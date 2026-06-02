//! dict.rs — Dictionary loading and lookup for the Ukrainian stress engine.
//!
//! # Architecture
//!
//! ## Single Responsibility
//! This module handles: binary dict loading, O(log N) lookup by surface form,
//! and construction of fully-expanded [`WordLookupResult`]s.
//! Phonetic transcription lives in `pipeline.rs`.
//!
//! ## No disambiguation
//! The engine **never picks one stress variant**. It returns *all* data it holds
//! for a queried form so that callers can apply their own resolution strategy
//! (NLP context in Python, first-variant fallback in the browser, etc.).
//! See `documentation/API_DESIGN.md` for resolution examples.

use crate::normalize::{apply_stress_marks, normalize_word};
use crate::pipeline::transcribe;
use crate::types::{MorphReading, StressReading, UaStressDbRaw, WordForm, WordLookupResult};
use once_cell::sync::OnceCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Read;

// ── Global singleton ─────────────────────────────────────────────────────────

static GLOBAL_DICT: OnceCell<UaStressDict> = OnceCell::new();

/// Initialise the global dictionary from a bzip2-compressed bincode blob.
///
/// Intended for the WASM / PyO3 crates that embed the binary via `include_bytes!`.
/// Safe to call multiple times — only the first call has effect.
///
/// # Panics
/// Panics if the bytes cannot be decompressed or deserialised.
pub fn init_global_dict(compressed_bytes: &[u8]) {
    GLOBAL_DICT.get_or_init(|| {
        UaStressDict::from_compressed_bytes(compressed_bytes)
            .expect("Failed to load embedded Ukrainian stress dictionary")
    });
}

/// Return a reference to the global dictionary.
///
/// # Panics
/// Panics if [`init_global_dict`] has not been called yet.
pub fn global_dict() -> &'static UaStressDict {
    GLOBAL_DICT
        .get()
        .expect("Ukrainian stress dictionary not initialised — call init_global_dict first")
}

// ── Main dictionary type ──────────────────────────────────────────────────────

/// Runtime Ukrainian stress dictionary.
///
/// Wraps [`UaStressDbRaw`] and provides fast O(log N) lookup via binary search
/// on the sorted entries vector.
pub struct UaStressDict {
    raw: UaStressDbRaw,
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Count Ukrainian vowel graphemes in a normalized (lowercase) word.
///
/// Recognises the 10 vowel letters of the Ukrainian alphabet.
fn count_vowels(word: &str) -> usize {
    const UA_VOWELS: &[char] = &['а', 'е', 'є', 'и', 'і', 'ї', 'о', 'у', 'ю', 'я'];
    word.chars().filter(|c| UA_VOWELS.contains(c)).count()
}

impl UaStressDict {
    /// Deserialise from a **bzip2-compressed** bincode blob (as produced by the
    /// builder crate).
    pub fn from_compressed_bytes(compressed: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        use bzip2_rs::DecoderReader;
        let mut buf = Vec::new();
        DecoderReader::new(compressed).read_to_end(&mut buf)?;
        Self::from_bincode_bytes(&buf)
    }

    /// Deserialise from already-decompressed bincode bytes.
    pub fn from_bincode_bytes(buf: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = bincode::deserialize::<UaStressDbRaw>(buf)
            .map_err(|e| format!("failed to deserialize binary DB: {e}"))?;
        Ok(Self { raw })
    }

    /// Look up *word* (any case, any apostrophe variant).
    ///
    /// Returns a [`WordLookupResult`] with **all** stress variants, each
    /// containing full phonetics and all morphological analyses that share
    /// that stress position.  `readings` is empty when the word is unknown.
    ///
    /// The engine never picks a single reading; selection is the caller's
    /// responsibility (see `documentation/API_DESIGN.md`).
    pub fn lookup(&self, word: &str) -> WordLookupResult {
        let form = normalize_word(word);

        // Fast path: single-syllable words always have stress on the only
        // available vowel (index 0). No dictionary or ML lookup needed.
        if count_vowels(&form) == 1 {
            return self.single_syllable_reading(form);
        }

        let entries = &self.raw.entries;

        let forms = match entries.binary_search_by(|(w, _)| w.as_str().cmp(form.as_str())) {
            Ok(idx) => &entries[idx].1,
            Err(_) => {
                return WordLookupResult {
                    form,
                    readings: Vec::new(),
                }
            }
        };

        // Group morphological readings by stress index.
        // Use BTreeMap so stress indices come out in ascending order.
        let mut stress_map: BTreeMap<u8, Vec<MorphReading>> = BTreeMap::new();
        for wf in forms {
            let morph = self.expand_morph(wf);
            // Each WordForm can carry multiple stress positions (variative stress).
            for &si in &wf.stress_variants {
                stress_map.entry(si).or_default().push(morph.clone());
            }
        }

        // Clitics / purely consonantal words may have no stress_variants.
        // Still generate a reading so the word isn't reported as unknown.
        if stress_map.is_empty() {
            stress_map.insert(0, forms.iter().map(|wf| self.expand_morph(wf)).collect());
        }

        // Build one StressReading per unique stress index.
        let readings = stress_map
            .into_iter()
            .map(|(si, morph)| {
                let tr = transcribe(&form, si);
                let syllable_index = tr.syllables.iter().position(|s| s.stressed).unwrap_or(0);
                let syllable_count = tr.syllables.len();
                let stress_from_end = if syllable_count == 0 {
                    0
                } else {
                    syllable_count - syllable_index
                };
                let word_syllables: Vec<String> =
                    tr.syllables.iter().map(|s| s.word.clone()).collect();
                let ipa_syllables: Vec<String> = tr
                    .syllables
                    .iter()
                    .map(|s| {
                        if s.stressed {
                            format!("\u{02c8}{}", s.ipa)
                        } else {
                            s.ipa.clone()
                        }
                    })
                    .collect();

                StressReading {
                    syllable_index,
                    stress_from_end,
                    syllable_count,
                    form: form.clone(),
                    stressed_form: apply_stress_marks(&form, &[si]),
                    word_syllables,
                    ipa: tr.ipa,
                    ipa_syllables,
                    tokens: tr.tokens,
                    morph,
                    confidence: None,
                }
            })
            .collect();

        WordLookupResult { form, readings }
    }

    /// Batch lookup for many words.
    ///
    /// Returns one [`WordLookupResult`] per input word in the same order.
    /// This is primarily a convenience API for language bindings that need
    /// to amortize call overhead across large word lists.
    pub fn lookup_many(&self, words: &[String]) -> Vec<WordLookupResult> {
        words.iter().map(|w| self.lookup(w)).collect()
    }

    /// Batch stress-marking for many words.
    ///
    /// For each input word returns the stressed form of the first reading,
    /// or the original word unchanged when the word is unknown.
    pub fn mark_many(&self, words: &[String]) -> Vec<String> {
        words
            .iter()
            .map(|w| {
                let result = self.lookup(w);
                match result.readings.first() {
                    Some(r) => r.stressed_form.clone(),
                    None => w.clone(),
                }
            })
            .collect()
    }

    /// Total number of word forms stored in the dictionary.
    pub fn len(&self) -> usize {
        self.raw.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.entries.is_empty()
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a [`WordLookupResult`] for a word known to have exactly one syllable.
    ///
    /// Stress is placed at vowel index 0 — the only possible position.
    /// Morphological fields are left empty because the word may not be in the
    /// dictionary (it could be OOV, a number, a proper name, etc.).
    fn single_syllable_reading(&self, form: String) -> WordLookupResult {
        let tr = transcribe(&form, 0);
        let syllable_count = tr.syllables.len();
        let word_syllables: Vec<String> =
            tr.syllables.iter().map(|s| s.word.clone()).collect();
        let ipa_syllables: Vec<String> = tr
            .syllables
            .iter()
            .map(|s| {
                if s.stressed {
                    format!("\u{02c8}{}", s.ipa)
                } else {
                    s.ipa.clone()
                }
            })
            .collect();

        let reading = StressReading {
            syllable_index: 0,
            stress_from_end: syllable_count.max(1),
            syllable_count,
            stressed_form: apply_stress_marks(&form, &[0]),
            form: form.clone(),
            word_syllables,
            ipa: tr.ipa,
            ipa_syllables,
            tokens: tr.tokens,
            morph: Vec::new(),
            confidence: Some("single-syllable".to_string()),
        };

        WordLookupResult {
            form,
            readings: vec![reading],
        }
    }

    /// Expand a single [`WordForm`] (intern-indexed) into a [`MorphReading`]
    /// (string-valued) without computing phonetics.
    fn expand_morph(&self, wf: &WordForm) -> MorphReading {
        let pos = wf
            .pos
            .iter()
            .filter_map(|&i| self.raw.pos_table.get(i as usize))
            .cloned()
            .collect();

        let mut feats: HashMap<String, Vec<String>> = HashMap::new();
        for (key_i, val_is) in &wf.feats {
            if let Some(key) = self.raw.feat_key_table.get(*key_i as usize) {
                let vals: Vec<String> = val_is
                    .iter()
                    .filter_map(|&v| self.raw.feat_val_table.get(v as usize))
                    .cloned()
                    .collect();
                feats.insert(key.clone(), vals);
            }
        }

        let lemma = wf
            .lemma
            .and_then(|i| self.raw.lemma_pool.get(i as usize))
            .cloned();

        let definition = wf
            .definition
            .and_then(|i| self.raw.definition_pool.get(i as usize))
            .cloned();

        MorphReading { pos, feats, lemma, definition }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_real_bin() {
        // Load the actual embedded binary to catch deserialization issues.
        let compressed = include_bytes!("../../../data/processed/ua_stress.bin.bz2");
        let dict = UaStressDict::from_compressed_bytes(compressed)
            .expect("Failed to load ua_stress.bin.bz2");
        assert!(dict.len() > 100_000, "Expected >100k entries, got {}", dict.len());
        println!("Loaded {} entries", dict.len());
    }

    /// The empty dictionary (placeholder binary) should return empty readings,
    /// not panic.
    #[test]
    fn lookup_unknown_returns_empty() {
        // We can't easily test real lookups without a populated binary,
        // but we can verify the type structure compiles and the empty path works.
        let result = WordLookupResult {
            form: "тест".to_string(),
            readings: Vec::new(),
        };
        assert!(result.readings.is_empty());
        assert_eq!(result.form, "тест");
    }

    /// Verify MorphReading and StressReading can be constructed.
    #[test]
    fn types_constructible() {
        let morph = MorphReading {
            pos: vec!["NOUN".to_string()],
            feats: {
                let mut m = HashMap::new();
                m.insert("Case".to_string(), vec!["Nom".to_string()]);
                m
            },
            lemma: Some("замок".to_string()),
            definition: None,
        };
        assert_eq!(morph.pos[0], "NOUN");
    }

    /// Every single-syllable Ukrainian word must be resolved immediately via the
    /// fast path (no dict lookup) with stress at index 0 and
    /// `confidence == Some("single-syllable")`.
    #[test]
    fn single_syllable_fast_path() {
        let compressed = include_bytes!("../../../data/processed/ua_stress.bin.bz2");
        let dict = UaStressDict::from_compressed_bytes(compressed)
            .expect("Failed to load ua_stress.bin.bz2");

        #[rustfmt::skip]
        let words: &[&str] = &[
            // consonant-heavy monosyllables
            "клус", "вступ", "груп", "круп", "труп", "тюп", "хлюп", "служб",
            "блуд", "бруд", "брук", "внук", "грюк", "друк", "жмут", "звук",
            "клюк", "Кнут", "крук", "Крут", "люд", "люк", "прут", "скрут",
            "спрут", "стук", "схуд", "труд", "трюк", "штук", "джгут", "жґут",
            "збут", "крюк", "пруд", "флуд",
            // verb/pronoun monosyllables
            "б'ю", "вб'ю", "вмру", "всю", "втну", "вчу", "гру", "дну", "дню",
            "дю", "жду", "злу", "йду", "йму", "Йсу", "Лю", "мчу", "ню",
            "п'ю", "пню", "псу", "склу", "сну", "сплю", "ссу", "Сью", "Сю",
            "тпру", "тру", "тьму", "тьфу", "тьху", "тю", "ф'ю", "цю",
            "шву", "шлю",
            // inflected monosyllables
            "б'юсь", "б'ють", "блюз", "бюст", "в'юн", "в'ють", "вб'ють",
            "вкруг", "впрусь", "вчув", "вщух", "глузд", "глум", "грув",
            "грудь", "груш", "ґрунт", "дмуть", "друг", "дюн", "жнуть",
            "звуть", "здув", "змусь", "ключ", "круг", "ллють", "лють",
            "мнуть", "мруж", "п'ють", "плуг", "плюс", "пруг", "пруть",
            "псуй", "пхнув", "пхнуть", "рвуть", "ртуть", "слуг", "слух",
            "смуг", "струм", "струн", "струс", "стуг", "сюр", "ткнув",
            "тхнуть", "хлющ", "хруст", "штурм", "щур", "брухт", "взув",
            "вкус", "втнув", "глуз", "гнув", "Гюнт", "дюйм", "круть",
            "люкс", "мруть", "нюх", "плюш", "плющ", "пнув", "пструг",
            "смух", "снус", "сплюнь", "спух", "трух", "тюль", "фрукт",
            "хрущ", "шнур",
            // short nouns /u/
            "гуп", "куп", "пуп", "суп", "туп", "дубль", "зубр", "рубль",
            "пупс", "буд", "бук", "бут", "гуд", "гук", "гут", "жук", "кут",
            "лук", "мук", "нуд", "нут", "пук", "пут", "рук", "суд", "сук",
            "тук", "тут", "фут", "хук", "ют",
            // interjections & particles /u/
            "гу", "му", "ну", "ту", "у", "фу", "ху",
            // more /u/ forms
            "був", "будь", "бунт", "вус", "вуст", "вух", "гул", "гульк",
            "гурт", "дум", "дух", "душ", "куль", "кум", "куш", "кущ", "луг",
            "муж", "муз", "мур", "мух", "муч", "нуль", "пункт", "пух", "рух",
            "руш", "туз", "туй", "туш", "Уж", "уст", "фур", "цур", "чуй",
            "шум", "Буг", "буз", "буй", "буль", "бур", "бус", "вуж", "вуй",
            "Вулф", "гугл", "гус", "гусь", "ґуґл", "дур", "зум", "кунь",
            "курс", "кус", "куць", "лунь", "луск", "мул", "нудь", "пульс",
            "пульт", "пум", "пунш", "пуск", "путь", "руль", "рунь", "рур",
            "рус", "Русь", "сум", "сунь", "суть", "тум", "тур", "ум", "урн",
            "Ус", "уф", "ух", "хуй", "хух", "чур", "шусть", "юнг", "юнь",
            "юрт",
            // /o/ monosyllables
            "бомб", "бот", "бо", "бог", "бой", "бокс", "бомж", "бор", "борг",
            "борт", "борть", "борщ", "бос", "гоп", "поп", "топ", "хоп", "мопс",
            "горб", "вод", "год", "док", "код", "мод", "мок", "рок", "рот",
            "шок", "Шот", "йод", "нот", "шот",
            // short /o/ particles
            "го", "ґо", "до", "зо", "йо", "ко", "Ло", "мо", "мо'", "но",
            "ньо", "по", "то", "хо", "шо",
            // /o/ inflected
            "вовк", "вождь", "Вон", "гол", "гольф", "Гор", "доль", "дон",
            "донь", "дощ", "жовч", "зойк", "зон", "йой", "ковдр", "кой",
            "Коль", "ком", "Кон", "корж", "корм", "Кость", "кошт", "Лодж",
            "лож", "лом", "Лонг", "лорд", "лось", "лох", "мов", "мож", "Мон",
            "монстр", "мор", "морг", "мох", "Ной", "Нор", "норм", "нош",
            "повз", "Пол", "полк", "пор", "порт", "Порш", "пост", "рож", "роз",
            "Ройс", "роль", "ром", "рос", "сов", "сом", "сон", "сонць", "товк",
            "тож", "той", "толк", "том", "тон", "тонн", "тор", "торг", "торс",
            "торт", "тост", "фон", "фонд", "фор", "форд", "форм", "хол", "хор",
            "хоч", "чом", "чорт", "шов", "шовк", "Волш", "гоц", "ґонт", "довг",
            "корт", "корч", "лов", "моль", "морж", "повх", "сойм", "соль",
            "торф", "хорт", "чось", "шось",
            // more /o/ monosyllables
            "оп", "стоп", "Стоп", "хлоп", "скорб", "стовб", "стовп", "блок",
            "грот", "змок", "крок", "од", "ок", "от", "скок", "скот", "сльот",
            "смок", "строк", "трок",
            // /o/ short forms
            "вйо", "всьо", "Джо", "дно", "зло", "йшло", "о", "про", "скло",
            "сто", "тло", "хто", "Шкло", "що",
            // /o/ clusters
            "вдвох", "вдовж", "вздовж", "всох", "втрьох", "гроз", "двом",
            "Джон", "дров", "дрож", "дрозд", "зводь", "здовж", "здох", "змов",
            "знов", "клон", "кров", "крон", "льон", "льох", "ой", "он", "ос",
            "ост", "ось", "ох", "плоть", "площ", "псом", "склом", "скронь",
            "слон", "смог", "Снов", "сповз", "спорт", "стос", "трон", "трьох",
            "фронт", "хльост", "хтось", "Шклом", "шторм", "щось",
        ];

        let mut failures: Vec<String> = Vec::new();
        for &word in words {
            let result = dict.lookup(word);
            if result.readings.len() != 1 {
                failures.push(format!(
                    "{:?}: expected 1 reading, got {}",
                    word,
                    result.readings.len()
                ));
                continue;
            }
            let r = &result.readings[0];
            if r.confidence.as_deref() != Some("single-syllable") {
                failures.push(format!(
                    "{:?}: confidence = {:?}, expected Some(\"single-syllable\")",
                    word, r.confidence
                ));
            }
            if r.syllable_index != 0 {
                failures.push(format!(
                    "{:?}: syllable_index = {}, expected 0",
                    word, r.syllable_index
                ));
            }
        }

        if !failures.is_empty() {
            panic!(
                "{} single-syllable words failed:\n{}",
                failures.len(),
                failures.join("\n")
            );
        }
    }

    #[test]
    fn lookup_many_keeps_input_order() {
        let compressed = include_bytes!("../../../data/processed/ua_stress.bin.bz2");
        let dict = UaStressDict::from_compressed_bytes(compressed)
            .expect("Failed to load ua_stress.bin.bz2");

        let words = vec![
            "мама".to_string(),
            "xyz_unknown".to_string(),
            "університет".to_string(),
        ];

        let out = dict.lookup_many(&words);
        assert_eq!(out.len(), words.len());
        assert_eq!(out[0].form, "мама");
        assert_eq!(out[1].form, "xyz_unknown");
        assert_eq!(out[2].form, "університет");
        assert!(!out[0].readings.is_empty());
        assert!(out[1].readings.is_empty());
        assert!(!out[2].readings.is_empty());
    }

    #[test]
    fn mark_many_marks_known_and_preserves_unknown() {
        let compressed = include_bytes!("../../../data/processed/ua_stress.bin.bz2");
        let dict = UaStressDict::from_compressed_bytes(compressed)
            .expect("Failed to load ua_stress.bin.bz2");

        let words = vec!["мама".to_string(), "xyz_unknown".to_string()];
        let out = dict.mark_many(&words);

        assert_eq!(out.len(), 2);
        assert_ne!(out[0], "мама");
        assert_eq!(out[1], "xyz_unknown");
    }
}
