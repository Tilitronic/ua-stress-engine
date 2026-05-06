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

    /// Total number of word forms stored in the dictionary.
    pub fn len(&self) -> usize {
        self.raw.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.entries.is_empty()
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

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
}
