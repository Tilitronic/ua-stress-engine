//! Ukrainian stress WASM bindings (wasm-bindgen).
//!
//! Exposes:
//!   - `lookup(word)`                   → JS object (WordLookupResult)
//!   - `mark(word)`                     → string
//!   - `wordCount()`                    → number
//!   - `transcribe(word, stressIndex)`  → JS object (TranscriptionResult)
//!
//! # No disambiguation in WASM
//!
//! The engine returns all data; the calling application picks the reading.
//! Typical web strategy: render `readings[0].stressedForm` and show a tooltip
//! with all variants. See `documentation/API_DESIGN.md` for details.

use js_sys::{Array, Object, Reflect};
use ua_stress_core::{UaStressDict, pipeline::transcribe as ua_transcribe};
use wasm_bindgen::prelude::*;

static DB_BIN: &[u8] = include_bytes!("../../../data/processed/ua_stress.bin.bz2");

thread_local! {
    static DICT: UaStressDict = UaStressDict::from_compressed_bytes(DB_BIN)
        .expect("Failed to load embedded Ukrainian stress dictionary");
}

/// Initialise the dictionary eagerly on WASM module load.
/// Called automatically by the bundler-target JS glue (`wasm.__wbindgen_start()`).
#[wasm_bindgen(start)]
pub fn start() {
    DICT.with(|_| {}); // force initialisation
}

// ── Serialisation helpers ────────────────────────────────────────────────────

fn set(obj: &Object, key: &str, value: JsValue) {
    Reflect::set(obj, &JsValue::from_str(key), &value).unwrap();
}

fn str_array(strings: &[String]) -> JsValue {
    let arr = Array::new();
    for s in strings {
        arr.push(&JsValue::from_str(s));
    }
    arr.into()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Look up a Ukrainian word and return all stress variants with full phonetics
/// and morphology.
///
/// Returns a JS object:
/// ```json
/// {
///   "form": "замок",
///   "readings": [
///     {
///       "syllableIndex": 0,
///       "form": "замок",
///       "stressedForm": "за́мок",
///       "wordSyllables": ["за", "мок"],
///       "ipa": "zɑmɔk",
///       "ipaSyllables": ["ˈzɑ", "mɔk"],
///       "syllableCount": 2,
///       "tokens": [...],
///       "morph": [{"pos":["NOUN"],"feats":{"Case":["Nom"]},"lemma":"замок"}]
///     },
///     ...
///   ]
/// }
/// ```
///
/// `readings` is an empty array when the word is not in the dictionary.
///
/// ```js
/// import { lookup } from 'ua-word-stress-wasm';
/// const result = lookup('замок');
/// result.readings[0].stressedForm;  // → 'за́мок'
/// ```
#[wasm_bindgen]
pub fn lookup(word: &str) -> Object {
    DICT.with(|d| {
    let result = d.lookup(word);
    let obj = Object::new();
    set(&obj, "form", JsValue::from_str(&result.form));

    let readings_arr = Array::new();
    for r in &result.readings {
        let rd = Object::new();
        set(&rd, "syllableIndex",  JsValue::from(r.syllable_index as u32));
        set(&rd, "stressFromEnd",  JsValue::from(r.stress_from_end as u32));
        set(&rd, "syllableCount",  JsValue::from(r.syllable_count as u32));
        set(&rd, "form",           JsValue::from_str(&r.form));
        set(&rd, "stressedForm",   JsValue::from_str(&r.stressed_form));
        set(&rd, "wordSyllables",  str_array(&r.word_syllables));
        set(&rd, "ipa",            JsValue::from_str(&r.ipa));
        set(&rd, "ipaSyllables",   str_array(&r.ipa_syllables));

        // Token-level phonetic detail
        let tokens_arr = Array::new();
        for t in &r.tokens {
            let td = Object::new();
            set(&td, "ipa",         JsValue::from_str(&t.ipa));
            set(&td, "source",      JsValue::from_str(&t.source));
            set(&td, "type",        JsValue::from_str(&format!("{:?}", t.token_type)));
            set(&td, "vowelIndex",  JsValue::from(t.vowel_index as i32));
            set(&td, "stressed",    JsValue::from(t.stressed));
            set(&td, "palatalized", JsValue::from(t.palatalized));
            tokens_arr.push(&td);
        }
        set(&rd, "tokens", tokens_arr.into());

        // Morphological readings
        let morph_arr = Array::new();
        for m in &r.morph {
            let md = Object::new();
            set(&md, "pos", str_array(&m.pos));
            let feats_obj = Object::new();
            for (k, vs) in &m.feats {
                set(&feats_obj, k, str_array(vs));
            }
            set(&md, "feats", feats_obj.into());
            let lemma = m.lemma.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
            set(&md, "lemma", lemma);
            let def = m.definition.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
            set(&md, "definition", def);
            morph_arr.push(&md);
        }
        set(&rd, "morph", morph_arr.into());
        let conf = r.confidence.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
        set(&rd, "confidence", conf);

        readings_arr.push(&rd);
    }
    set(&obj, "readings", readings_arr.into());

    obj
    })
}

/// Return *word* with a combining acute accent on the stressed vowel of the
/// **first** reading.  Returns the word unchanged if unknown.
///
/// For full control over which variant is shown, call `lookup()` and use
/// `readings[i].stressedForm` directly.
///
/// ```js
/// mark('мама')  // → 'ма́ма'
/// ```
#[wasm_bindgen]
pub fn mark(word: &str) -> String {
    DICT.with(|d| {
        let result = d.lookup(word);
        match result.readings.first() {
            Some(r) => r.stressed_form.clone(),
            None => word.to_string(),
        }
    })
}

/// Total number of word forms in the embedded dictionary.
#[wasm_bindgen(js_name = wordCount)]
pub fn word_count() -> u32 {
    DICT.with(|d| d.len() as u32)
}

/// Return the 0-based **syllable** index of the stressed syllable of the first
/// reading, or -1 if the word is unknown.
///
/// This is the same value as `readings[0].syllableIndex` from `lookup()`.
/// Use it as a direct index into `readings[0].wordSyllables`.
///
/// ```js
/// stressIndex('мама')  // → 0  (first syllable stressed)
/// stressIndex('ааааа') // → -1 (unknown)
/// ```
#[wasm_bindgen(js_name = stressIndex)]
pub fn stress_index(word: &str) -> i32 {
    DICT.with(|d| {
        let result = d.lookup(word);
        match result.readings.first() {
            Some(r) => r.syllable_index as i32,
            None => -1,
        }
    })
}

/// Batch stress-index lookup: takes a JS Array of word strings and returns a
/// JS Int32Array of stress indices (one per word, -1 = unknown).
///
/// Amortises the per-call JS↔WASM string encoding overhead across N words.
/// Significantly faster than calling `stressIndex()` N times individually.
///
/// ```js
/// const indices = stressIndexBatch(['мама', 'замок', 'xyz']);
/// // → Int32Array [1, 0, -1]
/// ```
#[wasm_bindgen(js_name = stressIndexBatch)]
pub fn stress_index_batch(words: &js_sys::Array) -> js_sys::Int32Array {
    DICT.with(|d| {
        let out = js_sys::Int32Array::new_with_length(words.length());
        for i in 0..words.length() {
            let word = words.get(i).as_string().unwrap_or_default();
            let result = d.lookup(&word);
            let idx = match result.readings.first() {
                Some(r) => r.syllable_index as i32,
                None => -1,
            };
            out.set_index(i, idx);
        }
        out
    })
}

/// Batch mark — stress-mark every word in a JS Array of strings.
///
/// Returns a JS Array of strings where each word has a combining acute accent
/// on its stressed vowel.  Words not in the dictionary are returned unchanged.
///
/// Much faster than calling `mark()` N times because JS↔WASM string encoding
/// overhead is amortised.
///
/// ```js
/// markBatch(['мама', 'тато', 'університет'])
/// // → ['ма́ма', 'та́то', 'університе́т']
/// ```
#[wasm_bindgen(js_name = markBatch)]
pub fn mark_batch(words: &js_sys::Array) -> js_sys::Array {
    DICT.with(|d| {
        let input: Vec<String> = (0..words.length())
            .map(|i| words.get(i).as_string().unwrap_or_default())
            .collect();
        let marked = d.mark_many(&input);

        let out = js_sys::Array::new_with_length(words.length());
        for (i, word) in marked.iter().enumerate() {
            out.set(i as u32, JsValue::from_str(word));
        }
        out
    })
}

/// Alias for `markBatch` with camelCase `Many` naming.
#[wasm_bindgen(js_name = markMany)]
pub fn mark_many(words: &js_sys::Array) -> js_sys::Array {
    mark_batch(words)
}

/// Batch full lookup — returns a JS Array of `LookupResult` objects (same
/// shape as `lookup()`), one per word.  Words not in the dictionary have an
/// empty `readings` array.
///
/// Significantly faster than calling `lookup()` N times.
///
/// ```js
/// const results = lookupBatch(['мама', 'замок']);
/// results[1].readings[0].stressedForm; // → 'за́мок'
/// ```
#[wasm_bindgen(js_name = lookupBatch)]
pub fn lookup_batch(words: &js_sys::Array) -> js_sys::Array {
    DICT.with(|d| {
        let input: Vec<String> = (0..words.length())
            .map(|i| words.get(i).as_string().unwrap_or_default())
            .collect();
        let results = d.lookup_many(&input);

        let out = js_sys::Array::new_with_length(words.length());
        for (i, r) in results.iter().enumerate() {
            let obj = Object::new();
            set(&obj, "form", JsValue::from_str(&r.form));

            let readings_arr = Array::new();
            for rr in &r.readings {
                let rd = Object::new();
                set(&rd, "syllableIndex", JsValue::from(rr.syllable_index as u32));
                set(&rd, "stressFromEnd", JsValue::from(rr.stress_from_end as u32));
                set(&rd, "syllableCount", JsValue::from(rr.syllable_count as u32));
                set(&rd, "form", JsValue::from_str(&rr.form));
                set(&rd, "stressedForm", JsValue::from_str(&rr.stressed_form));
                set(&rd, "wordSyllables", str_array(&rr.word_syllables));
                set(&rd, "ipa", JsValue::from_str(&rr.ipa));
                set(&rd, "ipaSyllables", str_array(&rr.ipa_syllables));

                let tokens_arr = Array::new();
                for t in &rr.tokens {
                    let td = Object::new();
                    set(&td, "ipa", JsValue::from_str(&t.ipa));
                    set(&td, "source", JsValue::from_str(&t.source));
                    set(&td, "type", JsValue::from_str(&format!("{:?}", t.token_type)));
                    set(&td, "vowelIndex", JsValue::from(t.vowel_index as i32));
                    set(&td, "stressed", JsValue::from(t.stressed));
                    set(&td, "palatalized", JsValue::from(t.palatalized));
                    tokens_arr.push(&td);
                }
                set(&rd, "tokens", tokens_arr.into());

                let morph_arr = Array::new();
                for m in &rr.morph {
                    let md = Object::new();
                    set(&md, "pos", str_array(&m.pos));
                    let feats_obj = Object::new();
                    for (k, vs) in &m.feats {
                        set(&feats_obj, k, str_array(vs));
                    }
                    set(&md, "feats", feats_obj.into());
                    let lemma = m.lemma.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
                    set(&md, "lemma", lemma);
                    let def = m.definition.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
                    set(&md, "definition", def);
                    morph_arr.push(&md);
                }
                set(&rd, "morph", morph_arr.into());
                let conf = rr.confidence.as_deref().map(JsValue::from_str).unwrap_or(JsValue::NULL);
                set(&rd, "confidence", conf);

                readings_arr.push(&rd);
            }
            set(&obj, "readings", readings_arr.into());
            out.set(i as u32, obj.into());
        }

        out
    })
}

/// Alias for `lookupBatch` with camelCase `Many` naming.
#[wasm_bindgen(js_name = lookupMany)]
pub fn lookup_many(words: &js_sys::Array) -> js_sys::Array {
    lookup_batch(words)
}

/// Transcribe a Ukrainian word to IPA using all 6 phonetic passes.
#[wasm_bindgen]
pub fn transcribe(word: &str, stress_index: u8) -> Object {
    let result = ua_transcribe(word, stress_index);
    let obj = Object::new();

    set(&obj, "word",        JsValue::from_str(&result.word));
    set(&obj, "ipa",         JsValue::from_str(&result.ipa));
    set(&obj, "stressIndex", JsValue::from(result.stress_index as u32));

    let tokens_arr = Array::new();
    for t in &result.tokens {
        let td = Object::new();
        set(&td, "ipa",         JsValue::from_str(&t.ipa));
        set(&td, "source",      JsValue::from_str(&t.source));
        set(&td, "type",        JsValue::from_str(&format!("{:?}", t.token_type)));
        set(&td, "vowelIndex",  JsValue::from(t.vowel_index as i32));
        set(&td, "stressed",    JsValue::from(t.stressed));
        set(&td, "palatalized", JsValue::from(t.palatalized));
        tokens_arr.push(&td);
    }
    set(&obj, "tokens", tokens_arr.into());

    let syl_arr = Array::new();
    for s in &result.syllables {
        let sd = Object::new();
        set(&sd, "ipa",      JsValue::from_str(&s.ipa));
        set(&sd, "word",     JsValue::from_str(&s.word));
        set(&sd, "stressed", JsValue::from(s.stressed));
        set(&sd, "isOpen",   JsValue::from(s.is_open));
        syl_arr.push(&sd);
    }
    set(&obj, "syllables", syl_arr.into());

    obj
}

