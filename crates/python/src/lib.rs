//! Ukrainian stress Python extension module (PyO3 + maturin).
//!
//! Exposes:
//!   - `lookup(word)`             в†’ dict with all stress variants + phonetics + morphology
//!   - `lookup_many(words)`       в†’ list[dict] (batch lookup)
//!   - `mark(word)`               в†’ str with combining acutes (first variant)
//!   - `mark_many(words)`         в†’ list[str] (batch mark)
//!   - `word_count()`             в†’ int
//!   - `transcribe(word, stress_index)` в†’ dict (full IPA + syllables)
//!
//! # No disambiguation in Rust
//!
//! The engine returns **all** data it holds for a queried word.  Callers pick
//! the right reading using their own strategy (spaCy/Stanza morphological
//! context, user preference, first-variant fallback, etc.).
//!
//! See `documentation/API_DESIGN.md` for resolution examples.

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ua_stress_core::{
    UaStressDict,
    pipeline::transcribe as ua_transcribe,
    types::{MorphReading, StressReading, WordLookupResult},
};

// в”Ђв”Ђ Embedded dictionary в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

static DB_BIN: &[u8] = include_bytes!("../../../data/processed/ua_stress.bin.bz2");

static DICT: Lazy<UaStressDict> = Lazy::new(|| {
    UaStressDict::from_compressed_bytes(DB_BIN)
        .expect("Failed to load embedded Ukrainian stress dictionary")
});

// в”Ђв”Ђ Serialisation helpers в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Convert a single `MorphReading` into a Python dict.
fn morph_to_py<'py>(py: Python<'py>, m: &MorphReading) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("pos", PyList::new_bound(py, &m.pos))?;
    let feats_d = PyDict::new_bound(py);
    for (k, vs) in &m.feats {
        feats_d.set_item(k, PyList::new_bound(py, vs))?;
    }
    d.set_item("feats", feats_d)?;
    d.set_item("lemma", m.lemma.as_deref())?;
    d.set_item("definition", m.definition.as_deref())?;
    Ok(d)
}

/// Convert a single `StressReading` into a Python dict.
fn reading_to_py<'py>(py: Python<'py>, r: &StressReading) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("syllable_index", r.syllable_index)?;
    d.set_item("stress_from_end", r.stress_from_end)?;
    d.set_item("syllable_count", r.syllable_count)?;
    d.set_item("form", &r.form)?;
    d.set_item("stressed_form", &r.stressed_form)?;
    d.set_item("word_syllables", PyList::new_bound(py, &r.word_syllables))?;
    d.set_item("ipa", &r.ipa)?;
    d.set_item("ipa_syllables", PyList::new_bound(py, &r.ipa_syllables))?;

    // Token-level phonetic detail
    let tokens_list = PyList::empty_bound(py);
    for t in &r.tokens {
        let td = PyDict::new_bound(py);
        td.set_item("ipa", &t.ipa)?;
        td.set_item("source", &t.source)?;
        td.set_item("type", format!("{:?}", t.token_type))?;
        td.set_item("vowel_index", t.vowel_index)?;
        td.set_item("stressed", t.stressed)?;
        td.set_item("palatalized", t.palatalized)?;
        tokens_list.append(td)?;
    }
    d.set_item("tokens", tokens_list)?;

    // Morphological readings
    let morph_list = PyList::empty_bound(py);
    for m in &r.morph {
        morph_list.append(morph_to_py(py, m)?)?;
    }
    d.set_item("morph", morph_list)?;
    d.set_item("confidence", r.confidence.as_deref())?;
    Ok(d)
}

/// Convert a `WordLookupResult` into a Python dict.
fn lookup_result_to_py<'py>(
    py: Python<'py>,
    result: &WordLookupResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("form", &result.form)?;
    let readings_list = PyList::empty_bound(py);
    for r in &result.readings {
        readings_list.append(reading_to_py(py, r)?)?;
    }
    d.set_item("readings", readings_list)?;
    Ok(d)
}

// в”Ђв”Ђ Public functions в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

/// Look up a Ukrainian word and return all stress variants with full phonetics
/// and morphology.
///
/// Returns a dict:
///
/// .. code-block:: python
///
///     {
///       "form": "Р·Р°РјРѕРє",
///       "readings": [
///         {
///           "stress_index": 0,
///           "form": "Р·Р°РјРѕРє",
///           "stressed_form": "Р·Р°МЃРјРѕРє",
///           "word_syllables": ["Р·Р°", "РјРѕРє"],
///           "ipa": "zЙ‘mЙ”k",
///           "ipa_syllables": ["Л€zЙ‘", "mЙ”k"],
///           "syllable_count": 2,
///           "tokens": [...],
///           "morph": [{"pos": ["NOUN"], "feats": {"Case": ["Nom"]}, "lemma": "Р·Р°РјРѕРє"}]
///         },
///         ...
///       ]
///     }
///
/// ``readings`` is an empty list if the word is not in the dictionary.
///
/// The engine never picks one reading.  Use the morphological context from your
/// NLP pipeline (spaCy, Stanza, etc.) to select the appropriate reading.
/// See ``documentation/API_DESIGN.md`` for resolution examples.
#[pyfunction]
fn lookup(py: Python<'_>, word: &str) -> PyResult<Py<PyDict>> {
    let result = DICT.lookup(word);
    Ok(lookup_result_to_py(py, &result)?.into())
}

/// Batch lookup for many words.
///
/// Args:
///     words: list of word strings.
///
/// Returns:
///     list of lookup result dicts (same shape as ``lookup``), in input order.
#[pyfunction]
fn lookup_many(py: Python<'_>, words: Vec<String>) -> PyResult<Py<PyList>> {
    let results = DICT.lookup_many(&words);
    let out = PyList::empty_bound(py);
    for r in &results {
        out.append(lookup_result_to_py(py, r)?)?;
    }
    Ok(out.into())
}

/// Return *word* with a combining acute accent (U+0301) placed after the
/// stressed vowel of the **first** (most common) stress variant.
///
/// Returns the word unchanged if it is not in the dictionary.
///
/// For full control over which variant is used, call ``lookup()`` and read
/// ``readings[i].stressed_form`` directly.
#[pyfunction]
fn mark(word: &str) -> String {
    let result = DICT.lookup(word);
    match result.readings.first() {
        Some(r) => r.stressed_form.clone(),
        None => word.to_string(),
    }
}

/// Batch stress-marking for many words.
///
/// Returns one output per input word. Unknown words are returned unchanged.
#[pyfunction]
fn mark_many(words: Vec<String>) -> Vec<String> {
    DICT.mark_many(&words)
}

/// Total number of word forms in the embedded dictionary.
#[pyfunction]
fn word_count() -> usize {
    DICT.len()
}

/// Transcribe a Ukrainian word to IPA using all 6 phonetic passes.
///
/// This is a lower-level function for when you already know the stress index
/// (e.g. from ``lookup``).  Use ``lookup`` to obtain all variants at once.
///
/// Args:
///     word:         Ukrainian word (any case, any apostrophe variant).
///     stress_index: 0-based index of the stressed vowel.
///
/// Returns a dict:
///
/// .. code-block:: python
///
///     {
///       "word": "РјР°РјР°",
///       "ipa": "mЙ‘mЙ‘",
///       "stress_index": 0,
///       "tokens": [{"ipa":"m","source":"Рј","type":"Consonant","vowel_index":-1,...}, ...],
///       "syllables": [{"ipa":"mЙ‘","word":"РјР°","stressed":true,"is_open":true}, ...]
///     }
#[pyfunction]
fn transcribe(py: Python<'_>, word: &str, si: u8) -> PyResult<Py<PyDict>> {
    let result = ua_transcribe(word, si);
    let d = PyDict::new_bound(py);
    d.set_item("word", &result.word)?;
    d.set_item("ipa", &result.ipa)?;
    d.set_item("stress_index", result.stress_index)?;

    let tokens_list = PyList::empty_bound(py);
    for t in &result.tokens {
        let td = PyDict::new_bound(py);
        td.set_item("ipa", &t.ipa)?;
        td.set_item("source", &t.source)?;
        td.set_item("type", format!("{:?}", t.token_type))?;
        td.set_item("vowel_index", t.vowel_index)?;
        td.set_item("stressed", t.stressed)?;
        td.set_item("palatalized", t.palatalized)?;
        tokens_list.append(td)?;
    }
    d.set_item("tokens", tokens_list)?;

    let syl_list = PyList::empty_bound(py);
    for s in &result.syllables {
        let sd = PyDict::new_bound(py);
        sd.set_item("ipa", &s.ipa)?;
        sd.set_item("word", &s.word)?;
        sd.set_item("stressed", s.stressed)?;
        sd.set_item("is_open", s.is_open)?;
        syl_list.append(sd)?;
    }
    d.set_item("syllables", syl_list)?;

    Ok(d.into())
}

// в”Ђв”Ђ Module entry point в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

#[pymodule]
fn ukrainian_stress(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lookup, m)?)?;
    m.add_function(wrap_pyfunction!(lookup_many, m)?)?;
    m.add_function(wrap_pyfunction!(mark, m)?)?;
    m.add_function(wrap_pyfunction!(mark_many, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_function(wrap_pyfunction!(transcribe, m)?)?;
    Ok(())
}
