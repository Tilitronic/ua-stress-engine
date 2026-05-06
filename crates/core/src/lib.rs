//! ua_stress_core — Ukrainian stress resolution + IPA phonetic pipeline.
//!
//! # Modules
//! - `types`     — all public types (phonetic + dict result)
//! - `data`      — static phoneme tables and grapheme maps
//! - `normalize` — word normalization (lowercase + canonical apostrophe)
//! - `tokenizer` — Pass 1: Cyrillic → PhoneticToken[]
//! - `geminate`  — Pass 1.5: merge double consonants
//! - `palatalization` — Pass 2: regressive dental softening
//! - `voicing`   — Pass 3: obstruent voicing assimilation
//! - `place`     — Pass 3b: sibilant + affricate place assimilation
//! - `vowel_allophones` — Pass 4: stressed vowel marking + vowel variants
//! - `v_allophones`     — Pass 5: /ʋ/ positional allophones
//! - `syllabifier`      — Pass 6: syllabification (produces word + IPA per syllable)
//! - `pipeline`  — orchestrates all 6 passes → `TranscriptionResult`
//! - `dict`      — binary dict loading and lookup → `WordLookupResult`

pub mod academic_tests;
pub mod data;
pub mod dict;
pub mod geminate;
pub mod normalize;
pub mod palatalization;
pub mod pipeline;
pub mod place;
pub mod syllabifier;
pub mod tokenizer;
pub mod types;
pub mod v_allophones;
pub mod voicing;
pub mod vowel_allophones;

// ── Convenience re-exports ────────────────────────────────────────────────────

pub use dict::{UaStressDict, global_dict, init_global_dict};
pub use normalize::{apply_stress_marks, normalize_word};
pub use pipeline::transcribe;
pub use types::{
    MorphReading, PhoneticToken, StressReading, TokenType, TranscriptionResult, UaStressDbRaw,
    UaSyllable, WordForm, WordLookupResult,
};
