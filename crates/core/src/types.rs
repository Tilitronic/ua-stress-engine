//! types.rs — All public types for the Ukrainian phonetic engine and stress dictionary.
//!
//! Split into two sections:
//!   1. Phonetic pipeline types  — PhoneticToken, UaSyllable, TranscriptionResult, …
//!   2. Dictionary / storage types — WordForm (binary), WordFormResult (resolved)
//!
//! The phonetic types intentionally mirror the TypeScript API from uaPhoneticPipeline.ts
//! so that the two codebases stay aligned.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1 — Phonetic pipeline types
// ═══════════════════════════════════════════════════════════════════════════

/// Voicing class of a consonant.  Sonorants (m, n, l, r, j, ʋ) are exempt
/// from obstruent voicing assimilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoicePower {
    Voiceless,
    Voiced,
    /// Nasals, liquids, glides — no voicing assimilation applies.
    Sonorant,
}

/// Articulatory place of a consonant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    Labial,
    Dental,
    Postalveolar,
    Palatal,
    Velar,
    Glottal,
}

/// Articulatory manner of a consonant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Manner {
    Plosive,
    Fricative,
    Affricate,
    Trill,
    Lateral,
    Nasal,
    Approximant,
}

/// Palatal (soft) vs. non-palatal (hard) consonant variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Softness {
    Hard,
    Soft,
}

/// Full 5-dimensional feature vector for a consonant phoneme.
/// Directly parallels the TypeScript `ConsonantFeatures` interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsonantFeatures {
    pub voice_power: VoicePower,
    pub place: Place,
    pub manner: Manner,
    pub softness: Softness,
    /// True only for nasal consonants (м, н, нь).  Redundant with
    /// `manner == Nasal` but kept for TypeScript API parity.
    pub nasal: bool,
}

/// Height position of a vowel in the oral cavity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VowelHeight {
    High,
    HighMid,
    Mid,
    Low,
}

/// Front-back position of a vowel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VowelBackness {
    Front,
    Central,
    Back,
}

/// Lip rounding of a vowel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VowelRounding {
    Rounded,
    Unrounded,
}

/// Full 3-dimensional feature vector for a vowel phoneme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VowelFeatures {
    pub height: VowelHeight,
    pub backness: VowelBackness,
    pub rounding: VowelRounding,
    /// Stable vowels (/і/, /у/, /а/) do not shift in unstressed position.
    pub stable: bool,
}

/// The kind of sound a `PhoneticToken` represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// A consonant phoneme.
    Consonant,
    /// A vowel nucleus.
    Vowel,
    /// A glide (j, w, u̯) — neither fully consonant nor vowel.
    Glide,
    /// A glottal stop/boundary — produced by the apostrophe grapheme.
    Glottal,
}

/// A single phoneme token produced by the tokenizer and transformed by
/// subsequent passes.
///
/// Passes operate on `&mut Vec<PhoneticToken>` (or return a new `Vec`
/// when tokens must be removed).
#[derive(Debug, Clone)]
pub struct PhoneticToken {
    /// Current IPA symbol for this phoneme.  May be mutated by passes
    /// (e.g., voicing assimilation changes "s" → "z").
    pub ipa: String,
    /// The Cyrillic source grapheme(s) that produced this token.
    pub source: String,
    /// Token classification.
    pub token_type: TokenType,
    /// 0-based index of this vowel among all vowels in the word.
    /// `-1` for non-vowel tokens.
    pub vowel_index: i8,
    /// Whether this is the stressed vowel.  Set by `vowel_allophones` pass.
    pub stressed: bool,
    /// Whether this consonant has been palatalized (softened).
    /// When true, `ipa` ends with ʲ (e.g. "nʲ").
    pub palatalized: bool,
    /// Feature vector for consonants.  `None` for vowels.
    pub consonant_features: Option<ConsonantFeatures>,
    /// Feature vector for vowels.  `None` for consonants.
    pub vowel_features: Option<VowelFeatures>,
}

/// One syllable in the syllabified output.
#[derive(Debug, Clone)]
pub struct UaSyllable {
    /// Concatenated IPA of all tokens in this syllable.
    pub ipa: String,
    /// Original Cyrillic graphemes that map to this syllable.
    /// Joining all `word` fields reconstructs the original word form:
    /// `syllables.iter().map(|s| s.word.as_str()).collect::<String>()`
    pub word: String,
    /// Ordered tokens that make up this syllable.
    pub tokens: Vec<PhoneticToken>,
    /// Whether this syllable bears the word stress.
    pub stressed: bool,
    /// Whether the syllable ends on a vowel (open syllable, e.g. "ма").
    pub is_open: bool,
}

/// The complete output of `transcribe()` — the result of running a word
/// through all 6 phonetic passes.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// The input word (unchanged).
    pub word: String,
    /// 0-based stressed vowel index that was actually used (clamped if
    /// the caller supplied an out-of-range value).
    pub stress_index: u8,
    /// Flat post-pipeline token array (all passes applied).
    pub tokens: Vec<PhoneticToken>,
    /// Syllabified output.
    pub syllables: Vec<UaSyllable>,
    /// Flat IPA string (tokens concatenated).
    pub ipa: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2 — Binary dictionary types (serialized by the builder crate)
// ═══════════════════════════════════════════════════════════════════════════

/// One morphological variant of a word form as stored in the compact binary
/// dictionary.  All string fields are encoded as integer indices into the
/// shared string tables inside [`UaStressDbRaw`] to minimise binary size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordForm {
    /// 0-based vowel indices at which stress may fall.
    /// Multiple indices → the form is heteronymous or variative.
    pub stress_variants: Vec<u8>,
    /// Indices into `UaStressDbRaw::pos_table`.
    pub pos: Vec<u8>,
    /// `(feat_key_idx, [feat_val_idx, …])` pairs.
    /// Feature keys/values are Universal Dependencies names
    /// (e.g. "Case", "Nom").
    pub feats: Vec<(u8, Vec<u8>)>,
    /// Index into `UaStressDbRaw::lemma_pool`.  `None` = no lemma stored.
    pub lemma: Option<u32>,
    /// Index into `UaStressDbRaw::definition_pool`.  `None` = no definition.
    pub definition: Option<u32>,
}

/// The raw in-memory database deserialized from `ua_stress.bin.bz2`.
///
/// `entries` is sorted lexicographically by word so that [`UaStressDict`]
/// can use binary search for O(log N) lookups.
#[derive(Debug, Serialize, Deserialize)]
pub struct UaStressDbRaw {
    /// Universal POS tag strings (e.g. "NOUN", "VERB").
    pub pos_table: Vec<String>,
    /// UD morphological feature key strings (e.g. "Case", "Gender").
    pub feat_key_table: Vec<String>,
    /// UD morphological feature value strings (e.g. "Nom", "Masc").
    pub feat_val_table: Vec<String>,
    /// All unique lemma strings.
    pub lemma_pool: Vec<String>,
    /// Short sense labels from Wiktionary (e.g. "castle", "lock").
    /// Indexed by `WordForm::definition`.
    pub definition_pool: Vec<String>,
    /// `(surface_form, [variant, …])` pairs sorted by surface_form.
    pub entries: Vec<(String, Vec<WordForm>)>,
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3 — Public lookup result types (returned to Python / JavaScript)
// ═══════════════════════════════════════════════════════════════════════════

/// One morphological analysis (POS + features + lemma) for a word form.
///
/// A single stress variant may represent multiple syncretised forms;
/// e.g. «ру́ки» is both nominative plural AND genitive singular of «рука».
/// In that case there will be two `MorphReading`s with the same `StressReading`.
///
/// All names follow the [Universal Dependencies](https://universaldependencies.org/)
/// annotation convention.
#[derive(Debug, Clone)]
pub struct MorphReading {
    /// Universal POS tags, e.g. `["NOUN"]`.  Typically one element.
    pub pos: Vec<String>,
    /// UD morphological features: key → allowed values.
    /// Multiple values per key encode syncretism
    /// (e.g. `Case → ["Acc", "Nom"]`).
    pub feats: HashMap<String, Vec<String>>,
    /// Base form (lemma), or `None` if not stored.
    pub lemma: Option<String>,
    /// Short sense label from Wiktionary (e.g. `"castle"` vs `"lock"` for «замок»).
    /// Used as a disambiguation hint when `pos` and `feats` are identical.
    /// `None` if not available in the source data.
    pub definition: Option<String>,
}

/// Complete analysis for **one stress variant** of a word form.
///
/// Carries three parallel representations of the same phonological unit:
///
/// | Level | Field | Example (`замок`, stress 0) |
/// |---|---|---|
/// | Written | `form` | `замок` |
/// | Stressed | `stressed_form` | `за́мок` |
/// | Syllabified written | `word_syllables` | `["за", "мок"]` |
/// | IPA | `ipa` | `zɑmɔk` |
/// | Syllabified IPA | `ipa_syllables` | `["ˈzɑ", "mɔk"]` |
///
/// `word_syllables[i]` and `ipa_syllables[i]` correspond to the same syllable.
/// Joining `word_syllables` reconstructs `form`.
/// Stripping leading `ˈ` from each `ipa_syllables[i]` and joining gives `ipa`.
#[derive(Debug, Clone)]
pub struct StressReading {
    // ── Stress position ───────────────────────────────────────────────────
    /// 0-based index of the stressed syllable (= stressed vowel index for
    /// normal words where each syllable has one vowel nucleus).
    /// `0` for zero-vowel (purely consonantal) words.
    pub syllable_index: usize,
    /// 1-based position from the end (2 = penultimate, 3 = antepenultimate).
    /// `0` for zero-vowel words.
    pub stress_from_end: usize,
    /// Number of syllables.  `1` for purely consonantal words (the whole
    /// word counts as one phonological unit).
    pub syllable_count: usize,

    // ── Written representation ────────────────────────────────────────────
    /// Normalized form: lowercased, canonical straight apostrophe `'`.
    pub form: String,
    /// `form` with a combining acute accent (U+0301) placed after the
    /// stressed vowel: `за́мок`.  Equals `form` when there is no vowel.
    pub stressed_form: String,
    /// Original graphemes grouped by syllable, aligned positionally with
    /// `ipa_syllables`.  Joining gives `form`.
    pub word_syllables: Vec<String>,

    // ── Phonetic representation ───────────────────────────────────────────
    /// Flat IPA string for the whole word.
    pub ipa: String,
    /// IPA per syllable.  The stressed syllable is prefixed with `ˈ` (U+02C8)
    /// following standard IPA notation.  No `ˈ` for purely consonantal words.
    pub ipa_syllables: Vec<String>,
    /// Token-level phonetic detail produced by the 6-pass pipeline.
    /// Useful for NLP/ML consumers; can be ignored for most use cases.
    pub tokens: Vec<PhoneticToken>,

    // ── Morphology ────────────────────────────────────────────────────────
    /// All morphological analyses that share this stress position.
    /// Multiple entries represent syncretism or same-stress homography.
    pub morph: Vec<MorphReading>,

    // ── Source quality ─────────────────────────────────────────────────────
    /// How stress was determined.  Always `None` for Ukrainian (all entries
    /// are confirmed dictionary forms).  `"exact"|"rule"|"default"` for Polish.
    pub confidence: Option<String>,
}

/// Top-level result of looking up a word form in the dictionary.
///
/// The engine **never picks one variant** — it returns all the data it holds
/// so that callers can apply their own resolution strategy (NLP context in
/// Python, tooltip in the browser, etc.).
#[derive(Debug, Clone)]
pub struct WordLookupResult {
    /// Normalized query form (lowercased, canonical apostrophe).
    pub form: String,
    /// All stress variants with full phonetics and morphology.
    /// Empty if the word is not in the dictionary.
    pub readings: Vec<StressReading>,
}
