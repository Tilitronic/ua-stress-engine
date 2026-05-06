//! data.rs — Static Ukrainian phoneme tables and lazy lookup maps.
//!
//! This module is the **single source of truth** for:
//!   - The consonant inventory (32 phonemes, 5-dimensional feature vector)
//!   - The vowel inventory (6 phonemes, 3-dimensional feature vector)
//!   - Grapheme-to-phoneme mappings (including digraphs, composites, iotation)
//!   - Voicing-assimilation pairs
//!   - Sibilant + affricate place-assimilation rules
//!   - Soft-map (hard IPA → palatalized IPA)
//!
//! Sources:
//!   Савченко І.С. (2014) — Фонетика, орфоепія і графіка сучасної украïнської мови
//!   Тоцька Н.І. (1997) — Фонетика, фонологія, орфоепія, графіка
//!   Мойсієнко А.К. et al. (2010) — Сучасна украïнська літературна мова

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::types::{
    ConsonantFeatures, Manner, Place, Softness, TokenType, VoicePower, VowelFeatures,
    VowelBackness, VowelHeight, VowelRounding,
};

// ── Consonant descriptor ────────────────────────────────────────────────────

/// Complete static descriptor for one consonant phoneme.
pub struct ConsonantDef {
    pub ipa: &'static str,
    pub voice_power: VoicePower,
    pub place: Place,
    pub manner: Manner,
    pub softness: Softness,
    /// IPA of the voiced counterpart (voicing assimilation target).
    /// `None` for sonorants and consonants with no voiced partner.
    pub voiced_pair: Option<&'static str>,
    /// IPA of the voiceless counterpart.
    pub voiceless_pair: Option<&'static str>,
}

impl ConsonantDef {
    pub fn to_features(&self) -> ConsonantFeatures {
        ConsonantFeatures {
            voice_power: self.voice_power,
            place: self.place,
            manner: self.manner,
            softness: self.softness,
            nasal: self.manner == Manner::Nasal,
        }
    }

    /// The `TokenType` produced by this consonant.  Glides (j) get their own type.
    pub fn token_type(&self) -> TokenType {
        if self.ipa == "j" {
            TokenType::Glide
        } else {
            TokenType::Consonant
        }
    }
}

// ── Consonant inventory — 32 phonemes + soft variants ──────────────────────
// Ordered: hard labials, hard dentals, hard velars/glottals, postalveolars,
//          affricates, sonorants — then their soft variants.
pub static CONSONANTS: &[ConsonantDef] = &[
    // ── Labial obstruents ─────────────────────────────────────────────────
    ConsonantDef { ipa: "b",  voice_power: VoicePower::Voiced,   place: Place::Labial,       manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("p")  },
    ConsonantDef { ipa: "p",  voice_power: VoicePower::Voiceless, place: Place::Labial,       manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: Some("b"), voiceless_pair: None       },
    ConsonantDef { ipa: "f",  voice_power: VoicePower::Voiceless, place: Place::Labial,       manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    // ── Dental obstruents ─────────────────────────────────────────────────
    ConsonantDef { ipa: "d",  voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("t")  },
    ConsonantDef { ipa: "t",  voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: Some("d"), voiceless_pair: None       },
    ConsonantDef { ipa: "z",  voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("s")  },
    ConsonantDef { ipa: "s",  voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: Some("z"), voiceless_pair: None       },
    // ── Dental affricates ─────────────────────────────────────────────────
    ConsonantDef { ipa: "dz", voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Affricate,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("ts") },
    ConsonantDef { ipa: "ts", voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Affricate,    softness: Softness::Hard, voiced_pair: Some("dz"),voiceless_pair: None       },
    // ── Velar obstruents ──────────────────────────────────────────────────
    ConsonantDef { ipa: "ɡ",  voice_power: VoicePower::Voiced,   place: Place::Velar,         manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("k")  },
    ConsonantDef { ipa: "k",  voice_power: VoicePower::Voiceless, place: Place::Velar,         manner: Manner::Plosive,      softness: Softness::Hard, voiced_pair: Some("ɡ"), voiceless_pair: None       },
    ConsonantDef { ipa: "x",  voice_power: VoicePower::Voiceless, place: Place::Velar,         manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: Some("ɦ"), voiceless_pair: None       },
    // ── Glottal ───────────────────────────────────────────────────────────
    ConsonantDef { ipa: "ɦ",  voice_power: VoicePower::Voiced,   place: Place::Glottal,       manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("x")  },
    // ── Postalveolar obstruents ───────────────────────────────────────────
    ConsonantDef { ipa: "ʒ",  voice_power: VoicePower::Voiced,   place: Place::Postalveolar,  manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("ʃ")  },
    ConsonantDef { ipa: "ʃ",  voice_power: VoicePower::Voiceless, place: Place::Postalveolar,  manner: Manner::Fricative,    softness: Softness::Hard, voiced_pair: Some("ʒ"), voiceless_pair: None       },
    ConsonantDef { ipa: "dʒ", voice_power: VoicePower::Voiced,   place: Place::Postalveolar,  manner: Manner::Affricate,    softness: Softness::Hard, voiced_pair: None,      voiceless_pair: Some("tʃ") },
    ConsonantDef { ipa: "tʃ", voice_power: VoicePower::Voiceless, place: Place::Postalveolar,  manner: Manner::Affricate,    softness: Softness::Hard, voiced_pair: Some("dʒ"),voiceless_pair: None       },
    // ── Sonorants (hard) ──────────────────────────────────────────────────
    ConsonantDef { ipa: "m",  voice_power: VoicePower::Sonorant,  place: Place::Labial,        manner: Manner::Nasal,        softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "n",  voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Nasal,        softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "l",  voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Lateral,      softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "r",  voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Trill,        softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "j",  voice_power: VoicePower::Sonorant,  place: Place::Palatal,       manner: Manner::Approximant,  softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
    /// Ukrainian /в/ — classified as sonorant (labial approximant).
    /// This is critical: it does NOT trigger voicing assimilation.
    ConsonantDef { ipa: "ʋ",  voice_power: VoicePower::Sonorant,  place: Place::Labial,        manner: Manner::Approximant,  softness: Softness::Hard, voiced_pair: None,      voiceless_pair: None       },
    // ── Soft (palatalized) variants ───────────────────────────────────────
    // Dental soft variants (participate in regressive palatalization propagation):
    ConsonantDef { ipa: "dʲ", voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Plosive,      softness: Softness::Soft, voiced_pair: None,      voiceless_pair: Some("tʲ") },
    ConsonantDef { ipa: "tʲ", voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Plosive,      softness: Softness::Soft, voiced_pair: Some("dʲ"),voiceless_pair: None       },
    ConsonantDef { ipa: "zʲ", voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Fricative,    softness: Softness::Soft, voiced_pair: None,      voiceless_pair: Some("sʲ") },
    ConsonantDef { ipa: "sʲ", voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Fricative,    softness: Softness::Soft, voiced_pair: Some("zʲ"),voiceless_pair: None       },
    ConsonantDef { ipa: "dzʲ",voice_power: VoicePower::Voiced,   place: Place::Dental,        manner: Manner::Affricate,    softness: Softness::Soft, voiced_pair: None,      voiceless_pair: Some("tsʲ")},
    ConsonantDef { ipa: "tsʲ",voice_power: VoicePower::Voiceless, place: Place::Dental,        manner: Manner::Affricate,    softness: Softness::Soft, voiced_pair: Some("dzʲ"),voiceless_pair: None      },
    ConsonantDef { ipa: "nʲ", voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Nasal,        softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "lʲ", voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Lateral,      softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "rʲ", voice_power: VoicePower::Sonorant,  place: Place::Dental,        manner: Manner::Trill,        softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
    // Labial soft variants (produced by iotated vowels after labials, but
    // do NOT propagate regressive palatalization):
    ConsonantDef { ipa: "bʲ", voice_power: VoicePower::Voiced,   place: Place::Labial,        manner: Manner::Plosive,      softness: Softness::Soft, voiced_pair: None,      voiceless_pair: Some("pʲ") },
    ConsonantDef { ipa: "pʲ", voice_power: VoicePower::Voiceless, place: Place::Labial,        manner: Manner::Plosive,      softness: Softness::Soft, voiced_pair: Some("bʲ"),voiceless_pair: None       },
    ConsonantDef { ipa: "mʲ", voice_power: VoicePower::Sonorant,  place: Place::Labial,        manner: Manner::Nasal,        softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
    ConsonantDef { ipa: "fʲ", voice_power: VoicePower::Voiceless, place: Place::Labial,        manner: Manner::Fricative,    softness: Softness::Soft, voiced_pair: None,      voiceless_pair: None       },
];

// ── Vowel inventory ─────────────────────────────────────────────────────────

/// Static descriptor for one vowel phoneme.
pub struct VowelDef {
    pub ipa: &'static str,
    pub height: VowelHeight,
    pub backness: VowelBackness,
    pub rounding: VowelRounding,
    /// Stable vowels don't shift in unstressed position.
    pub stable: bool,
}

impl VowelDef {
    pub fn to_features(&self) -> VowelFeatures {
        VowelFeatures {
            height: self.height,
            backness: self.backness,
            rounding: self.rounding,
            stable: self.stable,
        }
    }
}

pub static VOWELS: &[VowelDef] = &[
    VowelDef { ipa: "i",  height: VowelHeight::High,    backness: VowelBackness::Front,   rounding: VowelRounding::Unrounded, stable: true  },
    VowelDef { ipa: "ɪ",  height: VowelHeight::HighMid, backness: VowelBackness::Front,   rounding: VowelRounding::Unrounded, stable: false },
    VowelDef { ipa: "ɛ",  height: VowelHeight::Mid,     backness: VowelBackness::Front,   rounding: VowelRounding::Unrounded, stable: false },
    VowelDef { ipa: "u",  height: VowelHeight::High,    backness: VowelBackness::Back,    rounding: VowelRounding::Rounded,   stable: true  },
    VowelDef { ipa: "ɔ",  height: VowelHeight::Mid,     backness: VowelBackness::Back,    rounding: VowelRounding::Rounded,   stable: false },
    VowelDef { ipa: "ɑ",  height: VowelHeight::Low,     backness: VowelBackness::Back,    rounding: VowelRounding::Unrounded, stable: true  },
];

// ── Lazy lookup maps (built once from the static slices above) ──────────────

/// IPA string → ConsonantDef reference.
pub static CONSONANT_MAP: Lazy<HashMap<&'static str, &'static ConsonantDef>> =
    Lazy::new(|| CONSONANTS.iter().map(|c| (c.ipa, c)).collect());

/// IPA string → VowelDef reference.
pub static VOWEL_MAP: Lazy<HashMap<&'static str, &'static VowelDef>> =
    Lazy::new(|| VOWELS.iter().map(|v| (v.ipa, v)).collect());

/// Voiced ↔ voiceless pairing: each IPA maps to its opposite voicing partner.
/// Used bidirectionally by the voicing assimilation pass.
pub static VOICING_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for c in CONSONANTS {
        if let Some(vp) = c.voiced_pair    { m.insert(c.ipa, vp); }
        if let Some(vl) = c.voiceless_pair { m.insert(c.ipa, vl); }
    }
    // The voicing pass needs to look up current_ipa → paired_ipa.
    // We build both directions explicitly from the data above.
    // (Some entries may be overwritten; both directions are correct.)
    m
});

/// IPA sets used by the voicing assimilation pass.
pub static VOICED_OBSTRUENTS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    ["b", "d", "dʲ", "ɡ", "z", "zʲ", "ʒ", "ɦ", "dz", "dzʲ", "dʒ"]
        .iter()
        .copied()
        .collect()
});
pub static VOICELESS_OBSTRUENTS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    ["p", "t", "tʲ", "k", "s", "sʲ", "ʃ", "x", "f", "ts", "tsʲ", "tʃ"]
        .iter()
        .copied()
        .collect()
});
/// Sonorants + glides that are exempt from voicing assimilation.
pub static SONORANT_EXEMPT: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    ["m", "n", "nʲ", "l", "lʲ", "r", "rʲ", "ʋ", "j", "w", "u\u{032f}"]
        .iter()
        .copied()
        .collect()
});

// ── Grapheme → IPA mappings ─────────────────────────────────────────────────

/// Simple consonant Cyrillic → IPA (single grapheme, hard variant).
pub static CONSONANT_GRAPHEMES: &[(&str, &str)] = &[
    ("б", "b"), ("в", "ʋ"), ("г", "ɦ"), ("ґ", "ɡ"),
    ("д", "d"), ("ж", "ʒ"), ("з", "z"), ("й", "j"),
    ("к", "k"), ("л", "l"), ("м", "m"), ("н", "n"),
    ("п", "p"), ("р", "r"), ("с", "s"), ("т", "t"),
    ("ф", "f"), ("х", "x"), ("ц", "ts"), ("ч", "tʃ"),
    ("ш", "ʃ"),
];

/// Digraph Cyrillic sequences → single IPA affricate.
/// Must be checked BEFORE single graphemes.
pub static DIGRAPHS: &[(&str, &str)] = &[
    ("дж", "dʒ"),
    ("дз", "dz"),
];

/// Composite grapheme → two IPA tokens (щ = ʃtʃ).
pub static COMPOSITES: &[(&str, &[&str])] = &[
    ("щ", &["ʃ", "tʃ"]),
];

/// Simple vowel Cyrillic → IPA.
pub static SIMPLE_VOWELS: &[(&str, &str)] = &[
    ("а", "ɑ"), ("е", "ɛ"), ("и", "ɪ"),
    ("і", "i"), ("о", "ɔ"), ("у", "u"),
];

/// Iotated vowel → (vowel_ipa, glide_ipa, always_iotated).
/// `always_iotated` means the glide is produced even after a consonant (ї).
pub static IOTATED_VOWELS: &[(&str, &str, &str, bool)] = &[
    // grapheme, vowel_ipa, glide_ipa, always_iotated
    ("я", "ɑ", "j", false),
    ("ю", "u", "j", false),
    ("є", "ɛ", "j", false),
    ("ї", "i", "j", true),
];

/// Ukrainian vowel graphemes (needed to decide iotation context).
pub static VOWEL_GRAPHEMES: &[&str] = &[
    "а", "е", "є", "и", "і", "ї", "о", "у", "ю", "я",
];

/// Ukrainian vowel IPA symbols (for stress-mark placement and syllabification).
pub const UA_VOWEL_IPAS: &str = "aeiouɑɛɪɔuiʊ";
// A more precise set:
pub static VOWEL_IPA_SET: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    ["i", "ɪ", "ɛ", "u", "ɔ", "ɑ",
     // allophone variants
     "ɛ\u{031d}", "ɪ\u{031e}", "ɔ\u{031d}"]
        .iter().copied().collect()
});

/// Soft sign grapheme.
pub const SOFT_SIGN: char = 'ь';

/// Apostrophe characters recognized as word-internal boundaries.
pub static APOSTROPHE_CHARS: Lazy<std::collections::HashSet<char>> = Lazy::new(|| {
    ['\'', '\u{02bc}', '\u{2019}', '\u{0027}', '\u{2018}', '\u{0060}']
        .iter().copied().collect()
});

/// Dental IPA consonants that participate in REGRESSIVE palatalization
/// propagation (non-dentals are blocked).
pub static DENTAL_CONSONANTS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    CONSONANTS.iter()
        .filter(|c| c.place == Place::Dental)
        .map(|c| c.ipa)
        .collect()
});

/// Consonants that CAN be softened by the palatalization pass.
/// Labials and velars are excluded — they don't propagate softness regressively.
pub static SOFTENABLE_CONSONANTS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    ["d", "t", "z", "s", "dz", "ts", "l", "n", "r",
     "dʲ", "tʲ", "zʲ", "sʲ", "dzʲ", "tsʲ", "lʲ", "nʲ", "rʲ"]
        .iter().copied().collect()
});

/// Hard IPA → palatalized IPA for the softening pass.
pub static SOFT_MAP: &[(&str, &str)] = &[
    ("d", "dʲ"), ("t", "tʲ"), ("z", "zʲ"), ("s", "sʲ"),
    ("dz", "dzʲ"), ("ts", "tsʲ"), ("l", "lʲ"), ("n", "nʲ"), ("r", "rʲ"),
];

pub static SOFT_MAP_LOOKUP: Lazy<HashMap<&'static str, &'static str>> =
    Lazy::new(|| SOFT_MAP.iter().copied().collect());

// ── Sibilant place-assimilation rules ───────────────────────────────────────
// Direction: regressive. target_ipa + trigger_ipa → result_ipa for target.
// (The trigger token itself does not change.)

pub struct SibilantRule {
    pub target: &'static str,   // IPA of the token that changes
    pub trigger: &'static str,  // IPA of the immediately following token
    pub result: &'static str,   // IPA the target becomes
}

pub static SIBILANT_RULES: &[SibilantRule] = &[
    // з/с + ш → шш   (зшити → [ʃʃɪtɑ])
    SibilantRule { target: "z",  trigger: "ʃ",  result: "ʃ"  },
    SibilantRule { target: "s",  trigger: "ʃ",  result: "ʃ"  },
    // з/с + ж → жж
    SibilantRule { target: "z",  trigger: "ʒ",  result: "ʒ"  },
    SibilantRule { target: "s",  trigger: "ʒ",  result: "ʒ"  },
    // з/с + ч → шч  (zч → ʃtʃ)
    SibilantRule { target: "z",  trigger: "tʃ", result: "ʃ"  },
    SibilantRule { target: "s",  trigger: "tʃ", result: "ʃ"  },
    // з/с + дж → ждж
    SibilantRule { target: "z",  trigger: "dʒ", result: "ʒ"  },
    // зь/сь + ш → шш
    SibilantRule { target: "zʲ", trigger: "ʃ",  result: "ʃ"  },
    SibilantRule { target: "sʲ", trigger: "ʃ",  result: "ʃ"  },
    // зь/сь + ч → шч
    SibilantRule { target: "zʲ", trigger: "tʃ", result: "ʃ"  },
    SibilantRule { target: "sʲ", trigger: "tʃ", result: "ʃ"  },
];

// ── Affricate place-assimilation rules ──────────────────────────────────────
// Direction: regressive. first + second → result (second token is REMOVED).

pub struct AffricateRule {
    pub first: &'static str,   // IPA of the first token (gets replaced)
    pub second: &'static str,  // IPA of the second token (removed after merge)
    pub result: &'static str,  // IPA that replaces the first token
}

pub static AFFRICATE_RULES: &[AffricateRule] = &[
    // т + с → ц   (братство → [brɑtstvɔ] → [brɑtsvɔ] ... actually [brɑtsʲtvɔ])
    AffricateRule { first: "t",  second: "s",   result: "ts"  },
    // ть + с → ць
    AffricateRule { first: "tʲ", second: "s",   result: "tsʲ" },
    // т + ц → цц (geminate)
    AffricateRule { first: "t",  second: "ts",  result: "tsː" },
];

// ── /в/ positional allophones ────────────────────────────────────────────────

pub const V_DEFAULT: &str          = "ʋ";   // word-initial, before vowel
pub const V_POST_VOCALIC_PRE_C: &str = "w"; // after vowel, before consonant
pub const V_WORD_FINAL: &str       = "u\u{032f}"; // word-final after vowel (u̯)
