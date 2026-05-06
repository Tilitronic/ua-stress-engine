//! tokenizer.rs — Pass 1: Cyrillic graphemes → PhoneticToken[]
//!
//! Scans the input word left-to-right and produces an ordered array of
//! [`PhoneticToken`] objects.  This pass is **data-driven**: all grapheme
//! tables are imported from `data.rs`.
//!
//! Handled here:
//!   - Digraph affricates: дж, дз
//!   - Composite щ → [ʃ] + [tʃ]
//!   - Soft sign (ь) → merges palatalization into the preceding consonant
//!   - Apostrophe variants → glottal boundary token
//!   - Iotated vowels (я, ю, є, ї) → j+vowel or consonant palatalization
//!   - Simple vowels (а, е, и, і, о, у) → vowel token
//!   - All other consonants → consonant token
//!
//! Does NOT handle: palatalization propagation, voicing assimilation,
//! vowel allophones, /в/ allophones — those are separate passes.

use crate::data::{
    APOSTROPHE_CHARS, COMPOSITES, CONSONANT_GRAPHEMES, CONSONANT_MAP, DIGRAPHS, IOTATED_VOWELS,
    SIMPLE_VOWELS, SOFT_SIGN, VOWEL_GRAPHEMES, VOWEL_MAP,
};
use crate::types::{ConsonantFeatures, PhoneticToken, Softness, TokenType};

// ── Token factory helpers ────────────────────────────────────────────────────

/// Produce a consonant token, optionally palatalized.
///
/// If `palatalized` is true and the IPA doesn't already end with ʲ,
/// the diacritic is appended and the palatalized feature variant is
/// looked up first in the table.
fn make_consonant_token(ipa: &str, source: &str, palatalized: bool) -> PhoneticToken {
    // When palatalized, try the ʲ-variant first in the feature map
    let lookup_ipa = if palatalized && !ipa.ends_with('ʲ') {
        format!("{}ʲ", ipa)
    } else {
        ipa.to_string()
    };
    let final_ipa = lookup_ipa.clone();

    let features: Option<ConsonantFeatures> = CONSONANT_MAP
        .get(lookup_ipa.as_str())
        .or_else(|| CONSONANT_MAP.get(ipa))
        .map(|def| {
            let mut f = def.to_features();
            if palatalized {
                f.softness = Softness::Soft;
            }
            f
        });

    let tok_type = if final_ipa == "j" || final_ipa == "ʋ" {
        // ʋ (в) starts as Consonant; vAllophonePass may promote it to Glide
        if final_ipa == "j" { TokenType::Glide } else { TokenType::Consonant }
    } else {
        TokenType::Consonant
    };

    PhoneticToken {
        ipa: final_ipa,
        source: source.to_string(),
        token_type: tok_type,
        vowel_index: -1,
        stressed: false,
        palatalized,
        consonant_features: features,
        vowel_features: None,
    }
}

/// Produce a vowel token with the correct vowel_index.
fn make_vowel_token(ipa: &str, source: &str, vowel_index: i8) -> PhoneticToken {
    let features = VOWEL_MAP.get(ipa).map(|def| def.to_features());
    PhoneticToken {
        ipa: ipa.to_string(),
        source: source.to_string(),
        token_type: TokenType::Vowel,
        vowel_index,
        stressed: false, // filled in by vowel_allophones pass
        palatalized: false,
        consonant_features: None,
        vowel_features: features,
    }
}

/// Produce a glide token (j produced by iotated vowels).
fn make_glide_token(ipa: &str, source: &str) -> PhoneticToken {
    let features = CONSONANT_MAP.get(ipa).map(|def| def.to_features());
    PhoneticToken {
        ipa: ipa.to_string(),
        source: source.to_string(),
        token_type: TokenType::Glide,
        vowel_index: -1,
        stressed: false,
        palatalized: false,
        consonant_features: features,
        vowel_features: None,
    }
}

/// Produce a glottal stop boundary token for the apostrophe.
fn make_glottal_token(source: &str) -> PhoneticToken {
    PhoneticToken {
        ipa: "ʔ".to_string(),
        source: source.to_string(),
        token_type: TokenType::Glottal,
        vowel_index: -1,
        stressed: false,
        palatalized: false,
        consonant_features: None,
        vowel_features: None,
    }
}

// ── Iotation context ─────────────────────────────────────────────────────────

/// Decide whether an iotated vowel at `index` in `chars` should produce
/// [j]+vowel (iotation) or palatalize the preceding consonant.
///
/// Iotation occurs when:
/// - word-initial position
/// - after another vowel grapheme
/// - after an apostrophe (any variant)
/// - after soft sign (ь)
/// - `always_iotated` flag is set (ї is always iotated)
///
/// Palatalization occurs:
/// - after a consonant grapheme (the consonant gets ʲ; vowel produced without j)
fn should_iotate(chars: &[char], index: usize, always_iotated: bool) -> bool {
    if always_iotated || index == 0 {
        return true;
    }
    let prev = chars[index - 1];
    // After apostrophe → iotation (м'яч: apostrophe separates the labial from the vowel)
    if APOSTROPHE_CHARS.contains(&prev) {
        return true;
    }
    // After soft sign → iotation (the ь already palatalized the preceding consonant)
    if prev == SOFT_SIGN {
        return true;
    }
    // After a vowel grapheme → iotation (hiatus context)
    if VOWEL_GRAPHEMES.contains(&prev.to_string().as_str()) {
        return true;
    }
    // After a consonant → palatalize that consonant instead
    false
}

// ── Main tokenizer ────────────────────────────────────────────────────────────

/// Tokenize a Ukrainian word into [`PhoneticToken`]s.
///
/// All passes downstream consume and/or mutate this token array.
///
/// # Arguments
/// * `word` — Ukrainian word in Cyrillic (any case; normalize before calling)
pub fn tokenize(word: &str) -> Vec<PhoneticToken> {
    let chars: Vec<char> = word.chars().collect();
    let mut tokens: Vec<PhoneticToken> = Vec::with_capacity(chars.len() + 4);
    let mut vowel_index: i8 = 0;
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        let ch_str = ch.to_string();

        // ── Digraph check (дж, дз): consume two chars ──────────────────
        if i + 1 < chars.len() {
            let bigram = format!("{}{}", ch, chars[i + 1]);
            if let Some(&ipa) = DIGRAPHS.iter().find_map(|(g, p)| (*g == bigram).then_some(p)) {
                tokens.push(make_consonant_token(ipa, &bigram, false));
                i += 2;
                continue;
            }
        }

        // ── Composite щ → two tokens ────────────────────────────────────
        if let Some(&(_, ipas)) = COMPOSITES.iter().find(|(g, _)| **g == ch_str) {
            for (k, &ipa) in ipas.iter().enumerate() {
                let src = if k == 0 { ch_str.clone() } else { String::new() };
                tokens.push(make_consonant_token(ipa, &src, false));
            }
            i += 1;
            continue;
        }

        // ── Soft sign (ь) → palatalize the preceding consonant ──────────
        if ch == SOFT_SIGN {
            // Find the last consonant token and soften it
            if let Some(prev_tok) = tokens.iter_mut().rev().find(|t| {
                t.token_type == TokenType::Consonant && !t.ipa.ends_with('ʲ')
            }) {
                let hard_ipa = prev_tok.ipa.clone();
                prev_tok.ipa = format!("{}ʲ", hard_ipa);
                prev_tok.palatalized = true;
                prev_tok.source.push(SOFT_SIGN);
                if let Some(f) = &mut prev_tok.consonant_features {
                    f.softness = Softness::Soft;
                }
            }
            // ь itself produces no separate token
            i += 1;
            continue;
        }

        // ── Apostrophe → glottal boundary ───────────────────────────────
        if APOSTROPHE_CHARS.contains(&ch) {
            tokens.push(make_glottal_token(&ch_str));
            i += 1;
            continue;
        }

        // ── Iotated vowels (я, ю, є, ї) ─────────────────────────────────
        if let Some(&(_, vowel_ipa, glide_ipa, always_iotated)) =
            IOTATED_VOWELS.iter().find(|(g, _, _, _)| **g == ch_str)
        {
            if should_iotate(&chars, i, always_iotated) {
                // Produce [j] glide followed by the vowel
                tokens.push(make_glide_token(glide_ipa, &ch_str));
                tokens.push(make_vowel_token(vowel_ipa, "", vowel_index));
            } else {
                // After a consonant: palatalize the preceding consonant and
                // produce the bare vowel (no j)
                if let Some(prev_tok) = tokens.iter_mut().rev().find(|t| {
                    t.token_type == TokenType::Consonant
                }) {
                    if !prev_tok.ipa.ends_with('ʲ') {
                        let hard_ipa = prev_tok.ipa.clone();
                        // Try to get the palatalized variant
                        let soft_ipa = format!("{}ʲ", hard_ipa);
                        prev_tok.ipa = soft_ipa;
                        prev_tok.palatalized = true;
                        if let Some(f) = &mut prev_tok.consonant_features {
                            f.softness = Softness::Soft;
                        }
                    }
                }
                tokens.push(make_vowel_token(vowel_ipa, &ch_str, vowel_index));
            }
            vowel_index += 1;
            i += 1;
            continue;
        }

        // ── Simple vowels ────────────────────────────────────────────────
        if let Some(&(_, ipa)) = SIMPLE_VOWELS.iter().find(|(g, _)| **g == ch_str) {
            tokens.push(make_vowel_token(ipa, &ch_str, vowel_index));
            vowel_index += 1;
            i += 1;
            continue;
        }

        // ── Consonants ───────────────────────────────────────────────────
        if let Some(&(_, ipa)) = CONSONANT_GRAPHEMES.iter().find(|(g, _)| **g == ch_str) {
            tokens.push(make_consonant_token(ipa, &ch_str, false));
            i += 1;
            continue;
        }

        // ── Unknown character: skip silently ─────────────────────────────
        i += 1;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_consonants() {
        let tokens = tokenize("бпдтк");
        let ipas: Vec<&str> = tokens.iter().map(|t| t.ipa.as_str()).collect();
        assert_eq!(ipas, ["b", "p", "d", "t", "k"]);
    }

    #[test]
    fn ukrainian_specific_consonants() {
        // г → ɦ (not ɡ), ґ → ɡ, в → ʋ
        let tokens = tokenize("гґв");
        let ipas: Vec<&str> = tokens.iter().map(|t| t.ipa.as_str()).collect();
        assert_eq!(ipas, ["ɦ", "ɡ", "ʋ"]);
    }

    #[test]
    fn digraph_dzh() {
        let tokens = tokenize("джміль");
        assert_eq!(tokens[0].ipa, "dʒ");
        assert_eq!(tokens[0].source, "дж");
    }

    #[test]
    fn composite_shch() {
        // щ → ʃ + tʃ
        let tokens = tokenize("щука");
        assert_eq!(tokens[0].ipa, "ʃ");
        assert_eq!(tokens[1].ipa, "tʃ");
    }

    #[test]
    fn simple_vowels_mapped() {
        let tokens = tokenize("аеиіоу");
        let ipas: Vec<&str> = tokens.iter().map(|t| t.ipa.as_str()).collect();
        assert_eq!(ipas, ["ɑ", "ɛ", "ɪ", "i", "ɔ", "u"]);
    }

    #[test]
    fn iotated_word_initial_produces_glide() {
        // яма → j ɑ м ɑ
        let tokens = tokenize("яма");
        assert_eq!(tokens[0].ipa, "j");
        assert_eq!(tokens[0].token_type, TokenType::Glide);
        assert_eq!(tokens[1].ipa, "ɑ");
        assert_eq!(tokens[1].token_type, TokenType::Vowel);
    }

    #[test]
    fn iotated_after_consonant_palatalizes() {
        // ня → nʲ ɑ
        let tokens = tokenize("ня");
        assert_eq!(tokens[0].ipa, "nʲ");
        assert!(tokens[0].palatalized);
        assert_eq!(tokens[1].ipa, "ɑ");
    }

    #[test]
    fn yi_always_iotated() {
        // їх → j i x
        let tokens = tokenize("їх");
        assert_eq!(tokens[0].ipa, "j");
        assert_eq!(tokens[1].ipa, "i");
    }

    #[test]
    fn soft_sign_palatalizes_preceding_consonant() {
        // день → d ɛ nʲ  (ь merged into н)
        let tokens = tokenize("день");
        let ipas: Vec<&str> = tokens.iter().map(|t| t.ipa.as_str()).collect();
        assert_eq!(ipas, ["d", "ɛ", "nʲ"]);
        assert!(tokens[2].palatalized);
    }

    #[test]
    fn apostrophe_produces_glottal() {
        // м'яч → m ʔ j ɑ tʃ
        let tokens = tokenize("м\u{02bc}яч");
        assert_eq!(tokens[1].ipa, "ʔ");
        assert_eq!(tokens[1].token_type, TokenType::Glottal);
        // After apostrophe, я is iotated: j + ɑ
        assert_eq!(tokens[2].ipa, "j");
        assert_eq!(tokens[3].ipa, "ɑ");
    }

    #[test]
    fn vowel_indices_assigned() {
        // мама — two vowels at indices 0 and 1
        let tokens = tokenize("мама");
        let vowels: Vec<i8> = tokens.iter().map(|t| t.vowel_index).collect();
        assert_eq!(vowels, [-1, 0, -1, 1]);
    }
}
