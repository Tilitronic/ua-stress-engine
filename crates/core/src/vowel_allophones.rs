//! vowel_allophones.rs — Pass 4: Stressed/unstressed vowel variants.
//!
//! In Ukrainian, vowels in unstressed positions undergo reduction.
//! This module implements two tasks:
//!
//! 1. **Mark the stressed vowel** using the `stress_index` argument
//!    (0-based index into the sequence of vowels in the word).
//!    Sets `token.stressed = true`.
//!
//! 2. **Apply allophonic substitution** for unstressed vowels:
//!    - /ɛ/ (е) in an unstressed syllable before a syllable with a
//!      stressed high vowel → ɛ̝ (raised е, a high-mid front vowel)
//!    - /ɪ/ (и) in an unstressed syllable before a stressed high vowel → ɪ̞
//!    - /ɔ/ (о) in the same context → ɔ̝
//!
//! The "before stressed high vowel" rule targets the pre-stressed syllable only.
//! High vowels are /i/ and /u/.
//!
//! References:
//!   Тоцька 1997 §14: unstressed і, и, е raise toward і; о, у retain quality.
//!   The allophone notations follow the practice of the TypeScript phonetic pipeline.

use crate::types::{PhoneticToken, TokenType, VowelHeight};

/// High vowel IPA symbols that trigger allophonic raising in the preceding unstressed vowel.
const HIGH_VOWELS: &[&str] = &["i", "u"];

/// Apply stressed vowel marking and allophonic substitution.
///
/// # Arguments
/// * `tokens` — mutable slice to modify in place
/// * `stress_index` — 0-based vowel index that carries the word stress
pub fn apply_vowel_allophones(tokens: &mut Vec<PhoneticToken>, stress_index: u8) {
    // Phase 1 — mark the stressed vowel
    for tok in tokens.iter_mut() {
        if tok.token_type == TokenType::Vowel && tok.vowel_index == stress_index as i8 {
            tok.stressed = true;
        }
    }

    // Phase 2 — find the stressed vowel's IPA; if it's a high vowel,
    // apply allophonic raising to the vowel immediately preceding it.
    let stressed_ipa: Option<String> = tokens
        .iter()
        .find(|t| t.token_type == TokenType::Vowel && t.stressed)
        .map(|t| t.ipa.clone());

    let is_high_stressed = match &stressed_ipa {
        Some(ipa) => HIGH_VOWELS.contains(&ipa.as_str()),
        None => false,
    };

    if !is_high_stressed {
        return; // No allophonic raising needed
    }

    // The pre-stressed vowel is the last unstressed vowel before the stressed one.
    let stressed_vowel_pos = tokens
        .iter()
        .position(|t| t.token_type == TokenType::Vowel && t.stressed);

    if let Some(stressed_pos) = stressed_vowel_pos {
        // Find the last vowel token BEFORE stressed_pos
        let pre_stressed_vowel_pos = tokens[..stressed_pos]
            .iter()
            .rposition(|t| t.token_type == TokenType::Vowel);

        if let Some(idx) = pre_stressed_vowel_pos {
            let ipa = tokens[idx].ipa.clone();
            let new_ipa = match ipa.as_str() {
                "ɛ" => Some("ɛ\u{031d}"), // ɛ̝ — raised
                "ɪ" => Some("ɪ\u{031e}"), // ɪ̞ — lowered (closer to і)
                "ɔ" => Some("ɔ\u{031d}"), // ɔ̝ — raised
                _ => None,
            };
            if let Some(new_ipa_str) = new_ipa {
                tokens[idx].ipa = new_ipa_str.to_string();
                // Update features if present
                if let Some(f) = &mut tokens[idx].vowel_features {
                    f.height = VowelHeight::HighMid; // approximate — both "raised" variants
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn vowel_ipas_with_stress(word: &str, stress: u8) -> Vec<String> {
        let mut tokens = tokenize(word);
        apply_vowel_allophones(&mut tokens, stress);
        tokens
            .iter()
            .filter(|t| t.token_type == TokenType::Vowel)
            .map(|t| t.ipa.clone())
            .collect()
    }

    fn stressed_vowel(word: &str, stress: u8) -> Option<String> {
        let mut tokens = tokenize(word);
        apply_vowel_allophones(&mut tokens, stress);
        tokens
            .iter()
            .find(|t| t.token_type == TokenType::Vowel && t.stressed)
            .map(|t| t.ipa.clone())
    }

    #[test]
    fn marks_stressed_vowel() {
        // "мама" stress on first vowel (index 0)
        assert_eq!(stressed_vowel("мама", 0).as_deref(), Some("ɑ"));
    }

    #[test]
    fn marks_second_vowel_stressed() {
        assert_eq!(stressed_vowel("мама", 1).as_deref(), Some("ɑ"));
    }

    #[test]
    fn e_raises_before_high_stressed_vowel() {
        // "село" stress=1 (stressed "о"): not high → no raising
        let ipas = vowel_ipas_with_stress("село", 1);
        assert_eq!(ipas[0], "ɛ", "No raising before non-high vowel: {ipas:?}");
    }

    #[test]
    fn e_raises_before_stressed_i() {
        // "деліть" stress=1 (і is high)
        // Tokenizer: d ɛ l i tʲ
        // e (index 0) before stressed і (index 1) → ɛ̝
        let ipas = vowel_ipas_with_stress("делі", 1);
        assert_eq!(
            ipas[0], "ɛ\u{031d}",
            "ɛ should raise to ɛ̝ before stressed і: {ipas:?}"
        );
    }

    #[test]
    fn y_raises_before_stressed_i() {
        // "кидай" stress=0 — к і (high, stressed) d ɑ j
        // Before stressed і there is no unstressed vowel in that example.
        // Use "силіт" (made-up): s ɪ l i t → ɪ before stressed і → ɪ̞
        let ipas = vowel_ipas_with_stress("силі", 1);
        assert_eq!(
            ipas[0], "ɪ\u{031e}",
            "ɪ should lower to ɪ̞ before stressed і: {ipas:?}"
        );
    }
}
