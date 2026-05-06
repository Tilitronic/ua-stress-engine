//! normalize.rs — Word normalization before dictionary lookup or transcription.
//!
//! Converts any apostrophe variant to the canonical modifier letter (U+02BC, ʼ)
//! that is used as the storage key in the binary dictionary, then lowercases.
//!
//! The canonical apostrophe ʼ is also what the tokenizer recognizes as a
//! word-internal boundary that triggers iotation of the following vowel.

use crate::data::APOSTROPHE_CHARS;

/// Normalize a Ukrainian word for dictionary lookup or phonetic processing.
///
/// Steps applied (in order):
///   1. Lowercase each character.
///   2. Replace every recognized apostrophe variant with the canonical ʼ (U+02BC).
///
/// # Example
/// ```
/// use ua_stress_core::normalize_word;
/// assert_eq!(normalize_word("М'яч"), "м\u{02bc}яч");
/// ```
pub fn normalize_word(word: &str) -> String {
    word.chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| {
            if APOSTROPHE_CHARS.contains(&c) {
                '\u{02bc}' // ʼ — modifier letter apostrophe
            } else {
                c
            }
        })
        .collect()
}

/// Apply combining acute accent (U+0301) after each stressed vowel.
///
/// `stress_variants` contains 0-based indices into the sequence of Ukrainian
/// vowels in the word.  All listed positions are marked.
///
/// # Example
/// ```
/// use ua_stress_core::apply_stress_marks;
/// assert_eq!(apply_stress_marks("мама", &[0]), "ма\u{0301}ма");
/// ```
pub fn apply_stress_marks(word: &str, stress_variants: &[u8]) -> String {
    const COMBINING_ACUTE: char = '\u{0301}';
    const UA_VOWELS: &str = "аеєиіїоуюяАЕЄИІЇОУЮЯ";

    let mut out = String::with_capacity(word.len() + stress_variants.len() * 3);
    let mut vowel_idx: u8 = 0;
    for ch in word.chars() {
        out.push(ch);
        if UA_VOWELS.contains(ch) {
            if stress_variants.contains(&vowel_idx) {
                out.push(COMBINING_ACUTE);
            }
            vowel_idx += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_input() {
        assert_eq!(normalize_word("Мама"), "мама");
    }

    #[test]
    fn replaces_apostrophe_variants() {
        // straight apostrophe
        assert_eq!(normalize_word("м'яч"), "м\u{02bc}яч");
        // curly right quote
        assert_eq!(normalize_word("м\u{2019}яч"), "м\u{02bc}яч");
        // canonical ʼ stays unchanged
        assert_eq!(normalize_word("м\u{02bc}яч"), "м\u{02bc}яч");
    }

    #[test]
    fn stress_marks_single() {
        // мама, stress on first vowel (index 0)
        assert_eq!(apply_stress_marks("мама", &[0]), "ма\u{0301}ма");
        // stress on second vowel (index 1)
        assert_eq!(apply_stress_marks("мама", &[1]), "мама\u{0301}");
    }

    #[test]
    fn stress_marks_heteronym() {
        // замок — both vowels can be stressed
        let marked = apply_stress_marks("замок", &[0, 1]);
        assert!(marked.contains('\u{0301}'));
    }
}
