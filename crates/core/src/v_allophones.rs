//! v_allophones.rs — Pass 5: Positional allophones of Ukrainian /в/ (ʋ).
//!
//! The phoneme /в/ (Ukrainian в) is traditionally classified as a **sonorant**
//! (labial approximant /ʋ/), meaning it does NOT trigger voicing assimilation.
//! However, its surface realization depends on the phonological context:
//!
//! | Context                        | Allophone | Example              |
//! |-------------------------------|-----------|----------------------|
//! | After vowel, before consonant | w         | правда [ˈprɑwdɑ]     |
//! | Word-final after vowel         | u̯ (u̯)   | кров [krɔu̯]          |
//! | Elsewhere (default)            | ʋ         | він [ʋin], вчора [ʋtʃɔrɑ] |
//!
//! These allophones are post-vocalic: the preceding token must be a **vowel**.
//!
//! References:
//!   Тоцька 1997 §12: в/ф → у̯/w у певних позиціях.
//!   Мойсієнко 2010 §8.2: /в/ as bilabial approximant, positional variants.

use crate::data::{V_DEFAULT, V_POST_VOCALIC_PRE_C, V_WORD_FINAL};
use crate::types::{PhoneticToken, TokenType};

/// Check whether the token at `idx - 1` is a vowel (if `idx > 0`).
fn prev_is_vowel(tokens: &[PhoneticToken], idx: usize) -> bool {
    idx > 0 && tokens[idx - 1].token_type == TokenType::Vowel
}

/// Check whether the token at `idx + 1` is a consonant (if in bounds).
fn next_is_consonant(tokens: &[PhoneticToken], idx: usize) -> bool {
    tokens
        .get(idx + 1)
        .map(|t| t.token_type == TokenType::Consonant)
        .unwrap_or(false)
}

/// Check whether this is the last token OR followed only by non-consonants.
fn is_word_final_or_before_non_c(tokens: &[PhoneticToken], idx: usize) -> bool {
    // "Word final" means: no following consonant token at all.
    // (i.e., idx is the last token, or all remaining are vowels/glides)
    tokens[idx + 1..]
        .iter()
        .all(|t| t.token_type != TokenType::Consonant)
}

/// Apply positional allophony to all /ʋ/ tokens in the word.
///
/// Mutates `tokens` in place (only `ipa` and `token_type` may change).
pub fn apply_v_allophones(tokens: &mut Vec<PhoneticToken>) {
    for i in 0..tokens.len() {
        if tokens[i].ipa != V_DEFAULT {
            continue; // not a /ʋ/ token
        }

        if prev_is_vowel(tokens, i) {
            if next_is_consonant(tokens, i) {
                // Post-vocalic, pre-consonantal → w
                tokens[i].ipa = V_POST_VOCALIC_PRE_C.to_string();
                // Keep token_type = Consonant; w is a labial-velar approximant
            } else if i + 1 >= tokens.len() || is_word_final_or_before_non_c(tokens, i) {
                // Post-vocalic, word-final → u̯
                tokens[i].ipa = V_WORD_FINAL.to_string();
                tokens[i].token_type = TokenType::Glide;
            }
        }
        // Elsewhere: stays ʋ (default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn v_ipa_in(word: &str) -> Vec<String> {
        let mut tokens = tokenize(word);
        apply_v_allophones(&mut tokens);
        tokens.into_iter().map(|t| t.ipa).collect()
    }

    #[test]
    fn default_v_word_initial() {
        // він → ʋin (в is word-initial, not post-vocalic)
        let ipas = v_ipa_in("він");
        assert_eq!(ipas[0], "ʋ", "Word-initial в stays ʋ: {ipas:?}");
    }

    #[test]
    fn v_becomes_w_post_vocalic_pre_consonant() {
        // правда → prɑwdɑ (в is between ɑ and d)
        let ipas = v_ipa_in("правда");
        assert!(
            ipas.contains(&"w".to_string()),
            "Post-vocalic в before consonant → w: {ipas:?}"
        );
    }

    #[test]
    fn v_becomes_u_glide_word_final_after_vowel() {
        // кров → krɔu̯ (в is word-final after vowel)
        let ipas = v_ipa_in("кров");
        let v_ipa = ipas.last().unwrap();
        assert_eq!(v_ipa, V_WORD_FINAL, "Word-final в after vowel → u̯: {ipas:?}");
    }

    #[test]
    fn default_before_vowel() {
        // вчора — в is word-initial before consonant → stays ʋ
        let ipas = v_ipa_in("вчора");
        assert_eq!(ipas[0], "ʋ", "Word-initial в → ʋ: {ipas:?}");
    }
}
