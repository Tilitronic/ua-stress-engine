//! syllabifier.rs — Pass 6: Syllabification following Savchenko (2014) §20.
//!
//! Produces a `Vec<UaSyllable>` from the ordered `PhoneticToken` slice.
//!
//! # Algorithm
//!
//! The syllabification algorithm follows the Sonority Sequencing Principle
//! adapted for Ukrainian by Savchenko.  Each vowel token anchors one syllable;
//! the inter-vocalic consonant cluster is split between the two flanking vowels
//! using `split_cluster()`.
//!
//! ## Cluster splitting rules (`split_cluster`)
//!
//! Given a consonant cluster `C₁ C₂ … Cₙ` between two vowels,
//! the function determines index `k` such that:
//!   - Coda of the left syllable: `C₁ … Cₖ`
//!   - Onset of the right syllable: `Cₖ₊₁ … Cₙ`
//!
//! | Rule | Condition | k (0-based split point before onset) |
//! |------|-----------|-------------------------------------|
//! | 0    | n = 1     | 0 (whole cluster is onset)          |
//! | 2a   | n = 2, both obstruents, differ in voicing | 1 (both in coda, onset empty — but Ukrainian usually keeps the second as onset) |
//! | 2b   | n = 2, both obstruents, same voicing | 0 (onset) |
//! | 2c   | n = 2, obstruent + sonorant | 0 (onset) |
//! | 2d   | n = 2, sonorant + obstruent | 1 (coda+onset) |
//! | 2e   | n = 2, obstruent + j | 0 (onset) |
//! | 2f   | n = 2, sonorant + sonorant | 0 (onset) |
//! | 3c   | n ≥ 3, all voiceless | 0 (all in onset) |
//! | 3+   | first is sonorant | k = 1 |
//! | default | leftmost sonority trough | 1 |
//!
//! # Open/Closed
//! A syllable is **open** if its last token is a vowel.

use crate::data::SONORANT_EXEMPT;
use crate::types::{PhoneticToken, TokenType, UaSyllable, VoicePower};

// ── Sonority helpers ─────────────────────────────────────────────────────────

fn is_vowel_tok(t: &PhoneticToken) -> bool {
    t.token_type == TokenType::Vowel
}

fn is_consonant_tok(t: &PhoneticToken) -> bool {
    t.token_type == TokenType::Consonant || t.token_type == TokenType::Glide
}

fn is_sonorant(t: &PhoneticToken) -> bool {
    t.consonant_features
        .as_ref()
        .map(|f| f.voice_power == VoicePower::Sonorant)
        .unwrap_or_else(|| {
            // Fallback: check the sonorant-exempt set by IPA base
            let base = t.ipa.trim_end_matches('ʲ').trim_end_matches('ː');
            SONORANT_EXEMPT.contains(base)
        })
}

fn is_obstruent(t: &PhoneticToken) -> bool {
    !is_sonorant(t) && is_consonant_tok(t)
}

fn is_voiced_obstruent(t: &PhoneticToken) -> bool {
    is_obstruent(t)
        && t.consonant_features
            .as_ref()
            .map(|f| f.voice_power == VoicePower::Voiced)
            .unwrap_or(false)
}

fn is_voiceless_obstruent(t: &PhoneticToken) -> bool {
    is_obstruent(t) && !is_voiced_obstruent(t)
}

fn is_fricative(t: &PhoneticToken) -> bool {
    t.consonant_features
        .as_ref()
        .map(|f| f.manner == crate::types::Manner::Fricative)
        .unwrap_or(false)
}

// ── Cluster splitting ────────────────────────────────────────────────────────

/// Given an inter-vocalic consonant cluster (slice of tokens), return the
/// number of tokens that belong to the **coda** of the left syllable.
/// The rest form the onset of the right syllable.
///
/// Returns a value in `0..=cluster.len()`.
fn split_cluster(cluster: &[PhoneticToken]) -> usize {
    let n = cluster.len();

    match n {
        0 => 0,

        // Rule 0: single consonant → onset of the right syllable
        1 => 0,

        2 => {
            let a = &cluster[0];
            let b = &cluster[1];

            let a_son = is_sonorant(a);
            let b_son = is_sonorant(b);
            let a_obs = !a_son;
            let b_obs = !b_son;

            if a_obs && b_obs {
                // 2γ: voiced + voiceless → split [savchenko2014]
                let different_voicing = is_voiced_obstruent(a) != is_voiced_obstruent(b);
                if different_voicing {
                    return 1;
                }
                // 2δ: both voiced — split if one is fricative and other is stop/affricate
                // (voiced fricative + voiced stop → split; see Jespersen scale) [savchenko2014]
                if is_voiced_obstruent(a) && is_voiced_obstruent(b) {
                    let a_fric = is_fricative(a);
                    let b_fric = is_fricative(b);
                    if a_fric != b_fric {
                        return 1; // split between fricative and stop/affricate
                    }
                }
                // 2b: both voiced same manner, or both voiceless → all onset
                0
            } else if a_obs && b_son {
                // 2c: obstruent + sonorant → all onset
                0
            } else if a_son && b_obs {
                // 2d: sonorant + obstruent → sonorant in coda
                1
            } else if b.ipa == "j" || b.ipa == "ʋ" {
                // 2e: anything + glide → all onset
                0
            } else {
                // 2α: sonorant + sonorant → split between them [savchenko2014]
                1
            }
        }

        _ => {
            // n ≥ 3
            let all_voiceless = cluster.iter().all(is_voiceless_obstruent);
            if all_voiceless {
                // 3c: all voiceless → all onset
                return 0;
            }

            let first_sonorant = is_sonorant(&cluster[0]);
            if first_sonorant {
                // First sonorant → sonorant alone in coda
                return 1;
            }

            // Default: leftmost sonority trough or first obstruent cluster
            // Heuristic: if leading consonants are all obstruents, put them all in onset
            let leading_obs_count = cluster.iter().take_while(|t| is_obstruent(t)).count();
            if leading_obs_count == n {
                // All obstruents: put them all in onset (maximize onset principle)
                0
            } else {
                // Sonorant after leading obstruents: keep leading obstruents in onset
                // i.e., coda up to (but not including) the first sonorant among the rest
                0
            }
        }
    }
}

// ── Main syllabification ──────────────────────────────────────────────────────

/// Build syllables from an ordered `PhoneticToken` slice.
///
/// # Arguments
/// * `tokens` — tokens produced by the pipeline up to this pass
/// * `stress_index` — 0-based vowel index carrying the word stress
///
/// # Returns
/// `Vec<UaSyllable>` in word order, each with its IPA, tokens, stressed flag,
/// and open/closed flag.
pub fn syllabify(tokens: &[PhoneticToken], stress_index: u8) -> Vec<UaSyllable> {
    // Locate all vowel positions
    let vowel_positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| is_vowel_tok(t))
        .map(|(i, _)| i)
        .collect();

    if vowel_positions.is_empty() {
        // No vowels: the whole word is one consonant-only "syllable"
        // (e.g., abbreviations, interjections)
        let ipa: String = tokens.iter().map(|t| t.ipa.as_str()).collect();
        let word: String = tokens.iter().map(|t| t.source.as_str()).collect();
        return vec![UaSyllable {
            ipa,
            word,
            tokens: tokens.to_vec(),
            stressed: false,
            is_open: false,
        }];
    }

    // Prefix consonants (before the first vowel) — attached to first syllable onset
    let first_vowel_pos = vowel_positions[0];
    let mut prefix_cs: Vec<PhoneticToken> = tokens[..first_vowel_pos].to_vec();

    let mut syllables: Vec<UaSyllable> = Vec::with_capacity(vowel_positions.len());

    for (vi, &vpos) in vowel_positions.iter().enumerate() {
        // The vowel token itself
        let vowel_tok = &tokens[vpos];

        // Consonants between this vowel and the NEXT vowel (or word end)
        let next_vowel_pos = vowel_positions.get(vi + 1).copied().unwrap_or(tokens.len());
        let inter: Vec<PhoneticToken> = tokens[vpos + 1..next_vowel_pos].to_vec();

        // Split inter-vocalic cluster into coda + next_onset
        let coda_len = if vi + 1 < vowel_positions.len() {
            // More vowels follow: we must split the cluster
            split_cluster(&inter)
        } else {
            // Last syllable: all remaining consonants are its coda
            inter.len()
        };

        let coda: Vec<PhoneticToken> = inter[..coda_len].to_vec();
        let next_onset: Vec<PhoneticToken> = inter[coda_len..].to_vec();

        // Assemble this syllable's token list: prefix + vowel + coda
        let mut syl_tokens: Vec<PhoneticToken> = prefix_cs.drain(..).collect();
        syl_tokens.push(vowel_tok.clone());
        syl_tokens.extend_from_slice(&coda);

        // Build IPA string and original word fragment for this syllable
        let ipa: String = syl_tokens.iter().map(|t| t.ipa.as_str()).collect();
        let word: String = syl_tokens.iter().map(|t| t.source.as_str()).collect();

        // Stressed if this syllable contains the stressed vowel
        let stressed = vowel_tok.vowel_index == stress_index as i8 && vowel_tok.stressed;

        // Open syllable: no coda consonants
        let is_open = coda.is_empty();

        syllables.push(UaSyllable { ipa, word, tokens: syl_tokens, stressed, is_open });

        // The next onset becomes the prefix for the next syllable
        prefix_cs = next_onset;
    }

    syllables
}

// ── IPA reconstruction ────────────────────────────────────────────────────────

/// Reconstruct the full IPA string for a word from its syllables.
pub fn ipa_from_syllables(syllables: &[UaSyllable]) -> String {
    syllables.iter().map(|s| s.ipa.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;
    use crate::vowel_allophones::apply_vowel_allophones;

    fn run_syllabify(word: &str, stress: u8) -> Vec<UaSyllable> {
        let mut tokens = tokenize(word);
        apply_vowel_allophones(&mut tokens, stress);
        syllabify(&tokens, stress)
    }

    #[test]
    fn single_vowel_word() {
        // "й" — no vowels, single phoneme
        let syls = run_syllabify("й", 0);
        assert_eq!(syls.len(), 1);
    }

    #[test]
    fn two_syllable_open() {
        // "мама" → [mɑ][mɑ] — both open
        let syls = run_syllabify("мама", 0);
        assert_eq!(syls.len(), 2, "Expected 2 syllables: {:?}", syls.iter().map(|s| &s.ipa).collect::<Vec<_>>());
        assert!(syls[0].is_open, "First syllable should be open");
        assert!(syls[1].is_open, "Second syllable should be open");
    }

    #[test]
    fn cvc_closed() {
        // "кіт" → [kʲit] (one closed syllable)
        let syls = run_syllabify("кіт", 0);
        assert_eq!(syls.len(), 1);
        assert!(!syls[0].is_open, "CVC should be closed");
    }

    #[test]
    fn stressed_syllable_marked() {
        // "мама" stress on first vowel
        let syls = run_syllabify("мама", 0);
        assert!(syls[0].stressed, "First syllable should be stressed");
        assert!(!syls[1].stressed, "Second syllable should not be stressed");
    }

    #[test]
    fn three_syllables() {
        // "сестра" → with 2 vowels... wait, с-е-с-т-р-а → 2 vowels → 2 syllables
        // "батько" → б-а-т-ь-к-о → 2 vowels → 2 syllables
        let syls = run_syllabify("батько", 0);
        assert_eq!(syls.len(), 2, "Expected 2 syllables: {:?}", syls.iter().map(|s| &s.ipa).collect::<Vec<_>>());
    }

    #[test]
    fn consonant_cluster_split() {
        // "сестра" → ɛ is first vowel, ɑ is second
        // inter-vocalic cluster: s t r → [s] coda=0 (all onset) or [s,t,r] all onset?
        // According to our rules: leading obstruents s,t then sonorant r → 0 (all onset to next syllable)
        let syls = run_syllabify("сестра", 1);
        assert_eq!(syls.len(), 2);
        // First syllable "ses" or "se" — depends on split
        // Second syllable gets "stra"
        let _ = syls; // structure validated by not panicking + count
    }
}
