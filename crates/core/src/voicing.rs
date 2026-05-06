//! voicing.rs — Pass 3: Regressive obstruent voicing assimilation.
//!
//! Ukrainian obstruent clusters assimilate voicing regressively:
//!   - A voiceless obstruent becomes **voiced** before a voiced obstruent.
//!   - A voiced obstruent becomes **voiceless** before a voiceless obstruent.
//!
//! Critical constraints (Тоцька 1997):
//!   - Sonorants (м, н, л, р, й, /ʋ/) do NOT trigger assimilation and
//!     are NOT affected by it — they break the assimilation domain.
//!   - Ukrainian has **no word-final devoicing** (unlike Russian/German).
//!   - Only within-cluster (within-consonant-sequence) assimilation.
//!
//! Examples:
//!   просьба  → prɔzʲbɑ    (sʲ → zʲ before b)
//!   вокзал   → ʋɔɡzɑl     (k  → ɡ  before z)
//!   боротьба → bɔrɔdʲbɑ   (tʲ → dʲ before b)
//!   легко    → lɛxkɔ      (ɦ  → x  before k — devoicing)
//!
//! This pass mutates `tokens` **in place**.

use crate::data::{VOICED_OBSTRUENTS, VOICELESS_OBSTRUENTS, VOICING_MAP, SONORANT_EXEMPT};
use crate::types::{PhoneticToken, TokenType, VoicePower};
use crate::data::CONSONANT_MAP;

/// Apply regressive voicing assimilation across obstruent clusters.
///
/// Right-to-left scan: for each obstruent, the next consonant to the
/// right determines whether assimilation applies.
pub fn apply_voicing_assimilation(tokens: &mut Vec<PhoneticToken>) {
    // We need right-to-left access but will borrow two positions at once,
    // so we collect the changes into a side buffer first.
    let n = tokens.len();
    if n < 2 {
        return;
    }

    // Collect (index → new_ipa) mutations before applying them.
    let mut changes: Vec<(usize, String)> = Vec::new();

    for i in (0..n - 1).rev() {
        let current = &tokens[i];
        let next = &tokens[i + 1];

        // Both must be consonant-like
        if current.token_type != TokenType::Consonant {
            continue;
        }
        if next.token_type != TokenType::Consonant && next.token_type != TokenType::Glide {
            continue;
        }

        let cur_base = strip_length(current.ipa.as_str());
        let nxt_base = strip_length(next.ipa.as_str());

        // Current must be an obstruent
        if !is_obstruent(cur_base) {
            continue;
        }
        // Next must also be an obstruent (sonorants do not trigger assimilation)
        if SONORANT_EXEMPT.contains(nxt_base) || !is_obstruent(nxt_base) {
            continue;
        }

        let is_geminate = current.ipa.ends_with('ː');
        let suffix = if is_geminate { "ː" } else { "" };

        if is_voiced_obstruent(cur_base) && is_voiceless_obstruent(nxt_base) {
            // Voiced current → voiceless assimilation (devoicing)
            if let Some(&pair) = VOICING_MAP.get(cur_base) {
                if is_voiceless_obstruent(pair) {
                    changes.push((i, format!("{}{}", pair, suffix)));
                }
            }
        } else if is_voiceless_obstruent(cur_base) && is_voiced_obstruent(nxt_base) {
            // Voiceless current → voiced assimilation
            if let Some(&pair) = VOICING_MAP.get(cur_base) {
                if is_voiced_obstruent(pair) {
                    changes.push((i, format!("{}{}", pair, suffix)));
                }
            }
        }
    }

    // Apply collected mutations
    for (idx, new_ipa) in changes {
        let base = strip_length(&new_ipa).to_string();
        tokens[idx].ipa = new_ipa;
        tokens[idx].palatalized = tokens[idx].ipa.contains('ʲ');
        // Update features from the new IPA
        if let Some(def) = CONSONANT_MAP.get(base.as_str()) {
            tokens[idx].consonant_features = Some(def.to_features());
            if tokens[idx].palatalized {
                if let Some(f) = &mut tokens[idx].consonant_features {
                    f.softness = crate::types::Softness::Soft;
                    f.voice_power = VoicePower::Voiced; // or Voiceless — already set by def
                }
            }
        } else if let Some(f) = &mut tokens[idx].consonant_features {
            // Infer voicing from the new IPA set
            if VOICED_OBSTRUENTS.contains(base.as_str()) {
                f.voice_power = VoicePower::Voiced;
            } else if VOICELESS_OBSTRUENTS.contains(base.as_str()) {
                f.voice_power = VoicePower::Voiceless;
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn strip_length(ipa: &str) -> &str {
    ipa.strip_suffix('ː').unwrap_or(ipa)
}

fn is_obstruent(base: &str) -> bool {
    VOICED_OBSTRUENTS.contains(base) || VOICELESS_OBSTRUENTS.contains(base)
}

fn is_voiced_obstruent(base: &str) -> bool {
    VOICED_OBSTRUENTS.contains(base)
}

fn is_voiceless_obstruent(base: &str) -> bool {
    VOICELESS_OBSTRUENTS.contains(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn pipeline_ipas(word: &str) -> Vec<String> {
        let mut tokens = tokenize(word);
        apply_voicing_assimilation(&mut tokens);
        tokens.into_iter().map(|t| t.ipa).collect()
    }

    #[test]
    fn devoicing_before_voiceless() {
        // легко → lɛxkɔ  (ɦ → x before k)
        let ipas = pipeline_ipas("легко");
        // Tokens: l ɛ ɦ k ɔ → ɦ before k → ɦ → x
        let g_idx = ipas.iter().position(|s| s == "ɦ" || s == "x").unwrap();
        assert_eq!(ipas[g_idx], "x", "ɦ should become x before k: {ipas:?}");
    }

    #[test]
    fn voicing_before_voiced() {
        // просьба → prɔzʲbɑ (sʲ → zʲ before b)
        let ipas = pipeline_ipas("просьба");
        // After tokenizer: p r ɔ sʲ b ɑ
        // Voicing: sʲ before b → zʲ
        assert!(ipas.contains(&"zʲ".to_string()), "sʲ should become zʲ before b: {ipas:?}");
    }

    #[test]
    fn sonorant_blocks_assimilation() {
        // "слово" — sonorants l, r don't trigger assimilation
        let ipas = pipeline_ipas("слово");
        // No assimilation expected — no adjacent obstruent clusters
        assert!(ipas.contains(&"s".to_string()));
        assert!(ipas.contains(&"l".to_string()));
    }

    #[test]
    fn no_word_final_devoicing() {
        // "клад" — final д stays voiced (Ukrainian has no word-final devoicing)
        let ipas = pipeline_ipas("клад");
        assert_eq!(ipas.last().unwrap(), "d", "No word-final devoicing: {ipas:?}");
    }
}
