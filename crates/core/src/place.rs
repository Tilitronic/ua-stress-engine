//! place.rs — Pass 3b: Place-of-articulation assimilation.
//!
//! Two types of place assimilation operate in Ukrainian:
//!
//! ## Phase 1 — Sibilant assimilation (no token removal)
//! A sibilant (з/с/зь/сь) **before** a postalveolar (ш/ж/ч/дж) changes
//! its place to match the following consonant, but both tokens survive.
//! The rule is strictly regressive and affects only the preceding sibilant.
//!
//! ## Phase 2 — Affricate contraction (second token removed)
//! A stop/plosive + fricative sequence is contracted to an affricate.
//! The first token is replaced with the affricate; the second is removed.
//!
//! Examples:
//!   зшити  → [ʃʃɪtɪ]      (з→ʃ before ш)
//!   братство → [brɑtstvɔ] → [brɑtsvɔ] (тс→ts)
//!
//! This pass returns a **new** Vec (affricate contraction changes token count).

use crate::data::{AFFRICATE_RULES, SIBILANT_RULES};
use crate::types::PhoneticToken;

/// Strip ː (length diacritic) from an IPA string for rule matching.
fn strip_length(ipa: &str) -> &str {
    ipa.strip_suffix('ː').unwrap_or(ipa)
}

/// Apply place-of-articulation assimilation.
///
/// Phase 1 (sibilant) mutates tokens in-place.
/// Phase 2 (affricate) rebuilds the Vec (changes token count).
pub fn apply_place_assimilation(tokens: Vec<PhoneticToken>) -> Vec<PhoneticToken> {
    let mut tokens = apply_sibilant_assimilation(tokens);
    tokens = apply_affricate_contraction(tokens);
    tokens
}

// ── Phase 1: Sibilant assimilation ──────────────────────────────────────────

fn apply_sibilant_assimilation(mut tokens: Vec<PhoneticToken>) -> Vec<PhoneticToken> {
    let n = tokens.len();
    if n < 2 {
        return tokens;
    }

    // Collect index → new_ipa changes to avoid borrow conflicts
    let mut changes: Vec<(usize, String)> = Vec::new();

    for i in 0..n - 1 {
        let cur_base = strip_length(&tokens[i].ipa).to_string();
        let nxt_base = strip_length(&tokens[i + 1].ipa).to_string();

        for rule in SIBILANT_RULES {
            if rule.target == cur_base && rule.trigger == nxt_base {
                changes.push((i, rule.result.to_string()));
                break; // first matching rule wins
            }
        }
    }

    for (idx, new_ipa) in changes {
        tokens[idx].ipa = new_ipa;
    }

    tokens
}

// ── Phase 2: Affricate contraction ──────────────────────────────────────────

fn apply_affricate_contraction(tokens: Vec<PhoneticToken>) -> Vec<PhoneticToken> {
    if tokens.len() < 2 {
        return tokens;
    }

    let mut result: Vec<PhoneticToken> = Vec::with_capacity(tokens.len());
    let mut i = 0;

    while i < tokens.len() {
        if i + 1 < tokens.len() {
            let cur_base = strip_length(&tokens[i].ipa).to_string();
            let nxt_base = strip_length(&tokens[i + 1].ipa).to_string();

            // Find matching affricate rule
            let rule_match = AFFRICATE_RULES
                .iter()
                .find(|r| r.first == cur_base && r.second == nxt_base);

            if let Some(rule) = rule_match {
                // Replace first token with the affricate; skip second token
                let mut merged = tokens[i].clone();
                merged.ipa = rule.result.to_string();
                merged.source = format!("{}{}", tokens[i].source, tokens[i + 1].source);
                result.push(merged);
                i += 2; // skip the consumed second token
                continue;
            }
        }

        result.push(tokens[i].clone());
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn pipeline_ipas(word: &str) -> Vec<String> {
        let tokens = tokenize(word);
        let tokens = apply_place_assimilation(tokens);
        tokens.into_iter().map(|t| t.ipa).collect()
    }

    #[test]
    fn sibilant_z_before_sh() {
        // зшити → ʃʃɪtɪ (з becomes ʃ before ш)
        let ipas = pipeline_ipas("зшити");
        // Tokens: z ʃ ɪ t ɪ → first z becomes ʃ
        assert!(
            ipas.iter().filter(|s| s.as_str() == "ʃ").count() >= 2,
            "Expected two ʃ tokens (assimilation): {ipas:?}"
        );
    }

    #[test]
    fn affricate_t_plus_s() {
        // братство: б р а т с т в о
        // After tokenizer: b r ɑ t s t ʋ ɔ
        // Affricate rule: t + s → ts
        let ipas = pipeline_ipas("братство");
        assert!(ipas.contains(&"ts".to_string()), "t+s should contract to ts: {ipas:?}");
        // Ensure the 's' that was consumed is gone
        let ts_idx = ipas.iter().position(|s| s == "ts").unwrap();
        if ts_idx + 1 < ipas.len() {
            assert_ne!(ipas[ts_idx + 1], "s", "s should have been consumed: {ipas:?}");
        }
    }

    #[test]
    fn no_spurious_contractions() {
        // "час" has tʃ + a — the affricate comes from щ/ч, not from t+s
        let ipas = pipeline_ipas("час");
        assert!(ipas.contains(&"tʃ".to_string()));
        assert!(!ipas.contains(&"tsː".to_string()));
    }
}
