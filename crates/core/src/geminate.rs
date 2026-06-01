//! geminate.rs — Pass 1.5: Merge adjacent identical consonants into geminates.
//!
//! Ukrainian has phonemically long (geminate) consonants, typically at
//! morpheme boundaries.  When the tokenizer produces two consecutive tokens
//! with the **same base IPA**, this pass merges them into a single token
//! with the length diacritic `ː`.
//!
//! Examples (Савченко 2014, Тоцька 1997):
//!   життя  → ʒɪtːɑ   (тт → tː)
//!   зілля  → zʲilʲːɑ  (лл → lʲː, both soft → palatalized geminate)
//!   ніччю  → nʲitʃːu
//!   волосся → ʋɔlɔsːɑ
//!
//! Rules:
//!   1. Two adjacent consonant/glide tokens with the same **base** IPA
//!      (base = IPA with ʲ stripped) are merged.
//!   2. The merged token inherits the `source` of both, concatenated.
//!   3. If either token is palatalized, the merged token is palatalized
//!      (soft variant prevails: `lʲː` not `lː`).
//!   4. The merged IPA is `base + (ʲ if soft) + ː`.
//!
//! This pass runs AFTER tokenization but BEFORE palatalization propagation.

use crate::types::{PhoneticToken, TokenType};

/// Strip the palatalization diacritic (ʲ) from an IPA string to get the base.
/// Examples: "nʲ" → "n", "t͡sʲ" → "t͡s", "d͡ʒ" → "d͡ʒ"
fn base_ipa(ipa: &str) -> &str {
    ipa.strip_suffix('ʲ').unwrap_or(ipa)
}

/// Merge adjacent identical consonant tokens into geminate (long) tokens.
///
/// This is an in-place operation that may reduce the token count.
pub fn apply_geminates(tokens: &mut Vec<PhoneticToken>) {
    if tokens.len() < 2 {
        return;
    }

    let mut write = 0; // write cursor for the compacted output
    let mut read = 0;  // read cursor scanning through the original indices

    // We build the output in-place by swap-compacting matched pairs.
    // Using a temporary vec avoids borrow issues.
    let mut result: Vec<PhoneticToken> = Vec::with_capacity(tokens.len());

    while read < tokens.len() {
        let current = &tokens[read];

        // Only merge consonant-type or glide tokens
        if current.token_type != TokenType::Consonant && current.token_type != TokenType::Glide {
            result.push(tokens[read].clone());
            read += 1;
            continue;
        }

        // Look ahead: do the next token have the same base IPA?
        if let Some(next) = tokens.get(read + 1) {
            if (next.token_type == TokenType::Consonant || next.token_type == TokenType::Glide)
                && base_ipa(&current.ipa) == base_ipa(&next.ipa)
            {
                // Merge: the soft variant prevails
                let use_soft = current.palatalized || next.palatalized;
                let base = base_ipa(&current.ipa);
                let merged_ipa = if use_soft {
                    format!("{}ʲː", base)
                } else {
                    format!("{}ː", base)
                };

                let mut merged = current.clone();
                merged.ipa = merged_ipa;
                merged.source = format!("{}{}", current.source, next.source);
                merged.palatalized = use_soft;

                // Inherit features: prefer the palatalized variant's features
                if use_soft && next.palatalized {
                    if let Some(f) = &next.consonant_features {
                        merged.consonant_features = Some(f.clone());
                    }
                } else if use_soft {
                    if let Some(f) = &mut merged.consonant_features {
                        f.softness = crate::types::Softness::Soft;
                    }
                }

                result.push(merged);
                read += 2; // skip both input tokens
                continue;
            }
        }

        // No merge — copy as-is
        result.push(tokens[read].clone());
        read += 1;
        write += 1;
    }
    let _ = write; // suppress unused warning

    *tokens = result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    fn ipa_list(word: &str) -> Vec<String> {
        let mut tokens = tokenize(word);
        apply_geminates(&mut tokens);
        tokens.into_iter().map(|t| t.ipa).collect()
    }

    #[test]
    fn geminate_tt() {
        // життя: ʒ ɪ t tʲ ɑ  → ʒ ɪ tʲː ɑ
        // (soft т from я palatalizes and merges with the hard т)
        let ipas = ipa_list("життя");
        // After tokenizer: ʒ, ɪ, t, tʲ (from я palatalization), ɑ
        // After geminate: ʒ, ɪ, tʲː, ɑ
        assert!(ipas.iter().any(|s| s.contains('ː')), "Expected a geminate in {ipas:?}");
    }

    #[test]
    fn geminate_ll_soft() {
        // зілля → z ʲ i l lʲ ɑ → z ʲ i lʲː ɑ
        let ipas = ipa_list("зілля");
        assert!(ipas.iter().any(|s| s == "lʲː"), "Expected lʲː in {ipas:?}");
    }

    #[test]
    fn no_merge_different_consonants() {
        // брат → no geminates
        let ipas = ipa_list("брат");
        assert!(!ipas.iter().any(|s| s.contains('ː')));
    }
}
