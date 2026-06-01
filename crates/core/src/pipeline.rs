//! pipeline.rs — Orchestrator for all phonetic passes.
//!
//! Connects the six phonetic transformation passes in the correct order:
//!
//! ```text
//! normalize(word)
//!   → Pass 1:  tokenize()           — Cyrillic → PhoneticToken[]
//!   → Pass 1.5: apply_geminates()   — merge double consonants
//!   → Pass 2:  apply_palatalization() — regressive dental softening
//!   → Pass 3:  apply_voicing_assimilation() — obstruent voicing
//!   → Pass 3b: apply_place_assimilation()   — sibilant + affricate rules
//!   → Pass 4:  apply_vowel_allophones()    — stress mark + vowel variants
//!   → Pass 5:  apply_v_allophones()        — /ʋ/ positional allophones
//!   → Pass 6:  syllabify()                 — segment into syllables
//!   → collect full IPA, return TranscriptionResult
//! ```
//!
//! This module exports one public function: [`transcribe`].

use crate::{
    geminate::apply_geminates,
    normalize::normalize_word,
    palatalization::apply_palatalization,
    place::apply_place_assimilation,
    syllabifier::{ipa_from_syllables, syllabify},
    tokenizer::tokenize,
    types::TranscriptionResult,
    v_allophones::apply_v_allophones,
    voicing::apply_voicing_assimilation,
    vowel_allophones::apply_vowel_allophones,
};

/// Transcribe a single Ukrainian word to IPA.
///
/// # Arguments
/// * `word` — Ukrainian word, any case; apostrophe variants are normalized.
/// * `stress_index` — 0-based index of the stressed vowel (as stored in the dict).
///
/// # Returns
/// A [`TranscriptionResult`] containing the normalized word, all tokens,
/// syllables, and the complete IPA string.
///
/// # Example
/// ```
/// use ua_stress_core::transcribe;
/// let r = transcribe("мама", 0);
/// assert!(r.ipa.contains('ɑ'));
/// assert_eq!(r.syllables.len(), 2);
/// ```
pub fn transcribe(word: &str, stress_index: u8) -> TranscriptionResult {
    // Normalize: lowercase + canonical apostrophe
    let normalized = normalize_word(word);

    // Pass 1: grapheme-to-phoneme tokenization
    let mut tokens = tokenize(&normalized);

    // Pass 1.5: merge adjacent identical consonants into geminates
    apply_geminates(&mut tokens);

    // Pass 2: regressive dental palatalization propagation
    apply_palatalization(&mut tokens);

    // Pass 3: voicing assimilation (obstruent clusters)
    apply_voicing_assimilation(&mut tokens);

    // Pass 3b: sibilant + affricate place assimilation
    // (consumes `tokens` and returns a new Vec — token count may change)
    let tokens = apply_place_assimilation(tokens);
    let mut tokens = tokens;

    // Pass 4: mark stressed vowel + apply vowel allophones
    apply_vowel_allophones(&mut tokens, stress_index);

    // Pass 5: /ʋ/ positional allophones
    apply_v_allophones(&mut tokens);

    // Pass 6: syllabification
    let syllables = syllabify(&tokens, stress_index);

    // Collect full IPA representation
    let ipa = ipa_from_syllables(&syllables);

    TranscriptionResult {
        word: normalized,
        stress_index,
        tokens,
        syllables,
        ipa,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Assert the two reconstruction invariants for every word:
    ///   word_syllables.join("") == normalized_form
    ///   ipa_syllables (stripped of ˈ).join("") == ipa
    fn assert_reconstruction(word: &str, stress: u8) {
        let r = transcribe(word, stress);
        let reconstructed: String = r.syllables.iter().map(|s| s.word.as_str()).collect();
        assert_eq!(
            reconstructed, r.word,
            "{word}@{stress}: word_syllables.join != normalized form"
        );
        let ipa_reconstructed: String = r.syllables.iter().map(|s| s.ipa.as_str()).collect();
        assert_eq!(
            ipa_reconstructed, r.ipa,
            "{word}@{stress}: ipa_syllables.join != flat ipa"
        );
    }

    fn assert_syllable_count(word: &str, stress: u8, expected: usize) {
        let r = transcribe(word, stress);
        assert_eq!(
            r.syllables.len(), expected,
            "{word}: expected {expected} syllables, got {} (ipa={})",
            r.syllables.len(), r.ipa
        );
    }

    fn assert_stressed_syllable(word: &str, stress: u8, syllable_idx: usize) {
        let r = transcribe(word, stress);
        let stressed: Vec<usize> = r.syllables.iter().enumerate()
            .filter(|(_, s)| s.stressed)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(stressed, vec![syllable_idx],
            "{word}@{stress}: expected stressed syllable {syllable_idx}, got {:?}", stressed);
    }

    // ── Reconstruction invariants ─────────────────────────────────────────────

    #[test]
    fn reconstruction_invariants() {
        let cases = [
            ("мама",         0u8),
            ("батько",       0),
            ("університет",  2),
            ("сестра",       1),
            ("правда",       0),
            ("день",         0),
            ("ніч",          0),
            ("хліб",         0),
            ("вода",         1),
            ("земля",        1),
            ("книга",        0),
            ("школа",        0),
            ("народ",        1),
            ("серце",        0),
        ];
        for (word, stress) in cases {
            assert_reconstruction(word, stress);
        }
    }

    // ── Exactly-one-stressed-syllable invariant ───────────────────────────────

    #[test]
    fn exactly_one_stressed_syllable() {
        let cases = ["мама", "батько", "університет", "сестра", "правда"];
        for word in cases {
            let r = transcribe(word, 0);
            if r.syllables.is_empty() { continue; }
            let count = r.syllables.iter().filter(|s| s.stressed).count();
            assert_eq!(count, 1, "{word}: expected 1 stressed syllable, got {count}");
        }
    }

    // ── Syllable counts ───────────────────────────────────────────────────────

    #[test]
    fn syllable_counts() {
        assert_syllable_count("мама",         0, 2);
        assert_syllable_count("батько",       0, 2);
        assert_syllable_count("університет",  2, 5);
        assert_syllable_count("сестра",       1, 2);
        assert_syllable_count("день",         0, 1);
        assert_syllable_count("ніч",          0, 1);
        assert_syllable_count("хліб",         0, 1);
        assert_syllable_count("правда",       0, 2);
        assert_syllable_count("вода",         1, 2);
        assert_syllable_count("дерево",       0, 3);
    }

    // ── Zero-vowel (consonant-only) words ────────────────────────────────────

    #[test]
    fn zero_vowel_word_returns_one_syllable() {
        // Prepositions/particles with no vowel nucleus
        for word in &["з", "в", "й"] {
            let r = transcribe(word, 0);
            assert_eq!(r.syllables.len(), 1,
                "{word}: expected 1 phonological unit, got {}", r.syllables.len());
            // The 'syllable' contains the whole word
            assert!(!r.syllables[0].stressed,
                "{word}: consonant-only syllable should not be marked stressed");
            // Reconstruction still holds
            let reconstructed: String = r.syllables.iter().map(|s| s.word.as_str()).collect();
            assert_eq!(reconstructed, r.word, "{word}: reconstruction failed");
        }
    }

    // ── Exact IPA outputs ────────────────────────────────────────────────────

    #[test]
    fn ipa_mama_stress0() {
        // мама, stress on first vowel: both а → ɑ (stable vowel, no allophony)
        let r = transcribe("мама", 0);
        assert_eq!(r.ipa, "mɑmɑ");
        assert_eq!(r.syllables[0].ipa, "mɑ");
        assert_eq!(r.syllables[1].ipa, "mɑ");
        assert!(r.syllables[0].stressed);
    }

    #[test]
    fn ipa_den_stress0() {
        // день: д(d) е(ɛ) нь → nʲ (ь softens н)
        let r = transcribe("день", 0);
        assert_eq!(r.ipa, "dɛnʲ");
        assert_eq!(r.syllables.len(), 1);
        assert!(r.syllables[0].stressed);
        assert!(!r.syllables[0].is_open, "CVC(soft) should be closed");
    }

    #[test]
    fn ipa_batko_stress0() {
        // батько: б(b) а(ɑ) т(t) ь(soft) к(k) о(ɔ)
        // ь softens т → tʲ; then tʲk goes to onset of 2nd syllable
        let r = transcribe("батько", 0);
        assert_eq!(r.ipa, "bɑtʲkɔ");
        assert_eq!(r.syllables.len(), 2);
        assert!(r.syllables[0].stressed);
        // syllabifier assigns tʲk to the onset of syllable 2
        assert_eq!(r.syllables[0].ipa, "bɑ");
        assert_eq!(r.syllables[1].ipa, "tʲkɔ");
    }

    #[test]
    fn ipa_sestRA_stress1() {
        // сестра: stress on second vowel (а)
        // с(s) е(ɛ) с(s) т(t) р(r) а(ɑ)
        // inter-vocalic cluster [s,t,r] — all go to onset → syllables: [sɛ][strɑ]
        let r = transcribe("сестра", 1);
        assert_eq!(r.ipa, "sɛstrɑ");
        assert_eq!(r.syllables.len(), 2);
        assert_eq!(r.syllables[0].ipa, "sɛ");
        assert_eq!(r.syllables[1].ipa, "strɑ");
        assert!(!r.syllables[0].stressed);
        assert!(r.syllables[1].stressed);
    }

    #[test]
    fn ipa_apostrophe_myach() {
        // м'яч: м ʔ j a t͡ʃ (apostrophe → glottal, я → j + ɑ)
        let r = transcribe("м'яч", 0);
        assert!(r.tokens.iter().any(|t| t.ipa == "ʔ"), "expected glottal ʔ in м'яч");
        assert!(r.tokens.iter().any(|t| t.ipa == "j"),  "expected glide j in м'яч");
        assert_eq!(r.syllables.len(), 1);
    }

    #[test]
    fn ipa_shch_is_two_tokens() {
        // щука: щ → ʃ + t͡ʃ (two tokens), у к а
        let r = transcribe("щука", 0);
        assert_eq!(r.tokens[0].ipa, "ʃ");
        assert_eq!(r.tokens[1].ipa, "t͡ʃ");
        assert_eq!(r.syllables.len(), 2);
    }

    #[test]
    fn ipa_pravda_v_allophone() {
        // правда: в between vowel and consonant → w
        // п(p) р(r) а(ɑ, stressed) в(w) д(d) а(ɑ)
        // cluster [w, d]: sonorant w + obstruent d → w in coda → syllables: [prɑw][dɑ]
        let r = transcribe("правда", 0);
        assert!(r.tokens.iter().any(|t| t.ipa == "w"),
            "expected в→w in правда, got: {:?}", r.ipa);
        assert_eq!(r.ipa, "prɑwdɑ");
        assert_eq!(r.syllables.len(), 2);
        assert_eq!(r.syllables[0].ipa, "prɑw");
        assert_eq!(r.syllables[1].ipa, "dɑ");
    }

    #[test]
    fn ipa_voda_v_allophone_initial() {
        // вода: в word-initial before vowel → ʋ (approximant)
        // ʋ(ʋ) о(ɔ) д(d) а(ɑ, stressed at index 1)
        let r = transcribe("вода", 1);
        assert_eq!(r.tokens[0].ipa, "ʋ",
            "initial в before vowel should be ʋ, got: {}", r.tokens[0].ipa);
        assert_eq!(r.syllables.len(), 2);
        assert_eq!(r.syllables[1].ipa, "dɑ");
        assert!(r.syllables[1].stressed);
    }

    #[test]
    fn ipa_iotated_ya_word_initial() {
        // яма: initial я → j + ɑ
        let r = transcribe("яма", 0);
        assert_eq!(r.tokens[0].ipa, "j", "initial я should start with j");
        assert_eq!(r.syllables.len(), 2);
    }

    #[test]
    fn ipa_voicing_assimilation() {
        // просьба: с before б should voice → з
        // (regressive voicing assimilation: obstruent before voiced obstruent)
        let r = transcribe("просьба", 1);
        assert!(r.ipa.contains('z'),
            "expected з (voiced s) before б in просьба, ipa={}", r.ipa);
    }

    #[test]
    fn ipa_university_5_syllables() {
        // університет, stress on 3rd vowel (е in вер), stress_index=2
        let r = transcribe("університет", 2);
        assert_eq!(r.syllables.len(), 5, "університет should have 5 syllables");
        assert_eq!(r.syllables[2].ipa.trim_start_matches('\u{02c8}'), "ʋɛr",
            "3rd syllable ipa should be ʋɛr, got {}", r.syllables[2].ipa);
        assert!(r.syllables[2].stressed, "3rd syllable (вер) should be stressed");
        // word_syllables reconstruction
        let reconstructed: String = r.syllables.iter().map(|s| s.word.as_str()).collect();
        assert_eq!(reconstructed, "університет");
    }

    #[test]
    fn ipa_syllables_stress_prefix() {
        // Verify ˈ prefix is on the stressed syllable's IPA
        let r = transcribe("мама", 0);
        // syllables[0] is stressed
        assert!(r.syllables[0].stressed);
        // The ipa_syllables array built by dict.rs would prefix ˈ on stressed syllable
        // Here we verify the raw syllable ipa does NOT have ˈ (dict adds it)
        assert!(!r.syllables[0].ipa.starts_with('\u{02c8}'),
            "raw syllable ipa should not have ˈ — dict adds it");
    }

    #[test]
    fn word_syllables_alignment() {
        // For any word, word_syllables[i] and ipa_syllables[i] represent the same syllable.
        // The length of word_syllables and the number of syllables must match.
        let words = [("мама", 0), ("батько", 0), ("сестра", 1), ("правда", 0)];
        for (word, stress) in words {
            let r = transcribe(word, stress);
            let word_syls: Vec<&str> = r.syllables.iter().map(|s| s.word.as_str()).collect();
            let ipa_syls: Vec<&str>  = r.syllables.iter().map(|s| s.ipa.as_str()).collect();
            assert_eq!(word_syls.len(), ipa_syls.len(),
                "{word}: word_syllables and ipa_syllables must have same length");
        }
    }

    // ── Legacy tests (kept) ───────────────────────────────────────────────────

    #[test]
    fn basic_word() {
        let r = transcribe("мама", 0);
        assert_eq!(r.syllables.len(), 2);
        assert!(r.syllables[0].stressed);
        assert!(!r.syllables[1].stressed);
        assert!(r.ipa.contains('ɑ'));
    }

    #[test]
    fn word_with_apostrophe() {
        let r = transcribe("м'яч", 0);
        assert!(r.tokens.iter().any(|t| t.ipa == "ʔ"), "Glottal expected");
        assert!(r.tokens.iter().any(|t| t.ipa == "j"),  "Glide j expected");
    }

    #[test]
    fn soft_sign_word() {
        let r = transcribe("день", 0);
        assert!(r.tokens.iter().any(|t| t.ipa == "nʲ"), "nʲ expected");
    }

    #[test]
    fn digraph_dzh() {
        let r = transcribe("джміль", 0);
        assert_eq!(r.tokens[0].ipa, "d͡ʒ");
    }

    #[test]
    fn digraph_dz_denkit() {
        let r = transcribe("дзенькіт", 0);
        assert_eq!(r.tokens[0].ipa, "d͡z");
    }

    #[test]
    fn dzh_prefix_boundary_splits() {
        let r = transcribe("підживити", 1);
        let ipas: Vec<&str> = r.tokens.iter().map(|t| t.ipa.as_str()).collect();
        let d_idx = ipas.iter().position(|&x| x == "d").expect("must contain d");
        assert_eq!(ipas[d_idx + 1], "ʒ", "prefix boundary д+ж must stay split: {ipas:?}");
        assert!(!ipas.contains(&"d͡ʒ"), "must not merge to d͡ʒ at prefix boundary: {ipas:?}");
    }

    #[test]
    fn dz_prefix_boundary_splits() {
        let r = transcribe("надзвичайний", 2);
        let ipas: Vec<&str> = r.tokens.iter().map(|t| t.ipa.as_str()).collect();
        let d_idx = ipas.iter().position(|&x| x == "d").expect("must contain d");
        assert_eq!(ipas[d_idx + 1], "z", "prefix boundary д+з must stay split: {ipas:?}");
        assert!(!ipas.contains(&"d͡z"), "must not merge to d͡z at prefix boundary: {ipas:?}");
    }

    #[test]
    fn composite_shch() {
        let r = transcribe("щука", 0);
        assert_eq!(r.tokens[0].ipa, "ʃ");
        assert_eq!(r.tokens[1].ipa, "t͡ʃ");
    }

    #[test]
    fn iotated_initial() {
        let r = transcribe("яма", 0);
        assert_eq!(r.tokens[0].ipa, "j");
    }

    #[test]
    fn normalizes_uppercase() {
        let r = transcribe("Мама", 0);
        assert_eq!(r.word, "мама");
    }

    #[test]
    fn stress_marked() {
        let r = transcribe("батько", 0);
        let stressed_count = r.tokens.iter().filter(|t| t.stressed).count();
        assert_eq!(stressed_count, 1, "Exactly one stressed vowel");
    }

    #[test]
    fn v_allophone_post_vocalic() {
        let r = transcribe("правда", 0);
        assert!(r.tokens.iter().any(|t| t.ipa == "w"), "w expected in правда: {:?}", r.ipa);
    }
}
