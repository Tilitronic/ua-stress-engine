//! academic_tests.rs — Tests derived from academic sources.
//!
//! Sources:
//!   [savchenko2014] — Савченко (2014) §19-20: syllabification rules + examples
//!   [steriopolo2012] — Стеріополо (2012): IPA transcription rules + examples
//!   [kasyanova2015]  — Касьянова (2015): /в/ allophone rules + examples
//!
//! Each test references the rule and source that defines the expected behaviour.

#[cfg(test)]
mod tests {
    use crate::pipeline::transcribe;

    // ─────────────────────────────────────────────────────────────────────────
    // HELPERS
    // ─────────────────────────────────────────────────────────────────────────

    fn syllables_of(word: &str, stress: u8) -> Vec<String> {
        transcribe(word, stress)
            .syllables
            .into_iter()
            .map(|s| s.ipa)
            .collect()
    }

    fn syllable_count(word: &str, stress: u8) -> usize {
        transcribe(word, stress).syllables.len()
    }

    fn ipa_of(word: &str, stress: u8) -> String {
        transcribe(word, stress).ipa
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 1: single intervocalic consonant → next syllable
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule1_sadyba() {
        // /садиба/ → /са-ди-ба/ [savchenko2014 rule 1]
        assert_eq!(syllable_count("садиба", 1), 3, "садиба must split into 3 syllables");
    }

    #[test]
    fn rule1_derevo() {
        // /дерево/ → /де-ре-во/ [savchenko2014 rule 1]
        assert_eq!(syllable_count("дерево", 0), 3, "дерево must split into 3 syllables");
    }

    #[test]
    fn rule1_chuzhyna() {
        // /чужина/ → /чу-жи-на/ [savchenko2014 rule 1]
        assert_eq!(syllable_count("чужина", 1), 3, "чужина must split into 3 syllables");
    }

    #[test]
    fn rule1_podarunok() {
        // /подарунок/ → /по-да-ру-нок/ [savchenko2014 rule 1]
        assert_eq!(syllable_count("подарунок", 2), 4, "подарунок must split into 4 syllables");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2a: both voiceless obstruents → both to next syllable
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2a_gusto() {
        // /густо/ → /гу-сто/ — CC=[с,т], both voiceless → onset [savchenko2014]
        let syls = syllables_of("густо", 0);
        assert_eq!(syls.len(), 2, "густо: expected 2 syllables, got {syls:?}");
        // Second syllable must start with both consonants
        assert!(syls[1].starts_with("st"), "густо: 2nd syl should start 'st', got {syls:?}");
    }

    #[test]
    fn rule2a_shvapka() {
        // /шапка/ → /ша-пка/ [savchenko2014 rule 2a]
        let syls = syllables_of("шапка", 0);
        assert_eq!(syls.len(), 2, "шапка: expected 2 syllables, got {syls:?}");
        assert!(syls[1].starts_with("pk"), "шапка: 2nd syl should start 'pk', got {syls:?}");
    }

    #[test]
    fn rule2a_likhtar() {
        // /ліхтар/ → /л'і-хтар/ [savchenko2014 rule 2a]
        let syls = syllables_of("ліхтар", 1);
        assert_eq!(syls.len(), 2, "ліхтар: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn rule2a_krykhta() {
        // /крихта/ → /кри-хта/ [savchenko2014 rule 2a]
        let syls = syllables_of("крихта", 0);
        assert_eq!(syls.len(), 2, "крихта: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2b: both voiced obstruents same manner → both to next
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2b_prydbaты() {
        // /придбати/ → /при-дба-ти/ [savchenko2014 rule 2b]
        let syls = syllables_of("придбати", 1);
        assert_eq!(syls.len(), 3, "придбати: expected 3 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2c: obstruent + sonorant → both to next syllable
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2c_khutro() {
        // /хутро/ → /ху-тро/ [savchenko2014 rule 2c]
        let syls = syllables_of("хутро", 0);
        assert_eq!(syls.len(), 2, "хутро: expected 2 syllables, got {syls:?}");
        assert!(syls[1].starts_with("tr"), "хутро: 2nd syl should start 'tr', got {syls:?}");
    }

    #[test]
    fn rule2c_shablia() {
        // /шабля/ → /ша-бля/ [savchenko2014 rule 2c]
        let syls = syllables_of("шабля", 0);
        assert_eq!(syls.len(), 2, "шабля: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn rule2c_sriblo() {
        // /срібло/ → /ср'і-бло/ [savchenko2014 rule 2c]
        let syls = syllables_of("срібло", 0);
        assert_eq!(syls.len(), 2, "срібло: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2α: both sonorants → split between them
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2alpha_horlo() {
        // /горло/ → /гор-ло/ — CC=[р,л] both sonorants → split [savchenko2014]
        let syls = syllables_of("горло", 0);
        assert_eq!(syls.len(), 2, "горло: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('r'), "горло: 1st syl should end 'r', got {syls:?}");
    }

    #[test]
    fn rule2alpha_hryvnia() {
        // /гривня/ → /грив-ня/ [savchenko2014]
        let syls = syllables_of("гривня", 0);
        assert_eq!(syls.len(), 2, "гривня: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn rule2alpha_perly() {
        // /перли/ → /пер-ли/ [savchenko2014]
        let syls = syllables_of("перли", 0);
        assert_eq!(syls.len(), 2, "перли: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('r'), "перли: 1st syl should end 'r', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2β: sonorant first, any second → sonorant stays in coda
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2beta_stavky() {
        // /ставки/ → /став-ки/ — CC=[в,к]: sonorant + voiceless [savchenko2014]
        let syls = syllables_of("ставки", 1);
        assert_eq!(syls.len(), 2, "ставки: expected 2 syllables, got {syls:?}");
        // First syllable coda has в (realized as w post-vocalic)
        assert!(
            syls[0].ends_with('w') || syls[0].ends_with("ʋ"),
            "ставки: 1st syl should end in w/ʋ, got {syls:?}"
        );
    }

    #[test]
    fn rule2beta_chaika() {
        // /чайка/ → /чай-ка/ — CC=[й,к]: sonorant j + voiceless [savchenko2014]
        let syls = syllables_of("чайка", 0);
        assert_eq!(syls.len(), 2, "чайка: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('j'), "чайка: 1st syl should end 'j', got {syls:?}");
    }

    #[test]
    fn rule2beta_lombard() {
        // /ломбард/ → /лом-бард/ [savchenko2014]
        let syls = syllables_of("ломбард", 1);
        assert_eq!(syls.len(), 2, "ломбард: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('m'), "ломбард: 1st syl should end 'm', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2γ: voiced obstruent + voiceless → split
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2gamma_kazka() {
        // Phonological: /каз-ка/ — voiced з + voiceless к → split
        // Phonetic: з devoices before к → [с,к] both voiceless → both to onset → [ка-ска]
        // [savchenko2014 phonological vs phonetic divergence]
        let syls = syllables_of("казка", 0);
        assert_eq!(syls.len(), 2, "казка: expected 2 syllables, got {syls:?}");
        // Phonetic output: [kɑ][skɑ] — devoiced з→с, both voiceless → onset
        assert!(syls[1].starts_with('s'), "казка: phonetic 2nd syl starts 'sk', got {syls:?}");
    }

    #[test]
    fn rule2gamma_shvedka() {
        // Phonological: /швед-ка/ — voiced д + voiceless к → split
        // Phonetic: д devoices before к → [т,к] both voiceless → onset → [шве-тка]
        // [savchenko2014 phonological vs phonetic divergence]
        let syls = syllables_of("шведка", 0);
        assert_eq!(syls.len(), 2, "шведка: expected 2 syllables, got {syls:?}");
        // Phonetic output: [ʃwɛ][tkɑ] — both voiceless in onset
        assert!(syls[1].starts_with('t'), "шведка: phonetic 2nd syl starts 'tk', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 2δ: voiced fricative + voiced stop/affricate → split
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule2delta_druzhba() {
        // /дружба/ → /друж-ба/ — CC=[ж,б]: voiced fric + voiced stop [savchenko2014]
        let syls = syllables_of("дружба", 0);
        assert_eq!(syls.len(), 2, "дружба: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('ʒ'), "дружба: 1st syl should end 'ʒ', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 3a: first consonant is sonorant → stays in coda
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule3a_linhvist() {
        // /лінгвіст/ → /л'ін-гвіст/ [savchenko2014 rule 3a]
        let syls = syllables_of("лінгвіст", 1);
        assert_eq!(syls.len(), 2, "лінгвіст: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('n') || syls[0].ends_with("nʲ"),
            "лінгвіст: 1st syl should end in 'n', got {syls:?}");
    }

    #[test]
    fn rule3a_portfel() {
        // /портфель/ → /пор-тфель/ [savchenko2014 rule 3a]
        let syls = syllables_of("портфель", 1);
        assert_eq!(syls.len(), 2, "портфель: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('r'), "портфель: 1st syl should end 'r', got {syls:?}");
    }

    #[test]
    fn rule3a_tembrovy() {
        // /тембровий/ → /тем-бро-вий/ [savchenko2014 rule 3a]
        let syls = syllables_of("тембровий", 1);
        assert_eq!(syls.len(), 3, "тембровий: expected 3 syllables, got {syls:?}");
        assert!(syls[0].ends_with('m'), "тембровий: 1st syl should end 'm', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 3b: obstruents + sonorant → all to next syllable
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule3b_pysklia() {
        // /пискля/ → /пи-скля/ [savchenko2014 rule 3b]
        let syls = syllables_of("пискля", 0);
        assert_eq!(syls.len(), 2, "пискля: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn rule3b_postril() {
        // /постріл/ → /по-стріл/ [savchenko2014 rule 3b]
        let syls = syllables_of("постріл", 1);
        assert_eq!(syls.len(), 2, "постріл: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 3c: all voiceless obstruents → all to next syllable
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule3c_khustka() {
        // /хустка/ → /ху-стка/ [savchenko2014 rule 3c]
        let syls = syllables_of("хустка", 0);
        assert_eq!(syls.len(), 2, "хустка: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Rule 3d: voiced + voiceless(es) + sonorant → split after voiced
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rule3d_rozplata() {
        // Phonological: /роз-пла-та/ — voiced з before voiceless п → split after з
        // Phonetic: з devoices before п → [с,п,л] all voiceless+sonorant → rule 3b → onset → [ро-спла-та]
        // [savchenko2014 phonological vs phonetic divergence]
        let syls = syllables_of("розплата", 1);
        assert_eq!(syls.len(), 3, "розплата: expected 3 syllables, got {syls:?}");
        // Phonetic: [rɔ][splɑ][tɑ]
        assert!(syls[1].starts_with('s'), "розплата: phonetic 2nd syl starts 'spl', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Steriopolo 7-rule summary examples
    // [steriopolo2012]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn steriopolo_rule1_podorozhi() {
        // *по-до-ро-жі* [steriopolo2012 rule 1]
        assert_eq!(syllable_count("подорожі", 2), 4, "подорожі should have 4 syllables");
    }

    #[test]
    fn steriopolo_rule4_sumnyi() {
        // *сум-ний* [steriopolo2012 rule 4 — two sonorants split]
        let syls = syllables_of("сумний", 1);
        assert_eq!(syls.len(), 2, "сумний: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('m'), "сумний: 1st syl should end 'm', got {syls:?}");
    }

    #[test]
    fn steriopolo_rule6_haika() {
        // *гай-ка* [steriopolo2012 rule 6 — sonorant+obstruent: sonorant to coda]
        let syls = syllables_of("гайка", 0);
        assert_eq!(syls.len(), 2, "гайка: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('j'), "гайка: 1st syl should end 'j', got {syls:?}");
    }

    #[test]
    fn steriopolo_rule6_iamka() {
        // *ям-ка* [steriopolo2012 rule 6]
        let syls = syllables_of("ямка", 0);
        assert_eq!(syls.len(), 2, "ямка: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('m'), "ямка: 1st syl should end 'm', got {syls:?}");
    }

    #[test]
    fn steriopolo_rule6_synku() {
        // *син-ку* [steriopolo2012 rule 6]
        let syls = syllables_of("синку", 1);
        assert_eq!(syls.len(), 2, "синку: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('n') || syls[0].ends_with("nʲ"),
            "синку: 1st syl should end 'n', got {syls:?}");
    }

    #[test]
    fn steriopolo_rule7_liubliu() {
        // *лю-блю* [steriopolo2012 rule 7 — obstruent+sonorant → both to next]
        let syls = syllables_of("люблю", 1);
        assert_eq!(syls.len(), 2, "люблю: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn steriopolo_rule7_mudryi() {
        // *му-дрий* [steriopolo2012 rule 7]
        let syls = syllables_of("мудрий", 0);
        assert_eq!(syls.len(), 2, "мудрий: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Sonority profile examples from Totska
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn totska_pishla_2_syllables() {
        // пі-шла — profile: 14 — 134 [savchenko2014]
        let syls = syllables_of("пішла", 1);
        assert_eq!(syls.len(), 2, "пішла: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn totska_kazka_split() {
        // каз-ка — phonological profile: 142 — 14 [savchenko2014]
        // Phonetic: з devoiced before к → [ка-ска]
        let syls = syllables_of("казка", 0);
        assert_eq!(syls.len(), 2, "казка: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn totska_kombain_split() {
        // ком-байн — profile: 143 — 2433 [savchenko2014]
        let syls = syllables_of("комбайн", 1);
        assert_eq!(syls.len(), 2, "комбайн: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('m'), "комбайн: 1st syl ends 'm', got {syls:?}");
    }

    #[test]
    fn totska_parta_split() {
        // пар-та — profile: 143 — 14 [savchenko2014]
        let syls = syllables_of("парта", 0);
        assert_eq!(syls.len(), 2, "парта: expected 2 syllables, got {syls:?}");
        assert!(syls[0].ends_with('r'), "парта: 1st syl ends 'r', got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SYLLABIFICATION — Geminates at syllable boundary
    // [savchenko2014]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn geminate_zhyttia() {
        // /життя/ → phonetic [жие-т':а] — geminate attached to following syllable
        // [savchenko2014 geminates]
        let syls = syllables_of("життя", 1);
        assert_eq!(syls.len(), 2, "життя: expected 2 syllables, got {syls:?}");
    }

    #[test]
    fn geminate_statia() {
        // стат-тя [steriopolo2012 rule 5]
        let syls = syllables_of("стаття", 1);
        assert_eq!(syls.len(), 2, "стаття: expected 2 syllables, got {syls:?}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPA — /в/ allophones [kasyanova2015]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn v_allophone_mav_word_final() {
        // *мав* → після голосної, word-final → [ṷ] / u̯ [kasyanova2015]
        let r = transcribe("мав", 0);
        let last_ipa = r.tokens.last().unwrap().ipa.as_str();
        assert!(
            last_ipa == "u̯" || last_ipa == "ṷ" || last_ipa == "w",
            "мав: word-final в should be non-syllabic [u̯/ṷ], got '{last_ipa}'"
        );
    }

    #[test]
    fn v_allophone_pravda_post_vocalic_pre_c() {
        // *правда* → в between vowel and consonant → [w] [kasyanova2015]
        let r = transcribe("правда", 0);
        let v_tok = r.tokens.iter().find(|t| {
            t.ipa == "w" || t.ipa == "ʋ" || t.ipa == "u̯" || t.ipa == "ṷ"
        });
        assert!(v_tok.is_some(), "правда: must contain a /в/ allophone, got {}", r.ipa);
        assert_eq!(v_tok.unwrap().ipa, "w",
            "правда: post-vocalic pre-C в should be [w], got {}", r.ipa);
    }

    #[test]
    fn v_allophone_voda_word_initial() {
        // *вода* → word-initial before vowel → [ʋ] (default) [kasyanova2015]
        let r = transcribe("вода", 1);
        assert_eq!(r.tokens[0].ipa, "ʋ",
            "вода: word-initial в before vowel should be ʋ, got {}", r.tokens[0].ipa);
    }

    #[test]
    fn v_allophone_vin_word_initial_default() {
        // *він* → word-initial → [ʋ] default [kasyanova2015]
        let r = transcribe("він", 0);
        assert_eq!(r.tokens[0].ipa, "ʋ",
            "він: word-initial в should be ʋ, got {}", r.tokens[0].ipa);
    }

    #[test]
    fn v_allophone_krov_word_final() {
        // *кров* → word-final after vowel → u̯ [kasyanova2015]
        let r = transcribe("кров", 0);
        let last = r.tokens.last().unwrap().ipa.as_str();
        assert!(
            last == "u̯" || last == "ṷ" || last == "w",
            "кров: word-final в should be non-syllabic, got '{last}'"
        );
    }

    #[test]
    fn v_allophone_mavpa() {
        // *мавпа* → після голосної, перед приголосним → [w] [kasyanova2015]
        let r = transcribe("мавпа", 0);
        let v_pos = r.tokens.iter().position(|t| {
            t.ipa == "w" || t.ipa == "ʋ" || t.ipa == "u̯" || t.ipa == "ṷ"
        });
        assert!(v_pos.is_some(), "мавпа: must contain /в/ allophone");
        assert_eq!(r.tokens[v_pos.unwrap()].ipa, "w",
            "мавпа: post-vocalic pre-C в should be [w]");
    }

    #[test]
    fn v_allophone_shovk() {
        // *шовк* → після голосної, перед приголосним → [w] [kasyanova2015]
        let r = transcribe("шовк", 0);
        let has_w = r.tokens.iter().any(|t| t.ipa == "w");
        assert!(has_w, "шовк: post-vocalic pre-C в should be [w], ipa={}", r.ipa);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPA — Vowels [steriopolo2012]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn ipa_vowel_a_stressed_is_alpha() {
        // Stressed /а/ → [ɑ] [steriopolo2012]
        let r = transcribe("мама", 0);
        assert!(r.ipa.contains('ɑ'), "мама: stressed а should be ɑ, got {}", r.ipa);
    }

    #[test]
    fn ipa_vowel_o_stressed_is_open_o() {
        // Stressed /о/ → [ɔ] [steriopolo2012]
        let r = transcribe("кот", 0);
        assert!(r.ipa.contains('ɔ'), "кот: stressed о should be ɔ, got {}", r.ipa);
    }

    #[test]
    fn ipa_vowel_e_stressed_is_epsilon() {
        // Stressed /е/ → [ɛ] [steriopolo2012]
        let r = transcribe("день", 0);
        assert!(r.ipa.contains('ɛ'), "день: stressed е should be ɛ, got {}", r.ipa);
    }

    #[test]
    fn ipa_vowel_i_stressed_has_i() {
        // Stressed /і/ → [i] [steriopolo2012]
        let r = transcribe("ніч", 0);
        assert!(r.ipa.contains('i'), "ніч: stressed і should be i, got {}", r.ipa);
    }

    #[test]
    fn ipa_vowel_u_stressed_has_u() {
        // Stressed /у/ → [u] [steriopolo2012]
        let r = transcribe("рух", 0);
        assert!(r.ipa.contains('u'), "рух: stressed у should be u, got {}", r.ipa);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPA — Consonant voicing / devoicing assimilation [steriopolo2012]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn voicing_prosba_z_before_b() {
        // просьба: с before б → [з] (regressive voicing) [steriopolo2012]
        let r = transcribe("просьба", 1);
        assert!(r.ipa.contains('z'),
            "просьба: с before б should voice to з, ipa={}", r.ipa);
    }

    #[test]
    fn voicing_kazky_z_stays() {
        // казки: з before к → devoices to [с]? No — assimilation goes backward:
        // final obstruent before voiceless obstruent → devoiced
        let r = transcribe("казці", 1);
        // з before с (ц) should devoice
        assert!(!r.ipa.contains('z') || r.ipa.starts_with('z'),
            "казці: з before voiceless should devoice, ipa={}", r.ipa);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPA — Palatalization examples [steriopolo2012]
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn palatalization_den_final_soft() {
        // *день* → д-е-н-ь → [nʲ] (ь softens н) [steriopolo2012]
        let r = transcribe("день", 0);
        assert!(r.ipa.contains("nʲ"), "день: н before ь should be nʲ, ipa={}", r.ipa);
    }

    #[test]
    fn palatalization_batko_t_soft() {
        // *батько* → тьк → tʲ [steriopolo2012]
        let r = transcribe("батько", 0);
        assert!(r.ipa.contains("tʲ"), "батько: т before ь should be tʲ, ipa={}", r.ipa);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IPA — Full word IPA checks (from pipeline tests + rules docs)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn full_ipa_mama() {
        // мама, stress 0 → mɑmɑ [steriopolo2012 transcription examples]
        assert_eq!(ipa_of("мама", 0), "mɑmɑ");
    }

    #[test]
    fn full_ipa_den() {
        // день → dɛnʲ [steriopolo2012]
        assert_eq!(ipa_of("день", 0), "dɛnʲ");
    }

    #[test]
    fn full_ipa_batko() {
        // батько → bɑtʲkɔ [steriopolo2012]
        assert_eq!(ipa_of("батько", 0), "bɑtʲkɔ");
    }

    #[test]
    fn full_ipa_sestra() {
        // сестра, stress 1 → sɛstrɑ [steriopolo2012]
        assert_eq!(ipa_of("сестра", 1), "sɛstrɑ");
    }

    #[test]
    fn full_ipa_pravda() {
        // правда → prɑwdɑ [kasyanova2015 + steriopolo2012]
        assert_eq!(ipa_of("правда", 0), "prɑwdɑ");
    }
}
