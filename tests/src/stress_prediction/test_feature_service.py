"""Tests for feature_service — hashing helpers and the 97-feature builder.

Every test is pure (no DB, no I/O) and deterministic.
"""

import sys, os, json
import pytest

# Make the services package importable directly
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "src", "stress_prediction", "lightgbm",
    ),
)

from services.feature_service import (
    djb2_hash,
    find_vowels,
    syllable_onset_pattern,
    last_syllable_open,
    max_consonant_cluster,
    count_open_syllables,
    cv_shape,
    vowel_context_hash,
    vowel_wide_context_hash,
    syllable_weight,
    detect_compound_interfix,
    count_prefix_matches,
    parse_morph_features,
    build_features_v13,
)
from services.constants import (
    VOWEL_SET, EXPECTED_FEATURE_COUNT, PATTERN_INT, POS_INT,
)


# ════════════════════════════════════════════════════════════════════
# djb2_hash
# ════════════════════════════════════════════════════════════════════
class TestDjb2Hash:
    """djb2 must be deterministic, bounded, and handle edge cases."""

    def test_deterministic(self):
        assert djb2_hash("привіт", 1024) == djb2_hash("привіт", 1024)

    def test_different_strings_differ(self):
        assert djb2_hash("кіт", 1024) != djb2_hash("кот", 1024)

    def test_mod_bound(self):
        for mod in [1, 7, 256, 512, 1024, 2048]:
            val = djb2_hash("тестування", mod)
            assert 0 <= val < mod, f"hash={val} not in [0,{mod})"

    def test_empty_string(self):
        h = djb2_hash("", 512)
        assert 0 <= h < 512

    def test_single_char(self):
        h = djb2_hash("а", 256)
        assert 0 <= h < 256


# ════════════════════════════════════════════════════════════════════
# find_vowels
# ════════════════════════════════════════════════════════════════════
class TestFindVowels:

    def test_basic_word(self):
        assert find_vowels("мама") == [1, 3]

    def test_all_consonants(self):
        assert find_vowels("стрт") == []

    def test_all_vowels(self):
        # "аеі" — 3 vowels at 0,1,2
        assert find_vowels("аеі") == [0, 1, 2]

    def test_ukrainain_vowels_complete(self):
        """All 10 Ukrainian vowels are recognised."""
        for ch in "аеєиіїоуюя":
            assert find_vowels(ch) == [0], f"'{ch}' not found"

    def test_consonants_not_vowels(self):
        for ch in "бвгґджзклмнпрстфхцчшщ":
            assert find_vowels(ch) == [], f"'{ch}' wrongly found"

    def test_сонцесяйний(self):
        word = "сонцесяйний"
        vows = find_vowels(word)
        # с0 о1 н2 ц3 е4 с5 я6 й7 н8 и9 й10
        assert vows == [1, 4, 6, 9]

    def test_word_with_soft_sign(self):
        # ь is not a vowel
        assert find_vowels("тінь") == [1]


# ════════════════════════════════════════════════════════════════════
# syllable_onset_pattern
# ════════════════════════════════════════════════════════════════════
class TestSyllableOnsetPattern:

    def test_vowel_initial(self):
        # "арка" — first char is vowel → pattern "V" → 0
        assert syllable_onset_pattern("арка") == PATTERN_INT["V"]

    def test_cv_initial(self):
        # "мама" — м+а → "CV" → 1
        assert syllable_onset_pattern("мама") == PATTERN_INT["CV"]

    def test_ccv_initial(self):
        # "стіл" — ст+і → "CCV" → 2
        assert syllable_onset_pattern("стіл") == PATTERN_INT["CCV"]

    def test_no_vowels(self):
        assert syllable_onset_pattern("стр") == PATTERN_INT["no_vowel"]


# ════════════════════════════════════════════════════════════════════
# last_syllable_open
# ════════════════════════════════════════════════════════════════════
class TestLastSyllableOpen:

    def test_open(self):
        # "мама" — ends with vowel 'а'
        assert last_syllable_open("мама", [1, 3]) == 1

    def test_closed(self):
        # "мамин" — last vowel is 'и' at idx 3, word ends at idx 4
        assert last_syllable_open("мамин", [1, 3]) == 0

    def test_no_vowels(self):
        assert last_syllable_open("стр", []) == 0


# ════════════════════════════════════════════════════════════════════
# max_consonant_cluster
# ════════════════════════════════════════════════════════════════════
class TestMaxConsonantCluster:

    def test_basic(self):
        # "стрічка" — "стр" is a cluster of 3
        assert max_consonant_cluster("стрічка") == 3

    def test_no_consonants(self):
        assert max_consonant_cluster("аеі") == 0

    def test_all_consonants(self):
        assert max_consonant_cluster("бвгд") == 4


# ════════════════════════════════════════════════════════════════════
# count_open_syllables
# ════════════════════════════════════════════════════════════════════
class TestCountOpenSyllables:

    def test_mama(self):
        # "мама" (ма-ма) — both open
        assert count_open_syllables("мама", [1, 3]) == 2

    def test_closed_syllable(self):
        # "кіт" — one syllable, closed (ends with т)
        assert count_open_syllables("кіт", [1]) == 0


# ════════════════════════════════════════════════════════════════════
# cv_shape
# ════════════════════════════════════════════════════════════════════
class TestCvShape:

    def test_mama(self):
        assert cv_shape("мама") == "CVCV"

    def test_with_soft_sign(self):
        # ь is not alpha in Ukrainian? Actually it is. Let's check:
        # ь.isalpha() is True, and it's not a vowel → C
        assert cv_shape("тінь") == "CVCC"  # т→C, і→V, н→C, ь→C

    def test_with_apostrophe(self):
        # apostrophe is not alpha — should be skipped
        assert cv_shape("м'яч") == "CVC"  # м→C, ' skipped, я→V, ч→C


# ════════════════════════════════════════════════════════════════════
# syllable_weight
# ════════════════════════════════════════════════════════════════════
class TestSyllableWeight:

    def test_open_syllable_weight_zero(self):
        # "mama" as pure vowel ending: "аа" — vowels [0,1], syl 0 vowel at 0,
        # next vowel at 1, no consonants between → weight 0
        assert syllable_weight("аа", [0, 1], 0) == 0

    def test_light_syllable_weight_one(self):
        # "мама" vowels [1,3], syl 0: vowel at 1, chars 2='м' → 1 consonant → weight 1
        assert syllable_weight("мама", [1, 3], 0) == 1

    def test_closed_syllable_weight(self):
        # "кантор" к0 а1 н2 т3 о4 р5 — vowels [1,4]
        # syl 0: vowel at 1, next vowel at 4, coda = chars 2,3 = н,т → 2 consonants → weight 2
        assert syllable_weight("кантор", [1, 4], 0) == 2

    def test_single_coda(self):
        # "канка" к0 а1 н2 к3 а4 — vowels [1,4]
        # syl 0: vowel at 1, next vowel at 4, coda chars 2,3 = н,к → 2 → weight 2
        assert syllable_weight("канка", [1, 4], 0) == 2

    def test_out_of_range(self):
        assert syllable_weight("мама", [1, 3], 5) == 0


# ════════════════════════════════════════════════════════════════════
# detect_compound_interfix
# ════════════════════════════════════════════════════════════════════
class TestDetectCompoundInterfix:

    def test_compound_word(self):
        # "водовоз" — в0 о1 д2 о3 в4 о5 з6 — 'о' at idx 3
        # before: д (consonant), after: в (consonant), ratio 3/7≈0.43 — in [0.2, 0.7]
        has, pct = detect_compound_interfix("водовоз", find_vowels("водовоз"))
        assert has == 1
        assert 20 < pct < 70

    def test_short_word_no_interfix(self):
        has, pct = detect_compound_interfix("кіт", find_vowels("кіт"))
        assert has == 0
        assert pct == 0


# ════════════════════════════════════════════════════════════════════
# count_prefix_matches
# ════════════════════════════════════════════════════════════════════
class TestCountPrefixMatches:

    def test_bez_prefix(self):
        count, max_len = count_prefix_matches("безнебість")
        assert count >= 1
        assert max_len >= 3  # "без" has length 3

    def test_no_prefix(self):
        count, max_len = count_prefix_matches("кіт")
        assert count == 0
        assert max_len == 0

    def test_multiple_prefixes(self):
        # "перед" matches both "перед" (5) and "пере" (4)
        count, max_len = count_prefix_matches("перед")
        assert count >= 2
        assert max_len == 5


# ════════════════════════════════════════════════════════════════════
# parse_morph_features
# ════════════════════════════════════════════════════════════════════
class TestParseMorphFeatures:

    def test_valid_json(self):
        j = json.dumps({"Case": "Nom", "Gender": "Masc", "Number": "Sing"})
        result = parse_morph_features(j)
        assert result["case_int"] == 0
        assert result["gender_int"] == 0
        assert result["number_int"] == 0
        assert result["n_morph_features"] == 3

    def test_none_input(self):
        result = parse_morph_features(None)
        assert result["case_int"] == -1
        assert result["n_morph_features"] == 0

    def test_empty_string(self):
        result = parse_morph_features("")
        assert result["case_int"] == -1

    def test_invalid_json(self):
        result = parse_morph_features("{not valid}")
        assert result["case_int"] == -1

    def test_all_features(self):
        j = json.dumps({
            "Case": "Gen", "Gender": "Fem", "Number": "Plur",
            "Tense": "Past", "Aspect": "Perf", "Degree": "Sup",
        })
        result = parse_morph_features(j)
        assert result["case_int"] == 1
        assert result["gender_int"] == 1
        assert result["number_int"] == 1
        assert result["tense_int"] == 0
        assert result["aspect_int"] == 1
        assert result["degree_int"] == 2
        assert result["n_morph_features"] == 6


# ════════════════════════════════════════════════════════════════════
# build_features_v13  — THE KEY TEST
# ════════════════════════════════════════════════════════════════════
class TestBuildFeaturesV13:

    def test_returns_97_features(self):
        """The feature dict must have exactly 97 keys."""
        feats = build_features_v13("привіт", "NOUN")
        assert len(feats) == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feats)}: "
            f"extra={set(feats) - set(feats)} missing keys count mismatch"
        )

    def test_feature_count_with_morphology(self):
        morph = json.dumps({"Case": "Nom", "Gender": "Masc"})
        feats = build_features_v13("привіт", "NOUN", morph)
        assert len(feats) == EXPECTED_FEATURE_COUNT

    def test_all_values_numeric(self):
        feats = build_features_v13("тестування", "NOUN")
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"Feature '{k}' has type {type(v)}"

    def test_word_len_correct(self):
        feats = build_features_v13("кіт", "NOUN")
        assert feats["word_len"] == 3

    def test_vowel_count_correct(self):
        feats = build_features_v13("мама", "NOUN")
        assert feats["vowel_count"] == 2

    def test_pos_int_known(self):
        for pos_tag, expected_int in POS_INT.items():
            feats = build_features_v13("слово", pos_tag)
            assert feats["pos_int"] == expected_int

    def test_pos_int_unknown(self):
        feats = build_features_v13("щось", "UNKNOWN_POS")
        assert feats["pos_int"] == len(POS_INT)

    def test_deterministic(self):
        """Same input → identical output."""
        f1 = build_features_v13("перевірка", "NOUN")
        f2 = build_features_v13("перевірка", "NOUN")
        assert f1 == f2

    def test_apostrophe_detection(self):
        """Test that apostrophe variants are detected."""
        for apo_char in ["'", "\u02bc", "\u2019"]:
            word = f"м{apo_char}яч"
            feats = build_features_v13(word, "NOUN")
            assert feats["has_apostrophe"] == 1, (
                f"Apostrophe '{apo_char}' (U+{ord(apo_char):04X}) not detected"
            )

    def test_no_apostrophe(self):
        feats = build_features_v13("мама", "NOUN")
        assert feats["has_apostrophe"] == 0

    def test_is_infinitive(self):
        feats = build_features_v13("бігати", "VERB")
        assert feats["is_infinitive"] == 1

    def test_is_not_infinitive_for_noun(self):
        feats = build_features_v13("бігати", "NOUN")
        assert feats["is_infinitive"] == 0

    def test_is_reflexive(self):
        feats = build_features_v13("сміятися", "VERB")
        assert feats["is_reflexive"] == 1

    def test_is_bisyllable(self):
        feats = build_features_v13("мама", "NOUN")
        assert feats["is_bisyllable"] == 1
        feats2 = build_features_v13("тестування", "NOUN")
        assert feats2["is_bisyllable"] == 0

    def test_has_compound_interfix(self):
        feats = build_features_v13("водовоз", "NOUN")
        assert feats["has_compound_interfix"] == 1

    def test_suffix_hash_deterministic(self):
        f1 = build_features_v13("перевірка", "NOUN")
        f2 = build_features_v13("перевірка", "NOUN")
        assert f1["suffix_hash_3"] == f2["suffix_hash_3"]
        assert f1["suffix_hash_4"] == f2["suffix_hash_4"]

    def test_vowel_char_features(self):
        """vowel_char_0..4 should be VOWEL_CHAR_MAP indices."""
        feats = build_features_v13("аеіоу", "X")
        # vowels at 0,1,2,3,4 → а=0, е=1, і=4, о=6, у=7
        from services.constants import VOWEL_CHAR_MAP
        assert feats["vowel_char_0"] == VOWEL_CHAR_MAP["а"]
        assert feats["vowel_char_1"] == VOWEL_CHAR_MAP["е"]
        assert feats["vowel_char_2"] == VOWEL_CHAR_MAP["і"]
        assert feats["vowel_char_3"] == VOWEL_CHAR_MAP["о"]
        assert feats["vowel_char_4"] == VOWEL_CHAR_MAP["у"]

    def test_morph_features_populated(self):
        morph = json.dumps({"Case": "Gen", "Tense": "Past"})
        feats = build_features_v13("слова", "NOUN", morph)
        assert feats["morph_case"] == 1     # Gen
        assert feats["morph_tense"] == 0    # Past
        assert feats["morph_n_features"] == 2

    # ── Accent paradigm features (new) ──────────────────────────────
    def test_oxytone_mobile_suffix_ар(self):
        """'школяр' ends with 'яр' → paradigm B suffix → flag=1."""
        feats = build_features_v13("школяр", "NOUN")
        assert feats["has_oxytone_mobile_suffix"] == 1

    def test_oxytone_mobile_suffix_ач(self):
        feats = build_features_v13("ткач", "NOUN")
        assert feats["has_oxytone_mobile_suffix"] == 1

    def test_oxytone_mobile_suffix_absent(self):
        feats = build_features_v13("мама", "NOUN")
        assert feats["has_oxytone_mobile_suffix"] == 0

    def test_penult_stable_suffix_ість(self):
        """'радість' ends with 'ість' → stable penult stress → flag=1."""
        feats = build_features_v13("радість", "NOUN")
        assert feats["has_penult_stable_suffix"] == 1

    def test_penult_stable_suffix_ість2(self):
        """'злість' ends with 'ість' (2 syllables) → flag=1."""
        feats = build_features_v13("злість", "NOUN")
        assert feats["has_penult_stable_suffix"] == 1

    def test_penult_stable_suffix_absent(self):
        feats = build_features_v13("кіт", "NOUN")
        assert feats["has_penult_stable_suffix"] == 0

    def test_adcat_suffix_цький(self):
        """'козацький' ends with 'цький' + ADJ → paradigm A → flag=1."""
        feats = build_features_v13("козацький", "ADJ")
        assert feats["has_adcat_suffix"] == 1

    def test_adcat_suffix_ський(self):
        """'міський' ends with 'ський' + ADJ → paradigm A → flag=1."""
        feats = build_features_v13("міський", "ADJ")
        assert feats["has_adcat_suffix"] == 1

    def test_adcat_suffix_wrong_pos(self):
        """Same ending on NOUN → flag=0 (only ADJ counts)."""
        feats = build_features_v13("козацький", "NOUN")
        assert feats["has_adcat_suffix"] == 0

    def test_adcat_suffix_absent(self):
        feats = build_features_v13("мама", "ADJ")
        assert feats["has_adcat_suffix"] == 0

    def test_single_vowel_word(self):
        """Single-vowel word shouldn't crash."""
        feats = build_features_v13("кіт", "NOUN")
        assert feats["vowel_count"] == 1
        assert feats["vowel_span"] == 0.0
        assert feats["penult_ratio"] == -1.0

    def test_empty_word(self):
        """Edge case: empty string shouldn't crash."""
        feats = build_features_v13("", "X")
        assert feats["word_len"] == 0
        assert feats["vowel_count"] == 0
        assert len(feats) == EXPECTED_FEATURE_COUNT

    def test_known_handcrafted_words(self):
        """Smoke test: features build without error for real handcrafted words."""
        words = [
            ("сонцесяйний", "ADJ"),
            ("ніжносолов'їний", "ADJ"),
            ("льодопил", "NOUN"),
            ("мертвопетлювати", "VERB"),
            ("недоновонародження", "NOUN"),
            ("білоодежна", "ADJ"),
        ]
        for word, pos in words:
            feats = build_features_v13(word, pos)
            assert len(feats) == EXPECTED_FEATURE_COUNT, f"Failed on {word}"
            assert feats["word_len"] == len(word), f"word_len wrong for {word}"
