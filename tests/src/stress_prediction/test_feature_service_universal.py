"""Tests for feature_service_universal — 130-feature universal builder.

Every test is pure (no DB, no I/O) and deterministic.
Validates the 33 universal features on top of the 97-base set.
"""

import sys
import os

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "src", "stress_prediction", "lightGbm",
    ),
)

from services.feature_service_universal import (
    build_features_universal,
    EXPECTED_FEATURE_COUNT_UNIV,
)
from services.feature_service import build_features_v13
from services.constants import VOWEL_SET


# ════════════════════════════════════════════════════════════════════
# Feature count
# ════════════════════════════════════════════════════════════════════

class TestFeatureCount:

    def test_expected_count_constant(self):
        assert EXPECTED_FEATURE_COUNT_UNIV == 132

    def test_count_2syl_word(self):
        feat = build_features_universal("мама", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV

    def test_count_3syl_word(self):
        feat = build_features_universal("молоко", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV

    def test_count_4syl_word(self):
        feat = build_features_universal("документи", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV

    def test_count_5syl_word(self):
        feat = build_features_universal("університет", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV

    def test_count_monosyllable(self):
        """Monosyllable: features should still be 130, just with fallback values."""
        feat = build_features_universal("кіт", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV

    def test_no_vowels(self):
        """Consonant-only: should not crash, returns 130 features."""
        feat = build_features_universal("стрнг", "X")
        assert len(feat) == EXPECTED_FEATURE_COUNT_UNIV


# ════════════════════════════════════════════════════════════════════
# Base feature preservation
# ════════════════════════════════════════════════════════════════════

class TestBaseFeaturePreservation:
    """Universal features must include all 97 base features unchanged."""

    def test_base_keys_present(self):
        base = build_features_v13("слово", "NOUN")
        univ = build_features_universal("слово", "NOUN")
        for key in base:
            assert key in univ, f"Base key '{key}' missing from universal"

    def test_base_values_identical(self):
        base = build_features_v13("вечірня", "ADJ")
        univ = build_features_universal("вечірня", "ADJ")
        for key, val in base.items():
            assert univ[key] == val, f"Value mismatch for '{key}': {val} vs {univ[key]}"


# ════════════════════════════════════════════════════════════════════
# Syllable count features
# ════════════════════════════════════════════════════════════════════

class TestSyllableCountFeatures:

    def test_syllable_count_2syl(self):
        feat = build_features_universal("мама", "NOUN")
        assert feat["syllable_count_u"] == 2

    def test_syllable_count_3syl(self):
        feat = build_features_universal("молоко", "NOUN")
        assert feat["syllable_count_u"] == 3

    def test_syllable_count_4syl(self):
        feat = build_features_universal("документи", "NOUN")
        assert feat["syllable_count_u"] == 4

    def test_syllable_count_norm_bounded(self):
        feat = build_features_universal("університет", "NOUN")
        assert 0 <= feat["syllable_count_norm_u"] <= 1


# ════════════════════════════════════════════════════════════════════
# Ending hash features
# ════════════════════════════════════════════════════════════════════

class TestEndingHashFeatures:

    def test_hash_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        for key in ["ending_hash_4_u", "ending_hash_5_u", "ending_hash_6_u",
                     "ending_pos_hash_3_u", "ending_pos_hash_4_u"]:
            assert key in feat, f"Missing key: {key}"

    def test_hashes_bounded(self):
        feat = build_features_universal("документи", "NOUN")
        for key in ["ending_hash_4_u", "ending_hash_5_u", "ending_hash_6_u",
                     "ending_pos_hash_3_u", "ending_pos_hash_4_u"]:
            assert 0 <= feat[key] < 4096, f"{key}={feat[key]} out of range"

    def test_different_endings_differ(self):
        f1 = build_features_universal("молоко", "NOUN")
        f2 = build_features_universal("молоки", "NOUN")
        # Last 4 chars differ → ending_hash_4 should differ
        assert f1["ending_hash_4_u"] != f2["ending_hash_4_u"]

    def test_pos_compound_hash_differs_by_pos(self):
        f_noun = build_features_universal("вечір", "NOUN")
        f_verb = build_features_universal("вечір", "VERB")
        # POS-compound hashes should differ for same word, different POS
        assert (f_noun["ending_pos_hash_3_u"] != f_verb["ending_pos_hash_3_u"]
                or f_noun["ending_pos_hash_4_u"] != f_verb["ending_pos_hash_4_u"])


# ════════════════════════════════════════════════════════════════════
# Distance-from-end features
# ════════════════════════════════════════════════════════════════════

class TestDistFromEnd:

    def test_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        for key in ["dist_from_end_v_last_u", "dist_from_end_v_pen_u",
                     "dist_from_end_v_ante_u",
                     "dist_from_end_v_last_norm_u", "dist_from_end_v_pen_norm_u",
                     "dist_from_end_v_ante_norm_u"]:
            assert key in feat, f"Missing key: {key}"

    def test_last_vowel_distance(self):
        feat = build_features_universal("молоко", "NOUN")
        # "молоко" vowels at [1,3,5], wl=6. dist_last = 6-5-1 = 0
        assert feat["dist_from_end_v_last_u"] == 0

    def test_penultimate_distance(self):
        feat = build_features_universal("молоко", "NOUN")
        # dist_pen = 6-3-1 = 2
        assert feat["dist_from_end_v_pen_u"] == 2

    def test_antepenultimate_distance(self):
        feat = build_features_universal("молоко", "NOUN")
        # dist_ante = 6-1-1 = 4
        assert feat["dist_from_end_v_ante_u"] == 4

    def test_2syl_word_ante_is_zero(self):
        """2-syllable word has no antepenultimate vowel → 0."""
        feat = build_features_universal("мама", "NOUN")
        assert feat["dist_from_end_v_ante_u"] == -1

    def test_normalised_bounded(self):
        feat = build_features_universal("університет", "NOUN")
        for key in ["dist_from_end_v_last_norm_u", "dist_from_end_v_pen_norm_u",
                     "dist_from_end_v_ante_norm_u"]:
            assert -1 <= feat[key] <= 1, f"{key}={feat[key]}"


# ════════════════════════════════════════════════════════════════════
# Coda / vowel identity / inter-vowel features
# ════════════════════════════════════════════════════════════════════

class TestCodaAndVowelFeatures:

    def test_coda_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        for key in ["coda_last_u", "coda_pen_u", "coda_ante_u"]:
            assert key in feat

    def test_vowel_identity_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        for key in ["v_last_char_u", "v_pen_char_u", "v_ante_char_u"]:
            assert key in feat

    def test_interval_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        for key in ["iv_last_gap_u", "iv_pen_gap_u"]:
            assert key in feat

    def test_coda_nonnegative(self):
        feat = build_features_universal("документи", "NOUN")
        for key in ["coda_last_u", "coda_pen_u", "coda_ante_u"]:
            assert feat[key] >= 0

    def test_interval_nonnegative(self):
        feat = build_features_universal("документи", "NOUN")
        for key in ["iv_last_gap_u", "iv_pen_gap_u"]:
            assert feat[key] >= 0


# ════════════════════════════════════════════════════════════════════
# Linguistic rule flags
# ════════════════════════════════════════════════════════════════════

class TestLinguisticFlags:

    def test_all_flag_keys_present(self):
        feat = build_features_universal("молоко", "NOUN")
        flags = [
            "has_vy_prefix_u", "has_gerund_suffix_u",
            "has_verb_aty_u", "has_verb_yty_u", "has_verb_uvaty_u",
            "has_oxytone_mobile_u", "has_adcat_suffix_u",
            "has_num_adtsyat_u", "has_foreign_suffix_u",
            "has_greek_suffix_u", "has_penult_stable_u",
        ]
        for key in flags:
            assert key in feat, f"Missing flag: {key}"

    def test_flags_are_binary(self):
        for word in ["молоко", "виходити", "написання", "п'ятдесят"]:
            feat = build_features_universal(word, "NOUN")
            flags = [v for k, v in feat.items() if k.startswith("has_")]
            for val in flags:
                assert val in (0, 1, 0.0, 1.0), f"Non-binary flag: {val}"

    def test_vy_prefix_detected(self):
        feat = build_features_universal("виходити", "VERB")
        assert feat["has_vy_prefix_u"] == 1

    def test_vy_prefix_not_on_nonvy(self):
        feat = build_features_universal("молоко", "NOUN")
        assert feat["has_vy_prefix_u"] == 0

    def test_gerund_suffix_detected(self):
        feat = build_features_universal("читання", "NOUN")
        assert feat["has_gerund_suffix_u"] == 1

    def test_verb_aty_detected(self):
        feat = build_features_universal("читати", "VERB")
        assert feat["has_verb_aty_u"] == 1

    def test_verb_yty_detected(self):
        feat = build_features_universal("ходити", "VERB")
        assert feat["has_verb_yty_u"] == 1


# ════════════════════════════════════════════════════════════════════
# Determinism
# ════════════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_input_same_output(self):
        f1 = build_features_universal("молоко", "NOUN")
        f2 = build_features_universal("молоко", "NOUN")
        assert f1 == f2

    def test_different_words_differ(self):
        f1 = build_features_universal("молоко", "NOUN")
        f2 = build_features_universal("документи", "NOUN")
        # At minimum the syllable count should differ
        assert f1["syllable_count_u"] != f2["syllable_count_u"]


# ════════════════════════════════════════════════════════════════════
# Feature naming convention
# ════════════════════════════════════════════════════════════════════

class TestNamingConvention:
    """Universal-specific features should end with '_u' suffix."""

    def test_universal_features_have_suffix(self):
        base = set(build_features_v13("слово", "NOUN").keys())
        univ = set(build_features_universal("слово", "NOUN").keys())
        new_keys = univ - base
        assert len(new_keys) == 32, f"Expected 32 new keys, got {len(new_keys)}"
        for key in new_keys:
            assert key.endswith("_u"), f"Universal key '{key}' lacks '_u' suffix"
