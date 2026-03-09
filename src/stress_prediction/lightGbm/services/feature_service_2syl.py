"""Extended feature engineering for 2-syllable specialist.

Adds 2-syllable-specific features on top of the base 97 features:
    - first_vowel_char (10 categories: а,е,є,и,і,ї,о,у,ю,я)
    - second_vowel_char (10 categories)
    - consonant_cluster_between_vowels (complexity: 0-5)
    - suffix_after_v2_len (0-10)
    - consonant_before_v1_count (0-5)
    - consonant_after_v2_count (0-5)

Total: 97 + 6 = **103 features** for 2-syllable binary classifier.
"""

from typing import Dict, Optional
import json

from .feature_service import (
    build_features_v13, find_vowels, djb2_hash,
)
from .constants import VOWEL_SET, VOWEL_CHAR_MAP


def build_features_2syl(form: str, pos: str, features_json: Optional[str] = None) -> Dict[str, float]:
    """Build 103-feature vector for 2-syllable words.

    Extends base 97 features with 2-syllable-specific features.
    """
    # Start with base 97 features
    features = build_features_v13(form, pos, features_json)

    lower = form.lower()
    vowels = find_vowels(lower)

    # If not exactly 2 vowels, return base features + zeros
    if len(vowels) != 2:
        features.update({
            "first_vowel_char": 0,
            "second_vowel_char": 0,
            "consonant_cluster_between": 0,
            "suffix_after_v2_len": 0,
            "consonant_before_v1_count": 0,
            "consonant_after_v2_count": 0,
        })
        return features

    v1_pos, v2_pos = vowels[0], vowels[1]
    v1_char = lower[v1_pos]
    v2_char = lower[v2_pos]

    # Feature 98: first_vowel_char (0-9, mapped from VOWEL_CHAR_MAP)
    features["first_vowel_char"] = VOWEL_CHAR_MAP.get(v1_char, 0)

    # Feature 99: second_vowel_char (0-9)
    features["second_vowel_char"] = VOWEL_CHAR_MAP.get(v2_char, 0)

    # Feature 100: consonant cluster complexity between v1 and v2
    between = lower[v1_pos + 1: v2_pos]
    cons_count = sum(1 for c in between if c.isalpha() and c not in VOWEL_SET)
    features["consonant_cluster_between"] = min(cons_count, 5)

    # Feature 101: suffix length after v2 (up to 10)
    suffix_after_v2 = lower[v2_pos + 1:]
    features["suffix_after_v2_len"] = min(len(suffix_after_v2), 10)

    # Feature 102: consonant count before v1 (up to 5)
    prefix_before_v1 = lower[:v1_pos]
    cons_before = sum(1 for c in prefix_before_v1 if c.isalpha() and c not in VOWEL_SET)
    features["consonant_before_v1_count"] = min(cons_before, 5)

    # Feature 103: consonant count after v2 (up to 5)
    cons_after = sum(1 for c in suffix_after_v2 if c.isalpha() and c not in VOWEL_SET)
    features["consonant_after_v2_count"] = min(cons_after, 5)

    return features


def expected_feature_count_2syl() -> int:
    """Return expected feature count for 2-syllable specialist."""
    return 103  # 97 base + 6 2syl-specific


# Export count
EXPECTED_FEATURE_COUNT_2SYL = 103
