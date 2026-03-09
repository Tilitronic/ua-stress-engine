"""Feature engineering service — universal multisyllable extension.

Extends ``build_features_v13()`` (97 base features) with universal
features that scale across ALL syllable counts (2–10+).

Design rationale — why universal, not per-syllable-count
─────────────────────────────────────────────────────────
The 2S and 3S specialist models each have hard-coded feature sets (103
and 122 features respectively) with syllable-count-specific feature names.
A universal model must handle words from 2 to 10+ syllables with a SINGLE
feature schema.  The solution:

1. **Syllable-count-aware positional features** — stress-from-end encoding,
   normalised vowel positions, per-class distance features — all scale
   naturally with vowel count.

2. **"Distance from end" is the king predictor** (Караванський).
   We encode this directly: ``stress_from_end_norm``, coda lengths for
   the last 3 vowels, suffix-to-POS compound hashes at high resolution.

3. **Linguistic rule flags** — same high-confidence rules from 3S,
   but applicable to all syllable counts (ви- prefix, gerund suffix,
   -адцять, -ський, oxytone mobile suffixes).

4. **Syllable-count as an explicit feature** — the model learns
   syllable-count-specific patterns via interaction with other features.

5. **Relative position features** — normalised by syllable count so
   that "3rd vowel in a 5-vowel word" and "3rd vowel in a 10-vowel word"
   have the same relative position encoding.

Total feature count:  100 (base) + 32 (universal) = **132**
Exported constant: ``EXPECTED_FEATURE_COUNT_UNIV = 132``
"""

from typing import Dict, Optional

from .constants import (
    VOWEL_SET, VOWEL_CHAR_MAP, DEVERBAL_SUFFIXES,
    OXYTONE_MOBILE_SUFFIXES, ADCAT_SUFFIXES,
)
from .feature_service import build_features_v13, djb2_hash, find_vowels


# ── Public feature-count constant ────────────────────────────────────────────
EXPECTED_FEATURE_COUNT_UNIV: int = 132


# ── Suffix lists for linguistic rule flags ───────────────────────────────────
_VERB_ATY   = ("ати",)
_VERB_YTY   = ("ити",)
_VERB_UVATY = ("увати", "ювати")
_NUM_ADTSYAT = ("адцять",)
_VY_PREFIX = "ви"

# Foreign / Greek suffixes (strong stress attractors)
_FOREIGN_IST = ("іст", "ист", "ізм", "ант", "ент")
_GREEK_LOG   = ("лог", "граф", "фон", "скоп", "метр")

# Penult-stable suffixes
_PENULT_STABLE = ("ість", "ання", "ення", "іння")

# Compound interfixes — signal compound words
_COMPOUND_INTERFIXES = frozenset("оеєі")


def build_features_universal(
    form: str,
    pos: str,
    features_json: Optional[str] = None,
) -> Dict[str, float]:
    """Build a 130-feature vector for any Ukrainian word (2–10+ syllables).

    Starts with all 97 base features from ``build_features_v13``, then
    appends 33 universal features that scale with syllable count.

    Parameters
    ----------
    form:
        Surface word form (may include apostrophe / diaeresis).
    pos:
        Universal Dependencies POS tag (e.g. ``"NOUN"``, ``"VERB"``).
    features_json:
        Optional JSON string with UD morphological features.

    Returns
    -------
    Dict[str, float]
        Ordered dict with exactly ``EXPECTED_FEATURE_COUNT_UNIV`` entries.
    """
    # ── 1. Base features (97) ─────────────────────────────────────────────────
    feat = build_features_v13(form, pos, features_json)

    lower  = form.lower()
    vowels = find_vowels(lower)
    wl     = len(lower)
    ns     = len(vowels)

    # ── 2. Syllable count bucket (explicit feature) ───────────────────────────
    # The model needs to know ns to learn syllable-count-specific patterns.
    # Also provide a normalised version (ns / max_practical_ns).
    feat["syllable_count_u"]      = ns
    feat["syllable_count_norm_u"] = min(ns / 10.0, 1.0)  # cap at 10

    # ── 3. Ending hash features — higher resolution ──────────────────────────
    suf3 = lower[-3:] if wl >= 3 else lower
    suf4 = lower[-4:] if wl >= 4 else lower
    suf5 = lower[-5:] if wl >= 5 else lower
    suf6 = lower[-6:] if wl >= 6 else lower

    feat["ending_hash_4_u"]     = djb2_hash(suf4, 4096)
    feat["ending_hash_5_u"]     = djb2_hash(suf5, 4096)
    feat["ending_hash_6_u"]     = djb2_hash(suf6, 4096)
    feat["ending_pos_hash_3_u"] = djb2_hash(suf3 + "_" + pos, 4096)
    feat["ending_pos_hash_4_u"] = djb2_hash(suf4 + "_" + pos, 4096)

    # ── 4. Distance-from-end features (the king predictor) ───────────────────
    # For the last 3 vowels: how many chars from word-end to each vowel.
    # This directly encodes masculine/feminine/dactylic stress patterns.
    if ns >= 1:
        _d = wl - vowels[-1] - 1
        feat["dist_from_end_v_last_u"]      = _d
        feat["dist_from_end_v_last_norm_u"] = _d / wl if wl else 0
    else:
        feat["dist_from_end_v_last_u"]      = -1
        feat["dist_from_end_v_last_norm_u"] = -1.0

    if ns >= 2:
        _d = wl - vowels[-2] - 1
        feat["dist_from_end_v_pen_u"]       = _d
        feat["dist_from_end_v_pen_norm_u"]  = _d / wl if wl else 0
    else:
        feat["dist_from_end_v_pen_u"]       = -1
        feat["dist_from_end_v_pen_norm_u"]  = -1.0

    if ns >= 3:
        _d = wl - vowels[-3] - 1
        feat["dist_from_end_v_ante_u"]      = _d
        feat["dist_from_end_v_ante_norm_u"] = _d / wl if wl else 0
    else:
        feat["dist_from_end_v_ante_u"]      = -1
        feat["dist_from_end_v_ante_norm_u"] = -1.0

    # ── 5. Coda lengths — last 3 inter-vowel segments ────────────────────────
    # coda_last = chars after last vowel (final syllable coda)
    # coda_pen  = chars between penult and last vowel
    # coda_ante = chars between antepenult and penult
    if ns >= 1:
        feat["coda_last_u"] = wl - vowels[-1] - 1
    else:
        feat["coda_last_u"] = -1

    if ns >= 2:
        feat["coda_pen_u"] = vowels[-1] - vowels[-2] - 1
    else:
        feat["coda_pen_u"] = -1

    if ns >= 3:
        feat["coda_ante_u"] = vowels[-2] - vowels[-3] - 1
    else:
        feat["coda_ante_u"] = -1

    # ── 6. Vowel identity for last 3 positions ───────────────────────────────
    feat["v_last_char_u"]  = VOWEL_CHAR_MAP.get(lower[vowels[-1]], -1) if ns >= 1 else -1
    feat["v_pen_char_u"]   = VOWEL_CHAR_MAP.get(lower[vowels[-2]], -1) if ns >= 2 else -1
    feat["v_ante_char_u"]  = VOWEL_CHAR_MAP.get(lower[vowels[-3]], -1) if ns >= 3 else -1

    # ── 7. Inter-vowel intervals for the last 2 gaps ─────────────────────────
    if wl > 0 and ns >= 2:
        feat["iv_last_gap_u"] = (vowels[-1] - vowels[-2]) / wl
    else:
        feat["iv_last_gap_u"] = -1.0

    if wl > 0 and ns >= 3:
        feat["iv_pen_gap_u"] = (vowels[-2] - vowels[-3]) / wl
    else:
        feat["iv_pen_gap_u"] = -1.0

    # ── 8. Linguistic rule flags (universal) ─────────────────────────────────
    # ви- prefix (NOUN/VERB) → early stress
    feat["has_vy_prefix_u"] = int(
        lower.startswith(_VY_PREFIX) and pos in ("NOUN", "VERB")
    )
    # Deverbal gerunds -ання/-ення/-іння → penult stress
    feat["has_gerund_suffix_u"] = int(lower.endswith(DEVERBAL_SUFFIXES))

    # Verb infinitive suffixes
    feat["has_verb_aty_u"]   = int(lower.endswith(_VERB_ATY))
    feat["has_verb_yty_u"]   = int(lower.endswith(_VERB_YTY))
    feat["has_verb_uvaty_u"] = int(lower.endswith(_VERB_UVATY))

    # Oxytone mobile suffixes -ак/-яр/-ач/-ун
    feat["has_oxytone_mobile_u"] = int(lower.endswith(OXYTONE_MOBILE_SUFFIXES))

    # Adjective -ський/-зький/-цький → root stress
    feat["has_adcat_suffix_u"] = int(
        pos == "ADJ" and lower.endswith(ADCAT_SUFFIXES)
    )

    # Numeral -адцять → always penult
    feat["has_num_adtsyat_u"] = int(lower.endswith(_NUM_ADTSYAT))

    # Foreign/Greek suffixes (strong stress attractor to last syllable)
    feat["has_foreign_suffix_u"] = int(lower.endswith(_FOREIGN_IST))
    feat["has_greek_suffix_u"]   = int(lower.endswith(_GREEK_LOG))

    # Penult-stable suffixes -ість/-ання/-ення/-іння
    feat["has_penult_stable_u"] = int(lower.endswith(_PENULT_STABLE))

    return feat
