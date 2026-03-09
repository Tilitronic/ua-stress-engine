"""Feature engineering service — 3-syllable specialist extension.

Extends ``build_features_v13()`` (97 base features) with ~22 features
that are specifically informative for 3-syllable Ukrainian words.

Design rationale
────────────────
For 3-syllable words the stress can fall on one of exactly three positions:
  * class 0 — antepenult  (proparoxytone, "на першому складі")
  * class 1 — penult      (paroxytone,    "на другому складі")
  * class 2 — oxytone     (stress on final vowel)

Three complementary insights drive the extra features:

1. **Ending-based prediction cuts prefix noise.**
   Ukrainian word stress is heavily determined by the morphological ending,
   not the stem.  The base suffix hashes (suffix_hash_3/4/5) already capture
   this, but at low moduli (512–2048).  For 3-syl words the ending matters even
   more (e.g. -ання/-ення always → penult; -увати → у stressed), so we add
   higher-resolution end-trigram + end-tetragam hashes combined with POS.

2. **Distance-from-end is a better stress descriptor than distance-from-start.**
   For any word with exactly 3 vowels, ``stress_from_end = 2 - stress_idx``.
   Encoding coda lengths (chars after each vowel) gives the model direct access
   to the distance-from-end perspective without relying on the noisy prefix zone.

3. **Linguistically motivated binary flags** encode the highest-confidence rules:
   • ви- prefix (NOUN or VERB) → antepenult stress (class 0)  [weight 0.98]
   • -ання / -ення / -іння suffix → penult stress (class 1)   [weight ~1.0]
   • -ати ending → oxytone tendency (class 2)                 [weight 0.75]
   • -ити ending → paroxytone tendency (class 1)
   • -увати suffix → stress on у (class 1 for 3-syl)
   • -ак/-яр/-ач/-ун suffix → oxytone (class 2)               [weight 0.85]
   • -ський/-зький/-цький adj → root stress (class 0)          [weight 1.0]
   • -адцять numeral suffix → penult (class 1)

Inter-vowel interval ratios and per-syllable coda lengths round out the
feature set, giving the model a fine-grained syllable-weight picture.

Total feature count:  100 (base) + 22 (3-syl specific) = 122
Exported constant: ``EXPECTED_FEATURE_COUNT_3SYL = 122``
"""

from typing import Dict, Optional

from .constants import (
    VOWEL_SET, VOWEL_CHAR_MAP, DEVERBAL_SUFFIXES,
    OXYTONE_MOBILE_SUFFIXES, ADCAT_SUFFIXES,
)
from .feature_service import build_features_v13, djb2_hash, find_vowels


# ── Public feature-count constant ────────────────────────────────────────────
EXPECTED_FEATURE_COUNT_3SYL: int = 122


# ── Suffix lists specific to 3-syllable morphology ───────────────────────────

# Infinitive suffixes that tend to be oxytone (class 2) or penult (class 1)
_VERB_ATY   = ("ати",)                        # наголошення на -а-: несАти, класти…
_VERB_YTY   = ("ити",)                        # парокситонеза: стрОїти, вАрити
_VERB_UVATY = ("увати", "ювати")              # наголос на -у-: малювАти, групувАти

# Numeral suffix → always penult for 3-syl: один-АД-цять → index 1
_NUM_ADTSYAT = ("адцять",)

# Prepositive prefix ви- triggers antepenult stress for NOUN/VERB
_VY_PREFIX = "ви"


def build_features_3syl(
    form: str,
    pos: str,
    features_json: Optional[str] = None,
) -> Dict[str, float]:
    """Build a 119-feature vector for a 3-syllable Ukrainian word.

    Starts with all 97 base features from ``build_features_v13``, then
    appends 22 features that are specifically informative when
    ``len(find_vowels(form.lower())) == 3``.

    For words with fewer or more than 3 vowels the 3-syl features are
    still returned (using safe fallbacks), so the function can be called
    on any word — but the features are most meaningful for 3-syllable tokens.

    Parameters
    ----------
    form:
        Surface word form (may include apostrophe / diaeresis).
    pos:
        Universal Dependencies POS tag (e.g. ``"NOUN"``, ``"VERB"``).
    features_json:
        Optional JSON string with UD morphological features (``Case``,
        ``Gender``, ``Tense``, …) — passed through to ``build_features_v13``.

    Returns
    -------
    Dict[str, float]
        Ordered dict with exactly ``EXPECTED_FEATURE_COUNT_3SYL`` entries.
    """
    # ── 1. Base features (97) ─────────────────────────────────────────────────
    feat = build_features_v13(form, pos, features_json)

    lower  = form.lower()
    vowels = find_vowels(lower)
    wl     = len(lower)
    ns     = len(vowels)

    # ── 2. Ending hash features — higher resolution than base ─────────────────
    # The base already hashes suf3 @ mod=512 ("trigram_ending_hash") and
    # suffix_hash_3 @ mod=1024.  For 3-syl words we want richer signals:
    #   • ending_hash_4 @ 4096 — catches -ання/-ення/-іння/-увати
    #   • ending_hash_5 @ 4096 — catches -овати/-увати/-ській
    #   • ending_pos_hash_3 @ 4096 — (suf3 + POS) compound key
    #   • ending_pos_hash_4 @ 4096 — (suf4 + POS) compound key

    suf3 = lower[-3:] if wl >= 3 else lower
    suf4 = lower[-4:] if wl >= 4 else lower
    suf5 = lower[-5:] if wl >= 5 else lower

    feat["ending_hash_4"]       = djb2_hash(suf4, 4096)
    feat["ending_hash_5"]       = djb2_hash(suf5, 4096)
    feat["ending_pos_hash_3"]   = djb2_hash(suf3 + "_" + pos, 4096)
    feat["ending_pos_hash_4"]   = djb2_hash(suf4 + "_" + pos, 4096)

    # ── 3. Vowel identity for all three positions ─────────────────────────────
    # The base supplies vowel_char_0 … vowel_char_4 (5 slots, 0-based).
    # Here we add dedicated 3-syl aliases that are always well-defined:
    #   v1_char, v2_char, v3_char  →  identity of 1st/2nd/3rd vowel  (0-9)
    # Using -1 as a sentinel when the vowel is absent.

    feat["v1_char_3syl"] = VOWEL_CHAR_MAP.get(lower[vowels[0]], -1) if ns >= 1 else -1
    feat["v2_char_3syl"] = VOWEL_CHAR_MAP.get(lower[vowels[1]], -1) if ns >= 2 else -1
    feat["v3_char_3syl"] = VOWEL_CHAR_MAP.get(lower[vowels[2]], -1) if ns >= 3 else -1

    # ── 4. Coda lengths — distance-from-end perspective ───────────────────────
    # For a 3-vowel word [V1 … V2 … V3 … end]:
    #   coda_v3  = chars after V3 (= tail coda of the final syllable)
    #   coda_v2  = chars between V2 and V3 (= inter-syllable segment V2→V3)
    #   coda_v1  = chars between V1 and V2 (= inter-syllable segment V1→V2)
    #
    # These directly encode "how far is each vowel from the end of the word",
    # which is a better stress predictor than absolute position (prefix-neutral).

    if ns >= 3:
        coda_v3 = wl - vowels[2] - 1              # chars AFTER 3rd vowel
        coda_v2 = vowels[2] - vowels[1] - 1       # chars BETWEEN V2 and V3
        coda_v1 = vowels[1] - vowels[0] - 1       # chars BETWEEN V1 and V2
    elif ns == 2:
        coda_v3 = -1
        coda_v2 = wl - vowels[1] - 1
        coda_v1 = vowels[1] - vowels[0] - 1
    elif ns == 1:
        coda_v3 = -1
        coda_v2 = -1
        coda_v1 = wl - vowels[0] - 1
    else:
        coda_v3 = coda_v2 = coda_v1 = -1

    feat["coda_after_v3"]      = coda_v3
    feat["coda_between_v2_v3"] = coda_v2
    feat["coda_between_v1_v2"] = coda_v1

    # ── 5. Inter-vowel interval ratios ────────────────────────────────────────
    # The base already stores iv_dist_0/1/2 as (vowels[i+1]-vowels[i])/wl.
    # Here we add:
    #   iv_ratio_12 = (vowels[1] - vowels[0]) / wl  — density of syllable 1
    #   iv_ratio_23 = (vowels[2] - vowels[1]) / wl  — density of syllable 2
    # (These are the same as iv_dist_0 and iv_dist_1 from the base, but the
    #  explicit 3-syl names give LightGBM a dedicated signal path for trees
    #  that specialise on 3-syl words — preventing "pollution" of the base
    #  iv_dist features that cover all syllable counts.)

    if wl > 0 and ns >= 2:
        feat["iv_ratio_12_3syl"] = (vowels[1] - vowels[0]) / wl
    else:
        feat["iv_ratio_12_3syl"] = -1.0

    if wl > 0 and ns >= 3:
        feat["iv_ratio_23_3syl"] = (vowels[2] - vowels[1]) / wl
    else:
        feat["iv_ratio_23_3syl"] = -1.0

    # ── 6. Linguistic rule flags ──────────────────────────────────────────────
    # Each flag encodes one high-confidence Ukrainian accentuation rule.
    # Weight column from rules.json shown in parentheses.

    # NOUN-03 / analogue for VERB: ви- prefix → stress on prefix (class 0)
    # weight 0.98 — "systematic feature of deverbal derivatives"
    feat["has_vy_prefix_3syl"] = int(
        lower.startswith(_VY_PREFIX) and pos in ("NOUN", "VERB")
    )

    # Deverbal nouns -ання/-ення/-іння → ALWAYS penult stress (class 1)
    # Covered by has_deverbal_suffix in base, but that uses DEVERBAL_SUFFIXES
    # which equals ("ання","ення","іння") — an exact alias.  We keep it here
    # as a 3-syl-dedicated copy so the feature name makes the intent clear
    # in feature-importance plots.
    feat["has_gerund_suffix_3syl"] = int(lower.endswith(DEVERBAL_SUFFIXES))

    # Verb infinitives -ати → oxytone tendency (class 2)  [weight 0.75]
    feat["has_verb_aty_3syl"]  = int(lower.endswith(_VERB_ATY))

    # Verb infinitives -ити → paroxytone tendency (class 1)
    feat["has_verb_yty_3syl"]  = int(lower.endswith(_VERB_YTY))

    # Verb -увати/-ювати → stress on у (penult for 3-syl)  [class 1]
    feat["has_verb_uvaty_3syl"] = int(lower.endswith(_VERB_UVATY))

    # NOUN-01: -ак/-яр/-ач/-ун → oxytone (class 2)  [weight 0.85]
    feat["has_oxytone_mobile_3syl"] = int(lower.endswith(OXYTONE_MOBILE_SUFFIXES))

    # ADJ-01 / ADJ-02: -ський/-зький/-цький → root/initial stress (class 0)
    feat["has_adcat_suffix_3syl"] = int(
        pos == "ADJ" and lower.endswith(ADCAT_SUFFIXES)
    )

    # NUM-01: -адцять → always penult stress (class 1)
    feat["has_num_adtsyat_3syl"] = int(lower.endswith(_NUM_ADTSYAT))

    # Onset length: number of characters before the first vowel.
    # Large onset → likely a prefix (ви-, пере-, над-…) that may attract stress.
    feat["onset_before_v1_3syl"] = vowels[0] if ns >= 1 else wl

    # Onset ratio: onset / word_length.  Normalised to make it comparable
    # across words of different length.
    feat["onset_ratio_3syl"] = (vowels[0] / wl) if (ns >= 1 and wl > 0) else 0.0

    return feat
