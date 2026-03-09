"""Feature engineering service — hashing helpers and the v1.3 feature builder.

All pure functions: no I/O, no side-effects.  Takes a word form + POS + optional
morphological JSON and returns a dict of 97 numeric features.
"""

from typing import Dict, List, Optional, Tuple
import json
import numpy as np

from .constants import (
    VOWEL_SET, VOWEL_CHAR_MAP, POS_INT, PATTERN_INT,
    CASE_MAP, GENDER_MAP, NUMBER_MAP, TENSE_MAP, ASPECT_MAP, DEGREE_MAP,
    MASC_STRESS_SUFFIXES, FOREIGN_FINAL_SUFFIXES, GREEK_INTERFIX_SUFFIXES,
    DEVERBAL_SUFFIXES, MEASURE_SUFFIXES, DIMINUTIVE_ADJ, ROOT_STRESS_ADJ,
    COMPOUND_INTERFIXES, COMMON_PREFIXES, EXPECTED_FEATURE_COUNT,
    OXYTONE_MOBILE_SUFFIXES, PENULT_STABLE_SUFFIXES, ADCAT_SUFFIXES,
)


# ────────────────────────────────────────────────────────────────────
# Low-level helpers
# ────────────────────────────────────────────────────────────────────
def djb2_hash(s: str, mod: int) -> int:
    """DJB2 string hash (Bernstein, 1991) reduced modulo *mod*."""
    h = 5381
    for c in s:
        h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
    return h % mod


def find_vowels(word: str) -> List[int]:
    """Return 0-based positions of every Ukrainian vowel in *word*."""
    return [i for i, c in enumerate(word) if c in VOWEL_SET]


def syllable_onset_pattern(lower: str) -> int:
    vowels = find_vowels(lower)
    if not vowels:
        return PATTERN_INT["no_vowel"]
    fv = vowels[0]
    pat = "".join("V" if lower[i] in VOWEL_SET else "C" for i in range(fv + 1))
    return PATTERN_INT.get(pat, 6)


def last_syllable_open(lower: str, vowels: List[int]) -> int:
    if not vowels:
        return 0
    return int(vowels[-1] == len(lower) - 1)


def max_consonant_cluster(lower: str) -> int:
    max_run = cur_run = 0
    for c in lower:
        if c not in VOWEL_SET and c.isalpha():
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def count_open_syllables(lower: str, vowels: List[int]) -> int:
    count = 0
    for i, v_pos in enumerate(vowels):
        if v_pos == len(lower) - 1:
            count += 1
        elif i + 1 < len(vowels):
            gap = lower[v_pos + 1: vowels[i + 1]]
            if len(gap) <= 1:
                count += 1
    return count


def cv_shape(lower: str) -> str:
    return "".join("V" if c in VOWEL_SET else "C" for c in lower if c.isalpha())


def vowel_context_hash(lower: str, vowel_pos: int, mod: int) -> int:
    before = lower[vowel_pos - 1] if vowel_pos > 0 else "^"
    after = lower[vowel_pos + 1] if vowel_pos + 1 < len(lower) else "$"
    return djb2_hash(before + lower[vowel_pos] + after, mod)


def vowel_wide_context_hash(lower: str, vp: int, wl: int, mod: int) -> int:
    chars = []
    for off in (-2, -1, 0, 1, 2):
        p = vp + off
        chars.append(lower[p] if 0 <= p < wl else ("^" if p < 0 else "$"))
    return djb2_hash("".join(chars), mod)


def syllable_weight(lower: str, vowels: List[int], syl_idx: int) -> int:
    if not vowels or syl_idx >= len(vowels):
        return 0
    v_pos = vowels[syl_idx]
    end = vowels[syl_idx + 1] if syl_idx + 1 < len(vowels) else len(lower)
    coda = sum(1 for i in range(v_pos + 1, end)
               if lower[i] not in VOWEL_SET and lower[i].isalpha())
    return 0 if coda == 0 else (1 if coda == 1 else 2)


def detect_compound_interfix(lower: str, vowels: List[int]) -> Tuple[int, int]:
    wl = len(lower)
    if wl < 6 or len(vowels) < 3:
        return 0, 0
    for i in range(2, wl - 2):
        if lower[i] in COMPOUND_INTERFIXES:
            if (i > 0 and lower[i - 1] not in VOWEL_SET and lower[i - 1].isalpha()
                    and i + 1 < wl and lower[i + 1] not in VOWEL_SET
                    and lower[i + 1].isalpha()):
                ratio = i / wl
                if 0.2 < ratio < 0.7:
                    return 1, int(ratio * 100)
    return 0, 0


def count_prefix_matches(lower: str) -> Tuple[int, int]:
    max_len = 0
    count = 0
    for pre in COMMON_PREFIXES:
        if lower.startswith(pre):
            count += 1
            if len(pre) > max_len:
                max_len = len(pre)
    return count, max_len


def parse_morph_features(features_json_str: Optional[str]) -> dict:
    result = {
        "case_int": -1, "gender_int": -1, "number_int": -1,
        "tense_int": -1, "aspect_int": -1, "degree_int": -1,
        "n_morph_features": 0,
    }
    if not features_json_str:
        return result
    try:
        feats = json.loads(features_json_str)
        if not isinstance(feats, dict):
            return result
    except Exception:
        return result
    result["n_morph_features"] = len(feats)
    result["case_int"] = CASE_MAP.get(feats.get("Case", ""), -1)
    result["gender_int"] = GENDER_MAP.get(feats.get("Gender", ""), -1)
    result["number_int"] = NUMBER_MAP.get(feats.get("Number", ""), -1)
    result["tense_int"] = TENSE_MAP.get(feats.get("Tense", ""), -1)
    result["aspect_int"] = ASPECT_MAP.get(feats.get("Aspect", ""), -1)
    result["degree_int"] = DEGREE_MAP.get(feats.get("Degree", ""), -1)
    return result


# ────────────────────────────────────────────────────────────────────
# Main feature builder
# ────────────────────────────────────────────────────────────────────
def build_features_v13(form: str, pos: str,
                       features_json: Optional[str] = None) -> Dict[str, float]:
    """Build the v1.3-compatible 97-feature vector for a single word.

    Pure function — no side-effects.  Returns an ordered dict of
    ``{feature_name: numeric_value}``.
    """
    lower = form.lower()
    vowels = find_vowels(lower)
    wl = len(lower)
    ns = len(vowels)

    if vowels:
        fv, lv = vowels[0], vowels[-1]
    else:
        fv = lv = 0
    dist_first = fv if vowels else 0
    dist_last = (wl - lv - 1) if vowels else 0
    fvr = fv / wl if wl else 0.0
    lvr = lv / wl if wl else 0.0

    vr = [-1.0] * 5
    for i, vp in enumerate(vowels[:5]):
        vr[i] = vp / wl if wl else 0.0
    vs = (lv - fv) / wl if (wl and ns >= 2) else 0.0
    mvr = vowels[ns // 2] / wl if (wl and vowels) else 0.0

    penult_r = vowels[-2] / wl if (wl and ns >= 2) else -1.0
    antepen_r = vowels[-3] / wl if (wl and ns >= 3) else -1.0

    iv = [-1.0] * 4
    for i in range(min(ns - 1, 4)):
        iv[i] = (vowels[i + 1] - vowels[i]) / wl if wl else 0.0

    suf2 = lower[-2:] if wl >= 2 else lower
    suf3 = lower[-3:] if wl >= 3 else lower
    suf4 = lower[-4:] if wl >= 4 else lower
    suf5 = lower[-5:] if wl >= 5 else lower
    suf6 = lower[-6:] if wl >= 6 else lower
    pre2 = lower[:2] if wl >= 2 else lower
    pre3 = lower[:3] if wl >= 3 else lower
    pre4 = lower[:4] if wl >= 4 else lower

    bplv = djb2_hash(lower[lv - 2: lv], 512) if lv >= 2 else 0
    bpfv = djb2_hash(lower[fv + 1: fv + 3], 512) if (fv + 2 < wl and vowels) else 0
    max_cl = max_consonant_cluster(lower)
    n_open = count_open_syllables(lower, vowels) if vowels else 0
    last_syl_open_v = last_syllable_open(lower, vowels)

    s4c = lower[-4:] if wl >= 4 else lower
    s4vc = sum(1 for c in s4c if c in VOWEL_SET)

    has_masc = int(pos == "NOUN" and lower.endswith(MASC_STRESS_SUFFIXES))
    has_foreign = int(lower.endswith(FOREIGN_FINAL_SUFFIXES))
    has_greek = int(lower.endswith(GREEK_INTERFIX_SUFFIXES))
    has_vy_n = int(lower.startswith("ви") and pos == "NOUN")
    has_deverbal = int(lower.endswith(DEVERBAL_SUFFIXES))
    has_measure = int(lower.endswith(MEASURE_SUFFIXES))
    has_dim_adj = int(pos == "ADJ" and any(s in lower for s in DIMINUTIVE_ADJ))
    has_root_adj = int(pos == "ADJ" and any(s in lower for s in ROOT_STRESS_ADJ))
    is_inf = int(pos == "VERB" and lower.endswith(("ти", "сти", "зти")))
    is_v1p2p = int(pos == "VERB" and lower.endswith(("емо", "имо", "ємо", "ете", "ите")))
    has_vy_v = int(lower.startswith("ви") and pos == "VERB")
    is_refl = int(lower.endswith(("ся", "сь")))
    is_bisyl = int(ns == 2)
    stress_first = int(lower.startswith("ви") and pos in ("NOUN", "VERB"))

    # Accent paradigm features (new v1.4)
    has_oxytone_mobile = int(lower.endswith(OXYTONE_MOBILE_SUFFIXES))   # paradigm B: stress on suffix/ending
    has_penult_stable  = int(lower.endswith(PENULT_STABLE_SUFFIXES))    # -ість only (2-syl: радість, злість)
    has_adcat_suffix   = int(pos == "ADJ" and lower.endswith(ADCAT_SUFFIXES))  # paradigm A: ський/зький/цький

    vch = [-1] * 5
    for i, vp in enumerate(vowels[:5]):
        vch[i] = VOWEL_CHAR_MAP.get(lower[vp], -1)

    vctx = [0] * 5
    for i, vp in enumerate(vowels[:5]):
        vctx[i] = vowel_context_hash(lower, vp, 512)

    vctx_wide = [0] * 5
    for i, vp in enumerate(vowels[:5]):
        vctx_wide[i] = vowel_wide_context_hash(lower, vp, wl, 1024)

    rv = [-1.0] * 5
    for i in range(min(ns, 5)):
        rv[i] = i / ns if ns else 0.0

    cvsh = djb2_hash(cv_shape(lower), 2048)
    soft_cnt = lower.count("\u044c")
    soft_pr = lower.rfind("\u044c") / wl if (wl and "\u044c" in lower) else -1.0
    has_dbl = 0
    for i in range(wl - 1):
        if lower[i] == lower[i + 1] and lower[i] not in VOWEL_SET and lower[i].isalpha():
            has_dbl = 1
            break

    syl_weights = [syllable_weight(lower, vowels, si) for si in range(ns)] if ns else []
    heavy = sum(1 for w in syl_weights if w >= 2)
    final_sw = syl_weights[-1] if syl_weights else 0
    init_sw = syl_weights[0] if syl_weights else 0
    syl_weight_var = float(np.var(syl_weights)) if len(syl_weights) >= 2 else 0.0

    has_interfix, interfix_pos_pct = detect_compound_interfix(lower, vowels)
    prefix_count, longest_prefix = count_prefix_matches(lower)

    third = max(1, wl // 3)
    vd_first = sum(1 for c in lower[:third] if c in VOWEL_SET) / third if third > 0 else 0
    vd_mid = (sum(1 for c in lower[third:2 * third] if c in VOWEL_SET) / third
              if third > 0 else 0)
    vd_last = (sum(1 for c in lower[2 * third:] if c in VOWEL_SET) /
               max(1, wl - 2 * third)) if wl > 2 * third else 0

    morph = parse_morph_features(features_json)

    return {
        "word_len": wl, "vowel_count": ns,
        "dist_to_first_vowel": dist_first, "dist_from_last_vowel": dist_last,
        "first_vowel_ratio": fvr, "last_vowel_ratio": lvr,
        "vowel_ratio_0": vr[0], "vowel_ratio_1": vr[1], "vowel_ratio_2": vr[2],
        "vowel_ratio_3": vr[3], "vowel_ratio_4": vr[4],
        "vowel_span": vs, "mid_vowel_ratio": mvr,
        "vowel_pair_hash": djb2_hash(lower[fv] + lower[lv], 256) if vowels else 0,
        "penult_ratio": penult_r, "antepenult_ratio": antepen_r,
        "iv_dist_0": iv[0], "iv_dist_1": iv[1], "iv_dist_2": iv[2], "iv_dist_3": iv[3],
        "suffix_hash_2": djb2_hash(suf2, 512), "suffix_hash_3": djb2_hash(suf3, 1024),
        "suffix_hash_4": djb2_hash(suf4, 2048), "suffix_hash_5": djb2_hash(suf5, 2048),
        "suffix_hash_6": djb2_hash(suf6, 2048), "prefix_hash_2": djb2_hash(pre2, 512),
        "prefix_hash_3": djb2_hash(pre3, 1024), "prefix_hash_4": djb2_hash(pre4, 1024),
        "suffix_pos_hash": djb2_hash(suf3 + "_" + pos, 2048),
        "bigram_pre_lv_hash": bplv, "bigram_post_fv_hash": bpfv,
        "trigram_ending_hash": djb2_hash(suf3, 512),
        "onset_cluster_len": fv if vowels else 0, "coda_cluster_len": dist_last,
        "max_cluster_len": max_cl,
        "num_open_syllables": n_open, "last_syllable_open": last_syl_open_v,
        "suffix4_vowel_count": s4vc,
        "suffix4_vowel_ratio": s4vc / len(s4c) if s4c else 0.0,
        "char_diversity": len(set(lower)) / wl if wl else 0.0,
        "has_apostrophe": int("'" in form or "\u02bc" in form or "\u2019" in form),
        "pos_int": POS_INT.get(pos, len(POS_INT)),
        "syllable_pattern": syllable_onset_pattern(lower),
        "has_masc_stress_suffix": has_masc, "has_foreign_final_suffix": has_foreign,
        "has_greek_interfix": has_greek, "has_vy_prefix_noun": has_vy_n,
        "has_deverbal_suffix": has_deverbal, "has_measure_suffix": has_measure,
        "has_diminutive_adj": has_dim_adj, "has_root_stress_adj": has_root_adj,
        "is_infinitive": is_inf, "is_verb_1pl_2pl": is_v1p2p,
        "has_vy_prefix_verb": has_vy_v, "is_reflexive": is_refl,
        "is_bisyllable": is_bisyl, "stress_on_first_likely": stress_first,
        "vowel_char_0": vch[0], "vowel_char_1": vch[1], "vowel_char_2": vch[2],
        "vowel_char_3": vch[3], "vowel_char_4": vch[4],
        "vowel_ctx_hash_0": vctx[0], "vowel_ctx_hash_1": vctx[1],
        "vowel_ctx_hash_2": vctx[2], "vowel_ctx_hash_3": vctx[3],
        "vowel_ctx_hash_4": vctx[4],
        "rel_vowel_0": rv[0], "rel_vowel_1": rv[1], "rel_vowel_2": rv[2],
        "rel_vowel_3": rv[3], "rel_vowel_4": rv[4],
        "cv_shape_hash": cvsh,
        "soft_sign_count": soft_cnt, "soft_sign_pos_ratio": soft_pr,
        "has_double_consonant": has_dbl,
        "heavy_syllable_count": heavy, "final_syllable_weight": final_sw,
        "initial_syllable_weight": init_sw,
        "vowel_wide_ctx_0": vctx_wide[0], "vowel_wide_ctx_1": vctx_wide[1],
        "vowel_wide_ctx_2": vctx_wide[2], "vowel_wide_ctx_3": vctx_wide[3],
        "vowel_wide_ctx_4": vctx_wide[4],
        "has_compound_interfix": has_interfix,
        "interfix_pos_pct": interfix_pos_pct,
        "prefix_match_count": prefix_count,
        "longest_prefix_len": longest_prefix,
        "syllable_weight_var": syl_weight_var,
        "vowel_density_first_third": round(vd_first, 4),
        "vowel_density_mid_third": round(vd_mid, 4),
        "vowel_density_last_third": round(vd_last, 4),
        "morph_case": morph["case_int"],
        "morph_gender": morph["gender_int"],
        "morph_number": morph["number_int"],
        "morph_tense": morph["tense_int"],
        "morph_n_features": morph["n_morph_features"],
        # Accent paradigm features
        "has_oxytone_mobile_suffix": has_oxytone_mobile,
        "has_penult_stable_suffix": has_penult_stable,
        "has_adcat_suffix": has_adcat_suffix,
    }
