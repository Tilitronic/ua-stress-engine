"""Evaluation service — external, handcrafted, and composite fitness.

Unified service reusable across all specialist models (2-syllable, 3-syllable,
…) and the global multiclass model (Luscinia v1.0+).

Key design choices
──────────────────
* ``mode="multiclass"`` (default) — prediction via ``argmax`` over a probability
  vector; intended for the global model with N > 2 stress-position classes.
* ``mode="binary"`` — prediction via ``>= threshold`` on a scalar output;
  intended for syllable-count specialists (2-syl, 3-syl, …) that output a
  single stress-position probability.
* ``compute_fitness`` accepts an explicit ``weights`` dict so each training
  script can define its own objective without duplicating code.  When
  ``penalty_mode=True`` it returns ``(fitness, penalty_flag)`` instead of a
  bare float.

Fitness design (v2 — Luscinia redesign)
────────────────────────────────────────
Sanity accuracy (external DB sample) is NO LONGER part of the fitness formula.
It is a *data-quality guard* only:

  • If sanity_acc >= val_acc  →  no effect on score.
  • If sanity_acc < val_acc   →  the score is multiplied by a large penalty
    factor (``SANITY_BELOW_ACC_PENALTY = 0.50``).  This is intentionally
    severe: sanity < val_acc almost certainly means a data leak, label bug, or
    encoding mismatch — the model should not be selected under any
    circumstances.  A "⚠ SANITY BELOW ACC" warning is emitted by the training
    scripts.

The actual optimisation objective is split among three components only:

  Component        Weight   Rationale
  ──────────────── ──────   ─────────────────────────────────────────────────
  f1               45 %     Primary quality signal; class-balanced.
  acc              20 %     Val-set per-form accuracy; secondary discriminator.
  hand_acc         35 %     Handcrafted poetic words; closest to production.

A tiny **model-size tiebreaker** (+up to 2 % of the final score) favours
smaller models when everything else is equal:

  size_bonus = 0.02 * (1.0 - min(1.0, size_mb / SIZE_BONUS_CAP_MB))

  • At 0 MB  → +0.020 (theoretical max, never reached in practice).
  • At 50 MB → +0.000 (no bonus).
  • A 3 MB model vs a 6 MB model with identical accuracy: +0.018 vs +0.016.
    The difference (~0.002) is smaller than a single correct handcrafted word
    (~hand_acc_weight / hand_total ≈ 0.35 / 44 ≈ 0.008).  Size never beats
    accuracy.

Weight preset overview
──────────────────────
* ``"global_v1_4"``             — legacy multiclass global model (unchanged).
* ``"specialist_binary"``       — Phase 1/2 landscape mapping (all Luscinia).
* ``"luscinia_specialist"``     — Luscinia 2S Phase 3 TPE fine-tuning.
* ``"luscinia_3s_specialist"``  — Luscinia 3S Phase 3 (macro-F1 variant).

All three Luscinia presets share the same component weights; they differ only
in that the global preset retains the old ``syl_acc`` component for backward
compatibility with analysis scripts that read old result JSON files.

All functions are pure computation — no I/O or logging.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import SYLLABLE_BUCKETS
from .feature_service import build_features_v13


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _predict_one(booster, fvec: np.ndarray, mode: str,
                 threshold: float) -> Tuple[int, float]:
    """Return (predicted_class, confidence) for a single feature vector.

    Parameters
    ----------
    booster:
        Trained lightgbm booster.
    fvec:
        Shape ``(1, n_features)`` float32 array.
    mode:
        ``"multiclass"`` — argmax over probability vector.
        ``"binary"`` — threshold on scalar output.
    threshold:
        Decision boundary for binary mode (default 0.5).

    Returns
    -------
    predicted_class : int
    confidence : float  — P(predicted_class)
    """
    raw = booster.predict(fvec)
    if mode == "binary":
        p1 = float(raw.ravel()[0])
        predicted = 1 if p1 >= threshold else 0
        confidence = p1 if predicted == 1 else 1.0 - p1
    else:
        proba = raw[0]
        predicted = int(np.argmax(proba))
        confidence = float(proba[predicted])
    return predicted, confidence


# ────────────────────────────────────────────────────────────────────
# External DB-sample evaluation
# ────────────────────────────────────────────────────────────────────

def evaluate_external(
    booster,
    feature_columns: List[str],
    external_sample: List[dict],
    *,
    mode: str = "multiclass",
    threshold: float = 0.5,
) -> dict:
    """Evaluate on an external DB word sample with per-syllable breakdown.

    Parameters
    ----------
    booster:
        Trained lightgbm booster.
    feature_columns:
        Ordered list of feature names the booster was trained on.
    external_sample:
        List of dicts with keys ``form``, ``pos``, ``expected_label``,
        and optionally ``features_json``, ``n_syllables``, ``vowels``.
    mode:
        ``"multiclass"`` (default) or ``"binary"``.  Controls how the raw
        booster output is converted to a predicted class.
    threshold:
        Decision threshold for ``mode="binary"`` (default 0.5).

    Returns
    -------
    dict::

        {
            "accuracy": float,
            "correct": int,
            "sample_size": int,
            "per_syllable": {bucket: {"correct", "total", "accuracy"}},
            "mean_syllable_accuracy": float,
        }
    """
    ext_correct = 0
    ext_total = 0
    by_syl: Dict[int, dict] = {}

    for item in external_sample:
        features = build_features_v13(item["form"], item["pos"],
                                      item.get("features_json"))
        fvec = np.array([[features.get(c, 0) for c in feature_columns]],
                        dtype=np.float32)
        predicted, _ = _predict_one(booster, fvec, mode, threshold)
        ext_total += 1

        n_syl = item.get("n_syllables", len(item.get("vowels", [])))
        bucket = min(n_syl, max(SYLLABLE_BUCKETS))
        if bucket not in by_syl:
            by_syl[bucket] = {"correct": 0, "total": 0}
        by_syl[bucket]["total"] += 1

        if predicted == item["expected_label"]:
            ext_correct += 1
            by_syl[bucket]["correct"] += 1

    ext_acc = ext_correct / ext_total if ext_total > 0 else 0.0

    syl_accs: Dict[int, dict] = {}
    bucket_accs: List[float] = []
    for b in sorted(by_syl.keys()):
        v = by_syl[b]
        acc = v["correct"] / v["total"] if v["total"] > 0 else 0.0
        syl_accs[b] = {"correct": v["correct"], "total": v["total"],
                       "accuracy": round(acc, 4)}
        bucket_accs.append(acc)

    mean_syl_acc = sum(bucket_accs) / len(bucket_accs) if bucket_accs else 0.0

    return {
        "accuracy": round(ext_acc, 6),
        "correct": ext_correct,
        "sample_size": ext_total,
        "per_syllable": syl_accs,
        "mean_syllable_accuracy": round(mean_syl_acc, 6),
    }


# ────────────────────────────────────────────────────────────────────
# Handcrafted word-list evaluation
# ────────────────────────────────────────────────────────────────────

def evaluate_handcrafted(
    booster,
    feature_columns: List[str],
    handcrafted_tests: List[tuple],
    *,
    mode: str = "multiclass",
    threshold: float = 0.5,
) -> dict:
    """Evaluate on a handcrafted word-list with known (or unknown) labels.

    Entries with ``expected=None`` (stress unknown) are predicted and
    logged but excluded from the accuracy score.

    Parameters
    ----------
    booster:
        Trained lightgbm booster.
    feature_columns:
        Ordered list of feature names.
    handcrafted_tests:
        Each entry is a tuple of at least 3 elements:
        ``(word, pos, expected [, description [, features_json]])``
        where ``expected`` is an ``int``, a ``list[int]`` (ambiguous), or
        ``None`` (unknown).
    mode:
        ``"multiclass"`` (default) or ``"binary"``.
    threshold:
        Decision threshold for ``mode="binary"`` (default 0.5).

    Returns
    -------
    dict::

        {
            "accuracy": float,
            "correct": int,
            "total": int,            # scoreable words only
            "total_words": int,      # all words, including unknown
            "results": [per-word dicts],
        }

    Per-word dicts contain: ``word``, ``pos``, ``expected``, ``predicted``,
    ``correct``, ``scoreable``, ``confidence``.
    """
    hand_correct = 0
    hand_total = 0
    hand_results = []

    for entry in handcrafted_tests:
        if len(entry) >= 5:
            word, pos, expected, _desc, features_json = entry[:5]
        elif len(entry) == 4:
            word, pos, expected, _desc = entry
            features_json = None
        else:
            word, pos, expected = entry[:3]
            features_json = None

        features = build_features_v13(word, pos, features_json)
        fvec = np.array([[features.get(c, 0) for c in feature_columns]],
                        dtype=np.float32)
        predicted, confidence = _predict_one(booster, fvec, mode, threshold)

        scoreable = expected is not None
        if scoreable:
            hit: Optional[bool] = (predicted in expected
                                   if isinstance(expected, list)
                                   else predicted == expected)
            hand_total += 1
            if hit:
                hand_correct += 1
        else:
            hit = None

        hand_results.append({
            "word": word,
            "pos": pos,
            "expected": expected,
            "predicted": predicted,
            "correct": hit,
            "scoreable": scoreable,
            "confidence": round(confidence, 4),
        })

    hand_acc = hand_correct / hand_total if hand_total > 0 else 0.0
    return {
        "accuracy": round(hand_acc, 6),
        "correct": hand_correct,
        "total": hand_total,
        "total_words": len(handcrafted_tests),
        "results": hand_results,
    }


# ────────────────────────────────────────────────────────────────────
# Composite fitness — constants
# ────────────────────────────────────────────────────────────────────

# When sanity_acc < val_acc the model is penalised by this multiplier.
# 0.50 = score is halved — severe enough that no penalised model will ever
# beat an unpenalised one even if all other metrics are perfect.
SANITY_BELOW_ACC_PENALTY: float = 0.50

# Model size above this threshold receives zero size-tiebreaker bonus.
SIZE_BONUS_CAP_MB: float = 50.0

# Maximum contribution of the size tiebreaker to the final fitness score.
# Kept small enough that it NEVER overrides even a single correct handcrafted
# word (which contributes ~hand_weight / hand_total ≈ 0.35/44 ≈ 0.008).
SIZE_BONUS_MAX: float = 0.02

# ────────────────────────────────────────────────────────────────────
# Weight presets
# ────────────────────────────────────────────────────────────────────
#
# Sanity accuracy is intentionally ABSENT from all Luscinia presets.
# It is evaluated separately as a guard, not as an optimisation component.
#
# Component weights must sum to 1.0 (excluding the size tiebreaker,
# which is applied after the weighted sum as an additive bonus).
#
# "global_v1_4"           — legacy global multiclass model (unchanged for
#                           backward-compat with analysis scripts).
# "specialist_binary"     — Phase 1 / Phase 2 landscape mapping; used by
#                           all Luscinia scripts in early phases so that the
#                           fitness landscape shape is consistent across runs.
# "luscinia_specialist"   — Luscinia 2S Phase 3 TPE fine-tuning.
#                           hand_acc boosted vs specialist_binary (35% vs 25%)
#                           so Phase 3 aggressively targets real poetic words.
# "luscinia_3s_specialist"— Luscinia 3S Phase 3; identical weights to
#                           luscinia_specialist.  f1 is macro-F1 over 3 classes.
#
_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    # ── Legacy global preset (backward-compat, untouched) ──────────────────
    "global_v1_4": {
        "f1":        0.30,
        "ext_acc":   0.25,
        "hand_acc":  0.20,
        "syl_acc":   0.25,
    },
    # ── Phase 1 / Phase 2 landscape preset (all Luscinia scripts) ──────────
    # Moderate hand_acc weight so the search covers broad hyperparameter space
    # without prematurely over-fitting to the 44-word handcrafted set.
    "specialist_binary": {
        "f1":        0.45,   # primary: balanced val-set quality
        "acc":       0.30,   # secondary: per-form val accuracy
        "hand_acc":  0.25,   # tertiary: real poetic edge-cases
        # sanity: zero weight — evaluated as guard only (see SANITY_BELOW_ACC_PENALTY)
    },
    # ── Luscinia 2S Phase 3 preset ──────────────────────────────────────────
    # Hand accuracy raised to 35 % so TPE actively seeks configs that cover
    # real poetic vocabulary — closest proxy to production quality.
    "luscinia_specialist": {
        "f1":        0.45,   # primary: balanced val-set quality
        "acc":       0.20,   # secondary: per-form val accuracy
        "hand_acc":  0.35,   # ↑ boosted: Phase 3 targets edge-cases hard
        # sanity: zero weight — guard only
    },
    # ── Luscinia 3S Phase 3 preset ──────────────────────────────────────────
    # Same weights as luscinia_specialist; f1 is macro-F1 over 3 classes.
    "luscinia_3s_specialist": {
        "f1":        0.45,   # primary: macro-F1 across 3 stress positions
        "acc":       0.20,   # secondary: per-form val accuracy
        "hand_acc":  0.35,   # ↑ boosted: Phase 3 targets edge-cases hard
        # sanity: zero weight — guard only
    },
    # ── Luscinia Universal Phase 3 preset ───────────────────────────────────
    # Universal model covers ALL syllable counts (2–10+) with 11 classes.
    # f1 is macro-F1 across all 11 possible stress positions.
    # hand_acc is especially important because handcrafted words span ALL
    # syllable counts — closest proxy to production quality.
    # acc slightly reduced vs 3S: per-form accuracy is less meaningful
    # when mixing 2-syl and 8-syl words in the same metric.
    "luscinia_universal_specialist": {
        "f1":        0.45,   # primary: macro-F1 across all stress positions
        "acc":       0.20,   # secondary: per-form val accuracy
        "hand_acc":  0.35,   # ↑ boosted: spans all syllable counts
        # sanity: zero weight — guard only
    },
}


def compute_fitness(
    # positional args kept for backward-compat with v1.4 callers:
    f1: float,
    ext_acc: float,
    hand_acc: float,
    syl_acc_or_auc: float = 0.0,
    # keyword-only args for new specialist callers:
    *,
    auc: Optional[float] = None,
    acc: Optional[float] = None,
    hand_correct: int = 0,
    hand_total: int = 0,
    model_size_mb: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    preset: Optional[str] = None,
    penalty_mode: bool = False,
) -> Union[float, Tuple[float, bool, bool]]:
    """Flexible composite fitness score (Luscinia v2 redesign).

    Sanity accuracy as a guard — NOT a fitness component
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``ext_acc`` (sanity accuracy on an external DB sample) is evaluated as a
    data-quality guard rather than an optimisation component:

    * ``ext_acc >= acc``  →  no effect on score (normal case).
    * ``ext_acc <  acc``  →  score multiplied by ``SANITY_BELOW_ACC_PENALTY``
      (0.50 by default).  This situation indicates a data leak, label bug, or
      encoding mismatch and should never select the model.

    Fitness formula (Luscinia presets)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ::

        base_score  = f1 * w_f1  +  acc * w_acc  +  hand_acc * w_hand
        size_bonus  = SIZE_BONUS_MAX * (1 - min(1, size_mb / SIZE_BONUS_CAP_MB))
        raw_score   = base_score + size_bonus
        score       = raw_score * sanity_factor   (1.0 or SANITY_BELOW_ACC_PENALTY)

    Backward-compatible with v1.4 callers
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The original call signature::

        compute_fitness(f1_macro, ext_acc, hand_acc, mean_syl_acc) -> float

    still works unchanged.  The 4th positional arg is interpreted as
    ``syl_acc`` and the default ``global_v1_4`` weights apply.

    Parameters
    ----------
    f1:
        Binary F1 (specialist) or macro-F1 (global).
    ext_acc:
        Accuracy on the external DB sample — used ONLY as the sanity guard.
    hand_acc:
        Accuracy on handcrafted edge-case words.
    syl_acc_or_auc:
        4th positional arg for backward-compat (global_v1_4 only).
    auc:
        AUC-ROC; kept for backward-compat.
    acc:
        Val-set accuracy (per distinct grammatical form).
    hand_correct, hand_total:
        Used to compute the penalty flag when ``penalty_mode=True``.
    model_size_mb:
        Estimated model size in MB.  Smaller models receive a tiny bonus
        (up to ``SIZE_BONUS_MAX`` = 0.02) as a last-resort tiebreaker.
    weights:
        Explicit weight dict — takes priority over ``preset``.
    preset:
        One of the named presets.  Defaults to ``"global_v1_4"`` when neither
        ``weights`` nor ``preset`` is provided.
    penalty_mode:
        When ``True``, return ``(fitness, penalty_applied, sanity_violated)``
        where ``penalty_applied`` is ``True`` when any handcrafted word was
        missed, and ``sanity_violated`` is ``True`` when the sanity guard fired.
        The hand-penalty flag is *display-only* — it does not modify the score.
        The sanity guard IS applied to the numeric score regardless of this flag.

    Returns
    -------
    float
        Fitness score in ``[0, ~1.02]`` when ``penalty_mode=False``.
    Tuple[float, bool, bool]
        ``(fitness, hand_penalty_flag, sanity_violated)`` when ``penalty_mode=True``.
        ``sanity_violated`` is ``True`` when sanity_acc < val_acc and the score
        was halved by the sanity guard.
    """
    # ── Resolve weight preset ─────────────────────────────────────────────────
    if weights is not None:
        w = weights
    elif preset is not None:
        w = _WEIGHT_PRESETS[preset]
    else:
        w = _WEIGHT_PRESETS["global_v1_4"]

    # ── Component values ──────────────────────────────────────────────────────
    _auc = auc if auc is not None else syl_acc_or_auc
    _syl = syl_acc_or_auc if auc is None else syl_acc_or_auc
    _acc = acc if acc is not None else 0.0

    components: Dict[str, float] = {
        "f1":         f1,
        "ext_acc":    ext_acc,    # global_v1_4 key (backward-compat)
        "sanity_acc": ext_acc,    # alias (same value)
        "hand_acc":   hand_acc,
        "syl_acc":    _syl,
        "auc":        _auc,
        "acc":        _acc,
    }

    base_score = sum(w.get(k, 0.0) * v for k, v in components.items())

    # ── Model-size tiebreaker (tiny additive bonus) ───────────────────────────
    size_bonus = SIZE_BONUS_MAX * (1.0 - min(1.0, model_size_mb / SIZE_BONUS_CAP_MB))

    raw_score = base_score + size_bonus

    # ── Sanity guard — multiplicative penalty when sanity < val_acc ──────────
    # Only fires when:
    #   1. The caller explicitly provided acc= (acc is not None), AND
    #   2. ext_acc < acc (strict less-than; equal is fine).
    # Never fires for legacy global_v1_4 calls (which don't pass acc=).
    if acc is not None and _acc > 0.0 and ext_acc < _acc:
        sanity_factor = SANITY_BELOW_ACC_PENALTY
    else:
        sanity_factor = 1.0

    score = round(raw_score * sanity_factor, 6)

    if penalty_mode:
        hand_penalty = hand_total > 0 and hand_correct < hand_total
        sanity_violated = sanity_factor < 1.0
        return score, hand_penalty, sanity_violated
    return score
