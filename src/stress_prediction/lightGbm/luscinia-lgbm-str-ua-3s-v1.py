#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Luscinia 3S v1.0 — 3-Syllable Specialist
==========================================
Named after *Luscinia megarhynchos* (the Common Nightingale):
a specialist of extraordinary accuracy in its domain.

MOTIVATION
----------
Companion to ``luscinia-lgbm-str-ua-2s-v1.py``.  Luscinia 3S targets
words with exactly 3 vowels (3 syllables), where stress is a 3-class
classification problem rather than binary:

  * class 0 — antepenult  (proparoxytone, stress on 1st vowel)
  * class 1 — penult      (paroxytone,    stress on 2nd vowel)
  * class 2 — oxytone     (stress on 3rd / final vowel)

Key differences from the 2S script
-----------------------------------
1. **Multiclass objective** — ``objective=multiclass``, ``num_class=3``,
   ``metric=multi_logloss``.  Prediction uses argmax over the probability
   vector instead of a binary threshold.

2. **122-feature set** — ``build_features_3syl()`` adds 22 3-syl-specific
   features on top of the 100-feature base:
   • Richer ending hashes (mod=4096): captures -ання/-ення/-ити/-ати
   • Per-vowel identity (v1/v2/v3) with 3-syl-dedicated names
   • Coda lengths (chars after each vowel) — distance-from-end perspective
   • Inter-vowel interval ratios (dedicated 3-syl copies of iv_dist_0/1)
   • Linguistic rule flags: ви- prefix, gerund suffix, verb -ати/-ити/-увати,
     oxytone mobile suffix, adjective -ський, numeral -адцять

3. **Fitness preset** — ``luscinia_3s_specialist`` in Phase 3.  Same
   component weights as ``luscinia_specialist`` (f1=50%, hand_acc=25%)
   but ``f1`` here is macro-F1 over all 3 classes.

4. **Data filter** — exactly 3 vowels; no 2-syl or 4+-syl words.

5. **Model save threshold** — ``fitness > 0.82`` (3-class is harder than
   binary; the ceiling is lower than the 2S threshold of 0.87).

6. **Ensemble** — uses argmax prediction instead of binary threshold.

PHASES
------
Phase 1 (30% budget) — BROAD RANDOM EXPLORATION
  Wide random sampling to map the fitness landscape.  Checkpoint-based
  pruning via Hyperband: hopeless configs are killed early at checkpoints
  [50, 150, 300, 500, max_rounds].  Uses ``specialist_binary`` preset
  (compat with P2).

Phase 2 (25% budget) — CMA-ES FROM BEST SEEDS
  Continuous optimisation seeded from the top Phase 1 config.

Phase 3 (≈45% budget) — TPE BAYESIAN FINE-TUNING  [luscinia_3s_specialist]
  Multivariate TPE warm-started from the best Phase 1+2 config.  Narrow
  search space (±50% around winner).  Uses ``luscinia_3s_specialist`` preset:
  macro-F1 at 50%, hand_acc at 25%.

Phase 4 — ENSEMBLE
  Loads top-K saved models and combines via fitness-weighted soft vote
  (argmax of averaged probability vectors).

Final Refit — always executes after the winning model is determined.
  Retrains on 100% DB data + unambiguous 3-syl handcrafted words.

FITNESS PRESET — luscinia_3s_specialist
  f1        = 0.50   primary: macro-F1 over 3 stress positions
  acc       = 0.15   per-form val accuracy
  hand_acc  = 0.25   real poetic words — closest to prod
  sanity_acc= 0.10   external DB sample — catches encoding disasters

LINGUISTIC INSIGHTS (from syllable trend rules)
------------------------------------------------
The 22 extra features encode:
  • ви- prefix (NOUN/VERB) → class 0  [weight 0.98, rule NOUN-03]
  • -ання/-ення/-іння suffix → class 1  [deverbal gerunds, ~100%]
  • -ати → class 2 tendency  [infinitive oxytone, weight 0.75]
  • -ити → class 1 tendency  [paroxytone infinitives]
  • -увати/-ювати → class 1  [inchoative verbs]
  • -ак/-яр/-ач/-ун → class 2  [paradigm B, weight 0.85, rule NOUN-01]
  • -ський/-зький/-цький → class 0  [adj root-stable, rule ADJ-02]
  • -адцять → class 1  [numeral penult, weight 1.0, rule NUM-01]
  • Coda lengths encode distance-from-end: prefix noise is reduced because
    the model looks backwards from the stressed vowel, not forwards.

FILESYSTEM
----------
artifacts/luscinia-lgbm-str-ua-3s-v1/
  luscinia-lgbm-str-ua-3s-v1-results.json
  luscinia-lgbm-str-ua-3s-v1-results.csv
  leaderboard.txt
  optuna_study.db
  phase1_summary.json
  feature_importance/
  convergence/
  P4_ensemble/
  <trial_name>/
  <trial_name>_FINAL_FULLDATA/

USAGE
-----
  python luscinia-lgbm-str-ua-3s-v1.py                     # 24h full run
  python luscinia-lgbm-str-ua-3s-v1.py --budget-hours 8    # custom budget
  python luscinia-lgbm-str-ua-3s-v1.py --resume            # resume interrupted run
  python luscinia-lgbm-str-ua-3s-v1.py --phase1-only       # broad exploration only
  python luscinia-lgbm-str-ua-3s-v1.py --skip-ensemble     # skip P4 ensemble phase

REQUIREMENTS
------------
  lightgbm >= 4.0, optuna >= 3.0, scikit-learn >= 1.3
  numpy, pandas
"""

import csv
import gc
import json
import math
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score,
)

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import HyperbandPruner, PatientPruner, NopPruner

from services.constants import VOWEL_SET, EXPECTED_FEATURE_COUNT
from services.feature_service_3syl import build_features_3syl, EXPECTED_FEATURE_COUNT_3SYL
from services.data_service import (
    load_training_data, load_handcrafted_tests,
    stress_to_vowel_label, group_split,
)
from services.evaluation_service import (
    compute_fitness as _svc_compute_fitness,
)
from services.logging_service import (
    sp,
    log_trial_result,
    log_phase_progress,
    log_phase_summary,
    log_final_leaderboard,
    append_result_csv,
    log_training_summary,
)

try:
    from src.utils.normalize_apostrophe import normalize_apostrophe
except ImportError:
    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from src.utils.normalize_apostrophe import normalize_apostrophe

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Paths & constants
# ============================================================================
SCRIPT_NAME = "Luscinia-LGBM-STR-UA-3S-v1"
N_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_DB      = Path(__file__).parent.parent / "data" / "stress_training.db"
HANDCRAFTED_CSV = Path(__file__).parent.parent / "data" / "handcrafted_test_words.csv"
RESULTS_DIR     = Path(__file__).parent / "artifacts" / "luscinia-lgbm-str-ua-3s-v1"
RESULTS_CSV     = RESULTS_DIR / "luscinia-lgbm-str-ua-3s-v1-results.csv"
RESULTS_JSON    = RESULTS_DIR / "luscinia-lgbm-str-ua-3s-v1-results.json"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.txt"
OPTUNA_DB       = RESULTS_DIR / "optuna_study.db"
PHASE1_SUMMARY  = RESULTS_DIR / "phase1_summary.json"
CONVERGENCE_DIR = RESULTS_DIR / "convergence"
FEAT_IMP_DIR    = RESULTS_DIR / "feature_importance"
ENSEMBLE_DIR    = RESULTS_DIR / "P4_ensemble"
LOG_FILE        = RESULTS_DIR / "run.log"

NUM_CLASSES = 3                  # 3-way classification
EXTERNAL_SAMPLE_SIZE = 5000      # 5,000-word external sanity sample
EXTERNAL_SAMPLE_SEED = 42

# ── Phase budget fractions ───────────────────────────────────────────────────
PHASE1_FRACTION = 0.30
PHASE2_FRACTION = 0.25

# ── LightGBM base params ─────────────────────────────────────────────────────
BASE_LGBM = {
    "boosting_type":      "gbdt",
    "objective":          "multiclass",
    "num_class":          NUM_CLASSES,
    "metric":             "multi_logloss",
    "max_bin":            255,
    "num_threads":        N_WORKERS,
    "seed":               42,
    "verbosity":          -1,
    "force_col_wise":     True,
    "feature_pre_filter": False,
}

# ── Phase-specific max rounds ────────────────────────────────────────────────
MAX_ROUNDS_P1 = 800
MAX_ROUNDS_P2 = 1500
MAX_ROUNDS_P3 = 1200

# ── Phase 2 search bounds ────────────────────────────────────────────────────
P2_BOUNDS = {
    "num_leaves":              (31,    800),
    "max_depth":               (4,     14),
    "learning_rate":           (0.01,  0.30),
    "min_child_samples":       (5,     200),
    "lambda_l1":               (0.0,   2.0),
    "lambda_l2":               (0.0,   10.0),
    "subsample":               (0.4,   1.0),
    "colsample_bytree":        (0.2,   1.0),
    "feature_fraction_bynode": (0.3,   1.0),
    "min_sum_hessian_in_leaf": (1e-4,  30.0),
    "path_smooth":             (0.0,   1.0),
    "min_split_gain":          (0.0,   0.3),
}

# ── Ensemble settings ────────────────────────────────────────────────────────
ENSEMBLE_TOP_K = 5
ENSEMBLE_MIN_SANITY_GAIN = 0.001

# ── CSV schema ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "phase", "trial_number", "name", "fitness",
    "f1_macro", "accuracy",
    "sanity_accuracy", "sanity_correct", "sanity_sample_size",
    "hand_accuracy", "hand_correct", "hand_total",
    "hand_penalty_applied",
    "estimated_size_mb", "best_iteration", "num_trees",
    "train_time_sec", "wall_elapsed_sec", "wall_elapsed_min",
    "num_leaves", "max_depth", "learning_rate", "min_child_samples",
    "lambda_l1", "lambda_l2", "subsample", "colsample_bytree",
    "feature_fraction_bynode", "min_sum_hessian_in_leaf",
    "path_smooth", "min_split_gain",
    "data_sample_strategy",
    "boosting_rounds_at_convergence",
]


# ============================================================================
# Local wrappers around evaluation_service
# ============================================================================

def compute_fitness(
    f1: float,
    ext_acc: float,
    hand_acc: float,
    acc: float = 0.0,
    hand_correct: int = 0,
    hand_total: int = 0,
    *,
    model_size_mb: float = 0.0,
    preset: str = "luscinia_3s_specialist",
) -> Tuple[float, bool, bool]:
    """Luscinia 3S composite fitness (v2 redesign).

    Default preset: ``luscinia_3s_specialist``
      f1=45 % (macro-F1), acc=20 %, hand_acc=35 %  (no sanity weight)

    Returns ``(score, hand_penalty_flag, sanity_violated)``.
    """
    return _svc_compute_fitness(
        f1, ext_acc, hand_acc,
        acc=acc,
        hand_correct=hand_correct,
        hand_total=hand_total,
        model_size_mb=model_size_mb,
        preset=preset,
        penalty_mode=True,
    )


def _predict_one_3syl(booster, fvec: np.ndarray) -> Tuple[int, float]:
    """Argmax prediction for multiclass (3-class) booster.

    Returns (predicted_class, confidence).
    """
    raw = booster.predict(fvec)   # shape (1, 3) — probability vector
    proba = raw[0]
    predicted = int(np.argmax(proba))
    confidence = float(proba[predicted])
    return predicted, confidence


def evaluate_external(booster, feature_cols: list, external_sample: list) -> dict:
    """Evaluate the 3-syl booster on the external sanity sample."""
    correct = 0
    total = len(external_sample)
    for item in external_sample:
        feat = build_features_3syl(item["form"], item["pos"], item.get("features_json"))
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        predicted, _ = _predict_one_3syl(booster, x)
        if predicted == item["expected_label"]:
            correct += 1
    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(acc, 6),
        "correct": correct,
        "sample_size": total,
    }


def evaluate_handcrafted(booster, feature_cols: list, handcrafted_tests: list) -> dict:
    """Evaluate the 3-syl booster on the handcrafted word list."""
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

        feat = build_features_3syl(word, pos, features_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        predicted, confidence = _predict_one_3syl(booster, x)

        scoreable = expected is not None
        if scoreable:
            hit = (predicted in expected if isinstance(expected, list)
                   else predicted == expected)
            hand_total += 1
            if hit:
                hand_correct += 1
        else:
            hit = None

        hand_results.append({
            "word": word, "pos": pos,
            "expected": expected, "predicted": predicted,
            "correct": hit, "scoreable": scoreable,
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


# ============================================================================
# Data loading
# ============================================================================

class ThreeSylChunkProcessor:
    """Multiprocessing-safe chunk processor for 3-syllable feature extraction."""

    def __call__(self, chunk: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in chunk.iterrows():
            form = row["form"] or ""
            pos = row.get("pos") or ""
            features_json = row.get("features_json") or None

            lower = normalize_apostrophe(form).lower()
            vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
            if len(vowels) != 3:
                continue

            try:
                stress_raw = json.loads(row["stress_indices"] or "[]")
            except Exception:
                stress_raw = []
            label = stress_to_vowel_label(stress_raw, vowels)
            if label < 0 or label > 2:
                continue

            rec = build_features_3syl(form, pos, features_json)
            rec["__label__"] = label
            rec["__lemma__"] = row.get("lemma", "")
            records.append(rec)
        return pd.DataFrame(records)


def build_3syl_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the 3-syl feature matrix from a raw training DataFrame.

    Filters:
    * only rows with exactly 3 vowels
    * removes free_variant / grammatical_homonym
    * removes words where a single (form, pos, features_json) triple has
      conflicting stress annotations

    Returns (X, y, lemmas) where X has EXPECTED_FEATURE_COUNT_3SYL columns.
    """
    before = len(df)
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()
    df["stress_count"] = (df.groupby(["form", "pos", "features_json"])
                          ["stress_indices"].transform("nunique"))
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"])
    df = df.reset_index(drop=True)

    chunk_size = max(1, len(df) // N_WORKERS)
    chunks = [df.iloc[i: i + chunk_size]
              for i in range(0, len(df), chunk_size)]

    processor = ThreeSylChunkProcessor()
    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(processor, chunks)

    combined = pd.concat(results, ignore_index=True)
    y = combined["__label__"].astype(int)
    lemmas = combined["__lemma__"]
    X = combined.drop(columns=["__label__", "__lemma__"])
    return X, y, lemmas


def load_3syl_external(db_path: Path, size: int, seed: int) -> List[dict]:
    """Load a stratified external validation sample filtered to 3-vowel words.

    Returns up to ``size`` items.  Items with exactly 3 vowels and a valid
    stress label (0, 1, or 2) are kept; all others are discarded.
    """
    import sqlite3
    import random as _random

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT form, lemma, stress_indices, pos, features_json "
        "FROM training_entries "
        "WHERE variant_type NOT IN ('free_variant', 'grammatical_homonym')"
    )
    rows = cur.fetchall()
    conn.close()

    seen = set()
    pool: List[dict] = []
    for form, lemma, stress_raw_str, pos, features_json in rows:
        if not form or not stress_raw_str:
            continue
        lower = normalize_apostrophe(form).lower()
        key = (lower, pos or "")
        if key in seen:
            continue
        seen.add(key)

        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) != 3:
            continue
        try:
            stress_raw = json.loads(stress_raw_str)
        except Exception:
            continue
        if not stress_raw:
            continue
        vi = stress_raw[0]
        if vi < 0 or vi > 2:
            continue

        pool.append({
            "form": form, "pos": pos or "X",
            "expected_label": vi, "vowels": vowels, "lemma": lemma,
            "features_json": features_json, "n_syllables": 3,
        })

    rng = _random.Random(seed)
    rng.shuffle(pool)
    return pool[:size]


def load_3syl_handcrafted(csv_path: Path) -> list:
    """Load handcrafted test words from the shared CSV, keeping only entries
    with exactly 3 vowels and a valid stress index (0, 1, or 2).

    The shared ``handcrafted_test_words.csv`` contains words of all syllable
    counts.  This function:

    1. Loads the full table via :func:`load_handcrafted_tests` (handles
       apostrophe normalisation, comment lines, ambiguous ``"1,2"`` indices).
    2. Discards words whose normalised lowercase form does not have exactly
       3 vowels.
    3. Discards entries whose ``expected_vowel_index`` is out of range for
       a 3-class model (i.e. any index > 2), which would happen if a
       4+-syllable word were accidentally labelled 0–2 and slipped past step 2.
    """
    raw = load_handcrafted_tests(csv_path)
    filtered = []
    for entry in raw:
        word = entry[0]
        lower = normalize_apostrophe(word).lower()
        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) != 3:
            continue
        expected = entry[2]  # int | list[int] | None
        if expected is not None:
            indices = expected if isinstance(expected, list) else [expected]
            if any(idx > 2 for idx in indices):
                continue
        filtered.append(entry)
    return filtered


# ============================================================================
# JSON persistence  (CSV now handled by logging_service.append_result_csv)
# ============================================================================

def _append_json(result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_list: list = []
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
                results_list = json.load(fh)
        except Exception:
            results_list = []
    results_list.append(result)
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(results_list, fh, indent=2, ensure_ascii=False, default=str)


# ============================================================================
# Train and evaluate one trial
# ============================================================================

def train_and_evaluate(
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
    wall_start: float,
    max_rounds: int,
    early_stopping_rounds: int,
    trial: Optional[optuna.Trial],
    trial_name: str,
    phase: str,
    trial_number: int,
    fitness_preset: str,
    save_model: bool = False,
) -> Optional[dict]:
    """Train one LightGBM config and return a result dict.

    Uses checkpoint-based incremental training: the model is trained in
    segments and evaluated at each checkpoint.  Intermediate macro-F1 is
    reported to the Optuna ``trial`` so the Hyperband pruner can kill
    clearly hopeless configurations early, saving significant wall-time.

    Multiclass note: LightGBM with objective=multiclass auto-builds
    ``num_class`` trees per boosting round, so the effective number of
    trees = ``best_iteration * num_class``.

    Returns ``None`` if the trial is pruned or training fails.
    """
    # ── Checkpoint schedule (per phase) ───────────────────────────────────────
    checkpoints_map = {
        "P1": [50, 150, 300, 500,            max_rounds],
        "P2": [100, 300, 600, 1000,          max_rounds],
        "P3": [50, 150, 300, 600, 900,       max_rounds],
    }
    checkpoints = sorted(
        {c for c in checkpoints_map.get(phase, [max_rounds]) if c <= max_rounds}
    )
    if not checkpoints or checkpoints[-1] != max_rounds:
        checkpoints.append(max_rounds)

    full_params = {**BASE_LGBM, **params}
    full_params["num_class"] = NUM_CLASSES   # enforce 3 always

    t0 = time.perf_counter()
    train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_ds   = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False,
                           reference=train_ds)

    booster = None
    prev_cp = 0
    convergence_curve = []

    try:
        for cp in checkpoints:
            additional = cp - prev_cp
            if additional <= 0:
                continue
            callbacks = [
                lgb.log_evaluation(0),
                lgb.early_stopping(early_stopping_rounds, verbose=False),
            ]
            if booster is None:
                booster = lgb.train(full_params, train_ds,
                                    num_boost_round=cp,
                                    valid_sets=[val_ds], callbacks=callbacks)
            else:
                booster = lgb.train(full_params, train_ds,
                                    num_boost_round=additional,
                                    valid_sets=[val_ds], callbacks=callbacks,
                                    init_model=booster)
            prev_cp = cp

            # Evaluate macro-F1 at this checkpoint
            raw_cp = booster.predict(X_val.values.astype(np.float32))
            y_cp   = np.argmax(raw_cp, axis=1)
            cp_f1  = float(f1_score(y_val, y_cp, average="macro", zero_division=0))
            cp_acc = float(accuracy_score(y_val, y_cp))
            bs = booster.best_score.get("valid_0", {})
            best_logloss = bs.get("multi_logloss", None)
            convergence_curve.append({
                "cp":      cp,
                "f1":      round(cp_f1, 4),
                "acc":     round(cp_acc, 4),
                "logloss": round(best_logloss, 6) if best_logloss else None,
            })
            sp(f"       cp={cp:<5}  F1={cp_f1*100:.2f}%  acc={cp_acc*100:.2f}%"
               + (f"  loss={best_logloss:.5f}" if best_logloss else ""), flush=True)

            # Report to Optuna pruner
            if trial is not None:
                trial.report(cp_f1, step=cp)
                if trial.should_prune():
                    sp(f"       PRUNED at cp={cp} (F1={cp_f1*100:.2f}%)")
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
        raise
    except Exception as e:
        sp(f"   FAILED: {e}")
        import traceback; traceback.print_exc()
        return None

    train_time = time.perf_counter() - t0
    best_iter = booster.best_iteration or max_rounds

    # ── Internal val metrics ──────────────────────────────────────────────────
    raw_pred = booster.predict(X_val.values.astype(np.float32))   # (n, 3)
    y_pred = np.argmax(raw_pred, axis=1)
    val_acc = float(accuracy_score(y_val, y_pred))
    val_f1  = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    try:
        val_auc = float(roc_auc_score(
            y_val, raw_pred, multi_class="ovr", average="macro"
        ))
    except Exception:
        val_auc = 0.0

    # ── External sanity evaluation ────────────────────────────────────────────
    ext_res  = evaluate_external(booster, feature_cols, external_sample)
    hand_res = evaluate_handcrafted(booster, feature_cols, handcrafted_tests)

    # ── Model size estimate ───────────────────────────────────────────────────
    num_trees  = best_iter * NUM_CLASSES
    size_bytes = num_trees * 1000
    size_mb    = size_bytes / 1e6

    fitness, penalty, sanity_bad = compute_fitness(
        val_f1,
        ext_res["accuracy"],
        hand_res["accuracy"],
        acc=val_acc,
        hand_correct=hand_res["correct"],
        hand_total=hand_res["total"],
        model_size_mb=size_mb,
        preset=fitness_preset,
    )
    if sanity_bad:
        sp("\n  ⚠⚠⚠  SANITY BELOW VAL_ACC — sanity_acc={:.4f} < acc={:.4f}"
           " — possible data leak or label bug!  Score halved.".format(
               ext_res["accuracy"], val_acc))

    # ── Convergence plot ──────────────────────────────────────────────────────
    try:
        CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
        conv_path = CONVERGENCE_DIR / f"{trial_name}_curve.json"
        with open(conv_path, "w", encoding="utf-8") as fh:
            json.dump({"checkpoints": convergence_curve}, fh)
    except Exception:
        pass

    # ── Feature importance ────────────────────────────────────────────────────
    try:
        imp = booster.feature_importance(importance_type="gain")
        imp_pairs = sorted(zip(feature_cols, imp),
                           key=lambda x: x[1], reverse=True)[:20]
        FEAT_IMP_DIR.mkdir(parents=True, exist_ok=True)
        imp_path = FEAT_IMP_DIR / f"{trial_name}_top20.json"
        with open(imp_path, "w", encoding="utf-8") as fh:
            json.dump([{"feature": k, "gain": float(v)} for k, v in imp_pairs],
                      fh, indent=2)
    except Exception:
        pass

    # ── Save model if threshold reached ──────────────────────────────────────
    model_saved = False
    if save_model or fitness > 0.82:
        model_dir = RESULTS_DIR / trial_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{trial_name}.lgb"
        booster.save_model(str(model_path))
        model_saved = True

    wall_elapsed = time.perf_counter() - wall_start
    result = {
        "phase":             phase,
        "trial_number":      trial_number,
        "name":              trial_name,
        "timestamp":         datetime.now().isoformat(),
        "params":            params,
        "fitness_preset":    fitness_preset,
        "train_time_sec":    round(train_time, 2),
        "wall_elapsed_sec":  round(wall_elapsed, 2),
        "wall_elapsed_min":  round(wall_elapsed / 60, 2),
        "internal": {
            "f1":            round(val_f1, 6),
            "accuracy":      round(val_acc, 6),
            "auc":           round(val_auc, 6),
            "best_iteration": best_iter,
            "num_trees":     num_trees,
        },
        "external": {
            "accuracy":      ext_res["accuracy"],
            "correct":       ext_res["correct"],
            "sample_size":   ext_res["sample_size"],
        },
        "handcrafted": {
            "accuracy":      hand_res["accuracy"],
            "correct":       hand_res["correct"],
            "total":         hand_res["total"],
            "total_words":   hand_res["total_words"],
            "results":       hand_res["results"],
        },
        "model": {"size_bytes": size_bytes, "saved": model_saved},
        "fitness":             round(fitness, 6),
        "hand_penalty_applied": penalty,
        "sanity_violated":     sanity_bad,
        "boosting_rounds_at_convergence": best_iter,
    }

    append_result_csv(result, RESULTS_CSV, CSV_FIELDS)
    _append_json({k: v for k, v in result.items() if k != "handcrafted"})

    del booster, train_ds, val_ds
    gc.collect()
    return result


# ============================================================================
# Phase objectives
# ============================================================================

class Phase1Objective:
    """Wide random exploration — RandomSampler, ``specialist_binary`` preset."""

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget,
                 phase_budget: float = 0.0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.feature_cols = feature_cols
        self.external_sample = external_sample
        self.handcrafted_tests = handcrafted_tests
        self.wall_start = wall_start
        self.budget = budget
        self.phase_budget = phase_budget
        self.all_results: list = []
        self._trial_counter = 0

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget:
            trial.study.stop()
            return 0.0

        self._trial_counter += 1
        trial_name = f"P1_{self._trial_counter:04d}"

        num_leaves        = trial.suggest_int("num_leaves", 31, 511)
        max_depth         = trial.suggest_int("max_depth", 4, 14)
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.30, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 200, log=True)
        lambda_l1         = trial.suggest_float("lambda_l1", 0.0, 2.0)
        lambda_l2         = trial.suggest_float("lambda_l2", 0.0, 10.0)
        subsample         = trial.suggest_float("subsample", 0.4, 1.0)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        ffbn              = trial.suggest_float("feature_fraction_bynode", 0.3, 1.0)
        min_sum_h         = trial.suggest_float("min_sum_hessian_in_leaf", 1e-4, 30.0, log=True)
        path_smooth       = trial.suggest_float("path_smooth", 0.0, 1.0)
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.3)

        sp(f"\n   [P1 #{self._trial_counter}]  "
           f"leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={max_depth}")

        params = {
            "num_leaves": num_leaves, "max_depth": max_depth,
            "learning_rate": learning_rate, "min_child_samples": min_child_samples,
            "lambda_l1": lambda_l1, "lambda_l2": lambda_l2,
            "subsample": subsample, "colsample_bytree": colsample_bytree,
            "subsample_freq": 1,
            "feature_fraction_bynode": ffbn,
            "min_sum_hessian_in_leaf": min_sum_h,
            "path_smooth": path_smooth, "min_split_gain": min_split_gain,
            "is_unbalance": True,
            "data_sample_strategy": "bagging",
        }

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            self.wall_start, MAX_ROUNDS_P1,
            early_stopping_rounds=60,
            trial=trial,
            trial_name=trial_name,
            phase="P1", trial_number=self._trial_counter,
            fitness_preset="specialist_binary",
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P1", self.all_results, time.perf_counter() - self.wall_start, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        sys.stdout.flush()
        return result["fitness"]


class Phase2Objective:
    """CMA-ES from Phase 1 best seed — ``specialist_binary`` preset."""

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget,
                 seed_params: dict, phase_budget: float = 0.0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.feature_cols = feature_cols
        self.external_sample = external_sample
        self.handcrafted_tests = handcrafted_tests
        self.wall_start = wall_start
        self.budget = budget
        self.seed_params = seed_params
        self.phase_budget = phase_budget
        self.all_results: list = []
        self._trial_counter = 0

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget:
            trial.study.stop()
            return 0.0

        self._trial_counter += 1
        trial_name = f"P2_{self._trial_counter:04d}"

        num_leaves        = trial.suggest_int("num_leaves", P2_BOUNDS["num_leaves"][0], P2_BOUNDS["num_leaves"][1])
        max_depth         = trial.suggest_int("max_depth", P2_BOUNDS["max_depth"][0], P2_BOUNDS["max_depth"][1])
        learning_rate     = trial.suggest_float("learning_rate", P2_BOUNDS["learning_rate"][0], P2_BOUNDS["learning_rate"][1], log=True)
        min_child_samples = trial.suggest_int("min_child_samples", P2_BOUNDS["min_child_samples"][0], P2_BOUNDS["min_child_samples"][1], log=True)
        lambda_l1         = trial.suggest_float("lambda_l1", P2_BOUNDS["lambda_l1"][0], P2_BOUNDS["lambda_l1"][1])
        lambda_l2         = trial.suggest_float("lambda_l2", P2_BOUNDS["lambda_l2"][0], P2_BOUNDS["lambda_l2"][1])
        subsample         = trial.suggest_float("subsample", P2_BOUNDS["subsample"][0], P2_BOUNDS["subsample"][1])
        colsample_bytree  = trial.suggest_float("colsample_bytree", P2_BOUNDS["colsample_bytree"][0], P2_BOUNDS["colsample_bytree"][1])
        ffbn              = trial.suggest_float("feature_fraction_bynode", P2_BOUNDS["feature_fraction_bynode"][0], P2_BOUNDS["feature_fraction_bynode"][1])
        min_sum_h         = trial.suggest_float("min_sum_hessian_in_leaf", P2_BOUNDS["min_sum_hessian_in_leaf"][0], P2_BOUNDS["min_sum_hessian_in_leaf"][1], log=True)
        path_smooth       = trial.suggest_float("path_smooth", P2_BOUNDS["path_smooth"][0], P2_BOUNDS["path_smooth"][1])
        min_split_gain    = trial.suggest_float("min_split_gain", P2_BOUNDS["min_split_gain"][0], P2_BOUNDS["min_split_gain"][1])

        sp(f"\n   [P2 #{self._trial_counter}]  "
           f"leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={max_depth}")

        params = {
            "num_leaves": num_leaves, "max_depth": max_depth,
            "learning_rate": learning_rate, "min_child_samples": min_child_samples,
            "lambda_l1": lambda_l1, "lambda_l2": lambda_l2,
            "subsample": subsample, "colsample_bytree": colsample_bytree,
            "subsample_freq": 1,
            "feature_fraction_bynode": ffbn,
            "min_sum_hessian_in_leaf": min_sum_h,
            "path_smooth": path_smooth, "min_split_gain": min_split_gain,
            "is_unbalance": True,
            "data_sample_strategy": "bagging",
        }

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            self.wall_start, MAX_ROUNDS_P2,
            early_stopping_rounds=80,
            trial=trial,
            trial_name=trial_name,
            phase="P2", trial_number=self._trial_counter,
            fitness_preset="specialist_binary",
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P2", self.all_results, time.perf_counter() - self.wall_start, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        sys.stdout.flush()
        return result["fitness"]


class Phase3Objective:
    """Multivariate TPE fine-tuning — ``luscinia_3s_specialist`` preset.

    Searches within ±50% of the best combined Phase 1+2 params.
    """

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget,
                 best_params: dict, phase_budget: float = 0.0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.feature_cols = feature_cols
        self.external_sample = external_sample
        self.handcrafted_tests = handcrafted_tests
        self.wall_start = wall_start
        self.budget = budget
        self.best_params = best_params
        self.phase_budget = phase_budget
        self.all_results: list = []
        self._trial_counter = 0

    def _narrow(self, key: str, lo: float, hi: float) -> Tuple[float, float]:
        """Return ±50% bounds around the seed value, clamped to [lo, hi]."""
        seed = float(self.best_params.get(key, (lo + hi) / 2))
        w = max(seed * 0.5, (hi - lo) * 0.1)
        return max(lo, seed - w), min(hi, seed + w)

    def _narrow_int(self, key: str, lo: int, hi: int) -> Tuple[int, int]:
        seed = int(self.best_params.get(key, (lo + hi) // 2))
        w = max(int(seed * 0.5), 10, 1)
        return max(lo, seed - w), min(hi, seed + w)

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget:
            trial.study.stop()
            return 0.0

        self._trial_counter += 1
        trial_name = f"P3_{self._trial_counter:04d}"

        nl_lo, nl_hi = self._narrow_int("num_leaves", 31, 800)
        md_lo, md_hi = self._narrow_int("max_depth", 4, 14)
        lr_lo, lr_hi = self._narrow("learning_rate", 0.01, 0.30)
        mc_lo, mc_hi = self._narrow_int("min_child_samples", 5, 200)
        l1_lo, l1_hi = self._narrow("lambda_l1", 0.0, 2.0)
        l2_lo, l2_hi = self._narrow("lambda_l2", 0.0, 10.0)
        ss_lo, ss_hi = self._narrow("subsample", 0.4, 1.0)
        cs_lo, cs_hi = self._narrow("colsample_bytree", 0.2, 1.0)
        ff_lo, ff_hi = self._narrow("feature_fraction_bynode", 0.3, 1.0)
        mh_lo, mh_hi = self._narrow("min_sum_hessian_in_leaf", 1e-4, 30.0)
        ps_lo, ps_hi = self._narrow("path_smooth", 0.0, 1.0)
        mg_lo, mg_hi = self._narrow("min_split_gain", 0.0, 0.3)

        num_leaves        = trial.suggest_int("num_leaves", nl_lo, nl_hi)
        max_depth         = trial.suggest_int("max_depth", md_lo, md_hi)
        learning_rate     = trial.suggest_float("learning_rate", max(lr_lo, 0.001), lr_hi, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", mc_lo, mc_hi, log=True)
        lambda_l1         = trial.suggest_float("lambda_l1", l1_lo, l1_hi)
        lambda_l2         = trial.suggest_float("lambda_l2", l2_lo, l2_hi)
        subsample         = trial.suggest_float("subsample", ss_lo, ss_hi)
        colsample_bytree  = trial.suggest_float("colsample_bytree", cs_lo, cs_hi)
        ffbn              = trial.suggest_float("feature_fraction_bynode", ff_lo, ff_hi)
        min_sum_h         = trial.suggest_float("min_sum_hessian_in_leaf", mh_lo, mh_hi, log=True)
        path_smooth       = trial.suggest_float("path_smooth", ps_lo, ps_hi)
        min_split_gain    = trial.suggest_float("min_split_gain", mg_lo, mg_hi)

        sp(f"\n   [P3 #{self._trial_counter}]  "
           f"leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={max_depth}")

        params = {
            "num_leaves": num_leaves, "max_depth": max_depth,
            "learning_rate": learning_rate, "min_child_samples": min_child_samples,
            "lambda_l1": lambda_l1, "lambda_l2": lambda_l2,
            "subsample": subsample, "colsample_bytree": colsample_bytree,
            "subsample_freq": 1,
            "feature_fraction_bynode": ffbn,
            "min_sum_hessian_in_leaf": min_sum_h,
            "path_smooth": path_smooth, "min_split_gain": min_split_gain,
            "is_unbalance": True,
            "data_sample_strategy": "bagging",
        }

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            self.wall_start, MAX_ROUNDS_P3,
            early_stopping_rounds=70,
            trial=trial,
            trial_name=trial_name,
            phase="P3", trial_number=self._trial_counter,
            fitness_preset="luscinia_3s_specialist",
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P3", self.all_results, time.perf_counter() - self.wall_start, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        sys.stdout.flush()
        return result["fitness"]


# ============================================================================
# CMA-ES helper
# ============================================================================

def _clip_to_p2_bounds(x0: dict) -> dict:
    """Clip a param dict to P2_BOUNDS (ensures CMA-ES x0 is in feasible region)."""
    clipped = {}
    for k, v in x0.items():
        if k in P2_BOUNDS:
            lo, hi = P2_BOUNDS[k]
            clipped[k] = max(lo, min(hi, v))
        else:
            clipped[k] = v
    return clipped


# ============================================================================
# Phase 1 landscape analysis
# ============================================================================

def analyze_phase1_landscape(results: list) -> dict:
    if not results:
        return {}

    by_fit = sorted(results, key=lambda r: r["fitness"], reverse=True)
    top10 = by_fit[:10]
    top10_params = [r["params"] for r in top10]

    param_analysis = {}
    for key in ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                "lambda_l1", "lambda_l2", "colsample_bytree"]:
        vals = [p.get(key) for p in top10_params if p.get(key) is not None]
        if vals:
            param_analysis[key] = {
                "min": min(vals), "max": max(vals),
                "mean": sum(vals) / len(vals),
            }

    analysis = {
        "total_trials": len(results),
        "best_fitness": by_fit[0]["fitness"] if by_fit else 0.0,
        "best_sanity": by_fit[0]["external"]["accuracy"] if by_fit else 0.0,
        "top10_params": param_analysis,
        "generated": datetime.now().isoformat(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE1_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)
    sp(f"  Phase 1 summary → {PHASE1_SUMMARY}")
    return analysis


# ============================================================================
# Phase summary printer
# ============================================================================

def print_phase_summary(phase: str, results: list) -> None:
    """Print a concise summary for a completed phase."""
    log_phase_summary(phase, results, LOG_FILE)


# ============================================================================
# Final leaderboard
# ============================================================================

def print_final_leaderboard(all_results: list, elapsed: float) -> None:
    """Print the top-15 leaderboard and save to ``leaderboard.txt``."""
    log_final_leaderboard(
        all_results, elapsed, SCRIPT_NAME, LEADERBOARD_FILE, LOG_FILE
    )


# ============================================================================
# Handcrafted → training rows
# ============================================================================

def _handcrafted_to_training_rows(
    handcrafted_tests: list,
    feature_cols: list,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Convert unambiguous handcrafted entries into feature-matrix rows.

    Entries with ``expected=None`` (unknown) or a list (ambiguous) are
    skipped.  Only scalar labels in {0, 1, 2} are accepted.
    """
    rows_X, rows_y = [], []
    for entry in handcrafted_tests:
        if len(entry) >= 5:
            word, pos, expected, _desc, features_json = entry[:5]
        elif len(entry) == 4:
            word, pos, expected, _desc = entry
            features_json = None
        else:
            word, pos, expected = entry[:3]
            features_json = None

        if expected is None or isinstance(expected, list):
            continue
        label = int(expected)
        if label not in (0, 1, 2):
            continue

        feat = build_features_3syl(word, pos, features_json)
        rows_X.append({c: feat.get(c, 0) for c in feature_cols})
        rows_y.append(label)

    if not rows_X:
        return None, None
    return pd.DataFrame(rows_X, columns=feature_cols), pd.Series(rows_y)


# ============================================================================
# Phase 4: Ensemble
# ============================================================================

def run_ensemble_phase(
    all_results: list,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
    top_k: int = ENSEMBLE_TOP_K,
    min_sanity_gain: float = ENSEMBLE_MIN_SANITY_GAIN,
) -> Optional[dict]:
    """Combine top-K saved models via fitness-weighted soft vote (argmax).

    Strategy
    --------
    1. Select the K saved models with the highest ``luscinia_3s_specialist``
       fitness from Phase 3 (or Phase 2/1 if fewer P3 models saved).
    2. Weighted average of predicted probability vectors; weights = softmax
       of fitness scores.
    3. Predicted class = argmax of the weighted average probability vector.
    4. **Accept** only if ensemble strictly outperforms best solo on:
         - sanity_acc improvement ≥ ``min_sanity_gain``  OR
         - hand_correct > best solo hand_correct

    Returns
    -------
    dict or None
    """
    # ── 1. Collect saved models ───────────────────────────────────────────────
    p3_results    = [r for r in all_results if r.get("phase") == "P3"]
    other_results = [r for r in all_results if r.get("phase") != "P3"]
    candidates    = (sorted(p3_results,    key=lambda r: r["fitness"], reverse=True)
                   + sorted(other_results, key=lambda r: r["fitness"], reverse=True))

    saved_candidates = []
    for r in candidates:
        model_path = RESULTS_DIR / r["name"] / f"{r['name']}.lgb"
        if model_path.exists():
            saved_candidates.append((r, model_path))
        if len(saved_candidates) >= top_k:
            break

    if len(saved_candidates) < 2:
        sp(f"\n[P4 Ensemble] Only {len(saved_candidates)} saved model(s) found "
           f"(need ≥ 2 for ensemble).  Skipping.")
        return None

    sp(f"\n{'=' * 80}")
    sp(f"PHASE 4 — ENSEMBLE  ({len(saved_candidates)} models, 3-class argmax)")
    sp(f"  Strategy: fitness-weighted soft vote → argmax")
    sp(f"  Accept criterion: sanity_acc gain ≥ {min_sanity_gain:.3f}  OR  more hand_correct")
    sp(f"{'=' * 80}")

    # ── 2. Load boosters and compute softmax weights ──────────────────────────
    boosters = []
    fitnesses = []
    for result, model_path in saved_candidates:
        try:
            bst = lgb.Booster(model_file=str(model_path))
            boosters.append(bst)
            fitnesses.append(result["fitness"])
            sp(f"  Loaded {result['name']:12s}  fit={result['fitness']:.4f}  "
               f"sanity={result['external']['accuracy']*100:.2f}%  "
               f"hand={result['handcrafted']['correct']}/{result['handcrafted']['total']}")
        except Exception as e:
            sp(f"  WARNING: could not load {model_path}: {e}")

    if len(boosters) < 2:
        sp("  Not enough loadable models for ensemble.  Skipping.")
        return None

    fit_arr = np.array(fitnesses, dtype=float)
    fit_arr -= fit_arr.max()
    weights  = np.exp(fit_arr)
    weights /= weights.sum()
    sp(f"\n  Model weights (softmax of fitness):")
    for (r, _), w in zip(saved_candidates[:len(boosters)], weights):
        sp(f"    {r['name']:12s}  weight={w:.4f}")

    # ── 3. External sanity evaluation ─────────────────────────────────────────
    sp(f"\n  Evaluating ensemble on {len(external_sample)}-word sanity sample...")
    sanity_correct = 0
    for item in external_sample:
        feat = build_features_3syl(item["form"], item["pos"], item.get("features_json"))
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        # Weighted average of probability vectors
        avg_proba = sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights))
        predicted = int(np.argmax(avg_proba))
        if predicted == item["expected_label"]:
            sanity_correct += 1
    sanity_acc = sanity_correct / len(external_sample) if external_sample else 0.0

    # ── 4. Handcrafted evaluation ─────────────────────────────────────────────
    sp(f"  Evaluating ensemble on {len(handcrafted_tests)} handcrafted words...")
    hand_correct = 0
    hand_total   = 0
    hand_detail  = []
    for entry in handcrafted_tests:
        word      = entry[0]
        pos       = entry[1] if len(entry) > 1 else "X"
        expected  = entry[2] if len(entry) > 2 else None
        desc      = entry[3] if len(entry) > 3 else ""
        feat_json = entry[4] if len(entry) > 4 else None
        if expected is None:
            hand_detail.append({"word": word, "predicted": None,
                                 "correct": None, "scoreable": False})
            continue
        hand_total += 1
        feat = build_features_3syl(word, pos, feat_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        avg_proba = sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights))
        predicted = int(np.argmax(avg_proba))
        expected_set = set(expected) if isinstance(expected, list) else {int(expected)}
        is_correct = (predicted in expected_set)
        if is_correct:
            hand_correct += 1
        hand_detail.append({
            "word": word, "predicted": predicted,
            "correct": is_correct, "scoreable": True,
            "confidence": round(float(avg_proba[predicted]), 4),
        })
    hand_acc = hand_correct / hand_total if hand_total > 0 else 0.0

    sp(f"\n  Ensemble results:")
    sp(f"    sanity_acc:  {sanity_acc*100:.2f}%  ({sanity_correct}/{len(external_sample)})")
    sp(f"    hand:        {hand_correct}/{hand_total} ({hand_acc*100:.2f}%)")

    # ── 5. Compare against best solo ──────────────────────────────────────────
    best_solo        = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
    best_solo_sanity = best_solo["external"]["accuracy"]
    best_solo_hand   = best_solo["handcrafted"]["correct"]

    sanity_gain = sanity_acc - best_solo_sanity
    hand_gain   = hand_correct - best_solo_hand

    sp(f"\n  vs best solo ({best_solo['name']}):")
    sp(f"    sanity gain: {sanity_gain:+.4f}  (threshold: {min_sanity_gain:.3f})")
    sp(f"    hand gain:   {hand_gain:+d}")

    accepted = (sanity_gain >= min_sanity_gain) or (hand_gain > 0)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    if not accepted:
        sp(f"\n  [P4] REJECTED — ensemble does not improve over best solo.")
        sp(f"       Carrying forward {best_solo['name']} (fit={best_solo['fitness']:.4f})")
        report = {
            "decision":         "rejected",
            "reason":           "no improvement over best solo",
            "best_solo":        best_solo["name"],
            "best_solo_sanity": round(best_solo_sanity, 6),
            "best_solo_hand":   best_solo_hand,
            "ensemble_sanity":  round(sanity_acc, 6),
            "ensemble_hand":    hand_correct,
            "sanity_gain":      round(sanity_gain, 6),
            "hand_gain":        hand_gain,
            "models_used":      [r["name"] for r, _ in saved_candidates[:len(boosters)]],
            "weights":          weights.tolist(),
            "generated":        datetime.now().isoformat(),
        }
        with open(ENSEMBLE_DIR / "ensemble_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        sp(f"  Report → {ENSEMBLE_DIR / 'ensemble_report.json'}")
        return None

    sp(f"\n  [P4] ACCEPTED — ensemble is better!  "
       f"(+{sanity_gain:.4f} sanity, {hand_gain:+d} hand)")

    ensemble_fitness, ensemble_penalty, ensemble_sanity_bad = _svc_compute_fitness(
        best_solo["internal"]["f1"],
        sanity_acc,
        hand_acc,
        acc=best_solo["internal"]["accuracy"],
        hand_correct=hand_correct,
        hand_total=hand_total,
        model_size_mb=0.0,
        preset="luscinia_3s_specialist",
        penalty_mode=True,
    )
    if ensemble_sanity_bad:
        sp("  ⚠⚠⚠  ENSEMBLE SANITY BELOW ACC — sanity_acc={:.4f} < acc={:.4f}"
           " — data quality issue suspected!".format(
               sanity_acc, best_solo["internal"]["accuracy"]))

    ensemble_result = {
        "phase":              "P4",
        "trial_number":       -1,
        "name":               "P4_ensemble",
        "timestamp":          datetime.now().isoformat(),
        "params":             best_solo["params"],
        "fitness_preset":     "luscinia_3s_specialist",
        "train_time_sec":     0,
        "wall_elapsed_sec":   0,
        "wall_elapsed_min":   0,
        "internal":           best_solo["internal"],
        "external": {
            "accuracy":       round(sanity_acc, 6),
            "correct":        sanity_correct,
            "sample_size":    len(external_sample),
        },
        "handcrafted": {
            "accuracy":       round(hand_acc, 6),
            "correct":        hand_correct,
            "total":          hand_total,
            "total_words":    len(handcrafted_tests),
            "results":        hand_detail,
        },
        "model":              best_solo["model"],
        "fitness":            ensemble_fitness,
        "hand_penalty_applied": ensemble_penalty,
        "ensemble_models":    [r["name"] for r, _ in saved_candidates[:len(boosters)]],
        "ensemble_weights":   weights.tolist(),
        "best_solo_name":     best_solo["name"],
    }

    report = {
        "decision":           "accepted",
        "ensemble_sanity":    round(sanity_acc, 6),
        "ensemble_hand":      hand_correct,
        "ensemble_hand_total": hand_total,
        "ensemble_fitness":   ensemble_fitness,
        "best_solo":          best_solo["name"],
        "best_solo_sanity":   round(best_solo_sanity, 6),
        "best_solo_hand":     best_solo_hand,
        "sanity_gain":        round(sanity_gain, 6),
        "hand_gain":          hand_gain,
        "models_used":        [r["name"] for r, _ in saved_candidates[:len(boosters)]],
        "weights":            weights.tolist(),
        "wrong_handcrafted":  [d["word"] for d in hand_detail
                               if d.get("scoreable") and not d.get("correct")],
        "generated":          datetime.now().isoformat(),
    }
    with open(ENSEMBLE_DIR / "ensemble_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    sp(f"  Report → {ENSEMBLE_DIR / 'ensemble_report.json'}")

    del boosters
    gc.collect()
    return ensemble_result


# ============================================================================
# Final refit on full data
# ============================================================================

def refit_on_full_data(
    best_result: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
) -> None:
    """Retrain the winning configuration on 100% of the data.

    ``best_result["params"]`` and ``best_result["internal"]["best_iteration"]``
    are used for the refit.  Handcrafted rows are appended before training.
    """
    sp(f"\n{'=' * 80}")
    sp("FINAL REFIT ON FULL DATA  (always the last step)")
    sp(f"  Winner: {best_result['name']}  (phase: {best_result.get('phase', '?')})")

    best_name   = best_result["name"]
    best_iters  = best_result["internal"]["best_iteration"]
    best_params = dict(best_result["params"])

    # ── Append handcrafted rows ───────────────────────────────────────────────
    X_hand, y_hand = _handcrafted_to_training_rows(handcrafted_tests, feature_cols)
    if X_hand is not None:
        X_full = pd.concat([X[feature_cols], X_hand], ignore_index=True)
        y_full = pd.concat([y, y_hand], ignore_index=True)
        lc = {0: (y_hand == 0).sum(), 1: (y_hand == 1).sum(), 2: (y_hand == 2).sum()}
        sp(f"  Handcrafted added: {len(X_hand)} rows  "
           f"(class 0: {lc[0]}, class 1: {lc[1]}, class 2: {lc[2]})")
    else:
        X_full = X[feature_cols]
        y_full = y
        sp("  Handcrafted: nothing usable (all ambiguous or unlabelled)")

    sp(f"  sanity_acc (search phase): {best_result['external']['accuracy']*100:.2f}%")
    sp(f"  fitness   (search phase): {best_result['fitness']:.4f}")
    sp(f"  iterations: {best_iters}  (fixed from early-stopping on val)")
    sp(f"  total rows: {len(X_full):,}  (DB full + handcrafted)")
    sp(f"{'=' * 80}")
    sys.stdout.flush()

    # Remove val-set-specific params
    best_params.pop("metric", None)
    best_params["metric"]    = "None"
    best_params["num_class"] = NUM_CLASSES
    best_params["verbosity"] = -1

    t0 = time.perf_counter()
    full_ds = lgb.Dataset(X_full, label=y_full, free_raw_data=False)
    booster = lgb.train(
        best_params,
        full_ds,
        num_boost_round=best_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    train_time = time.perf_counter() - t0

    # ── Evaluate final model ──────────────────────────────────────────────────
    sanity_res = evaluate_external(booster, feature_cols, external_sample)
    hand_res   = evaluate_handcrafted(booster, feature_cols, handcrafted_tests)

    sp(f"  sanity_acc  (full-data model): {sanity_res['accuracy']*100:.2f}%  "
       f"({sanity_res['correct']}/{sanity_res['sample_size']})")
    sp(f"  handcrafted (now in train):    {hand_res['correct']}/{hand_res['total']}")
    sp(f"  train time: {train_time:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_dir  = RESULTS_DIR / f"{best_name}_FINAL_FULLDATA"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{best_name}_full.lgb"
    booster.save_model(str(model_path))

    meta = {
        "script":                        SCRIPT_NAME,
        "num_classes":                   NUM_CLASSES,
        "source_trial":                  best_name,
        "source_phase":                  best_result.get("phase", "?"),
        "source_fitness":                best_result["fitness"],
        "source_fitness_preset":         best_result.get("fitness_preset", "unknown"),
        "source_sanity_accuracy":        best_result["external"]["accuracy"],
        "full_data_sanity_accuracy":     sanity_res["accuracy"],
        "full_data_sanity_correct":      sanity_res["correct"],
        "full_data_sanity_sample_size":  sanity_res["sample_size"],
        "full_data_hand_correct":        hand_res["correct"],
        "full_data_hand_total":          hand_res["total"],
        "num_boost_round":               best_iters,
        "train_rows_db":                 len(X),
        "train_rows_handcrafted":        len(X_hand) if X_hand is not None else 0,
        "train_rows_total":              len(X_full),
        "train_time_sec":                round(train_time, 2),
        "params":                        best_params,
        "generated":                     datetime.now().isoformat(),
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)

    sp(f"  Saved → {model_path}")
    sp(f"  Meta  → {model_dir / 'meta.json'}")
    sys.stdout.flush()

    del booster
    gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    resume        = "--resume"        in sys.argv
    phase1_only   = "--phase1-only"   in sys.argv
    skip_ensemble = "--skip-ensemble" in sys.argv

    budget_hours = 24.0
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--budget-hours" and i < len(sys.argv) - 1:
            budget_hours = float(sys.argv[i + 1])
    budget_seconds = budget_hours * 3600

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not resume and OPTUNA_DB.exists():
        sp(f"  Removing old Optuna DB: {OPTUNA_DB}")
        OPTUNA_DB.unlink()

    sp("=" * 80)
    sp(f"{SCRIPT_NAME}")
    sp(f"  Phase 1: Wide random exploration     (30% budget) [specialist_binary]")
    sp(f"  Phase 2: CMA-ES from best seeds      (25% budget) [specialist_binary]")
    sp(f"  Phase 3: TPE Bayesian fine-tune      (45% budget) [luscinia_3s_specialist]")
    sp(f"  Phase 4: Ensemble top-{ENSEMBLE_TOP_K} models        (if improvement ≥ {ENSEMBLE_MIN_SANITY_GAIN:.3f})")
    sp(f"  Final:   Refit winner on 100% data   (always)")
    sp("=" * 80)
    sp(f"  Budget:    {budget_hours:.1f}h ({budget_seconds:.0f}s)")
    sp(f"  P1 budget: {budget_hours*PHASE1_FRACTION:.1f}h")
    sp(f"  P2 budget: {budget_hours*PHASE2_FRACTION:.1f}h")
    sp(f"  P3 budget: {budget_hours*(1-PHASE1_FRACTION-PHASE2_FRACTION):.1f}h")
    sp(f"  Val size:  {EXTERNAL_SAMPLE_SIZE} words")
    sp(f"  Rounds:    P1={MAX_ROUNDS_P1}  P2={MAX_ROUNDS_P2}  P3={MAX_ROUNDS_P3}")
    sp(f"  Classes:   {NUM_CLASSES}  (0=antepenult, 1=penult, 2=oxytone)")
    sp(f"  Features:  {EXPECTED_FEATURE_COUNT_3SYL}")
    sp(f"  Workers:   {N_WORKERS}")
    sp(f"  Resume:    {resume}")
    sp(f"  Skip ens:  {skip_ensemble}")

    # ── Load handcrafted words ────────────────────────────────────────────────
    handcrafted_tests = load_3syl_handcrafted(HANDCRAFTED_CSV)
    scoreable = sum(1 for t in handcrafted_tests if t[2] is not None)
    sp(f"  Handcraft: {len(handcrafted_tests)} 3-syl words, {scoreable} scoreable")

    # ── Load training data ────────────────────────────────────────────────────
    sp("\nLoading training data...")
    raw_df = load_training_data(DEFAULT_DB)
    sp(f"  {len(raw_df):,} raw rows")

    # ── Load external sanity sample ───────────────────────────────────────────
    sp(f"\nLoading {EXTERNAL_SAMPLE_SIZE}-word external sanity sample (3-syl only)...")
    external_sample = load_3syl_external(DEFAULT_DB, size=EXTERNAL_SAMPLE_SIZE,
                                          seed=EXTERNAL_SAMPLE_SEED)
    sp(f"  {len(external_sample)} 3-syl words loaded")
    lc: Dict[int, int] = {}
    for x in external_sample:
        lc[x["expected_label"]] = lc.get(x["expected_label"], 0) + 1
    for k in sorted(lc):
        sp(f"    class {k}: {lc[k]} ({lc[k]/len(external_sample)*100:.1f}%)")
    majority = max(lc.values())
    sp(f"  Naive baseline (predict majority): {majority/len(external_sample)*100:.1f}%")

    # ── Build feature matrix ──────────────────────────────────────────────────
    sp(f"\nBuilding 3-syl feature matrix ({EXPECTED_FEATURE_COUNT_3SYL} features)...")
    X, y, lemmas = build_3syl_dataset(raw_df)
    feature_cols = list(X.columns)
    sp(f"  Dataset: {len(X):,} rows, {len(feature_cols)} features")
    for k in sorted(y.unique()):
        sp(f"  Class {k}: {(y==k).sum():,} ({(y==k).mean()*100:.1f}%)")
    del raw_df
    gc.collect()

    # ── Group-split by lemma ──────────────────────────────────────────────────
    sp("\nGroup-split by lemma (90/10)...")
    X_train, X_val, y_train, y_val = group_split(lemmas, X, y)
    sp(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
    del X, y, lemmas
    gc.collect()

    # ── Resume: restore wall_start and previous results ─────────────────────
    resumed_elapsed_sec: float = 0.0
    all_results: List[dict] = []

    if resume and RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as _fh:
                _saved = json.load(_fh)
            if isinstance(_saved, list) and _saved:
                all_results = _saved
                resumed_elapsed_sec = max(
                    r.get("wall_elapsed_sec", 0.0) for r in _saved
                )
                sp(f"  [resume] Loaded {len(all_results)} previous results "
                   f"({resumed_elapsed_sec/3600:.2f}h already elapsed)")
        except Exception as _exc:
            sp(f"  [resume] Could not read results JSON: {_exc} — starting fresh")
            all_results = []
            resumed_elapsed_sec = 0.0

    wall_start    = time.perf_counter() - resumed_elapsed_sec
    phase1_budget = budget_seconds * PHASE1_FRACTION
    phase2_budget = budget_seconds * PHASE2_FRACTION
    existing_names: set = {r["name"] for r in all_results}

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Broad random exploration
    # ═══════════════════════════════════════════════════════════════════════════
    sp(f"\n{'=' * 80}")
    sp("PHASE 1 — BROAD RANDOM EXPLORATION")
    sp(f"  Budget: {phase1_budget/3600:.1f}h  Sampler: RandomSampler")
    sp(f"  Space: leaves=31-511, lr=0.01-0.30, depth=4-14")
    sp(f"  max_rounds={MAX_ROUNDS_P1}  Pruner: Hyperband(min=100,max={MAX_ROUNDS_P1},rf=3)")
    sp(f"{'=' * 80}")
    sys.stdout.flush()

    p1_sampler = RandomSampler(seed=42)
    p1_study = optuna.create_study(
        study_name="luscinia_lgbm_str_ua_3s_v1_p1",
        storage=f"sqlite:///{OPTUNA_DB}",
        direction="maximize",
        sampler=p1_sampler,
        pruner=HyperbandPruner(min_resource=100, max_resource=MAX_ROUNDS_P1,
                               reduction_factor=3),
        load_if_exists=resume,
    )

    p1_obj = Phase1Objective(
        X_train, y_train, X_val, y_val, feature_cols,
        external_sample, handcrafted_tests, wall_start, phase1_budget,
        phase_budget=phase1_budget,
    )
    p1_obj.all_results = [r for r in all_results if r.get("phase") == "P1"]
    p1_obj._trial_counter = len(p1_obj.all_results)

    try:
        p1_study.optimize(p1_obj, timeout=phase1_budget,
                          show_progress_bar=False, gc_after_trial=True)
    except KeyboardInterrupt:
        sp("Phase 1 interrupted by user")

    for _r in p1_obj.all_results:
        if _r["name"] not in existing_names:
            all_results.append(_r)
            existing_names.add(_r["name"])
    analyze_phase1_landscape(p1_obj.all_results)
    print_phase_summary("P1", p1_obj.all_results)

    if phase1_only or not p1_obj.all_results:
        sp("\nPhase 1 only — exiting.")
        log_training_summary(all_results, time.perf_counter() - wall_start,
                             SCRIPT_NAME, LOG_FILE)
        print_final_leaderboard(all_results, time.perf_counter() - wall_start)
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — CMA-ES from Phase 1 best seeds
    # ═══════════════════════════════════════════════════════════════════════════
    phase2_start   = time.perf_counter()
    time_remaining = budget_seconds - (phase2_start - wall_start)
    actual_p2_budget = min(phase2_budget, time_remaining * 0.55)

    if actual_p2_budget < 300:
        sp("\nNot enough time for Phase 2, skipping.")
    else:
        best_p1 = sorted(p1_obj.all_results, key=lambda r: r["fitness"], reverse=True)[0]
        best_p1_params = best_p1["params"]

        sp(f"\n{'=' * 80}")
        sp("PHASE 2 — CMA-ES OPTIMISATION")
        sp(f"  Budget: {actual_p2_budget/3600:.1f}h  Seeded from: {best_p1['name']}")
        sp(f"  Seed: sanity={best_p1['external']['accuracy']*100:.2f}%  "
           f"fit={best_p1['fitness']:.4f}")
        sp(f"  max_rounds={MAX_ROUNDS_P2}")
        sp(f"{'=' * 80}")
        sys.stdout.flush()

        x0_raw = {
            "num_leaves":              float(best_p1_params.get("num_leaves", 300)),
            "max_depth":               float(best_p1_params.get("max_depth", 9)),
            "learning_rate":           best_p1_params.get("learning_rate", 0.10),
            "min_child_samples":       float(best_p1_params.get("min_child_samples", 20)),
            "lambda_l1":               best_p1_params.get("lambda_l1", 0.1),
            "lambda_l2":               best_p1_params.get("lambda_l2", 1.0),
            "subsample":               best_p1_params.get("subsample", 0.8),
            "colsample_bytree":        best_p1_params.get("colsample_bytree", 0.7),
            "feature_fraction_bynode": best_p1_params.get("feature_fraction_bynode", 0.7),
            "min_sum_hessian_in_leaf": best_p1_params.get("min_sum_hessian_in_leaf", 1.0),
            "path_smooth":             best_p1_params.get("path_smooth", 0.1),
            "min_split_gain":          best_p1_params.get("min_split_gain", 0.0),
        }
        x0_clipped = _clip_to_p2_bounds(x0_raw)
        clipped_info = ", ".join(
            f"{k}={x0_raw[k]:.3g}->{x0_clipped[k]:.3g}"
            for k in x0_clipped if abs(x0_raw.get(k, 0) - x0_clipped[k]) > 1e-9
        )
        sp(f"  CMA-ES x0: {clipped_info or 'no clipping needed'}")

        p2_sampler = CmaEsSampler(
            x0=x0_clipped, sigma0=0.3, seed=42,
            warn_independent_sampling=False,
        )
        p2_study = optuna.create_study(
            study_name="luscinia_lgbm_str_ua_3s_v1_p2",
            storage=f"sqlite:///{OPTUNA_DB}",
            direction="maximize",
            sampler=p2_sampler,
            pruner=PatientPruner(
                HyperbandPruner(min_resource=50, max_resource=MAX_ROUNDS_P2,
                                reduction_factor=3),
                patience=0,
            ),
            load_if_exists=resume,
        )

        p2_obj = Phase2Objective(
            X_train, y_train, X_val, y_val, feature_cols,
            external_sample, handcrafted_tests, wall_start, budget_seconds,
            seed_params=best_p1_params,
            phase_budget=actual_p2_budget,
        )
        p2_obj.all_results = [r for r in all_results if r.get("phase") == "P2"]
        p2_obj._trial_counter = len(p2_obj.all_results)

        try:
            p2_study.optimize(p2_obj, timeout=actual_p2_budget,
                              show_progress_bar=False, gc_after_trial=True)
        except KeyboardInterrupt:
            sp("Phase 2 interrupted by user")

        for _r in p2_obj.all_results:
            if _r["name"] not in existing_names:
                all_results.append(_r)
                existing_names.add(_r["name"])
        print_phase_summary("P2", p2_obj.all_results)

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — TPE Bayesian fine-tuning  [luscinia_3s_specialist]
    # ═══════════════════════════════════════════════════════════════════════════
    time_remaining = budget_seconds - (time.perf_counter() - wall_start)
    if time_remaining < 300:
        sp("\nNot enough time for Phase 3, skipping.")
    else:
        best_overall = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
        best_params  = best_overall["params"]

        sp(f"\n{'=' * 80}")
        sp("PHASE 3 — TPE BAYESIAN FINE-TUNING  [luscinia_3s_specialist]")
        sp(f"  Budget: {time_remaining/3600:.1f}h  Seeded from: {best_overall['name']}")
        sp(f"  Best so far: sanity={best_overall['external']['accuracy']*100:.2f}%  "
           f"fit={best_overall['fitness']:.4f}")
        sp(f"  Fitness preset: luscinia_3s_specialist  (hand_acc weight = 25%)")
        sp(f"  max_rounds={MAX_ROUNDS_P3}  Narrow search ±50% around winner")
        sp(f"{'=' * 80}")
        sys.stdout.flush()

        p3_sampler = TPESampler(
            n_startup_trials=15,
            seed=42,
            multivariate=True,
            group=True,
            consider_endpoints=True,
            n_ei_candidates=48,
            warn_independent_sampling=False,
        )
        p3_study = optuna.create_study(
            study_name="luscinia_lgbm_str_ua_3s_v1_p3",
            storage=f"sqlite:///{OPTUNA_DB}",
            direction="maximize",
            sampler=p3_sampler,
            pruner=PatientPruner(
                HyperbandPruner(min_resource=50, max_resource=MAX_ROUNDS_P3,
                                reduction_factor=3),
                patience=0,
            ),
            load_if_exists=resume,
        )

        p3_obj = Phase3Objective(
            X_train, y_train, X_val, y_val, feature_cols,
            external_sample, handcrafted_tests, wall_start, budget_seconds,
            best_params=best_params,
            phase_budget=time_remaining,
        )
        p3_obj.all_results = [r for r in all_results if r.get("phase") == "P3"]
        p3_obj._trial_counter = len(p3_obj.all_results)

        try:
            p3_study.optimize(p3_obj, timeout=time_remaining,
                              show_progress_bar=False, gc_after_trial=True)
        except KeyboardInterrupt:
            sp("Phase 3 interrupted by user")

        for _r in p3_obj.all_results:
            if _r["name"] not in existing_names:
                all_results.append(_r)
                existing_names.add(_r["name"])
        print_phase_summary("P3", p3_obj.all_results)

    # ── Training summary (ML lab‐style end‐of‐run analysis) ──────────────────
    wall_elapsed = time.perf_counter() - wall_start
    log_training_summary(all_results, wall_elapsed, SCRIPT_NAME, LOG_FILE)
    print_final_leaderboard(all_results, wall_elapsed)
    sp(f"\nSearch complete. Total trials: {len(all_results)}")
    sp(f"Results saved to: {RESULTS_DIR}")

    if not all_results:
        sp("No results — aborting.")
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Ensemble
    # ═══════════════════════════════════════════════════════════════════════════
    ensemble_result: Optional[dict] = None
    if not skip_ensemble:
        ensemble_result = run_ensemble_phase(
            all_results, feature_cols,
            external_sample, handcrafted_tests,
            top_k=ENSEMBLE_TOP_K,
            min_sanity_gain=ENSEMBLE_MIN_SANITY_GAIN,
        )
    else:
        sp("\n[P4] Ensemble skipped (--skip-ensemble flag).")

    # ── Determine the winner ──────────────────────────────────────────────────
    if ensemble_result is not None:
        winning_result = ensemble_result
        sp(f"\n  Winner: P4_ensemble  "
           f"(sanity={ensemble_result['external']['accuracy']*100:.2f}%  "
           f"hand={ensemble_result['handcrafted']['correct']}/{ensemble_result['handcrafted']['total']})")
    else:
        winning_result = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
        sp(f"\n  Winner: {winning_result['name']}  "
           f"(fitness={winning_result['fitness']:.4f}  "
           f"sanity={winning_result['external']['accuracy']*100:.2f}%  "
           f"hand={winning_result['handcrafted']['correct']}/{winning_result['handcrafted']['total']})")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL REFIT
    # ═══════════════════════════════════════════════════════════════════════════
    sp("\nReconstructing full dataset for final refit...")
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    refit_on_full_data(
        winning_result, X_full, y_full, feature_cols,
        external_sample, handcrafted_tests,
    )


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    main()
