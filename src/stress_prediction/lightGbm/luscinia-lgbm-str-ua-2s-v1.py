#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Luscinia 2S v1.0 — 2-Syllable Specialist
==========================================
Named after *Luscinia megarhynchos* (the Common Nightingale):
a specialist of extraordinary accuracy in its domain.

MOTIVATION
----------
Successor to the "Bulbul v4 Extended Research" run.  Luscinia v1.0 builds
on three lessons learned:

  1. **Objective function** — the ``specialist_binary`` preset weighted
     general val-set accuracy (acc) at 25%, which is a weak discriminator
     at this accuracy level.  ``luscinia_specialist`` redistributes that
     10 pp to ``hand_acc`` (15% → 25%), so Phase 3 TPE is actively pushed
     toward configs that cover real poetic edge-case words — the closest
     proxy to production quality.

  2. **Phase 4 Ensemble** — the top-K solo models are combined via a
     fitness-weighted soft vote.  The ensemble is accepted *only* if it
     genuinely outperforms the best solo model on both the external sanity
     sample and the handcrafted set (threshold: ``+0.001`` in sanity_acc
     OR any gain in hand_correct).  If the ensemble does not improve, the
     best solo model is carried forward unchanged.

  3. **Guaranteed final refit** — after whichever model wins (solo or
     ensemble), the training script *always* retrains that configuration
     on 100% of the available data (DB + unambiguous handcrafted words)
     using the ``best_iteration`` from the search phase.

PHASES
------
Phase 1 (30% budget) — BROAD RANDOM EXPLORATION
  Wide random sampling to map the fitness landscape.  No pruning so every
  config is seen in full.  Covers all combinations: bagging/goss,
  shallow/deep, fast/slow lr.

Phase 2 (25% budget) — CMA-ES FROM BEST SEEDS
  Continuous optimisation seeded from the top Phase 1 config.  CMA-ES
  is superior to TPE for correlated continuous parameters.  Fixed:
  bagging + is_unbalance + max_bin=255 (confirmed by Phase 1 analysis).

Phase 3 (≈45% budget) — TPE BAYESIAN FINE-TUNING  [luscinia_specialist]
  Multivariate TPE warm-started from the best Phase 1+2 config.  Narrow
  search space (±50% around winner).  Uses the ``luscinia_specialist``
  fitness preset so hand_acc is weighted at 25% — substantially higher
  than in v4's Phase 3 (was 15%).

Phase 4 — ENSEMBLE
  Loads the top-K saved models (default K=5) and combines their
  predictions via a fitness-weighted soft vote.  Evaluated on the same
  external + handcrafted sets.  Accepted only if the ensemble is
  strictly better than the best solo model.

Final Refit — always executes after the winning model (solo or ensemble)
  is determined.  Retrains the winning *parameter configuration* on:
    - 100% of the DB rows (train + val combined)
    - + unambiguous 2-syl handcrafted words

FITNESS PRESET — luscinia_specialist
  f1        = 0.50   primary: captures class imbalance well
  acc       = 0.15   per-form val accuracy (lighter tiebreaker)
  hand_acc  = 0.25   real poetic words — closest to prod
  sanity_acc= 0.10   external DB sample — catches encoding disasters

  (Compare: specialist_binary has acc=0.25, hand_acc=0.15.)

TERMINOLOGY
-----------
``sanity_acc`` (formerly ``ext_acc`` in older scripts): accuracy on a
  random 5,000-word external DB sample.  It is a *sanity check*, not the
  primary optimisation target.  Its purpose is to catch catastrophic
  regressions (e.g. encoding bugs, overfitting to a narrow train split).
  The term was renamed to make the intent explicit.

FILESYSTEM
----------
artifacts/luscinia-lgbm-str-ua-2s-v1/
  luscinia-lgbm-str-ua-2s-v1-results.json   -- per-trial JSONL
  luscinia-lgbm-str-ua-2s-v1-results.csv    -- per-trial CSV with timing
  leaderboard.txt                           -- final leaderboard (after all phases)
  optuna_study.db                           -- Optuna SQLite storage
  phase1_summary.json                       -- Phase 1 landscape analysis
  feature_importance/                       -- per-trial top-20 gain features
  convergence/                              -- per-trial loss curves
  P4_ensemble/                              -- ensemble model(s) + eval report
  <trial_name>/                             -- model files for saved trials
  <trial_name>_FINAL_FULLDATA/              -- full-data refit output

USAGE
-----
  python luscinia-lgbm-str-ua-2s-v1.py                     # 24h full run
  python luscinia-lgbm-str-ua-2s-v1.py --budget-hours 8    # custom budget
  python luscinia-lgbm-str-ua-2s-v1.py --resume            # resume interrupted run
  python luscinia-lgbm-str-ua-2s-v1.py --phase1-only       # broad exploration only
  python luscinia-lgbm-str-ua-2s-v1.py --skip-ensemble     # skip P4 ensemble phase

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
    accuracy_score, f1_score, classification_report,
    roc_auc_score, precision_recall_fscore_support,
)

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import HyperbandPruner, PatientPruner, NopPruner

from services.constants import VOWEL_SET, EXPECTED_FEATURE_COUNT
from services.feature_service import build_features_v13
from services.data_service import (
    load_training_data, load_handcrafted_tests,
    stress_to_vowel_label, group_split,
)
from services.evaluation_service import (
    evaluate_external as _svc_evaluate_external,
    evaluate_handcrafted as _svc_evaluate_handcrafted,
    compute_fitness as _svc_compute_fitness,
)
from services.logging_service import (
    sp,
    log_trial_result,
    log_phase_progress,
    log_phase_summary,
    log_final_leaderboard,
    append_result_json,
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
SCRIPT_NAME = "Luscinia-LGBM-STR-UA-2S-v1"
N_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_DB     = Path(__file__).parent.parent / "data" / "stress_training.db"
HANDCRAFTED_CSV = Path(__file__).parent.parent / "data" / "handcrafted_test_words.csv"
RESULTS_DIR    = Path(__file__).parent / "artifacts" / "luscinia-lgbm-str-ua-2s-v1"
RESULTS_CSV    = RESULTS_DIR / "luscinia-lgbm-str-ua-2s-v1-results.csv"
RESULTS_JSON   = RESULTS_DIR / "luscinia-lgbm-str-ua-2s-v1-results.json"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.txt"
OPTUNA_DB      = RESULTS_DIR / "optuna_study.db"
PHASE1_SUMMARY = RESULTS_DIR / "phase1_summary.json"
CONVERGENCE_DIR = RESULTS_DIR / "convergence"
FEAT_IMP_DIR   = RESULTS_DIR / "feature_importance"
ENSEMBLE_DIR   = RESULTS_DIR / "P4_ensemble"
LOG_FILE       = RESULTS_DIR / "run.log"

NUM_CLASSES = 2
EXTERNAL_SAMPLE_SIZE = 5000   # 5,000-word external sanity sample
EXTERNAL_SAMPLE_SEED = 42

# ── Phase budget fractions ───────────────────────────────────────────────────
PHASE1_FRACTION = 0.30   # 30%: broad random exploration
PHASE2_FRACTION = 0.25   # 25%: CMA-ES continuous push
# Phase 3 uses the remaining 45%: multivariate TPE fine-tune

# ── lightgbm base params ─────────────────────────────────────────────────────
BASE_LGBM = {
    "boosting_type":      "gbdt",
    "objective":          "binary",
    "metric":             "binary_logloss",
    "max_bin":            255,
    "num_threads":        N_WORKERS,
    "seed":               42,
    "verbosity":          -1,
    "force_col_wise":     True,
    "feature_pre_filter": False,
}

# ── Phase-specific max rounds ────────────────────────────────────────────────
MAX_ROUNDS_P1 = 800    # Phase 1: fast exploration, keep it light
MAX_ROUNDS_P2 = 1500   # Phase 2: deeper push; CMA-ES benefits from longer runs
MAX_ROUNDS_P3 = 1200   # Phase 3: TPE fine-tune with Hyperband pruning

# ── Phase 2 search bounds ────────────────────────────────────────────────────
# Must bracket every value Phase 1 can produce so the CMA-ES x0 seed is
# always strictly inside the feasible region (prevents AssertionError).
P2_BOUNDS = {
    "num_leaves":              (31,    800),
    "max_depth":               (4,     14),
    "learning_rate":           (0.01,  0.30),   # log-scale
    "min_child_samples":       (5,     200),    # log-scale
    "lambda_l1":               (0.0,   2.0),
    "lambda_l2":               (0.0,   10.0),
    "subsample":               (0.4,   1.0),
    "colsample_bytree":        (0.2,   1.0),
    "feature_fraction_bynode": (0.3,   1.0),
    "min_sum_hessian_in_leaf": (1e-4,  30.0),  # log-scale
    "path_smooth":             (0.0,   1.0),
    "min_split_gain":          (0.0,   0.3),
}

# ── Ensemble settings ────────────────────────────────────────────────────────
ENSEMBLE_TOP_K = 5        # How many saved solo models to combine
# Accept ensemble only when it improves beyond this threshold on sanity_acc.
# This prevents accepting noise-level variations from being mistaken as gains.
ENSEMBLE_MIN_SANITY_GAIN = 0.001

# ── CSV schema ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "phase", "trial_number", "name", "fitness",
    "f1", "accuracy", "auc",
    "sanity_accuracy", "sanity_correct", "sanity_sample_size",   # formerly ext_*
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
# Data loading
# ============================================================================

class TwoSylChunkProcessor:
    """Multiprocessing-safe chunk processor for 2-syllable feature extraction."""

    def __call__(self, chunk: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in chunk.iterrows():
            form = row["form"] or ""
            pos = row.get("pos") or ""
            features_json = row.get("features_json") or None
            lower = normalize_apostrophe(form).lower()
            vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
            if len(vowels) != 2:
                continue
            try:
                stress_raw = json.loads(row["stress_indices"] or "[]")
            except Exception:
                stress_raw = []
            label = stress_to_vowel_label(stress_raw, vowels)
            if label < 0 or label >= NUM_CLASSES:
                continue
            rec = build_features_v13(form, pos, features_json)
            rec["__label__"] = label
            rec["__lemma__"] = row.get("lemma", "")
            records.append(rec)
        return pd.DataFrame(records)


def build_2syl_dataset(df: pd.DataFrame, n_workers: int = None):
    """Build feature matrix X and label vector y for 2-syllable words."""
    n_workers = n_workers or N_WORKERS
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()
    df["stress_count"] = (
        df.groupby(["form", "pos", "features_json"])["stress_indices"]
          .transform("nunique")
    )
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"]).reset_index(drop=True)
    chunk_size = max(1, len(df) // n_workers)
    chunks = [df.iloc[i: i + chunk_size] for i in range(0, len(df), chunk_size)]
    processor = TwoSylChunkProcessor()
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(processor, chunks)
    combined = pd.concat(results, ignore_index=True)
    y = combined["__label__"].astype(int)
    lemmas = combined["__lemma__"]
    X = combined.drop(columns=["__label__", "__lemma__"])
    return X, y, lemmas


def load_2syl_external(db_path: Path, size: int = 5000, seed: int = 42) -> list:
    """Load external validation pool (up to *size* unique 2-syl words).

    This sample is used as a *sanity check* (``sanity_acc``) — it catches
    catastrophic regressions but is not the primary optimisation target.
    """
    import sqlite3, random
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT form, lemma, stress_indices, pos, features_json "
        "FROM training_entries "
        "WHERE variant_type NOT IN ('free_variant', 'grammatical_homonym')"
    )
    rows = cur.fetchall()
    conn.close()

    pool = []
    seen = set()
    for form, lemma, stress_raw_str, pos, features_json in rows:
        if not form or not stress_raw_str:
            continue
        lower = normalize_apostrophe(form).lower()
        key = (lower, pos or "")
        if key in seen:
            continue
        seen.add(key)
        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) != 2:
            continue
        try:
            stress_raw = json.loads(stress_raw_str)
        except Exception:
            continue
        if not stress_raw:
            continue
        vi = stress_raw[0]
        if vi not in (0, 1):
            continue
        pool.append({
            "form":           form,
            "pos":            pos or "X",
            "expected_label": vi,
            "lemma":          lemma or "",
            "features_json":  features_json,
        })

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:size]


def load_2syl_handcrafted(csv_path: Path) -> list:
    """Load 2-syllable words from the handcrafted test CSV.

    These are real poetic / rare words curated by hand.  They represent
    the *production* use-case more faithfully than the DB sample.
    """
    all_tests = load_handcrafted_tests(csv_path)
    tests_2syl = []
    for entry in all_tests:
        word          = entry[0]
        pos           = entry[1]           if len(entry) > 1 else "X"
        expected      = entry[2]           if len(entry) > 2 else None
        desc          = entry[3]           if len(entry) > 3 else ""
        features_json = entry[4]           if len(entry) > 4 else None
        lower  = normalize_apostrophe(word).lower()
        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) == 2:
            tests_2syl.append((word, pos, expected, desc, features_json))
    return tests_2syl


# ============================================================================
# Evaluation  (thin wrappers around evaluation_service — binary mode)
# ============================================================================

def evaluate_external(booster, feature_cols: list, sample: list) -> dict:
    """Binary-mode external (sanity) evaluation for 2-syl specialist."""
    return _svc_evaluate_external(booster, feature_cols, sample, mode="binary")


def evaluate_handcrafted(booster, feature_cols: list, tests: list) -> dict:
    """Binary-mode handcrafted evaluation for 2-syl specialist."""
    return _svc_evaluate_handcrafted(booster, feature_cols, tests, mode="binary")


def compute_fitness(
    f1: float,
    ext_acc: float,
    hand_acc: float,
    acc: float,
    hand_correct: int,
    hand_total: int,
    *,
    model_size_mb: float = 0.0,
    preset: str = "luscinia_specialist",
) -> Tuple[float, bool, bool]:
    """Luscinia composite fitness (v2 redesign).

    Default preset: ``luscinia_specialist``
      f1=45 %, acc=20 %, hand_acc=35 %  (no sanity weight)

    Sanity is evaluated as a guard only: if ``ext_acc < acc`` the returned
    score is halved (SANITY_BELOW_ACC_PENALTY) and ``sanity_violated=True``.

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


# ============================================================================
# Core training function  (shared across Phases 1, 2, 3)
# ============================================================================

def train_and_evaluate(
    params: dict,
    X_train, y_train,
    X_val, y_val,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
    max_rounds: int,
    early_stopping_rounds: int,
    trial: optuna.Trial,
    phase: str,
    name: str,
    wall_start: float,
    fitness_preset: str = "luscinia_specialist",
) -> Optional[dict]:
    """Train one lightgbm model, evaluate fully, persist results.

    Parameters
    ----------
    params:
        lightgbm hyperparameters for this trial.
    X_train, y_train, X_val, y_val:
        Pre-split feature matrices and label vectors.
    feature_cols:
        Column names matching the feature matrices.
    external_sample:
        Sanity-check sample loaded by ``load_2syl_external``.
    handcrafted_tests:
        Curated poetic words loaded by ``load_2syl_handcrafted``.
    max_rounds:
        Hard ceiling on boosting rounds (per-phase constant).
    early_stopping_rounds:
        Patience for lightgbm's built-in early stopping.
    trial:
        Live Optuna trial (used for pruning + intermediate reporting).
    phase:
        String tag written into the result dict (``"P1"``, ``"P2"``, ``"P3"``).
    name:
        Unique trial name for file output (e.g. ``"P3_0042"``).
    wall_start:
        ``time.perf_counter()`` recorded at the very start of the run.
    fitness_preset:
        Weight preset forwarded to ``compute_fitness``.  P1 and P2 use
        ``"specialist_binary"`` to keep Phase 1/2 landscape consistent with
        prior v4 data.  P3 uses ``"luscinia_specialist"`` to push harder
        toward handcrafted accuracy.

    Returns
    -------
    dict or None
        Full result dict on success.  ``None`` if training failed.
    """
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

    td = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    vd = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False, reference=td)

    t0 = time.perf_counter()
    booster = None
    best_logloss = None
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
                booster = lgb.train(params, td, num_boost_round=cp,
                                    valid_sets=[vd], callbacks=callbacks)
            else:
                booster = lgb.train(params, td, num_boost_round=additional,
                                    valid_sets=[vd], callbacks=callbacks,
                                    init_model=booster)
            prev_cp = cp

            proba_val = booster.predict(X_val)
            y_pred    = (proba_val >= 0.5).astype(int)
            cp_f1     = f1_score(y_val, y_pred, average="binary")
            cp_acc    = accuracy_score(y_val, y_pred)
            bs = booster.best_score.get("valid_0", {})
            best_logloss = bs.get("binary_logloss", None)
            convergence_curve.append({
                "cp":      cp,
                "f1":      round(cp_f1, 4),
                "acc":     round(cp_acc, 4),
                "logloss": round(best_logloss, 6) if best_logloss else None,
            })
            sp(f"       cp={cp:<5}  F1={cp_f1*100:.2f}%  acc={cp_acc*100:.2f}%"
               + (f"  loss={best_logloss:.5f}" if best_logloss else ""), flush=True)

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

    train_time   = time.perf_counter() - t0
    wall_elapsed = time.perf_counter() - wall_start

    # ── Full evaluation ──────────────────────────────────────────────────────
    proba_val = booster.predict(X_val)
    y_pred    = (proba_val >= 0.5).astype(int)
    acc       = accuracy_score(y_val, y_pred)
    f1_bin    = f1_score(y_val, y_pred, average="binary")
    f1_w      = f1_score(y_val, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_val, proba_val)
    except Exception:
        auc = 0.0
    precision, recall, _, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    # sanity_acc = external DB sample accuracy (sanity-check only)
    sanity_res = evaluate_external(booster, feature_cols, external_sample)
    hand_res   = evaluate_handcrafted(booster, feature_cols, handcrafted_tests)

    num_trees = booster.num_trees()
    size_mb   = num_trees * 0.008

    fitness, penalty, sanity_bad = compute_fitness(
        f1_bin,
        sanity_res["accuracy"],   # sanity guard (zero weight, hard penalty if < acc)
        hand_res["accuracy"],
        acc,
        hand_res["correct"],
        hand_res["total"],
        model_size_mb=size_mb,
        preset=fitness_preset,
    )
    if sanity_bad:
        sp("\n  ⚠⚠⚠  SANITY BELOW VAL_ACC — sanity_acc={:.4f} < acc={:.4f}"
           " — possible data leak or label bug!  Score halved.".format(
               sanity_res["accuracy"], acc))

    # ── Feature importance (gain) ────────────────────────────────────────────
    feat_imp = sorted(
        zip(feature_cols, booster.feature_importance(importance_type="gain")),
        key=lambda x: x[1], reverse=True,
    )

    # ── Persist convergence curve ────────────────────────────────────────────
    CONVERGENCE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONVERGENCE_DIR / f"{name}_convergence.json", "w", encoding="utf-8") as fh:
        json.dump(convergence_curve, fh)

    # ── Persist feature importance (top 20) ──────────────────────────────────
    FEAT_IMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEAT_IMP_DIR / f"{name}_feat_imp.txt", "w", encoding="utf-8") as fh:
        for fn, imp in feat_imp[:20]:
            fh.write(f"{fn}: {imp:.2f}\n")

    result = {
        "phase":              phase,
        "trial_number":       trial.number if trial else -1,
        "name":               name,
        "timestamp":          datetime.now().isoformat(),
        "params":             params,
        "fitness_preset":     fitness_preset,
        "train_time_sec":     round(train_time, 2),
        "wall_elapsed_sec":   round(wall_elapsed, 2),
        "wall_elapsed_min":   round(wall_elapsed / 60, 2),
        "internal": {
            "accuracy":       round(acc, 6),
            "f1":             round(f1_bin, 6),
            "f1_weighted":    round(f1_w, 6),
            "precision":      round(precision, 6),
            "recall":         round(recall, 6),
            "auc":            round(auc, 6),
            "best_iteration": booster.best_iteration,
            "num_trees":      num_trees,
            "best_logloss":   round(best_logloss, 6) if best_logloss else None,
            "class_0_f1":     round(report.get("0", {}).get("f1-score", 0), 4),
            "class_1_f1":     round(report.get("1", {}).get("f1-score", 0), 4),
        },
        # "external" key kept for backward-compat with analysis scripts.
        # Semantically this is the *sanity* sample — not the optimisation target.
        "external":           sanity_res,
        "handcrafted":        hand_res,
        "model":              {"estimated_size_mb": round(size_mb, 2)},
        "fitness":            fitness,
        "hand_penalty_applied": penalty,
        "sanity_violated":    sanity_bad,
        "top20_features":     [fn for fn, _ in feat_imp[:20]],
        "convergence_curve":  convergence_curve,
    }

    # ── Persist to JSON + CSV ────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    append_result_json(result, RESULTS_JSON)
    append_result_csv(result, RESULTS_CSV, CSV_FIELDS)

    # ── Save model when quality is promising ─────────────────────────────────
    # Threshold: fitness > 0.87  OR  no penalty AND sanity_acc ≥ 95.5%
    if fitness > 0.87 or (not penalty and sanity_res["accuracy"] >= 0.955):
        model_dir = RESULTS_DIR / name
        model_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_dir / f"{name}.lgb"))
        sp(f"   Model saved -> {model_dir}")

    del booster
    gc.collect()
    return result


# ============================================================================
# PHASE 1 — Broad Random Exploration
# ============================================================================

class Phase1Objective:
    """Wide random search to map the fitness landscape.

    Fitness preset: ``specialist_binary`` (weights acc=25%, hand_acc=15%).
    Using the same preset as v4 Phase 1 keeps the landscape shape comparable
    and makes the best-seed selection compatible between runs.

    No pruning — every config is evaluated in full so we see the complete
    fitness distribution before committing to CMA-ES seeds.
    """

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget_sec,
                 phase_budget: float = 0.0):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.feature_cols          = feature_cols
        self.external_sample       = external_sample
        self.handcrafted_tests     = handcrafted_tests
        self.wall_start            = wall_start
        self.budget_sec            = budget_sec
        self.phase_budget          = phase_budget
        self.all_results: List[dict] = []

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget_sec:
            raise optuna.exceptions.OptunaError("Phase 1 budget exhausted")

        trial_num = trial.number
        name      = f"P1_{trial_num:04d}"

        # Wide search — intentionally broad to map the full landscape
        num_leaves  = trial.suggest_int("num_leaves", 31, 511)
        max_depth   = trial.suggest_int("max_depth", 4, 14)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.30, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 300, log=True)
        lambda_l1   = trial.suggest_float("lambda_l1", 0.0, 2.0)
        lambda_l2   = trial.suggest_float("lambda_l2", 0.0, 10.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.2, 1.0)
        min_sum_hessian = trial.suggest_float("min_sum_hessian_in_leaf", 0.0001, 50.0, log=True)
        path_smooth = trial.suggest_float("path_smooth", 0.0, 2.0)
        min_split_gain = trial.suggest_float("min_split_gain", 0.0, 0.5)
        data_strategy = trial.suggest_categorical("data_sample_strategy", ["bagging", "goss"])
        max_bin = trial.suggest_categorical("max_bin", [63, 127, 255, 511])

        params = {
            **BASE_LGBM,
            "max_bin":               max_bin,
            "num_leaves":            num_leaves,
            "max_depth":             max(max_depth,
                                         int(math.ceil(math.log2(max(2, num_leaves)))) + 1),
            "learning_rate":         learning_rate,
            "min_child_samples":     min_child_samples,
            "lambda_l1":             lambda_l1,
            "lambda_l2":             lambda_l2,
            "colsample_bytree":      colsample_bytree,
            "feature_fraction_bynode": feature_fraction_bynode,
            "min_sum_hessian_in_leaf": min_sum_hessian,
            "path_smooth":           path_smooth,
            "min_split_gain":        min_split_gain,
            "data_sample_strategy":  data_strategy,
            "is_unbalance":          True,
        }

        if data_strategy == "bagging":
            subsample = trial.suggest_float("subsample", 0.4, 1.0)
            params["subsample"]      = subsample
            params["subsample_freq"] = 1
        else:
            params["subsample"]      = 1.0
            params["subsample_freq"] = 0

        sp(f"\n   [P1 #{trial_num}] leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={params['max_depth']}  "
           f"strat={data_strategy}  max_bin={max_bin}")
        sys.stdout.flush()

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            max_rounds=MAX_ROUNDS_P1, early_stopping_rounds=60,
            trial=trial, phase="P1", name=name, wall_start=self.wall_start,
            fitness_preset="specialist_binary",   # landscape compat with v4
        )

        if result is None:
            return 0.0

        log_trial_result(result, LOG_FILE)
        self.all_results.append(result)
        elapsed = time.perf_counter() - self.wall_start
        log_phase_progress("P1", self.all_results, elapsed, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        return result["fitness"]


# ============================================================================
# PHASE 2 — CMA-ES from Phase 1 best seeds
# ============================================================================

def _clip_to_p2_bounds(params: dict) -> dict:
    """Clip parameter values to P2_BOUNDS.

    CMA-ES requires x0 to lie strictly inside the search box.
    If a Phase 1 winner has params outside P2_BOUNDS (e.g. ``num_leaves=800``
    when the upper bound is 800 — inclusive), the library raises
    ``AssertionError: invalid bounds`` on the first trial.  This helper
    prevents that crash by clamping to the open interior.
    """
    clipped = {}
    for key, (lo, hi) in P2_BOUNDS.items():
        val = params.get(key)
        if val is None:
            continue
        clipped[key] = float(max(lo, min(hi, val)))
    return clipped


class Phase2Objective:
    """CMA-ES optimizer seeded from the best Phase 1 configuration.

    CMA-ES excels at continuous correlated-parameter optimization once a
    good starting point is known.  Phase 2 fixes discrete choices confirmed
    by Phase 1 analysis (bagging, is_unbalance=True, max_bin=255) and lets
    CMA-ES explore the continuous hyperparameter surface.

    Fitness preset: ``specialist_binary`` (same as Phase 1) for consistent
    cross-phase comparison.  The ``luscinia_specialist`` preset is reserved
    for Phase 3 to maximise hand_acc in the final fine-tuning push.
    """

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget_sec,
                 seed_params: dict, phase_budget: float = 0.0):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.feature_cols          = feature_cols
        self.external_sample       = external_sample
        self.handcrafted_tests     = handcrafted_tests
        self.wall_start            = wall_start
        self.budget_sec            = budget_sec
        self.seed_params           = seed_params
        self.phase_budget          = phase_budget
        self.all_results: List[dict] = []

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget_sec:
            raise optuna.exceptions.OptunaError("Phase 2 budget exhausted")

        trial_num = trial.number
        name      = f"P2_{trial_num:04d}"

        b = P2_BOUNDS
        num_leaves = trial.suggest_int(
            "num_leaves", b["num_leaves"][0], b["num_leaves"][1])
        max_depth = trial.suggest_int(
            "max_depth", b["max_depth"][0], b["max_depth"][1])
        learning_rate = trial.suggest_float(
            "learning_rate", b["learning_rate"][0], b["learning_rate"][1], log=True)
        min_child_samples = trial.suggest_int(
            "min_child_samples", b["min_child_samples"][0], b["min_child_samples"][1], log=True)
        lambda_l1 = trial.suggest_float(
            "lambda_l1", b["lambda_l1"][0], b["lambda_l1"][1])
        lambda_l2 = trial.suggest_float(
            "lambda_l2", b["lambda_l2"][0], b["lambda_l2"][1])
        subsample = trial.suggest_float(
            "subsample", b["subsample"][0], b["subsample"][1])
        colsample_bytree = trial.suggest_float(
            "colsample_bytree", b["colsample_bytree"][0], b["colsample_bytree"][1])
        feature_fraction_bynode = trial.suggest_float(
            "feature_fraction_bynode",
            b["feature_fraction_bynode"][0], b["feature_fraction_bynode"][1])
        min_sum_hessian = trial.suggest_float(
            "min_sum_hessian_in_leaf",
            b["min_sum_hessian_in_leaf"][0], b["min_sum_hessian_in_leaf"][1], log=True)
        path_smooth = trial.suggest_float(
            "path_smooth", b["path_smooth"][0], b["path_smooth"][1])
        min_split_gain = trial.suggest_float(
            "min_split_gain", b["min_split_gain"][0], b["min_split_gain"][1])

        params = {
            **BASE_LGBM,
            "max_bin":               255,
            "is_unbalance":          True,
            "data_sample_strategy":  "bagging",
            "subsample_freq":        1,
            "num_leaves":            num_leaves,
            "max_depth":             max(max_depth,
                                         int(math.ceil(math.log2(max(2, num_leaves)))) + 1),
            "learning_rate":         learning_rate,
            "min_child_samples":     min_child_samples,
            "lambda_l1":             lambda_l1,
            "lambda_l2":             lambda_l2,
            "subsample":             subsample,
            "colsample_bytree":      colsample_bytree,
            "feature_fraction_bynode": feature_fraction_bynode,
            "min_sum_hessian_in_leaf": min_sum_hessian,
            "path_smooth":           path_smooth,
            "min_split_gain":        min_split_gain,
        }

        sp(f"\n   [P2 #{trial_num}] leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={params['max_depth']}")
        sys.stdout.flush()

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            max_rounds=MAX_ROUNDS_P2, early_stopping_rounds=80,
            trial=trial, phase="P2", name=name, wall_start=self.wall_start,
            fitness_preset="specialist_binary",   # consistent with Phase 1
        )

        if result is None:
            return 0.0

        log_trial_result(result, LOG_FILE)
        self.all_results.append(result)
        elapsed = time.perf_counter() - self.wall_start
        log_phase_progress("P2", self.all_results, elapsed, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        return result["fitness"]


# ============================================================================
# PHASE 3 — TPE Bayesian fine-tuning  [luscinia_specialist]
# ============================================================================

class Phase3Objective:
    """Multivariate TPE Bayesian optimisation warm-started from Phase 1+2 best.

    Key difference from Phase 1/2: uses the **``luscinia_specialist``** preset
    (hand_acc weight = 25%, acc weight = 15%).  This causes TPE to actively
    seek hyperparameter configs that improve accuracy on the curated poetic
    word set — the closest proxy to real production quality.

    Narrow search space (±50% around the best Phase 1+2 config) keeps
    the optimisation focused.  Hyperband pruning discards clearly bad configs
    early (at checkpoint 50) to save time for promising directions.
    """

    def __init__(self, X_train, y_train, X_val, y_val, feature_cols,
                 external_sample, handcrafted_tests, wall_start, budget_sec,
                 best_params: dict, phase_budget: float = 0.0):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.feature_cols          = feature_cols
        self.external_sample       = external_sample
        self.handcrafted_tests     = handcrafted_tests
        self.wall_start            = wall_start
        self.budget_sec            = budget_sec
        self.best_params           = best_params
        self.phase_budget          = phase_budget
        self.all_results: List[dict] = []

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget_sec:
            raise optuna.exceptions.OptunaError("Phase 3 budget exhausted")

        trial_num = trial.number
        name      = f"P3_{trial_num:04d}"

        bp = self.best_params

        def _r(key, lo, hi, log=False):
            return trial.suggest_float(key, lo, hi, log=log)

        def _ri(key, lo, hi):
            return trial.suggest_int(key, lo, hi)

        # Narrow range: ±50% around the Phase 1+2 winner
        best_leaves = bp.get("num_leaves", 300)
        best_lr     = bp.get("learning_rate", 0.10)
        best_mc     = bp.get("min_child_samples", 20)
        best_depth  = bp.get("max_depth", 9)

        num_leaves = _ri("num_leaves",
                          max(31,  int(best_leaves * 0.6)),
                          min(511, int(best_leaves * 1.4)))
        max_depth = _ri("max_depth",
                         max(6,   best_depth - 2),
                         min(13,  best_depth + 2))
        learning_rate = _r("learning_rate",
                            max(0.02, best_lr * 0.5),
                            min(0.30, best_lr * 2.0), log=True)
        min_child_samples = _ri("min_child_samples",
                                 max(5,   int(best_mc * 0.5)),
                                 min(200, int(best_mc * 2.5)))
        lambda_l1               = _r("lambda_l1",               0.0,    1.0)
        lambda_l2               = _r("lambda_l2",               0.0,    6.0)
        subsample               = _r("subsample",               0.4,    1.0)
        colsample_bytree        = _r("colsample_bytree",        0.2,    1.0)
        feature_fraction_bynode = _r("feature_fraction_bynode", 0.3,    1.0)
        min_sum_hessian         = _r("min_sum_hessian_in_leaf", 0.0001, 20.0, log=True)
        path_smooth             = _r("path_smooth",             0.0,    0.8)
        min_split_gain          = _r("min_split_gain",          0.0,    0.2)

        params = {
            **BASE_LGBM,
            "max_bin":               255,
            "is_unbalance":          True,
            "data_sample_strategy":  "bagging",
            "subsample_freq":        1,
            "num_leaves":            num_leaves,
            "max_depth":             max(max_depth,
                                         int(math.ceil(math.log2(max(2, num_leaves)))) + 1),
            "learning_rate":         learning_rate,
            "min_child_samples":     min_child_samples,
            "lambda_l1":             lambda_l1,
            "lambda_l2":             lambda_l2,
            "subsample":             subsample,
            "colsample_bytree":      colsample_bytree,
            "feature_fraction_bynode": feature_fraction_bynode,
            "min_sum_hessian_in_leaf": min_sum_hessian,
            "path_smooth":           path_smooth,
            "min_split_gain":        min_split_gain,
        }

        sp(f"\n   [P3 #{trial_num}] leaves={num_leaves}  lr={learning_rate:.5f}  "
           f"mc={min_child_samples}  depth={params['max_depth']}")
        sys.stdout.flush()

        result = train_and_evaluate(
            params, self.X_train, self.y_train, self.X_val, self.y_val,
            self.feature_cols, self.external_sample, self.handcrafted_tests,
            max_rounds=MAX_ROUNDS_P3, early_stopping_rounds=70,
            trial=trial, phase="P3", name=name, wall_start=self.wall_start,
            fitness_preset="luscinia_specialist",  # ← boosted hand_acc weight
        )

        if result is None:
            return 0.0

        log_trial_result(result, LOG_FILE)
        self.all_results.append(result)
        elapsed = time.perf_counter() - self.wall_start
        log_phase_progress("P3", self.all_results, elapsed, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        return result["fitness"]


# ============================================================================
# PHASE 4 — Ensemble
# ============================================================================

def run_ensemble_phase(
    all_results: list,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
    top_k: int = ENSEMBLE_TOP_K,
    min_sanity_gain: float = ENSEMBLE_MIN_SANITY_GAIN,
) -> Optional[dict]:
    """Combine the top-K saved solo models via fitness-weighted soft voting.

    Strategy
    --------
    1. Select the K saved models with the highest **luscinia_specialist** fitness
       from Phase 3 (or Phase 2/1 if fewer P3 models were saved).
    2. Build a weighted average of their predicted probabilities, where the
       weight of each model = softmax of its fitness score.  This gives better
       models proportionally more influence without hard-cutting weaker ones.
    3. Evaluate the ensemble on the external sanity sample and the handcrafted
       set using the same binary-mode evaluation as the solo models.
    4. **Accept** the ensemble only if it strictly outperforms the best solo
       model on *at least one* of:
         - sanity_acc improvement ≥ ``min_sanity_gain`` (default: +0.001)
         - hand_correct > best solo hand_correct
       This prevents marginal noise-level variations from being mistaken for
       genuine improvements.

    Returns
    -------
    dict or None
        Result dict (same schema as ``train_and_evaluate``) for the ensemble
        if accepted, or ``None`` if rejected (best solo is better).
    """
    # ── 1. Collect saved models from all phases ───────────────────────────────
    # Prefer P3 results (luscinia_specialist fitness), fall back to P2/P1
    p3_results = [r for r in all_results if r.get("phase") == "P3"]
    other_results = [r for r in all_results if r.get("phase") != "P3"]
    candidates = sorted(p3_results, key=lambda r: r["fitness"], reverse=True)
    candidates += sorted(other_results, key=lambda r: r["fitness"], reverse=True)

    # Keep only results whose model files were actually saved
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
    sp(f"PHASE 4 — ENSEMBLE  ({len(saved_candidates)} models)")
    sp(f"  Strategy: fitness-weighted soft vote")
    sp(f"  Accept criterion: sanity_acc gain ≥ {min_sanity_gain:.3f}  OR  more hand_correct")
    sp(f"{'=' * 80}")

    # ── 2. Load boosters and compute weights ──────────────────────────────────
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

    # Softmax weights (numerically stable)
    fit_arr = np.array(fitnesses, dtype=float)
    fit_arr -= fit_arr.max()          # shift for numerical stability
    weights  = np.exp(fit_arr)
    weights /= weights.sum()
    sp(f"\n  Model weights (softmax of fitness):")
    for (r, _), w in zip(saved_candidates[:len(boosters)], weights):
        sp(f"    {r['name']:12s}  weight={w:.4f}")

    # ── 3. Ensemble evaluation on external sanity sample ─────────────────────
    sp(f"\n  Evaluating ensemble on {len(external_sample)}-word sanity sample...")
    sanity_correct = 0
    for item in external_sample:
        form = item["form"]
        pos  = item.get("pos", "X")
        feat_json = item.get("features_json")
        expected  = item["expected_label"]
        feat = build_features_v13(form, pos, feat_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        # Weighted average probability (class 1)
        p1 = sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights))
        predicted = int(p1 >= 0.5)
        if predicted == expected:
            sanity_correct += 1
    sanity_acc = sanity_correct / len(external_sample) if external_sample else 0.0

    # ── 4. Ensemble evaluation on handcrafted set ─────────────────────────────
    sp(f"  Evaluating ensemble on {len(handcrafted_tests)} handcrafted words...")
    hand_correct = 0
    hand_total   = 0
    hand_detail  = []
    for entry in handcrafted_tests:
        word          = entry[0]
        pos           = entry[1] if len(entry) > 1 else "X"
        expected      = entry[2] if len(entry) > 2 else None
        desc          = entry[3] if len(entry) > 3 else ""
        feat_json     = entry[4] if len(entry) > 4 else None
        if expected is None:
            hand_detail.append({"word": word, "predicted": None,
                                 "correct": None, "scoreable": False})
            continue
        hand_total += 1
        feat = build_features_v13(word, pos, feat_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        p1 = float(sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights)))
        predicted = int(p1 >= 0.5)
        expected_set = set(expected) if isinstance(expected, list) else {int(expected)}
        is_correct = (predicted in expected_set)
        if is_correct:
            hand_correct += 1
        hand_detail.append({
            "word": word, "predicted": predicted,
            "correct": is_correct, "scoreable": True,
            "confidence": round(p1 if predicted == 1 else 1 - p1, 4),
        })
    hand_acc = hand_correct / hand_total if hand_total > 0 else 0.0

    sp(f"\n  Ensemble results:")
    sp(f"    sanity_acc:  {sanity_acc*100:.2f}%  ({sanity_correct}/{len(external_sample)})")
    sp(f"    hand:        {hand_correct}/{hand_total} ({hand_acc*100:.2f}%)")

    # ── 5. Compare against best solo ──────────────────────────────────────────
    best_solo = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
    best_solo_sanity = best_solo["external"]["accuracy"]
    best_solo_hand   = best_solo["handcrafted"]["correct"]

    sanity_gain = sanity_acc - best_solo_sanity
    hand_gain   = hand_correct - best_solo_hand

    sp(f"\n  vs best solo ({best_solo['name']}):")
    sp(f"    sanity gain: {sanity_gain:+.4f}  (threshold: {min_sanity_gain:.3f})")
    sp(f"    hand gain:   {hand_gain:+d}")

    # Accept criterion: meaningful sanity improvement OR any hand gain
    accepted = (sanity_gain >= min_sanity_gain) or (hand_gain > 0)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    if not accepted:
        sp(f"\n  [P4] REJECTED — ensemble does not improve over best solo.")
        sp(f"       Carrying forward {best_solo['name']} (fit={best_solo['fitness']:.4f})")
        # Persist rejection report
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
        sp(f"  Report -> {ENSEMBLE_DIR / 'ensemble_report.json'}")
        return None

    sp(f"\n  [P4] ACCEPTED — ensemble is better!  (+{sanity_gain:.4f} sanity, {hand_gain:+d} hand)")

    # ── 6. Build result dict in the standard format ───────────────────────────
    # Re-use best-solo params and iterations for the "winning config" record
    # (the ensemble itself has no single param set, so we tag it specially).
    ensemble_fitness, ensemble_penalty, ensemble_sanity_bad = _svc_compute_fitness(
        best_solo["internal"]["f1"],    # f1 is from the best solo (ensemble has no val set)
        sanity_acc,
        hand_acc,
        acc=best_solo["internal"]["accuracy"],
        hand_correct=hand_correct,
        hand_total=hand_total,
        model_size_mb=0.0,
        preset="luscinia_specialist",
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
        "params":             best_solo["params"],        # used for final refit
        "fitness_preset":     "luscinia_specialist",
        "train_time_sec":     0,
        "wall_elapsed_sec":   0,
        "wall_elapsed_min":   0,
        "internal":           best_solo["internal"],      # val metrics from best solo
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

    # Persist ensemble report
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
    sp(f"  Report -> {ENSEMBLE_DIR / 'ensemble_report.json'}")

    # Clean up boosters
    del boosters
    gc.collect()
    return ensemble_result


# ============================================================================
# Landscape analysis after Phase 1
# ============================================================================

def analyze_phase1_landscape(results: list) -> dict:
    """Analyse Phase 1 results to understand the fitness landscape."""
    if not results:
        return {}

    by_fit = sorted(results, key=lambda r: r["fitness"], reverse=True)
    top10       = by_fit[:10]
    top10_params = [r["params"] for r in top10]

    param_analysis = {}
    for key in ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                "lambda_l1", "lambda_l2", "colsample_bytree"]:
        vals = [p.get(key) for p in top10_params if p.get(key) is not None]
        if vals:
            param_analysis[key] = {
                "mean": sum(vals) / len(vals),
                "min":  min(vals),
                "max":  max(vals),
                "top1": top10_params[0].get(key),
            }

    ext_accs = [r["external"]["accuracy"] for r in results]
    bins = {"<90": 0, "90-92": 0, "92-94": 0, "94-95": 0, "95-96": 0, "96+": 0}
    for a in ext_accs:
        if   a < 0.90: bins["<90"]   += 1
        elif a < 0.92: bins["90-92"] += 1
        elif a < 0.94: bins["92-94"] += 1
        elif a < 0.95: bins["94-95"] += 1
        elif a < 0.96: bins["95-96"] += 1
        else:          bins["96+"]   += 1

    best = by_fit[0]
    summary = {
        "n_trials":              len(results),
        "best_fitness":          best["fitness"],
        "best_sanity_acc":       best["external"]["accuracy"],  # sanity check
        "best_name":             best["name"],
        "best_params":           best["params"],
        "mean_sanity_acc":       sum(ext_accs) / len(ext_accs),
        "max_sanity_acc":        max(ext_accs),
        "sanity_acc_distribution": bins,
        "top10_param_analysis":  param_analysis,
        "top5": [
            {
                "name":     r["name"],
                "fitness":  r["fitness"],
                "sanity":   r["external"]["accuracy"],
                "hand":     f"{r['handcrafted']['correct']}/{r['handcrafted']['total']}",
                "size_mb":  r["model"]["estimated_size_mb"],
                "params":   r["params"],
            }
            for r in by_fit[:5]
        ],
    }

    with open(PHASE1_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    return summary


def print_phase_summary(label: str, results: list) -> None:
    """Print a concise summary for a completed phase."""
    log_phase_summary(label, results, LOG_FILE)


def print_final_leaderboard(all_results: list, wall_elapsed: float) -> None:
    """Print the top-30 leaderboard and save to ``leaderboard.txt``."""
    log_final_leaderboard(
        all_results, wall_elapsed, SCRIPT_NAME, LEADERBOARD_FILE, LOG_FILE
    )


# ============================================================================
# Final refit on 100% of data  (always the last step)
# ============================================================================

def _handcrafted_to_training_rows(
    handcrafted_tests: list,
    feature_cols: list,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Convert unambiguous handcrafted entries to training rows.

    Skips entries where the expected label is ambiguous (list with >1 value)
    or unknown (``None``).  These words cannot provide a clean training signal.
    """
    rows, labels, skipped = [], [], []
    for entry in handcrafted_tests:
        word, pos, expected = entry[0], entry[1], entry[2]
        feat_json = entry[4] if len(entry) > 4 else None
        if isinstance(expected, list):
            if len(expected) != 1:
                skipped.append(f"{word}(ambiguous)")
                continue
            label = expected[0]
        elif expected is None:
            skipped.append(f"{word}(no_label)")
            continue
        else:
            label = int(expected)
        if label not in (0, 1):
            skipped.append(f"{word}(label={label})")
            continue
        feat = build_features_v13(word, pos, feat_json)
        rows.append({c: feat.get(c, 0) for c in feature_cols})
        labels.append(label)

    if skipped:
        sp(f"  Handcrafted skipped (ambiguous / no label): {', '.join(skipped)}")
    if not rows:
        return None, None
    return pd.DataFrame(rows, columns=feature_cols), pd.Series(labels, dtype=int)


def refit_on_full_data(
    best_result: dict,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    external_sample: list,
    handcrafted_tests: list,
) -> None:
    """Retrain the winning config on 100% of the data.

    This is **always the final step** regardless of whether the winner came
    from Phase 3 (solo) or Phase 4 (ensemble).

    Training data for the final model:
      * All DB rows (train + val combined, no group split needed)
      * + unambiguous 2-syl handcrafted words (poets' neologisms, rare words)

    The number of boosting rounds is fixed to ``best_iteration`` from the
    search phase.  No validation set is used — the round count was already
    determined by early stopping on held-out data.

    Parameters
    ----------
    best_result:
        The result dict of the winning configuration (Phase 3 or P4 ensemble).
        ``best_result["params"]`` and ``best_result["internal"]["best_iteration"]``
        are used for the refit.
    X, y:
        Full dataset (train + val concatenated) in feature-matrix form.
    feature_cols, external_sample, handcrafted_tests:
        Standard evaluation inputs.
    """
    sp(f"\n{'=' * 80}")
    sp("FINAL REFIT ON FULL DATA  (always the last step)")
    sp(f"  Winner: {best_result['name']}  (phase: {best_result.get('phase', '?')})")

    best_name  = best_result["name"]
    best_iters = best_result["internal"]["best_iteration"]
    best_params = dict(best_result["params"])

    # ── Append handcrafted rows ───────────────────────────────────────────────
    X_hand, y_hand = _handcrafted_to_training_rows(handcrafted_tests, feature_cols)
    if X_hand is not None:
        X_full = pd.concat([X[feature_cols], X_hand], ignore_index=True)
        y_full = pd.concat([y, y_hand], ignore_index=True)
        sp(f"  Handcrafted added: {len(X_hand)} rows  "
           f"(class 0: {(y_hand==0).sum()}, class 1: {(y_hand==1).sum()})")
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

    # Remove early-stopping params that require a validation set
    best_params.pop("metric", None)
    best_params["metric"]    = "None"
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
    # The handcrafted words are now IN the training data, so hand_correct
    # is expected to be near 100% here — a self-check, not an unbiased metric.
    sanity_res = evaluate_external(booster, feature_cols, external_sample)
    hand_res   = evaluate_handcrafted(booster, feature_cols, handcrafted_tests)

    sp(f"  sanity_acc  (full-data model): {sanity_res['accuracy']*100:.2f}%  "
       f"({sanity_res['correct']}/{sanity_res['sample_size']})")
    sp(f"  handcrafted (now in train):    {hand_res['correct']}/{hand_res['total']}")
    sp(f"  train time: {train_time:.0f}s")

    # ── Save model ────────────────────────────────────────────────────────────
    model_dir  = RESULTS_DIR / f"{best_name}_FINAL_FULLDATA"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{best_name}_full.lgb"
    booster.save_model(str(model_path))

    meta = {
        "script":                      SCRIPT_NAME,
        "source_trial":                best_name,
        "source_phase":                best_result.get("phase", "?"),
        "source_fitness":              best_result["fitness"],
        "source_fitness_preset":       best_result.get("fitness_preset", "unknown"),
        "source_sanity_accuracy":      best_result["external"]["accuracy"],
        "full_data_sanity_accuracy":   sanity_res["accuracy"],
        "full_data_sanity_correct":    sanity_res["correct"],
        "full_data_sanity_sample_size": sanity_res["sample_size"],
        "full_data_hand_correct":      hand_res["correct"],
        "full_data_hand_total":        hand_res["total"],
        "num_boost_round":             best_iters,
        "train_rows_db":               len(X),
        "train_rows_handcrafted":      len(X_hand) if X_hand is not None else 0,
        "train_rows_total":            len(X_full),
        "train_time_sec":              round(train_time, 2),
        "params":                      best_params,
        "generated":                   datetime.now().isoformat(),
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)

    sp(f"  Saved -> {model_path}")
    sp(f"  Meta  -> {model_dir / 'meta.json'}")
    sys.stdout.flush()

    del booster
    gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def main():
    import warnings
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    resume       = "--resume"        in sys.argv
    phase1_only  = "--phase1-only"   in sys.argv
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
    sp(f"  Phase 3: TPE Bayesian fine-tune      (45% budget) [luscinia_specialist]")
    sp(f"  Phase 4: Ensemble top-{ENSEMBLE_TOP_K} models        (if improvement ≥ {ENSEMBLE_MIN_SANITY_GAIN:.3f})")
    sp(f"  Final:   Refit winner on 100% data   (always)")
    sp("=" * 80)
    sp(f"  Budget:    {budget_hours:.1f}h ({budget_seconds:.0f}s)")
    sp(f"  P1 budget: {budget_hours*PHASE1_FRACTION:.1f}h")
    sp(f"  P2 budget: {budget_hours*PHASE2_FRACTION:.1f}h")
    sp(f"  P3 budget: {budget_hours*(1-PHASE1_FRACTION-PHASE2_FRACTION):.1f}h")
    sp(f"  Val size:  {EXTERNAL_SAMPLE_SIZE} words")
    sp(f"  Rounds:    P1={MAX_ROUNDS_P1}  P2={MAX_ROUNDS_P2}  P3={MAX_ROUNDS_P3}")
    sp(f"  Workers:   {N_WORKERS}")
    sp(f"  Resume:    {resume}")
    sp(f"  Skip ens:  {skip_ensemble}")

    # ── Load handcrafted words ─────────────────────────────────────────────────
    handcrafted_tests = load_2syl_handcrafted(HANDCRAFTED_CSV)
    scoreable = sum(1 for t in handcrafted_tests if t[2] is not None)
    sp(f"  Handcraft: {len(handcrafted_tests)} 2-syl words, {scoreable} scoreable")

    # ── Load training data ─────────────────────────────────────────────────────
    sp("\nLoading training data...")
    raw_df = load_training_data(DEFAULT_DB)
    sp(f"  {len(raw_df):,} raw rows")

    # ── Load external sanity sample ────────────────────────────────────────────
    sp(f"\nLoading {EXTERNAL_SAMPLE_SIZE}-word external sanity sample...")
    external_sample = load_2syl_external(DEFAULT_DB, size=EXTERNAL_SAMPLE_SIZE,
                                          seed=EXTERNAL_SAMPLE_SEED)
    sp(f"  {len(external_sample)} 2-syl words loaded")
    lc: Dict[int, int] = {}
    for x in external_sample:
        lc[x["expected_label"]] = lc.get(x["expected_label"], 0) + 1
    for k in sorted(lc):
        sp(f"    class {k}: {lc[k]} ({lc[k]/len(external_sample)*100:.1f}%)")
    majority = max(lc.values())
    sp(f"  Naive baseline (predict majority): {majority/len(external_sample)*100:.1f}%")

    # ── Build feature matrix ───────────────────────────────────────────────────
    sp(f"\nBuilding 2-syl feature matrix ({EXPECTED_FEATURE_COUNT} features)...")
    X, y, lemmas = build_2syl_dataset(raw_df)
    feature_cols = list(X.columns)
    sp(f"  Dataset: {len(X):,} rows, {len(feature_cols)} features")
    sp(f"  Class 0: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)  "
       f"Class 1: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    del raw_df
    gc.collect()

    # ── Group-split by lemma ───────────────────────────────────────────────────
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
        study_name="luscinia_lgbm_str_ua_2s_v1_p1",
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
            study_name="luscinia_lgbm_str_ua_2s_v1_p2",
            storage=f"sqlite:///{OPTUNA_DB}",
            direction="maximize",
            sampler=p2_sampler,
            pruner=PatientPruner(
                HyperbandPruner(min_resource=100, max_resource=MAX_ROUNDS_P2,
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
    # PHASE 3 — TPE Bayesian fine-tuning  [luscinia_specialist]
    # ═══════════════════════════════════════════════════════════════════════════
    time_remaining = budget_seconds - (time.perf_counter() - wall_start)
    if time_remaining < 300:
        sp("\nNot enough time for Phase 3, skipping.")
    else:
        best_overall = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
        best_params  = best_overall["params"]

        sp(f"\n{'=' * 80}")
        sp("PHASE 3 — TPE BAYESIAN FINE-TUNING  [luscinia_specialist]")
        sp(f"  Budget: {time_remaining/3600:.1f}h  Seeded from: {best_overall['name']}")
        sp(f"  Best so far: sanity={best_overall['external']['accuracy']*100:.2f}%  "
           f"fit={best_overall['fitness']:.4f}")
        sp(f"  Fitness preset: luscinia_specialist  (hand_acc weight = 25%)")
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
            study_name="luscinia_lgbm_str_ua_2s_v1_p3",
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

    # ══════════════════════════════════════════════════════════════════════════
    # Intermediate leaderboard (before ensemble / refit)
    # ══════════════════════════════════════════════════════════════════════════
    # ── Training summary (ML lab‐style end‐of‐run analysis) ──────────────────────
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

    # ── Determine the winner (ensemble if accepted, else best solo) ────────────
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
    # FINAL REFIT — always the last step
    # ═══════════════════════════════════════════════════════════════════════════
    sp("\nReconstructing full dataset for final refit...")
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)
    refit_on_full_data(
        winning_result, X_full, y_full, feature_cols,
        external_sample, handcrafted_tests,
    )


if __name__ == "__main__":
    # Force UTF-8 stdout/stderr on Windows without touching os.environ
    # (environment changes break mp.Pool child process pickling)
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    main()
