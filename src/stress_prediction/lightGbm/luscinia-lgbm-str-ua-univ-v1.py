#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Luscinia Universal v1.0 — Multisyllable Stress Predictor
==========================================================
Named after *Luscinia megarhynchos* (the Common Nightingale):
a universal model that handles ALL syllable counts in one shot.

MOTIVATION
----------
The 2S and 3S specialists achieve excellent accuracy within their syllable
buckets, but production deployment requires separate models for each count,
which is fragile and misses 4-10+ syllable words entirely.  Luscinia
Universal trains a SINGLE multiclass model on ALL syllable counts (2–10+)
using a unified 130-feature schema.

KEY DESIGN DECISIONS (informed by ``recommendations.md``)
─────────────────────────────────────────────────────────
1. **Target = vowel-ordinal from start** (0, 1, 2, … up to 10).
   The expert recommends counting stress from the END (masculine=1,
   feminine=2, dactylic=3, superdactylic=4), but the feature set encodes
   distance-from-end explicitly (``dist_from_end_v_last/pen/ante``), so
   the model can learn this mapping.  Using from-start labels keeps the
   label space bounded (max 11 classes) and compatible with the existing
   DB schema.

2. **Multiclass with num_class = MAX_VOWEL_CLASS + 1 = 11**.
   This covers words with up to 11 vowels (≈11 syllables).  Words with
   more vowels have their label capped at 10.

3. **Feature set: 132 universal features**.
   100 base (v1.3) + 32 universal extensions:
   • Explicit syllable count + normalised count
   • High-resolution ending hashes (mod=4096) + POS-compound hashes
   • Distance-from-end for last 3 vowels (absolute + normalised)
   • Coda lengths for last 3 inter-vowel segments
   • Vowel identity for last 3 positions
   • Inter-vowel intervals for last 2 gaps
   • 11 linguistic rule flags (ви- prefix, gerund, -ати/-ити/-увати,
     oxytone mobile, -ський, -адцять, foreign, Greek, penult-stable)

4. **lightgbm parameters (expert recommendations)**:
   • ``objective: multiclass`` with ``num_class: 11``
   • ``is_unbalance: True`` — handles severe class imbalance
     (class 0 and class 1 dominate; classes 5-10 are very rare)
   • ``num_leaves: 31-511`` — leaf-wise growth needs careful control
   • ``min_data_in_leaf: 50-200`` — prevents overfitting to rare classes
   • ``max_cat_threshold: 128`` — recommended for many suffix categories
   • ``bagging_fraction: 0.8 + bagging_freq: 5`` — regularisation
   • ``feature_fraction: 0.8`` — per-tree feature sampling

5. **Phase architecture**:
   Same proven 4-phase + refit pipeline as 2S/3S, but with adaptations:
   • Phase 1: wider num_leaves range (31-800) for more complex patterns
   • Phase 2: CMA-ES with higher max_rounds (2000) for convergence
   • Phase 3: TPE with ``luscinia_universal_specialist`` preset
   • Phase 4: Ensemble via argmax on averaged probability vectors
   • Final: Refit on 100% data

6. **Budget: 36h default** (vs 24h for 2S, 24h for 3S).
   Universal model has ~800K+ rows and 11 classes — needs more time.

FITNESS PRESET — luscinia_universal_specialist
  f1        = 0.45   primary: macro-F1 over all 11 classes
  acc       = 0.20   per-form val accuracy
  hand_acc  = 0.35   handcrafted poetic words — closest to prod
  (sanity: zero weight — guard only)

FILESYSTEM
----------
artifacts/luscinia-lgbm-str-ua-univ-v1/
  luscinia-lgbm-str-ua-univ-v1-results.json
  luscinia-lgbm-str-ua-univ-v1-results.csv
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
  python luscinia-lgbm-str-ua-univ-v1.py                     # 36h full run
  python luscinia-lgbm-str-ua-univ-v1.py --budget-hours 12   # custom budget
  python luscinia-lgbm-str-ua-univ-v1.py --resume            # resume interrupted
  python luscinia-lgbm-str-ua-univ-v1.py --phase1-only       # exploration only
  python luscinia-lgbm-str-ua-univ-v1.py --skip-ensemble     # skip P4

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

from services.constants import VOWEL_SET, MAX_VOWEL_CLASS
from services.feature_service_universal import (
    build_features_universal, EXPECTED_FEATURE_COUNT_UNIV,
)
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
SCRIPT_NAME = "Luscinia-LGBM-STR-UA-UNIV-v1"
N_WORKERS = max(1, mp.cpu_count() - 1)
DEFAULT_DB      = Path(__file__).parent.parent / "data" / "stress_training.db"
HANDCRAFTED_CSV = Path(__file__).parent.parent / "data" / "handcrafted_test_words.csv"
RESULTS_DIR     = Path(__file__).parent / "artifacts" / "luscinia-lgbm-str-ua-univ-v1"
RESULTS_CSV     = RESULTS_DIR / "luscinia-lgbm-str-ua-univ-v1-results.csv"
RESULTS_JSON    = RESULTS_DIR / "luscinia-lgbm-str-ua-univ-v1-results.json"
LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.txt"
OPTUNA_DB       = RESULTS_DIR / "optuna_study.db"
PHASE1_SUMMARY  = RESULTS_DIR / "phase1_summary.json"
CONVERGENCE_DIR = RESULTS_DIR / "convergence"
FEAT_IMP_DIR    = RESULTS_DIR / "feature_importance"
ENSEMBLE_DIR    = RESULTS_DIR / "P4_ensemble"
LOG_FILE        = RESULTS_DIR / "run.log"

NUM_CLASSES = MAX_VOWEL_CLASS + 1    # 11 classes (vowel indices 0..10)
MIN_VOWELS = 2                       # exclude monosyllabic words
EXTERNAL_SAMPLE_SIZE = 5000
EXTERNAL_SAMPLE_SEED = 42

# ── Phase budget fractions ───────────────────────────────────────────────────
PHASE1_FRACTION = 0.30
PHASE2_FRACTION = 0.25
# Phase 3 uses the remaining 45%

# ── lightgbm base params ─────────────────────────────────────────────────────
# Expert recommendations (recommendations.md):
# • objective: multiclass — positions are 0..10
# • is_unbalance: True — class distribution is severely skewed
# • max_bin: 255 — good default, higher adds cost for marginal gain
# • force_col_wise: True — faster for wide feature sets
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
# Universal model: 11 classes × more rows = slower convergence per round.
# Expert: "CMA-ES benefits from longer runs" → P2 gets 2000.
# P1 kept at 800 for fast exploration.  P3 at 1500 for deep fine-tuning.
MAX_ROUNDS_P1 = 800
MAX_ROUNDS_P2 = 1000  # reduced: ~30min/trial→~15min, doubles trial count
MAX_ROUNDS_P3 = 1500

# ── Phase 2 search bounds ────────────────────────────────────────────────────
P2_BOUNDS = {
    "num_leaves":              (31,    800),
    "max_depth":               (4,     16),    # deeper for 11-class
    "learning_rate":           (0.005, 0.20),  # lower floor for stability
    "min_child_samples":       (10,    300),   # higher floor: expert says 50-100
    "lambda_l1":               (0.0,   3.0),
    "lambda_l2":               (0.0,   15.0),  # wider for stronger reg
    "subsample":               (0.4,   1.0),
    "colsample_bytree":        (0.2,   1.0),
    "feature_fraction_bynode": (0.3,   1.0),
    "min_sum_hessian_in_leaf": (1e-3,  50.0),  # higher floor for multi
    "path_smooth":             (0.0,   2.0),
    "min_split_gain":          (0.0,   0.5),
}

# ── Ensemble settings ────────────────────────────────────────────────────────
ENSEMBLE_TOP_K = 5
ENSEMBLE_MIN_SANITY_GAIN = 0.001

# ── Model save threshold ─────────────────────────────────────────────────────
# Universal model on 11 classes is harder: lower threshold than 2S/3S.
MODEL_SAVE_FITNESS_THRESHOLD = 0.75

# ── CSV schema ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "phase", "trial_number", "name", "fitness",
    "f1_macro", "accuracy", "auc",
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
    preset: str = "luscinia_universal_specialist",
) -> Tuple[float, bool, bool]:
    """Luscinia Universal composite fitness (v2 redesign).

    Default preset: ``luscinia_universal_specialist``
      f1=45 % (macro-F1), acc=20 %, hand_acc=35 %  (no sanity weight)
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


def _predict_one_univ(booster, fvec: np.ndarray) -> Tuple[int, float]:
    """Argmax prediction for multiclass (11-class) booster."""
    raw = booster.predict(fvec)   # shape (1, NUM_CLASSES)
    proba = raw[0]
    predicted = int(np.argmax(proba))
    confidence = float(proba[predicted])
    return predicted, confidence


def evaluate_external(booster, feature_cols: list, external_sample: list) -> dict:
    """Evaluate the universal booster on the external sanity sample."""
    correct = 0
    total = len(external_sample)
    by_syl: Dict[int, dict] = {}
    for item in external_sample:
        feat = build_features_universal(
            item["form"], item["pos"], item.get("features_json"))
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        predicted, _ = _predict_one_univ(booster, x)
        n_syl = item.get("n_syllables", len(item.get("vowels", [])))
        bucket = min(n_syl, 7)
        if bucket not in by_syl:
            by_syl[bucket] = {"correct": 0, "total": 0}
        by_syl[bucket]["total"] += 1
        if predicted == item["expected_label"]:
            correct += 1
            by_syl[bucket]["correct"] += 1
    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(acc, 6),
        "correct": correct,
        "sample_size": total,
        "per_syllable": {
            k: {**v, "accuracy": round(v["correct"]/v["total"], 4) if v["total"] else 0}
            for k, v in sorted(by_syl.items())
        },
    }


def evaluate_handcrafted(booster, feature_cols: list, handcrafted_tests: list) -> dict:
    """Evaluate the universal booster on the handcrafted word list."""
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

        feat = build_features_universal(word, pos, features_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        predicted, confidence = _predict_one_univ(booster, x)

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

class UnivChunkProcessor:
    """Multiprocessing-safe chunk processor for universal feature extraction."""

    def __call__(self, chunk: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in chunk.iterrows():
            form = row["form"] or ""
            pos = row.get("pos") or ""
            features_json = row.get("features_json") or None

            lower = normalize_apostrophe(form).lower()
            vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
            if len(vowels) < MIN_VOWELS:
                continue

            try:
                stress_raw = json.loads(row["stress_indices"] or "[]")
            except Exception:
                stress_raw = []
            label = stress_to_vowel_label(stress_raw, vowels)
            if label < 0 or label >= NUM_CLASSES:
                continue

            rec = build_features_universal(form, pos, features_json)
            rec["__label__"] = label
            rec["__lemma__"] = row.get("lemma", "")
            records.append(rec)
        return pd.DataFrame(records)


def build_univ_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the universal feature matrix from a raw training DataFrame.

    Filters:
    * only rows with ≥ 2 vowels
    * removes free_variant / grammatical_homonym
    * removes conflicting stress annotations
    """
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()
    df["stress_count"] = (df.groupby(["form", "pos", "features_json"])
                          ["stress_indices"].transform("nunique"))
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"])
    df = df.reset_index(drop=True)

    chunk_size = max(1, len(df) // N_WORKERS)
    chunks = [df.iloc[i: i + chunk_size]
              for i in range(0, len(df), chunk_size)]

    processor = UnivChunkProcessor()
    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(processor, chunks)

    combined = pd.concat(results, ignore_index=True)
    y = combined["__label__"].astype(int)
    lemmas = combined["__lemma__"]
    X = combined.drop(columns=["__label__", "__lemma__"])
    return X, y, lemmas


def load_univ_external(db_path: Path, size: int, seed: int) -> List[dict]:
    """Load a stratified external sample — all syllable counts ≥ 2."""
    from services.data_service import load_external_sample
    return load_external_sample(db_path, size, seed)


def load_univ_handcrafted(csv_path: Path) -> list:
    """Load ALL handcrafted test words (no syllable filter).

    Validates that expected_vowel_index is within [0, MAX_VOWEL_CLASS].
    """
    raw = load_handcrafted_tests(csv_path)
    filtered = []
    for entry in raw:
        word = entry[0]
        lower = normalize_apostrophe(word).lower()
        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) < MIN_VOWELS:
            continue
        expected = entry[2]
        if expected is not None:
            indices = expected if isinstance(expected, list) else [expected]
            if any(idx >= NUM_CLASSES for idx in indices):
                continue
        filtered.append(entry)
    return filtered


# ============================================================================
# JSON persistence
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
    trial: Optional["optuna.Trial"],
    trial_name: str,
    phase: str,
    trial_number: int,
    fitness_preset: str,
    save_model: bool = False,
    train_frac: float = 1.0,
) -> Optional[dict]:
    """Train one lightgbm config and return a result dict.

    Checkpoint-based incremental training with Hyperband pruning support.
    Multiclass: lightgbm builds num_class trees per round,
    so effective trees = best_iteration * num_class.

    Args:
        train_frac: Fraction of training data to use (default 1.0). Set to
            0.5 for Phase 1 exploration to get ~2x speed-up; P2/P3 use
            full data for final quality. Val/external sets are always full.
    """
    full_params = {**BASE_LGBM, **params}
    full_params["num_class"] = NUM_CLASSES

    # ── Optional train subsampling for fast P1 exploration ────────────────────
    if train_frac < 1.0:
        rng_seed = trial_number if trial_number is not None else 42
        X_tr = X_train.sample(frac=train_frac, random_state=rng_seed)
        y_tr = y_train.loc[X_tr.index]
        sp(f"       train subsample: {len(X_tr):,} rows ({train_frac*100:.0f}% of {len(X_train):,})")
    else:
        X_tr, y_tr = X_train, y_train

    t0 = time.perf_counter()
    train_ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    val_ds   = lgb.Dataset(X_val,   label=y_val,   free_raw_data=False,
                           reference=train_ds)

    checkpoints_map = {
        "P1": [50, 150, 300, 500, max_rounds],
        "P2": [100, 300, 600, 1000, max_rounds],
        "P3": [50, 150, 300, 600, 900, max_rounds],
    }
    checkpoints = sorted(set(checkpoints_map.get(phase, [max_rounds]) + [max_rounds]))
    checkpoints = [cp for cp in checkpoints if cp <= max_rounds]

    booster = None
    prev_cp = 0
    convergence_curve = []

    try:
        for cp in checkpoints:
            additional = cp - prev_cp
            if additional <= 0:
                continue
            cbs = [lgb.log_evaluation(0),
                   lgb.early_stopping(early_stopping_rounds, verbose=False)]
            if booster is None:
                booster = lgb.train(full_params, train_ds,
                                    num_boost_round=cp,
                                    valid_sets=[val_ds], callbacks=cbs)
            else:
                booster = lgb.train(full_params, train_ds,
                                    num_boost_round=additional,
                                    valid_sets=[val_ds], callbacks=cbs,
                                    init_model=booster)
            prev_cp = cp

            raw_cp = booster.predict(X_val.values.astype(np.float32))
            y_cp   = np.argmax(raw_cp, axis=1)
            cp_f1  = float(f1_score(y_val, y_cp, average="macro", zero_division=0))
            cp_acc = float(accuracy_score(y_val, y_cp))
            bs = booster.best_score.get("valid_0", {})
            best_logloss = bs.get("multi_logloss", None)
            convergence_curve.append({"cp": cp, "f1": round(cp_f1, 6),
                                       "acc": round(cp_acc, 6),
                                       "logloss": round(best_logloss, 6) if best_logloss else None})
            sp(f"       cp={cp:<5}  F1={cp_f1*100:.2f}%  acc={cp_acc*100:.2f}%"
               + (f"  loss={best_logloss:.5f}" if best_logloss else ""))

            if trial is not None:
                trial.report(cp_f1, step=cp)
                if trial.should_prune():
                    sp(f"       PRUNED at cp={cp} (F1={cp_f1*100:.2f}%)")
                    raise optuna.TrialPruned()

    except optuna.TrialPruned:
        raise
    except Exception as e:
        sp(f"   FAILED: {e}")
        return None

    train_time = time.perf_counter() - t0
    best_iter = booster.best_iteration or max_rounds

    # ── Internal val metrics ──────────────────────────────────────────────────
    raw_pred = booster.predict(X_val.values.astype(np.float32))   # (n, 11)
    y_pred = np.argmax(raw_pred, axis=1)
    val_acc = float(accuracy_score(y_val, y_pred))
    val_f1  = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    try:
        # roc_auc_score requires all NUM_CLASSES columns in raw_pred and all
        # classes present in y_val; skip gracefully when rare classes missing
        if len(np.unique(y_val)) == NUM_CLASSES and raw_pred.shape[1] == NUM_CLASSES:
            val_auc = float(roc_auc_score(
                y_val, raw_pred, multi_class="ovr", average="macro"
            ))
        else:
            val_auc = 0.0
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

    # ── Convergence curve ─────────────────────────────────────────────────────
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
    if save_model or fitness > MODEL_SAVE_FITNESS_THRESHOLD:
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
    """Wide random exploration — RandomSampler, ``specialist_binary`` preset.

    Expert notes on Phase 1 parameter ranges:
    • num_leaves 31-800: leaf-wise growth can converge fast but overfit
    • min_child_samples 10-300: higher floor than 2S/3S (more classes)
    • learning_rate 0.005-0.30: lower floor for 11-class stability
    • lambda_l2 0-15: wider range for stronger regularisation
    • max_depth 4-16: deeper for more classes
    """

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

        # ── Wide search space ─────────────────────────────────────────────
        num_leaves        = trial.suggest_int("num_leaves", 31, 800)
        max_depth         = trial.suggest_int("max_depth", 4, 16)
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.20, log=True)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 300, log=True)
        lambda_l1         = trial.suggest_float("lambda_l1", 0.0, 3.0)
        lambda_l2         = trial.suggest_float("lambda_l2", 0.0, 15.0)
        subsample         = trial.suggest_float("subsample", 0.4, 1.0)
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.2, 1.0)
        ffbn              = trial.suggest_float("feature_fraction_bynode", 0.3, 1.0)
        min_sum_h         = trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 50.0, log=True)
        path_smooth       = trial.suggest_float("path_smooth", 0.0, 2.0)
        min_split_gain    = trial.suggest_float("min_split_gain", 0.0, 0.5)

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
            fitness_preset="luscinia_universal_specialist",
            train_frac=0.5,
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P1", self.all_results,
                           time.perf_counter() - self.wall_start, LOG_FILE,
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
        self.phase_budget = phase_budget
        self.seed_params = seed_params
        self.all_results: list = []
        self._trial_counter = 0

    def __call__(self, trial: optuna.Trial) -> float:
        if time.perf_counter() - self.wall_start >= self.budget:
            trial.study.stop()
            return 0.0

        self._trial_counter += 1
        trial_name = f"P2_{self._trial_counter:04d}"

        b = P2_BOUNDS
        num_leaves        = trial.suggest_int("num_leaves", b["num_leaves"][0], b["num_leaves"][1])
        max_depth         = trial.suggest_int("max_depth", b["max_depth"][0], b["max_depth"][1])
        learning_rate     = trial.suggest_float("learning_rate", b["learning_rate"][0], b["learning_rate"][1], log=True)
        min_child_samples = trial.suggest_int("min_child_samples", b["min_child_samples"][0], b["min_child_samples"][1], log=True)
        lambda_l1         = trial.suggest_float("lambda_l1", b["lambda_l1"][0], b["lambda_l1"][1])
        lambda_l2         = trial.suggest_float("lambda_l2", b["lambda_l2"][0], b["lambda_l2"][1])
        subsample         = trial.suggest_float("subsample", b["subsample"][0], b["subsample"][1])
        colsample_bytree  = trial.suggest_float("colsample_bytree", b["colsample_bytree"][0], b["colsample_bytree"][1])
        ffbn              = trial.suggest_float("feature_fraction_bynode", b["feature_fraction_bynode"][0], b["feature_fraction_bynode"][1])
        min_sum_h         = trial.suggest_float("min_sum_hessian_in_leaf", b["min_sum_hessian_in_leaf"][0], b["min_sum_hessian_in_leaf"][1], log=True)
        path_smooth       = trial.suggest_float("path_smooth", b["path_smooth"][0], b["path_smooth"][1])
        min_split_gain    = trial.suggest_float("min_split_gain", b["min_split_gain"][0], b["min_split_gain"][1])

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
            fitness_preset="luscinia_universal_specialist",
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P2", self.all_results,
                           time.perf_counter() - self.wall_start, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        sys.stdout.flush()
        return result["fitness"]


class Phase3Objective:
    """Multivariate TPE fine-tuning — ``luscinia_universal_specialist`` preset.

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
        self.phase_budget = phase_budget
        self.best_params = best_params
        self.all_results: list = []
        self._trial_counter = 0

    def _narrow(self, key: str, lo: float, hi: float) -> Tuple[float, float]:
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
        md_lo, md_hi = self._narrow_int("max_depth", 4, 16)
        lr_lo, lr_hi = self._narrow("learning_rate", 0.005, 0.20)
        mc_lo, mc_hi = self._narrow_int("min_child_samples", 10, 300)
        l1_lo, l1_hi = self._narrow("lambda_l1", 0.0, 3.0)
        l2_lo, l2_hi = self._narrow("lambda_l2", 0.0, 15.0)
        ss_lo, ss_hi = self._narrow("subsample", 0.4, 1.0)
        cs_lo, cs_hi = self._narrow("colsample_bytree", 0.2, 1.0)
        ff_lo, ff_hi = self._narrow("feature_fraction_bynode", 0.3, 1.0)
        mh_lo, mh_hi = self._narrow("min_sum_hessian_in_leaf", 1e-3, 50.0)
        ps_lo, ps_hi = self._narrow("path_smooth", 0.0, 2.0)
        mg_lo, mg_hi = self._narrow("min_split_gain", 0.0, 0.5)

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
            fitness_preset="luscinia_universal_specialist",
        )
        if result is None:
            return 0.0
        self.all_results.append(result)

        log_trial_result(result, LOG_FILE)
        log_phase_progress("P3", self.all_results,
                           time.perf_counter() - self.wall_start, LOG_FILE,
                           phase_budget_sec=self.phase_budget)
        sys.stdout.flush()
        return result["fitness"]


# ============================================================================
# CMA-ES helper
# ============================================================================

def _clip_to_p2_bounds(x0: dict) -> dict:
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


def print_phase_summary(phase: str, results: list) -> None:
    log_phase_summary(phase, results, LOG_FILE)


def print_final_leaderboard(all_results: list, elapsed: float) -> None:
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
    """Convert unambiguous handcrafted entries into training rows."""
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
        if label < 0 or label >= NUM_CLASSES:
            continue

        feat = build_features_universal(word, pos, features_json)
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
    """Combine top-K saved models via fitness-weighted soft vote (argmax)."""
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
        sp(f"\n[P4 Ensemble] Only {len(saved_candidates)} saved model(s) found. Skipping.")
        return None

    sp(f"\n{'=' * 80}")
    sp(f"PHASE 4 — ENSEMBLE  ({len(saved_candidates)} models, {NUM_CLASSES}-class argmax)")
    sp(f"{'=' * 80}")

    boosters, fitnesses = [], []
    for result, model_path in saved_candidates:
        try:
            bst = lgb.Booster(model_file=str(model_path))
            boosters.append(bst)
            fitnesses.append(result["fitness"])
            sp(f"  Loaded {result['name']:12s}  fit={result['fitness']:.4f}")
        except Exception as e:
            sp(f"  WARNING: could not load {model_path}: {e}")

    if len(boosters) < 2:
        return None

    fit_arr = np.array(fitnesses, dtype=float)
    fit_arr -= fit_arr.max()
    weights  = np.exp(fit_arr)
    weights /= weights.sum()

    # ── External evaluation ───────────────────────────────────────────────────
    sanity_correct = 0
    for item in external_sample:
        feat = build_features_universal(item["form"], item["pos"], item.get("features_json"))
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        avg_proba = sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights))
        if int(np.argmax(avg_proba)) == item["expected_label"]:
            sanity_correct += 1
    sanity_acc = sanity_correct / len(external_sample) if external_sample else 0.0

    # ── Handcrafted evaluation ────────────────────────────────────────────────
    hand_correct = hand_total = 0
    hand_detail = []
    for entry in handcrafted_tests:
        word = entry[0]
        pos  = entry[1] if len(entry) > 1 else "X"
        expected = entry[2] if len(entry) > 2 else None
        feat_json = entry[4] if len(entry) > 4 else None
        if expected is None:
            hand_detail.append({"word": word, "correct": None, "scoreable": False})
            continue
        hand_total += 1
        feat = build_features_universal(word, pos, feat_json)
        x = np.array([[feat.get(c, 0) for c in feature_cols]], dtype=np.float32)
        avg_proba = sum(w * bst.predict(x)[0] for bst, w in zip(boosters, weights))
        predicted = int(np.argmax(avg_proba))
        expected_set = set(expected) if isinstance(expected, list) else {int(expected)}
        is_correct = predicted in expected_set
        if is_correct:
            hand_correct += 1
        hand_detail.append({"word": word, "predicted": predicted,
                           "correct": is_correct, "scoreable": True})
    hand_acc = hand_correct / hand_total if hand_total > 0 else 0.0

    sp(f"  Ensemble: sanity={sanity_acc*100:.2f}%  hand={hand_correct}/{hand_total}")

    best_solo = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
    sanity_gain = sanity_acc - best_solo["external"]["accuracy"]
    hand_gain   = hand_correct - best_solo["handcrafted"]["correct"]
    accepted = (sanity_gain >= min_sanity_gain) or (hand_gain > 0)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "decision": "accepted" if accepted else "rejected",
        "ensemble_sanity": round(sanity_acc, 6),
        "ensemble_hand": hand_correct,
        "sanity_gain": round(sanity_gain, 6),
        "hand_gain": hand_gain,
        "models_used": [r["name"] for r, _ in saved_candidates[:len(boosters)]],
        "weights": weights.tolist(),
        "generated": datetime.now().isoformat(),
    }
    with open(ENSEMBLE_DIR / "ensemble_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    if not accepted:
        sp(f"  [P4] REJECTED — carrying forward {best_solo['name']}")
        del boosters
        gc.collect()
        return None

    sp(f"  [P4] ACCEPTED (+{sanity_gain:.4f} sanity, {hand_gain:+d} hand)")

    ensemble_fitness, ensemble_penalty, _ = _svc_compute_fitness(
        best_solo["internal"]["f1"], sanity_acc, hand_acc,
        acc=best_solo["internal"]["accuracy"],
        hand_correct=hand_correct, hand_total=hand_total,
        preset="luscinia_universal_specialist", penalty_mode=True,
    )

    ensemble_result = {
        "phase": "P4", "trial_number": -1, "name": "P4_ensemble",
        "timestamp": datetime.now().isoformat(),
        "params": best_solo["params"],
        "fitness_preset": "luscinia_universal_specialist",
        "train_time_sec": 0, "wall_elapsed_sec": 0, "wall_elapsed_min": 0,
        "internal": best_solo["internal"],
        "external": {"accuracy": round(sanity_acc, 6), "correct": sanity_correct,
                     "sample_size": len(external_sample)},
        "handcrafted": {"accuracy": round(hand_acc, 6), "correct": hand_correct,
                        "total": hand_total, "total_words": len(handcrafted_tests),
                        "results": hand_detail},
        "model": best_solo["model"],
        "fitness": ensemble_fitness,
        "hand_penalty_applied": ensemble_penalty,
        "ensemble_models": [r["name"] for r, _ in saved_candidates[:len(boosters)]],
        "ensemble_weights": weights.tolist(),
        "best_solo_name": best_solo["name"],
    }

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
    sp(f"\n{'=' * 80}")
    sp("FINAL REFIT ON FULL DATA  (always the last step)")
    sp(f"  Winner: {best_result['name']}  (phase: {best_result.get('phase', '?')})")

    best_name   = best_result["name"]
    best_iters  = best_result["internal"]["best_iteration"]
    best_params = dict(best_result["params"])

    X_hand, y_hand = _handcrafted_to_training_rows(handcrafted_tests, feature_cols)
    if X_hand is not None:
        X_full = pd.concat([X[feature_cols], X_hand], ignore_index=True)
        y_full = pd.concat([y, y_hand], ignore_index=True)
        sp(f"  Handcrafted added: {len(X_hand)} rows")
    else:
        X_full = X[feature_cols]
        y_full = y
        sp("  Handcrafted: nothing usable")

    sp(f"  iterations: {best_iters}  total rows: {len(X_full):,}")
    sp(f"{'=' * 80}")
    sys.stdout.flush()

    best_params.pop("metric", None)
    # Merge with BASE_LGBM so objective/num_class are always present,
    # then let best_params override hyperparams (leaves, lr, etc.)
    refit_params = {**BASE_LGBM, **best_params}
    refit_params["num_class"] = NUM_CLASSES        # enforce — never trust saved params alone
    refit_params["objective"] = "multiclass"       # enforce — same reason
    refit_params.pop("metric", None)               # disable metric: no val set, no callbacks
    refit_params["verbosity"] = -1

    t0 = time.perf_counter()
    full_ds = lgb.Dataset(X_full, label=y_full, free_raw_data=False)
    booster = lgb.train(
        refit_params, full_ds,
        num_boost_round=best_iters,
        callbacks=[lgb.log_evaluation(0)],
    )
    train_time = time.perf_counter() - t0

    sanity_res = evaluate_external(booster, feature_cols, external_sample)
    hand_res   = evaluate_handcrafted(booster, feature_cols, handcrafted_tests)

    sp(f"  sanity_acc:  {sanity_res['accuracy']*100:.2f}%  "
       f"({sanity_res['correct']}/{sanity_res['sample_size']})")
    sp(f"  handcrafted: {hand_res['correct']}/{hand_res['total']}")
    sp(f"  train time:  {train_time:.0f}s")

    # Per-syllable breakdown
    per_syl = sanity_res.get("per_syllable", {})
    if per_syl:
        sp("  Per-syllable breakdown:")
        for b in sorted(per_syl.keys()):
            v = per_syl[b]
            sp(f"    {b}-syl: {v['correct']}/{v['total']} "
               f"({v['accuracy']*100:.1f}%)")

    model_dir  = RESULTS_DIR / f"{best_name}_FINAL_FULLDATA"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{best_name}_full.lgb"
    booster.save_model(str(model_path))

    meta = {
        "script": SCRIPT_NAME, "num_classes": NUM_CLASSES,
        "source_trial": best_name,
        "source_fitness": best_result["fitness"],
        "full_data_sanity_accuracy": sanity_res["accuracy"],
        "full_data_hand_correct": hand_res["correct"],
        "full_data_hand_total": hand_res["total"],
        "full_data_per_syllable": per_syl,
        "num_boost_round": best_iters,
        "train_rows_total": len(X_full),
        "train_time_sec": round(train_time, 2),
        "params": best_params,
        "generated": datetime.now().isoformat(),
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)

    sp(f"  Saved → {model_path}")
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
    refit_only    = "--refit-only"    in sys.argv  # skip all phases, jump to refit

    budget_hours = 36.0   # Default: 36h (universal is ~800K+ rows, 11 classes)
    winner_name: Optional[str] = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--budget-hours" and i < len(sys.argv) - 1:
            budget_hours = float(sys.argv[i + 1])
        if arg == "--winner-name" and i < len(sys.argv) - 1:
            winner_name = sys.argv[i + 1]
    budget_seconds = budget_hours * 3600

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not resume and OPTUNA_DB.exists():
        sp(f"  Removing old Optuna DB: {OPTUNA_DB}")
        OPTUNA_DB.unlink()

    sp("=" * 80)
    sp(f"{SCRIPT_NAME}")
    sp(f"  Phase 1: Wide random exploration     (30% budget) [specialist_binary]")
    sp(f"  Phase 2: CMA-ES from best seeds      (25% budget) [specialist_binary]")
    sp(f"  Phase 3: TPE Bayesian fine-tune      (45% budget) [luscinia_univ_specialist]")
    sp(f"  Phase 4: Ensemble top-{ENSEMBLE_TOP_K} models        (if improvement ≥ {ENSEMBLE_MIN_SANITY_GAIN:.3f})")
    sp(f"  Final:   Refit winner on 100% data   (always)")
    sp("=" * 80)
    sp(f"  Budget:    {budget_hours:.1f}h ({budget_seconds:.0f}s)")
    sp(f"  P1 budget: {budget_hours*PHASE1_FRACTION:.1f}h  "
       f"P2: {budget_hours*PHASE2_FRACTION:.1f}h  "
       f"P3: {budget_hours*(1-PHASE1_FRACTION-PHASE2_FRACTION):.1f}h")
    sp(f"  Rounds:    P1={MAX_ROUNDS_P1}  P2={MAX_ROUNDS_P2}  P3={MAX_ROUNDS_P3}")
    sp(f"  Classes:   {NUM_CLASSES}  (vowel indices 0..{NUM_CLASSES-1})")
    sp(f"  Features:  {EXPECTED_FEATURE_COUNT_UNIV}")
    sp(f"  Workers:   {N_WORKERS}")
    sp(f"  Resume:    {resume}")
    sp(f"  Skip ens:  {skip_ensemble}")

    # ── Load handcrafted words ────────────────────────────────────────────────
    handcrafted_tests = load_univ_handcrafted(HANDCRAFTED_CSV)
    scoreable = sum(1 for t in handcrafted_tests if t[2] is not None)
    sp(f"  Handcraft: {len(handcrafted_tests)} words (all syllable counts), {scoreable} scoreable")

    # ── Load training data ────────────────────────────────────────────────────
    sp("\nLoading training data...")
    raw_df = load_training_data(DEFAULT_DB)
    sp(f"  {len(raw_df):,} raw rows")

    # ── Load external sanity sample ───────────────────────────────────────────
    sp(f"\nLoading {EXTERNAL_SAMPLE_SIZE}-word external sanity sample (all syl counts)...")
    external_sample = load_univ_external(DEFAULT_DB, size=EXTERNAL_SAMPLE_SIZE,
                                          seed=EXTERNAL_SAMPLE_SEED)
    sp(f"  {len(external_sample)} words loaded")
    lc: Dict[int, int] = {}
    for x in external_sample:
        lc[x["expected_label"]] = lc.get(x["expected_label"], 0) + 1
    for k in sorted(lc):
        sp(f"    class {k}: {lc[k]} ({lc[k]/len(external_sample)*100:.1f}%)")

    # ── Build feature matrix ──────────────────────────────────────────────────
    sp(f"\nBuilding universal feature matrix ({EXPECTED_FEATURE_COUNT_UNIV} features)...")
    X, y, lemmas = build_univ_dataset(raw_df)
    feature_cols = list(X.columns)
    sp(f"  Dataset: {len(X):,} rows, {len(feature_cols)} features")
    for k in sorted(y.unique()):
        cnt = (y == k).sum()
        sp(f"  Class {k}: {cnt:>8,} ({cnt/len(y)*100:>5.1f}%)")
    del raw_df
    gc.collect()

    # ── Group-split by lemma ──────────────────────────────────────────────────
    sp("\nGroup-split by lemma (90/10)...")
    X_train, X_val, y_train, y_val = group_split(lemmas, X, y)
    sp(f"  Train: {len(X_train):,}  Val: {len(X_val):,}")
    del X, y, lemmas
    gc.collect()

    # ── Resume: restore wall_start and already-completed results ─────────────
    # When --resume is set we want the budget clock to continue from where it
    # left off, not restart from 0.  We read the elapsed time recorded in the
    # last entry of the results JSON file and rewind wall_start accordingly.
    resumed_elapsed_sec: float = 0.0
    all_results: List[dict] = []

    if (resume or refit_only) and RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if isinstance(saved, list) and saved:
                all_results = saved
                # wall_elapsed_sec is recorded per-trial; take the max across all
                resumed_elapsed_sec = max(
                    r.get("wall_elapsed_sec", 0.0) for r in saved
                )
                sp(f"  [resume] Loaded {len(all_results)} previous results "
                   f"({resumed_elapsed_sec/3600:.2f}h already elapsed)")
        except Exception as exc:
            sp(f"  [resume] Could not read results JSON: {exc} — starting fresh")
            all_results = []
            resumed_elapsed_sec = 0.0

    # wall_start is pinned so that time.perf_counter() - wall_start equals the
    # TOTAL wall time (previously elapsed + current run).
    wall_start = time.perf_counter() - resumed_elapsed_sec
    # Track all result names to avoid duplicates when merging phase outputs.
    existing_names: set = {r["name"] for r in all_results}

    # ── --refit-only: skip all phases, refit winner immediately ──────────────
    if refit_only:
        if not all_results:
            sp("[refit-only] No saved results found — cannot refit. Run without --refit-only first.")
            return
        if winner_name:
            cands = [r for r in all_results if r["name"] == winner_name]
            if not cands:
                sp(f"[refit-only] --winner-name '{winner_name}' not found in results. "
                   f"Available: {[r['name'] for r in sorted(all_results, key=lambda r: r['fitness'], reverse=True)[:5]]}")
                return
            winning_result = cands[0]
        else:
            winning_result = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
        sp(f"[refit-only] Skipping phases — refitting {winning_result['name']} "
           f"(fitness={winning_result['fitness']:.4f})")
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = pd.concat([y_train, y_val], ignore_index=True)
        refit_on_full_data(
            winning_result, X_full, y_full, feature_cols,
            external_sample, handcrafted_tests,
        )
        return
    # ─────────────────────────────────────────────────────────────────────────

    phase1_budget  = budget_seconds * PHASE1_FRACTION
    phase2_budget  = budget_seconds * PHASE2_FRACTION

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Broad random exploration
    # ═══════════════════════════════════════════════════════════════════════════
    sp(f"\n{'=' * 80}")
    sp("PHASE 1 — BROAD RANDOM EXPLORATION")
    sp(f"  Budget: {phase1_budget/3600:.1f}h  Sampler: RandomSampler")
    sp(f"  Space: leaves=31-800, lr=0.005-0.20, depth=4-16, mc=10-300")
    sp(f"  max_rounds={MAX_ROUNDS_P1}  Pruner: Hyperband(min=100)")
    sp(f"{'=' * 80}")
    sys.stdout.flush()

    # When resuming, P1 may already be complete (all previous results have
    # phase="P1").  Compute how much P1 budget is left.
    p1_already_elapsed = sum(
        r.get("train_time_sec", 0.0) for r in all_results if r.get("phase") == "P1"
    ) if resume else 0.0
    p1_budget_remaining = max(0.0, phase1_budget - p1_already_elapsed)
    # Also cap by actual wall time already used against the phase1 window
    wall_used = time.perf_counter() - wall_start
    p1_budget_remaining = max(0.0, phase1_budget - wall_used)

    p1_sampler = RandomSampler(seed=42)
    p1_study = optuna.create_study(
        study_name="luscinia_lgbm_str_ua_univ_v1_p1",
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
    # Seed p1_obj with previously restored P1 results so phase progress
    # counting is correct from the first new trial onwards.
    p1_obj.all_results = [r for r in all_results if r.get("phase") == "P1"]
    p1_obj._trial_counter = len(p1_obj.all_results)

    if p1_budget_remaining > 60:
        try:
            p1_study.optimize(p1_obj, timeout=p1_budget_remaining,
                              show_progress_bar=False, gc_after_trial=True)
        except KeyboardInterrupt:
            sp("Phase 1 interrupted by user")
    else:
        sp(f"  [resume] Phase 1 budget already exhausted — skipping.")

    # Merge new P1 results (avoid duplicates by name)
    for r in p1_obj.all_results:
        if r["name"] not in existing_names:
            all_results.append(r)
            existing_names.add(r["name"])

    analyze_phase1_landscape([r for r in all_results if r.get("phase") == "P1"])
    print_phase_summary("P1", [r for r in all_results if r.get("phase") == "P1"])

    if phase1_only or not all_results:
        sp("\nPhase 1 only — exiting.")
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
        best_p1 = sorted(
            [r for r in all_results if r.get("phase") == "P1"],
            key=lambda r: r["fitness"], reverse=True
        )[0]
        best_p1_params = best_p1["params"]

        sp(f"\n{'=' * 80}")
        sp("PHASE 2 — CMA-ES OPTIMISATION")
        sp(f"  Budget: {actual_p2_budget/3600:.1f}h  Seeded from: {best_p1['name']}")
        sp(f"  max_rounds={MAX_ROUNDS_P2}")
        sp(f"{'=' * 80}")
        sys.stdout.flush()

        x0_raw = {
            "num_leaves":              float(best_p1_params.get("num_leaves", 300)),
            "max_depth":               float(best_p1_params.get("max_depth", 9)),
            "learning_rate":           best_p1_params.get("learning_rate", 0.05),
            "min_child_samples":       float(best_p1_params.get("min_child_samples", 50)),
            "lambda_l1":               best_p1_params.get("lambda_l1", 0.1),
            "lambda_l2":               best_p1_params.get("lambda_l2", 2.0),
            "subsample":               best_p1_params.get("subsample", 0.8),
            "colsample_bytree":        best_p1_params.get("colsample_bytree", 0.7),
            "feature_fraction_bynode": best_p1_params.get("feature_fraction_bynode", 0.7),
            "min_sum_hessian_in_leaf": best_p1_params.get("min_sum_hessian_in_leaf", 1.0),
            "path_smooth":             best_p1_params.get("path_smooth", 0.1),
            "min_split_gain":          best_p1_params.get("min_split_gain", 0.0),
        }
        x0_clipped = _clip_to_p2_bounds(x0_raw)

        p2_sampler = CmaEsSampler(
            x0=x0_clipped, sigma0=0.25, seed=42,  # balanced: wide enough to escape P1 region
            warn_independent_sampling=False,
        )
        p2_study = optuna.create_study(
            study_name="luscinia_lgbm_str_ua_univ_v1_p2",
            storage=f"sqlite:///{OPTUNA_DB}",
            direction="maximize",
            sampler=p2_sampler,
            pruner=PatientPruner(
                HyperbandPruner(min_resource=50, max_resource=MAX_ROUNDS_P2,
                                reduction_factor=4),  # prune @ 50/200/800 — stable for 11-class
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

        for r in p2_obj.all_results:
            if r["name"] not in existing_names:
                all_results.append(r)
                existing_names.add(r["name"])
        print_phase_summary("P2", [r for r in all_results if r.get("phase") == "P2"])

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — TPE Bayesian fine-tuning  [luscinia_universal_specialist]
    # ═══════════════════════════════════════════════════════════════════════════
    time_remaining = budget_seconds - (time.perf_counter() - wall_start)
    if time_remaining < 300:
        sp("\nNot enough time for Phase 3, skipping.")
    else:
        best_overall = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
        best_params  = best_overall["params"]

        sp(f"\n{'=' * 80}")
        sp("PHASE 3 — TPE BAYESIAN FINE-TUNING  [luscinia_universal_specialist]")
        sp(f"  Budget: {time_remaining/3600:.1f}h  Seeded from: {best_overall['name']}")
        sp(f"  Fitness: luscinia_universal_specialist  (hand_acc=35%)")
        sp(f"  max_rounds={MAX_ROUNDS_P3}  Narrow search ±50%")
        sp(f"{'=' * 80}")
        sys.stdout.flush()

        p3_sampler = TPESampler(
            n_startup_trials=15, seed=42,
            multivariate=True, group=True,
            consider_endpoints=True, n_ei_candidates=48,
            warn_independent_sampling=False,
        )
        p3_study = optuna.create_study(
            study_name="luscinia_lgbm_str_ua_univ_v1_p3",
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

        for r in p3_obj.all_results:
            if r["name"] not in existing_names:
                all_results.append(r)
                existing_names.add(r["name"])
        print_phase_summary("P3", [r for r in all_results if r.get("phase") == "P3"])

    # ── Training summary (ML lab‐style end‐of‐run analysis) ──────────────────
    wall_elapsed = time.perf_counter() - wall_start
    log_training_summary(all_results, wall_elapsed, SCRIPT_NAME, LOG_FILE)
    print_final_leaderboard(all_results, wall_elapsed)
    sp(f"\nSearch complete. Total trials: {len(all_results)}")

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
        )
    else:
        sp("\n[P4] Ensemble skipped (--skip-ensemble flag).")

    if ensemble_result is not None:
        winning_result = ensemble_result
    else:
        winning_result = sorted(all_results, key=lambda r: r["fitness"], reverse=True)[0]
    sp(f"\n  Winner: {winning_result['name']}  "
       f"(fitness={winning_result['fitness']:.4f}  "
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
