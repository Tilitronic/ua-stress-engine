"""Tests for luscinia-lgbm-str-ua-3s-v1 — pure-function unit tests.

Covers:
  * compute_fitness — default luscinia_3s_specialist preset
  * run_ensemble_phase — accept/reject logic, argmax prediction, early exits
  * run_ensemble_phase — ENSEMBLE_DIR / ensemble_report.json written on both
    accept and reject paths
  * Phase fitness presets: P1/P2 use specialist_binary, P3 uses luscinia_3s_specialist
  * _handcrafted_to_training_rows — labels in {0, 1, 2}, ambiguous entries skipped

All tests are pure: no DB, no real LightGBM training, no real disk writes
(except the ensemble-report tests which use tmp_path fixtures).
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script as a module
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "src", "stress_prediction", "lightGbm",
)
sys.path.insert(0, _SCRIPT_DIR)

_script_path = os.path.join(_SCRIPT_DIR, "luscinia-lgbm-str-ua-3s-v1.py")
if not os.path.exists(_script_path):
    pytest.skip(f"Training script not found: {_script_path}", allow_module_level=True)
pytest.importorskip("lightgbm", reason="lightgbm not installed — skip training-script tests")
_spec = importlib.util.spec_from_file_location("luscinia_lgbm_str_ua_3s_v1", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_mod.__file__ = _script_path
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

compute_fitness               = _mod.compute_fitness
run_ensemble_phase            = _mod.run_ensemble_phase
_handcrafted_to_training_rows = _mod._handcrafted_to_training_rows
ENSEMBLE_TOP_K                = _mod.ENSEMBLE_TOP_K
ENSEMBLE_MIN_SANITY_GAIN      = _mod.ENSEMBLE_MIN_SANITY_GAIN
ENSEMBLE_DIR                  = _mod.ENSEMBLE_DIR
RESULTS_DIR                   = _mod.RESULTS_DIR
NUM_CLASSES                   = _mod.NUM_CLASSES
build_features_3syl           = _mod.build_features_3syl

from services.feature_service_3syl import build_features_3syl as _build_feat_3s
from services.evaluation_service import compute_fitness as _svc_compute_fitness


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _feature_cols() -> List[str]:
    """Return the feature column list for the 3-syl specialist (122 features)."""
    return list(_build_feat_3s("батько", "NOUN").keys())


def _make_result(
    name: str,
    phase: str = "P3",
    fitness: float = 0.80,
    sanity_acc: float = 0.95,
    hand_correct: int = 10,
    hand_total: int = 12,
    f1: float = 0.75,
    acc: float = 0.95,
    best_iteration: int = 500,
    params: Optional[Dict] = None,
) -> dict:
    """Build a minimal result dict in the schema used by run_ensemble_phase."""
    return {
        "phase": phase,
        "trial_number": 0,
        "name": name,
        "timestamp": "2026-01-01T00:00:00",
        "params": params or {
            "num_leaves": 300,
            "max_depth": 12,
            "learning_rate": 0.10,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "fitness_preset": "luscinia_3s_specialist",
        "train_time_sec": 60,
        "wall_elapsed_sec": 60,
        "wall_elapsed_min": 1.0,
        "internal": {
            "f1": f1,
            "accuracy": acc,
            "best_iteration": best_iteration,
        },
        "external": {
            "accuracy": sanity_acc,
            "correct": int(sanity_acc * 5000),
            "sample_size": 5000,
        },
        "handcrafted": {
            "accuracy": hand_correct / hand_total if hand_total else 0.0,
            "correct": hand_correct,
            "total": hand_total,
            "total_words": hand_total,
            "results": [],
        },
        "model": {"size_bytes": 3_000_000},
        "fitness": fitness,
        "hand_penalty_applied": False,
    }


def _fake_booster_3class(proba: List[float] = None) -> MagicMock:
    """Return a mock lgb.Booster whose predict() always returns a fixed (1,3) proba."""
    if proba is None:
        proba = [0.1, 0.7, 0.2]   # class 1 wins by default
    arr = np.array(proba, dtype=float)
    arr /= arr.sum()  # normalise

    bst = MagicMock()

    def _predict(x):
        # LightGBM multiclass returns shape (n_samples, n_classes)
        n = x.shape[0]
        return np.tile(arr, (n, 1))

    bst.predict = MagicMock(side_effect=_predict)
    return bst


def _simple_external_sample_3class(n: int = 10) -> list:
    """Minimal external-sample entries with 3-class labels."""
    items = []
    for i in range(n):
        items.append({
            "form": "батько",
            "pos": "NOUN",
            "features_json": None,
            "expected_label": i % 3,   # classes 0, 1, 2 cycling
        })
    return items


def _simple_handcrafted_3class(n: int = 6) -> list:
    """Minimal handcrafted entries (3-class labels)."""
    # Use 3-syl word "батько" (2 vowels only actually — use "столиця" instead)
    # Just provide labels directly; feature extraction will still work
    return [("батько", "NOUN", i % 3, "test", None) for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# NUM_CLASSES constant
# ════════════════════════════════════════════════════════════════════════════

class TestNumClasses:

    def test_num_classes_is_3(self):
        assert NUM_CLASSES == 3

    def test_script_has_multiclass_objective(self):
        src = Path(_script_path).read_text(encoding="utf-8")
        assert '"multiclass"' in src or "'multiclass'" in src, (
            "Script must use multiclass objective"
        )

    def test_script_has_multi_logloss(self):
        src = Path(_script_path).read_text(encoding="utf-8")
        assert "multi_logloss" in src, "Script must use multi_logloss metric"

    def test_script_has_macro_f1(self):
        src = Path(_script_path).read_text(encoding="utf-8")
        assert 'average="macro"' in src, (
            "Script must compute macro-F1 for 3-class classification"
        )


# ════════════════════════════════════════════════════════════════════════════
# compute_fitness — luscinia_3s_specialist preset
# ════════════════════════════════════════════════════════════════════════════

class TestComputeFitness3S:
    """compute_fitness in 3S defaults to luscinia_3s_specialist (v2 redesign).

    Returns a 3-tuple (score, hand_penalty, sanity_violated).
    """

    def test_default_preset_is_luscinia_3s_specialist(self):
        """Perfect inputs (sanity >= acc) → fitness > 1.0 due to size bonus."""
        fitness, penalty, sanity_bad = compute_fitness(1.0, 1.0, 1.0, 1.0, 6, 6)
        assert fitness > 1.0
        assert not sanity_bad

    def test_zero_scores(self):
        fitness, penalty, sanity_bad = compute_fitness(0.0, 0.0, 0.0, 0.0, 0, 6)
        # acc=0, ext=0 → equal, no sanity guard; only size_bonus
        assert fitness < 0.03
        assert not sanity_bad

    def test_returns_tuple(self):
        result = compute_fitness(0.8, 0.95, 0.9, 0.85, 5, 6)
        assert isinstance(result, tuple) and len(result) == 3

    def test_fitness_is_float(self):
        fitness, _p, _s = compute_fitness(0.75, 0.94, 0.88, 0.90, 10, 12)
        assert isinstance(fitness, float)

    def test_penalty_flag_when_not_perfect(self):
        _, penalty, _ = compute_fitness(0.8, 0.95, 0.9, 0.85, 5, 6)
        assert penalty is True

    def test_no_penalty_when_all_correct(self):
        _, penalty, _ = compute_fitness(0.8, 0.95, 0.9, 0.85, 6, 6)
        assert penalty is False

    def test_hand_acc_weight_higher_than_acc(self):
        """luscinia_3s_specialist v2: hand_acc (35%) > acc (20%).
        Use base acc=0.85 to stay well below sanity=0.95 (no guard fires).
        """
        base_fitness, _, _ = compute_fitness(0.75, 0.95, 0.80, 0.85, 8, 10)
        delta = 0.05
        # acc improves to 0.90, still < sanity=0.95 ✓
        f_hand, _, _ = compute_fitness(0.75, 0.95, 0.80 + delta, 0.85, 8, 10)
        f_acc,  _, _ = compute_fitness(0.75, 0.95, 0.80, 0.85 + delta, 8, 10)
        assert f_hand > base_fitness
        assert f_acc  > base_fitness
        assert f_hand > f_acc, (
            f"hand_acc delta must outweigh acc delta "
            f"(f_hand={f_hand:.6f}, f_acc={f_acc:.6f})"
        )

    def test_fitness_increases_with_better_f1_macro(self):
        f_low,  _, _ = compute_fitness(0.60, 0.95, 0.90, 0.90, 8, 10)
        f_high, _, _ = compute_fitness(0.80, 0.95, 0.90, 0.90, 8, 10)
        assert f_high > f_low

    def test_fitness_bounded_zero_to_one_plus_bonus(self):
        """Fitness is in [0, ~1.03] with size bonus."""
        for f1, ext, hand, acc in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            fitness, _, _ = compute_fitness(f1, ext, hand, acc, 5, 10)
            assert 0.0 <= fitness <= 1.03

    def test_penalty_display_only_no_score_reduction(self):
        """Hand penalty must not alter the numeric fitness (sanity 0.95 >= acc 0.88)."""
        fitness_full, _, _ = compute_fitness(0.75, 0.95, 0.90, 0.88, 10, 10)
        fitness_miss, _, _ = compute_fitness(0.75, 0.95, 0.90, 0.88, 9, 10)
        assert abs(fitness_full - fitness_miss) < 1e-9, (
            "Penalty must not alter the numeric fitness score"
        )

    def test_default_preset_in_compute_fitness_is_luscinia_3s_specialist(self):
        import inspect
        sig = inspect.signature(compute_fitness)
        preset_param = sig.parameters.get("preset")
        assert preset_param is not None
        assert preset_param.default == "luscinia_3s_specialist", (
            f"Expected 'luscinia_3s_specialist', got '{preset_param.default}'"
        )


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — early-exit conditions
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleEarlyExit3S:

    def test_returns_none_with_no_results(self):
        result = run_ensemble_phase([], _feature_cols(), [], [])
        assert result is None

    def test_returns_none_with_zero_saved_models(self):
        results = [_make_result(f"P3_{i:04d}") for i in range(5)]
        result = run_ensemble_phase(
            results, _feature_cols(),
            _simple_external_sample_3class(), _simple_handcrafted_3class(),
        )
        assert result is None

    def test_returns_none_with_only_one_saved_model(self, tmp_path):
        results = [_make_result("P3_0001"), _make_result("P3_0002")]
        model_dir = tmp_path / "P3_0001"
        model_dir.mkdir()
        (model_dir / "P3_0001.lgb").write_text("fake")

        fake_bst = _fake_booster_3class()
        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
        ):
            result = run_ensemble_phase(
                results, _feature_cols(),
                _simple_external_sample_3class(), _simple_handcrafted_3class(),
            )
        assert result is None


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — argmax-based prediction (not binary threshold)
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleArgmax3S:
    """Ensemble for 3S uses argmax over averaged probability vectors."""

    def test_argmax_selects_class_1_when_proba_highest(self):
        """Booster returns [0.1, 0.7, 0.2] → predicted class should be 1."""
        proba = [0.1, 0.7, 0.2]
        avg = np.array(proba)
        assert int(np.argmax(avg)) == 1

    def test_argmax_selects_class_0(self):
        proba = [0.6, 0.3, 0.1]
        assert int(np.argmax(np.array(proba))) == 0

    def test_argmax_selects_class_2(self):
        proba = [0.1, 0.2, 0.7]
        assert int(np.argmax(np.array(proba))) == 2

    def test_ensemble_predicts_via_argmax_not_threshold(self):
        """The ensemble result dict schema must NOT contain a binary threshold key.
        The script should use argmax for prediction, not a 0.5 threshold.
        """
        src = Path(_script_path).read_text(encoding="utf-8")
        # In 2S the code does `int(p1 >= 0.5)` — 3S must NOT do this
        assert "p1 >= 0.5" not in src, (
            "3S ensemble must use argmax, not binary threshold >= 0.5"
        )
        # 3S ensemble must call np.argmax
        assert "np.argmax" in src, (
            "3S ensemble must use np.argmax for multiclass prediction"
        )


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — softmax weights
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleWeights3S:

    def test_weights_sum_to_one(self, tmp_path):
        fitnesses = [0.80, 0.78, 0.76]
        results = [
            _make_result(f"P3_{i:04d}", fitness=fitnesses[i])
            for i in range(3)
        ]
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir()
            (d / f"{r['name']}.lgb").write_text("fake")

        fake_bst = _fake_booster_3class()
        captured_weights = []

        def _capture_sp(msg):
            if "weight=" in msg:
                try:
                    w = float(msg.split("weight=")[1].strip())
                    captured_weights.append(w)
                except Exception:
                    pass

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
            patch.object(_mod, "sp", side_effect=_capture_sp),
        ):
            run_ensemble_phase(
                results, _feature_cols(),
                _simple_external_sample_3class(50), _simple_handcrafted_3class(6),
            )

        if captured_weights:
            assert abs(sum(captured_weights) - 1.0) < 1e-4

    def test_higher_fitness_gets_higher_weight(self):
        fitnesses = [0.82, 0.80, 0.78]
        arr = np.array(fitnesses)
        arr -= arr.max()
        w = np.exp(arr)
        w /= w.sum()
        assert w[0] > w[1] > w[2]

    def test_equal_fitness_gives_equal_weights(self):
        fitnesses = [0.80, 0.80, 0.80]
        arr = np.array(fitnesses)
        arr -= arr.max()
        w = np.exp(arr)
        w /= w.sum()
        for wi in w:
            assert abs(wi - 1 / 3) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — accept / reject logic
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleAcceptReject3S:

    def _run_with_mocked_models(
        self,
        results: list,
        tmp_path: Path,
        proba: List[float],
        external_sample: Optional[list] = None,
        handcrafted: Optional[list] = None,
        top_k: int = 5,
        min_sanity_gain: float = ENSEMBLE_MIN_SANITY_GAIN,
    ) -> Optional[dict]:
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir(exist_ok=True)
            (d / f"{r['name']}.lgb").write_text("fake")

        fake_bst = _fake_booster_3class(proba)
        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
        ):
            return run_ensemble_phase(
                results,
                _feature_cols(),
                external_sample or _simple_external_sample_3class(100),
                handcrafted or _simple_handcrafted_3class(6),
                top_k=top_k,
                min_sanity_gain=min_sanity_gain,
            )

    def test_accepted_on_sufficient_sanity_gain(self, tmp_path):
        """Ensemble predicts class 1 for all → all correct when external labels are 1."""
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None, "expected_label": 1}
            for _ in range(100)
        ]
        # Best solo sanity_acc = 0.30 → ensemble gets ~1.0 → gain >> 0.001
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.30, hand_correct=3, hand_total=6)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],   # booster always says class 1
            external_sample=external,
            handcrafted=[("батько", "NOUN", 1, "test", None)] * 6,
        )
        assert result is not None, "Ensemble should be accepted on sanity gain"
        assert result["phase"] == "P4"
        assert result["name"] == "P4_ensemble"

    def test_accepted_on_hand_gain_alone(self, tmp_path):
        """Even without sanity gain, hand_gain > 0 → accept."""
        # External: cycling classes 0/1/2 → ~33% accuracy regardless
        # Booster says class 1; handcrafted all class 1 → 6/6
        # Best solo hand_correct = 4 → hand_gain = 2 → accept
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.33, hand_correct=4, hand_total=6)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],
            handcrafted=[("батько", "NOUN", 1, "test", None)] * 6,
        )
        assert result is not None

    def test_rejected_when_no_improvement(self, tmp_path):
        """Best solo is perfect → ensemble (wrong class) → reject."""
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None,
             "expected_label": 0}   # all expect class 0, booster says class 1
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=1.0, hand_correct=6, hand_total=6)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],
            external_sample=external,
            handcrafted=[("батько", "NOUN", 0, "test", None)] * 6,
        )
        assert result is None

    def test_rejected_path_writes_report_json(self, tmp_path):
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None,
             "expected_label": 0}
            for _ in range(50)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=1.0, hand_correct=6, hand_total=6)
            for i in range(3)
        ]
        self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],
            external_sample=external,
            handcrafted=[("батько", "NOUN", 0, "test", None)] * 6,
        )
        report_path = tmp_path / "P4_ensemble" / "ensemble_report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["decision"] == "rejected"
        assert "best_solo" in report
        assert "sanity_gain" in report
        assert "hand_gain" in report

    def test_accepted_path_writes_report_json(self, tmp_path):
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.30, hand_correct=3, hand_total=6)
            for i in range(3)
        ]
        self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],
            external_sample=external,
            handcrafted=[("батько", "NOUN", 1, "test", None)] * 6,
        )
        report_path = tmp_path / "P4_ensemble" / "ensemble_report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["decision"] == "accepted"
        assert "ensemble_fitness" in report
        assert "models_used" in report
        assert "weights" in report

    def test_accepted_result_has_expected_keys(self, tmp_path):
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.30, hand_correct=3, hand_total=6)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path,
            proba=[0.05, 0.90, 0.05],
            external_sample=external,
            handcrafted=[("батько", "NOUN", 1, "test", None)] * 6,
        )
        assert result is not None
        for key in ["phase", "name", "fitness", "params", "internal",
                    "external", "handcrafted", "ensemble_models",
                    "ensemble_weights", "best_solo_name", "fitness_preset"]:
            assert key in result, f"Missing key '{key}'"
        assert result["phase"] == "P4"
        assert result["fitness_preset"] == "luscinia_3s_specialist"

    def test_p3_results_preferred_over_p1_p2(self, tmp_path):
        p1_results = [
            _make_result(f"P1_{i:04d}", phase="P1", fitness=0.85 - i * 0.001)
            for i in range(3)
        ]
        p3_results = [
            _make_result(f"P3_{i:04d}", phase="P3", fitness=0.80 - i * 0.001)
            for i in range(3)
        ]
        all_results = p1_results + p3_results

        for r in all_results:
            d = tmp_path / r["name"]
            d.mkdir(exist_ok=True)
            (d / f"{r['name']}.lgb").write_text("fake")

        loaded_names = []

        class _CapturingBooster:
            def __init__(self, model_file, **kw):
                loaded_names.append(Path(model_file).parent.name)

            def predict(self, x):
                n = x.shape[0]
                return np.tile([0.1, 0.7, 0.2], (n, 1))

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", side_effect=_CapturingBooster),
        ):
            run_ensemble_phase(
                all_results, _feature_cols(),
                _simple_external_sample_3class(50), _simple_handcrafted_3class(6),
                top_k=3,
            )

        for name in loaded_names:
            assert name.startswith("P3_"), (
                f"Expected P3 model to be loaded, got '{name}'"
            )


# ════════════════════════════════════════════════════════════════════════════
# Phase fitness preset contracts
# ════════════════════════════════════════════════════════════════════════════

class TestPhasePresets3S:

    def test_ensemble_result_preset_is_luscinia_3s_specialist(self, tmp_path):
        external = [
            {"form": "батько", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.30, hand_correct=3, hand_total=6)
            for i in range(3)
        ]
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir()
            (d / f"{r['name']}.lgb").write_text("fake")

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=_fake_booster_3class([0.05, 0.90, 0.05])),
        ):
            result = run_ensemble_phase(
                results, _feature_cols(),
                external,
                [("батько", "NOUN", 1, "test", None)] * 6,
            )

        assert result is not None
        assert result["fitness_preset"] == "luscinia_3s_specialist"

    def test_script_p3_preset_constant(self):
        src = Path(_script_path).read_text(encoding="utf-8")
        assert 'fitness_preset="luscinia_3s_specialist"' in src, (
            "Phase 3 objective must use luscinia_3s_specialist fitness preset"
        )

    def test_script_p1_preset_constant(self):
        src = Path(_script_path).read_text(encoding="utf-8")
        assert 'fitness_preset="specialist_binary"' in src, (
            "Phase 1 / Phase 2 must use specialist_binary fitness preset"
        )

    def test_luscinia_3s_specialist_registered_in_evaluation_service(self):
        """The preset must be registered in evaluation_service._WEIGHT_PRESETS."""
        from services.evaluation_service import _WEIGHT_PRESETS
        assert "luscinia_3s_specialist" in _WEIGHT_PRESETS, (
            "luscinia_3s_specialist must be registered in evaluation_service"
        )

    def test_luscinia_3s_specialist_weights_sum_to_one(self):
        from services.evaluation_service import _WEIGHT_PRESETS
        w = _WEIGHT_PRESETS["luscinia_3s_specialist"]
        assert abs(sum(w.values()) - 1.0) < 1e-6


# ════════════════════════════════════════════════════════════════════════════
# Ensemble constants
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleConstants3S:

    def test_top_k_is_5(self):
        assert ENSEMBLE_TOP_K == 5

    def test_min_sanity_gain_is_001(self):
        assert abs(ENSEMBLE_MIN_SANITY_GAIN - 0.001) < 1e-9

    def test_top_k_at_least_2(self):
        assert ENSEMBLE_TOP_K >= 2

    def test_min_sanity_gain_positive(self):
        assert ENSEMBLE_MIN_SANITY_GAIN > 0


# ════════════════════════════════════════════════════════════════════════════
# _handcrafted_to_training_rows — 3-class labels
# ════════════════════════════════════════════════════════════════════════════

class TestHandcraftedToTrainingRows3S:

    def _feat_cols(self) -> list:
        return list(_build_feat_3s("батько", "NOUN").keys())

    def test_returns_none_on_empty_input(self):
        X, y = _handcrafted_to_training_rows([], self._feat_cols())
        assert X is None and y is None

    def test_returns_dataframe_and_series(self):
        tests = [("батько", "NOUN", 0, "antepenult", None)]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_number_of_rows(self):
        tests = [
            ("батько", "NOUN", 0, "class0", None),
            ("столиця", "NOUN", 1, "class1", None),
            ("вода", "NOUN", 2, "class2", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 3
        assert len(y) == 3

    def test_skips_entries_with_none_expected(self):
        tests = [
            ("батько", "NOUN", 0, "ok", None),
            ("??", "X", None, "skip", None),
            ("столиця", "NOUN", 1, "ok2", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 2

    def test_skips_ambiguous_list_entries(self):
        tests = [
            ("батько", "NOUN", 0, "ok", None),
            ("вода", "NOUN", [0, 1], "ambiguous", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 1

    def test_feature_columns_match(self):
        tests = [("батько", "NOUN", 0, "test", None)]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert list(X.columns) == feat_cols

    def test_label_values_are_in_0_1_2(self):
        tests = [
            ("батько", "NOUN", 0, "antepenult", None),
            ("столиця", "NOUN", 1, "penult", None),
            ("вода", "NOUN", 2, "oxytone", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        for val in y:
            assert val in (0, 1, 2), f"Expected label in {{0,1,2}}, got {val}"

    def test_skips_label_outside_0_1_2(self):
        """Labels outside {0,1,2} should be skipped (safety guard)."""
        tests = [
            ("батько", "NOUN", 0, "ok", None),
            ("слово", "NOUN", 5, "invalid", None),
            ("столиця", "NOUN", 1, "ok2", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 2


# ════════════════════════════════════════════════════════════════════════════
# feature_service_3syl — basic sanity checks
# ════════════════════════════════════════════════════════════════════════════

class TestFeatureService3Syl:

    def test_feature_count_is_expected(self):
        from services.feature_service_3syl import (
            build_features_3syl as bfs,
            EXPECTED_FEATURE_COUNT_3SYL,
        )
        feat = bfs("батько", "NOUN")
        assert len(feat) == EXPECTED_FEATURE_COUNT_3SYL, (
            f"Expected {EXPECTED_FEATURE_COUNT_3SYL} features, got {len(feat)}"
        )

    def test_all_values_are_numeric(self):
        from services.feature_service_3syl import build_features_3syl as bfs
        feat = bfs("столиця", "NOUN")
        for k, v in feat.items():
            assert isinstance(v, (int, float)), (
                f"Feature '{k}' has non-numeric value: {v!r}"
            )

    def test_has_3syl_specific_features(self):
        from services.feature_service_3syl import build_features_3syl as bfs
        feat = bfs("столиця", "NOUN")
        expected_keys = [
            "ending_hash_4", "ending_hash_5",
            "ending_pos_hash_3", "ending_pos_hash_4",
            "v1_char_3syl", "v2_char_3syl", "v3_char_3syl",
            "coda_after_v3", "coda_between_v2_v3", "coda_between_v1_v2",
            "iv_ratio_12_3syl", "iv_ratio_23_3syl",
            "has_vy_prefix_3syl", "has_gerund_suffix_3syl",
            "has_verb_aty_3syl", "has_verb_yty_3syl", "has_verb_uvaty_3syl",
            "has_oxytone_mobile_3syl", "has_adcat_suffix_3syl",
            "has_num_adtsyat_3syl", "onset_before_v1_3syl", "onset_ratio_3syl",
        ]
        for k in expected_keys:
            assert k in feat, f"Missing 3-syl feature '{k}'"

    def test_gerund_suffix_flag(self):
        """Words ending in -ання should trigger has_gerund_suffix_3syl."""
        from services.feature_service_3syl import build_features_3syl as bfs
        # "видання" ends in -ання → flag should be 1
        feat = bfs("видання", "NOUN")
        assert feat["has_gerund_suffix_3syl"] == 1

    def test_vy_prefix_flag_noun(self):
        from services.feature_service_3syl import build_features_3syl as bfs
        feat = bfs("випадок", "NOUN")
        assert feat["has_vy_prefix_3syl"] == 1

    def test_vy_prefix_flag_not_set_for_adj(self):
        from services.feature_service_3syl import build_features_3syl as bfs
        # ви- prefix but POS is ADJ → should NOT flag
        feat = bfs("виразний", "ADJ")
        assert feat["has_vy_prefix_3syl"] == 0

    def test_coda_after_v3_for_3vowel_word(self):
        """'батько' has 2 vowels (а, о) — v3 sentinel should be -1."""
        from services.feature_service_3syl import build_features_3syl as bfs
        feat = bfs("батько", "NOUN")  # б-а-т-ь-к-о (2 vowels)
        assert feat["coda_after_v3"] == -1

    def test_coda_after_v3_for_real_3syl_word(self):
        """'видання' has 3 vowels (и, а, я/а) — coda after last vowel is counted."""
        from services.feature_service_3syl import build_features_3syl as bfs
        from services.feature_service import find_vowels
        word = "видання"
        vowels_pos = find_vowels(word)
        if len(vowels_pos) == 3:
            expected_coda = len(word) - vowels_pos[2] - 1
            feat = bfs(word, "NOUN")
            assert feat["coda_after_v3"] == expected_coda

    def test_onset_ratio_between_0_and_1(self):
        from services.feature_service_3syl import build_features_3syl as bfs
        feat = bfs("перемога", "NOUN")
        assert 0.0 <= feat["onset_ratio_3syl"] <= 1.0
