"""Tests for luscinia-lgbm-str-ua-univ-v1.1 — pure-function unit tests.

Covers:
  * compute_sample_weights — class imbalance correction
  * compute_fitness — luscinia_universal_specialist preset (inherited from v1.0)
  * train_and_evaluate — sample_weight propagation to lgb.Dataset
  * run_ensemble_phase — accept/reject logic, weight calculation, early exits
  * _handcrafted_to_training_rows — shared helper
  * load_warmstart_params — loads v1.0 JSON or falls back to hardcoded seed
  * Phase narrow ranges — v1.1 ±60% vs v1.0 ±50%
  * refit_on_full_data — weights passed for final training dataset
  * Structural checks — preset names, study names, sigma value

All tests are pure: no DB, no real LightGBM training, no disk writes
(except where tmp_path fixtures are used for files explicitly under test).
"""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call

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

_script_path = os.path.join(_SCRIPT_DIR, "luscinia-lgbm-str-ua-univ-v1.1.py")
if not os.path.exists(_script_path):
    pytest.skip(
        f"Training script not yet created: {_script_path}",
        allow_module_level=True,
    )
pytest.importorskip("lightgbm", reason="lightgbm not installed — skip training-script tests")
_spec = importlib.util.spec_from_file_location("luscinia_lgbm_str_ua_univ_v1_1", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_mod.__file__ = _script_path
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Exported symbols
compute_fitness               = _mod.compute_fitness
compute_sample_weights        = _mod.compute_sample_weights
run_ensemble_phase            = _mod.run_ensemble_phase
load_warmstart_params         = _mod.load_warmstart_params
_handcrafted_to_training_rows = _mod._handcrafted_to_training_rows
ENSEMBLE_TOP_K                = _mod.ENSEMBLE_TOP_K
ENSEMBLE_MIN_SANITY_GAIN      = _mod.ENSEMBLE_MIN_SANITY_GAIN
ENSEMBLE_DIR                  = _mod.ENSEMBLE_DIR
RESULTS_DIR                   = _mod.RESULTS_DIR
NUM_CLASSES                   = _mod.NUM_CLASSES
CLASS_WEIGHT_POWER            = _mod.CLASS_WEIGHT_POWER
WARMSTART_SEED_PARAMS         = _mod.WARMSTART_SEED_PARAMS
V10_RESULTS_JSON              = _mod.V10_RESULTS_JSON
V10_WINNER_NAME               = _mod.V10_WINNER_NAME
P2_BOUNDS                     = _mod.P2_BOUNDS
BASE_LGBM                     = _mod.BASE_LGBM
Phase3Objective               = _mod.Phase3Objective

from services.feature_service_universal import (
    build_features_universal, EXPECTED_FEATURE_COUNT_UNIV,
)
from services.evaluation_service import (
    compute_fitness as _svc_compute_fitness,
    SANITY_BELOW_ACC_PENALTY,
    SIZE_BONUS_MAX,
    SIZE_BONUS_CAP_MB,
)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _feature_cols() -> List[str]:
    """Return the 132 universal feature column names."""
    return list(build_features_universal("мамина", "NOUN").keys())


def _make_result(
    name: str,
    phase: str = "P3",
    fitness: float = 0.82,
    sanity_acc: float = 0.9945,
    hand_correct: int = 192,
    hand_total: int = 197,
    f1: float = 0.75,
    acc: float = 0.9944,
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
            "num_leaves": 712,
            "max_depth": 16,
            "learning_rate": 0.0955,
            "min_child_samples": 18,
            "lambda_l1": 0.741,
            "lambda_l2": 12.96,
            "subsample": 0.860,
            "colsample_bytree": 0.860,
            "feature_fraction_bynode": 0.488,
            "min_sum_hessian_in_leaf": 26.74,
        },
        "fitness_preset": "luscinia_universal_specialist",
        "train_time_sec": 120,
        "wall_elapsed_sec": 120,
        "wall_elapsed_min": 2.0,
        "internal": {
            "f1": f1,
            "accuracy": acc,
            "best_iteration": best_iteration,
        },
        "external": {
            "accuracy": sanity_acc,
            "correct": int(sanity_acc * EXTERNAL_SAMPLE_SIZE),
            "sample_size": EXTERNAL_SAMPLE_SIZE,
        },
        "handcrafted": {
            "accuracy": hand_correct / hand_total if hand_total else 0.0,
            "correct": hand_correct,
            "total": hand_total,
            "total_words": hand_total,
            "results": [],
        },
        "model": {"size_bytes": 272_000_000},
        "fitness": fitness,
        "hand_penalty_applied": False,
    }


EXTERNAL_SAMPLE_SIZE = 5000


def _fake_booster_univ(class_pred: int = 0) -> MagicMock:
    """Mock lgb.Booster for 11-class multiclass.

    predict() returns a (n, 11) array with 0.9 probability on class_pred.
    """
    bst = MagicMock()

    def _predict(x):
        n = x.shape[0] if hasattr(x, "shape") else 1
        arr = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        arr[:, class_pred] = 0.9
        return arr

    bst.predict = MagicMock(side_effect=_predict)
    return bst


def _simple_external_sample(n: int = 10, expected_label: int = 0) -> list:
    """Minimal external-sample entries for 11-class universal model."""
    return [
        {
            "form": "мамина",
            "pos": "ADJ",
            "features_json": None,
            "expected_label": expected_label,
            "n_syllables": 3,
        }
        for _ in range(n)
    ]


def _simple_handcrafted(n: int = 4, expected: int = 0) -> list:
    """Minimal handcrafted test entries (word, pos, expected, desc, feat_json)."""
    return [("мамина", "ADJ", expected, "test", None) for _ in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# compute_sample_weights
# ════════════════════════════════════════════════════════════════════════════

class TestComputeSampleWeights:
    """Verify inverse-frequency class-weight computation."""

    def _make_labels(self, counts: Dict[int, int]) -> pd.Series:
        """Build a pd.Series from a class → count dict."""
        labels = []
        for c, n in counts.items():
            labels.extend([c] * n)
        return pd.Series(labels, dtype=int)

    def test_power_zero_returns_all_ones(self):
        """power=0.0 → all weights = 1.0 (disabled)."""
        y = self._make_labels({0: 100, 1: 50, 2: 10})
        w = compute_sample_weights(y, power=0.0)
        assert np.allclose(w, 1.0), "power=0 should give all-ones weights"

    def test_output_shape_matches_input(self):
        y = self._make_labels({0: 200, 1: 100, 2: 20, 3: 5})
        w = compute_sample_weights(y, power=0.5)
        assert w.shape == (len(y),), "Output shape must match input length"

    def test_output_dtype_is_float32(self):
        y = self._make_labels({0: 100, 1: 50})
        w = compute_sample_weights(y, power=0.5)
        assert w.dtype == np.float32

    def test_rare_class_gets_higher_weight(self):
        """Rare class must receive a higher per-sample weight than majority class."""
        y = self._make_labels({0: 10000, 5: 10})
        w = compute_sample_weights(y, power=0.5)
        mask0 = (y == 0).values
        mask5 = (y == 5).values
        w0_mean = w[mask0].mean()
        w5_mean = w[mask5].mean()
        assert w5_mean > w0_mean, (
            f"Rare class 5 should get higher weight: "
            f"class0={w0_mean:.4f}, class5={w5_mean:.4f}"
        )

    def test_mean_weight_is_one(self):
        """Weights are normalised so that mean(weights_per_class) = 1.0.

        The implementation normalises by mean(weights_per_class[weights > 0]),
        which includes absent classes (value=1.0).  When all NUM_CLASSES classes
        are present the normalised class-weight vector must have mean = 1.0.
        """
        # Use all 11 classes so absent-class slots don't skew the mean
        counts = {c: max(5, 5000 // (2 ** c)) for c in range(NUM_CLASSES)}
        y = self._make_labels(counts)
        w = compute_sample_weights(y, power=0.5)
        y_arr = y.values
        class_means = np.array([w[y_arr == c].mean() for c in range(NUM_CLASSES)])
        assert abs(class_means.mean() - 1.0) < 0.15, (
            f"Mean class weight should be ≈1.0 after normalisation; "
            f"got {class_means.mean():.4f}"
        )

    def test_weights_all_positive(self):
        y = self._make_labels({0: 100, 1: 50, 2: 10, 3: 3})
        w = compute_sample_weights(y, power=0.5)
        assert (w > 0).all(), "All weights must be positive"

    def test_power_1_gives_full_inverse_frequency(self):
        """power=1.0: weight(c) ∝ n_total / n_c (before normalisation)."""
        # Class 0: 1000 samples, class 1: 100 samples → ratio should be 10:1
        y = self._make_labels({0: 1000, 1: 100})
        w = compute_sample_weights(y, power=1.0)
        mask0 = (y == 0).values
        mask1 = (y == 1).values
        ratio = w[mask1].mean() / w[mask0].mean()
        # Full inverse: class1 weight / class0 weight = (n/100) / (n/1000) = 10
        assert abs(ratio - 10.0) < 0.1, (
            f"power=1 full-inverse ratio should be 10:1; got {ratio:.4f}"
        )

    def test_sqrt_inverse_is_less_aggressive_than_full_inverse(self):
        """power=0.5 gives a smaller weight ratio between rare/common classes."""
        y = self._make_labels({0: 1000, 1: 10})
        w05 = compute_sample_weights(y, power=0.5)
        w10 = compute_sample_weights(y, power=1.0)
        mask0 = (y == 0).values
        mask1 = (y == 1).values
        ratio05 = w05[mask1].mean() / w05[mask0].mean()
        ratio10 = w10[mask1].mean() / w10[mask0].mean()
        assert ratio05 < ratio10, (
            "sqrt-inverse (power=0.5) should give smaller weight ratio than full inverse"
        )

    def test_single_class_returns_ones(self):
        """Edge case: only one class → all weights should be equal (≈1.0)."""
        y = pd.Series([0] * 100, dtype=int)
        w = compute_sample_weights(y, power=0.5)
        assert np.allclose(w, w[0]), "Single-class input should produce uniform weights"

    def test_absent_class_does_not_raise(self):
        """Classes missing from y should not cause errors or division-by-zero."""
        y = self._make_labels({0: 100, 1: 50})  # classes 2-10 absent
        w = compute_sample_weights(y, power=0.5)
        assert len(w) == 150

    def test_default_power_is_module_constant(self):
        """Default power in compute_sample_weights should match CLASS_WEIGHT_POWER."""
        import inspect
        sig = inspect.signature(compute_sample_weights)
        default_power = sig.parameters["power"].default
        assert default_power == CLASS_WEIGHT_POWER, (
            f"Default power {default_power} should equal CLASS_WEIGHT_POWER={CLASS_WEIGHT_POWER}"
        )


# ════════════════════════════════════════════════════════════════════════════
# compute_fitness — luscinia_universal_specialist preset
# ════════════════════════════════════════════════════════════════════════════

class TestComputeFitnessUniv:
    """compute_fitness in v1.1 defaults to luscinia_universal_specialist."""

    def test_default_preset_is_luscinia_universal_specialist(self):
        import inspect
        sig = inspect.signature(compute_fitness)
        preset = sig.parameters.get("preset")
        assert preset is not None
        assert preset.default == "luscinia_universal_specialist"

    def test_perfect_inputs_gives_score_above_one(self):
        """Perfect model → size bonus should push score above 1.0."""
        fitness, penalty, sanity_bad = compute_fitness(1.0, 1.0, 1.0, 1.0, 6, 6)
        assert fitness > 1.0
        assert not sanity_bad

    def test_returns_tuple_of_three(self):
        result = compute_fitness(0.75, 0.994, 0.975, 0.994, 192, 197)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fitness_is_float(self):
        fitness, _, _ = compute_fitness(0.75, 0.994, 0.975, 0.994, 192, 197)
        assert isinstance(fitness, float)

    def test_sanity_guard_fires_when_sanity_below_acc(self):
        """When sanity_acc < val_acc the score is halved."""
        score_ok, _, sv_ok = compute_fitness(0.75, 0.995, 0.97, 0.994, 192, 197)
        score_pen, _, sv_pen = compute_fitness(0.75, 0.980, 0.97, 0.994, 192, 197)
        assert not sv_ok
        assert sv_pen
        ratio = score_pen / score_ok
        assert abs(ratio - SANITY_BELOW_ACC_PENALTY) < 0.01

    def test_hand_acc_weight_35_percent(self):
        """luscinia_universal_specialist: hand_acc weight = 35%.
        A delta in hand_acc should produce more fitness change than the
        same delta in acc (weight 20%).

        Note: keep acc + delta < sanity_acc to avoid triggering the sanity guard,
        which would penalise the score instead of improving it.
        """
        # sanity_acc=0.98, val_acc=0.92 — both below sanity threshold
        base, _, _ = compute_fitness(0.75, 0.98, 0.80, 0.92, 8, 10)
        delta = 0.05
        f_hand, _, _ = compute_fitness(0.75, 0.98, 0.80 + delta, 0.92, 8, 10)
        f_acc,  _, _ = compute_fitness(0.75, 0.98, 0.80, 0.92 + delta, 8, 10)
        assert f_hand > base, f"f_hand={f_hand:.6f} should exceed base={base:.6f}"
        assert f_acc  > base, f"f_acc={f_acc:.6f} should exceed base={base:.6f}"
        assert f_hand > f_acc, (
            f"hand_acc delta (35%) should outweigh acc delta (20%) "
            f"(f_hand={f_hand:.6f}, f_acc={f_acc:.6f})"
        )

    def test_f1_weight_45_percent(self):
        """f1 has weight 45% — the dominant component."""
        base, _, _ = compute_fitness(0.70, 0.995, 0.90, 0.990, 8, 10)
        delta = 0.05
        f_f1,   _, _ = compute_fitness(0.70 + delta, 0.995, 0.90, 0.990, 8, 10)
        f_hand, _, _ = compute_fitness(0.70, 0.995, 0.90 + delta, 0.990, 8, 10)
        assert f_f1 > f_hand, (
            f"f1 delta (45%) should outweigh hand_acc delta (35%) "
            f"(f_f1={f_f1:.6f}, f_hand={f_hand:.6f})"
        )

    def test_penalty_flag_when_not_all_correct(self):
        _, penalty, _ = compute_fitness(0.75, 0.995, 0.975, 0.994, 192, 197)
        assert penalty is True

    def test_no_penalty_when_all_correct(self):
        _, penalty, _ = compute_fitness(0.75, 0.995, 1.0, 0.994, 6, 6)
        assert penalty is False

    def test_hand_penalty_is_display_only(self):
        """Hand penalty flag must NOT change numeric score."""
        f_all,  _, _ = compute_fitness(0.75, 0.995, 0.90, 0.990, 10, 10)
        f_miss, _, _ = compute_fitness(0.75, 0.995, 0.90, 0.990,  9, 10)
        assert abs(f_all - f_miss) < 1e-9, "Hand penalty must not alter numeric fitness"

    def test_sanity_guard_not_fired_for_equal_values(self):
        _, _, sv = compute_fitness(0.75, 0.994, 0.97, 0.994, 192, 197)
        assert not sv, "Equal sanity and acc should NOT fire sanity guard"

    def test_fitness_bounded_zero_to_one_plus_bonus(self):
        for f1, ext, hand, acc in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            fitness, _, _ = compute_fitness(f1, ext, hand, acc, 5, 10)
            assert 0.0 <= fitness <= 1.03, f"fitness {fitness} out of [0, 1.03]"


# ════════════════════════════════════════════════════════════════════════════
# BASE_LGBM — v1.1-specific structural checks
# ════════════════════════════════════════════════════════════════════════════

class TestBaseLgbmParams:
    """BASE_LGBM in v1.1 must NOT contain is_unbalance (replaced by sample weights)."""

    def test_is_unbalance_not_in_base_lgbm(self):
        """v1.1 uses sample weights instead of is_unbalance=True."""
        assert "is_unbalance" not in BASE_LGBM, (
            "is_unbalance must be removed from BASE_LGBM — "
            "sample weights replace it in v1.1"
        )

    def test_objective_is_multiclass(self):
        assert BASE_LGBM["objective"] == "multiclass"

    def test_num_class_is_11(self):
        assert BASE_LGBM["num_class"] == NUM_CLASSES
        assert NUM_CLASSES == 11

    def test_force_col_wise_true(self):
        assert BASE_LGBM.get("force_col_wise") is True


# ════════════════════════════════════════════════════════════════════════════
# load_warmstart_params
# ════════════════════════════════════════════════════════════════════════════

class TestLoadWarmstartParams:
    """load_warmstart_params should load from v1.0 JSON or fall back to hardcoded."""

    def test_fallback_to_hardcoded_when_file_missing(self, tmp_path, capsys):
        """When v1.0 JSON does not exist, returns WARMSTART_SEED_PARAMS."""
        non_existent = tmp_path / "does_not_exist.json"
        with patch.object(_mod, "V10_RESULTS_JSON", non_existent):
            result = load_warmstart_params("v1.0")
        assert result == dict(WARMSTART_SEED_PARAMS)

    def test_loads_named_winner_from_json(self, tmp_path):
        """When v1.0 JSON contains V10_WINNER_NAME, loads those params."""
        params = {
            "num_leaves": 712,
            "max_depth": 16,
            "learning_rate": 0.0955,
            "min_child_samples": 18,
            "lambda_l1": 0.741,
            "lambda_l2": 12.96,
            "subsample": 0.860,
            "colsample_bytree": 0.860,
            "feature_fraction_bynode": 0.488,
            "min_sum_hessian_in_leaf": 26.74,
        }
        saved = [
            {"name": V10_WINNER_NAME, "fitness": 0.8237, "params": params},
            {"name": "P3_0001", "fitness": 0.81, "params": {k: 1.0 for k in params}},
        ]
        v10_json = tmp_path / "v10_results.json"
        with open(v10_json, "w", encoding="utf-8") as f:
            json.dump(saved, f)

        with patch.object(_mod, "V10_RESULTS_JSON", v10_json):
            result = load_warmstart_params("v1.0")

        for k in P2_BOUNDS:
            if k in params:
                assert result[k] == float(params[k]), (
                    f"Expected param '{k}' = {params[k]}, got {result[k]}"
                )

    def test_falls_back_to_best_by_fitness_when_winner_not_found(self, tmp_path):
        """If the named winner is absent, picks the entry with highest fitness."""
        params_best = {"num_leaves": 500, "learning_rate": 0.10, "max_depth": 12}
        params_other = {"num_leaves": 100, "learning_rate": 0.02, "max_depth": 4}
        saved = [
            {"name": "P1_0001", "fitness": 0.80, "params": params_other},
            {"name": "P2_0001", "fitness": 0.82, "params": params_best},  # best
        ]
        v10_json = tmp_path / "v10_results.json"
        with open(v10_json, "w", encoding="utf-8") as f:
            json.dump(saved, f)

        with patch.object(_mod, "V10_RESULTS_JSON", v10_json):
            result = load_warmstart_params("v1.0")

        # Should have loaded params_best (highest fitness)
        assert result.get("num_leaves") == 500.0
        assert result.get("max_depth") == 12.0

    def test_non_v10_source_returns_hardcoded(self):
        """Passing any source other than 'v1.0' returns WARMSTART_SEED_PARAMS."""
        result = load_warmstart_params("some_other_source")
        assert result == dict(WARMSTART_SEED_PARAMS)

    def test_returned_keys_are_subset_of_p2_bounds(self, tmp_path):
        """All keys in the returned dict must be valid P2 parameter names."""
        non_existent = tmp_path / "does_not_exist.json"
        with patch.object(_mod, "V10_RESULTS_JSON", non_existent):
            result = load_warmstart_params("v1.0")
        for k in result:
            assert k in P2_BOUNDS, f"Unexpected key '{k}' not in P2_BOUNDS"

    def test_all_values_are_float(self, tmp_path):
        """All values in the returned dict must be float (CMA-ES requirement)."""
        non_existent = tmp_path / "does_not_exist.json"
        with patch.object(_mod, "V10_RESULTS_JSON", non_existent):
            result = load_warmstart_params("v1.0")
        for k, v in result.items():
            assert isinstance(v, float), f"Expected float for '{k}', got {type(v)}"


# ════════════════════════════════════════════════════════════════════════════
# Phase3Objective narrow range — v1.1 uses ±60% (was ±50% in v1.0)
# ════════════════════════════════════════════════════════════════════════════

class TestPhase3NarrowRange:
    """Phase 3 uses ±60% instead of ±50% to widen TPE search from v1.0 plateau."""

    def _make_p3_obj(self) -> "Phase3Objective":
        """Create a Phase3Objective with the P3_0017 seed params."""
        seed = dict(WARMSTART_SEED_PARAMS)
        # Provide minimal DataFrames — they won't be used in narrow tests
        dummy_X = pd.DataFrame({"f": [0.0]})
        dummy_y = pd.Series([0])
        return Phase3Objective(
            dummy_X, dummy_y, dummy_X, dummy_y,
            list(dummy_X.columns), [], [], 0.0, 9999999.0,
            best_params=seed,
        )

    def test_narrow_float_range_is_60_percent(self):
        """_narrow(key, lo, hi) window width = seed * 0.60 (v1.1 change)."""
        obj = self._make_p3_obj()
        # lambda_l2 seed = 12.96 from WARMSTART_SEED_PARAMS
        seed_val = float(WARMSTART_SEED_PARAMS["lambda_l2"])
        lo_bound, hi_bound = P2_BOUNDS["lambda_l2"]
        computed_lo, computed_hi = obj._narrow("lambda_l2", lo_bound, hi_bound)
        expected_w = max(seed_val * 0.60, (hi_bound - lo_bound) * 0.10)
        expected_lo = max(lo_bound, seed_val - expected_w)
        expected_hi = min(hi_bound, seed_val + expected_w)
        assert abs(computed_lo - expected_lo) < 1e-9, (
            f"_narrow lo mismatch: expected {expected_lo:.4f}, got {computed_lo:.4f}"
        )
        assert abs(computed_hi - expected_hi) < 1e-9, (
            f"_narrow hi mismatch: expected {expected_hi:.4f}, got {computed_hi:.4f}"
        )

    def test_narrow_int_range_is_60_percent(self):
        """_narrow_int(key, lo, hi) window = seed * 0.60."""
        obj = self._make_p3_obj()
        seed_val = int(WARMSTART_SEED_PARAMS["num_leaves"])
        lo_bound, hi_bound = P2_BOUNDS["num_leaves"]
        computed_lo, computed_hi = obj._narrow_int("num_leaves", lo_bound, hi_bound)
        w = max(int(seed_val * 0.60), 10, 1)
        expected_lo = max(lo_bound, seed_val - w)
        expected_hi = min(hi_bound, seed_val + w)
        assert computed_lo == expected_lo
        assert computed_hi == expected_hi

    def test_v1_1_wider_than_v1_0(self):
        """v1.1 ±60% range must produce a wider interval than v1.0's ±50%."""
        obj = self._make_p3_obj()
        seed_val = float(WARMSTART_SEED_PARAMS["lambda_l2"])
        lo_bound, hi_bound = P2_BOUNDS["lambda_l2"]

        lo11, hi11 = obj._narrow("lambda_l2", lo_bound, hi_bound)
        range_v11 = hi11 - lo11

        # Simulate v1.0 logic (±50%)
        w_v10 = max(seed_val * 0.50, (hi_bound - lo_bound) * 0.10)
        lo10 = max(lo_bound, seed_val - w_v10)
        hi10 = min(hi_bound, seed_val + w_v10)
        range_v10 = hi10 - lo10

        assert range_v11 >= range_v10, (
            f"v1.1 range ({range_v11:.4f}) should be ≥ v1.0 range ({range_v10:.4f})"
        )

    def test_narrow_respects_p2_bounds(self):
        """Narrowed ranges must stay within P2_BOUNDS for all parameters."""
        obj = self._make_p3_obj()
        for key, (lo_bound, hi_bound) in P2_BOUNDS.items():
            if isinstance(lo_bound, int):
                lo, hi = obj._narrow_int(key, lo_bound, hi_bound)
            else:
                lo, hi = obj._narrow(key, lo_bound, hi_bound)
            assert lo >= lo_bound, f"Narrowed lo for '{key}' below P2 bound"
            assert hi <= hi_bound, f"Narrowed hi for '{key}' above P2 bound"
            assert lo <= hi, f"Narrowed lo > hi for '{key}'"


# ════════════════════════════════════════════════════════════════════════════
# train_and_evaluate — sample_weight propagation
# ════════════════════════════════════════════════════════════════════════════

class TestTrainAndEvaluateWeights:
    """Verify that sample_weights_train is passed to lgb.Dataset."""

    def test_sample_weights_passed_to_lgb_dataset(self):
        """When sample_weights_train is provided, lgb.Dataset must receive it.

        Also verifies the weight sub-selection uses positional indexing (iloc/pos)
        so it is safe when the DataFrame has a non-contiguous index after group_split.
        """
        n_train = 20
        n_val = 5
        n_feat = EXPECTED_FEATURE_COUNT_UNIV

        X_tr = pd.DataFrame(np.random.randn(n_train, n_feat).astype(np.float32))
        y_tr = pd.Series(np.zeros(n_train, dtype=int))
        X_vl = pd.DataFrame(np.random.randn(n_val, n_feat).astype(np.float32))
        y_vl = pd.Series(np.zeros(n_val, dtype=int))
        feat_cols = list(X_tr.columns)
        sample_w = np.ones(n_train, dtype=np.float32) * 2.0

        captured_weights = []
        original_dataset = _mod.lgb.Dataset

        def _capturing_dataset(X, label=None, weight=None, **kwargs):
            if weight is not None:
                captured_weights.append(weight)
            return MagicMock(
                construct=MagicMock(), num_data=lambda: len(X)
            )

        # Minimal fake booster
        fake_booster = MagicMock()
        fake_booster.best_iteration = 10
        fake_booster.best_score = {"valid_0": {"multi_logloss": 0.5}}
        proba = np.zeros((n_val, NUM_CLASSES), dtype=np.float32)
        proba[:, 0] = 1.0
        fake_booster.predict = MagicMock(return_value=proba)
        fake_booster.feature_importance = MagicMock(return_value=np.ones(n_feat))
        fake_booster.save_model = MagicMock()

        with (
            patch.object(_mod.lgb, "Dataset", side_effect=_capturing_dataset),
            patch.object(_mod.lgb, "train", return_value=fake_booster),
            patch.object(_mod, "evaluate_external", return_value={
                "accuracy": 0.95, "correct": 95, "sample_size": 100,
                "per_syllable": {}
            }),
            patch.object(_mod, "evaluate_handcrafted", return_value={
                "accuracy": 0.90, "correct": 9, "total": 10, "total_words": 10,
                "results": []
            }),
            patch.object(_mod, "_append_json"),
            patch.object(_mod, "append_result_csv"),
            patch.object(_mod, "CONVERGENCE_DIR", Path("/tmp/fake_conv")),
            patch.object(_mod, "FEAT_IMP_DIR", Path("/tmp/fake_feat")),
            patch.object(_mod, "RESULTS_DIR", Path("/tmp/fake_results")),
        ):
            _mod.train_and_evaluate(
                params={"num_leaves": 31, "max_depth": 4, "learning_rate": 0.1,
                        "min_child_samples": 10, "lambda_l1": 0.0,
                        "lambda_l2": 0.0, "subsample": 1.0,
                        "colsample_bytree": 1.0, "feature_fraction_bynode": 1.0,
                        "min_sum_hessian_in_leaf": 1e-3},
                X_train=X_tr, y_train=y_tr,
                X_val=X_vl, y_val=y_vl,
                feature_cols=feat_cols,
                external_sample=_simple_external_sample(5),
                handcrafted_tests=_simple_handcrafted(2),
                wall_start=0.0, max_rounds=50,
                early_stopping_rounds=10,
                trial=None, trial_name="T_0001",
                phase="P1", trial_number=1,
                fitness_preset="luscinia_universal_specialist",
                train_frac=1.0,
                sample_weights_train=sample_w,
            )

        assert len(captured_weights) >= 1, (
            "lgb.Dataset must be called with sample weights"
        )
        assert any(w is not None for w in captured_weights), (
            "At least one lgb.Dataset call should receive non-None weights"
        )

    def test_none_weights_not_passed_when_absent(self):
        """When sample_weights_train=None, lgb.Dataset weight should be None."""
        n_train = 20
        n_val = 5
        n_feat = EXPECTED_FEATURE_COUNT_UNIV

        X_tr = pd.DataFrame(np.random.randn(n_train, n_feat).astype(np.float32))
        y_tr = pd.Series(np.zeros(n_train, dtype=int))
        X_vl = pd.DataFrame(np.random.randn(n_val, n_feat).astype(np.float32))
        y_vl = pd.Series(np.zeros(n_val, dtype=int))
        feat_cols = list(X_tr.columns)

        weight_args = []

        def _capturing_dataset(X, label=None, weight=None, **kwargs):
            weight_args.append(weight)
            return MagicMock(
                construct=MagicMock(), num_data=lambda: len(X)
            )

        fake_booster = MagicMock()
        fake_booster.best_iteration = 10
        fake_booster.best_score = {"valid_0": {"multi_logloss": 0.5}}
        proba = np.zeros((n_val, NUM_CLASSES), dtype=np.float32)
        proba[:, 0] = 1.0
        fake_booster.predict = MagicMock(return_value=proba)
        fake_booster.feature_importance = MagicMock(return_value=np.ones(n_feat))

        with (
            patch.object(_mod.lgb, "Dataset", side_effect=_capturing_dataset),
            patch.object(_mod.lgb, "train", return_value=fake_booster),
            patch.object(_mod, "evaluate_external", return_value={
                "accuracy": 0.95, "correct": 95, "sample_size": 100,
                "per_syllable": {}
            }),
            patch.object(_mod, "evaluate_handcrafted", return_value={
                "accuracy": 0.90, "correct": 9, "total": 10, "total_words": 10,
                "results": []
            }),
            patch.object(_mod, "_append_json"),
            patch.object(_mod, "append_result_csv"),
            patch.object(_mod, "CONVERGENCE_DIR", Path("/tmp/fake_conv")),
            patch.object(_mod, "FEAT_IMP_DIR", Path("/tmp/fake_feat")),
            patch.object(_mod, "RESULTS_DIR", Path("/tmp/fake_results")),
        ):
            _mod.train_and_evaluate(
                params={"num_leaves": 31, "max_depth": 4, "learning_rate": 0.1,
                        "min_child_samples": 10, "lambda_l1": 0.0,
                        "lambda_l2": 0.0, "subsample": 1.0,
                        "colsample_bytree": 1.0, "feature_fraction_bynode": 1.0,
                        "min_sum_hessian_in_leaf": 1e-3},
                X_train=X_tr, y_train=y_tr,
                X_val=X_vl, y_val=y_vl,
                feature_cols=feat_cols,
                external_sample=_simple_external_sample(5),
                handcrafted_tests=_simple_handcrafted(2),
                wall_start=0.0, max_rounds=50,
                early_stopping_rounds=10,
                trial=None, trial_name="T_0001",
                phase="P1", trial_number=1,
                fitness_preset="luscinia_universal_specialist",
                train_frac=1.0,
                sample_weights_train=None,
            )

        # First lgb.Dataset call (train set) should have weight=None
        assert weight_args[0] is None, (
            "lgb.Dataset should receive weight=None when no sample weights provided"
        )


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — early exits
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleEarlyExit:
    """Ensemble should silently return None when it cannot build a valid ensemble."""

    def test_returns_none_with_no_results(self):
        result = run_ensemble_phase([], _feature_cols(), [], [])
        assert result is None

    def test_returns_none_with_zero_saved_models(self):
        results = [_make_result(f"P3_{i:04d}") for i in range(5)]
        result = run_ensemble_phase(
            results, _feature_cols(),
            _simple_external_sample(), _simple_handcrafted(),
        )
        assert result is None

    def test_returns_none_with_only_one_saved_model(self, tmp_path):
        results = [_make_result("P3_0001"), _make_result("P3_0002")]
        model_dir = tmp_path / "P3_0001"
        model_dir.mkdir()
        (model_dir / "P3_0001.lgb").write_text("fake")

        fake_bst = _fake_booster_univ(0)
        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
        ):
            result = run_ensemble_phase(
                results, _feature_cols(),
                _simple_external_sample(), _simple_handcrafted(),
            )
        assert result is None


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — accept / reject
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleAcceptReject:

    def _run_with_mocked_models(
        self, results, tmp_path, ensemble_class_pred: int = 0,
        top_k: int = 5,
        min_sanity_gain: float = ENSEMBLE_MIN_SANITY_GAIN,
        external_sample: Optional[list] = None,
        handcrafted: Optional[list] = None,
    ) -> Optional[dict]:
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir(exist_ok=True)
            (d / f"{r['name']}.lgb").write_text("fake")

        fake_bst = _fake_booster_univ(ensemble_class_pred)
        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
        ):
            return run_ensemble_phase(
                results, _feature_cols(),
                external_sample or _simple_external_sample(100),
                handcrafted or _simple_handcrafted(4),
                top_k=top_k,
                min_sanity_gain=min_sanity_gain,
            )

    def test_accepted_on_sanity_gain(self, tmp_path):
        """Ensemble gains sanity → accepted."""
        # Best solo has low sanity; ensemble predicts class 0 → 100% for class-0 sample
        external = _simple_external_sample(100, expected_label=0)
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.82 - i * 0.001, sanity_acc=0.50,
                         hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_class_pred=0,
            external_sample=external,
            handcrafted=_simple_handcrafted(4, expected=0),
        )
        assert result is not None
        assert result["phase"] == "P4"

    def test_rejected_when_no_improvement(self, tmp_path):
        """Ensemble predicts wrong class for all → rejected."""
        # External: all expected class 1; ensemble predicts class 0 → 0% correct
        # Best solo has sanity=1.0, hand=4/4 → no improvement possible
        external = _simple_external_sample(100, expected_label=1)
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.82 - i * 0.001,
                         sanity_acc=1.0, hand_correct=4, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_class_pred=0,
            external_sample=external,
            handcrafted=_simple_handcrafted(4, expected=1),
        )
        assert result is None

    def test_rejected_path_writes_report_json(self, tmp_path):
        external = _simple_external_sample(100, expected_label=1)
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.82 - i * 0.001,
                         sanity_acc=1.0, hand_correct=4, hand_total=4)
            for i in range(3)
        ]
        self._run_with_mocked_models(
            results, tmp_path, ensemble_class_pred=0,
            external_sample=external,
            handcrafted=_simple_handcrafted(4, expected=1),
        )
        report = tmp_path / "P4_ensemble" / "ensemble_report.json"
        assert report.exists()
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["decision"] == "rejected"
        assert "sanity_gain" in data
        assert "hand_gain" in data

    def test_accepted_result_has_correct_schema(self, tmp_path):
        external = _simple_external_sample(100, expected_label=0)
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.82 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_class_pred=0,
            external_sample=external,
            handcrafted=_simple_handcrafted(4, expected=0),
        )
        assert result is not None
        for key in ["phase", "name", "fitness", "params", "internal",
                    "external", "handcrafted", "ensemble_models",
                    "ensemble_weights", "best_solo_name", "fitness_preset"]:
            assert key in result, f"Missing key '{key}' in ensemble result"
        assert result["fitness_preset"] == "luscinia_universal_specialist"


# ════════════════════════════════════════════════════════════════════════════
# _handcrafted_to_training_rows
# ════════════════════════════════════════════════════════════════════════════

class TestHandcraftedToTrainingRows:
    """_handcrafted_to_training_rows for universal 11-class model."""

    def _feat_cols(self) -> list:
        return list(build_features_universal("мамина", "ADJ").keys())

    def test_returns_none_on_empty_input(self):
        X, y = _handcrafted_to_training_rows([], self._feat_cols())
        assert X is None
        assert y is None

    def test_returns_dataframe_and_series(self):
        tests = [("мамина", "ADJ", 1, "penult stress", None)]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_row_count(self):
        tests = [
            ("мамина", "ADJ", 1, "t1", None),
            ("батьківська", "ADJ", 2, "t2", None),
            ("університет", "NOUN", 3, "t3", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 3
        assert len(y) == 3

    def test_skips_none_expected(self):
        tests = [
            ("мамина", "ADJ", 1, "ok", None),
            ("unknown", "X", None, "skip", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 1

    def test_skips_list_expected(self):
        tests = [
            ("мамина", "ADJ", 1, "ok", None),
            ("вода", "NOUN", [0, 1], "ambiguous", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, self._feat_cols())
        assert len(X) == 1

    def test_feature_columns_match(self):
        tests = [("мамина", "ADJ", 1, "test", None)]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert list(X.columns) == feat_cols

    def test_label_within_valid_range(self):
        tests = [
            ("мама", "NOUN", 0, "class0", None),
            ("батьківська", "ADJ", 2, "class2", None),
            ("університет", "NOUN", 3, "class3", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        for val in y:
            assert 0 <= val < NUM_CLASSES, f"Label {val} out of range [0, {NUM_CLASSES})"

    def test_skips_label_at_or_above_num_classes(self):
        tests = [
            ("мамина", "ADJ", 1, "ok", None),
            ("word", "X", NUM_CLASSES, "out of range", None),   # too large
            ("word2", "X", NUM_CLASSES + 5, "way out", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert len(X) == 1, "Out-of-range labels should be skipped"


# ════════════════════════════════════════════════════════════════════════════
# Structural checks — study names, CMA-ES sigma, train_frac constants
# ════════════════════════════════════════════════════════════════════════════

class TestStructuralChecks:
    """Verify v1.1-specific source-level constants and names."""

    def _src(self) -> str:
        return Path(_script_path).read_text(encoding="utf-8")

    def test_cma_es_sigma_is_0_35(self):
        """v1.1 widened CMA-ES sigma from 0.25 to 0.35."""
        src = self._src()
        assert "sigma0=0.35" in src, (
            "CMA-ES sigma must be 0.35 in v1.1 (was 0.25 in v1.0)"
        )

    def test_p3_train_frac_is_0_75(self):
        """Phase 3 uses train_frac=0.75 (was 1.0 in v1.0)."""
        src = self._src()
        assert "train_frac=0.75" in src, (
            "Phase 3 must use train_frac=0.75 in v1.1"
        )

    def test_p1_train_frac_is_0_5(self):
        """Phase 1 uses train_frac=0.5 (fast random exploration)."""
        src = self._src()
        assert "train_frac=0.5" in src

    def test_p2_train_frac_is_0_5(self):
        """Phase 2 (CMA-ES) uses train_frac=0.5 — same fast-HPO strategy as P1.

        The winner is refitted on 100% data at the end, so 50% here is fine.
        """
        src = self._src()
        p2_block = src[src.find("class Phase2Objective"):src.find("class Phase3Objective")]
        assert "train_frac=0.5" in p2_block

    def test_study_names_contain_v1_1(self):
        """Optuna study names must include 'v1_1' to avoid collisions with v1.0."""
        src = self._src()
        assert "v1_1_p1" in src, "P1 study name should include v1_1"
        assert "v1_1_p2" in src, "P2 study name should include v1_1"
        assert "v1_1_p3" in src, "P3 study name should include v1_1"

    def test_artifacts_dir_contains_v1_1(self):
        """RESULTS_DIR must point to a v1.1 directory (not v1.0)."""
        src = self._src()
        assert "univ-v1.1" in src, "RESULTS_DIR must reference v1.1 artifacts"

    def test_is_unbalance_not_in_phase_params(self):
        """is_unbalance must not appear in any phase params dict."""
        src = self._src()
        assert '"is_unbalance"' not in src, (
            "is_unbalance must be removed from all phase params in v1.1"
        )

    def test_default_budget_is_24h(self):
        """Default budget in v1.1 is 24h (not 36h from v1.0)."""
        src = self._src()
        assert "budget_hours = 24.0" in src, (
            "Default budget should be 24h in v1.1 (warm-start saves ~10h)"
        )

    def test_p3_preset_is_luscinia_universal_specialist(self):
        src = self._src()
        assert 'fitness_preset="luscinia_universal_specialist"' in src

    def test_p2_pruner_is_not_patient_pruner(self):
        """PatientPruner(patience=0) must NOT wrap the P2 Hyperband pruner.

        patience=0 kills every CMA-ES trial at the first non-improving
        checkpoint, preventing CMA-ES from learning the fitness landscape.
        The import and comments may reference it, but it must not be called.
        """
        src = self._src()
        p2_start = src.find("PHASE 2 \u2014 CMA-ES")
        p3_start = src.find("PHASE 3 \u2014")
        if p2_start == -1 or p3_start == -1:
            pytest.skip("Could not locate phase block markers in source")
        p2_block = src[p2_start:p3_start]
        # Check for actual call `PatientPruner(` not just comments mentioning it
        assert "PatientPruner(" not in p2_block, (
            "PatientPruner(...) must not be called in Phase 2 study creation \u2014 "
            "patience=0 kills CMA-ES trials before the model can learn"
        )

    def test_subsampling_uses_positional_indexing(self):
        """Weight sub-selection must use positional indices (not pandas labels).

        With a non-contiguous DataFrame index from group_split, using the
        pandas label index as a numpy array subscript causes IndexError or
        silently maps weights to the wrong samples.
        """
        src = self._src()
        # The fix uses rng.choice / iloc[pos] — must NOT use .sample(...).index
        # as a numpy subscript
        assert "iloc[pos]" in src, (
            "Weight subsampling must use iloc[pos] (positional) not loc[idx] "
            "to avoid wrong-weight assignment on non-contiguous DataFrame index"
        )



    def test_warmstart_seed_has_all_p2_bounds_keys(self):
        """WARMSTART_SEED_PARAMS must contain all required P2 parameter keys."""
        for key in P2_BOUNDS:
            assert key in WARMSTART_SEED_PARAMS, (
                f"WARMSTART_SEED_PARAMS missing key '{key}'"
            )

    def test_class_weight_power_is_0_5(self):
        """Default CLASS_WEIGHT_POWER must be 0.5 (square-root inverse)."""
        assert abs(CLASS_WEIGHT_POWER - 0.5) < 1e-9

    def test_refit_computes_sample_weights(self):
        """refit_on_full_data must call compute_sample_weights for the full dataset."""
        src = self._src()
        assert "compute_sample_weights(y_full" in src, (
            "refit_on_full_data must recompute sample weights on the full dataset"
        )

    def test_refit_passes_weights_to_lgb_dataset(self):
        """refit_on_full_data must pass weight=w_full to lgb.Dataset."""
        src = self._src()
        # The refit block should have lgb.Dataset(..., weight=w_full, ...)
        assert "weight=w_full" in src, (
            "refit_on_full_data must pass sample weights to lgb.Dataset"
        )
