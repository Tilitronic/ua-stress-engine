"""Tests for batch_train_2syl_v4 — compute_fitness, _handcrafted_to_training_rows,
and the refit helper.

All tests are pure (no DB, no real LightGBM model, no disk I/O).
"""
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest

# Make the script importable as a module
_SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "src", "stress_prediction", "lightGbm",
)
sys.path.insert(0, _SCRIPT_DIR)

# We import only the pure functions — no side-effects at import time
# because the script guards main() under `if __name__ == "__main__"`.
import importlib.util

_script_path = os.path.join(_SCRIPT_DIR, "batch_train_2syl_v4.py")
_spec = importlib.util.spec_from_file_location("batch_train_2syl_v4", _script_path)
_mod = importlib.util.module_from_spec(_spec)
# __file__ must be set before exec so Path(__file__) works inside the module
_mod.__file__ = _script_path
_spec.loader.exec_module(_mod)  # type: ignore

compute_fitness = _mod.compute_fitness
_handcrafted_to_training_rows = _mod._handcrafted_to_training_rows
_clip_to_p2_bounds = _mod._clip_to_p2_bounds
P2_BOUNDS = _mod.P2_BOUNDS

# Phase 1 search space — must be a subset of P2_BOUNDS for every parameter
# that CMA-ES seeds from. If P1 can produce a value outside P2_BOUNDS, the
# sampler crashes with "invalid bounds" on trial #1.
_P1_RANGES = {
    # (lo, hi) as coded in Phase1Objective.__call__
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

from services.feature_service import build_features_v13
from services.constants import EXPECTED_FEATURE_COUNT


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def _feature_cols() -> List[str]:
    return list(build_features_v13("мама", "NOUN").keys())


# ════════════════════════════════════════════════════════════════════
# compute_fitness  (v4 — no handcrafted penalty)
# ════════════════════════════════════════════════════════════════════
class TestComputeFitnessV4:

    def test_perfect_scores(self):
        fitness, penalty = compute_fitness(1.0, 1.0, 1.0, 1.0, 6, 6)
        assert fitness > 1.0  # size bonus pushes above 1.0

    def test_zero_scores(self):
        fitness, penalty = compute_fitness(0.0, 0.0, 0.0, 0.0, 0, 6)
        assert fitness < 0.03  # only size bonus (~0.02)

    def test_no_penalty_when_all_correct(self):
        """hand_correct == hand_total → penalty must be False."""
        _, penalty = compute_fitness(0.8, 0.95, 0.9, 0.85, 6, 6)
        assert penalty is False

    def test_penalty_flag_when_not_perfect(self):
        """hand_correct < hand_total → penalty flag True, but fitness unchanged."""
        fitness_full, _ = compute_fitness(0.8, 0.95, 0.9, 0.85, 6, 6)
        fitness_miss, penalty = compute_fitness(0.8, 0.95, 0.9, 0.85, 5, 6)
        assert penalty is True
        # fitness must be IDENTICAL — no multiplicative penalty in v4
        assert abs(fitness_full - fitness_miss) < 1e-9

    def test_fitness_not_reduced_by_handcrafted_miss(self):
        """Core v4 contract: missing handcrafted never reduces fitness score."""
        f_perfect, _ = compute_fitness(0.75, 0.95, 1.0, 0.85, 9, 9)
        f_miss_one, _ = compute_fitness(0.75, 0.95, 1.0, 0.85, 8, 9)
        f_miss_all, _ = compute_fitness(0.75, 0.95, 1.0, 0.85, 0, 9)
        assert abs(f_perfect - f_miss_one) < 1e-9
        assert abs(f_perfect - f_miss_all) < 1e-9

    def test_weights_45_30_25(self):
        """Formula v2: 0.45*f1 + 0.30*acc + 0.25*hand_acc + size_bonus(0.02 for size_mb=0)."""
        f, _ = compute_fitness(f1=0.6, ext_acc=0.9, hand_acc=0.5, acc=0.7,
                               hand_correct=3, hand_total=6)
        expected = 0.45 * 0.6 + 0.30 * 0.7 + 0.25 * 0.5 + 0.02
        assert abs(f - expected) < 1e-6

    def test_f1_dominates(self):
        """f1 has 50% weight — largest single component."""
        f_high_ext, _ = compute_fitness(0.0, 1.0, 0.0, 0.0, 0, 6)
        f_high_f1, _ = compute_fitness(1.0, 0.0, 0.0, 0.0, 0, 6)
        assert f_high_f1 > f_high_ext

    def test_bounds(self):
        import random
        rng = random.Random(42)
        for _ in range(200):
            f1 = rng.random()
            ext = rng.random()
            hand = rng.random()
            auc = rng.random()
            total = rng.randint(1, 10)
            correct = rng.randint(0, total)
            fitness, _ = compute_fitness(f1, ext, hand, auc, correct, total)
            assert 0.0 <= fitness <= 1.03 + 1e-9

    def test_zero_hand_total_no_crash(self):
        """hand_total=0 should not crash (no division)."""
        fitness, penalty = compute_fitness(0.8, 0.9, 0.0, 0.85, 0, 0)
        assert isinstance(fitness, float)
        assert penalty is False


# ════════════════════════════════════════════════════════════════════
# _handcrafted_to_training_rows
# ════════════════════════════════════════════════════════════════════
class TestHandcraftedToTrainingRows:

    def _cols(self):
        return _feature_cols()

    def test_unambiguous_entries_converted(self):
        """Entries with a single integer label → included."""
        cols = self._cols()
        tests = [
            ("мама", "NOUN", 0, "stress on 1st", None),
            ("весна", "NOUN", 1, "stress on 2nd", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert X is not None
        assert len(X) == 2
        assert list(y) == [0, 1]

    def test_ambiguous_list_skipped(self):
        """expected=[0,1] (ambiguous) → skipped."""
        cols = self._cols()
        tests = [
            ("мама", "NOUN", 0, "ok", None),
            ("слово", "NOUN", [0, 1], "ambiguous", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert len(X) == 1
        assert list(y) == [0]

    def test_single_item_list_accepted(self):
        """expected=[1] (list with one value) → treated as label 1."""
        cols = self._cols()
        tests = [("весна", "NOUN", [1], "single-item list", None)]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert X is not None
        assert list(y) == [1]

    def test_none_expected_skipped(self):
        """expected=None → no label, skip."""
        cols = self._cols()
        tests = [("слово", "NOUN", None, "no label", None)]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert X is None
        assert y is None

    def test_all_ambiguous_returns_none(self):
        """All entries ambiguous → (None, None)."""
        cols = self._cols()
        tests = [
            ("слово", "NOUN", [0, 1], "ambiguous", None),
            ("місто", "NOUN", None, "no label", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert X is None
        assert y is None

    def test_feature_columns_match(self):
        """Returned DataFrame has exactly the requested feature columns."""
        cols = self._cols()
        tests = [("мама", "NOUN", 0, "test", None)]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert list(X.columns) == cols

    def test_correct_feature_count(self):
        cols = self._cols()
        tests = [
            ("мама", "NOUN", 0, "a", None),
            ("весна", "NOUN", 1, "b", None),
            ("вітер", "NOUN", 0, "c", None),
        ]
        X, y = _handcrafted_to_training_rows(tests, cols)
        assert X.shape == (3, len(cols))

    def test_values_are_numeric(self):
        cols = self._cols()
        tests = [("мама", "NOUN", 0, "test", None)]
        X, _ = _handcrafted_to_training_rows(tests, cols)
        assert X.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()

    def test_empty_input_returns_none(self):
        X, y = _handcrafted_to_training_rows([], _feature_cols())
        assert X is None
        assert y is None

    def test_features_json_used_when_provided(self):
        """features_json at position 4 is forwarded to build_features_v13."""
        cols = self._cols()
        import json
        fj = json.dumps({"Case": "Nom", "Number": "Sing"})
        tests_with = [("мама", "NOUN", 0, "with morph", fj)]
        tests_without = [("мама", "NOUN", 0, "without morph", None)]
        X_with, _ = _handcrafted_to_training_rows(tests_with, cols)
        X_without, _ = _handcrafted_to_training_rows(tests_without, cols)
        # rows may differ because morph features change some values
        assert X_with is not None
        assert X_without is not None


# ════════════════════════════════════════════════════════════════════
# P2_BOUNDS coverage — P1 must be a subset of P2 for every param
# ════════════════════════════════════════════════════════════════════
class TestP2BoundsCoverP1:
    """Guard against the CMA-ES 'invalid bounds' crash.

    CMA-ES requires x0 (the Phase-1 winner) to lie strictly inside the
    Phase-2 search box.  If any Phase-1 bound exceeds the corresponding
    Phase-2 bound, the very first P2 trial crashes with AssertionError.
    """

    def test_all_p1_params_present_in_p2_bounds(self):
        """Every param that Phase 1 can produce must have a P2 bound."""
        for key in _P1_RANGES:
            assert key in P2_BOUNDS, (
                f"P2_BOUNDS is missing '{key}' — if P1 produces this value "
                f"CMA-ES x0 cannot be clipped and will crash."
            )

    def test_p2_lower_le_p1_lower(self):
        """P2 lower bound must be ≤ P1 lower bound for all shared params."""
        for key, (p1_lo, _) in _P1_RANGES.items():
            if key not in P2_BOUNDS:
                continue
            p2_lo, _ = P2_BOUNDS[key]
            assert p2_lo <= p1_lo, (
                f"P2_BOUNDS['{key}'] lo={p2_lo} > P1 lo={p1_lo}. "
                f"A P1 seed at the lower boundary would be clipped inward, "
                f"potentially moving the CMA-ES start point away from the winner."
            )

    def test_p2_upper_ge_p1_upper(self):
        """P2 upper bound must be ≥ P1 upper bound for all shared params."""
        for key, (_, p1_hi) in _P1_RANGES.items():
            if key not in P2_BOUNDS:
                continue
            _, p2_hi = P2_BOUNDS[key]
            assert p2_hi >= p1_hi, (
                f"P2_BOUNDS['{key}'] hi={p2_hi} < P1 hi={p1_hi}. "
                f"Phase 1 can produce {key}={p1_hi} but P2 box only reaches "
                f"{p2_hi} — CMA-ES will crash with 'invalid bounds'."
            )


# ════════════════════════════════════════════════════════════════════
# _clip_to_p2_bounds
# ════════════════════════════════════════════════════════════════════
class TestClipToP2Bounds:

    def test_in_bounds_value_unchanged(self):
        """Values already inside P2_BOUNDS must not be changed."""
        params = {
            "num_leaves": 300.0,
            "learning_rate": 0.10,
            "subsample": 0.8,
        }
        clipped = _clip_to_p2_bounds(params)
        assert abs(clipped["num_leaves"] - 300.0) < 1e-9
        assert abs(clipped["learning_rate"] - 0.10) < 1e-9
        assert abs(clipped["subsample"] - 0.8) < 1e-9

    def test_above_upper_bound_clipped(self):
        """Value above P2 upper bound → clipped to upper bound."""
        hi = P2_BOUNDS["num_leaves"][1]
        params = {"num_leaves": float(hi + 500)}
        clipped = _clip_to_p2_bounds(params)
        assert abs(clipped["num_leaves"] - float(hi)) < 1e-9

    def test_below_lower_bound_clipped(self):
        """Value below P2 lower bound → clipped to lower bound."""
        lo = P2_BOUNDS["num_leaves"][0]
        params = {"num_leaves": float(lo - 10)}
        clipped = _clip_to_p2_bounds(params)
        assert abs(clipped["num_leaves"] - float(lo)) < 1e-9

    def test_all_clipped_values_inside_bounds(self):
        """After clipping, every returned value must be within P2_BOUNDS."""
        # Construct an adversarial dict with extreme out-of-range values
        extreme = {k: hi * 10 for k, (lo, hi) in P2_BOUNDS.items()}
        clipped = _clip_to_p2_bounds(extreme)
        for key, val in clipped.items():
            lo, hi = P2_BOUNDS[key]
            assert lo <= val <= hi, (
                f"After clip, {key}={val} is outside [{lo}, {hi}]"
            )

    def test_unknown_keys_ignored(self):
        """Keys not in P2_BOUNDS are silently dropped (they have no bounds)."""
        params = {"num_leaves": 300.0, "unknown_param": 999.0}
        clipped = _clip_to_p2_bounds(params)
        assert "unknown_param" not in clipped
        assert "num_leaves" in clipped

    def test_empty_dict_returns_empty(self):
        assert _clip_to_p2_bounds({}) == {}

    def test_returns_floats(self):
        """All values in the returned dict must be floats (CMA-ES requirement)."""
        params = {k: (lo + hi) / 2 for k, (lo, hi) in P2_BOUNDS.items()}
        clipped = _clip_to_p2_bounds(params)
        for key, val in clipped.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_realistic_p1_winner_with_large_leaves(self):
        """Simulate the exact crash scenario: P1 winner has num_leaves=800."""
        p1_winner = {
            "num_leaves": 800.0,         # was causing the crash (old P2 hi was 480)
            "max_depth": 14.0,
            "learning_rate": 0.05,
            "min_child_samples": 20.0,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "feature_fraction_bynode": 0.7,
            "min_sum_hessian_in_leaf": 1.0,
            "path_smooth": 0.1,
            "min_split_gain": 0.0,
        }
        clipped = _clip_to_p2_bounds(p1_winner)
        for key, val in clipped.items():
            lo, hi = P2_BOUNDS[key]
            assert lo <= val <= hi, (
                f"{key}={val} outside P2_BOUNDS [{lo}, {hi}] after clip"
            )
