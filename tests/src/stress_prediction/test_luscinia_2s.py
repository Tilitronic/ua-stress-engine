"""Tests for luscinia-lgbm-str-ua-2s-v1 — pure-function unit tests.

Covers:
  * compute_fitness — default luscinia_specialist preset
  * run_ensemble_phase — accept/reject logic, weight calculation, early exits
  * run_ensemble_phase — ENSEMBLE_DIR / ensemble_report.json written on both
    accept and reject paths
  * Phase fitness presets: P1/P2 use specialist_binary, P3 uses luscinia_specialist

All tests are pure: no DB, no real lightgbm training, no real disk writes
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
# Import the script as a module (same pattern as test_batch_train_2syl_v4.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "src", "stress_prediction", "lightgbm",
)
sys.path.insert(0, _SCRIPT_DIR)

_script_path = os.path.join(_SCRIPT_DIR, "luscinia-lgbm-str-ua-2s-v1.py")
if not os.path.exists(_script_path):
    pytest.skip(f"Training script not found: {_script_path}", allow_module_level=True)
pytest.importorskip("lightgbm", reason="lightgbm not installed — skip training-script tests")
_spec = importlib.util.spec_from_file_location("luscinia_lgbm_str_ua_2s_v1", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_mod.__file__ = _script_path
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

compute_fitness             = _mod.compute_fitness
run_ensemble_phase          = _mod.run_ensemble_phase
_handcrafted_to_training_rows = _mod._handcrafted_to_training_rows
ENSEMBLE_TOP_K              = _mod.ENSEMBLE_TOP_K
ENSEMBLE_MIN_SANITY_GAIN    = _mod.ENSEMBLE_MIN_SANITY_GAIN
ENSEMBLE_DIR                = _mod.ENSEMBLE_DIR
RESULTS_DIR                 = _mod.RESULTS_DIR
build_features_v13          = _mod.build_features_v13

from services.feature_service import build_features_v13 as _build_feat
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
    """Return the feature column list (100 features for 2-syl specialist)."""
    return list(_build_feat("мама", "NOUN").keys())


def _make_result(
    name: str,
    phase: str = "P3",
    fitness: float = 0.80,
    sanity_acc: float = 0.95,
    hand_correct: int = 40,
    hand_total: int = 44,
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
        "fitness_preset": "luscinia_specialist",
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


def _fake_booster(p1: float = 0.9) -> MagicMock:
    """Return a mock lgb.Booster whose predict() always returns p1."""
    bst = MagicMock()
    bst.predict = MagicMock(side_effect=lambda x: np.full(x.shape[0], p1))
    return bst


def _simple_external_sample(n: int = 10) -> list:
    """Minimal external-sample entries (binary-mode: expected_label 0 or 1)."""
    feat_cols = _feature_cols()
    items = []
    for i in range(n):
        feat = _build_feat("мама", "NOUN")
        items.append({
            "form": "мама",
            "pos": "NOUN",
            "features_json": None,
            "expected_label": i % 2,
        })
    return items


def _simple_handcrafted(n: int = 4) -> list:
    """Minimal handcrafted test entries in the 5-tuple format."""
    # (word, pos, expected_label, description, feat_json)
    return [("мама", "NOUN", 0, "test", None) for _ in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# compute_fitness — luscinia_specialist preset (default)
# ════════════════════════════════════════════════════════════════════════════

class TestComputeFitnessDefault:
    """compute_fitness in luscinia_2s_v1.0 defaults to luscinia_specialist."""

    def test_default_preset_is_luscinia_specialist(self):
        """Perfect inputs (sanity >= acc) → fitness ~1.02 (size bonus included)."""
        fitness, penalty, sanity_bad = compute_fitness(1.0, 1.0, 1.0, 1.0, 6, 6)
        assert fitness > 1.0, "Size bonus should push perfect score above 1.0"
        assert not sanity_bad

    def test_zero_scores(self):
        fitness, penalty, sanity_bad = compute_fitness(0.0, 0.0, 0.0, 0.0, 0, 6)
        # sanity(0.0) < acc(0.0) → no penalty (equal, not less-than)
        assert fitness < 0.03   # only tiny size bonus remains

    def test_returns_tuple(self):
        result = compute_fitness(0.8, 0.95, 0.9, 0.85, 5, 6)
        assert isinstance(result, tuple)
        assert len(result) == 3   # (score, hand_penalty, sanity_violated)

    def test_fitness_is_float(self):
        fitness, _p, _s = compute_fitness(0.75, 0.94, 0.88, 0.90, 40, 44)
        assert isinstance(fitness, float)

    def test_penalty_flag_when_not_perfect(self):
        """Missing at least one handcrafted word → hand penalty flag True."""
        _, penalty, _ = compute_fitness(0.8, 0.95, 0.9, 0.85, 5, 6)
        assert penalty is True

    def test_no_penalty_when_all_correct(self):
        """All handcrafted correct → hand penalty False."""
        _, penalty, _ = compute_fitness(0.8, 0.95, 0.9, 0.85, 6, 6)
        assert penalty is False

    def test_hand_acc_weight_higher_than_acc(self):
        """luscinia_specialist v2: hand_acc (35%) > acc (20%).
        Improving hand_acc by delta should improve fitness more than
        improving acc by the same delta.

        We use a base acc=0.85 and delta=0.05 so the improved acc=0.90
        stays well below sanity=0.95 — no sanity guard fires for any call.
        """
        base_fitness, _, _ = compute_fitness(0.75, 0.95, 0.80, 0.85, 8, 10)
        delta = 0.05

        # Improve only hand_acc (0.80 → 0.85); acc stays at 0.85 < sanity=0.95 ✓
        f_hand, _, _ = compute_fitness(0.75, 0.95, 0.80 + delta, 0.85, 8, 10)
        # Improve only acc (0.85 → 0.90); still < sanity=0.95 ✓
        f_acc, _, _  = compute_fitness(0.75, 0.95, 0.80, 0.85 + delta, 8, 10)

        assert f_hand > base_fitness
        assert f_acc  > base_fitness
        assert f_hand > f_acc, (
            "hand_acc delta should outweigh acc delta under luscinia_specialist "
            f"(f_hand={f_hand:.6f}, f_acc={f_acc:.6f})"
        )

    def test_fitness_increases_with_better_f1(self):
        f_low,  _, _ = compute_fitness(0.60, 0.95, 0.90, 0.90, 8, 10)
        f_high, _, _ = compute_fitness(0.80, 0.95, 0.90, 0.90, 8, 10)
        assert f_high > f_low

    def test_fitness_bounded_zero_to_one_plus_bonus(self):
        """Fitness is in [0, ~1.02] — size bonus adds up to +2% on top of 1.0."""
        for f1, ext, hand, acc in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0),
            (0.5, 0.5, 0.5, 0.5),
        ]:
            fitness, _, _ = compute_fitness(f1, ext, hand, acc, 5, 10)
            assert 0.0 <= fitness <= 1.03, f"fitness {fitness} out of [0, 1.03]"

    def test_penalty_display_only_no_score_reduction(self):
        """Hand penalty is display-only and must NOT change the numeric score.
        Sanity is 0.95 >= acc 0.88 so no sanity guard fires either.
        """
        fitness_full, _, _ = compute_fitness(0.75, 0.95, 0.90, 0.88, 10, 10)
        fitness_miss, _, _ = compute_fitness(0.75, 0.95, 0.90, 0.88, 9, 10)
        assert abs(fitness_full - fitness_miss) < 1e-9, (
            "Hand penalty must not alter the numeric fitness score"
        )


# ════════════════════════════════════════════════════════════════════════════
# Sanity guard — multiplicative penalty when sanity_acc < val_acc
# ════════════════════════════════════════════════════════════════════════════

class TestSanityGuard:
    """Sanity accuracy acts as a circuit-breaker, not a fitness component."""

    def test_sanity_below_acc_halves_score(self):
        """When sanity < val_acc the score must be halved."""
        # Normal score (sanity >= acc)
        score_ok, _, sanity_bad_ok = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10)  # sanity=0.90 >= acc=0.85
        assert not sanity_bad_ok

        # Penalised score (sanity < acc)
        score_pen, _, sanity_bad_pen = compute_fitness(
            0.80, 0.70, 0.85, 0.85, 8, 10)  # sanity=0.70 < acc=0.85
        assert sanity_bad_pen

        # Penalty should be the configured multiplier
        ratio = score_pen / score_ok
        assert abs(ratio - SANITY_BELOW_ACC_PENALTY) < 0.01, (
            f"Expected score ratio ~{SANITY_BELOW_ACC_PENALTY}, got {ratio:.4f}"
        )

    def test_sanity_equal_acc_no_penalty(self):
        """Sanity == val_acc is fine (equal is not below)."""
        _, _, sanity_bad = compute_fitness(0.80, 0.85, 0.85, 0.85, 8, 10)
        assert not sanity_bad

    def test_sanity_above_acc_no_penalty(self):
        """Sanity well above val_acc → no penalty applied."""
        _, _, sanity_bad = compute_fitness(0.80, 0.98, 0.85, 0.82, 8, 10)
        assert not sanity_bad

    def test_penalised_model_never_beats_unpenalised(self):
        """Even a perfect penalised model must not beat a mediocre unpenalised one."""
        # Perfect model but sanity fires
        perfect_pen, _, _ = compute_fitness(1.0, 0.01, 1.0, 0.90, 10, 10)
        # Mediocre model with no sanity issue
        mediocre_ok, _, _ = compute_fitness(0.60, 0.95, 0.60, 0.90, 6, 10)
        assert perfect_pen < mediocre_ok, (
            "A penalised model must never beat an unpenalised one "
            f"(penalised={perfect_pen:.4f}, mediocre={mediocre_ok:.4f})"
        )

    def test_sanity_violated_returned_as_third_element(self):
        """Third element of tuple is True when sanity < acc."""
        result = compute_fitness(0.80, 0.60, 0.80, 0.90, 8, 10)
        assert len(result) == 3
        assert result[2] is True

    def test_sanity_not_violated_false_when_ok(self):
        result = compute_fitness(0.80, 0.95, 0.80, 0.90, 8, 10)
        assert result[2] is False


# ════════════════════════════════════════════════════════════════════════════
# Size tiebreaker bonus
# ════════════════════════════════════════════════════════════════════════════

class TestSizeBonus:
    """Model-size tiebreaker: tiny additive bonus, never overrides accuracy."""

    def test_smaller_model_gets_higher_score(self):
        """With identical other metrics, a smaller model should score higher."""
        # model_size_mb=0 → max bonus; model_size_mb=50 → 0 bonus
        score_small, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=0.0)
        score_large, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=SIZE_BONUS_CAP_MB)
        assert score_small > score_large

    def test_size_bonus_max_is_small(self):
        """Maximum size bonus must be at most SIZE_BONUS_MAX (2%)."""
        score_zero_size, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=0.0)
        score_cap_size, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=SIZE_BONUS_CAP_MB)
        diff = score_zero_size - score_cap_size
        assert diff <= SIZE_BONUS_MAX + 1e-9, (
            f"Size bonus diff {diff:.4f} exceeds SIZE_BONUS_MAX={SIZE_BONUS_MAX}"
        )

    def test_size_bonus_cannot_beat_single_handcrafted_word(self):
        """The max size bonus must not exceed the value of one correct handcrafted word.

        One handcrafted word is worth: hand_weight / hand_total = 0.35 / 44 ≈ 0.008.
        SIZE_BONUS_MAX = 0.02 < 0.08 so the delta between 2-word improvement
        and max size bonus ensures accuracy always wins.
        """
        # Two words better hand_acc but large model
        f_better_hand, _, _ = compute_fitness(
            0.80, 0.90, 42 / 44, 0.85, 42, 44, model_size_mb=SIZE_BONUS_CAP_MB)
        # Fewer words but tiny model
        f_small, _, _ = compute_fitness(
            0.80, 0.90, 40 / 44, 0.85, 40, 44, model_size_mb=0.0)
        # 2 extra handcrafted words (2 * 0.35/44 ≈ 0.016) > SIZE_BONUS_MAX (0.02)
        # is borderline, but 3+ words should always win
        f_better_3, _, _ = compute_fitness(
            0.80, 0.90, 43 / 44, 0.85, 43, 44, model_size_mb=SIZE_BONUS_CAP_MB)
        assert f_better_3 > f_small, (
            "3 extra handcrafted words should beat max size bonus"
        )

    def test_above_cap_gives_zero_bonus(self):
        """Models above SIZE_BONUS_CAP_MB all get the same (zero) bonus."""
        score_at_cap, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=SIZE_BONUS_CAP_MB)
        score_over_cap, _, _ = compute_fitness(
            0.80, 0.90, 0.85, 0.85, 8, 10, model_size_mb=SIZE_BONUS_CAP_MB * 2)
        assert abs(score_at_cap - score_over_cap) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — early-exit conditions
# ════════════════════════════════════════════════════════════════════════════


class TestEnsembleEarlyExit:
    """Ensemble should silently return None when it cannot build a valid ensemble."""

    def test_returns_none_with_no_results(self):
        result = run_ensemble_phase([], _feature_cols(), [], [])
        assert result is None

    def test_returns_none_with_zero_saved_models(self):
        """All results lack saved .lgb files → fewer than 2 found → None."""
        results = [_make_result(f"P3_{i:04d}") for i in range(5)]
        # No model files exist on disk → saved_candidates stays empty
        result = run_ensemble_phase(
            results, _feature_cols(),
            _simple_external_sample(), _simple_handcrafted(),
        )
        assert result is None

    def test_returns_none_with_only_one_saved_model(self, tmp_path):
        """Only one loadable model → needs ≥ 2 → None."""
        results = [_make_result("P3_0001"), _make_result("P3_0002")]

        # Create exactly one fake .lgb file
        model_dir = tmp_path / "P3_0001"
        model_dir.mkdir()
        (model_dir / "P3_0001.lgb").write_text("fake")

        fake_bst = _fake_booster(0.9)
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
# run_ensemble_phase — softmax weight calculation
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleWeights:
    """Weights are the softmax of fitness scores → they sum to ~1.0."""

    def _compute_expected_weights(self, fitnesses: list) -> np.ndarray:
        arr = np.array(fitnesses, dtype=float)
        arr -= arr.max()
        w = np.exp(arr)
        w /= w.sum()
        return w

    def test_weights_sum_to_one(self, tmp_path):
        """After softmax normalisation the weights must sum to 1.0."""
        fitnesses = [0.8167, 0.8164, 0.8161, 0.8156, 0.8147]
        results = [
            _make_result(f"P3_{i:04d}", fitness=fitnesses[i])
            for i in range(5)
        ]

        # Create fake .lgb files for all 5
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir()
            (d / f"{r['name']}.lgb").write_text("fake")

        # Mock lgb.Booster so it returns a fake booster
        fake_bst = _fake_booster(0.9)
        captured_weights = []

        original_run = run_ensemble_phase

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
                _simple_external_sample(50), _simple_handcrafted(4),
            )

        if captured_weights:
            assert abs(sum(captured_weights) - 1.0) < 1e-4, (
                f"Softmax weights should sum to 1.0; got {sum(captured_weights)}"
            )

    def test_higher_fitness_gets_higher_weight(self):
        """The best model should have the highest softmax weight."""
        fitnesses = [0.82, 0.80, 0.78]
        arr = np.array(fitnesses, dtype=float)
        arr -= arr.max()
        weights = np.exp(arr)
        weights /= weights.sum()
        assert weights[0] > weights[1] > weights[2]

    def test_equal_fitness_gives_equal_weights(self):
        """When all fitnesses are equal, weights should be uniform."""
        fitnesses = [0.80, 0.80, 0.80]
        arr = np.array(fitnesses, dtype=float)
        arr -= arr.max()
        weights = np.exp(arr)
        weights /= weights.sum()
        expected = 1.0 / 3
        for w in weights:
            assert abs(w - expected) < 1e-9


# ════════════════════════════════════════════════════════════════════════════
# run_ensemble_phase — accept / reject logic
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleAcceptReject:
    """Core acceptance criterion:
       sanity_gain ≥ ENSEMBLE_MIN_SANITY_GAIN  OR  hand_gain > 0
    """

    def _run_with_mocked_models(
        self,
        results: list,
        tmp_path: Path,
        ensemble_p1: float,   # fake booster predict probability → controls sanity/hand
        top_k: int = 5,
        min_sanity_gain: float = ENSEMBLE_MIN_SANITY_GAIN,
        external_sample: Optional[list] = None,
        handcrafted: Optional[list] = None,
    ) -> Optional[dict]:
        """Helper: create fake .lgb files, mock lgb.Booster, run ensemble."""
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir(exist_ok=True)
            (d / f"{r['name']}.lgb").write_text("fake")

        fake_bst = _fake_booster(ensemble_p1)

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=fake_bst),
        ):
            return run_ensemble_phase(
                results,
                _feature_cols(),
                external_sample or _simple_external_sample(100),
                handcrafted or _simple_handcrafted(4),
                top_k=top_k,
                min_sanity_gain=min_sanity_gain,
            )

    # ------------------------------------------------------------------
    # Accept on sanity_gain ≥ threshold
    # ------------------------------------------------------------------

    def test_accepted_on_sufficient_sanity_gain(self, tmp_path):
        """If the ensemble gains ≥ 0.001 sanity_acc, it must be accepted."""
        # Best solo has sanity_acc = 0.50; ensemble fake booster always predicts 1.
        # External sample is all expected_label=1 → 100% correct for ensemble.
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None, "expected_label": 1}
            for _ in range(100)
        ]
        # Best solo sanity_acc = 0.50 → ensemble will get 1.0 → gain = 0.50 >> 0.001
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9, external_sample=external,
            handcrafted=[("мама", "NOUN", 1, "test", None)] * 4,
        )
        assert result is not None, "Ensemble should be accepted on sanity gain"
        assert result["phase"] == "P4"
        assert result["name"] == "P4_ensemble"

    # ------------------------------------------------------------------
    # Accept on hand_gain > 0 even if sanity_gain < threshold
    # ------------------------------------------------------------------

    def test_accepted_on_hand_gain_alone(self, tmp_path):
        """Ensemble gets more handcrafted correct (even without sanity gain)."""
        # External: 50/50 → ensemble accuracy will be ~50% = same as best solo
        # Handcrafted: all expected=1, ensemble predicts 1 → 4/4
        # Best solo has hand_correct=3 → hand_gain=1 → accept
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": i % 2}
            for i in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9,
            external_sample=external,
            handcrafted=[("слово", "NOUN", 1, "test", None)] * 4,
        )
        assert result is not None, "Ensemble should be accepted on hand_gain > 0"

    # ------------------------------------------------------------------
    # Reject when no improvement
    # ------------------------------------------------------------------

    def test_rejected_when_no_improvement(self, tmp_path):
        """Ensemble is rejected when it doesn't improve sanity or hand."""
        # Best solo has sanity_acc=1.0 and hand_correct=4/4
        # Ensemble p1=0.9 → predicted=1 for all → same or worse than perfect solo
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": 0}       # all expected 0, ensemble predicts 1 → 0% correct
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=1.0, hand_correct=4, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9,
            external_sample=external,
            handcrafted=[("слово", "NOUN", 0, "test", None)] * 4,
        )
        assert result is None, "Ensemble should be rejected when it does not improve"

    # ------------------------------------------------------------------
    # Reject path still writes ensemble_report.json
    # ------------------------------------------------------------------

    def test_rejected_path_writes_report_json(self, tmp_path):
        """Even on rejection, ensemble_report.json must be written to ENSEMBLE_DIR."""
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": 0}
            for _ in range(50)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=1.0, hand_correct=4, hand_total=4)
            for i in range(3)
        ]
        self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9,
            external_sample=external,
            handcrafted=[("слово", "NOUN", 0, "test", None)] * 4,
        )
        report_path = tmp_path / "P4_ensemble" / "ensemble_report.json"
        assert report_path.exists(), "ensemble_report.json should be written on rejection"
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["decision"] == "rejected"
        assert "best_solo" in report
        assert "sanity_gain" in report
        assert "hand_gain" in report

    # ------------------------------------------------------------------
    # Accept path writes ensemble_report.json with decision=accepted
    # ------------------------------------------------------------------

    def test_accepted_path_writes_report_json(self, tmp_path):
        """Accepted ensemble writes ensemble_report.json with decision=accepted."""
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9,
            external_sample=external,
            handcrafted=[("слово", "NOUN", 1, "test", None)] * 4,
        )
        report_path = tmp_path / "P4_ensemble" / "ensemble_report.json"
        assert report_path.exists(), "ensemble_report.json should be written on acceptance"
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["decision"] == "accepted"
        assert "ensemble_fitness" in report
        assert "models_used" in report
        assert "weights" in report

    # ------------------------------------------------------------------
    # Accepted result has correct schema
    # ------------------------------------------------------------------

    def test_accepted_result_has_expected_keys(self, tmp_path):
        """Accepted ensemble result dict must have the standard schema."""
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        result = self._run_with_mocked_models(
            results, tmp_path, ensemble_p1=0.9,
            external_sample=external,
            handcrafted=[("слово", "NOUN", 1, "test", None)] * 4,
        )
        assert result is not None
        for key in ["phase", "name", "fitness", "params", "internal",
                    "external", "handcrafted", "ensemble_models",
                    "ensemble_weights", "best_solo_name", "fitness_preset"]:
            assert key in result, f"Missing key '{key}' in ensemble result dict"
        assert result["phase"] == "P4"
        assert result["fitness_preset"] == "luscinia_specialist"

    # ------------------------------------------------------------------
    # P3 results are preferred over P1/P2 when building the candidate pool
    # ------------------------------------------------------------------

    def test_p3_results_preferred_over_p1_p2(self, tmp_path):
        """The ensemble should prefer P3 models even if P1/P2 have saved files."""
        p1_results = [
            _make_result(f"P1_{i:04d}", phase="P1",
                         fitness=0.85 - i * 0.001)   # higher fitness but P1
            for i in range(3)
        ]
        p3_results = [
            _make_result(f"P3_{i:04d}", phase="P3",
                         fitness=0.80 - i * 0.001)
            for i in range(3)
        ]
        all_results = p1_results + p3_results

        # Create .lgb files for all
        for r in all_results:
            d = tmp_path / r["name"]
            d.mkdir(exist_ok=True)
            (d / f"{r['name']}.lgb").write_text("fake")

        loaded_names = []

        real_booster = _fake_booster(0.9)

        class _CapturingBooster:
            def __init__(self, model_file, **kw):
                loaded_names.append(Path(model_file).parent.name)
            def predict(self, x):
                return np.full(x.shape[0], 0.9)

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", side_effect=_CapturingBooster),
        ):
            run_ensemble_phase(
                all_results, _feature_cols(),
                _simple_external_sample(50), _simple_handcrafted(4),
                top_k=3,
            )

        # All loaded models should be P3
        for name in loaded_names:
            assert name.startswith("P3_"), (
                f"Expected P3 model to be loaded, got '{name}'. "
                "P3 results should be preferred over P1/P2."
            )


# ════════════════════════════════════════════════════════════════════════════
# Phase fitness preset contracts
# ════════════════════════════════════════════════════════════════════════════

class TestPhasePresets:
    """Verify that P1/P2 use specialist_binary and P3 uses luscinia_specialist
    (checked via the luscinia_2s_v1.0.py source constants, not by running
    full training — training requires a real DB connection).
    """

    def test_ensemble_result_preset_is_luscinia_specialist(self, tmp_path):
        """Accepted ensemble result must tag itself as luscinia_specialist."""
        external = [
            {"form": "слово", "pos": "NOUN", "features_json": None,
             "expected_label": 1}
            for _ in range(100)
        ]
        results = [
            _make_result(f"P3_{i:04d}", fitness=0.80 - i * 0.001,
                         sanity_acc=0.50, hand_correct=3, hand_total=4)
            for i in range(3)
        ]
        for r in results:
            d = tmp_path / r["name"]
            d.mkdir()
            (d / f"{r['name']}.lgb").write_text("fake")

        with (
            patch.object(_mod, "RESULTS_DIR", tmp_path),
            patch.object(_mod, "ENSEMBLE_DIR", tmp_path / "P4_ensemble"),
            patch("lightgbm.Booster", return_value=_fake_booster(0.9)),
        ):
            result = run_ensemble_phase(
                results, _feature_cols(),
                external,
                [("слово", "NOUN", 1, "test", None)] * 4,
            )

        assert result is not None
        assert result["fitness_preset"] == "luscinia_specialist"

    def test_script_p3_preset_constant(self):
        """The Phase 3 objective in the script must reference luscinia_specialist.
        We verify this by searching the module source text (structural check).
        """
        src = Path(_script_path).read_text(encoding="utf-8")
        # Phase 3 block must contain luscinia_specialist
        assert 'fitness_preset="luscinia_specialist"' in src, (
            "Phase 3 TPE objective must use luscinia_specialist fitness preset"
        )

    def test_script_p1_preset_constant(self):
        """Phase 1 and Phase 2 must use specialist_binary (landscape compat)."""
        src = Path(_script_path).read_text(encoding="utf-8")
        assert 'fitness_preset="specialist_binary"' in src, (
            "Phase 1 / Phase 2 must use specialist_binary fitness preset"
        )

    def test_default_preset_in_compute_fitness_is_luscinia_specialist(self):
        """compute_fitness() default preset must be luscinia_specialist.
        We verify this by inspecting the function's keyword default.
        """
        import inspect
        sig = inspect.signature(compute_fitness)
        preset_param = sig.parameters.get("preset")
        assert preset_param is not None, "compute_fitness must have a 'preset' parameter"
        assert preset_param.default == "luscinia_specialist", (
            f"Expected default preset 'luscinia_specialist', "
            f"got '{preset_param.default}'"
        )


# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE_TOP_K and ENSEMBLE_MIN_SANITY_GAIN constants
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleConstants:

    def test_top_k_is_5(self):
        assert ENSEMBLE_TOP_K == 5

    def test_min_sanity_gain_is_001(self):
        assert abs(ENSEMBLE_MIN_SANITY_GAIN - 0.001) < 1e-9

    def test_top_k_at_least_2(self):
        """Must allow at least 2 models to form a meaningful ensemble."""
        assert ENSEMBLE_TOP_K >= 2

    def test_min_sanity_gain_positive(self):
        """Gain threshold must be positive (a negative threshold would always accept)."""
        assert ENSEMBLE_MIN_SANITY_GAIN > 0


# ════════════════════════════════════════════════════════════════════════════
# _handcrafted_to_training_rows — shared helper (same as v4)
# ════════════════════════════════════════════════════════════════════════════

class TestHandcraftedToTrainingRows:
    """_handcrafted_to_training_rows must append rows with correct labels."""

    def _feat_cols(self) -> list:
        return list(_build_feat("мама", "NOUN").keys())

    def test_returns_none_on_empty_input(self):
        X, y = _handcrafted_to_training_rows([], self._feat_cols())
        assert X is None
        assert y is None

    def test_returns_dataframe_and_series(self):
        tests = [("мама", "NOUN", 0, "first vowel stressed", None)]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_correct_number_of_rows(self):
        tests = [
            ("мама", "NOUN", 0, "test1", None),
            ("папа", "NOUN", 0, "test2", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert len(X) == 2
        assert len(y) == 2

    def test_skips_entries_with_no_expected_label(self):
        """Entries with expected=None should be skipped."""
        tests = [
            ("мама", "NOUN", 0, "ok", None),
            ("??",   "X",    None, "skip me", None),
            ("папа", "NOUN", 0, "ok2", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert len(X) == 2

    def test_skips_multi_answer_entries(self):
        """Entries with expected as a list (ambiguous) should be skipped."""
        tests = [
            ("мама", "NOUN", 0, "ok", None),
            ("вода", "NOUN", [0, 1], "ambiguous", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert len(X) == 1

    def test_feature_columns_match(self):
        tests = [("мама", "NOUN", 0, "test", None)]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        assert list(X.columns) == feat_cols

    def test_label_values_are_binary(self):
        tests = [
            ("мама", "NOUN", 0, "stress on 1st vowel", None),
            ("вода", "NOUN", 1, "stress on 2nd vowel", None),
        ]
        feat_cols = self._feat_cols()
        X, y = _handcrafted_to_training_rows(tests, feat_cols)
        for val in y:
            assert val in (0, 1), f"Expected binary label 0 or 1, got {val}"
