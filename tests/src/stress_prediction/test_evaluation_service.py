"""Tests for evaluation_service — unified binary + multiclass evaluation.

Uses mock boosters (fake .predict()) to avoid needing a real lightgbm model.
Covers:
  * compute_fitness — v1.4 backward-compat (positional), preset="specialist_binary",
    custom weights, penalty_mode, bounds
  * evaluate_handcrafted — multiclass (default) and binary modes
  * evaluate_external — multiclass (default) and binary modes
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "src", "stress_prediction", "lightgbm",
    ),
)

from services.evaluation_service import (
    compute_fitness,
    evaluate_external,
    evaluate_handcrafted,
)
from services.feature_service import build_features_v13
from services.constants import EXPECTED_FEATURE_COUNT, NUM_CLASSES


# ════════════════════════════════════════════════════════════════════
# Fake boosters
# ════════════════════════════════════════════════════════════════════

class FakeBoosterMulticlass:
    """Stub that always predicts a fixed class via argmax (multiclass)."""

    def __init__(self, fixed_class: int = 0, num_classes: int = NUM_CLASSES):
        self._fixed = fixed_class
        self._nc = num_classes

    def predict(self, data):
        n = data.shape[0]
        proba = np.full((n, self._nc), 0.01)
        proba[:, self._fixed] = 0.90
        return proba


# Keep old name as alias so existing test code doesn't need changing
FakeBooster = FakeBoosterMulticlass


class FakeBoosterBinary:
    """Stub that always returns a fixed scalar probability (binary mode)."""

    def __init__(self, p1: float = 0.9):
        self._p1 = p1

    def predict(self, data):
        n = data.shape[0]
        return np.full(n, self._p1)


class FakeBoosterFromMap:
    """Stub that returns predictions from an ordinal call→class mapping."""

    def __init__(self, call_to_class: dict, num_classes: int = NUM_CLASSES):
        self._map = call_to_class
        self._nc = num_classes
        self._call_count = 0

    def predict(self, data):
        n = data.shape[0]
        proba = np.full((n, self._nc), 0.01)
        cls = self._map.get(self._call_count, 0)
        proba[:, cls] = 0.90
        self._call_count += 1
        return proba


# ════════════════════════════════════════════════════════════════════
# compute_fitness  — v1.4 backward-compat (global_v1_4 preset)
# ════════════════════════════════════════════════════════════════════
class TestComputeFitnessGlobalV14:
    """Original 4-arg positional API must stay unchanged (used by v1.4 script).

    NOTE: A size_bonus of up to +0.02 is now added to all results because
    model_size_mb defaults to 0.  Tests that check exact scores add 0.02.
    """

    def test_perfect_scores(self):
        f = compute_fitness(1.0, 1.0, 1.0, 1.0)
        # 1.0 weighted + 0.02 size bonus
        assert abs(f - 1.02) < 1e-9

    def test_zero_scores(self):
        f = compute_fitness(0.0, 0.0, 0.0, 0.0)
        # 0.0 weighted + 0.02 size bonus (model_size_mb=0)
        assert abs(f - 0.02) < 1e-9

    def test_weights_sum_to_one(self):
        """0.30+0.25+0.20+0.25=1.0, so perfect inputs + size_bonus = 1.02."""
        f = compute_fitness(1.0, 1.0, 1.0, 1.0)
        assert abs(f - 1.02) < 1e-9

    def test_partial_scores(self):
        f = compute_fitness(0.7, 0.95, 0.5, 0.8)
        expected = 0.30 * 0.7 + 0.25 * 0.95 + 0.20 * 0.5 + 0.25 * 0.8 + 0.02
        assert abs(f - expected) < 1e-9

    def test_bounds(self):
        """Fitness must stay in [0, 1.03] for inputs in [0, 1] (size bonus included)."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            args = [rng.random() for _ in range(4)]
            f = compute_fitness(*args)
            assert 0.0 <= f <= 1.03 + 1e-9

    def test_f1_is_dominant_weight(self):
        """F1 weight (0.30) > ext_acc weight (0.25); no sanity guard (acc not passed)."""
        f_high_f1 = compute_fitness(1.0, 0.0, 0.0, 0.0)   # no acc= → no sanity guard
        f_high_ext = compute_fitness(0.0, 1.0, 0.0, 0.0)
        # f_high_f1 = 0.30 + 0.02; f_high_ext = 0.25 + 0.02
        assert f_high_f1 > f_high_ext

    def test_returns_float_not_tuple(self):
        """Default call must return a bare float, not a tuple."""
        result = compute_fitness(0.8, 0.9, 0.7, 0.85)
        assert isinstance(result, float)


# ════════════════════════════════════════════════════════════════════
# compute_fitness  — specialist_binary preset  (used by 2-syl v4, etc.)
# ════════════════════════════════════════════════════════════════════
class TestComputeFitnessSpecialistBinary:
    """v2 weights: f1=45%, acc=30%, hand_acc=25% (sanity removed from formula).
    compute_fitness now returns a 3-tuple (score, hand_penalty, sanity_violated).
    """

    def test_perfect_scores(self):
        fitness, penalty, _sv = compute_fitness(
            1.0, 1.0, 1.0,
            acc=1.0, hand_correct=6, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        assert fitness > 1.0, "Perfect scores should exceed 1.0 due to size bonus"
        assert penalty is False

    def test_zero_scores(self):
        fitness, penalty, _sv = compute_fitness(
            0.0, 0.0, 0.0,
            acc=0.0, hand_correct=0, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        # acc=0 and ext_acc=0 → equal (not strict less-than) → no sanity penalty
        # score = 0 + size_bonus
        assert fitness < 0.03
        assert penalty is True

    def test_weights_45_30_25(self):
        """f1=45%, acc=30%, hand_acc=25% (v2)."""
        fitness, _, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=6, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        # sanity=0.95 >= acc=0.85 → no guard; model_size_mb=0 → max size bonus
        expected = 0.45 * 0.8 + 0.30 * 0.85 + 0.25 * 0.9 + 0.02
        assert abs(fitness - expected) < 1e-6

    def test_penalty_flag_only_does_not_change_score(self):
        """Missing handcrafted words raise penalty flag but DO NOT reduce score."""
        fitness_full, _, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=6, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        fitness_miss, penalty, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=5, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        assert abs(fitness_full - fitness_miss) < 1e-9
        assert penalty is True

    def test_no_penalty_when_all_correct(self):
        _, penalty, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=6, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        assert penalty is False

    def test_penalty_when_any_miss(self):
        _, penalty, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=5, hand_total=6,
            preset="specialist_binary", penalty_mode=True,
        )
        assert penalty is True

    def test_zero_hand_total_no_penalty(self):
        """hand_total=0 means no scoreable tests → no penalty."""
        _, penalty, _sv = compute_fitness(
            0.8, 0.9, 0.0,
            acc=0.85, hand_correct=0, hand_total=0,
            preset="specialist_binary", penalty_mode=True,
        )
        assert penalty is False

    def test_f1_dominates(self):
        """f1 (45%) is the single highest-weight component."""
        f_high_f1, _, _sv = compute_fitness(
            1.0, 0.95, 0.0,
            acc=0.0, hand_correct=0, hand_total=0,
            preset="specialist_binary", penalty_mode=True,
        )
        f_high_acc, _, _sv2 = compute_fitness(
            0.0, 0.95, 0.0,
            acc=1.0, hand_correct=0, hand_total=0,
            preset="specialist_binary", penalty_mode=True,
        )
        # f1=45% weight > acc=30% weight
        assert f_high_f1 > f_high_acc

    def test_bounds_random(self):
        import random
        rng = random.Random(99)
        for _ in range(200):
            f1 = rng.random()
            ext = rng.random()
            acc_v = rng.random()
            # ensure ext >= acc to avoid sanity guard mixing things
            acc_v_safe = min(ext, rng.random())
            total = rng.randint(0, 10)
            correct = rng.randint(0, total)
            fitness, _, _sv = compute_fitness(
                f1, ext, 0.5, acc=acc_v_safe,
                hand_correct=correct, hand_total=total,
                preset="specialist_binary", penalty_mode=True,
            )
            assert 0.0 <= fitness <= 1.03 + 1e-9


# ════════════════════════════════════════════════════════════════════
# compute_fitness  — custom weights
# ════════════════════════════════════════════════════════════════════
class TestComputeFitnessCustomWeights:
    """Custom weights override preset.  Note: size bonus (+0.02 max) is always added."""

    def test_custom_weights_applied(self):
        w = {"f1": 0.50, "ext_acc": 0.50}
        f = compute_fitness(0.8, 0.6, 0.0, weights=w)
        # 0.50*0.8 + 0.50*0.6 + 0.02 size_bonus = 0.70 + 0.02
        assert abs(f - (0.50 * 0.8 + 0.50 * 0.6 + 0.02)) < 1e-9

    def test_custom_weights_with_penalty_mode(self):
        w = {"ext_acc": 1.0}
        fitness, penalty, _sv = compute_fitness(
            0.0, 0.9, 0.0, weights=w,
            hand_correct=0, hand_total=3, penalty_mode=True,
        )
        # 1.0*0.9 + 0.02 size_bonus; no acc= passed → no sanity guard
        assert abs(fitness - (0.9 + 0.02)) < 1e-9
        assert penalty is True

    def test_custom_weights_override_preset(self):
        """Explicit weights take priority over preset.  Score includes size_bonus."""
        w = {"f1": 1.0}
        f = compute_fitness(0.7, 0.9, 0.0, weights=w, preset="specialist_binary")
        # 1.0*0.7 + 0.02 = 0.72
        assert abs(f - (0.7 + 0.02)) < 1e-9


# ════════════════════════════════════════════════════════════════════
# compute_fitness  — luscinia_specialist preset  (Luscinia v2.0, P3)
# ════════════════════════════════════════════════════════════════════
class TestComputeFitnessLusciniaSpecialist:
    """Verify the luscinia_specialist preset: f1=45%, acc=20%, hand_acc=35%.

    Redesign v2: sanity_acc removed from formula entirely.
    compute_fitness now returns a 3-tuple (score, hand_penalty, sanity_violated).
    """

    def test_perfect_scores(self):
        fitness, penalty, _sv = compute_fitness(
            1.0, 1.0, 1.0,
            acc=1.0, hand_correct=6, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert fitness > 1.0, "Perfect scores + size bonus should exceed 1.0"
        assert penalty is False

    def test_zero_scores(self):
        fitness, penalty, _sv = compute_fitness(
            0.0, 0.0, 0.0,
            acc=0.0, hand_correct=0, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        # acc=0 and ext_acc=0 → equal, not strict less-than → no sanity penalty
        assert fitness < 0.03   # only size_bonus
        assert penalty is True

    def test_weights_45_20_35(self):
        """v2: f1=45%, acc=20%, hand_acc=35% (no sanity weight)."""
        fitness, _, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=6, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        # sanity=0.95 >= acc=0.85 → no guard; model_size_mb=0 → max size bonus
        expected = 0.45 * 0.8 + 0.20 * 0.85 + 0.35 * 0.9 + 0.02
        assert abs(fitness - expected) < 1e-6

    def test_hand_acc_outweighs_acc(self):
        """hand_acc (35%) beats acc (20%): same f1, higher hand_acc wins."""
        # Use sanity well above acc to avoid any guard effects
        f_hand, _, _sv = compute_fitness(
            0.8, 0.9, 1.0,
            acc=0.5, hand_correct=6, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        f_acc, _, _sv2 = compute_fitness(
            0.8, 0.9, 0.5,
            acc=0.8, hand_correct=3, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert f_hand > f_acc

    def test_luscinia_vs_specialist_binary_hand_acc_matters_more(self):
        """A boost in hand_acc raises luscinia fitness more than specialist_binary."""
        base_kwargs = dict(
            acc=0.85, hand_correct=4, hand_total=6, penalty_mode=True,
        )
        low_hand = (0.8, 0.9, 0.7)
        high_hand = (0.8, 0.9, 1.0)

        gain_luscinia = (
            compute_fitness(*high_hand, preset="luscinia_specialist", **base_kwargs)[0]
            - compute_fitness(*low_hand,  preset="luscinia_specialist", **base_kwargs)[0]
        )
        gain_specialist = (
            compute_fitness(*high_hand, preset="specialist_binary", **base_kwargs)[0]
            - compute_fitness(*low_hand,  preset="specialist_binary", **base_kwargs)[0]
        )
        # luscinia rewards hand_acc improvement more (weight 35% vs 25%)
        assert gain_luscinia > gain_specialist

    def test_penalty_flag_does_not_reduce_score(self):
        """Hand penalty is display-only (sanity=0.95 >= acc=0.85, no guard)."""
        fitness_full, _, _sv = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=6, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        fitness_miss, penalty, _sv2 = compute_fitness(
            0.8, 0.95, 0.9,
            acc=0.85, hand_correct=5, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert abs(fitness_full - fitness_miss) < 1e-9
        assert penalty is True

    def test_f1_dominates(self):
        """f1 (45%) is the single highest-weight component."""
        f_high_f1, _, _sv = compute_fitness(
            1.0, 0.95, 0.0,
            acc=0.0, hand_correct=0, hand_total=0,
            preset="luscinia_specialist", penalty_mode=True,
        )
        f_high_hand, _, _sv2 = compute_fitness(
            0.0, 0.95, 1.0,
            acc=0.0, hand_correct=6, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert f_high_f1 > f_high_hand

    def test_bounds_random(self):
        import random
        rng = random.Random(42)
        for _ in range(200):
            f1 = rng.random()
            ext = rng.random()
            hand = rng.random()
            # keep acc <= ext to avoid sanity guard complicating bounds check
            acc_v = min(ext, rng.random())
            total = rng.randint(0, 10)
            correct = rng.randint(0, total)
            fitness, _, _sv = compute_fitness(
                f1, ext, hand,
                acc=acc_v, hand_correct=correct, hand_total=total,
                preset="luscinia_specialist", penalty_mode=True,
            )
            assert 0.0 <= fitness <= 1.03 + 1e-9

    def test_returns_tuple_in_penalty_mode(self):
        result = compute_fitness(
            0.8, 0.9, 0.85,
            acc=0.82, hand_correct=5, hand_total=6,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3   # (fitness, hand_penalty, sanity_violated)
        fitness, penalty, sanity_violated = result
        assert isinstance(fitness, float)
        assert isinstance(penalty, bool)
        assert isinstance(sanity_violated, bool)

    def test_returns_float_without_penalty_mode(self):
        result = compute_fitness(
            0.8, 0.9, 0.85,
            acc=0.82,
            preset="luscinia_specialist",
        )
        assert isinstance(result, float)

    def test_weights_sum_to_one_with_bonus(self):
        """45+20+35=100. Perfect inputs + size_bonus > 1.0."""
        fitness, _, _sv = compute_fitness(
            1.0, 1.0, 1.0,
            acc=1.0, hand_correct=1, hand_total=1,
            preset="luscinia_specialist", penalty_mode=True,
        )
        assert fitness > 1.0


# ════════════════════════════════════════════════════════════════════
# evaluate_handcrafted — multiclass (default)
# ════════════════════════════════════════════════════════════════════
class TestEvaluateHandcraftedMulticlass:

    def _cols(self):
        return list(build_features_v13("тест", "NOUN").keys())

    def test_all_correct(self):
        booster = FakeBooster(fixed_class=0)
        tests = [("мама", "NOUN", 0, "t1"), ("тато", "NOUN", 0, "t2")]
        result = evaluate_handcrafted(booster, self._cols(), tests)
        assert result["correct"] == 2
        assert result["total"] == 2
        assert abs(result["accuracy"] - 1.0) < 1e-6

    def test_none_excluded_from_score(self):
        booster = FakeBooster(fixed_class=0)
        tests = [
            ("мама", "NOUN", 0, "scoreable"),
            ("щось", "NOUN", None, "unknown"),
        ]
        result = evaluate_handcrafted(booster, self._cols(), tests)
        assert result["total"] == 1
        assert result["total_words"] == 2
        assert result["correct"] == 1

    def test_ambiguous_either_accepted(self):
        booster = FakeBooster(fixed_class=1)
        tests = [("цеглоїд", "NOUN", [1, 2], "ambiguous")]
        result = evaluate_handcrafted(booster, self._cols(), tests)
        assert result["correct"] == 1

    def test_wrong_prediction(self):
        booster = FakeBooster(fixed_class=5)
        tests = [("мама", "NOUN", 0, "miss")]
        result = evaluate_handcrafted(booster, self._cols(), tests)
        assert result["correct"] == 0
        assert result["accuracy"] == 0.0

    def test_empty(self):
        booster = FakeBooster(fixed_class=0)
        result = evaluate_handcrafted(booster, self._cols(), [])
        assert result["total"] == 0
        assert result["accuracy"] == 0.0

    def test_result_keys(self):
        booster = FakeBooster(fixed_class=0)
        result = evaluate_handcrafted(booster, self._cols(), [("мама", "NOUN", 0, "t")])
        for key in ("accuracy", "correct", "total", "total_words", "results"):
            assert key in result
        row = result["results"][0]
        for key in ("word", "predicted", "correct", "confidence", "scoreable"):
            assert key in row

    def test_tuple_without_description(self):
        """3-element tuples (word, pos, expected) should be accepted."""
        booster = FakeBooster(fixed_class=0)
        result = evaluate_handcrafted(booster, self._cols(), [("мама", "NOUN", 0)])
        assert result["total"] == 1

    def test_tuple_with_features_json(self):
        """5-element tuples with features_json should be accepted."""
        booster = FakeBooster(fixed_class=0)
        result = evaluate_handcrafted(
            booster, self._cols(),
            [("мама", "NOUN", 0, "desc", None)],
        )
        assert result["total"] == 1


# ════════════════════════════════════════════════════════════════════
# evaluate_handcrafted — binary mode  (2-syl specialist)
# ════════════════════════════════════════════════════════════════════
class TestEvaluateHandcraftedBinary:

    def _cols(self):
        return list(build_features_v13("тест", "NOUN").keys())

    def test_binary_correct_class1(self):
        """p1=0.9 → predicted=1; expected=1 → correct."""
        booster = FakeBoosterBinary(p1=0.9)
        result = evaluate_handcrafted(
            booster, self._cols(), [("сестра", "NOUN", 1, "t")],
            mode="binary",
        )
        assert result["correct"] == 1

    def test_binary_correct_class0(self):
        """p1=0.1 → predicted=0; expected=0 → correct."""
        booster = FakeBoosterBinary(p1=0.1)
        result = evaluate_handcrafted(
            booster, self._cols(), [("мама", "NOUN", 0, "t")],
            mode="binary",
        )
        assert result["correct"] == 1

    def test_binary_wrong(self):
        """p1=0.9 → predicted=1; expected=0 → miss."""
        booster = FakeBoosterBinary(p1=0.9)
        result = evaluate_handcrafted(
            booster, self._cols(), [("мама", "NOUN", 0, "t")],
            mode="binary",
        )
        assert result["correct"] == 0

    def test_binary_custom_threshold(self):
        """Threshold=0.8: p1=0.75 → predicted=0."""
        booster = FakeBoosterBinary(p1=0.75)
        result = evaluate_handcrafted(
            booster, self._cols(), [("мама", "NOUN", 0, "t")],
            mode="binary", threshold=0.8,
        )
        assert result["correct"] == 1

    def test_binary_ambiguous_list(self):
        booster = FakeBoosterBinary(p1=0.9)  # predicted=1
        result = evaluate_handcrafted(
            booster, self._cols(), [("слово", "NOUN", [0, 1], "t")],
            mode="binary",
        )
        assert result["correct"] == 1


# ════════════════════════════════════════════════════════════════════
# evaluate_external — multiclass (default)
# ════════════════════════════════════════════════════════════════════
class TestEvaluateExternalMulticlass:

    def _cols(self):
        return list(build_features_v13("тест", "NOUN").keys())

    def test_perfect(self):
        booster = FakeBooster(fixed_class=0)
        sample = [
            {"form": "мама", "pos": "NOUN", "expected_label": 0,
             "vowels": [1, 3], "n_syllables": 2},
            {"form": "тато", "pos": "NOUN", "expected_label": 0,
             "vowels": [1, 3], "n_syllables": 2},
        ]
        r = evaluate_external(booster, self._cols(), sample)
        assert r["correct"] == 2
        assert r["sample_size"] == 2
        assert abs(r["accuracy"] - 1.0) < 1e-6

    def test_all_wrong(self):
        booster = FakeBooster(fixed_class=5)
        sample = [{"form": "мама", "pos": "NOUN", "expected_label": 0,
                   "vowels": [1, 3], "n_syllables": 2}]
        r = evaluate_external(booster, self._cols(), sample)
        assert r["correct"] == 0
        assert r["accuracy"] == 0.0

    def test_per_syllable_present(self):
        booster = FakeBooster(fixed_class=0)
        sample = [
            {"form": "мама", "pos": "NOUN", "expected_label": 0,
             "vowels": [1, 3], "n_syllables": 2},
            {"form": "ведмедиця", "pos": "NOUN", "expected_label": 0,
             "vowels": [1, 4, 6, 8], "n_syllables": 4},
        ]
        r = evaluate_external(booster, self._cols(), sample)
        assert "per_syllable" in r
        assert "mean_syllable_accuracy" in r

    def test_empty(self):
        booster = FakeBooster(fixed_class=0)
        r = evaluate_external(booster, self._cols(), [])
        assert r["sample_size"] == 0
        assert r["accuracy"] == 0.0

    def test_result_keys(self):
        booster = FakeBooster(fixed_class=0)
        sample = [{"form": "мама", "pos": "NOUN", "expected_label": 0,
                   "n_syllables": 2}]
        r = evaluate_external(booster, self._cols(), sample)
        for key in ("accuracy", "correct", "sample_size",
                    "per_syllable", "mean_syllable_accuracy"):
            assert key in r


# ════════════════════════════════════════════════════════════════════
# evaluate_external — binary mode  (2-syl specialist)
# ════════════════════════════════════════════════════════════════════
class TestEvaluateExternalBinary:

    def _cols(self):
        return list(build_features_v13("тест", "NOUN").keys())

    def test_binary_correct(self):
        """p1=0.9 → predicted=1; item expected=1 → hit."""
        booster = FakeBoosterBinary(p1=0.9)
        sample = [{"form": "сестра", "pos": "NOUN", "expected_label": 1,
                   "n_syllables": 2}]
        r = evaluate_external(booster, self._cols(), sample, mode="binary")
        assert r["correct"] == 1
        assert abs(r["accuracy"] - 1.0) < 1e-6

    def test_binary_wrong(self):
        booster = FakeBoosterBinary(p1=0.9)  # predicts 1
        sample = [{"form": "мама", "pos": "NOUN", "expected_label": 0,
                   "n_syllables": 2}]
        r = evaluate_external(booster, self._cols(), sample, mode="binary")
        assert r["correct"] == 0

    def test_binary_threshold(self):
        booster = FakeBoosterBinary(p1=0.6)  # < 0.8 threshold → predicted=0
        sample = [{"form": "мама", "pos": "NOUN", "expected_label": 0,
                   "n_syllables": 2}]
        r = evaluate_external(
            booster, self._cols(), sample, mode="binary", threshold=0.8
        )
        assert r["correct"] == 1

    def test_binary_per_syllable_still_present(self):
        """Per-syllable breakdown works in binary mode too."""
        booster = FakeBoosterBinary(p1=0.9)
        sample = [
            {"form": "мама", "pos": "NOUN", "expected_label": 1, "n_syllables": 2},
            {"form": "сестра", "pos": "NOUN", "expected_label": 1, "n_syllables": 2},
        ]
        r = evaluate_external(booster, self._cols(), sample, mode="binary")
        assert "per_syllable" in r
        assert r["sample_size"] == 2
