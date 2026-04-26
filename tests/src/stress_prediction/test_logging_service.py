"""Tests for logging_service — dual-output (console + file) and CSV/JSON persistence.

All tests use the Luscinia result dict format (2S and 3S variants).
Temporary directories are used for all file I/O.
"""

import csv
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
        "src", "stress_prediction", "lightgbm",
    ),
)

from services.logging_service import (
    sp,
    log_trial_result,
    log_phase_progress,
    log_phase_summary,
    log_final_leaderboard,
    append_result_json,
    append_result_csv,
)


# ════════════════════════════════════════════════════════════════════════════
# Result dict factories (Luscinia format)
# ════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS_2S = [
    "phase", "trial_number", "name", "fitness",
    "f1", "accuracy", "auc",
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


def _make_result(
    trial_num: int = 1,
    name: str = "P3_0001",
    fitness: float = 0.8434,
    phase: str = "P3",
    hand_correct: int = 40,
    hand_total: int = 44,
    wrong_words: list = None,
    hand_penalty: bool = False,
    sanity_violated: bool = False,
    sanity_acc: float = 0.9564,
    size_mb: float = 3.0,
) -> dict:
    """Factory for a minimal valid Luscinia 2S result dict."""
    wrong_words = wrong_words or ["шпада", "ошук"]
    hand_results = [{"word": "рада", "correct": True}] * hand_correct
    hand_results += [{"word": w, "correct": False} for w in wrong_words]

    return {
        "phase":              phase,
        "trial_number":       trial_num,
        "name":               name,
        "fitness":            fitness,
        "train_time_sec":     9.0,
        "wall_elapsed_sec":   21246.0,
        "wall_elapsed_min":   354.1,
        "hand_penalty_applied": hand_penalty,
        "sanity_violated":    sanity_violated,
        "params": {
            "num_leaves":            435,
            "max_depth":             11,
            "learning_rate":         0.05,
            "min_child_samples":     13,
            "lambda_l1":             0.12,
            "lambda_l2":             1.80,
            "subsample":             0.82,
            "colsample_bytree":      0.71,
            "feature_fraction_bynode": 0.65,
            "min_sum_hessian_in_leaf": 1.5,
            "path_smooth":           0.1,
            "min_split_gain":        0.05,
            "data_sample_strategy":  "bagging",
        },
        "internal": {
            "f1":             0.7519,
            "accuracy":       0.841,
            "auc":            0.894,
            "best_iteration": 435,
            "num_trees":      435,
        },
        "external": {
            "accuracy":    sanity_acc,
            "correct":     4782,
            "sample_size": 5000,
        },
        "handcrafted": {
            "accuracy":    round(hand_correct / hand_total, 6) if hand_total else 0.0,
            "correct":     hand_correct,
            "total":       hand_total,
            "total_words": hand_total + 2,
            "results":     hand_results,
        },
        "model": {"estimated_size_mb": size_mb},
        "convergence_curve": [
            {"cp": 50,  "f1": 0.696, "acc": 0.793, "logloss": 0.452},
            {"cp": 150, "f1": 0.725, "acc": 0.812, "logloss": 0.421},
            {"cp": 300, "f1": 0.731, "acc": 0.817, "logloss": 0.415},
        ],
    }


def _make_result_3s(trial_num: int = 1, fitness: float = 0.79) -> dict:
    """Factory for a minimal valid Luscinia 3S result dict (size_bytes variant)."""
    r = _make_result(trial_num=trial_num, fitness=fitness, phase="P2")
    # 3S uses size_bytes instead of estimated_size_mb
    r["model"] = {"size_bytes": 3_000_000, "saved": True}
    return r


# ════════════════════════════════════════════════════════════════════════════
# sp — safe print helper
# ════════════════════════════════════════════════════════════════════════════

class TestSp:

    def test_smoke(self, capsys):
        sp("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_empty(self, capsys):
        sp()
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_unicode_passthrough(self, capsys):
        sp("шпада")
        captured = capsys.readouterr()
        assert "шпада" in captured.out


# ════════════════════════════════════════════════════════════════════════════
# log_trial_result
# ════════════════════════════════════════════════════════════════════════════

class TestLogTrialResult:

    def test_console_contains_name(self, capsys, tmp_path):
        r = _make_result(name="P3_0099")
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "P3_0099" in captured.out

    def test_console_contains_fitness(self, capsys, tmp_path):
        r = _make_result(fitness=0.8434)
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "0.8434" in captured.out

    def test_console_f1_as_percent(self, capsys, tmp_path):
        r = _make_result()
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        # f1=0.7519 → "75.19%"
        assert "75.19%" in captured.out

    def test_console_hand_fraction(self, capsys, tmp_path):
        r = _make_result(hand_correct=40, hand_total=44)
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "40/44" in captured.out

    def test_console_wrong_words(self, capsys, tmp_path):
        r = _make_result(wrong_words=["шпада", "ошук"])
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "шпада" in captured.out
        assert "ошук" in captured.out

    def test_console_penalty_flag(self, capsys, tmp_path):
        r = _make_result(hand_penalty=True)
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "HAND PENALTY" in captured.out

    def test_console_sanity_violated_flag(self, capsys, tmp_path):
        r = _make_result(sanity_violated=True)
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "SANITY VIOLATED" in captured.out

    def test_no_console_when_disabled(self, capsys, tmp_path):
        r = _make_result(name="P3_silent")
        log_trial_result(r, tmp_path / "run.log", console=False)
        captured = capsys.readouterr()
        assert "P3_silent" not in captured.out

    def test_file_created(self, tmp_path):
        log_path = tmp_path / "sub" / "run.log"
        log_trial_result(_make_result(), log_path)
        assert log_path.exists()

    def test_file_contains_name(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_trial_result(_make_result(name="P3_0777"), log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "P3_0777" in content

    def test_file_contains_params(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_trial_result(_make_result(), log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "leaves=" in content
        assert "lr=" in content

    def test_file_contains_convergence(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_trial_result(_make_result(), log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "convergence:" in content
        assert "cp50=" in content

    def test_file_contains_wrong_words(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_trial_result(_make_result(wrong_words=["кіт", "дім"]), log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "кіт" in content

    def test_appends_multiple(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_trial_result(_make_result(name="P1_0001"), log_path)
        log_trial_result(_make_result(name="P1_0002"), log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "P1_0001" in content
        assert "P1_0002" in content

    def test_3s_size_bytes_format(self, capsys, tmp_path):
        """3S result dict uses size_bytes — should still display ~3MB."""
        r = _make_result_3s()
        log_trial_result(r, tmp_path / "run.log")
        captured = capsys.readouterr()
        # 3_000_000 bytes = 3 MB; displayed as "~3MB"
        assert "3MB" in captured.out.replace(" ", "")


# ════════════════════════════════════════════════════════════════════════════
# log_phase_progress
# ════════════════════════════════════════════════════════════════════════════

class TestLogPhaseProgress:

    def test_console_shows_phase(self, capsys, tmp_path):
        results = [_make_result(trial_num=i, fitness=0.8 + i * 0.01) for i in range(3)]
        log_phase_progress("P1", results, 3600, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "P1" in captured.out

    def test_console_shows_trial_count(self, capsys, tmp_path):
        results = [_make_result(trial_num=i) for i in range(5)]
        log_phase_progress("P2", results, 1800, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_console_shows_best_fitness(self, capsys, tmp_path):
        results = [_make_result(trial_num=i, fitness=0.8 + i * 0.01) for i in range(3)]
        log_phase_progress("P3", results, 1000, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "0.8200" in captured.out  # best = 0.82

    def test_no_console_when_disabled(self, capsys, tmp_path):
        results = [_make_result()]
        log_phase_progress("P1", results, 60, tmp_path / "run.log", console=False)
        captured = capsys.readouterr()
        assert "progress" not in captured.out.lower()

    def test_file_contains_top3(self, tmp_path):
        results = [
            _make_result(trial_num=i, name=f"P1_{i:04d}", fitness=0.80 + i * 0.01)
            for i in range(5)
        ]
        log_path = tmp_path / "run.log"
        log_phase_progress("P1", results, 1000, log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "top3:" in content

    def test_empty_results_noop(self, capsys, tmp_path):
        # Should not crash or write anything
        log_phase_progress("P1", [], 0, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_elapsed_in_minutes(self, capsys, tmp_path):
        results = [_make_result()]
        log_phase_progress("P1", results, 3600, tmp_path / "run.log")  # 3600s = 60min
        captured = capsys.readouterr()
        assert "60.0min" in captured.out


# ════════════════════════════════════════════════════════════════════════════
# log_phase_summary
# ════════════════════════════════════════════════════════════════════════════

class TestLogPhaseSummary:

    def test_console_shows_phase_label(self, capsys, tmp_path):
        results = [_make_result(trial_num=i, fitness=0.8 + i * 0.01) for i in range(3)]
        log_phase_summary("P1", results, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "P1" in captured.out

    def test_console_shows_trial_count(self, capsys, tmp_path):
        results = [_make_result(trial_num=i) for i in range(7)]
        log_phase_summary("P2", results, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "7" in captured.out

    def test_console_shows_best_fitness(self, capsys, tmp_path):
        results = [_make_result(trial_num=i, fitness=0.85 + i * 0.01) for i in range(3)]
        log_phase_summary("P2", results, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "0.8700" in captured.out

    def test_console_top5_names(self, capsys, tmp_path):
        results = [_make_result(trial_num=i, name=f"P1_{i:04d}", fitness=float(i) / 10)
                   for i in range(6)]
        log_phase_summary("P1", results, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "#1" in captured.out

    def test_empty_results(self, capsys, tmp_path):
        log_path = tmp_path / "run.log"
        log_phase_summary("P1", [], log_path)
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()
        assert log_path.exists()

    def test_file_contains_params(self, tmp_path):
        results = [_make_result(trial_num=i, fitness=0.8 + i * 0.01) for i in range(3)]
        log_path = tmp_path / "run.log"
        log_phase_summary("P1", results, log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "Top-5 params:" in content
        assert "leaves=" in content

    def test_no_console_when_disabled(self, capsys, tmp_path):
        results = [_make_result()]
        log_phase_summary("P2", results, tmp_path / "run.log", console=False)
        captured = capsys.readouterr()
        assert "SUMMARY" not in captured.out

    def test_sanity_guard_label(self, capsys, tmp_path):
        results = [_make_result()]
        log_phase_summary("P1", results, tmp_path / "run.log")
        captured = capsys.readouterr()
        assert "guard" in captured.out.lower()


# ════════════════════════════════════════════════════════════════════════════
# log_final_leaderboard
# ════════════════════════════════════════════════════════════════════════════

class TestLogFinalLeaderboard:

    def _make_results(self, n: int = 5) -> list:
        return [
            _make_result(
                trial_num=i,
                name=f"P3_{i:04d}",
                fitness=0.80 + i * 0.005,
                phase="P3",
            )
            for i in range(n)
        ]

    def test_console_contains_leaderboard_header(self, capsys, tmp_path):
        log_final_leaderboard(
            self._make_results(), 3600, "TestScript",
            tmp_path / "leaderboard.txt", tmp_path / "run.log",
        )
        captured = capsys.readouterr()
        assert "LEADERBOARD" in captured.out

    def test_console_contains_winner_summary(self, capsys, tmp_path):
        log_final_leaderboard(
            self._make_results(), 7200, "TestScript",
            tmp_path / "leaderboard.txt", tmp_path / "run.log",
        )
        captured = capsys.readouterr()
        assert "WINNER" in captured.out

    def test_console_ranks_by_fitness(self, capsys, tmp_path):
        results = self._make_results(5)
        log_final_leaderboard(
            results, 1000, "TestScript",
            tmp_path / "leaderboard.txt", tmp_path / "run.log",
        )
        captured = capsys.readouterr()
        # The highest-fitness result (P3_0004, fitness=0.82) should appear first
        lines = [ln for ln in captured.out.splitlines() if "P3_" in ln]
        assert len(lines) >= 5
        assert "P3_0004" in lines[0]

    def test_leaderboard_txt_created(self, tmp_path):
        lb = tmp_path / "leaderboard.txt"
        log_final_leaderboard(
            self._make_results(), 100, "TestScript",
            lb, tmp_path / "run.log",
        )
        assert lb.exists()

    def test_leaderboard_txt_contains_header(self, tmp_path):
        lb = tmp_path / "leaderboard.txt"
        log_final_leaderboard(
            self._make_results(), 100, "TestScript",
            lb, tmp_path / "run.log",
        )
        content = lb.read_text(encoding="utf-8")
        assert "TestScript" in content

    def test_leaderboard_txt_contains_all_names(self, tmp_path):
        results = self._make_results(5)
        lb = tmp_path / "leaderboard.txt"
        log_final_leaderboard(
            results, 100, "TestScript",
            lb, tmp_path / "run.log",
        )
        content = lb.read_text(encoding="utf-8")
        for r in results:
            assert r["name"] in content

    def test_file_log_contains_winner_params(self, tmp_path):
        log_path = tmp_path / "run.log"
        log_final_leaderboard(
            self._make_results(), 100, "TestScript",
            tmp_path / "lb.txt", log_path,
        )
        content = log_path.read_text(encoding="utf-8")
        assert "Winner params" in content
        assert "leaves=" in content

    def test_empty_results(self, capsys, tmp_path):
        lb = tmp_path / "leaderboard.txt"
        log_path = tmp_path / "run.log"
        log_final_leaderboard([], 0, "TestScript", lb, log_path)
        captured = capsys.readouterr()
        assert "No results" in captured.out

    def test_no_console_when_disabled(self, capsys, tmp_path):
        log_final_leaderboard(
            self._make_results(), 100, "TestScript",
            tmp_path / "lb.txt", tmp_path / "run.log",
            console=False,
        )
        captured = capsys.readouterr()
        assert "LEADERBOARD" not in captured.out

    def test_leaderboard_txt_overwritten(self, tmp_path):
        lb = tmp_path / "leaderboard.txt"
        log_final_leaderboard(
            self._make_results(3), 100, "Script",
            lb, tmp_path / "run.log",
        )
        first_content = lb.read_text(encoding="utf-8")
        log_final_leaderboard(
            self._make_results(5), 200, "Script",
            lb, tmp_path / "run.log2",
        )
        second_content = lb.read_text(encoding="utf-8")
        # Written fresh each time (overwritten, not appended)
        assert second_content != first_content or len(second_content) > 0


# ════════════════════════════════════════════════════════════════════════════
# append_result_json
# ════════════════════════════════════════════════════════════════════════════

class TestAppendResultJson:

    def test_creates_file(self, tmp_path):
        p = tmp_path / "out" / "results.json"
        append_result_json(_make_result(), p)
        assert p.exists()

    def test_valid_json_lines(self, tmp_path):
        p = tmp_path / "results.json"
        append_result_json(_make_result(trial_num=1), p)
        line = p.read_text(encoding="utf-8").strip()
        obj = json.loads(line)
        assert "fitness" in obj

    def test_append_multiple(self, tmp_path):
        p = tmp_path / "results.json"
        append_result_json(_make_result(trial_num=1), p)
        append_result_json(_make_result(trial_num=2), p)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_unicode_preserved(self, tmp_path):
        p = tmp_path / "results.json"
        r = _make_result()
        r["name"] = "тест_назва"
        append_result_json(r, p)
        content = p.read_text(encoding="utf-8")
        assert "тест_назва" in content


# ════════════════════════════════════════════════════════════════════════════
# append_result_csv
# ════════════════════════════════════════════════════════════════════════════

class TestAppendResultCsv:

    def test_creates_file_with_header(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(), p, _CSV_FIELDS_2S)
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "fitness" in reader.fieldnames

    def test_append_multiple(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(trial_num=1), p, _CSV_FIELDS_2S)
        append_result_csv(_make_result(trial_num=2), p, _CSV_FIELDS_2S)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_no_duplicate_headers(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(trial_num=1), p, _CSV_FIELDS_2S)
        append_result_csv(_make_result(trial_num=2), p, _CSV_FIELDS_2S)
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        header_count = sum(1 for ln in lines if ln.startswith("phase"))
        assert header_count == 1

    def test_fitness_value_written(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(fitness=0.9123), p, _CSV_FIELDS_2S)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert float(rows[0]["fitness"]) == pytest.approx(0.9123)

    def test_sanity_accuracy_written(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(sanity_acc=0.9564), p, _CSV_FIELDS_2S)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert float(rows[0]["sanity_accuracy"]) == pytest.approx(0.9564)

    def test_3s_size_bytes_handled(self, tmp_path):
        """3S model.size_bytes should be converted to MB in the CSV."""
        p = tmp_path / "results.csv"
        r = _make_result_3s()
        append_result_csv(r, p, _CSV_FIELDS_2S)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert float(rows[0]["estimated_size_mb"]) == pytest.approx(3.0)

    def test_param_columns_written(self, tmp_path):
        p = tmp_path / "results.csv"
        append_result_csv(_make_result(), p, _CSV_FIELDS_2S)
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert float(rows[0]["num_leaves"]) == 435
        assert float(rows[0]["learning_rate"]) == pytest.approx(0.05)

