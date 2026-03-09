"""Logging service — dual-output training log (console + file) and persistence.

Public API
----------
log_trial_result(result, log_file, *, console=True)
    Console: informative 2-3 line block per trial with visual indicators.
    File:    full block — all metrics, params, wrong words.

log_phase_progress(phase, results, elapsed_sec, log_file, *, console=True)
    Console: compact progress bar with best snapshot.
    File:    same + top-3 listing.

log_phase_summary(phase, results, log_file, *, console=True)
    Console: bordered summary box + top-5 table.
    File:    top-5 with full params.

log_final_leaderboard(all_results, wall_elapsed, script_name,
                      leaderboard_file, log_file, *, console=True)
    Console: formatted table + winner summary.
    File:    same table + winner params.
    Also writes ``leaderboard_file`` (.txt) for post-run review.

append_result_json(result, results_json)  — persist raw result dict (JSON-lines)
append_result_csv(result, results_csv, csv_fields)  — persist to CSV

Helper (used internally and in training scripts)
-------------------------------------------------
sp(msg)  — safe print: falls back to ASCII on cp1251 terminals.

Design notes
------------
* All functions write to a ``run.log`` file (``log_file`` param) so every
  run produces a full technical report that can be summarised later.
* Console output is designed for readability during long runs:
  - Visual fitness bars show quality at a glance
  - Color-coded indicators (✓/✗/▲/★) highlight key events
  - Per-trial blocks are clearly separated
  - Progress lines show trend direction
* Both 2S (binary), 3S (multiclass), and universal (11-class) result dicts
  are supported.  The 3S/univ dict stores ``model.size_bytes`` instead of
  ``model.estimated_size_mb``; ``_model_size_mb`` handles both.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def sp(msg: str = "", **kwargs) -> None:
    """Safe print — falls back to ASCII on cp1251 / narrow terminals."""
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), **kwargs)


def _write_log(log_file: Path, text: str) -> None:
    """Append ``text`` to ``log_file``, creating parent dirs as needed."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(text)


def _model_size_mb(result: dict) -> float:
    """Extract model size in MB — handles both 2S and 3S result formats."""
    m = result.get("model", {})
    if "estimated_size_mb" in m:
        return float(m["estimated_size_mb"])
    if "size_bytes" in m:
        return float(m["size_bytes"]) / 1_000_000
    return 0.0


def _ts() -> str:
    """Current timestamp string for file log headers."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fitness_bar(fitness: float, width: int = 20) -> str:
    """Visual bar showing fitness level: [████████████--------] 0.8434"""
    filled = int(fitness * width)
    filled = max(0, min(width, filled))
    bar = chr(9608) * filled + chr(9472) * (width - filled)
    return f"[{bar}]"


def _fitness_indicator(fitness: float) -> str:
    """Return a visual indicator based on fitness quality."""
    if fitness >= 0.90:
        return "(*)"    # star: exceptional
    if fitness >= 0.85:
        return "(^)"    # up arrow: great
    if fitness >= 0.80:
        return "(+)"    # good
    if fitness >= 0.70:
        return "(~)"    # ok
    return "(.)"        # needs work


def _trend_arrow(current: float, previous: float) -> str:
    """Return trend indicator compared to previous best."""
    if current > previous + 0.001:
        return " NEW BEST ^"
    if current > previous - 0.005:
        return ""
    return ""


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


# ════════════════════════════════════════════════════════════════════════════
# log_trial_result
# ════════════════════════════════════════════════════════════════════════════

def log_trial_result(
    result: dict,
    log_file: Path,
    *,
    console: bool = True,
) -> None:
    """Log one trial result — informative block to console, full detail to file.

    Console output (example)::

        -- P3_0547 -------- fit=0.8434 (+) [================----] -----------
           F1=75.19%  acc=84.10%  AUC=89.4%  |  hand=40/44  sanity=95.64%
           ~3MB  iter=435  9s  |  leaves=435 lr=0.050 depth=11
           WRONG: shapada, oshuk, chiksa

    File block (example)::

        [2026-03-06 08:23:41] P3_0547
          fit=0.8434  F1=75.19%  acc=84.10%  AUC=89.4%  hand=40/44
          sanity=95.64% (guard)  ~3MB  iter=435  9s
          params: leaves=435 lr=0.05000 mc=13 depth=11
                  l1=0.12 l2=1.80 sub=0.82 col=0.71
          convergence: cp50=69.6% cp150=72.5% cp300=73.1%
          WRONG: shapada, oshuk, chiksa
    """
    iv   = result.get("internal", {})
    sv   = result.get("external", {})
    hv   = result.get("handcrafted", {})
    p    = result.get("params", {})
    name = result.get("name", "?")

    fit      = result.get("fitness", 0.0)
    f1_pct   = iv.get("f1", iv.get("f1_macro", iv.get("f1_weighted", 0.0))) * 100
    acc_pct  = iv.get("accuracy", 0.0) * 100
    auc_pct  = iv.get("auc", 0.0) * 100
    hand_c   = hv.get("correct", 0)
    hand_t   = hv.get("total", 0)
    sanity   = sv.get("accuracy", 0.0) * 100
    size_mb  = _model_size_mb(result)
    best_it  = iv.get("best_iteration", 0)
    t_sec    = result.get("train_time_sec", 0.0)
    penalty  = " [HAND PENALTY!]" if result.get("hand_penalty_applied") else ""
    sanity_warn = " !! SANITY VIOLATED!" if result.get("sanity_violated") else ""

    wrong = [x["word"] for x in hv.get("results", []) if not x.get("correct")]

    # ── Console output (informative multi-line block) ────────────────────────
    if console:
        indicator = _fitness_indicator(fit)
        bar = _fitness_bar(fit)
        hand_pct = (hand_c / hand_t * 100) if hand_t > 0 else 0
        time_str = _format_time(t_sec)

        # Line 1: trial header with fitness bar
        sp(f"   -- {name} -- fit={fit:.4f} {indicator} {bar} --")

        # Line 2: key metrics in clear groups
        sp(f"      F1={f1_pct:5.2f}%  acc={acc_pct:5.2f}%  AUC={auc_pct:4.1f}%  "
           f"|  hand={hand_c}/{hand_t} ({hand_pct:.0f}%)  "
           f"sanity={sanity:.2f}%{penalty}{sanity_warn}")

        # Line 3: model info + key params
        sp(f"      ~{size_mb:.0f}MB  iter={best_it}  {time_str}  "
           f"|  leaves={p.get('num_leaves', '?')} "
           f"lr={p.get('learning_rate', 0):.3f} "
           f"depth={p.get('max_depth', '?')}")

        # Line 4: wrong words (if any)
        if wrong:
            wrong_str = ", ".join(wrong[:8])
            extra = f" +{len(wrong)-8} more" if len(wrong) > 8 else ""
            sp(f"      WRONG: {wrong_str}{extra}")

        sys.stdout.flush()

    # ── File output (full detail) ────────────────────────────────────────────
    lines = [
        f"\n[{_ts()}] {name}\n",
        f"  fit={fit:.4f}  F1={f1_pct:.2f}%  acc={acc_pct:.2f}%  "
        f"AUC={auc_pct:.1f}%  hand={hand_c}/{hand_t}  "
        f"sanity={sanity:.2f}% (guard)  ~{size_mb:.0f}MB  "
        f"iter={best_it}  {t_sec:.0f}s{penalty}{sanity_warn}\n",
        f"  params: leaves={p.get('num_leaves','?')} "
        f"lr={p.get('learning_rate', 0):.5f} "
        f"mc={p.get('min_child_samples','?')} "
        f"depth={p.get('max_depth','?')} "
        f"l1={p.get('lambda_l1',0):.2f} "
        f"l2={p.get('lambda_l2',0):.2f} "
        f"sub={p.get('subsample','?')} "
        f"col={p.get('colsample_bytree',0):.2f}\n",
    ]

    # Convergence curve (compact: every checkpoint as cp{N}=F1%)
    curve = result.get("convergence_curve", [])
    if curve:
        curve_parts = [f"cp{pt['cp']}={pt['f1']*100:.1f}%" for pt in curve]
        lines.append(f"  convergence: {' '.join(curve_parts)}\n")

    if wrong:
        lines.append(f"  WRONG: {', '.join(wrong)}\n")

    _write_log(log_file, "".join(lines))


# ════════════════════════════════════════════════════════════════════════════
# log_phase_progress
# ════════════════════════════════════════════════════════════════════════════

def log_phase_progress(
    phase: str,
    results: list,
    elapsed_sec: float,
    log_file: Path,
    *,
    console: bool = True,
    phase_budget_sec: float = 0.0,
) -> None:
    """Log a progress snapshot after each trial.

    Console (3 lines): phase stats + best/mean fitness + trend + ETA.
    File: same + top-3 names with fitness.

    Args:
        phase_budget_sec: Total seconds allocated for this phase.  When > 0
            an ETA line is appended showing time remaining in the phase.
    """
    if not results:
        return
    _best = max(results, key=lambda r: r["fitness"])
    n       = len(results)
    elapsed = elapsed_sec / 60
    iv_b    = _best.get("internal", {})
    hv_b    = _best.get("handcrafted", {})
    sv_b    = _best.get("external", {})

    f1_pct  = iv_b.get("f1", iv_b.get("f1_macro", 0.0)) * 100
    hand_c  = hv_b.get("correct", 0)
    hand_t  = hv_b.get("total", 0)
    sanity  = sv_b.get("accuracy", 0.0) * 100
    fit     = _best.get("fitness", 0.0)

    # Calculate mean fitness and trend
    all_fits = [r.get("fitness", 0.0) for r in results]
    mean_fit = sum(all_fits) / len(all_fits) if all_fits else 0
    last5_fit = sum(all_fits[-5:]) / len(all_fits[-5:]) if len(all_fits) >= 5 else mean_fit

    # Calculate rate (trials per hour)
    rate = n / (elapsed / 60) if elapsed > 0 else 0

    # Trend: is the latest result the new best?
    trend = ""
    if n >= 2:
        prev_best = max(all_fits[:-1])
        if all_fits[-1] > prev_best + 0.001:
            trend = " NEW BEST ^"
        elif last5_fit > mean_fit + 0.002:
            trend = " trending up"

    line1 = (f"   {phase} [{n:>4} trials] {elapsed:.1f}min  "
             f"~{rate:.0f}t/h  |  "
             f"best={fit:.4f}  mean={mean_fit:.4f}  last5={last5_fit:.4f}"
             f"{trend}")
    line2 = (f"              "
             f"F1={f1_pct:.2f}%  hand={hand_c}/{hand_t}  "
             f"sanity={sanity:.2f}%")

    # ETA line: only when phase_budget_sec is known and there is time remaining
    line_eta = ""
    if phase_budget_sec > 0 and elapsed_sec < phase_budget_sec:
        remaining_sec = phase_budget_sec - elapsed_sec
        remaining_min = remaining_sec / 60
        # Estimate remaining trials based on average time per trial so far
        avg_sec_per_trial = elapsed_sec / n if n > 0 else 0
        est_trials_left = int(remaining_sec / avg_sec_per_trial) if avg_sec_per_trial > 0 else 0
        if remaining_min >= 60:
            eta_str = f"{remaining_min / 60:.1f}h"
        else:
            eta_str = f"{remaining_min:.0f}min"
        line_eta = (f"              "
                    f"ETA: ~{eta_str} remaining in {phase}  "
                    f"(~{est_trials_left} more trials @ {avg_sec_per_trial/60:.1f}min/trial)")

    if console:
        sp(line1)
        sp(line2)
        if line_eta:
            sp(line_eta)
        sys.stdout.flush()

    top3 = sorted(results, key=lambda r: r["fitness"], reverse=True)[:3]
    top3_str = "  ".join(
        f"{r['name']}={r['fitness']:.4f}" for r in top3
    )
    log_body = f"{line1}\n{line2}\n"
    if line_eta:
        log_body += f"{line_eta}\n"
    log_body += f"  top3: {top3_str}\n"
    _write_log(log_file, log_body)


# ════════════════════════════════════════════════════════════════════════════
# log_phase_summary
# ════════════════════════════════════════════════════════════════════════════

def log_phase_summary(
    phase: str,
    results: list,
    log_file: Path,
    *,
    console: bool = True,
) -> None:
    """Log a summary after a phase completes.

    Console: bordered box with key stats + top-5 table with aligned columns.
    File: same + params for each top-5 entry.
    """
    if not results:
        msg = f"\n{phase}: no results\n"
        if console:
            sp(msg.strip())
        _write_log(log_file, msg)
        return

    by_fit   = sorted(results, key=lambda r: r["fitness"], reverse=True)
    best     = by_fit[0]
    iv_b     = best.get("internal", {})
    hv_b     = best.get("handcrafted", {})
    sv_b     = best.get("external", {})
    ext_accs = [r["external"]["accuracy"] for r in results]
    all_fits = [r.get("fitness", 0.0) for r in results]

    f1_pct   = iv_b.get("f1", iv_b.get("f1_macro", 0.0)) * 100
    hand_c   = hv_b.get("correct", 0)
    hand_t   = hv_b.get("total", 0)
    sanity   = sv_b.get("accuracy", 0.0) * 100
    total_t  = sum(r.get("train_time_sec", 0) for r in results) / 60
    mean_fit = sum(all_fits) / len(all_fits)

    W = 80
    sep = "=" * W
    thin = "-" * W

    console_lines = [
        f"\n{sep}",
        f"  PHASE {phase} COMPLETE   |   {len(results)} trials in {total_t:.0f}min",
        f"{sep}",
        f"  Fitness:  best={best['fitness']:.4f}  mean={mean_fit:.4f}  "
        f"median={sorted(all_fits)[len(all_fits)//2]:.4f}",
        f"  Metrics:  F1={f1_pct:.2f}%  hand={hand_c}/{hand_t}  "
        f"sanity={sanity:.2f}%",
        f"  Sanity:   max={max(ext_accs)*100:.2f}%  "
        f"mean={sum(ext_accs)/len(ext_accs)*100:.2f}%  (guard only)",
        f"{thin}",
        f"  {'#':>2}  {'Name':<12}  {'Fitness':>7}  {'F1%':>6}  "
        f"{'Hand':>7}  {'Sanity%':>7}  {'~MB':>4}  {'Time':>5}",
        f"  {'--':>2}  {'-'*12}  {'-'*7}  {'-'*6}  "
        f"{'-'*7}  {'-'*7}  {'-'*4}  {'-'*5}",
    ]
    for i, r in enumerate(by_fit[:5], 1):
        iv = r.get("internal", {})
        sv = r.get("external", {})
        hv = r.get("handcrafted", {})
        pen = " *" if r.get("hand_penalty_applied") else ""
        f1_r = iv.get("f1", iv.get("f1_macro", 0.0)) * 100
        t = _format_time(r.get("train_time_sec", 0))
        console_lines.append(
            f"  #{i:<2} {r['name']:<12}  {r['fitness']:>7.4f}  "
            f"{f1_r:>5.2f}  {hv.get('correct',0):>2}/{hv.get('total',0)}{pen:<3}  "
            f"{sv.get('accuracy',0)*100:>6.2f}  "
            f"{_model_size_mb(r):>4.0f}  {t:>5}"
        )
    console_lines.append(f"{sep}")

    if console:
        for ln in console_lines:
            sp(ln)
        sys.stdout.flush()

    # File: same + params for each top-5
    file_lines = list(console_lines)
    file_lines.append("\n  Top-5 params:")
    for i, r in enumerate(by_fit[:5], 1):
        p = r.get("params", {})
        file_lines.append(
            f"    #{i} {r['name']}: "
            f"leaves={p.get('num_leaves','?')} "
            f"lr={p.get('learning_rate',0):.5f} "
            f"mc={p.get('min_child_samples','?')} "
            f"depth={p.get('max_depth','?')} "
            f"l1={p.get('lambda_l1',0):.2f} "
            f"l2={p.get('lambda_l2',0):.2f}"
        )

    _write_log(log_file, "\n".join(file_lines) + "\n")


# ════════════════════════════════════════════════════════════════════════════
# log_final_leaderboard
# ════════════════════════════════════════════════════════════════════════════

def log_final_leaderboard(
    all_results: list,
    wall_elapsed: float,
    script_name: str,
    leaderboard_file: Path,
    log_file: Path,
    *,
    console: bool = True,
) -> None:
    """Print/write the final leaderboard and persist ``leaderboard_file``.

    Console: top-30 table + winner summary box.
    File (``log_file``): same table + winner params.
    ``leaderboard_file``: plain-text table written fresh (overwrites previous).
    """
    if not all_results:
        msg = "No results to show.\n"
        if console:
            sp(msg.strip())
        _write_log(log_file, msg)
        return

    by_fit = sorted(all_results, key=lambda r: r["fitness"], reverse=True)

    # ── Header ───────────────────────────────────────────────────────────────
    W = 100
    header = [
        f"\n{'=' * W}",
        f"{script_name} — FINAL LEADERBOARD",
        f"{'=' * W}",
        (f"{'#':>3}  {'Name':<14}  {'Ph':>2}  {'Fit':>7}  {'F1%':>6}  "
         f"{'Acc%':>6}  {'Hand':>6}  {'AUC%':>5}  {'Sanity%':>7}  "
         f"{'~MB':>5}  {'Iter':>5}  {'Sec':>5}  {'@min':>5}"),
        f"{'─' * W}",
    ]

    # ── Rows ─────────────────────────────────────────────────────────────────
    rows = []
    for i, r in enumerate(by_fit[:30], 1):
        iv = r.get("internal", {})
        sv = r.get("external", {})
        hv = r.get("handcrafted", {})
        ph = r.get("phase", "??")
        pn = " *" if r.get("hand_penalty_applied") else "  "
        f1_r = iv.get("f1", iv.get("f1_macro", 0.0)) * 100
        rows.append(
            f"{i:>3}  {r['name']:<14}  {ph:>2}  "
            f"{r['fitness']:>7.4f}  "
            f"{f1_r:>5.2f}  {iv.get('accuracy',0)*100:>5.2f}  "
            f"{hv.get('correct',0):>2}/{hv.get('total',0)}{pn}  "
            f"{iv.get('auc',0)*100:>5.1f}  {sv.get('accuracy',0)*100:>7.2f}  "
            f"{_model_size_mb(r):>5.0f}  "
            f"{iv.get('best_iteration',0):>5}  "
            f"{r.get('train_time_sec',0):>5.0f}  "
            f"{r.get('wall_elapsed_min',0):>5.1f}"
        )

    # ── Winner summary ────────────────────────────────────────────────────────
    best        = by_fit[0]
    best_sv     = best.get("external", {})
    best_iv     = best.get("internal", {})
    best_hv     = best.get("handcrafted", {})
    best_p      = best.get("params", {})
    best_sanity = best_sv.get("accuracy", 0.0)
    best_size   = _model_size_mb(best)
    best_f1     = best_iv.get("f1", best_iv.get("f1_macro", 0.0)) * 100

    winner = [
        f"{'─' * W}",
        f"\n  WINNER SUMMARY:",
        f"    fitness:     {best['fitness']:.4f}",
        f"    F1:          {best_f1:.2f}%  "
        f"acc={best_iv.get('accuracy',0)*100:.2f}%  "
        f"hand={best_hv.get('correct',0)}/{best_hv.get('total',0)}",
        f"    sanity_acc:  {best_sanity*100:.2f}%  (guard)  "
        f"{'OK >96%' if best_sanity >= 0.96 else 'below 96% threshold'}",
        f"    model size:  ~{best_size:.0f}MB  "
        f"{'OK <20MB' if best_size <= 20 else 'LARGE'}",
        f"  Wall time: {wall_elapsed/3600:.1f}h  ({wall_elapsed/60:.0f} min)",
    ]

    # ── Console output ────────────────────────────────────────────────────────
    if console:
        for ln in header:
            sp(ln)
        for ln in rows:
            sp(ln)
        for ln in winner:
            sp(ln)
        sys.stdout.flush()

    # ── File output ───────────────────────────────────────────────────────────
    winner_params = [
        f"\n  Winner params ({best['name']}):",
        f"    leaves={best_p.get('num_leaves','?')} "
        f"lr={best_p.get('learning_rate',0):.5f} "
        f"mc={best_p.get('min_child_samples','?')} "
        f"depth={best_p.get('max_depth','?')} "
        f"l1={best_p.get('lambda_l1',0):.2f} "
        f"l2={best_p.get('lambda_l2',0):.2f} "
        f"sub={best_p.get('subsample','?')} "
        f"col={best_p.get('colsample_bytree',0):.2f}",
    ]
    file_body = (
        "\n".join(header) + "\n"
        + "\n".join(rows) + "\n"
        + "\n".join(winner) + "\n"
        + "\n".join(winner_params) + "\n"
    )
    _write_log(log_file, file_body)

    # ── leaderboard.txt ───────────────────────────────────────────────────────
    leaderboard_file.parent.mkdir(parents=True, exist_ok=True)
    with open(leaderboard_file, "w", encoding="utf-8") as fh:
        fh.write(f"{script_name} — FINAL LEADERBOARD\n")
        fh.write(f"Generated: {_ts()}\n")
        fh.write(f"Total trials: {len(all_results)}  "
                 f"Wall time: {wall_elapsed/60:.0f}min\n")
        if all_results:
            sample_sz = all_results[0].get("external", {}).get("sample_size", 0)
            fh.write(f"Sanity sample: {sample_sz} words\n")
        fh.write("\n")
        fh.write(f"{'#':>3}  {'Name':<14}  {'Ph':>2}  {'Fit':>7}  "
                 f"{'F1%':>6}  {'Hand':>6}  {'Sanity%':>7}  {'~MB':>5}  Params\n")
        fh.write(f"{'─' * 90}\n")
        for i, r in enumerate(by_fit[:30], 1):
            sv = r.get("external", {})
            hv = r.get("handcrafted", {})
            iv = r.get("internal", {})
            ph = r.get("phase", "??")
            p  = r.get("params", {})
            f1_r = iv.get("f1", iv.get("f1_macro", 0.0)) * 100
            fh.write(
                f"{i:>3}  {r['name']:<14}  {ph:>2}  "
                f"{r['fitness']:>7.4f}  "
                f"{f1_r:>5.2f}  {hv.get('correct',0):>2}/{hv.get('total',0)}  "
                f"{sv.get('accuracy',0)*100:>6.2f}  {_model_size_mb(r):>5.0f}  "
                f"leaves={p.get('num_leaves')}  "
                f"lr={p.get('learning_rate',0):.5f}  "
                f"mc={p.get('min_child_samples')}  "
                f"depth={p.get('max_depth')}\n"
            )


# ════════════════════════════════════════════════════════════════════════════
# Persistent result storage  (JSON-lines + CSV)
# ════════════════════════════════════════════════════════════════════════════

def append_result_json(result: dict, results_json: Path) -> None:
    """Append one trial result as a single JSON line."""
    results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


def append_result_csv(result: dict, results_csv: Path, csv_fields: list) -> None:
    """Append one trial result to the CSV log file.

    Parameters
    ----------
    result:
        Full Luscinia result dict (2S or 3S format).
    results_csv:
        Path to the CSV file (created with header if absent).
    csv_fields:
        Ordered list of column names — should be the ``CSV_FIELDS`` constant
        defined in the calling training script so each script controls its
        own schema.
    """
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    iv  = result.get("internal", {})
    sv  = result.get("external", {})
    hv  = result.get("handcrafted", {})
    p   = result.get("params", {})

    row = {
        "phase":               result.get("phase", ""),
        "trial_number":        result.get("trial_number", -1),
        "name":                result.get("name", ""),
        "fitness":             result.get("fitness", 0.0),
        "hand_penalty_applied": result.get("hand_penalty_applied", False),
        # internal metrics — both 2S ("f1") and 3S ("f1_macro" / "f1") keys
        "f1":                  iv.get("f1", iv.get("f1_macro", 0.0)),
        "f1_macro":            iv.get("f1", iv.get("f1_macro", 0.0)),
        "accuracy":            iv.get("accuracy", 0.0),
        "auc":                 iv.get("auc", 0.0),
        "best_iteration":      iv.get("best_iteration", 0),
        "num_trees":           iv.get("num_trees", 0),
        # sanity (external) sample
        "sanity_accuracy":     sv.get("accuracy", 0.0),
        "sanity_correct":      sv.get("correct", 0),
        "sanity_sample_size":  sv.get("sample_size", 0),
        # handcrafted
        "hand_accuracy":       hv.get("accuracy", 0.0),
        "hand_correct":        hv.get("correct", 0),
        "hand_total":          hv.get("total", 0),
        # model
        "estimated_size_mb":   _model_size_mb(result),
        # timing
        "train_time_sec":      result.get("train_time_sec", 0.0),
        "wall_elapsed_sec":    result.get("wall_elapsed_sec", 0.0),
        "wall_elapsed_min":    result.get("wall_elapsed_min", 0.0),
        "boosting_rounds_at_convergence": iv.get("best_iteration", 0),
    }
    # hyperparams
    for k in csv_fields:
        if k not in row:
            row[k] = p.get(k, "")

    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ════════════════════════════════════════════════════════════════════════════
# log_training_summary  — post-run ML lab-style analysis
# ════════════════════════════════════════════════════════════════════════════

def _detect_plateau(fits: list, window: int = 10, threshold: float = 0.002) -> dict:
    """Detect fitness plateau in the last ``window`` trials.

    Returns a dict with:
        plateau: bool  — True if last ``window`` trials improved < threshold
        window_best: float
        window_worst: float
        delta: float
        since_trial: int  — how many trials ago the last meaningful improvement was
    """
    if len(fits) < window:
        return {"plateau": False, "window_best": 0.0, "window_worst": 0.0,
                "delta": 0.0, "since_trial": 0}
    tail = fits[-window:]
    delta = max(tail) - min(tail)
    # find last trial where fitness improved by > threshold
    since = 0
    for i in range(len(fits) - 1, 0, -1):
        if fits[i] > fits[i - 1] + threshold:
            break
        since += 1
    return {
        "plateau": delta < threshold,
        "window_best": max(tail),
        "window_worst": min(tail),
        "delta": delta,
        "since_trial": since,
    }


def _phase_stats(results: list) -> dict:
    """Compute per-phase statistics from a flat results list."""
    from collections import defaultdict
    by_phase: dict = defaultdict(list)
    for r in results:
        by_phase[r.get("phase", "?")].append(r)
    stats = {}
    for ph, rs in sorted(by_phase.items()):
        fits = [r.get("fitness", 0.0) for r in rs]
        times = [r.get("train_time_sec", 0.0) for r in rs]
        pruned = sum(1 for r in rs if r.get("pruned", False))
        stats[ph] = {
            "n": len(rs),
            "best": max(fits),
            "mean": sum(fits) / len(fits),
            "worst": min(fits),
            "std": (sum((f - sum(fits)/len(fits))**2 for f in fits) / len(fits)) ** 0.5,
            "total_time_min": sum(times) / 60,
            "avg_time_min": sum(times) / len(times) / 60,
            "pruned": pruned,
            "pruned_pct": pruned / len(rs) * 100,
        }
    return stats


def _hyperparameter_insights(results: list, top_n: int = 10) -> dict:
    """Compute mean param values for top-N vs bottom-N trials.

    Returns dict of param → {"top_mean", "bottom_mean", "delta", "direction"}.
    Direction: "higher" means larger value → better fitness.
    """
    if len(results) < top_n * 2:
        return {}
    by_fit = sorted(results, key=lambda r: r.get("fitness", 0.0), reverse=True)
    top = [r.get("params", {}) for r in by_fit[:top_n]]
    bot = [r.get("params", {}) for r in by_fit[-top_n:]]
    keys = ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
            "lambda_l1", "lambda_l2", "subsample", "colsample_bytree"]
    insights = {}
    for k in keys:
        top_vals = [p[k] for p in top if k in p]
        bot_vals = [p[k] for p in bot if k in p]
        if not top_vals or not bot_vals:
            continue
        top_mean = sum(top_vals) / len(top_vals)
        bot_mean = sum(bot_vals) / len(bot_vals)
        delta = top_mean - bot_mean
        insights[k] = {
            "top_mean": round(top_mean, 5),
            "bottom_mean": round(bot_mean, 5),
            "delta": round(delta, 5),
            "direction": "higher" if delta > 0 else "lower",
        }
    return insights


def log_training_summary(
    all_results: list,
    wall_elapsed_sec: float,
    script_name: str,
    log_file: Path,
    *,
    console: bool = True,
) -> None:
    """Write a comprehensive ML lab-style training summary.

    Includes:
    - Overall statistics (total trials, wall time, best fitness)
    - Per-phase breakdown (n, best, mean, std, time, pruning rate)
    - Fitness progression (early/mid/late thirds comparison)
    - Plateau detection (last 10 trials)
    - Hyperparameter insights (top-10 vs bottom-10 param means)
    - Convergence quality (avg best_iteration vs max_rounds)
    - Recommendations based on observations

    Compatible with 2S (binary), 3S (3-class), and universal (11-class) result
    dicts — uses the same field extraction as the rest of logging_service.
    """
    if not all_results:
        msg = "\n[TRAINING SUMMARY] No results to summarize.\n"
        if console:
            sp(msg.strip())
        _write_log(log_file, msg)
        return

    W = 80
    sep = "=" * W
    thin = "-" * W

    by_fit    = sorted(all_results, key=lambda r: r.get("fitness", 0.0), reverse=True)
    all_fits  = [r.get("fitness", 0.0) for r in all_results]
    best      = by_fit[0]
    best_iv   = best.get("internal", {})
    best_hv   = best.get("handcrafted", {})
    best_sv   = best.get("external", {})

    n_total   = len(all_results)
    mean_fit  = sum(all_fits) / n_total
    std_fit   = (sum((f - mean_fit) ** 2 for f in all_fits) / n_total) ** 0.5
    wall_h    = wall_elapsed_sec / 3600

    # Fitness progression: split into thirds
    third = max(1, n_total // 3)
    early_fits = [r.get("fitness", 0.0) for r in all_results[:third]]
    late_fits  = [r.get("fitness", 0.0) for r in all_results[-third:]]
    early_mean = sum(early_fits) / len(early_fits) if early_fits else 0.0
    late_mean  = sum(late_fits) / len(late_fits) if late_fits else 0.0
    progression_delta = late_mean - early_mean

    # Plateau detection
    plateau = _detect_plateau(all_fits, window=min(10, n_total))

    # Per-phase stats
    phase_stats = _phase_stats(all_results)

    # Hyperparameter insights
    insights = _hyperparameter_insights(all_results, top_n=min(10, n_total // 4))

    # Convergence: average best_iteration / max_rounds per phase
    convergence_by_phase: dict = {}
    for r in all_results:
        ph = r.get("phase", "?")
        bi = r.get("internal", {}).get("best_iteration", 0)
        if bi > 0:
            convergence_by_phase.setdefault(ph, []).append(bi)

    # Build output lines
    lines = [
        f"\n{sep}",
        f"  TRAINING SUMMARY — {script_name}",
        f"  Generated: {_ts()}",
        f"{sep}",
        "",
        "  OVERVIEW",
        f"  {thin}",
        f"  Total trials:     {n_total}",
        f"  Wall time:        {wall_h:.2f}h  ({wall_elapsed_sec/60:.0f} min)",
        f"  Throughput:       {n_total / max(wall_h, 0.01):.1f} trials/h",
        f"  Best fitness:     {best['fitness']:.4f}  ({best['name']})",
        f"  Mean fitness:     {mean_fit:.4f}  ±{std_fit:.4f}",
        f"  Best F1:          {best_iv.get('f1', best_iv.get('f1_macro', 0.0))*100:.2f}%",
        f"  Best hand acc:    {best_hv.get('correct',0)}/{best_hv.get('total',0)}",
        f"  Best sanity acc:  {best_sv.get('accuracy',0)*100:.2f}%",
        "",
        "  PER-PHASE BREAKDOWN",
        f"  {thin}",
        f"  {'Phase':<4}  {'N':>4}  {'Best':>7}  {'Mean':>7}  {'Std':>6}  "
        f"{'Avg min':>7}  {'Total h':>7}  {'Pruned':>9}",
        f"  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}  "
        f"{'─'*7}  {'─'*7}  {'─'*9}",
    ]
    for ph, s in phase_stats.items():
        lines.append(
            f"  {ph:<4}  {s['n']:>4}  {s['best']:>7.4f}  {s['mean']:>7.4f}  "
            f"{s['std']:>6.4f}  {s['avg_time_min']:>6.1f}m  "
            f"{s['total_time_min']/60:>6.2f}h  "
            f"{s['pruned']:>4}/{s['n']} ({s['pruned_pct']:>4.0f}%)"
        )

    # Fitness progression
    lines += [
        "",
        "  FITNESS PROGRESSION",
        f"  {thin}",
        f"  Early trials (first 1/{3}):  mean={early_mean:.4f}",
        f"  Late  trials (last  1/{3}):  mean={late_mean:.4f}",
        f"  Delta (late - early):       {progression_delta:+.4f}  "
        + ("▲ improving" if progression_delta > 0.005 else
           "▼ declining" if progression_delta < -0.005 else "~ flat"),
    ]

    # Plateau
    lines += [
        "",
        "  PLATEAU DETECTION  (last 10 trials)",
        f"  {thin}",
    ]
    if plateau["plateau"]:
        lines.append(
            f"  ⚠  PLATEAU detected: Δfitness={plateau['delta']:.4f} "
            f"over last {min(10, n_total)} trials  "
            f"(no improvement for {plateau['since_trial']} trials)"
        )
    else:
        lines.append(
            f"  ✓  Active search: Δfitness={plateau['delta']:.4f} "
            f"in last {min(10, n_total)} trials  "
            f"(last improvement {plateau['since_trial']} trials ago)"
        )

    # Convergence
    if convergence_by_phase:
        lines += ["", "  CONVERGENCE (avg best_iteration per phase)", f"  {thin}"]
        for ph, iters in sorted(convergence_by_phase.items()):
            avg_iter = sum(iters) / len(iters)
            lines.append(f"  {ph}: avg best_iter={avg_iter:.0f}  "
                         f"(sample={len(iters)} trials with early stop)")

    # Hyperparameter insights
    if insights:
        lines += [
            "",
            f"  HYPERPARAMETER INSIGHTS  (top-{min(10, n_total//4)} vs bottom-{min(10, n_total//4)})",
            f"  {thin}",
            f"  {'Param':<28}  {'Top mean':>10}  {'Bot mean':>10}  {'Delta':>8}  Dir",
            f"  {'─'*28}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*7}",
        ]
        for param, info in sorted(insights.items(),
                                  key=lambda x: abs(x[1]["delta"]), reverse=True):
            lines.append(
                f"  {param:<28}  {info['top_mean']:>10.4f}  "
                f"{info['bottom_mean']:>10.4f}  "
                f"{info['delta']:>+8.4f}  {info['direction']}"
            )

    # Recommendations
    recs = []
    if plateau["plateau"] and plateau["since_trial"] > 5:
        recs.append("Search has plateaued — consider narrowing param space or increasing budget.")
    if progression_delta < -0.01:
        recs.append("Late fitness is lower than early — possible overfitting to sampler noise; "
                    "try CMA-ES restart.")
    if phase_stats.get("P1", {}).get("pruned_pct", 0) < 10 and \
            phase_stats.get("P1", {}).get("n", 0) > 20:
        recs.append("P1 pruning rate <10% — consider increasing min_resource or lowering "
                    "reduction_factor for Hyperband.")
    if best_hv.get("total", 0) > 0 and \
            best_hv.get("correct", 0) / best_hv["total"] < 0.80:
        recs.append("Handcrafted accuracy <80% — model struggles with poetic vocabulary; "
                    "consider adding more handcrafted examples.")
    if best_sv.get("accuracy", 0) < 0.90:
        recs.append("Sanity accuracy <90% — model quality may be insufficient for production.")

    if recs:
        lines += ["", "  RECOMMENDATIONS", f"  {thin}"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"  {i}. {rec}")

    lines.append(f"\n{sep}\n")

    text = "\n".join(lines)
    if console:
        sp(text)
        sys.stdout.flush()
    _write_log(log_file, text)
