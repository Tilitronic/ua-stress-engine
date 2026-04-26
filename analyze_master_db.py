#!/usr/bin/env python3
"""
analyze_master_db.py — Human-readable summary of the master SQLite database.

Prints record count, schema summary, field distributions, and SHA-256 hash.
Output is deterministic (no timestamps in body).

Usage:
    python src/data_management/analyze_master_db.py
    python src/data_management/analyze_master_db.py --verbose
    python src/data_management/analyze_master_db.py --db /path/to/db.sqlite3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the master DB
# ---------------------------------------------------------------------------

# __file__ = src/data_management/analyze_master_db.py
# parents[0] = src/data_management
# parents[1] = src
# parents[2] = ua-stress-engine (project root)
# parents[3] = poetykaAnalizerEngine
_CACHE_DIRS = [
    # Canonical: within this project
    Path(__file__).parent / "transform" / "cache",
    # Sibling VersaSenseBackend project (common dev layout)
    Path(__file__).parents[3] / "VersaSenseEngine" / "VersaSenseBackend"
    / "src" / "data_management" / "transform" / "cache",
]

def find_master_db() -> Path | None:
    """Return master DB path from env var or by globbing known cache directories."""
    env = os.environ.get("UA_STRESS_MASTER_DB")
    if env:
        p = Path(env)
        return p if p.exists() else None
    # Fallback: newest MERGEDSQL_*.sqlite3 in known cache dirs
    for cache_dir in _CACHE_DIRS:
        candidates = sorted(cache_dir.glob("MERGEDSQL_*.sqlite3"))
        if candidates:
            return candidates[-1]
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def _row_count(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _stress_index_distribution(con: sqlite3.Connection, limit: int = 10) -> list[tuple]:
    """Return top-N (stress_indices_json, count) pairs from word_form."""
    return con.execute(
        """
        SELECT stress_indices_json, COUNT(*) AS cnt
        FROM   word_form
        GROUP  BY stress_indices_json
        ORDER  BY cnt DESC
        LIMIT  ?
        """,
        (limit,),
    ).fetchall()


def _pos_distribution(con: sqlite3.Connection) -> list[tuple]:
    return con.execute(
        """
        SELECT pos, COUNT(*) AS cnt
        FROM   word_form
        WHERE  pos IS NOT NULL
        GROUP  BY pos
        ORDER  BY cnt DESC
        LIMIT  10
        """
    ).fetchall()


def _null_counts(con: sqlite3.Connection) -> dict[str, int]:
    cols = ["form", "stress_indices_json", "lemma", "pos"]
    result = {}
    for col in cols:
        n = con.execute(
            f"SELECT COUNT(*) FROM word_form WHERE {col} IS NULL"
        ).fetchone()[0]
        result[col] = n
    return result


def _sample_stress_json_invalid(con: sqlite3.Connection, n: int = 10_000) -> int:
    """Count rows (sampled) where stress_indices_json is not valid JSON."""
    rows = con.execute(
        f"SELECT stress_indices_json FROM word_form ORDER BY RANDOM() LIMIT {n}"
    ).fetchall()
    invalid = 0
    for (raw,) in rows:
        try:
            v = json.loads(raw)
            if not isinstance(v, list):
                invalid += 1
        except (json.JSONDecodeError, TypeError):
            invalid += 1
    return invalid


# ---------------------------------------------------------------------------
# Smoke-word verification
# ---------------------------------------------------------------------------

SMOKE_WORDS: list[tuple[str, int]] = [
    ("мама",         0),
    ("вода",         1),
    ("університет",  4),
    ("читати",       1),
    ("місто",        0),
    ("батько",       0),
    ("книга",        0),
    ("земля",        1),
]


def _verify_smoke_words(con: sqlite3.Connection) -> list[tuple[str, bool, str]]:
    """Return list of (word, passed, detail) for each smoke word."""
    results = []
    for word, expected_idx in SMOKE_WORDS:
        rows = con.execute(
            "SELECT stress_indices_json FROM word_form WHERE form = ?", (word,)
        ).fetchall()
        if not rows:
            results.append((word, False, "NOT FOUND"))
            continue
        indices_seen = set()
        for (raw,) in rows:
            try:
                v = json.loads(raw)
                if isinstance(v, list):
                    for i in v:
                        indices_seen.add(i)
            except Exception:
                pass
        passed = expected_idx in indices_seen
        results.append((word, passed, f"indices={sorted(indices_seen)}, expected={expected_idx}"))
    return results


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def analyze(db_path: Path, verbose: bool = False) -> None:
    from datetime import datetime, timezone
    print(f"# Master DB Analysis — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print()
    print(f"  Path        : {db_path}")
    print(f"  Size        : {_size_mb(db_path):.1f} MB")
    sha = _sha256(db_path)
    print(f"  SHA-256     : {sha}")
    print()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    # ── Row counts ────────────────────────────────────────────────────────────
    wf_count = _row_count(con, "word_form")
    le_count = _row_count(con, "lemma_entry")
    feat_count = _row_count(con, "feature")
    print("## Row counts")
    print(f"  word_form   : {wf_count:>12,}")
    print(f"  lemma_entry : {le_count:>12,}")
    print(f"  feature     : {feat_count:>12,}")
    print()

    # ── NULL checks ───────────────────────────────────────────────────────────
    nulls = _null_counts(con)
    print("## NULL counts in word_form")
    for col, n in nulls.items():
        flag = "  ← WARNING" if n > 0 and col in ("form", "stress_indices_json") else ""
        print(f"  {col:<25}: {n:>8,}{flag}")
    print()

    # ── Stress index distribution ─────────────────────────────────────────────
    print("## Top-10 stress_indices_json values")
    for idx_json, cnt in _stress_index_distribution(con):
        print(f"  {idx_json:<15}  {cnt:>9,} rows")
    print()

    # ── POS distribution ──────────────────────────────────────────────────────
    print("## Top-10 POS values")
    for pos, cnt in _pos_distribution(con):
        print(f"  {str(pos):<12}  {cnt:>9,} rows")
    print()

    # ── JSON validity sample ─────────────────────────────────────────────────
    sample_size = 10_000
    invalid = _sample_stress_json_invalid(con, sample_size)
    print(f"## JSON validity (stress_indices_json, sample n={sample_size:,})")
    print(f"  Invalid rows : {invalid}")
    print()

    # ── Smoke words ───────────────────────────────────────────────────────────
    print("## Smoke-word verification")
    smoke_results = _verify_smoke_words(con)
    all_pass = True
    for word, passed, detail in smoke_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {word:<16}  {detail}")
    print()

    if verbose:
        # ── Extended: tables ─────────────────────────────────────────────────
        print("## All tables")
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for (t,) in tables:
            cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<30}: {cnt:>12,} rows")
        print()

    con.close()

    print("## Summary")
    print(f"  word_form rows   : {wf_count:,}  (baseline 4,078,014)")
    delta_pct = abs(wf_count - 4_078_014) / 4_078_014 * 100
    within = delta_pct <= 0.1
    print(f"  Delta from base  : {delta_pct:.4f}%  ({'OK ≤0.1%' if within else 'WARNING >0.1%'})")
    print(f"  JSON validity    : {'OK' if invalid == 0 else f'{invalid} invalid'}")
    print(f"  Smoke words      : {'ALL PASS' if all_pass else 'SOME FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the ua-stress master SQLite DB.")
    parser.add_argument("--db", help="Path to the master DB (overrides UA_STRESS_MASTER_DB)")
    parser.add_argument("--verbose", action="store_true", help="Extended output")
    args = parser.parse_args()

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = find_master_db()

    if not db_path or not db_path.exists():
        print("ERROR: Master DB not found.")
        print("  Set UA_STRESS_MASTER_DB env var or pass --db <path>.")
        raise SystemExit(1)

    analyze(db_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
