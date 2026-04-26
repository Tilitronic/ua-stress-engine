#!/usr/bin/env python3
"""
analyze_stress_service.py
=========================
Human-readable summary of the LMDB Ukrainian stress database.

Prints: entry count, data.mdb size, smoke-word lookups, heteronym sample,
and morphological feature distribution.

Usage:
    python analyze_stress_service.py
    python analyze_stress_service.py --db path/to/stress.lmdb
    python analyze_stress_service.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.nlp.stress_service.lmdb_query import LMDBQuery

_DEFAULT_LMDB = _PROJECT_ROOT / "src/nlp/stress_service/stress.lmdb"

_SMOKE_WORDS = [
    ("мама",         [0]),
    ("вода",         [1]),
    ("університет",  [4]),
    ("читати",       [1]),
    ("місто",        [0]),
    ("батько",       [0]),
    ("книга",        [0]),
    ("земля",        [1]),
]


def _fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.1f} MB"


def analyze(db_path: Path, verbose: bool = False) -> None:
    data_mdb = db_path / "data.mdb"
    print(f"\nLMDB path : {db_path}")
    print(f"data.mdb  : {_fmt_mb(data_mdb)}" if data_mdb.exists() else "data.mdb  : MISSING")

    with LMDBQuery(db_path) as db:
        count = db.entry_count
        print(f"Entries   : {count:,}")

        print("\n── Smoke words ──────────────────────────────────────────────")
        all_pass = True
        for word, expected in _SMOKE_WORDS:
            result = db.lookup(word)
            if result is None:
                print(f"  MISS  {word}")
                all_pass = False
                continue
            found = [form["stress_variants"] for form in result]
            ok = expected in found
            mark = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{mark}]  {word:<20}  expected={expected}  found={found}")

        print("\n── Heteronym sample ─────────────────────────────────────────")
        heteronyms = ["замок", "атлас", "гвоздики", "оргáн"]
        for word in heteronyms:
            result = db.lookup(word)
            if result is None:
                print(f"  {word}: not found")
                continue
            stresses = [form["stress_variants"] for form in result]
            unique = sorted({s[0] for s in stresses if s})
            flag = "heteronym" if len(unique) > 1 else "single"
            print(f"  {word:<20}  stress positions={unique}  [{flag}]")

        if verbose:
            print("\n── Feature distribution (first 50k entries) ─────────────")
            import lmdb
            import msgpack
            pos_counts: dict[str, int] = {}
            n = 0
            env = lmdb.open(str(db_path), readonly=True, max_dbs=1, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for _, val in cursor.iternext(keys=False, values=True):
                    forms = msgpack.unpackb(val, raw=False)
                    for form in forms:
                        for pos in form.get("pos", ["_"]):
                            pos_counts[pos] = pos_counts.get(pos, 0) + 1
                    n += 1
                    if n >= 50_000:
                        break
            env.close()
            total = sum(pos_counts.values())
            for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
                print(f"  {pos:<12} {cnt:>8,}  ({cnt/total:.1%})")

    result_str = "ALL PASS" if all_pass else "SOME FAILED"
    print(f"\n{'═'*60}")
    print(f"  Smoke words: {result_str}")
    print(f"{'═'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LMDB Ukrainian stress database")
    parser.add_argument("--db", type=Path, default=_DEFAULT_LMDB,
                        help="Path to stress.lmdb directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Include POS distribution (scans first 50k entries)")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: LMDB not found: {args.db}", file=sys.stderr)
        print("Download data.mdb from LFS: git lfs pull", file=sys.stderr)
        sys.exit(1)

    analyze(args.db, verbose=args.verbose)


if __name__ == "__main__":
    main()
