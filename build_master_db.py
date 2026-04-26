#!/usr/bin/env python3
"""
build_master_db.py — Rebuild the master Ukrainian stress SQLite database.

Runs the full pipeline: parse 3 sources → intermediate LMDBs → merge → SQLite.
The output filename encodes a hash of the merged source data, so rebuilding
from identical sources produces the exact same filename — which is itself the
integrity proof.

Usage:
    python build_master_db.py
    python build_master_db.py --kaikki-jsonl /path/to/kaikki.org-dictionary-Ukrainian.jsonl
    python build_master_db.py --txt-dict /path/to/ua_word_stress_dictionary.txt
    python build_master_db.py --trie /path/to/stress.trie
    python build_master_db.py --verify-hash 33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665

Environment variable overrides (alternative to CLI args):
    UA_KAIKKI_JSONL   path to kaikki.org-dictionary-Ukrainian.jsonl
    UA_TXT_DICT       path to ua_word_stress_dictionary.txt
    UA_TRIE_DICT      path to stress.trie

After a successful build the script prints:
  - Path to the produced DB
  - Merged-source hash (embedded in filename)
  - SHA-256 of the SQLite file itself
  - Row counts for key tables
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# ── Project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Canonical paths for source data (can be in sibling VersaSenseBackend project)
# ---------------------------------------------------------------------------

_SIBLING_KAIKKI = (
    _ROOT.parent / "VersaSenseEngine" / "VersaSenseBackend"
    / "src" / "data_management" / "sources" / "kaikki"
    / "kaikki.org-dictionary-Ukrainian.jsonl"
)

BASELINE_MERGED_HASH = "33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665"
BASELINE_ROW_COUNT = 4_078_014


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_kaikki(cli_path: str | None) -> str:
    """Return resolved path to the kaikki JSONL, checking several locations."""
    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            print(f"ERROR: --kaikki-jsonl path not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        return str(p)

    env = os.environ.get("UA_KAIKKI_JSONL")
    if env:
        p = Path(env)
        if not p.exists():
            print(f"ERROR: UA_KAIKKI_JSONL env path not found: {p}", file=sys.stderr)
            raise SystemExit(1)
        return env

    # Built-in project path
    local = _ROOT / "src" / "data_management" / "sources" / "kaikki" / "kaikki.org-dictionary-Ukrainian.jsonl"
    if local.exists():
        return str(local)

    # Sibling VersaSenseBackend
    if _SIBLING_KAIKKI.exists():
        print(f"[INFO] Using kaikki JSONL from sibling project: {_SIBLING_KAIKKI}")
        return str(_SIBLING_KAIKKI)

    print(
        "ERROR: kaikki.org-dictionary-Ukrainian.jsonl not found.\n"
        "  Pass --kaikki-jsonl <path>  or set UA_KAIKKI_JSONL env var.\n"
        f"  Also tried: {local}\n"
        f"             {_SIBLING_KAIKKI}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _verify_db(db_path: Path, expected_merged_hash: str | None) -> bool:
    """Post-build verification: schema, row counts, smoke words, filename hash."""
    print("\n── Verification ────────────────────────────────────────────────")

    # 1. Filename hash check
    filename_hash = db_path.stem.replace("MERGEDSQL_", "")
    if expected_merged_hash:
        if filename_hash == expected_merged_hash:
            print(f"  [PASS] Merged-source hash matches baseline: {filename_hash[:16]}…")
        else:
            print(f"  [WARN] Merged-source hash differs from baseline.")
            print(f"         Expected : {expected_merged_hash}")
            print(f"         Got      : {filename_hash}")
            print("         This means source data has changed since the baseline was set.")
    else:
        print(f"  [INFO] Merged-source hash: {filename_hash}")

    # 2. File SHA-256
    sha = _sha256_file(db_path)
    print(f"  SHA-256 (file): {sha}")

    # 3. Row count
    con = sqlite3.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM word_form").fetchone()[0]
        delta = abs(count - BASELINE_ROW_COUNT) / BASELINE_ROW_COUNT * 100
        status = "PASS" if delta <= 0.1 else "WARN"
        print(f"  [{status}] word_form rows: {count:,}  (baseline {BASELINE_ROW_COUNT:,}, delta {delta:.3f}%)")

        # 4. Smoke words
        SMOKE = [("мама", 0), ("вода", 1), ("університет", 4),
                 ("читати", 1), ("місто", 0), ("батько", 0), ("книга", 0), ("земля", 1)]
        all_pass = True
        for word, expected_idx in SMOKE:
            rows = con.execute(
                "SELECT stress_indices_json FROM word_form WHERE form = ?", (word,)
            ).fetchall()
            indices: set[int] = set()
            for (raw,) in rows:
                try:
                    indices.update(json.loads(raw))
                except Exception:
                    pass
            passed = expected_idx in indices
            if not passed:
                all_pass = False
            print(f"  [{'PASS' if passed else 'FAIL'}] {word:<16} expected={expected_idx}  found={sorted(indices)}")
        return all_pass
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the master Ukrainian stress SQLite DB."
    )
    parser.add_argument("--kaikki-jsonl", help="Path to kaikki.org-dictionary-Ukrainian.jsonl")
    parser.add_argument("--txt-dict", help="Path to ua_word_stress_dictionary.txt")
    parser.add_argument("--trie", help="Path to stress.trie")
    parser.add_argument(
        "--verify-hash",
        default=BASELINE_MERGED_HASH,
        help="Expected merged-source hash (embedded in DB filename). Default: REMAKE.md baseline.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the build step and only run verification on an existing DB.",
    )
    args = parser.parse_args()

    # ── Resolve source paths ────────────────────────────────────────────────
    kaikki_jsonl = _resolve_kaikki(args.kaikki_jsonl)

    # Set env vars so ExportConfig picks them up at import time
    os.environ["UA_KAIKKI_JSONL"] = kaikki_jsonl
    if args.txt_dict:
        os.environ["UA_TXT_DICT"] = args.txt_dict
    if args.trie:
        os.environ["UA_TRIE_DICT"] = args.trie

    print("═" * 60)
    print("  Build Master DB")
    print("═" * 60)
    print(f"  TXT dict  : {os.environ.get('UA_TXT_DICT', '[project default]')}")
    print(f"  TRIE      : {os.environ.get('UA_TRIE_DICT', '[project default]')}")
    print(f"  KAIKKI    : {kaikki_jsonl}")
    print()

    if not args.skip_build:
        # Import AFTER setting env vars (ExportConfig reads them at import time)
        from src.data_management.transform.parsing_merging_service import main as pipeline_main

        t0 = time.time()
        print("[BUILD] Starting pipeline (this takes 30-90 min on first run)…")
        pipeline_main()
        elapsed = time.time() - t0
        print(f"\n[BUILD] Pipeline completed in {elapsed/60:.1f} min")

    # ── Find the produced DB ────────────────────────────────────────────────
    _cache_dirs = [
        _ROOT / "src" / "data_management" / "transform" / "cache",
        _ROOT.parent / "VersaSenseEngine" / "VersaSenseBackend"
        / "src" / "data_management" / "transform" / "cache",
    ]
    candidates = []
    for cache_dir in _cache_dirs:
        candidates.extend(cache_dir.glob("MERGEDSQL_*.sqlite3"))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("ERROR: No MERGEDSQL_*.sqlite3 found in cache/. Build may have failed.", file=sys.stderr)
        raise SystemExit(1)

    db_path = candidates[-1]
    print(f"\n[RESULT] DB: {db_path}")
    print(f"[RESULT] Size: {db_path.stat().st_size / (1024**2):.1f} MB")

    all_pass = _verify_db(db_path, args.verify_hash)

    print("\n" + ("═" * 60))
    print("  RESULT:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print("═" * 60)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
