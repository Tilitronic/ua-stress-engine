#!/usr/bin/env python3
"""
export_lmdb_to_json.py — Export the Ukrainian stress LMDB database to JSON.

The output JSON is consumed by the Rust builder crate:
    cargo run -p builder --release

Output format
-------------
{
  "замок": [
    {
      "stress_variants": [0],
      "pos": ["NOUN"],
      "feats": {"Case": ["Nom"], "Gender": ["Masc"], "Number": ["Sing"]},
      "lemma": "замок"
    },
    ...
  ],
  ...
}

Usage
-----
    python scripts/export_lmdb_to_json.py
    python scripts/export_lmdb_to_json.py --db path/to/stress.lmdb --out data/processed/ua_stress_export.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import lmdb
    import msgpack
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with:  pip install lmdb msgpack"
    )

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "src" / "nlp" / "stress_service" / "stress.lmdb"
DEFAULT_OUT = ROOT / "data" / "processed" / "ua_stress_export.json"


def export(db_path: Path, out_path: Path) -> None:
    if not db_path.exists():
        sys.exit(
            f"LMDB database not found: {db_path}\n"
            "Build it first with:  python build_master_db.py"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening LMDB: {db_path}")
    env = lmdb.open(
        str(db_path),
        readonly=True,
        lock=False,
        max_dbs=1,
        map_size=2 ** 30,
    )

    result: dict = {}
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for raw_key, raw_value in cursor.iternext(keys=True, values=True):
            word = raw_key.decode("utf-8")
            forms = msgpack.unpackb(raw_value, raw=False)
            # Ensure stress_variants contains plain Python ints (not numpy ints)
            clean_forms = []
            for form in forms:
                clean = {
                    "stress_variants": [int(v) for v in form.get("stress_variants", [])],
                    "pos": list(form.get("pos", [])),
                    "feats": {k: list(vs) for k, vs in form.get("feats", {}).items()},
                }
                if form.get("lemma") is not None:
                    clean["lemma"] = form["lemma"]
                clean_forms.append(clean)
            result[word] = clean_forms

    env.close()

    entry_count = len(result)
    print(f"Exporting {entry_count:,} word forms → {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Done — {out_path.name}: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"Path to the stress.lmdb directory (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()
    export(args.db, args.out)


if __name__ == "__main__":
    main()
