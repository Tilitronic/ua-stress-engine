#!/usr/bin/env python3
"""
build_web_stress_db.py
======================
Build the web-optimised Ukrainian stress trie (.ctrie) from the master SQLite
database and write the output to:

    src/data_management/export/web_stress_db/dist/
        ua_stress.ctrie          — raw binary trie
        ua_stress.ctrie.gz       — gzip-compressed (serve this to browsers)
        manifest.json            — metadata for the JS loader

    packages/ua-stress-web/data/
        ua_stress.ctrie.gz       — copy for the npm package

Tests are run automatically after every successful build and abort with a
non-zero exit code on failure.

Usage:
    python build_web_stress_db.py
    python build_web_stress_db.py --db path/to/master.sqlite3
    python build_web_stress_db.py --db path/to/master.sqlite3 --out path/to/dist/
    python build_web_stress_db.py --skip-tests   # not recommended
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_management.export.web_stress_db.trie import TrieBuilder, serialize
from src.data_management.export.web_stress_db.loader import load_from_master_db
from src.data_management.export.web_stress_db.tests import run_tests

# ── Defaults ──────────────────────────────────────────────────────────────────
_MASTER_DB_NAME = (
    "MERGEDSQL_33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665.sqlite3"
)
_CANDIDATE_MASTER_DBS = [
    _PROJECT_ROOT / "src/data_management/transform/cache" / _MASTER_DB_NAME,
    _PROJECT_ROOT.parent
    / "VersaSenseEngine/VersaSenseBackend/src/data_management/transform/cache"
    / _MASTER_DB_NAME,
]
_DEFAULT_MASTER_DB = next(
    (p for p in _CANDIDATE_MASTER_DBS if p.exists()),
    _CANDIDATE_MASTER_DBS[0],  # fall back to first (will fail with helpful error)
)
_DEFAULT_OUT = (
    _PROJECT_ROOT
    / "src/data_management/export/web_stress_db/dist"
)
_NPM_DATA_DIR = (
    _PROJECT_ROOT
    / "packages/ua-stress-web/data"
)
_NPM_README = (
    _PROJECT_ROOT
    / "packages/ua-stress-web/README.md"
)

# ── README auto-stats updater ─────────────────────────────────────────────────

def _update_npm_readme(manifest: dict, log: logging.Logger) -> None:
    """Replace the STATS_START…STATS_END block in the npm README."""
    readme = _NPM_README
    if not readme.exists():
        log.warning(f"npm README not found, skipping stats update: {readme}")
        return

    built_iso = manifest["built"].split(".")[0] + "Z"
    heteronym_count = manifest.get("heteronym_count", "n/a")

    new_block = (
        "<!-- STATS_START -->\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Word forms | {manifest['word_count']:,} |\n"
        f"| Heteronyms (context-dependent stress) | {heteronym_count:,} |\n"
        f"| Trie nodes | {manifest['node_count']:,} |\n"
        f"| Compressed size (`ua_stress.ctrie.gz`) | {manifest['gz_size_mb']} MB |\n"
        f"| Format | {manifest['format']} |\n"
        f"| Last built | {built_iso} |\n"
        "<!-- STATS_END -->"
    )

    text = readme.read_text(encoding="utf-8")
    import re
    updated = re.sub(
        r"<!-- STATS_START -->.*?<!-- STATS_END -->",
        new_block,
        text,
        flags=re.DOTALL,
    )
    if updated != text:
        readme.write_text(updated, encoding="utf-8")
        log.info(f"  npm README stats updated: {readme}")


# ── Progress bar ──────────────────────────────────────────────────────────────

class _Bar:
    """Minimal inline progress bar — no external dependencies."""
    WIDTH = 36

    def __init__(self, total: int, label: str) -> None:
        self.total = max(total, 1)
        self.label = label
        self._t0 = time.perf_counter()

    def update(self, current: int) -> None:
        pct = min(current / self.total, 1.0)
        filled = int(self.WIDTH * pct)
        bar = "█" * filled + "░" * (self.WIDTH - filled)
        elapsed = time.perf_counter() - self._t0
        rate = current / elapsed if elapsed > 0 else 0
        eta = (self.total - current) / rate if rate > 0 and current < self.total else 0
        eta_str = f"  ETA {eta:.0f}s" if eta > 1 else ""
        print(
            f"\r  [{bar}] {current:>10,} / {self.total:,}  "
            f"{pct:>5.1%}  {rate:>10,.0f}/s{eta_str}     ",
            end="",
            flush=True,
        )

    def finish(self, current: int | None = None) -> None:
        n = current if current is not None else self.total
        elapsed = time.perf_counter() - self._t0
        rate = n / elapsed if elapsed > 0 else 0
        print(
            f"\r  {self.label}: {n:,} done in {elapsed:.1f}s  "
            f"({rate:,.0f}/s)                           "
        )


# ── Build ─────────────────────────────────────────────────────────────────────

def build(
    db_path: Path,
    out_dir: Path,
    skip_tests: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load from master DB ─────────────────────────────────────────────
    log.info(f"Master DB: {db_path}")
    if not db_path.exists():
        log.error(f"Master DB not found: {db_path}")
        sys.exit(1)

    log.info("Loading stress data from master DB…")
    all_entries = list(load_from_master_db(db_path))
    total_entries = len(all_entries)
    log.info(f"  {total_entries:,} unique forms ready for insertion")

    # ── 2. Build trie ──────────────────────────────────────────────────────
    log.info(f"Building trie ({total_entries:,} forms)…")
    builder = TrieBuilder()
    inserted = skipped = 0
    bar = _Bar(total_entries, "Words inserted")

    for i, (norm_form, stress, heteronym) in enumerate(all_entries):
        ok = builder.insert(norm_form, stress, heteronym)
        if ok:
            inserted += 1
        else:
            skipped += 1
        if (i + 1) % 100_000 == 0 or (i + 1) == total_entries:
            bar.update(i + 1)

    bar.finish(inserted)
    log.info(
        f"  Trie: {inserted:,} words inserted, {skipped:,} skipped "
        f"(unmapped chars)  |  {builder.node_count:,} nodes"
    )

    # ── 3. Serialise ──────────────────────────────────────────────────────
    log.info("Serialising trie to binary format…")
    node_count = builder.node_count
    ser_bar = _Bar(node_count, "Nodes serialised")

    def _ser_progress(current: int, total: int) -> None:
        ser_bar.update(current)

    ctrie_bytes = serialize(builder, progress=_ser_progress)
    ser_bar.finish(node_count)
    raw_size_mb = len(ctrie_bytes) / 1024 / 1024
    log.info(f"  Raw size: {raw_size_mb:.2f} MB  ({node_count:,} nodes × 8 B)")

    # ── 4. Gzip compress ──────────────────────────────────────────────────
    log.info("Compressing (gzip level 9)…")
    gz_bytes = gzip.compress(ctrie_bytes, compresslevel=9)
    gz_size_mb = len(gz_bytes) / 1024 / 1024
    ratio = len(gz_bytes) / len(ctrie_bytes)
    log.info(f"  Compressed: {gz_size_mb:.2f} MB  ({ratio:.0%} of raw)")

    # ── 5. Write dist/ outputs ─────────────────────────────────────────────
    raw_path = out_dir / "ua_stress.ctrie"
    gz_path = out_dir / "ua_stress.ctrie.gz"
    manifest_path = out_dir / "manifest.json"

    raw_path.write_bytes(ctrie_bytes)
    gz_path.write_bytes(gz_bytes)
    log.info(f"  Written: {raw_path}")
    log.info(f"  Written: {gz_path}")

    heteronym_count = sum(1 for _, _, h in all_entries if h)
    manifest = {
        "version": "1.0.0",
        "built": datetime.now(timezone.utc).isoformat(),
        "format": "ctrie-v1",
        "word_count": builder.word_count,
        "node_count": builder.node_count,
        "heteronym_count": heteronym_count,
        "raw_size_bytes": len(ctrie_bytes),
        "gz_size_bytes": len(gz_bytes),
        "gz_size_mb": round(gz_size_mb, 2),
        "files": {
            "gz": "ua_stress.ctrie.gz",
            "raw": "ua_stress.ctrie",
        },
        "serving": {
            "note": (
                "Serve ua_stress.ctrie.gz with "
                "Content-Encoding: gzip and Content-Type: application/octet-stream"
            ),
            "nginx": (
                "add_header Content-Encoding gzip; "
                "add_header Content-Type application/octet-stream;"
            ),
            "express": (
                "res.setHeader('Content-Encoding','gzip'); "
                "res.setHeader('Content-Type','application/octet-stream'); "
                "res.sendFile('ua_stress.ctrie.gz');"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log.info(f"  Written: {manifest_path}")

    # ── 6. Copy gz to npm/data/ ────────────────────────────────────────────
    _NPM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    npm_gz = _NPM_DATA_DIR / "ua_stress.ctrie.gz"
    shutil.copy2(gz_path, npm_gz)
    log.info(f"  Copied to npm package: {npm_gz}")

    # ── 7. Update npm README stats block ──────────────────────────────────
    _update_npm_readme(manifest, log)

    elapsed = time.perf_counter() - t0
    log.info(f"Build finished in {elapsed:.1f}s")

    # ── 7. Tests (mandatory) ──────────────────────────────────────────────
    if skip_tests:
        log.warning("Tests SKIPPED (--skip-tests). Not recommended for production builds.")
    else:
        log.info("Running validation tests…")
        try:
            run_tests(ctrie_bytes, verbose=True)
            log.info("All tests passed — build is valid.")
        except AssertionError as e:
            log.error(f"TEST FAILED: {e}")
            log.error("Build output retained for inspection but is NOT valid.")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ukrainian web stress trie")
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_MASTER_DB,
        help="Path to master SQLite database (default: auto-detected)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory for dist files (default: web_stress_db/dist/)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip post-build validation (not recommended)",
    )
    args = parser.parse_args()
    build(db_path=args.db, out_dir=args.out, skip_tests=args.skip_tests)


if __name__ == "__main__":
    main()
