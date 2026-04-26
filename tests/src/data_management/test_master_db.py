"""
tests/src/data_management/test_master_db.py

Production-readiness tests for the master SQLite database.

The DB is reproducible — run `python build_master_db.py` to rebuild it.
When the rebuild produces the same filename (MERGEDSQL_<hash>.sqlite3) as the
BASELINE_MERGED_HASH constant below, the source data is byte-for-byte identical
to the reference build.

Requires the DB to be present — skipped automatically if not found.
Set UA_STRESS_MASTER_DB or run `python build_master_db.py` first.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Locate the master DB
# ---------------------------------------------------------------------------

# parents[3] = ua-stress-engine project root
# parents[4] = poetykaAnalizerEngine  (sibling projects live here)
_PROJECT_ROOT = Path(__file__).parents[3]

_CACHE_DIRS = [
    # Canonical: within this project
    _PROJECT_ROOT / "src" / "data_management" / "transform" / "cache",
    # Sibling VersaSenseBackend project (common dev layout)
    _PROJECT_ROOT.parent / "VersaSenseEngine" / "VersaSenseBackend"
    / "src" / "data_management" / "transform" / "cache",
]


def _find_master_db() -> Path | None:
    env = os.environ.get("UA_STRESS_MASTER_DB")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for cache_dir in _CACHE_DIRS:
        candidates = sorted(cache_dir.glob("MERGEDSQL_*.sqlite3"))
        if candidates:
            return candidates[-1]
    return None


_MASTER_DB: Path | None = _find_master_db()

pytestmark = pytest.mark.skipif(
    _MASTER_DB is None,
    reason=(
        "Master DB not found. Run `python build_master_db.py` to build it, "
        "or set UA_STRESS_MASTER_DB env var."
    ),
)

# ---------------------------------------------------------------------------
# Baseline constants (REMAKE.md §5)
# ---------------------------------------------------------------------------

# The merged-source hash is embedded in the DB filename:
#   MERGEDSQL_<BASELINE_MERGED_HASH>.sqlite3
# Rebuilding from the same source data produces the same hash → same filename.
# This is the primary data-integrity proof — no separate hash file needed.
BASELINE_MERGED_HASH = "33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665"

BASELINE_ROW_COUNT = 4_078_014
BASELINE_TOLERANCE = 0.001  # ±0.1 %

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

REQUIRED_TABLES = [
    "word_form",
    "lemma_entry",
    "feature",
    "definition",
    "etymology_text",
]

WORD_FORM_COLS = [
    "id", "form", "lemma", "pos", "stress_indices_json",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def con() -> sqlite3.Connection:
    assert _MASTER_DB is not None
    connection = sqlite3.connect(str(_MASTER_DB))
    yield connection
    connection.close()


# ---------------------------------------------------------------------------
# Integrity — filename hash proves source-data identity
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_db_file_exists_and_is_nonempty(self):
        assert _MASTER_DB is not None
        assert _MASTER_DB.exists(), f"DB not found: {_MASTER_DB}"
        assert _MASTER_DB.stat().st_size > 100_000_000, "DB is suspiciously small (<100 MB)"

    def test_filename_encodes_baseline_merged_hash(self):
        """The filename MERGEDSQL_<hash>.sqlite3 must match the baseline.

        This proves the DB was built from the exact same source data as the
        reference build — no separate checksum file required.
        To rebuild: python build_master_db.py
        """
        assert _MASTER_DB is not None
        actual_hash = _MASTER_DB.stem.replace("MERGEDSQL_", "")
        assert actual_hash == BASELINE_MERGED_HASH, (
            f"Merged-source hash mismatch.\n"
            f"  DB filename hash : {actual_hash}\n"
            f"  Expected baseline: {BASELINE_MERGED_HASH}\n"
            f"  This means source data has changed, or the DB was built with\n"
            f"  different input files. Rebuild with: python build_master_db.py"
        )



class TestSchema:
    def test_required_tables_exist(self, con):
        actual = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in REQUIRED_TABLES:
            assert t in actual, f"Missing table: {t}"

    def test_word_form_columns(self, con):
        cols = {c[1] for c in con.execute("PRAGMA table_info(word_form)").fetchall()}
        for col in WORD_FORM_COLS:
            assert col in cols, f"Missing column word_form.{col}"

    def test_word_form_has_index_on_form(self, con):
        indexes = con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='word_form'"
        ).fetchall()
        names = {r[0] for r in indexes}
        # At least one index must reference the form column
        # (verify via PRAGMA index_info)
        form_indexed = False
        for idx_name in names:
            cols = con.execute(f"PRAGMA index_info({idx_name})").fetchall()
            if any(c[2] == "form" for c in cols):
                form_indexed = True
                break
        assert form_indexed, "No index on word_form.form"


# ---------------------------------------------------------------------------
# Row count
# ---------------------------------------------------------------------------

class TestRowCount:
    def test_word_form_row_count_near_baseline(self, con):
        count = con.execute("SELECT COUNT(*) FROM word_form").fetchone()[0]
        delta = abs(count - BASELINE_ROW_COUNT) / BASELINE_ROW_COUNT
        assert delta <= BASELINE_TOLERANCE, (
            f"word_form has {count:,} rows; "
            f"expected {BASELINE_ROW_COUNT:,} ±{BASELINE_TOLERANCE*100}% "
            f"(delta {delta*100:.3f}%)"
        )

    def test_lemma_entry_not_empty(self, con):
        count = con.execute("SELECT COUNT(*) FROM lemma_entry").fetchone()[0]
        assert count > 100_000, f"lemma_entry has only {count:,} rows"


# ---------------------------------------------------------------------------
# NULL / integrity checks
# ---------------------------------------------------------------------------

class TestNullChecks:
    def test_no_null_form(self, con):
        n = con.execute(
            "SELECT COUNT(*) FROM word_form WHERE form IS NULL"
        ).fetchone()[0]
        assert n == 0, f"{n} rows have NULL form"

    def test_no_null_stress_indices(self, con):
        n = con.execute(
            "SELECT COUNT(*) FROM word_form WHERE stress_indices_json IS NULL"
        ).fetchone()[0]
        assert n == 0, f"{n} rows have NULL stress_indices_json"

    def test_no_empty_form(self, con):
        n = con.execute(
            "SELECT COUNT(*) FROM word_form WHERE TRIM(form) = ''"
        ).fetchone()[0]
        assert n == 0, f"{n} rows have empty form"


# ---------------------------------------------------------------------------
# JSON validity sample (10 000 rows)
# ---------------------------------------------------------------------------

class TestStressJsonValidity:
    def test_stress_json_is_valid_list_sample(self, con):
        rows = con.execute(
            "SELECT stress_indices_json FROM word_form ORDER BY RANDOM() LIMIT 10000"
        ).fetchall()
        invalid = 0
        for (raw,) in rows:
            try:
                v = json.loads(raw)
                if not isinstance(v, list):
                    invalid += 1
            except (json.JSONDecodeError, TypeError):
                invalid += 1
        assert invalid == 0, f"{invalid}/10,000 sampled rows have invalid stress_indices_json"

    def test_stress_indices_are_non_negative_ints_sample(self, con):
        rows = con.execute(
            "SELECT stress_indices_json FROM word_form ORDER BY RANDOM() LIMIT 10000"
        ).fetchall()
        bad = 0
        for (raw,) in rows:
            try:
                indices = json.loads(raw)
                if not all(isinstance(i, int) and i >= 0 for i in indices):
                    bad += 1
            except Exception:
                bad += 1
        assert bad == 0, f"{bad}/10,000 sampled rows have out-of-range stress indices"


# ---------------------------------------------------------------------------
# Smoke words
# ---------------------------------------------------------------------------

class TestSmokeWords:
    @pytest.mark.parametrize("word,expected_idx", SMOKE_WORDS)
    def test_smoke_word_stress_index(self, con, word, expected_idx):
        rows = con.execute(
            "SELECT stress_indices_json FROM word_form WHERE form = ?", (word,)
        ).fetchall()
        assert rows, f"Word '{word}' not found in word_form"
        indices_seen: set[int] = set()
        for (raw,) in rows:
            try:
                v = json.loads(raw)
                indices_seen.update(v)
            except Exception:
                pass
        assert expected_idx in indices_seen, (
            f"'{word}': expected stress index {expected_idx} "
            f"but found {sorted(indices_seen)}"
        )

    def test_heteronym_zamok_has_two_stress_variants(self, con):
        """замок (castle/lock) must appear with both stress index 0 and 1."""
        rows = con.execute(
            "SELECT DISTINCT stress_indices_json FROM word_form WHERE form = 'замок'"
        ).fetchall()
        indices = set()
        for (raw,) in rows:
            try:
                indices.update(json.loads(raw))
            except Exception:
                pass
        assert 0 in indices, "замок missing stress index 0 (замОк — lock)"
        assert 1 in indices, "замок missing stress index 1 (зАмок — castle)"
