"""
test_stress_service.py — Production tests for the LMDB Ukrainian stress service.

Requires: src/nlp/stress_service/stress.lmdb  (data.mdb from LFS, ~167 MB)
If not present, all tests are skipped.
"""

from __future__ import annotations

import pytest
from pathlib import Path

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LMDB_PATH = _PROJECT_ROOT / "src/nlp/stress_service/stress.lmdb"

_DB_AVAILABLE = _LMDB_PATH.exists()
_skip_no_lmdb = pytest.mark.skipif(
    not _DB_AVAILABLE,
    reason=f"LMDB not found: {_LMDB_PATH}. Download from LFS first.",
)

# ── Baselines ─────────────────────────────────────────────────────────────────
# Verified against the 167 MB stress.lmdb shipped via LFS.
_BASELINE_ENTRY_COUNT = 2_858_219
_ENTRY_COUNT_TOLERANCE = 0.001  # ±0.1%

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


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def db():
    """Open a single LMDBQuery for the entire test module."""
    from src.nlp.stress_service.lmdb_query import LMDBQuery

    with LMDBQuery(_LMDB_PATH) as _db:
        yield _db


@pytest.fixture(scope="module")
def service():
    """Open a single UkrainianStressService for the entire test module."""
    from src.nlp.stress_service.stress_service import UkrainianStressService

    svc = UkrainianStressService(db_path=_LMDB_PATH)
    yield svc
    svc.close()


# ── TestIntegrity ─────────────────────────────────────────────────────────────
class TestIntegrity:
    @_skip_no_lmdb
    def test_lmdb_directory_exists(self):
        assert _LMDB_PATH.is_dir(), f"Expected a directory: {_LMDB_PATH}"

    @_skip_no_lmdb
    def test_data_mdb_present_and_nonempty(self):
        data_mdb = _LMDB_PATH / "data.mdb"
        assert data_mdb.exists(), "data.mdb missing inside stress.lmdb/"
        assert data_mdb.stat().st_size > 10_000_000, "data.mdb suspiciously small (<10 MB)"

    @_skip_no_lmdb
    def test_entry_count_near_baseline(self, db):
        count = db.entry_count
        lo = int(_BASELINE_ENTRY_COUNT * (1 - _ENTRY_COUNT_TOLERANCE))
        hi = int(_BASELINE_ENTRY_COUNT * (1 + _ENTRY_COUNT_TOLERANCE))
        assert lo <= count <= hi, (
            f"Entry count {count:,} outside expected range [{lo:,}, {hi:,}]"
        )


# ── TestLMDBQuery ─────────────────────────────────────────────────────────────
class TestLMDBQuery:
    @_skip_no_lmdb
    def test_lookup_returns_list(self, db):
        result = db.lookup("мама")
        assert isinstance(result, list)
        assert len(result) >= 1

    @_skip_no_lmdb
    def test_lookup_missing_word_returns_none(self, db):
        assert db.lookup("хрюмпельштільцхен") is None

    @_skip_no_lmdb
    def test_result_has_stress_variants_key(self, db):
        result = db.lookup("вода")
        assert result is not None
        for form in result:
            assert "stress_variants" in form

    @_skip_no_lmdb
    def test_stress_variants_are_nonneg_ints(self, db):
        result = db.lookup("університет")
        assert result is not None
        for form in result:
            for idx in form["stress_variants"]:
                assert isinstance(idx, int) and idx >= 0

    @_skip_no_lmdb
    def test_context_manager_closes_cleanly(self):
        from src.nlp.stress_service.lmdb_query import LMDBQuery
        with LMDBQuery(_LMDB_PATH) as db2:
            assert db2.lookup("мама") is not None
        # After __exit__, further access should raise
        with pytest.raises(Exception):
            db2.lookup("мама")


# ── TestSmokeWords ────────────────────────────────────────────────────────────
class TestSmokeWords:
    @_skip_no_lmdb
    @pytest.mark.parametrize("word,expected_stress", _SMOKE_WORDS)
    def test_smoke_word_stress_index(self, db, word, expected_stress):
        result = db.lookup(word)
        assert result is not None, f"Word not found: {word!r}"
        found_stresses = [form["stress_variants"] for form in result]
        assert expected_stress in found_stresses, (
            f"{word!r}: expected stress {expected_stress} in {found_stresses}"
        )

    @_skip_no_lmdb
    def test_heteronym_zamok_has_two_stress_variants(self, db):
        """замок: за́мок (castle, stress=0) and замо́к (lock, stress=1)."""
        result = db.lookup("замок")
        assert result is not None
        all_stresses = [form["stress_variants"] for form in result]
        unique_primary = {s[0] for s in all_stresses if s}
        assert 0 in unique_primary, "Expected stress=0 variant for замок (castle)"
        assert 1 in unique_primary, "Expected stress=1 variant for замок (lock)"


# ── TestUkrainianStressService ─────────────────────────────────────────────────
class TestUkrainianStressService:
    @_skip_no_lmdb
    def test_lookup_returns_list(self, service):
        result = service.lookup("мама")
        assert isinstance(result, list) and len(result) >= 1

    @_skip_no_lmdb
    def test_lookup_normalises_uppercase(self, service):
        lower = service.lookup("університет")
        upper = service.lookup("УНІВЕРСИТЕТ")
        assert lower is not None
        assert lower == upper

    @_skip_no_lmdb
    def test_lookup_missing_returns_none(self, service):
        assert service.lookup("хрюмпельштільцхен") is None

    @_skip_no_lmdb
    def test_is_heteronym_true(self, service):
        assert service.is_heteronym("замок") is True

    @_skip_no_lmdb
    def test_is_heteronym_false(self, service):
        assert service.is_heteronym("мама") is False

    @_skip_no_lmdb
    def test_get_stress_variants_returns_list_of_strings(self, service):
        variants = service.get_stress_variants("замок")
        assert variants is not None
        assert len(variants) >= 2
        for v in variants:
            assert isinstance(v, str)
            assert "\u0301" in v, f"No stress mark in {v!r}"

    @_skip_no_lmdb
    def test_get_stress_variants_missing_returns_none(self, service):
        assert service.get_stress_variants("хрюмпельштільцхен") is None
