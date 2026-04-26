"""
Tests for src/data_management/sources/txt_ua_stresses/txt_stress_parser.py

All tests use sample_stress_dict.txt (bundled, ~30 entries).
No network access; no large file required.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[4]
TXT_SRC = ROOT / "src" / "data_management" / "sources" / "txt_ua_stresses"
SAMPLE_FILE = TXT_SRC / "sample_stress_dict.txt"
FULL_DICT = TXT_SRC / "ua_word_stress_dictionary.txt"

COMBINING_ACUTE = "\u0301"
UKRAINIAN_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")


# ---------------------------------------------------------------------------
# Helper — pure Python stress extraction (mirrors parser logic)
# ---------------------------------------------------------------------------

def _extract_stress_indices_pure(form: str) -> List[int]:
    """Extract 0-based vowel indices where combining-acute follows the vowel."""
    indices = []
    vowel_count = 0
    chars = list(unicodedata.normalize("NFC", form))
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch in UKRAINIAN_VOWELS:
            # Check if the next character is a combining acute
            if i + 1 < len(chars) and chars[i + 1] == COMBINING_ACUTE:
                indices.append(vowel_count)
                i += 1  # skip the combining accent
            vowel_count += 1
        i += 1
    return indices


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_sample_file_exists():
    """sample_stress_dict.txt must be present in the repo."""
    assert SAMPLE_FILE.exists(), f"Missing fixture: {SAMPLE_FILE}"


def test_sample_file_is_nonzero():
    assert SAMPLE_FILE.stat().st_size > 100


def test_sample_file_has_stress_marks():
    """At least some lines in the sample must contain combining-acute marks."""
    content = SAMPLE_FILE.read_text(encoding="utf-8")
    assert COMBINING_ACUTE in content, "Sample file has no stress marks (U+0301)"


def test_sample_file_no_blank_data_lines():
    """Non-comment, non-blank lines must contain at least one letter."""
    for line in SAMPLE_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert any(c.isalpha() for c in stripped), \
                f"Data line contains no letters: {stripped!r}"


# ---------------------------------------------------------------------------
# Pure utility: stress extraction from marked form strings
# ---------------------------------------------------------------------------

class TestExtractStressIndicesPure:
    """Verify our pure extraction helper against known words."""

    @pytest.mark.parametrize("marked, expected", [
        ("за\u0301мок",       [0]),   # за́мок  — castle, а=vowel 0
        ("замо\u0301к",       [1]),   # замо́к  — lock,   о=vowel 1
        ("вода\u0301",        [1]),   # вода́            а=vowel 1
        ("ма\u0301ма",        [0]),   # ма́ма             а=vowel 0
        ("мо\u0301ва",        [0]),   # мо́ва
        ("замо\u0301к",       [1]),
        ("по\u0301ми\u0301лка", [0, 1]),  # double-stress: both vowels
        ("книга",             []),    # no stress mark
    ])
    def test_indices(self, marked, expected):
        result = _extract_stress_indices_pure(marked)
        assert result == expected, f"Form {marked!r}: expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# Sample file smoke tests
# ---------------------------------------------------------------------------

class TestSampleFileSmoke:
    """Parse sample_stress_dict.txt and verify known entries."""

    # Expected: (unstressed_form, vowel_index)
    EXPECTED = [
        ("замок", 1),   # замо́к — lock
        ("атлас", 0),   # а́тлас
        ("атлас", 1),   # атла́с   (both forms present)
    ]

    def _load_entries(self):
        """Return list of (base_form, vowel_indices) from the sample file."""
        entries = []
        for line in SAMPLE_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indices = _extract_stress_indices_pure(stripped)
            base = "".join(
                c for c in unicodedata.normalize("NFD", stripped)
                if unicodedata.category(c) != "Mn"
            )
            entries.append((base.lower(), indices))
        return entries

    def test_zamok_lock_present(self):
        entries = self._load_entries()
        found = [idx for form, idx in entries if form == "замок" and 1 in idx]
        assert found, "замо́к (lock, index 1) not found in sample"

    def test_atlas_first_vowel_present(self):
        entries = self._load_entries()
        found = [idx for form, idx in entries if form == "атлас" and 0 in idx]
        assert found, "а́тлас (index 0) not found in sample"

    def test_atlas_second_vowel_present(self):
        entries = self._load_entries()
        found = [idx for form, idx in entries if form == "атлас" and 1 in idx]
        assert found, "атла́с (index 1) not found in sample"

    def test_all_entries_have_valid_form(self):
        entries = self._load_entries()
        assert len(entries) > 0
        for form, indices in entries:
            assert isinstance(form, str) and form, "Empty form"
            assert isinstance(indices, list), "Indices must be a list"

    def test_all_stress_indices_non_negative(self):
        entries = self._load_entries()
        for form, indices in entries:
            for idx in indices:
                assert idx >= 0, f"Negative index {idx} for form {form!r}"

    def test_mama_stress_index(self):
        """мама → stress on 0th vowel (МАма)."""
        entries = self._load_entries()
        found = [idx for form, idx in entries if form == "мама"]
        # мама may or may not be in sample — if present, check index
        for idx in found:
            assert 0 in idx, f"мама: expected vowel 0, got {idx}"

    def test_voda_stress_index(self):
        """вода → stress on 1st vowel (воДА)."""
        entries = self._load_entries()
        found = [idx for form, idx in entries if form == "вода"]
        for idx in found:
            assert 1 in idx, f"вода: expected vowel 1, got {idx}"


# ---------------------------------------------------------------------------
# Full-file smoke test (only runs if the large file is present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not FULL_DICT.exists() or FULL_DICT.stat().st_size < 1_000_000,
    reason="Full dictionary not available (Git LFS not pulled)",
)
def test_full_dict_word_count():
    """Full dictionary must have > 1 M lines (sanity check)."""
    with FULL_DICT.open(encoding="utf-8") as fh:
        count = sum(
            1 for line in fh
            if line.strip() and not line.startswith("#")
        )
    assert count > 1_000_000, f"Full dict has only {count} entries"
