"""
Tests for src/data_management/sources/ua_variative_stressed_words/variative_parser.py

Uses the bundled ua_variative_stressed_words.txt (no network, no large files).
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = ROOT / "src" / "data_management" / "sources" / "ua_variative_stressed_words"
DATA_FILE = SOURCE_DIR / "ua_variative_stressed_words.txt"


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from src.data_management.sources.ua_variative_stressed_words.variative_parser import (
    load_variative_words,
    is_variative,
    iter_variative_words,
)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_data_file_exists():
    """The curated word list must be present."""
    assert DATA_FILE.exists(), f"Missing: {DATA_FILE}"


def test_data_file_nonzero():
    assert DATA_FILE.stat().st_size > 10


def test_data_file_has_non_comment_entries():
    """The data file must have at least one non-comment, non-blank line."""
    with DATA_FILE.open(encoding="utf-8") as fh:
        count = sum(
            1 for line in fh
            if line.strip() and not line.strip().startswith("#")
        )
    assert count >= 50, f"Expected >= 50 entries, found {count}"


# ---------------------------------------------------------------------------
# Integrity tests
# ---------------------------------------------------------------------------

class TestLoadVariativeWords:

    def test_returns_set(self):
        words = load_variative_words()
        assert isinstance(words, set)

    def test_set_is_nonempty(self):
        words = load_variative_words()
        assert len(words) >= 50

    def test_all_entries_are_strings(self):
        for word in load_variative_words():
            assert isinstance(word, str), f"Non-string entry: {word!r}"

    def test_all_entries_are_lowercase(self):
        for word in load_variative_words():
            assert word == word.lower(), f"Entry not lowercase: {word!r}"

    def test_no_comment_lines_included(self):
        for word in load_variative_words():
            assert not word.startswith("#"), f"Comment line leaked into set: {word!r}"

    def test_no_blank_entries(self):
        for word in load_variative_words():
            assert word.strip(), "Blank entry found in set"


# ---------------------------------------------------------------------------
# Smoke tests: known variative words must be present
# ---------------------------------------------------------------------------

KNOWN_VARIATIVE = [
    "алфавіт",
    "договір",
    "завжди",
    "апостроф",
]

@pytest.mark.parametrize("word", KNOWN_VARIATIVE)
def test_known_word_in_set(word: str):
    words = load_variative_words()
    assert word in words, f"Expected '{word}' in variative set"


# ---------------------------------------------------------------------------
# is_variative() tests
# ---------------------------------------------------------------------------

class TestIsVariative:

    @pytest.fixture(scope="class")
    def variative_set(self):
        return load_variative_words()

    def test_known_word_returns_true(self, variative_set):
        assert is_variative("алфавіт", variative_set) is True

    def test_unknown_word_returns_false(self, variative_set):
        # замок is a heteronym, not a free variant
        assert is_variative("замок", variative_set) is False

    def test_normalises_uppercase_input(self, variative_set):
        assert is_variative("Алфавіт", variative_set) is True

    def test_empty_string_returns_false(self, variative_set):
        assert is_variative("", variative_set) is False


# ---------------------------------------------------------------------------
# iter_variative_words() tests
# ---------------------------------------------------------------------------

class TestIterVariativeWords:

    def test_yields_strings(self):
        for word in iter_variative_words():
            assert isinstance(word, str)
            break  # just check first item

    def test_skips_comments(self):
        for word in iter_variative_words():
            assert not word.startswith("#")

    def test_skips_blank_lines(self):
        for word in iter_variative_words():
            assert word.strip() != ""

    def test_yields_lowercase(self):
        for word in iter_variative_words():
            assert word == word.lower()

    def test_file_not_found_raises(self, tmp_path):
        missing = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            list(iter_variative_words(path=missing))

    def test_count_matches_load(self):
        """iter and load must agree on the count."""
        from_iter = list(iter_variative_words())
        from_load = load_variative_words()
        # load deduplicates; iter may have duplicates — compare unique counts
        assert len(set(from_iter)) == len(from_load)
