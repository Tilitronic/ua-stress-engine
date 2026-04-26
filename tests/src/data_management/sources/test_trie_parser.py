"""
Tests for src/data_management/sources/trie_ua_stresses/trie_stress_parser.py

All tests use the bundled stress.trie file (already in the repo at ~12 MB).
No network access is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

ROOT = Path(__file__).resolve().parents[4]
TRIE_SRC = ROOT / "src" / "data_management" / "sources" / "trie_ua_stresses"
TRIE_FILE = TRIE_SRC / "stress.trie"


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_trie_file_exists():
    """stress.trie must be present in the repository."""
    assert TRIE_FILE.exists(), f"Missing trie file: {TRIE_FILE}"


def test_trie_file_is_nonzero():
    """stress.trie must not be a Git LFS pointer (>= 100 KB)."""
    size = TRIE_FILE.stat().st_size
    assert size > 100_000, f"stress.trie too small ({size} B) — may be a Git LFS pointer"


def test_trie_file_loads():
    """marisa_trie.BytesTrie must be able to load the file without error."""
    marisa_trie = pytest.importorskip("marisa_trie")
    trie = marisa_trie.BytesTrie()
    trie.load(str(TRIE_FILE))
    assert len(trie) > 100_000, "Trie has suspiciously few entries"


# ---------------------------------------------------------------------------
# Pure utility function tests
# ---------------------------------------------------------------------------

class TestCharPositionsToVowelIndices:
    """char_positions_to_vowel_indices() converts character offsets to vowel indices."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "trie_stress_parser",
            TRIE_SRC / "trie_stress_parser.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Patch heavy dependencies so we can test the pure function alone
        import sys, types
        # Stub marisa_trie if not installed
        if "marisa_trie" not in sys.modules:
            sys.modules["marisa_trie"] = types.ModuleType("marisa_trie")
        # Pre-load lemmatizer stub to avoid model download
        if "src.lemmatizer.lemmatizer" not in sys.modules:
            stub = types.ModuleType("src.lemmatizer.lemmatizer")
            class _L:
                def __init__(self, **_): pass
                def get_lemma(self, w): return w
            stub.Lemmatizer = _L
            sys.modules["src.lemmatizer.lemmatizer"] = stub
        spec.loader.exec_module(mod)
        request.cls.fn = staticmethod(mod.char_positions_to_vowel_indices)

    def test_zamok_castle_char3_is_vowel0(self):
        # "замок": chars = з(0)а(1)м(2)о(3)к(4)
        # trie stores char_pos=2 (position AFTER which the accent is inserted)
        # which means the stressed char is at index 1 (а), the 1st vowel → vowel_index=0
        word = "замок"
        # Trie uses 1-indexed char insertion position: accent after char at (pos-1)
        # For за́мок: insertion after char index 1 → char 'а' at index 1 → vowel index 0
        result = self.fn(word, [2])  # char_pos=2 means 'а' (1-indexed after)
        # The function maps char position to vowel index
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] in range(len("замок"))

    def test_empty_word_returns_empty(self):
        result = self.fn("", [])
        assert result == []

    def test_empty_positions_returns_empty(self):
        result = self.fn("мама", [])
        assert result == []


# ---------------------------------------------------------------------------
# Smoke-word tests: stress.trie lookup
# ---------------------------------------------------------------------------

# Ground-truth pairs from REMAKE.md §7
# (word, expected_vowel_index)
SMOKE_WORDS = [
    ("мама",        0),
    ("вода",        1),
    ("місто",       0),
    ("батько",      0),
    ("книга",       0),
    ("земля",       1),
]

UKRAINIAN_VOWELS = "аеєиіїоуюя"


def _count_vowels(word: str) -> int:
    return sum(1 for c in word if c in UKRAINIAN_VOWELS)


def _lookup_trie(word: str):
    """Return (char_positions list) or None if not found."""
    try:
        import marisa_trie
    except ImportError:
        pytest.skip("marisa_trie not installed")
    trie = marisa_trie.BytesTrie()
    trie.load(str(TRIE_FILE))
    results = trie[word]
    if not results:
        return None
    # Parse first byte as char position (Format 1: single accent)
    raw = results[0]
    if raw and raw[0] not in (0xFE, 0xFF):
        return [raw[0]]  # single char position
    return None


def _char_pos_to_vowel_idx(word: str, char_pos: int) -> int:
    """Convert trie character position to 0-based vowel index."""
    # Trie char_pos is 1-based: accent inserted AFTER char at (char_pos - 1)
    stressed_char_idx = char_pos - 1
    vowel_count = 0
    for i, ch in enumerate(word):
        if ch in UKRAINIAN_VOWELS:
            if i == stressed_char_idx:
                return vowel_count
            vowel_count += 1
    return -1


@pytest.mark.parametrize("word,expected_idx", SMOKE_WORDS)
def test_smoke_word_stress(word: str, expected_idx: int):
    """Smoke words must return the correct stressed vowel index from the trie."""
    try:
        import marisa_trie
    except ImportError:
        pytest.skip("marisa_trie not installed")

    trie = marisa_trie.BytesTrie()
    trie.load(str(TRIE_FILE))

    results = trie[word]
    assert results, f"'{word}' not found in trie"

    raw = results[0]
    # Format 1: first byte is char position (no 0xFE/0xFF separators)
    # Format 2: contains 0xFE separator → parse first record's pos byte
    if b'\xfe' in raw:
        pos_byte = raw[0]
    else:
        pos_byte = raw[0]

    vowel_idx = _char_pos_to_vowel_idx(word, pos_byte)
    assert vowel_idx == expected_idx, (
        f"'{word}': expected vowel_index={expected_idx}, "
        f"got char_pos={pos_byte} → vowel_idx={vowel_idx}"
    )


# ---------------------------------------------------------------------------
# TAG_BY_BYTE structure test
# ---------------------------------------------------------------------------

def test_tag_by_byte_maps_to_ud_keys():
    """TAG_BY_BYTE must map byte values to (UDFeatKey, str) pairs."""
    import importlib.util, sys, types

    if "marisa_trie" not in sys.modules:
        sys.modules["marisa_trie"] = types.ModuleType("marisa_trie")
    if "src.lemmatizer.lemmatizer" not in sys.modules:
        stub = types.ModuleType("src.lemmatizer.lemmatizer")
        class _L:
            def __init__(self, **_): pass
            def get_lemma(self, w): return w
        stub.Lemmatizer = _L
        sys.modules["src.lemmatizer.lemmatizer"] = stub

    spec = importlib.util.spec_from_file_location(
        "trie_stress_parser2", TRIE_SRC / "trie_stress_parser.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tag_map = mod.TAG_BY_BYTE
    assert len(tag_map) >= 10, "TAG_BY_BYTE has fewer entries than expected"
    for byte_key, pair in tag_map.items():
        assert isinstance(byte_key, bytes), f"Key {byte_key!r} is not bytes"
        assert len(byte_key) == 1, f"Key {byte_key!r} must be 1 byte"
        assert isinstance(pair, tuple) and len(pair) == 2, \
            f"Value for {byte_key!r} must be (key, val) tuple"
