"""
tests.py — Validation suite for a built .ctrie file.

Called automatically at the end of every build.  Also importable as a
pytest module — pytest discovers it and runs all test_* functions.

Usage (standalone, called by build script):
    from src.data_management.export.web_stress_db.tests import run_tests
    run_tests(ctrie_bytes)   # raises AssertionError on failure

Usage (pytest):
    pytest tests/src/data_management/test_web_stress_db.py
"""

from __future__ import annotations

import gzip
import logging
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Ground-truth word samples for smoke tests ─────────────────────────────────
# Format: (word, expected_stress_index)
# Stress index = 0-based vowel position counting from word start.
# Verified manually against the source txt dictionary (U+0301 mark).
SMOKE_WORDS: list[tuple[str, int]] = [
    # word          vowels                   stressed   idx
    ("мама",        0),   # м-А-м-а  → а(0)  → МАма    (stress on 1st syllable)
    ("вода",        1),   # в-о-д-А  → а(1)  → водА
    ("університет", 4),   # у-н-і-в-е-р-с-и-т-Е-т → е(4) → університЕт
    ("читати",      1),   # ч-и-т-А-т-и → а(1) → читАти
    ("місто",       0),   # м-І-с-т-о → і(0) → Місто
    ("батько",      0),   # б-А-т-ь-к-о → а(0) → Батько
    ("книга",       0),   # к-н-И-г-а → и(0) → кнИга
    ("земля",       1),   # з-е-м-л-Я → я(1) → землЯ
]

# Words that must NOT be in the trie (contain spaces or are phrases)
PHRASE_WORDS = [
    "іди геть",
    "будь ласка",
    "добрий день",
]


def run_tests(ctrie_bytes: bytes, *, verbose: bool = True) -> None:
    """
    Validate a serialised .ctrie blob.

    Runs all checks; raises AssertionError with a descriptive message on failure.
    Prints a summary to logger on success.
    """
    log = logger.info if verbose else lambda *a, **k: None

    log("=" * 60)
    log("Running .ctrie validation suite")
    log("=" * 60)

    _test_magic(ctrie_bytes)
    log("  [OK] magic header")

    _test_header_counts(ctrie_bytes)
    log("  [OK] header counts consistent")

    _test_alphabet(ctrie_bytes)
    log("  [OK] alphabet table")

    _test_node_size(ctrie_bytes)
    log("  [OK] node array size matches node_count")

    hit, miss, wrong = _test_smoke_words(ctrie_bytes)
    log(f"  [OK] smoke words — {hit} hit / {miss} miss / {wrong} wrong stress")
    if wrong > 0:
        raise AssertionError(
            f"{wrong} smoke words returned wrong stress index. "
            "Check SMOKE_WORDS ground truth vs source dictionary."
        )

    _test_phrase_words_absent(ctrie_bytes)
    log("  [OK] phrase words correctly absent")

    _test_gzip_roundtrip(ctrie_bytes)
    log("  [OK] gzip roundtrip")

    _test_coverage(ctrie_bytes)
    log("  [OK] coverage (word_count > 500k)")

    log("=" * 60)
    log("All tests passed.")
    log("=" * 60)


# ── Individual test functions (also discovered by pytest) ─────────────────────

def _load_fixture() -> bytes:
    """Load the default built .ctrie file for pytest runs."""
    import pytest
    # Try gzipped first, then raw
    candidates = [
        Path("src/data_management/export/web_stress_db/dist/ua_stress.ctrie.gz"),
        Path("src/data_management/export/web_stress_db/dist/ua_stress.ctrie"),
    ]
    for p in candidates:
        if p.exists():
            data = p.read_bytes()
            if p.suffix == ".gz":
                return gzip.decompress(data)
            return data
    pytest.skip("No built .ctrie found. Run `python build_web_stress_db.py` first.")


def test_magic():
    _test_magic(_load_fixture())


def test_header_counts():
    _test_header_counts(_load_fixture())


def test_alphabet():
    _test_alphabet(_load_fixture())


def test_node_size():
    _test_node_size(_load_fixture())


def test_smoke_words():
    data = _load_fixture()
    hit, miss, wrong = _test_smoke_words(data)
    assert wrong == 0, f"{wrong} smoke words returned wrong stress index"
    assert miss <= len(SMOKE_WORDS) // 2, \
        f"Too many smoke words missing from trie: {miss}/{len(SMOKE_WORDS)}"


def test_phrases_absent():
    _test_phrase_words_absent(_load_fixture())


def test_gzip_roundtrip():
    _test_gzip_roundtrip(_load_fixture())


def test_coverage():
    _test_coverage(_load_fixture())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _test_magic(data: bytes) -> None:
    assert data[:4] == b"UKST", f"Bad magic bytes: {data[:4]!r}"
    assert data[4] in (0x01, 0x02), f"Unexpected version: {data[4]}"


def _test_header_counts(data: bytes) -> None:
    node_count, word_count, alpha_size = struct.unpack_from("<III", data, 6)
    assert node_count > 0, "node_count is 0"
    assert word_count > 0, "word_count is 0"
    assert 30 <= alpha_size <= 50, f"Unexpected alphabet_size: {alpha_size}"


def _test_alphabet(data: bytes) -> None:
    from .trie import HEADER_SIZE, ALPHABET_ENTRY_SIZE, deserialize
    _, alphabet = deserialize(data)
    assert len(alphabet) >= 33, f"Alphabet too small: {len(alphabet)}"
    # Must contain basic Ukrainian vowels
    for ch in "аеіоу":
        assert ch in alphabet, f"Missing vowel '{ch}' in alphabet"


def _test_node_size(data: bytes) -> None:
    from .trie import HEADER_SIZE, ALPHABET_ENTRY_SIZE, NODE_SIZE, deserialize
    node_count, _, alpha_size = struct.unpack_from("<III", data, 6)
    expected_node_bytes = node_count * NODE_SIZE
    alphabet_bytes = alpha_size * ALPHABET_ENTRY_SIZE
    actual_node_bytes = len(data) - HEADER_SIZE - alphabet_bytes
    assert actual_node_bytes == expected_node_bytes, (
        f"Node array size mismatch: expected {expected_node_bytes}, "
        f"got {actual_node_bytes}"
    )


def _test_smoke_words(data: bytes) -> tuple[int, int, int]:
    from .trie import lookup
    hit = miss = wrong = 0
    for word, expected_stress in SMOKE_WORDS:
        result = lookup(data, word)
        if result is None:
            miss += 1
            logger.warning(f"  MISS: '{word}' not found in trie")
        else:
            stress, uncertain = result
            if stress != expected_stress:
                wrong += 1
                logger.warning(
                    f"  WRONG: '{word}' → stress={stress}, expected={expected_stress}"
                )
            else:
                hit += 1
    return hit, miss, wrong


def _test_phrase_words_absent(data: bytes) -> None:
    from .trie import lookup
    for phrase in PHRASE_WORDS:
        # Phrases contain spaces; the trie skips them at build time.
        # lookup() will normalise and try anyway — it should return None
        # because spaces are outside the Ukrainian alphabet mapping.
        result = lookup(data, phrase)
        assert result is None, f"Phrase '{phrase}' should not be in trie"


def _test_gzip_roundtrip(data: bytes) -> None:
    compressed = gzip.compress(data, compresslevel=9)
    decompressed = gzip.decompress(compressed)
    assert decompressed == data, "gzip roundtrip failed"
    ratio = len(compressed) / len(data)
    assert ratio < 0.60, f"Compression ratio too poor: {ratio:.2%}"
    logger.info(
        f"  gzip: {len(data)/1024/1024:.1f} MB raw → "
        f"{len(compressed)/1024/1024:.1f} MB compressed ({ratio:.0%})"
    )


def _test_coverage(data: bytes) -> None:
    _, word_count, _ = struct.unpack_from("<III", data, 6)
    assert word_count > 500_000, (
        f"word_count={word_count:,} looks too low — expected >500 000"
    )
    logger.info(f"  word_count: {word_count:,}")
