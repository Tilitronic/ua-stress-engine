"""
variative_parser.py — Parser for the free-variant stress word list.

Reads `ua_variative_stressed_words.txt`: a line-delimited list of Ukrainian
lemmas that have two or more equally valid stress positions.  Lines starting
with `#` are comments; blank lines are ignored.

These words are NOT heteronyms (same word, different meaning depending on
stress) but rather *free variants* where both pronunciations are accepted by
the Ukrainian orthographic norm (e.g., "алфавіт" can be stressed on either
the second or third syllable).

During master-DB build, entries whose normalised form appears in this set
receive an extra flag in the merger so the stress service can report
ambiguity even when only one stress index was recorded in the raw sources.

Source file
-----------
  src/data_management/sources/ua_variative_stressed_words/ua_variative_stressed_words.txt

License
-------
  Curated manually; no third-party licence applies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Set

from src.utils.normalize_apostrophe import normalize_apostrophe

logger = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parent / "ua_variative_stressed_words.txt"


def _normalise(word: str) -> str:
    """Lowercase + normalise apostrophe."""
    return normalize_apostrophe(word.strip()).lower()


def iter_variative_words(path: Path = _DATA_FILE) -> Iterator[str]:
    """
    Yield normalised lemma strings from the variative word list.

    Skips blank lines and comment lines (starting with ``#``).

    Args:
        path: Path to the data file.  Defaults to the bundled
              ``ua_variative_stressed_words.txt``.

    Yields:
        Normalised lemma strings (lowercase, apostrophe-normalised).

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Variative word list not found: {path}"
        )

    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalised = _normalise(line)
            if normalised:
                yield normalised


def load_variative_words(path: Path = _DATA_FILE) -> Set[str]:
    """
    Load the full variative word list into a set for O(1) membership tests.

    Args:
        path: Path to the data file.

    Returns:
        Set of normalised lemma strings.
    """
    words = set(iter_variative_words(path))
    logger.info("Loaded %d variative words from %s", len(words), path)
    return words


def is_variative(word: str, variative_set: Set[str]) -> bool:
    """
    Return True if *word* is in the variative set.

    Args:
        word:          Raw word form (will be normalised internally).
        variative_set: Pre-loaded set returned by :func:`load_variative_words`.

    Returns:
        True if the word has free-variant stress.
    """
    return _normalise(word) in variative_set
