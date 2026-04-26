"""
ua_variative_stressed_words — Free-variant stress annotation source.

This source provides a curated list of Ukrainian lemmas that have two or more
equally valid stress positions (free variants, not heteronyms with different
meanings). During merging these words are flagged so the stress service can
report them as variative rather than resolving to a single canonical position.

Usage:
    from src.data_management.sources.ua_variative_stressed_words import (
        load_variative_words,
        is_variative,
    )

    variative = load_variative_words()
    print("алфавіт" in variative)  # True
"""

from .variative_parser import load_variative_words, is_variative, iter_variative_words

__all__ = [
    "load_variative_words",
    "is_variative",
    "iter_variative_words",
]
