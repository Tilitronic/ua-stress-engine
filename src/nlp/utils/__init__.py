"""
NLP Utils Package

Utilities for Ukrainian language processing.
"""

from .apostrophe import (
    normalize_apostrophe,
    normalize_word,
    normalize_text,
    has_wrong_apostrophe,
    get_apostrophe_info,
    CORRECT_APOSTROPHE,
    WRONG_APOSTROPHES,
)

__all__ = [
    'normalize_apostrophe',
    'normalize_word',
    'normalize_text',
    'has_wrong_apostrophe',
    'get_apostrophe_info',
    'CORRECT_APOSTROPHE',
    'WRONG_APOSTROPHES',
]

__version__ = '1.0.0'
