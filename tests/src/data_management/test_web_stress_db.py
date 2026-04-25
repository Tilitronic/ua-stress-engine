"""
Pytest entry point for web_stress_db validation tests.

Discovers and re-exports all test_* functions from the module's own test suite.
Run from project root:

    pytest tests/src/data_management/test_web_stress_db.py -v

The .ctrie file must already be built:

    python build_web_stress_db.py
"""

from src.data_management.export.web_stress_db.tests import (
    test_magic,
    test_header_counts,
    test_alphabet,
    test_node_size,
    test_smoke_words,
    test_phrases_absent,
    test_gzip_roundtrip,
    test_coverage,
)

__all__ = [
    "test_magic",
    "test_header_counts",
    "test_alphabet",
    "test_node_size",
    "test_smoke_words",
    "test_phrases_absent",
    "test_gzip_roundtrip",
    "test_coverage",
]
