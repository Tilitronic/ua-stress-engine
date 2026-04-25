"""
web_stress_db — Ukrainian stress trie builder for browser deployment.

Public API:
    TrieNode, TrieBuilder  — build the trie from word data
    serialize              — encode trie to binary .ctrie bytes
    load_from_master_db    — extract (form, stress, heteronym) rows from SQLite master
    run_tests              — validate a .ctrie file (called automatically after build)
"""

from .trie import TrieNode, TrieBuilder, serialize, deserialize
from .loader import load_from_master_db
from .tests import run_tests

__all__ = [
    "TrieNode",
    "TrieBuilder",
    "serialize",
    "deserialize",
    "load_from_master_db",
    "run_tests",
]
