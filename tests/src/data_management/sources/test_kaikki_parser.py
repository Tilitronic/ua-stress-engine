"""
Tests for src/data_management/sources/kaikki/kaikki_parser.py

Tests use kaikki.test.jsonl (5 entries: замок×2, блоха×2, помилка) and
do not require the full 100 MB JSONL file or network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
KAIKKI_SRC = ROOT / "src" / "data_management" / "sources" / "kaikki"
TEST_JSONL = KAIKKI_SRC / "kaikki.test.jsonl"

# ---------------------------------------------------------------------------
# Import helpers (avoid importing the full parser top-level so we can test
# pure utility functions without triggering LMDB / Lemmatizer initialisation)
# ---------------------------------------------------------------------------
import importlib.util, sys

def _load_parser():
    spec = importlib.util.spec_from_file_location(
        "kaikki_parser", KAIKKI_SRC / "kaikki_parser.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy-load so the expensive imports only fire when the test actually runs
@pytest.fixture(scope="module")
def parser():
    return _load_parser()


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_test_jsonl_exists():
    """Fixture file must be present in the repo."""
    assert TEST_JSONL.exists(), f"Missing fixture: {TEST_JSONL}"


def test_test_jsonl_is_valid_jsonl():
    """Every line in the fixture must parse as JSON."""
    lines = TEST_JSONL.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1, "Fixture is empty"
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert "word" in obj, f"Line {i}: missing 'word' key"
        assert "pos" in obj, f"Line {i}: missing 'pos' key"


def test_test_jsonl_contains_expected_words():
    """Fixture must include замок, блоха, помилка."""
    words = set()
    for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
        words.add(json.loads(line)["word"])
    assert "замок" in words
    assert "блоха" in words
    assert "помилка" in words


# ---------------------------------------------------------------------------
# Pure utility function tests (no heavy imports)
# ---------------------------------------------------------------------------

class TestStripStress:
    """strip_stress() must remove U+0301 combining acute marks."""

    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.strip = parser.strip_stress

    def test_removes_combining_acute(self):
        assert self.strip("за\u0301мок") == "замок"

    def test_leaves_unstressed_word_unchanged(self):
        assert self.strip("замок") == "замок"

    def test_removes_multiple_stress_marks(self):
        # по́ми́лка — two stress marks
        assert self.strip("по\u0301ми\u0301лка") == "помилка"

    def test_empty_string(self):
        assert self.strip("") == ""


class TestExtractStressIndices:
    """extract_stress_indices() returns 0-based vowel indices."""

    VOWELS = "аеєиіїоуюя"

    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.extract = parser.extract_stress_indices

    def test_castle_first_vowel(self):
        # за́мок — а is the 1st vowel, index 0
        indices = self.extract("за\u0301мок")
        assert indices == [0]

    def test_lock_second_vowel(self):
        # замо́к — о is the 2nd vowel, index 1
        indices = self.extract("замо\u0301к")
        assert indices == [1]

    def test_water_second_vowel(self):
        # вода́ — а is the 2nd vowel, index 1
        indices = self.extract("вода\u0301")
        assert indices == [1]

    def test_no_stress_returns_empty(self):
        assert self.extract("замок") == []

    def test_double_stress(self):
        # по́ми́лка — both о (idx 0) and и (idx 1)
        indices = self.extract("по\u0301ми\u0301лка")
        assert set(indices) == {0, 1}

    def test_empty_string(self):
        assert self.extract("") == []


class TestNormalizePos:
    """normalize_pos() maps Wiktionary POS strings to UPOS enum values."""

    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.normalize = parser.normalize_pos
        self.UPOS = parser.UPOS

    def test_noun(self):
        assert self.normalize("noun") == self.UPOS.NOUN

    def test_verb(self):
        assert self.normalize("verb") == self.UPOS.VERB

    def test_adjective_variants(self):
        assert self.normalize("adj") == self.UPOS.ADJ
        assert self.normalize("adjective") == self.UPOS.ADJ

    def test_unknown_maps_to_x(self):
        assert self.normalize("gibberish") == self.UPOS.X

    def test_none_returns_none(self):
        assert self.normalize(None) is None

    def test_case_insensitive(self):
        assert self.normalize("NOUN") == self.UPOS.NOUN
        assert self.normalize("Verb") == self.UPOS.VERB


class TestNormalizeTags:
    """normalize_tags() flattens and deduplicates tag lists."""

    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.normalize = parser.normalize_tags

    def test_list_of_strings(self):
        result = self.normalize(["nominative", "singular"])
        assert set(result) == {"nominative", "singular"}

    def test_deduplication(self):
        result = self.normalize(["singular", "singular"])
        assert result.count("singular") == 1

    def test_none_input(self):
        assert self.normalize(None) is None

    def test_empty_list(self):
        assert self.normalize([]) is None

    def test_string_input_wrapped(self):
        result = self.normalize("nominative")
        assert "nominative" in result


# ---------------------------------------------------------------------------
# Smoke-word integration via fixture JSONL
# ---------------------------------------------------------------------------

class TestKaikkiFixtureEntries:
    """Parse the 5 fixture entries and check expected outputs."""

    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.parser = parser

    def _parse_line(self, line: str):
        obj = json.loads(line)
        # Replicate what the parser does: extract forms and their stress indices
        forms_out = []
        for form_entry in obj.get("forms", []):
            form_str = form_entry.get("form", "")
            if not form_str or form_str in ("no-table-tags",):
                continue
            tags = form_entry.get("tags", [])
            # Skip romanization forms
            if "romanization" in tags:
                continue
            indices = self.parser.extract_stress_indices(form_str)
            base = self.parser.strip_stress(form_str)
            forms_out.append((base, indices, tags))
        return forms_out

    def test_zamok_castle_has_stress_index_0(self):
        """First замок entry (castle) has stressed first vowel: за́мок → index 0."""
        line = TEST_JSONL.read_text(encoding="utf-8").splitlines()[0]
        forms = self._parse_line(line)
        # Canonical form за́мок → а is vowel 0
        canonical = [(b, i) for b, i, t in forms if "canonical" in t]
        assert any(i == [0] for _, i in canonical), \
            f"Expected index [0] in canonical forms, got {canonical}"

    def test_zamok_lock_has_stress_index_1(self):
        """Second замок entry (lock) has stressed second vowel: замо́к → index 1."""
        lines = TEST_JSONL.read_text(encoding="utf-8").splitlines()
        line = lines[1]
        forms = self._parse_line(line)
        canonical = [(b, i) for b, i, t in forms if "canonical" in t]
        assert any(i == [1] for _, i in canonical), \
            f"Expected index [1] in canonical forms, got {canonical}"

    def test_blokha_has_stress_on_second_vowel(self):
        """блоха́ \u2014 second vowel (а) is stressed: index 1."""
        lines = TEST_JSONL.read_text(encoding="utf-8").splitlines()
        line = lines[2]
        forms = self._parse_line(line)
        canonical = [(b, i) for b, i, t in forms if "canonical" in t]
        assert any(1 in i for _, i in canonical), \
            f"Expected vowel index 1 in canonical forms, got {canonical}"
