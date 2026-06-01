"""
Tests for src/data_management/sources/sum11/sum11_parser.py

Tests use sum11.test.json (8 synthetic entries mirroring the DiktJson format)
and do not require the full 127 K-entry JSON file or any network access.

Test categories
---------------
1. Fixture sanity     — the fixture file exists and is valid JSON
2. _extract_stress_indices — 0-based vowel index extraction from combining-acute words
3. _bare              — stress-mark removal, superscript stripping, normalisation
4. _parse_pos_abbr    — Ukrainian abbreviation → UPOS mapping
5. _split_into_headword_blocks — HTML block splitting for heteronyms
6. _parse_entry       — per-entry WordForm extraction (simple, heteronym, cross-ref, …)
7. parse_sum11_to_unified_dict — integration against the fixture file
8. compute_source_hash  — deterministic SHA-256 of a file
9. download_sum11     — auto-download path (network mocked)
10. run_sum11_parser  — FileNotFoundError without auto-download; download triggered with it
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[4]
SUM11_SRC = ROOT / "src" / "data_management" / "sources" / "sum11"
FIXTURE_JSON = SUM11_SRC / "sum11.test.json"

COMBINING_ACUTE = "\u0301"


# ---------------------------------------------------------------------------
# Lazy module loader (same pattern as test_kaikki_parser.py)
# Avoids running heavy import side-effects at collection time.
# ---------------------------------------------------------------------------
import importlib.util
import sys


def _load_parser():
    spec = importlib.util.spec_from_file_location(
        "sum11_parser", SUM11_SRC / "sum11_parser.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parser():
    return _load_parser()


# ---------------------------------------------------------------------------
# 1. Fixture sanity
# ---------------------------------------------------------------------------

def test_fixture_exists():
    assert FIXTURE_JSON.exists(), f"Missing fixture: {FIXTURE_JSON}"


def test_fixture_is_valid_json():
    data = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    # At least the entries we care about
    for word in ("мама", "вода", "замок", "читати", "і", "копійка"):
        assert word in data, f"Fixture missing entry '{word}'"


def test_fixture_contains_stress_marks():
    # The JSON file stores U+0301 as a \u0301 escape sequence, so we must
    # parse the JSON to get the actual Unicode character.
    data = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
    combined = " ".join(v for v in data.values() if isinstance(v, str))
    assert COMBINING_ACUTE in combined, "Fixture values have no combining-acute stress marks"


# ---------------------------------------------------------------------------
# 2. _extract_stress_indices
# ---------------------------------------------------------------------------

class TestExtractStressIndices:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.extract = parser._extract_stress_indices

    def test_first_vowel_stressed(self):
        # МА́МА — А is the 1st vowel (index 0)
        assert self.extract("ма\u0301ма") == [0]

    def test_second_vowel_stressed(self):
        # ВОДА́ — А is the 2nd vowel (index 1)
        assert self.extract("вода\u0301") == [1]

    def test_heteronym_castle(self):
        # ЗА́МОК — А at index 0
        assert self.extract("за\u0301мок") == [0]

    def test_heteronym_lock(self):
        # ЗАМО́К — О at index 1
        assert self.extract("замо\u0301к") == [1]

    def test_middle_vowel(self):
        # ЧИТА́ТИ — А is the 2nd vowel (index 1)
        assert self.extract("чита\u0301ти") == [1]

    def test_no_stress_returns_empty(self):
        assert self.extract("замок") == []

    def test_empty_string(self):
        assert self.extract("") == []

    def test_stray_acute_skipped(self):
        # Combining acute with no preceding vowel — must not crash
        result = self.extract("\u0301")
        assert result == []


# ---------------------------------------------------------------------------
# 3. _bare
# ---------------------------------------------------------------------------

class TestBare:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.bare = parser._bare

    def test_removes_combining_acute(self):
        assert self.bare("за\u0301мок") == "замок"

    def test_removes_superscript_digits(self):
        assert self.bare("замок\u00b9") == "замок"
        assert self.bare("замок\u00b2") == "замок"

    def test_lowercases(self):
        assert self.bare("МАМА") == "мама"

    def test_full_pipeline(self):
        # Uppercase headword with stress and superscript
        assert self.bare("ЗА\u0301МОК\u00b9") == "замок"

    def test_empty_string(self):
        assert self.bare("") == ""


# ---------------------------------------------------------------------------
# 4. _parse_pos_abbr  (legacy shim — returns UPOS only)
# ---------------------------------------------------------------------------

class TestParsePosAbbr:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.parse = parser._parse_pos_abbr
        self.UPOS = parser.UPOS

    def test_feminine_noun(self):
        assert self.parse("ж.") == self.UPOS.NOUN

    def test_masculine_noun(self):
        assert self.parse("ч.") == self.UPOS.NOUN

    def test_neuter_noun(self):
        assert self.parse("с.") == self.UPOS.NOUN

    def test_adjective(self):
        assert self.parse("прикм.") == self.UPOS.ADJ

    def test_adverb(self):
        assert self.parse("присл.") == self.UPOS.ADV

    def test_imperfective_verb(self):
        assert self.parse("недок.") == self.UPOS.VERB

    def test_conjunction(self):
        assert self.parse("спол.") == self.UPOS.CCONJ

    def test_preposition(self):
        assert self.parse("прийм.") == self.UPOS.ADP

    def test_interjection(self):
        assert self.parse("виг.") == self.UPOS.INTJ

    def test_numeral(self):
        assert self.parse("числ.") == self.UPOS.NUM

    def test_unknown_maps_to_x(self):
        assert self.parse("xyz.") == self.UPOS.X

    def test_compound_abbr_uses_first_token(self):
        # "недок., перех." — take first token "недок"
        assert self.parse("недок., перех.") == self.UPOS.VERB


# ---------------------------------------------------------------------------
# 4b. _parse_pos_and_feats  (full UD output: UPOS, feats, style_tags)
# ---------------------------------------------------------------------------

class TestParsePosAndFeats:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.parse = parser._parse_pos_and_feats
        self.UPOS = parser.UPOS
        self.UDFeatKey = parser.UDFeatKey

    def _key(self, name: str):
        return self.UDFeatKey(name)

    # ── POS correctness ──────────────────────────────────────────────────────

    def test_masculine_noun_pos(self):
        pos, _, _ = self.parse("ч.")
        assert pos == self.UPOS.NOUN

    def test_feminine_noun_pos(self):
        pos, _, _ = self.parse("ж.")
        assert pos == self.UPOS.NOUN

    def test_neuter_noun_pos(self):
        pos, _, _ = self.parse("с.")
        assert pos == self.UPOS.NOUN

    def test_adjective_pos(self):
        pos, _, _ = self.parse("прикм.")
        assert pos == self.UPOS.ADJ

    def test_adverb_pos(self):
        pos, _, _ = self.parse("присл.")
        assert pos == self.UPOS.ADV

    def test_imperfective_verb_pos(self):
        pos, _, _ = self.parse("недок.")
        assert pos == self.UPOS.VERB

    def test_perfective_verb_pos(self):
        pos, _, _ = self.parse("док.")
        assert pos == self.UPOS.VERB

    def test_perfective_ocr_artefact(self):
        pos, feats, _ = self.parse("doc.")
        assert pos == self.UPOS.VERB
        assert feats.get(self._key("Aspect")) == "Perf"

    def test_participle_pos(self):
        pos, feats, _ = self.parse("дієприкм.")
        assert pos == self.UPOS.ADJ
        assert feats.get(self._key("VerbForm")) == "Part"

    def test_converb_pos(self):
        pos, feats, _ = self.parse("дієприсл.")
        assert pos == self.UPOS.VERB
        assert feats.get(self._key("VerbForm")) == "Conv"

    def test_conjunction_pos(self):
        pos, _, _ = self.parse("спол.")
        assert pos == self.UPOS.CCONJ

    def test_preposition_pos(self):
        pos, _, _ = self.parse("прийм.")
        assert pos == self.UPOS.ADP

    def test_pronoun_pos(self):
        pos, _, _ = self.parse("займ.")
        assert pos == self.UPOS.PRON

    def test_numeral_pos(self):
        pos, _, _ = self.parse("числ.")
        assert pos == self.UPOS.NUM

    def test_interjection_pos(self):
        pos, _, _ = self.parse("виг.")
        assert pos == self.UPOS.INTJ

    def test_particle_pos(self):
        pos, _, _ = self.parse("частк.")
        assert pos == self.UPOS.PART

    def test_unknown_maps_to_x(self):
        pos, feats, tags = self.parse("xyz.")
        assert pos == self.UPOS.X
        assert isinstance(feats, dict)
        assert isinstance(tags, list)

    # ── Gender features ──────────────────────────────────────────────────────

    def test_masculine_gender_feat(self):
        _, feats, _ = self.parse("ч.")
        assert feats.get(self._key("Gender")) == "Masc"

    def test_feminine_gender_feat(self):
        _, feats, _ = self.parse("ж.")
        assert feats.get(self._key("Gender")) == "Fem"

    def test_neuter_gender_feat(self):
        _, feats, _ = self.parse("с.")
        assert feats.get(self._key("Gender")) == "Neut"

    def test_neuter_short_form(self):
        # "с." is the standard neuter abbreviation; "ср." splits into ["с", "р"]
        # so "с" is matched for both POS and Gender.
        _, feats, _ = self.parse("с.")
        assert feats.get(self._key("Gender")) == "Neut"

    # ── Number features ──────────────────────────────────────────────────────

    def test_plural_number_feat(self):
        _, feats, _ = self.parse("мн.")
        assert feats.get(self._key("Number")) == "Plur"

    # ── Aspect features ──────────────────────────────────────────────────────

    def test_imperfective_aspect_feat(self):
        _, feats, _ = self.parse("недок.")
        assert feats.get(self._key("Aspect")) == "Imp"

    def test_perfective_aspect_feat(self):
        _, feats, _ = self.parse("dok.")
        assert feats.get(self._key("Aspect")) == "Perf"

    def test_perfective_cyrillic(self):
        _, feats, _ = self.parse("док.")
        assert feats.get(self._key("Aspect")) == "Perf"

    # ── VerbForm auto-injection ───────────────────────────────────────────────

    def test_verb_gets_inf_verbform(self):
        _, feats, _ = self.parse("дієсл.")
        assert feats.get(self._key("VerbForm")) == "Inf"

    def test_imperfective_verb_gets_inf_verbform(self):
        _, feats, _ = self.parse("недок.")
        assert feats.get(self._key("VerbForm")) == "Inf"

    def test_participle_does_not_get_inf(self):
        _, feats, _ = self.parse("дієприкм.")
        assert feats.get(self._key("VerbForm")) == "Part"

    def test_converb_does_not_get_inf(self):
        _, feats, _ = self.parse("дієприсл.")
        assert feats.get(self._key("VerbForm")) == "Conv"

    # ── Reflexive / animacy ───────────────────────────────────────────────────

    def test_reflexive_feat(self):
        _, feats, _ = self.parse("недок., зворот.")
        assert feats.get(self._key("Reflex")) == "Yes"

    # ── Compound abbreviation — multiple features + tags ─────────────────────

    def test_compound_abbr_aspect_plus_transitivity(self):
        pos, feats, tags = self.parse("недок., перех.")
        assert pos == self.UPOS.VERB
        assert feats.get(self._key("Aspect")) == "Imp"
        assert feats.get(self._key("VerbForm")) == "Inf"
        assert "transitive" in tags

    def test_compound_intransitive(self):
        _, _, tags = self.parse("недок., неперех.")
        assert "intransitive" in tags

    # ── Style / register tags ─────────────────────────────────────────────────

    def test_colloquial_tag(self):
        _, _, tags = self.parse("розм.")
        assert "colloquial" in tags

    def test_specialist_tag(self):
        _, _, tags = self.parse("спец.")
        assert "specialist" in tags

    def test_archaic_tag(self):
        _, _, tags = self.parse("заст.")
        assert "archaic" in tags

    def test_historical_tag(self):
        _, _, tags = self.parse("іст.")
        assert "historical" in tags

    def test_dialectal_tag(self):
        _, _, tags = self.parse("діал.")
        assert "dialectal" in tags

    def test_poetic_tag(self):
        _, _, tags = self.parse("поет.")
        assert "poetic" in tags

    def test_figurative_tag(self):
        _, _, tags = self.parse("перен.")
        assert "figurative" in tags

    def test_rare_tag(self):
        _, _, tags = self.parse("рідко")
        assert "rare" in tags

    def test_impersonal_tag(self):
        _, _, tags = self.parse("безос.")
        assert "impersonal" in tags

    def test_specialist_verb_compound(self):
        # "недок., перех. спец." → verb + transitive tag + specialist tag
        pos, feats, tags = self.parse("недок., перех. спец.")
        assert pos == self.UPOS.VERB
        assert "transitive" in tags
        assert "specialist" in tags

    def test_no_duplicate_tags(self):
        _, _, tags = self.parse("розм. розм.")
        assert tags.count("colloquial") == 1

    # ── Return type invariants ────────────────────────────────────────────────

    def test_feats_keys_are_enum_instances(self):
        _, feats, _ = self.parse("недок., перех., ж.")
        for k in feats:
            assert isinstance(k, self.UDFeatKey)

    def test_tags_is_list(self):
        _, _, tags = self.parse("розм.")
        assert isinstance(tags, list)

    def test_no_style_tags_returns_empty_list(self):
        _, _, tags = self.parse("ж.")
        assert tags == []


# ---------------------------------------------------------------------------
# 5. _split_into_headword_blocks
# ---------------------------------------------------------------------------

class TestSplitHeadwordBlocks:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.split = parser._split_into_headword_blocks

    def test_single_block(self):
        # _split_into_headword_blocks splits on <b> lookahead via re.split(r"(?=<b>)", ...).
        # The "<div>" prefix before the first <b> becomes its own non-empty block,
        # so a single headword yields 2 parts: ["<div>", "<b>...</b>..."].
        html = "<div><b>МА\u0301МА</b>, и, <i>ж.</i> Мати.</div>"
        blocks = self.split(html)
        assert len(blocks) == 2
        assert "<b>МА\u0301МА</b>" in blocks[1]

    def test_two_blocks_for_heteronym(self):
        # Two <b> headwords → 3 parts: ["<div>", "<b>block1</b>...", "<b>block2</b>..."]
        html = (
            "<div><b>ЗА\u0301МОК\u00b9</b>, мку, <i>ч.</i> Будівля."
            "<b>ЗАМО\u0301К\u00b2</b>, мку, <i>ч.</i> Пристрій.</div>"
        )
        blocks = self.split(html)
        assert len(blocks) == 3
        assert "<b>ЗА\u0301МОК\u00b9</b>" in blocks[1]
        assert "<b>ЗАМО\u0301К\u00b2</b>" in blocks[2]

    def test_empty_string(self):
        assert self.split("") == []


# ---------------------------------------------------------------------------
# 6. _parse_entry
# ---------------------------------------------------------------------------

class TestParseEntry:
    @pytest.fixture(autouse=True)
    def _setup(self, parser):
        self.parse = parser._parse_entry
        self.UPOS = parser.UPOS

    def _entry(self, key: str) -> list:
        data = json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))
        return self.parse(key, data[key])

    def test_simple_noun_мама(self):
        forms = self._entry("мама")
        assert len(forms) == 1
        wf = forms[0]
        assert wf.stress_indices == [0]
        assert wf.pos == self.UPOS.NOUN
        assert wf.form == "мама"
        assert wf.lemma == "мама"

    def test_noun_вода_second_vowel(self):
        forms = self._entry("вода")
        assert len(forms) == 1
        assert forms[0].stress_indices == [1]

    def test_heteronym_замок_two_forms(self):
        forms = self._entry("замок")
        assert len(forms) == 2
        indices = {tuple(wf.stress_indices) for wf in forms}
        assert (0,) in indices, "Castle stress (index 0) missing"
        assert (1,) in indices, "Lock stress (index 1) missing"

    def test_heteronym_forms_have_definitions(self):
        forms = self._entry("замок")
        for wf in forms:
            assert wf.main_definition, f"WordForm for замок has no definition: {wf}"

    def test_verb_читати_pos(self):
        forms = self._entry("читати")
        assert len(forms) == 1
        assert forms[0].pos == self.UPOS.VERB
        assert forms[0].stress_indices == [1]

    def test_adverb_швидко_pos(self):
        forms = self._entry("швидко")
        assert len(forms) == 1
        assert forms[0].pos == self.UPOS.ADV
        assert forms[0].stress_indices == [0]

    def test_adjective_прикметник_pos(self):
        # fixture has "ч." (masculine noun), so expect NOUN + Gender=Masc
        forms = self._entry("прикметник")
        assert len(forms) == 1
        assert forms[0].pos == self.UPOS.NOUN
        assert forms[0].stress_indices == [1]

    def test_noun_мама_has_gender_feat(self):
        # "ж." → Gender=Fem
        forms = self._entry("мама")
        assert len(forms) == 1
        feats = forms[0].feats
        assert any(k.value == "Gender" for k in feats), f"Gender missing from feats: {feats}"
        gender_val = next(v for k, v in feats.items() if k.value == "Gender")
        assert gender_val == "Fem"

    def test_verb_читати_has_verbform_inf(self):
        forms = self._entry("читати")
        assert len(forms) == 1
        feats = forms[0].feats
        vf_val = next((v for k, v in feats.items() if k.value == "VerbForm"), None)
        assert vf_val == "Inf"

    def test_adverb_швидко_has_no_gender(self):
        forms = self._entry("швидко")
        assert len(forms) == 1
        feats = forms[0].feats
        assert not any(k.value == "Gender" for k in feats)

    def test_monosyllable_no_stress_returns_empty(self):
        # "і" has no stress mark → no WordForms
        forms = self._entry("і")
        assert forms == []

    def test_cross_reference_skipped(self):
        # "копійка" entry has a second <b>копі́йки</b> which is a different bare lemma
        forms = self._entry("копійка")
        assert len(forms) == 1
        assert forms[0].stress_indices == [1]

    def test_form_matches_lemma_key(self):
        for key in ("мама", "вода", "читати"):
            forms = self._entry(key)
            for wf in forms:
                assert wf.form == key
                assert wf.lemma == key


# ---------------------------------------------------------------------------
# 7. parse_sum11_to_unified_dict (integration against fixture)
# ---------------------------------------------------------------------------

class TestParseSum11ToUnifiedDict:
    @pytest.fixture(scope="class")
    def entries(self, parser):
        return dict(
            parser.parse_sum11_to_unified_dict(str(FIXTURE_JSON), show_progress=False)
        )

    def test_yields_expected_lemmas(self, entries):
        # "і" has no stress so is not emitted
        for word in ("мама", "вода", "замок", "читати", "копійка"):
            assert word in entries, f"'{word}' not in parsed entries"

    def test_monosyllable_not_emitted(self, entries):
        assert "і" not in entries

    def test_замок_has_two_possible_stress_patterns(self, entries):
        entry = entries["замок"]
        patterns = entry.possible_stress_indices
        flat = [tuple(p) for p in patterns]
        assert (0,) in flat
        assert (1,) in flat

    def test_мама_possible_stress(self, entries):
        entry = entries["мама"]
        assert entry.possible_stress_indices == [[0]]

    def test_вода_possible_stress(self, entries):
        entry = entries["вода"]
        assert entry.possible_stress_indices == [[1]]

    def test_word_field_matches_key(self, entries):
        for lemma, entry in entries.items():
            assert entry.word == lemma

    def test_metadata_key_excluded(self, entries):
        # Keys starting with ## must not appear in output
        assert all(not k.startswith("##") for k in entries)

    def test_forms_have_lemma_set(self, entries):
        for lemma, entry in entries.items():
            for wf in entry.forms:
                assert wf.lemma == lemma

    def test_progress_callback_called(self, parser):
        calls: List[tuple] = []
        list(
            parser.parse_sum11_to_unified_dict(
                str(FIXTURE_JSON),
                show_progress=False,
                progress_callback=lambda cur, total: calls.append((cur, total)),
            )
        )
        assert len(calls) > 0
        # First call is (0, total)
        assert calls[0] == (0, calls[0][1])


# ---------------------------------------------------------------------------
# 8. compute_source_hash
# ---------------------------------------------------------------------------

class TestComputeSourceHash:
    def test_returns_hex_string(self, parser):
        digest = parser.compute_source_hash(str(FIXTURE_JSON))
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic(self, parser):
        d1 = parser.compute_source_hash(str(FIXTURE_JSON))
        d2 = parser.compute_source_hash(str(FIXTURE_JSON))
        assert d1 == d2

    def test_matches_hashlib_directly(self, parser):
        h = hashlib.sha256()
        with FIXTURE_JSON.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        expected = h.hexdigest()
        assert parser.compute_source_hash(str(FIXTURE_JSON)) == expected

    def test_different_files_different_hashes(self, parser, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text('{"key": "value_a"}', encoding="utf-8")
        b.write_text('{"key": "value_b"}', encoding="utf-8")
        assert parser.compute_source_hash(str(a)) != parser.compute_source_hash(str(b))


# ---------------------------------------------------------------------------
# 9. download_sum11
# ---------------------------------------------------------------------------

def _make_fake_zip(inner_path: str, content: bytes) -> bytes:
    """Build an in-memory ZIP with one member at *inner_path*."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr(inner_path, content)
    return buf.getvalue()


class TestDownloadSum11:
    def test_download_extracts_json(self, parser, tmp_path):
        fake_json = '{"##source": "test", "test": "<b>ТЕ\u0301СТ</b>"}'.encode("utf-8")
        fake_zip = _make_fake_zip(parser.SOURCE_ZIP_INNER_PATH, fake_json)
        dest = str(tmp_path / "sum11.json")

        def fake_urlretrieve(url, filename, reporthook=None):
            Path(filename).write_bytes(fake_zip)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            parser.download_sum11(dest, show_progress=False)

        assert Path(dest).exists()
        assert Path(dest).read_bytes() == fake_json

    def test_download_uses_correct_url(self, parser, tmp_path):
        fake_json = b"{}"
        fake_zip = _make_fake_zip(parser.SOURCE_ZIP_INNER_PATH, fake_json)
        dest = str(tmp_path / "sum11.json")

        captured_urls: list = []

        def fake_urlretrieve(url, filename, reporthook=None):
            captured_urls.append(url)
            Path(filename).write_bytes(fake_zip)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            parser.download_sum11(dest, show_progress=False)

        assert captured_urls == [parser.SOURCE_RELEASE_URL]

    def test_download_creates_parent_dirs(self, parser, tmp_path):
        fake_json = b"{}"
        fake_zip = _make_fake_zip(parser.SOURCE_ZIP_INNER_PATH, fake_json)
        dest = str(tmp_path / "nested" / "dir" / "sum11.json")

        def fake_urlretrieve(url, filename, reporthook=None):
            Path(filename).write_bytes(fake_zip)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            parser.download_sum11(dest, show_progress=False)

        assert Path(dest).exists()

    def test_sha256_mismatch_raises(self, parser, tmp_path):
        fake_json = b'{"key": "bad_content"}'
        fake_zip = _make_fake_zip(parser.SOURCE_ZIP_INNER_PATH, fake_json)
        dest = str(tmp_path / "sum11.json")

        def fake_urlretrieve(url, filename, reporthook=None):
            Path(filename).write_bytes(fake_zip)

        original_sha = parser.SOURCE_EXPECTED_SHA256
        try:
            # Inject a deliberately wrong expected hash
            parser.SOURCE_EXPECTED_SHA256 = "a" * 64
            with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                    parser.download_sum11(dest, show_progress=False)
        finally:
            parser.SOURCE_EXPECTED_SHA256 = original_sha


# ---------------------------------------------------------------------------
# 10. run_sum11_parser (file-missing behaviour)
# ---------------------------------------------------------------------------

class TestRunSum11ParserMissing:
    def test_raises_without_auto_download(self, parser, tmp_path):
        missing = str(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError):
            parser.run_sum11_parser(
                config={"db_path": missing, "parser_path": "dummy"},
                auto_download=False,
            )

    def test_auto_download_triggered_when_missing(self, parser, tmp_path):
        """When file is missing and auto_download=True, download_sum11 is called."""
        missing = str(tmp_path / "sum11.json")
        fake_json = FIXTURE_JSON.read_bytes()
        fake_zip = _make_fake_zip(parser.SOURCE_ZIP_INNER_PATH, fake_json)

        def fake_urlretrieve(url, filename, reporthook=None):
            Path(filename).write_bytes(fake_zip)

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
            # compute_parser_hash tries to open parser_path from disk; mock it out.
            with patch.object(parser, "compute_parser_hash", return_value="fakehash"):
                # We don't want to build a real LMDB in unit tests — stop after download
                with patch.object(
                    parser,
                    "stream_sum11_to_lmdb",
                    side_effect=RuntimeError("stop after download"),
                ):
                    with pytest.raises(RuntimeError, match="stop after download"):
                        parser.run_sum11_parser(
                            config={"db_path": missing, "parser_path": "dummy"},
                            auto_download=True,
                        )

        # The JSON must have been written to disk before stream_sum11_to_lmdb was called
        assert Path(missing).exists()
