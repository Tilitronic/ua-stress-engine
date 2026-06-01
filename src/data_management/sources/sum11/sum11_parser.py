"""
СУМ-11 (Словник Української Мови — 11 томів) Parser

Parses the digitized 11-volume Ukrainian explanatory dictionary (АН УРСР, 1970-1980)
and converts it to the unified linguistic format for merging into the master stress DB.

Source:
    https://github.com/bakustarver/ukr-dictionaries-list-opensource
    File: DiktJson-ukr-ukr_SUM-11_or_1.zip (extract to get sum11.json)
    License: Public domain (pre-1991 Ukrainian Soviet encyclopaedic work)

Input Format:
    JSON object: { "lemma_bare": "<div><b>ЛЕ́МА</b>, и, <i>ж.</i> Definition...</div>", ... }
    - Keys are unstressed bare lemma forms
    - Values are HTML with the stressed headword in <b> tags
    - Multiple bold headwords in one value = heteronyms (same spelling, different stress/meaning)
    - Superscript numbers (¹²³) distinguish homographs

Output:
    Streams (lemma, LinguisticEntry) pairs for disk-backed LMDB aggregation.
    Each distinct (stressed_form, pos) pair within an entry produces a separate WordForm.
    Entries with multiple stressed headwords (heteronyms) emit separate WordForms, each
    with its own stress_indices and main_definition, enabling downstream disambiguation.

Coverage:
    ~127K lemma entries, ~124K with stress marks, ~867 heteronym groups.
    Only lemma forms are present (no inflected paradigm), so form == lemma for all entries.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import glob
import html as html_module
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from src.data_management.transform.cache_utils import compute_parser_hash, to_serializable
from src.data_management.transform.data_unifier import LinguisticEntry, WordForm, UPOS, UDFeatKey
from src.data_management.transform.merger import LMDBExporter, LMDBExportConfig
from src.utils.normalize_apostrophe import normalize_apostrophe

logger = logging.getLogger("SUM11_Parser")

# Default path — override via UA_SUM11_JSON env var or parser config
DEFAULT_DB_PATH = "src/data_management/sources/sum11/sum11.json"

# GitHub release artefacts for automatic download
SOURCE_RELEASE_URL = (
    "https://github.com/bakustarver/ukr-dictionaries-list-opensource"
    "/releases/download/ukr-ukr_SUM-11_or_1/DiktJson-ukr-ukr_SUM-11_or_1.zip"
)
SOURCE_ZIP_INNER_PATH = (
    "Formats/DiktJson/ukr-ukr_SUM-11_or_1/ukr-ukr_SUM-11_or_1.json"
)
# Expected SHA-256 of the canonical sum11.json (set to None to skip verification)
SOURCE_EXPECTED_SHA256: Optional[str] = None

# ---------------------------------------------------------------------------
# Source-level utilities
# ---------------------------------------------------------------------------


def compute_source_hash(path: str) -> str:
    """Return the SHA-256 hex digest of the sum11 source JSON file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_sum11(dest_path: str, show_progress: bool = True) -> None:
    """
    Download the DiktJson-ukr-ukr_SUM-11_or_1.zip release and extract
    sum11.json to *dest_path*.

    Verifies SHA-256 against SOURCE_EXPECTED_SHA256 when it is set.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SUM11] Downloading from {SOURCE_RELEASE_URL} …")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if show_progress:
            def _progress(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(downloaded / total_size * 100, 100.0)
                    print(f"\r[SUM11] Downloading … {pct:.1f}%", end="", flush=True)
            urllib.request.urlretrieve(SOURCE_RELEASE_URL, tmp_path, _progress)
            print()  # newline after inline progress
        else:
            urllib.request.urlretrieve(SOURCE_RELEASE_URL, tmp_path)

        logger.info(f"[SUM11] Extracting {SOURCE_ZIP_INNER_PATH} …")
        with zipfile.ZipFile(tmp_path) as zf:
            with zf.open(SOURCE_ZIP_INNER_PATH) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        if SOURCE_EXPECTED_SHA256 is not None:
            actual = compute_source_hash(dest_path)
            if actual != SOURCE_EXPECTED_SHA256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {dest_path}: "
                    f"expected {SOURCE_EXPECTED_SHA256}, got {actual}"
                )
        logger.info(f"[SUM11] Saved to {dest_path}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


COMBINING_ACUTE = "\u0301"
UKRAINIAN_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")

# ---------------------------------------------------------------------------
# POS + morphological feature + style-tag parsing
#
# СУМ-11 italic blocks (<i>…</i>) carry three kinds of information:
#   1. Part of speech (ч., ж., прикм., дієсл., …)
#   2. Grammatical features (недок., перех., безос., …)
#   3. Style / register labels (розм., спец., поет., …)
#
# The *first* POS token maps to UPOS.  All other tokens are inspected for
# morphological features or style labels.  Unrecognised tokens are silently
# ignored — we never crash on unknown abbreviations.
# ---------------------------------------------------------------------------

# ── 1. POS abbreviations → UPOS ─────────────────────────────────────────────
# СУМ-11 encodes gender as the primary POS marker for nouns (ч./ж./с.).
# Multiple alternate spellings are listed where OCR variance is known.
_POS_MAP: Dict[str, UPOS] = {
    # Noun — СУМ-11 uses grammatical gender as the primary POS tag for nouns.
    # "ч.р"/"ж.р"/"с.р" forms are NOT listed because the dot splits them into
    # ["ч","р"] etc. — "ч", "ж", "с" tokens already cover everything.
    "ч":         UPOS.NOUN,   # чоловічий рід — masculine
    "ж":         UPOS.NOUN,   # жіночий рід — feminine
    "с":         UPOS.NOUN,   # середній рід — neuter
    "мн":        UPOS.NOUN,   # pluralia tantum (only-plural nouns)
    "одн":       UPOS.NOUN,   # singularia tantum (only-singular nouns)
    "імен":      UPOS.NOUN,   # explicit "іменник" label (rare)
    # Adjective
    "прикм":     UPOS.ADJ,
    # Numeral
    "числ":      UPOS.NUM,
    # Pronoun
    "займ":      UPOS.PRON,
    # Adverb / predicative
    "присл":     UPOS.ADV,
    "присудк":   UPOS.ADV,    # присудкове слово — UD has no dedicated tag, use ADV
    "предик":    UPOS.ADV,
    # Verb and verb-derived forms
    "дієсл":     UPOS.VERB,
    "недок":     UPOS.VERB,   # imperfective (primary POS label in СУМ-11)
    "dok":       UPOS.VERB,   # OCR artefact for "dok." (доконаний)
    "doc":       UPOS.VERB,   # OCR artefact
    "док":       UPOS.VERB,   # доконаний вид
    "дієприкм":  UPOS.ADJ,    # дієприкметник — UD: ADJ (with VerbForm=Part in feats)
    "дієприсл":  UPOS.VERB,   # дієприслівник — UD: VERB (with VerbForm=Conv)
    # Function words
    "прийм":     UPOS.ADP,
    "спол":      UPOS.CCONJ,  # СУМ-11 doesn't split coordinating/subordinating
    "підр":      UPOS.SCONJ,  # підрядний сполучник
    "частк":     UPOS.PART,
    "part":      UPOS.PART,   # OCR artefact
    "вст":       UPOS.PART,   # вставне слово — closest UD tag is PART
    "виг":       UPOS.INTJ,
    "вигук":     UPOS.INTJ,
}

# ── 2. Tokens → morphological feature (key, value) pairs ────────────────────
# A single <i> block may contribute multiple features; all matching tokens
# are applied.  Keys must be valid UD feature names (see UDFeatKey enum).
_FEATS_FROM_ABBR: Dict[str, Dict[str, str]] = {
    # Gender — UD Gender feature values per https://universaldependencies.org/u/feat/Gender.html
    "ч":         {"Gender": "Masc"},
    "ж":         {"Gender": "Fem"},
    "с":         {"Gender": "Neut"},
    # Number — UD Number feature per https://universaldependencies.org/u/feat/Number.html
    "мн":        {"Number": "Plur"},
    "одн":       {"Number": "Sing"},
    # Aspect
    "недок":     {"Aspect": "Imp"},
    "dok":       {"Aspect": "Perf"},
    "doc":       {"Aspect": "Perf"},
    "док":       {"Aspect": "Perf"},
    # Transitivity stored as Polarity marker is wrong; UD has no Transitivity.
    # We store it as a style tag instead (see _STYLE_TAG_MAP below).
    # VerbForm
    "дієприкм":  {"VerbForm": "Part"},
    "дієприсл":  {"VerbForm": "Conv"},
    # Voice
    "зворот":    {"Reflex": "Yes"},   # зворотне дієслово (reflexive)
    # Animacy — only when the source explicitly marks it
    "ос":        {"Animacy": "Anim"},  # особа → animate
    "неос":      {"Animacy": "Inan"},
    # Impersonal usage → PronType is not quite right, but Polarity/Person have
    # no clean mapping. We record it as a style tag only.
}

# ── 3. Style / register / domain labels ─────────────────────────────────────
# These go into WordForm.tags (a free-form list) rather than feats.
# Values are lowercase English labels compatible with UD/Kaikki conventions.
_STYLE_TAG_MAP: Dict[str, str] = {
    # ── Register / style labels ───────────────────────────────────────────────
    # Tag values follow Kaikki/Wiktionary conventions (lowercase English).
    "розм":     "colloquial",    # розмовне — colloquial register (high frequency: ~19k)
    "книжн":    "literary",      # книжне — literary/bookish register
    "поет":     "poetic",        # поетичне — poetic usage
    # "нар.-поет." splits into ["нар", "поет"] after _ABBR_SPLIT_RE (hyphen is splitter).
    # Each token is looked up independently → "folk" + "poetic".
    "нар":      "folk",          # народне — folk/popular usage
    # ── Domain / specialty ───────────────────────────────────────────────────
    "спец":     "specialist",    # спеціальне — specialist/technical term (~3k)
    "архт":     "architecture",  # архітектурний — architecture domain
    # ── Temporal / geographic labels ─────────────────────────────────────────
    "заст":     "archaic",       # застаріле — archaic/obsolete (~5.2k)
    "арх":      "archaic",       # архаїзм — archaic (alternate abbreviation)
    "іст":      "historical",    # історичне — historical term
    "діал":     "dialectal",     # діалектне — dialectal (~6k)
    "обл":      "regional",      # обласне — regional variant
    # ── Frequency ────────────────────────────────────────────────────────────
    "рідко":    "rare",          # рідко вживане — rarely used (~9k)
    # ── Connotation ──────────────────────────────────────────────────────────
    "пестл":    "diminutive",    # пестливе — diminutive/endearing
    "зневажл":  "pejorative",    # зневажливе — pejorative
    "ірон":     "ironic",        # іронічне — ironic
    # ── Semantic function ─────────────────────────────────────────────────────
    "перен":    "figurative",    # переносне значення — figurative sense (~10k)
    "знач":     "secondary-meaning",  # "в знач." — used in secondary/shifted meaning
    # ── Morphosyntactic notes (no UD feat equivalent) ─────────────────────────
    # These are stored as tags because UD has no standard features for them.
    "безос":    "impersonal",    # безособове — impersonal use
    "перех":    "transitive",    # перехідне — transitive (~13.7k)
    "неперех":  "intransitive",  # неперехідне — intransitive
    "збірн":    "collective",    # збірне — collective noun
    "незм":     "indeclinable",  # незмінюване — indeclinable word
}

# Split abbreviation blocks on: dots, commas, semicolons, slashes, hyphens, whitespace.
# Hyphens are included so "нар.-поет." splits into ["нар", "поет"] rather than a
# compound token that would never match any map key.
_ABBR_SPLIT_RE = re.compile(r"[.,;/\-\s]+")


def _parse_pos_and_feats(abbr_block: str) -> Tuple[UPOS, Dict[UDFeatKey, str], List[str]]:
    """
    Parse a full СУМ-11 italic abbreviation block and return
    ``(UPOS, feats_dict, style_tags)``.

    * **UPOS** — determined by the first token found in ``_POS_MAP``.
    * **feats_dict** — accumulated from *all* tokens in ``_FEATS_FROM_ABBR``;
      keys are ``UDFeatKey`` enum instances.
    * **style_tags** — list of lowercase English register/domain strings from
      ``_STYLE_TAG_MAP`` (stored in ``WordForm.tags``).

    Examples
    --------
    ``"недок., перех."``    → (VERB, {Aspect: Imp, VerbForm: Inf}, ["transitive"])
    ``"ж."``                → (NOUN, {Gender: Fem},                [])
    ``"прикм., розм."``     → (ADJ,  {},                           ["colloquial"])
    ``"дієприкм."``         → (ADJ,  {VerbForm: Part},             [])
    """
    tokens = [t.lower() for t in _ABBR_SPLIT_RE.split(abbr_block.strip()) if t]

    pos = UPOS.X
    raw_feats: Dict[str, str] = {}
    tags: List[str] = []

    for tok in tokens:
        # Determine POS from the first matching token
        if pos is UPOS.X and tok in _POS_MAP:
            pos = _POS_MAP[tok]
        # Accumulate morphological features
        if tok in _FEATS_FROM_ABBR:
            raw_feats.update(_FEATS_FROM_ABBR[tok])
        # Accumulate style tags (dedup)
        if tok in _STYLE_TAG_MAP:
            label = _STYLE_TAG_MAP[tok]
            if label not in tags:
                tags.append(label)

    # Verb lemmas in СУМ-11 are always infinitives unless a more specific
    # VerbForm was already set (Part for participles, Conv for converbs).
    if pos is UPOS.VERB and "VerbForm" not in raw_feats:
        raw_feats["VerbForm"] = "Inf"

    # Convert string keys → UDFeatKey enum; drop unrecognised keys gracefully
    typed_feats: Dict[UDFeatKey, str] = {}
    for k, v in raw_feats.items():
        try:
            typed_feats[UDFeatKey(k)] = v
        except ValueError:
            pass

    return pos, typed_feats, tags


def _parse_pos_abbr(abbr: str) -> UPOS:
    """Backwards-compatible shim that returns only UPOS."""
    pos, _, _ = _parse_pos_and_feats(abbr)
    return pos


# ---------------------------------------------------------------------------
# Regex helpers (defined after _parse_pos_and_feats to keep related code together)
# ---------------------------------------------------------------------------

# Matches the first italic abbreviation block after a headword
_ITALIC_RE = re.compile(r"<i>([^<]{1,60}?)</i>")
# Matches bold headword blocks: captures the full raw bold text
_BOLD_RE = re.compile(r"<b>([^<]+)</b>")
# Strips all HTML tags
_TAG_RE = re.compile(r"<[^>]+>")
# Superscript digits (homograph markers)
_SUPERSCRIPT_RE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹]")


def _strip_html(text: str) -> str:
    """Strip all HTML tags and decode entities."""
    return html_module.unescape(_TAG_RE.sub("", text)).strip()


def _extract_stress_indices(word: str) -> List[int]:
    """
    Return 0-based vowel indices of all stressed vowels.
    Handles combining acute (U+0301) after the vowel.
    """
    indices: List[int] = []
    vowel_idx = 0
    i = 0
    while i < len(word):
        c = word[i]
        if c in UKRAINIAN_VOWELS:
            if i + 1 < len(word) and word[i + 1] == COMBINING_ACUTE:
                indices.append(vowel_idx)
                i += 2  # skip vowel + combining acute
            else:
                i += 1
            vowel_idx += 1
        elif c == COMBINING_ACUTE:
            # Stray combining acute (shouldn't happen, but skip gracefully)
            i += 1
        else:
            i += 1
    return indices


def _bare(word: str) -> str:
    """Remove stress marks and superscripts, normalize apostrophe, lowercase."""
    clean = word.replace(COMBINING_ACUTE, "")
    clean = _SUPERSCRIPT_RE.sub("", clean)
    return normalize_apostrophe(clean).lower()


def _normalize_lemma(key: str) -> str:
    """Normalize the JSON key to a canonical bare lemma."""
    return normalize_apostrophe(key.strip().lower())


def _split_into_headword_blocks(html_val: str) -> List[str]:
    """
    Split the HTML value into per-headword blocks.
    Each block starts at a <b> tag and extends to the next one (or end of string).
    This allows extracting definition text per heteronym.
    """
    parts = re.split(r"(?=<b>)", html_val)
    return [p for p in parts if p.strip()]


def _parse_entry(key: str, html_val: str) -> List[WordForm]:
    """
    Parse a single СУМ-11 entry into one or more WordForms.

    Each <b>HEADWORD</b> occurrence with a stress mark produces one WordForm.
    Multiple headwords in one entry = heteronyms (different stress → different meaning).
    Cross-references (<b>WORD</b> <i>div.</i> ...) are skipped when the bare
    form of the cross-reference target differs from the JSON key.
    """
    lemma = _normalize_lemma(key)
    forms: List[WordForm] = []

    blocks = _split_into_headword_blocks(html_val)
    seen: set = set()  # (stressed_lower, pos) → deduplicate

    for block in blocks:
        bold_match = _BOLD_RE.search(block)
        if not bold_match:
            continue
        raw_bold = html_module.unescape(bold_match.group(1))
        # Remove superscript homograph markers before stress extraction
        raw_bold_clean = _SUPERSCRIPT_RE.sub("", raw_bold)

        # Only process headwords that have a stress mark
        if COMBINING_ACUTE not in raw_bold_clean:
            continue

        stressed_lower = raw_bold_clean.lower()
        bare_form = _bare(raw_bold_clean)

        # The form must match the lemma key (cross-references to other lemmas are skipped)
        if bare_form != lemma:
            continue

        stress_indices = _extract_stress_indices(stressed_lower)
        if not stress_indices:
            continue

        # Extract POS + morphological features from the first <i>…</i> after the bold
        after_bold = block[bold_match.end():]
        italic_match = _ITALIC_RE.search(after_bold[:200])  # limit search window
        if italic_match:
            pos, feats, style_tags = _parse_pos_and_feats(italic_match.group(1))
        else:
            pos, feats, style_tags = UPOS.X, {}, []

        dedup_key = (stressed_lower, pos)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Extract definition: plain text of the whole block minus the bold headword itself
        definition_text = _strip_html(block)
        # Remove the raw bold text from the start of the plain definition
        bare_bold = _bare(raw_bold)
        if definition_text.lower().startswith(bare_bold):
            definition_text = definition_text[len(bare_bold):].lstrip(" ,.:;—–-")
        main_definition = definition_text if definition_text else None

        forms.append(
            WordForm(
                form=lemma,
                stress_indices=stress_indices,
                pos=pos,
                feats=feats,
                lemma=lemma,
                main_definition=main_definition,
                alt_definitions=None,
                tags=style_tags if style_tags else None,
                examples=[],
            )
        )

    return forms


def parse_sum11_to_unified_dict(
    input_path: str,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[Tuple[str, LinguisticEntry]]:
    """
    Generator that yields (lemma, LinguisticEntry) pairs from the СУМ-11 JSON.

    Handles duplicate JSON keys (heteronyms stored under the same bare key)
    by using object_pairs_hook to collect all pairs including duplicates.
    """
    pairs: List[Tuple[str, str]] = []

    def _pairs_hook(lst: List[Tuple[str, str]]) -> dict:
        pairs.extend(lst)
        return {}

    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    json.loads(raw, object_pairs_hook=_pairs_hook)

    # Remove metadata keys (start with ##)
    data_pairs = [(k, v) for k, v in pairs if not k.startswith("##")]
    total = len(data_pairs)

    if progress_callback:
        progress_callback(0, total)

    iterator = tqdm(data_pairs, desc="[SUM11→LMDB]", disable=not show_progress)

    # Group by normalized lemma (handles true duplicate keys in JSON)
    grouped: Dict[str, List[WordForm]] = {}
    for idx, (key, html_val) in enumerate(iterator):
        try:
            lemma = _normalize_lemma(key)
            word_forms = _parse_entry(key, html_val)
            if not word_forms:
                continue
            grouped.setdefault(lemma, []).extend(word_forms)
        except Exception as e:
            logger.warning(f"Failed to parse entry '{key}': {e}")
        if progress_callback and (idx % 1000 == 0 or idx == total - 1):
            progress_callback(idx + 1, total)

    for lemma, forms in grouped.items():
        # Deduplicate forms by (stress_indices, pos)
        seen: set = set()
        unique_forms: List[WordForm] = []
        for wf in forms:
            key_tuple = (tuple(wf.stress_indices), wf.pos)
            if key_tuple not in seen:
                seen.add(key_tuple)
                unique_forms.append(wf)

        stress_patterns = sorted(
            {tuple(wf.stress_indices) for wf in unique_forms if wf.stress_indices}
        )
        possible_stress_indices = [list(p) for p in stress_patterns]

        entry = LinguisticEntry(
            word=lemma,
            forms=unique_forms,
            possible_stress_indices=possible_stress_indices,
        )
        yield lemma, entry


def stream_sum11_to_lmdb(
    input_path: str,
    lmdb_path: str,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Stream СУМ-11 entries to LMDB."""
    os.makedirs(lmdb_path, exist_ok=True)
    config_obj = LMDBExportConfig(db_path=Path(lmdb_path), overwrite=True)
    exporter = LMDBExporter(config_obj)

    def _serializable_iter():
        for lemma, entry in parse_sum11_to_unified_dict(
            input_path, show_progress=show_progress, progress_callback=progress_callback
        ):
            yield lemma, to_serializable(entry)

    exporter.export_streaming(_serializable_iter(), show_progress=False)
    logger.info(f"[SUM11→LMDB] Finished export at {lmdb_path}")


def run_sum11_parser(
    progress_callback: Optional[Callable[[int, int], None]] = None,
    config: Optional[Dict] = None,
    auto_download: bool = True,
) -> Tuple[str, Dict]:
    """
    Entry point called by parsing_merging_service.
    Returns (lmdb_path, stats).

    If the source JSON is missing and *auto_download* is True (default),
    the release ZIP is fetched from GitHub and the file is extracted
    automatically.  Pass auto_download=False to disable network access
    and receive a FileNotFoundError instead.
    """
    if config is None:
        db_path = os.environ.get("UA_SUM11_JSON", DEFAULT_DB_PATH)
        parser_path = "src/data_management/sources/sum11/sum11_parser.py"
    else:
        db_path = config["db_path"]
        parser_path = config["parser_path"]

    if not os.path.exists(db_path):
        if auto_download:
            logger.info(f"[SUM11] Source not found at '{db_path}'. Starting automatic download …")
            download_sum11(db_path, show_progress=True)
        else:
            raise FileNotFoundError(
                f"СУМ-11 JSON not found at '{db_path}'. "
                "Download DiktJson-ukr-ukr_SUM-11_or_1.zip from "
                f"{SOURCE_RELEASE_URL} "
                "and extract sum11.json there, or set UA_SUM11_JSON env var."
            )

    prefix = "SUM11"
    cache_key = compute_parser_hash(parser_path, db_path)
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "transform", "cache")
    )
    lmdb_dir = os.path.abspath(os.path.join(cache_dir, f"{prefix}_{cache_key}_lmdb"))

    if os.path.exists(lmdb_dir) and os.listdir(lmdb_dir):
        logger.info(f"[CACHE] Using LMDB cache for SUM11 at {lmdb_dir}")
        return lmdb_dir, {"cache_used": True, "lmdb_path": lmdb_dir}

    logger.info(f"[SUM11→LMDB] Streaming СУМ-11 to LMDB at {lmdb_dir}")
    stream_sum11_to_lmdb(
        input_path=db_path,
        lmdb_path=lmdb_dir,
        show_progress=True,
        progress_callback=progress_callback,
    )

    # Clean up stale caches for this prefix
    for d in glob.glob(os.path.join(cache_dir, f"{prefix}_*_lmdb")):
        if os.path.basename(d) != os.path.basename(lmdb_dir):
            try:
                shutil.rmtree(d)
                logger.info(f"[SUM11→LMDB] Deleted old cache: {d}")
            except Exception as e:
                logger.warning(f"[SUM11→LMDB] Failed to delete old cache {d}: {e}")

    return lmdb_dir, {"cache_used": False, "lmdb_path": lmdb_dir}
