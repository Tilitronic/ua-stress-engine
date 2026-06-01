"""
loader.py — Extract stress data from the master SQLite database.

Produces (normalised_form, stress_primary, stress_secondary, is_variative, is_heteronym)
tuples ready to be inserted into TrieBuilder.

Stress classification:
  stress_secondary is None → unique  (one unambiguous stress)
  stress_secondary is set, is_variative → variative  (both valid simultaneously)
  stress_secondary is set, is_heteronym → heteronym  (different meanings/forms)

Also provides load_variants_from_master_db() which builds the supplementary
ua_stress.variants.json.gz — per-variant morphological data for all words with
multiple stress positions (heteronyms + variatives).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Apostrophe normalisation — same logic as src/utils/normalize_apostrophe.py
# duplicated here so the module stays self-contained with no project imports.
_CORRECT_APOSTROPHE = "\u02bc"
_WRONG_APOSTROPHES = "\u2019\u0027\u02bb\u0060\u00b4"

# Path to the curated variative-stress word list.
_VARIATIVE_LIST = (
    Path(__file__).resolve().parents[4]
    / "src/data_management/sources/ua_variative_stressed_words"
    / "ua_variative_stressed_words.txt"
)


def _norm(word: str) -> str:
    """Lowercase + normalise apostrophe + strip combining accents."""
    w = word.lower()
    for ch in _WRONG_APOSTROPHES:
        w = w.replace(ch, _CORRECT_APOSTROPHE)
    # Strip combining diacritics (U+0301 stress marks embedded in some source forms)
    w = "".join(c for c in unicodedata.normalize("NFD", w)
                if unicodedata.category(c) != "Mn")
    return w


def _load_variative_set(path: Path) -> Set[str]:
    """Load normalised lemmas from the variative word list into a set."""
    result: Set[str] = set()
    if not path.exists():
        logger.warning(f"Variative word list not found, skipping: {path}")
        return result
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            w = _norm(line)
            if w:
                result.add(w)
    logger.info(f"  {len(result):,} variative lemmas loaded")
    return result


def load_from_master_db(
    db_path: Path,
    variative_list_path: Path = _VARIATIVE_LIST,
) -> Generator[Tuple[str, int, Optional[int], bool, bool], None, None]:
    """
    Yield (normalised_form, stress, stress2, is_variative, is_heteronym) for
    every single-word entry in the master SQLite database that has stress data.

    Args:
        db_path:              Path to the master SQLite DB.
        variative_list_path:  Path to ``ua_variative_stressed_words.txt``.
                              Words in this list are marked ``is_variative=True``
                              when they have multiple stress indices; all other
                              multi-stress words are marked ``is_heteronym=True``.

    Deduplication strategy:
      - Group by normalised form (case-folded, diacritics stripped).
      - If exactly one stress index seen → unique (stress2=None).
      - If multiple indices seen and form is in variative list
          → is_variative=True, stress2 = second index.
      - If multiple indices seen and form is NOT in variative list
          → is_heteronym=True, stress2 = second index.
      - Forms with no vowels or no stress data are skipped.
    """
    variative_set = _load_variative_set(variative_list_path)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    logger.info("Querying master DB for stress data...")
    rows = con.execute(
        """
        SELECT form, stress_indices_json
        FROM   word_form
        WHERE  form NOT LIKE '% %'
          AND  stress_indices_json IS NOT NULL
          AND  stress_indices_json != '[]'
          AND  stress_indices_json != ''
        """
    ).fetchall()
    con.close()
    logger.info(f"  {len(rows):,} raw rows fetched")

    # Aggregate: normalised_form → set of all seen stress indices (across sources)
    agg: dict[str, list[int]] = {}
    skipped = 0

    for row in rows:
        raw_form: str = row["form"]
        norm_form = _norm(raw_form)

        if not norm_form:
            skipped += 1
            continue

        try:
            indices: list[int] = json.loads(row["stress_indices_json"])
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        if not indices or not isinstance(indices[0], int):
            skipped += 1
            continue

        if norm_form not in agg:
            agg[norm_form] = []
        for idx in indices:
            if idx not in agg[norm_form]:
                agg[norm_form].append(idx)

    logger.info(f"  {len(agg):,} unique normalised forms  ({skipped:,} rows skipped)")

    heteronym_count = 0
    variative_count = 0
    for norm_form, indices in agg.items():
        stress = indices[0]                   # first = most-common (source order)
        stress2: Optional[int] = indices[1] if len(indices) > 1 else None
        is_variative = stress2 is not None and norm_form in variative_set
        is_heteronym = stress2 is not None and not is_variative
        if is_variative:
            variative_count += 1
        elif is_heteronym:
            heteronym_count += 1
        yield norm_form, stress, stress2, is_variative, is_heteronym

    logger.info(f"  {variative_count:,} variative words flagged")
    logger.info(f"  {heteronym_count:,} heteronyms flagged")


# ── Supplementary per-variant data ────────────────────────────────────────────

def load_variants_from_master_db(
    db_path: Path,
    variative_list_path: Path = _VARIATIVE_LIST,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build the supplementary variants map for all words with multiple stress positions.

    Returns a dict:
        { normalised_form: [ variant, ... ] }

    Each variant:
        {
            "stress": int,           # 0-based vowel index
            "pos":    str | null,    # UPOS tag, e.g. "NOUN"
            "feats":  { key: str },  # UD morphological features
            "lemma":  str | null,
            "definition": str | null # short Wiktionary gloss
        }

    Only forms with 2+ distinct stress positions are included.
    Variants are aggregated by (stress, pos, feats_canonical) — same grammatical
    slot from multiple sources is merged into one entry (richer definition wins).
    """
    variative_set = _load_variative_set(variative_list_path)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    logger.info("Querying master DB for per-variant morphological data...")

    rows = con.execute(
        """
        SELECT
            wf.form,
            wf.stress_indices_json,
            wf.pos,
            wf.lemma,
            d.text   AS definition
        FROM word_form wf
        LEFT JOIN definition d ON d.id = wf.main_definition_id
        WHERE wf.form NOT LIKE '% %'
          AND wf.stress_indices_json IS NOT NULL
          AND wf.stress_indices_json != '[]'
          AND wf.stress_indices_json != ''
        """
    ).fetchall()

    # Load features per word_form.id
    feat_rows = con.execute(
        """
        SELECT wf.form, wf.stress_indices_json, wf.pos, f.key, f.value
        FROM word_form wf
        JOIN feature f ON f.word_form_id = wf.id
        WHERE wf.form NOT LIKE '% %'
          AND wf.stress_indices_json IS NOT NULL
          AND wf.stress_indices_json != '[]'
        """
    ).fetchall()
    con.close()

    # Build feats lookup: (norm_form, stress_json, pos) -> {key: value}
    feats_map: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for r in feat_rows:
        norm = _norm(r["form"])
        key = (norm, r["stress_indices_json"], r["pos"] or "")
        if key not in feats_map:
            feats_map[key] = {}
        feats_map[key][r["key"]] = r["value"]

    logger.info(f"  {len(rows):,} raw rows, building per-form aggregation...")

    # First pass: find forms with multiple distinct stress indices
    form_stresses: Dict[str, Set[int]] = {}
    for row in rows:
        norm = _norm(row["form"])
        try:
            indices: List[int] = json.loads(row["stress_indices_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not indices:
            continue
        if norm not in form_stresses:
            form_stresses[norm] = set()
        form_stresses[norm].update(indices)

    multi_stress_forms = {f for f, s in form_stresses.items() if len(s) > 1}
    logger.info(f"  {len(multi_stress_forms):,} forms with multiple stress positions")

    # Second pass: aggregate variants per (norm_form, stress_index, pos, feats_canonical)
    # Key: (norm_form, stress, pos, feats_canonical_json)
    # Value: variant dict (definition wins from richest source)
    VariantKey = Tuple[str, int, str, str]
    variant_map: Dict[VariantKey, Dict[str, Any]] = {}

    for row in rows:
        norm = _norm(row["form"])
        if norm not in multi_stress_forms:
            continue
        try:
            indices: List[int] = json.loads(row["stress_indices_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not indices:
            continue

        pos = row["pos"] or None
        lemma = row["lemma"] or None
        definition = row["definition"] or None
        feats = feats_map.get((norm, row["stress_indices_json"], row["pos"] or ""), {})
        feats_canonical = json.dumps(feats, sort_keys=True, ensure_ascii=False)

        for stress in indices:
            vkey: VariantKey = (norm, stress, pos or "", feats_canonical)
            if vkey not in variant_map:
                variant_map[vkey] = {
                    "stress": stress,
                    "pos": pos,
                    "feats": feats,
                    "lemma": lemma,
                    "definition": definition,
                }
            else:
                # Enrich: prefer non-None definition and lemma
                existing = variant_map[vkey]
                if existing["definition"] is None and definition is not None:
                    existing["definition"] = definition
                if existing["lemma"] is None and lemma is not None:
                    existing["lemma"] = lemma

    # Third pass: group into output dict, order variants by stress index
    result: Dict[str, List[Dict[str, Any]]] = {}
    for (norm, stress, pos, feats_canonical), variant in variant_map.items():
        if norm not in result:
            result[norm] = []
        result[norm].append(variant)

    for norm in result:
        result[norm].sort(key=lambda v: v["stress"])

    logger.info(f"  variants dict built: {len(result):,} ambiguous forms")
    return result

