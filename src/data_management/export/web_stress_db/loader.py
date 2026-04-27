"""
loader.py — Extract stress data from the master SQLite database.

Produces (normalised_form, stress_primary, stress_secondary, is_variative, is_heteronym)
tuples ready to be inserted into TrieBuilder.

Stress classification:
  stress_secondary is None → unique  (one unambiguous stress)
  stress_secondary is set, is_variative → variative  (both valid simultaneously)
  stress_secondary is set, is_heteronym → heteronym  (different meanings/forms)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from pathlib import Path
from typing import Generator, Optional, Set, Tuple

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
