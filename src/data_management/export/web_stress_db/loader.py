"""
loader.py — Extract stress data from the master SQLite database.

Produces (normalised_form, stress_index, is_heteronym) triples
ready to be inserted into TrieBuilder.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import unicodedata
from pathlib import Path
from typing import Generator, Tuple

logger = logging.getLogger(__name__)

# Apostrophe normalisation — same logic as src/utils/normalize_apostrophe.py
# duplicated here so the module stays self-contained with no project imports.
_CORRECT_APOSTROPHE = "\u02bc"
_WRONG_APOSTROPHES = "\u2019\u0027\u02bb\u0060\u00b4"


def _norm(word: str) -> str:
    """Lowercase + normalise apostrophe + strip combining accents."""
    w = word.lower()
    for ch in _WRONG_APOSTROPHES:
        w = w.replace(ch, _CORRECT_APOSTROPHE)
    # Strip combining diacritics (U+0301 stress marks embedded in some source forms)
    w = "".join(c for c in unicodedata.normalize("NFD", w)
                if unicodedata.category(c) != "Mn")
    return w


def load_from_master_db(
    db_path: Path,
) -> Generator[Tuple[str, int, bool], None, None]:
    """
    Yield (normalised_form, stress_index, is_heteronym) for every single-word
    entry in the master SQLite database that has stress data.

    Deduplication strategy:
      - Group by normalised form (case-folded, diacritics stripped).
      - If all variants agree on the same stress index → not a heteronym.
      - If variants disagree → heteronym=True; emit the first (most-common) index.
      - Forms with no vowels or no stress data are skipped.
    """
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
    for norm_form, indices in agg.items():
        stress = indices[0]                   # first = most-common (source order)
        is_heteronym = len(indices) > 1
        if is_heteronym:
            heteronym_count += 1
        yield norm_form, stress, is_heteronym

    logger.info(f"  {heteronym_count:,} heteronyms flagged")
