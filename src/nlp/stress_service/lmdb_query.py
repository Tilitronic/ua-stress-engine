"""
lmdb_query.py — Self-contained LMDB reader for the Ukrainian stress database.

The database was built by an offline generation pipeline.  This module is the
*only* runtime reader; it carries no dependency on that pipeline.

Storage format
--------------
  Key   : word form, UTF-8 bytes
  Value : msgpack-encoded list of WordFormDict objects
          [
            {
              "stress_variants": [int, ...],   # vowel indices (0-based)
              "pos":             [str, ...],   # optional, e.g. ["NOUN"]
              "feats":           {str: [str]}, # optional morphological features
            },
            ...                                # one entry per homograph variant
          ]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import lmdb
import msgpack

from .types import WordLookupResult


class LMDBQuery:
    """
    Read-only accessor for the Ukrainian stress LMDB database.

    Thread-safety: a single LMDBQuery instance may be shared across threads
    (lmdb read-only transactions are re-entrant).

    Usage::

        db = LMDBQuery("src/nlp/stress_service/stress.lmdb")
        result = db.lookup("замок")  # → list[WordFormDict] or None
        db.close()

    Context-manager form::

        with LMDBQuery("src/nlp/stress_service/stress.lmdb") as db:
            result = db.lookup("мама")
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        if not self._path.exists():
            raise FileNotFoundError(
                f"LMDB stress database not found: {self._path}\n"
                "Download it from LFS or rebuild with the generation pipeline."
            )
        # map_size=0 → auto-detect on read-only open (lmdb will use the file size)
        self._env = lmdb.open(
            str(self._path),
            readonly=True,
            max_dbs=1,
            lock=False,       # skip lock file on read-only
            map_size=2 ** 30, # 1 GB ceiling; actual usage is ~167 MB
        )

    # ── public API ────────────────────────────────────────────────────────────

    def lookup(self, word: str) -> Optional[WordLookupResult]:
        """
        Look up *word* (already normalised) in the database.

        Returns a list of variant dicts on success, ``None`` if not found.
        """
        key = word.encode("utf-8")
        with self._env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            return None
        return msgpack.unpackb(raw, raw=False)

    @property
    def entry_count(self) -> int:
        """Total number of entries in the database."""
        return self._env.stat()["entries"]

    def close(self) -> None:
        """Close the LMDB environment."""
        self._env.close()

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "LMDBQuery":
        return self

    def __exit__(self, *_) -> None:
        self.close()
