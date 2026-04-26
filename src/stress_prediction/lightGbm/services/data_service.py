"""Data loading, label encoding, dataset building, and train/val splitting.

Handles I/O for the training DB, external validation sample, and
handcrafted test CSV.  Depends only on *feature_service* and *constants*.
"""

import csv
import json
import multiprocessing as mp
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    from src.utils.normalize_apostrophe import normalize_apostrophe
except ImportError:
    import sys
    from pathlib import Path
    # Add project root to path for training scripts run from subdirectories
    root = Path(__file__).resolve().parents[4]  # lightgbm/services -> lightgbm -> lightgbm -> stress_prediction -> src -> root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.utils.normalize_apostrophe import normalize_apostrophe

from .constants import (
    VOWEL_SET, MAX_VOWEL_CLASS, SYLLABLE_BUCKETS,
)
from .feature_service import build_features_v13, find_vowels

N_WORKERS = max(1, mp.cpu_count() - 1)


# ────────────────────────────────────────────────────────────────────
# Label encoding
# ────────────────────────────────────────────────────────────────────
def stress_to_vowel_label(stress_indices_raw: list,
                          vowels: List[int]) -> int:
    """Map a stress-index list to a 0-based vowel-ordinal label.

    Returns -1 when the label is invalid (no stress, out of range,
    beyond MAX_VOWEL_CLASS).
    """
    if not stress_indices_raw or not vowels:
        return -1
    vi = stress_indices_raw[0]
    if vi < 0 or vi >= len(vowels):
        return -1
    if vi > MAX_VOWEL_CLASS:
        return -1
    return vi


# ────────────────────────────────────────────────────────────────────
# Database loaders
# ────────────────────────────────────────────────────────────────────
def load_training_data(db_path: Path) -> pd.DataFrame:
    """Load raw training rows from SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        "SELECT form, lemma, stress_indices, pos, features_json, variant_type "
        "FROM training_entries", conn,
    )
    conn.close()
    return df


def load_external_sample(db_path: Path, n: int, seed: int) -> List[dict]:
    """Load a stratified external validation sample (~*n* words).

    Per-syllable-count quotas ensure 2-syllable words are represented
    as well as 7+ syllable words.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT form, lemma, stress_indices, pos, features_json "
        "FROM training_entries "
        "WHERE variant_type NOT IN ('free_variant', 'grammatical_homonym')"
    )
    rows = cur.fetchall()
    conn.close()

    by_syl: Dict[int, list] = {}
    seen = set()
    for form, lemma, stress_raw_str, pos, features_json in rows:
        if not form or not stress_raw_str:
            continue
        lower = normalize_apostrophe(form).lower()
        key = (lower, pos or "")
        if key in seen:
            continue
        seen.add(key)

        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) < 2:
            continue
        try:
            stress_raw = json.loads(stress_raw_str)
        except Exception:
            continue
        if not stress_raw:
            continue
        vi = stress_raw[0]
        if vi < 0 or vi >= len(vowels) or vi > MAX_VOWEL_CLASS:
            continue

        n_syl = len(vowels)
        bucket = min(n_syl, max(SYLLABLE_BUCKETS))
        by_syl.setdefault(bucket, []).append({
            "form": form, "pos": pos or "X",
            "expected_label": vi, "vowels": vowels, "lemma": lemma,
            "features_json": features_json, "n_syllables": n_syl,
        })

    rng = random.Random(seed)
    sample: List[dict] = []

    min_per_bucket = 50
    remaining = n
    for bucket in sorted(by_syl.keys()):
        pool = by_syl[bucket]
        take = min(min_per_bucket, len(pool))
        sample.extend(rng.sample(pool, take))
        remaining -= take

    if remaining > 0:
        leftover = []
        already_forms = {(normalize_apostrophe(s["form"]).lower(), s["pos"]) for s in sample}
        for bucket in sorted(by_syl.keys()):
            for item in by_syl[bucket]:
                key = (normalize_apostrophe(item["form"]).lower(), item["pos"])
                if key not in already_forms:
                    leftover.append(item)
        if leftover:
            extra = rng.sample(leftover, min(remaining, len(leftover)))
            sample.extend(extra)

    rng.shuffle(sample)
    return sample[:n]


# ────────────────────────────────────────────────────────────────────
# Handcrafted CSV loader
# ────────────────────────────────────────────────────────────────────
def load_handcrafted_tests(csv_path: Path) -> List[Tuple[str, str, Any, str, Optional[str]]]:
    """Load handcrafted test words from an external CSV file.

    CSV columns: ``word, pos, expected_vowel_index, description``
    Optional column: ``features_json``

    *   ``expected_vowel_index`` empty → ``None`` (excluded from score)
    *   ``"1,2"`` → ``[1, 2]`` (ambiguous — either accepted)
    *   Lines starting with ``#`` are comments.

    Returns ``[(word, pos, expected_or_None, description, features_json_or_None), ...]``.
    """
    if not csv_path.exists():
        return []

    tests: List[Tuple[str, str, Any, str, Optional[str]]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(
            (line for line in f if not line.lstrip().startswith("#"))
        )
        for row in reader:
            word = normalize_apostrophe(row["word"].strip())
            if not word:
                continue
            pos = row.get("pos", "X").strip() or "X"
            raw = row.get("expected_vowel_index", "").strip()
            raw_features = row.get("features_json", "").strip()
            features_json = raw_features if raw_features else None
            if not raw:
                expected: Any = None
            elif "," in raw:
                expected = [int(x) for x in raw.split(",")]
            else:
                expected = int(raw)
            desc = row.get("description", "").strip()
            tests.append((word, pos, expected, desc, features_json))

    return tests


# ────────────────────────────────────────────────────────────────────
# Parallel feature builder
# ────────────────────────────────────────────────────────────────────
class ChunkProcessor:
    """Callable for :pymod:`multiprocessing` — builds features for a chunk."""

    def __init__(self, min_vowels: int = 2):
        self.min_vowels = min_vowels

    def __call__(self, chunk: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in chunk.iterrows():
            form = row["form"] or ""
            pos = row.get("pos") or ""
            features_json = row.get("features_json") or None
            lower = normalize_apostrophe(form).lower()
            vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
            if len(vowels) < self.min_vowels:
                continue
            try:
                stress_raw = json.loads(row["stress_indices"] or "[]")
            except Exception:
                stress_raw = []
            label = stress_to_vowel_label(stress_raw, vowels)
            if label < 0:
                continue
            rec = build_features_v13(form, pos, features_json)
            rec["__label__"] = label
            rec["__lemma__"] = row.get("lemma", "")
            records.append(rec)
        return pd.DataFrame(records)


def build_dataset(df: pd.DataFrame, min_vowels: int = 2,
                  n_workers: Optional[int] = None):
    """Filter, deduplicate, and build the feature matrix.

    Returns ``(X, y, lemmas)``.
    """
    n_workers = n_workers or N_WORKERS
    before = len(df)
    df = df[~df["variant_type"].isin(["free_variant",
                                       "grammatical_homonym"])].copy()
    df["stress_count"] = (df.groupby(["form", "pos", "features_json"])
                          ["stress_indices"].transform("nunique"))
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"])
    df = df.reset_index(drop=True)

    chunk_size = max(1, len(df) // n_workers)
    chunks = [df.iloc[i: i + chunk_size]
              for i in range(0, len(df), chunk_size)]

    processor = ChunkProcessor(min_vowels=min_vowels)
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(processor, chunks)

    combined = pd.concat(results, ignore_index=True)
    y = combined["__label__"].astype(int)
    lemmas = combined["__lemma__"]
    X = combined.drop(columns=["__label__", "__lemma__"])
    return X, y, lemmas


def group_split(lemmas, X, y, test_size=0.1, seed=42):
    """Group-shuffle-split by lemma to avoid data leakage."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=seed)
    ti, vi = next(splitter.split(X, y, lemmas.values))
    return (
        X.iloc[ti].reset_index(drop=True),
        X.iloc[vi].reset_index(drop=True),
        y.iloc[ti].reset_index(drop=True),
        y.iloc[vi].reset_index(drop=True),
    )
