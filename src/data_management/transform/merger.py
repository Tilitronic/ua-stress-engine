import logging
import time
import os
import shutil
import tempfile
import sqlite3
import json
import hashlib
import itertools
import heapq
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple, TypedDict
from tqdm import tqdm
from pydantic import TypeAdapter
import msgpack
import lmdb
from .data_unifier import LinguisticEntry


# Type definitions for intermediate representations
class LinguisticEntryDict(TypedDict, total=False):
    """TypedDict for LinguisticEntry dict representation."""
    word: str
    lemma: str  # Alternative field name
    forms: List[Dict[str, Any]]  # List of WordForm dicts
    possible_stress_indices: List[List[int]]
    meta: Dict[str, Any]


class WordFormDict(TypedDict, total=False):
    """TypedDict for WordForm dict representation."""
    form: str
    stress_indices: List[int]
    pos: str
    feats: Dict[str, str]
    lemma: Optional[str]
    main_definition: Optional[str]
    alt_definitions: Optional[List[str]]
    translations: Optional[List[Dict[str, Any]]]
    etymology_templates: Optional[List[Dict[str, Any]]]
    etymology_number: Optional[int]
    tags: Optional[List[str]]
    examples: List[str]
    roman: Optional[str]
    ipa: Optional[str]
    etymology: Optional[str]
    inflection_templates: Optional[List[Dict[str, Any]]]
    categories: Optional[List[str]]
    sense_id: Optional[str]
from src.data_management.transform.cache_utils import (
    cache_path_for_key, save_to_cache_streaming, load_from_cache_streaming, to_serializable, compute_parser_hash
)

# NOTE: Type Safety Improvements
# This module mixes LinguisticEntry (Pydantic) with Dict[str, Any] for flexibility.
# Dict[str, Any] is used because:
# 1. LMDB serialization requires dict compatibility (msgpack)
# 2. Intermediate processing steps may have partial data
# 3. SQL normalization breaks up the structure across tables
#
# To improve type safety, consider:
# - Use Pydantic model_dump() to convert to dict explicitly
# - Add TypedDict for intermediate representations (instead of Dict[str, Any])
# - Use TypeAdapter for validation when converting back to LinguisticEntry
import os
import hashlib
from typing import Tuple


def merge_linguistic_dicts(dicts: List[Dict[str, LinguisticEntryDict]]) -> Dict[str, LinguisticEntry]:
    """
    Merge multiple dictionaries of LinguisticEntry losslessly.
    
    Args:
        dicts: List of dictionaries mapping lemma -> LinguisticEntry dict
        
    Returns:
        Dictionary mapping lemma -> merged LinguisticEntry object
        
    Details:
    - For each lemma, merge all WordForms (deduplicate by all fields)
    - Merge possible_stress_indices as unique arrays
    - Merge meta dicts (shallow merge)
    """
    merged: Dict[str, LinguisticEntry] = {}

    logger = logging.getLogger("merging")
    total_lemmas = sum(len(d) for d in dicts)
    logger.info(f"Starting merging of {len(dicts)} dictionaries, total lemmas: {total_lemmas}")

    def covers(form_a, form_b):
        # Returns True if form_a covers form_b (all fields in b are in a and equal)
        a = form_a.model_dump()
        b = form_b.model_dump()
        return all(k in a and a[k] == v for k, v in b.items())

    for d in dicts:
        for lemma, entry in d.items():
            entry_obj = TypeAdapter(LinguisticEntry).validate_python(entry) if isinstance(entry, dict) else entry
            if lemma not in merged:
                merged[lemma] = entry_obj.model_copy(deep=True)
            else:
                # --- Enhanced WordForm merge logic ---
                existing_forms = merged[lemma].forms
                new_forms = []
                for wf in entry_obj.forms:
                    add = True
                    to_replace = None
                    for i, ef in enumerate(existing_forms):
                        if covers(wf, ef):
                            to_replace = i
                            add = True
                            break
                        elif covers(ef, wf):
                            add = False
                            break
                    if to_replace is not None:
                        existing_forms[to_replace] = wf
                    elif add:
                        existing_forms.append(wf)
                # --- End WordForm merge logic ---
                # Merge possible_stress_indices (unique arrays)
                psi = merged[lemma].possible_stress_indices
                for arr in entry_obj.possible_stress_indices:
                    arr_sorted = sorted(arr)
                    if not any(sorted(x) == arr_sorted for x in psi):
                        psi.append(list(arr_sorted))
                # Merge meta (shallow)
                merged[lemma].meta.update(entry_obj.meta)
    logger.info(f"Merging complete. Unique lemmas: {len(merged)}")
    return merged



# --- Streaming, disk-backed, scientific-grade merge to LMDB ---
def merge_caches_and_save_lmdb(lmdb_dirs: List[str], *, merged_prefix="MERGED") -> str:
    """
    Streaming, disk-backed, scientific-grade merge directly into a new LMDB.
    No in-memory dicts. Merges per-lemma from all LMDB caches and writes directly to a merged LMDB.
    Returns the path to the merged LMDB directory.
    """


    logger = logging.getLogger("merger")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    from .parsing_merging_service import compute_merged_cache_key
    merged_cache_key = compute_merged_cache_key(lmdb_dirs)
    cache_folder = os.path.join(os.path.dirname(__file__), "cache")
    merged_lmdb_path = os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}_lmdb")
    # --- Delete all old MERGED_* lmdb folders before creating new merged cache ---
    for f in Path(cache_folder).glob(f"{merged_prefix}*_lmdb"):
        try:
            if f.is_dir():
                import shutil
                shutil.rmtree(f)
        except Exception as e:
            print(f"[CLEANUP] Could not delete {f}: {e}")
    start_time = time.time()
    # Remove existing merged LMDB if present
    if os.path.exists(merged_lmdb_path):
        import shutil
        shutil.rmtree(merged_lmdb_path)

    def stream_lmdb_entries(lmdb_path: str) -> Iterator[Tuple[str, LinguisticEntryDict]]:
        """Stream entries from LMDB as (lemma, entry_dict) tuples."""
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"[LMDB MERGE] LMDB path does not exist: {lmdb_path}")
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            for k, v in txn.cursor():
                yield k.decode('utf-8'), msgpack.unpackb(v, raw=False)
        env.close()

    def per_lemma_merge(existing: LinguisticEntryDict, new: LinguisticEntryDict) -> LinguisticEntry:
        """
        Merge two linguistic entries for the same lemma.
        
        Args:
            existing: Dict representation of existing LinguisticEntry (from msgpack/LMDB)
            new: Dict representation of new LinguisticEntry to merge in (from msgpack/LMDB)
            
        Returns:
            Merged LinguisticEntry Pydantic object with combined forms and stress indices
            
        Note:
            Uses TypeAdapter for safe validation from dict -> Pydantic model.
            This ensures all data is validated against the schema before merging.
        """
        # Use the scientific-grade merging logic from merge_linguistic_dicts, but for two entries
        from pydantic import TypeAdapter
        from .data_unifier import LinguisticEntry
        entry_obj1 = TypeAdapter(LinguisticEntry).validate_python(existing) if isinstance(existing, dict) else existing
        entry_obj2 = TypeAdapter(LinguisticEntry).validate_python(new) if isinstance(new, dict) else new
        merged = merge_linguistic_dicts([{entry_obj1.word: entry_obj1.model_dump()}, {entry_obj2.word: entry_obj2.model_dump()}])
        return merged[entry_obj1.word]

    # Open merged LMDB for writing
    # Count total lemmas for progress bar
    total_lemmas = 0
    for lmdb_path in lmdb_dirs:
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"[LMDB MERGE] LMDB path does not exist: {lmdb_path}")
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            total_lemmas += txn.stat()['entries']
        env.close()

    env = lmdb.open(merged_lmdb_path, map_size=100*1024*1024*1024, writemap=True, map_async=True, sync=True, metasync=True)
    txn = env.begin(write=True)
    count = 0
    batch_size = 1000
    import heapq, itertools
    streams = [stream_lmdb_entries(p) for p in lmdb_dirs]
    with tqdm(total=total_lemmas, desc="Merge→LMDB", unit='lemma', dynamic_ncols=True, leave=True) as pbar:
        for lemma, group in itertools.groupby(heapq.merge(*streams, key=lambda x: x[0]), key=lambda x: x[0]):
            entries = [entry for _, entry in group]
            key = lemma.encode('utf-8')
            def ensure_dict(obj: Any) -> LinguisticEntryDict:
                """Convert Pydantic model to dict for msgpack serialization."""
                return obj.model_dump() if hasattr(obj, "model_dump") else obj
            if txn.get(key):
                existing = msgpack.unpackb(txn.get(key), raw=False)
                merged = per_lemma_merge(existing, entries[0]) if len(entries) == 1 else per_lemma_merge(existing, merge_linguistic_dicts([{lemma: e} for e in entries])[lemma])
                merged = ensure_dict(merged)
                txn.put(key, msgpack.packb(merged, use_bin_type=True))
            elif len(entries) == 1:
                entry = ensure_dict(entries[0])
                txn.put(key, msgpack.packb(entry, use_bin_type=True))
            else:
                merged = merge_linguistic_dicts([{lemma: e} for e in entries])[lemma]
                merged = ensure_dict(merged)
                txn.put(key, msgpack.packb(merged, use_bin_type=True))
            count += 1
            pbar.update(1)
            if count % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
    txn.commit()
    env.close()
    elapsed = time.time() - start_time
    logger.info(f"Merging to LMDB complete. Unique lemmas: {count}. Time: {elapsed:.2f} seconds")
    print(f"\n=== Merging to LMDB Statistics ===")
    print(f"Merged LMDB cache path: {merged_lmdb_path}")
    print(f"Total merging time: {elapsed:.2f} seconds\n")

    # --- Compaction step to minimize LMDB size ---
    print(f"[LMDB] Starting compaction to minimal size at {merged_lmdb_path} ...")
    import tempfile
    compact_dir = Path(tempfile.mkdtemp(prefix="lmdb_compact_"))
    env = lmdb.open(merged_lmdb_path, readonly=True)
    before_size = 0
    for fname in ["data.mdb", "lock.mdb"]:
        f = Path(merged_lmdb_path) / fname
        if f.exists():
            before_size += f.stat().st_size
    print(f"[LMDB] Before compaction: data.mdb + lock.mdb size = {before_size/1024/1024:.2f} MB")
    env.copy(str(compact_dir), compact=True)
    env.close()
    # Ensure merged LMDB directory exists before replacing files
    Path(merged_lmdb_path).mkdir(parents=True, exist_ok=True)
    # Replace original LMDB files with compacted ones
    import shutil
    for fname in ["data.mdb", "lock.mdb"]:
        orig = Path(merged_lmdb_path) / fname
        compacted = compact_dir / fname
        if compacted.exists():
            try:
                # If on different drives, use copy2 then unlink
                if orig.drive != compacted.drive:
                    shutil.copy2(str(compacted), str(orig))
                    compacted.unlink()
                else:
                    compacted.replace(orig)
            except Exception as e:
                print(f"[LMDB] Error moving compacted file {compacted} to {orig}: {e}")
                raise
    after_size = 0
    for fname in ["data.mdb", "lock.mdb"]:
        f = Path(merged_lmdb_path) / fname
        if f.exists():
            after_size += f.stat().st_size
    import shutil
    shutil.rmtree(compact_dir)
    print(f"[LMDB] Compaction complete. LMDB at {merged_lmdb_path} is now minimal size.")
    print(f"[LMDB] After compaction: data.mdb + lock.mdb size = {after_size/1024/1024:.2f} MB (saved {(before_size-after_size)/1024/1024:.2f} MB)")
    print(f"[LMDB] LMDB files present: {[f.name for f in Path(merged_lmdb_path).glob('*')]}")
    return merged_lmdb_path

# --- SQL Exporter Config and Exporter ---
class SQLExportConfig:
    def __init__(self, db_path, overwrite: bool = False):
        self.db_path = Path(db_path)
        self.overwrite = overwrite

class SQLExporter:
    # Use field names from data_unifier and schema
    from src.data_management.transform.data_unifier import UPOS, UDFeatKey, WordForm, LinguisticEntry

    from enum import Enum
    class WordFormField(str, Enum):
        FORM = 'form'
        LEMMA = 'lemma'
        POS = 'pos'
        MAIN_DEFINITION = 'main_definition'
        ROMAN = 'roman'
        IPA = 'ipa'
        ETYMOLOGY = 'etymology'
        ETYMOLOGY_NUMBER = 'etymology_number'
        SENSE_ID = 'sense_id'

    WORD_FORM_FIELDS = [
        WordFormField.FORM.value,
        WordFormField.LEMMA.value,
        WordFormField.POS.value,
        WordFormField.MAIN_DEFINITION.value,
        WordFormField.ROMAN.value,
        WordFormField.IPA.value,
        WordFormField.ETYMOLOGY.value,
        WordFormField.ETYMOLOGY_NUMBER.value,
        WordFormField.SENSE_ID.value,
        'stress_indices_json',
    ]
    # For normalized tables
    NESTED_TABLES = [
        'feature', 'translation', 'etymology_template', 'inflection_template', 'category', 'tag', 'example', 'possible_stress_index', 'meta'
    ]

    def __init__(self, config: SQLExportConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger("SQLExporter")

    def create_schema(self, conn: sqlite3.Connection) -> None:
        # Read schema from file and add dedup tables for main_definition and etymology
        schema_path = Path(__file__).parent / 'linguistic_data_schema.sql'
        with open(schema_path, encoding='utf-8') as f:
            schema_sql = f.read()
        # Add dedup tables if not present
        schema_sql += '''
        CREATE TABLE IF NOT EXISTS definition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS etymology_text (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT UNIQUE NOT NULL
        );
        '''
        conn.executescript(schema_sql)

    def export_streaming(self, data_iter: Iterator[Tuple[str, Any]], total: Optional[int] = None, batch_size: int = 1000) -> None:
        """
        Export entries to SQLite using normalized schema, batch insert, and PRAGMA optimizations.
        Use msgpack only for large/nested fields (not for normalized tables).
        
        Args:
            data_iter: Iterator yielding (lemma: str, entry_data: dict) tuples
            total: Optional total count for progress bar
            batch_size: Batch size for SQL inserts
        """
        from tqdm import tqdm
        # --- Informative summary will be printed at the end ---
        if self.config.overwrite and self.config.db_path.exists():
            self.logger.info(f"Removing existing SQLite file at {self.config.db_path}")
            self.config.db_path.unlink()
        conn = sqlite3.connect(str(self.config.db_path))
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        self.create_schema(conn)
        cur = conn.cursor()
        # --- Deduplication caches ---
        definition_cache = {}
        etymology_cache = {}
        def get_definition_id(text):
            if not text:
                return None
            if text in definition_cache:
                return definition_cache[text]
            cur.execute("INSERT OR IGNORE INTO definition (text) VALUES (?)", (text,))
            cur.execute("SELECT id FROM definition WHERE text=?", (text,))
            id_ = cur.fetchone()[0]
            definition_cache[text] = id_
            return id_
        def get_etymology_id(text):
            if not text:
                return None
            if text in etymology_cache:
                return etymology_cache[text]
            cur.execute("INSERT OR IGNORE INTO etymology_text (text) VALUES (?)", (text,))
            cur.execute("SELECT id FROM etymology_text WHERE text=?", (text,))
            id_ = cur.fetchone()[0]
            etymology_cache[text] = id_
            return id_
        word_form_batch = []
        feature_batch = []
        translation_batch = []
        etymology_template_batch = []
        inflection_template_batch = []
        category_batch = []
        tag_batch = []
        example_batch = []
        possible_stress_index_batch = []
        meta_batch = []
        lemma_entry_batch = []  # (lemma, possible_stress_indices_json)
        lemma_stress_map = {}   # lemma -> set of tuple(stress_indices)
        count = 0
        # Estimate total for progress bar if not provided
        if total is None:
            # Try to estimate from LMDB if possible
            try:
                if hasattr(data_iter, '__length_hint__'):
                    total = data_iter.__length_hint__()
            except Exception:
                total = None
        pbar = tqdm(total=total, desc="SQL Export", ncols=100, dynamic_ncols=True, leave=True)
        for key, value in data_iter:
            # Support both dict and Pydantic model
            # Convert to dict for consistent processing
            value_dict: LinguisticEntryDict
            if hasattr(value, "model_dump"):
                value_dict = value.model_dump()
            else:
                value_dict = value
            forms = value_dict.get('forms', [])
            lemma = value_dict.get('word') or value_dict.get('lemma') or key
            # Collect all unique possible stress indices for this lemma
            for form_obj in forms:
                if hasattr(form_obj, "model_dump"):
                    form_dict = form_obj.model_dump()
                else:
                    form_dict = form_obj
                stress_indices = tuple(sorted(form_dict.get('stress_indices', []) or []))
                if lemma not in lemma_stress_map:
                    lemma_stress_map[lemma] = set()
                if stress_indices:
                    lemma_stress_map[lemma].add(stress_indices)
            for form_obj in forms:
                # Support both dict and Pydantic model for forms
                form_dict: WordFormDict
                if hasattr(form_obj, "model_dump"):
                    form_dict = form_obj.model_dump()
                else:
                    form_dict = form_obj
                form = form_dict.get('form')
                pos = form_dict.get('pos')
                main_definition_id = get_definition_id(form_dict.get('main_definition'))
                roman = form_dict.get('roman')
                ipa = form_dict.get('ipa')
                etymology_id = get_etymology_id(form_dict.get('etymology'))
                etymology_number = form_dict.get('etymology_number')
                sense_id = form_dict.get('sense_id')
                stress_indices = form_dict.get('stress_indices', []) or []
                word_form_batch.append((form, lemma, pos, main_definition_id, roman, ipa, etymology_id, etymology_number, sense_id, json.dumps(stress_indices, ensure_ascii=False, separators=(',', ':'))))
            # Prepare nested/related rows (normalize repeated strings)
            feats = form_dict.get('feats', {})
            for k, v in feats.items():
                feature_batch.append((None, k, v))  # word_form_id to be filled after insert
            translations = value_dict.get('translations', []) or []
            for t in translations:
                # Compact translation JSON fields if present
                lang = t.get('lang')
                text = t.get('text')
                sense = t.get('sense')
                translation_batch.append((None, lang, text, sense))
            etymology_templates = value_dict.get('etymology_templates', []) or []
            for et in etymology_templates:
                et_name = et.get('name')
                et_args = json.dumps(et.get('args', {}), ensure_ascii=False, separators=(',', ':'))
                etymology_template_batch.append((None, et_name, et_args))
            inflection_templates = value_dict.get('inflection_templates', []) or []
            for it in inflection_templates:
                it_name = it.get('name')
                it_args = json.dumps(it.get('args', {}), ensure_ascii=False, separators=(',', ':'))
                inflection_template_batch.append((None, it_name, it_args))
            categories = value_dict.get('categories', []) or []
            for cat in categories:
                category_batch.append((None, cat))
            tags = value_dict.get('tags', []) or []
            for tag in tags:
                tag_batch.append((None, tag))
            examples = value_dict.get('examples', []) or []
            for ex in examples:
                example_batch.append((None, ex))
            # (No longer insert possible_stress_indices per form; handled at lemma level)
            meta = value_dict.get('meta', {})
            if meta:
                meta_json = json.dumps(meta, ensure_ascii=False, separators=(',', ':'))
                meta_batch.append((None, meta_json))
            count += 1
            pbar.update(1)
            # Batch insert every batch_size word_forms
            if len(word_form_batch) >= batch_size:
                self._insert_batches(cur, word_form_batch, feature_batch, translation_batch, etymology_template_batch,
                                    inflection_template_batch, category_batch, tag_batch, example_batch,
                                    possible_stress_index_batch, meta_batch)
                conn.commit()
                word_form_batch.clear()
                feature_batch.clear()
                translation_batch.clear()
                etymology_template_batch.clear()
                inflection_template_batch.clear()
                category_batch.clear()
                tag_batch.clear()
                example_batch.clear()
                possible_stress_index_batch.clear()
                meta_batch.clear()
        pbar.close()
        import sys
        sys.stdout.flush()
        # Insert any remaining
        if word_form_batch:
            self._insert_batches(cur, word_form_batch, feature_batch, translation_batch, etymology_template_batch,
                                inflection_template_batch, category_batch, tag_batch, example_batch,
                                possible_stress_index_batch, meta_batch)
            conn.commit()

        # --- Insert lemma_entry table with possible_stress_indices per lemma ---
        for lemma, stress_set in lemma_stress_map.items():
            # Only keep non-empty stress patterns
            unique_patterns = [list(pattern) for pattern in sorted(stress_set) if pattern]
            if unique_patterns:
                psi_json = json.dumps(unique_patterns, ensure_ascii=False, separators=(',', ':'))
                lemma_entry_batch.append((lemma, psi_json))
        if lemma_entry_batch:
            cur.executemany('''
                INSERT OR REPLACE INTO lemma_entry (lemma, possible_stress_indices_json)
                VALUES (?, ?)
            ''', lemma_entry_batch)
            conn.commit()
        cur.close()
        conn.execute("PRAGMA optimize;")
        conn.close()
        # Print informative summary after export
        db_size = None
        try:
            db_size = self.config.db_path.stat().st_size
        except Exception:
            db_size = None
        summary = f"[SQLITE] Exported {count} entries to SQLite at {self.config.db_path}"
        if db_size is not None:
            summary += f" | Final DB size: {db_size/1024/1024:.2f} MB"
        print(summary)

    def _insert_batches(self, cur, word_form_batch, feature_batch, translation_batch, etymology_template_batch,
                       inflection_template_batch, category_batch, tag_batch, example_batch,
                       possible_stress_index_batch, meta_batch):
        # Insert word_form rows one by one to collect rowids
        word_form_ids = []
        for row in word_form_batch:
            cur.execute('''
                INSERT INTO word_form (form, lemma, pos, main_definition_id, roman, ipa, etymology_id, etymology_number, sense_id, stress_indices_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
            word_form_ids.append(cur.lastrowid)
        # Helper to assign word_form_id to each related row
        def assign_ids(batch):
            out = []
            idx = 0
            for row in batch:
                out.append((word_form_ids[idx],) + row[1:])
                if len(out) % len(word_form_ids) == 0:
                    idx += 1
            return out
        if feature_batch:
            cur.executemany('''
                INSERT INTO feature (word_form_id, key, value) VALUES (?, ?, ?)
            ''', assign_ids(feature_batch))
        if translation_batch:
            cur.executemany('''
                INSERT INTO translation (word_form_id, lang, text, sense) VALUES (?, ?, ?, ?)
            ''', assign_ids(translation_batch))
        if etymology_template_batch:
            cur.executemany('''
                INSERT INTO etymology_template (word_form_id, name, args_json) VALUES (?, ?, ?)
            ''', assign_ids(etymology_template_batch))
        if inflection_template_batch:
            cur.executemany('''
                INSERT INTO inflection_template (word_form_id, name, args_json) VALUES (?, ?, ?)
            ''', assign_ids(inflection_template_batch))
        if category_batch:
            cur.executemany('''
                INSERT INTO category (word_form_id, category) VALUES (?, ?)
            ''', assign_ids(category_batch))
        if tag_batch:
            cur.executemany('''
                INSERT INTO tag (word_form_id, tag) VALUES (?, ?)
            ''', assign_ids(tag_batch))
        if example_batch:
            cur.executemany('''
                INSERT INTO example (word_form_id, example) VALUES (?, ?)
            ''', assign_ids(example_batch))
        if possible_stress_index_batch:
            cur.executemany('''
                INSERT INTO possible_stress_index (word_form_id, stress_indices_json) VALUES (?, ?)
            ''', assign_ids(possible_stress_index_batch))
        if meta_batch:
            cur.executemany('''
                INSERT INTO meta (word_form_id, meta_json) VALUES (?, ?)
            ''', assign_ids(meta_batch))


class LMDBExportConfig:
    def __init__(self, db_path, map_size: Optional[int] = None, overwrite: bool = False):
        # Accept str or Path, always store as Path
        self.db_path = Path(db_path)
        self.map_size = map_size
        self.overwrite = overwrite


# --- LMDBExporter remains unchanged below (for optional use) ---
class LMDBExporter:
    def __init__(self, config: LMDBExportConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger("LMDBExporter")

    def estimate_map_size(self, data: Dict[str, Any], sample_size: int = 1000) -> int:
        total_size = 0
        count = 0
        for k, v in data.items():
            if count >= sample_size:
                break
            total_size += len(k.encode('utf-8'))
            total_size += len(msgpack.packb(v, use_bin_type=True))
            count += 1
        avg_size = total_size // max(count, 1)
        estimated = int(avg_size * len(data) * 1.25)
        return max(estimated, 100 * 1024 * 1024)  # At least 100MB

    def export_streaming(self, data_iter, total=None, show_progress=True) -> None:
        from tqdm import tqdm
        lmdb_data_file = self.config.db_path / "data.mdb"
        if self.config.overwrite and lmdb_data_file.exists():
            self.logger.info(f"Removing existing LMDB file at {lmdb_data_file}")
            lmdb_data_file.unlink()
        self.config.db_path.mkdir(parents=True, exist_ok=True)
        if self.config.map_size is None:
            self.config.map_size = 10 * 1024 * 1024 * 1024  # 10GB default, adjust as needed
        env = lmdb.open(str(self.config.db_path), map_size=self.config.map_size, writemap=True, map_async=True, sync=True, metasync=True)
        count = 0
        batch_size = 10000
        txn = env.begin(write=True)
        try:
            iterator = data_iter
            if show_progress:
                iterator = tqdm(data_iter, total=total, desc="LMDB Export", ncols=80)
            for key, value in iterator:
                packed = msgpack.packb(value, use_bin_type=True)
                txn.put(key.encode('utf-8'), packed)
                count += 1
                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        finally:
            env.close()
        self.logger.info(f"Stream-exported {count} entries to LMDB at {self.config.db_path}")
        print(f"[LMDB] Stream-exported {count} entries to LMDB at {self.config.db_path}")
        # Compaction step (unchanged)
        import os
        self.logger.info("[LMDB] Starting compaction to minimal size...")
        print(f"[LMDB] Starting compaction to minimal size at {self.config.db_path} ...")
        compact_dir = Path(tempfile.mkdtemp(prefix="lmdb_compact_"))
        env = lmdb.open(str(self.config.db_path), readonly=True)
        before_size = 0
        for fname in ["data.mdb", "lock.mdb"]:
            f = self.config.db_path / fname
            if f.exists():
                before_size += f.stat().st_size
        self.logger.info(f"[LMDB] Before compaction: data.mdb + lock.mdb size = {before_size/1024/1024:.2f} MB")
        print(f"[LMDB] Before compaction: data.mdb + lock.mdb size = {before_size/1024/1024:.2f} MB")
        env.copy(str(compact_dir), compact=True)
        env.close()
        for fname in ["data.mdb", "lock.mdb"]:
            orig = self.config.db_path / fname
            compacted = compact_dir / fname
            if compacted.exists():
                if orig.exists():
                    orig.unlink()
                shutil.move(str(compacted), str(orig))
        after_size = 0
        for fname in ["data.mdb", "lock.mdb"]:
            f = self.config.db_path / fname
            if f.exists():
                after_size += f.stat().st_size
        shutil.rmtree(compact_dir)
        self.logger.info(f"[LMDB] Compaction complete. LMDB at {self.config.db_path} is now minimal size.")
        self.logger.info(f"[LMDB] After compaction: data.mdb + lock.mdb size = {after_size/1024/1024:.2f} MB (saved {(before_size-after_size)/1024/1024:.2f} MB)")
        self.logger.info(f"[LMDB] LMDB files present: {[f.name for f in self.config.db_path.glob('*')]}")
        print(f"[LMDB] Compaction complete. LMDB at {self.config.db_path} is now minimal size.")
        print(f"[LMDB] After compaction: data.mdb + lock.mdb size = {after_size/1024/1024:.2f} MB (saved {(before_size-after_size)/1024/1024:.2f} MB)")
        print(f"[LMDB] LMDB files present: {[f.name for f in self.config.db_path.glob('*')]}")

    def verify(self, sample_words: List[str]) -> Dict[str, Any]:
        env = lmdb.open(str(self.config.db_path), readonly=True, lock=False)
        found = 0
        with env.begin() as txn:
            for word in sample_words:
                if txn.get(word.encode('utf-8')):
                    found += 1
            stats = txn.stat()
        env.close()
        return {"entries": stats["entries"], "sample_found": f"{found}/{len(sample_words)}"}


def compute_merged_cache_key(cache_paths):
    h = hashlib.sha256()
    for path in cache_paths:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


# --- Streaming, disk-backed, scientific-grade merge function ---
def merge_caches_and_save(lmdb_dirs: List[str], export_config=None, *, merged_prefix="MERGED") -> Tuple[None, str]:
    """
    Streaming, disk-backed, scientific-grade merge directly into SQLite.
    No in-memory dicts. Merges per-lemma from all LMDB caches and writes directly to SQLite.
    """
    # All imports are now at the top of the file

    def merge_linguistic_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple dictionaries of LinguisticEntry losslessly.
        - For each lemma, merge all WordForms (deduplicate by all fields)
        - Merge possible_stress_indices as unique arrays
        - Merge meta dicts (shallow merge)
        """
        merged = {}
        total_lemmas = sum(len(d) for d in dicts)
        logger = logging.getLogger("merging")
        logger.info(f"Starting merging of {len(dicts)} dictionaries, total lemmas: {total_lemmas}")

        def covers(form_a, form_b):
            # Returns True if form_a covers form_b (all fields in b are in a and equal)
            a = form_a.model_dump()
            b = form_b.model_dump()
            return all(k in a and a[k] == v for k, v in b.items())

        with tqdm(total=total_lemmas, desc="Merging", leave=True, ncols=80) as pbar:
            for d in dicts:
                for lemma, entry in d.items():
                    entry_obj = TypeAdapter(LinguisticEntry).validate_python(entry) if isinstance(entry, dict) else entry
                    if lemma not in merged:
                        merged[lemma] = entry_obj.model_copy(deep=True)
                    else:
                        existing_forms = merged[lemma].forms
                        new_forms = []
                        for wf in entry_obj.forms:
                            add = True
                            to_replace = None
                            for i, ef in enumerate(existing_forms):
                                if covers(wf, ef):
                                    to_replace = i
                                    add = True
                                    break
                                elif covers(ef, wf):
                                    add = False
                                    break
                            if to_replace is not None:
                                existing_forms[to_replace] = wf
                            elif add:
                                existing_forms.append(wf)
                        psi = merged[lemma].possible_stress_indices
                        for arr in entry_obj.possible_stress_indices:
                            arr_sorted = sorted(arr)
                            if not any(sorted(x) == arr_sorted for x in psi):
                                psi.append(list(arr_sorted))
                        merged[lemma].meta.update(entry_obj.meta)
                    pbar.update(1)
        logger.info(f"Merging complete. Unique lemmas: {len(merged)}")
        return merged
    from pydantic import TypeAdapter
    from .data_unifier import LinguisticEntry

    logger = logging.getLogger("merger")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    from .parsing_merging_service import compute_merged_cache_key
    merged_cache_key = compute_merged_cache_key(lmdb_dirs)
    cache_folder = os.path.join(os.path.dirname(__file__), "cache")
    sql_db_path = Path(os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}.sqlite3"))
    # --- Delete all old MERGED_* files before creating new merged cache ---
    for f in Path(cache_folder).glob(f"{merged_prefix}*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                import shutil
                shutil.rmtree(f)
        except Exception as e:
            print(f"[CLEANUP] Could not delete {f}: {e}")
    start_time = time.time()
    force_export = False
    if sql_db_path.exists():
        import sqlite3
        try:
            conn = sqlite3.connect(str(sql_db_path))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM word_form")
            count = cur.fetchone()[0]
            cur.close()
            conn.close()
            if count == 0:
                print(f"[SQLITE] Existing DB at {sql_db_path} has zero rows. Deleting and forcing export...")
                sql_db_path.unlink()
                force_export = True
        except Exception as e:
            print(f"[SQLITE] Could not check row count: {e}. Forcing export...")
            sql_db_path.unlink()
            force_export = True

    def stream_lmdb_entries(lmdb_path: str) -> Iterator[Tuple[str, LinguisticEntryDict]]:
        """Stream entries from LMDB as (lemma, entry_dict) tuples."""
        import lmdb, msgpack
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            for k, v in txn.cursor():
                yield k.decode('utf-8'), msgpack.unpackb(v, raw=False)
        env.close()

    def per_lemma_merge(lemma: str, entries: List[LinguisticEntryDict]) -> LinguisticEntry:
        # Use the scientific-grade merging logic from merge_linguistic_dicts, but per-lemma
        merged = None
        for entry in entries:
            entry_obj = TypeAdapter(LinguisticEntry).validate_python(entry) if isinstance(entry, dict) else entry
            # Ensure all word forms have stress_indices copied
            for wf in entry_obj.forms:
                if hasattr(wf, 'stress_indices') and wf.stress_indices is None:
                    wf.stress_indices = []
                # If stress_indices is missing, try to get from dict
                if not hasattr(wf, 'stress_indices') and isinstance(wf, dict) and 'stress_indices' in wf:
                    wf.stress_indices = wf['stress_indices']
            if merged is None:
                merged = entry_obj.model_copy(deep=True)
            else:
                # --- Enhanced WordForm merge logic ---
                existing_forms = merged.forms
                for wf in entry_obj.forms:
                    # Ensure stress_indices is preserved
                    if hasattr(wf, 'stress_indices') and wf.stress_indices is None:
                        wf.stress_indices = []
                    if not hasattr(wf, 'stress_indices') and isinstance(wf, dict) and 'stress_indices' in wf:
                        wf.stress_indices = wf['stress_indices']
                    add = True
                    to_replace = None
                    for i, ef in enumerate(existing_forms):
                        def covers(form_a, form_b):
                            a = form_a.model_dump()
                            b = form_b.model_dump()
                            return all(k in a and a[k] == v for k, v in b.items())
                        if covers(wf, ef):
                            to_replace = i
                            add = True
                            break
                        elif covers(ef, wf):
                            add = False
                            break
                    if to_replace is not None:
                        existing_forms[to_replace] = wf
                    elif add:
                        existing_forms.append(wf)
                # --- End WordForm merge logic ---
                # Merge possible_stress_indices (unique arrays)
                psi = merged.possible_stress_indices
                for arr in entry_obj.possible_stress_indices:
                    arr_sorted = sorted(arr)
                    if not any(sorted(x) == arr_sorted for x in psi):
                        psi.append(list(arr_sorted))
                # Merge meta (shallow)
                merged.meta.update(entry_obj.meta)
        return merged

    def merge_streaming_lmdbs(lmdb_paths: List[str]) -> Iterator[Tuple[str, LinguisticEntry]]:
        """Stream and merge all LMDBs, yielding (lemma, merged_entry) tuples.
        
        Args:
            lmdb_paths: List of paths to LMDB directories to merge
            
        Yields:
            Tuples of (lemma, merged_LinguisticEntry)
        """
        # Streams all LMDBs in sorted order by lemma, merges per-lemma, yields (lemma, merged_entry)
        streams = [stream_lmdb_entries(p) for p in lmdb_paths]
        # heapq.merge merges sorted streams by lemma
        for lemma, group in itertools.groupby(heapq.merge(*streams, key=lambda x: x[0]), key=lambda x: x[0]):
            entries = [entry for _, entry in group]
            merged = per_lemma_merge(lemma, entries)
            yield lemma, merged

    if not sql_db_path.exists() or force_export:
        logger.info(f"[SQLITE] No merged SQLite DB found at {sql_db_path}. Will merge and export to SQLite.")
        print(f"[SQLITE] No merged SQLite DB found at {sql_db_path}. Will merge and export to SQLite.")
        sql_export_config = SQLExportConfig(db_path=sql_db_path, overwrite=True)
        exporter = SQLExporter(sql_export_config, logger=logger)
        print(f"[SQLITE] About to export merged data to {sql_db_path}")
        # Stream and merge per-lemma, write directly to SQLite
        merged_iter = merge_streaming_lmdbs(lmdb_dirs)
        exporter.export_streaming(merged_iter)
        print(f"[SQLITE] Merged cache exported to {sql_db_path}")
        logger.info(f"[SQLITE] Merged cache exported to {sql_db_path}")
        if not sql_db_path.exists():
            raise RuntimeError(f"[CRITICAL] SQLite export failed: {sql_db_path} was not created!")
        cache_used = False
    else:
        logger.info(f"[SQLITE] Using merged SQLite DB at {sql_db_path}")
        print(f"[SQLITE] Using merged SQLite DB at {sql_db_path}")
        cache_used = True
    elapsed = time.time() - start_time
    logger.info(f"Merging complete. (Streaming, no in-memory dict). Time: {elapsed:.2f} seconds")
    print("\n=== Merging Statistics ===")
    print(f"Merged SQLite cache used: {cache_used}")
    print(f"Merged SQLite cache path: {sql_db_path}")
    print(f"Total merging time: {elapsed:.2f} seconds\n")
    return None, str(sql_db_path)

# --- Optional: LMDB export ---
## merge_caches_and_save_lmdb is deprecated and not used in the new LMDB-centric pipeline. Removed to avoid errors.
