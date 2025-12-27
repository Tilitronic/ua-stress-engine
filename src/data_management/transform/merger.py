from pathlib import Path

import lmdb
import msgpack
from typing import Dict, Any, Optional, List
import logging
import shutil, tempfile

import sqlite3
import json

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
        WordFormField.SENSE_ID.value
    ]
    # For normalized tables
    NESTED_TABLES = [
        'feature', 'translation', 'etymology_template', 'inflection_template', 'category', 'tag', 'example', 'possible_stress_index', 'meta'
    ]

    def __init__(self, config: SQLExportConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger("SQLExporter")

    def create_schema(self, conn):
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

    def export_streaming(self, data_iter, total=None, batch_size=1000):
        """
        Export entries to SQLite using normalized schema, batch insert, and PRAGMA optimizations.
        Use msgpack only for large/nested fields (not for normalized tables).
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
        count = 0
        for key, value in tqdm(data_iter, total=total, desc="SQL Export", ncols=80):
            # Map fields using data_unifier types
            form = value.get('form') or key
            lemma = value.get('lemma')
            pos = value.get('pos')
            # Deduplicate main_definition and etymology
            main_definition_id = get_definition_id(value.get('main_definition'))
            roman = value.get('roman')
            ipa = value.get('ipa')
            etymology_id = get_etymology_id(value.get('etymology'))
            etymology_number = value.get('etymology_number')
            sense_id = value.get('sense_id')
            # Insert word_form row (store IDs instead of text)
            word_form_batch.append((form, lemma, pos, main_definition_id, roman, ipa, etymology_id, etymology_number, sense_id))
            # Prepare nested/related rows (normalize repeated strings)
            feats = value.get('feats', {})
            for k, v in feats.items():
                feature_batch.append((None, k, v))  # word_form_id to be filled after insert
            translations = value.get('translations', []) or []
            for t in translations:
                # Compact translation JSON fields if present
                lang = t.get('lang')
                text = t.get('text')
                sense = t.get('sense')
                translation_batch.append((None, lang, text, sense))
            etymology_templates = value.get('etymology_templates', []) or []
            for et in etymology_templates:
                et_name = et.get('name')
                et_args = json.dumps(et.get('args', {}), ensure_ascii=False, separators=(',', ':'))
                etymology_template_batch.append((None, et_name, et_args))
            inflection_templates = value.get('inflection_templates', []) or []
            for it in inflection_templates:
                it_name = it.get('name')
                it_args = json.dumps(it.get('args', {}), ensure_ascii=False, separators=(',', ':'))
                inflection_template_batch.append((None, it_name, it_args))
            categories = value.get('categories', []) or []
            for cat in categories:
                category_batch.append((None, cat))
            tags = value.get('tags', []) or []
            for tag in tags:
                tag_batch.append((None, tag))
            examples = value.get('examples', []) or []
            for ex in examples:
                example_batch.append((None, ex))
            possible_stress_indices = value.get('possible_stress_indices', []) or []
            if possible_stress_indices:
                psi_json = json.dumps(possible_stress_indices, ensure_ascii=False, separators=(',', ':'))
                possible_stress_index_batch.append((None, psi_json))
            meta = value.get('meta', {})
            if meta:
                meta_json = json.dumps(meta, ensure_ascii=False, separators=(',', ':'))
                meta_batch.append((None, meta_json))
            count += 1
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
        # Insert any remaining
        if word_form_batch:
            self._insert_batches(cur, word_form_batch, feature_batch, translation_batch, etymology_template_batch,
                                inflection_template_batch, category_batch, tag_batch, example_batch,
                                possible_stress_index_batch, meta_batch)
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
                INSERT INTO word_form (form, lemma, pos, main_definition_id, roman, ipa, etymology_id, etymology_number, sense_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def export_streaming(self, data_iter, total=None) -> None:
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
            for key, value in tqdm(data_iter, total=total, desc="LMDB Export", ncols=80):
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
import os
import hashlib
from src.data_management.transform.cache_utils import cache_path_for_key, save_to_cache_streaming, load_from_cache_streaming, to_serializable
from src.data_management.transform.parsing_merging_service import SOURCES_CONFIGS, compute_parser_hash, merge_linguistic_dicts

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

def merge_caches_and_save(names, merged_prefix="MERGED"):
    import logging
    from tqdm import tqdm
    import time
    cache_keys = [compute_parser_hash(SOURCES_CONFIGS[name]["parser_path"], SOURCES_CONFIGS[name]["db_path"]) for name in names]
    cache_paths = [cache_path_for_key(key, prefix=name) for key, name in zip(cache_keys, names)]
    # Add hash of merger.py to merged cache key
    def file_hash(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    merger_py_path = Path(__file__).resolve()
    merger_hash = file_hash(merger_py_path)
    merged_cache_key = compute_merged_cache_key(cache_paths) + '_' + merger_hash[:16]
    merged_cache_path = cache_path_for_key(merged_cache_key, prefix=merged_prefix)
    cache_folder = os.path.join(os.path.dirname(__file__), "cache")
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
    sql_db_path = Path(os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}.sqlite3"))
    lmdb_dir = Path(os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}_lmdb"))

    logger = logging.getLogger("merger")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    start_time = time.time()
    # --- Default: Use SQL export ---
    force_export = False
    if sql_db_path.exists():
        # Check if DB has zero rows; if so, delete and force export
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
    if not sql_db_path.exists() or force_export:
        logger.info(f"[SQLITE] No merged SQLite DB found at {sql_db_path}. Will merge and export to SQLite.")
        print(f"[SQLITE] No merged SQLite DB found at {sql_db_path}. Will merge and export to SQLite.")
        dicts = []
        for name, cache_path in zip(names, cache_paths):
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                logger.info(f"[CACHE] Using cache for {name} from {cache_path}")
                print(f"[CACHE] Using cache for {name} from {cache_path}")
                d = load_from_cache_streaming(cache_keys[names.index(name)], prefix=name)
            else:
                logger.info(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                print(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                d = None
            dicts.append(d)
        logger.info(f"[MERGE] Merging {len(dicts)} dictionaries...")
        print(f"[MERGE] Merging {len(dicts)} dictionaries...")
        start = time.time()
        merged = merge_linguistic_dicts(dicts)
        elapsed = time.time() - start
        logger.info(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to SQLite...")
        print(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to SQLite...")
        export_config = SQLExportConfig(db_path=sql_db_path, overwrite=True)
        exporter = SQLExporter(export_config, logger=logger)
        print(f"[SQLITE] About to export merged data to {sql_db_path}")
        # Flatten merged dict: for each lemma, for each WordForm, yield (form, word_form_dict)
        def word_form_iter():
            for lemma, entry in merged.items():
                # entry is a LinguisticEntry (Pydantic model)
                for wf in entry.forms:
                    d = wf.model_dump()
                    # Add lemma as a field if not present
                    if 'lemma' not in d or d['lemma'] is None:
                        d['lemma'] = lemma
                    yield (d['form'], d)
        exporter.export_streaming(word_form_iter(), total=sum(len(entry.forms) for entry in merged.values()))
        print(f"[SQLITE] Merged cache exported to {sql_db_path}")
        logger.info(f"[SQLITE] Merged cache exported to {sql_db_path}")
        if not sql_db_path.exists():
            raise RuntimeError(f"[CRITICAL] SQLite export failed: {sql_db_path} was not created!")
        cache_used = False
    else:
        logger.info(f"[SQLITE] Using merged SQLite DB at {sql_db_path}")
        print(f"[SQLITE] Using merged SQLite DB at {sql_db_path}")
        cache_used = True
        merged = None  # Not loaded into memory

    elapsed = time.time() - start_time
    logger.info(f"Merging complete. Unique lemmas: {len(merged) if merged is not None else 'N/A (SQLITE only)'}")
    logger.info(f"Total merging time: {elapsed:.2f} seconds")
    print("\n=== Merging Statistics ===")
    print(f"Merged SQLite cache used: {cache_used}")
    print(f"Unique lemmas: {len(merged) if merged is not None else 'N/A (SQLITE only)'}")
    print(f"Merged SQLite cache path: {sql_db_path}")
    print(f"Total merging time: {elapsed:.2f} seconds\n")
    return merged, str(sql_db_path)

# --- Optional: LMDB export ---
def merge_caches_and_save_lmdb(names, merged_prefix="MERGED"):
    """
    Use this function to export to LMDB instead of SQLite. API is the same as merge_caches_and_save.
    """
    cache_keys = [compute_parser_hash(SOURCES_CONFIGS[name]["parser_path"], SOURCES_CONFIGS[name]["db_path"]) for name in names]
    cache_paths = [cache_path_for_key(key, prefix=name) for key, name in zip(cache_keys, names)]
    merged_cache_key = compute_merged_cache_key(cache_paths)
    cache_folder = os.path.join(os.path.dirname(__file__), "cache")
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
    lmdb_dir = Path(os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}_lmdb"))
    logger = logging.getLogger("merger")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    start_time = time.time()
    if lmdb_dir.exists():
        logger.info(f"[LMDB] Using merged LMDB cache at {lmdb_dir}")
        print(f"[LMDB] Using merged LMDB cache at {lmdb_dir}")
        cache_used = True
        merged = None
    else:
        logger.info(f"[LMDB] No merged LMDB cache found at {lmdb_dir}. Will merge and export to LMDB.")
        print(f"[LMDB] No merged LMDB cache found at {lmdb_dir}. Will merge and export to LMDB.")
        dicts = []
        for name, cache_path in zip(names, cache_paths):
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                logger.info(f"[CACHE] Using cache for {name} from {cache_path}")
                print(f"[CACHE] Using cache for {name} from {cache_path}")
                d = load_from_cache_streaming(cache_keys[names.index(name)], prefix=name)
            else:
                logger.info(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                print(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                d = None
            dicts.append(d)
        logger.info(f"[MERGE] Merging {len(dicts)} dictionaries...")
        print(f"[MERGE] Merging {len(dicts)} dictionaries...")
        start = time.time()
        merged = merge_linguistic_dicts(dicts)
        elapsed = time.time() - start
        logger.info(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to LMDB...")
        print(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to LMDB...")
        export_config = LMDBExportConfig(db_path=lmdb_dir, overwrite=True)
        exporter = LMDBExporter(export_config, logger=logger)
        print(f"[LMDB] About to export merged data to {lmdb_dir}")
        exporter.export_streaming(merged.items(), total=len(merged))
        print(f"[LMDB] Merged cache exported to {lmdb_dir}")
        logger.info(f"[LMDB] Merged cache exported to {lmdb_dir}")
        if not lmdb_dir.exists():
            raise RuntimeError(f"[CRITICAL] LMDB export failed: {lmdb_dir} was not created!")
        cache_used = False
    elapsed = time.time() - start_time
    logger.info(f"Merging complete. Unique lemmas: {len(merged) if merged is not None else 'N/A (LMDB only)'}")
    logger.info(f"Total merging time: {elapsed:.2f} seconds")
    print("\n=== Merging Statistics ===")
    print(f"Merged LMDB cache used: {cache_used}")
    print(f"Unique lemmas: {len(merged) if merged is not None else 'N/A (LMDB only)'}")
    print(f"Merged LMDB cache path: {lmdb_dir}")
    print(f"Total merging time: {elapsed:.2f} seconds\n")
    return merged, str(lmdb_dir)
