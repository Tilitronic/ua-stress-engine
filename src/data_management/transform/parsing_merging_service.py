
import os
import glob
import shutil
import logging
import multiprocessing
from multiprocessing import Manager, Process
from enum import Enum
from typing import Callable, List, Optional, Dict, Any, Tuple, TypedDict
from tqdm import tqdm
import time
import hashlib
import sys
import json
import sqlite3
from pathlib import Path

# Project-specific imports
from .data_unifier import LinguisticEntry, WordForm
from src.data_management.transform.cache_utils import (
    save_to_cache_streaming, load_from_cache_streaming, to_serializable,
)
from src.data_management.transform.cache_utils import compute_parser_hash
from src.data_management.sources.txt_ua_stresses.txt_stress_parser import stream_txt_to_lmdb
from src.data_management.sources.trie_ua_stresses.trie_stress_parser import stream_trie_to_lmdb
from src.data_management.sources.kaikki.kaikki_parser import stream_kaikki_to_lmdb
from src.data_management.transform.merger import merge_caches_and_save_lmdb, merge_caches_and_save, LMDBExporter, LMDBExportConfig



class ParserConfig(TypedDict):
    parser_func: str
    parser_path: str
    db_path: str

# --- Pipeline/Export/Query Configs ---
class ExportFormat(str, Enum):
    SQL = 'sql'
    LMDB = 'lmdb'


class ExportConfig:
    def __init__(self, format: ExportFormat = ExportFormat.SQL, query_words: Optional[List[str]] = None, sources_configs: Optional[Dict[str, 'ParserConfig']] = None):
        self.format = format
        self.query_words = query_words or []
        # Default sources_configs if not provided
        self.sources_configs: Dict[str, ParserConfig] = sources_configs or {
            "TXT": {
                "parser_func": "run_txt_parser",
                "parser_path": "src/data_management/sources/txt_ua_stresses/txt_stress_parser.py",
                "db_path": "src/data_management/sources/txt_ua_stresses/ua_word_stress_dictionary.txt",
            },
            "TRIE": {
                "parser_func": "run_trie_parser",
                "parser_path": "src/data_management/sources/trie_ua_stresses/trie_stress_parser.py",
                "db_path": "src/data_management/sources/trie_ua_stresses/stress.trie", 
            },
            "KAIKKI": {
                "parser_func": "run_kaikki_parser",
                "parser_path": "src/data_management/sources/kaikki/kaikki_parser.py",
                "db_path": "src/data_management/sources/kaikki/kaikki.org-dictionary-Ukrainian.jsonl",
            },
        }
        # Informative log for sources_configs
        logging.info("[CONFIG] sources_configs initialized:")
        for name, cfg in self.sources_configs.items():
            logging.info(f"[CONFIG] Source: {name}, Config: {cfg}")

# --- Export and Query Config ---
export_config = ExportConfig(
    format=ExportFormat.SQL,  # Change to ExportFormat.LMDB for LMDB
    query_words=["замок"]    # Add more words to query as needed, or leave empty for no test queries
    # sources_configs is defaulted inside ExportConfig
)



def run_txt_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Calls the TXT parser's own streaming-to-LMDB logic and returns the LMDB path and stats.
    No TXT-specific logic here; all details are encapsulated in the TXT parser module.
    """
    if config is None:
        config = export_config.sources_configs["TXT"]
    lmdb_path, stats = stream_txt_to_lmdb(progress_callback=progress_callback, config=config)
    return lmdb_path, stats


def run_trie_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if config is None:
        config = export_config.sources_configs["TRIE"]
    lmdb_path, stats = stream_trie_to_lmdb(progress_callback=progress_callback, config=config)
    return lmdb_path, stats


def run_kaikki_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calls the KAIKKI parser's streaming-to-LMDB logic and returns the LMDB path and stats.
    """
    if config is None:
        config = export_config.sources_configs["KAIKKI"]
    # Compute cache key and LMDB output path
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
    lmdb_dir = os.path.abspath(os.path.join(cache_dir, f"KAIKKI_{cache_key}_lmdb"))
    os.makedirs(lmdb_dir, exist_ok=True)
    lmdb_path = lmdb_dir
    if os.path.exists(os.path.join(lmdb_path, "data.mdb")):
        stats = {"cache_used": True, "lmdb_path": lmdb_path}
        return lmdb_path, stats
    input_path = config["db_path"]
    stream_kaikki_to_lmdb(input_path=input_path, lmdb_path=lmdb_path, show_progress=True, progress_callback=progress_callback)
    stats = {"cache_used": False, "lmdb_path": lmdb_path}
    return lmdb_path, stats



def parser_worker(
    func: Callable[[Optional[Callable[[int, int], None]], Optional[ParserConfig]], Tuple[Dict[str, Any], Dict[str, Any]]],
    name: str,
    progress_queue: Any,
    result_queue: Any
) -> None:
    def progress_callback(current, total):
        progress_queue.put((name, current, total))
    # ...existing code...
    # Memory tracking removed as per optimization
    class DevNull:
        def write(self, *_): pass
        def flush(self): pass
    sys.stdout = DevNull()
    sys.stderr = DevNull()
    try:
        # ...existing code...
        _, stats = func(progress_callback=progress_callback)
        # ...existing code...
        # Only send stats, not large data, to avoid pickling large objects
        result_queue.put((name, None, stats, None))
        progress_queue.put((name, None, None))  # Signal finished
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # ...existing code...
        result_queue.put((name, None, None, f"{e}\n{tb}"))
        progress_queue.put((name, None, None))

def run_parsers_concurrently_mp(
    parser_funcs: List[Callable[[Optional[Callable[[int, int], None]], Optional[ParserConfig]], Tuple[Dict[str, Any], Dict[str, Any]]]],
    names: List[str]
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]:
    """
    Run all parser callables concurrently in separate processes and return their results and stats, with progress reporting via queue.

    NOTE ON MEMORY USAGE AND MULTIPROCESSING:
    - Each parser process may consume significant memory, especially with large datasets.
    - The Manager().Queue() used for inter-process communication must pickle/unpickle all data sent through it.
    - Sending very large objects (e.g., huge dictionaries) through the queue can cause MemoryError or excessive memory usage.
    - Limiting the number of concurrent processes (max_concurrent) is critical to avoid exhausting system memory.
    - If you still encounter MemoryError, consider chunking results, using disk-backed queues, or further reducing concurrency.
    - Always monitor memory usage (see memory debug logs in parser_worker).
    """
    # If only one parser, run it directly (no multiprocessing)
    if len(parser_funcs) == 1:
        func = parser_funcs[0]
        name = names[0]
        logging.info(f"[PARSER] Running single parser '{name}' directly (no multiprocessing)...")
        data, stats = func()
        results = [data]
        stats_list = [(name, stats)]
        return results, stats_list

    # Otherwise, use multiprocessing as before
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    processes = []

    max_concurrent = len(parser_funcs)  # Allow all to run in parallel
    running = []
    func_name_pairs = list(zip(parser_funcs, names))
    idx = 0
    total = len(func_name_pairs)

    # Use full parser names for informative progress bar descriptions
    bar_descs = [f"[{name}] Parse→Cache" for name in names]
    pbar_list = []
    for i, desc in enumerate(bar_descs):
        pbar_list.append(tqdm(total=1, desc=desc, position=i, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]', leave=True, dynamic_ncols=True, unit='line'))
    finished_count = 0
    expected_finishes = len(names)
    totals = [None for _ in names]
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Start up to max_concurrent processes
    while idx < total or running:
        # Start new processes if under limit
        while len(running) < max_concurrent and idx < total:
            func, name = func_name_pairs[idx]
            logging.info(f"[PARSER] Launching process for {name}...")
            p = multiprocessing.Process(target=parser_worker, args=(func, name, progress_queue, result_queue))
            p.start()
            running.append((p, name))
            idx += 1
        # Wait for a progress update
        name, current, total_ = progress_queue.get()
        idx_ = name_to_idx[name]
        pbar = pbar_list[idx_]
        if current is None:
            finished_count += 1
            if totals[idx_] is not None:
                # Parser sent progress updates, complete the bar normally
                pbar.n = totals[idx_]
                pbar.refresh()
                pbar.close()
            else:
                # Parser used cache (no progress updates), remove bar completely
                pbar.leave = False
                pbar.close()
            # Remove finished process
            for i, (proc, n) in enumerate(running):
                if n == name:
                    proc.join()
                    running.pop(i)
                    break
        else:
            if totals[idx_] is None and total_:
                totals[idx_] = total_
                pbar.total = total_
                pbar.refresh()
            pbar.n = current
            pbar.refresh()

    results = []
    stats_list = []
    errors = []
    for _ in range(expected_finishes):
        name, data, stats, err = result_queue.get()
        if err:
            logging.error(f"[ERROR] Parser '{name}' failed: {err}")
            errors.append((name, err))
        else:
            # For large results, do not pass data through the queue, just pass None and rely on cache
            results.append(None)
            stats_list.append((name, stats))

    if errors:
        for name, err in errors:
            logging.error(f"[ERROR] Parser '{name}' failed: {err}")
        raise RuntimeError(f"One or more parsers failed: {errors}")

    # After all processes, load results from cache
    loaded_results = []
    for name in names:
        config = export_config.sources_configs[name]
        loaded = load_from_cache_streaming(
            compute_parser_hash(config["parser_path"], config["db_path"]),
            prefix=name
        )
        loaded_results.append(loaded)
    return loaded_results, stats_list



# Example main function for running txt and trie parsers and merging results

def compute_merged_cache_key(lmdb_dirs):
    """
    Compute a merged cache key by hashing the LMDB data.mdb files for each source.
    """
    h = hashlib.sha256()
    for lmdb_dir in lmdb_dirs:
        data_mdb = os.path.join(lmdb_dir, 'data.mdb')
        if not os.path.exists(data_mdb):
            raise FileNotFoundError(f"LMDB data file not found: {data_mdb}")
        with open(data_mdb, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()

def main():
    from tqdm import tqdm as _tqdm
    # Ensure stanza models are downloaded before multiprocessing
    try:
        import stanza
        stanza.download('uk')
    except Exception as e:
        _tqdm.write(f"[WARN] Could not download stanza model: {e}")
    _tqdm.write("\n=== Parsing and Merging Stress Dictionaries ===\n")
    # Dynamically select enabled sources and their parser functions
    parser_func_map = {
        "TXT": run_txt_parser,
        "TRIE": run_trie_parser,
        "KAIKKI": run_kaikki_parser,
    }
    enabled_names = list(export_config.sources_configs.keys())
    enabled_funcs = [parser_func_map[name] for name in enabled_names]

    # Start timer for total parsing time
    start_time = time.time()
    
    # --- Ensure all source LMDB caches exist, rerun missing ones if needed ---
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
    cache_keys = [compute_parser_hash(export_config.sources_configs[name]["parser_path"], export_config.sources_configs[name]["db_path"]) for name in enabled_names]
    # Use distinct prefix for merged LMDB
    def get_lmdb_dir_for_key(key, prefix):
        return os.path.abspath(os.path.join(cache_dir, f"{prefix}_{key}_lmdb"))
    lmdb_dirs = [get_lmdb_dir_for_key(key, name) for key, name in zip(cache_keys, enabled_names)]

    # Check for missing LMDBs and rerun parsers for those sources
    missing = [i for i, lmdb_dir in enumerate(lmdb_dirs) if not (os.path.exists(lmdb_dir) and os.path.exists(os.path.join(lmdb_dir, 'data.mdb')))]
    if missing:
        logging.info(f"[CACHE] Missing LMDB caches for: {[enabled_names[i] for i in missing]}. Re-running those parsers...")
        # Only run missing parsers
        missing_funcs = [enabled_funcs[i] for i in missing]
        missing_names = [enabled_names[i] for i in missing]
        _results, _stats_list = run_parsers_concurrently_mp(missing_funcs, missing_names)
        # Update lmdb_dirs after rerun
        cache_keys = [compute_parser_hash(export_config.sources_configs[name]["parser_path"], export_config.sources_configs[name]["db_path"]) for name in enabled_names]
        lmdb_dirs = [get_lmdb_dir_for_key(key, name) for key, name in zip(cache_keys, enabled_names)]

    # Now run all parsers (to get stats/results for all, but will use cache for existing)
    results, stats_list = run_parsers_concurrently_mp(
        enabled_funcs,
        enabled_names
    )

    # Now that LMDBs exist, compute merged cache key and handle merged cache cleanup
    merged_cache_key = compute_merged_cache_key(lmdb_dirs)
    merged_lmdb_dir = os.path.join(cache_dir, f"MERGEDLMDB_{merged_cache_key}_lmdb")
    merged_sqlite_path = os.path.join(cache_dir, f"MERGEDSQL_{merged_cache_key}.sqlite3")

    # Only purge old merged caches if creating a new merged cache
    if not (os.path.exists(merged_lmdb_dir) and os.path.exists(os.path.join(merged_lmdb_dir, 'data.mdb'))):
        merged_lmdb_pattern = os.path.join(cache_dir, "MERGEDLMDB_*_lmdb")
        for lmdb_folder in glob.glob(merged_lmdb_pattern):
            if merged_cache_key not in lmdb_folder:
                try:
                    shutil.rmtree(lmdb_folder)
                    logging.info(f"[CLEANUP] Deleted old merged LMDB cache: {lmdb_folder}")
                except Exception as e:
                    logging.warning(f"[CLEANUP] Failed to delete {lmdb_folder}: {e}")
    if not (os.path.exists(merged_sqlite_path) and os.path.getsize(merged_sqlite_path) > 0):
        merged_sqlite_pattern = os.path.join(cache_dir, "MERGEDSQL_*.sqlite3")
        for sqlite_file in glob.glob(merged_sqlite_pattern):
            if merged_cache_key not in sqlite_file:
                try:
                    os.remove(sqlite_file)
                    logging.info(f"[CLEANUP] Deleted old merged SQLite DB: {sqlite_file}")
                except Exception as e:
                    logging.warning(f"[CLEANUP] Failed to delete {sqlite_file}: {e}")
    # If results are LMDB paths (str), load dicts from LMDB for merging
    def load_dict_from_lmdb(lmdb_dir):
        import lmdb
        import msgpack
        d = {}
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                d[k.decode('utf-8')] = msgpack.unpackb(v, raw=False)
        env.close()
        return d
    # Only convert if results are str (LMDB dir), otherwise pass as is
    if results and isinstance(results[0], str):
        results = [load_dict_from_lmdb(lmdb_dir) for lmdb_dir in results]
    # Setup logging for merging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    total_elapsed = time.time() - start_time
    _tqdm.write("\n=== Parsing Summary (All Parsers) ===")
    for name, stats in stats_list:
        _tqdm.write(f"Parser {name} stats: {stats}")
    _tqdm.write(f"Total concurrent parsing time: {total_elapsed:.2f} seconds")

    # --- Merged cache logic ---
    # (No recomputation of merged_cache_key or merged_sqlite_path here; use the ones from above)

    # --- New: Streaming, disk-backed merge to LMDB, then export ---
    _tqdm.write("\n=== Streaming Merge to Merged LMDB ===")
    try:
        merged_lmdb_dir_actual = merge_caches_and_save_lmdb(lmdb_dirs, merged_prefix="MERGEDLMDB")
        _tqdm.write(f"[MERGE] Merged LMDB created at: {merged_lmdb_dir_actual}")
    except Exception as e:
        _tqdm.write(f"[MERGE] Exception during streaming merge: {e}")
        raise

    # --- Export Step ---
    if export_config.format == ExportFormat.LMDB:
        _tqdm.write("[LMDB] Merged LMDB is ready for use.")
    else:
        _tqdm.write("[SQLITE] Starting SQLite export from merged LMDB...")
        # Robust existence check for merged LMDB directory
        if not (os.path.exists(merged_lmdb_dir_actual) and os.path.exists(os.path.join(merged_lmdb_dir_actual, 'data.mdb'))):
            _tqdm.write(f"[ERROR] Merged LMDB directory not found: {merged_lmdb_dir_actual}\nAborting export. Please check the merge step for errors.")
            raise FileNotFoundError(f"Merged LMDB directory not found: {merged_lmdb_dir_actual}")
        try:
            _, sql_db_path = merge_caches_and_save([merged_lmdb_dir_actual], export_config, merged_prefix="MERGEDSQL")
            _tqdm.write(f"[SQLITE] SQLite export complete: {sql_db_path}")
        except Exception as e:
            _tqdm.write(f"[SQLITE] Exception during SQLite export: {e}")
            raise

    # --- Query exported data from LMDB or SQLite ---
    if export_config.query_words:
        _tqdm.write("\n=== Querying Exported Data ===")
        if export_config.format == ExportFormat.LMDB:
            _tqdm.write("[QUERY] LMDB export: querying keys...")
            config = LMDBExportConfig(db_path=Path(merged_lmdb_dir_actual), overwrite=False)
            import lmdb
            import msgpack
            env = lmdb.open(str(config.db_path), readonly=True, lock=False)
            with env.begin() as txn:
                for word in export_config.query_words:
                    val = txn.get(word.encode("utf-8"))
                    if val:
                        obj = msgpack.unpackb(val, raw=False)
                        _tqdm.write(f"[LMDB] {word}: {pprint.pformat(obj, width=120)}")
                    else:
                        _tqdm.write(f"[LMDB] {word}: NOT FOUND")
            env.close()
        else:
            _tqdm.write("[QUERY] SQLite export: querying by lemma or form...")
            sql_db_path = sql_db_path if 'sql_db_path' in locals() else merged_sqlite_path
            if not sql_db_path or not os.path.exists(sql_db_path):
                _tqdm.write(f"[SQLITE] DB not found at {sql_db_path}")
            else:
                conn = sqlite3.connect(sql_db_path)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM word_form")
                count = cur.fetchone()[0]
                _tqdm.write(f"[SQLITE] word_form row count: {count}")
                cur.execute("SELECT * FROM word_form LIMIT 5")
                sample = cur.fetchall()
                _tqdm.write(f"[SQLITE] word_form sample rows: {sample}")
                for word in export_config.query_words:
                    cur.execute("SELECT id, form, lemma, pos, stress_indices_json FROM word_form WHERE lemma=? OR form=?", (word, word))
                    rows = cur.fetchall()
                    if rows:
                        for row in rows:
                            wid, form, lemma, pos, stress_indices_json = row
                            try:
                                stress_indices = json.loads(stress_indices_json) if stress_indices_json else []
                            except Exception:
                                stress_indices = stress_indices_json
                            _tqdm.write(f"[SQLITE] {word}: form={form}, stress_indices={stress_indices}, lemma={lemma}, pos={pos}")
                    else:
                        _tqdm.write(f"[SQLITE] {word}: NOT FOUND")
                cur.close()
                conn.close()

if __name__ == "__main__":
    main()
