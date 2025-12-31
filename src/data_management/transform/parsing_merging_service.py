from enum import Enum
import logging
import os
import multiprocessing
from multiprocessing import Manager
from typing import Callable, List, Optional, Dict, Any, Tuple, TypedDict
from pydantic import TypeAdapter
from .data_unifier import LinguisticEntry, WordForm
from tqdm import tqdm
import multiprocessing
tqdm.set_lock(multiprocessing.RLock())
import pprint
from src.data_management.transform.cache_utils import cache_path_for_key, save_to_cache_streaming, load_from_cache_streaming, to_serializable

# Top-level parser functions for multiprocessing (must be importable/picklable)

# Absolute imports for parser runner functions
from src.data_management.sources.txt_ua_stresses.txt_stress_parser import parse_txt_to_unified_dict
from src.data_management.sources.trie_ua_stresses.trie_stress_parser import parse_trie_to_unified_dict
from src.data_management.sources.kaikki.kaikki_parser import parse_kaikki_to_unified_dict

from src.data_management.transform.cache_utils import compute_parser_hash, cache_path_for_key, save_to_cache, load_from_cache


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
    from src.data_management.sources.txt_ua_stresses.txt_stress_parser import stream_txt_to_lmdb
    if config is None:
        from .parsing_merging_service import export_config
        config = export_config.sources_configs["TXT"]
    lmdb_path, stats = stream_txt_to_lmdb(progress_callback=progress_callback, config=config)
    # ...existing code...
    return lmdb_path, stats


def run_trie_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache_streaming, to_serializable, save_to_cache_streaming
    if config is None:
        config = export_config.sources_configs["TRIE"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache_streaming(cache_key, prefix="TRIE")
    if cached:
        logging.info("[CACHE] Using streaming cache for TRIE")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    result = parse_trie_to_unified_dict(show_progress=True, progress_callback=progress_callback)
    # ...existing code...
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="TRIE")
    return result


def run_kaikki_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache_streaming, to_serializable, save_to_cache_streaming
    if config is None:
        config = export_config.sources_configs["KAIKKI"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache_streaming(cache_key, prefix="KAIKKI")
    if cached:
        logging.info("[CACHE] Using streaming cache for KAIKKI")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    input_path = config["db_path"]
    result = parse_kaikki_to_unified_dict(input_path, show_progress=True, progress_callback=progress_callback)
    # ...existing code...
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="KAIKKI")
    return result



class ParsingMergingService:
    """
    Service to run multiple parser processes concurrently and merge their outputs losslessly.
    - Accepts parser callables (functions/classes with .parse() or __call__ returning Dict[str, LinguisticEntry])
    - Runs all parsers concurrently
    - Merges all returned dictionaries under the same lemma keys, merging WordForms and stress indices losslessly
    """

def parser_worker(
    func: Callable[[Optional[Callable[[int, int], None]], Optional[ParserConfig]], Tuple[Dict[str, Any], Dict[str, Any]]],
    name: str,
    progress_queue: Any,
    result_queue: Any
) -> None:
    def progress_callback(current, total):
        progress_queue.put((name, current, total))
    import sys, os
    import psutil
    # ...existing code...
    def log_mem(stage):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"[MEMORY] [PID {os.getpid()}] {name} {stage}: {mem_mb:.2f} MB")
    log_mem('start')
    class DevNull:
        def write(self, *_): pass
        def flush(self): pass
    sys.stdout = DevNull()
    sys.stderr = DevNull()
    try:
        log_mem('before_func')
        # ...existing code...
        _, stats = func(progress_callback=progress_callback)
        log_mem('after_func')
        # ...existing code...
        # Only send stats, not large data, to avoid pickling large objects
        result_queue.put((name, None, stats, None))
        progress_queue.put((name, None, None))  # Signal finished
        log_mem('after_result_queue')
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # ...existing code...
        result_queue.put((name, None, None, f"{e}\n{tb}"))
        progress_queue.put((name, None, None))
        log_mem('exception')

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

    max_concurrent = 2
    running = []
    func_name_pairs = list(zip(parser_funcs, names))
    idx = 0
    total = len(func_name_pairs)

    # Use full parser names for informative progress bar descriptions
    bar_descs = [f"Parsing [{name}]" for name in names]
    pbar_list = []
    for i, desc in enumerate(bar_descs):
        pbar_list.append(tqdm(total=None, desc=desc, position=i, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', leave=True, dynamic_ncols=True))
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
                pbar.n = totals[idx_]
                pbar.refresh()
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

    for pbar in pbar_list:
        pbar.close()

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
    from src.data_management.transform.cache_utils import load_from_cache_streaming
    loaded_results = []
    for name in names:
        config = export_config.sources_configs[name]
        loaded = load_from_cache_streaming(
            compute_parser_hash(config["parser_path"], config["db_path"]),
            prefix=name
        )
        loaded_results.append(loaded)
    return loaded_results, stats_list


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
                pbar.update(1)
    logger.info(f"Merging complete. Unique lemmas: {len(merged)}")
    return merged


# Example main function for running txt and trie parsers and merging results

def compute_merged_cache_key(lmdb_dirs):
    """
    Compute a merged cache key by hashing the LMDB data.mdb files for each source.
    """
    import hashlib
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
    import glob
    import shutil
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "cache"))
    cache_keys = [compute_parser_hash(export_config.sources_configs[name]["parser_path"], export_config.sources_configs[name]["db_path"]) for name in enabled_names]
    def get_lmdb_dir_for_key(key, prefix):
        return os.path.abspath(os.path.join(cache_dir, f"{prefix}_{key}_lmdb"))
    lmdb_dirs = [get_lmdb_dir_for_key(key, name) for key, name in zip(cache_keys, enabled_names)]
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

    import time
    start_time = time.time()
    _tqdm.write("\n=== Parsing Summary (All Parsers) ===")
    for name, stats in stats_list:
        _tqdm.write(f"Parser {name} stats: {stats}")
    total_elapsed = time.time() - start_time
    _tqdm.write(f"Total concurrent parsing time: {total_elapsed:.2f} seconds")


    # --- Merged cache logic ---
    # (No recomputation of merged_cache_key or merged_sqlite_path here; use the ones from above)


    # --- New: Streaming, disk-backed merge to LMDB, then export ---
    from src.data_management.transform.merger import merge_caches_and_save_lmdb
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
            from src.data_management.transform.merger import merge_caches_and_save
            # Export from the merged LMDB only, using the exact path returned above
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
            from src.data_management.transform.merger import LMDBExporter, LMDBExportConfig
            from pathlib import Path
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
            import sqlite3
            # Use the actual path returned by the export step
            sql_db_path = sql_db_path if 'sql_db_path' in locals() else merged_sqlite_path
            if not sql_db_path or not os.path.exists(sql_db_path):
                _tqdm.write(f"[SQLITE] DB not found at {sql_db_path}")
            else:
                conn = sqlite3.connect(sql_db_path)
                cur = conn.cursor()
                # Debug: print row count and a sample
                cur.execute("SELECT COUNT(*) FROM word_form")
                count = cur.fetchone()[0]
                _tqdm.write(f"[SQLITE] word_form row count: {count}")
                cur.execute("SELECT * FROM word_form LIMIT 5")
                sample = cur.fetchall()
                _tqdm.write(f"[SQLITE] word_form sample rows: {sample}")
                for word in export_config.query_words:
                    # Query by lemma or form, print stress_indices_json
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
