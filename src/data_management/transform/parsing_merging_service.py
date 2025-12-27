# --- Export configuration ---
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

# --- Export and Query Config ---
export_config = ExportConfig(
    format=ExportFormat.SQL,  # Change to ExportFormat.LMDB for LMDB
    query_words=["замок"]    # Add more words to query as needed, or leave empty for no test queries
    # sources_configs is defaulted inside ExportConfig
)
class ParserConfig(TypedDict):
    parser_func: str
    parser_path: str
    db_path: str
# --- Export and Query Config ---
    format=ExportFormat.SQL,  # Change to ExportFormat.LMDB for LMDB
    query_words=["замок"]    # Add more words to query as needed, or leave empty for no test queries
    # sources_configs is defaulted inside ExportConfig

SOURCES_CONFIGS: Dict[str, ParserConfig] = {
    # "TXT": {
    #     "parser_func": "run_txt_parser",
    #     "parser_path": "src/data_management/sources/txt_ua_stresses/txt_stress_parser.py",
    #     "db_path": "src/data_management/sources/txt_ua_stresses/ua_word_stress_dictionary.txt",
    # },
    # "TRIE": {
    #     "parser_func": "run_trie_parser",
    #     "parser_path": "src/data_management/sources/trie_ua_stresses/trie_stress_parser.py",
    #     "db_path": "src/data_management/sources/trie_ua_stresses/stress.trie",
    # },
    "KAIKKI": {
        "parser_func": "run_kaikki_parser",
        "parser_path": "src/data_management/sources/kaikki/kaikki_parser.py",
        "db_path": "src/data_management/sources/kaikki/kaikki.org-dictionary-Ukrainian.jsonl",
    },
}



def run_txt_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache_streaming, to_serializable, save_to_cache_streaming
    if config is None:
        config = export_config.sources_configs["TXT"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    # Check for streaming cache
    cached = load_from_cache_streaming(cache_key, prefix="TXT")
    if cached:
        print(f"[CACHE] Using streaming cache for TXT")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_txt_parser called")
    result = parse_txt_to_unified_dict(show_progress=True, progress_callback=progress_callback)
    print(f"[DEBUG] [PID {os.getpid()}] run_txt_parser finished")
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="TXT")
    return result


def run_trie_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache_streaming, to_serializable, save_to_cache_streaming
    if config is None:
        config = export_config.sources_configs["TRIE"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache_streaming(cache_key, prefix="TRIE")
    if cached:
        print(f"[CACHE] Using streaming cache for TRIE")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_trie_parser called")
    result = parse_trie_to_unified_dict(show_progress=True, progress_callback=progress_callback)
    print(f"[DEBUG] [PID {os.getpid()}] run_trie_parser finished")
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="TRIE")
    return result


def run_kaikki_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache_streaming, to_serializable, save_to_cache_streaming
    if config is None:
        config = export_config.sources_configs["KAIKKI"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache_streaming(cache_key, prefix="KAIKKI")
    if cached:
        print(f"[CACHE] Using streaming cache for KAIKKI")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_kaikki_parser called")
    input_path = config["db_path"]
    result = parse_kaikki_to_unified_dict(input_path, show_progress=True, progress_callback=progress_callback)
    print(f"[DEBUG] [PID {os.getpid()}] run_kaikki_parser finished")
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
    print(f"[DEBUG] [PID {os.getpid()}] parser_worker for {name} starting")
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
        print(f"[DEBUG] [PID {os.getpid()}] Worker for {name} started.")
        _, stats = func(progress_callback=progress_callback)
        log_mem('after_func')
        print(f"[DEBUG] [PID {os.getpid()}] Worker for {name} finished.")
        # Only send stats, not large data, to avoid pickling large objects
        result_queue.put((name, None, stats, None))
        progress_queue.put((name, None, None))  # Signal finished
        log_mem('after_result_queue')
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[DEBUG] [PID {os.getpid()}] Worker for {name} exception: {e}\n{tb}")
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
        print(f"[DEBUG] Running single parser '{name}' directly (no multiprocessing)...")
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

    bar_descs = [name[:8].upper() for name in names]
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
            print(f"[DEBUG] Launching process for {name}...")
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
            print(f"❌ Parser '{name}' failed: {err}")
            errors.append((name, err))
        else:
            # For large results, do not pass data through the queue, just pass None and rely on cache
            results.append(None)
            stats_list.append((name, stats))

    if errors:
        for name, err in errors:
            print(f"❌ Parser '{name}' failed: {err}")
        raise RuntimeError(f"One or more parsers failed: {errors}")

    # After all processes, load results from cache
    from src.data_management.transform.cache_utils import load_from_cache_streaming
    loaded_results = []
    for name in names:
        loaded = load_from_cache_streaming(
            compute_parser_hash(SOURCES_CONFIGS[name]["parser_path"], SOURCES_CONFIGS[name]["db_path"]),
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

def compute_merged_cache_key(cache_paths):
    import hashlib
    h = hashlib.sha256()
    for path in cache_paths:
        with open(path, 'rb') as f:
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
    results, stats_list = run_parsers_concurrently_mp(
        enabled_funcs,
        enabled_names
    )
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
    cache_keys = [compute_parser_hash(export_config.sources_configs[name]["parser_path"], export_config.sources_configs[name]["db_path"]) for name in enabled_names]
    cache_paths = [cache_path_for_key(key, prefix=name) for key, name in zip(cache_keys, enabled_names)]
    merged_cache_key = compute_merged_cache_key(cache_paths)
    merged_cache_path = cache_path_for_key(merged_cache_key, prefix="MERGED")
    import glob
    import shutil
    cache_dir = os.path.dirname(merged_cache_path)
    merged_lmdb_pattern = os.path.join(cache_dir, "MERGED_*_lmdb")
    merged_sqlite_pattern = os.path.join(cache_dir, "MERGED_*.sqlite3")
    # Remove old merged cache folders/files if hashes do not match
    for lmdb_folder in glob.glob(merged_lmdb_pattern):
        if merged_cache_key not in lmdb_folder:
            try:
                shutil.rmtree(lmdb_folder)
                _tqdm.write(f"[CLEANUP] Deleted old merged LMDB cache: {lmdb_folder}")
            except Exception as e:
                _tqdm.write(f"[CLEANUP] Failed to delete {lmdb_folder}: {e}")
    for sqlite_file in glob.glob(merged_sqlite_pattern):
        if merged_cache_key not in sqlite_file:
            try:
                os.remove(sqlite_file)
                _tqdm.write(f"[CLEANUP] Deleted old merged SQLite DB: {sqlite_file}")
            except Exception as e:
                _tqdm.write(f"[CLEANUP] Failed to delete {sqlite_file}: {e}")

    merged = None
    if os.path.exists(merged_cache_path) and os.path.getsize(merged_cache_path) > 0:
        _tqdm.write(f"[CACHE] Using merged cache at {merged_cache_path}")
        merged = load_from_cache_streaming(merged_cache_key, prefix="MERGED")
    else:
        _tqdm.write("\n=== Merging Dictionaries ===")
        logger = logging.getLogger("merging")
        logger.info("Starting merging step...")
        merged = merge_linguistic_dicts(results)
        logger.info(f"Merging complete. Unique lemmas: {len(merged)}")
        _tqdm.write(f"Merged unique lemmas: {len(merged)}")

        # --- Export Step ---
        if export_config.format == ExportFormat.LMDB:
            _tqdm.write("[LMDB] Starting LMDB export...")
            logger.info("[LMDB] Starting LMDB export...")
            try:
                from src.data_management.transform.merger import LMDBExporter, LMDBExportConfig
                from pathlib import Path
                lmdb_dir = merged_cache_path + "_lmdb"
                config = LMDBExportConfig(db_path=Path(lmdb_dir), overwrite=True)
                exporter = LMDBExporter(config)
                logger.info(f"[LMDB] Streaming export to {lmdb_dir} ...")
                _tqdm.write(f"[LMDB] Streaming export to {lmdb_dir} ...")
                from src.data_management.transform.cache_utils import to_serializable
                def merged_iter():
                    for k, v in merged.items():
                        yield k, to_serializable(v)
                exporter.export_streaming(merged_iter(), total=len(merged))
                logger.info(f"[LMDB] Export complete. Verifying...")
                _tqdm.write(f"[LMDB] Export complete. Verifying...")
                if not os.path.exists(lmdb_dir) or not os.listdir(lmdb_dir):
                    logger.error(f"[LMDB] ERROR: LMDB directory {lmdb_dir} not created or empty!")
                    _tqdm.write(f"[LMDB] ERROR: LMDB directory {lmdb_dir} not created or empty!")
                    raise RuntimeError(f"[LMDB] Export failed: {lmdb_dir} not created or empty!")
                logger.info(f"[LMDB] LMDB directory {lmdb_dir} created successfully.")
                _tqdm.write(f"[LMDB] LMDB directory {lmdb_dir} created successfully.")
            except Exception as e:
                logger.error(f"[LMDB] Exception during LMDB export: {e}")
                _tqdm.write(f"[LMDB] Exception during LMDB export: {e}")
                raise
        else:
            _tqdm.write("[SQLITE] Starting SQLite export...")
            logger.info("[SQLITE] Starting SQLite export...")
            try:
                from src.data_management.transform.merger import merge_caches_and_save
                # merge_caches_and_save returns (merged, db_path)
                _, sql_db_path = merge_caches_and_save(enabled_names, merged_prefix="MERGED")
                logger.info(f"[SQLITE] SQLite export complete: {sql_db_path}")
                _tqdm.write(f"[SQLITE] SQLite export complete: {sql_db_path}")
            except Exception as e:
                logger.error(f"[SQLITE] Exception during SQLite export: {e}")
                _tqdm.write(f"[SQLITE] Exception during SQLite export: {e}")
                raise

    # --- Query exported data from LMDB or SQLite ---
    if export_config.query_words:
        _tqdm.write("\n=== Querying Exported Data ===")
        if export_config.format == ExportFormat.LMDB:
            _tqdm.write("[QUERY] LMDB export: querying keys...")
            from src.data_management.transform.merger import LMDBExporter, LMDBExportConfig
            from pathlib import Path
            lmdb_dir = merged_cache_path + "_lmdb"
            config = LMDBExportConfig(db_path=Path(lmdb_dir), overwrite=False)
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
            sql_db_path = None
            # Find the merged sqlite path (should match export)
            cache_dir = os.path.dirname(merged_cache_path)
            for fname in os.listdir(cache_dir):
                if fname.endswith(".sqlite3") and merged_cache_key in fname:
                    sql_db_path = os.path.join(cache_dir, fname)
                    break
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
                    # Query by lemma or form
                    cur.execute("SELECT * FROM word_form WHERE lemma=? OR form=?", (word, word))
                    rows = cur.fetchall()
                    if rows:
                        _tqdm.write(f"[SQLITE] {word}: {rows}")
                    else:
                        _tqdm.write(f"[SQLITE] {word}: NOT FOUND")
                cur.close()
                conn.close()

if __name__ == "__main__":
    main()
