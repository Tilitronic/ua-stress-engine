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

SOURCES_CONFIGS: Dict[str, ParserConfig] = {
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



def run_txt_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache, save_to_cache
    if config is None:
        config = SOURCES_CONFIGS["TXT"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache(cache_key)
    if cached:
        print(f"[CACHE] Using cached result for TXT")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_txt_parser called")
    from src.data_management.transform.cache_utils import to_serializable, save_to_cache_streaming
    result = parse_txt_to_unified_dict(show_progress=True, progress_callback=progress_callback)
    print(f"[DEBUG] [PID {os.getpid()}] run_txt_parser finished")
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="TXT")
    return result


def run_trie_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache, save_to_cache
    if config is None:
        config = SOURCES_CONFIGS["TRIE"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache(cache_key)
    if cached:
        print(f"[CACHE] Using cached result for TRIE")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_trie_parser called")
    from src.data_management.transform.cache_utils import to_serializable, save_to_cache_streaming
    result = parse_trie_to_unified_dict(show_progress=True, progress_callback=progress_callback)
    print(f"[DEBUG] [PID {os.getpid()}] run_trie_parser finished")
    save_to_cache_streaming(to_serializable(result[0]), cache_key, prefix="TRIE")
    return result


def run_kaikki_parser(progress_callback: Optional[Callable[[int, int], None]] = None, config: Optional[ParserConfig] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.data_management.transform.cache_utils import compute_parser_hash, load_from_cache, save_to_cache
    if config is None:
        config = SOURCES_CONFIGS["KAIKKI"]
    cache_key = compute_parser_hash(config["parser_path"], config["db_path"])
    cached = load_from_cache(cache_key)
    if cached:
        print(f"[CACHE] Using cached result for KAIKKI")
        return cached, {"cache_used": True, "unique_lemmas": len(cached)}
    print(f"[DEBUG] [PID {os.getpid()}] run_kaikki_parser called")
    input_path = config["db_path"]
    from src.data_management.transform.cache_utils import to_serializable, save_to_cache_streaming
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
    with tqdm(total=total_lemmas, desc="Merging", leave=True, ncols=80) as pbar:
        for d in dicts:
            for lemma, entry in d.items():
                # Use Pydantic type adapter for validation
                entry_obj = TypeAdapter(LinguisticEntry).validate_python(entry) if isinstance(entry, dict) else entry
                if lemma not in merged:
                    merged[lemma] = entry_obj.model_copy(deep=True)
                else:
                    # Merge forms (deduplicate by all fields)
                    existing_forms = merged[lemma].forms
                    for wf in entry_obj.forms:
                        if not any(wf.model_dump() == ef.model_dump() for ef in existing_forms):
                            existing_forms.append(wf)
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
def main():

    from tqdm import tqdm as _tqdm
    # Ensure stanza models are downloaded before multiprocessing
    try:
        import stanza
        stanza.download('uk')
    except Exception as e:
        _tqdm.write(f"[WARN] Could not download stanza model: {e}")
    _tqdm.write("\n=== Parsing and Merging Stress Dictionaries ===\n")
    # Run all parsers concurrently using multiprocessing with progress bars
    results, stats_list = run_parsers_concurrently_mp(
        [run_txt_parser, run_trie_parser, run_kaikki_parser],
        ["TXT", "TRIE", "KAIKKI"]
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

    # Merging step with visible progress/logs
    _tqdm.write("\n=== Merging Dictionaries ===")
    logger = logging.getLogger("merging")
    logger.info("Starting merging step...")
    merged = merge_linguistic_dicts(results)
    logger.info(f"Merging complete. Unique lemmas: {len(merged)}")
    _tqdm.write(f"Merged unique lemmas: {len(merged)}")

    # Pretty print a few sample entries
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    for key in ["замок", "блоха", "помилка"]:
        _tqdm.write(f"Merged entry for lemma: '{key}'")
        entry = merged.get(key)
        if not entry:
            _tqdm.write("  Not found in merged dictionary.")
            continue
        _tqdm.write(f"\033[1;36m{key}\033[0m:")
        _tqdm.write(f"\033[0;37m{pp.pformat(entry.model_dump())}\033[0m")

if __name__ == "__main__":
    main()
