


import logging
import multiprocessing
from multiprocessing import Manager
from typing import Callable, List, Optional
from pydantic import TypeAdapter
from .data_unifier import LinguisticEntry, WordForm
from tqdm import tqdm
import multiprocessing
tqdm.set_lock(multiprocessing.RLock())
import pprint

# Top-level parser functions for multiprocessing (must be importable/picklable)
def txt_parser(progress_callback=None):
    import importlib
    txt_parser_mod = importlib.import_module("src.data_management.sources.txt_ua_stresses.txt_stress_parser")
    return txt_parser_mod.parse_txt_to_unified_dict(show_progress=True, progress_callback=progress_callback)

def trie_parser(progress_callback=None):
    import importlib
    trie_parser_mod = importlib.import_module("src.data_management.sources.trie_ua_stresses.trie_stress_parser")
    return trie_parser_mod.parse_trie_to_unified_dict(show_progress=True, progress_callback=progress_callback)

class ParsingMergingService:
    """
    Service to run multiple parser processes concurrently and merge their outputs losslessly.
    - Accepts parser callables (functions/classes with .parse() or __call__ returning Dict[str, LinguisticEntry])
    - Runs all parsers concurrently
    - Merges all returned dictionaries under the same lemma keys, merging WordForms and stress indices losslessly
    """

def parser_worker(func, name, progress_queue, result_queue):
    def progress_callback(current, total):
        progress_queue.put((name, current, total))
    # Suppress all stdout/stderr in child process during parsing
    import sys, os
    class DevNull:
        def write(self, *_): pass
        def flush(self): pass
    sys.stdout = DevNull()
    sys.stderr = DevNull()
    try:
        data, stats = func(progress_callback=progress_callback)
        result_queue.put((name, data, stats, None))
        progress_queue.put((name, None, None))  # Signal finished
    except Exception as e:
        result_queue.put((name, None, None, str(e)))
        progress_queue.put((name, None, None))

def run_parsers_concurrently_mp(parser_funcs: List[Callable], names: List[str]) -> (List[dict], List[dict]):
    """Run all parser callables concurrently in separate processes and return their results and stats, with progress reporting via queue."""
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    processes = []

    for func, name in zip(parser_funcs, names):
        p = multiprocessing.Process(target=parser_worker, args=(func, name, progress_queue, result_queue))
        p.start()
        processes.append(p)

    # Use fixed order and short descriptions for bars, scalable for N processes
    bar_descs = [name[:8].upper() for name in names]  # Short, unique descs
    pbar_list = []
    for i, desc in enumerate(bar_descs):
        pbar_list.append(tqdm(total=None, desc=desc, position=i, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', leave=True, dynamic_ncols=True))
    finished_count = 0
    expected_finishes = len(names)
    totals = [None for _ in names]

    name_to_idx = {name: i for i, name in enumerate(names)}

    while finished_count < expected_finishes:
        name, current, total = progress_queue.get()
        idx = name_to_idx[name]
        pbar = pbar_list[idx]
        if current is None:
            finished_count += 1
            if totals[idx] is not None:
                pbar.n = totals[idx]
                pbar.refresh()
        else:
            if totals[idx] is None and total:
                totals[idx] = total
                pbar.total = total
                pbar.refresh()
            pbar.n = current
            pbar.refresh()

    for pbar in pbar_list:
        pbar.close()

    results = []
    stats_list = []
    errors = []
    for _ in processes:
        name, data, stats, err = result_queue.get()
        if err:
            errors.append((name, err))
        else:
            results.append(data)
            stats_list.append((name, stats))

    for p in processes:
        p.join()

    if errors:
        for name, err in errors:
            print(f"❌ Parser '{name}' failed: {err}")
        raise RuntimeError(f"One or more parsers failed: {errors}")

    return results, stats_list


def merge_linguistic_dicts(dicts: List[dict]) -> dict:
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
    _tqdm.write("\n=== Parsing and Merging Stress Dictionaries ===\n")
    # Run both parsers concurrently using multiprocessing with progress bars
    results, stats_list = run_parsers_concurrently_mp([txt_parser, trie_parser], ["TXT", "TRIE"])
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
    for key in ["замок", "блоха", "клаксон", "помилка", "обід"]:
        _tqdm.write(f"Merged entry for lemma: '{key}'")
        entry = merged.get(key)
        if not entry:
            _tqdm.write("  Not found in merged dictionary.")
            continue
        _tqdm.write(f"\033[1;36m{key}\033[0m:")
        _tqdm.write(f"\033[0;37m{pp.pformat(entry.model_dump())}\033[0m")

if __name__ == "__main__":
    main()
