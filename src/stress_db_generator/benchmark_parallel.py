#!/usr/bin/env python3
"""
Benchmark: Sequential vs Parallel Parsing

Compares the performance of sequential vs parallel parsing
to measure actual time savings.
"""

from pathlib import Path
import sys
import time
import multiprocessing
from multiprocessing import Manager

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.trie_adapter import parse_trie_data
from src.stress_db_generator.txt_parser import parse_txt_dictionary


def parse_trie_worker(trie_path, queue, result_queue):
    """Worker for trie parsing"""
    try:
        start = time.time()
        data = parse_trie_data(trie_path, None)
        elapsed = time.time() - start
        result_queue.put(('trie', data, elapsed))
    except Exception as e:
        result_queue.put(('trie_error', str(e), 0))


def parse_txt_worker(txt_path, queue, result_queue):
    """Worker for txt parsing"""
    try:
        start = time.time()
        data = parse_txt_dictionary(txt_path, None)
        elapsed = time.time() - start
        result_queue.put(('txt', data, elapsed))
    except Exception as e:
        result_queue.put(('txt_error', str(e), 0))


if __name__ == "__main__":
    print("=" * 80)
    print("PARSING BENCHMARK: Sequential vs Parallel")
    print("=" * 80)
    
    # File paths
    base_path = Path(__file__).parent
    trie_path = base_path / "raw_data" / "stress.trie"
    txt_path = base_path / "raw_data" / "ua_word_stress_dictionary.txt"
    
    print(f"\n📂 Files:")
    print(f"  Trie: {trie_path.name} ({trie_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  TXT:  {txt_path.name} ({txt_path.stat().st_size / (1024*1024):.2f} MB)")
    
    # ========================================================================
    # Test 1: Sequential Parsing
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 1: Sequential Parsing")
    print("-" * 80)
    
    seq_start = time.time()
    
    print("  Parsing Trie...")
    trie_start = time.time()
    trie_data = parse_trie_data(trie_path, None)
    trie_time = time.time() - trie_start
    print(f"    ✓ Completed in {trie_time:.2f}s ({len(trie_data):,} words)")
    
    print("  Parsing TXT...")
    txt_start = time.time()
    txt_data = parse_txt_dictionary(txt_path, None)
    txt_time = time.time() - txt_start
    print(f"    ✓ Completed in {txt_time:.2f}s ({len(txt_data):,} words)")
    
    seq_total = time.time() - seq_start
    print(f"\n  Total Sequential Time: {seq_total:.2f}s")
    
    # ========================================================================
    # Test 2: Parallel Parsing
    # ========================================================================
    print("\n" + "-" * 80)
    print("TEST 2: Parallel Parsing (Multiprocessing)")
    print("-" * 80)
    
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    
    par_start = time.time()
    
    print("  Starting both parsers simultaneously...")
    p_trie = multiprocessing.Process(target=parse_trie_worker, args=(trie_path, progress_queue, result_queue))
    p_txt = multiprocessing.Process(target=parse_txt_worker, args=(txt_path, progress_queue, result_queue))
    
    p_trie.start()
    p_txt.start()
    
    # Wait for completion
    p_trie.join()
    p_txt.join()
    
    par_total = time.time() - par_start
    
    # Collect results
    trie_par_data = None
    txt_par_data = None
    trie_par_time = 0
    txt_par_time = 0
    
    while not result_queue.empty():
        source, data, elapsed = result_queue.get()
        if source == 'trie':
            trie_par_data = data
            trie_par_time = elapsed
            print(f"    ✓ Trie completed in {elapsed:.2f}s ({len(data):,} words)")
        elif source == 'txt':
            txt_par_data = data
            txt_par_time = elapsed
            print(f"    ✓ TXT completed in {elapsed:.2f}s ({len(data):,} words)")
    
    print(f"\n  Total Parallel Time: {par_total:.2f}s")
    
    # ========================================================================
    # Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nSequential Execution:")
    print(f"  Trie:  {trie_time:.2f}s")
    print(f"  TXT:   {txt_time:.2f}s")
    print(f"  Total: {seq_total:.2f}s")
    
    print(f"\nParallel Execution:")
    print(f"  Trie:  {trie_par_time:.2f}s (in background)")
    print(f"  TXT:   {txt_par_time:.2f}s (in background)")
    print(f"  Total: {par_total:.2f}s")
    
    time_saved = seq_total - par_total
    speedup = seq_total / par_total
    efficiency = (speedup / 2) * 100  # 2 cores used
    
    print(f"\n⚡ Performance Gain:")
    print(f"  Time Saved:  {time_saved:.2f}s ({(time_saved/seq_total*100):.1f}%)")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Efficiency:  {efficiency:.1f}% (parallel efficiency on 2 cores)")
    
    print("\n" + "=" * 80)
