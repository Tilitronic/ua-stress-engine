#!/usr/bin/env python3
"""
Database Generator - LMDB Stress Database Builder

Generates LMDB stress database from multiple sources.
Pipeline: Download/Update → Trie → TXT → Merge → spaCy → LMDB Export

Data Attribution:
    This tool processes data from lang-uk organization:
    
    1. stress.trie - Trie database
       Source: https://github.com/lang-uk/ukrainian-word-stress
       License: MIT License, Copyright (c) 2022 lang-uk
    
    2. ua_word_stress_dictionary.txt - Text dictionary
       Source: https://github.com/lang-uk/ukrainian-word-stress-dictionary
       Based on "Словники України" (Ukrainian Linguistic Information Fund)
    
    See raw_data/DATA_ATTRIBUTION.md for complete licensing information.
"""

from pathlib import Path
import sys
import time
import json
import itertools
import threading
import multiprocessing
from multiprocessing import Manager
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.trie_adapter import parse_trie_data
from src.stress_db_generator.txt_parser import parse_txt_dictionary
from src.stress_db_generator.merger import DictionaryMerger
from src.stress_db_generator.spacy_transformer import SpaCyTransformer
from src.stress_db_generator.download_data_files import check_and_download_files
from src.stress_db_generator.lmdb_exporter import LMDBExporter


# ============================================================================
# Multiprocessing Workers with Progress Reporting
# ============================================================================

def parse_trie_worker(trie_path: Path, queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """Worker function for trie parsing with queue-based progress"""
    def progress_callback(current, total):
        # Report progress every 5000 items
        if current % 5000 == 0 or current == total:
            queue.put(('trie', current, total))
    
    try:
        data = parse_trie_data(trie_path, progress_callback)
        result_queue.put(('trie', data))
        queue.put(('trie', None, None))  # Signal finished
    except Exception as e:
        result_queue.put(('trie_error', str(e)))
        queue.put(('trie', None, None))


def parse_txt_worker(txt_path: Path, queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """Worker function for txt parsing with queue-based progress"""
    def progress_callback(current, total):
        # Report progress every 5000 items
        if current % 5000 == 0 or current == total:
            queue.put(('txt', current, total))
    
    try:
        data = parse_txt_dictionary(txt_path, progress_callback)
        result_queue.put(('txt', data))
        queue.put(('txt', None, None))  # Signal finished
    except Exception as e:
        result_queue.put(('txt_error', str(e)))
        queue.put(('txt', None, None))


# ============================================================================
# Progress Utilities
# ============================================================================

class ProgressSpinner:
    """Simple console spinner for long-running operations"""
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.running = False
        self.thread = None
    
    def _spin(self):
        while self.running:
            sys.stdout.write(f'\r{self.message} {next(self.spinner)}')
            sys.stdout.flush()
            time.sleep(0.1)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')
        if final_message:
            print(final_message)
        sys.stdout.flush()


class ProgressBar:
    """Simple progress bar for operations with known total"""
    
    def __init__(self, total: int, prefix: str = "", length: int = 40):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
    
    def update(self, current: int):
        self.current = current
        percent = min(100, int(100.0 * current / self.total))
        filled = int(self.length * percent // 100)
        bar = '█' * filled + '░' * (self.length - filled)
        sys.stdout.write(f'\r{self.prefix} [{bar}] {percent}% ({current:,}/{self.total:,})')
        sys.stdout.flush()
    
    def finish(self, message: str = ""):
        sys.stdout.write('\r' + ' ' * (len(self.prefix) + self.length + 30) + '\r')
        if message:
            print(message)
        sys.stdout.flush()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    pipeline_start = time.time()
    
    print("=" * 80)
    print("STRESS DATABASE GENERATOR")
    print("=" * 80)
    print("\nPipeline: Download → Trie → TXT → Merge → spaCy → LMDB Export\n")
    
    # ========================================================================
    # STEP 0: Check and Download Data Files
    # ========================================================================
    print("\n")
    step_start = time.time()
    if not check_and_download_files():
        print("\n❌ ERROR: Failed to prepare data files")
        return 1
    print(f"\n⏱️  Data check completed in {time.time() - step_start:.2f}s")
    
    # Paths
    trie_path = Path(__file__).parent / "raw_data" / "stress.trie"
    txt_path = Path(__file__).parent / "raw_data" / "ua_word_stress_dictionary.txt"
    output_path = Path(__file__).parent.parent / "nlp" / "stress_service" / "stress.lmdb"
    
    # Validate input files exist (should be guaranteed by step 0)
    if not trie_path.exists():
        print(f"❌ ERROR: Trie file not found at {trie_path}")
        return 1
    
    if not txt_path.exists():
        print(f"⚠️  WARNING: TXT file not found at {txt_path}")
        print("    Continuing with trie data only...")
        txt_path = None
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEPS 1-2: Parse Trie and TXT Dictionary (Parallel)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEPS 1-2: Parsing Trie and TXT Dictionary (Parallel)")
    print("=" * 80)
    print(f"📂 Trie:  {trie_path.name} ({trie_path.stat().st_size / (1024*1024):.2f} MB)")
    if txt_path:
        print(f"📂 TXT:   {txt_path.name} ({txt_path.stat().st_size / (1024*1024):.2f} MB)")
    
    print("\n🚀 Starting parallel parsing with multiprocessing (bypasses GIL)...\n")
    parallel_start = time.time()
    
    # Create manager and queues for inter-process communication
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Start parsing processes
    p_trie = multiprocessing.Process(target=parse_trie_worker, args=(trie_path, progress_queue, result_queue))
    p_txt = multiprocessing.Process(target=parse_txt_worker, args=(txt_path, progress_queue, result_queue)) if txt_path else None
    
    p_trie.start()
    if p_txt:
        p_txt.start()
    
    # Create stacked progress bars
    pbar_trie = tqdm(total=100, desc="Trie", position=0, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    pbar_txt = tqdm(total=100, desc="TXT ", position=1, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') if txt_path else None
    
    # Track progress from both processes
    finished_count = 0
    expected_finishes = 2 if txt_path else 1
    trie_total = 0
    txt_total = 0
    
    while finished_count < expected_finishes:
        source, current, total = progress_queue.get()
        
        if current is None:  # Process finished
            finished_count += 1
            if source == 'trie':
                pbar_trie.n = trie_total  # Set to actual total, not 100
                pbar_trie.refresh()
            elif source == 'txt' and pbar_txt:
                pbar_txt.n = txt_total  # Set to actual total, not 100
                pbar_txt.refresh()
        else:
            if source == 'trie':
                if trie_total == 0:
                    trie_total = total
                    pbar_trie.total = total
                    pbar_trie.reset(total=total)  # Reset with proper total
                pbar_trie.n = current
                pbar_trie.refresh()
            elif source == 'txt' and pbar_txt:
                if txt_total == 0:
                    txt_total = total
                    pbar_txt.total = total
                    pbar_txt.reset(total=total)  # Reset with proper total
                pbar_txt.n = current
                pbar_txt.refresh()
    
    # Wait for processes to complete
    pbar_trie.write("⏳ Waiting for Trie parser process to finish...")
    p_trie.join()
    pbar_trie.write("✓ Trie parser process finished")
    
    if p_txt:
        pbar_txt.write("⏳ Waiting for TXT parser process to finish...")
        p_txt.join()
        pbar_txt.write("✓ TXT parser process finished")
    
    # Close progress bars
    pbar_trie.close()
    if pbar_txt:
        pbar_txt.close()
    
    print()  # Add newline after progress bars
    print("📦 Collecting parsed data from worker processes...")
    
    # Collect results
    trie_data = {}
    txt_data = {}
    
    while not result_queue.empty():
        source, data = result_queue.get()
        if source == 'trie':
            trie_data = data
            print(f"✓ Received Trie data: {len(trie_data):,} words")
        elif source == 'txt':
            txt_data = data
            print(f"✓ Received TXT data: {len(txt_data):,} words")
        elif source.endswith('_error'):
            print(f"❌ Error in {source}: {data}")
            sys.exit(1)
    
    parallel_elapsed = time.time() - parallel_start
    
    print(f"✓ Parallel parsing completed in {parallel_elapsed:.2f}s")
    print(f"  📊 Trie: {len(trie_data):,} unique words")
    if txt_data:
        print(f"  📊 TXT:  {len(txt_data):,} unique words")
    print(f"⏱️  Step completed in {parallel_elapsed:.2f}s")
    
    # ========================================================================
    # STEP 3: Merge Dictionaries
    # ========================================================================
    step_num = 3
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Merging Dictionaries")
    print("=" * 80)
    
    step_start = time.time()
    spinner = ProgressSpinner("⏳ Combining data with intelligent merging")
    spinner.start()
    
    merger = DictionaryMerger()
    
    merger.add_trie_data(trie_data)
    if txt_data:
        merger.add_txt_data(txt_data)
    merged_dict = merger.get_dictionary()
    elapsed = time.time() - step_start
    
    spinner.stop()
    print(f"✓ Completed in {elapsed:.2f}s")
    
    stats = merger.get_statistics()
    print(f"\n  📊 Statistics:")
    print(f"     Total unique words:     {stats['total_unique_words']:>10,}")
    print(f"     Total word forms:       {stats['total_forms']:>10,}")
    print(f"     Heteronyms:             {stats['heteronyms']:>10,}")
    print(f"     Words with morphology:  {stats['words_with_morphology']:>10,}")
    if txt_data:
        print(f"     From trie only:         {stats['trie_only']:>10,}")
        print(f"     From txt only:          {stats['txt_only']:>10,}")
        print(f"     Merged sources:         {stats['merged']:>10,}")
    
    # ========================================================================
    # STEP 4: Transform to spaCy Format
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: spaCy/UD Format Transformation")
    print("=" * 80)
    
    step_start = time.time()
    spinner = ProgressSpinner("⏳ Validating morphological features")
    spinner.start()
    
    transformer = SpaCyTransformer(strict=False)
    
    spacy_dict = transformer.transform(merged_dict)
    
    spinner.stop()
    print(f"✓ Validation completed")
    print(f"  📊 Validated words: {len(spacy_dict):,}")
    
    # Convert to export format
    spinner = ProgressSpinner("⏳ Preparing export data")
    spinner.start()
    
    export_dict = {}
    for key, forms in spacy_dict.items():
        export_dict[key] = [form.to_dict() for form in forms]
    
    spinner.stop()
    
    # Purge empty entries (no stress, no POS, no features)
    print(f"\n  🧹 Purging empty entries...")
    original_count = len(export_dict)
    
    empty_words = []
    for word, forms in list(export_dict.items()):
        # Check if all forms are completely empty
        all_empty = True
        for form in forms:
            stress = form.get('stress_variants', [])
            pos = form.get('pos', [])
            feats = form.get('feats', {})
            
            # If any form has stress, POS, or features, keep the word
            if stress or pos or feats:
                all_empty = False
                break
        
        if all_empty:
            empty_words.append(word)
            del export_dict[word]
    
    purged_count = len(empty_words)
    if purged_count > 0:
        print(f"     Purged empty words:     {purged_count:>10,} ({purged_count/original_count*100:.2f}%)")
        # Show a few examples
        if purged_count <= 10:
            print(f"     Examples: {', '.join(empty_words)}")
        else:
            print(f"     Examples: {', '.join(empty_words[:10])} ...")
    else:
        print(f"     No empty words found ✓")
    
    elapsed = time.time() - step_start
    print(f"  ✓ Export data ready: {len(export_dict):,} entries")
    print(f"⏱️  Step completed in {elapsed:.2f}s")
    
    # ========================================================================
    # STEP 5: Export to LMDB
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Exporting to LMDB")
    print("=" * 80)
    print(f"📦 Output: {output_path.relative_to(Path(__file__).parent.parent.parent)}")
    
    try:
        step_start = time.time()
        
        # Create exporter and export with progress tracking
        exporter = LMDBExporter(output_path)
        
        # Progress bar for export
        export_progress_bar = ProgressBar(len(export_dict), prefix="      Progress")
        
        def export_progress(current, total):
            export_progress_bar.update(current)
        
        exporter.export_raw(export_dict, progress_callback=export_progress)
        export_progress_bar.finish()
        
        elapsed = time.time() - step_start
        print(f"      ⏱️  Export completed in {elapsed:.2f}s ({len(export_dict) / elapsed:,.0f} words/sec)")
        
    except Exception as e:
        print(f"      ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 6: Verification
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Verification")
    print("=" * 80)
    
    test_words = ["атлас", "блохи", "замок", "помилка"]
    
    try:
        verify_result = exporter.verify(test_words)
        
        if verify_result["status"] == "success":
            print(f"  ✓ Entries:  {verify_result['entries']:>10,}")
            print(f"  ✓ Size:     {verify_result['size_bytes'] / (1024*1024):.2f} MB")
            print(f"  ✓ Samples:  {verify_result['sample_found']:>10}")
        else:
            print(f"  ⚠️  Verification failed: {verify_result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"  ❌ Verification error: {e}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ DATABASE GENERATION COMPLETE")
    print("=" * 80)
    
    total_elapsed = time.time() - pipeline_start
    
    print(f"\n📊 Summary:")
    print(f"   Total words:      {len(export_dict):,}")
    print(f"   Heteronyms:       {stats['heteronyms']:,}")
    print(f"   Total time:       {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
    
    # Calculate LMDB directory size (sum of all files)
    lmdb_size = 0
    if output_path.exists() and output_path.is_dir():
        for file in output_path.iterdir():
            if file.is_file():
                lmdb_size += file.stat().st_size
    
    print(f"\n📁 Output:")
    print(f"   Path: {output_path}")
    print(f"   Size: {lmdb_size / (1024 * 1024):.2f} MB")
    
    # ========================================================================
    # STEP 7: Run Database Tests
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Running Database Tests")
    print("=" * 80)
    
    try:
        from src.stress_db_generator.lmdb_exporter import LMDBQuery
        
        print(f"\n📂 Testing database: {output_path.relative_to(Path(__file__).parent.parent.parent)}\n")
        
        with LMDBQuery(output_path) as db:
            # Quick validation tests
            test_words = ["атлас", "блохи", "помилка", "замок", "мама"]
            
            print("✓ Database opened successfully")
            
            # Get stats
            stats_db = db.get_stats()
            print(f"  📊 Entries: {stats_db['entries']:,}")
            print(f"  💾 Size: {stats_db['size_bytes'] / (1024*1024):.2f} MB")
            
            # Test lookups
            print(f"\n  Testing {len(test_words)} sample lookups...")
            found = 0
            for word in test_words:
                forms = db.lookup(word)
                if forms:
                    found += 1
            
            print(f"  ✓ Found: {found}/{len(test_words)} words")
            
            # Quick performance test
            print(f"\n  Performance test (1000 queries)...")
            start = time.time()
            for _ in range(1000):
                for word in test_words:
                    db.lookup(word)
            elapsed = time.time() - start
            qps = (len(test_words) * 1000) / elapsed
            
            print(f"  🚀 Speed: {qps:,.0f} queries/second")
            print(f"  ⚡ Latency: {(elapsed / (len(test_words) * 1000)) * 1000:.3f}ms per query")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n⚠️  Test error: {e}")
        print("   Database created but tests failed")
    
    print(f"\n🎯 Ready for NLP pipeline integration\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
