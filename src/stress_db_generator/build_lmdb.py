buil#!/usr/bin/env python3
"""
Build LMDB database from trie and txt files.
Complete modular pipeline: Trie → TXT → Merge → spaCy Transform → LMDB
Output: Database for stress_service in src/nlp/stress_service/
"""

from pathlib import Path
import sys
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.trie_adapter import parse_trie_data
from src.stress_db_generator.txt_parser import parse_txt_dictionary
from src.stress_db_generator.merger import DictionaryMerger
from src.stress_db_generator.spacy_transformer import SpaCyTransformer
from src.stress_db_generator.lmdb_exporter import LMDBExporter

def main():
    print("=" * 80)
    print("LMDB DATABASE BUILDER")
    print("=" * 80)
    print("\nModular Pipeline: Trie → TXT → Merge → spaCy → LMDB\n")
    
    # Paths
    trie_path = Path(__file__).parent / "raw_data" / "stress.trie"
    txt_path = Path(__file__).parent / "raw_data" / "ua_word_stress_dictionary.txt"
    output_dir = Path(__file__).parent.parent / "nlp" / "stress_service"
    lmdb_path = output_dir / "stress.lmdb"
    
    # Validate input files
    if not trie_path.exists():
        print(f"❌ ERROR: Trie file not found at {trie_path}")
        return 1
    
    if not txt_path.exists():
        print(f"⚠️  WARNING: TXT file not found at {txt_path}")
        print("    Continuing with trie data only...")
        txt_path = None
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Parse Trie
    # ========================================================================
    print("=" * 80)
    print("STEP 1: Parsing Trie")
    print("=" * 80)
    
    print(f"📂 Input: {trie_path}")
    print("⏳ Processing ~2.9M entries (this takes ~30 seconds)...\n")
    
    start = time.time()
    trie_data = parse_trie_data(trie_path)
    elapsed = time.time() - start
    
    print(f"\n✓ Trie parsing complete in {elapsed:.2f}s")
    print(f"  📊 Unique words: {len(trie_data):,}")
    
    # ========================================================================
    # STEP 2: Parse TXT Dictionary (if available)
    # ========================================================================
    txt_data = {}
    
    if txt_path:
        print("\n" + "=" * 80)
        print("STEP 2: Parsing TXT Dictionary")
        print("=" * 80)
        
        print(f"📂 Input: {txt_path}")
        print("⏳ Processing dictionary file...\n")
        
        start = time.time()
        txt_data = parse_txt_dictionary(txt_path)
        elapsed = time.time() - start
        
        print(f"\n✓ TXT parsing complete in {elapsed:.2f}s")
        print(f"  📊 Unique words: {len(txt_data):,}")
    
    # ========================================================================
    # STEP 3: Merge Dictionaries
    # ========================================================================
    step_num = 3 if txt_path else 2
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Merging Dictionaries")
    print("=" * 80)
    
    print("⏳ Combining trie and txt data with intelligent merging...\n")
    
    merger = DictionaryMerger()
    
    start = time.time()
    merger.add_trie_data(trie_data)
    if txt_data:
        merger.add_txt_data(txt_data)
    merged_dict = merger.get_dictionary()
    elapsed = time.time() - start
    
    print(f"\n✓ Merging complete in {elapsed:.2f}s")
    
    stats = merger.get_statistics()
    print(f"\n  📊 Merge Statistics:")
    print(f"    Total unique words:     {stats['total_unique_words']:>12,}")
    print(f"    Total word forms:       {stats['total_forms']:>12,}")
    print(f"    Heteronyms:             {stats['heteronyms']:>12,}")
    print(f"    Words with morphology:  {stats['words_with_morphology']:>12,}")
    if txt_data:
        print(f"    From trie only:         {stats['trie_only']:>12,}")
        print(f"    From txt only:          {stats['txt_only']:>12,}")
        print(f"    Merged sources:         {stats['merged']:>12,}")
    
    # ========================================================================
    # STEP 4: Transform to spaCy Format
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: spaCy/UD Format Transformation")
    print("=" * 80)
    
    print("⏳ Validating and standardizing morphological features...\n")
    
    transformer = SpaCyTransformer(strict=False)
    
    start = time.time()
    spacy_dict = transformer.transform(merged_dict)
    elapsed = time.time() - start
    
    print(f"\n✓ Transformation complete in {elapsed:.2f}s")
    print(f"  📊 Validated words: {len(spacy_dict):,}")
    
    # ========================================================================
    # STEP 5: Export to LMDB
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Exporting to LMDB")
    print("=" * 80)
    
    print(f"📂 Output: {lmdb_path}")
    print(f"⏳ Writing {len(spacy_dict):,} words to LMDB...\n")
    
    exporter = LMDBExporter(lmdb_path, map_size=2 * 1024 * 1024 * 1024)  # 2GB max
    
    start = time.time()
    
    # Convert SpaCyWordForm to dict for export
    export_dict = {}
    for key, forms in spacy_dict.items():
        export_dict[key] = [form.to_dict() for form in forms]
    
    exporter.export_raw(export_dict)
    elapsed = time.time() - start
    
    print(f"\n✓ Export complete in {elapsed:.2f}s")
    
    # ========================================================================
    # STEP 6: Verification
    # ========================================================================
    step_num += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_num}: Verification")
    print("=" * 80)
    
    from src.stress_db_generator.lmdb_exporter import LMDBQuery
    
    with LMDBQuery(lmdb_path) as db:
        db_stats = db.get_stats()
        
        print(f"\n📊 Database Statistics:")
        print(f"  Entries: {db_stats['entries']:,}")
        print(f"  Size: {db_stats['size_bytes'] / (1024*1024):.2f} MB")
        
        # Test lookups
        print(f"\n🔍 Test Lookups:")
        test_words = ["атлас", "блохи", "замок", "помилка"]
        
        for word in test_words:
            forms = db.lookup(word)
            if not forms:
                print(f"  ❌ '{word}': NOT FOUND")
                continue
            
            print(f"\n  ✓ '{word}' — {len(forms)} form(s):")
            for i, form in enumerate(forms, 1):
                print(f"    [{i}] {form}")
    
    print("\n" + "=" * 80)
    print("✅ LMDB DATABASE BUILD COMPLETE")
    print("=" * 80)
    print(f"\n📁 Database location: {lmdb_path.absolute()}")
    print(f"📊 Total words: {len(spacy_dict):,}")
    print(f"🎯 Ready for NLP pipeline integration\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
