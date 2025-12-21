#!/usr/bin/env python3
"""
Test LMDB query performance and functionality.
"""

from pathlib import Path
import sys
import time
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.lmdb_exporter import LMDBQuery


def test_lookup_performance(db: LMDBQuery, test_words: list, iterations: int = 1000):
    """Test lookup performance"""
    print(f"\n🔥 Performance Test: {iterations:,} lookups")
    print(f"   Testing {len(test_words)} words × {iterations} iterations = {len(test_words) * iterations:,} total queries\n")
    
    start = time.time()
    
    for _ in range(iterations):
        for word in test_words:
            forms = db.lookup(word)
    
    elapsed = time.time() - start
    total_queries = len(test_words) * iterations
    qps = total_queries / elapsed
    
    print(f"   ⏱️  Time: {elapsed:.3f}s")
    print(f"   🚀 Speed: {qps:,.0f} queries/second")
    print(f"   ⚡ Avg latency: {(elapsed / total_queries) * 1000:.3f}ms per query")


def main():
    print("=" * 80)
    print("LMDB QUERY TEST & PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    lmdb_path = Path(__file__).parent.parent / "nlp" / "stress_service" / "stress.lmdb"
    
    if not lmdb_path.exists():
        print(f"\n❌ ERROR: LMDB database not found at {lmdb_path}")
        print(f"   Run 'python src/stress_db_generator/build_lmdb.py' first to create the database.")
        return 1
    
    print(f"\n📂 Opening database: {lmdb_path}\n")
    
    with LMDBQuery(lmdb_path) as db:
        # Database stats
        print("=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        
        stats = db.get_stats()
        print(f"  📊 Total entries: {stats['entries']:,}")
        print(f"  💾 Database size: {stats['size_bytes'] / (1024*1024):.2f} MB")
        
        # Test exact lookups
        print("\n" + "=" * 80)
        print("EXACT WORD LOOKUPS")
        print("=" * 80)
        
        test_words = ["атлас", "блохи", "помилка", "коридор", "замок", "мама", "вода"]
        
        for word in test_words:
            forms = db.lookup(word)
            
            if not forms:
                print(f"\n❌ '{word}': NOT FOUND")
                continue
            
            print(f"\n✓ '{word}' — {len(forms)} form(s):")
            for i, form in enumerate(forms, 1):
                print(f"  [{i}] {json.dumps(form, ensure_ascii=False)}")
        
        # Test prefix search
        print("\n" + "=" * 80)
        print("PREFIX SEARCH")
        print("=" * 80)
        
        prefixes = ["атл", "замок", "кор"]
        
        for prefix in prefixes:
            matches = db.prefix_search(prefix, limit=10)
            print(f"\n'{prefix}*' → {len(matches)} matches (showing max 10):")
            for word in matches[:10]:
                print(f"  • {word}")
        
        # Performance test
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK")
        print("=" * 80)
        
        test_lookup_performance(db, test_words, iterations=10000)
        
        # List sample words
        print("\n" + "=" * 80)
        print("SAMPLE WORDS (first 20)")
        print("=" * 80)
        
        sample_words = db.list_words(limit=20)
        print()
        for i, word in enumerate(sample_words, 1):
            print(f"  {i:2d}. {word}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\n🚀 LMDB is ready for production use!")
    print("   Ultra-fast lookups with memory-mapped access\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
