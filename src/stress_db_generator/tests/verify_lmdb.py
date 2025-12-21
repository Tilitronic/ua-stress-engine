#!/usr/bin/env python3
"""Quick verification of LMDB database"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.lmdb_exporter import LMDBQuery

lmdb_path = Path(__file__).parent.parent / "nlp" / "stress_service" / "stress.lmdb"

print("=" * 80)
print("LMDB DATABASE VERIFICATION")
print("=" * 80)

with LMDBQuery(lmdb_path) as db:
    stats = db.get_stats()
    print(f"\n📊 Database Statistics:")
    print(f"  Entries: {stats['entries']:,}")
    print(f"  Size: {stats['size_bytes'] / (1024*1024):.2f} MB")
    
    print(f"\n🔍 Test Lookups:\n")
    test_words = ["атлас", "блохи", "замок", "помилка"]
    
    for word in test_words:
        forms = db.lookup(word)
        if not forms:
            print(f"  ❌ '{word}': NOT FOUND\n")
            continue
        
        print(f"  ✓ '{word}' — {len(forms)} form(s):")
        for i, form in enumerate(forms, 1):
            print(f"    [{i}] {form}")
        print()

print("=" * 80)
print("✅ Verification complete!")
print("=" * 80)
