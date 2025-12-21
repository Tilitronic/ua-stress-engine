#!/usr/bin/env python3
"""Inspect raw data structure from LMDB"""

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.lmdb_exporter import LMDBQuery

lmdb_path = Path(__file__).parent.parent / "nlp" / "stress_service" / "stress.lmdb"

print("=" * 80)
print("RAW DATA STRUCTURE INSPECTION")
print("=" * 80)

with LMDBQuery(lmdb_path) as db:
    word = "атлас"
    forms = db.lookup(word)
    
    print(f"\nWord: '{word}'\n")
    print("Raw JSON from LMDB:")
    print(json.dumps(forms, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("TYPE INSPECTION")
    print("=" * 80)
    
    for i, form in enumerate(forms, 1):
        print(f"\nForm #{i}:")
        print(f"  stress_variants: {type(form['stress_variants']).__name__} = {form['stress_variants']}")
        print(f"  pos: {type(form['pos']).__name__} = {form['pos']}")
        print(f"  feats: {type(form['feats']).__name__}")
        
        if form.get('feats'):
            for key, value in form['feats'].items():
                print(f"    '{key}': {type(value).__name__} = {value}")
                if isinstance(value, list):
                    print(f"           ✓ IS A LIST")
                else:
                    print(f"           ❌ NOT A LIST - THIS IS WRONG!")
