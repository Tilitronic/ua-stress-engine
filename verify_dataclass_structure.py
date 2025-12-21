#!/usr/bin/env python3
"""
Verify that the dataclass structure uses lists for ALL properties.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.stress_db_generator.parser import StressDictionary

def inspect_structure(obj, indent=0):
    """Recursively inspect object structure"""
    prefix = "  " * indent
    if isinstance(obj, list):
        print(f"{prefix}LIST with {len(obj)} items:")
        if obj and len(obj) <= 5:
            for i, item in enumerate(obj):
                print(f"{prefix}  [{i}] {type(item).__name__}: {repr(item)}")
    elif isinstance(obj, dict):
        print(f"{prefix}DICT with {len(obj)} keys:")
        for key, value in obj.items():
            print(f"{prefix}  '{key}': {type(value).__name__} = {repr(value)}")
    else:
        print(f"{prefix}{type(obj).__name__}: {repr(obj)}")

def main():
    print("=" * 80)
    print("DATACLASS STRUCTURE VERIFICATION")
    print("=" * 80)
    print("\nVerifying that ALL properties are lists (not strings)\n")
    
    # Load dictionary
    trie_path = Path("src/stress_db_generator/raw_data/stress.trie")
    dictionary = StressDictionary()
    
    print("⏳ Loading trie (this will take ~30 seconds)...")
    dictionary.parse_trie(trie_path)
    
    # Test word
    test_word = "атлас"
    forms = dictionary.lookup(test_word)
    
    print(f"\n{'=' * 80}")
    print(f"Inspecting structure for word: '{test_word}'")
    print(f"{'=' * 80}\n")
    
    if not forms:
        print("Word not found!")
        return 1
    
    for i, form in enumerate(forms, 1):
        print(f"\n{'─' * 80}")
        print(f"FORM #{i}")
        print(f"{'─' * 80}\n")
        
        # Check raw dataclass attributes
        print("📋 Raw Dataclass Attributes:")
        print(f"  stress_variants: {type(form.stress_variants).__name__} = {form.stress_variants}")
        print(f"  pos: {type(form.pos).__name__} = {form.pos}")
        print(f"  feats: {type(form.feats).__name__}")
        
        if form.feats:
            print(f"  feats contents:")
            for key, value in form.feats.items():
                print(f"    '{key}': {type(value).__name__} = {value}")
        
        # Check to_dict() output
        print(f"\n📦 to_dict() output:")
        dict_output = form.to_dict()
        inspect_structure(dict_output, indent=1)
        
        # Check to_spacy_format() output
        print(f"\n📝 to_spacy_format() output:")
        spacy_output = form.to_spacy_format()
        print(f"  Type: {type(spacy_output).__name__}")
        print(f"  Value: '{spacy_output}'")
        
        # Verify structure
        print(f"\n✓ Structure Verification:")
        print(f"  stress_variants is list: {isinstance(form.stress_variants, list)} ✓" if isinstance(form.stress_variants, list) else f"  ❌ stress_variants is {type(form.stress_variants).__name__}")
        print(f"  pos is list: {isinstance(form.pos, list)} ✓" if isinstance(form.pos, list) else f"  ❌ pos is {type(form.pos).__name__}")
        
        all_feats_are_lists = all(isinstance(v, list) for v in form.feats.values())
        print(f"  ALL feats values are lists: {all_feats_are_lists} ✓" if all_feats_are_lists else "  ❌ Some feats values are NOT lists")
        
        if form.feats and not all_feats_are_lists:
            print(f"  Problem feats:")
            for key, value in form.feats.items():
                if not isinstance(value, list):
                    print(f"    '{key}': {type(value).__name__} = {value}")
    
    print(f"\n{'=' * 80}")
    print("VERIFICATION COMPLETE")
    print(f"{'=' * 80}\n")
    
    # Summary
    print("Expected structure:")
    print("""
    WordForm(
        stress_variants=[0],           # ← LIST of ints
        pos=["NOUN"],                  # ← LIST of strings
        feats={
            "Case": ["Acc", "Nom"],    # ← LIST of strings
            "Gender": ["Masc"],        # ← LIST of strings (even single!)
            "Number": ["Sing"]         # ← LIST of strings (even single!)
        }
    )
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
