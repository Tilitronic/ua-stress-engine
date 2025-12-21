#!/usr/bin/env python3
"""
Test the integrated parser that merges data from both text files and trie.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.stress_db_generator.parser import StressDictionary

def main():
    print("=" * 80)
    print("Testing Parser Integration (Text + Trie)")
    print("=" * 80)
    
    # Initialize dictionary
    dictionary = StressDictionary()
    
    # Parse trie file
    trie_path = Path("src/stress_db_generator/raw_data/stress.trie")
    
    if not trie_path.exists():
        print(f"ERROR: Trie file not found at {trie_path}")
        return 1
    
    print(f"\nParsing trie file: {trie_path}")
    print("This may take a few minutes for ~2.9M entries...")
    print()
    
    try:
        dictionary.parse_trie(trie_path)
        
        print("\n" + "=" * 80)
        print("Dictionary Statistics")
        print("=" * 80)
        
        stats = dictionary.get_statistics()
        print(f"Total unique words: {stats['total_unique_words']:,}")
        print(f"Total word forms: {stats['total_forms']:,}")
        print(f"Words with multiple stress patterns: {stats['words_with_unique_stress']:,}")
        print(f"Average forms per word: {stats['avg_forms_per_word']}")
        
        print("\n" + "=" * 80)
        print("Test Lookups")
        print("=" * 80)
        
        # Test specific words
        test_words = [
            "атлас",  # heteronym
            "блохи",
            "помилка",
            "коридор"
        ]
        
        for word in test_words:
            print(f"\nWord: '{word}'")
            forms = dictionary.lookup(word)
            
            if not forms:
                print("  NOT FOUND")
                continue
            
            print(f"  Found {len(forms)} form(s):")
            for i, form in enumerate(forms, 1):
                print(f"    Form {i}:")
                print(f"      Stress indices: {form.stress_variants}")
                if form.morphology:
                    print(f"      Morphology: {form.morphology}")
        
        # Show some heteronyms
        print("\n" + "=" * 80)
        print("Sample Heteronyms (words with multiple stress patterns)")
        print("=" * 80)
        
        heteronyms = dictionary.get_words_with_unique_stress()
        print(f"\nShowing first 10 of {len(heteronyms):,} heteronyms:\n")
        
        for word, forms in heteronyms[:10]:
            print(f"'{word}' - {len(forms)} forms:")
            for form in forms:
                stress_str = ", ".join(str(i) for i in form.stress_variants)
                morph_str = f" [{form.morphology}]" if form.morphology else ""
                print(f"  Stress at vowel(s): {stress_str}{morph_str}")
        
        print("\n" + "=" * 80)
        print("Integration test complete!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
