#!/usr/bin/env python3
"""
Find words with no stress data in LMDB database.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nlp.stress_service.stress_service import UkrainianStressService


def find_words_without_stress():
    """Find all words that have no stress data"""
    print("=" * 80)
    print("Finding words with no stress data...")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        # Get all words from database
        all_words = service.db.list_words()
        total_words = len(all_words)
        
        print(f"\nTotal words in database: {total_words:,}")
        print("Scanning for words without stress data...\n")
        
        no_stress_words = []
        
        for i, word in enumerate(all_words, 1):
            if i % 100000 == 0:
                print(f"  Processed {i:,}/{total_words:,} words...")
            
            result = service.lookup(word, normalize=False)
            
            if result:
                # Check if any form has stress data
                has_stress = False
                for form in result:
                    stress = form.get('stress_variants', [])
                    if stress:
                        has_stress = True
                        break
                
                if not has_stress:
                    no_stress_words.append(word)
        
        print(f"\n{'=' * 80}")
        print(f"Results: {len(no_stress_words):,} words without stress data")
        print(f"Percentage: {len(no_stress_words)/total_words*100:.2f}%")
        print(f"{'=' * 80}")
        
        # Show first 100 examples
        if no_stress_words:
            print(f"\nFirst 100 examples:")
            for i, word in enumerate(no_stress_words[:100], 1):
                print(f"  {i:3}. {word}")
            
            if len(no_stress_words) > 100:
                print(f"\n  ... and {len(no_stress_words) - 100:,} more")
        
        # Save to file
        output_file = Path(__file__).parent / "words_without_stress.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in no_stress_words:
                f.write(f"{word}\n")
        
        print(f"\n💾 Full list saved to: {output_file}")
        
        return no_stress_words


if __name__ == "__main__":
    find_words_without_stress()
