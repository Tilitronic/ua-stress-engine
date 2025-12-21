#!/usr/bin/env python3
"""
Verify the correctness of stress positions by showing actual stressed word forms.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.stress_db_generator.trie_parser import TrieParser

def apply_stress_marks(word, stress_indices):
    """Apply stress marks to word at given vowel indices"""
    VOWELS = "уеіїаояиюєУЕІАОЯИЮЄЇ"
    STRESS_MARK = "́"  # combining acute
    
    chars = list(word)
    vowel_count = 0
    result = []
    
    for i, char in enumerate(chars):
        result.append(char)
        if char.lower() in VOWELS.lower():
            if vowel_count in stress_indices:
                result.append(STRESS_MARK)
            vowel_count += 1
    
    return "".join(result)


def main():
    print("=" * 80)
    print("STRESS VERIFICATION - Checking if vowel indices produce correct stress")
    print("=" * 80)
    
    trie_path = Path("src/stress_db_generator/raw_data/stress.trie")
    parser = TrieParser(trie_path)
    
    test_cases = [
        ("атлас", "Expected: а́тлас (stress on first 'а') or атла́с (stress on second 'а')"),
        ("блохи", "Expected: бло́хи (stress on 'о') or блохи́ (stress on 'и')"),
        ("помилка", "Expected: по́милка or поми́лка (or both)"),
        ("коридор", "Expected: коридо́р (stress on last 'о')"),
        ("замок", "Expected: за́мок (castle) or замо́к (lock/verb)")
    ]
    
    for word, expected in test_cases:
        print(f"\n{'='*80}")
        print(f"Word: '{word}'")
        print(f"{expected}")
        print(f"{'='*80}")
        
        forms = parser.get_word_forms(word)
        
        if not forms:
            print("  ❌ NOT FOUND in trie")
            continue
        
        print(f"\n✓ Found {len(forms)} form(s) in trie:\n")
        
        for i, form in enumerate(forms, 1):
            stressed_word = apply_stress_marks(word, form.stress_positions)
            morph_info = []
            
            if form.morphology:
                if 'upos' in form.morphology:
                    morph_info.append(f"POS={form.morphology['upos']}")
                for key in sorted(form.morphology.keys()):
                    if key != 'upos':
                        morph_info.append(f"{key}={form.morphology[key]}")
            
            morph_str = " | " + ", ".join(morph_info) if morph_info else ""
            
            print(f"  [{i}] '{stressed_word}' — Stress at vowel index {form.stress_positions}{morph_str}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print("\nCheck if the stressed forms match your linguistic knowledge!")
    print("Vowel indices are 0-based: first vowel = 0, second = 1, etc.\n")


if __name__ == "__main__":
    main()
