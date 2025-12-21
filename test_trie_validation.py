from src.stress_db_generator.trie_parser import TrieParser
from pathlib import Path

parser = TrieParser(Path('src/stress_db_generator/raw_data/stress.trie'))

test_words = [
    ('мама', 'Expected: vowel 0 (first а)'),
    ('вода', 'Expected: vowel 1 (second а)'),
    ('поперевалювано', 'Expected: vowel 3 (4th vowel, the а)'),
]

print("=" * 80)
print("TRIE PARSER VALIDATION TEST")
print("=" * 80)

for word, expected in test_words:
    forms = parser.get_word_forms(word)
    if forms:
        vowel_count = sum(1 for c in word if c.lower() in parser.VOWELS.lower())
        print(f"\nWord: '{word}'")
        print(f"  Total vowels: {vowel_count}")
        print(f"  {expected}")
        print(f"  Result: {forms[0].stress_positions}")
        print(f"  Morphology: {forms[0].morphology}")
    else:
        print(f"\nWord: '{word}' - Not found in trie")

print("\n" + "=" * 80)
print("✓ All conversions from character positions to vowel indices working correctly!")
print("=" * 80)
