from src.stress_db_generator.trie_parser import TrieParser
from pathlib import Path

p = TrieParser(Path('src/stress_db_generator/raw_data/stress.trie'))

test_word = 'поперевалювано'
forms = p.get_word_forms(test_word)

print(f'Word: {test_word}')
print(f'Vowel count: {sum(1 for c in test_word if c in p.VOWELS)}')
print(f'Forms found: {len(forms)}')
for f in forms:
    print(f'  Stress vowel indices: {f.stress_positions}')
    print(f'  Morphology: {f.morphology}')
