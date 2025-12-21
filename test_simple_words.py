import marisa_trie

t = marisa_trie.BytesTrie()
t.load('src/stress_db_generator/raw_data/stress.trie')

# Test with simple words
test_words = ['мама', 'вода', 'рука', 'нога']

VOWELS = 'аеиіоуяюєї'

for word in test_words:
    if word in t:
        value = t[word][0]
        stored_pos = list(value)
        vowel_positions = [i for i, c in enumerate(word) if c.lower() in VOWELS]
        
        print(f"\nWord: '{word}'")
        print(f"  Stored position: {stored_pos}")
        print(f"  Vowel char positions: {vowel_positions}")
        print(f"  Characters: {list(enumerate(word))}")
        
        if stored_pos:
            pos = stored_pos[0]
            print(f"  Position {pos} is: '{word[pos] if pos < len(word) else 'OUT OF RANGE'}'")
            if pos > 0:
                print(f"  Position {pos-1} is: '{word[pos-1]}'")
