import marisa_trie

t = marisa_trie.BytesTrie()
t.load('src/stress_db_generator/raw_data/stress.trie')

# Check the exact word and test the application logic
word = 'поперевалювано'
if word in t:
    stored_value = list(t[word][0])
    print(f"Word: {word}")
    print(f"Stored value: {stored_value}")
    print(f"\nApplying the reference library's logic:")
    print(f"s[:position] + accent + s[position:]")
    
    for pos in stored_value:
        result = word[:pos] + '́' + word[pos:]
        print(f"\nPosition {pos}:")
        print(f"  Result: {result}")
        print(f"  Breakdown: '{word[:pos]}' + '́' + '{word[pos:]}'")
        if pos > 0:
            print(f"  Character BEFORE position {pos}: '{word[pos-1]}'")
        if pos < len(word):
            print(f"  Character AT position {pos}: '{word[pos]}'")
        
        # Count which vowel this is
        VOWELS = 'уеіїаояиюєУЕІАОЯИЮЄЇ'
        vowel_count = 0
        for i in range(pos):
            if word[i] in VOWELS:
                vowel_count += 1
        print(f"  Number of vowels BEFORE position {pos}: {vowel_count}")
        
        # The accented vowel is the one just before the mark
        if pos > 0 and word[pos-1] in VOWELS:
            accented_vowel_pos = pos - 1
            vowels_before_accented = sum(1 for i in range(accented_vowel_pos) if word[i] in VOWELS)
            print(f"  ✓ Stressed vowel '{word[pos-1]}' is at vowel index: {vowels_before_accented}")
