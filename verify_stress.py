word = 'поперевалювано'

print("Character-by-character breakdown:")
print("Pos | Char | Vowel#")
print("----|------|-------")
vowel_num = 0
for i, c in enumerate(word):
    is_vowel = c.lower() in 'уеіїаояиюєУЕІАОЯИЮЄЇ'
    if is_vowel:
        print(f"{i:3d} | {c:^4s} | {vowel_num}")
        vowel_num += 1
    else:
        print(f"{i:3d} | {c:^4s} |")

print(f"\nTotal vowels: {vowel_num}")
print(f"\nIf stress is on vowel #4 (0-indexed), that's the 5th vowel")
print(f"The 5th vowel (index 4) is 'ю' at character position 9")
print(f"\nTrie stores: [8]")
print(f"Character at position 8: '{word[8]}'")
print(f"Character at position 9: '{word[9]}'")
print(f"\nSo if trie stores 8, and we need vowel index 4...")
print(f"Then position 8 should point to BEFORE the stressed vowel")
print(f"Position 9 (ю) is vowel index 4 ✓")
