word = 'поперевалювано'
vowels = 'уеіїаояиюєУЕІАОЯИЮЄЇ'

print('Pos  Char  IsVowel  VowelIdx')
print('-' * 40)
vowel_idx = 0
for i, c in enumerate(word):
    is_v = c in vowels
    v_idx = vowel_idx if is_v else '-'
    marker = ' <-- STRESS HERE (pos 8)' if i == 8 else ''
    print(f'{i:3d}  {c:4s}  {str(is_v):7s}  {v_idx}{marker}')
    if is_v:
        vowel_idx += 1

print(f'\nTrie stores: [8] (character position)')
print(f'Character at position 8: {word[8]}')
print(f'This should convert to vowel index: 4')
