`ua_word_stress_dictionary.txt`

- **Source:** https://github.com/lang-uk/ukrainian-word-stress
- **License:** MIT License
- **Copyright:** (c) 2022 lang-uk
- **Format:** marisa_trie.BytesTrie
- **Size:** ~12 MB
- **Contains:** ~2.9M word forms with stress positions and morphological tags

## Dictionary Format

The dictionary is stored as a `marisa_trie.BytesTrie`. It maps words as they are written (without stress marks) to one or more possible stress positions, along with morphological information that helps resolve ambiguity.

- **Key:** A word (base form) without any stress marks.
- **Value:** A byte string in one of the following formats:

### Value Format #1: Single Accent Position

If the base word has only one possible accent position, each byte in the value is a character position of a stressed vowel. Most often, there will be only one, but more is also possible.

**Example:**

    b'\x02'

means that the accent is placed on the character with index 2 (zero-based).

### Value Format #2: Multiple Accent Positions (Ambiguity)

If the base word has multiple possible accent positions, the format is:

    b'{entry_1}{entry_2}...{entry_N}'

where each entry is:

    b'{pos}\xFE{compressed_tags}\xFF'

- `pos`: a single byte indicating the character index of the stressed vowel
- `\xFE`: separator byte
- `compressed_tags`: a sequence of bytes, each corresponding to a morphological or POS tag (see `ukrainian_word_stress.tags.TAGS`)
- `\xFF`: separator byte

**Example:**

    b'\x02\xFE\x10\x11\xFF\x04\xFE\x12\xFF'

This means two possible accent positions: index 2 (with tags 0x10, 0x11) and index 4 (with tag 0x12).

**Note:** 0xFE and 0xFF are one-byte separators.

## Attribution

When using this data or any derived databases, please include:

```
Ukrainian Word Stress Data
- Trie Database: Copyright (c) 2022 lang-uk, MIT License
  https://github.com/lang-uk/ukrainian-word-stress
```
