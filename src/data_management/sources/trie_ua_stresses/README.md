# trie_ua_stresses — lang-uk TRIE Stress Source

Ukrainian word-stress data from the `lang-uk/ukrainian-word-stress` project: a compact `marisa_trie.BytesTrie` encoding ~2.9 M word forms with stress positions and morphological tags.

## What it contains

- `stress.trie` — compiled `marisa_trie.BytesTrie` file (~12 MB). Keys are word forms; values encode stress positions and morphological tags in a compact byte format.
- `trie_stress_parser.py` — parser: `char_positions_to_vowel_indices`, `stream_trie_to_lmdb`, tag decompression via `TAG_BY_BYTE`.
- `stress_db_file_manager.py` — optional auto-download/update of `stress.trie` from the upstream GitHub repo.
- `__init__.py` — package init.

## Version & integrity

| Field           | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Source          | https://github.com/lang-uk/ukrainian-word-stress             |
| License         | MIT License, Copyright (c) 2022 lang-uk                      |
| Format          | `marisa_trie.BytesTrie`                                      |
| Approx. entries | ~2.9 M word forms                                            |
| Size (raw)      | ~12 MB                                                       |

## How it works

The trie maps word forms (no stress marks) to a compact byte value:

**Format 1 — single stress position:**
```
b'\x02'   →  stress at character index 2
```

**Format 2 — multiple stress positions (ambiguity):**
```
b'\x02\xFE\x10\x11\xFF\x04\xFE\x12\xFF'
  pos=2, tags=[0x10,0x11] | pos=4, tag=[0x12]
```
- `\xFE` separates position from tags
- `\xFF` separates records
- Each tag byte maps to a `(UDFeatKey, value)` pair via `TAG_BY_BYTE`

The parser converts trie character positions to 0-based vowel indices using `char_positions_to_vowel_indices()`, then yields `(lemma, LinguisticEntry)` pairs.

## How to build / rebuild

`stress.trie` is included in the repo.  If the upstream is updated, refresh it:

```bash
conda activate verseSense-py312
python -c "
from src.data_management.sources.trie_ua_stresses.stress_db_file_manager import ensure_latest_db_file
ensure_latest_db_file()
"
```

Then re-run the full pipeline:

```bash
python -m src.data_management.transform.parsing_merging_service
```

## How to use

```python
import marisa_trie
from src.data_management.sources.trie_ua_stresses.trie_stress_parser import (
    char_positions_to_vowel_indices,
    TAG_BY_BYTE,
)

trie = marisa_trie.BytesTrie()
trie.load("src/data_management/sources/trie_ua_stresses/stress.trie")

# Raw lookup
results = trie["замок"]   # list of byte values
print(results[0])          # e.g. b'\x03\xFE\x61\x20\x11\xFF\x05\xFE\x61\x20\x12\xFF'
```

## Tests

```bash
conda activate verseSense-py312
pytest tests/src/data_management/sources/test_trie_parser.py -v
```

Expected: all tests pass; trie file loads; smoke words return correct vowel indices.

## Dependencies

- `marisa-trie` — trie access (`pip install marisa-trie`)
- `src/data_management/transform/data_unifier.py`
- `src/data_management/transform/merger.py`
- `src/utils/normalize_apostrophe.py`
- `src/lemmatizer/lemmatizer.py` — lemmatisation during parsing

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
