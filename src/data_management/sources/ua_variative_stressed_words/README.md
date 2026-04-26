# ua_variative_stressed_words

Curated list of Ukrainian lemmas that have two or more equally valid stress positions (free variants, not meaning-dependent heteronyms).

## What it contains

- `ua_variative_stressed_words.txt` — line-delimited list of lemmas with free-variant stress (~150 entries). Comment lines start with `#`.
- `variative_parser.py` — parser: reads the list, normalises apostrophes, yields/loads word set.
- `__init__.py` — package exports: `load_variative_words`, `is_variative`, `iter_variative_words`.

## Version & integrity

| Field        | Value                                    |
| ------------ | ---------------------------------------- |
| Version      | 1.0.0                                    |
| Format       | Plain text, UTF-8, line-delimited        |
| Entry count  | ~150 lemmas (manual curation)            |
| Size on disk | < 5 KB                                   |
| License      | Curated manually; no third-party licence |

## How it works

Each non-blank, non-comment line is a Ukrainian lemma whose stress is _freely variable_ — i.e. both stress positions are accepted by the orthographic norm. Examples: `алфавіт`, `договір`, `завжди`.

During the master-DB build (`parsing_merging_service.py`), the merger can query this set to annotate entries with a "variative" flag so the stress service can report ambiguity even when only one stress index was recorded in the raw source data.

The parser does **not** assign specific stress indices; it only identifies _which_ lemmas are variative. The actual stress index data comes from the trie, txt, and kaikki sources.

## How to build / rebuild

No build step required — this is a static curated text file. To add new variative words, append lemmas to `ua_variative_stressed_words.txt` one per line.

```bash
conda activate verseSense-py312
python -c "
from src.data_management.sources.ua_variative_stressed_words import load_variative_words
words = load_variative_words()
print(f'Loaded {len(words)} variative words')
"
```

## How to use

```python
from src.data_management.sources.ua_variative_stressed_words import (
    load_variative_words,
    is_variative,
)

variative = load_variative_words()

# membership test
print(is_variative("алфавіт", variative))   # True
print(is_variative("замок",   variative))   # False  (heteronym, not free variant)

# iterate
from src.data_management.sources.ua_variative_stressed_words import iter_variative_words
for word in iter_variative_words():
    print(word)
```

## Tests

```bash
conda activate verseSense-py312
pytest tests/src/data_management/sources/test_variative_parser.py -v
```

Expected: all tests pass, ~150 words loaded, known entries present.

## Dependencies

- `src/utils/normalize_apostrophe.py` — apostrophe normalisation
- No external libraries required
