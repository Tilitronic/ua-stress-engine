# txt_ua_stresses — Text Stress Dictionary Source

Ukrainian word-stress dictionary from `lang-uk/ukrainian-word-stress-dictionary`: plain text with embedded combining-acute stress marks (U+0301), ~2.9 M word forms.

## What it contains

- `ua_word_stress_dictionary.txt` (Git LFS) — main dictionary, ~69 MB, one stressed word form per line.
- `sample_stress_dict.txt` — 30-entry fixture file used for offline unit tests.
- `txt_stress_parser.py` — parser: `split_words`, `extract_stress_indices`, `stream_txt_to_lmdb`.
- `stress_db_file_manager.py` — optional auto-download/update from the upstream GitHub repo.

## Version & integrity

| Field           | Value                                                             |
| --------------- | ----------------------------------------------------------------- |
| Source          | https://github.com/lang-uk/ukrainian-word-stress-dictionary       |
| Based on        | "Словники України", ULIF, National Academy of Sciences of Ukraine |
| License         | See upstream repo (data from public dictionary corpus)            |
| Format          | Plain text, UTF-8, one form per line                              |
| Approx. entries | ~2.9 M word forms                                                 |
| Size (raw)      | ~69 MB (Git LFS)                                                  |

## How it works

Each non-blank, non-comment line contains a Ukrainian word form with the stressed vowel marked by a combining acute accent (U+0301) immediately after the vowel character.

Examples:

```
замо́к    →  form="замок", stress_vowel_index=2   (3rd char counting from 0)
за́мок    →  form="замок", stress_vowel_index=0   (1st vowel)
по́ми́лка  →  form="помилка", stress_vowel_index=[0,1]  (double stress)
```

The parser:

1. Reads each line and calls `extract_stress_indices(form)` to find which vowel is stressed.
2. Strips the acute mark to get the base form.
3. Lemmatises via `Lemmatizer.get_lemma()`.
4. Yields `(lemma, LinguisticEntry)` pairs for aggregation.

## How to build / rebuild

```bash
conda activate verseSense-py312
python -m src.data_management.transform.parsing_merging_service
```

To refresh `ua_word_stress_dictionary.txt` from upstream:

```bash
python -c "
from src.data_management.sources.txt_ua_stresses.stress_db_file_manager import ensure_latest_db_file
ensure_latest_db_file()
"
```

## How to use

```python
from src.data_management.sources.txt_ua_stresses.txt_stress_parser import (
    extract_stress_indices,
    split_words,
)
import re, unicodedata

# Extract stress from a marked form
def strip_stress(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

marked = "замо́к"
indices = extract_stress_indices(marked)  # [2] — char index of 'о'
base = strip_stress(marked)               # "замок"
```

## Tests

```bash
conda activate verseSense-py312
pytest tests/src/data_management/sources/test_txt_parser.py -v
```

Expected: all tests pass using `sample_stress_dict.txt` (no network access, no large file).

## Dependencies

- `src/data_management/transform/data_unifier.py`
- `src/data_management/transform/merger.py`
- `src/utils/normalize_apostrophe.py`
- `src/lemmatizer/lemmatizer.py`
- `tqdm`, `lmdb`, `msgpack`

## Attribution

When using this data or any derived databases, please include:

```
Ukrainian Word Stress Data
- Text Dictionary: Словники України
  https://github.com/lang-uk/ukrainian-word-stress-dictionary
```
