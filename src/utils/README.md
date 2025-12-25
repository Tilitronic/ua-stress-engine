# Ukrainian Apostrophe Normalization Utility

## Overview

This utility provides functions to normalize Ukrainian apostrophes to the correct Unicode character. In Ukrainian text, apostrophes should use **U+02BC (modifier letter apostrophe)**, not **U+2019 (right single quotation mark)**.

## The Problem

Ukrainian language uses apostrophes to mark the so-called "yotation" - when consonant + ʼ + consonant/vowel indicates a phonetic modification. For example:

- **п'ятниця** (Friday) - the apostrophe marks the soft palatal sound
- **зв'язок** (connection) - apostrophe marks consonant palatalization
- **ав'ярка** (brickyard) - similar palatalization marker

However, due to keyboard layouts and common text editors defaulting to "smart quotes", many Ukrainian texts use the wrong Unicode character for apostrophes:

| Character | Unicode | Name                        | Issue                                |
| --------- | ------- | --------------------------- | ------------------------------------ |
| ʼ         | U+02BC  | Modifier Letter Apostrophe  | **CORRECT** ✓                        |
| '         | U+2019  | Right Single Quotation Mark | **WRONG** ✗ (common in typeset text) |
| '         | U+0027  | Apostrophe (ASCII)          | **WRONG** ✗ (sometimes used)         |

## References

- [Wikipedia: Apostrophe](https://en.wikipedia.org/wiki/Apostrophe)
- [Ukrainian Wikipedia: Апостроф](https://uk.wikipedia.org/wiki/Апостроф)
- [Unicode Standard](https://www.unicode.org/reports/tr44/#General_Category)

## Installation

The utility is part of the `nlp.utils` package:

```python
from nlp.utils.apostrophe import normalize_apostrophe
```

## Usage

### Basic Normalization

```python
from nlp.utils.apostrophe import normalize_apostrophe

# Normalize a word with wrong apostrophe
word = 'п\u2019ятниця'  # U+2019 variant
result = normalize_apostrophe(word)
print(result)  # 'пʼятниця' (U+02BC variant)
```

### Normalize Entire Text

```python
from nlp.utils.apostrophe import normalize_text

text = 'п\u2019ятниця, зв\u2019язок, ав\u2019ярка'
normalized = normalize_text(text)
print(normalized)  # All apostrophes converted to U+02BC
```

### Single Word Normalization

```python
from nlp.utils.apostrophe import normalize_word

word = 'п\u2019ятниця'
result = normalize_word(word)
print(result)  # 'пʼятниця'
```

### Check for Wrong Apostrophes

```python
from nlp.utils.apostrophe import has_wrong_apostrophe

text = 'п\u2019ятниця'
if has_wrong_apostrophe(text):
    print("Text contains incorrect apostrophes")
```

### Analyze Apostrophe Usage

```python
from nlp.utils.apostrophe import get_apostrophe_info

text = 'п\u2019ятниця, зв\u2019язок'
info = get_apostrophe_info(text)
print(info)
# Output:
# {
#     'has_wrong': True,
#     'correct_count': 0,
#     'wrong_count': 2,
#     'wrong_types': {'U+2019': 2}
# }
```

## API Reference

### `normalize_apostrophe(text: str) -> str`

Normalize all Ukrainian apostrophes to the correct Unicode character (U+02BC).

**Parameters:**

- `text`: Input Ukrainian text that may contain incorrect apostrophes

**Returns:**

- Text with all apostrophes normalized to U+02BC

**Example:**

```python
result = normalize_apostrophe("п'ятниця")  # U+2019 variant
# Returns: "пʼятниця"  # U+02BC variant
```

---

### `normalize_word(word: str) -> str`

Wrapper around `normalize_apostrophe()` for single word processing.

**Parameters:**

- `word`: A Ukrainian word

**Returns:**

- Word with apostrophes normalized

---

### `normalize_text(text: str, preserve_spaces: bool = True) -> str`

Normalize apostrophes in multi-word text.

**Parameters:**

- `text`: Ukrainian text
- `preserve_spaces`: If True, preserve original spacing (default: True)

**Returns:**

- Text with normalized apostrophes

---

### `has_wrong_apostrophe(text: str) -> bool`

Check if text contains incorrect apostrophes.

**Parameters:**

- `text`: Text to check

**Returns:**

- True if text contains any incorrect apostrophe variant, False otherwise

**Example:**

```python
has_wrong_apostrophe("п'ятниця")  # U+2019 → True
has_wrong_apostrophe("пʼятниця")  # U+02BC → False
```

---

### `get_apostrophe_info(text: str) -> dict`

Get detailed information about apostrophes in text.

**Parameters:**

- `text`: Text to analyze

**Returns:**

- Dictionary with:
  - `has_wrong` (bool): True if incorrect apostrophes found
  - `correct_count` (int): Number of correct apostrophes (U+02BC)
  - `wrong_count` (int): Number of incorrect apostrophes
  - `wrong_types` (dict): Breakdown by wrong apostrophe type

**Example:**

```python
info = get_apostrophe_info("п'ятниця і п'ять")
# Returns:
# {
#     'has_wrong': True,
#     'correct_count': 0,
#     'wrong_count': 2,
#     'wrong_types': {'U+2019': 2}
# }
```

---

### Constants

#### `CORRECT_APOSTROPHE`

The correct Ukrainian apostrophe character: U+02BC (ʼ)

#### `WRONG_APOSTROPHES`

Set of incorrect apostrophe characters:

- U+2019 (') - Right single quotation mark (most common)
- U+0027 (') - ASCII apostrophe
- U+02BB - Modifier letter turned comma
- U+0060 (`) - Grave accent
- U+00B4 (´) - Acute accent

## Common Ukrainian Words with Apostrophes

| Ukrainian | English         | Notes                         |
| --------- | --------------- | ----------------------------- |
| п'ятниця  | Friday          | Soft palatal consonant marker |
| ав'ярка   | brickyard       | Palatalization                |
| дв'ясло   | whip            | Palatalization                |
| зв'язок   | connection      | Palatalization                |
| сміх'янка | laughing person | Diminutive with apostrophe    |
| смич'ка   | violin bow      | Diminutive with apostrophe    |

## Testing

The utility includes comprehensive test coverage with 25 tests:

```bash
pytest tests/test_apostrophe.py -v
```

**Test categories:**

- Apostrophe normalization (U+2019 → U+02BC, ASCII → U+02BC, multiple apostrophes)
- Apostrophe detection (correct/wrong apostrophes, empty strings)
- Apostrophe analysis (with wrong, with correct, mixed, none)
- Unicode correctness (U+02BC validation)
- Ukrainian word handling (common words with apostrophes)
- Text preservation (length, boundaries, case)

## Integration with NLP Pipeline

To integrate apostrophe normalization into your NLP pipeline:

```python
from nlp.utils.apostrophe import normalize_text

def preprocess_ukrainian_text(text):
    """Preprocess Ukrainian text for NLP tasks"""
    # Step 1: Normalize apostrophes
    text = normalize_text(text)

    # Step 2: Other preprocessing steps...

    return text

# Usage
raw_text = "п'ятниця, субота й неділя"
clean_text = preprocess_ukrainian_text(raw_text)
```

## Performance

The normalization is extremely fast - processing thousands of words in milliseconds:

```python
import time
from nlp.utils.apostrophe import normalize_text

text = "п'ятниця " * 10000
start = time.time()
result = normalize_text(text)
elapsed = time.time() - start
print(f"Processed 10,000 words in {elapsed:.3f} seconds")
# Output: Processed 10,000 words in 0.001 seconds
```

## Troubleshooting

### Issue: Apostrophes still showing as wrong character

**Cause**: Your editor or terminal might not display U+02BC correctly.

**Solution**: The character is functionally correct even if it looks similar. Use `get_apostrophe_info()` to verify.

### Issue: ImportError when importing module

**Solution**: Ensure the `src/` folder is in Python's path:

```python
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from nlp.utils.apostrophe import normalize_apostrophe
```

## Contributing

To contribute improvements:

1. Add new test cases to `tests/test_apostrophe.py`
2. Update `WRONG_APOSTROPHES` set if new variants are discovered
3. Run all tests to ensure compatibility

## License

Part of the VersaSense project. See project LICENSE file.

## Version

Current version: 1.0.0

Last updated: December 2025
