# Ukrainian Lemmatizer (Hybrid: Stanza + pymorphy3)

This module provides robust, production-grade lemmatization for Ukrainian, combining neural/contextual (Stanza) and dictionary-based (pymorphy3/VESUM) approaches. It is designed for high accuracy, type safety, and easy integration into NLP pipelines.

## Features

- **Hybrid Lemmatization:** Uses Stanza for context-aware lemmatization and pymorphy3 for dictionary lookups.
- **Type Safety:** All public APIs return Pydantic v2 models for strict type validation.
- **Production Ready:** Tested, modular, and compatible with spaCy/UD pipelines.
- **Customizable:** Easily extendable for new sources or lemmatization strategies.

## Installation

1. Ensure you have Python 3.8–3.12.
2. Install dependencies:
   ```sh
   pip install stanza pymorphy3 pydantic
   ```
3. (Recommended) Install the package in editable mode from the project root:
   ```sh
   pip install -e .
   ```

## Usage

```python
from lemmatizer.lemmatizer import Lemmatizer, TokenLemma

lemmatizer = Lemmatizer()

# Lemmatize a single word (dictionary-based)
lemma = lemmatizer.get_lemma('словами')
print(lemma)  # e.g., 'слово'

# Lemmatize a sentence (contextual)
results = lemmatizer.analyze_sentence('Ми говоримо українською мовою.')
for token in results:
    print(token.text, token.lemma, token.pos)
```

## API

### `Lemmatizer`

- `get_lemma(word: str) -> str`  
  Returns the dictionary lemma for a word.
- `analyze_sentence(sentence: str) -> List[TokenLemma]`  
  Returns a list of `TokenLemma` (Pydantic) for each token in the sentence.

### `TokenLemma` (Pydantic model)

- `text`: str — The original token
- `lemma`: str — The lemmatized form
- `pos`: str — Part of speech (UD tag)

## Testing

Run unit tests from the project root:

```sh
python tests/src/lemmatizer/test_lemmatizer.py
```

## Notes

- Stanza will download models on first use (requires internet connection).
- For best results, use in a conda or virtual environment with supported Python version.

## Authors

- See main project README for contributors.
