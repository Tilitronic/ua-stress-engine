# VerseSense Backend

Ukrainian word stress analysis engine for NLP tasks.

## Setup

Before running the project, download the required large data files:

```bash
python download_data_files.py
```

This will automatically download:

- `stress.trie` - Stress dictionary trie database
- `ua_word_stress_dictionary.txt` - Ukrainian word stress dictionary

## Project Structure

```
src/
  nlp/
    stress/          # Stress analysis module
    syllables/       # Syllabification module
    tokenization/    # Tokenization module
    phonetic/        # Phonetic analysis
    wordStress/      # Word stress handling
tests/               # Unit tests
```

## Installation

1. Clone the repository
2. Run `python download_data_files.py`
3. Install dependencies from `environment.yml`

```bash
conda env create -f environment.yml
conda activate versasense
```
