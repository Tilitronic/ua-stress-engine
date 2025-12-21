# Stress Database Generator

Generates optimized LMDB database for Ukrainian word stress lookup from multiple data sources.

## Architecture

### Main Pipeline: `generate_db.py`

The main entry point that orchestrates the complete database generation pipeline:

```
STEP 0: Download/Update Data Files
   ↓
STEP 1-2: Parse Sources (Parallel)
   ├── Trie Parser (stress.trie)
   └── TXT Parser (ua_word_stress_dictionary.txt)
   ↓
STEP 3: Merge Dictionaries
   ↓
STEP 4: spaCy Morphology Validation
   ↓
STEP 5: LMDB Export (MsgPack + Append Mode)
   ↓
STEP 6: Quick Verification
   ↓
STEP 7: Automatic Testing
```

**Usage:**

```bash
python generate_db.py
```

## Core Modules

### Data Acquisition

- **`download_data_files.py`** - Downloads and verifies source data files from lang-uk repositories

### Parsers

- **`trie_parser.py`** - Low-level MARISA trie format parser
- **`trie_adapter.py`** - High-level adapter for trie data parsing
- **`txt_parser.py`** - Text dictionary parser for ua_word_stress_dictionary.txt

### Processing

- **`merger.py`** - Intelligent dictionary merger with heteronym handling
- **`spacy_transformer.py`** - Validates and enriches morphological features using spaCy

### Export

- **`lmdb_exporter.py`** - Optimized LMDB exporter and query interface
  - **Optimizations:**
    - MsgPack serialization (30% smaller than JSON)
    - MDB_APPEND mode with sorted keys (10x-50x faster writes)
    - Memory-mapped I/O with zero-copy reads (`buffers=True`)
    - Automatic map_size calculation (1.3x overhead + 1.2x safety)
    - Read-ahead optimization for sequential scans

## Output

**Database Location:** `src/nlp/stress_service/stress.lmdb/`

**Database Stats:**

- ~2.9M entries
- ~200-250 MB file size (MsgPack optimized)
- 288k+ queries/second
- 0.003ms average latency

**Data Format:**

```python
{
    "stress_variants": [0],      # Stress position indices
    "pos": ["NOUN"],             # Part-of-speech tags
    "feats": {                   # Morphological features
        "Case": ["Nom", "Acc"],
        "Gender": ["Masc"],
        "Number": ["Sing"]
    }
}
```

## Data Attribution

This generator processes data from the lang-uk organization:

1. **stress.trie** - Trie database

   - Source: https://github.com/lang-uk/ukrainian-word-stress
   - License: MIT License, Copyright (c) 2022 lang-uk

2. **ua_word_stress_dictionary.txt** - Text dictionary
   - Source: https://github.com/lang-uk/ukrainian-word-stress-dictionary
   - Based on "Словники України" (Ukrainian Linguistic Information Fund)

See `raw_data/DATA_ATTRIBUTION.md` for complete licensing information.

## Directory Structure

```
stress_db_generator/
├── generate_db.py              # Main pipeline orchestrator
├── download_data_files.py      # Data acquisition
├── trie_parser.py              # Trie format parser
├── trie_adapter.py             # Trie data adapter
├── txt_parser.py               # Text dictionary parser
├── merger.py                   # Dictionary merger
├── spacy_transformer.py        # Morphology validator
├── lmdb_exporter.py            # LMDB export/query (MsgPack optimized)
├── raw_data/                   # Source data files
│   ├── stress.trie
│   ├── ua_word_stress_dictionary.txt
│   └── DATA_ATTRIBUTION.md
└── tests/                      # Test suite
    ├── test_lmdb_query.py      # Database testing
    ├── test_modular_pipeline.py # Module tests
    ├── verify_lmdb.py          # Quick verification
    └── README.md
```

## Performance Optimizations

### Write Performance

- **Sorted Keys + MDB_APPEND:** Keys sorted alphabetically before writing enables sequential append mode (10x-50x faster)
- **MsgPack Serialization:** Binary format reduces size by ~30% and CPU overhead
- **Memory-Mapped Writes:** `writemap=True` + `map_async=True` for maximum OS throughput

### Read Performance

- **Zero-Copy Access:** `buffers=True` reads directly from memory map without copying
- **Read-Ahead:** OS-level prefetching for sequential scans
- **No Locking:** `lock=False` for read-only access (multiple readers)
- **Memory-Mapped:** Direct file access, no deserialization overhead until needed

### Space Efficiency

- **Smart Map Sizing:** Samples 1000 entries, estimates total size with 1.56x multiplier
- **MsgPack Compact:** Binary format more space-efficient than JSON
- **Reduced Overhead:** 1.3x LMDB overhead (vs 1.5x for JSON) due to predictable binary format

## Requirements

- Python 3.13+
- lmdb
- msgpack
- marisa-trie
- spacy (with uk_core_news_sm model)

## Testing

See `tests/README.md` for test documentation.

Quick test:

```bash
cd tests
python test_lmdb_query.py
```
