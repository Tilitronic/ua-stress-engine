# Quick Reference Guide

## Build Database

```bash
# Build complete database from trie + txt
python src/stress_db_generator/build_lmdb.py

# Expected output:
# ✓ Trie parsing:     ~25s (2.9M words)
# ✓ TXT parsing:      ~28s (2.9M words)
# ✓ Merging:          ~10s
# ✓ Transform:         ~6s
# ✓ LMDB export:      ~15s
# ✓ Verification:      <1s
# Total: ~84 seconds
```

## Query Database

```python
from pathlib import Path
from src.stress_db_generator.lmdb_exporter import LMDBQuery

# Open database
db_path = Path("src/nlp/stress_service/stress.lmdb")
with LMDBQuery(db_path) as db:
    # Lookup word
    forms = db.lookup("атлас")

    # Print results
    for form in forms:
        print(form)
        # {'stress_variants': [0], 'pos': ['NOUN'], 'feats': {...}}

    # Get statistics
    stats = db.get_stats()
    print(f"Entries: {stats['entries']:,}")
    print(f"Size: {stats['size_bytes'] / (1024*1024):.2f} MB")

    # Prefix search
    words = db.prefix_search("атл", limit=10)
    print(words)
```

## Module Usage

### TXT Parser

```python
from pathlib import Path
from src.stress_db_generator.txt_parser import parse_txt_dictionary

# Parse text file
txt_data = parse_txt_dictionary(Path("dictionary.txt"))

# Result: Dict[str, List[Tuple[List[int], Optional[str]]]]
# {"слово": [([0], "definition"), ([1], None)]}
```

### Trie Adapter

```python
from pathlib import Path
from src.stress_db_generator.trie_adapter import parse_trie_data

# Parse trie file
trie_data = parse_trie_data(Path("stress.trie"))

# Result: Dict[str, List[Tuple[List[int], Dict]]]
# {"слово": [([0], {"upos": "NOUN", "Case": "Nom"})]}
```

### Merger

```python
from src.stress_db_generator.merger import DictionaryMerger

# Create merger
merger = DictionaryMerger()

# Add data sources
merger.add_trie_data(trie_data)
merger.add_txt_data(txt_data)

# Get merged dictionary
merged = merger.get_dictionary()

# Get statistics
stats = merger.get_statistics()
print(f"Total words: {stats['total_unique_words']:,}")
print(f"Merged sources: {stats['merged']:,}")
```

### spaCy Transformer

```python
from src.stress_db_generator.spacy_transformer import SpaCyTransformer

# Create transformer
transformer = SpaCyTransformer(strict=False)

# Transform to spaCy format
spacy_dict = transformer.transform(merged_dict)

# Check for warnings
if transformer.warnings:
    print(f"Warnings: {len(transformer.warnings)}")

# Use transformed data
for word, forms in spacy_dict.items():
    for form in forms:
        print(form.to_spacy_format())  # "Case=Nom|Number=Sing"
```

### LMDB Exporter

```python
from pathlib import Path
from src.stress_db_generator.lmdb_exporter import LMDBExporter

# Create exporter
exporter = LMDBExporter(
    db_path=Path("stress.lmdb"),
    map_size=2 * 1024 * 1024 * 1024  # 2 GB
)

# Export to LMDB
export_dict = {
    key: [form.to_dict() for form in forms]
    for key, forms in spacy_dict.items()
}
exporter.export_raw(export_dict)
```

## Test Modules

```bash
# Test all modules independently
python src/stress_db_generator/test_modular_pipeline.py

# Expected output:
# ✓ TEST 1: TXT Parser Module
# ✓ TEST 2: Trie Adapter Module
# ✓ TEST 3: Merger Module
# ✓ TEST 4: spaCy Transformer Module
# ✓ TEST 5: Full Pipeline Integration
```

## Data Structures

### WordForm (Merger Output)

```python
from src.stress_db_generator.merger import WordForm

form = WordForm(
    stress_variants=[0, 1],           # Vowel indices
    pos=["NOUN"],                     # List of POS tags
    feats={                           # Dict of feature lists
        "Case": ["Nom", "Acc"],
        "Number": ["Sing"]
    },
    lemma="слово",                    # Optional base form
    source="merged"                   # Data source
)
```

### SpaCyWordForm (Transformer Output)

```python
from src.stress_db_generator.spacy_transformer import SpaCyWordForm

form = SpaCyWordForm(
    stress_variants=[0],
    pos=["NOUN"],
    feats={"Case": ["Nom"], "Number": ["Sing"]}
)

# Export to dict
form_dict = form.to_dict()
# {'stress_variants': [0], 'pos': ['NOUN'], 'feats': {...}}

# Export to spaCy format
morph_str = form.to_spacy_format()
# "Case=Nom|Number=Sing"

# Check morphology match
matches = form.matches_morphology({"Case": "Nom", "Number": "Sing"})
# True
```

## File Paths

```
Project Structure:

src/stress_db_generator/
├── raw_data/
│   ├── stress.trie                    # Input: Trie data
│   └── ua_word_stress_dictionary.txt  # Input: TXT data
│
├── txt_parser.py                      # Module: Parse TXT
├── trie_adapter.py                    # Module: Parse trie
├── merger.py                          # Module: Merge data
├── spacy_transformer.py               # Module: Transform
├── lmdb_exporter.py                   # Module: Export/query
├── build_lmdb.py                      # Script: Build DB
├── test_modular_pipeline.py           # Script: Test modules
│
├── PIPELINE_ARCHITECTURE.md           # Docs: Architecture
├── REFACTORING_SUMMARY.md            # Docs: Summary
├── ARCHITECTURE_DIAGRAMS.md           # Docs: Diagrams
└── QUICK_REFERENCE.md                 # Docs: This file

Output:
src/nlp/stress_service/
└── stress.lmdb/                       # Output: Database
```

## Common Tasks

### Add New Data Source

```python
# 1. Create new parser module
# my_parser.py
def parse_my_format(path):
    # Parse your format
    return {
        "word": [
            ([stress_indices], {"upos": "NOUN", ...})
        ]
    }

# 2. Add to merger
from src.stress_db_generator.merger import DictionaryMerger
merger = DictionaryMerger()
merger.add_trie_data(trie_data)
merger.add_txt_data(txt_data)
# Add custom method if needed
# merger.add_custom_data(my_data)
```

### Export to Different Format

```python
# 1. Create new exporter
# pickle_exporter.py
import pickle

def export_to_pickle(dictionary, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(dictionary, f)

# 2. Use in pipeline
export_dict = {
    key: [form.to_dict() for form in forms]
    for key, forms in spacy_dict.items()
}
export_to_pickle(export_dict, Path("stress.pkl"))
```

### Validate Data Quality

```python
from src.stress_db_generator.merger import DictionaryMerger

merger = DictionaryMerger()
merger.add_trie_data(trie_data)
merger.add_txt_data(txt_data)

stats = merger.get_statistics()

# Check coverage
print(f"Words with morphology: {stats['words_with_morphology']:,}")
print(f"Heteronyms: {stats['heteronyms']:,}")
print(f"Merge rate: {stats['merged'] / stats['total_unique_words'] * 100:.1f}%")

# Check for issues
if stats['words_with_morphology'] < stats['total_unique_words'] * 0.5:
    print("⚠️  Warning: Less than 50% morphology coverage")
```

## Performance Tips

### Optimize Build Time

```python
# Skip TXT parsing if not needed (use trie only)
# Comment out in build_lmdb.py:
# txt_data = parse_txt_dictionary(txt_path)
# merger.add_txt_data(txt_data)

# Result: ~40 seconds faster
```

### Reduce Memory Usage

```python
# Use smaller map_size for LMDB
exporter = LMDBExporter(
    db_path=lmdb_path,
    map_size=1 * 1024 * 1024 * 1024  # 1 GB instead of 2 GB
)
```

### Optimize Queries

```python
# Use context manager for automatic cleanup
with LMDBQuery(db_path) as db:
    # Multiple queries in same session
    forms1 = db.lookup("слово1")
    forms2 = db.lookup("слово2")
    forms3 = db.lookup("слово3")
# Automatically closed

# Don't do this (creates multiple connections):
db = LMDBQuery(db_path)
forms1 = db.lookup("слово1")
db.close()
db = LMDBQuery(db_path)
forms2 = db.lookup("слово2")
db.close()
```

## Troubleshooting

### Database Build Fails

```bash
# Check input files exist
ls src/stress_db_generator/raw_data/

# Expected:
# stress.trie
# ua_word_stress_dictionary.txt

# Rebuild from scratch
rm -rf src/nlp/stress_service/stress.lmdb
python src/stress_db_generator/build_lmdb.py
```

### Query Returns None

```python
# Check key normalization
from src.stress_db_generator.txt_parser import TXTParser
parser = TXTParser()

# Your query
word = "Слово"
normalized = parser.generate_key(word)
print(f"'{word}' → '{normalized}'")  # "слово"

# Query with normalized key
with LMDBQuery(db_path) as db:
    forms = db.lookup(normalized)
```

### Memory Issues

```bash
# Monitor memory during build
# On Windows:
tasklist /FI "IMAGENAME eq python.exe"

# If out of memory:
# 1. Close other applications
# 2. Use smaller dataset for testing
# 3. Increase system virtual memory
```

### Validation Warnings

```python
# Check warnings from transformer
transformer = SpaCyTransformer(strict=False)
spacy_dict = transformer.transform(merged_dict)

if transformer.warnings:
    print(f"Total warnings: {len(transformer.warnings)}")
    for warning in transformer.warnings[:10]:
        print(f"  • {warning}")

# To fail on invalid data:
transformer = SpaCyTransformer(strict=True)
try:
    spacy_dict = transformer.transform(merged_dict)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Statistics Reference

### Typical Build Statistics

```
Merge Statistics:
  Total unique words:      2,858,922    # All unique dictionary entries
  Total word forms:        2,908,444    # Including stress variants
  Heteronyms:                 46,667    # Words with multiple stress
  Words with morphology:      36,887    # Words with POS/features
  From trie only:                  0    # Only in trie source
  From txt only:               6,723    # Only in txt source
  Merged sources:          2,848,862    # Combined from both
```

### Database Statistics

```
Database Statistics:
  Entries: 2,858,922                    # Total words in database
  Size: 20-30 MB                        # Compressed size on disk

Query Performance:
  Single lookup:    <1 ms               # Memory-mapped access
  Prefix search:    <10 ms              # For 100 results
  Throughput:       100,000+ queries/s  # Sustained rate
```

## Resources

- **Architecture:** `PIPELINE_ARCHITECTURE.md`
- **Refactoring:** `REFACTORING_SUMMARY.md`
- **Diagrams:** `ARCHITECTURE_DIAGRAMS.md`
- **Tests:** `test_modular_pipeline.py`
- **Build:** `build_lmdb.py`

## Support

For issues or questions:

1. Check documentation files
2. Run test suite: `python test_modular_pipeline.py`
3. Verify input files exist
4. Check database build logs
5. Review error messages carefully
