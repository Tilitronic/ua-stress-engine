# Stress Database Generator Tests

This directory contains test scripts for the LMDB stress database generator pipeline.

## Test Files

### `test_lmdb_query.py`

Comprehensive LMDB database testing:

- Database statistics and integrity
- Exact word lookups (7 test words)
- Prefix search functionality
- Performance benchmarking (queries/second)
- Sample word listing

**Usage:**

```bash
cd src/stress_db_generator/tests
python test_lmdb_query.py
```

### `test_modular_pipeline.py`

Tests individual pipeline modules:

- TXT parser functionality
- Trie adapter functionality
- Dictionary merger logic
- spaCy transformer validation
- End-to-end pipeline integration

**Usage:**

```bash
cd src/stress_db_generator/tests
python test_modular_pipeline.py
```

### `verify_lmdb.py`

Quick verification script for LMDB database:

- Basic database statistics
- Sample word lookups
- Database health check

**Usage:**

```bash
cd src/stress_db_generator/tests
python verify_lmdb.py
```

### `test_parser.py`

Legacy test for old parser implementation (may be deprecated).

## Running Tests

All tests expect to be run from within the `tests` directory or with proper Python path configuration. They automatically add parent directories to `sys.path` for imports.

The main database should be located at:

```
src/nlp/stress_service/stress.lmdb/
```

## Test Data

Test files use sample Ukrainian words to verify:

- Stress position marking
- Multiple stress variants (heteronyms)
- Morphological features (POS, Case, Gender, Number)
- Data integrity and serialization
