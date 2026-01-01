# Data Export Module

Exports derived datasets from the merged linguistic database.

## Structure

```
export/
├── __init__.py
├── export_training_db.py      # Export training database for ML
└── README.md                   # This file
```

## Available Exporters

### export_training_db.py

**Purpose**: Creates optimized SQLite training database for stress prediction models

**Input**: Merged SQL database from `transform/cache/MERGEDSQL_*.sqlite3`

**Output**: Training database at `stress_prediction/data/stress_training.db`

**Features**:

- Variant type classification (single, free_variant, grammatical_homonym, morphological_variant)
- Streaming two-pass algorithm (O(batch_size) memory)
- Reproducibility hashing
- Training/validation/test split support
- Comprehensive statistics

**Usage**:

```bash
# Default paths (auto-detects merged SQL)
python -m src.data_management.export.export_training_db --overwrite

# Custom paths
python -m src.data_management.export.export_training_db \
    --merged-db path/to/merged.sqlite3 \
    --output-db path/to/training.db \
    --overwrite \
    --batch-size 1000
```

**Output Schema**:

- `training_entries` - Main training data table
- `db_metadata` - Export metadata (timestamp, hash, source)
- `db_stats` - Dataset statistics

## Data Flow

```
Sources (Kaikki, TRIE, TXT)
    ↓
transform/merger.py
    ↓
transform/cache/MERGEDSQL_*.sqlite3  ← INPUT
    ↓
export/export_training_db.py
    ↓
stress_prediction/data/stress_training.db  ← OUTPUT (used by ML training)
```

## Future Exporters

- `export_to_lmdb.py` - Export runtime LMDB for stress service
- `export_phonetic_dict.py` - Export IPA pronunciation dictionary
- `export_lemmatizer_data.py` - Export lemmatization lookup tables
