# Database Generator - Multi-Format Export Guide

## Overview

`generate_db.py` is a scalable database generator that supports multiple export formats from a single pipeline run.

## Supported Formats

| Format          | File Size | Speed   | Use Case                           |
| --------------- | --------- | ------- | ---------------------------------- |
| **LMDB**        | ~20 MB    | Fastest | Production queries (memory-mapped) |
| **Pickle**      | ~110 MB   | Fast    | Python-only, native serialization  |
| **JSON**        | ~250 MB   | Slow    | Human-readable, cross-platform     |
| **MessagePack** | ~150 MB   | Medium  | Compact binary, cross-language     |

## Usage

### Default (LMDB only)

```bash
python src/stress_db_generator/generate_db.py
```

### Multiple Formats

```bash
# Export to LMDB and Pickle
python src/stress_db_generator/generate_db.py lmdb pickle

# Export to all formats
python src/stress_db_generator/generate_db.py lmdb pickle json msgpack

# Export to JSON only
python src/stress_db_generator/generate_db.py json
```

## Architecture

### Abstract Base Class

```python
class DatabaseExporter(ABC):
    """Base class for all exporters"""

    @abstractmethod
    def export(self, data: Dict[str, List[Dict]]) -> None:
        """Export data to format"""
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Return format name"""
        pass

    def verify(self, sample_words: List[str]) -> Dict:
        """Verify exported database"""
        pass
```

### Built-in Exporters

#### 1. LMDBExporter

- **Format:** Memory-mapped database
- **Speed:** Ultra-fast reads (millions/sec)
- **Size:** ~20 MB (compressed)
- **Dependencies:** `lmdb`
- **Best for:** Production queries

```python
exporter = LMDBExporter(
    output_path=Path("stress.lmdb"),
    map_size=2 * 1024 * 1024 * 1024  # 2 GB
)
```

#### 2. PickleExporter

- **Format:** Python binary serialization
- **Speed:** Fast
- **Size:** ~110 MB
- **Dependencies:** Built-in
- **Best for:** Python-only applications

```python
exporter = PickleExporter(
    output_path=Path("stress.pkl")
)
```

#### 3. JSONExporter

- **Format:** Human-readable text
- **Speed:** Slower (large files)
- **Size:** ~250 MB
- **Dependencies:** Built-in
- **Best for:** Debugging, cross-platform

```python
exporter = JSONExporter(
    output_path=Path("stress.json")
)
```

#### 4. MessagePackExporter

- **Format:** Compact binary
- **Speed:** Medium
- **Size:** ~150 MB
- **Dependencies:** `msgpack`
- **Best for:** Cross-language applications

```python
exporter = MessagePackExporter(
    output_path=Path("stress.msgpack")
)
```

## Adding New Formats

### Step 1: Create Exporter Class

```python
class MyCustomExporter(DatabaseExporter):
    """Export to my custom format"""

    def get_format_name(self) -> str:
        return "My Custom Format"

    def export(self, data: Dict[str, List[Dict]]) -> None:
        # Implement export logic
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, 'wb') as f:
            # Write your format
            pass

        print(f"\n      ✓ Custom export complete: {len(data):,} words")

    def verify(self, sample_words: List[str]) -> Dict:
        # Implement verification
        return {
            "status": "success",
            "entries": 0,
            "size_bytes": 0,
            "sample_found": "0/0"
        }
```

### Step 2: Register in EXPORTERS

```python
EXPORTERS = {
    "lmdb": LMDBExporter,
    "pickle": PickleExporter,
    "json": JSONExporter,
    "msgpack": MessagePackExporter,
    "custom": MyCustomExporter,  # Add here
}
```

### Step 3: Add Output Path

```python
output_paths = {
    "lmdb": output_dir / "stress.lmdb",
    "pickle": output_dir / "stress.pkl",
    "json": output_dir / "stress.json",
    "msgpack": output_dir / "stress.msgpack",
    "custom": output_dir / "stress.custom",  # Add here
}
```

### Step 4: Use

```bash
python src/stress_db_generator/generate_db.py custom
```

## Example: SQLite Exporter

```python
class SQLiteExporter(DatabaseExporter):
    """Export to SQLite database"""

    def get_format_name(self) -> str:
        return "SQLite (Relational Database)"

    def export(self, data: Dict[str, List[Dict]]) -> None:
        import sqlite3

        # Create parent directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create database
        conn = sqlite3.connect(self.output_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS words (
                word TEXT PRIMARY KEY,
                forms TEXT
            )
        """)

        # Insert data
        for word, forms in data.items():
            forms_json = json.dumps(forms, ensure_ascii=False)
            cursor.execute("INSERT INTO words VALUES (?, ?)", (word, forms_json))

        conn.commit()
        conn.close()

        print(f"\n      ✓ SQLite export complete: {len(data):,} words")

    def verify(self, sample_words: List[str]) -> Dict:
        import sqlite3

        conn = sqlite3.connect(self.output_path)
        cursor = conn.cursor()

        # Count total
        cursor.execute("SELECT COUNT(*) FROM words")
        total = cursor.fetchone()[0]

        # Test samples
        found = 0
        for word in sample_words:
            cursor.execute("SELECT forms FROM words WHERE word = ?", (word,))
            if cursor.fetchone():
                found += 1

        conn.close()

        return {
            "status": "success",
            "entries": total,
            "size_bytes": self.output_path.stat().st_size,
            "sample_found": f"{found}/{len(sample_words)}"
        }
```

Register it:

```python
EXPORTERS = {
    # ... existing ...
    "sqlite": SQLiteExporter,
}
```

Use it:

```bash
python src/stress_db_generator/generate_db.py sqlite
```

## Performance Comparison

### Export Speed (2.86M words)

```
LMDB:        ~10 seconds
Pickle:      ~3 seconds
JSON:        ~45 seconds
MessagePack: ~8 seconds
```

### File Sizes

```
LMDB:        ~20 MB (memory-mapped, compressed)
Pickle:      ~110 MB (Python binary)
JSON:        ~250 MB (text, formatted)
MessagePack: ~150 MB (binary, compact)
```

### Query Speed

```
LMDB:        <1ms per lookup
Pickle:      ~50ms (full load), then instant
JSON:        ~200ms (full load), then instant
MessagePack: ~100ms (full load), then instant
```

## Output Structure

All formats export the same data structure:

```python
{
    "атлас": [
        {
            "stress_variants": [0],
            "pos": ["NOUN"],
            "feats": {
                "Case": ["Acc", "Nom"],
                "Gender": ["Masc"],
                "Number": ["Sing"]
            }
        },
        {
            "stress_variants": [1],
            "pos": ["NOUN"],
            "feats": {
                "Case": ["Acc", "Nom"],
                "Gender": ["Masc"],
                "Number": ["Sing"]
            }
        }
    ]
}
```

## Best Practices

### For Production

```bash
# Use LMDB for fastest queries
python generate_db.py lmdb
```

### For Development

```bash
# Use JSON for debugging + LMDB for testing
python generate_db.py json lmdb
```

### For Distribution

```bash
# Use Pickle for Python packages + JSON for docs
python generate_db.py pickle json
```

### For Cross-Platform

```bash
# Use MessagePack for multi-language support
python generate_db.py msgpack
```

## Verification

Each exporter includes verification:

- Counts total entries
- Measures file size
- Tests sample lookups
- Returns detailed statistics

Example output:

```
📦 Verifying LMDB...
    ✓ Entries: 2,858,922
    ✓ Size: 20.34 MB
    ✓ Sample lookups: 4/4

📦 Verifying PICKLE...
    ✓ Entries: 2,858,922
    ✓ Size: 108.47 MB
    ✓ Sample lookups: 4/4
```

## Migration from build_lmdb.py

### Old

```bash
python src/stress_db_generator/build_lmdb.py
```

### New

```bash
# Same result (LMDB default)
python src/stress_db_generator/generate_db.py

# Or explicit
python src/stress_db_generator/generate_db.py lmdb
```

## Extensibility Benefits

✅ **Single Pipeline:** Parse data once, export to multiple formats
✅ **No Duplication:** All exporters use same validated data
✅ **Easy Testing:** Compare formats side-by-side
✅ **Future-Proof:** Add new formats without changing pipeline
✅ **Format-Specific Options:** Each exporter can have custom parameters

## Troubleshooting

### Missing Dependencies

```bash
# For LMDB
pip install lmdb

# For MessagePack
pip install msgpack

# Built-in (no install needed)
# - Pickle
# - JSON
```

### Format Not Found

```
❌ ERROR: Unknown format 'xyz'
   Supported formats: lmdb, pickle, json, msgpack
```

Check available formats in EXPORTERS dict.

### Memory Issues

Use single format at a time for large datasets:

```bash
# Instead of all at once
python generate_db.py lmdb pickle json

# Do one by one
python generate_db.py lmdb
python generate_db.py pickle
python generate_db.py json
```

## Summary

The scalable `generate_db.py` architecture provides:

- ✅ Multiple export formats from single pipeline
- ✅ Easy to add new formats
- ✅ Format-specific optimizations
- ✅ Built-in verification
- ✅ Consistent data structure
- ✅ Production-ready performance
