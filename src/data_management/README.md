# Data Management — Layer 1

The data pipeline that builds the master SQLite database from four raw Ukrainian stress sources. Everything downstream (LMDB lookup service, binary trie, ML training) is built from this master DB.

## What it contains

```
src/data_management/
  sources/            — four raw source sub-packages (parsers + data)
    kaikki/           — Wiktionary JSONL extract
    trie_ua_stresses/ — lang-uk marisa-trie file
    txt_ua_stresses/  — lang-uk plain-text dictionary
    ua_variative_stressed_words/ — curated free-variant stress list
  transform/          — shared types, merge logic, schema
    data_unifier.py   — Pydantic models: LinguisticEntry, WordForm, UPOS, UDFeatKey
    merger.py         — LMDB-backed merge + SQLite export
    parsing_merging_service.py — orchestrator (runs all parsers → master DB)
    cache_utils.py    — msgpack streaming cache helpers
    linguistic_data_schema.sql — SQLite schema definition
  export/             — downstream export targets
    export_training_db.py — export to training SQLite
    web_stress_db/    — binary .ctrie builder + npm package
```

## Version & integrity

| Field         | Value                                           |
|---------------|-------------------------------------------------|
| Version       | 1.0.0                                           |
| Master DB rows| ~4,078,014 (±0.1%)                              |
| Sources       | kaikki, trie_ua_stresses, txt_ua_stresses, ua_variative_stressed_words |
| Schema        | See `transform/linguistic_data_schema.sql`      |

## How it works

```
kaikki_parser     ─┐
trie_stress_parser─┤─► merger.py (LMDB-backed merge) ─► SQLite master DB
txt_stress_parser ─┤
variative_parser  ─┘  (annotation only)
```

1. Each parser streams `(lemma, LinguisticEntry)` pairs to an intermediate LMDB cache.
2. `merger.py` merges all caches losslessly, deduplicating by normalised form.
3. The merged result is exported to the master SQLite using `linguistic_data_schema.sql`.

### SQLite Schema (key tables)

```sql
-- Main table: one row per word form
CREATE TABLE word_form (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    form                 TEXT NOT NULL,
    lemma                TEXT,
    pos                  TEXT,
    stress_indices_json  TEXT NOT NULL   -- e.g. '[0]' or '[0,1]'
);

-- Normalised morphological features (UD)
CREATE TABLE feature (
    word_form_id  INTEGER NOT NULL,
    key           TEXT NOT NULL,   -- e.g. 'Case'
    value         TEXT NOT NULL    -- e.g. 'Nom'
);
```

Full schema: [transform/linguistic_data_schema.sql](transform/linguistic_data_schema.sql)

### Source provenance

| Source                        | ~Records    | License       |
|-------------------------------|-------------|---------------|
| kaikki (Wiktionary extract)   | ~2 M forms  | CC BY-SA 4.0  |
| trie_ua_stresses (lang-uk)    | ~2.9 M forms| MIT           |
| txt_ua_stresses (lang-uk)     | ~2.9 M forms| MIT / ULIF    |
| ua_variative_stressed_words   | ~150 lemmas | Manual curation |

## How to build / rebuild

```bash
conda activate verseSense-py312
python -m src.data_management.transform.parsing_merging_service
```

This will:
1. Run all four source parsers (takes ~30–60 min depending on hardware)
2. Merge intermediate LMDB caches
3. Export the master SQLite DB (681 MB)
4. Print a row-count summary

The output path is controlled by `UA_STRESS_MASTER_DB` env var.

## How to use

```python
import sqlite3, json, os

db_path = os.environ.get("UA_STRESS_MASTER_DB", "master.sqlite3")
con = sqlite3.connect(db_path)

# Look up a word
rows = con.execute(
    "SELECT form, stress_indices_json FROM word_form WHERE form = ?",
    ("замок",)
).fetchall()
for form, idx_json in rows:
    print(form, json.loads(idx_json))  # e.g. замок [0]
```

## Tests

```bash
conda activate verseSense-py312

# Source parsers
pytest tests/src/data_management/sources/ -v

# Web trie (needs built .ctrie)
pytest tests/src/data_management/test_web_stress_db.py -v

# Master DB (needs master DB available)
pytest tests/src/data_management/test_master_db.py -v
```

## Dependencies

- `lmdb`, `msgpack`, `tqdm`, `pydantic` — core pipeline
- `marisa-trie` — trie source parser
- `src/lemmatizer/` — lemmatisation during parsing
- `src/utils/normalize_apostrophe.py` — apostrophe normalisation
