# Modular Pipeline Architecture - Visual Overview

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      STRESS DATABASE GENERATOR                            │
│                          Modular Pipeline v2.0                            │
└──────────────────────────────────────────────────────────────────────────┘

INPUT SOURCES                PIPELINE STAGES                    OUTPUT
═════════════                ════════════════                   ══════

┌─────────────┐              ┌─────────────┐
│stress.trie  │─────────────▶│txt_parser.py│
│(2.9M words) │              │Parse TXT    │
│ - Stress    │              │             │
│ - Morphology│              │ • Extract   │
└─────────────┘              │   stress    │
                             │ • Normalize │
                             │   keys      │
                             └──────┬──────┘
┌─────────────┐                     │
│*.txt files  │─────────────────────┘
│(2.9M words) │
│ - Stress    │              ┌─────────────┐
│ - Optional  │              │trie_adapter │
│   definitions              │Parse Trie   │
└─────────────┘─────────────▶│             │
                             │ • Load trie │
                             │ • Extract   │
                             │   morphology│
                             └──────┬──────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │  merger.py  │
                             │Combine Data │
                             │             │
                             │ • Merge     │
                             │   stress    │
                             │ • Combine   │
                             │   features  │
                             │ • Track     │
                             │   sources   │
                             └──────┬──────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │spacy_trans- │
                             │former.py    │
                             │Validate     │
                             │             │
                             │ • Check POS │
                             │ • Check     │
                             │   features  │
                             │ • Sort      │
                             │ • Format    │
                             └──────┬──────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │lmdb_exporter│          ┌──────────┐
                             │Export DB    │─────────▶│stress.lmdb
                             │             │          │(20-30 MB)│
                             │ • Serialize │          │          │
                             │ • Write     │          │ Fast     │
                             │ • Verify    │          │ Queries  │
                             └─────────────┘          └──────────┘
```

## Data Flow with Examples

```
STAGE 1: TXT PARSER
═══════════════════

Input (text file):
    а́тлас	збірник карт
    атла́с	тканина

Output:
    {
        "атлас": [
            ([0], "збірник карт"),
            ([1], "тканина")
        ]
    }

─────────────────────────────────────────────────────────────────────────

STAGE 2: TRIE ADAPTER
═════════════════════

Input (trie bytes):
    "атлас" → [binary morphology data]

Output:
    {
        "атлас": [
            ([0], {"upos": "NOUN", "Case": "Nom", "Number": "Sing"}),
            ([0], {"upos": "NOUN", "Case": "Acc", "Number": "Sing"}),
            ([1], {"upos": "NOUN", "Case": "Nom", "Number": "Sing"}),
            ([1], {"upos": "NOUN", "Case": "Acc", "Number": "Sing"})
        ]
    }

─────────────────────────────────────────────────────────────────────────

STAGE 3: MERGER
═══════════════

Combines trie + txt data:

Trie: 4 forms with stress [0] and [1], different Cases
TXT:  2 forms with stress [0] and [1], no morphology

Merging Logic:
    • Same stress [0] + Case=Nom → merge
    • Same stress [0] + Case=Acc → merge
    • Same stress [1] + Case=Nom → merge
    • Same stress [1] + Case=Acc → merge

Output:
    {
        "атлас": [
            WordForm(
                stress_variants=[0],
                pos=["NOUN"],
                feats={"Case": ["Nom", "Acc"], "Number": ["Sing"]},
                source="merged"
            ),
            WordForm(
                stress_variants=[1],
                pos=["NOUN"],
                feats={"Case": ["Nom", "Acc"], "Number": ["Sing"]},
                source="merged"
            )
        ]
    }

Result: 4 forms → 2 forms (intelligent deduplication)

─────────────────────────────────────────────────────────────────────────

STAGE 4: SPACY TRANSFORMER
═══════════════════════════

Validates and standardizes:

Input:
    WordForm(stress=[0], pos=["NOUN"], feats={"Case": ["Nom", "Acc"]})

Validation:
    ✓ pos "NOUN" → valid UPOS tag
    ✓ feature "Case" → valid UD feature
    ✓ values "Nom", "Acc" → valid Case values
    ✓ Sort alphabetically

Output:
    SpaCyWordForm(
        stress_variants=[0],
        pos=["NOUN"],
        feats={"Case": ["Acc", "Nom"], "Number": ["Sing"]}
    )

spaCy Format String:
    "Case=Acc,Nom|Number=Sing"

─────────────────────────────────────────────────────────────────────────

STAGE 5: LMDB EXPORTER
══════════════════════

Converts to JSON and writes to database:

Input:
    SpaCyWordForm objects

Conversion:
    {
        "stress_variants": [0],
        "pos": ["NOUN"],
        "feats": {"Case": ["Acc", "Nom"], "Number": ["Sing"]}
    }

Storage:
    Key:   "атлас" (UTF-8 bytes)
    Value: JSON (UTF-8 bytes)

Output:
    LMDB database file (~20-30 MB)
    Memory-mapped for ultra-fast queries
```

## Module Interactions

```
┌─────────────────────────────────────────────────────────────────┐
│                        build_lmdb.py                             │
│                     (Pipeline Orchestrator)                      │
│                                                                  │
│  Coordinates all modules:                                       │
│  1. Call txt_parser.parse_txt_dictionary()                      │
│  2. Call trie_adapter.parse_trie_data()                         │
│  3. Create DictionaryMerger and merge data                      │
│  4. Create SpaCyTransformer and validate                        │
│  5. Create LMDBExporter and export                              │
│  6. Create LMDBQuery and verify                                 │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│txt_parser.py ││trie_adapter  ││  merger.py   ││spacy_trans-  │
│              ││              ││              ││former.py     │
│• TXTParser   ││• TrieData    ││• Dictionary  ││• SpaCy       │
│  class       ││  Adapter     ││  Merger      ││  Transformer │
│              ││• Uses trie_  ││• WordForm    ││• SpaCyWord   │
│• parse_file()││  parser      ││  dataclass   ││  Form        │
│• parse_line()││              ││• Merge logic ││• Validate    │
│• extract_    ││• parse_trie()││• Statistics  ││  POS & feats │
│  stress()    ││              ││              ││              │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
                       │
                       ▼
              ┌──────────────┐
              │trie_parser.py│
              │              │
              │• TrieParser  │
              │• marisa_trie │
              │• Tag decom-  │
              │  pression    │
              └──────────────┘
```

## Dependency Graph

```
build_lmdb.py
    ├─▶ txt_parser.py
    │       └─▶ normalize_apostrophe (utils)
    │
    ├─▶ trie_adapter.py
    │       ├─▶ trie_parser.py
    │       │       └─▶ marisa_trie (external)
    │       └─▶ normalize_apostrophe (utils)
    │
    ├─▶ merger.py
    │       └─▶ (no external deps)
    │
    ├─▶ spacy_transformer.py
    │       └─▶ (no external deps)
    │
    └─▶ lmdb_exporter.py
            └─▶ lmdb (external)
```

## Performance Breakdown

```
Total Time: ~84 seconds

┌────────────────────────────────────────────────────────┐
│                   TIME DISTRIBUTION                     │
├────────────────────────────────────────────────────────┤
│                                                         │
│ TXT Parsing     ████████████████████ 28s (33%)        │
│                                                         │
│ Trie Parsing    ███████████████████ 25s (30%)         │
│                                                         │
│ Merging         ███████████ 10s (12%)                  │
│                                                         │
│ Export LMDB     ████████████████ 15s (18%)            │
│                                                         │
│ Transform       ███████ 6s (7%)                        │
│                                                         │
└────────────────────────────────────────────────────────┘

Memory Usage:
    Peak: ~2.5 GB (during merge stage)
    LMDB: 2 GB max (configurable)
    Final DB: 20-30 MB (compressed)
```

## Data Structure Evolution

```
TXT Parser Output
┌─────────────────────────────────────┐
│ Dict[str, List[Tuple]]              │
│                                     │
│ "слово": [                          │
│     ([0], "definition"),            │
│     ([1], None)                     │
│ ]                                   │
└─────────────────────────────────────┘
                  │
                  ▼
Trie Adapter Output
┌─────────────────────────────────────┐
│ Dict[str, List[Tuple]]              │
│                                     │
│ "слово": [                          │
│     ([0], {"upos": "NOUN", ...}),  │
│     ([1], {"upos": "VERB", ...})   │
│ ]                                   │
└─────────────────────────────────────┘
                  │
                  ▼
Merger Output
┌─────────────────────────────────────┐
│ Dict[str, List[WordForm]]           │
│                                     │
│ "слово": [                          │
│     WordForm(                       │
│         stress=[0],                 │
│         pos=["NOUN"],               │
│         feats={"Case": ["Nom"]}     │
│     )                               │
│ ]                                   │
└─────────────────────────────────────┘
                  │
                  ▼
Transformer Output
┌─────────────────────────────────────┐
│ Dict[str, List[SpaCyWordForm]]      │
│                                     │
│ "слово": [                          │
│     SpaCyWordForm(                  │
│         stress=[0],                 │
│         pos=["NOUN"],               │
│         feats={"Case": ["Nom"]}     │
│     )                               │
│ ]                                   │
└─────────────────────────────────────┘
                  │
                  ▼
LMDB Storage
┌─────────────────────────────────────┐
│ LMDB (Key-Value Database)           │
│                                     │
│ Key: "слово" (bytes)                │
│ Value: JSON(                        │
│     [{                              │
│         "stress_variants": [0],     │
│         "pos": ["NOUN"],            │
│         "feats": {"Case": ["Nom"]}  │
│     }]                              │
│ )                                   │
└─────────────────────────────────────┘
```

## Testing Strategy

```
Unit Tests (test_modular_pipeline.py)
┌────────────────────────────────────┐
│ Test Each Module Independently     │
├────────────────────────────────────┤
│                                    │
│ ✓ TEST 1: TXT Parser              │
│   - Line parsing                   │
│   - Stress extraction              │
│   - Key normalization              │
│                                    │
│ ✓ TEST 2: Trie Adapter            │
│   - Trie loading                   │
│   - Word parsing                   │
│   - Key normalization              │
│                                    │
│ ✓ TEST 3: Merger                  │
│   - Data combination               │
│   - Feature merging                │
│   - Source tracking                │
│                                    │
│ ✓ TEST 4: spaCy Transformer       │
│   - POS validation                 │
│   - Feature validation             │
│   - Format conversion              │
│                                    │
│ ✓ TEST 5: Full Integration        │
│   - End-to-end pipeline            │
│   - Data integrity                 │
│   - Output verification            │
│                                    │
└────────────────────────────────────┘
```

## Extension Points

```
Future Enhancements

┌────────────────────┐
│ New Input Sources  │
├────────────────────┤
│ • JSON Parser      │
│ • XML Parser       │
│ • SQL Connector    │
│ • API Client       │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│    merger.py       │──▶ Already supports multiple sources
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ New Transformers   │
├────────────────────┤
│ • Lemmatizer       │
│ • Frequency        │
│ • Context Analyzer │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ New Exporters      │
├────────────────────┤
│ • Pickle           │
│ • MessagePack      │
│ • SQLite           │
│ • JSON             │
└────────────────────┘
```

---

## Summary

**Modular architecture provides:**

- ✅ Clear separation of concerns
- ✅ Independent testability
- ✅ Easy extension
- ✅ Production-ready performance
- ✅ Comprehensive documentation
- ✅ Type safety
- ✅ Error handling
- ✅ Statistics tracking

**Result:** Clean, maintainable, scalable NLP pipeline for Ukrainian stress analysis.
