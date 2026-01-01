# VerseSense Stress Service - Optimized Database Design

## Overview

Your current system has:

- **Current SQL DB**: 681 MB (normalized schema with full linguistic data)
- **Goal**: Design two optimized databases:
  1. **Stress Resolution DB** (LMDB): Fast, compact lookup for known words
  2. **Model Training DB** (SQLite): Curated dataset for stress prediction ML model

---

## 1. Architecture Comparison

### Current Approach (681 MB SQL)

```
merged_stress.sqlite3
├─ word_form (843K rows) - all fields including etymology
├─ lemma_entry (843K rows) - all data
├─ feature (5.2M rows)
├─ translation (1.2M rows)
├─ etymology_text
├─ definition
└─ 9 other normalized tables
```

**Pros**: Normalized, queryable, comprehensive
**Cons**: 681 MB (large), overhead for etymology/translations, slow lookups by form

### Proposed Approach

```
A. Stress Resolution DB (LMDB) - 80-120 MB
   ├─ Keyed by FORM (word variant with morphology)
   └─ Fast O(1) lookups for stress resolution

B. Model Training DB (SQLite) - 150-200 MB
   ├─ Curated dataset with stress patterns
   ├─ Morphological features for learning
   └─ Word frequency/confidence scores
```

---

## 2. Database Design #1: Stress Resolution DB (LMDB)

### Purpose

Fast, memory-efficient lookup for stress resolution. Given a word form + morphology, return stress indices and definition.

### Data Structure (CORRECTED with Variant Types)

**Single Key Per Lemma** with variant type classification for intelligent disambiguation.

Three types of stress variants:

1. **Grammatical Homonyms** (замок): Same pos+features, different meaning+stress
2. **Morphological Variants** (блохи): Different pos/features, different stress
3. **Free Variants** (помилка): Same meaning+pos+features, both stresses allowed

```python
# Type 1: GRAMMATICAL HOMONYM (замок)
db["замок"] = {
    "lemma": "замок",
    "possible_stress_forms": [[0], [1]],      # Two distinct stresses
    "variant_type": "grammatical_homonym",    # Type indicator
    "forms": [
        {
            "stress_indices": [0],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            "definition": "Lock (device)"      # CRITICAL for disambiguation
        },
        {
            "stress_indices": [1],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            "definition": "Castle"             # CRITICAL for disambiguation
        }
    ]
}

# Type 2: MORPHOLOGICAL VARIANT (блохи)
db["блоха"] = {
    "lemma": "блоха",
    "possible_stress_forms": [[0], [1]],      # Two distinct stresses
    "variant_type": "morphological_variant",  # Type indicator
    "forms": [
        {
            "stress_indices": [0],
            "pos": "NOUN",
            "features": {"Case": "Acc", "Gender": "Fem", "Number": "Sing"},
            "definition": "Flea (direct object - WHAT)"
        },
        {
            "stress_indices": [1],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Fem", "Number": "Plur"},
            "definition": "Fleas (subject - WHO)"
        }
    ]
}

# Type 3: FREE VARIANT (помилка)
db["помилка"] = {
    "lemma": "помилка",
    "possible_stress_forms": [[0], [1]],      # Both stresses equally valid
    "variant_type": "free_variant",           # Type indicator
    "forms": [
        {
            "stress_indices": [0],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Fem", "Number": "Sing"},
            "definition": "Pardon/forgiveness" # Same for both
        },
        {
            "stress_indices": [1],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Fem", "Number": "Sing"},
            "definition": "Pardon/forgiveness" # Same for both
        }
    ]
}

# Simple unambiguous word:
db["книга"] = {
    "lemma": "книга",
    "possible_stress_forms": [[0]],           # Single stress - instant lookup
    "variant_type": "single",
    "forms": [
        {
            "stress_indices": [0],
            "pos": "NOUN",
            "features": {"Case": "Nom", "Gender": "Fem", "Number": "Sing"},
            "definition": "Book"
        }
    ]
}
```

**Advantages**:

- ✅ Single key per lemma (843K keys)
- ✅ Variant type enables intelligent disambiguation strategy
- ✅ Fast path: if `len(possible_stress_forms) == 1`, return immediately
- ✅ Slow path: uses variant_type to decide disambiguation method
- ✅ Definition included for grammatical homonyms
- ✅ Morphological features included for morphological variants
- ✅ Free variants clearly marked for training data weighting

---

### Why This Structure Works

1. **`variant_type`**: Indicates disambiguation strategy

   - `"single"`: Only one stress → return immediately
   - `"morphological_variant"`: Different pos/features → use morphology to disambiguate
   - `"grammatical_homonym"`: Same pos/features → use definition/context to disambiguate
   - `"free_variant"`: Multiple stresses equally valid → return any (test robustness)

2. **`possible_stress_forms`**: Quick check without deserializing forms

   - O(1) length check: if 1, fast path
   - If >1, only then examine forms array

3. **`forms`**: Array of (stress, pos, features, definition) combinations

   - Grouped by stress pattern (all [0] forms together, all [1] forms together)
   - Definition disambiguates grammatical homonyms
   - Morphology disambiguates morphological variants

4. **Key = lemma**: 843K keys (clean, efficient)
   - Single lookup per word
   - Natural linguistic organization

### LMDB Build Process

```python
def build_stress_resolution_db(merged_lmdb_path, output_lmdb_path):
    """
    Build optimized stress resolution DB from merged LMDB.

    Source: merged LMDB (843K lemmas)
    Output: stress resolution LMDB (843K lemmas)
    Size: ~100-140 MB (6-7x smaller than SQL)
    """
    env_in = lmdb.open(merged_lmdb_path, readonly=True)
    env_out = lmdb.open(output_lmdb_path, map_size=1024*1024*1024)

    txn_out = env_out.begin(write=True)

    for lemma, entry in stream_merged_lmdb(env_in):
        # Group forms by their stress patterns
        stress_forms = {}
        for form in entry['forms']:
            stress_key = tuple(form['stress_indices'])
            if stress_key not in stress_forms:
                stress_forms[stress_key] = []
            stress_forms[stress_key].append(form)

        # Build entry with possible_stress_forms for fast path
        db_entry = {
            'lemma': lemma,
            'possible_stress_forms': sorted(list(stress_forms.keys())),
            'forms': [stress_forms[key] for key in sorted(stress_forms.keys())]
        }

        key = lemma.encode('utf-8')
        txn_out.put(key, msgpack.packb(db_entry))

    txn_out.commit()
    compact_lmdb(env_out)
    env_out.close()
```

### Size Estimate (CORRECTED)

**Input**: 843K lemmas
**Key space**: 843K keys (one per lemma)
**Data per entry**: ~150-200 bytes (lemma, possible_stress_forms array, forms array with definitions/features)
**Total**: 843K × 0.175 KB ≈ **148 MB before compression**
**After LMDB compaction**: **100-140 MB** ✓

---

## 3. Database Design #2: Model Training DB (SQLite)

### Purpose

Curated dataset for training stress prediction model. Include only data needed for ML (not definitions, etymologies, etc.).

### Recommended Schema

#### Table 1: `training_entries`

```sql
CREATE TABLE training_entries (
    id INTEGER PRIMARY KEY,

    -- Word & Morphology
    form TEXT NOT NULL,           -- Word form without stress
    lemma TEXT NOT NULL,          -- Lemma for reference
    pos TEXT NOT NULL,            -- NOUN, VERB, ADJ, etc.

    -- Morphological features (JSON for flexibility)
    features_json TEXT,           -- {"Case": "Nom", "Number": "Sing", ...}

    -- Stress (Target for ML)
    stress_indices TEXT NOT NULL, -- JSON: [0], [1], [0,1], etc.

    -- Metadata for training
    confidence REAL,              -- 0.0-1.0 (how certain is the stress)
    frequency INTEGER,            -- How many times seen in sources
    sources TEXT,                 -- JSON: ["TRIE", "TXT"]

    -- Training metadata
    split TEXT DEFAULT 'train',   -- 'train', 'val', 'test'
    is_homonym INTEGER DEFAULT 0, -- Has multiple stress patterns

    INDEX idx_form (form),
    INDEX idx_lemma (lemma),
    INDEX idx_pos (pos)
);
```

#### Table 2: `phonetic_features` (Optional, for future use)

```sql
CREATE TABLE phonetic_features (
    id INTEGER PRIMARY KEY,
    form_id INTEGER NOT NULL,

    -- Phonetic properties
    syllable_count INTEGER,           -- Number of syllables
    consonant_clusters TEXT,          -- C, CC, CCC
    vowel_pattern TEXT,               -- Pattern of vowels (VCV, VCCV, etc.)
    morpheme_boundaries TEXT,         -- Morpheme structure

    FOREIGN KEY (form_id) REFERENCES training_entries(id)
);
```

#### Table 3: `definitions` (Optional, for context)

```sql
CREATE TABLE definitions (
    id INTEGER PRIMARY KEY,
    form_id INTEGER NOT NULL,
    definition TEXT,
    FOREIGN KEY (form_id) REFERENCES training_entries(id)
);
```

### Data Selection Strategy

**Include**:

- ✅ All lemmas with unambiguous stress (confidence > 0.8)
- ✅ Common homonyms (замок с stress [0] and [1])
- ✅ Different POS and morphological features
- ✅ Multi-word expressions (they have patterns too)
- ✅ Frequency counts for sampling

**Exclude**:

- ✗ Etymology (not relevant for stress prediction)
- ✗ Translations (not relevant)
- ✗ Examples (too large)
- ✗ Inflection templates (not needed, only features matter)
- ✗ Very rare words (frequency < 2)

### Size Estimate

**Input**: 843K unique lemmas × 2-3 forms avg = 2M form entries
**Data per entry**: ~100 bytes (form, lemma, POS, features JSON, stress, metadata)
**Total**: 2M × 0.1 KB = **200 MB**
**After compression**: **150-200 MB** ✓

---

## 4. Corrected Comparison: Sense-Based vs Alternative Approaches

### For Stress Resolution Service:

| Approach                      | Lookup Time              | Space              | Use Case                           |
| ----------------------------- | ------------------------ | ------------------ | ---------------------------------- |
| **Sense-based (RECOMMENDED)** | O(1) direct sense lookup | 120-160 MB         | ✅ Best - definition disambiguates |
| **Old: Merged stresses**      | O(1) but ambiguous       | 80-120 MB          | ✗ Ambiguity (замок = 2 stresses)   |
| **Lemma-based**               | O(n) scan lemmas         | 200 MB (less data) | ✗ Slow for form lookup             |

**Recommendation**: **Sense-based with composite key** (form + sense_id) because:

1. Definition is the natural disambiguator for homophones
2. Each sense has exactly ONE stress - no ambiguity
3. Training can learn: definition context → correct stress
4. Storage cost minimal (120-160 MB vs 120 MB) but correctness is critical
5. Matches linguistic reality: homophones are different words

---

## 5. Integration with Your Stress Service

### Current Architecture (from diagram.md)

```
Input: analyzed word with spaCy data (morphology, pos, context relations)
   ↓
Is one syllable? → Auto-stress
   ↓
Is in db? → Lookup stress + morphology
   ↓
Is homonym? → Use morphology to decide
   ↓
Unknown word → Stress prediction model
```

### Proposed Enhancement (CORRECTED)

```
Input: word_form + morphology + context from spaCy

Step 1: Lookup lemma in DB
   ├─ Get entry with possible_stress_forms + forms array
   └─ Key: single LMDB lookup (O(1))

Step 2: Check possible_stress_forms for fast path
   ├─ If len(possible_stress_forms) == 1:
   │  └─ Return that stress immediately ✓
   └─ If len(possible_stress_forms) > 1:
      ├─ Score each form in forms array by:
      │  ├─ Morphology match (Case, Number, Gender vs entry)
      │  ├─ Definition similarity to context
      │  └─ Word frequency
      └─ Return top form's stress

Step 3: If not found in DB
   └─ Use stress prediction model
      ├─ Input: form, POS, morphology, syllable structure, context
      └─ Output: stress probability distribution
```

Step 1: Stress Resolution DB Lookup
├─ Key: form_normalized (e.g., "замок")
├─ Get: stress_patterns with morphology
└─ If found:
├─ Filter by morphology (Case, Number, Gender, etc.)
├─ If unique → return stress
└─ If multiple matches:
├─ Score by feature match accuracy
└─ Return top candidate + confidence

Step 2: If not found in DB
└─ Use stress prediction model
├─ Input: form, POS, morphology, syllable structure
└─ Output: stress probability distribution

````

---

## 6. Build Scripts (CORRECTED for Single-Key Design)

### Script 1: Build Stress Resolution DB

```python
# src/data_management/sources/stress_service/build_stress_resolution_db.py

def build_stress_resolution_db(
    merged_lmdb_path: str,
    output_lmdb_path: str,
    batch_size: int = 1000
) -> Tuple[str, dict]:
    """
    Build optimized LMDB for stress resolution.

    Input: Merged LMDB (843K lemmas)
    Output: Stress resolution LMDB (843K lemmas with stress forms)

    Key = lemma (e.g., "замок")
    Value = {
        "lemma": "замок",
        "possible_stress_forms": [[0], [1]],
        "forms": [
            {"stress_indices": [0], "pos": "NOUN", "features": {...}, "definition": "..."},
            {"stress_indices": [1], "pos": "NOUN", "features": {...}, "definition": "..."}
        ]
    }
    """
    env_in = lmdb.open(merged_lmdb_path, readonly=True)
    env_out = lmdb.open(output_lmdb_path, map_size=1024*1024*1024)

    txn_out = env_out.begin(write=True)

    for lemma, entry in stream_merged_lmdb(env_in):
        # Group forms by stress pattern
        stress_to_forms = {}
        for form in entry['forms']:
            stress_key = tuple(sorted(form['stress_indices']))
            if stress_key not in stress_to_forms:
                stress_to_forms[stress_key] = []
            stress_to_forms[stress_key].append({
                'stress_indices': form['stress_indices'],
                'pos': form['pos'],
                'features': form.get('features', {}),
                'definition': form.get('definition', '')
            })

        # Determine variant type
        variant_type = determine_variant_type(entry, stress_to_forms)

        # Create DB entry
        db_entry = {
            'lemma': lemma,
            'variant_type': variant_type,  # NEW: type indicator
            'possible_stress_forms': sorted([list(k) for k in stress_to_forms.keys()]),
            'forms': [stress_to_forms[k] for k in sorted(stress_to_forms.keys())]
        }

        key = lemma.encode('utf-8')
        txn_out.put(key, msgpack.packb(db_entry))

    txn_out.commit()
    compact_lmdb(env_out)
    env_out.close()

    return output_lmdb_path, {'lemmas_processed': 843000}
````

### Service Lookup (With Variant-Type Intelligence)

```python
def lookup_stress(lemma: str, morphology: Dict = None, context: Optional[str] = None) -> Dict:
    """
    Lookup stress with intelligent disambiguation by variant type.

    Args:
        lemma: Word lemma
        morphology: Dict with Case, Number, Gender, etc. from spaCy
        context: Optional context words for definition matching

    Returns:
        Dict with stress (or multiple stresses) and confidence
    """
    entry = db.get(lemma.encode('utf-8'))

    if not entry:
        return {'found': False, 'stress': None}

    entry = msgpack.unpackb(entry)

    # Fast path: single stress
    if len(entry['possible_stress_forms']) == 1:
        return {
            'found': True,
            'stress': entry['possible_stress_forms'][0][0],
            'type': entry['variant_type'],
            'forms': entry['forms']
        }

    # Slow path: multiple stresses - use variant_type to disambiguate
    variant_type = entry['variant_type']

    if variant_type == 'morphological_variant':
        # Disambiguate by morphology (Case, Number, Gender)
        if morphology:
            matching_form = find_form_by_morphology(entry['forms'], morphology)
            if matching_form:
                return {
                    'found': True,
                    'stress': matching_form['stress_indices'][0],
                    'type': 'morphological_variant',
                    'definition': matching_form.get('definition'),
                    'matched_by': 'morphology'
                }
        # No morphology provided - return both
        return {
            'found': True,
            'possible_stresses': entry['possible_stress_forms'],
            'type': 'morphological_variant',
            'forms': entry['forms']
        }

    elif variant_type == 'grammatical_homonym':
        # Disambiguate by definition/context
        if context:
            matching_form = find_form_by_context_similarity(entry['forms'], context)
            if matching_form:
                return {
                    'found': True,
                    'stress': matching_form['stress_indices'][0],
                    'type': 'grammatical_homonym',
                    'definition': matching_form['definition'],
                    'matched_by': 'context'
                }
        # No context - return both possibilities with definitions
        return {
            'found': True,
            'possible_stresses': entry['possible_stress_forms'],
            'type': 'grammatical_homonym',
            'options': [
                {
                    'stress': f['stress_indices'][0],
                    'definition': f.get('definition')
                }
                for f in entry['forms']
            ]
        }

    elif variant_type == 'free_variant':
        # Both stresses are equally valid - return primary, note alternative
        return {
            'found': True,
            'stress': entry['possible_stress_forms'][0][0],
            'type': 'free_variant',
            'alternative_stresses': entry['possible_stress_forms'][1:],
            'note': 'Both stresses are acceptable'
        }

    return {'found': True, 'possible_stresses': entry['possible_stress_forms']}
```

### Script 2: Build Model Training DB

```python
# src/data_management/sources/stress_service/build_training_db.py

def build_training_db(
    stress_resolution_lmdb_path: str,
    output_sqlite_path: str
) -> Tuple[str, dict]:
    """
    Build curated SQLite DB for model training.

    Flatten all forms from stress resolution DB into training rows.
    Each row = one (lemma, stress, definition, features, variant_type) combination.
    """
    conn = sqlite3.connect(output_sqlite_path)
    create_training_schema(conn)

    env = lmdb.open(stress_resolution_lmdb_path, readonly=True)

    with env.begin() as txn:
        for lemma_bytes, entry_bytes in txn.cursor():
            lemma = lemma_bytes.decode('utf-8')
            entry = msgpack.unpackb(entry_bytes)

            variant_type = entry.get('variant_type', 'single')
            is_disambiguable = 1 if variant_type in ['morphological_variant', 'grammatical_homonym'] else 0

            for form_group in entry['forms']:
                # form_group is a list of forms with same stress
                for form in form_group:
                    conn.execute("""
                        INSERT INTO training_entries
                        (lemma, stress_indices, pos, features_json, definition,
                         variant_type, is_disambiguable)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        lemma,
                        json.dumps(form['stress_indices']),
                        form['pos'],
                        json.dumps(form.get('features', {})),
                        form.get('definition', ''),
                        variant_type,
                        is_disambiguable
                    ))

    conn.commit()
    conn.executescript("""
        CREATE INDEX idx_variant_type ON training_entries(variant_type);
        CREATE INDEX idx_lemma ON training_entries(lemma);
    """)
    conn.close()
    env.close()
```

### Updated Training Table Schema (With Variant Classification)

```sql
CREATE TABLE training_entries (
    id INTEGER PRIMARY KEY,

    lemma TEXT NOT NULL,                    -- Word lemma
    stress_indices TEXT NOT NULL,           -- JSON: [0] or [1]
    pos TEXT NOT NULL,                      -- NOUN, VERB, ADJ, etc.
    features_json TEXT,                     -- Full morphology (Case, Gender, Number)
    definition TEXT,                        -- Definition for context (important for grammatical homonyms)

    variant_type TEXT,                      -- 'single' | 'grammatical_homonym' | 'morphological_variant' | 'free_variant'
    is_disambiguable INTEGER,               -- 1 if morphology/definition disambiguates, 0 for free variants

    split TEXT DEFAULT 'train',             -- 'train', 'val', 'test'

    INDEX idx_variant_type (variant_type),
    INDEX idx_lemma (lemma)
);

-- Training data distribution strategy:
-- SELECT COUNT(*) FROM training_entries WHERE variant_type='single';
--     → Most data (85-90%) - straightforward learning
--
-- SELECT COUNT(*) FROM training_entries WHERE variant_type='morphological_variant';
--     → Learn morphology → stress mapping (features important)
--
-- SELECT COUNT(*) FROM training_entries WHERE variant_type='grammatical_homonym';
--     → Learn definition → stress mapping (use definition as feature)
--
-- SELECT COUNT(*) FROM training_entries WHERE variant_type='free_variant';
--     → Learn robustness - both stresses produce same result
--     → Use for testing model flexibility
```

### Training Strategy (Weighted by Variant Type)

```python
# Data loading for training:
# 1. Load single-stress words: weight=1.0 (baseline)
# 2. Load morphological variants: weight=1.5 (learn morphology matters)
# 3. Load grammatical homonyms: weight=2.0 (harder, need context)
# 4. Load free variants: weight=0.5 (test only - sanity check)

def get_weighted_training_batch(batch_size=32):
    """Load training batch with variant-type weighting."""
    queries = {
        'single': f'SELECT * FROM training_entries WHERE variant_type="single" AND split="train" ORDER BY RANDOM() LIMIT {int(batch_size*0.85)}',
        'morphological': f'SELECT * FROM training_entries WHERE variant_type="morphological_variant" AND split="train" ORDER BY RANDOM() LIMIT {int(batch_size*0.10)}',
        'grammatical': f'SELECT * FROM training_entries WHERE variant_type="grammatical_homonym" AND split="train" ORDER BY RANDOM() LIMIT {int(batch_size*0.05)}',
    }

    batch = []
    for variant_type, query in queries.items():
        rows = conn.execute(query).fetchall()
        batch.extend(rows)

    random.shuffle(batch)
    return batch[:batch_size]
```

    INDEX idx_lemma (lemma),
    INDEX idx_stress (stress_indices)

);

```
                    lemma,
                    form['pos'],
                    json.dumps(form.get('feats', {})),
                    json.dumps(form['stress_indices']),
                    confidence,
                    entry.get('frequency', 1)
                ))

    conn.commit()
    # Add indexes
    conn.executescript("CREATE INDEX idx_form ON training_entries(form);")
    conn.close()
```

---

## 7. Implementation Roadmap

### Phase 1: Design & Validation ✓ (CORRECTED)

- [x] Review current 681 MB SQL DB
- [x] Design single-key-per-lemma LMDB with possible_stress_forms fast path
- [x] Design training DB schema (flattened forms)
- [x] Remove redundant fields (confidence, sources, sense_ids)
- [x] Size estimates (100-140 MB for stress DB)

### Phase 2: Implementation

- [ ] Build stress resolution LMDB from merged LMDB with stress grouping
- [ ] Build training DB from stress resolution LMDB
- [ ] Add unit tests for both
- [ ] Performance benchmarks

### Phase 3: Integration

- [ ] Create stress resolution service using single-key LMDB
- [ ] Implement fast path (single stress) + slow path (context-based disambiguation)
- [ ] Test with real words (замок/lock vs замок/castle, блоха)
- [ ] Measure lookup latency (target: <2ms for fast path, <5ms for slow path)

### Phase 4: ML Model Training

- [ ] Use training DB to train stress prediction model with definition context
- [ ] Evaluate on test split (including homonym cases)
- [ ] Integrate into service for unknown words

---

## 8. Performance Targets

| Metric                 | Target  | Single-Key LMDB |
| ---------------------- | ------- | --------------- |
| Size                   | <150 MB | ✅ 100-140 MB   |
| Fast path (1 stress)   | <2ms    | ✅ O(1)         |
| Slow path (N stresses) | <5ms    | ✅ O(forms)     |
| Build time             | <3 min  | ✅ ~90s         |
| Memory footprint       | <500 MB | ✅ ~200 MB      |
| Key space              | ~1M     | ✅ 843K (clean) |

---

## 9. Recommendation Summary (FINAL - With Variant Classification)

### For Stress Resolution Service (Real-time)

- **Use**: Single-key LMDB with variant_type classification
- **Key Format**: lemma (e.g., "замок", "блоха", "помилка")
- **Size**: ~100-140 MB
- **Variant Types**:
  - `"single"`: One stress only → return immediately
  - `"morphological_variant"`: Different pos/features → disambiguate by morphology
  - `"grammatical_homonym"`: Same pos/features → disambiguate by definition/context
  - `"free_variant"`: Multiple stresses equally valid → return any
- **Lookup Strategy**:
  - Fast path: O(1) for single-stress words
  - Morphological path: Match by Case/Number/Gender
  - Context path: Match by definition similarity
  - Free variant path: Return primary + note alternative

### For Model Training (SQLite)

- **Use**: SQLite with flattened forms and variant classification
- **Size**: ~180-200 MB
- **Schema**: lemma, stress, pos, features, definition, variant_type, is_disambiguable
- **Training Strategy**: Weighted sampling
  - 85% single-stress words (baseline)
  - 10% morphological variants (learn morphology)
  - 5% grammatical homonyms (hardest - learn context/definitions)
  - 0% free variants in training, 100% in validation (sanity check)
- **Includes**: All variant types (complete representation of language)
- **Excludes**: Etymology, translations, sources, confidence (not needed)

### For Reference (Queryable Data)

- **Keep**: Current 681 MB SQL for analytics/exploration
- **Use**: Single-key LMDB for production stress service

---

## 10. Key Insight: Three Types of Ambiguity Handled Intelligently

```
Type 1: GRAMMATICAL HOMONYMS (замок)
  Same pos+features, different meaning
  → Disambiguate by definition/context
  → Service: lookup definition in context
  → Training: learn definition → stress mapping

Type 2: MORPHOLOGICAL VARIANTS (блохи)
  Different pos/features (Case, Number, Gender)
  → Disambiguate by morphology
  → Service: lookup morphology from spaCy
  → Training: learn morphology → stress mapping

Type 3: FREE VARIANTS (помилка)
  Same meaning+pos+features, multiple stresses OK
  → Return primary, note alternative
  → Service: return either stress confidently
  → Training: test model robustness (both should work)

db["замок"] = {
    "variant_type": "grammatical_homonym",
    "possible_stress_forms": [[0], [1]],
    "forms": [
        {"stress": [0], "definition": "Lock"},      ← disambiguator
        {"stress": [1], "definition": "Castle"}     ← disambiguator
    ]
}

db["блоха"] = {
    "variant_type": "morphological_variant",
    "possible_stress_forms": [[0], [1]],
    "forms": [
        {"stress": [0], "features": {"Case": "Acc"}},  ← disambiguator
        {"stress": [1], "features": {"Case": "Nom", "Number": "Plur"}}  ← disambiguator
    ]
}

db["помилка"] = {
    "variant_type": "free_variant",
    "possible_stress_forms": [[0], [1]],
    "forms": [
        {"stress": [0], "definition": "Pardon"},   ← same
        {"stress": [1], "definition": "Pardon"}    ← same (both OK)
    ]
}
```

---

## Files to Create

```
src/data_management/sources/stress_service/
├─ __init__.py
├─ README.md                               # Documentation
├─ build_stress_resolution_db.py           # Single-key LMDB builder
├─ build_training_db.py                    # Training DB builder (flattened)
├─ stress_resolution_service.py            # Service with fast/slow paths
├─ stress_resolution_service_test.py       # Test cases (замок, блоха, etc.)
└─ training_db_schema.sql                  # SQLite schema
```

---

## Next Steps

1. ✅ Review and approve single-key design
2. Implement Phase 2 (build scripts)
3. Test size estimates with real data
4. Benchmark lookup performance (fast path vs slow path)
5. Proceed to Phase 3 (service integration)
