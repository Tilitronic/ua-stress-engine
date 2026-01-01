-- Stress Prediction Training Database Schema
-- 
-- This database is optimized for stress prediction model training.
-- It contains:
-- - Word forms with stress indices
-- - Morphological features (Case, Gender, Number, etc.)
-- - Variant type classification for intelligent training
-- - Metadata and statistics tracking
--
-- Source: Derived from merged linguistic database
-- Purpose: Train stress prediction ML model

-- Core training table
-- Each row represents a (form, lemma, stress, pos, features) combination
-- Used for training stress prediction models
CREATE TABLE IF NOT EXISTS training_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Word data
    lemma TEXT NOT NULL,                    -- Base lemma (e.g., "замок")
    form TEXT NOT NULL,                     -- Specific word form (surface form for training)
    stress_indices TEXT NOT NULL,           -- JSON: [0] or [1] or [0,1]
    
    -- Linguistic features
    pos TEXT NOT NULL,                      -- Part of speech (NOUN, VERB, ADJ, etc.)
    features_json TEXT,                     -- JSON: morphological features
                                            -- {"Case": "Nom", "Gender": "Masc", "Number": "Sing", ...}
    
    -- Classification
    variant_type TEXT NOT NULL,             -- Indicates disambiguation strategy
                                            -- 'single': only one stress
                                            -- 'grammatical_homonym': same pos+features, different stress (by sense)
                                            -- 'morphological_variant': different pos/features, different stress
                                            -- 'free_variant': same everything, multiple stresses allowed
    is_disambiguable INTEGER DEFAULT 0,     -- 1 if morphology disambiguates (grammatical/morphological)
    is_homonym INTEGER DEFAULT 0,           -- 1 if multiple stress patterns exist for this lemma
    
    -- Training metadata
    split TEXT DEFAULT 'train',             -- 'train', 'val', 'test'
                                            -- default is 'train'
                                            -- validation: stratified by variant_type
                                            -- test: includes all variant types for robustness
    
    -- Provenance
    
    -- Uniqueness constraint: don't duplicate identical entries
    UNIQUE(lemma, form, stress_indices, pos, features_json)
);

-- Indexes for fast querying during training
CREATE INDEX IF NOT EXISTS idx_variant_type ON training_entries(variant_type);
CREATE INDEX IF NOT EXISTS idx_lemma ON training_entries(lemma);
CREATE INDEX IF NOT EXISTS idx_form ON training_entries(form);
CREATE INDEX IF NOT EXISTS idx_pos ON training_entries(pos);
CREATE INDEX IF NOT EXISTS idx_split ON training_entries(split);
CREATE INDEX IF NOT EXISTS idx_is_disambiguable ON training_entries(is_disambiguable);
CREATE INDEX IF NOT EXISTS idx_is_homonym ON training_entries(is_homonym);

-- Metadata table: tracks source and reproducibility
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Statistics table: summary counts
CREATE TABLE IF NOT EXISTS db_stats (
    metric TEXT PRIMARY KEY,
    value TEXT
);

-- Sample queries for model training:
--
-- 1. Load training data with variant-type weighting:
--   SELECT * FROM training_entries 
--   WHERE split='train' AND variant_type='single'
--   LIMIT 27000  -- 85% of 32K
--   UNION ALL
--   SELECT * FROM training_entries 
--   WHERE split='train' AND variant_type='morphological_variant'
--   LIMIT 3200   -- 10% of 32K
--   UNION ALL
--   SELECT * FROM training_entries 
--   WHERE split='train' AND variant_type='grammatical_homonym'
--   LIMIT 1600   -- 5% of 32K
--
-- 2. Validation data for each variant type:
--   SELECT * FROM training_entries WHERE split='val' AND variant_type='single'
--   SELECT * FROM training_entries WHERE split='val' AND variant_type='morphological_variant'
--   SELECT * FROM training_entries WHERE split='val' AND variant_type='grammatical_homonym'
--
-- 3. Test model robustness with free variants:
--   SELECT * FROM training_entries WHERE split='test' AND variant_type='free_variant'
--
-- 4. Distribution analysis:
--   SELECT variant_type, COUNT(*) as count, COUNT(DISTINCT lemma) as unique_lemmas, COUNT(DISTINCT form) as unique_forms
--   FROM training_entries
--   WHERE split='train'
--   GROUP BY variant_type
--
-- 5. Morphology learning:
--   SELECT features_json, stress_indices, COUNT(*) as freq
--   FROM training_entries
--   WHERE variant_type='morphological_variant'
--   GROUP BY features_json, stress_indices
--   ORDER BY freq DESC
--
-- 6. Homonym analysis:
--   SELECT is_homonym, COUNT(*) FROM training_entries GROUP BY is_homonym
