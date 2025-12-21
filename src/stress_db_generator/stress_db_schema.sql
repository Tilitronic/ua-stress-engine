-- SQLite Database Schema for Ukrainian Word Stress and Morphology
-- Google-inspired design with proper normalization and indexing

CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL UNIQUE,
    lemma TEXT,
    primary_pos TEXT CHECK(primary_pos IN ('ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 
                                           'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 
                                           'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X')),
    frequency INTEGER DEFAULT 1,
    is_lemma BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS word_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    form_text TEXT NOT NULL,
    form_type TEXT CHECK(form_type IN ('base', 'inflected', 'compound')),
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE,
    UNIQUE(word_id, form_text)
);

CREATE TABLE IF NOT EXISTS morphological_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL,
    feature_value TEXT NOT NULL,
    category TEXT,  -- 'nominal', 'verbal', 'pos', etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(feature_name, feature_value)
);

CREATE TABLE IF NOT EXISTS word_form_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (word_form_id) REFERENCES word_forms(id) ON DELETE CASCADE,
    FOREIGN KEY (feature_id) REFERENCES morphological_features(id) ON DELETE CASCADE,
    UNIQUE(word_form_id, feature_id)
);

CREATE TABLE IF NOT EXISTS stress_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    word_form_id INTEGER NOT NULL,
    stress_positions TEXT NOT NULL,  -- JSON array: "[0]", "[2,6]", "[8,10]"
    is_primary BOOLEAN DEFAULT 0,
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE,
    FOREIGN KEY (word_form_id) REFERENCES word_forms(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS morphology_ud (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL UNIQUE,
    ud_string TEXT,  -- Universal Dependencies format: "Case=Nom|Gender=Masc|Number=Sing"
    pos TEXT NOT NULL,  -- NOUN, VERB, ADJ, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (word_form_id) REFERENCES word_forms(id) ON DELETE CASCADE
);

-- INDEXES FOR FAST QUERIES
CREATE INDEX idx_words_text ON words(text);
CREATE INDEX idx_words_lemma ON words(lemma);
CREATE INDEX idx_words_pos ON words(primary_pos);
CREATE INDEX idx_words_frequency ON words(frequency DESC);

CREATE INDEX idx_word_forms_word_id ON word_forms(word_id);
CREATE INDEX idx_word_forms_text ON word_forms(form_text);
CREATE INDEX idx_word_forms_frequency ON word_forms(frequency DESC);

CREATE INDEX idx_morphological_features_name ON morphological_features(feature_name);
CREATE INDEX idx_morphological_features_value ON morphological_features(feature_value);

CREATE INDEX idx_word_form_features_word_form ON word_form_features(word_form_id);
CREATE INDEX idx_word_form_features_feature ON word_form_features(feature_id);

CREATE INDEX idx_stress_patterns_word ON stress_patterns(word_id);
CREATE INDEX idx_stress_patterns_form ON stress_patterns(word_form_id);
CREATE INDEX idx_stress_patterns_primary ON stress_patterns(is_primary);
CREATE INDEX idx_stress_patterns_frequency ON stress_patterns(frequency DESC);

CREATE INDEX idx_morphology_ud_form ON morphology_ud(word_form_id);
CREATE INDEX idx_morphology_ud_pos ON morphology_ud(pos);

-- VIEWS FOR COMMON QUERIES
CREATE VIEW word_stress_summary AS
SELECT 
    w.id,
    w.text,
    w.primary_pos,
    GROUP_CONCAT(DISTINCT sp.stress_positions) as all_stress_positions,
    MAX(sp.frequency) as max_frequency,
    COUNT(DISTINCT sp.id) as variant_count
FROM words w
LEFT JOIN stress_patterns sp ON w.id = sp.word_id
GROUP BY w.id, w.text, w.primary_pos;

CREATE VIEW word_form_morphology AS
SELECT 
    wf.id as form_id,
    wf.word_id,
    wf.form_text,
    m.pos,
    m.ud_string,
    GROUP_CONCAT(mf.feature_name || '=' || mf.feature_value, '|') as features,
    sp.stress_positions
FROM word_forms wf
LEFT JOIN morphology_ud m ON wf.id = m.word_form_id
LEFT JOIN word_form_features wff ON wf.id = wff.word_form_id
LEFT JOIN morphological_features mf ON wff.feature_id = mf.id
LEFT JOIN stress_patterns sp ON wf.id = sp.word_form_id AND sp.is_primary = 1
GROUP BY wf.id;
