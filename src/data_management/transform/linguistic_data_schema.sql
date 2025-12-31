-- Linguistic Data SQLite Schema (optimized, normalized)
-- For use with Python's sqlite3 or any SQL tool

PRAGMA foreign_keys = ON;

-- Main table: word forms (atomic unit)
CREATE TABLE word_form (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    form TEXT NOT NULL,
    lemma TEXT,
    pos TEXT,
    main_definition_id INTEGER REFERENCES definition(id),
    roman TEXT,
    ipa TEXT,
    etymology_id INTEGER REFERENCES etymology_text(id),
    etymology_number INTEGER,
    sense_id TEXT,
    stress_indices_json TEXT NOT NULL -- JSON array of stress indices for this word form
);

-- Lemma-level possible stress indices
CREATE TABLE lemma_entry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma TEXT NOT NULL UNIQUE,
    possible_stress_indices_json TEXT NOT NULL -- JSON array of all possible stress indices for this lemma
);
-- Deduplicated definition and etymology tables
CREATE TABLE IF NOT EXISTS definition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS etymology_text (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT UNIQUE NOT NULL
);
CREATE INDEX idx_word_form_form ON word_form(form);
CREATE INDEX idx_word_form_lemma ON word_form(lemma);
CREATE INDEX idx_word_form_pos ON word_form(pos);

-- Morphological features (UD-compliant)
CREATE TABLE feature (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_feature_word_form_id ON feature(word_form_id);
CREATE INDEX idx_feature_key_value ON feature(key, value);

-- Translations
CREATE TABLE translation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    lang TEXT NOT NULL,
    text TEXT NOT NULL,
    sense TEXT,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_translation_word_form_id ON translation(word_form_id);
CREATE INDEX idx_translation_lang ON translation(lang);

-- Etymology templates
CREATE TABLE etymology_template (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_etymology_template_word_form_id ON etymology_template(word_form_id);

-- Inflection templates
CREATE TABLE inflection_template (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_inflection_template_word_form_id ON inflection_template(word_form_id);

-- Categories
CREATE TABLE category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_category_word_form_id ON category(word_form_id);

-- Tags
CREATE TABLE tag (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_tag_word_form_id ON tag(word_form_id);

-- Examples
CREATE TABLE example (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    example TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_example_word_form_id ON example(word_form_id);

-- Possible stress indices (as JSON array)
CREATE TABLE possible_stress_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    stress_indices_json TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_possible_stress_index_word_form_id ON possible_stress_index(word_form_id);

-- Meta (as JSON object)
CREATE TABLE meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_form_id INTEGER NOT NULL,
    meta_json TEXT NOT NULL,
    FOREIGN KEY(word_form_id) REFERENCES word_form(id) ON DELETE CASCADE
);
CREATE INDEX idx_meta_word_form_id ON meta(word_form_id);

-- For memory efficiency:
--  - Use TEXT for JSON fields (compact, flexible, can be indexed with SQLite JSON1 extension if needed)
--  - Use indexes on all foreign keys and frequent query fields
--  - Use ON DELETE CASCADE for easy cleanup
--  - No duplicate data, all many-to-one/many-to-many relations are normalized
--  - No unnecessary tables for rarely used fields
