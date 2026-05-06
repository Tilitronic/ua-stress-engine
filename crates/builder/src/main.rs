//! Builder — reads from the master SQLite DB and produces `ua_stress.bin.bz2`.
//!
//! The SQLite database is the **single source of truth** for all linguistic data.
//! This crate queries it directly (via rusqlite with bundled SQLite) and
//! builds the compact binary dictionary embedded in the Python and WASM crates.
//!
//! # Usage
//! ```
//! cargo run -p builder --release -- path/to/linguistics.db
//! ```
//! If the path argument is omitted, defaults to `data/linguistics.db`.
//!
//! Produces: `data/processed/ua_stress.bin.bz2`

use anyhow::{Context, Result};
use bzip2::write::BzEncoder;
use bzip2::Compression;
use rusqlite::Connection;
use serde_json;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use ua_stress_core::types::{UaStressDbRaw, WordForm};

// ── String-interning helpers ──────────────────────────────────────────────────

/// Interns strings into a compact u8-indexed table (max 256 unique values).
struct Interner {
    table: Vec<String>,
    index: HashMap<String, usize>,
}

impl Interner {
    fn new() -> Self {
        Self { table: Vec::new(), index: HashMap::new() }
    }

    /// Intern a string, returning its table index as u8.
    ///
    /// # Panics
    /// Panics if more than 256 unique strings are interned (table overflow).
    fn intern(&mut self, s: &str) -> u8 {
        if let Some(&i) = self.index.get(s) {
            return i as u8;
        }
        let i = self.table.len();
        assert!(i < 256, "Interner table overflow (>256 unique values) at: {s}");
        self.table.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i as u8
    }
}

/// Interns strings into a u32-indexed table (no size limit).
struct LemmaInterner {
    table: Vec<String>,
    index: HashMap<String, usize>,
}

impl LemmaInterner {
    fn new() -> Self {
        Self { table: Vec::new(), index: HashMap::new() }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i as u32;
        }
        let i = self.table.len();
        self.table.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i as u32
    }
}

// ── SQLite reading ────────────────────────────────────────────────────────────

/// Read all features from the `feature` table, grouped by `word_form_id`.
fn read_features(conn: &Connection) -> Result<HashMap<i64, Vec<(String, String)>>> {
    let mut stmt = conn
        .prepare("SELECT word_form_id, key, value FROM feature ORDER BY word_form_id")
        .context("Failed to prepare feature query")?;

    let mut map: HashMap<i64, Vec<(String, String)>> = HashMap::new();
    let mut rows = stmt.query([]).context("Failed to query features")?;

    while let Some(row) = rows.next().context("Error reading feature row")? {
        let wf_id: i64 = row.get(0)?;
        let key: String = row.get(1)?;
        let value: String = row.get(2)?;
        map.entry(wf_id).or_default().push((key, value));
    }
    Ok(map)
}

/// Read all word forms from `word_form` and build the compact binary DB.
fn build_db_from_sqlite(conn: &Connection) -> Result<UaStressDbRaw> {
    let mut pos_int = Interner::new();
    let mut key_int = Interner::new();
    let mut val_int = Interner::new();
    let mut lemma_int = LemmaInterner::new();
    let mut def_int = LemmaInterner::new();

    // Check if the `definition` column exists in word_form (added in later schema versions)
    let has_definition: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('word_form') WHERE name='definition'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) > 0;

    // Pre-load all features grouped by word_form_id
    eprintln!("[2/4] Reading feature table …");
    let all_features = read_features(conn)?;

    // Read word forms in sorted order (ORDER BY form ensures binary-search correctness)
    eprintln!("[3/4] Reading word_form table …");
    let sql = if has_definition {
        "SELECT id, form, lemma, pos, stress_indices_json, definition \
         FROM word_form ORDER BY form"
    } else {
        "SELECT id, form, lemma, pos, stress_indices_json, NULL \
         FROM word_form ORDER BY form"
    };
    let mut stmt = conn.prepare(sql).context("Failed to prepare word_form query")?;

    // Group by surface form: one form → multiple morphological variants
    let mut form_map: HashMap<String, Vec<WordForm>> = HashMap::new();

    let mut rows = stmt.query([]).context("Failed to query word_form")?;
    while let Some(row) = rows.next().context("Error reading word_form row")? {
        let id: i64 = row.get(0)?;
        let form: String = row.get(1)?;
        let lemma: Option<String> = row.get(2)?;
        let pos_str: Option<String> = row.get(3)?;
        let stress_json: Option<String> = row.get(4)?;
        let definition: Option<String> = row.get(5)?;

        // Parse stress indices (JSON array of integers)
        let stress_variants: Vec<u8> = match &stress_json {
            Some(json) => serde_json::from_str(json)
                .with_context(|| format!("Bad stress_indices_json for form {form:?}: {json}"))?,
            None => vec![],
        };

        // POS — stored as a single string (e.g. "NOUN")
        let pos_intern: Vec<u8> = match &pos_str {
            Some(p) if !p.is_empty() => vec![pos_int.intern(p)],
            _ => vec![],
        };

        // Features for this word_form_id
        let raw_feats = all_features.get(&id).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut feat_map: HashMap<&str, Vec<&str>> = HashMap::new();
        for (k, v) in raw_feats {
            feat_map.entry(k.as_str()).or_default().push(v.as_str());
        }
        let feats: Vec<(u8, Vec<u8>)> = feat_map
            .into_iter()
            .map(|(k, vs)| {
                let ki = key_int.intern(k);
                let vis: Vec<u8> = vs.iter().map(|v| val_int.intern(v)).collect();
                (ki, vis)
            })
            .collect();

        let lemma_idx = lemma.as_deref().map(|l| lemma_int.intern(l));
        let def_idx = definition.as_deref()
            .filter(|s| !s.is_empty())
            .map(|d| def_int.intern(d));

        let wf = WordForm {
            stress_variants,
            pos: pos_intern,
            feats,
            lemma: lemma_idx,
            definition: def_idx,
        };
        form_map.entry(form).or_default().push(wf);
    }

    // Sort entries lexicographically for binary-search lookup
    let mut entries: Vec<(String, Vec<WordForm>)> = form_map.into_iter().collect();
    entries.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

    Ok(UaStressDbRaw {
        pos_table: pos_int.table,
        feat_key_table: key_int.table,
        feat_val_table: val_int.table,
        lemma_pool: lemma_int.table,
        definition_pool: def_int.table,
        entries,
    })
}

// ── Binary serialization ──────────────────────────────────────────────────────

fn write_compressed(db: &UaStressDbRaw, path: &PathBuf) -> Result<()> {
    let bincode_bytes = bincode::serialize(db).context("bincode serialization failed")?;
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create {path:?}"))?;
    let mut encoder = BzEncoder::new(file, Compression::best());
    encoder.write_all(&bincode_bytes).context("bzip2 compression failed")?;
    encoder.finish().context("bzip2 finalization failed")?;

    let original_mb = bincode_bytes.len() as f64 / 1_048_576.0;
    let file_size = std::fs::metadata(path)?.len();
    let compressed_mb = file_size as f64 / 1_048_576.0;
    eprintln!(
        "    bincode: {original_mb:.1} MB → compressed: {compressed_mb:.1} MB  ({:.0}% reduction)",
        (1.0 - compressed_mb / original_mb) * 100.0
    );
    Ok(())
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Accept optional path to the SQLite DB as the first argument.
    let db_path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("data/linguistics.db")
        });

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("data/processed");
    let bin_path = out_dir.join("ua_stress.bin.bz2");

    std::fs::create_dir_all(&out_dir).context("Cannot create data/processed/ directory")?;

    eprintln!("[1/4] Opening SQLite DB: {db_path:?}");
    let conn = Connection::open(&db_path)
        .with_context(|| format!("Cannot open SQLite database at {db_path:?}"))?;

    // Performance pragmas — read-only usage
    conn.execute_batch(
        "PRAGMA journal_mode = OFF;
         PRAGMA synchronous = OFF;
         PRAGMA cache_size = -32768;",
    )?;

    let db = build_db_from_sqlite(&conn)?;

    eprintln!(
        "[4/4] Serialising {} entries → {bin_path:?} …",
        db.entries.len()
    );
    write_compressed(&db, &bin_path)?;

    eprintln!(
        "Done. Rebuild the wasm/python crates to embed the updated binary."
    );
    Ok(())
}
