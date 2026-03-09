"""
Export stress prediction training database from merged SQL database.

This script creates an optimized SQLite database for stress prediction model training,
derived from the merged linguistic database. It includes:
- All word forms with their stress patterns
- Morphological features (Case, Gender, Number, etc.) - enriched with spaCy (default) and Stanza fallback
- Variant type classification (single, grammatical_homonym, morphological_variant, free_variant)
- Confidence scores for POS and morphological features
- Training/validation/test splits

The database is smaller and more focused than the full merged SQL DB.

Usage:
    python -m src.data_management.export.export_training_db
        --merged-db <path_to_merged_sql_db>
        --output-db <path_to_output_db>
        --overwrite
"""

import sqlite3
import json
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from tqdm import tqdm
import sys


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_management.transform.data_unifier import LinguisticEntry, WordForm
from pydantic import BaseModel, Field, ConfigDict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import stanza
    STANZA_AVAILABLE = True
except ImportError:
    STANZA_AVAILABLE = False


# --- Enrichment providers -------------------------------------------------

EnrichFn = Callable[[str], Tuple[Optional[str], Dict[str, str], float]]


def _build_spacy_enricher(logger: logging.Logger, cache_limit: int = 20000) -> Optional[EnrichFn]:
    """Create a spaCy-based enricher (default path)."""
    if not SPACY_AVAILABLE:
        logger.info("spaCy not installed; skipping spaCy enrichment")
        return None

    try:
        # Disable NER to keep the pipeline light; POS/morph still available.
        nlp = spacy.load("uk_core_news_lg", disable=["ner"])
        logger.info("Using spaCy 'uk_core_news_lg' for POS/morph enrichment (default)")
    except OSError:
        logger.warning(
            "spaCy model 'uk_core_news_lg' not found. Install with: python -m spacy download uk_core_news_lg"
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"spaCy failed to load: {exc}")
        return None

    cache: Dict[str, Tuple[Optional[str], Dict[str, str], float]] = {}

    def enrich(form: str) -> Tuple[Optional[str], Dict[str, str], float]:
        if form in cache:
            return cache[form]

        try:
            doc = nlp(form)
            token = next((t for t in doc if not t.is_space), None)
            if not token:
                return None, {}, 0.0

            features = token.morph.to_dict() if token.morph else {}
            result = (token.pos_ or "X", features, 0.85)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"spaCy enrichment error for '{form}': {exc}")
            result = (None, {}, 0.0)

        if len(cache) >= cache_limit:
            cache.pop(next(iter(cache)))
        cache[form] = result
        return result

    return enrich


def _build_stanza_enricher(logger: logging.Logger, cache_limit: int = 20000) -> Optional[EnrichFn]:
    """Create a Stanza-based enricher (secondary, slower)."""
    if not STANZA_AVAILABLE:
        logger.info("Stanza not installed; skipping Stanza enrichment")
        return None

    try:
        stanza_dir = Path(__file__).parent / "stanza_resources"
        stanza_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Ensuring Stanza 'uk' models exist in {stanza_dir} ...")
        stanza.download("uk", model_dir=str(stanza_dir), verbose=False)

        logger.info("Loading Stanza pipeline for Ukrainian (secondary fallback)...")
        nlp = stanza.Pipeline(
            lang="uk",
            processors="tokenize,mwt,pos,lemma",
            dir=str(stanza_dir),
            use_gpu=False,
            logging_level="WARN",
            verbose=False,
        )
        logger.info("Stanza pipeline ready (fallback)")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not initialize Stanza enrichment: {exc}")
        return None

    cache: Dict[str, Tuple[Optional[str], Dict[str, str], float]] = {}

    def enrich(form: str) -> Tuple[Optional[str], Dict[str, str], float]:
        if form in cache:
            return cache[form]

        try:
            doc = nlp(form)
            if not doc.sentences or not doc.sentences[0].words:
                return None, {}, 0.0

            word = doc.sentences[0].words[0]
            pos = (word.upos or word.pos or "X")

            features: Dict[str, str] = {}
            if word.feats:
                for feat_pair in word.feats.split("|"):
                    if "=" in feat_pair:
                        key, val = feat_pair.split("=", 1)
                        features[key] = val

            result = (pos, features, 0.7)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Stanza enrichment error for '{form}': {exc}")
            result = (None, {}, 0.0)

        if len(cache) >= cache_limit:
            cache.pop(next(iter(cache)))
        cache[form] = result
        return result

    return enrich


class TrainingDBConfig(BaseModel):
    """Configuration for training DB export."""
    db_path: Path = Field(..., description="Path to output training database")
    merged_db_path: Path = Field(..., description="Path to input merged SQL database")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing database")
    batch_size: int = Field(default=1000, description="Batch size for SQL inserts", ge=1, le=10000)
    compute_hash: bool = Field(default=True, description="Whether to compute reproducibility hash")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow Path objects


class StressTrainingDBExporter:
    """
    Export stress prediction training database from merged SQL.
    
    Creates an optimized SQLite DB with:
    - training_entries table (form, lemma, stress, pos, features, definition, variant_type)
    - Indexes for fast querying
    - Hash tracking for reproducibility
    """
    
    SCHEMA_SQL = """
    -- Core training table
    CREATE TABLE IF NOT EXISTS training_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        
        -- Word data
        lemma TEXT NOT NULL,                    -- Base lemma (e.g., "замок")
        form TEXT NOT NULL,                     -- Specific inflected form (surface form)
        stress_indices TEXT NOT NULL,           -- JSON: [0] or [1] or [0,1]
        
        -- Linguistic features
        pos TEXT NOT NULL,                      -- NOUN, VERB, ADJ, etc.
        features_json TEXT,                     -- JSON: {"Case": "Nom", "Gender": "Masc", ...}
        
        -- Feature confidence scores (0.0-1.0)
        pos_confidence REAL DEFAULT 1.0,        -- 1.0=from data, 0.9=from lemma, 0.7=from Stanza
        features_confidence REAL DEFAULT 1.0,   -- 1.0=from data, 0.7=from Stanza
        
        -- Classification
        variant_type TEXT NOT NULL,             -- 'single' | 'grammatical_homonym' | 'morphological_variant' | 'free_variant'
        is_disambiguable INTEGER DEFAULT 0,    -- 1 if morphology/definition disambiguates
        is_homonym INTEGER DEFAULT 0,          -- 1 if multiple stress patterns exist for lemma/form
        
        -- Training metadata
        split TEXT DEFAULT 'train',             -- 'train', 'val', 'test'
        
        UNIQUE(lemma, form, stress_indices, pos, features_json)
    );
    
    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_variant_type ON training_entries(variant_type);
    CREATE INDEX IF NOT EXISTS idx_lemma ON training_entries(lemma);
    CREATE INDEX IF NOT EXISTS idx_form ON training_entries(form);
    CREATE INDEX IF NOT EXISTS idx_pos ON training_entries(pos);
    CREATE INDEX IF NOT EXISTS idx_split ON training_entries(split);
    CREATE INDEX IF NOT EXISTS idx_is_disambiguable ON training_entries(is_disambiguable);
    CREATE INDEX IF NOT EXISTS idx_is_homonym ON training_entries(is_homonym);
    
    -- Metadata table
    CREATE TABLE IF NOT EXISTS db_metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    
    -- Stats table
    CREATE TABLE IF NOT EXISTS db_stats (
        metric TEXT PRIMARY KEY,
        value TEXT
    );
    """
    
    def __init__(self, config: TrainingDBConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger("StressTrainingDBExporter")
        self.enrichment_cache_limit = 20000
        self.enrichers: List[Tuple[str, EnrichFn]] = []

        spacy_enricher = _build_spacy_enricher(self.logger, self.enrichment_cache_limit)
        if spacy_enricher:
            self.enrichers.append(("spaCy", spacy_enricher))

        stanza_enricher = _build_stanza_enricher(self.logger, self.enrichment_cache_limit)
        if stanza_enricher:
            self.enrichers.append(("Stanza", stanza_enricher))

        if not self.enrichers:
            self.logger.info("No NLP enrichers available; using lemma-based fallback only.")
    
    def create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema."""
        conn.executescript(self.SCHEMA_SQL)
        conn.commit()
    
    def _enrich_form(self, form: str) -> Tuple[Optional[str], float, Dict[str, str], float]:
        """Try available enrichers (spaCy first, then Stanza) for POS/features."""
        pos: Optional[str] = None
        pos_conf = 0.0
        feats: Dict[str, str] = {}
        feats_conf = 0.0

        for name, enrich_fn in self.enrichers:
            try:
                enriched_pos, enriched_feats, confidence = enrich_fn(form)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug(f"{name} enrichment failed for '{form}': {exc}")
                continue

            if enriched_pos and enriched_pos != "X" and pos is None:
                pos = enriched_pos
                pos_conf = confidence

            if enriched_feats and not feats:
                feats = enriched_feats
                feats_conf = confidence

            if pos and feats:
                break

        return pos, pos_conf, feats, feats_conf
    
    def determine_variant_type(
        self, 
        lemma: str, 
        forms: List[WordForm]
    ) -> str:
        """
        Determine variant type based on forms.
        
        Classification rules:
        - single: only one unique stress pattern across all forms
        - free_variant: multiple stresses with identical POS, features, and definition
          Example: 'помилка' - пОмилка/помИлка both NOUN, feminine, nominative, same meaning
        - grammatical_homonym: same POS/features but different definitions and stresses
          Example: 'замок' - замОк (castle) vs замЫк (lock), both NOUN but different senses
        - morphological_variant: different POS or features with different stresses
          Example: 'краї' (edges, NOUN) vs 'краї' (will edge, VERB), different POS
        
        Args:
            lemma: The base word form (for logging/debugging)
            forms: List of WordForm objects with complete morphological and semantic data
            
        Returns:
            One of: 'single', 'free_variant', 'grammatical_homonym', 'morphological_variant'
        """
        if not forms:
            return 'single'
        
        stress_defs: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        
        for form in forms:
            # Create a hashable key for stress pattern (sorted tuple)
            stress_key = tuple(sorted(form.stress_indices or []))
            
            # Create hashable keys for morphology and semantics
            pos_feats_json = json.dumps(
                {'pos': str(form.pos), 'feats': form.feats or {}},
                sort_keys=True
            )
            definition = form.main_definition or ""
            
            if stress_key not in stress_defs:
                stress_defs[stress_key] = {
                    'pos_feats_set': set(),
                    'definitions_set': set(),
                    'forms': []
                }
            
            stress_defs[stress_key]['pos_feats_set'].add(pos_feats_json)
            stress_defs[stress_key]['definitions_set'].add(definition)
            stress_defs[stress_key]['forms'].append(form)
        
        # Single stress pattern only
        if len(stress_defs) <= 1:
            return 'single'
        
        # Multiple stress patterns - analyze variation
        all_pos_feats = set()
        all_definitions = set()
        
        for stress_key, data in stress_defs.items():
            all_pos_feats.update(data['pos_feats_set'])
            all_definitions.update(data['definitions_set'])
        
        # Free variant: same morphology and semantics, different stresses
        # Example: 'помилка' has multiple acceptable stresses, same meaning
        if len(all_pos_feats) == 1 and len(all_definitions) <= 1:
            return 'free_variant'
        
        # Grammatical homonym: same POS/features but different definitions/senses
        # Example: 'замок' (castle) vs 'замок' (lock) - same morphology, different senses
        if len(all_pos_feats) == 1 and len(all_definitions) > 1:
            return 'grammatical_homonym'
        
        # Morphological variant: different POS or features
        # Example: different inflectional forms with different stress patterns
        return 'morphological_variant'
    
    def export_from_merged_sql(self) -> Tuple[int, str]:
        """
        Export training data from merged SQL database.
        
        Returns:
            Tuple of (rows_inserted, hash_of_input_data)
        """
        if self.config.overwrite and self.config.db_path.exists():
            self.logger.info(f"Removing existing database at {self.config.db_path}")
            self.config.db_path.unlink()
        
        # Create output dir
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to merged SQL DB
        merged_conn = sqlite3.connect(str(self.config.merged_db_path))
        merged_conn.row_factory = sqlite3.Row
        
        # Create training DB
        training_conn = sqlite3.connect(str(self.config.db_path))
        training_conn.execute("PRAGMA synchronous = OFF;")
        training_conn.execute("PRAGMA journal_mode = WAL;")
        training_conn.execute("PRAGMA temp_store = MEMORY;")
        self.create_schema(training_conn)
        
        # Compute input hash if requested
        input_hash = ""
        if self.config.compute_hash:
            input_hash = self._compute_input_hash()
        
        try:
            rows_inserted = self._export_entries(merged_conn, training_conn)
            self._save_metadata(training_conn, input_hash)
            self._save_stats(training_conn)
            
            training_conn.commit()
            self.logger.info(f"✓ Training DB export complete: {rows_inserted} rows")
            
            return rows_inserted, input_hash
        
        finally:
            merged_conn.close()
            training_conn.close()
    
    def _export_entries(self, merged_conn: sqlite3.Connection, training_conn: sqlite3.Connection) -> int:
        """
        Export entries from merged SQL to training DB using true streaming approach.
        
        Two-pass algorithm to avoid unbounded memory growth:
        1. First pass: Group by lemma, compute variant types, store in temp table
        2. Second pass: Stream from temp table + word_form, insert into training DB
        
        This ensures O(batch_size) memory usage, not O(total_lemmas).
        """
        
        self.logger.info("Pass 1: Computing variant types per lemma...")
        
        # Create temporary table for variant types and lemma POS
        training_conn.execute("""
            CREATE TEMPORARY TABLE lemma_variant_types (
                lemma TEXT PRIMARY KEY,
                variant_type TEXT NOT NULL,
                is_disambiguable INTEGER NOT NULL,
                lemma_pos TEXT  -- Most common POS for this lemma
            )
        """)
        
        # First pass: Compute variant types per lemma (streaming by lemma)
        lemma_cursor = merged_conn.cursor()
        lemma_cursor.execute("""
            SELECT DISTINCT lemma
            FROM word_form
            ORDER BY lemma
        """)
        
        lemma_count = 0
        lemma_batch = []
        
        for lemma_row in tqdm(lemma_cursor, desc="Computing variant types", unit="lemma"):
            lemma = lemma_row[0]
            
            # Get all forms for this lemma
            forms_cursor = merged_conn.cursor()
            forms_cursor.execute("""
                SELECT DISTINCT
                    wf.form,
                    wf.pos,
                    GROUP_CONCAT(f.key || ':' || f.value, '|') as features_str,
                    wf.stress_indices_json,
                    d.text as main_definition
                FROM word_form wf
                LEFT JOIN feature f ON wf.id = f.word_form_id
                LEFT JOIN definition d ON wf.main_definition_id = d.id
                WHERE wf.lemma = ?
                GROUP BY wf.id
            """, (lemma,))

            lemma_forms = []
            for form_row in forms_cursor:
                form_val = form_row[0]
                pos_val = form_row[1]
                features_str = form_row[2]
                stress_json = form_row[3]
                definition_val = form_row[4]

                form_features = {}
                if features_str:
                    for feat_pair in features_str.split('|'):
                        if ':' in feat_pair:
                            key, value = feat_pair.split(':', 1)
                            form_features[key] = value

                try:
                    form_obj = WordForm(
                        form=form_val or "",
                        stress_indices=json.loads(stress_json or '[]'),
                        pos=pos_val,
                        feats=form_features,
                        main_definition=definition_val or None
                    )
                    lemma_forms.append(form_obj)
                except Exception as e:
                    self.logger.debug(f"Skip form for lemma={lemma}: {e}")
                    continue
            
            if lemma_forms:
                variant_type = self.determine_variant_type(lemma, lemma_forms)
                is_disambiguable = 1 if variant_type in ['morphological_variant', 'grammatical_homonym'] else 0
                
                # Determine most common POS for lemma (use first non-X POS, or X if all missing)
                lemma_pos = "X"
                pos_counts = {}
                for form_obj in lemma_forms:
                    pos_val = form_obj.pos or "X"
                    pos_counts[pos_val] = pos_counts.get(pos_val, 0) + 1
                
                # Prefer non-X POS
                for pos_val in sorted(pos_counts.keys(), key=lambda x: (x == "X", -pos_counts[x])):
                    if pos_val != "X":
                        lemma_pos = pos_val
                        break
                
                lemma_batch.append((lemma, variant_type, is_disambiguable, lemma_pos))
                lemma_count += 1
            
            # Flush batch to temp table
            if len(lemma_batch) >= self.config.batch_size:
                training_conn.executemany(
                    "INSERT OR REPLACE INTO lemma_variant_types (lemma, variant_type, is_disambiguable, lemma_pos) VALUES (?, ?, ?, ?)",
                    lemma_batch
                )
                training_conn.commit()
                lemma_batch = []
        
        # Final batch
        if lemma_batch:
            training_conn.executemany(
                "INSERT OR REPLACE INTO lemma_variant_types (lemma, variant_type, is_disambiguable, lemma_pos) VALUES (?, ?, ?, ?)",
                lemma_batch
            )
            training_conn.commit()
        
        self.logger.info(f"✓ Pass 1 complete: {lemma_count} unique lemmas")
        
        # Second pass: Stream word forms and join with variant types
        self.logger.info("Pass 2: Exporting training entries...")
        
        cursor = merged_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM word_form")
        total_rows = cursor.fetchone()[0]
        
        cursor = merged_conn.cursor()
        cursor.execute("""
            SELECT 
                wf.form,
                wf.lemma,
                wf.pos,
                wf.stress_indices_json,
                d.text as main_definition,
                GROUP_CONCAT(f.key || ':' || f.value, '|') as features_str
            FROM word_form wf
            LEFT JOIN feature f ON wf.id = f.word_form_id
            LEFT JOIN definition d ON wf.main_definition_id = d.id
            GROUP BY wf.id, wf.form, wf.lemma, wf.pos, wf.stress_indices_json, d.text
            ORDER BY wf.lemma
        """)
        
        batch = []
        rows_inserted = 0
        
        pbar = tqdm(total=total_rows, desc="Exporting entries", ncols=100)
        
        for row in cursor:
            form_val = row['form']
            lemma = row['lemma']
            pos = row['pos']
            stress_indices_json = row['stress_indices_json']
            definition = row['main_definition'] or ""  # used only for variant typing, not stored
            features_str = row['features_str']
            
            # Parse features
            features = {}
            pos_confidence = 1.0  # Original data
            features_confidence = 1.0  # Original data
            
            if features_str:
                for feat_pair in features_str.split('|'):
                    if ':' in feat_pair:
                        key, value = feat_pair.split(':', 1)
                        features[key] = value
            
            # Get variant type and lemma POS from temp table
            variant_cursor = training_conn.cursor()
            variant_cursor.execute(
                "SELECT variant_type, is_disambiguable, lemma_pos FROM lemma_variant_types WHERE lemma = ?",
                (lemma,)
            )
            variant_row = variant_cursor.fetchone()
            
            if not variant_row:
                self.logger.warning(f"No variant type found for lemma={lemma}")
                pbar.update(1)
                continue
            
            variant_type = variant_row[0]
            is_disambiguable = variant_row[1]
            lemma_pos = variant_row[2]
            is_homonym = 0 if variant_type == 'single' else 1
            
            # Enrichment (spaCy preferred, Stanza as secondary)
            enriched_pos = None
            enriched_feats: Dict[str, str] = {}
            enriched_pos_conf = 0.0
            enriched_feats_conf = 0.0

            if ((pos in ("X", None, "")) or not features) and self.enrichers:
                enriched_pos, enriched_pos_conf, enriched_feats, enriched_feats_conf = self._enrich_form(form_val)

            # Enrich POS: data > lemma > spaCy/Stanza
            if pos in ("X", None, ""):
                pos = lemma_pos
                pos_confidence = 0.9  # From lemma

                if (pos in ("X", None, "")) and enriched_pos:
                    pos = enriched_pos
                    pos_confidence = enriched_pos_conf

            # Enrich morphology if missing
            if not features and enriched_feats:
                features.update(enriched_feats)
                features_confidence = enriched_feats_conf
            
            batch.append((
                lemma,                          # lemma
                form_val,                       # form (surface form)
                stress_indices_json,            # stress_indices
                pos,                            # pos
                json.dumps(features),           # features_json
                pos_confidence,                 # pos_confidence
                features_confidence,            # features_confidence
                variant_type,                   # variant_type
                is_disambiguable,               # is_disambiguable
                is_homonym,                     # is_homonym
                'train'                        # split
            ))
            
            if len(batch) >= self.config.batch_size:
                training_conn.executemany("""
                    INSERT OR IGNORE INTO training_entries
                    (lemma, form, stress_indices, pos, features_json,
                     pos_confidence, features_confidence,
                     variant_type, is_disambiguable, is_homonym, split)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                rows_inserted += len(batch)
                batch = []
            
            pbar.update(1)
        
        # Final batch
        if batch:
            training_conn.executemany("""
                INSERT OR IGNORE INTO training_entries
                (lemma, form, stress_indices, pos, features_json,
                 pos_confidence, features_confidence,
                 variant_type, is_disambiguable, is_homonym, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            rows_inserted += len(batch)
        
        pbar.close()
        return rows_inserted
    
    def _compute_input_hash(self) -> str:
        """
        Compute hash of input data sources for reproducibility.
        Hash includes:
        - Merged SQL DB file
        - This script
        - STRESS_SERVICE_DB_DESIGN.md
        """
        hasher = hashlib.sha256()
        
        # Hash merged SQL DB
        if self.config.merged_db_path.exists():
            with open(self.config.merged_db_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
        
        # Hash this script
        script_path = Path(__file__)
        if script_path.exists():
            with open(script_path, 'rb') as f:
                hasher.update(f.read())
        
        # Hash design doc
        design_path = Path(__file__).parent.parent.parent / "STRESS_SERVICE_DB_DESIGN.md"
        if design_path.exists():
            with open(design_path, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _save_metadata(self, conn: sqlite3.Connection, input_hash: str) -> None:
        """Save metadata about the export."""
        from datetime import datetime
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'input_hash': input_hash,
            'source_db': str(self.config.merged_db_path),
            'script': 'export_training_db.py',
            'purpose': 'Stress prediction model training dataset'
        }
        
        for key, value in metadata.items():
            conn.execute(
                "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
                (key, str(value))
            )
    
    def _save_stats(self, conn: sqlite3.Connection) -> None:
        """Save statistics about the exported data."""
        cursor = conn.cursor()
        
        stats = {
            'total_entries': "SELECT COUNT(*) FROM training_entries",
            'unique_lemmas': "SELECT COUNT(DISTINCT lemma) FROM training_entries",
            'unique_forms': "SELECT COUNT(DISTINCT form) FROM training_entries",
            'single_stress': "SELECT COUNT(*) FROM training_entries WHERE variant_type='single'",
            'grammatical_homonym': "SELECT COUNT(*) FROM training_entries WHERE variant_type='grammatical_homonym'",
            'morphological_variant': "SELECT COUNT(*) FROM training_entries WHERE variant_type='morphological_variant'",
            'free_variant': "SELECT COUNT(*) FROM training_entries WHERE variant_type='free_variant'",
            'unique_pos': "SELECT COUNT(DISTINCT pos) FROM training_entries",
            'disambiguable': "SELECT COUNT(*) FROM training_entries WHERE is_disambiguable=1",
            'homonym': "SELECT COUNT(*) FROM training_entries WHERE is_homonym=1"
        }
        
        for metric, query in stats.items():
            cursor.execute(query)
            value = cursor.fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO db_stats (metric, value) VALUES (?, ?)",
                (metric, str(value))
            )
            self.logger.info(f"  {metric}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Export stress prediction training database from merged SQL DB"
    )
    parser.add_argument(
        '--merged-db',
        type=Path,
        default=Path(__file__).parent.parent / "transform/cache/MERGEDSQL_33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665.sqlite3",
        help="Path to merged SQL database (default: transform/cache/MERGEDSQL_*.sqlite3)"
    )
    parser.add_argument(
        '--output-db',
        type=Path,
        default=Path(__file__).parent.parent.parent / "stress_prediction/data/stress_training.db",
        help="Path to output training database (default: stress_prediction/data/stress_training.db)"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing database"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help="Batch size for inserts"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("StressTrainingExport")
    
    config = TrainingDBConfig(
        db_path=args.output_db,
        merged_db_path=args.merged_db,
        overwrite=args.overwrite,
        batch_size=args.batch_size
    )
    
    logger.info(f"Starting stress prediction training DB export")
    logger.info(f"  Merged DB: {config.merged_db_path}")
    logger.info(f"  Output DB: {config.db_path}")
    
    if not config.merged_db_path.exists():
        logger.error(f"Merged DB not found: {config.merged_db_path}")
        return 1
    
    exporter = StressTrainingDBExporter(config, logger)
    
    try:
        rows_inserted, input_hash = exporter.export_from_merged_sql()
        logger.info(f"✓ Export successful!")
        logger.info(f"  Rows: {rows_inserted}")
        logger.info(f"  Hash: {input_hash[:16]}...")
        return 0
    
    except Exception as e:
        logger.error(f"✗ Export failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
