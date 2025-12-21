#!/usr/bin/env python3
"""
Ukrainian Stress Service

Provides word stress lookup functionality using LMDB database.
Returns all possible stress variants and morphological data for Ukrainian words.
"""

from pathlib import Path
from typing import Optional, List
from logging import getLogger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_db_generator.lmdb_exporter import LMDBQuery
from src.nlp.stress_service.types import WordLookupResult, WordFormDict, format_stress_display
from src.nlp.utils.normalize_apostrophe import normalize_apostrophe

logger = getLogger(__name__)


class UkrainianStressService:
    """
    Service for looking up Ukrainian word stress and morphology.
    
    Features:
    - Fast LMDB-based lookups (300k+ queries/sec)
    - Automatic word normalization (lowercase, apostrophe)
    - Support for heteronyms (words with multiple stress patterns)
    - Rich morphological data (POS, Case, Gender, Number, etc.)
    
    Usage:
        service = UkrainianStressService()
        results = service.lookup("замок")
        # Returns all possible forms with different stress positions
        service.close()
    
    Or with context manager:
        with UkrainianStressService() as service:
            results = service.lookup("замок")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize stress service.
        
        Args:
            db_path: Path to LMDB database directory.
                    If None, uses default location: src/nlp/stress_service/stress.lmdb
        """
        if db_path is None:
            # Default path relative to this file
            db_path = Path(__file__).parent / "stress.lmdb"
        
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Stress database not found at {self.db_path}. "
                f"Run generate_db.py to create the database."
            )
        
        self.db = LMDBQuery(self.db_path)
        logger.info(f"Stress service initialized with database: {self.db_path}")
    
    @staticmethod
    def normalize_word(word: str) -> str:
        """
        Normalize word for lookup.
        
        Args:
            word: Word to normalize
        
        Returns:
            Normalized word (lowercase, correct apostrophe)
        """
        # Normalize apostrophe
        normalized = normalize_apostrophe(word)
        # Convert to lowercase
        return normalized.lower()
    
    def lookup(self, word: str, normalize: bool = True) -> Optional[WordLookupResult]:
        """
        Look up word stress and morphology.
        
        Args:
            word: Word to look up (e.g., "замок", "атлас")
            normalize: Whether to normalize the word (default: True)
        
        Returns:
            List of word forms with stress and morphology data, or None if not found.
            
            Example result for "замок":
            [
                {
                    "stress_variants": [0],
                    "pos": ["NOUN"],
                    "feats": {"Case": ["Nom"], "Gender": ["Masc"], "Number": ["Sing"]}
                },
                {
                    "stress_variants": [2],
                    "pos": ["NOUN"],
                    "feats": {"Case": ["Nom"], "Gender": ["Masc"], "Number": ["Sing"]}
                }
            ]
        """
        # Normalize if requested
        lookup_word = self.normalize_word(word) if normalize else word
        
        # Query database
        result = self.db.lookup(lookup_word)
        
        if result is None:
            logger.debug(f"Word not found: {word}")
            return None
        
        return result
    
    def get_stress_variants(self, word: str, normalize: bool = True) -> Optional[List[str]]:
        """
        Get all stress variants for a word with visual stress marks.
        
        Args:
            word: Word to look up
            normalize: Whether to normalize the word (default: True)
        
        Returns:
            List of stress-marked variants, or None if word not found.
            
            Example for "замок":
            ["за́мок", "замо́к"]
        """
        result = self.lookup(word, normalize)
        
        if result is None:
            return None
        
        # Extract unique stress patterns
        stress_variants = []
        seen_patterns = set()
        
        for form in result:
            stress_indices = tuple(form.get("stress_variants", []))
            if stress_indices not in seen_patterns:
                seen_patterns.add(stress_indices)
                # Apply stress marks to original word (not normalized)
                stressed = format_stress_display(word, list(stress_indices))
                stress_variants.append(stressed)
        
        return stress_variants if stress_variants else None
    
    def is_heteronym(self, word: str, normalize: bool = True) -> bool:
        """
        Check if word is a heteronym (has multiple stress patterns).
        
        Args:
            word: Word to check
            normalize: Whether to normalize the word (default: True)
        
        Returns:
            True if word has multiple stress patterns, False otherwise.
        """
        variants = self.get_stress_variants(word, normalize)
        return variants is not None and len(variants) > 1
    
    def get_pos_tags(self, word: str, normalize: bool = True) -> Optional[List[str]]:
        """
        Get all part-of-speech tags for a word.
        
        Args:
            word: Word to look up
            normalize: Whether to normalize the word (default: True)
        
        Returns:
            List of unique POS tags, or None if word not found.
        """
        result = self.lookup(word, normalize)
        
        if result is None:
            return None
        
        # Extract unique POS tags
        pos_tags = set()
        for form in result:
            pos_tags.update(form.get("pos", []))
        
        return sorted(list(pos_tags)) if pos_tags else None
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database stats (entries, size, etc.)
        """
        return self.db.get_stats()
    
    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("Stress service closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()
