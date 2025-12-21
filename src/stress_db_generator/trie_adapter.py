#!/usr/bin/env python3
"""
Trie Data Adapter

Bridges the trie_parser output format to the format expected by merger.
Converts TrieEntry objects to (stress_indices, morphology) tuples.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from logging import getLogger

from src.stress_db_generator.trie_parser import TrieParser
from src.nlp.utils.normalize_apostrophe import normalize_apostrophe

logger = getLogger(__name__)


class TrieDataAdapter:
    """
    Adapter for converting trie data to merger-compatible format.
    
    Responsibilities:
    - Parse trie file using TrieParser
    - Normalize word keys (lowercase, apostrophe)
    - Convert TrieEntry objects to (stress_indices, morphology) tuples
    """
    
    def __init__(self, trie_path: Path):
        """
        Initialize adapter with trie file.
        
        Args:
            trie_path: Path to stress.trie file
        """
        self.parser = TrieParser(trie_path)
    
    @staticmethod
    def normalize_key(word: str) -> str:
        """
        Normalize word key for dictionary lookup.
        
        Args:
            word: Word to normalize
        
        Returns:
            Normalized key (lowercase, correct apostrophe)
        """
        normalized = normalize_apostrophe(word)
        return normalized.lower()
    
    def parse_trie(self, progress_callback=None) -> Dict[str, List[Tuple[List[int], Dict]]]:
        """
        Parse entire trie file and convert to merger format.
        
        Args:
            progress_callback: Optional callback function(current, total) for progress tracking
        
        Returns:
            Dictionary mapping normalized keys to list of (stress_indices, morphology) tuples
            
            Example:
            {
                "атлас": [
                    ([0], {"upos": "NOUN", "Case": "Nom", "Number": "Sing"}),
                    ([1], {"upos": "NOUN", "Case": "Nom", "Number": "Sing"})
                ]
            }
        """
        logger.info("Parsing trie data...")
        
        result: Dict[str, List[Tuple[List[int], Dict]]] = {}
        
        # Get all keys first to know total count
        all_keys = list(self.parser.trie.keys())
        total = len(all_keys)
        
        # Iterate through all words in trie
        total_words = 0
        total_forms = 0
        
        for idx, word in enumerate(all_keys, 1):
            # Normalize the key
            key = self.normalize_key(word)
            
            # Get all forms for this word
            forms = self.parser.get_word_forms(word)
            
            if key not in result:
                result[key] = []
            
            for form in forms:
                # Convert TrieEntry to (stress_indices, morphology) tuple
                result[key].append((form.stress_positions, form.morphology))
                total_forms += 1
            
            total_words += 1
            
            # Progress callback
            if progress_callback and (idx % 5000 == 0 or idx == total):
                progress_callback(idx, total)
            
            # Log progress every 100k words
            if total_words % 100000 == 0:
                logger.info(f"Processed {total_words:,} words, extracted {total_forms:,} forms")
        
        logger.info(f"Trie parsing complete: {total_words:,} words, {total_forms:,} forms")
        
        return result


def parse_trie_data(trie_path: Path, progress_callback=None) -> Dict[str, List[Tuple[List[int], Dict]]]:
    """
    Convenience function to parse trie data.
    
    Args:
        trie_path: Path to stress.trie file
        progress_callback: Optional callback function(current, total) for progress tracking
    
    Returns:
        Dictionary in merger-compatible format
    """
    adapter = TrieDataAdapter(trie_path)
    return adapter.parse_trie(progress_callback)
