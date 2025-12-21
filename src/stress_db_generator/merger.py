#!/usr/bin/env python3
"""
Dictionary Merger

Merges data from multiple sources (trie + txt) into unified dictionary structure.
Implements intelligent merging of stress patterns and morphological features.

Responsibilities:
- Combine trie data (with morphology) and txt data (stress-only)
- Merge identical stress patterns with different morphology
- Handle conflicts and precedence rules
"""

from typing import Dict, List, Optional, Tuple
from logging import getLogger
from dataclasses import dataclass, field

logger = getLogger(__name__)


@dataclass
class WordForm:
    """
    Single form of a word with stress and optional morphology.
    Uses lists for ALL feature values (consistent design pattern).
    
    Attributes:
        stress_variants: List of vowel indices where stress occurs (0-indexed)
        pos: List of Universal POS tags (usually single item)
        feats: Morphological features dict where each value is a list
        lemma: Base form of the word (optional)
        source: Data source ("trie", "txt", "merged")
    """
    stress_variants: List[int]
    pos: List[str] = field(default_factory=list)
    feats: Dict[str, List[str]] = field(default_factory=dict)
    lemma: Optional[str] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Export to dictionary format (for LMDB)"""
        result = {
            "stress_variants": self.stress_variants,
        }
        if self.pos:
            result["pos"] = self.pos
        if self.feats:
            # Sort features alphabetically, keep lists
            result["feats"] = {k: sorted(v) for k, v in sorted(self.feats.items())}
        if self.lemma:
            result["lemma"] = self.lemma
        return result


class DictionaryMerger:
    """
    Merges dictionary data from multiple sources.
    
    Merge Strategy:
    1. Start with trie data (has morphology)
    2. Add txt data (stress-only, may have additional stress patterns)
    3. For identical stress patterns: merge morphological features
    4. For new stress patterns: add as new forms
    
    Priority: Trie morphology > txt stress-only entries
    """
    
    def __init__(self):
        # Unified dictionary: normalized_key → List[WordForm]
        self.words: Dict[str, List[WordForm]] = {}
    
    def add_trie_data(self, trie_data: Dict[str, List[Tuple[List[int], Dict]]]) -> None:
        """
        Add data from trie parser.
        
        Args:
            trie_data: Dict mapping keys to list of (stress_indices, morphology) tuples
                Example: {"атлас": [([0], {"upos": "NOUN", "Case": "Nom"}), ...]}
        """
        logger.info(f"Adding trie data: {len(trie_data):,} words")
        
        for key, forms in trie_data.items():
            for stress_indices, morphology in forms:
                # Parse morphology into pos + feats (with lists)
                pos, feats = self._parse_morphology(morphology)
                
                # Add with merging
                self._add_word_form_with_merge(
                    key=key,
                    stress_variants=stress_indices,
                    pos=pos,
                    feats=feats,
                    source="trie"
                )
        
        logger.info(f"Trie data added: {len(self.words):,} unique words")
    
    def add_txt_data(self, txt_data: Dict[str, List[Tuple[List[int], Optional[str]]]]) -> None:
        """
        Add data from txt parser.
        
        Args:
            txt_data: Dict mapping keys to list of (stress_indices, definition) tuples
                Example: {"атлас": [([0], "збірник карт"), ([1], "тканина")]}
        """
        logger.info(f"Adding txt data: {len(txt_data):,} words")
        
        new_words = 0
        enriched_words = 0
        
        for key, forms in txt_data.items():
            word_existed = key in self.words
            
            for stress_indices, definition in forms:
                # Txt data has no morphology, just stress
                self._add_word_form_with_merge(
                    key=key,
                    stress_variants=stress_indices,
                    pos=[],
                    feats={},
                    source="txt"
                )
            
            if not word_existed:
                new_words += 1
            elif key in self.words:
                enriched_words += 1
        
        logger.info(f"TXT data added: {new_words:,} new words, {enriched_words:,} enriched with additional stress patterns")
        logger.info(f"Total unique words: {len(self.words):,}")
    
    def _add_word_form_with_merge(self, key: str, stress_variants: List[int], 
                                   pos: List[str], feats: Dict[str, List[str]],
                                   source: str = "unknown") -> None:
        """
        Add word form with intelligent merging of identical stress patterns.
        
        Merging Rules:
        1. Same stress + same POS (or empty POS) → merge features
        2. Same stress + different POS → keep separate forms
        3. Different stress → always keep separate
        
        Args:
            key: Word key
            stress_variants: Stress vowel indices
            pos: List of POS tags
            feats: Dict of feature lists
            source: Data source identifier
        """
        if key not in self.words:
            self.words[key] = []
        
        # Look for existing form with same stress pattern and compatible POS
        for existing_form in self.words[key]:
            if existing_form.stress_variants == stress_variants:
                # Check if POS is compatible (same or one is subset of other)
                pos_compatible = (
                    not pos or not existing_form.pos or 
                    set(pos) == set(existing_form.pos)
                )
                
                if pos_compatible:
                    # Merge POS
                    if pos:
                        existing_form.pos = sorted(list(set(existing_form.pos + pos)))
                    
                    # Merge features (combine lists for each key)
                    for feat_key, feat_values in feats.items():
                        if feat_key in existing_form.feats:
                            # Merge and deduplicate
                            existing_form.feats[feat_key] = sorted(list(set(
                                existing_form.feats[feat_key] + feat_values
                            )))
                        else:
                            # New feature
                            existing_form.feats[feat_key] = sorted(feat_values)
                    
                    # Update source to "merged" if combining different sources
                    if existing_form.source != source:
                        existing_form.source = "merged"
                    
                    return  # Merged successfully
        
        # No matching form found - create new one
        word_form = WordForm(
            stress_variants=stress_variants,
            pos=pos,
            feats=feats,
            source=source
        )
        self.words[key].append(word_form)
    
    def _parse_morphology(self, morphology: Optional[Dict]) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Parse morphology dict into pos + feats structure with lists.
        
        Args:
            morphology: Flat dict like {'upos': 'NOUN', 'Case': 'Nom', 'Number': 'Sing'}
        
        Returns:
            (pos_list, feats_dict) where:
            - pos_list: List like ["NOUN"] or []
            - feats_dict: Dict like {"Case": ["Nom"], "Number": ["Sing"]}
        """
        if not morphology:
            return [], {}
        
        # Extract POS as list
        pos = [morphology['upos']] if morphology.get('upos') else []
        
        # Extract features as dict of lists
        feats = {}
        for key, value in morphology.items():
            if key != 'upos' and value:
                feats[key] = [value]  # Wrap single value in list
        
        return pos, feats
    
    def get_dictionary(self) -> Dict[str, List[WordForm]]:
        """
        Get the merged dictionary.
        
        Returns:
            Dictionary mapping keys to list of WordForm objects
        """
        return self.words
    
    def get_statistics(self) -> Dict:
        """Get merger statistics"""
        total_forms = sum(len(forms) for forms in self.words.values())
        
        # Count source distribution
        trie_only = sum(1 for forms in self.words.values() 
                       if all(f.source == "trie" for f in forms))
        txt_only = sum(1 for forms in self.words.values() 
                      if all(f.source == "txt" for f in forms))
        merged = sum(1 for forms in self.words.values() 
                    if any(f.source == "merged" for f in forms))
        
        # Count heteronyms
        heteronyms = sum(1 for forms in self.words.values() 
                        if len({tuple(f.stress_variants) for f in forms}) > 1)
        
        # Count words with morphology
        words_with_morph = sum(1 for forms in self.words.values() 
                              if any(f.pos or f.feats for f in forms))
        
        return {
            "total_unique_words": len(self.words),
            "total_forms": total_forms,
            "heteronyms": heteronyms,
            "words_with_morphology": words_with_morph,
            "trie_only": trie_only,
            "txt_only": txt_only,
            "merged": merged,
            "avg_forms_per_word": round(total_forms / len(self.words), 2) if self.words else 0
        }


def merge_dictionaries(trie_data: Dict[str, List[Tuple[List[int], Dict]]],
                       txt_data: Dict[str, List[Tuple[List[int], Optional[str]]]]) -> Dict[str, List[WordForm]]:
    """
    Convenience function to merge trie and txt dictionaries.
    
    Args:
        trie_data: Data from trie parser
        txt_data: Data from txt parser
    
    Returns:
        Merged dictionary with WordForm objects
    """
    merger = DictionaryMerger()
    merger.add_trie_data(trie_data)
    merger.add_txt_data(txt_data)
    return merger.get_dictionary()
