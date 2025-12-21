#!/usr/bin/env python3
"""
Ukrainian Word Dictionary Generator (Professional NLP Structure)

Parses Ukrainian dictionary and keeps structured data in RAM.
Uses spaCy-compatible Universal Dependencies format for morphological features.

Data Structure (spaCy-compatible):
{
  "атлас": [
    {
      "stress_variants": [0],
      "pos": "NOUN",
      "feats": {
        "Case": "Acc",
        "Gender": "Masc",
        "Number": "Sing"
      }
    },
    {
      "stress_variants": [1],
      "pos": "NOUN",
      "feats": {
        "Case": "Acc",
        "Gender": "Masc",
        "Number": "Sing"
      }
    }
  ]
}

Follows Universal Dependencies (UD) annotation scheme:
- pos: Part-of-speech (UPOS tags)
- feats: Morphological features (sorted alphabetically)
"""

from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.nlp.utils.normalize_apostrophe import normalize_apostrophe

try:
    from src.stress_db_generator.trie_parser import TrieParser
    TRIE_AVAILABLE = True
except ImportError:
    TRIE_AVAILABLE = False

logger = getLogger(__name__)


# Unicode constants
APOSTROPHE_WRONG = "'"  # U+2019 ' - WRONG
APOSTROPHE_CORRECT = "ʼ"  # U+02BC ʼ - CORRECT

STRESS_MARK_ACUTE = "´"  # U+00B4 ´ - acute accent
STRESS_MARK_COMBINING = "́"  # U+0301 ́ - combining acute

UKRAINIAN_VOWELS = list("аеєиіїоуюяАЕЄИІЇОУЮЯ")


@dataclass
class WordForm:
    """
    Single form of a word with spaCy-compatible structure.
    Uses lists for ALL feature values (consistent design pattern).
    
    Follows Universal Dependencies (UD) annotation scheme:
    https://universaldependencies.org/
    
    Design principle: ALL values are lists, even single values.
    This provides consistency and handles morphological ambiguity naturally.
    
    Example:
        {
            "stress_variants": [0],
            "pos": ["NOUN"],
            "feats": {
                "Case": ["Nom", "Acc", "Voc"],  # Ambiguous cases
                "Gender": ["Fem"],              # Single value, still list
                "Number": ["Plur"]              # Single value, still list
            }
        }
    
    Attributes:
        stress_variants: List of vowel indices where stress occurs (0-indexed)
        pos: List of Universal POS tags (usually single item)
        feats: Morphological features dict where each value is a list
        lemma: Base form of the word (optional)
    """
    stress_variants: List[int]
    pos: List[str] = field(default_factory=list)  # Always list: ["NOUN"] or []
    feats: Dict[str, List[str]] = field(default_factory=dict)  # Each value is a list
    lemma: Optional[str] = None  # Base form
    
    def to_dict(self) -> Dict:
        """Export to dictionary format"""
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
    
    def to_spacy_format(self) -> str:
        """
        Export morphology to spaCy token.morph format.
        For features with multiple values, joins them with comma.
        
        Returns:
            String like "Case=Nom,Acc,Voc|Gender=Fem|Number=Plur"
        """
        if not self.feats:
            return ""
        
        parts = []
        for key in sorted(self.feats.keys()):
            values = sorted(self.feats[key])
            values_str = ",".join(values)
            parts.append(f"{key}={values_str}")
        
        return "|".join(parts)
    
    def matches_morphology(self, required_feats: Dict[str, str]) -> bool:
        """
        Check if this form matches required morphological features.
        
        Args:
            required_feats: Dict like {"Case": "Nom", "Number": "Plur"}
        
        Returns:
            True if all required features match (exist in this form's lists)
        """
        for key, value in required_feats.items():
            if key not in self.feats or value not in self.feats[key]:
                return False
        return True


class StressExtractor:
    """Extract stress indices from Ukrainian words"""
    
    @staticmethod
    def normalize_apostrophe_word(word: str) -> str:
        """Normalize apostrophe using utility function"""
        return normalize_apostrophe(word)
    
    @staticmethod
    def extract_stress_indices(word: str) -> Tuple[List[int], str]:
        """
        Extract stress positions and clean text.
        
        Returns:
            (stress_indices, clean_text)
            stress_indices: List of 0-based vowel indices where stress occurs
            clean_text: Word without stress marks
        """
        # Track which character positions are stressed (before removing marks)
        stressed_char_positions = set()
        
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i + 1] in (STRESS_MARK_ACUTE, STRESS_MARK_COMBINING):
                # Next char is stress mark - current char is stressed
                stressed_char_positions.add(i)
                i += 1  # Skip the stress mark
            i += 1
        
        # Build clean text without stress marks
        clean_chars = []
        char_index = 0
        for i, char in enumerate(word):
            if char not in (STRESS_MARK_ACUTE, STRESS_MARK_COMBINING):
                if i in stressed_char_positions:
                    stressed_char_positions.add(char_index)  # Map to position in clean text
                clean_chars.append(char)
                char_index += 1
        
        clean_text = "".join(clean_chars)
        
        # Convert character positions to vowel indices
        vowel_indices = []
        vowel_count = 0
        for i, char in enumerate(clean_text.lower()):
            if char in UKRAINIAN_VOWELS:
                if i in stressed_char_positions:
                    vowel_indices.append(vowel_count)
                vowel_count += 1
        
        return vowel_indices, clean_text
    
    @staticmethod
    def generate_key(word: str) -> str:
        """
        Generate normalized searchable key.
        
        Steps:
        1. Remove stress marks (´ and ́)
        2. Normalize apostrophe (' → ʼ)
        3. Lowercase
        """
        key = word.replace(STRESS_MARK_ACUTE, "")
        key = key.replace(STRESS_MARK_COMBINING, "")
        key = StressExtractor.normalize_apostrophe_word(key)
        return key.lower()


class StressDictionary:
    """
    In-memory stress dictionary with flat structure.
    
    Structure: Dict[str, List[WordForm]]
    - Key: normalized word (lowercase, no stress)
    - Value: List of forms (each with stress_variants)
    
    Heteronym detection: len > 1 with different stress patterns
    """
    
    def __init__(self):
        # Main dictionary: normalized_key → List[WordForm]
        self.words: Dict[str, List[WordForm]] = {}
    
    def add_word(self, stressed_form: str, morphology: Optional[Dict] = None) -> None:
        """
        Add a word to the dictionary.
        
        Args:
            stressed_form: Word with stress marks (e.g., "обі´ді")
            morphology: Optional morphology data (can be flat dict or structured)
        """
        # Normalize apostrophe
        stressed_form = StressExtractor.normalize_apostrophe_word(stressed_form)
        
        # Generate key
        key = StressExtractor.generate_key(stressed_form)
        
        # Extract stress
        stress_indices, _ = StressExtractor.extract_stress_indices(stressed_form)
        
        # Parse morphology into pos + feats (now returns lists)
        pos, feats = self._parse_morphology(morphology) if morphology else ([], {})
        
        # Add to dictionary with merging logic
        self._add_word_form_with_merge(key, stress_indices, pos, feats)
    
    def add_word_form(self, key: str, stress_variants: List[int], morphology: Optional[Dict] = None) -> None:
        """
        Add a word form directly with normalized key and stress indices.
        Used for trie integration.
        
        Args:
            key: Normalized word key (lowercase, no stress, normalized apostrophe)
            stress_variants: List of vowel indices where stress occurs
            morphology: Optional morphology data (flat dict with 'upos', 'Case', etc.)
        """
        # Parse morphology into pos + feats (now returns lists)
        pos, feats = self._parse_morphology(morphology) if morphology else ([], {})
        
        # Add to dictionary with merging logic
        self._add_word_form_with_merge(key, stress_variants, pos, feats)
    
    def _add_word_form_with_merge(self, key: str, stress_variants: List[int], 
                                   pos: List[str], feats: Dict[str, List[str]]) -> None:
        """
        Add word form with intelligent merging of identical stress patterns.
        
        If a form with same stress already exists, merge the morphological features
        by combining the lists (e.g., Case=["Nom"] + Case=["Acc"] → Case=["Nom", "Acc"]).
        
        Args:
            key: Word key
            stress_variants: Stress vowel indices
            pos: List of POS tags
            feats: Dict of feature lists
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
                    
                    return  # Merged successfully
        
        # No matching form found - create new one
        word_form = WordForm(
            stress_variants=stress_variants,
            pos=pos,
            feats=feats
        )
        self.words[key].append(word_form)
    
    def _parse_morphology(self, morphology: Dict) -> Tuple[List[str], Dict[str, List[str]]]:
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
    
    def lookup(self, word: str) -> Optional[List[WordForm]]:
        """
        Look up word by any form.
        
        Args:
            word: Word to look up (can be with or without stress marks)
        
        Returns:
            List of WordForm objects, or None if not found
        """
        key = StressExtractor.generate_key(word)
        return self.words.get(key)
    
    def has_unique_stress(self, word: str) -> bool:
        """
        Check if word has a single unique stress pattern.
        
        Returns True if word has only ONE stress pattern (unique/unambiguous).
        Returns False if word has MULTIPLE stress patterns (ambiguous/potential heteronym).
        
        Note: Dictionary may not cover all word forms, so this is not definitive.
        """
        forms = self.lookup(word)
        if not forms:
            return False
        
        # Check if only one unique stress pattern exists
        unique_stresses = {tuple(f.stress_variants) for f in forms}
        return len(unique_stresses) == 1
    
    def get_words_with_unique_stress(self) -> List[Tuple[str, List[WordForm]]]:
        """Get all words with multiple unique stress patterns (potential heteronyms)"""
        return [(key, forms) for key, forms in self.words.items() 
                if len(forms) > 1 and len({tuple(f.stress_variants) for f in forms}) > 1]
    
    def parse_file(self, file_path: Path) -> None:
        """
        Parse dictionary file.
        
        Format: One word per line, optional tab-separated metadata
        Example:
            обі´ді
            а́тлас	збірник карт
            атла́с	тканина
        """
        logger.info(f"Parsing dictionary from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Split by tab if metadata exists
                parts = line.split('\t')
                stressed_form = parts[0].strip()
                
                if not stressed_form:
                    continue
                
                # Add word
                self.add_word(stressed_form)
        
        logger.info(f"Parsing complete: {len(self.words)} unique words")
    
    def parse_trie(self, trie_path: Path) -> None:
        """
        Parse stress.trie file and merge data into dictionary.
        
        Args:
            trie_path: Path to stress.trie file
        """
        if not TRIE_AVAILABLE:
            logger.warning("Trie parser not available - skipping trie parsing")
            return
        
        logger.info(f"Parsing trie from {trie_path}")
        
        try:
            trie_parser = TrieParser(trie_path)
            
            # Track statistics
            total_processed = 0
            total_forms_added = 0
            
            # Iterate through all words in trie
            for word in trie_parser.trie.keys():
                # Normalize the key (lowercase, normalize apostrophe)
                normalized_word = StressExtractor.normalize_apostrophe_word(word)
                key = normalized_word.lower()
                
                # Get all forms for this word from trie
                forms = trie_parser.get_word_forms(word)
                
                for form in forms:
                    # Add each form to dictionary
                    self.add_word_form(
                        key=key,
                        stress_variants=form.stress_positions,
                        morphology=form.morphology if form.morphology else None
                    )
                    total_forms_added += 1
                
                total_processed += 1
                
                # Progress indicator every 100k words
                if total_processed % 100000 == 0:
                    logger.info(f"Processed {total_processed:,} words, added {total_forms_added:,} forms")
            
            logger.info(f"Trie parsing complete: processed {total_processed:,} words, added {total_forms_added:,} forms")
            logger.info(f"Total unique words in dictionary: {len(self.words):,}")
            
        except Exception as e:
            logger.error(f"Error parsing trie: {e}", exc_info=True)
            raise
    
    def get_statistics(self) -> Dict:
        """Get dictionary statistics"""
        unique_stress_words = self.get_words_with_unique_stress()
        total_forms = sum(len(forms) for forms in self.words.values())
        
        return {
            "total_unique_words": len(self.words),
            "total_forms": total_forms,
            "words_with_unique_stress": len(unique_stress_words),
            "avg_forms_per_word": round(total_forms / len(self.words), 2) if self.words else 0
        }
    
    def print_summary(self) -> None:
        """Print dictionary summary"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("DICTIONARY STATISTICS")
        print("="*80)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Print sample entries
        print("\n" + "="*80)
        print("SAMPLE ENTRIES")
        print("="*80)
        for i, (key, forms) in enumerate(list(self.words.items())[:5]):
            print(f"\n{i+1}. Key: '{key}'")
            print(f"   Forms: {len(forms)}")
            for j, form in enumerate(forms):
                print(f"     [{j}] stress_variants={form.stress_variants}, morphology={form.morphology}")
        
        # Print words with unique stress patterns
        unique_stress_words = self.get_words_with_unique_stress()
        if unique_stress_words:
            print("\n" + "="*80)
            print(f"WORDS WITH UNIQUE STRESS PATTERNS ({len(unique_stress_words)})")
            print("="*80)
            for key, forms in unique_stress_words[:10]:  # Show first 10
                stress_info = [f"stress={f.stress_variants}" for f in forms]
                print(f"'{key}' ({len(forms)} forms): {', '.join(stress_info)}")


def build_dictionary_from_file(file_path: Path) -> StressDictionary:
    """
    Build stress dictionary from file.
    
    Args:
        file_path: Path to dictionary file
    
    Returns:
        StressDictionary instance with loaded data
    """
    dictionary = StressDictionary()
    dictionary.parse_file(file_path)
    return dictionary
