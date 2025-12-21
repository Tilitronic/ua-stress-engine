#!/usr/bin/env python3
"""
TXT Dictionary Parser

Parses Ukrainian stress dictionary from text file format.
Handles stress mark extraction and word normalization.

Input Format:
    обі´ді
    а́тлас	збірник карт
    атла́с	тканина
    
Output: Dict[str, List[Tuple[List[int], Optional[str]]]]
    {
        "атлас": [
            ([0], "збірник карт"),  # Stress on vowel 0, optional definition
            ([1], "тканина")        # Stress on vowel 1, optional definition
        ]
    }
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from logging import getLogger

from src.nlp.utils.normalize_apostrophe import normalize_apostrophe

logger = getLogger(__name__)

# Ukrainian vowels
UKRAINIAN_VOWELS = set('аеєиіїоуюя')


def get_vowel_positions(word: str) -> List[int]:
    """Get positions of all vowels in a word."""
    return [i for i, char in enumerate(word.lower()) if char in UKRAINIAN_VOWELS]


def auto_stress_single_vowel(word: str, stress_positions: List[int]) -> List[int]:
    """
    Automatically add stress for single-vowel words if no stress is specified.
    
    Args:
        word: The word to check
        stress_positions: Current stress positions (may be empty)
    
    Returns:
        Stress positions (original or auto-detected for single vowel)
    """
    # If stress is already specified, return as-is
    if stress_positions:
        return stress_positions
    
    # Find vowel positions
    vowel_positions = get_vowel_positions(word)
    
    # If exactly one vowel and no stress specified, stress it
    if len(vowel_positions) == 1:
        return [vowel_positions[0]]
    
    # Otherwise return empty (no stress data)
    return stress_positions


# Unicode constants
APOSTROPHE_WRONG = "'"  # U+2019 ' - WRONG
APOSTROPHE_CORRECT = "ʼ"  # U+02BC ʼ - CORRECT

STRESS_MARK_ACUTE = "´"  # U+00B4 ´ - acute accent
STRESS_MARK_COMBINING = "́"  # U+0301 ́ - combining acute

UKRAINIAN_VOWELS = list("аеєиіїоуюяАЕЄИІЇОУЮЯ")


class TXTParser:
    """
    Parser for Ukrainian stress dictionary in text format.
    
    Responsibilities:
    - Extract stress positions from marked text
    - Normalize apostrophes and case
    - Generate searchable keys
    - Parse optional definitions/metadata
    """
    
    @staticmethod
    def normalize_apostrophe_word(word: str) -> str:
        """Normalize apostrophe using utility function"""
        return normalize_apostrophe(word)
    
    @staticmethod
    def extract_stress_indices(word: str) -> Tuple[List[int], str]:
        """
        Extract stress positions and clean text.
        
        Args:
            word: Word with stress marks (e.g., "обі´ді")
        
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
        
        Args:
            word: Word to normalize
        
        Returns:
            Normalized key for dictionary lookup
        """
        key = word.replace(STRESS_MARK_ACUTE, "")
        key = key.replace(STRESS_MARK_COMBINING, "")
        key = TXTParser.normalize_apostrophe_word(key)
        return key.lower()
    
    @staticmethod
    def parse_line(line: str) -> Optional[Tuple[str, List[int], Optional[str]]]:
        """
        Parse a single line from dictionary file.
        
        Args:
            line: Line from file (e.g., "обі´ді" or "а́тлас	збірник карт")
        
        Returns:
            (key, stress_indices, definition) or None if invalid
            - key: Normalized word key
            - stress_indices: List of vowel positions with stress
            - definition: Optional definition/metadata after tab
        """
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            return None
        
        # Split by tab if metadata exists
        parts = line.split('\t')
        stressed_form = parts[0].strip()
        definition = parts[1].strip() if len(parts) > 1 else None
        
        if not stressed_form:
            return None
        
        # Normalize apostrophe
        stressed_form = TXTParser.normalize_apostrophe_word(stressed_form)
        
        # Generate key
        key = TXTParser.generate_key(stressed_form)
        
        # Extract stress
        stress_indices, _ = TXTParser.extract_stress_indices(stressed_form)
        
        return key, stress_indices, definition
    
    def parse_file(self, file_path: Path, progress_callback=None) -> Dict[str, List[Tuple[List[int], Optional[str]]]]:
        """
        Parse entire dictionary file.
        
        Args:
            file_path: Path to dictionary file
            progress_callback: Optional callback function(current, total) for progress tracking
        
        Returns:
            Dictionary mapping normalized keys to list of (stress_indices, definition) tuples
            
            Example:
            {
                "атлас": [
                    ([0], "збірник карт"),
                    ([1], "тканина")
                ]
            }
        """
        logger.info(f"Parsing TXT dictionary from {file_path}")
        
        # Get total lines for progress tracking
        total_lines = 0
        if progress_callback:
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        
        result: Dict[str, List[Tuple[List[int], Optional[str]]]] = {}
        line_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                parsed = self.parse_line(line)
                
                if parsed is None:
                    continue
                
                key, stress_indices, definition = parsed
                
                # Auto-stress single-vowel words if no stress specified
                stress_indices = auto_stress_single_vowel(key, stress_indices)
                
                if key not in result:
                    result[key] = []
                
                result[key].append((stress_indices, definition))
                
                # Progress callback
                if progress_callback and total_lines > 0:
                    if line_count % 10000 == 0 or line_count == total_lines:
                        progress_callback(line_count, total_lines)
                
                # Log progress
                if line_num % 100000 == 0:
                    logger.info(f"Processed {line_num:,} lines, {len(result):,} unique words")
        
        logger.info(f"Parsing complete: {len(result):,} unique words")
        
        return result


def parse_txt_dictionary(file_path: Path, progress_callback=None) -> Dict[str, List[Tuple[List[int], Optional[str]]]]:
    """
    Convenience function to parse TXT dictionary file.
    
    Args:
        file_path: Path to dictionary file
        progress_callback: Optional callback function(current, total) for progress tracking
    
    Returns:
        Dictionary mapping normalized keys to list of (stress_indices, definition) tuples
    """
    parser = TXTParser()
    return parser.parse_file(file_path, progress_callback)
