#!/usr/bin/env python3
"""
Stress Trie Parser

Parses marisa_trie format stress.trie file and extracts:
- Stress positions (vowel indices)
- Morphological features (Case, Number, Gender, etc.)
- POS tags (NOUN, VERB, ADJ, etc.)

Data Source:
    Repository: https://github.com/lang-uk/ukrainian-word-stress
    License: MIT License, Copyright (c) 2022 lang-uk
    Format: marisa_trie.BytesTrie (see raw_data/DICTIONARY_FORMAT.md)

Based on ukrainian-word-stress library format.
See raw_data/DATA_ATTRIBUTION.md for complete licensing information.
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import marisa_trie
except ImportError:
    raise ImportError(
        "marisa-trie is required. Install with: pip install marisa-trie"
    )

logger = getLogger(__name__)


# Ukrainian vowels for position conversion
UKRAINIAN_VOWELS = "уеіїаояиюєУЕІАОЯИЮЄЇ"

# Trie format constants
POS_SEPARATOR = b'\xFE'  # Separates stress from morphology
RECORD_SEPARATOR = b'\xFF'  # Separates multiple records

# Tag decompression mapping 
TAG_BY_BYTE = {
    b'\x11': "Number=Sing",
    b'\x12': "Number=Plur",
    b'\x20': "Case=Nom",
    b'\x21': "Case=Gen",
    b'\x22': "Case=Dat",
    b'\x23': "Case=Acc",
    b'\x24': "Case=Ins",
    b'\x25': "Case=Loc",
    b'\x26': "Case=Voc",
    b'\x30': "Gender=Neut",
    b'\x31': "Gender=Masc",
    b'\x32': "Gender=Fem",
    b'\x41': "VerbForm=Inf",
    b'\x42': "VerbForm=Conv",
    b'\x50': "Person=0",
    b'\x61': "upos=NOUN",
    b'\x62': "upos=ADJ",
    b'\x63': "upos=INTJ",
    b'\x64': "upos=CCONJ",
    b'\x65': "upos=PART",
    b'\x66': "upos=PRON",
    b'\x67': "upos=VERB",
    b'\x68': "upos=PROPN",
    b'\x69': "upos=ADV",
    b'\x6A': "upos=NOUN",
    b'\x6B': "upos=NUM",
    b'\x6C': "upos=ADP",
}


@dataclass
class TrieEntry:
    """Single entry from trie (one word form with specific morphology)"""
    stress_positions: List[int]  # Vowel indices where stress occurs (converted from char positions)
    morphology: Optional[Dict[str, str]] = None  # Parsed morphology features
    
    def to_dict(self) -> Dict:
        return {
            "stress_variants": self.stress_positions,
            "morphology": self.morphology
        }


class TrieParser:
    """
    Parser for stress.trie marisa_trie format.
    
    Extracts stress and morphological information in spaCy-compatible format.
    """
    
    def __init__(self, trie_path: Path):
        """
        Initialize parser with trie file.
        
        Args:
            trie_path: Path to stress.trie file
        """
        self.trie_path = trie_path
        self.trie = marisa_trie.BytesTrie()
        self.trie.load(str(trie_path))
        logger.info(f"Loaded trie from {trie_path} ({len(self.trie)} entries)")
    
    VOWELS = "уеіїаояиюєУЕІАОЯИЮЄЇ"

    def _char_positions_to_vowel_indices(self, word: str, char_positions: List[int]) -> List[int]:
        """
        Converts character positions (from trie) to vowel indices (for our dictionary).
        
        The trie stores insertion positions for the combining accent mark.
        Reference code: s[:position] + accent + s[position:]
        This places the accent AFTER the character at (position - 1).
        
        Since accent marks in Ukrainian come AFTER vowels, the stressed vowel
        is the character at position (stored_position - 1).
        
        Examples from testing:
        - 'мама' with stored [2]: accent after pos 1 ('а') -> vowel index 0
        - 'вода' with stored [4]: accent after pos 3 ('а') -> vowel index 1  
        - 'поперевалювано' with stored [8]: accent after pos 7 ('а') -> vowel index 3
        """
        vowel_indices = []
        
        for char_pos in char_positions:
            # The stressed character is at position (char_pos - 1)
            if char_pos > 0:
                stressed_char_pos = char_pos - 1
                
                # Count how many vowels come before this position
                vowel_index = sum(1 for i in range(stressed_char_pos) 
                                if word[i].lower() in self.VOWELS.lower())
                
                # Verify this position is actually a vowel
                if stressed_char_pos < len(word) and word[stressed_char_pos].lower() in self.VOWELS.lower():
                    vowel_indices.append(vowel_index)
        
        return sorted(list(set(vowel_indices)))
    
    def parse_word(self, word: str) -> List[TrieEntry]:
        """
        Parse all entries for a word.
        
        Args:
            word: Normalized word (lowercase, no stress marks)
        
        Returns:
            List of TrieEntry objects with stress and morphology
        """
        if word not in self.trie:
            return []
        
        # Get raw bytes from trie (marisa_trie returns a list with one element)
        values = self.trie[word]
        
        if len(values) != 1:
            logger.warning(f"Expected 1 value for '{word}', got {len(values)}")
            return []
        
        raw_value = values[0]
        
        # Parse the value
        entries = self._parse_trie_value(raw_value, word)
        
        return entries
    
    def get_word_forms(self, word: str) -> List[TrieEntry]:
        """
        Alias for parse_word to maintain compatibility.
        """
        return self.parse_word(word)
    
    def _parse_trie_value(self, value: bytes, word: str) -> List[TrieEntry]:
        """
        Parses the byte value from trie following the exact logic from the reference implementation.
        
        Based on ukrainian-word-stress library's _parse_dictionary_value function.
        """
        POS_SEP = b'\xFE'
        REC_SEP = b'\xFF'
        
        entries = []
        
        if REC_SEP not in value:
            # Simple case: single item, all bytes are accent positions (character indices)
            char_positions = [int(b) for b in value if b != 0]
            vowel_indices = self._char_positions_to_vowel_indices(word, char_positions)
            if vowel_indices:
                entries.append(TrieEntry(stress_positions=vowel_indices, morphology={}))
        else:
            # Complex case: multiple records with morphological tags
            items = value.split(REC_SEP)
            for item in items:
                if item:
                    # Split into accent positions and tags
                    accents_bytes, _, tags_bytes = item.partition(POS_SEP)
                    char_positions = [int(b) for b in accents_bytes if b != 0]
                    vowel_indices = self._char_positions_to_vowel_indices(word, char_positions)
                    
                    # Decompress tags
                    morphology = self._decompress_tags(tags_bytes)
                    
                    if vowel_indices:
                        entries.append(TrieEntry(stress_positions=vowel_indices, morphology=morphology))
        
        return entries
    
    def _decompress_tags(self, tags_bytes: bytes) -> Dict[str, str]:
        """
        Decompress tag bytes to dictionary following the reference implementation.
        """
        tags = {}
        for byte in tags_bytes:
            tag_str = TAG_BY_BYTE.get(bytes([byte]))
            if tag_str and '=' in tag_str:
                key, value = tag_str.split('=', 1)
                tags[key] = value
        return tags


def main():
    """
    Main function to demonstrate the TrieParser.
    """
    # The path to the stress.trie file
    trie_path = Path(__file__).parent / "raw_data" / "stress.trie"
    
    # Create a parser instance
    parser = TrieParser(trie_path)

    # First, let's check what's actually in the trie
    print("--- Checking Trie Contents ---")
    sample_count = 0
    for word in parser.trie.keys():
        print(f"Sample word: '{word}'")
        sample_count += 1
        if sample_count >= 5:
            break
    
    # --- Test Cases ---
    test_words = ["замок", "заводи", "блохи", "поперек", "поперекові", "атлас"]
    
    print("\n--- Trie Parser Test ---")
    for word in test_words:
        # Try both lowercase and as-is
        forms = parser.get_word_forms(word)
        if not forms and word != word.lower():
            forms = parser.get_word_forms(word.lower())
        
        if forms:
            print(f"\nWord: '{word}'")
            # Count vowels in word for validation
            vowel_count = sum(1 for c in word.lower() if c in parser.VOWELS.lower())
            print(f"  Total vowels in word: {vowel_count}")
            for i, form in enumerate(forms, 1):
                print(f"  - Form {i}:")
                print(f"    - Stress (vowel indices): {form.stress_positions}")
                print(f"    - Morphology: {form.morphology}")
        else:
            print(f"\nWord: '{word}' - Not found in trie")
    
    # Manual test with raw data
    print("\n--- Manual Raw Data Check ---")
    if "поперек" in parser.trie:
        raw_value = parser.trie["поперек"][0]
        print(f"Raw value for 'поперек': {raw_value.hex()}")
        print(f"Raw bytes: {list(raw_value)}")
    
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    main()
