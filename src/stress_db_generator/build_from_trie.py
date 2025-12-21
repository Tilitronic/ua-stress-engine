#!/usr/bin/env python3
"""
Build StressDictionary from stress.trie

Populates in-memory dictionary with all data from marisa trie.
"""

from logging import getLogger
from pathlib import Path

from src.stress_db_generator.parser import StressDictionary, WordForm
from src.stress_db_generator.trie_parser import TrieParser

logger = getLogger(__name__)


def populate_dictionary_from_trie(dictionary: StressDictionary, trie_parser: TrieParser) -> None:
    """
    Populate existing StressDictionary with data from TrieParser.
    
    Args:
        dictionary: StressDictionary instance to populate
        trie_parser: TrieParser instance with loaded trie
    """
    # Get all words from trie
    all_words = trie_parser.get_all_words()
    
    logger.info(f"Populating dictionary with {len(all_words)} words...")
    
    for i, word in enumerate(all_words):
        # Progress logging
        if (i + 1) % 100000 == 0:
            logger.info(f"Processed {i + 1}/{len(all_words)} words...")
        
        # Parse entries from trie
        entries = trie_parser.parse_word(word)
        
        # Add each entry to dictionary
        for entry in entries:
            word_form = WordForm(
                stress_variants=entry.stress_positions,
                morphology=entry.morphology
            )
            
            # Add to dictionary if not duplicate
            if word not in dictionary.words:
                dictionary.words[word] = []
            
            # Check for duplicates
            is_duplicate = False
            for existing_form in dictionary.words[word]:
                if (existing_form.stress_variants == word_form.stress_variants and
                    existing_form.morphology == word_form.morphology):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                dictionary.words[word].append(word_form)
    
    logger.info("Dictionary population complete")


def build_dictionary_from_trie(trie_path: Path) -> StressDictionary:
    """
    Build StressDictionary from stress.trie file.
    
    Args:
        trie_path: Path to stress.trie file
    
    Returns:
        StressDictionary populated with all trie data
    """
    logger.info(f"Loading trie from {trie_path}...")
    trie_parser = TrieParser(trie_path)
    
    logger.info("Creating dictionary...")
    dictionary = StressDictionary()
    
    # Populate dictionary
    populate_dictionary_from_trie(dictionary, trie_parser)
    
    logger.info("Dictionary built successfully")
    return dictionary


def main():
    """Build and test dictionary from trie"""
    
    # Find trie file
    trie_path = Path(__file__).parent / "raw_data" / "stress.trie"
    
    if not trie_path.exists():
        print(f"Error: {trie_path} not found")
        return
    
    # Build dictionary
    print("Building dictionary from trie...")
    dictionary = build_dictionary_from_trie(trie_path)
    
    # Print statistics
    dictionary.print_summary()
    
    # Test lookups
    print("\n" + "="*80)
    print("TEST LOOKUPS WITH MORPHOLOGY")
    print("="*80)
    
    test_words = ["атлас", "україна", "котик", "блохи"]
    
    for word in test_words:
        forms = dictionary.lookup(word)
        if forms:
            print(f"\n✓ Found '{word}':")
            print(f"  Forms: {len(forms)}")
            print(f"  Has unique stress: {dictionary.has_unique_stress(word)}")
            for i, form in enumerate(forms[:3]):  # Show first 3
                print(f"    [{i}] stress={form.stress_variants}")
                if form.morphology:
                    print(f"        {form.morphology}")
        else:
            print(f"\n✗ NOT found: '{word}'")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
