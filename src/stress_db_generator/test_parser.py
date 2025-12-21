#!/usr/bin/env python3
"""
Test script for Ukrainian Stress Dictionary Builder

Demonstrates usage of parser.py
"""

from pathlib import Path

from src.stress_db_generator.parser import build_dictionary_from_file


def main():
    """Test dictionary builder with sample file"""
    
    # Build dictionary from sample file
    sample_file = Path(__file__).parent / "sample_stress_dict.txt"
    
    if not sample_file.exists():
        print(f"Error: {sample_file} not found")
        return
    
    print(f"Building dictionary from {sample_file.name}...")
    dictionary = build_dictionary_from_file(sample_file)
    
    # Print summary
    dictionary.print_summary()
    
    # Test lookups
    print("\n" + "="*80)
    print("TEST LOOKUPS")
    print("="*80)
    
    test_words = ["обіді", "атлас", "блохи", "україна", "катамаран", "коридор"]
    for word in test_words:
        forms = dictionary.lookup(word)
        if forms:
            print(f"\n✓ Found '{word}':")
            print(f"  Forms: {len(forms)}")
            print(f"  Has unique stress: {dictionary.has_unique_stress(word)}")
            for i, form in enumerate(forms):
                print(f"    [{i}] stress_variants={form.stress_variants}")
        else:
            print(f"\n✗ NOT found: '{word}'")
    
    # Show dictionary structure as dict
    print("\n" + "="*80)
    print("RAW DICTIONARY STRUCTURE (first 3 entries)")
    print("="*80)
    for i, (key, forms) in enumerate(list(dictionary.words.items())[:3]):
        print(f"\n'{key}': [")
        for form in forms:
            print(f"  {form.to_dict()},")
        print("]")


if __name__ == "__main__":
    main()
