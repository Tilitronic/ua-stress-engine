"""
Ukrainian Apostrophe Normalization Utility

This module provides functions to normalize Ukrainian apostrophes to the correct
Unicode character. In Ukrainian text, apostrophes should use U+02BC (modifier letter
apostrophe), not U+2019 (right single quotation mark).

Reference:
    - https://en.wikipedia.org/wiki/Apostrophe
    - https://uk.wikipedia.org/wiki/Апостроф
    - Unicode Standard: U+02BC ʼ vs U+2019 '

The issue:
    U+2019 (') - RIGHT SINGLE QUOTATION MARK - commonly used in typeset text
    U+02BC (ʼ) - MODIFIER LETTER APOSTROPHE - correct for Ukrainian language

Example:
    >>> normalize_apostrophe("п'ятниця")  # with U+2019
    "п'ятниця"  # returns with U+02BC
"""

from typing import Set


# Unicode characters for apostrophes
CORRECT_APOSTROPHE = '\u02BC'  # U+02BC ʼ (modifier letter apostrophe) - CORRECT FOR UKRAINIAN
WRONG_APOSTROPHES = {
    '\u2019',  # U+2019 ' (right single quotation mark) - WRONG but common in typeset
    '\u0027',  # U+0027 ' (ASCII apostrophe) - WRONG sometimes used in plain text
    '\u02BB',  # U+02BB ʻ (modifier letter turned comma) - WRONG variant
    '\u0060',  # U+0060 ` (grave accent) - WRONG sometimes confused
    '\u00B4',  # U+00B4 ´ (acute accent) - WRONG sometimes confused
}


def normalize_apostrophe(text: str) -> str:
    """
    Normalize all Ukrainian apostrophes to the correct Unicode character.
    
    Converts all incorrect apostrophe variations (U+2019, U+0027, etc.) to the
    correct Ukrainian apostrophe U+02BC.
    
    Args:
        text: Input Ukrainian text that may contain incorrect apostrophes
        
    Returns:
        Text with all apostrophes normalized to U+02BC ʼ (modifier letter apostrophe)
        
    Example:
        >>> normalize_apostrophe("п'ятниця")  # U+2019 ' variant
        "п'ятниця"  # U+02BC ʼ variant
        
        >>> normalize_apostrophe("Батько́ва хата")
        "Батько́ва хата"  # No apostrophes, returns unchanged
    """
    if not text:
        return text
    
    result = text
    for wrong_apostrophe in WRONG_APOSTROPHES:
        result = result.replace(wrong_apostrophe, CORRECT_APOSTROPHE)
    
    return result


def normalize_word(word: str) -> str:
    """
    Normalize apostrophes in a single word.
    
    Wrapper around normalize_apostrophe() for single word processing.
    
    Args:
        word: A Ukrainian word that may contain incorrect apostrophes
        
    Returns:
        Word with apostrophes normalized to U+02BC
        
    Example:
        >>> normalize_word("п'ятниця")
        "п'ятниця"
    """
    return normalize_apostrophe(word)


def has_wrong_apostrophe(text: str) -> bool:
    """
    Check if text contains incorrect apostrophes.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains any incorrect apostrophe variant
        
    Example:
        >>> has_wrong_apostrophe("п'ятниця")  # U+2019 '
        True
        
        >>> has_wrong_apostrophe("п'ятниця")  # U+02BC ʼ
        False
    """
    if not text:
        return False
    
    for wrong_apostrophe in WRONG_APOSTROPHES:
        if wrong_apostrophe in text:
            return True
    
    return False


def normalize_text(text: str, preserve_spaces: bool = True) -> str:
    """
    Normalize apostrophes in multi-word text.
    
    Args:
        text: Ukrainian text that may contain incorrect apostrophes
        preserve_spaces: If True, preserve original spacing (default: True)
        
    Returns:
        Text with all apostrophes normalized to U+02BC ʼ
        
    Example:
        >>> normalize_text("п'ятниця і п'ять")
        "п'ятниця і п'ять"
    """
    return normalize_apostrophe(text)


def get_apostrophe_info(text: str) -> dict:
    """
    Get detailed information about apostrophes in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with:
        - 'has_wrong': bool - True if incorrect apostrophes found
        - 'correct_count': int - number of correct apostrophes (U+02BC ʼ)
        - 'wrong_count': int - number of incorrect apostrophes
        - 'wrong_types': dict - breakdown by wrong apostrophe type (U+XXXX)
        
    Example:
        >>> info = get_apostrophe_info("п'ятниця і п'ять")
        >>> info['has_wrong']
        True
        >>> info['wrong_count']
        2
    """
    info = {
        'has_wrong': False,
        'correct_count': text.count(CORRECT_APOSTROPHE),
        'wrong_count': 0,
        'wrong_types': {}
    }
    
    for wrong_apostrophe in WRONG_APOSTROPHES:
        count = text.count(wrong_apostrophe)
        if count > 0:
            info['has_wrong'] = True
            info['wrong_count'] += count
            info['wrong_types'][f'U+{ord(wrong_apostrophe):04X}'] = count
    
    return info


# Example Ukrainian words that commonly have apostrophes
EXAMPLE_WORDS = {
    'п\'ятниця': 'Friday',
    'ав\'ярка': 'brickyard',
    'дв\'ясло': 'whip',
    'зв\'язок': 'connection',
    'сміх\'янка': 'laughing person',
    'смич\'ка': 'violin bow',
}


if __name__ == '__main__':
    """Example usage and testing"""
    
    print("=" * 80)
    print("UKRAINIAN APOSTROPHE NORMALIZATION UTILITY")
    print("=" * 80)
    print()
    
    print("UNICODE REFERENCE:")
    print("-" * 80)
    print(f"Correct:   U+02BC ʼ (modifier letter apostrophe)")
    print(f"Wrong:     U+2019 ' (right single quotation mark)")
    print(f"Wrong:     U+0027 ' (ASCII apostrophe)")
    print(f"Wrong:     U+02BB ʻ (modifier letter turned comma)")
    print(f"Wrong:     U+0060 ` (grave accent)")
    print(f"Wrong:     U+00B4 ´ (acute accent)")
    print()
    print()
    
    print("EXAMPLE WORDS WITH APOSTROPHES:")
    print("-" * 80)
    
    # Create test words with wrong apostrophes
    test_words = [
        ('п' + '\u2019' + 'ятниця', 'п' + '\u02BC' + 'ятниця'),  # wrong to correct
        ('зв' + '\u2019' + 'язок', 'зв' + '\u02BC' + 'язок'),    # wrong to correct
        ('ав' + '\u2019' + 'ярка', 'ав' + '\u02BC' + 'ярка'),    # wrong to correct
    ]
    
    for wrong_word, expected in test_words:
        result = normalize_apostrophe(wrong_word)
        status = "✓" if result == expected else "✗"
        print(f"\n{status} Input:    {wrong_word!r}")
        print(f"  Output:   {result!r}")
        print(f"  Expected: {expected!r}")
        
        # Show info
        info = get_apostrophe_info(wrong_word)
        print(f"  Info: has_wrong={info['has_wrong']}, correct={info['correct_count']}, wrong={info['wrong_count']}")
    
    print()
    print()
    print("BATCH NORMALIZATION:")
    print("-" * 80)
    
    text = 'п' + '\u2019' + 'ятниця, зв' + '\u2019' + 'язок, ав' + '\u2019' + 'ярка'
    print(f"Input text:  {text!r}")
    
    normalized = normalize_text(text)
    print(f"Output text: {normalized!r}")
    
    info = get_apostrophe_info(text)
    print(f"\nAnalysis before: {info}")
    
    info_after = get_apostrophe_info(normalized)
    print(f"Analysis after:  {info_after}")
