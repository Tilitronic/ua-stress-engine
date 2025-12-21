#!/usr/bin/env python3
"""
Ukrainian Phonetic Transcription Service (DRAFT)

Converts Ukrainian text to IPA (International Phonetic Alphabet) transcription.
This is a draft implementation with basic rules.

Features:
- Cyrillic to IPA mapping
- Stress-based vowel reduction
- Basic consonant assimilation
- Syllabification

TODO for production:
- Context-dependent allophonic rules
- Voice assimilation rules
- Palatalization rules
- Prosody and intonation
"""

from typing import List, Optional
from logging import getLogger

logger = getLogger(__name__)


# Ukrainian vowels (Cyrillic)
UKRAINIAN_VOWELS = "аеєиіїоуюя"
UKRAINIAN_VOWELS_SET = set(UKRAINIAN_VOWELS + UKRAINIAN_VOWELS.upper())


class UkrainianPhoneticTranscriber:
    """
    Draft IPA transcriber for Ukrainian.
    
    Basic phoneme mapping with stress-based vowel reduction.
    """
    
    # Base phoneme mappings (stressed vowels)
    VOWEL_MAP = {
        'а': 'ɑ',   # open back unrounded
        'е': 'ɛ',   # open-mid front unrounded
        'є': 'jɛ',  # je
        'и': 'ɪ',   # near-close front unrounded
        'і': 'i',   # close front unrounded
        'ї': 'ji',  # ji
        'о': 'ɔ',   # open-mid back rounded
        'у': 'u',   # close back rounded
        'ю': 'ju',  # ju
        'я': 'jɑ',  # ja
    }
    
    # Reduced vowels (unstressed)
    VOWEL_REDUCED_MAP = {
        'а': 'ɐ',   # near-open central
        'е': 'e',   # close-mid front
        'є': 'je',
        'и': 'ɪ',   # same as stressed
        'і': 'i',   # same as stressed
        'ї': 'ji',
        'о': 'o',   # close-mid back rounded
        'у': 'u',   # same as stressed
        'ю': 'ju',
        'я': 'jɐ',
    }
    
    # Consonant mappings
    CONSONANT_MAP = {
        'б': 'b',
        'в': 'ʋ',   # labiodental approximant
        'г': 'ɦ',   # voiced glottal fricative
        'ґ': 'ɡ',   # voiced velar stop
        'д': 'd',
        'ж': 'ʒ',   # voiced postalveolar fricative
        'з': 'z',
        'й': 'j',   # palatal approximant
        'к': 'k',
        'л': 'l',
        'м': 'm',
        'н': 'n',
        'п': 'p',
        'р': 'r',   # alveolar trill
        'с': 's',
        'т': 't',
        'ф': 'f',
        'х': 'x',   # voiceless velar fricative
        'ц': 'ts',  # voiceless alveolar affricate
        'ч': 'tʃ',  # voiceless postalveolar affricate
        'ш': 'ʃ',   # voiceless postalveolar fricative
        'щ': 'ʃtʃ', # sh-ch
    }
    
    # Soft sign
    SOFT_SIGN = {'ь': 'ʲ'}  # palatalization marker
    
    def __init__(self):
        """Initialize phonetic transcriber."""
        logger.info("Initialized Ukrainian Phonetic Transcriber (DRAFT)")
    
    def transcribe(self, word: str, stress_position: Optional[int] = None) -> str:
        """
        Convert Ukrainian word to IPA transcription.
        
        Args:
            word: Ukrainian word (Cyrillic)
            stress_position: Index of stressed vowel (0-indexed)
        
        Returns:
            IPA transcription string
        
        Example:
            transcribe("замок", 0) → "ˈzɑmɔk"  (castle)
            transcribe("замок", 2) → "zɐˈmɔk"  (lock)
        """
        if not word:
            return ""
        
        word = word.lower()
        ipa = []
        vowel_count = 0
        stress_marker_added = False
        
        i = 0
        while i < len(word):
            char = word[i]
            
            # Check if this is a vowel
            is_vowel = char in self.VOWEL_MAP
            
            # Add stress marker before stressed vowel
            if is_vowel and stress_position is not None and vowel_count == stress_position:
                ipa.append('ˈ')  # Primary stress
                stress_marker_added = True
            
            # Transcribe character
            if is_vowel:
                # Vowel: stressed or reduced
                if stress_position is not None and vowel_count == stress_position:
                    ipa.append(self.VOWEL_MAP[char])
                else:
                    ipa.append(self.VOWEL_REDUCED_MAP[char])
                vowel_count += 1
            
            elif char in self.CONSONANT_MAP:
                # Consonant
                ipa.append(self.CONSONANT_MAP[char])
            
            elif char in self.SOFT_SIGN:
                # Soft sign: palatalization
                ipa.append(self.SOFT_SIGN[char])
            
            elif char == 'ʼ' or char == "'":
                # Apostrophe: glottal stop
                ipa.append('ʔ')
            
            else:
                # Unknown character: keep as-is
                ipa.append(char)
            
            i += 1
        
        return ''.join(ipa)
    
    def syllabify(self, ipa: str) -> List[str]:
        """
        Split IPA transcription into syllables (draft implementation).
        
        Basic rule: Split between consonants and vowels.
        TODO: Implement proper Ukrainian syllable rules.
        
        Args:
            ipa: IPA transcription
        
        Returns:
            List of syllable strings
        
        Example:
            syllabify("ˈzɑmɔk") → ["ˈzɑ", "mɔk"]
        """
        if not ipa:
            return []
        
        # Draft: Simple vowel-based splitting
        syllables = []
        current = []
        
        ipa_vowels = {'ɑ', 'ɐ', 'ɛ', 'e', 'ɪ', 'i', 'ɔ', 'o', 'u'}
        
        for i, char in enumerate(ipa):
            current.append(char)
            
            # If this is a vowel and next is a consonant, might be syllable boundary
            if char in ipa_vowels:
                # Look ahead to see if we should split
                if i + 1 < len(ipa) and ipa[i + 1] not in ipa_vowels and ipa[i + 1] != 'ˈ':
                    # Check if next consonant should start new syllable
                    if i + 2 < len(ipa) and ipa[i + 2] in ipa_vowels:
                        # Split after this vowel
                        syllables.append(''.join(current))
                        current = []
        
        # Add remaining
        if current:
            syllables.append(''.join(current))
        
        return syllables if syllables else [ipa]
    
    def get_vowel_positions(self, word: str) -> List[int]:
        """
        Get character positions of all vowels in word.
        
        Args:
            word: Ukrainian word
        
        Returns:
            List of vowel character indices
        """
        word = word.lower()
        return [i for i, char in enumerate(word) if char in UKRAINIAN_VOWELS_SET]


# Example usage
if __name__ == '__main__':
    transcriber = UkrainianPhoneticTranscriber()
    
    print("=" * 80)
    print("UKRAINIAN PHONETIC TRANSCRIPTION (DRAFT)")
    print("=" * 80)
    print()
    
    # Test words
    test_words = [
        ("замок", 0, "castle (stress on first syllable)"),
        ("замок", 2, "lock (stress on second syllable)"),
        ("слово", 1, "word"),
        ("мова", 0, "language"),
        ("дім", 0, "house"),
        ("п'ять", 0, "five"),
    ]
    
    print("Examples:")
    print("-" * 80)
    for word, stress, meaning in test_words:
        ipa = transcriber.transcribe(word, stress)
        syllables = transcriber.syllabify(ipa)
        print(f"{word:15} → [{ipa:20}] → {syllables}")
        print(f"{'':15}   {meaning}")
        print()
    
    print("=" * 80)
    print("NOTE: This is a DRAFT implementation!")
    print("TODO: Implement full allophonic and assimilation rules")
    print("=" * 80)
