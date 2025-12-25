#!/usr/bin/env python3
"""
Ukrainian TXT Stress Dictionary Parser

Parses Ukrainian stress dictionary from text format and converts to unified format.

Input Format:
    обі´ді
    а́тлас	збірник карт
    атла́с	тканина

"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from logging import getLogger
import spacy

from src.data_management.transform.data_unifier import LinguisticEntry, WordForm, UPOS

from src.utils.normalize_apostrophe import normalize_apostrophe
from src.lemmatizer import UkrLinguisticsService


lemmatizer = UkrLinguisticsService(use_gpu=False)

def get_lemma(word: str) -> str:
    return lemmatizer.get_lemma(word)


logger = getLogger(__name__)

# Unicode stress marks
STRESS_MARK_ACUTE = "´"  # U+00B4 ´ - acute accent
STRESS_MARK_COMBINING = "́"  # U+0301 ́ - combining acute

# Ukrainian vowels
UKRAINIAN_VOWELS = set('аеєиіїоуюяАЕЄИІЇОУЮЯ')


def extract_stress_indices(word: str) -> List[int]:
    """
    Extract stress positions from a word with stress marks.

    Args:
        word: Word with stress marks (e.g., "за́мок")

    Returns:
        List of 0-based vowel indices where stress occurs
    """
    # Remove stress marks and track stressed positions
    clean_word = ""
    stressed_positions = set()

    i = 0
    while i < len(word):
        char = word[i]
        if char in (STRESS_MARK_ACUTE, STRESS_MARK_COMBINING):
            # Previous character was stressed
            if clean_word:
                stressed_positions.add(len(clean_word) - 1)
        else:
            clean_word += char
        i += 1

    # Convert character positions to vowel indices
    vowel_indices = []
    vowel_count = 0
    for i, char in enumerate(clean_word.lower()):
        if char in UKRAINIAN_VOWELS:
            if i in stressed_positions:
                vowel_indices.append(vowel_count)
            vowel_count += 1

    return vowel_indices


class UkrainianStressTXTParser:
    """
    Parser for Ukrainian stress dictionary in text format.

    Converts individual stressed word forms to unified LinguisticEntry format
    using spaCy lemmatization to determine lemma keys.
    """

    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line from the stress dictionary.

        Args:
            line: Line like "за́мок" or "а́тлас	збірник карт"

        Returns:
            Dict with parsed data or None if invalid
        """
        line = line.strip()
        if not line or line.startswith('#'):
            return None

        # Split by tab for optional definition
        parts = line.split('\t', 1)
        stressed_form = parts[0].strip()
        definition = parts[1].strip() if len(parts) > 1 else None

        if not stressed_form:
            return None

        # Extract stress indices
        stress_indices = extract_stress_indices(stressed_form)

        # Get lemma using spaCy
        lemma = self.lemmatize_word(stressed_form)

        # Clean form (no stress marks)
        clean_form = stressed_form.replace(STRESS_MARK_ACUTE, "").replace(STRESS_MARK_COMBINING, "")

        return {
            'stressed_form': stressed_form,
            'clean_form': clean_form,
            'lemma': lemma,
            'stress_indices': stress_indices,
            'definition': definition
        }

    def parse_file(self, file_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse entire stress dictionary file.

        Args:
            file_path: Path to the txt file

        Returns:
            Dict mapping lemmas to list of form data
        """
        logger.info(f"Parsing stress dictionary: {file_path}")

        result: Dict[str, List[Dict[str, Any]]] = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parsed = self.parse_line(line)
                if parsed is None:
                    continue

                lemma = parsed['lemma']
                if lemma not in result:
                    result[lemma] = []
                result[lemma].append(parsed)

        logger.info(f"Parsed {len(result)} unique lemmas from {file_path}")
        return result

    def to_unified_format(self, parsed_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, LinguisticEntry]:
        """
        Convert parsed data to unified LinguisticEntry format.

        Args:
            parsed_data: Output from parse_file()

        Returns:
            Dict mapping lemmas to LinguisticEntry objects
        """
        result: Dict[str, LinguisticEntry] = {}

        for lemma, forms_data in parsed_data.items():
            word_forms = []

            for form_data in forms_data:
                # Create WordForm
                word_form = WordForm(
                    stress_indices=form_data['stress_indices'],
                    pos=UPOS.NOUN,  # Most stress dict entries are nouns, could be enhanced
                    lemma=lemma,
                    main_definition=form_data['definition'],
                    # Store original stressed form as example or in meta
                )

                # Add metadata
                word_form.meta = {
                    'source': 'txt_stress_dict',
                    'stressed_form': form_data['stressed_form'],
                    'clean_form': form_data['clean_form']
                }

                word_forms.append(word_form)

            # Create LinguisticEntry
            entry = LinguisticEntry(
                word=lemma,
                forms=word_forms
            )

            result[lemma] = entry

        return result


def build_stress_dict(file_path: Path) -> Dict[str, LinguisticEntry]:
    """
    Convenience function to parse stress dictionary and return unified format.

    Args:
        file_path: Path to stress dictionary txt file

    Returns:
        Unified dictionary with lemmas as keys
    """
    parser = UkrainianStressTXTParser()
    parsed_data = parser.parse_file(file_path)
    return parser.to_unified_format(parsed_data)
