#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API*", category=UserWarning)
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional

from src.data_management.transform.data_unifier import LinguisticEntry, WordForm, UPOS

from src.utils.normalize_apostrophe import normalize_apostrophe
from lemmatizer.lemmatizer import Lemmatizer
from src.data_management.sources.txt_ua_stresses.stress_db_file_manager import ensure_latest_db_file

import logging
"""

Ukrainian TXT Stress Dictionary Parser

Parses Ukrainian stress dictionary from text format and converts to unified format.

Input Format:
    замо́к
    за́мок
    по́ми́лка
    п'я́тниця
    клаксо́нив
    кла́сико-романти́чний
    клуб-чита́льня
"""



# NOTE: Do not configure logging handlers or levels here.
# Logging is managed by the main process for concurrency compatibility and clean progress bar output.
logger = logging.getLogger("TXT_Stress_Parser")

TEST_MODE = False

DB_PATH = "src/data_management/sources/txt_ua_stresses/ua_word_stress_dictionary.txt"
TEST_DB_PATH = "src/data_management/sources/txt_ua_stresses/sample_stress_dict.txt"



# Unicode stress marks
STRESS_MARK_ACUTE = "´"  # U+00B4 ´ - acute accent
STRESS_MARK_COMBINING = "́"  # U+0301 ́ - combining acute

# Ukrainian vowels
UKRAINIAN_VOWELS = set('аеєиіїоуюяАЕЄИІЇОУЮЯ')

def get_db_path() -> Path:
    if TEST_MODE:
        return Path(TEST_DB_PATH)
    return Path(DB_PATH)

PATH = get_db_path()

lemmatizer = Lemmatizer(use_gpu=False)

def get_lemma(word: str) -> str:
    return lemmatizer.get_lemma(word)



def split_words(input_path: str) -> List[str]:
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    split_words = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Split by tab (ignore definitions)
        word_part = line.split('\t', 1)[0]
        # Split by space and hyphen
        for token in word_part.replace('-', ' ').split():
            split_words.append(token)
    return split_words


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





def extract_stress_indices_fn(word: str) -> List[int]:
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

def extract_stress_indices(word: str) -> List[int]:
        """
        Extract stress positions and clean text.
        
        Args:
            word: Word with stress marks (e.g., "обі´ді")
        
        Returns:
            (stress_indices)
            stress_indices: List of 0-based vowel indices where stress occurs

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
        
        return vowel_indices

def strip_stress_marks(word: str) -> str:
    """Remove stress marks from a word."""
    return word.replace(STRESS_MARK_ACUTE, '').replace(STRESS_MARK_COMBINING, '')

def clean_up_word(word: str) -> str:
    """Normalize apostrophe and strip stress marks."""
    no_stress_word = strip_stress_marks(word)
    cleaned_word = normalize_apostrophe(no_stress_word)
    return cleaned_word


def parse_txt_to_unified_dict(input_path: Optional[str] = None, show_progress: bool = False, progress_callback=None) -> Dict[str, LinguisticEntry]:
    if input_path is None:
        input_path = str(get_db_path())
    unified_data = {}
    total_tokens = 0
    skipped_multisyllable = 0
    word_forms_count = 0
    start_time = time.time()
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read input file '{input_path}': {e}")
        return {}
    filtered_lines = [l for l in all_lines if l.strip() and not l.strip().startswith('#')]
    total_lines = len(filtered_lines)
    lines_iter = tqdm(filtered_lines, desc='[Parsing]', unit='line') if show_progress else filtered_lines
    unified_data = {}
    for idx, line in enumerate(lines_iter):
        try:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for token in line.replace('-', ' ').split():
                total_tokens += 1
                try:
                    clean_word_form = clean_up_word(token)
                except Exception as e:
                    logger.warning(f"Failed to clean word '{token}': {e}")
                    continue
                try:
                    lemma = get_lemma(clean_word_form)
                except Exception as e:
                    logger.warning(f"Failed to lemmatize '{clean_word_form}': {e}")
                    lemma = clean_word_form
                try:
                    stress_indices = extract_stress_indices(token)
                except Exception as e:
                    logger.warning(f"Failed to extract stress indices for '{token}': {e}")
                    continue
                # Handle single-syllable words: if no stress, assign stress to the only vowel
                if not stress_indices:
                    vowel_positions = get_vowel_positions(token)
                    if len(vowel_positions) == 1:
                        stress_indices = [0]  # Only one vowel, index 0
                    else:
                        # Multi-syllable word with no stress marker: skip
                        skipped_multisyllable += 1
                        continue
                pos = UPOS.X  # Unknown, since not provided
                feats = {}
                try:
                    word_form = WordForm(
                        stress_indices=stress_indices,
                        pos=pos,
                        feats=feats,
                        lemma=lemma,
                        examples=[],
                        form=clean_word_form,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create WordForm for '{token}': {e}")
                    continue
                # Merge by lemma and stress_indices only
                if lemma not in unified_data:
                    unified_data[lemma] = LinguisticEntry(word=lemma, forms=[], possible_stress_indices=[])
                # Only add unique WordForms by stress_indices
                if not any(wf.stress_indices == word_form.stress_indices for wf in unified_data[lemma].forms):
                    unified_data[lemma].forms.append(word_form)
                    word_forms_count += 1
        except Exception as e:
            logger.error(f"Error processing line: {line}\n{e}")
            continue
        # Progress callback every 100 lines
        if progress_callback and (idx % 100 == 0 or idx == total_lines - 1):
            progress_callback(idx + 1, total_lines)

    # After all forms are collected, set possible_stress_indices as unique stress arrays for each lemma
    for lemma, entry in unified_data.items():
        unique_stress_arrays = []
        for wf in entry.forms:
            sorted_indices = tuple(sorted(wf.stress_indices))
            if sorted_indices not in [tuple(sorted(arr)) for arr in unique_stress_arrays]:
                unique_stress_arrays.append(list(sorted_indices))
        entry.possible_stress_indices = unique_stress_arrays

    elapsed = time.time() - start_time
    stats = {
        "total_lines": total_lines,
        "total_tokens": total_tokens,
        "unique_lemmas": len(unified_data),
        "word_forms_created": word_forms_count,
        "skipped_multisyllable": skipped_multisyllable,
        "elapsed": elapsed,
    }
    return unified_data, stats

# Example usage:
def main():
    logger.info("Ukrainian TXT Stress Dictionary Parser")
    if not TEST_MODE:
        ensure_latest_db_file(str(get_db_path()))
    try:
        unified_data, stats = parse_txt_to_unified_dict(str(PATH), show_progress=True)
    except Exception as e:
        logger.error(f"Critical error during parsing: {e}")
        return

    # Query and print raw structure for 'замок' and 'блоха'
    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    for key in ["замок", "блоха", "клаксон", "помилка", "обід"]:
        logger.info(f"Entry for lemma: '{key}'")
        entry = unified_data.get(key)
        if not entry:
            logger.warning("  Not found in dictionary.")
            continue
        # Pretty-print the entry to stdout with color
        print(f"\033[1;36m{key}\033[0m:")
        print(f"\033[0;37m{pp.pformat(entry.model_dump())}\033[0m")
    return unified_data

if __name__ == "__main__":
    main()

