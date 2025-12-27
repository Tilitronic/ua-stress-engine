from src.data_management.sources.trie_ua_stresses.stress_db_file_manager import ensure_latest_db_file, DEFAULT_LOCAL_PATH
# Standalone function to convert char positions to vowel indices


#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API*", category=UserWarning)

import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional

from src.data_management.transform.data_unifier import LinguisticEntry, WordForm, UPOS

from src.utils.normalize_apostrophe import normalize_apostrophe
from src.lemmatizer.lemmatizer import Lemmatizer
import logging
from pydantic import BaseModel, Field, ConfigDict
from src.data_management.transform.data_unifier import UDFeatKey, UPOS

try:
    import marisa_trie
except ImportError:
    raise ImportError(
        "marisa-trie is required. Install with: pip install marisa-trie"
    )

""""
Description

"""


# NOTE: Do not configure logging handlers or levels here.
# Logging is managed by the main process for concurrency compatibility and clean progress bar output.
logger = logging.getLogger("TRIE_Stress_Parser")

DB_PATH = "src/data_management/sources/trie_ua_stresses/stress.trie"

UKRAINIAN_VOWELS = "уеіїаояиюєУЕІАОЯИЮЄЇ"

# Trie format constants
POS_SEPARATOR = b'\xFE'  # Separates stress from morphology
RECORD_SEPARATOR = b'\xFF'  # Separates multiple records

# Tag decompression mapping using UDFeatKey and UPOS enums for type safety
TAG_BY_BYTE = {
    b'\x11': (UDFeatKey.Number, "Sing"),
    b'\x12': (UDFeatKey.Number, "Plur"),
    b'\x20': (UDFeatKey.Case, "Nom"),
    b'\x21': (UDFeatKey.Case, "Gen"),
    b'\x22': (UDFeatKey.Case, "Dat"),
    b'\x23': (UDFeatKey.Case, "Acc"),
    b'\x24': (UDFeatKey.Case, "Ins"),
    b'\x25': (UDFeatKey.Case, "Loc"),
    b'\x26': (UDFeatKey.Case, "Voc"),
    b'\x30': (UDFeatKey.Gender, "Neut"),
    b'\x31': (UDFeatKey.Gender, "Masc"),
    b'\x32': (UDFeatKey.Gender, "Fem"),
    b'\x41': (UDFeatKey.VerbForm, "Inf"),
    b'\x42': (UDFeatKey.VerbForm, "Conv"),
    b'\x50': (UDFeatKey.Person, "0"),
    b'\x61': ("upos", UPOS.NOUN),
    b'\x62': ("upos", UPOS.ADJ),
    b'\x63': ("upos", UPOS.INTJ),
    b'\x64': ("upos", UPOS.CCONJ),
    b'\x65': ("upos", UPOS.PART),
    b'\x66': ("upos", UPOS.PRON),
    b'\x67': ("upos", UPOS.VERB),
    b'\x68': ("upos", UPOS.PROPN),
    b'\x69': ("upos", UPOS.ADV),
    b'\x6A': ("upos", UPOS.NOUN),
    b'\x6B': ("upos", UPOS.NUM),
    b'\x6C': ("upos", UPOS.ADP),
}



class TrieEntry(BaseModel):
    """
    Single entry from trie (one word form with specific morphology).
    - stress_indices: Vowel indices where stress occurs (converted from char positions)
    - feats: Morphological features (UD-compliant, if available)
    """
    stress_indices: List[int] = Field(
        ...,
        description="Vowel indices where stress occurs (converted from char positions)",
        examples=[[0], [1], [0, 1]]
    )
    feats: Dict[str, str] = Field(
        default_factory=dict,
        description="Morphological features (UD-compliant, if available)"
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

def extract_stress_indices() -> List[int]:
    return []   

lemmatizer = Lemmatizer(use_gpu=False)
def get_lemma(word: str) -> str:
    return lemmatizer.get_lemma(word)

def char_positions_to_vowel_indices(word: str, char_positions: List[int]) -> List[int]:
    """
    Converts character positions (from trie) to vowel indices (for our dictionary).
    The trie stores insertion positions for the combining accent mark.
    Reference code: s[:position] + accent + s[position:]
    This places the accent AFTER the character at (position - 1).
    Since accent marks in Ukrainian come AFTER vowels, the stressed vowel
    is the character at position (stored_position - 1).
    """
    vowel_indices = []
    VOWELS = UKRAINIAN_VOWELS
    for char_pos in char_positions:
        if char_pos > 0:
            stressed_char_pos = char_pos - 1
            vowel_index = sum(1 for i in range(stressed_char_pos)
                             if word[i].lower() in VOWELS.lower())
            if stressed_char_pos < len(word) and word[stressed_char_pos].lower() in VOWELS.lower():
                vowel_indices.append(vowel_index)
    return sorted(list(set(vowel_indices)))

def parse_trie_to_unified_dict(input_path: Optional[str] = None, show_progress: bool = False, progress_callback=None) -> Dict[str, LinguisticEntry]:
    """
    Build a unified dictionary: lemma -> LinguisticEntry(forms=[WordForm, ...]) using lemmatizer for grouping.
    """
    if input_path is None:
        input_path = DB_PATH
    trie_path = Path(input_path)
    # Ensure the trie DB file is present and up-to-date
    ensure_latest_db_file(str(trie_path))
    trie = marisa_trie.BytesTrie()
    trie.load(str(trie_path))
    keys = list(trie.keys())
    iterator = tqdm(keys, desc="Parsing trie", unit="word") if show_progress else keys

    def char_positions_to_vowel_indices(word: str, char_positions: List[int]) -> List[int]:
        VOWELS = UKRAINIAN_VOWELS
        vowel_indices = []
        for char_pos in char_positions:
            if char_pos > 0:
                stressed_char_pos = char_pos - 1
                vowel_index = sum(1 for i in range(stressed_char_pos) if word[i].lower() in VOWELS.lower())
                if stressed_char_pos < len(word) and word[stressed_char_pos].lower() in VOWELS.lower():
                    vowel_indices.append(vowel_index)
        return sorted(set(vowel_indices))

    def decompress_tags(tags_bytes: bytes) -> Dict[str, str]:
        tags = {}
        for byte in tags_bytes:
            tag = TAG_BY_BYTE.get(bytes([byte]))
            if tag and isinstance(tag, tuple):
                key, value = tag
                if hasattr(key, 'value'):
                    key_str = key.value
                else:
                    key_str = str(key)
                if hasattr(value, 'value'):
                    value_str = value.value
                else:
                    value_str = str(value)
                tags[key_str] = value_str
            elif tag and isinstance(tag, str) and '=' in tag:
                key_str, value_str = tag.split('=', 1)
                tags[key_str] = value_str
        return tags

    unified_data: Dict[str, LinguisticEntry] = {}
    total_tokens = 0
    word_forms_count = 0
    start_time = time.time()

    for idx, word in enumerate(iterator):
        norm_word = normalize_apostrophe(word).lower()
        lemma = get_lemma(norm_word)
        values = trie[word]
        if not values or len(values) != 1:
            continue
        total_tokens += 1
        raw_value = values[0]
        forms = []
        if RECORD_SEPARATOR not in raw_value:
            char_positions = [int(b) for b in raw_value if b != 0]
            vowel_indices = char_positions_to_vowel_indices(word, char_positions)
            if vowel_indices:
                forms.append((vowel_indices, {}))
        else:
            items = raw_value.split(RECORD_SEPARATOR)
            for item in items:
                if item:
                    accents_bytes, _, tags_bytes = item.partition(POS_SEPARATOR)
                    char_positions = [int(b) for b in accents_bytes if b != 0]
                    vowel_indices = char_positions_to_vowel_indices(word, char_positions)
                    morphology = decompress_tags(tags_bytes)
                    if vowel_indices:
                        forms.append((vowel_indices, morphology))

        # Add forms to the correct lemma entry in unified_data
        if lemma not in unified_data:
            unified_data[lemma] = LinguisticEntry(
                word=lemma,
                forms=[],
                possible_stress_indices=[],
                meta={}
            )
        entry = unified_data[lemma]
        for stress_indices, feats in forms:
            # Ensure unique stress arrays (order-insensitive)
            sorted_indices = tuple(sorted(stress_indices))
            if sorted_indices not in [tuple(sorted(arr)) for arr in entry.possible_stress_indices]:
                entry.possible_stress_indices.append(list(sorted_indices))
            pos = UPOS(feats.get('upos')) if 'upos' in feats else UPOS.X
            feats_clean = {k: v for k, v in feats.items() if k != 'upos'}
            # Only add unique WordForms by stress_indices and form
            if not any((wf.stress_indices == stress_indices and wf.form == norm_word) for wf in entry.forms):
                entry.forms.append(WordForm(
                    form=norm_word,
                    stress_indices=stress_indices,
                    pos=pos,
                    feats=feats_clean,
                    lemma=lemma,
                    main_definition=None,
                    alt_definitions=None,
                    translations=None,
                    etymology_templates=None,
                    etymology_number=None,
                    tags=None,
                    examples=[],
                    roman=None,
                    ipa=None,
                    etymology=None,
                    inflection_templates=None,
                    categories=None,
                    sense_id=None
                ))
                word_forms_count += 1
        # Progress callback every 10000 words
        if progress_callback and (idx % 10000 == 0 or idx == len(keys) - 1):
            progress_callback(idx + 1, len(keys))
    elapsed = time.time() - start_time
    stats = {
        "total_tokens": total_tokens,
        "unique_lemmas": len(unified_data),
        "elapsed": elapsed,
        "word_forms_created": word_forms_count,
    }
    return unified_data, stats


def main():

    """
    Main function to demonstrate building a unified dictionary from the trie with progress and pretty-printing.
    """
    # Use the default DB_PATH or override as needed
    print("--- Building Unified Dictionary from Trie ---")
    unified_dict, stats = parse_trie_to_unified_dict(DB_PATH, show_progress=True)

    import pprint
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    for key in ["замок", "блоха", "клаксон", "помилка", "обід"]:
        logger.info(f"Entry for lemma: '{key}'")
        entry = unified_dict.get(key)
        if not entry:
            logger.warning("  Not found in dictionary.")
            continue
        # Pretty-print the entry to stdout with color
        print(f"\033[1;36m{key}\033[0m:")
        print(f"\033[0;37m{pp.pformat(entry.model_dump())}\033[0m")


if __name__ == "__main__":
    main()