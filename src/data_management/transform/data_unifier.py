"""
Linguistic Data Unifier

Transforms parsed data from any source into a unified, type-safe, spaCy/UD-inspired format using Pydantic v2 models.
This format is designed for merging, processing, and exporting comprehensive linguistic data from multiple sources.

References:
- Universal Dependencies: https://universaldependencies.org/
- spaCy Token API: https://spacy.io/api/token
- Pydantic v2: https://docs.pydantic.dev/latest/
"""

from typing import List, Dict, Optional, Any, Union, TypedDict
# --- TypedDicts for Structured Dicts ---

class TranslationEntry(TypedDict, total=False):
    lang: str
    text: str
    sense: Optional[str]

class EtymologyTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, Any]

class InflectionTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, Any]
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum

# --- Universal Dependencies Enums ---
# https://universaldependencies.org/u/pos/
# https://universaldependencies.org/u/feat/

class UPOS(str, Enum):
    """
    Universal POS (Part-of-Speech) tags following UD standard.
    Ref: https://universaldependencies.org/u/pos/
    """
    ADJ = "ADJ"      # прикметник (adjective)
    ADP = "ADP"      # прийменник (adposition)
    ADV = "ADV"      # прислівник (adverb)
    AUX = "AUX"      # допоміжне дієслово (auxiliary verb)
    CCONJ = "CCONJ"  # сурядний сполучник (coordinating conjunction)
    DET = "DET"      # детермінатив (determiner/article)
    INTJ = "INTJ"    # вигук (interjection)
    NOUN = "NOUN"    # іменник (noun)
    NUM = "NUM"      # числівник (numeral)
    PART = "PART"    # частка (particle)
    PRON = "PRON"    # займенник (pronoun)
    PROPN = "PROPN"  # власна назва (proper noun)
    PUNCT = "PUNCT"  # пунктуація (punctuation)
    SCONJ = "SCONJ"  # підрядний сполучник (subordinating conjunction)
    SYM = "SYM"      # символ (symbol)
    VERB = "VERB"    # дієслово (verb)
    X = "X"          # інше/невідоме (other/unknown)

# Core UD features (expand as needed)
class UDFeatKey(str, Enum):
    """
    Morphological feature keys following UD standard.
    Ref: https://universaldependencies.org/u/feat/
    """
    Gender = "Gender"         # рід (grammatical gender)
    Animacy = "Animacy"       # одушевленість (animate/inanimate)
    Number = "Number"         # число (singular/plural/dual)
    Case = "Case"             # відмінок (grammatical case)
    Definite = "Definite"     # означеність (definiteness)
    Degree = "Degree"         # ступінь порівняння (degree of comparison)
    VerbForm = "VerbForm"     # форма дієслова (verb form type)
    Mood = "Mood"             # спосіб (verb mood)
    Tense = "Tense"           # час (verb tense)
    Aspect = "Aspect"         # вид (verb aspect)
    Voice = "Voice"           # стан (verb voice)
    Evident = "Evident"       # евиденціальність (evidentiality)
    Polarity = "Polarity"     # заперечення (negation/polarity)
    Person = "Person"         # особа (grammatical person)
    Polite = "Polite"         # ввічливість (politeness)
    Clusivity = "Clusivity"   # включеність (inclusive/exclusive)
    PronType = "PronType"     # тип займенника (pronoun type)
    NumType = "NumType"       # тип числівника (numeral type)
    Poss = "Poss"             # присвійність (possessive)
    Reflex = "Reflex"         # зворотність (reflexive)
    Foreign = "Foreign"       # іншомовність (foreign language)
    Abbr = "Abbr"             # абревіатура (abbreviation)
    Typo = "Typo"             # описка (spelling error)

# Example values for some features (expand as needed)
class GenderVal(str, Enum):
    """Grammatical gender values (Gender feature)."""
    Masc = "Masc"  # Masculine (чоловічий)
    Fem = "Fem"    # Feminine (жіночий)
    Neut = "Neut"  # Neuter (середній)

class NumberVal(str, Enum):
    """Number values (Number feature)."""
    Sing = "Sing"  # Singular (однина)
    Plur = "Plur"  # Plural (множина)

class CaseVal(str, Enum):
    """Case values (Case feature) - Ukrainian cases."""
    Nom = "Nom"    # Nominative (називний)
    Gen = "Gen"    # Genitive (родовий)
    Dat = "Dat"    # Dative (давальний)
    Acc = "Acc"    # Accusative (знахідний)
    Ins = "Ins"    # Instrumental (орудним)
    Loc = "Loc"    # Locative (місцевий)
    Voc = "Voc"    # Vocative (кличний)

# --- Unified Data Models ---

# --- Unified Data Models ---

class WordForm(BaseModel):
    """
    Represents a unique word form as defined by its normalized word form (no stress marks, normalized apostrophe), stress pattern(s), POS, features, and all available metadata.
    - The 'form' field stores the normalized, correctly spelled word form (e.g., 'блохи', 'мати', "п'ятниця").
    - This model does NOT attempt to merge by meaning/sense unless the source provides explicit sense IDs or markers.
    - If stress changes meaning, it results in a separate WordForm entity.
    - If stress does not change meaning (as indicated by the source), all valid stress positions are grouped in one WordForm.
    - All metadata (definitions, translations, etymology, etc.) is attached as provided by the source, per form.
    - This approach is robust for polysemy, homographs, and ambiguous cases where sense disambiguation is not possible.
    """
    form: str = Field(
        ...,
        description="Normalized, correctly spelled word form (no stress marks, normalized apostrophe)",
        examples=["блохи", "мати", "п'ятниця"]
    )
    stress_indices: List[int] = Field(
        ...,
        description=(
            "Indices of stressed vowels (0-based, left-to-right in word). "
            "If a word form allows multiple stress patterns with no change in meaning, POS, or features (e.g., 'помилка' → пОмилка/помИлка), "
            "all valid stress positions are listed together in one entry. For most forms, this will be a single-element list; "
            "for true free-stress forms, it may contain multiple indices."
        ),
        examples=[[0], [1], [0, 1]]
    )
    pos: UPOS = Field(
        ...,
        description="Universal POS tag (UPOS) for this atomic form. Strictly validated against UD standard.",
        examples=[UPOS.NOUN, UPOS.VERB]
    )
    feats: Dict[UDFeatKey, str] = Field(
        default_factory=dict,
        description="Morphological features (UD-compliant), one value per feature for this atomic form. Keys strictly validated.",
        examples=[{UDFeatKey.Gender: "Fem", UDFeatKey.Case: "Nom"}]
    )
    lemma: Optional[str] = Field(
        default=None,
        description="Base form of the word",
        examples=["мати"]
    )
    main_definition: Optional[str] = Field(
        default=None,
        description="Main definition/gloss for this sense (Wiktionary/Kaikki)",
        examples=["castle"]
    )
    alt_definitions: Optional[List[str]] = Field(
        default=None,
        description="Alternative glosses for this sense (Wiktionary/Kaikki)",
        examples=[["fortress", "stronghold"]]
    )
    translations: Optional[List[TranslationEntry]] = Field(
        default=None,
        description="List of translation entries (Wiktionary/Kaikki)",
        examples=[[{"lang": "en", "text": "castle"}]]
    )
    etymology_templates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Structured etymology templates (Wiktionary/Kaikki)",
        examples=[[{"name": "bor", "args": {"1": "uk", "2": "pl", "3": "zamek"}}]]
    )
    etymology_number: Optional[int] = Field(
        default=None,
        description="Etymology number (Wiktionary/Kaikki)",
        examples=[1]
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Semantic or grammatical tags (Wiktionary/Kaikki)",
        examples=[["inanimate", "masculine"]]
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example sentences",
        examples=[["Вона моя мати."]]
    )
    roman: Optional[str] = Field(
        default=None,
        description="Romanization of the word form (if available)",
        examples=["zamók"]
    )
    ipa: Optional[str] = Field(
        default=None,
        description="IPA pronunciation string (if available)",
        examples=["[ˈzamɔk]"]
    )
    etymology: Optional[str] = Field(
        default=None,
        description="Etymology text (if available)",
        examples=["Borrowed from Polish zamek (‘castle’)"]
    )
    inflection_templates: Optional[List[InflectionTemplate]] = Field(
        default=None,
        description="Inflection templates (Wiktionary/Kaikki format)",
        examples=[[{"name": "uk-ndecl", "args": {"1": "замо́к<*>"}}]]
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="List of semantic or grammatical categories",
        examples=[["Buildings", "Mechanisms"]]
    )
    sense_id: Optional[str] = Field(
        default=None,
        description="Sense ID (Wiktionary/Kaikki)",
        examples=["en-замок-uk-noun-kTpT198v"]
    )
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_validator('feats', mode='before')
    def validate_feats_keys(cls, v: Any) -> Dict[UDFeatKey, str]:
        # Accept string keys but convert to Enum, error if not allowed
        if not isinstance(v, dict):
            raise TypeError("feats must be a dict")
        new = {}
        for k, val in v.items():
            try:
                key_enum = UDFeatKey(k) if not isinstance(k, UDFeatKey) else k
            except ValueError:
                raise ValueError(f"Invalid UD feature key: {k}")
            new[key_enum] = val
        return new

class LinguisticEntry(BaseModel):
    word: str = Field(..., description="Normalized word key", examples=["мати"])
    forms: List[WordForm] = Field(..., description="All forms/variants for this word")
    possible_stress_indices: List[List[int]] = Field(
        default_factory=list,
        description="All unique stress index arrays for this word across all forms (e.g., [[0], [1], [0, 1]]). Each entry is a unique, sorted list of stressed vowel indices for a form.",
    )
    meta: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata for the word")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

# --- Unifier Class ---

# class LinguisticDataUnifier:
#     """
#     Transforms parsed source data into the unified linguistic format.
#     - Accepts data from any parser (trie, txt, Kaikki, etc.)
#     - Outputs a dict: word -> LinguisticEntry, where each form is atomic (single stress, single POS, single feature set)
#     - Allows duplicates for same stress with different POS/features
#     """
#     def transform(self, parsed_data: Dict[str, List[Dict[str, Any]]], source: str) -> Dict[str, LinguisticEntry]:
#         """
#         For each word, produce a list of atomic forms. If multiple forms have identical lemma, POS, and features
#         (and other properties except stress), merge their stress_indices into a single entry. This is crucial for
#         words like 'помилка', where multiple stresses are possible but do not change meaning or features.
#         """
#         from collections import defaultdict
#         from itertools import product

#         result: Dict[str, LinguisticEntry] = {}
#         for word, forms in parsed_data.items():
#             # Temporary dict to merge forms by (lemma, pos, feats, definition, examples, source, meta)
#             merge_map: Dict[Any, Any] = {}
#             for form in forms:
#                 # --- Explode legacy/ambiguous input to atomic forms ---
#                 stress_variants = form.get('stress_indices', [])
#                 if not isinstance(stress_variants, list):
#                     stress_variants = [stress_variants]
#                 if len(stress_variants) == 0:
#                     stress_variants = [[]]
#                 elif not isinstance(stress_variants[0], list):
#                     # Single variant, wrap in list
#                     stress_variants = [stress_variants]

#                 pos_list = form.get('pos', [])
#                 if isinstance(pos_list, str):
#                     pos_list = [pos_list]
#                 elif not pos_list:
#                     pos_list = [None]

#                 feats_dict = form.get('feats', {})
#                 # If values are lists, explode; else treat as atomic
#                 if feats_dict and any(isinstance(v, list) for v in feats_dict.values()):
#                     keys = list(feats_dict.keys())
#                     value_lists = [v if isinstance(v, list) else [v] for v in feats_dict.values()]
#                     feats_variants = [dict(zip(keys, prod)) for prod in product(*value_lists)]
#                 else:
#                     feats_variants = [feats_dict or {}]

#                 # --- Explode all combinations to atomic forms ---
#                 for stress in stress_variants:
#                     for pos in pos_list:
#                         if pos is None:
#                             continue
#                         for feats in feats_variants:
#                             # --- Merge key: all properties except stress_indices ---
#                             merge_key = (
#                                 pos,
#                                 tuple(sorted(feats.items())),
#                                 form.get('lemma'),
#                                 form.get('main_definition'),
#                                 tuple(form.get('examples', [])),
#                                 source,
#                                 frozenset(form.get('meta', {}).items())
#                             )
#                             if merge_key not in merge_map:
#                                 merge_map[merge_key] = {
#                                     'stress_indices': set(),
#                                     'pos': pos,
#                                     'feats': feats,
#                                     'lemma': form.get('lemma'),
#                                     'main_definition': form.get('main_definition'),
#                                     'alt_definitions': form.get('alt_definitions'),
#                                     'translations': form.get('translations'),
#                                     'etymology_templates': form.get('etymology_templates'),
#                                     'etymology_number': form.get('etymology_number'),
#                                     'tags': form.get('tags'),
#                                     'roman': form.get('roman'),
#                                     'ipa': form.get('ipa'),
#                                     'etymology': form.get('etymology'),
#                                     'inflection_templates': form.get('inflection_templates'),
#                                     'categories': form.get('categories'),
#                                     'sense_id': form.get('sense_id'),
#                                     'examples': form.get('examples', []),
#                                     'source': source,
#                                     'meta': form.get('meta', {})
#                                 }
#                             # Add all stress indices for this atomic form
#                             # (for words like 'помилка', this will collect [0, 1] in one entry)
#                             merge_map[merge_key]['stress_indices'].update(stress)

#             # --- Build final atomic forms, merging stresses where appropriate ---
#             entry_forms: List[WordForm] = []
#             for merged in merge_map.values():
#                 wordform_kwargs = dict(
#                     stress_indices=sorted(merged['stress_indices']),
#                     pos=merged['pos'],
#                     feats=merged['feats'],
#                     lemma=merged.get('lemma'),
#                     examples=merged.get('examples', []),
#                 )
#                 for field in [
#                     'main_definition', 'alt_definitions', 'translations', 'etymology_templates', 'etymology_number',
#                     'tags', 'roman', 'ipa', 'etymology', 'inflection_templates', 'categories', 'sense_id']:
#                     if field in merged:
#                         wordform_kwargs[field] = merged[field]
#                 entry_forms.append(WordForm(**wordform_kwargs))
#             result[word] = LinguisticEntry(word=word, forms=entry_forms)
#         return result

# --- Example Usage ---
# from data_management.transform.data_unifier import LinguisticDataUnifier
# txt_data = TXTParser().parse_file(txt_path)
# unifier = LinguisticDataUnifier()
# txt_unified = unifier.transform(txt_data, source='txt')
# Now txt_unified is a type-safe, UD-compatible, spaCy-inspired dictionary
