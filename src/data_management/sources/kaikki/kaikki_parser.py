from typing import Dict
from utils.normalize_apostrophe import normalize_apostrophe
from data_management.transform.data_unifier import LinguisticDataUnifier, UPOS, WordForm
import logging
import time
from tqdm import tqdm
def build_kaikki_dict(filepath: str, verbose: bool = True) -> Dict[str, list]:
    """
    Parses a Kaikki.org JSONL file, normalizes word keys, and builds a dictionary
    of normalized_word -> list of unified entries (using LinguisticDataUnifier).
    Shows progress bar, timing, and logs if verbose is True.
    """
    logger = logging.getLogger("kaikki_parser")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    start_time = time.time()
    logger.info(f"Parsing Kaikki JSONL: {filepath}")

    # Count lines for progress bar
    with open(filepath, encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())

    import re
    word_entries: Dict[str, list] = {}

    def normalize_pos(pos):
        if not pos:
            return None
        pos_map = {
            'noun': UPOS.NOUN, 'verb': UPOS.VERB, 'adj': UPOS.ADJ, 'adjective': UPOS.ADJ, 'adv': UPOS.ADV, 'adverb': UPOS.ADV,
            'pron': UPOS.PRON, 'pronoun': UPOS.PRON, 'propn': UPOS.PROPN, 'proper_noun': UPOS.PROPN, 'num': UPOS.NUM, 'number': UPOS.NUM,
            'det': UPOS.DET, 'determiner': UPOS.DET, 'adp': UPOS.ADP, 'adposition': UPOS.ADP, 'aux': UPOS.AUX, 'auxiliary': UPOS.AUX,
            'cconj': UPOS.CCONJ, 'conj': UPOS.CCONJ, 'sconj': UPOS.SCONJ, 'part': UPOS.PART, 'particle': UPOS.PART,
            'intj': UPOS.INTJ, 'interjection': UPOS.INTJ, 'punct': UPOS.PUNCT, 'punctuation': UPOS.PUNCT,
            'sym': UPOS.SYM, 'symbol': UPOS.SYM, 'x': UPOS.X, 'other': UPOS.X
        }
        return pos_map.get(str(pos).lower(), UPOS.X)

    def normalize_tags(tags):
        if tags is None:
            return None
        if isinstance(tags, str):
            tags = [tags]
        flat = []
        for tag in tags:
            if isinstance(tag, list):
                flat.extend(tag)
            elif tag:
                flat.append(tag)
        flat = [str(t).strip().lower() for t in flat if t and str(t).strip()]
        return sorted(set(flat)) if flat else None

    def extract_stress_index(form: str) -> list:
        # Find the index of the accented vowel in the vowel sequence
        accented_vowels = 'а́е́є́и́і́ї́о́у́ю́я́А́Е́Є́И́І́Ї́О́У́Ю́Я́'
        combining_acute = '\u0301'
        vowels = 'аеєиіїоуюяАЕЄИІЇОУЮЯ'
        i = 0
        for j, c in enumerate(form):
            if c in vowels or c in accented_vowels:
                if c in accented_vowels or (j+1 < len(form) and form[j+1] == combining_acute and c in vowels):
                    return [i]
                if c in vowels:
                    i += 1
        return []

    with open(filepath, encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Parsing entries", disable=not verbose):
            if not line.strip():
                continue
            entry = json.loads(line)
            word = entry.get('word')
            if not word:
                continue
            norm_word = normalize_apostrophe(word)
            pos = normalize_pos(entry.get('pos'))
            inflection_templates = entry.get('inflection_templates')
            etymology_number = entry.get('etymology_number')
            etymology = entry.get('etymology_text')
            etymology_templates = entry.get('etymology_templates')
            sounds = entry.get('sounds', [])
            ipa = None
            for s in sounds:
                if 'ipa' in s:
                    ipa = s['ipa']
                    break
            categories = [c['name'] if isinstance(c, dict) and 'name' in c else c for c in entry.get('categories', [])]
            forms = entry.get('forms', [])
            form_map = {f['form']: f for f in forms if 'form' in f}
            # DEBUG: Only print for 'замок'
            if norm_word == 'замок':
                logger.info(f"Processing word: {word}")
                logger.info(f"Forms: {[f['form'] for f in forms if 'form' in f]}")
                logger.info(f"Senses: {[s.get('glosses', [None])[0] for s in entry.get('senses', [])]}")
            # For each sense, build a dict for the unifier
            for sense in entry.get('senses', []):
                glosses = sense.get('glosses', [])
                main_definition = glosses[0] if glosses else None
                alt_definitions = glosses[1:] if len(glosses) > 1 else None
                sense_id = sense.get('id') or None
                translations = sense.get('translations')
                sense_tags = normalize_tags(sense.get('tags'))
                sense_categories = [c['name'] if isinstance(c, dict) and 'name' in c else c for c in sense.get('categories', [])] if 'categories' in sense else None

                if norm_word == 'замок':
                    logger.info(f"Sense data: glosses={glosses}, main_definition={main_definition}, sense_id={sense_id}")
                # Try to find the correct accented form for this sense
                # 1. Try to match head_templates expansion or args[1] (if present)
                headword = None
                if 'head_templates' in entry and entry['head_templates']:
                    ht = entry['head_templates'][0]
                    if 'expansion' in ht and ht['expansion']:
                        # Take first word in expansion (may have gender etc. after)
                        headword = ht['expansion'].split()[0]
                    elif 'args' in ht and '1' in ht['args']:
                        headword = ht['args']['1']
                # 2. Try to match a form with the same accented spelling as headword
                matched_form = None
                if headword:
                    for f in forms:
                        if f.get('form') == headword:
                            matched_form = f
                            break
                # 3. Fallback: use first accented form that matches the sense (by gloss or by order)
                if not matched_form:
                    for f in forms:
                        form_str = f.get('form', '')
                        if any(ch in form_str for ch in '\u0301а́е́є́и́і́ї́о́у́ю́я́А́Е́Є́И́І́Ї́О́У́Ю́Я́'):
                            matched_form = f
                            break
                # 4. Fallback: use the word itself
                if not matched_form:
                    matched_form = {'form': word, 'tags': []}

                form_str = matched_form['form']
                stress_indices = extract_stress_index(form_str)
                feats = {}  # TODO: extract UD features from tags if possible
                roman = matched_form.get('roman')
                wordform = {
                    'pos': pos,
                    'feats': feats,
                    'lemma': word,
                    'main_definition': main_definition,
                    'alt_definitions': alt_definitions,
                    'translations': translations,
                    'etymology_templates': etymology_templates,
                    'etymology_number': etymology_number,
                    'tags': sense_tags,
                    'roman': roman,
                    'ipa': ipa,
                    'etymology': etymology,
                    'inflection_templates': inflection_templates,
                    'categories': (categories or sense_categories),
                    'sense_id': sense_id,
                    'examples': [],  # TODO: extract examples if available
                    'stress_indices': stress_indices,
                }
                # Only add if main_definition or sense_id are present (to avoid merging)
                if (main_definition or sense_id) and stress_indices:
                    word_entries.setdefault(norm_word, []).append(wordform)
                if norm_word == 'замок':
                    logger.info(f"Sense '{main_definition}' | Form: {form_str} | Stress indices: {stress_indices}")
                word_entries.setdefault(norm_word, []).append(wordform)

    logger.info(f"Parsed {total_lines:,} lines. Unique normalized words: {len(word_entries):,}")

    logger.info("Unifying entries with LinguisticDataUnifier...")
    unify_start = time.time()
    unifier = LinguisticDataUnifier()
    unified = unifier.transform(word_entries, source='kaikki')
    unify_time = time.time() - unify_start
    logger.info(f"Unified entries in {unify_time:.2f}s. Final dict size: {len(unified):,}")

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.2f}s")
    return unified
# --- Kaikki.org TypedDicts ---
from typing import TypedDict, List, Dict, Optional, Any, Iterator
import json

class HeadTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]
    expansion: str

class FormEntry(TypedDict, total=False):
    form: str
    tags: List[str]
    source: Optional[str]
    roman: Optional[str]

class InflectionTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]

class EtymologyTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]
    expansion: Optional[str]

class SoundEntry(TypedDict, total=False):
    ipa: Optional[str]
    audio: Optional[str]
    mp3_url: Optional[str]
    ogg_url: Optional[str]
    rhymes: Optional[str]
    tags: Optional[List[str]]
    text: Optional[str]

class CategoryEntry(TypedDict, total=False):
    name: str
    kind: str
    parents: List[str]
    source: Optional[str]
    orig: Optional[str]
    langcode: Optional[str]
    _dis: Optional[str]

class SenseEntry(TypedDict, total=False):
    glosses: List[str]
    id: Optional[str]
    links: Optional[List[List[str]]]
    tags: Optional[List[str]]
    categories: Optional[List[CategoryEntry]]
    translations: Optional[List['TranslationEntry']]

class TranslationEntry(TypedDict, total=False):
    code: str
    lang: str
    sense: Optional[str]
    word: str
    alt: Optional[str]
    note: Optional[str]
    roman: Optional[str]
    tags: Optional[List[str]]

class KaikkiEntry(TypedDict, total=False):
    word: str
    lang: str
    lang_code: str
    pos: str
    head_templates: List[HeadTemplate]
    forms: List[FormEntry]
    inflection_templates: List[InflectionTemplate]
    etymology_number: Optional[int]
    etymology_text: Optional[str]
    etymology_templates: List[EtymologyTemplate]
    sounds: List[SoundEntry]
    categories: List[str]
    senses: List[SenseEntry]
    translations: Optional[List[TranslationEntry]]

# --- Kaikki Parser ---

def parse_kaikki_jsonl(filepath: str) -> Iterator[KaikkiEntry]:
    """
    Yields KaikkiEntry objects from a Kaikki.org JSONL file.
    Each line is parsed and type-checked.
    """
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Optionally: validate required fields here
            yield data  # type: ignore
