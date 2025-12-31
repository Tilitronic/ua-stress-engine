
import os
import glob
import shutil
import json
import re
from pathlib import Path
from typing import Callable, Tuple, TypedDict, List, Dict, Optional, Any, Iterator
from tqdm import tqdm
from src.utils.normalize_apostrophe import normalize_apostrophe
from src.data_management.transform.data_unifier import LinguisticEntry, WordForm, UPOS, UDFeatKey, GenderVal, NumberVal, CaseVal
from src.data_management.transform.merger import LMDBExporter, LMDBExportConfig
from src.data_management.transform.cache_utils import to_serializable

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
        # word_entries: Dict[str, list] = {}

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

def strip_stress(form: str) -> str:
    # Remove acute accents from vowels (both precomposed and combining)
    form = re.sub(r'[\u0301]', '', form)  # Remove combining acute
    form = re.sub(r'([аеєиіїоуюяАЕЄИІЇОУЮЯ])\u0301', r'\1', form)  # Remove combining after vowel
    form = re.sub(r'([аеиієїоуюяАЕІЄЇОУЮЯ])́', r'\1', form)  # Remove precomposed acute
    return form

def extract_stress_indices(form: str) -> list:
    # Return the index/indices of accented vowels (0-based, left-to-right)
    vowels = 'аеєиіїоуюяАЕЄИІЇОУЮЯ'
    precomposed = {
        'а́': 'а', 'е́': 'е', 'є́': 'є', 'и́': 'и', 'і́': 'і', 'ї́': 'ї', 'о́': 'о', 'у́': 'у', 'ю́': 'ю', 'я́': 'я',
        'А́': 'А', 'Е́': 'Е', 'Є́': 'Є', 'И́': 'И', 'І́': 'І', 'Ї́': 'Ї', 'О́': 'О', 'У́': 'У', 'Ю́': 'Ю', 'Я́': 'Я',
    }
    indices = []
    vowel_idx = 0
    i = 0
    while i < len(form):
        c = form[i]
        # Precomposed accented vowel (2 chars)
        if i+1 < len(form) and form[i:i+2] in precomposed:
            indices.append(vowel_idx)
            vowel_idx += 1
            i += 2
            continue
        # Vowel + combining acute
        if c in vowels:
            if i+1 < len(form) and form[i+1] == '\u0301':
                indices.append(vowel_idx)
                i += 2
            else:
                i += 1
            vowel_idx += 1
        else:
            i += 1
    return indices

# --- Kaikki tag to UD feature value mapping ---
tag_to_ud = {
    # Number
    'singular': (UDFeatKey.Number, NumberVal.Sing),
    'plural': (UDFeatKey.Number, NumberVal.Plur),

    # Gender
    'masculine': (UDFeatKey.Gender, GenderVal.Masc),
    'feminine': (UDFeatKey.Gender, GenderVal.Fem),
    'neuter': (UDFeatKey.Gender, GenderVal.Neut),

    # Case
    'nominative': (UDFeatKey.Case, CaseVal.Nom),
    'genitive': (UDFeatKey.Case, CaseVal.Gen),
    'dative': (UDFeatKey.Case, CaseVal.Dat),
    'accusative': (UDFeatKey.Case, CaseVal.Acc),
    'instrumental': (UDFeatKey.Case, CaseVal.Ins),
    'locative': (UDFeatKey.Case, CaseVal.Loc),
    'vocative': (UDFeatKey.Case, CaseVal.Voc),

    # Animacy
    'inanimate': (UDFeatKey.Animacy, 'Inan'),
    'animate': (UDFeatKey.Animacy, 'Anim'),

    # Aspect
    'imperfective': (UDFeatKey.Aspect, 'Imp'),
    'perfective': (UDFeatKey.Aspect, 'Perf'),

    # Tense
    'present': (UDFeatKey.Tense, 'Pres'),
    'past': (UDFeatKey.Tense, 'Past'),
    'future': (UDFeatKey.Tense, 'Fut'),

    # Person
    'first-person': (UDFeatKey.Person, '1'),
    'second-person': (UDFeatKey.Person, '2'),
    'third-person': (UDFeatKey.Person, '3'),

    # VerbForm
    'infinitive': (UDFeatKey.VerbForm, 'Inf'),
    'imperative': (UDFeatKey.VerbForm, 'Imp'),
    'participle': (UDFeatKey.VerbForm, 'Part'),
    'adverbial': (UDFeatKey.VerbForm, 'Conv'),

    # Voice
    'active': (UDFeatKey.Voice, 'Act'),
    'passive': (UDFeatKey.Voice, 'Pass'),

    # Degree
    'comparative': (UDFeatKey.Degree, 'Cmp'),
    'superlative': (UDFeatKey.Degree, 'Sup'),

    # Reflexivity
    'reflexive': (UDFeatKey.Reflex, 'Yes'),

    # Polarity
    'negative': (UDFeatKey.Polarity, 'Neg'),
    'positive': (UDFeatKey.Polarity, 'Pos'),

    # PronType, NumType, Poss, etc. can be added as needed
}

def merge_wordforms(forms, double_stress_lemmas):
    merge_map = {}
    if not hasattr(merge_wordforms, '_variative_stress_lemmas'):
        # Use correct, project-root-relative path for variative stress words
        merge_wordforms._variative_stress_lemmas = load_variative_stress_lemmas(os.path.join(os.path.dirname(__file__), '..', 'ua_variative_stressed_words', 'ua_variative_stressed_words.txt'))
    variative_stress_lemmas = merge_wordforms._variative_stress_lemmas

    lemma_set = {wf.lemma.lower() for wf in forms if wf.lemma}
    is_double_stress = any(lemma in double_stress_lemmas for lemma in lemma_set)
    is_variative_stress = any(lemma in variative_stress_lemmas for lemma in lemma_set)
    if is_variative_stress:
        # Merge by (form, sense_id, pos, number, gender, case), treating None/empty as wildcard only if one side is missing
        def extract_feat(feats, key):
            return feats.get(key, None) if feats else None

        # Merge forms with same form, sense_id, pos, and (Number or None) as a group
        def group_key(wf):
            number = wf.feats.get(UDFeatKey.Number) if wf.feats else None
            # treat None and 'Sing' as the same group for merging
            if number is None or number == 'Sing':
                number_group = 'SingOrNone'
            else:
                number_group = number
            return (wf.form, wf.sense_id, wf.pos, number_group)

        group_map = {}
        for wf in forms:
            key = group_key(wf)
            if key not in group_map:
                group_map[key] = []
            group_map[key].append(wf)
        merged = []
        for group in group_map.values():
            all_stress = set()
            for wf in group:
                all_stress.update(wf.stress_indices)
            base = group[0].model_copy()
            base.stress_indices = sorted(all_stress)
            merged.append(base)
        return merged
    elif is_double_stress:
        # For double-stress: merge by (sense_id, form, pos, feats)
        for wf in forms:
            key = (
                wf.sense_id,
                wf.form,
                wf.pos,
                tuple(sorted(wf.feats.items())),
            )
            if key not in merge_map:
                merge_map[key] = wf.model_copy()
                merge_map[key].stress_indices = list(wf.stress_indices)
            else:
                merge_map[key].stress_indices = sorted(set(merge_map[key].stress_indices) | set(wf.stress_indices))
        return list(merge_map.values())
    else:
        return forms

def load_variative_stress_lemmas(resource_path=None):
    if resource_path is None:
        # Use correct, project-root-relative path for robust loading
        resource_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ua_variative_stressed_words', 'ua_variative_stressed_words.txt'))
    lemmas = set()
    try:
        with open(resource_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lemmas.add(line.lower())
    except Exception as e:
        print(f"Warning: Could not load variative-stress lemma list: {e}")
    return lemmas

def parse_kaikki_to_unified_dict(
    input_path: str,
    show_progress: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[Dict[str, LinguisticEntry], dict]:
    """
    Parse a Kaikki.org JSONL file and return a dict: lemma -> LinguisticEntry.
    Supports progress bar and progress_callback for multiprocessing.
    """
    # Count lines for progress bar
    word_entries = {}
    with open(input_path, encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())

    unified_data: Dict[str, LinguisticEntry] = {}


    with open(input_path, encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="[Parsing Kaikki]", disable=not show_progress)):
            if not line.strip():
                continue
            entry = json.loads(line)
            word = entry.get('word')
            if not word:
                continue
            norm_word = normalize_apostrophe(strip_stress(word))
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
            # Filter out invalid forms
            valid_forms = [
                f for f in forms
                if 'form' in f
                and f['form'] not in {"no-table-tags", "uk-ndecl"}
                and not set(f.get('tags', [])) & {"inflection-template", "table-tags", "romanization"}
            ]

            # Find canonical form string from head_templates or inflection_templates
            canonical_form_str = None
            if 'head_templates' in entry and entry['head_templates']:
                ht = entry['head_templates'][0]
                if 'expansion' in ht and ht['expansion']:
                    canonical_form_str = ht['expansion'].split()[0]
                elif 'args' in ht and '1' in ht['args']:
                    canonical_form_str = ht['args']['1']
            if not canonical_form_str and inflection_templates:
                it = inflection_templates[0]
                if 'args' in it and '1' in it['args']:
                    canonical_form_str = it['args']['1']

            for sense in entry.get('senses', []):
                glosses = sense.get('glosses', [])
                main_definition = glosses[0] if glosses else None
                alt_definitions = glosses[1:] if len(glosses) > 1 else None
                sense_id = sense.get('id') or None
                translations = sense.get('translations')
                sense_tags = normalize_tags(sense.get('tags'))
                sense_categories = [c['name'] if isinstance(c, dict) and 'name' in c else c for c in sense.get('categories', [])] if 'categories' in sense else None
                for f in valid_forms:
                    form_str = f.get('form', '')
                    norm_form = normalize_apostrophe(strip_stress(form_str))
                    # Always extract stress from the actual form string
                    stress_indices = extract_stress_indices(form_str)
                    if norm_word == 'замок':
                        print(f"DEBUG: form_str='{form_str}' norm_form='{norm_form}' stress_indices={stress_indices}")
                    if not stress_indices:
                        continue
                    # Map tags to UD features using the pedantic mapping
                    feats = {}
                    for tag in f.get('tags', []):
                        mapping = tag_to_ud.get(tag)
                        if mapping:
                            k, v = mapping
                            feats[k] = v
                    roman = f.get('roman')
                    wordform = WordForm(
                        form=norm_form,
                        pos=pos,
                        feats=feats,
                        lemma=norm_word,
                        main_definition=main_definition,
                        alt_definitions=alt_definitions,
                        translations=translations,
                        etymology_templates=etymology_templates,
                        etymology_number=etymology_number,
                        tags=sense_tags,
                        roman=roman,
                        ipa=ipa,
                        etymology=etymology,
                        inflection_templates=inflection_templates,
                        categories=(categories or sense_categories),
                        sense_id=sense_id,
                        examples=[],
                        stress_indices=stress_indices,
                    )
                    word_entries.setdefault(norm_word, []).append(wordform)
            if progress_callback and (idx % 1000 == 0 or idx == total_lines - 1):
                progress_callback(idx + 1, total_lines)





    # Load double-stress lemmas inside the function to avoid global heavy work
    double_stress_lemmas = load_double_stress_lemmas()
    unified = {}
    for word, forms in word_entries.items():
        try:
            merged_forms = merge_wordforms(forms, double_stress_lemmas)
            # Collect all unique, sorted stress index arrays from forms
            stress_patterns = set()
            for wf in merged_forms:
                if wf.stress_indices:
                    stress_patterns.add(tuple(sorted(wf.stress_indices)))
            possible_stress_indices = [list(pattern) for pattern in sorted(stress_patterns)] if stress_patterns else []
            entry = LinguisticEntry(word=word, forms=merged_forms, possible_stress_indices=possible_stress_indices)
            unified[word] = entry
        except Exception as e:
            print(f"[ERROR] Failed to merge forms for word '{word}': {e}")
    stats = {
        'total_lines': total_lines,
        'unique_lemmas': len(unified),
        'lmdb_path': None  # Will be set by caller if needed
    }
    return unified, stats


# Load double-stress lemmas
def load_double_stress_lemmas(resource_path=None):
    if resource_path is None:
        # Use the same file as variative-stress for now, or handle gracefully if not found
        resource_path = os.path.join(os.path.dirname(__file__), '..', 'ua_variative_stressed_words', 'ua_variative_stressed_words.txt')
    lemmas = set()
    try:
        with open(resource_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lemmas.add(line.lower())
    except Exception as e:
        print(f"Warning: Could not load double-stress lemma list: {e}")
    return lemmas

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

def stream_kaikki_to_lmdb(input_path: str, lmdb_path: str, map_size: int = 10 * 1024 * 1024 * 1024, show_progress: bool = True, batch_size: int = 10000):
    """
    Stream Kaikki.org JSONL to LMDB, applying post-parsing merging logic per lemma.
    """
    double_stress_lemmas = load_double_stress_lemmas()
    # Prepare cache dir
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path)
    # Count total lines for progress bar
    with open(input_path, encoding="utf-8") as f:
        total_lines = sum(1 for line in f if line.strip())
    # Streaming parse and yield (lemma, entry) pairs
    def entry_iter():
        word_entries = {}
        with open(input_path, encoding="utf-8") as f, tqdm(total=total_lines, desc="[Kaikki→LMDB]", disable=not show_progress) as pbar:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                word = entry.get('word')
                if not word:
                    pbar.update(1)
                    continue
                norm_word = normalize_apostrophe(strip_stress(word))
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
                valid_forms = [
                    f for f in forms
                    if 'form' in f
                    and f['form'] not in {"no-table-tags", "uk-ndecl"}
                    and not set(f.get('tags', [])) & {"inflection-template", "table-tags", "romanization"}
                ]
                canonical_form_str = None
                if 'head_templates' in entry and entry['head_templates']:
                    ht = entry['head_templates'][0]
                    if 'expansion' in ht and ht['expansion']:
                        canonical_form_str = ht['expansion'].split()[0]
                    elif 'args' in ht and '1' in ht['args']:
                        canonical_form_str = ht['args']['1']
                if not canonical_form_str and inflection_templates:
                    it = inflection_templates[0]
                    if 'args' in it and '1' in it['args']:
                        canonical_form_str = it['args']['1']
                for sense in entry.get('senses', []):
                    glosses = sense.get('glosses', [])
                    main_definition = glosses[0] if glosses else None
                    alt_definitions = glosses[1:] if len(glosses) > 1 else None
                    sense_id = sense.get('id') or None
                    translations = sense.get('translations')
                    sense_tags = normalize_tags(sense.get('tags'))
                    sense_categories = [c['name'] if isinstance(c, dict) and 'name' in c else c for c in sense.get('categories', [])] if 'categories' in sense else None
                    for f in valid_forms:
                        form_str = f.get('form', '')
                        norm_form = normalize_apostrophe(strip_stress(form_str))
                        stress_indices = extract_stress_indices(form_str)
                        if not stress_indices:
                            continue
                        feats = {}
                        for tag in f.get('tags', []):
                            mapping = tag_to_ud.get(tag)
                            if mapping:
                                k, v = mapping
                                feats[k] = v
                        roman = f.get('roman')
                        wordform = WordForm(
                            form=norm_form,
                            pos=pos,
                            feats=feats,
                            lemma=norm_word,
                            main_definition=main_definition,
                            alt_definitions=alt_definitions,
                            translations=translations,
                            etymology_templates=etymology_templates,
                            etymology_number=etymology_number,
                            tags=sense_tags,
                            roman=roman,
                            ipa=ipa,
                            etymology=etymology,
                            inflection_templates=inflection_templates,
                            categories=(categories or sense_categories),
                            sense_id=sense_id,
                            examples=[],
                            stress_indices=stress_indices,
                        )
                        word_entries.setdefault(norm_word, []).append(wordform)
                # Periodically yield and clear
                if (idx + 1) % batch_size == 0:
                    for word, forms in word_entries.items():
                        merged_forms = merge_wordforms(forms, double_stress_lemmas)
                        stress_patterns = set(tuple(sorted(wf.stress_indices)) for wf in merged_forms if wf.stress_indices)
                        possible_stress_indices = [list(pattern) for pattern in sorted(stress_patterns)] if stress_patterns else []
                        entry_obj = LinguisticEntry(word=word, forms=merged_forms, possible_stress_indices=possible_stress_indices)
                        yield word, to_serializable(entry_obj)
                    word_entries.clear()
                pbar.update(1)
            # Final flush
            for word, forms in word_entries.items():
                merged_forms = merge_wordforms(forms, double_stress_lemmas)
                stress_patterns = set(tuple(sorted(wf.stress_indices)) for wf in merged_forms if wf.stress_indices)
                possible_stress_indices = [list(pattern) for pattern in sorted(stress_patterns)] if stress_patterns else []
                entry_obj = LinguisticEntry(word=word, forms=merged_forms, possible_stress_indices=possible_stress_indices)
                yield word, to_serializable(entry_obj)

    config_obj = LMDBExportConfig(db_path=Path(lmdb_path), overwrite=True)
    exporter = LMDBExporter(config_obj)
    exporter.export_streaming(entry_iter(), show_progress=False)
    print(f"[Kaikki→LMDB] Finished LMDB export at {lmdb_path}")

    # --- Old cache cleanup ---
    prefix = "KAIKKI"
    cache_root = os.path.dirname(lmdb_path)
    current_key = os.path.basename(lmdb_path)
    for d in glob.glob(os.path.join(cache_root, f"{prefix}_*_lmdb")):
        if os.path.basename(d) != current_key:
            try:
                shutil.rmtree(d)
                print(f"[Kaikki→LMDB] Deleted old cache: {d}")
            except Exception as e:
                print(f"[Kaikki→LMDB] Failed to delete old cache {d}: {e}")


def main():
    from tqdm import tqdm as _tqdm
    import pprint
    import sys
    # Test mode: use small test file for fast runs
    TEST_PATH = os.path.join(os.path.dirname(__file__), "kaikki.test.jsonl")
    FULL_PATH = os.path.join(os.path.dirname(__file__), "kaikki.org-dictionary-Ukrainian.jsonl")
    test_mode = True
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_mode = False
    path = TEST_PATH if test_mode else FULL_PATH
    _tqdm.write(f"\n--- Building Unified Dictionary from Kaikki ({'TEST' if test_mode else 'FULL'}) ---")
    unified, stats = parse_kaikki_to_unified_dict(path, show_progress=True)
    _tqdm.write(f"Stats: {stats}")
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=False)
    for key in ["помилка"]:
        _tqdm.write(f"Entry for lemma: '{key}'")
        entry = unified.get(key)
        if not entry:
            _tqdm.write("  Not found in dictionary.")
            continue
        _tqdm.write(f"\033[1;36m{key}\033[0m:")
        _tqdm.write(f"\033[0;37m{pp.pformat(entry.model_dump())}\033[0m")
    _tqdm.write(f"\nTo run on full data, use: python -m src.data_management.sources.kaikki.kaikki_parser --full")


if __name__ == "__main__":
    main()