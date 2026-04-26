# kaikki — Ukrainian Dictionary Source

Kaikki.org Ukrainian dictionary (Wiktionary extract): structured JSONL with inflected forms, stress marks, POS, morphology, etymology, and translations.

## What it contains

- `kaikki.org-dictionary-Ukrainian.jsonl` (Git LFS) — full Wiktionary-derived JSONL; ~100 MB, one JSON object per word/POS entry.
- `kaikki.test.jsonl` — 5-entry fixture file used for offline unit tests.
- `kaikki_parser.py` — parser: `extract_stress_indices`, `strip_stress`, `normalize_pos`, `stream_kaikki_to_lmdb`.
- `__init__.py` — package init.
- `pretty.jsonl` — human-readable sample (pretty-printed first entries).

## Version & integrity

| Field           | Value                                      |
|-----------------|--------------------------------------------|
| Source          | https://kaikki.org/dictionary/Ukrainian/   |
| License         | CC BY-SA 4.0                               |
| Format          | JSONL, UTF-8                               |
| Approx. entries | ~200,000 lemmas, ~2 M inflected forms      |
| Size (raw)      | ~100 MB (Git LFS)                          |

## How it works

Each line is a JSON object for one lemma+POS combination.  The parser:

1. Reads `.forms[]` entries and strips combining-acute stress marks (U+0301) to get the base form.
2. Calls `extract_stress_indices(form)` to recover the 0-based vowel index of the stress mark.
3. Maps Wiktionary `tags` (e.g. `["nominative", "singular"]`) to UD feature keys (`Case=Nom`, `Number=Sing`).
4. Maps Wiktionary `pos` string to a `UPOS` enum value.
5. Yields `(lemma, LinguisticEntry)` pairs for disk-backed aggregation by the merger.

Stress mark detection recognises both the combining acute (U+0301) and the precomposed acute variants.

## How to build / rebuild

The source JSONL is downloaded from kaikki.org.  Place it at:

```
src/data_management/sources/kaikki/kaikki.org-dictionary-Ukrainian.jsonl
```

Then run the full pipeline:

```bash
conda activate verseSense-py312
python -m src.data_management.transform.parsing_merging_service
```

## How to use

```python
from src.data_management.sources.kaikki.kaikki_parser import (
    strip_stress,
    extract_stress_indices,
    normalize_pos,
)

# Stress extraction from a Wiktionary form string
form = "за́мок"  # castle (stress mark U+0301 after з)
base = strip_stress(form)              # "замок"
indices = extract_stress_indices(form) # [0]  — first vowel stressed

# Stream to LMDB (used by parsing_merging_service)
from src.data_management.sources.kaikki.kaikki_parser import stream_kaikki_to_lmdb
stream_kaikki_to_lmdb(
    input_path="src/data_management/sources/kaikki/kaikki.org-dictionary-Ukrainian.jsonl",
    lmdb_path="/tmp/kaikki_cache.lmdb",
)
```

## Tests

```bash
conda activate verseSense-py312
pytest tests/src/data_management/sources/test_kaikki_parser.py -v
```

Expected: all tests pass using `kaikki.test.jsonl` (no network access required).

## Dependencies

- `src/data_management/transform/data_unifier.py` — `LinguisticEntry`, `WordForm`, `UPOS`, `UDFeatKey`
- `src/data_management/transform/merger.py` — `LMDBExporter`
- `src/utils/normalize_apostrophe.py`
- `tqdm`, `lmdb`, `msgpack` (conda env)

---

## Dictionary Format

Each entry is a JSON object representing a word, its forms, senses, etymology, inflections, and more. The main fields are:

- `word`: The lemma (base form) of the word
- `lang`: Language name (e.g., "Ukrainian")
- `lang_code`: ISO language code (e.g., "uk")
- `pos`: Part of speech (e.g., "noun", "verb")
- `head_templates`: List of headword templates (see below)
- `forms`: List of inflected forms (see below)
- `inflection_templates`: List of inflection templates (see below)
- `etymology_number`: Integer, distinguishes homonyms
- `etymology_text`: Etymology as plain text
- `etymology_templates`: List of etymology templates (see below)
- `sounds`: List of pronunciation and audio data (see below)
- `categories`: List of semantic or grammatical categories
- `senses`: List of sense objects (see below)

---

## Data Types (TypedDicts)

### HeadTemplate

```python
class HeadTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]
    expansion: str
```

- Describes the headword template used in Wiktionary (e.g., "uk-noun").
- `args` are template arguments (e.g., {"1": "за́мок<\*.genu>"}).

### FormEntry

```python
class FormEntry(TypedDict, total=False):
    form: str
    tags: List[str]
    source: Optional[str]
    roman: Optional[str]
```

- Each inflected form of the word.
- `tags` describe grammatical features (e.g., ["nominative", "plural"]).
- `roman` is the romanized form, if available.

### InflectionTemplate

```python
class InflectionTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]
```

- Describes the inflectional paradigm (e.g., "uk-ndecl").

### EtymologyTemplate

```python
class EtymologyTemplate(TypedDict, total=False):
    name: str
    args: Dict[str, str]
    expansion: Optional[str]
```

- Structured etymology information, often referencing source languages or roots.

### SoundEntry

```python
class SoundEntry(TypedDict, total=False):
    ipa: Optional[str]
    audio: Optional[str]
    mp3_url: Optional[str]
    ogg_url: Optional[str]
    rhymes: Optional[str]
    tags: Optional[List[str]]
    text: Optional[str]
```

- Pronunciation and audio data for the word.

### CategoryEntry

```python
class CategoryEntry(TypedDict, total=False):
    name: str
    kind: str
    parents: List[str]
    source: Optional[str]
    orig: Optional[str]
    langcode: Optional[str]
    _dis: Optional[str]
```

- Semantic or grammatical categories, with metadata.

### SenseEntry

```python
class SenseEntry(TypedDict, total=False):
    glosses: List[str]
    id: Optional[str]
    links: Optional[List[List[str]]]
    tags: Optional[List[str]]
    categories: Optional[List[CategoryEntry]]
```

- Each sense (meaning) of the word.
- `glosses` are definitions (in English).
- `tags` describe usage, register, or grammatical features.

### TranslationEntry

```python
class TranslationEntry(TypedDict, total=False):
    code: str
    lang: str
    sense: Optional[str]
    word: str
    alt: Optional[str]
    note: Optional[str]
    roman: Optional[str]
    tags: Optional[List[str]]
```

- Translations of the sense into other languages.

---

## Example Entry

```json
{
  "word": "замок",
  "lang": "Ukrainian",
  "lang_code": "uk",
  "pos": "noun",
  "head_templates": [
    {
      "name": "uk-noun",
      "args": { "1": "за́мок<*.genu>" },
      "expansion": "за́мок • (zámok) m inan (genitive за́мку, nominative plural за́мки, genitive plural за́мків)"
    }
  ],
  "forms": [
    { "form": "за́мок", "tags": ["canonical", "inanimate", "masculine"] },
    { "form": "zámok", "tags": ["romanization"] },
    {
      "form": "за́мки",
      "tags": ["nominative", "plural"],
      "source": "declension",
      "roman": "zámky"
    }
    // ... more forms
  ],
  "inflection_templates": [
    { "name": "uk-ndecl", "args": { "1": "за́мок<*.genu>" } }
  ],
  "etymology_number": 1,
  "etymology_text": "Borrowed from Polish zamek (“castle”).",
  "etymology_templates": [
    {
      "name": "bor",
      "args": { "1": "uk", "2": "pl", "3": "zamek", "4": "", "5": "castle" },
      "expansion": "Polish zamek (“castle”)"
    }
  ],
  "sounds": [{ "ipa": "[ˈzamɔk]" }],
  "categories": ["Buildings", "Mechanisms"],
  "senses": [
    {
      "glosses": ["castle"],
      "id": "en-замок-uk-noun-kTpT198v",
      "categories": [
        {
          "name": "Buildings",
          "kind": "other",
          "parents": [],
          "source": "w+disamb"
        }
      ]
    }
  ]
}
```

---

## Contribution Guidelines

- **Field Types:** Use the above TypedDicts for new code and contributions.
- **Extending Schema:** If you add new fields, document them and update the TypedDicts.
- **Type Safety:** Avoid using `Any` unless the structure is truly open-ended.
- **Examples:** Always provide real data examples for new or complex fields.
- **Testing:** Validate new data types with mypy or Pyright for static type safety.

---

## Notes

- All glosses/definitions are in English (for Ukrainian, use mapping or semantic similarity for downstream tasks).
- IPA parsing may require custom logic to extract stress indices.
- Categories and tags are open-ended; document new values as needed.

---

## Attribution

When using this data or any derived databases, please include:

```
Ukrainian Machine-Readable Dictionary. Accessed December 25, 2025. https://kaikki.org/dictionary/Ukrainian/index.html. CC BY-SA 4.0.
```

If you use Wiktextract or the data on this site in academic work, please cite:

Tatu Ylonen. "Wiktextract: Wiktionary as Machine-Readable Structured Data." Proceedings of the 13th Conference on Language Resources and Evaluation (LREC), pp. 1317-1325, Marseille, 20-25 June 2022.

---

## References

- [Kaikki.org Data Model](https://kaikki.org/dictionary/)
- [Wiktionary Data Model](https://en.wiktionary.org/wiki/Wiktionary:Entry_layout_explained)
