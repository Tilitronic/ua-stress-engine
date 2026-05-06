# Ukrainian Stress Engine — Public API Design

> **Status**: Canonical specification.  
> **Target**: `ua-stress-core` (Rust), `ukrainian_stress` (Python/PyO3), `ua-word-stress` (WASM/npm).  
> **Shared format**: identical JSON-compatible shape in all three runtimes.  
> **Companion**: `pl-stress-engine` will adopt the same format.

---

## Design Principles

### 1 · Return everything, decide nothing

The Rust package **never picks one variant**. It returns all the data it holds for
every possible stress position so that callers can apply their own resolution
strategy:

| Runtime                 | Typical resolution strategy                              |
| ----------------------- | -------------------------------------------------------- |
| Python (spaCy / Stanza) | Use morphological context from the NLP pipeline          |
| Web app                 | Pick first variant; show a tooltip with all alternatives |
| Plain script            | Pick first variant; done                                 |

### 2 · Parallel written ↔ phonetic representation

For every stress variant the engine returns **three parallel representations**:

| Level            | Field            | Example (`замок`, stress 0) |
| ---------------- | ---------------- | --------------------------- |
| Word             | `form`           | `замок`                     |
| Stressed word    | `stressed_form`  | `за́мок`                     |
| Syllabified word | `word_syllables` | `["за", "мок"]`             |
| IPA              | `ipa`            | `zɑmɔk`                     |
| Syllabified IPA  | `ipa_syllables`  | `["ˈzɑ", "mɔk"]`            |

The syllabified arrays are **positionally aligned**: `word_syllables[i]` is the
original graphemes for the same syllable as `ipa_syllables[i]`.

This makes it possible to:

- Reconstruct the original word: `word_syllables.join("")`
- Reconstruct the IPA: `ipa_syllables.map(s => s.replace("ˈ","")).join("")`
- Build highlighted / annotated views syllable-by-syllable

### 3 · Universal Dependencies morphology

All morphological data uses [Universal Dependencies](https://universaldependencies.org/)
naming conventions:

- **POS tags**: `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `AUX`,
  `CCONJ`, `SCONJ`, `PART`, `INTJ`, `PUNCT`, `SYM`, `NUM`, `X`
- **Feature keys**: `Case`, `Number`, `Gender`, `Animacy`, `Degree`, `Aspect`,
  `Tense`, `Mood`, `Person`, `VerbForm`, `Voice`, `Polarity`, …
- **Feature values**: `Nom`, `Gen`, `Dat`, `Acc`, `Ins`, `Loc`, `Voc` for Case; etc.

### 4 · Stable, additive evolution

Fields may be **added** in future versions without breaking existing consumers.
Removals or renames are breaking changes and require a major version bump.

---

## Core Types (Rust → JSON)

### `MorphReading`

One morphological analysis of a word form. A single stress variant may cover
multiple syncretised forms (e.g. `ру́ки` is both nominative plural and genitive
singular of `рука`).

```
MorphReading {
  pos:   string[]                   // UD POS tags, e.g. ["NOUN"]
  feats: { [key: string]: string[] } // UD feature map, e.g. {"Case":["Nom"],"Number":["Sing"]}
  lemma: string | null              // base form, or null if unknown
}
```

### `PhoneticToken`

Token-level detail produced by the 6-pass phonetic pipeline.

```
PhoneticToken {
  ipa:         string   // IPA symbol(s) for this phoneme
  source:      string   // original Cyrillic grapheme(s)
  type:        "Consonant" | "Vowel" | "Glide" | "Glottal"
  vowelIndex:  number   // 0-based index among vowels; -1 for non-vowels
  stressed:    boolean
  palatalized: boolean
}
```

### `StressReading`

A complete analysis for **one stress variant** of a word form.

```
StressReading {
  // ── Stress position ──────────────────────────────────────────────────────
  stressIndex:   number   // 0-based vowel index

  // ── Written representation ───────────────────────────────────────────────
  form:          string   // normalized form (lowercased, canonical apostrophe)
  stressedForm:  string   // form with combining acute U+0301 after stressed vowel
  wordSyllables: string[] // original graphemes per syllable, aligned with ipaSyllables

  // ── Phonetic representation ──────────────────────────────────────────────
  ipa:           string   // flat IPA string
  ipaSyllables:  string[] // IPA per syllable; stressed syllable prefixed with "ˈ"
  syllableCount: number
  tokens:        PhoneticToken[]  // token-level detail (for NLP / expert use)

  // ── Morphology ───────────────────────────────────────────────────────────
  morph: MorphReading[]   // all morphological analyses sharing this stress position
}
```

### `WordLookupResult`

The top-level return value of `lookup()`.

```
WordLookupResult {
  form:     string          // normalized query form
  readings: StressReading[] // all stress variants; empty array if word is unknown
}
```

---

## API Functions

### `lookup(word) → WordLookupResult`

Look up a Ukrainian word. Case-insensitive. Any apostrophe variant accepted
(`'`, `ʼ`, `'`).

Returns all stress variants with full phonetics and morphology.

```python
# Python
import ukrainian_stress
result = ukrainian_stress.lookup("замок")
# result.readings[0].stressed_form == "за́мок"
# result.readings[1].stressed_form == "замо́к"
```

```js
// JavaScript / TypeScript
import { lookup } from "ua-word-stress";
const result = lookup("замок");
// result.readings[0].stressedForm === 'за́мок'
```

### `mark(word) → string`

Return the word with a combining acute after the stressed vowel of the **first**
(most common) stress variant. Returns the word unchanged if unknown.

```python
ukrainian_stress.mark("мама")   # → "ма́ма"
ukrainian_stress.mark("__unk")  # → "__unk"
```

### `wordCount() → number`

Total word forms stored in the embedded dictionary.

### `transcribe(word, stressIndex) → TranscriptionResult`

Run the full 6-pass IPA pipeline for a word at a given stress index.
Useful when you already know the stress (e.g. from `lookup`) and want
only the phonetic output.

```
TranscriptionResult {
  word:        string          // normalized input
  ipa:         string          // flat IPA
  stressIndex: number
  tokens:      PhoneticToken[]
  syllables:   SyllableDetail[]
}

SyllableDetail {
  ipa:      string    // syllable IPA (no stress mark)
  word:     string    // original graphemes for this syllable
  stressed: boolean
  isOpen:   boolean   // ends on a vowel
}
```

---

## Resolution Strategies (consumer-side)

### Python with spaCy

```python
import ukrainian_stress
import spacy

nlp = spacy.load("uk_core_news_sm")

def resolve_stress(word: str, sentence: str) -> str:
    result = ukrainian_stress.lookup(word)
    if not result["readings"]:
        return word  # unknown

    doc = nlp(sentence)
    tok = next((t for t in doc if t.text.lower() == word.lower()), None)

    if tok is None or len(result["readings"]) == 1:
        return result["readings"][0]["stressed_form"]

    # Use spaCy POS + features to pick the right reading
    for reading in result["readings"]:
        for morph in reading["morph"]:
            if morph["pos"] and morph["pos"][0] == tok.pos_:
                return reading["stressed_form"]

    return result["readings"][0]["stressed_form"]
```

### Web (show first, tooltip for alternatives)

```ts
import { lookup } from "ua-word-stress";

function annotateWord(word: string): string {
  const { readings } = lookup(word);
  if (readings.length === 0) return word;
  const primary = readings[0].stressedForm;
  if (readings.length === 1) return primary;
  const tip = readings.map((r) => r.stressedForm).join(" / ");
  return `<span title="${tip}">${primary}</span>`;
}
```

---

## Syllabification & Reconstruction Examples

| Word        | `wordSyllables`                   | `ipaSyllables`                      |
| ----------- | --------------------------------- | ----------------------------------- |
| мама        | `["ма", "ма"]`                    | `["ˈmɑ", "mɑ"]`                     |
| батько      | `["бать", "ко"]`                  | `["ˈbɑtʲ", "kɔ"]`                   |
| університет | `["у", "ні", "вер", "си", "тет"]` | `["u", "nʲi", "ˈʋɛr", "sɪ", "tɛt"]` |

Reconstruction:

```
original word = wordSyllables.join("")
IPA (no marks) = ipaSyllables.map(s => s.replace(/^ˈ/, "")).join("")
stressed word  = stressedForm  (pre-computed)
```

---

## Notes on Token-Level Data

The `tokens` array is included for consumers that need phonetic detail beyond IPA
strings (e.g. a poetry meter analyser that needs vowel quality, or an ML model
that uses articulatory features). The field can safely be ignored for most uses.

The `UaSyllable.tokens` field available from `transcribe()` is the same data
split by syllable.

---

## Compatibility with `pl-stress-engine`

`pl-stress-engine` will export the same `WordLookupResult` / `StressReading` /
`MorphReading` shape. Differences:

| Field         | Ukrainian           | Polish           |
| ------------- | ------------------- | ---------------- |
| `ipa`         | Ukrainian IPA rules | Polish IPA rules |
| `feats`       | UD morphology       | UD morphology    |
| `morph[].pos` | UD UPOS             | UD UPOS          |
| Module name   | `ukrainian_stress`  | `polish_stress`  |
| npm package   | `ua-word-stress`    | `pl-word-stress` |
