# ua-word-stress

Offline Ukrainian word stress lookup via a compact binary trie.  
**Zero dependencies.** Works in browsers (ESM) and Node.js.

---

## Database statistics

<!-- AUTO-GENERATED — do not edit this block manually -->
<!-- STATS_START -->

| Metric                                 | Value                |
| -------------------------------------- | -------------------- |
| Word forms                             | 2,874,507            |
| Variative stress (both valid)          | 221                  |
| Heteronyms (context-dependent stress)  | 114,362              |
| Trie nodes                             | 4,463,020            |
| Compressed size (`ua_stress.ctrie.gz`) | 9.4 MB               |
| Format                                 | ctrie-v2             |
| Last built                             | 2026-04-27T16:28:35Z |

<!-- STATS_END -->

---

## Installation

```bash
# pnpm (recommended)
pnpm add ua-word-stress

# npm
npm install ua-word-stress

# yarn
yarn add ua-word-stress
```

### Serving the data file

**Vite / Quasar / Nuxt (recommended):** import the data file as a URL — Vite resolves it to a content-hashed asset with no manual copy step:

```ts
// vite.config.ts (or quasar.config.js > build.extendViteConf)
// Required only if Vite doesn't recognise .ctrie.gz as an asset:
export default defineConfig({
  assetsInclude: ["**/*.ctrie.gz"],
});
```

```ts
import trieUrl from "ua-word-stress/data/ua_stress.ctrie.gz?url";
const trie = await UaStressTrie.fromUrl(trieUrl);
```

**Webpack / plain HTML:** copy the file to your public directory:

```bash
cp node_modules/ua-word-stress/data/ua_stress.ctrie.gz public/
```

---

## Quick start

### Browser (Vite / Webpack / plain ESM)

```ts
import { UaStressTrie } from "ua-word-stress";

const trie = await UaStressTrie.fromUrl("/ua_stress.ctrie.gz");

trie.lookup("університет"); // → 4
trie.mark("університет"); // → 'університе́т'
trie.lookupFull("замок"); // → { stress: 0, stresses: [0, 1], type: 'heteronym', uncertain: true }
trie.lookupFull("помилка"); // → { stress: 0, stresses: [0, 1], type: 'variative', uncertain: true }
trie.lookupFull("мама"); // → { stress: 0, stresses: [0], type: 'unique', uncertain: false }
```

### Node.js

```ts
import { UaStressTrie } from "ua-word-stress";

const trie = await UaStressTrie.fromFile("./data/ua_stress.ctrie.gz");
console.log(trie.lookup("читати")); // → 1
```

### Vue 3 / Quasar example

```ts
// composables/useStressTrie.ts
import { ref, shallowRef } from "vue";
import { UaStressTrie } from "ua-word-stress";

const trie = shallowRef<UaStressTrie | null>(null);
const loading = ref(false);

export function useStressTrie() {
  async function init(dataUrl = "/ua_stress.ctrie.gz") {
    loading.value = true;
    trie.value = await UaStressTrie.fromUrl(dataUrl);
    loading.value = false;
  }

  function markWord(word: string): string {
    return trie.value?.mark(word) ?? word;
  }

  return { trie, loading, init, markWord };
}
```

---

## API reference

### `UaStressTrie`

#### Factory methods

| Method                            | Description                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `UaStressTrie.fromUrl(url)`       | Fetch + parse `.ctrie.gz` from a URL (browser or Node.js). Auto-detects gzip. |
| `UaStressTrie.fromFile(path)`     | Load from a local file path (Node.js only).                                   |
| `UaStressTrie.fromBuffer(buffer)` | Parse from an already-decompressed `ArrayBuffer`.                             |

#### Instance methods

| Method                  | Return type            | Description                                                               |
| ----------------------- | ---------------------- | ------------------------------------------------------------------------- |
| `trie.lookup(word)`     | `number \| null`       | Stressed vowel index (0-based), or `null` if not in trie.                 |
| `trie.lookupFull(word)` | `LookupResult \| null` | Full result with all stress positions and type classification, or `null`. |
| `trie.mark(word)`       | `string \| null`       | Word with U+0301 combining accent on stressed vowel, or `null`.           |

#### Properties

| Property         | Type        | Description                                            |
| ---------------- | ----------- | ------------------------------------------------------ |
| `trie.wordCount` | `number`    | Total word forms in the database.                      |
| `trie.nodeCount` | `number`    | Total trie nodes.                                      |
| `trie.stats`     | `TrieStats` | Summary object with wordCount, nodeCount, gzSizeBytes. |

### Utility functions

```ts
import { applyStressMark, normaliseApostrophe } from "ua-word-stress";

applyStressMark("університет", 4); // → 'університе́т'
normaliseApostrophe("п'ять"); // → 'п\u02bcять'
```

### Types

```ts
interface LookupResult {
  /** Primary stress position — 0-based vowel index. */
  stress: number;
  /** All valid stress positions for this word form. */
  stresses: number[];
  /**
   * How to interpret multiple stress positions:
   * - `"unique"`    — one stress, no ambiguity.
   * - `"variative"` — both positions are always valid simultaneously
   *                    (e.g. по́милка / поми́лка). Display both or pick either.
   * - `"heteronym"` — positions correspond to different meanings or forms
   *                    (e.g. за́мок "lock" vs замо́к "castle").
   *                    Context is required to pick the right one.
   */
  type: "unique" | "variative" | "heteronym";
  /** @deprecated Use `type !== "unique"` instead. */
  uncertain: boolean;
}

interface TrieStats {
  wordCount: number;
  nodeCount: number;
  gzSizeBytes: number;
}
```

---

## Serving the data file

The `.ctrie.gz` binary is pre-compressed. Serve it with the `Content-Encoding: gzip`
header so the browser decompresses it transparently:

**nginx**

```nginx
location /ua_stress.ctrie.gz {
    add_header Content-Encoding gzip;
    add_header Content-Type application/octet-stream;
}
```

**Express.js**

```js
app.get("/ua_stress.ctrie.gz", (req, res) => {
  res.setHeader("Content-Encoding", "gzip");
  res.setHeader("Content-Type", "application/octet-stream");
  res.sendFile(path.join(__dirname, "public/ua_stress.ctrie.gz"));
});
```

**Without `Content-Encoding: gzip`** the library still works — it detects the
gzip magic bytes and decompresses client-side using the
[DecompressionStream API](https://developer.mozilla.org/en-US/docs/Web/API/DecompressionStream)
(supported in all modern browsers since 2022).

---

## Variative stress and heteronyms

`lookupFull()` classifies every word into one of three types:

### `type: "unique"` — no ambiguity

The vast majority of words. Use `result.stress` directly.

```ts
const r = trie.lookupFull("університет");
// { stress: 4, stresses: [4], type: 'unique', uncertain: false }
```

### `type: "variative"` — both stresses are always correct

These are words where both stress positions are accepted as standard in contemporary Ukrainian (e.g. _по́милка_ / _поми́лка_, _до́говір_ / _догові́р_). Both positions in `stresses[]` are valid — you can display both or pick either freely.

```ts
const r = trie.lookupFull("помилка");
// { stress: 0, stresses: [0, 1], type: 'variative', uncertain: true }
// → 'по́милка' or 'поми́лка' — both correct
```

### `type: "heteronym"` — different meanings or grammatical forms

These are words where stress distinguishes meaning or grammatical form (e.g. _за́мок_ "lock" vs _замо́к_ "castle"; _бло́хи_ gen.sg vs _блохи́_ nom.pl). Both positions are returned in `stresses[]`, but they are **not interchangeable** — the correct one depends on context.

```ts
const r = trie.lookupFull("замок");
// { stress: 0, stresses: [0, 1], type: 'heteronym', uncertain: true }
// stress 0 → 'за́мок' (lock)
// stress 1 → 'замо́к' (castle) — consumer must choose
```

Context-aware heteronym disambiguation requires sentence-level analysis and is outside the scope of this library. For poetry and annotation use cases, displaying both stress options is often the right choice.

### Out-of-vocabulary words

`lookup()` returns `null` for words not in the database (rare proper nouns, neologisms, typos). The **Luscinia** LightGBM model covers this case — it predicts stress from character-level features for any unknown Ukrainian word:

```ts
const trie = await UaStressTrie.fromUrl("/ua_stress.ctrie.gz");

function getStress(word: string): number | null {
  const result = trie.lookupFull(word);
  if (result) return result.stress; // trie hit — use primary stress
  return null; // OOV: hand off to Luscinia or skip
}
```

> **Note:** Luscinia is a character-n-gram model — it predicts stress from the letter pattern of an isolated word, **not** from sentence context. It resolves OOV words reliably but **cannot disambiguate heteronyms** (which are known words with context-dependent stress).

---

## Binary format (`.ctrie v2`)

The format is designed for minimal parse overhead — the JS constructor is O(alphabet_size) and each `lookup()` call is O(word_length).

```
Header         20 bytes
  [0..3]   magic         b'UKST' (0x55 0x4B 0x53 0x54)
  [4]      version       0x02
  [5]      reserved      0x00
  [6..9]   node_count    uint32 LE
  [10..13] word_count    uint32 LE
  [14..17] alphabet_size uint32 LE

Alphabet table   alphabet_size × 3 bytes
  [0]    char_byte   uint8   (codepoint if < 256, else 0xFF)
  [1..2] codepoint   uint16 LE

Node array   node_count × 8 bytes
  [0]    char_id     uint8   index into alphabet table
  [1]    stress      uint8   0xFF = not a word end; 0–10 = primary stressed vowel index
  [2]    flags       uint8   0x01 = FLAG_VARIATIVE (both stresses always valid)
                             0x02 = FLAG_HAS_CHILDREN
                             0x04 = FLAG_HETERONYM (meaning/form-dependent stress)
                             0x08 = FLAG_HAS_SECONDARY (stress2 byte is valid)
  [3]    stress2     uint8   secondary stress index (valid only if FLAG_HAS_SECONDARY set)
  [4..7] first_child uint32 LE  index of first child node (0 = no children)
```

Children of each node are **contiguous and sorted by char_id** in BFS order,
enabling a simple linear scan over at most 35 nodes (Ukrainian alphabet size).

The parser accepts both `0x01` (v1, backward compat) and `0x02` (current). v1 files have no secondary stress byte and only a single generic "uncertain" flag.

---

## Publishing to npm

```bash
cd packages/ua-stress-web

# Install dev deps
pnpm install

# 1. Build the TypeScript source
pnpm build

# 2. Run tests (requires data/ua_stress.ctrie.gz — build with Python first)
pnpm test

# 3. Publish (will run build + test automatically via prepublishOnly)
pnpm publish --access public
```

To publish as a scoped package, update `"name"` in `package.json`:

```json
"name": "@your-org/ua-word-stress"
```

---

## Building the database

The database is generated from the master SQLite source by the Python build script
at the project root:

```bash
# From VersaSenseBackend/
python build_web_stress_db.py
```

This writes `data/ua_stress.ctrie.gz` into this directory automatically.

---

## Data sources & attribution

The compiled `data/ua_stress.ctrie.gz` trie is built from three open sources:

### 1. lang-uk/ukrainian-word-stress

- **Repository:** https://github.com/lang-uk/ukrainian-word-stress
- **License:** [MIT](https://github.com/lang-uk/ukrainian-word-stress/blob/main/LICENSE) — Copyright (c) 2022 lang-uk
- **What we use:** The pre-built binary stress trie (`uk.stress.trie`) packaged with the library, covering ~400 k word forms with heteronym annotations.

### 2. lang-uk/ukrainian-word-stress-dictionary

- **Repository:** https://github.com/lang-uk/ukrainian-word-stress-dictionary
- **License:** No explicit open-source license. The data is derived from the ["Dictionaries of Ukraine"](https://lcorp.ulif.org.ua/dictua/) corpus published by the Ukrainian Language and Information Fund (ULIF), National Academy of Sciences of Ukraine — a public state institution. It is used here in accordance with the established convention in the Ukrainian open-source NLP community.
- **What we use:** `stress.txt` — ~2.9 M word forms with stress marks (U+0301).

### 3. kaikki.org — Wiktionary Ukrainian extract

- **Source:** https://kaikki.org/dictionary/Ukrainian/
- **Upstream:** [Wiktionary](https://en.wiktionary.org) contributors
- **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **What we use:** Parsed Ukrainian headword entries with stress annotations.

### 4. ua_variative_stressed_words (original)

- **Source:** Original curation — included in this repository.
- **License:** AGPL-3.0 (same as code).
- **What we use:** ~150 Ukrainian lemmas with freely variable stress (e.g. _алфавіт_, _договір_) for ambiguity annotation.

---

## License

This package uses **dual licensing**:

| Component                                      | License                                                         |
| ---------------------------------------------- | --------------------------------------------------------------- |
| TypeScript source code (`src/`, `dist/`)       | [AGPL-3.0](LICENSE)                                             |
| Compiled data file (`data/ua_stress.ctrie.gz`) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The data file carries CC BY-SA 4.0 because it is a derived database of Wiktionary content. Any derivative database you build from `ua_stress.ctrie.gz` must also be released under CC BY-SA 4.0 or a compatible license.

The AGPL-3.0 code license is compatible with the CC BY-SA 4.0 data: the code and data are separately licensed works.

See [LICENSE](LICENSE) for the full AGPL-3.0 text.
