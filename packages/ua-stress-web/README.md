# ua-stress-trie

Offline Ukrainian word stress lookup via a compact binary trie.  
**Zero dependencies.** Works in browsers (ESM) and Node.js.

---

## Database statistics

<!-- AUTO-GENERATED — do not edit this block manually -->
<!-- STATS_START -->
| Metric | Value |
|--------|-------|
| Word forms | 2,874,507 |
| Heteronyms (context-dependent stress) | 114,583 |
| Trie nodes | 4,463,020 |
| Compressed size (`ua_stress.ctrie.gz`) | 9.37 MB |
| Format | ctrie-v1 |
| Last built | 2026-04-25T13:49:36Z |
<!-- STATS_END -->

---

## Installation

```bash
# pnpm (recommended)
pnpm add ua-stress-trie

# npm
npm install ua-stress-trie

# yarn
yarn add ua-stress-trie
```

Copy the data file to your project's public directory:

```bash
cp node_modules/ua-stress-trie/data/ua_stress.ctrie.gz public/
```

---

## Quick start

### Browser (Vite / Webpack / plain ESM)

```ts
import { UaStressTrie } from "ua-stress-trie";

const trie = await UaStressTrie.fromUrl("/ua_stress.ctrie.gz");

trie.lookup("університет"); // → 4
trie.mark("університет"); // → 'університе́т'
trie.lookupFull("замок"); // → { stress: 0, uncertain: true }
```

### Node.js

```ts
import { UaStressTrie } from "ua-stress-trie";

const trie = await UaStressTrie.fromFile("./data/ua_stress.ctrie.gz");
console.log(trie.lookup("читати")); // → 1
```

### Vue 3 / Quasar example

```ts
// composables/useStressTrie.ts
import { ref, shallowRef } from "vue";
import { UaStressTrie } from "ua-stress-trie";

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

| Method                  | Return type            | Description                                                     |
| ----------------------- | ---------------------- | --------------------------------------------------------------- |
| `trie.lookup(word)`     | `number \| null`       | Stressed vowel index (0-based), or `null` if not in trie.       |
| `trie.lookupFull(word)` | `LookupResult \| null` | Stress index + heteronym flag, or `null`.                       |
| `trie.mark(word)`       | `string \| null`       | Word with U+0301 combining accent on stressed vowel, or `null`. |

#### Properties

| Property         | Type        | Description                                            |
| ---------------- | ----------- | ------------------------------------------------------ |
| `trie.wordCount` | `number`    | Total word forms in the database.                      |
| `trie.nodeCount` | `number`    | Total trie nodes.                                      |
| `trie.stats`     | `TrieStats` | Summary object with wordCount, nodeCount, gzSizeBytes. |

### Utility functions

```ts
import { applyStressMark, normaliseApostrophe } from "ua-stress-trie";

applyStressMark("університет", 4); // → 'університе́т'
normaliseApostrophe("п'ять"); // → 'п\u02bcять'
```

### Types

```ts
interface LookupResult {
  stress: number; // 0-based vowel index
  uncertain: boolean; // true = heteronym, context needed
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

## Heteronyms and ONNX fallback

Words with `uncertain: true` have **multiple valid stress positions** depending
on grammatical context (e.g. _за́мок_ "lock" vs _замо́к_ "castle"). For those
words — and for out-of-vocabulary words (`lookup()` returns `null`) — use the
**Luscinia ONNX model** for context-aware prediction:

```ts
import { UaStressTrie } from "ua-stress-trie";
import { InferenceSession, Tensor } from "onnxruntime-web";

const trie = await UaStressTrie.fromUrl("/ua_stress.ctrie.gz");
const onnx = await InferenceSession.create("/luscinia.onnx");

async function resolveStress(word: string): Promise<number> {
  const result = trie.lookupFull(word);

  if (result && !result.uncertain) {
    return result.stress; // fast trie hit
  }

  return await lusciniaPredict(onnx, word); // ONNX fallback
}
```

---

## Binary format (`.ctrie v1`)

The format is designed for minimal parse overhead — the JS constructor is O(alphabet_size) and each `lookup()` call is O(word_length).

```
Header         20 bytes
  [0..3]   magic        b'UKST' (0x55 0x4B 0x53 0x54)
  [4]      version      0x01
  [5]      reserved     0x00
  [6..9]   node_count   uint32 LE
  [10..13] word_count   uint32 LE
  [14..17] alphabet_size uint32 LE

Alphabet table   alphabet_size × 3 bytes
  [0]    char_byte   uint8  (codepoint if < 256, else 0xFF)
  [1..2] codepoint   uint16 LE

Node array   node_count × 8 bytes
  [0]    char_id     uint8   index into alphabet table
  [1]    stress      uint8   0xFF = not a word end; 0–10 = stressed vowel index
  [2]    flags       uint8   bit 0 = heteronym; bit 1 = has_children
  [3]    reserved    uint8
  [4..7] first_child uint32 LE  index of first child node (0 = no children)
```

Children of each node are **contiguous and sorted by char_id** in BFS order,
enabling O(1) child access with a simple linear scan over at most 35 nodes
(Ukrainian alphabet size).

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
"name": "@your-org/ua-stress-trie"
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

## License

AGPL-3.0 — see [LICENSE](LICENSE).
