# ua-stress-ml

LightGBM/ONNX stress predictor for out-of-vocabulary Ukrainian words.

This package contains **Luscinia** — a character-level machine-learning model that predicts which syllable carries primary stress in a Ukrainian word. It is an OOV-only predictor: for in-vocabulary words, use [`ua-word-stress`](https://www.npmjs.com/package/ua-word-stress) first (2.9 million word forms, zero dependencies).

## Quick start

```ts
import ort from "onnxruntime-web";
import { LusciniaPredictor } from "ua-stress-ml";

// Load the model (bundled separately — see "Model file" below)
const predictor = await LusciniaPredictor.fromUrl(
  new URL("./data/P3_0017_full.onnx.gz", import.meta.url).href,
  ort,
);

// Returns 0-based vowel index
const idx = await predictor.predict("університет"); // 4
//  у-н-і-в-е-р-с-и-т-е-т
//  0   1   2     3   4
//                        ^  stress on 5th vowel (е)

// Providing the POS tag improves accuracy (but is optional)
const idx2 = await predictor.predict("виходити", "VERB"); // 0

// Batch prediction
const results = await predictor.predictBatch(
  ["університет", "любити", "красивий"],
  "NOUN",
);

await predictor.dispose();
```

## What this package does

Luscinia uses **character-level** features only (suffix/prefix hashes, vowel positions, consonant clusters, and 20+ linguistic rule flags). It has no dictionary and no context window — it cannot resolve heteronyms (`замок` = castle vs. `замок` = lock) based on surrounding text. For heteronym disambiguation you need a POS tagger or a full NLP pipeline.

Intended use: as an OOV fallback _after_ a trie lookup fails, not as a standalone stress engine.

## Model file

The ONNX model (`P3_0017_full.onnx.gz`, ~30 MB compressed) is shipped inside this package at `<package>/data/P3_0017_full.onnx.gz`. You need to copy or serve it yourself depending on your build setup:

### Vite / Rollup

```ts
import modelUrl from "ua-stress-ml/data/P3_0017_full.onnx.gz?url";
const predictor = await LusciniaPredictor.fromUrl(modelUrl, ort);
```

### webpack (asset/resource)

```ts
// webpack.config.js — add rule for .onnx.gz
{ test: /\.onnx\.gz$/, type: "asset/resource" }
```

```ts
// app code
import modelUrl from "ua-stress-ml/data/P3_0017_full.onnx.gz";
const predictor = await LusciniaPredictor.fromUrl(modelUrl, ort);
```

### Vite + Web Worker

Running inference inside a Web Worker is the recommended approach for browser apps — it keeps the main thread free while the model loads and predicts.

Two pitfalls specific to this setup:

**1. Exclude `ua-stress-ml` from Vite's dep pre-bundler.**
Vite pre-bundles dependencies into hashed chunks. If you update the package, Vite will keep serving the old cached bundle until you manually clear `.vite` / `.q-cache`. Excluding the package forces Vite to always serve `node_modules/ua-stress-ml/dist/index.js` directly.

```ts
// vite.config.ts (or inside extendViteConf in quasar.config.ts)
export default defineConfig({
  optimizeDeps: {
    exclude: ["ua-stress-ml"],
  },
});
```

**2. Set `ort.env.wasm.wasmPaths` before the first `fromUrl()` call.**
Inside a Web Worker the base URL is the worker script's URL, not the page URL. ORT cannot resolve its `.wasm` files via relative paths, so you must point it to an explicit location — either a CDN or a local path served by your bundler.

```ts
// stress-worker.ts
import * as ort from "onnxruntime-web";
import { LusciniaPredictor } from "ua-stress-ml";

// Set BEFORE any LusciniaPredictor.fromUrl() call.
ort.env.wasm.wasmPaths =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";

let predictorPromise: Promise<LusciniaPredictor> | null = null;

function getPredictor(modelUrl: string): Promise<LusciniaPredictor> {
  predictorPromise ??= LusciniaPredictor.fromUrl(modelUrl, ort);
  return predictorPromise;
}

self.onmessage = async (
  e: MessageEvent<{ word: string; modelUrl: string }>,
) => {
  const { word, modelUrl } = e.data;
  try {
    const predictor = await getPredictor(modelUrl);
    const result = await predictor.predict(word);
    self.postMessage({ result });
  } catch (err) {
    self.postMessage({ result: null, error: String(err) });
  }
};
```

Spawn the worker from your main thread:

```ts
// main.ts
const worker = new Worker(new URL("./stress-worker.ts", import.meta.url), {
  type: "module",
});

worker.postMessage({
  word: "університет",
  modelUrl: "/models/luscinia.onnx.gz",
});
worker.onmessage = (e) => console.log(e.data.result); // 4
```

### Node.js

```ts
import { LusciniaPredictor } from "ua-stress-ml";
import ort from "onnxruntime-web";
import { fileURLToPath } from "url";
import { join, dirname } from "path";

const modelPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "node_modules/ua-stress-ml/data/P3_0017_full.onnx.gz",
);
const predictor = await LusciniaPredictor.fromFile(modelPath, ort);
```

## API

### `LusciniaPredictor`

```ts
class LusciniaPredictor {
  // Load from URL (browser / CDN). Supports .onnx and .onnx.gz
  static fromUrl(
    url: string,
    ort: OrtNamespace,
    options?: LusciniaOptions,
  ): Promise<LusciniaPredictor>;

  // Load from an already-decompressed ArrayBuffer
  static fromBuffer(
    buffer: ArrayBuffer,
    ort: OrtNamespace,
    options?: LusciniaOptions,
  ): Promise<LusciniaPredictor>;

  // Load from a gzip-compressed ArrayBuffer
  static fromGzipBuffer(
    gzipBuffer: ArrayBuffer,
    ort: OrtNamespace,
    options?: LusciniaOptions,
  ): Promise<LusciniaPredictor>;

  // Load from a file system path (Node.js 18+)
  static fromFile(
    path: string,
    ort: OrtNamespace,
    options?: LusciniaOptions,
  ): Promise<LusciniaPredictor>;

  // Predict the 0-based vowel index carrying primary stress
  predict(word: string, pos?: string): Promise<number>;

  // Batch inference — all words share the same pos tag
  predictBatch(words: string[], pos?: string): Promise<number[]>;

  // Release ONNX runtime resources
  dispose(): Promise<void>;
}

interface LusciniaOptions {
  executionProviders?: string[]; // default: ["wasm"]
}
```

### Feature utilities

```ts
// Build the 132-feature vector as a plain object (useful for debugging)
buildFeaturesUniversal(word: string, pos: string): Record<string, number>

// Build the 132-feature vector as a Float32Array (model input order)
featurise(word: string, pos?: string): Float32Array

// Feature names in model-input order
FEATURE_NAMES: readonly string[]  // length 132
EXPECTED_FEATURE_COUNT: number    // 132

// Low-level helpers
djb2Hash(s: string, mod: number): number
findVowels(word: string): number[]
```

## POS tags

POS tags use the [Universal Dependencies](https://universaldependencies.org/u/pos/) tagset. Supported values:

| Tag    | Meaning                   |
| ------ | ------------------------- |
| `NOUN` | Noun                      |
| `VERB` | Verb                      |
| `ADJ`  | Adjective                 |
| `ADV`  | Adverb                    |
| `NUM`  | Numeral                   |
| `PRON` | Pronoun                   |
| `DET`  | Determiner                |
| `PART` | Particle                  |
| `CONJ` | Conjunction               |
| `ADP`  | Adposition                |
| `INTJ` | Interjection              |
| `X`    | Unknown / other (default) |

## Combined usage with `ua-word-stress`

The recommended pattern is trie-first, ML-fallback:

```ts
import { UaStressTrie } from "ua-word-stress";
import { LusciniaPredictor } from "ua-stress-ml";
import ort from "onnxruntime-web";

const trie = await UaStressTrie.fromUrl("/ua_stress.ctrie.gz");
const luscinia = await LusciniaPredictor.fromUrl("/P3_0017_full.onnx.gz", ort);

async function getStress(word: string, pos = "X") {
  const result = trie.lookupFull(word);
  if (result !== null) return result; // fast path: trie hit

  // OOV — fall back to ML
  const vowelIdx = await luscinia.predict(word, pos);
  return {
    stress: vowelIdx,
    stresses: [vowelIdx],
    type: "unique" as const,
    uncertain: false,
  };
}
```

## Model details

| Property   | Value                            |
| ---------- | -------------------------------- |
| Model ID   | `luscinia-lgbm-str-ua-univ-v1.0` |
| Algorithm  | LightGBM → ONNX (opset 15)       |
| Features   | 132 (character-level)            |
| Classes    | 11 (vowel index 0–10)            |
| Checkpoint | `P3_0017_full`                   |
| Exported   | 2026-03-09                       |

## Peer dependency

`onnxruntime-web` is a **peer dependency** and must be installed separately. This keeps the main package free of large binaries and lets you control the version:

```bash
npm install onnxruntime-web ua-stress-ml
```

The package is compatible with `onnxruntime-web >= 1.18.0`.

## License

AGPL-3.0. See [LICENSE](./LICENSE).

The training data includes material from:

- [Ukrainian Wiktionary](https://uk.wiktionary.org/) (CC BY-SA)
- [kaikki.org](https://kaikki.org/) Ukrainian dictionary (CC BY-SA)
- [lang-uk](https://github.com/lang-uk) resources (MIT / CC BY)
