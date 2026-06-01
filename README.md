# ua-stress-engine

Ukrainian word stress engine — dictionary lookup with full IPA transcription, ML stress prediction, and published packages for Python, Node.js, and the browser.

The centrepiece is **Luscinia** — a LightGBM model that predicts the stressed vowel in
any Ukrainian word with **99.44 % accuracy** across all syllable counts.
The model is also exported to ONNX for browser-side inference via `onnxruntime-web`.

## Published packages

| Package               | Registry                                                 | Source                    | Description                                             |
| --------------------- | -------------------------------------------------------- | ------------------------- | ------------------------------------------------------- |
| `ua-word-stress`      | [npm](https://www.npmjs.com/package/ua-word-stress)      | `packages/ua-stress-web/` | Zero-dependency TypeScript trie (~9 MB, browser + Node) |
| `ua-word-stress-wasm` | [npm](https://www.npmjs.com/package/ua-word-stress-wasm) | `crates/wasm/`            | Rust/WASM — full IPA, morphology, batch API             |
| `ua-stress-ml`        | [npm](https://www.npmjs.com/package/ua-stress-ml)        | `packages/ua-stress-ml/`  | ONNX Luscinia predictor for OOV words (browser/worker)  |
| `ua-stress-engine`    | [PyPI](https://pypi.org/project/ua-stress-engine/)       | `crates/python/`          | PyO3 extension (`ukrainian_stress`) — same API as WASM  |
| `luscinia`            | [PyPI](https://pypi.org/project/luscinia/)               | `packages/luscinia/`      | Python ONNX Luscinia predictor (OOV fallback)           |

## Highlights

|                   |                                                              |
| ----------------- | ------------------------------------------------------------ |
| Model             | `luscinia-lgbm-str-ua-univ-v1`                               |
| Task              | Ukrainian word stress prediction (multiclass, vowel-ordinal) |
| Accuracy          | 99.44 % (sanity sample) · 192 / 197 hand-checked             |
| Syllable coverage | 2 – 10 + syllable words, single universal model              |
| Features          | 132 linguistic / hash features                               |
| Runtimes          | lightgbm (Python) · ONNX (browser via `onnxruntime-web`)     |
| Training data     | 2.875 M word forms                                           |
| License           | AGPL-3.0                                                     |

## Installation

### JavaScript / TypeScript

```bash
# Trie-based lookup (browser + Node, no WASM)
npm install ua-word-stress

# Full engine — IPA, morphology, batch API (WASM, bundler required)
npm install ua-word-stress-wasm
```

### Python

Published packages:

```bash
pip install ua-stress-engine
pip install luscinia
```

The package is a compiled Rust extension (PyO3 + maturin). Runtime Python dependencies:

| Extra    | Packages                       | Purpose                                    |
| -------- | ------------------------------ | ------------------------------------------ |
| _(core)_ | none                           | Dictionary lookup + IPA via Rust extension |
| `ml`     | `lightgbm>=4.0`, `numpy>=1.24` | Luscinia LightGBM resolver                 |
| `nlp`    | `spacy>=3.7`                   | spaCy tokenization pipeline                |
| `full`   | all of the above               | Everything                                 |

For local development (requires [Rust toolchain](https://rustup.rs/) and `maturin`):

```bash
pip install maturin
pip install -e '.[full]'   # builds the Rust extension in-place
```

## Quick start — JavaScript / TypeScript

### Trie-based lookup (`ua-word-stress`)

Pure TypeScript, no WASM, works in any environment:

```ts
import { UaStressTrie } from "ua-word-stress";

const trie = new UaStressTrie();
trie.mark("університет"); // → 'університе́т'
trie.lookup("замок"); // → 0 (first syllable — замок-lock)
trie.markBatch(["мама", "тато"]); // → ['ма́ма', 'та́то']
```

### Full WASM engine (`ua-word-stress-wasm`)

Rust/WASM with IPA transcription, morphology, and batch API. No `init()` call needed — the dictionary loads automatically at module import (bundler target):

```ts
import { mark, lookup, stressIndex, transcribe } from "ua-word-stress-wasm";

mark("університет"); // → 'університе́т'
stressIndex("мама"); // → 0  (0-based syllable index)

const r = lookup("замок");
r.readings[0].stressedForm; // → 'за́мок'
r.readings[0].ipa; // → 'zɑmɔk'
r.readings[0].syllableIndex; // → 0
r.readings[1].stressedForm; // → 'замо́к'  (heteronym)

transcribe("слово", 0); // → { ipa: 'slɔwɔ', ipaSyllables: ['ˈslɔ', 'wɔ'], … }
```

See the [WASM package README](crates/wasm/pkg/README.md) for the full API reference.

## Quick start — Python

### Dictionary + IPA (Rust extension)

```python
import ukrainian_stress

ukrainian_stress.mark('університет')   # → 'університе́т'

r = ukrainian_stress.lookup('замок')
r['readings'][0]['stressed_form']      # → 'за́мок'
r['readings'][0]['ipa']               # → 'zɑmɔk'
r['readings'][0]['syllable_index']    # → 0
```

### Full pipeline (Rust dict + LightGBM ML fallback)

Requires `pip install -e '.[full]'`:

```python
from src.stress_resolver.resolver_factory import create_pipeline_kwargs
from src.stress_resolver.pipeline import UkrainianPipeline

pipeline = UkrainianPipeline(**create_pipeline_kwargs())

doc = pipeline.process("Мама варила борщ на кухні.")
for sentence in doc.sentences:
    for token in sentence.tokens:
        print(f"{token.text:15} {token.stress_pattern}")
```

### Raw Luscinia prediction (LightGBM)

```python
import lightgbm as lgb
import numpy as np
from src.stress_prediction.lightgbm.services.feature_service_universal import (
    build_features_universal,
)

MODEL_PATH = (
    "src/stress_prediction/lightgbm/artifacts/"
    "luscinia-lgbm-str-ua-univ-v1/P3_0017_FINAL_FULLDATA/P3_0017_full.lgb"
)
bst = lgb.Booster(model_file=MODEL_PATH)

VOWELS = set("аеєиіїоуюя")

def predict_stress(word: str, pos: str = "NOUN") -> str:
    feat = build_features_universal(word, pos)
    X = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
    vowel_idx = int(bst.predict(X).argmax(axis=1)[0])
    vpos = [i for i, c in enumerate(word.lower()) if c in VOWELS]
    cp = vpos[vowel_idx]
    return word[: cp + 1] + "\u0301" + word[cp + 1 :]

print(predict_stress("університет", "NOUN"))  # → університе́т
print(predict_stress("читати",      "VERB"))  # → чита́ти
```

> **POS tags** — use Universal Dependencies tags: `NOUN VERB ADJ ADV PRON DET NUM PART CCONJ X`. Pass `"X"` when POS is unknown.

## Quick start — browser (ONNX)

The 30 MB gzip-compressed ONNX artifact (`P3_0017_full.onnx.gz`) is stored in
Git LFS. Serve it with `Content-Encoding: gzip` so browsers decompress it
transparently.

```js
import * as ort from "onnxruntime-web";

const session = await ort.InferenceSession.create(
  "/models/P3_0017_full.onnx.gz",
);

// Build a Float32Array of 132 features (see manifest.json for order)
const tensor = new ort.Tensor("float32", featureArray, [1, 132]);
const results = await session.run({ float_input: tensor });
const vowelIndex = Number(results["label"].data[0]);
```

See [src/stress_prediction/lightgbm/documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md](src/stress_prediction/lightgbm/documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md)
for the full deployment guide (nginx / Express serving, batch inference, feature order).

## API status

Canonical API contract is documented in [documentation/API_DESIGN.md](documentation/API_DESIGN.md).
Current runtime/API mapping:

- `ua-word-stress` (npm): trie lookup API (`lookup`, `lookupFull`, `mark`, batch methods)
- `ua-word-stress-wasm` (npm): full Rust API (`lookup`, `mark`, `stressIndex`, `transcribe`)
- `ua-stress-engine` (PyPI): Python binding module `ukrainian_stress` with the same data shape as WASM
- `ua-stress-ml` (npm) and `luscinia` (PyPI): ML OOV fallback predictors (132-feature Luscinia ONNX model)

## Modules

| Module                          | Path                              | What it does                                                               |
| ------------------------------- | --------------------------------- | -------------------------------------------------------------------------- |
| **`ua-word-stress`** (npm)      | `packages/ua-stress-web/`         | Zero-dependency TypeScript trie — `mark`, `lookup`, batch API              |
| **`ua-word-stress-wasm`** (npm) | `crates/wasm/`                    | Rust/WASM — IPA, morphology, batch API, no init() required                 |
| **`ua-stress-ml`** (npm)        | `packages/ua-stress-ml/`          | Browser/worker ONNX Luscinia predictor for OOV stress                      |
| **`ukrainian_stress`** (Python) | `crates/python/`                  | PyO3 extension — same API as WASM, for Python                              |
| **`luscinia`** (Python)         | `packages/luscinia/`              | Python ONNX Luscinia predictor package (PyPI)                              |
| **Rust core**                   | `crates/core/`                    | Dictionary embed, phonetic pipeline, syllabifier (shared by WASM + Python) |
| **ML resolver** (LightGBM)      | `src/stress_prediction/lightgbm/` | Luscinia model — 99.44 % accuracy, 132 features, ONNX export               |
| **NLP pipeline**                | `src/stress_resolver/`            | spaCy tokenization → Rust dict lookup → ML fallback                        |
| **Data management**             | `src/data_management/`            | Source parsers, master SQLite DB builder, binary trie exporter             |

## Project structure

```
ua-stress-engine/
├── crates/
│   ├── core/                      # Rust core library (dict embed, phonetics, syllabifier)
│   ├── wasm/                      # ua-word-stress-wasm (wasm-pack, bundler target)
│   │   ├── src/lib.rs
│   │   └── pkg/                   # built npm package (gitignored except README + package.json)
│   ├── python/                    # ukrainian_stress PyO3 extension (maturin)
│   │   └── src/lib.rs
│   └── builder/                   # CLI tool to compile the embedded binary dictionary
├── packages/
│   └── ua-stress-web/             # ua-word-stress npm package (TypeScript, zero deps)
│       ├── src/                   # UaStressTrie.ts, types.ts, utils.ts
│       ├── tests/
│       └── package.json
├── src/
│   ├── stress_resolver/           # Python NLP pipeline + resolver chain
│   │   ├── pipeline.py            # UkrainianPipeline
│   │   ├── stress_resolver.py     # Rust-extension-based resolver
│   │   ├── ml_stress_resolver.py  # LightGBM-based resolver
│   │   └── resolver_factory.py    # Auto-configure resolver chain
│   ├── nlp/
│   │   ├── stress_service/        # Stress lookup wrapper
│   │   ├── phonetic/              # IPA transcription (Python side)
│   │   └── tokenization_service/  # spaCy tokenizer wrapper
│   ├── stress_prediction/
│   │   └── lightgbm/              # Luscinia model, training scripts, services, artifacts
│   └── data_management/
│       ├── sources/               # Source parsers (kaikki, trie, txt, variative)
│       ├── transform/             # Master DB builder (SQLite)
│       └── export/
│           └── web_stress_db/     # Binary .ctrie builder → packages/ua-stress-web/data/
├── build_master_db.py             # Build master SQLite from all sources
├── build_web_stress_db.py         # Build + export binary trie
├── pyproject.toml                 # maturin build config (points to crates/python/)
└── tests/
    └── src/
        ├── stress_resolver/       # Pipeline + resolver tests
        ├── stress_prediction/     # LightGBM model tests
        ├── data_management/       # Source parser + DB tests
        └── nlp/                   # Stress service tests
```

## Data sources

The embedded dictionary is compiled from five open Ukrainian stress resources:

| Source                                                                                                                                                   | License                                         | Entries              | Notes                                    |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | -------------------- | ---------------------------------------- |
| [kaikki.org Ukrainian](https://kaikki.org/dictionary/Ukrainian/) — Wiktionary extract                                                                    | CC BY-SA 4.0                                    | ~2 M inflected forms | POS + full morphology                    |
| [lang-uk/ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) — marisa-trie                                                          | MIT                                             | ~2.9 M word forms    | compact trie with morph tags             |
| [lang-uk/ukrainian-word-stress-dictionary](https://github.com/lang-uk/ukrainian-word-stress-dictionary) — text dict                                      | see upstream                                    | ~2.9 M word forms    | based on ULIF / NASU corpora             |
| [bakustarver/ukr-dictionaries-list-opensource](https://github.com/bakustarver/ukr-dictionaries-list-opensource) — SUM11 DiktJson (`ukr-ukr_SUM-11_or_1`) | public domain (original SUM-11), digitised JSON | ~127 K lemmas        | classic 11-volume explanatory dictionary |
| `ua_variative_stressed_words` — curated free-variant list                                                                                                | original work                                   | ~150 lemmas          | marks freely variable stress             |

All five sources are merged into a single master SQLite (~680 MB) and then compiled into the embedded binary (`ua_stress.bin.bz2`) shipped inside the Rust crates.

## Running tests

```bash
# Python tests (requires ml + nlp extras installed)
python -m pytest tests/ -q

# TypeScript trie package
cd packages/ua-stress-web && pnpm test

# WASM package
cd crates/wasm && wasm-pack test --node
```

## Large files (Git LFS)

The following binary artifacts are stored in Git LFS:

| File                   | Size   |
| ---------------------- | ------ |
| `P3_0017_full.lgb`     | 259 MB |
| `P3_0017_full.onnx`    | 185 MB |
| `P3_0017_full.onnx.gz` | 30 MB  |
| `stress.lmdb`          | varies |

## License

[AGPL-3.0](LICENSE)
