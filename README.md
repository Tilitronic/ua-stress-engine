# ua-stress-engine

Ukrainian word stress engine — dictionary lookup, ML prediction, and a zero-dependency JS/TS trie package.

The centrepiece is **Luscinia** — a LightGBM model that predicts the stressed vowel in
any Ukrainian word with **99.44 % accuracy** across all syllable counts.
The model is also exported to ONNX for browser-side inference via `onnxruntime-web`.

## Highlights

|                   |                                                              |
| ----------------- | ------------------------------------------------------------ |
| Model             | `luscinia-lgbm-str-ua-univ-v1`                               |
| Task              | Ukrainian word stress prediction (multiclass, vowel-ordinal) |
| Accuracy          | 99.44 % (sanity sample) · 192 / 197 hand-checked             |
| Syllable coverage | 2 – 10 + syllable words, single universal model              |
| Features          | 132 linguistic / hash features                               |
| Runtimes          | lightgbm (Python) · ONNX (browser via `onnxruntime-web`)     |
| Training data     | 2.7 M word forms                                             |
| License           | AGPL-3.0                                                     |

## Installation

```bash
conda env create -f environment.yml
conda activate verseSense-py312
pip install -e .
```

## Quick start — Python

### Low-level: raw model prediction

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
    """Returns the word with a combining acute accent on the stressed vowel."""
    feat = build_features_universal(word, pos)
    X = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
    vowel_idx = int(bst.predict(X).argmax(axis=1)[0])
    vpos = [i for i, c in enumerate(word.lower()) if c in VOWELS]
    cp = vpos[vowel_idx]
    return word[: cp + 1] + "\u0301" + word[cp + 1 :]

print(predict_stress("університет", "NOUN"))  # → університе́т
print(predict_stress("читати",      "VERB"))  # → чита́ти
```

### Full pipeline (LMDB + ML fallback)

```python
from src.stress_resolver.resolver_factory import create_pipeline_kwargs
from src.stress_resolver.pipeline import UkrainianPipeline

# Auto mode: uses LMDB lookup + LightGBM fallback if model is available,
# silently falls back to LMDB-only if lightgbm is not installed.
pipeline = UkrainianPipeline(**create_pipeline_kwargs())

doc = pipeline.process("Мама варила борщ на кухні.")
for sentence in doc.sentences:
    for token in sentence.tokens:
        print(f"{token.text:15} {token.stress_pattern}")
```

> **POS tags** — use Universal Dependencies tags:
> `NOUN VERB ADJ ADV PRON DET NUM PART CCONJ X`.
> Pass `"X"` when POS is unknown.
>
> **Apostrophes** — pre-normalise to U+02BC (`ʼ`) using
> `src/utils/normalize_apostrophe.py` before calling the model.

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

## Modules

| Module                         | Path                              | What it does                                                       |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------------ |
| **Dictionary resolver** (LMDB) | `src/nlp/stress_service/`         | Sub-millisecond stress lookup across 2.86 M word forms             |
| **ML resolver** (LightGBM)     | `src/stress_prediction/lightgbm/` | Luscinia model — 99.44 % accuracy, 132 features, ONNX export       |
| **NLP pipeline**               | `src/stress_resolver/`            | spaCy tokenization → LMDB lookup → ML fallback → IPA transcription |
| **JS trie package**            | `packages/ua-stress-web/`         | `ua-stress-trie` — zero-dependency browser/Node trie (~9 MB)       |
| **Data management**            | `src/data_management/`            | Source parsers, master SQLite DB builder, trie exporter            |

## Project structure

```
ua-stress-engine/
├── packages/
│   └── ua-stress-web/             # ua-stress-trie npm package (TypeScript, zero deps)
│       ├── src/                   # UaStressTrie.ts, types.ts, utils.ts
│       ├── tests/
│       └── package.json
├── src/
│   ├── stress_resolver/           # Python NLP pipeline + resolver chain
│   │   ├── pipeline.py            # UkrainianPipeline
│   │   ├── stress_resolver.py     # LMDB-based resolver (morphology matching)
│   │   ├── ml_stress_resolver.py  # LightGBM-based resolver
│   │   └── resolver_factory.py    # Auto-configure DB + optional ML resolver
│   ├── nlp/
│   │   ├── stress_service/        # LMDB stress lookup (2.86 M entries)
│   │   ├── phonetic/              # IPA transcription
│   │   └── tokenization_service/  # spaCy tokenizer wrapper
│   ├── stress_prediction/
│   │   └── lightgbm/              # Luscinia model, training scripts, services, artifacts
│   └── data_management/
│       ├── sources/               # Source parsers (kaikki, trie, txt, variative)
│       ├── transform/             # Master DB builder (SQLite, 681 MB)
│       └── export/
│           └── web_stress_db/     # Binary trie builder (Python → ua-stress-trie data)
├── build_master_db.py             # Build master SQLite from all sources
├── build_web_stress_db.py         # Build + export binary trie → packages/ua-stress-web/data/
├── analyze_master_db.py           # Inspect master DB
├── analyze_stress_service.py      # Inspect LMDB
├── analyze_luscinia.py            # Inspect LightGBM model
└── tests/
    └── src/
        ├── stress_resolver/       # Pipeline + resolver tests (30 tests)
        ├── stress_prediction/     # LightGBM model tests (44+ tests)
        ├── data_management/       # Source parser + DB tests (21 tests)
        └── nlp/                   # Stress service tests (24 tests)
```

## Running tests

```bash
# All tests (requires verseSense-py312 for LightGBM and spaCy)
conda activate verseSense-py312
python -m pytest tests/ -q

# JS/TS trie package
cd packages/ua-stress-web && pnpm test
```

    sources/               # Kaikki, trie, txt, UA variative stressed words
    transform/             # merger, data unifier, cache utils
    export/                # training DB export (SQL schema)

lemmatizer/ # Ukrainian lemmatizer
utils/ # shared utilities (apostrophe normalization, …)
tests/ # pytest test suite mirroring src/

````

## Running tests

```bash
pytest
````

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
