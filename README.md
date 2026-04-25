# VersaSense Backend

Ukrainian NLP backend for word stress prediction, phonetic analysis, and text processing.

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
| Runtimes          | LightGBM (Python) · ONNX (browser via `onnxruntime-web`)     |
| Training data     | 2.7 M word forms                                             |
| License           | AGPL-3.0                                                     |

## Installation

```bash
conda env create -f environment.yml
conda activate verseSense-py312
pip install -e .
```

## Quick start — Python

```python
import lightgbm as lgb
import numpy as np
from src.stress_prediction.lightGbm.services.feature_service_universal import (
    build_features_universal,
)

MODEL_PATH = (
    "src/stress_prediction/lightGbm/artifacts/"
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

See [src/stress_prediction/lightGbm/documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md](src/stress_prediction/lightGbm/documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md)
for the full deployment guide (nginx / Express serving, batch inference, feature order).

## Project structure

```
src/
  stress_prediction/
    lightGbm/              # Luscinia model — training scripts, services, artifacts
      services/            # feature_service_universal.py and helpers
      artifacts/           # trained .lgb + .onnx files (Git LFS)
      documentation/       # LUSCINIA_LGBM_V1_DEPLOYMENT.md
  nlp/
    stress_service/        # dictionary-based stress lookup (LMDB trie)
    pipeline/              # NLP pipeline (stress resolver, ML stress resolver)
    phonetic/              # IPA transcription for Ukrainian
    tokenization_service/  # tokenizer
  data_management/         # data ingestion, parsing, merging pipeline
    sources/               # Kaikki, trie, txt, UA variative stressed words
    transform/             # merger, data unifier, cache utils
    export/                # training DB export (SQL schema)
  lemmatizer/              # Ukrainian lemmatizer
  utils/                   # shared utilities (apostrophe normalization, …)
tests/                     # pytest test suite mirroring src/
```

## Running tests

```bash
pytest
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
