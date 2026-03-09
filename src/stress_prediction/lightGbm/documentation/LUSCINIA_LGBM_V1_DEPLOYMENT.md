# Luscinia v1.0 — Deployment Guide

**Model:** `luscinia-lgbm-str-ua-univ-v1`  
**Accuracy:** 99.44% sanity · 192/197 hand-checked · 100% ONNX class agreement  
**Output:** 0-based vowel index of the stressed vowel from the start of the word

---

## Table of Contents

1. [What the model outputs](#1-what-the-model-outputs)
2. [Python service (backend)](#2-python-service-backend)
3. [Web browser (ONNX)](#3-web-browser-onnx)
4. [Artifacts reference](#4-artifacts-reference)

---

## 1. What the model outputs

The model predicts a **vowel index** — the 0-based position among Ukrainian vowels
(`аеєиіїоуюя`) counting from the start of the word.

```
Word:      у  н  і  в  е  р  с  и  т  е  т
           ^     ^     ^        ^     ^
Vowels:    0     1     2        3     4   ← predicted = 4

→ stressed vowel is the 5th one: університЕт
```

To convert vowel index to a character position:

```python
VOWELS = set("аеєиіїоуюя")

def vowel_positions(word: str) -> list[int]:
    """Returns character indices of all vowels in the word."""
    return [i for i, c in enumerate(word.lower()) if c in VOWELS]

def stressed_char_index(word: str, vowel_idx: int) -> int:
    """Returns character index of the stressed vowel."""
    return vowel_positions(word)[vowel_idx]
```

---

## 2. Python service (backend)

### Installation

```bash
pip install lightgbm numpy
```

### Minimal usage

```python
import lightgbm as lgb
import numpy as np
import sys

sys.path.insert(0, "/path/to/VersaSenseBackend")  # project root
from src.stress_prediction.lightGbm.services.feature_service_universal import (
    build_features_universal,
)

# ── Load once at service startup (takes ~2 s, model is 259 MB) ────────────────
MODEL_PATH = (
    "src/stress_prediction/lightGbm/artifacts/"
    "luscinia-lgbm-str-ua-univ-v1/P3_0017_FINAL_FULLDATA/P3_0017_full.lgb"
)
bst = lgb.Booster(model_file=MODEL_PATH)

# ── Predict stress for a single word ─────────────────────────────────────────
def predict_stress(word: str, pos: str = "NOUN") -> int:
    """
    Returns 0-based vowel index of the stressed vowel.

    Parameters
    ----------
    word : str
        Ukrainian word, apostrophes normalised to U+02BC.
    pos : str
        Universal POS tag: NOUN, VERB, ADJ, ADV, PRON, DET, NUM, PART, CCONJ, X
        Use "X" when POS is unknown — the model gracefully degrades.

    Returns
    -------
    int
        0-based index of the stressed vowel from the start of the word.
        e.g. 0 → first vowel, 1 → second vowel, etc.
    """
    feat = build_features_universal(word, pos)
    X    = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
    probs = bst.predict(X)            # shape (1, 11)
    return int(probs.argmax(axis=1)[0])


# ── Example ───────────────────────────────────────────────────────────────────
VOWELS = set("аеєиіїоуюя")

def apply_stress_mark(word: str, pos: str = "NOUN") -> str:
    """Returns the word with a combining acute accent on the stressed vowel."""
    idx  = predict_stress(word, pos)
    vpos = [i for i, c in enumerate(word.lower()) if c in VOWELS]
    char_pos = vpos[idx]
    return word[:char_pos + 1] + "\u0301" + word[char_pos + 1:]   # U+0301 = ́

print(apply_stress_mark("університет", "NOUN"))   # → університе́т
print(apply_stress_mark("читати",      "VERB"))   # → чита́ти
print(apply_stress_mark("місто",       "NOUN"))   # → мі́сто
```

### Batch prediction (much faster)

```python
words_and_pos = [
    ("університет", "NOUN"),
    ("читати",      "VERB"),
    ("місто",       "NOUN"),
    ("вода",        "NOUN"),
]

# Build feature matrix for all words at once
feat_matrix = np.array(
    [list(build_features_universal(w, p).values()) for w, p in words_and_pos],
    dtype=np.float32,
)                                     # shape (N, 132)

all_probs   = bst.predict(feat_matrix)   # shape (N, 11)
all_indices = all_probs.argmax(axis=1)   # shape (N,)  ← vowel indices

for (word, _), idx in zip(words_and_pos, all_indices):
    print(word, "→ vowel", idx)
```

### Important notes

| Topic               | Detail                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Feature count**   | Always 132. Never pass a raw array — always go through `build_features_universal()`.                                                                         |
| **Feature order**   | The dict returned by `build_features_universal()` is an ordered dict. `list(feat.values())` produces features in the exact order the model was trained with. |
| **POS tags**        | Use Universal Dependencies tags: `NOUN VERB ADJ ADV PRON DET NUM PART CCONJ X`. If you don't know POS, pass `"X"`.                                           |
| **Apostrophes**     | Pre-normalise to U+02BC (`ʼ`). Use `src/utils/normalize_apostrophe.py`.                                                                                      |
| **Thread safety**   | `lgb.Booster.predict()` is thread-safe. One `bst` instance can serve all threads.                                                                            |
| **Max vowel index** | Model outputs classes 0–10. Words with > 11 vowels are truncated at vowel 10.                                                                                |

---

## 3. Web browser (ONNX)

### Artifacts

```
artifacts/luscinia-lgbm-str-ua-univ-v1/web/
├── P3_0017_full.onnx.gz   ← 30 MB  — serve this to the browser
├── P3_0017_full.onnx      ← 185 MB — keep as source of truth, don't serve directly
└── manifest.json          ← serving metadata
```

### Serving the file

The browser receives a **pre-compressed** file. Configure your server to send it
with the correct headers so the browser decompresses it transparently:

**nginx:**

```nginx
location /models/ {
    root /srv;
    add_header Content-Encoding gzip;
    add_header Content-Type application/octet-stream;
    # Do NOT use gzip_static here — file is already gzipped
}
```

**Express (Node.js):**

```js
app.get("/models/P3_0017_full.onnx.gz", (req, res) => {
  res.setHeader("Content-Encoding", "gzip");
  res.setHeader("Content-Type", "application/octet-stream");
  res.sendFile("/path/to/P3_0017_full.onnx.gz");
});
```

### Browser-side inference

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

```js
import * as ort from "onnxruntime-web";

// ── Load once ─────────────────────────────────────────────────────────────────
// The browser downloads 30 MB, decompresses it (Content-Encoding: gzip),
// then onnxruntime-web initialises the session.
const session = await ort.InferenceSession.create(
  "/models/P3_0017_full.onnx.gz",
);

// ── Input ─────────────────────────────────────────────────────────────────────
// You must build the 132-element feature vector.
// Option A (recommended): call your Python API endpoint that returns the feature
//                         vector as JSON, then pass it here.
// Option B: port build_features_universal() to JavaScript (see notes below).

const featureArray = new Float32Array(132); // fill with feature values
// featureArray[0] = feat["syllable_count"]  etc. — order from manifest.json

const tensor = new ort.Tensor("float32", featureArray, [1, 132]);

// ── Run inference ─────────────────────────────────────────────────────────────
const results = await session.run({ float_input: tensor });

// results has two outputs:
//   results['label']         → Int64 tensor, shape [1]   — predicted class (vowel index)
//   results['probabilities'] → sequence of maps (class → probability)
const vowelIndex = Number(results["label"].data[0]);

// ── Convert to character position ─────────────────────────────────────────────
const VOWELS = new Set([..."аеєиіїоуюя"]);

function vowelPositions(word) {
  const positions = [];
  for (let i = 0; i < word.length; i++) {
    if (VOWELS.has(word[i].toLowerCase())) positions.push(i);
  }
  return positions;
}

function applyStressMark(word, vowelIdx) {
  const positions = vowelPositions(word);
  const charPos = positions[vowelIdx];
  // Insert combining acute U+0301 after the stressed vowel
  return word.slice(0, charPos + 1) + "\u0301" + word.slice(charPos + 1);
}

console.log(applyStressMark("університет", vowelIndex)); // → університе́т
```

### Feature order for the browser

The `manifest.json` in `web/` contains the canonical list of all 132 feature names
in order. Your JavaScript feature builder must produce values in exactly this order.

```js
const manifest = await fetch("/models/manifest.json").then((r) => r.json());
// manifest.feature_names[0..131] — feed values in this order to Float32Array
```

### Notes

| Topic                | Detail                                                                                                                                                                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ONNX opset**       | 9 (onnxmltools fixed value). Supported by onnxruntime-web 1.16+.                                                                                                                                                                         |
| **Accuracy**         | 100% class agreement with the LightGBM model on 2 000 random inputs (verified by `test_onnx_identical_argmax`).                                                                                                                          |
| **Feature building** | The bottleneck for browser deployment. The safest approach is a thin API endpoint that accepts `(word, pos)` and returns the 132-element feature array as JSON. This keeps the feature logic in Python and the inference in the browser. |
| **WASM threads**     | Enable `ort.env.wasm.numThreads` for better performance on multi-core devices.                                                                                                                                                           |
| **Bundle size**      | onnxruntime-web WASM backend: ~6 MB. Model: 30 MB download. Total cold start: ~36 MB.                                                                                                                                                    |

---

## 4. Artifacts reference

```
src/stress_prediction/lightGbm/artifacts/luscinia-lgbm-str-ua-univ-v1/
│
├── P3_0017_FINAL_FULLDATA/
│   ├── P3_0017_full.lgb       259 MB  LightGBM text format — Python service model
│   ├── meta.json                       Training metadata (accuracy, params, etc.)
│   └── manifest.json                   Export manifest (feature names, version)
│
├── web/
│   ├── P3_0017_full.onnx.gz    30 MB  ← deploy to web server
│   ├── P3_0017_full.onnx      185 MB  source ONNX (do not serve directly)
│   └── manifest.json                   Web manifest with serving instructions
│
├── leaderboard.txt                     HPO trial results summary
└── results.json / results.csv          Full Optuna study data
```

### manifest.json fields (both Python and web versions)

| Field                                     | Value                                            |
| ----------------------------------------- | ------------------------------------------------ |
| `version`                                 | `"luscinia-lgbm-str-ua-univ-v1.0"`               |
| `num_features`                            | `132`                                            |
| `num_classes`                             | `11`                                             |
| `num_boost_round`                         | `908`                                            |
| `feature_names`                           | Array of 132 feature names in model order        |
| `lgb_file` / `onnx_file` / `onnx_gz_file` | Filename (not full path)                         |
| `onnx_gz_size_mb`                         | `~30.0`                                          |
| `serving_note`                            | Instructions for `Content-Encoding: gzip` header |

---

## Quick reference

```
┌─────────────────────────────────────────────────────────────────────┐
│  PYTHON SERVICE                      WEB BROWSER                    │
│  ─────────────────────────────────   ──────────────────────────     │
│  import lightgbm as lgb              import * as ort from            │
│  import numpy as np                    'onnxruntime-web'             │
│                                                                     │
│  bst = lgb.Booster(                  session = await                 │
│    model_file='P3_0017_full.lgb')      InferenceSession.create(     │
│                                         '/models/P3.onnx.gz')       │
│  feat = build_features_universal(                                    │
│    word, pos)                        tensor = new ort.Tensor(        │
│                                        'float32', features, [1,132])│
│  X = np.array(list(feat.values()),                                   │
│    dtype=np.float32).reshape(1,-1)   results = await                 │
│                                        session.run({float_input:t}) │
│  probs = bst.predict(X)  # (1,11)                                    │
│  idx = probs.argmax(axis=1)[0]       idx = results['label'].data[0] │
└─────────────────────────────────────────────────────────────────────┘
```
