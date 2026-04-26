# Luscinia — lightgbm Ukrainian Stress Predictor

**Model:** `luscinia-lgbm-str-ua-univ-v1`  
**Accuracy:** 99.44 % (sanity) · 192 / 197 hand-checked · 100 % ONNX class agreement  
**Task:** predict the 0-based vowel index of the stressed syllable in a Ukrainian word

---

## Files

```
lightgbm/
  services/
    feature_service_universal.py   ← build the 132-feature input vector
    feature_service_2syl.py        ← specialist helpers (internal use)
    feature_service_3syl.py
    data_service.py
    evaluation_service.py
    model_export.py
    constants.py
  artifacts/
    luscinia-lgbm-str-ua-univ-v1/
      P3_0017_FINAL_FULLDATA/
        P3_0017_full.lgb           ← lightgbm model (Git LFS, 259 MB)
        meta.json                  ← training metadata
        manifest.json              ← feature names / version
      web/
        P3_0017_full.onnx.gz       ← ONNX for browser (Git LFS, 30 MB)
        manifest.json              ← serving metadata
      leaderboard.txt              ← HPO trial results
  documentation/
    LUSCINIA_LGBM_V1_DEPLOYMENT.md ← full usage guide (Python + browser)
  luscinia-lgbm-str-ua-univ-v1.py ← training script
  TRAINING_HISTORY.md             ← 12-month iteration history
```

## Usage

See [documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md](documentation/LUSCINIA_LGBM_V1_DEPLOYMENT.md)
for complete examples: single-word prediction, batch prediction, ONNX browser setup,
and nginx / Express serving instructions.

**Minimal Python example:**

```python
import lightgbm as lgb
import numpy as np
from src.stress_prediction.lightgbm.services.feature_service_universal import (
    build_features_universal,
)

bst = lgb.Booster(
    model_file="artifacts/luscinia-lgbm-str-ua-univ-v1/P3_0017_FINAL_FULLDATA/P3_0017_full.lgb"
)

feat = build_features_universal("університет", "NOUN")
X = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
vowel_idx = int(bst.predict(X).argmax(axis=1)[0])   # → 4
```

## Key design decisions

| Decision                                              | Rationale                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- |
| Single universal model (not per-syllable specialists) | One model covers 2–10 + syllable words; simpler deployment                      |
| Vowel-ordinal labels (0 = first vowel)                | Bounded label space (11 classes); compatible with DB schema                     |
| 132 hash features                                     | DJB2 suffix/prefix hashes massively outperform categorical encoding in lightgbm |
| `is_unbalance: True`                                  | Class 0/1 dominate; rare classes (5-10) need balancing                          |
| ONNX opset 15                                         | Supported by `onnxruntime-web` 1.16 +                                           |
