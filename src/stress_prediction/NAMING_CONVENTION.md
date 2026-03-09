# Luscinia Model Naming Convention

> **Scope:** All models under `src/stress_prediction/` — LightGBM, XGBoost,
> neural, transformer, and any future architectures.
>
> **Date established:** 2026-03-05

---

## 1. Canonical Format

```
Luscinia-[ARCH]-[TASK]-[LANG]-v[VERSION]
```

| Segment     | Meaning                                | Example      |
| ----------- | -------------------------------------- | ------------ |
| `Luscinia`  | Model family name (always capitalised) | `Luscinia`   |
| `ARCH`      | Architecture / technology code        | `LGBM`       |
| `TASK`      | Task code                              | `STR`        |
| `LANG`      | Language ISO code (uppercase)          | `UA`         |
| `v[VERSION]`| Semantic version: `vMAJOR[.MINOR]`     | `v1`, `v1.2` |

### Full example

```
Luscinia-LGBM-STR-UA-v1
```

> *"We introduce **Luscinia-LGBM-STR-UA**, a gradient boosting model for
> Ukrainian lexical stress prediction."*

---

## 2. Architecture Codes (`ARCH`)

| Technology         | Code        |
| ------------------ | ----------- |
| LightGBM           | `LGBM`      |
| XGBoost            | `XGB`       |
| CatBoost           | `CAT`       |
| Random Forest      | `RF`        |
| Generic Neural Net | `NN`        |
| Bidirectional LSTM | `BiLSTM`    |
| Transformer        | `TRF`       |

---

## 3. Task Codes (`TASK`)

| Task                          | Code   |
| ----------------------------- | ------ |
| Lexical stress                | `STR`  |
| Stress + morphology           | `STRM` |
| Phonetics / transcription     | `PHON` |

---

## 4. Language Codes (`LANG`)

Use uppercase [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) codes.

| Language   | Code |
| ---------- | ---- |
| Ukrainian  | `UA` |
| English    | `EN` |
| Polish     | `PL` |

---

## 5. Version Scheme

| Situation                          | Format         | Example                       |
| ---------------------------------- | -------------- | ----------------------------- |
| Initial stable release             | `v1`           | `Luscinia-LGBM-STR-UA-v1`    |
| Minor improvement, same arch       | `v1.1`, `v1.2` | `Luscinia-LGBM-STR-UA-v1.2`  |
| New architecture                   | New `ARCH`     | `Luscinia-TRF-STR-UA-v1`     |
| Experimental / work-in-progress    | `-exp[N]` suffix | `Luscinia-LGBM-STR-UA-v1-exp3` |

---

## 6. File & Directory Naming

### Training scripts

```
luscinia-lgbm-str-ua-v1.py
luscinia-trf-str-ua-v1.py
```

### Artifact directories

```
artifacts/luscinia-lgbm-str-ua-v1/
artifacts/luscinia-trf-str-ua-v1/
```

### Result files (inside artifact dir)

```
luscinia-lgbm-str-ua-v1-results.csv
luscinia-lgbm-str-ua-v1-results.json
```

### Optuna study names (underscores — SQLite key)

```
luscinia_lgbm_str_ua_v1_p1
luscinia_lgbm_str_ua_v1_p2
luscinia_lgbm_str_ua_v1_p3
```

---

## 7. Specialist / Syllable-Count Variants

When a model is trained on a specific syllable-count subset, append a
**scope suffix** *before* the version:

```
Luscinia-LGBM-STR-UA-2S-v1     ← 2-syllable specialist
Luscinia-LGBM-STR-UA-3S-v1     ← 3-syllable specialist
Luscinia-LGBM-STR-UA-v1        ← combined (all syllable counts)
```

The scope suffix (`2S`, `3S`, …) is optional and only used when a model
is intentionally scope-limited.

---

## 8. Repository Names (for future splits)

```
luscinia-core          — shared dataset + preprocessing pipeline
luscinia-lgbm          — LightGBM stress models
luscinia-transformer   — transformer / neural models
```

---

## 9. Current Models

| Model ID                          | Script                              | Artifacts dir                              | Status       |
| --------------------------------- | ----------------------------------- | ------------------------------------------ | ------------ |
| `Luscinia-LGBM-STR-UA-2S-v1`     | `luscinia-lgbm-str-ua-2s-v1.py`    | `artifacts/luscinia-lgbm-str-ua-2s-v1/`   | Training     |
| *(predecessor)*  Bulbul v4 2S     | `batch_train_2syl_v4.py`           | `artifacts/v2syl_v4_research/`             | Archived     |

---

## 10. Anti-Patterns to Avoid

| ❌ Don't                        | ✅ Do instead                              |
| ------------------------------- | ------------------------------------------ |
| `luscinia_2s_v1`                | `luscinia-lgbm-str-ua-2s-v1`              |
| `batch_train_2syl_v5.py`        | `luscinia-lgbm-str-ua-2s-v1.py`           |
| `bulbul_v5_extended_research`   | `luscinia-lgbm-str-ua-v2` (new major ver) |
| `model_v3_final_FINAL2.lgb`     | `<trial_name>_FINAL_FULLDATA/<name>.lgb`  |
| `luscinia_2S_v1_p1` (study name)| `luscinia_lgbm_str_ua_2s_v1_p1`          |
