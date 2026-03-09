# Bulbul Training History — Lessons Learned & Archive

> **Date:** 2026-03-02  
> **Purpose:** Historical documentation before cleanup of 26 deprecated training scripts and ~15 GB of old artifacts.  
> **Kept:** `batch_train_v1.3_evo.py`, `batch_train_v1.4_optuna.py`, `artifacts/v1.3_evo/`, `plan.md`

---

## Timeline: 27 Scripts, 12 Months of Iteration

### Phase 1: Naive baselines (`train.py` → `train_v7.py`)

| Script        | Lines | Key Idea                                                    | Outcome                                             |
| ------------- | ----- | ----------------------------------------------------------- | --------------------------------------------------- |
| `train.py`    | 233   | First LightGBM. Form-level, pos+morph features              | Baseline established                                |
| `train_v2.py` | 280   | Positions 0-5 only, syllable boundary features              | Marginal improvement                                |
| `train_v3.py` | 422   | spaCy/Stanza morph as categoricals, rhyme features          | **Mistake:** noisy categoricals hurt more than help |
| `train_v4.py` | 438   | Noise reduction: drop variants, filter ambiguous            | **Key lesson:** data cleaning > feature engineering |
| `train_v5.py` | 511   | Vowel-index labels (0=1st vowel, 1=2nd, etc.)               | **Breakthrough:** correct label encoding, ~95% acc  |
| `train_v6.py` | 499   | DART boosting, lr=0.015, 3000 rounds                        | DART slow, minimal gain over GBDT                   |
| `train_v7.py` | 757   | 20 linguistic boolean features from Ukrainian stress papers | 95.84% but 416 MB model — **overfitting**           |

**Lessons from Phase 1:**

- ❌ Categorical features (suffix as string category) = LightGBM creates too many bins → model bloat
- ❌ DART boosting: 10× slower, negligible accuracy gain
- ❌ More features ≠ better: v7 (20 linguistic features, 416 MB) LOST to web_v2 (hash features, 2.9 MB)
- ✅ Vowel-index labels (v5) was the correct encoding all along
- ✅ Data filtering (remove ambiguous stress, variants) is essential

### Phase 2: Web deployment (`train_web_v1.py` → `train_web_v3.py`)

| Script            | Lines | Key Idea                                     | Outcome                                |
| ----------------- | ----- | -------------------------------------------- | -------------------------------------- |
| `train_web_v1.py` | 537   | Small ONNX model, hashed suffixes            | 90.57%, 878 KB gzip                    |
| `train_web_v2.py` | 647   | Better hash features, no categoricals        | **96.29%, 2.9 MB** — beat v7's 416 MB! |
| `train_web_v3.py` | 731   | Synthesis of v7 linguistics + web_v2 hashing | 96.5%                                  |

**Key insight:** Hash features (djb2 of suffix/prefix into 512-2048 buckets) massively outperform categorical encoding for LightGBM. This became the foundation for all future versions.

### Phase 3: Bulbul series (`train_stress_bulbul_v0.1.py` → `v1.0.py`)

| Script        | Lines | Key Idea                                    | Outcome                      |
| ------------- | ----- | ------------------------------------------- | ---------------------------- |
| `bulbul_v0.1` | 758   | Max accuracy, no size limits, hash features | 97.71%, F1=84.86%            |
| `bulbul_v0.2` | 753   | Push further with new features              | Overfitting, worse than v0.1 |
| `bulbul_v0.3` | 554   | Best of v0.1 + v0.2 features − mistakes     | Slight improvement           |
| `bulbul_v0.4` | 595   | Push to 98%, controlled size                | Parse errors, abandoned      |
| `bulbul_v0.5` | 725   | Synthesis of 27 overnight experiments       | Best single-shot model       |
| `bulbul_v1.0` | 898   | Production release                          | Production baseline          |

**Lessons from Phase 3:**

- ❌ Manual grid search is exhausting and error-prone
- ❌ More rounds ≠ better when lr is wrong (v0.2 overtrained)
- ✅ Vowel context hashes (bigram around each vowel) are powerful features
- ✅ Onset/coda pattern features help for 2-syllable words

### Phase 4: Batch & automated search (`overnight` → `v1.1` → `v1.2`)

| Script                        | Lines | Key Idea                              | Outcome                                   |
| ----------------------------- | ----- | ------------------------------------- | ----------------------------------------- |
| `overnight_batch_train.py`    | 1157  | 27 experiments, run overnight         | Good data but manual configs              |
| `batch_train_v1.1.py`         | 1017  | Bug-fix release, structured batch     | Working but limited search                |
| `batch_train_v1.1_unified.py` | 1442  | Scientific pipeline, group split      | **First proper train/val split by lemma** |
| `batch_train_v1.2_evo.py`     | 1503  | Evolutionary search (user's original) | 5-phase evo, external validation          |

**Lessons from Phase 4:**

- ✅ Group split by lemma prevents data leakage (forms of same lemma in both train and val)
- ✅ External validation sample catches overfitting that internal val misses
- ✅ Handcrafted test words reveal qualitative model behavior
- ❌ v1.1 unified had 20 hardcoded configs — too few to find optimum
- ❌ v1.2 evo: codebase was v1.1 feature set (fewer features), never got to run long enough

### Phase 5: v1.3 EVO — the big run (current knowledge base)

| Script                    | Lines | Key Idea                                       | Outcome                          |
| ------------------------- | ----- | ---------------------------------------------- | -------------------------------- |
| `batch_train_v1.3_evo.py` | 1995  | 97 features, 5-phase evo, 27 handcrafted tests | **17.1h, 80 configs, F1=69.56%** |

**v1.3 Results (80 experiments, 17.1 hours):**

- Winner: `A_g1_i2` — lr=0.05, leaves=512, mc=30, depth=12, bagging, 500 trees
- F1=69.56%, ext=95.5%, hand=22/27, ~187 MB, fitness=0.7762
- Sweet spot confirmed: lr≈0.05, bagging >> GOSS, extra_trees=False
- External accuracy 95-98% across all top configs
- 2-syllable words (~90%) much harder than 6+ syllable (99%+)

**Critical bugs found in v1.3:**

1. **Genome caching bug:** `genome_hash()` hashed only hyperparams, not `max_rounds`. Phases D (2500 trees) and E (4000 trees) saw same genomes as A (500 trees) → cache hit → 0 new trainings. Result: 80 configs were really all trained with ≤1200 trees.
2. **Size penalty bias:** Fitness function penalized model size. Phase A (500 trees, ~187 MB) always beat Phase C (1200 trees, ~449 MB) even when C had better F1 and ext accuracy. This made the multi-phase design pointless.
3. **All top-5 fail on identical 5 words:** ніжносолов'їний, віху, відторкнутися, куйовдлик, стрибкограй — compound/rare words need more features or data.

### Phase 6: v1.4 Optuna

Fixes all v1.3 bugs, uses TPE + Hyperband, warm-starts from v1.3 results.

---

### Phase 7: Bulbul v4 Extended Research — 2-Syllable Specialist (2026-03-04)

| Script                   | Lines | Key Idea                                         | Outcome                              |
| ------------------------ | ----- | ------------------------------------------------ | ------------------------------------ |
| `batch_train_2syl_v4.py` | ~1900 | 3-phase Optuna (Random/CMA-ES/TPE), 100 features | **8h, 2153 trials, best fit=0.8167** |

**Full run summary:**

- P1 (Random): 944 trials — broad landscape exploration
- P2 (CMA-ES): 394 trials — continuous optimisation
- P3 (TPE): 815 trials — Bayesian fine-tuning
- Optuna study: `specialist_binary` fitness preset throughout

**Leaderboard top-5:**

| Rank | Trial   | Phase | ext_acc | hand  | F1     | Size  | Fitness | Key params                              |
| ---- | ------- | ----- | ------- | ----- | ------ | ----- | ------- | --------------------------------------- |
| #1   | P3_0175 | P3    | 95.74%  | 40/44 | 74.94% | ~3 MB | 0.8167  | leaves=331, lr=0.15151, mc=24, depth=13 |
| #2   | P3_0811 | P3    | 95.84%  | 40/44 | 74.89% | ~4 MB | 0.8164  | leaves=394, lr=0.12578, mc=24, depth=12 |
| #3   | P2_0419 | P2    | 95.86%  | 40/44 | 74.80% | ~6 MB | 0.8161  | leaves=318, lr=0.07713, mc=24, depth=14 |
| #4   | P3_0097 | P3    | 95.28%  | 40/44 | 74.86% | ~5 MB | 0.8156  | leaves=212, lr=0.14516, mc=37, depth=12 |
| #5   | P2_0356 | P2    | 95.70%  | 40/44 | 74.65% | ~8 MB | 0.8147  | leaves=183, lr=0.08366, mc=22, depth=13 |

**Winner (full-data refit):** `P3_0175_FINAL_FULLDATA/` in `artifacts/v2syl_v4_research/`

**Key lessons from v4:**

- ✅ `specialist_binary` preset works well for P1/P2 coarse search
- ⚠️ `acc` weight at 25% is a weak discriminator once all top configs are ≥95% ext_acc
- ⚠️ Stubborn misses (hand=40/44): шпада, чікса (10/10 configs), кислій (9/10) — data gaps, not model failures
- ✅ Top-5 all converge to similar lr range: 0.075–0.155 (higher than v1.3's 0.05)
- ✅ Smaller models (3-6 MB) win: leaves=183-394, mc=22-37 — compact trees generalise better

---

### Phase 8: Luscinia-LGBM-STR-UA-2S-v1 (current generation)

| Script                          | Lines | Key Idea                                                | Status   |
| ------------------------------- | ----- | ------------------------------------------------------- | -------- |
| `luscinia-lgbm-str-ua-2s-v1.py` | 1960  | 4-phase (Random/CMA-ES/TPE + Ensemble), luscinia preset | Training |

**Architecture improvements over v4:**

- `luscinia_specialist` fitness preset for Phase 3: `hand_acc=25%` (was 15%), `acc=15%` (was 25%)
- Phase 4 Ensemble: top-K models (default K=5) via fitness-weighted soft vote; auto-accept if `sanity_gain ≥ 0.001` OR `hand_gain > 0`
- Guaranteed final refit on 100% data after any winning config (solo or ensemble)
- **Naming convention formalised:** `Luscinia-[ARCH]-[TASK]-[LANG]-[SCOPE]-v[N]` — see `../NAMING_CONVENTION.md`

**Artifacts:** `artifacts/luscinia-lgbm-str-ua-2s-v1/`

---

### Phase 9: Luscinia-LGBM-STR-UA-2S-v1 — Final Results (2026-03-05)

**Script:** `luscinia-lgbm-str-ua-2s-v1.py` | **Run:** 38h total, 99 trials (P1: 20, P2: 22, P3: 57 via resume)  
**Winner:** `P3_0547` → refit as `P3_0547_FINAL_FULLDATA/P3_0547_full.lgb` (15 MB)

| Metric                  | Val (90%) | Full-data refit        |
| ----------------------- | --------- | ---------------------- |
| Fitness                 | 0.8434    | —                      |
| Val accuracy            | —         | —                      |
| Sanity acc (5000 words) | —         | **97.32%** (4866/5000) |
| Handcrafted (44 words)  | —         | **44/44 (100%)**       |
| Train time              | —         | 5.1s                   |
| Model size              | —         | 15 MB                  |

**Winner params:** `leaves=341, lr=0.135, mc=13, depth=12, l2=2.77, col=0.347`

**Key observation:** The 2S specialist trained in 5 seconds on full data (124k rows, binary objective) — orders of magnitude faster than the universal model.

---

### Phase 10: Luscinia-LGBM-STR-UA-UNIV-v1 — Universal model (2026-03-06 → 03-08)

| Script                            | Lines | Key Idea                                            | Status       |
| --------------------------------- | ----- | --------------------------------------------------- | ------------ |
| `luscinia-lgbm-str-ua-univ-v1.py` | 1711  | Universal 11-class model, all syllable counts 2–10+ | **Complete** |

**Architecture:** single LightGBM multiclass model replacing per-syllable-count specialists.  
**Feature set:** `build_features_universal` — 132 features (97 base v13 + 35 universal extensions).  
**Label encoding:** vowel-index 0..10 (class = position of stressed vowel in word, 0-indexed).  
**Classes:** 11 (vowel indices 0–10). Class distribution severely skewed: class 1 = 35%, class 10 = 0.001%.  
**Training data:** 2,725,281 rows (after dedup/filter), group-split 90/10 by lemma.

#### Training run summary

| Phase                   | Trials | Duration  | Best fitness | Δ vs prev   |
| ----------------------- | ------ | --------- | ------------ | ----------- |
| P1 — Random exploration | 53     | ~631 min  | 0.8111       | baseline    |
| P2 — CMA-ES             | 22     | ~653 min  | 0.8133       | **+0.0022** |
| P3 — TPE Bayesian       | 24     | ~1008 min | **0.8237**   | **+0.0104** |
| **Total**               | **99** | **38.2h** | **0.8237**   | —           |

**Winner:** `P3_0017` → refit as `P3_0017_FINAL_FULLDATA/P3_0017_full.lgb` (272 MB)

#### Final refit metrics (full data, 2,725,477 rows)

| Metric                                  | Value                  |
| --------------------------------------- | ---------------------- |
| Sanity acc (5000 words, all syl counts) | **99.44%** (4972/5000) |
| Handcrafted (197 words)                 | **192/197 (97.5%)**    |
| Train time                              | 1734s (~29 min)        |
| Model size                              | 272 MB                 |
| Boosting rounds                         | 908                    |

**Per-syllable sanity breakdown:**

| Syl count | Correct | Total | Accuracy |
| --------- | ------- | ----- | -------- |
| 2         | 213     | 218   | 97.7%    |
| 3         | 957     | 966   | 99.1%    |
| 4         | 1437    | 1445  | 99.5%    |
| 5         | 1216    | 1221  | 99.6%    |
| 6         | 689     | 690   | 99.9%    |
| 7         | 460     | 460   | 100.0%   |

**Winner params:** `leaves=712, lr=0.0955, mc=18, depth=16, l1=0.741, l2=12.96, col=0.860, ffn=0.488, msh=26.74`

#### Convergence analysis

**P1:** Made all meaningful progress. 4 improvements in 631 min; best reached at trial 20/53 (min 184). Remaining 33 P1 trials added **nothing** — classic random search saturation.

**P2 (CMA-ES):** Gained only +0.0022 over P1 best. CMA-ES was seeded from P1 winners and got trapped in the same fitness basin. The algorithm explored but never escaped. This is a known failure mode for CMA-ES on highly multimodal landscapes.

**P3 (TPE):** The real workhorse. P3_0017 at trial 17/24 jumped +0.0097 above the prior record — a genuine find, not a trend. The other 23 P3 trials clustered tightly around 0.810–0.815 (stdev=0.017). **7 consecutive stalls before the winner, 7 consecutive stalls after.** The winner was isolated in parameter space.

**Saturation verdict:** The model is plateau-converged at this data/feature scale. Val accuracy is stuck at ~89.5%, val F1 macro at ~0.745. More HPO on the same setup will not yield meaningful gains.

#### What made P3_0017 the outlier vs the plateau cluster

| Parameter                 | Plateau cluster mean (7 trials) | P3_0017   | Interpretation                          |
| ------------------------- | ------------------------------- | --------- | --------------------------------------- |
| `num_leaves`              | 540                             | **712**   | More complex trees                      |
| `learning_rate`           | 0.106                           | **0.096** | Slightly lower — more careful           |
| `lambda_l2`               | 9.76                            | **12.96** | Stronger L2 regularisation              |
| `feature_fraction_bynode` | 0.694                           | **0.488** | Aggressive per-node feature subsampling |
| `min_sum_hessian_in_leaf` | 32.5                            | **26.74** | Slightly lower leaf floor               |

**Key insight:** big tree + aggressive per-node subsampling + strong L2 = better generalisation on imbalanced 11-class multiclass. The `feature_fraction_bynode=0.488` is the most distinctive differentiator.

#### Lessons learned

- ✅ Universal model (1 model, all syl counts) **matches or beats** per-specialist models even on their home turf (2-syllable words, see comparison below)
- ✅ 272 MB model on 2.7M rows at 11 classes is reasonable — 908 rounds × 11 classes = 9988 trees
- ⚠️ CMA-ES P2 was largely wasted here: gained only +0.002 in 653 min. With this problem size, P2 should use a wider CMA-ES sigma or more diverse seeds
- ⚠️ Val F1 macro (0.745) vs val acc (0.895) shows class imbalance is the primary bottleneck, not model expressiveness. Classes 5–10 are underserved at training time
- ⚠️ `best_iter=908` out of `MAX_ROUNDS_P3=1500` — budget not the constraint; early stopping fired at 908
- ❌ 5 handcrafted misses (192/197): 3 are rare proper nouns / toponyms not in training data — data gap, not model failure

---

### Phase 10b: Universal vs 2S Specialist — Head-to-Head (2026-03-08)

**Script:** `compare_2s_vs_univ.py`  
**Test set:** ALL 2-syllable words in the training DB (124,285 records, group-filtered, non-conflicting)  
**Class 0** (stress on 1st vowel): 83,464 (67.2%) | **Class 1** (stress on 2nd vowel): 40,821 (32.8%)

| Metric                  | 2S specialist (P3_0547) | Universal (P3_0017) | Δ          |
| ----------------------- | ----------------------- | ------------------- | ---------- |
| **Accuracy**            | 99.32%                  | **99.45%**          | **+0.13%** |
| F1 class-0 (stress=1st) | 0.9949                  | **0.9959**          | +0.0010    |
| F1 class-1 (stress=2nd) | 0.9898                  | **0.9916**          | +0.0018    |
| F1 macro                | 0.9924                  | **0.9938**          | +0.0014    |

**Confusion matrices (rows=true, cols=pred):**

```
2S specialist:          Universal:
[[82667,  797]]         [[83271,  193]]
[   45, 40776]]         [  491, 40330]]
```

**Agreement analysis:**

| Category                     | Count   | %     |
| ---------------------------- | ------- | ----- |
| Both correct                 | 122,963 | 98.9% |
| Both wrong                   | 204     | 0.2%  |
| Only 2S correct (univ fails) | 480     | 0.4%  |
| Only Univ correct (2S fails) | 638     | 0.5%  |

**Verdict: Universal WINS by +0.13% accuracy on the 2-syllable domain.**

The universal model has drastically fewer false positives for class 0 (193 vs 797), meaning it predicts "stress on 1st vowel" far more precisely. The 2S specialist wins on 480 specific cases — mostly class-1 nouns/adjectives with distinctive Ukrainian suffixes (Бляшка, Гірське etc.) — while the universal model wins on 638 different cases.

**Key lesson:** A single well-trained universal model is production-viable. Dedicated specialists are not necessary for 2-syllable words at this data scale.

---

### Improvement Roadmap: Universal v1.1 Estimate (2026-03-08)

**Current ceiling:** val acc ~89.5%, val F1 macro ~0.745, fitness ~0.824  
**Primary bottleneck:** class imbalance — classes 5–10 (5–11 syllables) have very few training examples

#### Proposed improvements and effort estimates

| Improvement                                                                                                      | Expected gain                      | Est. training time           | Complexity                         |
| ---------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ---------------------------- | ---------------------------------- |
| **A. Class weights / focal loss** — upweight rare classes 5–10 in the objective                                  | +0.005–0.015 F1 macro              | ~38h (full HPO run)          | Low — config change only           |
| **B. Feature selection** — remove/consolidate the ~30% of 132 features that `ffn=0.488` suggests are conflicting | +0.002–0.008 fitness               | ~20h (1 refit + short HPO)   | Medium — needs importance analysis |
| **C. More data for rare classes** — augment 5–10 syllable words from additional sources (Wiktionary UA, ВЕСЬ)    | +0.010–0.030 F1 macro              | 1–2 days data prep + 38h HPO | High — data pipeline work          |
| **D. P2 sigma tuning** — wider CMA-ES sigma (0.35–0.4) and more diverse seeds to escape the P1 basin             | +0.002–0.005 fitness               | ~38h                         | Low — config change                |
| **E. Warm-start v1.1 from P3_0017** — use winner params as P2 seed + tighter P3 search box                       | Reduce P1/P2 waste, more P3 budget | ~20h (skip P1 exploration)   | Low                                |

#### Recommendation: is v1.1 worth it?

**Short answer: Yes, but only with improvement A + C combined.**

- **A alone** (class weights): ~38h for +0.005–0.015 F1 macro. Gainworthy if you care about rare-class quality. The val F1 macro at 0.745 hiding behind 89.5% accuracy means the model is weak on 5+ syllable words — which matters for poetry (longer words appear often in verse).
- **B alone** (feature selection): marginal gains only, not worth a full run. Best done as a preprocessing pass before the next training.
- **E alone** (warm-start): cuts ~10h from P1 wasted exploration. Free win — do this regardless.
- **C** (more data): highest impact but also highest effort. Deferred to v1.2.

**Recommended v1.1 plan:**

1. Apply class weights: `class_weight = {5: 10, 6: 20, 7: 40, 8: 80, 9: 160, 10: 320}` (inverse-frequency scaling)
2. Warm-start from P3_0017 params (skip broad P1 random, start P2 from known-good region)
3. Widen CMA-ES sigma to 0.35
4. Budget: **24h** (reduced from 38h since P1 is skipped)
5. Expected outcome: fitness 0.826–0.835, F1 macro 0.755–0.760

**Total calendar time for v1.1: ~24–26 hours of unattended training.**

---

## Utility Scripts Removed

| Script                | Lines | Purpose                                      |
| --------------------- | ----- | -------------------------------------------- |
| `diagnostic.py`       | 181   | DB inspection and label distribution         |
| `_error_analysis.py`  | 163   | Post-hoc error analysis of model predictions |
| `_error_analysis2.py` | 128   | Improved error analysis                      |
| `_explore_data.py`    | 56    | Quick DB exploration                         |
| `_explore_data2.py`   | 53    | Extended exploration                         |

---

## Artifacts Removed (~15 GB)

| Path                              | Content                              | Size    |
| --------------------------------- | ------------------------------------ | ------- |
| `artifacts/stress_model*.pkl`     | v1-v6 sklearn pickles                | ~2 GB   |
| `artifacts/stress_model_v7.*`     | v7 LGB + pkl                         | ~500 MB |
| `artifacts/stress_model_web_v*.*` | web v1-v3 LGB + ONNX + pkl           | ~1 GB   |
| `artifacts/stress_bulbul_v0.*.*`  | bulbul v0.1-v1.0 LGB + ONNX + pkl    | ~3 GB   |
| `artifacts/model_config*.json`    | Config files for all above           | <1 MB   |
| `artifacts/overnight/`            | 27 overnight experiment results      | ~100 MB |
| `artifacts/v1.1_batch/`           | Empty directory                      | 0       |
| `artifacts/v1.1_unified/`         | 20 subdirs with LGB models + results | ~5 GB   |
| `artifacts/v1.2_evo/`             | Evo state + results CSV/JSON         | ~5 MB   |
| `artifacts/browser/`              | Browser-ready ONNX exports           | ~100 MB |

**Kept:** `artifacts/v1.3_evo/` (4 files, 2 MB) — warm-start data for v1.4.

---

## Top Insights (for future reference)

1. **Hash features > categorical features** for LightGBM. djb2 hash of suffix/prefix into 512-2048 buckets preserves enough information while avoiding categorical explosion.

2. **Group split by lemma** is mandatory. Without it, "наприклад" in train and "наприклад" in val = data leakage. All metrics are inflated by ~3-5%.

3. **External validation sample** catches things internal val misses. Must be stratified by syllable count — otherwise 4-5 syllable words dominate (60%+ of data).

4. **Handcrafted test words** reveal qualitative failures (compound words, rare morphology) that F1 average hides.

5. **Model size ≠ quality indicator.** v7 (416 MB) lost to web_v2 (2.9 MB). Size penalty in fitness is counterproductive.

6. **DART boosting:** tested in v6, bulbul_v1.1_unified (F1 configs). 10× slower training, <0.5% accuracy gain. Not worth it.

7. **GOSS vs bagging:** v1.3 EVO with 80 configs confirmed: bagging consistently beats GOSS by ~1% F1. GOSS is only useful for speed.

8. **Learning rate:** Sweet spot 0.03-0.08. Below 0.02 = too slow to converge in reasonable time. Above 0.10 = overfitting on tail classes.

9. **2-syllable words** are the hardest category (~90% ext accuracy vs 99%+ for 6+ syllable). They need special attention — possibly a dedicated sub-model.

10. **Caching by hyperparameters only** is a subtle bug when you vary training budget (num_rounds) across phases. Always include ALL training parameters in cache keys.

---

## Luscinia Universal Model v1.0 and v1.1

> **Date:** 2026-03-08 / 2026-03-09
> **Scripts:** `luscinia-lgbm-str-ua-univ-v1.py`, `luscinia-lgbm-str-ua-univ-v1.1.py`
> **Final model kept:** `artifacts/luscinia-lgbm-str-ua-univ-v1/P3_0017_FINAL_FULLDATA/P3_0017_full.lgb`

### v1.0 - Production Model (SELECTED)

Training run: ~38.2 h wall time, 99 trials (P1 + P2 + P3 + ensemble)

#### Final refit metrics (P3_0017, full dataset - 2,725,477 rows)

| Metric                      | Value              |
| --------------------------- | ------------------ |
| HPO fitness                 | 0.8237             |
| Sanity accuracy (full data) | 99.44%             |
| Hand-curated words correct  | 192 / 197 (97.5%)  |
| 2-syllable accuracy         | 97.71% (213/218)   |
| 3-syllable accuracy         | 99.07% (957/966)   |
| 4-syllable accuracy         | 99.45% (1437/1445) |
| 5-syllable accuracy         | 99.59% (1216/1221) |
| 6-syllable accuracy         | 99.86% (689/690)   |
| 7-syllable accuracy         | 100.00% (460/460)  |
| Boost rounds                | 908 (converged)    |
| Train time (refit)          | 1734 s             |

Winner params: num_leaves=712 max_depth=16 learning_rate=0.0955 min_child_samples=18 lambda_l1=0.741 lambda_l2=12.96

HPO leaderboard top-5 (5000-word sanity sample):
#1 P3_0017 fit=0.8237 F1=74.56% hand=165/197 sanity=98.26%
#2 P3_0018 fit=0.8195 F1=74.69% hand=162/197 sanity=98.26%
#3 P3_0022 fit=0.8173 F1=74.99% hand=160/197 sanity=98.36%
#4 P3_0021 fit=0.8151 F1=74.25% hand=161/197 sanity=98.20%
#5 P3_0015 fit=0.8140 F1=75.02% hand=160/197 sanity=98.24%

---

### v1.1 - Class-weighting experiment (REJECTED - worse on all metrics)

Training run: 24.77 h wall time, 37 trials (15 P2 + 22 P3), warm-started from v1.0 P3_0017
Decision: Rejected. Regressed on every metric vs v1.0.

Motivation: v1.0 had lower F1 on rare high-syllable classes (8-10 syl). v1.1 introduced
sqrt-inverse class weighting (class_weight_power=0.5) to boost minority classes, plus pruner
fixes and warm-start HPO seeded from v1.0 winner params.

#### Head-to-head (full-data refit)

| Metric             | v1.0            | v1.1            | Delta              |
| ------------------ | --------------- | --------------- | ------------------ |
| HPO fitness        | 0.8237          | 0.7601          | -0.0636            |
| Sanity accuracy    | 99.44%          | 96.80%          | -2.64 pp           |
| Hand correct       | 192/197 (97.5%) | 174/197 (88.3%) | -18 words          |
| 2-syllable         | 97.71%          | 89.91%          | -7.80 pp           |
| 3-syllable         | 99.07%          | 93.58%          | -5.49 pp           |
| 4-syllable         | 99.45%          | 96.47%          | -2.98 pp           |
| 5-syllable         | 99.59%          | 98.20%          | -1.39 pp           |
| 6-syllable         | 99.86%          | 99.57%          | -0.29 pp           |
| 7-syllable         | 100%            | 100%            | 0                  |
| Boost rounds       | 908 (converged) | 1500 (hit max)  | not converged      |
| Train time (refit) | 1734 s          | 2217 s          | +28%               |
| P3 trials          | 65+             | 22              | 3x fewer           |
| P3 pruned          | --              | 0/22            | pruner ineffective |

v1.1 winner params: num_leaves=543 max_depth=13 learning_rate=0.051
min_child_samples=36 lambda_l1=0.608 lambda_l2=3.086 class_weight_power=0.5

Root causes:

1. Class weights slowed convergence past budget. All 22 P3 trials hit hard cap 1500
   (avg best_iter=1500 exactly). Model never converged within budget.
2. Weights hurt majority classes. Classes 8-10 (28-686 samples) upweighted up to 57x.
   2-syl and 3-syl took -7.8pp / -5.5pp -- the bulk of real Ukrainian text usage.
3. Pruner never fired -> only 22 P3 trials. Every trial ran full 1500 rounds.
4. Warm-start CMA-ES wandered: sigma=0.35 in +/-60% window produced winner
   (leaves=543, lr=0.051) far from seed (leaves=712, lr=0.0955) after only 22 trials.

HPO leaderboard top-3 (5000-word sanity sample):
#1 P3_0021 fit=0.7601 F1=64.01% hand=160/197 sanity=94.96%
#2 P3_0019 fit=0.7574 F1=64.57% hand=157/197 sanity=95.06%
#3 P3_0020 fit=0.7556 F1=63.85% hand=158/197 sanity=94.82%

Recommendations if revisiting class weighting in v1.2:

- CLASS_WEIGHT_POWER=0.0 first (control run), then try 0.25
- MAX_ROUNDS_P3 -> 2500 (v1.1 never converged at 1500)
- Early-stopping patience in P3 -> 100-120 rounds (weighted plateau is wider)
- Remove warm-start or use tight sigma=0.15 (refinement only, not exploration)
