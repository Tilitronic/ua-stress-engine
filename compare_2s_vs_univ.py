#!/usr/bin/env python3
"""
Compare 2-syllable specialist (luscinia-lgbm-str-ua-2s-v1) vs
universal model (luscinia-lgbm-str-ua-univ-v1) on ALL 2-syllable words
in the training database.

Both models are evaluated on the exact same set of 2-syllable words.
The 2S model uses build_features_v13 (97 features, binary: class 0 vs 1).
The universal model uses build_features_universal (132 features, multiclass 0..10,
but for 2-syl words only classes 0 and 1 are valid).

Usage:
    python compare_2s_vs_univ.py
    python compare_2s_vs_univ.py --db path/to/stress_training.db
"""
import sys
import json
import sqlite3
from pathlib import Path
import multiprocessing as mp
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).resolve().parent
_LGBM  = _HERE / "src" / "stress_prediction" / "lightgbm"
_SVC   = _LGBM / "services"

for p in [str(_HERE), str(_LGBM), str(_SVC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import lightgbm as lgb
from services.feature_service import build_features_v13
from services.feature_service_universal import build_features_universal
from services.constants import VOWEL_SET
from services.data_service import stress_to_vowel_label
from src.utils.normalize_apostrophe import normalize_apostrophe

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_2S_PATH   = _LGBM / "artifacts" / "luscinia-lgbm-str-ua-2s-v1"   / "P3_0547_FINAL_FULLDATA" / "P3_0547_full.lgb"
MODEL_UNIV_PATH = _LGBM / "artifacts" / "luscinia-lgbm-str-ua-univ-v1" / "P3_0017_FINAL_FULLDATA" / "P3_0017_full.lgb"
DEFAULT_DB      = _LGBM.parent / "data" / "stress_training.db"

N_WORKERS = max(1, mp.cpu_count() - 1)


# ── Feature column order ──────────────────────────────────────────────────────
def _get_feature_cols(build_fn, sample_form: str = "мова") -> list:
    """Infer ordered feature column names by calling the builder once."""
    d = build_fn(sample_form, "NOUN", None)
    return list(d.keys())


# ── Data loading ──────────────────────────────────────────────────────────────
def load_2syl_words(db_path: Path) -> list:
    """
    Load ALL unique (form, pos, features_json, stress_indices) rows
    that are exactly 2-syllable, non-conflicting, non-variant, and have
    a valid stress label in {0, 1}.

    Returns list of dicts:
      { form, pos, features_json, label }
    """
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql(
        "SELECT form, lemma, stress_indices, pos, features_json, variant_type "
        "FROM training_entries",
        conn,
    )
    conn.close()

    # Remove ambiguous variant types
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()

    # Remove conflicting stress annotations for the same (form, pos, features_json)
    df["stress_count"] = (
        df.groupby(["form", "pos", "features_json"])["stress_indices"]
          .transform("nunique")
    )
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"]).reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        form = row["form"] or ""
        pos  = row.get("pos") or ""
        features_json = row.get("features_json") or None

        lower = normalize_apostrophe(form).lower()
        vowels = [i for i, c in enumerate(lower) if c in VOWEL_SET]
        if len(vowels) != 2:
            continue

        try:
            stress_raw = json.loads(row["stress_indices"] or "[]")
        except Exception:
            stress_raw = []

        label = stress_to_vowel_label(stress_raw, vowels)
        if label < 0 or label > 1:   # 2-syl: only 0 or 1
            continue

        records.append({
            "form":          form,
            "pos":           pos,
            "features_json": features_json,
            "label":         label,
            "lemma":         row.get("lemma", ""),
        })

    return records


# ── Prediction helpers ────────────────────────────────────────────────────────
def predict_2s(booster: lgb.Booster, records: list, feature_cols: list) -> np.ndarray:
    """Binary booster: returns predicted class 0/1 for each record."""
    rows = []
    for r in records:
        feat = build_features_v13(r["form"], r["pos"], r["features_json"])
        rows.append([feat.get(c, 0) for c in feature_cols])
    X = np.array(rows, dtype=np.float32)
    proba = booster.predict(X)          # shape (N,) — prob of class 1
    return (proba >= 0.5).astype(int)


def predict_univ(booster: lgb.Booster, records: list, feature_cols: list) -> np.ndarray:
    """Multiclass booster: returns argmax class for each record."""
    rows = []
    for r in records:
        feat = build_features_universal(r["form"], r["pos"], r["features_json"])
        rows.append([feat.get(c, 0) for c in feature_cols])
    X = np.array(rows, dtype=np.float32)
    proba = booster.predict(X)          # shape (N, 11)
    return np.argmax(proba, axis=1).astype(int)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc  = accuracy_score(y_true, y_pred)
    f1_0 = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    f1_1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": acc,
        "f1_class0": f1_0,
        "f1_class1": f1_1,
        "f1_macro":  f1_m,
        "confusion_matrix": cm,
        "n": len(y_true),
        "class0_count": int((y_true == 0).sum()),
        "class1_count": int((y_true == 1).sum()),
    }


def analyse_errors(records: list, y_true: np.ndarray,
                   pred_2s: np.ndarray, pred_univ: np.ndarray,
                   top_n: int = 20) -> None:
    """Print words where the models disagree or where both are wrong."""
    both_wrong  = []
    only_2s_ok  = []
    only_univ_ok = []

    for i, r in enumerate(records):
        t  = y_true[i]
        p2 = pred_2s[i]
        pu = pred_univ[i]
        if p2 == t and pu == t:
            continue
        if p2 != t and pu != t:
            both_wrong.append((r["form"], r["pos"], t, p2, pu))
        elif p2 == t and pu != t:
            only_univ_ok.append((r["form"], r["pos"], t, p2, pu))  # 2S right, univ wrong
        else:
            only_2s_ok.append((r["form"], r["pos"], t, p2, pu))    # univ right, 2S wrong

    def _fmt(lst, title, max_rows=top_n):
        if not lst:
            return
        print(f"\n  {title} ({len(lst)} cases):")
        for form, pos, true, p2, pu in lst[:max_rows]:
            print(f"    {form:25s} [{pos:6s}]  true={true}  2S={p2}  univ={pu}")
        if len(lst) > max_rows:
            print(f"    … and {len(lst) - max_rows} more")

    _fmt(both_wrong,   "Both wrong")
    _fmt(only_univ_ok, "2S right / univ WRONG  (2S wins)")
    _fmt(only_2s_ok,   "Univ right / 2S WRONG  (Univ wins)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    db_path = DEFAULT_DB
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--db" and i < len(sys.argv) - 1:
            db_path = Path(sys.argv[i + 1])

    print("=" * 72)
    print("  2-SYLLABLE SPECIALIST  vs  UNIVERSAL MODEL  — head-to-head")
    print("=" * 72)

    # ── Check model files ────────────────────────────────────────────────────
    for p, name in [(MODEL_2S_PATH, "2S specialist"), (MODEL_UNIV_PATH, "Universal")]:
        if not p.exists():
            print(f"[ERROR] {name} model not found: {p}")
            return 1
    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}")
        return 1

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading 2-syllable words from DB: {db_path}")
    records = load_2syl_words(db_path)
    if not records:
        print("[ERROR] No 2-syllable records found.")
        return 1
    y_true = np.array([r["label"] for r in records], dtype=int)
    n_total = len(records)
    n0 = int((y_true == 0).sum())
    n1 = int((y_true == 1).sum())
    print(f"  Total 2-syl records : {n_total:,}")
    print(f"  Class 0 (1st vowel) : {n0:,}  ({100*n0/n_total:.1f}%)")
    print(f"  Class 1 (2nd vowel) : {n1:,}  ({100*n1/n_total:.1f}%)")

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"\nLoading 2S specialist  : {MODEL_2S_PATH.name}")
    booster_2s = lgb.Booster(model_file=str(MODEL_2S_PATH))

    print(f"Loading universal model: {MODEL_UNIV_PATH.name}")
    booster_univ = lgb.Booster(model_file=str(MODEL_UNIV_PATH))

    # ── Feature column lists ─────────────────────────────────────────────────
    feat_cols_2s   = _get_feature_cols(build_features_v13)
    feat_cols_univ = _get_feature_cols(build_features_universal)
    print(f"\n  2S features   : {len(feat_cols_2s)}")
    print(f"  Univ features : {len(feat_cols_univ)}")

    # ── Predict ──────────────────────────────────────────────────────────────
    print(f"\nPredicting with 2S specialist …")
    pred_2s = predict_2s(booster_2s, records, feat_cols_2s)

    print(f"Predicting with universal model …")
    pred_univ = predict_univ(booster_univ, records, feat_cols_univ)

    # ── Metrics ──────────────────────────────────────────────────────────────
    m2s   = compute_metrics(y_true, pred_2s)
    muniv = compute_metrics(y_true, pred_univ)

    print()
    print("=" * 72)
    print("  RESULTS ON 2-SYLLABLE WORDS")
    print("=" * 72)
    header = f"  {'Metric':<28}  {'2S specialist':>14}  {'Universal':>14}  {'Δ (univ−2S)':>12}"
    print(header)
    print("  " + "-" * 70)

    def row(label, k, fmt=".4f"):
        v2  = m2s[k]
        vu  = muniv[k]
        delta = vu - v2
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<28}  {v2:>14{fmt}}  {vu:>14{fmt}}  {sign}{delta:>11{fmt}}")

    row("Accuracy",      "accuracy")
    row("F1 class-0 (stress=1st)", "f1_class0")
    row("F1 class-1 (stress=2nd)", "f1_class1")
    row("F1 macro",      "f1_macro")
    print("  " + "-" * 70)

    # Confusion matrices
    print()
    cm2s_str  = f"  [[{m2s['confusion_matrix'][0,0]:>7},{m2s['confusion_matrix'][0,1]:>7}]\n"  \
                f"   [{m2s['confusion_matrix'][1,0]:>7},{m2s['confusion_matrix'][1,1]:>7}]]"
    cmu_str   = f"  [[{muniv['confusion_matrix'][0,0]:>7},{muniv['confusion_matrix'][0,1]:>7}]\n" \
                f"   [{muniv['confusion_matrix'][1,0]:>7},{muniv['confusion_matrix'][1,1]:>7}]]"
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"    2S specialist:          Universal model:")
    cm2 = m2s['confusion_matrix']
    cmu = muniv['confusion_matrix']
    print(f"    [[{cm2[0,0]:>7},{cm2[0,1]:>7}]]       [[{cmu[0,0]:>7},{cmu[0,1]:>7}]]")
    print(f"    [{cm2[1,0]:>8},{cm2[1,1]:>7}]]        [{cmu[1,0]:>8},{cmu[1,1]:>7}]]")

    # ── Agreement analysis ───────────────────────────────────────────────────
    agree      = int((pred_2s == pred_univ).sum())
    both_right = int(((pred_2s == y_true) & (pred_univ == y_true)).sum())
    both_wrong = int(((pred_2s != y_true) & (pred_univ != y_true)).sum())
    only_2s_right  = int(((pred_2s == y_true) & (pred_univ != y_true)).sum())
    only_univ_right = int(((pred_univ == y_true) & (pred_2s != y_true)).sum())

    print()
    print("  Agreement analysis:")
    print(f"    Both predict same class       : {agree:,} / {n_total:,}  ({100*agree/n_total:.1f}%)")
    print(f"    Both correct                  : {both_right:,}  ({100*both_right/n_total:.1f}%)")
    print(f"    Both wrong                    : {both_wrong:,}  ({100*both_wrong/n_total:.1f}%)")
    print(f"    Only 2S correct (univ fails)  : {only_2s_right:,}  ({100*only_2s_right/n_total:.1f}%)")
    print(f"    Only Univ correct (2S fails)  : {only_univ_right:,}  ({100*only_univ_right/n_total:.1f}%)")

    # ── Verdict ──────────────────────────────────────────────────────────────
    acc_delta = muniv["accuracy"] - m2s["accuracy"]
    print()
    print("=" * 72)
    if abs(acc_delta) < 0.001:
        verdict = "DRAW (< 0.1% difference)"
    elif acc_delta > 0:
        verdict = f"UNIVERSAL WINS by {acc_delta*100:.2f}% accuracy"
    else:
        verdict = f"2S SPECIALIST WINS by {abs(acc_delta)*100:.2f}% accuracy"
    print(f"  VERDICT: {verdict}")
    print("=" * 72)

    # ── Error analysis ───────────────────────────────────────────────────────
    print("\nError analysis (sample):")
    analyse_errors(records, y_true, pred_2s, pred_univ, top_n=15)

    return 0


if __name__ == "__main__":
    sys.exit(main())
