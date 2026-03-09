#!/usr/bin/env python3
"""Compare v3 vs v4 stress models side-by-side."""

import sqlite3
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Paths
DB_PATH = Path("src/stress_prediction/data/stress_training.db")
MODEL_V3_PATH = Path("src/stress_prediction/lightGbm/artifacts/stress_model_v3.pkl")
MODEL_V4_PATH = Path("src/stress_prediction/lightGbm/artifacts/stress_model_v4.pkl")
CONFIG_V3_PATH = Path("src/stress_prediction/lightGbm/artifacts/model_config_v3.json")
CONFIG_V4_PATH = Path("src/stress_prediction/lightGbm/artifacts/model_config_v4.json")

VOWELS = "аеєиіїоуюя"

def find_vowels(word: str):
    return [i for i, ch in enumerate(word.lower()) if ch in VOWELS]

def get_syllable_info(word: str):
    word_lower = word.lower()
    vowels = find_vowels(word_lower)
    if not vowels:
        return 0, 0, 0, "no_vowel"
    num_syllables = len(vowels)
    first_vowel_pos = vowels[0]
    pattern = "".join("V" if word_lower[i] in VOWELS else "C" for i in range(first_vowel_pos + 1))
    dist_to_first = first_vowel_pos
    dist_from_last = len(word_lower) - vowels[-1] - 1
    return num_syllables, dist_to_first, dist_from_last, pattern

def stress_to_class(stress):
    if not stress:
        return -1, 0
    primary = stress[0]
    label = primary if primary <= 5 else 6
    return label, len(stress)

def load_training_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT
            form, lemma, stress_indices, pos, features_json,
            variant_type, is_homonym, is_disambiguable,
            pos_confidence, features_confidence
        FROM training_entries
        """,
        conn,
    )
    conn.close()
    return df

def build_features_v3(row):
    """v3 feature set (original)."""
    feats_json = row.get("features_json") or "{}"
    feats = json.loads(feats_json)
    form = row["form"] or ""
    vowels = find_vowels(form)
    num_syl, dist_first, dist_last, syl_pattern = get_syllable_info(form)
    last_vowel_idx = vowels[-1] if vowels else -1

    stress = json.loads(row["stress_indices"] or "[]")
    label, num_stresses = stress_to_class(stress)

    feature_row = {
        "word_len": len(form),
        "vowel_count": len(vowels),
        "num_syllables": num_syl,
        "dist_to_first_vowel": dist_first,
        "dist_from_last_vowel": dist_last,
        "last_vowel_pos": last_vowel_idx,
        "pos": row.get("pos") or "",
        "case": feats.get("Case", ""),
        "number": feats.get("Number", ""),
        "gender": feats.get("Gender", ""),
        "is_homonym": row.get("is_homonym", 0),
        "is_disambiguable": row.get("is_disambiguable", 0),
        "num_stresses": num_stresses,
        "is_multi_stress": 1 if num_stresses > 1 else 0,
        "pos_confidence": row.get("pos_confidence", 1.0),
        "features_confidence": row.get("features_confidence", 1.0),
        "suffix_2": form[-2:] if len(form) >= 2 else "",
        "suffix_3": form[-3:] if len(form) >= 3 else "",
        "suffix_4": form[-4:] if len(form) >= 4 else "",
        "prefix_2": form[:2] if len(form) >= 2 else "",
        "first_syllable_type": syl_pattern,
    }
    return feature_row, label

def build_features_v4(row):
    """v4 feature set (enhanced with suffix_5, suffix3_pos, vowel ratios)."""
    feats_json = row.get("features_json") or "{}"
    feats = json.loads(feats_json)
    form = row["form"] or ""
    vowels = find_vowels(form)
    num_syl, dist_first, dist_last, syl_pattern = get_syllable_info(form)
    last_vowel_idx = vowels[-1] if vowels else -1

    stress = json.loads(row["stress_indices"] or "[]")
    label, num_stresses = stress_to_class(stress)

    feature_row = {
        "word_len": len(form),
        "vowel_count": len(vowels),
        "num_syllables": num_syl,
        "dist_to_first_vowel": dist_first,
        "dist_from_last_vowel": dist_last,
        "last_vowel_pos": last_vowel_idx,
        "pos": row.get("pos") or "",
        "case": feats.get("Case", ""),
        "number": feats.get("Number", ""),
        "gender": feats.get("Gender", ""),
        "is_homonym": row.get("is_homonym", 0),
        "is_disambiguable": row.get("is_disambiguable", 0),
        "num_stresses": num_stresses,
        "is_multi_stress": 1 if num_stresses > 1 else 0,
        "pos_confidence": row.get("pos_confidence", 1.0),
        "features_confidence": row.get("features_confidence", 1.0),
        "suffix_2": form[-2:] if len(form) >= 2 else "",
        "suffix_3": form[-3:] if len(form) >= 3 else "",
        "suffix_4": form[-4:] if len(form) >= 4 else "",
        "suffix_5": form[-5:] if len(form) >= 5 else "",
        "suffix3_pos": f"{form[-3:] if len(form) >= 3 else ''}_{row.get('pos') or ''}",
        "first_vowel_ratio": vowels[0] / len(form) if vowels and len(form) > 0 else 0.0,
        "last_vowel_ratio": vowels[-1] / len(form) if vowels and len(form) > 0 else 0.0,
        "prefix_2": form[:2] if len(form) >= 2 else "",
        "first_syllable_type": syl_pattern,
    }
    return feature_row, label

def build_dataset_v3(df):
    """v3 dataset pipeline."""
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()
    df["stress_count"] = df.groupby(["form", "pos", "features_json"])["stress_indices"].transform("nunique")
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"])

    features = []
    labels = []
    kept = []
    for _, row in df.iterrows():
        feats, label = build_features_v3(row)
        if label < 0:
            continue
        features.append(feats)
        labels.append(label)
        kept.append(row)
    
    X = pd.DataFrame(features)
    y = pd.Series(labels)
    kept_df = pd.DataFrame(kept)
    
    cat_features = ["pos", "case", "number", "gender", "suffix_2", "suffix_3", "suffix_4", "prefix_2", "first_syllable_type"]
    for col in cat_features:
        X[col] = X[col].astype("category")
    
    return X, y, kept_df

def build_dataset_v4(df):
    """v4 dataset pipeline."""
    df = df[~df["variant_type"].isin(["free_variant", "grammatical_homonym"])].copy()
    df["stress_count"] = df.groupby(["form", "pos", "features_json"])["stress_indices"].transform("nunique")
    df = df[df["stress_count"] == 1].drop(columns=["stress_count"])

    features = []
    labels = []
    kept = []
    for _, row in df.iterrows():
        feats, label = build_features_v4(row)
        if label < 0:
            continue
        features.append(feats)
        labels.append(label)
        kept.append(row)
    
    X = pd.DataFrame(features)
    y = pd.Series(labels)
    kept_df = pd.DataFrame(kept)
    
    cat_features = ["pos", "case", "number", "gender", "suffix_2", "suffix_3", "suffix_4", "suffix_5", "suffix3_pos", "prefix_2", "first_syllable_type"]
    for col in cat_features:
        X[col] = X[col].astype("category")
    
    return X, y, kept_df

def group_split(df, X, y, test_size=0.1, seed=42):
    from sklearn.model_selection import GroupShuffleSplit
    groups = df["lemma"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        train_idx,
        val_idx,
    )

def main():
    print("=" * 70)
    print("📊 COMPARING STRESS MODEL v3 vs v4")
    print("=" * 70)

    # Load data
    print("\n📚 Loading training data...")
    df = load_training_data(DB_PATH)
    print(f"   Total rows: {len(df):,}")

    # Build datasets
    print("\n🔧 Building v3 dataset...")
    X_v3, y_v3, kept_df_v3 = build_dataset_v3(df.copy())
    print(f"   X shape: {X_v3.shape}")
    print(f"   Features: {list(X_v3.columns)}")

    print("\n🔧 Building v4 dataset...")
    X_v4, y_v4, kept_df_v4 = build_dataset_v4(df.copy())
    print(f"   X shape: {X_v4.shape}")
    print(f"   Features: {list(X_v4.columns)}")

    # Split
    print("\n✂️  Splitting data (same split for both)...")
    X_train_v3, X_val_v3, y_train_v3, y_val_v3, _, _ = group_split(kept_df_v3, X_v3, y_v3)
    X_train_v4, X_val_v4, y_train_v4, y_val_v4, _, _ = group_split(kept_df_v4, X_v4, y_v4)
    
    # Load models
    print("\n🤖 Loading models...")
    if not MODEL_V3_PATH.exists():
        print(f"   ⚠️  v3 model not found at {MODEL_V3_PATH}")
        return 1
    if not MODEL_V4_PATH.exists():
        print(f"   ⚠️  v4 model not found at {MODEL_V4_PATH}")
        return 1
    
    model_v3 = joblib.load(MODEL_V3_PATH)
    model_v4 = joblib.load(MODEL_V4_PATH)
    print(f"   ✓ v3 model loaded")
    print(f"   ✓ v4 model loaded")

    # Evaluate
    print("\n" + "=" * 70)
    print("📈 EVALUATION RESULTS")
    print("=" * 70)

    print("\n### MODEL v3 (Original) ###")
    y_pred_v3 = model_v3.predict(X_val_v3)
    acc_v3 = accuracy_score(y_val_v3, y_pred_v3)
    f1_v3 = f1_score(y_val_v3, y_pred_v3, average="macro")
    report_v3 = classification_report(y_val_v3, y_pred_v3, output_dict=True)
    
    print(f"Accuracy: {acc_v3*100:.2f}%")
    print(f"Macro F1: {f1_v3*100:.2f}%")
    print("\nPer-class metrics:")
    for cls in sorted(k for k in report_v3.keys() if str(k).isdigit()):
        metrics = report_v3[cls]
        print(f"  class {cls}: p={metrics['precision']:.3f}, r={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

    print("\n### MODEL v4 (Enhanced) ###")
    y_pred_v4 = model_v4.predict(X_val_v4)
    acc_v4 = accuracy_score(y_val_v4, y_pred_v4)
    f1_v4 = f1_score(y_val_v4, y_pred_v4, average="macro")
    report_v4 = classification_report(y_val_v4, y_pred_v4, output_dict=True)
    
    print(f"Accuracy: {acc_v4*100:.2f}%")
    print(f"Macro F1: {f1_v4*100:.2f}%")
    print("\nPer-class metrics:")
    for cls in sorted(k for k in report_v4.keys() if str(k).isdigit()):
        metrics = report_v4[cls]
        print(f"  class {cls}: p={metrics['precision']:.3f}, r={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

    print("\n" + "=" * 70)
    print("🎯 COMPARISON SUMMARY")
    print("=" * 70)
    
    acc_diff = (acc_v4 - acc_v3) * 100
    f1_diff = (f1_v4 - f1_v3) * 100
    
    print(f"\nAccuracy:   v3={acc_v3*100:.2f}%  →  v4={acc_v4*100:.2f}%  ({acc_diff:+.2f}%)")
    print(f"Macro F1:   v3={f1_v3*100:.2f}%  →  v4={f1_v4*100:.2f}%  ({f1_diff:+.2f}%)")
    
    if acc_v4 > acc_v3:
        print(f"\n✅ v4 is BETTER by {acc_diff:.2f}% accuracy")
    else:
        print(f"\n⚠️  v4 is WORSE by {abs(acc_diff):.2f}% accuracy")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("🔍 FEATURE IMPORTANCE")
    print("=" * 70)
    
    print("\n### v3 Top 10 Features ###")
    importances_v3 = model_v3.feature_importances_
    features_v3 = list(X_v3.columns)
    top_idx_v3 = np.argsort(importances_v3)[::-1][:10]
    for i, idx in enumerate(top_idx_v3, 1):
        print(f"  {i:2d}. {features_v3[idx]:20s} {importances_v3[idx]:8.1f}")
    
    print("\n### v4 Top 10 Features ###")
    importances_v4 = model_v4.feature_importances_
    features_v4 = list(X_v4.columns)
    top_idx_v4 = np.argsort(importances_v4)[::-1][:10]
    for i, idx in enumerate(top_idx_v4, 1):
        print(f"  {i:2d}. {features_v4[idx]:20s} {importances_v4[idx]:8.1f}")
    
    # New features importance
    new_features = {"suffix_5", "suffix3_pos", "first_vowel_ratio", "last_vowel_ratio"}
    print("\n### New Feature Importance (v4) ###")
    for feat in new_features:
        if feat in features_v4:
            idx = features_v4.index(feat)
            print(f"  {feat:20s} {importances_v4[idx]:8.1f}")

    print("\n" + "=" * 70)
    return 0

if __name__ == "__main__":
    exit(main())
