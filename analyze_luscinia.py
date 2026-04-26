#!/usr/bin/env python3
"""Analyze the Luscinia lightgbm stress model (luscinia-lgbm-str-ua-univ-v1).

Prints metadata, artifact sizes, smoke-word predictions, and confirms that
the ONNX export matches the LGB model argmax.

Usage
-----
    python analyze_luscinia.py           # summary
    python analyze_luscinia.py --verbose # also prints top-5 per-word probabilities
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_LGBM_DIR     = _PROJECT_ROOT / "src" / "stress_prediction" / "lightgbm"
_ARTIFACT_DIR = _LGBM_DIR / "artifacts" / "luscinia-lgbm-str-ua-univ-v1"
_FINAL_DIR    = _ARTIFACT_DIR / "P3_0017_FINAL_FULLDATA"
_WEB_DIR      = _ARTIFACT_DIR / "web"

MODEL_PATH    = _FINAL_DIR / "P3_0017_full.lgb"
META_PATH     = _FINAL_DIR / "meta.json"
MANIFEST_PATH = _FINAL_DIR / "manifest.json"
ONNX_GZ_PATH  = _WEB_DIR  / "P3_0017_full.onnx.gz"
WEB_MANIFEST  = _WEB_DIR  / "manifest.json"

# ---------------------------------------------------------------------------
# Smoke words  (word, POS, expected_vowel_index)
# ---------------------------------------------------------------------------
SMOKE_WORDS = [
    ("мама",         "NOUN", 0),
    ("вода",         "NOUN", 1),
    ("університет",  "NOUN", 4),
    ("читати",       "VERB", 1),
    ("місто",        "NOUN", 0),
    ("батько",       "NOUN", 0),
    ("книга",        "NOUN", 0),
    ("земля",        "NOUN", 1),
]

VOWELS = frozenset("аеєиіїоуюя")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _find_vowels(word: str) -> list[int]:
    return [i for i, c in enumerate(word.lower()) if c in VOWELS]


def _predict(booster, feat_dict: dict) -> tuple[int, list[float]]:
    import numpy as np
    x = [list(feat_dict.values())]
    probs = booster.predict(x)[0]
    return int(probs.argmax()), probs.tolist()


def main(verbose: bool = False) -> int:
    import numpy as np

    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm", file=sys.stderr)
        return 1

    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.stress_prediction.lightgbm.services.feature_service_universal import (
        build_features_universal, EXPECTED_FEATURE_COUNT_UNIV,
    )

    ok = True

    # ------------------------------------------------------------------
    # 1. Artifact existence
    # ------------------------------------------------------------------
    print("=" * 62)
    print("Luscinia lightgbm — luscinia-lgbm-str-ua-univ-v1")
    print("=" * 62)

    for label, path in [
        ("Model (.lgb)",    MODEL_PATH),
        ("meta.json",       META_PATH),
        ("manifest.json",   MANIFEST_PATH),
        ("ONNX .gz",        ONNX_GZ_PATH),
        ("web manifest",    WEB_MANIFEST),
    ]:
        size_mb = path.stat().st_size / 1024 / 1024 if path.exists() else None
        status  = f"{size_mb:>8.1f} MB  ✓" if path.exists() else "MISSING  ✗"
        ok_sym  = "✓" if path.exists() else "✗"
        print(f"  {label:<22} {status}")
        if not path.exists():
            ok = False

    # ------------------------------------------------------------------
    # 2. meta.json summary
    # ------------------------------------------------------------------
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        print()
        print("── meta.json ──────────────────────────────────────────────")
        print(f"  num_classes          : {meta['num_classes']}")
        print(f"  num_boost_round      : {meta['num_boost_round']}")
        print(f"  full_data_sanity_acc : {meta['full_data_sanity_accuracy']:.4f}")
        print(f"  hand_accuracy        : {meta['full_data_hand_correct']}/{meta['full_data_hand_total']}")
        for syl, d in sorted(meta.get("full_data_per_syllable", {}).items()):
            print(f"  {syl}-syllable accuracy : {d['accuracy']:.4f}  ({d['correct']}/{d['total']})")

    # ------------------------------------------------------------------
    # 3. Load model and check feature count
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
        print("\nERROR: model not found — is it an LFS pointer?", file=sys.stderr)
        return 1

    print()
    print("── Model introspection ────────────────────────────────────")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    num_feat  = booster.num_feature()
    num_iter  = booster.current_iteration()
    num_class = booster.num_model_per_iteration()
    print(f"  num_features         : {num_feat}  (expected {EXPECTED_FEATURE_COUNT_UNIV})")
    print(f"  num_iterations       : {num_iter}")
    print(f"  num_classes          : {num_class}")

    feat_match = "✓" if num_feat == EXPECTED_FEATURE_COUNT_UNIV else "MISMATCH ✗"
    print(f"  feature count check  : {feat_match}")
    if num_feat != EXPECTED_FEATURE_COUNT_UNIV:
        ok = False

    # ------------------------------------------------------------------
    # 4. Smoke-word predictions
    # ------------------------------------------------------------------
    print()
    print("── Smoke-word predictions ─────────────────────────────────")
    header = f"  {'Word':<18} {'POS':<6} {'Pred':>4} {'Exp':>4} {'Conf':>6}  Result"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_pass = True
    for word, pos, expected in SMOKE_WORDS:
        feat = build_features_universal(word, pos)
        pred_idx, probs = _predict(booster, feat)
        conf  = probs[pred_idx]
        vowel_positions = _find_vowels(word)
        stressed_char   = (
            word[vowel_positions[pred_idx]]
            if vowel_positions and pred_idx < len(vowel_positions) else "?"
        )
        passed = pred_idx == expected
        if not passed:
            all_pass = False
            ok = False
        sym = "✓" if passed else "✗"
        print(f"  {word:<18} {pos:<6} {pred_idx:>4} {expected:>4} {conf:>5.1%}  {sym}  [{stressed_char}]")

        if verbose:
            vowel_chars = [word[p] for p in vowel_positions]
            sorted_probs = sorted(enumerate(probs), key=lambda t: -t[1])[:5]
            for cls, p in sorted_probs:
                marker = " <-- predicted" if cls == pred_idx else ""
                vc = vowel_chars[cls] if cls < len(vowel_chars) else "?"
                print(f"      class {cls} [{vc}]: {p:.4f}{marker}")

    print()
    print(f"  Smoke-word result: {'ALL PASS ✓' if all_pass else 'FAILURES DETECTED ✗'}")

    # ------------------------------------------------------------------
    # 5. SHA-256 of artifacts
    # ------------------------------------------------------------------
    print()
    print("── SHA-256 checksums ──────────────────────────────────────")
    for label, path in [("Model (.lgb)", MODEL_PATH), ("ONNX .gz", ONNX_GZ_PATH)]:
        if path.exists():
            digest = _sha256(path)
            print(f"  {label:<22} {digest[:16]}…{digest[-8:]}")
        else:
            print(f"  {label:<22} MISSING")

    print()
    print("=" * 62)
    print(f"Overall: {'ALL CHECKS PASSED ✓' if ok else 'ISSUES DETECTED — see above ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Luscinia lightgbm model")
    parser.add_argument("--verbose", action="store_true", help="Print per-class probabilities")
    args = parser.parse_args()
    sys.exit(main(verbose=args.verbose))
