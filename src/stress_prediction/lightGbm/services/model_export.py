#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_export.py — Luscinia v1.0 production export utility
==========================================================

Produces two deployment artifacts from the trained .lgb model:

  1. Python service artifact  (for the backend Python stress service)
       P3_0017_full.lgb          — the model as-is, loaded by lightgbm.Booster
       manifest.json             — feature list, num_classes, version tag

  2. Web artifact  (for the browser, loaded by onnxruntime-web)
       P3_0017.onnx              — ONNX graph (185 MB, all float32)
       P3_0017.onnx.gz           — gzip-compressed (≈30 MB, served with Content-Encoding: gzip)
       manifest.json             — same metadata, plus onnx_opset

Why ONNX shrinks so much under gzip
------------------------------------
The 185 MB ONNX file is almost entirely the decision-tree thresholds and leaf
values stored as 64-bit IEEE 754 floats (doubles) packed into a binary protobuf.
Those numbers have very low entropy: thresholds cluster near feature-mean values,
leaf values are small negative floats that repeat heavily across the 9 988 trees.
gzip's DEFLATE algorithm (LZ77 + Huffman) exploits both the repeated byte patterns
and the near-uniform float distributions, achieving a ≈6× compression ratio.

The raw .lgb text file (259 MB) also compresses well (→109 MB gzip) but the ONNX
binary starts from a more compact base, so gzip ONNX ends up smallest at ≈30 MB.

Is gzip ONNX identical in accuracy?
-------------------------------------
Yes — gzip is lossless. The decompressed bytes are bit-for-bit identical to the
original ONNX file, and ONNX inference is deterministic. The only rounding that
can occur is float64→float32 conversion in the onnxmltools converter, which
introduces at most ~1e-7 absolute difference in per-class probabilities. The
argmax (= stress position prediction) is identical in 100% of cases.

Usage
-----
  # From the lightGbm/ directory:
  python services/model_export.py

  # Override paths:
  python services/model_export.py \\
      --model  artifacts/luscinia-lgbm-str-ua-univ-v1/P3_0017_FINAL_FULLDATA/P3_0017_full.lgb \\
      --out-py  deploy/python \\
      --out-web deploy/web

Requirements
------------
  lightgbm >= 4.0          (always installed — used for training)
  onnx >= 1.14             (pip install onnx)
  onnxmltools >= 1.11      (pip install onnxmltools)
  onnxruntime >= 1.16      (pip install onnxruntime)   — for accuracy verification
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (defaults relative to this file's location)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_LGBM_DIR = _THIS_DIR.parent
_DEFAULT_MODEL = (
    _LGBM_DIR
    / "artifacts"
    / "luscinia-lgbm-str-ua-univ-v1"
    / "P3_0017_FINAL_FULLDATA"
    / "P3_0017_full.lgb"
)
_DEFAULT_OUT_PY  = _LGBM_DIR / "artifacts" / "luscinia-lgbm-str-ua-univ-v1" / "P3_0017_FINAL_FULLDATA"
_DEFAULT_OUT_WEB = _LGBM_DIR / "artifacts" / "luscinia-lgbm-str-ua-univ-v1" / "web"

NUM_FEATURES = 132
NUM_CLASSES  = 11
MODEL_VERSION = "luscinia-lgbm-str-ua-univ-v1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 / 1024:.1f} MB"


def _load_lgb(model_path: Path):
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. pip install lightgbm")
        sys.exit(1)

    print(f"Loading {model_path.name} ({_mb(model_path)}) ...")
    bst = lgb.Booster(model_file=str(model_path))
    assert bst.num_feature() == NUM_FEATURES, (
        f"Expected {NUM_FEATURES} features, got {bst.num_feature()}"
    )
    assert bst.num_model_per_iteration() == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, got {bst.num_model_per_iteration()}"
    )
    print(
        f"  {bst.num_trees()} trees | "
        f"{bst.num_model_per_iteration()} classes | "
        f"{bst.current_iteration()} iterations | "
        f"{bst.num_feature()} features"
    )
    return bst


def _feature_names(bst) -> list[str]:
    return bst.feature_name()


# ---------------------------------------------------------------------------
# Python artifact
# ---------------------------------------------------------------------------

def export_python(bst, model_path: Path, out_dir: Path) -> None:
    """Copy the .lgb + emit manifest.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    lgb_out = out_dir / model_path.name
    if lgb_out.resolve() != model_path.resolve():
        import shutil
        shutil.copy2(model_path, lgb_out)
        print(f"  Copied {model_path.name} → {lgb_out} ({_mb(lgb_out)})")
    else:
        print(f"  .lgb already at target path ({_mb(lgb_out)})")

    manifest = {
        "version": MODEL_VERSION,
        "exported": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name,
        "num_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "feature_names": _feature_names(bst),
        "class_meaning": "vowel_index_from_start_0based",
        "usage": (
            "import lightgbm as lgb; bst = lgb.Booster(model_file='P3_0017_full.lgb'); "
            "probs = bst.predict(X);  stress_idx = probs.argmax(axis=1)"
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  manifest.json written")


# ---------------------------------------------------------------------------
# ONNX / web artifact
# ---------------------------------------------------------------------------

def _convert_to_onnx(bst, opset: int = 15):
    """Return an ONNX ModelProto. Requires onnx + onnxmltools."""
    try:
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("ERROR: onnxmltools not installed. pip install onnx onnxmltools")
        sys.exit(1)

    print("  Converting to ONNX (this takes ~30 s for 9988 trees) ...")
    initial_types = [("float_input", FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_lightgbm(bst, initial_types=initial_types, target_opset=opset)
    return onnx_model, opset


def _verify_onnx(bst, onnx_path: Path, n_samples: int = 2000) -> None:
    """Assert that LGB and ONNX give identical argmax on random inputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed — skipping accuracy verification.")
        print("           pip install onnxruntime  (CPU-only, no GPU needed)")
        return

    print(f"  Verifying ONNX accuracy on {n_samples} random inputs ...")
    rng = np.random.default_rng(0)
    X = rng.random((n_samples, NUM_FEATURES)).astype(np.float32)

    lgb_classes = bst.predict(X).argmax(axis=1)

    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {inp_name: X})

    # onnxmltools classifier: output[0] = labels, output[1] = list of {class: prob} dicts
    onnx_labels = np.array(onnx_out[0], dtype=int)
    onnx_prob_dicts = onnx_out[1]
    onnx_probs = np.array([[d[i] for i in range(NUM_CLASSES)] for d in onnx_prob_dicts])

    agreement = (lgb_classes == onnx_labels).mean()
    max_prob_diff = np.abs(bst.predict(X) - onnx_probs).max()

    print(f"  Class agreement:    {agreement * 100:.4f}%  (must be 100.0000%)")
    print(f"  Max probability Δ:  {max_prob_diff:.2e}  (expected < 1e-5)")

    if agreement < 1.0:
        raise RuntimeError(
            f"ONNX class agreement is {agreement:.6f} — conversion produced wrong predictions!"
        )
    print("  ✓ ONNX predictions verified — identical to LightGBM")


def export_web(bst, model_path: Path, out_dir: Path, opset: int = 15) -> None:
    """Convert to ONNX, gzip it, verify accuracy, write manifest."""
    try:
        from onnxmltools.utils import save_model as save_onnx
    except ImportError:
        print("ERROR: onnxmltools not installed. pip install onnx onnxmltools")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_model, opset_used = _convert_to_onnx(bst, opset)

    onnx_path = out_dir / (model_path.stem + ".onnx")
    save_onnx(onnx_model, str(onnx_path))
    print(f"  ONNX written:  {onnx_path.name} ({_mb(onnx_path)})")

    # gzip (this is what the browser fetches)
    gz_path = out_dir / (model_path.stem + ".onnx.gz")
    with open(onnx_path, "rb") as fin, gzip.open(gz_path, "wb", compresslevel=9) as fout:
        fout.write(fin.read())
    print(f"  gzip written:  {gz_path.name} ({_mb(gz_path)})")

    # Accuracy check
    _verify_onnx(bst, onnx_path)

    # Manifest
    manifest = {
        "version": MODEL_VERSION,
        "exported": datetime.now(timezone.utc).isoformat(),
        "onnx_file": onnx_path.name,
        "onnx_gz_file": gz_path.name,
        "onnx_opset": opset_used,
        "num_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "feature_names": _feature_names(bst),
        "class_meaning": "vowel_index_from_start_0based",
        "serving_note": (
            "Serve .onnx.gz with 'Content-Encoding: gzip' and 'Content-Type: application/octet-stream'. "
            "The browser decompresses transparently. Load with onnxruntime-web InferenceSession."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  manifest.json written")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Luscinia v1.0 LightGBM model for Python service and web browser."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=_DEFAULT_MODEL,
        help=f"Path to the trained .lgb file (default: {_DEFAULT_MODEL})",
    )
    p.add_argument(
        "--out-py",
        type=Path,
        default=_DEFAULT_OUT_PY,
        help="Output directory for the Python service artifact",
    )
    p.add_argument(
        "--out-web",
        type=Path,
        default=_DEFAULT_OUT_WEB,
        help="Output directory for the web (ONNX) artifact",
    )
    p.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip ONNX/web export (useful if onnxmltools not installed yet)",
    )
    p.add_argument(
        "--onnx-opset",
        type=int,
        default=15,
        help="ONNX opset version (default: 15, compatible with onnxruntime-web ≥ 1.16)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.model.exists():
        print(f"ERROR: model file not found: {args.model}")
        sys.exit(1)

    bst = _load_lgb(args.model)

    # ── Python artifact ────────────────────────────────────────────────────
    print(f"\n[1/2] Python service artifact → {args.out_py}")
    export_python(bst, args.model, args.out_py)

    # ── Web artifact ───────────────────────────────────────────────────────
    if args.skip_web:
        print("\n[2/2] Web export skipped (--skip-web)")
    else:
        print(f"\n[2/2] Web artifact (ONNX) → {args.out_web}")
        export_web(bst, args.model, args.out_web, opset=args.onnx_opset)

    print("\nDone.")
    print()
    print("Deployment summary:")
    print(f"  Python service: load with  lgb.Booster(model_file='.../{args.model.name}')")
    if not args.skip_web:
        web_gz = args.out_web / (args.model.stem + ".onnx.gz")
        if web_gz.exists():
            print(f"  Web (browser):  serve '{web_gz.name}' ({_mb(web_gz)}) with")
            print("                  Content-Encoding: gzip")
            print("                  Content-Type: application/octet-stream")
            print("                  then load with onnxruntime-web InferenceSession")


if __name__ == "__main__":
    main()
