#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for model_export.py — Luscinia v1.0 production export
============================================================

Covers:
  1. Feature pipeline  — build_features_universal() produces exactly 132 features
                         for a variety of real Ukrainian words
  2. LightGBM inference — the .lgb model loads, predicts correct stress for
                          known words, and output shape is consistent
  3. Stress index → character position  — utility that maps model output back
                                           to the stressed character in the word
  4. Python service contract  — the exact API a backend service should use
  5. ONNX export (optional) — if onnx + onnxmltools installed, verify the
                               exported .onnx gives identical argmax to lgb
  6. ONNX web serving notes — structural check of the manifest.json

Run from project root:
    pytest tests/src/stress_prediction/test_model_export.py -v

Or with coverage:
    pytest tests/src/stress_prediction/test_model_export.py -v --tb=short
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LGBM_DIR     = PROJECT_ROOT / "src" / "stress_prediction" / "lightGbm"
SERVICES_DIR = LGBM_DIR / "services"
ARTIFACTS_DIR = LGBM_DIR / "artifacts" / "luscinia-lgbm-str-ua-univ-v1"
FINAL_DIR    = ARTIFACTS_DIR / "P3_0017_FINAL_FULLDATA"
MODEL_PATH   = FINAL_DIR / "P3_0017_full.lgb"
META_PATH    = FINAL_DIR / "meta.json"
WEB_DIR      = ARTIFACTS_DIR / "web"

sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def booster():
    """Load the LightGBM model once for the whole test session."""
    lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found (LFS asset): {MODEL_PATH}")
    bst = lgb.Booster(model_file=str(MODEL_PATH))
    return bst


@pytest.fixture(scope="session")
def meta():
    assert META_PATH.exists(), f"meta.json not found: {META_PATH}"
    return json.loads(META_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def features_for(form: str, pos: str = "NOUN") -> dict:
    from src.stress_prediction.lightGbm.services.feature_service_universal import (
        build_features_universal,
    )
    return build_features_universal(form, pos)


def predict_stress_index(booster, form: str, pos: str = "NOUN") -> int:
    """
    Full inference pipeline for a single word.

    Returns
    -------
    int
        0-based index of the stressed vowel from the start of the word.
        e.g. "замок" (castle) → stressed on vowel 0 → returns 0
             "замок" (lock)   → stressed on vowel 1 → returns 1
    """
    feat_dict = features_for(form, pos)
    X = np.array(list(feat_dict.values()), dtype=np.float32).reshape(1, -1)
    probs = booster.predict(X)          # shape (1, 11)
    return int(probs.argmax(axis=1)[0])


def vowel_positions(word: str) -> list[int]:
    """Return 0-based character indices of Ukrainian vowels in *word*."""
    VOWELS = set("аеєиіїоуюя")
    return [i for i, c in enumerate(word.lower()) if c in VOWELS]


def stressed_char_index(word: str, vowel_index: int) -> Optional[int]:
    """Convert vowel_index (0-based count of vowels) → character position."""
    positions = vowel_positions(word)
    if not positions or vowel_index >= len(positions):
        return None
    return positions[vowel_index]


# ---------------------------------------------------------------------------
# 1. Feature pipeline
# ---------------------------------------------------------------------------

class TestFeaturePipeline:
    """Verify that build_features_universal() produces a well-formed vector."""

    EXPECTED_FEATURE_COUNT = 132

    def test_feature_count_noun(self):
        feat = features_for("навчання", "NOUN")
        assert len(feat) == self.EXPECTED_FEATURE_COUNT, (
            f"Expected {self.EXPECTED_FEATURE_COUNT} features, got {len(feat)}"
        )

    def test_feature_count_verb(self):
        feat = features_for("читати", "VERB")
        assert len(feat) == self.EXPECTED_FEATURE_COUNT

    def test_feature_count_adj(self):
        feat = features_for("київський", "ADJ")
        assert len(feat) == self.EXPECTED_FEATURE_COUNT

    def test_feature_count_short_word(self):
        # 2-syllable word — minimum supported
        feat = features_for("книга", "NOUN")
        assert len(feat) == self.EXPECTED_FEATURE_COUNT

    def test_feature_count_long_word(self):
        # 6-syllable word
        feat = features_for("загальноосвітній", "ADJ")
        assert len(feat) == self.EXPECTED_FEATURE_COUNT

    def test_all_values_are_finite_floats(self):
        feat = features_for("університет", "NOUN")
        vals = list(feat.values())
        assert all(isinstance(v, (int, float)) for v in vals)
        arr = np.array(vals, dtype=np.float32)
        assert np.all(np.isfinite(arr)), "Feature vector contains NaN or Inf"

    def test_syllable_count_feature(self):
        # "університет" → у-ні-вер-си-тет → 5 vowels
        feat = features_for("університет", "NOUN")
        assert feat["syllable_count_u"] == 5

    def test_syllable_count_feature_2syl(self):
        feat = features_for("місто", "NOUN")
        assert feat["syllable_count_u"] == 2

    def test_vy_prefix_flag(self):
        feat_vy  = features_for("вивчення", "NOUN")
        feat_nov = features_for("навчання", "NOUN")
        assert feat_vy["has_vy_prefix_u"]  == 1
        assert feat_nov["has_vy_prefix_u"] == 0

    def test_adcat_suffix_flag(self):
        feat_adj   = features_for("київський", "ADJ")
        feat_other = features_for("красивий",  "ADJ")
        assert feat_adj["has_adcat_suffix_u"]   == 1
        assert feat_other["has_adcat_suffix_u"] == 0

    def test_apostrophe_word(self):
        # Words with apostrophe should parse correctly
        feat = features_for("п\u02BCятниця", "NOUN")   # п'ятниця with U+02BC
        assert len(feat) == self.EXPECTED_FEATURE_COUNT
        assert np.all(np.isfinite(np.array(list(feat.values()), dtype=np.float32)))

    def test_reproducible(self):
        f1 = features_for("навчання", "NOUN")
        f2 = features_for("навчання", "NOUN")
        assert f1 == f2


# ---------------------------------------------------------------------------
# 2. LightGBM model: shape and output
# ---------------------------------------------------------------------------

class TestLGBMModel:
    """Verify the model loads and produces sensible predictions."""

    def test_model_loads(self, booster):
        assert booster is not None

    def test_num_features(self, booster):
        assert booster.num_feature() == 132

    def test_num_classes(self, booster):
        assert booster.num_model_per_iteration() == 11

    def test_num_iterations(self, booster):
        assert booster.current_iteration() == 908

    def test_predict_shape_single(self, booster):
        X = np.zeros((1, 132), dtype=np.float32)
        out = booster.predict(X)
        assert out.shape == (1, 11), f"Expected (1, 11), got {out.shape}"

    def test_predict_shape_batch(self, booster):
        X = np.zeros((50, 132), dtype=np.float32)
        out = booster.predict(X)
        assert out.shape == (50, 11)

    def test_predict_probabilities_sum_to_one(self, booster):
        rng = np.random.default_rng(1)
        X = rng.random((200, 132)).astype(np.float32)
        probs = booster.predict(X)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5,
                                   err_msg="Softmax probabilities do not sum to 1")

    def test_predict_all_probs_non_negative(self, booster):
        rng = np.random.default_rng(2)
        X = rng.random((200, 132)).astype(np.float32)
        probs = booster.predict(X)
        assert np.all(probs >= 0.0)

    def test_meta_matches_model(self, booster, meta):
        # meta.json stores num_classes and num_boost_round; num_features is in the model itself
        assert booster.num_feature() == 132                                # hard constant
        assert booster.num_model_per_iteration() == meta["num_classes"]   # 11
        assert booster.current_iteration() == meta["num_boost_round"]     # 908


# ---------------------------------------------------------------------------
# 3. Known-word stress predictions
# ---------------------------------------------------------------------------

# Ground-truth: (word, pos, expected_vowel_index_0based)
# Vowel index is counted from the START of the word, 0-based.
#
# Examples:
#   "місто"      → м[і]сто      → vowel 0 stressed  → index 0
#   "вода"       → вод[а]       → vowel 1 stressed  → index 1
#   "навчання"   → навч[а]ння   → vowel 2 stressed  → index 2
#   "університет"→ університ[е]т→ vowel 4 stressed  → index 4
KNOWN_WORDS = [
    # word,             pos,    expected_vowel_idx
    # Vowel index = 0-based count of Ukrainian vowels from start of word.
    # е.g. "місто"→ [м,і,с,т,о] vowels=[1,4] → index 0 = 'і' → М[І]СТО
    ("місто",          "NOUN",  0),   # М[І]СТО → vowels at [1,4] → stress on 'і' (idx 0)
    ("вода",           "NOUN",  1),   # ВОД[А]  → vowels at [1,3] → stress on 'а' (idx 1)
    ("навчання",       "NOUN",  1),   # НА[В]ЧАННЯ → навч[А]ння → vowels at [1,4,7] → 'а' (idx 1)
    ("читати",         "VERB",  1),   # ЧИТ[А]ТИ → чи-ТА-ти → vowels at [1,3,5] → 'а' (idx 1)
    ("університет",    "NOUN",  4),   # УНІВЕРСИТ[Е]Т → stress on 5th vowel (idx 4)
    ("київський",      "ADJ",   0),   # [КИ]ЇВСЬКИЙ → КИ stressed → vowels at [1,2,7] → 'и' (idx 0)
    ("вивчення",       "NOUN",  0),   # [ВИ]ВЧЕННЯ → ВИ-prefix stressed → vowels at [1,4,7] → 'и' (idx 0)
]


class TestKnownWordStress:
    """The model must predict the correct vowel index for well-known words."""

    @pytest.mark.parametrize("word,pos,expected", KNOWN_WORDS)
    def test_known_word(self, booster, word, pos, expected):
        predicted = predict_stress_index(booster, word, pos)
        assert predicted == expected, (
            f"'{word}' ({pos}): expected vowel index {expected}, got {predicted}\n"
            f"  vowels at: {vowel_positions(word)}\n"
            f"  stressed char: {stressed_char_index(word, predicted)!r} "
            f"(predicted) vs {stressed_char_index(word, expected)!r} (expected)"
        )


# ---------------------------------------------------------------------------
# 4. Python service contract
# ---------------------------------------------------------------------------

class TestPythonServiceContract:
    """
    Documents and validates the exact API a Python stress service must use.

    This class is the reference implementation — copy this pattern into your
    backend service.
    """

    def test_full_pipeline_single_word(self, booster):
        """
        PYTHON SERVICE PATTERN
        ----------------------
        from src.stress_prediction.lightGbm.services.feature_service_universal import (
            build_features_universal,
        )
        import lightgbm as lgb
        import numpy as np

        # 1. Load once at service startup
        bst = lgb.Booster(model_file="path/to/P3_0017_full.lgb")

        # 2. For each word to stress:
        def predict_stress(word: str, pos: str = "NOUN") -> int:
            feat = build_features_universal(word, pos)
            X = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
            probs = bst.predict(X)          # shape (1, 11)
            vowel_idx = int(probs.argmax(axis=1)[0])
            return vowel_idx

        # 3. Map vowel_idx back to a character position:
        VOWELS = set("аеєиіїоуюя")
        vowel_positions = [i for i, c in enumerate(word.lower()) if c in VOWELS]
        stressed_char_pos = vowel_positions[vowel_idx]
        """
        from src.stress_prediction.lightGbm.services.feature_service_universal import (
            build_features_universal,
        )

        word = "навчання"
        pos  = "NOUN"

        # Step 1 — build features
        feat = build_features_universal(word, pos)
        assert len(feat) == 132

        # Step 2 — predict
        X = np.array(list(feat.values()), dtype=np.float32).reshape(1, -1)
        probs = booster.predict(X)
        assert probs.shape == (1, 11)

        vowel_idx = int(probs.argmax(axis=1)[0])

        # Step 3 — map to character position
        VOWELS = set("аеєиіїоуюя")
        vowel_positions_list = [i for i, c in enumerate(word.lower()) if c in VOWELS]
        stressed_char = word[vowel_positions_list[vowel_idx]]

        # навч[А]ння → stress on 2nd vowel (index 1), char='а'
        assert stressed_char == "а", (
            f"Expected stressed char 'а' in навч[а]ння (vowel index 1), got '{stressed_char}' (vowel index {vowel_idx})"
        )

    def test_batch_prediction(self, booster):
        """
        BATCH PATTERN — process many words at once (much faster than one-by-one).

        feat_matrix = np.array(
            [list(build_features_universal(w, p).values()) for w, p in word_pos_pairs],
            dtype=np.float32
        )
        all_probs   = bst.predict(feat_matrix)    # shape (N, 11)
        all_indices = all_probs.argmax(axis=1)    # shape (N,)
        """
        from src.stress_prediction.lightGbm.services.feature_service_universal import (
            build_features_universal,
        )

        words = [
            ("місто",       "NOUN"),
            ("вода",        "NOUN"),
            ("навчання",    "NOUN"),
            ("університет", "NOUN"),
        ]

        feat_matrix = np.array(
            [list(build_features_universal(w, p).values()) for w, p in words],
            dtype=np.float32,
        )
        assert feat_matrix.shape == (4, 132)

        all_probs   = booster.predict(feat_matrix)
        all_indices = all_probs.argmax(axis=1)

        assert all_probs.shape == (4, 11)
        assert all_indices.shape == (4,)

    def test_feature_order_is_stable(self, booster):
        """
        Feature dict is an ordered dict (Python 3.7+).
        Calling values() twice in the same Python process gives the same order.
        """
        from src.stress_prediction.lightGbm.services.feature_service_universal import (
            build_features_universal,
        )
        feat = build_features_universal("навчання", "NOUN")
        keys1 = list(feat.keys())
        keys2 = list(build_features_universal("навчання", "NOUN").keys())
        assert keys1 == keys2

    def test_feature_names_match_model(self, booster):
        """
        The order of features in the dict must match the order the model was
        trained with. Verify using the model's own feature_name() list.
        """
        from src.stress_prediction.lightGbm.services.feature_service_universal import (
            build_features_universal,
        )
        feat = build_features_universal("навчання", "NOUN")
        dict_keys   = list(feat.keys())
        model_names = booster.feature_name()
        assert dict_keys == model_names, (
            "Feature dict key order does not match model's expected feature order!\n"
            f"First mismatch at index "
            f"{next(i for i,(a,b) in enumerate(zip(dict_keys,model_names)) if a!=b)}"
        )


# ---------------------------------------------------------------------------
# 5. ONNX export — optional (skipped if packages not installed)
# ---------------------------------------------------------------------------

import os as _os

try:
    import onnx          # noqa: F401
    import onnxmltools   # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    import onnxruntime   # noqa: F401
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

# ONNX conversion of the 259 MB model takes several minutes.
# Only run when explicitly opted in: PYTEST_RUN_ONNX_EXPORT=1
_RUN_ONNX_EXPORT = _os.environ.get("PYTEST_RUN_ONNX_EXPORT", "").strip() == "1"


@pytest.mark.skipif(
    not _ONNX_AVAILABLE or not _RUN_ONNX_EXPORT,
    reason="ONNX export skipped — set PYTEST_RUN_ONNX_EXPORT=1 to enable (slow, one-time task)",
)
class TestONNXExport:
    """Verify ONNX conversion and accuracy against the LightGBM model."""

    @pytest.fixture(scope="class")
    def onnx_path(self, tmp_path_factory, booster):
        import onnxmltools
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        from onnxmltools.utils import save_model as save_onnx

        tmp = tmp_path_factory.mktemp("onnx_export")
        onnx_file = tmp / "P3_0017.onnx"

        initial_types = [("float_input", FloatTensorType([None, 132]))]
        onnx_model = convert_lightgbm(booster, initial_types=initial_types, target_opset=15)
        save_onnx(onnx_model, str(onnx_file))
        return onnx_file

    def test_onnx_file_created(self, onnx_path):
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 10_000_000  # > 10 MB

    def test_onnx_loads(self, onnx_path):
        import onnx
        model = onnx.load(str(onnx_path))
        assert model is not None

    def test_onnx_opset(self, onnx_path):
        import onnx
        model = onnx.load(str(onnx_path))
        # onnxmltools convert_lightgbm fixes opset at 9 regardless of target_opset;
        # onnxruntime-web supports opset 9+, so this is fine.
        assert model.opset_import[0].version >= 9

    def test_onnx_gzip_size(self, onnx_path, tmp_path_factory):
        """gzip ONNX should be well under 50 MB (typically ~30 MB)."""
        tmp = tmp_path_factory.mktemp("onnx_gz")
        gz_path = tmp / "P3_0017.onnx.gz"
        with open(onnx_path, "rb") as fin, gzip.open(gz_path, "wb", compresslevel=9) as fout:
            fout.write(fin.read())
        size_mb = gz_path.stat().st_size / 1024 / 1024
        assert size_mb < 50, f"gzip ONNX is {size_mb:.1f} MB, expected < 50 MB"

    @pytest.mark.skipif(not _ORT_AVAILABLE, reason="onnxruntime not installed")
    def test_onnx_identical_argmax(self, onnx_path, booster):
        """
        ONNX must give exactly the same predicted class as LightGBM on every input.
        This is the critical accuracy check before deploying to web.
        """
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        X = rng.random((2000, 132)).astype(np.float32)

        lgb_classes = booster.predict(X).argmax(axis=1)

        sess     = ort.InferenceSession(str(onnx_path))
        inp_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {inp_name: X})

        onnx_labels = np.array(onnx_out[0], dtype=int)
        agreement   = (lgb_classes == onnx_labels).mean()

        assert agreement == 1.0, (
            f"ONNX class agreement is {agreement:.6f} — "
            f"{int((1-agreement)*2000)} / 2000 inputs disagree"
        )

    @pytest.mark.skipif(not _ORT_AVAILABLE, reason="onnxruntime not installed")
    def test_onnx_prob_diff(self, onnx_path, booster):
        """Probability differences should be negligible (float64→float32 rounding)."""
        import onnxruntime as ort

        rng = np.random.default_rng(1)
        X = rng.random((500, 132)).astype(np.float32)

        lgb_probs = booster.predict(X)

        sess     = ort.InferenceSession(str(onnx_path))
        inp_name = sess.get_inputs()[0].name
        onnx_out = sess.run(None, {inp_name: X})

        onnx_prob_dicts = onnx_out[1]
        onnx_probs = np.array([[d[i] for i in range(11)] for d in onnx_prob_dicts])

        max_diff = np.abs(lgb_probs - onnx_probs).max()
        assert max_diff < 1e-5, f"Max probability diff {max_diff:.2e} exceeds 1e-5"


# ---------------------------------------------------------------------------
# 6. Web serving manifest
# ---------------------------------------------------------------------------

class TestWebManifest:
    """
    If the web export has been run, verify the manifest.json is correct.
    Skipped if web/ directory doesn't exist yet.
    """

    @pytest.fixture
    def manifest(self):
        manifest_path = WEB_DIR / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Web export not yet run — execute: python services/model_export.py")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def test_manifest_version(self, manifest):
        assert manifest["version"] == "luscinia-lgbm-str-ua-univ-v1.0"

    def test_manifest_num_features(self, manifest):
        assert manifest["num_features"] == 132

    def test_manifest_num_classes(self, manifest):
        assert manifest["num_classes"] == 11

    def test_manifest_feature_names_count(self, manifest):
        assert len(manifest["feature_names"]) == 132

    def test_manifest_onnx_gz_file_exists(self, manifest):
        gz_file = WEB_DIR / manifest["onnx_gz_file"]
        assert gz_file.exists(), f"ONNX gz file not found: {gz_file}"

    def test_manifest_onnx_file_exists(self, manifest):
        # The uncompressed .onnx is a derivable artifact (gunzip of .onnx.gz).
        # Only the .gz is stored; skip if the uncompressed file isn't present.
        onnx_file = WEB_DIR / manifest["onnx_file"]
        if not onnx_file.exists():
            pytest.skip("Uncompressed .onnx not present — derivable from .onnx.gz")
        assert onnx_file.exists(), f"ONNX file not found: {onnx_file}"

    def test_manifest_gz_is_valid_gzip(self, manifest):
        gz_file = WEB_DIR / manifest["onnx_gz_file"]
        with gzip.open(gz_file, "rb") as f:
            header = f.read(4)
        # ONNX protobuf magic bytes start with 0x0a (field 1, wire type 2)
        assert len(header) == 4

    def test_manifest_has_serving_note(self, manifest):
        assert "serving_note" in manifest
        assert "Content-Encoding: gzip" in manifest["serving_note"]


# ---------------------------------------------------------------------------
# 7. meta.json integrity
# ---------------------------------------------------------------------------

class TestMetaJson:
    def test_meta_sanity_accuracy(self, meta):
        assert meta["full_data_sanity_accuracy"] >= 0.99

    def test_meta_hand_correct(self, meta):
        assert meta["full_data_hand_correct"] == 192
        assert meta["full_data_hand_total"]   == 197

    def test_meta_boost_rounds(self, meta):
        assert meta["num_boost_round"] == 908

    def test_meta_per_syllable_keys(self, meta):
        for syl in ["2", "3", "4", "5", "6", "7"]:
            assert syl in meta["full_data_per_syllable"]

    def test_meta_per_syllable_4syl_accuracy(self, meta):
        acc = meta["full_data_per_syllable"]["4"]["accuracy"]
        assert acc >= 0.99, f"4-syllable accuracy {acc:.4f} below 99%"
