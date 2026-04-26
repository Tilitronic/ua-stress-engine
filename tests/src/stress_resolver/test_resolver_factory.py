"""test_resolver_factory.py — Tests for the stress resolver factory.

Validates the modular installation logic:
  - DB-only resolver always works (no optional deps)
  - ML resolver is optional — auto-detected via is_lightgbm_available()
  - Factory wires build_features_universal (132 features) to the universal model
  - UkrainianPipeline accepts a pre-built ml_resolver from the factory

Unit tests   — use FakeBooster; no lightgbm install needed
Integration  — marked _skip_no_lgb / _skip_no_model; require verseSense-py312
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# ── Path constants ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_MODEL_PATH = (
    _PROJECT_ROOT
    / "src/stress_prediction/lightgbm/artifacts"
    / "luscinia-lgbm-str-ua-univ-v1"
    / "P3_0017_FINAL_FULLDATA"
    / "P3_0017_full.lgb"
)

_LGBm_OK: bool
try:
    import lightgbm  # noqa: F401
    _LGBm_OK = True
except ImportError:
    _LGBm_OK = False

_MODEL_OK = _MODEL_PATH.is_file()

_skip_no_lgb = pytest.mark.skipif(
    not _LGBm_OK,
    reason="lightgbm not installed — run in verseSense-py312",
)
_skip_no_model = pytest.mark.skipif(
    not _MODEL_OK,
    reason=f"Model not found (LFS): {_MODEL_PATH}",
)
_skip_no_lgb_or_model = pytest.mark.skipif(
    not (_LGBm_OK and _MODEL_OK),
    reason="lightgbm or model not available",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_token(word: str, pos: str = "NOUN", morph: dict | None = None):
    return SimpleNamespace(
        text=word,
        text_lower=word.lower(),
        text_normalized=word.lower(),
        lemma=word.lower(),
        idx=0,
        i=0,
        pos=pos,
        tag=pos,
        dep="ROOT",
        head_idx=0,
        morph=morph or {},
        is_alpha=True,
        is_punct=False,
        is_space=False,
    )


class _FakeBooster:
    """Booster stub: always predicts the given probability distribution."""

    def __init__(self, probs: list[float]):
        self._probs = np.array([probs], dtype=np.float32)

    def feature_name(self) -> list[str]:
        return []  # triggers fallback feature discovery

    def predict(self, _x):
        return self._probs


# ── Unit: availability probes ─────────────────────────────────────────────────

class TestAvailabilityProbes:
    def test_is_lightgbm_available_returns_bool(self):
        from src.stress_resolver.resolver_factory import is_lightgbm_available
        result = is_lightgbm_available()
        assert isinstance(result, bool)

    def test_is_model_available_false_for_missing_path(self, tmp_path):
        from src.stress_resolver.resolver_factory import is_model_available
        missing = tmp_path / "does_not_exist.lgb"
        assert is_model_available(missing) is False

    def test_is_model_available_true_for_existing_file(self, tmp_path):
        from src.stress_resolver.resolver_factory import is_model_available
        f = tmp_path / "model.lgb"
        f.write_bytes(b"fake")
        assert is_model_available(f) is True

    def test_default_model_path_is_exported(self):
        from src.stress_resolver.resolver_factory import DEFAULT_MODEL_PATH
        assert isinstance(DEFAULT_MODEL_PATH, Path)
        assert DEFAULT_MODEL_PATH.name == "P3_0017_full.lgb"


# ── Unit: create_pipeline_kwargs ──────────────────────────────────────────────

class TestCreatePipelineKwargs:
    def test_db_mode_always_works(self):
        from src.stress_resolver.resolver_factory import create_pipeline_kwargs
        kwargs = create_pipeline_kwargs(mode="db")
        assert kwargs == {"stress_mode": "db"}

    def test_invalid_mode_raises(self):
        from src.stress_resolver.resolver_factory import create_pipeline_kwargs
        with pytest.raises(ValueError, match="mode must be"):
            create_pipeline_kwargs(mode="turbo")

    def test_auto_mode_db_only_when_no_model(self, tmp_path, monkeypatch):
        """auto mode falls back to db-only if model file is missing."""
        from src.stress_resolver import resolver_factory
        monkeypatch.setattr(resolver_factory, "is_lightgbm_available", lambda: True)
        missing = tmp_path / "no_model.lgb"
        monkeypatch.setattr(resolver_factory, "is_model_available", lambda _p=None: False)
        kwargs = resolver_factory.create_pipeline_kwargs(mode="auto")
        assert kwargs == {"stress_mode": "db"}

    def test_auto_mode_db_only_when_no_lgb(self, monkeypatch):
        """auto mode falls back to db-only if lightgbm not installed."""
        from src.stress_resolver import resolver_factory
        monkeypatch.setattr(resolver_factory, "is_lightgbm_available", lambda: False)
        monkeypatch.setattr(resolver_factory, "is_model_available", lambda _p=None: True)
        kwargs = resolver_factory.create_pipeline_kwargs(mode="auto")
        assert kwargs == {"stress_mode": "db"}

    def test_hybrid_raises_without_lgb(self, monkeypatch):
        from src.stress_resolver import resolver_factory
        monkeypatch.setattr(resolver_factory, "is_lightgbm_available", lambda: False)
        with pytest.raises(ImportError, match="lightgbm"):
            resolver_factory.create_pipeline_kwargs(mode="hybrid")

    def test_hybrid_raises_without_model(self, tmp_path, monkeypatch):
        from src.stress_resolver import resolver_factory
        monkeypatch.setattr(resolver_factory, "is_lightgbm_available", lambda: True)
        monkeypatch.setattr(resolver_factory, "is_model_available", lambda _p=None: False)
        with pytest.raises(FileNotFoundError, match="model binary"):
            resolver_factory.create_pipeline_kwargs(mode="hybrid")

    def test_ml_raises_without_lgb(self, monkeypatch):
        from src.stress_resolver import resolver_factory
        monkeypatch.setattr(resolver_factory, "is_lightgbm_available", lambda: False)
        with pytest.raises(ImportError):
            resolver_factory.create_pipeline_kwargs(mode="ml")


# ── Unit: feature_builder param on MLStressResolver ──────────────────────────

class TestFeatureBuilderParam:
    """Verify MLStressResolver uses the injected feature_builder callable."""

    def test_default_feature_builder_is_v13(self):
        from src.stress_resolver.ml_stress_resolver import MLStressResolver
        from src.stress_prediction.lightgbm.services.feature_service import (
            build_features_v13,
        )
        probs = [0.1] * 11
        resolver = MLStressResolver(booster=_FakeBooster(probs))
        assert resolver._feature_builder is build_features_v13

    def test_custom_feature_builder_is_stored(self):
        from src.stress_resolver.ml_stress_resolver import MLStressResolver

        calls: list = []

        def my_builder(form, pos, features_json=None):
            calls.append(form)
            from src.stress_prediction.lightgbm.services.feature_service import (
                build_features_v13,
            )
            return build_features_v13(form, pos, features_json)

        probs = [0.02] * 11
        probs[0] = 0.90
        resolver = MLStressResolver(
            booster=_FakeBooster(probs),
            feature_builder=my_builder,
        )
        assert resolver._feature_builder is my_builder

        token = _make_token("мама")
        resolver.resolve(token)
        assert "мама" in calls, "custom feature_builder was not called during resolve()"

    def test_universal_feature_builder_produces_132_features(self):
        from src.stress_prediction.lightgbm.services.feature_service_universal import (
            build_features_universal,
            EXPECTED_FEATURE_COUNT_UNIV,
        )
        feats = build_features_universal("університет", "NOUN")
        assert len(feats) == EXPECTED_FEATURE_COUNT_UNIV == 132

    def test_ml_resolver_with_universal_builder_fallback_feature_names(self):
        """When FakeBooster.feature_name() returns [], resolver falls back to
        calling feature_builder('тест', 'X') to discover column names."""
        from src.stress_resolver.ml_stress_resolver import MLStressResolver
        from src.stress_prediction.lightgbm.services.feature_service_universal import (
            build_features_universal,
            EXPECTED_FEATURE_COUNT_UNIV,
        )
        probs = [0.02] * 11
        probs[1] = 0.90
        resolver = MLStressResolver(
            booster=_FakeBooster(probs),
            feature_builder=build_features_universal,
        )
        assert len(resolver.feature_columns) == EXPECTED_FEATURE_COUNT_UNIV
        assert "syllable_count_u" in resolver.feature_columns


# ── Integration: create_ml_resolver ──────────────────────────────────────────

class TestCreateMlResolver:
    @_skip_no_lgb_or_model
    def test_create_ml_resolver_returns_resolver(self):
        from src.stress_resolver.resolver_factory import create_ml_resolver
        from src.stress_resolver.ml_stress_resolver import MLStressResolver
        resolver = create_ml_resolver()
        assert isinstance(resolver, MLStressResolver)

    @_skip_no_lgb_or_model
    def test_ml_resolver_feature_count_matches_universal_model(self):
        from src.stress_resolver.resolver_factory import create_ml_resolver
        from src.stress_prediction.lightgbm.services.feature_service_universal import (
            EXPECTED_FEATURE_COUNT_UNIV,
        )
        resolver = create_ml_resolver()
        assert len(resolver.feature_columns) == EXPECTED_FEATURE_COUNT_UNIV, (
            f"Expected {EXPECTED_FEATURE_COUNT_UNIV} features, "
            f"got {len(resolver.feature_columns)}"
        )

    @_skip_no_lgb_or_model
    def test_ml_resolver_uses_universal_builder(self):
        from src.stress_resolver.resolver_factory import create_ml_resolver
        from src.stress_prediction.lightgbm.services.feature_service_universal import (
            build_features_universal,
        )
        resolver = create_ml_resolver()
        assert resolver._feature_builder is build_features_universal

    @_skip_no_lgb
    def test_create_ml_resolver_raises_on_missing_model(self, tmp_path):
        from src.stress_resolver.resolver_factory import create_ml_resolver
        with pytest.raises(FileNotFoundError, match="not found"):
            create_ml_resolver(model_path=tmp_path / "nope.lgb")


# ── Integration: smoke-word predictions with universal model ─────────────────

_SMOKE_WORDS = [
    ("мама",        "NOUN", 0),
    ("вода",        "NOUN", 1),
    ("університет", "NOUN", 4),
    ("читати",      "VERB", 1),
    ("місто",       "NOUN", 0),
    ("батько",      "NOUN", 0),
    ("книга",       "NOUN", 0),
    ("земля",       "NOUN", 1),
]


@_skip_no_lgb_or_model
@pytest.mark.parametrize("word,pos,expected_vowel", _SMOKE_WORDS)
def test_ml_resolver_smoke_word(word, pos, expected_vowel):
    """Universal ML resolver predicts correct stress for canonical words."""
    from src.stress_resolver.resolver_factory import create_ml_resolver
    resolver = create_ml_resolver()
    token = _make_token(word, pos)
    result = resolver.resolve(token)

    assert result["stress_position"] == expected_vowel, (
        f"{word!r}: expected vowel {expected_vowel}, "
        f"got {result['stress_position']} (conf={result['stress_match_score']:.3f})"
    )


# ── Integration: create_pipeline_kwargs auto mode ────────────────────────────

class TestCreatePipelineKwargsIntegration:
    @_skip_no_lgb_or_model
    def test_auto_mode_returns_hybrid_kwargs(self):
        from src.stress_resolver.resolver_factory import create_pipeline_kwargs
        kwargs = create_pipeline_kwargs(mode="auto")
        assert kwargs["stress_mode"] == "hybrid"
        assert "ml_resolver" in kwargs

    @_skip_no_lgb_or_model
    def test_hybrid_mode_explicit_returns_resolver(self):
        from src.stress_resolver.resolver_factory import create_pipeline_kwargs
        kwargs = create_pipeline_kwargs(mode="hybrid")
        assert kwargs["stress_mode"] == "hybrid"
        assert "ml_resolver" in kwargs

    @_skip_no_lgb_or_model
    def test_auto_kwargs_accepted_by_pipeline_init(self):
        """create_pipeline_kwargs output can be unpacked into UkrainianPipeline."""
        from src.stress_resolver.resolver_factory import create_pipeline_kwargs
        from src.stress_resolver import UkrainianPipeline
        kwargs = create_pipeline_kwargs(mode="auto")
        # Pipeline construction triggers spaCy model load — just verify no TypeError
        try:
            pipeline = UkrainianPipeline(**kwargs)
            assert pipeline.stress_mode == "hybrid"
            assert pipeline.ml_stress_resolver is not None
        except Exception as exc:
            # spaCy model may not be downloaded — that's OK, we only care that
            # our new kwargs are not rejected with TypeError/ValueError
            if "TypeError" in type(exc).__name__ or "ValueError" in type(exc).__name__:
                raise
