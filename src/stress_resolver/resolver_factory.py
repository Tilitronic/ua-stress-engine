"""Stress resolver factory — auto-configure DB + optional ML resolver.

Usage
-----
    from src.nlp.pipeline.resolver_factory import create_pipeline_kwargs
    from src.nlp.pipeline import UkrainianPipeline

    # Auto mode: hybrid (DB + ML) if LightGBM + model available,
    # falls back to db-only silently.
    pipeline = UkrainianPipeline(**create_pipeline_kwargs())

    # Explicit hybrid — raises if prerequisites missing
    pipeline = UkrainianPipeline(**create_pipeline_kwargs(mode="hybrid"))

Auto mode ("auto") enables hybrid when BOTH of:
  1. ``lightgbm`` package is importable
  2. Default universal model file exists on disk

Default model
-------------
luscinia-lgbm-str-ua-univ-v1 / P3_0017_FINAL_FULLDATA / P3_0017_full.lgb
Feature builder: ``build_features_universal`` (132 features, universal model)
"""

from __future__ import annotations

import importlib
from logging import getLogger
from pathlib import Path
from typing import Optional

logger = getLogger(__name__)

# ── Default model path (universal, 132-feature, luscinia-lgbm-str-ua-univ-v1) ─
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH: Path = (
    _PROJECT_ROOT
    / "src/stress_prediction/lightGbm/artifacts"
    / "luscinia-lgbm-str-ua-univ-v1"
    / "P3_0017_FINAL_FULLDATA"
    / "P3_0017_full.lgb"
)


# ── Availability probes ───────────────────────────────────────────────────────

def is_lightgbm_available() -> bool:
    """Return True if the ``lightgbm`` package can be imported."""
    try:
        importlib.import_module("lightgbm")
        return True
    except ImportError:
        return False


def is_model_available(model_path: Optional[Path] = None) -> bool:
    """Return True if the model binary exists on disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    return path.is_file()


# ── ML resolver constructor ───────────────────────────────────────────────────

def create_ml_resolver(
    model_path: Optional[Path] = None,
    *,
    high_confidence_threshold: float = 0.80,
    low_confidence_threshold: float = 0.50,
) -> "MLStressResolver":
    """Create a :class:`MLStressResolver` wired to the universal feature builder.

    Uses ``build_features_universal`` (132 features) which matches
    ``luscinia-lgbm-str-ua-univ-v1``.

    Raises
    ------
    ImportError
        If ``lightgbm`` is not installed.
    FileNotFoundError
        If the model binary is not found at *model_path* (or the default).
    """
    if not is_lightgbm_available():
        raise ImportError(
            "create_ml_resolver requires lightgbm — install it with: "
            "conda install -c conda-forge lightgbm"
        )

    from src.stress_prediction.lightGbm.services.feature_service_universal import (
        build_features_universal,
    )
    from src.nlp.pipeline.ml_stress_resolver import MLStressResolver

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"LightGBM model not found: {path}")

    resolver = MLStressResolver(
        model_path=path,
        feature_builder=build_features_universal,
        high_confidence_threshold=high_confidence_threshold,
        low_confidence_threshold=low_confidence_threshold,
    )
    logger.info(
        "resolver_factory: ML resolver ready — %s features, model=%s",
        len(resolver.feature_columns),
        path.name,
    )
    return resolver


# ── Pipeline kwargs factory ───────────────────────────────────────────────────

def create_pipeline_kwargs(
    mode: str = "auto",
    model_path: Optional[Path] = None,
    ml_confidence_threshold: float = 0.70,
) -> dict:
    """Return kwargs for :class:`UkrainianPipeline` for the requested mode.

    Parameters
    ----------
    mode:
        ``"auto"``   — hybrid when lgb + model available; db-only otherwise *(default)*
        ``"db"``     — always db-only, no ML
        ``"hybrid"`` — force hybrid; raises if prerequisites absent
        ``"ml"``     — force ml-only; raises if prerequisites absent
    model_path:
        Override model path (default: ``DEFAULT_MODEL_PATH``).
    ml_confidence_threshold:
        Hybrid cutoff: below this probability the DB resolver is tried first.

    Returns
    -------
    dict
        Kwargs to unpack into ``UkrainianPipeline(**kwargs)``.

    Raises
    ------
    ValueError
        If *mode* is not one of the accepted values.
    ImportError
        If ``mode in {"hybrid","ml"}`` and lightgbm is not installed.
    FileNotFoundError
        If ``mode in {"hybrid","ml"}`` and model binary is missing.
    """
    mode = mode.lower()
    if mode not in {"auto", "db", "hybrid", "ml"}:
        raise ValueError(f"mode must be auto|db|hybrid|ml, got: {mode!r}")

    if mode == "db":
        return {"stress_mode": "db"}

    lgb_ok = is_lightgbm_available()
    model_ok = is_model_available(model_path)

    if mode == "auto":
        if lgb_ok and model_ok:
            ml_resolver = create_ml_resolver(
                model_path=model_path,
                high_confidence_threshold=0.80,
                low_confidence_threshold=0.50,
            )
            logger.info("resolver_factory: auto → hybrid mode enabled")
            return {
                "stress_mode": "hybrid",
                "ml_resolver": ml_resolver,
                "ml_confidence_threshold": ml_confidence_threshold,
            }
        logger.info(
            "resolver_factory: auto → db-only mode (lgb_ok=%s, model_ok=%s)",
            lgb_ok, model_ok,
        )
        return {"stress_mode": "db"}

    # "hybrid" or "ml" — hard requirements
    if not lgb_ok:
        raise ImportError(
            f"stress_mode={mode!r} requires lightgbm — install it with: "
            "conda install -c conda-forge lightgbm"
        )
    if not model_ok:
        path = model_path or DEFAULT_MODEL_PATH
        raise FileNotFoundError(
            f"stress_mode={mode!r} requires model binary at {path}"
        )

    ml_resolver = create_ml_resolver(
        model_path=model_path,
        high_confidence_threshold=0.80,
        low_confidence_threshold=0.50,
    )
    return {
        "stress_mode": mode,
        "ml_resolver": ml_resolver,
        "ml_confidence_threshold": ml_confidence_threshold,
    }
