#!/usr/bin/env python3
"""ML-based stress resolver for Ukrainian pipeline tokens.

This resolver reuses the same runtime token objects as the existing DB resolver
(`TokenData`) and the same feature builder used during model training
(`build_features_v13`).

Design goals:
- Keep inference path as close as possible to real pipeline runtime
- Reuse token-level morphology produced by spaCy tokenization service
- Maintain output compatibility with `StressResolver.resolve()`
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.nlp.tokenization_service.types import TokenData
from src.stress_prediction.lightgbm.services.constants import MAX_VOWEL_CLASS
from src.stress_prediction.lightgbm.services.feature_service import (
    build_features_v13,
    find_vowels,
)

logger = getLogger(__name__)


class MLStressResolver:
    """Resolve stress with a trained lightgbm multiclass model.

    Expected model objective: multiclass over vowel index classes
    ``0..MAX_VOWEL_CLASS``.
    """

    VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        booster=None,
        high_confidence_threshold: float = 0.80,
        low_confidence_threshold: float = 0.50,
        feature_builder: Optional[Callable] = None,
    ):
        """Initialize ML resolver.

        Args:
            model_path: Path to serialized lightgbm model file.
            booster: Pre-initialized booster-like object (useful for tests).
            high_confidence_threshold: Probability threshold for ``exact``.
            low_confidence_threshold: Probability threshold for ``partial``.
            feature_builder: Callable with signature ``(form, pos, features_json)``
                that returns a ``Dict[str, float]`` feature vector.  Defaults to
                ``build_features_v13`` (97 features, backward-compatible).  Pass
                ``build_features_universal`` for the 132-feature universal model.
        """
        if booster is None and model_path is None:
            raise ValueError("Either booster or model_path must be provided")

        if booster is None:
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(model_path))
            self.model_path = Path(model_path)
        else:
            self.booster = booster
            self.model_path = Path(model_path) if model_path else None

        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self._feature_builder: Callable = feature_builder or build_features_v13

        feature_names = []
        if hasattr(self.booster, "feature_name"):
            try:
                feature_names = list(self.booster.feature_name())
            except Exception:
                feature_names = []

        if not feature_names:
            fallback = self._feature_builder("тест", "X")
            feature_names = list(fallback.keys())

        self.feature_columns = feature_names
        logger.info(
            "MLStressResolver initialized (%s features)",
            len(self.feature_columns),
        )

    def resolve(self, token: TokenData) -> Dict:
        """Resolve token stress using model prediction."""
        if not token.is_alpha or token.is_punct or token.is_space:
            return self._no_stress_result()

        word = token.text_normalized
        if not word:
            return self._no_stress_result()

        features_json = None
        if token.morph:
            features_json = json.dumps(token.morph, ensure_ascii=False)

        features = self._feature_builder(word, token.pos or "X", features_json)
        fvec = np.array(
            [[features.get(column, 0.0) for column in self.feature_columns]],
            dtype=np.float32,
        )

        try:
            pred_proba = self.booster.predict(fvec)
        except Exception as exc:
            logger.warning("ML prediction failed for '%s': %s", word, exc)
            return self._no_stress_result()

        probs = np.asarray(pred_proba)
        if probs.ndim == 2:
            probs = probs[0]

        if probs.size == 0:
            return self._no_stress_result()

        predicted = int(np.argmax(probs))
        confidence_score = float(probs[predicted])

        vowels = find_vowels(word.lower())
        if not vowels:
            return self._no_stress_result()

        if predicted > MAX_VOWEL_CLASS or predicted >= len(vowels):
            logger.debug(
                "Predicted class %s out of range for '%s' (%s vowels)",
                predicted,
                word,
                len(vowels),
            )
            return self._no_stress_result()

        if confidence_score >= self.high_confidence_threshold:
            confidence = "exact"
        elif confidence_score >= self.low_confidence_threshold:
            confidence = "partial"
        else:
            confidence = "fallback"

        stress_pattern = self._add_stress_mark(word, predicted)
        return {
            "stress_position": predicted,
            "stress_pattern": stress_pattern,
            "stress_confidence": confidence,
            "stress_match_score": round(confidence_score, 6),
        }

    def _no_stress_result(self) -> Dict:
        return {
            "stress_position": None,
            "stress_pattern": "",
            "stress_confidence": "none",
            "stress_match_score": 0.0,
        }

    def _add_stress_mark(self, word: str, stress_position: int) -> str:
        vowel_positions = [i for i, char in enumerate(word) if char in self.VOWELS]
        if stress_position >= len(vowel_positions):
            return word
        insert_pos = vowel_positions[stress_position] + 1
        return word[:insert_pos] + "\u0301" + word[insert_pos:]
