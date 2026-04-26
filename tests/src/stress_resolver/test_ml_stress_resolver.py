import os
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ml_module_path = ROOT / "src" / "stress_resolver" / "ml_stress_resolver.py"
spec = importlib.util.spec_from_file_location("ml_stress_resolver", ml_module_path)
ml_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(ml_module)
MLStressResolver = ml_module.MLStressResolver


class FakeBooster:
    def __init__(self, probs):
        self._probs = np.array([probs], dtype=np.float32)

    def feature_name(self):
        return []

    def predict(self, _):
        return self._probs


def make_token(word: str, pos: str = "NOUN", morph=None):
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


def test_predicts_stress_position_and_pattern():
    probs = [0.02] * 11
    probs[1] = 0.91
    resolver = MLStressResolver(booster=FakeBooster(probs))

    token = make_token("мама")
    result = resolver.resolve(token)

    assert result["stress_position"] == 1
    assert result["stress_confidence"] == "exact"
    assert result["stress_pattern"] == "мама́"
    assert result["stress_match_score"] >= 0.9


def test_returns_none_when_predicted_class_exceeds_vowels():
    probs = [0.01] * 11
    probs[5] = 0.95
    resolver = MLStressResolver(booster=FakeBooster(probs))

    token = make_token("мама")
    result = resolver.resolve(token)

    assert result["stress_position"] is None
    assert result["stress_confidence"] == "none"


def test_non_alpha_token_returns_none():
    probs = [0.02] * 11
    probs[0] = 0.95
    resolver = MLStressResolver(booster=FakeBooster(probs))

    token = make_token(".")
    token.is_alpha = False
    token.is_punct = True
    result = resolver.resolve(token)

    assert result["stress_position"] is None
    assert result["stress_pattern"] == ""
