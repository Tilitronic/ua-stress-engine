"""Tests for Luscinia predictor."""

import pytest
import numpy as np
from luscinia import LusciniaPredictor, predict_stress, featurize


def test_predictor_initialization():
    """Test that predictor initializes correctly."""
    predictor = LusciniaPredictor()
    assert predictor.num_features == 132
    assert predictor.num_classes == 11
    assert predictor.model_version == "luscinia-lgbm-str-ua-univ-v1.0"


def test_single_prediction():
    """Test single word prediction."""
    predictor = LusciniaPredictor()
    
    # Test known words
    assert predictor.predict("мама") == 0  # ма́ма
    assert predictor.predict("університет") == 4  # університе́т


def test_prediction_with_pos():
    """Test prediction with POS tags."""
    predictor = LusciniaPredictor()
    
    # Verb with prefix
    idx = predictor.predict("виходити", pos="VERB")
    assert isinstance(idx, int)
    assert 0 <= idx < 11


def test_batch_prediction():
    """Test batch prediction."""
    predictor = LusciniaPredictor()
    
    words = ["мама", "тато", "університет"]
    indices = predictor.predict_batch(words)
    
    assert len(indices) == 3
    assert all(isinstance(idx, int) for idx in indices)
    assert all(0 <= idx < 11 for idx in indices)


def test_batch_prediction_with_pos():
    """Test batch prediction with POS tags."""
    predictor = LusciniaPredictor()
    
    words = ["читати", "читання", "читач"]
    pos_tags = ["VERB", "NOUN", "NOUN"]
    
    indices = predictor.predict_batch(words, pos_tags=pos_tags)
    assert len(indices) == 3


def test_predict_proba():
    """Test probability prediction."""
    predictor = LusciniaPredictor()
    
    probs = predictor.predict_proba("університет")
    assert probs.shape == (11,)
    assert np.isclose(probs.sum(), 1.0, atol=0.01)  # Probabilities sum to 1
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_batch_proba():
    """Test batch probability prediction."""
    predictor = LusciniaPredictor()
    
    words = ["мама", "тато"]
    probs = predictor.predict_batch_proba(words)
    
    assert probs.shape == (2, 11)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_featurize():
    """Test feature extraction."""
    features = featurize("мама")
    assert len(features) == 132
    assert all(isinstance(f, float) for f in features)
    
    # Test with POS
    features_with_pos = featurize("мама", pos="NOUN")
    assert len(features_with_pos) == 132


def test_convenience_function():
    """Test convenience function."""
    idx = predict_stress("університет")
    assert isinstance(idx, int)
    assert 0 <= idx < 11


def test_model_info():
    """Test model metadata."""
    predictor = LusciniaPredictor()
    info = predictor.model_info
    
    assert 'num_features' in info
    assert 'num_classes' in info
    assert 'version' in info
    assert info['num_features'] == 132
    assert info['num_classes'] == 11


def test_repr():
    """Test string representation."""
    predictor = LusciniaPredictor()
    repr_str = repr(predictor)
    assert "LusciniaPredictor" in repr_str
    assert "132" in repr_str
    assert "11" in repr_str


def test_empty_word():
    """Test handling of edge cases."""
    predictor = LusciniaPredictor()
    
    # Empty or single-char words should still return valid predictions
    idx = predictor.predict("я")
    assert isinstance(idx, int)


def test_uppercase_handling():
    """Test that uppercase words are handled correctly."""
    predictor = LusciniaPredictor()
    
    idx_lower = predictor.predict("мама")
    idx_upper = predictor.predict("МАМА")
    idx_mixed = predictor.predict("Мама")
    
    # All should give same result (case-insensitive)
    assert idx_lower == idx_upper == idx_mixed


def test_apostrophe_handling():
    """Test words with apostrophes."""
    predictor = LusciniaPredictor()
    
    # Test different apostrophe types
    idx1 = predictor.predict("п'ять")
    idx2 = predictor.predict("п'ять")  # Different apostrophe character
    
    assert isinstance(idx1, int)
    assert isinstance(idx2, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
