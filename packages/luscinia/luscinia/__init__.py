"""Luscinia: Ukrainian stress predictor for out-of-vocabulary words.

Luscinia is a machine learning model (LightGBM/ONNX) that predicts which syllable
carries primary stress in Ukrainian words. It achieves 99.44% accuracy on held-out data.

Quick Start:
    >>> from luscinia import LusciniaPredictor
    >>> predictor = LusciniaPredictor()
    >>> predictor.predict("університет")
    4  # stress on 5th vowel (е)
    
    >>> # Batch prediction
    >>> predictor.predict_batch(["мама", "тато", "університет"])
    [0, 0, 4]
    
    >>> # With POS tags for better accuracy
    >>> predictor.predict("виходити", pos="VERB")
    0

For in-vocabulary words, use ua-stress-engine package first (dictionary lookup).
Luscinia is designed for OOV (out-of-vocabulary) prediction.
"""

__version__ = "1.0.0"
__author__ = "Rostyslav Lukan"
__license__ = "AGPL-3.0-or-later"

from .predictor import LusciniaPredictor, predict_stress
from .featurizer import featurize

__all__ = [
    "LusciniaPredictor",
    "predict_stress",
    "featurize",
    "__version__",
]
