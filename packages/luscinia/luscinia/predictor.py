"""Luscinia predictor for Ukrainian word stress.

This module provides the main LusciniaPredictor class for predicting stress
positions in out-of-vocabulary Ukrainian words using an ONNX model.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import gzip
import json
import importlib.resources

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "Required dependencies not installed. "
        "Install with: pip install luscinia"
    ) from e

from .featurizer import featurize


class LusciniaPredictor:
    """Ukrainian stress predictor using the Luscinia ONNX model.
    
    The model predicts which syllable (vowel) carries primary stress in a Ukrainian word.
    Returns 0-based vowel index (0 = first vowel, 1 = second vowel, etc.).
    
    Example:
        >>> predictor = LusciniaPredictor()
        >>> predictor.predict("університет")
        4  # stress on 5th vowel (е)
        >>> predictor.predict("виходити", pos="VERB")
        0  # stress on first vowel (и)
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None, session_options: Optional[Any] = None):
        """Initialize the predictor with an ONNX model.
        
        Args:
            model_path: Path to the ONNX model file (gz compressed or uncompressed).
                       If None, uses the bundled model.
            session_options: Optional ONNX Runtime session options for performance tuning.
        """
        if model_path is None:
            # Load bundled model from package data
            model_path = self._get_bundled_model_path()
        else:
            model_path = Path(model_path)
        
        # Load model bytes
        if str(model_path).endswith('.gz'):
            with gzip.open(model_path, 'rb') as f:
                model_bytes = f.read()
        else:
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
        
        # Create ONNX Runtime session
        if session_options is None:
            session_options = ort.SessionOptions()
            # Optimize for inference
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 1  # Single-threaded for latency
        
        self.session = ort.InferenceSession(model_bytes, session_options, providers=['CPUExecutionProvider'])
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape (batch_size, num_features)
        input_shape = self.session.get_inputs()[0].shape
        self.num_features = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        
        # Get output shape (batch_size, num_classes) or (batch_size, num_classes, 1)
        output_shape = self.session.get_outputs()[0].shape
        if len(output_shape) >= 2:
            self.num_classes = output_shape[1]
        else:
            self.num_classes = 11  # Default for luscinia model
        
        # Load manifest if available
        self.manifest = self._load_manifest()
    
    def _get_bundled_model_path(self) -> Path:
        """Get path to the bundled ONNX model."""
        try:
            # Python 3.9+
            if hasattr(importlib.resources, 'files'):
                data_path = importlib.resources.files('luscinia') / 'data' / 'P3_0017_full.onnx.gz'
                return Path(str(data_path))
            else:
                # Python 3.8 fallback
                with importlib.resources.path('luscinia.data', 'P3_0017_full.onnx.gz') as p:
                    return p
        except Exception as e:
            raise RuntimeError(f"Failed to locate bundled model: {e}") from e
    
    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load model manifest if available."""
        try:
            if hasattr(importlib.resources, 'files'):
                manifest_path = importlib.resources.files('luscinia') / 'data' / 'manifest.json'
                with open(str(manifest_path), 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with importlib.resources.path('luscinia.data', 'manifest.json') as p:
                    with open(p, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception:
            return None
    
    def predict(self, word: str, pos: str = "X") -> int:
        """Predict stress position for a single word.
        
        Args:
            word: Ukrainian word to predict stress for
            pos: Part-of-speech tag (UD tagset). Defaults to "X" (unknown)
                Providing specific POS (NOUN, VERB, etc.) improves accuracy.
        
        Returns:
            0-based vowel index indicating which vowel is stressed
            (0 = first vowel, 1 = second vowel, etc.)
        
        Example:
            >>> predictor.predict("університет")
            4  # 5th vowel (е) is stressed
        """
        result = self.predict_batch([word], [pos])
        return result[0]
    
    def predict_batch(
        self,
        words: List[str],
        pos_tags: Optional[List[str]] = None
    ) -> List[int]:
        """Predict stress positions for multiple words efficiently.
        
        Args:
            words: List of Ukrainian words
            pos_tags: Optional list of POS tags (one per word). Defaults to "X" for each word.
        
        Returns:
            List of 0-based vowel indices for each word
        
        Example:
            >>> predictor.predict_batch(["мама", "тато", "університет"])
            [0, 0, 4]
        """
        if not words:
            return []
        
        if pos_tags is None:
            pos_tags = ["X"] * len(words)
        
        if len(pos_tags) != len(words):
            raise ValueError(f"pos_tags length ({len(pos_tags)}) must match words length ({len(words)})")
        
        # Extract features for all words
        batch_size = len(words)
        all_features = [featurize(word, pos) for word, pos in zip(words, pos_tags)]
        input_array = np.array(all_features, dtype=np.float32)
        
        # Run batch inference
        # Only fetch the first output ("label": int64 [batch_size])
        # The model has already performed argmax internally
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        label_data = outputs[0]  # Shape: (batch_size,) int64 array
        
        # Convert int64 to Python int
        return [int(label) for label in label_data]
    
    @property
    def model_version(self) -> str:
        """Get model version from manifest."""
        if self.manifest:
            return self.manifest.get('version', 'unknown')
        return 'unknown'
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        info = {
            'num_features': self.num_features,
            'num_classes': self.num_classes,
            'input_name': self.input_name,
            'output_name': self.output_name,
        }
        if self.manifest:
            info.update({
                'version': self.manifest.get('version'),
                'exported': self.manifest.get('exported'),
                'opset': self.manifest.get('onnx_opset'),
            })
        return info
    
    def __repr__(self) -> str:
        return f"LusciniaPredictor(version={self.model_version}, features={self.num_features}, classes={self.num_classes})"


# Convenience function for simple usage
def predict_stress(word: str, pos: str = "X", predictor: Optional[LusciniaPredictor] = None) -> int:
    """Predict stress position for a Ukrainian word (convenience function).
    
    Args:
        word: Ukrainian word
        pos: POS tag (defaults to "X" for unknown)
        predictor: Optional pre-initialized predictor (for reuse)
    
    Returns:
        0-based vowel index
    
    Example:
        >>> from luscinia import predict_stress
        >>> predict_stress("університет")
        4
    """
    if predictor is None:
        predictor = LusciniaPredictor()
    return predictor.predict(word, pos)
