# Luscinia

**Luscinia** is a machine learning model for predicting stress (accent) positions in out-of-vocabulary Ukrainian words. It uses a LightGBM model exported to ONNX format for efficient inference.

## Features

- **99.44% accuracy** on held-out validation data
- **132 linguistic features** including character n-grams, vowel patterns, suffix analysis, and POS tags
- **ONNX Runtime** for fast, cross-platform inference
- **Batch prediction** support for efficient processing
- **Zero external model downloads** — model bundled in the package (~30 MB)
- **Type hints** and comprehensive documentation

## Installation

```bash
pip install luscinia
```

## Quick Start

```python
from luscinia import LusciniaPredictor

# Initialize predictor (loads bundled ONNX model)
predictor = LusciniaPredictor()

# Predict stress position (returns 0-based vowel index)
stress_idx = predictor.predict("університет")
print(stress_idx)  # 4 (5th vowel is stressed: універси<те>т)

# Provide POS tag for better accuracy
stress_idx = predictor.predict("виходити", pos="VERB")
print(stress_idx)  # 0 (<ви>ходити)

# Batch prediction
words = ["мама", "тато", "дитина"]
indices = predictor.predict_batch(words)
print(indices)  # [0, 0, 2]

# Get probability distributions
probs = predictor.predict_proba("університет")
print(probs[4])  # High probability for position 4
```

## Advanced Usage

### Custom Model Path

```python
# Load a custom ONNX model
predictor = LusciniaPredictor(model_path="path/to/custom_model.onnx.gz")
```

### Performance Tuning

```python
import onnxruntime as ort

# Custom session options
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4  # Multi-threading
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

predictor = LusciniaPredictor(session_options=session_options)
```

### Batch Processing with POS Tags

```python
words = ["читати", "читання", "читач"]
pos_tags = ["VERB", "NOUN", "NOUN"]

indices = predictor.predict_batch(words, pos_tags=pos_tags)
```

## Model Information

```python
print(predictor.model_info)
# {
#     'num_features': 132,
#     'num_classes': 11,
#     'version': 'luscinia-lgbm-str-ua-univ-v1.0',
#     'opset': 15,
#     ...
# }
```

## Use Case

Luscinia is designed for **out-of-vocabulary (OOV)** stress prediction. For maximum accuracy:

1. **Check dictionary first** — Use [ua-stress-engine](https://pypi.org/project/ua-stress-engine/) for dictionary lookup (2.7M word forms with full morphology)
2. **Fall back to Luscinia** — For unknown words, use Luscinia to predict stress

```python
# Recommended workflow
from ukrainian_stress import lookup  # from ua-stress-engine
from luscinia import LusciniaPredictor

predictor = LusciniaPredictor()

def get_stress_index(word, pos=None):
    # Try dictionary first
    result = lookup(word)
    if result['readings']:
        return result['readings'][0]['syllable_index']

    # Fall back to ML prediction
    return predictor.predict(word, pos=pos)
```

## Supported POS Tags

The model supports the following part-of-speech tags for improved accuracy:

- `NOUN` — noun
- `VERB` — verb
- `ADJ` — adjective
- `ADV` — adverb
- `NUM` — numeral
- `PRON` — pronoun
- `DET` — determiner
- `PART` — particle
- `CONJ` — conjunction
- `ADP` — adposition
- `INTJ` — interjection
- `X` — other

POS tags are optional but recommended when available.

## Model Details

- **Model**: luscinia-lgbm-str-ua-univ-v1
- **Algorithm**: LightGBM multiclass (11 classes for up to 11-syllable words)
- **Training Data**: 2.7M Ukrainian word forms
- **Features**: 132 (100 base + 32 universal extensions)
- **Accuracy**: 99.44% on validation set
- **Export Format**: ONNX (opset 15)
- **File Size**: ~30 MB (gzip compressed)

## Performance

- **Single prediction**: ~1-2 ms (CPU)
- **Batch prediction (100 words)**: ~10-20 ms (CPU)
- **Model loading**: ~100-200 ms (one-time cost)

## Requirements

- Python 3.8+
- numpy >= 1.24.0
- onnxruntime >= 1.16.0

## License

AGPL-3.0-or-later

## Citation

If you use Luscinia in academic work, please cite:

```
Lukan, R. (2026). Luscinia: A LightGBM-based stress predictor for Ukrainian.
https://github.com/Rostyslav-Lukan/ua-stress-engine
```

## Related Projects

- [ua-stress-engine](https://pypi.org/project/ua-stress-engine/) — Ukrainian stress dictionary with Rust extension
- [ua-word-stress](https://www.npmjs.com/package/ua-word-stress) — JavaScript/TypeScript dictionary (npm)
- [ua-word-stress-wasm](https://www.npmjs.com/package/ua-word-stress-wasm) — WebAssembly version (npm)

## Contributing

Contributions welcome! See [GitHub repository](https://github.com/Rostyslav-Lukan/ua-stress-engine) for details.

## Support

- Issues: https://github.com/Rostyslav-Lukan/ua-stress-engine/issues
- Documentation: https://github.com/Rostyslav-Lukan/ua-stress-engine/tree/main/packages/luscinia
