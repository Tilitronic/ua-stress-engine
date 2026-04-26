"""
Ukrainian NLP Pipeline Package

Complete text processing pipeline:
- Tokenization (spaCy)
- Stress Resolution (LMDB lookup + morphology matching)
- Phonetic Transcription (IPA)
"""

from .pipeline import (
    UkrainianPipeline,
    EnrichedTokenData,
    EnrichedSentenceData,
    EnrichedDocumentData,
    process_text,
)
from .stress_resolver import StressResolver
from .ml_stress_resolver import MLStressResolver
from .resolver_factory import (
    create_ml_resolver,
    create_pipeline_kwargs,
    is_lightgbm_available,
    is_model_available,
    DEFAULT_MODEL_PATH,
)

__all__ = [
    'UkrainianPipeline',
    'EnrichedTokenData',
    'EnrichedSentenceData',
    'EnrichedDocumentData',
    'process_text',
    'StressResolver',
    'MLStressResolver',
    'create_ml_resolver',
    'create_pipeline_kwargs',
    'is_lightgbm_available',
    'is_model_available',
    'DEFAULT_MODEL_PATH',
]
