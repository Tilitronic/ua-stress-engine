"""
Stress Service Package

Provides ultra-fast stress and morphology lookup for Ukrainian words.
Uses LMDB (Lightning Memory-Mapped Database) for optimal performance.

Usage:
    from src.nlp.stress_service import StressService
    
    service = StressService()
    forms = service.lookup("атлас")
    
    for form in forms:
        print(form["stress_variants"])  # [0] or [1]
        print(form["pos"])              # ["NOUN"]
        print(form["feats"])            # {"Case": ["Nom"], ...}
"""

from .types import WordFormDict, WordLookupResult, format_stress_display, format_morphology_spacy
from .stress_service import UkrainianStressService

__all__ = [
    "UkrainianStressService",
    "WordFormDict",
    "WordLookupResult",
    "format_stress_display",
    "format_morphology_spacy",
]
