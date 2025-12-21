#!/usr/bin/env python3
"""
Tokenization Service Package

Provides comprehensive tokenization for Ukrainian text using spaCy.

Usage:
    from src.nlp.tokenization_service import UkrainianTokenizationService
    
    service = UkrainianTokenizationService()
    result = service.tokenize("Привіт!")
    
    for sentence in result.sentences:
        for token in sentence.tokens:
            print(f"{token.text}: {token.pos}")
"""

from .tokenization_service import UkrainianTokenizationService
from .types import TokenData, SentenceData, DocumentData, TokenList, SentenceList

__all__ = [
    "UkrainianTokenizationService",
    "TokenData",
    "SentenceData",
    "DocumentData",
    "TokenList",
    "SentenceList",
]
