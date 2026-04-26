#!/usr/bin/env python3
"""
Ukrainian Text Processing Pipeline

Comprehensive NLP pipeline for Ukrainian text:
1. Tokenization (spaCy) - linguistic analysis
2. Stress Resolution - word stress using morphology matching
3. Phonetic Transcription - IPA transcription (draft)

Usage:
    from src.nlp.pipeline import UkrainianPipeline
    
    pipeline = UkrainianPipeline()
    result = pipeline.process("Привіт, світ!")
    
    for sentence in result.sentences:
        for token in sentence.tokens:
            print(f"{token.text} → {token.stress_pattern} → {token.phonetic}")
"""

from typing import Optional, List
from pathlib import Path
from logging import getLogger

from src.nlp.tokenization_service import (
    UkrainianTokenizationService,
    TokenData,
    SentenceData,
    DocumentData,
)
from src.nlp.stress_service import UkrainianStressService
from src.nlp.phonetic import UkrainianPhoneticTranscriber
from src.nlp.pipeline.stress_resolver import StressResolver
from src.nlp.pipeline.ml_stress_resolver import MLStressResolver
from pydantic import BaseModel, Field

logger = getLogger(__name__)


class EnrichedTokenData(TokenData):
    """
    Token with additional pipeline data: stress and phonetic transcription.
    
    Extends TokenData with:
    - stress_position: Vowel index with stress (0-indexed)
    - stress_pattern: Visual representation (e.g., "замо́к")
    - stress_confidence: Match quality (exact/partial/none)
    - phonetic: IPA transcription
    - phonetic_syllables: Syllabified IPA
    """
    
    # Stress information
    stress_position: Optional[int] = Field(
        default=None,
        description="Index of stressed vowel (0-indexed), None if no stress or not a word"
    )
    
    stress_pattern: str = Field(
        default="",
        description="Text with stress mark (e.g., 'замо́к'), empty if no stress"
    )
    
    stress_confidence: str = Field(
        default="none",
        description="Stress resolution confidence: exact/partial/fallback/none"
    )
    
    stress_match_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Morphology match score (0.0-1.0), 1.0 = perfect match"
    )
    
    # Phonetic information
    phonetic: str = Field(
        default="",
        description="IPA phonetic transcription"
    )
    
    phonetic_syllables: List[str] = Field(
        default_factory=list,
        description="IPA transcription split into syllables"
    )


class EnrichedSentenceData(BaseModel):
    """Sentence with enriched tokens."""
    
    text: str
    text_normalized: str
    start_char: int
    end_char: int
    tokens: List[EnrichedTokenData] = Field(default_factory=list)


class EnrichedDocumentData(BaseModel):
    """Document processing result with full pipeline data."""
    
    text: str
    text_normalized: str
    sentences: List[EnrichedSentenceData] = Field(default_factory=list)
    lang: str = "uk"
    has_vectors: bool = False
    
    # Pipeline statistics
    total_tokens: int = 0
    words_processed: int = 0
    words_with_stress: int = 0
    words_with_phonetic: int = 0
    
    @property
    def stress_coverage(self) -> float:
        """Percentage of words with stress information."""
        if self.words_processed == 0:
            return 0.0
        return (self.words_with_stress / self.words_processed) * 100
    
    @property
    def phonetic_coverage(self) -> float:
        """Percentage of words with phonetic transcription."""
        if self.words_processed == 0:
            return 0.0
        return (self.words_with_phonetic / self.words_processed) * 100


class UkrainianPipeline:
    """
    Complete Ukrainian text processing pipeline.
    
    Stages:
    1. Tokenization: spaCy analysis (POS, morphology, dependencies)
    2. Stress Resolution: Match morphology to find correct stress
    3. Phonetic Transcription: Convert to IPA with stress
    
    Example:
        pipeline = UkrainianPipeline()
        result = pipeline.process("Замок на замку")
        
        # Access enriched tokens
        for sent in result.sentences:
            for token in sent.tokens:
                if token.is_alpha:
                    print(f"{token.text} [{token.phonetic}] stress={token.stress_position}")
    """
    
    def __init__(
        self,
        tokenizer_model: Optional[str] = None,
        stress_db_path: Optional[Path] = None,
        stress_mode: str = "db",
        ml_model_path: Optional[Path] = None,
        ml_confidence_threshold: float = 0.70,
        ml_resolver: Optional["MLStressResolver"] = None,
    ):
        """
        Initialize pipeline with all services.
        
        Args:
            tokenizer_model: spaCy model name (default: uk_core_news_lg)
            stress_db_path: Path to stress LMDB database
            stress_mode: Stress engine mode: "db", "ml", or "hybrid"
            ml_model_path: Path to trained LightGBM model (required for "ml")
            ml_confidence_threshold: Hybrid cutoff for accepting ML prediction
            ml_resolver: Pre-built MLStressResolver (from resolver_factory);
                when provided, skips internal construction and uses it directly.
        """
        logger.info("Initializing Ukrainian NLP Pipeline...")

        mode = (stress_mode or "db").lower()
        if mode not in {"db", "ml", "hybrid"}:
            raise ValueError("stress_mode must be one of: db, ml, hybrid")
        self.stress_mode = mode
        self.ml_confidence_threshold = ml_confidence_threshold
        
        # Initialize services
        self.tokenizer = UkrainianTokenizationService(model_name=tokenizer_model)
        self.stress_service = UkrainianStressService(db_path=stress_db_path)
        self.stress_resolver = StressResolver(self.stress_service)
        self.ml_stress_resolver = None
        if ml_resolver is not None:
            # Pre-built resolver from factory — use as-is (feature builder already wired)
            if mode not in {"ml", "hybrid"}:
                logger.warning(
                    "ml_resolver provided but stress_mode=%r — resolver ignored", mode
                )
            else:
                self.ml_stress_resolver = ml_resolver
        elif mode in {"ml", "hybrid"}:
            if ml_model_path is None:
                if mode == "ml":
                    raise ValueError("ml_model_path is required when stress_mode='ml'")
                logger.warning(
                    "hybrid mode requested without ml_model_path; falling back to db mode"
                )
                self.stress_mode = "db"
            else:
                self.ml_stress_resolver = MLStressResolver(model_path=ml_model_path)
        self.phonetic = UkrainianPhoneticTranscriber()
        
        logger.info(
            "Pipeline ready: tokenization → stress(%s) → phonetic",
            self.stress_mode,
        )
    
    def process(self, text: str, normalize: bool = True) -> EnrichedDocumentData:
        """
        Process text through the complete pipeline.
        
        Args:
            text: Ukrainian text to process
            normalize: Whether to normalize apostrophes
        
        Returns:
            EnrichedDocumentData with tokenization, stress, and phonetic data
        """
        logger.debug(f"Processing text: {text[:50]}...")
        
        # Stage 1: Tokenization
        doc_data = self.tokenizer.tokenize(text, normalize=normalize)
        
        # Stage 2 & 3: Stress resolution + phonetic transcription
        enriched_sentences = []
        stats = {
            'total_tokens': 0,
            'words_processed': 0,
            'words_with_stress': 0,
            'words_with_phonetic': 0,
        }
        
        for sentence in doc_data.sentences:
            enriched_tokens = []
            
            for token in sentence.tokens:
                stats['total_tokens'] += 1
                
                # Convert to enriched token
                enriched = self._enrich_token(token)
                enriched_tokens.append(enriched)
                
                # Update statistics
                if token.is_alpha and not token.is_punct:
                    stats['words_processed'] += 1
                    if enriched.stress_position is not None:
                        stats['words_with_stress'] += 1
                    if enriched.phonetic:
                        stats['words_with_phonetic'] += 1
            
            enriched_sentence = EnrichedSentenceData(
                text=sentence.text,
                text_normalized=sentence.text_normalized,
                start_char=sentence.start_char,
                end_char=sentence.end_char,
                tokens=enriched_tokens,
            )
            enriched_sentences.append(enriched_sentence)
        
        result = EnrichedDocumentData(
            text=doc_data.text,
            text_normalized=doc_data.text_normalized,
            sentences=enriched_sentences,
            lang=doc_data.lang,
            has_vectors=doc_data.has_vectors,
            **stats,
        )
        
        logger.debug(
            f"Pipeline complete: {result.total_tokens} tokens, "
            f"{result.stress_coverage:.1f}% stress coverage, "
            f"{result.phonetic_coverage:.1f}% phonetic coverage"
        )
        
        return result
    
    def _enrich_token(self, token: TokenData) -> EnrichedTokenData:
        """
        Enrich token with stress and phonetic information.
        
        Process:
        1. Skip non-words (punctuation, spaces, etc.)
        2. Resolve stress using morphology matching
        3. Generate phonetic transcription using stress
        
        Args:
            token: Base token from tokenization
        
        Returns:
            Enriched token with stress and phonetic data
        """
        # Start with base token data
        enriched_dict = token.model_dump()
        
        # Skip non-words
        if not token.is_alpha or token.is_punct or token.is_space:
            return EnrichedTokenData(**enriched_dict)
        
        # Stage 2: Resolve stress
        stress_info = self._resolve_stress(token)
        enriched_dict.update(stress_info)
        
        # Stage 3: Phonetic transcription
        if stress_info['stress_position'] is not None:
            phonetic_info = self._transcribe_phonetic(
                token.text_normalized,
                stress_info['stress_position']
            )
            enriched_dict.update(phonetic_info)
        
        return EnrichedTokenData(**enriched_dict)

    def _resolve_stress(self, token: TokenData) -> dict:
        """Resolve stress according to configured engine mode."""
        if self.stress_mode == "db" or self.ml_stress_resolver is None:
            return self.stress_resolver.resolve(token)

        if self.stress_mode == "ml":
            return self.ml_stress_resolver.resolve(token)

        ml_info = self.ml_stress_resolver.resolve(token)
        ml_score = ml_info.get("stress_match_score", 0.0)
        ml_has_stress = ml_info.get("stress_position") is not None

        if ml_has_stress and ml_score >= self.ml_confidence_threshold:
            return ml_info

        db_info = self.stress_resolver.resolve(token)
        if db_info.get("stress_position") is not None:
            return db_info

        return ml_info
    
    def _transcribe_phonetic(self, word: str, stress_position: int) -> dict:
        """
        Generate IPA phonetic transcription with stress.
        
        Args:
            word: Normalized word text
            stress_position: Index of stressed vowel
        
        Returns:
            Dict with phonetic and phonetic_syllables
        """
        # Use phonetic transcriber (draft implementation)
        ipa = self.phonetic.transcribe(word, stress_position)
        syllables = self.phonetic.syllabify(ipa)
        
        return {
            'phonetic': ipa,
            'phonetic_syllables': syllables,
        }
    
    def close(self):
        """Close all services."""
        self.tokenizer.close()
        self.stress_service.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def process_text(text: str) -> EnrichedDocumentData:
    """
    Quick processing of Ukrainian text.
    
    Args:
        text: Ukrainian text
    
    Returns:
        Enriched document with full pipeline data
    """
    with UkrainianPipeline() as pipeline:
        return pipeline.process(text)
