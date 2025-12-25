#!/usr/bin/env python3
"""
Ukrainian Tokenization Service

Complete tokenization service using spaCy with uk_core_news_lg model.
Provides comprehensive linguistic analysis for Ukrainian text.

Features:
- Sentence segmentation
- Word tokenization
- Lemmatization
- POS tagging (Universal Dependencies)
- Morphological analysis
- Dependency parsing
- Automatic apostrophe normalization

Usage:
    service = UkrainianTokenizationService()
    result = service.tokenize("Привіт! Як справи?")
    
    for sentence in result.sentences:
        for token in sentence.tokens:
            print(f"{token.text}: {token.pos} ({token.lemma})")
    
    service.close()

Or with context manager:
    with UkrainianTokenizationService() as service:
        result = service.tokenize("Привіт!")
"""

import spacy
from pathlib import Path
from typing import Optional
from logging import getLogger

from utils.normalize_apostrophe import normalize_apostrophe
from .types import TokenData, SentenceData, DocumentData

logger = getLogger(__name__)


class UkrainianTokenizationService:
    """
    Ukrainian text tokenization service with full linguistic analysis.
    
    Uses spaCy's uk_core_news_lg model for:
    - Tokenization
    - Lemmatization  
    - POS tagging (Universal Dependencies UPOS)
    - Morphological features
    - Dependency parsing
    - Named entity recognition (if available)
    
    All text is automatically normalized (apostrophes, etc.) before processing.
    """
    
    # Model name (can be changed to uk_core_news_sm for smaller size)
    DEFAULT_MODEL = "uk_core_news_lg"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize tokenization service.
        
        Args:
            model_name: spaCy model to use (default: uk_core_news_lg)
                       Options: uk_core_news_sm, uk_core_news_md, uk_core_news_lg
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading spaCy model: {self.model_name}")
        
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Model loaded successfully: {self.model_name}")
        except OSError:
            logger.error(f"Model '{self.model_name}' not found. Install with:")
            logger.error(f"  python -m spacy download {self.model_name}")
            raise
        
        # Check capabilities
        logger.info(f"Pipeline components: {self.nlp.pipe_names}")
        logger.info(f"Language: {self.nlp.lang}")
        logger.info(f"Vocab size: {len(self.nlp.vocab)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def tokenize(self, text: str, normalize: bool = True) -> DocumentData:
        """
        Tokenize Ukrainian text with full linguistic analysis.
        
        Args:
            text: Text to tokenize
            normalize: Whether to normalize apostrophes (default: True)
        
        Returns:
            DocumentData object with sentences and tokens
        """
        # Normalize text if requested
        original_text = text
        if normalize:
            text = normalize_apostrophe(text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Build result structure
        sentences = []
        
        for sent in doc.sents:
            # Create sentence data
            sentence_data = SentenceData(
                text=sent.text,
                text_normalized=sent.text if normalize else sent.text,
                start_char=sent.start_char,
                end_char=sent.end_char,
            )
            
            # Process tokens
            for token in sent:
                token_data = self._extract_token_data(token, normalize)
                sentence_data.tokens.append(token_data)
            
            sentences.append(sentence_data)
        
        # Create document data
        result = DocumentData(
            text=original_text,
            text_normalized=text if normalize else original_text,
            sentences=sentences,
            lang=doc.lang_,
            has_vectors=doc.vocab.vectors.shape[0] > 0 if hasattr(doc.vocab, 'vectors') else False,
        )
        
        logger.debug(
            f"Tokenized: {result.sentence_count} sentences, "
            f"{result.token_count} tokens"
        )
        
        return result
    
    def _extract_token_data(self, token, normalize: bool) -> TokenData:
        """
        Extract all relevant data from spaCy token.
        
        Args:
            token: spaCy Token object
            normalize: Whether text is normalized
        
        Returns:
            TokenData object with all token information
        """
        # Extract morphological features (UD format)
        morph_dict = {}
        if token.morph:
            # Parse spaCy's MorphAnalysis to dict
            morph_dict = token.morph.to_dict()
        
        # Create TokenData
        token_data = TokenData(
            # Basic properties
            text=token.text,
            text_lower=token.lower_,
            text_normalized=normalize_apostrophe(token.lower_) if normalize else token.lower_,
            lemma=token.lemma_,
            
            # Position
            idx=token.idx,
            i=token.i,
            
            # Linguistic annotations
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            head_idx=token.head.i,
            head_lemma=token.head.lemma_,
            
            # Morphological features
            morph=morph_dict,
            
            # Token properties
            is_alpha=token.is_alpha,
            is_ascii=token.is_ascii,
            is_digit=token.is_digit,
            is_lower=token.is_lower,
            is_upper=token.is_upper,
            is_title=token.is_title,
            is_punct=token.is_punct,
            is_space=token.is_space,
            is_stop=token.is_stop,
            is_oov=token.is_oov,
            
            # Pattern matching
            like_num=token.like_num,
            like_url=token.like_url,
            like_email=token.like_email,
            
            # Shape
            shape=token.shape_,
            
            # Whitespace
            whitespace=token.whitespace_,
            
            # Named Entity Recognition
            ent_type=token.ent_type_,
            ent_iob=token.ent_iob_,
            
            # Sentence boundaries
            is_sent_start=token.is_sent_start,
            is_sent_end=token.is_sent_end if hasattr(token, 'is_sent_end') else False,
            
            # Lexical features
            norm=token.norm_,
            prefix=token.prefix_,
            suffix=token.suffix_,
            
            # Additional classification
            is_bracket=token.is_bracket,
            is_quote=token.is_quote,
            is_currency=token.is_currency,
            is_left_punct=token.is_left_punct,
            is_right_punct=token.is_right_punct,
            
            # Syntactic tree
            n_lefts=token.n_lefts,
            n_rights=token.n_rights,
            
            # Statistical & Vectors
            lang=token.lang_,
            has_vector=token.has_vector,
            vector_norm=float(token.vector_norm) if token.has_vector else 0.0,
            rank=token.rank if hasattr(token, 'rank') else 0,
            prob=float(token.prob) if hasattr(token, 'prob') else 0.0,
            cluster=token.cluster if hasattr(token, 'cluster') else 0,
            sentiment=float(token.sentiment) if hasattr(token, 'sentiment') else 0.0,
            
            # Extended NER
            ent_id=token.ent_id_,
            ent_kb_id=token.ent_kb_id_,
        )
        
        return token_data
    
    def tokenize_batch(self, texts: list[str], normalize: bool = True, 
                       batch_size: int = 50) -> list[DocumentData]:
        """
        Tokenize multiple texts efficiently in batches.
        
        Args:
            texts: List of texts to tokenize
            normalize: Whether to normalize apostrophes
            batch_size: Number of texts to process at once
        
        Returns:
            List of DocumentData objects
        """
        # Normalize if requested
        if normalize:
            texts_to_process = [normalize_apostrophe(text) for text in texts]
        else:
            texts_to_process = texts
        
        results = []
        
        # Process in batches using spaCy's pipe
        for i, doc in enumerate(self.nlp.pipe(texts_to_process, batch_size=batch_size)):
            # Build DocumentData same as tokenize()
            sentences = []
            
            for sent in doc.sents:
                sentence_data = SentenceData(
                    text=sent.text,
                    text_normalized=sent.text if normalize else sent.text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
                
                for token in sent:
                    token_data = self._extract_token_data(token, normalize)
                    sentence_data.tokens.append(token_data)
                
                sentences.append(sentence_data)
            
            result = DocumentData(
                text=texts[i],
                text_normalized=texts_to_process[i] if normalize else texts[i],
                sentences=sentences,
                lang=doc.lang_,
                has_vectors=doc.vocab.vectors.shape[0] > 0 if hasattr(doc.vocab, 'vectors') else False,
            )
            
            results.append(result)
        
        logger.info(f"Batch tokenized {len(texts)} documents")
        return results
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded spaCy model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "name": self.model_name,
            "lang": self.nlp.lang,
            "pipeline": self.nlp.pipe_names,
            "has_vectors": self.nlp.vocab.vectors.shape[0] > 0 if hasattr(self.nlp.vocab, 'vectors') else False,
            "vocab_size": len(self.nlp.vocab),
        }
    
    def close(self):
        """Clean up resources."""
        logger.info("Tokenization service closed")
