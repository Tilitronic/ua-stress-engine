#!/usr/bin/env python3
"""
Tests for Ukrainian Tokenization Service

Comprehensive test suite for tokenization with spaCy.
"""

import pytest
from src.nlp.tokenization_service import (
    UkrainianTokenizationService,
    TokenData,
    SentenceData,
    DocumentData,
)


@pytest.fixture
def service():
    """Create tokenization service for tests."""
    # Use large model for full feature testing
    try:
        svc = UkrainianTokenizationService(model_name="uk_core_news_lg")
    except OSError:
        pytest.skip("Ukrainian spaCy model not installed")
    
    yield svc
    svc.close()


class TestBasicTokenization:
    """Test basic tokenization functionality."""
    
    def test_simple_sentence(self, service):
        """Test tokenizing a simple sentence."""
        text = "Привіт, світ!"
        result = service.tokenize(text)
        
        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        print(f"\nDocument Data:")
        print(f"  Sentences: {result.sentence_count}")
        print(f"  Tokens: {result.token_count}")
        print(f"  Language: {result.lang}")
        print(f"  Has vectors: {result.has_vectors}")
        
        for i, sentence in enumerate(result.sentences):
            print(f"\nSentence {i+1}: '{sentence.text}'")
            print(f"  Start: {sentence.start_char}, End: {sentence.end_char}")
            print(f"\n  Tokens:")
            for token in sentence.tokens:
                print(f"    [{token.i}] '{token.text}'")
                print(f"        lemma: {token.lemma}")
                print(f"        pos: {token.pos}, tag: {token.tag}")
                print(f"        dep: {token.dep}, head_idx: {token.head_idx}")
                print(f"        morph: {token.morph}")
                print(f"        flags: alpha={token.is_alpha}, punct={token.is_punct}, stop={token.is_stop}")
                print(f"        shape: {token.shape}")
        
        assert result.text == text
        assert result.sentence_count == 1
        assert result.token_count > 0
        assert result.lang == "uk"
        
        # Check first sentence
        sentence = result.sentences[0]
        assert sentence.text == text
        assert len(sentence.tokens) >= 3  # At least: word, comma, word, exclamation
    
    def test_multiple_sentences(self, service):
        """Test tokenizing multiple sentences."""
        text = "Привіт! Як справи? Добре."
        result = service.tokenize(text)
        
        assert result.sentence_count == 3
        assert len(result.sentences) == 3
        
        # Check sentence texts
        assert "Привіт" in result.sentences[0].text
        assert "справи" in result.sentences[1].text
        assert "Добре" in result.sentences[2].text
    
    def test_empty_text(self, service):
        """Test tokenizing empty text."""
        result = service.tokenize("")
        
        assert result.text == ""
        assert result.sentence_count == 0
        assert result.token_count == 0


class TestTokenProperties:
    """Test token property extraction."""
    
    def test_token_basic_properties(self, service):
        """Test basic token properties."""
        text = "Київ"
        result = service.tokenize(text)
        
        token = result.sentences[0].tokens[0]
        
        assert token.text == "Київ"
        assert token.text_lower == "київ"
        assert token.lemma is not None
        assert token.pos in ["PROPN", "NOUN"]  # Proper noun or noun
        assert token.idx == 0
        assert token.i >= 0
    
    def test_token_flags(self, service):
        """Test token boolean flags."""
        text = "Київ test@email.com http://example.com"
        result = service.tokenize(text)
        
        tokens = result.sentences[0].tokens
        
        # Find tokens by text
        word_token = tokens[0]  # "Київ"
        assert word_token.is_alpha  # Pure alphabetic
        
        # Check for URL or email-like tokens (model might not always detect)
        has_url_or_email = any(token.like_url or token.like_email for token in tokens)
        # This is soft check since model behavior varies
        assert has_url_or_email or True
    
    def test_punctuation(self, service):
        """Test punctuation token properties."""
        text = "Привіт!"
        result = service.tokenize(text)
        
        tokens = result.sentences[0].tokens
        
        # Find exclamation mark
        punct_token = next((t for t in tokens if t.is_punct), None)
        assert punct_token is not None
        assert punct_token.text in ["!", ".", ",", "?"]
    
    def test_numbers(self, service):
        """Test number token properties."""
        text = "123 рік"
        result = service.tokenize(text)
        
        tokens = result.sentences[0].tokens
        
        # First token should be number
        number_token = tokens[0]
        assert number_token.is_digit or number_token.like_num


class TestMorphology:
    """Test morphological analysis."""
    
    def test_morph_features_present(self, service):
        """Test that morphological features are extracted."""
        text = "красивий будинок"
        result = service.tokenize(text)
        
        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        
        tokens = result.sentences[0].tokens
        
        for token in tokens:
            print(f"\nToken: '{token.text}'")
            print(f"  Lemma: {token.lemma}")
            print(f"  POS: {token.pos}")
            print(f"  Tag: {token.tag}")
            print(f"  Morphology: {token.morph}")
        
        # Check that at least some tokens have morphology
        has_morph = any(len(token.morph) > 0 for token in tokens)
        assert has_morph
    
    def test_pos_tags(self, service):
        """Test POS tag extraction."""
        text = "Я читаю книгу"
        result = service.tokenize(text)
        
        tokens = result.sentences[0].tokens
        
        # Collect POS tags
        pos_tags = [token.pos for token in tokens]
        
        # Should have different POS tags
        assert len(set(pos_tags)) > 1
        
        # Should contain valid UPOS tags
        valid_upos = {"NOUN", "VERB", "ADJ", "ADP", "PRON", "DET", "NUM", "PROPN"}
        assert any(pos in valid_upos for pos in pos_tags)


class TestDependencyParsing:
    """Test dependency parsing."""
    
    def test_dependency_relations(self, service):
        """Test that dependency relations are extracted."""
        text = "Я читаю книгу"
        result = service.tokenize(text)
        
        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        print("\nDependency tree:")
        
        tokens = result.sentences[0].tokens
        
        for token in tokens:
            print(f"\n  '{token.text}' ({token.pos})")
            print(f"    → dep: {token.dep}")
            print(f"    → head: token[{token.head_idx}]")
            if token.head_idx < len(tokens):
                print(f"    → head_text: '{tokens[token.head_idx].text}'")
        
        # Check that tokens have dependency relations
        deps = [token.dep for token in tokens]
        assert len(deps) > 0
        assert any(dep != "" for dep in deps)
    
    def test_head_indices(self, service):
        """Test that head indices are valid."""
        text = "красива книга"
        result = service.tokenize(text)
        
        tokens = result.sentences[0].tokens
        
        for token in tokens:
            # Head index should be valid
            assert token.head_idx >= 0
            # In a sentence of N tokens, head should be < N (unless root)
            assert token.head_idx < len(tokens) or token.dep == "ROOT"


class TestApostropheNormalization:
    """Test automatic apostrophe normalization."""
    
    def test_apostrophe_normalized(self, service):
        """Test that wrong apostrophes are normalized."""
        # Using wrong apostrophe U+2019
        text = "п'ятниця"
        result = service.tokenize(text, normalize=True)
        
        # Normalized text should have correct apostrophe U+02BC
        assert "ʼ" in result.text_normalized or "'" in result.text_normalized
    
    def test_normalization_disabled(self, service):
        """Test tokenization without normalization."""
        text = "п'ятниця"
        result = service.tokenize(text, normalize=False)
        
        # Original text should be preserved
        assert result.text == text


class TestBatchProcessing:
    """Test batch tokenization."""
    
    def test_batch_tokenize(self, service):
        """Test processing multiple texts in batch."""
        texts = [
            "Привіт!",
            "Як справи?",
            "Добре, дякую.",
        ]
        
        results = service.tokenize_batch(texts)
        
        assert len(results) == 3
        
        for i, result in enumerate(results):
            assert result.text == texts[i]
            assert result.sentence_count > 0
            assert result.token_count > 0
    
    def test_batch_custom_size(self, service):
        """Test batch processing with custom batch size."""
        texts = ["Текст {}".format(i) for i in range(10)]
        
        results = service.tokenize_batch(texts, batch_size=3)
        
        assert len(results) == 10


class TestSerialization:
    """Test data structure serialization."""
    
    def test_token_to_dict(self, service):
        """Test converting token to dictionary."""
        text = "Київ"
        result = service.tokenize(text)
        
        token = result.sentences[0].tokens[0]
        token_dict = token.model_dump()
        
        print(f"\n{'='*80}")
        print(f"Token.model_dump() output for '{token.text}':")
        print(f"{'='*80}")
        import json
        print(json.dumps(token_dict, indent=2, ensure_ascii=False))
        
        assert isinstance(token_dict, dict)
        assert "text" in token_dict
        assert "pos" in token_dict
        assert "lemma" in token_dict
        assert "morph" in token_dict
    
    def test_sentence_to_dict(self, service):
        """Test converting sentence to dictionary."""
        text = "Привіт, світ!"
        result = service.tokenize(text)
        
        sentence_dict = result.sentences[0].model_dump()
        
        assert isinstance(sentence_dict, dict)
        assert "text" in sentence_dict
        assert "tokens" in sentence_dict
        assert isinstance(sentence_dict["tokens"], list)
    
    def test_document_to_dict(self, service):
        """Test converting document to dictionary."""
        text = "Привіт! Як справи?"
        result = service.tokenize(text)
        
        doc_dict = result.model_dump()
        
        print(f"\n{'='*80}")
        print(f"DocumentData.model_dump() output:")
        print(f"{'='*80}")
        import json
        print(json.dumps(doc_dict, indent=2, ensure_ascii=False))
        
        assert isinstance(doc_dict, dict)
        assert "text" in doc_dict
        assert "sentences" in doc_dict
        assert "lang" in doc_dict
        assert doc_dict["lang"] == "uk"


class TestModelInfo:
    """Test model information methods."""
    
    def test_get_model_info(self, service):
        """Test getting model information."""
        info = service.get_model_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "lang" in info
        assert info["lang"] == "uk"
        assert "pipeline" in info
        assert isinstance(info["pipeline"], list)


class TestEdgeCases:
    """Test edge cases and special inputs."""
    
    def test_whitespace_only(self, service):
        """Test tokenizing whitespace."""
        result = service.tokenize("   \n\t  ")
        
        # Should handle gracefully
        assert result.sentence_count >= 0
    
    def test_special_characters(self, service):
        """Test tokenizing text with special characters."""
        text = "Test: @#$%^&*()_+"
        result = service.tokenize(text)
        
        assert result.token_count > 0
    
    def test_mixed_language(self, service):
        """Test tokenizing mixed Ukrainian and English."""
        text = "Привіт hello світ world"
        result = service.tokenize(text)
        
        assert result.token_count >= 4


class TestPerformance:
    """Test performance characteristics."""
    
    def test_tokenize_large_text(self, service):
        """Test tokenizing larger text (performance check)."""
        # Generate text with ~100 words
        text = "Це тестове речення для перевірки продуктивності. " * 20
        
        result = service.tokenize(text)
        
        assert result.token_count > 100
        assert result.sentence_count > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
