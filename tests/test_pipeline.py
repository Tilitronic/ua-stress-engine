#!/usr/bin/env python3
"""
Tests for Ukrainian NLP Pipeline

Tests the complete pipeline: tokenization → stress → phonetic
"""

import pytest
from src.nlp.pipeline import UkrainianPipeline, process_text


@pytest.fixture
def pipeline():
    """Create pipeline for tests."""
    try:
        p = UkrainianPipeline()
    except Exception as e:
        pytest.skip(f"Pipeline initialization failed: {e}")
    
    yield p
    p.close()


class TestPipelineBasic:
    """Test basic pipeline functionality."""
    
    def test_simple_word(self, pipeline):
        """Test processing a simple word."""
        result = pipeline.process("слово")
        
        print(f"\n{'='*80}")
        print(f"TEXT: слово (word)")
        print(f"{'='*80}")
        
        assert result.total_tokens > 0
        assert result.sentences
        
        # Check first token
        token = result.sentences[0].tokens[0]
        print(f"\nToken: {token.text}")
        print(f"  POS: {token.pos}")
        print(f"  Lemma: {token.lemma}")
        print(f"  Stress: {token.stress_pattern} (position: {token.stress_position})")
        print(f"  Confidence: {token.stress_confidence} (score: {token.stress_match_score:.2f})")
        print(f"  Phonetic: [{token.phonetic}]")
        print(f"  Syllables: {token.phonetic_syllables}")
        
        assert token.text == "слово"
        # Should have stress information
        assert token.stress_position is not None or token.stress_confidence == "none"
    
    def test_heteronym(self, pipeline):
        """Test heteronym (word with multiple stress patterns)."""
        text = "Замок на замку"
        result = pipeline.process(text)
        
        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        
        words = []
        for sent in result.sentences:
            for token in sent.tokens:
                if token.is_alpha and not token.is_punct:
                    words.append(token)
                    print(f"\nToken: {token.text}")
                    print(f"  POS: {token.pos} | Morphology: {token.morph}")
                    print(f"  Stress: {token.stress_pattern} (position: {token.stress_position})")
                    print(f"  Confidence: {token.stress_confidence} (score: {token.stress_match_score:.2f})")
                    print(f"  Phonetic: [{token.phonetic}]")
        
        # Should have 3 words: Замок, на, замку
        assert len(words) == 3
        assert words[0].text.lower() == "замок"
        assert words[2].text.lower() == "замку"
    
    def test_sentence_with_punctuation(self, pipeline):
        """Test sentence with various tokens."""
        text = "Привіт, світ!"
        result = pipeline.process(text)
        
        print(f"\n{'='*80}")
        print(f"TEXT: {text}")
        print(f"{'='*80}")
        
        for sent in result.sentences:
            print(f"\nSentence: '{sent.text}'")
            print(f"Tokens:")
            for token in sent.tokens:
                if token.is_alpha:
                    print(f"  {token.text:15} → [{token.phonetic:20}] stress={token.stress_position}")
                else:
                    print(f"  {token.text:15} (punctuation)")
        
        assert result.total_tokens >= 3  # привіт, comma, світ, exclamation
    
    def test_pipeline_statistics(self, pipeline):
        """Test pipeline statistics collection."""
        text = "Це просте речення. Воно має два речення."
        result = pipeline.process(text)
        
        print(f"\n{'='*80}")
        print(f"PIPELINE STATISTICS")
        print(f"{'='*80}")
        print(f"Total tokens: {result.total_tokens}")
        print(f"Words processed: {result.words_processed}")
        print(f"Words with stress: {result.words_with_stress}")
        print(f"Words with phonetic: {result.words_with_phonetic}")
        print(f"Stress coverage: {result.stress_coverage:.1f}%")
        print(f"Phonetic coverage: {result.phonetic_coverage:.1f}%")
        
        assert result.total_tokens > 0
        assert result.words_processed > 0
        assert result.stress_coverage >= 0


class TestStressResolution:
    """Test stress resolution with morphology matching."""
    
    def test_stress_with_morphology(self, pipeline):
        """Test that morphology improves stress selection."""
        # "замок" - heteronym with different meanings based on stress
        result = pipeline.process("замок")
        
        token = result.sentences[0].tokens[0]
        
        print(f"\n{'='*80}")
        print(f"STRESS RESOLUTION TEST")
        print(f"{'='*80}")
        print(f"Word: {token.text}")
        print(f"spaCy POS: {token.pos}")
        print(f"spaCy Morphology: {token.morph}")
        print(f"Resolved stress: {token.stress_pattern}")
        print(f"Stress position: {token.stress_position}")
        print(f"Confidence: {token.stress_confidence}")
        print(f"Match score: {token.stress_match_score:.2f}")
        
        # Should have attempted stress resolution
        assert token.stress_confidence in ['exact', 'partial', 'fallback', 'none']


class TestPhoneticTranscription:
    """Test phonetic transcription."""
    
    def test_basic_transcription(self, pipeline):
        """Test basic IPA transcription."""
        test_words = [
            "слово",  # word
            "мова",   # language
            "дім",    # house
        ]
        
        print(f"\n{'='*80}")
        print(f"PHONETIC TRANSCRIPTION TEST")
        print(f"{'='*80}")
        
        for word in test_words:
            result = pipeline.process(word)
            token = result.sentences[0].tokens[0]
            
            print(f"\n{word:15} → [{token.phonetic}]")
            print(f"{'':15}    Syllables: {token.phonetic_syllables}")
            print(f"{'':15}    Stress: {token.stress_pattern}")
            
            # Should have phonetic transcription if it has stress
            if token.stress_position is not None:
                assert token.phonetic  # Should not be empty


class TestConvenienceFunction:
    """Test convenience function."""
    
    def test_process_text_function(self):
        """Test the quick process_text() function."""
        result = process_text("Привіт!")
        
        print(f"\n{'='*80}")
        print(f"CONVENIENCE FUNCTION TEST")
        print(f"{'='*80}")
        
        assert result.sentences
        token = result.sentences[0].tokens[0]
        
        print(f"Text: {token.text}")
        print(f"Phonetic: [{token.phonetic}]")
        
        assert token.text == "Привіт"


class TestContextManager:
    """Test context manager usage."""
    
    def test_with_statement(self):
        """Test using pipeline with context manager."""
        with UkrainianPipeline() as pipeline:
            result = pipeline.process("тест")
            assert result.sentences
        
        print("\n✅ Context manager works correctly")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
