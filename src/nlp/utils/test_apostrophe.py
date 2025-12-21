"""
Tests for Ukrainian Apostrophe Normalization Utility

Tests the normalize_apostrophe() function and related utilities
for correct handling of Ukrainian apostrophe characters.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from nlp.utils.apostrophe import (
    normalize_apostrophe,
    normalize_word,
    normalize_text,
    has_wrong_apostrophe,
    get_apostrophe_info,
    CORRECT_APOSTROPHE,
    WRONG_APOSTROPHES,
)


class TestApostropheNormalization:
    """Test basic apostrophe normalization"""
    
    def test_normalize_u2019_to_u02bc(self):
        """Test normalization of U+2019 (right single quotation mark)"""
        # U+2019 variant
        word = 'п' + '\u2019' + 'ятниця'
        result = normalize_apostrophe(word)
        
        # Should contain U+02BC
        assert CORRECT_APOSTROPHE in result
        # Should not contain U+2019
        assert '\u2019' not in result
        # Should equal expected result
        assert result == 'п' + '\u02BC' + 'ятниця'
    
    def test_normalize_ascii_apostrophe(self):
        """Test normalization of U+0027 (ASCII apostrophe)"""
        word = 'зв' + "'" + 'язок'  # ASCII apostrophe
        result = normalize_apostrophe(word)
        
        assert CORRECT_APOSTROPHE in result
        assert "'" not in result or result.count("'") == 0
    
    def test_normalize_multiple_apostrophes(self):
        """Test normalization with multiple wrong apostrophes"""
        text = 'п' + '\u2019' + 'ятниця, зв' + '\u2019' + 'язок'
        result = normalize_text(text)
        
        # Count apostrophes
        correct_count = result.count(CORRECT_APOSTROPHE)
        assert correct_count == 2
        
        # Should not contain wrong apostrophes
        assert '\u2019' not in result
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string"""
        result = normalize_apostrophe("")
        assert result == ""
    
    def test_normalize_text_without_apostrophes(self):
        """Test text without apostrophes returns unchanged"""
        text = "Україна та Батьківщина"
        result = normalize_apostrophe(text)
        assert result == text


class TestApostropheDetection:
    """Test apostrophe detection functions"""
    
    def test_has_wrong_apostrophe_true(self):
        """Test detection of wrong apostrophes"""
        word = 'п' + '\u2019' + 'ятниця'
        assert has_wrong_apostrophe(word) is True
    
    def test_has_wrong_apostrophe_false(self):
        """Test that correct apostrophes are not flagged"""
        word = 'п' + CORRECT_APOSTROPHE + 'ятниця'
        assert has_wrong_apostrophe(word) is False
    
    def test_has_wrong_apostrophe_no_apostrophe(self):
        """Test text without apostrophes"""
        text = "Україна"
        assert has_wrong_apostrophe(text) is False
    
    def test_has_wrong_apostrophe_empty_string(self):
        """Test empty string"""
        assert has_wrong_apostrophe("") is False


class TestApostropheAnalysis:
    """Test apostrophe analysis function"""
    
    def test_get_apostrophe_info_with_wrong(self):
        """Test analysis of text with wrong apostrophes"""
        word = 'п' + '\u2019' + 'ятниця'
        info = get_apostrophe_info(word)
        
        assert info['has_wrong'] is True
        assert info['wrong_count'] == 1
        assert info['correct_count'] == 0
        assert 'U+2019' in info['wrong_types']
    
    def test_get_apostrophe_info_with_correct(self):
        """Test analysis of text with correct apostrophes"""
        word = 'п' + CORRECT_APOSTROPHE + 'ятниця'
        info = get_apostrophe_info(word)
        
        assert info['has_wrong'] is False
        assert info['wrong_count'] == 0
        assert info['correct_count'] == 1
    
    def test_get_apostrophe_info_mixed(self):
        """Test analysis of mixed apostrophes"""
        text = ('п' + '\u2019' + 'ятниця ' +  # wrong
                'зв' + CORRECT_APOSTROPHE + 'язок')  # correct
        info = get_apostrophe_info(text)
        
        assert info['has_wrong'] is True
        assert info['wrong_count'] == 1
        assert info['correct_count'] == 1
    
    def test_get_apostrophe_info_no_apostrophes(self):
        """Test analysis of text without apostrophes"""
        text = "Україна та Батьківщина"
        info = get_apostrophe_info(text)
        
        assert info['has_wrong'] is False
        assert info['wrong_count'] == 0
        assert info['correct_count'] == 0


class TestNormalizeWord:
    """Test normalize_word wrapper function"""
    
    def test_normalize_word_single(self):
        """Test normalization of single word"""
        word = 'п' + '\u2019' + 'ятниця'
        result = normalize_word(word)
        
        assert result == 'п' + CORRECT_APOSTROPHE + 'ятниця'
    
    def test_normalize_word_empty(self):
        """Test normalization of empty word"""
        result = normalize_word("")
        assert result == ""


class TestCorrectApostropheUsage:
    """Test that correct apostrophe is used"""
    
    def test_correct_apostrophe_is_u02bc(self):
        """Test that CORRECT_APOSTROPHE is U+02BC"""
        assert CORRECT_APOSTROPHE == '\u02BC'
        assert ord(CORRECT_APOSTROPHE) == 0x02BC
    
    def test_wrong_apostrophes_include_u2019(self):
        """Test that WRONG_APOSTROPHES includes U+2019"""
        assert '\u2019' in WRONG_APOSTROPHES
    
    def test_wrong_apostrophes_count(self):
        """Test that we track multiple wrong apostrophe types"""
        assert len(WRONG_APOSTROPHES) >= 4  # At least common ones


class TestUkrainianWords:
    """Test with real Ukrainian words"""
    
    @pytest.mark.parametrize("wrong_form,expected_form", [
        ('п' + '\u2019' + 'ятниця', 'п' + '\u02BC' + 'ятниця'),
        ('зв' + '\u2019' + 'язок', 'зв' + '\u02BC' + 'язок'),
        ('ав' + '\u2019' + 'ярка', 'ав' + '\u02BC' + 'ярка'),
        ('дв' + '\u2019' + 'ясло', 'дв' + '\u02BC' + 'ясло'),
    ])
    def test_common_ukrainian_words(self, wrong_form, expected_form):
        """Test normalization of common Ukrainian words with apostrophes"""
        result = normalize_apostrophe(wrong_form)
        assert result == expected_form


class TestPreservation:
    """Test that normalization preserves text structure"""
    
    def test_preserves_text_length_approximately(self):
        """Test that normalization doesn't significantly change length"""
        text = 'п' + '\u2019' + 'ятниця, зв' + '\u2019' + 'язок'
        result = normalize_apostrophe(text)
        
        # Length should be same (just character replacement)
        assert len(text) == len(result)
    
    def test_preserves_word_boundaries(self):
        """Test that spaces and punctuation are preserved"""
        text = 'п' + '\u2019' + 'ятниця, зв' + '\u2019' + 'язок!'
        result = normalize_apostrophe(text)
        
        assert ', ' in result
        assert '!' in result
    
    def test_preserves_case(self):
        """Test that letter case is preserved"""
        text = 'П' + '\u2019' + 'ятниця'
        result = normalize_apostrophe(text)
        
        assert result.startswith('П')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
