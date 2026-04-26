#!/usr/bin/env python3
"""
Stress Resolution for Ukrainian Pipeline

Resolves word stress by matching morphological features between
spaCy tokens and stress database entries. Handles heteronyms by
selecting the best-matching stress variant based on context.

Key Features:
- Morphology-based matching (POS + grammatical features)
- Confidence scoring (exact/partial/fallback/none)
- Support for words with multiple stress patterns
- Stress mark insertion in Unicode (combining acute accent U+0301)

Usage:
    from src.nlp.pipeline.stress_resolver import StressResolver
    
    resolver = StressResolver(stress_service)
    stress_info = resolver.resolve(token)
    print(f"{token.text} → {stress_info['stress_pattern']}")
"""

from typing import Optional, Dict, List
from logging import getLogger

from src.nlp.tokenization_service.types import TokenData
from src.nlp.stress_service import UkrainianStressService

logger = getLogger(__name__)


class StressResolver:
    """
    Resolves word stress using morphological matching.
    
    Strategy for heteronym disambiguation:
    1. Lookup all stress variants from database
    2. Calculate match score for each variant (POS + features + semantic context)
    3. Select variant with highest confidence score
    4. Fallback to frequency-based selection if no good match
    
    Scoring system:
    - POS match: 40% weight
    - Morphological features: 40% weight  
    - Semantic context (future): 20% weight
    - Score range: 0.0 (no match) to 1.0 (perfect match)
    
    Confidence levels:
    - exact: score >= 0.8 (high confidence in disambiguation)
    - partial: score >= 0.5 (moderate confidence)
    - fallback: score < 0.5 (using most common variant)
    - none: no stress data available
    """
    
    # Ukrainian vowels for stress mark insertion
    VOWELS = "аеєиіїоуюяАЕЄИІЇОУЮЯ"
    
    def __init__(self, stress_service: UkrainianStressService):
        """
        Initialize resolver with stress service.
        
        Args:
            stress_service: Service for stress database lookups
        """
        self.stress_service = stress_service
        logger.debug("StressResolver initialized")
    
    def resolve(self, token: TokenData) -> Dict:
        """
        Resolve word stress with morphological context.
        
        Args:
            token: Token with POS and morphology from spaCy
        
        Returns:
            Dict with:
                - stress_position: int | None (0-indexed vowel position)
                - stress_pattern: str (word with stress mark, e.g., "замо́к")
                - stress_confidence: str (exact/partial/fallback/none)
                - stress_match_score: float (0.0-1.0)
        """
        # Lookup all stress variants
        stress_result = self.stress_service.lookup(token.text_normalized)
        
        if not stress_result:
            return self._no_stress_result()
        
        # Single variant - use it directly
        if len(stress_result) == 1:
            return self._build_result(
                token=token,
                form=stress_result[0],
                score=1.0,
                confidence='exact'
            )
        
        # Multiple variants - match morphology
        best_form, best_score = self._find_best_match(token, stress_result)
        
        # Determine confidence level
        if best_score >= 0.8:
            confidence = 'exact'
        elif best_score >= 0.5:
            confidence = 'partial'
        else:
            confidence = 'fallback'
            logger.debug(
                f"Low match score for '{token.text}': {best_score:.2f}. "
                f"Token POS={token.pos}, morph={token.morph}"
            )
        
        return self._build_result(token, best_form, best_score, confidence)
    
    def _find_best_match(
        self, 
        token: TokenData, 
        stress_forms: List[Dict]
    ) -> tuple[Dict, float]:
        """
        Find stress form with best morphology match.
        
        Handles heteronyms like:
        - "замок" (за́мок - castle, замо́к - lock)
        - "атлас" (а́тлас - atlas book, атла́с - satin fabric)
        - "блохи" (бло́хи - fleas nominative, блохи́ - flea genitive)
        
        For cases where morphology is identical (like "атлас" in accusative),
        falls back to frequency-based selection (first variant in database).
        
        Args:
            token: spaCy token with morphology
            stress_forms: List of stress database entries
        
        Returns:
            Tuple of (best_form, best_score)
        """
        best_form = stress_forms[0]  # Fallback to first (most frequent)
        best_score = 0.0
        
        for form in stress_forms:
            score = self._calculate_morph_match(token, form)
            
            if score > best_score:
                best_score = score
                best_form = form
                logger.debug(
                    f"Better match for '{token.text}': score={score:.2f}, "
                    f"form_pos={form.get('pos')}, form_feats={form.get('feats')}"
                )
        
        return best_form, best_score
    
    def _calculate_morph_match(self, token: TokenData, stress_form: Dict) -> float:
        """
        Calculate morphological similarity between token and stress form.
        
        Scoring weights:
        - POS match: 40% (0.4 points)
        - Feature matches: 40% (0.4 points)
        - Reserved for semantic context: 20% (future enhancement)
        
        Feature matching:
        - Each matching feature contributes proportionally
        - Case, Number, Gender, Animacy have equal weight
        - Partial match better than no match
        
        Examples:
            Token: NOUN, Case=Nom, Number=Sing
            Form:  ['NOUN'], {'Case': ['Nom'], 'Number': ['Sing', 'Plur']}
            Score: 0.4 (POS) + 0.4 (2/2 features) = 0.8 (exact)
            
            Token: NOUN, Case=Acc, Gender=Masc
            Form:  ['NOUN'], {'Case': ['Acc'], 'Gender': ['Fem']}
            Score: 0.4 (POS) + 0.2 (1/2 features) = 0.6 (partial)
        
        Args:
            token: Token with spaCy morphology
            stress_form: Stress database form
        
        Returns:
            Score 0.0-1.0
        """
        score = 0.0
        
        # Weight distribution
        POS_WEIGHT = 0.4
        FEATURES_WEIGHT = 0.4
        # SEMANTIC_WEIGHT = 0.2  # Reserved for future context analysis
        
        # Check POS match
        token_pos = token.pos.upper()
        stress_pos_list = stress_form.get('pos', [])
        
        if token_pos in stress_pos_list:
            score += POS_WEIGHT
            logger.debug(f"POS match: {token_pos} in {stress_pos_list}")
        else:
            logger.debug(f"POS mismatch: {token_pos} not in {stress_pos_list}")
        
        # Check morphological features
        token_morph = token.morph or {}
        stress_feats = stress_form.get('feats', {})
        
        if token_morph and stress_feats:
            matching_features = 0
            total_features = len(token_morph)
            
            for feat, value in token_morph.items():
                stress_values = stress_feats.get(feat, [])
                if value in stress_values:
                    matching_features += 1
                    logger.debug(f"Feature match: {feat}={value} in {stress_values}")
                else:
                    logger.debug(f"Feature mismatch: {feat}={value} not in {stress_values}")
            
            if total_features > 0:
                feature_ratio = matching_features / total_features
                feature_score = FEATURES_WEIGHT * feature_ratio
                score += feature_score
                logger.debug(
                    f"Feature score: {matching_features}/{total_features} = "
                    f"{feature_ratio:.2f} × {FEATURES_WEIGHT} = {feature_score:.2f}"
                )
        
        # Future: Add semantic context analysis (word collocations, syntax)
        # This would help disambiguate cases like "атлас" where morphology is identical:
        # - "гортає атлас" (flipping through) → book
        # - "атлас тканини" (fabric texture) → satin
        
        return score
    
    def _build_result(
        self,
        token: TokenData,
        form: Dict,
        score: float,
        confidence: str
    ) -> Dict:
        """
        Build stress resolution result dictionary.
        
        Args:
            token: Original token
            form: Selected stress database form
            score: Match score
            confidence: Confidence level
        
        Returns:
            Complete stress info dict
        """
        # Get stress position (use first variant, typically most common)
        stress_variants = form.get('stress_variants', [])
        stress_position = stress_variants[0] if stress_variants else None
        
        # Generate stress pattern with combining accent
        stress_pattern = self._add_stress_mark(
            token.text_normalized,
            stress_position
        ) if stress_position is not None else ''
        
        return {
            'stress_position': stress_position,
            'stress_pattern': stress_pattern,
            'stress_confidence': confidence,
            'stress_match_score': score,
            # Note: selected_variant excluded from return to keep result compatible
            # with EnrichedTokenData model. For debugging, log it instead.
        }
    
    def _no_stress_result(self) -> Dict:
        """Return result for words without stress data."""
        return {
            'stress_position': None,
            'stress_pattern': '',
            'stress_confidence': 'none',
            'stress_match_score': 0.0,
        }
    
    def _add_stress_mark(self, word: str, stress_position: int) -> str:
        """
        Add Unicode combining acute accent (U+0301) after stressed vowel.
        
        The combining accent visually appears above the vowel but is
        inserted after it in the string: "замок" + stress on first vowel
        → "за́мок" (z-a-COMBINING_ACUTE-m-o-k).
        
        Args:
            word: Word text
            stress_position: Index of stressed vowel (0-indexed)
        
        Returns:
            Word with stress mark (e.g., "замо́к")
        
        Examples:
            >>> _add_stress_mark("замок", 0)
            "за́мок"  # Stress on first vowel (а)
            
            >>> _add_stress_mark("замок", 1)
            "замо́к"  # Stress on second vowel (о)
        """
        if stress_position is None:
            return word
        
        # Find all vowel positions
        vowel_positions = [i for i, char in enumerate(word) if char in self.VOWELS]
        
        # Validate stress position
        if stress_position >= len(vowel_positions):
            logger.warning(
                f"Invalid stress position {stress_position} for '{word}' "
                f"(only {len(vowel_positions)} vowels)"
            )
            return word
        
        # Insert combining acute accent after the stressed vowel
        insert_pos = vowel_positions[stress_position] + 1
        return word[:insert_pos] + '\u0301' + word[insert_pos:]
