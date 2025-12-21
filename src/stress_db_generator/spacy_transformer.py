#!/usr/bin/env python3
"""
spaCy Format Transformer

Transforms merged dictionary into spaCy-compatible Universal Dependencies format.
Ensures all morphological features follow UD annotation scheme.

Responsibilities:
- Validate and standardize POS tags (UPOS)
- Validate and standardize morphological features
- Sort features alphabetically
- Generate spaCy Token.morph compatible strings
"""

from typing import Dict, List, Optional
from logging import getLogger
from dataclasses import dataclass, field

logger = getLogger(__name__)


# Universal POS tags (UPOS)
# https://universaldependencies.org/u/pos/
VALID_UPOS_TAGS = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
}

# Common morphological features for Ukrainian
# https://universaldependencies.org/u/feat/index.html
VALID_FEATURES = {
    # Lexical features
    "PronType", "NumType", "Poss", "Reflex", "Foreign", "Abbr", "Typo",
    
    # Inflectional features - nominal
    "Gender", "Animacy", "Number", "Case", "Definite", "Degree",
    
    # Inflectional features - verbal
    "VerbForm", "Mood", "Tense", "Aspect", "Voice", "Evident", "Polarity",
    "Person", "Polite", "Clusivity",
}


@dataclass
class SpaCyWordForm:
    """
    Word form in spaCy-compatible format.
    Ready for LMDB export and NLP pipeline use.
    
    Attributes:
        stress_variants: List of vowel indices where stress occurs
        pos: List of UPOS tags (validated)
        feats: Dict of morphological features (validated, sorted)
        lemma: Base form (optional)
    """
    stress_variants: List[int]
    pos: List[str] = field(default_factory=list)
    feats: Dict[str, List[str]] = field(default_factory=dict)
    lemma: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Export to dictionary format (for LMDB)"""
        result = {
            "stress_variants": self.stress_variants,
        }
        if self.pos:
            result["pos"] = self.pos
        if self.feats:
            # Sort features alphabetically, keep lists
            result["feats"] = {k: sorted(v) for k, v in sorted(self.feats.items())}
        if self.lemma:
            result["lemma"] = self.lemma
        return result
    
    def to_spacy_format(self) -> str:
        """
        Export morphology to spaCy token.morph format.
        
        Returns:
            String like "Case=Nom,Acc|Gender=Fem|Number=Plur"
        """
        if not self.feats:
            return ""
        
        parts = []
        for key in sorted(self.feats.keys()):
            values = sorted(self.feats[key])
            values_str = ",".join(values)
            parts.append(f"{key}={values_str}")
        
        return "|".join(parts)
    
    def matches_morphology(self, required_feats: Dict[str, str]) -> bool:
        """
        Check if this form matches required morphological features.
        
        Args:
            required_feats: Dict like {"Case": "Nom", "Number": "Plur"}
        
        Returns:
            True if all required features match
        """
        for key, value in required_feats.items():
            if key not in self.feats or value not in self.feats[key]:
                return False
        return True


class SpaCyTransformer:
    """
    Transforms dictionary data into spaCy-compatible format.
    
    Transformation Steps:
    1. Validate POS tags against UPOS standards
    2. Validate morphological features against UD standards
    3. Sort features alphabetically
    4. Convert to SpaCyWordForm objects
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize transformer.
        
        Args:
            strict: If True, raise errors on invalid tags/features.
                   If False, log warnings and skip invalid data.
        """
        self.strict = strict
        self.warnings = []
    
    def transform(self, merged_dict: Dict[str, List]) -> Dict[str, List[SpaCyWordForm]]:
        """
        Transform merged dictionary into spaCy format.
        
        Args:
            merged_dict: Dictionary from merger (keys → List[WordForm])
        
        Returns:
            Dictionary with SpaCyWordForm objects (validated and standardized)
        """
        logger.info(f"Transforming {len(merged_dict):,} words to spaCy format")
        
        result: Dict[str, List[SpaCyWordForm]] = {}
        
        for key, forms in merged_dict.items():
            spacy_forms = []
            
            for form in forms:
                # Validate and transform
                validated_pos = self._validate_pos(form.pos, key)
                validated_feats = self._validate_features(form.feats, key)
                
                # Create spaCy form
                spacy_form = SpaCyWordForm(
                    stress_variants=form.stress_variants,
                    pos=validated_pos,
                    feats=validated_feats,
                    lemma=form.lemma
                )
                
                spacy_forms.append(spacy_form)
            
            if spacy_forms:
                result[key] = spacy_forms
        
        logger.info(f"Transformation complete: {len(result):,} words")
        
        if self.warnings:
            logger.warning(f"Total warnings during transformation: {len(self.warnings)}")
            # Show first 10 warnings
            for warning in self.warnings[:10]:
                logger.warning(warning)
            if len(self.warnings) > 10:
                logger.warning(f"... and {len(self.warnings) - 10} more warnings")
        
        return result
    
    def _validate_pos(self, pos_list: List[str], word_key: str) -> List[str]:
        """
        Validate POS tags against UPOS standards.
        
        Args:
            pos_list: List of POS tags
            word_key: Word key (for error messages)
        
        Returns:
            Validated and deduplicated POS list
        """
        if not pos_list:
            return []
        
        validated = []
        
        for pos in pos_list:
            if pos in VALID_UPOS_TAGS:
                validated.append(pos)
            else:
                msg = f"Invalid UPOS tag '{pos}' for word '{word_key}'"
                self.warnings.append(msg)
                
                if self.strict:
                    raise ValueError(msg)
        
        return sorted(list(set(validated)))
    
    def _validate_features(self, feats: Dict[str, List[str]], word_key: str) -> Dict[str, List[str]]:
        """
        Validate morphological features against UD standards.
        
        Args:
            feats: Dict of feature lists
            word_key: Word key (for error messages)
        
        Returns:
            Validated features dict (sorted keys and values)
        """
        if not feats:
            return {}
        
        validated = {}
        
        for feat_key, feat_values in feats.items():
            # Check if feature is valid
            if feat_key not in VALID_FEATURES:
                msg = f"Unknown feature '{feat_key}' for word '{word_key}'"
                self.warnings.append(msg)
                
                if self.strict:
                    raise ValueError(msg)
                else:
                    # In non-strict mode, keep the feature anyway
                    # (might be language-specific extension)
                    pass
            
            # Deduplicate and sort values
            validated[feat_key] = sorted(list(set(feat_values)))
        
        return validated


def transform_to_spacy_format(merged_dict: Dict[str, List], strict: bool = False) -> Dict[str, List[SpaCyWordForm]]:
    """
    Convenience function to transform dictionary to spaCy format.
    
    Args:
        merged_dict: Merged dictionary from DictionaryMerger
        strict: If True, raise errors on invalid data
    
    Returns:
        Dictionary with SpaCyWordForm objects
    """
    transformer = SpaCyTransformer(strict=strict)
    return transformer.transform(merged_dict)
