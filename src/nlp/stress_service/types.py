"""
Type definitions for stress service.
Defines the data structure returned from LMDB database queries.
"""

from typing import TypedDict, List, Dict


class WordFormDict(TypedDict, total=False):
    """
    Dictionary representation of a word form from LMDB.
    
    This is the format returned by LMDBQuery.lookup().
    All morphological features use lists for consistency.
    
    Example:
        {
            "stress_variants": [0],
            "pos": ["NOUN"],
            "feats": {
                "Case": ["Acc", "Nom"],
                "Gender": ["Masc"],
                "Number": ["Sing"]
            }
        }
    """
    stress_variants: List[int]  # Vowel indices where stress occurs (0-indexed)
    pos: List[str]  # Part-of-speech tags (e.g., ["NOUN"], ["VERB"])
    feats: Dict[str, List[str]]  # Morphological features with list values
    lemma: str  # Optional: base form of the word


# Type alias for the result of a word lookup
WordLookupResult = List[WordFormDict]


def format_stress_display(word: str, stress_indices: List[int]) -> str:
    """
    Apply visual stress marks to a word for display.
    
    Args:
        word: Base word without stress marks
        stress_indices: List of vowel indices to mark (0-indexed)
    
    Returns:
        Word with combining acute accents (́) on stressed vowels
    
    Example:
        >>> format_stress_display("атлас", [0])
        'а́тлас'
        >>> format_stress_display("атлас", [1])
        'атла́с'
    """
    VOWELS = "уеіїаояиюєУЕІАОЯИЮЄЇ"
    COMBINING_ACUTE = "\u0301"  # ́
    
    chars = list(word)
    vowel_count = 0
    result = []
    
    for char in chars:
        result.append(char)
        if char in VOWELS:
            if vowel_count in stress_indices:
                result.append(COMBINING_ACUTE)
            vowel_count += 1
    
    return "".join(result)


def format_morphology_spacy(form: WordFormDict) -> str:
    """
    Format morphological features in spaCy format.
    
    Args:
        form: WordFormDict with morphological data
    
    Returns:
        String like "Case=Acc,Nom|Gender=Masc|Number=Sing"
        Empty string if no features
    
    Example:
        >>> form = {"pos": ["NOUN"], "feats": {"Case": ["Nom"], "Number": ["Sing"]}}
        >>> format_morphology_spacy(form)
        'Case=Nom|Number=Sing'
    """
    if not form.get("feats"):
        return ""
    
    parts = []
    for key in sorted(form["feats"].keys()):
        values = sorted(form["feats"][key])
        values_str = ",".join(values)
        parts.append(f"{key}={values_str}")
    
    return "|".join(parts)
