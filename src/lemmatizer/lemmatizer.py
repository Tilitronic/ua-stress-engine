import pymorphy3
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

_stanza_available = False
try:
    import stanza
    _stanza_available = True
except ImportError:
    stanza = None

class TokenLemma(BaseModel):
    word: str = Field(..., description="Original token text", examples=["слово", "імені"])
    lemma: str = Field(..., description="Canonical lemma for the token", examples=["слово", "ім'я"])
    pos: str = Field(..., description="Universal POS tag (UPOS)", examples=["NOUN", "VERB"])
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class Lemmatizer:
    """
    Comprehensive lemmatization service for Ukrainian.
    Combines dictionary-based (VESUM/pymorphy3) and neural (Stanza) methods.
    All public methods use Pydantic models for type safety.
    """
    
    def __init__(self, use_gpu: bool = True):
        # 1. Initialize VESUM (via pymorphy3)
        self.morph = pymorphy3.MorphAnalyzer(lang='uk')
        self._nlp = None
        self._use_gpu = use_gpu
        # Mapping: Stanza Universal POS -> pymorphy3 (VESUM) POS
        self._pos_map = {
            "NOUN": "NOUN", "VERB": "VERB", "ADJ": "ADJF",
            "ADV": "ADVB", "PRON": "NPRO", "DET": "ADJF",
            "NUM": "NUMR", "AUX": "VERB", "ADP": "PREP"
        }

    def get_lemma(self, word_form: str) -> str:
        """
        DB-building mode: converts any word form to its canonical lemma key.
        Uses only the dictionary (fast, context-free).
        Assumes stress has already been removed upstream.
        Returns:
            str: Canonical lemma for the word form
        """
        clean = word_form.strip()
        parsed = self.morph.parse(clean)
        # Return normal_form of the most probable parse
        return parsed[0].normal_form if parsed else clean.lower()

    def analyze_sentence(self, sentence: str) -> List[TokenLemma]:
        """
        Text analysis mode: returns a list of tokens with context-sensitive lemmas.
        Uses a hybrid Stanza + VESUM approach.
        Assumes stress has already been removed upstream.
        Returns:
            List[TokenLemma]: List of token/lemma/POS objects (Pydantic models)
        """
        if self._nlp is None:
            if not _stanza_available:
                raise ImportError("stanza is not installed but is required for analyze_sentence")
            self._nlp = stanza.Pipeline(
                lang='uk',
                processors='tokenize,mwt,pos,lemma',
                use_gpu=self._use_gpu,
                logging_level='WARN'
            )
        clean_sent = sentence.strip()
        doc = self._nlp(clean_sent)
        results: List[TokenLemma] = []

        for sent in doc.sentences:
            for word in sent.words:
                stanza_lemma: str = word.lemma
                stanza_pos: str = word.upos
                # Try to refine lemma using VESUM dictionary, guided by Stanza POS
                choices = self.morph.parse(word.text)
                target_vesum_pos = self._pos_map.get(stanza_pos)
                # Snap to dictionary: look for lemma matching POS in context
                best_match: Optional[str] = next(
                    (p.normal_form for p in choices if p.tag.POS == target_vesum_pos),
                    None
                )
                results.append(TokenLemma(
                    word=word.text,
                    lemma=best_match if best_match else stanza_lemma,
                    pos=stanza_pos
                ))
        return results