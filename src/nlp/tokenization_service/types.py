#!/usr/bin/env python3
"""
Tokenization Service Data Types

Comprehensive Pydantic models for Ukrainian text tokenization with spaCy.
Provides runtime type validation and automatic data coercion for API safety.

Uses Pydantic for:
- Runtime type enforcement and validation
- Automatic data conversion (e.g., "5" -> 5)
- JSON serialization/deserialization
- API compatibility (FastAPI integration)
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class TokenData(BaseModel):
    """
    Complete token information from spaCy processing with runtime validation.
    
    This model captures all linguistic properties extracted by spaCy's Ukrainian
    language model (uk_core_news_lg), providing comprehensive token-level analysis
    including morphology, syntax, semantics, and named entity information.
    
    All fields are validated at runtime using Pydantic, ensuring type safety when
    processing external data (API requests, file uploads, database records).
    """
    
    # === BASIC TEXT PROPERTIES ===
    # Raw and normalized forms of the token text
    
    text: str = Field(
        ...,
        description="Original token text exactly as it appears in the source document",
        examples=["слово", "123", "!", "ім'я"]
    )
    
    text_lower: str = Field(
        ...,
        description="Lowercase version of token text for case-insensitive operations",
        examples=["слово", "123", "!", "ім'я"]
    )
    
    text_normalized: str = Field(
        ...,
        description="Normalized form: lowercase + apostrophe normalization (U+02BC → U+0027)",
        examples=["слово", "ім'я"]  # Note: apostrophe normalized
    )
    
    lemma: str = Field(
        ...,
        description="Dictionary/base form of the word (lemmatization result from spaCy)",
        examples=["слово", "бути", "великий"]  # "слова"→"слово", "був"→"бути"
    )
    
    # === POSITION INFORMATION ===
    # Location of token within the document
    
    idx: int = Field(
        ...,
        ge=0,
        description="Character offset from start of document (0-indexed byte position)",
        examples=[0, 5, 142]
    )
    
    i: int = Field(
        ...,
        ge=0,
        description="Token index within the document (0-indexed sequential position)",
        examples=[0, 1, 2]  # First token is 0, second is 1, etc.
    )
    
    # === LINGUISTIC ANNOTATIONS ===
    # Part-of-speech and syntactic structure
    
    pos: str = Field(
        ...,
        description="Universal POS tag (UPOS) from Universal Dependencies standard",
        examples=["NOUN", "VERB", "ADJ", "ADP", "PUNCT"]
    )
    
    tag: str = Field(
        ...,
        description="Language-specific fine-grained POS tag (e.g., Ukrainian morphological tag)",
        examples=["Ncmsnn", "Vmis-sm", "Afpnsnf"]  # Ukrainian-specific morphosyntactic descriptors
    )
    
    dep: str = Field(
        ...,
        description="Dependency relation to syntactic head (Universal Dependencies label)",
        examples=["nsubj", "ROOT", "obj", "amod", "case"]  # Subject, root, object, modifier, etc.
    )
    
    head_idx: int = Field(
        ...,
        ge=0,
        description="Token index of syntactic head (the word this token depends on)",
        examples=[2, 0, 5]  # Index points to another token in the sentence
    )
    
    head_lemma: str = Field(
        default="",
        description="Lemma of the syntactic head token (for dependency-based disambiguation)",
        examples=["читати", "стояти", "великий"]  # Helps identify semantic context
    )
    
    # === MORPHOLOGICAL FEATURES ===
    # Grammatical properties from Universal Dependencies
    
    morph: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Morphological features following Universal Dependencies standard. "
            "Contains grammatical categories like Case, Gender, Number, Tense, Aspect, etc. "
            "Each feature has a standardized name and value set."
        ),
        examples=[
            {"Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            {"Tense": "Past", "VerbForm": "Fin", "Aspect": "Perf"},
            {"Degree": "Pos"}  # Positive degree for adjectives
        ]
    )
    
    # === TOKEN CLASSIFICATION FLAGS ===
    # Boolean properties describing token character composition
    
    is_alpha: bool = Field(
        default=False,
        description="True if token contains only alphabetic characters (no digits/punctuation)",
        examples=[True]  # "слово" → True, "слово123" → False
    )
    
    is_ascii: bool = Field(
        default=False,
        description="True if all characters are ASCII (basic Latin, no Cyrillic/Unicode)",
        examples=[False]  # Ukrainian text is typically non-ASCII (Cyrillic)
    )
    
    is_digit: bool = Field(
        default=False,
        description="True if token contains only numeric digits",
        examples=[True]  # "123" → True, "12a" → False
    )
    
    is_lower: bool = Field(
        default=False,
        description="True if all cased characters are lowercase",
        examples=[True]  # "слово" → True, "Слово" → False
    )
    
    is_upper: bool = Field(
        default=False,
        description="True if all cased characters are uppercase",
        examples=[True]  # "СЛОВО" → True, "Слово" → False
    )
    
    is_title: bool = Field(
        default=False,
        description="True if first character is uppercase and rest are lowercase (title case)",
        examples=[True]  # "Слово" → True, "слово" → False, "СЛОВО" → False
    )
    
    is_punct: bool = Field(
        default=False,
        description="True if token is punctuation (period, comma, bracket, etc.)",
        examples=[True]  # ".", ",", "!", "?" → True
    )
    
    is_space: bool = Field(
        default=False,
        description="True if token consists entirely of whitespace characters",
        examples=[False]  # Usually False as spaCy typically excludes whitespace tokens
    )
    
    is_stop: bool = Field(
        default=False,
        description="True if token is a stop word (common words: і, в, на, то, etc.)",
        examples=[True]  # "і", "в", "на" → True (common function words)
    )
    
    is_oov: bool = Field(
        default=False,
        description=(
            "True if token is Out-Of-Vocabulary (not found in spaCy's lexicon). "
            "Indicates rare/unknown words, typos, or domain-specific terminology."
        ),
        examples=[True]  # Neologisms, proper names, typos → True
    )
    
    # === PATTERN MATCHING FLAGS ===
    # Heuristic detection of special token types
    
    like_num: bool = Field(
        default=False,
        description="True if token resembles a number (digits, roman numerals, written numbers)",
        examples=[True]  # "123", "5.5", "п'ять", "XII" → True
    )
    
    like_url: bool = Field(
        default=False,
        description="True if token looks like a URL (contains http://, www., domain pattern)",
        examples=[True]  # "google.com", "http://example.org" → True
    )
    
    like_email: bool = Field(
        default=False,
        description="True if token resembles an email address (contains @ with domain)",
        examples=[True]  # "user@example.com" → True
    )
    
    # === WORD SHAPE ===
    # Abstract representation of token's character pattern
    
    shape: str = Field(
        default="",
        description=(
            "Abstract word shape showing capitalization and character type patterns. "
            "X = uppercase, x = lowercase, d = digit. Used for named entity recognition."
        ),
        examples=["Xxxxx", "XXXX", "dddd", "Xx."]  # "Слово" → "Xxxxx", "2023" → "dddd"
    )
    
    # === WHITESPACE ===
    # Trailing space after token
    
    whitespace: str = Field(
        default="",
        description="Trailing whitespace characters immediately after token (for reconstruction)",
        examples=[" ", "", "\\n", "\\t"]  # Space, no space, newline, tab
    )
    
    # === NAMED ENTITY RECOGNITION ===
    # Entity identification and classification
    
    ent_type: str = Field(
        default="",
        description=(
            "Named Entity type label if token is part of an entity. "
            "Common types: PERSON, ORG, LOC, GPE, DATE, MONEY, etc."
        ),
        examples=["PERSON", "ORG", "LOC", ""]  # "Київ" → "GPE", regular word → ""
    )
    
    ent_iob: str = Field(
        default="",
        description=(
            "IOB (Inside-Outside-Begin) tag for entity boundaries. "
            "B = begin entity, I = inside entity, O = outside any entity."
        ),
        examples=["B", "I", "O"]  # "B-PERSON I-PERSON O" for "Тарас Шевченко любить"
    )
    
    ent_id: str = Field(
        default="",
        description="Entity ID from knowledge base (if linked to external entity database)",
        examples=["Q1899", ""]  # Wikidata ID or similar knowledge base identifier
    )
    
    ent_kb_id: str = Field(
        default="",
        description="Knowledge base identifier for entity linking (explicit KB ID field)",
        examples=["Q1899", "KB:123456", ""]  # External knowledge base reference
    )
    
    # === SENTENCE BOUNDARIES ===
    # Sentence segmentation markers
    
    is_sent_start: Optional[bool] = Field(
        default=None,
        description=(
            "Sentence boundary marker: True = sentence start, False = not start, "
            "None = uncertain (spaCy couldn't determine)."
        ),
        examples=[True, False, None]  # First token of sentence → True
    )
    
    is_sent_end: bool = Field(
        default=False,
        description="True if token is the last token of a sentence",
        examples=[True]  # Last token before period/end → True
    )
    
    # === LEXICAL FEATURES ===
    # Normalized forms and affixes
    
    norm: str = Field(
        default="",
        description=(
            "Normalized form of the token for better matching. "
            "Lowercased, may include spelling corrections or standard forms."
        ),
        examples=["слово", "100"]  # Normalized representation
    )
    
    prefix: str = Field(
        default="",
        description="First 1-3 characters of token (prefix for pattern matching)",
        examples=["сло", "не", "пр"]  # First few characters
    )
    
    suffix: str = Field(
        default="",
        description="Last 1-3 characters of token (suffix for morphological patterns)",
        examples=["ово", "ить", "ний"]  # Last few characters, often indicate morphology
    )
    
    # === PUNCTUATION CLASSIFICATION ===
    # Detailed punctuation type flags
    
    is_bracket: bool = Field(
        default=False,
        description="True if token is a bracket character: (, ), [, ], {, }",
        examples=[True]  # "(", "[" → True
    )
    
    is_quote: bool = Field(
        default=False,
        description="True if token is a quotation mark: \", ', «, », etc.",
        examples=[True]  # '"', "«" → True
    )
    
    is_currency: bool = Field(
        default=False,
        description="True if token is a currency symbol: $, €, ₴, £, ¥, etc.",
        examples=[True]  # "$", "₴" → True (hryvnia symbol)
    )
    
    is_left_punct: bool = Field(
        default=False,
        description="True if token is left/opening punctuation: (, [, {, «, etc.",
        examples=[True]  # "(", "«" → True
    )
    
    is_right_punct: bool = Field(
        default=False,
        description="True if token is right/closing punctuation: ), ], }, », etc.",
        examples=[True]  # ")", "»" → True
    )
    
    # === SYNTACTIC TREE ===
    # Dependency tree structure
    
    n_lefts: int = Field(
        default=0,
        ge=0,
        description="Number of syntactic children to the left of this token",
        examples=[0, 1, 2]  # Count of left dependents
    )
    
    n_rights: int = Field(
        default=0,
        ge=0,
        description="Number of syntactic children to the right of this token",
        examples=[0, 1, 2]  # Count of right dependents
    )
    
    # === STATISTICAL & VECTOR PROPERTIES ===
    # Corpus statistics and word embeddings
    
    lang: str = Field(
        default="",
        description="ISO 639-1 language code (should be 'uk' for Ukrainian)",
        examples=["uk", "en", "ru"]  # Two-letter language code
    )
    
    has_vector: bool = Field(
        default=False,
        description=(
            "True if token has a word vector (word embedding). "
            "Vectors enable semantic similarity calculations."
        ),
        examples=[True]  # Common words have vectors, rare words may not
    )
    
    vector_norm: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "L2 norm (magnitude) of the word vector. "
            "Used for cosine similarity: similarity = dot(v1,v2) / (norm(v1) * norm(v2)). "
            "Zero if has_vector is False."
        ),
        examples=[5.123, 0.0]  # Typically 1-10 for normalized vectors
    )
    
    rank: int = Field(
        default=0,
        ge=0,
        description=(
            "Corpus frequency rank: lower number = more common word. "
            "Rank 1 = most frequent word, rank 100000 = very rare. "
            "Zero if not available."
        ),
        examples=[1, 100, 50000]  # "і" might be rank 1, rare words > 100000
    )
    
    prob: float = Field(
        default=0.0,
        description=(
            "Smoothed log probability estimate of token in corpus. "
            "Negative values: more negative = less frequent. "
            "Based on word frequency in training data."
        ),
        examples=[-5.5, -15.2, 0.0]  # Logarithmic scale, typically negative
    )
    
    cluster: int = Field(
        default=0,
        ge=0,
        description=(
            "Brown cluster ID: hierarchical word class from corpus statistics. "
            "Words with similar distributions get same cluster. "
            "Useful for handling rare words. Zero if not available."
        ),
        examples=[0, 42, 512]  # Cluster IDs from Brown clustering algorithm
    )
    
    sentiment: float = Field(
        default=0.0,
        description=(
            "Sentiment polarity score if available from lexicon. "
            "Range: -1.0 (negative) to +1.0 (positive), 0.0 = neutral or unavailable."
        ),
        examples=[0.8, -0.5, 0.0]  # "чудовий" → +0.8, "поганий" → -0.5
    )
    
    # Pydantic configuration
    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="forbid",
        # Validate on assignment for runtime safety
        validate_assignment=True,
        # Use enum values (if we add enums later)
        use_enum_values=True,
        # Strict types by default
        strict=False,  # Allow coercion for convenience (e.g., "5" -> 5)
    )


class SentenceData(BaseModel):
    """
    Complete sentence information with runtime validation.
    
    Represents a single sentence from the tokenized document, containing
    the sentence text and all its constituent tokens with full linguistic analysis.
    """
    
    text: str = Field(
        ...,
        description="Original sentence text exactly as it appears in source",
        examples=["Це перше речення.", "Яка гарна погода!"]
    )
    
    text_normalized: str = Field(
        ...,
        description="Normalized sentence text (apostrophe normalization applied)",
        examples=["Це перше речення.", "Ім'я автора відоме."]
    )
    
    start_char: int = Field(
        ...,
        ge=0,
        description="Character offset of sentence start in the document (0-indexed)",
        examples=[0, 23, 156]
    )
    
    end_char: int = Field(
        ...,
        ge=0,
        description="Character offset of sentence end in the document (exclusive)",
        examples=[22, 45, 200]
    )
    
    tokens: list[TokenData] = Field(
        default_factory=list,
        description="List of all tokens in this sentence with full linguistic analysis",
        examples=[[]]  # List of TokenData objects
    )
    
    # Pydantic configuration
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )



class DocumentData(BaseModel):
    """
    Complete document tokenization result with runtime validation.
    
    Top-level container for all tokenization results, containing sentences,
    tokens, and document-level metadata. Provides properties for quick
    access to document statistics.
    
    Ideal for API responses, database storage, and inter-service communication.
    """
    
    text: str = Field(
        ...,
        description="Original document text before any normalization",
        examples=["Перше речення. Друге речення.", "Короткий текст."]
    )
    
    text_normalized: str = Field(
        ...,
        description="Normalized document text (apostrophe normalization applied)",
        examples=["Перше речення. Друге речення.", "Ім'я та прізвище."]
    )
    
    sentences: list[SentenceData] = Field(
        default_factory=list,
        description="List of all sentences in the document with their tokens",
        examples=[[]]  # List of SentenceData objects
    )
    
    lang: str = Field(
        default="uk",
        description="ISO 639-1 language code of the document",
        examples=["uk", "en", "ru"]
    )
    
    has_vectors: bool = Field(
        default=False,
        description=(
            "Whether word vectors (embeddings) are available in the model. "
            "True enables semantic similarity calculations."
        ),
        examples=[True, False]
    )
    
    @property
    def token_count(self) -> int:
        """
        Total number of tokens across all sentences in the document.
        
        Returns:
            Integer count of all tokens
        """
        return sum(len(sent.tokens) for sent in self.sentences)
    
    @property
    def sentence_count(self) -> int:
        """
        Total number of sentences in the document.
        
        Returns:
            Integer count of sentences
        """
        return len(self.sentences)
    
    # Pydantic configuration
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


# === TYPE ALIASES ===
# Convenient shortcuts for type hints

TokenList = list[TokenData]
"""Type alias: List of TokenData objects for function signatures"""

SentenceList = list[SentenceData]
"""Type alias: List of SentenceData objects for function signatures"""