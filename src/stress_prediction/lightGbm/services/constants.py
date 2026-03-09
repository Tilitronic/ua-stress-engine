"""Feature engineering constants shared across all training services.

Single source of truth for vowels, morphological maps, POS integers,
suffix/prefix lists, and feature-set configuration. Import these
instead of duplicating magic values.
"""

from typing import Dict, Tuple

# ── Vowel inventory ──
VOWELS: str = "аеєиіїоуюя"
VOWEL_SET: frozenset = frozenset(VOWELS)
VOWEL_CHAR_MAP: Dict[str, int] = {c: i for i, c in enumerate(VOWELS)}

MAX_VOWEL_CLASS: int = 10
NUM_CLASSES: int = MAX_VOWEL_CLASS + 1

# ── POS → integer encoding ──
POS_INT: Dict[str, int] = {
    "NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, "NUM": 4,
    "PRON": 5, "DET": 6, "PART": 7, "CONJ": 8, "ADP": 9, "INTJ": 10, "X": 11,
}

# ── Syllable onset pattern codes ──
PATTERN_INT: Dict[str, int] = {
    "V": 0, "CV": 1, "CCV": 2, "CCCV": 3, "CCCCV": 4, "no_vowel": 5,
}

# ── Morphological maps ──
CASE_MAP: Dict[str, int] = {
    "Nom": 0, "Gen": 1, "Dat": 2, "Acc": 3, "Ins": 4, "Loc": 5, "Voc": 6,
}
GENDER_MAP: Dict[str, int] = {"Masc": 0, "Fem": 1, "Neut": 2}
NUMBER_MAP: Dict[str, int] = {"Sing": 0, "Plur": 1}
TENSE_MAP: Dict[str, int] = {"Past": 0, "Pres": 1, "Fut": 2}
ASPECT_MAP: Dict[str, int] = {"Imp": 0, "Perf": 1}
DEGREE_MAP: Dict[str, int] = {"Pos": 0, "Cmp": 1, "Sup": 2}

# ── Suffix / prefix linguistics ──
MASC_STRESS_SUFFIXES: Tuple[str, ...] = (
    "ак", "як", "аль", "ань", "ач", "ій", "іж", "чук", "ун", "няк", "усь",
    "ар", "яр", "іст", "ист",
)
FOREIGN_FINAL_SUFFIXES: Tuple[str, ...] = ("ист", "іст", "ізм", "ант", "ент")
GREEK_INTERFIX_SUFFIXES: Tuple[str, ...] = ("лог", "граф", "фон", "скоп", "метр")
DEVERBAL_SUFFIXES: Tuple[str, ...] = ("ання", "ення", "іння")
MEASURE_SUFFIXES: Tuple[str, ...] = ("метр", "грам", "літр")
DIMINUTIVE_ADJ: Tuple[str, ...] = ("еньк", "есеньк", "юсіньк")
ROOT_STRESS_ADJ: Tuple[str, ...] = ("лив", "аст", "ист", "ев", "ав", "ів", "зьк", "цьк")
COMPOUND_INTERFIXES: Tuple[str, ...] = ("о", "е", "є", "і")
COMMON_PREFIXES: Tuple[str, ...] = (
    "без", "від", "до", "з", "за", "на", "над", "не", "об", "пере",
    "перед", "під", "по", "при", "про", "роз", "ви",
)

# Акцентні парадигми: парадигма B — наголос на суфіксі/флексії (окситонеза)
OXYTONE_MOBILE_SUFFIXES: Tuple[str, ...] = (
    "ак", "як", "ар", "яр", "ач", "ун", "няк", "аль", "ань",
)
# Передостаннє наголошення: тільки -ість дає 2-складові слова (радість, злість)
# -ання/-ення/-іння/-атня завжди 3+ склади → прибрані для 2syl-спеціаліста
PENULT_STABLE_SUFFIXES: Tuple[str, ...] = (
    "ість",
)
# Прикметникові суфікси -зьк/-цьк/-ськ — наголос на основі (парадигма A)
# Вказуємо повне закінчення форми, бо endswith() порівнює кінець рядка
ADCAT_SUFFIXES: Tuple[str, ...] = ("зький", "цький", "ський")

# ── Per-syllable breakdown buckets ──
SYLLABLE_BUCKETS = [2, 3, 4, 5, 6, 7]  # 7+ grouped together

# ── Expected feature count ──
EXPECTED_FEATURE_COUNT: int = 100
