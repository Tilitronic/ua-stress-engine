Based on the comprehensive materials provided, here is the structured documentation for Ukrainian vowel phonemes.

## Ukrainian Vowel System (Вокалізм)

The modern Ukrainian literary language consists of **6 vowel phonemes**. They are defined by three primary articulatory parameters: **Height** (vertical tongue movement), **Backness/Row** (horizontal tongue movement), and **Labialization** (lip rounding).

---

### 1. Classification Table (Phonemes)

This table represents the phonemes in their "strong" (stressed) position.

| Height (Підняття)             | Front (Передній) | Central (Середній) | Back (Задній)   |
| ----------------------------- | ---------------- | ------------------ | --------------- |
| **High** (Високе)             | /і/ (unrounded)  | —                  | /у/ (rounded)   |
| **High-Mid** (Високо-середнє) | /и/ (unrounded)  | —                  | —               |
| **Mid** (Середнє)             | /е/ (unrounded)  | —                  | /о/ (rounded)   |
| **Low** (Низьке)              | —                | —                  | /а/ (unrounded) |

> **Note on /а/:** While phonologically often grouped with the back vowels, phonetically it is the most open (low) sound where the tongue is retracted.

---

### 2. Detailed Phoneme Profiles

| Phoneme | IPA | Ukrainian Name          | English Description | Labialization  |
| ------- | --- | ----------------------- | ------------------- | -------------- |
| **/і/** | [i] | передній високе         | Front High          | Non-labialized |
| **/и/** | [ɪ] | передній високо-середнє | Front High-Mid      | Non-labialized |
| **/е/** | [ɛ] | передній середнє        | Front Mid           | Non-labialized |
| **/у/** | [u] | задній високе           | Back High           | **Labialized** |
| **/о/** | [ɔ] | задній середнє          | Back Mid            | **Labialized** |
| **/а/** | [ɑ] | задній низьке           | Back Low            | Non-labialized |

---

### 3. Allophonic Variation (Unstressed Position)

In Ukrainian, unstressed vowels (especially /е/ and /и/) undergo **accommodation**, moving toward each other in the vowel space.

- **[е]** in unstressed position pulls toward [и]: `[еи]` (e.g., _несе_ [несеи]).
- **[и]** in unstressed position pulls toward [е]: `[ие]` (e.g., _життя_ [жиетʲːа]).
- **[о]** remains relatively stable but may narrow toward [у] before a stressed high vowel (e.g., _зозуля_ [зоузулʲа]).

---

### 4. Mathematical Numerical Representation

To represent these phonemes in a coordinate system (relevant for programming or acoustic analysis), we can map the qualitative features to a 3D vector .

#### Coordinate Mapping Logic:

1. **X-axis (Backness/Row):** Front = 1, Central = 2, Back = 3.
2. **Y-axis (Height):** High = 3, Mid = 2, Low = 1.
3. **Z-axis (Rounding):** Unrounded = 0, Rounded = 1.

#### Phoneme Vector Space:

#### Euclidean Distance in Python:

If you are implementing a phonetic similarity algorithm, you can represent the phonemes as a dictionary of NumPy arrays:

```python
import numpy as np

# Phoneme vectors: (Backness, Height, Rounding)
vowels = {
    'i': np.array([1, 3, 0]),
    'u': np.array([3, 3, 1]),
    'y': np.array([1, 2.5, 0]), # Ukrainian /и/
    'e': np.array([1, 2, 0]),
    'o': np.array([3, 2, 1]),
    'a': np.array([3, 1, 0])
}

def get_similarity(p1, p2):
    # Calculate Euclidean distance
    distance = np.linalg.norm(vowels[p1] - vowels[p2])
    return distance

# Example: distance between /i/ and /e/ is 1.0,
# while /i/ and /u/ is approx 2.23

```

**Would you like me to generate a similar technical breakdown for Ukrainian consonant clusters and their assimilation rules?**

## Bibliography

Primary Sources used for the Documentation:

    Савченко І. С. Фонетика, орфоепія і графіка сучасної української мови : навч. посіб. — Черкаси: ЧНУ ім. Б. Хмельницького, 2014. — 184 с.

    Used for: General classification of vowels, detailed articulatory descriptions of phonemes in stressed and unstressed positions, and the comparative table of vowel realizations.

Мойсієнко А. К., Бас-Кононенко О. В., Бондаренко В. В. та ін. Сучасна українська літературна мова: Лексикологія. Фонетика: підручник. — К.: Знання, 2010. — 270 с.

    Used for: Theoretical framework of the vowel system and systematic classification.

Бойко Н. І., Хомич Т. Л. Сучасна українська літературна мова. Хрестоматія з фонетики: навч. посіб. — 2-ге вид., змінене і доп. — Ніжин: НДУ ім. М. Гоголя, 2020. — 305 с.

    Used for: Contextualizing modern phonological debates and historical perspectives on vowel counts.

Supplementary Materials:

    Ющук І. Фонетика української мови (матеріал для вчителя).

    Used for: Basic distinctions between sounds (phonemes) and letters (graphemes).

Українська фонетика — Вікіпедія.

        Used for: General overview and IPA conventions.

Specialized Research (Reference for broader context):

    Касьянова О. А. Звукове поле фонеми /в/: комбінаторні та позиційні алофони. — 2015.

Касьянова О. А. Артикуляційна та акустична характеристика звукових реалізацій української фонеми /m/. — 2018.
