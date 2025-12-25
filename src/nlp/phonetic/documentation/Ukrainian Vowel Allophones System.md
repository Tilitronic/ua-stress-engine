In Ukrainian phonology, the relationship between a **phoneme** (the abstract mental unit) and its **allophones** (the physical realizations in speech) is governed by specific rules of position and combinatorics.

The general system differentiates between the **invariant** (the primary form of the phoneme in a "strong" or stressed position) and its **variants** or **allophones** (realizations in "weak" or unstressed positions).

---

### 1. General Principles of the System

- **Invariant:** The main manifestation of a phoneme, realized as an identical sound when not influenced by surrounding sounds or its position in a word.

- **Acommodation:** Vowels change slightly depending on the neighboring consonants (e.g., whether the preceding consonant is soft/palatalized).

- **Reduction/Neutralization:** In unstressed positions, certain phonemes lose their distinct qualities and "pull" toward other phonemes, creating specific allophones.

---

### 2. Allophone Rules for Specific Vowel Phonemes

In the modern Ukrainian literary language, there are **6 vowel phonemes**. Below are their specific allophonic behaviors:

#### **Phonemes /і/, /у/, /а/**

These are considered the most "stable" phonemes. Their pronunciation does not change significantly between stressed and unstressed positions.

- **/і/**: Always remains clear.
- **/у/**: Remains stable, though in a weak position it may move slightly toward the central row.

- **/а/**: Remains stable; in very rapid speech, an unstressed /а/ might move toward the mid-height position.

#### **The /е/ – /и/ Pair (The "Mobile" Vowels)**

The most complex allophone rules apply to these two phonemes because they are phonetically close (front row, mid and high-mid height).

- **/е/ in unstressed position:** Pulls toward [и]. It is transcribed as **[еи]**.
- _Example:_ **несе** is pronounced as [несеи].

- **/и/ in unstressed position:** Pulls toward [е]. It is transcribed as **[ие]**.
- _Example:_ **життя** is pronounced as [жиетʲːа].

- **Rule of Assimilation:** If the next syllable contains a stressed [і], [и], or [у], the unstressed [е] becomes even higher, moving closer to [и] (e.g., _ниеси́_).

#### **Phoneme /о/**

- **/о/ in stressed position:** A clear back-row, mid-height, labialized sound.

- **/о/ in unstressed position:** Usually remains [о], but **before a syllable with a stressed [у] or [і]**, it narrows and becomes labialized toward [у]. This allophone is transcribed as **[оу]**.

- _Example:_ **зозуля** [зоузулʲа], **собі** [соуб'і].

---

### Summary Table of Allophones

| Phoneme | Stressed (Invariant) | Unstressed (Allophone) | Condition                  |
| ------- | -------------------- | ---------------------- | -------------------------- |
| **/і/** | [і]                  | [і]                    | Stable                     |
| **/у/** | [у]                  | [у]                    | Stable                     |
| **/а/** | [а]                  | [а]                    | Stable                     |
| **/е/** | [е]                  | **[еи]**               | Always when unstressed     |
| **/и/** | [и]                  | **[ие]**               | Always when unstressed     |
| **/о/** | [о]                  | **[оу]**               | Before stressed [у] or [і] |

### Visual Representation of Allophonic Shift

These "living" (phonetic) changes are predictable based on the environment and do not change the underlying phonemic structure of the word.

To extend the documentation, here is the mathematical formalization of Ukrainian allophone rules and the bibliography of the sources used to construct this system.

### 1. Mathematical Model for Allophones (Transformation Mapping)

In a computational linguistic context, allophones can be modeled as a function that maps a **phoneme vector** to a **realization vector** based on its environment (Context ).

Let the Phoneme Vector be .

#### A. The Reduction Function (for /е/ and /и/)

The most significant shifts occur in the unstressed front-vowel space. We can define a **Centering Shift** where the height of /е/ increases and the height of /и/ decreases toward a common mid-point.

If is the stress bit (1 for stressed, 0 for unstressed):

- For **/е/** (): Result (Height 2.25)
- For **/и/** (): Result (Height 2.25)

#### B. The Assimilative Labialization (for /о/)

The shift of unstressed /о/ toward [у] happens only if the following syllable contains a high-vowel phoneme .

#### C. Vector Space Visualization

In a 2D plot of (Height) vs (Backness), allophones are represented as "clouds" or "vectors of displacement" from the central phoneme coordinate.

---

### 2. General Allophone Rules Summary

| Phoneme       | Rule Type    | Trigger Condition                | Resulting Allophone (IPA/Transcr.) |
| ------------- | ------------ | -------------------------------- | ---------------------------------- |
| **/е/**       | Position     | Unstressed                       | / [ɛ̝]                              |
| **/и/**       | Position     | Unstressed                       | / [ɪ̞]                              |
| **/о/**       | Assimilation | Unstressed + or in next syllable | / [o̝]                              |
| **/і, у, а/** | Invariance   | General                          | Stable (minimal variance)          |

---

### Python Implementation Hint (Allophone Logic)

```python
def realize_phoneme(phoneme, is_stressed, next_syllable_vowel=None):
    if is_stressed:
        return phoneme # Invariant

    if phoneme == 'e': return 'e_y'
    if phoneme == 'y': return 'y_e'
    if phoneme == 'o' and next_syllable_vowel in ['u', 'i']:
        return 'o_u'

    return phoneme # Stable vowels a, i, u

```

### Bibliography & Source Attribution

The following academic sources provided the articulatory data and phonological rules used for this documentation:

1. **Савченко, І. С. (2014).** _Фонетика, орфоепія і графіка сучасної української мови_. Черкаси: ЧНУ ім. Б. Хмельницького.

- _Role:_ Primary source for the classification of vowels by row and height (§15) and the description of unstressed vowel qualities.

2. **Мойсієнко, А. К. та ін. (2010).** _Сучасна українська літературна мова: Лексикологія. Фонетика_. Київ: Знання.

- _Role:_ Theoretical basis for the 6-phoneme system and the functional definition of phonemes vs. sounds.

3. **Бойко, Н. І., & Хомич, Т. Л. (2020).** _Сучасна українська літературна мова. Хрестоматія з фонетики_. Ніжин: НДУ ім. М. Гоголя.

- _Role:_ Used for historical context on the /і/ - /и/ distinction and definitions of articulatory tension.

4. **Касьянова, О. А. (2015/2018).** _Звукове поле фонеми..._. Інститут філології КНУ імені Тараса Шевченка.

- _Role:_ Methodology for acoustic analysis (using Praat) which informs the numerical/mathematical representation of sound "fields" rather than static points.

5. **Ющук, І. (2010).** _Фонетика української мови_.

- _Role:_ Clarification on the stability of /і, у, а/ in the Ukrainian phonetic system.
