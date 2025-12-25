Based on the provided academic materials, here is the technical documentation of the Ukrainian consonant system translated into English.

## Ukrainian Consonant System (Консонантизм)

The Ukrainian consonant system is an organized hierarchy where each of the **32 phonemes** is defined by a specific set of differential features.

---

### 1. Classification of Consonant Phonemes

The classification is based on five primary criteria:

#### A. By the Ratio of Voice to Noise (За співвідношенням голосу і шуму)

1. **Sonorants (Сонорні):** (Voice > Noise) — `/m, n, n', r, r', l, l', v, j/` (`/м, н, н', р, р', л, л', в, й/`).
2. **Noise Consonants (Шумні):** (Noise > Voice or Noise only) — all other phonemes.

- **Voiced (Дзвінкі):** (Noise + Voice) — `/b, d, d', z, z', zh, dz, dz', dzh, h, g/` (`/б, д, д', з, з', ж, дз, дз', дж, г, ґ/`).
- **Voiceless (Глухі):** (Noise only) — `/p, t, t', s, s', sh, ts, ts', ch, kh, k, f/` (`/п, т, т', с, с', ш, ц, ц', ч, х, к, ф/`).

#### B. By Place of Articulation / Active Organ (За місцем творення / Активним органом)

1. **Labials (Губні):** `/b, p, v, m, f/`.
2. **Linguals (Язикові):**

- **Front-lingual (Передньоязикові):** `/d, t, z, s, dz, ts, l, n, r, zh, ch, sh, dzh/` (and their soft counterparts).
- **Mid-lingual (Середньоязикові):** `/j/` (`/й/`).
- **Back-lingual (Задньоязикові):** `/g, k, kh/` (`/ґ, к, х/`).

3. **Pharyngeal (Глотковий):** `/h/` (`/г/`).

#### C. By Manner of Articulation (За способом творення)

1. **Occlusives/Plosives (Зімкнені):** Created by a complete blockage of airflow — `/b, p, d, t, g, k, m, n/`.
2. **Fricatives (Щілинні):** Created by a narrowing (slit) — `/v, f, z, s, zh, sh, h, kh, j/`.
3. **Affricates (Африкати):** Start as a stop and release as a fricative — `/dz, ts, dzh, ch/`.
4. **Trills/Vibrants (Дрижачі):** Created by the vibration of the tongue — `/r, r'/`.

#### D. By Palatalization / Softness (За наявністю чи відсутністю м’якості)

- **Hard/Soft Correlative Pairs:** `/d-d', t-t', z-z', s-s', dz-dz', ts-ts', l-l', n-n', r-r'/`.
- **Always Hard (Завжди тверді):** Labials, Sibilants (`/zh, ch, sh, dzh/`), Back-linguals, and the Pharyngeal.
- **Always Soft (Завжди м'яка):** `/j/` (`/й/`).

#### E. By Nasal Resonance (За участю носового резонатора)

- **Nasals (Носові):** Air passes through the nasal cavity — `/m, n, n'/`.
- **Non-nasals/Oral (Неносові):** All other phonemes.

---

### 2. Mathematical Vector Space Model

To represent a consonant phoneme digitally, we use a 5-dimensional vector .

**Coordinate Encoding:**

1. **Voice Power ():** Voiceless = 0, Voiced = 1, Sonorant = 2.
2. **Place ():** Labial = 1, Front-lingual = 2, Mid-lingual = 3, Back-lingual = 4, Pharyngeal = 5.
3. **Manner ():** Occlusive = 1, Fricative = 2, Affricate = 3, Trill = 4.
4. **Softness ():** Hard = 0, Soft = 1.
5. **Nasality ():** Oral = 0, Nasal = 1.

#### Vector Examples:

- **/d/** = (Voiced, Front-lingual, Occlusive, Hard, Oral).
- **/n'//** = (Sonorant, Front-lingual, Occlusive, Soft, Nasal).
- **/j/** = (Sonorant, Mid-lingual, Fricative, Soft, Oral).

---

### 3. Bibliography

This documentation was synthesized from the following analyzed sources:

1. **Savchenko, I. S. (2014).** _Phonetics, Orthoepy, and Graphics of Modern Ukrainian_. (Classification according to H. R. Peredriy).
2. **Moisiienko, A. K. et al. (2010).** _Modern Ukrainian Literary Language: Lexicology. Phonetics_. (Definition of the 32-phoneme system and correlative pairs).
3. **Totska, N. I. (1997).** _Phonetics, Phonology, Orthoepy, Graphics_. (Active and passive organ differentiation).
4. **Kasianova, O. A. (2018).** _Articulatory and Acoustic Characteristics of the Ukrainian Phoneme /m/_. (Sonority and nasality experimental data).
5. **Boiko, N. I. & Khomych, T. L. (2020).** _Chrestomathy of Phonetics_. (Historical views on phoneme counts).
