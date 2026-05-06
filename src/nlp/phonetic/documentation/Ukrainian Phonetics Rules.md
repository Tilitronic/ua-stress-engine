# Ukrainian Phonetics: IPA Rules and Transcription Examples

This file accumulates rules and verified transcription examples extracted from experimental phonetic research on Ukrainian.
Each rule cites its source. Use these as the ground truth for IPA transcriber tests and implementation.

---

## Table of Contents

- [Phoneme /в/ — Labial Approximant](#phoneme-в--labial-approximant)
- [Phoneme /m/ — Bilabial Nasal](#phoneme-m--bilabial-nasal)
- [Ukrainian Phonological System — IPA Overview](#ukrainian-phonological-system--ipa-overview)
- [Vowel and Consonant Classification (Academic Standard)](#vowel-and-consonant-classification-academic-standard)
- [References](#references)

---

## Phoneme /в/ — Labial Approximant

> Source: [kasyanova2015]

### IPA Symbols

| Symbol | Description (UA)                               | Description (EN)                            | IPA number |
| ------ | ---------------------------------------------- | ------------------------------------------- | ---------- |
| `[w]`  | губно-губний задньопіднебінний апроксимант     | bilabial velar approximant                  | 170        |
| `[w̹]`  | огублений губно-губний (більш огублений)       | rounded bilabial (more rounded, before [у]) | —          |
| `[w̜]`  | огублений губно-губний (менш огублений)        | rounded bilabial (less rounded, before [о]) | —          |
| `[ʋ]`  | губно-зубний апроксимант                       | labiodental approximant                     | —          |
| `[ʋʲ]` | напівм'який губно-зубний апроксимант           | soft labiodental approximant                | —          |
| `[ṷ]`  | нескладовий голосний "у"                       | non-syllabic [u]                            | 308+432    |
| `[ʍ]`  | губно-губний задньопіднебінний щілинний шумний | voiceless bilabial fricative                | 169        |

> IPA has no standardised dedicated symbols for Ukrainian-specific realisations. The symbols above follow the convention established in Ukrainian phonetic literature [kasyanova2015].

### Allophone Selection Rules

#### [w] — primary bilabial allophone

The **primary allophone** of /в/ in standard Ukrainian is bilabial [w] [kasyanova2015].

| Phonological context                                   | Example         | Transcription       |
| ------------------------------------------------------ | --------------- | ------------------- |
| Syllable onset before [а]                              | _ваш_           | `[wáʃ]`             |
| Syllable onset before [е]                              | _вечір_         | `[wɛtʃir]`          |
| Syllable onset before [о]                              | _вона_          | `[wɔnɑ]`            |
| Before vowels [о], [у] (basic, without extra rounding) | _вода_, _кавун_ | `[wɔdɑ]`, `[kɑwún]` |
| Before voiced consonants (not after a vowel)           | _вниз_          | `[wɲɪz]`            |

#### [w̹] — more rounded bilabial (before [у])

**Condition:** syllable onset immediately before rounded back vowel [у].

| Example | Transcription | Note                                                                                 |
| ------- | ------------- | ------------------------------------------------------------------------------------ |
| _вухо_  | `[w̹úxɔ]`      | lips pushed forward; oscillogram boundary with [у] indistinguishable [kasyanova2015] |

Acoustic: F1 = 294 Hz, F2 = 1310 Hz. Low F1 = rounding [kasyanova2015].

#### [w̜] — less rounded bilabial (before [о])

**Condition:** syllable onset immediately before rounded back vowel [о].

| Example | Transcription |
| ------- | ------------- |
| _во-_   | `[w̜ó]`        |

#### [ʋ] — labiodental approximant

**Condition:** syllable onset before [а], [е] (coarticulatory variant alongside [w]); primary before [и] [kasyanova2015].

| Example | Transcription |
| ------- | ------------- |
| _ваза_  | `[ʋɑzɑ]`      |
| _видно_ | `[ʋɪdnɔ]`     |
| _вечір_ | `[ʋɛtʃir]`    |
| _вада_  | `[ʋádɑ]`      |

Acoustic: F1 = 406 Hz, F2 = 1643 Hz, F3 = 3127 Hz. Higher F2 than [w] ("diez" quality) [kasyanova2015].

> **Brovchenko & Totska rule:** [ʋ] and [w] coexist before [а], [е], [и] — even in the same word and same speaker [kasyanova2015].

#### [ʋʲ] — soft labiodental approximant

**Condition:** syllable onset immediately before front high vowel [і].

| Example        | Transcription |
| -------------- | ------------- |
| _він_          | `[ʋʲin]`      |
| _свято_        | `[sʲʋʲɑtɔ]`   |
| _ви_ (variant) | `[ʋʲý]`       |

Acoustic: F1 = 228 Hz, F2 = 2100 Hz. High F2 near F3 is the defining signature [kasyanova2015].

#### [ṷ] — non-syllabic vocalised allophone

**Condition:** syllable coda, word-final after vowel, or after vowel in any position.

| Position                              | Example    | Transcription |
| ------------------------------------- | ---------- | ------------- |
| Word-initial before consonant         | _вже_      | `[ṷʒɛ]`       |
| Word-initial before consonant         | _внук_     | `[ṷnúk]`      |
| Word-initial before consonant cluster | _вчора_    | `[ṷtʃɔrɑ]`    |
| Word-initial before consonant cluster | _вдова_    | `[ṷdɔwɑ]`     |
| After vowel, word-final               | _мав_      | `[mɑṷ]`       |
| After vowel, word-final               | _був_      | `[buṷ]`       |
| After vowel, before consonant         | _мавпа_    | `[mɑṷpɑ]`     |
| After vowel, before consonant         | _шовк_     | `[ʃɔṷk]`      |
| After vowel, word-internal            | _правда_   | `[prɑṷdɑ]`    |
| After vowel, word-internal            | _кривда_   | `[krɪṷdɑ]`    |
| After vowel, word-final               | _упав_     | `[upɑṷ]`      |
| After vowel, across word boundary     | _а вперше_ | `[ɐṷpɛrʃɛ]`   |

Acoustic: periodic oscillations, high amplitude, F1 well separated from other formants — resembles vowel [у] but shorter [kasyanova2015].

> **Historical note:** Vocalisation of /в/ across all dialects. _вовк_ (← _влъкъ_), _повен/повний_ (← _плънъ_). Alternation: _вчити/учити_, _вкрасти/украсти_ [kasyanova2015].

#### [ʍ] — voiceless bilabial fricative

**Condition:** word-initial before voiceless consonant, not preceded by vowel [kasyanova2015].

| Example  | Transcription |
| -------- | ------------- |
| _вперше_ | `[ʍpɛrʃɛ]`    |

> Normative status debated — described in Bilous (2005) and Buk, Rovenchak & Machutec [kasyanova2015].

### Summary Table — /в/ Allophone Selection

| Context                                               | Allophone                           | Examples                           |
| ----------------------------------------------------- | ----------------------------------- | ---------------------------------- |
| Onset + before [а], [е]                               | `[w]` primary; `[ʋ]` coarticulatory | _ваш_ `[wáʃ]`, _вада_ `[ʋádɑ]`     |
| Onset + before [о]                                    | `[w̜]`                               | _во-_ `[w̜ó]`                       |
| Onset + before [у]                                    | `[w̹]`                               | _вухо_ `[w̹úxɔ]`                    |
| Onset + before [и]                                    | `[ʋ]`                               | _ви_ `[ʋý]`                        |
| Onset + before [і]                                    | `[ʋʲ]`                              | _він_ `[ʋʲin]`                     |
| Onset + before voiced consonant (not post-vocalic)    | `[w]`                               | _вниз_ `[wnɪz]`                    |
| Onset + before voiceless consonant (not post-vocalic) | `[ʍ]`                               | _вперше_ `[ʍpɛrʃɛ]`                |
| Coda / word-final / before consonant                  | `[ṷ]`                               | _вже_ `[ṷʒɛ]`, _мав_ `[mɑṷ]`       |
| Post-vocalic (any position)                           | `[ṷ]`                               | _мавпа_ `[mɑṷpɑ]`, _шовк_ `[ʃɔṷk]` |

### /в/ ~ [у] Alternation

Both forms normative at word boundaries before consonant clusters [kasyanova2015]:

| /в/ form  | [у] form  |
| --------- | --------- |
| _вчити_   | _учити_   |
| _вкрасти_ | _украсти_ |

### Final Devoicing

Ukrainian does **not** have final devoicing of /в/. Partial devoicing is non-normative [kasyanova2015].

### Acoustic Reference — /в/ Allophones

Formant measurements for isolated allophones, speaker O.B. [kasyanova2015]:

| Allophone               | F1 (Hz) | F2 (Hz) | F3 (Hz) | Characteristic                       |
| ----------------------- | ------- | ------- | ------- | ------------------------------------ |
| `[w]` bilabial          | 354     | 913     | 2399    | F1+F2 close ("bemol"); F3 distant    |
| `[ʋ]` labiodental       | 406     | 1643    | 3127    | F1+F2 separated ("diez"); F2 rises   |
| `[ṷ]` non-syllabic      | 342     | 1007    | 2653    | Periodic, high amplitude; vowel-like |
| `[ʋʲ]` soft labiodental | 228     | 2100    | —       | F2 near F3; very low F1              |
| `[w̹]` rounded bilabial  | 294     | 1310    | 2856    | Low F1 (rounding); F2 variable       |

General /в/ range: F1 = 200–500 Hz, F2 = 1000–1700 Hz, F3 = 2000–2500 Hz [kasyanova2015].

### Dialect Notes

- **South-western dialects:** bilabial [w] dominates
- **Northern and eastern dialects:** labiodental [ʋ] dominates [kasyanova2015]

### Russian Interference

Ukrainian [ʋ] is a sonorant approximant; Russian/Polish /v/ is a fricative obstruent. Substitution is non-normative [kasyanova2015].

---

## Phoneme /m/ — Bilabial Nasal

> Source: [kasyanova2018]

### Basic Classification

| Feature                | Value                                                  |
| ---------------------- | ------------------------------------------------------ |
| Place of articulation  | bilabial (губно-губна)                                 |
| Manner of articulation | nasal continuant (зімкнено-прохідна / зімкнено-носова) |
| Voicing                | voiced sonorant                                        |
| Focus                  | single-focus (однофокусна)                             |
| IPA number             | 114                                                    |

Articulation: lips close completely; velum lowers, airflow diverted through nasal cavity; weak oral airstream also exits through labial closure [kasyanova2018].

### Allophones

| Allophone | Condition                      | Example          | Transcription         |
| --------- | ------------------------------ | ---------------- | --------------------- |
| `[m]`     | basic — all other positions    | _ма_             | `[má]`                |
| `[m˚]`    | before rounded vowels [о], [u] | _морж_, _тріумф_ | `[m˚ó]рж`, `трі[ṹm]ф` |
| `[mʲ]`    | before front high vowel [і]    | _мі-_            | `[mʲí]`               |

#### [m] — basic allophone, IPA 114

| Example | Transcription |
| ------- | ------------- |
| _ма_    | `[má]`        |
| _обман_ | `[obm̃án]`     |

#### [m˚] — rounded (labialized) allophone

**Condition:** immediately before or after rounded back vowels [о] or [u].

| Example  | Transcription | Type                            |
| -------- | ------------- | ------------------------------- |
| _морж_   | `[m˚ṍrʒ]`     | progressive nasalisation of [о] |
| _тріумф_ | `трі[ṹm˚]ф`   | regressive nasalisation of [u]  |

Acoustic: all formant values lower due to increased resonator volume from lip protrusion [kasyanova2018].

#### [mʲ] — semi-palatal allophone

**Condition:** immediately before front high vowel [і].

| Example | Transcription |
| ------- | ------------- |
| _мі-_   | `[mʲí]`       |

Acoustic: F1 distant from F2; F3 close to F2 — tongue dorsum raised toward hard palate [kasyanova2018].

### Nasalisation of Adjacent Vowels

| Type                      | Direction               | Example  | Transcription |
| ------------------------- | ----------------------- | -------- | ------------- |
| Progressive               | nasal → following vowel | _морж_   | `[m˚ṍrʒ]`     |
| Regressive                | preceding vowel ← nasal | _тріумф_ | `трі[ṹm]ф`    |
| Full (between two nasals) | both                    | _мама_   | `[mãmã]`      |

### Homorganic Clusters [bm], [pm]

In connected speech, [b]/[p] before [m] forms one articulatory gesture [kasyanova2018]:

- Stop acquires nasal colouring (progressive assimilation)
- Nasal loses smooth articulation, becomes explosive (regressive assimilation)

| Example | Transcription | Note                                                  |
| ------- | ------------- | ----------------------------------------------------- |
| _обман_ | `[obm̃án]`     | no initial impulse of [m]; boundary indistinguishable |

Formant comparison within _обман_ [kasyanova2018]:

| Segment | F1  | F2   | F3   | F4   |
| ------- | --- | ---- | ---- | ---- |
| `[b]`   | 424 | 1046 | 2715 | 3857 |
| `[m]`   | 280 | 1053 | 2780 | 3753 |

### Acoustic Characteristics — /m/

**Spectrogram signatures** [kasyanova2018]:

1. Anti-resonances in F1–F2 bands, especially F2 nulls; spectral energy loss in band (more prominent in connected speech)
2. Low F1 = 200–400 Hz (locus); Praat average 400–600 Hz. Reflects bilabial focus and posterior dorsal position
3. Clear formant pattern typical of nasals (Fant). Vowel forms "inside" the labial — boundary indistinguishable
4. Darkened region below ~1300 Hz — sonorant marker; resembles vowel spectrogram
5. Unstable F3, F4 — linked to laryngopharyngeal tension and nasopharyngeal resonance; shadowed bands at 3500–4000 Hz
6. Intensity: rising at word-onset, falling at word-final, rising-then-falling word-medially

**Oscillogram signatures** [kasyanova2018]:

- Short closure impulse at onset, then gradual opening with increasing amplitude
- Brush-like waveform throughout
- Periodic oscillations (sonorant marker)
- Boundary with adjacent vowel difficult to identify

**Formant reference data** [kasyanova2018]:

| Context               | Allophone | F1      | F2        | F3   | F4   |
| --------------------- | --------- | ------- | --------- | ---- | ---- |
| `[má]` isolated       | [m]       | 200–400 | anti-res. | —    | —    |
| _і в на прямі_ (text) | [mʲ]      | 330     | 1600      | 2650 | 3900 |
| `[m˚ó]рж`             | [m˚]      | 275     | 1196      | 2797 | 3959 |
| `[mʲí]` syllable      | [mʲ]      | 756     | 2253      | 3207 | 4500 |
| `трі[ṹm]ф`            | [m˚]      | 542     | 1188      | 2812 | 4100 |

> Lowest formants before [о] or after [u]; highest before [і] [kasyanova2018].

**Palatalisation signature:** F1 distant from F2; F3 close to F2.

**Rounding signature:** all formant values decrease; active orbicularis oris.

### /m/ Positional Summary

| Position            | Allophone           | Behaviour                                             |
| ------------------- | ------------------- | ----------------------------------------------------- |
| Before [о], [u]     | `[m˚]`              | rounding; progressive nasalisation of following vowel |
| Before [і]          | `[mʲ]`              | palatalisation; F1↓, F2↑ toward F3                    |
| After [u]           | `[m˚]`              | regressive nasalisation of preceding vowel            |
| Between two nasals  | `[m]`               | full nasalisation of intervening vowel `[mãmã]`       |
| After [b] or [p]    | `[m]` (assimilated) | homorganic cluster; stop nasalised, nasal devoiced    |
| All other positions | `[m]`               | basic allophone                                       |

### /m/ vs /f/ — Labial Comparison

| Feature   | `[m]` bilabial nasal                                      | `[f]` labiodental fricative           |
| --------- | --------------------------------------------------------- | ------------------------------------- |
| Voicing   | voiced sonorant                                           | voiceless obstruent                   |
| Wave      | periodic                                                  | aperiodic, low-amplitude, interrupted |
| Vocality  | strong                                                    | consonantal                           |
| Dark band | below ~1300 Hz                                            | 2100–3300 Hz                          |
| Shared    | anti-resonances at lower formants; maximum coarticulation | same                                  |

---

## Ukrainian Phonological System — IPA Overview

> Source: [steriopolo2012]

### Phonological Inventory

**32 consonant phonemes + 6 vowel phonemes** [steriopolo2012].

---

### Consonant System

#### Table: Ukrainian Consonants (IPA, after Steriopolo 2012)

|                      | Bilabial  | Labiodental | Dental/Alveolar | Post-alveolar | Palatal | Velar     | Glottal |
| -------------------- | --------- | ----------- | --------------- | ------------- | ------- | --------- | ------- |
| **Plosive** hard     | p b       |             | t̪ d̪             |               |         | k ɡ       |         |
| **Plosive** soft     | (pʲ) (bʲ) |             | tʲ dʲ           |               |         | (kʲ) (ɡʲ) |         |
| **Nasal** hard       | m         |             | n̪               |               |         |           |         |
| **Nasal** soft       | (mʲ)      |             | nʲ              |               |         |           |         |
| **Fricative** hard   |           | f v         | s z             | (ʃ) (ʒ)       |         | x         | ɦ       |
| **Fricative** soft   |           | (fʲ) (vʲ)   | sʲ zʲ           |               |         | (xʲ)      | (ɦʲ)    |
| **Affricate** hard   |           |             | ts dz           | (tʃ) (dʒ)     |         |           |         |
| **Affricate** soft   |           |             | tsʲ dzʲ         |               |         |           |         |
| **Trill** hard       |           |             | r               |               |         |           |         |
| **Trill** soft       |           |             | rʲ              |               |         |           |         |
| **Approximant** hard | w         |             |                 |               |         |           |         |
| **Approximant** soft |           |             |                 |               | j       |           |         |
| **Lateral** hard     |           |             | l               |               |         |           |         |
| **Lateral** soft     |           |             | lʲ              |               |         |           |         |

> Items in parentheses are **context-conditioned semi-soft allophones** of the corresponding hard consonants [steriopolo2012].
> Left cell = voiceless, right cell = voiced (IPA convention).

#### Retroflex vs. Soft Realization of Sibilants and Affricates

Ukrainian /ʃ, ʒ, tʃ, dʒ/ have two main realizations [steriopolo2012]:

| Phoneme | Hard (retroflex) symbol | Soft (post-alveolar) symbol | Condition                                                                                |
| ------- | ----------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| /ʃ/     | `[ʂ]`                   | `[ʃ]`                       | [ʂ] = tongue tip raised/curled toward palate; [ʃ] = tongue advanced toward lower alveoli |
| /ʒ/     | `[ʐ]`                   | `[ʒ]`                       | same distinction                                                                         |
| /tʃ/    | `[ʈʂ]`                  | `[tʃ]`                      | same distinction                                                                         |
| /dʒ/    | `[ɖʐ]`                  | `[dʒ]`                      | same distinction                                                                         |

> In Ukrainian phonological literature the sibilants are traditionally classified as "front-lingual" (передньоязикові), but articulatorily the tongue tip rises and curls toward the hard palate — hence the retroflex symbol is more precise for the hard realizations [steriopolo2012].

---

### Word Examples for Each Consonant Phoneme

| Phoneme   | IPA transcription | Word                             | Ukrainian  |
| --------- | ----------------- | -------------------------------- | ---------- |
| p         | `pɨu̯`             | _пив_                            | пив        |
| pʲ        | `pʲiu̯`            | _пів_                            | пів        |
| b         | `baz`             | _баз_                            | баз        |
| bʲ        | `ˈbʲazʲ`          | _бязь_                           | бязь       |
| m         | `mau̯`             | _мав_                            | мав        |
| mʲ        | `mʲulʲ`           | _мюль_                           | мюль       |
| f         | `ˈfaza`           | _фаза_                           | фаза       |
| fʲ        | `ˈfʲikus`         | _фікус_                          | фікус      |
| v         | `svat̪`            | _сват_                           | сват       |
| vʲ        | `sʲvʲatɨi̯`        | _святий_                         | святий     |
| t̪         | `tam̪`             | _там_                            | там        |
| tʲ        | `tʲam`            | _тям_                            | тям        |
| d̪         | `lad̪`             | _лад_                            | лад        |
| dʲ        | `ladʲ`            | _ладь_                           | ладь       |
| n̪         | `stan`            | _стан_                           | стан       |
| nʲ        | `stanʲ`           | _стань_                          | стань      |
| s         | `rɨs`             | _рис_                            | рис        |
| sʲ        | `rɨsʲ`            | _рись_                           | рись       |
| z         | `pɛrɛˈlaz`        | _перелаз_                        | перелаз    |
| zʲ        | `pɛrɛˈlazʲ`       | _перелазь_                       | перелазь   |
| ʂ (hard)  | `ʃar`             | _шар_                            | шар        |
| ʐ (hard)  | `ʒar`             | _жар_                            | жар        |
| ʃ (soft)  | `pʲiˈdːaʃːa`      | _піддашшя_                       | піддашшя   |
| ʒ (soft)  | `ˈpɔdɔrɔʒi`       | _подорожі_                       | подорожі   |
| ts        | `tsɛp`            | _цеп_                            | цеп        |
| tsʲ       | `tsʲɔm`           | _цьом_                           | цьом       |
| ʈʂ (hard) | `tʃɔm`            | _чом_                            | чом        |
| tʃ (soft) | `ˈnʲitʃːu`        | _ніччю_                          | ніччю      |
| dz        | `ˈdzvɔnɨ`         | _дзвони_                         | дзвони     |
| dzʲ       | `dzʲurˈʈʂatɨ`     | _дзюрчати_                       | дзюрчати   |
| ɖʐ (hard) | `suˈdʒu`          | _суджу_                          | суджу      |
| dʒ (soft) | `ˈbdʒilka`        | _бджілка_                        | бджілка    |
| k         | `kɨnʲ`            | _кинь_                           | кинь       |
| kʲ        | `kʲinʲ`           | _кінь_                           | кінь       |
| ɡ         | `ɡɛdzʲ`           | _ґедзь_                          | ґедзь      |
| ɡʲ        | `ɡʲilʲɔˈtɨna`     | _ґільйотина_                     | ґільйотина |
| x         | `xai̯`             | _хай_                            | хай        |
| xʲ        | `xʲid̪`            | _хід_                            | хід        |
| ɦ         | `ɦai̯`             | _гай_                            | гай        |
| ɦʲ        | `ɦʲisʲtʲ`         | _гість_                          | гість      |
| r         | `rad̪`             | _рад_                            | рад        |
| rʲ        | `rʲad̪`            | _ряд_                            | ряд        |
| l         | `luk`             | _лук_                            | лук        |
| lʲ        | `lʲuk`            | _люк_                            | люк        |
| j         | `ˈjiʒa`           | _їжа_                            | їжа        |
| w         | —                 | (approximant, hard /в/ in onset) | —          |

---

### Consonant Phonological Rules

#### (1) Voicing

- **No aspiration** of voiceless stops (unlike Germanic languages); tension stronger than in Russian [steriopolo2012].
- **No final devoicing** — voiced consonants remain voiced in coda and word-finally [steriopolo2012]:

  | Example  | Transcription                                         |
  | -------- | ----------------------------------------------------- |
  | _дуб_    | `[dub]`                                               |
  | _стежка_ | `[ˈstɛʒka]` — /ʒ/ remains voiced before voiceless /k/ |

- **Regressive voicing assimilation** — voiceless consonant becomes voiced before a voiced consonant [steriopolo2012]:

  | Example     | Transcription   | Note                 |
  | ----------- | --------------- | -------------------- |
  | _баскетбол_ | `[baskɛɨdˈbɔl]` | /t/ → [d] before /b/ |

#### (2) Palatalization

18 phonemes form 9 palatalization pairs (hard : soft) [steriopolo2012]:

`/t̪ : tʲ/`, `/d̪ : dʲ/`, `/s : sʲ/`, `/z : zʲ/`, `/n̪ : nʲ/`, `/l : lʲ/`, `/r : rʲ/`, `/ts : tsʲ/`, `/dz : dzʲ/`

Palatalization = secondary articulation: front dorsum of tongue raises toward hard palate [steriopolo2012].

**Semi-palatalized consonants** ("peripheral subsystem", Totska) — context-conditioned soft allophones of hard consonants, mainly in loanwords. /ʒ, ʃ/ and /tʃ/ can occur before [i], [a], [u] [steriopolo2012]:

| Example    | Transcription  |
| ---------- | -------------- |
| _суміш+шю_ | `[ˈsumʲitʃːu]` |
| _ніч+чю_   | `[ˈnʲitʃːu]`   |

#### (3) Special Consonant Behaviors

- **/r, rʲ/** — coronal; may partially devoice before pause (syllabic sonorant possible) [steriopolo2012]:

  | Example   | Transcription |
  | --------- | ------------- |
  | _семестр_ | `[sɛᶤˈmɛstr̥]` |

- **/ɦ/** — voiced, energetic; articulated in the lower pharynx/glottis (deeper than German [h]) [steriopolo2012].

- **/v/** → vocalic `[u̯]` in coda position and word-initially before consonants [steriopolo2012]:

  | Example | Transcription |
  | ------- | ------------- |
  | _пив_   | `[pɨu̯]`       |
  | _бив_   | `[bɨu̯]`       |
  | _впав_  | `[u̯ˈpau̯]`     |
  | _вовк_  | `[wɔu̯k]`      |

- **Partial regressive palatal assimilation** — fricative before palatalised [rʲ] may partially palatalize [steriopolo2012]:

  | Example  | Transcription |
  | -------- | ------------- |
  | _срібло_ | `[ˈsʲrʲiblɔ]` |

---

### Geminates (Doubled/Long Consonants)

Four structural sources of gemination in Ukrainian [steriopolo2012]:

| Group                                                 | Condition                               | Examples                                 | Transcription                                            |
| ----------------------------------------------------- | --------------------------------------- | ---------------------------------------- | -------------------------------------------------------- |
| (i) Morpheme boundary, identical consonants           | prefix + root or root + suffix          | _без+зубий_, _під+дати_, _причин+на_     | `[bɛᶤˈzːubɨj]`, `[pʲіˈdːatɨ]`, `[prɨˈʈʂɨnːa]`            |
| (ii) Historical progressive assimilation              | noun/verb suffix -ття, -ддя, -ссся etc. | _життя_, _погруддя_, _волосся_, _стаття_ | `[ʒɨˈtʲːa]`, `[pɔɦˈrudʲːa]`, `[vɔˈlɔsʲːa]`, `[staˈtʲːa]` |
| (iii) Word/syllable boundary, regressive assimilation | proclitic + noun; prefix у/в + root     | _над Дніпром_, _ввічливий_, _квітчати_   | `[nadːnʲiˈprɔm]`, `[ˈvʲːitʃlɨvɨi̯]`, `[kvʲiˈtʃːatɨ]`      |
| (iv) Foreign loanwords                                | orthographic double consonants          | _ванна_, _бароккo_                       | `[ˈvanːa]`, `[baˈrɔkːɔ]`                                 |

**Minimal pairs** showing phonemic status of geminates [steriopolo2012]:

| Short     | Transcription | Long       | Transcription  |
| --------- | ------------- | ---------- | -------------- |
| _причина_ | `[prɨˈʈʂɨna]` | _причинна_ | `[prɨˈʈʂɨnːa]` |
| _істина_  | `[ˈistɨna]`   | _істинна_  | `[ˈistɨnːa]`   |
| _стіна_   | `[stiˈna]`    | _стінна_   | `[stiˈnːa]`    |

> Note: Phonological status debated — long consonant may represent two underlying phonemes merged at a morpheme boundary rather than a single geminate phoneme [steriopolo2012].

---

### Vowel System

Six vowel phonemes: **/i, ɨ, ɛ, a, ɔ, u/** [steriopolo2012].

Orthographic mapping: `<і, и, е, а, о, у>`.

#### Vowel Examples

| Phoneme | Transcription | Word            | Ukrainian |
| ------- | ------------- | --------------- | --------- |
| /i/     | `dʲim`        | _дім_           | дім       |
| /ɨ/     | `dɨm`         | _дим_           | дим       |
| /ɛ/     | `sɛm`         | _сем_ (gen.pl.) | сем       |
| /a/     | `sam`         | _сам_           | сам       |
| /ɔ/     | `sɔm`         | _сом_           | сом       |
| /u/     | `sum`         | _сум_           | сум       |

#### Vowel Positions in IPA Trapezoid

- **/i, ɨ, ɛ/** — front vowels [steriopolo2012]
- **/a, ɔ, u/** — back vowels [steriopolo2012]
- **No central vowels** in the phonological system

**Key deviations from standard IPA positions** (Steriopolo's experimental data):

| Vowel    | Standard IPA position            | Ukrainian articulatory reality                                                                                                                                |
| -------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| /a/      | back open [ɑ]                    | **Fronted toward center** — should be transcribed as light /a/ not dark back [ɑ]; may be placed in a triangle rather than quadrilateral [steriopolo2012]      |
| /ɨ/      | central high [ɨ] (as in Russian) | **Front row, lowered** — articulated in the front of the oral cavity with tongue slightly retracted; front-lowered, not central-high [steriopolo2012; Totska] |
| /ɛ/, /ɔ/ | mid [e], [o]                     | More **open** than [e], [o]; transcribed as [ɛ], [ɔ] — not [e], [o] [steriopolo2012]                                                                          |

---

### Vowel Reduction Rules

Difference in quality/duration between stressed and unstressed vowels is **less pronounced than in Russian** [steriopolo2012]. Exceptions: pre-stress /ɛ, ɨ, ɔ/.

| Pre-stress vowel | Environment                                | Shift                                   | Example      | Phonemic → Phonetic               |
| ---------------- | ------------------------------------------ | --------------------------------------- | ------------ | --------------------------------- |
| /ɛ/              | before stressed syllable with /Cɨ/ or /Ci/ | → [ɨ] (regressive distant assimilation) | _вершина_    | `/vɛᶤrˈʂɨna/` → `[vɨᶹrˈʂɨna]`     |
| /ɔ/              | before stressed syllable with /Cu/         | → [u]                                   | _подушка_    | `/pɔᶹˈduʂka/` → `[puˈduʂka]`      |
| /ɨ/ (unstressed) | before stressed /Cɛ/ or /Ca/               | → [ɛ]                                   | _письменник_ | `/pɨsʲˈmɛnːɨk/` → `[pɛsʲˈmɛnːɨk]` |
| /ɨ/ (unstressed) | before stressed /Ca/                       | → [ɛ]                                   | _зима_       | `/zɨɛˈma/` → `[zɛɛˈma]`           |

> /ɛ, ɔ/ before vowels of their own row (closed) tend toward **closure**; /ɨ/ before open vowels becomes **more open** [steriopolo2012].

---

### Syllable Structure

Template: **(CCCC) V (CCCC)** — up to 4 consonants in onset and coda [steriopolo2012].

Types: open (_село_ `[sɛᶤ.ˈlɔ]`), closed (_рік_ `[rʲik]`), covered (_парта_ `[ˈpar.ta]`), uncovered (_осінь_ `[ˈɔ.sʲinʲ]`).

Ukrainian, like all Slavic languages, has a **tendency toward open syllables** [steriopolo2012].

Examples of onset/coda complexity:

| Word          | Transcription       | Onset/Coda consonants |
| ------------- | ------------------- | --------------------- |
| _за_          | `[za]`              | CV                    |
| _бра_         | `[bra]`             | CCV                   |
| _вклад_       | `[fklad]`           | CCCVC                 |
| _встрибувати_ | `[ˈvstrɨ.bu.va.tɨ]` | CCCC onset            |
| _мак_         | `[mak]`             | CVC                   |
| _старт_       | `[start̥]`           | CCVCC                 |
| _акт_         | `[akt]`             | VCC                   |
| _міністр_     | `[mʲi.ˈnʲistr̥ɛ]`    | VCCC coda             |
| _безумств_    | `[bɛᶤ.ˈᶎumstv]`     | VCCCC coda            |

---

### Syllabification Rules

Rules based on **sonority theory** (vowel = sonority peak) [steriopolo2012]:

| Rule | Condition                                                              | Boundary placement                  | Examples                                                                            |
| ---- | ---------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| 1    | Single consonant after vowel                                           | before consonant                    | _по-до-ро-жі_ `[ˈpɔ.dɔ.rɔ.ʒi]`                                                      |
| 2    | Two voiced **or** two voiceless consonants                             | both go to next syllable            | _су-джу_ `[su.ˈɖʐu]`                                                                |
| 3    | One voiced + one voiceless                                             | split between them                  | _бджіл-ка_ `[ˈbdʒil.ka]`, _гав-ка-ти_ `[ˈɦav.ka.tɨ]`, _дзюр-чати_ `[dzʲur.ˈʈʂa.tɨ]` |
| 4    | Two adjacent sonorants                                                 | split between different syllables   | _сум-ний_ `[ˈsum.nɨi̯]`                                                              |
| 5    | Geminate                                                               | split across syllable boundary      | _стат-тя_ `[staˈtʲːa]`                                                              |
| 6    | Two consonants of different sonority: sonorant first, obstruent second | sonorant goes to preceding syllable | _гай-ка_ `[ˈɦaj.ka]`, _ям-ка_ `[ˈjam.ka]`, _син-ку_ `[ˈsɨn.ku]`                     |
| 7    | Two consonants: obstruent first, sonorant second                       | both go to next syllable            | _лю-блю_ `[lʲu.ˈblʲu]`, _му-дрий_ `[ˈmu.drɨi̯]`, _по-свист_ `[ˈpɔ.svɨst]`            |

> IPA syllable boundary marker: `.` (not a hyphen) [steriopolo2012].

---

### Word Stress

- **Free and mobile** — not fixed to a particular syllable [steriopolo2012].
- **Phonemically distinctive** (minimal pairs) [steriopolo2012]:

  | Word A    | Transcription | Word B    | Transcription |
  | --------- | ------------- | --------- | ------------- |
  | _ку́ри_    | `[ˈkurɨ]`     | _кури́_    | `[kuˈrɨ]`     |
  | _пла́кати_ | `[ˈplakatɨ]`  | _плака́ти_ | `[plaˈkatɨ]`  |
  | _доро́га_  | `[dɔˈrɔɦa]`   | _дорога́_  | `[dɔrɔˈɦa]`   |

- **Shifts with inflection** [steriopolo2012]:

  | Form       | Transcription  |
  | ---------- | -------------- |
  | _сестра́_   | `[sɛᶤsˈtra]`   |
  | _сестри_   | `[ˈsɛstrɨ]`    |
  | _стіл_     | `[sʲtʲil]`     |
  | _на столі_ | `[na ͡stɔˈlʲi]` |

- **Three degrees**: stressed / secondary / unstressed [steriopolo2012].
- **Primary acoustic correlate**: "cumulative energy" (muscle tension); duration and intensity are secondary (Brovchenko) [steriopolo2012].
- IPA stress: marked **before** the stressed syllable, not above the vowel.

---

### Intonation

Suprasegmental parameters: melody (F0 contour), phrasal stress, pauses, rhythm, tempo, timbre [steriopolo2012].

**Falling contour** (спадна мелодика):

- Declarative sentences (especially final portion)
- Declarative sentences with enumeration
- WH-questions
- Vocatives
- Imperatives expressing strong emotion

**Rising contour** (висхідна мелодика):

- Yes/no questions (without question word)
- Echo questions / counter-questions
- Incomplete utterances

**Stressed syllable contour**: rising-falling or falling; melody rises on stressed syllable then falls [steriopolo2012].

**Post-stress syllable** may also lengthen in Ukrainian utterances [steriopolo2012].

**F0 contour shape**: clearly wave-like, especially in spontaneous speech. Spontaneous speech has more pauses and more frequent tonal level changes than read speech [steriopolo2012].

**Semantic centers** (most important words): realized at highest tonal level followed by falling melody + maximum intensity rise. In read text: typically at end of utterance; in spontaneous speech: can occur anywhere [steriopolo2012].

---

### IPA Transcription Conventions for Ukrainian

| Convention              | Ukrainian IPA practice                  | Note                                                                       |
| ----------------------- | --------------------------------------- | -------------------------------------------------------------------------- |
| Stress mark             | Before stressed syllable: `ˈ`           | Not above the vowel as in Ukrainian orthography                            |
| Palatalization          | Superscript: `tʲ`                       | Not apostrophe or right-above dash as in Cyrillic tradition                |
| Syllable boundary       | `.`                                     | Per IPA standard                                                           |
| Aspiration              | Absent                                  | Unlike Germanic languages                                                  |
| Final devoicing         | Absent                                  | Voiced consonants voiced in all positions                                  |
| Voiced stop in coda     | `[dub]` for _дуб_                       | Not `[dup]`                                                                |
| Voiceless before voiced | Voices: `[baskɛɨdˈbɔl]` for _баскетбол_ | /t/ → [d] before /b/                                                       |
| /r, rʲ/ before pause    | Partial devoicing: `[sɛᶤˈmɛstr̥]`        | Diacritic `̥` below                                                         |
| /ɦ/                     | Deep energetic glottal                  | Deeper than German [h]; hook on right of [h] indicates sub-glottal quality |

---

### Phonetic Transcription — Aesop's Fable "The Northwind and the Sun"

Full text transcription (read by Bukovyna speaker) [steriopolo2012]:

**Orthographic text:**

> Одного разу засперечалися Сонце і Північний вітер з приводу того, хто з них двох сильніший. Аж раптом вони помітили мандрівника, який саме проходив повз них, кутаючись у пальто. Обидва дійшли спільної думки, що той буде визнаний сильнішим, хто вимусить мандрівника зняти своє пальто. Північний вітер дув з усієї сили, але чим дужче він дув, тим щільніше загортався мандрівник у своє пальто. Врешті-решт Північний вітер перестав боротися. І тут Сонце зігріло повітря своїми привітними променями. І вже через декілька хвилин мандрівник зняв своє пальто. Отож Північний вітер вимушений був визнати, що Сонце з-поміж них двох було сильнішим.

**Phonetic transcription:**

```
ɔdˈnoɦo ˈrazu | zaspeˈreʈʂaɫˈis̪a ˈsɔnts̪e i
pʲiu̯ˈnʲitʃnʲij ˈvʲite̪r s ˈprɨwɔdu ˈtɔɦo | xtɔ z n̪x
dwɔx sɪᶤlˈnʲiʃij || aʒ ˈraptɔm | wɔˈn̪i pɔˈmʲitɪłr
mandrʲiunᵊka | jaˈktɪj ˈsame̪ prɔˈxɔd̪u pɔuz n̪x |
zafiorˈtajuʃts̪ u palˈtɔ || ɔˈbɪdwa dʲiіʃˈlˈit ˈs̪pʲilˈn̪oji
ˈdumkɪ | ʃtʃɔ ˈtɔj ˈmɔʒɛˈ ˈbutɪ ˈvɨznanɪj sɪᶤlˈnʲiʃᶤm
| xtɔ ˈvɨmusᵗ mandrʲiunᵊka ˈzʲnʲatᶤ swoˈje
palˈtɔ || pʲiu̯nˈitʃnʲij ˈvʲite̪r prᶤˈptᶤn̪u bɔˈrɔdˡbu || i tut
ˈsɔnts̪e zʲiɦˈrʲilɔ pɔˈvʲitrˈa swɔᵘjim lasˈkavɪm
prɔˈmʲinˈam | i ˈuʒe ˈtʃereˈz ˈdɛkˈilˈka xvˡiˈlɪn |
mandrʲiuˈntk zʲnˈlau palˈtɔ || a | pʲiu̯nˈitʃnʲij
ˈvʲite̪r ˈzmuʃentʲi buu ˈvɨznatɪ | ʃtʃɔ ˈsɔnts̪e z
ˈpɔmʲiʒ nix dwɔx | ˈvujavaɫɔsˈa sɪᶤlˈnʲiʃᶤm||
```

---

## Vowel and Consonant Classification (Academic Standard)

> Source: [savchenko2014]

### Phoneme Inventory (consensus)

**38 phonemes total: 6 vowel + 32 consonant** [savchenko2014].  
Note: some sources count 36–102 depending on school (Moscow vs St Petersburg traditions). Modern Ukrainian phonetists agree on 38.

---

### Vowel Classification

**Three criteria for classifying vowels** [savchenko2014]:

1. **Row** (horizontal tongue position)
2. **Height** (vertical tongue raising)
3. **Lip participation** (labialization)

#### By Row

| Row                | Phonemes      | Condition                                                       |
| ------------------ | ------------- | --------------------------------------------------------------- |
| Front (передній)   | /і/, /и/, /е/ | Main tongue mass moved forward toward hard palate               |
| Central (середній) | —             | **Does not exist in Ukrainian**                                 |
| Back (задній)      | /у/, /о/, /а/ | Tongue body retracted; back of tongue raised toward soft palate |

#### By Height — Standard (Table 1)

| Height        | Front    | Back |
| ------------- | -------- | ---- |
| High (високе) | /і/, /и/ | /у/  |
| Mid (середнє) | /е/      | /о/  |
| Low (низьке)  |          | /а/  |

#### By Height — Totska's 4-Level System (Table 2)

| Height                    | Front | Back |
| ------------------------- | ----- | ---- |
| High (високе)             | /і/   | /у/  |
| High-mid (високо-середнє) | /и/   |      |
| Mid (середнє)             | /е/   | /о/  |
| Low (низьке)              |       | /а/  |

> **Note:** /и/ occupies a unique high-mid position — higher than /е/ but lower than /і/ [savchenko2014].

#### By Lip Participation

| Type                             | Phonemes           | Description                        |
| -------------------------------- | ------------------ | ---------------------------------- |
| Labialized (лабіалізовані)       | /о/, /у/           | Lips rounded and protruded forward |
| Non-labialized (нелабіалізовані) | /е/, /и/, /і/, /а/ | Lips neutral                       |

---

### Vowel Articulation — Detailed Descriptions

Based on Totska's experimental data [savchenko2014]:

#### [а] — stressed

- **Row/height**: back, low, non-labialized
- **Mouth**: maximally open
- **Lips**: pressed to teeth; no protrusion, no spreading
- **Tongue**: visibly retracted toward pharyngeal wall
- **Velum**: closed against pharyngeal wall, blocking nasal passage
- **Articulation**: relaxed (muscles do not contract)
- **Unstressed**: at fast tempo, may shift to **central row, mid height**

#### [о] — stressed

- **Row/height**: back, mid, labialized
- **Tongue**: back part slightly raised toward soft palate, not touching
- **Mouth**: half-open
- **Lips**: rounded, tense, slightly protruded — actively participate
- **Unstressed**: narrower, shorter, less labialized; before stressed [у] or [і] shifts closer to them

#### [у] — stressed

- **Row/height**: back, high, labialized
- **Tongue**: very high; back of tongue raised to soft palate
- **Channel**: very narrow
- **Lips**: strongly protruded forward
- **Unstressed**: shifts to **central row, high height**; tongue advances, lower and wider than in stressed position

#### [е] — stressed

- **Row/height**: front, mid, non-labialized
- **Mouth**: very open
- **Tongue**: main body advanced forward toward middle of hard palate
- **Lips**: spread to sides; do not actively participate
- **Articulation**: relaxed
- **Unstressed**: rises toward [и] (before syllables with [і], [и], [у]): e.g. _несе_ → `[ниесé]`

#### [и] — stressed

- **Row/height**: front, high, non-labialized
- **Pharynx**: wider than oral cavity (distinguishing feature)
- **Tongue**: advanced forward
- **Lips**: corners spread wider than for [е], pressed to teeth
- **Height**: higher than [е], lower than [і]
- **Unstressed**: shifts to **front, mid** position

#### [і] — stressed and unstressed identical

- **Row/height**: front, high, non-labialized
- **Tongue**: advanced to front hard palate or upper teeth + alveoli; swollen shape
- **Lips**: spread significantly more than for [и] or [е]; form narrow slit
- **Unstressed**: **no change** — same as stressed position

#### Positional Stability Summary

| Vowel | Stable between stressed/unstressed? | Unstressed shift               |
| ----- | ----------------------------------- | ------------------------------ |
| [і]   | Yes (no change)                     | —                              |
| [у]   | Mostly (shifts to central row)      | central-high                   |
| [о]   | Mostly (slight narrowing)           | → [у] before stressed /у/, /і/ |
| [и]   | No                                  | → front-mid                    |
| [е]   | No                                  | → [и] before /і/, /и/, /у/     |
| [а]   | No (at fast tempo)                  | → central-mid                  |

**Table 3 — Unstressed vowel positions** [savchenko2014]:

| Height | Front  | Central | Back |
| ------ | ------ | ------- | ---- |
| High   | [і]    |         | [у]  |
| Mid    | [е, и] | [а]     | [о]  |

---

### Phoneme Realization Types in Speech

Vowel and consonant phonemes realize as three types of sounds [savchenko2014]:

| Type                     | Description                             | Example                                                                                           |
| ------------------------ | --------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Invariant** (stressed) | Phoneme realized as its canonical form  | /мито/ → `[мито]`; /и/ = stressed [и]                                                             |
| **Unstressed invariant** | Same quality, unstressed position       | /сходи/ → `[сходи]`; /и/ = unstressed [и] (stress on preceding syllable)                          |
| **Positional variant**   | Quality altered by phonetic environment | /митарства/ → `[миетарства]`; /и/ = positional variant [иe] (unstressed before stressed syllable) |

Consonant variants [savchenko2014]:

| Type                      | Example                                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Invariant**             | /пора/ → `[пора]` — each phoneme as invariant                                                                                                   |
| **Positional variant**    | /з'ійшо́вся/ → `[з'іĭшоŭс'а]` — [ĭ], [ŭ] appear word-initially before consonant, medially after vowel before consonant, word-finally after vowel |
| **Combinatorial variant** | /кізочці/ → `[к'ізоц':і]` — /к/ → [к'] by accommodation; /ч/ + /ц'/ → [ц':] by assimilation + lengthening                                       |

---

### Consonant Classification Criteria

Seven criteria for classifying consonant **phonemes** [savchenko2014]; sounds add criteria 7–8:

1. Voice/noise ratio (voicing)
2. Voice participation (voiced vs voiceless)
3. Place of articulation — active organ
4. Place of articulation — passive organ
5. Manner of articulation
6. Palatalization (hard/soft)
7. Nasal resonator participation
8. _(sound level only)_ Duration
9. _(sound level only)_ Acoustic impression

---

### Criterion 1+2: Voice/Noise Ratio

**Sonorants** (голос > шум): /м/, /н/, /н'/, /р/, /р'/, /л/, /л'/, /в/, /й/ — 9 phonemes; sounds: [м], [м'], [в], [в'], [й], [л], [л'], [н], [н'], [р], [р'] [savchenko2014]

> Sonorants: organs approximate as in obstruents, but the gap is wide enough that air passes without strong noise [Totska, cited in savchenko2014].

**Noisy consonants** (шумні): all remaining 23 phonemes.

**Voiced** (дзвінкі): noise + voice.  
**Voiceless** (глухі): noise only; requires greater airstream pressure.

#### Correlative voiced/voiceless pairs

Debated across sources [savchenko2014]:

| #   | Pair       | Note                                                                           |
| --- | ---------- | ------------------------------------------------------------------------------ |
| 1   | /б/–/п/    | correlative                                                                    |
| 2   | /д/–/т/    | correlative                                                                    |
| 3   | /д'/–/т'/  | correlative                                                                    |
| 4   | /з/–/с/    | correlative                                                                    |
| 5   | /з'/–/с'/  | correlative                                                                    |
| 6   | /ж/–/ш/    | correlative                                                                    |
| 7   | /дж/–/ч/   | correlative                                                                    |
| 8   | /ґ/–/к/    | correlative                                                                    |
| 9   | /дз/–/ц/   | correlative                                                                    |
| 10  | /дз'/–/ц'/ | correlative                                                                    |
| 11  | /г/–/х/    | **disjunctive** — differ by BOTH voicing AND place: /г/ = glottal, /х/ = velar |

> **Totska and Zhovtobryukh**: 11 correlative pairs (above). **Prokopova**: 13 pairs (adds soft pairs [б']–[п'], [дж']–[ч']). **Karpenko**: only 7 pairs (/з–с/, /ж–ш/, /дз–ц/, /дж–ч/, /б–п/, /д–т/, /ґ–к/); considers [ф], [х] and [г] to lack correlative partners. **Historical note**: /г/–/х/ were once both velar; /г/ later shifted to glottal, breaking the pair [savchenko2014].

> /ф/ has **no voiced correlate** — borrowed into Ukrainian consonantism from other languages [savchenko2014].

---

### Criterion 3: Place of Articulation — Active Organ

| Group                               | Phonemes                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Labial** (губні)                  | /б/, /п/, /в/, /м/, /ф/                                                                                                  |
| **Front-lingual** (передньоязикові) | /д/, /д'/, /т/, /т'/, /з/, /з'/, /с/, /с'/, /л/, /л'/, /н/, /н'/, /р/, /р'/, /дз/, /дз'/, /ц/, /ц'/, /ж/, /ч/, /ш/, /дж/ |
| **Mid-lingual** (середньоязикові)   | /й/                                                                                                                      |
| **Back-lingual** (задньоязикові)    | /ґ/, /к/, /х/                                                                                                            |
| **Glottal/pharyngeal** (глоткова)   | /г/ (only one)                                                                                                           |

> **Zhovtobryukh** also includes soft [д'], [т'], [л'], [н'] in mid-lingual, because their i-like raising is as high as [й] [savchenko2014].

---

### Criterion 4: Place of Articulation — Passive Organ

**Labial subclasses** [savchenko2014]:

| Type                        | Phonemes           | Description                                                                                  |
| --------------------------- | ------------------ | -------------------------------------------------------------------------------------------- |
| Labial-labial (губно-губні) | /б/, /п/, /в/, /м/ | Lower lip approaches upper lip                                                               |
| Labial-dental (губно-зубні) | /ф/                | Lower lip approaches upper teeth. Note: /в/ also labial-dental in some positions (Prokopova) |

**Front-lingual subclasses** (Totska classification) [savchenko2014]:

| Type                                         | Phonemes                                                                                 | Description                                                      |
| -------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Dental (зубні)                               | /д/, /д'/, /т/, /т'/, /з/, /з'/, /с/, /с'/, /дз/, /дз'/, /ц/, /ц'/, /л/, /л'/, /н/, /н'/ | Front tongue + tip approach inner surface of upper teeth         |
| Palatal-dental / alveolar (піднебінно-зубні) | /ж/, /ч/, /ш/, /дж/, /р/, /р'/                                                           | Tongue tip curled up to zone between teeth and front hard palate |

> **Apical vs dorsal**: when only the tongue tip contacts the passive organ → **apical** sounds; when front dorsum without tip → **dorsal** sounds. In Ukrainian, all consonants except [л], [л'], [р], [р'] are dorsal [Totska, cited in savchenko2014].

---

### Criterion 5: Manner of Articulation

| Type                          | Phonemes                                                            | Description                                                               |
| ----------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Stops** (зімкнені/проривні) | /б/, /п/, /м/, /д/, /д'/, /т/, /т'/, /н/, /н'/, /ґ/, /к/, /л/, /л'/ | Complete closure between articulators                                     |
| **Fricatives** (щілинні)      | /в/, /ф/, /с/, /с'/, /з/, /з'/, /ж/, /ш/, /г/, /х/, /й/             | Articulators approximate, forming a gap; airstream creates friction       |
| **Affricates** (африкати)     | /дз/, /дз'/, /ц/, /ц'/, /ч/, /дж/                                   | Begin with complete closure, gradually opening into a gap; no final burst |
| **Trills** (дрижачі/вібранти) | /р/, /р'/                                                           | Periodic tongue approximation/withdrawal producing rhythmic interruptions |

**Sonorant manner sub-classification** (Totska) [savchenko2014]:

| Sub-type            | Sounds              |
| ------------------- | ------------------- |
| Nasal stops         | [м], [н], [н']      |
| Fricative sonorants | [в], [й], [л], [л'] |
| Trills              | [р], [р']           |

> Note on [л], [л']: during articulation there is no complete closure — air exits through sides of tongue (lateral). Classified as lateral fricatives by Karpenko and Totska. Zhovtobryukh and Avanessov group them with sonorants as "stop-passage" (зімкнено-прохідні) [savchenko2014].

---

### Criterion 6: Palatalization

**Three palatalization degrees** (Totska) [savchenko2014]:

| Degree                                 | Sounds                      | Description                                                                            |
| -------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------- |
| **Inherently soft** (власне м'який)    | [й]                         | Middle dorsum **touches** hard palate; i-like articulation IS the primary articulation |
| **Palatalized** (пом'якшені)           | [т'], [н'], [с'], [ц'] etc. | i-like articulation is **additional** (secondary) to basic articulation                |
| **Semi-palatalized** (напівпом'якшені) | [в'], [м'], [ж'], [х'] etc. | Additional i-like articulation is **weakened**                                         |

**9 hard/soft correlative pairs** [savchenko2014]:
/д/–/д'/, /т/–/т'/, /з/–/з'/, /с/–/с'/, /дз/–/дз'/, /ц/–/ц'/, /л/–/л'/, /н/–/н'/, /р/–/р'/

**Always hard** (no soft counterpart): labials /б/, /п/, /в/, /м/, /ф/; velars /ґ/, /к/, /х/; glottal /г/; palatal-dental /ж/, /ч/, /ш/, /дж/

**Always soft**: /й/ (no hard counterpart)

> **Acoustic basis** (Totska): hard consonants have lower inherent tone; soft consonants have higher inherent tone. Tone height depends on resonator size: palatalization reduces oral resonator volume → higher tone [savchenko2014].

**Historical note**: labials and hushing consonants hardened in most positions. Hushing consonants were originally soft and had no hard counterparts [savchenko2014].

**Orfoepic dictionary** distinguishes three sub-levels of soft: м'які `[д'], [т'], [н'], [л']` / пом'якшені `[з'], [с'], [дз'], [ц']` / напівпом'якшені `[р'], [ш'], [ж'], [ч'], [дж'], [б'], [п'], [в'], [ф'], [м'], [к'], [г'], [ґ'], [х']` — but notes the difference between м'які and пом'якшені is so small that the automated dictionary does not distinguish them [savchenko2014].

---

### Criterion 7: Nasal Resonator

| Type                                  | Phonemes       | Description                           |
| ------------------------------------- | -------------- | ------------------------------------- |
| **Nasal** (носові)                    | /м/, /н/, /н'/ | Airstream passes through nasal cavity |
| **Non-nasal / pure** (неносові/чисті) | All others     | No nasal resonance                    |

> [м]: lip closure, airstream → nasal cavity; [н]: tongue tip against inner surface of upper teeth, airstream → nasal cavity [savchenko2014].

---

### Criterion 8: Duration (Sound Level Only)

| Type                        | Sounds                                                                                                                                                                                                              | Examples            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Regular (звичайної довготи) | [д], [д'], [т], [т'], [з], [з'], [с], [с'], [ц], [ц'], [дз], [дз'], [л], [л'], [н], [н'], [б], [б'], [п], [п'], [в], [в'], [ф], [ф'], [ж], [ж'], [ч], [ч'], [ш], [ш'], [дж], [дж'], [г], [г'], [к], [к'], [х], [х'] | _жар_, _жінка_      |
| Geminate (подовжені)        | [д:], [д':], [т:], [т':], [з:], [з':], [с:], [с':] etc. (all consonants with `:`)                                                                                                                                   | _збіж':а_, _бут':а_ |

> Geminate property applies to hard, soft, and semi-soft consonants alike [savchenko2014].

---

### Criterion 9: Acoustic Impression (Sound Level Only)

| Group                           | Sounds                                       | Feature                                |
| ------------------------------- | -------------------------------------------- | -------------------------------------- |
| **Sibilant-hissing** (свистячі) | [з], [з'], [с], [с'], [дз], [дз'], [ц], [ц'] | Accompanied by whistling/hissing noise |
| **Hushing** (шиплячі)           | [ж], [ш], [ч], [дж]                          | Accompanied by hushing/sizzling noise  |

---

## References

[kasyanova2015]: Касьянова, О. А. "Звукове поле фонеми /в/: комбінаторні та позиційні алофони." _Мовні і концептуальні картини світу_ 1 (2015): 324–336.

[kasyanova2018]: Касьянова, О. А. "Артикуляційна та акустична характеристика звукових реалізацій української фонеми /m/ (за матеріалами експериментально-фонетичного дослідження)." _Science and Education a New Dimension. Philology_, VI(53), Issue 182 (2018): 18–21. https://doi.org/10.31174/SEND-Ph2018-182VI53-04

[steriopolo2012]: Стеріополо, Олена. "Українська фонетична система у парадигмі міжнародної фонетичної асоціації (МФА)." _Науковий вісник Ужгородського університету. Серія: Філологія. Соціальні комунікації_ 27 (2012): 51–58.

[savchenko2014]: Савченко, І. С. _Фонетика, орфоепія і графіка сучасної української мови: навч. посіб._ (2014).
