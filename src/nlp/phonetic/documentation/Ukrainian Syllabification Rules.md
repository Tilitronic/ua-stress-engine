# Ukrainian Syllabification Rules

All rules and examples extracted from academic sources.
Use for syllabifier implementation and tests. Each rule cites its source.

---

## Table of Contents

- [Syllable Basics](#syllable-basics)
- [Theories of the Syllable](#theories-of-the-syllable)
- [Sonority Scale](#sonority-scale)
- [Three Tendencies in Ukrainian Syllabification](#three-tendencies-in-ukrainian-syllabification)
- [Syllabification Rules — Phonological Level](#syllabification-rules--phonological-level)
- [Syllabification Rules — Phonetic Level](#syllabification-rules--phonetic-level)
- [Phonological vs Phonetic Divergence](#phonological-vs-phonetic-divergence)
- [Geminates at Syllable Boundary](#geminates-at-syllable-boundary)
- [Morpheme Boundary vs Phonetic Syllable](#morpheme-boundary-vs-phonetic-syllable)
- [Quick Reference — Steriopolo 7-Rule Summary](#quick-reference--steriopolo-7-rule-summary)
- [References](#references)

---

## Syllable Basics

**Syllable** = minimum unit of both phonetic and phonological levels [savchenko2014].

- A word has as many syllables as it has vowels: _ко-ро-на_ (3 vowels → 3 syllables)
- Syllable types: open (кінцевий = голосний), closed (кінцевий = приголосний), covered (початок = приголосний), uncovered (початок = голосний)
- Ukrainian tends strongly toward **open syllables** — ~78% of syllables are open [savchenko2014]

**Syllable structure template**: (CCCC) V (CCCC) — up to 4 consonants in onset or coda [steriopolo2012]

**IPA syllable boundary marker**: `.` (dot, not hyphen) [steriopolo2012]

### Syllable Count Examples

| Word                  | Syllabification | Count |
| --------------------- | --------------- | ----- |
| _мир, плач, дзвін_    | mono            | 1     |
| _ша-фа, до-ля_        | disyllabic      | 2     |
| _ко-ро-вай, до-ро-га_ | trisyllabic     | 3     |
| _пе-ре-мог-ти_        | polysyllabic    | 4     |

---

## Theories of the Syllable

Three phonetic theories describe the syllable [savchenko2014]:

### 1. Expiratory Theory (Stetson)

> **Syllable** = sound or combination of sounds produced by one push of exhaled air.

**Problem**: both the interjection _ау_ and the sound-cluster _тс_ are produced by one push, but only _ау_ is a syllable. Cannot account for syllabification of consonant clusters [savchenko2014].

_(Still found in school textbooks, as it was long the standard in Ukrainian phonetics.)_

### 2. Muscular Tension Theory (Grammont, Shcherba)

> **Syllable** = articulatorily indivisible unit produced by one impulse of muscular tension.

Speech = chain of impulses. Each consonant has three forms [savchenko2014]:

| Form                                   | Description                                                         | Example                       |
| -------------------------------------- | ------------------------------------------------------------------- | ----------------------------- |
| **Strong-ending** (сильнокінцева)      | End of consonant stronger than beginning → forms syllable **onset** | _дім_ → [d] is onset          |
| **Strong-beginning** (сильнопочаткова) | Beginning stronger than end → forms syllable **coda**               | _пар-там_ → [r] is coda       |
| **Double-peak** (двовершинна)          | Beginning and end equally strong → stands on syllable **boundary**  | _жит-тя_ → [t] is on boundary |

### 3. Sonority / Acoustic Theory (Totska)

> **Syllable** = sequence of sounds with increasing sonority from start to peak.

Sonority levels (Totska scale) [savchenko2014]:

| Level | Sound type                            |
| ----- | ------------------------------------- |
| 4     | Vowels (most sonorous)                |
| 3     | Sonorant consonants                   |
| 2     | Voiced obstruents                     |
| 1     | Voiceless obstruents (least sonorous) |

**Sonority-based syllabification examples** (Totska) [savchenko2014]:

| Word       | Sonority profile | Syllabification |
| ---------- | ---------------- | --------------- |
| _пі-шла_   | 14 — 134         | `[pʲi.ʃla]`     |
| _каз-ка_   | 142 — 14         | `[kaz.ka]`      |
| _ком-байн_ | 143 — 2433       | `[kɔm.bain]`    |
| _пар-та_   | 143 — 14         | `[par.ta]`      |

---

## Sonority Scale

**Jespersen's sonority scale** (cited in Semchynsky 1986 [savchenko2014]):

| Value | Sound type           | Examples                                                         |
| ----- | -------------------- | ---------------------------------------------------------------- |
| 0     | Voiceless stops      | [п], [т], [к], [к'], [т'], [д'] — wait, voiceless: [п], [т], [к] |
| 1     | Voiced stops         | [б], [д], [ґ]                                                    |
| 2     | Voiceless fricatives | [ф], [с], [х], [ш]                                               |
| 3     | Voiced fricatives    | [в], [з], [ж], [г]                                               |
| 4     | Nasals               | [м], [н], [н']                                                   |
| 5     | Laterals             | [л], [л']                                                        |
| 6     | Trills               | [р], [р']                                                        |
| 7     | High vowels          | [і], [и], [у]                                                    |
| 8     | Mid vowels           | [е], [о]                                                         |
| 9     | Low vowels           | [а]                                                              |

**Principle**: syllable boundary falls at a **sonority valley** — the point between two sounds where sonority drops to a local minimum.

**Numeric examples** [savchenko2014]:

| Phonological | Numerics | Phonetic   | Numerics |
| ------------ | -------- | ---------- | -------- |
| /дру́ж-ба/    | 1673-19  | [ро́з-д'іл] | 683-175  |
| /біг-це́м/    | 173-084  | [йіж-джу]  | 573-07   |
| /ко́б-зар/    | 081-396  | [нуд'-га́]  | 471-39   |

---

## Three Tendencies in Ukrainian Syllabification

[savchenko2014]:

| #   | Tendency                     | Description                                                                            | Example                                 |
| --- | ---------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------- |
| 1   | **Open syllable attraction** | Consonants are pulled toward the following vowel (historical: old open syllable law)   | _до-до-му_                              |
| 2   | **Rising sonority**          | Within a syllable, sounds arranged from less to more sonorous                          | _пі-сня_ (14-134)                       |
| 3   | **Euphony**                  | Avoidance of consonant clusters within one syllable; avoidance of sharp sonority drops | Ukrainian _чесний_ vs Russian _честный_ |

---

## Syllabification Rules — Phonological Level

Rules operate on underlying phoneme sequences. Written in `/phonemic/` notation [savchenko2014].

---

### Rule 1: Single Intervocalic Consonant

> **Always goes to the following syllable.**

| Phonological | Syllabification |
| ------------ | --------------- |
| /садиба/     | /са-ди-ба/      |
| /дерево/     | /де-ре-во/      |
| /чужина/     | /чу-жи-на/      |
| /подарунок/  | /по-да-ру-нок/  |

---

### Rule 2: Two Intervocalic Consonants

#### 2a — Both voiceless obstruents → both go to next syllable

| Phonological | Syllabification |
| ------------ | --------------- |
| /густо/      | /гу-сто/        |
| /тещча/      | /те-шча/        |
| /ліхтар/     | /л'і-хтар/      |
| /ряска/      | /р'я-ска/       |
| /шапка/      | /ша-пка/        |
| /крихта/     | /кри-хта/       |

#### 2b — Both voiced obstruents, same manner of articulation → both go to next syllable

| Phonological | Syllabification |
| ------------ | --------------- |
| /придбати/   | /при-дба-ти/    |

#### 2c — Obstruent + sonorant → both go to next syllable

| Phonological | Syllabification |
| ------------ | --------------- |
| /хутро/      | /ху-тро/        |
| /шабля/      | /ша-бл'а/       |
| /срібло/     | /ср'і-бло/      |
| /скісний/    | /скі-сний/      |

---

#### Split cases — boundary between the two consonants:

#### 2α — Both sonorants → split between them

| Phonological | Syllabification |
| ------------ | --------------- |
| /горло/      | /гор-ло/        |
| /гривня/     | /грив-н'а/      |
| /перли/      | /пер-ли/        |
| /пильно/     | /пил'-но/       |

#### 2β — Sonorant first, any consonant second → sonorant stays in preceding syllable

| Phonological | Syllabification |
| ------------ | --------------- |
| /скл'анка/   | /скл'ан-ка/     |
| /ставки/     | /став-ки/       |
| /чайка/      | /чай-ка/        |
| /рейдер/     | /рей-дер/       |
| /ломбард/    | /лом-бард/      |
| /куранти/    | /ку-ран-ти/     |

#### 2γ — Voiced obstruent first, voiceless second → split between them

| Phonological | Syllabification |
| ------------ | --------------- |
| /казка/      | /каз-ка/        |
| /шведка/     | /швед-ка/       |
| /дьогту/     | /д'ог-ту/       |

#### 2δ — Voiced fricative + voiced stop or affricate (or reverse) → split

| Phonological | Syllabification |
| ------------ | --------------- |
| /кузбас/     | /куз-бас/       |
| /дружба/     | /друж-ба/       |
| /обжати/     | /об-жа-ти/      |

---

### Rule 3: Three or More Intervocalic Consonants

#### 3a — First consonant is sonorant → stays in preceding syllable; rest go to next

| Phonological | Syllabification |
| ------------ | --------------- |
| /лінгвіст/   | /л'ін-гвіст/    |
| /портфель/   | /пор-тфел'/     |
| /тембровий/  | /тем-бро-вий/   |
| /шаленство/  | /ша-лен-ство/   |

#### 3b — Obstruents precede a sonorant → all go to next syllable

| Phonological | Syllabification |
| ------------ | --------------- |
| /пискля/     | /пи-скл'а/      |
| /постріл/    | /по-стр'іл/     |
| /кравецтво/  | /кра-ве-цтво/   |

#### 3c — All voiceless obstruents → all go to next syllable

| Phonological      | Syllabification      |
| ----------------- | -------------------- |
| /хустка/          | /ху-стка/            |
| /культуристський/ | /кул'-ту-ри-стс'кий/ |

#### 3d — Voiced obstruent + one or two voiceless + sonorant → boundary after voiced

| Phonological | Syllabification |
| ------------ | --------------- |
| /розплата/   | /роз-пла-та/    |
| /розспів/    | /роз-спів/      |

---

## Syllabification Rules — Phonetic Level

Rules operate on realized sounds. Written in `[phonetic]` notation [savchenko2014].

Many rules parallel the phonological rules, but applied after assimilation, voicing changes, and gemination.

### Rule 1: Single Intervocalic Consonant

> Always to following syllable.

| Phonetic             | Syllabification      |
| -------------------- | -------------------- |
| `[лата-т':а]`        | `[ла-та-т':а]`       |
| `[воло-с':а]`        | `[во-ло-с':а]`       |
| `[беи-з:а-ко-н:иек]` | `[беи-з:а-ко-н:иек]` |

### Rule 2: Two Consonants

#### 2a — Both voiceless → both to next syllable (phonetic)

| Phonetic        | Syllabification  |
| --------------- | ---------------- |
| `[роспиета-ти]` | `[ро-спиета-ти]` |
| `[лехко]`       | `[ле-хко]`       |
| `[ростули-ти]`  | `[ро-сту-ли-ти]` |
| `[н'іхт'і]`     | `[н'і-хт'і]`     |
| `[шиешпи-на]`   | `[шие-шпи-на]`   |

#### 2b — Both voiced obstruents, same manner → both to next (phonetic)

| Phonetic      | Syllabification |
| ------------- | --------------- |
| `[боро-д'ба]` | `[бо-ро-д'ба]`  |
| `[фудбол]`    | `[фу-дбол]`     |
| `[л'ізгосп]`  | `[л'і-згосп]`   |

#### 2c — Obstruent + sonorant → both to next (phonetic)

| Phonetic    | Syllabification |
| ----------- | --------------- |
| `[ша-шлик]` | `[ша-шлик]`     |
| `[о-браз]`  | `[о-браз]`      |

#### 2α — Two sonorants / sonorant + non-syllabic → split (phonetic)

| Phonetic            | Syllabification     |
| ------------------- | ------------------- |
| `[диў-но]`          | `[диў-но]`          |
| `[мр'іĭ-ниек]`      | `[мр'іĭ-ниек]`      |
| `[ку-п'іў-л'а]`     | `[ку-п'іў-л'а]`     |
| `[йу-веи-л'ір-ниĭ]` | `[йу-веи-л'ір-ниĭ]` |
| `[сум-но]`          | `[сум-но]`          |
| `[зеим-л'а]`        | `[зеим-л'а]`        |

#### 2β — Sonorant/non-syllabic first → stays in preceding syllable (phonetic)

| Phonetic     | Syllabification |
| ------------ | --------------- |
| `[крейда]`   | `[крей-да]`     |
| `[граўс'а]`  | `[граў-с'а]`    |
| `[гончар]`   | `[гон-чар]`     |
| `[воўки]`    | `[воў-ки]`      |
| `[гоĭдалка]` | `[гоĭ-дал-ка]`  |

#### 2γ — Voiced first, voiceless second → split (phonetic)

| Phonetic   | Syllabification |
| ---------- | --------------- |
| `[л'іжко]` | `[л'іж-ко]`     |
| `[р'ідко]` | `[р'ід-ко]`     |
| `[обшук]`  | `[об-шук]`      |

#### 2δ — Voiced fricative + voiced stop/affricate (or reverse) → split (phonetic)

| Phonetic     | Syllabification |
| ------------ | --------------- |
| `[оз'-де]`   | `[оз'-де]`      |
| `[проз'-ба]` | `[проз'-ба]`    |
| `[руґзак]`   | `[ру-ґзак]`     |
| `[йаґже]`    | `[йа-ґже]`      |

### Rule 3: Three or More (phonetic)

#### 3a — Sonorant or non-syllabic first → stays in preceding syllable

| Phonetic        | Syllabification   |
| --------------- | ----------------- |
| `[тоўсто]`      | `[тоў-сто]`       |
| `[скул'птор]`   | `[скул'-птор]`    |
| `[хут'ірс'киĭ]` | `[ху-т'ір-с'киĭ]` |
| `[циеганс'киĭ]` | `[циеган-с'киĭ]`  |
| `[кримс'киĭ]`   | `[крим-с'киĭ]`    |

#### 3b — Obstruents preceding sonorant → all to next syllable

| Phonetic        | Syllabification   |
| --------------- | ----------------- |
| `[актриса]`     | `[а-ктри-са]`     |
| `[оркеистрант]` | `[ор-кеи-странт]` |
| `[ткацтво]`     | `[тка-цтво]`      |

#### 3c — All voiceless obstruents → next syllable

| Phonetic      | Syllabification |
| ------------- | --------------- |
| `[стажистка]` | `[ста-жи-стка]` |

#### 3d — Voiced + voiceless(es) + sonorant → boundary after voiced

| Phonetic         | Syllabification     |
| ---------------- | ------------------- |
| `[ростр'іл'ати]` | `[ро-стр'і-л'а-ти]` |
| `[розстрочка]`   | `[роз-стро-чка]`    |

---

## Phonological vs Phonetic Divergence

Rules at the phonological and phonetic levels mostly align, but diverge in several cases [savchenko2014]:

### Assimilation-induced divergence

| Phenomenon             | Phonological   | Phonetic                                                        |
| ---------------------- | -------------- | --------------------------------------------------------------- |
| Voicing assimilation   | /бо-рот'-ба/   | `[бо-ро-д'ба]` — [т'] voiced to [д'] → shifts syllable boundary |
| Devoicing assimilation | /роз-ка-за-ти/ | `[ро-ска-за-ти]` — [з] devoiced to [с] → boundary shifts        |
| Place + devoicing      | /вог-ко/       | `[во-хко]` — [г] → [х] → obstruent + voiceless → both to next   |

### Voiced obstruent manner sub-rule

When choosing which of two voiced obstruents stays in preceding syllable [savchenko2014]:

- Keep the **voiced fricative** in preceding syllable, send the **voiced stop or affricate** to next syllable (voiced fricative is more sonorous than voiced stop; see Jespersen scale: fricative = 3, stop = 1)
- And conversely (reverse order also applies)

| Example  | Rule application                                         | Syllabification |
| -------- | -------------------------------------------------------- | --------------- |
| /дружба/ | voiced fricative [ж] stays, voiced stop [б] goes to next | /друж-ба/       |
| /бігцем/ | voiced stop [г] stays, voiced affricate [ц] goes to next | /біг-цем/       |
| /кобзар/ | voiced stop [б] stays, voiced fricative [з] goes to next | /коб-зар/       |

---

## Geminates at Syllable Boundary

### Phonological level

For same sonorant geminates: split between them [savchenko2014]:

- /тонна/ → `/тон-на/`
- /сіллю/ → `/сіл'-л'у/`

For same voiceless obstruent geminates: potentially ambiguous, but split is possible:

- /над дорогою/ → `/над-до-ро-го-йу/` (both /д/ are voiced → either /над-до-.../ or /на-ддо-.../)
- /віссю/ → `/від-с'с'у/` — both /с'/ are voiceless → `/ві-с'с'у/`

### Phonetic level

Two identical phonemes merge into **one geminate sound** in intervocalic position → the geminate goes to the **following syllable** as a unit [savchenko2014]:

| Phonological | Phonetic    | Syllabification |
| ------------ | ----------- | --------------- |
| /життя/      | `[жиет':а]` | `[жие-т':а]`    |
| /піддон/     | `[п'ід:он]` | `[п'і-д:он]`    |

Contrast with non-geminate (for testing):

| Phonological | Phonetic   | Syllabification |
| ------------ | ---------- | --------------- |
| /жити/       | `[жити]`   | `[жи-ти]`       |
| /бідон/      | `[б'ідон]` | `[б'і-дон]`     |

### Shcherba's double-peak rule

Long consonants are "double-peak" — equally strong at both ends, weakened in the middle — and therefore stand **at the syllable boundary** [Shcherba, cited in savchenko2014]. This justifies splitting them: _об-бити_, _ніч-чю_, _зіл-ля_, _насін-ня_.

However, at fast speech rate, long consonants become strong-ending and attach to the following syllable (like _бе-ссада_ / _без сада_ in Russian). The same applies in Ukrainian [savchenko2014].

---

## Morpheme Boundary vs Phonetic Syllable

### Prefix-root boundary

At morpheme boundaries, two syllabification variants are both acceptable [savchenko2014]:

| Word        | Variant 1    | Variant 2    |
| ----------- | ------------ | ------------ |
| _безdonний_ | _без-донний_ | _бе-здонний_ |
| _відвів_    | _від-вів_    | _ві-двів_    |
| _розбити_   | _роз-бити_   | _ро-збити_   |

### Connected speech — phonetic syllable dominates

In the phonetic stream, syllable boundaries ignore morpheme and even word boundaries [savchenko2014]:

| Phrase                 | Phonetic syllabification |
| ---------------------- | ------------------------ |
| _побіг у степ_         | _по-бі-гу-степ_          |
| _спогад у пам'яті_     | _спо-га-ду-пам'-я-ті_    |
| _між лісом і струмком_ | _мі-жлі-со-мі-струм-ком_ |

---

## Quick Reference — Steriopolo 7-Rule Summary

Simplified rule set from [steriopolo2012]:

| Rule | Condition                              | Placement                      | Examples                                                                            |
| ---- | -------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------- |
| 1    | Single consonant after vowel           | Before consonant               | _по-до-ро-жі_ `[ˈpɔ.dɔ.rɔ.ʒi]`                                                      |
| 2    | Two voiced or two voiceless consonants | Both to next syllable          | _су-джу_ `[su.ˈɖʐu]`                                                                |
| 3    | One voiced + one voiceless             | Split between them             | _бджіл-ка_ `[ˈbdʒil.ka]`, _гав-ка-ти_ `[ˈɦav.ka.tɨ]`, _дзюр-чати_ `[dzʲur.ˈʈʂa.tɨ]` |
| 4    | Two adjacent sonorants                 | Split between them             | _сум-ний_ `[ˈsum.nɨi̯]`                                                              |
| 5    | Geminate                               | Split across syllable          | _стат-тя_ `[staˈtʲːa]`                                                              |
| 6    | Sonorant + obstruent                   | Sonorant to preceding syllable | _гай-ка_ `[ˈɦaj.ka]`, _ям-ка_ `[ˈjam.ka]`, _син-ку_ `[ˈsɨn.ku]`                     |
| 7    | Obstruent + sonorant                   | Both to next syllable          | _лю-блю_ `[lʲu.ˈblʲu]`, _му-дрий_ `[ˈmu.drɨi̯]`, _по-свист_ `[ˈpɔ.svɨst]`            |

---

## References

[steriopolo2012]: Стеріополо, Олена. "Українська фонетична система у парадигмі міжнародної фонетичної асоціації (МФА)." _Науковий вісник Ужгородського університету. Серія: Філологія. Соціальні комунікації_ 27 (2012): 51–58.

[savchenko2014]: Савченко, І. С. _Фонетика, орфоепія і графіка сучасної української мови: навч. посіб._ (2014).
