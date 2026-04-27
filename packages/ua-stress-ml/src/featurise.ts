/**
 * featurise.ts
 * ============
 * TypeScript port of the Python feature pipeline:
 *   feature_service.py      → build_features_v13()   (100 base features)
 *   feature_service_universal.py → build_features_universal()  (+32 → 132 total)
 *
 * All hash functions and numeric encodings are bit-for-bit identical to the
 * Python originals so that the ONNX model receives exactly the same input
 * as during training.
 *
 * Morphological features (morph_case, morph_gender, morph_number,
 * morph_tense, morph_n_features) default to -1 / 0, which is correct for
 * OOV words — the same default used in training when pymorphy3 had no parse.
 */

// ── Vowel inventory (matches Python VOWELS string exactly) ───────────────────
const VOWELS = "аеєиіїоуюя";
const VOWEL_SET = new Set(VOWELS.split(""));

/** Maps vowel character → integer index 0–9. */
const VOWEL_CHAR_MAP: ReadonlyMap<string, number> = new Map(
  VOWELS.split("").map((c, i) => [c, i] as [string, number]),
);

// ── POS encoding ─────────────────────────────────────────────────────────────
const POS_INT: ReadonlyMap<string, number> = new Map([
  ["NOUN", 0],
  ["VERB", 1],
  ["ADJ", 2],
  ["ADV", 3],
  ["NUM", 4],
  ["PRON", 5],
  ["DET", 6],
  ["PART", 7],
  ["CONJ", 8],
  ["ADP", 9],
  ["INTJ", 10],
  ["X", 11],
]);
const POS_INT_UNKNOWN = POS_INT.size; // 12

// ── Syllable onset pattern codes ─────────────────────────────────────────────
const PATTERN_INT: ReadonlyMap<string, number> = new Map([
  ["V", 0],
  ["CV", 1],
  ["CCV", 2],
  ["CCCV", 3],
  ["CCCCV", 4],
  ["no_vowel", 5],
]);

// ── Suffix / prefix lists (must match Python constants.py exactly) ───────────
const MASC_STRESS_SUFFIXES = [
  "ак",
  "як",
  "аль",
  "ань",
  "ач",
  "ій",
  "іж",
  "чук",
  "ун",
  "няк",
  "усь",
  "ар",
  "яр",
  "іст",
  "ист",
] as const;
const FOREIGN_FINAL_SUFFIXES = ["ист", "іст", "ізм", "ант", "ент"] as const;
const GREEK_INTERFIX_SUFFIXES = ["лог", "граф", "фон", "скоп", "метр"] as const;
const DEVERBAL_SUFFIXES = ["ання", "ення", "іння"] as const;
const MEASURE_SUFFIXES = ["метр", "грам", "літр"] as const;
const DIMINUTIVE_ADJ = ["еньк", "есеньк", "юсіньк"] as const;
const ROOT_STRESS_ADJ = [
  "лив",
  "аст",
  "ист",
  "ев",
  "ав",
  "ів",
  "зьк",
  "цьк",
] as const;
const COMPOUND_INTERFIXES = new Set(["о", "е", "є", "і"]);
const COMMON_PREFIXES = [
  "без",
  "від",
  "до",
  "з",
  "за",
  "на",
  "над",
  "не",
  "об",
  "пере",
  "перед",
  "під",
  "по",
  "при",
  "про",
  "роз",
  "ви",
] as const;
const OXYTONE_MOBILE_SUFFIXES = [
  "ак",
  "як",
  "ар",
  "яр",
  "ач",
  "ун",
  "няк",
  "аль",
  "ань",
] as const;
const PENULT_STABLE_SUFFIXES = ["ість"] as const;
const ADCAT_SUFFIXES = ["зький", "цький", "ський"] as const;

// Universal-model suffix lists
const VERB_ATY = ["ати"] as const;
const VERB_YTY = ["ити"] as const;
const VERB_UVATY = ["увати", "ювати"] as const;
const NUM_ADTSYAT = ["адцять"] as const;
const FOREIGN_IST = ["іст", "ист", "ізм", "ант", "ент"] as const;
const GREEK_LOG = ["лог", "граф", "фон", "скоп", "метр"] as const;
const PENULT_STABLE_U = ["ість", "ання", "ення", "іння"] as const;

// ── Low-level helpers ─────────────────────────────────────────────────────────

/**
 * DJB2 hash (Bernstein 1991) reduced modulo `mod`.
 * Matches the Python implementation exactly — 32-bit unsigned arithmetic.
 */
export function djb2Hash(s: string, mod: number): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) {
    // (h << 5) + h === h * 33; keep to 32-bit unsigned
    h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  }
  return h % mod;
}

/** Returns 0-based character positions of every Ukrainian vowel in `word`. */
export function findVowels(word: string): number[] {
  const result: number[] = [];
  for (let i = 0; i < word.length; i++) {
    if (VOWEL_SET.has(word[i])) result.push(i);
  }
  return result;
}

function endsWith(s: string, suffixes: readonly string[]): boolean {
  return suffixes.some((suf) => s.endsWith(suf));
}

function syllableOnsetPattern(lower: string): number {
  const vowels = findVowels(lower);
  if (vowels.length === 0) return PATTERN_INT.get("no_vowel")!;
  const fv = vowels[0];
  let pat = "";
  for (let i = 0; i <= fv; i++) {
    pat += VOWEL_SET.has(lower[i]) ? "V" : "C";
  }
  return PATTERN_INT.get(pat) ?? 6;
}

function lastSyllableOpen(lower: string, vowels: number[]): number {
  if (vowels.length === 0) return 0;
  return vowels[vowels.length - 1] === lower.length - 1 ? 1 : 0;
}

function maxConsonantCluster(lower: string): number {
  let maxRun = 0,
    curRun = 0;
  for (let i = 0; i < lower.length; i++) {
    const c = lower[i];
    if (!VOWEL_SET.has(c) && /\p{L}/u.test(c)) {
      curRun++;
      if (curRun > maxRun) maxRun = curRun;
    } else {
      curRun = 0;
    }
  }
  return maxRun;
}

function countOpenSyllables(lower: string, vowels: number[]): number {
  let count = 0;
  const wl = lower.length;
  for (let i = 0; i < vowels.length; i++) {
    const vPos = vowels[i];
    if (vPos === wl - 1) {
      count++;
    } else if (i + 1 < vowels.length) {
      const gap = lower.slice(vPos + 1, vowels[i + 1]);
      if (gap.length <= 1) count++;
    }
  }
  return count;
}

function cvShape(lower: string): string {
  let s = "";
  for (let i = 0; i < lower.length; i++) {
    const c = lower[i];
    if (/\p{L}/u.test(c)) s += VOWEL_SET.has(c) ? "V" : "C";
  }
  return s;
}

function vowelContextHash(
  lower: string,
  vowelPos: number,
  mod: number,
): number {
  const before = vowelPos > 0 ? lower[vowelPos - 1] : "^";
  const after = vowelPos + 1 < lower.length ? lower[vowelPos + 1] : "$";
  return djb2Hash(before + lower[vowelPos] + after, mod);
}

function vowelWideContextHash(
  lower: string,
  vp: number,
  wl: number,
  mod: number,
): number {
  const chars: string[] = [];
  for (const off of [-2, -1, 0, 1, 2]) {
    const p = vp + off;
    chars.push(p < 0 ? "^" : p >= wl ? "$" : lower[p]);
  }
  return djb2Hash(chars.join(""), mod);
}

function syllableWeight(
  lower: string,
  vowels: number[],
  sylIdx: number,
): number {
  if (vowels.length === 0 || sylIdx >= vowels.length) return 0;
  const vPos = vowels[sylIdx];
  const end = sylIdx + 1 < vowels.length ? vowels[sylIdx + 1] : lower.length;
  let coda = 0;
  for (let i = vPos + 1; i < end; i++) {
    if (!VOWEL_SET.has(lower[i]) && /\p{L}/u.test(lower[i])) coda++;
  }
  return coda === 0 ? 0 : coda === 1 ? 1 : 2;
}

function detectCompoundInterfix(
  lower: string,
  vowels: number[],
): [number, number] {
  const wl = lower.length;
  if (wl < 6 || vowels.length < 3) return [0, 0];
  for (let i = 2; i < wl - 2; i++) {
    if (COMPOUND_INTERFIXES.has(lower[i])) {
      const prevOk =
        i > 0 && !VOWEL_SET.has(lower[i - 1]) && /\p{L}/u.test(lower[i - 1]);
      const nextOk =
        i + 1 < wl &&
        !VOWEL_SET.has(lower[i + 1]) &&
        /\p{L}/u.test(lower[i + 1]);
      if (prevOk && nextOk) {
        const ratio = i / wl;
        if (ratio > 0.2 && ratio < 0.7) return [1, Math.round(ratio * 100)];
      }
    }
  }
  return [0, 0];
}

function countPrefixMatches(lower: string): [number, number] {
  let maxLen = 0,
    count = 0;
  for (const pre of COMMON_PREFIXES) {
    if (lower.startsWith(pre)) {
      count++;
      if (pre.length > maxLen) maxLen = pre.length;
    }
  }
  return [count, maxLen];
}

function variance(arr: number[]): number {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((s, x) => s + (x - mean) ** 2, 0) / arr.length;
}

// ── Base feature builder (100 features — v1.3 compatible) ───────────────────

function buildFeaturesV13(form: string, pos: string): Record<string, number> {
  const lower = form.toLowerCase();
  const vowels = findVowels(lower);
  const wl = lower.length;
  const ns = vowels.length;

  const fv = vowels.length > 0 ? vowels[0] : 0;
  const lv = vowels.length > 0 ? vowels[vowels.length - 1] : 0;
  const distFirst = vowels.length > 0 ? fv : 0;
  const distLast = vowels.length > 0 ? wl - lv - 1 : 0;
  const fvr = wl > 0 ? fv / wl : 0;
  const lvr = wl > 0 ? lv / wl : 0;

  const vr: number[] = [-1, -1, -1, -1, -1];
  for (let i = 0; i < Math.min(ns, 5); i++) vr[i] = wl > 0 ? vowels[i] / wl : 0;
  const vs = wl > 0 && ns >= 2 ? (lv - fv) / wl : 0;
  const mvr = wl > 0 && vowels.length > 0 ? vowels[Math.floor(ns / 2)] / wl : 0;

  const penultR = wl > 0 && ns >= 2 ? vowels[ns - 2] / wl : -1;
  const antepenR = wl > 0 && ns >= 3 ? vowels[ns - 3] / wl : -1;

  const iv: number[] = [-1, -1, -1, -1];
  for (let i = 0; i < Math.min(ns - 1, 4); i++) {
    iv[i] = wl > 0 ? (vowels[i + 1] - vowels[i]) / wl : 0;
  }

  const suf2 = wl >= 2 ? lower.slice(-2) : lower;
  const suf3 = wl >= 3 ? lower.slice(-3) : lower;
  const suf4 = wl >= 4 ? lower.slice(-4) : lower;
  const suf5 = wl >= 5 ? lower.slice(-5) : lower;
  const suf6 = wl >= 6 ? lower.slice(-6) : lower;
  const pre2 = wl >= 2 ? lower.slice(0, 2) : lower;
  const pre3 = wl >= 3 ? lower.slice(0, 3) : lower;
  const pre4 = wl >= 4 ? lower.slice(0, 4) : lower;

  const bplv = lv >= 2 ? djb2Hash(lower.slice(lv - 2, lv), 512) : 0;
  const bpfv =
    vowels.length > 0 && fv + 2 < wl
      ? djb2Hash(lower.slice(fv + 1, fv + 3), 512)
      : 0;

  const maxCl = maxConsonantCluster(lower);
  const nOpen = vowels.length > 0 ? countOpenSyllables(lower, vowels) : 0;
  const lastSylOpenV = lastSyllableOpen(lower, vowels);

  const s4c = wl >= 4 ? lower.slice(-4) : lower;
  const s4vc = Array.from(s4c).filter((c) => VOWEL_SET.has(c)).length;

  const hasMasc =
    pos === "NOUN" && endsWith(lower, MASC_STRESS_SUFFIXES) ? 1 : 0;
  const hasForeign = endsWith(lower, FOREIGN_FINAL_SUFFIXES) ? 1 : 0;
  const hasGreek = endsWith(lower, GREEK_INTERFIX_SUFFIXES) ? 1 : 0;
  const hasVyN = lower.startsWith("ви") && pos === "NOUN" ? 1 : 0;
  const hasDeverbal = endsWith(lower, DEVERBAL_SUFFIXES) ? 1 : 0;
  const hasMeasure = endsWith(lower, MEASURE_SUFFIXES) ? 1 : 0;
  const hasDimAdj =
    pos === "ADJ" && DIMINUTIVE_ADJ.some((s) => lower.includes(s)) ? 1 : 0;
  const hasRootAdj =
    pos === "ADJ" && ROOT_STRESS_ADJ.some((s) => lower.includes(s)) ? 1 : 0;
  const isInf = pos === "VERB" && endsWith(lower, ["ти", "сти", "зти"]) ? 1 : 0;
  const isV1p2p =
    pos === "VERB" && endsWith(lower, ["емо", "имо", "ємо", "ете", "ите"])
      ? 1
      : 0;
  const hasVyV = lower.startsWith("ви") && pos === "VERB" ? 1 : 0;
  const isRefl = endsWith(lower, ["ся", "сь"]) ? 1 : 0;
  const isBisyl = ns === 2 ? 1 : 0;
  const stressFirst =
    lower.startsWith("ви") && (pos === "NOUN" || pos === "VERB") ? 1 : 0;

  const hasOxytone = endsWith(lower, OXYTONE_MOBILE_SUFFIXES) ? 1 : 0;
  const hasPenult = endsWith(lower, PENULT_STABLE_SUFFIXES) ? 1 : 0;
  const hasAdcat = pos === "ADJ" && endsWith(lower, ADCAT_SUFFIXES) ? 1 : 0;

  const vch: number[] = [-1, -1, -1, -1, -1];
  for (let i = 0; i < Math.min(ns, 5); i++) {
    vch[i] = VOWEL_CHAR_MAP.get(lower[vowels[i]]) ?? -1;
  }

  const vctx: number[] = [0, 0, 0, 0, 0];
  for (let i = 0; i < Math.min(ns, 5); i++) {
    vctx[i] = vowelContextHash(lower, vowels[i], 512);
  }

  const vctxWide: number[] = [0, 0, 0, 0, 0];
  for (let i = 0; i < Math.min(ns, 5); i++) {
    vctxWide[i] = vowelWideContextHash(lower, vowels[i], wl, 1024);
  }

  const rv: number[] = [-1, -1, -1, -1, -1];
  for (let i = 0; i < Math.min(ns, 5); i++) {
    rv[i] = ns > 0 ? i / ns : 0;
  }

  const cvsh = djb2Hash(cvShape(lower), 2048);
  const softCnt = Array.from(lower).filter((c) => c === "\u044c").length;
  const softPr = lower.includes("\u044c")
    ? lower.lastIndexOf("\u044c") / wl
    : -1;

  let hasDbl = 0;
  for (let i = 0; i < wl - 1; i++) {
    if (
      lower[i] === lower[i + 1] &&
      !VOWEL_SET.has(lower[i]) &&
      /\p{L}/u.test(lower[i])
    ) {
      hasDbl = 1;
      break;
    }
  }

  const sylWeights: number[] = [];
  for (let i = 0; i < ns; i++)
    sylWeights.push(syllableWeight(lower, vowels, i));
  const heavy = sylWeights.filter((w) => w >= 2).length;
  const finalSw = sylWeights.length > 0 ? sylWeights[sylWeights.length - 1] : 0;
  const initSw = sylWeights.length > 0 ? sylWeights[0] : 0;
  const sylWeightVar = variance(sylWeights);

  const [hasInterfix, interfixPosPct] = detectCompoundInterfix(lower, vowels);
  const [prefixCount, longestPrefix] = countPrefixMatches(lower);

  const third = Math.max(1, Math.floor(wl / 3));
  const vdFirst =
    third > 0
      ? Array.from(lower.slice(0, third)).filter((c) => VOWEL_SET.has(c))
          .length / third
      : 0;
  const vdMid =
    third > 0
      ? Array.from(lower.slice(third, 2 * third)).filter((c) =>
          VOWEL_SET.has(c),
        ).length / third
      : 0;
  const vdLast =
    wl > 2 * third
      ? Array.from(lower.slice(2 * third)).filter((c) => VOWEL_SET.has(c))
          .length / Math.max(1, wl - 2 * third)
      : 0;

  const hasApostrophe =
    form.includes("'") || form.includes("\u02bc") || form.includes("\u2019")
      ? 1
      : 0;

  // Morph features default to -1/0 (correct for OOV at inference time)
  const morphCase = -1;
  const morphGender = -1;
  const morphNumber = -1;
  const morphTense = -1;
  const morphN = 0;

  return {
    word_len: wl,
    vowel_count: ns,
    dist_to_first_vowel: distFirst,
    dist_from_last_vowel: distLast,
    first_vowel_ratio: fvr,
    last_vowel_ratio: lvr,
    vowel_ratio_0: vr[0],
    vowel_ratio_1: vr[1],
    vowel_ratio_2: vr[2],
    vowel_ratio_3: vr[3],
    vowel_ratio_4: vr[4],
    vowel_span: vs,
    mid_vowel_ratio: mvr,
    vowel_pair_hash:
      vowels.length > 0 ? djb2Hash(lower[fv] + lower[lv], 256) : 0,
    penult_ratio: penultR,
    antepenult_ratio: antepenR,
    iv_dist_0: iv[0],
    iv_dist_1: iv[1],
    iv_dist_2: iv[2],
    iv_dist_3: iv[3],
    suffix_hash_2: djb2Hash(suf2, 512),
    suffix_hash_3: djb2Hash(suf3, 1024),
    suffix_hash_4: djb2Hash(suf4, 2048),
    suffix_hash_5: djb2Hash(suf5, 2048),
    suffix_hash_6: djb2Hash(suf6, 2048),
    prefix_hash_2: djb2Hash(pre2, 512),
    prefix_hash_3: djb2Hash(pre3, 1024),
    prefix_hash_4: djb2Hash(pre4, 1024),
    suffix_pos_hash: djb2Hash(suf3 + "_" + pos, 2048),
    bigram_pre_lv_hash: bplv,
    bigram_post_fv_hash: bpfv,
    trigram_ending_hash: djb2Hash(suf3, 512),
    onset_cluster_len: vowels.length > 0 ? fv : 0,
    coda_cluster_len: distLast,
    max_cluster_len: maxCl,
    num_open_syllables: nOpen,
    last_syllable_open: lastSylOpenV,
    suffix4_vowel_count: s4vc,
    suffix4_vowel_ratio: s4c.length > 0 ? s4vc / s4c.length : 0,
    char_diversity: wl > 0 ? new Set(lower).size / wl : 0,
    has_apostrophe: hasApostrophe,
    pos_int: POS_INT.get(pos) ?? POS_INT_UNKNOWN,
    syllable_pattern: syllableOnsetPattern(lower),
    has_masc_stress_suffix: hasMasc,
    has_foreign_final_suffix: hasForeign,
    has_greek_interfix: hasGreek,
    has_vy_prefix_noun: hasVyN,
    has_deverbal_suffix: hasDeverbal,
    has_measure_suffix: hasMeasure,
    has_diminutive_adj: hasDimAdj,
    has_root_stress_adj: hasRootAdj,
    is_infinitive: isInf,
    is_verb_1pl_2pl: isV1p2p,
    has_vy_prefix_verb: hasVyV,
    is_reflexive: isRefl,
    is_bisyllable: isBisyl,
    stress_on_first_likely: stressFirst,
    vowel_char_0: vch[0],
    vowel_char_1: vch[1],
    vowel_char_2: vch[2],
    vowel_char_3: vch[3],
    vowel_char_4: vch[4],
    vowel_ctx_hash_0: vctx[0],
    vowel_ctx_hash_1: vctx[1],
    vowel_ctx_hash_2: vctx[2],
    vowel_ctx_hash_3: vctx[3],
    vowel_ctx_hash_4: vctx[4],
    rel_vowel_0: rv[0],
    rel_vowel_1: rv[1],
    rel_vowel_2: rv[2],
    rel_vowel_3: rv[3],
    rel_vowel_4: rv[4],
    cv_shape_hash: cvsh,
    soft_sign_count: softCnt,
    soft_sign_pos_ratio: softPr,
    has_double_consonant: hasDbl,
    heavy_syllable_count: heavy,
    final_syllable_weight: finalSw,
    initial_syllable_weight: initSw,
    vowel_wide_ctx_0: vctxWide[0],
    vowel_wide_ctx_1: vctxWide[1],
    vowel_wide_ctx_2: vctxWide[2],
    vowel_wide_ctx_3: vctxWide[3],
    vowel_wide_ctx_4: vctxWide[4],
    has_compound_interfix: hasInterfix,
    interfix_pos_pct: interfixPosPct,
    prefix_match_count: prefixCount,
    longest_prefix_len: longestPrefix,
    syllable_weight_var: sylWeightVar,
    vowel_density_first_third: Math.round(vdFirst * 10000) / 10000,
    vowel_density_mid_third: Math.round(vdMid * 10000) / 10000,
    vowel_density_last_third: Math.round(vdLast * 10000) / 10000,
    morph_case: morphCase,
    morph_gender: morphGender,
    morph_number: morphNumber,
    morph_tense: morphTense,
    morph_n_features: morphN,
    has_oxytone_mobile_suffix: hasOxytone,
    has_penult_stable_suffix: hasPenult,
    has_adcat_suffix: hasAdcat,
  };
}

// ── Universal extension (+32 features → 132 total) ───────────────────────────

/**
 * Build the 132-feature vector for a Ukrainian word.
 *
 * `pos` should be a Universal Dependencies POS tag: `"NOUN"`, `"VERB"`,
 * `"ADJ"`, `"ADV"`, `"NUM"`, etc. Pass `"X"` when unknown.
 *
 * Feature order matches the `feature_names` array in `manifest.json`
 * exactly, so the resulting `Float32Array` can be fed directly to the
 * Luscinia ONNX model.
 */
export function buildFeaturesUniversal(
  form: string,
  pos: string,
): Record<string, number> {
  const feat = buildFeaturesV13(form, pos);

  const lower = form.toLowerCase();
  const vowels = findVowels(lower);
  const wl = lower.length;
  const ns = vowels.length;

  feat["syllable_count_u"] = ns;
  feat["syllable_count_norm_u"] = Math.min(ns / 10, 1);

  const suf3 = wl >= 3 ? lower.slice(-3) : lower;
  const suf4 = wl >= 4 ? lower.slice(-4) : lower;
  const suf5 = wl >= 5 ? lower.slice(-5) : lower;
  const suf6 = wl >= 6 ? lower.slice(-6) : lower;

  feat["ending_hash_4_u"] = djb2Hash(suf4, 4096);
  feat["ending_hash_5_u"] = djb2Hash(suf5, 4096);
  feat["ending_hash_6_u"] = djb2Hash(suf6, 4096);
  feat["ending_pos_hash_3_u"] = djb2Hash(suf3 + "_" + pos, 4096);
  feat["ending_pos_hash_4_u"] = djb2Hash(suf4 + "_" + pos, 4096);

  if (ns >= 1) {
    const d = wl - vowels[ns - 1] - 1;
    feat["dist_from_end_v_last_u"] = d;
    feat["dist_from_end_v_last_norm_u"] = wl > 0 ? d / wl : 0;
  } else {
    feat["dist_from_end_v_last_u"] = -1;
    feat["dist_from_end_v_last_norm_u"] = -1;
  }
  if (ns >= 2) {
    const d = wl - vowels[ns - 2] - 1;
    feat["dist_from_end_v_pen_u"] = d;
    feat["dist_from_end_v_pen_norm_u"] = wl > 0 ? d / wl : 0;
  } else {
    feat["dist_from_end_v_pen_u"] = -1;
    feat["dist_from_end_v_pen_norm_u"] = -1;
  }
  if (ns >= 3) {
    const d = wl - vowels[ns - 3] - 1;
    feat["dist_from_end_v_ante_u"] = d;
    feat["dist_from_end_v_ante_norm_u"] = wl > 0 ? d / wl : 0;
  } else {
    feat["dist_from_end_v_ante_u"] = -1;
    feat["dist_from_end_v_ante_norm_u"] = -1;
  }

  feat["coda_last_u"] = ns >= 1 ? wl - vowels[ns - 1] - 1 : -1;
  feat["coda_pen_u"] = ns >= 2 ? vowels[ns - 1] - vowels[ns - 2] - 1 : -1;
  feat["coda_ante_u"] = ns >= 3 ? vowels[ns - 2] - vowels[ns - 3] - 1 : -1;

  feat["v_last_char_u"] =
    ns >= 1 ? (VOWEL_CHAR_MAP.get(lower[vowels[ns - 1]]) ?? -1) : -1;
  feat["v_pen_char_u"] =
    ns >= 2 ? (VOWEL_CHAR_MAP.get(lower[vowels[ns - 2]]) ?? -1) : -1;
  feat["v_ante_char_u"] =
    ns >= 3 ? (VOWEL_CHAR_MAP.get(lower[vowels[ns - 3]]) ?? -1) : -1;

  feat["iv_last_gap_u"] =
    wl > 0 && ns >= 2 ? (vowels[ns - 1] - vowels[ns - 2]) / wl : -1;
  feat["iv_pen_gap_u"] =
    wl > 0 && ns >= 3 ? (vowels[ns - 2] - vowels[ns - 3]) / wl : -1;

  feat["has_vy_prefix_u"] =
    lower.startsWith("ви") && (pos === "NOUN" || pos === "VERB") ? 1 : 0;
  feat["has_gerund_suffix_u"] = endsWith(lower, DEVERBAL_SUFFIXES) ? 1 : 0;
  feat["has_verb_aty_u"] = endsWith(lower, VERB_ATY) ? 1 : 0;
  feat["has_verb_yty_u"] = endsWith(lower, VERB_YTY) ? 1 : 0;
  feat["has_verb_uvaty_u"] = endsWith(lower, VERB_UVATY) ? 1 : 0;
  feat["has_oxytone_mobile_u"] = endsWith(lower, OXYTONE_MOBILE_SUFFIXES)
    ? 1
    : 0;
  feat["has_adcat_suffix_u"] =
    pos === "ADJ" && endsWith(lower, ADCAT_SUFFIXES) ? 1 : 0;
  feat["has_num_adtsyat_u"] = endsWith(lower, NUM_ADTSYAT) ? 1 : 0;
  feat["has_foreign_suffix_u"] = endsWith(lower, FOREIGN_IST) ? 1 : 0;
  feat["has_greek_suffix_u"] = endsWith(lower, GREEK_LOG) ? 1 : 0;
  feat["has_penult_stable_u"] = endsWith(lower, PENULT_STABLE_U) ? 1 : 0;

  return feat;
}

/** Feature names in the exact order expected by the ONNX model. */
export const FEATURE_NAMES: readonly string[] = [
  "word_len",
  "vowel_count",
  "dist_to_first_vowel",
  "dist_from_last_vowel",
  "first_vowel_ratio",
  "last_vowel_ratio",
  "vowel_ratio_0",
  "vowel_ratio_1",
  "vowel_ratio_2",
  "vowel_ratio_3",
  "vowel_ratio_4",
  "vowel_span",
  "mid_vowel_ratio",
  "vowel_pair_hash",
  "penult_ratio",
  "antepenult_ratio",
  "iv_dist_0",
  "iv_dist_1",
  "iv_dist_2",
  "iv_dist_3",
  "suffix_hash_2",
  "suffix_hash_3",
  "suffix_hash_4",
  "suffix_hash_5",
  "suffix_hash_6",
  "prefix_hash_2",
  "prefix_hash_3",
  "prefix_hash_4",
  "suffix_pos_hash",
  "bigram_pre_lv_hash",
  "bigram_post_fv_hash",
  "trigram_ending_hash",
  "onset_cluster_len",
  "coda_cluster_len",
  "max_cluster_len",
  "num_open_syllables",
  "last_syllable_open",
  "suffix4_vowel_count",
  "suffix4_vowel_ratio",
  "char_diversity",
  "has_apostrophe",
  "pos_int",
  "syllable_pattern",
  "has_masc_stress_suffix",
  "has_foreign_final_suffix",
  "has_greek_interfix",
  "has_vy_prefix_noun",
  "has_deverbal_suffix",
  "has_measure_suffix",
  "has_diminutive_adj",
  "has_root_stress_adj",
  "is_infinitive",
  "is_verb_1pl_2pl",
  "has_vy_prefix_verb",
  "is_reflexive",
  "is_bisyllable",
  "stress_on_first_likely",
  "vowel_char_0",
  "vowel_char_1",
  "vowel_char_2",
  "vowel_char_3",
  "vowel_char_4",
  "vowel_ctx_hash_0",
  "vowel_ctx_hash_1",
  "vowel_ctx_hash_2",
  "vowel_ctx_hash_3",
  "vowel_ctx_hash_4",
  "rel_vowel_0",
  "rel_vowel_1",
  "rel_vowel_2",
  "rel_vowel_3",
  "rel_vowel_4",
  "cv_shape_hash",
  "soft_sign_count",
  "soft_sign_pos_ratio",
  "has_double_consonant",
  "heavy_syllable_count",
  "final_syllable_weight",
  "initial_syllable_weight",
  "vowel_wide_ctx_0",
  "vowel_wide_ctx_1",
  "vowel_wide_ctx_2",
  "vowel_wide_ctx_3",
  "vowel_wide_ctx_4",
  "has_compound_interfix",
  "interfix_pos_pct",
  "prefix_match_count",
  "longest_prefix_len",
  "syllable_weight_var",
  "vowel_density_first_third",
  "vowel_density_mid_third",
  "vowel_density_last_third",
  "morph_case",
  "morph_gender",
  "morph_number",
  "morph_tense",
  "morph_n_features",
  "has_oxytone_mobile_suffix",
  "has_penult_stable_suffix",
  "has_adcat_suffix",
  // Universal extension (32 features)
  "syllable_count_u",
  "syllable_count_norm_u",
  "ending_hash_4_u",
  "ending_hash_5_u",
  "ending_hash_6_u",
  "ending_pos_hash_3_u",
  "ending_pos_hash_4_u",
  "dist_from_end_v_last_u",
  "dist_from_end_v_last_norm_u",
  "dist_from_end_v_pen_u",
  "dist_from_end_v_pen_norm_u",
  "dist_from_end_v_ante_u",
  "dist_from_end_v_ante_norm_u",
  "coda_last_u",
  "coda_pen_u",
  "coda_ante_u",
  "v_last_char_u",
  "v_pen_char_u",
  "v_ante_char_u",
  "iv_last_gap_u",
  "iv_pen_gap_u",
  "has_vy_prefix_u",
  "has_gerund_suffix_u",
  "has_verb_aty_u",
  "has_verb_yty_u",
  "has_verb_uvaty_u",
  "has_oxytone_mobile_u",
  "has_adcat_suffix_u",
  "has_num_adtsyat_u",
  "has_foreign_suffix_u",
  "has_greek_suffix_u",
  "has_penult_stable_u",
] as const;

export const EXPECTED_FEATURE_COUNT = FEATURE_NAMES.length; // 132

/**
 * Convert a feature dict to a `Float32Array` in model-input order.
 * Missing features are filled with 0.
 */
export function featuresToFloat32(feat: Record<string, number>): Float32Array {
  const arr = new Float32Array(EXPECTED_FEATURE_COUNT);
  for (let i = 0; i < FEATURE_NAMES.length; i++) {
    arr[i] = feat[FEATURE_NAMES[i]] ?? 0;
  }
  return arr;
}

/**
 * Full pipeline: word + POS → Float32Array ready for ONNX inference.
 */
export function featurise(word: string, pos = "X"): Float32Array {
  return featuresToFloat32(buildFeaturesUniversal(word, pos));
}
