/** Ukrainian vowels as a Set for fast membership checks. */
export const UA_VOWELS: ReadonlySet<string> = new Set([..."аеєиіїоуюя"]);

/** Correct Ukrainian apostrophe: U+02BC modifier letter apostrophe. */
export const CORRECT_APOSTROPHE = "\u02bc";

const WRONG_APOSTROPHES = ["\u2019", "\u0027", "\u02bb", "\u0060", "\u00b4"];

/**
 * Normalise all apostrophe variants to U+02BC (modifier letter apostrophe),
 * which is the correct character per Ukrainian orthography.
 *
 * @example
 * normaliseApostrophe("п\u2019ять") // → "п\u02bcять"
 */
export function normaliseApostrophe(text: string): string {
  let t = text;
  for (const ch of WRONG_APOSTROPHES) {
    t = t.replaceAll(ch, CORRECT_APOSTROPHE);
  }
  return t;
}

/**
 * Apply a combining acute accent (U+0301) after the `vowelIndex`-th
 * Ukrainian vowel (0-based count) in `word`.
 *
 * Returns the original word unchanged if `vowelIndex` is out of range.
 *
 * @example
 * applyStressMark('університет', 4) // → 'університе\u0301т'
 * applyStressMark('мама', 0)        // → 'ма\u0301ма'
 */
export function applyStressMark(word: string, vowelIndex: number): string {
  let count = 0;
  for (let i = 0; i < word.length; i++) {
    if (UA_VOWELS.has(word[i].toLowerCase())) {
      if (count === vowelIndex) {
        return word.slice(0, i + 1) + "\u0301" + word.slice(i + 1);
      }
      count++;
    }
  }
  return word;
}

/** Internal: lowercase + normalise apostrophe + strip combining diacritics. */
export function normWord(word: string): string {
  let w = normaliseApostrophe(word.toLowerCase());
  // Strip combining diacritics (e.g. U+0301 stress marks already present in input)
  w = w.normalize("NFD").replace(/\p{Mn}/gu, "");
  return w;
}
