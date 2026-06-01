/**
 * ua-word-stress — public API surface
 *
 * @packageDocumentation
 */

export {
  UaStressTrie,
  applyStressMark,
  normaliseApostrophe,
} from "./UaStressTrie.js";
export { UA_VOWELS, CORRECT_APOSTROPHE, normWord } from "./utils.js";
export type {
  LookupResult,
  StressVariant,
  TrieStats,
  UkrainianVowel,
} from "./types.js";
