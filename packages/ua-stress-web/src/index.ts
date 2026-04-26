/**
 * ua-stress-trie — public API surface
 *
 * @packageDocumentation
 */

export {
  UaStressTrie,
  applyStressMark,
  normaliseApostrophe,
} from "./UaStressTrie.js";
export { UA_VOWELS, CORRECT_APOSTROPHE, normWord } from "./utils.js";
export type { LookupResult, TrieStats, UkrainianVowel } from "./types.js";
