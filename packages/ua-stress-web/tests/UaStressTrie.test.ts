/**
 * ua-word-stress — Vitest test suite
 *
 * Requires the data file built by `python build_web_stress_db.py`:
 *   data/ua_stress.ctrie.gz
 *
 * Run from the npm/ directory:
 *   npx vitest run
 */

import { describe, it, expect, beforeAll } from "vitest";
import { fileURLToPath } from "url";
import { resolve, dirname } from "path";

import { UaStressTrie } from "../src/UaStressTrie.js";
import { applyStressMark, normaliseApostrophe } from "../src/utils.js";

// ── Data path ─────────────────────────────────────────────────────────────────
const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_FILE = resolve(__dirname, "../data/ua_stress.ctrie.gz");

// ── Shared trie instance ──────────────────────────────────────────────────────
let trie: UaStressTrie;

beforeAll(async () => {
  trie = await UaStressTrie.fromFile(DATA_FILE);
});

// ── Loading ───────────────────────────────────────────────────────────────────
describe("loading", () => {
  it("constructs a UaStressTrie instance", () => {
    expect(trie).toBeInstanceOf(UaStressTrie);
  });

  it("wordCount > 2 million", () => {
    expect(trie.wordCount).toBeGreaterThan(2_000_000);
  });

  it("nodeCount > 1 million", () => {
    expect(trie.nodeCount).toBeGreaterThan(1_000_000);
  });

  it("stats object is consistent", () => {
    const s = trie.stats;
    expect(s.wordCount).toBe(trie.wordCount);
    expect(s.nodeCount).toBe(trie.nodeCount);
    expect(s.gzSizeBytes).toBeGreaterThan(0);
  });

  it("throws on corrupt data", () => {
    const bad = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]);
    expect(() => UaStressTrie.fromBuffer(bad.buffer as ArrayBuffer)).toThrow(
      /bad magic/i,
    );
  });
});

// ── Smoke words (stress index ground truth) ───────────────────────────────────
const SMOKE: Array<[string, number]> = [
  ["мама", 0], // МА-ма     а(0)
  ["вода", 1], // во-ДА     а(1)
  ["університет", 4], // університЕт  е(4)
  ["читати", 1], // чи-ТА-ти  а(1)
  ["місто", 0], // МІс-то    і(0)
  ["батько", 0], // БАТь-ко   а(0)
  ["земля", 1], // зем-ЛЯ    я(1)
  ["книга", 0], // кни-га    и(0)  (кнИга)
];

describe("lookup — smoke words", () => {
  for (const [word, expectedStress] of SMOKE) {
    it(`lookup('${word}') === ${expectedStress}`, () => {
      expect(trie.lookup(word)).toBe(expectedStress);
    });
  }
});

// ── Normalisation ─────────────────────────────────────────────────────────────
describe("input normalisation", () => {
  it("is case-insensitive", () => {
    expect(trie.lookup("УНІВЕРСИТЕТ")).toBe(trie.lookup("університет"));
    expect(trie.lookup("Місто")).toBe(trie.lookup("місто"));
  });

  it("normalises apostrophe variants to U+02BC", () => {
    // U+2019 right single quotation mark  →  U+02BC modifier letter apostrophe
    // м'яч (ball) — stress index 0, confirmed present in the trie
    const withWrong = "\u043c\u2019\u044f\u0447";   // м + U+2019 + яч
    const withCorrect = "\u043c\u02bc\u044f\u0447";  // м + U+02BC + яч
    const r1 = trie.lookup(withWrong);
    const r2 = trie.lookup(withCorrect);
    expect(r1).not.toBeNull();
    expect(r1).toBe(r2);
    expect(r1).toBe(0);
  });

  it("strips pre-existing combining accent marks in input", () => {
    // Input already has U+0301 on е
    expect(trie.lookup("університе\u0301т")).toBe(4);
  });
});

// ── lookupFull ────────────────────────────────────────────────────────────────
describe("lookupFull", () => {
  it("returns null for OOV words", () => {
    expect(trie.lookupFull("нездійсненненнеслово123")).toBeNull();
  });

  it("returns null for multi-word phrases (spaces)", () => {
    expect(trie.lookupFull("добрий день")).toBeNull();
  });

  it("returns uncertain=false for unambiguous words", () => {
    const r = trie.lookupFull("університет");
    expect(r).not.toBeNull();
    expect(r!.uncertain).toBe(false);
  });

  it("returns a valid stress index between 0 and 10", () => {
    for (const [word] of SMOKE) {
      const r = trie.lookupFull(word);
      expect(r).not.toBeNull();
      expect(r!.stress).toBeGreaterThanOrEqual(0);
      expect(r!.stress).toBeLessThanOrEqual(10);
    }
  });
});

// ── mark ─────────────────────────────────────────────────────────────────────
describe("mark", () => {
  it("inserts U+0301 after the stressed vowel", () => {
    // мама → stress=0 → а(0) → ма́ма
    expect(trie.mark("мама")).toBe("ма\u0301ма");
  });

  it("returns null for unknown words", () => {
    expect(trie.mark("нездійсненненнеслово123")).toBeNull();
  });

  it("marks університет correctly", () => {
    // vowels: у(0) і(1) е(2) и(3) е(4) — stress=4 → е at index 4
    expect(trie.mark("університет")).toBe("університе\u0301т");
  });
});

// ── applyStressMark utility ───────────────────────────────────────────────────
describe("applyStressMark()", () => {
  it("inserts accent after vowel at index 0", () => {
    expect(applyStressMark("мама", 0)).toBe("ма\u0301ма");
  });

  it("inserts accent after vowel at index 1", () => {
    expect(applyStressMark("вода", 1)).toBe("вода\u0301");
  });

  it("is a no-op when vowelIndex is out of range", () => {
    expect(applyStressMark("мама", 99)).toBe("мама");
  });

  it("handles words with no vowels", () => {
    expect(applyStressMark("бжж", 0)).toBe("бжж");
  });
});

// ── normaliseApostrophe utility ───────────────────────────────────────────────
describe("normaliseApostrophe()", () => {
  const cases: Array<[string, string]> = [
    ["п\u2019ять", "п\u02bcять"], // U+2019
    ["п'ять", "п\u02bcять"], // U+0027 ASCII apostrophe
    ["п\u02bcять", "п\u02bcять"], // already correct
  ];
  for (const [input, expected] of cases) {
    it(`normalises U+${input.charCodeAt(1).toString(16).toUpperCase().padStart(4, "0")}`, () => {
      expect(normaliseApostrophe(input)).toBe(expected);
    });
  }
});

// ── Data structure integrity ──────────────────────────────────────────────────
describe("data structure integrity", () => {
  it("all smoke words are found (coverage)", () => {
    const missing = SMOKE.filter(([w]) => trie.lookup(w) === null).map(
      ([w]) => w,
    );
    expect(missing).toEqual([]);
  });

  it("wordCount matches independently verified lower bound", () => {
    // We know the master DB has ~2.87M single-word forms
    expect(trie.wordCount).toBeGreaterThan(2_500_000);
  });
});
