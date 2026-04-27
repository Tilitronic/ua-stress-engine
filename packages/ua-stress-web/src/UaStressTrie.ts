import type { LookupResult, TrieStats } from "./types.js";
import { normWord, applyStressMark, normaliseApostrophe } from "./utils.js";

// ── Binary format constants — v1 (0x01) + v2 (0x02) ———————————————————————————————
const MAGIC = 0x54534b55; // 'UKST' as LE uint32
const SUPPORTED_VERSIONS = new Set([0x01, 0x02]);
const HEADER_SIZE = 20;
const ALPHABET_ENTRY_SIZE = 3; // char_byte(u8) + codepoint(u16 LE)
const NODE_SIZE = 8; // char_id(u8) + stress(u8) + flags(u8) + stress2(u8) + first_child(u32 LE)

const NO_STRESS = 0xff;
// v1 flag (bit 0 was "heteronym"): kept for backward-compat when reading v1 files
const FLAG_V1_UNCERTAIN   = 0x01;
// v2 flags
const FLAG_VARIATIVE      = 0x01; // both stresses orthographically valid
const FLAG_HAS_CHILDREN   = 0x02;
const FLAG_HETERONYM      = 0x04; // different meanings / forms
const FLAG_HAS_SECONDARY  = 0x08; // stress2 byte (byte[3]) is valid

// ── Gzip decompression ────────────────────────────────────────────────────────

async function decompressGzip(compressed: ArrayBuffer): Promise<ArrayBuffer> {
  // Modern browsers (2022+): DecompressionStream API
  if (typeof DecompressionStream !== "undefined") {
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(new Uint8Array(compressed));
    await writer.close();
    const chunks: Uint8Array[] = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    const total = chunks.reduce((acc, c) => acc + c.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const chunk of chunks) {
      out.set(chunk, off);
      off += chunk.length;
    }
    return out.buffer as ArrayBuffer;
  }

  // Node.js fallback (dynamic import — keeps browser bundle clean)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const zlib = await import("zlib" as any);
  const result: Buffer = zlib.gunzipSync(Buffer.from(compressed));
  return result.buffer.slice(
    result.byteOffset,
    result.byteOffset + result.byteLength,
  ) as ArrayBuffer;
}

// ── Main class ────────────────────────────────────────────────────────────────

/**
 * Offline Ukrainian word stress resolver backed by a compact binary trie.
 *
 * Lookup is **O(word_length)** and entirely in-memory — no network I/O after
 * the initial load.
 *
 * @example
 * ```ts
 * // Browser
 * const trie = await UaStressTrie.fromUrl('/static/ua_stress.ctrie.gz')
 *
 * // Node.js
 * const trie = await UaStressTrie.fromFile('./data/ua_stress.ctrie.gz')
 *
 * trie.lookup('університет')   // → 4
 * trie.mark('університет')     // → 'університе\u0301т'
 * trie.lookupFull('замок')     // → { stress: 0, stresses: [0, 1], type: 'heteronym', uncertain: true }
 * trie.lookupFull('помилка')   // → { stress: 0, stresses: [0, 1], type: 'variative', uncertain: true }
 * ```
 */
export class UaStressTrie {
  private readonly _view: DataView;
  private readonly _charToId: Map<number, number>;
  private readonly _nodesOffset: number;
  private readonly _nodeCount: number;
  private readonly _wordCount: number;
  private readonly _alphaSize: number;
  private readonly _gzSizeBytes: number;
  private readonly _version: number;

  /**
   * Construct from a raw (already-decompressed) `.ctrie` `ArrayBuffer`.
   *
   * You normally don't call this directly — use a factory method instead.
   */
  constructor(buffer: ArrayBuffer, gzSizeBytes = 0) {
    const view = new DataView(buffer);

    const magic = view.getUint32(0, true);
    if (magic !== MAGIC) {
      throw new Error(
        `Not a .ctrie file — bad magic: 0x${magic.toString(16).toUpperCase()} ` +
          `(expected 0x${MAGIC.toString(16).toUpperCase()})`,
      );
    }

    const version = view.getUint8(4);
    if (!SUPPORTED_VERSIONS.has(version)) {
      throw new Error(
        `Unsupported .ctrie version: ${version} (supported: ${[...SUPPORTED_VERSIONS].join(", ")})`,
      );
    }

    this._nodeCount = view.getUint32(6, true);
    this._wordCount = view.getUint32(10, true);
    this._alphaSize = view.getUint32(14, true);
    this._gzSizeBytes = gzSizeBytes;
    this._version = version;

    // Parse alphabet table → Map<Unicode codepoint, char_id>
    this._charToId = new Map();
    let offset = HEADER_SIZE;
    for (let i = 0; i < this._alphaSize; i++) {
      const cp = view.getUint16(offset + 1, true);
      this._charToId.set(cp, i);
      offset += ALPHABET_ENTRY_SIZE;
    }

    this._nodesOffset = offset;
    this._view = view;
  }

  // ── Factory methods ─────────────────────────────────────────────────────────

  /**
   * Fetch a `.ctrie` or `.ctrie.gz` file and parse it into a `UaStressTrie`.
   *
   * Automatically detects and decompresses gzip data (works both with and
   * without `Content-Encoding: gzip` on the server).
   *
   * **Recommended serving setup** (avoids client-side decompression work):
   * ```
   * Content-Encoding: gzip
   * Content-Type: application/octet-stream
   * ```
   *
   * @param url - URL to `.ctrie.gz` (or raw `.ctrie`)
   */
  static async fromUrl(url: string): Promise<UaStressTrie> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
    const rawBuf = await resp.arrayBuffer();
    const bytes = new Uint8Array(rawBuf);

    // gzip magic bytes: 0x1F 0x8B
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      const decompressed = await decompressGzip(rawBuf);
      return new UaStressTrie(decompressed, rawBuf.byteLength);
    }

    return new UaStressTrie(rawBuf);
  }

  /**
   * Load from a local file path (**Node.js only**).
   *
   * Handles both `.ctrie.gz` (gzip-compressed) and raw `.ctrie` files.
   *
   * @param filePath - Absolute or relative path to the data file
   */
  static async fromFile(filePath: string): Promise<UaStressTrie> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fs = await import("fs" as any);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const zlib = await import("zlib" as any);

    const raw: Buffer = fs.readFileSync(filePath);
    const isGzip = raw[0] === 0x1f && raw[1] === 0x8b;

    if (isGzip) {
      const decompressed: Buffer = zlib.gunzipSync(raw);
      const ab = decompressed.buffer.slice(
        decompressed.byteOffset,
        decompressed.byteOffset + decompressed.byteLength,
      ) as ArrayBuffer;
      return new UaStressTrie(ab, raw.byteLength);
    }

    const ab = raw.buffer.slice(
      raw.byteOffset,
      raw.byteOffset + raw.byteLength,
    ) as ArrayBuffer;
    return new UaStressTrie(ab);
  }

  /**
   * Parse from an already-decompressed `ArrayBuffer`.
   *
   * Useful when you loaded the binary data yourself (e.g. from a Service
   * Worker cache, IndexedDB, or a custom fetch wrapper).
   *
   * @param buffer - Raw, decompressed `.ctrie` bytes
   */
  static fromBuffer(buffer: ArrayBuffer): UaStressTrie {
    return new UaStressTrie(buffer);
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  /** Total number of word forms stored in the trie. */
  get wordCount(): number {
    return this._wordCount;
  }

  /** Total number of internal nodes (structural metric). */
  get nodeCount(): number {
    return this._nodeCount;
  }

  /** Metadata summary (wordCount, nodeCount, gzSizeBytes). */
  get stats(): TrieStats {
    return {
      wordCount: this._wordCount,
      nodeCount: this._nodeCount,
      gzSizeBytes: this._gzSizeBytes,
    };
  }

  /**
   * Look up the stressed vowel index (0-based) for a word.
   *
   * Returns `null` if the word is not in the trie — use an ONNX-based
   * fallback (e.g. Luscinia) for OOV words.
   *
   * Apostrophe variants and casing are normalised automatically.
   *
   * @example
   * trie.lookup('університет')   // → 4
   * trie.lookup('мама')          // → 0
   * trie.lookup('unknownword')   // → null
   */
  lookup(word: string): number | null {
    return this.lookupFull(word)?.stress ?? null;
  }

  /**
   * Full lookup returning all stress positions and type classification.
   *
   * @returns `LookupResult` or `null` if the word is not in the database.
   *
   * The `type` field tells you how to interpret multiple stress positions:
   *
   * - `"unique"`    — one stress, no ambiguity.
   * - `"variative"` — both positions in `stresses[]` are orthographically
   *                    valid simultaneously (e.g. _по́милка_ / _поми́лка_).
   *                    Present both to the user, or pick either — both are correct.
   * - `"heteronym"` — different positions correspond to different meanings
   *                    or grammatical forms (e.g. _за́мок_ "lock" vs _замо́к_ "castle";
   *                    _бло́хи_ gen.sg vs _блохи́_ nom.pl).
   *                    Context-aware disambiguation is needed to pick the right one.
   *
   * @example
   * trie.lookupFull('університет')
   * // → { stress: 4, stresses: [4], type: 'unique', uncertain: false }
   *
   * trie.lookupFull('помилка')
   * // → { stress: 0, stresses: [0, 1], type: 'variative', uncertain: true }
   *
   * trie.lookupFull('замок')
   * // → { stress: 0, stresses: [0, 1], type: 'heteronym', uncertain: true }
   *
   * trie.lookupFull('xyz123')
   * // → null
   */
  lookupFull(word: string): LookupResult | null {
    const norm = normWord(word);
    let nodeIdx = 0; // root

    for (const ch of norm) {
      const cid = this._charToId.get(ch.codePointAt(0)!);
      if (cid === undefined) return null;

      const childIdx = this._findChild(nodeIdx, cid);
      if (childIdx === -1) return null;
      nodeIdx = childIdx;
    }

    const stress = this._nodeByte(nodeIdx, 1);
    if (stress === NO_STRESS) return null;

    const flags = this._nodeByte(nodeIdx, 2);

    let stresses: number[];
    let type: "unique" | "variative" | "heteronym";

    if (this._version >= 2) {
      // v2: explicit flag bits + stress2 byte
      const stress2 = this._nodeByte(nodeIdx, 3);
      stresses =
        (flags & FLAG_HAS_SECONDARY) !== 0 && stress2 !== NO_STRESS
          ? [stress, stress2]
          : [stress];
      if (flags & FLAG_VARIATIVE) type = "variative";
      else if (flags & FLAG_HETERONYM) type = "heteronym";
      else type = "unique";
    } else {
      // v1 backward compat: bit 0 was generic "uncertain" (heteronym-only)
      stresses = [stress];
      type = (flags & FLAG_V1_UNCERTAIN) !== 0 ? "heteronym" : "unique";
    }

    return { stress, stresses, type, uncertain: type !== "unique" };
  }

  /**
   * Return the word with a combining acute accent (U+0301) on the stressed
   * vowel, or `null` if the word is not in the trie.
   *
   * The accent is inserted *after* the stressed vowel so it renders correctly
   * in all Unicode-compliant environments.
   *
   * @example
   * trie.mark('університет') // → 'університе\u0301т'
   * trie.mark('мама')        // → 'ма\u0301ма'
   * trie.mark('xyz123')      // → null
   */
  mark(word: string): string | null {
    const result = this.lookupFull(word);
    if (result === null) return null;
    return applyStressMark(word, result.stress);
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  private _nodeByte(nodeIdx: number, byteOffset: number): number {
    return this._view.getUint8(
      this._nodesOffset + nodeIdx * NODE_SIZE + byteOffset,
    );
  }

  private _firstChild(nodeIdx: number): number {
    return this._view.getUint32(
      this._nodesOffset + nodeIdx * NODE_SIZE + 4,
      true,
    );
  }

  /**
   * Find the child of `parentIdx` whose char_id equals `targetId`.
   *
   * Children are contiguous in BFS order and sorted by char_id (ascending),
   * so we stop early when `char_id > targetId`.
   *
   * Returns the child node index, or -1 if not found.
   */
  private _findChild(parentIdx: number, targetId: number): number {
    if (!(this._nodeByte(parentIdx, 2) & FLAG_HAS_CHILDREN)) return -1;
    const fc = this._firstChild(parentIdx);
    if (fc === 0) return -1;

    for (let i = 0; i < this._alphaSize; i++) {
      const idx = fc + i;
      if (idx >= this._nodeCount) break;
      const cid = this._nodeByte(idx, 0);
      if (cid === targetId) return idx;
      if (cid > targetId) break; // sorted → target absent
    }
    return -1;
  }
}

export { applyStressMark, normaliseApostrophe };
