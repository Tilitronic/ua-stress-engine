/**
 * LusciniaPredictor
 * =================
 * Wraps the Luscinia LightGBM/ONNX model for Ukrainian syllable stress
 * prediction. Accepts a word and optional POS tag; returns the 0-based
 * vowel index that carries the stress.
 *
 * This is an OOV-only predictor: it predicts based on character-level
 * features only and has no dictionary lookup. For in-vocabulary words,
 * use `ua-word-stress` first.
 *
 * Requires `onnxruntime-web` as a peer dependency.
 */

import type { InferenceSession } from "onnxruntime-web";
import { featurise, EXPECTED_FEATURE_COUNT } from "./featurise.js";

/** Minimal ORT namespace shape we rely on. */
type OrtNamespace = {
  InferenceSession: typeof InferenceSession;
  Tensor: new (
    type: "float32",
    data: Float32Array,
    dims: number[],
  ) => InstanceType<typeof import("onnxruntime-web").Tensor>;
};

/** Decompresses a gzip-compressed ArrayBuffer. */
async function decompressGzip(compressed: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new DecompressionStream("deflate-raw");
  const writer = stream.writable.getWriter();
  const reader = stream.readable.getReader();
  writer.write(new Uint8Array(compressed));
  writer.close();
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const totalLen = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Uint8Array(totalLen);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out.buffer;
}

/**
 * Strips the 10-byte gzip header (magic + flags + mtime + xfl + OS) and
 * any optional header extensions, returning the deflate-compressed payload
 * so that `DecompressionStream("deflate-raw")` can decompress it.
 */
function stripGzipHeader(buf: ArrayBuffer): ArrayBuffer {
  const view = new DataView(buf);
  // Verify gzip magic bytes 0x1f 0x8b
  if (view.getUint8(0) !== 0x1f || view.getUint8(1) !== 0x8b) {
    throw new Error("Not a valid gzip stream");
  }
  const flags = view.getUint8(3);
  let offset = 10;
  // FEXTRA
  if (flags & 0x04) {
    const xlen = view.getUint16(offset, true);
    offset += 2 + xlen;
  }
  // FNAME — null-terminated string
  if (flags & 0x08) {
    while (view.getUint8(offset++) !== 0);
  }
  // FCOMMENT — null-terminated string
  if (flags & 0x10) {
    while (view.getUint8(offset++) !== 0);
  }
  // FHCRC
  if (flags & 0x02) offset += 2;
  // Strip the trailing 8-byte gzip footer (CRC32 + ISIZE)
  return buf.slice(offset, buf.byteLength - 8);
}

async function decompressGzipBuffer(buf: ArrayBuffer): Promise<ArrayBuffer> {
  const payload = stripGzipHeader(buf);
  return decompressGzip(payload);
}

// ── Types ────────────────────────────────────────────────────────────────────

/** Options forwarded to `onnxruntime-web` session creation. */
export interface LusciniaOptions {
  /**
   * Execution providers to pass to `InferenceSession.create`.
   * Defaults to `["wasm"]`.
   */
  executionProviders?: string[];
}

export type { OrtNamespace };

// ── Predictor ─────────────────────────────────────────────────────────────────

/**
 * Luscinia ONNX stress predictor.
 *
 * @example
 * ```ts
 * import ort from "onnxruntime-web";
 * import { LusciniaPredictor } from "ua-stress-ml";
 *
 * const predictor = await LusciniaPredictor.fromUrl(
 *   new URL("./data/P3_0017_full.onnx.gz", import.meta.url).href,
 *   ort,
 * );
 * const vowelIdx = await predictor.predict("університет");
 * // vowelIdx === 4  (stress on 5th vowel, 0-based)
 * ```
 */
export class LusciniaPredictor {
  private readonly _session: InferenceSession;
  private readonly _inputName: string;
  private readonly _ort: OrtNamespace;

  private constructor(session: InferenceSession, ort: OrtNamespace) {
    this._session = session;
    this._inputName = session.inputNames[0];
    this._ort = ort;
  }

  // ── Factory methods ────────────────────────────────────────────────────────

  /**
   * Load from a URL (e.g. CDN or `new URL(…, import.meta.url).href`).
   * Supports both `.onnx` (raw) and `.onnx.gz` (gzip-compressed) files.
   */
  static async fromUrl(
    url: string,
    ort: OrtNamespace,
    options: LusciniaOptions = {},
  ): Promise<LusciniaPredictor> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch model from ${url}: ${response.status} ${response.statusText}`,
      );
    }
    const buf = await response.arrayBuffer();
    const modelBuf = url.endsWith(".gz")
      ? await decompressGzipBuffer(buf)
      : buf;
    return LusciniaPredictor._createFromBuffer(modelBuf, ort, options);
  }

  /**
   * Load from a raw `ArrayBuffer` (already decompressed ONNX bytes).
   * Useful when you embed the model via a bundler or load it yourself.
   */
  static async fromBuffer(
    buffer: ArrayBuffer,
    ort: OrtNamespace,
    options: LusciniaOptions = {},
  ): Promise<LusciniaPredictor> {
    return LusciniaPredictor._createFromBuffer(buffer, ort, options);
  }

  /**
   * Load from a gzip-compressed `ArrayBuffer` (e.g. fetched `.onnx.gz`).
   */
  static async fromGzipBuffer(
    gzipBuffer: ArrayBuffer,
    ort: OrtNamespace,
    options: LusciniaOptions = {},
  ): Promise<LusciniaPredictor> {
    const buf = await decompressGzipBuffer(gzipBuffer);
    return LusciniaPredictor._createFromBuffer(buf, ort, options);
  }

  /**
   * Load from a Node.js file path (`.onnx` or `.onnx.gz`).
   * Works in Node.js 18+ environments only.
   */
  static async fromFile(
    filePath: string,
    ort: OrtNamespace,
    options: LusciniaOptions = {},
  ): Promise<LusciniaPredictor> {
    const { readFile } = await import("fs/promises");
    const fileBytes = await readFile(filePath);
    const arrayBuf = fileBytes.buffer.slice(
      fileBytes.byteOffset,
      fileBytes.byteOffset + fileBytes.byteLength,
    ) as ArrayBuffer;
    const modelBuf = filePath.endsWith(".gz")
      ? await decompressGzipBuffer(arrayBuf)
      : arrayBuf;
    return LusciniaPredictor._createFromBuffer(modelBuf, ort, options);
  }

  private static async _createFromBuffer(
    buffer: ArrayBuffer,
    ort: OrtNamespace,
    options: LusciniaOptions,
  ): Promise<LusciniaPredictor> {
    const session = await ort.InferenceSession.create(buffer, {
      executionProviders: options.executionProviders ?? ["wasm"],
    });
    return new LusciniaPredictor(session, ort);
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  /**
   * Predict the 0-based vowel index that carries primary stress.
   *
   * @param word - Ukrainian word (may include apostrophe / diacritic).
   * @param pos  - UD POS tag. Defaults to `"X"` (unknown) when omitted.
   * @returns    - 0-based index into the word's vowel sequence.
   */
  async predict(word: string, pos = "X"): Promise<number> {
    const [result] = await this.predictBatch([word], pos);
    return result;
  }

  /**
   * Predict stress for multiple words in a single batched inference call.
   *
   * All words in a batch share the same `pos` tag. If your words have
   * different POS tags, call `predict()` individually or group by POS.
   *
   * @param words - Array of Ukrainian words.
   * @param pos   - UD POS tag applied to all words. Defaults to `"X"`.
   * @returns     - Array of 0-based vowel indices, same order as input.
   */
  async predictBatch(words: string[], pos = "X"): Promise<number[]> {
    if (words.length === 0) return [];

    const batchSize = words.length;
    const data = new Float32Array(batchSize * EXPECTED_FEATURE_COUNT);
    for (let i = 0; i < batchSize; i++) {
      const features = featurise(words[i], pos);
      data.set(features, i * EXPECTED_FEATURE_COUNT);
    }

    const inputTensor = new this._ort.Tensor("float32", data, [
      batchSize,
      EXPECTED_FEATURE_COUNT,
    ]);

    const feeds: Record<string, typeof inputTensor> = {
      [this._inputName]: inputTensor,
    };

    // Only fetch the first output ("label": int64 [batchSize]).
    // The second output is a ZipMap (sequence of class→probability maps) which
    // ORT-web cannot materialise as a typed array — requesting it causes:
    //   "Reading data from non-tensor typed value is not supported."
    const outputName = this._session.outputNames[0];
    const outputMap = await this._session.run(feeds, [outputName]);
    const output = outputMap[outputName];

    // The ONNX graph already performs the argmax internally, so `label` is an
    // int64 tensor of shape [batchSize] containing the predicted vowel index.
    // ORT-web exposes int64 data as BigInt64Array; use Number() to normalise.
    const labelData = output.data as BigInt64Array | Int32Array;
    const results: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      results.push(Number(labelData[i]));
    }
    return results;
  }

  /**
   * Release ONNX runtime resources.
   * Call this when you are done with the predictor.
   */
  async dispose(): Promise<void> {
    await this._session.release();
  }
}
