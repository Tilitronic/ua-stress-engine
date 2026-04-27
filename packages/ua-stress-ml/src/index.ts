/**
 * ua-stress-ml
 * ============
 * Luscinia LightGBM/ONNX stress predictor for out-of-vocabulary Ukrainian words.
 *
 * @example
 * ```ts
 * import ort from "onnxruntime-web";
 * import { LusciniaPredictor } from "ua-stress-ml";
 *
 * const predictor = await LusciniaPredictor.fromUrl("./model.onnx.gz", ort);
 * const idx = await predictor.predict("університет"); // 4
 * ```
 */
export { LusciniaPredictor } from "./LusciniaPredictor.js";
export type { LusciniaOptions } from "./LusciniaPredictor.js";
export {
  featurise,
  buildFeaturesUniversal,
  djb2Hash,
  findVowels,
  featuresToFloat32,
  FEATURE_NAMES,
  EXPECTED_FEATURE_COUNT,
} from "./featurise.js";
