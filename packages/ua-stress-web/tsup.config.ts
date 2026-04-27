import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  target: "es2020",
  // Keep platform-neutral — no Node.js built-ins in the main bundle.
  // fs/zlib are marked external so esbuild doesn't try to bundle them;
  // in browsers the DecompressionStream path runs first and they are
  // never imported.
  platform: "neutral",
  external: ["fs", "zlib"],
  esbuildOptions(options) {
    options.conditions = ["import", "default"];
  },
});
