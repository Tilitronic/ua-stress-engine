import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  target: "es2020",
  // Keep platform-neutral — no Node.js built-ins in the main bundle
  platform: "neutral",
  esbuildOptions(options) {
    options.conditions = ["import", "default"];
  },
});
