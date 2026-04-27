import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  splitting: false,
  platform: "neutral",
  external: ["onnxruntime-web", "fs", "path"],
  target: "es2021",
});
