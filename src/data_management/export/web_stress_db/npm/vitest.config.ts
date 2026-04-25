import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    testTimeout: 60_000, // loading 9 MB + 4 M nodes takes a few seconds
    globals: false,
  },
});
