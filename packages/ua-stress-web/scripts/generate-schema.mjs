/**
 * generate-schema.mjs
 *
 * Runs the Rust `schema-export` binary and writes its stdout (JSON Schema)
 * to `src/generated/schema.json` with correct UTF-8 encoding.
 *
 * Using Node.js child_process avoids Windows PowerShell console-encoding bugs
 * that corrupt non-ASCII characters in pipe redirections.
 */

import { spawnSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = join(__dirname, "../../..");
const outFile = join(__dirname, "../src/generated/schema.json");

mkdirSync(dirname(outFile), { recursive: true });

const result = spawnSync(
  "cargo",
  ["run", "-p", "builder", "--bin", "schema-export"],
  {
    cwd: workspaceRoot,
    encoding: "utf8",
    env: { ...process.env, CARGO_TERM_COLOR: "never" },
    // stderr goes to terminal so progress is visible; stdout is captured
    stdio: ["ignore", "pipe", "inherit"],
  },
);

if (result.status !== 0) {
  console.error("schema-export failed");
  process.exit(1);
}

writeFileSync(outFile, result.stdout, "utf8");
console.log(
  `Written ${result.stdout.length} chars → src/generated/schema.json`,
);
