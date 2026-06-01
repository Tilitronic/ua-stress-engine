/**
 * check-schema-drift.mjs
 *
 * CI guard: regenerates the JSON schema from Rust types and verifies that the
 * committed `src/generated/schema.json` is up-to-date.
 *
 * Exits 0 if up-to-date, 1 if drift is detected.
 *
 * Usage:
 *   node scripts/check-schema-drift.mjs          # from packages/ua-stress-web/
 *   pnpm --filter ua-word-stress check:drift     # from workspace root
 */

import { execSync, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(__dirname, "..");
const workspaceRoot = join(pkgRoot, "../..");
const committedSchema = join(pkgRoot, "src/generated/schema.json");

// ── 1. Run schema-export binary ─────────────────────────────────────────────
const tmpFile = join(tmpdir(), `schema-${randomUUID()}.json`);

const result = spawnSync(
  "cargo",
  ["run", "-p", "builder", "--bin", "schema-export", "--quiet"],
  { cwd: workspaceRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
);

if (result.status !== 0) {
  console.error("schema-export failed:\n", result.stderr);
  process.exit(1);
}

writeFileSync(tmpFile, result.stdout);

// ── 2. Compare with committed schema ────────────────────────────────────────
let committed;
try {
  committed = readFileSync(committedSchema, "utf8");
} catch {
  console.error(
    `Committed schema not found at ${committedSchema}.\n` +
      "Run: pnpm --filter ua-word-stress generate",
  );
  process.exit(1);
}

if (result.stdout.trim() === committed.trim()) {
  console.log("✔ Schema is up-to-date.");
  process.exit(0);
} else {
  console.error(
    "✖ Schema drift detected!\n" +
      "The Rust types have changed but src/generated/schema.json was not regenerated.\n" +
      "Fix: pnpm --filter ua-word-stress generate && git add src/generated/",
  );
  process.exit(1);
}
