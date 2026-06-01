//! schema-export — emit the JSON Schema for `WordLookupResult` to stdout.
//!
//! Usage (from workspace root):
//!   cargo run -p builder --bin schema-export \
//!     > packages/ua-stress-web/src/generated/schema.json
//!
//! This binary is called by the npm `generate:schema` script and by the CI
//! drift-guard script.  It has no runtime state — it is purely a compile-time
//! type introspection tool.

fn main() {
    let schema = schemars::schema_for!(ua_stress_core::types::WordLookupResult);
    let json = serde_json::to_string_pretty(&schema)
        .expect("schemars schema must be serialisable to JSON");
    println!("{json}");
}
