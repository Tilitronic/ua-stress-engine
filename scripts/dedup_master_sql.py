#!/usr/bin/env python3
"""Deduplicate merged master SQLite database in-place.

This script performs:
1) Exact deduplication of word_form rows on canonical fields.
2) Rewiring child-table foreign keys to the kept word_form id.
3) Deduplication of child rows (feature/tag/example/etc.) per word_form.

Usage:
  python scripts/dedup_master_sql.py --db path/to/MERGEDSQL_xxx.sqlite3
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


WORD_FORM_KEY = """
ifnull(w.form,'') = ifnull(k.form,'') AND
ifnull(w.lemma,'') = ifnull(k.lemma,'') AND
ifnull(w.pos,'') = ifnull(k.pos,'') AND
ifnull(w.main_definition_id,-1) = ifnull(k.main_definition_id,-1) AND
ifnull(w.roman,'') = ifnull(k.roman,'') AND
ifnull(w.ipa,'') = ifnull(k.ipa,'') AND
ifnull(w.etymology_id,-1) = ifnull(k.etymology_id,-1) AND
ifnull(w.etymology_number,-1) = ifnull(k.etymology_number,-1) AND
ifnull(w.sense_id,'') = ifnull(k.sense_id,'') AND
ifnull(w.stress_indices_json,'[]') = ifnull(k.stress_indices_json,'[]')
"""


def scalar(cur: sqlite3.Cursor, sql: str) -> int:
    cur.execute(sql)
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def dedup_child(cur: sqlite3.Cursor, table: str, cols: list[str]) -> int:
    col_csv = ", ".join(cols)
    sql = f"""
    DELETE FROM {table}
    WHERE id NOT IN (
        SELECT MIN(id) FROM {table}
        GROUP BY {col_csv}
    )
    """
    cur.execute(sql)
    return cur.rowcount if cur.rowcount is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate merged master SQLite database in-place")
    ap.add_argument("--db", required=True, help="Path to MERGEDSQL_*.sqlite3")
    ap.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deduplication to compact the database file",
    )
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    con = sqlite3.connect(str(db))
    cur = con.cursor()

    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA cache_size=-200000")

    t0 = time.time()

    before_word_form = scalar(cur, "SELECT COUNT(*) FROM word_form")
    print(f"word_form before: {before_word_form}")

    # 1) Build keeper table for exact word_form duplicates.
    cur.execute("DROP TABLE IF EXISTS _wf_keep")
    cur.execute(
        """
        CREATE TEMP TABLE _wf_keep AS
        SELECT
            MIN(id) AS keep_id,
            form, lemma, pos, main_definition_id, roman, ipa,
            etymology_id, etymology_number, sense_id, stress_indices_json
        FROM word_form
        GROUP BY
            ifnull(form,''), ifnull(lemma,''), ifnull(pos,''),
            ifnull(main_definition_id,-1), ifnull(roman,''), ifnull(ipa,''),
            ifnull(etymology_id,-1), ifnull(etymology_number,-1),
            ifnull(sense_id,''), ifnull(stress_indices_json,'[]')
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS _wf_keep_keep_id_idx ON _wf_keep(keep_id)")

    cur.execute("DROP TABLE IF EXISTS _wf_dups")
    cur.execute(
        f"""
        CREATE TEMP TABLE _wf_dups AS
        SELECT w.id AS dup_id, k.keep_id AS keep_id
        FROM word_form w
        JOIN _wf_keep k
          ON {WORD_FORM_KEY}
        WHERE w.id <> k.keep_id
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS _wf_dups_dup_id_idx ON _wf_dups(dup_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS _wf_dups_keep_id_idx ON _wf_dups(keep_id)")

    dup_word_form = scalar(cur, "SELECT COUNT(*) FROM _wf_dups")
    print(f"word_form duplicates identified: {dup_word_form}")

    # 2) Rewire child tables to keep_id.
    child_tables = [
        "feature", "translation", "etymology_template", "inflection_template",
        "category", "tag", "example", "possible_stress_index", "meta",
    ]

    for t in child_tables:
        cur.execute(
            f"""
            UPDATE {t}
            SET word_form_id = (
                SELECT keep_id FROM _wf_dups d WHERE d.dup_id = {t}.word_form_id
            )
            WHERE EXISTS (
                SELECT 1 FROM _wf_dups d WHERE d.dup_id = {t}.word_form_id
            )
            """
        )

    # 3) Delete duplicate word_form rows.
    cur.execute("DELETE FROM word_form WHERE id IN (SELECT dup_id FROM _wf_dups)")
    print(f"word_form duplicates deleted: {dup_word_form}")

    # 4) Deduplicate child rows after rewiring.
    removed_children = 0
    removed_children += dedup_child(cur, "feature", ["word_form_id", "key", "value"])
    removed_children += dedup_child(cur, "translation", ["word_form_id", "lang", "text", "sense"])
    removed_children += dedup_child(cur, "etymology_template", ["word_form_id", "name", "args_json"])
    removed_children += dedup_child(cur, "inflection_template", ["word_form_id", "name", "args_json"])
    removed_children += dedup_child(cur, "category", ["word_form_id", "category"])
    removed_children += dedup_child(cur, "tag", ["word_form_id", "tag"])
    removed_children += dedup_child(cur, "example", ["word_form_id", "example"])
    removed_children += dedup_child(cur, "possible_stress_index", ["word_form_id", "stress_indices_json"])
    removed_children += dedup_child(cur, "meta", ["word_form_id", "meta_json"])

    con.commit()
    print(f"child duplicates removed: {removed_children}")

    after_word_form = scalar(cur, "SELECT COUNT(*) FROM word_form")
    print(f"word_form after: {after_word_form}")

    # 5) Rebuild and compact.
    cur.execute("REINDEX")
    if args.vacuum:
        print("running VACUUM...")
        cur.execute("VACUUM")
        print("VACUUM complete")
    con.close()

    dt = time.time() - t0
    print(
        "\n".join(
            [
                f"DB: {db}",
                f"word_form before: {before_word_form}",
                f"word_form duplicates removed: {dup_word_form}",
                f"word_form after: {after_word_form}",
                f"child duplicates removed: {removed_children}",
                f"elapsed_sec: {dt:.2f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
