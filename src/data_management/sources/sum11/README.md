# СУМ-11 — Словник Української Мови (11 томів)

## Attribution

> **Словник Української Мови** в 11 томах  
> Академія Наук Української РСР, Інститут мовознавства ім. О. О. Потебні  
> Київ: Наукова думка, 1970–1980

The digitised version used here is the **DiktJson** format published by
[@bakustarver](https://github.com/bakustarver):

```
https://github.com/bakustarver/ukr-dictionaries-list-opensource
```

Release: `ukr-ukr_SUM-11_or_1`  
Archive: `DiktJson-ukr-ukr_SUM-11_or_1.zip`

**License**: The original work predates 1991 and is in the public domain under
Ukrainian law. The digitised JSON distribution carries no additional
restrictions.

---

## Contents of this directory

| File | Purpose |
|------|---------|
| `sum11_parser.py` | Parser — converts СУМ-11 DiktJson → `LinguisticEntry` pairs for LMDB |
| `sum11.json` | Source data (**not committed** — downloaded automatically on first run) |
| `sum11.test.json` | Tiny 8-entry fixture used by the unit-test suite (committed) |
| `README.md` | This file |
| `__init__.py` | Python package marker |

---

## Source format (DiktJson)

The source is a single JSON object where every key is an unstressed bare lemma
and every value is an HTML fragment containing the stressed headword inside
`<b>` tags:

```json
{
  "замок": "<div><b>ЗА́МОК¹</b>, мку, <i>ч.</i> Велика мурована будівля.<b>ЗАМО́К²</b>, мку, <i>ч.</i> Пристрій для замикання.</div>",
  "вода":  "<div><b>ВОДА́</b>, и, <i>ж.</i> Прозора рідина без кольору.</div>"
}
```

Key observations:

* **Stress** is encoded as Unicode combining acute U+0301 placed *after* the
  stressed vowel in the bold headword (`А\u0301` = `А́`).
* **Heteronyms** (same spelling, different stress/meaning) share one JSON key
  and appear as multiple `<b>…</b>` blocks in the value.
* **Superscript digits** `¹²³` (U+00B9, U+00B2, U+00B3) distinguish
  homographs and are stripped before bare-form comparison.
* **Metadata** keys start with `##` and are skipped by the parser.
* Approximately 127 K lemma entries; ~124 K carry explicit stress marks;
  ~867 entries contain two or more stressed headwords (heteronym groups).

---

## How the parser works

```
sum11.json (DiktJson)
      │
      ▼  parse_sum11_to_unified_dict()
      │  • reads all (key, HTML) pairs via object_pairs_hook
      │    (handles duplicate JSON keys that encode heteronyms)
      │  • calls _parse_entry() per key → List[WordForm]
      │  • groups by normalised lemma
      │  • deduplicates (stress_indices, pos) pairs
      │  • yields (lemma, LinguisticEntry)
      │
      ▼  stream_sum11_to_lmdb()
      │  • serialises each LinguisticEntry via to_serializable()
      │  • writes to a disk-backed LMDB via LMDBExporter
      │
      ▼  run_sum11_parser()          ← called by parsing_merging_service
         • resolves path (env UA_SUM11_JSON → default → auto-download)
         • checks LMDB cache (keyed by parser+source hash)
         • returns (lmdb_path, stats)
```

---

## Automatic download

If `sum11.json` is not found at the expected path, the parser downloads and
extracts it automatically from the GitHub release:

```python
from src.data_management.sources.sum11.sum11_parser import download_sum11
download_sum11("src/data_management/sources/sum11/sum11.json")
```

Or set the `UA_SUM11_JSON` environment variable to a custom path before
running `build_master_db.py`.

To disable automatic download (e.g. in CI without network access), pass
`auto_download=False` to `run_sum11_parser()` — it will raise
`FileNotFoundError` with instructions.

---

## Source version hashing

Every parser run computes a combined hash of the **parser source file** and
the **sum11.json data file** via `compute_parser_hash()`.  This hash is
embedded in the LMDB cache directory name, so changing either the parser
logic or the source data automatically invalidates the cache and forces a
full re-parse.

You can also obtain the SHA-256 of the data file alone:

```python
from src.data_management.sources.sum11.sum11_parser import compute_source_hash
print(compute_source_hash("src/data_management/sources/sum11/sum11.json"))
```

---

## Running the tests

```bash
pytest tests/src/data_management/sources/test_sum11_parser.py -v
```

All tests are self-contained: they use `sum11.test.json` and mock network
calls, so no internet access or full source file is required.
