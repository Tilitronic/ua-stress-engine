"""
src.data_management.sources

Four raw data sources that feed the master SQLite DB:

  kaikki              — Wiktionary-derived JSONL (CC BY-SA 4.0)
  trie_ua_stresses    — lang-uk marisa-trie file (MIT)
  txt_ua_stresses     — lang-uk plain-text dictionary (MIT / ULIF)
  ua_variative_stressed_words — curated free-variant stress list

Each sub-package exposes a parser that yields (lemma, LinguisticEntry)
pairs for downstream merging.
"""