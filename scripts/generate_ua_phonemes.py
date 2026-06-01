"""
Generate ua_phonemes.json — Panphon feature data for all IPA symbols
used in the Ukrainian phonetic engine (crates/core/src/data.rs + allophones).

Output format (compact array-per-phoneme):
{
  "source": "panphon 0.22.2",
  "features": ["syl", "son", ...],           // ordered feature names
  "descriptions": {"syl": "syllabic", ...},  // human-readable
  "phonemes": [
    {"ipa": "i", "f": ["+","+","-",...], "base": true},
    ...
  ]
}

Run from workspace root:
  python scripts/generate_ua_phonemes.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIBLING_PHONEMES = (
    _REPO_ROOT.parent.parent
    / "poetryEngine/phonetic-poetry-engine/IPA/phonemes.json"
)

# Fallback search paths
_SEARCH_PATHS = [
    _SIBLING_PHONEMES,
    Path(r"W:/Projects/poetryEngine/phonetic-poetry-engine/IPA/phonemes.json"),
]

_OUT_WEB = _REPO_ROOT / "packages/ua-stress-web/data/ua_phonemes.json"

# ---------------------------------------------------------------------------
# Canonical Ukrainian IPA symbol inventory
# (All symbols actually emitted by crates/core/)
# ---------------------------------------------------------------------------
UA_IPA_SYMBOLS: list[str] = [
    # ── Vowels ──────────────────────────────────────────────────────────────
    "i",      # і  — high front unrounded        (stable)
    "ɪ",      # и  — high-mid front unrounded    (reduces unstressed)
    "ɛ",      # е  — mid front unrounded          (reduces unstressed)
    "u",      # у  — high back rounded            (stable)
    "ɔ",      # о  — mid back rounded             (reduces unstressed)
    "ɑ",      # а  — low back unrounded           (stable)

    # ── Base consonants ─────────────────────────────────────────────────────
    "b",      # б
    "p",      # п
    "f",      # ф
    "ʋ",      # в  (labiodental approximant, default allophone)
    "m",      # м
    "d",      # д
    "t",      # т
    "z",      # з
    "s",      # с
    "n",      # н
    "l",      # л
    "r",      # р
    "ʒ",      # ж
    "ʃ",      # ш
    "ɡ",      # ґ
    "k",      # к
    "x",      # х
    "ɦ",      # г  (voiced glottal fricative)
    "j",      # й  (palatal glide)
    # ʔ (glottal stop) is NOT emitted — apostrophe is a boundary marker only

    # ── Affricates (Unicode tie-bar U+0361 = ͡) ─────────────────────────────
    "d\u0361z",   # дз
    "t\u0361s",   # ц
    "d\u0361ʒ",   # дж
    "t\u0361ʃ",   # ч

    # ── Palatalized (soft) consonants (U+02B2 = ʲ) ──────────────────────────
    "bʲ",   "pʲ",   "fʲ",   "ʋʲ",   "mʲ",
    "dʲ",   "tʲ",   "zʲ",   "sʲ",   "nʲ",   "lʲ",   "rʲ",
    "d\u0361zʲ",   "t\u0361sʲ",

    # ── /ʋ/ positional allophones ────────────────────────────────────────────
    "w",      # post-vocalic pre-consonant (e.g. правда → [prɑwdɑ])
    "u\u032f",  # u̯  word-final after vowel (e.g. кров → [krɔu̯])
]

FEATURE_ORDER = [
    "syl", "son", "cons", "cont", "delrel", "lat",
    "nas", "strid", "voi", "sg", "cg", "ant", "cor",
    "distr", "lab", "hi", "lo", "back", "round",
    "velaric", "tense", "long", "hitone", "hireg",
]

FEATURE_DESCRIPTIONS = {
    "syl":     "syllabic",
    "son":     "sonorant",
    "cons":    "consonantal",
    "cont":    "continuant",
    "delrel":  "delayed release",
    "lat":     "lateral",
    "nas":     "nasal",
    "strid":   "strident",
    "voi":     "voice",
    "sg":      "spread glottis",
    "cg":      "constricted glottis",
    "ant":     "anterior",
    "cor":     "coronal",
    "distr":   "distributed",
    "lab":     "labial",
    "hi":      "high",
    "lo":      "low",
    "back":    "back",
    "round":   "round",
    "velaric": "velaric",
    "tense":   "tense",
    "long":    "long",
    "hitone":  "high tone",
    "hireg":   "high register",
}


def _find_phonemes_json() -> Path:
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "phonemes.json not found. Expected at:\n"
        + "\n".join(f"  {p}" for p in _SEARCH_PATHS)
    )


def main() -> None:
    src_path = _find_phonemes_json()
    print(f"Reading Panphon data from: {src_path}")

    with open(src_path, encoding="utf-8") as f:
        panphon_data = json.load(f)

    # Build lookup: ipa → {feat: value, ..., is_base: bool}
    panphon_map: dict[str, dict] = {
        p["ipa"]: p for p in panphon_data["phonemes"]
    }

    phoneme_entries = []
    missing: list[str] = []

    for sym in UA_IPA_SYMBOLS:
        if sym not in panphon_map:
            missing.append(sym)
            continue
        p = panphon_map[sym]
        feat_values = [p["features"][feat]["value"] for feat in FEATURE_ORDER]
        phoneme_entries.append({
            "ipa":  sym,
            "f":    feat_values,
            "base": p["is_base"],
        })

    if missing:
        print(f"\n[WARN] {len(missing)} symbols not found in Panphon:", file=sys.stderr)
        for s in missing:
            print(f"  {repr(s)}", file=sys.stderr)

    output = {
        "source":       panphon_data["metadata"]["source"],
        "features":     FEATURE_ORDER,
        "descriptions": {k: FEATURE_DESCRIPTIONS[k] for k in FEATURE_ORDER},
        "encoding": {
            "+": "positively specified (present)",
            "-": "negatively specified (absent)",
            "0": "unspecified / not applicable",
        },
        "total":    len(phoneme_entries),
        "phonemes": phoneme_entries,
    }

    for out_path in [_OUT_WEB]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        size_kb = out_path.stat().st_size / 1024
        print(f"Written: {out_path}  ({size_kb:.1f} KB, {len(phoneme_entries)} phonemes)")


if __name__ == "__main__":
    main()
