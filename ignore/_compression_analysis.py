from __future__ import annotations

import gzip
import json
import lzma
from pathlib import Path

ROOT = Path(r"w:\Projects\poetykaAnalizerEngine\VersaSenseEngine\VersaSenseBackend")
ART = ROOT / "src" / "stress_prediction" / "lightgbm" / "artifacts" / "v1.4_optuna"


def mb(n: int) -> float:
    return n / (1024 * 1024)


def ratio(orig: int, comp: int) -> float:
    if orig == 0:
        return 0.0
    return comp / orig


def analyze_file(path: Path) -> dict:
    data = path.read_bytes()
    original = len(data)
    gz = gzip.compress(data, compresslevel=9)
    xz = lzma.compress(data, preset=9)
    return {
        "file": str(path),
        "original_bytes": original,
        "original_mb": round(mb(original), 3),
        "gzip_bytes": len(gz),
        "gzip_mb": round(mb(len(gz)), 3),
        "gzip_ratio": round(ratio(original, len(gz)), 4),
        "gzip_saving_pct": round((1 - ratio(original, len(gz))) * 100, 2),
        "xz_bytes": len(xz),
        "xz_mb": round(mb(len(xz)), 3),
        "xz_ratio": round(ratio(original, len(xz)), 4),
        "xz_saving_pct": round((1 - ratio(original, len(xz))) * 100, 2),
    }


def main() -> None:
    targets = [
        ART / "T0087" / "T0087.lgb",
        ART / "v1.4_results.json",
        ART / "v1.4_results.csv",
        ART / "optuna_study.db",
    ]
    report = {
        "targets": [],
        "missing": [],
    }

    for p in targets:
        if p.exists():
            report["targets"].append(analyze_file(p))
        else:
            report["missing"].append(str(p))

    all_size = 0
    all_files = []
    if ART.exists():
        for f in ART.rglob("*"):
            if f.is_file():
                s = f.stat().st_size
                all_size += s
                all_files.append((str(f), s))

    all_files.sort(key=lambda x: x[1], reverse=True)
    report["artifact_dir_total_mb"] = round(mb(all_size), 3)
    report["artifact_top10"] = [
        {"file": name, "size_mb": round(mb(size), 3)}
        for name, size in all_files[:10]
    ]

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
