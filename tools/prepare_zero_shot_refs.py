from __future__ import annotations

import csv
import shutil
from pathlib import Path

# -------------------------
# Config
# -------------------------
SOURCE_SEGMENTS_DIR = Path("segments_clean")
FALLBACK_SEGMENTS_DIR = Path("segments")
REVIEW_CSV = Path("segments/review_list.csv")
OUTPUT_DIR = Path("refs_zero_shot")
NUM_REFS = 5


def _load_review_scores():
    if not REVIEW_CSV.exists():
        return []
    rows = []
    with REVIEW_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sim = float(row.get("sim_min", ""))
            except Exception:
                sim = -1.0
            rows.append((row.get("file", ""), sim))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = _load_review_scores()
    candidates = [f for f, _s in scores if f]

    if not candidates:
        # fallback: take any wavs from segments_clean
        candidates = [p.name for p in SOURCE_SEGMENTS_DIR.glob("*.wav")]

    picked = 0
    for fname in candidates:
        if picked >= NUM_REFS:
            break
        src = SOURCE_SEGMENTS_DIR / fname
        if not src.exists():
            src = FALLBACK_SEGMENTS_DIR / fname
        if not src.exists():
            continue
        dst = OUTPUT_DIR / fname
        shutil.copy2(src, dst)
        picked += 1

    print(f"OK. Refs copied: {picked} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
