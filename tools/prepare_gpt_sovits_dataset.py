from __future__ import annotations

import csv
import shutil
from pathlib import Path

# -------------------------
# Config
# -------------------------
SOURCE_DIR = Path("segments_clean")
OUT_DIR = Path("datasets/gpt_sovits")
WAV_DIR = OUT_DIR / "wavs"
METADATA = OUT_DIR / "metadata.csv"


def main() -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    wavs = list(SOURCE_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit("No wavs found in segments_clean")

    rows = []
    for wav in wavs:
        dst = WAV_DIR / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
        # transcript will be filled by ASR
        rows.append([wav.name, ""])

    with METADATA.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "text"])
        writer.writerows(rows)

    print(f"OK. Dataset prepared: {OUT_DIR}")


if __name__ == "__main__":
    main()
