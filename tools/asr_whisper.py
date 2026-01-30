from __future__ import annotations

import csv
import os
from pathlib import Path

# -------------------------
# Config
# -------------------------
DATASET_DIR = Path("datasets/gpt_sovits")
WAV_DIR = DATASET_DIR / "wavs"
METADATA = DATASET_DIR / "metadata.csv"
LANG = os.getenv("ASR_LANG", "pt")
MODEL_SIZE = os.getenv("ASR_MODEL", "medium")  # tiny/base/small/medium/large
DEVICE = os.getenv("ASR_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cpu")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE", "float16")  # float16/int8/int8_float16


def _load_rows():
    rows = []
    with METADATA.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _save_rows(rows):
    with METADATA.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["file", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _asr_faster_whisper(wav_path: Path) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, _info = model.transcribe(str(wav_path), language=LANG, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


def _asr_openai_whisper(wav_path: Path) -> str:
    import whisper
    model = whisper.load_model(MODEL_SIZE)
    result = model.transcribe(str(wav_path), language=LANG)
    return result.get("text", "").strip()


def main() -> None:
    if not METADATA.exists():
        raise SystemExit("metadata.csv not found. Run prepare_gpt_sovits_dataset.py first.")
    if not WAV_DIR.exists():
        raise SystemExit("wavs directory not found.")

    rows = _load_rows()
    updated = 0

    # Prefer faster-whisper if installed, else fallback to openai-whisper
    use_faster = False
    try:
        import faster_whisper  # noqa: F401
        use_faster = True
    except Exception:
        use_faster = False

    for row in rows:
        if row.get("text"):
            continue
        wav_path = WAV_DIR / row["file"]
        if not wav_path.exists():
            continue
        if use_faster:
            text = _asr_faster_whisper(wav_path)
        else:
            text = _asr_openai_whisper(wav_path)
        row["text"] = text
        updated += 1
        if updated % 50 == 0:
            _save_rows(rows)

    _save_rows(rows)
    print(f"OK. Transcribed: {updated} (faster_whisper={use_faster})")


if __name__ == "__main__":
    main()
