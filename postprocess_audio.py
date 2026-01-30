from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# -------------------------
# Config
# -------------------------
INPUT_DIR = "segments_new"  # folder with clean wavs
OUTPUT_DIR = "segments_clean"
INPUT_WAV = ""  # optional: process single wav instead of folder

TRIM_SILENCE = True
SILENCE_THRESHOLD_DB = -40  # lower = more aggressive trimming
SILENCE_MIN_DURATION = 0.4  # seconds

NORMALIZE_MODE = "peak"  # "peak" or "loudnorm" or "none"
PEAK_TARGET_DB = -1.0
LOUDNORM_TARGET = -20.0  # LUFS

FFMPEG_BIN = "ffmpeg"  # relies on PATH


def _ffmpeg_available() -> bool:
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _build_filter() -> str:
    filters = []
    if TRIM_SILENCE:
        # remove leading/trailing silence
        filters.append(
            f"silenceremove=start_periods=1:start_threshold={SILENCE_THRESHOLD_DB}dB:"
            f"start_duration={SILENCE_MIN_DURATION}:"
            f"stop_periods=1:stop_threshold={SILENCE_THRESHOLD_DB}dB:"
            f"stop_duration={SILENCE_MIN_DURATION}"
        )

    if NORMALIZE_MODE == "peak":
        filters.append(f"alimiter=limit={PEAK_TARGET_DB}dB")
    elif NORMALIZE_MODE == "loudnorm":
        filters.append(f"loudnorm=I={LOUDNORM_TARGET}:TP=-1.5:LRA=11")

    return ",".join(filters)


def _process_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    filt = _build_filter()
    cmd = [FFMPEG_BIN, "-y", "-i", str(src)]
    if filt:
        cmd += ["-af", filt]
    cmd += ["-c:a", "pcm_s16le", str(dst)]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Erro processando {src.name}")
        print(result.stderr.decode(errors="ignore"))
        return


def main() -> None:
    if not _ffmpeg_available():
        print("Erro: ffmpeg nao encontrado no PATH.")
        sys.exit(1)

    if INPUT_WAV:
        src = Path(INPUT_WAV)
        if not src.exists():
            print(f"Erro: arquivo nao encontrado: {src}")
            sys.exit(1)
        dst = Path(OUTPUT_DIR) / src.name
        _process_file(src, dst)
        print(f"OK: {dst}")
        return

    src_dir = Path(INPUT_DIR)
    if not src_dir.exists():
        print(f"Erro: pasta nao encontrada: {src_dir}")
        sys.exit(1)

    wavs = list(src_dir.glob("*.wav"))
    if not wavs:
        print(f"Nenhum wav encontrado em {src_dir}")
        sys.exit(1)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav in wavs:
        dst = out_dir / wav.name
        _process_file(wav, dst)

    print(f"OK. Processados: {len(wavs)} -> {out_dir}")


if __name__ == "__main__":
    main()
