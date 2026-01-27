from __future__ import annotations

import csv
import os
import sys


# -------------------------
# Config
# -------------------------
REVIEW_CSV = os.path.join("segments", "review_list.csv")
OUTPUT_WAV = "jarvis_dublador_clean.wav"
ADD_SILENCE_MS = 0


def _is_approved(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"1", "true", "yes", "ok", "approved", "y", "sim"}


def main() -> None:
    if not os.path.exists(REVIEW_CSV):
        print(f"Erro: arquivo nao encontrado: {REVIEW_CSV}")
        sys.exit(1)

    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(REVIEW_CSV))
    rows = []
    with open(REVIEW_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _is_approved(row.get("approved", "")):
                rows.append(row)

    if not rows:
        print("Nenhum trecho aprovado. Marque a coluna 'approved' com 1/yes/ok.")
        sys.exit(1)

    with sf.SoundFile(os.path.join(base_dir, rows[0]["file"])) as first:
        sr = first.samplerate
        channels = first.channels
        subtype = first.subtype

    silence = None
    silence_samples = int((ADD_SILENCE_MS / 1000.0) * sr)
    if silence_samples > 0:
        silence = np.zeros((silence_samples, channels), dtype="float32")

    kept = 0
    with sf.SoundFile(
        OUTPUT_WAV, mode="w", samplerate=sr, channels=channels, subtype=subtype
    ) as out:
        for row in rows:
            rel_path = row.get("file", "")
            if not rel_path:
                continue
            if os.path.isabs(rel_path):
                seg_path = rel_path
            else:
                seg_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(seg_path):
                print(f"Aviso: arquivo nao encontrado: {seg_path}")
                continue
            with sf.SoundFile(seg_path) as seg:
                data = seg.read(dtype="float32", always_2d=True)
                if data.size == 0:
                    continue
                out.write(data)
                kept += 1
                if silence is not None:
                    out.write(silence)

    print(f"OK. Trechos concatenados: {kept}. Arquivo: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
