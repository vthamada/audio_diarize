from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple


# -------------------------
# Config
# -------------------------
SPEAKERS_FILE = "speakers.txt"
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
TARGET_SPEAKER = "SPEAKER_00"  # ajuste aqui
MIN_SEGMENT_MS = 300
ADD_SILENCE_MS = 0  # 0 = sem silencia extra entre trechos
OUTPUT_WAV = "jarvis_dublador_raw.wav"


def _parse_speakers(path: str) -> Iterable[Tuple[float, float, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # expected: "start --> end | SPEAKER_XX"
            try:
                time_part, speaker = line.split("|", 1)
                start_str, end_str = time_part.split("-->")
                start = float(start_str.strip())
                end = float(end_str.strip())
                speaker = speaker.strip()
                yield start, end, speaker
            except Exception:
                continue


def _unique_speakers(segments: Iterable[Tuple[float, float, str]]) -> list[str]:
    seen = set()
    for _, _, spk in segments:
        if spk not in seen:
            seen.add(spk)
    return sorted(seen)


def main() -> None:
    if not os.path.exists(SPEAKERS_FILE):
        print(f"Erro: arquivo nao encontrado: {SPEAKERS_FILE}")
        sys.exit(1)
    if not os.path.exists(ORIGINAL_AUDIO_FILE):
        print(f"Erro: arquivo nao encontrado: {ORIGINAL_AUDIO_FILE}")
        sys.exit(1)

    segments = list(_parse_speakers(SPEAKERS_FILE))
    if not segments:
        print("Erro: speakers.txt vazio ou formato invalido.")
        sys.exit(1)

    if not TARGET_SPEAKER:
        print("Defina TARGET_SPEAKER no topo do arquivo.")
        print("Speakers encontrados:", ", ".join(_unique_speakers(segments)))
        sys.exit(1)

    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    with sf.SoundFile(ORIGINAL_AUDIO_FILE) as src:
        sr = src.samplerate
        channels = src.channels
        subtype = src.subtype

        min_samples = int((MIN_SEGMENT_MS / 1000.0) * sr)
        silence_samples = int((ADD_SILENCE_MS / 1000.0) * sr)
        silence = None
        if silence_samples > 0:
            silence = np.zeros((silence_samples, channels), dtype="float32")

        kept = 0
        with sf.SoundFile(
            OUTPUT_WAV, mode="w", samplerate=sr, channels=channels, subtype=subtype
        ) as out:
            for start, end, speaker in segments:
                if speaker != TARGET_SPEAKER:
                    continue
                start_frame = max(0, int(start * sr))
                end_frame = max(0, int(end * sr))
                if end_frame <= start_frame:
                    continue
                if (end_frame - start_frame) < min_samples:
                    continue

                src.seek(start_frame)
                data = src.read(end_frame - start_frame, dtype="float32", always_2d=True)
                if data.size == 0:
                    continue
                out.write(data)
                kept += 1
                if silence is not None:
                    out.write(silence)

    print(f"OK. Trechos extraidos: {kept}. Arquivo gerado: {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
