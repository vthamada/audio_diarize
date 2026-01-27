from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple


# -------------------------
# Config
# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
CACHE_DIR = os.path.join(".cache", "pyannote")
DIARIZE_AUDIO_FILE = "podcast_completo_16k_mono.wav"
OUTPUT_OVERLAP_FILE = "overlaps.txt"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
OVERLAP_DILATION_SECONDS = 0.2
MIN_OVERLAP_SECONDS = 0.1


def _dilate(seg: Tuple[float, float], delta: float) -> Tuple[float, float]:
    start, end = seg
    return max(0.0, start - delta), end + delta


def _collect_segments(result) -> list[Tuple[float, float]]:
    if hasattr(result, "get_timeline"):
        timeline = result.get_timeline()
        return [(s.start, s.end) for s in timeline.itersegments()]
    if hasattr(result, "itersegments"):
        return [(s.start, s.end) for s in result.itersegments()]
    raise RuntimeError("Formato de saida de overlap nao reconhecido.")


def main() -> None:
    if not HF_TOKEN:
        print("Erro: defina HF_TOKEN no ambiente.")
        sys.exit(1)
    if not os.path.exists(DIARIZE_AUDIO_FILE):
        print(f"Erro: arquivo nao encontrado: {DIARIZE_AUDIO_FILE}")
        sys.exit(1)

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        print("Erro: pyannote.audio nao instalado corretamente.")
        print(f"Detalhe: {exc}")
        sys.exit(1)

    pipeline = Pipeline.from_pretrained(
        OVERLAP_MODEL, token=HF_TOKEN, cache_dir=CACHE_DIR
    )

    overlap = pipeline(DIARIZE_AUDIO_FILE)
    segments = _collect_segments(overlap)

    if not segments:
        print("Aviso: nenhum overlap detectado.")

    # dilate + filter small
    cleaned: list[Tuple[float, float]] = []
    for seg in segments:
        start, end = _dilate(seg, OVERLAP_DILATION_SECONDS)
        if (end - start) < MIN_OVERLAP_SECONDS:
            continue
        cleaned.append((start, end))

    cleaned.sort(key=lambda x: x[0])

    with open(OUTPUT_OVERLAP_FILE, "w", encoding="utf-8") as f:
        for start, end in cleaned:
            f.write(f"{start:.3f}\t{end:.3f}\n")

    print(f"OK. Overlaps detectados: {len(cleaned)}. Salvo em {OUTPUT_OVERLAP_FILE}")


if __name__ == "__main__":
    main()
