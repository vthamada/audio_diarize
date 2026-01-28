from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from typing import Iterable, Tuple


# -------------------------
load_dotenv()

try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        def _ta_list_audio_backends() -> list[str]:
            return []
        torchaudio.list_audio_backends = _ta_list_audio_backends
except Exception:
    pass

# Config
# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY", "")
CACHE_DIR = os.path.join(".cache", "pyannote")
DIARIZE_AUDIO_FILE = "podcast_completo_16k_mono.wav"
OUTPUT_OVERLAP_FILE = "overlaps.txt"
SPEAKERS_FILE = "speakers.txt"
OVERLAP_MODEL = "pyannote/overlapped-speech-detection"
IDENTIFY_JSON = "identify_output.json"
IDENTIFY_JSON_NON_EXCLUSIVE = "identify_output_non_exclusive.json"
OVERLAP_DILATION_SECONDS = 0.2
MIN_OVERLAP_SECONDS = 0.1
FALLBACK_TO_SPEAKERS = True
PREFER_OVERLAP_MODEL = os.getenv("PREFER_OVERLAP_MODEL", "0") == "1"
REQUIRE_OVERLAP_MODEL = os.getenv("REQUIRE_OVERLAP_MODEL", "0") == "1"




def _parse_speakers(path: str):
    segments = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    time_part, speaker = line.split("|", 1)
                    start_str, end_str = time_part.split("-->")
                    start = float(start_str.strip())
                    end = float(end_str.strip())
                    speaker = speaker.strip()
                    if end > start:
                        segments.append((start, end, speaker))
                except Exception:
                    continue
    except Exception:
        return []
    return segments




def _load_identify_segments(path: str):
    if not os.path.exists(path):
        return []
    try:
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []

    diar = data.get("exclusiveDiarization") or data.get("diarization") or data
    segs = diar.get("segments") if isinstance(diar, dict) else diar
    if not isinstance(segs, list):
        return []
    out = []
    for s in segs:
        if isinstance(s, dict):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
            speaker = s.get("speaker") or s.get("label") or s.get("name") or "SPEAKER"
        elif isinstance(s, (list, tuple)) and len(s) >= 2:
            start = float(s[0]); end = float(s[1]);
            speaker = str(s[2]) if len(s) >= 3 else "SPEAKER"
        else:
            continue
        if end > start:
            out.append((start, end, str(speaker)))
    return out


def _dilate(seg: Tuple[float, float], delta: float) -> Tuple[float, float]:
    start, end = seg
    return max(0.0, start - delta), end + delta




def _overlaps_from_speakers(segments):
    # sweep line to find overlaps between different speakers
    events = []
    for start, end, spk in segments:
        events.append((start, 1, spk, end))
        events.append((end, -1, spk, end))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = []
    overlaps = []
    prev_t = None

    for t, typ, spk, end in events:
        if prev_t is not None and t > prev_t and len(active) >= 2:
            overlaps.append((prev_t, t))
        if typ == 1:
            active.append((spk, end))
        else:
            # remove one instance of this speaker
            for i, (s, e) in enumerate(active):
                if s == spk and abs(e - end) < 1e-6:
                    active.pop(i)
                    break
        prev_t = t

    return overlaps


def _collect_segments(result) -> list[Tuple[float, float]]:
    if hasattr(result, "get_timeline"):
        timeline = result.get_timeline()
        return [(s.start, s.end) for s in timeline.itersegments()]
    if hasattr(result, "itersegments"):
        return [(s.start, s.end) for s in result.itersegments()]
    raise RuntimeError("Formato de saida de overlap nao reconhecido.")


def _patch_torchaudio_backend() -> None:
    try:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            def _ta_list_audio_backends() -> list[str]:
                return []
            torchaudio.list_audio_backends = _ta_list_audio_backends
    except Exception:
        return

def main() -> None:
    if not HF_TOKEN:
        print("Erro: defina HF_TOKEN no ambiente.")
        sys.exit(1)
    if not os.path.exists(DIARIZE_AUDIO_FILE):
        print(f"Erro: arquivo nao encontrado: {DIARIZE_AUDIO_FILE}")
        sys.exit(1)

    segments = []

    # 1) Prefer overlap model if requested
    if PREFER_OVERLAP_MODEL or REQUIRE_OVERLAP_MODEL:
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                OVERLAP_MODEL, token=HF_TOKEN, cache_dir=CACHE_DIR
            )
            overlap = pipeline(DIARIZE_AUDIO_FILE)
            segments = _collect_segments(overlap)
            print(f"Overlaps via pyannote model: {len(segments)}")
        except Exception as exc:
            if REQUIRE_OVERLAP_MODEL:
                raise
            print(f"Aviso: overlap model falhou: {exc}")

    # 2) Prefer Precision-2 JSON if available
    if not segments and os.path.exists(IDENTIFY_JSON_NON_EXCLUSIVE):
        spk_segments = _load_identify_segments(IDENTIFY_JSON_NON_EXCLUSIVE)
        if spk_segments:
            segments = _overlaps_from_speakers(spk_segments)
            print(f"Usando identify_output_non_exclusive.json para overlaps: {len(segments)}")

    if not segments and os.path.exists(IDENTIFY_JSON):
        spk_segments = _load_identify_segments(IDENTIFY_JSON)
        if spk_segments:
            segments = _overlaps_from_speakers(spk_segments)
            print(f"Usando identify_output.json (Precision-2) para overlaps: {len(segments)}")

    # 3) Try pyannote overlap model if not attempted yet
    if not segments and not (PREFER_OVERLAP_MODEL or REQUIRE_OVERLAP_MODEL):
        try:
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(
                OVERLAP_MODEL, token=HF_TOKEN, cache_dir=CACHE_DIR
            )
            overlap = pipeline(DIARIZE_AUDIO_FILE)
            segments = _collect_segments(overlap)
            print(f"Overlaps via pyannote model: {len(segments)}")
        except Exception as exc:
            print(f"Aviso: overlap model falhou: {exc}")

    # 4) Final fallback: speakers.txt
    if not segments:
        spk_segments = _parse_speakers(SPEAKERS_FILE)
        segments = _overlaps_from_speakers(spk_segments)
        print(f"Usando speakers.txt para overlaps: {len(segments)}")

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
