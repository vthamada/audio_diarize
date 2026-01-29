from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple


# -------------------------
# Config (modo ultra conservador)
# -------------------------
SPEAKERS_FILE = os.getenv("REF_SPEAKERS_FILE", "speakers.txt").strip()
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
TARGET_SPEAKER = os.getenv("REF_TARGET_SPEAKER", "SPEAKER_01").strip()
OUTPUT_DIR = "ref_speaker"
MAX_REFERENCES = int(os.getenv("REF_MAX_REFERENCES", "10"))
REF_SECONDS = float(os.getenv("REF_SECONDS", "5.0"))
MIN_SEGMENT_SECONDS = float(os.getenv("REF_MIN_SEGMENT_SECONDS", "40.0"))
EDGE_TRIM_SECONDS = float(os.getenv("REF_EDGE_TRIM_SECONDS", "2.0"))
STRICT_EXTRA_SECONDS = float(os.getenv("REF_STRICT_EXTRA_SECONDS", "6.0"))
OVERLAP_FILE = os.getenv("REF_OVERLAP_FILE", "overlaps.txt")
OVERLAP_MAX_RATIO = float(os.getenv("REF_OVERLAP_MAX_RATIO", "0.0"))
MIN_SPEECH_RATIO = float(os.getenv("REF_MIN_SPEECH_RATIO", "0.0"))
SPEECH_THRESHOLD_DB = float(os.getenv("REF_SPEECH_THRESHOLD_DB", "-40.0"))


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


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_overlaps(path: str) -> list[Tuple[float, float]]:
    if not path or not os.path.exists(path):
        return []
    overlaps: list[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
            except Exception:
                continue
            if end > start:
                overlaps.append((start, end))
    overlaps.sort(key=lambda x: x[0])
    return overlaps


def _overlap_ratio(
    seg_start: float, seg_end: float, overlaps: list[Tuple[float, float]]
) -> float:
    if seg_end <= seg_start or not overlaps:
        return 0.0
    total = 0.0
    for ov_start, ov_end in overlaps:
        if ov_end <= seg_start:
            continue
        if ov_start >= seg_end:
            break
        inter_start = max(seg_start, ov_start)
        inter_end = min(seg_end, ov_end)
        if inter_end > inter_start:
            total += inter_end - inter_start
    return total / (seg_end - seg_start)


def main() -> None:
    if not os.path.exists(SPEAKERS_FILE):
        print(f"Erro: arquivo nao encontrado: {SPEAKERS_FILE}")
        sys.exit(1)
    if not os.path.exists(ORIGINAL_AUDIO_FILE):
        print(f"Erro: arquivo nao encontrado: {ORIGINAL_AUDIO_FILE}")
        sys.exit(1)
    if not TARGET_SPEAKER:
        print("Defina TARGET_SPEAKER no topo do arquivo.")
        sys.exit(1)

    try:
        import soundfile as sf
        import numpy as np
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    min_required = max(
        MIN_SEGMENT_SECONDS,
        REF_SECONDS + (2.0 * EDGE_TRIM_SECONDS) + STRICT_EXTRA_SECONDS,
    )

    overlaps = _load_overlaps(OVERLAP_FILE)

    segments = []
    for start, end, spk in _parse_speakers(SPEAKERS_FILE):
        if spk != TARGET_SPEAKER:
            continue
        if end <= start:
            continue
        if (end - start) < min_required:
            continue
        if overlaps:
            ratio = _overlap_ratio(start, end, overlaps)
            if ratio > OVERLAP_MAX_RATIO:
                continue
        segments.append((start, end))

    if not segments:
        print("Erro: nao ha segmentos suficientes para referencias.")
        sys.exit(1)

    segments.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    _ensure_dir(OUTPUT_DIR)

    report_path = os.path.join(OUTPUT_DIR, "_ref_candidates.csv")
    with open(report_path, "w", encoding="utf-8") as report:
        report.write(
            "file,orig_start_s,orig_end_s,trim_start_s,trim_end_s,"
            "final_start_s,final_end_s,duration_s\n"
        )

        with sf.SoundFile(ORIGINAL_AUDIO_FILE) as src:
            sr = src.samplerate
            channels = src.channels
            subtype = src.subtype

            ref_samples = int(REF_SECONDS * sr)
            edge_trim = int(EDGE_TRIM_SECONDS * sr)

            exported = 0
            for idx, (start, end) in enumerate(segments):
                if exported >= MAX_REFERENCES:
                    break

                start_frame = max(0, int(start * sr) + edge_trim)
                end_frame = max(0, int(end * sr) - edge_trim)
                if end_frame <= start_frame:
                    continue

                seg_len = end_frame - start_frame
                if seg_len < ref_samples:
                    continue

                center = start_frame + (seg_len // 2)
                ref_start = max(0, center - (ref_samples // 2))
                ref_end = ref_start + ref_samples

                src.seek(ref_start)
                data = src.read(ref_end - ref_start, dtype="float32", always_2d=True)
                if data.size == 0:
                    continue
                if MIN_SPEECH_RATIO > 0.0:
                    mono = data.mean(axis=1)
                    thresh = 10 ** (SPEECH_THRESHOLD_DB / 20.0)
                    speech_ratio = float((abs(mono) >= thresh).mean())
                    if speech_ratio < MIN_SPEECH_RATIO:
                        continue

                start_ms = int((ref_start / sr) * 1000)
                end_ms = int((ref_end / sr) * 1000)
                out_name = f"ref_{exported+1:02d}_{start_ms:09d}_{end_ms:09d}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                with sf.SoundFile(
                    out_path,
                    mode="w",
                    samplerate=sr,
                    channels=channels,
                    subtype=subtype,
                ) as out:
                    out.write(data)

                report.write(
                    f"{out_name},{start:.3f},{end:.3f},"
                    f"{(int(start * sr) + edge_trim)/sr:.3f},"
                    f"{(int(end * sr) - edge_trim)/sr:.3f},"
                    f"{ref_start/sr:.3f},{ref_end/sr:.3f},"
                    f"{REF_SECONDS:.3f}\n"
                )

                exported += 1

    print(f"OK. Referencias geradas: {exported} em {OUTPUT_DIR}\\")
    print(f"Relatorio: {report_path}")


if __name__ == "__main__":
    main()
