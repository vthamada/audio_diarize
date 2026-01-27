from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple


# -------------------------
# Config (modo ultra conservador)
# -------------------------
SPEAKERS_FILE = "speakers.txt"
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
TARGET_SPEAKER = "SPEAKER_02"
OUTPUT_DIR = "ref_speaker"
MAX_REFERENCES = 6
REF_SECONDS = 5.0
MIN_SEGMENT_SECONDS = 40.0
EDGE_TRIM_SECONDS = 2.0
STRICT_EXTRA_SECONDS = 6.0


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
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    min_required = max(
        MIN_SEGMENT_SECONDS,
        REF_SECONDS + (2.0 * EDGE_TRIM_SECONDS) + STRICT_EXTRA_SECONDS,
    )

    segments = []
    for start, end, spk in _parse_speakers(SPEAKERS_FILE):
        if spk != TARGET_SPEAKER:
            continue
        if end <= start:
            continue
        if (end - start) < min_required:
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
