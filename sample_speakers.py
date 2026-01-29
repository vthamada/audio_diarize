from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Iterable, Tuple


def _default_speakers_file() -> str:
    return "speakers_exclusive.txt" if os.path.exists("speakers_exclusive.txt") else "speakers.txt"

# -------------------------
# Config
# -------------------------
SPEAKERS_FILE = os.getenv("SAMPLE_SPEAKERS_FILE", _default_speakers_file()).strip()
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
OUTPUT_DIR = "samples"
TARGET_SPEAKER = os.getenv("SAMPLE_TARGET_SPEAKER", "").strip()
SAMPLES_PER_SPEAKER = int(os.getenv("SAMPLE_SAMPLES_PER_SPEAKER", "3"))
SAMPLE_SECONDS = float(os.getenv("SAMPLE_SECONDS", "4.0"))
MIN_SEGMENT_SECONDS = float(os.getenv("SAMPLE_MIN_SEGMENT_SECONDS", "20.0"))
EDGE_TRIM_SECONDS = float(os.getenv("SAMPLE_EDGE_TRIM_SECONDS", "1.0"))
STRICT_EXTRA_SECONDS = float(os.getenv("SAMPLE_STRICT_EXTRA_SECONDS", "4.0"))


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


def _write_report(
    output_dir: str,
    config: dict[str, str],
    audio_info: dict[str, str],
    per_speaker: dict[str, tuple[int, int, list[str]]],
    samples_meta: list[tuple[str, float, float, float, float, float, float, str]],
) -> None:
    config_path = os.path.join(output_dir, "_config.txt")
    summary_path = os.path.join(output_dir, "_summary.txt")
    samples_path = os.path.join(output_dir, "_samples.csv")

    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
        for key, value in audio_info.items():
            f.write(f"{key}={value}\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("speaker,available_segments,exported_samples,files\n")
        for spk in sorted(per_speaker.keys()):
            available, exported, files = per_speaker[spk]
            files_str = "|".join(files)
            f.write(f"{spk},{available},{exported},{files_str}\n")

    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(
            "speaker,orig_start_s,orig_end_s,trim_start_s,trim_end_s,"
            "final_start_s,final_end_s,duration_s,file\n"
        )
        for row in samples_meta:
            f.write(
                f"{row[0]},{row[1]:.3f},{row[2]:.3f},"
                f"{row[3]:.3f},{row[4]:.3f},"
                f"{row[5]:.3f},{row[6]:.3f},"
                f"{row[6]-row[5]:.3f},{row[7]}\n"
            )


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

    try:
        import soundfile as sf
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    min_required = max(
        MIN_SEGMENT_SECONDS,
        SAMPLE_SECONDS + (2.0 * EDGE_TRIM_SECONDS) + STRICT_EXTRA_SECONDS,
    )

    segments_by_spk: dict[str, list[Tuple[float, float]]] = defaultdict(list)
    for start, end, spk in segments:
        if end <= start:
            continue
        if TARGET_SPEAKER and spk != TARGET_SPEAKER:
            continue
        if (end - start) < min_required:
            continue
        segments_by_spk[spk].append((start, end))

    if not segments_by_spk:
        print("Erro: nao ha segmentos validos para amostragem.")
        sys.exit(1)

    _ensure_dir(OUTPUT_DIR)

    config = {
        "SPEAKERS_FILE": SPEAKERS_FILE,
        "ORIGINAL_AUDIO_FILE": ORIGINAL_AUDIO_FILE,
        "OUTPUT_DIR": OUTPUT_DIR,
        "TARGET_SPEAKER": TARGET_SPEAKER or "ALL",
        "SAMPLES_PER_SPEAKER": str(SAMPLES_PER_SPEAKER),
        "SAMPLE_SECONDS": str(SAMPLE_SECONDS),
        "MIN_SEGMENT_SECONDS": str(MIN_SEGMENT_SECONDS),
        "EDGE_TRIM_SECONDS": str(EDGE_TRIM_SECONDS),
        "STRICT_EXTRA_SECONDS": str(STRICT_EXTRA_SECONDS),
        "MIN_REQUIRED_SECONDS": str(min_required),
    }

    with sf.SoundFile(ORIGINAL_AUDIO_FILE) as src:
        sr = src.samplerate
        channels = src.channels
        subtype = src.subtype
        max_samples = int(SAMPLE_SECONDS * sr)
        edge_trim = int(EDGE_TRIM_SECONDS * sr)

        audio_info = {
            "SAMPLE_RATE": str(sr),
            "CHANNELS": str(channels),
            "SUBTYPE": str(subtype),
        }

        per_speaker: dict[str, tuple[int, int, list[str]]] = {}
        total_written = 0
        samples_meta: list[tuple[str, float, float, float, float, float, float, str]] = []
        for spk, segs in sorted(segments_by_spk.items()):
            # pick longest segments first
            segs_sorted = sorted(segs, key=lambda x: (x[1] - x[0]), reverse=True)
            written = 0
            files: list[str] = []
            for idx, (start, end) in enumerate(segs_sorted):
                if written >= SAMPLES_PER_SPEAKER:
                    break
                start_frame = max(0, int(start * sr) + edge_trim)
                end_frame = max(0, int(end * sr) - edge_trim)
                if end_frame <= start_frame:
                    continue
                seg_len = end_frame - start_frame
                if seg_len <= 0:
                    continue
                # take a centered chunk to avoid boundary bleed
                if seg_len > max_samples:
                    center = start_frame + seg_len // 2
                    start_frame = max(0, center - max_samples // 2)
                    end_frame = start_frame + max_samples
                    nframes = max_samples
                else:
                    nframes = seg_len
                if nframes <= 0:
                    continue

                src.seek(start_frame)
                data = src.read(nframes, dtype="float32", always_2d=True)
                if data.size == 0:
                    continue

                out_name = f"{spk}_{idx+1:04d}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                with sf.SoundFile(
                    out_path,
                    mode="w",
                    samplerate=sr,
                    channels=channels,
                    subtype=subtype,
                ) as out:
                    out.write(data)
                written += 1
                files.append(out_name)
                total_written += 1
                trim_start_s = (int(start * sr) + edge_trim) / sr
                trim_end_s = (int(end * sr) - edge_trim) / sr
                final_start_s = start_frame / sr
                final_end_s = end_frame / sr
                samples_meta.append(
                    (
                        spk,
                        float(start),
                        float(end),
                        float(trim_start_s),
                        float(trim_end_s),
                        float(final_start_s),
                        float(final_end_s),
                        out_name,
                    )
                )

            per_speaker[spk] = (len(segs_sorted), written, files)

        _write_report(OUTPUT_DIR, config, audio_info, per_speaker, samples_meta)

        print(f"OK. Amostras geradas: {total_written} em {OUTPUT_DIR}\\")
        print(f"Config salvo em: {os.path.join(OUTPUT_DIR, '_config.txt')}")
        print(f"Resumo salvo em: {os.path.join(OUTPUT_DIR, '_summary.txt')}")
        print(f"Detalhes por arquivo: {os.path.join(OUTPUT_DIR, '_samples.csv')}")


if __name__ == "__main__":
    main()
