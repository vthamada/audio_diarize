from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple


# -------------------------
# Config
# -------------------------
SPEAKERS_FILE = os.getenv("EXTRACT_SPEAKERS_FILE", "speakers_exclusive.txt")
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
TARGET_SPEAKER = os.getenv("EXTRACT_TARGET_SPEAKER", "SPEAKER_01")  # foco no dublador
MIN_SEGMENT_SECONDS = 2.5
EDGE_TRIM_SECONDS = 0.5
ADD_SILENCE_MS = 0  # 0 = sem silencia extra entre trechos
OUTPUT_WAV = os.getenv("EXTRACT_OUTPUT_WAV", "jarvis_dublador_raw.wav")
EXPORT_INDIVIDUAL = True
INDIVIDUAL_DIR = os.getenv("EXTRACT_SEGMENTS_DIR", "segments")
INDIVIDUAL_PREFIX = "segment"
REVIEW_CSV = os.path.join(INDIVIDUAL_DIR, "review_list.csv")
USE_SIMILARITY_FILTER = True
REFERENCE_DIR = "ref_speaker"
REFERENCE_MAX_SECONDS = 15.0
EMBED_SAMPLE_RATE = 16000
EMBED_SECONDS = 3.0
EMBED_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.65
SIMILARITY_CSV = "segments_similarity.csv"
EMBED_CACHE_DIR = os.path.join(".cache", "speechbrain_spkrec")
OVERLAP_FILE = "overlaps.txt"
OVERLAP_MAX_RATIO = 0.05


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
    seg_start: float, seg_end: float, overlaps: list[Tuple[float, float]], idx: int
) -> Tuple[float, int]:
    if seg_end <= seg_start:
        return 0.0, idx
    total = 0.0
    while idx < len(overlaps) and overlaps[idx][1] <= seg_start:
        idx += 1
    j = idx
    while j < len(overlaps) and overlaps[j][0] < seg_end:
        ov_start, ov_end = overlaps[j]
        inter_start = max(seg_start, ov_start)
        inter_end = min(seg_end, ov_end)
        if inter_end > inter_start:
            total += inter_end - inter_start
        if ov_end >= seg_end:
            break
        j += 1
    ratio = total / (seg_end - seg_start)
    return ratio, idx


def _patch_hf_hub_download() -> None:
    try:
        import huggingface_hub
        import inspect
    except Exception:
        return

    try:
        sig = inspect.signature(huggingface_hub.hf_hub_download)
        if "use_auth_token" in sig.parameters:
            return
    except Exception:
        return

    orig = huggingface_hub.hf_hub_download

    def _hf_hub_download_shim(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        try:
            return orig(*args, **kwargs)
        except Exception as exc:
            filename = kwargs.get("filename")
            if filename == "custom.py":
                cache_dir = kwargs.get("cache_dir") or os.path.join(".cache", "hf_custom")
                os.makedirs(cache_dir, exist_ok=True)
                local_path = os.path.join(cache_dir, "custom.py")
                if not os.path.exists(local_path):
                    with open(local_path, "w", encoding="utf-8") as f:
                        f.write("# empty custom module\n")
                return local_path
            raise exc

    huggingface_hub.hf_hub_download = _hf_hub_download_shim


def _patch_speechbrain_symlinks() -> None:
    try:
        import speechbrain.utils.fetching as sb_fetching
    except Exception:
        return
    try:
        local_strategy = sb_fetching.LocalStrategy.COPY
    except Exception:
        local_strategy = "copy"

    try:
        orig_fetch = sb_fetching.fetch
    except Exception:
        return

    def _fetch_wrapper(*args, **kwargs):
        if "local_strategy" not in kwargs or kwargs["local_strategy"] is None:
            kwargs["local_strategy"] = local_strategy
        else:
            try:
                if kwargs["local_strategy"] == sb_fetching.LocalStrategy.SYMLINK:
                    kwargs["local_strategy"] = local_strategy
            except Exception:
                if str(kwargs["local_strategy"]).lower().endswith("symlink"):
                    kwargs["local_strategy"] = local_strategy
        return orig_fetch(*args, **kwargs)

    sb_fetching.fetch = _fetch_wrapper


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

    use_similarity = USE_SIMILARITY_FILTER
    if use_similarity:
        _patch_hf_hub_download()
        try:
            import torch
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                def _ta_list_audio_backends() -> list[str]:
                    return []
                torchaudio.list_audio_backends = _ta_list_audio_backends  # type: ignore[attr-defined]
            _patch_speechbrain_symlinks()
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as exc:
            print("Erro: dependencias para similarity filter nao disponiveis.")
            print(f"Detalhe: {exc}")
            sys.exit(1)

        if not os.path.exists(REFERENCE_DIR):
            print(f"Erro: pasta de referencia nao encontrada: {REFERENCE_DIR}")
            sys.exit(1)

    similarity_log = None
    if use_similarity and SIMILARITY_CSV:
        similarity_log = open(SIMILARITY_CSV, "w", encoding="utf-8")
        similarity_log.write(
            "segment,start_s,end_s,duration_s,chunk_sims_min,chunk_sims_avg,chunk_sims_max,kept\n"
        )

    ref_embedding = None
    resamplers = {}
    device = "cpu"
    model = None
    if use_similarity:
        os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=EMBED_CACHE_DIR,
            run_opts={"device": device},
        )
        model.eval()

        ref_files = [
            os.path.join(REFERENCE_DIR, f)
            for f in os.listdir(REFERENCE_DIR)
            if f.lower().endswith(".wav")
        ]
        if not ref_files:
            print(f"Erro: coloque WAVs limpos em {REFERENCE_DIR}")
            sys.exit(1)

        ref_embs = []
        for ref_path in sorted(ref_files):
            data, sr = sf.read(ref_path, dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            max_samples = int(REFERENCE_MAX_SECONDS * sr)
            if max_samples > 0 and mono.shape[0] > max_samples:
                mono = mono[:max_samples]
            wf = torch.from_numpy(mono).float().unsqueeze(0)
            if sr != EMBED_SAMPLE_RATE:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(
                        sr, EMBED_SAMPLE_RATE
                    )
                wf = resamplers[sr](wf)
            wf = wf.to(device)
            with torch.no_grad():
                emb = model.encode_batch(wf).squeeze(0).squeeze(0)
                emb = torch.nn.functional.normalize(emb, dim=0)
                ref_embs.append(emb)

        if not ref_embs:
            print("Erro: nao foi possivel gerar embedding de referencia.")
            sys.exit(1)
        ref_embedding = torch.stack(ref_embs, dim=0).mean(dim=0)
        ref_embedding = torch.nn.functional.normalize(ref_embedding, dim=0)

    overlaps = _load_overlaps(OVERLAP_FILE)
    overlap_idx = 0

    with sf.SoundFile(ORIGINAL_AUDIO_FILE) as src:
        sr = src.samplerate
        channels = src.channels
        subtype = src.subtype

        min_samples = int(MIN_SEGMENT_SECONDS * sr)
        edge_trim = int(EDGE_TRIM_SECONDS * sr)
        silence_samples = int((ADD_SILENCE_MS / 1000.0) * sr)
        silence = None
        if silence_samples > 0:
            silence = np.zeros((silence_samples, channels), dtype="float32")

        if EXPORT_INDIVIDUAL:
            os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

        kept = 0
        review_rows: list[tuple[str, float, float, float, float | None, float | None, float | None]] = []
        out = None
        if OUTPUT_WAV:
            out = sf.SoundFile(
                OUTPUT_WAV, mode="w", samplerate=sr, channels=channels, subtype=subtype
            )
        try:
            segment_idx = 0
            for start, end, speaker in segments:
                segment_idx += 1
                if speaker != TARGET_SPEAKER:
                    continue
                start_frame = max(0, int(start * sr) + edge_trim)
                end_frame = max(0, int(end * sr) - edge_trim)
                if end_frame <= start_frame:
                    continue
                if (end_frame - start_frame) < min_samples:
                    continue

                if overlaps:
                    seg_start_s = start_frame / sr
                    seg_end_s = end_frame / sr
                    ratio, overlap_idx = _overlap_ratio(
                        seg_start_s, seg_end_s, overlaps, overlap_idx
                    )
                    if ratio > OVERLAP_MAX_RATIO:
                        continue

                src.seek(start_frame)
                data = src.read(end_frame - start_frame, dtype="float32", always_2d=True)
                if data.size == 0:
                    continue

                sim_min = None
                sim_avg = None
                sim_max = None
                if use_similarity:
                    mono = data.mean(axis=1)
                    chunk_len = int(EMBED_SECONDS * sr)
                    if chunk_len <= 0 or mono.shape[0] < chunk_len:
                        if similarity_log is not None:
                            similarity_log.write(
                                f"{segment_idx},{start:.3f},{end:.3f},"
                                f"{(end-start):.3f},0,0,0,0\n"
                            )
                        continue

                    chunk_count = EMBED_CHUNKS if EMBED_CHUNKS > 0 else 1
                    max_chunks = max(1, mono.shape[0] // chunk_len)
                    chunk_count = min(chunk_count, max_chunks)
                    gap = mono.shape[0] - chunk_len
                    if chunk_count == 1:
                        offsets = [gap // 2]
                    else:
                        step = gap // (chunk_count + 1) if gap > 0 else 0
                        offsets = [step * (i + 1) for i in range(chunk_count)]

                    sims = []
                    for off in offsets:
                        chunk = mono[off : off + chunk_len]
                        wf = torch.from_numpy(chunk).float().unsqueeze(0)
                        if sr != EMBED_SAMPLE_RATE:
                            if sr not in resamplers:
                                resamplers[sr] = torchaudio.transforms.Resample(
                                    sr, EMBED_SAMPLE_RATE
                                )
                            wf = resamplers[sr](wf)
                        wf = wf.to(device)
                        with torch.no_grad():
                            emb = model.encode_batch(wf).squeeze(0).squeeze(0)
                            emb = torch.nn.functional.normalize(emb, dim=0)
                            sim = torch.sum(emb * ref_embedding).item()
                            sims.append(sim)

                    if not sims:
                        continue
                    sim_min = min(sims)
                    sim_avg = sum(sims) / len(sims)
                    sim_max = max(sims)
                    keep = sim_min >= SIMILARITY_THRESHOLD
                    if similarity_log is not None:
                        similarity_log.write(
                            f"{segment_idx},{start:.3f},{end:.3f},"
                            f"{(end-start):.3f},{sim_min:.4f},"
                            f"{sim_avg:.4f},{sim_max:.4f},{1 if keep else 0}\n"
                        )
                    if not keep:
                        continue

                kept += 1

                if out is not None:
                    out.write(data)
                    if silence is not None:
                        out.write(silence)

                if EXPORT_INDIVIDUAL:
                    start_ms = int((start_frame / sr) * 1000)
                    end_ms = int((end_frame / sr) * 1000)
                    out_name = (
                        f"{INDIVIDUAL_PREFIX}_{kept:04d}_"
                        f"{start_ms:09d}_{end_ms:09d}.wav"
                    )
                    out_path = os.path.join(INDIVIDUAL_DIR, out_name)
                    with sf.SoundFile(
                        out_path,
                        mode="w",
                        samplerate=sr,
                        channels=channels,
                        subtype=subtype,
                    ) as seg_out:
                        seg_out.write(data)
                    final_start_s = start_frame / sr
                    final_end_s = end_frame / sr
                    duration_s = final_end_s - final_start_s
                    review_rows.append(
                        (
                            out_name,
                            final_start_s,
                            final_end_s,
                            duration_s,
                            sim_min,
                            sim_avg,
                            sim_max,
                        )
                    )
        finally:
            if out is not None:
                out.close()
            if similarity_log is not None:
                similarity_log.close()

    if EXPORT_INDIVIDUAL and review_rows:
        os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
        review_rows_sorted = sorted(
            review_rows,
            key=lambda r: (r[4] if r[4] is not None else -1.0),
            reverse=True,
        )
        with open(REVIEW_CSV, "w", encoding="utf-8") as f:
            f.write("file,start_s,end_s,duration_s,sim_min,sim_avg,sim_max,approved\n")
            for row in review_rows_sorted:
                sim_min, sim_avg, sim_max = row[4], row[5], row[6]
                f.write(
                    f"{row[0]},{row[1]:.3f},{row[2]:.3f},{row[3]:.3f},"
                    f"{'' if sim_min is None else f'{sim_min:.4f}'},"
                    f"{'' if sim_avg is None else f'{sim_avg:.4f}'},"
                    f"{'' if sim_max is None else f'{sim_max:.4f}'},"
                    "0\n"
                )

    msg = f"OK. Trechos extraidos: {kept}."
    if OUTPUT_WAV:
        msg += f" Arquivo gerado: {OUTPUT_WAV}."
    if EXPORT_INDIVIDUAL:
        msg += f" Segments: {INDIVIDUAL_DIR}\\"
        msg += f" Review CSV: {REVIEW_CSV}"
    print(msg)


if __name__ == "__main__":
    main()
