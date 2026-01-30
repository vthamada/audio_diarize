from __future__ import annotations

import csv
import os
import sys


# -------------------------
# Config
# -------------------------
REVIEW_CSV = os.path.join("segments", "review_list.csv")
REVIEW_CSVS = os.getenv("REVIEW_CSVS", "")
OUTPUT_WAV = "jarvis_dublador_clean.wav"
ADD_SILENCE_MS = 0
DEDUP_TOLERANCE_SEC = 0.25


def _is_approved(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"1", "true", "yes", "ok", "approved", "y", "sim"}


def main() -> None:
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        print("Erro: soundfile nao instalado. Rode: pip install soundfile")
        sys.exit(1)

    csv_paths = []
    if REVIEW_CSVS:
        for part in REVIEW_CSVS.replace(';', ',').split(','):
            p = part.strip()
            if p:
                csv_paths.append(p)
    else:
        csv_paths = [REVIEW_CSV]

    rows = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Erro: arquivo nao encontrado: {csv_path}")
            sys.exit(1)
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _is_approved(row.get("approved", "")):
                    row["__base_dir"] = base_dir
                    rows.append(row)

    if not rows:
        print("Nenhum trecho aprovado. Marque a coluna 'approved' com 1/yes/ok.")
        sys.exit(1)

    # deduplicate by (start,end) within tolerance
    def _parse_float(v):
        try:
            return float(v)
        except Exception:
            return None

    deduped = []
    seen = []
    for row in rows:
        start = _parse_float(row.get("start_s", ""))
        end = _parse_float(row.get("end_s", ""))
        if start is None or end is None:
            deduped.append(row)
            continue
        dup = False
        for s, e in seen:
            if abs(start - s) <= DEDUP_TOLERANCE_SEC and abs(end - e) <= DEDUP_TOLERANCE_SEC:
                dup = True
                break
        if not dup:
            seen.append((start, end))
            deduped.append(row)

    rows = deduped

    def _row_start(r):
        try:
            return float(r.get("start_s", ""))
        except Exception:
            return float("inf")

    rows.sort(key=_row_start)

    first_path = os.path.join(rows[0].get("__base_dir", ""), rows[0]["file"])
    with sf.SoundFile(first_path) as first:
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
                seg_path = os.path.join(row.get("__base_dir", ""), rel_path)
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
