from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import requests


# -------------------------
# Config
# -------------------------
API_KEY = os.getenv("PYANNOTE_API_KEY", "")
API_BASE = "https://api.pyannote.ai/v1"
MODEL = "precision-2"
EXCLUSIVE = True
MATCHING_THRESHOLD = 50  # 0-100, higher = more strict
MIN_SPEAKERS = 0
MAX_SPEAKERS = 0
NUM_SPEAKERS = 0

MAIN_AUDIO_FILE = "podcast_completo.WAV"
REF_DIR = "ref_speaker"
VOICEPRINT_CACHE = "voiceprints.json"
MEDIA_CACHE = "pyannote_media.json"
OUTPUT_JSON = "identify_output.json"
OUTPUT_SPEAKERS = "speakers.txt"

POLL_SECONDS = 5
UPLOAD_TIMEOUT = 3600


# -------------------------
# Helpers
# -------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(url, json=payload, headers=_headers(), timeout=60)
    if not resp.ok:
        raise RuntimeError(f"POST {url} failed: {resp.status_code} {resp.text}")
    return resp.json()


def _get_json(url: str) -> dict[str, Any]:
    resp = requests.get(url, headers=_headers(), timeout=60)
    if not resp.ok:
        raise RuntimeError(f"GET {url} failed: {resp.status_code} {resp.text}")
    return resp.json()


def _upload_media(path: str, object_key: str) -> str:
    payload = {"url": f"media://{object_key}"}
    meta = _post_json(f"{API_BASE}/media/input", payload)
    upload_url = meta.get("url")
    if not upload_url:
        raise RuntimeError("media/input did not return upload url")

    _log(f"Uploading {os.path.basename(path)}...")
    with open(path, "rb") as f:
        put = requests.put(
            upload_url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
            timeout=UPLOAD_TIMEOUT,
        )
    if not put.ok:
        raise RuntimeError(f"Upload failed: {put.status_code} {put.text}")

    return payload["url"]


def _wait_job(job_id: str) -> dict[str, Any]:
    url = f"{API_BASE}/jobs/{job_id}"
    while True:
        data = _get_json(url)
        status = data.get("status", "")
        if status in {"succeeded", "failed", "canceled"}:
            return data
        time.sleep(POLL_SECONDS)


def _make_object_key(prefix: str, path: str) -> str:
    base = Path(path).stem
    return f"{prefix}/{base}-{uuid.uuid4().hex}.wav"


def _extract_segments(diar: Any) -> list[Any] | None:
    if isinstance(diar, list):
        return diar
    if isinstance(diar, dict):
        segs = diar.get("segments")
        if isinstance(segs, list):
            return segs
    return None


def _read_segment(seg: Any) -> tuple[float, float, str]:
    if isinstance(seg, dict):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        speaker = (
            seg.get("speaker")
            or seg.get("label")
            or seg.get("name")
            or "SPEAKER"
        )
        return start, end, str(speaker)
    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
        start = float(seg[0])
        end = float(seg[1])
        speaker = str(seg[2]) if len(seg) >= 3 else "SPEAKER"
        return start, end, speaker
    return 0.0, 0.0, "SPEAKER"


# -------------------------
# Main
# -------------------------

def main() -> None:
    if not API_KEY:
        raise SystemExit("Erro: defina PYANNOTE_API_KEY no ambiente.")
    if not os.path.exists(MAIN_AUDIO_FILE):
        raise SystemExit(f"Erro: arquivo nao encontrado: {MAIN_AUDIO_FILE}")
    if not os.path.isdir(REF_DIR):
        raise SystemExit(f"Erro: pasta de referencia nao encontrada: {REF_DIR}")

    ref_files = [
        os.path.join(REF_DIR, f)
        for f in os.listdir(REF_DIR)
        if f.lower().endswith(".wav")
    ]
    if not ref_files:
        raise SystemExit(f"Erro: coloque WAVs limpos em {REF_DIR}")

    media_cache = _read_json(MEDIA_CACHE, {})

    # Upload main audio
    if MAIN_AUDIO_FILE in media_cache:
        main_media = media_cache[MAIN_AUDIO_FILE]
    else:
        main_key = _make_object_key("audio", MAIN_AUDIO_FILE)
        main_media = _upload_media(MAIN_AUDIO_FILE, main_key)
        media_cache[MAIN_AUDIO_FILE] = main_media
        _write_json(MEDIA_CACHE, media_cache)

    # Voiceprints cache
    vp_cache = _read_json(VOICEPRINT_CACHE, {})
    voiceprints: list[dict[str, str]] = []

    for ref_path in sorted(ref_files):
        label = Path(ref_path).stem
        mtime = os.path.getmtime(ref_path)
        cached = vp_cache.get(ref_path)
        if cached and cached.get("mtime") == mtime and cached.get("voiceprint"):
            voiceprints.append({"label": label, "voiceprint": cached["voiceprint"]})
            continue

        ref_key = _make_object_key("ref", ref_path)
        ref_media = _upload_media(ref_path, ref_key)
        payload = {"url": ref_media, "model": MODEL}
        job = _post_json(f"{API_BASE}/voiceprint", payload)
        job_id = job.get("jobId") or job.get("job_id")
        if not job_id:
            raise RuntimeError("voiceprint did not return jobId")
        _log(f"Voiceprint job started: {job_id}")
        result = _wait_job(job_id)
        if result.get("status") != "succeeded":
            raise RuntimeError(f"Voiceprint failed: {result}")
        output = result.get("output", {})
        voiceprint = output.get("voiceprint")
        if not voiceprint:
            raise RuntimeError(f"Voiceprint missing in output: {output}")
        voiceprints.append({"label": label, "voiceprint": voiceprint})
        vp_cache[ref_path] = {"mtime": mtime, "voiceprint": voiceprint}
        _write_json(VOICEPRINT_CACHE, vp_cache)

    _log(f"Voiceprints prontos: {len(voiceprints)}")

    payload: dict[str, Any] = {
        "url": main_media,
        "model": MODEL,
        "exclusive": EXCLUSIVE,
        "turnLevelConfidence": True,
        "confidence": True,
        "matching": {"exclusive": True, "threshold": MATCHING_THRESHOLD},
        "voiceprints": voiceprints,
    }
    if NUM_SPEAKERS > 0:
        payload["numSpeakers"] = NUM_SPEAKERS
    if MIN_SPEAKERS > 0:
        payload["minSpeakers"] = MIN_SPEAKERS
    if MAX_SPEAKERS > 0:
        payload["maxSpeakers"] = MAX_SPEAKERS

    _log("Iniciando identify (precision-2)...")
    job = _post_json(f"{API_BASE}/identify", payload)
    job_id = job.get("jobId") or job.get("job_id")
    if not job_id:
        raise RuntimeError("identify did not return jobId")

    result = _wait_job(job_id)
    if result.get("status") != "succeeded":
        raise RuntimeError(f"identify failed: {result}")

    output = result.get("output", {})
    _write_json(OUTPUT_JSON, output)

    diar = output.get("exclusiveDiarization") if EXCLUSIVE else None
    if diar is None:
        diar = output.get("diarization")
    if diar is None:
        raise RuntimeError("No diarization found in output")

    segments = _extract_segments(diar)
    if not segments:
        raise RuntimeError("Unexpected diarization format")

    segments_sorted = sorted(segments, key=lambda s: _read_segment(s)[0])
    with open(OUTPUT_SPEAKERS, "w", encoding="utf-8") as f:
        for seg in segments_sorted:
            start, end, speaker = _read_segment(seg)
            f.write(f"{start:.2f} --> {end:.2f} | {speaker}\n")

    _log(f"OK. speakers.txt gerado: {OUTPUT_SPEAKERS}")


if __name__ == "__main__":
    main()
