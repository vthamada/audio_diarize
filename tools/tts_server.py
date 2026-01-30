from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Jarvis TTS Router")

OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "tts_outputs"))
OUTPUT_DIR.mkdir(exist_ok=True)

# Choose backend: gpt_sovits | fish | f5
TTS_BACKEND = os.getenv("TTS_BACKEND", "fish")

# Paths to backend scripts (configure after cloning repos)
GPT_SOVITS_CMD = os.getenv("GPT_SOVITS_CMD", "")
FISH_CMD = os.getenv("FISH_CMD", "")
F5_CMD = os.getenv("F5_CMD", "")

# Reference audio for zero-shot backends
REF_WAV = os.getenv("TTS_REF_WAV", "")

class TTSRequest(BaseModel):
    text: str
    backend: Optional[Literal["gpt_sovits", "fish", "f5"]] = None
    reference_wav: Optional[str] = None


def _run_cmd(cmd: str) -> None:
    if not cmd:
        raise RuntimeError("Backend command not configured.")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


@app.post("/tts")
def tts(req: TTSRequest):
    backend = req.backend or TTS_BACKEND
    ref = req.reference_wav or REF_WAV

    if backend not in {"gpt_sovits", "fish", "f5"}:
        raise HTTPException(status_code=400, detail="Invalid backend")

    out_path = OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.wav"

    # NOTE: these command templates are placeholders.
    # After you install each backend, set env vars with the exact CLI.
    if backend == "gpt_sovits":
        if not GPT_SOVITS_CMD:
            raise HTTPException(status_code=500, detail="GPT-SoVITS command not configured")
        cmd = GPT_SOVITS_CMD.format(text=req.text, out=str(out_path), ref=ref)
    elif backend == "fish":
        if not FISH_CMD:
            raise HTTPException(status_code=500, detail="Fish command not configured")
        cmd = FISH_CMD.format(text=req.text, out=str(out_path), ref=ref)
    else:
        if not F5_CMD:
            raise HTTPException(status_code=500, detail="F5 command not configured")
        cmd = F5_CMD.format(text=req.text, out=str(out_path), ref=ref)

    try:
        _run_cmd(cmd)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"output_wav": str(out_path)}
