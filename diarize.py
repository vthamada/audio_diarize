from __future__ import annotations

import collections
import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(path: str = ".env", **_kwargs):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = value
            return True
        except Exception:
            return False


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".tmp_write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _ensure_temp_dir() -> None:
    # Ensure a writable temp folder before importing heavy libs (lightning/torchmetrics).
    candidates: list[str] = []
    for key in ("TEMP", "TMP", "TMPDIR"):
        value = os.environ.get(key)
        if value:
            candidates.append(value)

    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        candidates.append(os.path.join(local_app, "Temp"))

    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        candidates.append(os.path.join(user_profile, "Temp"))

    candidates.extend(
        [
            os.path.join(os.getcwd(), "tmp"),
            r"C:\Temp",
            r"C:\Windows\Temp",
        ]
    )

    for candidate in candidates:
        if _is_writable_dir(candidate):
            os.environ["TEMP"] = candidate
            os.environ["TMP"] = candidate
            return


_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(_ENV_PATH)
_ensure_temp_dir()

# -------------------------
# Config
# -------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Erro: defina HF_TOKEN no ambiente.")
    sys.exit(1)

CACHE_DIR = os.path.join(os.getcwd(), ".cache")
ORIGINAL_AUDIO_FILE = "podcast_completo.WAV"
DIARIZE_AUDIO_FILE = "podcast_completo_16k_mono.wav"
DIARIZE_SAMPLE_RATE = 16000
DIARIZE_FORCE_CONVERT = False
USE_GPU = os.environ.get("DIARIZE_USE_GPU", "1") == "1"
try:
    SEGMENTATION_BATCH_SIZE = int(os.environ.get("DIARIZE_SEG_BATCH", "32"))
except Exception:
    SEGMENTATION_BATCH_SIZE = 32
try:
    EMBEDDING_BATCH_SIZE = int(os.environ.get("DIARIZE_EMB_BATCH", "64"))
except Exception:
    EMBEDDING_BATCH_SIZE = 64
try:
    DIARIZE_TEST_SECONDS = int(os.environ.get("DIARIZE_TEST_SECONDS", "0"))
except Exception:
    DIARIZE_TEST_SECONDS = 0
try:
    DIARIZE_LOG_EVERY = int(os.environ.get("DIARIZE_LOG_EVERY", "60"))
except Exception:
    DIARIZE_LOG_EVERY = 60

os.makedirs(CACHE_DIR, exist_ok=True)


# -------------------------
# Compatibility patches
# -------------------------

def _patch_hf_hub_download() -> None:
    try:
        import huggingface_hub as hf
    except Exception:
        return

    orig = hf.hf_hub_download

    def patched(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        elif "use_auth_token" in kwargs:
            kwargs.pop("use_auth_token")
        filename = kwargs.get("filename")
        try:
            return orig(*args, **kwargs)
        except Exception as e:
            try:
                from huggingface_hub.utils import EntryNotFoundError
            except Exception:
                EntryNotFoundError = None
            is_missing = False
            if EntryNotFoundError is not None and isinstance(e, EntryNotFoundError):
                is_missing = True
            if "404" in str(e) or "Entry Not Found" in str(e):
                is_missing = True
            if is_missing and str(filename).endswith("custom.py"):
                # speechbrain treats missing custom.py as optional
                raise ValueError("File not found on HF hub") from e
            raise

    hf.hf_hub_download = patched


def _patch_torch_serialization() -> None:
    try:
        import torch
    except Exception:
        return

    # allow common classes for safe deserialization
    safe_globals: list[Any] = [
        list,
        dict,
        tuple,
        set,
        object,
        int,
        float,
        collections.defaultdict,
    ]
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        from typing import Any as TypingAny

        safe_globals.extend(
            [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, TypingAny]
        )
    except Exception:
        pass
    try:
        from pyannote.audio.core.model import Introspection
        safe_globals.append(Introspection)
    except Exception:
        pass
    try:
        from pyannote.audio.core.task import (
            Problem,
            Resolution,
            Specifications,
            Task,
            TrainDataset,
            ValDataset,
            UnknownSpecificationsError,
        )

        safe_globals.extend(
            [
                Problem,
                Resolution,
                Specifications,
                Task,
                TrainDataset,
                ValDataset,
                UnknownSpecificationsError,
            ]
        )
    except Exception:
        pass

    try:
        torch.serialization.add_safe_globals(safe_globals)
    except Exception:
        try:
            torch.serialization.safe_globals(safe_globals)
        except Exception:
            pass

    # force weights_only=False if not provided
    orig_torch_load = getattr(torch, "load", None)
    if orig_torch_load is not None:
        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)
        torch.load = patched_load

    if hasattr(torch, "serialization"):
        orig_serialization_load = getattr(torch.serialization, "load", None)
        if orig_serialization_load is not None:
            def patched_serialization_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return orig_serialization_load(*args, **kwargs)
            torch.serialization.load = patched_serialization_load


def _patch_torchaudio() -> None:
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends():
            return []
        torchaudio.list_audio_backends = _list_audio_backends


def _patch_speechbrain() -> None:
    try:
        from speechbrain.inference import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except Exception:
        return

    orig = EncoderClassifier.from_hparams

    def patched(*args, **kwargs):
        kwargs.setdefault("local_strategy", LocalStrategy.COPY)
        return orig(*args, **kwargs)

    EncoderClassifier.from_hparams = patched


def _patch_pyannote_revision() -> None:
    try:
        from pyannote.audio.core.model import Model
    except Exception:
        return

    orig = Model.from_pretrained

    def patched(checkpoint, *args, **kwargs):
        if (
            isinstance(checkpoint, str)
            and "@" in checkpoint
            and "revision" not in kwargs
        ):
            checkpoint, revision = checkpoint.split("@", 1)
            kwargs["revision"] = revision
        return orig(checkpoint, *args, **kwargs)

    Model.from_pretrained = patched


def _apply_patches() -> None:
    _patch_hf_hub_download()
    _patch_torch_serialization()
    _patch_torchaudio()
    _patch_speechbrain()
    _patch_pyannote_revision()


def _select_device():
    try:
        import torch
    except Exception:
        return None
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_batch_sizes(pipeline) -> None:
    # Try common attributes first.
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = SEGMENTATION_BATCH_SIZE
    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = EMBEDDING_BATCH_SIZE

    # Fallback: adjust Inference objects if present.
    for attr, value in (("_segmentation", SEGMENTATION_BATCH_SIZE), ("_embedding", EMBEDDING_BATCH_SIZE)):
        obj = getattr(pipeline, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "batch_size"):
            obj.batch_size = value


# -------------------------
# Audio helpers
# -------------------------

def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        pattern = os.path.join(
            local_app,
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg_*",
            "ffmpeg-*",
            "bin",
            "ffmpeg.exe",
        )
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return "ffmpeg"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def _start_heartbeat(label: str, every_seconds: int):
    if every_seconds <= 0:
        return None, None
    stop_event = threading.Event()

    def run():
        start = time.monotonic()
        while not stop_event.wait(every_seconds):
            elapsed = int(time.monotonic() - start)
            _log(f"{label}... {elapsed}s")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return stop_event, thread


def _ensure_diarization_audio() -> str:
    if os.path.exists(DIARIZE_AUDIO_FILE) and not DIARIZE_FORCE_CONVERT:
        return DIARIZE_AUDIO_FILE
    if not os.path.exists(ORIGINAL_AUDIO_FILE):
        raise FileNotFoundError(f"Arquivo de audio nao encontrado: {ORIGINAL_AUDIO_FILE}")
    _log("Gerando audio para diarizacao (16k mono)...")
    ffmpeg_exe = _resolve_ffmpeg()
    if ffmpeg_exe == "ffmpeg" and not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg nao encontrado no PATH. Instale o FFmpeg ou reinicie o terminal."
        )
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-y",
        "-i",
        ORIGINAL_AUDIO_FILE,
        "-ac",
        "1",
        "-ar",
        str(DIARIZE_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        DIARIZE_AUDIO_FILE,
    ]
    subprocess.run(cmd, check=True)
    return DIARIZE_AUDIO_FILE


def _load_audio_for_pipeline(path: str, max_seconds: int = 0) -> dict[str, Any]:
    # Prefer soundfile to avoid torchcodec dependency when possible.
    try:
        import soundfile as sf
        import torch

        with sf.SoundFile(path) as f:
            sample_rate = f.samplerate
            frames = int(max_seconds * sample_rate) if max_seconds else -1
            data = f.read(frames if frames > 0 else -1, dtype="float32", always_2d=True)

        waveform = torch.from_numpy(data.T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        duration = waveform.shape[-1] / float(sample_rate)
        return {"waveform": waveform, "sample_rate": sample_rate, "duration": duration}
    except Exception:
        pass

    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(path)
        if max_seconds:
            max_frames = int(max_seconds * sample_rate)
            waveform = waveform[..., :max_frames]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        duration = waveform.shape[-1] / float(sample_rate)
        return {"waveform": waveform, "sample_rate": sample_rate, "duration": duration}
    except Exception as e:
        raise RuntimeError(
            "Audio loading failed with soundfile and torchaudio. "
            "Install libsndfile or fix torchcodec/ffmpeg."
        ) from e


def main() -> None:
    _apply_patches()

    from pyannote.audio import Pipeline

    _log(f"Arquivo original: {ORIGINAL_AUDIO_FILE}")
    diarize_audio_path = _ensure_diarization_audio()
    _log(f"Usando audio para diarizacao: {diarize_audio_path}")
    if DIARIZE_TEST_SECONDS:
        _log(f"Test mode: {DIARIZE_TEST_SECONDS}s")

    _log("Carregando pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    _apply_batch_sizes(pipeline)
    _log(
        f"Batch sizes: segmentation={SEGMENTATION_BATCH_SIZE}, embedding={EMBEDDING_BATCH_SIZE}"
    )

    device = _select_device()
    if device is not None:
        try:
            pipeline.to(device)
            _log(f"Usando device: {device}")
        except Exception:
            pass

    _log("Carregando audio para pipeline...")
    audio = _load_audio_for_pipeline(
        diarize_audio_path, max_seconds=DIARIZE_TEST_SECONDS
    )
    if "duration" in audio:
        _log(f"Duracao carregada: {audio['duration']:.1f}s")
    if device is not None and "waveform" in audio:
        try:
            audio["waveform"] = audio["waveform"].to(device)
        except Exception:
            pass
    _log("Iniciando diarizacao...")
    stop_event, thread = _start_heartbeat("Diarizacao em andamento", DIARIZE_LOG_EVERY)
    t0 = time.monotonic()
    try:
        diarization = pipeline(audio)
    finally:
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1)
    _log(f"Diarizacao concluida em {time.monotonic() - t0:.1f}s")

    annotation = getattr(diarization, "speaker_diarization", diarization)
    _log("Escrevendo speakers.txt...")
    with open("speakers.txt", "w", encoding="utf-8") as f:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            line = f"{turn.start:.2f} --> {turn.end:.2f} | {speaker}"
            print(line)
            f.write(line + "\n")

    _log("OK. Diarizacao concluida. Arquivo gerado: speakers.txt")


if __name__ == "__main__":
    main()
