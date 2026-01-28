# Context / Spec - jarvis_diarizacao

## Goal
Create a clean dataset of a target speaker (dublador Eduardo Borgerth) from a long podcast.
Primary objective: maximum cleanliness for voice cloning (TTS), even if quantity is reduced.

## Environment
- OS: Windows
- Python: 3.14
- Running on GPU when available
- HF token required: `HF_TOKEN`

## Current pipeline (high level)
1) Diarization -> `speakers.txt`
2) Identify target speaker label (currently: `SPEAKER_02`)
3) Generate clean reference clips (manual review)
4) Extract segments with strong similarity filter + overlap removal
5) Review `segments/review_list.csv` and approve only clean segments
6) Concatenate approved segments into final clean WAV

## Key scripts and what they do

### `diarize.py`
Runs pyannote diarization and writes `speakers.txt`.
CPU/GPU support already added. Uses soundfile fallback to avoid torchcodec issues.

### `sample_speakers.py`
Generates samples per speaker to help identify target speaker.
Writes reports:
- `samples/_config.txt`
- `samples/_summary.txt`
- `samples/_samples.csv`

Current config (conservative):
- `TARGET_SPEAKER = "SPEAKER_02"`
- `SAMPLE_SECONDS = 4.0`
- `MIN_SEGMENT_SECONDS = 20.0`
- `EDGE_TRIM_SECONDS = 1.0`
- `STRICT_EXTRA_SECONDS = 4.0`

### `make_ref_speaker.py`
Generates **ultra-conservative** reference clips for similarity filtering.
Outputs into `ref_speaker/` + report `ref_speaker/_ref_candidates.csv`.

Current config (very strict):
- `TARGET_SPEAKER = "SPEAKER_02"`
- `REF_SECONDS = 5.0`
- `MIN_SEGMENT_SECONDS = 40.0`
- `EDGE_TRIM_SECONDS = 2.0`
- `STRICT_EXTRA_SECONDS = 6.0`
- `MAX_REFERENCES = 6`

User manually deleted contaminated refs and kept 3 clean WAVs in `ref_speaker/`.

### `detect_overlap.py`
Runs overlapped speech detection and outputs `overlaps.txt`.
This is used to discard any segments that intersect overlap.

Important config:
- `OVERLAP_MODEL = "pyannote/overlapped-speech-detection"`
- `OVERLAP_DILATION_SECONDS = 0.2`
- `MIN_OVERLAP_SECONDS = 0.1`

### `extract_dublador.py`
Main extractor with:
- similarity filter using SpeechBrain ECAPA embeddings
- overlap filtering (`overlaps.txt`)
- individual segment export + review CSV

Current config (ultra-clean mode):
- `TARGET_SPEAKER = "SPEAKER_02"`
- `MIN_SEGMENT_SECONDS = 20.0`
- `EDGE_TRIM_SECONDS = 1.5`
- `SIMILARITY_THRESHOLD = 0.75`
- `EMBED_SECONDS = 3.0`
- `EMBED_CHUNKS = 5`
- `USE_SIMILARITY_FILTER = True`
- `OVERLAP_MAX_RATIO = 0.0`

Outputs:
- `jarvis_dublador_raw.wav`
- `segments/` individual WAVs
- `segments_similarity.csv`
- `segments/review_list.csv` (sorted by similarity, includes `approved` column)

Notes:
- `extract_dublador.py` includes compatibility patches:
  - `hf_hub_download` shim for `use_auth_token`
  - `custom.py` 404 fallback (creates empty custom.py)
  - SpeechBrain symlink patch (forces copy)
  - torchaudio missing `list_audio_backends` shim

### `concat_approved.py`
Reads `segments/review_list.csv` and concatenates only approved rows.
Output: `jarvis_dublador_clean.wav`.

## Known issues / warnings
- Windows "Controlled Folder Access" may block writes in Desktop.
  Workaround: run as Admin or move repo to `C:\work\`.
- SpeechBrain warnings about torchaudio backends are expected; processing still works.
- HF token not set => download rate limits and some warnings.

## Recommended operating procedure
1) Ensure `HF_TOKEN` is set.
2) Run diarization if needed:
   - `.\venv\Scripts\python.exe .\diarize.py`
3) Generate overlap list:
   - `.\venv\Scripts\python.exe .\detect_overlap.py`
4) Ensure clean refs in `ref_speaker/` (3-5 WAVs, 5-15s each).
5) Extract segments:
   - `.\venv\Scripts\python.exe .\extract_dublador.py`
6) Review `segments/review_list.csv`, mark `approved = 1` only for clean segments.
7) Concatenate approved:
   - `.\venv\Scripts\python.exe .\concat_approved.py`

## Current pain points
- Even with strong filtering, some contamination remains.
- Ultra-clean settings may reduce total duration drastically.
- Expect iterative tuning:
  - Adjust `SIMILARITY_THRESHOLD` (0.70-0.80)
  - Adjust `MIN_SEGMENT_SECONDS` (12-30)
  - Adjust `EDGE_TRIM_SECONDS` (1.0-2.0)

## Git state
New files added:
- `detect_overlap.py`
- `make_ref_speaker.py`
- `concat_approved.py`
- `CONTEXT_SPEC.md`

Updated:
- `extract_dublador.py`
- `sample_speakers.py`
- `.gitignore`

Large outputs (`segments/`, `samples/`, `ref_speaker/`, WAVs) are gitignored.

### `identify_precision2.py`
Uses pyannoteAI Precision-2 API with voiceprints for highest-precision diarization.
- Uploads main audio and reference WAVs from `ref_speaker/`.
- Builds/loads voiceprints and runs `/identify` with `exclusive=true`.
- Writes `speakers.txt` and saves raw output to `identify_output.json`.
- Requires `PYANNOTE_API_KEY` in environment.

Recommended API flow (best quality):
1) Ensure clean refs in `ref_speaker/` (3-5 WAVs, 5-15s each).
2) Run:
   - `\venv\Scripts\python.exe .\identify_precision2.py`
3) Continue with overlap detection and extraction if needed.

## Latest updates (2026-01-28)
- Added `identify_precision2.py` to use pyannoteAI Precision-2 with voiceprints.
- Added `requests` dependency and gitignored API artifacts (`voiceprints.json`, `identify_output.json`, `pyannote_media.json`).
- Voiceprints workflow in use; user kept **3 clean refs** in `ref_speaker/`.
- `PYANNOTE_API_KEY` set in `.env` (do not commit).

## Precision-2 voiceprints workflow (recommended)
1) Ensure clean refs in `ref_speaker/` (3-5 WAVs, 5-15s each).
2) Run identify from terminal:
   - `$env:PYANNOTE_API_KEY="<your_key>"`
   - `\venv\Scripts\python.exe .\identify_precision2.py`
3) `speakers.txt` is generated from Precision-2 exclusive diarization.
4) (Optional) run overlap detection for extra safety:
   - `\venv\Scripts\python.exe .\detect_overlap.py`
5) Extract segments:
   - `\venv\Scripts\python.exe .\extract_dublador.py`
6) Review and approve:
   - `segments/review_list.csv` (set `approved=1` only for clean segments)
7) Concatenate approved:
   - `\venv\Scripts\python.exe .\concat_approved.py`

## Next steps / planned improvements
- Add env config for `MATCHING_THRESHOLD`, `EXCLUSIVE`, `MIN/MAX_SPEAKERS` in `identify_precision2.py`.
- Consider raising `OVERLAP_DILATION_SECONDS` (0.4–0.6) for stricter overlap removal.
- Optional: add energy/RMS filter before similarity to drop weak/noisy segments.
