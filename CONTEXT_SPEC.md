# Spec - audio_diarizacao

## Problem Statement
We need to build a clean voice dataset for a single target speaker (Eduardo Borgerth) from a long podcast. The dataset will be used for voice cloning (TTS). Audio contains frequent overlaps (interjections, laughs, agreement sounds), so the pipeline must aggressively remove contamination.

## Goals
- Produce a dataset that is as clean as possible (target speaker only).
- Preserve original timbre and pacing (no EQ/normalization at this stage).
- Support CPU and GPU execution on Windows.
- Be robust for very large audio files (>1 GB).

## Non-Goals
- Real-time processing.
- Perfect automatic cleaning without any manual review.
- Audio enhancement (denoise, EQ, compression).

## Functional Requirements
- Diarize the podcast into speaker segments.
- Identify the target speaker label (e.g., SPEAKER_02).
- Generate clean reference clips for similarity filtering.
- Detect overlaps and remove any segments that intersect overlaps.
- Filter segments by voice similarity to references.
- Export segments individually for manual review.
- Concatenate only approved segments into the final dataset.

## Non-Functional Requirements
- Windows compatibility.
- CPU-only fallback if GPU unavailable.
- Clear logs and minimal manual intervention.
- Use relative paths by default.

## Inputs
- `podcast_completo.WAV` (original audio)
- `podcast_completo_16k_mono.wav` (diarization audio)
- `ref_speaker/*.wav` (clean reference clips)
- `HF_TOKEN` for Hugging Face models
- `PYANNOTE_API_KEY` for Precision-2 API

## Outputs
- `speakers_exclusive.txt` / `speakers_non_exclusive.txt`
- `identify_output_exclusive.json` / `identify_output_non_exclusive.json`
- `overlaps.txt` (overlap intervals)
- `segments/` (individual extracted clips)
- `segments/review_list.csv` (review + approval list)
- `jarvis_dublador_clean.wav` (final approved concatenation)

## Recommended Workflow (Happy Path)
1) Run Precision-2 diarization (exclusive and non-exclusive):
   - `RUN_BOTH_EXCLUSIVE=1` then run `identify_precision2.py`
2) Generate overlaps:
   - run `detect_overlap.py`
3) Extract target speaker segments:
   - run `extract_dublador.py`
4) Review `segments/review_list.csv` and mark `approved=1` for clean clips.
5) Concatenate approved clips:
   - run `concat_approved.py`

## Acceptance Criteria
- Final dataset contains only target speaker voice (no audible other speakers).
- Approved clips are traceable in `review_list.csv`.
- Pipeline can be rerun end-to-end without manual reconfiguration.

## Scripts and Responsibilities
- `diarize.py`: local pyannote diarization -> `speakers.txt`
- `identify_precision2.py`: Precision-2 API diarization + voiceprints
  - Supports dual run via `RUN_BOTH_EXCLUSIVE=1`
  - Outputs:
    - `identify_output_non_exclusive.json`, `speakers_non_exclusive.txt`
    - `identify_output_exclusive.json`, `speakers_exclusive.txt`
- `detect_overlap.py`:
  - Prefers `identify_output_non_exclusive.json` when available
  - Fallback order: non-exclusive JSON -> exclusive JSON -> overlap model -> speakers.txt
  - Env flags:
    - `PREFER_OVERLAP_MODEL=1`
    - `REQUIRE_OVERLAP_MODEL=1`
- `make_ref_speaker.py`: generate ultra-clean references
- `sample_speakers.py`: sample speakers for identification
- `extract_dublador.py`: similarity filter + overlap filter + review list
- `concat_approved.py`: concatenates approved clips only

## Key Configuration (current defaults)
- Target speaker: `SPEAKER_01`
- Similarity filter:
  - `SIMILARITY_THRESHOLD = 0.65`
  - `EMBED_SECONDS = 3.0`
  - `EMBED_CHUNKS = 5`
- Segment filtering:
  - `MIN_SEGMENT_SECONDS = 2.5`
  - `EDGE_TRIM_SECONDS = 0.5`
- Overlap filtering:
  - `OVERLAP_MAX_RATIO = 0.05`
  - `OVERLAP_DILATION_SECONDS = 0.5`
  - `MIN_OVERLAP_SECONDS = 0.2`

## Known Issues / Warnings
- Windows Controlled Folder Access may block writes on Desktop.
- `torchaudio.list_audio_backends` missing in some versions; shim used.
- pyannote overlap model may fail in local env; use Precision-2 JSON fallback.

## Decision Log
- Use Precision-2 for highest diarization quality when available.
- Use non-exclusive Precision-2 output to infer overlaps.
- Keep strong similarity filtering and manual review for cleanliness.

## Current State (2026-02-02)
- Training will be done on GPU machine; WSL2 preferred for stability.
- Precision-2 dual-run completed: `speakers_exclusive.txt` / `speakers_non_exclusive.txt` generated.
- Target speaker confirmed as `SPEAKER_01`.
- References cleaned: 9 refs kept in `ref_speaker/` (voiceprints cached).
- Overlaps regenerated from non-exclusive output with aggressive dilation.
- Extraction thresholds tuned; manual review completed.
- `segments_clean/` contains processed segments for final use.
- `jarvis_dublador_clean.wav` generated from `segments_clean` (249 clips, ~18 min).
- GPT-SoVITS dataset prepared in `datasets/gpt_sovits` (248 wavs + transcripts).
- Zero-shot refs prepared in `refs_zero_shot/` (5 clips).
- TTS server scaffold in `tools/tts_server.py`.
- Transfer bundle created: `transfer_package.zip` (datasets + refs + final wav).

## Next Steps (for another machine)
1) Copy `transfer_package.zip` to GPU machine and extract.
2) Prefer WSL2 (Ubuntu) for training on RTX 3050 4GB.
3) Install Python 3.10/3.11, Git, FFmpeg, GPU drivers inside WSL2.
4) Clone repo and install: `pip install -r requirements.txt`.
5) Install model deps inside each repo:
   - `models/GPT-SoVITS`
   - `models/fish-speech`
   - `models/F5-TTS`
6) Create `.env` with `HF_TOKEN` and `PYANNOTE_API_KEY`.
7) Train GPT-SoVITS using `datasets/gpt_sovits`.
8) Run zero-shot tests using `refs_zero_shot`.
9) Configure `tools/tts_server.py` to point to the chosen backend.

## Open Questions
- Should we add an RMS/energy filter before similarity?
- Should we store approved segments list separate from review CSV?

## Change Log
- 2026-01-28: Added Precision-2 dual-run and overlap detection fallback chain.
- 2026-01-28: Added review_list.csv + concat_approved workflow.
- 2026-01-29: Target speaker updated to SPEAKER_01; overlap dilation increased; extraction thresholds relaxed; clean 8-min output generated.
- 2026-02-02: Added ASR + GPT-SoVITS dataset prep; zero-shot refs; transfer bundle; TTS tooling updates.
- 2026-02-02: Documented WSL2 training preference for GPU machine.
