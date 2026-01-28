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
- `speakers.txt` (speaker segments)
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
- Target speaker: `SPEAKER_02`
- Similarity filter:
  - `SIMILARITY_THRESHOLD = 0.75`
  - `EMBED_SECONDS = 3.0`
  - `EMBED_CHUNKS = 5`
- Segment filtering:
  - `MIN_SEGMENT_SECONDS = 20.0`
  - `EDGE_TRIM_SECONDS = 1.5`
- Overlap filtering:
  - `OVERLAP_MAX_RATIO = 0.0`

## Known Issues / Warnings
- Windows Controlled Folder Access may block writes on Desktop.
- `torchaudio.list_audio_backends` missing in some versions; shim used.
- pyannote overlap model may fail in local env; use Precision-2 JSON fallback.

## Decision Log
- Use Precision-2 for highest diarization quality when available.
- Use non-exclusive Precision-2 output to infer overlaps.
- Keep strong similarity filtering and manual review for cleanliness.

## Open Questions
- Should we add an RMS/energy filter before similarity?
- Should we store approved segments list separate from review CSV?

## Change Log
- 2026-01-28: Added Precision-2 dual-run and overlap detection fallback chain.
- 2026-01-28: Added review_list.csv + concat_approved workflow.
