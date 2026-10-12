# Source grounding, hallucination safeguards, and the three QA tracks

This document describes how AutoClipper keeps generated clips traceable to the
source video and how it separates technical, editorial, and factual validation.
It accompanies `AUDIT_ROOT_CAUSES.md` (root-cause analysis of the knife-safety
failure) and the implementation in `autoclipper/grounding.py`,
`autoclipper/analyze.py`, and `autoclipper/pipeline.py`.

## Evidence hierarchy

1. The source media and its verified timestamps (strongest).
2. The actual transcript text with its timestamps.
3. Visual observations from frames that were genuinely sampled and inspected.
4. Model-generated summaries, chapters, topics, and suggested angles.
5. Model-generated inferences (weakest).

Items 4–5 are hypotheses about the source. They never silently override 1–3.
Every generated candidate must cite an `evidence_quote` that exists verbatim in
the transcript or it is dropped (`analyze._verify_evidence_quotes`).

## Timestamp convention

All intervals are `[start, end)` in seconds on the **original source timeline**.
Clip boundaries are validated against the real source duration; caption cues
are remapped to clip-relative time from word timestamps; montage segments keep
their source intervals in `source_segments`. A boundary within
`grounding.BOUNDARY_TOLERANCE` (0.45 s) of a transcript segment edge counts as
sentence-aligned (Whisper segments are sentence-level).

## Deterministic factual grounding (`grounding.py`)

`ground_candidate()` produces the per-clip `factual_qa` record with:

- `evidence_quote` — verbatim verification + location (in-span or outside).
- `generated_text` — title/hook/description grounding: fabricated-quotation
  detection (quoted phrases must exist verbatim in the transcript), enumerated
  count claims ("3 rules" must be deliverable from the span), and lexical
  support ratios. Never rewrites text; reports only.
- `boundary_alignment` + `boundary_risks` — mid-sentence cuts, negations cut at
  a boundary ("do not …" excluded before the clip starts = critical), qualifiers
  or safety statements cut at the ending.
- `topic_relevance` — lexical overlap with the user's explicit direction;
  below `RELEVANCE_CRITICAL` an AI-generated candidate is rejected as unrelated.
- `transcript_uncertainty` — low-confidence words when the provider exposes
  probabilities (Groq currently does not; never invented).
- `visual_verification` — `frame_sampled_model_review`, `frame_sampled_no_review`,
  or `transcript_only`. A transcript-only path can NEVER be reported as visual
  verification; model-written visual claims for candidates without sampled
  frames are overwritten with "not verifiable from sampled frames".

Statuses: `passed`, `warning`, `failed` (plus `not_run` / `unverified` where
applicable). Critical failures reject AI-generated candidates before rendering;
user-selected candidates are retained but visibly flagged, because discarding
the user's explicit choice silently was one of the original bugs.

## Independent candidate review states (`analyze.review_clip_candidates`)

Review outcomes are distinguished, never collapsed:

| status | meaning |
|---|---|
| `reviewed` | complete valid review; `reviewed_count`/`rejected_count` populated |
| `malformed` | model answered but the payload was unusable/incomplete/contradictory |
| `timed_out` | provider deadline exceeded |
| `unavailable` | client/server/network failure or unexpected error |
| `skipped_no_key` | no Gemini key configured; review never attempted |
| `not_run` | no candidates to review |

"Review unavailable" is **not** "review passed": unreviewed candidates are
labeled `review_state != reviewed` in their factual record and must still pass
all deterministic grounding checks. Failures carry `failure_reason`.

The reviewer never controls numeric timestamps: boundary moves are derived only
from exact quotes matched to real transcript segments, bounded by duration
limits and a maximum shift.

## Duration honesty

- A user-selected angle is validated against timestamp integrity and the 120 s
  hard ceiling only — never against the target-duration window.
- A short selection may be extended toward the target, but only by adding
  contiguous transcript segments that are topically relevant to the selection
  (`_extend_selection_to_target`); unrelated padding is impossible.
- When the source lacks enough relevant material, the shorter accurate clip is
  delivered with `duration_note` / `duration_delta` explaining the deviation.
- The 120 s hard maximum is enforced everywhere (`config.HARD_MAX_CLIP_SECONDS`).

## The three QA tracks (`validation_summary`)

- **technical** — `quality_assurance.verify_rendered_file` on the actual final
  file: streams, duration, aspect ratio, audio, SRT + burned-output artifacts,
  decodable frame samples. Subtitle verification is artifact-based; no OCR is
  performed and that is disclosed in the report.
- **editorial** — deterministic structure checks: opening/ending sentence
  alignment, duration fit vs. target, montage stitching gaps (stitched shots
  are reported as separate moments, never as one continuous event).
- **factual** — the grounding record above plus independent review state.

Each track reports its own status; a technical pass never implies editorial or
factual pass. Montage entries additionally carry `source_segments` and an
editorial warning whenever the stitched shots are not continuous in the source.

## What cannot be verified automatically

- Semantic nuance beyond lexical grounding (paraphrases, sarcasm, implied
  meaning) still needs human or model review; deterministic checks flag, they
  do not prove.
- Burned-in caption pixels are not OCR-verified; we verify the burned artifact
  exists, decodes, and matches a valid in-range SRT built from real words.
- Visual verification depends on sparse frame sampling; actions between sampled
  frames are not established.
- Per-word transcription confidence is not exposed by Groq Whisper today; the
  pipeline consumes it if a provider ever returns it, and never invents scores.
