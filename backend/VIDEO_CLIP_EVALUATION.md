# Video clip evaluation workflow

The video evaluator compares actual generated clip intervals with **manually annotated reference spans** and summarizes only human ratings that have been entered. It does not create ground truth, infer virality, or claim an accuracy improvement from synthetic tests. Temporal overlap measures interval agreement, not whether a clip is interesting; human review is a separate, subjective measure.

## 1. Prepare an authorized evaluation set

Use local videos you own or have explicit permission to process and evaluate. Keep the files, manifest, generated media, and report private when their rights or contents require it. Every manifest and sample carries an explicit authorization attestation; the evaluator will refuse to proceed if either is missing. Do not include provider keys in this file.

Create a JSON manifest. Paths are relative to the manifest unless absolute:

```json
{
  "version": 1,
  "authorized_for_evaluation": true,
  "samples": [
    {
      "id": "interview-01",
      "source": "authorized/interview-01.mp4",
      "authorized": true,
      "duration": 412.6,
      "tags": ["interview", "single-speaker"],
      "manual_references": [
        {"start": 34.2, "end": 58.8, "label": "Manually selected complete answer"},
        {"start": 181.0, "end": 206.4, "label": "Manually selected story"}
      ],
      "generated_metadata": "runs/interview-01_metadata.json",
      "human_judgments": [
        {
          "reviewer": "reviewer-1",
          "ratings": {
            "hook_strength": 4,
            "coherence": 5,
            "transcript_accuracy": 4,
            "visual_fit": 4,
            "standalone_value": 5,
            "overall": 4
          },
          "notes": "Optional reviewer notes; do not include private information."
        }
      ]
    }
  ]
}
```

Annotate references independently of the system output when practical. Timestamps are seconds from the beginning of the source. Reference spans must have a non-negative start, a later end, and stay within the declared duration (a 250 ms tolerance is allowed). `generated_metadata` points to the JSON result from a normal AutoClipper job and is the safe, offline default. It can be omitted if the manifest contains a `generated_clips` array with manually supplied or previously generated intervals (`start` and `end` required).

Human ratings are optional integers from 1 (low) to 5 (high) in these dimensions: `hook_strength`, `coherence`, `transcript_accuracy`, `visual_fit`, `standalone_value`, and `overall`. Add one judgment per reviewer per clip/sample as appropriate. Invalid, absent, or out-of-range values are not counted. The tool never fills missing ratings. Keep a stable rubric and use independent reviewers if comparing runs; the report does not attempt to infer agreement or statistical significance.

## 2. Offline evaluation of existing outputs

From `backend/`, run:

```bash
PYTHONPATH=. python scripts/evaluate_clips.py \
  --manifest /path/to/authorized-video-evaluation.json \
  --output ../output/video-evaluation/report.json
```

The default IoU threshold is 0.5. Matching is greedy one-to-one: each generated interval and each reference can be matched at most once. The report includes per-sample and micro-aggregated precision, recall, F1, matched interval IoU, clip QA counts, and entered human-rating averages. Use `--iou-threshold 0.3` (or another value in `[0,1]`) only when a different temporal tolerance is justified; record that choice when comparing reports.

## 3. Optional explicit generation run

To run the current pipeline on every authorized local source before evaluating, use the opt-in flag:

```bash
GROQ_API_KEY=... GEMINI_API_KEY=... PYTHONPATH=. \
  python scripts/evaluate_clips.py \
  --manifest /path/to/authorized-video-evaluation.json \
  --run-pipeline \
  --artifact-dir ../output/video-evaluation/run-01 \
  --output ../output/video-evaluation/run-01-report.json
```

This makes real provider calls and creates media artifacts. Do not run it in CI or against media without permission. The evaluator reads credentials only from the process environment or the local ignored `backend/.env` through application configuration; credentials are neither accepted in the manifest nor written to the report. Set ordinary per-sample `pipeline_options` such as `clip_count`, `min_clip`, `max_clip`, `aspect_ratio`, or `subtitles` in the manifest if you need a controlled setting. Keep settings consistent between compared runs.

## 4. Read results honestly

- Temporal precision/recall/F1 measure agreement with the supplied spans at the chosen IoU threshold. They do **not** measure viral performance, semantic quality, or transcription accuracy.
- A low score can reflect generation, an unsuitable threshold, or inconsistent manual annotation. Inspect per-sample intervals before changing the pipeline. Invalid generated timestamps are counted as unmatched predictions, not silently dropped.
- Human rating averages are `null` when no human score exists. A report with no ratings explicitly says so.
- QA status counts summarize the stored per-clip QA status where available; legacy metadata without QA is counted as `unknown`.
- Run on a representative, permissioned sample and report sample counts, settings, and limitations. No real benchmark result is checked into this repository; tests use synthetic intervals and validate metric behavior only.
