# Whisper transcription benchmark

This benchmark compares Groq's `whisper-large-v3-turbo` and `whisper-large-v3` on the **same audio files**, scored against manually corrected references. It does not use either model as ground truth and does not change AutoClipper's default model.

## Existing transcription path

- `autoclipper/transcribe.py::_transcribe_file` sends audio to Groq's transcription endpoint and requests `verbose_json` with word and segment timestamps.
- `autoclipper/config.py::GROQ_WHISPER_MODEL` defaults to `whisper-large-v3-turbo` and can be overridden by `GROQ_WHISPER_MODEL`.
- The per-user `whisper_model` setting flows from the Settings UI through `/api/process` and `/api/analyze-video` into `transcribe()`.
- `frontend/src/components/SettingsPanel.jsx` already offers Turbo and Large V3. The benchmark does not alter that setting or the transcript structure.
- Analysis-first stores the transcript and reuses it when the user later requests clips.

## Prepare a reference set

Use several short, representative audio files, preferably extracted from difficult sections of real videos. Correct each transcript manually. Include names, technical terms, pauses/silence, topic changes, and overlapping speech as applicable. Add tags so the report can show results for each case type.

Create a JSON manifest, for example:

```json
{
  "samples": [
    {
      "id": "podcast-names-overlap",
      "audio": "audio/podcast-names-overlap.wav",
      "tags": ["proper-names", "overlap"],
      "reference_text": "Manually corrected words go here.",
      "reference_words": [
        {"word": "Manually", "start": 0.12, "end": 0.48},
        {"word": "corrected", "start": 0.49, "end": 0.91}
      ]
    },
    {
      "id": "tutorial-pause-and-terms",
      "audio": "audio/tutorial-pause-and-terms.wav",
      "tags": ["technical-terms", "pause"],
      "reference_text": "A second manually corrected transcript."
    }
  ]
}
```

Audio paths are relative to the manifest unless absolute. Reference word timestamps are optional; without them, that sample's timestamp metrics are reported as unavailable. Timestamps must be in seconds relative to the beginning of the audio sample. Reference word text should match the corrected transcript and preserve the words actually spoken.

Keep the manifest, audio, and generated report private if the recordings are not public. Do not put API keys in the manifest or results.

## Run

From `backend/`, configure `GROQ_API_KEY` in the process environment or an ignored local `.env`, then run:

```bash
PYTHONPATH=. python scripts/benchmark_transcription.py \
  --manifest /path/to/whisper-reference-set.json \
  --output ../output/whisper-benchmark.json
```

The default runs both models sequentially against every sample. To run one model or use a different provider model ID, pass `--models model-id ...`. The current pipeline default remains Turbo regardless of benchmark results.

Transient network/provider errors and rate limits are retried with bounded backoff. If an API call still fails (including unsupported model IDs), the report records its category and continues with the remaining model/sample pairs. The command exits non-zero if any pair failed, but still writes the report.

## Metrics

- **WER** is `(substitutions + deletions + insertions) / reference words`; lower is better. Deletions are omitted speech; insertions are extra/hallucinated words; substitutions are incorrect words.
- **Timestamp MAE** compares start/end times only for exactly aligned words with manual word-level reference times. Substituted or omitted words are not presented as successful timing matches. `within_tolerance_rate` reports the share within 250 ms on both boundaries.
- **Per-tag results** help compare cases such as proper names, technical terms, pauses, and overlap. They are only meaningful if those tags are used consistently and the sample is large enough.

Compare aggregate and per-sample results; do not change the production default from a tiny sample or a single metric. A real benchmark still requires actual representative audio and manually corrected references; mocked tests validate the code, not model accuracy.
