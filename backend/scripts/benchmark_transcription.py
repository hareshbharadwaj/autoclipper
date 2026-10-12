#!/usr/bin/env python3
"""Run Groq Whisper models against manually corrected audio references."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

# Support both `python scripts/benchmark_transcription.py` from backend/ and
# `python backend/scripts/benchmark_transcription.py` from the repository root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from autoclipper import config  # noqa: E402 - config loads a local ignored .env if present
from autoclipper.transcription_benchmark import (  # noqa: E402
    DEFAULT_BENCHMARK_MODELS,
    run_transcription_benchmark,
)


def load_manifest(path):
    """Read a sample manifest and resolve audio paths relative to that manifest."""
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("Manifest must be a JSON object with a 'samples' list.")
    if not manifest["samples"]:
        raise ValueError("Manifest 'samples' must contain at least one audio sample.")

    samples = []
    seen_ids = set()
    for index, raw in enumerate(manifest["samples"]):
        if not isinstance(raw, dict):
            raise ValueError(f"samples[{index}] must be an object.")
        sample_id = str(raw.get("id") or f"sample-{index + 1}").strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"samples[{index}] has a missing or duplicate id.")
        seen_ids.add(sample_id)

        audio_value = raw.get("audio")
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise ValueError(f"samples[{index}].audio must be a file path.")
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise ValueError(f"Audio file for sample '{sample_id}' does not exist: {audio_path}")

        reference_text = raw.get("reference_text")
        reference_words = raw.get("reference_words", [])
        if not isinstance(reference_words, list):
            raise ValueError(f"samples[{index}].reference_words must be a list when provided.")
        if reference_text is not None and not isinstance(reference_text, str):
            raise ValueError(f"samples[{index}].reference_text must be a string.")
        if reference_text is None and not reference_words:
            raise ValueError(f"sample '{sample_id}' needs reference_text or reference_words.")
        for word_index, word in enumerate(reference_words):
            if not isinstance(word, dict) or not isinstance(word.get("word", word.get("text")), str):
                raise ValueError(f"samples[{index}].reference_words[{word_index}] needs word/text.")
            try:
                start, end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"samples[{index}].reference_words[{word_index}] needs numeric start/end.")
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError(f"samples[{index}].reference_words[{word_index}] has invalid timestamps.")

        tags = raw.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"samples[{index}].tags must be a list of strings.")
        samples.append({
            "id": sample_id,
            "audio": str(audio_path),
            "tags": tags,
            "reference_text": reference_text,
            "reference_words": reference_words,
        })
    return samples


def _format_metric(value, suffix=""):
    return "n/a" if value is None else f"{value:.4f}{suffix}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest with audio and manual references.")
    parser.add_argument("--output", required=True, help="Path for the JSON results file (keep it out of Git).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_MODELS),
        help="Groq model IDs; defaults to Turbo and Large V3.",
    )
    parser.add_argument("--api-key-env", default="GROQ_API_KEY", help="Environment variable holding the Groq key.")
    args = parser.parse_args(argv)

    try:
        samples = load_manifest(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))
    api_key = os.environ.get(args.api_key_env) or config.GROQ_API_KEY
    if not api_key:
        parser.error(f"Set {args.api_key_env} in the environment or local ignored .env; never put keys in the manifest.")

    try:
        report = run_transcription_benchmark(samples, api_key, args.models)
    except ValueError as exc:
        parser.error(str(exc))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Benchmark report saved to {output_path}")
    for model, metrics in report["summary"]["by_model"].items():
        print(
            f"{model}: WER={_format_metric(metrics['wer'])}, "
            f"boundary MAE={_format_metric(metrics['boundary_mae_seconds'], 's')}, "
            f"samples={metrics['sample_count']}, failed={metrics.get('failed_sample_count', 0)}"
        )
    failures = sum(record["status"] != "ok" for record in report["results"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
