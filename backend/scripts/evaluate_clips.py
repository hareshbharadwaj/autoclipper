#!/usr/bin/env python3
"""Evaluate AutoClipper intervals against authorized manual references.

By default this is offline and reads generated metadata files. Pass --run-pipeline
only when you intentionally want to make real provider calls for an authorized
local source collection.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from autoclipper import config  # noqa: E402
from autoclipper.video_benchmark import (  # noqa: E402
    build_video_evaluation_report,
    load_video_evaluation_manifest,
)

_ALLOWED_PIPELINE_OPTIONS = {
    "clip_count", "min_clip", "max_clip", "target_duration", "aspect_ratio",
    "subtitles", "vertical", "use_yolo", "content_type", "force_hd",
}


def _safe_slug(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "sample")).strip("._")
    return value[:80] or "sample"


def _run_authorized_pipeline(manifest, artifact_dir):
    """Explicit opt-in path: generate clips for each manifest-authorized source."""
    api_key = os.environ.get("GROQ_API_KEY") or config.GROQ_API_KEY
    gemini_key = os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY
    if not api_key or not gemini_key:
        raise ValueError(
            "--run-pipeline requires GROQ_API_KEY and GEMINI_API_KEY in the environment "
            "or local ignored backend .env; credentials are never read from the manifest."
        )

    from autoclipper.pipeline import run as run_pipeline

    os.makedirs(artifact_dir, exist_ok=True)
    generated = {}
    for sample in manifest["samples"]:
        options = sample.get("pipeline_options") or {}
        if not isinstance(options, dict):
            raise ValueError(f"pipeline_options for sample '{sample['id']}' must be an object.")
        unknown_options = set(options) - _ALLOWED_PIPELINE_OPTIONS
        if unknown_options:
            raise ValueError(
                f"Unsupported pipeline_options for sample '{sample['id']}': "
                + ", ".join(sorted(unknown_options))
            )
        output_dir = os.path.join(artifact_dir, _safe_slug(sample["id"]))
        result = run_pipeline(
            sample["source"],
            output_dir=output_dir,
            groq_key=api_key,
            gemini_key=gemini_key,
            whisper_model=config.GROQ_WHISPER_MODEL,
            gemini_model=config.GEMINI_MODEL,
            **options,
        )
        generated[sample["id"]] = [
            clip for clip in result.get("clips", [])
            if isinstance(clip, dict) and not clip.get("is_montage")
        ]
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Authorized manual-reference JSON manifest.")
    parser.add_argument("--output", required=True, help="Path for the evaluation JSON report; keep it out of Git.")
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5,
        help="Temporal intersection-over-union threshold for one-to-one matching (default: 0.5).",
    )
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Opt in to real Groq/Gemini calls; processes only manifest-authorized local files.",
    )
    parser.add_argument(
        "--artifact-dir", default=None,
        help="Where generated media/metadata go with --run-pipeline (default: backend/output/video-evaluation/<UTC timestamp>).",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1.")

    try:
        manifest = load_video_evaluation_manifest(args.manifest)
        generated = None
        artifact_dir = None
        if args.run_pipeline:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            artifact_dir = os.path.abspath(
                args.artifact_dir or os.path.join(config.OUTPUT_ROOT, "video-evaluation", stamp)
            )
            generated = _run_authorized_pipeline(manifest, artifact_dir)
        report = build_video_evaluation_report(
            manifest, generated_by_id=generated, iou_threshold=args.iou_threshold
        )
        # The report is honest about the execution path and does not save secrets.
        report["execution"] = {
            "pipeline_run_requested": bool(args.run_pipeline),
            "artifact_dir": os.path.basename(artifact_dir) if artifact_dir else None,
            "provider_keys_recorded": False,
        }
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Evaluation report saved to {output_path}")
    print(f"Temporal span F1: {report['summary']['temporal']['f1']:.4f}")
    human = report["summary"]["human_judgments"]
    print(f"Human reviewer judgments recorded: {human['reviewer_judgment_count']}")
    if report["warnings"]:
        for warning in report["warnings"]:
            print(f"Note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
