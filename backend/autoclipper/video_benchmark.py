"""Offline temporal and human-judgment evaluation for generated video clips.

This module never creates references or ratings. Evaluation inputs must be
manually prepared and the manifest must attest that every source is authorized.
"""
import json
import math
import os
from datetime import datetime, timezone


RATING_MIN = 1
RATING_MAX = 5
HUMAN_RATING_DIMENSIONS = (
    "hook_strength",
    "coherence",
    "transcript_accuracy",
    "visual_fit",
    "standalone_value",
    "overall",
)


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _validate_span(span, name, duration=None):
    if not isinstance(span, dict):
        raise ValueError(f"{name} must be an object with start/end seconds.")
    start = _finite_number(span.get("start"), f"{name}.start")
    end = _finite_number(span.get("end"), f"{name}.end")
    if start < 0 or end <= start:
        raise ValueError(f"{name} has invalid start/end timestamps.")
    if duration is not None and end > float(duration) + 0.25:
        raise ValueError(f"{name} extends beyond the declared source duration.")
    return {"start": start, "end": end}


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation JSON: {os.path.basename(path)} ({type(exc).__name__}).") from exc


def load_video_evaluation_manifest(path):
    """Load and validate a version-1 manual evaluation manifest."""
    path = os.path.abspath(path)
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Video evaluation manifest must be a JSON object with version: 1.")
    if payload.get("authorized_for_evaluation") is not True:
        raise ValueError(
            "Manifest must set authorized_for_evaluation: true after confirming permission to use every source."
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Manifest must contain a non-empty samples array.")

    base_dir = os.path.dirname(path)
    normalized = []
    seen_ids = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object.")
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"samples[{index}].id must be non-empty and unique.")
        seen_ids.add(sample_id)
        if sample.get("authorized") is not True:
            raise ValueError(f"Sample '{sample_id}' must set authorized: true.")
        source = str(sample.get("source") or "").strip()
        if not source:
            raise ValueError(f"Sample '{sample_id}' needs a local source file path.")
        source_path = os.path.abspath(os.path.join(base_dir, source)) if not os.path.isabs(source) else source
        if not os.path.isfile(source_path):
            raise ValueError(f"Authorized source file for sample '{sample_id}' does not exist.")

        duration = None
        if sample.get("duration") is not None:
            duration = _finite_number(sample["duration"], f"Sample '{sample_id}' duration")
            if duration <= 0:
                raise ValueError(f"Sample '{sample_id}' duration must be greater than zero.")

        references = sample.get("manual_references", sample.get("references"))
        if not isinstance(references, list) or not references:
            raise ValueError(f"Sample '{sample_id}' needs at least one manually annotated reference span.")
        checked_references = []
        for ref_index, reference in enumerate(references):
            checked = _validate_span(reference, f"Sample '{sample_id}' reference {ref_index + 1}", duration)
            checked["label"] = str(reference.get("label") or "")[:160]
            checked_references.append(checked)

        generated_metadata = sample.get("generated_metadata")
        if generated_metadata is not None:
            if not isinstance(generated_metadata, str) or not generated_metadata.strip():
                raise ValueError(f"Generated metadata path for sample '{sample_id}' must be a file path.")
            generated_metadata = (
                os.path.abspath(os.path.join(base_dir, generated_metadata))
                if not os.path.isabs(generated_metadata)
                else os.path.abspath(generated_metadata)
            )
            if not os.path.isfile(generated_metadata):
                raise ValueError(f"Generated metadata for sample '{sample_id}' does not exist.")
        generated_clips = sample.get("generated_clips")
        if generated_clips is not None and not isinstance(generated_clips, list):
            raise ValueError(f"generated_clips for sample '{sample_id}' must be an array.")
        if generated_metadata is None and generated_clips is None:
            raise ValueError(
                f"Sample '{sample_id}' needs generated_metadata or a generated_clips array."
            )
        tags = sample.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"Tags for sample '{sample_id}' must be an array of strings.")
        human_judgments = sample.get("human_judgments") or []
        if not isinstance(human_judgments, list):
            raise ValueError(f"human_judgments for sample '{sample_id}' must be an array.")
        pipeline_options = sample.get("pipeline_options") or {}
        if not isinstance(pipeline_options, dict):
            raise ValueError(f"pipeline_options for sample '{sample_id}' must be an object.")

        normalized.append({
            "id": sample_id,
            "source": source_path,
            "duration": duration,
            "tags": [tag.strip()[:80] for tag in tags if tag.strip()],
            "manual_references": checked_references,
            "generated_metadata": generated_metadata,
            "generated_clips": generated_clips,
            "human_judgments": human_judgments,
            "pipeline_options": pipeline_options,
        })
    return {"version": 1, "manifest_path": path, "samples": normalized}


def _temporal_iou(left, right):
    intersection = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return intersection / union if union > 0 else 0.0


def temporal_span_metrics(generated, references, iou_threshold=0.5, unmatched_generated_count=0):
    """Greedy one-to-one temporal matching using intersection over union."""
    generated = list(generated or [])
    references = list(references or [])
    try:
        unmatched_generated_count = max(0, int(unmatched_generated_count))
    except (TypeError, ValueError):
        unmatched_generated_count = 0
    pairs = []
    for gen_index, gen in enumerate(generated):
        for ref_index, ref in enumerate(references):
            iou = _temporal_iou(gen, ref)
            if iou > 0.0 and iou >= float(iou_threshold):
                pairs.append((iou, gen_index, ref_index))
    pairs.sort(reverse=True)
    used_generated, used_references, matched_ious = set(), set(), []
    for iou, gen_index, ref_index in pairs:
        if gen_index in used_generated or ref_index in used_references:
            continue
        used_generated.add(gen_index)
        used_references.add(ref_index)
        matched_ious.append(iou)

    total_generated = len(generated) + unmatched_generated_count
    precision = len(used_generated) / total_generated if total_generated else 0.0
    recall = len(used_references) / len(references) if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou_threshold": float(iou_threshold),
        "generated_count": total_generated,
        "valid_generated_count": len(generated),
        "invalid_generated_count": unmatched_generated_count,
        "reference_count": len(references),
        "matched_count": len(matched_ious),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_matched_iou": round(sum(matched_ious) / len(matched_ious), 4) if matched_ious else 0.0,
        "evaluated": bool(references),
    }


def summarize_human_judgments(judgments):
    """Aggregate only ratings explicitly entered by a human, on a 1-5 scale."""
    totals = {dimension: [] for dimension in HUMAN_RATING_DIMENSIONS}
    reviewer_count = 0
    for judgment in judgments or []:
        if not isinstance(judgment, dict):
            continue
        ratings = judgment.get("ratings") or {}
        valid_any = False
        for dimension in HUMAN_RATING_DIMENSIONS:
            if dimension not in ratings:
                continue
            try:
                score = _finite_number(ratings[dimension], f"human rating {dimension}")
            except ValueError:
                continue
            if score.is_integer() and RATING_MIN <= score <= RATING_MAX:
                totals[dimension].append(int(score))
                valid_any = True
        if valid_any:
            reviewer_count += 1
    return {
        "reviewer_judgment_count": reviewer_count,
        "rating_count_by_dimension": {key: len(values) for key, values in totals.items()},
        "average_by_dimension": {
            key: round(sum(values) / len(values), 3) if values else None
            for key, values in totals.items()
        },
    }


def _metadata_clips(path):
    payload = _load_json(path)
    clips = payload.get("clips") if isinstance(payload, dict) else None
    if not isinstance(clips, list):
        raise ValueError(f"Generated metadata has no clips array: {os.path.basename(path)}.")
    return [clip for clip in clips if isinstance(clip, dict) and not clip.get("is_montage")]


def _checked_generated_clips(clips, duration):
    checked = []
    invalid_count = 0
    for index, clip in enumerate(clips or []):
        try:
            span = _validate_span(clip, f"generated clip {index + 1}", duration)
        except ValueError:
            invalid_count += 1
            continue
        qa_report = clip.get("quality_assurance")
        if not isinstance(qa_report, dict):
            qa_report = {}
        qa_status = clip.get("qa_status") or qa_report.get("status")
        checked.append({
            **span,
            "title": str(clip.get("title") or "")[:160],
            "qa_status": qa_status if isinstance(qa_status, str) else None,
        })
    return checked, invalid_count


def _aggregate_temporal(sample_metrics, iou_threshold=0.5):
    generated = sum(row["temporal"]["generated_count"] for row in sample_metrics)
    references = sum(row["temporal"]["reference_count"] for row in sample_metrics)
    matched = sum(row["temporal"]["matched_count"] for row in sample_metrics)
    precision = matched / generated if generated else 0.0
    recall = matched / references if references else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ious = [
        row["temporal"]["mean_matched_iou"]
        for row in sample_metrics
        if row["temporal"]["matched_count"]
    ]
    return {
        "iou_threshold": float(iou_threshold),
        "sample_count": len(sample_metrics),
        "generated_count": generated,
        "reference_count": references,
        "matched_count": matched,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_sample_matched_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }


def build_video_evaluation_report(manifest, generated_by_id=None, iou_threshold=0.5):
    """Score generated intervals against manual spans and aggregate human ratings."""
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1.")
    generated_by_id = generated_by_id or {}
    per_sample = []
    all_judgments = []
    qa_status_counts = {"pass": 0, "warning": 0, "fail": 0, "unknown": 0}

    for sample in manifest["samples"]:
        if sample["id"] in generated_by_id:
            raw_clips = generated_by_id[sample["id"]]
        elif sample.get("generated_metadata"):
            raw_clips = _metadata_clips(sample["generated_metadata"])
        else:
            raw_clips = sample.get("generated_clips") or []
        generated, invalid_generated_count = _checked_generated_clips(raw_clips, sample.get("duration"))
        temporal = temporal_span_metrics(
            generated,
            sample["manual_references"],
            iou_threshold,
            unmatched_generated_count=invalid_generated_count,
        )
        human = summarize_human_judgments(sample.get("human_judgments"))
        all_judgments.extend(sample.get("human_judgments") or [])
        for clip in generated:
            qa_status_counts[clip.get("qa_status") if clip.get("qa_status") in {"pass", "warning", "fail"} else "unknown"] += 1
        qa_status_counts["unknown"] += invalid_generated_count
        per_sample.append({
            "id": sample["id"],
            "tags": sample["tags"],
            "generated_count": temporal["generated_count"],
            "valid_generated_count": len(generated),
            "invalid_generated_count": invalid_generated_count,
            "manual_reference_count": len(sample["manual_references"]),
            "manual_references": sample["manual_references"],
            "temporal": temporal,
            "human_judgments": human,
            "generated_clips": generated,
        })

    warnings = []
    if not any(row["human_judgments"]["reviewer_judgment_count"] for row in per_sample):
        warnings.append("No human ratings were supplied; subjective quality averages are intentionally empty.")
    if any(row["invalid_generated_count"] for row in per_sample):
        warnings.append("Invalid generated clip intervals were counted as unmatched predictions.")

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": os.path.basename(manifest["manifest_path"]),
        "authorization_attested": True,
        "summary": {
            "temporal": _aggregate_temporal(per_sample, iou_threshold),
            "human_judgments": summarize_human_judgments(all_judgments),
            "clip_qa_status_counts": qa_status_counts,
        },
        "per_sample": per_sample,
        "warnings": warnings,
        "interpretation": (
            "Temporal overlap measures agreement with the supplied manual spans, not virality or causal performance. "
            "Human ratings are reported only when explicitly provided."
        ),
    }
