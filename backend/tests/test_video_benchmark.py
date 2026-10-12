import json

import pytest

from autoclipper.video_benchmark import (
    build_video_evaluation_report,
    load_video_evaluation_manifest,
    summarize_human_judgments,
    temporal_span_metrics,
)
from scripts.evaluate_clips import main as evaluate_clips_main


def test_temporal_matching_is_one_to_one_and_reports_overlap_metrics():
    report = temporal_span_metrics(
        [{"start": 1, "end": 9}, {"start": 1.2, "end": 8.8}],
        [{"start": 0, "end": 10}],
        iou_threshold=0.5,
    )
    assert report["generated_count"] == 2
    assert report["reference_count"] == 1
    assert report["matched_count"] == 1
    assert report["precision"] == 0.5
    assert report["recall"] == 1.0
    assert report["f1"] == pytest.approx(2 / 3, abs=0.0001)
    assert report["mean_matched_iou"] == 0.8
    assert temporal_span_metrics(
        [{"start": 10, "end": 12}], [{"start": 0, "end": 2}], iou_threshold=0.0
    )["matched_count"] == 0


def test_human_averages_only_include_explicit_valid_1_to_5_ratings():
    report = summarize_human_judgments([
        {"reviewer": "one", "ratings": {"overall": 4, "visual_fit": 5}},
        {"reviewer": "two", "ratings": {"overall": 2, "visual_fit": 7}},
        {"reviewer": "three", "ratings": {"overall": "not scored"}},
    ])
    assert report["reviewer_judgment_count"] == 2
    assert report["rating_count_by_dimension"]["overall"] == 2
    assert report["average_by_dimension"]["overall"] == 3.0
    assert report["average_by_dimension"]["visual_fit"] == 5.0
    assert report["average_by_dimension"]["hook_strength"] is None


def test_manifest_requires_explicit_authorization_and_manual_references(tmp_path):
    source = tmp_path / "authorized.mp4"
    source.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    base = {
        "version": 1,
        "authorized_for_evaluation": True,
        "samples": [{
            "id": "sample-1",
            "source": source.name,
            "authorized": True,
            "duration": 20,
            "manual_references": [{"start": 2, "end": 7, "label": "manual highlight"}],
            "generated_clips": [{"start": 2.5, "end": 6.5, "qa_status": "pass"}],
        }],
    }
    manifest_path.write_text(json.dumps(base), encoding="utf-8")
    manifest = load_video_evaluation_manifest(manifest_path)
    report = build_video_evaluation_report(manifest)
    assert report["authorization_attested"] is True
    assert report["summary"]["temporal"]["f1"] == 1.0
    assert report["summary"]["clip_qa_status_counts"]["pass"] == 1
    assert report["summary"]["human_judgments"]["reviewer_judgment_count"] == 0
    assert report["summary"]["human_judgments"]["average_by_dimension"]["overall"] is None
    assert "No human ratings" in report["warnings"][0]

    base["authorized_for_evaluation"] = False
    manifest_path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized_for_evaluation"):
        load_video_evaluation_manifest(manifest_path)


def test_manifest_loads_generated_metadata_and_rejects_reference_outside_source(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"placeholder")
    metadata = tmp_path / "generated.json"
    metadata.write_text(json.dumps({
        "clips": [
            {"start": 4, "end": 9, "title": "Good", "quality_assurance": {"status": "warning"}},
            {"start": 0, "end": 0, "title": "Bad interval"},
            {"start": 1, "end": 3, "is_montage": True},
        ],
    }), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "version": 1,
        "authorized_for_evaluation": True,
        "samples": [{
            "id": "sample",
            "source": "source.mp4",
            "authorized": True,
            "duration": 10,
            "manual_references": [{"start": 4, "end": 9}],
            "generated_metadata": "generated.json",
        }],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_video_evaluation_manifest(manifest_path)
    report = build_video_evaluation_report(manifest)
    assert report["per_sample"][0]["generated_count"] == 2
    assert report["per_sample"][0]["valid_generated_count"] == 1
    assert report["per_sample"][0]["invalid_generated_count"] == 1
    assert report["summary"]["temporal"]["precision"] == 0.5
    assert report["summary"]["clip_qa_status_counts"]["warning"] == 1
    assert report["summary"]["clip_qa_status_counts"]["unknown"] == 1

    payload["samples"][0]["manual_references"] = [{"start": 9, "end": 11}]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="beyond the declared source duration"):
        load_video_evaluation_manifest(manifest_path)


def test_offline_cli_writes_report_without_inventing_human_ratings(tmp_path, capsys):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"authorized test placeholder")
    metadata = tmp_path / "job_metadata.json"
    metadata.write_text(json.dumps({"clips": [{"start": 2, "end": 7}]}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "authorized_for_evaluation": True,
        "samples": [{
            "id": "cli-sample",
            "source": source.name,
            "authorized": True,
            "duration": 10,
            "manual_references": [{"start": 2, "end": 7}],
            "generated_metadata": metadata.name,
        }],
    }), encoding="utf-8")
    output = tmp_path / "report.json"

    assert evaluate_clips_main([
        "--manifest", str(manifest), "--output", str(output), "--iou-threshold", "0.4",
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["temporal"]["iou_threshold"] == 0.4
    assert report["summary"]["human_judgments"]["average_by_dimension"]["overall"] is None
    assert report["execution"]["pipeline_run_requested"] is False
    assert "Human reviewer judgments recorded: 0" in capsys.readouterr().out
