import os

import pytest

from autoclipper import pipeline


def test_clip_count_validation_accepts_supported_values_and_rejects_invalid():
    assert pipeline._normalize_clip_count(None) is None
    assert pipeline._normalize_clip_count("1") == 1
    assert pipeline._normalize_clip_count(10) == 10
    assert pipeline._normalize_clip_count(30) == 30
    for invalid in (0, -1, 31, 1.5, True, "many"):
        with pytest.raises(ValueError, match="whole number between 1 and 30"):
            pipeline._normalize_clip_count(invalid)


def test_selected_angle_is_kept_and_duplicate_moments_do_not_fill_clip_count():
    selected = [{"start": 10, "end": 20, "title": "Chosen angle"}]
    generated = [
        {"start": 12, "end": 22, "title": "Duplicate of selected"},
        {"start": 30, "end": 40, "title": "Second"},
        {"start": 50, "end": 60, "title": "Third"},
    ]
    result = pipeline._merge_unique_clip_candidates(selected, generated, clip_count=3)
    assert len(result) == 3
    assert {item["title"] for item in result} == {"Chosen angle", "Second", "Third"}


@pytest.mark.parametrize(("requested_count", "available_count"), [(3, 3), (5, 2)])
def test_pipeline_reports_actual_individual_clip_count_and_shortfall(
    tmp_path, monkeypatch, requested_count, available_count
):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"stub")
    transcript = {
        "duration": 100,
        "language": "English",
        "segments": [{"start": 0, "end": 100, "text": "A transcript."}],
        "words": [],
    }
    candidates = [
        {"start": 5, "end": 15, "title": "One"},
        {"start": 25, "end": 35, "title": "Two"},
        {"start": 50, "end": 60, "title": "Three"},
    ]
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(video_path), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 1920, "height": 1080, "duration": 100, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(
        pipeline,
        "find_viral_clips",
        lambda *args, **kwargs: candidates[:available_count],
    )
    monkeypatch.setattr(
        pipeline,
        "cut_all",
        lambda video, clips, out_dir, title: [
            {"index": i + 1, "file": f"clip_{i + 1}.mp4", **candidate}
            for i, candidate in enumerate(clips)
        ],
    )
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path), "checks": {},
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    progress_events = []
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        source="stub-source",
        output_dir=str(tmp_path / "job"),
        vertical=False,
        subtitles=False,
        clip_count=requested_count,
        min_clip=5,
        max_clip=30,
        on_progress=progress_events.append,
    )
    assert [event["stage"] for event in progress_events][0] == "ingesting"
    assert "reviewing_candidates" in [event["stage"] for event in progress_events]
    assert "quality_assurance" in [event["stage"] for event in progress_events]
    assert progress_events[-1]["stage"] == "completed"
    assert result["quality_assurance"]["status"] == "pass"
    assert result["requested_clip_count"] == requested_count
    assert result["individual_clip_count"] == available_count
    assert len(result["clips"]) == available_count
    if available_count < requested_count:
        assert any(
            "requested 5 individual clips" in warning.lower()
            and "generated 2" in warning.lower()
            for warning in result["processing_warnings"]
        )
    else:
        assert result["processing_warnings"] == []
