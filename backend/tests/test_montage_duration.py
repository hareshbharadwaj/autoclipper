import os

from autoclipper import pipeline
from autoclipper import montage as montage_module


def test_pipeline_reports_actual_montage_duration_without_cutting_transcript_segments(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"stub")
    transcript = {
        "duration": 102,
        "language": "English",
        "segments": [
            {"start": 0, "end": 17, "text": "First complete thought."},
            {"start": 17, "end": 34, "text": "Second complete thought."},
            {"start": 34, "end": 51, "text": "Third complete thought."},
            {"start": 51, "end": 68, "text": "Fourth complete thought."},
            {"start": 68, "end": 85, "text": "Fifth complete thought."},
            {"start": 85, "end": 102, "text": "Sixth complete thought."},
        ],
        "words": [],
    }
    selected = [
        {"title": "First", "start": 0, "end": 17},
        {"title": "Third", "start": 34, "end": 51},
        {"title": "Fifth", "start": 68, "end": 85},
    ]
    captured = {}
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(video_path), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 1920, "height": 1080, "duration": 102, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "probe_duration", lambda path: 50.5)
    monkeypatch.setattr(
        pipeline,
        "cut_all",
        lambda video, clips, out_dir, title: captured.setdefault(
            "clips",
            [{"index": i + 1, "file": f"clip_{i + 1}.mp4", **clip} for i, clip in enumerate(clips)],
        ),
    )

    def fake_stitch(paths, output_path, crossfade_seconds=0.25):
        captured["montage_inputs"] = paths
        with open(output_path, "wb") as output:
            output.write(b"synthetic montage")
        return output_path

    monkeypatch.setattr(montage_module, "stitch_montage", fake_stitch)
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path), "checks": {},
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
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
        montage=True,
        selected_clips=selected,
        target_duration=60,
    )

    assert [(clip["start"], clip["end"]) for clip in captured["clips"]] == [
        (0, 17), (34, 51), (68, 85),
    ]
    assert len(captured["montage_inputs"]) == 3
    assert result["individual_clip_count"] == 3
    assert result["montage_file"] == "source_montage.mp4"
    montage_result = result["clips"][-1]
    assert montage_result["is_montage"] is True
    assert montage_result["duration"] == 50.5
    assert montage_result["duration_delta"] == -9.5
    assert any("actual duration is 50.50s" in warning for warning in result["processing_warnings"])
