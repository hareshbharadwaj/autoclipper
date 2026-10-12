import os

from autoclipper import pipeline


def test_pipeline_restores_previous_verified_render_when_final_qa_fails(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source placeholder")
    output_dir = tmp_path / "job"
    transcript = {
        "duration": 30,
        "language": "English",
        "segments": [{"start": 0, "end": 30, "text": "A coherent transcript."}],
        "words": [],
    }

    monkeypatch.setattr(pipeline.config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(source), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda _path: {
        "width": 1920, "height": 1080, "duration": 30, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *args, **kwargs: [
        {"start": 5, "end": 15, "title": "A moment", "candidate_score": 80},
    ])

    def fake_cut_all(_video, candidates, folder, _title):
        os.makedirs(folder, exist_ok=True)
        (output_dir / "clip_1.mp4").write_bytes(b"verified original render")
        return [{"index": 1, "file": "clip_1.mp4", **candidates[0]}]

    monkeypatch.setattr(pipeline, "cut_all", fake_cut_all)

    def fake_reframe(_src, output, **_kwargs):
        with open(output, "wb") as handle:
            handle.write(b"bad reframe")
        return True

    monkeypatch.setattr(pipeline, "reframe_clip", fake_reframe)
    qa_calls = []

    def fake_qa(path, **_kwargs):
        qa_calls.append(os.path.basename(path))
        if os.path.basename(path) == "clip_1_9x16.mp4":
            return {
                "status": "fail", "checks": {}, "warnings": [],
                "errors": ["Synthetic final-render verification failure."],
            }
        return {
            "status": "fail",
            "checks": {
                "file": {"ok": True}, "streams": {"ok": True},
                "duration": {"ok": True}, "audio": {"ok": True},
                "visual_samples": {"ok": True}, "aspect_ratio": {"ok": False},
            },
            "warnings": [],
            "errors": ["Output dimensions do not match the requested aspect ratio."],
        }

    monkeypatch.setattr(pipeline, "verify_rendered_file", fake_qa)
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        "synthetic-source",
        output_dir=str(output_dir),
        clip_count=1,
        min_clip=5,
        max_clip=20,
        subtitles=False,
    )

    assert result["clips"][0]["file"] == "clip_1.mp4"
    assert result["clips"][0]["qa_status"] == "warning"
    assert result["quality_assurance"]["reports"][0]["fallback_used"] is True
    assert "clip_1_9x16.mp4" in qa_calls
    assert "clip_1.mp4" in qa_calls
    assert os.path.exists(output_dir / "clip_1.mp4")
    assert not os.path.exists(output_dir / "clip_1_9x16.mp4")
    assert any("earlier render was restored" in warning for warning in result["processing_warnings"])
    assert any("final render failed this check" in warning.lower() for warning in result["processing_warnings"])


def test_pipeline_excludes_unplayable_outputs_and_recomputes_counts_after_qa(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source placeholder")
    output_dir = tmp_path / "job"
    transcript = {
        "duration": 30,
        "language": "English",
        "segments": [{"start": 0, "end": 30, "text": "A coherent transcript."}],
        "words": [],
    }
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(source), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda _path: {
        "width": 1920, "height": 1080, "duration": 30, "has_audio": True,
    })
    monkeypatch.setattr(pipeline.config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *args, **kwargs: [
        {"start": 5, "end": 15, "title": "A moment"},
    ])
    monkeypatch.setattr(pipeline, "cut_all", lambda *_args: [
        {"index": 1, "file": "missing.mp4", "start": 5, "end": 15, "title": "A moment"},
    ])
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        "synthetic-source",
        output_dir=str(output_dir),
        vertical=False,
        subtitles=False,
        clip_count=1,
        min_clip=5,
        max_clip=20,
    )

    assert result["clips"] == []
    assert result["individual_clip_count"] == 0
    assert result["quality_assurance"]["status"] == "fail"
    assert result["quality_assurance"]["failed_clip_count"] == 1
    assert result["quality_assurance"]["failed_outputs"][0]["file"] == "missing.mp4"
    assert any("passed render QA" in warning for warning in result["processing_warnings"])
