"""Post-render QA must deliver, repair, explicitly downgrade, or fail requested captions."""
import os

from autoclipper import pipeline


WORDS = [
    {"start": 5.0, "end": 7.0, "word": "A"},
    {"start": 7.0, "end": 9.0, "word": "moment"},
]

BASE_CHECKS = {
    "file": {"ok": True},
    "streams": {"ok": True},
    "duration": {"ok": True},
    "audio": {"ok": True},
    "visual_samples": {"ok": True},
    "aspect_ratio": {"ok": True},
}

SUBTITLE_ERROR = (
    "Requested subtitles could not be verified: no burned-caption output file was produced."
)


def _subtitle_failure_report():
    return {
        "status": "fail",
        "checks": {
            **BASE_CHECKS,
            "subtitles": {
                "ok": False,
                "present": True,
                "valid": True,
                "burned_output_present": False,
            },
        },
        "warnings": [],
        "errors": [SUBTITLE_ERROR],
    }


def _prepare(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source placeholder")
    output_dir = tmp_path / "job"
    transcript = {
        "duration": 30,
        "language": "English",
        "segments": [{"start": 0, "end": 30, "text": "A coherent transcript."}],
        "words": WORDS,
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
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    def fake_cut_all(_video, candidates, folder, _title):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "clip_1.mp4"), "wb") as handle:
            handle.write(b"verified original render")
        return [{"index": 1, "file": "clip_1.mp4", **candidates[0]}]

    monkeypatch.setattr(pipeline, "cut_all", fake_cut_all)
    return output_dir


def _run(output_dir):
    return pipeline.run(
        "synthetic-source",
        output_dir=str(output_dir),
        vertical=False,
        clip_count=1,
        min_clip=5,
        max_clip=20,
        subtitles=True,
    )


def test_pipeline_repairs_failed_subtitle_burn_on_second_attempt(tmp_path, monkeypatch):
    output_dir = _prepare(tmp_path, monkeypatch)
    calls = {"captions": 0, "qa": []}

    def fake_add_subtitles(clips, transcript, output_dir, burn=True, preserve_source=False):
        calls["captions"] += 1
        for clip in clips:
            srt_path = os.path.join(output_dir, "clip_1.srt")
            with open(srt_path, "w", encoding="utf-8") as handle:
                handle.write("1\n00:00:00,200 --> 00:00:01,200\nA moment\n")
            clip["srt"] = "clip_1.srt"
            if calls["captions"] == 1:
                clip["subtitle_warning"] = (
                    "Captions were generated as SRT but could not be burned in: boom"
                )
            else:
                with open(os.path.join(output_dir, "clip_1_sub.mp4"), "wb") as handle:
                    handle.write(b"repaired subtitled render")
                clip["subtitled_file"] = "clip_1_sub.mp4"
                clip["file"] = "clip_1_sub.mp4"
                clip.pop("subtitle_warning", None)
        return clips

    def fake_qa(path, **kwargs):
        name = os.path.basename(path)
        calls["qa"].append(name)
        if name == "clip_1_sub.mp4":
            return {
                "status": "pass",
                "checks": {**BASE_CHECKS, "subtitles": {"ok": True}},
                "warnings": [],
                "errors": [],
            }
        return _subtitle_failure_report()

    monkeypatch.setattr(pipeline, "add_subtitles", fake_add_subtitles)
    monkeypatch.setattr(pipeline, "verify_rendered_file", fake_qa)

    result = _run(output_dir)

    assert calls["captions"] == 2  # initial attempt + one repair attempt
    assert calls["qa"] == ["clip_1.mp4", "clip_1_sub.mp4"]
    clip = result["clips"][0]
    assert clip["file"] == "clip_1_sub.mp4"
    assert clip["subtitled_file"] == "clip_1_sub.mp4"
    assert clip["qa_status"] == "pass"
    assert "subtitle_warning" not in clip
    assert any("captions were repaired" in warning.lower() for warning in result["processing_warnings"])
    # The captionless interim render was superseded and cleaned up.
    assert os.path.exists(output_dir / "clip_1_sub.mp4")
    assert not os.path.exists(output_dir / "clip_1.mp4")
    assert result["quality_assurance"]["status"] == "pass"


def test_pipeline_downgrades_impossible_captions_to_explicit_warning(tmp_path, monkeypatch):
    output_dir = _prepare(tmp_path, monkeypatch)
    calls = {"captions": 0}

    def fake_add_subtitles(clips, transcript, output_dir, burn=True, preserve_source=False):
        calls["captions"] += 1
        for clip in clips:
            clip["subtitle_warning"] = (
                "No transcript words overlap this clip; captions were not generated."
            )
        return clips

    monkeypatch.setattr(pipeline, "add_subtitles", fake_add_subtitles)
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: _subtitle_failure_report())

    result = _run(output_dir)

    clip = result["clips"][0]
    assert clip["file"] == "clip_1.mp4"
    assert clip["qa_status"] == "warning"  # acceptable fallback, not a silent pass
    assert calls["captions"] == 1  # impossible case must not trigger a burn retry
    assert any(
        "no transcript words overlap" in warning.lower() and "without captions" in warning.lower()
        for warning in result["processing_warnings"]
    )
    assert result["quality_assurance"]["status"] == "warning"
    assert result["quality_assurance"]["failed_clip_count"] == 0


def test_pipeline_excludes_clip_when_caption_repair_also_fails(tmp_path, monkeypatch):
    output_dir = _prepare(tmp_path, monkeypatch)
    calls = {"captions": 0}

    def fake_add_subtitles(clips, transcript, output_dir, burn=True, preserve_source=False):
        calls["captions"] += 1
        for clip in clips:
            srt_path = os.path.join(output_dir, "clip_1.srt")
            with open(srt_path, "w", encoding="utf-8") as handle:
                handle.write("1\n00:00:00,200 --> 00:00:01,200\nA moment\n")
            clip["srt"] = "clip_1.srt"
            clip["subtitle_warning"] = (
                "Captions were generated as SRT but could not be burned in: boom"
            )
        return clips

    monkeypatch.setattr(pipeline, "add_subtitles", fake_add_subtitles)
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: _subtitle_failure_report())

    result = _run(output_dir)

    assert calls["captions"] == 2  # repair was attempted and failed
    assert result["clips"] == []
    qa_summary = result["quality_assurance"]
    assert qa_summary["status"] == "fail"
    assert qa_summary["failed_clip_count"] == 1
    failed = qa_summary["failed_outputs"][0]
    assert any("requested subtitles" in error.lower() for error in failed["quality_assurance"]["errors"])
    assert any("requested subtitles" in warning.lower() for warning in result["processing_warnings"])
