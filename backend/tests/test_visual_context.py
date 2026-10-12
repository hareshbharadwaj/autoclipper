import json
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest

from autoclipper.visual_context import (
    extract_video_frames,
    sample_timestamps,
    sample_video_frames,
)


def test_sample_timestamps_are_evenly_distributed_and_strictly_bounded():
    times = sample_timestamps(900, max_frames=5)
    assert len(times) == 5
    assert times == pytest.approx([150, 300, 450, 600, 750])
    assert all(0 < time < 900 for time in times)
    assert len(sample_timestamps(40, max_frames=50)) == 1
    assert sample_timestamps(float("nan")) == []
    assert sample_timestamps(20, max_frames=0) == []


def test_extract_video_frames_keeps_samples_in_memory_and_crops_at_most_the_cap(
    tmp_path, monkeypatch
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=b"jpeg-frame", stderr=b"")

    monkeypatch.setattr("autoclipper.visual_context.subprocess.run", fake_run)
    result = extract_video_frames(
        str(video), [1, 2, 3, 4], max_frames=2, ffmpeg_path="/fake/ffmpeg"
    )

    assert [frame["timestamp"] for frame in result["frames"]] == [1.0, 2.0]
    assert all(frame["data"] == b"jpeg-frame" for frame in result["frames"])
    assert result["warning"] is None
    assert len(calls) == 2
    assert all(call[1]["timeout"] <= 3.0 for call in calls)
    assert all("image2pipe" in call[0] for call in calls)
    assert not list(tmp_path.glob("*.jpg"))


def test_sampling_returns_a_warning_for_timeout_or_oversized_frames(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    monkeypatch.setattr(
        "autoclipper.visual_context.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ffmpeg", 1)),
    )
    result = sample_video_frames(str(video), 200, ffmpeg_path="/fake/ffmpeg")
    assert result["frames"] == []
    assert "transcript evidence only" in result["warning"]

    monkeypatch.setattr(
        "autoclipper.visual_context.subprocess.run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=b"x" * 20, stderr=b""
        ),
    )
    result = extract_video_frames(
        str(video), [10], max_frame_bytes=10, ffmpeg_path="/fake/ffmpeg"
    )
    assert result["frames"] == []
    assert "transcript evidence only" in result["warning"]


def _fake_google(monkeypatch):
    part_calls = []

    class FakePart:
        @staticmethod
        def from_bytes(data, mime_type):
            part_calls.append((data, mime_type))
            return {"image_part": len(data)}

    fake_genai = types.SimpleNamespace(
        Client=Mock(return_value=object()),
        types=types.SimpleNamespace(Part=FakePart),
    )
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    return part_calls


def _report_response():
    return types.SimpleNamespace(
        text=json.dumps({
            "summary": "A person demonstrates a repair.",
            "chapters": [{"start": 0, "end": 20, "label": "Repair"}],
            "key_topics": ["repair"],
            "suggested_angles": [],
        })
    )


def test_whole_video_analysis_sends_timestamped_visual_parts(monkeypatch):
    import autoclipper.analyze as analyze_mod

    part_calls = _fake_google(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        analyze_mod.config,
        "gemini_generate",
        lambda _client, _model, contents: captured.setdefault("contents", contents) or _report_response(),
    )
    # setdefault above returns the stored contents, so use a dedicated capture.
    def fake_generate(_client, _model, contents):
        captured["contents"] = contents
        return _report_response()
    monkeypatch.setattr(analyze_mod.config, "gemini_generate", fake_generate)

    result = analyze_mod.analyze_video_summary_and_angles(
        {
            "duration": 20,
            "language": "English",
            "segments": [{"start": 0, "end": 20, "text": "The repair is complete."}],
        },
        api_key="test-key",
        visual_samples=[{"timestamp": 5.25, "data": b"jpeg", "mime_type": "image/jpeg"}],
    )

    assert part_calls == [(b"jpeg", "image/jpeg")]
    contents = captured["contents"]
    assert any("5.250s (00:00:05)" in item for item in contents if isinstance(item, str))
    assert any(isinstance(item, dict) and item.get("image_part") == 4 for item in contents)
    assert result["visual_context"]["status"] == "used"
    assert result["visual_context"]["frame_count"] == 1
    assert result["analysis_warnings"] == []


def test_visual_gemini_failure_falls_back_to_transcript_only_with_visible_warning(monkeypatch):
    import autoclipper.analyze as analyze_mod

    _fake_google(monkeypatch)
    calls = []

    def fake_generate(_client, _model, contents):
        calls.append(contents)
        if len(calls) == 1:
            raise RuntimeError("provider rejected inline image")
        return _report_response()

    monkeypatch.setattr(analyze_mod.config, "gemini_generate", fake_generate)
    result = analyze_mod.analyze_video_summary_and_angles(
        {
            "duration": 20,
            "language": "English",
            "segments": [{"start": 0, "end": 20, "text": "The repair is complete."}],
        },
        api_key="test-key",
        visual_samples=[{"timestamp": 5, "data": b"jpeg", "mime_type": "image/jpeg"}],
    )

    assert len(calls) == 2
    assert isinstance(calls[0], list)
    assert isinstance(calls[1], str)
    assert result["visual_context"]["status"] == "transcript_only"
    assert "transcript evidence only" in result["analysis_warnings"][0]


def test_sampling_failure_warning_is_attached_even_when_analysis_succeeds(monkeypatch):
    import autoclipper.analyze as analyze_mod

    _fake_google(monkeypatch)
    monkeypatch.setattr(analyze_mod.config, "gemini_generate", lambda *_args: _report_response())
    result = analyze_mod.analyze_video_summary_and_angles(
        {
            "duration": 20,
            "language": "English",
            "segments": [{"start": 0, "end": 20, "text": "The repair is complete."}],
        },
        api_key="test-key",
        visual_sampling_warning="Visual frame sampling failed; analysis will use transcript evidence only.",
    )
    assert result["visual_context"]["status"] == "transcript_only"
    assert result["analysis_warnings"]
