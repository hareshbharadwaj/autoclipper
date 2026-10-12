import importlib
import json
import subprocess
import sys
import types
from unittest.mock import Mock

import pytest

import autoclipper.visual_context as visual_context_mod
from autoclipper.visual_context import (
    extract_video_frames,
    sample_candidate_frames,
    sample_timestamps,
    sample_video_frames,
)


def test_sample_timestamps_are_evenly_distributed_and_strictly_bounded():
    times = sample_timestamps(900, max_frames=5)
    assert len(times) == 5
    assert times == pytest.approx([150, 300, 450, 600, 750])
    assert all(0 < time < 900 for time in times)
    # Short videos are lifted to the minimum sample count instead of one frame.
    short = sample_timestamps(40, max_frames=50)
    assert len(short) == 3
    assert short == pytest.approx([10, 20, 30])
    assert sample_timestamps(float("nan")) == []
    assert sample_timestamps(20, max_frames=0) == []


def test_short_videos_receive_minimum_visual_coverage():
    # A 90s video used to get a single midpoint frame; it now gets three.
    times = sample_timestamps(90)
    assert len(times) == 3
    assert times == pytest.approx([22.5, 45.0, 67.5])

    ten_seconds = sample_timestamps(10)
    assert len(ten_seconds) == 3
    assert ten_seconds == pytest.approx([2.5, 5.0, 7.5])

    # Very short videos collapse so frames stay ~1s apart.
    tiny = sample_timestamps(0.5)
    assert len(tiny) == 1
    assert tiny == pytest.approx([0.25])
    two_seconds = sample_timestamps(2.0)
    assert len(two_seconds) == 2


def test_medium_and_long_videos_scale_with_the_sampling_interval():
    medium = sample_timestamps(300)  # ceil(300/90) = 4
    assert len(medium) == 4
    assert medium == pytest.approx([60, 120, 180, 240])

    long_video = sample_timestamps(3600)  # capped at MAX_ANALYSIS_FRAMES (8)
    assert len(long_video) == visual_context_mod.MAX_ANALYSIS_FRAMES
    assert all(0 < time < 3600 for time in long_video)

    denser = sample_timestamps(300, interval=60)  # ceil(300/60) = 5
    assert len(denser) == 5
    sparse_floor = sample_timestamps(40, min_frames=1)
    assert len(sparse_floor) == 1


def test_sampling_limits_are_env_configurable_but_hard_bounded(monkeypatch):
    try:
        monkeypatch.setenv("AUTOCLIPPER_MAX_ANALYSIS_FRAMES", "100")  # over the cap
        monkeypatch.setenv("AUTOCLIPPER_MIN_ANALYSIS_FRAMES", "6")
        monkeypatch.setenv("AUTOCLIPPER_VISUAL_SAMPLE_INTERVAL_SECONDS", "30")
        monkeypatch.setenv("AUTOCLIPPER_MAX_FRAME_BYTES", "999999999")  # over the cap
        importlib.reload(visual_context_mod)

        assert visual_context_mod.MAX_ANALYSIS_FRAMES == 16  # hard upper bound
        assert visual_context_mod.MIN_ANALYSIS_FRAMES == 6
        assert visual_context_mod.VISUAL_SAMPLE_INTERVAL_SECONDS == 30.0
        assert visual_context_mod.MAX_FRAME_BYTES == 2_000_000  # hard upper bound
        assert visual_context_mod.MAX_TOTAL_FRAME_BYTES >= visual_context_mod.MAX_FRAME_BYTES
        assert len(visual_context_mod.sample_timestamps(600)) == 16
    finally:
        monkeypatch.delenv("AUTOCLIPPER_MAX_ANALYSIS_FRAMES", raising=False)
        monkeypatch.delenv("AUTOCLIPPER_MIN_ANALYSIS_FRAMES", raising=False)
        monkeypatch.delenv("AUTOCLIPPER_VISUAL_SAMPLE_INTERVAL_SECONDS", raising=False)
        monkeypatch.delenv("AUTOCLIPPER_MAX_FRAME_BYTES", raising=False)
        importlib.reload(visual_context_mod)
    assert visual_context_mod.MAX_ANALYSIS_FRAMES == 8
    assert visual_context_mod.MIN_ANALYSIS_FRAMES == 3


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


def _stub_frame_extraction(monkeypatch, calls=None):
    def fake_run(command, **kwargs):
        if calls is not None:
            calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout=b"jpeg-frame", stderr=b"")

    monkeypatch.setattr("autoclipper.visual_context.subprocess.run", fake_run)


def test_short_whole_video_sampling_extracts_minimum_frames(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    calls = []
    _stub_frame_extraction(monkeypatch, calls)

    result = sample_video_frames(str(video), 90, ffmpeg_path="/fake/ffmpeg")

    # A 90-second video previously yielded a single midpoint frame.
    assert [frame["timestamp"] for frame in result["frames"]] == [22.5, 45.0, 67.5]
    assert len(calls) == 3
    assert result["warning"] is None


def test_candidate_sampling_adds_context_frames_around_candidate_moments(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    _stub_frame_extraction(monkeypatch)
    candidates = [
        {"start": 10, "end": 20},
        {"start": 30, "end": 50},
        {"start": 60, "end": 64},
    ]

    result = sample_candidate_frames(str(video), candidates, ffmpeg_path="/fake/ffmpeg")

    frames = result["frames"]
    timestamps = [frame["timestamp"] for frame in frames]
    assert len(frames) <= visual_context_mod.MAX_ANALYSIS_FRAMES
    assert timestamps == sorted(timestamps)
    # Every candidate midpoint is guaranteed coverage.
    for midpoint in (15.0, 40.0, 62.0):
        assert midpoint in timestamps
    # Remaining budget adds 25%/75% context points around the same moments.
    assert 12.5 in timestamps and 17.5 in timestamps
    owners = {frame["timestamp"]: frame["candidate_index"] for frame in frames}
    assert owners[15.0] == 0
    assert owners[12.5] == 0
    assert owners[40.0] == 1
    assert owners[62.0] == 2
    assert all(index in (0, 1, 2) for index in owners.values())


def test_candidate_sampling_respects_explicit_frame_cap(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    calls = []
    _stub_frame_extraction(monkeypatch, calls)
    candidates = [
        {"start": 10, "end": 20},
        {"start": 30, "end": 50},
        {"start": 60, "end": 64},
    ]

    result = sample_candidate_frames(
        str(video), candidates, max_frames=3, ffmpeg_path="/fake/ffmpeg"
    )

    assert len(result["frames"]) == 3
    assert len(calls) == 3
    assert [frame["timestamp"] for frame in result["frames"]] == [15.0, 40.0, 62.0]


def test_candidate_sampling_ignores_malformed_candidates(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"stub")
    _stub_frame_extraction(monkeypatch)
    candidates = [
        {"start": 10, "end": 20},
        {"start": "bad"},
        {"start": 30, "end": 25},  # end <= start
        None,
    ]

    result = sample_candidate_frames(str(video), candidates, ffmpeg_path="/fake/ffmpeg")

    timestamps = [frame["timestamp"] for frame in result["frames"]]
    assert 15.0 in timestamps
    assert all(frame["candidate_index"] == 0 for frame in result["frames"] if frame["timestamp"] == 15.0)
    assert result["warning"] is None or "partial" in result["warning"]


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
