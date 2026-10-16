"""Real-media tests for the silent / speech-less analysis paths (Objective 4).

Unlike the rest of the suite, these tests do **not** mock the media boundary:
they synthesise a real MP4 with ffmpeg and drive the real pipeline, so clip
rendering, scene detection, QA and the audio probe all run for real. Only the
paid model calls are stubbed, because no API key exists in CI.

They skip cleanly when ffmpeg is not installed. They do not prove the *semantic*
quality of visual analysis — that needs a real speech-less video and a live
Gemini key; see OBJECTIVES_REPORT.md.
"""

import json
import os
import shutil
import subprocess

import pytest

from autoclipper import pipeline

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required for real-media tests"
)

VIDEO_SECONDS = 40


def _make_silent_video(path, *, with_silent_audio: bool) -> bool:
    """Render a real test video: no audio stream, or a digital-silence track."""
    args = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i",
        f"testsrc2=size=640x360:rate=25:duration={VIDEO_SECONDS}",
    ]
    if with_silent_audio:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
                 "-shortest", "-c:a", "aac"]
    args += ["-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)]
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _run_pipeline(source, output_dir, monkeypatch, **kwargs):
    """Run the real pipeline; fail loudly if transcription is attempted."""
    calls = {"transcribe": 0}

    def _guard(*args, **kw):
        calls["transcribe"] += 1
        raise AssertionError("transcription must not run for a speech-less source")

    monkeypatch.setattr(pipeline, "transcribe", _guard)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = pipeline.run(source=source, output_dir=output_dir, vertical=False, **kwargs)
    return result, calls


def test_real_silent_video_skips_transcription_and_renders_clips(tmp_path, monkeypatch):
    """A video with no audio stream must never be sent to transcription."""
    source = tmp_path / "no_audio.mp4"
    if not _make_silent_video(source, with_silent_audio=False):
        pytest.skip("could not synthesise a test video with this ffmpeg build")

    result, calls = _run_pipeline(
        str(source), str(tmp_path / "job"), monkeypatch,
        clip_count=1, target_duration=20, gemini_key=None,
    )

    assert calls["transcribe"] == 0
    profile = result["source_profile"]
    assert profile["analysis_mode"] == "visual_only"
    assert profile["has_audio"] is False
    assert profile["has_speech"] is False
    assert profile["uses_visual_analysis"] is True
    assert "no audio stream" in " ".join(profile["reasons"]).lower()

    # The visual path ran and the job produced a real, playable file.
    assert result["candidate_review"]["status"] == "visual_analysis"
    assert result["individual_clip_count"] >= 1
    clip = result["clips"][0]
    assert os.path.getsize(os.path.join(str(tmp_path / "job"), clip["file"])) > 0


def test_real_silent_audio_track_is_measured_and_not_treated_as_speech(tmp_path, monkeypatch):
    """Audio present but silent: measured, not transcribed, still rendered."""
    source = tmp_path / "silent_audio.mp4"
    if not _make_silent_video(source, with_silent_audio=True):
        pytest.skip("could not synthesise a test video with this ffmpeg build")

    result, calls = _run_pipeline(
        str(source), str(tmp_path / "job"), monkeypatch,
        clip_count=1, target_duration=20, gemini_key=None,
    )

    assert calls["transcribe"] == 0, "a silent track was sent to transcription"
    profile = result["source_profile"]
    assert profile["has_audio"] is True, "the audio stream was not detected"
    assert profile["has_speech"] is False
    assert profile["analysis_mode"] != "speech"
    # A silent-but-present track is still the source's audio and is preserved.
    assert profile["preserve_audio"] is True
    assert result["individual_clip_count"] >= 1


def test_real_visual_only_job_reports_a_shortfall_honestly(tmp_path, monkeypatch):
    """With no semantic review available the job must not claim a full delivery."""
    source = tmp_path / "no_audio.mp4"
    if not _make_silent_video(source, with_silent_audio=False):
        pytest.skip("could not synthesise a test video with this ffmpeg build")

    result, _ = _run_pipeline(
        str(source), str(tmp_path / "job"), monkeypatch,
        clip_count=3, target_duration=30, gemini_key=None,
    )

    compliance = result["request_compliance"]
    assert compliance["requested_individual_clips"] == 3
    assert compliance["delivered_individual_clips"] == result["individual_clip_count"]
    if compliance["delivered_individual_clips"] < 3:
        assert compliance["satisfied"] is False
        assert compliance["count_satisfied"] is False
        assert compliance["failures"]
        # The honest degradation is surfaced, not hidden.
        warnings = " ".join(result["processing_warnings"]).lower()
        assert "visual" in warnings or "scene" in warnings or "individual clip" in warnings
    # Metadata must be serialisable for the API response.
    json.dumps(result["source_profile"])
    json.dumps(result["request_compliance"])
