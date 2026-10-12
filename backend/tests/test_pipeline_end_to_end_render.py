"""End-to-end render verification on a synthetic but REAL media file.

The knife-safety source video is not available in this environment, so this
test exercises every non-model stage of the pipeline on a genuine rendered
file: FFmpeg cutting, SRT generation and burned captions, post-render QA
(streams/duration/aspect/audio/frame decode), and source provenance. Only the
two model calls (candidate generation and independent review) are mocked.
"""
import json
import os
import shutil
import subprocess

import pytest

from autoclipper import pipeline
from autoclipper.utils import probe_duration, probe_video_info


FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")


def _make_source(path, seconds=40):
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi", "-i",
            f"testsrc2=s=320x240:r=25:d={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _synthetic_transcript():
    """Verbatim word timings for the rules block at 10-20s of the fixture."""
    words = []
    text = (
        "Rule one always place the knife flat on the board after use. "
        "Rule two hold the knife pointing down when walking."
    ).split()
    start = 10.0
    for word in text:
        words.append({"word": word, "start": round(start, 2), "end": round(start + 0.4, 2)})
        start += 0.45
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Welcome to the kitchen basics intro."},
        {"start": 10.0, "end": 19.0,
         "text": "Rule one always place the knife flat on the board after use."},
        {"start": 19.0, "end": 24.0,
         "text": "Rule two hold the knife pointing down when walking."},
        {"start": 24.0, "end": 40.0,
         "text": "Next week we look at sharpening stones for the chef knife."},
    ]
    return {
        "language": "english",
        "duration": 40.0,
        "text": " ".join(seg["text"] for seg in segments),
        "segments": segments,
        "words": words,
    }


def test_user_selected_clip_renders_with_burned_captions_and_honest_duration(
    tmp_path, monkeypatch
):
    source = tmp_path / "fixture_source.mp4"
    _make_source(source, seconds=40)
    transcript = _synthetic_transcript()

    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **k: (str(source), "Knife Fixture"))
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: pytest.fail(
        "A valid user selection must not trigger regeneration."
    ))
    monkeypatch.setattr(pipeline, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        source="fixture",
        output_dir=str(tmp_path / "job"),
        subtitles=True,
        aspect_ratio="16:9",
        clip_count=1,
        target_duration=60,
        custom_prompt="knife safety rules placement and walking",
        selected_clip={"title": "Knife rules", "start": 10.0, "end": 24.0},
    )

    assert len(result["clips"]) == 1
    clip = result["clips"][0]

    # Honest duration: the real render is ~14s, not padded to the 60s target.
    final_path = os.path.join(str(tmp_path / "job"), clip["file"])
    assert os.path.isfile(final_path) and os.path.getsize(final_path) > 0
    actual = probe_duration(final_path)
    assert 13.0 <= actual <= 15.0
    assert clip["requested_duration"] == 60.0
    assert clip["duration_delta"] < -40
    assert clip["editorial_qa"]["duration_note"]
    assert result["quality_assurance"]["status"] in ("pass", "warning")

    # Captions are burned from the REAL transcript words only.
    srt_path = os.path.join(str(tmp_path / "job"), clip["srt"])
    assert os.path.isfile(srt_path)
    srt_text = open(srt_path, encoding="utf-8").read()
    assert "Rule one" in srt_text or "Rule" in srt_text
    assert "sharpening" not in srt_text  # nothing outside the span leaked in

    # Technical QA inspected the real rendered file.
    checks = clip["quality_assurance"]["checks"]
    assert checks["streams"]["ok"] is True
    assert checks["duration"]["ok"] is True
    assert checks["subtitles"]["ok"] is True
    assert checks["visual_samples"]["decoded_frame_count"] >= 1
    assert checks["aspect_ratio"]["ok"] is True
    info = probe_video_info(final_path)
    assert info["width"] == 320 and info["height"] == 240  # landscape preserved

    # Provenance survived all the way to the result.
    assert clip["source_intervals"] == [{"start": 10.0, "end": 24.0}]
    assert "Rule one" in clip["transcript_span"]
    assert result["source_asset"]["transcript"]["segment_count"] == 4
    assert result["validation_summary"]["technical"] in ("pass", "warning")
    assert result["candidate_review"]["status"] == "skipped_no_key"
