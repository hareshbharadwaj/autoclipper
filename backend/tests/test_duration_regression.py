"""Regression tests for target duration enforcement, QA reporting, subtitles, and reframing."""
import os
import shutil
import subprocess

import pytest

from autoclipper.analyze import extend_candidate_to_target_duration
from autoclipper.quality_assurance import verify_rendered_file
from autoclipper.reframe import (
    VERTICAL_ASPECT, _detect_scenes_opencv, plan_aspect_dims, SpeakerTracker,
)
from autoclipper.subtitles import subtitle_layout, _wrap_caption_text, build_srt


FFMPEG = shutil.which("ffmpeg")


def test_extend_candidate_reaches_60s_target_when_context_is_sufficient():
    """Reproduces the 7.2s candidate extended along contiguous transcript to ~60s."""
    candidate = {
        "start": 90.6,
        "end": 97.8,
        "title": "Never Put Knives in the Sink!",
        "hook": "Never put sharp knives in soapy water!",
        "score": 95,
    }
    # Segments representing a kitchen knife safety explanation
    segments = [
        {"start": 80.0, "end": 85.0, "text": "Before you begin cooking, let's talk about safety."},
        {"start": 85.5, "end": 90.0, "text": "Here is the number one rule in professional kitchens."},
        {"start": 90.6, "end": 97.8, "text": "Never put knives in the sink. Always set your knives on the counter."},
        {"start": 98.2, "end": 105.0, "text": "When you put knives into soapy water, nobody can see the blade."},
        {"start": 105.5, "end": 115.0, "text": "Someone will reach in to grab a sponge and get a severe laceration."},
        {"start": 115.4, "end": 125.0, "text": "Instead, wash the knife immediately under running water with a sponge moving away from the blade."},
        {"start": 125.5, "end": 135.0, "text": "Dry it immediately with a clean kitchen towel pointing the tip down."},
        {"start": 135.5, "end": 145.0, "text": "Then place it directly back into your knife block or onto the magnetic strip."},
        {"start": 145.5, "end": 150.8, "text": "Follow this simple habit and you will prevent emergency room visits."},
        {"start": 151.2, "end": 160.0, "text": "Now let us move on to proper cutting board preparation."},
    ]
    transcript = {"segments": segments, "duration": 200.0}

    extended = extend_candidate_to_target_duration(
        candidate,
        transcript,
        target_duration=60.0,
        total_duration=200.0,
    )

    duration = extended["end"] - extended["start"]
    # Should be close to 60s (e.g. 55-65s) along sentence boundaries
    assert 52.0 <= duration <= 68.0
    # Must preserve the opening hook setup
    assert extended["start"] <= 90.6
    # Must cover the explanation and payoff
    assert extended["end"] >= 145.0
    assert extended["duration_target_met"] is True
    assert extended.get("duration_warning") is None
    assert extended["title"] == "Never Put Knives in the Sink!"


def test_extend_candidate_handles_insufficient_context_with_clear_warning():
    """When source runs out of contiguous speech, returns max coherent span with warning."""
    candidate = {
        "start": 90.6,
        "end": 97.8,
        "title": "Short Tip",
        "score": 90,
    }
    # Only 12 seconds of dialogue followed by silence or topic shift
    segments = [
        {"start": 90.6, "end": 97.8, "text": "Never put knives in the sink."},
        {"start": 98.0, "end": 102.5, "text": "Always wash them immediately."},
        # 10 second speech gap indicating scene change or silence
        {"start": 112.5, "end": 120.0, "text": "Completely unrelated next topic about oven cleaning."},
    ]
    transcript = {"segments": segments, "duration": 150.0}

    extended = extend_candidate_to_target_duration(
        candidate,
        transcript,
        target_duration=60.0,
        total_duration=150.0,
    )

    duration = extended["end"] - extended["start"]
    # Coherent contiguous span should stop before the 10s gap
    assert duration <= 15.0
    assert extended["duration_target_met"] is False
    assert "Requested duration was 60s" in extended["duration_warning"]
    assert "coherent" in extended["duration_warning"]


@pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")
def test_qa_fails_when_actual_rendered_clip_is_far_shorter_than_requested(tmp_path):
    """A valid MP4 of 7.2s must NOT pass QA when the user requested 60s."""
    clip_path = tmp_path / "short_clip.mp4"
    # Create a 7.2 second video
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi", "-i",
            "color=c=black:s=320x180:r=25:d=7.2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=7.2",
            "-t", "7.2", "-c:v", "libx264", "-c:a", "aac",
            str(clip_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    report = verify_rendered_file(
        str(clip_path),
        expected_duration=7.2,
        target_duration=60.0,
        output_dir=str(tmp_path),
    )

    # Must fail QA because 7.2s violates the requested 60s target
    assert report["status"] == "fail"
    dur_check = report["checks"]["duration"]
    assert dur_check["ok"] is False
    assert dur_check["target_duration_passed"] is False
    assert dur_check["target_seconds"] == 60.0
    assert any("Requested duration of 60.0s was not respected" in err for err in report["errors"])


@pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")
def test_qa_passes_when_rendered_clip_meets_requested_target_duration(tmp_path):
    """A clip close to the requested target passes with all duration metrics recorded."""
    clip_path = tmp_path / "target_clip.mp4"
    # Create a 5.0 second video with target 5.0s
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi", "-i",
            "color=c=black:s=320x180:r=25:d=5.0",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5.0",
            "-t", "5.0", "-c:v", "libx264", "-c:a", "aac",
            str(clip_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    report = verify_rendered_file(
        str(clip_path),
        expected_duration=5.0,
        target_duration=5.0,
        output_dir=str(tmp_path),
    )

    assert report["status"] == "pass"
    dur_check = report["checks"]["duration"]
    assert dur_check["ok"] is True
    assert dur_check["target_duration_passed"] is True
    assert dur_check["target_seconds"] == 5.0
    assert abs(dur_check["actual_seconds"] - 5.0) <= 0.5


def test_portrait_subtitle_layout_avoids_dominating_screen():
    """Portrait subtitles scale relative to aspect ratio and enforce safe margins."""
    # Standard 1080x1920 vertical video
    portrait = subtitle_layout(1080, 1920)
    assert 40 <= portrait["font_size"] <= 48
    # Bottom margin clears platform UI (14% - 22% of height)
    assert portrait["margin_v"] >= 268
    # Side margin ensures captions do not touch edges
    assert portrait["margin_l"] >= 80

    # Low-resolution 540x960 portrait
    low_res = subtitle_layout(540, 960)
    assert 20 <= low_res["font_size"] <= 26
    assert low_res["margin_v"] >= 134

    # Landscape 1920x1080
    landscape = subtitle_layout(1920, 1080)
    assert landscape["font_size"] == 80
    assert landscape["margin_v"] <= 162

    # Square 1080x1080
    square = subtitle_layout(1080, 1080)
    assert 60 <= square["font_size"] <= 72


def test_wrap_caption_balances_lines_and_breaks_long_words():
    """Balanced line wrapping distributes text evenly without orphan words."""
    text = "Never put sharp kitchen knives inside a soapy sink"
    wrapped = _wrap_caption_text(text, max_chars=25)
    lines = wrapped.split("\n")
    assert len(lines) >= 2
    # Check that lines are balanced (no single orphan word on line 2)
    assert all(len(line) > 5 for line in lines)

    # Word exceeding max_chars is broken safely
    long_word_wrapped = _wrap_caption_text("antidisestablishmentarianism", max_chars=12)
    long_lines = long_word_wrapped.split("\n")
    assert all(len(l) <= 12 for l in long_lines)


def test_speaker_tracker_hysteresis_avoids_rapid_switching():
    """SpeakerTracker maintains active speaker without frame-by-frame ping-pong."""
    tracker = SpeakerTracker(cooldown_frames=30)
    # Frame 1: Speaker A on left
    faces_a = [{"box": [100, 200, 80, 80], "score": 6400}]
    box1 = tracker.get_target(faces_a, frame_number=1, width=1920)
    assert box1 == [100, 200, 80, 80]

    # Frame 5: Speaker A and Speaker B both present; Speaker B slightly higher raw score
    faces_ab = [
        {"box": [100, 200, 80, 80], "score": 6000},
        {"box": [1500, 200, 80, 80], "score": 6500},
    ]
    # Cooldown and active speaker bias should maintain Speaker A
    box2 = tracker.get_target(faces_ab, frame_number=5, width=1920)
    assert box2 == [100, 200, 80, 80]


def test_aspect_dims_planning():
    """Aspect planning preserves resolution across aspect presets."""
    # 1920x1080 landscape to 9:16 portrait
    out_w, out_h, crop_w, crop_h = plan_aspect_dims(1920, 1080, aspect=9 / 16)
    assert out_w == 1080
    assert out_h == 1920
    assert crop_h == 1080
    assert crop_w == 606 or crop_w == 608

    # 1920x1080 landscape to 1:1 square
    out_w_sq, out_h_sq, crop_w_sq, crop_h_sq = plan_aspect_dims(1920, 1080, aspect=1.0)
    assert out_w_sq == 1080
    assert out_h_sq == 1080
    assert crop_w_sq == 1080
    assert crop_h_sq == 1080
