import shutil
import subprocess
import sys
import types
from unittest.mock import Mock, patch

import pytest

from autoclipper.analyze import (
    _build_transcript_text, _filter_clips_to_bounds, _parse_clips,
    _resolve_clip_bounds, _validate_clips, _validate_report,
    analyze_video_summary_and_angles, find_viral_clips,
)
from autoclipper.subtitles import _ass_time, build_srt, hex_to_ass


def test_transcript_lines_keep_timestamps_and_skip_empty_segments():
    transcript = {
        "segments": [
            {"start": 0.25, "end": 1.5, "text": " Hello world "},
            {"start": 1.5, "end": 2.0, "text": "  "},
        ]
    }
    assert _build_transcript_text(transcript) == "[0.2 - 1.5] Hello world"


def test_parse_clips_accepts_fenced_json():
    assert _parse_clips('before\n```json\n{"clips":[{"start":1,"end":8}]}\n```') == [
        {"start": 1, "end": 8}
    ]


def test_parse_clips_repairs_missing_commas_between_objects():
    assert _parse_clips('{"clips":[{"start":1,"end":8} {"start":9,"end":16}]}') == [
        {"start": 1, "end": 8},
        {"start": 9, "end": 16},
    ]


def test_target_duration_overrides_stale_legacy_range():
    assert _resolve_clip_bounds(15, 60, 100) == (97.0, 103.0)
    assert _resolve_clip_bounds(15, 60, 5) == (5.0, 8.0)
    assert _resolve_clip_bounds(15, 60, 120) == (117.0, 120.0)


def test_duration_filter_accepts_100_second_target_despite_old_60_second_max():
    candidates = [
        {"start": 0, "end": 100, "title": "target"},
        {"start": 0, "end": 90, "title": "too short"},
        {"start": 0, "end": 110, "title": "too long"},
    ]
    assert _filter_clips_to_bounds(candidates, 97, 103) == [candidates[0]]


def test_legacy_duration_filter_keeps_legacy_tolerance():
    candidates = [
        {"start": 0, "end": 13.5},
        {"start": 0, "end": 62.5},
        {"start": 0, "end": 63.1},
    ]
    assert _filter_clips_to_bounds(
        candidates, 15, 60, min_tolerance=2, max_tolerance=3
    ) == candidates[:2]


@pytest.mark.parametrize("target", [0, 4.99, 120.01, -5])
def test_target_duration_rejects_values_outside_supported_range(target):
    with pytest.raises(ValueError, match="between 5 and 120"):
        _resolve_clip_bounds(15, 60, target)


def test_validate_clips_clamps_end_to_transcript_duration():
    clips = _validate_clips(
        [{"start": 10, "end": 22, "title": "Clip"}],
        total_duration=20,
        min_clip=5,
        max_clip=15,
    )
    assert len(clips) == 1
    assert clips[0]["end"] == 20


def test_validate_clips_rejects_clip_whose_start_is_past_duration_after_clamp():
    clips = _validate_clips(
        [{"start": 21, "end": 25, "title": "Impossible"}],
        total_duration=20,
        min_clip=1,
        max_clip=120,
    )
    assert clips == []


def test_validate_clips_rejects_invalid_and_out_of_range_candidates():
    clips = _validate_clips(
        [
            {"start": 10, "end": 10},
            {"start": "bad", "end": 20},
            {"start": 1, "end": 2},
            {"start": 1, "end": 40},
        ],
        total_duration=100,
        min_clip=5,
        max_clip=20,
    )
    assert clips == []


def test_find_viral_clips_passes_target_and_custom_direction_to_model():
    fake_genai = types.SimpleNamespace(Client=Mock(return_value=object()))
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    response = types.SimpleNamespace(
        text='{"clips":[{"start":10,"end":50,"title":"Test"}]}'
    )
    with patch.dict(sys.modules, {"google": fake_google}), patch(
        "autoclipper.analyze.config.gemini_generate", return_value=response
    ) as generate:
        clips = find_viral_clips(
            {
                "duration": 100,
                "language": "English",
                "segments": [{"start": 10, "end": 50, "text": "A complete thought."}],
            },
            api_key="test-key",
            clip_count=1,
            min_clip=15,
            max_clip=60,
            target_duration=40,
            custom_prompt="Focus on the conclusion",
        )
    assert len(clips) == 1
    prompt = generate.call_args.kwargs["contents"] if "contents" in generate.call_args.kwargs else generate.call_args.args[2]
    assert "40 seconds" in prompt
    assert "Focus on the conclusion" in prompt


def test_build_srt_clips_word_times_to_requested_window():
    srt = build_srt(
        [
            {"word": "before", "start": 0.0, "end": 1.0},
            {"word": "inside", "start": 2.0, "end": 4.0},
            {"word": "after", "start": 4.8, "end": 5.5},
        ],
        start=2.5,
        end=5.0,
    )
    assert "inside" in srt
    assert "after" in srt
    assert "00:00:00,000 --> 00:00:01,500" in srt
    # A source word crosses the output boundary at 2.5s; its cue must be clipped.
    assert "00:00:02,300 --> 00:00:02,500" in srt
    assert "00:00:02,500 --> 00:00:03,000" not in srt


def test_build_srt_returns_empty_when_no_words_overlap():
    assert build_srt([{"word": "outside", "start": 0, "end": 1}], 2, 5) == ""


def test_ffmpeg_can_burn_captions_into_a_synthetic_clip(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is not installed in this environment")

    source = tmp_path / "synthetic.mp4"
    srt_path = tmp_path / "captions.srt"
    output = tmp_path / "captioned.mp4"
    subprocess.run(
        [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=2",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(source),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    srt_path.write_text(
        "1\n00:00:00,200 --> 00:00:01,500\nAutomated caption validation\n",
        encoding="utf-8",
    )
    from autoclipper.subtitles import burn_subtitles

    burn_subtitles(str(source), str(srt_path), str(output))
    assert output.is_file()
    assert output.stat().st_size > 0
    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe", "-v", "error",
            "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(output),
        ],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert 1.8 <= float(probe.stdout.strip()) <= 2.2


def test_add_subtitles_reports_missing_transcript_words(tmp_path):
    from autoclipper.subtitles import add_subtitles

    clips = [{"index": 1, "file": "clip.mp4", "start": 10, "end": 20}]
    result = add_subtitles(clips, {"words": []}, str(tmp_path), burn=True)
    assert result[0]["subtitle_warning"]
    assert "captions were not generated" in result[0]["subtitle_warning"]


def test_ass_timestamp_carries_milliseconds_into_next_second():
    assert _ass_time(1.9999) == "00:00:02.00"


def test_hex_to_ass_converts_rgb_to_bgr():
    assert hex_to_ass("#12ABEF") == "&H00EFAB12&"


def test_analysis_report_sanitizes_chapters_and_angle_hints():
    report = _validate_report(
        {
            "summary": "A concise summary.",
            "chapters": [
                {"start": 0, "end": 20, "label": "Intro"},
                {"start": 18, "end": 45, "label": "Main"},
                {"start": 60, "end": 70, "label": "Out of bounds"},
            ],
            "key_topics": ["specific technique", "", 42],
            "suggested_angles": [
                {
                    "title": "Key moment", "target_duration": 600,
                    "start_hint": 10, "end_hint": 200, "custom_prompt": "Focus on payoff",
                },
                {"title": "", "start_hint": 1, "end_hint": 5},
            ],
        },
        duration=50,
    )
    assert report["summary"] == "A concise summary."
    assert len(report["chapters"]) == 2
    assert report["chapters"][-1]["end"] == 45
    assert report["key_topics"] == ["specific technique"]
    assert len(report["suggested_angles"]) == 1
    assert report["suggested_angles"][0]["target_duration"] == 120
    assert report["suggested_angles"][0]["end_hint"] is None


def test_whole_video_analysis_keeps_summary_chapters_topics_and_angles():
    import json

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=Mock(return_value=object()))
    response = types.SimpleNamespace(
        text=json.dumps({
            "summary": "The video teaches one specific technique.",
            "chapters": [{"start": 0, "end": 20, "label": "Setup"}],
            "key_topics": ["specific grip"],
            "suggested_angles": [{
                "title": "The technique", "hook": "Do this first",
                "description": "A focused demonstration.",
                "target_duration": 35, "start_hint": 3, "end_hint": 18,
                "custom_prompt": "Keep the result at the end.",
            }],
        })
    )
    transcript = {
        "duration": 20,
        "language": "English",
        "segments": [{"start": 0, "end": 20, "text": "This is the full transcript."}],
    }
    with patch.dict(sys.modules, {"google": fake_google}), patch(
        "autoclipper.analyze.config.gemini_generate", return_value=response
    ):
        report = analyze_video_summary_and_angles(
            transcript, api_key="test-key", video_info={"duration": 20, "width": 1920, "height": 1080}
        )
    assert report["summary"] == "The video teaches one specific technique."
    assert report["chapters"][0]["label"] == "Setup"
    assert report["key_topics"] == ["specific grip"]
    assert report["suggested_angles"][0]["title"] == "The technique"
    assert report["suggested_angles"][0]["end_hint"] == 18.0
