import shutil
import subprocess

import pytest

from autoclipper.audio_mix import mix_audio_layers
from autoclipper.montage import stitch_montage
from autoclipper.pipeline import (
    _fit_montage_candidates, _load_cached_transcript, _normalize_target_duration,
)
from autoclipper.reframe import plan_aspect_dims
from autoclipper.utils import classify_aspect, probe_duration, probe_video_info


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")


def _run(args):
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_video(tmp_path, name, color, seconds=2, audio=True, size="320x240"):
    path = tmp_path / name
    args = [
        FFMPEG, "-y", "-f", "lavfi", "-i",
        f"color=c={color}:s={size}:r=25:d={seconds}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    _run(args)
    return path


def _make_tone(tmp_path, name, frequency=880, seconds=2):
    path = tmp_path / name
    _run([
        FFMPEG, "-y", "-f", "lavfi", "-i",
        f"sine=frequency={frequency}:duration={seconds}", str(path),
    ])
    return path


def test_probe_video_info_reads_duration_dimensions_and_audio(tmp_path):
    video = _make_video(tmp_path, "probe.mp4", "black", seconds=2)
    info = probe_video_info(str(video))
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["has_audio"] is True
    assert 1.8 <= info["duration"] <= 2.2
    assert info["aspect"]["recommended"] in {"9:16", "1:1", "16:9"}


def test_aspect_classifier_handles_portrait_square_and_landscape():
    assert classify_aspect(1080, 1920) == "9:16"
    assert classify_aspect(1080, 1080) == "1:1"
    assert classify_aspect(1920, 1080) == "16:9"


def test_montage_stitches_two_clips_with_crossfaded_audio(tmp_path):
    first = _make_video(tmp_path, "first.mp4", "black", seconds=2)
    second = _make_video(tmp_path, "second.mp4", "blue", seconds=2)
    output = tmp_path / "montage.mp4"
    stitch_montage([str(first), str(second)], str(output), crossfade_seconds=0.25)
    assert output.is_file() and output.stat().st_size > 0
    info = probe_video_info(str(output))
    assert info["has_audio"] is True
    assert info["width"] == 320 and info["height"] == 240
    assert 3.4 <= probe_duration(str(output)) <= 4.1


def test_montage_requires_two_or_three_segments(tmp_path):
    with pytest.raises(ValueError, match="2 or 3"):
        stitch_montage([str(tmp_path / "missing.mp4")], str(tmp_path / "out.mp4"))


def test_voiceover_and_music_mix_preserves_video_and_audio(tmp_path):
    video = _make_video(tmp_path, "base.mp4", "black", seconds=2)
    voice = _make_tone(tmp_path, "voice.wav", frequency=700, seconds=1.5)
    music = _make_tone(tmp_path, "music.wav", frequency=180, seconds=3)
    output = tmp_path / "mixed.mp4"
    mix_audio_layers(
        str(video), str(output), voiceover_path=str(voice),
        music_path=str(music), music_volume=0.18,
    )
    assert output.is_file() and output.stat().st_size > 0
    info = probe_video_info(str(output))
    assert info["has_audio"] is True
    assert info["width"] == 320 and info["height"] == 240
    assert 1.8 <= probe_duration(str(output)) <= 2.2


def test_music_only_mix_preserves_clip_duration(tmp_path):
    video = _make_video(tmp_path, "silent.mp4", "black", seconds=2, audio=False)
    music = _make_tone(tmp_path, "music.wav", frequency=180, seconds=3)
    output = tmp_path / "music_only.mp4"
    mix_audio_layers(str(video), str(output), music_path=str(music), music_volume=0.2)
    info = probe_video_info(str(output))
    assert info["has_audio"] is True
    assert 1.8 <= probe_duration(str(output)) <= 2.2


def test_montage_candidate_picker_uses_transcript_boundaries_and_avoids_overlap():
    transcript = {
        "segments": [
            {"start": float(t), "end": float(t + 2), "text": f"sentence {t}"}
            for t in range(0, 60, 2)
        ]
    }
    angles = [
        {"title": "First", "start_hint": 0, "end_hint": 12},
        {"title": "Second", "start_hint": 20, "end_hint": 32},
        {"title": "Third", "start_hint": 40, "end_hint": 52},
    ]
    clips = _fit_montage_candidates(angles, transcript, target_duration=36, total_duration=60)
    assert len(clips) == 3
    assert clips[0]["end"] <= clips[1]["start"]
    assert clips[1]["end"] <= clips[2]["start"]
    assert all(clip["start"] in {float(t) for t in range(0, 60, 2)} for clip in clips)
    assert all(8 <= clip["end"] - clip["start"] <= 16 for clip in clips)


def test_aspect_dimensions_match_requested_publish_formats():
    assert plan_aspect_dims(1920, 1080, 9 / 16)[:2] == (1080, 1920)
    assert plan_aspect_dims(1920, 1080, 1.0)[:2] == (1080, 1080)
    assert plan_aspect_dims(1920, 1080, 16 / 9)[:2] == (1920, 1080)


def test_target_duration_is_hard_capped_at_120_seconds(monkeypatch):
    from autoclipper import config

    # A deployment environment override must not weaken the product ceiling.
    monkeypatch.setattr(config, "HARD_MAX_CLIP_SECONDS", 600.0)
    assert _normalize_target_duration(600) == 120.0
    assert _normalize_target_duration(120) == 120.0
    assert _normalize_target_duration(None) is None
    with pytest.raises(ValueError, match="number between 5 and 120"):
        _normalize_target_duration("not-a-number")


def test_reframe_renders_portrait_and_square_outputs_with_audio(tmp_path):
    from autoclipper.reframe import reframe_clip

    source = _make_video(tmp_path, "wide_source.mp4", "black", seconds=1, size="320x180")
    for ratio, expected in [(9 / 16, (180, 320)), (1.0, (180, 180))]:
        output = tmp_path / f"reframed_{str(ratio).replace('.', '_')}.mp4"
        assert reframe_clip(str(source), str(output), use_yolo=False, aspect=ratio)
        info = probe_video_info(str(output))
        assert (info["width"], info["height"]) == expected
        assert info["has_audio"] is True


def test_cached_transcript_loader_reuses_valid_json_without_api_calls(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        '{"duration": 12, "segments": [{"start": 0, "end": 12, "text": "Cached"}]}',
        encoding="utf-8",
    )
    transcript = _load_cached_transcript(str(transcript_path))
    assert transcript["duration"] == 12
    assert transcript["segments"][0]["text"] == "Cached"


def test_cached_transcript_loader_rejects_missing_or_malformed_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="cache is missing"):
        _load_cached_transcript(str(tmp_path / "missing.json"))
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"words": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _load_cached_transcript(str(malformed))
