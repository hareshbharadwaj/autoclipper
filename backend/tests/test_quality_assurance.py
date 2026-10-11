import shutil
import subprocess

import pytest

from autoclipper.quality_assurance import verify_rendered_file


FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")


def _run(args):
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_video(tmp_path, *, audio=True, duration=2):
    path = tmp_path / ("with_audio.mp4" if audio else "silent.mp4")
    command = [
        FFMPEG, "-y", "-f", "lavfi", "-i",
        f"color=c=blue:s=320x180:r=25:d={duration}",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    command += ["-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        command += ["-c:a", "aac", "-shortest"]
    command += [str(path)]
    _run(command)
    return path


def test_qa_checks_real_file_streams_duration_aspect_and_decodable_frames(tmp_path):
    video = _make_video(tmp_path)
    report = verify_rendered_file(
        str(video),
        expected_duration=2,
        expected_aspect="16:9",
        expected_audio=True,
        output_dir=str(tmp_path),
    )

    assert report["status"] in {"pass", "warning"}
    assert report["checks"]["file"]["ok"] is True
    assert report["checks"]["streams"]["video"] is True
    assert report["checks"]["audio"]["actual"] is True
    assert report["checks"]["duration"]["ok"] is True
    assert report["checks"]["aspect_ratio"]["ok"] is True
    assert report["checks"]["visual_samples"]["decoded_frame_count"] >= 1


def test_qa_reports_missing_file_without_running_media_probes(tmp_path):
    report = verify_rendered_file(str(tmp_path / "missing.mp4"), expected_duration=2)
    assert report["status"] == "fail"
    assert report["checks"]["file"]["exists"] is False
    assert "missing" in report["errors"][0].lower()


def test_qa_flags_duration_audio_or_aspect_mismatch(tmp_path):
    video = _make_video(tmp_path, audio=False, duration=2)
    report = verify_rendered_file(
        str(video),
        expected_duration=5,
        expected_aspect="9:16",
        expected_audio=True,
        output_dir=str(tmp_path),
    )
    assert report["status"] == "fail"
    assert report["checks"]["duration"]["ok"] is False
    assert report["checks"]["aspect_ratio"]["ok"] is False
    assert report["checks"]["audio"]["ok"] is False


def test_qa_checks_caption_sidecar_and_burned_output_artifact(tmp_path):
    video = _make_video(tmp_path)
    srt = tmp_path / "captions.srt"
    srt.write_text(
        "1\n00:00:00,200 --> 00:00:01,200\nA visible caption\n",
        encoding="utf-8",
    )
    clip = {"srt": "captions.srt", "subtitled_file": video.name}

    report = verify_rendered_file(
        str(video),
        expected_duration=2,
        expected_aspect="16:9",
        expected_audio=True,
        subtitles_requested=True,
        clip=clip,
        output_dir=str(tmp_path),
    )

    assert report["checks"]["subtitles"]["present"] is True
    assert report["checks"]["subtitles"]["valid"] is True
    assert report["checks"]["subtitles"]["cue_count"] == 1
    assert report["checks"]["subtitles"]["ok"] is True
    assert "no OCR" in report["checks"]["subtitles"]["verification_method"]


def test_qa_rejects_subtitle_blocks_with_no_caption_text(tmp_path):
    video = _make_video(tmp_path)
    srt = tmp_path / "empty-cue.srt"
    srt.write_text("1\n00:00:00,200 --> 00:00:01,200\n", encoding="utf-8")
    report = verify_rendered_file(
        str(video),
        expected_duration=2,
        subtitles_requested=True,
        clip={"srt": srt.name, "subtitled_file": video.name},
        output_dir=str(tmp_path),
    )
    assert report["checks"]["subtitles"]["present"] is True
    assert report["checks"]["subtitles"]["valid"] is False
    assert report["checks"]["subtitles"]["ok"] is False
