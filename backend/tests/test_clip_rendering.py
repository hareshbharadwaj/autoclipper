import math
import os
import shutil
import subprocess

import pytest

from autoclipper import clip as clip_mod
from autoclipper.utils import probe_duration, probe_video_info


FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")


def _run(args):
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _make_source(tmp_path):
    path = tmp_path / "render_source.mp4"
    _run([
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(path),
    ])
    return path


def test_cut_clip_reencodes_requested_window_and_preserves_audio(tmp_path):
    source = _make_source(tmp_path)
    output = tmp_path / "cut.mp4"

    assert clip_mod.cut_clip(str(source), 0.6, 1.8, str(output)) == str(output)
    info = probe_video_info(str(output))
    assert output.is_file() and output.stat().st_size > 0
    assert (info["width"], info["height"]) == (320, 180)
    assert info["has_audio"] is True
    assert 1.05 <= probe_duration(str(output)) <= 1.35


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 2), (math.nan, 2), (1, math.inf), (2, 1), (0, 0)],
)
def test_cut_clip_rejects_invalid_boundaries_before_running_ffmpeg(tmp_path, monkeypatch, start, end):
    source = tmp_path / "exists.mp4"
    source.write_bytes(b"stub")
    monkeypatch.setattr(
        clip_mod,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("FFmpeg should not run for an invalid interval"),
    )

    with pytest.raises(ValueError):
        clip_mod.cut_clip(str(source), start, end, str(tmp_path / "bad.mp4"))


def test_cut_all_omits_partial_outputs_and_continues(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []

    def fake_cut(_source, start, end, output):
        calls.append((start, end))
        if start == 10:
            with open(output, "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("synthetic ffmpeg error")
        with open(output, "wb") as handle:
            handle.write(b"valid")

    monkeypatch.setattr(clip_mod, "cut_clip", fake_cut)
    results = clip_mod.cut_all(
        str(source),
        [
            {"start": 0, "end": 5, "title": "First"},
            {"start": 10, "end": 15, "title": "Broken"},
            {"start": 20, "end": 25, "title": "Last"},
        ],
        str(tmp_path / "out"),
        "demo",
    )

    assert len(results) == 2
    assert [item["index"] for item in results] == [1, 3]
    assert not (tmp_path / "out" / "demo_clip_2.mp4").exists()
    assert len(calls) == 3
