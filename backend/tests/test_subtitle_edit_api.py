"""The subtitle editor endpoint must keep SRT sidecars in sync with edited video."""
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import app as app_module
from autoclipper.subtitles import cues_to_srt


FFMPEG = shutil.which("ffmpeg")


def test_cues_to_srt_serializes_edited_cues_and_skips_invalid():
    srt = cues_to_srt([
        {"start": 0.2, "end": 1.0, "text": "Edited caption"},
        {"start": 1.0, "end": 0.5, "text": "backwards cue"},   # end <= start
        {"start": 2.0, "end": 3.5, "text": ""},                 # empty text
        {"start": None, "end": 4.0, "text": "bad start"},       # unparsable
        "not-a-cue",                                            # wrong type
        {"start": 4.0, "end": 5.0, "text": "Second\ncue"},      # newline flattened
    ])
    assert srt.startswith("1\n00:00:00,200 --> 00:00:01,000\nEdited caption\n")
    assert "2\n00:00:04,000 --> 00:00:05,000\nSecond cue" in srt
    assert "backwards cue" not in srt
    assert "bad start" not in srt
    assert cues_to_srt([]) == ""
    assert cues_to_srt(None) == ""


@pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")
def test_subtitle_edit_endpoint_returns_matching_srt_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(tmp_path))
    job_dir = tmp_path / "job1"
    job_dir.mkdir()
    video = job_dir / "clip_1.mp4"
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=25:d=2",
            "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # A stale pre-edit sidecar exists; the response must point at the EDITED one.
    (job_dir / "clip_1.srt").write_text(
        "1\n00:00:00,200 --> 00:00:01,200\nOld caption\n", encoding="utf-8"
    )

    with TestClient(app_module.app) as client:
        response = client.post("/api/subtitle", json={
            "job_id": "job1",
            "filename": "clip_1.mp4",
            "cues": [{"start": 0.2, "end": 1.0, "text": "Edited caption"}],
            "style": {},
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "clip_1_edited.mp4"
    assert payload["srt"] == "clip_1_edited.srt"
    assert (job_dir / "clip_1_edited.mp4").stat().st_size > 0
    srt_text = (job_dir / "clip_1_edited.srt").read_text(encoding="utf-8")
    assert "00:00:00,200 --> 00:00:01,000" in srt_text
    assert "Edited caption" in srt_text
    assert "Old caption" not in srt_text


@pytest.mark.skipif(not FFMPEG, reason="FFmpeg is not installed")
def test_subtitle_edit_endpoint_reports_no_sidecar_for_empty_cues(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(tmp_path))
    job_dir = tmp_path / "job2"
    job_dir.mkdir()
    video = job_dir / "clip_1.mp4"
    subprocess.run(
        [
            FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:r=25:d=1",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    with TestClient(app_module.app) as client:
        response = client.post("/api/subtitle", json={
            "job_id": "job2",
            "filename": "clip_1.mp4",
            "cues": [{"start": 1.0, "end": 0.5, "text": "invalid"}],
            "style": {},
        })

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "clip_1_edited.mp4"
    # No usable cues: no sidecar is claimed, so the UI omits the stale SRT.
    assert payload["srt"] is None
    assert not (job_dir / "clip_1_edited.srt").exists()
