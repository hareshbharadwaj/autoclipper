import asyncio
import os
import time

from fastapi.testclient import TestClient

import app as app_module
from autoclipper.utils import save_to_library


def test_job_runner_streams_structured_progress_and_status_endpoint(tmp_path, monkeypatch):
    job_id = "progress-qa-test"
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(tmp_path))
    app_module.jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "source": "authorized-local-source.mp4",
        "created_at": time.time(),
        "finished_at": None,
        "logs": [],
        "result": None,
        "error": None,
        "progress": {"stage": "queued", "message": "Waiting for a processing slot."},
    }

    def fake_pipeline_run(source, output_dir, on_clip, on_progress, **_options):
        on_progress({
            "stage": "quality_assurance",
            "message": "Checking rendered outputs.",
            "current": 0,
            "total": 1,
            "qa_status": "running",
        })
        on_clip({"file": "clip_1.mp4", "index": 1}, 1, 1)
        return {
            "title": "Synthetic",
            "source": source,
            "job_id": os.path.basename(output_dir),
            "clips": [{
                "file": "clip_1.mp4", "index": 1, "qa_status": "pass",
                "quality_assurance": {"status": "pass"},
            }],
            "quality_assurance": {
                "status": "pass", "checked_clip_count": 1, "failed_clip_count": 0,
                "reports": [], "failed_outputs": [],
            },
            "processing_warnings": [],
        }

    monkeypatch.setattr(app_module.pipeline, "run", fake_pipeline_run)
    try:
        asyncio.run(app_module._run_job(job_id, "authorized-local-source.mp4", {}))
        with TestClient(app_module.app) as client:
            response = client.get(f"/api/status/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "completed"
        assert payload["progress"]["stage"] == "completed"
        assert payload["progress"]["qa_status"] == "pass"
        assert payload["result"]["clips"][0]["qa_status"] == "pass"
        assert payload["result"]["quality_assurance"]["checked_clip_count"] == 1
    finally:
        app_module.jobs.pop(job_id, None)


def test_automatic_library_copy_preserves_clip_subtitle_sidecar(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    job_dir = output_root / "job-with-subtitles"
    job_dir.mkdir(parents=True)
    (job_dir / "final-hook-audio.mp4").write_bytes(b"video")
    (job_dir / "source-caption.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nCaption text\n", encoding="utf-8"
    )
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(output_root))

    library_dir = save_to_library(
        str(output_root),
        "Source title",
        "job-with-subtitles",
        [{
            "index": 1,
            "file": "final-hook-audio.mp4",
            "srt": "source-caption.srt",
            "title": "A clip",
        }],
    )

    assert os.path.isfile(os.path.join(library_dir, "clip1.mp4"))
    assert os.path.isfile(os.path.join(library_dir, "clip1.srt"))


def test_manual_library_save_copies_explicit_subtitle_sidecar(tmp_path, monkeypatch):
    output_root = tmp_path / "output"
    job_dir = output_root / "job-subtitle"
    job_dir.mkdir(parents=True)
    (job_dir / "rendered.mp4").write_bytes(b"video")
    (job_dir / "subtitle-source.srt").write_text("caption", encoding="utf-8")
    library_root = output_root / "library"
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(output_root))
    monkeypatch.setattr(app_module, "LIBRARY_ROOT", str(library_root))

    with TestClient(app_module.app) as client:
        response = client.post("/api/library/save", json={
            "job_id": "job-subtitle",
            "filename": "rendered.mp4",
            "srt": "subtitle-source.srt",
            "title": "Saved source",
            "clip_title": "A clip",
        })

    assert response.status_code == 200, response.text
    folder = response.json()["folder"]
    assert (library_root / folder / "clip1.mp4").is_file()
    assert (library_root / folder / "clip1.srt").is_file()
