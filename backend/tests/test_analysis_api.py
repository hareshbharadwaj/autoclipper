import time
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import app as app_module


def _fake_user():
    return SimpleNamespace(
        username="ci-test",
        groq_key=None,
        gemini_key=None,
        whisper_model=None,
        gemini_model=None,
        youtube_cookies=None,
    )


def test_analyze_video_rejects_empty_source():
    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user
    try:
        with TestClient(app_module.app) as client:
            response = client.post("/api/analyze-video", data={"source": ""})
        assert response.status_code == 400
        assert "source" in response.json()["detail"].lower()
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_analyze_video_unknown_id_returns_404():
    with TestClient(app_module.app) as client:
        response = client.get("/api/analyze-video/unknown-ci-analysis-id")
    assert response.status_code == 404


def test_analyze_video_stubbed_worker_returns_expected_report(monkeypatch):
    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user

    def fake_worker(analysis_id, source, opts):
        assert opts["content_type"] is None
        entry = app_module.analyses[analysis_id]
        entry["status"] = "done"
        entry["finished_at"] = time.time()
        entry["result"] = {
            "summary": "A synthetic test video summary.",
            "chapters": [{"start": 0.0, "end": 10.0, "label": "Main section"}],
            "key_topics": ["test topic"],
            "suggested_angles": [
                {
                    "title": "Best moment",
                    "hook": "Watch this",
                    "description": "A complete test moment.",
                    "target_duration": 10,
                    "start_hint": 1.0,
                    "end_hint": 9.0,
                    "custom_prompt": "Preserve the ending.",
                }
            ],
            "source": source,
        }

    monkeypatch.setattr(app_module, "_run_analysis", fake_worker)
    try:
        with TestClient(app_module.app) as client:
            started = client.post(
                "/api/analyze-video",
                data={"source": "https://example.invalid/synthetic.mp4", "angle_count": "4"},
            )
            assert started.status_code == 200, started.text
            analysis_id = started.json()["analysis_id"]
            report = None
            for _ in range(50):
                polled = client.get(f"/api/analyze-video/{analysis_id}")
                assert polled.status_code == 200
                report = polled.json()
                if report["status"] in {"done", "error"}:
                    break
                time.sleep(0.02)
        assert report["status"] == "done"
        assert report["result"]["summary"] == "A synthetic test video summary."
        assert len(report["result"]["chapters"]) == 1
        assert report["result"]["suggested_angles"][0]["target_duration"] == 10
        assert report["result"]["source"] == "https://example.invalid/synthetic.mp4"
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


@pytest.mark.parametrize("clip_count", ["0", "31", "-1"])
def test_process_rejects_invalid_clip_count_before_starting_work(clip_count):
    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user
    try:
        with TestClient(app_module.app) as client:
            response = client.post("/api/process", data={"clip_count": clip_count})
        assert response.status_code == 400
        assert "clip_count" in response.json()["detail"]
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_process_rejects_unknown_analysis_id():
    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user
    try:
        with TestClient(app_module.app) as test_client:
            response = test_client.post(
                "/api/process",
                data={
                    "source": "https://example.invalid/synthetic.mp4",
                    "analysis_id": "unknown-ci-analysis-id",
                },
            )
        assert response.status_code == 404
        assert "analysis id" in response.json()["detail"].lower()
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_analysis_worker_persists_transcript_for_generation_reuse(tmp_path, monkeypatch):
    import json
    import autoclipper.analyze as analyze_mod
    import autoclipper.ingest as ingest_mod
    import autoclipper.transcribe as transcribe_mod
    import autoclipper.utils as utils_mod

    output_root = tmp_path / "output"
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(output_root))
    transcript = {
        "duration": 12.0,
        "language": "English",
        "segments": [{"start": 0.0, "end": 12.0, "text": "Cached transcript."}],
        "words": [{"word": "Cached", "start": 0.0, "end": 1.0}],
    }
    monkeypatch.setattr(ingest_mod, "ingest", lambda source, out_dir, **kwargs: (str(tmp_path / "video.mp4"), "test-video"))
    monkeypatch.setattr(utils_mod, "probe_video_info", lambda path: {
        "width": 320, "height": 180, "duration": 12.0, "has_audio": True,
        "aspect": {"detected": "16:9", "recommended": "9:16"},
    })
    monkeypatch.setattr(transcribe_mod, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(analyze_mod, "analyze_video_summary_and_angles", lambda *args, **kwargs: {
        "summary": "A stubbed analysis report.",
        "chapters": [{"start": 0, "end": 12, "label": "Main"}],
        "key_topics": ["topic"],
        "suggested_angles": [],
    })
    analysis_id = "cache-test"
    app_module.analyses[analysis_id] = {
        "analysis_id": analysis_id, "status": "queued", "source": "input",
        "logs": [], "result": None, "error": None,
    }
    app_module._run_analysis(analysis_id, "input", {})
    entry = app_module.analyses[analysis_id]
    assert entry["status"] == "done"
    assert entry["transcript_path"]
    with open(entry["transcript_path"], encoding="utf-8") as handle:
        cached = json.load(handle)
    assert cached["segments"] == transcript["segments"]
    assert entry["result"]["source"] == str(tmp_path / "video.mp4")


def test_process_resolves_source_and_transcript_from_analysis_id(tmp_path, monkeypatch):
    import asyncio
    import json

    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user
    output_root = tmp_path / "output"
    analysis_id = "generation-cache-test"
    analysis_dir = output_root / "analysis" / analysis_id
    analysis_dir.mkdir(parents=True)
    transcript_path = analysis_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps({"duration": 12, "segments": [{"start": 0, "end": 12, "text": "Cached"}]}),
        encoding="utf-8",
    )
    source_path = str(analysis_dir / "source.mp4")
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(output_root))
    app_module.analyses[analysis_id] = {
        "analysis_id": analysis_id,
        "status": "done",
        "result": {"source": source_path},
        "transcript_path": str(transcript_path),
    }
    captured = {}

    async def fake_job(job_id, source, opts):
        captured.update({"job_id": job_id, "source": source, "opts": opts})
        await asyncio.sleep(0)

    monkeypatch.setattr(app_module, "_run_job", fake_job)
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/api/process",
                data={"analysis_id": analysis_id, "target_duration": "40", "clip_count": "3"},
            )
            assert response.status_code == 200, response.text
            for _ in range(20):
                if captured:
                    break
                time.sleep(0.01)
        assert captured["source"] == source_path
        assert captured["opts"]["cached_transcript_path"] == str(transcript_path)
        assert captured["opts"]["clip_count"] == 3
    finally:
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)
        app_module.analyses.pop(analysis_id, None)



def test_analysis_admission_rejects_when_queue_is_full(monkeypatch):
    app_module.app.dependency_overrides[app_module.get_current_user] = _fake_user
    monkeypatch.setattr(app_module, "MAX_PENDING_ANALYSES", 1)
    app_module.pending_analysis_ids.add("already-running")
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/api/analyze-video",
                data={"source": "https://example.invalid/large-video.mp4"},
            )
        assert response.status_code == 429
        assert response.headers.get("retry-after") == "30"
        assert "queue is full" in response.json()["detail"].lower()
    finally:
        app_module.pending_analysis_ids.discard("already-running")
        app_module.app.dependency_overrides.pop(app_module.get_current_user, None)


def test_analysis_worker_respects_force_hd_and_removes_admission_slot(tmp_path, monkeypatch):
    import autoclipper.analyze as analyze_mod
    import autoclipper.ingest as ingest_mod
    import autoclipper.utils as utils_mod

    calls = {}
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"synthetic")
    monkeypatch.setattr(app_module.config, "OUTPUT_ROOT", str(tmp_path / "output"))

    def fake_ingest(source, out_dir, **kwargs):
        calls.update(kwargs)
        return str(video_path), "synthetic"

    monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)
    monkeypatch.setattr(utils_mod, "probe_video_info", lambda path: {
        "width": 320, "height": 180, "duration": 12.0, "has_audio": True,
        "aspect": {"detected": "16:9", "recommended": "9:16"},
    })
    import autoclipper.transcribe as transcribe_mod
    monkeypatch.setattr(transcribe_mod, "transcribe", lambda *args, **kwargs: {
        "duration": 12.0, "language": "English",
        "segments": [{"start": 0, "end": 12, "text": "Synthetic"}],
        "words": [],
    })
    monkeypatch.setattr(analyze_mod, "analyze_video_summary_and_angles", lambda *args, **kwargs: {
        "summary": "Synthetic", "chapters": [], "key_topics": [], "suggested_angles": [],
    })
    analysis_id = "force-hd-test"
    app_module.analyses[analysis_id] = {
        "analysis_id": analysis_id, "status": "queued", "source": "input",
        "logs": [], "result": None, "error": None,
    }
    app_module.pending_analysis_ids.add(analysis_id)
    app_module._run_analysis(analysis_id, "input", {"force_hd": True})
    assert calls["force_hd"] is True
    assert calls["max_height"] is None
    assert analysis_id not in app_module.pending_analysis_ids
