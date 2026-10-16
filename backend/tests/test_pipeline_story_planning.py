"""Pipeline integration for story planning: dynamic duration and QA split.

Uses synthetic transcripts and mocked model/media stages (deterministic). Real
FFmpeg behaviour is covered by the existing media/quality tests.
"""
import os

import pytest

from autoclipper import pipeline
from autoclipper.edit_planner import build_edit_plan
from autoclipper.subtitles import rebase_srt_file


def _stub_transcript():
    return {
        "duration": 40,
        "language": "English",
        "segments": [
            {"start": 0, "end": 8,
             "text": "The weather has been extremely rainy this entire week."},
            {"start": 8, "end": 12,
             "text": "Traffic downtown was completely awful today."},
            {"start": 12, "end": 17,
             "text": "Your friends make you happy."},
            {"start": 17, "end": 28,
             "text": "I bought new shoes over the weekend and they were on sale."},
            {"start": 28, "end": 40,
             "text": "Anyway the football match got cancelled because of the storm."},
        ],
        "words": [],
    }


def _patch_common(monkeypatch, tmp_path, transcript, candidates):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source placeholder")
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **k: (str(source), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda _p: {
        "width": 1920, "height": 1080, "duration": 40, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: candidates)
    monkeypatch.setattr(pipeline, "review_clip_candidates", lambda *a, **k: {
        "candidates": candidates, "reviewed_count": len(candidates), "rejected_count": 0,
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "save_to_library",
        lambda *a, **k: os.path.join(str(tmp_path), "library"),
    )
    return source


def _fake_cut_all_factory(folder):
    def fake_cut_all(_video, clips, out_dir, _title):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for i, candidate in enumerate(clips, start=1):
            name = f"clip_{i}.mp4"
            with open(os.path.join(out_dir, name), "wb") as handle:
                handle.write(b"rendered clip bytes")
            results.append({"index": i, "file": name, **candidate})
        return results
    return fake_cut_all



def _short_candidate():
    return {
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy",
        "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner",
        "story_arc": "one line about friends",
        "candidate_score": 90,
    }

def test_pipeline_reports_shortfall_instead_of_silently_dropping_clips(tmp_path, monkeypatch):
    """Scenario 8 (planning side): a 60s request with a 5s topic must be
    explained, never silently returned as a much shorter clip or dropped."""
    transcript = _stub_transcript()
    candidates = [{
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy",
        "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner",
        "story_arc": "one line about friends",
        "candidate_score": 90,
    }]
    _patch_common(monkeypatch, tmp_path, transcript, candidates)
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)

    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=60, generation_mode="source_only",
        vertical=False, subtitles=False,
    )
    assert len(result["clips"]) == 1
    clip = result["clips"][0]
    assert clip["edit_plan"]["duration_status"] == "short"
    assert clip["editorial_qa"]["status"] in {"pass", "warning", "fail"}
    joined = " ".join(result["processing_warnings"])
    assert "60s" in joined  # the requested target is named
    assert "keeps source audio" in joined.lower() or "never generates narration" in joined.lower()
    # The rendered-vs-target gap is reported as well.
    assert any("requested 60s" in warning.lower() for warning in result["processing_warnings"])


def test_pipeline_rendered_duration_mismatch_is_reported(tmp_path, monkeypatch):
    """Scenario 8 (render side): an unexpectedly short final render triggers
    an actionable warning against the requested target."""
    transcript = _stub_transcript()
    candidates = [{
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy", "hook": "Friends and happiness",
        "description": "one line", "reason": "short", "story_arc": "one line",
        "candidate_score": 90,
    }]
    _patch_common(monkeypatch, tmp_path, transcript, candidates)
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_short(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 8.5},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_short)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=60, generation_mode="source_only",
        vertical=False, subtitles=False,
    )
    warnings = " ".join(result["processing_warnings"]).lower()
    assert "requested 60s" in warnings
    assert "8.5s" in warnings


def test_pipeline_technical_failure_keeps_editorial_report(tmp_path, monkeypatch):
    """Scenario 11: render failure is a technical verdict; the editorial
    review of the planned story is still reported."""
    transcript = _stub_transcript()
    candidates = [{
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy", "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner", "story_arc": "one line about friends",
        "candidate_score": 90,
    }]
    _patch_common(monkeypatch, tmp_path, transcript, candidates)
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_fail(path, **_kwargs):
        return {
            "status": "fail", "file": os.path.basename(path),
            "checks": {"audio": {"ok": False}},
            "warnings": [],
            "errors": ["Expected source audio is missing from the rendered output."],
        }

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_fail)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=40, generation_mode="source_only",
        vertical=False, subtitles=False,
    )
    assert result["quality_assurance"]["status"] == "fail"
    assert result["clips"] == []
    assert result["quality_assurance"]["failed_outputs"], "failed output must be reported"
    failed = result["quality_assurance"]["failed_outputs"][0]
    assert failed.get("editorial_qa", {}).get("status") in {"pass", "warning", "fail"}
    # The job-level editorial summary reflects the planned story review.
    assert result["editorial_qa"]["checked_clip_count"] == 1
    assert result["editorial_qa"]["reports"][0]["checks"]["opening_not_fragment"]["ok"] is True


def test_pipeline_rejects_unknown_generation_mode(tmp_path, monkeypatch):
    _patch_common(monkeypatch, tmp_path, _stub_transcript(), [])
    with pytest.raises(ValueError, match="generation_mode"):
        pipeline.run(
            "stub", output_dir=str(tmp_path / "job"),
            clip_count=1, generation_mode="make_up_dialogue",
            vertical=False, subtitles=False,
        )


# --- Real-media integration (FFmpeg/FFprobe) --------------------------------
def test_pipeline_never_generates_captions_during_render(tmp_path, monkeypatch):
    """Captions are not generated or burned during generation/reframing.

    The legacy ``subtitles=`` argument is accepted but ignored; users add
    captions later from the clip editor.
    """
    transcript = _stub_transcript()
    _patch_common(monkeypatch, tmp_path, transcript, [_short_candidate()])
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_pass(path, **kwargs):
        assert "subtitles_requested" not in kwargs
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=15, generation_mode="source_only",
        vertical=False, subtitles=True,  # legacy kwarg must be a no-op
    )
    clip = result["clips"][0]
    assert "srt" not in clip
    assert "subtitled_file" not in clip
    job_dir = tmp_path / "job"
    assert not list(job_dir.glob("*.srt"))
    assert not [name for name in os.listdir(job_dir) if "sub" in name.lower()]


def test_rebase_srt_file_fails_loudly_when_captions_cannot_rebase(tmp_path):
    """Review issue 2: an unmappable or missing caption sidecar must raise,
    never silently return mistimed captions as if they were rebased."""
    plan = build_edit_plan(
        {
            "duration": 40, "language": "English",
            "segments": [
                {"start": 0, "end": 8, "text": "The weather has been rainy."},
                {"start": 8, "end": 12, "text": "Traffic was awful."},
                {"start": 12, "end": 17, "text": "Your friends make you happy."},
            ],
        },
        {
            "start": 12.0, "end": 17.0,
            "title": "Friends make you happy", "hook": "Friends",
            "description": "one line", "reason": "short", "story_arc": "one line",
        },
        target_duration=15,
        generation_mode="source_only",
    )
    with pytest.raises(ValueError, match="missing"):
        rebase_srt_file(str(tmp_path / "nope.srt"), plan, clip_source_start=12.0)

    unmappable = tmp_path / "unmappable.srt"
    unmappable.write_text(
        "1\n00:01:20,000 --> 00:01:22,000\nway outside\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No caption cues could be mapped"):
        rebase_srt_file(str(unmappable), plan, clip_source_start=12.0)

    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no parseable cues"):
        rebase_srt_file(str(empty), plan, clip_source_start=12.0)


def test_pipeline_marks_clip_unreviewed_when_edit_planning_fails(tmp_path, monkeypatch):
    """Review issue 3: a planning failure is marked and can never look like a
    passed editorial review."""
    transcript = {"duration": 20, "language": "English", "segments": [], "words": []}
    _patch_common(monkeypatch, tmp_path, transcript, [_short_candidate()])
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=15, generation_mode="source_only",
        vertical=False, subtitles=False,
    )
    clip = result["clips"][0]
    assert clip.get("planning_failed") is True
    assert clip.get("edit_plan") is None
    assert clip["editorial_qa"]["status"] == "not_run"
    # Job summary must not read as a successful review.
    assert result["editorial_qa"]["status"] == "warning"
    assert result["editorial_qa"]["unchecked_clip_count"] == 1
    assert any("unreviewed" in w.lower() for w in result["processing_warnings"])

def test_legacy_ai_expanded_request_stays_source_audio_only(tmp_path, monkeypatch):
    """Removed narration mode must not create a narration card or change audio."""
    transcript = _stub_transcript()
    _patch_common(monkeypatch, tmp_path, transcript, [_short_candidate()])
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "audio": {"ok": True}, "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=15, generation_mode="ai_expanded",
        vertical=False, subtitles=False,
    )

    clip = result["clips"][0]
    assert clip["edit_plan"]["mode"] == "source_only"
    assert clip["edit_plan"].get("narration") is None
    assert clip.get("ai_narration") is None
    assert not any("voiceover" in warning.lower() or "background music" in warning.lower()
                   for warning in result["processing_warnings"])

