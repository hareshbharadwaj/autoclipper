"""Pipeline integration for story planning: dynamic duration and QA split.

Uses synthetic transcripts and mocked model/media stages (deterministic). Real
FFmpeg behaviour is covered by the existing media/quality tests.
"""
import os

import pytest

from autoclipper import pipeline
from autoclipper.edit_planner import AI_NARRATION_LABEL


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
    assert "AI-expanded" in joined or "ai_expanded" in joined.lower()
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


def test_pipeline_ai_expanded_adds_labelled_narration_card(tmp_path, monkeypatch):
    """Scenario 7 integration: AI-expanded mode prepends a labelled narration
    card built from grounded quotes; source-only mode never does."""
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

    import autoclipper.voiceover as voiceover_mod

    def fake_generate_voiceover(text, out_path, engine="auto", voice=None, gemini_key=None):
        assert "friends make you happy" in text.lower()
        with open(out_path, "wb") as handle:
            handle.write(b"wav-bytes")
        return out_path

    def fake_build_intro(video_in, wav_in, out_path, hook_text="", **kwargs):
        assert hook_text == AI_NARRATION_LABEL
        with open(out_path, "wb") as handle:
            handle.write(b"card-bytes")
        return out_path

    def fake_concat_videos(a, b, out_path):
        with open(out_path, "wb") as handle:
            handle.write(b"final-bytes")
        return out_path

    monkeypatch.setattr(voiceover_mod, "generate_voiceover", fake_generate_voiceover)
    monkeypatch.setattr(voiceover_mod, "build_intro", fake_build_intro)
    monkeypatch.setattr(voiceover_mod, "concat_videos", fake_concat_videos)
    monkeypatch.setattr(voiceover_mod, "wav_duration", lambda _p: 6.5)

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 11.5},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)

    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=40, generation_mode="ai_expanded",
        vertical=False, subtitles=False,
    )
    clip = result["clips"][0]
    narration = clip.get("ai_narration")
    assert narration is not None
    assert narration["origin"] == "ai"
    assert narration["label"] == AI_NARRATION_LABEL
    assert narration["actual_duration"] == 6.5
    assert clip["file"].endswith("_ainarr.mp4")
    assert clip["planned_rendered_duration"] == pytest.approx(
        6.5 + clip["edit_plan"]["source_duration"], abs=0.5
    )
    assert result["generation_mode"] == "ai_expanded"


def test_pipeline_narration_failure_falls_back_to_source_only_with_warning(tmp_path, monkeypatch):
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

    import autoclipper.voiceover as voiceover_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("TTS engine unavailable")

    monkeypatch.setattr(voiceover_mod, "generate_voiceover", boom)

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=40, generation_mode="ai_expanded",
        vertical=False, subtitles=False,
    )
    clip = result["clips"][0]
    assert clip.get("ai_narration") is None
    assert clip["file"] == "clip_1.mp4"  # source-only clip delivered
    warnings = " ".join(result["processing_warnings"]).lower()
    assert "ai-expanded narration could not be generated" in warnings
    assert "source-only" in warnings


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
import shutil
import subprocess

from autoclipper.clip import cut_clip
from autoclipper.edit_planner import (
    build_edit_plan,
    finalize_narration_timing,
    rebase_cues,
)
from autoclipper.subtitles import parse_srt, rebase_srt_file
from autoclipper.utils import probe_duration

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark_media = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="FFmpeg/FFprobe are not installed"
)


@pytestmark_media
def test_real_render_matches_planned_edit_timeline(tmp_path):
    """Render the planned edit for real: cut -> narration card -> concat.

    Verifies the rendered timeline (via FFprobe) against the plan's edit
    timeline and rebases a caption sidecar onto it.
    """
    source = tmp_path / "source.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=20",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
         "-t", "20", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(source)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    transcript = {
        "duration": 20,
        "language": "English",
        "segments": [
            {"start": 0, "end": 5, "text": "The weather has been rainy all week."},
            {"start": 5, "end": 10, "text": "Your friends make you happy."},
            {"start": 10, "end": 15, "text": "I bought new shoes on sale."},
            {"start": 15, "end": 20, "text": "The football match got cancelled."},
        ],
    }
    candidate = {
        "start": 5.0, "end": 10.0,
        "title": "Friends make you happy", "hook": "Friends and happiness",
        "description": "A short remark about friends", "reason": "Short topic",
        "story_arc": "one line about friends",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=12,
                           generation_mode="ai_expanded", total_duration=20)
    assert plan["narration"] is not None

    clip_path = tmp_path / "clip.mp4"
    cut_clip(str(source), plan["source_start"], plan["source_end"], str(clip_path))
    source_span = plan["source_end"] - plan["source_start"]
    assert probe_duration(str(clip_path)) == pytest.approx(source_span, abs=0.75)

    # Stand in for TTS with real audio: a sine wav of the planned length.
    narration = plan["narration"]
    wav = tmp_path / "narration.wav"
    wav_seconds = 3.0
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i",
         f"sine=frequency=220:duration={wav_seconds}", str(wav)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    finalize_narration_timing(plan, wav_seconds)

    from autoclipper.voiceover import build_intro, concat_videos

    card = tmp_path / "card.mp4"
    build_intro(str(clip_path), str(wav), str(card),
                hook_text=narration["label"], duration=wav_seconds)
    final = tmp_path / "final.mp4"
    concat_videos(str(card), str(clip_path), str(final))

    rendered = probe_duration(str(final))
    assert rendered == pytest.approx(
        plan["estimated_duration"], abs=max(1.0, plan["estimated_duration"] * 0.1)
    )
    # Rendered timeline = narration card + source passages, never mixed up
    # with the transcript's own duration.
    assert rendered == pytest.approx(wav_seconds + source_span, abs=1.0)
    assert rendered != pytest.approx(20.0, abs=0.5)

    # Caption sidecar rebases onto the edit timeline (narration first).
    srt_path = tmp_path / "clip.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:02,500\nYour friends\n\n"
        "2\n00:00:02,500 --> 00:00:05,000\nmake you happy.\n",
        encoding="utf-8",
    )
    rebase_srt_file(str(srt_path), plan, clip_source_start=plan["source_start"])
    cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    assert cues[0]["start"] == pytest.approx(wav_seconds, abs=0.1)
    assert cues[1]["end"] == pytest.approx(wav_seconds + 5.0, abs=0.1)
    assert all(cue["end"] <= rendered + 0.15 for cue in cues)

    # Pure cue mapping agrees with the sidecar rewrite.
    rebased = rebase_cues(
        [{"start": 0.0, "end": 2.5, "text": "Your friends"}],
        plan,
        clip_source_start=plan["source_start"],
    )
    assert rebased[0]["start"] == pytest.approx(cues[0]["start"], abs=0.1)


# --- Review regressions: failure paths must keep metadata honest ------------

def _short_candidate():
    return {
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy", "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner", "story_arc": "one line about friends",
        "candidate_score": 90,
    }


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


@pytest.mark.parametrize("failing_stage", ["build_intro", "concat_videos"])
def test_narration_render_failure_restores_source_only_timing(
    tmp_path, monkeypatch, failing_stage
):
    """Review issue 1: when narration rendering fails, the delivered clip is
    source-only and the plan's duration metadata must say so — never include
    narration that was planned but never produced."""
    transcript = _stub_transcript()
    _patch_common(monkeypatch, tmp_path, transcript, [_short_candidate()])
    monkeypatch.setattr(pipeline, "cut_all", _fake_cut_all_factory(tmp_path))

    import autoclipper.voiceover as voiceover_mod

    def fake_generate_voiceover(text, out_path, engine="auto", voice=None, gemini_key=None):
        with open(out_path, "wb") as handle:
            handle.write(b"wav-bytes")
        return out_path

    def fake_build_intro(video_in, wav_in, out_path, hook_text="", **kwargs):
        with open(out_path, "wb") as handle:
            handle.write(b"card-bytes")
        return out_path

    def boom(*_args, **_kwargs):
        raise RuntimeError("narration render failed")

    monkeypatch.setattr(voiceover_mod, "generate_voiceover", fake_generate_voiceover)
    monkeypatch.setattr(voiceover_mod, "wav_duration", lambda _p: 6.5)
    if failing_stage == "build_intro":
        monkeypatch.setattr(voiceover_mod, "build_intro", boom)
    else:
        monkeypatch.setattr(voiceover_mod, "build_intro", fake_build_intro)
        monkeypatch.setattr(voiceover_mod, "concat_videos", boom)

    captured = {}

    def qa_pass(path, **kwargs):
        captured.setdefault("expected_duration", kwargs.get("expected_duration"))
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 5.0},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=40, generation_mode="ai_expanded",
        vertical=False, subtitles=False,
    )
    clip = result["clips"][0]
    plan = clip["edit_plan"]
    # The plan is back on the source-only timeline.
    assert plan["narration"] is None
    assert plan.get("narration_failed", {}).get("script")
    assert not any(seg.get("origin") == "ai" for seg in plan["segments"])
    assert plan["estimated_duration"] == pytest.approx(plan["source_duration"], abs=0.05)
    # No inflated rendered-duration expectation survives the failure: any
    # recorded duration matches the source-only clip that was actually kept.
    planned = clip.get("planned_rendered_duration")
    assert planned is None or planned == pytest.approx(plan["source_duration"], abs=0.05)
    assert captured["expected_duration"] == pytest.approx(plan["source_duration"], abs=0.05)
    # The untouched source clip is delivered with an explicit warning.
    assert clip["file"] == "clip_1.mp4"
    warnings = " ".join(result["processing_warnings"]).lower()
    assert "ai-expanded narration could not be generated" in warnings
    assert "source-only" in warnings


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


def test_pipeline_warns_when_caption_sidecar_cannot_rebase(tmp_path, monkeypatch):
    """Review issue 2 at pipeline level: the mismatch is surfaced as a warning
    instead of passing stale captions off as correct."""
    transcript = _stub_transcript()
    _patch_common(monkeypatch, tmp_path, transcript, [_short_candidate()])

    def cut_all_with_srt(_video, clips, out_dir, _title):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for i, candidate in enumerate(clips, start=1):
            name = f"clip_{i}.mp4"
            with open(os.path.join(out_dir, name), "wb") as handle:
                handle.write(b"rendered clip bytes")
            srt_name = f"clip_{i}.srt"
            with open(os.path.join(out_dir, srt_name), "w", encoding="utf-8") as handle:
                handle.write("1\n00:01:20,000 --> 00:01:22,000\nway outside\n")
            results.append({"index": i, "file": name, "srt": srt_name, **candidate})
        return results

    monkeypatch.setattr(pipeline, "cut_all", cut_all_with_srt)

    import autoclipper.voiceover as voiceover_mod

    def fake_generate_voiceover(text, out_path, engine="auto", voice=None, gemini_key=None):
        with open(out_path, "wb") as handle:
            handle.write(b"wav-bytes")
        return out_path

    def fake_build_intro(video_in, wav_in, out_path, hook_text="", **kwargs):
        with open(out_path, "wb") as handle:
            handle.write(b"card-bytes")
        return out_path

    def fake_concat(a, b, out_path):
        with open(out_path, "wb") as handle:
            handle.write(b"final-bytes")
        return out_path

    monkeypatch.setattr(voiceover_mod, "generate_voiceover", fake_generate_voiceover)
    monkeypatch.setattr(voiceover_mod, "build_intro", fake_build_intro)
    monkeypatch.setattr(voiceover_mod, "concat_videos", fake_concat)
    monkeypatch.setattr(voiceover_mod, "wav_duration", lambda _p: 6.5)

    def qa_pass(path, **_kwargs):
        return {"status": "pass", "file": os.path.basename(path), "checks": {
            "duration": {"ok": True, "actual_seconds": 11.5},
        }, "warnings": [], "errors": []}

    monkeypatch.setattr(pipeline, "verify_rendered_file", qa_pass)
    result = pipeline.run(
        "stub", output_dir=str(tmp_path / "job"),
        clip_count=1, target_duration=40, generation_mode="ai_expanded",
        vertical=False, subtitles=False,
    )
    warnings = " ".join(result["processing_warnings"]).lower()
    assert "caption sidecar could not be rebased" in warnings
    assert "does not match the final video" in warnings


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
