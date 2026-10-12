"""Integration tests: source-grounded selection, honest fallbacks, no padding.

These tests reproduce the failure modes behind the knife-safety case
(91-117s selection silently widened to 84-144s to hit a 60s target while the
independent review was unavailable) and the hallucination-resistance
requirements: every output must be traceable to source timestamps, generated
text must be supported by the source, and failed checks are never reported
as passes.
"""
import json
import os
import types

import pytest

from autoclipper import pipeline
from autoclipper.analyze import (
    CandidateReviewError,
    CandidateReviewTimeoutError,
)


KNIFE_DIRECTION = (
    "Create a fast-paced summary of crucial knife safety rules, including safe "
    "placement after use, safe handling while walking, and knife orientation "
    "when not actively in use."
)


def _knife_transcript():
    """Synthetic stand-in for the 10-minute knife-safety video.

    The real media file is not available in this environment; this fixture
    models its structure: a safety-rules block at 91-117s surrounded by
    unrelated material.
    """
    segments = [
        {"start": 0.0, "end": 30.0,
         "text": "Welcome back to the channel, today we prep the marinade for the barbecue."},
        {"start": 30.0, "end": 60.0,
         "text": "Mix the soy sauce, garlic, and brown sugar in a large bowl."},
        {"start": 60.0, "end": 91.0,
         "text": "Let the meat rest while we talk about the grill setup outside."},
        {"start": 91.0, "end": 100.0,
         "text": "Rule one, after you use the knife place it flat on the board."},
        {"start": 100.0, "end": 109.0,
         "text": "Rule two, when walking hold the knife pointing down at your side."},
        {"start": 109.0, "end": 117.0,
         "text": "Rule three, keep the blade facing away while it rests on the bench."},
        {"start": 117.0, "end": 140.0,
         "text": "Now let us switch topics and talk about the garden tomato harvest."},
        {"start": 140.0, "end": 170.0,
         "text": "Tomatoes need sun and water, and the harvest begins in late summer."},
    ]
    return {
        "language": "english",
        "duration": 604.0,
        "text": " ".join(seg["text"] for seg in segments),
        "segments": segments,
        "words": [],
    }


def _patch_common(monkeypatch, tmp_path, transcript, *, duration=604.0):
    video_path = tmp_path / "knife-safety-fixture.mp4"
    video_path.write_bytes(b"synthetic fixture media placeholder")
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *a, **k: (str(video_path), "Knife Safety Fixture"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 854, "height": 480, "duration": duration, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *a, **k: transcript)
    monkeypatch.setattr(
        pipeline,
        "cut_all",
        lambda video, clips, out_dir, title: [
            {"index": i + 1, "file": f"clip_{i + 1}.mp4", **clip}
            for i, clip in enumerate(clips)
        ],
    )
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path), "checks": {
            "file": {"ok": True}, "streams": {"ok": True},
            "duration": {"ok": True}, "audio": {"ok": True},
            "visual_samples": {"ok": True}, "aspect_ratio": {"ok": True},
        },
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )
    return video_path


def test_knife_safety_selection_is_not_widened_with_unrelated_footage(tmp_path, monkeypatch):
    """The reproduction case: 91-117s angle, 60s target, review unavailable.

    Before the fix the selection was dropped by the target window and the
    generator invented 84.2-144.0s. Now the user's span is kept, never widened
    with unrelated material, and the duration shortfall is explained.
    """
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: pytest.fail(
        "The pipeline must not regenerate a replacement for a valid user selection."
    ))

    result = pipeline.run(
        source="fixture",
        output_dir=str(tmp_path / "job"),
        subtitles=False,
        aspect_ratio="16:9",
        clip_count=1,
        target_duration=60,
        custom_prompt=KNIFE_DIRECTION,
        selected_clip={
            "title": "3 Knife Rules You NEED to Know!",
            "hook": "Three knife rules",
            "start_hint": 91.0,
            "end_hint": 117.0,
            "start": 91.0,
            "end": 117.0,
        },
    )

    assert len(result["clips"]) == 1
    clip = result["clips"][0]
    # The boundaries stayed on the requested safety-rules span: no backward
    # drift into the intro, no forward drift into the tomato segment.
    assert clip["start"] >= 91.0
    assert clip["end"] <= 117.0
    assert clip["planned_duration"] == 26.0
    # Honest duration reporting: requested 60s, actual 26s, with a reason.
    assert clip["requested_duration"] == 60.0
    assert clip["duration_delta"] == -34.0
    assert clip["editorial_qa"]["duration_note"]
    assert any("padding" in warning for warning in result["processing_warnings"])
    # Source traceability survived to the final metadata.
    assert clip["source_intervals"] == [{"start": 91.0, "end": 117.0}]
    assert "Rule one" in clip["transcript_span"]
    assert "tomato" not in clip["transcript_span"]
    # Factual QA ran and did not fail the user's selection.
    assert clip["factual_qa"]["status"] in ("passed", "warning")
    assert result["validation_summary"]["factual"] in ("pass", "warning")


def test_extension_only_uses_contiguous_relevant_segments(tmp_path, monkeypatch):
    """A short selection may grow toward the target, but only with speech that
    is topically continuous with it — never with unrelated filler."""
    transcript = _knife_transcript()
    # Append a genuine continuation of the rules right after 117s.
    transcript["segments"][6:6] = [
        {"start": 117.0, "end": 133.0,
         "text": "Rule four, never leave the knife blade facing up on the counter "
                 "and always keep the knife flat on the board after you use it."},
    ]
    # Re-number the following unrelated segments.
    transcript["segments"][7]["start"] = 133.0
    transcript["segments"][7]["end"] = 160.0
    transcript["segments"][8]["start"] = 160.0
    transcript["segments"][8]["end"] = 190.0
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: pytest.fail(
        "The pipeline must not regenerate a replacement for a valid user selection."
    ))

    result = pipeline.run(
        source="fixture",
        output_dir=str(tmp_path / "job"),
        subtitles=False,
        clip_count=1,
        target_duration=40,
        custom_prompt=KNIFE_DIRECTION,
        selected_clip={
            "title": "3 Knife Rules You NEED to Know!",
            "start": 91.0, "end": 117.0,
        },
    )
    clip = result["clips"][0]
    # The relevant continuation was added; the unrelated tomato segment was not.
    assert clip["start"] == 91.0
    assert clip["end"] == 133.0
    assert "tomato" not in clip["transcript_span"]
    assert "Rule four" in clip["transcript_span"]
    assert any("Extended the user-selected span" in w for w in result["processing_warnings"])


def test_review_failure_is_labeled_unreviewed_not_passed(tmp_path, monkeypatch):
    """When independent review is unavailable, candidates must be labeled
    unreviewed and still pass deterministic grounding — never reported as
    successfully reviewed."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife rules",
            "candidate_score": 80.0,
            "evidence_quote": "Rule one, after you use the knife",
            "evidence_quote_verified": True,
        },
    ])
    monkeypatch.setattr(
        pipeline, "review_clip_candidates",
        lambda *a, **k: (_ for _ in ()).throw(
            CandidateReviewError("Independent candidate review was unavailable (ClientError).", "client_error")
        ),
    )

    result = pipeline.run(
        source="fixture",
        output_dir=str(tmp_path / "job"),
        subtitles=False,
        clip_count=1,
        gemini_key="dummy",  # force the review path to be attempted
        min_clip=20,
        max_clip=30,
    )
    assert result["candidate_review"]["status"] == "unavailable"
    assert result["candidate_review"]["failure_reason"]
    assert any("UNREVIEWED" in warning for warning in result["processing_warnings"])
    clip = result["clips"][0]
    assert clip["factual_qa"]["checks"]["review_state"] == "unavailable"
    # The technical QA pass must not be misread as a full verification.
    assert result["validation_summary"]["technical"] == "pass"


def test_review_timeout_and_malformed_are_distinguished(tmp_path, monkeypatch):
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {"start": 91.0, "end": 117.0, "title": "Knife rules",
         "candidate_score": 80.0,
         "evidence_quote": "Rule one, after you use the knife",
         "evidence_quote_verified": True},
    ])

    monkeypatch.setattr(
        pipeline, "review_clip_candidates",
        lambda *a, **k: (_ for _ in ()).throw(
            CandidateReviewTimeoutError("Independent candidate review timed out.")
        ),
    )
    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job1"), subtitles=False,
        clip_count=1, gemini_key="dummy", min_clip=20, max_clip=30,
    )
    assert result["candidate_review"]["status"] == "timed_out"

    monkeypatch.setattr(
        pipeline, "review_clip_candidates",
        lambda *a, **k: (_ for _ in ()).throw(
            CandidateReviewError("Independent candidate review returned no reviews array.", "malformed")
        ),
    )
    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job2"), subtitles=False,
        clip_count=1, gemini_key="dummy", min_clip=20, max_clip=30,
    )
    assert result["candidate_review"]["status"] == "malformed"


def test_off_topic_generated_candidate_is_rejected_before_render(tmp_path, monkeypatch):
    """An AI-selected segment unrelated to the user's prompt must not render."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife safety rules",
            "candidate_score": 90.0,
            "evidence_quote": "Rule one, after you use the knife",
            "evidence_quote_verified": True,
        },
        {
            "start": 140.0, "end": 170.0, "title": "Tomato harvest tips",
            "candidate_score": 95.0,
            "evidence_quote": "Tomatoes need sun and water",
            "evidence_quote_verified": True,
        },
    ])

    result = pipeline.run(
        source="fixture",
        output_dir=str(tmp_path / "job"),
        subtitles=False,
        clip_count=2,
        min_clip=20,
        max_clip=40,
        custom_prompt=KNIFE_DIRECTION,
    )
    assert len(result["clips"]) == 1
    assert result["clips"][0]["start"] == 91.0
    rejected = result["candidate_review"]["grounding_rejected"]
    assert any("Tomato" in str(item["title"]) for item in rejected)
    assert any("unrelated" in reason for item in rejected for reason in item["reasons"])


def test_fabricated_evidence_quote_never_renders_from_generation(tmp_path, monkeypatch):
    """Generated candidates whose evidence quote is invented are unusable."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife rules",
            "candidate_score": 90.0,
            # This sentence exists nowhere in the transcript.
            "evidence_quote": "Knives are the most dangerous tool in any home",
            "evidence_quote_verified": True,
        },
    ])

    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job"), subtitles=False,
        clip_count=1, min_clip=20, max_clip=40,
    )
    assert result["clips"] == []
    rejected = result["candidate_review"]["grounding_rejected"]
    assert rejected and any("evidence quote" in r for item in rejected for r in item["reasons"])


def test_negation_flipping_candidate_is_rejected(tmp_path, monkeypatch):
    """A generated cut that strips the 'do not' out of a sentence is a
    critical factual failure and must not render."""
    transcript = {
        "language": "english",
        "duration": 40.0,
        "segments": [
            {"start": 0.0, "end": 8.0, "text": "Do not walk around with a raised blade."},
            {"start": 8.0, "end": 20.0, "text": "Keep the point down beside your leg."},
            {"start": 20.0, "end": 32.0, "text": "Hand the handle first when passing a knife."},
        ],
        "words": [],
    }
    transcript["text"] = " ".join(seg["text"] for seg in transcript["segments"])
    _patch_common(monkeypatch, tmp_path, transcript, duration=40.0)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            # Starts at 3.0s: after the negation, inside the first sentence.
            "start": 3.0, "end": 32.0, "title": "How to carry a knife",
            "candidate_score": 88.0,
            "evidence_quote": "Keep the point down beside your leg.",
            "evidence_quote_verified": True,
        },
    ])

    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job"), subtitles=False,
        clip_count=1, min_clip=20, max_clip=40,
    )
    assert result["clips"] == []
    rejected = result["candidate_review"]["grounding_rejected"]
    assert any("negation" in reason for item in rejected for reason in item["reasons"])


def test_target_longer_than_material_returns_shorter_clip_with_explanation(tmp_path, monkeypatch):
    """Duration is a goal, not permission to pad: when the source lacks enough
    relevant material, deliver the best valid shorter clip and explain."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    # Generation returns the only relevant span in the whole video (26s) even
    # though the prompt demanded a 60s window.
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife safety rules",
            "candidate_score": 90.0,
            "evidence_quote": "Rule one, after you use the knife",
            "evidence_quote_verified": True,
        },
    ])

    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job"), subtitles=False,
        clip_count=1, target_duration=60, custom_prompt=KNIFE_DIRECTION,
    )
    assert len(result["clips"]) == 1
    clip = result["clips"][0]
    assert clip["planned_duration"] == 26.0
    assert clip["duration_delta"] == -34.0
    assert clip["editorial_qa"]["duration_note"]
    assert any(
        "did not provide enough relevant material" in warning
        for warning in result["processing_warnings"]
    )
    # Nothing outside the rules block was added to reach 60 seconds.
    assert clip["source_intervals"] == [{"start": 91.0, "end": 117.0}]


def test_transcript_only_pipeline_never_reports_visual_verification(tmp_path, monkeypatch):
    """With no sampled frames, no clip may claim visual verification — even if
    the reviewer model asserts it saw something."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife safety rules",
            "candidate_score": 90.0,
            "evidence_quote": "Rule one, after you use the knife",
            "evidence_quote_verified": True,
        },
    ])
    import autoclipper.visual_context as visual_context_mod
    monkeypatch.setattr(
        visual_context_mod, "sample_candidate_frames",
        lambda *a, **k: {"frames": [], "warning": "FFmpeg unavailable in this test."},
    )

    def fake_review(transcript_arg, candidates, **kwargs):
        reviewed = []
        for candidate in candidates:
            reviewed.append({
                **candidate,
                "review_status": "accepted",
                "review_score": 90.0,
                "review_reason": "ok",
                "review_evidence": candidate["evidence_quote"],
                # The model claims it saw the demonstration — but NO frames
                # were attached, so this claim must not survive.
                "visual_evidence": "The knife technique is clearly visible.",
                "visual_frames_attached": False,
            })
        return {"candidates": reviewed, "reviewed_count": len(reviewed), "rejected_count": 0}

    monkeypatch.setattr(pipeline, "review_clip_candidates", fake_review)

    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job"), subtitles=False,
        clip_count=1, gemini_key="dummy", min_clip=20, max_clip=30,
    )
    assert result["candidate_review"]["status"] == "reviewed"
    clip = result["clips"][0]
    factual = clip["factual_qa"]["checks"]
    # Zero frames were attached, so the verification level must be transcript-
    # only even though the review itself completed.
    assert factual["visual_verification"] == "transcript_only"
    assert factual["review_state"] == "reviewed"
    # The model's claim that it saw the technique cannot stand without frames.
    assert clip.get("visual_frames_attached") is False
    assert "not verifiable" in (clip.get("visual_evidence") or "").casefold()


def test_fabricated_hook_quote_is_never_burned(tmp_path, monkeypatch):
    """A hook that invents a quotation must be skipped, not rendered."""
    transcript = _knife_transcript()
    _patch_common(monkeypatch, tmp_path, transcript)

    def fail_if_burned(*args, **kwargs):
        pytest.fail("An ungrounded hook must never reach the renderer.")

    import autoclipper.hooks as hooks_mod
    monkeypatch.setattr(hooks_mod, "add_hook_to_video", fail_if_burned)

    # Case 1: an AI-generated candidate whose hook fabricates a quotation is
    # rejected outright by factual grounding (never rendered).
    monkeypatch.setattr(pipeline, "find_viral_clips", lambda *a, **k: [
        {
            "start": 91.0, "end": 117.0, "title": "Knife rules",
            "candidate_score": 90.0,
            "hook": '"Never buy cheap knives" - every chef knows this',
            "evidence_quote": "Rule one, after you use the knife",
            "evidence_quote_verified": True,
        },
    ])
    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job"), subtitles=False,
        clip_count=1, min_clip=20, max_clip=40, add_hook=True,
    )
    assert result["clips"] == []
    rejected = result["candidate_review"]["grounding_rejected"]
    assert any("hook" in reason.lower() for item in rejected for reason in item["reasons"])

    # Case 2: the user explicitly selected this moment, so it is retained —
    # but the fabricated hook overlay must still be skipped at render time.
    result = pipeline.run(
        source="fixture", output_dir=str(tmp_path / "job2"), subtitles=False,
        clip_count=1, target_duration=26, add_hook=True,
        custom_prompt=KNIFE_DIRECTION,
        selected_clip={
            "title": "Knife rules",
            "hook": '"Never buy cheap knives" - every chef knows this',
            "start": 91.0, "end": 117.0,
        },
    )
    clip = result["clips"][0]
    assert clip.get("hook_skipped_ungrounded") is True
    assert not clip["file"].endswith("_hook.mp4")
    assert any("hook text is not supported" in w for w in result["processing_warnings"])
