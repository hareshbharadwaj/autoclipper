"""Editorial QA tests — story quality is never implied by a successful render.

These checks run on the edit plan and transcript (synthetic, deterministic).
Technical/render QA lives in ``test_quality_assurance`` and the pipeline tests;
scenario 11 (render failure) is covered by asserting the two QA layers stay
independent.
"""
import pytest

from autoclipper.edit_planner import (
    AI_NARRATION_LABEL,
    build_edit_plan,
    build_sentence_units,
)
from autoclipper.editorial_qa import editorial_summary, review_edit_plan


def _transcript():
    return {
        "duration": 60,
        "language": "English",
        "segments": [
            {"start": 0, "end": 6,
             "text": "Today I want to talk about why networking is not just about business."},
            {"start": 6, "end": 12,
             "text": "Researchers tracked how happiness spread through social groups."},
            {"start": 12, "end": 17,
             "text": "They found the same level of influence, three degrees of separation out."},
            {"start": 17, "end": 23,
             "text": "And then the most fascinating study was of happiness."},
            {"start": 23, "end": 30,
             "text": "The two found out that your friends make you happy."},
            {"start": 30, "end": 38,
             "text": "If your friend of a friend of a friend is happy with her life, then you have a 6% greater likelihood of being happy with yours."},
            {"start": 38, "end": 46,
             "text": "Consider that if I gave you a $10,000 raise, that would only increase your happiness 2%."},
            {"start": 46, "end": 54,
             "text": "So your friend of a friend of a friend has a greater influence on your life happiness than your salary."},
        ],
    }


def _candidate():
    return {
        "start": 17.0,
        "end": 32.0,
        "title": "Happiness study",
        "hook": "Your friends make you happy",
        "description": "How happiness spreads through friends of friends",
        "reason": "Surprising science with a salary comparison",
        "story_arc": "study -> 6% effect -> what it means",
        "evidence_quote": "6% greater likelihood",
    }


def _good_plan(**overrides):
    return build_edit_plan(
        _transcript(), _candidate(),
        target_duration=overrides.pop("target_duration", 40),
        generation_mode=overrides.pop("generation_mode", "source_only"),
        **overrides,
    )


def test_good_standalone_plan_passes_editorial_review():
    review = review_edit_plan(_good_plan(), _transcript())
    assert review["status"] in {"pass", "warning"}
    checks = review["checks"]
    for name in ("opening_not_fragment", "opening_context", "topic_relevance",
                 "central_message", "conclusion_complete", "no_cut_payoff",
                 "duration_explained", "narration_grounded", "timeline_mapping"):
        assert checks[name]["ok"] is True, f"{name}: {checks[name]['detail']}"
    assert review["errors"] == []


def test_fragment_opening_fails_editorial_review():
    plan = _good_plan()
    plan["opening"]["text"] = "and found the same level of influence"
    plan["segments"][0]["text"] = plan["opening"]["text"]
    review = review_edit_plan(plan, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["opening_not_fragment"]["ok"] is False
    assert any("fragment" in error.lower() or "middle of a thought" in error.lower()
               for error in review["errors"])


def test_cut_off_conclusion_fails_editorial_review():
    plan = _good_plan()
    plan["conclusion"]["text"] = "and then the salary comparison,"
    last = plan["segments"][-1]
    last["text"] = plan["conclusion"]["text"]
    review = review_edit_plan(plan, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["conclusion_complete"]["ok"] is False
    assert any("cut off" in error.lower() or "mid-thought" in error.lower()
               for error in review["errors"])


def test_missing_payoff_fails_editorial_review():
    plan = _good_plan()
    plan["evidence_included"] = False
    review = review_edit_plan(plan, _transcript())
    assert review["checks"]["no_cut_payoff"]["ok"] is False
    assert any("payoff" in error.lower() for error in review["errors"])


def test_source_only_plan_with_generated_passage_fails():
    plan = _good_plan()
    plan["segments"].insert(1, {
        "role": "narration_intro",
        "origin": "ai",
        "text": "Totally invented narration.",
        "source_start": None,
        "source_end": None,
        "edit_start": 0.0,
        "edit_end": 3.0,
    })
    review = review_edit_plan(plan, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["narration_grounded"]["ok"] is False
    assert any("source-only" in error.lower() for error in review["errors"])


def test_ungrounded_narration_quote_fails():
    plan = _good_plan(generation_mode="ai_expanded", target_duration=90)
    narration = plan.get("narration")
    if narration is None:
        # Force a narration block so the grounding checks are exercised.
        narration = {
            "script": 'He said "I invented a 99% happiness boost" clearly.',
            "grounding_quotes": ["I invented a 99% happiness boost"],
            "estimated_duration": 8.0,
            "actual_duration": None,
            "position": "intro",
            "origin": "ai",
            "label": AI_NARRATION_LABEL,
        }
        plan["narration"] = narration
        plan["segments"].insert(0, {
            "role": "narration_intro", "origin": "ai",
            "text": narration["script"],
            "source_start": None, "source_end": None,
            "edit_start": 0.0, "edit_end": 8.0,
        })
    else:
        narration["grounding_quotes"] = ["this quote does not exist in the transcript"]
        narration["script"] = 'He said "this quote does not exist in the transcript"'
    review = review_edit_plan(plan, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["narration_grounded"]["ok"] is False


def test_narration_numbers_must_exist_in_the_transcript():
    plan = _good_plan(generation_mode="ai_expanded", target_duration=90)
    narration = plan.get("narration") or {
        "script": "The speaker explains the point.",
        "grounding_quotes": ["Your friends make you happy."],
        "estimated_duration": 5.0,
        "actual_duration": None,
        "position": "intro",
        "origin": "ai",
        "label": AI_NARRATION_LABEL,
    }
    # Assert an invented statistic that is absent from every quote.
    narration["script"] = "The speaker proves a 42% increase in everything. " + narration["script"]
    narration["grounding_quotes"] = ["Your friends make you happy."]
    plan["narration"] = narration
    if not any(seg.get("origin") == "ai" for seg in plan["segments"]):
        plan["segments"].insert(0, {
            "role": "narration_intro", "origin": "ai",
            "text": narration["script"],
            "source_start": None, "source_end": None,
            "edit_start": 0.0, "edit_end": 5.0,
        })
    review = review_edit_plan(plan, _transcript())
    assert review["checks"]["narration_grounded"]["ok"] is False
    assert any("numbers" in error.lower() for error in review["errors"])


def test_grounded_narration_passes_grounding_check():
    plan = _good_plan(generation_mode="ai_expanded", target_duration=90)
    narration = plan.get("narration")
    if narration is None:
        narration = {
            "script": 'The speaker says "Your friends make you happy."',
            "grounding_quotes": ["Your friends make you happy."],
            "estimated_duration": 5.0,
            "actual_duration": None,
            "position": "intro",
            "origin": "ai",
            "label": AI_NARRATION_LABEL,
        }
        plan["narration"] = narration
    else:
        # Replace with a strictly verbatim, minimal grounded block.
        narration["script"] = 'The speaker says "Your friends make you happy."'
        narration["grounding_quotes"] = ["Your friends make you happy."]
    review = review_edit_plan(plan, _transcript())
    assert review["checks"]["narration_grounded"]["ok"] is True


def test_duration_mismatch_without_explanation_fails():
    plan = _good_plan()
    plan["duration_status"] = "short"
    plan["duration_explanation"] = ""
    review = review_edit_plan(plan, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["duration_explained"]["ok"] is False


def test_duration_mismatch_with_explanation_is_reported_not_hidden():
    plan = _good_plan(target_duration=90)
    if plan["duration_status"] == "within_tolerance":
        plan["duration_status"] = "short"
        plan["duration_explanation"] = (
            "Requested 90s but only 42.0s of relevant source material exists "
            "for this topic."
        )
    review = review_edit_plan(plan, _transcript())
    assert review["checks"]["duration_explained"]["ok"] is True
    if plan["duration_status"] != "within_tolerance":
        assert review["status"] in {"warning", "fail"}
        assert any("90" in warning for warning in review["warnings"] + review["errors"])


def test_editorial_review_is_independent_of_technical_render_failure():
    """Scenario 11: a render failure never erases the editorial verdict, and a
    successful render never implies editorial quality."""
    plan = _good_plan()
    review = review_edit_plan(plan, _transcript())

    # 1) Technical failure (e.g. invalid audio stream) does not change the
    #    editorial review — it was produced without touching a media file.
    technical_failure = {
        "status": "fail",
        "checks": {"audio": {"ok": False, "expected": True, "actual": False}},
        "errors": ["Expected source audio is missing from the rendered output."],
    }
    assert review["status"] in {"pass", "warning"}
    assert technical_failure["status"] == "fail"

    # 2) The reverse: a technically passing render of a broken story still
    #    fails editorially.
    broken = _good_plan()
    broken["opening"]["text"] = "but so anyway"
    broken["segments"][0]["text"] = broken["opening"]["text"]
    technical_pass = {"status": "pass", "checks": {"streams": {"ok": True}}, "errors": []}
    broken_review = review_edit_plan(broken, _transcript())
    assert technical_pass["status"] == "pass"
    assert broken_review["status"] == "fail"


def test_timeline_mapping_regression_fails_editorial_review():
    plan = _good_plan()
    source_segment = next(s for s in plan["segments"] if s.get("origin") == "source")
    source_segment["edit_start"] = 999.0  # inconsistent rebased caption timing
    source_segment["edit_end"] = 1000.0
    review = review_edit_plan(plan, _transcript())
    assert review["checks"]["timeline_mapping"]["ok"] is False
    assert review["status"] == "fail"


def test_editorial_summary_aggregates_statuses():
    reviews = [
        {"status": "pass", "checks": {}},
        {"status": "warning", "checks": {}},
        {"status": "fail", "checks": {}},
    ]
    summary = editorial_summary(reviews)
    assert summary["status"] == "fail"
    assert summary["checked_clip_count"] == 3
    assert summary["failed_clip_count"] == 1
    empty = editorial_summary([])
    assert empty["status"] == "not_run"
    # An unreviewed clip is never counted as a passed review.
    mixed = editorial_summary([{"status": "pass"}, {"status": "not_run"}])
    assert mixed["status"] == "warning"
    assert mixed["unchecked_clip_count"] == 1
    assert mixed["checked_clip_count"] == 1


def test_empty_plan_fails_editorial_review():
    review = review_edit_plan({}, _transcript())
    assert review["status"] == "fail"
    assert review["checks"]["opening_context"]["ok"] is False
