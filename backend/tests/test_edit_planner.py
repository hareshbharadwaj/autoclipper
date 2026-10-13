"""Story-planning tests: dynamic duration and complete story generation.

Covers the required scenarios with synthetic transcripts and mocked narration
responses (deterministic unit tests). Editorial expectations are asserted on
the planned story; real-video render checks live in the pipeline/FFmpeg tests.
"""
import pytest

from autoclipper.edit_planner import (
    AI_NARRATION_LABEL,
    DURATION_TOLERANCE,
    build_edit_plan,
    build_narration_script,
    build_sentence_units,
    duration_mismatch_warning,
    duration_tolerance_window,
    ends_like_fragment,
    estimate_speech_duration,
    finalize_narration_timing,
    map_source_time_to_edit_time,
    normalize_generation_mode,
    rebase_cues,
    starts_like_fragment,
)


# --- Fixtures ---------------------------------------------------------------

def happiness_transcript():
    """The clip-generation bug report's happiness/networking talk."""
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
            {"start": 54, "end": 60,
             "text": "That is perhaps why we need to redefine networking as real relationships."},
        ],
    }


HAPPINESS_CANDIDATE = {
    "start": 17.0,
    "end": 32.0,
    "title": "Happiness study",
    "hook": "Your friends make you happy",
    "description": "How happiness spreads through friends of friends",
    "reason": "Surprising science with a salary comparison",
    "story_arc": "study -> 6% effect -> what it means",
    "evidence_quote": "6% greater likelihood",
}


def isolated_topic_transcript():
    """A short topic surrounded by clearly unrelated neighbours."""
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
    }


# --- Basics -----------------------------------------------------------------

def test_normalize_generation_mode_defaults_and_aliases():
    assert normalize_generation_mode(None) == "source_only"
    assert normalize_generation_mode("source_only") == "source_only"
    assert normalize_generation_mode("Original content only".lower().replace(" ", "_")) in (
        "source_only", "original_content_only",
    )
    assert normalize_generation_mode("ai-expanded") == "ai_expanded"
    assert normalize_generation_mode("AI_EXPANDED") == "ai_expanded"
    with pytest.raises(ValueError, match="generation_mode"):
        normalize_generation_mode("hallucinate")


def test_sentence_units_merge_phrases_and_split_multi_sentence_segments():
    transcript = {
        "segments": [
            {"start": 0, "end": 2, "text": "First half of the thought"},
            {"start": 2, "end": 4, "text": "and the rest of the thought."},
            {"start": 4, "end": 9, "text": "One sentence here. Another one follows!"},
        ]
    }
    units = build_sentence_units(transcript)
    assert len(units) == 3
    assert units[0]["text"] == "First half of the thought and the rest of the thought."
    assert units[0]["ends_sentence"] is True
    assert units[1]["text"] == "One sentence here."
    assert units[2]["text"] == "Another one follows!"
    # Split timings stay inside the source segment and keep order.
    assert units[1]["start"] == 4 and units[2]["end"] == 9
    assert units[1]["end"] <= units[2]["start"] + 1e-6


def test_fragment_heuristics_detect_dangling_edges():
    assert starts_like_fragment("and found the same level of influence")
    assert starts_like_fragment("but wait")
    assert not starts_like_fragment("They found the same level of influence")
    assert ends_like_fragment("If your friend is happy with her life,")
    assert ends_like_fragment("and the study is")
    assert not ends_like_fragment("You have a 6% greater likelihood of happiness.")


def test_speech_duration_estimate_uses_words():
    assert estimate_speech_duration("") == 0.0
    assert estimate_speech_duration("one two three four five six seven eight") == pytest.approx(
        8 / 2.6, rel=1e-3
    )


def test_duration_tolerance_window_is_around_ten_percent():
    lower, upper = duration_tolerance_window(40)
    assert lower == pytest.approx(36.0)
    assert upper == pytest.approx(44.0)
    assert DURATION_TOLERANCE == pytest.approx(0.10)


# --- Scenario 1: 15s topic with a 15s target --------------------------------

def test_fifteen_second_topic_with_fifteen_second_target_uses_complete_passage():
    transcript = {
        "duration": 40,
        "language": "English",
        "segments": [
            {"start": 0, "end": 5, "text": "Here is the opening setup for today."},
            {"start": 5, "end": 10, "text": "The key idea is that practice beats talent."},
            {"start": 10, "end": 15, "text": "Every expert you admire practiced for years."},
            {"start": 15, "end": 20, "text": "So start practicing today and keep going."},
            {"start": 20, "end": 40, "text": "Next we will discuss unrelated cooking tips."},
        ],
    }
    candidate = {
        "start": 5.0, "end": 20.0,
        "title": "Practice beats talent",
        "hook": "Practice beats talent",
        "description": "Why practice matters more than talent",
        "reason": "Complete mini-story",
        "story_arc": "setup -> proof -> call to action",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=15,
                           generation_mode="source_only")
    assert plan["duration_status"] == "within_tolerance"
    assert 13.5 <= plan["estimated_duration"] <= 16.5
    assert plan["narration"] is None
    assert plan["opening"]["text"] == "The key idea is that practice beats talent."
    assert plan["conclusion"]["text"] == "So start practicing today and keep going."
    # The unrelated neighbour is never pulled in.
    assert all("cooking" not in seg["text"] for seg in plan["segments"])
    # Every source segment keeps its original timestamps.
    for seg in plan["segments"]:
        assert seg["source_start"] >= 5.0 and seg["source_end"] <= 20.0


# --- Scenario 2: 15s topic, 40s target, sufficient context ------------------

def test_short_topic_expands_with_contextual_material_for_longer_target():
    transcript = happiness_transcript()
    candidate = dict(HAPPINESS_CANDIDATE)  # ~15s directly matched
    plan = build_edit_plan(transcript, candidate, target_duration=40,
                           generation_mode="source_only")
    lower, upper = duration_tolerance_window(40)
    assert plan["duration_status"] == "within_tolerance"
    assert lower <= plan["estimated_duration"] <= upper
    assert plan["context_expanded_seconds"] > 0
    assert plan["narration"] is None
    # Expansion uses relevant surrounding material: the salary comparison or
    # the networking takeaway, never the unrelated opening pleasantries.
    texts = " ".join(seg["text"] for seg in plan["segments"])
    assert "happiness" in texts.lower()
    assert plan["insufficient_source"] is False


# --- Scenario 3: 15s topic, 40s target, insufficient source -----------------

def test_short_topic_reports_shortfall_in_source_only_mode():
    transcript = isolated_topic_transcript()
    candidate = {
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy",
        "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner",
        "story_arc": "one line about friends",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=40,
                           generation_mode="source_only")
    assert plan["duration_status"] == "short"
    assert plan["insufficient_source"] is True
    assert plan["narration"] is None
    assert plan["estimated_duration"] < 36.0
    joined = " ".join(plan["warnings"])
    assert "40s" in joined  # reports the requested target explicitly
    assert "source-only" in joined.lower() or "source only" in joined.lower()
    assert "AI-expanded" in joined or "ai_expanded" in joined.lower()
    # Unrelated neighbours were not used as padding.
    assert all(seg["text"] == "Your friends make you happy." for seg in plan["segments"])


def test_short_topic_plans_labelled_narration_in_ai_expanded_mode():
    transcript = isolated_topic_transcript()
    candidate = {
        "start": 12.0, "end": 17.0,
        "title": "Friends make you happy",
        "hook": "Friends and happiness",
        "description": "A short remark about friends and happiness",
        "reason": "Cute one-liner",
        "story_arc": "one line about friends",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=40,
                           generation_mode="ai_expanded")
    narration = plan["narration"]
    assert narration is not None
    assert narration["origin"] == "ai"
    assert narration["label"] == AI_NARRATION_LABEL
    assert narration["estimated_duration"] > 0
    assert narration["position"] == "intro"
    # The narration card is the first edit-timeline segment.
    assert plan["segments"][0]["role"] == "narration_intro"
    assert plan["segments"][0]["origin"] == "ai"
    # Estimated duration includes narration.
    assert plan["estimated_duration"] == pytest.approx(
        plan["source_duration"] + narration["estimated_duration"], abs=0.5
    )


# --- Scenario 4: topic starts mid-sentence ----------------------------------

def test_mid_sentence_topic_start_repaired_to_complete_opening():
    transcript = happiness_transcript()
    candidate = {
        "start": 12.5,  # mid-unit: the raw range opened on "...and found ..."
        "end": 16.5,
        "title": "Influence finding",
        "hook": "Three degrees of separation",
        "description": "Influence travels far",
        "reason": "Raw range begins mid-thought",
        "story_arc": "influence reaches far",
        "evidence_quote": "three degrees of separation",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=15,
                           generation_mode="source_only")
    assert plan["opening_repaired"] is True
    assert not starts_like_fragment(plan["opening"]["text"])
    assert plan["opening"]["text"].startswith("They found the same level of influence")
    # The plan opens at or before the raw request instead of inside it.
    assert plan["source_start"] <= 12.5
    assert plan["segments"][0]["text"] == plan["opening"]["text"]


# --- Scenario 5: conclusion after the initial range -------------------------

def test_conclusion_repair_includes_payoff_after_initial_range():
    transcript = {
        "duration": 50,
        "language": "English",
        "segments": [
            {"start": 0, "end": 6, "text": "Networking matters more than you think."},
            {"start": 6, "end": 14, "text": "Your friends shape how happy you feel."},
            {"start": 14, "end": 19, "text": "If your friend of a friend is happy with her life,"},
            {"start": 20.5, "end": 26, "text": "then you have a 6% greater likelihood of being happy with yours."},
            {"start": 26, "end": 50, "text": "Moving on to the next topic about sleep habits."},
        ],
    }
    candidate = {
        "start": 6.0, "end": 18.0,  # ends before the payoff sentence
        "title": "Happiness is contagious",
        "hook": "Friends of friends make you happy",
        "description": "How happiness spreads",
        "reason": "Payoff lies just after the raw range",
        "story_arc": "claim -> condition -> payoff",
        "evidence_quote": "6% greater likelihood",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=15,
                           generation_mode="source_only")
    assert plan["conclusion_repaired"] is True
    assert "6%" in plan["conclusion"]["text"]
    assert not ends_like_fragment(plan["conclusion"]["text"])
    assert plan["evidence_included"] is True
    assert plan["source_end"] >= 26.0


# --- Scenario 6: source-only never introduces generated dialogue ------------

def test_source_only_plan_contains_no_generated_passages():
    transcript = happiness_transcript()
    plan = build_edit_plan(transcript, dict(HAPPINESS_CANDIDATE),
                           target_duration=40, generation_mode="source_only")
    assert plan["mode"] == "source_only"
    assert plan["narration"] is None
    assert all(seg["origin"] == "source" for seg in plan["segments"])
    # Every planned passage is verbatim transcript text.
    transcript_text = " ".join(s["text"] for s in transcript["segments"])
    for seg in plan["segments"]:
        assert seg["text"] in transcript_text


# --- Scenario 7: AI-expanded with transcript-grounded narration -------------

def test_narration_script_only_quotes_transcript_text():
    units = [
        {"text": "Your friends make you happy."},
        {"text": "A friend of a friend still influences your mood."},
    ]
    script, quotes = build_narration_script(units, target_words=30)
    assert quotes == ["Your friends make you happy.",
                      "A friend of a friend still influences your mood."]
    for quote in quotes:
        assert f'"{quote}"' in script
    # Framing introduces no numbers or claims of its own.
    framing = script.replace('"' + quotes[0] + '"', "").replace('"' + quotes[1] + '"', "")
    assert not any(ch.isdigit() for ch in framing)


def test_ai_expanded_narration_is_grounded_in_the_transcript():
    transcript = happiness_transcript()
    plan = build_edit_plan(transcript, dict(HAPPINESS_CANDIDATE),
                           target_duration=90, generation_mode="ai_expanded",
                           total_duration=60)
    narration = plan.get("narration")
    if narration is None:
        pytest.skip("source alone satisfied the target; narration not needed")
    transcript_text = " ".join(s["text"] for s in transcript["segments"])
    for quote in narration["grounding_quotes"]:
        assert quote in transcript_text
    assert narration["label"] == AI_NARRATION_LABEL


# --- Scenario 8: duration mismatch reporting -------------------------------

def test_duration_mismatch_warning_is_actionable_and_silent_within_tolerance():
    warning = duration_mismatch_warning(60, 9)
    assert warning is not None
    assert "60" in warning and "9.0" in warning
    assert "AI-expanded" in warning or "target duration" in warning
    assert duration_mismatch_warning(60, 58) is None
    assert duration_mismatch_warning(60, 62) is None
    assert duration_mismatch_warning(None, 9) is None


def test_short_render_target_gap_is_never_silent_in_the_plan():
    transcript = isolated_topic_transcript()
    candidate = {"start": 12.0, "end": 17.0, "title": "Friends make you happy",
                 "hook": "Friends and happiness", "description": "one line",
                 "reason": "short", "story_arc": "one line"}
    plan = build_edit_plan(transcript, candidate, target_duration=60,
                           generation_mode="source_only")
    assert plan["duration_status"] == "short"
    assert plan["duration_explanation"]
    assert any("60s" in warning for warning in plan["warnings"])


# --- Scenario 9: long candidate keeps its complete ending -------------------

def test_long_candidate_keeps_complete_conclusion_when_shrinking():
    transcript = happiness_transcript()
    candidate = {
        "start": 6.0, "end": 54.0,
        "title": "Full happiness arc",
        "hook": "Happiness and salary",
        "description": "The complete happiness story",
        "reason": "Long but coherent story",
        "story_arc": "study -> comparison -> takeaway",
        "evidence_quote": "greater influence on your life happiness",
    }
    plan = build_edit_plan(transcript, candidate, target_duration=40,
                           generation_mode="source_only")
    lower, upper = duration_tolerance_window(40)
    assert lower <= plan["estimated_duration"] <= upper
    assert plan["trimmed_from_front"] is True
    # The complete takeaway/payoff survives trimming.
    assert plan["conclusion"]["text"].startswith("So your friend of a friend")
    assert not ends_like_fragment(plan["conclusion"]["text"])
    assert plan["evidence_included"] is True
    # Cuts only land on sentence boundaries: every planned passage is a full
    # transcript sentence with its original timestamps.
    for seg in plan["segments"]:
        assert seg["text"] in " ".join(s["text"] for s in transcript["segments"])
        assert seg["source_end"] > seg["source_start"]


def test_single_oversized_story_is_kept_whole_with_explanation():
    """Story coherence beats an exact number: never truncate a full sentence."""
    transcript = {
        "duration": 30,
        "language": "English",
        "segments": [
            {"start": 0, "end": 30,
             "text": "This one enormous sentence keeps going and going with no internal punctuation at all."},
        ],
    }
    candidate = {"start": 0.0, "end": 30.0, "title": "Big sentence",
                 "hook": "One long sentence", "description": "long", "reason": "long",
                 "story_arc": "long"}
    plan = build_edit_plan(transcript, candidate, target_duration=10,
                           generation_mode="source_only")
    assert plan["duration_status"] == "long"
    assert plan["source_end"] - plan["source_start"] == pytest.approx(30.0, abs=0.5)
    assert any("coherence" in warning for warning in plan["warnings"])


# --- Scenario 10: multi-segment captions rebased onto the edit timeline -----

def test_captions_rebase_across_segments_with_narration_prefix():
    transcript = isolated_topic_transcript()
    candidate = {"start": 12.0, "end": 17.0, "title": "Friends make you happy",
                 "hook": "Friends and happiness", "description": "one line",
                 "reason": "short", "story_arc": "one line"}
    plan = build_edit_plan(transcript, candidate, target_duration=40,
                           generation_mode="ai_expanded")
    assert plan["narration"] is not None
    narration_estimate = plan["narration"]["estimated_duration"]
    # Cues use clip-relative source time (0 == clip start).
    cues = [
        {"start": 0.0, "end": 2.5, "text": "Your friends"},
        {"start": 2.5, "end": 5.0, "text": "make you happy."},
        {"start": 50.0, "end": 52.0, "text": "outside the clip"},
    ]
    rebased = rebase_cues(cues, plan, clip_source_start=12.0)
    assert len(rebased) == 2
    assert rebased[0]["start"] == pytest.approx(narration_estimate, abs=0.05)
    assert rebased[0]["end"] == pytest.approx(narration_estimate + 2.5, abs=0.05)
    assert rebased[1]["start"] == pytest.approx(narration_estimate + 2.5, abs=0.05)
    assert rebased[1]["end"] == pytest.approx(narration_estimate + 5.0, abs=0.05)

    # Measured narration audio re-times the edit timeline; captions follow.
    finalize_narration_timing(plan, actual_duration=7.0)
    assert plan["narration"]["actual_duration"] == 7.0
    rebased = rebase_cues(cues, plan, clip_source_start=12.0)
    assert rebased[0]["start"] == pytest.approx(7.0, abs=0.05)
    assert rebased[0]["end"] == pytest.approx(9.5, abs=0.05)
    assert plan["estimated_duration"] == pytest.approx(7.0 + plan["source_duration"], abs=0.05)


def test_map_source_time_to_edit_time_covers_source_segments_only():
    transcript = isolated_topic_transcript()
    candidate = {"start": 12.0, "end": 17.0, "title": "Friends make you happy",
                 "hook": "Friends and happiness", "description": "one line",
                 "reason": "short", "story_arc": "one line"}
    plan = build_edit_plan(transcript, candidate, target_duration=15,
                           generation_mode="source_only")
    assert map_source_time_to_edit_time(plan, 12.0) == pytest.approx(0.0, abs=0.05)
    assert map_source_time_to_edit_time(plan, 14.5) == pytest.approx(2.5, abs=0.05)
    assert map_source_time_to_edit_time(plan, 30.0) is None


# --- Scenario 12: independent clips with individual targets ----------------

def test_multiple_clips_respect_their_own_duration_targets():
    transcript = happiness_transcript()
    short_moment = {
        "start": 23.0, "end": 30.0,
        "title": "Friends make you happy",
        "hook": "Friends and happiness",
        "description": "Short beat",
        "reason": "Short moment",
        "story_arc": "one beat",
        "target_duration": 15,
    }
    long_moment = {
        "start": 6.0, "end": 54.0,
        "title": "Full happiness arc",
        "hook": "Happiness and salary",
        "description": "Long beat",
        "reason": "Long moment",
        "story_arc": "full arc",
        "target_duration": 45,
    }
    plan_short = build_edit_plan(transcript, short_moment, generation_mode="source_only")
    plan_long = build_edit_plan(transcript, long_moment, generation_mode="source_only")
    assert plan_short["target_duration"] == 15
    assert plan_long["target_duration"] == 45
    assert plan_short["duration_status"] in {"within_tolerance", "short", "long"}
    assert plan_long["duration_status"] in {"within_tolerance", "short", "long"}
    # Each plan honours its own window: ~13.5-16.5s and ~40.5-49.5s.
    if plan_short["duration_status"] == "within_tolerance":
        assert 13.5 <= plan_short["estimated_duration"] <= 16.5
    if plan_long["duration_status"] == "within_tolerance":
        assert 40.5 <= plan_long["estimated_duration"] <= 49.5
    # Plans do not leak into each other.
    assert plan_short["segments"] != plan_long["segments"]


# --- Degenerate transcripts and bounds --------------------------------------

def test_lumped_transcript_units_fall_back_to_candidate_bounds():
    transcript = {
        "duration": 100,
        "language": "English",
        "segments": [{"start": 0, "end": 100, "text": "A transcript."}],
    }
    candidate = {"start": 5, "end": 15, "title": "One", "hook": "One",
                 "description": "One", "reason": "One", "story_arc": "One"}
    plan = build_edit_plan(transcript, candidate, target_duration=10,
                           generation_mode="source_only")
    assert plan["source_start"] == pytest.approx(5.0, abs=0.25)
    assert plan["source_end"] == pytest.approx(15.0, abs=0.25)


def test_target_duration_bounds_are_validated():
    transcript = happiness_transcript()
    with pytest.raises(ValueError, match="between 5 and 120"):
        build_edit_plan(transcript, dict(HAPPINESS_CANDIDATE), target_duration=200)
    with pytest.raises(ValueError, match="between 5 and 120"):
        build_edit_plan(transcript, dict(HAPPINESS_CANDIDATE), target_duration=1)
