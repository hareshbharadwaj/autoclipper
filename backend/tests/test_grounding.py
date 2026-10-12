"""Unit tests for deterministic source-grounding and factual validation."""
import pytest

from autoclipper import grounding


KNIFE_TRANSCRIPT = {
    "duration": 60,
    "text": "",
    "segments": [
        {"start": 0.0, "end": 6.0, "text": "Welcome back to the kitchen basics series."},
        {"start": 6.0, "end": 14.0, "text": "Today we talk about knife safety."},
        {"start": 14.0, "end": 24.0,
         "text": "Rule one, always place the knife flat on the board after use."},
        {"start": 24.0, "end": 34.0,
         "text": "Rule two, when walking hold the knife pointing down at your side."},
        {"start": 34.0, "end": 44.0,
         "text": "Rule three, keep the blade facing away when it rests on the bench."},
        {"start": 44.0, "end": 52.0,
         "text": "And never leave a knife sitting in a sink full of soapy water."},
        {"start": 52.0, "end": 60.0,
         "text": "Next week we sharpen our favorite chef knife."},
    ],
    "words": [],
}
KNIFE_TRANSCRIPT["text"] = " ".join(s["text"] for s in KNIFE_TRANSCRIPT["segments"])


class TestVerifyQuote:
    def test_exact_quote_is_verified_and_located(self):
        result = grounding.verify_quote(
            KNIFE_TRANSCRIPT, "Rule one, always place the knife flat", 10.0, 50.0
        )
        assert result["verified"] is True
        assert result["in_span"] is True
        assert result["location"] == (14.0, 24.0)

    def test_invented_quote_is_not_verified(self):
        result = grounding.verify_quote(
            KNIFE_TRANSCRIPT, "Never touch a falling knife", 10.0, 50.0
        )
        assert result["verified"] is False

    def test_quote_outside_span_is_flagged(self):
        result = grounding.verify_quote(
            KNIFE_TRANSCRIPT, "we sharpen our favorite chef knife", 6.0, 44.0
        )
        assert result["verified"] is True
        assert result["in_span"] is False


class TestBoundaryChecks:
    def test_aligned_boundaries_pass(self):
        alignment = grounding.boundary_alignment(KNIFE_TRANSCRIPT, 14.0, 44.0)
        assert alignment["start_aligned"] is True
        assert alignment["end_aligned"] is True
        assert grounding.boundary_context_risks(KNIFE_TRANSCRIPT, 14.0, 44.0) == []

    def test_mid_sentence_start_is_detected(self):
        alignment = grounding.boundary_alignment(KNIFE_TRANSCRIPT, 18.0, 44.0)
        assert alignment["start_aligned"] is False

    def test_cutting_after_do_not_is_critical_negation(self):
        transcript = {
            "segments": [
                {"start": 0.0, "end": 8.0,
                 "text": "Do not walk around with a raised blade."},
                {"start": 8.0, "end": 16.0,
                 "text": "Walk with the knife pointing down."},
            ],
        }
        # Clip starts mid-sentence, after the negation.
        risks = grounding.boundary_context_risks(transcript, 3.0, 16.0)
        assert any(
            risk["type"] == "negation_cut_at_start" and risk["severity"] == "critical"
            for risk in risks
        )

    def test_ending_inside_safety_statement_is_warned(self):
        # Ending at 51 cuts the safety sentence that runs 44-52.
        risks = grounding.boundary_context_risks(KNIFE_TRANSCRIPT, 14.0, 51.0)
        assert any(
            risk["type"] in ("qualifier_cut_at_end", "safety_statement_cut_at_end")
            for risk in risks
        )

    def test_ending_just_before_later_warning_is_warned(self):
        risks = grounding.boundary_context_risks(KNIFE_TRANSCRIPT, 14.0, 43.0)
        assert any(risk["type"] == "warning_cut_at_end" for risk in risks)

    def test_qualifier_tail_cut_is_warned(self):
        transcript = {
            "segments": [
                {"start": 0.0, "end": 10.0,
                 "text": "Carry it at your side, but only when the blade is guarded."},
            ],
        }
        risks = grounding.boundary_context_risks(transcript, 0.0, 5.0)
        assert any(risk["type"] == "qualifier_cut_at_end" for risk in risks)


class TestGeneratedTextGrounding:
    def test_supported_title_passes(self):
        result = grounding.check_generated_text(
            "Three knife rules: placement, walking, blade direction",
            KNIFE_TRANSCRIPT, 6.0, 52.0,
        )
        assert result["status"] in ("passed", "warning")
        assert result["count_claim"]["supported"] is True

    def test_title_promising_three_rules_with_one_rule_span_fails(self):
        result = grounding.check_generated_text(
            "3 knife rules you must know", KNIFE_TRANSCRIPT, 14.0, 24.0
        )
        assert result["status"] == "failed"
        assert result["count_claim"]["claimed"] == 3
        assert result["count_claim"]["supported"] is False

    def test_fabricated_quotation_fails(self):
        result = grounding.check_generated_text(
            'He said "never buy cheap knives" in the video',
            KNIFE_TRANSCRIPT, 6.0, 52.0,
        )
        assert result["status"] == "failed"
        assert result["fabricated_quotes"] == ["never buy cheap knives"]

    def test_unsupported_claim_is_at_least_warned(self):
        result = grounding.check_generated_text(
            "Sharpening stones compared and ranked", KNIFE_TRANSCRIPT, 6.0, 44.0
        )
        assert result["status"] in ("warning", "failed")


class TestGroundCandidate:
    def test_clean_user_direction_candidate_passes(self):
        candidate = {
            "start": 6.0, "end": 52.0, "title": "Knife safety rules",
            "hook": "Knife safety basics",
            "evidence_quote": "Rule one, always place the knife flat",
        }
        record = grounding.ground_candidate(
            candidate, KNIFE_TRANSCRIPT,
            user_direction="knife safety rules placement walking orientation",
            review_state="reviewed", frames_attached=True,
        )
        assert record["status"] == "passed"
        assert record["checks"]["visual_verification"] == "frame_sampled_model_review"
        assert record["checks"]["topic_relevance"] is not None

    def test_transcript_only_path_never_claims_visual_verification(self):
        candidate = {"start": 6.0, "end": 44.0, "title": "Knife safety rules"}
        record = grounding.ground_candidate(
            candidate, KNIFE_TRANSCRIPT, review_state="fallback_unreviewed",
            frames_attached=False,
        )
        assert record["checks"]["visual_verification"] == "transcript_only"

    def test_unrelated_ai_candidate_fails_topic_gate(self):
        candidate = {
            "start": 52.0, "end": 60.0,
            "title": "Sharpening stones ranked",
            "description": "A full comparison of sharpening stones.",
        }
        record = grounding.ground_candidate(
            candidate, KNIFE_TRANSCRIPT,
            user_direction="knife safety rules placement walking blade orientation",
        )
        assert record["status"] == "failed"
        assert any("unrelated" in message for message in record["critical"])

    def test_invented_evidence_quote_fails_ai_candidate(self):
        candidate = {
            "start": 6.0, "end": 44.0, "title": "Knife safety",
            "evidence_quote": "Knives are dangerous toys for children",
        }
        record = grounding.ground_candidate(candidate, KNIFE_TRANSCRIPT)
        assert record["status"] == "failed"

    def test_missing_quote_is_unverified_not_passed_silently(self):
        candidate = {"start": 6.0, "end": 44.0, "title": "Knife safety"}
        record = grounding.ground_candidate(
            candidate, KNIFE_TRANSCRIPT, review_state="fallback_unreviewed"
        )
        assert any("unverified" in w for w in record["warnings"])

    def test_negation_cut_rejects_generated_candidate(self):
        transcript = {
            "segments": [
                {"start": 0.0, "end": 8.0,
                 "text": "Do not walk around with a raised blade."},
                {"start": 8.0, "end": 16.0,
                 "text": "Keep the point down beside your leg."},
            ],
        }
        candidate = {"start": 3.0, "end": 16.0, "title": "Carry the knife right"}
        record = grounding.ground_candidate(candidate, transcript)
        assert record["status"] == "failed"
        assert any("negation" in message for message in record["critical"])

    def test_user_selected_criticals_downgrade_to_warnings(self):
        candidate = {
            "start": 3.0, "end": 16.0, "title": "Carry the knife right",
            "_user_selected": True,
            "evidence_quote": "totally invented quote line",
        }
        transcript = {
            "segments": [
                {"start": 0.0, "end": 8.0,
                 "text": "Do not walk around with a raised blade."},
                {"start": 8.0, "end": 16.0,
                 "text": "Keep the point down beside your leg."},
            ],
        }
        record = grounding.ground_candidate(candidate, transcript)
        assert record["status"] == "warning"
        assert record["critical"] == []
        assert record["warnings"]


class TestTranscriptUncertainty:
    def test_missing_probability_is_not_invented(self):
        result = grounding.transcript_uncertainty(KNIFE_TRANSCRIPT, 0, 60)
        assert result["provider_exposes_confidence"] is False
        assert result["uncertain_words"] == []

    def test_low_probability_words_are_flagged(self):
        transcript = {
            "words": [
                {"word": "knife", "start": 0.0, "end": 0.4, "probability": 0.98},
                {"word": "gnife", "start": 0.4, "end": 0.9, "probability": 0.31},
            ],
        }
        result = grounding.transcript_uncertainty(transcript, 0, 2)
        assert result["provider_exposes_confidence"] is True
        assert result["uncertain_words"] == ["gnife"]
