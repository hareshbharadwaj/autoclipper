"""Unit tests for the analyze-level grounding behaviors.

Covers: review failure classification, visual-honesty enforcement in the
independent review, evidence-quote verification at generation time, and the
user-selection exemption from the target-duration window.
"""
import json
import sys
import types
from unittest.mock import Mock

import pytest

from autoclipper import grounding
from autoclipper.analyze import (
    CandidateReviewError,
    CandidateReviewTimeoutError,
    _classify_review_exception,
    _validate_clips,
    _verify_evidence_quotes,
    find_viral_clips,
    review_clip_candidates,
)


def _install_google(monkeypatch):
    fake_part = types.SimpleNamespace(from_bytes=lambda data, mime_type: Mock())
    fake_genai = types.SimpleNamespace(
        Client=Mock(return_value=object()),
        types=types.SimpleNamespace(Part=fake_part),
    )
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)


TRANSCRIPT = {
    "duration": 40,
    "segments": [
        {"start": 0.0, "end": 10.0, "text": "Rule one, keep the blade facing away."},
        {"start": 10.0, "end": 20.0, "text": "Rule two, hold the handle when walking."},
        {"start": 20.0, "end": 30.0, "text": "That covers the two knife rules."},
    ],
}


class TestReviewFailureClassification:
    def test_timeout_exceptions_map_to_timed_out(self):
        class FakeDeadlineExceeded(Exception):
            pass

        error = _classify_review_exception(FakeDeadlineExceeded("504 Deadline Exceeded"))
        assert isinstance(error, CandidateReviewTimeoutError)
        assert error.category == "timed_out"

    def test_client_errors_map_to_client_error(self):
        class FakeClientError(Exception):
            pass

        try:
            from google.api_core import exceptions as gexc

            exc = gexc.PermissionDenied("403 forbidden")
        except Exception:
            pytest.skip("google.api_core not available")
        error = _classify_review_exception(exc)
        assert isinstance(error, CandidateReviewError)
        assert error.category == "client_error"

    def test_unknown_errors_map_to_unavailable(self):
        error = _classify_review_exception(RuntimeError("connection reset"))
        assert isinstance(error, CandidateReviewError)
        assert error.category == "unavailable"

    def test_candidate_review_error_is_value_error_compatible(self):
        # Existing callers/tests catch ValueError; classification keeps that.
        assert issubclass(CandidateReviewError, ValueError)


class TestReviewVisualHonesty:
    def _run_review(self, monkeypatch, visual_samples):
        _install_google(monkeypatch)
        import autoclipper.analyze as analyze_mod

        payload = {
            "reviews": [{
                "candidate_index": 0,
                "decision": "accept",
                "score": 90,
                "reason": "Coherent and supported.",
                "transcript_evidence": "Rule one, keep the blade facing away.",
                "visual_evidence": "The knife technique is clearly visible in the frame.",
            }]
        }
        monkeypatch.setattr(
            analyze_mod.config, "gemini_generate",
            lambda *a, **k: types.SimpleNamespace(text=json.dumps(payload)),
        )
        result = review_clip_candidates(
            TRANSCRIPT,
            [{"start": 0.0, "end": 20.0, "title": "Rules", "candidate_score": 80}],
            api_key="test-key",
            visual_samples=visual_samples,
            total_duration=40,
        )
        return result["candidates"][0]

    def test_visual_claim_discarded_when_no_frames_attached(self, monkeypatch):
        reviewed = self._run_review(monkeypatch, visual_samples=None)
        assert reviewed["visual_frames_attached"] is False
        assert reviewed["visual_evidence"] == "not verifiable from sampled frames"

    def test_visual_claim_kept_when_frames_were_attached(self, monkeypatch):
        frames = [{
            "timestamp": 5.0,
            "data": b"\xff\xd8fakejpeg",
            "mime_type": "image/jpeg",
            "candidate_index": 0,
        }]
        reviewed = self._run_review(monkeypatch, visual_samples=frames)
        assert reviewed["visual_frames_attached"] is True
        assert "clearly visible" in reviewed["visual_evidence"]

    def test_review_timeout_is_raised_with_category(self, monkeypatch):
        _install_google(monkeypatch)
        import autoclipper.analyze as analyze_mod

        def raise_timeout(*args, **kwargs):
            raise CandidateReviewTimeoutError("provider deadline exceeded")

        monkeypatch.setattr(analyze_mod.config, "gemini_generate", raise_timeout)
        with pytest.raises(CandidateReviewTimeoutError) as err:
            review_clip_candidates(
                TRANSCRIPT,
                [{"start": 0.0, "end": 20.0, "title": "Rules"}],
                api_key="test-key",
            )
        assert err.value.category == "timed_out"

    def test_malformed_review_payload_is_classified(self, monkeypatch):
        _install_google(monkeypatch)
        import autoclipper.analyze as analyze_mod

        monkeypatch.setattr(
            analyze_mod.config, "gemini_generate",
            lambda *a, **k: types.SimpleNamespace(text="this is not json at all"),
        )
        with pytest.raises(CandidateReviewError) as err:
            review_clip_candidates(
                TRANSCRIPT,
                [{"start": 0.0, "end": 20.0, "title": "Rules"}],
                api_key="test-key",
            )
        assert err.value.category == "malformed"


class TestEvidenceQuoteVerification:
    def test_fabricated_quote_drops_generated_candidate(self):
        clips = [
            {
                "start": 0.0, "end": 20.0, "title": "Good",
                "evidence_quote": "Rule one, keep the blade facing away.",
            },
            {
                "start": 0.0, "end": 20.0, "title": "Fabricated",
                "evidence_quote": "Knives should always be thrown, never stored",
            },
        ]
        verified = _verify_evidence_quotes(TRANSCRIPT, clips)
        assert [c["title"] for c in verified] == ["Good"]
        assert verified[0]["evidence_quote_verified"] is True

    def test_missing_quote_is_kept_but_marked_unverified(self):
        verified = _verify_evidence_quotes(
            TRANSCRIPT, [{"start": 0.0, "end": 20.0, "title": "No quote"}]
        )
        assert len(verified) == 1
        assert verified[0]["evidence_quote_verified"] is False

    def test_find_viral_clips_verifies_quotes_end_to_end(self, monkeypatch):
        _install_google(monkeypatch)
        import autoclipper.analyze as analyze_mod

        payload = {
            "clips": [
                {"start": 0, "end": 20, "title": "Real",
                 "evidence_quote": "keep the blade facing away"},
                {"start": 0, "end": 20, "title": "Fake",
                 "evidence_quote": "totally invented supporting words"},
            ]
        }
        monkeypatch.setattr(
            analyze_mod.config, "gemini_generate",
            lambda *a, **k: types.SimpleNamespace(text=json.dumps(payload)),
        )
        clips = find_viral_clips(
            {**TRANSCRIPT, "language": "english"},
            api_key="test-key", clip_count=2, min_clip=10, max_clip=30,
        )
        assert [c["title"] for c in clips] == ["Real"]


class TestUserSelectionExemption:
    def test_short_user_selection_survives_target_window_validation(self):
        # A 26s user selection with a 60s target window (57-63s) must survive.
        candidate = {
            "start": 91.0, "end": 117.0, "title": "Chosen angle",
            "_user_selected": True,
        }
        validated = _validate_clips(
            [candidate], total_duration=604.0,
            min_clip=57.0, max_clip=63.0,
            exempt_user_selected=True,
        )
        assert len(validated) == 1
        assert validated[0]["_user_selected"] is True
        assert (validated[0]["start"], validated[0]["end"]) == (91.0, 117.0)

    def test_generated_candidates_still_respect_the_window(self):
        candidate = {"start": 91.0, "end": 117.0, "title": "Generated"}
        validated = _validate_clips(
            [candidate], total_duration=604.0,
            min_clip=57.0, max_clip=63.0,
            exempt_user_selected=True,
        )
        assert validated == []


class TestPipelineExtensionHelper:
    def test_extension_stops_at_off_topic_segments(self):
        from autoclipper.pipeline import _extend_selection_to_target

        transcript = {
            "duration": 100,
            "segments": [
                {"start": 0.0, "end": 10.0,
                 "text": "Keep the knife flat on the board after use."},
                {"start": 10.0, "end": 20.0,
                 "text": "Hold the knife low while walking with the knife."},
                {"start": 20.0, "end": 30.0,
                 "text": "Now the weather forecast calls for heavy rain."},
            ],
        }
        candidate = {"start": 0.0, "end": 10.0, "title": "Knife care"}
        extended = _extend_selection_to_target(candidate, transcript, 25.0, 100.0)
        assert extended["end"] == 20.0
        assert extended["start"] == 0.0
        assert "extension_note" in extended

    def test_extension_never_exceeds_target_plus_tolerance(self):
        from autoclipper.pipeline import _extend_selection_to_target

        transcript = {
            "duration": 200,
            "segments": [
                {"start": float(i * 10), "end": float(i * 10 + 10),
                 "text": "Knife safety rule about the knife blade and the board."}
                for i in range(20)
            ],
        }
        candidate = {"start": 0.0, "end": 10.0, "title": "Rules"}
        extended = _extend_selection_to_target(candidate, transcript, 30.0, 200.0)
        assert extended["end"] - extended["start"] <= 33.0

    def test_no_extension_when_already_long_enough(self):
        from autoclipper.pipeline import _extend_selection_to_target

        transcript = {"duration": 100, "segments": [
            {"start": 0.0, "end": 40.0, "text": "Long knife safety explanation."},
        ]}
        candidate = {"start": 0.0, "end": 40.0, "title": "Rules"}
        assert _extend_selection_to_target(candidate, transcript, 30.0, 100.0) is candidate
