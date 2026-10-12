import json
import sys
import types
from unittest.mock import Mock

import pytest

from autoclipper.analyze import rank_and_diversify_candidates, review_clip_candidates


def _install_google(monkeypatch):
    fake_genai = types.SimpleNamespace(Client=Mock(return_value=object()))
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)


def test_independent_review_derives_boundary_changes_from_exact_segment_quotes(monkeypatch):
    import autoclipper.analyze as analyze_mod

    _install_google(monkeypatch)
    transcript = {
        "duration": 20,
        "segments": [
            {"start": 2.0, "end": 4.0, "text": "The repair works."},
            {"start": 4.0, "end": 6.0, "text": "Now watch the clean finish."},
            {"start": 6.0, "end": 8.0, "text": "And the result is clean."},
        ],
    }
    candidate = {
        "start": 2.5,
        "end": 7.5,
        "title": "A clean repair",
        "hook": "It works",
        "candidate_score": 70,
    }
    review_payload = {
        "reviews": [{
            "candidate_index": 0,
            "decision": "accept",
            "score": 91,
            "reason": "Complete demonstration and payoff.",
            "transcript_evidence": "The repair works.",
            "visual_evidence": "The repaired item is visible in the sampled frame.",
            "start_quote": "The repair works.",
            "end_quote": "And the result is clean.",
            # The reviewer must not control endpoint values directly.
            "start": 0,
            "end": 20,
        }]
    }
    monkeypatch.setattr(
        analyze_mod.config,
        "gemini_generate",
        lambda *_args, **_kwargs: types.SimpleNamespace(text=json.dumps(review_payload)),
    )

    result = review_clip_candidates(
        transcript,
        [candidate],
        api_key="test-key",
        total_duration=20,
        min_clip=5,
        max_clip=10,
    )

    reviewed = result["candidates"][0]
    assert (reviewed["start"], reviewed["end"]) == (2.0, 8.0)
    assert reviewed["review_score"] == 91
    assert reviewed["timestamp_revision"]["from"] == {"start": 2.5, "end": 7.5}
    assert reviewed["timestamp_revision"]["to"] == {"start": 2.0, "end": 8.0}


def test_unverifiable_review_quote_is_rejected_instead_of_trusted(monkeypatch):
    import autoclipper.analyze as analyze_mod

    _install_google(monkeypatch)
    monkeypatch.setattr(
        analyze_mod.config,
        "gemini_generate",
        lambda *_args, **_kwargs: types.SimpleNamespace(text=json.dumps({
            "reviews": [{
                "candidate_index": 0,
                "decision": "accept",
                "score": 95,
                "transcript_evidence": "invented words not in the source",
                "start_quote": "invented words",
                "end_quote": "invented words",
            }]
        })),
    )
    transcript = {"segments": [{"start": 0, "end": 10, "text": "Actual spoken words."}]}
    with pytest.raises(ValueError, match="verifiable transcript quote"):
        review_clip_candidates(
            transcript,
            [{"start": 0, "end": 10, "title": "Candidate"}],
            api_key="test-key",
            total_duration=10,
        )


def test_review_rejects_weak_candidates_but_keeps_flagged_user_selection(monkeypatch):
    import autoclipper.analyze as analyze_mod

    _install_google(monkeypatch)
    transcript = {
        "segments": [
            {"start": 0, "end": 5, "text": "One distinct statement."},
            {"start": 10, "end": 15, "text": "A second distinct statement."},
        ]
    }
    candidates = [
        {"start": 0, "end": 5, "title": "Auto candidate"},
        {"start": 10, "end": 15, "title": "Chosen by user", "_user_selected": True},
    ]
    reviews = {
        "reviews": [
            {
                "candidate_index": 0, "decision": "reject", "score": 20,
                "transcript_evidence": "One distinct statement.",
            },
            {
                "candidate_index": 1, "decision": "reject", "score": 30,
                "transcript_evidence": "A second distinct statement.",
            },
        ]
    }
    monkeypatch.setattr(
        analyze_mod.config,
        "gemini_generate",
        lambda *_args, **_kwargs: types.SimpleNamespace(text=json.dumps(reviews)),
    )

    result = review_clip_candidates(transcript, candidates, api_key="test-key")
    assert result["rejected_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["review_status"] == "flagged_user_selected"


def test_ranker_uses_review_score_and_removes_overlapping_or_semantic_duplicates():
    candidates = [
        {
            "start": 0, "end": 10, "title": "Strong repair tutorial",
            "hook": "Fix it now", "candidate_score": 65, "review_score": 95,
        },
        {
            "start": 1, "end": 10, "title": "Duplicate repair tutorial",
            "hook": "Fix it now", "candidate_score": 98, "review_score": 40,
        },
        {
            "start": 20, "end": 30, "title": "Unexpected interview confession",
            "hook": "I did not expect this", "candidate_score": 75, "review_score": 80,
        },
    ]

    selected = rank_and_diversify_candidates(candidates, target_count=2)
    assert [candidate["title"] for candidate in selected] == [
        "Strong repair tutorial", "Unexpected interview confession",
    ]
    assert selected[0]["candidate_rank"] == 1
    assert selected[0]["quality_score"] > selected[1]["quality_score"]
