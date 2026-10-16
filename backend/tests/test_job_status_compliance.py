"""The job status must reflect the request, not merely that files exist."""

import pytest

from app import _job_status_for


def _result(*, satisfied, qa_state, failures=None, compliance=None):
    extra = {"satisfied": satisfied}
    if failures is not None:
        extra["failures"] = failures
    return {
        "quality_assurance": {"status": qa_state},
        "request_compliance": {**extra, **(compliance or {})},
    }


def test_a_fully_satisfied_job_is_completed():
    status, message = _job_status_for(_result(satisfied=True, qa_state="pass"))
    assert status == "completed"
    assert "complete" in message.lower()


@pytest.mark.parametrize("qa_state", ["shortfall", "warning", "fail"])
def test_a_short_or_partial_delivery_is_not_reported_as_a_clean_pass(qa_state):
    status, message = _job_status_for(
        _result(
            satisfied=False, qa_state=qa_state,
            failures=["1 of 2 requested individual clip(s) were not delivered."],
        )
    )
    assert status == "completed_with_warnings"
    assert "not fully met" in message
    assert "requested individual clip" in message


def test_a_duration_shortfall_is_described_in_the_job_message():
    status, message = _job_status_for(_result(satisfied=False, qa_state="shortfall"))
    assert status == "completed_with_warnings"
    assert "shorter than the requested duration" in message


def test_the_26_second_against_60_second_case_is_not_a_clean_success():
    """The exact regression: a 60s request delivered as ~26.6s."""
    result = _result(
        satisfied=False,
        qa_state="shortfall",
        failures=[
            "1 clip(s) are shorter than the requested duration: clip_1.mp4 "
            "(26.6s vs 60s target)"
        ],
        compliance={"duration_shortfall_clips": 1, "delivered_individual_clips": 1},
    )
    status, message = _job_status_for(result)
    assert status == "completed_with_warnings"
    assert "26.6s vs 60s target" in message


def test_a_legacy_result_without_compliance_still_reports_qa_state():
    assert _job_status_for({"quality_assurance": {"status": "pass"}})[0] == "completed"
    assert (
        _job_status_for({"quality_assurance": {"status": "warning"}})[0]
        == "completed_with_warnings"
    )
