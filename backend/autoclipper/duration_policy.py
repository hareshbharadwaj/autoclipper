"""The single source of truth for what a requested clip duration means.

Every stage that asks "is this clip the right length?" — candidate selection,
edit planning, render QA, job status and the UI — resolves its answer through
this module, so the numbers can never disagree with each other.

Vocabulary
----------
``target_seconds``
    What the user asked for. A preference, not a promise.
``minimum_acceptable`` / ``maximum_acceptable``
    The band inside which a rendered clip counts as meeting the request.
``shorter_allowed``
    Whether a shorter clip may be *delivered at all*. It is ``True`` by
    default because a complete short story beats padding with unrelated
    footage — but a shorter result is never silently treated as a full-length
    success: it produces a :data:`STATUS_SHORTFALL` verdict, a per-clip QA
    status of ``shortfall``, and a user-facing warning.
``hard_max_seconds``
    A product rule (2 minutes) that no request or environment variable can
    raise.

The distinction the product depends on is that "within tolerance" and
"delivered, but short" are *different outcomes*. Collapsing them is what let a
26.6s clip be reported as a successful 60s clip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import config

# Fraction of the requested duration a clip may miss by before the result is
# reported as a shortfall rather than accepted.
DEFAULT_TOLERANCE_RATIO = 0.15

# Never demand a tighter band than this, so short targets are not judged in
# fractions of a second.
MIN_TOLERANCE_SECONDS = 2.0

STATUS_WITHIN_TOLERANCE = "within_tolerance"
STATUS_SHORTFALL = "shortfall"
STATUS_OVERLONG = "overlong"
STATUS_UNKNOWN = "unknown"

# Render-QA statuses derived from a verdict. ``shortfall`` is deliberately not
# ``pass`` and not ``fail``: the clip is usable, but it did not meet the ask.
QA_PASS = "pass"
QA_WARNING = "warning"
QA_SHORTFALL = "shortfall"
QA_FAIL = "fail"


@dataclass(frozen=True)
class DurationVerdict:
    """How one actual duration compares against one requested duration."""

    status: str
    target_seconds: float | None
    actual_seconds: float | None
    delta_seconds: float
    minimum_acceptable: float | None
    maximum_acceptable: float | None
    message: str = ""

    @property
    def within_tolerance(self) -> bool:
        return self.status == STATUS_WITHIN_TOLERANCE

    @property
    def is_shortfall(self) -> bool:
        return self.status == STATUS_SHORTFALL

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "target_seconds": self.target_seconds,
            "actual_seconds": self.actual_seconds,
            "delta_seconds": round(self.delta_seconds, 2),
            "minimum_acceptable": (
                round(self.minimum_acceptable, 2)
                if self.minimum_acceptable is not None else None
            ),
            "maximum_acceptable": (
                round(self.maximum_acceptable, 2)
                if self.maximum_acceptable is not None else None
            ),
            "message": self.message,
        }


@dataclass(frozen=True)
class DurationPolicy:
    """The resolved duration contract for one clip request.

    Build it once per job with :meth:`from_request` and pass the *same*
    instance to candidate selection, the edit planner, render QA and the job
    summary.
    """

    target_seconds: float | None = None
    tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO
    shorter_allowed: bool = True
    hard_max_seconds: float = float(config.HARD_MAX_CLIP_SECONDS)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_request(
        cls,
        target_duration,
        *,
        tolerance_ratio: float = DEFAULT_TOLERANCE_RATIO,
        shorter_allowed: bool = True,
    ) -> "DurationPolicy":
        """Resolve a raw user request into a policy.

        ``None`` means "no duration requested" — legacy min/max mode — and
        every duration check becomes a no-op rather than inventing a target.
        """
        if target_duration is None:
            return cls(
                target_seconds=None,
                tolerance_ratio=tolerance_ratio,
                shorter_allowed=shorter_allowed,
            )
        try:
            value = float(target_duration)
        except (TypeError, ValueError):
            raise ValueError("target_duration must be a number of seconds.")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("target_duration must be a positive number of seconds.")
        return cls(
            target_seconds=min(value, float(config.HARD_MAX_CLIP_SECONDS)),
            tolerance_ratio=tolerance_ratio,
            shorter_allowed=shorter_allowed,
        )

    # -- derived bands -------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.target_seconds is not None

    @property
    def tolerance_seconds(self) -> float | None:
        if self.target_seconds is None:
            return None
        return max(MIN_TOLERANCE_SECONDS, self.target_seconds * self.tolerance_ratio)

    @property
    def minimum_acceptable(self) -> float | None:
        if self.target_seconds is None:
            return None
        return max(0.0, self.target_seconds - self.tolerance_seconds)

    @property
    def maximum_acceptable(self) -> float | None:
        if self.target_seconds is None:
            return None
        return min(self.hard_max_seconds, self.target_seconds + self.tolerance_seconds)

    # -- evaluation ----------------------------------------------------------

    def verdict(self, actual_seconds, *, context: str = "") -> DurationVerdict:
        """Compare one measured duration against this policy."""
        total = self.target_seconds
        if actual_seconds is None:
            return DurationVerdict(
                status=STATUS_UNKNOWN, target_seconds=total, actual_seconds=None,
                delta_seconds=0.0, minimum_acceptable=self.minimum_acceptable,
                maximum_acceptable=self.maximum_acceptable,
                message="The rendered duration could not be measured.",
            )
        try:
            actual = float(actual_seconds)
        except (TypeError, ValueError):
            actual = float("nan")
        if not math.isfinite(actual) or actual <= 0:
            return DurationVerdict(
                status=STATUS_UNKNOWN, target_seconds=total, actual_seconds=None,
                delta_seconds=0.0, minimum_acceptable=self.minimum_acceptable,
                maximum_acceptable=self.maximum_acceptable,
                message="The rendered duration could not be measured.",
            )

        if total is None:
            # No duration was requested: only the hard ceiling applies.
            if actual > self.hard_max_seconds + 0.25:
                return DurationVerdict(
                    status=STATUS_OVERLONG, target_seconds=None, actual_seconds=actual,
                    delta_seconds=actual - self.hard_max_seconds,
                    minimum_acceptable=None, maximum_acceptable=self.hard_max_seconds,
                    message=(
                        f"{context + ': ' if context else ''}the clip is {actual:.1f}s, "
                        f"over the {self.hard_max_seconds:.0f}s hard limit."
                    ),
                )
            return DurationVerdict(
                status=STATUS_WITHIN_TOLERANCE, target_seconds=None,
                actual_seconds=actual, delta_seconds=0.0, minimum_acceptable=None,
                maximum_acceptable=self.hard_max_seconds,
            )

        delta = actual - total
        prefix = f"{context}: " if context else ""
        if actual > self.maximum_acceptable:
            return DurationVerdict(
                status=STATUS_OVERLONG, target_seconds=total, actual_seconds=actual,
                delta_seconds=delta, minimum_acceptable=self.minimum_acceptable,
                maximum_acceptable=self.maximum_acceptable,
                message=(
                    f"{prefix}{actual:.1f}s is {delta:+.1f}s against the "
                    f"{total:.0f}s target (allowed {self.minimum_acceptable:.1f}s–"
                    f"{self.maximum_acceptable:.1f}s)."
                ),
            )
        if actual < self.minimum_acceptable:
            return DurationVerdict(
                status=STATUS_SHORTFALL, target_seconds=total, actual_seconds=actual,
                delta_seconds=delta, minimum_acceptable=self.minimum_acceptable,
                maximum_acceptable=self.maximum_acceptable,
                message=(
                    f"{prefix}{actual:.1f}s delivered against a {total:.0f}s request — "
                    f"{abs(delta):.1f}s short of the minimum acceptable "
                    f"{self.minimum_acceptable:.1f}s. The source did not contain enough "
                    "relevant material to reach the target without padding with "
                    "unrelated footage."
                ),
            )
        return DurationVerdict(
            status=STATUS_WITHIN_TOLERANCE, target_seconds=total, actual_seconds=actual,
            delta_seconds=delta, minimum_acceptable=self.minimum_acceptable,
            maximum_acceptable=self.maximum_acceptable,
        )

    def qa_status(self, actual_seconds, *, context: str = "") -> tuple[str, DurationVerdict]:
        """Render-QA status implied by a measured duration.

        ``shortfall`` is its own status: the clip is playable and honest, but
        it did not meet the request, so the job must say so.
        """
        verdict = self.verdict(actual_seconds, context=context)
        return {
            STATUS_SHORTFALL: QA_SHORTFALL,
            STATUS_OVERLONG: QA_WARNING,
            STATUS_UNKNOWN: QA_WARNING,
            STATUS_WITHIN_TOLERANCE: QA_PASS,
        }[verdict.status], verdict

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "target_seconds": self.target_seconds,
            "tolerance_ratio": self.tolerance_ratio,
            "tolerance_seconds": (
                round(self.tolerance_seconds, 2) if self.tolerance_seconds else None
            ),
            "minimum_acceptable": (
                round(self.minimum_acceptable, 2) if self.minimum_acceptable else None
            ),
            "maximum_acceptable": (
                round(self.maximum_acceptable, 2) if self.maximum_acceptable else None
            ),
            "shorter_allowed": self.shorter_allowed,
            "hard_max_seconds": self.hard_max_seconds,
        }
