"""Global candidate selection for a clip-generation job.

The pipeline used to pick clip slots one at a time: a candidate was accepted
because it looked good *in isolation*, with at most a cheap interval-overlap
check against its predecessors. Two consequences showed up in real output:

* the same footage resurfacing under a different title, and
* several clips that were really the same explanation told twice.

This module makes selection *global*. Every candidate in the pool is scored
against the user's actual request **and** against the candidates already
accepted, and a candidate is only accepted when it adds something:

* **relevance** — overlap with the user's prompt/topic vocabulary,
* **quality** — the pipeline's own candidate/review score,
* **duration fit** — how close the moment is to the requested length,
* **duplicate penalty** — how much footage it reuses,
* **repetition penalty** — how similar its wording is to what was accepted.

Selection is greedy and deterministic: highest composite score first, ties
broken by source position. When the pool cannot fill the request, the outcome
reports the shortfall and *why* rather than fabricating candidates.

A second, explicitly-labelled pass may allow limited overlap when a slot
cannot otherwise be filled, because a slightly overlapping clip with a real
story is better than no clip — but those candidates are tagged so QA and the
UI can say so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .duration_policy import DurationPolicy
from .edit_planner import content_words, relevance_score

# Footage reuse tolerated between two clips before the later one is rejected.
DEFAULT_MAX_OVERLAP_RATIO = 0.15
# Limited reuse allowed only by the context pass.
CONTEXT_OVERLAP_RATIO = 0.45
# Wording similarity above which two clips count as the same explanation.
DEFAULT_MAX_TEXT_SIMILARITY = 0.72
# A clip shorter than this is never a usable standalone short.
MIN_USABLE_DURATION = 3.0

_REJECTION_DUPLICATE = "duplicate_footage"
_REJECTION_REPEATED = "repeated_explanation"
_REJECTION_DURATION = "duration_out_of_range"
_REJECTION_INVALID = "invalid_bounds"
_REJECTION_SURPLUS = "request_already_filled"


@dataclass
class SelectionOutcome:
    """The result of one global selection pass."""

    candidates: list = field(default_factory=list)
    requested_count: int = 0
    rejected: list = field(default_factory=list)
    exhausted: bool = False
    reasons: list = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return len(self.candidates)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_count - self.selected_count)

    @property
    def relented_for_context(self) -> int:
        return sum(1 for item in self.candidates if item.get("overlap_allowed_for_context"))

    def summary(self) -> str:
        if not self.shortfall:
            return (
                f"Selected {self.selected_count} distinct clip(s) for the "
                f"{self.requested_count} requested."
            )
        detail = "; ".join(self.reasons) if self.reasons else "the source ran out of distinct material"
        return (
            f"Requested {self.requested_count} individual clip(s) but only "
            f"{self.selected_count} distinct moment(s) passed selection: {detail}."
        )

    def to_dict(self) -> dict:
        return {
            "requested_count": self.requested_count,
            "selected_count": self.selected_count,
            "shortfall": self.shortfall,
            "exhausted": self.exhausted,
            "summary": self.summary(),
            "reasons": list(self.reasons),
            "rejected": list(self.rejected),
            "relented_for_context": self.relented_for_context,
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bounds(candidate) -> tuple[float, float] | None:
    try:
        start = float(candidate.get("start"))
        end = float(candidate.get("end"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
        return None
    return start, end


def _overlap_ratio(left, right) -> float:
    """Fraction of the shorter clip that both clips share."""
    left_bounds, right_bounds = _bounds(left), _bounds(right)
    if not left_bounds or not right_bounds:
        return 0.0
    l_start, l_end = left_bounds
    r_start, r_end = right_bounds
    overlap = max(0.0, min(l_end, r_end) - max(l_start, r_start))
    shorter = min(l_end - l_start, r_end - r_start)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _evidence_text(candidate) -> str:
    """All text that describes what the clip shows/says."""
    parts = [
        candidate.get("title"),
        candidate.get("hook"),
        candidate.get("description"),
        candidate.get("reason"),
        candidate.get("story_arc"),
        candidate.get("transcript_excerpt"),
        candidate.get("visual_summary"),
        candidate.get("topic"),
    ]
    return " ".join(str(part) for part in parts if part)


def text_similarity(left, right) -> float:
    """Jaccard similarity of the two candidates' content words.

    Catches the "same explanation, different title" case that interval overlap
    alone cannot see: a re-edit of the same passage usually keeps its nouns.
    """
    left_words = content_words(_evidence_text(left))
    right_words = content_words(_evidence_text(right))
    if not left_words or not right_words:
        return 0.0
    union = left_words | right_words
    if not union:
        return 0.0
    return len(left_words & right_words) / len(union)


def _quality_score(candidate) -> float:
    """Blend the generator score with the independent review score."""
    def _read(key, default):
        try:
            value = float(candidate.get(key, default))
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value):
            return default
        return max(0.0, min(100.0, value))

    base = _read("candidate_score", 50.0)
    review = _read("review_score", base)
    return (0.35 * base) + (0.65 * review)


def _duration_fit(duration: float, policy: DurationPolicy) -> float:
    """1.0 when the moment already sits at the requested length, else lower."""
    if not policy.enabled or not duration:
        return 1.0
    target = float(policy.target_seconds)
    if target <= 0:
        return 1.0
    deviation = abs(duration - target) / target
    return max(0.0, 1.0 - min(1.0, deviation / 0.5))


def _prompt_keywords(prompt) -> set:
    return content_words(prompt) if prompt else set()


def score_candidate(candidate, *, keywords, policy, accepted) -> dict:
    """Score one candidate against the request and the already-accepted set."""
    bounds = _bounds(candidate)
    duration = (bounds[1] - bounds[0]) if bounds else 0.0

    relevance_raw = relevance_score(_evidence_text(candidate), keywords) if keywords else 0.0
    # Normalise the raw hit-per-sqrt(words) score into 0..1.
    relevance = min(1.0, relevance_raw / 2.0) if keywords else 0.5
    quality = _quality_score(candidate) / 100.0
    duration_fit = _duration_fit(duration, policy)

    overlap = max((_overlap_ratio(candidate, prior) for prior in accepted), default=0.0)
    similarity = max((text_similarity(candidate, prior) for prior in accepted), default=0.0)

    # Relevance and duration fit are what make a clip *right*; duplicate and
    # repetition penalties are what stop it repeating a clip we already have.
    composite = (
        0.30 * quality
        + 0.30 * relevance
        + 0.25 * duration_fit
        - 0.10 * min(1.0, overlap)
        - 0.05 * min(1.0, similarity)
    )
    return {
        "relevance": round(relevance, 4),
        "quality": round(quality, 4),
        "duration_fit": round(duration_fit, 4),
        "duplicate_penalty": round(overlap, 4),
        "repetition_penalty": round(similarity, 4),
        "selection_score": round(max(0.0, composite), 4),
        "duration": round(duration, 2),
    }


def _record_evidence(candidate, *, scores, source, visual_tags=None) -> dict:
    """Attach the evidence and scores that justified accepting a candidate."""
    evidence = {
        "source": source,
        "topic": candidate.get("topic") or candidate.get("title"),
        "transcript_excerpt": candidate.get("transcript_excerpt"),
        "visual_tags": list(visual_tags or candidate.get("visual_tags") or []),
        "quality_score": scores["quality"],
    }
    return {
        **candidate,
        "evidence": evidence,
        "selection_scores": scores,
        "quality_score": round(scores["quality"] * 100, 2),
    }


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def select_distinct_candidates(
    pool,
    *,
    requested_count: int,
    policy: DurationPolicy | None = None,
    prompt: str | None = None,
    keywords=None,
    used_intervals=None,
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
    max_text_similarity: float = DEFAULT_MAX_TEXT_SIMILARITY,
    min_duration: float = MIN_USABLE_DURATION,
    allow_context_overlap: bool = True,
) -> SelectionOutcome:
    """Choose up to ``requested_count`` genuinely distinct clips.

    ``used_intervals`` seeds the "already used" set so a later stage (for
    example a montage built after the individual clips) cannot reuse footage
    the individual clips already claimed, unless it opts in.

    The outcome is explicit about shortfalls: it never returns fabricated
    candidates to hit the requested number.
    """
    policy = policy or DurationPolicy()
    requested = max(0, int(requested_count or 0))
    outcome = SelectionOutcome(requested_count=requested)
    if not requested:
        return outcome

    accepted: list[dict] = []
    seeded = [item for item in (used_intervals or []) if isinstance(item, dict)]
    scored: list[dict] = []

    target_keywords = set(keywords) if keywords else _prompt_keywords(prompt)

    pool_size = 0
    for candidate in pool or []:
        pool_size += 1
        bounds = _bounds(candidate)
        duration = (bounds[1] - bounds[0]) if bounds else 0.0
        scores = score_candidate(
            candidate, keywords=target_keywords, policy=policy, accepted=seeded or accepted,
        )
        scored.append({"candidate": candidate, "scores": scores, "duration": duration})

    # Deterministic: best composite score first, then earliest source position.
    def _sort_key(item):
        candidate = item["candidate"]
        start = _bounds(candidate)
        return (
            -item["scores"]["selection_score"],
            start[0] if start else 0.0,
            start[1] if start else 0.0,
        )

    scored.sort(key=_sort_key)

    def _try_accept(item, *, overlap_limit: float, similarity_limit: float,
                    for_context: bool) -> None:
        candidate = item["candidate"]
        if len(accepted) >= requested:
            return
        if _bounds(candidate) is None:
            outcome.rejected.append({"candidate": candidate, "reason": _REJECTION_INVALID})
            return
        duration = item["duration"]
        if duration < min_duration:
            outcome.rejected.append({"candidate": candidate, "reason": _REJECTION_DURATION})
            return
        if policy.enabled and duration > policy.hard_max_seconds:
            outcome.rejected.append({"candidate": candidate, "reason": _REJECTION_DURATION})
            return
        against = seeded + accepted
        overlap = max((_overlap_ratio(candidate, prior) for prior in against), default=0.0)
        similarity = max((text_similarity(candidate, prior) for prior in against), default=0.0)
        if similarity >= similarity_limit:
            outcome.rejected.append({"candidate": candidate, "reason": _REJECTION_REPEATED})
            return
        if overlap > overlap_limit:
            outcome.rejected.append({"candidate": candidate, "reason": _REJECTION_DUPLICATE})
            return

        scores = score_candidate(
            candidate, keywords=target_keywords, policy=policy, accepted=against,
        )
        accepted_candidate = _record_evidence(
            candidate,
            scores=scores,
            source="visual" if candidate.get("visual_only") else "transcript",
        )
        accepted_candidate["candidate_rank"] = len(accepted) + 1
        if for_context:
            accepted_candidate["overlap_allowed_for_context"] = True
            outcome.reasons.append(
                "Limited footage overlap was allowed to fill a remaining slot with a "
                "candidate that still carries its own story."
            )
        accepted.append(accepted_candidate)

    # Pass 1: strict — no reused footage, no repeated explanations.
    strict_candidates = []
    for item in scored:
        if len(accepted) >= requested:
            break
        before = len(accepted)
        _try_accept(item, overlap_limit=max_overlap_ratio,
                    similarity_limit=max_text_similarity, for_context=False)
        if len(accepted) > before:
            strict_candidates.append(item)
        elif outcome.rejected and outcome.rejected[-1]["reason"] == _REJECTION_SURPLUS:
            pass

    # Pass 2: relax overlap only, and only for slots still empty.
    if allow_context_overlap and len(accepted) < requested:
        accepted_keys = {id(item["candidate"]) for item in strict_candidates}
        for item in scored:
            if len(accepted) >= requested:
                break
            if id(item["candidate"]) in accepted_keys:
                continue
            _try_accept(item, overlap_limit=CONTEXT_OVERLAP_RATIO,
                        similarity_limit=max_text_similarity, for_context=True)

    if pool_size and not scored:
        outcome.exhausted = True
        outcome.reasons.append("The candidate pool was empty.")

    if outcome.shortfall:
        counts: dict[str, int] = {}
        for entry in outcome.rejected:
            counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
        for reason, count in sorted(counts.items()):
            label = {
                _REJECTION_DUPLICATE: "reused footage already claimed by an accepted clip",
                _REJECTION_REPEATED: "repeated an explanation already accepted",
                _REJECTION_DURATION: "outside the usable duration range",
                _REJECTION_INVALID: "invalid timestamps",
                _REJECTION_SURPLUS: "surplus candidate",
            }.get(reason, reason)
            outcome.reasons.append(f"{count} candidate(s) rejected because they {label}.")
        if len(outcome.reasons) <= (1 if not pool_size else len(counts)):
            outcome.reasons.append(
                "The source did not contain enough distinct relevant moments."
            )
        outcome.exhausted = True

    outcome.reasons = list(dict.fromkeys(outcome.reasons))
    accepted.sort(key=lambda item: float(item.get("start", 0.0)))
    for index, candidate in enumerate(accepted, start=1):
        candidate["candidate_rank"] = index
    outcome.candidates = accepted
    return outcome


def validate_candidates_against_prompt(candidates, *, prompt, keywords=None,
                                       min_relevance: float = 0.0) -> dict:
    """Re-check accepted clips against the user's request before rendering.

    Returns a per-clip verdict plus a roll-up. A clip with no evidential link
    to the prompt is reported so QA can downgrade the job instead of shipping
    unrelated footage as a success.
    """
    target_keywords = set(keywords) if keywords else _prompt_keywords(prompt)
    checks = []
    for candidate in candidates or []:
        evidence_text = _evidence_text(candidate)
        raw = relevance_score(evidence_text, target_keywords) if target_keywords else 0.0
        relevance = min(1.0, raw / 2.0) if target_keywords else 0.5
        checks.append({
            "file": candidate.get("file"),
            "index": candidate.get("index"),
            "relevance": round(relevance, 4),
            "prompt_match": bool(not target_keywords or relevance > min_relevance),
            "evidence_source": (candidate.get("evidence") or {}).get("source"),
        })
    matched = sum(1 for check in checks if check["prompt_match"])
    return {
        "checked_count": len(checks),
        "matched_count": matched,
        "unmatched_count": len(checks) - matched,
        "prompt_provided": bool(target_keywords),
        "checks": checks,
    }
