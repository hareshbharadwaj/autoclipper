"""Editorial QA for planned clips — separate from technical render QA.

Technical QA (``quality_assurance.verify_rendered_file``) answers "did FFmpeg
produce a valid file?". Editorial QA answers the different question "does this
clip work as a standalone story?" — and it must never be marked passed merely
because a render succeeded.

The checks here run on the *edit plan* (and the transcript), before and
independently of rendering:

- the opening has enough context and is not a sentence fragment;
- every selected passage relates to the requested topic;
- the central message is understandable;
- the conclusion is complete and no payoff is cut off;
- any AI-generated narration is grounded in the available transcript evidence;
- duration mismatches are explained, never silent.

Everything is deterministic so it can be unit-tested with synthetic transcripts.
"""
import re

from .edit_planner import (
    ends_like_fragment,
    map_source_time_to_edit_time,
    relevance_score,
    starts_like_fragment,
    content_words,
)

_DIGIT_RE = re.compile(r"\d+")
_QUOTED_RE = re.compile(r"[\"“]([^\"”]{6,})[\"”]")


def _normalized(text):
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _check(name, ok, detail):
    return {"ok": bool(ok), "detail": str(detail or "")}


def _source_segments(plan):
    return [s for s in plan.get("segments", []) or [] if s.get("origin") == "source"]


def _plan_text(plan):
    parts = []
    for segment in _source_segments(plan):
        parts.append(str(segment.get("text") or ""))
    return " ".join(parts)


def review_edit_plan(plan, transcript=None):
    """Review one edit plan for standalone-story quality.

    Returns ``{"status": "pass"|"warning"|"fail", "checks": {...},
    "warnings": [...], "errors": [...]}``. A plan that cannot be trusted as a
    standalone story fails editorially even when it is technically renderable.
    """
    plan = plan or {}
    checks = {}
    warnings = []
    errors = []

    source_segments = _source_segments(plan)
    opening = plan.get("opening") or {}
    conclusion = plan.get("conclusion") or {}
    narration = plan.get("narration")
    selected_text = _plan_text(plan)

    # 1. Opening is a standalone introduction, not a sentence fragment.
    opening_text = str(opening.get("text") or "")
    if not source_segments:
        checks["opening_context"] = _check(
            "opening_context", False, "The plan contains no source passages."
        )
        errors.append("The plan has no source passages to build a clip from.")
    else:
        first = source_segments[0]
        starts_with_source = (
            opening.get("source_start") is not None
            and abs(float(first.get("source_start", -1)) - float(opening.get("source_start", -2))) < 0.05
        )
        fragment = starts_like_fragment(opening_text)
        checks["opening_not_fragment"] = _check(
            "opening_not_fragment",
            not fragment,
            "The opening is a complete sentence."
            if not fragment else
            f"The opening ('{opening_text[:70]}') reads like the middle of a thought.",
        )
        if fragment:
            errors.append(
                "The clip opens on a sentence fragment or dangling connective; "
                "viewers would need the missing source context."
            )
        enough_context = (
            bool(opening_text.split())
            and len(opening_text.split()) >= 3
            and starts_with_source
        )
        checks["opening_context"] = _check(
            "opening_context",
            enough_context,
            "The opening establishes the topic from the start of a complete "
            "source sentence."
            if enough_context else
            "The opening does not establish enough context for a viewer who "
            "has not seen the source.",
        )
        if not enough_context:
            warnings.append(
                "Opening context is thin; consider a moment whose setup is in "
                "the source video."
            )
        if plan.get("opening_repaired"):
            warnings.append(
                "The opening was extended back to earlier source context so the "
                "clip starts on a complete thought."
            )

    # 2. All selected passages relate to the requested topic.
    topic = plan.get("topic") or ""
    central = plan.get("central_message") or ""
    keywords = content_words(" ".join([topic, central]))
    relevance = relevance_score(selected_text, keywords) if keywords else 0.0
    relevance_ok = (not keywords) or relevance >= 0.35
    checks["topic_relevance"] = _check(
        "topic_relevance",
        relevance_ok,
        f"Selected passages score {relevance:.2f} topical relevance against "
        f"'{(topic or central)[:60]}'.",
    )
    if not relevance_ok:
        warnings.append(
            "Some planned passages only weakly match the requested topic; "
            "review the edit before publishing."
        )

    # 3. The central message is stated and understandable.
    message_ok = bool(central.strip()) and len(central.split()) >= 3
    checks["central_message"] = _check(
        "central_message",
        message_ok,
        f"Central message: '{central[:100]}'"
        if message_ok else "The plan does not state a clear central message.",
    )
    if not message_ok:
        warnings.append("The clip's central message is unclear.")

    # 4. Complete conclusion; no important payoff cut off.
    conclusion_text = str(conclusion.get("text") or "")
    if not source_segments:
        conclusion_ok = False
        conclusion_detail = "No conclusion exists without source passages."
    else:
        last = source_segments[-1]
        conclusion_is_last = (
            conclusion.get("source_start") is not None
            and abs(float(last.get("source_start", -1)) - float(conclusion.get("source_start", -2))) < 0.05
        )
        dangling = ends_like_fragment(conclusion_text)
        conclusion_ok = conclusion_is_last and not dangling
        if conclusion_ok:
            conclusion_detail = "The clip ends on a finished sentence."
        elif dangling:
            conclusion_detail = (
                f"The closing line ('{conclusion_text[:70]}') trails off mid-thought."
            )
        else:
            conclusion_detail = "The planned conclusion is not the final passage."
    checks["conclusion_complete"] = _check(
        "conclusion_complete", conclusion_ok, conclusion_detail
    )
    if not conclusion_ok:
        errors.append(
            "The clip ends mid-thought or drops the planned conclusion; a "
            "payoff or closing sentence would be cut off."
        )

    # Payoff protection: a promised takeaway (evidence) and the planned
    # conclusion must both survive into the final selection.
    evidence_included = plan.get("evidence_included", True)
    checks["no_cut_payoff"] = _check(
        "no_cut_payoff",
        conclusion_ok and evidence_included,
        "No promised payoff or closing sentence is cut off."
        if (conclusion_ok and evidence_included) else
        "A promised payoff or closing sentence would be cut off by this edit.",
    )
    if conclusion_ok and not evidence_included:
        errors.append("The promised payoff/evidence is missing from the edit.")
    if plan.get("conclusion_repaired"):
        warnings.append(
            "The ending was extended forward so the complete payoff is included."
        )

    # 5. Duration is explained whenever it leaves the requested window.
    target = plan.get("target_duration")
    status = plan.get("duration_status")
    explanation = str(plan.get("duration_explanation") or "")
    if target is None or status == "no_target":
        checks["duration_explained"] = _check(
            "duration_explained", True, "No duration target was requested."
        )
    elif status == "within_tolerance":
        checks["duration_explained"] = _check(
            "duration_explained", True, explanation or "Duration is within tolerance."
        )
    else:
        explained = len(explanation.strip()) >= 20
        checks["duration_explained"] = _check(
            "duration_explained",
            explained,
            explanation if explained else
            f"Duration status '{status}' has no actionable explanation.",
        )
        if explained:
            warnings.append(explanation)
        else:
            errors.append(
                "The planned duration misses the requested target without an "
                "actionable explanation."
            )

    # 6. AI narration must be grounded in transcript evidence and labelled.
    if narration:
        script = str(narration.get("script") or "")
        quotes = [str(q) for q in narration.get("grounding_quotes") or [] if str(q).strip()]
        transcript_text = _normalized(
            " ".join(
                str(seg.get("text") or "")
                for seg in (transcript or {}).get("segments", []) or []
                if isinstance(seg, dict)
            )
        ) if transcript else _normalized(selected_text)
        origin_ok = narration.get("origin") == "ai" and bool(narration.get("label"))
        quotes_ok = bool(quotes) and all(
            _normalized(quote) in transcript_text for quote in quotes
        )
        # Every quoted phrase inside the script must appear in the transcript.
        script_quotes = _QUOTED_RE.findall(script)
        script_quotes_ok = all(
            _normalized(quote) in transcript_text for quote in script_quotes
        ) if script_quotes else bool(quotes)
        # Numbers asserted in narration must exist in the source evidence.
        narrated_numbers = set(_DIGIT_RE.findall(script))
        quoted_numbers = set(_DIGIT_RE.findall(" ".join(quotes)))
        numbers_ok = narrated_numbers <= (quoted_numbers | set(_DIGIT_RE.findall(transcript_text)))
        grounded = origin_ok and quotes_ok and script_quotes_ok and numbers_ok
        detail_parts = []
        if not origin_ok:
            detail_parts.append("narration is not labelled as AI-generated")
        if not quotes_ok or not script_quotes_ok:
            detail_parts.append("narration quotes do not all appear in the transcript")
        if not numbers_ok:
            detail_parts.append("narration asserts numbers absent from the transcript")
        checks["narration_grounded"] = _check(
            "narration_grounded",
            grounded,
            "AI narration is labelled and every quote is verbatim transcript "
            "evidence." if grounded else "; ".join(detail_parts),
        )
        if not grounded:
            errors.append(
                "AI-generated narration is not fully grounded in the source "
                "transcript; generated speech must not invent quotes, numbers, "
                "events, or claims."
            )
    else:
        checks["narration_grounded"] = _check(
            "narration_grounded",
            True,
            "No AI-generated narration in this plan (source-only edit).",
        )
        if plan.get("mode") == "source_only":
            # Source-only mode must never introduce generated dialogue.
            has_generated = any(
                seg.get("origin") != "source" for seg in plan.get("segments", []) or []
            )
            if has_generated:
                checks["narration_grounded"] = _check(
                    "narration_grounded",
                    False,
                    "Source-only plan contains generated passages.",
                )
                errors.append(
                    "A source-only edit must never introduce generated dialogue."
                )

    # 7. Edit-timeline mapping must stay consistent: passages play back to
    #    back from 0 with no gaps/overlaps, and source times map correctly.
    mapping_ok = True
    mapping_detail = "Edit timeline maps every source passage from its original timestamps."
    previous_end = 0.0
    ordered_segments = plan.get("segments", []) or []
    for position, segment in enumerate(ordered_segments):
        try:
            edit_start = float(segment.get("edit_start"))
            edit_end = float(segment.get("edit_end"))
        except (KeyError, TypeError, ValueError):
            mapping_ok = False
            mapping_detail = "A planned passage is missing timeline timestamps."
            break
        if edit_end <= edit_start:
            mapping_ok = False
            mapping_detail = "A planned passage has a non-positive duration."
            break
        if abs(edit_start - previous_end) > 0.15:
            mapping_ok = False
            mapping_detail = (
                "Edit timeline has a gap or overlap: captions would drift out "
                "of sync with the rendered clip."
            )
            break
        previous_end = edit_end
        if segment.get("origin") == "source":
            try:
                source_start = float(segment["source_start"])
                source_end = float(segment["source_end"])
            except (KeyError, TypeError, ValueError):
                mapping_ok = False
                mapping_detail = "A planned passage is missing source timestamps."
                break
            if source_end <= source_start:
                mapping_ok = False
                mapping_detail = "A planned passage has a non-positive source duration."
                break
            mapped = map_source_time_to_edit_time(plan, source_start + 0.001)
            if mapped is None or abs(mapped - edit_start) > 0.15:
                mapping_ok = False
                mapping_detail = "Caption/source timestamps would not rebase onto the edit timeline."
                break
    checks["timeline_mapping"] = _check("timeline_mapping", mapping_ok, mapping_detail)
    if not mapping_ok:
        errors.append(mapping_detail)

    if errors:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def editorial_summary(clip_reviews):
    """Aggregate per-clip editorial reviews into one report for job results."""
    reviews = list(clip_reviews or [])
    if not reviews:
        return {"status": "not_run", "checked_clip_count": 0, "failed_clip_count": 0, "reports": []}
    failed = sum(1 for review in reviews if review.get("status") == "fail")
    warned = sum(1 for review in reviews if review.get("status") == "warning")
    if failed:
        status = "fail"
    elif warned:
        status = "warning"
    else:
        status = "pass"
    return {
        "status": status,
        "checked_clip_count": len(reviews),
        "failed_clip_count": failed,
        "reports": reviews,
    }
