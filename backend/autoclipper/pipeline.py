"""End-to-end orchestration: ingest -> transcribe -> analyze -> cut."""
import json
import math
import os
import time

from . import config
from .analyze import (
    find_viral_clips, review_clip_candidates, rank_and_diversify_candidates,
    _filter_clips_to_bounds, _resolve_clip_bounds, _validate_clips,
)
from .clip import cut_all
from .edit_planner import (
    build_edit_plan,
    drop_narration,
    duration_mismatch_warning,
    finalize_narration_timing,
    normalize_generation_mode,
    content_words,
    relevance_score,
)
from .editorial_qa import editorial_summary, review_edit_plan
from .ingest import ingest
from .reframe import VERTICAL_ASPECT, reframe_clip
from .quality_assurance import verify_rendered_file
from .transcribe import transcribe
from .utils import (
    ASPECT_PRESETS,
    ensure_ffmpeg,
    logger,
    probe_video_info,
    probe_duration,
    sanitize_filename,
    write_json,
    save_to_library,
)





def _load_cached_transcript(path):
    """Load a prior analysis transcript and reject malformed cache files."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("The analyzed transcript cache is missing; analyze the video again.")
    with open(path, "r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
        raise ValueError("The cached transcript is malformed; analyze the video again.")
    return transcript


def _normalize_target_duration(target_duration):
    """Clamp oversized requests to the hard ceiling; reject malformed values."""
    if target_duration is None:
        return None
    try:
        value = float(target_duration)
    except (TypeError, ValueError):
        raise ValueError("target_duration must be a number between 5 and 120 seconds.")
    return min(value, 120.0)


MAX_CLIP_COUNT = 30  # Preserve the classic generator's existing 1-30 control.


def _normalize_clip_count(clip_count):
    """Validate an optional clip-count request while preserving legacy defaults."""
    if clip_count is None:
        return None
    if isinstance(clip_count, bool):
        raise ValueError("clip_count must be a whole number between 1 and 30.")
    try:
        value = float(clip_count)
    except (TypeError, ValueError):
        raise ValueError("clip_count must be a whole number between 1 and 30.")
    if not math.isfinite(value) or not value.is_integer() or not 1 <= value <= MAX_CLIP_COUNT:
        raise ValueError("clip_count must be a whole number between 1 and 30.")
    return int(value)


def _same_clip_moment(candidate, existing):
    """Return true when most of a candidate's duration duplicates another."""
    try:
        candidate_start, candidate_end = float(candidate["start"]), float(candidate["end"])
        existing_start, existing_end = float(existing["start"]), float(existing["end"])
        overlap = max(0.0, min(candidate_end, existing_end) - max(candidate_start, existing_start))
        shorter = min(candidate_end - candidate_start, existing_end - existing_start)
    except (KeyError, TypeError, ValueError):
        return False
    return shorter > 0 and overlap / shorter >= 0.35


def _merge_unique_clip_candidates(selected, generated, clip_count):
    """Keep the selected moment and fill remaining slots with distinct candidates."""
    limit = max(0, int(clip_count or 0))
    if not limit:
        return []
    unique = []
    for candidate in list(selected or []) + list(generated or []):
        if any(_same_clip_moment(candidate, accepted) for accepted in unique):
            continue
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return sorted(unique, key=lambda item: float(item.get("start", 0.0)))


def _candidate_pool_size(requested_count):
    """Over-generate a bounded candidate pool for ranking and independent review."""
    try:
        requested = max(1, int(requested_count))
    except (TypeError, ValueError):
        requested = 1
    return min(40, requested * 2)


def _add_render_warning(clip, warning, warnings):
    clip.setdefault("render_warnings", []).append(warning)
    if warning not in warnings:
        warnings.append(warning)
    logger.warning("Clip %s: %s", clip.get("index", "?"), warning)


def _nonempty_file(path):
    return bool(path and os.path.isfile(path) and os.path.getsize(path) > 0)


def _rank_montage_candidates(candidates):
    """Order user-selected montage moments chronologically without de-duplication.

    `_fit_montage_candidates` already guarantees 2-3 distinct, non-overlapping,
    transcript-aligned shots, so every candidate is retained and ranked; only
    the internal selection marker is normalized away like the ranking pass does.
    """
    ordered = []
    for item in candidates or []:
        try:
            start = float(item.get("start", 0.0))
        except (TypeError, ValueError):
            start = 0.0
        ordered.append((start, dict(item)))
    ordered.sort(key=lambda pair: pair[0])
    ranked = []
    for index, (_, item) in enumerate(ordered):
        item["candidate_rank"] = index + 1
        item.pop("_user_selected", None)
        ranked.append(item)
    return ranked


def _emit_progress(callback, stage, message, *, current=None, total=None, **details):
    if callback is None:
        return
    event = {"stage": stage, "message": message, "updated_at": time.time()}
    if current is not None:
        event["current"] = int(current)
    if total is not None:
        event["total"] = int(total)
    event.update(details)
    try:
        callback(event)
    except Exception:
        pass


def _montage_duration_tolerance(target_duration):
    """Maximum acceptable target variance before explaining the actual duration."""
    target = max(0.0, float(target_duration or 0.0))
    return max(3.0, min(8.0, target * 0.08))


def _fit_montage_candidates(candidates, transcript, target_duration, total_duration,
                            crossfade_seconds=0.25):
    """Choose topic-grounded, sentence-aligned shots within the montage time budget.

    Analysis timestamps are hints, not proof that the hinted passage matches the
    angle. Prefer transcript passages whose words support the selected angle, and
    treat the montage target as a shared budget across all selected shots.
    """
    from .edit_planner import content_words, relevance_score

    segments = []
    for segment in transcript.get("segments", []) or []:
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = (segment.get("text") or "").strip()
        if end <= start or not text:
            continue
        if total_duration:
            start = max(0.0, min(start, float(total_duration)))
            end = max(0.0, min(end, float(total_duration)))
        if end > start:
            segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not segments:
        raise ValueError("Montage needs timestamped transcript segments.")
    if not candidates or not 2 <= len(candidates) <= 3:
        raise ValueError("Montage mode requires 2 or 3 selected video moments.")

    target = min(float(config.HARD_MAX_CLIP_SECONDS), max(5.0, float(target_duration)))
    shot_count = len(candidates)
    fade = max(0.0, float(crossfade_seconds))
    desired_source_total = min(
        float(config.HARD_MAX_CLIP_SECONDS), target + fade * (shot_count - 1)
    )
    desired_per_shot = max(3.0, desired_source_total / shot_count)
    # A shot may vary from the ideal only as much as sentence boundaries require.
    # Crucially, the original angle hint is not a minimum-duration requirement.
    max_per_shot = min(
        float(config.HARD_MAX_CLIP_SECONDS),
        desired_per_shot + max(3.0, desired_per_shot * 0.25),
    )

    def _expanded_terms(text):
        terms = set(content_words(text))
        # Small deterministic morphology expansion handles common transcript
        # variants (rolling/roll, vegetables/vegetable, chopping/chop) without
        # relying on another model call.
        for term in list(terms):
            if term.endswith("ing") and len(term) > 5:
                stem = term[:-3]
                terms.add(stem)
                if len(stem) > 2 and stem[-1] == stem[-2]:
                    terms.add(stem[:-1])
            if term.endswith("ies") and len(term) > 4:
                terms.add(term[:-3] + "y")
            if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
                terms.add(term[:-1])
        synonym_groups = (
            {"veggie", "veggies", "vegetable", "vegetables"},
            {"roll", "rolling", "rolled", "rolls"},
            {"knife", "knives"},
            {"chop", "chopping", "chopped", "chops"},
            {"slice", "slicing", "sliced", "slices"},
        )
        for group in synonym_groups:
            if terms & group:
                terms.update(group)
        return terms

    chosen = []
    for angle in candidates:
        try:
            hint_start = float(angle.get("start", angle.get("start_hint", 0)))
            hint_end = float(angle.get("end", angle.get("end_hint", 0)))
        except (TypeError, ValueError):
            hint_start, hint_end = 0.0, 0.0
        if total_duration:
            hint_start = max(0.0, min(hint_start, float(total_duration)))
            hint_end = max(0.0, min(hint_end, float(total_duration)))

        angle_text = " ".join(
            str(angle.get(key) or "")
            for key in ("title", "hook", "description", "custom_prompt", "reason", "story_arc")
        )
        keywords = _expanded_terms(angle_text)
        if not keywords:
            raise ValueError(
                f"Selected montage angle '{angle.get('title') or 'Untitled'}' has no usable topic terms."
            )
        segment_scores = [relevance_score(seg["text"], keywords) for seg in segments]

        # Select the best transcript-supported anchor across the whole video. The
        # model timestamp is used only to break ties; it cannot force an unrelated
        # section into a selected angle.
        available_indices = [
            i for i, seg in enumerate(segments)
            if not any(seg["start"] < item["end"] and seg["end"] > item["start"] for item in chosen)
        ]
        matching_indices = [i for i in available_indices if segment_scores[i] > 0]
        if not matching_indices:
            raise ValueError(
                f"Could not ground selected angle '{angle.get('title') or 'Untitled'}' "
                "in the transcript. Choose a different angle or analyze the source again."
            )
        anchor_idx = max(
            matching_indices,
            key=lambda i: (
                segment_scores[i],
                -abs(segments[i]["start"] - hint_start) if hint_end > hint_start else 0.0,
            ),
        )

        # Evaluate contiguous windows containing the topical anchor. Keep complete
        # transcript segments, avoid already-selected footage, and prefer the
        # requested per-shot budget without padding with unrelated material.
        windows = []
        for lo in range(anchor_idx, -1, -1):
            if segments[anchor_idx]["end"] - segments[lo]["start"] > max_per_shot:
                break
            for hi in range(anchor_idx, len(segments)):
                shot_start = segments[lo]["start"]
                shot_end = segments[hi]["end"]
                duration = shot_end - shot_start
                if duration > max_per_shot:
                    break
                if duration < 3.0:
                    continue
                if any(
                    shot_start < item["end"] and shot_end > item["start"]
                    for item in chosen
                ):
                    continue
                scores = segment_scores[lo:hi + 1]
                avg_relevance = sum(scores) / max(1, len(scores))
                peak_relevance = max(scores, default=0.0)
                duration_penalty = abs(duration - desired_per_shot) / desired_per_shot
                # Peak relevance ensures the requested topic is present; average
                # relevance favors windows that stay on topic instead of drifting.
                score = (avg_relevance * 4.0) + (peak_relevance * 0.5) - (duration_penalty * 2.0)
                windows.append((score, duration, shot_start, shot_end, lo, hi))

        if not windows:
            raise ValueError(
                f"Could not find a distinct transcript-aligned shot for angle "
                f"'{angle.get('title') or 'Untitled'}' within its montage time budget."
            )
        _, duration, shot_start, shot_end, lo, hi = max(
            windows, key=lambda w: (w[0], -abs(w[1] - desired_per_shot))
        )
        chosen.append({
            "start": round(shot_start, 2),
            "end": round(shot_end, 2),
            "title": (angle.get("title") or "Montage moment")[:80],
            "description": (angle.get("description") or "")[:220],
            "hook": (angle.get("hook") or "")[:120],
            "custom_prompt": (angle.get("custom_prompt") or "")[:1200],
            "reason": angle.get("reason") or "Selected from whole-video analysis",
            "story_arc": angle.get("story_arc") or "Transcript-grounded selected angle",
            # Override any angle-level estimate: montage clips share one total budget.
            "target_duration": round(desired_per_shot, 2),
            "topic_match_score": round(segment_scores[anchor_idx], 3),
            "_user_selected": True,
        })

    chosen.sort(key=lambda item: item["start"])
    if not 2 <= len(chosen) <= 3:
        raise ValueError("Could not find 2-3 distinct, topic-grounded montage moments.")
    return chosen


def _enforce_clip_constraints(clips, min_clip, max_clip, clip_count, video_dur, trim_long=True, pad_short=True):
    """Adjust Gemini's candidates to respect the user's count and duration window."""
    cleaned = []
    for c in clips:
        try:
            start = float(c.get("start", 0.0))
            end = float(c.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        dur = end - start

        # Trim clips longer than the requested max.
        if trim_long and max_clip and dur > max_clip:
            end = start + max_clip
            dur = max_clip

        # Pad short clips outward (within video bounds) up to the min duration.
        if pad_short and min_clip and dur < min_clip:
            need = min_clip - dur
            pad_before = min(start, need / 2.0)
            pad_after = min(max(0.0, video_dur - end), need - pad_before)
            # If we still need more, take the rest from the front.
            if pad_before + pad_after < need:
                pad_before = min(start, need - pad_after)
            start = max(0.0, start - pad_before)
            end = min(video_dur or end, end + pad_after)
            # Final clamp so we never exceed min by padding past video end.
            if video_dur and end - start < min_clip and start > 0:
                start = max(0.0, end - min_clip)

        cleaned.append({**c, "start": round(start, 2), "end": round(end, 2)})

    # Cap the total number of clips at the requested count.
    if clip_count and len(cleaned) > clip_count:
        cleaned = cleaned[:clip_count]

    return cleaned


def _plan_clip_edits(candidates, transcript, video_dur, target_duration,
                     generation_mode, custom_prompt, processing_warnings):
    """Shape each candidate into a complete, standalone planned edit.

    The edit planner — not a blind segment-extension rule — decides the final
    source bounds: it repairs the opening and conclusion at sentence
    boundaries, expands with relevant context (or trims the front of the body)
    toward the requested duration when the story allows it, and reports any
    shortfall explicitly instead of padding with unrelated material.
    """
    planned = []
    for candidate in candidates or []:
        try:
            plan = build_edit_plan(
                transcript,
                candidate,
                target_duration=target_duration,
                generation_mode=generation_mode,
                custom_prompt=custom_prompt,
                total_duration=video_dur,
            )
            editorial = review_edit_plan(plan, transcript)
        except Exception as exc:  # noqa: BLE001 — keep the candidate, explain why
            logger.warning(
                "Edit planning failed (%s); keeping original candidate bounds.",
                type(exc).__name__,
            )
            _add_render_warning(
                candidate,
                f"Edit planning could not run ({type(exc).__name__}); the "
                "original candidate timestamps were kept without story "
                "structure checks. Treat this clip as unreviewed output.",
                processing_warnings,
            )
            planned.append({
                **candidate,
                "planning_failed": True,
                "edit_plan": None,
                "editorial_qa": {
                    "status": "not_run",
                    "checks": {},
                    "warnings": [
                        "Editorial QA did not run because edit planning "
                        "failed; this clip is unreviewed as a standalone story."
                    ],
                    "errors": [],
                },
            })
            continue
        shaped = {
            **candidate,
            "start": plan["source_start"],
            "end": plan["source_end"],
            "edit_plan": plan,
            "editorial_qa": editorial,
        }
        if plan.get("target_duration") is not None:
            shaped["target_duration"] = plan["target_duration"]
            shaped["duration_delta"] = round(
                plan["estimated_duration"] - plan["target_duration"], 2
            )
        planned.append(shaped)
        for warning in plan.get("warnings", []):
            _add_render_warning(shaped, warning, processing_warnings)
        if editorial.get("status") == "fail":
            detail = "; ".join(editorial.get("errors") or ["story structure check failed"])
            _add_render_warning(
                shaped, f"Editorial QA failed: {detail}", processing_warnings
            )
    return planned


def run(
    source: str,
    output_dir: str = "output",
    groq_key: str = None,
    gemini_key: str = None,
    whisper_model: str = None,
    gemini_model: str = None,
    vertical: bool = True,
    use_yolo: bool = True,
    # Deprecated and ignored: captions are no longer generated or burned during
    # clip generation/reframing. Users add captions later from the clip editor.
    # Kept only so older clients passing ``subtitles=`` keep working.
    subtitles: bool = False,
    split_screen: bool = False,
    force_hd: bool = False,
    cookies_text: str = None,
    clip_count: int = None,
    min_clip: float = None,
    max_clip: float = None,
    content_type: str = None,
    target_duration: float = None,
    generation_mode: str = "source_only",
    custom_prompt: str = None,
    selected_clip: dict = None,
    aspect_ratio: str = None,
    add_hook: bool = False,
    montage: bool = False,
    selected_clips: list = None,
    cached_transcript_path: str = None,
    on_clip=None,
    on_progress=None,
) -> dict:
    """Run the full auto-clip pipeline and return a result summary dict.

    If *on_clip* is provided it will be called with ``(clip_dict, index, total)``
    after each clip has been rendered, before final QA. *on_progress*, when set,
    receives structured stage and per-clip QA events for progress polling.

    Captions are intentionally not generated here (the ``subtitles`` argument is
    deprecated and ignored): the render flow produces a clean clip, and users
    add captions afterwards from the clip editor whenever they want.
    """
    # Clamp oversized requests and validate count before any ingest/model work.
    target_duration = _normalize_target_duration(target_duration)
    generation_mode = normalize_generation_mode(generation_mode)
    clip_count = _normalize_clip_count(clip_count)
    if montage:
        if not selected_clips or not 2 <= len(selected_clips) <= 3:
            raise ValueError("Montage mode requires 2 or 3 selected video moments.")
        requested_individual_count = len(selected_clips)
        effective_min_clip, effective_max_clip = 3.0, float(config.HARD_MAX_CLIP_SECONDS)
    else:
        requested_individual_count = (
            clip_count if clip_count is not None
            else (1 if selected_clip else config.TARGET_CLIP_COUNT)
        )
        effective_min_clip, effective_max_clip = _resolve_clip_bounds(
            min_clip, max_clip, target_duration
        )
    if max_clip is not None:
        try:
            max_clip = min(float(max_clip), float(config.HARD_MAX_CLIP_SECONDS))
        except (TypeError, ValueError):
            max_clip = None
    ensure_ffmpeg()
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingest
    _emit_progress(on_progress, "ingesting", "Preparing the source video.")
    video_path, title = ingest(source, output_dir, force_hd=force_hd, cookies_text=cookies_text)
    base_name = sanitize_filename(title)
    video_info = probe_video_info(video_path)
    if not video_info.get("duration"):
        video_info["duration"] = 0.0

    # 1b. Resolve the output aspect ratio (9:16 portrait / 1:1 square / 16:9).
    #     `vertical` is kept for backwards compatibility with older clients.
    target_aspect = None
    aspect_key = (aspect_ratio or "").strip()
    if aspect_key and aspect_key in ASPECT_PRESETS:
        if aspect_key != "16:9":
            target_aspect = ASPECT_PRESETS[aspect_key]["ratio"]
    elif aspect_key and aspect_key not in ASPECT_PRESETS:
        logger.warning("Unknown aspect_ratio '%s'; falling back to 9:16.", aspect_key)
        target_aspect = VERTICAL_ASPECT
    elif vertical:
        target_aspect = VERTICAL_ASPECT
    if target_aspect is not None:
        logger.info("Reframing to %s (aspect %.4f)", aspect_key or "9:16", target_aspect)
    else:
        logger.info("Keeping original 16:9 framing (no crop).")

    # 2. Reuse the whole-video transcript when analysis-first already paid for it.
    _emit_progress(on_progress, "transcribing", "Transcribing or loading the source transcript.")
    if cached_transcript_path:
        transcript = _load_cached_transcript(cached_transcript_path)
        logger.info("Reusing cached whole-video transcript; skipping a second Groq call.")
    else:
        transcript = transcribe(
            video_path, output_dir, api_key=groq_key, model=whisper_model
        )

    # 3. Generate a bounded candidate pool, independently review it, then rank
    #    and diversify. Transcript-grounded timestamps remain authoritative.
    _emit_progress(
        on_progress,
        "reviewing_candidates",
        "Finding distinct moments and checking them against transcript/visual evidence.",
    )
    video_dur = video_info.get("duration") or transcript.get("duration") or 0.0
    candidate_pool_size = _candidate_pool_size(requested_individual_count)
    candidate_warnings = []
    candidate_review = {"status": "not_run", "reviewed_count": 0, "rejected_count": 0}

    if montage:
        candidate_pool = _fit_montage_candidates(
            selected_clips, transcript,
            target_duration=float(target_duration or 40),
            total_duration=video_dur,
        )
    elif selected_clip:
        # Keep the selected angle as a real output when its hinted timestamps are
        # valid. Other slots draw from an over-generated, distinct candidate pool.
        selected_candidates = _validate_clips(
            [selected_clip],
            total_duration=video_dur,
            min_clip=effective_min_clip,
            max_clip=effective_max_clip,
        )
        selected_candidates = [{**item, "_user_selected": True} for item in selected_candidates]
        if clip_count is not None and len(selected_candidates) < requested_individual_count:
            extra_prompt = (custom_prompt or "").strip()
            if selected_candidates:
                chosen = selected_candidates[0]
                direction = (
                    f"The user-selected moment from {chosen['start']:.1f}s to "
                    f"{chosen['end']:.1f}s is already one output. Return other "
                    "distinct, non-overlapping moments only."
                )
            else:
                angle_title = (selected_clip.get("title") or "").strip()
                direction = (
                    f"Use the selected angle '{angle_title}' as creative direction "
                    "where it fits, but return valid transcript-aligned moments."
                )
            extra_prompt = f"{extra_prompt}\n\n{direction}".strip()
            generated_candidates = find_viral_clips(
                transcript, api_key=gemini_key, model=gemini_model,
                clip_count=candidate_pool_size,
                min_clip=effective_min_clip if target_duration is not None else min_clip,
                max_clip=effective_max_clip if target_duration is not None else max_clip,
                content_type=content_type, target_duration=target_duration,
                custom_prompt=extra_prompt,
            )
            candidate_pool = _merge_unique_clip_candidates(
                selected_candidates, generated_candidates, candidate_pool_size
            )
        else:
            candidate_pool = selected_candidates
    else:
        candidate_pool = find_viral_clips(
            transcript, api_key=gemini_key, model=gemini_model,
            clip_count=candidate_pool_size,
            min_clip=effective_min_clip if target_duration is not None else min_clip,
            max_clip=effective_max_clip if target_duration is not None else max_clip,
            content_type=content_type, target_duration=target_duration,
            custom_prompt=custom_prompt,
        )

    if not montage:
        candidate_pool = _merge_unique_clip_candidates(
            [], candidate_pool, candidate_pool_size
        )

    review_key = gemini_key or config.GEMINI_API_KEY
    if candidate_pool and review_key:
        try:
            from .visual_context import sample_candidate_frames

            sampled = sample_candidate_frames(video_path, candidate_pool)
            if sampled.get("warning"):
                candidate_warnings.append(
                    "Candidate visual sampling was incomplete; independent review used transcript evidence and any extracted frames."
                )
            review = review_clip_candidates(
                transcript,
                candidate_pool,
                api_key=review_key,
                model=gemini_model,
                visual_samples=sampled.get("frames") or [],
                total_duration=video_dur,
                min_clip=effective_min_clip,
                max_clip=effective_max_clip,
            )
            candidate_pool = review["candidates"]
            candidate_review = {
                "status": "reviewed",
                "reviewed_count": review["reviewed_count"],
                "rejected_count": review["rejected_count"],
            }
        except Exception as exc:  # fall back to validated, unchanged candidates
            logger.warning(
                "Independent candidate review unavailable (%s); retaining original timestamps.",
                type(exc).__name__,
            )
            candidate_review = {
                "status": "fallback_unreviewed",
                "reviewed_count": 0,
                "rejected_count": 0,
            }
            candidate_warnings.append(
                "Independent candidate review could not be completed; original transcript-validated timestamps were retained. Check provider availability and retry if needed."
            )
    elif candidate_pool:
        candidate_review = {"status": "skipped_no_key", "reviewed_count": 0, "rejected_count": 0}

    _emit_progress(
        on_progress,
        "ranking_candidates",
        "Ranking and diversifying the reviewed moments.",
        candidate_count=len(candidate_pool),
        candidate_review=candidate_review,
    )
    if montage:
        # Every fitted montage moment is a required user selection; text-similarity
        # de-duplication must never drop one (two selected angles with similar
        # titles/hooks would otherwise lose a shot and the montage would fail).
        clip_candidates = _rank_montage_candidates(candidate_pool)
    else:
        clip_candidates = rank_and_diversify_candidates(candidate_pool, requested_individual_count)
    logger.info(
        "Clip params -> requested count=%s min=%s max=%s | reviewed pool=%d | selected %d candidate(s)",
        requested_individual_count, min_clip, max_clip, len(candidate_pool), len(clip_candidates),
    )

    # Story planning replaces blind duration enforcement. Each candidate is
    # shaped into a complete standalone edit (opening -> body -> conclusion)
    # before rendering: the planner repairs mid-thought boundaries at sentence
    # edges, expands with relevant context or trims toward the requested
    # duration when the story allows, and reports any mismatch explicitly
    # instead of silently dropping clips or padding with unrelated material.
    if montage and clip_candidates:
        # Montage shots use their own share of the overall target and must pass
        # the same standalone-story planner as ordinary clips.
        clip_candidates = _plan_clip_edits(
            clip_candidates, transcript, video_dur, None,
            generation_mode, custom_prompt, candidate_warnings,
        )
        clip_candidates = _enforce_clip_constraints(
            clip_candidates, effective_min_clip, effective_max_clip,
            len(clip_candidates), video_dur,
            trim_long=False, pad_short=False,
        )
    elif clip_candidates:
        clip_candidates = _plan_clip_edits(
            clip_candidates, transcript, video_dur, target_duration,
            generation_mode, custom_prompt, candidate_warnings,
        )
        if target_duration is None:
            # Legacy min/max mode keeps its established bounds check; the
            # target-duration path never silently drops a planned story.
            clip_candidates = _filter_clips_to_bounds(
                clip_candidates,
                min_clip=effective_min_clip,
                max_clip=effective_max_clip,
                min_tolerance=2.0,
                max_tolerance=3.0,
            )
        clip_candidates = _enforce_clip_constraints(
            clip_candidates, effective_min_clip, effective_max_clip,
            requested_individual_count, video_dur,
            trim_long=False, pad_short=False,
        )
        logger.info("After planning: %d clip(s)", len(clip_candidates))
    if not clip_candidates:
        logger.warning("No viral clips detected.")

    # 4. Cut clips (FFmpeg)
    _emit_progress(
        on_progress,
        "rendering",
        "Rendering selected clips and requested transformations.",
        current=0,
        total=len(clip_candidates),
        candidate_review=candidate_review,
    )
    clips = cut_all(video_path, clip_candidates, output_dir, base_name)
    total_clips = len(clips)

    # Process each clip one-by-one so partial results can stream to the UI.
    # Keep earlier stages until post-render QA confirms the final file is usable.
    processed = []
    processing_warnings = list(candidate_warnings)
    for idx, clip in enumerate(clips):
        fallbacks = []

        # 4b. Chat-split. Keep the source clip if the transform fails.
        if split_screen:
            from .gaming_layout import auto_chat_split_clip as _auto_split
            source_name = clip["file"]
            src = os.path.join(output_dir, source_name)
            s_name = source_name.rsplit(".", 1)[0] + "_split.mp4"
            s_path = os.path.join(output_dir, s_name)
            try:
                if not _nonempty_file(src):
                    raise RuntimeError("The source clip is missing or empty.")
                _auto_split(src, s_path)
                if not _nonempty_file(s_path):
                    raise RuntimeError("No non-empty split-screen file was created.")
                fallbacks.append(source_name)
                clip["file"] = s_name
                clip["split_screen"] = True
            except Exception as exc:  # noqa: BLE001
                _add_render_warning(
                    clip,
                    f"Split-screen processing was skipped ({type(exc).__name__}); the prior clip was retained.",
                    processing_warnings,
                )
                try:
                    if os.path.exists(s_path):
                        os.remove(s_path)
                except OSError:
                    pass

        # 5. Reframe to the requested aspect ratio, retaining the prior render
        #    if subject tracking or the output stream check fails.
        if target_aspect is not None and not clip.get("split_screen"):
            prior_name = clip["file"]
            src = os.path.join(output_dir, prior_name)
            suffix = "9x16" if abs(target_aspect - (9 / 16)) < 1e-6 else (
                "1x1" if abs(target_aspect - 1.0) < 1e-6 else "crop"
            )
            v_name = prior_name.rsplit(".", 1)[0] + f"_{suffix}.mp4"
            v_path = os.path.join(output_dir, v_name)
            try:
                if not _nonempty_file(src):
                    raise RuntimeError("The source clip is missing or empty.")
                if not reframe_clip(src, v_path, use_yolo=use_yolo, aspect=target_aspect):
                    raise RuntimeError("The reframed output failed verification.")
                if not _nonempty_file(v_path):
                    raise RuntimeError("No non-empty reframed file was created.")
                fallbacks.append(prior_name)
                clip["vertical_file"] = v_name
                clip["file"] = v_name
            except Exception as exc:  # noqa: BLE001
                _add_render_warning(
                    clip,
                    f"Aspect-ratio reframing failed ({type(exc).__name__}); the prior framing was retained.",
                    processing_warnings,
                )
                try:
                    if os.path.exists(v_path):
                        os.remove(v_path)
                except OSError:
                    pass

        # 6. Generate the SRT sidecar for the clip (unburned) so captions can be edited
        #    and burned later from the clip editor whenever desired.
        try:
            from .subtitles import add_subtitles
            add_subtitles([clip], transcript, output_dir, burn=False, preserve_source=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not generate SRT sidecar for clip %s: %s", clip.get("index"), exc)

        # 6b. Burn the hook banner only if a complete output is produced.
        if add_hook and clip.get("hook"):
            prior_name = clip["file"]
            src = os.path.join(output_dir, prior_name)
            h_name = prior_name.rsplit(".", 1)[0] + "_hook.mp4"
            h_path = os.path.join(output_dir, h_name)
            try:
                if not _nonempty_file(src):
                    raise RuntimeError("The source clip is missing or empty.")
                from .hooks import add_hook_to_video
                add_hook_to_video(src, clip["hook"], h_path, position="top", duration=4.0)
                if not _nonempty_file(h_path):
                    raise RuntimeError("No non-empty hook-overlay file was created.")
                fallbacks.append(prior_name)
                clip["file"] = h_name
                clip["hook_file"] = h_name
            except Exception as exc:  # noqa: BLE001
                _add_render_warning(
                    clip,
                    f"Hook overlay was skipped ({type(exc).__name__}); the prior clip was retained.",
                    processing_warnings,
                )
                try:
                    if os.path.exists(h_path):
                        os.remove(h_path)
                except OSError:
                    pass

        clip["_render_fallbacks"] = list(dict.fromkeys(fallbacks))

        processed.append(clip)
        _emit_progress(
            on_progress,
            "rendering",
            f"Clip {idx + 1} of {total_clips} rendered; final QA is pending.",
            current=idx + 1,
            total=total_clips,
            candidate_review=candidate_review,
        )

        # Do not expose internal fallback filenames in incremental job polling.
        if on_clip is not None:
            try:
                public_clip = {key: value for key, value in clip.items() if not key.startswith("_")}
                on_clip(public_clip, idx + 1, total_clips)
            except Exception:
                pass

    actual_individual_count = len(processed)

    # Plan B: stitch the selected, independently captioned moments into one montage.
    if montage:
        _emit_progress(on_progress, "montage", "Stitching the selected clips into a montage.")
        try:
            from .montage import stitch_montage
            montage_name = f"{base_name}_montage.mp4"
            montage_path = os.path.join(output_dir, montage_name)
            stitch_montage(
                [os.path.join(output_dir, item["file"]) for item in processed],
                montage_path,
            )
            montage_duration = probe_duration(montage_path)
            if montage_duration > float(config.HARD_MAX_CLIP_SECONDS) + 0.25:
                raise RuntimeError(
                    f"Montage duration {montage_duration:.1f}s exceeds the hard "
                    f"{config.HARD_MAX_CLIP_SECONDS:.0f}s limit; choose fewer/shorter moments."
                )
            duration_delta = montage_duration - float(target_duration or 40)
            duration_tolerance = _montage_duration_tolerance(target_duration or 40)
            if abs(duration_delta) > duration_tolerance:
                processing_warnings.append(
                    f"Montage actual duration is {montage_duration:.2f}s versus the "
                    f"{float(target_duration or 40):.0f}s target ({duration_delta:+.2f}s; "
                    f"tolerance ±{duration_tolerance:.1f}s). Complete transcript boundaries "
                    "were preserved; meeting the target would require cutting speech or "
                    "reusing footage."
                )
            montage_clip_dict = {
                "index": len(processed) + 1,
                "file": montage_name,
                "title": f"{title} — Montage",
                "start": 0.0,
                "end": round(montage_duration, 2),
                "duration": round(montage_duration, 2),
                "target_duration": float(target_duration or 40),
                "duration_delta": round(duration_delta, 2),
                "is_montage": True,
                "reason": "Stitched from selected complementary moments with video/audio crossfades.",
            }
            # Generate SRT sidecar for the stitched montage clip
            try:
                from .subtitles import cues_to_srt, parse_srt
                montage_cues = []
                current_offset = 0.0
                fade = 0.25
                for p_clip in processed:
                    if p_clip.get("is_montage"):
                        continue
                    if p_clip.get("srt"):
                        p_srt_path = os.path.join(output_dir, p_clip["srt"])
                        if os.path.isfile(p_srt_path):
                            with open(p_srt_path, "r", encoding="utf-8") as _f:
                                p_cues = parse_srt(_f.read())
                            shot_dur = p_clip.get("duration") or (p_clip["end"] - p_clip["start"])
                            for cue in p_cues:
                                montage_cues.append({
                                    "start": round(cue["start"] + current_offset, 2),
                                    "end": round(cue["end"] + current_offset, 2),
                                    "text": cue["text"],
                                })
                            current_offset += max(0.0, shot_dur - fade)
                if montage_cues:
                    montage_srt_name = f"{base_name}_montage.srt"
                    montage_srt_path = os.path.join(output_dir, montage_srt_name)
                    with open(montage_srt_path, "w", encoding="utf-8") as _f:
                        _f.write(cues_to_srt(montage_cues))
                    montage_clip_dict["srt"] = montage_srt_name
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not generate montage SRT sidecar: %s", exc)

            processed.append(montage_clip_dict)
        except Exception as exc:
            processing_warnings.append(f"Montage stitching failed: {exc}")
            logger.exception("Montage stitching failed")

    # 6d. Verify the actual final artifacts. If a late transform is invalid,
    #     try earlier known-good render stages before marking the clip unusable.
    _emit_progress(
        on_progress,
        "quality_assurance",
        "Checking rendered files, streams, durations, subtitles, audio, and sampled frames.",
        current=0,
        total=len(processed),
        candidate_review=candidate_review,
        qa_status="running",
    )
    quality_reports = []
    failed_outputs = []
    verified_clips = []
    editorial_reviews = []
    for clip in list(processed):
        is_montage_clip = bool(clip.get("is_montage"))
        try:
            if is_montage_clip:
                expected_duration = float(clip.get("duration"))
            else:
                # Expect the edit-timeline length (source passages plus any
                # measured narration card), never a bare transcript guess.
                planned_rendered = clip.get("planned_rendered_duration")
                if planned_rendered is None:
                    edit_plan = clip.get("edit_plan") or {}
                    if edit_plan.get("estimated_duration") is not None:
                        planned_rendered = edit_plan.get("estimated_duration")
                        clip["planned_rendered_duration"] = planned_rendered
                expected_duration = (
                    float(planned_rendered) if planned_rendered is not None
                    else float(clip.get("end", 0.0)) - float(clip.get("start", 0.0))
                )
        except (TypeError, ValueError):
            expected_duration = None
        # Fallback renders produced before the narration card are source-only.
        # When QA falls back to one of them, it must be verified against the
        # source-only edit duration — never the narrated total.
        try:
            edit_plan = clip.get("edit_plan") or {}
            if edit_plan.get("source_duration") is not None:
                source_only_duration = float(edit_plan["source_duration"])
            else:
                source_only_duration = (
                    float(clip.get("end", 0.0)) - float(clip.get("start", 0.0))
                )
        except (TypeError, ValueError):
            source_only_duration = None
        narrated_file = os.path.basename(
            (clip.get("ai_narration") or {}).get("file") or ""
        )
        expected_audio = (
            True if is_montage_clip or clip.get("audio_mixed")
            else bool(video_info.get("has_audio"))
        )
        qa_args = {
            "expected_duration": expected_duration,
            "expected_aspect": target_aspect,
            "expected_audio": expected_audio,
            "clip": clip,
            "output_dir": output_dir,
        }
        current_name = clip.get("file")
        current_path = os.path.join(output_dir, current_name or "")
        try:
            qa = verify_rendered_file(current_path, **qa_args)
        except Exception as exc:  # QA failure should be reported, not mask the job result
            qa = {
                "status": "fail",
                "file": current_name,
                "checks": {},
                "warnings": [],
                "errors": [f"Post-render QA could not inspect this file ({type(exc).__name__})."],
            }

        fallbacks = list(clip.get("_render_fallbacks") or [])
        if qa.get("status") == "fail":
            for fallback_name in reversed(fallbacks):
                fallback_base = os.path.basename(fallback_name)
                fallback_path = os.path.join(output_dir, fallback_base)
                fallback_clip = {**clip, "file": fallback_base}
                # A render that predates the narration card is source-only:
                # verify it against the source-only edit duration, not the
                # narrated total the failed final render was measured against.
                is_source_only_render = bool(narrated_file) and fallback_base != narrated_file
                fallback_expected = expected_duration
                if is_source_only_render and source_only_duration is not None:
                    fallback_expected = source_only_duration
                fallback_args = {
                    **qa_args,
                    "clip": fallback_clip,
                    "expected_duration": fallback_expected,
                }
                try:
                    fallback_qa = verify_rendered_file(fallback_path, **fallback_args)
                except Exception:
                    continue
                fallback_checks = fallback_qa.get("checks") or {}
                playable_fallback = all(
                    isinstance(fallback_checks.get(check), dict)
                    and fallback_checks[check].get("ok") is True
                    for check in ("file", "streams", "duration", "audio", "visual_samples")
                )
                if fallback_qa.get("status") != "fail" or playable_fallback:
                    clip["file"] = fallback_clip["file"]
                    if clip.get("hook_file") != clip["file"]:
                        clip.pop("hook_file", None)
                    if is_source_only_render and clip.get("ai_narration"):
                        # The narrated render failed QA and the source-only
                        # clip is being delivered instead: drop the narration
                        # metadata too, so duration/plan data describes the
                        # file that actually ships.
                        if clip.get("edit_plan"):
                            drop_narration(clip["edit_plan"])
                        clip.pop("ai_narration", None)
                        clip.pop("planned_rendered_duration", None)
                        fallback_qa.setdefault("warnings", []).append(
                            "The AI narration card was dropped with the failed "
                            "final render; delivering the verified source-only "
                            "clip."
                        )
                    if clip.get("audio_mixed"):
                        # Every earlier render predates the music/voiceover mix
                        # (the mixed render itself is what failed QA): clear
                        # the mix-only metadata for whichever fallback ships.
                        clip.pop("audio_mixed", None)
                        clip.pop("voiceover_duration", None)
                        fallback_qa.setdefault("warnings", []).append(
                            "The music/voiceover mix was dropped with the "
                            "failed final render; delivering a clip without "
                            "that audio mix."
                        )
                    if fallback_qa.get("status") == "fail":
                        tolerated = list(fallback_qa.get("errors") or [])
                        fallback_qa["tolerated_failures"] = tolerated
                        fallback_qa["errors"] = []
                        fallback_qa.setdefault("warnings", []).extend(
                            f"The restored playable render does not meet this requested check: {error}"
                            for error in tolerated
                        )
                    fallback_qa.setdefault("warnings", []).append(
                        "Final render failed QA; a verified earlier render was restored. Recheck requested overlays/audio."
                    )
                    fallback_qa["status"] = "warning"
                    fallback_qa["fallback_used"] = True
                    superseded_errors = list(qa.get("errors") or [])
                    fallback_qa["superseded_final_render"] = {
                        "status": qa.get("status"),
                        "errors": superseded_errors,
                    }
                    fallback_qa.setdefault("warnings", []).extend(
                        f"The final render failed this check: {error}"
                        for error in superseded_errors
                    )
                    qa = fallback_qa
                    try:
                        if current_path != fallback_path and os.path.exists(current_path):
                            os.remove(current_path)
                    except OSError:
                        pass
                    break

        clip["quality_assurance"] = qa
        clip["qa_status"] = qa.get("status", "fail")
        checks = qa.get("checks") or {}
        # Editorial QA is tracked separately from the render verdict: a clip
        # that fails technical QA still keeps its story review (and vice versa
        # a technical pass is never an editorial pass).
        if clip.get("editorial_qa"):
            editorial_reviews.append({
                "index": clip.get("index"),
                "file": clip.get("file"),
                **clip.get("editorial_qa"),
            })
        # Never silently miss the requested duration: compare the rendered
        # length against the planning target and explain any mismatch.
        rendered_duration = (checks.get("duration") or {}).get("actual_seconds")
        plan_target = (clip.get("edit_plan") or {}).get("target_duration")
        if plan_target is None:
            plan_target = target_duration
        if rendered_duration and plan_target:
            mismatch = duration_mismatch_warning(
                plan_target,
                rendered_duration,
                context=f"Clip {clip.get('index', '?')}",
            )
            if mismatch:
                _add_render_warning(clip, mismatch, processing_warnings)
        if clip.get("vertical_file") in fallbacks:
            if (checks.get("aspect_ratio") or {}).get("ok"):
                clip["vertical_file"] = clip["file"]
            else:
                clip.pop("vertical_file", None)
        if clip.get("hook_file") != clip.get("file"):
            clip.pop("hook_file", None)
        quality_reports.append({
            "index": clip.get("index"),
            "file": clip.get("file"),
            "status": clip["qa_status"],
            "checks": qa.get("checks", {}),
            "warnings": qa.get("warnings", []),
            "errors": qa.get("errors", []),
            "fallback_used": bool(qa.get("fallback_used")),
        })
        _emit_progress(
            on_progress,
            "quality_assurance",
            f"Checked {len(quality_reports)} of {len(processed)} rendered output(s).",
            current=len(quality_reports),
            total=len(processed),
            qa_status="running",
            last_clip_qa_status=clip["qa_status"],
            clip_index=clip.get("index"),
            candidate_review=candidate_review,
        )
        for warning in qa.get("warnings", []) or []:
            _add_render_warning(clip, warning, processing_warnings)
        for error in qa.get("errors", []) or []:
            _add_render_warning(clip, error, processing_warnings)

        if clip["qa_status"] == "fail":
            failed_outputs.append({
                "index": clip.get("index"),
                "title": clip.get("title"),
                "file": clip.get("file"),
                "quality_assurance": qa,
                "editorial_qa": clip.get("editorial_qa"),
            })
            clip.pop("_render_fallbacks", None)
            continue

        # Clean superseded intermediates only after a final/fallback render passes.
        for fallback_name in fallbacks:
            fallback_path = os.path.join(output_dir, os.path.basename(fallback_name))
            if os.path.abspath(fallback_path) == os.path.abspath(os.path.join(output_dir, clip.get("file", ""))):
                continue
            try:
                if os.path.isfile(fallback_path):
                    os.remove(fallback_path)
            except OSError:
                pass
        clip.pop("_render_fallbacks", None)
        verified_clips.append(clip)

    processed = verified_clips
    actual_individual_count = sum(1 for clip in processed if not clip.get("is_montage"))
    if not montage and actual_individual_count < requested_individual_count:
        warning = (
            f"Requested {requested_individual_count} individual clips, but generated "
            f"{actual_individual_count}; {actual_individual_count} passed render QA; "
            "review the per-clip QA details."
        )
        if warning not in processing_warnings:
            processing_warnings.append(warning)
            logger.warning(warning)

    failed_qa_count = sum(report["status"] == "fail" for report in quality_reports)
    if not quality_reports:
        qa_status = "not_run"
    elif failed_qa_count:
        qa_status = "fail"
    elif any(report["status"] == "warning" for report in quality_reports):
        qa_status = "warning"
    else:
        qa_status = "pass"
    quality_assurance = {
        "status": qa_status,
        "checked_clip_count": len(quality_reports),
        "failed_clip_count": failed_qa_count,
        "reports": quality_reports,
        "failed_outputs": failed_outputs,
    }
    _emit_progress(
        on_progress,
        "quality_assurance",
        f"Post-render QA finished with status: {qa_status}.",
        current=len(quality_reports),
        total=len(quality_reports),
        qa_status=qa_status,
        failed_clip_count=failed_qa_count,
        candidate_review=candidate_review,
    )
    _emit_progress(on_progress, "persisting", "Saving clip metadata and library copies.")

    # 7. Persist metadata
    try:
        video_info = probe_video_info(video_path)
    except Exception:  # noqa: BLE001
        video_info = {}
    result = {
        "title": title,
        "source": source,
        "job_id": os.path.basename(output_dir),
        "video_file": os.path.basename(video_path),
        "language": transcript.get("language"),
        "duration": transcript.get("duration"),
        "video_info": video_info,
        "aspect_ratio": aspect_key or ("16:9" if target_aspect is None else "9:16"),
        "clips": processed,
        "requested_clip_count": requested_individual_count,
        "individual_clip_count": actual_individual_count,
        "montage_file": next((item["file"] for item in processed if item.get("is_montage")), None),
        "candidate_review": candidate_review,
        "quality_assurance": quality_assurance,
        "editorial_qa": editorial_summary(editorial_reviews),
        "generation_mode": generation_mode,
        "processing_warnings": list(dict.fromkeys(processing_warnings)),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    write_json(metadata_path, {**result, "transcript": transcript})

    # 8. Save verified clips into the Saved Library (folder per source video).
    if processed:
        try:
            library_path = save_to_library(output_dir, title, result["job_id"], processed)
            result["library_path"] = os.path.relpath(library_path, output_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not save to library: %s", e)
    else:
        logger.info("No clips passed render QA; skipping the Saved Library copy.")

    logger.info(
        "Done in %.1fs — %d clip(s) written to %s",
        result["elapsed_seconds"],
        len(processed),
        output_dir,
    )
    _emit_progress(
        on_progress,
        "completed",
        "Clip generation, QA, and metadata saving are complete.",
        current=len(processed),
        total=len(processed),
        qa_status=qa_status,
        failed_clip_count=failed_qa_count,
        candidate_review=candidate_review,
    )
    return result
