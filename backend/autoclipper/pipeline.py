"""End-to-end orchestration: ingest -> transcribe -> analyze -> cut."""
import json
import math
import os
import time

from . import config
from . import grounding
from .analyze import (
    CandidateReviewError,
    CandidateReviewTimeoutError,
    find_viral_clips, review_clip_candidates, rank_and_diversify_candidates,
    _filter_clips_to_bounds, _resolve_clip_bounds, _validate_clips,
)
from .clip import cut_all
from .ingest import ingest
from .reframe import VERTICAL_ASPECT, reframe_clip
from .subtitles import add_subtitles
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


def _subtitle_repair_needed(qa):
    """True when the only blocking QA problem is the requested captions."""
    checks = qa.get("checks") or {}
    subtitle_check = checks.get("subtitles")
    if not isinstance(subtitle_check, dict) or subtitle_check.get("ok") is not False:
        return False
    required = ("file", "streams", "duration", "audio", "visual_samples")
    return all(
        isinstance(checks.get(name), dict) and checks[name].get("ok") is True
        for name in required
    )


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


def _repair_subtitle_failure(clip, transcript, output_dir, qa, qa_args, fallbacks, processing_warnings):
    """Try once to still deliver the requested captions before falling back/failing.

    Returns the (possibly updated) QA report and fallback list.

    - If no transcript words overlap the clip, captions are impossible; the clip
      is an acceptable subtitle-free fallback, so the failure is downgraded to
      an explicit warning that explains the missing feature.
    - Otherwise retry the caption stage once on the current (otherwise verified)
      render. If the repaired file passes QA it becomes the final render and the
      captionless file is kept as a fallback; if not, the original QA report is
      returned unchanged so the normal fallback/failure handling runs.
    """
    if not _subtitle_repair_needed(qa):
        return qa, fallbacks

    warning = clip.get("subtitle_warning") or ""
    if warning.startswith("No transcript words overlap"):
        remaining = [
            error for error in qa.get("errors") or []
            if not error.lower().startswith("requested subtitles")
        ]
        explanation = (
            "Requested subtitles could not be generated because no transcript words "
            "overlap this clip; it was delivered without captions."
        )
        repaired = {
            **qa,
            "errors": remaining,
            "warnings": list(qa.get("warnings") or []) + [explanation],
        }
        repaired["status"] = "fail" if remaining else "warning"
        return repaired, fallbacks

    try:
        base_name = clip.get("file")
        repair_clip = {**clip}
        repair_clip.pop("subtitled_file", None)
        repair_clip.pop("subtitle_warning", None)
        add_subtitles([repair_clip], transcript, output_dir, burn=True, preserve_source=True)
        repaired_name = repair_clip.get("file")
        if not repaired_name or repaired_name == base_name or repair_clip.get("subtitle_warning"):
            raise RuntimeError(
                repair_clip.get("subtitle_warning")
                or "The caption retry did not produce a burned subtitle file."
            )
        repaired_path = os.path.join(output_dir, os.path.basename(repaired_name))
        if not _nonempty_file(repaired_path):
            raise RuntimeError("The caption retry produced an empty subtitle file.")
        repaired_qa = verify_rendered_file(repaired_path, **{**qa_args, "clip": repair_clip})
        if repaired_qa.get("status") == "fail":
            raise RuntimeError("The repaired caption output failed QA verification.")
        # Adopt the repaired render; keep the captionless file as a fallback.
        fallbacks = list(dict.fromkeys(list(fallbacks) + [base_name]))
        clip["file"] = repaired_name
        clip["subtitled_file"] = repaired_name
        if repair_clip.get("srt"):
            clip["srt"] = repair_clip["srt"]
        clip.pop("subtitle_warning", None)
        repaired_qa.setdefault("warnings", []).append(
            "Captions were re-generated and burned successfully after an initial subtitle failure."
        )
        processing_warnings.append(
            "Subtitle rendering initially failed; captions were repaired on a second attempt."
        )
        logger.info("Subtitle repair succeeded for clip %s.", clip.get("index"))
        return repaired_qa, fallbacks
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Subtitle repair attempt failed for clip %s (%s); continuing with fallback handling.",
            clip.get("index"), type(exc).__name__,
        )
        return qa, fallbacks


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
    """Fit selected angles to complete transcript segments near the target length.

    Segment boundaries come from Whisper timestamps. The per-shot target reserves
    time for the montage crossfades. When no boundary is close enough, the closest
    complete segment is retained (never duplicated or cut mid-sentence); the caller
    reports the actual stitched duration and its deviation from the target.
    """
    segments = []
    for segment in transcript.get("segments", []) or []:
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or not (segment.get("text") or "").strip():
            continue
        if total_duration:
            start = max(0.0, min(start, float(total_duration)))
            end = max(0.0, min(end, float(total_duration)))
        if end > start:
            segments.append({"start": start, "end": end})
    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not segments:
        raise ValueError("Montage needs word/sentence timestamps from the transcript.")
    if not candidates or not 2 <= len(candidates) <= 3:
        raise ValueError("Montage mode requires 2 or 3 selected video moments.")

    target = min(float(config.HARD_MAX_CLIP_SECONDS), max(3.0, float(target_duration)))
    shot_count = len(candidates)
    fade = max(0.0, float(crossfade_seconds))
    # The output duration is sum(segment durations) - fade * (shots - 1).
    desired_source_total = min(
        float(config.HARD_MAX_CLIP_SECONDS), target + fade * (shot_count - 1)
    )
    desired_per_shot = max(3.0, desired_source_total / shot_count)
    hard_per_shot = (float(config.HARD_MAX_CLIP_SECONDS) + fade * (shot_count - 1)) / shot_count

    chosen = []
    for angle in candidates:
        try:
            hint_start = float(angle.get("start", angle.get("start_hint")))
            hint_end = float(angle.get("end", angle.get("end_hint")))
        except (TypeError, ValueError):
            continue
        if hint_end <= hint_start:
            continue
        if total_duration:
            hint_start = max(0.0, min(hint_start, float(total_duration)))
            hint_end = max(0.0, min(hint_end, float(total_duration)))
        if hint_end <= hint_start:
            continue

        # Start on the complete transcript segment containing the angle's opening
        # hint; otherwise use the nearest boundary. This includes the opening setup.
        containing = [
            i for i, segment in enumerate(segments)
            if segment["start"] <= hint_start < segment["end"]
        ]
        start_idx = containing[0] if containing else min(
            range(len(segments)), key=lambda i: abs(segments[i]["start"] - hint_start)
        )
        shot_start = segments[start_idx]["start"]

        # Do not silently trim off the selected angle's hinted ending. The nearest
        # eligible complete boundary is chosen, with the overall hard cap preserved.
        minimum_end_idx = next(
            (i for i in range(start_idx, len(segments)) if segments[i]["end"] >= hint_end),
            len(segments) - 1,
        )
        possible_ends = []
        for end_idx in range(minimum_end_idx, len(segments)):
            shot_end = segments[end_idx]["end"]
            shot_duration = shot_end - shot_start
            if shot_duration <= 0:
                continue
            if shot_duration > hard_per_shot:
                break
            possible_ends.append((shot_duration, shot_end))
        if not possible_ends:
            continue

        shot_duration, shot_end = min(
            possible_ends,
            key=lambda item: (abs(item[0] - desired_per_shot), item[0] > desired_per_shot),
        )
        if any(shot_start < item["end"] and shot_end > item["start"] for item in chosen):
            continue
        chosen.append({
            "start": round(shot_start, 2),
            "end": round(shot_end, 2),
            "title": (angle.get("title") or "Montage moment")[:80],
            "description": (angle.get("description") or "")[:220],
            "hook": (angle.get("hook") or "")[:120],
            "reason": angle.get("reason") or "Selected from whole-video analysis",
            "story_arc": "Selected angle; transcript-aligned montage segment",
            "_user_selected": True,
        })

    chosen.sort(key=lambda item: item["start"])
    if not 2 <= len(chosen) <= 3:
        raise ValueError("Could not find 2-3 distinct, non-overlapping transcript moments for this montage.")
    return chosen


# --- Source-grounded selection helpers ------------------------------------
def _extend_selection_to_target(candidate, transcript, target_duration, total_duration,
                                max_extra_seconds=30.0):
    """Extend a user-selected interval only with topically adjacent material.

    A short, relevant selection is kept honest: instead of padding it with
    unrelated footage to hit the duration target, it may only grow by adding
    transcript segments that are directly contiguous AND topically relevant to
    the selection's own speech. The target is a goal, never a reason to add
    filler. Returns a new candidate dict (the input is not mutated) plus an
    ``extension_note`` explaining any growth, or the unchanged candidate.
    """
    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
        target = float(target_duration)
    except (KeyError, TypeError, ValueError):
        return candidate
    if target <= 0 or (end - start) >= target:
        return candidate

    core_text = grounding.span_text(transcript, start, end)
    if not core_text.strip():
        return candidate

    segments = grounding.segments_in_span(transcript, 0.0, float(total_duration or 0.0) or float("inf"))
    if not segments:
        return candidate

    hard_limit = min(
        float(config.HARD_MAX_CLIP_SECONDS),
        end - start + max(0.0, float(max_extra_seconds)),
        target + 3.0,
    )
    new_start, new_end = start, end
    added_texts = []

    def _relevance(text):
        return grounding.token_overlap(core_text, text)

    # Grow forward first, then backward, only onto contiguous relevant speech.
    for direction in (1, -1):
        while (new_end - new_start) < target:
            boundary = new_end if direction == 1 else new_start
            adjacent = None
            for seg in segments:
                if direction == 1 and abs(seg["start"] - boundary) <= 1.0 and seg["end"] > boundary:
                    adjacent = seg
                    break
                if direction == -1 and abs(seg["end"] - boundary) <= 1.0 and seg["start"] < boundary:
                    adjacent = seg
                    break
            if adjacent is None:
                break
            if _relevance(adjacent["text"]) < 0.15:
                break  # next material is off-topic: stop rather than pad
            if direction == 1:
                candidate_end = adjacent["end"]
                if candidate_end - new_start > hard_limit:
                    break
                new_end = candidate_end
            else:
                candidate_start = adjacent["start"]
                if new_end - candidate_start > hard_limit:
                    break
                new_start = candidate_start
            added_texts.append(adjacent["text"])

    new_start = round(max(0.0, new_start), 2)
    new_end = round(min(float(total_duration or new_end), new_end), 2)
    if new_start == start and new_end == end:
        return candidate
    extended = {**candidate, "start": new_start, "end": new_end}
    extended["extension_note"] = (
        f"Extended the user-selected span from {start:.1f}-{end:.1f}s to "
        f"{new_start:.1f}-{new_end:.1f}s using only contiguous, topically "
        "matching transcript segments."
    )
    return extended


def _apply_factual_grounding(candidates, transcript, user_direction, review_status):
    """Run deterministic factual-grounding QA over every candidate.

    AI-generated candidates that fail critical grounding (fabricated evidence,
    meaning-flipping cuts, or clear off-topic selection) are rejected; user-
    selected candidates are retained but flagged. Every candidate carries its
    factual record downstream so no clip is ever presented as verified when it
    was not. Returns ``(kept, rejected)``.
    """
    kept = []
    rejected = []
    for candidate in candidates or []:
        # Defense in depth for visual honesty: even if a review stage failed to
        # strip a visual claim, a candidate with no sampled frames can never
        # carry an assertion that something was seen.
        if not candidate.get("visual_frames_attached"):
            visual_evidence = (candidate.get("visual_evidence") or "").strip()
            if visual_evidence and "not verifiable" not in visual_evidence.casefold():
                candidate["visual_evidence"] = "not verifiable from sampled frames"
        record = grounding.ground_candidate(
            candidate,
            transcript,
            user_direction=user_direction,
            review_state=review_status,
            frames_attached=bool(candidate.get("visual_frames_attached")),
        )
        candidate["factual_qa"] = record
        if record["status"] == "failed" and not candidate.get("_user_selected"):
            rejected.append({
                "title": candidate.get("title"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "reasons": list(record["critical"]),
            })
            logger.info(
                "Rejected candidate %.1f-%.1f on factual grounding: %s",
                candidate.get("start", 0.0), candidate.get("end", 0.0),
                "; ".join(record["critical"]),
            )
            continue
        kept.append(candidate)
    return kept, rejected


def _duration_tolerance(target_duration):
    """Documented target-duration tolerance for individual clips."""
    return _montage_duration_tolerance(target_duration)


def _attach_provenance(clip, transcript, target_duration):
    """Attach source-traceability and an editorial QA record to one clip.

    Adds ``source_intervals``, the exact ``transcript_span`` used, requested vs
    planned duration with an honest note, and a deterministic editorial QA
    record (opening/ending alignment, duration fit). Model-dependent editorial
    judgment (relevance quality, pacing) is reported via the review/factual
    records, never claimed here.
    """
    try:
        start = float(clip["start"])
        end = float(clip["end"])
    except (KeyError, TypeError, ValueError):
        return clip
    duration = end - start
    clip["source_intervals"] = [{"start": round(start, 2), "end": round(end, 2)}]
    clip["transcript_span"] = grounding.span_text(transcript, start, end)[:1600]
    clip["requested_duration"] = (
        round(float(target_duration), 2) if target_duration is not None else None
    )
    clip["planned_duration"] = round(duration, 2)

    alignment = grounding.boundary_alignment(transcript, start, end)
    editorial_warnings = []
    if not alignment["start_aligned"]:
        editorial_warnings.append("Opening may begin mid-thought (not sentence-aligned).")
    if not alignment["end_aligned"]:
        editorial_warnings.append("Ending may cut a sentence (not sentence-aligned).")

    duration_note = None
    if target_duration is not None:
        tolerance = _duration_tolerance(target_duration)
        delta = duration - float(target_duration)
        clip["duration_delta"] = round(delta, 2)
        if abs(delta) > tolerance:
            direction = "shorter" if delta < 0 else "longer"
            duration_note = (
                f"Planned duration is {duration:.1f}s versus the "
                f"{float(target_duration):.0f}s target ({delta:+.1f}s, {direction}). "
                "The source did not provide enough relevant material to match the "
                "target without adding unrelated footage or cutting speech."
            )
            editorial_warnings.append(duration_note)

    factual = clip.get("factual_qa") or {}
    editorial_status = "warning" if editorial_warnings else "passed"
    if factual.get("status") == "failed":
        editorial_status = "failed"
    clip["editorial_qa"] = {
        "status": editorial_status,
        "warnings": editorial_warnings,
        "opening_aligned": alignment["start_aligned"],
        "ending_aligned": alignment["end_aligned"],
        "duration_note": duration_note,
        "deterministic": True,
    }
    return clip


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


def run(
    source: str,
    output_dir: str = "output",
    groq_key: str = None,
    gemini_key: str = None,
    whisper_model: str = None,
    gemini_model: str = None,
    vertical: bool = True,
    use_yolo: bool = True,
    subtitles: bool = True,
    split_screen: bool = False,
    force_hd: bool = False,
    cookies_text: str = None,
    clip_count: int = None,
    min_clip: float = None,
    max_clip: float = None,
    content_type: str = None,
    target_duration: float = None,
    custom_prompt: str = None,
    selected_clip: dict = None,
    aspect_ratio: str = None,
    add_hook: bool = False,
    montage: bool = False,
    selected_clips: list = None,
    voiceover_text: str = None,
    voiceover_engine: str = "auto",
    voiceover_voice: str = None,
    background_music_path: str = None,
    music_volume: float = 0.18,
    cached_transcript_path: str = None,
    on_clip=None,
    on_progress=None,
) -> dict:
    """Run the full auto-clip pipeline and return a result summary dict.

    If *on_clip* is provided it will be called with ``(clip_dict, index, total)``
    after each clip has been rendered, before final QA. *on_progress*, when set,
    receives structured stage and per-clip QA events for progress polling.
    """
    # Clamp oversized requests and validate count before any ingest/model work.
    target_duration = _normalize_target_duration(target_duration)
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

    # Topic direction used by deterministic grounding: the user's own words.
    user_direction = (custom_prompt or "").strip()
    if selected_clip and not user_direction:
        user_direction = " ".join(
            str(selected_clip.get(key) or "")
            for key in ("title", "hook", "description")
        ).strip()

    if montage:
        candidate_pool = _fit_montage_candidates(
            selected_clips, transcript,
            target_duration=float(target_duration or 40),
            total_duration=video_dur,
        )
    elif selected_clip:
        # Keep the selected angle as a real output whenever its timestamps are
        # valid. It is validated against timestamp integrity and the hard
        # ceiling only — the target-duration window must never silently drop a
        # user's explicit selection and force the generator to pad with
        # unrelated footage (the knife-safety 91-117s -> 84-144s failure).
        selected_candidates = _validate_clips(
            [{**selected_clip, "_user_selected": True}],
            total_duration=video_dur,
            min_clip=3.0,
            max_clip=float(config.HARD_MAX_CLIP_SECONDS),
            exempt_user_selected=True,
        )
        if selected_candidates and target_duration is not None:
            chosen = selected_candidates[0]
            chosen_duration = float(chosen["end"]) - float(chosen["start"])
            if chosen_duration < float(target_duration) - _duration_tolerance(target_duration):
                extended = _extend_selection_to_target(
                    chosen, transcript, target_duration, video_dur
                )
                if extended is not chosen:
                    selected_candidates = [extended]
                    if extended.get("extension_note"):
                        candidate_warnings.append(extended["extension_note"])
                else:
                    candidate_warnings.append(
                        f"The selected angle spans {chosen_duration:.1f}s; the "
                        f"{float(target_duration):.0f}s target needs more relevant "
                        "material than the source provides around it. The shorter, "
                        "accurate selection was kept instead of padding with "
                        "unrelated footage."
                    )
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
                hint_start = selected_clip.get("start_hint") or selected_clip.get("start")
                hint_end = selected_clip.get("end_hint") or selected_clip.get("end")
                hint_note = ""
                try:
                    hint_note = (
                        f" The original hints pointed near "
                        f"{float(hint_start):.0f}-{float(hint_end):.0f}s."
                    )
                except (TypeError, ValueError):
                    hint_note = ""
                direction = (
                    f"Use the selected angle '{angle_title}' as creative direction "
                    f"where it fits, but return valid transcript-aligned moments."
                    f"{hint_note}"
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
                "failure_reason": None,
            }
        except CandidateReviewTimeoutError as exc:
            logger.warning("Independent candidate review timed out; candidates remain unreviewed.")
            candidate_review = {
                "status": "timed_out",
                "reviewed_count": 0,
                "rejected_count": 0,
                "failure_reason": str(exc),
            }
            candidate_warnings.append(
                "Independent candidate review timed out; the candidates below are UNREVIEWED. "
                "They still passed deterministic factual checks, but no model review confirmed them."
            )
        except CandidateReviewError as exc:
            status = "malformed" if exc.category in ("malformed", "invalid") else "unavailable"
            logger.warning(
                "Independent candidate review failed (%s: %s); candidates remain unreviewed.",
                exc.category, exc,
            )
            candidate_review = {
                "status": status,
                "reviewed_count": 0,
                "rejected_count": 0,
                "failure_reason": str(exc),
            }
            candidate_warnings.append(
                f"Independent candidate review could not be completed ({exc.category}); "
                "the candidates below are UNREVIEWED. They still passed deterministic "
                "factual checks, but no model review confirmed them."
            )
        except Exception as exc:  # noqa: BLE001 - unexpected provider failure
            logger.warning(
                "Independent candidate review unavailable (%s); candidates remain unreviewed.",
                type(exc).__name__,
            )
            candidate_review = {
                "status": "unavailable",
                "reviewed_count": 0,
                "rejected_count": 0,
                "failure_reason": f"{type(exc).__name__}",
            }
            candidate_warnings.append(
                "Independent candidate review could not be completed; the candidates "
                "below are UNREVIEWED. They still passed deterministic factual checks."
            )
    elif candidate_pool:
        candidate_review = {
            "status": "skipped_no_key", "reviewed_count": 0, "rejected_count": 0,
            "failure_reason": None,
        }

    # Deterministic factual grounding runs ALWAYS, reviewed or not. "Review
    # unavailable" is never equivalent to "review passed": unreviewed candidates
    # must still prove their evidence quotes, boundaries, and topic relevance.
    if candidate_pool:
        review_state_for_grounding = candidate_review.get("status", "not_run")
        grounded_pool, rejected_candidates = _apply_factual_grounding(
            candidate_pool,
            transcript,
            user_direction=user_direction if not montage else None,
            review_status=review_state_for_grounding,
        )
        if rejected_candidates:
            candidate_review["grounding_rejected"] = rejected_candidates
            logger.info(
                "Deterministic grounding rejected %d candidate(s) before rendering.",
                len(rejected_candidates),
            )
        candidate_pool = grounded_pool

    # Collect user-visible warnings from here on (candidate review, grounding,
    # duration deviations, render/QA stages).
    processing_warnings = list(candidate_warnings)

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

    # Apply hard duration bounds without blindly padding with unrelated footage or
    # truncating a sentence/payoff. Candidate selection should respect the bounds.
    if effective_min_clip is not None or effective_max_clip is not None or requested_individual_count:
        constraint_count = len(clip_candidates) if montage else requested_individual_count
        clip_candidates = _enforce_clip_constraints(
            clip_candidates, effective_min_clip, effective_max_clip,
            constraint_count, video_dur,
            trim_long=False, pad_short=False,
        )
        if not montage:
            # User-selected moments are exempt from the duration window: a
            # shorter accurate selection beats a padded one (RC1/RC2).
            exempt = [c for c in clip_candidates if c.get("_user_selected")]
            filterable = [c for c in clip_candidates if not c.get("_user_selected")]
            filtered = _filter_clips_to_bounds(
                filterable,
                min_clip=effective_min_clip,
                max_clip=effective_max_clip,
                min_tolerance=0.0 if target_duration is not None else 2.0,
                max_tolerance=0.0 if target_duration is not None else 3.0,
            )
            if not filtered and filterable:
                # The strict window rejected every generated candidate. Rather
                # than returning nothing, keep the best candidate that fits the
                # hard ceiling and report the deviation honestly.
                within_hard_max = [
                    c for c in filterable
                    if float(c["end"]) - float(c["start"]) <= float(config.HARD_MAX_CLIP_SECONDS)
                ]
                if within_hard_max:
                    filtered = sorted(
                        within_hard_max,
                        key=lambda c: -float(c.get("quality_score", c.get("candidate_score", 0.0))),
                    )[:requested_individual_count]
                    processing_warnings.append(
                        "No generated candidate matched the requested duration window; "
                        "the closest transcript-grounded candidate was kept and the "
                        "duration deviation is reported per clip."
                    )
            clip_candidates = exempt + filtered
            clip_candidates.sort(key=lambda c: float(c.get("start", 0.0)))
        logger.info("After enforcement: %d clip(s)", len(clip_candidates))
    if not clip_candidates:
        logger.warning("No viral clips detected.")

    # Attach source provenance + deterministic editorial QA to every planned
    # clip BEFORE rendering, so timestamp traceability can never be lost
    # between analysis and the final file.
    for candidate in clip_candidates:
        _attach_provenance(
            candidate, transcript,
            None if montage else target_duration,
        )
        if candidate.get("_user_selected"):
            candidate["user_selected"] = True
            candidate.pop("_user_selected", None)
        if candidate.get("editorial_qa", {}).get("duration_note"):
            processing_warnings.append(candidate["editorial_qa"]["duration_note"])

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

        # 6. Generate/burn captions. Preserve the uncaptained video for fallback.
        if subtitles:
            prior_name = clip["file"]
            add_subtitles([clip], transcript, output_dir, burn=True, preserve_source=True)
            if clip.get("file") != prior_name:
                fallbacks.append(prior_name)
            if clip.get("subtitle_warning"):
                _add_render_warning(clip, clip["subtitle_warning"], processing_warnings)

        # 6b. Burn the hook banner only if a complete output is produced and the
        #     hook text is actually grounded in this clip's transcript span.
        #     A fabricated quotation must never be burned onto the footage.
        if add_hook and clip.get("hook"):
            hook_check = grounding.check_generated_text(
                clip["hook"], transcript, clip.get("start"), clip.get("end")
            )
            if hook_check["status"] == "failed":
                _add_render_warning(
                    clip,
                    "Hook overlay was skipped: the generated hook text is not "
                    "supported by this clip's transcript ("
                    + "; ".join(hook_check["notes"][:1]) + ").",
                    processing_warnings,
                )
                clip["hook_skipped_ungrounded"] = True
            else:
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
            montage_segments = [
                {
                    "order": position + 1,
                    "file": item.get("file"),
                    "source_start": item.get("start"),
                    "source_end": item.get("end"),
                    "title": item.get("title"),
                }
                for position, item in enumerate(processed)
                if not item.get("is_montage")
            ]
            # Stitching creates a NEW editorial sequence: the shots are not
            # continuous in the source and must never be presented as one event.
            montage_gaps = []
            for previous, following in zip(montage_segments, montage_segments[1:]):
                try:
                    gap = float(following["source_start"]) - float(previous["source_end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if gap > 1.0:
                    montage_gaps.append(round(gap, 1))
            montage_editorial = {
                "status": "warning" if montage_gaps else "passed",
                "warnings": (
                    [
                        "The montage stitches separate source moments "
                        f"(gaps of {', '.join(str(g) + 's' for g in montage_gaps)}); "
                        "they are not continuous in the original video."
                    ] if montage_gaps else []
                ),
                "deterministic": True,
            }
            processed.append({
                "index": len(processed) + 1,
                "file": montage_name,
                "title": f"{title} — Montage",
                "start": 0.0,
                "end": round(montage_duration, 2),
                "duration": round(montage_duration, 2),
                "target_duration": float(target_duration or 40),
                "duration_delta": round(duration_delta, 2),
                "is_montage": True,
                "source_segments": montage_segments,
                "editorial_qa": montage_editorial,
                "reason": "Stitched from selected complementary moments with video/audio crossfades.",
            })
        except Exception as exc:
            processing_warnings.append(f"Montage stitching failed: {exc}")
            logger.exception("Montage stitching failed")

    # Plan C: optional narration and/or background music. For multiple normal clips,
    # apply the same requested audio treatment to the first clip only and say so.
    audio_target = None
    if montage:
        if processed and processed[-1].get("is_montage"):
            audio_target = processed[-1]
        elif voiceover_text or background_music_path:
            processing_warnings.append(
                "Voiceover/music was skipped because montage stitching failed."
            )
    elif len(processed) == 1:
        audio_target = processed[0]
    elif voiceover_text or background_music_path:
        audio_target = processed[0] if processed else None
        if len(processed) > 1:
            processing_warnings.append(
                "Voiceover/music was applied to the first generated clip only; enable montage to mix one combined output."
            )
    if audio_target and (voiceover_text or background_music_path):
        _emit_progress(on_progress, "audio", "Mixing requested voiceover and/or background music.")
        try:
            from .audio_mix import mix_audio_layers
            voiceover_path = None
            if voiceover_text and voiceover_text.strip():
                from .voiceover import generate_voiceover
                voiceover_path = os.path.join(output_dir, f"{base_name}_voiceover.wav")
                generate_voiceover(
                    voiceover_text.strip()[:5000],
                    voiceover_path,
                    engine=voiceover_engine or "auto",
                    voice=voiceover_voice,
                    gemini_key=gemini_key,
                )
            src = os.path.join(output_dir, audio_target["file"])
            mixed_name = audio_target["file"].rsplit(".", 1)[0] + "_audio.mp4"
            mixed_path = os.path.join(output_dir, mixed_name)
            if voiceover_path and os.path.isfile(voiceover_path):
                voice_duration = probe_duration(voiceover_path)
                output_duration = probe_duration(src)
                if voice_duration > 0 and output_duration > 0:
                    audio_target["voiceover_duration"] = round(voice_duration, 2)
                    if voice_duration < output_duration - max(2.0, output_duration * 0.05):
                        warning = (
                            f"Voiceover is {voice_duration:.1f}s for a {output_duration:.1f}s output. "
                            "It plays once from the start; it is not stretched or repeated. "
                            "Extend the script if narration should cover more of the video."
                        )
                        processing_warnings.append(warning)
                        logger.info(warning)
            mix_audio_layers(
                src, mixed_path,
                voiceover_path=voiceover_path,
                music_path=background_music_path,
                music_volume=music_volume,
            )
            if _nonempty_file(mixed_path):
                audio_target.setdefault("_render_fallbacks", []).append(audio_target["file"])
                audio_target["file"] = mixed_name
                audio_target["audio_mixed"] = True
            else:
                raise RuntimeError("Audio mix did not produce a non-empty output file.")
        except Exception as exc:
            warning = f"Voiceover/music mixing failed ({type(exc).__name__}); the unmixed clip was retained."
            processing_warnings.append(warning)
            logger.exception("Voiceover/music mixing failed")

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
    for clip in list(processed):
        is_montage_clip = bool(clip.get("is_montage"))
        try:
            expected_duration = (
                float(clip.get("duration")) if is_montage_clip
                else float(clip.get("end", 0.0)) - float(clip.get("start", 0.0))
            )
        except (TypeError, ValueError):
            expected_duration = None
        expected_audio = (
            True if is_montage_clip or clip.get("audio_mixed")
            else bool(video_info.get("has_audio"))
        )
        qa_args = {
            "expected_duration": expected_duration,
            "expected_aspect": target_aspect,
            "expected_audio": expected_audio,
            "subtitles_requested": bool(subtitles and not is_montage_clip),
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
        if qa.get("status") == "fail" and subtitles and not is_montage_clip:
            # Requested captions must be delivered, repaired, explicitly
            # downgraded (impossible case), or fail — never pass silently.
            qa, fallbacks = _repair_subtitle_failure(
                clip, transcript, output_dir, qa, qa_args, fallbacks, processing_warnings
            )
            clip["_render_fallbacks"] = list(dict.fromkeys(fallbacks))
            current_name = clip.get("file")
            current_path = os.path.join(output_dir, current_name or "")
        if qa.get("status") == "fail":
            for fallback_name in reversed(fallbacks):
                fallback_path = os.path.join(output_dir, os.path.basename(fallback_name))
                fallback_clip = {**clip, "file": os.path.basename(fallback_name)}
                if fallback_clip.get("subtitled_file") != fallback_clip["file"]:
                    fallback_clip.pop("subtitled_file", None)
                fallback_args = {**qa_args, "clip": fallback_clip}
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
                    if fallback_clip.get("subtitled_file") == clip["file"]:
                        clip["subtitled_file"] = clip["file"]
                    elif not (fallback_qa.get("checks", {}).get("subtitles") or {}).get("ok"):
                        clip.pop("subtitled_file", None)
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
        if clip.get("vertical_file") in fallbacks:
            if (checks.get("aspect_ratio") or {}).get("ok"):
                clip["vertical_file"] = clip["file"]
            else:
                clip.pop("vertical_file", None)
        if clip.get("hook_file") != clip.get("file"):
            clip.pop("hook_file", None)
        if (checks.get("subtitles") or {}).get("ok"):
            clip["subtitled_file"] = clip["file"]
        elif clip.get("subtitled_file") in fallbacks and clip.get("subtitled_file") != clip.get("file"):
            clip.pop("subtitled_file", None)
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

    # Source-of-truth record: every editing decision must be traceable back to
    # this asset and its transcript.
    transcript_words = transcript.get("words") or []
    source_asset = {
        "file": os.path.basename(video_path),
        "title": title,
        "duration": video_info.get("duration") or transcript.get("duration"),
        "width": video_info.get("width"),
        "height": video_info.get("height"),
        "has_audio": bool(video_info.get("has_audio")),
        "probe_status": "probed" if video_info.get("width") else "probe_failed",
        "transcript": {
            "language": transcript.get("language"),
            "segment_count": len(transcript.get("segments") or []),
            "word_count": len(transcript_words),
            "provider_confidence_available": any(
                isinstance(word, dict) and word.get("probability") is not None
                for word in transcript_words
            ),
            "cached": bool(cached_transcript_path),
        },
    }

    # Three independent validation tracks. A technical pass never implies an
    # editorial or factual pass; each track reports its own status and reasons.
    def _aggregate_validation(track):
        statuses = []
        for clip in processed:
            record = clip.get(track)
            if isinstance(record, dict) and record.get("status"):
                statuses.append(record["status"])
        if not statuses:
            return "not_run"
        if any(status == "failed" for status in statuses):
            return "fail"
        if any(status == "warning" for status in statuses):
            return "warning"
        return "pass"

    validation_summary = {
        "technical": qa_status,
        "editorial": _aggregate_validation("editorial_qa"),
        "factual": _aggregate_validation("factual_qa"),
        "note": (
            "technical = rendered file checks; editorial = structure/duration/"
            "boundary checks (deterministic); factual = source-grounding checks "
            "(deterministic) plus any independent model review. A pass in one "
            "track does not imply a pass in the others."
        ),
    }

    result = {
        "title": title,
        "source": source,
        "job_id": os.path.basename(output_dir),
        "video_file": os.path.basename(video_path),
        "language": transcript.get("language"),
        "duration": transcript.get("duration"),
        "video_info": video_info,
        "source_asset": source_asset,
        "aspect_ratio": aspect_key or ("16:9" if target_aspect is None else "9:16"),
        "clips": processed,
        "requested_clip_count": requested_individual_count,
        "individual_clip_count": actual_individual_count,
        "montage_file": next((item["file"] for item in processed if item.get("is_montage")), None),
        "candidate_review": candidate_review,
        "quality_assurance": quality_assurance,
        "validation_summary": validation_summary,
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
