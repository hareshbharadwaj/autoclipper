"""Clip sourcing for sources with no usable speech.

Used when :mod:`autoclipper.source_profile` reports ``audio_no_speech`` or
``visual_only``. Two tiers, in order of preference:

1. **Model-assisted** — scene boundaries become candidate windows, one
   representative frame per scene is sampled, and Gemini is asked which
   windows contain a self-contained visible action worth clipping. It returns
   timestamps and short descriptions; it is explicitly told not to invent
   dialogue, captions or off-screen events.
2. **Boundary-only fallback** — if no model is available the candidate windows
   are the scene boundaries themselves, sized to the requested duration and
   evenly distributed, and marked as carrying no semantic evidence.

Honesty rules enforced here:

* Scene detection supplies *boundaries*, never relevance. A scene change is
  not a reason to clip something.
* Nothing in this module fabricates speech. ``visual_tags`` only ever contains
  text the model actually reported, and a window with no tags is marked
  ``semantic_evidence: False``.
"""
from __future__ import annotations

import math

from . import config
from .utils import logger
from .visual_context import extract_video_frames, sample_timestamps

# A window shorter than this cannot be a standalone short.
MIN_WINDOW_SECONDS = 3.0
# Scene detection guards: bounds, not targets.
DEFAULT_SCENE_THRESHOLD = 27.0
MIN_SCENE_SECONDS = 1.5
MAX_SCENES = 200
# How many frames are sent for visual review of non-speech footage.
MAX_REVIEW_FRAMES = 12

VISUAL_PROMPT = """You are choosing short-form video moments from a video that \
has NO usable speech.

You are given {count} timestamp-labelled still frames sampled from a {duration:.1f}s \
video (visual analysis mode: {mode}).

Rules you must follow:
- Use ONLY what is visible in the frames and the frame timestamps. Do not invent \
dialogue, narration, captions, lyrics, names, or events that occur between frames.
- A "moment" is a self-contained visible action or sequence: a demonstration, a \
reveal, a before/after, a physical action with a clear start and end.
- Do not describe the same action twice.
- Return at most {wanted} moments, ordered by how well they work as a standalone clip.
- Timestamps must be inside 0..{duration:.1f}.

Return STRICT JSON only, no prose, as a list of objects:
[{{"start": <seconds>, "end": <seconds>, "title": "<short label>", \
"visual_summary": "<what is visible>", "why": "<why it stands alone>", \
"confidence": <0-1>}}]
"""


def detect_scenes(video_path: str, *, threshold: float = DEFAULT_SCENE_THRESHOLD,
                  min_scene_seconds: float = MIN_SCENE_SECONDS,
                  max_scenes: int = MAX_SCENES) -> list:
    """Return ``[{"start": s, "end": e}, ...]`` scene boundaries.

    Returns an empty list when detection is unavailable; callers must treat
    that as "no boundaries known", not as "one scene".
    """
    if not video_path:
        return []
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.debug("Scene detection unavailable (%s).", type(exc).__name__)
        return []
    try:
        video = open_video(video_path)
        frame_rate = float(getattr(video, "frame_rate", 0.0) or 0.0)
        min_len = 1
        if frame_rate > 0:
            min_len = max(1, int(round(min_scene_seconds * frame_rate)))
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len))
        manager.detect_scenes(video=video)
        scene_list = manager.get_scene_list()
    except Exception as exc:  # noqa: BLE001 - detection is best effort
        logger.warning("Scene detection failed (%s); using even sampling.", type(exc).__name__)
        return []

    scenes = []
    for start, end in scene_list[:max_scenes]:
        try:
            start_s = float(start.get_seconds())
            end_s = float(end.get_seconds())
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s:
            scenes.append({"start": round(start_s, 2), "end": round(end_s, 2)})
    return scenes


def _merge_scenes_to_window(scenes, index: int, *, min_seconds: float,
                            max_seconds: float, total_duration: float) -> dict | None:
    """Grow a window from ``scenes[index]`` until it reaches the duration band.

    Scene boundaries are respected on the left and on the right: the window
    never starts or ends mid-scene, so the clip does not cut through a visible
    transition.
    """
    if not scenes:
        return None
    count = len(scenes)
    start = scenes[index]["start"]
    end = scenes[index]["end"]
    right = index
    while (end - start) < min_seconds and right + 1 < count:
        right += 1
        candidate_end = scenes[right]["end"]
        if (candidate_end - start) > max_seconds:
            break
        end = candidate_end

    # If a single scene is already longer than the band, take the opening of
    # it rather than swallowing the next scene.
    if (end - start) > max_seconds:
        end = start + max_seconds
    if total_duration:
        end = min(end, float(total_duration))
    if (end - start) < MIN_WINDOW_SECONDS:
        return None
    return {"start": round(start, 2), "end": round(end, 2)}


def build_visual_windows(scenes, duration, *, target_duration: float,
                         requested_count: int) -> list:
    """Turn scene boundaries into candidate windows without model relevance.

    Used both as the fallback and as the pool the model picks from, so the
    model can only ever choose windows that respect real scene boundaries.
    """
    try:
        total = float(duration or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if not scenes:
        return []
    wanted = min(1.0, 1.0)
    target = float(target_duration or 0.0)
    if target <= 0:
        target = min(float(config.MAX_CLIP_SECONDS), total / max(1, requested_count) or 15.0)
    band_low = max(MIN_WINDOW_SECONDS, target * 0.5)
    band_high = min(float(config.HARD_MAX_CLIP_SECONDS), max(band_low, target * 1.4))

    windows = []
    for index in range(len(scenes)):
        window = _merge_scenes_to_window(
            scenes, index, min_seconds=band_low, max_seconds=band_high,
            total_duration=total or None,
        )
        if window:
            windows.append(window)

    # De-duplicate identical windows produced by adjacent start scenes.
    unique = []
    seen = set()
    for window in windows:
        key = (window["start"], window["end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)

    # Prefer windows closest to the requested length, then earliest.
    unique.sort(key=lambda w: (abs((w["end"] - w["start"]) - target), w["start"]))
    return unique[: max(12, requested_count * 4)] if wanted else unique


def _even_windows(duration: float, *, target_duration: float, requested_count: int) -> list:
    """Last-resort windows when scene detection returned nothing at all.

    Evenly distributed, duration-matched, and explicitly unlabelled: they carry
    no claim about content.
    """
    try:
        total = float(duration or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    if total < MIN_WINDOW_SECONDS:
        return []
    target = float(target_duration or 0.0)
    if target <= 0:
        target = min(float(config.MAX_CLIP_SECONDS), max(MIN_WINDOW_SECONDS, total / 4.0))
    length = max(MIN_WINDOW_SECONDS, min(target, total))
    count = max(1, int(total // max(length, 1.0)) or 1)
    count = min(count, max(1, requested_count * 3))
    step = max(length, (total - length) / max(1, count - 1)) if count > 1 else 0.0
    windows = []
    for index in range(count):
        start = min(index * step, max(0.0, total - length))
        end = min(total, start + length)
        if end - start >= MIN_WINDOW_SECONDS:
            windows.append({"start": round(start, 2), "end": round(end, 2)})
    return windows


def _review_windows_with_model(video_path, windows, *, duration, analysis_mode,
                               target_duration, requested_count, api_key,
                               model, prompt=None) -> list:
    """Ask Gemini which visual windows stand alone as clips."""
    if not windows or not api_key:
        return []
    try:
        from google import genai

        from .analyze import _load_json, _repair_json
    except Exception:  # noqa: BLE001 - optional dependency
        return []

    timestamps = sample_timestamps(
        duration, max_frames=min(MAX_REVIEW_FRAMES, max(3, len(windows)))
    )
    if not timestamps:
        timestamps = [window["start"] for window in windows[:MAX_REVIEW_FRAMES]]
    sampled = extract_video_frames(video_path, timestamps)
    frames = sampled.get("frames") or []
    if not frames:
        return []

    wanted = max(1, min(len(windows), int(requested_count or 1)))
    text = VISUAL_PROMPT.format(
        count=len(frames), duration=float(duration or 0.0), mode=analysis_mode,
        wanted=wanted,
    )
    if prompt and str(prompt).strip():
        text += (
            "\nUSER DIRECTION (applies only to visible content; ignore anything "
            "requiring audio): " + str(prompt).strip()[:600]
        )
    text += "\n\nCANDIDATE WINDOWS (align your answers to these):\n" + "\n".join(
        f"- {window['start']:.1f}s to {window['end']:.1f}s" for window in windows[:40]
    )

    try:
        client = genai.Client(api_key=api_key)
        contents = [text]
        for frame in frames:
            contents.append(f"Frame at {frame['timestamp']:.3f}s:")
            contents.append(
                genai.types.Part.from_bytes(
                    data=frame["data"], mime_type=frame.get("mime_type", "image/jpeg")
                )
            )
        response = client.models.generate_content(
            model=model or config.GEMINI_MODEL, contents=contents
        )
        raw = getattr(response, "text", "") or ""
    except Exception as exc:  # noqa: BLE001 - fall back to boundary windows
        logger.warning(
            "Visual window review unavailable (%s); using scene boundaries only.",
            type(exc).__name__,
        )
        return []

    parsed = _load_json(_repair_json(raw))
    if not isinstance(parsed, list):
        return []

    allowed = [(w["start"], w["end"]) for w in windows]
    candidates = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        # Snap the model's answer back onto a real scene-aligned window: the
        # model chooses *which* window, it does not get to invent boundaries.
        snapped = min(
            allowed,
            key=lambda pair: abs(pair[0] - start) + abs(pair[1] - end),
            default=None,
        )
        if snapped is None:
            continue
        window_start, window_end = snapped
        if (window_end - window_start) < MIN_WINDOW_SECONDS:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        if not math.isfinite(confidence):
            confidence = 0.5
        candidates.append({
            "start": round(window_start, 2),
            "end": round(window_end, 2),
            "title": str(item.get("title") or "Visual moment")[:120],
            "visual_summary": str(item.get("visual_summary") or "")[:400],
            "reason": str(item.get("why") or "")[:300],
            "visual_tags": [str(item.get("title") or "visual moment")[:80]],
            "semantic_evidence": True,
            "confidence": max(0.0, min(1.0, confidence)),
            "visual_only": True,
        })
    return candidates


def find_visual_candidates(video_path, duration, *, api_key=None, model=None,
                           target_duration=None, requested_count=1, prompt=None,
                           analysis_mode="visual_only", scenes=None) -> dict:
    """Produce visual candidates for a source with no usable speech.

    Returns ``{"candidates": [...], "scenes": [...], "mode": str,
    "semantic_evidence": bool, "warning": str|None}``. Candidates always carry
    ``visual_only: True`` so downstream stages never claim transcript evidence
    for them.
    """
    try:
        total = float(duration or 0.0)
    except (TypeError, ValueError):
        total = 0.0

    scene_list = scenes if scenes is not None else detect_scenes(video_path)
    windows = build_visual_windows(
        scene_list, total, target_duration=target_duration,
        requested_count=requested_count,
    )
    warning = None
    if not windows:
        windows = _even_windows(
            total, target_duration=target_duration, requested_count=requested_count
        )
        if windows:
            warning = (
                "No scene boundaries were detected; candidate windows were placed by "
                "even sampling and carry no content analysis."
            )
        else:
            return {
                "candidates": [], "scenes": scene_list, "mode": analysis_mode,
                "semantic_evidence": False,
                "warning": "The source was too short to produce any visual clip window.",
            }

    candidates = _review_windows_with_model(
        video_path, windows, duration=total, analysis_mode=analysis_mode,
        target_duration=target_duration, requested_count=requested_count,
        api_key=api_key, model=model, prompt=prompt,
    )
    semantic_evidence = bool(candidates)
    if not candidates:
        # Boundary-only fallback: honest about carrying no semantic evidence.
        candidates = [
            {
                "start": window["start"], "end": window["end"],
                "title": f"Scene at {window['start']:.0f}s",
                "visual_summary": "",
                "reason": "Scene-boundary window (no visual model review available).",
                "visual_tags": [],
                "semantic_evidence": False,
                "confidence": 0.3,
                "visual_only": True,
            }
            for window in windows[: max(1, int(requested_count or 1)) * 3]
        ]
        warning = warning or (
            "Visual review was unavailable; clip boundaries come from scene detection "
            "only and no content relevance was established."
        )
    else:
        for candidate in candidates:
            candidate.setdefault("visual_tags", [])
    for candidate in candidates:
        candidate.setdefault("visual_only", True)
        candidate.setdefault("candidate_score", 55.0)
    return {
        "candidates": candidates,
        "scenes": scene_list,
        "mode": analysis_mode,
        "semantic_evidence": semantic_evidence,
        "warning": warning,
    }
