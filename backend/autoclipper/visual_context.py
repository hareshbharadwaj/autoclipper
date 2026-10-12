"""Bounded, timestamped video-frame sampling for multimodal review.

Frames are extracted in memory and are never written beside user media or
included in logs. Sampling is deliberately capped by both count and wall time.

Every limit is configurable through the environment but hard-bounded here so a
misconfigured deployment can never request unbounded frames, payloads, or time.
"""
import math
import os
import shutil
import subprocess
import time

from .utils import logger


def _bounded_int_env(name, default, low, high):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _bounded_float_env(name, default, low, high):
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(low, min(high, value))


# Frame-count limits. Short videos still receive a sensible minimum number of
# samples so multi-scene clips are not reduced to a single midpoint frame.
MAX_ANALYSIS_FRAMES = _bounded_int_env("AUTOCLIPPER_MAX_ANALYSIS_FRAMES", 8, 1, 16)
MIN_ANALYSIS_FRAMES = _bounded_int_env(
    "AUTOCLIPPER_MIN_ANALYSIS_FRAMES", min(3, MAX_ANALYSIS_FRAMES), 1, MAX_ANALYSIS_FRAMES
)
# One sample per this many seconds for long videos (above the minimum).
VISUAL_SAMPLE_INTERVAL_SECONDS = _bounded_float_env(
    "AUTOCLIPPER_VISUAL_SAMPLE_INTERVAL_SECONDS", 90.0, 5.0, 900.0
)
# Frames closer together than this collapse (avoids near-duplicates on very
# short videos, which would waste payload without adding visual evidence).
MIN_FRAME_SPACING_SECONDS = 1.0

# Payload and execution-time limits (still bounded after env overrides).
MAX_FRAME_BYTES = _bounded_int_env("AUTOCLIPPER_MAX_FRAME_BYTES", 600_000, 10_000, 2_000_000)
MAX_TOTAL_FRAME_BYTES = _bounded_int_env(
    "AUTOCLIPPER_MAX_TOTAL_FRAME_BYTES",
    max(4_000_000, MAX_FRAME_BYTES),
    max(100_000, MAX_FRAME_BYTES),
    12_000_000,
)
MAX_SAMPLING_SECONDS = _bounded_float_env("AUTOCLIPPER_MAX_SAMPLING_SECONDS", 18.0, 1.0, 120.0)
FRAME_TIMEOUT_SECONDS = _bounded_float_env("AUTOCLIPPER_FRAME_TIMEOUT_SECONDS", 3.0, 0.5, 30.0)
MAX_FRAME_DIMENSION = _bounded_int_env("AUTOCLIPPER_MAX_FRAME_DIMENSION", 640, 128, 1920)


def sample_timestamps(duration, max_frames=None, min_frames=None, interval=None):
    """Choose evenly distributed interior timestamps, with a strict frame cap.

    Long videos get about one frame per `interval` seconds (default 90s) up to
    `max_frames`; short videos are lifted to at least `min_frames` samples
    (default 3) so a sub-90s video with several scenes is not reduced to a
    single midpoint frame. Very short videos collapse so consecutive samples
    stay at least ~1s apart.
    """
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(duration) or duration <= 0:
        return []
    try:
        cap = int(max_frames) if max_frames is not None else MAX_ANALYSIS_FRAMES
    except (TypeError, ValueError):
        cap = MAX_ANALYSIS_FRAMES
    try:
        floor = int(min_frames) if min_frames is not None else MIN_ANALYSIS_FRAMES
    except (TypeError, ValueError):
        floor = MIN_ANALYSIS_FRAMES
    try:
        step = float(interval) if interval is not None else VISUAL_SAMPLE_INTERVAL_SECONDS
    except (TypeError, ValueError):
        step = VISUAL_SAMPLE_INTERVAL_SECONDS
    if cap <= 0:
        return []
    if not math.isfinite(step) or step <= 0:
        step = VISUAL_SAMPLE_INTERVAL_SECONDS

    floor = max(1, min(floor, cap))
    # Very short videos: never sample faster than one frame per second.
    floor = max(1, min(floor, int(duration // MIN_FRAME_SPACING_SECONDS) or 1))
    count = int(math.ceil(duration / step))
    count = max(floor, min(count, cap))
    # Interior points avoid title cards/black end frames while still covering
    # the full timeline, including its opening and conclusion.
    return [duration * (index + 1) / (count + 1) for index in range(count)]


def _extract_one(video_path, timestamp, timeout, ffmpeg):
    """Extract one JPEG through stdout; no temporary frame file is created."""
    filter_expr = (
        f"scale={MAX_FRAME_DIMENSION}:{MAX_FRAME_DIMENSION}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    completed = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{max(0.0, float(timestamp)):.3f}", "-i", video_path,
            "-frames:v", "1", "-vf", filter_expr,
            "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(0.05, timeout),
    )
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


def extract_video_frames(
    video_path,
    timestamps,
    *,
    max_frames=MAX_ANALYSIS_FRAMES,
    total_timeout=MAX_SAMPLING_SECONDS,
    frame_timeout=FRAME_TIMEOUT_SECONDS,
    max_frame_bytes=MAX_FRAME_BYTES,
    max_total_bytes=MAX_TOTAL_FRAME_BYTES,
    ffmpeg_path=None,
):
    """Return timestamped JPEG frame bytes plus a safe, user-facing warning.

    Individual FFmpeg calls and the whole sampling pass have deadlines. Invalid
    files, missing FFmpeg, oversized frames, and partial extraction degrade to
    transcript-only use instead of aborting video analysis.
    """
    requested = []
    for value in list(timestamps or [])[: max(0, int(max_frames))]:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(timestamp) and timestamp >= 0:
            requested.append(timestamp)

    if not requested:
        return {"frames": [], "warning": "No usable video timestamps were available for visual sampling."}
    if not video_path or not os.path.isfile(video_path):
        return {"frames": [], "warning": "Visual frame sampling was unavailable because the local video file could not be read."}

    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg:
        return {"frames": [], "warning": "Visual frame sampling was unavailable because FFmpeg is not installed."}

    frames = []
    total_bytes = 0
    deadline = time.monotonic() + max(0.1, float(total_timeout))
    failed = False
    for timestamp in requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failed = True
            break
        try:
            data = _extract_one(
                video_path,
                timestamp,
                min(float(frame_timeout), remaining),
                ffmpeg,
            )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            logger.warning("Video frame sampling stopped (%s).", type(exc).__name__)
            failed = True
            break
        if not data:
            failed = True
            continue
        if len(data) > max_frame_bytes or total_bytes + len(data) > max_total_bytes:
            failed = True
            break
        frames.append({"timestamp": round(timestamp, 3), "data": data, "mime_type": "image/jpeg"})
        total_bytes += len(data)

    warning = None
    if len(frames) < len(requested) or failed:
        warning = (
            f"Visual frame sampling was partial ({len(frames)} of {len(requested)} frames); "
            "analysis will use the frames that were available and transcript evidence."
            if frames
            else "Visual frame sampling failed; analysis will use transcript evidence only."
        )
    return {"frames": frames, "warning": warning}


def sample_video_frames(video_path, duration, max_frames=None, min_frames=None, interval=None, **kwargs):
    """Sample evenly spaced, bounded visual context from a whole video."""
    if max_frames is None:
        max_frames = MAX_ANALYSIS_FRAMES
    timestamps = sample_timestamps(
        duration, max_frames=max_frames, min_frames=min_frames, interval=interval
    )
    return extract_video_frames(video_path, timestamps, max_frames=max_frames, **kwargs)


def sample_candidate_frames(video_path, candidates, max_frames=None, **kwargs):
    """Sample bounded visual evidence focused on candidate moments.

    Every selected candidate contributes its midpoint first (guaranteeing
    coverage across candidates); any remaining frame budget adds 25%/75%
    context points around those same candidates so reviewers see how each
    moment begins and resolves, not just its middle.
    """
    items = list(candidates or [])
    if max_frames is None:
        max_frames = MAX_ANALYSIS_FRAMES
    try:
        limit = max(0, int(max_frames))
    except (TypeError, ValueError):
        limit = 0
    if not items or limit <= 0:
        return {"frames": [], "warning": None}

    if len(items) <= limit:
        indices = list(range(len(items)))
    elif limit == 1:
        indices = [len(items) // 2]
    else:
        indices = sorted({round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)})

    def _bounds(index):
        try:
            start = float(items[index]["start"])
            end = float(items[index]["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            return None
        return start, end

    requested = []
    owner_by_time = {}

    def _add(timestamp, index):
        if timestamp is None or not math.isfinite(timestamp) or timestamp < 0:
            return
        if len(requested) >= limit:
            return
        rounded = round(timestamp, 3)
        if rounded in owner_by_time:
            return
        requested.append(rounded)
        owner_by_time[rounded] = index

    # Round 1: candidate midpoints (primary visual evidence).
    for index in indices:
        bounds = _bounds(index)
        if bounds:
            _add((bounds[0] + bounds[1]) / 2.0, index)
    # Round 2: additional context frames around the same candidate moments.
    for index in indices:
        bounds = _bounds(index)
        if bounds:
            span = bounds[1] - bounds[0]
            _add(bounds[0] + span * 0.25, index)
            _add(bounds[0] + span * 0.75, index)

    requested.sort()
    result = extract_video_frames(
        video_path, requested, max_frames=limit, total_timeout=12.0, **kwargs
    )
    for frame in result["frames"]:
        frame["candidate_index"] = owner_by_time.get(round(frame["timestamp"], 3), -1)
    return result


def format_timestamp(seconds):
    """Render a non-negative timestamp as HH:MM:SS for multimodal prompts."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
