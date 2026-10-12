"""Bounded, timestamped video-frame sampling for multimodal review.

Frames are extracted in memory and are never written beside user media or
included in logs. Sampling is deliberately capped by both count and wall time.
"""
import math
import os
import shutil
import subprocess
import time

from .utils import logger

MAX_ANALYSIS_FRAMES = 8
MAX_FRAME_BYTES = 600_000
MAX_TOTAL_FRAME_BYTES = 4_000_000
MAX_SAMPLING_SECONDS = 18.0
FRAME_TIMEOUT_SECONDS = 3.0
MAX_FRAME_DIMENSION = 640


def sample_timestamps(duration, max_frames=MAX_ANALYSIS_FRAMES):
    """Choose evenly distributed interior timestamps, with a strict frame cap."""
    try:
        duration = float(duration)
        max_frames = int(max_frames)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(duration) or duration <= 0 or max_frames <= 0:
        return []
    count = min(max_frames, max(1, int(math.ceil(duration / 90.0))))
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


def sample_video_frames(video_path, duration, max_frames=MAX_ANALYSIS_FRAMES, **kwargs):
    """Sample evenly spaced, bounded visual context from a whole video."""
    timestamps = sample_timestamps(duration, max_frames=max_frames)
    return extract_video_frames(video_path, timestamps, max_frames=max_frames, **kwargs)


def sample_candidate_frames(video_path, candidates, max_frames=8, **kwargs):
    """Sample a bounded, evenly distributed set of candidate midpoints."""
    items = list(candidates or [])
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

    requested = []
    owner_by_time = {}
    for index in indices:
        try:
            timestamp = (float(items[index]["start"]) + float(items[index]["end"])) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or timestamp < 0:
            continue
        rounded = round(timestamp, 3)
        requested.append(rounded)
        owner_by_time[rounded] = index

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
