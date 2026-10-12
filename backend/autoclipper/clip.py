"""Cut and validate clips from a source video with FFmpeg."""
import math
import os

from .utils import logger, probe_duration, probe_video_info, run_command


def _as_timestamp(value, name):
    if isinstance(value, bool):
        raise ValueError(f"Clip {name} must be a finite non-negative number.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Clip {name} must be a finite non-negative number.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Clip {name} must be a finite non-negative number.")
    return value


def _validate_rendered_clip(path, expected_duration=None):
    """Reject empty, unprobeable, or severely truncated FFmpeg outputs."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("FFmpeg did not produce a non-empty clip file.")
    info = probe_video_info(path)
    if not info.get("width") or not info.get("height"):
        raise RuntimeError("FFmpeg output has no readable video stream.")
    actual_duration = probe_duration(path)
    if actual_duration <= 0:
        raise RuntimeError("FFmpeg output duration could not be verified.")
    if expected_duration is not None:
        tolerance = max(0.75, float(expected_duration) * 0.08)
        if abs(actual_duration - float(expected_duration)) > tolerance:
            raise RuntimeError(
                f"Rendered clip duration check failed (expected about {expected_duration:.2f}s, "
                f"got {actual_duration:.2f}s)."
            )
    return {**info, "duration": actual_duration}


def cut_clip(video_path: str, start: float, end: float, out_path: str) -> str:
    """Extract [start, end] as an H.264/AAC MP4 and verify the actual output."""
    start = _as_timestamp(start, "start")
    end = _as_timestamp(end, "end")
    if end <= start:
        raise ValueError("Clip end must be later than clip start.")
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError("Source video for clip rendering does not exist.")
    duration = end - start
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        run_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                video_path,
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-sn",
                "-dn",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-x264-params",
                "ref=4:me=hex:subme=7:trellis=1",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                out_path,
            ]
        )
        _validate_rendered_clip(out_path, expected_duration=duration)
    except Exception:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        raise
    return out_path


def cut_all(video_path: str, clips: list, output_dir: str, base_name: str) -> list:
    """Cut every candidate; omit failures instead of exposing partial files."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, clip in enumerate(clips, start=1):
        out_name = f"{base_name}_clip_{i}.mp4"
        out_path = os.path.join(output_dir, out_name)
        logger.info(
            "Cutting clip %d/%d [%.1f-%.1f]: %s",
            i,
            len(clips),
            clip["start"],
            clip["end"],
            clip.get("title", ""),
        )
        try:
            cut_clip(video_path, clip["start"], clip["end"], out_path)
            enriched = {**clip, "index": i, "file": out_name}
            results.append(enriched)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Failed to cut clip %d (%s).", i, type(exc).__name__)
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except OSError:
                pass
    return results
