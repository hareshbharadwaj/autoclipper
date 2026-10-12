"""Structured checks for rendered clip files and their expected outputs."""
import json
import math
import os
import re
import shutil
import subprocess

from . import config
from .utils import probe_video_info
from .visual_context import extract_video_frames

_SRT_TIME = re.compile(
    r"^(\d+):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2}),(\d{3})$"
)


def _time_seconds(hours, minutes, seconds, millis):
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _read_srt(path, expected_duration):
    if not path or not os.path.isfile(path):
        return {"present": False, "valid": False, "cue_count": 0}
    try:
        if os.path.getsize(path) == 0:
            return {"present": True, "valid": False, "cue_count": 0}
        with open(path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
    except (OSError, UnicodeError):
        return {"present": True, "valid": False, "cue_count": 0}

    rows = []
    invalid_cue = False
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((index for index, line in enumerate(lines) if _SRT_TIME.match(line)), None)
        if time_index is None:
            invalid_cue = True
            continue
        match = _SRT_TIME.match(lines[time_index])
        values = match.groups()
        start = _time_seconds(*values[:4])
        end = _time_seconds(*values[4:])
        if any(int(values[index]) >= 60 for index in (1, 2, 5, 6)):
            invalid_cue = True
        if not any(line for line in lines[time_index + 1:]):
            invalid_cue = True
        rows.append((start, end))
    valid = bool(rows) and not invalid_cue and all(
        0 <= start < end <= float(expected_duration or 0) + 0.15
        for start, end in rows
    )
    return {"present": True, "valid": valid, "cue_count": len(rows)}


def _nonempty_file(path):
    try:
        return bool(path and os.path.isfile(path) and os.path.getsize(path) > 0)
    except OSError:
        return False


def _probe_streams(path, timeout=12.0):
    """Probe actual container streams; use the existing FFmpeg fallback if needed."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe, "-v", "error", "-show_streams", "-show_format",
                    "-of", "json", path,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                payload = json.loads(result.stdout or "{}")
                streams = payload.get("streams") or []
                video = next((s for s in streams if s.get("codec_type") == "video"), None)
                audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
                duration_values = []
                for raw in [
                    (payload.get("format") or {}).get("duration"),
                    (video or {}).get("duration"),
                    (audio or {}).get("duration"),
                ]:
                    try:
                        value = float(raw)
                        if math.isfinite(value) and value > 0:
                            duration_values.append(value)
                    except (TypeError, ValueError):
                        continue
                return {
                    "video": video is not None,
                    "audio": audio is not None,
                    "video_codec": (video or {}).get("codec_name"),
                    "audio_codec": (audio or {}).get("codec_name"),
                    "width": int((video or {}).get("width") or 0),
                    "height": int((video or {}).get("height") or 0),
                    "duration": max(duration_values, default=0.0),
                    "probe_method": "ffprobe",
                    "probe_warning": None,
                }
        except (OSError, subprocess.TimeoutExpired, ValueError, json.JSONDecodeError):
            pass

    info = probe_video_info(path)
    return {
        "video": bool(info.get("width") and info.get("height")),
        "audio": bool(info.get("has_audio")),
        "video_codec": None,
        "audio_codec": None,
        "width": int(info.get("width") or 0),
        "height": int(info.get("height") or 0),
        "duration": float(info.get("duration") or 0.0),
        "probe_method": "ffmpeg_fallback",
        "probe_warning": "FFprobe was unavailable or could not parse this file; fallback media probing was used.",
    }


def _ratio_value(ratio):
    if ratio is None:
        return None
    if isinstance(ratio, str):
        if ratio in {"9:16", "1:1", "16:9"}:
            width, height = ratio.split(":", 1)
            return float(width) / float(height)
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            return None
    try:
        value = float(ratio)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def verify_rendered_file(
    path,
    *,
    expected_duration=None,
    expected_aspect=None,
    expected_audio=None,
    subtitles_requested=False,
    clip=None,
    output_dir=None,
    max_visual_samples=3,
):
    """Return a machine-readable QA report for the actual file on disk.

    Subtitle verification confirms a valid in-range SRT and that the successful
    burn stage produced a file; it does not claim OCR-based pixel-level text
    recognition. Visual checks decode a few bounded frames from the final file.
    """
    clip = clip or {}
    report = {
        "status": "fail",
        "file": os.path.basename(path or "") or None,
        "checks": {},
        "warnings": [],
        "errors": [],
    }
    if not path or not os.path.isfile(path):
        report["checks"]["file"] = {"ok": False, "exists": False, "size_bytes": 0}
        report["errors"].append("Rendered output file is missing.")
        return report

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    file_ok = size > 0
    report["checks"]["file"] = {"ok": file_ok, "exists": True, "size_bytes": size}
    if not file_ok:
        report["errors"].append("Rendered output file is empty.")
        report["status"] = "fail"
        return report

    streams = _probe_streams(path)
    report["checks"]["streams"] = {
        "ok": bool(streams["video"]),
        "video": bool(streams["video"]),
        "audio": bool(streams["audio"]),
        "video_codec": streams["video_codec"],
        "audio_codec": streams["audio_codec"],
        "width": streams["width"],
        "height": streams["height"],
        "probe_method": streams["probe_method"],
    }
    if not streams["video"]:
        report["errors"].append("Output has no readable video stream.")
    if streams.get("probe_warning"):
        report["warnings"].append(streams["probe_warning"])

    actual_duration = float(streams.get("duration") or 0.0)
    duration_ok = actual_duration > 0
    duration_check = {
        "ok": duration_ok,
        "expected_seconds": None,
        "actual_seconds": round(actual_duration, 3) if actual_duration > 0 else None,
        "delta_seconds": None,
        "tolerance_seconds": None,
    }
    try:
        expected = float(expected_duration) if expected_duration is not None else None
    except (TypeError, ValueError):
        expected = None
    if expected is not None and math.isfinite(expected) and expected > 0 and actual_duration > 0:
        tolerance = max(0.75, expected * 0.08)
        delta = actual_duration - expected
        duration_ok = abs(delta) <= tolerance and actual_duration <= float(config.HARD_MAX_CLIP_SECONDS) + 0.25
        duration_check.update({
            "ok": duration_ok,
            "expected_seconds": round(expected, 3),
            "delta_seconds": round(delta, 3),
            "tolerance_seconds": round(tolerance, 3),
        })
    report["checks"]["duration"] = duration_check
    if not duration_ok:
        report["errors"].append("Rendered duration is missing, outside tolerance, or above the hard clip limit.")

    aspect = _ratio_value(expected_aspect)
    actual_aspect = (
        streams["width"] / float(streams["height"])
        if streams.get("width") and streams.get("height") else None
    )
    aspect_ok = True
    if aspect is not None:
        aspect_ok = actual_aspect is not None and abs(actual_aspect - aspect) <= max(0.025, aspect * 0.04)
    report["checks"]["aspect_ratio"] = {
        "ok": aspect_ok,
        "expected": round(aspect, 5) if aspect is not None else None,
        "actual": round(actual_aspect, 5) if actual_aspect is not None else None,
        "width": streams.get("width"),
        "height": streams.get("height"),
    }
    if not aspect_ok:
        report["errors"].append("Output dimensions do not match the requested aspect ratio.")

    audio_ok = expected_audio is None or bool(streams["audio"]) == bool(expected_audio)
    report["checks"]["audio"] = {
        "ok": audio_ok,
        "expected": bool(expected_audio) if expected_audio is not None else None,
        "actual": bool(streams["audio"]),
    }
    if not audio_ok:
        report["errors"].append(
            "Expected source audio is missing from the rendered output."
            if expected_audio else "Unexpected audio stream was found in the rendered output."
        )

    if subtitles_requested:
        srt_name = clip.get("srt")
        srt_path = os.path.join(output_dir or os.path.dirname(path), os.path.basename(srt_name)) if srt_name else None
        srt_check = _read_srt(srt_path, expected or actual_duration)
        burned_name = clip.get("subtitled_file")
        burned_path = os.path.join(output_dir or os.path.dirname(path), os.path.basename(burned_name)) if burned_name else None
        burned_present = _nonempty_file(burned_path)
        captions_ok = bool(srt_check["present"] and srt_check["valid"] and burned_present)
        subtitle_check = {
            **srt_check,
            "burned_output_present": burned_present,
            "ok": captions_ok,
            "verification_method": "SRT cue range plus successful burned-output artifact; no OCR performed.",
        }
        report["checks"]["subtitles"] = subtitle_check
        if not captions_ok:
            warning = clip.get("subtitle_warning") or (
                "Caption QA could not confirm a valid in-range SRT and burned caption output."
            )
            report["warnings"].append(warning)

    visual_timestamps = []
    if actual_duration > 0:
        count = max(1, min(int(max_visual_samples or 3), 3))
        visual_timestamps = sorted({
            round(actual_duration * (index + 1) / (count + 1), 3)
            for index in range(count)
        })
    visual = extract_video_frames(
        path,
        visual_timestamps,
        max_frames=max(1, min(int(max_visual_samples or 3), 3)),
        total_timeout=6.0,
        frame_timeout=2.0,
    ) if visual_timestamps else {"frames": [], "warning": "Output duration was not available for visual sampling."}
    decoded_timestamps = [frame["timestamp"] for frame in visual.get("frames") or []]
    visual_ok = bool(decoded_timestamps)
    visual_check = {
        "ok": visual_ok,
        "requested_timestamps": visual_timestamps,
        "decoded_timestamps": decoded_timestamps,
        "decoded_frame_count": len(decoded_timestamps),
    }
    report["checks"]["visual_samples"] = visual_check
    if not visual_ok:
        report["errors"].append("No visual samples could be decoded from the final rendered file.")
    elif visual.get("warning"):
        report["warnings"].append("Visual sampling was partial during QA; inspect the clip preview.")

    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warning"
    else:
        report["status"] = "pass"
    return report
