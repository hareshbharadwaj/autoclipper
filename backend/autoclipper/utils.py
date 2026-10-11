"""Shared helpers: logging, filenames, and FFmpeg/FFprobe wrappers."""
import json
import logging
import os
import re
import shutil
import subprocess
import sys

logger = logging.getLogger("autoclipper")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def sanitize_filename(name: str) -> str:
    """Strip characters that are invalid in filenames across platforms."""
    name = re.sub(r'[<>:"/\\|?*#]', "", name or "")
    name = name.replace(" ", "_").strip("._")
    return (name or "video")[:100]


def ensure_ffmpeg():
    """Verify ffmpeg is available on PATH, falling back to imageio_ffmpeg and ~/bin."""
    # Check ~/bin and /usr/local/bin
    for extra_dir in (os.path.expanduser("~/bin"), "/usr/local/bin"):
        if os.path.isdir(extra_dir) and extra_dir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{extra_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                bin_dir = os.path.dirname(exe)
                if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                target = os.path.join(bin_dir, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
                if not os.path.exists(target):
                    try:
                        shutil.copy2(exe, target)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("imageio_ffmpeg fallback failed: %s", e)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "'ffmpeg' not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html"
        )
    if shutil.which("ffprobe") is None:
        logger.warning("'ffprobe' not found on PATH; duration/stream probes may use fallback.")


def run_command(cmd, check: bool = True, cwd: str = None) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing output. Raises on non-zero exit when check=True."""
    logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def probe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe (with ffmpeg fallback)."""
    if shutil.which("ffprobe"):
        try:
            result = run_command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    path,
                ],
                check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except Exception:
            pass

    if shutil.which("ffmpeg"):
        try:
            res = run_command(["ffmpeg", "-i", path], check=False)
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr or "")
            if match:
                h, m, s = match.groups()
                return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass

    return 0.0


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


# --- Aspect ratio presets -------------------------------------------------
# ratio = width / height
ASPECT_PRESETS = {
    "9:16": {
        "ratio": 9 / 16,
        "label": "9:16 Portrait",
        "blurb": "Reels, TikTok & YouTube Shorts",
    },
    "1:1": {
        "ratio": 1.0,
        "label": "1:1 Square",
        "blurb": "Feed posts & in-stream playback",
    },
    "16:9": {
        "ratio": 16 / 9,
        "label": "16:9 Landscape",
        "blurb": "Original widescreen (no crop)",
    },
}


def classify_aspect(width: int, height: int) -> str:
    """Return the closest '9:16' / '1:1' / '16:9' preset for raw dimensions."""
    if not width or not height:
        return "16:9"
    ratio = width / float(height)
    return min(ASPECT_PRESETS, key=lambda k: abs(ASPECT_PRESETS[k]["ratio"] - ratio))


def recommend_aspect(width: int, height: int) -> dict:
    """Detect the source format and recommend the best publishing format.

    Short-form platforms reward vertical video, so landscape sources are
    recommended for a 9:16 reframe (with subject tracking); vertical sources
    stay vertical.
    """
    detected = classify_aspect(width, height)
    if detected == "16:9":
        recommended = "9:16"
        reason = (
            "Source is landscape. 9:16 Portrait performs best on Reels, TikTok "
            "and Shorts — AutoClipper will smart-crop with face/subject tracking."
        )
    elif detected == "1:1":
        recommended = "9:16"
        reason = (
            "Source is square. 9:16 Portrait fills the whole screen on Reels, "
            "TikTok and Shorts."
        )
    else:
        recommended = "9:16"
        reason = "Source is already vertical — 9:16 is the native short-form format."
    return {
        "detected": detected,
        "detected_label": ASPECT_PRESETS[detected]["label"],
        "recommended": recommended,
        "recommended_label": ASPECT_PRESETS[recommended]["label"],
        "reason": reason,
        "options": [
            {"id": k, "label": v["label"], "blurb": v["blurb"]} for k, v in ASPECT_PRESETS.items()
        ],
    }


def probe_video_info(path: str) -> dict:
    """Probe a media file for dimensions, duration, audio and aspect metadata.

    Uses ffprobe when available and degrades gracefully to an `ffmpeg -i`
    parse (and finally to zeros) so analysis never hard-fails on a probe.
    """
    info = {
        "width": 0,
        "height": 0,
        "duration": 0.0,
        "has_audio": False,
        "aspect_ratio": None,
        "aspect": recommend_aspect(0, 0),
    }
    if not path or not os.path.exists(path):
        return info

    width = height = 0
    duration = 0.0
    has_audio = False

    if shutil.which("ffprobe"):
        try:
            result = run_command(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "stream=index,codec_type,width,height,duration",
                    "-show_entries", "format=duration",
                    "-of", "json",
                    path,
                ],
                check=False,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout or "{}")
                for stream in data.get("streams") or []:
                    if stream.get("codec_type") == "audio":
                        has_audio = True
                        continue
                    if stream.get("codec_type") != "video":
                        continue
                    try:
                        width = int(stream.get("width") or 0)
                        height = int(stream.get("height") or 0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        duration = duration or float(stream.get("duration") or 0.0)
                    except (TypeError, ValueError):
                        pass
                try:
                    duration = duration or float(((data.get("format") or {}).get("duration")) or 0.0)
                except (TypeError, ValueError):
                    pass
        except Exception as e:  # noqa: BLE001
            logger.debug("ffprobe stream probe failed for %s: %s", path, e)

    if not (width and height) and shutil.which("ffmpeg"):
        try:
            res = run_command(["ffmpeg", "-i", path], check=False)
            stderr = res.stderr or ""
            match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
            if match:
                h, m, s = match.groups()
                duration = duration or int(h) * 3600 + int(m) * 60 + float(s)
            match = re.search(r",\s*(\d{2,5})x(\d{2,5})", stderr)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
            if "Audio:" in stderr:
                has_audio = True
        except Exception as e:  # noqa: BLE001
            logger.debug("ffmpeg dimension probe failed for %s: %s", path, e)

    if not duration:
        duration = probe_duration(path)

    info.update(
        {
            "width": width,
            "height": height,
            "duration": round(duration, 2),
            "has_audio": has_audio,
            "aspect_ratio": (width / float(height)) if width and height else None,
            "aspect": recommend_aspect(width, height),
        }
    )
    return info


def write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_to_library(output_root: str, title: str, job_id: str, clips: list) -> str:
    """Persist generated clips into a Saved Library folder per source video.

    Layout: <output_root>/library/<video_title>/clip1.mp4, clip2.mp4, ...
    Already-existing files are skipped so re-runs do not overwrite previous
    saves. Returns the library folder path.
    """
    from . import config

    safe_title = sanitize_filename(title) or "video"
    # Avoid collisions when the same title is processed twice.
    lib_root = os.path.join(config.OUTPUT_ROOT if not output_root else output_root, "library")
    base = os.path.join(lib_root, safe_title)
    n = 1
    while os.path.exists(base):
        base = os.path.join(lib_root, f"{safe_title}_{n}")
        n += 1
    os.makedirs(base, exist_ok=True)

    saved = []
    for clip in clips:
        src = os.path.join(output_root, job_id, clip.get("file", ""))
        if not os.path.isfile(src):
            continue
        idx = clip.get("index") or (len(saved) + 1)
        dest = os.path.join(base, f"clip{idx}.mp4")
        if os.path.exists(dest):
            continue
        try:
            shutil.copy2(src, dest)
            saved.append({"index": idx, "file": f"clip{idx}.mp4"})
        except OSError as e:
            logger.warning("Failed to save clip %s to library: %s", idx, e)

    # Save a small manifest for the dashboard to render titles/captions.
    # Key each clip by its destination filename so the API can match clips
    # reliably (index alone is ambiguous across re-runs).
    clip_meta = []
    for clip in clips:
        idx = clip.get("index") or (len(clip_meta) + 1)
        dest_name = f"clip{idx}.mp4"
        clip_meta.append(
            {
                "file": dest_name,
                **{k: clip.get(k) for k in ("index", "title", "description", "hook", "start", "end")},
            }
        )
    manifest = {
        "title": title,
        "job_id": job_id,
        "clips": clip_meta,
    }
    write_json(os.path.join(base, "manifest.json"), manifest)
    logger.info("Saved %d clip(s) to library: %s", len(saved), base)
    return base
