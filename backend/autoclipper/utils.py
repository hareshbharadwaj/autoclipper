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
