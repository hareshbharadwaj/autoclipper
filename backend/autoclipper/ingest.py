"""Video ingest: download from a URL (yt-dlp) or use a local file."""
import os
import shutil

from . import config
from .utils import logger, sanitize_filename


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _write_cookies(output_dir: str, cookies_text: str = None):
    """Materialize YouTube cookies (from request or YOUTUBE_COOKIES env) into a
    Netscape cookies file. Returns the path or None if no cookies provided."""
    text = cookies_text or config.YOUTUBE_COOKIES
    if not text or not text.strip():
        return None
    cookies_path = os.path.join(output_dir, "cookies.txt")
    try:
        with open(cookies_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Using YouTube cookies (%d bytes)", os.path.getsize(cookies_path))
        return cookies_path
    except OSError as e:
        logger.warning("Failed to write cookies file: %s", e)
        return None


def _finalize_opts(force_hd: bool = False, cookiefile: str = None, extractor_args: dict = None):
    if force_hd:
        fmt = (
            "bestvideo[height>=720]+bestaudio/"
            "best[height>=720]/best"
        )
    else:
        fmt = (
            "bestvideo+bestaudio/best[ext=mp4]/best"
        )

    last_pct = {"v": -1}

    def _progress(d):
        if d.get("status") == "downloading":
            try:
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                got = d.get("downloaded_bytes", 0)
                if total:
                    pct = int(got / total * 100)
                    if pct >= last_pct["v"] + 5:
                        last_pct["v"] = pct
                        logger.info("YouTube download: %d%%", pct)
            except Exception:
                pass

    opts = {
        "quiet": True,
        "no_warnings": True,
        "cookiefile": cookiefile,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": True,
        "cachedir": False,
        "format": fmt,
        "merge_output_format": "mp4",
        "progress_hooks": [_progress],
    }
    if extractor_args:
        opts["extractor_args"] = {"youtube": extractor_args}
    return opts


def _download_youtube(url: str, output_dir: str, force_hd: bool = False, cookies_text: str = None, max_height: int = None):
    import yt_dlp

    cookies_path = _write_cookies(output_dir, cookies_text)
    has_cookies = bool(cookies_path)

    # Sequence of client configurations to try:
    # 1. Default (let yt-dlp auto-select best client like visionos/default; avoids 403)
    # 2. Android/mweb fallback without cookies
    # 3. With cookies (if provided)
    attempts = [
        (None, None, "default client chain (anonymous)"),
        ({"player_client": ["android", "mweb", "web"]}, None, "android/mweb client chain"),
    ]
    if has_cookies:
        attempts.append(({"player_client": ["mweb", "web", "ios"]}, cookies_path, "with cookies"))

    last_err = None
    for extractor_args, cookiefile, label in attempts:
        opts = _finalize_opts(force_hd=force_hd, cookiefile=cookiefile, extractor_args=extractor_args)
        if max_height:
            opts["format"] = (
                f"bestvideo[height<={max_height}]+bestaudio/"
                f"best[height<={max_height}]/best"
            )

        out_template = os.path.join(output_dir, "%(title).90B.%(ext)s")
        opts["outtmpl"] = out_template
        opts["overwrites"] = True

        try:
            logger.info("YouTube download attempt: %s", label)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)

            raw_title = (info.get("title") if isinstance(info, dict) else None) or "youtube_video"
            title = sanitize_filename(raw_title)

            # Locate the downloaded file
            final_path = None
            if info:
                prepared = ydl.prepare_filename(info)
                base, _ = os.path.splitext(prepared)
                candidate_mp4 = f"{base}.mp4"
                if os.path.exists(candidate_mp4):
                    final_path = candidate_mp4
                elif os.path.exists(prepared):
                    final_path = prepared

            if not final_path or not os.path.exists(final_path):
                for f in os.listdir(output_dir):
                    if f.endswith(".mp4"):
                        final_path = os.path.join(output_dir, f)
                        break

            if not final_path or not os.path.exists(final_path):
                raise RuntimeError("yt-dlp finished but no output file was found.")

            logger.info("Downloaded successfully: %s", final_path)
            return final_path, title
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("YouTube attempt %s failed: %s", label, e)

    raise RuntimeError(f"YouTube download failed: {last_err}")


def ingest(source: str, output_dir: str, force_hd: bool = False, cookies_text: str = None, max_height: int = None):
    """Return (video_path, title) for a URL or local file `source`."""
    os.makedirs(output_dir, exist_ok=True)

    if _is_url(source):
        return _download_youtube(source, output_dir, force_hd=force_hd, cookies_text=cookies_text, max_height=max_height)

    if not os.path.exists(source):
        raise FileNotFoundError(f"Input file not found: {source}")

    title = sanitize_filename(os.path.splitext(os.path.basename(source))[0])
    dest = os.path.join(output_dir, f"{title}{os.path.splitext(source)[1] or '.mp4'}")
    if os.path.abspath(source) != os.path.abspath(dest):
        shutil.copy2(source, dest)
    logger.info("Using local file: %s", dest)
    return dest, title
