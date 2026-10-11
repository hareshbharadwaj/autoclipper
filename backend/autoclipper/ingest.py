"""Video ingest: download from a URL (yt-dlp) or use a local file."""
import os
import shutil
import socket
import ssl
import time
import urllib.error

from . import config
from .utils import logger, sanitize_filename


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


_TRANSIENT_NETWORK_MARKERS = (
    "tls/ssl connection has been closed",
    "unexpected eof while reading",
    "ssleoferror",
    "ssl_error_syscall",
    "ssl_connect:",
    "curl: (35)",
    "connection closed abruptly",
    "connection reset by peer",
    "remote end closed connection",
    "timed out",
    "temporary failure in name resolution",
    "network is unreachable",
)


def _is_transient_network_error(error: Exception) -> bool:
    """Recognize retryable connection failures wrapped by yt-dlp errors."""
    message = str(error).lower()
    if any(marker in message for marker in _TRANSIENT_NETWORK_MARKERS):
        return True

    current = error
    seen = set()
    for _ in range(5):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, urllib.error.HTTPError):
            return False
        if isinstance(current, (socket.timeout, TimeoutError, ConnectionError, ssl.SSLError)):
            return True
        reason = getattr(current, "reason", None)
        current = reason or getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False


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


def _finalize_opts(
    force_hd: bool = False,
    cookiefile: str = None,
    extractor_args: dict = None,
    impersonate=None,
):
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
        "extractor_retries": 5,
        "fragment_retries": 10,
        "cachedir": False,
        "format": fmt,
        "merge_output_format": "mp4",
        "progress_hooks": [_progress],
    }
    if extractor_args:
        opts["extractor_args"] = {"youtube": extractor_args}
    if impersonate is not None:
        opts["impersonate"] = impersonate
    return opts


def _chrome_impersonation_target():
    """Return a Chrome TLS target when yt-dlp's curl-cffi extra is installed."""
    try:
        import curl_cffi  # noqa: F401
        from yt_dlp.networking.impersonate import ImpersonateTarget

        return ImpersonateTarget(client="chrome")
    except (ImportError, OSError):
        return None


def _download_youtube(url: str, output_dir: str, force_hd: bool = False, cookies_text: str = None, max_height: int = None):
    import yt_dlp

    cookies_path = _write_cookies(output_dir, cookies_text)
    has_cookies = bool(cookies_path)

    # Sequence of client configurations to try:
    # 1. Default (let yt-dlp auto-select best client like visionos/default; avoids 403)
    # 2. Android/mweb fallback without cookies
    # 3. With cookies (if provided)
    attempts = [
        (None, None, "default client chain (anonymous)", None),
        ({"player_client": ["android", "mweb", "web"]}, None, "android/mweb client chain", None),
    ]
    if has_cookies:
        attempts.append(({"player_client": ["mweb", "web", "ios"]}, cookies_path, "with cookies", None))

    # If normal requests hit TLS EOFs, retry once with curl-cffi's browser-like
    # TLS handshake. This is only attempted after a transient network failure.
    chrome_target = _chrome_impersonation_target()
    if chrome_target is not None:
        attempts.append((None, cookies_path if has_cookies else None, "Chrome TLS fallback", chrome_target))

    last_err = None
    for extractor_args, cookiefile, label, impersonate in attempts:
        if impersonate is not None and not _is_transient_network_error(last_err):
            continue

        opts = _finalize_opts(
            force_hd=force_hd,
            cookiefile=cookiefile,
            extractor_args=extractor_args,
            impersonate=impersonate,
        )
        if max_height:
            opts["format"] = (
                f"bestvideo[height<={max_height}]+bestaudio/"
                f"best[height<={max_height}]/best"
            )

        out_template = os.path.join(output_dir, "%(title).90B.%(ext)s")
        opts["outtmpl"] = out_template
        opts["overwrites"] = True

        try:
            info = None
            ydl = None
            for retry_number in range(3):
                try:
                    logger.info("YouTube download attempt: %s", label)
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if retry_number >= 2 or not _is_transient_network_error(e):
                        raise
                    delay = 2 ** retry_number
                    logger.warning(
                        "Transient YouTube TLS/network failure (%s); retry %d/2 in %ds.",
                        type(e).__name__, retry_number + 1, delay,
                    )
                    time.sleep(delay)

            raw_title = (info.get("title") if isinstance(info, dict) else None) or "youtube_video"
            title = sanitize_filename(raw_title)

            # Locate the downloaded file.
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
                for filename in os.listdir(output_dir):
                    if filename.endswith(".mp4"):
                        final_path = os.path.join(output_dir, filename)
                        break

            if not final_path or not os.path.exists(final_path):
                raise RuntimeError("yt-dlp finished but no output file was found.")

            logger.info("Downloaded successfully: %s", final_path)
            return final_path, title
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("YouTube attempt %s failed (%s).", label, type(e).__name__)

    try:
        yt_dlp_version = yt_dlp.version.__version__
    except AttributeError:
        yt_dlp_version = "unknown"
    if _is_transient_network_error(last_err):
        raise RuntimeError(
            "YouTube download hit a transient TLS/network error after retries "
            f"(yt-dlp {yt_dlp_version}). Retry the job; if it keeps failing, "
            "rebuild the backend with the current requirements and check outbound "
            f"TLS access to YouTube. Last error: {last_err}"
        ) from last_err
    raise RuntimeError(f"YouTube download failed: {last_err}") from last_err


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
