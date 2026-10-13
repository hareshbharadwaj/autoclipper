"""Central configuration loaded from environment / .env."""
import os
import time

from dotenv import load_dotenv

_this_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_this_dir)
_root_dir = os.path.dirname(_backend_dir)
load_dotenv(os.path.join(_root_dir, ".env"))
load_dotenv(os.path.join(_backend_dir, ".env"))
load_dotenv()

# --- API keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

# --- Models ---
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Fallback models tried in order when the primary is overloaded (503 / 429).
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.5-flash",
]


def _is_transient_gemini_error(err: Exception) -> bool:
    """Return True if an exception represents a retryable transient failure."""
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True

    err_str = f"{type(err).__name__}: {err}".lower()
    transient_cues = (
        "503",
        "429",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "resourceexhausted",
        "serviceunavailable",
        "rate limit",
        "spikes in demand",
        "temporarily unavailable",
        "try again later",
        "overloaded",
    )
    return any(cue in err_str for cue in transient_cues)


def gemini_generate(client, model: str, contents, max_retries: int = 2, **kwargs):
    """Call Gemini with automatic retry + fallback on 503/429 overload.

    Tries the requested model up to max_retries times with backoff, then falls through
    to GEMINI_FALLBACK_MODELS.
    """
    from .utils import logger

    # Build unique, ordered list of models to try
    models_to_try = []
    for m in [model] + GEMINI_FALLBACK_MODELS:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_err = None
    for m in models_to_try:
        for attempt in range(max_retries + 1):
            try:
                return client.models.generate_content(model=m, contents=contents, **kwargs)
            except Exception as e:
                if _is_transient_gemini_error(e):
                    last_err = e
                    if attempt < max_retries:
                        wait = (2 ** attempt) + 1
                        logger.warning(
                            "Gemini transient error on %s (attempt %d/%d); retrying in %ds: %s",
                            m, attempt + 1, max_retries + 1, wait, e,
                        )
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(
                            "Gemini %s exhausted all retries; falling back to next available model",
                            m,
                        )
                        break
                else:
                    raise  # non-transient errors propagate immediately
    raise last_err or RuntimeError("All Gemini models failed")

# --- Content-type analysis modes ---
# Each mode tailors the Gemini prompt for better-targeted viral clips.
CONTENT_TYPES = {
    "general": {
        "label": "General / Other",
        "focus": "any moment with a strong hook, emotion, insight, humor, or controversy",
    },
    "podcast": {
        "label": "Podcast",
        "focus": "hot takes, debates, controversial opinions, personal stories, and quotable one-liners that spark discussion",
    },
    "gaming": {
        "label": "Gaming",
        "focus": "clutch plays, epic fails, funny reactions, rage moments, and skill highlights that gaming audiences love",
    },
    "tutorial": {
        "label": "Tutorial / How-To",
        "focus": "quick wins, surprising tips, 'I wish I knew this' moments, and clear step-by-step hacks viewers will save",
    },
    "irl": {
        "label": "IRL / Live Stream",
        "focus": "unexpected real-life moments, funny encounters, raw reactions, and spontaneous viral interactions",
    },
}

# --- Groq audio limits ---
# Free tier caps uploads at 25 MB; keep a safety margin. Dev tier allows 100 MB.
GROQ_MAX_AUDIO_MB = float(os.environ.get("GROQ_MAX_AUDIO_MB", "24"))
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# --- Clip generation defaults ---
MIN_CLIP_SECONDS = float(os.environ.get("MIN_CLIP_SECONDS", "15"))
MAX_CLIP_SECONDS = float(os.environ.get("MAX_CLIP_SECONDS", "60"))
TARGET_CLIP_COUNT = int(os.environ.get("TARGET_CLIP_COUNT", "8"))

# Hard ceiling: a single short-form clip is never longer than 2 minutes.
# Deliberately not configurable: API callers and deployment env vars must not
# be able to raise this safety/product limit.
HARD_MAX_CLIP_SECONDS = 120.0

# --- Misc ---
YOUTUBE_COOKIES = os.environ.get("YOUTUBE_COOKIES")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
YOUTUBE_REGION = os.environ.get("YOUTUBE_REGION", "ID")

# --- Backend job queue ---
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "output")

# --- Burned-in subtitles ---
SUBTITLE_FONT = os.environ.get("SUBTITLE_FONT", "Arial")
# Font size / vertical margin are reference values for a 1080x1920 design canvas;
# the subtitle renderer scales them to each clip's actual dimensions and aspect ratio.
SUBTITLE_FONT_SIZE = int(os.environ.get("SUBTITLE_FONT_SIZE", "80"))
SUBTITLE_WORDS_PER_CUE = int(os.environ.get("SUBTITLE_WORDS_PER_CUE", "1"))
SUBTITLE_MARGIN_V = int(os.environ.get("SUBTITLE_MARGIN_V", "280"))
SUBTITLE_PRIMARY = os.environ.get("SUBTITLE_PRIMARY", "&H00FFFFFF")      # white (opaque)
SUBTITLE_OUTLINE = os.environ.get("SUBTITLE_OUTLINE", "&H00000000")      # black
SUBTITLE_BACKGROUND = os.environ.get("SUBTITLE_BACKGROUND", "&H80000000")  # 50% black


def require(value, name: str):
    """Return value or raise a clear error if it is missing."""
    if not value:
        raise RuntimeError(
            f"Missing required configuration: {name}. "
            f"Set it in your environment or .env file."
        )
    return value
