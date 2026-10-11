"""Viral moment detection using Google Gemini."""
import json
import re

from . import config
from .utils import logger


def _repair_json(text: str) -> str:
    """Fix common Gemini JSON issues: missing commas between array elements."""
    # Insert missing comma between } and { (objects in an array)
    text = re.sub(r'\}\s*\{', '}, {', text)
    # Insert missing comma between ] and [ (arrays in an array)
    text = re.sub(r'\]\s*\[', '], [', text)
    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _build_transcript_text(transcript: dict) -> str:
    """Render segments as timestamped lines for the LLM to reason over."""
    lines = []
    for seg in transcript.get("segments", []):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(f"[{start:.1f} - {end:.1f}] {text}")
    return "\n".join(lines)


PROMPT_TEMPLATE = """You are a viral short-form video editor. Given a timestamped \
transcript of a long video, identify the {count} best self-contained moments that \
would perform well as vertical short clips (TikTok / Reels / YouTube Shorts).

The transcript language is: {language}.
IMPORTANT: Write ALL output fields (title, description, hook, reason) in the SAME \
language as the transcript. If the transcript is Indonesian, write everything in \
natural Indonesian. Do not translate to English unless the transcript itself is in \
English.

Rules:
- Each clip must be between {min_s:.0f} and {max_s:.0f} seconds long.
- Prefer a duration close to {target_s:.0f} seconds when a target is supplied; target duration is a preference, not permission to cut a sentence or remove the payoff.
- Clips must start and end on natural sentence/thought boundaries (no mid-sentence cuts). Preserve the question/setup and the answer/payoff when they belong together.
- Prioritize coherent standalone stories over isolated shocking lines. Include enough context for a viewer who has not seen the source.
- Keep the opening hook within the first 1–3 seconds of the selected moment. End immediately after the conclusion, punchline, or useful takeaway.
- Do not invent speech or infer words not present in the transcript.
- Content focus: {focus}
- Do not overlap clips. Use timestamps that exist in the transcript.
- "start" and "end" are in seconds (numbers).

  Return ONLY valid JSON in exactly this shape, with no markdown fences:
  {{
    "clips": [
      {{
        "start": 12.3,
        "end": 48.9,
        "title": "punchy viral title (max 60 chars)",
        "description": "engaging social caption with 3-5 relevant hashtags (max 200 chars)",
        "hook": "one-line on-screen hook text",
        "reason": "why this moment is compelling"
      }}
    ]
  }}

TRANSCRIPT:
{transcript}
"""


def find_viral_clips(transcript: dict, api_key: str = None, model: str = None,
                      clip_count: int = None, min_clip: float = None,
                      max_clip: float = None, content_type: str = None,
                      target_duration: float = None, custom_prompt: str = None) -> list:
    """Ask Gemini for a list of viral clip candidates. Returns a list of dicts."""
    from google import genai

    api_key = config.require(api_key or config.GEMINI_API_KEY, "GEMINI_API_KEY")
    model = model or config.GEMINI_MODEL

    count = clip_count or config.TARGET_CLIP_COUNT
    min_s = min_clip if min_clip is not None else config.MIN_CLIP_SECONDS
    max_s = max_clip if max_clip is not None else config.MAX_CLIP_SECONDS

    ctype = config.CONTENT_TYPES.get(content_type, config.CONTENT_TYPES["general"])
    focus = ctype["focus"]
    target_s = float(target_duration) if target_duration is not None else (min_s + max_s) / 2.0
    if target_s <= 0:
        raise ValueError("target_duration must be greater than zero.")
    if target_duration is not None:
        # Give the model a small editing window around the target, while preserving
        # natural sentence boundaries. Hard validation still happens after generation.
        min_s = max(5.0, min(min_s, target_s - 3.0))
        max_s = min(120.0, max(max_s, target_s + 3.0))

    transcript_text = _build_transcript_text(transcript)
    if not transcript_text.strip():
        raise RuntimeError("Transcript is empty; cannot analyze for clips.")

    language = transcript.get("language") or "auto-detected"
    prompt = PROMPT_TEMPLATE.format(
        count=count,
        min_s=min_s,
        max_s=max_s,
        target_s=target_s,
        language=language,
        focus=focus,
        transcript=transcript_text,
    )
    if custom_prompt and custom_prompt.strip():
        prompt += (
            "\n\nUSER EDITING DIRECTION (follow this while preserving transcript truth):\n"
            + custom_prompt.strip()[:1200]
        )

    logger.info("Analyzing transcript with Gemini model '%s'...", model)
    client = genai.Client(api_key=api_key)
    response = config.gemini_generate(client, model, prompt)
    raw = (response.text or "").strip()

    clips = _parse_clips(raw)
    clips = _validate_clips(
        clips,
        total_duration=transcript.get("duration", 0.0),
        min_clip=min_s,
        max_clip=max_s,
    )
    logger.info("Found %d viral clip candidate(s)", len(clips))
    return clips


def _parse_clips(raw: str) -> list:
    """Parse the model output, tolerating markdown fences and bare arrays."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence.
        text = text.split("```", 2)[1] if "```" in text else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()

    # Extract the outermost JSON object or array if wrapped in prose.
    if not (text.startswith("{") or text.startswith("[")):
        for opener, closer in (("{", "}"), ("[", "]")):
            first, last = text.find(opener), text.rfind(closer)
            if first != -1 and last != -1 and last > first:
                text = text[first : last + 1]
                break

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_json(text)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Could not parse Gemini response as JSON: {e}\nRaw: {raw[:500]}")

    if isinstance(data, list):
        return data
    return data.get("clips", []) if isinstance(data, dict) else []


def _validate_clips(clips: list, total_duration: float,
                    min_clip: float = None, max_clip: float = None) -> list:
    valid = []
    for c in clips:
        try:
            start = float(c["start"])
            end = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        if total_duration and end > total_duration + 1:
            end = total_duration
        length = end - start
        lower = (min_clip if min_clip is not None else config.MIN_CLIP_SECONDS) - 2
        upper = (max_clip if max_clip is not None else config.MAX_CLIP_SECONDS) + 3
        if length < max(3.0, lower) or length > upper:
            continue
        valid.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "title": (c.get("title") or "Untitled Clip").strip()[:80],
                "description": (c.get("description") or "").strip()[:220],
                "hook": (c.get("hook") or "").strip()[:120],
                "reason": (c.get("reason") or "").strip(),
            }
        )
    valid.sort(key=lambda x: x["start"])
    return valid
