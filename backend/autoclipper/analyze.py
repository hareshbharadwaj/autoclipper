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

STORY ARC — every clip MUST have a beginning, a middle and an end:
1. HOOK (first 0–3s): open exactly on a complete, attention-grabbing sentence — \
a bold claim, a question, a warning, or a surprising statement. Never start \
mid-word or mid-sentence, and never fade in over the tail of a previous thought.
2. BODY (middle): the explanation, demonstration, steps, or dialogue that delivers \
the value promised by the hook. Keep it in the speaker's original order.
3. CONCLUSION (end): land on a finished sentence — the payoff, result, takeaway, \
or punchline. The very last frame must be the end of that sentence, so extend the \
end timestamp to the end of the closing sentence rather than cutting it off.
A clip that ends mid-sentence or starts mid-sentence is a FAILURE. Reject it and \
pick a different moment.

  Return ONLY valid JSON in exactly this shape, with no markdown fences:
  {{
    "clips": [
      {{
        "start": 12.3,
        "end": 48.9,
        "title": "punchy viral title (max 60 chars)",
        "description": "engaging social caption with 3-5 relevant hashtags (max 200 chars)",
        "hook": "one-line on-screen hook text",
        "reason": "why this moment is compelling",
        "story_arc": "one line: hook -> body -> conclusion"
      }}
    ]
  }}

TRANSCRIPT:
{transcript}
"""


SUMMARY_PROMPT_TEMPLATE = """You are a senior short-form content strategist. You are \
given the FULL timestamped transcript of a video (duration: {duration:.0f} seconds, \
source resolution: {resolution}).

The transcript language is: {language}.
IMPORTANT: Write every field in the SAME language as the transcript. If the \
transcript is Indonesian, write natural Indonesian — do NOT translate to English.

Your job is NOT to touch the video yet. You are producing an analysis report that a \
human will read before deciding what clip to make.

Step 1 — Understand the whole video, from the first second to the last.
Step 2 — Write "summary": exactly 2-3 sentences describing what this video is about, \
who it is for, and what the viewer walks away with. No timestamps in the summary.
Step 3 — Write "chapters": 3-8 timestamped sections covering the ENTIRE video in \
order. Every chapter needs a start, an end, and a short label. Use "start" and "end" \
as seconds (numbers). Chapter ranges must be continuous and cover the whole duration.
Step 4 — Write "key_topics": 3-5 short topic tags that are SPECIFIC to this video \
(e.g. "Tunnel cutting grip" not "cooking").
Step 5 — Write "suggested_angles": {angle_count} distinct, video-specific clip ideas. \
These are NOT generic categories — each one must come from something that actually \
happens in THIS video. For each angle give:
  - "title": a punchy angle name (max 60 chars)
  - "hook": the on-screen hook text for the first 2 seconds (max 90 chars)
  - "description": 1-2 sentences on what the clip shows and why it will perform
  - "target_duration": a whole number of seconds between 20 and 120 that this \
specific idea actually needs to breathe (do not force every angle to the same length)
  - "start_hint" / "end_hint": your best estimate (in seconds) of where this moment \
lives in the video, taken from real transcript timestamps
  - "custom_prompt": a ready-to-use editing instruction (1-2 sentences) that tells \
the clip editor exactly what to focus on, and demands a clean hook → body → ending \
story arc. This is what gets sent to the editor when the user picks this angle.

Rules:
- Never invent speech, events, or timestamps that are not in the transcript.
- Angles must not be near-duplicates of each other — cover different moments/topics.
- Order suggested_angles from strongest to weakest viral potential.

  Return ONLY valid JSON in exactly this shape, with no markdown fences:
  {{
    "summary": "2-3 sentence overview of the whole video",
    "chapters": [
      {{"start": 0.0, "end": 75.0, "label": "Cold open & setup"}}
    ],
    "key_topics": ["topic one", "topic two", "topic three"],
    "suggested_angles": [
      {{
        "title": "punchy angle name",
        "hook": "on-screen hook text",
        "description": "what the clip shows and why it works",
        "target_duration": 45,
        "start_hint": 120.0,
        "end_hint": 165.0,
        "custom_prompt": "editing instruction for the clip editor"
      }}
    ]
  }}

TRANSCRIPT:
{transcript}
"""


def _truncate_transcript(text: str, max_chars: int = 180000) -> str:
    """Keep the prompt inside a safe context budget while covering the whole video."""
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3) :]
    return (
        head
        + "\n\n[... middle of transcript trimmed for length ...]\n\n"
        + tail
    )


def analyze_video_summary_and_angles(
    transcript: dict,
    api_key: str = None,
    model: str = None,
    video_info: dict = None,
    angle_count: int = 4,
    content_type: str = None,
) -> dict:
    """Produce a whole-video understanding report before any clip is cut.

    Returns a dict with: summary, chapters, key_topics, suggested_angles.
    """
    from google import genai

    api_key = config.require(api_key or config.GEMINI_API_KEY, "GEMINI_API_KEY")
    model = model or config.GEMINI_MODEL

    transcript_text = _build_transcript_text(transcript)
    if not transcript_text.strip():
        raise RuntimeError("Transcript is empty; cannot analyze the video.")

    video_info = video_info or {}
    duration = float(
        video_info.get("duration") or transcript.get("duration") or 0.0
    )
    width = video_info.get("width") or 0
    height = video_info.get("height") or 0
    resolution = f"{width}x{height}" if width and height else "unknown"
    language = transcript.get("language") or "auto-detected"

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        duration=duration,
        resolution=resolution,
        language=language,
        angle_count=max(1, min(int(angle_count or 4), 6)),
        transcript=_truncate_transcript(transcript_text),
    )
    ctype = config.CONTENT_TYPES.get(content_type, config.CONTENT_TYPES["general"])
    if content_type and content_type != "general":
        prompt += (
            "\n\nCONTENT NICHE (weight your angle selection toward this):\n"
            + ctype["focus"]
        )

    logger.info("Building whole-video analysis with Gemini model '%s'...", model)
    client = genai.Client(api_key=api_key)
    response = config.gemini_generate(client, model, prompt)
    raw = (response.text or "").strip()

    data = _parse_clips(raw)
    if isinstance(data, dict):
        report = data
    elif isinstance(data, list):
        report = data[0] if data and isinstance(data[0], dict) else {}
    else:
        report = {}

    return _validate_report(report, duration)


def _validate_report(report: dict, duration: float) -> dict:
    """Normalize / sanity-check the Gemini analysis report."""
    summary = (report.get("summary") or "").strip()
    if not summary:
        summary = "No summary could be generated for this video."

    chapters = []
    for ch in report.get("chapters") or []:
        if not isinstance(ch, dict):
            continue
        try:
            start = float(ch.get("start", 0.0))
            end = float(ch.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if duration and start > duration:
            continue
        chapters.append(
            {
                "start": round(max(0.0, start), 2),
                "end": round(min(end, duration or end), 2),
                "label": (ch.get("label") or "Untitled section").strip()[:90],
            }
        )
    chapters.sort(key=lambda c: c["start"])

    topics = []
    for t in report.get("key_topics") or []:
        if isinstance(t, str) and t.strip():
            topics.append(t.strip()[:60])
    topics = topics[:6]

    angles = []
    for a in report.get("suggested_angles") or []:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        if not title:
            continue
        try:
            target = int(float(a.get("target_duration") or 45))
        except (TypeError, ValueError):
            target = 45
        target = max(10, min(120, target))

        def _hint(key):
            try:
                value = float(a.get(key))
            except (TypeError, ValueError):
                return None
            if duration and value > duration:
                return None
            return round(max(0.0, value), 2)

        angles.append(
            {
                "title": title[:80],
                "hook": (a.get("hook") or "").strip()[:120],
                "description": (a.get("description") or "").strip()[:400],
                "target_duration": target,
                "start_hint": _hint("start_hint"),
                "end_hint": _hint("end_hint"),
                "custom_prompt": (a.get("custom_prompt") or "").strip()[:1200],
            }
        )
    angles = angles[:6]

    return {
        "summary": summary,
        "chapters": chapters,
        "key_topics": topics,
        "suggested_angles": angles,
        "duration": round(duration, 2),
    }


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

    # Hard ceiling: no clip is ever longer than HARD_MAX_CLIP_SECONDS (2 min).
    max_s = min(max_s, config.HARD_MAX_CLIP_SECONDS)
    target_s = float(target_duration) if target_duration is not None else (min_s + max_s) / 2.0
    target_s = min(target_s, config.HARD_MAX_CLIP_SECONDS)
    if target_s <= 0:
        raise ValueError("target_duration must be greater than zero.")
    if target_duration is not None:
        # Give the model a small editing window around the target, while preserving
        # natural sentence boundaries. Hard validation still happens after generation.
        min_s = max(5.0, target_s - 3.0)
        max_s = min(config.HARD_MAX_CLIP_SECONDS, target_s + 3.0)

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


def _load_json(raw: str):
    """Parse model output into JSON, tolerating markdown fences and prose."""
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

    return data


def _parse_clips(raw: str) -> list:
    """Parse the model output into a list of clip dicts."""
    data = _load_json(raw)
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
        upper = min(upper, config.HARD_MAX_CLIP_SECONDS + 3)
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
                "story_arc": (c.get("story_arc") or "").strip()[:300],
            }
        )
    valid.sort(key=lambda x: x["start"])
    return valid
