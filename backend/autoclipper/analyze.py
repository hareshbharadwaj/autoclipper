"""Viral moment detection using Google Gemini."""
import json
import math
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
- Generate up to {count} plausible candidates, ordered by likely short-form value; do not pad with weak or duplicate moments.
- Give every candidate a numeric "score" from 0 to 100 based on hook strength, narrative completeness, specificity, and standalone clarity.
- Include "evidence_quote": a short exact quote copied from the supplied transcript that supports the moment. Never invent or paraphrase this quote.
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
        "story_arc": "one line: hook -> body -> conclusion",
        "score": 0,
        "evidence_quote": "an exact short quote from the transcript"
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
IMPORTANT: Write every natural-language field in the SAME language as the transcript. \
Keep "content_type" as one of the exact lowercase enum tokens below. If the transcript \
is Indonesian, write natural Indonesian — do NOT translate to English.

Your job is NOT to touch the video yet. You are producing an analysis report that a \
human will read before deciding what clip to make.

Step 1 — Understand the whole video, from the first second to the last. Use both \
the transcript and sampled visual frames when available.
Step 2 — Classify the OVERALL video format as exactly one of: "general", "podcast", \
"gaming", "tutorial", or "irl". Classify what the source mainly is, not a topic \
mentioned in passing. Use "general" when the evidence is ambiguous. Do not assume a \
format from the title alone.
Step 3 — Write "content_type_reason": one short sentence explaining the evidence for \
the format choice. Let this classification inform which moments are strongest for \
short clips, without forcing a type-specific angle when the video does not support it.
Step 4 — Write "summary": exactly 2-3 sentences describing what this video is about, \
who it is for, and what the viewer walks away with. No timestamps in the summary.
Step 5 — Write "chapters": 3-8 timestamped sections covering the ENTIRE video in \
order. Every chapter needs a start, an end, and a short label. Use "start" and "end" \
as seconds (numbers). Chapter ranges must be continuous and cover the whole duration.
Step 6 — Write "key_topics": 3-5 short topic tags that are SPECIFIC to this video \
(e.g. "Tunnel cutting grip" not "cooking").
Step 7 — Write "suggested_angles": {angle_count} distinct, video-specific clip ideas. \
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
    "content_type": "tutorial",
    "content_type_reason": "brief evidence-based explanation",
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


def _format_visual_timestamp(seconds):
    """Render a sampled frame time in an unambiguous prompt-friendly format."""
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_video_summary_and_angles(
    transcript: dict,
    api_key: str = None,
    model: str = None,
    video_info: dict = None,
    angle_count: int = 4,
    content_type: str = None,
    visual_samples: list = None,
    visual_sampling_warning: str = None,
) -> dict:
    """Produce a whole-video understanding report before any clip is cut.

    Returns a dict with: content_type, content_type_reason, summary, chapters,
    key_topics, and suggested_angles.
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

    samples = [
        sample for sample in (visual_samples or [])
        if isinstance(sample, dict) and isinstance(sample.get("data"), bytes) and sample["data"]
    ]
    visual_context = {
        "status": "used" if samples and not visual_sampling_warning else (
            "partial" if samples else ("transcript_only" if visual_sampling_warning else "not_requested")
        ),
        "frame_count": len(samples),
        "timestamps": [round(float(sample.get("timestamp") or 0.0), 3) for sample in samples],
        "warning": visual_sampling_warning or None,
    }
    analysis_warnings = [visual_sampling_warning] if visual_sampling_warning else []

    logger.info("Building whole-video analysis with Gemini model '%s'...", model)
    client = genai.Client(api_key=api_key)
    contents = prompt
    if samples:
        prompt += (
            "\\n\\nVISUAL EVIDENCE: The following timestamp-labeled still frames are "
            "sparse samples from this source video. Use them only to describe visible "
            "context; do not infer speech, actions between frames, identities, or "
            "events that the frames do not establish. Treat the transcript as the "
            "authority for spoken content and timestamps."
        )
        contents = [prompt]
        try:
            for sample in samples:
                timestamp = float(sample.get("timestamp") or 0.0)
                mime_type = sample.get("mime_type") or "image/jpeg"
                part = genai.types.Part.from_bytes(data=sample["data"], mime_type=mime_type)
                contents.extend([
                    f"Visual frame sampled at {timestamp:.3f}s ({_format_visual_timestamp(timestamp)}).",
                    part,
                ])
        except Exception as exc:  # malformed images / SDK mismatch: use transcript only
            logger.warning("Could not prepare Gemini visual frames (%s); using transcript only.", type(exc).__name__)
            contents = prompt.removesuffix(
                "\\n\\nVISUAL EVIDENCE: The following timestamp-labeled still frames are "
                "sparse samples from this source video. Use them only to describe visible "
                "context; do not infer speech, actions between frames, identities, or "
                "events that the frames do not establish. Treat the transcript as the "
                "authority for spoken content and timestamps."
            )
            visual_context["status"] = "transcript_only"
            visual_context["frame_count"] = 0
            visual_context["timestamps"] = []
            visual_context["warning"] = "Visual frames could not be prepared; analysis used transcript evidence only."
            analysis_warnings.append(visual_context["warning"])

    try:
        response = config.gemini_generate(client, model, contents)
    except Exception as exc:
        if not samples or visual_context["status"] == "transcript_only":
            raise
        logger.warning(
            "Gemini visual-context analysis failed (%s); retrying with transcript only.",
            type(exc).__name__,
        )
        response = config.gemini_generate(client, model, prompt.rsplit("\\n\\nVISUAL EVIDENCE:", 1)[0])
        visual_context["status"] = "transcript_only"
        visual_context["warning"] = "Gemini could not analyze the sampled frames; this report uses transcript evidence only."
        analysis_warnings.append(visual_context["warning"])
    raw = (response.text or "").strip()

    # Whole-video reports are JSON objects, not {"clips": [...]} payloads.
    # Parsing through _parse_clips would discard summary/chapters/angles.
    data = _load_json(raw)
    if isinstance(data, dict):
        report = data
    elif isinstance(data, list):
        report = data[0] if data and isinstance(data[0], dict) else {}
    else:
        report = {}

    result = _validate_report(report, duration)
    result["visual_context"] = visual_context
    result["analysis_warnings"] = list(dict.fromkeys(analysis_warnings))
    return result


def _validate_report(report: dict, duration: float) -> dict:
    """Normalize / sanity-check the Gemini analysis report."""
    summary = (report.get("summary") or "").strip()
    if not summary:
        summary = "No summary could be generated for this video."

    raw_content_type = report.get("content_type")
    content_type = raw_content_type.strip().lower() if isinstance(raw_content_type, str) else "general"
    if content_type not in config.CONTENT_TYPES:
        content_type = "general"
    raw_type_reason = report.get("content_type_reason")
    content_type_reason = raw_type_reason.strip()[:240] if isinstance(raw_type_reason, str) else ""

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
        start = max(0.0, start)
        end = min(end, duration or end)
        if end <= start:
            continue
        chapters.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
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
    for angle in angles:
        start_hint, end_hint = angle.get("start_hint"), angle.get("end_hint")
        if start_hint is not None and end_hint is not None and end_hint <= start_hint:
            angle["start_hint"] = None
            angle["end_hint"] = None

    return {
        "content_type": content_type,
        "content_type_reason": content_type_reason,
        "summary": summary,
        "chapters": chapters,
        "key_topics": topics,
        "suggested_angles": angles,
        "duration": round(duration, 2),
    }



def _resolve_clip_bounds(min_clip=None, max_clip=None, target_duration=None):
    """Resolve hard clip bounds; an explicit target uses a +/-3s window."""
    if target_duration is None:
        return min_clip, max_clip
    try:
        target = float(target_duration)
    except (TypeError, ValueError):
        raise ValueError("target_duration must be between 5 and 120 seconds.")
    if not 5.0 <= target <= float(config.HARD_MAX_CLIP_SECONDS):
        raise ValueError("target_duration must be between 5 and 120 seconds.")
    return max(5.0, target - 3.0), min(float(config.HARD_MAX_CLIP_SECONDS), target + 3.0)


def extend_candidate_to_target_duration(
    candidate: dict,
    transcript: dict,
    target_duration: float,
    total_duration: float = None,
    tolerance: float = None,
) -> dict:
    """Extend a short clip candidate along complete transcript sentence boundaries.

    Expands along contiguous dialogue (speech gap <= 4.0s) so that the clip achieves
    approximately the user's requested duration (target_duration) while preserving
    coherent story context (opening setup, explanation, and payoff).

    If coherent context cannot reach the target (e.g. source has no more contiguous speech),
    the maximum coherent span is returned with duration_target_met = False and a clear
    duration_warning. Filler or unrelated footage is never added.
    """
    if target_duration is None:
        return dict(candidate)
    try:
        target = float(target_duration)
    except (TypeError, ValueError):
        return dict(candidate)
    if target <= 0:
        return dict(candidate)

    target = min(float(config.HARD_MAX_CLIP_SECONDS), target)
    if tolerance is None:
        tolerance = max(4.0, min(10.0, target * 0.15))

    try:
        cand_start = float(candidate.get("start", 0.0))
        cand_end = float(candidate.get("end", 0.0))
    except (TypeError, ValueError):
        return dict(candidate)

    if cand_end <= cand_start:
        return dict(candidate)

    cand_dur = cand_end - cand_start

    segments = []
    for s in (transcript.get("segments") or []):
        if not isinstance(s, dict):
            continue
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        text = (s.get("text") or "").strip()
        if end > start and text:
            if total_duration:
                start = max(0.0, min(start, float(total_duration)))
                end = max(0.0, min(end, float(total_duration)))
            if end > start:
                segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda s: (s["start"], s["end"]))

    if not segments:
        updated = dict(candidate)
        updated["requested_duration"] = target
        updated["duration_target_met"] = abs(cand_dur - target) <= tolerance
        if not updated["duration_target_met"]:
            updated["duration_warning"] = (
                f"Requested target duration was {target:.0f}s, but transcript has no segments "
                f"to extend candidate (current {cand_dur:.1f}s)."
            )
        return updated

    overlapping = [
        i for i, s in enumerate(segments)
        if s["start"] < cand_end and s["end"] > cand_start
    ]
    if overlapping:
        seed_start_idx = min(overlapping)
        seed_end_idx = max(overlapping)
    else:
        closest_idx = min(
            range(len(segments)),
            key=lambda i: min(abs(segments[i]["start"] - cand_start), abs(segments[i]["end"] - cand_end)),
        )
        seed_start_idx = closest_idx
        seed_end_idx = closest_idx

    # Contiguous bounds (speech gap <= 4.0s)
    max_gap = 4.0
    min_i = seed_start_idx
    while min_i > 0 and (segments[min_i]["start"] - segments[min_i - 1]["end"]) <= max_gap:
        min_i -= 1

    max_j = seed_end_idx
    while max_j < len(segments) - 1 and (segments[max_j + 1]["start"] - segments[max_j]["end"]) <= max_gap:
        max_j += 1

    # Search for the best (i, j) that encompasses the seed hook/explanation
    # and gets closest to target duration without exceeding hard ceiling.
    best_i, best_j = seed_start_idx, seed_end_idx
    best_score = float("inf")

    # We evaluate spans [i, j] covering seed_start_idx to seed_end_idx.
    # When expanding to fill a target, forward expansion (explanation/payoff)
    # is evaluated first, then backward expansion (setup).
    for i in range(seed_start_idx, min_i - 1, -1):
        for j in range(seed_end_idx, max_j + 1):
            s_start = segments[i]["start"]
            s_end = segments[j]["end"]
            s_dur = s_end - s_start
            if s_dur <= 0:
                continue
            if s_dur > float(config.HARD_MAX_CLIP_SECONDS) + 0.5:
                continue

            # Score by delta to target.
            delta = abs(s_dur - target)
            # Slight preference for forward context (payoff) over backward setup.
            forward_bonus = 0.05 * (j - seed_end_idx)
            score = delta - forward_bonus

            if score < best_score:
                best_score = score
                best_i, best_j = i, j

    best_start = segments[best_i]["start"]
    best_end = segments[best_j]["end"]
    best_dur = best_end - best_start
    target_met = abs(best_dur - target) <= tolerance

    updated = dict(candidate)
    updated["start"] = round(best_start, 2)
    updated["end"] = round(best_end, 2)
    updated["requested_duration"] = target
    updated["duration_target_met"] = target_met

    if not target_met and best_dur < target - tolerance:
        updated["duration_warning"] = (
            f"Requested duration was {target:.0f}s, but only {best_dur:.1f}s of coherent "
            f"contiguous context was available in this moment without adding unrelated footage."
        )
    elif not target_met and best_dur > target + tolerance:
        updated["duration_warning"] = (
            f"Requested duration was {target:.0f}s, but candidate was kept at {best_dur:.1f}s "
            "to avoid cutting off a sentence or payoff mid-thought."
        )
    else:
        updated.pop("duration_warning", None)

    return updated


def _filter_clips_to_bounds(clips, min_clip=None, max_clip=None,
                            min_tolerance=0.0, max_tolerance=0.0):
    """Filter candidate durations without trimming away a sentence or payoff."""
    filtered = []
    for clip in clips or []:
        try:
            duration = float(clip["end"]) - float(clip["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        if min_clip is not None and duration < float(min_clip) - min_tolerance:
            # If the clip has an explicit duration warning explaining why it could not be
            # extended further, preserve it rather than leaving zero clips!
            if clip.get("duration_warning"):
                filtered.append(clip)
            continue
        if max_clip is not None and duration > float(max_clip) + max_tolerance:
            continue
        filtered.append(clip)
    return filtered


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
    if target_duration is not None:
        min_s, max_s = _resolve_clip_bounds(min_s, max_s, target_duration)
        target_s = min(float(target_duration), config.HARD_MAX_CLIP_SECONDS)
    elif target_s <= 0:
        raise ValueError("The resolved target duration must be greater than zero.")

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
        start = max(0.0, start)
        if total_duration:
            if start >= total_duration:
                continue
            end = min(end, total_duration)
        if end <= start:
            continue
        length = end - start
        lower = (min_clip if min_clip is not None else config.MIN_CLIP_SECONDS) - 2
        upper = (max_clip if max_clip is not None else config.MAX_CLIP_SECONDS) + 3
        upper = min(upper, config.HARD_MAX_CLIP_SECONDS + 3)
        if length < max(3.0, lower) or length > upper:
            continue
        try:
            candidate_score = float(c.get("score", 50))
            if not math.isfinite(candidate_score):
                candidate_score = 50.0
        except (TypeError, ValueError):
            candidate_score = 50.0
        valid.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "title": (c.get("title") or "Untitled Clip").strip()[:80],
                "description": (c.get("description") or "").strip()[:220],
                "hook": (c.get("hook") or "").strip()[:120],
                "reason": (c.get("reason") or "").strip()[:500],
                "story_arc": (c.get("story_arc") or "").strip()[:300],
                "candidate_score": round(max(0.0, min(100.0, candidate_score)), 2),
                "evidence_quote": (c.get("evidence_quote") or "").strip()[:320],
            }
        )
    valid.sort(key=lambda x: x["start"])
    return valid


def _normalize_quote(text):
    """Normalize whitespace/case only so evidence still has to be verbatim."""
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _candidate_transcript_excerpt(transcript, candidate, context_seconds=4.0, max_chars=1800):
    """Build a compact, timestamped transcript excerpt around a candidate."""
    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
    except (KeyError, TypeError, ValueError):
        return ""
    lines = []
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        try:
            seg_start, seg_end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = (segment.get("text") or "").strip()
        if text and seg_end >= start - context_seconds and seg_start <= end + context_seconds:
            lines.append(f"[{seg_start:.2f} - {seg_end:.2f}] {text}")
    return "\n".join(lines)[:max_chars]


def _match_quote_segment(transcript, quote):
    """Return an actual transcript segment containing the exact review quote."""
    normalized_quote = _normalize_quote(quote)
    if not normalized_quote:
        return None
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        if normalized_quote in _normalize_quote(segment.get("text")):
            return segment
    return None


def _review_frame_contents(genai, prompt, visual_samples):
    contents = [prompt]
    for sample in visual_samples or []:
        if not isinstance(sample, dict) or not isinstance(sample.get("data"), bytes):
            continue
        timestamp = float(sample.get("timestamp") or 0.0)
        candidate_index = int(sample.get("candidate_index", -1))
        part = genai.types.Part.from_bytes(
            data=sample["data"], mime_type=sample.get("mime_type") or "image/jpeg"
        )
        contents.extend([
            f"Candidate {candidate_index} visual frame at {timestamp:.3f}s ({_format_visual_timestamp(timestamp)}).",
            part,
        ])
    return contents


def review_clip_candidates(
    transcript,
    candidates,
    *,
    api_key=None,
    model=None,
    visual_samples=None,
    total_duration=None,
    min_clip=None,
    max_clip=None,
    max_timestamp_shift=8.0,
):
    """Independently review candidates and accept only transcript-grounded edits.

    The reviewer can accept/reject candidates and suggest boundary evidence, but
    its numeric timestamps are never trusted. A moved boundary is derived from
    the real start/end of a transcript segment containing the reviewer's exact
    quoted words, then checked against duration limits.
    """
    candidates = list(candidates or [])
    if not candidates:
        return {"candidates": [], "rejected_count": 0, "reviewed_count": 0}

    from google import genai

    api_key = config.require(api_key or config.GEMINI_API_KEY, "GEMINI_API_KEY")
    model = model or config.GEMINI_MODEL
    compact_candidates = []
    for index, candidate in enumerate(candidates):
        excerpt = _candidate_transcript_excerpt(transcript, candidate)
        if not excerpt:
            raise ValueError(f"Candidate {index} has no transcript excerpt to review.")
        compact_candidates.append({
            "candidate_index": index,
            "start": candidate.get("start"),
            "end": candidate.get("end"),
            "title": candidate.get("title", ""),
            "hook": candidate.get("hook", ""),
            "description": candidate.get("description", ""),
            "story_arc": candidate.get("story_arc", ""),
            "transcript_excerpt": excerpt,
            "user_selected": bool(candidate.get("_user_selected")),
        })

    prompt = (
        "You are an independent quality reviewer for short-video clip candidates. "
        "Review each candidate separately against only its timestamped transcript excerpt "
        "and any timestamped still frames. Assess whether it starts/ends coherently, has "
        "a complete hook-to-payoff arc, is supported by the transcript, and whether its "
        "visual claim (if any) is visible in the supplied stills. Do not infer unseen "
        "events or speech. Return ONLY JSON with a 'reviews' array. Each review must have "
        "candidate_index, decision ('accept' or 'reject'), score (0-100), reason, "
        "transcript_evidence (an exact short quote copied from the excerpt), "
        "visual_evidence (what is directly visible, or 'not verifiable from sampled frames'), "
        "start_quote and end_quote (exact quotes from individual transcript segments only "
        "if a boundary should move; otherwise empty strings). Do not provide proposed "
        "timestamps: accepted boundary changes will be derived from the quoted segment times.\n\n"
        + json.dumps({"candidates": compact_candidates}, ensure_ascii=False)
    )
    client = genai.Client(api_key=api_key)
    contents = _review_frame_contents(genai, prompt, visual_samples)
    response = config.gemini_generate(client, model, contents)
    data = _load_json((response.text or "").strip())
    rows = data.get("reviews") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Independent candidate review returned no reviews array.")

    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("candidate_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(candidates) and index not in by_index:
            by_index[index] = row
    if len(by_index) != len(candidates):
        raise ValueError("Independent candidate review was incomplete; original timestamps were retained.")

    accepted = []
    rejected_count = 0
    for index, candidate in enumerate(candidates):
        row = by_index[index]
        decision = (row.get("decision") or "").strip().lower()
        if decision not in {"accept", "accepted", "keep", "approve", "approved", "reject", "rejected"}:
            raise ValueError(f"Independent candidate review returned an invalid decision for candidate {index}.")
        evidence = (row.get("transcript_evidence") or "").strip()
        excerpt = _candidate_transcript_excerpt(transcript, candidate)
        if not evidence or _normalize_quote(evidence) not in _normalize_quote(excerpt):
            raise ValueError(f"Candidate {index} review did not provide a verifiable transcript quote.")
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            raise ValueError(f"Candidate {index} review score was invalid.")
        if not math.isfinite(score):
            raise ValueError(f"Candidate {index} review score was invalid.")
        score = max(0.0, min(100.0, score))

        is_rejected = decision in {"reject", "rejected"}
        if is_rejected and not candidate.get("_user_selected"):
            rejected_count += 1
            continue

        reviewed = {
            **candidate,
            "review_status": "flagged_user_selected" if is_rejected else "accepted",
            "review_score": round(score, 2),
            "review_reason": (row.get("reason") or "").strip()[:500],
            "review_evidence": evidence[:320],
            "visual_evidence": (row.get("visual_evidence") or "").strip()[:320],
        }
        if is_rejected:
            reviewed["review_warning"] = "The independent reviewer flagged this user-selected moment; it was retained for user review."

        # The reviewer never controls numeric boundaries. If it supplies exact
        # segment quotes, derive each endpoint from that segment and apply strict
        # bounds; otherwise retain the original interval unchanged.
        start_quote = (row.get("start_quote") or "").strip()
        end_quote = (row.get("end_quote") or "").strip()
        start_segment = _match_quote_segment(transcript, start_quote) if start_quote else None
        end_segment = _match_quote_segment(transcript, end_quote) if end_quote else None
        if start_segment is not None and end_segment is not None:
            try:
                revised_start = float(start_segment["start"])
                revised_end = float(end_segment["end"])
                old_start, old_end = float(candidate["start"]), float(candidate["end"])
                lower_bound = float(min_clip) if min_clip is not None else 3.0
                upper_bound = min(
                    float(max_clip) if max_clip is not None else float(config.HARD_MAX_CLIP_SECONDS),
                    float(config.HARD_MAX_CLIP_SECONDS),
                )
                duration_ok = lower_bound <= revised_end - revised_start <= upper_bound
                video_ok = not total_duration or (
                    0 <= revised_start < revised_end <= float(total_duration)
                )
                local_ok = (
                    abs(revised_start - old_start) <= float(max_timestamp_shift)
                    and abs(revised_end - old_end) <= float(max_timestamp_shift)
                )
                if duration_ok and video_ok and local_ok:
                    if revised_start != old_start or revised_end != old_end:
                        reviewed["timestamp_revision"] = {
                            "from": {"start": round(old_start, 2), "end": round(old_end, 2)},
                            "to": {"start": round(revised_start, 2), "end": round(revised_end, 2)},
                            "evidence": "Exact start/end quotes matched transcript segments; endpoints use their timestamps.",
                        }
                    reviewed["start"] = round(revised_start, 2)
                    reviewed["end"] = round(revised_end, 2)
            except (KeyError, TypeError, ValueError, OverflowError):
                # Keep the original candidate untouched if segment timing is bad.
                pass
        accepted.append(reviewed)

    return {
        "candidates": accepted,
        "rejected_count": rejected_count,
        "reviewed_count": len(by_index),
    }


def _candidate_text_similarity(left, right):
    left_words = set(re.findall(r"[\w']+", " ".join(str(left.get(key, "")) for key in ("title", "hook", "description")).casefold()))
    right_words = set(re.findall(r"[\w']+", " ".join(str(right.get(key, "")) for key in ("title", "hook", "description")).casefold()))
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def _candidate_intervals_duplicate(left, right, overlap_threshold=0.35):
    try:
        left_start, left_end = float(left["start"]), float(left["end"])
        right_start, right_end = float(right["start"]), float(right["end"])
    except (KeyError, TypeError, ValueError):
        return False
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shorter = min(left_end - left_start, right_end - right_start)
    return shorter > 0 and overlap / shorter >= overlap_threshold


def rank_and_diversify_candidates(candidates, target_count):
    """Rank reviewed candidates and suppress temporal/textual near-duplicates."""
    try:
        limit = max(0, int(target_count))
    except (TypeError, ValueError):
        limit = 0
    if not limit:
        return []

    ranked = []
    for candidate in candidates or []:
        try:
            base_score = float(candidate.get("candidate_score", 50.0))
            if not math.isfinite(base_score):
                base_score = 50.0
        except (TypeError, ValueError):
            base_score = 50.0
        try:
            review_score = float(candidate.get("review_score", base_score))
            if not math.isfinite(review_score):
                review_score = base_score
        except (TypeError, ValueError):
            review_score = base_score
        score = (0.35 * max(0.0, min(100.0, base_score))) + (
            0.65 * max(0.0, min(100.0, review_score))
        )
        ranked.append({**candidate, "quality_score": round(score, 2)})

    ranked.sort(
        key=lambda item: (
            not bool(item.get("_user_selected")),
            -float(item.get("quality_score", 0)),
            float(item.get("start", 0)),
        )
    )
    selected = []
    for candidate in ranked:
        if any(
            _candidate_intervals_duplicate(candidate, prior)
            or _candidate_text_similarity(candidate, prior) >= 0.78
            for prior in selected
        ):
            continue
        candidate = {**candidate, "candidate_rank": len(selected) + 1}
        candidate.pop("_user_selected", None)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: float(item.get("start", 0)))
    return selected
