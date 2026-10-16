"""Story-first edit planning: source timeline -> edit timeline.

The planner turns a raw candidate moment (``[start, end]`` from analysis) into a
structured edit plan *before* anything is rendered. Every plan separates three
timelines:

1. **Source timeline** — original video timestamps and sentence-level units.
2. **Edit timeline** — the ordered passages (optional AI narration card, opening,
   body, conclusion) mapped to output timestamps.
3. **Rendered timeline** — verified after FFmpeg by the technical QA layer; the
   planner only estimates it (``estimated_duration``) and never assumes the two
   are interchangeable.

Duration is a planning constraint (default tolerance +/-10%), never a blind
segment-extension rule: the planner repairs the opening and the conclusion first
(expanding context before/after the topic, at sentence boundaries only), then
shapes the story toward the requested length. Story coherence wins over hitting
an exact number, and any shortfall is reported explicitly instead of padding
with unrelated footage or cutting a sentence in half.

The planner is deterministic and dependency-free so it can be unit-tested with
synthetic transcripts and mocked narration responses.
"""
import math
import re

from . import config

# --- Tunables ---------------------------------------------------------------

# Requested duration is a preference band, not an exact target.
DURATION_TOLERANCE = 0.10

# Hard ceiling is a product rule (2 minutes) and is never configurable.
HARD_MAX_CLIP_SECONDS = float(config.HARD_MAX_CLIP_SECONDS)

# How far the opening/conclusion repair may walk through the source.
OPENING_LOOKBACK_SECONDS = 30.0
CONCLUSION_LOOKAHEAD_SECONDS = 30.0

# A real spoken sentence never spans this long; units beyond it are lumped
# transcript segments and cannot be trusted as sentence boundaries.
_MAX_SENTENCE_SECONDS = 30.0

# Window for searching relevant contextual material near the topic.
CONTEXT_SEARCH_WINDOW_SECONDS = 90.0

# Speaking rate used for narration duration estimates (~156 words/minute).
SPEAKING_RATE_WORDS_PER_SECOND = 2.6

# Minimum length of a generated supplemental narration card.
MIN_NARRATION_SECONDS = 2.5
MAX_NARRATION_SECONDS = 25.0

GENERATION_MODE_SOURCE_ONLY = "source_only"
GENERATION_MODE_AI_EXPANDED = "ai_expanded"
GENERATION_MODES = (GENERATION_MODE_SOURCE_ONLY, GENERATION_MODE_AI_EXPANDED)

# Label burned onto AI narration cards so generated speech is always
# distinguishable from the original speaker's words.
AI_NARRATION_LABEL = "AI narration (quotes from the transcript)"

_DANGLING_STARTERS = {
    "and", "but", "so", "or", "also", "then", "because", "which", "that", "plus",
    "now", "well", "like", "okay", "ok", "right", "yeah", "yes", "no", "actually",
    "basically", "literally", "however", "therefore", "moreover", "furthermore",
    "besides", "instead", "meanwhile", "afterwards", "afterward", "anyway",
    "anyways", "anyhow", "thus", "hence", "meanwhile", "anyway", "although",
    "though", "whereas", "meanwhile", "accordingly", "consequently", "otherwise",
    "specifically", "particularly", "especially", "remember", "again", "anyway",
    "i mean", "you know", "kind of", "sort of",
}

_DANGLING_ENDINGS = {
    "and", "but", "or", "so", "the", "a", "an", "of", "to", "with", "on", "in",
    "at", "from", "for", "is", "was", "are", "were", "be", "been", "being",
    "that", "which", "who", "whom", "this", "these", "those", "it", "they",
    "he", "she", "we", "you", "i", "my", "our", "your", "their", "his", "her",
    "its", "as", "if", "when", "while", "because", "about", "into", "over",
    "under", "than", "then", "by", "not", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "may", "might", "must",
    "there", "here", "where", "how", "what", "when", "whether", "both", "either",
    "neither", "each", "every", "some", "any", "all", "most", "more", "less",
    "during", "before", "after", "until", "unless", "since", "through",
}

# Very common words carry no topical signal for relevance scoring.
_STOPWORDS = {
    "the", "and", "that", "this", "with", "for", "you", "your", "they", "them",
    "their", "have", "has", "had", "was", "were", "are", "from", "there", "their",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "not",
    "but", "can", "will", "would", "could", "should", "shall", "may", "might",
    "must", "about", "into", "than", "then", "them", "these", "those", "some",
    "any", "all", "more", "most", "other", "such", "only", "very", "also", "just",
    "because", "while", "after", "before", "during", "through", "over", "under",
    "again", "ever", "never", "always", "really", "like", "know", "think", "want",
    "going", "right", "well", "yeah", "okay", "here", "now", "get", "got", "let",
    "say", "said", "tell", "told", "make", "made", "take", "took", "come", "came",
    "one", "two", "three", "way", "thing", "things", "people", "something",
    "anything", "everything", "nothing", "still", "even", "much", "many", "every",
    "each", "between", "both", "too", "own", "same", "being", "been", "does",
    "did", "doing", "having", "him", "her", "she", "himself", "herself", "itself",
    "ourselves", "themselves", "myself", "yourself", "yours", "ours", "mine",
    "out", "off", "down", "back", "here's", "there's", "it's", "don't", "i'm",
    "i've", "we're", "you're", "they're", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "can't", "cannot", "couldn't", "wouldn't",
    "shouldn't", "mustn't", "let's", "that's", "what's", "who's", "how's",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_TERMINAL_RE = re.compile(r"[.!?…][\"'”’)\]]*$")
_TRAILING_CONNECTIVE_RE = re.compile(
    r"[,\:;\—\–\-…]+$|"
    r"\b(and|but|or|so|the|a|an|of|to|with|on|in|at|from|for|is|was|are|were|"
    r"be|been|that|which|who|this|these|those|it|they|he|she|we|you|as|if|"
    r"when|while|because|about|into|than|then|by|not|there|here|where|how|"
    r"what|whether|since|until|unless|after|before|during|through|over|under|"
    r"than|such|both|either|neither)\s*$",
    re.IGNORECASE,
)
_QUOTE_STRIP_RE = re.compile(r"^[\"'“‘(\[\s]+|[\"'”’)\]\s]+$")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_DIGIT_RE = re.compile(r"\d")


# --- Public helpers ---------------------------------------------------------

def normalize_generation_mode(value):
    """Return a valid generation mode; reject unknown values loudly."""
    if value is None:
        return GENERATION_MODE_SOURCE_ONLY
    mode = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "original": GENERATION_MODE_SOURCE_ONLY,
        "original_content_only": GENERATION_MODE_SOURCE_ONLY,
        "source": GENERATION_MODE_SOURCE_ONLY,
        "none": GENERATION_MODE_SOURCE_ONLY,
        "ai": GENERATION_MODE_AI_EXPANDED,
        "ai_expansion": GENERATION_MODE_AI_EXPANDED,
        "expanded": GENERATION_MODE_AI_EXPANDED,
    }
    mode = aliases.get(mode, mode)
    if mode not in GENERATION_MODES:
        raise ValueError(
            "generation_mode must be 'source_only' (original content only) or "
            "'ai_expanded'."
        )
    return mode


def duration_tolerance_window(target_duration, tolerance_ratio=DURATION_TOLERANCE):
    """Return the (lower, upper) planning window for a requested duration."""
    target = float(target_duration)
    tolerance = max(0.5, abs(target) * float(tolerance_ratio))
    return target - tolerance, target + tolerance


def estimate_speech_duration(text):
    """Estimate how long ``text`` takes to speak at the configured rate."""
    words = [w for w in _WORD_RE.findall(str(text or "")) if w]
    return max(0.0, len(words) / SPEAKING_RATE_WORDS_PER_SECOND)


def content_words(text):
    """Content words used for topical relevance scoring."""
    words = []
    for word in _WORD_RE.findall(str(text or "").lower()):
        if len(word) >= 3 and word not in _STOPWORDS:
            words.append(word)
    return set(words)


def relevance_score(text, keywords):
    """Simple, deterministic relevance of a unit to the topic keywords."""
    words = content_words(text)
    if not words or not keywords:
        return 0.0
    hits = len(words & set(keywords))
    return hits / math.sqrt(len(words))


def starts_like_fragment(text):
    """True when text opens like the middle of a thought, not an introduction."""
    cleaned = _QUOTE_STRIP_RE.sub("", str(text or "")).strip()
    if not cleaned:
        return True
    first = cleaned.split()[0] if cleaned.split() else ""
    first_word = re.sub(r"[^\w']+", "", first).lower()
    if first_word in _DANGLING_STARTERS:
        return True
    if re.match(r"^[,;:.…\-—]", cleaned):
        return True
    # A lowercase opening usually continues a previous spoken sentence.
    if cleaned[0].islower():
        return True
    # Fewer than two words cannot establish a topic.
    if len(cleaned.split()) < 2:
        return True
    return False


def ends_like_fragment(text):
    """True when text closes mid-thought (dangling word or no conclusion)."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return True
    if _TRAILING_CONNECTIVE_RE.search(cleaned):
        return True
    words = cleaned.split()
    last_word = re.sub(r"[^\w']+", "", words[-1]).lower() if words else ""
    if last_word in _DANGLING_ENDINGS:
        return True
    if not _TERMINAL_RE.search(cleaned):
        # Whisper output is inconsistent with punctuation: treat a missing
        # terminal mark as incomplete only when the phrase clearly trails off.
        return cleaned.endswith(("...", "…", ",", ";", ":", "-", "—"))
    return False


def is_complete_unit(text):
    """A unit that can open and close a standalone clip."""
    return not starts_like_fragment(text) and not ends_like_fragment(text)


# --- Sentence-level source units -------------------------------------------

def _split_segment_text(text):
    """Split a segment's text at sentence boundaries, keeping punctuation."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(str(text or "").strip())]
    return [p for p in parts if p]


def build_sentence_units(transcript):
    """Group transcript segments into sentence-level source units.

    Consecutive segments accumulate until the accumulated text ends a sentence
    (terminal punctuation) or a silence gap separates them, so a cut can only
    ever land on a thought boundary. A single segment containing several
    sentences is split with proportional timing. Each unit keeps its original
    source timestamps — the source timeline is never rewritten.
    """
    segments = []
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = (segment.get("text") or "").strip()
        if not text or end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda item: (item["start"], item["end"]))

    units = []
    buffer = None

    def _flush():
        nonlocal buffer
        if buffer is None:
            return
        text = " ".join(buffer["text_parts"]).strip()
        if text:
            units.append({
                "start": round(buffer["start"], 3),
                "end": round(buffer["end"], 3),
                "text": text,
                "ends_sentence": bool(_TERMINAL_RE.search(text)),
            })
        buffer = None

    for segment in segments:
        parts = _split_segment_text(segment["text"])
        if not parts:
            continue
        span = segment["end"] - segment["start"]
        total_chars = sum(len(p) for p in parts) or 1
        cursor = segment["start"]
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                part_end = segment["end"]
            else:
                part_end = min(segment["end"], cursor + span * (len(part) / total_chars))
            if buffer is None:
                buffer = {
                    "start": cursor,
                    "end": part_end,
                    "text_parts": [part],
                }
            else:
                buffer["end"] = part_end
                buffer["text_parts"].append(part)
            closes = bool(_TERMINAL_RE.search(part))
            next_gap = None
            # Close at a pause before the next segment as a thought boundary.
            if index == len(parts) - 1:
                following = next(
                    (s for s in segments if s["start"] >= segment["end"] - 1e-6),
                    None,
                )
                if following is not None:
                    next_gap = following["start"] - segment["end"]
            if closes or (next_gap is not None and next_gap > 1.2):
                _flush()
            cursor = part_end
    _flush()
    return units


# --- Narration --------------------------------------------------------------

def build_narration_script(key_units, target_words):
    """Compose supplemental narration from verbatim transcript quotes.

    Framing lines assert nothing about the world beyond "these are the
    speaker's words"; every factual phrase is a verbatim unit quote that is
    recorded in ``grounding_quotes`` for the editorial QA to verify against the
    transcript. No statistics, quotations, events or claims are invented.
    """
    quotes = []
    seen = set()
    words_used = 0
    for unit in key_units:
        text = str(unit.get("text") or "").strip()
        if not text or text in seen:
            continue
        word_count = len(_WORD_RE.findall(text))
        if quotes and words_used + word_count > max(8, int(target_words) * 1.15):
            break
        seen.add(text)
        quotes.append(text)
        words_used += word_count
        if words_used >= target_words:
            break
    if not quotes:
        return "", []

    frame = (
        "Before the clip, here is the context in the speaker's own words. "
    ) if len(quotes) == 1 else (
        "Before the clip, here is the context in the speaker's own words. "
    )
    spoken = " ".join(f'"{q}"' for q in quotes)
    script = f"{frame}{spoken}".strip()
    return script, quotes


def _pick_narration_units(units, selected_indices, keywords, budget_words):
    """Choose the most relevant complete units to quote in the narration."""
    candidates = []
    for index, unit in enumerate(units):
        if index in selected_indices:
            continue
        if starts_like_fragment(unit["text"]) and ends_like_fragment(unit["text"]):
            continue
        score = relevance_score(unit["text"], keywords)
        if score <= 0:
            continue
        candidates.append((score, -abs(index - min(selected_indices or [0])), index))
    candidates.sort(reverse=True)
    picked = []
    total = 0
    for _, _, index in candidates:
        words = len(_WORD_RE.findall(units[index]["text"]))
        if picked and total + words > budget_words * 1.15:
            continue
        picked.append(units[index])
        total += words
        if total >= budget_words:
            break
    return picked


# --- Timeline mapping -------------------------------------------------------

def map_source_time_to_edit_time(plan, source_time):
    """Map a source-timeline timestamp onto the edit timeline (or None)."""
    for segment in plan.get("segments", []) or []:
        if segment.get("origin") != "source":
            continue
        start = float(segment.get("source_start", 0.0))
        end = float(segment.get("source_end", 0.0))
        if start - 1e-6 <= float(source_time) <= end + 1e-6:
            edit_start = float(segment.get("edit_start", 0.0))
            return edit_start + (float(source_time) - start)
    return None


def finalize_narration_timing(plan, actual_duration):
    """Replace the narration estimate with the measured synthesis duration.

    Re-times the edit timeline (narration card first, then the source
    passages) so caption rebasing and rendered-duration checks use the real
    audio length. Mutates and returns the plan; never changes source
    timestamps.
    """
    narration = plan.get("narration")
    if not narration:
        return plan
    try:
        actual = float(actual_duration)
    except (TypeError, ValueError):
        return plan
    if not math.isfinite(actual) or actual <= 0:
        return plan

    narration["actual_duration"] = round(actual, 2)
    cursor = 0.0
    for segment in plan.get("segments", []):
        try:
            old_start = float(segment.get("edit_start", 0.0))
            old_end = float(segment.get("edit_end", 0.0))
        except (TypeError, ValueError):
            old_start, old_end = 0.0, 0.0
        if segment.get("origin") == "ai":
            duration = actual
        else:
            duration = max(0.0, old_end - old_start)
        segment["edit_start"] = round(cursor, 3)
        segment["edit_end"] = round(cursor + duration, 3)
        cursor += duration
    plan["estimated_duration"] = round(cursor, 2)
    plan["source_duration"] = round(cursor - actual, 2)
    return plan


def drop_narration(plan):
    """Restore a plan to its source-only timeline after a failed narration render.

    Removes the AI narration segment and re-times the edit timeline so
    duration metadata and caption rebasing always describe what is actually
    delivered (the untouched source clip) — never narration that was planned
    but never rendered. The narration script is preserved under
    ``narration_failed`` for reporting. Idempotent.
    """
    narration = plan.get("narration")
    has_ai_segments = any(
        segment.get("origin") == "ai" for segment in plan.get("segments", []) or []
    )
    if not narration and not has_ai_segments:
        return plan

    plan["narration_failed"] = dict(narration or {})
    plan["narration"] = None
    plan["segments"] = [
        segment
        for segment in plan.get("segments", []) or []
        if segment.get("origin") != "ai"
    ]
    cursor = 0.0
    for segment in plan["segments"]:
        try:
            old_start = float(segment.get("edit_start", 0.0))
            old_end = float(segment.get("edit_end", 0.0))
        except (TypeError, ValueError):
            old_start, old_end = 0.0, 0.0
        duration = max(0.0, old_end - old_start)
        segment["edit_start"] = round(cursor, 3)
        segment["edit_end"] = round(cursor + duration, 3)
        cursor += duration
    plan["estimated_duration"] = round(cursor, 2)
    plan["source_duration"] = round(cursor, 2)
    return plan


def rebase_cues(cues, plan, clip_source_start=0.0):
    """Rebase clip-relative caption cues onto the edit timeline.

    Cues use clip-relative source time (0 = ``clip_source_start``). Every cue is
    mapped through the plan's source->edit segments so captions stay aligned
    after a narration card is prepended or passages are reordered. Cues that
    would cross a clip boundary are clipped to it; cues fully outside every
    planned segment are dropped.
    """
    rebased = []
    for cue in cues or []:
        try:
            start = float(cue["start"]) + float(clip_source_start)
            end = float(cue["end"]) + float(clip_source_start)
        except (KeyError, TypeError, ValueError):
            continue
        edit_start = map_source_time_to_edit_time(plan, start)
        edit_end = map_source_time_to_edit_time(plan, end)
        if edit_start is None or edit_end is None:
            continue
        if edit_end <= edit_start:
            continue
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        rebased.append({
            "start": round(edit_start, 3),
            "end": round(edit_end, 3),
            "text": text,
        })
    return rebased


# --- Duration reporting -----------------------------------------------------

def duration_mismatch_warning(target_duration, actual_duration,
                              tolerance_ratio=DURATION_TOLERANCE, context=""):
    """Actionable warning when a duration misses the planning window.

    Returns None when the duration is within tolerance or no target exists.
    """
    if target_duration is None:
        return None
    try:
        target = float(target_duration)
        actual = float(actual_duration)
    except (TypeError, ValueError):
        return None
    lower, upper = duration_tolerance_window(target, tolerance_ratio)
    if lower <= actual <= upper:
        return None
    delta = actual - target
    direction = "shorter" if delta < 0 else "longer"
    prefix = f"{context}: " if context else ""
    return (
        f"{prefix}requested {target:.0f}s but planned/actual duration is "
        f"{actual:.1f}s ({abs(delta):.1f}s {direction}, tolerance "
        f"+/-{target * tolerance_ratio:.1f}s). Complete sentences and the "
        "payoff were preserved over hitting the exact number. Adjust the "
        "target, pick a richer moment, or enable AI-expanded mode to fill the "
        "gap with labelled transcript-grounded narration."
    )


# --- The planner ------------------------------------------------------------

def _keyword_set(topic, candidate, selected_texts):
    parts = [topic or ""]
    for key in ("title", "description", "hook", "reason", "story_arc", "evidence_quote"):
        parts.append(str(candidate.get(key) or ""))
    parts.extend(selected_texts or [])
    return content_words(" ".join(parts))


def _unit_overlaps(unit, start, end):
    return unit["end"] > start + 0.05 and unit["start"] < end - 0.05


def _find_covering_index(units, position):
    for index, unit in enumerate(units):
        if unit["start"] <= position < unit["end"]:
            return index
    # Nearest unit by start distance for positions between units.
    return min(
        range(len(units)),
        key=lambda i: min(
            abs(units[i]["start"] - position),
            abs(units[i]["end"] - position),
        ),
    )


def _repair_opening(units, first_index, budget_seconds):
    """Walk back to a complete standalone opener within the lookback budget."""
    anchor = units[first_index]["start"]
    repaired = False
    index = first_index
    while index > 0 and (anchor - units[index - 1]["end"]) <= budget_seconds:
        text = units[index]["text"]
        if not starts_like_fragment(text):
            break
        index -= 1
        repaired = True
    # Even a non-fragment opener may need one sentence of prior setup when the
    # original candidate began mid-explanation (its first unit ran past the
    # requested start).
    return index, repaired


def _repair_conclusion(units, last_index, budget_seconds, evidence_unit_index=None):
    """Walk forward to a finished closing sentence within the lookahead budget."""
    anchor = units[last_index]["end"]
    repaired = False
    index = last_index
    while index + 1 < len(units) and (units[index + 1]["start"] - anchor) <= budget_seconds:
        text = units[index]["text"]
        if not ends_like_fragment(text):
            break
        index += 1
        repaired = True
    if evidence_unit_index is not None and evidence_unit_index > index:
        # The promised payoff lives after the initial range: never cut it off.
        if units[evidence_unit_index]["end"] - anchor <= budget_seconds:
            index = evidence_unit_index
            repaired = True
    return index, repaired


def _find_evidence_unit(units, candidate):
    quote = str(candidate.get("evidence_quote") or "").strip()
    if len(quote) < 6:
        return None
    normalized_quote = re.sub(r"\s+", " ", quote.casefold()).strip()
    for index, unit in enumerate(units):
        normalized_text = re.sub(r"\s+", " ", unit["text"].casefold()).strip()
        if normalized_quote in normalized_text:
            return index
    return None


def _contiguous_expansion(units, lo, hi, keywords, budget_seconds, total_duration):
    """Grow the selection with relevant neighbouring units at sentence edges."""
    original_duration = units[hi]["end"] - units[lo]["start"]
    limit = min(original_duration + max(0.0, budget_seconds), HARD_MAX_CLIP_SECONDS)
    while True:
        grew = False
        before = None
        after = None
        if lo > 0:
            candidate = units[lo - 1]
            before = (relevance_score(candidate["text"], keywords), -1, candidate)
        if hi + 1 < len(units):
            candidate = units[hi + 1]
            after = (relevance_score(candidate["text"], keywords), 1, candidate)

        options = [item for item in (before, after) if item is not None]
        options.sort(key=lambda item: (item[0], item[1] > 0), reverse=True)
        for score, direction, unit in options:
            if score <= 0:
                continue
            new_lo = lo - 1 if direction < 0 else lo
            new_hi = hi + 1 if direction > 0 else hi
            start = units[new_lo]["start"]
            end = units[new_hi]["end"]
            if total_duration and end > float(total_duration) + 0.25:
                continue
            new_duration = end - start
            if new_duration > limit:
                continue
            lo, hi = new_lo, new_hi
            grew = True
            break
        if not grew:
            break
        if (units[hi]["end"] - units[lo]["start"]) >= limit:
            break
    return lo, hi, max(0.0, (units[hi]["end"] - units[lo]["start"]) - original_duration)

def _shrink_to_window(units, lo, hi, target, upper_limit):
    """Shrink a too-long story into the tolerance window.

    Only the front of the body is trimmed (unit boundary by unit boundary) so
    the complete conclusion/payoff always survives. The new start must itself
    be a viable standalone opener — a sentence is never cut in half to fit.
    Returns (lo, hi, trimmed).
    """
    def _duration(a, b):
        return units[b]["end"] - units[a]["start"]

    if _duration(lo, hi) <= upper_limit:
        return lo, hi, False

    best = None
    for new_lo in range(lo, hi + 1):
        if new_lo < hi and starts_like_fragment(units[new_lo]["text"]):
            continue
        duration = _duration(new_lo, hi)
        if duration > upper_limit:
            # Keep trimming more setup from the front.
            continue
        distance = abs(duration - target)
        if best is None or distance < best[0]:
            best = (distance, new_lo, duration)
        # Front-trimming shrinks duration monotonically, so the first window
        # that fits keeps the most context while respecting the limit.
        break
    if best is None:
        # Even the conclusion alone exceeds the window; keep the coherent
        # story and let the caller report the mismatch.
        return lo, hi, False
    _, new_lo, _ = best
    return new_lo, hi, new_lo != lo


def build_edit_plan(transcript, candidate, *, target_duration=None,
                    generation_mode=GENERATION_MODE_SOURCE_ONLY, topic=None,
                    custom_prompt=None, total_duration=None):
    """Plan a complete, standalone edit for one candidate moment.

    Returns a JSON-serializable plan describing the opening, body and
    conclusion passages (original source timestamps), their edit-timeline
    mapping, an optional transcript-grounded narration card, duration status
    with an explicit explanation, and warnings for anything the user must know
    (shortfalls, repaired boundaries, insufficient source material).
    """
    mode = normalize_generation_mode(generation_mode)
    candidate = candidate or {}
    try:
        requested_start = float(candidate.get("start", 0.0))
        requested_end = float(candidate.get("end", 0.0))
    except (TypeError, ValueError):
        raise ValueError("Candidate start/end must be numeric seconds.")
    if requested_end <= requested_start:
        raise ValueError("Candidate end must be later than candidate start.")
    if total_duration:
        requested_start = max(0.0, min(requested_start, float(total_duration)))
        requested_end = max(0.0, min(requested_end, float(total_duration)))
        if requested_end <= requested_start:
            raise ValueError("Candidate lies outside the source duration.")

    units = build_sentence_units(transcript)
    if not units:
        raise ValueError("Transcript has no timestamped sentences to plan an edit from.")

    # A per-moment target (analysis angle) overrides the job-level default so
    # multiple requested clips stay independent of each other.
    if candidate.get("target_duration") is not None:
        target_duration = candidate.get("target_duration")
    target = None
    if target_duration is not None:
        try:
            target = float(target_duration)
        except (TypeError, ValueError):
            raise ValueError("target_duration must be between 5 and 120 seconds.")
        if not 5.0 <= target <= HARD_MAX_CLIP_SECONDS:
            raise ValueError("target_duration must be between 5 and 120 seconds.")

    topic_text = topic or " ".join(
        str(candidate.get(key) or "") for key in ("title", "hook", "description")
    ).strip()

    # 1. Matched source units (whole sentences only — never partial).
    matched = [i for i, unit in enumerate(units) if _unit_overlaps(unit, requested_start, requested_end)]
    if not matched:
        matched = [_find_covering_index(units, (requested_start + requested_end) / 2.0)]
    # Degenerate transcripts (one lumped "sentence" spanning most of the
    # video) have no usable sentence granularity. Fall back to the candidate's
    # own bounds for those units instead of swallowing the whole video. A unit
    # is lumped when it is far longer than the candidate (or absurdly long for
    # spoken sentences); normal sentence-sized units are kept whole so a cut
    # never lands mid-thought.
    if matched:
        candidate_span = requested_end - requested_start
        for index in (matched[0], matched[-1]):
            unit = units[index]
            unit_span = unit["end"] - unit["start"]
            lumped = (
                unit_span > _MAX_SENTENCE_SECONDS
                or unit_span > 2.0 * candidate_span + 5.0
            )
            if not lumped:
                continue
            if index == matched[0] and requested_start > unit["start"] + 0.25:
                unit["start"] = min(requested_start, unit["end"] - 0.5)
            if index == matched[-1] and requested_end < unit["end"] - 0.25:
                unit["end"] = max(requested_end, unit["start"] + 0.5)

    lo, hi = matched[0], matched[-1]

    warnings = []
    # 2. Standalone introduction: repair a dangling/mid-thought opening.
    original_lo = lo
    lo, opening_repaired = _repair_opening(units, lo, OPENING_LOOKBACK_SECONDS)
    if requested_start > units[original_lo]["start"] + 0.25 or original_lo != lo:
        opening_repaired = True
    if starts_like_fragment(units[lo]["text"]):
        warnings.append(
            "Opening could not be repaired to a complete standalone sentence "
            "within the source lookback window; the clip may assume earlier "
            "context."
        )

    # 3. Complete conclusion: repair a dangling ending, keep the promised payoff.
    original_hi = hi
    evidence_index = _find_evidence_unit(units, candidate)
    hi, conclusion_repaired = _repair_conclusion(
        units, hi, CONCLUSION_LOOKAHEAD_SECONDS, evidence_unit_index=evidence_index
    )
    if requested_end < units[original_hi]["end"] - 0.25:
        conclusion_repaired = True
    if ends_like_fragment(units[hi]["text"]):
        warnings.append(
            "Conclusion could not be extended to a finished sentence within "
            "the source lookahead window; the ending may feel abrupt."
        )

    # 4. Relevance keywords for contextual expansion / narration selection.
    core_texts = [units[i]["text"] for i in range(lo, hi + 1)]
    keywords = _keyword_set(topic_text, candidate, [] if candidate.get("_montage_selected") else core_texts)

    core_duration = units[hi]["end"] - units[lo]["start"]

    # 5. Shape toward the requested duration at sentence boundaries only.
    source_lo, source_hi = lo, hi
    expansion_added = 0.0
    trimmed = False
    narration = None
    insufficient_source = False

    if target is not None:
        lower, upper = duration_tolerance_window(target)
        lower = max(0.5, lower)
        upper = min(HARD_MAX_CLIP_SECONDS, max(upper, lower + 1.0))

        if core_duration > upper:
            source_lo, source_hi, trimmed = _shrink_to_window(units, lo, hi, target, upper)
            core_duration = units[source_hi]["end"] - units[source_lo]["start"]

        if core_duration < lower:
            budget = min(upper - core_duration + 2.0, CONTEXT_SEARCH_WINDOW_SECONDS)
            source_lo, source_hi, expansion_added = _contiguous_expansion(
                units, source_lo, source_hi, keywords, budget,
                total_duration=total_duration,
            )
            core_duration = units[source_hi]["end"] - units[source_lo]["start"]

    selected_indices = set(range(source_lo, source_hi + 1))
    source_duration = core_duration

    # 6. Report or fill any remaining gap — never pad with unrelated footage.
    duration_status = "no_target"
    duration_explanation = (
        "No target duration was requested; the edit keeps the complete "
        "selected story."
    )
    estimated_duration = source_duration

    if target is not None:
        lower, upper = duration_tolerance_window(target)
        if source_duration < max(0.5, lower):
            shortfall = max(0.0, target - source_duration)
            insufficient_source = True
            if mode == GENERATION_MODE_AI_EXPANDED:
                budget_words = max(6, int(shortfall * SPEAKING_RATE_WORDS_PER_SECOND))
                key_units = _pick_narration_units(
                    units, selected_indices, keywords, budget_words
                )
                if not key_units:
                    key_units = [
                        units[i]
                        for i in sorted(selected_indices)
                        if is_complete_unit(units[i]["text"])
                    ][:2]
                script, grounding_quotes = build_narration_script(key_units, budget_words)
                if script:
                    narration = {
                        "script": script,
                        "grounding_quotes": grounding_quotes,
                        "estimated_duration": round(
                            max(MIN_NARRATION_SECONDS,
                                min(MAX_NARRATION_SECONDS, estimate_speech_duration(script))),
                            2,
                        ),
                        "actual_duration": None,
                        "position": "intro",
                        "origin": "ai",
                        "label": AI_NARRATION_LABEL,
                    }
                    estimated_duration = source_duration + narration["estimated_duration"]
            if estimated_duration < max(0.5, lower):
                duration_status = "short"
                duration_explanation = (
                    f"Requested {target:.0f}s but only {source_duration:.1f}s of "
                    f"relevant source material exists for this topic"
                    + (
                        f"; transcript-grounded AI narration of about "
                        f"{narration['estimated_duration']:.0f}s was planned to fill the gap "
                        f"(final estimate {estimated_duration:.1f}s)."
                        if narration else
                        ". A source-only edit keeps the complete story instead of "
                        "padding with unrelated footage or cutting sentences. "
                        "AutoClipper keeps source audio and never generates narration; "
                        "lower the target duration to fit the available source story."
                    )
                )
                warnings.append(duration_explanation)
            else:
                duration_status = "within_tolerance" if narration else "short"
                duration_explanation = (
                    f"Requested {target:.0f}s; the complete source story is "
                    f"{source_duration:.1f}s and AI narration of about "
                    f"{narration['estimated_duration']:.0f}s brings the edit to "
                    f"about {estimated_duration:.1f}s."
                )
        elif source_duration > min(HARD_MAX_CLIP_SECONDS, max(0.5, upper)):
            duration_status = "long"
            duration_explanation = (
                f"Requested {target:.0f}s but the complete story (including its "
                f"opening and payoff) is {source_duration:.1f}s. Story coherence "
                "takes priority: the clip keeps finished sentences rather than "
                "truncating them to hit the number."
            )
            warnings.append(duration_explanation)
        else:
            duration_status = "within_tolerance"
            duration_explanation = (
                f"Requested {target:.0f}s; the planned complete story is "
                f"{source_duration:.1f}s (tolerance +/-"
                f"{target * DURATION_TOLERANCE:.1f}s)."
            )

    # 7. Build the edit timeline (source passages, optional narration card).
    segments = []
    edit_cursor = 0.0
    if narration is not None and narration.get("position") == "intro":
        segments.append({
            "role": "narration_intro",
            "origin": "ai",
            "text": narration["script"],
            "source_start": None,
            "source_end": None,
            "edit_start": round(edit_cursor, 3),
            "edit_end": round(edit_cursor + narration["estimated_duration"], 3),
        })
        edit_cursor += narration["estimated_duration"]

    roles = {}
    for index in range(source_lo, source_hi + 1):
        if index == source_lo:
            roles[index] = "opening"
        elif index == source_hi:
            roles[index] = "conclusion"
        else:
            roles[index] = "body"
    if source_lo == source_hi:
        roles[source_lo] = "opening_conclusion"

    body_segments = []
    for index in range(source_lo, source_hi + 1):
        unit = units[index]
        duration = unit["end"] - unit["start"]
        entry = {
            "role": roles[index],
            "origin": "source",
            "text": unit["text"],
            "source_start": round(unit["start"], 3),
            "source_end": round(unit["end"], 3),
            "edit_start": round(edit_cursor, 3),
            "edit_end": round(edit_cursor + duration, 3),
        }
        segments.append(entry)
        edit_cursor += duration
        if entry["role"] == "body":
            body_segments.append({
                "text": unit["text"],
                "source_start": entry["source_start"],
                "source_end": entry["source_end"],
                "origin": "source",
            })

    opening_unit = units[source_lo]
    conclusion_unit = units[source_hi]
    opening = {
        "text": opening_unit["text"],
        "source_start": round(opening_unit["start"], 3),
        "source_end": round(opening_unit["end"], 3),
        "origin": "source",
    }
    conclusion = {
        "text": conclusion_unit["text"],
        "source_start": round(conclusion_unit["start"], 3),
        "source_end": round(conclusion_unit["end"], 3),
        "origin": "source",
    }

    central_message = (
        str(candidate.get("story_arc") or "").strip()
        or str(candidate.get("reason") or "").strip()
        or str(candidate.get("description") or "").strip()
        or topic_text
    ).strip()

    plan_explanation = (
        f"Planned a standalone edit of about {estimated_duration:.1f}s from the "
        f"source timeline [{opening['source_start']:.1f}s-{conclusion['source_end']:.1f}s]: "
        f"an opening ('{opening['text'][:60]}'), "
        f"{len(body_segments)} development passage(s), and a complete closing "
        f"line ('{conclusion['text'][:60]}')."
    )
    if opening_repaired:
        plan_explanation += " The opening was extended back to a complete sentence."
    if conclusion_repaired:
        plan_explanation += " The ending was extended forward to a finished thought."
    if narration is not None:
        plan_explanation += (
            f" A clearly labelled AI narration card of about "
            f"{narration['estimated_duration']:.0f}s (verbatim transcript quotes) "
            "completes the requested length."
        )

    return {
        "version": 1,
        "mode": mode,
        "topic": topic_text[:200],
        "central_message": central_message[:300],
        "target_duration": round(target, 2) if target is not None else None,
        "duration_tolerance_ratio": DURATION_TOLERANCE,
        "duration_tolerance": round((target or 0.0) * DURATION_TOLERANCE, 2),
        "estimated_duration": round(estimated_duration, 2),
        "source_duration": round(source_duration, 2),
        "requested_start": round(requested_start, 2),
        "requested_end": round(requested_end, 2),
        "source_start": round(units[source_lo]["start"], 3),
        "source_end": round(units[source_hi]["end"], 3),
        "duration_status": duration_status,
        "duration_explanation": duration_explanation,
        "insufficient_source": insufficient_source,
        "opening_repaired": opening_repaired,
        "conclusion_repaired": conclusion_repaired,
        "evidence_included": (
            evidence_index is None or source_lo <= evidence_index <= source_hi
        ),
        "context_expanded_seconds": round(expansion_added, 2),
        "trimmed_from_front": trimmed,
        "opening": opening,
        "body_segments": body_segments,
        "conclusion": conclusion,
        "segments": segments,
        "narration": narration,
        "warnings": warnings,
        "explanation": plan_explanation,
    }
