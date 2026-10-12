"""Source-grounding and factual validation for clip candidates.

Every check in this module is deterministic and transcript-based: it never
calls a model, never invents confidence scores, and never claims visual
verification. Its purpose is to answer, with evidence from the source
transcript, whether a candidate edit and its generated text (title, hook,
description, evidence quote) are actually supported by the source material.

Evidence hierarchy enforced throughout the pipeline:

1. The source media and its verified timestamps (strongest).
2. The actual transcript text with its timestamps.
3. Visual observations from frames that were genuinely inspected.
4. Model-generated summaries, topics, chapters, and suggested angles.
5. Model-generated inferences (weakest — never silently override 1-3).

Status vocabulary used here and by the pipeline:
``passed``, ``warning``, ``failed``, ``not_run``, ``unverified``.
"""
import re

# Timestamp convention: all intervals are [start, end) in seconds on the
# ORIGINAL source timeline. A boundary that lands within `BOUNDARY_TOLERANCE`
# of a transcript segment edge counts as sentence-aligned.
BOUNDARY_TOLERANCE = 0.45

# Lexical topic-relevance thresholds (documented heuristics, not calibrated
# probabilities). Below CRITICAL an AI-generated candidate is treated as
# unrelated to the requested topic; below SOFT it earns a visible warning.
RELEVANCE_CRITICAL = 0.2
RELEVANCE_SOFT = 0.3

# Cues that a cut may flip or weaken the speaker's meaning. Kept as cues, not
# proof: each hit is reported as a risk that a reviewer should see, and only a
# negation on the EXCLUDED side of a boundary is treated as critical.
_NEGATION_CUES = (
    "do not", "don't", "does not", "doesn't", "did not", "didn't",
    "never", "no,", "not ", "cannot", "can't", "avoid", "stop ",
)
_QUALIFIER_CUES = (
    "but ", "unless", "except", "only when", "only if", "however",
    "if you", "make sure", "as long as", "otherwise",
)
_SAFETY_CUES = (
    "warning", "danger", "careful", "safety", "safe", "injur",
    "never", "always", "important",
)

_COUNT_NOUNS = (
    "rules", "tips", "ways", "steps", "reasons", "mistakes", "things",
    "secrets", "lessons", "facts", "habits", "tricks", "ideas", "points",
)
_DIGIT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_LIST_MARKERS = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "number one", "number two", "number three",
    "number four", "number five", "rule one", "rule two", "rule three",
    "tip one", "tip two", "tip three", "step one", "step two", "step three",
    "1.", "2.", "3.", "4.", "5.", "#1", "#2", "#3",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "so", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "up", "was", "we", "what", "when", "which",
    "who", "will", "with", "you", "your", "about", "just", "like", "really",
    "very", "can", "could", "would", "should", "do", "does", "did", "not",
    "no", "yes", "all", "any", "more", "most", "some", "than", "too", "also",
    "its", "his", "her", "my", "me", "us", "he", "she", "were",
}

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_QUOTE_RE = re.compile(r"[\"\u201c]([^\u201d\"]{8,160})[\"\u201d]")


def normalize_text(text):
    """Casefold and collapse whitespace for verbatim-style comparisons."""
    return re.sub(r"\s+", " ", str(text or "")).casefold().strip()


def content_words(text):
    """Extract lowercased content words (stopwords removed)."""
    return [w for w in _WORD_RE.findall(str(text or "").casefold())
            if w not in _STOPWORDS and len(w) > 1]


def token_overlap(text_a, text_b):
    """Symmetric content-word overlap in [0, 1] between two texts."""
    words_a = set(content_words(text_a))
    words_b = set(content_words(text_b))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def segments_in_span(transcript, start, end, pad=0.0):
    """Return transcript segments overlapping [start - pad, end + pad)."""
    selected = []
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = (segment.get("text") or "").strip()
        if not text or seg_end <= seg_start:
            continue
        if seg_end > float(start) - pad and seg_start < float(end) + pad:
            selected.append({"start": seg_start, "end": seg_end, "text": text})
    selected.sort(key=lambda item: (item["start"], item["end"]))
    return selected


def span_text(transcript, start, end, pad=0.0):
    """Exact transcript text overlapping a source interval (verbatim)."""
    return " ".join(seg["text"] for seg in segments_in_span(transcript, start, end, pad))


def words_in_span(transcript, start, end):
    """Return transcript words overlapping [start, end), preserving order."""
    selected = []
    for word in transcript.get("words", []) or []:
        if not isinstance(word, dict):
            continue
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if word_end > float(start) and word_start < float(end):
            selected.append(word)
    return selected


def verify_quote(transcript, quote, start=None, end=None):
    """Check that a quote appears verbatim (normalized) in the transcript.

    Returns ``{"verified": bool, "in_span": bool, "location": (s, e) | None}``.
    ``in_span`` is only evaluated when start/end are provided.
    """
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        return {"verified": False, "in_span": False, "location": None}
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        if normalized_quote in normalize_text(segment.get("text")):
            try:
                seg_start = float(segment["start"])
                seg_end = float(segment["end"])
            except (KeyError, TypeError, ValueError):
                seg_start = seg_end = None
            in_span = True
            if start is not None and end is not None and seg_start is not None:
                in_span = seg_end > float(start) and seg_start < float(end)
            return {"verified": True, "in_span": in_span,
                    "location": (seg_start, seg_end)}
    return {"verified": False, "in_span": False, "location": None}


def find_quoted_text(text):
    """Return quoted substrings (double/curly quotes) found in generated text."""
    return [match.strip() for match in _QUOTE_RE.findall(str(text or ""))
            if match.strip()]


def boundary_alignment(transcript, start, end, tolerance=BOUNDARY_TOLERANCE):
    """Determine whether clip boundaries land on transcript sentence edges.

    Whisper segments are sentence-level, so a boundary that does not match a
    segment edge (within tolerance) indicates a probable mid-sentence cut.
    """
    segments = segments_in_span(transcript, start, end)
    result = {
        "start_aligned": False,
        "end_aligned": False,
        "start_gap": None,
        "end_gap": None,
    }
    if not segments:
        return result
    start_gaps = [abs(seg["start"] - float(start)) for seg in segments]
    end_gaps = [abs(seg["end"] - float(end)) for seg in segments]
    result["start_gap"] = round(min(start_gaps), 3)
    result["end_gap"] = round(min(end_gaps), 3)
    result["start_aligned"] = min(start_gaps) <= tolerance
    result["end_aligned"] = min(end_gaps) <= tolerance
    return result


def boundary_context_risks(transcript, start, end):
    """Detect cuts that may strip negations, conditions, or safety qualifiers.

    Returns a list of risk dicts: ``{"type", "severity", "detail"}``.
    Severity is ``critical`` only when excluded speech on the far side of a
    boundary carries a negation/qualifier belonging to a sentence the clip
    partially includes — the classic meaning-flip cut. Everything else is a
    ``warning`` a human or model reviewer should see.
    """
    risks = []
    segments = []
    for segment in transcript.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = (segment.get("text") or "").strip()
        if text:
            segments.append({"start": seg_start, "end": seg_end, "text": text})

    start, end = float(start), float(end)

    # A segment the clip only partially includes at its opening: the excluded
    # head may carry a negation that flips what remains ("do not walk ...").
    for seg in segments:
        # The clip starts strictly inside this segment (beyond the boundary
        # tolerance from both of the segment's edges).
        if seg["start"] + BOUNDARY_TOLERANCE < start < seg["end"] - BOUNDARY_TOLERANCE:
            head = seg["text"]
            head_words = normalize_text(head)
            if any(cue in head_words for cue in _NEGATION_CUES):
                risks.append({
                    "type": "negation_cut_at_start",
                    "severity": "critical",
                    "detail": (
                        f"The clip starts inside the sentence \"{head}\" "
                        f"({seg['start']:.1f}-{seg['end']:.1f}s); the excluded "
                        "opening contains a negation, so the retained speech may "
                        "say the opposite of the speaker's intent."
                    ),
                })
            elif any(cue in head_words for cue in _QUALIFIER_CUES):
                risks.append({
                    "type": "qualifier_cut_at_start",
                    "severity": "warning",
                    "detail": (
                        f"The clip starts inside the sentence \"{head}\"; the "
                        "excluded opening contains a condition/qualifier."
                    ),
                })

    # A segment the clip only partially includes at its ending.
    for seg in segments:
        # The clip ends strictly inside this segment (beyond the boundary
        # tolerance from both of the segment's edges).
        if seg["start"] + BOUNDARY_TOLERANCE < end < seg["end"] - BOUNDARY_TOLERANCE:
            tail = normalize_text(seg["text"])
            if any(cue in tail for cue in _NEGATION_CUES + _QUALIFIER_CUES):
                risks.append({
                    "type": "qualifier_cut_at_end",
                    "severity": "warning",
                    "detail": (
                        f"The clip ends inside the sentence \"{seg['text']}\" "
                        f"({seg['start']:.1f}-{seg['end']:.1f}s); the excluded "
                        "tail contains a condition, negation, or qualifier."
                    ),
                })
            elif any(cue in tail for cue in _SAFETY_CUES):
                risks.append({
                    "type": "safety_statement_cut_at_end",
                    "severity": "warning",
                    "detail": (
                        f"The clip ends inside a safety-related sentence "
                        f"\"{seg['text']}\" ({seg['start']:.1f}-{seg['end']:.1f}s); "
                        "verify the cut does not drop the warning's point."
                    ),
                })

    # Ending near a safety statement that follows immediately. A stop exactly
    # on a sentence boundary is a natural cut point and is not flagged; only
    # unaligned stops within ~2s of a safety statement earn the warning.
    end_aligned = any(
        abs(seg["end"] - end) <= BOUNDARY_TOLERANCE for seg in segments
    )
    if not end_aligned:
        for seg in segments:
            if 0 <= seg["start"] - end <= 2.0:
                if any(cue in normalize_text(seg["text"]) for cue in _SAFETY_CUES):
                    risks.append({
                        "type": "warning_cut_at_end",
                        "severity": "warning",
                        "detail": (
                            f"A safety-related statement begins {seg['start'] - end:.1f}s "
                            f"after the clip ends: \"{seg['text'][:120]}\". Verify the "
                            "ending does not drop a required warning."
                        ),
                    })
    return risks


def transcript_uncertainty(transcript, start, end, low_probability=0.5):
    """Report low-confidence words in a span when the provider exposes them.

    Groq Whisper does not currently return per-word probabilities; when a
    provider does, words below ``low_probability`` are listed so captions and
    hooks avoid presenting uncertain words as confident. Never invents scores.
    """
    uncertain = []
    has_signal = False
    for word in words_in_span(transcript, start, end):
        probability = word.get("probability")
        if probability is None:
            continue
        has_signal = True
        try:
            if float(probability) < float(low_probability):
                uncertain.append(str(word.get("word") or ""))
        except (TypeError, ValueError):
            continue
    return {"provider_exposes_confidence": has_signal, "uncertain_words": uncertain}


def extract_count_claim(text):
    """Detect an enumerated-count claim such as '3 rules' in generated text.

    Returns ``(count, noun)`` or ``None``. Only explicit counts paired with a
    countable noun are treated as claims the clip must deliver.
    """
    lowered = str(text or "").casefold()
    for match in re.finditer(r"(\d+)\s+(?:\w+\s+){0,2}(" + "|".join(_COUNT_NOUNS) + r")", lowered):
        try:
            return int(match.group(1)), match.group(2)
        except ValueError:
            continue
    for match in re.finditer(
        r"\b(" + "|".join(_DIGIT_WORDS) + r")\s+(?:\w+\s+){0,2}(" + "|".join(_COUNT_NOUNS) + r")",
        lowered,
    ):
        return _DIGIT_WORDS[match.group(1)], match.group(2)
    return None


def count_enumerated_items(text):
    """Count distinct list markers (first/second/number one/1.) in text."""
    lowered = normalize_text(text)
    found = set()
    for marker in _LIST_MARKERS:
        if marker in lowered:
            found.add(marker.rstrip(".#"))
    # Collapse "number N" and "rule N" style duplicates of bare ordinals.
    return len(found)


def ground_text_claim(claim, evidence_text):
    """Grade how well generated text is supported by source transcript text.

    Returns ``{"status": supported|weak|unsupported, "matched_ratio": float}``.
    This is a lexical grounding signal, not proof of semantic agreement; the
    pipeline treats ``unsupported`` as a blocking condition only for generated
    (non-user-selected) candidates, and always reports the ratio.
    """
    claim_words = list(dict.fromkeys(content_words(claim)))
    if not claim_words:
        return {"status": "unsupported", "matched_ratio": 0.0}
    evidence = set(content_words(evidence_text))
    matched = [w for w in claim_words if w in evidence]
    ratio = len(matched) / len(claim_words)
    status = "supported" if ratio >= 0.5 else ("weak" if ratio >= 0.25 else "unsupported")
    return {"status": status, "matched_ratio": round(ratio, 3)}


def check_generated_text(text, transcript, start=None, end=None):
    """Factual checks for one AI-written string (title/hook/description).

    - Quoted passages must exist verbatim in the transcript (fabrication check).
    - Enumerated-count claims ('3 rules') must be deliverable from the span.
    - Overall lexical grounding against the span (or the whole transcript when
      no span is given).
    Never rewrites the text; it only reports. Rewriting/dropping is the
    pipeline's decision.
    """
    text = str(text or "")
    result = {
        "status": "passed",
        "fabricated_quotes": [],
        "count_claim": None,
        "grounding": None,
        "notes": [],
    }
    if not text.strip():
        result["status"] = "not_run"
        return result

    for quote in find_quoted_text(text):
        check = verify_quote(transcript, quote, start, end)
        if not check["verified"]:
            result["fabricated_quotes"].append(quote)
            result["notes"].append(
                f"Generated text quotes \"{quote}\" but this exact wording does "
                "not appear anywhere in the transcript."
            )

    claim = extract_count_claim(text)
    if claim is not None and start is not None and end is not None:
        count, noun = claim
        span = span_text(transcript, start, end)
        items = count_enumerated_items(span)
        digit_in_span = str(count) in span.casefold() or any(
            word == _number_word(count) for word in content_words(span)
        )
        delivered = max(items, 1 if digit_in_span else 0)
        result["count_claim"] = {
            "claimed": count, "noun": noun, "markers_found": items,
            "supported": delivered >= count,
        }
        if delivered < count:
            result["notes"].append(
                f"The text promises {count} {noun}, but the selected span only "
                f"shows evidence for about {delivered}."
            )

    evidence = span_text(transcript, start, end) if start is not None and end is not None \
        else transcript.get("text") or span_text(transcript, 0, float("inf"))
    result["grounding"] = ground_text_claim(text, evidence)

    if result["fabricated_quotes"]:
        result["status"] = "failed"
    elif result["count_claim"] and not result["count_claim"]["supported"]:
        result["status"] = "failed"
    elif result["grounding"]["status"] == "unsupported":
        result["status"] = "warning"
    return result


def _number_word(count):
    for word, value in _DIGIT_WORDS.items():
        if value == count:
            return word
    return ""


def ground_candidate(candidate, transcript, *, user_direction=None,
                     review_state="not_run", frames_attached=False):
    """Build the factual-grounding QA record for one clip candidate.

    Deterministic only. The returned record separates:
      - evidence verification (quote present in transcript?),
      - generated-text grounding (title/hook/description),
      - boundary semantics (alignment + negation/qualifier risks),
      - transcript uncertainty inside the span,
      - topic relevance against explicit user direction,
      - visual verification status (never claimed without real frames).
    """
    try:
        start = float(candidate["start"])
        end = float(candidate["end"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "failed",
            "critical": ["Candidate has no valid source timestamps."],
            "warnings": [], "checks": {},
        }

    critical = []
    warnings = []
    span = span_text(transcript, start, end)
    user_selected = bool(candidate.get("_user_selected"))

    # 1. Evidence quote verification (model claims must be traceable).
    evidence_quote = (candidate.get("evidence_quote") or "").strip()
    evidence_check = None
    if evidence_quote:
        evidence_check = verify_quote(transcript, evidence_quote, start, end)
        if not evidence_check["verified"]:
            if user_selected:
                warnings.append(
                    "The candidate's evidence quote was not found verbatim in the "
                    "transcript; treating the quote as unverified."
                )
            else:
                critical.append(
                    "The candidate's evidence quote does not appear in the source "
                    "transcript; the claim of support is unfounded."
                )
        elif not evidence_check["in_span"]:
            warnings.append(
                "The evidence quote exists in the transcript but outside this "
                "candidate's interval."
            )
    elif review_state not in ("reviewed",):
        warnings.append(
            "No verbatim evidence quote is attached to this candidate; its "
            "support is unverified."
        )

    # 2. Generated text grounding (never display invented claims as fact).
    text_checks = {}
    for field in ("title", "hook", "description"):
        value = (candidate.get(field) or "").strip()
        if not value:
            continue
        check = check_generated_text(value, transcript, start, end)
        text_checks[field] = check
        if check["status"] == "failed":
            message = f"Generated {field} is not supported by the clip's transcript span."
            if user_selected:
                warnings.append(message)
            else:
                critical.append(message)
        elif check["status"] == "warning":
            warnings.append(
                f"Generated {field} has weak lexical support from the clip's "
                "transcript span; verify before publishing."
            )

    # 3. Boundary semantics.
    alignment = boundary_alignment(transcript, start, end)
    if not alignment["start_aligned"]:
        warnings.append(
            "The opening boundary does not align to a transcript sentence start; "
            "the clip may begin mid-thought."
        )
    if not alignment["end_aligned"]:
        warnings.append(
            "The ending boundary does not align to a transcript sentence end; "
            "the clip may cut off a sentence."
        )
    risks = boundary_context_risks(transcript, start, end)
    for risk in risks:
        if risk["severity"] == "critical":
            if user_selected:
                warnings.append(risk["detail"])
            else:
                critical.append(risk["detail"])
        else:
            warnings.append(risk["detail"])

    # 4. Topic relevance against explicit user direction. These are lexical
    # heuristics (documented thresholds, not calibrated probabilities).
    relevance = None
    if user_direction and str(user_direction).strip():
        relevance = round(token_overlap(user_direction, span), 3)
        if relevance < RELEVANCE_CRITICAL and not user_selected:
            critical.append(
                "The candidate's transcript span shares almost no content with "
                "the user's requested topic; it appears unrelated."
            )
        elif relevance < RELEVANCE_SOFT:
            warnings.append(
                "The selected span shares little vocabulary with the stated "
                "direction; double-check it covers the requested topic."
            )

    # 5. Transcript uncertainty within the span.
    uncertainty = transcript_uncertainty(transcript, start, end)
    if uncertainty["uncertain_words"]:
        warnings.append(
            "Some words in this span have low transcription confidence: "
            + ", ".join(uncertainty["uncertain_words"][:8])
        )

    # 6. Visual verification honesty: transcript-only evidence must never be
    #    reported as visual verification.
    if frames_attached:
        visual_status = "frame_sampled_model_review" if review_state == "reviewed" \
            else "frame_sampled_no_review"
    else:
        visual_status = "transcript_only"

    checks = {
        "evidence_quote": evidence_check,
        "generated_text": text_checks,
        "boundary_alignment": alignment,
        "boundary_risks": risks,
        "topic_relevance": relevance,
        "transcript_uncertainty": uncertainty,
        "visual_verification": visual_status,
        "review_state": review_state,
    }
    status = "failed" if critical else ("warning" if warnings else "passed")
    return {
        "status": status,
        "critical": critical,
        "warnings": warnings,
        "checks": checks,
        "transcript_span": span[:1200],
    }
