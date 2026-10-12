"""Transcription via Groq's OpenAI-compatible Whisper endpoint."""
import math
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx

from . import config
from .audio import extract_audio, split_audio
from .utils import logger


RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 60.0


class GroqTranscriptionError(RuntimeError):
    """A safe, classified Groq transcription failure (never includes request headers)."""

    def __init__(self, message, category, model=None, status_code=None):
        super().__init__(message)
        self.category = category
        self.model = model
        self.status_code = status_code


def _error_detail(response, api_key):
    """Extract a short provider detail and redact the credential defensively."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                detail = error.get("message") or error.get("detail") or ""
            elif isinstance(error, str):
                detail = error
    except (ValueError, TypeError):
        pass
    if not detail:
        detail = getattr(response, "text", "") or ""
    detail = str(detail).strip()
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    return detail[:400]


def _http_error_category(status_code, detail):
    lowered = (detail or "").lower()
    if status_code == 429:
        return "rate_limited"
    if status_code in (401, 403):
        return "authentication_error"
    if status_code == 404 or (
        status_code == 400 and any(token in lowered for token in ("model", "unsupported", "not found"))
    ):
        return "unsupported_model"
    if status_code >= 500:
        return "provider_error"
    return "request_error"


def _retry_delay(response, attempt):
    """Use Retry-After when present; otherwise apply bounded exponential backoff."""
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After")
    delay = None
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = None
    if delay is None or not math.isfinite(delay):
        delay = float(2 ** attempt)
    return max(0.0, min(delay, MAX_RETRY_AFTER_SECONDS))


def _invalid_response(message, model):
    return GroqTranscriptionError(
        f"Groq returned an invalid timestamped transcription for model '{model}': {message}",
        category="invalid_response",
        model=model,
    )


def _validate_timestamp_items(items, field, model):
    """Validate word/segment timestamp records while preserving provider fields."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise _invalid_response(f"'{field}' must be a list.", model)

    normalized = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise _invalid_response(f"'{field}[{index}]' must be an object.", model)
        text_key = "word" if field == "words" else "text"
        text = item.get(text_key)
        if not isinstance(text, str):
            raise _invalid_response(f"'{field}[{index}].{text_key}' must be text.", model)
        try:
            raw_start, raw_end = item["start"], item["end"]
            if isinstance(raw_start, bool) or isinstance(raw_end, bool):
                raise ValueError
            start, end = float(raw_start), float(raw_end)
        except (KeyError, TypeError, ValueError):
            raise _invalid_response(f"'{field}[{index}]' needs numeric start/end timestamps.", model)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise _invalid_response(f"'{field}[{index}]' has invalid start/end timestamps.", model)
        normalized.append({**item, "start": start, "end": end})
    return normalized


def _validate_transcription_payload(payload, model):
    """Check verbose_json before it enters the shared transcript structure."""
    if not isinstance(payload, dict):
        raise _invalid_response("the response must be a JSON object.", model)
    if not isinstance(payload.get("text"), str):
        raise _invalid_response("'text' must be a string.", model)

    words = _validate_timestamp_items(payload.get("words"), "words", model)
    segments = _validate_timestamp_items(payload.get("segments"), "segments", model)
    if payload["text"].strip() and not words:
        raise _invalid_response("the response contains text but no word timestamps.", model)
    if payload["text"].strip() and not segments:
        raise _invalid_response("the response contains text but no segment timestamps.", model)

    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise _invalid_response("'language' must be a string when provided.", model)
    return {**payload, "words": words, "segments": segments}


def _transcribe_file(audio_path: str, api_key: str, model: str, max_retries: int = 2) -> dict:
    """Transcribe one audio file, retry transient failures, and validate timestamps."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("A non-empty Groq Whisper model name is required.")
    if not api_key:
        raise ValueError("GROQ_API_KEY is required for transcription.")

    try:
        max_retries = max(0, int(max_retries))
    except (TypeError, ValueError):
        raise ValueError("max_retries must be a non-negative integer.")
    try:
        audio_handle = open(audio_path, "rb")
    except OSError as exc:
        raise GroqTranscriptionError(
            f"Could not open audio for Groq transcription ({type(exc).__name__}).",
            category="input_error",
            model=model,
        ) from exc
    with audio_handle as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
        data = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            with httpx.Client(timeout=300) as client:
                for attempt in range(max(0, int(max_retries)) + 1):
                    try:
                        f.seek(0)
                        response = client.post(
                            config.GROQ_TRANSCRIBE_URL,
                            headers=headers,
                            data=data,
                            files=files,
                        )
                    except httpx.RequestError as exc:
                        if attempt < max_retries:
                            time.sleep(float(2 ** attempt))
                            continue
                        raise GroqTranscriptionError(
                            f"Groq transcription request failed for model '{model}' after retries "
                            f"({type(exc).__name__}).",
                            category="network_error",
                            model=model,
                        ) from exc

                    if response.status_code in RETRYABLE_HTTP_STATUSES and attempt < max_retries:
                        delay = _retry_delay(response, attempt)
                        logger.warning(
                            "Groq transcription returned HTTP %d for %s; retrying in %.1fs (%d/%d).",
                            response.status_code, model, delay, attempt + 1, max_retries,
                        )
                        if delay:
                            time.sleep(delay)
                        continue
                    if response.status_code != 200:
                        detail = _error_detail(response, api_key)
                        category = _http_error_category(response.status_code, detail)
                        message = (
                            f"Groq transcription failed for model '{model}' "
                            f"(HTTP {response.status_code}, {category})"
                        )
                        if detail:
                            message += f": {detail}"
                        raise GroqTranscriptionError(
                            message,
                            category=category,
                            model=model,
                            status_code=response.status_code,
                        )

                    try:
                        payload = response.json()
                    except (ValueError, TypeError) as exc:
                        raise _invalid_response("the response body was not valid JSON.", model) from exc
                    return _validate_transcription_payload(payload, model)
        except GroqTranscriptionError:
            raise
        except (OSError, ValueError) as exc:
            raise GroqTranscriptionError(
                f"Could not prepare the audio file for Groq transcription ({type(exc).__name__}).",
                category="input_error",
                model=model,
            ) from exc

    raise GroqTranscriptionError(
        f"Groq transcription failed for model '{model}' without a provider response.",
        category="provider_error",
        model=model,
    )


def _offset_items(items, offset: float, keys=("start", "end")):
    for item in items or []:
        for k in keys:
            if k in item and item[k] is not None:
                item[k] = round(item[k] + offset, 3)
    return items or []


def transcribe(video_path: str, output_dir: str, api_key: str = None, model: str = None) -> dict:
    """Transcribe a video's audio, handling Groq's file-size limit via chunking.

    Returns a normalized dict:
        {
          "language": str,
          "duration": float,
          "text": str,
          "words":    [{"word", "start", "end"}, ...],
          "segments": [{"start", "end", "text"}, ...],
        }
    """
    api_key = config.require(api_key or config.GROQ_API_KEY, "GROQ_API_KEY")
    model = model or config.GROQ_WHISPER_MODEL

    logger.info("Transcribing with Groq model '%s'...", model)
    audio_path = extract_audio(video_path, output_dir)
    chunks = split_audio(audio_path, output_dir)

    all_words, all_segments = [], []
    full_text_parts = []
    language = None

    for idx, (chunk_path, offset) in enumerate(chunks):
        logger.info("Transcribing chunk %d/%d (offset %.1fs)", idx + 1, len(chunks), offset)
        result = _transcribe_file(chunk_path, api_key, model)
        language = language or result.get("language")
        full_text_parts.append((result.get("text") or "").strip())
        all_words.extend(_offset_items(result.get("words"), offset))
        all_segments.extend(
            _offset_items(
                [
                    {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "").strip()}
                    for s in result.get("segments", [])
                ],
                offset,
            )
        )

    transcript = {
        "language": language,
        "duration": all_segments[-1]["end"] if all_segments else 0.0,
        "text": " ".join(p for p in full_text_parts if p),
        "words": all_words,
        "segments": all_segments,
    }
    logger.info(
        "Transcription complete: %d words, %d segments", len(all_words), len(all_segments)
    )
    return transcript
