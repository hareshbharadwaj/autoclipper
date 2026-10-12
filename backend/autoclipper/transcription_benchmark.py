"""Repeatable Groq Whisper comparison against manually corrected references."""
from datetime import datetime, timezone
import math
import re
import unicodedata

from .transcribe import GroqTranscriptionError, _transcribe_file


DEFAULT_BENCHMARK_MODELS = (
    "whisper-large-v3-turbo",
    "whisper-large-v3",
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def normalize_tokens(text):
    """Normalize punctuation/case without erasing accents or word distinctions."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _TOKEN_RE.findall(normalized)


def _align_tokens(reference, hypothesis):
    """Levenshtein alignment with deterministic match/substitute/delete/insert ops."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        costs[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        costs[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                back[i][j] = "match"
                continue
            choices = [
                (costs[i - 1][j - 1] + 1, 0, "substitute"),
                (costs[i - 1][j] + 1, 1, "delete"),
                (costs[i][j - 1] + 1, 2, "insert"),
            ]
            costs[i][j], _, back[i][j] = min(choices)

    alignment = []
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation == "match" or operation == "substitute":
            alignment.append((operation, i - 1, j - 1))
            i -= 1
            j -= 1
        elif operation == "delete":
            alignment.append((operation, i - 1, None))
            i -= 1
        else:
            alignment.append(("insert", None, j - 1))
            j -= 1
    alignment.reverse()
    return alignment


def word_error_metrics(reference_text, hypothesis_text):
    """Return WER edit counts, omission/hallucination rates, and token counts."""
    reference = normalize_tokens(reference_text)
    hypothesis = normalize_tokens(hypothesis_text)
    alignment = _align_tokens(reference, hypothesis)
    substitutions = sum(operation == "substitute" for operation, _, _ in alignment)
    deletions = sum(operation == "delete" for operation, _, _ in alignment)
    insertions = sum(operation == "insert" for operation, _, _ in alignment)
    edits = substitutions + deletions + insertions
    return {
        "wer": edits / len(reference) if reference else None,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "omission_rate": deletions / len(reference) if reference else None,
        "hallucination_rate": insertions / len(hypothesis) if hypothesis else 0.0,
        "reference_word_count": len(reference),
        "hypothesis_word_count": len(hypothesis),
        "matched_word_count": sum(operation == "match" for operation, _, _ in alignment),
    }


def _timed_tokens(words):
    tokens = []
    for item in words or []:
        if isinstance(item, str):
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("word", item.get("text", ""))
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            continue
        for token in normalize_tokens(text):
            tokens.append({"token": token, "start": start, "end": end})
    return tokens


def timestamp_error_metrics(reference_words, hypothesis_words, tolerance_seconds=0.25):
    """Compare timestamps for exactly aligned words with manually timed references.

    Substitutions are excluded from the timing score because they do not establish
    that the same word was aligned. Word omissions remain visible in the WER fields.
    """
    reference = _timed_tokens(reference_words)
    hypothesis = _timed_tokens(hypothesis_words)
    alignment = _align_tokens(
        [item["token"] for item in reference],
        [item["token"] for item in hypothesis],
    )
    start_errors, end_errors = [], []
    for operation, ref_index, hyp_index in alignment:
        if operation != "match":
            continue
        ref_word, hyp_word = reference[ref_index], hypothesis[hyp_index]
        start_errors.append(abs(ref_word["start"] - hyp_word["start"]))
        end_errors.append(abs(ref_word["end"] - hyp_word["end"]))

    count = len(start_errors)
    if not count:
        return {
            "timestamp_word_count": 0,
            "start_mae_seconds": None,
            "end_mae_seconds": None,
            "boundary_mae_seconds": None,
            "within_tolerance_rate": None,
        }
    start_mae = sum(start_errors) / count
    end_mae = sum(end_errors) / count
    within = sum(
        start_error <= tolerance_seconds and end_error <= tolerance_seconds
        for start_error, end_error in zip(start_errors, end_errors)
    )
    return {
        "timestamp_word_count": count,
        "start_mae_seconds": start_mae,
        "end_mae_seconds": end_mae,
        "boundary_mae_seconds": (sum(start_errors) + sum(end_errors)) / (2 * count),
        "within_tolerance_rate": within / count,
    }


def score_transcription(reference_text, prediction, reference_words=None):
    """Score one provider result without treating another model as ground truth."""
    if not isinstance(prediction, dict) or not isinstance(prediction.get("text"), str):
        raise ValueError("Prediction must be an object containing a text string.")
    metrics = word_error_metrics(reference_text, prediction["text"])
    metrics.update(timestamp_error_metrics(reference_words, prediction.get("words", [])))
    return metrics


def _aggregate(records):
    successful = [record for record in records if record.get("status") == "ok"]
    if not successful:
        return {
            "sample_count": 0,
            "failed_sample_count": len(records),
            "wer": None,
            "timestamp_word_count": 0,
        }

    metrics = [record["metrics"] for record in successful]
    reference_count = sum(item["reference_word_count"] for item in metrics)
    hypothesis_count = sum(item["hypothesis_word_count"] for item in metrics)
    substitutions = sum(item["substitutions"] for item in metrics)
    deletions = sum(item["deletions"] for item in metrics)
    insertions = sum(item["insertions"] for item in metrics)
    timestamp_count = sum(item["timestamp_word_count"] for item in metrics)

    def weighted_mean(field):
        available = [item for item in metrics if item[field] is not None and item["timestamp_word_count"]]
        count = sum(item["timestamp_word_count"] for item in available)
        if not count:
            return None
        return sum(item[field] * item["timestamp_word_count"] for item in available) / count

    return {
        "sample_count": len(successful),
        "failed_sample_count": len(records) - len(successful),
        "reference_word_count": reference_count,
        "hypothesis_word_count": hypothesis_count,
        "wer": (substitutions + deletions + insertions) / reference_count if reference_count else None,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "omission_rate": deletions / reference_count if reference_count else None,
        "hallucination_rate": insertions / hypothesis_count if hypothesis_count else 0.0,
        "timestamp_word_count": timestamp_count,
        "start_mae_seconds": weighted_mean("start_mae_seconds"),
        "end_mae_seconds": weighted_mean("end_mae_seconds"),
        "boundary_mae_seconds": weighted_mean("boundary_mae_seconds"),
    }


def run_transcription_benchmark(samples, api_key, models=DEFAULT_BENCHMARK_MODELS, transcriber=None):
    """Run each model over each exact same audio sample and score manual references.

    The optional `transcriber` argument is intended for deterministic tests. In
    production, it is the existing Groq `_transcribe_file` request function.
    """
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in the environment before benchmarking.")
    models = tuple(model.strip() for model in models if isinstance(model, str) and model.strip())
    if not models:
        raise ValueError("At least one Whisper model must be selected.")
    if not samples:
        raise ValueError("At least one manually referenced audio sample is required.")
    transcribe_file = transcriber or _transcribe_file

    records = []
    for model in models:
        for sample in samples:
            sample_id = str(sample.get("id") or sample.get("audio") or "unnamed")
            tags = sorted({str(tag).strip() for tag in sample.get("tags", []) if str(tag).strip()})
            reference_text = sample.get("reference_text")
            if not isinstance(reference_text, str):
                reference_words = sample.get("reference_words") or []
                reference_text = " ".join(
                    str(item.get("word", item.get("text", "")))
                    for item in reference_words if isinstance(item, dict)
                )
            base = {
                "sample_id": sample_id,
                "audio": sample.get("audio"),
                "tags": tags,
                "model": model,
            }
            try:
                prediction = transcribe_file(sample["audio"], api_key, model)
                metrics = score_transcription(
                    reference_text,
                    prediction,
                    sample.get("reference_words"),
                )
                records.append({
                    **base,
                    "status": "ok",
                    "reference_text": reference_text,
                    "prediction_text": prediction["text"],
                    "prediction_words": prediction.get("words", []),
                    "metrics": metrics,
                })
            except GroqTranscriptionError as exc:
                records.append({
                    **base,
                    "status": "error",
                    "error_category": exc.category,
                    "status_code": exc.status_code,
                    "error": str(exc),
                })
            except Exception as exc:  # one sample/model failure should not stop the benchmark
                records.append({
                    **base,
                    "status": "error",
                    "error_category": "invalid_response" if isinstance(exc, ValueError) else "unexpected_error",
                    "error": str(exc)[:400],
                })

    by_model = {
        model: _aggregate([record for record in records if record["model"] == model])
        for model in models
    }
    tags = sorted({tag for record in records for tag in record["tags"]})
    by_tag = {
        tag: {
            model: _aggregate([
                record for record in records
                if record["model"] == model and tag in record["tags"]
            ])
            for model in models
        }
        for tag in tags
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": list(models),
        "sample_count": len(samples),
        "results": records,
        "summary": {"by_model": by_model, "by_tag": by_tag},
    }
