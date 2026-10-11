import json

import httpx
import pytest

import autoclipper.transcribe as transcribe_mod
from autoclipper.transcribe import GroqTranscriptionError
from autoclipper.transcription_benchmark import (
    DEFAULT_BENCHMARK_MODELS,
    run_transcription_benchmark,
    score_transcription,
    timestamp_error_metrics,
    word_error_metrics,
)
from scripts.benchmark_transcription import load_manifest


def _valid_payload():
    return {
        "text": "Hello there.",
        "language": "English",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "there.", "start": 0.41, "end": 0.8},
        ],
        "segments": [{"start": 0.0, "end": 0.8, "text": "Hello there."}],
    }


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, invalid_json=False):
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else "provider error"
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("not json")
        return self.payload


class FakeClient:
    def __init__(self, responses, error=None):
        self.responses = list(responses)
        self.error = error
        self.calls = []
        self.file_payloads = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        file_handle = kwargs["files"]["file"][1]
        self.file_payloads.append(file_handle.read())
        if self.error:
            raise self.error
        return self.responses.pop(0)


def _install_client(monkeypatch, client):
    monkeypatch.setattr(transcribe_mod.httpx, "Client", lambda **_kwargs: client)


def test_existing_model_setting_selects_turbo_by_default_and_large_v3_on_request(monkeypatch):
    monkeypatch.setattr(transcribe_mod.config, "GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    monkeypatch.setattr(transcribe_mod, "extract_audio", lambda *_args: "sample.mp3")
    monkeypatch.setattr(transcribe_mod, "split_audio", lambda *_args: [("sample.mp3", 0.0)])
    selected = []

    def fake_transcribe(_path, _key, model):
        selected.append(model)
        return _valid_payload()

    monkeypatch.setattr(transcribe_mod, "_transcribe_file", fake_transcribe)
    default_result = transcribe_mod.transcribe("source.mp4", "out", api_key="test-key")
    large_result = transcribe_mod.transcribe(
        "source.mp4", "out", api_key="test-key", model="whisper-large-v3"
    )

    assert selected == ["whisper-large-v3-turbo", "whisper-large-v3"]
    assert default_result["words"][0]["word"] == "Hello"
    assert large_result["segments"][0]["text"] == "Hello there."
    assert set(large_result) == {"language", "duration", "text", "words", "segments"}


def test_transcribe_file_retries_rate_limit_and_rewinds_audio(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"same audio bytes")
    client = FakeClient([
        FakeResponse(429, {"error": {"message": "busy"}}, {"Retry-After": "0.25"}),
        FakeResponse(200, _valid_payload()),
    ])
    sleeps = []
    _install_client(monkeypatch, client)
    monkeypatch.setattr(transcribe_mod.time, "sleep", sleeps.append)

    result = transcribe_mod._transcribe_file(str(audio), "test-secret", "whisper-large-v3", max_retries=1)

    assert len(client.calls) == 2
    assert client.file_payloads == [b"same audio bytes", b"same audio bytes"]
    assert sleeps == [0.25]
    assert client.calls[0][1]["data"]["model"] == "whisper-large-v3"
    assert client.calls[0][1]["data"]["timestamp_granularities[]"] == ["word", "segment"]
    assert result["words"][0]["start"] == 0.0


def test_transcribe_file_classifies_unsupported_model_without_retries(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    client = FakeClient([FakeResponse(400, {"error": {"message": "model is not supported; token test-secret was rejected"}})])
    _install_client(monkeypatch, client)

    with pytest.raises(GroqTranscriptionError) as error:
        transcribe_mod._transcribe_file(str(audio), "test-secret", "not-a-model")

    assert error.value.category == "unsupported_model"
    assert error.value.status_code == 400
    assert "test-secret" not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert len(client.calls) == 1


def test_transcribe_file_exhausted_rate_limit_is_classified(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    client = FakeClient([
        FakeResponse(429, {"error": "slow down"}),
        FakeResponse(429, {"error": "slow down"}),
    ])
    _install_client(monkeypatch, client)
    monkeypatch.setattr(transcribe_mod.time, "sleep", lambda _seconds: None)

    with pytest.raises(GroqTranscriptionError) as error:
        transcribe_mod._transcribe_file(str(audio), "test-key", "whisper-large-v3", max_retries=1)
    assert error.value.category == "rate_limited"
    assert error.value.status_code == 429
    assert len(client.calls) == 2


def test_transcribe_file_handles_network_failure_and_invalid_json(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    _install_client(monkeypatch, FakeClient([], httpx.ConnectError("offline")))
    monkeypatch.setattr(transcribe_mod.time, "sleep", lambda _seconds: None)
    with pytest.raises(GroqTranscriptionError) as network_error:
        transcribe_mod._transcribe_file(str(audio), "test-key", "whisper-large-v3", max_retries=1)
    assert network_error.value.category == "network_error"

    _install_client(monkeypatch, FakeClient([FakeResponse(200, invalid_json=True)]))
    with pytest.raises(GroqTranscriptionError) as response_error:
        transcribe_mod._transcribe_file(str(audio), "test-key", "whisper-large-v3")
    assert response_error.value.category == "invalid_response"


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [("words", "no word timestamps"), ("segments", "no segment timestamps")],
)
def test_transcribe_file_rejects_missing_timestamp_granularity(monkeypatch, tmp_path, missing_field, message):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    payload = _valid_payload()
    payload[missing_field] = []
    _install_client(monkeypatch, FakeClient([FakeResponse(200, payload)]))

    with pytest.raises(GroqTranscriptionError, match=message) as error:
        transcribe_mod._transcribe_file(str(audio), "test-key", "whisper-large-v3")
    assert error.value.category == "invalid_response"


def test_transcribe_file_rejects_malformed_timestamps(monkeypatch, tmp_path):
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"audio")
    payload = _valid_payload()
    payload["words"][0]["start"] = float("nan")
    _install_client(monkeypatch, FakeClient([FakeResponse(200, payload)]))

    with pytest.raises(GroqTranscriptionError, match="invalid start/end timestamps") as error:
        transcribe_mod._transcribe_file(str(audio), "test-key", "whisper-large-v3")
    assert error.value.category == "invalid_response"


def test_wer_reports_substitutions_deletions_and_hallucinations():
    metrics = word_error_metrics(
        "We will launch tomorrow",
        "We launch tomorrow plus",
    )
    assert metrics["substitutions"] == 0
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 1
    assert metrics["wer"] == pytest.approx(0.5)
    assert metrics["omission_rate"] == pytest.approx(0.25)
    assert metrics["hallucination_rate"] == pytest.approx(0.25)


def test_timestamp_metrics_score_exactly_aligned_words_only():
    reference = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.51, "end": 1.0},
    ]
    hypothesis = [
        {"word": "hello", "start": 0.1, "end": 0.6},
        {"word": "world", "start": 0.7, "end": 1.2},
        {"word": "extra", "start": 1.3, "end": 1.5},
    ]
    metrics = timestamp_error_metrics(reference, hypothesis)
    assert metrics["timestamp_word_count"] == 2
    assert metrics["start_mae_seconds"] == pytest.approx(0.145)
    assert metrics["end_mae_seconds"] == pytest.approx(0.15)
    assert metrics["boundary_mae_seconds"] == pytest.approx(0.1475)
    assert metrics["within_tolerance_rate"] == 1.0


def test_benchmark_runs_both_models_on_same_audio_and_groups_tags():
    calls = []
    sample = {
        "id": "names-and-overlap",
        "audio": "/samples/one.wav",
        "tags": ["proper-names", "overlap"],
        "reference_text": "Hello there.",
        "reference_words": _valid_payload()["words"],
    }

    def fake_transcriber(audio, key, model):
        calls.append((audio, key, model))
        return _valid_payload()

    report = run_transcription_benchmark(
        [sample], "test-key", transcriber=fake_transcriber
    )
    assert [call[2] for call in calls] == list(DEFAULT_BENCHMARK_MODELS)
    assert {call[0] for call in calls} == {sample["audio"]}
    assert report["summary"]["by_model"]["whisper-large-v3"]["wer"] == 0
    assert set(report["summary"]["by_tag"]) == {"proper-names", "overlap"}


def test_benchmark_records_provider_failures_and_continues():
    def partial_transcriber(_audio, _key, model):
        if model == "whisper-large-v3":
            raise GroqTranscriptionError("model unavailable", "unsupported_model", model, 404)
        return _valid_payload()

    report = run_transcription_benchmark(
        [{"audio": "sample.wav", "reference_text": "Hello there."}],
        "test-key",
        transcriber=partial_transcriber,
    )
    statuses = {record["model"]: record["status"] for record in report["results"]}
    categories = {
        record["model"]: record.get("error_category") for record in report["results"]
    }
    assert statuses == {"whisper-large-v3-turbo": "ok", "whisper-large-v3": "error"}
    assert categories["whisper-large-v3"] == "unsupported_model"
    assert report["summary"]["by_model"]["whisper-large-v3"]["failed_sample_count"] == 1


def test_load_manifest_resolves_audio_and_requires_valid_manual_reference(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"synthetic placeholder")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "samples": [{
            "id": "pause-case",
            "audio": "sample.wav",
            "tags": ["pause"],
            "reference_text": "Hello there.",
        }]
    }), encoding="utf-8")

    samples = load_manifest(manifest)
    assert samples[0]["audio"] == str(audio.resolve())
    assert samples[0]["tags"] == ["pause"]

    manifest.write_text(json.dumps({
        "samples": [{
            "id": "bad-time",
            "audio": "sample.wav",
            "reference_text": "Hello.",
            "reference_words": [{"word": "Hello", "start": 2, "end": 1}],
        }]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid timestamps"):
        load_manifest(manifest)


def test_score_transcription_requires_a_timestamped_text_result():
    with pytest.raises(ValueError, match="text string"):
        score_transcription("hello", {"segments": []})
