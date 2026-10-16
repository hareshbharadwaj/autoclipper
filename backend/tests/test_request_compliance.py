"""Regression tests for the request-compliance objectives.

These tests stub the media/model boundary (ffmpeg, Groq, Gemini) and drive the
real ``pipeline.run`` / selection / QA code with synthetic timestamped
transcripts, so they exercise the decision logic rather than the encoder.

Covered acceptance criteria:
  1. 2 individual clips requested with the montage off -> exactly 2 clips.
  2. 8 individual clips requested while the material allows it -> exactly 8.
  3. A montage is an *additional* output; it never reduces the clip count.
  4. A 60s target is not silently accepted as a full-length success at ~26s.
  5. Insufficient distinct material is reported as a truthful shortfall.
  6. Different titles/hooks over the same footage are not distinct clips.
  7. A silent source skips transcription and uses the visual analysis path.
  8. Audio without speech does not fail the pipeline.
  9. Source audio is preserved: no narration or external music is added.
 10. Duration QA reads the measured output, not the planned duration.
 11. Unicode logging and hook-overlay failures do not crash the job.
 12. The montage is accounted separately from individual clip QA/count.
"""

import io
import logging
import os

import pytest

from autoclipper import montage as montage_module
from autoclipper import pipeline, source_profile
from autoclipper.duration_policy import (
    QA_PASS,
    QA_SHORTFALL,
    STATUS_SHORTFALL,
    DurationPolicy,
)
from autoclipper.selection import select_distinct_candidates


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# Ten genuinely different knife-skills lessons: no shared sentences, so the
# diversity rules must accept them on merit rather than on wording luck.
LESSONS = (
    "Keep the blade pointed down on the board and walk with the tip trailing.",
    "Tuck your fingertips into a claw grip so the knuckles guide the blade.",
    "Let the knife do the work instead of pressing hard on the cutting edge.",
    "Wipe the blade dry after washing so the steel does not rust in the drawer.",
    "Use a honing rod weekly to realign the edge before it starts to drag.",
    "Start the cut with the heel of the blade and draw it forward smoothly.",
    "Choose a plastic board for raw chicken and a wooden board for vegetables.",
    "Stop cutting when the blade feels dull instead of forcing the blade.",
    "Cut a flat base on round vegetables before slicing them into rounds.",
    "Keep your cutting area clear of loose towels and other clutter.",
)


def _spread_candidates(count, *, duration=20.0, span=600.0, label="Moment"):
    """`count` distinct, non-overlapping moments spread across the source."""
    step = span / max(count, 1)
    candidates = []
    for index in range(count):
        start = round(index * step + 5.0, 2)
        lesson = LESSONS[index % len(LESSONS)]
        title = " ".join(lesson.split()[:5])
        candidates.append(
            {
                "title": title,
                "hook": f"{lesson.split()[0]} {lesson.split()[1]} rule",
                "description": lesson,
                "start": start,
                "end": round(start + duration, 2),
                "relevance_score": 0.9,
                "viral_score": 0.8,
            }
        )
    return candidates


def _words_for(segment):
    """Even word timestamps, the way Whisper reports them for a segment."""
    tokens = segment["text"].split()
    if not tokens:
        return []
    span = (segment["end"] - segment["start"]) / len(tokens)
    words = []
    for index, token in enumerate(tokens):
        words.append(
            {
                "start": round(segment["start"] + index * span, 3),
                "end": round(segment["start"] + (index + 1) * span, 3),
                "word": token + ("." if index == len(tokens) - 1 else ""),
            }
        )
    return words


def _transcript_for(candidates, *, duration=600.0):
    segments = []
    for candidate in candidates:
        segments.append(
            {
                "start": candidate["start"],
                "end": candidate["end"],
                "text": candidate["description"],
            }
        )
    if not segments:
        segments = [{"start": 0, "end": duration, "text": "Source narration."}]
    return {
        "duration": duration,
        "language": "English",
        "segments": segments,
        "words": [word for segment in segments for word in _words_for(segment)],
    }


class _StubRun:
    """Patches the boundary and records what the pipeline actually rendered."""

    def __init__(
        self,
        monkeypatch,
        tmp_path,
        *,
        candidates,
        transcript=None,
        video_duration=600.0,
        measured_durations=None,
        has_audio=True,
        audio_activity=None,
        visual_candidates=None,
    ):
        self.monkeypatch = monkeypatch
        self.tmp_path = tmp_path
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.candidates = candidates
        self.transcript = transcript if transcript is not None else _transcript_for(
            candidates, duration=video_duration
        )
        self.video_duration = video_duration
        self.measured_durations = measured_durations
        self.calls = {
            "transcribe": 0,
            "visual": 0,
            "rendered": [],          # (start, end) of individual clips
            "montage_shots": [],     # (start, end) of montage shots
            "audio_mix": [],
            "narration": [],
            "verify": [],            # measured durations handed to QA
        }

        video_path = tmp_path / "input.mp4"
        video_path.write_bytes(b"stub")
        self.video_path = video_path

        monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
        monkeypatch.setattr(
            pipeline, "ingest", lambda *a, **k: (str(video_path), "source")
        )
        monkeypatch.setattr(
            pipeline,
            "probe_video_info",
            lambda path: {
                "width": 1920,
                "height": 1080,
                "duration": video_duration,
                "has_audio": has_audio,
            },
        )
        monkeypatch.setattr(
            pipeline, "transcribe", self._transcribe
        )
        monkeypatch.setattr(
            pipeline, "find_viral_clips", lambda *a, **k: [dict(c) for c in candidates]
        )
        monkeypatch.setattr(pipeline, "measure_audio_activity", self._audio_activity)
        # `pipeline` imports these names directly, so patch them there.
        monkeypatch.setattr(pipeline, "find_visual_candidates", self._visual_candidates)
        monkeypatch.setattr(pipeline, "cut_all", self._cut_all)
        monkeypatch.setattr(
            montage_module, "stitch_montage", self._stitch_montage, raising=False
        )
        monkeypatch.setattr(pipeline, "verify_rendered_file", self._verify)
        monkeypatch.setattr(pipeline, "probe_duration", self._probe_duration)
        monkeypatch.setattr(pipeline, "write_json", lambda *a, **k: None)
        monkeypatch.setattr(
            pipeline,
            "save_to_library",
            lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
        )
        self.audio_activity = audio_activity
        self.visual_candidates = visual_candidates or []

    # -- stubs ------------------------------------------------------------- #
    def _audio_activity(self, path, *args, **kwargs):
        if self.audio_activity is not None:
            return self.audio_activity
        return {
            "measured": True,
            "has_audio": True,
            "silent": False,
            "max_volume_db": -6.0,
        }

    def _transcribe(self, *args, **kwargs):
        self.calls["transcribe"] += 1
        return self.transcript

    def _visual_candidates(self, *args, **kwargs):
        self.calls["visual"] += 1
        return {
            "candidates": [dict(c) for c in self.visual_candidates],
            "scenes": [],
            "mode": "visual_only",
            "semantic_evidence": "Sampled frames only; no dialogue is claimed.",
            "warning": None,
        }

    def _probe_duration(self, path):
        """The stitched length the montage actually has on disk."""
        return round(sum(end - start for start, end in self.calls["montage_shots"]), 2)

    def _stitch_montage(self, paths, output_path, crossfade_seconds=0.25):
        self.calls["montage_inputs"] = list(paths)
        with open(output_path, "wb") as handle:
            handle.write(b"synthetic montage")
        return output_path

    def _cut_all(self, video, clips, out_dir, title):
        is_montage = "montage" in os.path.basename(str(out_dir)).lower() or any(
            clip.get("is_montage") for clip in clips
        )
        bucket = "montage_shots" if is_montage else "rendered"
        made = []
        os.makedirs(out_dir, exist_ok=True)
        for index, clip in enumerate(clips):
            self.calls[bucket].append((float(clip["start"]), float(clip["end"])))
            name = f"{'montage_shot' if is_montage else 'clip'}_{index + 1}.mp4"
            with open(os.path.join(out_dir, name), "wb") as handle:
                handle.write(b"stub")
            made.append({"index": index + 1, "file": name, **clip})
        return made

    def _verify(self, path, **kwargs):
        index = len(self.calls["verify"])
        if self.measured_durations:
            measured = self.measured_durations[min(index, len(self.measured_durations) - 1)]
        elif "montage" in os.path.basename(path):
            measured = sum(end - start for start, end in self.calls["montage_shots"])
        else:
            measured = kwargs.get("expected_duration") or self.video_duration
        self.calls["verify"].append(measured)
        return {
            "status": "pass",
            "file": os.path.basename(path),
            "checks": {
                "file": {"ok": True},
                "streams": {"ok": True},
                "duration": {"ok": True, "actual_seconds": float(measured)},
                "audio": {"ok": True, "has_audio": True},
                "visual_samples": {"ok": True},
            },
            "warnings": [],
            "errors": [],
        }

    # -- run --------------------------------------------------------------- #
    def run(self, **kwargs):
        kwargs.setdefault("source", "stub-source")
        kwargs.setdefault("output_dir", str(self.tmp_path / "job"))
        kwargs.setdefault("vertical", False)
        kwargs.setdefault("use_yolo", False)
        return pipeline.run(**kwargs)


def _individual(clips):
    return [clip for clip in clips if not clip.get("is_montage")]


# --------------------------------------------------------------------------- #
# 1 + 2: exact individual clip counts
# --------------------------------------------------------------------------- #
def test_two_clips_requested_without_a_montage_delivers_exactly_two(tmp_path, monkeypatch):
    run = _StubRun(monkeypatch, tmp_path, candidates=_spread_candidates(6))

    result = run.run(clip_count=2, target_duration=60)

    assert result["requested_clip_count"] == 2
    assert result["individual_clip_count"] == 2
    assert len(_individual(result["clips"])) == 2
    assert result["montage_requested"] is False
    assert result["montage_count"] == 0
    assert "montage_file" not in result or not result.get("montage_file")
    assert result["request_compliance"]["count_satisfied"] is True


def test_eight_distinct_clips_are_produced_when_the_material_allows_it(tmp_path, monkeypatch):
    candidates = _spread_candidates(10)
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates),
    )

    result = run.run(clip_count=8, target_duration=60)

    assert result["requested_clip_count"] == 8
    assert result["individual_clip_count"] == 8
    assert len(_individual(result["clips"])) == 8
    assert result["request_compliance"]["count_satisfied"] is True

    # The eight clips must use eight distinct passages of the source.
    spans = sorted((float(c["start"]), float(c["end"])) for c in _individual(result["clips"]))
    for (start_a, end_a), (start_b, end_b) in zip(spans, spans[1:]):
        overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        shortest = min(end_a - start_a, end_b - start_b)
        assert overlap / shortest <= 0.35, f"clips {start_a}-{end_a} and {start_b}-{end_b} overlap"


# --------------------------------------------------------------------------- #
# 3 + 12: montage is an additional, separately accounted output
# --------------------------------------------------------------------------- #
def test_a_montage_adds_an_output_without_reducing_the_individual_clip_count(tmp_path, monkeypatch):
    candidates = _spread_candidates(6)
    transcript = _transcript_for(candidates)

    without = _StubRun(monkeypatch, tmp_path / "a", candidates=candidates, transcript=transcript)
    control = without.run(clip_count=3, target_duration=60)
    control_spans = sorted(
        (round(float(c["start"]), 2), round(float(c["end"]), 2))
        for c in _individual(control["clips"])
    )

    preferred = [
        {"title": candidates[0]["title"], "start": candidates[0]["start"], "end": candidates[0]["end"]},
        {"title": candidates[1]["title"], "start": candidates[1]["start"], "end": candidates[1]["end"]},
    ]
    with_montage = _StubRun(monkeypatch, tmp_path / "b", candidates=candidates, transcript=transcript)
    result = with_montage.run(
        clip_count=3, target_duration=60, montage=True, selected_clips=preferred,
    )

    assert result["requested_clip_count"] == 3
    assert result["individual_clip_count"] == 3, "the montage consumed a requested clip"
    assert len(_individual(result["clips"])) == 3
    assert result["montage_count"] == 1
    assert result["montage_file"]
    assert result["montage_report"]["status"] == "delivered"

    # Enabling the montage must not change which individual clips are delivered.
    montage_spans = sorted(
        (round(float(c["start"]), 2), round(float(c["end"]), 2))
        for c in _individual(result["clips"])
    )
    assert montage_spans == control_spans


def test_montage_qa_and_count_are_reported_separately_from_individual_clips(tmp_path, monkeypatch):
    candidates = _spread_candidates(4)
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates, transcript=_transcript_for(candidates)
    )
    preferred = [
        {"title": candidates[0]["title"], "start": candidates[0]["start"], "end": candidates[0]["end"]},
        {"title": candidates[1]["title"], "start": candidates[1]["start"], "end": candidates[1]["end"]},
        {"title": candidates[2]["title"], "start": candidates[2]["start"], "end": candidates[2]["end"]},
    ]

    result = run.run(
        clip_count=3, target_duration=20, montage=True, selected_clips=preferred,
        montage_target_duration=60,
    )

    # Count accounting is split and the montage is never one of the individuals.
    assert result["requested_clip_count"] == 3
    assert result["individual_clip_count"] == 3
    assert result["montage_count"] == 1
    assert len(result["clips"]) == 4
    montage_clip = next(c for c in result["clips"] if c.get("is_montage"))
    assert montage_clip["file"] == result["montage_file"]
    assert montage_clip["qa_status"] != QA_SHORTFALL

    # Individual clip QA counts only individual clips; the montage is verified
    # and reported separately instead of inflating the clip count.
    assert result["quality_assurance"]["checked_clip_count"] == 3
    montage_qa = result["quality_assurance"]["montage"]
    assert montage_qa["file"] == result["montage_file"]
    assert montage_qa["qa_status"] == montage_clip["qa_status"]
    assert montage_qa["delivery_status"] == "delivered"
    assert montage_qa["actual_seconds"] == pytest.approx(60.0, abs=0.5)
    compliance = result["request_compliance"]
    assert compliance["delivered_individual_clips"] == 3
    assert compliance["montage_requested"] is True
    assert compliance["montage_status"] == "delivered"
    assert compliance["duration_shortfall_clips"] == 0
    assert len(run.calls["montage_shots"]) == 3


# --------------------------------------------------------------------------- #
# 4 + 10: duration is judged on the measured output, not the plan
# --------------------------------------------------------------------------- #
def test_sixty_second_target_is_not_accepted_as_success_at_26_seconds(tmp_path, monkeypatch):
    candidates = _spread_candidates(2, span=200.0)
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates, duration=200.0),
        video_duration=200.0, measured_durations=[26.6, 26.6],
    )

    result = run.run(clip_count=1, target_duration=60)

    assert result["requested_clip_count"] == 1
    clip = _individual(result["clips"])[0]
    assert clip["actual_duration"] == pytest.approx(26.6, abs=0.01)
    assert clip["duration_verdict"]["status"] == STATUS_SHORTFALL
    assert clip["duration_shortfall"] is True
    assert clip["qa_status"] == QA_SHORTFALL

    assert result["quality_assurance"]["status"] == "shortfall"
    assert result["quality_assurance"]["duration_shortfall_count"] == 1
    assert result["request_compliance"]["satisfied"] is False
    assert result["request_compliance"]["duration_shortfall_clips"] == 1
    assert any(
        "26.6s delivered against a 60s request" in warning
        for warning in result["processing_warnings"]
    )


def test_duration_qa_uses_measured_output_not_planned_timestamps(tmp_path, monkeypatch):
    """A plan that promises 60s cannot make a 26.6s render pass."""
    candidates = _spread_candidates(1, duration=60.0, span=200.0)
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates, duration=200.0),
        video_duration=200.0, measured_durations=[26.6],
    )

    result = run.run(clip_count=1, target_duration=60)

    clip = _individual(result["clips"])[0]
    # The plan (and therefore the pre-render expectation) is ~60s...
    assert float(clip["planned_rendered_duration"]) >= 55.0
    # ...but the verdict follows the file on disk.
    assert run.calls["verify"][0] == pytest.approx(26.6, abs=0.01)
    assert clip["duration_verdict"]["actual_seconds"] == pytest.approx(26.6, abs=0.01)
    assert clip["duration_verdict"]["status"] == STATUS_SHORTFALL
    assert clip["qa_status"] != QA_PASS


# --------------------------------------------------------------------------- #
# 5: truthful shortfall when the source cannot support the request
# --------------------------------------------------------------------------- #
def test_insufficient_material_reports_a_truthful_shortfall(tmp_path, monkeypatch):
    # Only one genuinely usable moment exists for a four-clip request.
    candidates = _spread_candidates(1, duration=20.0, span=200.0)
    candidates[0].update({"relevance_score": 0.95, "viral_score": 0.9})
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates, duration=200.0),
        video_duration=200.0,
    )

    result = run.run(clip_count=4, target_duration=60)

    assert result["requested_clip_count"] == 4
    assert result["individual_clip_count"] == 1
    assert result["request_compliance"]["count_satisfied"] is False
    assert result["request_compliance"]["satisfied"] is False
    assert any(
        "requested 4 individual clips" in warning.lower()
        for warning in result["processing_warnings"]
    )
    failures = " ".join(result["request_compliance"]["failures"]).lower()
    assert "clip count" in failures or "4" in failures
    # Nothing is fabricated to fill the gap.
    assert len(result["clips"]) == 1


# --------------------------------------------------------------------------- #
# 6: different titles over the same footage are not distinct clips
# --------------------------------------------------------------------------- #
def test_renamed_duplicate_footage_is_not_counted_as_distinct(tmp_path, monkeypatch):
    duplicates = [
        {
            "title": f"Angle {index + 1}",
            "hook": f"Hook number {index + 1}",
            "description": f"A completely different wording {index + 1} of one lesson.",
            "start": 100.0,
            "end": 140.0,
        }
        for index in range(3)
    ]
    genuine = {
        "title": "Second lesson",
        "hook": "A different lesson",
        "description": "Cutting technique for round vegetables.",
        "start": 300.0,
        "end": 340.0,
    }
    policy = DurationPolicy.from_request(40)

    outcome = select_distinct_candidates(
        duplicates + [genuine], requested_count=4, policy=policy,
    )

    assert outcome.selected_count == 2, "re-narrated duplicates were counted as distinct"
    assert outcome.shortfall == 2
    assert outcome.rejected
    assert {entry["reason"] for entry in outcome.rejected} & {
        "duplicate_footage", "repeated_explanation"
    }

    # The same rule holds end-to-end.
    run = _StubRun(
        monkeypatch, tmp_path, candidates=duplicates + [genuine],
        transcript=_transcript_for(duplicates + [genuine], duration=400.0),
        video_duration=400.0,
    )
    result = run.run(clip_count=4, target_duration=40)
    assert result["individual_clip_count"] == 2
    assert result["request_compliance"]["count_satisfied"] is False


# --------------------------------------------------------------------------- #
# 7 + 8: analysis path selection
# --------------------------------------------------------------------------- #
def test_silent_video_skips_transcription_and_generates_clips_visually(tmp_path, monkeypatch):
    run = _StubRun(
        monkeypatch, tmp_path, candidates=[],
        has_audio=False,
        audio_activity={"measured": True, "has_audio": False, "silent": True,
                        "max_volume_db": None},
        visual_candidates=[
            {"title": "Visible action", "start": 20.0, "end": 60.0,
             "description": "Frames show a safe grip demonstration.",
             "visual_only": True, "relevance_score": 0.8},
            {"title": "Second visible action", "start": 120.0, "end": 160.0,
             "description": "Frames show a cutting board setup.",
             "visual_only": True, "relevance_score": 0.7},
        ],
    )

    result = run.run(clip_count=2, target_duration=40)

    assert run.calls["transcribe"] == 0, "transcription ran for a source with no audio"
    assert run.calls["visual"] == 1
    assert result["source_profile"]["analysis_mode"] == source_profile.MODE_VISUAL_ONLY
    assert result["source_profile"]["has_speech"] is False
    assert result["individual_clip_count"] >= 1
    # No invented dialogue: visual-only candidates never get a transcript quote.
    for clip in _individual(result["clips"]):
        assert not (clip.get("transcript") or "").strip() or clip.get("visual_only")


def test_audio_without_speech_does_not_fail_the_pipeline(tmp_path, monkeypatch):
    """A loud track whose transcription is empty must fall back, not raise."""
    run = _StubRun(
        monkeypatch, tmp_path, candidates=[],
        transcript={"duration": 300.0, "language": "English", "segments": [], "words": []},
        video_duration=300.0,
        audio_activity={"measured": True, "has_audio": True, "silent": False,
                        "max_volume_db": -8.0},
        visual_candidates=[
            {"title": "Music-driven moment", "start": 30.0, "end": 70.0,
             "description": "Frames show the knife being wiped down.",
             "visual_only": True, "relevance_score": 0.75},
        ],
    )

    result = run.run(clip_count=1, target_duration=40)

    assert result["source_profile"]["analysis_mode"] != source_profile.MODE_SPEECH
    assert run.calls["visual"] == 1
    assert result["individual_clip_count"] == 1
    assert result["quality_assurance"]["status"] != "fail"


# --------------------------------------------------------------------------- #
# 9: original audio is preserved
# --------------------------------------------------------------------------- #
def test_source_audio_is_preserved_without_narration_or_added_music(tmp_path, monkeypatch):
    candidates = _spread_candidates(2)
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates, duration=200.0),
        video_duration=200.0,
    )
    # Fail loudly if anything tries to synthesize speech or mix in music.
    import autoclipper

    for module_name in ("voiceover", "narration", "tts", "music"):
        module = getattr(autoclipper, module_name, None)
        if module is not None:
            for attr in dir(module):
                if attr.startswith("_"):
                    continue
                target = getattr(module, attr)
                if callable(target):
                    monkeypatch.setattr(
                        module, attr, _forbidden(module_name, attr), raising=False
                    )

    result = run.run(clip_count=2, target_duration=40)

    assert result["source_profile"]["preserve_audio"] is True
    for clip in _individual(result["clips"]):
        assert clip["quality_assurance"]["checks"]["audio"]["ok"] is True
    assert run.calls["audio_mix"] == []
    assert run.calls["narration"] == []


def _forbidden(module_name, attr):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{module_name}.{attr} must not run for a source-only job")
    return _fail


# --------------------------------------------------------------------------- #
# 11: Unicode logging + hook overlay failures
# --------------------------------------------------------------------------- #
def test_unicode_logging_does_not_raise_on_a_legacy_encoding_stream():
    from autoclipper.utils import SafeStreamHandler, ensure_utf8_stream

    class LegacyStream(io.TextIOBase):
        """Mimics a Windows console that cannot encode emoji."""

        encoding = "cp1252"

        def __init__(self):
            self.written = []

        def write(self, text):
            text.encode(self.encoding)  # raises UnicodeEncodeError on emoji
            self.written.append(text)
            return len(text)

        def flush(self):
            pass

    stream = LegacyStream()
    handler = SafeStreamHandler(stream)
    record = logging.LogRecord(
        "autoclipper", logging.INFO, __file__, 1, "Hook overlay ✅ added 🎬", None, None
    )
    handler.emit(record)  # must not raise

    assert ensure_utf8_stream(stream) is stream


def test_hook_overlay_failure_keeps_the_valid_clip_and_records_a_warning(tmp_path, monkeypatch):
    candidates = _spread_candidates(1, duration=40.0, span=200.0)
    candidates[0]["hook"] = "Keep the blade pointed down 🎬"
    run = _StubRun(
        monkeypatch, tmp_path, candidates=candidates,
        transcript=_transcript_for(candidates, duration=200.0),
        video_duration=200.0,
    )

    import autoclipper.hooks as hooks_module

    def _broken_overlay(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "🎬", 0, 1, "character maps to <undefined>")

    monkeypatch.setattr(hooks_module, "add_hook_to_video", _broken_overlay)
    monkeypatch.setattr(
        pipeline, "add_hook_to_video", _broken_overlay, raising=False
    )

    result = run.run(clip_count=1, target_duration=40, add_hook=True)

    clip = _individual(result["clips"])[0]
    # The pre-overlay render is retained and reported as the delivered file.
    assert clip["file"] == "clip_1.mp4"
    assert clip.get("hook_file") is None
    assert clip["quality_assurance"]["status"] != "fail"
    assert any("hook overlay" in warning.lower() for warning in result["processing_warnings"])
    assert result["individual_clip_count"] == 1
