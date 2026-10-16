import os
import pathlib

from autoclipper import pipeline
from autoclipper import montage as montage_module


def _stub_source(monkeypatch, video_path, transcript, *, montage_duration=50.5,
                 individual_candidates=None, probe=None):
    """Patch the media/model boundary; capture what actually got rendered."""
    captured = {"clips": [], "montage_shots": []}
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(video_path), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 1920, "height": 1080,
        "duration": transcript.get("duration") or 100, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "probe_duration", lambda path: montage_duration)
    monkeypatch.setattr(
        pipeline, "find_viral_clips",
        lambda *args, **kwargs: [dict(c) for c in (individual_candidates or [])],
    )

    def fake_cut_all(video, clips, out_dir, title):
        bucket = "montage_shots" if "montage_shots" in str(out_dir) else "clips"
        captured[bucket].extend(
            [(float(clip["start"]), float(clip["end"])) for clip in clips]
        )
        made = []
        for index, clip in enumerate(clips):
            name = f"{bucket}_{index + 1}.mp4"
            (pathlib.Path(out_dir) / name).write_bytes(b"synthetic clip")
            made.append({"index": index + 1, "file": name, **clip})
        return made

    monkeypatch.setattr(pipeline, "cut_all", fake_cut_all)

    def fake_stitch(paths, output_path, crossfade_seconds=0.25):
        captured["montage_inputs"] = list(paths)
        with open(output_path, "wb") as handle:
            handle.write(b"synthetic montage")
        return output_path

    montage_module.stitch_montage = fake_stitch
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path),
        "checks": {"duration": {"ok": True, "actual_seconds": probe or 60.0}},
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline, "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )
    return captured


def test_montage_is_a_separate_output_counted_apart_from_individual_clips(
    tmp_path, monkeypatch
):
    """A montage is an additional output: it never replaces requested clips.

    Regression for the coupled behaviour where enabling montage rewrote the
    requested individual clip count to the number of selected angles.
    """
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"stub")
    transcript = {
        "duration": 102,
        "language": "English",
        "segments": [
            {"start": 0, "end": 17, "text": "Same hook is the first complete thought."},
            {"start": 17, "end": 34, "text": "Second complete thought."},
            {"start": 34, "end": 51, "text": "Same hook is the third complete thought."},
            {"start": 51, "end": 68, "text": "Fourth complete thought."},
            {"start": 68, "end": 85, "text": "Fifth complete thought."},
            {"start": 85, "end": 102, "text": "Sixth complete thought."},
        ],
        "words": [],
    }
    montage_moments = [
        {"title": "First", "start": 0, "end": 17},
        {"title": "Third", "start": 34, "end": 51},
        {"title": "Fifth", "start": 68, "end": 85},
    ]
    # Two distinct individual clips requested *in addition* to the montage.
    individual = [
        {"title": "Opening", "start": 0, "end": 34, "description": "first lesson"},
        {"title": "Closing", "start": 68, "end": 102, "description": "final lesson"},
    ]
    captured = _stub_source(
        monkeypatch, video_path, transcript,
        individual_candidates=individual, montage_duration=50.5,
    )

    result = pipeline.run(
        source="stub-source",
        output_dir=str(tmp_path / "job"),
        vertical=False,
        subtitles=False,
        montage=True,
        selected_clips=montage_moments,
        clip_count=2,
        target_duration=60,
    )

    # Individual clips are the requested two, planned independently.
    assert result["requested_clip_count"] == 2
    assert result["individual_clip_count"] == 2
    assert len(captured["clips"]) == 2

    # The montage is built from the selected moments, on transcript boundaries.
    assert captured["montage_shots"] == [(0.0, 17.0), (34.0, 51.0), (68.0, 85.0)]
    assert len(captured["montage_inputs"]) == 3
    assert result["montage_file"] == "source_montage.mp4"
    assert result["montage_count"] == 1

    # Montage accounting is separate and reports the real stitched duration.
    montage_result = next(c for c in result["clips"] if c.get("is_montage"))
    assert montage_result["is_montage"] is True
    assert montage_result["duration"] == 50.5
    assert montage_result["duration_delta"] == -9.5
    assert result["montage_report"]["actual_seconds"] == 50.5
    assert any("actual duration is 50.50s" in warning for warning in result["processing_warnings"])
    # The montage is never counted as one of the individual clips.
    assert sum(1 for c in result["clips"] if not c.get("is_montage")) == 2


def test_montage_keeps_user_selected_moments_with_identical_text(tmp_path, monkeypatch):
    """Text-similarity de-duplication must never drop a required montage shot."""
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"stub")
    transcript = {
        "duration": 68,
        "language": "English",
        "segments": [
            {"start": 0, "end": 17, "text": "Same hook is the first complete thought."},
            {"start": 17, "end": 34, "text": "Second complete thought."},
            {"start": 34, "end": 51, "text": "Same hook is the third complete thought."},
            {"start": 51, "end": 68, "text": "Fourth complete thought."},
        ],
        "words": [],
    }
    # Two NON-overlapping user selections whose metadata is textually identical.
    montage_moments = [
        {"title": "Same hook", "hook": "Identical hook text",
         "description": "Identical description", "start": 0, "end": 17},
        {"title": "Same hook", "hook": "Identical hook text",
         "description": "Identical description", "start": 34, "end": 51},
    ]
    individual = [
        {"title": "Opening", "start": 0, "end": 34, "description": "distinct opening lesson"},
        {"title": "Closing", "start": 51, "end": 68, "description": "distinct closing lesson"},
    ]
    captured = _stub_source(
        monkeypatch, video_path, transcript,
        individual_candidates=individual, montage_duration=33.5,
    )

    result = pipeline.run(
        source="stub-source",
        output_dir=str(tmp_path / "job"),
        vertical=False,
        subtitles=False,
        montage=True,
        selected_clips=montage_moments,
        clip_count=2,
        target_duration=40,
    )

    assert captured["montage_shots"] == [(0.0, 17.0), (34.0, 51.0)]
    assert len(captured["montage_inputs"]) == 2
    assert result["montage_file"] == "source_montage.mp4"
    assert result["individual_clip_count"] == 2


def test_montage_uses_topic_evidence_instead_of_wrong_analysis_timestamp():
    """A bad hint must not force a selected angle onto an unrelated passage."""
    transcript = {
        "duration": 60,
        "segments": [
            {"start": 0, "end": 15, "text": "Knife safety starts with keeping the blade pointed down."},
            {"start": 15, "end": 30, "text": "Never wave a knife and always keep knife safety in mind."},
            {"start": 30, "end": 45, "text": "When cutting a round vegetable that can roll, use the tunnel method."},
            {"start": 45, "end": 60, "text": "Now they will not roll away, and you have a flat surface to cut safely."},
        ],
    }
    selected = [
        {
            "title": "Knife Safety Rules",
            "description": "Avoid dangerous knife handling mistakes.",
            "start": 0, "end": 30,
        },
        {
            "title": "Stop Your Veggies From Rolling",
            "description": "Use the tunnel method so round vegetables do not roll away.",
            # Deliberately wrong timestamp: semantic matching should find 30-60.
            "start": 0, "end": 15,
        },
    ]

    clips = pipeline._fit_montage_candidates(selected, transcript, 60, 60)

    assert len(clips) == 2
    assert clips[0]["start"] == 0
    assert clips[0]["end"] == 30
    assert clips[1]["start"] == 30
    assert clips[1]["end"] == 60
    assert all(clip["target_duration"] == 30.12 or clip["target_duration"] == 30.0
               for clip in clips)
    assert clips[1]["topic_match_score"] > 0


def test_montage_rejects_selected_angle_not_grounded_in_transcript():
    transcript = {
        "duration": 20,
        "segments": [
            {"start": 0, "end": 10, "text": "The tomato is red and ripe."},
            {"start": 10, "end": 20, "text": "Cut the tomato into small pieces."},
        ],
    }
    selected = [
        {"title": "Quantum Physics", "description": "Black holes and quantum gravity",
         "start": 0, "end": 10},
        {"title": "Space Travel", "description": "Rockets travelling to Mars",
         "start": 10, "end": 20},
    ]

    import pytest
    with pytest.raises(ValueError, match="Could not ground selected angle"):
        pipeline._fit_montage_candidates(selected, transcript, 20, 20)



def test_short_montages_use_word_timestamps_when_whisper_segments_are_too_coarse():
    """10s/30s montages should not fail just because Whisper segments are 15s long."""
    cases = [
        (10, [
            {"title": "Knife safety", "description": "Keep the blade pointed down."},
            {"title": "Vegetable cutting", "description": "Use the tunnel method for round vegetables."},
        ]),
        (30, [
            {"title": "Knife safety", "description": "Keep the blade pointed down."},
            {"title": "Vegetable cutting", "description": "Use the tunnel method for round vegetables."},
            {"title": "Portrait framing", "description": "Keep the speaker centered in portrait video."},
        ]),
    ]
    source_texts = [
        "Knife safety starts here. Keep the blade pointed down.",
        "Vegetable cutting uses a tunnel method. Stop round vegetables rolling away.",
        "Portrait framing keeps the speaker centered. Portrait video follows motion.",
    ]

    for target, angles in cases:
        segments = []
        words = []
        for segment_index, text in enumerate(source_texts[:len(angles)]):
            start = segment_index * 15.0
            end = start + 15.0
            segments.append({"start": start, "end": end, "text": text})
            tokens = text.split()
            step = 15.0 / len(tokens)
            for word_index, token in enumerate(tokens):
                words.append({
                    "word": token,
                    "start": start + word_index * step,
                    "end": start + (word_index + 1) * step,
                })
        transcript = {"duration": 15.0 * len(angles), "segments": segments, "words": words}
        selected = [
            {**angle, "start": i * 15.0, "end": (i + 1) * 15.0}
            for i, angle in enumerate(angles)
        ]

        clips = pipeline._fit_montage_candidates(
            selected, transcript, target, transcript["duration"]
        )

        assert len(clips) == len(angles)
        assert all(clip["end"] - clip["start"] <= target / len(angles) * 1.6
                   for clip in clips)
        assert all(clip["topic_match_score"] > 0 for clip in clips)
