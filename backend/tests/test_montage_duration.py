import os

from autoclipper import pipeline
from autoclipper import montage as montage_module


def test_pipeline_reports_actual_montage_duration_without_cutting_transcript_segments(
    tmp_path, monkeypatch
):
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
    selected = [
        {"title": "First", "start": 0, "end": 17},
        {"title": "Third", "start": 34, "end": 51},
        {"title": "Fifth", "start": 68, "end": 85},
    ]
    captured = {}
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(video_path), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 1920, "height": 1080, "duration": 102, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "probe_duration", lambda path: 50.5)
    monkeypatch.setattr(
        pipeline,
        "cut_all",
        lambda video, clips, out_dir, title: captured.setdefault(
            "clips",
            [{"index": i + 1, "file": f"clip_{i + 1}.mp4", **clip} for i, clip in enumerate(clips)],
        ),
    )

    def fake_stitch(paths, output_path, crossfade_seconds=0.25):
        captured["montage_inputs"] = paths
        with open(output_path, "wb") as output:
            output.write(b"synthetic montage")
        return output_path

    monkeypatch.setattr(montage_module, "stitch_montage", fake_stitch)
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path), "checks": {},
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        source="stub-source",
        output_dir=str(tmp_path / "job"),
        vertical=False,
        subtitles=False,
        montage=True,
        selected_clips=selected,
        target_duration=60,
    )

    assert [(clip["start"], clip["end"]) for clip in captured["clips"]] == [
        (0, 17), (34, 51), (68, 85),
    ]
    assert len(captured["montage_inputs"]) == 3
    assert result["individual_clip_count"] == 3
    assert result["montage_file"] == "source_montage.mp4"
    montage_result = result["clips"][-1]
    assert montage_result["is_montage"] is True
    assert montage_result["duration"] == 50.5
    assert montage_result["duration_delta"] == -9.5
    assert any("actual duration is 50.50s" in warning for warning in result["processing_warnings"])


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
    # Two NON-overlapping user selections whose metadata is textually identical:
    # the generic ranking pass would score their similarity >= 0.78 and drop one.
    selected = [
        {"title": "Same hook", "hook": "Identical hook text", "description": "Identical description",
         "start": 0, "end": 17},
        {"title": "Same hook", "hook": "Identical hook text", "description": "Identical description",
         "start": 34, "end": 51},
    ]
    captured = {}
    rank_calls = []
    monkeypatch.setattr(pipeline, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "ingest", lambda *args, **kwargs: (str(video_path), "source"))
    monkeypatch.setattr(pipeline, "probe_video_info", lambda path: {
        "width": 1920, "height": 1080, "duration": 68, "has_audio": True,
    })
    monkeypatch.setattr(pipeline, "transcribe", lambda *args, **kwargs: transcript)
    monkeypatch.setattr(pipeline, "probe_duration", lambda path: 33.5)
    monkeypatch.setattr(
        pipeline,
        "rank_and_diversify_candidates",
        lambda *args, **kwargs: rank_calls.append(args) or [],
    )
    monkeypatch.setattr(
        pipeline,
        "cut_all",
        lambda video, clips, out_dir, title: captured.setdefault(
            "clips",
            [{"index": i + 1, "file": f"clip_{i + 1}.mp4", **clip} for i, clip in enumerate(clips)],
        ),
    )

    def fake_stitch(paths, output_path, crossfade_seconds=0.25):
        captured["montage_inputs"] = paths
        with open(output_path, "wb") as output:
            output.write(b"synthetic montage")
        return output_path

    monkeypatch.setattr(montage_module, "stitch_montage", fake_stitch)
    monkeypatch.setattr(pipeline, "verify_rendered_file", lambda path, **kwargs: {
        "status": "pass", "file": os.path.basename(path), "checks": {},
        "warnings": [], "errors": [],
    })
    monkeypatch.setattr(pipeline, "write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "save_to_library",
        lambda output_root, title, job_id, clips: os.path.join(output_root, "library"),
    )

    result = pipeline.run(
        source="stub-source",
        output_dir=str(tmp_path / "job"),
        vertical=False,
        subtitles=False,
        montage=True,
        selected_clips=selected,
        target_duration=40,
    )

    assert rank_calls == []  # montage mode must bypass similarity de-duplication
    assert [(clip["start"], clip["end"]) for clip in captured["clips"]] == [(0, 17), (34, 51)]
    assert len(captured["montage_inputs"]) == 2
    assert result["individual_clip_count"] == 2
    assert result["montage_file"] == "source_montage.mp4"


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
    import pytest

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
