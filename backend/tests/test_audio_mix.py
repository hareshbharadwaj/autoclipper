import pytest

from autoclipper.audio_mix import (
    _concise_ffmpeg_error,
    build_audio_filtergraph,
    mix_audio_layers,
)


def test_voiceover_only_graph_has_no_unused_asplit_output():
    graph = build_audio_filtergraph(
        duration=51.51,
        has_audio=True,
        has_voiceover=True,
        has_music=False,
    )
    assert "asplit" not in graph
    assert "[1:a:0]" in graph and "[vo]" in graph
    assert "amix=inputs=2:duration=longest" in graph
    assert "apad=whole_dur=51.5100" in graph


def test_voiceover_and_music_graph_connects_both_voiceover_branches():
    graph = build_audio_filtergraph(
        duration=12,
        has_audio=True,
        has_voiceover=True,
        has_music=True,
        music_volume=0.2,
    )
    assert "asplit=2[vo][vo_side]" in graph
    assert "[music0][vo_side]sidechaincompress" in graph
    assert "[orig][vo][music]amix=inputs=3" in graph
    assert graph.count("[vo_side]") == 2  # split output + sidechain input


def test_music_only_graph_and_silent_video_are_normalized_to_stereo():
    graph = build_audio_filtergraph(
        duration=3,
        has_audio=False,
        has_voiceover=False,
        has_music=True,
    )
    assert "anullsrc=r=44100:cl=stereo" in graph
    assert "aformat=sample_fmts=fltp:channel_layouts=stereo" in graph
    assert "[orig][music]amix=inputs=2" in graph
    assert "asplit" not in graph


def test_voiceover_only_graph_supports_video_without_original_audio():
    graph = build_audio_filtergraph(
        duration=5,
        has_audio=False,
        has_voiceover=True,
        has_music=False,
    )
    assert "anullsrc=r=44100:cl=stereo" in graph
    assert "[orig][vo]amix=inputs=2" in graph
    assert "asplit" not in graph


def test_graph_rejects_no_optional_tracks():
    with pytest.raises(ValueError, match="At least one optional"):
        build_audio_filtergraph(5, has_audio=True, has_voiceover=False, has_music=False)


def test_no_optional_audio_tracks_leave_the_original_video_untouched(tmp_path):
    video = tmp_path / "original.mp4"
    video.write_bytes(b"placeholder")
    output = tmp_path / "unused-output.mp4"
    assert mix_audio_layers(str(video), str(output)) == str(video)
    assert not output.exists()


def test_ffmpeg_error_summary_keeps_actionable_filter_message_only():
    summary = _concise_ffmpeg_error(
        "Command failed (1): ffmpeg -y -i private/path.mp4\n"
        "ffmpeg version blah\n"
        "[AVFilterGraph] Filter 'asplit' has output 0 (vo_side) unconnected\n"
        "Conversion failed!"
    )
    assert summary == "[AVFilterGraph] Filter 'asplit' has output 0 (vo_side) unconnected"
    assert "ffmpeg version" not in summary
    assert "private/path" not in summary
