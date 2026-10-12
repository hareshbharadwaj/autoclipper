"""Optional voiceover and background-music mixing with automatic music ducking."""
import os

from .utils import logger, probe_duration, probe_video_info, run_command


def _clamp_volume(value, default):
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def build_audio_filtergraph(
    duration,
    has_audio,
    has_voiceover,
    has_music,
    music_volume=0.18,
    original_volume=0.45,
):
    """Build a duration-bounded stereo mix graph for the enabled tracks.

    The voiceover is split only when the music sidechain actually consumes the
    second branch. All tracks are resampled and padded/trimmed to the video
    duration so short narration never shortens the output.
    """
    duration = max(0.01, float(duration))
    duration_arg = f"{duration:.4f}"
    graph = []
    labels = ["orig"]

    if has_audio:
        volume = _clamp_volume(original_volume, 0.45)
        graph.append(
            f"[0:a:0]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={volume:.3f},apad=whole_dur={duration_arg},"
            f"atrim=0:{duration_arg}[orig]"
        )
    else:
        graph.append(
            f"anullsrc=r=44100:cl=stereo:d={duration_arg},atrim=0:{duration_arg}[orig]"
        )

    input_index = 1
    if has_voiceover:
        voice_filter = (
            f"[{input_index}:a:0]aresample=44100,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0,"
            f"apad=whole_dur={duration_arg},atrim=0:{duration_arg}"
        )
        if has_music:
            graph.append(f"{voice_filter},asplit=2[vo][vo_side]")
        else:
            graph.append(f"{voice_filter}[vo]")
        labels.append("vo")
        input_index += 1

    if has_music:
        volume = _clamp_volume(music_volume, 0.18)
        graph.append(
            f"[{input_index}:a:0]aresample=44100,"
            f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={volume:.3f},apad=whole_dur={duration_arg},"
            f"atrim=0:{duration_arg}[music0]"
        )
        if has_voiceover:
            graph.append(
                "[music0][vo_side]sidechaincompress=threshold=0.025:ratio=8:"
                "attack=20:release=320[music]"
            )
        else:
            graph.append("[music0]anull[music]")
        labels.append("music")

    if has_voiceover and has_music:
        mix_inputs = "[orig][vo][music]"
    elif has_voiceover:
        mix_inputs = "[orig][vo]"
    elif has_music:
        mix_inputs = "[orig][music]"
    else:
        raise ValueError("At least one optional audio track is required for a mix.")

    graph.append(
        f"{mix_inputs}amix=inputs={len(labels)}:duration=longest:dropout_transition=2,"
        f"alimiter=limit=0.95,atrim=0:{duration_arg}[aout]"
    )
    return ";".join(graph)


def _concise_ffmpeg_error(error):
    """Return one actionable FFmpeg diagnostic without its banner/command dump."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    priority_markers = (
        "unconnected", "invalid argument", "no such file", "matches no streams",
        "error initializing", "failed to configure", "could not", "unable to",
    )
    for marker in priority_markers:
        for line in reversed(lines):
            if marker in line.lower():
                return line[-240:]
    for line in reversed(lines):
        lowered = line.lower()
        if "conversion failed" not in lowered and not lowered.startswith("command failed"):
            return line[-240:]
    return "FFmpeg returned a non-zero exit status."


def mix_audio_layers(video_in, video_out, voiceover_path=None, music_path=None,
                     music_volume=0.18, original_volume=0.45):
    """Mix optional VO/music while ducking music under narration.

    Existing clip audio is retained at a lower level. Background music is
    sidechain-ducked under narration only when both optional tracks are present.
    A shorter voiceover plays once from the beginning; the other tracks and the
    exported video continue to the original video duration (no stretching or
    looping of speech).
    """
    voiceover_path = voiceover_path if voiceover_path and os.path.isfile(voiceover_path) else None
    music_path = music_path if music_path and os.path.isfile(music_path) else None
    if not voiceover_path and not music_path:
        return video_in

    duration = probe_duration(video_in)
    if duration <= 0:
        raise RuntimeError("Cannot mix audio because the video duration is unknown.")
    info = probe_video_info(video_in)
    has_audio = bool(info.get("has_audio"))
    inputs = ["-i", video_in]
    if voiceover_path:
        inputs.extend(["-i", voiceover_path])
    if music_path:
        inputs.extend(["-stream_loop", "-1", "-i", music_path])

    graph = build_audio_filtergraph(
        duration,
        has_audio=has_audio,
        has_voiceover=bool(voiceover_path),
        has_music=bool(music_path),
        music_volume=music_volume,
        original_volume=original_volume,
    )

    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)
    command = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex_threads", "1",
        "-filter_complex", graph,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.4f}", "-movflags", "+faststart", video_out,
    ]
    try:
        run_command(command)
    except RuntimeError as exc:
        detail = _concise_ffmpeg_error(exc)
        logger.error("FFmpeg audio mix failed: %s", detail)
        raise RuntimeError(f"FFmpeg audio mix failed: {detail}") from exc

    if not os.path.isfile(video_out) or os.path.getsize(video_out) == 0:
        raise RuntimeError("FFmpeg did not produce an audio-mixed video.")
    actual = probe_duration(video_out)
    if actual <= 0 or abs(actual - duration) > max(1.0, duration * 0.05):
        raise RuntimeError(
            f"Audio mix duration check failed (expected {duration:.2f}s, got {actual:.2f}s)."
        )
    logger.info("Audio mix created: %s (voiceover=%s, music=%s, duration=%.2fs)",
                video_out, bool(voiceover_path), bool(music_path), actual)
    return video_out
