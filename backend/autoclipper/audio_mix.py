"""Optional voiceover and background-music mixing with automatic music ducking."""
import os

from .utils import logger, probe_duration, probe_video_info, run_command


def mix_audio_layers(video_in, video_out, voiceover_path=None, music_path=None,
                     music_volume=0.18, original_volume=0.45):
    """Mix optional VO/music while ducking music under narration.

    Existing clip audio is retained at a lower level. Music is ducked by the
    voiceover sidechain when both are supplied. With no optional tracks, this
    function returns the input unchanged.
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

    graph = []
    labels = []
    if has_audio:
        graph.append(f"[0:a:0]volume={max(0.0, min(float(original_volume), 1.0)):.3f}[orig]")
    else:
        graph.append(f"anullsrc=r=44100:cl=stereo:d={duration:.4f}[orig]")
    labels.append("orig")

    input_index = 1
    voice_label = None
    if voiceover_path:
        voice_label = "vo"
        graph.append(f"[{input_index}:a:0]aresample=44100,volume=1.0,asplit=2[vo][vo_side]")
        labels.append("vo")
        input_index += 1

    if music_path:
        vol = max(0.0, min(float(music_volume), 1.0))
        graph.append(f"[{input_index}:a:0]aresample=44100,volume={vol:.3f}[music0]")
        if voice_label:
            graph.append(
                "[music0][vo_side]sidechaincompress=threshold=0.025:ratio=8:"
                "attack=20:release=320[music]"
            )
        else:
            graph.append("[music0]anull[music]")
        labels.append("music")

    if voice_label and music_path:
        mix_inputs = "[orig][vo][music]"
    elif voice_label:
        mix_inputs = "[orig][vo]"
    else:
        mix_inputs = "[orig][music]"
    graph.append(f"{mix_inputs}amix=inputs={len(labels)}:duration=first:"
                 "dropout_transition=2,alimiter=limit=0.95[aout]")

    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)
    run_command([
        "ffmpeg", "-y", *inputs,
        "-filter_complex_threads", "1",
        "-filter_complex", ";".join(graph),
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.4f}", "-movflags", "+faststart", video_out,
    ])
    if not os.path.isfile(video_out) or os.path.getsize(video_out) == 0:
        raise RuntimeError("FFmpeg did not produce an audio-mixed video.")
    actual = probe_duration(video_out)
    if actual <= 0 or abs(actual - duration) > max(1.0, duration * 0.05):
        raise RuntimeError(
            f"Audio mix duration check failed (expected {duration:.2f}s, got {actual:.2f}s)."
        )
    logger.info("Audio mix created: %s (voiceover=%s, music=%s)",
                video_out, bool(voiceover_path), bool(music_path))
    return video_out
