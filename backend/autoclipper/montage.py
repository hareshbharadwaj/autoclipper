"""Multi-shot montage assembly with video and audio crossfades."""
import os

from .utils import logger, probe_duration, probe_video_info, run_command


def stitch_montage(video_paths, output_path, crossfade_seconds=0.25):
    """Stitch 2-3 processed clips with short visual/audio crossfades.

    Inputs are normalized to the first clip's dimensions, 30 fps and stereo
    audio. Silent inputs receive a generated silent track so concat remains safe.
    """
    paths = [os.path.abspath(p) for p in (video_paths or []) if p and os.path.isfile(p)]
    if not 2 <= len(paths) <= 3:
        raise ValueError("A montage requires 2 or 3 existing clip files.")

    durations = [probe_duration(path) for path in paths]
    if any(duration <= 0 for duration in durations):
        raise RuntimeError("Could not determine the duration of every montage segment.")

    info = probe_video_info(paths[0])
    width = int(info.get("width") or 720)
    height = int(info.get("height") or 1280)
    width += width % 2
    height += height % 2
    fade = max(0.0, min(float(crossfade_seconds), min(durations) / 4.0))
    inputs = []
    graph = []
    video_labels = []
    audio_labels = []

    for i, (path, duration) in enumerate(zip(paths, durations)):
        inputs.extend(["-i", path])
        graph.append(
            f"[{i}:v:0]fps=30,settb=AVTB,setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{i}]"
        )
        if probe_video_info(path).get("has_audio"):
            graph.append(
                f"[{i}:a:0]aresample=44100,aformat=sample_fmts=fltp:"
                f"channel_layouts=stereo[a{i}]"
            )
        else:
            graph.append(f"anullsrc=r=44100:cl=stereo:d={duration:.4f}[a{i}]")
        video_labels.append(f"v{i}")
        audio_labels.append(f"a{i}")

    v_current, a_current = video_labels[0], audio_labels[0]
    elapsed = durations[0]
    for i in range(1, len(paths)):
        v_next = f"vx{i}"
        a_next = f"ax{i}"
        offset = max(0.0, elapsed - fade * i)
        graph.append(
            f"[{v_current}][{video_labels[i]}]xfade=transition=fade:"
            f"duration={fade:.4f}:offset={offset:.4f}[{v_next}]"
        )
        graph.append(
            f"[{a_current}][{audio_labels[i]}]acrossfade=d={fade:.4f}:"
            f"c1=tri:c2=tri[{a_next}]"
        )
        v_current, a_current = v_next, a_next
        elapsed += durations[i]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    run_command([
        "ffmpeg", "-y", *inputs,
        "-filter_complex_threads", "1",
        "-filter_complex", ";".join(graph),
        "-map", f"[{v_current}]", "-map", f"[{a_current}]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        output_path,
    ])
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("FFmpeg did not produce a valid montage file.")
    expected = sum(durations) - fade * (len(paths) - 1)
    actual = probe_duration(output_path)
    if actual <= 0 or abs(actual - expected) > max(1.5, expected * 0.08):
        raise RuntimeError(
            f"Montage duration check failed (expected about {expected:.2f}s, got {actual:.2f}s)."
        )
    logger.info("Montage created: %s (%.2fs, %d segments)", output_path, actual, len(paths))
    return output_path
