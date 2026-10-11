"""End-to-end orchestration: ingest -> transcribe -> analyze -> cut."""
import json
import os
import time

from . import config
from .analyze import (
    find_viral_clips, _filter_clips_to_bounds,
    _resolve_clip_bounds, _validate_clips,
)
from .clip import cut_all
from .ingest import ingest
from .reframe import VERTICAL_ASPECT, reframe_clip
from .subtitles import add_subtitles
from .transcribe import transcribe
from .utils import (
    ASPECT_PRESETS,
    ensure_ffmpeg,
    logger,
    probe_video_info,
    probe_duration,
    sanitize_filename,
    write_json,
    save_to_library,
)





def _load_cached_transcript(path):
    """Load a prior analysis transcript and reject malformed cache files."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("The analyzed transcript cache is missing; analyze the video again.")
    with open(path, "r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    if not isinstance(transcript, dict) or not isinstance(transcript.get("segments"), list):
        raise ValueError("The cached transcript is malformed; analyze the video again.")
    return transcript


def _normalize_target_duration(target_duration):
    """Clamp oversized requests to the hard ceiling; reject malformed values."""
    if target_duration is None:
        return None
    try:
        value = float(target_duration)
    except (TypeError, ValueError):
        raise ValueError("target_duration must be a number between 5 and 120 seconds.")
    return min(value, 120.0)


def _fit_montage_candidates(candidates, transcript, target_duration, total_duration):
    """Turn 2-3 selected analysis angles into transcript-aligned montage shots.

    Segment boundaries come from Whisper timestamps. We preserve whole transcript
    segments rather than arbitrarily cutting a word to hit the duration exactly.
    """
    segments = []
    for segment in transcript.get("segments", []) or []:
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start and (segment.get("text") or "").strip():
            segments.append({"start": start, "end": end})
    segments.sort(key=lambda item: item["start"])
    if not segments:
        raise ValueError("Montage needs word/sentence timestamps from the transcript.")

    shot_count = max(1, len(candidates))
    per_shot = max(3.0, float(target_duration) / shot_count)
    # Leave a little room for sentence-boundary rounding, but never exceed the
    # 120-second total ceiling when the requested target is already at the cap.
    max_shot = per_shot + min(0.75, max(0.0, (120.0 - float(target_duration)) / shot_count))
    chosen = []
    for angle in candidates:
        try:
            hint_start = float(angle.get("start", angle.get("start_hint")))
            hint_end = float(angle.get("end", angle.get("end_hint")))
        except (TypeError, ValueError):
            continue
        if hint_end <= hint_start:
            continue
        desired_start = max(0.0, min(hint_start, max(0.0, total_duration - 0.1)))
        start_idx = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - desired_start))
        end_idx = start_idx
        while end_idx + 1 < len(segments) and segments[end_idx]["end"] - segments[start_idx]["start"] < per_shot - 1.0:
            end_idx += 1
        shot_start = segments[start_idx]["start"]
        shot_end = segments[end_idx]["end"]
        if shot_end - shot_start > max_shot:
            possible = [seg for seg in segments[start_idx:end_idx + 1]
                        if seg["end"] - shot_start <= max_shot]
            if not possible:
                continue
            shot_end = possible[-1]["end"]
        if total_duration:
            shot_start = max(0.0, min(shot_start, total_duration))
            shot_end = max(0.0, min(shot_end, total_duration))
        if shot_end <= shot_start:
            continue
        if any(shot_start < item["end"] and shot_end > item["start"] for item in chosen):
            # Avoid duplicate/overlapping moments; leave a clear error if all angles overlap.
            continue
        chosen.append({
            "start": round(shot_start, 2),
            "end": round(shot_end, 2),
            "title": (angle.get("title") or "Montage moment")[:80],
            "description": (angle.get("description") or "")[:220],
            "hook": (angle.get("hook") or "")[:120],
            "reason": angle.get("reason") or "Selected from whole-video analysis",
            "story_arc": "Selected angle; transcript-aligned montage segment",
        })
    chosen.sort(key=lambda item: item["start"])
    if not 2 <= len(chosen) <= 3:
        raise ValueError("Could not find 2-3 distinct, non-overlapping transcript moments for this montage.")
    return chosen


def _enforce_clip_constraints(clips, min_clip, max_clip, clip_count, video_dur, trim_long=True, pad_short=True):
    """Adjust Gemini's candidates to respect the user's count and duration window."""
    cleaned = []
    for c in clips:
        try:
            start = float(c.get("start", 0.0))
            end = float(c.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        dur = end - start

        # Trim clips longer than the requested max.
        if trim_long and max_clip and dur > max_clip:
            end = start + max_clip
            dur = max_clip

        # Pad short clips outward (within video bounds) up to the min duration.
        if pad_short and min_clip and dur < min_clip:
            need = min_clip - dur
            pad_before = min(start, need / 2.0)
            pad_after = min(max(0.0, video_dur - end), need - pad_before)
            # If we still need more, take the rest from the front.
            if pad_before + pad_after < need:
                pad_before = min(start, need - pad_after)
            start = max(0.0, start - pad_before)
            end = min(video_dur or end, end + pad_after)
            # Final clamp so we never exceed min by padding past video end.
            if video_dur and end - start < min_clip and start > 0:
                start = max(0.0, end - min_clip)

        cleaned.append({**c, "start": round(start, 2), "end": round(end, 2)})

    # Cap the total number of clips at the requested count.
    if clip_count and len(cleaned) > clip_count:
        cleaned = cleaned[:clip_count]

    return cleaned


def run(
    source: str,
    output_dir: str = "output",
    groq_key: str = None,
    gemini_key: str = None,
    whisper_model: str = None,
    gemini_model: str = None,
    vertical: bool = True,
    use_yolo: bool = True,
    subtitles: bool = True,
    split_screen: bool = False,
    force_hd: bool = False,
    cookies_text: str = None,
    clip_count: int = None,
    min_clip: float = None,
    max_clip: float = None,
    content_type: str = None,
    target_duration: float = None,
    custom_prompt: str = None,
    selected_clip: dict = None,
    aspect_ratio: str = None,
    add_hook: bool = False,
    montage: bool = False,
    selected_clips: list = None,
    voiceover_text: str = None,
    voiceover_engine: str = "auto",
    voiceover_voice: str = None,
    background_music_path: str = None,
    music_volume: float = 0.18,
    cached_transcript_path: str = None,
    on_clip=None,
) -> dict:
    """Run the full auto-clip pipeline and return a result summary dict.

    If *on_clip* is provided it will be called with ``(clip_dict, index, total)``
    after each clip is fully processed (cut + reframe + subtitle), enabling the
    caller to stream partial results to the frontend.
    """
    # Clamp oversized requests before resolving target bounds.
    target_duration = _normalize_target_duration(target_duration)
    if max_clip is not None:
        try:
            max_clip = min(float(max_clip), float(config.HARD_MAX_CLIP_SECONDS))
        except (TypeError, ValueError):
            max_clip = None
    if montage:
        if not selected_clips or not 2 <= len(selected_clips) <= 3:
            raise ValueError("Montage mode requires 2 or 3 selected video moments.")
        effective_min_clip, effective_max_clip = 3.0, float(config.HARD_MAX_CLIP_SECONDS)
    else:
        effective_min_clip, effective_max_clip = _resolve_clip_bounds(
            min_clip, max_clip, target_duration
        )
    ensure_ffmpeg()
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Ingest
    video_path, title = ingest(source, output_dir, force_hd=force_hd, cookies_text=cookies_text)
    base_name = sanitize_filename(title)
    video_info = probe_video_info(video_path)
    if not video_info.get("duration"):
        video_info["duration"] = 0.0

    # 1b. Resolve the output aspect ratio (9:16 portrait / 1:1 square / 16:9).
    #     `vertical` is kept for backwards compatibility with older clients.
    target_aspect = None
    aspect_key = (aspect_ratio or "").strip()
    if aspect_key and aspect_key in ASPECT_PRESETS:
        if aspect_key != "16:9":
            target_aspect = ASPECT_PRESETS[aspect_key]["ratio"]
    elif aspect_key and aspect_key not in ASPECT_PRESETS:
        logger.warning("Unknown aspect_ratio '%s'; falling back to 9:16.", aspect_key)
        target_aspect = VERTICAL_ASPECT
    elif vertical:
        target_aspect = VERTICAL_ASPECT
    if target_aspect is not None:
        logger.info("Reframing to %s (aspect %.4f)", aspect_key or "9:16", target_aspect)
    else:
        logger.info("Keeping original 16:9 framing (no crop).")

    # 2. Reuse the whole-video transcript when analysis-first already paid for it.
    if cached_transcript_path:
        transcript = _load_cached_transcript(cached_transcript_path)
        logger.info("Reusing cached whole-video transcript; skipping a second Groq call.")
    else:
        transcript = transcribe(
            video_path, output_dir, api_key=groq_key, model=whisper_model
        )

    # 3. Analyze (Gemini) — language is auto-read from the transcript so that
    # titles / descriptions / hooks are written in the video's own language.
    if montage:
        clip_candidates = _fit_montage_candidates(
            selected_clips, transcript,
            target_duration=float(target_duration or 40),
            total_duration=video_info.get("duration") or transcript.get("duration", 0.0),
        )
    elif selected_clip:
        # Reuse the selected candidate only after validating its source timestamps.
        clip_candidates = _validate_clips(
            [selected_clip],
            total_duration=video_info.get("duration") or transcript.get("duration", 0.0),
            min_clip=effective_min_clip,
            max_clip=effective_max_clip,
        )
    else:
        clip_candidates = find_viral_clips(
            transcript, api_key=gemini_key, model=gemini_model,
            clip_count=clip_count,
            min_clip=effective_min_clip if target_duration is not None else min_clip,
            max_clip=effective_max_clip if target_duration is not None else max_clip,
            content_type=content_type, target_duration=target_duration,
            custom_prompt=custom_prompt,
        )
    logger.info(
        "Clip params -> requested count=%s min=%s max=%s | Gemini returned %d candidates",
        clip_count, min_clip, max_clip, len(clip_candidates),
    )

    # Apply hard duration bounds without blindly padding with unrelated footage or
    # truncating a sentence/payoff. Candidate selection should respect the bounds.
    video_dur = video_info.get("duration") or transcript.get("duration") or 0.0
    if effective_min_clip is not None or effective_max_clip is not None or clip_count:
        clip_candidates = _enforce_clip_constraints(
            clip_candidates, effective_min_clip, effective_max_clip,
            1 if selected_clip else clip_count, video_dur,
            trim_long=False, pad_short=False,
        )
        if not montage:
            clip_candidates = _filter_clips_to_bounds(
                clip_candidates,
                min_clip=effective_min_clip,
                max_clip=effective_max_clip,
                min_tolerance=0.0 if target_duration is not None else 2.0,
                max_tolerance=0.0 if target_duration is not None else 3.0,
            )
        logger.info("After enforcement: %d clip(s)", len(clip_candidates))
    if not clip_candidates:
        logger.warning("No viral clips detected.")

    # 4. Cut clips (FFmpeg)
    clips = cut_all(video_path, clip_candidates, output_dir, base_name)
    total_clips = len(clips)

    # Process each clip one-by-one so that partial results stream to the UI.
    processed = []
    for idx, clip in enumerate(clips):
        # 4b. Chat-split
        if split_screen:
            from .gaming_layout import auto_chat_split_clip as _auto_split
            src = os.path.join(output_dir, clip["file"])
            if os.path.exists(src):
                s_name = clip["file"].rsplit(".", 1)[0] + "_split.mp4"
                s_path = os.path.join(output_dir, s_name)
                try:
                    _auto_split(src, s_path)
                    os.remove(src)
                    clip["file"] = s_name
                    clip["split_screen"] = True
                    logger.info("Chat-split applied to %s -> %s", src, s_name)
                except Exception as e:
                    logger.warning("Chat-split skipped for %s: %s", src, e)

        # 5. Reframe to the requested aspect ratio (9:16 / 1:1), skipping the
        #    crop entirely for 16:9 so the original framing is preserved.
        if target_aspect is not None and not clip.get("split_screen"):
            src = os.path.join(output_dir, clip["file"])
            if os.path.exists(src):
                suffix = "9x16" if abs(target_aspect - (9 / 16)) < 1e-6 else (
                    "1x1" if abs(target_aspect - 1.0) < 1e-6 else "crop"
                )
                v_name = clip["file"].rsplit(".", 1)[0] + f"_{suffix}.mp4"
                v_path = os.path.join(output_dir, v_name)
                if reframe_clip(src, v_path, use_yolo=use_yolo, aspect=target_aspect):
                    clip["vertical_file"] = v_name
                    os.remove(src)
                    clip["file"] = v_name

        # 6. Generate + burn subtitle tracks (word-level karaoke captions).
        #    Captions are burned in by default so every exported clip has them.
        if subtitles:
            add_subtitles([clip], transcript, output_dir, burn=True)

        # 6b. Burn the curiosity-hook banner on top of the finished clip.
        if add_hook and clip.get("hook"):
            try:
                from .hooks import add_hook_to_video

                src = os.path.join(output_dir, clip["file"])
                if os.path.exists(src):
                    h_name = clip["file"].rsplit(".", 1)[0] + "_hook.mp4"
                    h_path = os.path.join(output_dir, h_name)
                    add_hook_to_video(
                        src, clip["hook"], h_path, position="top", duration=4.0
                    )
                    os.remove(src)
                    clip["file"] = h_name
                    clip["hook_file"] = h_name
                    logger.info("Hook banner burned into %s", h_name)
            except Exception as e:  # noqa: BLE001
                logger.warning("Hook banner skipped for clip %s: %s", clip.get("index"), e)

        processed.append(clip)

        # Notify caller that this clip is ready
        if on_clip is not None:
            try:
                on_clip(clip, idx + 1, total_clips)
            except Exception:
                pass

    processing_warnings = []

    # Plan B: stitch the selected, independently captioned moments into one montage.
    if montage:
        try:
            from .montage import stitch_montage
            montage_name = f"{base_name}_montage.mp4"
            montage_path = os.path.join(output_dir, montage_name)
            stitch_montage(
                [os.path.join(output_dir, item["file"]) for item in processed],
                montage_path,
            )
            montage_duration = probe_duration(montage_path)
            if montage_duration > float(config.HARD_MAX_CLIP_SECONDS) + 0.25:
                raise RuntimeError(
                    f"Montage duration {montage_duration:.1f}s exceeds the hard "
                    f"{config.HARD_MAX_CLIP_SECONDS:.0f}s limit; choose fewer/shorter moments."
                )
            if target_duration and abs(montage_duration - float(target_duration)) > 3.0:
                processing_warnings.append(
                    f"Montage is {montage_duration:.1f}s versus the {float(target_duration):.0f}s target; "
                    "whole transcript boundaries were prioritized."
                )
            processed.append({
                "index": len(processed) + 1,
                "file": montage_name,
                "title": f"{title} — Montage",
                "start": 0.0,
                "end": round(montage_duration, 2),
                "duration": round(montage_duration, 2),
                "is_montage": True,
                "reason": "Stitched from selected complementary moments with video/audio crossfades.",
            })
        except Exception as exc:
            processing_warnings.append(f"Montage stitching failed: {exc}")
            logger.exception("Montage stitching failed")

    # Plan C: optional narration and/or background music. For multiple normal clips,
    # apply the same requested audio treatment to the first clip only and say so.
    audio_target = None
    if montage:
        if processed and processed[-1].get("is_montage"):
            audio_target = processed[-1]
        elif voiceover_text or background_music_path:
            processing_warnings.append(
                "Voiceover/music was skipped because montage stitching failed."
            )
    elif len(processed) == 1:
        audio_target = processed[0]
    elif voiceover_text or background_music_path:
        audio_target = processed[0] if processed else None
        if len(processed) > 1:
            processing_warnings.append(
                "Voiceover/music was applied to the first generated clip only; enable montage to mix one combined output."
            )
    if audio_target and (voiceover_text or background_music_path):
        try:
            from .audio_mix import mix_audio_layers
            voiceover_path = None
            if voiceover_text and voiceover_text.strip():
                from .voiceover import generate_voiceover
                voiceover_path = os.path.join(output_dir, f"{base_name}_voiceover.wav")
                generate_voiceover(
                    voiceover_text.strip()[:5000],
                    voiceover_path,
                    engine=voiceover_engine or "auto",
                    voice=voiceover_voice,
                    gemini_key=gemini_key,
                )
            src = os.path.join(output_dir, audio_target["file"])
            mixed_name = audio_target["file"].rsplit(".", 1)[0] + "_audio.mp4"
            mixed_path = os.path.join(output_dir, mixed_name)
            mix_audio_layers(
                src, mixed_path,
                voiceover_path=voiceover_path,
                music_path=background_music_path,
                music_volume=music_volume,
            )
            if os.path.isfile(mixed_path) and os.path.getsize(mixed_path) > 0:
                os.remove(src)
                audio_target["file"] = mixed_name
                audio_target["audio_mixed"] = True
        except Exception as exc:
            processing_warnings.append(f"Voiceover/music mixing failed: {exc}")
            logger.exception("Voiceover/music mixing failed")

    # 7. Persist metadata
    try:
        video_info = probe_video_info(video_path)
    except Exception:  # noqa: BLE001
        video_info = {}
    result = {
        "title": title,
        "source": source,
        "job_id": os.path.basename(output_dir),
        "video_file": os.path.basename(video_path),
        "language": transcript.get("language"),
        "duration": transcript.get("duration"),
        "video_info": video_info,
        "aspect_ratio": aspect_key or ("16:9" if target_aspect is None else "9:16"),
        "clips": processed,
        "montage_file": next((item["file"] for item in processed if item.get("is_montage")), None),
        "processing_warnings": processing_warnings,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
    write_json(metadata_path, {**result, "transcript": transcript})

    # 8. Save clips into the Saved Library (folder per source video).
    try:
        library_path = save_to_library(output_dir, title, result["job_id"], processed)
        result["library_path"] = os.path.relpath(library_path, output_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not save to library: %s", e)

    logger.info(
        "Done in %.1fs — %d clip(s) written to %s",
        result["elapsed_seconds"],
        len(processed),
        output_dir,
    )
    return result
