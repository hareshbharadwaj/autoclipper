"""Inspect a source *before* deciding how to analyse it.

The pipeline used to assume every video was a speech video: it always
extracted audio, always transcribed, and treated an empty transcript as an
empty video. That is wrong in two common cases:

* **No audio stream at all** (screen recordings, silent exports, GIF-derived
  MP4s). Transcription is impossible; the footage still contains usable
  moments.
* **Audio but no speech** (music videos, gameplay with music, ambience). A
  Whisper transcript is legitimately empty, but the video is not empty.

This module separates "is there audio?" from "is there speech?" and returns an
explicit ``analysis_mode`` the pipeline branches on:

``speech``
    Timestamped transcription, sentence boundaries, topic relevance, plus
    visual evidence where frames are available.
``audio_no_speech``
    Skip speech reasoning. Keep the audio track in the clip, and drive clip
    boundaries from visual evidence.
``visual_only``
    No audio stream (or a silent one): skip transcription entirely and drive
    boundaries from sampled frames and scene changes.

Limits are stated honestly: the visual path can describe visible actions,
objects and on-screen text, and can find scene boundaries. It cannot recover
dialogue that was never recorded, and it never invents captions or narration.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass, field

from .utils import logger

MODE_SPEECH = "speech"
MODE_AUDIO_NO_SPEECH = "audio_no_speech"
MODE_VISUAL_ONLY = "visual_only"
ANALYSIS_MODES = (MODE_SPEECH, MODE_AUDIO_NO_SPEECH, MODE_VISUAL_ONLY)

# A track quieter than this is treated as silent rather than as content.
SILENCE_MAX_VOLUME_DB = -50.0

# A transcript shorter than this cannot support topic analysis, so it is
# treated as "no speech" even though the track had energy.
MIN_SPEECH_CHARACTERS = 20

_VOLUME_RE = re.compile(r"(max_volume|mean_volume):\s*(-?[\d.]+)\s*dB")


@dataclass(frozen=True)
class SourceProfile:
    """What the pipeline knows about a source before analysing it."""

    analysis_mode: str
    duration: float = 0.0
    has_audio: bool = False
    has_speech: bool = False
    audio_max_volume_db: float | None = None
    audio_mean_volume_db: float | None = None
    speech_characters: int = 0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_transcript(self) -> bool:
        """True when speech evidence should drive clip selection."""
        return self.analysis_mode == MODE_SPEECH

    @property
    def uses_visual_analysis(self) -> bool:
        """True when frames must carry the analysis (both non-speech modes)."""
        return self.analysis_mode in (MODE_AUDIO_NO_SPEECH, MODE_VISUAL_ONLY)

    @property
    def preserve_audio(self) -> bool:
        """The source audio track is kept whenever the source has one.

        This is deliberately *not* "has speech": a present-but-silent track is
        still the source's own audio and must survive rendering (the clip keeps
        an audio stream instead of silently becoming a video-only file).
        """
        return bool(self.has_audio)

    def summary(self) -> str:
        return {
            MODE_SPEECH: "Speech detected; using timestamped transcription with visual evidence.",
            MODE_AUDIO_NO_SPEECH: (
                "Audio track present but no usable speech; analysing the video visually "
                "and preserving the original audio."
            ),
            MODE_VISUAL_ONLY: (
                "No usable audio stream; skipping transcription and analysing the video "
                "visually."
            ),
        }.get(self.analysis_mode, "Source analysis mode unknown.")

    def to_dict(self) -> dict:
        return {
            "analysis_mode": self.analysis_mode,
            "duration": round(float(self.duration or 0.0), 2),
            "has_audio": self.has_audio,
            "has_speech": self.has_speech,
            "audio_max_volume_db": self.audio_max_volume_db,
            "audio_mean_volume_db": self.audio_mean_volume_db,
            "speech_characters": self.speech_characters,
            "preserve_audio": self.preserve_audio,
            "uses_transcript": self.uses_transcript,
            "uses_visual_analysis": self.uses_visual_analysis,
            "reasons": list(self.reasons),
            "summary": self.summary(),
        }


def measure_audio_activity(video_path: str, timeout: float = 90.0) -> dict:
    """Measure the loudest/mean level of the first audio stream.

    Returns ``{"max_volume_db": float|None, "mean_volume_db": float|None,
    "measured": bool}``. A failure to measure is reported, never raised: an
    unmeasurable track must not stop the job.
    """
    if not video_path or not os.path.isfile(video_path):
        return {"max_volume_db": None, "mean_volume_db": None, "measured": False}
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats",
                "-i", video_path, "-map", "0:a:0",
                "-af", "volumedetect", "-f", "null",
                os.devnull,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Audio activity probe failed (%s); continuing.", type(exc).__name__)
        return {"max_volume_db": None, "mean_volume_db": None, "measured": False}

    values = {name: float(value) for name, value in _VOLUME_RE.findall(result.stderr or "")}
    measured = bool(values)
    return {
        "max_volume_db": values.get("max_volume"),
        "mean_volume_db": values.get("mean_volume"),
        "measured": measured,
    }


def classify_source(
    video_info: dict | None = None,
    *,
    audio_activity: dict | None = None,
    transcript: dict | None = None,
    assume_speech_when_unmeasured: bool = True,
) -> SourceProfile:
    """Decide which analysis path a source needs.

    ``transcript`` is optional: when it is ``None`` the caller intends to
    transcribe, so speech is *assumed* for a loud audio track. Pass the
    transcribed result afterwards (see ``pipeline``) to downgrade to
    ``audio_no_speech`` when it comes back empty.

    The downgrade requires *positive* evidence. An empty transcript is only
    treated as "no speech" when the audio level could actually be measured and
    showed energy — otherwise a working transcript path would be discarded on
    the strength of evidence we never gathered. Set
    ``assume_speech_when_unmeasured=False`` to always downgrade.
    """
    info = video_info or {}
    try:
        duration = float(info.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration):
        duration = 0.0

    has_audio = bool(info.get("has_audio"))
    activity = audio_activity or {}
    max_db = activity.get("max_volume_db")
    mean_db = activity.get("mean_volume_db")
    reasons: list[str] = []

    if not has_audio:
        reasons.append("The source has no audio stream.")
        return SourceProfile(
            analysis_mode=MODE_VISUAL_ONLY, duration=duration, has_audio=False,
            has_speech=False, audio_max_volume_db=None, audio_mean_volume_db=None,
            reasons=tuple(reasons),
        )

    if max_db is not None and max_db <= SILENCE_MAX_VOLUME_DB:
        reasons.append(
            f"The source audio track is silent (peak {max_db:.1f} dB)."
        )
        return SourceProfile(
            analysis_mode=MODE_VISUAL_ONLY, duration=duration, has_audio=True,
            has_speech=False, audio_max_volume_db=max_db, audio_mean_volume_db=mean_db,
            reasons=tuple(reasons),
        )

    if transcript is None:
        reasons.append("Audio detected; speech is assumed until transcription returns text.")
        return SourceProfile(
            analysis_mode=MODE_SPEECH, duration=duration, has_audio=True, has_speech=True,
            audio_max_volume_db=max_db, audio_mean_volume_db=mean_db,
            reasons=tuple(reasons),
        )

    text = " ".join(
        str(part) for part in (
            transcript.get("text"),
            *[segment.get("text") for segment in transcript.get("segments") or []],
        ) if part
    ).strip()
    characters = len(re.sub(r"\s+", " ", text))
    transcript_duration = 0.0
    try:
        transcript_duration = float(transcript.get("duration") or 0.0)
    except (TypeError, ValueError):
        transcript_duration = 0.0

    measured = bool(activity.get("measured"))
    if characters < MIN_SPEECH_CHARACTERS and not measured and assume_speech_when_unmeasured:
        reasons.append(
            "Transcription produced little text and the audio level could not be "
            "measured; keeping the transcript path rather than switching modes on "
            "unverified evidence."
        )
        return SourceProfile(
            analysis_mode=MODE_SPEECH, duration=duration or transcript_duration,
            has_audio=True, has_speech=True, audio_max_volume_db=max_db,
            audio_mean_volume_db=mean_db, speech_characters=characters,
            reasons=tuple(reasons),
        )

    if characters < MIN_SPEECH_CHARACTERS:
        reasons.append(
            "The audio track has energy, but transcription produced no usable "
            f"speech ({characters} characters)."
        )
        return SourceProfile(
            analysis_mode=MODE_AUDIO_NO_SPEECH, duration=duration or transcript_duration,
            has_audio=True, has_speech=False, audio_max_volume_db=max_db,
            audio_mean_volume_db=mean_db, speech_characters=characters,
            reasons=tuple(reasons),
        )

    reasons.append(f"Transcription produced {characters} characters of speech.")
    return SourceProfile(
        analysis_mode=MODE_SPEECH, duration=duration or transcript_duration, has_audio=True,
        has_speech=True, audio_max_volume_db=max_db, audio_mean_volume_db=mean_db,
        speech_characters=characters, reasons=tuple(reasons),
    )
