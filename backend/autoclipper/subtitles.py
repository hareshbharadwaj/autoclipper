"""Burned-in subtitles generated from word-level timestamps.

Builds an SRT from the words that fall inside each clip, then burns it into the
final clip with ffmpeg's subtitles filter (libass). Word-by-word timing gives
the classic short-form caption look.
"""
import os
import re
import subprocess
import textwrap

from . import config
from .utils import logger, probe_video_info, run_command


REFERENCE_SHORT_SIDE = 1080
REFERENCE_FRAME_HEIGHT = 1920


def subtitle_layout(width=1080, height=1920, font_size=None, outline_width=4, margin_v=None):
    """Scale subtitle styling from the 1080x1920 design canvas to a video frame.

    `SUBTITLE_FONT_SIZE`, `font_size`, `outline_width`, and `margin_v` are
    reference-canvas values, not literal pixels for every output resolution.
    The short side drives text size so both portrait and landscape footage stay
    readable without letting captions dominate low-resolution frames.
    """
    try:
        width = max(2, int(width or 1080))
        height = max(2, int(height or 1920))
    except (TypeError, ValueError):
        width, height = 1080, 1920

    short_side = min(width, height)
    scale = short_side / float(REFERENCE_SHORT_SIDE)
    requested_size = config.SUBTITLE_FONT_SIZE if font_size is None else font_size
    try:
        requested_size = max(1.0, float(requested_size))
    except (TypeError, ValueError):
        requested_size = float(config.SUBTITLE_FONT_SIZE)
    max_size = max(12, int(round(short_side * 0.085)))
    size = max(12, min(int(round(requested_size * scale)), max_size, int(round(width * 0.12))))

    try:
        requested_outline = max(1.0, float(outline_width))
    except (TypeError, ValueError):
        requested_outline = 4.0
    outline = max(1, min(int(round(requested_outline * scale)), int(round(size * 0.08)), 6))

    requested_margin = config.SUBTITLE_MARGIN_V if margin_v is None else margin_v
    try:
        requested_margin = max(0.0, float(requested_margin))
    except (TypeError, ValueError):
        requested_margin = float(config.SUBTITLE_MARGIN_V)
    scaled_margin = int(round(requested_margin * height / float(REFERENCE_FRAME_HEIGHT)))
    vertical_safe_margin = int(round(height * 0.05))
    vertical_max_margin = int(round(height * 0.15))
    margin = max(vertical_safe_margin, min(scaled_margin, vertical_max_margin))

    side_margin = max(2, int(round(width * 0.06)))
    available_width = max(size * 6, width - (2 * side_margin))
    max_chars = max(10, min(32, int(available_width / max(1.0, size * 0.58))))
    return {
        "width": width,
        "height": height,
        "scale": scale,
        "font_size": size,
        "outline_width": outline,
        "margin_v": margin,
        "margin_l": side_margin,
        "margin_r": side_margin,
        "max_text_width": available_width,
        "max_chars": max_chars,
    }


def _wrap_caption_text(text, max_chars, line_break="\n"):
    """Wrap caption text to the safe width, breaking unusually long words too."""
    width = max(1, int(max_chars or 20))
    lines = textwrap.wrap(
        " ".join((text or "").split()),
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return line_break.join(lines)


def _wrap_words(words, max_chars):
    """Wrap ASS cue words, preserving the order used for karaoke timing."""
    width = max(1, int(max_chars or 20))
    lines, line, line_size = [], [], 0
    for word in words:
        chunks = textwrap.wrap(str(word), width=width, break_long_words=True, break_on_hyphens=False) or [str(word)]
        for chunk in chunks:
            extra = len(chunk) + (1 if line else 0)
            if line and line_size + extra > width:
                lines.append(line)
                line, line_size = [], 0
            line.append(chunk)
            line_size += len(chunk) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(line)
    return lines


def build_srt(words, start, end, max_chars=20, max_duration=2.0):
    """Return an SRT string for words inside [start, end], relative to clip start.

    Word timings are clipped to the requested output interval, preventing subtitle
    cues from extending beyond the rendered clip's duration.
    """
    try:
        clip_start = float(start)
        clip_end = float(end)
    except (TypeError, ValueError):
        return ""
    if clip_end <= clip_start:
        return ""
    clip_duration = clip_end - clip_start

    selected = []
    for word in words or []:
        text = (word.get("word") or "").strip()
        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if text and word_end > word_start and word_end > clip_start and word_start < clip_end:
            selected.append({"word": text, "start": word_start, "end": word_end})
    if not selected:
        return ""

    cues = []
    block = []
    block_start = None

    for word in selected:
        s = max(0.0, min(clip_duration, word["start"] - clip_start))
        e = max(0.0, min(clip_duration, word["end"] - clip_start))
        if e <= s:
            continue
        word["start_rel"] = s
        word["end_rel"] = e

        if not block:
            block = [word]
            block_start = s
        else:
            current_text_len = sum(len(item["word"]) + 1 for item in block)
            duration = e - block_start
            if current_text_len + len(word["word"]) > max_chars or duration > max_duration:
                block_end = block[-1]["end_rel"]
                text = " ".join(item["word"] for item in block)
                cues.append((block_start, block_end, text))
                block = [word]
                block_start = s
            else:
                block.append(word)

    if block:
        block_end = block[-1]["end_rel"]
        text = " ".join(item["word"] for item in block)
        cues.append((block_start, block_end, text))

    def _fmt(t):
        total_ms = max(0, int(round(float(t) * 1000)))
        h, rem = divmod(total_ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for idx, (cue_start, cue_end, cue_text) in enumerate(cues, start=1):
        wrapped_text = _wrap_caption_text(cue_text, max_chars=max_chars)
        lines.append(f"{idx}\n{_fmt(cue_start)} --> {_fmt(cue_end)}\n{wrapped_text}\n")
    return "\n".join(lines)


def _srt_timestamp(t) -> str:
    total_ms = max(0, int(round(float(t) * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def cues_to_srt(cues: list) -> str:
    """Serialize edited caption cues ({"start","end","text"} in seconds) to SRT.

    Used by the subtitle editor endpoint so an edited clip is saved with an
    SRT sidecar that matches its burned-in captions — never the stale
    pre-edit SRT. Invalid or empty cues are skipped; returns "" when no cue
    is usable.
    """
    blocks = []
    for cue in cues or []:
        if not isinstance(cue, dict):
            continue
        text = str(cue.get("text") or "").strip()
        try:
            start = float(cue.get("start"))
            end = float(cue.get("end"))
        except (TypeError, ValueError):
            continue
        if not text or start < 0 or end <= start:
            continue
        blocks.append((start, end, text.replace("\r", " ").replace("\n", " ")))

    if not blocks:
        return ""
    lines = []
    for idx, (start, end, text) in enumerate(blocks, start=1):
        lines.append(f"{idx}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}\n")
    return "\n".join(lines)


_SRT_TIME_RE = re.compile(
    r"^(\d+):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2}),(\d{3})$"
)


def parse_srt(text: str) -> list:
    """Parse SRT content into [{"start","end","text"}] cue dicts (seconds)."""
    cues = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_index = next(
            (i for i, line in enumerate(lines) if _SRT_TIME_RE.match(line)), None
        )
        if time_index is None:
            continue
        match = _SRT_TIME_RE.match(lines[time_index])
        h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
        start = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000.0
        end = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000.0
        body = " ".join(lines[time_index + 1:]).strip()
        if body and end > start:
            cues.append({"start": start, "end": end, "text": body})
    return cues


def rebase_srt_file(srt_path: str, plan: dict, clip_source_start: float = 0.0,
                    out_path: str = None) -> str:
    """Rewrite an SRT sidecar onto a clip's edit timeline.

    Clip-relative cue times are mapped through the edit plan's source->edit
    segments (see ``edit_planner.rebase_cues``) so captions stay aligned when a
    narration card is prepended or passages are re-ordered. Cues that would
    cross a planned segment boundary are clipped to it.
    """
    from .edit_planner import rebase_cues

    if not srt_path or not os.path.isfile(srt_path):
        return srt_path
    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        cues = parse_srt(handle.read())
    rebased = rebase_cues(cues, plan, clip_source_start=clip_source_start)
    if not rebased:
        return srt_path
    destination = out_path or srt_path
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(cues_to_srt(rebased))
    return destination


def _escape_filter_path(path: str) -> str:
    """Return a filtergraph-safe reference to a subtitle file.

    We run ffmpeg from the file's own directory and use a relative basename,
    which sidesteps Windows drive-letter escaping. Spaces/special chars are
    wrapped in single quotes.
    """
    name = os.path.basename(path)
    if any(ch in name for ch in " ':,[]=;"):
        return "'" + name + "'"
    return name


def burn_subtitles(video_path: str, srt_path: str, out_path: str, layout=None) -> str:
    """Burn an SRT using resolution-aware font, outline, and safe margins."""
    if layout is None:
        info = probe_video_info(video_path)
        layout = subtitle_layout(info.get("width"), info.get("height"))
    style = (
        f"FontName={_map_font(config.SUBTITLE_FONT)},"
        f"FontSize={layout['font_size']},"
        f"PrimaryColour={config.SUBTITLE_PRIMARY},"
        f"OutlineColour={config.SUBTITLE_OUTLINE},"
        f"BackColour={config.SUBTITLE_BACKGROUND},"
        f"Bold=1,Outline={layout['outline_width']},Shadow=0,Alignment=2,WrapStyle=2,"
        f"MarginL={layout['margin_l']},MarginR={layout['margin_r']},MarginV={layout['margin_v']}"
    )
    vf = f"subtitles={_escape_filter_path(srt_path)}:force_style='{style}'"
    cwd = os.path.dirname(os.path.abspath(video_path))
    run_command(
        [
            "ffmpeg", "-y", "-i", os.path.basename(video_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart",
            os.path.basename(out_path),
        ],
        cwd=cwd,
    )
    return out_path


def add_subtitles(clips, transcript, output_dir, burn=True, preserve_source=False):
    """For each clip, generate and (optionally) burn subtitles.

    Updates each clip dict with 'srt' (path) and, when burned, replaces 'file'
    with the subtitled version and records 'subtitled_file'. When ``preserve_source``
    is true, the pre-subtitle render remains available as a safe fallback.
    """
    words = transcript.get("words", [])
    for clip in clips:
        cs, ce = clip.get("start"), clip.get("end")
        if cs is None or ce is None:
            continue
        src = os.path.join(output_dir, clip.get("file", ""))
        info = probe_video_info(src) if os.path.isfile(src) else {}
        layout = subtitle_layout(info.get("width"), info.get("height"))
        srt = build_srt(words, cs, ce, max_chars=layout["max_chars"])
        if not srt:
            warning = "No transcript words overlap this clip; captions were not generated."
            clip["subtitle_warning"] = warning
            logger.warning("%s Clip index: %s", warning, clip.get("index"))
            continue

        srt_name = clip["file"].rsplit(".", 1)[0] + ".srt"
        srt_path = os.path.join(output_dir, srt_name)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
        clip["srt"] = srt_name

        if not burn:
            continue

        if not os.path.isfile(src):
            clip["subtitle_warning"] = "Clip file was missing; subtitles could not be burned."
            continue
        sub_name = clip["file"].rsplit(".", 1)[0] + "_sub.mp4"
        sub_path = os.path.join(output_dir, sub_name)
        try:
            burn_subtitles(src, srt_path, sub_path, layout=layout)
            if not os.path.isfile(sub_path) or os.path.getsize(sub_path) == 0:
                raise RuntimeError("FFmpeg returned without creating a non-empty subtitled video.")
            clip["subtitled_file"] = sub_name
            clip["file"] = sub_name
            clip.pop("subtitle_warning", None)
            if not preserve_source:
                try:
                    os.remove(src)
                except OSError:
                    pass
        except Exception as e:  # noqa: BLE001
            try:
                if os.path.exists(sub_path):
                    os.remove(sub_path)
            except OSError:
                pass
            clip["subtitle_warning"] = f"Captions were generated as SRT but could not be burned in: {e}"
            logger.error("Failed to burn subtitles for clip %s (%s).", clip.get("index"), type(e).__name__)
    return clips


# --- Editable / styled subtitle re-burn (dashboard "Auto Subtitle") ---------
def hex_to_ass(hex_color: str) -> str:
    """Convert '#RRGGBB' to ASS colour '&H00BBGGRR&' (alpha 00 = opaque)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H00{b}{g}{r}&"
    return "&H00FFFFFF&"


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    cc = cs % 100
    return f"{h:02d}:{m:02d}:{s:02d}.{cc:02d}"


def _map_font(name: str) -> str:
    """Map a font name to one actually available in the container so the
    burned result matches what the browser preview shows."""
    n = (name or "").strip().lower()
    aliases = {
        "arial": "Liberation Sans",
        "arial black": "Liberation Sans:bold",
        "helvetica": "Liberation Sans",
        "liberation sans": "Liberation Sans",
        "times new roman": "Liberation Serif",
        "times": "Liberation Serif",
        "courier new": "Liberation Mono",
        "verdana": "DejaVu Sans",
        "georgia": "DejaVu Serif",
        "impact": "Liberation Sans",
    }
    return aliases.get(n, name or config.SUBTITLE_FONT)


def build_ass(cues: list, style: dict, video_width=1080, video_height=1920) -> str:
    """Build an ASS subtitle string from edited cues + styling options.

    `cues` is a list of {"start", "end", "text"} (seconds, relative to clip).
    `style` keys: font, font_size, text_color, outline_color, border_width,
    box_color, box_opacity (0-100), highlight_color, highlight (bool), box (bool),
    margin_v, position ("top" | "middle" | "bottom"), animation ("pop"|"glow"|"karaoke"|"none").
    When highlight is on, each cue uses karaoke timing so the spoken word fills
    with `highlight_color` (the classic short-form caption effect).
    Maps to OpenShorts animation styles: karaoke/glow → highlight, none → no highlight, pop → highlight.
    """
    style = style or {}
    font = _map_font(style.get("font", config.SUBTITLE_FONT))
    layout = subtitle_layout(
        video_width,
        video_height,
        font_size=style.get("font_size", config.SUBTITLE_FONT_SIZE),
        outline_width=style.get("border_width", 2),
        margin_v=style.get("margin_v", config.SUBTITLE_MARGIN_V),
    )
    # Keep the editor's existing 0.85 visual scale, then derive wrapping from
    # the actual output frame and scaled font rather than a fixed 1080x1920 canvas.
    size = max(10, int(round(layout["font_size"] * 0.85)))
    primary = hex_to_ass(style.get("text_color", "#FFFFFF"))
    outline = hex_to_ass(style.get("outline_color", "#000000"))
    highlight = hex_to_ass(style.get("highlight_color", "#FFFF00"))
    margin_v = layout["margin_v"]
    max_chars = max(10, min(32, int(layout["max_text_width"] / max(1.0, size * 0.58))))
    position = (style.get("position", "bottom") or "bottom").lower()
    use_box = bool(style.get("box", False))
    animation = (style.get("animation", "karaoke") or "karaoke").lower()
    use_highlight = bool(style.get("highlight", animation in ("karaoke", "glow", "pop")))

    box_opacity = int(style.get("box_opacity", 50))
    alpha = max(0, min(255, int(round(box_opacity / 100 * 255))))
    box_color = hex_to_ass(style.get("box_color", "#000000"))
    back = f"&H{alpha:02X}{box_color[2:8]}"  # &H<AABBGGRR>

    # OpenShorts-style: BorderStyle 3 for box, BorderStyle 1 for outline with variable width
    border_style = 3 if use_box else 1
    outline_width = 1 if use_box else layout["outline_width"]
    shadow = 0
    # Alignment: 1=top-left, 2=bottom-center, 5=middle-center, etc. We use the
    # center variants: top=6, middle=10, bottom=2 (horizontal center).
    alignment = {"top": 6, "middle": 10, "bottom": 2}.get(position, 2)
    # Karaoke reveals SecondaryColour over PrimaryColour as each word plays.
    secondary = highlight if use_highlight else primary

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {layout['width']}",
        f"PlayResY: {layout['height']}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV",
        f"Style: Default,{font},{size},{primary},{secondary},{outline},{back},"
        f"-1,0,0,0,100,100,0,0,{border_style},{outline_width},{shadow},{alignment},"
        f"{layout['margin_l']},{layout['margin_r']},{margin_v}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    events = []
    for cue in cues or []:
        text = (cue.get("text") or "").strip()
        if not text:
            continue
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", start + 1.0))
        words = text.split()
        wrapped_words = _wrap_words(words, max_chars)
        wrapped_count = sum(len(line_words) for line_words in wrapped_words)
        if use_highlight and wrapped_count > 0:
            dur = max(1, int(round((end - start) * 100 / wrapped_count)))
            line = "\\N".join(
                "".join(f"{{\\k{dur}}}{word} " for word in line_words).strip()
                for line_words in wrapped_words
            )
        else:
            line = "\\N".join(" ".join(line_words) for line_words in wrapped_words)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{line}"
        )

    return "\n".join(header + events) + "\n"


def burn_ass(video_path: str, ass_path: str, out_path: str) -> str:
    """Burn an ASS subtitle file into a video, returning the output path."""
    cwd = os.path.dirname(os.path.abspath(video_path))
    vf = f"subtitles={_escape_filter_path(ass_path)}"
    run_command(
        [
            "ffmpeg", "-y", "-i", os.path.basename(video_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart",
            os.path.basename(out_path),
        ],
        cwd=cwd,
    )
    return out_path
