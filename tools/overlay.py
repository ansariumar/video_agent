"""
Overlay tools — text, watermark, subtitles, thumbnail extraction.
"""

import os
from typing import Optional

from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


def _resolve_font_file(font_file: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Resolve a usable font path; fallback to common system fonts on Windows/macOS/Linux."""
    if font_file:
        font_path = os.path.abspath(font_file)
        if not os.path.exists(font_path):
            return None, f"Font file not found: '{font_path}'"
        return font_path, None

    # Auto-pick a common system font if available
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),  # Windows
        "/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, None

    return None, None


@tool
def add_text_overlay(
    input_path: str,
    text: str,
    job_id: str,
    font_file: Optional[str] = None,
    position: str = "bottom-center",
    font_size: int = 48,
    font_color: str = "white",
    start_time: float = 0,
    end_time: float = -1,
) -> str:
    """
    Burn text onto a video. The text is permanently embedded in the video frames.
    Use this when the user wants to add a title, caption, label, or any text to the video.

    Args:
        input_path: Absolute path to the input video file.
        text: The text string to display on the video.
        job_id: The current job ID for naming the output file.
        font_file: Path to a .ttf/.otf font. If omitted, the function will try to locate a common
               system font (Arial/DejaVuSans) before letting FFmpeg pick a default.
        position: Where to place the text. Options: 'top-left', 'top-center', 'top-right',
              'center', 'bottom-left', 'bottom-center', 'bottom-right'. Default: 'bottom-center'.
        font_size: Font size in pixels. Default: 48.
        font_color: Text color name or hex (e.g., 'white', 'yellow', '#FF0000'). Default: 'white'.
        start_time: Time in seconds when text appears. Default: 0 (from start).
        end_time: Time in seconds when text disappears. Default: -1 (until end).
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    font_path, font_err = _resolve_font_file(font_file)
    if font_err:
        return f"Error: {font_err}"

    position_map = {
        "top-left":      ("10", "10"),
        "top-center":    ("(w-text_w)/2", "10"),
        "top-right":     ("w-text_w-10", "10"),
        "center":        ("(w-text_w)/2", "(h-text_h)/2"),
        "bottom-left":   ("10", "h-text_h-10"),
        "bottom-center": ("(w-text_w)/2", "h-text_h-10"),
        "bottom-right":  ("w-text_w-10", "h-text_h-10"),
    }

    pos = position_map.get(position.lower(), position_map["bottom-center"])
    x_expr, y_expr = pos

    # Escape special chars for FFmpeg drawtext
    safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")

    font_clause = ""
    if font_path:
        safe_font = font_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        font_clause = f"fontfile='{safe_font}':"

    time_filter = ""
    if start_time > 0 or end_time > 0:
        t_start = f":enable='between(t,{start_time},{end_time})'" if end_time > 0 else f":enable='gte(t,{start_time})'"
        time_filter = t_start

    vf = (
        f"drawtext={font_clause}"
        f"text='{safe_text}'"
        f":fontsize={font_size}"
        f":fontcolor={font_color}"
        f":x={x_expr}:y={y_expr}"
        f":box=1:boxcolor=black@0.4:boxborderw=5"
        f"{time_filter}"
    )

    output_path = get_output_path(job_id, suffix="_text")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Text overlay added. Output saved to: {output_path}"
    return f"Text overlay failed: {result.error_message}"


@tool
def add_image_watermark(
    input_path: str,
    watermark_path: str,
    job_id: str,
    position: str = "bottom-right",
    opacity: float = 0.7,
    scale: float = 0.15,
) -> str:
    """
    Add an image watermark (logo) to a video.
    Use this when the user wants to brand a video, add a logo, or watermark it.

    Args:
        input_path: Absolute path to the input video file.
        watermark_path: Absolute path to the watermark image (PNG recommended for transparency).
        job_id: The current job ID for naming the output file.
        position: Watermark position. Options: 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'.
        opacity: Watermark opacity from 0.0 (invisible) to 1.0 (fully opaque). Default: 0.7.
        scale: Watermark size as a fraction of video width (0.0-1.0). Default: 0.15 (15% of width).
    """
    for path, label in [(input_path, "input video"), (watermark_path, "watermark image")]:
        valid, err = validate_input_file(path)
        if not valid:
            return f"Error with {label}: {err}"

    position_map = {
        "top-left":     "10:10",
        "top-right":    "W-w-10:10",
        "bottom-left":  "10:H-h-10",
        "bottom-right": "W-w-10:H-h-10",
        "center":       "(W-w)/2:(H-h)/2",
    }
    overlay_pos = position_map.get(position.lower(), "W-w-10:H-h-10")

    output_path = get_output_path(job_id, suffix="_watermarked")

    result = run_ffmpeg([
        "-i", input_path,
        "-i", watermark_path,
        "-filter_complex",
        f"[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];"
        f"[0:v][wm]overlay={overlay_pos}[out]",
        "-map", "[out]",
        "-map", "0:a?",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Watermark added at {position}. Output saved to: {output_path}"
    return f"Watermark failed: {result.error_message}"


@tool
def add_subtitles(input_path: str, srt_path: str, job_id: str) -> str:
    """
    Burn subtitles from an SRT file directly into the video frames.
    Use this when the user wants to add subtitles, captions, or burn in text from an SRT file.

    Args:
        input_path: Absolute path to the input video file.
        srt_path: Absolute path to the SRT subtitle file.
        job_id: The current job ID for naming the output file.
    """
    for path, label in [(input_path, "input video"), (srt_path, "SRT subtitle file")]:
        valid, err = validate_input_file(path)
        if not valid:
            return f"Error with {label}: {err}"

    output_path = get_output_path(job_id, suffix="_subtitled")

    # Escape the srt path for FFmpeg subtitles filter
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"subtitles='{safe_srt}'",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Subtitles added. Output saved to: {output_path}"
    return f"Subtitle burning failed: {result.error_message}"


@tool
def generate_thumbnail(input_path: str, timestamp: str, job_id: str) -> str:
    """
    Extract a single frame from a video and save it as a JPEG image (thumbnail).
    Use this when the user wants a thumbnail, preview image, or screenshot from a video.

    Args:
        input_path: Absolute path to the input video file.
        timestamp: Time in the video to extract the frame from (HH:MM:SS or seconds).
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_thumbnail", extension=".jpg")

    result = run_ffmpeg([
        "-i", input_path,
        "-ss", timestamp,
        "-vframes", "1",
        "-q:v", "2",        # High quality JPEG
        output_path
    ])

    if result.success:
        return f"Thumbnail saved to: {output_path}"
    return f"Thumbnail generation failed: {result.error_message}"
