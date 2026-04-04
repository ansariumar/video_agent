"""
Freeze-frame / still-frame segments for tutorial-style videos (no voice).
Extracts a frame at each timestamp, holds it for a duration, optionally burns text, then concatenates.
"""

import json
import os
import tempfile
from typing import Any, Optional, Union

from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file
from tools.overlay import _resolve_font_file


def _escape_drawtext_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _escape_drawtext_font(font_path: str) -> str:
    return font_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _position_exprs(position: str) -> tuple[str, str]:
    position_map = {
        "top-left": ("10", "10"),
        "top-center": ("(w-text_w)/2", "10"),
        "top-right": ("w-text_w-10", "10"),
        "center": ("(w-text_w)/2", "(h-text_h)/2"),
        "bottom-left": ("10", "h-text_h-10"),
        "bottom-center": ("(w-text_w)/2", "h-text_h-10"),
        "bottom-right": ("w-text_w-10", "h-text_h-10"),
    }
    return position_map.get(position.lower(), position_map["center"])


def _normalize_segments_arg(raw: Any) -> tuple[Optional[list[Any]], Optional[str]]:
    """
    Accept JSON string, a list of segment dicts, or a single segment dict.
    Returns (segments, error_message). Agents often pass a Python list instead of a string.
    """
    if raw is None:
        return None, "Error: segments_json is required."

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None, "Error: segments_json must not be empty."
        try:
            parsed: Any = json.loads(s)
        except json.JSONDecodeError as e:
            return None, f"Error: segments_json is not valid JSON: {e}"
    elif isinstance(raw, list):
        parsed = raw
    elif isinstance(raw, dict):
        parsed = [raw]
    else:
        return None, (
            "Error: segments_json must be a JSON string, a list of segment objects, "
            f"or one segment object (got {type(raw).__name__})."
        )

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None, "Error: segments_json must be a non-empty list of segment objects."
    return parsed, None


@tool
def build_freeze_frame_tutorial(
    input_path: str,
    job_id: str,
    segments_json: Union[str, list, dict],
    width: int = 1920,
    height: int = 1080,
    output_fps: int = 30,
    font_size: int = 48,
    font_color: str = "white",
    font_file: Optional[str] = None,
    position: str = "center",
) -> str:
    """
    Build a silent tutorial-style video from multiple freeze-frame segments.
    For each segment: grabs a still frame from the source video at a timestamp, holds it for a
    fixed duration (like a slide), optionally draws instruction text on top, then concatenates
    all segments in order. Use this for step-by-step screen tutorials without narration.

    Each segment object in the JSON array supports:
    - timestamp (required): time to grab the frame from the source video (HH:MM:SS or seconds).
    - duration (required): how long to show that still, in seconds (float).
    - text (optional): instruction text; omit or use empty string for a plain freeze with no caption.

    Args:
        input_path: Absolute path to the source video to sample frames from.
        job_id: The current job ID for naming the output file.
        segments_json: Segment list — **prefer a plain list of dicts** (what tool APIs expect), e.g.
            [{"timestamp": "00:00:15", "duration": 5, "text": "Click here"}, ...].
            A JSON string of that array, or a single segment dict, also works.
        width: Output frame width in pixels. Default: 1920.
        height: Output frame height in pixels. Default: 1080.
        output_fps: Output frame rate for each still clip (helps concat compatibility). Default: 30.
        font_size: Caption font size in pixels. Default: 48.
        font_color: Caption color (name or hex). Default: white.
        font_file: Path to a .ttf/.otf font; if omitted, a common system font is used when needed.
        position: Text position for captions: top-left, top-center, top-right, center,
            bottom-left, bottom-center, bottom-right. Default: center.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    segments, norm_err = _normalize_segments_arg(segments_json)
    if norm_err or not segments:
        return norm_err or "Error: no segments to process."

    needs_text = any(
        str(s.get("text", "") or "").strip()
        for s in segments
        if isinstance(s, dict)
    )
    font_path: Optional[str] = None
    if needs_text:
        font_path, font_err = _resolve_font_file(font_file)
        if font_err:
            return f"Error: {font_err}"

    x_expr, y_expr = _position_exprs(position)
    font_clause = ""
    if font_path:
        font_clause = f"fontfile='{_escape_drawtext_font(font_path)}':"

    segment_paths: list[str] = []

    with tempfile.TemporaryDirectory(prefix=f"{job_id}_freeze_") as tmp:
        for i, raw in enumerate(segments):
            if not isinstance(raw, dict):
                return f"Error: segment {i} must be an object, got {type(raw).__name__}."

            ts = raw.get("timestamp")
            if ts is None or (isinstance(ts, str) and not str(ts).strip()):
                return f"Error: segment {i} missing non-empty 'timestamp'."
            timestamp = str(ts).strip()

            dur = raw.get("duration")
            if dur is None:
                return f"Error: segment {i} missing 'duration'."
            try:
                duration = float(dur)
            except (TypeError, ValueError):
                return f"Error: segment {i} 'duration' must be a number."
            if duration <= 0:
                return f"Error: segment {i} 'duration' must be positive."

            text = raw.get("text", "")
            if text is None:
                text = ""
            text = str(text)

            png_path = os.path.join(tmp, f"frame_{i:03d}.png")
            seg_path = os.path.join(tmp, f"seg_{i:03d}.mp4")

            extract = run_ffmpeg([
                "-ss", timestamp,
                "-i", input_path,
                "-frames:v", "1",
                "-q:v", "2",
                png_path,
            ])
            if not extract.success:
                return (
                    f"Freeze segment {i}: failed to extract frame at {timestamp}. "
                    f"{extract.error_message}"
                )

            scale_pad = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            )

            if text.strip():
                safe_text = _escape_drawtext_text(text)
                vf = (
                    f"{scale_pad},"
                    f"drawtext={font_clause}"
                    f"text='{safe_text}'"
                    f":fontsize={font_size}"
                    f":fontcolor={font_color}"
                    f":x={x_expr}:y={y_expr}"
                    f":box=1:boxcolor=black@0.5:boxborderw=10"
                )
            else:
                vf = scale_pad

            enc = run_ffmpeg([
                "-loop", "1",
                "-i", png_path,
                "-t", str(duration),
                "-vf", vf,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(output_fps),
                "-an",
                seg_path,
            ])
            if not enc.success:
                return (
                    f"Freeze segment {i}: failed to build still clip ({duration}s). "
                    f"{enc.error_message}"
                )

            segment_paths.append(seg_path)

        concat_list = os.path.join(tmp, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in segment_paths:
                # FFmpeg concat demuxer: forward slashes are safest on Windows
                safe = p.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        output_path = get_output_path(job_id, suffix="_freeze_tutorial")

        merged = run_ffmpeg([
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ])

    if merged.success:
        return (
            f"Freeze-frame tutorial created ({len(segment_paths)} segments, {width}x{height}). "
            f"Output saved to: {output_path}"
        )
    return f"Concatenation failed: {merged.error_message}"
