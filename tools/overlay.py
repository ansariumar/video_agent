"""
Overlay tools — text, watermark, subtitles, thumbnail, freeze-frame annotations.
"""

import os
import json
import tempfile
import shutil
from typing import Optional
from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg, run_ffprobe
from utils.file_utils import get_output_path, validate_input_file, OUTPUTS_DIR


# ─── Shared font resolver ─────────────────────────────────────────────────────

def _resolve_font_file(font_file: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Locate a usable font file for FFmpeg drawtext.

    Returns:
        (font_path, error_message)
        font_path is None if no font found but FFmpeg should still try its default.
        error_message is non-None only if an explicitly given path doesn't exist.
    """
    if font_file:
        if os.path.exists(font_file):
            return font_file, None
        return None, f"Font file not found: '{font_file}'"

    # Search common system font locations (Linux → macOS → Windows)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "/System/Library/Fonts/Helvetica.ttc",     # macOS
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",              # Windows
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path, None

    # No font found — return None without error so FFmpeg uses its built-in default
    return None, None


def _make_drawtext_filter(
    text: str,
    position: str = "bottom-center",
    font_size: int = 48,
    font_color: str = "white",
    font_path: Optional[str] = None,
    start_time: float = 0,
    end_time: float = -1,
) -> str:
    position_map = {
        "top-left":      ("10",             "10"),
        "top-center":    ("(w-text_w)/2",   "10"),
        "top-right":     ("w-text_w-10",    "10"),
        "center":        ("(w-text_w)/2",   "(h-text_h)/2"),
        "bottom-left":   ("10",             "h-text_h-30"),
        "bottom-center": ("(w-text_w)/2",   "h-text_h-30"),
        "bottom-right":  ("w-text_w-10",    "h-text_h-30"),
    }

    x_expr, y_expr = position_map.get(position.lower(), position_map["bottom-center"])

    safe_text = (
        text
        .replace("\\", "\\\\")
        .replace("'",  "\\'")
        .replace(":",  "\\:")
        .replace("[",  "\\[")
        .replace("]",  "\\]")
        .replace(",",  "\\,")
    )

    font_clause = ""
    if font_path:
        safe_font = font_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        font_clause = f"fontfile='{safe_font}':"

    # ✅ FIXED LOGIC
    time_filter = ""
    if end_time >= 0:
        time_filter = f":enable='between(t,{start_time},{end_time})'"
    elif start_time > 0:
        time_filter = f":enable='gte(t,{start_time})'"

    return (
        f"drawtext={font_clause}"
        f"text='{safe_text}'"
        f":fontsize={font_size}"
        f":fontcolor={font_color}"
        f":x={x_expr}:y={y_expr}"
        f":box=1:boxcolor=black@0.5:boxborderw=8"
        f"{time_filter}"
    )   


# ─── Timestamp helpers ────────────────────────────────────────────────────────

def _ts_to_seconds(ts) -> float:
    """Convert HH:MM:SS, MM:SS, or raw seconds string/float to float seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _seconds_to_ts(secs: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm string."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ─── Video probe helpers ──────────────────────────────────────────────────────

def _probe_video(input_path: str) -> Optional[dict]:
    """Return parsed ffprobe JSON for a video file, or None on failure."""
    result = run_ffprobe([
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        input_path,
    ])
    if not result.success:
        return None
    try:
        return json.loads(result.stdout)
    except Exception:
        return None


def _get_video_specs(input_path: str) -> dict:
    """
    Return a dict with: width, height, fps, duration, has_audio, sample_rate, channels.
    Falls back to safe defaults if probe fails.
    """
    info = _probe_video(input_path)
    specs = {
        "width": 1280, "height": 720,
        "fps": 25.0,
        "duration": None,
        "has_audio": False,
        "sample_rate": 44100,
        "channels": 2,
    }
    if not info:
        return specs

    fmt = info.get("format", {})
    if fmt.get("duration"):
        specs["duration"] = float(fmt["duration"])

    for stream in info.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video":
            specs["width"]  = stream.get("width",  specs["width"])
            specs["height"] = stream.get("height", specs["height"])
            fps_str = stream.get("avg_frame_rate") or stream.get("r_frame_rate", "25/1")
            try:
                if "/" in fps_str:
                    n, d = fps_str.split("/")
                    specs["fps"] = float(n) / float(d) if float(d) else 25.0
                else:
                    specs["fps"] = float(fps_str)
            except Exception:
                pass

        elif codec_type == "audio":
            specs["has_audio"]   = True
            specs["sample_rate"] = int(stream.get("sample_rate", 44100))
            specs["channels"]    = int(stream.get("channels", 2))

    return specs


# ─── Tools ───────────────────────────────────────────────────────────────────

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
        font_file: Optional path to a .ttf/.otf font file. Auto-detected if omitted.
        position: Text position. Options: 'top-left', 'top-center', 'top-right',
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

    vf = _make_drawtext_filter(
        text=text, position=position, font_size=font_size,
        font_color=font_color, font_path=font_path,
        start_time=start_time, end_time=end_time,
    )
    output_path = get_output_path(job_id, suffix="_text")

    result = run_ffmpeg(["-i", input_path, "-vf", vf, "-c:a", "copy", output_path])

    if result.success:
        return f"Text overlay added. Output saved to: {output_path}"
    return f"Text overlay failed: {result.error_message}"

@tool
def insert_freeze_frames(
    input_path: str,
    freeze_points: list,
    job_id: str,
) -> str:
    """
    Frame-perfect freeze-frame insertion using a single FFmpeg filter graph.
    Eliminates frame mismatch and ensures seamless transitions.
    """

    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    if not freeze_points:
        return "Error: freeze_points list is empty."

    # Normalize & sort
    normalised = []
    for fp in freeze_points:
        ts = _ts_to_seconds(fp["timestamp"])
        normalised.append({
            "ts": ts,
            "duration": float(fp["duration"]),
            "text": str(fp["text"]),
            "position": fp.get("position", "center"),
            "font_size": int(fp.get("font_size", 52)),
            "font_color": fp.get("font_color", "white"),
        })

    normalised.sort(key=lambda x: x["ts"])

    specs = _get_video_specs(input_path)
    if specs["duration"] is None:
        return "Error: Could not determine video duration."

    total_duration = specs["duration"]
    FPS = specs["fps"]

    font_path, font_err = _resolve_font_file(None)
    if font_err:
        return f"Font error: {font_err}"

    output_path = get_output_path(job_id, suffix="_freeze_annotated")

    # Build filter graph
    filters = []
    last_t = 0
    v_labels = []

    for i, fp in enumerate(normalised):
        t = fp["ts"]
        d = fp["duration"]

        # segment before freeze
        filters.append(
            f"[0:v]trim={last_t}:{t},setpts=PTS-STARTPTS[v{i}]"
        )

        # freeze using tpad (clone last frame)
        segment_duration = t - last_t

        drawtext = _make_drawtext_filter(
        text=fp["text"],
        position=fp["position"],
        font_size=fp["font_size"],
        font_color=fp["font_color"],
        font_path=font_path,
        start_time=segment_duration,
        end_time=segment_duration + d,
        )

        filters.append(
        f"[v{i}]tpad=stop_mode=clone:stop_duration={d},{drawtext}[vf{i}]"
        )

        v_labels.append(f"[vf{i}]")
        last_t = t

    # tail segment
    filters.append(
        f"[0:v]trim={last_t}:{total_duration},setpts=PTS-STARTPTS[v_last]"
    )
    v_labels.append("[v_last]")

    # concat all video parts
    filters.append(
        f"{''.join(v_labels)}concat=n={len(v_labels)}:v=1:a=0[outv]"
    )

    filter_complex = ";".join(filters)

    result = run_ffmpeg([
        "-hwaccel", "cuda",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-cq", "19",
        "-c:a", "aac",
        output_path
    ])

    if result.success:
        return f"Freeze-frame annotation complete. Output saved to: {output_path}"
    return f"Failed: {result.error_message}"


# @tool
# def insert_freeze_frames(
#     input_path: str,
#     freeze_points: list,
#     job_id: str,
# ) -> str:
#     """
#     Insert freeze-frame annotation pauses at multiple points in a video — exactly like
#     tutorial videos where playback pauses on a frozen frame, displays an instruction,
#     then continues. Multiple freeze points can be injected in a single call.

#     The output video plays normally → freezes at timestamp → shows text for the
#     specified duration → continues playing → freezes again (if more points) → etc.

#     Use this when the user wants tutorial-style annotations, step-by-step instructions
#     overlaid at specific moments, or any 'pause and explain' effect.

#     Args:
#         input_path: Absolute path to the input video file.
#         freeze_points: List of annotation dicts. Each dict must have:
#             - "timestamp" (str|float): When to freeze. HH:MM:SS or seconds. e.g. "00:00:15" or 15
#             - "duration"  (float):     How long to hold the freeze in seconds. e.g. 5.0
#             - "text"      (str):       Instruction text to display on the frozen frame.
#             Optional per-point keys:
#             - "position"   (str):   Text position. Default "center".
#                                     Options: top-left, top-center, top-right,
#                                              center, bottom-left, bottom-center, bottom-right
#             - "font_size"  (int):   Font size in pixels. Default 52.
#             - "font_color" (str):   Text color. Default "white".
#         job_id: The current job ID for naming the output file.

#     Example freeze_points:
#         [
#             {"timestamp": "00:00:10", "duration": 4, "text": "Step 1: Click the Settings button"},
#             {"timestamp": "00:00:28", "duration": 5, "text": "Step 2: Enter your username here"},
#             {"timestamp": "00:00:55", "duration": 4, "text": "Step 3: Click Save to confirm"},
#         ]
#     """
#     valid, err = validate_input_file(input_path)
#     if not valid:
#         return f"Error: {err}"

#     if not freeze_points:
#         return "Error: freeze_points list is empty. Provide at least one freeze annotation."

#     # ── Validate and normalise freeze points ──────────────────────────────────
#     normalised = []
#     for i, fp in enumerate(freeze_points):
#         if not isinstance(fp, dict):
#             return f"Error: freeze_points[{i}] must be a dict, got {type(fp).__name__}"
#         for required in ("timestamp", "duration", "text"):
#             if required not in fp:
#                 return f"Error: freeze_points[{i}] is missing required key '{required}'"
#         try:
#             ts_secs = _ts_to_seconds(fp["timestamp"])
#         except Exception as e:
#             return f"Error: freeze_points[{i}] has invalid timestamp '{fp['timestamp']}': {e}"
#         normalised.append({
#             "ts_secs":    ts_secs,
#             "duration":   float(fp["duration"]),
#             "text":       str(fp["text"]),
#             "position":   fp.get("position",   "center"),
#             "font_size":  int(fp.get("font_size",  52)),
#             "font_color": fp.get("font_color", "white"),
#         })

#     # Sort by timestamp so we process in order
#     normalised.sort(key=lambda x: x["ts_secs"])

#     # ── Probe input video ─────────────────────────────────────────────────────
#     specs = _get_video_specs(input_path)
#     if specs["duration"] is None:
#         return "Error: Could not determine video duration. Is this a valid video file?"

#     total_duration = specs["duration"]
#     W, H          = specs["width"], specs["height"]
#     FPS           = specs["fps"]
#     has_audio     = specs["has_audio"]
#     sample_rate   = specs["sample_rate"]
#     channels      = specs["channels"]

#     # Validate all timestamps are within the video
#     for i, fp in enumerate(normalised):
#         if fp["ts_secs"] >= total_duration:
#             return (
#                 f"Error: freeze_points[{i}] timestamp {fp['ts_secs']:.2f}s "
#                 f"is beyond video duration {total_duration:.2f}s"
#             )

#     # Resolve font once for all annotations
#     font_path, font_err = _resolve_font_file(None)
#     if font_err:
#         return f"Error resolving font: {font_err}"

#     # ── Work in a temp directory ──────────────────────────────────────────────
#     tmp_dir = tempfile.mkdtemp(prefix=f"freeze_{job_id}_")
#     concat_parts = []   # ordered list of file paths for final concat

#     try:
#         # ── Common re-encode settings (all parts must be identical for concat) ──
#         # We re-encode everything to h264/aac so concat -c copy works cleanly.
#         VIDEO_ENC = [
#     "-c:v", "h264_nvenc",
#     "-preset", "p5",
#     "-cq", "19",
#     "-vf", f"scale={W}:{H},fps={FPS:.3f}"
# ]
#         AUDIO_ENC = ["-c:a", "aac", "-ar", str(sample_rate), "-ac", str(channels)]
#         SILENT_AUDIO = ["-f", "lavfi", "-i",
#                         f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}"]

#         # Build segment boundaries:
#         # [0 → T0], [T0 → T1], [T1 → T2], ..., [TN → end]
#         boundaries = [0.0] + [fp["ts_secs"] for fp in normalised] + [total_duration]

#         # ── Step 1: Re-encode each segment of the original video ──────────────
#         for seg_idx in range(len(boundaries) - 1):
#             seg_start = boundaries[seg_idx]
#             seg_end   = boundaries[seg_idx + 1]
#             seg_dur   = seg_end - seg_start

#             # Skip zero-length or near-zero segments (e.g. freeze at t=0)
#             if seg_dur < 0.05:
#                 continue

#             seg_path = os.path.join(tmp_dir, f"seg_{seg_idx:03d}.mp4")

#             args = [
#     "-hwaccel", "cuda",
#     "-i", input_path,
#     "-ss", _seconds_to_ts(seg_start),
#     "-to", _seconds_to_ts(seg_end),
# ]

#             if has_audio:
#                 args += VIDEO_ENC + AUDIO_ENC
#             else:
#                 # No audio in source → add silent audio so concat is uniform
#                 args += VIDEO_ENC + SILENT_AUDIO + AUDIO_ENC + ["-shortest"]

#             args.append(seg_path)
#             result = run_ffmpeg(args)
#             if not result.success:
#                 return f"Error re-encoding segment {seg_idx}: {result.error_message}"

#             concat_parts.append(seg_path)

#             # ── Step 2: After each segment (except the last), insert freeze clip ─
#             # The freeze goes AFTER this segment if it's not the final segment
#             if seg_idx < len(normalised):
#                 fp = normalised[seg_idx]
#                 freeze_ts  = fp["ts_secs"]
#                 freeze_dur = fp["duration"]

#                 # 2a. Extract the frame image at the freeze timestamp
#                 frame_path = os.path.join(tmp_dir, f"frame_{seg_idx:03d}.png")
#                 frame_result = run_ffmpeg([
#     "-hwaccel", "cuda",
#     "-i", input_path,
#     "-ss", _seconds_to_ts(freeze_ts),
#     "-frames:v", "1",
#                     "-q:v", "1",
#                     frame_path,
#                 ])
#                 if not frame_result.success:
#                     return f"Error extracting frame at {_seconds_to_ts(freeze_ts)}: {frame_result.error_message}"

#                 # 2b. Build the drawtext filter for this annotation
#                 drawtext = _make_drawtext_filter(
#                     text=fp["text"],
#                     position=fp["position"],
#                     font_size=fp["font_size"],
#                     font_color=fp["font_color"],
#                     font_path=font_path,
#                 )

#                 # 2c. Create the freeze clip in ONE ffmpeg call:
#                 #     - Loop the frame image for `freeze_dur` seconds
#                 #     - Add silent audio track
#                 #     - Apply drawtext overlay
#                 #     - Re-encode to same spec as segments
#                 freeze_path = os.path.join(tmp_dir, f"freeze_{seg_idx:03d}.mp4")
#                 freeze_result = run_ffmpeg([
#                     "-loop", "1",
#                     "-i", frame_path,
#                     "-f", "lavfi",
#                     "-i", f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}",
#                     "-t", str(freeze_dur),
#                     "-vf", f"scale={W}:{H},fps={FPS:.3f},{drawtext}",
#                     "-c:v", "h264_nvenc",
#                     "-preset", "p5", "-cq", "19",
#                     "-c:a", "aac", "-ar", str(sample_rate), "-ac", str(channels),
#                     "-shortest",
#                     freeze_path,
#                 ])
#                 if not freeze_result.success:
#                     return (
#                         f"Error creating freeze clip at {_seconds_to_ts(freeze_ts)}: "
#                         f"{freeze_result.error_message}"
#                     )

#                 concat_parts.append(freeze_path)

#         if not concat_parts:
#             return "Error: No video segments were produced. Check timestamps and video duration."

#         # ── Step 3: Write concat list file ────────────────────────────────────
#         concat_list_path = os.path.join(tmp_dir, "concat.txt")
#         with open(concat_list_path, "w") as f:
#             for part in concat_parts:
#                 f.write(f"file '{part}'\n")

#         # ── Step 4: Final concatenation ───────────────────────────────────────
#         output_path = get_output_path(job_id, suffix="_freeze_annotated")

#         concat_result = run_ffmpeg([
#             "-f", "concat",
#             "-safe", "0",
#             "-i", concat_list_path,
#             "-c", "copy",       # all parts are already encoded to the same spec
#             output_path,
#         ])

#         if not concat_result.success:
#             return f"Error in final concat: {concat_result.error_message}"

#         n_freezes = len(normalised)
#         total_added = sum(fp["duration"] for fp in normalised)
#         return (
#             f"Freeze-frame annotation complete. "
#             f"Inserted {n_freezes} freeze pause{'s' if n_freezes != 1 else ''} "
#             f"({total_added:.1f}s added). "
#             f"Output saved to: {output_path}"
#         )

#     finally:
#         # Always clean up temp files
#         shutil.rmtree(tmp_dir, ignore_errors=True)


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
        position: Watermark position. Options: top-left, top-right, bottom-left, bottom-right, center.
        opacity: Watermark opacity 0.0–1.0. Default: 0.7.
        scale: Watermark size as fraction of video width. Default: 0.15 (15%).
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
        "-i", input_path, "-i", watermark_path,
        "-filter_complex",
        f"[1:v]scale=iw*{scale}:-1,format=rgba,colorchannelmixer=aa={opacity}[wm];"
        f"[0:v][wm]overlay={overlay_pos}[out]",
        "-map", "[out]", "-map", "0:a?", "-c:a", "copy",
        output_path,
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
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")

    result = run_ffmpeg(["-i", input_path, "-vf", f"subtitles='{safe_srt}'", "-c:a", "copy", output_path])

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
        "-i", input_path, "-ss", timestamp,
        "-vframes", "1", "-q:v", "2",
        output_path,
    ])

    if result.success:
        return f"Thumbnail saved to: {output_path}"
    return f"Thumbnail generation failed: {result.error_message}"