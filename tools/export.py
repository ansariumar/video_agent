"""
Export and format tools — conversion, compression, resolution presets, GIF, frames.
"""

import os
from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


@tool
def convert_format(input_path: str, output_format: str, job_id: str) -> str:
    """
    Convert a video file from one format to another.
    Use this when the user wants to change the file type or container of a video.

    Supported output formats: mp4, mkv, avi, mov, webm, flv, ts, gif

    Args:
        input_path: Absolute path to the input video file.
        output_format: Target format without dot (e.g., 'mp4', 'webm', 'mkv').
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    fmt = output_format.lower().strip().lstrip(".")
    output_path = get_output_path(job_id, suffix=f"_converted", extension=f".{fmt}")

    result = run_ffmpeg(["-i", input_path, output_path])

    if result.success:
        return f"Converted to {fmt.upper()}. Output saved to: {output_path}"
    return f"Format conversion failed: {result.error_message}"


@tool
def compress_video(input_path: str, job_id: str, quality: str = "medium") -> str:
    """
    Compress a video to reduce file size using H.264 encoding.
    Use this when the user wants to reduce file size, compress, or make a video smaller.

    Quality options:
    - 'high': Minimal compression, best quality (CRF 18). Large file.
    - 'medium': Balanced compression (CRF 23). Default.
    - 'low': Strong compression, smaller file (CRF 28). Some quality loss.
    - 'tiny': Maximum compression (CRF 35). Noticeable quality loss.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        quality: Compression level preset. Options: 'high', 'medium', 'low', 'tiny'. Default: 'medium'.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    crf_map = {"high": "18", "medium": "23", "low": "28", "tiny": "35"}
    crf = crf_map.get(quality.lower(), "23")

    output_path = get_output_path(job_id, suffix=f"_compressed_{quality}")

    result = run_ffmpeg([
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", crf,
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ])

    if result.success:
        return f"Compressed at '{quality}' quality (CRF {crf}). Output saved to: {output_path}"
    return f"Compression failed: {result.error_message}"


@tool
def change_resolution(input_path: str, preset: str, job_id: str) -> str:
    """
    Change video resolution to a standard preset.
    Use this when the user mentions a specific resolution like 1080p, 720p, or 4K.

    Available presets: '4k' (3840x2160), '1080p' (1920x1080), '720p' (1280x720),
                       '480p' (854x480), '360p' (640x360), '240p' (426x240)

    Args:
        input_path: Absolute path to the input video file.
        preset: Resolution preset name (e.g., '1080p', '720p', '4k').
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    preset_map = {
        "4k":    "3840:2160",
        "2k":    "2560:1440",
        "1080p": "1920:1080",
        "720p":  "1280:720",
        "480p":  "854:480",
        "360p":  "640:360",
        "240p":  "426:240",
    }

    scale = preset_map.get(preset.lower().strip())
    if not scale:
        available = ", ".join(preset_map.keys())
        return f"Error: Unknown preset '{preset}'. Available: {available}"

    output_path = get_output_path(job_id, suffix=f"_{preset}")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={scale}",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Resolution changed to {preset}. Output saved to: {output_path}"
    return f"Resolution change failed: {result.error_message}"


@tool
def create_gif(
    input_path: str,
    job_id: str,
    start_time: str = "0",
    duration: float = 5.0,
    fps: int = 10,
    width: int = 480,
) -> str:
    """
    Convert a segment of a video into an animated GIF.
    Use this when the user wants to create a GIF, animation, or shareable clip.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        start_time: Start time of the segment to convert (HH:MM:SS or seconds). Default: '0'.
        duration: Duration in seconds of the GIF. Default: 5.0 seconds.
        fps: Frames per second for the GIF (higher = smoother but larger file). Default: 10.
        width: Width of the GIF in pixels (height auto-calculated). Default: 480.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_clip", extension=".gif")

    # Two-pass GIF: generate optimal palette first, then apply it
    palette_path = get_output_path(job_id, suffix="_palette", extension=".png")

    pass1 = run_ffmpeg([
        "-ss", start_time,
        "-t", str(duration),
        "-i", input_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
        palette_path
    ])

    if not pass1.success:
        return f"GIF palette generation failed: {pass1.error_message}"

    pass2 = run_ffmpeg([
        "-ss", start_time,
        "-t", str(duration),
        "-i", input_path,
        "-i", palette_path,
        "-filter_complex", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        output_path
    ])

    # Clean up palette
    try:
        os.unlink(palette_path)
    except OSError:
        pass

    if pass2.success:
        return f"GIF created ({duration}s, {fps}fps, {width}px wide). Output saved to: {output_path}"
    return f"GIF creation failed: {pass2.error_message}"


@tool
def extract_frames(
    input_path: str,
    job_id: str,
    fps: float = 1.0,
    output_format: str = "jpg",
) -> str:
    """
    Extract frames from a video as individual image files.
    Use this when the user wants to save frames, screenshots, or individual images from a video.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output directory.
        fps: How many frames to extract per second of video. Default: 1.0 (one per second).
        output_format: Image format for extracted frames ('jpg' or 'png'). Default: 'jpg'.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    from config import OUTPUTS_DIR
    frames_dir = os.path.join(OUTPUTS_DIR, f"{job_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    output_pattern = os.path.join(frames_dir, f"frame_%04d.{output_format}")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern
    ])

    if result.success:
        # Count extracted frames
        frame_count = len([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
        return f"Extracted {frame_count} frames to directory: {frames_dir}"
    return f"Frame extraction failed: {result.error_message}"


@tool
def get_video_info(input_path: str) -> str:
    """
    Get technical information about a video file: duration, resolution, codec, fps, file size.
    Use this when you need to inspect a video before editing or when the user asks about video properties.

    Args:
        input_path: Absolute path to the video file to inspect.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    from utils.ffmpeg_runner import run_ffprobe
    import os

    result = run_ffprobe([
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        input_path
    ])

    if not result.success:
        return f"Could not read video info: {result.error_message}"

    try:
        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        duration = float(fmt.get("duration", 0))
        size_mb = os.path.getsize(input_path) / (1024 * 1024)

        info = [
            f"File: {os.path.basename(input_path)}",
            f"Duration: {duration:.1f}s ({int(duration//60)}m {int(duration%60)}s)",
            f"Size: {size_mb:.1f} MB",
            f"Video: {video_stream.get('codec_name','?').upper()} "
            f"{video_stream.get('width','?')}x{video_stream.get('height','?')} "
            f"@ {eval(video_stream.get('avg_frame_rate','0/1')):.1f}fps",
        ]
        if audio_stream:
            info.append(f"Audio: {audio_stream.get('codec_name','?').upper()} "
                        f"{audio_stream.get('sample_rate','?')}Hz")
        else:
            info.append("Audio: None")

        return "\n".join(info)
    except Exception as e:
        return f"Could not parse video info: {e}. Raw: {result.stdout[:300]}"
