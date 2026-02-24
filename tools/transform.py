"""
Visual transformation tools — resize, crop, rotate, flip, speed, reverse.
"""

from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


@tool
def resize_video(input_path: str, width: int, height: int, job_id: str) -> str:
    """
    Resize a video to the specified width and height in pixels.
    Use -1 for width or height to maintain the aspect ratio automatically.
    Use this when the user wants to scale, resize, or change the resolution of a video.

    Args:
        input_path: Absolute path to the input video file.
        width: Target width in pixels. Use -1 to auto-calculate from height.
        height: Target height in pixels. Use -1 to auto-calculate from width.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_resized")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"scale={width}:{height}",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Resized to {width}x{height}. Output saved to: {output_path}"
    return f"Resize failed: {result.error_message}"


@tool
def crop_video(input_path: str, width: int, height: int, x: int, y: int, job_id: str) -> str:
    """
    Crop a video to a rectangular region. The region starts at pixel position (x, y)
    and extends to the given width and height.
    Use this when the user wants to crop, cut out, or focus on a specific area of the video.

    Args:
        input_path: Absolute path to the input video file.
        width: Width of the crop region in pixels.
        height: Height of the crop region in pixels.
        x: X offset (left edge) of the crop region in pixels.
        y: Y offset (top edge) of the crop region in pixels.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_cropped")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"crop={width}:{height}:{x}:{y}",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Cropped to {width}x{height} at ({x},{y}). Output saved to: {output_path}"
    return f"Crop failed: {result.error_message}"


@tool
def rotate_video(input_path: str, degrees: int, job_id: str) -> str:
    """
    Rotate a video by the specified number of degrees.
    Supported values: 90, 180, 270 (clockwise). Use 270 for 90° counter-clockwise.
    Use this when a video is sideways, upside down, or needs rotation.

    Args:
        input_path: Absolute path to the input video file.
        degrees: Rotation in degrees. Must be 90, 180, or 270.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    transpose_map = {
        90: "1",    # 90 clockwise
        180: "2,transpose=2",  # 180
        270: "2",   # 90 counter-clockwise
    }

    if degrees not in transpose_map:
        return f"Error: degrees must be 90, 180, or 270. Got: {degrees}"

    output_path = get_output_path(job_id, suffix=f"_rotated{degrees}")
    vf = f"transpose={transpose_map[degrees]}"

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Rotated {degrees}°. Output saved to: {output_path}"
    return f"Rotation failed: {result.error_message}"


@tool
def flip_video(input_path: str, direction: str, job_id: str) -> str:
    """
    Flip a video horizontally (mirror) or vertically (upside down).
    Use this when the user wants to mirror or flip a video.

    Args:
        input_path: Absolute path to the input video file.
        direction: Either 'horizontal' (left-right mirror) or 'vertical' (upside down).
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    direction = direction.lower().strip()
    if direction in ("horizontal", "hflip", "h"):
        vf = "hflip"
    elif direction in ("vertical", "vflip", "v"):
        vf = "vflip"
    else:
        return f"Error: direction must be 'horizontal' or 'vertical'. Got: '{direction}'"

    output_path = get_output_path(job_id, suffix=f"_flip_{direction[0]}")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"Flipped {direction}. Output saved to: {output_path}"
    return f"Flip failed: {result.error_message}"


@tool
def change_speed(input_path: str, speed_factor: float, job_id: str) -> str:
    """
    Change the playback speed of a video. Also adjusts audio pitch.
    Use this for slow motion (< 1.0) or timelapse / fast-forward (> 1.0) effects.
    Examples: 0.5 = half speed (slow motion), 2.0 = double speed (timelapse).

    Args:
        input_path: Absolute path to the input video file.
        speed_factor: Playback speed multiplier. 0.5 = half speed, 2.0 = double speed.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    if speed_factor <= 0:
        return "Error: speed_factor must be greater than 0."

    output_path = get_output_path(job_id, suffix=f"_speed{speed_factor}x")

    # Video: setpts filter. Audio: atempo (supports 0.5-2.0, chain for extremes)
    video_filter = f"setpts={1/speed_factor}*PTS"
    audio_filter = _build_atempo_chain(speed_factor)

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", video_filter,
        "-af", audio_filter,
        output_path
    ])

    if result.success:
        return f"Speed changed to {speed_factor}x. Output saved to: {output_path}"
    return f"Speed change failed: {result.error_message}"


def _build_atempo_chain(factor: float) -> str:
    """
    FFmpeg's atempo filter only supports 0.5–2.0.
    Chain multiple atempo filters for values outside that range.
    """
    filters = []
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.4f}")
    return ",".join(filters)


@tool
def reverse_video(input_path: str, job_id: str) -> str:
    """
    Reverse a video so it plays backwards.
    Use this when the user wants a reverse, rewind, or backwards effect.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_reversed")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", "reverse",
        "-af", "areverse",
        output_path
    ])

    if result.success:
        return f"Reversed video saved to: {output_path}"
    return f"Reverse failed: {result.error_message}"
