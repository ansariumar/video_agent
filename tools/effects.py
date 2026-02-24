"""
Visual effects tools — color filters, brightness, contrast, fade, stabilization.
"""

from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


@tool
def apply_color_filter(input_path: str, filter_name: str, job_id: str) -> str:
    """
    Apply a named color filter/effect to a video.
    Use this when the user wants a visual style or color effect.

    Available filters:
    - 'grayscale' or 'black_and_white': Convert to black and white.
    - 'sepia': Warm brownish vintage tone.
    - 'invert': Invert all colors (negative effect).
    - 'vignette': Darken the edges of the frame.
    - 'sharpen': Sharpen the image.
    - 'blur': Apply a soft blur.
    - 'vintage': Faded, desaturated vintage look.

    Args:
        input_path: Absolute path to the input video file.
        filter_name: Name of the filter to apply (see available filters above).
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    filter_map = {
        "grayscale":       "hue=s=0",
        "black_and_white": "hue=s=0",
        "sepia":           "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
        "invert":          "negate",
        "vignette":        "vignette=PI/4",
        "sharpen":         "unsharp=5:5:1.5:5:5:0.0",
        "blur":            "boxblur=2:1",
        "vintage":         "curves=vintage,hue=s=0.6",
    }

    vf = filter_map.get(filter_name.lower().strip())
    if not vf:
        available = ", ".join(filter_map.keys())
        return f"Error: Unknown filter '{filter_name}'. Available: {available}"

    output_path = get_output_path(job_id, suffix=f"_{filter_name}")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return f"'{filter_name}' filter applied. Output saved to: {output_path}"
    return f"Color filter failed: {result.error_message}"


@tool
def adjust_brightness_contrast(
    input_path: str,
    job_id: str,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> str:
    """
    Adjust brightness, contrast, and saturation of a video.
    Use this when the user wants to make a video brighter, darker, more/less vivid, or adjust contrast.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        brightness: Brightness adjustment. Range -1.0 to 1.0. 0.0 = no change, 0.3 = brighter, -0.3 = darker.
        contrast: Contrast multiplier. 1.0 = no change, 1.5 = more contrast, 0.5 = less contrast.
        saturation: Color saturation multiplier. 1.0 = no change, 2.0 = vivid colors, 0.0 = grayscale.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_adjusted")

    result = run_ffmpeg([
        "-i", input_path,
        "-vf", f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}",
        "-c:a", "copy",
        output_path
    ])

    if result.success:
        return (
            f"Adjusted brightness={brightness}, contrast={contrast}, saturation={saturation}. "
            f"Output saved to: {output_path}"
        )
    return f"Brightness/contrast adjustment failed: {result.error_message}"


@tool
def add_fade(
    input_path: str,
    job_id: str,
    fade_in_duration: float = 1.0,
    fade_out_duration: float = 1.0,
) -> str:
    """
    Add a fade-in at the start and/or fade-out at the end of a video.
    Both video and audio are faded. Use 0 to skip either fade.
    Use this when the user wants smooth transitions at the beginning or end of a video.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        fade_in_duration: Duration of the fade-in effect in seconds. Use 0 to skip. Default: 1.0.
        fade_out_duration: Duration of the fade-out effect in seconds. Use 0 to skip. Default: 1.0.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    # Get video duration via ffprobe
    probe = run_ffmpeg(["-i", input_path, "-f", "null", "-"])
    # Parse duration from stderr (FFmpeg always prints duration to stderr)
    duration = _parse_duration(probe.stderr)
    if duration is None:
        return "Error: Could not determine video duration for fade-out calculation."

    video_filters = []
    audio_filters = []

    if fade_in_duration > 0:
        video_filters.append(f"fade=t=in:st=0:d={fade_in_duration}")
        audio_filters.append(f"afade=t=in:st=0:d={fade_in_duration}")

    if fade_out_duration > 0:
        fade_out_start = max(0, duration - fade_out_duration)
        video_filters.append(f"fade=t=out:st={fade_out_start:.3f}:d={fade_out_duration}")
        audio_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_duration}")

    if not video_filters:
        return "Error: Both fade_in_duration and fade_out_duration are 0. Nothing to do."

    output_path = get_output_path(job_id, suffix="_fade")

    args = ["-i", input_path]
    if video_filters:
        args += ["-vf", ",".join(video_filters)]
    if audio_filters:
        args += ["-af", ",".join(audio_filters)]
    args.append(output_path)

    result = run_ffmpeg(args)

    if result.success:
        return f"Fade applied (in={fade_in_duration}s, out={fade_out_duration}s). Output saved to: {output_path}"
    return f"Fade failed: {result.error_message}"


def _parse_duration(stderr: str) -> float | None:
    """Extract video duration in seconds from FFmpeg stderr output."""
    import re
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", stderr)
    if match:
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        return h * 3600 + m * 60 + s
    return None


@tool
def stabilize_video(input_path: str, job_id: str, smoothing: int = 10) -> str:
    """
    Stabilize a shaky video using FFmpeg's vidstab filter (two-pass process).
    Use this when the user has shaky, handheld, or unstable footage.
    Note: Requires the libvidstab FFmpeg plugin to be installed.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        smoothing: Smoothing strength (1-100). Higher = smoother but more crop. Default: 10.
    """
    import tempfile
    import os

    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    transforms_file = os.path.join(tempfile.gettempdir(), f"{job_id}_transforms.trf")
    output_path = get_output_path(job_id, suffix="_stabilized")

    # Pass 1: detect motion and save transforms
    pass1 = run_ffmpeg([
        "-i", input_path,
        "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file}",
        "-f", "null", "-"
    ])

    if not pass1.success:
        return (
            f"Stabilization pass 1 failed. This usually means libvidstab is not installed. "
            f"Error: {pass1.error_message}"
        )

    # Pass 2: apply stabilization
    pass2 = run_ffmpeg([
        "-i", input_path,
        "-vf", f"vidstabtransform=input={transforms_file}:smoothing={smoothing}:crop=black",
        "-c:a", "copy",
        output_path
    ])

    # Cleanup transforms file
    try:
        os.unlink(transforms_file)
    except OSError:
        pass

    if pass2.success:
        return f"Video stabilized. Output saved to: {output_path}"
    return f"Stabilization pass 2 failed: {pass2.error_message}"
