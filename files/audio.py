"""
Audio manipulation tools.
"""

from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


@tool
def extract_audio(input_path: str, job_id: str, output_format: str = "mp3") -> str:
    """
    Extract the audio track from a video file and save it as an audio file.
    Use this when the user wants to rip, save, or export audio from a video.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
        output_format: Audio format to export (mp3, aac, wav, flac). Default is mp3.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_audio", extension=f".{output_format}")

    result = run_ffmpeg([
        "-i", input_path,
        "-vn",              # No video
        "-acodec", "copy" if output_format in ("aac", "mp3") else output_format,
        output_path
    ])

    if result.success:
        return f"Audio extracted to: {output_path}"
    # Retry with re-encode if copy failed
    result = run_ffmpeg(["-i", input_path, "-vn", output_path])
    if result.success:
        return f"Audio extracted to: {output_path}"
    return f"Audio extraction failed: {result.error_message}"


@tool
def mute_video(input_path: str, job_id: str) -> str:
    """
    Remove all audio from a video, producing a silent video file.
    Use this when the user wants to mute, silence, or strip audio from a video.

    Args:
        input_path: Absolute path to the input video file.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_muted")

    result = run_ffmpeg([
        "-i", input_path,
        "-an",              # Remove audio
        "-c:v", "copy",
        output_path
    ])

    if result.success:
        return f"Muted video saved to: {output_path}"
    return f"Mute failed: {result.error_message}"


@tool
def replace_audio(video_path: str, audio_path: str, job_id: str) -> str:
    """
    Replace the audio track of a video with a different audio file.
    Use this when the user wants to swap, change, or dub the audio in a video.

    Args:
        video_path: Absolute path to the input video file.
        audio_path: Absolute path to the new audio file.
        job_id: The current job ID for naming the output file.
    """
    for path, label in [(video_path, "video"), (audio_path, "audio")]:
        valid, err = validate_input_file(path)
        if not valid:
            return f"Error with {label} file: {err}"

    output_path = get_output_path(job_id, suffix="_replaced_audio")

    result = run_ffmpeg([
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",    # Video from first input
        "-map", "1:a:0",    # Audio from second input
        "-shortest",        # End when shortest stream ends
        output_path
    ])

    if result.success:
        return f"Audio replaced. Output saved to: {output_path}"
    return f"Audio replacement failed: {result.error_message}"


@tool
def adjust_volume(input_path: str, volume_factor: float, job_id: str) -> str:
    """
    Adjust the volume of a video's audio track.
    Use this when the user wants to make audio louder or quieter.
    A factor of 2.0 doubles the volume, 0.5 halves it, 1.0 is unchanged.

    Args:
        input_path: Absolute path to the input video file.
        volume_factor: Volume multiplier. Values > 1 increase volume, < 1 decrease it.
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_volume")

    result = run_ffmpeg([
        "-i", input_path,
        "-af", f"volume={volume_factor}",
        "-c:v", "copy",
        output_path
    ])

    if result.success:
        return f"Volume adjusted (factor: {volume_factor}). Output saved to: {output_path}"
    return f"Volume adjustment failed: {result.error_message}"


@tool
def add_background_music(
    video_path: str,
    music_path: str,
    job_id: str,
    music_volume: float = 0.3,
    original_volume: float = 1.0
) -> str:
    """
    Mix background music into a video while keeping the original audio.
    Use this when the user wants to add background music, soundtrack, or ambient audio.

    Args:
        video_path: Absolute path to the input video file.
        music_path: Absolute path to the background music file.
        job_id: The current job ID for naming the output file.
        music_volume: Volume of the background music (0.0-1.0). Default 0.3 (30%).
        original_volume: Volume of the original video audio (0.0-1.0). Default 1.0 (100%).
    """
    for path, label in [(video_path, "video"), (music_path, "music")]:
        valid, err = validate_input_file(path)
        if not valid:
            return f"Error with {label} file: {err}"

    output_path = get_output_path(job_id, suffix="_with_music")

    # Mix original audio + music using amix filter
    result = run_ffmpeg([
        "-i", video_path,
        "-i", music_path,
        "-filter_complex",
        f"[0:a]volume={original_volume}[a0];[1:a]volume={music_volume}[a1];[a0][a1]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-shortest",
        output_path
    ])

    if result.success:
        return f"Background music added. Output saved to: {output_path}"
    return f"Adding background music failed: {result.error_message}"
