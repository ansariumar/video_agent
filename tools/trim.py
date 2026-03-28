"""
Trimming, splitting, and merging tools.
"""

import os
import tempfile
from langchain.tools import tool

from utils.ffmpeg_runner import run_ffmpeg
from utils.file_utils import get_output_path, validate_input_file


@tool
def trim_video(input_path: str, start_time: str, end_time: str, job_id: str) -> str:
    """
    Trim a video to keep only the segment between start_time and end_time.
    Use this when the user wants to cut a clip, extract a segment, or shorten a video.

    Args:
        input_path: Absolute path to the input video file.
        start_time: Start timestamp in HH:MM:SS or seconds (e.g., '00:00:10' or '10').
        end_time: End timestamp in HH:MM:SS or seconds (e.g., '00:00:30' or '30').
        job_id: The current job ID for naming the output file.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    output_path = get_output_path(job_id, suffix="_trimmed")

    result = run_ffmpeg([
        "-i", input_path,
        "-ss", start_time,
        "-to", end_time,
        "-c", "copy",       # Stream copy — fast, no re-encode
        output_path
    ])

    if result.success:
        return f"Trimmed video saved to: {output_path}"
    return f"Trim failed: {result.error_message}"


@tool
def merge_videos(input_paths: list[str], job_id: str) -> str:
    """
    Concatenate multiple video files into a single video in the given order.
    Use this when the user wants to join, combine, or stitch videos together.

    Args:
        input_paths: Ordered list of absolute paths to video files to merge.
        job_id: The current job ID for naming the output file.
    """
    for path in input_paths:
        valid, err = validate_input_file(path)
        if not valid:
            return f"Error with file '{path}': {err}"

    # FFmpeg concat requires a temporary file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")
        concat_list = f.name

    output_path = get_output_path(job_id, suffix="_merged")

    try:
        result = run_ffmpeg([
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path
        ])
    finally:
        os.unlink(concat_list)

    if result.success:
        return f"Merged video saved to: {output_path}"
    return f"Merge failed: {result.error_message}"


@tool
def split_video(input_path: str, split_timestamps: list[str], job_id: str) -> str:
    """
    Split a video into multiple segments at the given timestamps.
    Use this when the user wants to divide a video into parts.

    Args:
        input_path: Absolute path to the input video file.
        split_timestamps: List of timestamps where splits should occur (e.g., ['00:01:00', '00:02:30']).
        job_id: The current job ID for naming output files.
    """
    valid, err = validate_input_file(input_path)
    if not valid:
        return f"Error: {err}"

    from utils.file_utils import get_output_path
    import os

    output_dir = os.path.dirname(get_output_path(job_id))
    output_paths = []
    errors = []

    # Build segment boundaries: [start, end] pairs
    boundaries = ["0"] + split_timestamps + ["999999"]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        out = os.path.join(output_dir, f"{job_id}_part{i+1}.mp4")

        result = run_ffmpeg([
            "-i", input_path,
            "-ss", start,
            "-to", end,
            "-c", "copy",
            out
        ])

        if result.success:
            output_paths.append(out)
        else:
            errors.append(f"Part {i+1}: {result.error_message}")

    if errors:
        return f"Split partially failed. Errors: {'; '.join(errors)}"
    return f"Split into {len(output_paths)} parts: {', '.join(output_paths)}"