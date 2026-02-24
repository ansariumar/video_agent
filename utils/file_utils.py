"""
File utility helpers.
Handles path validation, safe output path generation, and workspace management.
"""

import os
import uuid
from pathlib import Path

from config import INPUTS_DIR, OUTPUTS_DIR


# Supported video/audio formats
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv",
    ".wmv", ".m4v", ".ts", ".3gp"
}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".ogg", ".m4a"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def validate_input_file(path: str) -> tuple[bool, str]:
    """
    Validate that an input file exists and is readable.

    Returns:
        (is_valid, error_message) — error_message is empty string if valid.
    """
    if not path:
        return False, "File path is empty."

    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return False, f"File not found: '{abs_path}'"

    if not os.path.isfile(abs_path):
        return False, f"Path is not a file: '{abs_path}'"

    if not os.access(abs_path, os.R_OK):
        return False, f"File is not readable: '{abs_path}'"

    return True, ""


def get_output_path(job_id: str, suffix: str = "", extension: str = ".mp4") -> str:
    """
    Generate a safe, unique output file path in the outputs directory.

    Args:
        job_id: The job UUID.
        suffix: Optional label appended to filename (e.g., "_trimmed").
        extension: Output file extension including dot (e.g., ".mp4").

    Returns:
        Absolute path string for the output file.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filename = f"{job_id}{suffix}{extension}"
    return os.path.join(OUTPUTS_DIR, filename)


def get_file_extension(path: str) -> str:
    """Return the lowercase file extension including the dot."""
    return Path(path).suffix.lower()


def is_video_file(path: str) -> bool:
    return get_file_extension(path) in SUPPORTED_VIDEO_EXTENSIONS


def is_audio_file(path: str) -> bool:
    return get_file_extension(path) in SUPPORTED_AUDIO_EXTENSIONS


def is_image_file(path: str) -> bool:
    return get_file_extension(path) in SUPPORTED_IMAGE_EXTENSIONS


def ensure_workspace_dirs():
    """Create workspace directories if they don't exist."""
    os.makedirs(INPUTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_file_size_mb(path: str) -> float:
    """Return file size in megabytes, or 0 if file doesn't exist."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def resolve_input_path(path: str) -> str:
    """
    Resolve an input path. If relative, check workspace/inputs first,
    then fall back to treating it as relative to cwd.
    """
    if os.path.isabs(path):
        return path

    # Try workspace/inputs first
    candidate = os.path.join(INPUTS_DIR, path)
    if os.path.exists(candidate):
        return candidate

    # Fall back to absolute resolution from cwd
    return os.path.abspath(path)
