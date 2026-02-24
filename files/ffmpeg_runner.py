"""
FFmpeg command executor.
Runs FFmpeg as a subprocess, captures stdout/stderr, and returns
a structured result the agent can reason about on failure.
"""

import subprocess
import shlex
from dataclasses import dataclass
from typing import Optional

from config import FFMPEG_BIN, FFPROBE_BIN


@dataclass
class FFmpegResult:
    success: bool
    command: str
    stdout: str
    stderr: str
    returncode: int
    error_message: Optional[str] = None

    def __str__(self):
        if self.success:
            return f"Success. Command: {self.command}"
        return (
            f"Failed (exit code {self.returncode}).\n"
            f"Command: {self.command}\n"
            f"Error: {self.error_message or self.stderr[:500]}"
        )


def run_ffmpeg(args: list[str], timeout: int = 300) -> FFmpegResult:
    """
    Execute an FFmpeg command.

    Args:
        args: List of FFmpeg arguments (do NOT include 'ffmpeg' itself).
        timeout: Max seconds to wait before killing the process.

    Returns:
        FFmpegResult with success status and captured output.
    """
    cmd = [FFMPEG_BIN, "-y"] + args  # -y = overwrite output without asking
    cmd_str = shlex.join(cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        error_message = None

        if not success:
            # Extract the most useful part of FFmpeg stderr
            stderr_lines = result.stderr.strip().splitlines()
            # FFmpeg puts the actual error near the end
            error_lines = [
                line for line in stderr_lines
                if any(kw in line.lower() for kw in ["error", "invalid", "no such", "unable", "failed"])
            ]
            error_message = "\n".join(error_lines[-5:]) if error_lines else result.stderr[-500:]

        return FFmpegResult(
            success=success,
            command=cmd_str,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error_message=error_message,
        )

    except subprocess.TimeoutExpired:
        return FFmpegResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr="",
            returncode=-1,
            error_message=f"FFmpeg timed out after {timeout} seconds.",
        )
    except FileNotFoundError:
        return FFmpegResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr="",
            returncode=-1,
            error_message=(
                f"FFmpeg binary not found at '{FFMPEG_BIN}'. "
                "Make sure FFmpeg is installed and on your PATH."
            ),
        )
    except Exception as e:
        return FFmpegResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr="",
            returncode=-1,
            error_message=f"Unexpected error running FFmpeg: {str(e)}",
        )


def run_ffprobe(args: list[str]) -> FFmpegResult:
    """
    Execute an FFprobe command (for video inspection).

    Args:
        args: List of ffprobe arguments (do NOT include 'ffprobe' itself).

    Returns:
        FFmpegResult with probe output in stdout.
    """
    cmd = [FFPROBE_BIN] + args
    cmd_str = shlex.join(cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return FFmpegResult(
            success=result.returncode == 0,
            command=cmd_str,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            error_message=result.stderr if result.returncode != 0 else None,
        )
    except Exception as e:
        return FFmpegResult(
            success=False,
            command=cmd_str,
            stdout="",
            stderr="",
            returncode=-1,
            error_message=str(e),
        )
