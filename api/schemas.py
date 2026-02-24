"""
API request and response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    """Request body for submitting a new video editing task."""
    prompt: str = Field(
        ...,
        description="Natural language description of the editing task.",
        examples=["Trim the video from 00:00:10 to 00:00:30"]
    )
    input_file: str = Field(
        ...,
        description="Path to the input video file. Can be absolute or relative to workspace/inputs/.",
        examples=["workspace/inputs/my_video.mp4", "/home/user/videos/clip.mp4"]
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the default Ollama model for this job.",
        examples=["qwen2.5:7b", "mistral:7b"]
    )


class TaskResponse(BaseModel):
    """Response when a task is successfully submitted."""
    job_id: str
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
