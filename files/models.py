"""
Job data models.
Defines the structure of a job throughout its lifecycle.
"""

from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AgentStep(BaseModel):
    """A single step in the agent's reasoning trace."""
    step_number: int
    type: str  # "thought", "tool_call", "tool_result", "final_answer"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Job(BaseModel):
    """Represents a video editing job from creation to completion."""
    id: str
    status: JobStatus = JobStatus.PENDING
    prompt: str
    input_file: str

    # Results
    output_file: Optional[str] = None
    result_message: Optional[str] = None
    error: Optional[str] = None

    # Agent reasoning trace — every step the agent took
    steps: List[AgentStep] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class JobSummary(BaseModel):
    """Lightweight job info for list responses."""
    id: str
    status: JobStatus
    prompt: str
    input_file: str
    output_file: Optional[str] = None
    created_at: datetime
    duration_seconds: Optional[float] = None
