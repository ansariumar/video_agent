"""
Job Manager.
Simple in-memory store for job lifecycle management.
All jobs live in a dict keyed by job_id.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from jobs.models import Job, JobStatus, AgentStep, JobSummary


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def create_job(self, prompt: str, input_file: str) -> Job:
        """Create a new job and store it. Returns the Job object."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            prompt=prompt,
            input_file=input_file,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID. Returns None if not found."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[JobSummary]:
        """Return a summary list of all jobs, newest first."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [
            JobSummary(
                id=j.id,
                status=j.status,
                prompt=j.prompt,
                input_file=j.input_file,
                output_file=j.output_file,
                created_at=j.created_at,
                duration_seconds=j.duration_seconds,
            )
            for j in jobs
        ]

    def mark_running(self, job_id: str):
        """Transition job to RUNNING status."""
        job = self._get_or_raise(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

    def mark_done(self, job_id: str, output_file: str, result_message: str):
        """Transition job to DONE with output path and summary message."""
        job = self._get_or_raise(job_id)
        job.status = JobStatus.DONE
        job.output_file = output_file
        job.result_message = result_message
        job.completed_at = datetime.utcnow()

    def mark_failed(self, job_id: str, error: str):
        """Transition job to FAILED with an error message."""
        job = self._get_or_raise(job_id)
        job.status = JobStatus.FAILED
        job.error = error
        job.completed_at = datetime.utcnow()

    def add_step(self, job_id: str, step: AgentStep):
        """Append an agent reasoning step to the job's trace."""
        job = self._get_or_raise(job_id)
        step.step_number = len(job.steps) + 1
        job.steps.append(step)

    def _get_or_raise(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"Job '{job_id}' not found.")
        return job


# Singleton instance — imported by routes and agent
job_manager = JobManager()
