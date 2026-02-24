"""
FastAPI route definitions.
All API endpoints are defined here.
"""

import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from api.schemas import TaskRequest, TaskResponse, ErrorResponse
from jobs.manager import job_manager
from jobs.models import Job, JobSummary
from agent.core import run_job
from utils.file_utils import validate_input_file, resolve_input_path

router = APIRouter()


# ─── Submit a Task ────────────────────────────────────────────────────────────

@router.post(
    "/task",
    response_model=TaskResponse,
    summary="Submit a video editing task",
    description="Submit a natural language video editing request. Returns a job_id immediately. Poll /status/{job_id} for progress.",
)
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    # Validate the input file before even creating the job
    resolved = resolve_input_path(request.input_file)
    valid, err = validate_input_file(resolved)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Input file error: {err}")

    # Create job
    job = job_manager.create_job(
        prompt=request.prompt,
        input_file=request.input_file,
    )

    # Fire the agent as a background task — non-blocking
    background_tasks.add_task(run_job, job.id, request.model)

    return TaskResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Job created. Poll GET /status/{job.id} for updates.",
    )


# ─── Job Status ───────────────────────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    response_model=Job,
    summary="Get job status and reasoning trace",
    description="Returns the full job object including status, output path, and every agent step.",
)
async def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


# ─── List All Jobs ────────────────────────────────────────────────────────────

@router.get(
    "/jobs",
    response_model=list[JobSummary],
    summary="List all jobs",
    description="Returns a summary list of all submitted jobs, newest first.",
)
async def list_jobs():
    return job_manager.list_jobs()


# ─── Download Output File ─────────────────────────────────────────────────────

@router.get(
    "/download/{job_id}",
    summary="Download the output file for a completed job",
    description="Returns the output video file as a downloadable response.",
)
async def download_output(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.status.value != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not done yet. Current status: {job.status.value}"
        )

    if not job.output_file:
        raise HTTPException(status_code=404, detail="No output file available for this job.")

    if not os.path.exists(job.output_file):
        raise HTTPException(
            status_code=404,
            detail=f"Output file not found on disk: {job.output_file}"
        )

    return FileResponse(
        path=job.output_file,
        filename=os.path.basename(job.output_file),
        media_type="application/octet-stream",
    )


# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    summary="Health check",
)
async def health():
    return {"status": "ok", "message": "Video Agent is running."}
