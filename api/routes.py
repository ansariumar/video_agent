"""
FastAPI route definitions.
All API endpoints are defined here.
"""

import os
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from api.schemas import TaskRequest, TaskResponse, ErrorResponse
from jobs.manager import job_manager
from jobs.models import Job, JobSummary
from agent.core import run_job, subscribe, unsubscribe
from utils.file_utils import validate_input_file, resolve_input_path

router = APIRouter()


# ─── Submit a Task ────────────────────────────────────────────────────────────

@router.post(
    "/task",
    response_model=TaskResponse,
    summary="Submit a video editing task",
)
async def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
    resolved = resolve_input_path(request.input_file)
    valid, err = validate_input_file(resolved)
    if not valid:
        raise HTTPException(status_code=400, detail=f"Input file error: {err}")

    job = job_manager.create_job(
        prompt=request.prompt,
        input_file=request.input_file,
    )
    background_tasks.add_task(run_job, job.id, request.model)

    return TaskResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Job created. Poll GET /status/{job.id} for updates.",
    )


# ─── Job Status ───────────────────────────────────────────────────────────────

@router.get("/status/{job_id}", response_model=Job, summary="Get job status and reasoning trace")
async def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


# ─── List All Jobs ────────────────────────────────────────────────────────────

@router.get("/jobs", response_model=list[JobSummary], summary="List all jobs")
async def list_jobs():
    return job_manager.list_jobs()


# ─── Download Output File ─────────────────────────────────────────────────────

@router.get("/download/{job_id}", summary="Download output file")
async def download_output(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status.value != "done":
        raise HTTPException(status_code=400, detail=f"Job not done yet. Status: {job.status.value}")
    if not job.output_file or not os.path.exists(job.output_file):
        raise HTTPException(status_code=404, detail="Output file not found.")

    return FileResponse(
        path=job.output_file,
        filename=os.path.basename(job.output_file),
        media_type="application/octet-stream",
    )


# ─── WebSocket: Live Agent Stream ─────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def websocket_stream(websocket: WebSocket, job_id: str):
    """
    Real-time WebSocket stream of agent events for a job.
    If job is already done, replays its full step history then closes.
    """
    await websocket.accept()

    job = job_manager.get_job(job_id)
    if not job:
        await websocket.send_text(json.dumps({"type": "error", "message": f"Job '{job_id}' not found."}))
        await websocket.close()
        return

    # Replay history if job already finished
    if job.status.value in ("done", "failed"):
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": job.status.value,
            "message": job.result_message or job.error or "",
            "output_file": job.output_file,
        }))
        for step in job.steps:
            await websocket.send_text(json.dumps({
                "type": step.type,
                "tool": step.tool_name,
                "args": step.tool_args,
                "message": step.content,
                "ts": step.timestamp.isoformat(),
                "replayed": True,
            }))
        await websocket.send_text(json.dumps({"type": "done", "replayed": True}))
        await websocket.close()
        return

    # Live stream
    queue = subscribe(job_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event, default=str))
            if event.get("type") == "done":
                break
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe(job_id, queue)


# ─── Health Check ─────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "message": "Video Agent is running."}