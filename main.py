"""
Video Agent — FastAPI Application Entry Point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Dashboard:  http://localhost:8000/
API Docs:   http://localhost:8000/docs
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes import router
from utils.file_utils import ensure_workspace_dirs
from config import API_HOST, API_PORT

ensure_workspace_dirs()

app = FastAPI(
    title="Video Editing Agent",
    description="AI-powered video editing agent. Ollama + LangChain + FFmpeg.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# ─── Serve dashboard ──────────────────────────────────────────────────────────
_DASHBOARD = os.path.join(os.path.dirname(__file__), "dashboard.html")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    with open(_DASHBOARD, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)