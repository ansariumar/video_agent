"""
Video Agent — FastAPI Application Entry Point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

API Docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.file_utils import ensure_workspace_dirs
from config import API_HOST, API_PORT

# ─── Startup ──────────────────────────────────────────────────────────────────
ensure_workspace_dirs()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Video Editing Agent",
    description=(
        "An AI-powered video editing agent. "
        "Send natural language editing requests, get back edited videos. "
        "Powered by Ollama + LangChain + FFmpeg."
    ),
    version="1.0.0",
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes
app.include_router(router, prefix="/api/v1")


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Video Editing Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ─── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
