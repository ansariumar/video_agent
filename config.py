"""
Central configuration for the Video Agent.
This is your control panel — change settings here, nowhere else.
"""

# ─── Ollama Model ────────────────────────────────────────────────────────────
# Swap this to test different models. Must be pulled in Ollama first.
# Examples: "minimax-m2.7:cloud", "qwen3.5:2b", "granite4:latest", "lfm2.5-thinking:latest"
OLLAMA_MODEL = "minimax-m2.7:cloud"

# Ollama server base URL (default local)
OLLAMA_BASE_URL = "http://localhost:11434"

# ─── Agent Settings ──────────────────────────────────────────────────────────
# Maximum number of tool calls the agent can make in a single job
AGENT_MAX_ITERATIONS = 15

# Temperature for the model (0 = deterministic, good for tool use)
AGENT_TEMPERATURE = 0

# ─── Paths ───────────────────────────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
INPUTS_DIR = os.path.join(WORKSPACE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(WORKSPACE_DIR, "outputs")

# ─── FFmpeg ──────────────────────────────────────────────────────────────────
# Path to ffmpeg binary (assumes it's on PATH)
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000