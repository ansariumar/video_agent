"""
Agent Core — with robust streaming and full error visibility.

Handles both LangGraph streaming formats:
  - v2 format (LangGraph >= 1.1): chunks are dicts {"type": ..., "data": ...}
  - v1 format (older):            chunks are tuples (mode, data)

The version="v2" parameter is passed but we defensively handle both so
the code works regardless of what langchain/langgraph version is installed.
"""

import re
import threading
import asyncio
import traceback
import logging
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, AGENT_TEMPERATURE
from tools.registry import ALL_TOOLS
from jobs.manager import job_manager
from jobs.models import AgentStep
from agent.prompts import SYSTEM_PROMPT
from utils.file_utils import resolve_input_path, validate_input_file

# Full tracebacks go to the server console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent.core")


# ─── Event Bus ────────────────────────────────────────────────────────────────
_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers[job_id].append(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue):
    try:
        _subscribers[job_id].remove(q)
    except ValueError:
        pass
    if not _subscribers[job_id]:
        _subscribers.pop(job_id, None)


def _emit(job_id: str, event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    for q in list(_subscribers.get(job_id, [])):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


# ─── Agent Builder ────────────────────────────────────────────────────────────

def build_agent(model_name=None):
    llm = ChatOllama(
        model=model_name or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=AGENT_TEMPERATURE,
        # thinking={"type": "enabled", "budget_tokens": 1000},
    )
    return create_agent(llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


# ─── Chunk Normaliser ─────────────────────────────────────────────────────────

def _normalise_chunk(raw) -> tuple[str, any]:
    """
    Normalise a raw stream chunk into (mode, data) regardless of version.

    v2  → dict {"type": "updates"|"messages", "data": ...}
    v1  → tuple ("updates"|"messages", data)
    """
    if isinstance(raw, dict):
        return raw.get("type", ""), raw.get("data")
    elif isinstance(raw, tuple) and len(raw) == 2:
        return raw[0], raw[1]
    else:
        log.warning("Unknown chunk format: %s (type=%s)", repr(raw)[:200], type(raw).__name__)
        return "", None


# ─── Chunk Processor ─────────────────────────────────────────────────────────

def _process_chunk(raw) -> list[dict]:
    """Convert a raw stream chunk into a list of typed UI events."""
    events = []
    mode, data = _normalise_chunk(raw)

    if not mode or data is None:
        return events

    # ── "messages" mode: individual tokens as the model generates them ───────
    if mode == "messages":
        # v2: data is (token, metadata)
        # v1: data might be a message directly or also (token, metadata)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            token, metadata = data
            node = metadata.get("langgraph_node", "model") if isinstance(metadata, dict) else "model"
        else:
            token = data
            node = "model"

        if not isinstance(token, AIMessageChunk):
            return events

        content = token.content

        if isinstance(content, str) and content:
            events.append({"type": "token", "text": content, "node": node})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    events.append({"type": "token", "text": block["text"], "node": node})
                elif btype in ("reasoning", "thinking"):
                    text = block.get(btype) or block.get("text", "")
                    if text:
                        events.append({"type": "token_reasoning", "text": text, "node": node})

        for tc_chunk in getattr(token, "tool_call_chunks", []) or []:
            if isinstance(tc_chunk, dict) and tc_chunk.get("name"):
                events.append({"type": "token_tool_name", "tool": tc_chunk["name"], "node": node})

    # ── "updates" mode: completed step outputs ───────────────────────────────
    elif mode == "updates":
        if not isinstance(data, dict):
            return events

        for node_name, update in data.items():
            if node_name == "__interrupt__":
                continue
            if not isinstance(update, dict):
                continue

            messages = update.get("messages", [])
            if not messages:
                continue
            msg = messages[-1]

            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    events.append({"type": "token_end"})
                    for tc in msg.tool_calls:
                        events.append({
                            "type": "tool_call",
                            "tool": tc.get("name", "unknown"),
                            "args": tc.get("args", {}),
                        })
                else:
                    raw_content = msg.content
                    if isinstance(raw_content, list):
                        text = "".join(
                            b.get("text", "") for b in raw_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        text = str(raw_content) if raw_content else ""

                    if text.strip():
                        events.append({"type": "token_end"})
                        events.append({"type": "final_answer", "message": text.strip()})

            elif isinstance(msg, ToolMessage):
                content = str(msg.content).strip()
                success = not any(
                    content.lower().startswith(kw)
                    for kw in ("error", "fail", "could not", "trim failed",
                               "merge failed", "conversion failed")
                )
                events.append({"type": "tool_result", "success": success, "message": content})

    return events


# ─── Job Runner ───────────────────────────────────────────────────────────────

async def run_job(job_id: str, model_name=None):
    """
    Execute a video editing job with token-level streaming.

    agent.stream() is blocking, so it runs in a daemon thread.
    A thread-safe bridge queue moves events back to the asyncio loop.
    """
    job = job_manager.get_job(job_id)
    if not job:
        log.error("run_job called with unknown job_id=%s", job_id)
        return

    effective_model = model_name or OLLAMA_MODEL
    log.info("Starting job %s  model=%s  prompt=%r", job_id[:8], effective_model, job.prompt[:80])

    job_manager.mark_running(job_id)
    _emit(job_id, {
        "type": "status",
        "status": "running",
        "message": f"Job started — model: {effective_model}",
        "model": effective_model,
    })

    # Validate input file
    input_path = resolve_input_path(job.input_file)
    valid, err = validate_input_file(input_path)
    if not valid:
        msg = f"Input file error: {err}"
        log.error("Job %s — %s", job_id[:8], msg)
        job_manager.mark_failed(job_id, msg)
        _emit(job_id, {"type": "status", "status": "failed", "message": msg})
        _emit(job_id, {"type": "done"})
        return

    _emit(job_id, {"type": "info", "message": f"Input validated: {input_path}"})
    log.info("Job %s — input OK: %s", job_id[:8], input_path)

    user_message = (
        f"{job.prompt}\n\n"
        f"Input file: {input_path}\n"
        f"Job ID: {job_id}\n"
        f"Use the job_id '{job_id}' as the job_id argument in all tool calls."
    )

    # Thread-safe bridge: blocking thread → asyncio event loop
    loop = asyncio.get_event_loop()
    bridge: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _stream_thread():
        try:
            log.info("Job %s — building agent (model=%s)", job_id[:8], effective_model)
            agent = build_agent(model_name)

            log.info("Job %s — starting agent.stream()", job_id[:8])
            chunk_count = 0

            for raw_chunk in agent.stream(
                {"messages": [{"role": "user", "content": user_message}]},
                stream_mode=["updates", "messages"],
                version="v2",
            ):
                chunk_count += 1
                log.debug("Job %s — chunk #%d type=%s",
                          job_id[:8], chunk_count, type(raw_chunk).__name__)

                try:
                    events = _process_chunk(raw_chunk)
                    for ev in events:
                        loop.call_soon_threadsafe(bridge.put_nowait, ev)
                except Exception as chunk_err:
                    tb = traceback.format_exc()
                    log.error("Job %s — error processing chunk #%d:\n%s",
                              job_id[:8], chunk_count, tb)
                    # Surface chunk-level errors to the dashboard too
                    loop.call_soon_threadsafe(bridge.put_nowait, {
                        "type": "agent_error",
                        "message": f"Chunk processing error: {chunk_err}",
                        "traceback": tb,
                    })

            log.info("Job %s — stream finished (%d chunks)", job_id[:8], chunk_count)

        except Exception as e:
            tb = traceback.format_exc()
            log.error("Job %s — agent stream failed:\n%s", job_id[:8], tb)
            loop.call_soon_threadsafe(bridge.put_nowait, {
                "type": "status",
                "status": "failed",
                "message": str(e),
                "traceback": tb,
            })
        finally:
            loop.call_soon_threadsafe(bridge.put_nowait, _SENTINEL)

    threading.Thread(target=_stream_thread, daemon=True).start()

    # Consume bridge, forward to WebSocket, persist to job trace
    output_path = None
    final_answer = None

    while True:
        event = await bridge.get()
        if event is _SENTINEL:
            break

        _emit(job_id, event)
        etype = event.get("type", "")

        if etype == "tool_call":
            log.info("Job %s — tool_call: %s  args=%s",
                     job_id[:8], event.get("tool"), event.get("args"))
            job_manager.add_step(job_id, AgentStep(
                step_number=0, type="tool_call",
                content=f"Called: {event.get('tool')}",
                tool_name=event.get("tool"),
                tool_args=event.get("args"),
            ))

        elif etype == "tool_result":
            content = event.get("message", "")
            log.info("Job %s — tool_result (success=%s): %s",
                     job_id[:8], event.get("success"), content[:120])
            job_manager.add_step(job_id, AgentStep(
                step_number=0, type="tool_result", content=content,
            ))
            if "saved to:" in content.lower():
                m = re.search(r"(?:saved to:|output saved to:)\s*(.+)", content, re.IGNORECASE)
                if m:
                    output_path = m.group(1).strip()
                    log.info("Job %s — output detected: %s", job_id[:8], output_path)

        elif etype == "final_answer":
            final_answer = event.get("message", "")
            log.info("Job %s — final_answer: %s", job_id[:8], final_answer[:120])
            job_manager.add_step(job_id, AgentStep(
                step_number=0, type="final_answer", content=final_answer,
            ))

        elif etype == "agent_error":
            # Non-fatal chunk error — logged already, shown in dashboard
            log.warning("Job %s — non-fatal agent_error: %s", job_id[:8], event.get("message"))

        elif etype == "status" and event.get("status") == "failed":
            log.error("Job %s — marked FAILED: %s", job_id[:8], event.get("message"))
            job_manager.mark_failed(job_id, event.get("message", "Unknown error"))
            _emit(job_id, {"type": "done"})
            return

    # Finalise
    log.info("Job %s — completing. output_path=%s", job_id[:8], output_path)
    job_manager.mark_done(
        job_id,
        output_file=output_path or "",
        result_message=final_answer or "Task completed.",
    )
    _emit(job_id, {
        "type": "status",
        "status": "done",
        "message": final_answer or "Task completed.",
        "output_file": output_path,
    })
    _emit(job_id, {"type": "done"})