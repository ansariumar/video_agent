"""
Agent Core.
Builds the LangChain agent using create_agent() + ChatOllama.
Runs jobs asynchronously and writes the reasoning trace to the job manager.
"""

import asyncio
from datetime import datetime
from typing import Optional

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage, HumanMessage

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, AGENT_TEMPERATURE, AGENT_MAX_ITERATIONS
from tools.registry import ALL_TOOLS
from jobs.manager import job_manager
from jobs.models import AgentStep
from agent.prompts import SYSTEM_PROMPT
from utils.file_utils import resolve_input_path, validate_input_file


def build_agent(model_name: Optional[str] = None):
    """
    Build and return a LangChain agent instance.
    Optionally override the model name at runtime (for model switching).
    """
    llm = ChatOllama(
        model=model_name or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=AGENT_TEMPERATURE,
    )

    agent = create_agent(
        llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


async def run_job(job_id: str, model_name: Optional[str] = None):
    """
    Execute a video editing job asynchronously.

    1. Marks the job as running.
    2. Builds the agent (with optional model override).
    3. Invokes the agent with the job prompt + input file.
    4. Streams steps into the job's reasoning trace.
    5. Marks the job done or failed.
    """
    job = job_manager.get_job(job_id)
    if not job:
        return

    job_manager.mark_running(job_id)

    # Resolve and validate the input file
    input_path = resolve_input_path(job.input_file)
    valid, err = validate_input_file(input_path)
    if not valid:
        job_manager.mark_failed(job_id, f"Input file error: {err}")
        return

    # Build the user message — include job_id so tools can name outputs correctly
    user_message = (
        f"{job.prompt}\n\n"
        f"Input file: {input_path}\n"
        f"Job ID: {job_id}\n"
        f"Use the job_id '{job_id}' as the job_id argument in all tool calls."
    )

    try:
        agent = build_agent(model_name)

        # Run agent in a thread so it doesn't block the async event loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
            )
        )

        # Parse the message trace and write steps to job manager
        messages = result.get("messages", [])
        output_path = None
        final_answer = None

        for msg in messages:
            if isinstance(msg, HumanMessage):
                continue  # Skip the original user message

            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        job_manager.add_step(job_id, AgentStep(
                            step_number=0,  # will be set by manager
                            type="tool_call",
                            content=f"Calling tool: {tc['name']}",
                            tool_name=tc["name"],
                            tool_args=tc.get("args", {}),
                        ))
                elif msg.content:
                    final_answer = msg.content
                    job_manager.add_step(job_id, AgentStep(
                        step_number=0,
                        type="final_answer",
                        content=msg.content,
                    ))

            elif isinstance(msg, ToolMessage):
                content = str(msg.content)
                job_manager.add_step(job_id, AgentStep(
                    step_number=0,
                    type="tool_result",
                    content=content,
                ))
                # Extract output path from successful tool results
                if "saved to:" in content.lower() or "output saved to:" in content.lower():
                    import re
                    match = re.search(r"(?:saved to:|output saved to:)\s*(.+)", content, re.IGNORECASE)
                    if match:
                        output_path = match.group(1).strip()

        if output_path:
            job_manager.mark_done(
                job_id,
                output_file=output_path,
                result_message=final_answer or "Task completed successfully.",
            )
        else:
            # Agent finished but no output file detected — still mark done with message
            job_manager.mark_done(
                job_id,
                output_file="",
                result_message=final_answer or "Task completed. Check agent steps for details.",
            )

    except Exception as e:
        job_manager.mark_failed(job_id, error=f"Agent error: {str(e)}")
