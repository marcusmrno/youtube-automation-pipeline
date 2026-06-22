"""
vidIQ agent runners — Claude agents that use the vidIQ MCP for research and vetting.
"""
from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from claude_agent_sdk import query as agent_query, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock
from dotenv import load_dotenv

from prompts import (
    _build_agent_system_prompt,
    _build_agent_script_prompt,
    _build_vet_prompt,
    _extract,
)

if TYPE_CHECKING:
    from profile import Profile

load_dotenv()

VIDIQ_MCP_URL = "https://mcp.vidiq.com/mcp"
_VIDIQ_KEY    = (os.getenv("VIDIQ_API_KEY") or "").strip()


VET_MODEL = "claude-haiku-4-5-20251001"


def _vidiq_options(max_turns: int, model: str | None = None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        mcp_servers={
            "vidiq": {
                "type": "http",
                "url": VIDIQ_MCP_URL,
                "headers": {"Authorization": f"Bearer {_VIDIQ_KEY}"},
            }
        },
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model,
    )


async def _run_agent(
    system_prompt: str,
    user_prompt: str,
    max_turns: int,
    extract_tag: str,
    log_fn,
    model: str | None = None,
) -> str:
    """Core agent loop — streams messages and returns extracted tagged section."""
    options = _vidiq_options(max_turns, model=model)
    options.system_prompt = system_prompt

    full_text = ""
    async for message in agent_query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    log_fn(f"  {block.text[:120].strip()}")
                    full_text += block.text
        elif isinstance(message, ResultMessage):
            if message.result:
                full_text += message.result

    return _extract(extract_tag, full_text)


async def _run_script_agent(topic: str, profile: "Profile", log_fn) -> str:
    system_prompt = _build_agent_system_prompt(topic, profile)
    user_prompt   = f"Topic: {topic}\n\n{_build_agent_script_prompt(profile)}"
    return await _run_agent(system_prompt, user_prompt, max_turns=30, extract_tag="SCRIPT", log_fn=log_fn)


async def _run_vet_agent(topic: str, script: str, profile: "Profile", log_fn) -> str:
    system_prompt = _build_agent_system_prompt(topic, profile) + f"\n\nCURRENT SCRIPT TO VET:\n{script}"
    user_prompt   = f"Topic: {topic}\n\n{_build_vet_prompt(profile)}"
    return await _run_agent(system_prompt, user_prompt, max_turns=20, extract_tag="SCRIPT", log_fn=log_fn, model=VET_MODEL)


def run_script_agent(topic: str, profile: "Profile", log_fn) -> str:
    return asyncio.run(_run_script_agent(topic, profile, log_fn))


def run_vet_agent(topic: str, script: str, profile: "Profile", log_fn) -> str:
    return asyncio.run(_run_vet_agent(topic, script, profile, log_fn))
