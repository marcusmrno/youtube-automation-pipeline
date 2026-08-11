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
VIDIQ_KEY     = (os.getenv("VIDIQ_API_KEY") or "").strip()


SCRIPT_MODEL = "claude-opus-5"
VET_MODEL    = "claude-haiku-4-5-20251001"


async def run_vidiq_agent(
    system_prompt: str,
    user_prompt: str,
    max_turns: int,
    log_fn,
    model: str | None = None,
) -> str:
    """Core vidIQ agent loop — streams messages and returns the concatenated text."""
    options = ClaudeAgentOptions(
        mcp_servers={
            "vidiq": {
                "type": "http",
                "url": VIDIQ_MCP_URL,
                "headers": {"Authorization": f"Bearer {VIDIQ_KEY}"},
            }
        },
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model,
    )
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
    return full_text


def split_agent_output(raw: str) -> tuple[str, str]:
    """Split agent output into (script, research_notes).

    Notes are whatever the agent said before the SCRIPT block — its vidIQ
    findings (keywords, outliers, title scores). Empty if it went straight
    to the script.
    """
    return _extract("SCRIPT", raw), raw.split("===SCRIPT===")[0].strip()


def run_script_agent(topic: str, profile: "Profile", log_fn, approach_context: str = "") -> tuple[str, str]:
    system_prompt = _build_agent_system_prompt(topic, profile)
    user_prompt   = f"Topic: {topic}\n\n{_build_agent_script_prompt(profile, approach_context)}"
    raw = asyncio.run(run_vidiq_agent(system_prompt, user_prompt, max_turns=30, log_fn=log_fn, model=SCRIPT_MODEL))
    return split_agent_output(raw)


def run_vet_agent(topic: str, script: str, profile: "Profile", log_fn) -> str:
    system_prompt = _build_agent_system_prompt(topic, profile) + f"\n\nCURRENT SCRIPT TO VET:\n{script}"
    user_prompt   = f"Topic: {topic}\n\n{_build_vet_prompt(profile)}"
    raw = asyncio.run(run_vidiq_agent(system_prompt, user_prompt, max_turns=20, log_fn=log_fn, model=VET_MODEL))
    return _extract("SCRIPT", raw)
