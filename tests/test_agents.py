import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import agents


def test_unpatched_agent_call_fails_loudly():
    # conftest stubs the SDK; reaching it must fail the test, not quietly return ""
    with pytest.raises(RuntimeError, match="patch the agent call"):
        asyncio.run(agents.run_vidiq_agent("sys", "user", 1, lambda m: None))


# Runs in a subprocess: conftest stubs the SDK in-process, and this needs the real
# command builder. It captures the options run_vidiq_agent passes; nothing is spawned.
_BUILD_CMD = r'''
import asyncio, json
from unittest import mock
import agents
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

seen = {}
async def fake_query(prompt, options):
    seen["options"] = options
    return
    yield

with mock.patch.object(agents, "agent_query", fake_query):
    asyncio.run(agents.run_vidiq_agent("sys", "user", 1, lambda m: None))
t = SubprocessCLITransport(prompt="x", options=seen["options"])
t._cli_path = "claude"
print(json.dumps(t._build_command()))
'''


def test_vidiq_agent_cli_gets_no_builtin_tools_or_user_settings():
    # the agent reads third-party YouTube text; it must not get Bash/Write/WebFetch,
    # the user's hooks and CLAUDE.md, or other configured MCP servers
    env = {**os.environ, "ANTHROPIC_API_KEY": "", "VIDIQ_API_KEY": "", "GOOGLE_API_KEY": "",
           "ELEVENLABS_API_KEY": "", "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_USER_ID": "0"}
    out = subprocess.run([sys.executable, "-c", _BUILD_CMD], cwd=Path(agents.__file__).parent,
                         env=env, capture_output=True, text=True, check=True).stdout
    cmd = json.loads(out.splitlines()[-1])
    assert "--tools" in cmd and cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert "--setting-sources=" in cmd


class _Text:          # stand-ins for the SDK message types (conftest stubs the SDK)
    def __init__(self, text): self.text = text


class _Assistant:
    def __init__(self, text): self.content = [_Text(text)]


class _Result:
    def __init__(self, result): self.result = result


def test_final_agent_message_is_counted_once(monkeypatch):
    final = "Keyword 'why cats purr' is strong. Here is the script:\n===SCRIPT===\nTITLE: X\nbody"

    async def fake_query(prompt, options):
        yield _Assistant("Calling vidIQ tools.")
        yield _Assistant(final)
        yield _Result(final)   # the CLI repeats the last assistant text as the result

    monkeypatch.setattr(agents, "agent_query", fake_query)
    monkeypatch.setattr(agents, "AssistantMessage", _Assistant)
    monkeypatch.setattr(agents, "TextBlock", _Text)
    monkeypatch.setattr(agents, "ResultMessage", _Result)
    raw = asyncio.run(agents.run_vidiq_agent("sys", "user", 1, lambda m: None))
    script, notes = agents.split_agent_output(raw)
    assert script == "TITLE: X\nbody"   # no preamble glued onto the end of the script
    assert notes == "Calling vidIQ tools.\nKeyword 'why cats purr' is strong. Here is the script:"
