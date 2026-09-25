"""
conftest.py — stub the Claude Agent SDK so no test can spawn a real (billed) agent.
"""
import sys
from unittest.mock import MagicMock

# claude-agent-sdk is a real dependency (requirements.txt), but calling it spawns Claude.
# ponytail: loud stub — an unpatched agent call fails the test instead of returning "".
mock_sdk = MagicMock()
mock_sdk.query.side_effect = RuntimeError("test reached claude_agent_sdk.query — patch the agent call")
sys.modules["claude_agent_sdk"] = mock_sdk
sys.modules["claude_agent_sdk.types"] = mock_sdk
