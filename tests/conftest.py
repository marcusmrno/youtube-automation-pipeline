"""
conftest.py — blank the API keys and stub the Claude Agent SDK so no test can make a paid call.
"""
import os
import sys
from unittest.mock import MagicMock

# Blank every key before any module imports: load_dotenv never overrides a variable that is
# already set, so a test that forgets to fake a paid call fails authentication instead of billing.
for _key in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID",
             "VIDIQ_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID"):
    os.environ[_key] = ""

# claude-agent-sdk is a real dependency (requirements.txt), but calling it spawns Claude.
# ponytail: loud stub — an unpatched agent call fails the test instead of returning "".
mock_sdk = MagicMock()
mock_sdk.query.side_effect = RuntimeError("test reached claude_agent_sdk.query — patch the agent call")
sys.modules["claude_agent_sdk"] = mock_sdk
sys.modules["claude_agent_sdk.types"] = mock_sdk
