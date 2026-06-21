"""
conftest.py — mock unavailable optional dependencies before any test imports pipeline.
"""
import sys
from unittest.mock import MagicMock

# claude_agent_sdk is a private/internal package not available on PyPI.
# Mock it so pipeline.py can be imported in tests.
if "claude_agent_sdk" not in sys.modules:
    mock_sdk = MagicMock()
    sys.modules["claude_agent_sdk"] = mock_sdk
    sys.modules["claude_agent_sdk.types"] = mock_sdk
