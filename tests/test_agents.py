import asyncio

import pytest

import agents


def test_unpatched_agent_call_fails_loudly():
    # conftest stubs the SDK; reaching it must fail the test, not quietly return ""
    with pytest.raises(RuntimeError, match="patch the agent call"):
        asyncio.run(agents.run_vidiq_agent("sys", "user", 1, lambda m: None))
