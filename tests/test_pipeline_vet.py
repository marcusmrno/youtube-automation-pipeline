import json
import sys
from unittest.mock import MagicMock, patch, call

if "claude_agent_sdk" not in sys.modules:
    mock_sdk = MagicMock()
    sys.modules["claude_agent_sdk"] = mock_sdk
    sys.modules["claude_agent_sdk.types"] = mock_sdk

from pipeline import _vet_image_prompts


def _make_prompts(n=5):
    return [
        {"num": f"{i:03d}", "source": f"source line {i}", "prompt": f"prompt text {i}"}
        for i in range(1, n + 1)
    ]


def _mock_profile():
    p = MagicMock()
    p.characters_block.return_value = "Anchor cat: orange tabby"
    p.image_style = {"art_style_block": "flat vector art style", "scene_rules": ""}
    return p


def test_vet_returns_original_when_nothing_flagged():
    prompts = _make_prompts(3)
    profile = _mock_profile()
    client  = MagicMock()

    # Stage 1: Haiku returns empty array (nothing flagged)
    stage1_resp = MagicMock()
    stage1_resp.content = [MagicMock(text="[]")]
    client.messages.create.return_value = stage1_resp

    result = _vet_image_prompts(prompts, profile, client, lambda m: None)

    assert result == prompts
    assert client.messages.create.call_count == 1  # only stage 1 ran


def test_vet_rewrites_flagged_prompts():
    prompts = _make_prompts(3)
    profile = _mock_profile()
    client  = MagicMock()

    # Stage 1: flag prompt 002
    flagged = json.dumps([{"num": "002", "reason": "RELEVANCE", "detail": "wrong content"}])
    stage1_resp = MagicMock()
    stage1_resp.content = [MagicMock(text=flagged)]

    # Stage 2: return rewritten prompt 002
    stage2_resp = MagicMock()
    stage2_resp.content = [MagicMock(text="002 | source line 2 | rewritten prompt for 002")]

    client.messages.create.side_effect = [stage1_resp, stage2_resp]

    result = _vet_image_prompts(prompts, profile, client, lambda m: None)

    assert client.messages.create.call_count == 2
    assert result[1]["prompt"] == "rewritten prompt for 002"
    assert result[1]["source"] == "source line 2"
    assert result[0]["prompt"] == "prompt text 1"  # untouched
    assert result[2]["prompt"] == "prompt text 3"  # untouched


def test_vet_returns_original_on_invalid_stage1_json():
    prompts = _make_prompts(2)
    profile = _mock_profile()
    client  = MagicMock()

    stage1_resp = MagicMock()
    stage1_resp.content = [MagicMock(text="not valid json")]
    client.messages.create.return_value = stage1_resp

    logs = []
    result = _vet_image_prompts(prompts, profile, client, logs.append)

    assert result == prompts
    assert any("⚠️" in m for m in logs)


def test_vet_returns_original_on_empty_stage2_response():
    prompts = _make_prompts(2)
    profile = _mock_profile()
    client  = MagicMock()

    flagged = json.dumps([{"num": "001", "reason": "RELEVANCE", "detail": "bad"}])
    stage1_resp = MagicMock()
    stage1_resp.content = [MagicMock(text=flagged)]

    stage2_resp = MagicMock()
    stage2_resp.content = [MagicMock(text="")]
    client.messages.create.side_effect = [stage1_resp, stage2_resp]

    logs = []
    result = _vet_image_prompts(prompts, profile, client, logs.append)

    assert result == prompts
    assert any("⚠️" in m for m in logs)


def test_vet_handles_missing_keys_in_flagged_entry():
    """Stage 1 returns valid JSON but entries missing num/reason — should not crash."""
    prompts = _make_prompts(3)
    profile = _mock_profile()
    client  = MagicMock()

    # Entry with missing "num" key — should be skipped gracefully
    flagged = json.dumps([{"reason": "RELEVANCE", "detail": "bad"}])
    stage1_resp = MagicMock()
    stage1_resp.content = [MagicMock(text=flagged)]

    # Stage 2 won't run since no valid flagged lines built
    client.messages.create.return_value = stage1_resp

    logs = []
    result = _vet_image_prompts(prompts, profile, client, logs.append)

    # Should return originals without crashing (skipped the malformed entry)
    assert result == prompts
