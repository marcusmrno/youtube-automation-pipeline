import pytest
import sys
from unittest.mock import MagicMock

# Ensure claude_agent_sdk is mocked before importing pipeline
if "claude_agent_sdk" not in sys.modules:
    mock_sdk = MagicMock()
    sys.modules["claude_agent_sdk"] = mock_sdk
    sys.modules["claude_agent_sdk.types"] = mock_sdk

from pipeline import parse_image_prompts


def test_parse_new_format_source_field():
    raw = "001 | The cat explains photosynthesis | Cat at a blackboard drawing a leaf, chalk lines"
    result = parse_image_prompts(raw)
    assert len(result) == 1
    assert result[0]["num"] == "001"
    assert result[0]["source"] == "The cat explains photosynthesis"
    assert result[0]["prompt"] == "Cat at a blackboard drawing a leaf, chalk lines"


def test_parse_new_format_no_ts_key():
    raw = "001 | source line here | some prompt"
    result = parse_image_prompts(raw)
    assert "ts" not in result[0]


def test_parse_multiple_prompts_sorted():
    raw = (
        "003 | third line | third prompt\n"
        "001 | first line | first prompt\n"
        "002 | second line | second prompt\n"
    )
    result = parse_image_prompts(raw)
    assert [p["num"] for p in result] == ["001", "002", "003"]


def test_parse_deduplicates_by_num_keeps_last():
    raw = (
        "001 | first version source | first version prompt\n"
        "001 | second version source | second version prompt\n"
    )
    result = parse_image_prompts(raw)
    assert len(result) == 1
    assert result[0]["source"] == "second version source"


def test_parse_skips_lines_without_three_parts():
    raw = "001 | only two parts\nnot a prompt line\n002 | source | prompt"
    result = parse_image_prompts(raw)
    assert len(result) == 1
    assert result[0]["num"] == "002"


def test_parse_pipe_in_prompt_body_preserved():
    raw = "001 | source line | prompt with | pipe inside it"
    result = parse_image_prompts(raw)
    assert result[0]["prompt"] == "prompt with | pipe inside it"


def test_parse_empty_string_returns_empty():
    assert parse_image_prompts("") == []
