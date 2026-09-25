import pytest

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



# ── agents.split_agent_output ──────────────────────────────────────────────────

def test_split_agent_output_separates_research_from_script():
    from agents import split_agent_output
    raw = (
        "Keyword research: 'why we dream' — 40k/mo, low competition.\n"
        "Top outlier reframes it as a survival mechanism.\n"
        "===SCRIPT===\n"
        "TITLE: Why You Dream\n[00:00-00:35] HOOK\nYour brain never sleeps."
    )
    script, notes = split_agent_output(raw)
    assert script.startswith("TITLE: Why You Dream")
    assert "40k/mo" in notes
    assert "===SCRIPT===" not in notes


def test_split_agent_output_no_notes_when_script_only():
    from agents import split_agent_output
    script, notes = split_agent_output("===SCRIPT===\nTITLE: X")
    assert script == "TITLE: X"
    assert notes == ""


@pytest.mark.parametrize("tag,text,expected", [
    ("SCRIPT", "===SCRIPT===\nA\n=== Part 2 ===\nB", "A\n=== Part 2 ===\nB"),       # section divider in the script
    ("IMAGE_PROMPTS", "===IMAGE_PROMPTS===\n001 | a\n======\n002 | b", "001 | a\n======\n002 | b"),
    ("SCRIPT", "I'll put it under ===SCRIPT=== below.\n===SCRIPT===\nbody", "body"),  # tag mentioned first
    ("SCRIPT", "=== SCRIPT ===\nbody", "body"),                                     # spaced tag
    ("SCRIPT", "===script===\nbody", "body"),                                       # lower-case tag
    ("SCRIPT", "**===SCRIPT===**\nTITLE: X", "TITLE: X"),                           # bold tag
    ("TITLES", "===TITLES===\n1. a\n===END===", "1. a"),
    ("THUMBNAIL_1", "===THUMBNAIL_1===\nHOOK: x\nbody1\n===THUMBNAIL_2===\nbody2", "HOOK: x\nbody1"),
    ("SCRIPT", "no tags here", ""),
])
def test_extract(tag, text, expected):
    from prompts import _extract
    assert _extract(tag, text) == expected
