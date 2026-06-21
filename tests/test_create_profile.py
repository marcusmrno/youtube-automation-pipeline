import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from create_profile import next_version_name
from create_profile_image_gen import generate_anchor
from create_profile_claude import extract_fenced_block


def test_base_name_gets_v2():
    assert next_version_name("my-channel") == "my-channel-v2"


def test_v2_gets_v3():
    assert next_version_name("my-channel-v2") == "my-channel-v3"


def test_v9_gets_v10():
    assert next_version_name("my-channel-v9") == "my-channel-v10"


def test_name_ending_in_number_not_version():
    # "channel-2" is not a version slug — treat as base
    assert next_version_name("channel-2") == "channel-2-v2"


def test_extract_yaml_block():
    text = 'Some text\n```yaml\nkey: value\n```\nMore text'
    assert extract_fenced_block(text, "yaml") == "key: value"


def test_extract_missing_block_returns_none():
    text = "No fenced blocks here"
    assert extract_fenced_block(text, "yaml") is None


def test_extract_markdown_block():
    text = '```markdown\n# Title\n```'
    assert extract_fenced_block(text, "markdown") == "# Title"


from create_profile_anchors import run_full_anchors


def test_run_full_anchors_skips_existing(tmp_path):
    anchors_dir = tmp_path / "anchors"
    anchors_dir.mkdir()
    # Pre-create anchor-03.png so it should be skipped
    (anchors_dir / "anchor-03.png").write_bytes(b"fake")

    profile_yaml = {
        "image_gen": {"default_model": "gemini-3.1-flash-image"},
        "image_style": {"anchor_priority": ["anchor-01", "anchor-02", "anchor-03"], "max_anchors": 14},
    }
    prompts = ["prompt1", "prompt2", "prompt3"]

    with patch("create_profile_anchors.generate_anchor", return_value=True) as mock_gen:
        result = run_full_anchors(profile_yaml, anchors_dir, prompts, start_from=3)

    # anchor-03 already exists, should not be generated
    called_names = [call.args[1].name for call in mock_gen.call_args_list]
    assert not any("anchor-03" in n for n in called_names)


def test_generate_anchor_returns_false_on_exception(tmp_path):
    profile_yaml = {
        "image_gen": {"default_model": "gemini-3.1-flash-image"},
        "image_style": {"anchor_priority": [], "max_anchors": 14},
    }
    with patch("create_profile_image_gen.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content.side_effect = Exception("API error")
        result = generate_anchor("test prompt", tmp_path / "anchor-01.png", profile_yaml, tmp_path)
    assert result is False
