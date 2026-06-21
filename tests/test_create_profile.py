import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from create_profile import next_version_name
from create_profile_image_gen import generate_anchor


def test_base_name_gets_v2():
    assert next_version_name("my-channel") == "my-channel-v2"


def test_v2_gets_v3():
    assert next_version_name("my-channel-v2") == "my-channel-v3"


def test_v9_gets_v10():
    assert next_version_name("my-channel-v9") == "my-channel-v10"


def test_name_ending_in_number_not_version():
    # "channel-2" is not a version slug — treat as base
    assert next_version_name("channel-2") == "channel-2-v2"


def test_generate_anchor_returns_false_on_exception(tmp_path):
    profile_yaml = {
        "image_gen": {"default_model": "gemini-3.1-flash-image"},
        "image_style": {"anchor_priority": [], "max_anchors": 14},
    }
    with patch("create_profile_image_gen.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content.side_effect = Exception("API error")
        result = generate_anchor("test prompt", tmp_path / "anchor-01.png", profile_yaml, tmp_path)
    assert result is False
