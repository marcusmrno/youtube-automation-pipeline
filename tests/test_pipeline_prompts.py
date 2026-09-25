import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_profile():
    from profile import load_profile
    return load_profile("test-channel", profiles_root=FIXTURES)


def test_generate_image_google_uses_profile_anchors(test_profile, tmp_path):
    """generate_image_google should load anchors from profile.anchors_dir, not style-anchors/."""
    calls = []

    def fake_generate(model, contents, config):
        calls.append(contents)
        raise RuntimeError("stop after capture")

    import pipeline
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = fake_generate
    # patch the cached client getter itself: patching pipeline.genai misses once it's cached
    with patch.object(pipeline, "_get_genai_client", return_value=mock_client), \
         patch.object(pipeline.time, "sleep"):
        pipeline.generate_image_google("test prompt", tmp_path / "001.png", test_profile, print)

    contents = calls[0]
    assert isinstance(contents[0], str) and contents[0].endswith("test prompt")
    # the fixture's two anchors ride along as image parts
    assert len(contents) == 3 and not any(isinstance(c, str) for c in contents[1:])


def test_build_script_prompt_injects_channel_identity(test_profile):
    from pipeline import _build_script_prompt
    prompt = _build_script_prompt("test topic", "fake research", test_profile)
    assert test_profile.channel["niche"] in prompt
    assert test_profile.channel["audience"] in prompt
    assert test_profile.channel["tone"] in prompt
    assert test_profile.channel["reference_channel"] in prompt
    assert str(test_profile.script["target_mins"]) in prompt
    assert str(test_profile.script["wpm"]) in prompt


def test_build_tts_prompt_injects_tone_description(test_profile):
    from pipeline import _build_tts_prompt
    prompt = _build_tts_prompt("Some script text here.", test_profile)
    assert test_profile.voice["tone_description"] in prompt
    assert "Some script text here." in prompt


def test_build_image_prompt_instructions_injects_characters(test_profile):
    from pipeline import _build_image_prompt_instructions
    instructions = _build_image_prompt_instructions(test_profile)
    assert "Test Cat" in instructions
    assert "round head" in instructions
    assert "Other Cat" in instructions
    assert test_profile.image_style["art_style_block"].strip() in instructions


def test_build_agent_script_prompt_injects_channel_identity(test_profile):
    from prompts import _build_agent_script_prompt
    prompt = _build_agent_script_prompt(test_profile)
    assert test_profile.channel["niche"] in prompt
    assert test_profile.channel["tone"] in prompt
    assert str(test_profile.script["target_mins"]) in prompt


def test_build_agent_system_prompt_uses_profile(test_profile):
    from pipeline import _build_agent_system_prompt
    prompt = _build_agent_system_prompt("test topic", test_profile)
    assert test_profile.channel["niche"] in prompt
    assert "test topic" in prompt


def test_generate_voiceover_uses_profile_voice_settings(test_profile, tmp_path):
    """generate_voiceover should use voice_id and settings from profile."""
    (tmp_path / "audio").mkdir()

    captured_payloads = []

    def fake_post(url, headers, params, json, timeout):
        captured_payloads.append(json)
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.content = b"ID3" + b"\x00" * 100
        return mock_resp

    import pipeline
    with patch("pipeline.requests.post", side_effect=fake_post):
        pipeline.generate_voiceover("Hello world test.", tmp_path, test_profile, print)

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["model_id"] == "eleven_v3"
    assert payload["voice_settings"]["stability"] == 0.68
