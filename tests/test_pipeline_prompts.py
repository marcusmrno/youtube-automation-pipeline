import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_profile():
    from channel_profile import load_profile
    return load_profile("test-channel", profiles_root=FIXTURES)


def test_generate_image_google_uses_profile_anchors(test_profile, tmp_path):
    """generate_image_google should load anchors from profile.anchors_dir, not style-anchors/."""
    calls = []

    def fake_generate(model, contents, config):
        calls.append(contents)
        raise RuntimeError("stop after capture")

    import images
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = fake_generate
    # patch the cached client getter itself: patching images.genai misses once it's cached
    with patch.object(images, "_get_genai_client", return_value=mock_client), \
         patch.object(images.time, "sleep"):
        images.generate_image_google("test prompt", tmp_path / "001.png", test_profile, print)

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


def test_image_prompt_request_does_not_ask_for_timestamps(test_profile, monkeypatch):
    # the line format has no timestamp column; asking for one shifts every field when obeyed
    import re
    import pipeline
    sent = []
    monkeypatch.setattr(pipeline, "_stream_text", lambda client, **k: sent.append(k["messages"][0]["content"])
                        or "===IMAGE_PROMPTS===\n001 | s | none | scene")
    pipeline._generate_image_prompts("SCRIPT BODY", test_profile, None, lambda m: None)
    assert not re.search(r"(?i)timestamps (are|must)|contiguous|segment of the script", sent[0])


def test_script_agent_and_vet_prompts_share_one_word_range(test_profile):
    # the vet (Haiku, rewriting Opus's script) capped narration at the target the agent may exceed
    from prompts import _word_budget, _build_script_prompt, _build_agent_script_prompt, _build_vet_prompt
    _, min_words, max_words, *_ = _word_budget(test_profile.script)
    floor = round(min_words * 1.1)
    assert f"under {floor}" in _build_script_prompt("t", "research", test_profile)
    assert f"under {floor}" in _build_agent_script_prompt(test_profile)
    assert f"{floor}-{max_words} words" in _build_vet_prompt(test_profile)


def test_prompts_follow_the_profiles_durations(test_profile):
    from dataclasses import replace
    from prompts import (_build_script_prompt, _build_agent_script_prompt, _build_tts_prompt,
                         _build_image_prompt_instructions)
    p = replace(test_profile, script={**test_profile.script, "hook_duration_s": 75, "target_mins": 8})
    for hook_prompt in (_build_script_prompt("t", "r", p), _build_agent_script_prompt(p)):
        assert "0:75" not in hook_prompt and "1:15" in hook_prompt      # a 75 s hook ends at 1:15
    assert "-02:00]" not in _build_agent_script_prompt(p)              # SECTION 1 no longer ends at 02:00
    tts, images = _build_tts_prompt("s", p), _build_image_prompt_instructions(p)
    assert "8-minute" in tts and "12-minute" not in tts
    assert "8-minute" in images and "14-minute" not in images
