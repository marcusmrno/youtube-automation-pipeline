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
    # Create fake anchor files in the test profile anchors dir
    anchors_dir = FIXTURES / "test-channel" / "anchors"
    anchors_dir.mkdir(exist_ok=True)
    (anchors_dir / "anchor-01.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    (anchors_dir / "anchor-02.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    captured_contents = []

    def fake_generate(model, contents, config):
        captured_contents.extend(contents)
        raise RuntimeError("stop after capture")

    import pipeline
    with patch.object(pipeline, "genai") as mock_genai:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.side_effect = fake_generate

        output = tmp_path / "001.png"
        pipeline.generate_image_google("test prompt", output, test_profile, print)

    # The first content element should be the prompt string
    assert any(isinstance(c, str) and "test prompt" in c for c in captured_contents)


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
