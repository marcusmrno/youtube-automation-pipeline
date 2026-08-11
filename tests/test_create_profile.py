import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from profile_creator.revise import next_version_name
from profile_creator.claude_helpers import extract_fenced_block
from profile_creator.anchors import run_full_anchors


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


def test_run_full_anchors_skips_existing(tmp_path):
    anchors_dir = tmp_path / "anchors"
    anchors_dir.mkdir()
    # Pre-create anchor-03.png so it should be skipped
    (anchors_dir / "anchor-03.png").write_bytes(b"fake")

    profile_yaml = {
        "image_gen": {"default_model": "gemini-3.1-flash-image"},
        "image_style": {"max_anchors": 14},
    }
    prompts = {"anchor-01": "prompt1", "anchor-02": "prompt2", "anchor-03": "prompt3"}
    plan = [
        {"label": "anchor-01", "tier": "full", "purpose": "test slot 1"},
        {"label": "anchor-02", "tier": "full", "purpose": "test slot 2"},
        {"label": "anchor-03", "tier": "full", "purpose": "test slot 3"},
    ]

    with patch("profile_creator.anchors._generate_anchor", return_value=True) as mock_gen:
        result = run_full_anchors(profile_yaml, anchors_dir, prompts, plan)

    # anchor-03 already exists, should not be generated
    called_names = [call.args[1].name for call in mock_gen.call_args_list]
    assert not any("anchor-03" in n for n in called_names)


def test_run_create_writes_profile_files(tmp_path, monkeypatch):
    import profile_creator.new as m

    monkeypatch.setattr(m, "PROFILES_ROOT", tmp_path)

    fake_yaml = """channel:
  name: Test Channel
  niche: test
  audience: test
  tone: test
  reference_channel: test
  title_format: test
script:
  target_mins: 12
  min_mins: 9
  max_mins: 15
  wpm: 160
  hook_duration_s: 35
  cta_duration_s: 30
  section_count: "10-14"
  section_duration_s: "60-90"
characters:
  roster:
    - name: Test Cat
      description: a test cat
  behavior: alternate
image_style:
  art_style_block: flat
  max_anchors: 2
voice:
  voice_id: ${ELEVENLABS_VOICE_ID}
  model: eleven_v3
  stability: 0.68
  similarity_boost: 0.85
  style: 0.0
  use_speaker_boost: true
  tone_description: test
image_gen:
  default_model: gemini-3.1-flash-image
  pro_model: gemini-3-pro-image"""

    with patch("profile_creator.new.anthropic.Anthropic"), \
         patch("profile_creator.new.clarification_loop", return_value=[]), \
         patch("profile_creator.new.generate_profile_content", return_value=(fake_yaml, "# Style\n")), \
         patch("profile_creator.new.generate_anchor_prompts", return_value=(
             {"anchor-01": "p1", "anchor-02": "p2"},
             [{"label": "anchor-01", "purpose": "cast sheet", "tier": "verification"},
              {"label": "anchor-02", "purpose": "props",      "tier": "full"}],
         )), \
         patch("profile_creator.new.run_verification_anchors", return_value=True), \
         patch("profile_creator.new.run_full_anchors", return_value={"ok": [], "failed": []}), \
         patch("builtins.input", side_effect=["some channel concept", "---", "my-channel"]):
        m.run_create()

    profile_dir = tmp_path / "my-channel"
    assert (profile_dir / "profile.yaml").exists()
    assert (profile_dir / "style-sheet.md").exists()
    # Anchor slot descriptions must be persisted — image generation reads them back
    manifest = yaml.safe_load((profile_dir / "anchors" / "manifest.yaml").read_text())
    assert [s["label"] for s in manifest] == ["anchor-01", "anchor-02"]


def test_run_revise_creates_v2_folder(tmp_path, monkeypatch):
    import profile_creator.revise as m
    from profile_creator.revise import next_version_name

    src_dir = tmp_path / "my-channel"
    src_dir.mkdir()
    (src_dir / "anchors").mkdir()
    (src_dir / "profile.yaml").write_text("""channel:
  name: My Channel
  niche: test
  audience: test
  tone: test
  reference_channel: test
  title_format: test
script:
  target_mins: 12
  min_mins: 9
  max_mins: 15
  wpm: 160
  hook_duration_s: 35
  cta_duration_s: 30
  section_count: "10-14"
  section_duration_s: "60-90"
characters:
  roster:
    - name: Test Cat
      description: a cat
  behavior: alternate
image_style:
  art_style_block: flat
  max_anchors: 2
voice:
  voice_id: ${ELEVENLABS_VOICE_ID}
  model: eleven_v3
  stability: 0.68
  similarity_boost: 0.85
  style: 0.0
  use_speaker_boost: true
  tone_description: test
image_gen:
  default_model: gemini-3.1-flash-image
  pro_model: gemini-3-pro-image""")
    (src_dir / "style-sheet.md").write_text("# Style")

    monkeypatch.setattr(m, "PROFILES_ROOT", tmp_path)

    updated_yaml = (src_dir / "profile.yaml").read_text()

    with patch("profile_creator.revise.anthropic.Anthropic"), \
         patch("profile_creator.revise.clarification_loop", return_value=[]), \
         patch("profile_creator.revise.generate_profile_content", return_value=(updated_yaml, "# Style\n")), \
         patch("builtins.input", return_value="make it darker"):
        m.run_revise("my-channel")

    v2_dir = tmp_path / "my-channel-v2"
    assert v2_dir.exists()
    assert (v2_dir / "profile.yaml").exists()
    assert (v2_dir / "style-sheet.md").exists()
