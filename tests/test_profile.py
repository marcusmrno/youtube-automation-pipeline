import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_profile_returns_profile():
    from profile import load_profile, Profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert isinstance(p, Profile)
    assert p.name == "test-channel"


def test_load_profile_channel_fields():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.channel["name"] == "Test Channel"
    assert p.channel["niche"] == "Testing / fixtures"


def test_load_profile_script_fields():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.script["target_mins"] == 12
    assert p.script["wpm"] == 160


def test_load_profile_characters():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert len(p.characters) == 2
    assert p.characters[0]["name"] == "Test Cat"
    assert "round head" in p.characters[0]["description"]


def test_load_profile_character_behavior():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert "Alternate" in p.character_behavior


def test_load_profile_image_style():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert "flat 2D test illustration" in p.image_style["art_style_block"]
    assert p.image_style["max_anchors"] == 2


def test_load_profile_voice_id_literal():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.voice["voice_id"] == "test-voice-id-123"


def test_load_profile_voice_id_env_var(monkeypatch):
    from profile import load_profile
    # Patch the fixture yaml temporarily to use env var syntax
    import yaml, shutil, tempfile
    src = FIXTURES / "test-channel" / "profile.yaml"
    data = yaml.safe_load(src.read_text())
    data["voice"]["voice_id"] = "${TEST_VOICE_ENV}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copytree(FIXTURES / "test-channel", tmp_path / "test-channel")
        (tmp_path / "test-channel" / "profile.yaml").write_text(yaml.dump(data))

        monkeypatch.setenv("TEST_VOICE_ENV", "resolved-voice-id")
        p = load_profile("test-channel", profiles_root=tmp_path)
        assert p.voice["voice_id"] == "resolved-voice-id"


def test_load_profile_missing_raises():
    from profile import load_profile
    with pytest.raises(ValueError, match="not found"):
        load_profile("nonexistent-channel", profiles_root=FIXTURES)


def test_load_profile_anchors_dir():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.anchors_dir == FIXTURES / "test-channel" / "anchors"


def test_load_profile_dir():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.dir == FIXTURES / "test-channel"


def test_list_profiles():
    from profile import list_profiles
    profiles = list_profiles(profiles_root=FIXTURES)
    assert "test-channel" in profiles


def test_list_profiles_empty_dir(tmp_path):
    from profile import list_profiles
    assert list_profiles(profiles_root=tmp_path) == []


def test_list_profiles_missing_dir(tmp_path):
    from profile import list_profiles
    assert list_profiles(profiles_root=tmp_path / "nonexistent") == []


def test_characters_block():
    from profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    block = p.characters_block()
    assert "**Test Cat:**" in block
    assert "round head" in block
    assert "**Other Cat:**" in block



@pytest.mark.parametrize("name", ["../fixtures/test-channel", str(FIXTURES / "test-channel"),
                                  "test-channel/../test-channel"])
def test_load_profile_rejects_paths(name):
    # UI request bodies reach load_profile; only plain profile names may load
    from profile import load_profile
    with pytest.raises(ValueError):
        load_profile(name, profiles_root=FIXTURES)
