import pytest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_profile_returns_profile():
    from channel_profile import load_profile, Profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert isinstance(p, Profile)
    assert p.name == "test-channel"


def test_load_profile_channel_fields():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.channel["name"] == "Test Channel"
    assert p.channel["niche"] == "Testing / fixtures"


def test_load_profile_script_fields():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.script["target_mins"] == 12
    assert p.script["wpm"] == 160


def test_load_profile_characters():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert len(p.characters) == 2
    assert p.characters[0]["name"] == "Test Cat"
    assert "round head" in p.characters[0]["description"]


def test_load_profile_character_behavior():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert "Alternate" in p.character_behavior


def test_load_profile_image_style():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert "flat 2D test illustration" in p.image_style["art_style_block"]
    assert p.image_style["max_anchors"] == 2


def test_load_profile_voice_id_literal():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.voice["voice_id"] == "test-voice-id-123"


def test_load_profile_voice_id_env_var(monkeypatch):
    from channel_profile import load_profile
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
    from channel_profile import load_profile
    with pytest.raises(ValueError, match="not found"):
        load_profile("nonexistent-channel", profiles_root=FIXTURES)


def test_load_profile_anchors_dir():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.anchors_dir == FIXTURES / "test-channel" / "anchors"


def test_load_profile_dir():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    assert p.dir == FIXTURES / "test-channel"


def test_list_profiles():
    from channel_profile import list_profiles
    profiles = list_profiles(profiles_root=FIXTURES)
    assert "test-channel" in profiles


def test_list_profiles_empty_dir(tmp_path):
    from channel_profile import list_profiles
    assert list_profiles(profiles_root=tmp_path) == []


def test_list_profiles_missing_dir(tmp_path):
    from channel_profile import list_profiles
    assert list_profiles(profiles_root=tmp_path / "nonexistent") == []


def test_characters_block():
    from channel_profile import load_profile
    p = load_profile("test-channel", profiles_root=FIXTURES)
    block = p.characters_block()
    assert "**Test Cat:**" in block
    assert "round head" in block
    assert "**Other Cat:**" in block



@pytest.mark.parametrize("name", ["../fixtures/test-channel", str(FIXTURES / "test-channel"),
                                  "test-channel/../test-channel"])
def test_load_profile_rejects_paths(name):
    # UI request bodies reach load_profile; only plain profile names may load
    from channel_profile import load_profile
    with pytest.raises(ValueError):
        load_profile(name, profiles_root=FIXTURES)


@pytest.mark.parametrize("broken,named", [
    (lambda y: y.pop("image_gen"), "image_gen"),                              # missing section
    (lambda y: y["voice"].pop("voice_id"), "voice_id"),                       # missing key
    (lambda y: y["script"].update(section_duration_s=90), "script"),          # must be "lo-hi"
    (lambda y: y["script"].update(hook_duration_s="35"), "script"),           # must be a number
])
def test_broken_profiles_raise_a_readable_valueerror(tmp_path, broken, named):
    # a bare KeyError after a paid call (or an HTML 500 in the UI) was all you got
    import yaml
    from channel_profile import load_profile
    data = yaml.safe_load((FIXTURES / "test-channel" / "profile.yaml").read_text())
    broken(data)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "profile.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=named):
        load_profile("bad", profiles_root=tmp_path)


def test_empty_profile_yaml_raises_valueerror(tmp_path):
    from channel_profile import load_profile
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "profile.yaml").write_text("")
    with pytest.raises(ValueError, match="channel"):
        load_profile("empty", profiles_root=tmp_path)
