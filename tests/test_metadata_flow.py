import json
from unittest.mock import MagicMock

import pytest

def _seed_run_with_script(tmp_path, slug="abc", script="SCRIPT BODY"):
    run = tmp_path / slug
    run.mkdir()
    (run / "script.txt").write_text(script)
    (run / "research.txt").write_text("RESEARCH")
    return run


def _stub_metadata_internals(monkeypatch, metadata):
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "")  # turn off vidIQ
    monkeypatch.setattr(metadata, "_vidiq_keywords", lambda topic, log_fn: [{"keyword": "kw"}])
    monkeypatch.setattr(
        metadata, "_generate_titles",
        lambda *a, **kw: ["T1", "T2", "T3", "T4", "T5"],
    )
    monkeypatch.setattr(
        metadata, "_score_titles",
        lambda titles, log_fn: [
            {"text": t, "score": 100 - i, "score_breakdown": {}}
            for i, t in enumerate(titles)
        ],
    )
    monkeypatch.setattr(
        metadata, "_generate_description_hashtags",
        lambda *a, **kw: {"description": "DESC", "hashtags": ["#a", "#b", "#c", "#d", "#e"]},
    )
    monkeypatch.setattr(
        metadata, "_generate_thumbnail_prompts",
        lambda *a, **kw: [
            {"prompt": "p1", "hook_text": "H1"},
            {"prompt": "p2", "hook_text": "H2"},
            {"prompt": "p3", "hook_text": "H3"},
        ],
    )


def test_full_flow_writes_all_artifacts(tmp_path, monkeypatch):
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)

    def fake_render(prompts, run_dir, profile, log_fn):
        out = []
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        for i, p in enumerate(prompts, 1):
            fn = f"thumbnails/thumb-{i:02d}.png"
            (run_dir / fn).write_bytes(b"PNG")
            out.append({**p, "filename": fn})
        return out

    monkeypatch.setattr(metadata, "_render_thumbnails", fake_render)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())

    run_dir = _seed_run_with_script(tmp_path)
    profile = MagicMock(
        channel={"niche": "n", "audience": "a", "tone": "t", "title_format": "tf"},
        characters_block=lambda: "chars",
        image_style={"art_style_block": "art"},
        image_gen={"pro_model": "gemini-3-pro-image", "default_model": "gemini-3.1-flash-image"},
    )
    data = metadata.generate_metadata("abc", profile, log_fn=lambda _: None)

    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "thumbnail.png").exists()
    assert (run_dir / "thumbnails" / "thumb-01.png").exists()
    assert data["chosen_thumbnail_index"] == 0
    assert "chosen_title_index" not in data
    # Sorted by score desc — T1 has score 100, should be first
    assert data["titles"][0]["text"] == "T1"
    assert data["tags"] == ["Abc", "kw"]  # topic (title-cased slug), then stubbed vidIQ keyword


def test_existing_metadata_blocks_without_regenerate(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = _seed_run_with_script(tmp_path)
    (run_dir / "metadata.json").write_text("{}")
    with pytest.raises(ValueError) as exc:
        metadata.generate_metadata("abc", MagicMock(), lambda _: None)
    assert "already exists" in str(exc.value).lower() or "regenerate" in str(exc.value).lower()


def test_existing_metadata_with_regenerate_overwrites(tmp_path, monkeypatch):
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)

    def fake_render(prompts, run_dir, profile, log_fn):
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        out = []
        for i, p in enumerate(prompts, 1):
            fn = f"thumbnails/thumb-{i:02d}.png"
            (run_dir / fn).write_bytes(b"PNG")
            out.append({**p, "filename": fn})
        return out

    monkeypatch.setattr(metadata, "_render_thumbnails", fake_render)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())

    run_dir = _seed_run_with_script(tmp_path)
    (run_dir / "metadata.json").write_text(json.dumps({"stale": True}))
    profile = MagicMock(
        channel={"niche": "n", "audience": "a", "tone": "t", "title_format": "tf"},
        characters_block=lambda: "c",
        image_style={"art_style_block": "x"},
        image_gen={"pro_model": "gemini-3-pro-image", "default_model": "gemini-3.1-flash-image"},
    )
    metadata.generate_metadata("abc", profile, lambda _: None, regenerate=True)
    data = json.loads((run_dir / "metadata.json").read_text())
    assert "stale" not in data
    assert len(data["titles"]) == 5
    assert [t["filename"] for t in data["thumbnails"]] == [
        f"thumbnails/thumb-{i:02d}.png" for i in (1, 2, 3)]


def test_missing_script_raises(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    (tmp_path / "abc").mkdir()
    with pytest.raises(ValueError) as exc:
        metadata.generate_metadata("abc", MagicMock(), lambda _: None)
    assert "script.txt" in str(exc.value).lower() or "incomplete" in str(exc.value).lower()


def test_thumbnail_render_failure_marks_slot(tmp_path, monkeypatch):
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)

    def fake_render(prompts, run_dir, profile, log_fn):
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        out = []
        for i, p in enumerate(prompts, 1):
            fn = f"thumbnails/thumb-{i:02d}.png"
            if i == 2:
                out.append({**p, "filename": fn, "render_error": "boom"})
            else:
                (run_dir / fn).write_bytes(b"PNG")
                out.append({**p, "filename": fn})
        return out

    monkeypatch.setattr(metadata, "_render_thumbnails", fake_render)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())

    run_dir = _seed_run_with_script(tmp_path)
    profile = MagicMock(
        channel={"niche": "n", "audience": "a", "tone": "t", "title_format": "tf"},
        characters_block=lambda: "c",
        image_style={"art_style_block": "x"},
        image_gen={"pro_model": "gemini-3-pro-image", "default_model": "gemini-3.1-flash-image"},
    )
    data = metadata.generate_metadata("abc", profile, lambda _: None)
    assert data["thumbnails"][1]["render_error"] == "boom"
    assert (run_dir / "thumbnail.png").exists()  # slot 0 succeeded, chosen by default


def test_all_thumbnails_fail_no_chosen_file(tmp_path, monkeypatch):
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)

    def fake_render_all_fail(prompts, run_dir, profile, log_fn):
        (run_dir / "thumbnails").mkdir(exist_ok=True)
        return [
            {**p, "filename": f"thumbnails/thumb-{i:02d}.png", "render_error": "boom"}
            for i, p in enumerate(prompts, 1)
        ]

    monkeypatch.setattr(metadata, "_render_thumbnails", fake_render_all_fail)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())

    run_dir = _seed_run_with_script(tmp_path)
    profile = MagicMock(
        channel={"niche": "n", "audience": "a", "tone": "t", "title_format": "tf"},
        characters_block=lambda: "c",
        image_style={"art_style_block": "x"},
        image_gen={"pro_model": "gemini-3-pro-image", "default_model": "gemini-3.1-flash-image"},
    )
    data = metadata.generate_metadata("abc", profile, lambda _: None)

    # All three slots errored
    assert all("render_error" in t for t in data["thumbnails"])
    # chosen_thumbnail_index defaults to 0 (no non-errored slot was found)
    assert data["chosen_thumbnail_index"] == 0
    # thumbnail.png NOT written because chosen slot has render_error
    assert not (run_dir / "thumbnail.png").exists()


def _fake_render(prompts, run_dir, profile, log_fn):
    (run_dir / "thumbnails").mkdir(exist_ok=True)
    for i in range(1, len(prompts) + 1):
        (run_dir / f"thumbnails/thumb-{i:02d}.png").write_bytes(b"PNG")
    return [{**p, "filename": f"thumbnails/thumb-{i:02d}.png"} for i, p in enumerate(prompts, 1)]


def test_description_is_told_the_real_audio_length(tmp_path, monkeypatch):
    # chapters came from the script's planned times: 3 of 12 started after a real run's audio ended
    import metadata
    from prompts import _build_metadata_desc_hashtags_prompt
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(metadata, "_render_thumbnails", _fake_render)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())
    seen = {}
    monkeypatch.setattr(metadata, "_generate_description_hashtags",
                        lambda *a, **kw: seen.update(kw) or {"description": "D", "hashtags": []})
    run_dir = _seed_run_with_script(tmp_path)
    (run_dir / "audio").mkdir()
    (run_dir / "audio" / "voiceover.mp3").write_bytes(b"\0" * 24_000)   # 1.0 s at 192 kbps
    metadata.generate_metadata("abc", MagicMock(image_gen={}), log_fn=lambda _: None)
    assert seen.get("audio_secs") == 1.0

    profile = MagicMock(channel={"niche": "n", "tone": "t"})
    prompt = _build_metadata_desc_hashtags_prompt("T", "S", [], profile, audio_secs=636.1)
    assert "10:36" in prompt


def test_topic_comes_from_the_scripts_title(tmp_path, monkeypatch):
    # the slug is cut at 60 chars and loses apostrophes; it became tag #1 and the vidIQ query
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(metadata, "_render_thumbnails", _fake_render)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())
    _seed_run_with_script(tmp_path, slug="why-cats-purr-heals",
                          script="**TITLE:** Why Your Cat's Purr Heals\n[00:00-00:30] HOOK\nbody")
    data = metadata.generate_metadata("why-cats-purr-heals", MagicMock(image_gen={}), log_fn=lambda _: None)
    assert data["topic"] == "Why Your Cat's Purr Heals" and data["tags"][0] == "Why Your Cat's Purr Heals"


def test_failed_regenerate_does_not_keep_the_old_pick(tmp_path, monkeypatch):
    import metadata
    _stub_metadata_internals(monkeypatch, metadata)
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(metadata, "anthropic", MagicMock())
    monkeypatch.setattr(metadata, "_render_thumbnails", lambda prompts, run_dir, profile, log_fn: [
        {**p, "filename": f"thumbnails/thumb-0{i}.png", "render_error": "quota"} for i, p in enumerate(prompts, 1)])
    run_dir = _seed_run_with_script(tmp_path)
    (run_dir / "thumbnail.png").write_bytes(b"OLD PICK")
    data = metadata.generate_metadata("abc", MagicMock(image_gen={}), log_fn=lambda _: None, regenerate=True)
    assert "render_error" in data["thumbnails"][data["chosen_thumbnail_index"]]
    assert not (run_dir / "thumbnail.png").exists()      # it no longer matches metadata.json
