import json
from unittest.mock import MagicMock

import pytest

def test_load_metadata_missing(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    (tmp_path / "abc").mkdir()
    assert metadata.load_metadata("abc") is None


def test_load_metadata_run_dir_missing(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    assert metadata.load_metadata("nope") is None


def test_load_metadata_roundtrip(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    data = {"run_slug": "abc", "titles": []}
    metadata._save_metadata(run_dir, data)
    loaded = metadata.load_metadata("abc")
    assert loaded == data


def test_load_metadata_corrupt_json(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{not json")
    with pytest.raises(ValueError) as exc:
        metadata.load_metadata("abc")
    assert "metadata.json" in str(exc.value)


def test_save_metadata_atomic_no_tmp_left(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run_dir = tmp_path / "abc"
    run_dir.mkdir()
    metadata._save_metadata(run_dir, {"x": 1})
    assert (run_dir / "metadata.json").exists()
    assert not (run_dir / "metadata.json.tmp").exists()


def test_vidiq_keywords_no_key_returns_empty(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "")
    logs = []
    assert metadata._vidiq_keywords("anything", logs.append) == []
    assert any("vidIQ" in m for m in logs)


def test_vidiq_keywords_call_failure_returns_empty(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    def bad(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(metadata, "_call_vidiq_agent", bad)
    logs = []
    result = metadata._vidiq_keywords("anything", logs.append)
    assert result == []
    assert any("boom" in m or "vidIQ" in m for m in logs)


def test_vidiq_keywords_parses_agent_json(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    fake_payload = '===KEYWORDS===\n[{"keyword":"vitamins","search_volume":9000}]\n'
    monkeypatch.setattr(metadata, "_call_vidiq_agent", lambda *a, **kw: fake_payload)
    out = metadata._vidiq_keywords("vitamins", lambda _: None)
    assert out == [{"keyword": "vitamins", "search_volume": 9000}]


def test_build_video_tags_dedup_and_topic_first():
    import metadata
    tags = metadata._build_video_tags(
        "Vitamin D", [{"keyword": "vitamin d"}, {"keyword": "sunlight"}, {"keyword": "Vitamin D"}],
    )
    assert tags == ["Vitamin D", "sunlight"]  # case-insensitive dedup, topic kept first


def test_build_video_tags_caps_at_500_chars():
    import metadata
    keywords = [{"keyword": f"keyword number {i}"} for i in range(50)]
    tags = metadata._build_video_tags("topic", keywords)
    # YouTube counts one comma between tags and quotes around any tag containing a space
    assert sum(len(t) + 2 * (" " in t) for t in tags) + len(tags) - 1 <= 500
    assert len(tags) < 51  # got truncated, not all 51 candidates fit


def test_parse_titles_five_clean():
    raw = (
        "===TITLES===\n"
        "1. First title here\n"
        "2. Second title here\n"
        "3. Third\n"
        "4. Fourth one\n"
        "5. Fifth and final\n"
    )
    import metadata
    assert metadata._parse_titles(raw) == [
        "First title here",
        "Second title here",
        "Third",
        "Fourth one",
        "Fifth and final",
    ]


def test_parse_titles_with_preamble_and_quotes():
    raw = (
        "Here are 5 titles:\n"
        "===TITLES===\n"
        '1. "Wrapped in quotes"\n'
        "2. Plain\n"
        "3. Another\n"
        "4. Fourth\n"
        "5. Fifth\n"
        "===END===\n"
    )
    import metadata
    titles = metadata._parse_titles(raw)
    assert titles[0] == "Wrapped in quotes"
    assert len(titles) == 5


def test_parse_titles_short_returns_what_we_got():
    raw = "===TITLES===\n1. only\n2. two\n"
    import metadata
    assert metadata._parse_titles(raw) == ["only", "two"]


def test_parse_titles_missing_block_returns_empty():
    import metadata
    assert metadata._parse_titles("nothing here") == []


class _FakeContent:
    def __init__(self, text): self.text = text
class _FakeResponse:
    def __init__(self, text): self.content = [_FakeContent(text)]


def test_generate_titles_returns_parsed_list(monkeypatch):
    import metadata

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse(
                    "===TITLES===\n"
                    "1. A\n2. B\n3. C\n4. D\n5. E\n"
                )

    titles = metadata._generate_titles(
        "topic", "script body", "research", [{"keyword": "kw"}],
        profile=MagicMock(channel={"niche": "n", "audience": "a", "title_format": "tf", "tone": "t"}),
        client=FakeClient,
        log_fn=lambda _: None,
    )
    assert titles == ["A", "B", "C", "D", "E"]


def test_generate_titles_logs_warning_on_short(monkeypatch):
    import metadata

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse("===TITLES===\n1. only one\n")

    logs = []
    titles = metadata._generate_titles(
        "topic", "script", "research", [],
        profile=MagicMock(channel={"niche": "n", "audience": "a", "title_format": "tf", "tone": "t"}),
        client=FakeClient,
        log_fn=logs.append,
    )
    assert titles == ["only one"]
    assert any("fewer than 5" in m.lower() or "1 title" in m.lower() for m in logs)


def test_score_titles_no_vidiq_returns_null_scores(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "")
    result = metadata._score_titles(["a", "b", "c"], lambda _: None)
    assert [r["text"] for r in result] == ["a", "b", "c"]
    assert all(r["score"] is None for r in result)
    assert all(r["score_breakdown"] == {} for r in result)


def test_score_titles_per_title_failure_is_null(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")

    def fake_call(system, user, max_turns, log_fn):
        # Agent silently drops "fail" from its response — simulates a per-title tool failure.
        return ('===SCORES===\n'
                '[{"title": "ok", "score": 80, "breakdown": {"ctr": 7.0}}, '
                '{"title": "ok2", "score": 80, "breakdown": {"ctr": 7.0}}]')

    monkeypatch.setattr(metadata, "_call_vidiq_agent", fake_call)
    result = metadata._score_titles(["ok", "fail", "ok2"], lambda _: None)
    assert result[0] == {"text": "ok", "score": 80, "score_breakdown": {"ctr": 7.0}}
    assert result[1]["text"] == "fail" and result[1]["score"] is None
    assert result[2]["score"] == 80


def test_score_titles_preserves_input_order(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")

    def fake_call(system, user, max_turns, log_fn):
        # Agent returns entries out of order — output must still match input order.
        return ('===SCORES===\n'
                '[{"title": "z", "score": 60, "breakdown": {}}, '
                '{"title": "x", "score": 50, "breakdown": {}}, '
                '{"title": "y", "score": 70, "breakdown": {}}]')

    monkeypatch.setattr(metadata, "_call_vidiq_agent", fake_call)
    result = metadata._score_titles(["x", "y", "z"], lambda _: None)
    assert [r["text"] for r in result] == ["x", "y", "z"]
    assert [r["score"] for r in result] == [50, 70, 60]


def test_description_hashtags_parses_blocks():
    import metadata

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse(
                    "===DESCRIPTION===\n"
                    "First line of description.\n\n00:00 Hook\n00:35 Setup\n"
                    "===HASHTAGS===\n"
                    "#vitamins #nutrition #health #science #biology #extra\n"
                )

    out = metadata._generate_description_hashtags(
        "Title here", "script body", [],
        profile=MagicMock(channel={"niche": "n", "audience": "a", "tone": "t"}),
        client=FakeClient,
        log_fn=lambda _: None,
    )
    assert "First line of description" in out["description"]
    assert out["hashtags"][:3] == ["#vitamins", "#nutrition", "#health"]
    assert len(out["hashtags"]) == 5  # capped at 5


def test_description_hashtags_missing_blocks_raises():
    import metadata

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse("nothing useful here")

    with pytest.raises(ValueError) as exc:
        metadata._generate_description_hashtags(
            "T", "s", [],
            profile=MagicMock(channel={"niche": "n", "audience": "a", "tone": "t"}),
            client=FakeClient,
            log_fn=lambda _: None,
        )
    assert "description" in str(exc.value).lower() or "hashtag" in str(exc.value).lower()


def test_thumbnail_prompts_three_clean():
    import metadata

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse(
                    "===THUMBNAIL_1===\n"
                    "HOOK: DO NOTHING\n"
                    "Orange cat holding giant pill...\n"
                    "===THUMBNAIL_2===\n"
                    "HOOK: WRONG\n"
                    "White cat pointing at chart...\n"
                    "===THUMBNAIL_3===\n"
                    "HOOK: MYTH\n"
                    "Orange cat next to glowing diagram...\n"
                    "===END===\n"
                )

    out = metadata._generate_thumbnail_prompts(
        "script body", "topic",
        profile=MagicMock(
            channel={"niche": "n", "audience": "a", "tone": "t"},
            characters_block=lambda: "Orange Cat: ...; White Cat: ...",
            image_style={"art_style_block": "ART STYLE BLOCK"},
        ),
        client=FakeClient,
        log_fn=lambda _: None,
    )
    assert len(out) == 3
    assert out[0]["hook_text"] == "DO NOTHING"
    assert "giant pill" in out[0]["prompt"]


def test_thumbnail_prompts_retry_then_fail():
    import metadata

    class FakeClient:
        calls = 0
        class messages:
            @staticmethod
            def create(**kw):
                FakeClient.calls += 1
                return _FakeResponse("not valid")

    with pytest.raises(ValueError):
        metadata._generate_thumbnail_prompts(
            "s", "t",
            profile=MagicMock(
                channel={"niche": "n", "audience": "a", "tone": "t"},
                characters_block=lambda: "x",
                image_style={"art_style_block": "y"},
            ),
            client=FakeClient,
            log_fn=lambda _: None,
        )
    assert FakeClient.calls == 2  # one initial call + one retry


def test_render_thumbnails_all_succeed(tmp_path, monkeypatch):
    import metadata

    def fake_gen(prompt, output_path, profile, log_fn, model=None, anchor_parts=None):
        output_path.write_bytes(b"PNGDATA")
        return True

    monkeypatch.setattr(metadata, "generate_image_google", fake_gen)
    monkeypatch.setattr(metadata, "_load_anchor_parts", lambda p: [])

    out = metadata._render_thumbnails(
        [
            {"prompt": "p1", "hook_text": "A"},
            {"prompt": "p2", "hook_text": "B"},
            {"prompt": "p3", "hook_text": "C"},
        ],
        tmp_path,
        profile=MagicMock(image_gen={"pro_model": "gemini-3-pro-image"}),
        log_fn=lambda _: None,
    )
    assert len(out) == 3
    assert all((tmp_path / r["filename"]).exists() for r in out)
    assert all("render_error" not in r for r in out)


def test_render_thumbnails_middle_fails(tmp_path, monkeypatch):
    import metadata

    def fake_gen(prompt, output_path, profile, log_fn, model=None, anchor_parts=None):
        if prompt == "p2":
            return False
        output_path.write_bytes(b"PNGDATA")
        return True

    monkeypatch.setattr(metadata, "generate_image_google", fake_gen)
    monkeypatch.setattr(metadata, "_load_anchor_parts", lambda p: [])

    out = metadata._render_thumbnails(
        [{"prompt": "p1", "hook_text": "A"},
         {"prompt": "p2", "hook_text": "B"},
         {"prompt": "p3", "hook_text": "C"}],
        tmp_path,
        profile=MagicMock(image_gen={"pro_model": "gemini-3-pro-image"}),
        log_fn=lambda _: None,
    )
    assert len(out) == 3
    assert "render_error" not in out[0]
    assert "render_error" in out[1]
    assert "render_error" not in out[2]
    on_disk = sorted(p.name for p in (tmp_path / "thumbnails").iterdir())
    assert "thumb-01.png" in on_disk
    assert "thumb-03.png" in on_disk
    assert "thumb-02.png" not in on_disk


def test_render_thumbnails_exception_raised(tmp_path, monkeypatch):
    import metadata

    def fake_gen(prompt, output_path, profile, log_fn, model=None, anchor_parts=None):
        if prompt == "p2":
            raise RuntimeError("kaboom")
        output_path.write_bytes(b"PNGDATA")
        return True

    monkeypatch.setattr(metadata, "generate_image_google", fake_gen)
    monkeypatch.setattr(metadata, "_load_anchor_parts", lambda p: [])

    out = metadata._render_thumbnails(
        [{"prompt": "p1", "hook_text": "A"},
         {"prompt": "p2", "hook_text": "B"},
         {"prompt": "p3", "hook_text": "C"}],
        tmp_path,
        profile=MagicMock(image_gen={"pro_model": "gemini-3-pro-image"}),
        log_fn=lambda _: None,
    )
    assert len(out) == 3
    assert "render_error" not in out[0]
    assert "render_error" in out[1]
    assert "kaboom" in out[1]["render_error"]
    assert "render_error" not in out[2]
    on_disk = sorted(p.name for p in (tmp_path / "thumbnails").iterdir())
    assert "thumb-01.png" in on_disk
    assert "thumb-03.png" in on_disk
    assert "thumb-02.png" not in on_disk


def _seed_run(tmp_path):
    run = tmp_path / "abc"
    run.mkdir()
    (run / "thumbnails").mkdir()
    (run / "thumbnails" / "thumb-01.png").write_bytes(b"ONE")
    (run / "thumbnails" / "thumb-02.png").write_bytes(b"TWO")
    (run / "thumbnails" / "thumb-03.png").write_bytes(b"THREE")
    data = {
        "run_slug": "abc",
        "titles": [{"index": i, "text": f"t{i}"} for i in range(3)],
        "thumbnails": [
            {"index": 0, "filename": "thumbnails/thumb-01.png"},
            {"index": 1, "filename": "thumbnails/thumb-02.png"},
            {"index": 2, "filename": "thumbnails/thumb-03.png"},
        ],
        "chosen_thumbnail_index": 0,
    }
    (run / "metadata.json").write_text(json.dumps(data))
    return run


def test_pick_thumbnail_copies_file(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run = _seed_run(tmp_path)
    metadata.pick_thumbnail("abc", 1)
    chosen = (run / "thumbnail.png").read_bytes()
    src = (run / "thumbnails" / "thumb-02.png").read_bytes()
    assert chosen == src


def test_pick_thumbnail_skips_errored_slot(tmp_path, monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "OUTPUT_ROOT", tmp_path)
    run = _seed_run(tmp_path)
    data = json.loads((run / "metadata.json").read_text())
    data["thumbnails"][1] = {**data["thumbnails"][1], "render_error": "boom"}
    (run / "thumbnails" / "thumb-02.png").unlink()
    (run / "metadata.json").write_text(json.dumps(data))
    with pytest.raises(ValueError) as exc:
        metadata.pick_thumbnail("abc", 1)
    assert "render_error" in str(exc.value) or "not available" in str(exc.value).lower()


@pytest.mark.parametrize("payload,expected", [
    ('===KEYWORDS===\n["cat purring", "why cats purr"]', [{"keyword": "cat purring"}, {"keyword": "why cats purr"}]),
    ('===KEYWORDS===\n```json\n[{"keyword": "cat purring"}]\n```', [{"keyword": "cat purring"}]),
    ('===KEYWORDS===\n{"keyword": "not a list"}', []),
])
def test_vidiq_keywords_tolerate_the_agents_json_shape(monkeypatch, payload, expected):
    # a string list crashed _build_video_tags after the paid session; a ```json fence gave []
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    monkeypatch.setattr(metadata, "_call_vidiq_agent", lambda *a, **kw: payload)
    keywords = metadata._vidiq_keywords("cats", lambda _: None)
    assert keywords == expected
    metadata._build_video_tags("cats", keywords)   # must not raise


def test_score_titles_tolerates_odd_scores_and_fences(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    payload = ('===SCORES===\n```json\n[{"title": "A", "score": "85/100", "breakdown": {}},'
               ' {"title": "B", "score": 71, "breakdown": {}}]\n```')
    monkeypatch.setattr(metadata, "_call_vidiq_agent", lambda *a, **kw: payload)
    assert [t["score"] for t in metadata._score_titles(["A", "B"], lambda _: None)] == [None, 71]


def test_parse_titles_strips_markdown_labels_and_accepts_bullets():
    import metadata
    raw = ("===TITLES===\n1. **Why Cats Purr**\n2. Title: Cats Rule\n3) “Smart Quotes”\n"
           "- Bullet Title\n4: Colon Numbered\n===END===")
    assert metadata._parse_titles(raw) == ["Why Cats Purr", "Cats Rule", "Smart Quotes",
                                           "Bullet Title", "Colon Numbered"]


def test_no_titles_means_no_scoring_session(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    monkeypatch.setattr(metadata, "_call_vidiq_agent", lambda *a, **k: pytest.fail("paid session for 0 titles"))
    assert metadata._score_titles([], lambda _: None) == []


def test_scores_match_titles_despite_punctuation_changes(monkeypatch):
    import metadata
    monkeypatch.setattr(metadata, "VIDIQ_KEY", "fake-key")
    payload = '===SCORES===\n[{"title": "Why Your Cat’s Purr Heals.", "score": 77, "breakdown": {}}]'
    monkeypatch.setattr(metadata, "_call_vidiq_agent", lambda *a, **kw: payload)
    assert metadata._score_titles(["Why Your Cat's Purr Heals"], lambda _: None)[0]["score"] == 77


@pytest.mark.parametrize("first_line", ["HOOK: BIG 1", "**HOOK:** BIG 1", "Hook - BIG 1"])
def test_thumbnail_hook_line_variants(first_line):
    import metadata
    raw = "".join(f"===THUMBNAIL_{n}===\n{first_line}\nscene {n}\n" for n in (1, 2, 3)) + "===END==="
    thumbs = metadata._parse_thumbnail_prompts(raw)
    assert [t["hook_text"] for t in thumbs] == ["BIG 1"] * 3
    assert [t["prompt"] for t in thumbs] == ["scene 1", "scene 2", "scene 3"]   # hook not sent to Gemini


def test_overlong_titles_are_dropped():
    import metadata
    raw = "===TITLES===\n1. " + "x" * 101 + "\n2. Short enough\n===END==="
    assert metadata._parse_titles(raw) == ["Short enough"]   # YouTube rejects titles over 100 chars


def test_overlong_description_is_flagged():
    import metadata
    logs = []

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kw):
                return _FakeResponse("===DESCRIPTION===\n" + "word " * 1100 + "\n===HASHTAGS===\n#a #b\n")

    metadata._generate_description_hashtags(
        "t", "s", [], profile=MagicMock(channel={"niche": "n", "audience": "a", "tone": "t"}),
        client=FakeClient, log_fn=logs.append)
    assert any("5000" in m for m in logs)                    # YouTube's description limit
