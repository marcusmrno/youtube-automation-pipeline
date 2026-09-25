"""Image-prompt expansion, stream retry, preamble, premade-script titles, stop handling."""
import threading
import time
import types

import anthropic
import pytest
import images
import pipeline
import writing
from images import build_preamble, generate_all_images, GENERIC_ANCHOR_REFS
from pipeline import _topic_from_script, slugify
from writing import _expand_short_prompts, _stream_text


class _Profile:
    characters = [
        {"name": "Orange Cat", "description": "Large orange tabby cat, bowtie"},
        {"name": "White Cat", "description": "Small white fluffy cat, collar"},
    ]
    image_style = {"art_style_block": "flat 2D illustration, marker texture"}
    image_gen = {"flicker": {"enabled": False}}


P = _Profile()

RAW = """
001 | Ancient humans let food rot. | Orange Cat | holds a crock — plain background
002 | They did it on purpose. | White Cat | stamps a label — plain background
003 | A diagram with no cat. | none | a timeline of fermentation — plain background
"""


def test_expand_short_prompts():
    out = _expand_short_prompts(RAW, P)
    assert [p["num"] for p in out] == ["001", "002", "003"], out
    assert out[0]["source"] == "Ancient humans let food rot."

    # character description is prepended, style block appended
    assert out[0]["prompt"] == (
        "Large orange tabby cat, bowtie — holds a crock — plain background"
        " — flat 2D illustration, marker texture")
    assert out[1]["prompt"].startswith("Small white fluffy cat, collar — stamps a label")
    # `none` character -> no description, still gets the style block
    assert out[2]["prompt"] == (
        "a timeline of fermentation — plain background"
        " — flat 2D illustration, marker texture")
    assert all(p["prompt"].endswith("flat 2D illustration, marker texture") for p in out)

    # multiple characters: both descriptions, in the order named
    multi = _expand_short_prompts("008 | s | Orange Cat + White Cat | face off — bg", P)[0]["prompt"]
    assert multi == ("Large orange tabby cat, bowtie and Small white fluffy cat, collar"
                     " — face off — bg — flat 2D illustration, marker texture"), multi

    # unknown name inside a multi-character field is dropped, the known one survives
    assert _expand_short_prompts("009 | s | Grey Cat + White Cat | meet — bg", P)[0]["prompt"] == \
        "Small white fluffy cat, collar — meet — bg — flat 2D illustration, marker texture"

    # unknown character name degrades to scene-only rather than crashing
    assert _expand_short_prompts("004 | s | Grey Cat | sits — bg", P)[0]["prompt"] == \
        "sits — bg — flat 2D illustration, marker texture"

    # malformed lines (wrong field count, empty scene) are skipped, not crashed on
    assert _expand_short_prompts("005 | only | three", P) == []
    assert _expand_short_prompts("006 | s | Orange Cat | ", P) == []

    # later duplicate number wins
    dupe = _expand_short_prompts("007 | a | Orange Cat | first — bg\n007 | b | Orange Cat | second — bg", P)
    assert len(dupe) == 1 and "second" in dupe[0]["prompt"]


class _Blk:
    def __init__(self, t, x): self.type, self.text = t, x


class _FakeStream:
    def __init__(self, fail): self.fail = fail
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self):
        if self.fail:
            raise anthropic.APIConnectionError(request=None)
        return types.SimpleNamespace(content=[_Blk("thinking", ""), _Blk("text", "OK")])


class _FakeClient:
    def __init__(self, fails): self.fails, self.calls = fails, 0
    @property
    def messages(self):
        def stream(**kw):
            self.calls += 1
            return _FakeStream(self.calls <= self.fails)
        return types.SimpleNamespace(stream=stream)


def test_stream_text_retry(monkeypatch):
    monkeypatch.setattr(writing.time, "sleep", lambda s: None)   # don't actually back off

    c = _FakeClient(fails=2)
    assert _stream_text(c, model="m", max_tokens=1, messages=[]) == "OK"
    assert c.calls == 3, c.calls

    c = _FakeClient(fails=99)
    try:
        _stream_text(c, model="m", max_tokens=1, messages=[])
        raise AssertionError("should have raised after exhausting attempts")
    except anthropic.APIConnectionError:
        pass
    assert c.calls == 3, c.calls


def test_build_preamble(tmp_path):
    # style_constraints wins when present
    pre = build_preamble({"art_style_block": "LONG BLOCK", "style_constraints": "SHORT RULES"})
    assert "SHORT RULES" in pre and "LONG BLOCK" not in pre, pre
    # falls back to the full block when it isn't
    assert "LONG BLOCK" in build_preamble({"art_style_block": "LONG BLOCK"})
    # no anchors dir -> generic reference line, no invented slot descriptions
    assert GENERIC_ANCHOR_REFS in pre and "anchor-01" not in pre, pre
    # nothing from any one profile's look leaks in
    for leak in ("marker", "off-white", "gradient", "cat"):
        assert leak not in build_preamble({"art_style_block": "watercolour"}).lower(), leak

    # missing manifest -> generic line, never a crash
    assert GENERIC_ANCHOR_REFS in build_preamble({"art_style_block": "x"}, tmp_path)
    (tmp_path / "manifest.yaml").write_text("- label: anchor-01\n  purpose: cast sheet\n")
    (tmp_path / "anchor-01.png").write_bytes(b"x")          # only anchors actually sent are described
    pre = build_preamble({"art_style_block": "x"}, tmp_path)
    assert "anchor-01: cast sheet" in pre and GENERIC_ANCHOR_REFS not in pre, pre


def test_topic_from_script():
    assert _topic_from_script("TITLE: The Real Reason\nKEYWORDS: a, b\n") == "The Real Reason"
    assert _topic_from_script("\n\n  title: lower case works\n") == "lower case works"
    assert _topic_from_script("No title line here\nsecond line") == "No title line here"
    assert _topic_from_script("   \n\n") == "untitled"
    assert slugify(_topic_from_script("TITLE: Why Cats Rule — Part 2")) == "why-cats-rule-part-2"


def test_stop_halts_queued_images(monkeypatch, tmp_path):
    generated: list[str] = []
    stop = threading.Event()

    def _fake_gen(prompt, path, profile, log_fn, **kw):
        generated.append(path.name)
        time.sleep(0.05)   # the submit loop queues every remaining prompt during this
        stop.set()         # ...then the user hits stop
        return True

    monkeypatch.setattr(images, "_load_anchor_parts", lambda profile: [])
    monkeypatch.setattr(images, "generate_image_google", _fake_gen)
    (tmp_path / "images").mkdir()
    results = generate_all_images(
        [{"num": f"{i:03d}", "prompt": "x"} for i in range(1, 21)],
        tmp_path, P, lambda m: None, stop_event=stop, max_workers=1,
    )

    assert generated == ["001.png"], generated   # 19 queued images must not bill
    assert list(results) == ["001"], results


def test_requirements_allow_output_config():
    # every Sonnet call passes output_config=; anthropic < 0.77.0 raises TypeError on it
    import re
    from pathlib import Path
    reqs = (Path(pipeline.__file__).parent / "requirements.txt").read_text()
    floor = re.search(r"^anthropic>=([\d.]+)", reqs, re.M)[1]
    assert tuple(map(int, floor.split("."))) >= (0, 77, 0), floor


def test_ctrl_c_drops_queued_images(monkeypatch, tmp_path):
    # the CLI has no stop_event; Ctrl-C must not leave ~150 queued images billing
    import _thread
    import pytest
    generated: list[str] = []

    def _fake_gen(prompt, path, profile, log_fn, **kw):
        generated.append(path.name)
        if len(generated) == 2:
            _thread.interrupt_main()   # Ctrl-C while image 2 is in flight, 18 still queued
        time.sleep(0.05)
        return True

    monkeypatch.setattr(images, "_load_anchor_parts", lambda profile: [])
    monkeypatch.setattr(images, "generate_image_google", _fake_gen)
    (tmp_path / "images").mkdir()
    with pytest.raises(KeyboardInterrupt):
        generate_all_images([{"num": f"{i:03d}", "prompt": "x"} for i in range(1, 21)],
                            tmp_path, P, lambda m: None, max_workers=1)
    assert len(generated) <= 3, generated


def test_tts_chunks_respect_the_limit_and_keep_every_word():
    from voiceover import _split_into_chunks
    for text in ("word " * 2000,                                   # no terminal punctuation
                 'He said "stop." Then more. ' * 400,              # sentences ending in a quote
                 "Short one. " * 900):
        chunks = _split_into_chunks(text)
        assert max(map(len, chunks)) <= 4500, max(map(len, chunks))
        assert " ".join(chunks).split() == text.split()           # nothing dropped
    assert _split_into_chunks("   \n ") == []                       # no empty ElevenLabs request


@pytest.mark.parametrize("topic", ["Café culture", "¿Por qué soñamos?", "— Why cats", "Why cats —",
                                   "a" * 59 + " b", "你好世界", "???", "🔥🔥"])
def test_every_slug_is_one_the_ui_accepts(topic):
    # the UI silently dropped review edits (and 400'd every route) for runs it couldn't name
    import ui
    slug = slugify(topic)
    assert slug and ui._safe_slug(slug), slug


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "/etc", "a\nb"])
def test_ui_slug_check_still_rejects_paths(bad):
    import ui
    assert not ui._safe_slug(bad)


@pytest.mark.parametrize("script,topic", [
    ("TITLE:\nWhy Cats Rule\n", "Why Cats Rule"),                 # empty TITLE: value
    ("```\nTITLE: Fenced Title\n```", "Fenced Title"),           # pasted from a chat
    ("**TITLE:** Bold Title\nbody", "Bold Title"),
    ("1. TITLE: Numbered Title", "Numbered Title"),
    ("TITLE : Spaced Title", "Spaced Title"),
    ("Subtitle: not a title line", "Subtitle: not a title line"),
])
def test_topic_from_script_finds_the_title(script, topic):
    assert _topic_from_script(script) == topic


def test_preamble_describes_exactly_the_anchors_sent(tmp_path):
    # a seed (anchor-00) or a failed anchor shifted every description by one
    for label in ("anchor-00", "anchor-01", "anchor-03"):               # anchor-02 failed to generate
        (tmp_path / f"{label}.png").write_bytes(b"x")
    (tmp_path / "manifest.yaml").write_text(
        "- {label: anchor-00, purpose: seed}\n- {label: anchor-01, purpose: cast sheet}\n"
        "- {label: anchor-02, purpose: props}\n- {label: anchor-03, purpose: wide shot}\n")
    pre = build_preamble({"art_style_block": "x", "max_anchors": 3}, tmp_path)
    described = [line.split(":")[0].strip() for line in pre.splitlines() if line.startswith("  anchor-")]
    assert described == ["anchor-00", "anchor-01", "anchor-03"]
