"""Image-prompt expansion, stream retry, preamble, premade-script titles, stop handling."""
import threading
import time
import types

import anthropic
import pipeline
from pipeline import (_expand_short_prompts, _stream_text, _topic_from_script, build_preamble,
                      generate_all_images, slugify, GENERIC_ANCHOR_REFS)


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
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)   # don't actually back off

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

    monkeypatch.setattr(pipeline, "_load_anchor_parts", lambda profile: [])
    monkeypatch.setattr(pipeline, "generate_image_google", _fake_gen)
    (tmp_path / "images").mkdir()
    results = generate_all_images(
        [{"num": f"{i:03d}", "prompt": "x"} for i in range(1, 21)],
        tmp_path, P, lambda m: None, stop_event=stop, max_workers=1,
    )

    assert generated == ["001.png"], generated   # 19 queued images must not bill
    assert list(results) == ["001"], results
