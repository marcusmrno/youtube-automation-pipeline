"""Bot handlers driven with fake Telegram updates. Pipeline entry points are faked; nothing reaches Telegram."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import bot

_REAL_RELAY = bot._relay_and_finish   # the fixture stubs it for runs


async def _noop(*a, **k):
    return None


class _Msg:
    def __init__(self, text=None, document=None):
        self.text, self.document, self.replies = text, document, []

    async def reply_text(self, text, **kw):
        self.replies.append(text)

    async def reply_document(self, document, filename, **kw):
        self.replies.append(("document", filename, document.getbuffer().nbytes))


def _update(text=None, data=None, document=None):
    msg = _Msg(text, document)
    query = SimpleNamespace(data=data, message=msg, answer=_noop,
                            edit_message_reply_markup=_noop, edit_message_text=_noop) if data else None
    return SimpleNamespace(message=msg, callback_query=query, effective_user=SimpleNamespace(id=1),
                           effective_chat=SimpleNamespace(id=1, send_message=_noop, send_media_group=_noop))


def _context(*args):
    sent = []

    async def send_message(*a, **k):
        sent.append(k.get("text") or (a[1] if len(a) > 1 else a[0]))

    tg = SimpleNamespace(send_message=send_message)
    return SimpleNamespace(args=list(args), bot=tg, application=SimpleNamespace(bot=tg), sent=sent)


run = asyncio.run


@pytest.fixture(autouse=True)
def fresh_bot(monkeypatch, tmp_path):
    """Fresh bot state, output in tmp_path, one profile, and recorded (never real) pipeline runs."""
    calls = []
    monkeypatch.setattr(bot, "ALLOWED_USER_ID", 1)
    monkeypatch.setattr(bot, "_state", dict(bot._state, milestone_sent=set()))
    monkeypatch.setattr(bot, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(bot, "list_profiles", lambda: ["p"])
    monkeypatch.setattr(bot, "load_profile", lambda name: f"<profile {name}>")
    monkeypatch.setattr(bot, "_relay_and_finish", _noop)
    for name in ("run_pipeline", "resume_pipeline", "run_from_script"):
        monkeypatch.setattr(bot, name, lambda *a, _n=name, **k: calls.append(_n) or {"status": "complete"})
    return calls


def _wait_for(calls, n=1, timeout=2.0):
    """Runs start on a worker thread; give it a moment to call the (faked) pipeline."""
    end = time.monotonic() + timeout
    while len(calls) < n and time.monotonic() < end:
        time.sleep(0.02)


def test_approach_tap_is_refused_while_a_run_is_live(fresh_bot):
    bot._state.update(running=True, clarifying_topic="cats")
    ctx = _context()
    run(bot.on_button(_update(data="approach:1"), ctx))
    _wait_for(fresh_bot, timeout=0.3)
    assert fresh_bot == []
    assert any("already running" in m for m in ctx.sent)


def test_stop_disarms_revision_and_planning(monkeypatch, tmp_path):
    paid = []
    monkeypatch.setattr(bot, "revise_script", lambda *a, **k: paid.append("revise") or "REVISED")
    monkeypatch.setattr(bot, "_send_approach_pitches", lambda *a, **k: paid.append("pitches") or _noop())
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "script.txt").write_text("ORIGINAL")
    bot._state.update(running=True, run_slug="r1", revision_mode=True, clarifying_mode=True,
                      approval_event=threading.Event())
    run(bot.cmd_stop(_update(), _context()))
    run(bot.on_text(_update(text="ok thanks, never mind"), _context()))
    assert paid == []                                              # no paid Sonnet/Haiku call
    assert (tmp_path / "r1" / "script.txt").read_text() == "ORIGINAL"


def test_stale_review_buttons_say_nothing_is_waiting():
    bot._state["approval_event"] = threading.Event()
    run(bot.cmd_stop(_update(), _context()))                       # sets the event
    u = _update(data="approve")
    run(bot.on_button(u, _context()))
    assert u.message.replies == ["Nothing is waiting for approval."]


@pytest.mark.parametrize("arg", ["..", "../..", "/etc", "r1/..", "."])
def test_slugs_outside_output_are_refused(fresh_bot, tmp_path, arg):
    # "/download .." zipped the repo, .env included, and uploaded it to Telegram
    (tmp_path / "r1").mkdir()
    u = _update()
    run(bot.cmd_download(u, _context(arg)))
    assert not any(isinstance(r, tuple) for r in u.message.replies), u.message.replies
    run(bot.cmd_resume(_update(), _context(arg)))
    _wait_for(fresh_bot, timeout=0.3)
    assert fresh_bot == []


def test_a_failed_start_does_not_leave_the_bot_running(monkeypatch, tmp_path):
    def load(name):
        if name == "renamed-profile":
            raise ValueError("Profile 'renamed-profile' not found. Available: ['p']")
        return f"<profile {name}>"
    monkeypatch.setattr(bot, "load_profile", load)
    (tmp_path / "old-run").mkdir()
    (tmp_path / "old-run" / "profile.txt").write_text("renamed-profile")
    ctx = _context("old-run")
    run(bot.cmd_resume(_update(), ctx))
    assert bot._state["running"] is False
    assert any("renamed-profile" in m for m in ctx.sent)          # the user is told why


def test_a_stale_approach_tap_says_the_plan_expired():
    u = _update(data="approach:2")                                 # e.g. from before a bot restart
    run(bot.on_button(u, _context()))
    assert bot._state["running"] is False
    assert u.message.replies == ["That plan expired — send /run again."]


def test_thumbnail_buttons_fit_telegrams_64_byte_limit(monkeypatch):
    slug = "a-topic-long-enough-to-hit-slugifys-sixty-character-cap-here"   # 60 chars, the slugify cap
    markups, picked = [], []

    async def send_message(*a, **k):
        markups.append(k.get("reply_markup"))

    chat = SimpleNamespace(send_message=send_message, send_media_group=_noop)
    data = {"titles": [], "thumbnails": [{"filename": f"thumbnails/thumb-0{i}.png"} for i in (1, 2, 3)]}
    run(bot._send_metadata_view(chat, data, slug))
    buttons = [b for m in markups if m for row in m.inline_keyboard for b in row]
    assert len(buttons) == 3
    assert all(len(b.callback_data.encode()) <= 64 for b in buttons), [b.callback_data for b in buttons]

    monkeypatch.setattr(bot._metadata_mod, "pick_thumbnail", lambda s, i: picked.append((s, i)) or data)
    u = _update(data=buttons[1].callback_data)
    u.callback_query.message.chat = chat
    run(bot.on_button(u, _context()))
    assert picked == [(slug, 1)]


@pytest.mark.parametrize("reply,expected", [
    ("2. adults only", ["history", "adults only", "playful"]),            # only Q2 answered
    ("1: science\n2: kids\n3: serious", ["science", "kids", "serious"]),
    ("1) science\n3 - serious", ["science", "teens", "serious"]),
    ("just make it fun", ["just make it fun", "teens", "playful"]),      # unnumbered -> Q1
])
def test_answers_pair_with_the_number_typed(monkeypatch, reply, expected):
    captured = []
    monkeypatch.setattr(bot, "_send_approach_pitches", lambda u, c, answers: captured.append(answers) or _noop())
    bot._state.update(clarifying_mode=True, clarifying_questions=[
        {"question": "Angle?", "default": "history"},
        {"question": "Audience?", "default": "teens"},
        {"question": "Tone?", "default": "playful"}])
    run(bot.on_text(_update(text=reply), _context()))
    got = [line.split("Answer: ", 1)[1] for line in captured[0].splitlines() if "Answer: " in line]
    assert got == expected


def _longest_stall(coro):
    """Run coro next to a 20 ms ticker; return the longest gap the event loop left it."""
    async def main():
        gaps, last = [0.0], time.monotonic()

        async def ticker():
            nonlocal last
            while True:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)   # let the ticker start...
        await coro
        await asyncio.sleep(0.05)   # ...and record the gap the handler caused
        t.cancel()
        return max(gaps)
    return run(main())


def test_revision_does_not_freeze_the_bot(monkeypatch, tmp_path):
    # while Sonnet revises, /stop and the buttons must still be handled
    monkeypatch.setattr(bot, "revise_script", lambda *a, **k: time.sleep(0.5) or "REVISED")
    monkeypatch.setattr(bot, "_send_script_for_review", _noop)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "script.txt").write_text("ORIGINAL")
    bot._state.update(running=True, run_slug="r1", revision_mode=True)
    assert _longest_stall(bot.on_text(_update(text="make it punchier"), _context())) < 0.3


def test_zipping_does_not_freeze_the_bot(monkeypatch, tmp_path):
    import zipfile
    real_write = zipfile.ZipFile.write
    monkeypatch.setattr(zipfile.ZipFile, "write", lambda self, *a, **k: time.sleep(0.25) or real_write(self, *a, **k))
    (tmp_path / "r1").mkdir()
    for f in ("script.txt", "tts_script.txt"):
        (tmp_path / "r1" / f).write_text("x")
    assert _longest_stall(bot.cmd_download(_update(), _context("r1"))) < 0.3


def test_download_parts_fit_telegrams_upload_limit(tmp_path):
    import os
    images = tmp_path / "r1" / "images"
    images.mkdir(parents=True)
    for i in range(60):                                   # 60 MB of incompressible "images"
        (images / f"{i:03d}.png").write_bytes(os.urandom(1 << 20))
    u = _update()
    run(bot.cmd_download(u, _context("r1")))
    sizes = [r[2] for r in u.message.replies if isinstance(r, tuple)]
    assert len(sizes) >= 2 and max(sizes) <= 50_000_000, sizes   # Bot API upload cap: 50 MB


class _Doc:
    """Telegram Document stand-in."""
    def __init__(self, data: bytes, name: str):
        self.data, self.file_name = data, name

    async def get_file(self):
        async def download_as_bytearray():
            return bytearray(self.data)
        return SimpleNamespace(download_as_bytearray=download_as_bytearray)


@pytest.mark.parametrize("doc,started_with", [
    (_Doc("TITLE: Why Cats Rule\nbody".encode("utf-8-sig"), "script.txt"), "TITLE: Why Cats Rule"),  # Notepad BOM
    (_Doc(b"{\\rtf1\\ansi TITLE: Why Cats Rule}", "script.rtf"), None),                            # TextEdit default
])
def test_uploads_must_be_plain_text_scripts(monkeypatch, doc, started_with):
    started = []
    monkeypatch.setattr(bot, "_start_pipeline", lambda u, c, **kw: started.append(kw["script"]) or _noop())
    bot._state["script_mode"] = True
    u = _update(document=doc)
    run(bot.on_document(u, _context()))
    if started_with:
        assert started and started[0].startswith(started_with)   # the TITLE: line still names the run
    else:
        assert started == [] and "txt" in u.message.replies[0]


def test_per_image_retries_are_not_relayed():
    # a Gemini outage over 150 images sent ~750 of these to the phone
    assert not bot._should_relay("  ⚠️  Google AI attempt 1 error: 429 RESOURCE_EXHAUSTED")
    assert not bot._should_relay("  ❌  Image 001.png failed after 3 attempts — skipping")
    assert bot._should_relay("❌  Voiceover failed on chunk 1")
    assert bot._should_relay("✅  148/150 images ready")


def test_metadata_generation_is_tracked_and_not_duplicated(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bot._metadata_mod, "generate_metadata",
                        lambda slug, *a, **k: calls.append(slug) or time.sleep(0.4))
    monkeypatch.setattr(bot._metadata_mod, "load_metadata", lambda slug: None)
    (tmp_path / "r1").mkdir()
    first, second, status = _update(), _update(), _update()

    async def both():
        async def check_status():
            await asyncio.sleep(0.1)
            await bot.cmd_status(status, _context())
        await asyncio.gather(bot.cmd_metadata(first, _context("r1", "regenerate")),
                             bot.cmd_metadata(second, _context("r1", "regenerate")),
                             check_status())
    run(both())
    assert calls == ["r1"]                                      # the second request didn't race the first
    assert any("already" in r for r in second.message.replies + first.message.replies)
    assert any("metadata" in r.lower() for r in status.message.replies)   # /status knows about it


def test_help_lists_every_command():
    u = _update()
    run(bot.cmd_start(u, _context()))
    for cmd in ("/run", "/script", "/runs", "/resume", "/download", "/metadata", "/profile", "/status", "/stop"):
        assert cmd in u.message.replies[0], cmd


def test_image_milestones_say_started_not_done():
    # the pipeline logs "Generating image i/N" when image i starts, so 150/150 isn't "done"
    import queue
    lq = queue.Queue()
    lq.put({"type": "log", "msg": "🖼  Generating image 150/150 (150)"})
    lq.put({"type": "done"})
    sent = []
    app = SimpleNamespace(bot=SimpleNamespace(send_message=lambda **k: sent.append(k["text"]) or _noop()))
    run(_REAL_RELAY(app, 1, lq))
    assert sent == ["🖼 Images: 100% started (150/150)"]


def test_bot_does_not_poll_for_edited_messages(monkeypatch):
    # PTB handlers match edited messages, where update.message is None, so editing an old message crashed them
    polled = {}
    app = SimpleNamespace(add_handler=lambda h: None, run_polling=lambda **k: polled.update(k))
    builder = SimpleNamespace(build=lambda: app)
    builder.token = lambda t: builder
    builder.concurrent_updates = lambda c: builder
    monkeypatch.setattr(bot, "BOT_TOKEN", "token")
    monkeypatch.setattr(bot, "Application", SimpleNamespace(builder=lambda: builder))
    bot.main()
    assert set(polled["allowed_updates"]) == {"message", "callback_query"}


def test_several_profiles_and_none_chosen_asks_for_profile(monkeypatch, fresh_bot):
    # with profiles/example shipped, every real user has two; the bot silently used the first
    asked = []
    monkeypatch.setattr(bot, "list_profiles", lambda: ["example", "my-channel"])
    monkeypatch.setattr(bot, "generate_clarifying_questions", lambda *a: asked.append(a) or "1. Q?")
    u = _update()
    run(bot.cmd_run(u, _context("cats")))
    ctx = _context()
    run(bot._start_pipeline(_update(), ctx, script="TITLE: T\nbody"))
    _wait_for(fresh_bot, timeout=0.3)
    assert asked == [] and fresh_bot == []
    assert "/profile" in u.message.replies[-1] and "/profile" in ctx.sent[-1]
    bot._state["profile_name"] = "my-channel"                 # an explicit choice works
    run(bot._start_pipeline(_update(), _context(), script="TITLE: T\nbody"))
    _wait_for(fresh_bot)
    assert fresh_bot == ["run_from_script"]


def test_metadata_and_revise_use_the_runs_profile(monkeypatch, tmp_path):
    used = []
    monkeypatch.setattr(bot, "list_profiles", lambda: ["alpha", "beta"])
    monkeypatch.setattr(bot._metadata_mod, "load_metadata", lambda slug: None)
    monkeypatch.setattr(bot._metadata_mod, "generate_metadata", lambda slug, profile, **k: used.append(profile))
    monkeypatch.setattr(bot, "revise_script", lambda script, fb, topic, profile, client: used.append(profile) or "R")
    monkeypatch.setattr(bot, "_send_script_for_review", _noop)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "profile.txt").write_text("beta")
    (tmp_path / "r1" / "script.txt").write_text("S")
    run(bot.cmd_metadata(_update(), _context("r1", "regenerate")))
    bot._state.update(running=True, run_slug="r1", revision_mode=True)
    run(bot.on_text(_update(text="shorter"), _context()))
    assert used == ["<profile beta>", "<profile beta>"]


def test_resume_lists_incomplete_runs_beyond_the_newest_ten(monkeypatch, tmp_path):
    import os
    for i in range(12):                                         # 11 complete runs, then an older incomplete one
        d = tmp_path / f"run-{i:02d}"
        d.mkdir()
        os.utime(d, (1_000_000 - i, 1_000_000 - i))
    monkeypatch.setattr(bot, "run_status", lambda slug: {"missing": ["voiceover"] if slug == "run-11" else []})
    u = _update()
    run(bot.cmd_resume(u, _context()))
    assert "run-11" in u.message.replies[0]


CANONICAL = [{"question": "Angle?", "default": "history"}, {"question": "Audience?", "default": "teens"}]


@pytest.mark.parametrize("text", [
    "1. Angle?\n   Default: history\n\n2. Audience?\n   Default: teens",
    "1) Angle?\n   Default: history\n2) Audience?\n   Default: teens",
    "**1. Angle?**\n   **Default:** history\n**2. Audience?**\n   **Default:** teens",
    "Here are some questions:\n\n1. Angle?\n   Default: history\n2. Audience?\n   Default: teens",
    "1. Angle?\n   DEFAULT: history\n2. Audience?\n   default: teens",
])
def test_clarifying_questions_parse_in_common_variants(text):
    # the same parse rules live in templates/index.html (parseClarifyingQuestions)
    assert bot._parse_clarifying_questions(text) == CANONICAL
