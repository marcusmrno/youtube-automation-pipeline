"""Bot handlers driven with fake Telegram updates. Pipeline entry points are faked; nothing reaches Telegram."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import bot


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
