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


def _wait_for(calls, n=1):
    for _ in range(40):
        if len(calls) >= n:
            return
        time.sleep(0.05)


def test_approach_tap_is_refused_while_a_run_is_live(fresh_bot):
    bot._state.update(running=True, clarifying_topic="cats")
    ctx = _context()
    run(bot.on_button(_update(data="approach:1"), ctx))
    _wait_for(fresh_bot)
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
