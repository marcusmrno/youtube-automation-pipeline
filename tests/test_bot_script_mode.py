"""/script routing: arming, paste, upload, and mode exclusivity."""
import asyncio
import time
import types

import pytest

import bot

_real_start_pipeline = bot._start_pipeline
SCRIPT = "TITLE: Why Cats Rule\n\n[00:00-00:35] HOOK\nbody"


class _Msg:
    def __init__(self, text=None, document=None):
        self.text, self.document, self.replies = text, document, []

    async def reply_text(self, text, **kw):
        self.replies.append(text)


def _update(text=None, document=None):
    msg = _Msg(text, document)
    return types.SimpleNamespace(message=msg, effective_user=types.SimpleNamespace(id=1),
                                 effective_chat=types.SimpleNamespace(id=1))


class _Doc:
    """Telegram Document stand-in: get_file().download_as_bytearray()."""
    def __init__(self, data: bytes, name="script.txt"):
        self.data, self.file_name = data, name

    async def get_file(self):
        outer = self
        async def download_as_bytearray():
            return bytearray(outer.data)
        return types.SimpleNamespace(download_as_bytearray=download_as_bytearray)


run = asyncio.run


@pytest.fixture(autouse=True)
def started(monkeypatch):
    """Fresh bot state per test; _start_pipeline records its kwargs instead of running."""
    calls: list[dict] = []

    async def _fake_start_pipeline(update, context, **kw):
        calls.append(kw)

    monkeypatch.setattr(bot, "ALLOWED_USER_ID", 1)
    monkeypatch.setattr(bot, "_start_pipeline", _fake_start_pipeline)
    monkeypatch.setattr(bot, "_state", dict(bot._state, running=False, script_mode=False,
                                            clarifying_mode=False, revision_mode=False))
    return calls


# ── /script arms, and only when idle ─────────────────────────────────────────
def test_script_arms_when_idle():
    u = _update()
    run(bot.cmd_script(u, None))
    assert bot._state["script_mode"] is True
    assert "Send the script" in u.message.replies[0]


def test_script_refuses_while_running():
    bot._state["running"] = True
    u = _update()
    run(bot.cmd_script(u, None))
    assert bot._state["script_mode"] is False, "must not arm while a run is in flight"
    assert "already running" in u.message.replies[0]


# ── pasted script starts production and disarms ──────────────────────────────
def test_paste_starts_and_disarms(started):
    bot._state["script_mode"] = True
    run(bot.on_text(_update(text=SCRIPT), None))
    assert started == [{"script": SCRIPT}], started
    assert bot._state["script_mode"] is False


def test_empty_paste_stays_armed(started):
    bot._state["script_mode"] = True
    u = _update(text="   ")
    run(bot.on_text(u, None))
    assert started == [] and bot._state["script_mode"] is True
    assert "empty" in u.message.replies[0]


# ── uploaded .txt is decoded and used ────────────────────────────────────────
def test_upload_is_decoded_and_used(started):
    bot._state["script_mode"] = True
    run(bot.on_document(_update(document=_Doc(SCRIPT.encode())), None))
    assert started == [{"script": SCRIPT}], started


def test_non_utf8_upload_is_reported(started):
    bot._state["script_mode"] = True
    u = _update(document=_Doc(b"\xff\xfe\x00binary"))
    run(bot.on_document(u, None))
    assert started == [] and bot._state["script_mode"] is True
    assert "UTF-8" in u.message.replies[0]


def test_unrequested_upload_does_not_start(started):
    u = _update(document=_Doc(SCRIPT.encode()))
    run(bot.on_document(u, None))
    assert started == []   # an upload nobody asked for does not start a paid run
    assert "/script" in u.message.replies[0]


# ── script_mode never shadows the other text modes ───────────────────────────
def test_armed_script_beats_stale_revision_flag(started):
    bot._state.update(script_mode=True, revision_mode=True)
    run(bot.on_text(_update(text=SCRIPT), None))
    assert started == [{"script": SCRIPT}], "armed /script wins over a stale revision flag"


def test_run_disarms_script_mode(monkeypatch):
    bot._state["script_mode"] = True
    monkeypatch.setattr(bot, "generate_clarifying_questions", lambda *a: "1. Angle?\nDefault: history")
    monkeypatch.setattr(bot, "_resolve_profile_for_bot", lambda: (None, "p"))
    context = types.SimpleNamespace(bot=types.SimpleNamespace(
        send_message=lambda *a, **k: asyncio.sleep(0)))
    run(bot._send_clarifying_questions(_update(), context, topic="cats"))
    assert bot._state["clarifying_mode"] is True
    assert bot._state["script_mode"] is False, "an armed /script would eat the answers"


# ── _start_pipeline's script branch reaches run_from_script ──────────────────
def test_start_pipeline_script_branch(monkeypatch):
    produced: list[tuple] = []
    monkeypatch.setattr(bot, "run_from_script", lambda script, profile, topic, **kw:
                        produced.append((script, profile, topic, sorted(kw))) or {"status": "complete"})
    monkeypatch.setattr(bot, "list_profiles", lambda: ["prof-a"])
    monkeypatch.setattr(bot, "load_profile", lambda name: f"<profile {name}>")
    monkeypatch.setattr(bot, "_relay_and_finish", lambda *a: asyncio.sleep(0))
    bot._state["profile_name"] = None
    ctx = types.SimpleNamespace(application=types.SimpleNamespace(bot=None))
    ctx.bot = types.SimpleNamespace(send_message=lambda *a, **k: asyncio.sleep(0))
    ctx.application.bot = ctx.bot

    run(_real_start_pipeline(_update(), ctx, script=SCRIPT))
    for _ in range(50):
        if produced:
            break
        time.sleep(0.05)

    (script_arg, profile_arg, topic_arg, kwargs), = produced
    assert script_arg == SCRIPT
    assert profile_arg == "<profile prof-a>", profile_arg
    assert topic_arg == "Why Cats Rule", topic_arg          # from the TITLE: line
    assert kwargs == ["progress_callback", "stop_event"], kwargs
    assert bot._state["run_slug"] == "why-cats-rule", bot._state["run_slug"]
