"""/script routing: arming, paste, upload, and mode exclusivity. Run: python -m tests.test_bot_script_mode"""
import asyncio
import types

import bot

bot.ALLOWED_USER_ID = 1

started: list[dict] = []
_real_start_pipeline = bot._start_pipeline


async def _fake_start_pipeline(update, context, **kw):
    started.append(kw)


bot._start_pipeline = _fake_start_pipeline


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


def run(coro):
    return asyncio.run(coro)


def reset():
    started.clear()
    bot._state.update(running=False, script_mode=False, clarifying_mode=False, revision_mode=False)


SCRIPT = "TITLE: Why Cats Rule\n\n[00:00-00:35] HOOK\nbody"

# ── /script arms, and only when idle ─────────────────────────────────────────
reset()
u = _update()
run(bot.cmd_script(u, None))
assert bot._state["script_mode"] is True
assert "Send the script" in u.message.replies[0]

reset()
bot._state["running"] = True
u = _update()
run(bot.cmd_script(u, None))
assert bot._state["script_mode"] is False, "must not arm while a run is in flight"
assert "already running" in u.message.replies[0]

# ── pasted script starts production and disarms ──────────────────────────────
reset()
bot._state["script_mode"] = True
run(bot.on_text(_update(text=SCRIPT), None))
assert started == [{"script": SCRIPT}], started
assert bot._state["script_mode"] is False

# an empty paste stays armed instead of starting an empty run
reset()
bot._state["script_mode"] = True
u = _update(text="   ")
run(bot.on_text(u, None))
assert started == [] and bot._state["script_mode"] is True
assert "empty" in u.message.replies[0]

# ── uploaded .txt is decoded and used ────────────────────────────────────────
reset()
bot._state["script_mode"] = True
run(bot.on_document(_update(document=_Doc(SCRIPT.encode())), None))
assert started == [{"script": SCRIPT}], started

# non-UTF8 upload is reported, not crashed on, and stays armed
reset()
bot._state["script_mode"] = True
u = _update(document=_Doc(b"\xff\xfe\x00binary"))
run(bot.on_document(u, None))
assert started == [] and bot._state["script_mode"] is True
assert "UTF-8" in u.message.replies[0]

# an upload nobody asked for does not start a paid run
reset()
u = _update(document=_Doc(SCRIPT.encode()))
run(bot.on_document(u, None))
assert started == []
assert "/script" in u.message.replies[0]

# ── script_mode never shadows the other text modes ───────────────────────────
reset()
bot._state.update(script_mode=True, revision_mode=True)
run(bot.on_text(_update(text=SCRIPT), None))
assert started == [{"script": SCRIPT}], "armed /script wins over a stale revision flag"

# /run disarms it, so clarifying answers aren't swallowed as a script
reset()
bot._state["script_mode"] = True
bot.generate_clarifying_questions = lambda *a: "1. Angle?\nDefault: history"
bot._resolve_profile_for_bot = lambda: (None, "p")
context = types.SimpleNamespace(bot=types.SimpleNamespace(
    send_message=lambda *a, **k: asyncio.sleep(0)))
run(bot._send_clarifying_questions(_update(), context, topic="cats"))
assert bot._state["clarifying_mode"] is True
assert bot._state["script_mode"] is False, "an armed /script would eat the answers"


# ── _start_pipeline's script branch reaches run_from_script ──────────────────
import time

produced: list[tuple] = []
bot.run_from_script = lambda script, profile, topic, **kw: produced.append((script, profile, topic, sorted(kw))) or {"status": "complete"}
bot.list_profiles   = lambda: ["prof-a"]
bot.load_profile    = lambda name: f"<profile {name}>"
bot._relay_and_finish = lambda *a: asyncio.sleep(0)

reset()
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

print("test_bot_script_mode: PASS")
