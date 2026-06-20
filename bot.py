"""
YouTube Pipeline — Telegram Bot Interface
Run: python bot.py
"""

import asyncio
import io
import os
import re
import queue
import threading
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

import pipeline
from pipeline import (
    OUTPUT_ROOT,
    ANTHROPIC_KEY,
    CLAUDE_MODEL,
    run_pipeline,
    resume_pipeline,
    run_status,
    slugify,
    load_claude_md,
    load_style_sheet,
)

load_dotenv()

BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ALLOWED_USER_ID = int(os.getenv("TELEGRAM_USER_ID", "0"))

_state: dict = {
    "running":         False,
    "run_slug":        None,
    "stop_event":      None,
    "loop":            None,
    "approval_event":  None,
    "approval_result": None,
    "revision_mode":   False,
    "chat_id":         None,
    "log_queue":       None,
    "total_images":    0,
    "done_images":     0,
    "milestone_sent":  set(),
}


# ── Auth ───────────────────────────────────────────────────────────────────────

def auth(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or update.effective_user.id != ALLOWED_USER_ID:
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


# ── Log helpers ────────────────────────────────────────────────────────────────

def _should_relay(msg: str) -> bool:
    """Filter out per-image sub-step chatter; keep top-level progress lines."""
    m = msg.strip()
    if not m:
        return False
    # Skip indented verbose sub-steps (Google AI request attempts, TTS chunks, etc.)
    # but still forward failures/warnings that start with an emoji
    if msg.startswith("  ") and not any(m.startswith(c) for c in ("❌", "⚠️")):
        return False
    # Per-image lines are tracked via milestones — skip them to avoid flood
    if "Generating image " in msg and "/" in msg:
        return False
    return True


def _check_milestone(msg: str) -> int | None:
    """Parse image progress from a log line; return newly crossed milestone % or None."""
    m = re.search(r"Generating image\s+(\d+)/(\d+)", msg)
    if not m:
        return None
    done  = int(m.group(1))
    total = int(m.group(2))
    _state["done_images"]  = done
    _state["total_images"] = total
    if total == 0:
        return None
    pct = int(done / total * 100)
    for milestone in (25, 50, 75, 100):
        if pct >= milestone and milestone not in _state["milestone_sent"]:
            _state["milestone_sent"].add(milestone)
            return milestone
    return None


# ── Script review ──────────────────────────────────────────────────────────────

async def _send_script_for_review(bot, chat_id: int, script: str) -> None:
    await bot.send_message(chat_id=chat_id, text="📋 *Script ready for review:*", parse_mode="Markdown")

    chunks = [script[i:i+4000] for i in range(0, max(len(script), 1), 4000)]
    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        if is_last:
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data="approve"),
                    InlineKeyboardButton("❌ Reject",  callback_data="reject"),
                ],
                [InlineKeyboardButton("✏️ Revise", callback_data="revise")],
            ])
            await bot.send_message(
                chat_id=chat_id,
                text=f"```\n{chunk}\n```",
                parse_mode="Markdown",
                reply_markup=keyboard,
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=f"```\n{chunk}\n```",
                parse_mode="Markdown",
            )


def _make_approval_callback(bot, chat_id: int, loop: asyncio.AbstractEventLoop):
    """Returns a sync approval_callback for run_pipeline — blocks until user responds."""
    def approval_callback(script_text: str) -> bool:
        evt = threading.Event()
        _state["approval_event"]  = evt
        _state["approval_result"] = None

        asyncio.run_coroutine_threadsafe(
            _send_script_for_review(bot, chat_id, script_text),
            loop,
        )
        evt.wait()
        return bool(_state["approval_result"])
    return approval_callback


# ── Pipeline relay + completion ────────────────────────────────────────────────

async def _relay_and_finish(app: Application, chat_id: int, lq: queue.Queue) -> None:
    complete_result = None

    while True:
        try:
            event = lq.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.5)
            continue

        etype = event.get("type")

        if etype == "log":
            msg = event.get("msg", "")
            milestone = _check_milestone(msg)
            if milestone:
                done  = _state["done_images"]
                total = _state["total_images"]
                try:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=f"🖼 Images: {milestone}% done ({done}/{total})",
                    )
                except Exception:
                    pass
            if _should_relay(msg):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=msg[:4096])
                except Exception:
                    pass

        elif etype == "complete":
            complete_result = event.get("result")

        elif etype == "done":
            break

    _state["running"] = False

    if complete_result:
        await _deliver_completion(app, chat_id, complete_result)


async def _deliver_completion(app: Application, chat_id: int, result: dict) -> None:
    slug     = result.get("out_dir", "")
    slug     = Path(slug).name if slug else (_state["run_slug"] or "")
    n_images = result.get("images", 0)
    audio_ok = result.get("audio", "failed") != "failed"

    await app.bot.send_message(
        chat_id=chat_id,
        text=(
            f"🎬 *Pipeline complete!*\n\n"
            f"Run: `{slug}`\n"
            f"Images: {n_images}\n"
            f"Audio: {'✅' if audio_ok else '❌'}"
        ),
        parse_mode="Markdown",
    )

    img_dir = OUTPUT_ROOT / slug / "images"
    if not img_dir.exists():
        return
    imgs = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not imgs:
        return

    step    = max(1, len(imgs) // 10)
    samples = imgs[::step][:10]
    handles = [open(p, "rb") for p in samples]
    try:
        media = [InputMediaPhoto(media=f) for f in handles]
        await app.bot.send_media_group(chat_id=chat_id, media=media)
    except Exception as e:
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Image delivery failed: {e}")
    finally:
        for f in handles:
            f.close()


# ── Command handlers ───────────────────────────────────────────────────────────

@auth
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎬 *YT Pipeline Bot*\n\n"
        "/run <topic> — start a new pipeline run\n"
        "/runs — list recent runs and their status\n"
        "/resume [slug] — resume an incomplete run\n"
        "/download [slug] — download run assets as zip (latest if omitted)\n"
        "/status — show current run status\n"
        "/stop — stop the current run",
        parse_mode="Markdown",
    )


@auth
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _state["running"]:
        slug  = _state["run_slug"] or "?"
        done  = _state["done_images"]
        total = _state["total_images"]
        pct   = f"{int(done/total*100)}%" if total else "starting…"
        await update.message.reply_text(
            f"🔄 Running: `{slug}`\n"
            f"Images: {done}/{total} ({pct})",
            parse_mode="Markdown",
        )
        return

    if not OUTPUT_ROOT.exists():
        await update.message.reply_text("💤 No pipeline running. No output folder yet.")
        return

    dirs = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        await update.message.reply_text("💤 No pipeline running.")
        return

    s       = run_status(dirs[0])
    missing = ", ".join(s["missing"]) if s["missing"] else "none"
    await update.message.reply_text(
        f"💤 No pipeline running\n\n"
        f"Last run: `{s['run_slug']}`\n"
        f"Images: {s['images_on_disk']}/{s['total_prompts']}\n"
        f"Audio: {'✅' if s['has_audio'] else '❌'}\n"
        f"Missing: {missing}",
        parse_mode="Markdown",
    )


@auth
async def cmd_runs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not OUTPUT_ROOT.exists():
        await update.message.reply_text("No output folder yet.")
        return
    dirs = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        await update.message.reply_text("No runs found.")
        return

    lines = []
    for slug in dirs[:15]:
        s     = run_status(slug)
        img   = f"{s['images_on_disk']}/{s['total_prompts']}" if s["total_prompts"] else "?"
        audio = "🔊" if s["has_audio"] else "🔇"
        flag  = " ⚠️" if s["missing"] else " ✅"
        lines.append(f"{audio} `{slug}` — {img} imgs{flag}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@auth
async def cmd_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Resolve slug: explicit arg, or most recent run
    if context.args:
        slug = context.args[0].strip()
    elif OUTPUT_ROOT.exists():
        dirs = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()], reverse=True)
        slug = dirs[0] if dirs else None
    else:
        slug = None

    if not slug:
        await update.message.reply_text("Usage: /download <slug>  (or omit for the latest run)")
        return

    run_dir = OUTPUT_ROOT / slug
    if not run_dir.exists():
        await update.message.reply_text(f"❌ Run `{slug}` not found.", parse_mode="Markdown")
        return

    await update.message.reply_text(f"📦 Zipping `{slug}`…", parse_mode="Markdown")

    LIMIT = 49 * 1_048_576  # 49 MB — Telegram bot limit is 50 MB

    def _make_zip(paths: list) -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in paths:
                zf.write(path, path.relative_to(run_dir))
        buf.seek(0)
        return buf

    # Collect all files — text assets first, then images
    all_files = sorted(run_dir.rglob("*"), key=lambda p: (
        0 if "images" not in p.parts else 1, p
    ))
    all_files = [p for p in all_files if not p.is_dir()]

    # Split into batches under the Telegram limit
    batches: list[list] = []
    current_batch: list = []
    current_size = 0
    for path in all_files:
        file_size = path.stat().st_size
        if current_batch and current_size + file_size > LIMIT:
            batches.append(current_batch)
            current_batch, current_size = [], 0
        current_batch.append(path)
        current_size += file_size
    if current_batch:
        batches.append(current_batch)

    total = len(batches)
    for i, batch in enumerate(batches, 1):
        part_label = f"part {i}/{total}" if total > 1 else "complete"
        filename   = f"{slug}-part{i}.zip" if total > 1 else f"{slug}.zip"
        buf        = _make_zip(batch)
        size_mb    = buf.getbuffer().nbytes / 1_048_576
        await update.message.reply_document(
            document=buf,
            filename=filename,
            caption=f"✅ {slug} ({part_label}) — {size_mb:.1f} MB",
        )


@auth
async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    se = _state.get("stop_event")
    if se:
        se.set()
    ae = _state.get("approval_event")
    if ae:
        _state["approval_result"] = False
        ae.set()
    _state["running"] = False
    await update.message.reply_text("🛑 Stop signal sent.")


@auth
async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _state["running"]:
        await update.message.reply_text("⚠️ A pipeline is already running. Use /stop first.")
        return
    topic = " ".join(context.args).strip()
    if not topic:
        await update.message.reply_text("Usage: /run <topic>")
        return
    await _start_pipeline(update, context, topic=topic)


@auth
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _state["running"]:
        await update.message.reply_text("⚠️ A pipeline is already running. Use /stop first.")
        return

    if not context.args:
        if not OUTPUT_ROOT.exists():
            await update.message.reply_text("No runs found.")
            return
        dirs = sorted([d.name for d in OUTPUT_ROOT.iterdir() if d.is_dir()], reverse=True)
        incomplete = []
        for slug in dirs[:10]:
            s = run_status(slug)
            if s["missing"]:
                incomplete.append(f"• `{slug}` — missing: {', '.join(s['missing'])}")
        if not incomplete:
            await update.message.reply_text(
                "No incomplete runs found.\n\nUse /resume <slug> to force-resume a specific run."
            )
            return
        await update.message.reply_text(
            "Incomplete runs:\n" + "\n".join(incomplete) + "\n\nUse /resume <slug> to continue.",
            parse_mode="Markdown",
        )
        return

    await _start_pipeline(update, context, run_slug=context.args[0].strip())


# ── Pipeline runner ────────────────────────────────────────────────────────────

async def _start_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    topic: str | None = None,
    run_slug: str | None = None,
) -> None:
    chat_id = update.effective_chat.id
    app     = context.application
    loop    = asyncio.get_running_loop()

    lq = queue.Queue()
    se = threading.Event()

    _state.update({
        "running":         True,
        "run_slug":        run_slug or slugify(topic or ""),
        "stop_event":      se,
        "loop":            loop,
        "approval_event":  None,
        "approval_result": None,
        "revision_mode":   False,
        "chat_id":         chat_id,
        "log_queue":       lq,
        "total_images":    0,
        "done_images":     0,
        "milestone_sent":  set(),
    })

    def progress_cb(msg: str):
        lq.put({"type": "log", "msg": msg})

    if topic:
        approval_cb = _make_approval_callback(app.bot, chat_id, loop)
        label = f"▶️ Starting pipeline: *{topic}*"
        fn    = lambda: run_pipeline(
            topic,
            progress_callback=progress_cb,
            stop_event=se,
            approval_callback=approval_cb,
        )
    else:
        label = f"▶️ Resuming: *{run_slug}*"
        fn    = lambda: resume_pipeline(
            run_slug,
            progress_callback=progress_cb,
            stop_event=se,
        )

    await update.message.reply_text(label, parse_mode="Markdown")

    def worker():
        try:
            result = fn()
            status = result.get("status", "?")
            if status == "complete":
                lq.put({"type": "complete", "result": result})
            elif status == "rejected":
                lq.put({"type": "log", "msg": "🚫 Script rejected — pipeline stopped."})
            elif status == "cancelled":
                lq.put({"type": "log", "msg": "🛑 Pipeline cancelled."})
            elif status == "error":
                lq.put({"type": "log", "msg": f"❌ Pipeline error: {result.get('reason', '?')}"})
        except Exception as e:
            lq.put({"type": "log", "msg": f"❌ Pipeline error: {e}"})
        finally:
            lq.put({"type": "done"})

    threading.Thread(target=worker, daemon=True).start()
    asyncio.create_task(_relay_and_finish(app, chat_id, lq))


# ── Inline button callback ─────────────────────────────────────────────────────

@auth
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    ae   = _state.get("approval_event")

    if data == "approve":
        _state["approval_result"] = True
        if ae:
            ae.set()
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("✅ Approved — starting production...")

    elif data == "reject":
        _state["approval_result"] = False
        _state["running"]         = False
        se = _state.get("stop_event")
        if se:
            se.set()
        if ae:
            ae.set()
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("❌ Rejected — pipeline stopped.")

    elif data == "revise":
        _state["revision_mode"] = True
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("✏️ Send your revision feedback:")


# ── Text handler (revision feedback) ──────────────────────────────────────────

@auth
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _state.get("revision_mode"):
        if not _state.get("running"):
            await update.message.reply_text("No pipeline running. Use /run <topic> to start.")
        return

    feedback = update.message.text.strip()
    if not feedback:
        return

    _state["revision_mode"] = False

    slug        = _state.get("run_slug", "")
    script_path = OUTPUT_ROOT / slug / "script.txt"
    if not script_path.exists():
        await update.message.reply_text("❌ script.txt not found — cannot revise")
        _state["approval_result"] = False
        ae = _state.get("approval_event")
        if ae:
            ae.set()
        return

    await update.message.reply_text("✏️ Revising script…")

    import anthropic
    script = script_path.read_text()
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    prompt = (
        f"{load_claude_md()}\n\n---\nSTYLE SHEET:\n{load_style_sheet()}\n\n---\n"
        "You are revising a YouTube video script based on feedback. Apply the feedback precisely.\n"
        "Keep everything not mentioned in the feedback exactly as-is.\n"
        "Return only the revised script — no preamble, no explanation.\n\n"
        f"FEEDBACK:\n{feedback}\n\nCURRENT SCRIPT:\n{script}\n\n===SCRIPT===\n"
    )
    try:
        # This blocks the event loop for ~10-20s — acceptable for a personal single-user bot
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        text    = response.content[0].text
        m       = re.search(r"===SCRIPT===(.*)", text, re.DOTALL)
        revised = m.group(1).strip() if m else text.strip()
    except Exception as e:
        await update.message.reply_text(f"❌ Revision error: {e}")
        _state["revision_mode"] = True
        return

    if not revised:
        await update.message.reply_text("⚠️ Revision returned empty — please try again")
        _state["revision_mode"] = True
        return

    script_path.write_text(revised)
    await update.message.reply_text("✅ Script revised — sending for re-review…")
    await _send_script_for_review(context.bot, update.effective_chat.id, revised)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in .env")
    if not ALLOWED_USER_ID:
        raise RuntimeError("TELEGRAM_USER_ID not set in .env")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_start))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("runs",     cmd_runs))
    app.add_handler(CommandHandler("stop",     cmd_stop))
    app.add_handler(CommandHandler("run",      cmd_run))
    app.add_handler(CommandHandler("resume",   cmd_resume))
    app.add_handler(CommandHandler("download", cmd_download))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print(f"✅ Bot running — authorized user ID: {ALLOWED_USER_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
