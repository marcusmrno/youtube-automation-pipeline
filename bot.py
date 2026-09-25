"""
YouTube Pipeline — Telegram Bot Interface
Run: python bot.py
"""

import asyncio
import functools
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

import anthropic
import metadata as _metadata_mod
from pipeline import (
    OUTPUT_ROOT,
    ANTHROPIC_KEY,
    run_pipeline,
    resume_pipeline,
    run_from_script,
    _topic_from_script,
    revise_script,
    run_status,
    slugify,
    generate_clarifying_questions,
    generate_approach_pitches,
)
from profile import load_profile, list_profiles

load_dotenv()

BOT_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_user_id        = os.getenv("TELEGRAM_USER_ID", "").strip()
ALLOWED_USER_ID = int(_user_id) if _user_id.isdigit() else 0   # blank or "# comment" -> main() says "not set"

_state: dict = {
    "running":           False,
    "run_slug":          None,
    "stop_event":        None,
    "approval_event":    None,
    "approval_result":   None,
    "revision_mode":     False,
    "script_mode":       False,   # waiting for a premade script (pasted or uploaded)
    "clarifying_mode":   False,   # waiting for answers to clarifying questions
    "clarifying_topic":  None,    # topic being planned
    "clarifying_questions": None, # parsed list of {question, default} dicts
    "approach_pitches":  None,    # raw pitches text for context storage
    "approach_answers":  None,    # user's answers to clarifying questions
    "log_queue":         None,    # identity guard so a stale worker can't clear a newer run's "running" flag
    "total_images":      0,
    "done_images":       0,
    "milestone_sent":    set(),
    "profile_name":      None,   # None = use first available
    "meta_slug":         None,   # run whose thumbnail picker was sent last
}


# ── Auth ───────────────────────────────────────────────────────────────────────

def auth(func):
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.effective_user or update.effective_user.id != ALLOWED_USER_ID:
            return
        return await func(update, context)
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
    crossed = [m for m in (25, 50, 75, 100) if pct >= m and m not in _state["milestone_sent"]]
    if not crossed:
        return None
    _state["milestone_sent"].update(crossed)
    return max(crossed)


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

        def _on_send_done(fut):
            # e.g. Markdown parse error on the script text — don't hang the worker forever
            if fut.exception() is not None:
                _state["approval_result"] = False
                evt.set()

        future = asyncio.run_coroutine_threadsafe(
            _send_script_for_review(bot, chat_id, script_text),
            loop,
        )
        future.add_done_callback(_on_send_done)
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

    if _state.get("log_queue") is lq:
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


# ── Clarifying questions + approach pitches ────────────────────────────────────

def _parse_clarifying_questions(text: str) -> list[dict]:
    items = []
    for block in re.split(r'\n(?=\d+\.)', text.strip()):
        lines = block.strip().split('\n')
        question = re.sub(r'^\d+\.\s*', '', lines[0]).strip()
        default_line = next((l for l in lines if l.strip().lower().startswith('default:')), '')
        default = re.sub(r'(?i)^\s*default:\s*', '', default_line).strip()
        if question:
            items.append({'question': question, 'default': default})
    return items


async def _send_clarifying_questions(update: Update, context: ContextTypes.DEFAULT_TYPE, topic: str) -> None:
    chat_id = update.effective_chat.id
    try:
        profile, _ = _resolve_profile_for_bot()
    except RuntimeError as e:
        await update.message.reply_text(f"❌ {e}")
        return
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    await update.message.reply_text(f"🤔 Thinking about your topic: *{topic}*…", parse_mode="Markdown")

    loop = asyncio.get_running_loop()
    def _generate():
        return generate_clarifying_questions(topic, profile, client, lambda _: None)

    try:
        questions_text = await loop.run_in_executor(None, _generate)
    except Exception as e:
        await update.message.reply_text(f"❌ Error generating questions: {e}")
        return

    parsed = _parse_clarifying_questions(questions_text)
    _state["script_mode"]          = False   # these two both eat the next text message
    _state["clarifying_mode"]      = True
    _state["clarifying_topic"]     = topic
    _state["clarifying_questions"] = parsed

    lines = ["📋 *A few quick questions before we write the script:*\n"]
    for i, item in enumerate(parsed, 1):
        lines.append(f"*{i}. {item['question']}*")
        if item['default']:
            lines.append(f"   _Default: {item['default']}_")
        lines.append("")

    lines.append("Reply with your answers (numbered), or send `default` to use all suggested answers.")

    await context.bot.send_message(chat_id=chat_id, text='\n'.join(lines), parse_mode="Markdown")


async def _send_approach_pitches(update: Update, context: ContextTypes.DEFAULT_TYPE, answers: str) -> None:
    chat_id = update.effective_chat.id
    topic   = _state["clarifying_topic"]

    try:
        profile, _ = _resolve_profile_for_bot()
    except RuntimeError as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ {e}")
        return
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    await context.bot.send_message(chat_id=chat_id, text="💡 Pitching approaches…")

    loop = asyncio.get_running_loop()
    def _generate():
        return generate_approach_pitches(topic, answers, profile, client, lambda _: None)

    try:
        pitches_text = await loop.run_in_executor(None, _generate)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error generating pitches: {e}")
        return

    _state["approach_pitches"] = pitches_text
    _state["approach_answers"] = answers

    # Trim to Telegram message limit
    display = pitches_text[:3800] + ("…" if len(pitches_text) > 3800 else "")
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Approach 1", callback_data="approach:1"),
         InlineKeyboardButton("✅ Approach 2", callback_data="approach:2"),
         InlineKeyboardButton("✅ Approach 3", callback_data="approach:3")],
        [InlineKeyboardButton("❌ Cancel", callback_data="approach:cancel")],
    ])
    await context.bot.send_message(
        chat_id=chat_id,
        text=display,
        reply_markup=keyboard,
    )


# ── Command handlers ───────────────────────────────────────────────────────────

@auth
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎬 *YT Pipeline Bot*\n\n"
        "/run <topic> — plan & start a pipeline run\n"
        "   ↳ asks clarifying questions, pitches 3 approaches, then runs\n"
        "   ↳ reply `default` to auto-fill all suggested answers\n"
        "/script — produce from a script you already wrote\n"
        "   ↳ then paste it, or upload it as a .txt file\n"
        "/runs — list recent runs and their status\n"
        "/resume [slug] — resume an incomplete run\n"
        "/download [slug] — download run assets as zip (latest if omitted)\n"
        "/profile — view or switch the active style profile\n"
        "/status — show current run status\n"
        "/stop — stop the current run",
        parse_mode="Markdown",
    )


@auth
async def cmd_profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    available = list_profiles()
    if not available:
        await update.message.reply_text("❌ No profiles found. Create profiles/<name>/profile.yaml first.")
        return

    # If arg given, set directly
    if context.args:
        name = context.args[0].strip()
        if name not in available:
            await update.message.reply_text(
                f"❌ Profile `{name}` not found.\n\nAvailable: {', '.join(f'`{p}`' for p in available)}",
                parse_mode="Markdown",
            )
            return
        _state["profile_name"] = name
        await update.message.reply_text(f"✅ Profile set to `{name}`", parse_mode="Markdown")
        return

    # Otherwise show current + inline buttons to switch
    current = _state["profile_name"] or available[0]
    buttons = [
        [InlineKeyboardButton(
            f"{'✅ ' if p == current else ''}{p}",
            callback_data=f"profile:{p}",
        )]
        for p in available
    ]
    await update.message.reply_text(
        f"Active profile: `{current}`\n\nSelect a profile:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(buttons),
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

    dirs = _run_dirs()
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
    dirs = _run_dirs()
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


def _run_dirs() -> list[str]:
    """Run slugs, most recently modified first."""
    if not OUTPUT_ROOT.exists():
        return []
    dirs = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def _resolve_run_slug(arg: str | None) -> str | None:
    """A run folder directly under output/ (the newest when arg is empty); None for anything else."""
    if arg and arg != "regenerate":
        arg = arg.strip()
        # '..' or an absolute path would zip (and upload) anything on disk, .env included
        inside = (OUTPUT_ROOT / arg).resolve().parent == OUTPUT_ROOT.resolve()
        return arg if arg and inside else None
    dirs = _run_dirs()
    return dirs[0] if dirs else None


def _resolve_profile_for_bot():
    name = _state.get("profile_name")
    available = list_profiles()
    if not (name and name in available):
        name = available[0] if available else None
    if not name:
        raise RuntimeError("No profiles available")
    return load_profile(name), name


async def _send_metadata_view(chat, data, run_slug):
    titles = data.get("titles") or []
    chosen_th = data.get("chosen_thumbnail_index", 0)

    if titles:
        lines = [
            f"[{t.get('vidiq_score') if t.get('vidiq_score') is not None else '—'}] {t['text']}"
            for t in titles
        ]
        await chat.send_message("*Titles (vidIQ score in brackets):*\n" + "\n".join(lines), parse_mode="Markdown")
    else:
        await chat.send_message("*Titles:* (none)", parse_mode="Markdown")

    desc = data.get("description") or ""
    if desc:
        await chat.send_message(f"```\n{desc}\n```", parse_mode="Markdown")

    tags = " ".join(data.get("hashtags") or [])
    if tags:
        await chat.send_message(tags)

    thumbs = data.get("thumbnails") or []
    media: list[InputMediaPhoto] = []
    handles = []
    for t in thumbs:
        if "render_error" in t:
            continue
        path = OUTPUT_ROOT / run_slug / t["filename"]
        if path.exists():
            fh = open(path, "rb")
            handles.append(fh)
            media.append(InputMediaPhoto(media=fh, caption=t.get("hook_text", "")))
    if media:
        try:
            await chat.send_media_group(media=media)
        finally:
            for fh in handles:
                fh.close()

    # Telegram caps callback_data at 64 bytes and slugs run to 60, so the slug stays here.
    # ponytail: the newest picker wins; map ids to slugs if older pickers must stay live
    _state["meta_slug"] = run_slug
    thumb_buttons = [
        InlineKeyboardButton(
            f"{'★' if i == chosen_th else ''}Thumb {i+1}",
            callback_data=f"meta_thumb:{i}",
        )
        for i, t in enumerate(thumbs) if "render_error" not in t
    ]
    if thumb_buttons:
        await chat.send_message(
            "Pick a thumbnail:",
            reply_markup=InlineKeyboardMarkup([thumb_buttons]),
        )


@auth
async def cmd_metadata(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    regenerate = False
    slug_arg = None
    for a in args:
        if a == "regenerate":
            regenerate = True
        else:
            slug_arg = a

    slug = _resolve_run_slug(slug_arg)
    if not slug:
        await update.message.reply_text("Usage: /metadata [<run-slug>] [regenerate]")
        return

    existing = _metadata_mod.load_metadata(slug)
    if existing and not regenerate:
        await update.message.reply_text(f"📦 Loading existing metadata for `{slug}`", parse_mode="Markdown")
        await _send_metadata_view(update.effective_chat, existing, slug)
        return

    await update.message.reply_text(f"📦 Generating metadata for `{slug}`…", parse_mode="Markdown")
    try:
        profile, _ = _resolve_profile_for_bot()
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")
        return

    log_queue: queue.Queue = queue.Queue()

    def runner():
        try:
            _metadata_mod.generate_metadata(slug, profile, log_fn=lambda m: log_queue.put(m), regenerate=True)
            log_queue.put("__DONE__")
        except Exception as e:
            log_queue.put(f"__ERROR__:{e}")

    threading.Thread(target=runner, daemon=True).start()

    # Forward filtered progress lines
    while True:
        try:
            msg = await asyncio.get_running_loop().run_in_executor(None, log_queue.get, True, 60)
        except queue.Empty:
            await update.message.reply_text("⏳ still working…")
            continue
        if msg == "__DONE__":
            break
        if msg.startswith("__ERROR__:"):
            await update.message.reply_text("❌ " + msg[len("__ERROR__:"):])
            return
        if _should_relay(msg):
            await update.message.reply_text(msg)

    data = _metadata_mod.load_metadata(slug)
    if data:
        await _send_metadata_view(update.effective_chat, data, slug)


@auth
async def cmd_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Resolve slug: explicit arg, or most recent run
    slug = _resolve_run_slug(context.args[0].strip() if context.args else None)

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
    # disarm every text mode: otherwise the next message still triggers a paid revise or pitch call
    _state.update(running=False, script_mode=False, revision_mode=False,
                  clarifying_mode=False, clarifying_topic=None)
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
    await _send_clarifying_questions(update, context, topic=topic)


@auth
async def cmd_script(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _state["running"]:
        await update.message.reply_text("⚠️ A pipeline is already running. Use /stop first.")
        return
    _state["script_mode"] = True
    await update.message.reply_text(
        "📄 Send the script now — paste it, or upload it as a .txt file.\n\n"
        "No research, writing, or approval step: it goes straight to TTS, image prompts, "
        "images, and voiceover. The run is named after the script's `TITLE:` line, and a run "
        "folder of that name is overwritten.",
        parse_mode="Markdown",
    )


async def _run_premade_script(update, context: ContextTypes.DEFAULT_TYPE, script: str) -> None:
    """Consume a pasted or uploaded script and start production."""
    if not script.strip():
        await update.message.reply_text("⚠️ That script is empty — send another, or /stop to cancel.")
        return
    _state["script_mode"] = False
    await _start_pipeline(update, context, script=script)


@auth
async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A .txt upload is only meaningful while /script is waiting for one."""
    if not _state.get("script_mode"):
        await update.message.reply_text("Use /script first if that's a script to produce from.")
        return

    doc = update.message.document
    try:
        raw = await (await doc.get_file()).download_as_bytearray()
        script = bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        await update.message.reply_text(f"❌ `{doc.file_name}` isn't UTF-8 text — send a .txt file.",
                                        parse_mode="Markdown")
        return
    except Exception as e:
        await update.message.reply_text(f"❌ Could not read that file: {e}")
        return

    await _run_premade_script(update, context, script)


@auth
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _state["running"]:
        await update.message.reply_text("⚠️ A pipeline is already running. Use /stop first.")
        return

    if not context.args:
        if not OUTPUT_ROOT.exists():
            await update.message.reply_text("No runs found.")
            return
        dirs = _run_dirs()
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

    slug = _resolve_run_slug(context.args[0])
    if not slug:
        await update.message.reply_text("❌ That isn't a run folder. Use /resume to list incomplete runs.")
        return
    await _start_pipeline(update, context, run_slug=slug)


# ── Pipeline runner ────────────────────────────────────────────────────────────

async def _start_pipeline(
    update,
    context: ContextTypes.DEFAULT_TYPE,
    topic: str | None = None,
    run_slug: str | None = None,
    approach_context: str = "",
    script: str | None = None,
) -> None:
    chat_id = update.effective_chat.id
    app     = context.application
    loop    = asyncio.get_running_loop()

    if script and not topic:
        topic = _topic_from_script(script)

    lq = queue.Queue()
    se = threading.Event()

    available = list_profiles()
    if not available:
        await context.bot.send_message(chat_id, "❌ No profiles found. Create profiles/<name>/profile.yaml first.")
        return
    # Resume uses the run's saved profile; everything else the selected (or first) one
    saved_profile_file = OUTPUT_ROOT / run_slug / "profile.txt" if (run_slug and not (script or topic)) else None
    if saved_profile_file and saved_profile_file.exists():
        profile_name = saved_profile_file.read_text().strip()
    else:
        profile_name = _state["profile_name"] or available[0]
    try:
        profile = load_profile(profile_name)
    except Exception as e:   # e.g. profile.txt names a renamed profile
        await context.bot.send_message(chat_id, f"❌ {e}")
        return

    # One guard for every caller (the Approach buttons bypassed the per-command checks).
    # running is set only after the last step that can fail, with no await in between,
    # so two taps can't both pass and a failed start can't leave the bot "running".
    if _state["running"]:
        await context.bot.send_message(chat_id, "⚠️ A pipeline is already running. Use /stop first.")
        return

    _state.update({
        "running":         True,
        "run_slug":        run_slug or slugify(topic or ""),
        "stop_event":      se,
        "approval_event":  None,
        "approval_result": None,
        "revision_mode":   False,
        "script_mode":     False,
        "clarifying_mode": False,   # a started run ends any planning session left open
        "clarifying_topic": None,
        "log_queue":       lq,
        "total_images":    0,
        "done_images":     0,
        "milestone_sent":  set(),
    })

    def progress_cb(msg: str):
        lq.put({"type": "log", "msg": msg})

    if script:
        label = f"▶️ Producing from script: *{topic}*\nProfile: `{profile_name}`"
        fn    = lambda: run_from_script(
            script,
            profile,
            topic,
            progress_callback=progress_cb,
            stop_event=se,
        )
    elif topic:
        approval_cb = _make_approval_callback(app.bot, chat_id, loop)
        label = f"▶️ Starting pipeline: *{topic}*\nProfile: `{profile_name}`"
        fn    = lambda: run_pipeline(
            topic,
            profile,
            progress_callback=progress_cb,
            stop_event=se,
            approval_callback=approval_cb,
            approach_context=approach_context,
        )
    else:
        label = f"▶️ Resuming: *{run_slug}*\nProfile: `{profile_name}`"
        fn    = lambda: resume_pipeline(
            run_slug,
            profile,
            progress_callback=progress_cb,
            stop_event=se,
        )

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
    await context.bot.send_message(chat_id, label, parse_mode="Markdown")


# ── Inline button callback ─────────────────────────────────────────────────────

@auth
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data
    ae   = _state.get("approval_event")

    # meta_thumb answers with a toast message — skip the generic blank answer
    if not data.startswith("meta_thumb:"):
        await query.answer()

    if data.startswith("approach:"):
        choice = data.split(":", 1)[1]
        await query.edit_message_reply_markup(None)
        if choice == "cancel":
            _state["clarifying_topic"]     = None
            _state["approach_answers"]     = None
            _state["approach_pitches"]     = None
            await query.message.reply_text("❌ Pipeline planning cancelled.")
            return
        answers = _state.get("approach_answers") or ""
        pitches = _state.get("approach_pitches") or ""
        topic   = _state["clarifying_topic"]
        if not topic:   # buttons from a cancelled plan or from before a bot restart
            await query.message.reply_text("That plan expired — send /run again.")
            return
        approach_context = f"{answers}\n\nApproach pitches:\n{pitches}\n\nUser chose Approach {choice}."
        _state["clarifying_topic"]  = None
        _state["approach_answers"]  = None
        _state["approach_pitches"]  = None
        await query.message.reply_text(f"✅ Approach {choice} selected — starting pipeline…")
        await _start_pipeline(update, context, topic=topic, approach_context=approach_context)
        return

    if data.startswith("profile:"):
        name = data.split(":", 1)[1]
        available = list_profiles()
        if name in available:
            _state["profile_name"] = name
            await query.edit_message_text(f"✅ Profile set to `{name}`", parse_mode="Markdown")
        else:
            await query.edit_message_text(f"❌ Profile `{name}` no longer exists.", parse_mode="Markdown")
        return

    if data.startswith("meta_thumb:"):
        idx, slug = data.split(":", 1)[1], _state.get("meta_slug")
        if not slug:
            await query.answer("That picker expired — send /metadata again", show_alert=True)
            return
        try:
            updated = _metadata_mod.pick_thumbnail(slug, int(idx))
            await query.answer("Thumbnail set")
            await _send_metadata_view(query.message.chat, updated, slug)
        except ValueError as e:
            await query.answer(f"Error: {e}", show_alert=True)
        return

    if data in ("approve", "reject", "revise") and not (ae and not ae.is_set()):
        # a review left over from a stopped or finished run
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("Nothing is waiting for approval.")
        return

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
    # A pasted premade script (Telegram caps these at 4096 chars — longer ones come as a file)
    if _state.get("script_mode"):
        await _run_premade_script(update, context, update.message.text)
        return

    # Handle clarifying question answers
    if _state.get("clarifying_mode"):
        text = update.message.text.strip()
        parsed = _state.get("clarifying_questions") or []

        if text.lower() == "default":
            # Use all defaults
            parts = []
            for i, item in enumerate(parsed, 1):
                parts.append(f"{i}. {item['question']}\n   Answer: {item['default'] or '(no default)'}")
            answers = '\n\n'.join(parts)
        else:
            # Pair user's numbered answers back to questions
            raw_answers = re.split(r'\n(?=\d+[\.\)])', text)
            parts = []
            for i, item in enumerate(parsed, 1):
                user_ans = raw_answers[i - 1].strip() if i <= len(raw_answers) else ''
                user_ans = re.sub(r'^\d+[\.\)]\s*', '', user_ans).strip()
                parts.append(f"{i}. {item['question']}\n   Answer: {user_ans or item['default'] or '(no answer)'}")
            answers = '\n\n'.join(parts)

        _state["clarifying_mode"] = False
        await _send_approach_pitches(update, context, answers)
        return

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

    script = script_path.read_text()
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    _available = list_profiles()
    _profile_name = _state["profile_name"] or (_available[0] if _available else None)
    _profile = load_profile(_profile_name) if _profile_name else None

    try:
        # This blocks the event loop for ~10-20s — acceptable for a personal single-user bot
        revised = revise_script(script, feedback, slug or "video", _profile, client)
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

    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_start))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("runs",     cmd_runs))
    app.add_handler(CommandHandler("stop",     cmd_stop))
    app.add_handler(CommandHandler("run",      cmd_run))
    app.add_handler(CommandHandler("script",   cmd_script))
    app.add_handler(CommandHandler("resume",   cmd_resume))
    app.add_handler(CommandHandler("download", cmd_download))
    app.add_handler(CommandHandler("metadata", cmd_metadata))
    app.add_handler(CommandHandler("profile",  cmd_profile))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    print(f"✅ Bot running — authorized user ID: {ALLOWED_USER_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
