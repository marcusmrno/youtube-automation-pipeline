import re
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).parent.parent


def _blanked(text):
    # what a user who skips an optional key ends up with: KEY= plus whatever follows
    return re.sub(r"(?m)^(\w+)=(?:sk-ant-)?\.\.\.", r"\1=", text)


def test_env_templates_never_turn_comments_into_values(tmp_path):
    readme_block = re.search(r"```env\n(.*?)```", (ROOT / "README.md").read_text(), re.S)[1]
    for name, text in [(".env.example", (ROOT / ".env.example").read_text()),
                       ("README env block", _blanked(readme_block))]:
        f = tmp_path / "env"
        f.write_text(text)
        commented = {k: v for k, v in dotenv_values(f).items() if (v or "").startswith("#")}
        assert not commented, (name, commented)   # a comment value is truthy: VIDIQ_KEY would turn on


def test_bot_imports_with_blank_or_commented_user_id():
    # a blank or comment-valued TELEGRAM_USER_ID must reach main()'s "not set" message, not crash import
    import os, subprocess, sys
    for value in ("", "# your numeric Telegram user ID"):
        env = {**os.environ, "TELEGRAM_USER_ID": value, "TELEGRAM_BOT_TOKEN": "",
               "ANTHROPIC_API_KEY": "", "VIDIQ_API_KEY": ""}
        r = subprocess.run([sys.executable, "-c", "import bot; print(bot.ALLOWED_USER_ID)"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip() == "0", (value, r.stderr[-200:])
