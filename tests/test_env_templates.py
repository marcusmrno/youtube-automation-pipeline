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
