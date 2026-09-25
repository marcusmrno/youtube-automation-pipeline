import ast
from pathlib import Path

import ui


def test_ui_binds_localhost_only():
    tree = ast.parse(Path(ui.__file__).read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "run" and getattr(n.func.value, "id", "") == "app")
    host = next(k.value.value for k in run.keywords if k.arg == "host")
    assert host == "127.0.0.1"
