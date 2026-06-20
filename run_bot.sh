#!/bin/zsh
cd "$(dirname "$0")"
PYTHON=/Users/marcusmoreno/miniforge3/bin/python3.12
$PYTHON -m pip install -q "python-telegram-bot>=20.0"
$PYTHON bot.py
