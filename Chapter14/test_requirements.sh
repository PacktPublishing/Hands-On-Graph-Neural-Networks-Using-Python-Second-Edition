#!/bin/bash
set -e
deactivate 2>/dev/null || true
python3.11 -m venv ~/venvs/ch14-test
source ~/venvs/ch14-test/bin/activate
pip install -r requirements.txt
python check_env.py
