#!/bin/bash

if [ ! -d ".venv" ]; then
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv .venv
    elif [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
        /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
    else
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python baseline.py
python timing_test.py
