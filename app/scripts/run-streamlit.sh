#!/bin/bash
set -e

# Ensure venv exists and is up to date
if [ ! -d "/home/adminuser/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv /home/adminuser/venv
fi

# Upgrade pip and install dependencies
echo "Installing dependencies..."
/home/adminuser/venv/bin/pip install --upgrade pip setuptools wheel
/home/adminuser/venv/bin/pip install -r /app/requirements.txt

# Run Streamlit
echo "Starting Streamlit app..."
/home/adminuser/venv/bin/streamlit run /app/eod_swing_app.py "$@"
