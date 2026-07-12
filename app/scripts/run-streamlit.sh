#!/bin/bash
# Streamlit Community Cloud / container entrypoint helper.
# Prefer plain streamlit; avoid curl_cffi native crashes during Yahoo downloads.
set -euo pipefail

export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${STREAMLIT_SERVER_FILE_WATCHER_TYPE:-none}"
export YF_USE_CURL_CFFI="${YF_USE_CURL_CFFI:-0}"

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${APP_ROOT}"

if [ -x "/home/adminuser/venv/bin/streamlit" ]; then
  STREAMLIT_BIN="/home/adminuser/venv/bin/streamlit"
elif [ -x "${APP_ROOT}/.venv/bin/streamlit" ]; then
  STREAMLIT_BIN="${APP_ROOT}/.venv/bin/streamlit"
else
  STREAMLIT_BIN="streamlit"
fi

exec "${STREAMLIT_BIN}" run "${APP_ROOT}/eod_swing_app.py" \
  --server.fileWatcherType=none \
  "$@"
