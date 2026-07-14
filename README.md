# EOD Swing Scanner

NIFTY 50 + NIFTY 100 end-of-day swing scanner with Streamlit UI, pivot levels, buy/sell zones, and optional Telegram alerts.

Shared data helpers live in **`eod_swing_lib.py`** (single self-contained module) so Streamlit Cloud deploy works without extra files or the parent `test/` tree.

## Setup

```bash
cd /Users/admin/Desktop/Codebase/ri/test
python3 -m venv .venv
source .venv/bin/activate
pip install -r eod-swing/requirements.txt
```

## Streamlit app

```bash
cd eod-swing
streamlit run eod_swing_app.py
```

## CLI scanner

```bash
cd eod-swing
python eod_swing_scanner.py
python eod_swing_scanner.py -o eod_swing_hits.csv
python eod_swing_scanner.py --nifty50-only
```

## Telegram (optional)

1. Create a bot with [@BotFather](https://t.me/BotFather).
2. Copy `.env.example` → `.env` and set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

```bash
cd eod-swing
python eod_swing_telegram.py --dry-run
python eod_swing_telegram.py
```

## Layout

| Path | Purpose |
|------|---------|
| `eod_swing_app.py` | Streamlit UI (scan, pivots, daily chart) |
| `eod_swing_scanner.py` | Core scan logic + CLI |
| `eod_swing_telegram.py` | Scan → Telegram |
| `eod_swing_lib.py` | Yahoo download, EMA/RSI, S/R, NIFTY symbols, patterns (self-contained) |
| `telegram_notify.py` | Telegram Bot API helper |

## Streamlit Cloud

Deploy this folder as the app root (or set **Main file path** to `eod_swing_app.py`).

**Requirements file:** `requirements.txt` (in this directory)

Ensure the repo includes **`eod_swing_lib.py`** alongside `eod_swing_app.py` and `eod_swing_scanner.py`.
