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

`/app/scripts/run-streamlit.sh` is **Streamlit’s own container script** (not in this repo). A
`Segmentation fault` there usually means bad Python / native wheels (`yfinance`+`curl_cffi`,
`pyarrow`) — not a missing file in git.

**Deploy checklist**

1. App root = this `eod-swing` folder (Main file: `eod_swing_app.py`)
2. In Cloud → **⋮ → Settings → General → Advanced** → set **Python version = 3.11**
3. Commit these files so Cloud rebuilds with pinned deps:
   - `requirements.txt` (pinned)
   - `runtime.txt` / `.python-version`
   - `.streamlit/config.toml` (`fileWatcherType = "none"`)
   - `eod_swing_lib.py` (Yahoo via plain `requests`, no `curl_cffi`)
4. **Reboot app** (or delete + redeploy) so the old venv is wiped

Do **not** use unpinned `streamlit` / `yfinance` on Cloud — that pulls 3.13 + curl_cffi and segfaults.
