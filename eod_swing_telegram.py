#!/usr/bin/env python3
"""
Run the EOD swing scanner and send results to Telegram for next-day swing ideas.

Requires:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your user/group chat id

Optional `.env` in this directory (see .env.example).

Usage:
  cd eod-swing && python eod_swing_telegram.py
  python eod_swing_telegram.py --dry-run          # print message, do not send
  python eod_swing_telegram.py --nifty50-only
  TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python eod_swing_telegram.py

Schedule (IST, after NSE close ~15:30):
  15 16 * * 1-5 cd /path/to/test/eod-swing && ../.venv/bin/python eod_swing_telegram.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from eod_swing_scanner import ScanHit, ScannerConfig, hits_to_dataframe, run_eod_swing_scan
from telegram_notify import html_escape, send_telegram_message


def _yn(flag: bool) -> str:
    return "✓" if flag else "✗"


def format_hit_block(hit: ScanHit, index: int) -> str:
    patterns = ", ".join(hit.patterns) if hit.patterns else "—"
    return (
        f"<b>{index}. {html_escape(hit.symbol)}</b> "
        f"({html_escape(hit.universe)})\n"
        f"Close <b>{hit.close}</b> | RSI <b>{hit.rsi}</b> | Vol +{hit.vol_vs_avg_pct}%\n"
        f"EMA20 {hit.ema20} | EMA50 {hit.ema50}\n"
        f"<b>Pivots:</b> S1 {hit.s1} | S2 {hit.s2} | R1 {hit.r1} | R2 {hit.r2}\n"
        f"Near EMA {_yn(hit.near_ema)} | Near Sup {_yn(hit.near_support)} | "
        f"Breakout {_yn(hit.breakout_resistance)}\n"
        f"Patterns: {html_escape(patterns)}\n"
        f"SR zone: Sup {hit.support} → Res {hit.resistance}"
    )


def format_swing_scan_telegram(
    hits: list[ScanHit],
    scanned_label: str,
    *,
    missing: Optional[list[str]] = None,
    errors: Optional[list[str]] = None,
) -> str:
    ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    as_of = hits[0].as_of if hits else ist.strftime("%Y-%m-%d")
    header = (
        f"📈 <b>EOD Swing Watchlist</b>\n"
        f"<i>Next session ideas · scanned {html_escape(scanned_label)}</i>\n"
        f"Bar date: <b>{html_escape(as_of)}</b> | "
        f"Sent {ist.strftime('%Y-%m-%d %H:%M')} IST\n"
        f"Filters: Close&gt;20EMA&gt;50EMA · RSI&gt;55 · Vol&gt;avg\n"
        f"Matches: <b>{len(hits)}</b>\n"
    )

    if not hits:
        footer = ""
        if missing:
            footer += f"\n<i>Missing data: {len(missing)} symbols</i>"
        return (
            header
            + "\n⚠️ <b>No stocks passed all core filters today.</b>"
            + footer
        )

    body_parts = [format_hit_block(h, i + 1) for i, h in enumerate(hits)]
    body = "\n\n".join(body_parts)

    footer_lines = [
        "\n<i>Not financial advice. Pivots = floor levels from scanned session H/L/C.</i>",
    ]
    if missing:
        footer_lines.append(f"<i>Missing data: {len(missing)}</i>")
    if errors:
        footer_lines.append(f"<i>Download errors: {len(errors)}</i>")

    return header + "\n" + body + "\n".join(footer_lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EOD swing scan → Telegram alert")
    p.add_argument("--dry-run", action="store_true", help="Print message only; do not send")
    p.add_argument("--min-rsi", type=float, default=55.0)
    p.add_argument("--near-ema-pct", type=float, default=2.0)
    p.add_argument("--near-support-pct", type=float, default=3.5)
    p.add_argument("--period", default="1y")
    p.add_argument("--nifty50-only", action="store_true")
    p.add_argument("--static-symbols", action="store_true")
    p.add_argument("--delay", type=float, default=0.12)
    p.add_argument("--token", help="Telegram bot token (overrides TELEGRAM_BOT_TOKEN)")
    p.add_argument("--chat-id", help="Telegram chat id (overrides TELEGRAM_CHAT_ID)")
    p.add_argument("-o", "--output", help="Also save CSV of hits")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ScannerConfig(
        period=args.period,
        min_rsi=args.min_rsi,
        near_ema_pct=args.near_ema_pct,
        near_support_pct=args.near_support_pct,
        prefer_live_symbols=not args.static_symbols,
        nifty50_only=args.nifty50_only,
        download_delay=args.delay,
    )

    hits, scanned_label, missing, errors = run_eod_swing_scan(cfg)
    message = format_swing_scan_telegram(hits, scanned_label, missing=missing, errors=errors)

    if args.output and hits:
        hits_to_dataframe(hits).to_csv(args.output, index=False)
        print(f"Wrote {args.output}")

    if args.dry_run:
        print(message.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
        print(f"\n[dry-run] Would send {len(message)} chars in Telegram message(s).")
        return 0

    ids = send_telegram_message(message, token=args.token, chat_id=args.chat_id)
    print(f"Sent to Telegram ({len(ids)} message part(s)), {len(hits)} hit(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
