#!/usr/bin/env python3
"""
EOD swing scanner for NSE stocks (NIFTY 50 + NIFTY 100 universe).

Hard filters (all required):
  1. Close > 20 EMA and 20 EMA > 50 EMA
  2. Session volume > 20-day average volume
  3. RSI(14) > 55

Context flags (reported, not required):
  - Near 20 EMA (within tolerance)
  - Near previous support
  - Breakout above resistance

Candlestick detection on latest completed daily bar:
  - Bullish Engulfing, Hammer, Marubozu

Usage:
  cd eod-swing && python eod_swing_scanner.py
  python eod_swing_scanner.py -o eod_swing_hits.csv
  python eod_swing_scanner.py --nifty50-only
  python eod_swing_scanner.py --min-rsi 58 --near-ema-pct 1.5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from eod_swing_lib import (
    LiveQuote,
    compute_ema,
    compute_rsi,
    detect_marubozu,
    download_daily_batch,
    download_daily_single,
    fetch_live_quotes_batch,
    get_nifty50_and_100_universe,
    get_nifty50_symbols,
    infer_support_resistance,
    merge_realtime_session,
    session_bar_indices,
    to_yahoo_nse,
)


@dataclass
class ScannerConfig:
    period: str = "1y"
    batch_size: int = 12
    download_delay: float = 0.12
    prefer_live_symbols: bool = True
    min_rsi: float = 55.0
    ema_fast: int = 20
    ema_slow: int = 50
    volume_ma_period: int = 20
    rsi_period: int = 14
    min_history_bars: int = 60
    sr_lookback_days: int = 120
    near_ema_pct: float = 2.0
    near_support_pct: float = 3.5
    breakout_buffer_pct: float = 0.15
    nifty50_only: bool = False
    use_realtime: bool = False
    live_quote_workers: int = 10


@dataclass
class ScanHit:
    symbol: str
    universe: str
    close: float
    ema20: float
    ema50: float
    rsi: float
    volume: int
    avg_volume: int
    vol_vs_avg_pct: float
    near_ema: bool
    near_support: bool
    breakout_resistance: bool
    support: float
    resistance: float
    pivot: Optional[float] = None
    s1: Optional[float] = None
    s2: Optional[float] = None
    r1: Optional[float] = None
    r2: Optional[float] = None
    patterns: list[str] = field(default_factory=list)
    as_of: str = ""
    live_ltp: Optional[float] = None
    scan_mode: str = "eod"


def _is_hammer_bar(o: float, h: float, low: float, c: float) -> bool:
    rng = h - low
    if rng <= 0:
        return False
    body = abs(c - o)
    lower_wick = min(o, c) - low
    upper_wick = h - max(o, c)
    return (
        c > o
        and lower_wick >= body * 2.0
        and upper_wick <= body * 0.5
        and (body / rng) <= 0.35
    )


def _is_bullish_engulfing(prev_o: float, prev_c: float, o: float, c: float) -> bool:
    return prev_c < prev_o and c > o and c >= prev_o and o <= prev_c


def detect_patterns_at_idx(df: pd.DataFrame, idx: int) -> list[str]:
    """Bullish patterns on bar `idx` (Hammer / Engulfing / Marubozu)."""
    if idx < 0 or idx >= len(df):
        return []
    patterns: list[str] = []
    row = df.iloc[idx : idx + 1]
    bull_maru, _ = detect_marubozu(row, min_volume_mult_vs_ma=None)
    if not bull_maru.empty:
        patterns.append("marubozu")

    o = float(df["Open"].iloc[idx])
    h = float(df["High"].iloc[idx])
    low = float(df["Low"].iloc[idx])
    c = float(df["Close"].iloc[idx])
    if _is_hammer_bar(o, h, low, c):
        patterns.append("hammer")

    if idx >= 1:
        prev_o = float(df["Open"].iloc[idx - 1])
        prev_c = float(df["Close"].iloc[idx - 1])
        if _is_bullish_engulfing(prev_o, prev_c, o, c):
            patterns.append("bullish_engulfing")

    return patterns


def immediate_floor_pivots(df: pd.DataFrame, idx: int) -> tuple[float, float, float, float, float]:
    """
    Classic floor pivots from the scanned session H/L/C (bar at `idx`).
    These are the immediate S1/S2/R1/R2 levels for the next trading session.
    """
    bar = df.iloc[idx]
    h = float(bar["High"])
    low = float(bar["Low"])
    c = float(bar["Close"])
    pivot = (h + low + c) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - h
    r2 = pivot + (h - low)
    s2 = pivot - (h - low)
    return (
        round(pivot, 2),
        round(s1, 2),
        round(s2, 2),
        round(r1, 2),
        round(r2, 2),
    )


def evaluate_symbol(
    symbol: str,
    yahoo: str,
    df: pd.DataFrame,
    cfg: ScannerConfig,
    universe_label: str,
    live_quote: Optional[LiveQuote] = None,
) -> Optional[ScanHit]:
    if len(df) < cfg.min_history_bars:
        return None

    work = df
    live_ltp: Optional[float] = None
    scan_mode = "eod"

    if cfg.use_realtime and live_quote and live_quote.price > 0:
        work = merge_realtime_session(
            df,
            live_quote.price,
            day_high=live_quote.day_high,
            day_low=live_quote.day_low,
            day_volume=live_quote.volume if live_quote.volume > 0 else None,
        )
        live_ltp = live_quote.price
        scan_mode = "live"

    price_idx, vol_idx, pat_idx = session_bar_indices(
        work,
        cfg.volume_ma_period,
        realtime=cfg.use_realtime,
    )

    close_s = work["Close"].astype(float)
    price = live_ltp if live_ltp is not None else float(close_s.iloc[price_idx])
    ema20 = float(compute_ema(close_s, cfg.ema_fast).iloc[price_idx])
    ema50 = float(compute_ema(close_s, cfg.ema_slow).iloc[price_idx])

    if not (price > ema20 > ema50):
        return None

    rsi = float(compute_rsi(close_s, cfg.rsi_period).iloc[price_idx])
    if rsi <= cfg.min_rsi:
        return None

    vol = work["Volume"].astype(float)
    avg_vol = float(vol.rolling(cfg.volume_ma_period).mean().iloc[vol_idx])
    last_vol = float(vol.iloc[vol_idx])
    if avg_vol <= 0 or last_vol <= avg_vol:
        return None

    support, resistance, _ = infer_support_resistance(work, price, cfg.sr_lookback_days)
    dist_ema_pct = abs(price - ema20) / ema20 * 100.0 if ema20 > 0 else 999.0
    dist_support_pct = (price - support) / price * 100.0 if price > 0 else 999.0

    near_ema = dist_ema_pct <= cfg.near_ema_pct
    near_support = dist_support_pct <= cfg.near_support_pct

    prev_close = float(close_s.iloc[price_idx - 1]) if price_idx > 0 else price
    breakout = price > resistance * (1.0 + cfg.breakout_buffer_pct / 100.0) and prev_close <= resistance

    patterns = detect_patterns_at_idx(work, pat_idx)
    vol_vs_avg = (last_vol - avg_vol) / avg_vol * 100.0 if avg_vol > 0 else 0.0
    bar_date = pd.Timestamp(work.index[price_idx]).strftime("%Y-%m-%d")
    as_of = f"{bar_date} (live)" if live_ltp is not None else bar_date

    pivot, s1, s2, r1, r2 = immediate_floor_pivots(work, price_idx)

    return ScanHit(
        symbol=symbol,
        universe=universe_label,
        close=round(price, 2),
        ema20=round(ema20, 2),
        ema50=round(ema50, 2),
        rsi=round(rsi, 1),
        volume=int(last_vol),
        avg_volume=int(avg_vol),
        vol_vs_avg_pct=round(vol_vs_avg, 1),
        near_ema=near_ema,
        near_support=near_support,
        breakout_resistance=breakout,
        support=round(support, 2),
        resistance=round(resistance, 2),
        pivot=pivot,
        s1=s1,
        s2=s2,
        r1=r1,
        r2=r2,
        patterns=patterns,
        as_of=as_of,
        live_ltp=live_ltp,
        scan_mode=scan_mode,
    )


def _universe_label(symbol: str, n50_set: set[str]) -> str:
    return "NIFTY50" if symbol in n50_set else "NIFTY100"


def reevaluate_hits_from_cache(
    frame_cache: dict[str, pd.DataFrame],
    cfg: ScannerConfig,
    n50_set: set[str],
) -> list[ScanHit]:
    """Re-run filters with fresh live LTP (no re-download of history)."""
    if not frame_cache:
        return []
    quotes = fetch_live_quotes_batch(
        list(frame_cache.keys()),
        max_workers=cfg.live_quote_workers,
    )
    hits: list[ScanHit] = []
    for symbol, frame in frame_cache.items():
        label = _universe_label(symbol, n50_set)
        hit = evaluate_symbol(
            symbol,
            to_yahoo_nse(symbol),
            frame,
            cfg,
            label,
            live_quote=quotes.get(symbol),
        )
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: (len(h.patterns), h.rsi, h.vol_vs_avg_pct), reverse=True)
    return hits


def _scan_universe(
    symbols: list[str],
    cfg: ScannerConfig,
    n50_set: set[str],
) -> tuple[list[ScanHit], list[str], list[str], dict[str, pd.DataFrame]]:
    yahoo_map = {s: to_yahoo_nse(s) for s in symbols}
    yahoo_tickers = [yahoo_map[s] for s in symbols]
    hits: list[ScanHit] = []
    missing: list[str] = []
    errors: list[str] = []
    frame_cache: dict[str, pd.DataFrame] = {}

    total_batches = max(1, (len(yahoo_tickers) + cfg.batch_size - 1) // cfg.batch_size)
    for batch_idx, i in enumerate(range(0, len(yahoo_tickers), cfg.batch_size)):
        batch_yahoo = yahoo_tickers[i : i + cfg.batch_size]
        batch_nse = symbols[i : i + cfg.batch_size]
        print(f"  batch {batch_idx + 1}/{total_batches} — {batch_nse[0]}…", flush=True)

        try:
            frames = download_daily_batch(batch_yahoo, cfg.period)
        except Exception as exc:
            errors.append(f"{batch_nse[0]}: {exc}")
            frames = {}

        live_quotes: dict[str, LiveQuote] = {}
        if cfg.use_realtime:
            live_quotes = fetch_live_quotes_batch(
                batch_nse,
                max_workers=cfg.live_quote_workers,
            )

        for nse_symbol, yahoo_ticker in zip(batch_nse, batch_yahoo):
            frame = frames.get(yahoo_ticker)
            if frame is None or frame.empty:
                frame = download_daily_single(yahoo_ticker, cfg.period)
            if frame is None or frame.empty:
                missing.append(nse_symbol)
                continue
            frame_cache[nse_symbol] = frame.copy()
            label = _universe_label(nse_symbol, n50_set)
            hit = evaluate_symbol(
                nse_symbol,
                yahoo_ticker,
                frame,
                cfg,
                label,
                live_quote=live_quotes.get(nse_symbol),
            )
            if hit is not None:
                hits.append(hit)

        if i + cfg.batch_size < len(yahoo_tickers) and cfg.download_delay > 0:
            time.sleep(cfg.download_delay)

    hits.sort(key=lambda h: (len(h.patterns), h.rsi, h.vol_vs_avg_pct), reverse=True)
    return hits, missing, errors, frame_cache


def run_eod_swing_scan(
    cfg: Optional[ScannerConfig] = None,
) -> tuple[list[ScanHit], str, list[str], list[str], dict[str, pd.DataFrame], set[str]]:
    cfg = cfg or ScannerConfig()

    mode = "Realtime (live LTP)" if cfg.use_realtime else "EOD (last completed session)"
    print("EOD swing scanner")
    print(f"  Mode:   {mode}")
    print(f"  Trend:  Close > {cfg.ema_fast} EMA > {cfg.ema_slow} EMA")
    print(f"  Volume: > {cfg.volume_ma_period}-day average")
    print(f"  RSI:    > {cfg.min_rsi}")
    if cfg.use_realtime:
        print("  Note:   Live LTP for trend/RSI/pivots; volume uses last completed session until today fills in")
    print()

    if cfg.nifty50_only:
        symbols = get_nifty50_symbols(prefer_live=cfg.prefer_live_symbols)
        n50_set = set(symbols)
        scanned_label = "NIFTY 50 only"
    else:
        symbols, n50_set = get_nifty50_and_100_universe(prefer_live=cfg.prefer_live_symbols)
        n50_only = len(n50_set)
        n100_only = len(symbols) - n50_only
        scanned_label = f"NIFTY 50 + NIFTY 100 ({len(symbols)} symbols)"
        print(f"  Universe: {len(symbols)} symbols ({n50_only} NIFTY 50 + {n100_only} NIFTY 100-only)")
        print()

    hits, missing, errors, frame_cache = _scan_universe(symbols, cfg, n50_set)

    if errors:
        print(f"  Download errors: {len(errors)} (first: {errors[0]})")
    if missing:
        print(f"  Missing data: {len(missing)} symbols")

    return hits, scanned_label, missing, errors, frame_cache, n50_set


@dataclass
class SwingEntrySuggestion:
    suggested_entry: float
    entry_low: float
    entry_high: float
    entry_style: str
    note: str


def suggest_swing_entry(hit: ScanHit) -> SwingEntrySuggestion:
    """
    Next-session swing long entry levels (aligned with jindalstel_trade_plan logic).

    Dip setup: buy zone around S1 / swing support / pivot / 20 EMA.
    Breakout setup: enter on hold above resistance when breakout flag is set.
    """
    price = hit.close
    s1 = hit.s1 or 0.0
    pivot = hit.pivot or 0.0
    support = hit.support
    ema20 = hit.ema20
    resistance = hit.resistance

    entry_primary = round(max(s1, support, pivot * 0.998), 2)
    entry_secondary = round(min(ema20, (s1 + pivot) / 2), 2)
    if entry_secondary > price:
        entry_secondary = round(ema20 * 0.995, 2)
    entry_breakout = round(resistance * 1.0015, 2)

    if hit.breakout_resistance:
        return SwingEntrySuggestion(
            suggested_entry=entry_breakout,
            entry_low=round(resistance, 2),
            entry_high=entry_breakout,
            entry_style="breakout",
            note="Breakout — enter on hold above resistance with volume",
        )

    entry_low = round(min(entry_primary, entry_secondary), 2)
    entry_high = round(max(entry_primary, entry_secondary), 2)
    suggested = round((entry_primary + entry_secondary) / 2, 2)
    if suggested > price:
        suggested = round(min(entry_primary, ema20 * 0.998), 2)
        entry_high = min(entry_high, price)

    return SwingEntrySuggestion(
        suggested_entry=suggested,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_style="dip",
        note=f"Dip buy zone ₹{entry_low:,.2f}–₹{entry_high:,.2f} (S1 / pivot / 20 EMA)",
    )


def hits_to_dataframe(hits: list[ScanHit]) -> pd.DataFrame:
    if not hits:
        return pd.DataFrame()
    rows = []
    for h in hits:
        entry = suggest_swing_entry(h)
        rows.append(
            {
                "symbol": h.symbol,
                "universe": h.universe,
                "as_of": h.as_of,
                "close": h.close,
                "ema20": h.ema20,
                "ema50": h.ema50,
                "rsi": h.rsi,
                "vol_vs_avg_pct": h.vol_vs_avg_pct,
                "near_ema": h.near_ema,
                "near_support": h.near_support,
                "breakout_resistance": h.breakout_resistance,
                "support": h.support,
                "resistance": h.resistance,
                "pivot": h.pivot,
                "s1": h.s1,
                "s2": h.s2,
                "r1": h.r1,
                "r2": h.r2,
                "suggested_entry": entry.suggested_entry,
                "entry_low": entry.entry_low,
                "entry_high": entry.entry_high,
                "entry_style": entry.entry_style,
                "entry_note": entry.note,
                "patterns": ", ".join(h.patterns) if h.patterns else "—",
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EOD swing scanner (NIFTY 50 + NIFTY 100)")
    p.add_argument("--min-rsi", type=float, default=55.0)
    p.add_argument("--near-ema-pct", type=float, default=2.0, help="Within %% of 20 EMA")
    p.add_argument("--near-support-pct", type=float, default=3.5)
    p.add_argument("--period", default="1y")
    p.add_argument("--nifty50-only", action="store_true", help="Scan only NIFTY 50 (skip NIFTY 100-only names)")
    p.add_argument("--static-symbols", action="store_true", help="Skip Wikipedia symbol fetch")
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Use live LTP on today's session bar (default: last completed EOD bar)",
    )
    p.add_argument("-o", "--output", help="Write CSV path")
    p.add_argument("--delay", type=float, default=0.12)
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
        use_realtime=args.realtime,
    )

    hits, label, missing, errors, _frame_cache, _n50 = run_eod_swing_scan(cfg)
    df = hits_to_dataframe(hits)

    print()
    print(f"Scanned: {label}")
    print(f"Matches: {len(hits)}")
    if df.empty:
        print("\nNo stocks passed all three core filters.")
        return 0

    display = df.rename(
        columns={
            "symbol": "Symbol",
            "universe": "Universe",
            "as_of": "As Of",
            "close": "Close",
            "ema20": "EMA20",
            "ema50": "EMA50",
            "rsi": "RSI",
            "vol_vs_avg_pct": "Vol vs Avg %",
            "near_ema": "Near EMA",
            "near_support": "Near Support",
            "breakout_resistance": "Breakout",
            "support": "Support",
            "resistance": "Resistance",
            "pivot": "Pivot",
            "s1": "S1",
            "s2": "S2",
            "r1": "R1",
            "r2": "R2",
            "suggested_entry": "Suggested Entry",
            "patterns": "Patterns",
        }
    )
    cols = [
        "Symbol",
        "Universe",
        "Close",
        "Suggested Entry",
        "S1",
        "S2",
        "R1",
        "R2",
        "RSI",
        "Vol vs Avg %",
        "Near EMA",
        "Near Support",
        "Breakout",
        "Patterns",
        "As Of",
    ]
    print("  Pivot S1/S2/R1/R2: floor pivots from scanned session H/L/C (levels for the next session)")
    print()
    print(display[cols].to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
