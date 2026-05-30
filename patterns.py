"""Candlestick pattern detection shared by the Streamlit app and scanners."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def detect_marubozu(
    df: pd.DataFrame,
    wick_max_frac: float = 0.03,
    min_body_frac: float = 0.62,
    min_volume_mult_vs_ma: Optional[float] = 1.2,
    volume_ma_period: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bullish Marubozu: Close > Open, open ≈ low, close ≈ high.
    Bearish Marubozu: Close < Open, open ≈ high, close ≈ low.
    """
    if df.empty or not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    low = df["Low"].astype(float)
    c = df["Close"].astype(float)
    rng = (h - low).replace(0, np.nan)
    body = (c - o).abs()
    body_ok = (body / rng) >= min_body_frac

    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - low
    upper_ok = (upper_wick / rng) <= wick_max_frac
    lower_ok = (lower_wick / rng) <= wick_max_frac

    bullish = (c > o) & body_ok & upper_ok & lower_ok & rng.notna()
    bearish = (c < o) & body_ok & upper_ok & lower_ok & rng.notna()

    has_volume = "Volume" in df.columns
    vol_ma: Optional[pd.Series] = None
    if has_volume:
        vol = df["Volume"].astype(float)
        vol_ma = vol.rolling(volume_ma_period).mean()
        if min_volume_mult_vs_ma is not None:
            vol_ok = vol_ma.notna() & (vol_ma > 0) & (vol >= vol_ma * float(min_volume_mult_vs_ma))
            bullish = bullish & vol_ok
            bearish = bearish & vol_ok

    out_cols = ["Open", "High", "Low", "Close"] + (["Volume"] if has_volume else [])
    bull_df = df.loc[bullish, [col for col in out_cols if col in df.columns]].copy()
    bear_df = df.loc[bearish, [col for col in out_cols if col in df.columns]].copy()
    if has_volume and vol_ma is not None:
        if not bull_df.empty:
            bull_df["Vol_MA20"] = vol_ma.loc[bullish].astype(float)
        if not bear_df.empty:
            bear_df["Vol_MA20"] = vol_ma.loc[bearish].astype(float)

    return bull_df, bear_df


def latest_bullish_confirmation(
    df: pd.DataFrame,
    lookback: int = 3,
) -> tuple[bool, str]:
    """
    True if any bullish confirmation pattern appears in the last `lookback` completed bars.
    Checks Marubozu, hammer, and bullish engulfing on daily OHLCV.
    """
    if df.empty or len(df) < 2:
        return False, ""

    work = df.tail(max(lookback + 1, 5)).copy()
    recent = work.tail(lookback)

    bull_maru, _ = detect_marubozu(recent, min_volume_mult_vs_ma=None)
    if not bull_maru.empty:
        return True, "bullish_marubozu"

    o = work["Open"].astype(float)
    h = work["High"].astype(float)
    low = work["Low"].astype(float)
    c = work["Close"].astype(float)
    rng = (h - low).replace(0, np.nan)
    body = (c - o).abs()
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - low
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)

    hammer = (
        (c > o)
        & (lower_wick >= body * 2.0)
        & (upper_wick <= body * 0.5)
        & ((body / rng) <= 0.35)
        & rng.notna()
    )
    if hammer.loc[recent.index].any():
        return True, "hammer"

    prev_o = o.shift(1)
    prev_c = c.shift(1)
    engulf = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    if engulf.loc[recent.index].any():
        return True, "bullish_engulfing"

    return False, ""


def drop_forming_week(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the latest row when it belongs to the still-forming calendar week."""
    if len(df) < 2:
        return df
    last_ts = pd.Timestamp(df.index[-1])
    now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz is not None else pd.Timestamp.now()
    if last_ts.isocalendar()[:2] == now.isocalendar()[:2]:
        return df.iloc[:-1].copy()
    return df


def load_weekly_ohlcv(yahoo_ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(yahoo_ticker, period=period, interval="1wk", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna().copy()


def marubozu_hits_in_window(
    df: pd.DataFrame,
    lookback_weeks: int,
    *,
    wick_max_frac: float = 0.03,
    min_body_frac: float = 0.62,
    min_volume_mult_vs_ma: Optional[float] = 1.2,
    volume_ma_period: int = 20,
    exclude_forming_week: bool = True,
) -> list[dict[str, Any]]:
    """Return bullish/bearish Marubozu rows whose week falls in the last `lookback_weeks` bars."""
    if df.empty:
        return []

    work = drop_forming_week(df) if exclude_forming_week else df.copy()
    if work.empty:
        return []

    history_weeks = max(lookback_weeks, volume_ma_period + 2)
    scan_df = work.tail(history_weeks)
    recent_index = set(work.tail(lookback_weeks).index)

    bull_df, bear_df = detect_marubozu(
        scan_df,
        wick_max_frac=wick_max_frac,
        min_body_frac=min_body_frac,
        min_volume_mult_vs_ma=min_volume_mult_vs_ma,
        volume_ma_period=volume_ma_period,
    )

    hits: list[dict[str, Any]] = []
    for pattern, frame in (("bullish", bull_df), ("bearish", bear_df)):
        for ts, row in frame.iterrows():
            if ts not in recent_index:
                continue
            rng = float(row["High"] - row["Low"])
            body_pct = float(abs(row["Close"] - row["Open"]) / rng * 100.0) if rng > 0 else 0.0
            hit: dict[str, Any] = {
                "week_ending": ts,
                "pattern": pattern,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "body_pct_of_range": round(body_pct, 2),
            }
            if "Volume" in row.index:
                hit["volume"] = float(row["Volume"])
            if "Vol_MA20" in row.index:
                hit["vol_ma20"] = float(row["Vol_MA20"])
            hits.append(hit)
    return hits
