"""
Self-contained helpers for the EOD swing scanner.

Bundled here so Streamlit Cloud / standalone deploy only needs files in this
folder — no separate ``filter_pipeline``, ``nifty50_symbols``, or ``patterns``
modules required at import time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")

# --- NIFTY symbol lists -------------------------------------------------------

NIFTY_50_FALLBACK: list[str] = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HINDALCO",
    "HINDUNILVR",
    "HINDZINC",
    "ICICIBANK",
    "INDIGO",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
]

NSE_SYMBOL_RENAMES: dict[str, str] = {
    "ZOMATO": "ETERNAL",
}

YAHOO_TICKER_ALIASES: dict[str, str] = {
    "TATAMOTORS": "TMPV.NS",
    "ZOMATO": "ETERNAL.NS",
}

NIFTY_100_EXTRA_FALLBACK: list[str] = [
    "ABB",
    "ADANIGREEN",
    "ADANIPOWER",
    "AMBUJACEM",
    "DMART",
    "GAIL",
    "HAL",
    "HAVELLS",
    "ICICIPRULI",
    "INDUSTOWER",
    "IOC",
    "IRFC",
    "JINDALSTEL",
    "LICI",
    "LODHA",
    "NAUKRI",
    "PIDILITIND",
    "PNB",
    "SIEMENS",
    "VEDL",
]


def normalize_nse_symbol(symbol: str) -> str:
    key = symbol.strip().upper()
    return NSE_SYMBOL_RENAMES.get(key, key)


def to_yahoo_nse(symbol: str) -> str:
    raw = symbol.strip().upper()
    key = normalize_nse_symbol(raw)
    if raw in YAHOO_TICKER_ALIASES:
        return YAHOO_TICKER_ALIASES[raw]
    return f"{key}.NS"


def _symbols_from_wikipedia() -> list[str]:
    tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        symbol_col = None
        for key in ("symbol", "ticker", "nse symbol"):
            if key in cols:
                symbol_col = cols[key]
                break
        if symbol_col is None:
            continue
        symbols = (
            table[symbol_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "": None})
            .dropna()
            .tolist()
        )
        symbols = [s for s in symbols if s.isalnum() or "-" in s]
        if len(symbols) >= 45:
            return sorted({normalize_nse_symbol(s) for s in symbols})
    raise ValueError("Could not parse NIFTY 50 symbols from Wikipedia")


def _symbols_from_wikipedia_title(title: str, min_count: int = 45) -> list[str]:
    tables = pd.read_html(f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        symbol_col = None
        for key in ("symbol", "ticker", "nse symbol"):
            if key in cols:
                symbol_col = cols[key]
                break
        if symbol_col is None:
            continue
        symbols = (
            table[symbol_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": None, "": None})
            .dropna()
            .tolist()
        )
        symbols = [s for s in symbols if s.isalnum() or "-" in s]
        if len(symbols) >= min_count:
            return sorted({normalize_nse_symbol(s) for s in symbols})
    raise ValueError(f"Could not parse symbols from Wikipedia: {title}")


def get_nifty50_symbols(prefer_live: bool = True) -> list[str]:
    if prefer_live:
        try:
            return _symbols_from_wikipedia()
        except Exception:
            pass
    return sorted({normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK})


def get_nifty100_symbols(prefer_live: bool = True) -> list[str]:
    if prefer_live:
        try:
            return _symbols_from_wikipedia_title("NIFTY 100", min_count=90)
        except Exception:
            pass
    return sorted(
        {normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK}
        | {normalize_nse_symbol(s) for s in NIFTY_100_EXTRA_FALLBACK}
    )


def get_nifty50_and_100_universe(prefer_live: bool = True) -> tuple[list[str], set[str]]:
    n50 = get_nifty50_symbols(prefer_live=prefer_live)
    n100 = get_nifty100_symbols(prefer_live=prefer_live)
    n50_set = {normalize_nse_symbol(s) for s in n50}
    all_symbols = sorted(n50_set | {normalize_nse_symbol(s) for s in n100})
    return all_symbols, n50_set


# --- Yahoo OHLCV download + indicators ----------------------------------------

def _trim_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.columns = [str(c) for c in frame.columns]
    keep = [c for c in _OHLCV_COLS if c in frame.columns]
    if not keep:
        return pd.DataFrame()
    return frame[keep].dropna(how="all")


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1)) if data.columns.nlevels > 1 else set()
        if ticker in level0:
            return _trim_ohlcv(data[ticker].copy())
        if ticker in level1:
            return _trim_ohlcv(data.xs(ticker, axis=1, level=1).copy())
        return pd.DataFrame()
    if len({str(c) for c in data.columns} & set(_OHLCV_COLS)):
        return _trim_ohlcv(data.copy())
    return pd.DataFrame()


def download_daily_batch(yahoo_tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    if not yahoo_tickers:
        return {}
    raw = yf.download(
        yahoo_tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return out
    if len(yahoo_tickers) == 1:
        frame = _extract_ticker_frame(raw, yahoo_tickers[0])
        if not frame.empty:
            out[yahoo_tickers[0]] = frame
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        return out
    for ticker in yahoo_tickers:
        frame = _extract_ticker_frame(raw, ticker)
        if not frame.empty:
            out[ticker] = frame.dropna()
    return out


def download_daily_single(yahoo_ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(yahoo_ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return _trim_ohlcv(df).dropna()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.astype(float).ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    val = float(atr.iloc[-1]) if len(atr) and not np.isnan(atr.iloc[-1]) else np.nan
    return val


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> tuple[list[float], list[float]]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for i in range(lookback, len(df) - lookback):
        high_window = highs[i - lookback : i + lookback + 1]
        low_window = lows[i - lookback : i + lookback + 1]
        if highs[i] == np.max(high_window):
            swing_highs.append(float(highs[i]))
        if lows[i] == np.min(low_window):
            swing_lows.append(float(lows[i]))
    return swing_highs, swing_lows


def cluster_levels(levels: list[float], tolerance: float) -> list[float]:
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]
    for level in sorted_levels[1:]:
        if abs(level - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [round(float(np.mean(cluster)), 2) for cluster in clusters]


def previous_day_pivots(df: pd.DataFrame) -> list[float]:
    if len(df) < 2:
        return []
    prev = df.iloc[-2]
    pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3.0
    r1 = 2 * pivot - prev["Low"]
    s1 = 2 * pivot - prev["High"]
    r2 = pivot + (prev["High"] - prev["Low"])
    s2 = pivot - (prev["High"] - prev["Low"])
    return [round(x, 2) for x in [pivot, r1, s1, r2, s2]]


def infer_support_resistance(
    df: pd.DataFrame,
    current_price: float,
    lookback_days: int,
) -> tuple[float, float, float]:
    """Return (nearest_support, nearest_resistance, atr)."""
    atr_value = compute_atr(df, 14)
    if np.isnan(atr_value):
        atr_value = current_price * 0.01
    tolerance = max(atr_value * 0.35, current_price * 0.004)

    recent = df.tail(lookback_days)
    swing_highs, swing_lows = find_swing_points(recent, lookback=3)
    pivots = previous_day_pivots(df)

    support_candidates = cluster_levels(swing_lows + [p for p in pivots if p < current_price], tolerance)
    resistance_candidates = cluster_levels(swing_highs + [p for p in pivots if p > current_price], tolerance)

    supports = sorted([x for x in support_candidates if x < current_price], reverse=True)
    resistances = sorted([x for x in resistance_candidates if x > current_price])

    nearest_support = supports[0] if supports else current_price * 0.95
    nearest_resistance = resistances[0] if resistances else current_price * 1.08
    return nearest_support, nearest_resistance, atr_value


def _last_completed_bar_idx(df: pd.DataFrame) -> int:
    """Use prior session when the latest row is still today's incomplete daily bar."""
    if len(df) < 2:
        return -1
    last_ts = pd.Timestamp(df.index[-1])
    if last_ts.tz is not None:
        last_ts = last_ts.tz_convert("Asia/Kolkata")
    else:
        last_ts = last_ts.tz_localize("Asia/Kolkata")
    today = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    if last_ts.normalize() >= today:
        return -2
    return -1


# --- Candlestick patterns -----------------------------------------------------

def detect_marubozu(
    df: pd.DataFrame,
    wick_max_frac: float = 0.03,
    min_body_frac: float = 0.62,
    min_volume_mult_vs_ma: Optional[float] = 1.2,
    volume_ma_period: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bullish/bearish Marubozu on daily OHLCV rows."""
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
