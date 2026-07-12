"""
Self-contained helpers for the EOD swing scanner.

Bundled here so Streamlit Cloud / standalone deploy only needs files in this
folder — no separate ``filter_pipeline``, ``nifty50_symbols``, or ``patterns``
modules required at import time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NSE_MARKET_OPEN = time(9, 15)
NSE_MARKET_CLOSE = time(15, 30)

_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")


@lru_cache(maxsize=1)
def _yf_session() -> requests.Session:
    """
    Plain requests session for Yahoo Finance.

    Newer yfinance defaults to curl_cffi, which Segmentation-faults on
    Streamlit Community Cloud (seen as crash in /app/scripts/run-streamlit.sh).
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return session

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
        threads=False,
        session=_yf_session(),
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
    df = yf.download(
        yahoo_ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        session=_yf_session(),
    )
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


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    atr = compute_atr_series(df, period)
    val = float(atr.iloc[-1]) if len(atr) and not np.isnan(atr.iloc[-1]) else np.nan
    return val


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    macd_line = compute_ema(close, fast) - compute_ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=close.index,
    )


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength)."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = compute_atr_series(df, period)
    atr = tr.replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    SuperTrend bands. ``direction``: 1 = bullish (line below price), -1 = bearish.
    """
    if df.empty:
        return pd.DataFrame(columns=["supertrend", "direction"])

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    atr = compute_atr_series(df, period)
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    n = len(df)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    final_upper[0] = basic_upper.iloc[0]
    final_lower[0] = basic_lower.iloc[0]
    supertrend[0] = final_lower[0]
    direction[0] = 1

    for i in range(1, n):
        if np.isnan(atr.iloc[i]):
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue

        bu = float(basic_upper.iloc[i])
        bl = float(basic_lower.iloc[i])
        c = float(close.iloc[i])
        c_prev = float(close.iloc[i - 1])

        if bu < final_upper[i - 1] or c_prev > final_upper[i - 1]:
            final_upper[i] = bu
        else:
            final_upper[i] = final_upper[i - 1]

        if bl > final_lower[i - 1] or c_prev < final_lower[i - 1]:
            final_lower[i] = bl
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == 1:
            if c < final_lower[i]:
                direction[i] = -1
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]
        elif c > final_upper[i]:
            direction[i] = 1
            supertrend[i] = final_lower[i]
        else:
            direction[i] = -1
            supertrend[i] = final_upper[i]

    return pd.DataFrame(
        {"supertrend": supertrend, "direction": direction},
        index=df.index,
    )


def supertrend_flip_bars_ago(direction: pd.Series, idx: int, lookback: int = 5) -> Optional[int]:
    """Bars since last bullish flip (1), or None if no flip within lookback."""
    if idx < 1 or lookback < 1:
        return None
    start = max(1, idx - lookback + 1)
    for i in range(idx, start - 1, -1):
        if direction.iloc[i] == 1 and direction.iloc[i - 1] == -1:
            return idx - i
    return None


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


def ist_now() -> datetime:
    return datetime.now(IST)


def _today_ts() -> pd.Timestamp:
    return pd.Timestamp(ist_now().date())


def market_session_status() -> dict[str, str | bool]:
    """NSE cash session phase for UI banners."""
    now = ist_now()
    weekday = now.weekday()
    t = now.time()
    as_of = now.strftime("%a %d %b %Y, %H:%M IST")

    if weekday >= 5:
        return {"phase": "closed (weekend)", "is_open": False, "as_of": as_of}
    if t < NSE_MARKET_OPEN:
        return {"phase": "pre-open", "is_open": False, "as_of": as_of}
    if t <= NSE_MARKET_CLOSE:
        return {"phase": "open", "is_open": True, "as_of": as_of}
    return {"phase": "closed", "is_open": False, "as_of": as_of}


@dataclass
class LiveQuote:
    symbol: str
    yahoo: str
    price: float
    day_high: float
    day_low: float
    volume: float


def normalize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(IST)
    else:
        idx = idx.tz_localize(IST)
    out.index = idx.normalize().tz_localize(None)
    return out.sort_index()


def merge_realtime_session(
    df: pd.DataFrame,
    live_price: float,
    *,
    day_high: Optional[float] = None,
    day_low: Optional[float] = None,
    day_volume: Optional[float] = None,
) -> pd.DataFrame:
    """Patch or append today's daily bar with live LTP (and optional session H/L/volume)."""
    out = normalize_daily_index(df)
    if out.empty:
        return out

    live_price = float(live_price)
    today_ts = _today_ts()
    hi = float(day_high) if day_high and day_high > 0 else live_price
    low = float(day_low) if day_low and day_low > 0 else live_price

    if out.index[-1] >= today_ts:
        o = float(out["Open"].iloc[-1])
        h = max(float(out["High"].iloc[-1]), hi, live_price)
        low_val = min(float(out["Low"].iloc[-1]), low, live_price)
        out.iloc[-1, out.columns.get_loc("High")] = h
        out.iloc[-1, out.columns.get_loc("Low")] = low_val
        out.iloc[-1, out.columns.get_loc("Close")] = live_price
        if day_volume is not None and day_volume > 0 and "Volume" in out.columns:
            out.iloc[-1, out.columns.get_loc("Volume")] = float(day_volume)
    else:
        prev_close = float(out["Close"].iloc[-1])
        vol = float(day_volume) if day_volume and day_volume > 0 else 0.0
        out.loc[today_ts] = {
            "Open": prev_close,
            "High": max(prev_close, hi, live_price),
            "Low": min(prev_close, low, live_price),
            "Close": live_price,
            "Volume": vol,
        }
    return out


def has_today_bar(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return pd.Timestamp(df.index[-1]).normalize() >= _today_ts()


def _last_completed_bar_idx(df: pd.DataFrame) -> int:
    """Use prior session when the latest row is still today's incomplete daily bar."""
    if len(df) < 2:
        return -1
    if has_today_bar(df):
        return -2
    return -1


def session_is_developing(df: pd.DataFrame, volume_ma_period: int = 20) -> bool:
    """
    True when today's bar should not drive volume/pattern filters yet.

    During NSE hours we always treat volume as developing; after hours we use
    today's bar only when session volume is a meaningful fraction of the 20-day avg.
    """
    if not has_today_bar(df) or len(df) < 2:
        return False

    now = ist_now()
    if now.weekday() < 5 and NSE_MARKET_OPEN <= now.time() <= NSE_MARKET_CLOSE:
        return True

    idx = len(df) - 1
    vol = float(df["Volume"].astype(float).iloc[idx])
    if vol <= 0:
        return True

    avg_series = df["Volume"].astype(float).rolling(volume_ma_period).mean()
    avg_vol = float(avg_series.iloc[idx - 1]) if idx >= 1 else 0.0
    if avg_vol > 0 and vol < avg_vol * 0.25:
        return True
    return False


def session_bar_indices(
    df: pd.DataFrame,
    volume_ma_period: int = 20,
    *,
    realtime: bool,
) -> tuple[int, int, int]:
    """Return (price_idx, volume_idx, pattern_idx) for filter evaluation."""
    if not realtime:
        idx = _last_completed_bar_idx(df)
        return idx, idx, idx

    price_idx = len(df) - 1
    if session_is_developing(df, volume_ma_period):
        vol_idx = _last_completed_bar_idx(df)
    else:
        vol_idx = price_idx
    return price_idx, vol_idx, vol_idx


def _pick_positive(*values: object) -> Optional[float]:
    for val in values:
        if val is None:
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num > 0:
            return num
    return None


def fetch_live_quote(nse_symbol: str) -> Optional[LiveQuote]:
    """Latest LTP for an NSE symbol via Yahoo Finance."""
    yahoo = to_yahoo_nse(nse_symbol)
    try:
        ticker = yf.Ticker(yahoo, session=_yf_session())
        price: Optional[float] = None
        day_high = 0.0
        day_low = 0.0
        volume = 0.0

        try:
            fi = ticker.fast_info
            price = _pick_positive(
                getattr(fi, "last_price", None),
                getattr(fi, "lastPrice", None),
            )
            day_high = float(getattr(fi, "day_high", 0) or getattr(fi, "dayHigh", 0) or price or 0)
            day_low = float(getattr(fi, "day_low", 0) or getattr(fi, "dayLow", 0) or price or 0)
            volume = float(getattr(fi, "last_volume", 0) or getattr(fi, "lastVolume", 0) or 0)
        except Exception:
            pass

        if price is None:
            info = ticker.info or {}
            price = _pick_positive(
                info.get("regularMarketPrice"),
                info.get("currentPrice"),
                info.get("previousClose"),
            )
            day_high = float(info.get("dayHigh") or info.get("regularMarketDayHigh") or price or 0)
            day_low = float(info.get("dayLow") or info.get("regularMarketDayLow") or price or 0)
            volume = float(info.get("volume") or info.get("regularMarketVolume") or 0)

        if price is None:
            hist = ticker.history(period="5d", interval="1d")
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
                day_high = float(hist["High"].iloc[-1])
                day_low = float(hist["Low"].iloc[-1])
                volume = float(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0.0

        if price is None or price <= 0:
            return None

        return LiveQuote(
            symbol=nse_symbol,
            yahoo=yahoo,
            price=round(price, 2),
            day_high=round(day_high or price, 2),
            day_low=round(day_low or price, 2),
            volume=volume,
        )
    except Exception:
        return None


def fetch_live_quotes_batch(
    nse_symbols: list[str],
    *,
    max_workers: int = 10,
) -> dict[str, LiveQuote]:
    if not nse_symbols:
        return {}
    out: dict[str, LiveQuote] = {}
    workers = min(max_workers, max(1, len(nse_symbols)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_live_quote, sym): sym for sym in nse_symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                quote = fut.result()
            except Exception:
                quote = None
            if quote and quote.price > 0:
                out[sym] = quote
    return out


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
