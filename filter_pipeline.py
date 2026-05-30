"""
Multi-stage NIFTY 50 stock filter pipeline.

Universe → Trend → Momentum → Volume → S/R → Candlestick → Risk/Reward → Watchlist
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from nifty50_symbols import get_nifty50_symbols, to_yahoo_nse
from patterns import latest_bullish_confirmation

_OHLCV_COLS = ("Open", "High", "Low", "Close", "Volume")

STAGE_NAMES = (
    "Universe",
    "Trend",
    "Momentum",
    "Volume",
    "Support/Resistance",
    "Candlestick",
    "Risk/Reward",
    "Watchlist",
)


@dataclass
class FilterConfig:
    """Thresholds for each pipeline stage."""

    period: str = "2y"
    batch_size: int = 12
    download_delay: float = 0.12
    prefer_live_symbols: bool = True

    # Trend: price > EMA20 > EMA50 and price above EMA200
    require_above_ema200: bool = True

    # Momentum
    min_rsi: float = 45.0
    max_rsi: float = 72.0
    require_macd_bullish: bool = True

    # Volume
    min_volume_mult: float = 1.1
    volume_ma_period: int = 20

    # Support / resistance (daily swing + pivots)
    sr_lookback_days: int = 180
    max_dist_to_support_pct: float = 0.035
    min_room_to_resistance_pct: float = 0.02

    # Candlestick
    candlestick_lookback: int = 3

    # Risk / reward
    min_rr_ratio: float = 2.0
    max_stop_distance_pct: float = 0.06


@dataclass
class StageCounts:
    """Symbol counts after each stage (cumulative survivors)."""

    universe: int = 0
    trend: int = 0
    momentum: int = 0
    volume: int = 0
    support_resistance: int = 0
    candlestick: int = 0
    risk_reward: int = 0
    watchlist: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "Universe": self.universe,
            "Trend": self.trend,
            "Momentum": self.momentum,
            "Volume": self.volume,
            "Support/Resistance": self.support_resistance,
            "Candlestick": self.candlestick,
            "Risk/Reward": self.risk_reward,
            "Watchlist": self.watchlist,
        }


@dataclass
class WatchlistRow:
    symbol: str
    yahoo_ticker: str
    price: float
    trend: str
    rsi: float
    macd_hist: float
    volume_ratio: float
    nearest_support: float
    nearest_resistance: float
    dist_support_pct: float
    dist_resistance_pct: float
    candlestick_pattern: str
    risk_reward: float
    stop: float
    target: float
    score: float


@dataclass
class PipelineResult:
    config: FilterConfig
    counts: StageCounts
    watchlist: list[WatchlistRow] = field(default_factory=list)
    stage_details: dict[str, list[str]] = field(default_factory=dict)
    missing_symbols: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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


def compute_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = close.astype(float)
    ema12 = compute_ema(close, 12)
    ema26 = compute_ema(close, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


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


def infer_support_resistance(df: pd.DataFrame, current_price: float, lookback_days: int) -> tuple[float, float, float]:
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


@dataclass
class _Eval:
    symbol: str
    yahoo: str
    df: pd.DataFrame
    passed_trend: bool = False
    passed_momentum: bool = False
    passed_volume: bool = False
    passed_sr: bool = False
    passed_candle: bool = False
    passed_rr: bool = False
    row: Optional[WatchlistRow] = None


def _evaluate_symbol(symbol: str, yahoo: str, df: pd.DataFrame, cfg: FilterConfig) -> _Eval:
    ev = _Eval(symbol=symbol, yahoo=yahoo, df=df)
    if len(df) < 220:
        return ev

    close = df["Close"].astype(float)
    price = float(close.iloc[-1])
    ema20 = float(compute_ema(close, 20).iloc[-1])
    ema50 = float(compute_ema(close, 50).iloc[-1])
    ema200 = float(compute_ema(close, 200).iloc[-1])

    trend_ok = price > ema20 > ema50
    if cfg.require_above_ema200:
        trend_ok = trend_ok and price > ema200
    if not trend_ok:
        return ev
    ev.passed_trend = True

    rsi = float(compute_rsi(close).iloc[-1])
    _, _, macd_hist = compute_macd(close)
    hist = float(macd_hist.iloc[-1])
    momentum_ok = cfg.min_rsi <= rsi <= cfg.max_rsi
    if cfg.require_macd_bullish:
        momentum_ok = momentum_ok and hist > 0
    if not momentum_ok:
        return ev
    ev.passed_momentum = True

    vol = df["Volume"].astype(float)
    vol_ma = vol.rolling(cfg.volume_ma_period).mean()
    vol_ratio = float(vol.iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 0.0
    if vol_ratio < cfg.min_volume_mult:
        return ev
    ev.passed_volume = True

    support, resistance, _ = infer_support_resistance(df, price, cfg.sr_lookback_days)
    dist_support_pct = (price - support) / price if price > 0 else 1.0
    dist_resistance_pct = (resistance - price) / price if price > 0 else 0.0

    sr_ok = dist_support_pct <= cfg.max_dist_to_support_pct and dist_resistance_pct >= cfg.min_room_to_resistance_pct
    if not sr_ok:
        return ev
    ev.passed_sr = True

    confirmed, pattern = latest_bullish_confirmation(df, lookback=cfg.candlestick_lookback)
    if not confirmed:
        return ev
    ev.passed_candle = True

    risk = price - support
    reward = resistance - price
    if risk <= 0 or reward <= 0:
        return ev
    rr = reward / risk
    stop_pct = risk / price
    if rr < cfg.min_rr_ratio or stop_pct > cfg.max_stop_distance_pct:
        return ev
    ev.passed_rr = True

    score = min(100.0, rr * 12 + vol_ratio * 8 + (70 - abs(rsi - 58)) * 0.5)
    ev.row = WatchlistRow(
        symbol=symbol,
        yahoo_ticker=yahoo,
        price=round(price, 2),
        trend="Bullish",
        rsi=round(rsi, 1),
        macd_hist=round(hist, 3),
        volume_ratio=round(vol_ratio, 2),
        nearest_support=round(support, 2),
        nearest_resistance=round(resistance, 2),
        dist_support_pct=round(dist_support_pct * 100, 2),
        dist_resistance_pct=round(dist_resistance_pct * 100, 2),
        candlestick_pattern=pattern,
        risk_reward=round(rr, 2),
        stop=round(support, 2),
        target=round(resistance, 2),
        score=round(score, 1),
    )
    return ev


def run_pipeline(
    cfg: Optional[FilterConfig] = None,
    *,
    progress_callback: Optional[Any] = None,
) -> PipelineResult:
    cfg = cfg or FilterConfig()
    symbols = get_nifty50_symbols(prefer_live=cfg.prefer_live_symbols)
    yahoo_map = {s: to_yahoo_nse(s) for s in symbols}
    yahoo_tickers = [yahoo_map[s] for s in symbols]

    counts = StageCounts(universe=len(symbols))
    stage_details: dict[str, list[str]] = {name: [] for name in STAGE_NAMES[1:-1]}
    evaluations: list[_Eval] = []
    missing: list[str] = []
    errors: list[str] = []

    total_batches = (len(yahoo_tickers) + cfg.batch_size - 1) // cfg.batch_size
    for batch_idx, i in enumerate(range(0, len(yahoo_tickers), cfg.batch_size)):
        batch_yahoo = yahoo_tickers[i : i + cfg.batch_size]
        batch_nse = symbols[i : i + cfg.batch_size]
        if progress_callback:
            progress_callback(batch_idx + 1, total_batches, batch_nse[0])

        try:
            frames = download_daily_batch(batch_yahoo, cfg.period)
        except Exception as exc:
            errors.append(f"Batch {batch_nse[0]}: {exc}")
            frames = {}

        for nse_symbol, yahoo_ticker in zip(batch_nse, batch_yahoo):
            df = frames.get(yahoo_ticker)
            if df is None or df.empty:
                df = download_daily_single(yahoo_ticker, cfg.period)
            if df is None or df.empty:
                missing.append(nse_symbol)
                continue
            evaluations.append(_evaluate_symbol(nse_symbol, yahoo_ticker, df, cfg))

        if i + cfg.batch_size < len(yahoo_tickers) and cfg.download_delay > 0:
            time.sleep(cfg.download_delay)

    trend_syms = [e.symbol for e in evaluations if e.passed_trend]
    counts.trend = len(trend_syms)
    stage_details["Trend"] = trend_syms

    momentum_syms = [e.symbol for e in evaluations if e.passed_momentum]
    counts.momentum = len(momentum_syms)
    stage_details["Momentum"] = momentum_syms

    volume_syms = [e.symbol for e in evaluations if e.passed_volume]
    counts.volume = len(volume_syms)
    stage_details["Volume"] = volume_syms

    sr_syms = [e.symbol for e in evaluations if e.passed_sr]
    counts.support_resistance = len(sr_syms)
    stage_details["Support/Resistance"] = sr_syms

    candle_syms = [e.symbol for e in evaluations if e.passed_candle]
    counts.candlestick = len(candle_syms)
    stage_details["Candlestick"] = candle_syms

    rr_syms = [e.symbol for e in evaluations if e.passed_rr]
    counts.risk_reward = len(rr_syms)
    stage_details["Risk/Reward"] = rr_syms

    watchlist = [e.row for e in evaluations if e.row is not None]
    watchlist.sort(key=lambda r: r.score, reverse=True)
    counts.watchlist = len(watchlist)

    return PipelineResult(
        config=cfg,
        counts=counts,
        watchlist=watchlist,
        stage_details=stage_details,
        missing_symbols=missing,
        errors=errors,
    )


def watchlist_to_dataframe(rows: list[WatchlistRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in rows])


@dataclass
class ShortlistConfig:
    """Quick screen: Close > EMA20, RSI > min_rsi, Volume > avg volume."""

    period: str = "1y"
    batch_size: int = 12
    download_delay: float = 0.12
    prefer_live_symbols: bool = True
    ema_period: int = 20
    rsi_period: int = 14
    min_rsi: float = 55.0
    volume_ma_period: int = 20
    min_history_bars: int = 30


@dataclass
class ShortlistRow:
    symbol: str
    yahoo_ticker: str
    close: float
    ema20: float
    pct_above_ema20: float
    rsi: float
    volume: float
    avg_volume: float
    volume_vs_avg_pct: float
    as_of: str


@dataclass
class ShortlistResult:
    config: ShortlistConfig
    universe: int
    passed_ema: int
    passed_rsi: int
    passed_volume: int
    shortlist: list[ShortlistRow] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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


def _evaluate_shortlist(
    symbol: str,
    yahoo: str,
    df: pd.DataFrame,
    cfg: ShortlistConfig,
) -> tuple[Optional[ShortlistRow], dict[str, bool]]:
    flags = {"ema": False, "rsi": False, "volume": False}
    if len(df) < cfg.min_history_bars:
        return None, flags

    idx = _last_completed_bar_idx(df)
    close = df["Close"].astype(float)
    price = float(close.iloc[idx])
    ema20 = float(compute_ema(close, cfg.ema_period).iloc[idx])
    rsi = float(compute_rsi(close, cfg.rsi_period).iloc[idx])

    vol = df["Volume"].astype(float)
    avg_vol = float(vol.rolling(cfg.volume_ma_period).mean().iloc[idx])
    last_vol = float(vol.iloc[idx])

    flags["ema"] = price > ema20
    flags["rsi"] = rsi > cfg.min_rsi
    flags["volume"] = avg_vol > 0 and last_vol > avg_vol

    if not all(flags.values()):
        return None, flags

    pct_above = ((price - ema20) / ema20 * 100.0) if ema20 > 0 else 0.0
    vol_vs_avg = ((last_vol - avg_vol) / avg_vol * 100.0) if avg_vol > 0 else 0.0
    as_of = pd.Timestamp(df.index[idx]).strftime("%Y-%m-%d")

    row = ShortlistRow(
        symbol=symbol,
        yahoo_ticker=yahoo,
        close=round(price, 2),
        ema20=round(ema20, 2),
        pct_above_ema20=round(pct_above, 2),
        rsi=round(rsi, 1),
        volume=int(last_vol),
        avg_volume=int(avg_vol),
        volume_vs_avg_pct=round(vol_vs_avg, 1),
        as_of=as_of,
    )
    return row, flags


def run_shortlist_screener(
    cfg: Optional[ShortlistConfig] = None,
    *,
    progress_callback: Optional[Any] = None,
) -> ShortlistResult:
    cfg = cfg or ShortlistConfig()
    symbols = get_nifty50_symbols(prefer_live=cfg.prefer_live_symbols)
    yahoo_map = {s: to_yahoo_nse(s) for s in symbols}
    yahoo_tickers = [yahoo_map[s] for s in symbols]

    passed_ema = 0
    passed_rsi = 0
    passed_volume = 0
    shortlist: list[ShortlistRow] = []
    missing: list[str] = []
    errors: list[str] = []

    total_batches = (len(yahoo_tickers) + cfg.batch_size - 1) // cfg.batch_size
    for batch_idx, i in enumerate(range(0, len(yahoo_tickers), cfg.batch_size)):
        batch_yahoo = yahoo_tickers[i : i + cfg.batch_size]
        batch_nse = symbols[i : i + cfg.batch_size]
        if progress_callback:
            progress_callback(batch_idx + 1, total_batches, batch_nse[0])

        try:
            frames = download_daily_batch(batch_yahoo, cfg.period)
        except Exception as exc:
            errors.append(f"Batch {batch_nse[0]}: {exc}")
            frames = {}

        for nse_symbol, yahoo_ticker in zip(batch_nse, batch_yahoo):
            df = frames.get(yahoo_ticker)
            if df is None or df.empty:
                df = download_daily_single(yahoo_ticker, cfg.period)
            if df is None or df.empty:
                missing.append(nse_symbol)
                continue

            row, flags = _evaluate_shortlist(nse_symbol, yahoo_ticker, df, cfg)
            if flags["ema"]:
                passed_ema += 1
            if flags["ema"] and flags["rsi"]:
                passed_rsi += 1
            if flags["ema"] and flags["rsi"] and flags["volume"]:
                passed_volume += 1
            if row is not None:
                shortlist.append(row)

        if i + cfg.batch_size < len(yahoo_tickers) and cfg.download_delay > 0:
            time.sleep(cfg.download_delay)

    shortlist.sort(key=lambda r: (r.rsi, r.volume_vs_avg_pct), reverse=True)
    failed = [s for s in symbols if s not in {r.symbol for r in shortlist} and s not in missing]

    return ShortlistResult(
        config=cfg,
        universe=len(symbols),
        passed_ema=passed_ema,
        passed_rsi=passed_rsi,
        passed_volume=passed_volume,
        shortlist=shortlist,
        failed_symbols=failed,
        missing_symbols=missing,
        errors=errors,
    )


def shortlist_to_dataframe(rows: list[ShortlistRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in rows])
