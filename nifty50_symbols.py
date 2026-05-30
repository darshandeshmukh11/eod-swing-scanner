"""NIFTY 50 constituent symbols (NSE tickers without .NS suffix)."""

from __future__ import annotations

import pandas as pd

# Fallback when Wikipedia is unreachable; update after index rebalances.
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


def get_nifty50_symbols(prefer_live: bool = True) -> list[str]:
    if prefer_live:
        try:
            return _symbols_from_wikipedia()
        except Exception:
            pass
    return sorted({normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK})


# NSE renames (Zomato → Eternal, etc.) — use current NSE symbol in scan output.
NSE_SYMBOL_RENAMES: dict[str, str] = {
    "ZOMATO": "ETERNAL",
}

# NSE symbol → Yahoo ticker when the default {SYMBOL}.NS does not work.
YAHOO_TICKER_ALIASES: dict[str, str] = {
    "TATAMOTORS": "TMPV.NS",  # post-demerger; TATAMOTORS.NS is delisted on Yahoo
    "ZOMATO": "ETERNAL.NS",  # renamed to Eternal Limited (Mar 2025)
}


def normalize_nse_symbol(symbol: str) -> str:
    key = symbol.strip().upper()
    return NSE_SYMBOL_RENAMES.get(key, key)


def to_yahoo_nse(symbol: str) -> str:
    raw = symbol.strip().upper()
    key = normalize_nse_symbol(raw)
    if raw in YAHOO_TICKER_ALIASES:
        return YAHOO_TICKER_ALIASES[raw]
    return f"{key}.NS"


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


# Extra names often in NIFTY 100 but outside the static NIFTY 50 fallback (for offline extended scan).
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


def get_nifty100_symbols(prefer_live: bool = True) -> list[str]:
    """NIFTY 100 constituents (NSE symbols). Falls back to static union if live fetch fails."""
    if prefer_live:
        try:
            return _symbols_from_wikipedia_title("NIFTY 100", min_count=90)
        except Exception:
            pass
    return sorted(
        {normalize_nse_symbol(s) for s in NIFTY_50_FALLBACK}
        | {normalize_nse_symbol(s) for s in NIFTY_100_EXTRA_FALLBACK}
    )


def get_extended_universe_symbols(prefer_live: bool = True) -> list[str]:
    """Symbols outside NIFTY 50 but inside NIFTY 100."""
    n50 = set(get_nifty50_symbols(prefer_live=prefer_live))
    n100 = get_nifty100_symbols(prefer_live=prefer_live)
    return sorted(s for s in n100 if s not in n50)


def get_nifty50_and_100_universe(prefer_live: bool = True) -> tuple[list[str], set[str]]:
    """
    Full scan universe: union of NIFTY 50 and NIFTY 100 (deduplicated).
    Returns (sorted symbols, set of NIFTY 50 members for per-hit labeling).
    """
    n50 = get_nifty50_symbols(prefer_live=prefer_live)
    n100 = get_nifty100_symbols(prefer_live=prefer_live)
    n50_set = {normalize_nse_symbol(s) for s in n50}
    all_symbols = sorted(n50_set | {normalize_nse_symbol(s) for s in n100})
    return all_symbols, n50_set
