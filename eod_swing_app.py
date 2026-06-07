#!/usr/bin/env python3
"""
EOD swing scanner — Streamlit UI.

Runs ``eod_swing_scanner.run_eod_swing_scan`` and displays shortlisted NSE names
with floor pivot levels (S1/S2/R1/R2) to help plan stop loss and targets.

Run:
  cd eod-swing && streamlit run eod_swing_app.py
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from eod_swing_lib import (
    IST,
    compute_ema,
    compute_supertrend,
    download_daily_single,
    fetch_live_quote,
    ist_now,
    market_session_status,
    merge_realtime_session,
    to_yahoo_nse,
)
from eod_swing_scanner import (
    ScannerConfig,
    hits_to_dataframe,
    reevaluate_hits_from_cache,
    run_eod_swing_scan,
)

st.set_page_config(
    page_title="EOD Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIVOT_NOTE = (
    "Floor pivots from the **scanned session** high / low / close — levels for the "
    "**next** session. **Buy zone** = scale-in area · **Sell zone** = target trim area. "
    "Use **S1 / S2** for stop-loss. With **Realtime** on, close/RSI/pivots use **live LTP**; "
    "volume filter uses the **last completed session** until today's volume is meaningful."
)


def _inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0a0a0a; }
        [data-testid="stAppViewContainer"] { background-color: #0a0a0a; }
        [data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #262626;
        }
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
            color: #a1a1aa !important;
        }
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #f4f4f5 !important;
        }
        p, .stMarkdown, label, span, div { color: #d4d4d8; }
        .stCaption, [data-testid="stCaptionContainer"] { color: #a1a1aa !important; }
        [data-testid="stMetricLabel"] { color: #a1a1aa !important; }
        [data-testid="stMetricValue"] { color: #e4e4e7 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_config() -> ScannerConfig:
    st.sidebar.header("Scanner filters")
    st.sidebar.caption("Core: trend · volume · RSI · quality score")

    nifty50_only = st.sidebar.checkbox("NIFTY 50 only", value=False)
    prefer_live = st.sidebar.checkbox("Live symbol list (Wikipedia)", value=True)
    period = st.sidebar.selectbox("Price history", ["6mo", "1y", "2y"], index=1)

    st.sidebar.subheader("Thresholds")
    min_rsi = st.sidebar.slider("Min RSI(14)", 50, 75, 55)
    near_ema_pct = st.sidebar.slider("Near 20 EMA (%)", 0.5, 5.0, 2.0, 0.25)
    near_support_pct = st.sidebar.slider("Near support (%)", 1.0, 8.0, 3.5, 0.25)

    st.sidebar.subheader("SuperTrend")
    require_st = st.sidebar.checkbox(
        "Require bullish SuperTrend",
        value=True,
        help="Close above SuperTrend line with bullish direction.",
    )
    st_atr = st.sidebar.slider("ST ATR period", 7, 14, 10)
    st_mult = st.sidebar.slider("ST multiplier", 2.0, 4.0, 3.0, 0.5)

    st.sidebar.subheader("Quality / leading signals")
    min_quality = st.sidebar.slider(
        "Min quality score (0–6)",
        0,
        6,
        2,
        help="Count of: ST bull, MACD bull, MACD rising, RSI rising, ST flip, ADX trend.",
    )
    require_macd = st.sidebar.checkbox("Require MACD bullish", value=False)
    require_macd_hist = st.sidebar.checkbox(
        "Require MACD histogram rising",
        value=False,
        help="Leading: MACD momentum turning up vs prior bar.",
    )
    require_rsi_rise = st.sidebar.checkbox(
        "Require RSI rising",
        value=False,
        help="Leading: RSI higher than N bars ago.",
    )
    rsi_rise_bars = st.sidebar.slider("RSI rising lookback (bars)", 3, 10, 5, disabled=not require_rsi_rise)
    require_st_flip = st.sidebar.checkbox(
        "Require recent ST bullish flip",
        value=False,
        help="Leading: SuperTrend turned bullish within lookback window.",
    )
    st_flip_lb = st.sidebar.slider("ST flip lookback (bars)", 3, 10, 5, disabled=not require_st_flip)
    require_adx = st.sidebar.checkbox("Require ADX trend strength", value=False)
    min_adx = st.sidebar.slider("Min ADX", 15.0, 35.0, 20.0, 1.0, disabled=not require_adx)

    st.sidebar.subheader("Realtime")
    use_realtime = st.sidebar.checkbox(
        "Realtime (live LTP)",
        value=True,
        help="Merge Yahoo LTP into today's bar for trend, RSI, and next-session pivots.",
    )
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh LTP",
        value=use_realtime,
        disabled=not use_realtime,
    )
    refresh_sec = st.sidebar.slider(
        "Refresh interval (sec)",
        30,
        300,
        90,
        15,
        disabled=not (use_realtime and auto_refresh),
    )

    return ScannerConfig(
        period=period,
        min_rsi=float(min_rsi),
        near_ema_pct=float(near_ema_pct),
        near_support_pct=float(near_support_pct),
        prefer_live_symbols=prefer_live,
        nifty50_only=nifty50_only,
        use_realtime=use_realtime,
        require_supertrend=require_st,
        st_atr_period=int(st_atr),
        st_multiplier=float(st_mult),
        min_quality_score=int(min_quality),
        require_macd_bullish=require_macd,
        require_macd_hist_rising=require_macd_hist,
        require_rsi_rising=require_rsi_rise,
        rsi_rising_bars=int(rsi_rise_bars),
        require_st_flip=require_st_flip,
        st_flip_lookback=int(st_flip_lb),
        require_adx_trend=require_adx,
        min_adx=float(min_adx),
    ), auto_refresh, refresh_sec


def _prepare_display_df(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    out = raw.copy()
    out["patterns"] = out["patterns"].fillna("—")
    out["Stop (S1)"] = out["s1"]
    out["Stop (S2)"] = out["s2"]
    out["Target (R1)"] = out["r1"]
    out["Target (R2)"] = out["r2"]
    if "suggested_entry" in out.columns:
        out["Suggested entry"] = out["suggested_entry"]
        out["Entry zone"] = out.apply(
            lambda r: (
                f"₹{r['entry_low']:,.2f} – ₹{r['entry_high']:,.2f}"
                if pd.notna(r.get("entry_low")) and pd.notna(r.get("entry_high"))
                else "—"
            ),
            axis=1,
        )
        out["Entry style"] = out["entry_style"].map(
            {"dip": "Dip buy", "breakout": "Breakout"}
        ).fillna("—")
    out["Near EMA"] = out["near_ema"].map({True: "Yes", False: "—"})
    out["Near Support"] = out["near_support"].map({True: "Yes", False: "—"})
    out["Breakout"] = out["breakout_resistance"].map({True: "Yes", False: "—"})
    if "quality_score" in out.columns:
        out["Quality"] = out["quality_score"]
    if "supertrend" in out.columns:
        out["SuperTrend"] = out["supertrend"]
    if "quality_flags" in out.columns:
        out["Quality signals"] = out["quality_flags"].fillna("—")

    return out.rename(
        columns={
            "symbol": "Symbol",
            "universe": "Universe",
            "as_of": "As of",
            "close": "Close",
            "pivot": "Pivot",
            "rsi": "RSI",
            "adx": "ADX",
            "vol_vs_avg_pct": "Vol vs avg %",
            "support": "Support",
            "resistance": "Resistance",
            "patterns": "Patterns",
        }
    )


DISPLAY_COLS = [
    "Symbol",
    "Universe",
    "Quality",
    "Quality signals",
    "Close",
    "Suggested entry",
    "Entry zone",
    "Entry style",
    "Pivot",
    "SuperTrend",
    "Stop (S1)",
    "Stop (S2)",
    "Target (R1)",
    "Target (R2)",
    "RSI",
    "ADX",
    "Vol vs avg %",
    "Support",
    "Resistance",
    "Near EMA",
    "Near Support",
    "Breakout",
    "Patterns",
    "As of",
]


def _style_results_table(df: pd.DataFrame):
    stop_cols = ["Stop (S1)", "Stop (S2)"]
    target_cols = ["Target (R1)", "Target (R2)"]
    entry_cols = ["Suggested entry"]
    money_cols = [
        "Close",
        "Suggested entry",
        "Pivot",
        "SuperTrend",
        *stop_cols,
        *target_cols,
        "Support",
        "Resistance",
    ]
    show_cols = [c for c in DISPLAY_COLS if c in df.columns]

    styler = df[show_cols].style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [("background-color", "#1e293b"), ("color", "#f8fafc")],
            },
            {
                "selector": "td",
                "props": [("background-color", "#0f172a"), ("color", "#e2e8f0")],
            },
        ]
    )
    for col in entry_cols:
        if col in show_cols:
            styler = styler.set_properties(
                subset=[col], **{"color": "#60a5fa", "font-weight": "600"}
            )
    for col in stop_cols:
        if col in show_cols:
            styler = styler.set_properties(subset=[col], **{"color": "#f87171", "font-weight": "600"})
    for col in target_cols:
        if col in show_cols:
            styler = styler.set_properties(subset=[col], **{"color": "#4ade80", "font-weight": "600"})
    fmt = {c: "₹{:,.2f}" for c in money_cols if c in show_cols}
    return styler.format(fmt)


def _render_stock_detail(
    row: pd.Series,
    raw_row: pd.Series,
    *,
    period: str,
    use_live: bool = False,
) -> None:
    close = float(row["Close"])
    s1, s2 = float(row["Stop (S1)"]), float(row["Stop (S2)"])
    r1, r2 = float(row["Target (R1)"]), float(row["Target (R2)"])
    pivot = float(row["Pivot"])
    entry = float(row["Suggested entry"]) if "Suggested entry" in row and pd.notna(row["Suggested entry"]) else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Close", f"₹{close:,.2f}")
    if entry is not None:
        c2.metric(
            "Suggested entry",
            f"₹{entry:,.2f}",
            f"{(entry - close) / close * 100:+.2f}% vs close",
        )
    else:
        c2.metric("Suggested entry", "—")
    c3.metric("Pivot", f"₹{pivot:,.2f}")
    c4.metric("Stop S1", f"₹{s1:,.2f}", f"{(s1 - close) / close * 100:+.2f}% vs close")
    c5.metric("Target R1", f"₹{r1:,.2f}", f"{(r1 - close) / close * 100:+.2f}% vs close")
    c6.metric("Target R2", f"₹{r2:,.2f}", f"{(r2 - close) / close * 100:+.2f}% vs close")

    if entry is not None and row.get("Entry zone"):
        style = row.get("Entry style", "—")
        st.info(
            f"**{style}** · Zone: {row['Entry zone']} · "
            f"Use limit near **₹{entry:,.2f}** or scale in across the zone."
        )

    buy_zone, sell_zone = _compute_buy_sell_zones(row, raw_row)
    z1, z2 = st.columns(2)
    with z1:
        if buy_zone:
            st.success(f"**Buy zone:** ₹{buy_zone[0]:,.2f} – ₹{buy_zone[1]:,.2f}")
        else:
            st.caption("Buy zone: —")
    with z2:
        if sell_zone:
            st.error(f"**Sell zone:** ₹{sell_zone[0]:,.2f} – ₹{sell_zone[1]:,.2f}")
        else:
            st.caption("Sell zone: —")

    entry_row = (
        f"| **Suggested entry** | ₹{entry:,.2f} | Next-session swing entry (dip or breakout) |\n"
        f"| **Entry zone** | {row.get('Entry zone', '—')} | Scale-in range |\n"
        if entry is not None
        else ""
    )
    st.markdown(
        f"""
| Level | Price | Notes |
|-------|------:|-------|
{entry_row}| **S2** (wider stop) | ₹{s2:,.2f} | Below S1 — use if you want more room |
| **S1** (tighter stop) | ₹{s1:,.2f} | First support pivot below entry |
| **Pivot** | ₹{pivot:,.2f} | Session pivot — break below weakens bias |
| **R1** (first target) | ₹{r1:,.2f} | First resistance pivot above entry |
| **R2** (stretch target) | ₹{r2:,.2f} | Extended target |
| Swing support | ₹{float(row['Support']):,.2f} | From swing lookback |
| Swing resistance | ₹{float(row['Resistance']):,.2f} | From swing lookback |
"""
    )

    flags = []
    if row.get("Near EMA") == "Yes":
        flags.append("Near 20 EMA")
    if row.get("Near Support") == "Yes":
        flags.append("Near support")
    if row.get("Breakout") == "Yes":
        flags.append("Breakout")
    if flags:
        st.success("Context flags: " + " · ".join(flags))
    if row.get("Patterns") and row["Patterns"] != "—":
        st.info(f"Candlestick: **{row['Patterns']}**")
    if row.get("Quality signals") and row["Quality signals"] != "—":
        q = row.get("Quality", "—")
        st.success(f"Quality score **{q}/6** · {row['Quality signals']}")

    _render_daily_chart(str(row["Symbol"]), period, row, raw_row, use_live=use_live)


@st.cache_data(ttl=120, show_spinner=False)
def _load_daily_bars(symbol: str, period: str, use_live: bool = False) -> pd.DataFrame:
    df = download_daily_single(to_yahoo_nse(symbol), period)
    if df.empty:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out.sort_index()
    if use_live:
        quote = fetch_live_quote(symbol)
        if quote and quote.price > 0:
            out = merge_realtime_session(
                out,
                quote.price,
                day_high=quote.day_high,
                day_low=quote.day_low,
                day_volume=quote.volume if quote.volume > 0 else None,
            )
    return out


def _parse_as_of_date(as_of: object) -> pd.Timestamp:
    """Parse scanner ``As of`` value (e.g. ``2026-06-04`` or ``2026-06-04 (live)``)."""
    text = str(as_of).strip()
    if not text or text == "—":
        return pd.Timestamp.today().normalize()
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    return pd.Timestamp(text).normalize()


def _nearest_bar_index(index: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[int]:
    if index.empty:
        return None
    target = pd.Timestamp(target).normalize()
    pos = index.get_indexer([target], method="nearest")[0]
    return int(pos) if pos >= 0 else None


def _compute_buy_sell_zones(
    display_row: pd.Series,
    raw_row: pd.Series,
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Buy zone (entry scale-in) and sell zone (R1–R2 targets) for the next session."""
    entry = float(display_row["Suggested entry"]) if pd.notna(display_row.get("Suggested entry")) else None
    entry_low = float(raw_row["entry_low"]) if pd.notna(raw_row.get("entry_low")) else None
    entry_high = float(raw_row["entry_high"]) if pd.notna(raw_row.get("entry_high")) else None
    s1 = float(display_row["Stop (S1)"])
    pivot = float(display_row["Pivot"])
    support = float(display_row["Support"])
    resistance = float(display_row["Resistance"])
    r1 = float(display_row["Target (R1)"])
    r2 = float(display_row["Target (R2)"])
    is_breakout = display_row.get("Entry style") == "Breakout" or raw_row.get("entry_style") == "breakout"

    if is_breakout and entry is not None:
        buy_lo = round(min(resistance, entry), 2)
        buy_hi = round(max(resistance * 1.001, entry), 2)
    elif entry_low is not None and entry_high is not None and entry_low <= entry_high:
        buy_lo, buy_hi = round(entry_low, 2), round(entry_high, 2)
    elif entry is not None:
        band = max(entry * 0.004, 0.5)
        buy_lo, buy_hi = round(entry - band, 2), round(entry + band, 2)
    else:
        buy_lo = round(min(s1, support, pivot), 2)
        buy_hi = round(max(s1, support, pivot), 2)

    sell_lo, sell_hi = round(min(r1, r2), 2), round(max(r1, r2), 2)
    buy_zone = (buy_lo, buy_hi) if buy_lo < buy_hi else None
    sell_zone = (sell_lo, sell_hi) if sell_lo < sell_hi else None
    return buy_zone, sell_zone


def build_daily_swing_chart(
    ohlcv: pd.DataFrame,
    display_row: pd.Series,
    raw_row: pd.Series,
    *,
    lookback_bars: int = 120,
) -> tuple[go.Figure, Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Daily candlestick chart with EMAs, pivot levels, and EOD entry signal."""
    chart = ohlcv.tail(lookback_bars).copy()
    close = chart["Close"].astype(float)
    ema20 = compute_ema(close, 20)
    ema50 = compute_ema(close, 50)
    st_df = compute_supertrend(chart, period=10, multiplier=3.0)

    entry = float(display_row["Suggested entry"]) if pd.notna(display_row.get("Suggested entry")) else None
    entry_low = float(raw_row["entry_low"]) if pd.notna(raw_row.get("entry_low")) else None
    entry_high = float(raw_row["entry_high"]) if pd.notna(raw_row.get("entry_high")) else None
    s1 = float(display_row["Stop (S1)"])
    s2 = float(display_row["Stop (S2)"])
    r1 = float(display_row["Target (R1)"])
    r2 = float(display_row["Target (R2)"])
    pivot = float(display_row["Pivot"])
    support = float(display_row["Support"])
    resistance = float(display_row["Resistance"])
    signal_date = _parse_as_of_date(display_row["As of"])
    buy_zone, sell_zone = _compute_buy_sell_zones(display_row, raw_row)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.76, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="Daily",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=ema20,
            mode="lines",
            name="EMA 20",
            line=dict(color="#60a5fa", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=ema50,
            mode="lines",
            name="EMA 50",
            line=dict(color="#fbbf24", width=1.5),
        ),
        row=1,
        col=1,
    )
    st_colors = ["#22c55e" if d == 1 else "#ef4444" for d in st_df["direction"]]
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=st_df["supertrend"],
            mode="lines",
            name="SuperTrend",
            line=dict(color="#a78bfa", width=1.8),
            customdata=st_colors,
            hovertemplate="SuperTrend %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    vol_colors = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(chart["Close"], chart["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=chart.index,
            y=chart["Volume"],
            marker_color=vol_colors,
            name="Volume",
            opacity=0.45,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if buy_zone:
        fig.add_hrect(
            y0=buy_zone[0],
            y1=buy_zone[1],
            line_width=1,
            line_color="rgba(34, 197, 94, 0.55)",
            fillcolor="rgba(34, 197, 94, 0.22)",
            annotation_text=f"Buy zone ₹{buy_zone[0]:,.2f} – ₹{buy_zone[1]:,.2f}",
            annotation_position="top left",
            annotation=dict(font=dict(color="#86efac", size=11)),
            row=1,
            col=1,
        )
    if sell_zone:
        fig.add_hrect(
            y0=sell_zone[0],
            y1=sell_zone[1],
            line_width=1,
            line_color="rgba(239, 68, 68, 0.55)",
            fillcolor="rgba(239, 68, 68, 0.18)",
            annotation_text=f"Sell zone ₹{sell_zone[0]:,.2f} – ₹{sell_zone[1]:,.2f}",
            annotation_position="bottom left",
            annotation=dict(font=dict(color="#fca5a5", size=11)),
            row=1,
            col=1,
        )

    level_lines = [
        ("Suggested entry", entry, "#60a5fa", "dash"),
        ("S1 stop", s1, "#f87171", "solid"),
        ("S2 stop", s2, "#fca5a5", "dot"),
        ("Pivot", pivot, "#c4b5fd", "dash"),
        ("R1 target", r1, "#4ade80", "solid"),
        ("R2 target", r2, "#86efac", "dot"),
        ("Swing support", support, "#16a34a", "solid"),
        ("Swing resistance", resistance, "#dc2626", "solid"),
    ]
    for label, price, color, dash in level_lines:
        if price is None:
            continue
        fig.add_hline(
            y=price,
            line_width=1.2,
            line_color=color,
            line_dash=dash,
            annotation_text=f"{label} {price:,.2f}",
            annotation_position="right",
            row=1,
            col=1,
        )

    sig_idx = _nearest_bar_index(chart.index, signal_date)
    if sig_idx is not None:
        sig_x = chart.index[sig_idx]
        sig_close = float(chart["Close"].iloc[sig_idx])
        fig.add_trace(
            go.Scatter(
                x=[sig_x],
                y=[sig_close],
                mode="markers+text",
                name="EOD signal",
                marker=dict(
                    symbol="star",
                    size=16,
                    color="#60a5fa",
                    line=dict(width=1.5, color="#ffffff"),
                ),
                text=["EOD signal"],
                textposition="top center",
                textfont=dict(color="#93c5fd", size=11),
                hovertemplate=(
                    f"Signal bar<br>{sig_x.date()}<br>Close %{{y:,.2f}}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        if entry is not None:
            fig.add_trace(
                go.Scatter(
                    x=[sig_x],
                    y=[entry],
                    mode="markers",
                    name="Suggested entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#2563eb",
                        line=dict(width=1, color="#ffffff"),
                    ),
                    hovertemplate=f"Suggested entry<br>₹{entry:,.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=sig_x,
                y=entry,
                ax=sig_x,
                ay=sig_close,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#60a5fa",
                text=f"Entry ₹{entry:,.2f}",
                font=dict(color="#93c5fd", size=11),
                bgcolor="rgba(15, 23, 42, 0.85)",
            )

    symbol = display_row["Symbol"]
    style = display_row.get("Entry style", "—")
    fig.update_layout(
        title=dict(
            text=f"{symbol} — daily · {style} · signal {signal_date.date()}",
            font=dict(color="#f4f4f5", size=16),
        ),
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        height=640,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(10, 10, 10, 0.6)",
            font=dict(color="#e2e8f0"),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="Price (₹)",
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Volume",
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.12)",
        row=2,
        col=1,
    )
    return fig, buy_zone, sell_zone


def _render_daily_chart(
    symbol: str,
    period: str,
    display_row: pd.Series,
    raw_row: pd.Series,
    *,
    use_live: bool = False,
) -> None:
    st.subheader("Daily chart — entry signal & levels")
    ohlcv = _load_daily_bars(symbol, period, use_live=use_live)
    if ohlcv.empty:
        st.warning(f"No daily price data for **{symbol}**.")
        return
    fig, buy_zone, sell_zone = build_daily_swing_chart(ohlcv, display_row, raw_row)
    st.plotly_chart(fig, use_container_width=True)
    cap = (
        "★ **EOD signal** on scanner bar · ▲ **Suggested entry** marker · "
        "Green band = **buy zone** (scale in) · Red band = **sell zone** (targets R1–R2)."
    )
    if buy_zone and sell_zone:
        cap += f" Buy ₹{buy_zone[0]:,.2f}–₹{buy_zone[1]:,.2f} · Sell ₹{sell_zone[0]:,.2f}–₹{sell_zone[1]:,.2f}."
    st.caption(cap)


def _scan_timestamp_iso() -> str:
    """IST timestamp for scan / LTP refresh (matches NSE session banner)."""
    return ist_now().isoformat()


def _parse_updated_ist(iso_value: str) -> datetime | None:
    if not iso_value:
        return None
    try:
        ts = datetime.fromisoformat(iso_value)
        if ts.tzinfo is None:
            # Legacy scans stored naive UTC on Streamlit Cloud.
            ts = ts.replace(tzinfo=timezone.utc).astimezone(IST)
        else:
            ts = ts.astimezone(IST)
        return ts
    except ValueError:
        return None


def _render_summary_stat(
    container: Any,
    title: str,
    primary: str,
    *,
    secondary: str = "",
    primary_size: str = "1.5rem",
    nowrap: bool = True,
) -> None:
    """Metric-style block without st.metric truncation (ellipsis in narrow columns)."""
    wrap = "white-space:nowrap;" if nowrap else "white-space:normal;line-height:1.35;"
    secondary_html = ""
    if secondary:
        secondary_html = (
            '<p style="margin:0.15rem 0 0 0;font-size:0.875rem;color:rgba(250,250,250,0.65);'
            f'white-space:nowrap;">{secondary}</p>'
        )
    container.markdown(
        '<p style="margin:0 0 0.2rem 0;font-size:0.875rem;color:rgba(250,250,250,0.65);">'
        f"{title}</p>"
        f'<p style="margin:0;font-size:{primary_size};font-weight:600;{wrap}">{primary}</p>'
        f"{secondary_html}",
        unsafe_allow_html=True,
    )


def _run_scan(cfg: ScannerConfig) -> dict[str, Any]:
    hits, label, missing, errors, frame_cache, n50_set = run_eod_swing_scan(cfg)
    return {
        "raw_df": hits_to_dataframe(hits),
        "label": label,
        "missing": missing,
        "errors": errors,
        "match_count": len(hits),
        "period": cfg.period,
        "frame_cache": frame_cache,
        "n50_set": n50_set,
        "updated_at": _scan_timestamp_iso(),
        "use_realtime": cfg.use_realtime,
    }


def _refresh_scan_live(scan: dict[str, Any], cfg: ScannerConfig) -> dict[str, Any]:
    frame_cache = scan.get("frame_cache") or {}
    n50_set = scan.get("n50_set") or set()
    hits = reevaluate_hits_from_cache(frame_cache, cfg, n50_set)
    out = dict(scan)
    out["raw_df"] = hits_to_dataframe(hits)
    out["match_count"] = len(hits)
    out["updated_at"] = _scan_timestamp_iso()
    return out


def main() -> None:
    _inject_dark_theme()
    st.title("EOD Swing Scanner")
    st.caption(
        "NIFTY 50 + NIFTY 100 universe · Close > 20 EMA > 50 EMA · Volume > 20-day avg · RSI > threshold"
    )
    st.markdown(PIVOT_NOTE)

    cfg, auto_refresh, refresh_sec = _sidebar_config()
    status = market_session_status()

    run_clicked = st.sidebar.button("Run full scan", type="primary", use_container_width=True)
    refresh_clicked = st.sidebar.button(
        "Refresh live LTP",
        use_container_width=True,
        disabled=not cfg.use_realtime,
        help="Re-fetch LTP and re-filter cached history (no full re-download).",
    )
    if st.sidebar.button("Clear results", use_container_width=True):
        st.session_state.pop("eod_scan", None)
        st.rerun()

    if run_clicked:
        with st.status("Downloading prices and scanning universe…", expanded=True) as scan_status:
            try:
                st.session_state["eod_scan"] = _run_scan(cfg)
                scan_status.update(label="Scan complete", state="complete")
            except Exception as exc:
                scan_status.update(label=f"Scan failed: {exc}", state="error")
                st.error(str(exc))
                return

    if refresh_clicked and st.session_state.get("eod_scan"):
        with st.spinner("Refreshing live LTP…"):
            st.session_state["eod_scan"] = _refresh_scan_live(st.session_state["eod_scan"], cfg)
        st.rerun()

    scan = st.session_state.get("eod_scan")

    if scan and cfg.use_realtime and auto_refresh and scan.get("frame_cache"):
        updated = scan.get("updated_at")
        if updated:
            try:
                ts = datetime.fromisoformat(updated)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc).astimezone(IST)
                else:
                    ts = ts.astimezone(IST)
                if ist_now() - ts >= timedelta(seconds=refresh_sec):
                    st.session_state["eod_scan"] = _refresh_scan_live(scan, cfg)
                    st.rerun()
            except ValueError:
                pass

    if cfg.use_realtime:
        if status["is_open"]:
            st.success(
                f"**NSE open** · {status['as_of']} · Live LTP for trend/RSI/pivots; "
                f"volume filter uses the **last completed session** until today's volume is in."
            )
        else:
            st.info(
                f"**NSE {status['phase']}** · {status['as_of']} · Live LTP still updates last bar; "
                f"pivots use latest session H/L/C."
            )
    if not scan:
        st.info("Configure filters in the sidebar, then click **Run full scan**.")
        st.markdown(
            """
**Core filters (all required)**
- Close above 20 EMA and 20 EMA above 50 EMA
- Session volume above 20-day average
- RSI(14) above your minimum
- **SuperTrend** bullish (optional) + **quality score** from MACD / RSI slope / ST flip / ADX

**Pivot columns** help you set stop loss (S1/S2) and targets (R1/R2) for the next session.
**Suggested entry** is a dip-buy zone (S1 / pivot / 20 EMA) or breakout above resistance.
"""
        )
        return

    raw_df: pd.DataFrame = scan["raw_df"]
    display_df = _prepare_display_df(raw_df)

    mode_label = "Live LTP" if scan.get("use_realtime") else "EOD"
    updated_ts = _parse_updated_ist(scan.get("updated_at", ""))

    m1, m2, m3, m4, m5 = st.columns([0.9, 0.9, 1.8, 0.9, 1.3])
    m1.metric("Matches", scan["match_count"])
    m2.metric("Mode", mode_label)
    universe_label = str(scan["label"])
    universe_primary = universe_label
    universe_secondary = ""
    if " (" in universe_label and universe_label.endswith(")"):
        universe_primary, universe_secondary = universe_label.split(" (", 1)
        universe_secondary = f"({universe_secondary}"
    _render_summary_stat(
        m3,
        "Universe",
        universe_primary,
        secondary=universe_secondary,
        primary_size="1rem",
        nowrap=False,
    )
    m4.metric("Missing data", len(scan["missing"]))
    if updated_ts is not None:
        _render_summary_stat(
            m5,
            "Updated (IST)",
            updated_ts.strftime("%H:%M:%S"),
            secondary=updated_ts.strftime("%d %b %Y"),
        )
    else:
        _render_summary_stat(m5, "Updated (IST)", "—")
    if scan["errors"]:
        st.caption(f"Download errors: {len(scan['errors'])}")

    with st.expander(f"Download errors ({len(scan['errors'])})", expanded=False):
        if scan["errors"]:
            st.code("\n".join(scan["errors"][:20]))
        else:
            st.caption("None")

    if display_df.empty:
        st.warning("No stocks passed all core filters for the current settings.")
        return

    st.subheader("Shortlisted stocks — pivot levels")
    st.dataframe(
        _style_results_table(display_df),
        use_container_width=True,
        hide_index=True,
        height=min(560, 38 + len(display_df) * 35),
    )

    csv_buf = io.StringIO()
    show_cols = [c for c in DISPLAY_COLS if c in display_df.columns]
    display_df[show_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name="eod_swing_hits.csv",
        mime="text/csv",
    )

    st.subheader("Stock detail — entry, stop & target")
    symbols = display_df["Symbol"].tolist()
    pick = st.selectbox("Select symbol", symbols, index=0)
    detail_row = display_df.loc[display_df["Symbol"] == pick].iloc[0]
    raw_row = raw_df.loc[raw_df["symbol"] == pick].iloc[0]
    period = scan.get("period", "1y")
    _render_stock_detail(
        detail_row,
        raw_row,
        period=period,
        use_live=bool(scan.get("use_realtime")),
    )


if __name__ == "__main__":
    main()
