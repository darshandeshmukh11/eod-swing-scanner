# Agent Guide: EOD Swing Scanner

This project scans NSE stocks for potential next-session swing long ideas. The main scanner is `eod_swing_scanner.py`; the Streamlit app and Telegram notifier both use it.

## Main Flow

1. Build the stock universe.
   - Default: NIFTY 50 + NIFTY 100.
   - Optional: NIFTY 50 only.
2. Download daily OHLCV data from Yahoo Finance.
3. Optionally merge live LTP into the latest daily bar.
4. Apply all hard filters.
5. Calculate support, resistance, pivots, suggested entry, stop levels, and targets.
6. Sort hits by quality score, pattern count, RSI, and volume strength.

## Hard Filters For A Potential Buy

A stock is shortlisted only if all of these pass:

| Filter | Default Rule |
|---|---|
| Trend | `price > EMA20 > EMA50` |
| RSI | `RSI(14) > 55` |
| Volume | Session volume > 20-day average volume |
| SuperTrend | Bullish SuperTrend required by default |
| Quality score | At least `2` out of `6` quality signals |

The quality score counts these six signals:

| Signal | Condition |
|---|---|
| ST bull | SuperTrend is bullish |
| MACD bull | MACD line is above signal line |
| MACD rising | MACD histogram is higher than previous bar |
| RSI rising | RSI is higher than it was N bars ago; default N is `5` |
| ST flip | SuperTrend flipped bullish within the lookback; default lookback is `5` bars |
| ADX trend | ADX(14) is at least `20` |

Some quality signals can be turned into mandatory filters from the Streamlit sidebar: MACD bullish, MACD histogram rising, RSI rising, recent SuperTrend flip, and ADX trend strength. They are off by default.

## Indicators Used

| Indicator | Default Parameters | Used For |
|---|---:|---|
| EMA | 20, 50 | Trend filter and dip-entry planning |
| RSI | 14 | Momentum filter and quality score |
| Volume average | 20 sessions | Volume confirmation |
| SuperTrend | ATR 10, multiplier 3.0 | Trend confirmation and quality score |
| MACD | EMA 12, EMA 26, signal 9 | Quality score |
| ADX | 14 | Quality score |
| ATR | 14 | Support/resistance clustering tolerance |
| Candlestick patterns | latest completed daily bar | Context only |

Candlestick patterns currently detected for context are hammer, bullish engulfing, and marubozu. They are reported and used in sorting, but they are not hard filters in the main scanner.

## Support And Resistance

Support/resistance comes from `infer_support_resistance` in `eod_swing_lib.py`.

The scanner looks back `120` daily bars by default. It finds swing highs and swing lows using a `3`-bar lookback. It also includes previous-session floor pivot levels.

The clustering tolerance is:

```text
max(ATR(14) * 0.35, current_price * 0.004)
```

Support candidates are clustered swing lows plus previous-session pivot levels below the current price. Resistance candidates are clustered swing highs plus previous-session pivot levels above the current price.

If no support is found, fallback support is:

```text
current_price * 0.95
```

If no resistance is found, fallback resistance is:

```text
current_price * 1.08
```

## Pivot, S1, S2, Target 1, Target 2

The app uses classic floor pivots from the scanned session high, low, and close. This is implemented in `immediate_floor_pivots` in `eod_swing_scanner.py`.

In EOD mode, the input is the latest completed daily bar. In realtime mode, the input can be today's bar after live LTP has been merged.

Formulas:

```text
Pivot = (High + Low + Close) / 3
R1 = 2 * Pivot - Low
S1 = 2 * Pivot - High
R2 = Pivot + (High - Low)
S2 = Pivot - (High - Low)
```

The app labels them like this:

| Level | App Meaning |
|---|---|
| S1 | Stop (S1), tighter stop |
| S2 | Stop (S2), wider stop |
| R1 | Target (R1), first target |
| R2 | Target (R2), stretch target |

So "Target 1" is R1, and "Target 2" is R2.

## How Buy Zones Are Decided

Buy-zone logic starts in `suggest_swing_entry` in `eod_swing_scanner.py`, then the chart zone is prepared in `_compute_buy_sell_zones` in `eod_swing_app.py`.

### Breakout Setup

If the stock has broken above resistance:

```text
breakout = price > resistance * (1 + 0.15 / 100)
           and previous_close <= resistance
```

Then:

```text
suggested_entry = resistance * 1.0015
entry_low = resistance
entry_high = suggested_entry
```

The chart buy zone becomes:

```text
low = min(resistance, suggested_entry)
high = max(resistance * 1.001, suggested_entry)
```

The intent is to enter only if the stock holds above resistance with volume.

### Dip-Buy Setup

If the stock is not a breakout, the scanner plans a dip-buy zone around S1, support, pivot, and EMA20.

```text
entry_primary = max(S1, support, Pivot * 0.998)
entry_secondary = min(EMA20, (S1 + Pivot) / 2)
```

If `entry_secondary` is above current price, it is adjusted down:

```text
entry_secondary = EMA20 * 0.995
```

Then:

```text
entry_low = min(entry_primary, entry_secondary)
entry_high = max(entry_primary, entry_secondary)
suggested_entry = (entry_primary + entry_secondary) / 2
```

If the suggested entry is still above current price:

```text
suggested_entry = min(entry_primary, EMA20 * 0.998)
entry_high = min(entry_high, current_price)
```

The chart buy zone is normally:

```text
[entry_low, entry_high]
```

If entry values are missing, the app falls back to:

```text
[min(S1, support, Pivot), max(S1, support, Pivot)]
```

## Sell Zone

The sell zone is always based on R1 and R2:

```text
sell_zone = [min(R1, R2), max(R1, R2)]
```

In normal floor-pivot conditions, this is simply:

```text
[R1, R2]
```

## Important Defaults

| Setting | Default |
|---|---:|
| History period | 1 year |
| Min RSI | 55 |
| EMA fast / slow | 20 / 50 |
| Volume average | 20 days |
| Support/resistance lookback | 120 days |
| Near EMA threshold | 2.0% |
| Near support threshold | 3.5% |
| Breakout buffer | 0.15% |
| SuperTrend ATR period | 10 |
| SuperTrend multiplier | 3.0 |
| Min quality score | 2 |
| ADX trend threshold | 20 |

## Other Scanner Code

`filter_pipeline.py` contains a separate multi-stage NIFTY 50 pipeline. It is stricter and includes EMA200, RSI range, MACD, volume multiple, support proximity, candlestick confirmation, and risk/reward checks. It is useful context, but it is not the primary buy-idea scanner used by the app and Telegram workflow.
