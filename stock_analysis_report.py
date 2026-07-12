#!/usr/bin/env python3
"""
Comprehensive stock analysis report generator for EOD Swing Scanner.

Produces detailed trade quality analysis for each scanned stock, similar to HAL example.
Generates both console-friendly and structured report output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from eod_swing_scanner import ScanHit


@dataclass
class QualityVerdict:
    """Detailed quality assessment and trade decision."""
    parameter: str
    scanner_value: str
    market_data_value: str
    verdict: str
    impact: str
    is_red_flag: bool = False


class StockAnalysisReport:
    """Generate detailed analysis report for a single stock."""

    def __init__(self, hit: ScanHit):
        self.hit = hit

    def _assess_rsi(self) -> list[QualityVerdict]:
        """Assess RSI signal."""
        rsi = self.hit.rsi
        verdicts = []
        
        if 50 <= rsi <= 70:
            verdict = "✅ Healthy momentum, not overbought"
            impact = "Good momentum without extreme conditions"
            flag = False
        elif rsi < 50:
            verdict = "⚠️ Below 50 — weak momentum"
            impact = "Low conviction signal"
            flag = True
        else:  # > 70
            verdict = "⚠️ Above 70 — overbought, prone to pullback"
            impact = "High risk of reversal"
            flag = True
        
        verdicts.append(QualityVerdict(
            parameter="RSI",
            scanner_value=f"{rsi}",
            market_data_value=f"{rsi}",
            verdict=verdict,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _assess_adx(self) -> list[QualityVerdict]:
        """Assess ADX trend strength."""
        adx = self.hit.adx or 0.0
        verdicts = []
        
        if adx < 20:
            verdict = "⚠️ Weak trend strength"
            impact = "High risk of reversal or consolidation"
            flag = True
        elif 20 <= adx < 25:
            verdict = "⚠️ Marginal trend strength"
            impact = "Moderate conviction, watch for ADX rising"
            flag = False
        elif 25 <= adx < 30:
            verdict = "✅ Moderate trend strength"
            impact = "Good directional move"
            flag = False
        else:
            verdict = "✅ Strong trend strength"
            impact = "Excellent directional move"
            flag = False
        
        verdicts.append(QualityVerdict(
            parameter="ADX",
            scanner_value=f"{adx:.1f}",
            market_data_value=f"{adx:.1f}",
            verdict=verdict,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _assess_volume(self) -> list[QualityVerdict]:
        """Assess volume confirmation."""
        vol_pct = self.hit.vol_vs_avg_pct
        verdicts = []
        
        if vol_pct > 30:
            verdict = "✅ Strong volume confirmation"
            impact = "Excellent move confirmation"
            flag = False
        elif vol_pct > 10:
            verdict = "✅ Good volume confirmation"
            impact = "Solid move confirmation"
            flag = False
        else:
            verdict = "⚠️ Weak volume"
            impact = "Low confirmation"
            flag = True
        
        verdicts.append(QualityVerdict(
            parameter="Volume vs Avg",
            scanner_value=f"+{vol_pct:.1f}%",
            market_data_value=f"{self.hit.volume / self.hit.avg_volume:.2f}x",
            verdict=verdict,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _assess_supertrend(self) -> list[QualityVerdict]:
        """Assess SuperTrend signal."""
        verdicts = []
        
        if self.hit.st_bullish:
            verdict = "✅ Bullish trend confirmed"
            impact = "Price above ST line, uptrend active"
            flag = False
        else:
            verdict = "⚠️ SuperTrend not bullish"
            impact = "Trend not confirmed"
            flag = True
        
        verdicts.append(QualityVerdict(
            parameter="SuperTrend",
            scanner_value=f"₹{self.hit.supertrend:.2f}" if self.hit.supertrend else "—",
            market_data_value="Bullish" if self.hit.st_bullish else "Bearish",
            verdict=verdict,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _assess_macd(self) -> list[QualityVerdict]:
        """Assess MACD momentum."""
        verdicts = []
        
        flags = []
        if self.hit.macd_bullish:
            flags.append("MACD bullish")
        if self.hit.macd_hist_rising:
            flags.append("histogram rising")
        
        if self.hit.macd_bullish and self.hit.macd_hist_rising:
            verdict = "✅ Strong bullish momentum"
            impact = "Momentum strengthening"
            flag = False
        elif self.hit.macd_bullish:
            verdict = "✅ Bullish momentum"
            impact = "Momentum is bullish"
            flag = False
        else:
            verdict = "⚠️ MACD not bullish"
            impact = "Momentum signal weak"
            flag = True
        
        verdict_str = verdict
        if flags:
            verdict_str += f" ({', '.join(flags)})"
        
        verdicts.append(QualityVerdict(
            parameter="MACD",
            scanner_value="Bull + Rising" if (self.hit.macd_bullish and self.hit.macd_hist_rising) else ("Bull" if self.hit.macd_bullish else "Bear"),
            market_data_value="Confirmed",
            verdict=verdict_str,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _assess_entry_zone(self) -> list[QualityVerdict]:
        """Assess entry zone distance from current price."""
        verdicts = []
        
        # Check if near support
        if self.hit.near_support:
            verdict = "✅ Near support zone"
            impact = "Good dip-buy opportunity"
            flag = False
        else:
            verdict = "⚠️ Far from support"
            impact = "Less attractive dip buy"
            flag = True
        
        verdicts.append(QualityVerdict(
            parameter="Entry Zone",
            scanner_value=f"₹{self.hit.support:.2f}–₹{self.hit.resistance:.2f}",
            market_data_value=f"₹{self.hit.support:.2f}–₹{self.hit.resistance:.2f} zone",
            verdict=verdict,
            impact=impact,
            is_red_flag=flag
        ))
        return verdicts

    def _calculate_quality_score_percentage(self) -> tuple[int, str]:
        """Calculate quality score as percentage and get color coding."""
        score = self.hit.quality_score
        max_score = 6
        percentage = (score / max_score) * 100
        
        if score < 2:
            return int(percentage), "🔴 VERY LOW"
        elif score < 3:
            return int(percentage), "🟠 LOW"
        elif score < 4:
            return int(percentage), "🟡 MODERATE"
        elif score < 5:
            return int(percentage), "🟢 GOOD"
        else:
            return int(percentage), "🟢 EXCELLENT"

    def _generate_trade_verdict(self) -> str:
        """Generate final trade decision based on all factors."""
        red_flags = sum(1 for v in self._get_all_verdicts() if v.is_red_flag)
        quality_score = self.hit.quality_score
        adx = self.hit.adx or 0.0
        
        # Decision logic
        if quality_score < 2:
            return "❌ SKIP — Quality score below minimum threshold"
        elif adx < 15:
            return "⚠️ CONDITIONAL PASS — Very weak trend; only take with strict stops"
        elif red_flags >= 3:
            return "❌ SKIP — Multiple red flags detected"
        elif quality_score >= 4 and adx >= 25:
            return "✅ STRONG BUY — High quality with strong trend"
        elif quality_score >= 3 and adx >= 20:
            return "✅ BUY — Good quality setup with moderate trend"
        elif quality_score >= 2 and adx >= 20:
            return "✅ CONDITIONAL PASS — Can trade with proper risk management"
        else:
            return "⚠️ BORDERLINE — Trade only if you accept higher volatility"

    def _get_all_verdicts(self) -> list[QualityVerdict]:
        """Get all verdicts from all assessment methods."""
        all_verdicts = []
        all_verdicts.extend(self._assess_supertrend())
        all_verdicts.extend(self._assess_macd())
        all_verdicts.extend(self._assess_rsi())
        all_verdicts.extend(self._assess_adx())
        all_verdicts.extend(self._assess_volume())
        all_verdicts.extend(self._assess_entry_zone())
        return all_verdicts

    def generate_table_report(self) -> pd.DataFrame:
        """Generate a pandas DataFrame table with all verdicts."""
        verdicts = self._get_all_verdicts()
        rows = [
            {
                "Parameter": v.parameter,
                "Scanner Value": v.scanner_value,
                "Market Data": v.market_data_value,
                "Verdict": v.verdict,
                "Impact": v.impact,
            }
            for v in verdicts
        ]
        return pd.DataFrame(rows)

    def generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        score_pct, score_level = self._calculate_quality_score_percentage()
        trade_verdict = self._generate_trade_verdict()
        
        report_lines = [
            f"\n{'='*80}",
            f"STOCK ANALYSIS REPORT: {self.hit.symbol}",
            f"{'='*80}\n",
            f"Quality Score: {self.hit.quality_score}/6 ({score_pct}%) {score_level}",
            f"Trade Verdict: {trade_verdict}\n",
        ]
        
        report_lines.append("📊 CORE METRICS\n" + "-"*80)
        report_lines.append(f"Close/LTP:        ₹{self.hit.close:,.2f}")
        report_lines.append(f"20 EMA:           ₹{self.hit.ema20:,.2f}")
        report_lines.append(f"50 EMA:           ₹{self.hit.ema50:,.2f}")
        report_lines.append(f"RSI(14):          {self.hit.rsi:.1f}")
        report_lines.append(f"Volume:           {self.hit.volume:,} ({self.hit.vol_vs_avg_pct:+.1f}% vs avg)")
        report_lines.append(f"ADX:              {self.hit.adx:.1f}" if self.hit.adx else "ADX:              —")
        report_lines.append("")
        
        report_lines.append("📈 SUPPORT & RESISTANCE\n" + "-"*80)
        report_lines.append(f"Support:          ₹{self.hit.support:,.2f}")
        report_lines.append(f"Resistance:       ₹{self.hit.resistance:,.2f}")
        report_lines.append(f"Pivot (S1/S2):    ₹{self.hit.s1:,.2f} / ₹{self.hit.s2:,.2f}")
        report_lines.append(f"Target (R1/R2):   ₹{self.hit.r1:,.2f} / ₹{self.hit.r2:,.2f}")
        report_lines.append("")
        
        report_lines.append("🎯 ENTRY & EXIT\n" + "-"*80)
        report_lines.append(f"Suggested Entry:  ₹{self.hit.s1:,.2f}–₹{self.hit.support:,.2f}")
        report_lines.append(f"Stop Loss (S1):   ₹{self.hit.s1:,.2f}")
        report_lines.append(f"Target 1 (R1):    ₹{self.hit.r1:,.2f}")
        report_lines.append(f"Target 2 (R2):    ₹{self.hit.r2:,.2f}")
        
        if self.hit.s1 and self.hit.r1:
            risk = self.hit.close - self.hit.s1
            reward = self.hit.r1 - self.hit.close
            if risk > 0:
                rr_ratio = reward / risk
                report_lines.append(f"Risk/Reward:      {rr_ratio:.1f}:1")
        report_lines.append("")
        
        report_lines.append("✅ QUALITY SIGNALS\n" + "-"*80)
        if self.hit.quality_flags:
            for flag in self.hit.quality_flags:
                report_lines.append(f"  • {flag}")
        else:
            report_lines.append("  (None)")
        report_lines.append("")
        
        report_lines.append("📋 DETAILED ASSESSMENT\n" + "-"*80)
        for verdict in self._get_all_verdicts():
            flag_marker = "🚩" if verdict.is_red_flag else "✓"
            report_lines.append(f"\n{flag_marker} {verdict.parameter}")
            report_lines.append(f"  Scanner:  {verdict.scanner_value}")
            report_lines.append(f"  Market:   {verdict.market_data_value}")
            report_lines.append(f"  Verdict:  {verdict.verdict}")
            report_lines.append(f"  Impact:   {verdict.impact}")
        
        report_lines.append("\n" + "="*80 + "\n")
        
        return "\n".join(report_lines)

    def generate_markdown_report(self) -> str:
        """Generate a markdown-formatted report suitable for Streamlit."""
        score_pct, score_level = self._calculate_quality_score_percentage()
        trade_verdict = self._generate_trade_verdict()
        
        md_lines = [
            f"## {self.hit.symbol} — Detailed Analysis",
            "",
            f"**Quality Score:** {self.hit.quality_score}/6 ({score_pct}%) {score_level}",
            "",
            f"### 🎯 Trade Decision",
            f"{trade_verdict}",
            "",
        ]
        
        md_lines.append("### 📊 Core Metrics")
        md_lines.append(
            f"| Metric | Value | Status |"
            f"\n|--------|-------|--------|"
            f"\n| Close/LTP | ₹{self.hit.close:,.2f} | — |"
            f"\n| EMA 20 | ₹{self.hit.ema20:,.2f} | {"✅" if self.hit.close > self.hit.ema20 else "❌"} |"
            f"\n| EMA 50 | ₹{self.hit.ema50:,.2f} | {"✅" if self.hit.ema20 > self.hit.ema50 else "❌"} |"
            f"\n| RSI(14) | {self.hit.rsi:.1f} | {"✅" if 50 <= self.hit.rsi <= 70 else "⚠️"} |"
            f"\n| Volume vs Avg | +{self.hit.vol_vs_avg_pct:.1f}% | {"✅" if self.hit.vol_vs_avg_pct > 0 else "❌"} |"
            f"\n| ADX | {self.hit.adx:.1f if self.hit.adx else "—"} | {"✅" if (self.hit.adx and self.hit.adx >= 20) else "⚠️"} |"
        )
        md_lines.append("")
        
        md_lines.append("### 📈 Support & Resistance")
        md_lines.append(
            f"| Level | Price |"
            f"\n|-------|-------|"
            f"\n| Support | ₹{self.hit.support:,.2f} |"
            f"\n| S1 (Stop) | ₹{self.hit.s1:,.2f} |"
            f"\n| Pivot | ₹{self.hit.pivot:,.2f} |"
            f"\n| Resistance | ₹{self.hit.resistance:,.2f} |"
            f"\n| R1 (Target 1) | ₹{self.hit.r1:,.2f} |"
            f"\n| R2 (Target 2) | ₹{self.hit.r2:,.2f} |"
        )
        md_lines.append("")
        
        md_lines.append("### 🎯 Entry & Exit Plan")
        entry_style = "Breakout" if self.hit.breakout_resistance else "Dip Buy"
        md_lines.append(f"**Style:** {entry_style}")
        md_lines.append(
            f"- **Suggested Entry:** ₹{self.hit.s1:,.2f}–₹{self.hit.support:,.2f}"
            f"\n- **Stop Loss:** ₹{self.hit.s1:,.2f}"
            f"\n- **Target 1:** ₹{self.hit.r1:,.2f}"
            f"\n- **Target 2:** ₹{self.hit.r2:,.2f}"
        )
        
        if self.hit.s1 and self.hit.r1:
            risk = self.hit.close - self.hit.s1
            reward = self.hit.r1 - self.hit.close
            if risk > 0:
                rr_ratio = reward / risk
                md_lines.append(f"- **Risk/Reward:** {rr_ratio:.1f}:1")
        md_lines.append("")
        
        md_lines.append("### ✅ Quality Signals")
        if self.hit.quality_flags:
            for flag in self.hit.quality_flags:
                md_lines.append(f"- {flag}")
        else:
            md_lines.append("- (None)")
        md_lines.append("")
        
        md_lines.append("### 📋 Detailed Assessment")
        for verdict in self._get_all_verdicts():
            flag_marker = "🚩" if verdict.is_red_flag else "✓"
            md_lines.append(f"\n**{flag_marker} {verdict.parameter}**")
            md_lines.append(f"- Scanner: {verdict.scanner_value}")
            md_lines.append(f"- Market: {verdict.market_data_value}")
            md_lines.append(f"- Verdict: {verdict.verdict}")
            md_lines.append(f"- Impact: {verdict.impact}")
        
        return "\n".join(md_lines)


def generate_batch_report(hits: list[ScanHit]) -> pd.DataFrame:
    """Generate summary analysis for all scanned stocks."""
    rows = []
    for hit in hits:
        report = StockAnalysisReport(hit)
        score_pct, score_level = report._calculate_quality_score_percentage()
        
        rows.append({
            "Symbol": hit.symbol,
            "Universe": hit.universe,
            "Close": f"₹{hit.close:,.2f}",
            "Quality Score": f"{hit.quality_score}/6",
            "Quality %": score_pct,
            "ADX": f"{hit.adx:.1f}" if hit.adx else "—",
            "RSI": f"{hit.rsi:.1f}",
            "Volume": f"+{hit.vol_vs_avg_pct:.1f}%",
            "Entry": f"₹{hit.s1:,.2f}–₹{hit.support:,.2f}",
            "Stop (S1)": f"₹{hit.s1:,.2f}",
            "Target (R1)": f"₹{hit.r1:,.2f}",
            "Target (R2)": f"₹{hit.r2:,.2f}",
            "Verdict": report._generate_trade_verdict(),
            "Quality Signals": ", ".join(hit.quality_flags) if hit.quality_flags else "—",
        })
    
    return pd.DataFrame(rows)
