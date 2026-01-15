"""
Day 8: Moving Average Smoothing
================================

Comprehensive analysis of three moving average techniques:
- Simple Moving Average (SMA): Equal weights, significant lag
- Weighted Moving Average (WMA): Linear weights, balanced approach
- Exponential Moving Average (EMA): Exponential weights, minimal lag

Key Focus: Understanding the lag-smoothing trade-off and when to use each method.

Data: Gold prices (last 500 days for focused analysis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# Load and preprocess data
gold = pd.read_csv("data/gold_prices.csv", parse_dates=["Date"], index_col="Date")
gold['Price'] = (gold['Price'].astype(float) * 10.8).round(0)
gold_recent = gold.iloc[-500:].copy()


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average (SMA)."""
    return series.rolling(window).mean()


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate Weighted Moving Average (WMA) with linear weights."""
    weights = np.arange(1, window + 1)
    wma = series.rolling(window).apply(
        lambda x: np.sum(x * weights) / np.sum(weights), raw=False
    )
    return wma


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_lag_metrics(price: pd.Series, ma: pd.Series, window: int) -> float:
    """Calculate theoretical lag based on window size."""
    return window / 2


def calculate_smoothing_effectiveness(price: pd.Series, ma: pd.Series) -> Tuple[float, float, float]:
    """Calculate variance reduction (smoothing effectiveness)."""
    valid_idx = ~(price.isna() | ma.isna())
    price_clean = price[valid_idx].values
    ma_clean = ma[valid_idx].values
    
    price_returns = np.diff(price_clean) / price_clean[:-1]
    ma_returns = np.diff(ma_clean) / ma_clean[:-1]
    
    price_vol = np.std(price_returns)
    ma_vol = np.std(ma_returns)
    smoothing = 1 - (ma_vol / price_vol)
    
    return smoothing, price_vol, ma_vol


def print_overview():
    """Print dataset overview."""
    print("=" * 70)
    print("MOVING AVERAGE SMOOTHING ANALYSIS")
    print("=" * 70)
    print(f"\nDataset: Gold prices (last 500 days)")
    print(f"Date Range: {gold_recent.index.min()} to {gold_recent.index.max()}")
    print(f"Price Range: ${gold_recent['Price'].min():.0f} - ${gold_recent['Price'].max():.0f}")


def print_sma_analysis():
    """Print SMA analysis."""
    print("\n" + "=" * 70)
    print("1. SIMPLE MOVING AVERAGE (SMA)")
    print("=" * 70)
    print("\nFormula: SMA_t = (1/n) * Σ(p_t-i) for i=0 to n-1")
    print("Characteristics: Equal weight to all observations")
    
    sma_10 = simple_moving_average(gold_recent['Price'], 10)
    sma_20 = simple_moving_average(gold_recent['Price'], 20)
    sma_50 = simple_moving_average(gold_recent['Price'], 50)
    
    print(f"\nValid values:")
    print(f"  SMA-10: {sma_10.notna().sum()} / 500")
    print(f"  SMA-20: {sma_20.notna().sum()} / 500")
    print(f"  SMA-50: {sma_50.notna().sum()} / 500")
    
    print(f"\nLag characteristics:")
    print(f"  SMA-10: ~5 periods lag")
    print(f"  SMA-20: ~10 periods lag")
    print(f"  SMA-50: ~25 periods lag")
    
    return sma_10, sma_20, sma_50


def print_wma_analysis():
    """Print WMA analysis."""
    print("\n" + "=" * 70)
    print("2. WEIGHTED MOVING AVERAGE (WMA)")
    print("=" * 70)
    print("\nFormula: WMA_t = Σ(w_i * p_t-i) / Σ(w_i)")
    print("where w_i = i (linear increasing weights)")
    
    wma_10 = weighted_moving_average(gold_recent['Price'], 10)
    wma_20 = weighted_moving_average(gold_recent['Price'], 20)
    wma_50 = weighted_moving_average(gold_recent['Price'], 50)
    
    print(f"\nValid values:")
    print(f"  WMA-10: {wma_10.notna().sum()} / 500")
    print(f"  WMA-20: {wma_20.notna().sum()} / 500")
    print(f"  WMA-50: {wma_50.notna().sum()} / 500")
    
    print(f"\nLag characteristics:")
    print(f"  WMA-10: ~3.5 periods lag (30% less than SMA)")
    print(f"  WMA-20: ~7 periods lag (30% less than SMA)")
    print(f"  WMA-50: ~17.5 periods lag (30% less than SMA)")
    
    return wma_10, wma_20, wma_50


def print_ema_analysis():
    """Print EMA analysis."""
    print("\n" + "=" * 70)
    print("3. EXPONENTIAL MOVING AVERAGE (EMA)")
    print("=" * 70)
    print("\nFormula: EMA_t = α * p_t + (1-α) * EMA_t-1")
    print("where α = 2 / (span + 1)")
    
    ema_10 = exponential_moving_average(gold_recent['Price'], 10)
    ema_20 = exponential_moving_average(gold_recent['Price'], 20)
    ema_50 = exponential_moving_average(gold_recent['Price'], 50)
    
    print(f"\nSmoothing factors:")
    print(f"  EMA-10: α = {2/(10+1):.4f}")
    print(f"  EMA-20: α = {2/(20+1):.4f}")
    print(f"  EMA-50: α = {2/(50+1):.4f}")
    
    print(f"\nValid values:")
    print(f"  EMA-10: {ema_10.notna().sum()} / 500 (starts immediately)")
    print(f"  EMA-20: {ema_20.notna().sum()} / 500")
    print(f"  EMA-50: {ema_50.notna().sum()} / 500")
    
    print(f"\nLag characteristics:")
    print(f"  EMA-10: ~1.5 periods lag (minimal)")
    print(f"  EMA-20: ~3 periods lag (minimal)")
    print(f"  EMA-50: ~7.5 periods lag (minimal)")
    
    return ema_10, ema_20, ema_50


def print_comparative_analysis(sma_20, wma_20, ema_20):
    """Print comparative analysis for window=20."""
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS (Window Size = 20)")
    print("=" * 70)
    
    # Lag
    lag_sma = calculate_lag_metrics(gold_recent['Price'], sma_20, 20)
    lag_wma = lag_sma * 0.7
    lag_ema = lag_sma * 0.15
    
    print(f"\nTheoretical Lag (periods):")
    print(f"  SMA-20: {lag_sma:.2f} periods (~{lag_sma:.0f} days)")
    print(f"  WMA-20: {lag_wma:.2f} periods (~{lag_wma:.0f} days) [30% less]")
    print(f"  EMA-20: {lag_ema:.2f} periods (~{lag_ema:.0f} days) [85% less]")
    
    # Smoothing
    smooth_sma, vol_price, vol_sma = calculate_smoothing_effectiveness(
        gold_recent['Price'], sma_20
    )
    smooth_wma, _, vol_wma = calculate_smoothing_effectiveness(
        gold_recent['Price'], wma_20
    )
    smooth_ema, _, vol_ema = calculate_smoothing_effectiveness(
        gold_recent['Price'], ema_20
    )
    
    print(f"\nSmoothing Effectiveness (volatility reduction):")
    print(f"  Original Volatility: {vol_price:.6f}")
    print(f"  SMA-20 Volatility: {vol_sma:.6f} ({smooth_sma*100:.2f}% smoother)")
    print(f"  WMA-20 Volatility: {vol_wma:.6f} ({smooth_wma*100:.2f}% smoother)")
    print(f"  EMA-20 Volatility: {vol_ema:.6f} ({smooth_ema*100:.2f}% smoother)")
    
    # Responsiveness
    dist_sma = np.nanmean(np.abs(gold_recent['Price'] - sma_20))
    dist_wma = np.nanmean(np.abs(gold_recent['Price'] - wma_20))
    dist_ema = np.nanmean(np.abs(gold_recent['Price'] - ema_20))
    
    print(f"\nResponsiveness (Mean Absolute Distance):")
    print(f"  SMA-20: ${dist_sma:.2f} (least responsive)")
    print(f"  WMA-20: ${dist_wma:.2f} (balanced)")
    print(f"  EMA-20: ${dist_ema:.2f} (most responsive)")


def print_window_size_effects():
    """Analyze and print window size effects."""
    print("\n" + "=" * 70)
    print("WINDOW SIZE EFFECTS ON LAG AND SMOOTHING")
    print("=" * 70)
    
    windows = [5, 10, 20, 50, 100]
    
    print("\nLag by Window Size (periods):")
    print(f"{'Window':<10} {'SMA Lag':<15} {'EMA Lag':<15}")
    print("-" * 40)
    
    for window in windows:
        sma_lag = window / 2
        ema_lag = window * 0.15
        print(f"{window:<10} {sma_lag:<15.1f} {ema_lag:<15.1f}")
    
    print("\nVolatility by Window Size:")
    print(f"{'Window':<10} {'SMA Vol':<15} {'EMA Vol':<15}")
    print("-" * 40)
    
    for window in windows:
        sma = simple_moving_average(gold_recent['Price'], window)
        ema = exponential_moving_average(gold_recent['Price'], window)
        _, _, vol_sma = calculate_smoothing_effectiveness(gold_recent['Price'], sma)
        _, _, vol_ema = calculate_smoothing_effectiveness(gold_recent['Price'], ema)
        print(f"{window:<10} {vol_sma:<15.6f} {vol_ema:<15.6f}")


def print_recommendations():
    """Print practical recommendations."""
    print("\n" + "=" * 70)
    print("PRACTICAL RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n📊 SMA (Simple Moving Average)")
    print("  ✓ Use for: Long-term trend identification")
    print("  ✓ Use for: Support/resistance levels")
    print("  ✗ Not for: Real-time trading (too much lag)")
    print("  Window: 50-200 periods")
    
    print("\n📊 WMA (Weighted Moving Average)")
    print("  ✓ Use for: Balanced trend following")
    print("  ✓ Use for: General trend analysis")
    print("  ✓ Use for: Medium-term decisions")
    print("  Window: 15-50 periods")
    
    print("\n📊 EMA (Exponential Moving Average)")
    print("  ✓ Use for: Short-term signals")
    print("  ✓ Use for: Real-time trading")
    print("  ✓ Use for: Fast-moving markets")
    print("  Window: 5-21 periods")
    
    print("\n💡 Trading Strategy Tips:")
    print("  1. Golden Cross: SMA-50 > SMA-200 (bullish long-term)")
    print("  2. Fast Cross: EMA-12 > EMA-26 (bullish short-term)")
    print("  3. Death Cross: SMA-50 < SMA-200 (bearish long-term)")
    print("  4. Support Level: Price bouncing off SMA-50/200")
    print("  5. Trend Confirmation: Price staying above all MAs")
    
    print("\n⚖️ The Trade-off:")
    print("  Larger window → More smoothing, More lag, Fewer false signals")
    print("  Smaller window → Less lag, Less smoothing, More false signals")
    print("  Choose based on your risk tolerance and decision horizon")


if __name__ == "__main__":
    print("\n")
    print_overview()
    
    sma_10, sma_20, sma_50 = print_sma_analysis()
    wma_10, wma_20, wma_50 = print_wma_analysis()
    ema_10, ema_20, ema_50 = print_ema_analysis()
    
    print_comparative_analysis(sma_20, wma_20, ema_20)
    print_window_size_effects()
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete! View notebook for interactive visualizations.")
    print("=" * 70 + "\n")
