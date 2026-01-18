"""
Day 11: Holt-Winters' Seasonal Method
======================================

Triple exponential smoothing for data with both trend and seasonality.

Key Concepts:
- Extends Holt's by adding a seasonal component
- Three smoothing parameters: alpha (level), beta (trend), gamma (seasonal)
- Two approaches: additive (constant seasonal magnitude) or multiplicative (seasonal grows with trend)
- Best for data with recurring patterns

Formula (Additive):
  ŷ(t+h) = ℓ(t) + h·b(t) + s(t-m+h)

Formula (Multiplicative):
  ŷ(t+h) = (ℓ(t) + h·b(t)) × s(t-m+h)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_gold_data_monthly():
    """Load and aggregate gold price data to monthly."""
    gold = yf.download('GLD', start='2015-01-01', end='2026-01-18', progress=False)
    gold['Price'] = (gold['Close'] * 10.8).round(0)
    gold_monthly = gold[['Price']].resample('MS').mean()
    return gold_monthly.squeeze()


def decompose_series(series):
    """Perform seasonal decomposition."""
    decomposition = seasonal_decompose(series, model='additive', period=12)
    return decomposition


def fit_holt_winters(train, seasonal_periods, seasonal_type):
    """
    Fit Holt-Winters model.
    
    Parameters:
    -----------
    train : Series
        Training data
    seasonal_periods : int
        Number of periods in seasonal cycle (12 for monthly, 4 for quarterly, etc.)
    seasonal_type : str
        'add' for additive, 'mul' for multiplicative
    
    Returns:
    --------
    fit : HoltWinters result object
    """
    model = ExponentialSmoothing(
        train,
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal=seasonal_type,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True)
    return fit


def evaluate_model(test, forecast, train):
    """Calculate forecast performance metrics."""
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    # Naive seasonal baseline (last year)
    naive_seasonal = train.iloc[-12:].values
    naive_forecast = np.tile(naive_seasonal, len(test)//12 + 1)[:len(test)]
    naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'naive_rmse': naive_rmse,
        'improvement': ((naive_rmse - rmse) / naive_rmse * 100)
    }


def print_header():
    """Print analysis header."""
    print("\n" + "=" * 70)
    print("HOLT-WINTERS' SEASONAL METHOD (TRIPLE EXPONENTIAL SMOOTHING)")
    print("=" * 70)


def print_decomposition(decomposition):
    """Print decomposition statistics."""
    print("\n" + "=" * 70)
    print("SEASONAL DECOMPOSITION ANALYSIS")
    print("=" * 70)
    
    print("\nOriginal Series:")
    print(f"  Mean: {decomposition.observed.mean():.2f}")
    print(f"  Std Dev: {decomposition.observed.std():.2f}")
    print(f"  Min: {decomposition.observed.min():.2f}")
    print(f"  Max: {decomposition.observed.max():.2f}")
    
    print("\nTrend Component:")
    print(f"  Mean: {decomposition.trend.mean():.2f}")
    print(f"  Initial: {decomposition.trend.iloc[0]:.2f}")
    print(f"  Final: {decomposition.trend.iloc[-1]:.2f}")
    print(f"  Direction: {'↑ Uptrend' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else '↓ Downtrend'}")
    
    print("\nSeasonal Component:")
    print(f"  Mean: {decomposition.seasonal.mean():.2f} (should be ~0)")
    print(f"  Std Dev: {decomposition.seasonal.std():.2f}")
    print(f"  Min: {decomposition.seasonal.min():.2f}")
    print(f"  Max: {decomposition.seasonal.max():.2f}")
    
    print("\nResidual Component:")
    print(f"  Mean: {decomposition.resid.mean():.2f}")
    print(f"  Std Dev: {decomposition.resid.std():.2f}")


def print_model_parameters(fit_add, fit_mul):
    """Print model parameters."""
    print("\n" + "=" * 70)
    print("OPTIMAL MODEL PARAMETERS")
    print("=" * 70)
    
    alpha_add = fit_add.params['smoothing_level']
    beta_add = fit_add.params['smoothing_trend']
    gamma_add = fit_add.params['smoothing_seasonal']
    
    alpha_mul = fit_mul.params['smoothing_level']
    beta_mul = fit_mul.params['smoothing_trend']
    gamma_mul = fit_mul.params['smoothing_seasonal']
    
    print(f"\nAdditive Model:")
    print(f"  α (level):   {alpha_add:.4f} - Weight on current observation")
    print(f"  β (trend):   {beta_add:.4f} - Weight on trend change")
    print(f"  γ (seasonal): {gamma_add:.4f} - Weight on seasonal change")
    
    print(f"\nMultiplicative Model:")
    print(f"  α (level):   {alpha_mul:.4f}")
    print(f"  β (trend):   {beta_mul:.4f}")
    print(f"  γ (seasonal): {gamma_mul:.4f}")
    
    return alpha_add, beta_add, gamma_add, alpha_mul, beta_mul, gamma_mul


def print_model_comparison(metrics_add, metrics_mul):
    """Compare model performance."""
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\nAdditive Holt-Winters:")
    print(f"  MAE:  {metrics_add['mae']:.4f}")
    print(f"  RMSE: {metrics_add['rmse']:.4f}")
    print(f"  MAPE: {metrics_add['mape']:.2f}%")
    
    print(f"\nMultiplicative Holt-Winters:")
    print(f"  MAE:  {metrics_mul['mae']:.4f}")
    print(f"  RMSE: {metrics_mul['rmse']:.4f}")
    print(f"  MAPE: {metrics_mul['mape']:.2f}%")
    
    print(f"\nNaive Seasonal Baseline:")
    print(f"  RMSE: {metrics_add['naive_rmse']:.4f}")
    
    if metrics_add['rmse'] < metrics_mul['rmse']:
        better = 'Additive'
        rmse_better = metrics_add['rmse']
        rmse_worse = metrics_mul['rmse']
    else:
        better = 'Multiplicative'
        rmse_better = metrics_mul['rmse']
        rmse_worse = metrics_add['rmse']
    
    print(f"\n✓ {better} model performs better")
    print(f"  Difference: {abs(rmse_worse - rmse_better):.4f} RMSE"
          f" ({abs((rmse_worse - rmse_better)/rmse_worse)*100:.1f}%)")


def print_when_to_use():
    """Print when to use each model."""
    print("\n" + "=" * 70)
    print("WHEN TO USE ADDITIVE VS MULTIPLICATIVE")
    print("=" * 70)
    
    print("\n✓ Use ADDITIVE when:")
    print("  - Seasonal swings are roughly constant over time")
    print("  - Example: Temperature (winter always ~20°F colder)")
    print("  - Formula: ŷ(t+h) = level + trend + seasonal")
    print("  - Seasonal component is independent of level")
    
    print("\n✓ Use MULTIPLICATIVE when:")
    print("  - Seasonal swings increase with trend")
    print("  - Example: Sales (busy season grows with company size)")
    print("  - Formula: ŷ(t+h) = (level + trend) × seasonal_factor")
    print("  - Seasonal component scales with level")
    
    print("\n✓ Quick Decision Rule:")
    print("  Plot decomposition → seasonal variance changes with trend → Multiplicative")
    print("  Plot decomposition → seasonal variance constant → Additive")


def print_three_components():
    """Explain three components."""
    print("\n" + "=" * 70)
    print("THREE COMPONENTS OF HOLT-WINTERS")
    print("=" * 70)
    
    print("\n1. LEVEL (ℓ)")
    print("   - Smoothed baseline value")
    print("   - Removes noise and seasonal effects")
    print("   - Similar to simple moving average")
    
    print("\n2. TREND (b)")
    print("   - Slope of the series")
    print("   - Can be positive (uptrend) or negative (downtrend)")
    print("   - Captured from recent changes")
    
    print("\n3. SEASONAL (s)")
    print("   - Repeating pattern with fixed period (12 for monthly)")
    print("   - Additive: Fixed offset each season")
    print("   - Multiplicative: Fixed multiplier each season")


def print_progression():
    """Print learning progression."""
    print("\n" + "=" * 70)
    print("EXPONENTIAL SMOOTHING PROGRESSION")
    print("=" * 70)
    
    print("\n  Day 9:  Simple Exponential Smoothing (SES)")
    print("          Component: Level only")
    print("          ŷ(t+h) = ℓ(t)")
    print("          Use: Stationary data\n")
    
    print("  Day 10: Holt's Linear Trend")
    print("          Components: Level + Trend")
    print("          ŷ(t+h) = ℓ(t) + h·b(t)")
    print("          Use: Data with trend, no seasonality\n")
    
    print("  Day 11: Holt-Winters' Seasonal ← YOU ARE HERE")
    print("          Components: Level + Trend + Seasonal")
    print("          ŷ(t+h) = ℓ(t) + h·b(t) + s(t)")
    print("          Use: Data with trend AND seasonality")


if __name__ == "__main__":
    # Load data
    gold_monthly = load_gold_data_monthly()
    
    # Train-test split
    train_size = int(len(gold_monthly) * 0.8)
    train = gold_monthly.iloc[:train_size]
    test = gold_monthly.iloc[train_size:]
    
    print_header()
    print(f"\nDataset: Monthly Gold Prices")
    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")
    
    # Decomposition
    decomp = decompose_series(gold_monthly)
    print_decomposition(decomp)
    
    # Fit both models
    fit_add = fit_holt_winters(train, seasonal_periods=12, seasonal_type='add')
    fit_mul = fit_holt_winters(train, seasonal_periods=12, seasonal_type='mul')
    
    # Print parameters
    alpha_add, beta_add, gamma_add, alpha_mul, beta_mul, gamma_mul = print_model_parameters(fit_add, fit_mul)
    
    # Generate forecasts
    forecast_add = fit_add.forecast(steps=len(test))
    forecast_mul = fit_mul.forecast(steps=len(test))
    print(f"\n✓ Generated {len(forecast_add)} forecasts for each model")
    
    # Evaluate
    metrics_add = evaluate_model(test, forecast_add, train)
    metrics_mul = evaluate_model(test, forecast_mul, train)
    print_model_comparison(metrics_add, metrics_mul)
    
    # Guidance
    print_when_to_use()
    print_three_components()
    print_progression()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete! View notebook for interactive visualizations.")
    print("=" * 70 + "\n")
