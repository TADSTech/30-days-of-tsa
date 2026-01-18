"""
Day 10: Holt's Linear Trend Method
===================================

Double exponential smoothing for data with trend but no seasonality.

Key Concepts:
- Extends SES by adding a trend component
- Two smoothing parameters: alpha (level) and beta (trend)
- Produces trending forecasts instead of flat forecasts
- Best for data with clear upward/downward movement

Formula: 
  ŷ(t+1) = ℓ(t) + b(t)
  where ℓ(t) = α·y(t) + (1-α)·(ℓ(t-1) + b(t-1))
        b(t) = β·(ℓ(t) - ℓ(t-1)) + (1-β)·b(t-1)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_gold_data():
    """Load and prepare gold price data."""
    gold = yf.download('GLD', start='2015-01-01', end='2026-01-18', progress=False)
    gold['Price'] = (gold['Close'] * 10.8).round(0)
    gold['Price_Diff'] = gold[['Price']].diff()
    gold_diff = gold['Price_Diff'].dropna()
    return gold_diff


def manual_holts(data, alpha, beta):
    """
    Manual Holt's Linear Trend implementation.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    alpha : float
        Level smoothing parameter (0 < alpha < 1)
    beta : float
        Trend smoothing parameter (0 < beta < 1)
    
    Returns:
    --------
    fitted : array
        Fitted values (one-step ahead forecasts)
    level : array
        Level component
    trend : array
        Trend component
    """
    n = len(data)
    fitted = np.zeros(n)
    level = np.zeros(n)
    trend = np.zeros(n)
    
    # Initialize
    level[0] = data[0]
    trend[0] = data[1] - data[0]
    fitted[0] = level[0] + trend[0]
    
    # Recursive update
    for t in range(1, n):
        level[t] = alpha * data[t] + (1 - alpha) * (level[t-1] + trend[t-1])
        trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
        fitted[t] = level[t] + trend[t]
    
    return fitted, level, trend


def test_alpha_beta_grid(train, alphas, betas):
    """Grid search for optimal alpha and beta parameters."""
    results = []
    
    for alpha in alphas:
        for beta in betas:
            try:
                model = Holt(train)
                fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
                fitted_values = fit.fittedvalues
                
                mse = mean_squared_error(train[1:], fitted_values[1:])
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(train[1:], fitted_values[1:])
                
                results.append({
                    'alpha': alpha,
                    'beta': beta,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                })
            except:
                pass
    
    return pd.DataFrame(results)


def evaluate_forecast(test, forecast, train):
    """Calculate comprehensive forecast evaluation metrics."""
    # Convert forecast to Series with same index as test for proper alignment
    if isinstance(forecast, np.ndarray):
        forecast = pd.Series(forecast, index=test.index)
    
    mse = mean_squared_error(test, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast)
    
    # Avoid division by zero in MAPE calculation
    mask = test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((test[mask] - forecast[mask]) / test[mask])) * 100
    else:
        mape = np.nan
    
    # Naive baseline
    naive_forecast = np.full(len(test), train.iloc[-1])
    naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
    improvement = ((naive_rmse - rmse) / naive_rmse * 100)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'naive_rmse': naive_rmse,
        'improvement': improvement
    }


def print_header():
    """Print analysis header."""
    print("\n" + "=" * 70)
    print("HOLT'S LINEAR TREND METHOD (DOUBLE EXPONENTIAL SMOOTHING)")
    print("=" * 70)


def print_manual_demo(train):
    """Demonstrate manual Holt's implementation."""
    print("\n" + "=" * 70)
    print("MANUAL HOLT'S IMPLEMENTATION DEMO")
    print("=" * 70)
    
    alpha, beta = 0.3, 0.1
    fitted, level, trend = manual_holts(train.values, alpha, beta)
    
    print(f"\nParameters: α = {alpha}, β = {beta}")
    print(f"\nFormulas:")
    print(f"  Level: ℓ(t) = α·y(t) + (1-α)·(ℓ(t-1) + b(t-1))")
    print(f"  Trend: b(t) = β·(ℓ(t) - ℓ(t-1)) + (1-β)·b(t-1)")
    print(f"  Fitted: ŷ(t) = ℓ(t) + b(t)")
    print(f"\nFirst 5 fitted values: {fitted[:5]}")
    print(f"Last 5 fitted values: {fitted[-5:]}")
    print(f"\nFinal level: {level[-1]:.4f}")
    print(f"Final trend: {trend[-1]:.6f} (slope of series)")


def print_optimization_results(results_df):
    """Print grid search results."""
    print("\n" + "=" * 70)
    print("ALPHA-BETA PARAMETER GRID SEARCH")
    print("=" * 70)
    
    print("\nTop 10 parameter combinations by RMSE:")
    print(results_df.nsmallest(10, 'rmse')[['alpha', 'beta', 'mse', 'rmse', 'mae']].to_string(index=False))
    
    best_idx = results_df['rmse'].idxmin()
    best = results_df.loc[best_idx]
    
    print(f"\n✓ Best parameters by RMSE:")
    print(f"  Alpha: {best['alpha']:.2f}")
    print(f"  Beta: {best['beta']:.2f}")
    print(f"  RMSE: {best['rmse']:.4f}")


def print_optimized_model(fit):
    """Print statsmodels optimized model results."""
    print("\n" + "=" * 70)
    print("STATSMODELS OPTIMIZED HOLT'S MODEL")
    print("=" * 70)
    
    alpha = fit.params['smoothing_level']
    beta = fit.params['smoothing_trend']
    
    print(f"\nOptimized Parameters:")
    print(f"  Alpha (level smoothing): {alpha:.4f}")
    print(f"  Beta (trend smoothing): {beta:.4f}")
    
    print(f"\nInterpretation:")
    print(f"  - Level: {alpha*100:.1f}% weight on current observation")
    print(f"  - Trend: {beta*100:.1f}% weight on recent trend change")
    
    level_final = fit.level.iloc[-1]
    trend_final = fit.trend.iloc[-1]
    
    print(f"\nFinal Components:")
    print(f"  Level: {level_final:.4f}")
    print(f"  Trend: {trend_final:.6f}")
    print(f"\nForecast formula: ŷ(t+h) = {level_final:.4f} + h × {trend_final:.6f}")


def print_forecast_evaluation(test, forecast, train):
    """Print forecast performance metrics."""
    metrics = evaluate_forecast(test, forecast, train)
    
    print("\n" + "=" * 70)
    print("FORECAST PERFORMANCE EVALUATION")
    print("=" * 70)
    
    print(f"\nError Metrics:")
    print(f"  Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    
    print(f"\nBaseline Comparison:")
    print(f"  Naive forecast RMSE: {metrics['naive_rmse']:.4f}")
    print(f"  Holt's improvement: {metrics['improvement']:.2f}%")
    
    if metrics['improvement'] > 0:
        print(f"\n✓ Holt's performs {metrics['improvement']:.1f}% better than naive forecast")
    else:
        print(f"\n⚠ Holt's performs {abs(metrics['improvement']):.1f}% worse than naive forecast")
    
    return metrics


def print_key_insights(fit, metrics):
    """Print key insights."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    print("\n✓ How Holt's differs from SES:")
    print("  SES: Flat forecasts (all h-step ahead = same value)")
    print("  Holt's: Trending forecasts (continue slope into future)")
    
    print("\n✓ Parameter Meanings:")
    alpha = fit.params['smoothing_level']
    beta = fit.params['smoothing_trend']
    print(f"  α = {alpha:.4f}: How much to weight current observation")
    print(f"  β = {beta:.4f}: How much to weight current trend change")
    
    print("\n✓ When to Use Holt's:")
    print("  ✓ Data has clear upward or downward trend")
    print("  ✓ No seasonal patterns")
    print("  ✓ Need forecasts that continue the trend")
    print("  ✗ Data is stationary (use SES)")
    print("  ✗ Data has seasonality (use Holt-Winters)")
    
    print("\n✓ Limitations:")
    print("  1. Assumes trend continues (may not hold in reality)")
    print("  2. Cannot handle seasonality")
    print("  3. No uncertainty intervals in base model")
    print("  4. Requires careful parameter selection")


def print_progression():
    """Print learning progression."""
    print("\n" + "=" * 70)
    print("EXPONENTIAL SMOOTHING PROGRESSION")
    print("=" * 70)
    
    print("\n  Day 9:  Simple Exponential Smoothing (SES)")
    print("          Level component only")
    print("          Formula: ŷ(t+h) = ℓ(t)")
    print("          → Flat forecasts")
    print("\n  Day 10: Holt's Linear Trend ← YOU ARE HERE")
    print("          Level + Trend components")
    print("          Formula: ŷ(t+h) = ℓ(t) + h·b(t)")
    print("          → Trending forecasts")
    print("\n  Day 11: Holt-Winters' Seasonal")
    print("          Level + Trend + Seasonal components")
    print("          Formula: ŷ(t+h) = (ℓ(t) + h·b(t)) × s(t)")
    print("          → Trending + seasonal forecasts")


if __name__ == "__main__":
    # Load data
    gold_diff = load_gold_data()
    
    # Train-test split
    train_size = int(len(gold_diff) * 0.8)
    train = gold_diff.iloc[:train_size]
    test = gold_diff.iloc[train_size:]
    
    print_header()
    print(f"\nDataset: Differenced Gold Prices")
    print(f"Training set: {len(train)} observations")
    print(f"Test set: {len(test)} observations")
    
    # Manual implementation demo
    print_manual_demo(train)
    
    # Grid search
    alphas = np.arange(0.1, 1.0, 0.2)
    betas = np.arange(0.01, 0.4, 0.05)
    results_df = test_alpha_beta_grid(train, alphas, betas)
    print_optimization_results(results_df)
    
    # Fit optimized model
    model = Holt(train)
    fit = model.fit(optimized=True)
    print_optimized_model(fit)
    
    # Generate forecasts
    forecast = fit.forecast(steps=len(test))
    # Convert to Series with proper index
    forecast = pd.Series(forecast.values, index=test.index)
    print(f"\n✓ Generated {len(forecast)} forecasts")
    
    # Evaluate
    metrics = print_forecast_evaluation(test, forecast, train)
    
    # Insights
    print_key_insights(fit, metrics)
    print_progression()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete! View notebook for interactive visualizations.")
    print("=" * 70 + "\n")
