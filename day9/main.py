"""
Day 9: Simple Exponential Smoothing (SES)
==========================================

Forecasting stationary time series data without trend using Simple Exponential Smoothing.

Key Concepts:
- SES assigns exponentially decreasing weights to past observations
- Alpha parameter controls weight given to recent vs. historical data
- Best for stationary data (no trend, no seasonality)
- Provides simple, interpretable forecasts

Formula: y_hat(t+1) = α * y(t) + (1-α) * y_hat(t)

Data: Differenced gold prices (stationary series from Day 6)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_and_prepare_data():
    """Load gold price data and create differenced series."""
    gold = pd.read_csv("data/gold_prices.csv", parse_dates=["Date"], index_col="Date")
    gold['Price'] = (gold['Price'].astype(float) * 10.8).round(0)
    gold['Price_Diff'] = gold['Price'].diff()
    gold_diff = gold['Price_Diff'].dropna()
    return gold_diff


def manual_ses(data, alpha):
    """
    Manual Simple Exponential Smoothing implementation.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    alpha : float
        Smoothing parameter (0 < alpha < 1)
    
    Returns:
    --------
    fitted : array
        Fitted values (one-step ahead forecasts)
    """
    fitted = np.zeros(len(data))
    fitted[0] = data[0]  # Initialize with first observation
    
    for t in range(1, len(data)):
        fitted[t] = alpha * data[t-1] + (1 - alpha) * fitted[t-1]
    
    return fitted


def test_alpha_values(train):
    """Test different alpha values and return performance metrics."""
    alphas = np.arange(0.1, 1.0, 0.1)
    results = []
    
    for alpha in alphas:
        model = SimpleExpSmoothing(train)
        fit = model.fit(smoothing_level=alpha, optimized=False)
        fitted_values = fit.fittedvalues
        
        mse = mean_squared_error(train[1:], fitted_values[1:])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(train[1:], fitted_values[1:])
        
        results.append({
            'alpha': alpha,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        })
    
    return pd.DataFrame(results)


def evaluate_forecast(test, forecast, train):
    """Calculate comprehensive forecast evaluation metrics."""
    forecast_mse = mean_squared_error(test, forecast)
    forecast_rmse = np.sqrt(forecast_mse)
    forecast_mae = mean_absolute_error(test, forecast)
    
    # MAPE
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    # Naive forecast comparison
    naive_forecast = np.full(len(test), train.iloc[-1])
    naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
    
    improvement = ((naive_rmse - forecast_rmse) / naive_rmse * 100)
    
    return {
        'mse': forecast_mse,
        'rmse': forecast_rmse,
        'mae': forecast_mae,
        'mape': mape,
        'naive_rmse': naive_rmse,
        'improvement': improvement
    }


def print_header():
    """Print analysis header."""
    print("\n" + "=" * 70)
    print("SIMPLE EXPONENTIAL SMOOTHING (SES) ANALYSIS")
    print("=" * 70)


def print_data_overview(gold_diff, train, test):
    """Print dataset overview."""
    print(f"\n📊 Dataset Overview:")
    print(f"  Total observations: {len(gold_diff)}")
    print(f"  Training set: {len(train)} ({len(train)/len(gold_diff)*100:.1f}%)")
    print(f"  Test set: {len(test)} ({len(test)/len(gold_diff)*100:.1f}%)")
    print(f"\n  Date range: {gold_diff.index.min()} to {gold_diff.index.max()}")
    print(f"  Train period: {train.index.min()} to {train.index.max()}")
    print(f"  Test period: {test.index.min()} to {test.index.max()}")


def print_manual_ses_demo(train):
    """Demonstrate manual SES implementation."""
    print("\n" + "=" * 70)
    print("MANUAL SES IMPLEMENTATION DEMO")
    print("=" * 70)
    
    alpha_test = 0.3
    manual_fit = manual_ses(train.values, alpha_test)
    
    print(f"\nTesting manual SES with α = {alpha_test}")
    print(f"\nFormula: y_hat(t+1) = α * y(t) + (1-α) * y_hat(t)")
    print(f"Where α = {alpha_test}, giving {alpha_test*100}% weight to recent observation")
    print(f"\nFirst 5 fitted values: {manual_fit[:5]}")
    print(f"Last 5 fitted values: {manual_fit[-5:]}")


def print_alpha_tuning_results(results_df):
    """Print alpha parameter tuning results."""
    print("\n" + "=" * 70)
    print("ALPHA PARAMETER TUNING")
    print("=" * 70)
    
    print("\nTesting alpha values from 0.1 to 0.9:")
    print("\n" + results_df.to_string(index=False))
    
    best_alpha = results_df.loc[results_df['rmse'].idxmin(), 'alpha']
    best_rmse = results_df['rmse'].min()
    
    print(f"\n✓ Best alpha by RMSE: {best_alpha:.1f} (RMSE = {best_rmse:.4f})")


def print_optimal_model_results(fit_final):
    """Print results from optimized model."""
    print("\n" + "=" * 70)
    print("STATSMODELS OPTIMIZED SES MODEL")
    print("=" * 70)
    
    optimal_alpha = fit_final.params['smoothing_level']
    print(f"\n✓ Optimized α (smoothing level): {optimal_alpha:.4f}")
    print(f"\nInterpretation:")
    print(f"  - {optimal_alpha*100:.2f}% weight on most recent observation")
    print(f"  - {(1-optimal_alpha)*100:.2f}% weight on historical smoothed value")
    
    # Show weight decay
    print(f"\n  Weight Decay Pattern:")
    for i in range(5):
        weight = optimal_alpha * ((1-optimal_alpha) ** i)
        print(f"    {i} periods ago: {weight*100:.2f}%")


def print_forecast_results(test, forecast):
    """Print forecast generation results."""
    print("\n" + "=" * 70)
    print("FORECAST GENERATION")
    print("=" * 70)
    
    print(f"\nForecast horizon: {len(forecast)} steps")
    print(f"Forecast period: {forecast.index[0]} to {forecast.index[-1]}")
    print(f"\nFirst 5 forecasts:")
    for i in range(5):
        print(f"  {forecast.index[i]}: {forecast.iloc[i]:.2f} (Actual: {test.iloc[i]:.2f})")


def print_evaluation_metrics(metrics):
    """Print forecast evaluation metrics."""
    print("\n" + "=" * 70)
    print("FORECAST PERFORMANCE EVALUATION")
    print("=" * 70)
    
    print(f"\n📈 Error Metrics:")
    print(f"  Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    
    print(f"\n📊 Baseline Comparison:")
    print(f"  Naive forecast RMSE: {metrics['naive_rmse']:.4f}")
    print(f"  SES improvement: {metrics['improvement']:.2f}%")
    
    if metrics['improvement'] > 0:
        print(f"\n✓ SES performs {metrics['improvement']:.1f}% better than naive forecast")
    else:
        print(f"\n⚠ SES performs {abs(metrics['improvement']):.1f}% worse than naive forecast")


def print_residual_analysis(residuals):
    """Print residual analysis."""
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    
    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.4f} (should be ~0)")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.2f}")
    print(f"  Max: {residuals.max():.2f}")
    print(f"\nResidual Range: [{residuals.quantile(0.25):.2f}, {residuals.quantile(0.75):.2f}]")


def print_recommendations():
    """Print practical recommendations."""
    print("\n" + "=" * 70)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n✅ When to Use SES:")
    print("  • Data is stationary (no trend, no seasonality)")
    print("  • Short-term forecasting (1-10 steps ahead)")
    print("  • Need simple, interpretable model")
    print("  • Computational efficiency is important")
    
    print("\n❌ When NOT to Use SES:")
    print("  • Data has clear trend (use Holt's method)")
    print("  • Data has seasonality (use Holt-Winters)")
    print("  • Need long-term forecasts (forecasts flatten)")
    print("  • Multiple predictors available (use regression)")
    
    print("\n📊 Alpha Parameter Guidelines:")
    print("  • α → 1: Responsive to recent changes (volatile series)")
    print("  • α → 0: More smoothing (stable series)")
    print("  • Typical range: 0.1 to 0.3 for most applications")
    print("  • Use optimization to find optimal α")
    
    print("\n⚠️ Limitations:")
    print("  1. Flat forecasts: All future forecasts = same value")
    print("  2. No uncertainty: Base model lacks prediction intervals")
    print("  3. Stationarity required: Must transform non-stationary data")
    print("  4. Short horizon: Best for 1-10 steps ahead")


if __name__ == "__main__":
    # Load data
    gold_diff = load_and_prepare_data()
    
    # Train-test split
    train_size = int(len(gold_diff) * 0.8)
    train = gold_diff.iloc[:train_size]
    test = gold_diff.iloc[train_size:]
    
    # Print analysis
    print_header()
    print_data_overview(gold_diff, train, test)
    print_manual_ses_demo(train)
    
    # Alpha tuning
    results_df = test_alpha_values(train)
    print_alpha_tuning_results(results_df)
    
    # Fit optimized model
    model_final = SimpleExpSmoothing(train)
    fit_final = model_final.fit(optimized=True)
    print_optimal_model_results(fit_final)
    
    # Generate forecasts
    forecast = fit_final.forecast(steps=len(test))
    print_forecast_results(test, forecast)
    
    # Evaluate
    metrics = evaluate_forecast(test, forecast, train)
    print_evaluation_metrics(metrics)
    
    # Residual analysis
    residuals = test - forecast
    print_residual_analysis(residuals)
    
    # Recommendations
    print_recommendations()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete! View notebook for interactive visualizations.")
    print("=" * 70 + "\n")
