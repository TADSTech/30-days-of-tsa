#!/usr/bin/env python3
"""
Day 15: SARIMA Models - Seasonal ARIMA
========================================
Extend ARIMA with seasonal components for data with recurring yearly patterns.
Uses MONTHLY aggregated data for lightweight computation.

Progression:
- Day 12: AR(p)           → Autoregressive only
- Day 13: ARMA(p,q)       → AR + Moving Average
- Day 14: ARIMA(p,d,q)    → + Differencing (handles trends)
- Day 15: SARIMA(p,d,q)(P,D,Q)s → + Seasonal patterns ← YOU ARE HERE
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Time Series Analysis
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================================
# 1. LOADING AND PREPARING DATA (MONTHLY)
# ============================================================================
print("\n" + "="*70)
print("DAY 15: SARIMA MODELS - SEASONAL ARIMA")
print("="*70)
print("\n" + "="*70)
print("1. LOADING AND PREPARING MONTHLY GOLD PRICE DATA")
print("="*70)

import os
try:
    df = pd.read_csv('data/gold_prices.csv', parse_dates=['Date'])
    print("✓ Using cached daily data")
except:
    try:
        df = pd.read_csv('day15/data/gold_prices.csv', parse_dates=['Date'])
        print("✓ Using cached daily data")
    except:
        print("⚠ No cached data found. Using synthetic example.")
        # Create synthetic data if needed
        dates = pd.date_range('2015-01-01', periods=2781, freq='D')
        np.random.seed(42)
        prices = 1500 + np.cumsum(np.random.normal(0.5, 30, 2781))
        df = pd.DataFrame({'Date': dates, 'Price': prices})

# Ensure formatting
if 'Price' not in df.columns:
    df = df.rename(columns={'Adj Close': 'Price'})
df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

print(f"Daily data shape: {df.shape}")

# Aggregate to MONTHLY for lightweight SARIMA
df['YearMonth'] = df['Date'].dt.to_period('M')
df_monthly = df.groupby('YearMonth')['Price'].mean().reset_index()
df_monthly['Date'] = df_monthly['YearMonth'].dt.to_timestamp()
df_monthly = df_monthly[['Date', 'Price']].reset_index(drop=True)

print(f"Monthly data shape: {df_monthly.shape}")
print(f"Date range: {df_monthly['Date'].min()} to {df_monthly['Date'].max()}")
print("Price summary:")
print(df_monthly['Price'].describe())

# ============================================================================
# 2. SEASONAL DECOMPOSITION
# ============================================================================
print("\n" + "="*70)
print("2. SEASONAL DECOMPOSITION ANALYSIS")
print("="*70)

# Need at least 24 months (2 years) for decomposition
if len(df_monthly) >= 24:
    decomposition = seasonal_decompose(df_monthly['Price'], model='additive', period=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    print(f"\nDecomposition Components:")
    print(f"  Trend: Mean={trend.dropna().mean():.2f}, Std={trend.dropna().std():.2f}")
    print(f"  Seasonal: Mean={seasonal.dropna().mean():.2f}, Std={seasonal.dropna().std():.2f}")
    print(f"  Residual: Mean={residual.dropna().mean():.2f}, Std={residual.dropna().std():.2f}")

    # Seasonal strength
    seasonal_var = seasonal.dropna().var()
    residual_var = residual.dropna().var()
    seasonal_strength = 1 - (residual_var / (seasonal_var + residual_var))
    print(f"\nSeasonal Strength: {seasonal_strength:.4f}")
    print(f"  Interpretation: {seasonal_strength*100:.1f}% of variation is seasonal")
    if seasonal_strength > 0.1:
        print("  ✓ Strong seasonal pattern → SARIMA appropriate")
    else:
        print("  ⚠ Weak seasonal pattern → ARIMA may suffice")
else:
    print("⚠ Less than 2 years of data, skipping decomposition")

# ============================================================================
# 3. DIFFERENCING FOR STATIONARITY
# ============================================================================
print("\n" + "="*70)
print("3. DIFFERENCING ANALYSIS")
print("="*70)

price_monthly = df_monthly['Price'].values

# ADF test helper
def adf_test(series, name):
    result = adfuller(series, autolag='AIC')
    p_val = result[1]
    is_stat = p_val <= 0.05
    print(f"\n{name}: ADF p-value={p_val:.6f} → {'✓ STATIONARY' if is_stat else '✗ NON-STAT'}")
    return is_stat, p_val

# Test original
stat_orig, p_orig = adf_test(price_monthly, "Original Series")

# First difference
diff_d1 = np.diff(price_monthly, n=1)
stat_d1, p_d1 = adf_test(diff_d1, "First Difference (d=1)")

# Seasonal difference (12 months)
diff_seasonal = np.diff(price_monthly, n=12)
stat_seasonal, p_seasonal = adf_test(diff_seasonal, "Seasonal Diff (D=1, 12m)")

print(f"\n✓ Recommendation: d=1, D={'1' if p_seasonal <= 0.05 else '0'} (monthly data)")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("4. TRAIN-TEST SPLIT")
print("="*70)

train_size = int(len(df_monthly) * 0.8)
train_data = df_monthly[:train_size].copy()
test_data = df_monthly[train_size:].copy()

print(f"Training set: {len(train_data)} months ({len(train_data)/len(df_monthly)*100:.1f}%)")
print(f"Test set: {len(test_data)} months ({len(test_data)/len(df_monthly)*100:.1f}%)")
print(f"Train: {train_data['Date'].min().date()} to {train_data['Date'].max().date()}")
print(f"Test: {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")

# ============================================================================
# 5. FITTING SARIMA MODELS (LIGHTWEIGHT)
# ============================================================================
print("\n" + "="*70)
print("5. FITTING SARIMA MODELS")
print("="*70)

# Use ONLY 3 simple SARIMA configs to avoid crashes
# With monthly data (100 observations), these will fit quickly
sarima_configs = [
    ((1, 1, 0), (0, 0, 0, 12)),  # Non-seasonal ARIMA
    ((0, 1, 1), (1, 0, 0, 12)),  # Seasonal AR
    ((0, 1, 1), (0, 1, 0, 12)),  # Seasonal differencing
]

print(f"\nFitting 3 SARIMA models on {len(train_data)} months of data...")
print(f"{'Model':<30} {'AIC':<12} {'BIC':<12} {'RMSE':<10}")
print("-" * 64)

results = []
fitted_models = {}

for order, seasonal_order in sarima_configs:
    try:
        model = SARIMAX(train_data['Price'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fitted_model = model.fit(disp=False, maxiter=50)
        
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        forecast = fitted_model.get_forecast(steps=len(test_data))
        forecast_values = forecast.predicted_mean.values
        
        rmse = np.sqrt(mean_squared_error(test_data['Price'], forecast_values))
        
        results.append({
            'Model': f"SARIMA{order}{seasonal_order}",
            'AIC': aic,
            'BIC': bic,
            'RMSE': rmse,
            'Forecast': forecast_values,
            'FittedModel': fitted_model
        })
        
        model_str = f"ARIMA{order}{seasonal_order}"
        print(f"{model_str:<30} {aic:<12.2f} {bic:<12.2f} {rmse:<10.2f}")
        
    except Exception as e:
        model_str = f"ARIMA{order}{seasonal_order}"
        print(f"{model_str:<30} Failed: {str(e)[:40]}")

if not results:
    print("\n⚠ All models failed. Check data format.")
else:
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['BIC'].idxmin()
    print(f"\n✓ Best Model (BIC): {results_df.loc[optimal_idx, 'Model']}")
    print(f"  BIC: {results_df.loc[optimal_idx, 'BIC']:.2f}")
    optimal_model = results_df.loc[optimal_idx, 'FittedModel']
    optimal_forecast = results_df.loc[optimal_idx, 'Forecast']

# ============================================================================
# 6. MODEL SUMMARY AND DIAGNOSTICS
# ============================================================================
if optimal_model is not None:
    print("\n" + "="*70)
    print("6. OPTIMAL SARIMA MODEL SUMMARY")
    print("="*70)
    
    rmse = np.sqrt(mean_squared_error(test_data['Price'], optimal_forecast))
    mae = mean_absolute_error(test_data['Price'], optimal_forecast)
    mape = np.mean(np.abs((test_data['Price'] - optimal_forecast) / test_data['Price'])) * 100
    
    naive_forecast = np.full(len(test_data), train_data['Price'].iloc[-1])
    naive_rmse = np.sqrt(mean_squared_error(test_data['Price'], naive_forecast))
    improvement = (naive_rmse - rmse) / naive_rmse * 100
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"\nNaive Baseline RMSE: {naive_rmse:.2f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    print(f"\nResidual Analysis:")
    residuals = optimal_model.resid
    print(f"  Mean: {residuals.mean():.4f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.2f}, Max: {residuals.max():.2f}")
    
    print(f"\nModel Summary:")
    print(optimal_model.summary())

# ============================================================================
# 7. KEY INSIGHTS: SARIMA VS ARIMA
# ============================================================================
print("\n" + "="*70)
print("7. KEY INSIGHTS: SARIMA MODELS")
print("="*70)

print("\nSARIMA(p,d,q)(P,D,Q)s vs ARIMA(p,d,q):")
print("  SARIMA adds 3 seasonal components:")
print("    P = Seasonal AR order")
print("    D = Seasonal differencing")
print("    Q = Seasonal MA order")
print("    s = Seasonal period (12 for monthly)")

print("\nWhy Use SARIMA:")
print("  ✓ Handles seasonal patterns automatically")
print("  ✓ Separates seasonal from trend components")
print("  ✓ Better forecasts for seasonal data")
print("  ✓ Gold prices have weak seasonality")

print("\nWhen to Use SARIMA:")
print("  ✓ Retail sales (holiday peaks)")
print("  ✓ Weather (seasonal cycles)")
print("  ✓ Tourism (summer/winter patterns)")
print("  ✗ Non-seasonal data (use ARIMA)")

print("\nModel Family Progression:")
print("  Day 12: AR(p)               - Past values only")
print("  Day 13: ARMA(p,q)           - AR + MA components")
print("  Day 14: ARIMA(p,d,q)        - + Differencing for trends")
print("  Day 15: SARIMA(p,d,q)(P,D,Q)s - + Seasonal patterns")

print("\n" + "="*70)
print("✓ Day 15 SARIMA Analysis Complete!")
print("="*70 + "\n")
