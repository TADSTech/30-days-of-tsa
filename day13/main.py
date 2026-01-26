#!/usr/bin/env python3
"""
Day 13: ARMA Models
Combining AR and MA components for time series modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("✓ Libraries imported successfully")

# ============================================================================
# 1. Load and Prepare Data
# ============================================================================
print("\n" + "="*70)
print("1. LOADING AND PREPARING GOLD PRICE DATA")
print("="*70)

# Fetch gold price data
gold = yf.download('GLD', start='2015-01-01', end='2026-01-25', progress=False)
gold['Price'] = (gold['Close'] * 10.8).round(0)
gold = gold[['Price']].copy()

print(f"Gold data shape: {gold.shape}")
print(f"Date range: {gold.index.min()} to {gold.index.max()}")
print(f"\nData summary:")
print(gold['Price'].describe())

# First difference for stationarity
gold_diff = gold['Price'].diff().dropna()

# Test stationarity
adf_result = adfuller(gold_diff, autolag='AIC')
print(f"\nADF Test on Differenced Series:")
print(f"  Test Statistic: {adf_result[0]:.6f}")
print(f"  P-value: {adf_result[1]:.6f}")
print(f"  Result: {'STATIONARY' if adf_result[1] < 0.05 else 'NON-STATIONARY'}")

# ============================================================================
# 2. ACF and PACF Analysis
# ============================================================================
print("\n" + "="*70)
print("2. ACF AND PACF ANALYSIS FOR ARMA IDENTIFICATION")
print("="*70)

# Calculate ACF and PACF
acf_vals = acf(gold_diff, nlags=40)
pacf_vals = pacf(gold_diff, nlags=40, method='ywm')

# Confidence intervals
ci = 1.96 / np.sqrt(len(gold_diff))

print(f"\nACF Analysis:")
print(f"  Lag 1-5: {acf_vals[1:6]}")
print(f"  95% CI: ±{ci:.4f}")

print(f"\nPACF Analysis:")
print(f"  Lag 1-5: {pacf_vals[1:6]}")
print(f"  95% CI: ±{ci:.4f}")

# Identify significant lags
sig_acf = [i for i in range(1, 21) if abs(acf_vals[i]) > ci]
sig_pacf = [i for i in range(1, 21) if abs(pacf_vals[i]) > ci]

print(f"\nSignificant ACF lags (1-20): {sig_acf[:10]}")
print(f"Significant PACF lags (1-20): {sig_pacf[:10]}")

# Suggest ARMA orders
print(f"\nSuggested Orders:")
print(f"  AR order (p): {sig_pacf[0] if sig_pacf else 0} (from PACF cutoff)")
print(f"  MA order (q): {sig_acf[0] if sig_acf else 0} (from ACF cutoff)")

# ============================================================================
# 3. Train-Test Split
# ============================================================================
print("\n" + "="*70)
print("3. TRAIN-TEST SPLIT")
print("="*70)

train_size = int(len(gold_diff) * 0.8)
train = gold_diff.iloc[:train_size]
test = gold_diff.iloc[train_size:]

print(f"Training set: {len(train)} observations ({len(train)/len(gold_diff)*100:.1f}%)")
print(f"Test set: {len(test)} observations ({len(test)/len(gold_diff)*100:.1f}%)")
print(f"Train period: {train.index.min()} to {train.index.max()}")
print(f"Test period: {test.index.min()} to {test.index.max()}")

# ============================================================================
# 4. Fit ARMA Models with Different Orders
# ============================================================================
print("\n" + "="*70)
print("4. FITTING ARMA MODELS WITH DIFFERENT ORDERS")
print("="*70)

# Define ARMA orders to test (reduced for faster execution)
arma_orders = [
    (1, 0),  # AR(1)
    (2, 0),  # AR(2)
    (0, 1),  # MA(1)
    (0, 2),  # MA(2)
    (1, 1),  # ARMA(1,1)
    (2, 1),  # ARMA(2,1)
]

models = {}
results_table = []

print(f"\n{'Model':<12} {'AIC':<15} {'BIC':<15} {'RMSE':<15}")
print("-" * 57)

for p, q in arma_orders:
    try:
        # Fit ARIMA(p,0,q) which is ARMA(p,q) on differenced data
        model = ARIMA(train, order=(p, 0, q), trend='c')
        fit = model.fit()
        
        # Generate forecast
        forecast = fit.forecast(steps=len(test))
        forecast_series = pd.Series(forecast.values, index=test.index)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test, forecast_series))
        
        models[(p, q)] = fit
        results_table.append({
            'order': (p, q),
            'aic': fit.aic,
            'bic': fit.bic,
            'rmse': rmse,
            'forecast': forecast_series
        })
        
        print(f"ARMA({p},{q}){' '*(6-len(str((p,q))))} {fit.aic:<15.2f} {fit.bic:<15.2f} {rmse:<15.6f}")
    except Exception as e:
        print(f"ARMA({p},{q}){' '*(6-len(str((p,q))))} Failed: {str(e)[:30]}")

# Find best model by BIC
results_df = pd.DataFrame(results_table)
best_idx = results_df['bic'].idxmin()
best_order = results_df.loc[best_idx, 'order']
best_model = models[best_order]

print(f"\nOptimal Model: ARMA{best_order} (by BIC)")
print(f"  AIC: {best_model.aic:.2f}")
print(f"  BIC: {best_model.bic:.2f}")
print(f"  RMSE: {results_df.loc[best_idx, 'rmse']:.6f}")

# ============================================================================
# 5. Detailed Analysis of Optimal Model
# ============================================================================
print("\n" + "="*70)
print(f"5. DETAILED ANALYSIS OF ARMA{best_order}")
print("="*70)

print(f"\nModel Summary:")
print(best_model.summary())

# ============================================================================
# 6. Extract and Interpret Parameters
# ============================================================================
print("\n" + "="*70)
print("6. PARAMETER EXTRACTION AND INTERPRETATION")
print("="*70)

params = best_model.params
p, q = best_order

print(f"\nARMA{best_order} Parameters:")
print(f"  Constant: {params.iloc[0]:.6f}")

# AR parameters
if p > 0:
    print(f"\n  AR Coefficients (φ):")
    for i in range(1, p + 1):
        param_name = f'ar.L{i}'
        if param_name in params.index:
            print(f"    φ_{i}: {params[param_name]:.6f}")

# MA parameters
if q > 0:
    print(f"\n  MA Coefficients (θ):")
    for i in range(1, q + 1):
        param_name = f'ma.L{i}'
        if param_name in params.index:
            print(f"    θ_{i}: {params[param_name]:.6f}")

print(f"\nInterpretation:")
print(f"  AR component: Models dependency on past values")
print(f"  MA component: Models dependency on past forecast errors")
print(f"  Formula: y(t) = c + Σφᵢy(t-i) + εₜ + Σθⱼε(t-j)")

# ============================================================================
# 7. Generate Forecasts and Evaluate
# ============================================================================
print("\n" + "="*70)
print("7. FORECAST GENERATION AND EVALUATION")
print("="*70)

best_forecast = results_df.loc[best_idx, 'forecast']

# Compare with pure AR and pure MA models
ar_forecast = results_df[results_df['order'] == (2, 0)]['forecast'].values[0] if (2, 0) in [r['order'] for r in results_table] else None
ma_forecast = results_df[results_df['order'] == (0, 2)]['forecast'].values[0] if (0, 2) in [r['order'] for r in results_table] else None

print(f"\nForecast Statistics:")
print(f"  ARMA{best_order} - Mean: {best_forecast.mean():.4f}, Std: {best_forecast.std():.4f}")
print(f"  First 5 forecasts: {best_forecast.iloc[:5].values}")
print(f"  Last 5 forecasts: {best_forecast.iloc[-5:].values}")

# Calculate metrics
def evaluate_forecast(test, forecast, model_name):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    # Naive forecast
    naive = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    naive_rmse = np.sqrt(mean_squared_error(test, naive))
    improvement = ((naive_rmse - rmse) / naive_rmse * 100)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Naive RMSE: {naive_rmse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'improvement': improvement}

print("="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

metrics_best = evaluate_forecast(test, best_forecast, f"ARMA{best_order}")

if ar_forecast is not None:
    metrics_ar = evaluate_forecast(test, ar_forecast, "AR(2)")
    
if ma_forecast is not None:
    metrics_ma = evaluate_forecast(test, ma_forecast, "MA(2)")

# ============================================================================
# 8. Residual Analysis
# ============================================================================
print("\n" + "="*70)
print("8. RESIDUAL ANALYSIS")
print("="*70)

# Get residuals from best model
residuals_insample = best_model.resid

print(f"\nIn-Sample Residuals (Training Data):")
print(f"  Mean: {residuals_insample.mean():.6f} (should be ~0)")
print(f"  Std Dev: {residuals_insample.std():.6f}")
print(f"  Min: {residuals_insample.min():.6f}")
print(f"  Max: {residuals_insample.max():.6f}")

# Out-of-sample residuals
residuals_outofsample = test - best_forecast

print(f"\nOut-of-Sample Residuals (Test Data):")
print(f"  Mean: {residuals_outofsample.mean():.6f}")
print(f"  Std Dev: {residuals_outofsample.std():.6f}")
print(f"  Min: {residuals_outofsample.min():.6f}")
print(f"  Max: {residuals_outofsample.max():.6f}")

# Test for autocorrelation in residuals
residuals_acf = acf(residuals_outofsample.dropna(), nlags=20)
sig_residuals = [i for i in range(1, 21) if abs(residuals_acf[i]) > ci]
print(f"\nResidual ACF - Significant lags: {sig_residuals}")
if len(sig_residuals) == 0:
    print("  ✓ No significant autocorrelation (good!)")
else:
    print(f"  ⚠ Some autocorrelation remains at lags {sig_residuals[:5]}")

# ============================================================================
# 9. Model Comparison Summary
# ============================================================================
print("\n" + "="*70)
print("9. MODEL COMPARISON SUMMARY")
print("="*70)

print(f"\n{'Model':<12} {'AIC':<15} {'BIC':<15} {'RMSE':<15} {'Improvement':<12}")
print("-" * 69)

for result in results_table:
    order = result['order']
    aic = result['aic']
    bic = result['bic']
    rmse = result['rmse']
    
    # Calculate improvement
    naive_rmse = metrics_best['rmse'] / (1 - metrics_best['improvement']/100)
    improvement = ((naive_rmse - rmse) / naive_rmse * 100)
    
    marker = " ✓" if order == best_order else ""
    print(f"ARMA{order}{' '*(6-len(str(order)))} {aic:<15.2f} {bic:<15.2f} {rmse:<15.6f} {improvement:<11.2f}%{marker}")

# ============================================================================
# 10. Key Insights
# ============================================================================
print("\n" + "="*70)
print("10. KEY INSIGHTS: ARMA MODELS")
print("="*70)

print(f"\nARMA Model Fundamentals:")
print(f"  - ARMA(p,q): Combines AR(p) and MA(q) components")
print(f"  - AR part: Captures dependency on past values")
print(f"  - MA part: Captures dependency on past errors")
print(f"  - More flexible than pure AR or MA models")

print(f"\nOptimal Model Selection:")
print(f"  - Best: ARMA{best_order} (by BIC criterion)")
print(f"  - BIC: {best_model.bic:.2f} (lower is better)")
print(f"  - RMSE: {metrics_best['rmse']:.6f}")
print(f"  - Improvement: {metrics_best['improvement']:.2f}% over naive")

print(f"\nACF/PACF Patterns for Identification:")
print(f"  - AR(p): PACF cuts off at lag p, ACF decays")
print(f"  - MA(q): ACF cuts off at lag q, PACF decays")
print(f"  - ARMA(p,q): Both ACF and PACF decay exponentially")

print(f"\nWhen to Use ARMA:")
print(f"  ✓ Stationary time series")
print(f"  ✓ Mixed autocorrelation patterns")
print(f"  ✓ When pure AR or MA is insufficient")
print(f"  ✗ Trending data (use ARIMA with d>0)")
print(f"  ✗ Seasonal data (use SARIMA)")

print(f"\nComparison with Previous Methods:")
print(f"  Day 12 - AR: Uses only past values")
print(f"  Day 13 - ARMA: Uses past values + past errors ← HERE")

print("\n" + "="*70)
print("✓ Day 13 Analysis Complete!")
print("="*70)
