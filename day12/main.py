#!/usr/bin/env python3
"""
Day 12: Autoregressive (AR) Models
Fundamentals of AR processes, order selection using PACF, and AR model fitting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("✓ Libraries imported successfully")

# ============================================================================
# 1. Load and Explore Data
# ============================================================================
print("\n" + "="*70)
print("1. LOADING AND EXPLORING GOLD PRICE DATA")
print("="*70)

# Fetch gold price data
gold = yf.download('GLD', start='2015-01-01', end='2026-01-18', progress=False)
gold['Price'] = (gold['Close'] * 10.8).round(0)  # Convert to approximate gold price per ounce
gold = gold[['Price']].copy()

print(f"Gold data shape: {gold.shape}")
print(f"Date range: {gold.index.min()} to {gold.index.max()}")
print(f"\nData summary:")
print(gold['Price'].describe())
print(f"Missing values: {gold['Price'].isna().sum()}")

# Plot original series
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=gold.index, y=gold['Price'],
    mode='lines',
    name='Gold Price (Daily)',
    line=dict(color='goldenrod', width=1)
))
fig.update_layout(
    title="Gold Price Time Series (2015-2026)",
    xaxis_title='Date',
    yaxis_title='Price ($)',
    template='plotly_white',
    height=500,
    hovermode='x unified'
)
print("\n✓ Original series plotted")

# ============================================================================
# 2. Prepare Data (Differencing for Stationarity)
# ============================================================================
print("\n" + "="*70)
print("2. PREPARING DATA - DIFFERENCING FOR STATIONARITY")
print("="*70)

# First difference to make series stationary
gold_diff = gold['Price'].diff().dropna()

print(f"Original series mean: {gold['Price'].mean():.2f}")
print(f"Differenced series mean: {gold_diff.mean():.6f}")
print(f"Original series std: {gold['Price'].std():.2f}")
print(f"Differenced series std: {gold_diff.std():.2f}")

# Plot original vs differenced
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Original Series', 'First Differenced'),
    shared_xaxes=True,
    vertical_spacing=0.1
)

fig.add_trace(
    go.Scatter(x=gold.index, y=gold['Price'], mode='lines', name='Original', line=dict(color='blue')),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=gold_diff.index, y=gold_diff.values, mode='lines', name='Differenced', line=dict(color='red')),
    row=2, col=1
)

fig.update_layout(height=600, showlegend=True, title_text="Original vs First Differenced Series")
print("✓ Original vs differenced plotted")

# ============================================================================
# 3. ACF and PACF Analysis
# ============================================================================
print("\n" + "="*70)
print("3. ACF AND PACF ANALYSIS FOR ORDER SELECTION")
print("="*70)

# Plot ACF and PACF on the differenced series
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
)

# Calculate ACF
acf_values = pd.Series(index=range(40))
pacf_values = pd.Series(index=range(40))

from statsmodels.tsa.stattools import acf, pacf
acf_vals = acf(gold_diff, nlags=40)
pacf_vals = pacf(gold_diff, nlags=40, method='ywm')

# Add ACF
fig.add_trace(
    go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name='ACF', marker=dict(color='blue')),
    row=1, col=1
)

# Add confidence interval for ACF
ci_acf = 1.96 / np.sqrt(len(gold_diff))
fig.add_hline(y=ci_acf, line_dash='dash', line_color='red', row=1, col=1, annotation_text="95% CI")
fig.add_hline(y=-ci_acf, line_dash='dash', line_color='red', row=1, col=1)

# Add PACF
fig.add_trace(
    go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name='PACF', marker=dict(color='green')),
    row=1, col=2
)

# Add confidence interval for PACF
ci_pacf = 1.96 / np.sqrt(len(gold_diff))
fig.add_hline(y=ci_pacf, line_dash='dash', line_color='red', row=1, col=2, annotation_text="95% CI")
fig.add_hline(y=-ci_pacf, line_dash='dash', line_color='red', row=1, col=2)

fig.update_layout(height=500, showlegend=True, title_text="ACF and PACF of Differenced Series")
print("✓ ACF and PACF plotted")

# Determine AR order from PACF
print(f"\nPACF Analysis (Differenced Series):")
print(f"  Lags 1-5 PACF values: {pacf_vals[1:6]}")
print(f"  95% Confidence interval: ±{ci_pacf:.4f}")

# Find significant lags in PACF
significant_lags = [i for i in range(1, 21) if abs(pacf_vals[i]) > ci_pacf]
print(f"  Significant lags (within first 20): {significant_lags[:10]}")  # Show first 10
ar_order = significant_lags[0] if significant_lags else 1
print(f"\nSuggested AR order (p): {ar_order} (first significant lag in PACF)")

# ============================================================================
# 4. Train-Test Split
# ============================================================================
print("\n" + "="*70)
print("4. TRAIN-TEST SPLIT")
print("="*70)

# Use differenced series for AR modeling
train_size = int(len(gold_diff) * 0.8)
train = gold_diff.iloc[:train_size]
test = gold_diff.iloc[train_size:]

print(f"Training set: {len(train)} observations ({len(train)/len(gold_diff)*100:.1f}%)")
print(f"Test set: {len(test)} observations ({len(test)/len(gold_diff)*100:.1f}%)")
print(f"Train period: {train.index.min()} to {train.index.max()}")
print(f"Test period: {test.index.min()} to {test.index.max()}")

# ============================================================================
# 5. Fit AR Models with Different Orders
# ============================================================================
print("\n" + "="*70)
print("5. FITTING AR MODELS WITH DIFFERENT ORDERS")
print("="*70)

# Fit multiple AR models
ar_orders = [1, 2, 3, 5, 7, 10]
models = {}
aic_values = []
bic_values = []

print(f"\n{'Order':<8} {'AIC':<15} {'BIC':<15} {'Params':<40}")
print("-" * 80)

for p in ar_orders:
    model = AutoReg(train, lags=p, seasonal=False)
    fit = model.fit()
    models[p] = fit
    aic_values.append(fit.aic)
    bic_values.append(fit.bic)
    
    # Get AR coefficients
    params = fit.params
    print(f"{p:<8} {fit.aic:<15.2f} {fit.bic:<15.2f} {str(list(np.round(params.values[1:4], 4)))[1:-1]:<40}")

optimal_order_aic = ar_orders[np.argmin(aic_values)]
optimal_order_bic = ar_orders[np.argmin(bic_values)]

print(f"\nOptimal AR order by AIC: {optimal_order_aic}")
print(f"Optimal AR order by BIC: {optimal_order_bic}")

# Use BIC optimal model (more conservative)
optimal_order = optimal_order_bic
fit_optimal = models[optimal_order]

print(f"\n✓ Selected AR({optimal_order}) model")
print(f"\nAR({optimal_order}) Summary:")
print(fit_optimal.summary())

# ============================================================================
# 6. Extract AR Coefficients
# ============================================================================
print("\n" + "="*70)
print("6. AR COEFFICIENTS AND INTERPRETATION")
print("="*70)

params = fit_optimal.params
print(f"\nAR({optimal_order}) Coefficients:")
print(f"  Constant: {params.iloc[0]:.6f}")

for i in range(1, optimal_order + 1):
    coef = params.iloc[i]
    print(f"  φ_{i} (lag {i}): {coef:.6f}")

print(f"\nInterpretation:")
print(f"  Formula: ŷ(t) = c + Σ(φ_i × y(t-i)) + ε(t)")
print(f"  - Each coefficient φ_i shows the weight of past {i} step(s)")
print(f"  - Positive coefficient: positive autocorrelation at that lag")
print(f"  - Negative coefficient: negative autocorrelation at that lag")
print(f"  - Magnitude indicates strength of the relationship")

# ============================================================================
# 7. Generate Forecasts
# ============================================================================
print("\n" + "="*70)
print("7. GENERATING FORECASTS")
print("="*70)

# Forecast on test set
forecast = fit_optimal.forecast(steps=len(test))
forecast_series = pd.Series(forecast.values, index=test.index)

print(f"Forecast horizon: {len(test)} steps")
print(f"First 5 forecasts: {forecast_series.iloc[:5].values}")
print(f"Last 5 forecasts: {forecast_series.iloc[-5:].values}")

# ============================================================================
# 8. Evaluate Model Performance
# ============================================================================
print("\n" + "="*70)
print("8. MODEL PERFORMANCE EVALUATION")
print("="*70)

# Calculate metrics
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

# Naive forecast (use last value as forecast for all steps)
naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))
improvement = ((naive_rmse - rmse) / naive_rmse * 100)

print(f"\nAR({optimal_order}) Model Performance:")
print(f"  RMSE: {rmse:.6f}")
print(f"  MAE: {mae:.6f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  Naive RMSE (last value): {naive_rmse:.6f}")
print(f"  Improvement over naive: {improvement:.2f}%")

# Residual analysis
residuals = test - forecast
print(f"\nResidual Analysis:")
print(f"  Mean: {residuals.mean():.6f} (should be ~0)")
print(f"  Std Dev: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")

# ============================================================================
# 9. Visualize Results
# ============================================================================
print("\n" + "="*70)
print("9. VISUALIZING RESULTS")
print("="*70)

# Create comparison plot
fig = go.Figure()

# Training data
fig.add_trace(go.Scatter(
    x=train.index, y=train.values,
    mode='lines',
    name='Training Data',
    line=dict(color='blue', width=1)
))

# Test data (actual)
fig.add_trace(go.Scatter(
    x=test.index, y=test.values,
    mode='lines',
    name='Test Data (Actual)',
    line=dict(color='green', width=2)
))

# Forecast
fig.add_trace(go.Scatter(
    x=test.index, y=forecast_series.values,
    mode='lines',
    name=f'AR({optimal_order}) Forecast',
    line=dict(color='red', width=2, dash='dash')
))

# Naive forecast
fig.add_trace(go.Scatter(
    x=test.index, y=naive_forecast.values,
    mode='lines',
    name='Naive Forecast',
    line=dict(color='orange', width=2, dash='dot')
))

fig.update_layout(
    title=f"AR({optimal_order}) Model: Forecast vs Actual",
    xaxis_title='Date',
    yaxis_title='Price Change ($)',
    hovermode='x unified',
    template='plotly_white',
    height=500
)
print("✓ Forecast comparison plotted")

# AIC and BIC comparison
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=ar_orders, y=aic_values,
    mode='lines+markers',
    name='AIC',
    marker=dict(size=10, color='blue')
))

fig2.add_trace(go.Scatter(
    x=ar_orders, y=bic_values,
    mode='lines+markers',
    name='BIC',
    marker=dict(size=10, color='red')
))

fig2.update_layout(
    title="Model Selection: AIC vs BIC",
    xaxis_title='AR Order (p)',
    yaxis_title='Information Criterion',
    hovermode='x unified',
    template='plotly_white',
    height=500
)
print("✓ AIC/BIC comparison plotted")

# ============================================================================
# 10. Summary and Key Insights
# ============================================================================
print("\n" + "="*70)
print("10. KEY INSIGHTS: AUTOREGRESSIVE (AR) MODELS")
print("="*70)

print(f"\nAR Model Fundamentals:")
print(f"  - AR(p): Current value depends on p past values")
print(f"  - Formula: y(t) = c + φ₁×y(t-1) + φ₂×y(t-2) + ... + φₚ×y(t-p) + ε(t)")
print(f"  - Assumes: Series is stationary (constant mean, variance)")
print(f"  - Requires: Differencing for non-stationary data")

print(f"\nOrder Selection (AR order = p):")
print(f"  - PACF: Partial Autocorrelation Function identifies cutoff")
print(f"  - AIC: Penalizes model complexity (selected: {optimal_order_aic})")
print(f"  - BIC: Stronger complexity penalty (selected: {optimal_order_bic})")
print(f"  - Chosen: AR({optimal_order}) by BIC")

print(f"\nOptimal AR({optimal_order}) Parameters:")
for i in range(1, min(4, optimal_order + 1)):
    coef = params.iloc[i]
    print(f"  φ_{i}: {coef:.6f}")

print(f"\nForecast Performance:")
print(f"  - RMSE: {rmse:.6f} vs Naive: {naive_rmse:.6f}")
print(f"  - Improvement: {improvement:.2f}%")
print(f"  - Model {'outperforms' if improvement > 0 else 'underperforms'} naive forecast")

print(f"\nWhen to Use AR Models:")
print(f"  ✓ Stationary time series")
print(f"  ✓ Recent history strongly affects future values")
print(f"  ✓ Differenced financial returns")
print(f"  ✗ Trending series (need differencing first)")
print(f"  ✗ Series with strong seasonality (need SARIMA)")

print("\n" + "="*70)
print("✓ Day 12 Analysis Complete!")
print("="*70)
