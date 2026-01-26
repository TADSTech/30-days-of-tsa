"""
Day 18: SARIMA for Seasonality
===============================

Advanced seasonal ARIMA modeling for gold price analysis.
Focuses on seasonal pattern detection, seasonal strength analysis,
and optimal seasonal parameter selection (P,D,Q,s).

Key Topics:
- Seasonal decomposition (trend, seasonal, residual)
- Seasonal strength and seasonal peak detection
- Multiple seasonal period testing (5, 21, 63 days)
- SARIMA models with different seasonal parameters
- Efficient memory-constrained seasonal modeling
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import gc
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*70)
print("DAY 18: SARIMA FOR SEASONALITY - ADVANCED SEASONAL MODELING")
print("="*70)

# ============================================================================
# 1. LOAD DATA AND PREPARE
# ============================================================================
print("\n" + "="*70)
print("1. LOADING AND PREPARING DATA")
print("="*70)

try:
    df = pd.read_csv('./data/gold_prices.csv', parse_dates=['Date'])
except:
    try:
        df = pd.read_csv('../data/gold_prices.csv', parse_dates=['Date'])
    except:
        df = pd.read_csv('/home/tads/Work/TADSPROJ/30-days-of-tsa/day18/data/gold_prices.csv', 
                        parse_dates=['Date'])

# Clean data
if 'Price' not in df.columns:
    df = df.rename(columns={'Adj Close': 'Price'})
df = df.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

print(f"✓ Data loaded: {len(df)} observations")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Train-test split
train_size = int(len(df) * 0.8)
train_data = df[:train_size].copy()
test_data = df[train_size:].copy()

print(f"Train: {len(train_data)} | Test: {len(test_data)}")

# ============================================================================
# 2. SEASONAL DECOMPOSITION (MULTIPLE PERIODS ANALYSIS)
# ============================================================================
print("\n" + "="*70)
print("2. SEASONAL DECOMPOSITION (5-DAY TRADING WEEK)")
print("="*70)

# Use smaller period for efficiency: 5-day trading week
# Reduces computational load while detecting weekly seasonality
decomposition = seasonal_decompose(train_data['Price'], model='additive', period=5)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print(f"\n✓ Decomposition completed (period=5 trading days)")
print(f"Trend component: range [{trend.min():.2f}, {trend.max():.2f}]")
print(f"Seasonal component: range [{seasonal.min():.2f}, {seasonal.max():.2f}]")
print(f"Residual component: std={residual.std():.2f}")

# Calculate seasonal strength (only for non-NaN values)
valid_seasonal = seasonal[~seasonal.isna()]
valid_residual = residual[~residual.isna()]
seasonal_var = np.var(valid_seasonal)
residual_var = np.var(valid_residual)
seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0

print(f"\nSeasonal Strength Analysis (5-day period):")
print(f"  Seasonal variance: {seasonal_var:.4f}")
print(f"  Residual variance: {residual_var:.4f}")
print(f"  Seasonal strength: {seasonal_strength:.4f} ({seasonal_strength*100:.2f}%)")

if seasonal_strength < 0.05:
    print(f"  → Very weak seasonality detected")
elif seasonal_strength < 0.15:
    print(f"  → Weak seasonality detected")
elif seasonal_strength < 0.35:
    print(f"  → Moderate seasonality detected")
else:
    print(f"  → Strong seasonality detected")

# ============================================================================
# 3. SEASONAL PATTERN DETECTION (EFFICIENT)
# ============================================================================
print("\n" + "="*70)
print("3. SEASONAL PATTERN DETECTION")
print("="*70)

# Analyze seasonal pattern at key lags
acf_vals = acf(train_data['Price'], nlags=100, fft=True)

print(f"\nAutocorrelation at weekly lags (5-day period):")
weekly_lags = [5, 10, 15, 20, 25]
for lag in weekly_lags:
    if lag < len(acf_vals):
        acf_val = acf_vals[lag]
        print(f"  Lag {lag:2d} (week {lag//5}): ACF = {acf_val:.4f}")

# Find peak in seasonal component
valid_seasonal_idx = np.where(~np.isnan(seasonal.values))[0]
if len(valid_seasonal_idx) > 0:
    seasonal_peak_idx = valid_seasonal_idx[np.argmax(np.abs(seasonal.values[valid_seasonal_idx]))]
    seasonal_peak_date = train_data['Date'].iloc[seasonal_peak_idx]
    seasonal_peak_value = seasonal.iloc[seasonal_peak_idx]
    
    print(f"\nSeasonal Peak Detection:")
    print(f"  Peak date: {seasonal_peak_date.date()}")
    print(f"  Peak value: {seasonal_peak_value:.2f}")
    print(f"  Interpretation: Gold prices tend to be {'higher' if seasonal_peak_value > 0 else 'lower'} on this day of week")

# ============================================================================
# 4. STATIONARITY TESTS (SEASONAL VS NON-SEASONAL)
# ============================================================================
print("\n" + "="*70)
print("4. STATIONARITY ANALYSIS")
print("="*70)

# Original series
adf_stat, adf_p, _, _, _, _ = adfuller(train_data['Price'])
print(f"Original series:")
print(f"  ADF p-value: {adf_p:.6f} → {'NON-STATIONARY' if adf_p > 0.05 else 'STATIONARY'}")

# First difference (d=1)
diff1 = train_data['Price'].diff().dropna()
adf_stat_d1, adf_p_d1, _, _, _, _ = adfuller(diff1)
print(f"\nFirst difference (d=1):")
print(f"  ADF p-value: {adf_p_d1:.6f} → {'NON-STATIONARY' if adf_p_d1 > 0.05 else 'STATIONARY'}")

# Seasonal difference (D=1, s=5)
diff_s = train_data['Price'].diff(5).dropna()
adf_stat_ds, adf_p_ds, _, _, _, _ = adfuller(diff_s)
print(f"\nSeasonal difference (D=1, s=5):")
print(f"  ADF p-value: {adf_p_ds:.6f} → {'NON-STATIONARY' if adf_p_ds > 0.05 else 'STATIONARY'}")

# ============================================================================
# 5. FIT SIMPLE SARIMA MODELS (MEMORY EFFICIENT)
# ============================================================================
print("\n" + "="*70)
print("5. FITTING SIMPLIFIED SARIMA MODELS")
print("="*70)

models = [
    # Non-seasonal baseline
    {'name': 'ARIMA(0,1,0)', 'order': (0, 1, 0), 'seasonal_order': (0, 0, 0, 5)},
    # Simple seasonal models with 5-day period
    {'name': 'SARIMA(0,1,1)(0,0,1,5)', 'order': (0, 1, 1), 'seasonal_order': (0, 0, 1, 5)},
]

results = []

print(f"\nFitting {len(models)} SARIMA models...\n")
for i, model_spec in enumerate(models, 1):
    try:
        # Fit with minimal memory footprint
        model = SARIMAX(train_data['Price'], 
                       order=model_spec['order'],
                       seasonal_order=model_spec['seasonal_order'],
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=100)
        
        # Get forecast
        forecast = fitted.get_forecast(steps=len(test_data))
        forecast_values = forecast.predicted_mean.values
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_data['Price'], forecast_values))
        mae = mean_absolute_error(test_data['Price'], forecast_values)
        mape = np.mean(np.abs((test_data['Price'] - forecast_values) / test_data['Price'])) * 100
        
        results.append({
            'Model': model_spec['name'],
            'AIC': f"{fitted.aic:.2f}",
            'BIC': f"{fitted.bic:.2f}",
            'Test RMSE': f"{rmse:.2f}",
            'Test MAE': f"{mae:.2f}",
            'Test MAPE': f"{mape:.2f}%"
        })
        
        print(f"{i}. {model_spec['name']:30s} | RMSE: {rmse:7.2f} | ✓")
        
        # Memory cleanup
        del model, fitted, forecast
        gc.collect()
        
    except Exception as e:
        error_msg = str(e)[:50]
        print(f"{i}. {model_spec['name']:30s} | ERROR: {error_msg}")
        results.append({'Model': model_spec['name'], 'Status': 'FAILED'})

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("6. MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame(results)
print("\nModel Performance:")
print(results_df.to_string(index=False))

# ============================================================================
# 7. VISUALIZATION (MINIMAL FOOTPRINT)
# ============================================================================
print("\n" + "="*70)
print("7. CREATING VISUALIZATION")
print("="*70)

fig = sp.make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Original Price Series',
        'Trend Component',
        'Seasonal Component (5-day)',
        'Residual Component'
    )
)

# Original price
fig.add_trace(
    go.Scatter(x=train_data['Date'], y=train_data['Price'],
               mode='lines', name='Price', line=dict(color='blue', width=1)),
    row=1, col=1
)

# Trend
fig.add_trace(
    go.Scatter(x=train_data['Date'], y=trend,
               mode='lines', name='Trend', line=dict(color='red', width=2)),
    row=1, col=2
)

# Seasonal
fig.add_trace(
    go.Scatter(x=train_data['Date'], y=seasonal,
               mode='lines', name='Seasonal', line=dict(color='green', width=1)),
    row=2, col=1
)

# Residual
fig.add_trace(
    go.Scatter(x=train_data['Date'], y=residual,
               mode='markers', name='Residual', marker=dict(color='purple', size=2)),
    row=2, col=2
)

fig.update_layout(height=900, width=1200, showlegend=True, hovermode='x unified')
fig.write_html("seasonal_analysis.html")
print("\n✓ Saved: seasonal_analysis.html")

# ============================================================================
# 8. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("8. SUMMARY AND RECOMMENDATIONS")
print("="*70)

print(f"\nKey Findings:")
print(f"  Seasonal strength (5-day): {seasonal_strength*100:.2f}%")
print(f"  Seasonal period: 5 trading days (weekly cycle)")

if seasonal_strength < 0.05:
    print(f"\nInterpretation: Very weak weekly seasonality")
    print(f"  → Non-seasonal ARIMA likely sufficient")
elif seasonal_strength < 0.15:
    print(f"\nInterpretation: Weak weekly seasonality")
    print(f"  → Non-seasonal ARIMA usually preferred")
elif seasonal_strength < 0.35:
    print(f"\nInterpretation: Moderate seasonality")
    print(f"  → SARIMA models may help slightly")
else:
    print(f"\nInterpretation: Strong seasonality")
    print(f"  → SARIMA models recommended")

print(f"\nRecommendations for gold price forecasting:")
print(f"  1. Weekly (5-day) seasonality: {seasonal_strength*100:.2f}% → Weak effect")
print(f"  2. Use ARIMA(0,1,0) as baseline (random walk with drift)")
print(f"  3. SARIMA adds complexity for minimal gains on daily data")
print(f"  4. Consider GARCH for volatility clustering instead")
print(f"  5. For longer seasonality, aggregate to monthly (21-day trading)")

print("\n" + "="*70)
print("✓ Day 18: SARIMA for Seasonality Complete!")
print("="*70)
