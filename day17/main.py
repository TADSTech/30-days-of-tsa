"""
Day 17: ARIMA Diagnostics
=========================

Comprehensive residual analysis for ARIMA model validation.
Tests include: Ljung-Box (autocorrelation), Jarque-Bera (normality),
heteroskedasticity, and ACF/PACF analysis.

Key Topics:
- Residual properties and white noise tests
- Ljung-Box test for autocorrelation
- Jarque-Bera test for normality
- Heteroskedasticity testing
- ACF and PACF analysis of residuals
- Visual diagnostics and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*70)
print("DAY 17: ARIMA DIAGNOSTICS - RESIDUAL ANALYSIS")
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
        df = pd.read_csv('/home/tads/Work/TADSPROJ/30-days-of-tsa/day17/data/gold_prices.csv', 
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
# 2. FIT OPTIMAL ARIMA(0,1,0) MODEL (FROM DAY 16)
# ============================================================================
print("\n" + "="*70)
print("2. FITTING OPTIMAL ARIMA(0,1,0) MODEL")
print("="*70)

model = ARIMA(train_data['Price'], order=(0, 1, 0))
fitted_model = model.fit()

print(f"\n✓ Model: ARIMA(0,1,0)")
print(f"✓ AIC: {fitted_model.aic:.2f}")
print(f"✓ BIC: {fitted_model.bic:.2f}")
print(f"\nModel Summary:")
print(fitted_model.summary())

# ============================================================================
# 3. EXTRACT RESIDUALS
# ============================================================================
print("\n" + "="*70)
print("3. RESIDUAL PROPERTIES")
print("="*70)

residuals_in_sample = fitted_model.resid
residuals = residuals_in_sample.dropna()

print(f"\nIn-Sample Residuals:")
print(f"  Count: {len(residuals)}")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std Dev: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.2f}")
print(f"  Max: {residuals.max():.2f}")
print(f"  Median: {residuals.median():.2f}")

# Test set residuals
forecast = fitted_model.get_forecast(steps=len(test_data))
forecast_values = forecast.predicted_mean.values
test_residuals = test_data['Price'].values - forecast_values

print(f"\nTest Set Residuals:")
print(f"  Count: {len(test_residuals)}")
print(f"  Mean: {test_residuals.mean():.2f}")
print(f"  Std Dev: {test_residuals.std():.2f}")
print(f"  Min: {test_residuals.min():.2f}")
print(f"  Max: {test_residuals.max():.2f}")

# ============================================================================
# 4. LJUNG-BOX TEST (AUTOCORRELATION)
# ============================================================================
print("\n" + "="*70)
print("4. LJUNG-BOX TEST FOR AUTOCORRELATION")
print("="*70)

print("\nNull Hypothesis: Residuals are independently distributed (white noise)")
print("Alternative: Residuals exhibit autocorrelation")

# Ljung-Box test at multiple lags
lags_to_test = [5, 10, 20, 40]
lb_results = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)

print(f"\nLjung-Box Test Results:")
print(lb_results.to_string())

alpha = 0.05
print(f"\nInterpretation (α = {alpha}):")
for lag in lags_to_test:
    p_value = lb_results.loc[lag, 'lb_pvalue']
    is_white_noise = p_value > alpha
    status = "✓ PASS" if is_white_noise else "✗ FAIL"
    print(f"  Lag {lag:2d}: p-value = {p_value:.4f} → {status} (white noise)" if is_white_noise 
          else f"  Lag {lag:2d}: p-value = {p_value:.4f} → {status} (autocorrelation detected)")

# ============================================================================
# 5. JARQUE-BERA TEST (NORMALITY)
# ============================================================================
print("\n" + "="*70)
print("5. JARQUE-BERA TEST FOR NORMALITY")
print("="*70)

jb_stat, jb_pvalue = stats.jarque_bera(residuals)
skewness = stats.skew(residuals)
kurtosis = stats.kurtosis(residuals)

print(f"\nNull Hypothesis: Residuals follow normal distribution")
print(f"Alternative: Residuals deviate from normality")

print(f"\nJarque-Bera Test Results:")
print(f"  Test Statistic: {jb_stat:.4f}")
print(f"  P-value: {jb_pvalue:.4f}")
print(f"  Skewness: {skewness:.4f} (ideally 0)")
print(f"  Kurtosis: {kurtosis:.4f} (ideally 0)")

alpha = 0.05
is_normal = jb_pvalue > alpha
status = "✓ PASS" if is_normal else "✗ FAIL"
print(f"\nInterpretation (α = {alpha}):")
print(f"  Result: {status} - Residuals {'are' if is_normal else 'are NOT'} normally distributed")

if abs(skewness) < 0.5:
    print(f"  Skewness: ✓ Acceptable (|{skewness:.4f}| < 0.5)")
else:
    print(f"  Skewness: ✗ High asymmetry ({skewness:.4f})")

if abs(kurtosis) < 1.0:
    print(f"  Kurtosis: ✓ Acceptable (|{kurtosis:.4f}| < 1.0)")
else:
    print(f"  Kurtosis: ✗ Heavy/light tails ({kurtosis:.4f})")

# ============================================================================
# 6. HETEROSKEDASTICITY TEST
# ============================================================================
print("\n" + "="*70)
print("6. HETEROSKEDASTICITY ANALYSIS")
print("="*70)

# Calculate rolling variance
window = 50
rolling_var = pd.Series(residuals).rolling(window=window).var()

mean_var_first = rolling_var[:len(rolling_var)//2].mean()
mean_var_second = rolling_var[len(rolling_var)//2:].mean()
h_statistic = mean_var_second / mean_var_first if mean_var_first > 0 else 1.0

print(f"\nNull Hypothesis: Residuals have constant variance (homoscedastic)")
print(f"Alternative: Variance changes over time (heteroscedastic)")

print(f"\nRolling Variance Analysis (window={window}):")
print(f"  First half mean variance: {mean_var_first:.2f}")
print(f"  Second half mean variance: {mean_var_second:.2f}")
print(f"  Ratio (H-statistic): {h_statistic:.4f}")

if 0.8 < h_statistic < 1.25:
    print(f"  ✓ PASS - Variances are relatively constant")
else:
    print(f"  ✗ FAIL - Evidence of heteroscedasticity")

# ============================================================================
# 7. ACF AND PACF OF RESIDUALS
# ============================================================================
print("\n" + "="*70)
print("7. ACF AND PACF OF RESIDUALS")
print("="*70)

acf_values = acf(residuals, nlags=40, fft=True)
pacf_values = pacf(residuals, nlags=40, method='ywm')

significant_acf = np.sum(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(residuals)))
significant_pacf = np.sum(np.abs(pacf_values[1:]) > 1.96/np.sqrt(len(residuals)))

print(f"\nAutocorrelation Function (ACF):")
print(f"  Significant lags (40): {significant_acf}")
print(f"  Threshold: ~1-2 expected by chance at α=0.05")
print(f"  Status: {'✓ PASS' if significant_acf <= 3 else '✗ FAIL'} (white noise)" if significant_acf <= 3 
      else f"  Status: ✗ FAIL (autocorrelation present)")

print(f"\nPartial Autocorrelation Function (PACF):")
print(f"  Significant lags (40): {significant_pacf}")
print(f"  Threshold: ~1-2 expected by chance")
print(f"  Status: {'✓ PASS' if significant_pacf <= 3 else '✗ FAIL'}")

# ============================================================================
# 8. VISUALIZATION: RESIDUAL PLOTS
# ============================================================================
print("\n" + "="*70)
print("8. CREATING RESIDUAL DIAGNOSTIC PLOTS")
print("="*70)

fig = sp.make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Residuals Over Time',
        'Residual Distribution',
        'Q-Q Plot',
        'ACF of Residuals',
        'PACF of Residuals',
        'Rolling Mean and Variance'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'histogram'}],
        [{'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'scatter'}, {'type': 'scatter'}]
    ]
)

# Residuals over time
fig.add_trace(
    go.Scatter(
        x=train_data['Date'],
        y=residuals,
        mode='markers',
        marker=dict(color='#FF6B6B', size=4),
        name='Residuals'
    ),
    row=1, col=1
)
fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=1)

# Distribution
fig.add_trace(
    go.Histogram(
        x=residuals,
        nbinsx=40,
        marker=dict(color='#FFD700'),
        name='Distribution'
    ),
    row=1, col=2
)

# Q-Q Plot
sorted_residuals = np.sort(residuals)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))

fig.add_trace(
    go.Scatter(
        x=theoretical_quantiles,
        y=sorted_residuals,
        mode='markers',
        marker=dict(color='#4ECDC4', size=5),
        name='Q-Q Plot'
    ),
    row=2, col=1
)
# Add 45-degree line
min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Normal Line'
    ),
    row=2, col=1
)

# ACF plot
fig.add_trace(
    go.Bar(
        x=np.arange(len(acf_values)),
        y=acf_values,
        marker=dict(color='#95E1D3'),
        name='ACF'
    ),
    row=2, col=2
)
# Confidence interval
ci = 1.96 / np.sqrt(len(residuals))
fig.add_hline(y=ci, line_dash='dash', line_color='red', row=2, col=2)
fig.add_hline(y=-ci, line_dash='dash', line_color='red', row=2, col=2)

# PACF plot
fig.add_trace(
    go.Bar(
        x=np.arange(len(pacf_values)),
        y=pacf_values,
        marker=dict(color='#A8D8EA'),
        name='PACF'
    ),
    row=3, col=1
)
fig.add_hline(y=ci, line_dash='dash', line_color='red', row=3, col=1)
fig.add_hline(y=-ci, line_dash='dash', line_color='red', row=3, col=1)

# Rolling statistics
fig.add_trace(
    go.Scatter(
        x=train_data['Date'],
        y=pd.Series(residuals).rolling(window=50).mean(),
        mode='lines',
        name='Rolling Mean',
        line=dict(color='blue')
    ),
    row=3, col=2
)
fig.add_trace(
    go.Scatter(
        x=train_data['Date'],
        y=pd.Series(residuals).rolling(window=50).std(),
        mode='lines',
        name='Rolling Std',
        line=dict(color='red')
    ),
    row=3, col=2
)

fig.update_yaxes(title_text="Residual", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)
fig.update_yaxes(title_text="Theoretical", row=2, col=1)
fig.update_yaxes(title_text="ACF", row=2, col=2)
fig.update_yaxes(title_text="PACF", row=3, col=1)
fig.update_yaxes(title_text="Value", row=3, col=2)

fig.update_layout(height=1200, width=1400, showlegend=True, hovermode='x unified')
fig.write_html("residual_diagnostics.html")
print("\n✓ Saved: residual_diagnostics.html")

# ============================================================================
# 9. DIAGNOSTIC SUMMARY TABLE
# ============================================================================
print("\n" + "="*70)
print("9. DIAGNOSTIC SUMMARY")
print("="*70)

diagnostic_results = {
    'Test': [
        'Ljung-Box (lag 20)',
        'Jarque-Bera',
        'Heteroskedasticity',
        'ACF Significant Lags',
        'PACF Significant Lags'
    ],
    'Statistic': [
        f"{lb_results.loc[20, 'lb_stat']:.4f}",
        f"{jb_stat:.4f}",
        f"{h_statistic:.4f}",
        f"{significant_acf}",
        f"{significant_pacf}"
    ],
    'P-Value': [
        f"{lb_results.loc[20, 'lb_pvalue']:.4f}",
        f"{jb_pvalue:.4f}",
        "N/A",
        "N/A",
        "N/A"
    ],
    'Status': [
        '✓ PASS' if lb_results.loc[20, 'lb_pvalue'] > 0.05 else '✗ FAIL',
        '✓ PASS' if jb_pvalue > 0.05 else '✗ FAIL',
        '✓ PASS' if 0.8 < h_statistic < 1.25 else '✗ FAIL',
        '✓ PASS' if significant_acf <= 3 else '✗ FAIL',
        '✓ PASS' if significant_pacf <= 3 else '✗ FAIL'
    ]
}

summary_df = pd.DataFrame(diagnostic_results)
print("\n", summary_df.to_string(index=False))

# ============================================================================
# 10. MODEL DIAGNOSTICS INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("10. OVERALL MODEL ASSESSMENT")
print("="*70)

all_passed = all([
    lb_results.loc[20, 'lb_pvalue'] > 0.05,
    jb_pvalue > 0.05,
    0.8 < h_statistic < 1.25,
    significant_acf <= 3,
    significant_pacf <= 3
])

print(f"\nDiagnostic Test Results:")
print(f"  Autocorrelation (Ljung-Box): {'✓ PASS' if lb_results.loc[20, 'lb_pvalue'] > 0.05 else '✗ FAIL'}")
print(f"  Normality (Jarque-Bera): {'✓ PASS' if jb_pvalue > 0.05 else '✗ FAIL'}")
print(f"  Homoscedasticity: {'✓ PASS' if 0.8 < h_statistic < 1.25 else '✗ FAIL'}")
print(f"  ACF Structure: {'✓ PASS' if significant_acf <= 3 else '✗ FAIL'}")
print(f"  PACF Structure: {'✓ PASS' if significant_pacf <= 3 else '✗ FAIL'}")

print(f"\nOverall Assessment:")
if all_passed:
    print(f"  ✓ EXCELLENT - All diagnostic tests passed!")
    print(f"  Model assumptions satisfied. Residuals are white noise.")
    print(f"  Safe to use for forecasting and inference.")
else:
    print(f"  ⚠ PARTIAL - Some diagnostic tests failed.")
    print(f"  Review failing tests and consider model refinement.")

print(f"\nForecast Performance:")
rmse = np.sqrt(mean_squared_error(test_data['Price'], forecast_values))
mae = mean_absolute_error(test_data['Price'], forecast_values)
mape = np.mean(np.abs((test_data['Price'] - forecast_values) / test_data['Price'])) * 100

print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")
print(f"  MAPE: {mape:.2f}%")

# Naive baseline
naive_forecast = np.full(len(test_data), train_data['Price'].iloc[-1])
naive_rmse = np.sqrt(mean_squared_error(test_data['Price'], naive_forecast))

print(f"\nNaive Baseline Comparison:")
print(f"  Naive RMSE: {naive_rmse:.2f}")
print(f"  ARIMA RMSE: {rmse:.2f}")
improvement = (naive_rmse - rmse) / naive_rmse * 100
print(f"  Improvement: {improvement:+.2f}%")

print("\n" + "="*70)
print("✓ Day 17: ARIMA Diagnostics Complete!")
print("="*70)
