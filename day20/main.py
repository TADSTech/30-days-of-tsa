"""
Day 20: Confidence Intervals and Prediction Bands
Quantifying uncertainty in ARIMA forecasts with prediction intervals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

print("\n" + "="*70)
print("DAY 20: CONFIDENCE INTERVALS & PREDICTION BANDS")
print("="*70)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
print("\n" + "="*70)
print("1. LOADING AND PREPARING GOLD PRICE DATA")
print("="*70)

gold = yf.download('GLD', start='2015-01-01', end='2026-01-25', progress=False)
gold['Price'] = (gold['Close'] * 10.8).round(0)
gold = gold[['Price']].copy()

print(f"Gold data shape: {gold.shape}")
print(f"Date range: {gold.index.min()} to {gold.index.max()}")
print(f"\nData summary:")
print(gold['Price'].describe())

# =============================================================================
# 2. STATIONARITY CHECK
# =============================================================================
print("\n" + "="*70)
print("2. STATIONARITY VERIFICATION")
print("="*70)

adf_result = adfuller(gold['Price'].diff().dropna(), autolag='AIC')
print(f"\nADF Test on First Differences:")
print(f"  Test Statistic: {adf_result[0]:.6f}")
print(f"  P-value: {adf_result[1]:.6f}")
print(f"  Result: {'STATIONARY' if adf_result[1] < 0.05 else 'NON-STATIONARY'}")

# =============================================================================
# 3. TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "="*70)
print("3. TRAIN-TEST SPLIT")
print("="*70)

train_size = int(len(gold) * 0.8)
train_data = gold['Price'].iloc[:train_size]
test_data = gold['Price'].iloc[train_size:]

print(f"Training set: {len(train_data)} observations ({len(train_data)/len(gold)*100:.1f}%)")
print(f"Test set: {len(test_data)} observations ({len(test_data)/len(gold)*100:.1f}%)")
print(f"Train period: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test period: {test_data.index.min()} to {test_data.index.max()}")

# =============================================================================
# 4. FIT ARIMA MODEL
# =============================================================================
print("\n" + "="*70)
print("4. FITTING ARIMA(1,1,0) MODEL")
print("="*70)

# Use ARIMA(1,1,0) from Day 14 - optimal model for this data
model = ARIMA(gold['Price'], order=(1, 1, 0))
fit = model.fit()

print(f"\nARIMA(1,1,0) Model Fitted")
print(f"  AIC: {fit.aic:.2f}")
print(f"  BIC: {fit.bic:.2f}")
print(f"  Observations: {fit.nobs}")

# =============================================================================
# 5. GENERATE FORECASTS WITH CONFIDENCE INTERVALS
# =============================================================================
print("\n" + "="*70)
print("5. FORECAST WITH PREDICTION INTERVALS")
print("="*70)

# Get forecasts with prediction intervals for test period
forecast_result = fit.get_forecast(steps=len(test_data))
forecast_values = forecast_result.predicted_mean
forecast_ci_95 = forecast_result.conf_int(alpha=0.05)  # 95% CI
forecast_ci_90 = forecast_result.conf_int(alpha=0.10)  # 90% CI
forecast_ci_99 = forecast_result.conf_int(alpha=0.01)  # 99% CI

print(f"\nForecast Statistics (Test Period):")
print(f"  Forecast Mean: {forecast_values.mean():.2f}")
print(f"  Forecast Std: {forecast_values.std():.2f}")
print(f"  Forecast Min: {forecast_values.min():.2f}")
print(f"  Forecast Max: {forecast_values.max():.2f}")

# Extract confidence bounds
ci_95_lower = forecast_ci_95.iloc[:, 0]
ci_95_upper = forecast_ci_95.iloc[:, 1]
ci_90_lower = forecast_ci_90.iloc[:, 0]
ci_90_upper = forecast_ci_90.iloc[:, 1]
ci_99_lower = forecast_ci_99.iloc[:, 0]
ci_99_upper = forecast_ci_99.iloc[:, 1]

print(f"\nConfidence Interval Widths (at first forecast point):")
print(f"  90% CI Width: {ci_90_upper.iloc[0] - ci_90_lower.iloc[0]:.2f}")
print(f"  95% CI Width: {ci_95_upper.iloc[0] - ci_95_lower.iloc[0]:.2f}")
print(f"  99% CI Width: {ci_99_upper.iloc[0] - ci_99_lower.iloc[0]:.2f}")

# =============================================================================
# 6. PERFORMANCE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("6. FORECAST PERFORMANCE EVALUATION")
print("="*70)

rmse = np.sqrt(mean_squared_error(test_data, forecast_values))
mae = mean_absolute_error(test_data, forecast_values)

# Check how many actuals fall within confidence intervals
# Convert to numpy arrays to avoid pandas index alignment issues
test_data_np = test_data.values
ci_95_lower_np = ci_95_lower.values
ci_95_upper_np = ci_95_upper.values
ci_90_lower_np = ci_90_lower.values
ci_90_upper_np = ci_90_upper.values
ci_99_lower_np = ci_99_lower.values
ci_99_upper_np = ci_99_upper.values

within_95 = np.sum((test_data_np >= ci_95_lower_np) & (test_data_np <= ci_95_upper_np))
within_90 = np.sum((test_data_np >= ci_90_lower_np) & (test_data_np <= ci_90_upper_np))
within_99 = np.sum((test_data_np >= ci_99_lower_np) & (test_data_np <= ci_99_upper_np))

print(f"\nForecast Accuracy:")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE: {mae:.2f}")

print(f"\nActual Values Within Confidence Intervals:")
print(f"  90% CI: {within_90}/{len(test_data)} = {within_90/len(test_data)*100:.1f}%")
print(f"  95% CI: {within_95}/{len(test_data)} = {within_95/len(test_data)*100:.1f}%")
print(f"  99% CI: {within_99}/{len(test_data)} = {within_99/len(test_data)*100:.1f}%")

# Naive forecast comparison
naive_forecast = pd.Series([train_data.iloc[-1]] * len(test_data), index=test_data.index)
naive_rmse = np.sqrt(mean_squared_error(test_data, naive_forecast))
improvement = ((naive_rmse - rmse) / naive_rmse * 100)

print(f"\nComparison with Naive Forecast:")
print(f"  Naive RMSE: {naive_rmse:.2f}")
print(f"  ARIMA RMSE: {rmse:.2f}")
print(f"  Improvement: {improvement:+.2f}%")

# =============================================================================
# 7. INTERVAL WIDTH ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("7. CONFIDENCE INTERVAL ANALYSIS")
print("="*70)

# Calculate interval widths over time
width_95 = ci_95_upper - ci_95_lower
width_90 = ci_90_upper - ci_90_lower
width_99 = ci_99_upper - ci_99_lower

print(f"\n95% Confidence Interval:")
print(f"  Mean Width: {width_95.mean():.2f}")
print(f"  Min Width: {width_95.min():.2f}")
print(f"  Max Width: {width_95.max():.2f}")
print(f"  Std Dev: {width_95.std():.2f}")

print(f"\n90% Confidence Interval:")
print(f"  Mean Width: {width_90.mean():.2f}")
print(f"  Min Width: {width_90.min():.2f}")
print(f"  Max Width: {width_90.max():.2f}")
print(f"  Std Dev: {width_90.std():.2f}")

print(f"\n99% Confidence Interval:")
print(f"  Mean Width: {width_99.mean():.2f}")
print(f"  Min Width: {width_99.min():.2f}")
print(f"  Max Width: {width_99.max():.2f}")
print(f"  Std Dev: {width_99.std():.2f}")

# =============================================================================
# 8. RESIDUAL ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("8. RESIDUAL AND UNCERTAINTY ANALYSIS")
print("="*70)

residuals = fit.resid
print(f"\nIn-Sample Residuals:")
print(f"  Mean: {residuals.mean():.2f}")
print(f"  Std Dev: {residuals.std():.2f}")
print(f"  Min: {residuals.min():.2f}")
print(f"  Max: {residuals.max():.2f}")

# Forecast error analysis
forecast_errors = test_data.values - forecast_values.values
print(f"\nOut-of-Sample Forecast Errors:")
print(f"  Mean: {forecast_errors.mean():.2f}")
print(f"  Std Dev: {forecast_errors.std():.2f}")
print(f"  Min: {forecast_errors.min():.2f}")
print(f"  Max: {forecast_errors.max():.2f}")

# =============================================================================
# 9. KEY INSIGHTS
# =============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS: CONFIDENCE INTERVALS AND UNCERTAINTY")
print("="*70)

print(f"\nWhat Confidence Intervals Tell Us:")
print(f"  - 90% CI: Wider, captures actuals 90% of the time")
print(f"  - 95% CI: Standard choice, captures actuals ~95% of the time")
print(f"  - 99% CI: Widest, very conservative for important decisions")

print(f"\nInterpretation:")
print(f"  - CI Width increases with forecast horizon")
print(f"  - Wider CIs = More uncertainty further in future")
print(f"  - Narrow CIs = High confidence in near-term forecasts")

print(f"\nActual Coverage Rates:")
if within_95/len(test_data) >= 0.93:
    print(f"  ✓ 95% CI coverage: {within_95/len(test_data)*100:.1f}% (Good!)")
else:
    print(f"  ⚠ 95% CI coverage: {within_95/len(test_data)*100:.1f}% (Below expected)")

print(f"\nModel Calibration:")
if abs(within_95/len(test_data) - 0.95) < 0.05:
    print(f"  ✓ Model well-calibrated for uncertainty")
else:
    print(f"  ⚠ Model may under/overestimate uncertainty")

print("\n" + "="*70)
print("✓ Day 20 Analysis Complete!")
print("="*70)
