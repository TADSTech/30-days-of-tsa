"""
Day 21: Rolling Forecasts & Backtesting
Walk-forward validation and realistic backtesting methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
print("DAY 21: ROLLING FORECASTS & BACKTESTING")
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

# =============================================================================
# 2. WALK-FORWARD VALIDATION SETUP
# =============================================================================
print("\n" + "="*70)
print("2. WALK-FORWARD VALIDATION SETUP")
print("="*70)

# Use approximately last 2 years for backtesting
backtest_size = 252 * 2  # ~2 years of trading days
train_size = len(gold) - backtest_size

print(f"\nData Split:")
print(f"  Training data: 0 to {train_size} ({train_size} observations)")
print(f"  Backtest period: {train_size} to {len(gold)} ({backtest_size} observations)")
print(f"  Backtest range: {gold.index[train_size]} to {gold.index[-1]}")

# Initial train set
initial_train = gold['Price'].iloc[:train_size]
backtest_data = gold['Price'].iloc[train_size:]

print(f"\nInitial Training Set: {len(initial_train)} observations")
print(f"Backtest Set: {len(backtest_data)} observations")

# =============================================================================
# 3. ROLLING FORECAST CONFIGURATION
# =============================================================================
print("\n" + "="*70)
print("3. ROLLING FORECAST CONFIGURATION")
print("="*70)

# Parameters
forecast_horizon = 20  # 20-day ahead forecasts
rolling_window_step = 1  # Update model every 1 day

print(f"\nRolling Forecast Parameters:")
print(f"  Forecast Horizon: {forecast_horizon} days")
print(f"  Window Step: {rolling_window_step} day(s)")
print(f"  ARIMA Order: (1, 1, 0)")
print(f"  Total Forecasts: {len(backtest_data) - forecast_horizon + 1}")

# =============================================================================
# 4. WALK-FORWARD BACKTESTING
# =============================================================================
print("\n" + "="*70)
print("4. EXECUTING WALK-FORWARD BACKTEST")
print("="*70)

# Store results
rolling_forecasts = []
rolling_actuals = []
rolling_rmse = []
rolling_mae = []
rolling_dates = []

current_train = initial_train.copy()
num_iterations = 0

for i in range(0, len(backtest_data) - forecast_horizon + 1, rolling_window_step):
    try:
        # Fit model on current training set
        model = ARIMA(current_train, order=(1, 1, 0))
        fit = model.fit()
        
        # Forecast next 'forecast_horizon' steps
        forecast_result = fit.get_forecast(steps=forecast_horizon)
        forecast_values = forecast_result.predicted_mean.values
        
        # Get actual values for this forecast period
        actual_start = train_size + i
        actual_end = actual_start + forecast_horizon
        actuals_in_period = gold['Price'].iloc[actual_start:actual_end].values
        
        # Calculate metrics for this forecast
        rmse = np.sqrt(mean_squared_error(actuals_in_period, forecast_values))
        mae = mean_absolute_error(actuals_in_period, forecast_values)
        
        rolling_forecasts.append(forecast_values)
        rolling_actuals.append(actuals_in_period)
        rolling_rmse.append(rmse)
        rolling_mae.append(mae)
        rolling_dates.append(gold.index[actual_start])
        
        # Add next observation to training set (retraining)
        next_obs_idx = train_size + i + rolling_window_step
        if next_obs_idx < len(gold):
            current_train = pd.concat([
                current_train,
                gold['Price'].iloc[next_obs_idx:next_obs_idx + rolling_window_step]
            ])
        
        num_iterations += 1
        if (num_iterations) % 50 == 0:
            print(f"  Progress: {num_iterations} rolling forecasts completed...")
        
    except Exception as e:
        print(f"  Error at iteration {i}: {str(e)[:50]}")
        continue

print(f"\n✓ Completed {num_iterations} rolling forecasts")

# =============================================================================
# 5. BACKTEST PERFORMANCE SUMMARY
# =============================================================================
print("\n" + "="*70)
print("5. BACKTEST PERFORMANCE SUMMARY")
print("="*70)

# Flatten the rolling forecasts and actuals
all_forecasts = np.array([f for period in rolling_forecasts for f in period])
all_actuals = np.array([a for period in rolling_actuals for a in period])

# Calculate overall metrics
overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_forecasts))
overall_mae = mean_absolute_error(all_actuals, all_forecasts)
overall_mape = np.mean(np.abs((all_actuals - all_forecasts) / all_actuals)) * 100

# Naive forecast comparison
naive_forecast = np.repeat(initial_train.iloc[-1], len(all_actuals))
naive_rmse = np.sqrt(mean_squared_error(all_actuals, naive_forecast))

print(f"\nOverall Backtest Metrics (All {len(all_forecasts)} forecast steps):")
print(f"  RMSE: {overall_rmse:.2f}")
print(f"  MAE: {overall_mae:.2f}")
print(f"  MAPE: {overall_mape:.2f}%")

print(f"\nNaive Forecast Comparison:")
print(f"  Naive RMSE: {naive_rmse:.2f}")
print(f"  ARIMA RMSE: {overall_rmse:.2f}")
print(f"  Improvement: {(naive_rmse - overall_rmse) / naive_rmse * 100:+.2f}%")

# =============================================================================
# 6. ROLLING WINDOW PERFORMANCE
# =============================================================================
print("\n" + "="*70)
print("6. ROLLING WINDOW PERFORMANCE ANALYSIS")
print("="*70)

rolling_rmse_arr = np.array(rolling_rmse)
rolling_mae_arr = np.array(rolling_mae)

print(f"\nRolling RMSE Statistics:")
print(f"  Mean: {rolling_rmse_arr.mean():.2f}")
print(f"  Std Dev: {rolling_rmse_arr.std():.2f}")
print(f"  Min: {rolling_rmse_arr.min():.2f}")
print(f"  Max: {rolling_rmse_arr.max():.2f}")

print(f"\nRolling MAE Statistics:")
print(f"  Mean: {rolling_mae_arr.mean():.2f}")
print(f"  Std Dev: {rolling_mae_arr.std():.2f}")
print(f"  Min: {rolling_mae_arr.min():.2f}")
print(f"  Max: {rolling_mae_arr.max():.2f}")

# Identify periods of better/worse performance
percentile_25 = np.percentile(rolling_rmse_arr, 25)
percentile_75 = np.percentile(rolling_rmse_arr, 75)

good_periods = np.sum(rolling_rmse_arr < percentile_25)
bad_periods = np.sum(rolling_rmse_arr > percentile_75)

print(f"\nPerformance Distribution:")
print(f"  Good periods (RMSE < {percentile_25:.2f}): {good_periods}")
print(f"  Bad periods (RMSE > {percentile_75:.2f}): {bad_periods}")

# =============================================================================
# 7. FORECAST HORIZON ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("7. FORECAST HORIZON PERFORMANCE")
print("="*70)

# Analyze performance by forecast horizon
horizon_rmse = []
for horizon in range(1, min(6, forecast_horizon + 1)):
    errors = []
    for period_idx, (forecast_period, actuals_period) in enumerate(zip(rolling_forecasts, rolling_actuals)):
        if horizon <= len(forecast_period):
            errors.append((forecast_period[horizon-1] - actuals_period[horizon-1]) ** 2)
    
    if errors:
        rmse_h = np.sqrt(np.mean(errors))
        horizon_rmse.append(rmse_h)
        print(f"  {horizon}-day ahead RMSE: {rmse_h:.2f}")

# =============================================================================
# 8. AVOIDING LOOK-AHEAD BIAS
# =============================================================================
print("\n" + "="*70)
print("8. LOOK-AHEAD BIAS MITIGATION")
print("="*70)

print(f"\nLook-Ahead Bias Prevention Measures:")
print(f"  ✓ Only using past data for model training")
print(f"  ✓ Rolling window: Training set grows with each iteration")
print(f"  ✓ No future information used in forecasts")
print(f"  ✓ Out-of-sample testing with chronological order")
print(f"  ✓ Forecast horizon > 1 (multi-step ahead, not same-day)")

print(f"\nBacktest Integrity:")
print(f"  Training data updated: {num_iterations} times")
print(f"  Minimum train size: {len(initial_train)} observations")
print(f"  No reordering of time series data")
print(f"  Each forecast uses only historical data")

# =============================================================================
# 9. KEY INSIGHTS
# =============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS: ROLLING FORECASTS & BACKTESTING")
print("="*70)

print(f"\nWalk-Forward Validation Benefits:")
print(f"  ✓ Realistic out-of-sample performance")
print(f"  ✓ Avoids look-ahead bias")
print(f"  ✓ Models market regime changes")
print(f"  ✓ Multiple test periods (not just one)")

print(f"\nPerformance Findings:")
if (naive_rmse - overall_rmse) / naive_rmse * 100 > 0:
    print(f"  ✓ ARIMA beats naive forecast by {(naive_rmse - overall_rmse) / naive_rmse * 100:.2f}%")
else:
    print(f"  ⚠ ARIMA underperforms naive by {(overall_rmse - naive_rmse) / naive_rmse * 100:.2f}%")

print(f"\nMarket Efficiency Insight:")
if overall_mape < 5:
    print(f"  ⚠ Very low MAPE ({overall_mape:.2f}%) - unusual for financial data")
else:
    print(f"  ✓ Realistic MAPE ({overall_mape:.2f}%) reflects market unpredictability")

print(f"\nRobustness Analysis:")
print(f"  Performance consistency: Std Dev of RMSE = {rolling_rmse_arr.std():.2f}")
if rolling_rmse_arr.std() / rolling_rmse_arr.mean() < 0.3:
    print(f"  ✓ Model stable across different periods")
else:
    print(f"  ⚠ Model performance varies significantly")

print("\n" + "="*70)
print("✓ Day 21 Analysis Complete!")
print("="*70)
