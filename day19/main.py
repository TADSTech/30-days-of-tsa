"""
Day 19: Model Evaluation Metrics
=================================

Comprehensive evaluation framework for time series forecasts.
Implements and compares MAE, RMSE, MAPE, SMAPE, MDA, and directional accuracy.

Key Topics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric MAPE (SMAPE)
- Mean Directional Accuracy (MDA)
- Theil's U statistic
- Visual error analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

print("="*70)
print("DAY 19: MODEL EVALUATION METRICS - COMPREHENSIVE FORECAST ASSESSMENT")
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
        df = pd.read_csv('/home/tads/Work/TADSPROJ/30-days-of-tsa/day19/data/gold_prices.csv', 
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
# 2. FIT MULTIPLE MODELS FOR COMPARISON
# ============================================================================
print("\n" + "="*70)
print("2. FITTING CANDIDATE MODELS")
print("="*70)

models_fitted = {}

# Model 1: ARIMA(0,1,0) - Random walk
try:
    model1 = ARIMA(train_data['Price'], order=(0, 1, 0))
    fitted1 = model1.fit()
    forecast1 = fitted1.get_forecast(steps=len(test_data))
    models_fitted['ARIMA(0,1,0)'] = {
        'forecast': forecast1.predicted_mean.values,
        'model': fitted1
    }
    print(f"✓ ARIMA(0,1,0): AIC={fitted1.aic:.2f}")
except Exception as e:
    print(f"✗ ARIMA(0,1,0): {str(e)[:50]}")

# Model 2: ARIMA(1,1,0) - AR component
try:
    model2 = ARIMA(train_data['Price'], order=(1, 1, 0))
    fitted2 = model2.fit()
    forecast2 = fitted2.get_forecast(steps=len(test_data))
    models_fitted['ARIMA(1,1,0)'] = {
        'forecast': forecast2.predicted_mean.values,
        'model': fitted2
    }
    print(f"✓ ARIMA(1,1,0): AIC={fitted2.aic:.2f}")
except Exception as e:
    print(f"✗ ARIMA(1,1,0): {str(e)[:50]}")

# Model 3: ARIMA(0,1,1) - MA component
try:
    model3 = ARIMA(train_data['Price'], order=(0, 1, 1))
    fitted3 = model3.fit()
    forecast3 = fitted3.get_forecast(steps=len(test_data))
    models_fitted['ARIMA(0,1,1)'] = {
        'forecast': forecast3.predicted_mean.values,
        'model': fitted3
    }
    print(f"✓ ARIMA(0,1,1): AIC={fitted3.aic:.2f}")
except Exception as e:
    print(f"✗ ARIMA(0,1,1): {str(e)[:50]}")

# Model 4: Naive baseline (last value)
models_fitted['Naive'] = {
    'forecast': np.full(len(test_data), train_data['Price'].iloc[-1]),
    'model': None
}
print(f"✓ Naive (last value): Baseline model")

# Model 5: Seasonal naive (252-day annual)
if len(train_data) > 252:
    seasonal_forecast = []
    for i in range(len(test_data)):
        idx = len(train_data) - 252 + (i % 252)
        if idx >= 0 and idx < len(train_data):
            seasonal_forecast.append(train_data['Price'].iloc[idx])
        else:
            seasonal_forecast.append(train_data['Price'].iloc[-1])
    models_fitted['Seasonal Naive'] = {
        'forecast': np.array(seasonal_forecast),
        'model': None
    }
    print(f"✓ Seasonal Naive (252-day): Seasonal baseline")

# ============================================================================
# 3. CALCULATE EVALUATION METRICS
# ============================================================================
print("\n" + "="*70)
print("3. CALCULATING EVALUATION METRICS")
print("="*70)

actual = test_data['Price'].values
metrics_results = []

for model_name, model_data in models_fitted.items():
    forecast = model_data['forecast']
    
    # Basic errors
    errors = actual - forecast
    abs_errors = np.abs(errors)
    
    # 1. MAE (Mean Absolute Error)
    mae = np.mean(abs_errors)
    
    # 2. RMSE (Root Mean Squared Error)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # 3. MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # 4. SMAPE (Symmetric MAPE)
    smape = np.mean(2.0 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))) * 100
    
    # 5. Mean Direction Accuracy (MDA)
    actual_direction = np.diff(actual)
    forecast_direction = np.diff(forecast)
    correct_direction = np.sum((actual_direction > 0) == (forecast_direction > 0))
    mda = (correct_direction / len(actual_direction)) * 100
    
    # 6. Theil's U statistic
    theil_numerator = np.sum((actual[1:] - forecast[1:]) ** 2)
    theil_denominator = np.sum(actual[1:] ** 2)
    theil_u = np.sqrt(theil_numerator / theil_denominator) if theil_denominator > 0 else np.inf
    
    # 7. Mean Error (bias)
    me = np.mean(errors)
    
    # 8. Mean Percentage Error (signed)
    mpe = np.mean((actual - forecast) / actual) * 100
    
    # 9. Directional Accuracy (price up/down) - align arrays by using differences
    # Use first 502 elements to match the difference arrays
    da = mda  # DA is same as MDA (directional accuracy)
    
    # 10. Variance of errors
    error_std = np.std(errors)
    
    metrics_results.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MDA': mda,
        'Theil U': theil_u,
        'ME': me,
        'MPE': mpe,
        'Error Std': error_std
    })

# ============================================================================
# 4. DISPLAY METRICS SUMMARY
# ============================================================================
print("\nMetrics Summary:\n")

metrics_df = pd.DataFrame(metrics_results)
print(metrics_df.to_string(index=False))

# ============================================================================
# 5. DETAILED METRICS EXPLANATION
# ============================================================================
print("\n" + "="*70)
print("4. METRICS INTERPRETATION")
print("="*70)

best_mae_idx = metrics_df['MAE'].idxmin()
best_rmse_idx = metrics_df['RMSE'].idxmin()
best_mape_idx = metrics_df['MAPE'].idxmin()
best_mda_idx = metrics_df['MDA'].idxmax()

print(f"\nBest Models by Criterion:")
print(f"  MAE:  {metrics_df.loc[best_mae_idx, 'Model']:20s} ({metrics_df.loc[best_mae_idx, 'MAE']:.2f})")
print(f"  RMSE: {metrics_df.loc[best_rmse_idx, 'Model']:20s} ({metrics_df.loc[best_rmse_idx, 'RMSE']:.2f})")
print(f"  MAPE: {metrics_df.loc[best_mape_idx, 'Model']:20s} ({metrics_df.loc[best_mape_idx, 'MAPE']:.2f}%)")
print(f"  MDA:  {metrics_df.loc[best_mda_idx, 'Model']:20s} ({metrics_df.loc[best_mda_idx, 'MDA']:.2f}%)")

# ============================================================================
# 6. ERROR DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("5. ERROR DISTRIBUTION ANALYSIS")
print("="*70)

best_model = metrics_df.loc[best_rmse_idx, 'Model']
best_forecast = models_fitted[best_model]['forecast']
best_errors = actual - best_forecast

print(f"\nBest Model: {best_model}")
print(f"Error Statistics:")
print(f"  Mean: {np.mean(best_errors):.2f}")
print(f"  Std Dev: {np.std(best_errors):.2f}")
print(f"  Min: {np.min(best_errors):.2f}")
print(f"  Max: {np.max(best_errors):.2f}")
print(f"  Median: {np.median(best_errors):.2f}")
print(f"  Skewness: {stats.skew(best_errors):.2f}")
print(f"  Kurtosis: {stats.kurtosis(best_errors):.2f}")

# Percentage of errors within bounds
within_1std = np.sum(np.abs(best_errors) <= np.std(best_errors)) / len(best_errors) * 100
within_2std = np.sum(np.abs(best_errors) <= 2 * np.std(best_errors)) / len(best_errors) * 100

print(f"\nError Bounds:")
print(f"  Within ±1σ: {within_1std:.1f}%")
print(f"  Within ±2σ: {within_2std:.1f}%")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("6. CREATING VISUALIZATIONS")
print("="*70)

fig = sp.make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Actual vs Forecasts (Best Model)',
        'Forecast Errors',
        'Error Distribution',
        'MAE Comparison',
        'RMSE Comparison',
        'MAPE Comparison'
    ),
    specs=[
        [{'type': 'scatter'}, {'type': 'scatter'}],
        [{'type': 'histogram'}, {'type': 'bar'}],
        [{'type': 'bar'}, {'type': 'bar'}]
    ]
)

# Plot 1: Actual vs Best Forecast
fig.add_trace(
    go.Scatter(x=test_data['Date'], y=actual, mode='lines', name='Actual',
               line=dict(color='blue', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=test_data['Date'], y=best_forecast, mode='lines', name='Forecast',
               line=dict(color='red', dash='dash', width=2)),
    row=1, col=1
)

# Plot 2: Errors over time
fig.add_trace(
    go.Scatter(x=test_data['Date'], y=best_errors, mode='markers', name='Error',
               marker=dict(color='purple', size=5)),
    row=1, col=2
)
fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=2)

# Plot 3: Error histogram
fig.add_trace(
    go.Histogram(x=best_errors, nbinsx=30, name='Error Distribution',
                 marker=dict(color='green')),
    row=2, col=1
)

# Plot 4: MAE comparison
fig.add_trace(
    go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'],
           marker=dict(color='orange'), name='MAE'),
    row=2, col=2
)

# Plot 5: RMSE comparison
fig.add_trace(
    go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'],
           marker=dict(color='red'), name='RMSE'),
    row=3, col=1
)

# Plot 6: MAPE comparison
fig.add_trace(
    go.Bar(x=metrics_df['Model'], y=metrics_df['MAPE'],
           marker=dict(color='cyan'), name='MAPE'),
    row=3, col=2
)

fig.update_layout(height=1200, width=1400, showlegend=True, hovermode='x unified')
fig.write_html("evaluation_metrics.html")
print("\n✓ Saved: evaluation_metrics.html")

# ============================================================================
# 8. SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("7. SUMMARY AND RECOMMENDATIONS")
print("="*70)

print(f"\nMetrics Guide:")
print(f"  MAE:   Average absolute error (same units as data)")
print(f"  RMSE:  Root mean square error (penalizes large errors)")
print(f"  MAPE:  Percentage error (scale-independent)")
print(f"  SMAPE: Symmetric MAPE (avoids zero division)")
print(f"  MDA:   Direction accuracy (up/down movement)")
print(f"  U:     Theil's U statistic (0=perfect, 1=naive)")

print(f"\nBest Model Selection:")
print(f"  For accuracy: Use {metrics_df.loc[best_rmse_idx, 'Model']} (lowest RMSE)")
print(f"  For robustness: Use {metrics_df.loc[best_mae_idx, 'Model']} (lowest MAE)")
print(f"  For percentage: Use {metrics_df.loc[best_mape_idx, 'Model']} (lowest MAPE)")
print(f"  For direction: Use {metrics_df.loc[best_mda_idx, 'Model']} (highest MDA)")

print(f"\nKey Insights:")
print(f"  • Multiple metrics needed for complete evaluation")
print(f"  • RMSE sensitive to outliers; MAE more robust")
print(f"  • MAPE useful for percentage comparison across scales")
print(f"  • MDA important for trading (direction accuracy)")
print(f"  • Naive baseline provides lower bound comparison")

print("\n" + "="*70)
print("✓ Day 19: Model Evaluation Metrics Complete!")
print("="*70)
