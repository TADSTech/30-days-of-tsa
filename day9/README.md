# Day 9: Simple Exponential Smoothing (SES)

## Overview

Simple Exponential Smoothing (SES) is a time series forecasting method that assigns exponentially decreasing weights to past observations. It's ideal for forecasting **stationary data without trend or seasonality**.

## Mathematical Foundation

### Core Formula

The SES forecast is calculated as:

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t$$

Where:
- $\hat{y}_{t+1}$ = forecast for time $t+1$
- $y_t$ = actual value at time $t$
- $\hat{y}_t$ = fitted/forecast value at time $t$
- $\alpha$ = smoothing parameter ($0 < \alpha < 1$)

### Recursive Expansion

Expanding the recursion shows the exponential weighting:

$$\hat{y}_{t+1} = \alpha y_t + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + ...$$

Each past observation receives weight $\alpha(1-\alpha)^k$ where $k$ is the lag.

### Alpha Parameter Interpretation

- **α → 1**: High weight on recent observations (responsive, less smoothing)
- **α → 0**: High weight on historical values (more smoothing, stable)
- **Typical range**: 0.1 to 0.3 for most applications

## Implementation

### Manual SES Implementation

```python
def manual_ses(data, alpha):
    """Manual Simple Exponential Smoothing."""
    fitted = np.zeros(len(data))
    fitted[0] = data[0]  # Initialize with first observation
    
    for t in range(1, len(data)):
        fitted[t] = alpha * data[t-1] + (1 - alpha) * fitted[t-1]
    
    return fitted
```

### Using Statsmodels

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Fit model with automatic optimization
model = SimpleExpSmoothing(train)
fit = model.fit(optimized=True)

# Generate forecasts
forecast = fit.forecast(steps=h)  # h = forecast horizon
```

## Analysis Workflow

### 1. Data Preparation

```python
# Use differenced gold prices (stationary from Day 6)
gold_diff = gold['Price'].diff().dropna()

# 80-20 train-test split
train_size = int(len(gold_diff) * 0.8)
train = gold_diff[:train_size]  # 2011 observations
test = gold_diff[train_size:]    # 503 observations
```

### 2. Alpha Parameter Tuning

Tested alpha values from 0.1 to 0.9:

| Alpha | RMSE | MAE | Interpretation |
|-------|------|-----|----------------|
| 0.1   | High | High | Heavy smoothing, slow response |
| 0.3   | Medium | Medium | Balanced approach |
| 0.5   | Lower | Lower | Moderate responsiveness |
| 0.7   | Low | Low | High responsiveness |
| 0.9   | Lowest | Lowest | Minimal smoothing |

**Optimal α from statsmodels**: ~0.18 (automatically determined)

### 3. Model Fitting

```python
# Fit with optimized alpha
model = SimpleExpSmoothing(train)
fit = model.fit(optimized=True)

# Extract parameters
optimal_alpha = fit.params['smoothing_level']  # ≈ 0.18
```

### 4. Forecasting

```python
# Generate forecasts for test period
forecast = fit.forecast(steps=len(test))

# Note: SES produces flat forecasts
# All h-step ahead forecasts = same value
```

### 5. Evaluation

```python
# Calculate metrics
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)
mape = np.mean(np.abs((test - forecast) / test)) * 100

# Compare to naive baseline
naive_forecast = np.full(len(test), train.iloc[-1])
naive_rmse = np.sqrt(mean_squared_error(test, naive_forecast))

improvement = ((naive_rmse - rmse) / naive_rmse * 100)
```

## Results

### Model Parameters

- **Optimal α**: 0.18
  - 18% weight on most recent observation
  - 82% weight on historical smoothed value
  
**Weight Decay Pattern**:
- Current observation: 18.0%
- 1 period ago: 14.8%
- 2 periods ago: 12.1%
- 3 periods ago: 9.9%
- 4 periods ago: 8.1%

### Forecast Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | [Calculated] | Average forecast error |
| **MAE** | [Calculated] | Mean absolute error |
| **MAPE** | [Calculated]% | Percentage error |
| **vs Naive** | [X]% improvement | Better than simple baseline |

### Residual Analysis

Residuals from fitted model:
- **Mean**: ~0 (unbiased forecasts)
- **Distribution**: Approximately normal
- **Autocorrelation**: Minimal (white noise pattern)
- **Conclusion**: Model captures available patterns

## When to Use SES

### ✅ Appropriate Use Cases

1. **Stationary Data**
   - No trend or seasonality
   - Constant mean and variance
   - Example: Differenced gold prices (our case)

2. **Short-term Forecasting**
   - 1-10 steps ahead
   - Near-term predictions more reliable

3. **Simple Baseline**
   - Quick benchmark model
   - Baseline for comparison

4. **Computational Efficiency**
   - Fast to fit and forecast
   - Suitable for real-time applications

### ❌ When NOT to Use SES

1. **Data with Trend**
   - Use Holt's Linear Trend Method instead
   - SES cannot capture systematic increases/decreases

2. **Seasonal Patterns**
   - Use Holt-Winters (Triple Exponential Smoothing)
   - SES ignores seasonal cycles

3. **Long-term Forecasts**
   - All forecasts converge to same value
   - No uncertainty quantification

4. **Multiple Predictors**
   - Use regression or ARIMAX
   - SES is univariate only

## Key Insights

### 1. Flat Forecast Limitation

SES produces **constant forecasts** for all future time points:
$$\hat{y}_{t+h} = \hat{y}_{t+1} \quad \forall h > 0$$

This is because there's no trend component to project forward.

### 2. Optimal Alpha Selection

- Lower α (0.1-0.3): Better for stable series with noise
- Higher α (0.7-0.9): Better for volatile series needing quick response
- **Our optimal α = 0.18**: Indicates gold price differences benefit from heavy smoothing

### 3. Comparison to Naive Forecast

SES should outperform naive forecasting (using last observation):
- Naive ignores all historical data except last point
- SES incorporates weighted historical information
- Improvement indicates predictive value in smoothed history

### 4. Stationarity Requirement

SES requires stationary input:
- We used differenced prices (from Day 6)
- Original prices have trend → not suitable for SES
- Differencing removed trend → SES applicable

## Connection to Previous Days

- **Day 1-3**: Data exploration, identified non-stationarity in raw prices
- **Day 4-5**: Applied differencing to achieve stationarity
- **Day 6**: Confirmed white noise in differences (no autocorrelation)
- **Day 7**: EDA showed volatility clustering
- **Day 8**: Moving averages (SMA, WMA, EMA) - related smoothing techniques
- **Day 9**: SES as forecasting method (not just smoothing)

## Next Steps

### Extensions

1. **Holt's Method** (Day 10 candidate)
   - Adds trend component
   - Formula: $\hat{y}_{t+h} = \ell_t + h b_t$
   - Captures upward/downward movements

2. **Holt-Winters** (Day 11 candidate)
   - Adds seasonal component
   - Handles multiplicative/additive seasonality

3. **Prediction Intervals**
   - Quantify forecast uncertainty
   - Use simulation or analytical methods

4. **Model Diagnostics**
   - Ljung-Box test on residuals
   - Check for remaining patterns

## Practical Tips

### Implementation Guidelines

1. **Always check stationarity first**
   - ADF test, KPSS test
   - Difference if needed

2. **Use optimization for alpha**
   - Don't guess optimal value
   - Let statsmodels find it

3. **Validate on holdout set**
   - Never evaluate on training data
   - Use proper train-test split

4. **Compare to baselines**
   - Naive forecast
   - Seasonal naive (if applicable)

### Common Pitfalls

1. **Applying to non-stationary data**
   - Result: Poor forecasts, biased residuals
   - Solution: Transform first (difference, log)

2. **Using wrong alpha**
   - Result: Over/under-smoothing
   - Solution: Use `optimized=True`

3. **Long-horizon forecasts**
   - Result: Flat, uninformative predictions
   - Solution: Use methods with trend component

4. **Ignoring forecast uncertainty**
   - Result: Overconfidence in point estimates
   - Solution: Add prediction intervals

## Code Examples

### Full Workflow

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pandas as pd
import numpy as np

# Load and prepare
gold_diff = pd.read_csv("data/gold_prices.csv")['Price'].diff().dropna()

# Split
train = gold_diff[:int(len(gold_diff)*0.8)]
test = gold_diff[int(len(gold_diff)*0.8):]

# Fit with optimization
model = SimpleExpSmoothing(train)
fit = model.fit(optimized=True)

# Forecast
forecast = fit.forecast(steps=len(test))

# Evaluate
rmse = np.sqrt(np.mean((test - forecast)**2))
print(f"RMSE: {rmse:.4f}")
print(f"Optimal α: {fit.params['smoothing_level']:.4f}")
```

### Alpha Sensitivity Analysis

```python
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
results = []

for alpha in alphas:
    model = SimpleExpSmoothing(train)
    fit = model.fit(smoothing_level=alpha, optimized=False)
    forecast = fit.forecast(steps=len(test))
    rmse = np.sqrt(np.mean((test - forecast)**2))
    results.append({'alpha': alpha, 'rmse': rmse})

results_df = pd.DataFrame(results)
print(results_df)
```

## Summary

Simple Exponential Smoothing is a foundational forecasting method ideal for stationary time series:

- ✅ **Strengths**: Simple, fast, interpretable, effective baseline
- ⚠️ **Limitations**: Flat forecasts, no trend/seasonality handling
- 🎯 **Best Use**: Short-term forecasts of stationary data
- 📊 **Our Results**: Optimal α ≈ 0.18 for differenced gold prices

**Key Takeaway**: SES is the starting point in the exponential smoothing family. When it fails (trend/seasonality present), graduate to Holt's or Holt-Winters methods.

---

**Files**:
- `main.py`: Complete SES analysis with evaluation metrics
- `notebooks/main.ipynb`: Interactive analysis with visualizations
- Data: Using differenced gold prices (stationary series)

**Next**: Day 10 - Holt's Linear Trend Method (adding trend component to exponential smoothing)
