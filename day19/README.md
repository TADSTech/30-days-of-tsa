# Day 19: Model Evaluation Metrics
## Comprehensive Forecast Assessment Framework

**Objective:** Master quantitative evaluation metrics for time series forecasts. Implement and compare MAE, RMSE, MAPE, SMAPE, MDA, and directional accuracy to select optimal forecasting models.

---

## Table of Contents
1. [Overview](#overview)
2. [Core Metrics](#core-metrics)
3. [Implementation](#implementation)
4. [Metric Comparisons](#metric-comparisons)
5. [When to Use](#when-to-use)

---

## Overview

After fitting multiple forecasting models, we need quantitative metrics to evaluate performance. Different metrics reveal different aspects:
- **Accuracy**: How close predictions are to actual values
- **Robustness**: How sensitive the metric is to outliers
- **Scale**: Whether the metric depends on absolute values or percentages
- **Interpretability**: How easy it is to explain to stakeholders

### The Problem with Single Metrics

Using only one metric can be misleading:
- **RMSE only**: Overstates large errors
- **MAE only**: Ignores magnitude of errors
- **MAPE only**: Fails on near-zero actual values
- **MDA only**: Ignores magnitude, only cares about direction

**Solution**: Use multiple complementary metrics

---

## Core Metrics

### 1. Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Characteristics:**
- **Units**: Same as actual data (e.g., $/oz for gold)
- **Robustness**: Robust to outliers (linear penalty)
- **Interpretation**: Average prediction error magnitude
- **Best for**: General accuracy assessment

**Example:**
```
Actual:    [100, 110, 105, 115]
Forecast:  [102, 108, 107, 113]
Errors:    [2, 2, 2, 2]
MAE:       2.0
```

**Pros:**
- ✓ Same units as data (interpretable)
- ✓ Robust to outliers
- ✓ Every error weighted equally

**Cons:**
- ✗ Absolute value not differentiable at zero
- ✗ Doesn't penalize large errors heavily
- ✗ Can't compare across different scales

---

### 2. Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Characteristics:**
- **Units**: Same as actual data
- **Robustness**: Sensitive to outliers (quadratic penalty)
- **Interpretation**: Standard deviation of errors
- **Best for**: Penalizing large errors

**Example:**
```
Actual:    [100, 110, 105, 115]
Forecast:  [102, 108, 109, 111]  ← One error is larger (4)
MSE:       (4+4+16+16)/4 = 10
RMSE:      √10 = 3.16
```

**Pros:**
- ✓ Penalizes large errors (quadratic)
- ✓ Differentiable everywhere
- ✓ Statistical properties well-understood

**Cons:**
- ✗ Sensitive to outliers
- ✗ Larger values harder to interpret
- ✗ Can't compare across different scales

---

### 3. Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$$

**Characteristics:**
- **Units**: Percentage (scale-independent)
- **Robustness**: Sensitive to small actual values
- **Interpretation**: Average percentage error
- **Best for**: Comparing across different scales

**Example:**
```
Actual:    [100, 200, 150]
Forecast:  [102, 198, 153]
Errors:    [2%, 1%, 2%]
MAPE:      1.67%
```

**Pros:**
- ✓ Scale-independent (percentages)
- ✓ Easy to interpret ("5% error")
- ✓ Comparable across different datasets

**Cons:**
- ✗ Undefined when actual = 0
- ✗ Biased toward underprediction
- ✗ Explodes on small actual values

---

### 4. Symmetric Mean Absolute Percentage Error (SMAPE)

$$\text{SMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|} \times 100\%$$

**Characteristics:**
- **Units**: Percentage (scale-independent)
- **Robustness**: Symmetric (fixes MAPE bias)
- **Interpretation**: Average percentage error (symmetric)
- **Best for**: Fixing MAPE zero-division problems

**Difference from MAPE:**
```
         Actual  Forecast  MAPE   SMAPE
Case 1:  100     50       50%    66.7%   (MAPE lower)
Case 2:  50      100      100%   66.7%   (MAPE higher)
Case 3:  0       10       ∞      200%    (MAPE fails!)
```

**Pros:**
- ✓ Symmetric (same penalty for over/under)
- ✓ Handles zero values better
- ✓ Always bounded [0, 200%]

**Cons:**
- ✗ Less intuitive than MAPE
- ✗ Still sensitive to near-zero values
- ✗ Bounded range can be limiting

---

### 5. Mean Directional Accuracy (MDA)

$$\text{MDA} = \frac{1}{n-1} \sum_{i=1}^{n-1} \mathbb{1}[\text{sign}(y_i - y_{i-1}) = \text{sign}(\hat{y}_i - \hat{y}_{i-1})] \times 100\%$$

**Characteristics:**
- **Units**: Percentage of correct directions
- **Robustness**: Ignores magnitude (only direction)
- **Interpretation**: How often forecast gets direction right
- **Best for**: Trading and risk management

**Example:**
```
Date  Actual  ActualDir  Forecast  ForecastDir  Correct?
t-1   100     -          102       -            -
t     110     ↑          108       ↑            ✓
t+1   105     ↓          107       ↑            ✗
t+2   115     ↑          113       ↑            ✓

MDA = 2/3 = 66.67%
```

**Pros:**
- ✓ Crucial for trading (correct direction valuable)
- ✓ Ignores magnitude (scale-independent)
- ✓ Intuitive interpretation

**Cons:**
- ✗ Ignores magnitude of error
- ✗ Can be high even with poor accuracy
- ✗ No partial credit for close calls

---

### 6. Theil's U Statistic

$$U = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} y_i^2}}$$

**Characteristics:**
- **Range**: [0, ∞) with interpretation:
  - U = 0: Perfect forecast
  - U = 1: Naive baseline performance
  - U > 1: Worse than naive
- **Interpretation**: Relative to naive baseline
- **Best for**: Comparing to naive (last value carried forward)

**Interpretation:**
```
U = 0.5  → 50% of naive RMSE (very good)
U = 1.0  → Same as naive (no improvement)
U = 2.0  → 2× worse than naive (poor model)
```

**Pros:**
- ✓ Normalized (comparable across datasets)
- ✓ Built-in baseline comparison
- ✓ Easy interpretation

**Cons:**
- ✗ Denominator can be small (instability)
- ✗ Undefined for stationary series
- ✗ Less commonly used

---

### 7. Mean Error (ME) and Mean Percentage Error (MPE)

$$\text{ME} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

$$\text{MPE} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right) \times 100\%$$

**Characteristics:**
- **Units**: Same as data (ME) or percentage (MPE)
- **Meaning**: Average bias (not absolute)
- **Interpretation**: Systematic over/under prediction

**Example:**
```
Actual:    [100, 110, 105, 115]
Forecast:  [102, 108, 107, 113]
Errors:    [2, -2, 2, -2]
ME:        0.0    (balanced, no bias)

Forecast2: [105, 115, 110, 120]
Errors:    [5, 5, 5, 5]
ME:        5.0    (consistently overpredicts)
```

**Pros:**
- ✓ Reveals systematic bias
- ✓ Simple to compute
- ✓ Interpretable sign

**Cons:**
- ✗ Errors can cancel out (masked by averaging)
- ✗ Doesn't show magnitude of errors
- ✗ Use with absolute error metrics

---

## Implementation

### Basic Implementation

```python
import numpy as np

# Data
actual = np.array([100, 110, 105, 115])
forecast = np.array([102, 108, 107, 113])

# 1. MAE
mae = np.mean(np.abs(actual - forecast))

# 2. RMSE
rmse = np.sqrt(np.mean((actual - forecast) ** 2))

# 3. MAPE
mape = np.mean(np.abs((actual - forecast) / actual)) * 100

# 4. SMAPE
smape = np.mean(2 * np.abs(actual - forecast) / 
                (np.abs(actual) + np.abs(forecast))) * 100

# 5. MDA (direction accuracy)
actual_direction = np.diff(actual)
forecast_direction = np.diff(forecast)
mda = np.mean((actual_direction > 0) == (forecast_direction > 0)) * 100

# 6. Theil's U
theil_u = np.sqrt(np.sum((actual - forecast) ** 2) / np.sum(actual ** 2))

# 7. ME and MPE
me = np.mean(actual - forecast)
mpe = np.mean((actual - forecast) / actual) * 100
```

### Using Scikit-learn

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

mae = mean_absolute_error(actual, forecast)
rmse = np.sqrt(mean_squared_error(actual, forecast))
```

---

## Metric Comparisons

### Robustness to Outliers

```
Setup: Gold price forecast with one large error
Actual:    [100, 110, 105, 115, 100]
Forecast:  [102, 108, 107, 113, 50]  ← One large error (50 vs 100)

MAE:   7.2   (moderate increase)
RMSE:  14.8  (large increase due to squaring)
MAPE:  14.7% (moderate increase)
MDA:   75%   (ignores magnitude)
```

**Conclusion**: RMSE amplifies outlier impact (quadratic), MAE is moderate, MAPE percentage-based, MDA completely ignores magnitude.

### Behavior with Different Scales

```
Test A (Gold prices $100-120):
  MAE = 2.0, RMSE = 2.5, MAPE = 1.8%

Test B (Stock prices $10-12):
  MAE = 0.2, RMSE = 0.25, MAPE = 1.8%

Conclusion:
  • MAE, RMSE not comparable (different scales)
  • MAPE comparable (same percentage)
```

---

## When to Use

### Decision Framework

| Scenario | Best Metric | Reason |
|----------|-------------|--------|
| **Financial accuracy** | RMSE | Penalizes large errors heavily |
| **Robustness needed** | MAE | Linear penalty, less outlier-sensitive |
| **Percentage comparison** | MAPE/SMAPE | Scale-independent |
| **Trading signals** | MDA | Only direction matters |
| **Model comparison** | Theil's U | Normalized by baseline |
| **Bias detection** | ME/MPE | Shows systematic error direction |
| **General evaluation** | MAE + RMSE | Complementary information |

### Financial Context (Gold Prices)

**For trading**: Use MDA (direction accuracy)
- Correct direction → Profitable trade
- Wrong direction → Losing trade
- Magnitude matters less than direction

**For risk management**: Use MAE + RMSE
- MAE: Average loss magnitude
- RMSE: Tail risk (large errors)
- Together: Full picture of risk

**For forecast quality**: Use MAPE
- Show percentage error to stakeholders
- "Average error is 2.5%" is interpretable
- Scale-independent for reporting

---

## Example: Selecting Best Model

```python
# Metrics for 3 models
models = {
    'ARIMA': {'MAE': 2.1, 'RMSE': 2.8, 'MAPE': 1.9, 'MDA': 55},
    'ETS':   {'MAE': 1.8, 'RMSE': 2.5, 'MAPE': 1.6, 'MDA': 58},
    'Naive': {'MAE': 3.0, 'RMSE': 4.2, 'MAPE': 2.7, 'MDA': 50}
}

# Decision:
# ETS best: Lowest MAE, RMSE, MAPE (best accuracy)
# ETS second: Highest MDA (best direction)
# Naive worst: Highest errors (baseline)
# Conclusion: Use ETS
```

---

## Results on Gold Price Data

### Model Comparison (Test Set: 503 observations, 2024-01-09 to 2026-01-09)

| Model | MAE | RMSE | MAPE | SMAPE | MDA | Theil U |
|-------|-----|------|------|-------|-----|---------|
| **ARIMA(0,1,0)** | 83.98 | 103.55 | 27.60% | 33.83% | 41.43% | 0.372 |
| ARIMA(1,1,0) | 84.02 | 103.58 | 27.62% | 33.85% | 41.43% | 0.372 |
| ARIMA(0,1,1) | 84.02 | 103.58 | 27.62% | 33.85% | 41.43% | 0.372 |
| Naive | 83.98 | 103.55 | 27.60% | 33.83% | 41.43% | 0.372 |
| Seasonal Naive | 91.06 | 108.89 | 30.42% | 37.57% | 47.01% | 0.391 |

**Key Findings:**
- **Best Accuracy**: ARIMA(0,1,0) with MAE=83.98, RMSE=103.55 (same as Naive baseline)
- **Best Direction**: Seasonal Naive with MDA=47.01% (slightly better than random 50%)
- **Theil's U**: 0.372 (37% of naive RMSE) - shows these models explain 63% more variance than naive
- **Surprising Result**: ARIMA models provide NO improvement over naive random walk
- **Seasonal Effect**: Weak (Seasonal Naive adds 4.6% to MAE)

### Error Distribution Analysis

**Best Model: ARIMA(0,1,0)**
- Mean Error: $83.91 (consistent overprediction)
- Error Std Dev: $60.68
- Min Error: -$3.45 | Max Error: $228.87
- Median Error: $65.12
- Skewness: 0.62 (right-skewed, occasional large negative errors)
- Kurtosis: -0.59 (flatter than normal, fewer extreme outliers)

**Error Bounds:**
- Within ±1σ: 47.1% of errors (below normal 68%)
- Within ±2σ: 75.7% of errors (below normal 95%)

**Interpretation:** Errors have slightly heavier tails than normal distribution; larger errors occur more frequently than expected.

## Summary

### 10 Essential Metrics

1. **MAE**: Average absolute error (interpretable, robust)
2. **RMSE**: Root mean squared error (penalizes outliers)
3. **MAPE**: Mean absolute percentage error (scale-independent)
4. **SMAPE**: Symmetric MAPE (fixes MAPE issues)
5. **MDA**: Mean directional accuracy (direction-focused)
6. **Theil's U**: Normalized vs naive baseline
7. **ME**: Mean bias (systematic error direction)
8. **MPE**: Mean percentage bias
9. **Error Std**: Volatility of errors
10. **Max Error**: Worst-case prediction error

### Best Practices

1. **Always use multiple metrics**: Different metrics reveal different aspects
2. **Compare to baseline**: Naive is minimum requirement
3. **Show confidence**: Report error bounds (±1σ, ±2σ)
4. **Visualize errors**: Time series plots reveal patterns
5. **Report bias**: Include signed errors (ME, MPE)
6. **Context matters**: Financial vs academic vs operational
7. **Transparent reporting**: Show all metrics, not just best one

### Gold Price Insights

- **No ARIMA Advantage**: All ARIMA variants match naive random walk (d=1 makes them equivalent)
- **Directional Accuracy Low**: 41-47% MDA suggests insufficient signal for profitable trading
- **Consistent Overprediction**: ME=$83.91 indicates systematic bias (forecasts too high)
- **Fat Tails**: Error distribution more heavy-tailed than normal (kurtosis=-0.59)
- **Next Step**: Consider GARCH for volatility, exogenous variables, or ensemble methods

---

## References

- Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? Geoscientific Model Development Discussions.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting.
- Makridakis, S. (1993). Accuracy measures: critical and practical considerations.
