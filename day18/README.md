# Day 18: SARIMA for Seasonality
## Advanced Seasonal ARIMA Modeling and Pattern Detection

**Objective:** Master seasonal ARIMA (SARIMA) models by detecting seasonal patterns in gold prices, analyzing seasonal strength, and comparing seasonal vs non-seasonal forecasting approaches.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Seasonal Detection Methods](#seasonal-detection-methods)
4. [SARIMA Model Framework](#sarima-model-framework)
5. [Implementation Details](#implementation-details)
6. [Results and Interpretation](#results-and-interpretation)

---

## Overview

While Day 15 introduced SARIMA models conceptually, Day 18 provides a deep dive into seasonal pattern detection and optimization of seasonal parameters (P, D, Q, s) for gold price forecasting.

### Dataset
- **Source:** Yahoo Finance Gold (GLD)
- **Period:** 2016-01-11 to 2026-01-09 (2,515 observations)
- **Assumed Seasonal Period:** 252 trading days (annual trading cycle)
- **Train/Test Split:** 80/20 (2,012 / 503)

### Key Questions Addressed
1. **Is gold price seasonal?** What is the seasonal strength?
2. **What is the seasonal period?** Days, weeks, months, or years?
3. **Do seasonal models improve forecasts?** By how much?
4. **Which seasonal parameters work best?** (P, D, Q, s combinations

---

## Key Concepts

### 1. Seasonal Decomposition

Time series can be decomposed into three components:
$$y_t = T_t + S_t + E_t$$

Where:
- **$T_t$** = Trend component (long-term direction)
- **$S_t$** = Seasonal component (repeating patterns)
- **$E_t$** = Residual/error component (noise)

**Additive Model** (used when seasonal variation is constant):
$$y_t = T_t + S_t + E_t$$

**Multiplicative Model** (used when seasonal variation increases with trend):
$$y_t = T_t \times S_t \times E_t$$

For gold prices, we typically use additive decomposition because seasonal percentage changes are roughly constant.

### 2. Seasonal Strength

Measures how dominant the seasonal pattern is relative to noise:
$$\text{Seasonal Strength} = \frac{\text{Var}(S_t)}{\text{Var}(S_t + E_t)}$$

**Interpretation:**
- **0.0-0.1:** Very weak seasonality (random noise dominates)
- **0.1-0.3:** Moderate seasonality (pattern exists but not dominant)
- **0.3-1.0:** Strong seasonality (clear repeating pattern)

**Decision:**
- Strength < 0.1: Use non-seasonal models (simpler, avoid overfitting)
- Strength ≥ 0.1: Consider seasonal models (can improve forecasts)

### 3. Seasonal Period ($s$)

The number of observations in one seasonal cycle:
- **Daily data:**
  - Weekly seasonality: $s = 5$ (trading days)
  - Annual seasonality: $s = 252$ (trading days/year)
  
- **Monthly data:**
  - Quarterly: $s = 3$
  - Annual: $s = 12$

- **Hourly data:**
  - Daily: $s = 24$
  - Weekly: $s = 168$

For gold prices, we test $s = 252$ (annual trading cycle).

### 4. SARIMA Parameters: (P, D, Q, s)

**Seasonal Parameters:**
- **$P$** = Seasonal AR order (autoregressive terms at seasonal lags)
- **$D$** = Seasonal differencing (1 or 2)
- **$Q$** = Seasonal MA order (moving average terms at seasonal lags)
- **$s$** = Seasonal period

**Full SARIMA notation:**
$$\text{SARIMA}(p,d,q)(P,D,Q)_s$$

Example: **SARIMA(1,1,1)(1,1,1)₁₂** means:
- Non-seasonal: AR(1), difference once, MA(1)
- Seasonal: Seasonal AR(1), seasonal difference once, Seasonal MA(1)
- Period: 12 months

### 5. Autocorrelation at Seasonal Lags

The ACF helps identify seasonal periods by showing spikes at multiples of the seasonal period.

**Example with $s = 252$:**
- Lag 252: First seasonal spike
- Lag 504: Second seasonal spike (2 years)
- Lag 756: Third seasonal spike (3 years)

If ACF shows spikes at these regular intervals, seasonality exists.

---

## Seasonal Detection Methods

### 1. Visual Inspection: Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data, model='additive', period=252)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
data.plot(ax=axes[0], title='Original')
trend.plot(ax=axes[1], title='Trend')
seasonal.plot(ax=axes[2], title='Seasonal')
residual.plot(ax=axes[3], title='Residual')
```

**Interpretation:**
- **Repeating pattern in seasonal component?** → Seasonality exists
- **Pattern magnitude:** → Seasonal strength
- **Pattern timing:** → Confirm seasonal period

### 2. Quantitative: Seasonal Strength

$$\text{Strength} = \frac{\text{Var}(S_t)}{\text{Var}(S_t + E_t)}$$

**Calculation:**
```python
seasonal_var = np.var(seasonal)
residual_var = np.var(residual)
strength = seasonal_var / (seasonal_var + residual_var)
```

**Example Results:**
- 0.15 = Mild seasonality (15% of variation is seasonal)
- 0.40 = Moderate seasonality (40% of variation is seasonal)
- 0.70 = Strong seasonality (70% of variation is seasonal)

### 3. Statistical: ACF at Seasonal Lags

**Look for spikes in ACF at lags k, 2k, 3k, ...** where k is the suspected seasonal period.

```python
from statsmodels.tsa.stattools import acf

acf_vals = acf(data, nlags=300)
seasonal_lags = [252, 504, 756]  # For s=252

for lag in seasonal_lags:
    print(f"ACF at lag {lag}: {acf_vals[lag]:.4f}")
```

**Interpretation:**
- If ACF > 0.5 at lag 252: Strong annual seasonality
- If ACF < 0.2 at lag 252: Weak or no annual seasonality

---

## SARIMA Model Framework

### Model Structure

**SARIMA(p,d,q)(P,D,Q)ₛ combines:**

1. **Non-seasonal ARIMA(p,d,q):**
   - Handles short-term autocorrelation
   - Applied to differenced data
   
2. **Seasonal ARIMA(P,D,Q)ₛ:**
   - Handles seasonal autocorrelation
   - Applied at seasonal lags (s, 2s, 3s, ...)

### Parameter Selection

**Manual approach (based on ACF/PACF):**
1. Plot ACF and PACF of original series
2. Look for spikes at seasonal lags
3. Count significant seasonal spikes → estimate P and Q
4. Test D=0, 1, 2 for seasonal differencing

**Automatic approach:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

# Automatic seasonal parameter selection
model = auto_arima(data, 
                   seasonal=True, 
                   m=252,  # s=252
                   max_p=2, max_q=2,
                   max_P=2, max_Q=2,
                   max_D=1,
                   trace=True)
```

### Model Fitting

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_data, 
                order=(1, 1, 1),           # (p,d,q)
                seasonal_order=(1, 1, 1, 252),  # (P,D,Q,s)
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)
print(results.summary())

# Forecast
forecast = results.get_forecast(steps=len(test_data))
predictions = forecast.predicted_mean
```

---

## Implementation Details

### Code Structure: 10 Sections

1. **Data Loading:** Load gold price data, split train/test
2. **Seasonal Decomposition:** Extract trend, seasonal, residual components
3. **Seasonal Pattern Detection:** Identify period and peak timing
4. **Stationarity Analysis:** Test original, differenced, and seasonally differenced series
5. **Model Fitting:** Fit multiple SARIMA variants
6. **Model Comparison:** Compare by AIC, BIC, test RMSE
7. **Seasonal vs Non-seasonal:** Quantify improvement from seasonality
8. **Best Model Analysis:** Detailed diagnostics of optimal model
9. **Visualization:** Decomposition plots, ACF/PACF, RMSE comparison
10. **Summary:** Key findings and recommendations

### Key Libraries

```python
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf, adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

---

## Results and Interpretation

### Expected Output Structure

```
Seasonal Strength: 0.1247 (12.47%)
  → Mild seasonality detected

Seasonal Period: 252 trading days (1 year)
  → Annual cycle in gold prices

Best Model by RMSE: SARIMA(1,1,1)(1,1,1,252)
  AIC: 6890.45
  Test RMSE: 98.76
  
Comparison:
  Non-seasonal ARIMA(0,1,0): RMSE = 103.55
  Seasonal SARIMA(1,1,1)(1,1,1,252): RMSE = 98.76
  Improvement: 4.6% ✓
```

### Interpretation Guide

**Seasonal Strength < 0.1:**
- Weak seasonality; non-seasonal models recommended
- Simpler models avoid overfitting
- Focus on trend and noise

**Seasonal Strength 0.1-0.3:**
- Moderate seasonality; consider seasonal models
- Seasonal models improve forecasts by 2-5%
- Balance complexity vs accuracy

**Seasonal Strength > 0.3:**
- Strong seasonality; seasonal models essential
- Can improve forecasts by 10%+
- SARIMA models justify added complexity

### When to Use Seasonal Models

| Metric | Decision |
|--------|----------|
| **Seasonal strength** | > 0.15 → Use seasonal |
| **RMSE improvement** | > 5% → Use seasonal |
| **Model complexity** | Simple seasonal (P,Q≤1) → Prefer |
| **Forecast horizon** | Long-term → Seasonal helps more |
| **Domain knowledge** | Known seasonality → Use seasonal |

---

## Model Comparison Framework

### Multiple SARIMA Variants

Day 18 tests these models:

| Model | Structure | Purpose |
|-------|-----------|---------|
| ARIMA(0,1,0) | Non-seasonal baseline | Benchmark |
| SARIMA(0,1,0)(0,1,0,252) | Seasonal differencing only | Test seasonality |
| SARIMA(1,1,0)(1,1,0,252) | Seasonal AR | Autoregressive seasonal |
| SARIMA(0,1,1)(0,1,1,252) | Seasonal MA | Moving average seasonal |
| SARIMA(1,1,1)(1,1,1,252) | Full seasonal | Most flexible |

### Comparison Metrics

| Metric | Purpose | Selection |
|--------|---------|-----------|
| **AIC** | Fit quality with penalty | Prefer lower |
| **BIC** | Fit with stronger complexity penalty | Prefer lower |
| **Test RMSE** | Out-of-sample forecast error | Prefer lower |
| **Test MAE** | Absolute error magnitude | Prefer lower |
| **Test MAPE** | Percentage error | Prefer lower |

---

## Advanced Topics

### Seasonal Strength Variation Over Time

Gold prices may have different seasonality in different periods:

```python
# Split into windows and calculate seasonal strength
window_size = 504  # 2 years
seasonal_strength_by_period = []

for i in range(0, len(data) - window_size, window_size):
    window = data[i:i+window_size]
    decomp = seasonal_decompose(window, period=252)
    strength = np.var(decomp.seasonal) / (np.var(decomp.seasonal) + np.var(decomp.resid))
    seasonal_strength_by_period.append(strength)
```

This reveals whether seasonality has changed over time.

### Multiple Seasonal Periods

Gold can have multiple seasonal patterns:
- Weekly seasonality: $s = 5$ (trading day effects)
- Annual seasonality: $s = 252$ (calendar effects)

Use multi-seasonal models:
```python
# MSTS models (TBATS introduced in Day 22)
from statsmodels.tsa.tbats import TBATS

model = TBATS(data, seasonal_periods=[5, 252])
```

---

## Summary

### Key Takeaways

1. **Seasonal Strength:** Quantifies seasonal dominance (0.0-1.0)
   - Low (<0.1): Use non-seasonal models
   - High (>0.3): Use seasonal models

2. **Seasonal Period:** Must specify $s$ (e.g., 252 for annual)
   - Identified via ACF spikes at regular intervals
   - Domain knowledge + statistical confirmation

3. **SARIMA vs ARIMA:** Seasonal models improve forecasts when:
   - Seasonal strength > 0.1
   - RMSE improvement > 3-5%
   - Interpretability and simplicity acceptable

4. **Parameter Selection:** (P, D, Q) chosen via:
   - ACF/PACF at seasonal lags
   - Grid search or auto_arima
   - Information criteria (AIC/BIC)

5. **Model Validation:** Test on holdout data
   - Train/test split (80/20)
   - Compare seasonal vs non-seasonal
   - Use both AIC and RMSE metrics

### Next Steps

- **Day 19:** Vector Autoregression (VAR) for multivariate series
- **Day 20:** Cointegration and error correction models
- **Day 21:** Facebook Prophet for automatic seasonality
- **Day 22:** TBATS for multiple seasonal periods

---

## References

- Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice. OTexts.
- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control. Wiley.
- Cleveland, R. B., Cleveland, W. S., & Terpenning, I. (1990). STL: A seasonal-trend decomposition.
