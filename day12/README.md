# Day 12: Autoregressive (AR) Models

## Overview

Autoregressive (AR) models form the foundation of time series analysis. They predict future values based on past values of the series itself. Today we explore AR process fundamentals, order selection techniques, and practical implementation.

**Key Concepts:**
- AR(p): Autoregressive model of order p
- PACF: Partial Autocorrelation Function for order selection
- Stationarity: Required assumption for AR models
- Differencing: Transformation to achieve stationarity

## Learning Objectives

By the end of this analysis, you will understand:

1. **AR Model Fundamentals**
   - How autoregressive models work mathematically
   - The AR(p) formula and coefficient interpretation
   - Why stationarity is critical

2. **Order Selection**
   - Using PACF plots to determine optimal AR order
   - Comparing AIC and BIC criteria
   - Trade-offs between model complexity and fit

3. **Coefficient Interpretation**
   - What each AR coefficient represents
   - Positive vs negative autocorrelation
   - Confidence intervals for coefficients

4. **Model Evaluation**
   - Comparing AR models of different orders
   - Residual diagnostics
   - Forecast accuracy assessment

## Dataset

**Gold Price Data (2015-2026)**
- Source: Yahoo Finance (GLD)
- Frequency: Daily
- Observations: 2,777 daily prices
- Transformation: First differencing for stationarity

## Methodology

### 1. Stationarity Testing

First, we test whether the original gold price series is stationary using the Augmented Dickey-Fuller (ADF) test.

**Results:**
- Original Series ADF Test: Test Statistic = -0.7832 (p-value > 0.05) → Non-stationary
- First Difference ADF Test: Test Statistic = -13.2847 (p-value < 0.001) → Stationary ✓

**Conclusion:** First differencing makes the series stationary, suitable for AR modeling.

### 2. Autocorrelation Analysis

**ACF (Autocorrelation Function):**
- Shows correlation between observations at different lags
- ACF at lag 1: -0.0298 (very weak)
- Decays quickly → Suggests differencing was successful

**PACF (Partial Autocorrelation Function):**
- Removes intermediate lag effects to show direct dependencies
- Significant spikes at lags: 4, 7, 9, 18
- First cutoff at lag 4 → Suggests AR(4) minimum order

### 3. Data Preparation

- **Training Set:** 2,220 observations (80% of differenced data)
- **Test Set:** 556 observations (20% of differenced data)
- **Period:** 2015-01-05 to 2023-10-27 (train) | 2023-10-30 to 2026-01-16 (test)

### 4. Model Fitting

AR models with orders p = {1, 2, 3, 5, 7, 10} were fitted using conditional maximum likelihood.

**Information Criteria Comparison:**

| AR Order | AIC | BIC | Best |
|----------|-----|-----|------|
| 1 | 18,078.95 | 18,096.07 | - |
| 2 | 18,068.85 | 18,091.67 | - |
| 3 | 18,062.36 | 18,090.88 | - |
| 5 | 18,048.43 | 18,088.36 | - |
| 7 | 18,034.74 | 18,086.06 | - |
| 10 | 18,010.50 | 18,078.91 | ✓ Both |

**Optimal Order:** AR(10) by both AIC and BIC criteria

### 5. AR(10) Model Parameters

**Coefficients (φᵢ):**
- Constant (c): 0.3296
- φ₁ (lag 1): +0.0270 - Small positive autocorrelation at t-1
- φ₂ (lag 2): -0.0409 - Moderate negative autocorrelation at t-2
- φ₃ (lag 3): +0.0199 - Weak positive autocorrelation at t-3
- φ₄ (lag 4): +0.0033 - Negligible at t-4
- φ₅ (lag 5): -0.0372 - Moderate negative at t-5
- φ₆ (lag 6): -0.0121 - Weak negative at t-6
- φ₇ (lag 7): -0.0294 - Moderate negative at t-7
- φ₈ (lag 8): +0.0147 - Weak positive at t-8
- φ₉ (lag 9): +0.0218 - Weak positive at t-9
- φ₁₀ (lag 10): -0.0221 - Weak negative at t-10

**Interpretation:**
- Most coefficients are small (<±0.05), suggesting weak autocorrelation
- The differenced series has limited dependency on past values
- This is typical for financial price changes (random walk property)

## Results

### Forecast Performance (Test Set: 556 observations)

**AR(10) Model:**
- RMSE: **35.60**
- MAE: **24.46**
- Naive RMSE: 39.87
- **Improvement over Naive: 10.69%** ✓

The AR(10) model outperforms the naive forecast (using last observed value) by approximately 10.69%, demonstrating that historical patterns capture some predictable information about price changes.

### Model Diagnostics

**Residual Properties:**
- Mean: ~0 (unbiased predictions)
- Standard Deviation: ~14.16
- Distribution: Approximately normal with slight skewness

**Key Finding:** 
Residuals show minimal autocorrelation, suggesting the AR(10) model has captured the main autocorrelative structure in the differenced series.

## Interpretation

### What Does AR(10) Tell Us?

The AR(10) model reveals:

1. **Weak Autocorrelation:** All coefficients are small (< 0.05), indicating gold price changes have weak dependency on past values
2. **Multi-lag Effects:** The significant lags are distributed (2, 5, 7), suggesting various time horizons influence current changes
3. **Random Walk Component:** Small improvement (10.69%) vs naive suggests a near-random walk behavior typical of efficient markets
4. **Practical Utility:** While AR(10) beats naive, the modest improvement suggests limited predictability in daily gold prices

### Why AR(10) and Not Lower Orders?

- **AR(1):** Too simple, misses complex dynamics
- **AR(5):** BIC = 18,088.36 (higher than AR(10))
- **AR(10):** Optimal by both AIC and BIC
- **AR(15+):** Would add complexity without meaningful improvement

## Key Insights

### 1. AR Model Fundamentals
$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t$$

- Current value is a linear combination of past values plus noise
- Each coefficient shows the weight given to that lag's contribution
- Assumes: Stationarity, linearity, constant parameters

### 2. Order Selection Strategy

| Method | Process | Advantage |
|--------|---------|-----------|
| **PACF** | Look for cutoff point | Visual, intuitive |
| **AIC** | Minimize AIC values | Balances fit and complexity |
| **BIC** | Minimize BIC values | Stronger penalty (preferred) |

**For this data:** BIC suggested AR(10) as optimal

### 3. Stationarity Requirement

- AR models require stationary series
- Non-stationary gold prices needed first differencing
- Differencing creates stationary price changes suitable for AR

### 4. Coefficient Interpretation

| Sign | Meaning |
|------|---------|
| **+0.027 (lag 1)** | Positive change today slightly increases tomorrow's change |
| **-0.041 (lag 2)** | Change 2 days ago has small negative effect on today |
| **-0.037 (lag 5)** | Week-old changes show negative feedback |

Small coefficients indicate weak predictability.

### 5. When to Use AR Models

**✓ Good for:**
- Stationary time series
- Recent history affects future values
- Differenced financial returns
- Short-term forecasting

**✗ Not suitable for:**
- Trending (non-stationary) series
- Strong seasonal patterns
- Structural breaks or regime changes
- Very short series (need p+1 observations minimum)

## Progression of Exponential Smoothing Family

```
Day 9:  Simple Exponential Smoothing (SES)
        → Level only
        → Flat forecasts for all horizons
        
Day 10: Holt's Linear Trend
        → Level + Trend
        → Trending forecasts
        
Day 11: Holt-Winters Seasonal
        → Level + Trend + Seasonality
        → Trending + Seasonal forecasts

Day 12: Autoregressive (AR) ← YOU ARE HERE
        → Based on past values of the series
        → Alternative approach to exponential smoothing
```

## Comparison with Previous Methods

| Method | Components | Best For | Day |
|--------|-----------|----------|-----|
| SES | Level only | Stationary, no trend | 9 |
| Holt | Level + Trend | Trending series | 10 |
| H-W Seasonal | Level + Trend + Season | Seasonal data | 11 |
| AR | Autocorrelation | Stationary, weak seasonality | 12 |

## Next Steps

Day 13 will introduce the **Moving Average (MA) Model**, completing the ARIMA family:
- How MA models work
- Combining AR and MA: ARIMA models
- Automatic order selection using auto_arima

## Conclusion

AR models provide a powerful framework for modeling stationary time series based on their own history. For the gold price differenced series, AR(10) captures meaningful autocorrelative patterns, improving forecasts by 10.69% over naive methods. However, the modest improvement reflects the challenging nature of financial market prediction and the approximate random walk behavior of commodity prices.

The small AR coefficients suggest that while historical information is somewhat useful, daily gold price changes are largely unpredictable—a finding consistent with financial market efficiency.

## Files

- `main.py` - Complete analysis script
- `notebooks/main.ipynb` - Interactive Jupyter notebook
- `README.md` - This documentation
