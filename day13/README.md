# Day 13: ARMA Models

## Overview

ARMA (AutoRegressive Moving Average) models combine the power of AR and MA components to capture more complex time series patterns. Today we explore how to combine these two approaches and use ACF/PACF patterns for model identification.

**Key Concepts:**
- ARMA(p,q): Combines AR(p) and MA(q) components
- AR component: Models dependency on past values
- MA component: Models dependency on past forecast errors  
- ACF/PACF patterns for model identification
- Information criteria (AIC/BIC) for model selection

## Learning Objectives

By the end of this analysis, you will understand:

1. **ARMA Model Structure**
   - How AR and MA components work together
   - Mathematical formulation of ARMA(p,q)
   - When ARMA is superior to pure AR or MA

2. **Model Identification**
   - Using ACF for MA order selection
   - Using PACF for AR order selection
   - Interpreting mixed ACF/PACF patterns

3. **Parameter Estimation**
   - Maximum likelihood estimation
   - Coefficient interpretation
   - Model diagnostics

4. **Model Selection**
   - Comparing ARMA models with different orders
   - Using AIC and BIC criteria
   - Residual analysis for model adequacy

## Dataset

**Gold Price Data (2015-2026)**
- Source: Yahoo Finance (GLD)
- Frequency: Daily
- Observations: 2,781 daily prices
- Transformation: First differencing for stationarity
- Train/Test Split: 80/20 (2,224 / 556 observations)

## Methodology

### 1. Stationarity Check

**ADF Test on Differenced Series:**
- Test Statistic: -6.144
- P-value: < 0.001
- **Result: STATIONARY** ✓

First differencing successfully converts the non-stationary gold price series into a stationary price change series suitable for ARMA modeling.

### 2. ACF and PACF Analysis

**ACF (Autocorrelation Function):**
- Lag 1: -0.0126 (weak negative)
- Lag 2: +0.0351 (weak positive)
- Lag 3: -0.0127 (weak negative)
- Lag 4: -0.0431 (moderate negative) - **Significant**
- 95% Confidence Interval: ±0.0372

**Significant ACF lags:** 4, 8, 13, 14, 16, 18

**PACF (Partial Autocorrelation Function):**
- Lag 1: -0.0126 (weak negative)
- Lag 2: +0.0350 (weak positive)
- Lag 3: -0.0119 (weak negative)
- Lag 4: -0.0447 (moderate negative) - **Significant**
- 95% Confidence Interval: ±0.0372

**Significant PACF lags:** 4, 12, 13, 14, 16, 18

**Initial Suggestions:**
- AR order (p): 4 (from PACF first cutoff)
- MA order (q): 4 (from ACF first cutoff)

### 3. Model Fitting and Comparison

ARMA models with various (p,q) combinations were fitted using maximum likelihood estimation:

| Model | AIC | BIC | RMSE |
|-------|-----|-----|------|
| AR(1) = ARMA(1,0) | 18,118.06 | 18,135.18 | 36.7438 |
| AR(2) = ARMA(2,0) | 18,115.85 | 18,138.68 | 36.7435 |
| MA(1) = ARMA(0,1) | **18,117.91** | **18,135.04** ✓ | 36.7438 |
| MA(2) = ARMA(0,2) | 18,115.71 | 18,138.54 | 36.7436 |
| ARMA(1,1) | 18,117.31 | 18,140.13 | 36.7435 |
| ARMA(2,1) | 18,117.27 | 18,145.80 | 36.7436 |

**Optimal Model: MA(1) = ARMA(0,1)** (by BIC criterion)

### 4. Optimal Model: ARMA(0,1) = MA(1)

The BIC criterion selected **MA(1)** as the optimal model, which is interesting because it's a pure Moving Average model rather than a mixed ARMA model.

**Model Parameters:**
- **Constant (c):** 0.3399
- **MA Coefficient (θ₁):** 0.0289
- **Sigma² (error variance):** 201.61

**Model Formula:**
$$y_t = 0.3399 + \epsilon_t + 0.0289 \cdot \epsilon_{t-1}$$

Where:
- $y_t$ = Price change at time t
- $\epsilon_t$ = Forecast error at time t
- $\theta_1 = 0.0289$ = Weight given to previous forecast error

**Interpretation:**
- The MA(1) coefficient of +0.0289 indicates a small positive relationship with the previous forecast error
- The constant term suggests an average daily price increase of about $0.34
- The small MA coefficient reflects weak predictability in daily gold price changes

### 5. Model Diagnostics

**In-Sample Residuals (Training):**
- Mean: 0.000071 ≈ 0 ✓ (unbiased)
- Std Dev: 14.20
- Range: [-111.16, 74.75]

**Out-of-Sample Residuals (Test):**
- Mean: 4.98
- Std Dev: 36.44
- Range: [-280.34, 171.66]

**Residual Autocorrelation:**
- Significant lags remaining: 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19
- ⚠️ **Issue**: Substantial autocorrelation remains in residuals

This suggests the MA(1) model may be inadequate for capturing all the serial correlation in the differenced gold price series, though it was selected as optimal by BIC.

## Results

### Forecast Performance (Test Set: 556 observations)

**MA(1) = ARMA(0,1) Model:**
- **RMSE: 36.74**
- **MAE: 25.10**
- **Naive RMSE: 36.41**
- **Improvement: -0.91%** ❌

**Key Finding:** The MA(1) model actually **underperforms** the naive forecast by 0.91%, suggesting that daily gold price changes are extremely difficult to predict and behave like a near-random walk.

**Comparison Across Models:**
All ARMA models tested showed nearly identical performance (RMSE ≈ 36.74), with slight underperformance relative to the naive forecast. This indicates:
1. Weak autocorrelation in differenced series
2. Random walk behavior in gold prices
3. Limited predictability from historical patterns

## Interpretation

### Why MA(1) Over More Complex Models?

Despite suggesting ARMA(4,4) from ACF/PACF, BIC selected the simpler MA(1) model because:

1. **Parsimony Principle:** BIC strongly penalizes model complexity
2. **Weak Signals:** The small improvements from additional parameters don't justify added complexity
3. **Overfitting Prevention:** Complex models risk overfitting noise rather than signal

### ARMA Model Selection Guidelines

| ACF Pattern | PACF Pattern | Suggested Model |
|-------------|--------------|-----------------|
| Cuts off after lag q | Decays gradually | **MA(q)** |
| Decays gradually | Cuts off after lag p | **AR(p)** |
| Decays gradually | Decays gradually | **ARMA(p,q)** |
| Few significant lags | Few significant lags | **Low-order ARMA** |

**For Gold Prices:**
- Both ACF and PACF show weak, scattered significant lags
- No clear cutoff pattern → Suggests random walk behavior
- MA(1) captures the dominant (though weak) pattern

### Financial Implications

The inability of ARMA models to beat naive forecasts for daily gold prices is consistent with:

1. **Efficient Market Hypothesis:** Prices already reflect available information
2. **Random Walk Theory:** Daily price changes are largely unpredictable
3. **High Noise-to-Signal Ratio:** Daily returns dominated by random shocks

## Key Insights

### 1. ARMA Model Formula

**General Form:**
$$y_t = c + \phi_1 y_{t-1} + \cdots + \phi_p y_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}$$

**Our MA(1) Model:**
$$y_t = 0.3399 + \epsilon_t + 0.0289 \cdot \epsilon_{t-1}$$

Where:
- **AR terms** ($\phi_i y_{t-i}$): Use past values directly
- **MA terms** ($\theta_j \epsilon_{t-j}$): Use past forecast errors
- **Constant** (c): Base level or drift term

### 2. Model Identification Strategy

```
Step 1: Check Stationarity (ADF test)
  └─ If non-stationary → Difference the series

Step 2: Examine ACF and PACF
  ├─ ACF cutoff at q, PACF decays → MA(q)
  ├─ PACF cutoff at p, ACF decays → AR(p)
  └─ Both decay → ARMA(p,q)

Step 3: Fit Multiple Models
  └─ Compare using AIC/BIC

Step 4: Validate with Residual Diagnostics
  └─ Check for remaining autocorrelation
```

### 3. When to Use ARMA vs AR/MA

| Model Type | Best For | Example Use Case |
|------------|----------|------------------|
| **AR(p)** | Direct dependency on past values | Temperature persistence |
| **MA(q)** | Dependency on past shocks/errors | Inventory shocks |
| **ARMA(p,q)** | Mixed patterns | Economic indicators |

**For Financial Returns:**
- Typically MA(1) or MA(2) sufficient
- AR components less common (due to efficient markets)
- ARMA rarely needed for daily frequencies

### 4. Limitations Observed

**Why ARMA Failed to Beat Naive:**

1. **Weak Autocorrelation:** θ₁ = 0.0289 is very small
2. **Random Walk Behavior:** Gold prices follow approximate random walk
3. **High Volatility:** σ² = 201.61 indicates large random shocks
4. **Efficient Markets:** Historical patterns quickly arbitraged away

**Residual Autocorrelation:**
- 15 significant lags in test residuals
- Indicates MA(1) incomplete
- But more complex models don't improve forecasts (due to overfitting)

## Comparison with Previous Methods

| Day | Method | Components | Gold Price RMSE | Improvement |
|-----|--------|-----------|-----------------|-------------|
| 9 | SES | Level | - | - |
| 10 | Holt | Level + Trend | 180.93 | -353.85% |
| 11 | Holt-Winters | Level + Trend + Season | 1098.08 | +10.8% |
| 12 | AR(10) | Past values | 35.60 | +10.69% |
| **13** | **MA(1)** | **Past errors** | **36.74** | **-0.91%** |

**Key Observation:** AR(10) from Day 12 performed better (35.60 vs 36.74 RMSE) than MA(1), suggesting past values contain more information than past errors for gold prices.

## Progression of ARIMA Family

```
Day 12: AR(p) - Autoregressive
        └─ Uses: Past values
        └─ Formula: y(t) = c + Σφᵢy(t-i) + εₜ
        
Day 13: ARMA(p,q) - AutoRegressive Moving Average ← YOU ARE HERE
        └─ Uses: Past values + Past errors
        └─ Formula: y(t) = c + Σφᵢy(t-i) + εₜ + Σθⱼε(t-j)
        
Day 14: ARIMA(p,d,q) - Integrated ARMA (Coming Next)
        └─ Uses: Differencing + ARMA
        └─ Handles: Non-stationary series directly
```

## Next Steps

Day 14 will introduce **ARIMA Models**, which incorporate differencing directly into the model:
- Automatic differencing selection
- Box-Jenkins methodology
- Complete model identification workflow
- Auto-ARIMA for automated order selection

## Conclusion

ARMA models provide a flexible framework combining AR and MA components. For gold price changes, the optimal model turned out to be a simple MA(1), reflecting:

1. **Weak Predictability:** MA coefficient of 0.0289 indicates minimal dependency on past errors
2. **Random Walk Dominance:** -0.91% performance vs naive confirms near-random walk behavior
3. **Model Selection Trade-offs:** BIC correctly favored simplicity over marginal complexity gains

The failure of ARMA to beat naive forecasts for daily gold prices reinforces the efficient market hypothesis and highlights the challenge of short-term financial prediction. This finding is typical for liquid financial markets where information is quickly incorporated into prices.

**Practical Takeaway:** For daily gold price changes, simple models like MA(1) are sufficient, though even they struggle to beat naive forecasts. Longer forecast horizons or alternative approaches (volatility modeling, fundamental analysis) may be more productive.

## Files

- `main.py` - Complete ARMA analysis script
- `notebooks/main.ipynb` - Interactive Jupyter notebook
- `README.md` - This documentation with actual results
