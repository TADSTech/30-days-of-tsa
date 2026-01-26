# Day 14: ARIMA Models - Integrated Autoregressive Moving Average

## Objective

Build and evaluate ARIMA models that handle both non-stationary data and autocorrelation patterns through automatic differencing and combined AR/MA components.

## Key Deliverables

- Non-stationarity testing and differencing analysis
- ACF/PACF analysis for (p,d,q) parameter selection
- ARIMA models with 6 different configurations
- Model selection using AIC and BIC criteria
- Comprehensive residual diagnostics
- Forecast performance evaluation and comparison

## Dataset

- **Source**: Yahoo Finance Gold (GLD)
- **Period**: 2015-01-02 to 2022-08-13
- **Observations**: 2,781 daily price points
- **Train-Test Split**: 80/20 (2,224 training, 557 test observations)

## Key Findings

### 1. Stationarity Analysis

**Original Series (Non-Stationary)**:
- ADF Test Statistic: -0.316
- P-value: 0.923
- Result: ❌ Non-stationary (cannot use AR/MA directly)

**First Difference (d=1)**:
- ADF Test Statistic: -52.121
- P-value: 0.000000
- Result: ✅ Stationary
- **Recommendation**: d=1 is optimal

**Second Difference (d=2)**:
- ADF Test Statistic: -17.516
- P-value: 0.000000
- Result: ✅ Stationary (but over-differencing not needed)

### 2. ACF/PACF Analysis

**Confidence Interval (95%)**: ±0.0372

| Pattern | Lag 1-5 Values | Interpretation |
|---------|---|---|
| ACF | [0.012, -0.014, -0.020, -0.013, 0.013] | No significant lags → Suggests MA(0) |
| PACF | No significant lags | No significant lags → Suggests AR(0) |

**Suggested Orders**: p=0, d=1, q=0 (Random Walk Model)

### 3. ARIMA Model Comparison

| Model | AIC | BIC | RMSE | Status |
|-------|-----|-----|------|--------|
| ARIMA(1,1,0) | 21043.55 | 21055.41 | 160.66 | ✓ Competitive |
| ARIMA(2,1,0) | 21045.02 | 21062.81 | 160.96 | - |
| **ARIMA(0,1,1)** | **21043.54** | **21055.40** | **160.66** | **✓ OPTIMAL** |
| ARIMA(0,1,2) | 21045.04 | 21062.83 | 160.95 | - |
| ARIMA(1,1,1) | 21045.38 | 21063.17 | 160.72 | - |
| ARIMA(2,1,1) | 21046.36 | 21070.08 | 161.57 | - |

**Optimal Model**: ARIMA(0, 1, 1)
- **Model Type**: Random Walk with Moving Average error correction
- **AIC**: 21043.54 (best)
- **BIC**: 21055.40 (best)
- **Parsimony**: Simplest model with lowest information criteria

### 4. Optimal ARIMA(0,1,1) Model Details

#### Model Equation
$$y_t = y_{t-1} + \epsilon_t + 0.0125 \cdot \epsilon_{t-1}$$

Where:
- $y_t$ = Gold price at time t
- $\epsilon_t$ = Random noise/error term
- θ₁ = 0.0125 (MA coefficient, very small)

#### Parameter Interpretation

| Parameter | Value | Std Error | P-value | Interpretation |
|-----------|-------|-----------|---------|-----------------|
| ma.L1 | 0.0125 | 0.019 | 0.515 | Not statistically significant |
| σ² | 113.33 | 3.111 | <0.001 | Error variance (highly significant) |

#### Model Diagnostics

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Ljung-Box Q | 0.00 | ✅ No autocorrelation (p=0.97) |
| Jarque-Bera | 0.96 | ✅ Residuals normally distributed (p=0.62) |
| Heteroskedasticity | 1.00 | ✅ Constant variance (p=0.96) |
| Skewness | -0.00 | ✅ Symmetric distribution |
| Kurtosis | 2.91 | ✅ Normal tail behavior |

### 5. Forecast Performance

#### Test Set Evaluation
- **ARIMA(0,1,1) RMSE**: 160.66
- **Naive Forecast RMSE**: 221.65
- **Improvement**: +27.52% ✓

#### Error Metrics
- **MAE** (Mean Absolute Error): 143.31
- **MAPE** (Mean Absolute Percentage Error): 6.72%
- Typical forecast error: $143-$144 per ounce

#### Residual Analysis

**In-Sample Residuals**:
- Mean: 0.823 (close to zero ✓)
- Std Dev: 32.67
- Range: -33.84 to 1629.00 (outlier present)

**Out-of-Sample Residuals**:
- Mean: -112.33 (slight negative bias)
- Std Dev: 114.86
- Range: -293.90 to 154.10

**Autocorrelation Check**:
- Significant lags at 1-20 in full series
- Lags 1-5 show some remaining autocorrelation
- Overall diagnostics pass (Ljung-Box p=0.97)

## Mathematical Foundation

### ARIMA(p,d,q) Components

| Component | Role | Symbol | Example |
|-----------|------|--------|---------|
| **p** (Order) | Autoregressive lags | φᵢ | p=0 (no past values) |
| **d** (Integration) | Differencing level | ∇^d | d=1 (first difference) |
| **q** (Order) | Moving Average lags | θⱼ | q=1 (one error lag) |

### General ARIMA Formula

$$y_t = c + \sum_{i=1}^{p} \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

Where:
- $y_t$ = Value at time t
- $c$ = Constant (drift term)
- $\phi_i$ = AR coefficients (how past values affect current)
- $\theta_j$ = MA coefficients (how past errors affect current)
- $\epsilon_t$ = White noise error

### Differencing Transformation

**First Difference**:
$$\nabla y_t = y_t - y_{t-1}$$

- Removes linear trends
- Most common (d=1)
- Stabilizes mean

**Second Difference** (if needed):
$$\nabla^2 y_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) = y_t - 2y_{t-1} + y_{t-2}$$

- Removes quadratic trends
- Rarely needed (d=2)
- Risks over-differencing

## Model Selection Criteria Explanation

### AIC (Akaike Information Criterion)
$$AIC = 2k + n \ln(RSS/n)$$

- Penalizes model complexity (k = number of parameters)
- Useful for in-sample fit and short-term forecasts
- Lower is better

### BIC (Bayesian Information Criterion)
$$BIC = k \ln(n) + n \ln(RSS/n)$$

- Stronger penalty for complexity than AIC
- Prefers simpler models (parsimony principle)
- Better for model comparison and interpretation
- Used for final selection in this analysis

## Why ARIMA Superior to ARMA

| Aspect | ARMA | ARIMA |
|--------|------|-------|
| **Stationarity** | Requires stationary input | Handles non-stationary directly |
| **Preprocessing** | Manual differencing needed | Automatic via d parameter |
| **Trend Handling** | Cannot model trends | Models via differencing |
| **Practicality** | Extra steps required | All-in-one approach |
| **Flexibility** | Limited to stationary data | Works with most real data |

## When to Use ARIMA

### ✅ Best For:
- Non-stationary time series with trend
- Financial data (stock prices, commodities)
- Macroeconomic indicators
- Short to medium-term forecasting (1-12 steps ahead)
- Data with mixed autocorrelation patterns
- When ACF decays and PACF cuts off (or vice versa)

### ❌ Not Suitable For:
- Seasonal data (use SARIMA instead)
- Multiple structural breaks or regime changes
- Very short series (< 50 observations)
- Non-linear relationships
- Data requiring external variables (use ARIMAX)

## Key Insights

### 1. Gold Prices Follow Random Walk with Error Correction

The optimal ARIMA(0,1,1) model indicates:
- **AR(0)**: Past prices don't help predict future prices
- **I(1)**: Price level is non-stationary; changes are what matter
- **MA(1)**: Price changes have small error correlation (θ=0.0125)

**Implication**: Daily gold price changes are largely **unpredictable** based on historical patterns alone.

### 2. Small MA Coefficient Validates Efficient Market Hypothesis

θ₁ = 0.0125 is:
- Very small (1.25% effect)
- Not statistically significant (p=0.515)
- Explains minimal variance

This suggests markets incorporate information efficiently—no exploitable patterns.

### 3. Model Performance vs. Market Behavior

- ARIMA beats naive by 27.52%
- But 6.72% MAPE shows forecasting is difficult
- Random walk dominance indicates weak predictability

### 4. Differencing Solves Stationarity Problem

| Series | ADF p-value | Result | ARIMA d |
|--------|-------------|--------|---------|
| Original | 0.923 | Non-stationary | - |
| First Diff | <0.001 | Stationary | d=1 ✓ |
| Second Diff | <0.001 | Stationary | Unnecessary |

d=1 is sufficient and follows parsimony principle.

### 5. Progression of Time Series Models

```
Day 12: AR(p)       → Autoregressive only
Day 13: ARMA(p,q)   → Auto + Moving Average
Day 14: ARIMA(p,d,q) → + Integrated (differencing) ← YOU ARE HERE
Day 15: SARIMA      → + Seasonal patterns (coming next)
```

## Outcomes Achieved

✅ Successfully built and evaluated ARIMA models
✅ Demonstrated automatic handling of non-stationary data
✅ Achieved 27.52% improvement over naive forecast
✅ Selected optimal model using information criteria
✅ Comprehensive residual diagnostics validation
✅ Clear model interpretation and forecasting capability
✅ Foundation for seasonal ARIMA (SARIMA) extension

## Files Generated

- `main.py` - Full ARIMA analysis pipeline
- `main.ipynb` - Interactive exploration and visualizations
- `README.md` - This comprehensive documentation

## Next Steps

**Day 15: SARIMA Models**
- Add seasonal component S(P,D,Q)s
- Handle yearly/quarterly/monthly patterns
- Extended model: SARIMA(p,d,q)(P,D,Q)s
- Apply to seasonal gold price patterns
