# Day 15: SARIMA Models - Seasonal ARIMA

## Objective

Extend ARIMA models with explicit seasonal components to handle time series data with recurring yearly (or periodic) patterns, completing the ARIMA family progression.

## Key Deliverables

- Seasonal decomposition analysis (trend, seasonal, residual)
- Seasonal strength assessment and interpretation
- Seasonal vs non-seasonal differencing analysis
- Multiple SARIMA model configurations tested
- Model selection using AIC and BIC criteria
- Performance comparison with naive baseline
- Comprehensive residual diagnostics

## Dataset

- **Source**: Yahoo Finance Gold (GLD)
- **Aggregation**: Monthly averages (121 observations)
- **Period**: 2016-01-01 to 2026-01-01
- **Train-Test Split**: 80/20 (96 training months, 25 test months)

## Key Findings

### 1. Seasonal Decomposition

**Components Analysis**:
- **Trend**: Mean = 166.12, Std = 46.74
  - Long-term upward trend with variation
- **Seasonal**: Mean = -0.01, Std = 2.21
  - Relatively small seasonal component
- **Residual**: Mean = -0.56, Std = 5.33
  - Random variation around trend + seasonality

**Seasonal Strength**: 0.1468 (14.7% of variation is seasonal)
- ✅ Above 10% threshold → SARIMA is appropriate
- Weak but detectable seasonal pattern
- Suggests seasonality matters but won't dominate

### 2. Stationarity Testing

| Series | ADF p-value | Status |
|--------|-------------|--------|
| Original | 1.000000 | ✗ Non-stationary |
| First Difference (d=1) | 0.000000 | ✓ Stationary |
| Seasonal Difference (D=1, 12m) | 0.000000 | ✓ Stationary |

**Recommendation**: d=1, D=1
- First differencing handles trend
- Seasonal differencing captures 12-month cycle
- Both achieve stationarity

### 3. Data Summary

- **Monthly Data Shape**: 121 observations (10 years)
- **Price Mean**: $173.13
- **Price Std Dev**: $63.85
- **Price Range**: $105.37 - $409.21

### 4. SARIMA Model Comparison

| Model | AIC | BIC | RMSE |
|-------|-----|-----|------|
| ARIMA(1,1,0)(0,0,0,12) | 547.68 | 552.77 | 105.97 |
| **ARIMA(0,1,1)(1,0,0,12)** | **487.80** | **495.05** | **105.05** |
| ARIMA(0,1,1)(0,1,0,12) | 523.08 | 527.87 | 81.40 |

**Optimal Model**: SARIMA(0,1,1)(1,0,0,12)
- Selected by BIC (parsimony principle)
- AIC: 487.80
- BIC: 495.05
- Lowest AIC among top performers

### 5. Optimal Model: SARIMA(0,1,1)(1,0,0,12)

**Model Structure**:
- **(p,d,q) = (0,1,1)**: Non-seasonal component
  - p=0: No autoregressive term
  - d=1: First differencing
  - q=1: One moving average term
- **(P,D,Q)s = (1,0,0,12)**: Seasonal component
  - P=1: Seasonal autoregressive order 1
  - D=0: No seasonal differencing needed
  - Q=0: No seasonal moving average
  - s=12: Monthly seasonality (12 months)

**Parameter Estimates**:

| Parameter | Value | Std Err | P-value | Interpretation |
|-----------|-------|---------|---------|-----------------|
| ma.L1 | 0.2264 | 0.100 | 0.024 | ✓ Significant MA(1) term |
| ar.S.L12 | 0.0906 | 0.115 | 0.433 | ⚠ Not significant seasonal AR |
| σ² | 19.4196 | 3.352 | <0.001 | ✓ Significant error variance |

**Diagnostic Tests**:
- **Ljung-Box Test**: Q=0.02, p=0.87 → ✓ No autocorrelation
- **Jarque-Bera Test**: JB=0.47, p=0.79 → ✓ Normal residuals
- **Heteroskedasticity**: H=3.82, p<0.01 → ⚠ Unequal variance detected
- **Skewness**: 0.01 → ✓ Symmetric
- **Kurtosis**: 2.63 → ✓ Normal tails

### 6. Forecast Performance

**Test Set Results**:
- **RMSE**: 105.05
- **MAE**: 83.99
- **MAPE**: 27.06%

**Naive Baseline**:
- **RMSE**: 107.09

**Improvement**: +1.91%
- Modest improvement over naive forecast
- Gold prices are difficult to forecast with seasonal models
- Reflects efficient market hypothesis

**Residual Statistics**:
- **Mean**: 1.73 (close to zero, slight positive bias)
- **Std Dev**: 11.56
- **Range**: -9.76 to 105.37 (outlier present)

## Mathematical Foundation

### SARIMA(p,d,q)(P,D,Q)s Components

**Non-seasonal AR component (p)**:
$$y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \epsilon_t$$

**Non-seasonal MA component (q)**:
$$y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots$$

**Seasonal AR component (P)**:
$$y_t = \Phi_1 y_{t-s} + \Phi_2 y_{t-2s} + \cdots$$

**Seasonal Differencing (D)**:
$$\nabla_s^D y_t = (1 - B^s)^D y_t$$

**Full SARIMA Equation** (multiplicative):
$$(1 - \phi_1 B - \cdots)(1 - \Phi_1 B^s - \cdots) (1-B)^d(1-B^s)^D y_t = (1 + \theta_1 B + \cdots)(1 + \Theta_1 B^s + \cdots) \epsilon_t$$

Where:
- $B$ = Backshift operator
- $s$ = Seasonal period (12 for monthly)
- $\phi_i$ = AR coefficients
- $\theta_j$ = MA coefficients
- $\Phi_k$ = Seasonal AR coefficients
- $\Theta_l$ = Seasonal MA coefficients

### Seasonal Strength Metric

$$\text{Seasonal Strength} = 1 - \frac{\text{Var}(\text{Residual})}{\text{Var}(\text{Seasonal}) + \text{Var}(\text{Residual})}$$

- Range: 0 to 1
- **0**: No seasonality
- **1**: Perfect seasonality
- **0.1468**: Weak but present seasonality

## SARIMA vs ARIMA Comparison

| Aspect | ARIMA(p,d,q) | SARIMA(p,d,q)(P,D,Q)s |
|--------|--------------|----------------------|
| **Seasonal Patterns** | Cannot capture | ✓ Explicit component |
| **Computational Cost** | Low | Moderate |
| **Data Requirements** | ~50+ obs | ~50+ obs per season |
| **Use Case** | Non-seasonal data | Seasonal data |
| **Forecasting Accuracy** | Good for trends | Better for cycles |
| **Flexibility** | Fixed parameters | Separate seasonal params |

## When to Use SARIMA

### ✅ Best For:
- **Retail Sales**: Strong holiday seasonality
- **Weather**: Temperature/precipitation cycles
- **Tourism**: Seasonal visitor patterns
- **Utilities**: Seasonal energy/water usage
- **Electricity**: Daily/monthly demand patterns
- **Gold Prices**: Weak annual seasonality (commodity demand)

### ❌ Not Suitable For:
- **Stock Indices**: Weak/absent seasonality
- **Financial Returns**: Random walk behavior
- **Non-seasonal Data**: Use ARIMA instead
- **Structural Breaks**: Use Markov regime switching
- **Multiple Seasonalities**: Use TBATS/Prophet

## ARIMA Family Progression

```
Day 12: AR(p)
        ↓ Only uses past values
        Weak autocorrelation detection

Day 13: ARMA(p,q)
        ↓ Adds past errors
        Mixed autocorrelation patterns

Day 14: ARIMA(p,d,q)
        ↓ Adds differencing
        Handles trends and non-stationarity

Day 15: SARIMA(p,d,q)(P,D,Q)s ← YOU ARE HERE
        ↓ Adds seasonal components
        Captures repeating yearly patterns
```

## Model Selection Criteria

### AIC (Akaike Information Criterion)
$$\text{AIC} = 2k + n \ln(\text{RSS}/n)$$
- **Use when**: Comparing models on same data
- **Advantage**: Tends to select more complex models
- **Disadvantage**: May overfit

### BIC (Bayesian Information Criterion)
$$\text{BIC} = k \ln(n) + n \ln(\text{RSS}/n)$$
- **Use when**: Prefer simpler, more interpretable models
- **Advantage**: Stronger penalty for complexity (parsimony)
- **Disadvantage**: May underfit with large samples

**In This Analysis**: Used BIC to select SARIMA(0,1,1)(1,0,0,12)
- Lowest BIC among stable models
- Simpler than alternatives
- Interpretable coefficients

## Key Insights

### 1. Weak Seasonality in Gold Prices
- 14.7% of variation is seasonal
- Suggests some annual cyclicality
- But trend dominates the dynamics
- Not as strong as retail/weather data

### 2. Model Performance
- 1.91% improvement over naive forecast
- Modest gain reflects efficient markets
- Daily price changes unpredictable
- Monthly aggregation helps but limited

### 3. Parameter Interpretation
- **ma.L1 = 0.226** (p=0.024): Significant moving average
  - Past month's error affects current forecast
  - 22.6% weight on recent error
- **ar.S.L12 = 0.091** (p=0.433): Not significant
  - Seasonal AR term doesn't help much
  - Could potentially be removed

### 4. Diagnostic Quality
- ✓ No autocorrelation (Ljung-Box p=0.87)
- ✓ Normal residuals (Jarque-Bera p=0.79)
- ⚠ Unequal variance (Heteroskedasticity p<0.01)
  - Suggests volatility increases with price level
  - Could use multiplicative model instead

### 5. Why Monthly Data?
- Daily SARIMA with period=252 (annual trading days) crashes systems
- Monthly aggregation: 121 observations instead of 2,500+
- Seasonal period = 12 months (reasonable)
- Computationally efficient without losing patterns

## Outcomes Achieved

✅ Successfully implemented SARIMA on gold price data
✅ Identified optimal model using information criteria
✅ Detected weak but present seasonality
✅ Achieved 1.91% improvement over naive forecast
✅ Comprehensive diagnostic testing (all tests passed except heteroskedasticity)
✅ Foundation for advanced forecasting methods
✅ Completed ARIMA family progression (AR → ARMA → ARIMA → SARIMA)

## Recommendations for Further Analysis

1. **Model Enhancement**:
   - Try multiplicative SARIMA for heteroskedasticity
   - Test auto_arima for automated parameter selection
   - Consider ARIMAX (add external variables like interest rates)

2. **Advanced Methods**:
   - TBATS for multiple seasonalities
   - Prophet for holiday/event effects
   - LSTM neural networks for non-linear patterns

3. **Ensemble Approach**:
   - Combine SARIMA with other methods
   - Weighted average of predictions
   - Kalman filter for adaptive forecasting

4. **Domain Knowledge**:
   - Incorporate macroeconomic factors
   - Fed interest rate decisions
   - Geopolitical events
   - Currency exchange rates

## Files Generated

- `main.py` - Complete SARIMA analysis pipeline
- `main.ipynb` - Interactive exploration notebook
- `README.md` - This comprehensive documentation

## Next Steps

**Day 16 and Beyond**:
- Auto ARIMA/SARIMA for automated selection
- Ensemble methods combining multiple models
- Deep learning approaches (LSTM, Transformer)
- Real-time forecasting systems
