# Day 17: ARIMA Diagnostics
## Residual Analysis for Model Validation

**Objective:** Validate ARIMA model assumptions through comprehensive residual diagnostics including white noise testing, normality assessment, and heteroskedasticity detection.

---

## Table of Contents
1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Diagnostic Tests](#diagnostic-tests)
4. [Implementation Details](#implementation-details)
5. [Results Interpretation](#results-interpretation)
6. [When to Use](#when-to-use)

---

## Overview

After fitting an ARIMA model, we must validate that:
1. **Residuals are white noise** (no autocorrelation)
2. **Residuals are normally distributed** (for confidence intervals)
3. **Variance is constant over time** (homoscedasticity)
4. **No patterns remain** in residual structure

A well-specified ARIMA model should produce residuals that satisfy these properties. Violation suggests model misspecification.

### Dataset
- **Source:** Yahoo Finance Gold (GLD)
- **Period:** 2016-01-11 to 2026-01-09 (2,515 observations)
- **Target Model:** ARIMA(0,1,0) from Day 16 (optimal by BIC)
- **Train/Test Split:** 80/20 (2,012 / 503)

---

## Key Concepts

### 1. Residuals
Residuals are differences between observed and fitted values:
$$\hat{e}_t = y_t - \hat{y}_t$$

For ARIMA(p,d,q):
$$\hat{e}_t = y_t - \alpha_1 y_{t-1} - ... - \alpha_p y_{t-p} + \theta_1 \hat{e}_{t-1} + ... + \theta_q \hat{e}_{t-q}$$

**Good residuals:** Small, random, no patterns
**Bad residuals:** Large spikes, trends, cycles, outliers

### 2. White Noise Process
A time series is white noise if:
- **Mean = 0:** $E[\hat{e}_t] = 0$
- **Constant variance:** $\text{Var}(\hat{e}_t) = \sigma^2$
- **Uncorrelated:** $\text{Cov}(\hat{e}_t, \hat{e}_{t-k}) = 0$ for $k \neq 0$

This is the **gold standard** for ARIMA residuals.

### 3. Autocorrelation
Residuals should NOT be autocorrelated:
$$\text{Corr}(\hat{e}_t, \hat{e}_{t-k}) = 0 \text{ for } k > 0$$

If residuals are correlated, the model missed patterns.

---

## Diagnostic Tests

### 1. Ljung-Box Test (Autocorrelation)

Tests whether residuals are **independently distributed** (white noise).

**Null Hypothesis:** $H_0:$ Residuals are white noise (no autocorrelation)
**Alternative:** $H_A:$ Autocorrelation exists

**Test Statistic:**
$$Q_{LB}(m) = n(n+2) \sum_{k=1}^{m} \frac{\rho_k^2}{n-k}$$

where:
- $n$ = sample size
- $m$ = number of lags tested
- $\rho_k$ = autocorrelation at lag $k$
- $Q_{LB} \sim \chi^2_m$ approximately

**Decision Rule:**
- If $p\text{-value} > 0.05$: ✓ PASS (white noise)
- If $p\text{-value} \leq 0.05$: ✗ FAIL (autocorrelation present)

**Common Lags to Test:**
- Lag 5-10: Short-term dependencies
- Lag 20-40: Medium-term cycles
- Lag 40+: Long-term patterns

**Interpretation:**
- **PASS:** Model captured all autocorrelation
- **FAIL:** Model incomplete; try larger p or q

### 2. Jarque-Bera Test (Normality)

Tests whether residuals follow a **normal distribution**.

**Null Hypothesis:** $H_0:$ Residuals ~ $N(\mu, \sigma^2)$
**Alternative:** $H_A:$ Residuals deviate from normality

**Test Statistic:**
$$JB = n \left[ \frac{S^2}{6} + \frac{(K-3)^2}{24} \right]$$

where:
- $n$ = sample size
- $S$ = skewness = $\frac{1}{n}\sum \left(\frac{\hat{e}_t - \bar{\hat{e}}}{\hat{\sigma}}\right)^3$
- $K$ = kurtosis = $\frac{1}{n}\sum \left(\frac{\hat{e}_t - \bar{\hat{e}}}{\hat{\sigma}}\right)^4$
- $JB \sim \chi^2_2$ approximately

**Components:**
- **Skewness:** Asymmetry (ideally 0)
  - Positive: Right tail longer
  - Negative: Left tail longer
  - $|S| < 0.5$ is acceptable
  
- **Kurtosis:** Tail heaviness (ideally 0 for excess kurtosis)
  - Positive: Heavy tails (outliers common)
  - Negative: Light tails (fewer outliers)
  - $|K| < 1.0$ is acceptable

**Decision Rule:**
- If $p\text{-value} > 0.05$: ✓ PASS (normal)
- If $p\text{-value} \leq 0.05$: ✗ FAIL (non-normal)

**Implications:**
- **Normal residuals:** Confidence intervals and hypothesis tests valid
- **Non-normal residuals:** Use robust methods or transformation

### 3. Heteroskedasticity Test (Variance Stability)

Tests whether **variance is constant** over time.

**Null Hypothesis:** $H_0:$ $\text{Var}(\hat{e}_t) = \sigma^2$ (constant)
**Alternative:** $H_A:$ Variance changes over time

**Method: H-Statistic (Rolling Variance Ratio)**

Divide the series into two halves and compute:
$$H = \frac{\text{Var}(\hat{e}_{t, \text{second half}})}{\text{Var}(\hat{e}_{t, \text{first half}})}$$

**Decision Rule:**
- If $0.8 < H < 1.25$: ✓ PASS (homoscedastic)
- If $H \leq 0.8$ or $H \geq 1.25$: ✗ FAIL (heteroscedastic)

**Interpretation:**
- **Homoscedastic:** Variance stable (good for forecasting)
- **Heteroscedastic:** Volatility clusters exist (May need GARCH models)

### 4. ACF/PACF of Residuals

**ACF (Autocorrelation Function):**
$$\rho_k = \frac{\sum_{t=k+1}^n (\hat{e}_t - \bar{\hat{e}})(\hat{e}_{t-k} - \bar{\hat{e}})}{\sum_{t=1}^n (\hat{e}_t - \bar{\hat{e}})^2}$$

**PACF (Partial Autocorrelation Function):**
Controls for intermediate lags using:
$$\phi_{kk} = \text{Corr}(\hat{e}_t, \hat{e}_{t-k} | \hat{e}_{t-1},...,\hat{e}_{t-k+1})$$

**Confidence Interval:**
$$\text{CI} = \pm \frac{1.96}{\sqrt{n}}$$

**Good residuals:** Most correlations within confidence bands

---

## Implementation Details

### Data Processing
```python
# Load and split data
df = pd.read_csv('gold_prices.csv')
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]
```

### Model Fitting
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_data['Price'], order=(0, 1, 0))
fitted_model = model.fit()
residuals = fitted_model.resid
```

### Ljung-Box Test
```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_results = acorr_ljungbox(residuals, lags=[5, 10, 20, 40], return_df=True)
print(lb_results)
# p-value > 0.05 → white noise ✓
```

### Jarque-Bera Test
```python
from scipy import stats

jb_stat, jb_pvalue = stats.jarque_bera(residuals)
skewness = stats.skew(residuals)
kurtosis = stats.kurtosis(residuals)
# p-value > 0.05 → normal ✓
```

### Heteroskedasticity Test
```python
window = 50
rolling_var = pd.Series(residuals).rolling(window).var()
h_stat = rolling_var[-1] / rolling_var[0]
# 0.8 < h_stat < 1.25 → homoscedastic ✓
```

### ACF/PACF Analysis
```python
from statsmodels.tsa.stattools import acf, pacf

acf_vals = acf(residuals, nlags=40)
pacf_vals = pacf(residuals, nlags=40, method='ywm')
ci = 1.96 / np.sqrt(len(residuals))
# Most values within [-ci, ci] → good ✓
```

---

## Results Interpretation

### Scenario 1: All Tests Pass ✓
```
Ljung-Box (lag 20): p-value = 0.45 → ✓ PASS
Jarque-Bera: p-value = 0.82 → ✓ PASS
Heteroskedasticity: H = 1.02 → ✓ PASS
ACF: 1 significant lag → ✓ PASS
PACF: 0 significant lags → ✓ PASS
```

**Interpretation:** 
- Model well-specified
- Residuals are white noise
- Safe for forecasting and inference
- Confidence intervals and hypothesis tests valid

---

### Scenario 2: Ljung-Box Fails ✗
```
Ljung-Box (lag 20): p-value = 0.01 → ✗ FAIL
Ljung-Box (lag 10): p-value = 0.03 → ✗ FAIL
ACF: 5+ significant lags → ✗ FAIL
```

**Interpretation:**
- Residuals have **autocorrelation**
- Model missed patterns in the data
- Suggestions:
  1. Increase p or q parameters
  2. Check for seasonal patterns (increase P or Q)
  3. Try different differencing order (d)
  4. Consider ARIMA extensions (SARIMA, VAR)

---

### Scenario 3: Jarque-Bera Fails ✗
```
Jarque-Bera: p-value = 0.02 → ✗ FAIL
Skewness: 0.45 → Moderate
Kurtosis: 1.85 → Heavy tails
```

**Interpretation:**
- Residuals are **non-normal**
- Often caused by:
  1. Outliers/extreme values
  2. Heavy-tailed distributions
  3. Mixture of regimes
- Implications:
  1. Confidence intervals may be inaccurate
  2. Hypothesis tests unreliable
  3. Consider robust methods or transformation

---

### Scenario 4: Heteroskedasticity Detected ✗
```
Rolling Variance (first half): 100.5
Rolling Variance (second half): 250.3
H-statistic: 2.49 → ✗ FAIL
```

**Interpretation:**
- Variance is **changing over time** (volatility clustering)
- Forecast uncertainty increases over time
- Options:
  1. Accept increasing uncertainty in forecasts
  2. Use exponential smoothing of variance
  3. Fit GARCH/ARCH model for conditional variance
  4. Consider regime-switching models

---

## When to Use

### Primary Applications
1. **Model Validation:** After fitting any ARIMA model
2. **Model Comparison:** Comparing candidate models
3. **Confidence Assessment:** Before using for forecasting
4. **Research:** Publishing time series studies (required)
5. **Production Systems:** Monitoring model performance

### Decision Framework

| Test | Passes | Action |
|------|--------|--------|
| Ljung-Box | ✓ | Model adequate for forecasting |
| Ljung-Box | ✗ | Increase p, d, or q; try SARIMA |
| Jarque-Bera | ✓ | Confidence intervals valid |
| Jarque-Bera | ✗ | Use robust methods or transform |
| Homoscedasticity | ✓ | Standard forecasting OK |
| Homoscedasticity | ✗ | Consider GARCH/volatility model |
| ACF/PACF | ✓ | Residuals are white noise |
| ACF/PACF | ✗ | Check model specification |

---

## Code Example: Complete Diagnostic Flow

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import numpy as np

# Fit model
model = ARIMA(data, order=(0, 1, 0))
fitted = model.fit()
residuals = fitted.resid

# 1. Ljung-Box Test
lb = acorr_ljungbox(residuals, lags=20, return_df=True)
white_noise = lb['lb_pvalue'].iloc[-1] > 0.05

# 2. Jarque-Bera Test
jb_stat, jb_p = stats.jarque_bera(residuals)
normal = jb_p > 0.05

# 3. Heteroskedasticity
rolling_var = residuals.rolling(50).var()
h_stat = rolling_var.iloc[-1] / rolling_var.iloc[0]
homoscedastic = 0.8 < h_stat < 1.25

# Decision
if white_noise and normal and homoscedastic:
    print("✓ Model diagnostics EXCELLENT")
    print("Safe to use for forecasting")
else:
    print("✗ Some diagnostic tests failed")
    if not white_noise:
        print("  - Try larger p/q")
    if not normal:
        print("  - Consider transformation")
    if not homoscedastic:
        print("  - Consider GARCH model")
```

---

## Summary

### Key Takeaways

1. **Ljung-Box Test:** Checks for autocorrelation in residuals
   - Best use: Validating ARIMA model adequacy
   - Decision: $p > 0.05$ → white noise ✓

2. **Jarque-Bera Test:** Checks for normality
   - Best use: Validating inference assumptions
   - Decision: $p > 0.05$ → normal ✓

3. **Heteroskedasticity Test:** Checks for constant variance
   - Best use: Assessing forecast uncertainty
   - Decision: $0.8 < H < 1.25$ → constant ✓

4. **ACF/PACF Analysis:** Visual white noise check
   - Best use: Identifying remaining patterns
   - Decision: Most within confidence bands ✓

5. **Integration:** Use ALL tests together for complete assessment

### Common Issues and Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Ljung-Box fails | Missed autocorrelation | Increase p or q |
| Non-normal residuals | Heavy tails/outliers | Use robust methods |
| Non-constant variance | Volatility clustering | Add GARCH component |
| ACF spikes at lag k | Seasonal pattern | Add seasonal terms (P, Q) |

### Next Steps

- **If diagnostics PASS:** Proceed to forecasting with confidence
- **If diagnostics FAIL:** 
  1. Identify failing test
  2. Adjust model parameters
  3. Rerun diagnostics
  4. Iterate until all tests pass

---

## References

- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control. Wiley.
- Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometric reviews, 1(1), 29-59.
- Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review, 55(2), 163-172.
- Ljung, G. M., & Box, G. E. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.
