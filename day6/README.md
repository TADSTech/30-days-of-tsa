# Day 6: Autocorrelation Analysis (ACF & PACF)

## Project Overview
Comprehensive analysis of Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to identify optimal ARIMA model parameters. This project demonstrates how correlograms guide the selection of p (AR), d (I), and q (MA) orders.

## Key Concepts

### Autocorrelation Function (ACF)
- **Measures**: Correlation between a time series and its lagged values
- **Scope**: Includes both direct and indirect relationships through intermediate lags
- **Use**: Identifies the Moving Average (MA) order → **q parameter**
- **Pattern**: In MA(q) models, ACF typically cuts off sharply after lag q
- **Interpretation**: Significant lags indicate past values that influence current values

### Partial Autocorrelation Function (PACF)
- **Measures**: Correlation between a time series and its lagged values after removing the effect of intermediate lags
- **Scope**: Shows only direct (direct) autoregressive relationships
- **Use**: Identifies the Autoregressive (AR) order → **p parameter**
- **Pattern**: In AR(p) models, PACF typically cuts off sharply after lag p
- **Interpretation**: Significant lags show direct dependencies requiring AR terms

### Significance Testing
- Both ACF and PACF plots display **95% confidence intervals** (blue shaded area)
- Bars extending beyond confidence bands indicate **statistically significant** correlations
- These significant lags must be modeled by ARIMA parameters

## Dataset
- **Source**: Gold price historical data
- **Period**: 10 years of daily prices
- **Data File**: gold_prices.csv
- **Preprocessing**: First-order differencing applied (d=1)

## Methodology

### 1. Data Preparation
- Load gold price data
- Apply first-order differencing (from Day 5)
- Remove NaN values to create clean dataset
- Verify stationarity of differenced series

### 2. ACF Calculation
- Compute autocorrelation for 40 lags
- Identify lag where ACF becomes statistically insignificant
- Count significant lags to estimate q parameter

### 3. PACF Calculation
- Compute partial autocorrelation for 40 lags
- Identify lag where PACF becomes statistically insignificant
- Count significant lags to estimate p parameter

### 4. Visual Interpretation
- Side-by-side ACF and PACF plots (40 lags)
- Confidence interval bands for significance testing
- Pattern recognition for AR vs MA processes

## Reading the Correlograms

### ACF Pattern Analysis

**Sharp Cutoff (e.g., after lag 1-2)**
- Suggests **MA(q)** process
- Count the number of significant lags before cutoff
- Example: If cutoff at lag 2 → q ≈ 2

**Gradual Exponential Decay**
- Suggests **AR(p)** process
- Indicates trend or strong memory in data
- Look to PACF for p order

**Significant Individual Spikes**
- Seasonal or non-stationary patterns
- May indicate missed differencing

### PACF Pattern Analysis

**Sharp Cutoff (e.g., after lag 1-2)**
- Suggests **AR(p)** process
- Count the number of significant lags before cutoff
- Example: If cutoff at lag 1 → p ≈ 1

**Gradual Exponential Decay**
- Suggests **MA(q)** process
- Look to ACF for q order

**Both Decay Gradually**
- May require both AR and MA terms: **ARMA(p,q)**

## Model Parameter Identification

### Parameter Guidelines

**ARIMA(p, d, q) where:**
- **d = 1** - Confirmed from Day 5 (first-order differencing)
- **p** - AR order from PACF analysis
- **q** - MA order from ACF analysis

### Typical Starting Values

| Pattern | p Values | q Values | Suggested Model |
|---------|----------|----------|------------------|
| ACF cuts off, PACF decays | 0-3 | 1-3 | ARIMA(0-3,1,1-3) |
| ACF decays, PACF cuts off | 1-3 | 0-3 | ARIMA(1-3,1,0-3) |
| Both decay gradually | 1-2 | 1-2 | ARIMA(1-2,1,1-2) |
| Both cut off early | 0-1 | 0-1 | ARIMA(0-1,1,0-1) |

### Recommended Models to Test

1. **ARIMA(1,1,1)** - Balanced AR and MA, common for financial data
2. **ARIMA(2,1,2)** - More complex patterns, if significant lags persist
3. **ARIMA(1,1,0)** - Pure AR process, if PACF shows clear pattern
4. **ARIMA(0,1,1)** - Pure MA process, if ACF shows clear pattern
5. **ARIMA(2,1,0)** - Stronger AR, if PACF shows 2+ significant lags
6. **ARIMA(0,1,2)** - Stronger MA, if ACF shows 2+ significant lags

## Technical Implementation

### Libraries Used
- **statsmodels**: `plot_acf()`, `plot_pacf()`, `acf()`, `pacf()`
- **pandas**: Data manipulation and differencing
- **matplotlib**: ACF and PACF visualization
- **numpy**: Numerical operations

### Key Functions
- `plot_acf()` - Creates ACF correlogram with confidence intervals
- `plot_pacf()` - Creates PACF correlogram with confidence intervals
- `acf()` - Calculates ACF values numerically
- `pacf()` - Calculates PACF values numerically

## Analysis Workflow

1. **Visual Inspection**
   - Examine ACF plot for cutoff or decay pattern
   - Examine PACF plot for cutoff or decay pattern
   - Note significance threshold violations

2. **Lag Counting**
   - Count lags before significance threshold
   - ACF cutoff → q parameter estimate
   - PACF cutoff → p parameter estimate

3. **Model Selection**
   - List candidate models based on patterns
   - Prioritize simpler models (parsimony principle)
   - Plan grid search across p and q ranges

4. **Model Comparison**
   - Fit multiple ARIMA variants
   - Compare using AIC/BIC (lower is better)
   - Check residual diagnostics

5. **Validation**
   - Ensure residuals are white noise
   - Verify no autocorrelation in residuals
   - Test on holdout test set

## Insights

1. **Parameter Estimation**: ACF and PACF provide data-driven guidance for ARIMA parameters
2. **Significance Matters**: Only statistically significant lags should be included
3. **Parsimony Principle**: Simpler models (smaller p,q) preferred if equally valid
4. **Financial Data**: Often exhibits AR(1) or ARMA(1,1) characteristics
5. **Iterative Process**: May need to test multiple models and compare metrics

## Next Steps

1. **Model Fitting**: Implement grid search across ARIMA(p,1,q) combinations
2. **Comparison**: Use AIC/BIC to rank candidate models
3. **Diagnostics**: Analyze residuals for white noise properties
4. **Forecasting**: Generate price predictions with confidence intervals
5. **Validation**: Test on independent test dataset

## Conclusion

ACF and PACF correlograms are indispensable tools for identifying ARIMA model parameters. By correctly interpreting these plots, we can systematically narrow down the search space for optimal (p,q) values. This analysis demonstrates how statistical principles guide practical time series modeling, moving from stationarity testing (Day 4-5) to parameter identification (Day 6) and ultimately to model fitting and forecasting.
