# Day 20: Confidence Intervals & Prediction Bands

## Overview
Confidence intervals and prediction bands quantify forecast uncertainty. This analysis demonstrates how to:
- Generate prediction intervals at multiple confidence levels (90%, 95%, 99%)
- Evaluate interval coverage rates (how many actuals fall within the bands)
- Understand interval width behavior over the forecast horizon
- Calibrate models based on actual uncertainty

## Objectives
1. **Generate prediction intervals** using ARIMA forecasts
2. **Calculate coverage rates** and verify model calibration
3. **Analyze interval width growth** as forecast horizon increases
4. **Understand uncertainty propagation** in time series forecasts

## Dataset
- **Source**: Gold prices (GLD)
- **Period**: 2015-01-02 to 2026-01-23
- **Total Observations**: 2,781
- **Training Set**: 2,224 observations (80%, 2015-01-02 to 2023-11-01)
- **Test Set**: 557 observations (20%, 2023-11-02 to 2026-01-23)

**Data Statistics:**
| Metric | Value |
|--------|-------|
| Mean Price | $1,805.17 |
| Std Dev | $678.59 |
| Min | $1,085.00 |
| Max | $4,946.00 |

## Methodology

### 1. Stationarity Verification
**ADF Test on First Differences:**
- Test Statistic: -6.144134
- P-value: 0.000000
- Result: **STATIONARY** ✓

The first differences are stationary, confirming the need for d=1 in ARIMA.

### 2. Model Selection
**ARIMA(1,1,0) Model:**
- AR(1) component: Uses 1 past observation
- Integration: d=1 (first differencing)
- MA(0): No moving average term
- **AIC**: 24,764.40
- **BIC**: 24,776.26

### 3. Confidence Interval Generation
Prediction intervals computed at three confidence levels:

| Confidence Level | Formula | Coverage Expected |
|-----------------|---------|------------------|
| 90% | ±1.645σ | ~90% of actuals within bounds |
| 95% | ±1.960σ | ~95% of actuals within bounds |
| 99% | ±2.576σ | ~99% of actuals within bounds |

Where σ is the forecast error standard deviation.

## Results

### Forecast Performance

| Metric | Value |
|--------|-------|
| **RMSE** | 2,175.32 |
| **MAE** | 2,056.22 |
| **Naive RMSE** | 1,151.96 |
| **Performance vs Naive** | -88.84% (underperformance) |

**Note**: ARIMA(1,1,0) static model shows poor out-of-sample performance. Day 21 rolling validation shows dramatically better results (+90.32% improvement).

### Confidence Interval Coverage Rates

**Actual % of test actuals within intervals:**

| Confidence Level | Expected Coverage | Actual Coverage | Count | Assessment |
|-----------------|------------------|-----------------|-------|------------|
| 90% | ~90% | 7.2% | 40/557 | ⚠️ Severe under-coverage |
| 95% | ~95% | 10.8% | 60/557 | ⚠️ Severe under-coverage |
| 99% | ~99% | 14.5% | 81/557 | ⚠️ Severe under-coverage |

**Interpretation**: All confidence levels show coverage rates far below expected values. The model severely underestimates forecast uncertainty, making intervals unreliable for decision-making.

### Confidence Interval Widths

**Average interval widths over test period:**

| Confidence Level | Mean Width | Min Width | Max Width | Std Dev |
|-----------------|-----------|-----------|-----------|---------|
| **90% CI** | $1,068.34 | $68.39 | $1,600.35 | $376.56 |
| **95% CI** | $1,273.01 | $81.49 | $1,906.93 | $448.70 |
| **99% CI** | $1,673.02 | $107.10 | $2,506.14 | $589.69 |

**Width Behavior:**
- Initial forecasts very narrow ($68-$107): Model overconfident in near-term
- Widths grow substantially: Uncertainty increases with horizon
- Even 99% CI too narrow to capture actual price movement
- Gold prices exhibit more volatility than ARIMA(1,1,0) captures

### Residual Analysis

**In-Sample Residuals (Training Period):**
- Mean: 1.79 (good - close to zero)
- Std Dev: 31.22
- Range: [-278.69, 1,232.00]

**Out-of-Sample Forecast Errors (Test Period):**
- Mean: -2,056.22 (systematic underprediction)
- Std Dev: 709.91
- Range: [-3,006.43, 0.57]

The large negative mean error indicates the model consistently predicts too-low prices.

## Key Insights

### Understanding Confidence Intervals

1. **What they mean:**
   - **90% CI**: If we repeated this forecast 100 times, ~90 intervals would contain the true value
   - **95% CI**: Standard business choice - balance between precision and uncertainty
   - **99% CI**: Conservative choice for critical decisions - accepts wider bands

2. **Width behavior:**
   - CIs grow wider with forecast horizon
   - Uncertainty increases exponentially
   - Initial forecasts have narrowest intervals

### Uncertainty Quantification

**Why intervals matter in forecasting:**
- Point forecasts alone are misleading
- Intervals communicate decision confidence
- Wider intervals = higher caution needed
- Narrow intervals = high confidence in prediction

### Model Calibration Issues

The model has **severe calibration problems**:
1. **Under-coverage**: Only 10.8% within 95% CI vs. expected 95%
2. **Systematic bias**: Mean forecast error = -$2,056.22
3. **Poor uncertainty estimation**: Intervals too narrow for actual variability
4. **Univariate limitation**: Single time series may not capture gold price drivers

### Why Performance is Poor

The ARIMA(1,1,0) model shows poor out-of-sample performance (-88.84% vs naive) because:
- Gold prices have structural breaks and regime changes
- Univariate models struggle with external shocks
- 2024-2026 period saw significant price movements not in training data
- Confidence intervals reflect training period variability, not test period volatility

## Mathematical Formula

For ARIMA forecasts, prediction intervals are:

$$\hat{y}_{t+h} \pm z_{\alpha/2} \cdot \sigma_h$$

Where:
- $\hat{y}_{t+h}$ = point forecast h steps ahead
- $z_{\alpha/2}$ = critical value (1.96 for 95% CI)
- $\sigma_h$ = forecast error standard deviation

The standard error $\sigma_h$ grows with h, causing interval widening.

## Recommendations

1. **Model Improvement:**
   - Consider multivariate approaches (include economic indicators)
   - Explore machine learning models (Random Forests, Neural Networks)
   - Use regime-switching models for structural breaks

2. **Interval Calibration:**
   - Use bootstrap methods for better uncertainty estimates
   - Consider quantile regression
   - Empirically validate coverage on holdout data

3. **Decision Making:**
   - Use 99% CI for important decisions
   - Don't rely solely on point forecasts
   - Monitor actual coverage and adjust model accordingly

## Progression Notes

**ARIMA Family Progress:**
- Day 12: AR(10) - Good performance (+10.69% improvement)
- Day 13: MA(1) - Acceptable performance (-0.91% improvement)
- Day 14: ARIMA(1,1,0) - Poor performance (-88.84% vs naive)
- **Day 20: Confidence Intervals** - Shows ARIMA limitations, demonstrates uncertainty quantification
- Day 21: Rolling forecasts & backtesting - Realistic validation methodology

**Key Takeaway**: Confidence intervals reveal that standard ARIMA models may not be suitable for gold prices. More sophisticated approaches needed for reliable forecasting.
