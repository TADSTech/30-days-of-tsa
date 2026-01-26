# Day 16: ARIMA Parameter Selection

## Overview

Systematic exploration of ARIMA parameter selection using grid search, information criteria (AIC/BIC), and automated parameter selection with `pmdarima`'s `auto_arima` function.

## Objectives

- Understand the complete parameter space for ARIMA models
- Learn grid search methodology for hyperparameter optimization
- Master information criteria (AIC and BIC) for model comparison
- Implement automated parameter selection with auto_arima
- Compare manual grid search vs automated approaches
- Visualize the information criteria landscape

## Dataset

**Source**: Yahoo Finance Gold (GLD)  
**Time Period**: 2016-01-11 to 2026-01-09 (2,515 daily observations)  
**Price Range**: $103.02 to $416.74  
**Train/Test Split**: 80/20 (2,012 train / 503 test)  
**Aggregation**: Daily prices (no temporal aggregation)

## Key Concepts

### ARIMA Parameter Space

ARIMA models are defined by three parameters: **(p, d, q)**

- **p (AR order)**: Number of autoregressive lags (0-5 in grid search)
- **d (Integration/differencing)**: Number of difference transformations (0-2 in grid search)
- **q (MA order)**: Number of moving average lags (0-5 in grid search)

**Total Combinations**: 6 × 3 × 6 = **108 models (107 successful, 1 failed)**

### Information Criteria

#### AIC (Akaike Information Criterion)

**Formula**:
$$\text{AIC} = -2 \ln(L) + 2k$$

Where:
- L = maximum likelihood estimate
- k = number of parameters

**Characteristics**:
- Penalizes model complexity with factor 2k
- Favors models with better fit
- Tends to select more complex models
- Asymptotically efficient (better predictive power)

**Interpretation**: Lower AIC indicates better model; difference of 10+ suggests strong evidence against higher AIC model

#### BIC (Bayesian Information Criterion)

**Formula**:
$$\text{BIC} = -2 \ln(L) + k \ln(n)$$

Where:
- L = maximum likelihood estimate
- k = number of parameters
- n = sample size

**Characteristics**:
- Stronger penalty for complexity: $k \ln(n)$ vs 2k in AIC
- Favors simpler, more parsimonious models
- Asymptotically consistent (selects true model as n → ∞)
- More conservative than AIC

**Interpretation**: Lower BIC indicates better model; consistent with AIC when difference > 10

### AIC vs BIC Comparison

| Criterion | AIC | BIC |
|-----------|-----|-----|
| **Penalty** | 2k | k·ln(n) |
| **Complexity Penalizing** | Moderate | Aggressive |
| **Model Preference** | Complex (better fit) | Simple (parsimony) |
| **Asymptotic Property** | Efficient | Consistent |
| **Best For** | Prediction accuracy | Interpretation & avoiding overfitting |
| **Under True Simple Model** | May overfit | Selects correctly |
| **Under Complex Data** | Performs well | May underfit |

### Grid Search Methodology

**Step-by-step process**:

1. **Define parameter ranges**: p ∈ [0,5], d ∈ [0,2], q ∈ [0,5]
2. **Iterate all combinations**: 108 total
3. **Fit each model**: Use MLE (Maximum Likelihood Estimation)
4. **Calculate metrics**: AIC, BIC, test RMSE
5. **Rank by criterion**: Find minimum AIC, BIC, or RMSE
6. **Compare results**: Analyze agreement between criteria

**Advantages**:
- Exhaustive search of parameter space
- Transparent: Can see all models and their metrics
- Educational: Understand parameter sensitivity
- Guaranteed to find best model in search space

**Disadvantages**:
- Computationally expensive: O(p × d × q) = O(108) = ~2 minutes
- Impractical for larger spaces: 6×3×6 is manageable, but 10×3×10 = 300 is slow
- No guidance on appropriate parameter ranges

### Auto ARIMA (Automated Selection)

`pmdarima`'s `auto_arima` function implements:

**Algorithm**: Stepwise algorithm (Hyndman-Khandakar)

1. Start at (p=0, d=0, q=0) or custom starting point
2. Test neighboring (p±1, d±1, q±1) models
3. Move to neighbor with lowest information criterion
4. Repeat until no improvement
5. Return best model found

**Key Features**:
- **Speed**: 1-2 minutes vs grid search's longer duration
- **Scalability**: Works for larger parameter spaces
- **Flexibility**: Can set constraints, information criterion, seasonal parameters
- **Practical**: Designed for production use

**Parameters**:
```python
auto_arima(
    data,
    start_p=0, max_p=5,      # AR order range
    start_d=0, max_d=2,      # Differencing range
    start_q=0, max_q=5,      # MA order range
    seasonal=False,          # Include seasonal? (for SARIMA)
    stepwise=True,           # Use stepwise? (vs brute force)
    information_criterion='bic',  # 'aic' or 'bic'
    trace=False,             # Print each step?
    error_action='ignore',   # Handle errors?
)
```

**Advantages**:
- Automatic: No manual parameter tuning
- Fast: Efficient stepwise algorithm
- Robust: Error handling built-in
- Practical: Good balance between speed and quality

**Disadvantages**:
- Greedy algorithm: May get stuck in local optimum
- Less transparency: Black box optimization
- May miss global optimum: Depends on starting point

## Results

### Best Models by Criterion

| Criterion | Model | AIC | BIC | Test RMSE |
|-----------|-------|-----|-----|-----------|
| **AIC** | ARIMA(3,1,2) | 6929.17 | 6962.81 | 103.45 |
| **BIC** | ARIMA(0,1,0) | 6936.60 | 6942.21 | 103.55 |
| **RMSE** | ARIMA(4,2,5) | 6938.31 | 6994.37 | 91.55 |
| **Auto ARIMA** | ARIMA(0,1,0) | 6936.60 | 6942.21 | 103.55 |

*Note: Specific values from actual execution; see main.py output for details*

### Key Findings

1. **Information Criteria Often Disagree**
   - AIC selected ARIMA(3,1,2) - more complex model with better fit
   - BIC selected ARIMA(0,1,0) - simplest model by parsimony
   - RMSE selected ARIMA(4,2,5) - highest complexity with best forecast
   - Disagreement is common with different penalty structures

2. **RMSE May Not Match Information Criteria**
   - Best by AIC: RMSE 103.45
   - Best by BIC: RMSE 103.55
   - Best by RMSE: 91.55 (but complexity penalty higher)
   - Different optimization objectives lead to different selections

3. **Auto ARIMA Performs Well**
   - Stepped search returned ARIMA(0,1,0)
   - Identical to BIC selection (parsimonious approach)
   - RMSE 103.55 (excellent balance of simplicity and accuracy)
   - Much faster than exhaustive grid search

4. **Parameter Ranges Matter**
   - d=1 (first differencing) appears in almost all top models
   - p and q rarely exceed 2 in top 10 BIC models
   - Higher p,q do not significantly improve test RMSE
   - Grid search confirms d=1 as essential for stationarity

### Visualization Insights

**AIC/BIC Heatmap (d=1)**:
- Shows relationship between p and q for fixed d
- Lower values (darker) = better by that criterion
- Patterns reveal optimal region in parameter space

**AIC vs BIC Scatter Plot**:
- Each point is one ARIMA model
- Color indicates test RMSE
- Shows trade-off between criteria
- Highlights disagreement between criteria

## Mathematical Foundation

### Information Criterion Definition

Both AIC and BIC balance model fit with model complexity:

$$\text{IC} = -2 \ln(L) + \text{Penalty}$$

Where the penalty differs:
- **AIC**: Penalty = 2k (constant per parameter)
- **BIC**: Penalty = k ln(n) (grows with sample size)

As sample size n increases, BIC penalty grows logarithmically, making BIC increasingly conservative.

### Log-Likelihood in ARIMA

For ARIMA model with residuals {ε_t}:

$$\ln(L) = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{t=1}^{n} \epsilon_t^2$$

Where σ² is estimated from residuals.

### Parameter Counting

For ARIMA(p, d, q):
- k = p + q + 1 (plus intercept/constant term)
- Differencing (d) doesn't add parameters but affects available data

## When to Use Each Method

### Grid Search

**Use when**:
- ✅ Parameter space is small (< 200 combinations)
- ✅ You want to understand all options
- ✅ Educational context or research
- ✅ Interpretability is critical
- ✅ Validation on multiple test sets

**Avoid when**:
- ❌ Parameter space is large (> 1000)
- ❌ Time-sensitive analysis required
- ❌ Production forecasting system

### Auto ARIMA

**Use when**:
- ✅ Fast decision needed (< 5 minutes)
- ✅ Production forecasting systems
- ✅ Large parameter spaces
- ✅ No domain knowledge about parameters
- ✅ Good enough solution acceptable

**Avoid when**:
- ❌ Need guaranteed global optimum
- ❌ Model interpretability critical
- ❌ Research publication (need more thorough validation)

### Information Criteria Selection

**Choose AIC when**:
- Forecasting accuracy is primary goal
- Sample size is moderate
- Can afford slight overfitting risk
- Historical prediction success matters

**Choose BIC when**:
- Model simplicity is important
- Avoiding overfitting is priority
- Statistical inference on parameters matters
- Interpretability is valued

**Both important when**:
- Different criteria suggest different models
- Need to validate both approaches
- Model uncertainty is a concern

## Model Selection Strategy

### Recommended Approach

1. **Start with Auto ARIMA**
   - Fast overview of optimal region
   - Use BIC for parsimony

2. **Grid Search Around Best**
   - Refine search around auto_arima result
   - Check AIC vs BIC agreement
   - Evaluate final 3-5 candidates

3. **Out-of-Sample Validation**
   - Test on multiple test sets
   - Evaluate forecast accuracy vs simplicity
   - Use domain knowledge for final selection

4. **Final Model**
   - Balance between AIC, BIC, and RMSE
   - Prefer simpler models (Occam's Razor)
   - Validate residuals meet assumptions

## Residual Diagnostics for Selected Model

For the selected ARIMA model, perform:

1. **Autocorrelation Test** (Ljung-Box)
   - H₀: Residuals are white noise
   - ✓ Pass: p-value > 0.05

2. **Normality Test** (Jarque-Bera)
   - H₀: Residuals are normally distributed
   - ✓ Pass: p-value > 0.05

3. **Homoscedasticity** (constant variance)
   - Visual inspection: residual plot
   - ✓ Pass: No clear patterns

## Key Insights

### Information Criteria Trade-off

The fundamental trade-off in AIC vs BIC reflects philosophical differences:
- **AIC**: Frequentist approach, emphasis on prediction
- **BIC**: Bayesian approach, emphasis on model selection

With gold price data (n=2,011 large), BIC penalty k·ln(2011) ≈ 7k becomes very strong, making BIC much more conservative than AIC.

### Parameter Sensitivity

Grid search reveals:
- **d=1 most robust**: First differencing achieves stationarity
- **p, q low values**: Rarely benefit from p,q > 3
- **Flat regions**: Multiple models may perform similarly
- **Pareto frontier**: Some models dominate others (lower AIC AND RMSE)

### When Information Criteria Agree

Strong agreement (both select same model) indicates:
- ✓ High confidence in model selection
- ✓ Model likely robust
- ✓ Less risk of overfitting or underfitting
- ✓ Good choice for production use

### When They Disagree

Disagreement indicates:
- Complex parameter space
- Multiple local optima
- Trade-off between fit and parsimony
- Need for additional validation

## Practical Recommendation

For gold price forecasting:

1. **Start with Auto ARIMA** (fast, good results)
2. **Verify with Grid Search** (educational, confirm results)
3. **Compare AIC and BIC** (understand trade-off)
4. **Validate on holdout set** (ensure generalization)
5. **Use simpler model if similar RMSE** (Occam's Razor principle)

## Code Examples

### Grid Search Approach

```python
results = []
for p in range(6):
    for d in range(3):
        for q in range(6):
            model = ARIMA(data, order=(p,d,q))
            fitted = model.fit()
            results.append({
                'order': (p,d,q),
                'AIC': fitted.aic,
                'BIC': fitted.bic
            })
```

### Auto ARIMA Approach

```python
from pmdarima import auto_arima

model = auto_arima(
    data,
    start_p=0, max_p=5,
    start_d=0, max_d=2,
    start_q=0, max_q=5,
    information_criterion='bic'
)
print(model.order)  # Selected (p, d, q)
```

## Summary Table

| Aspect | Grid Search | Auto ARIMA |
|--------|-------------|-----------|
| **Computation Time** | ~2 minutes | ~1-2 minutes |
| **Parameter Combinations** | All 108 | ~20-30 iterative |
| **Result Transparency** | All models visible | Best model only |
| **Algorithm** | Exhaustive | Stepwise |
| **Optimal Result** | Guaranteed | High probability |
| **Use Case** | Learning/research | Production |
| **Interpretability** | High | Low |

## Next Steps

Day 17 will cover:
- **Seasonal ARIMA Extension**: SARIMA(p,d,q)(P,D,Q)s
- **Automated Seasonal Selection**: auto_arima with seasonal=True
- **Monthly/Quarterly Seasonality**: Identifying seasonal periods
- **SARIMA Parameter Grid**: (P,D,Q,s) parameter space

## References

- Akaike, H. (1974). "A New Look at the Statistical Model Identification"
- Schwarz, G. (1978). "Estimating the Dimension of a Model"
- Hyndman, R.J. & Khandakar, Y. (2008). "Automatic time series forecasting"
- pmdarima documentation: https://alkaline-ml.com/pmdarima/

## File Structure

```
day16/
├── main.py                 # Grid search and auto_arima implementation
├── README.md              # This file
├── notebooks/
│   └── main.ipynb         # Interactive exploration notebook
├── aic_bic_heatmap.html   # Information criteria heatmap (d=1)
└── aic_bic_scatter.html   # AIC vs BIC scatter plot
```

## Conclusion

Day 16 demonstrates that ARIMA parameter selection is both an art and science:
- **Science**: Information criteria provide mathematical framework
- **Art**: Domain knowledge and validation inform final choice
- **Practice**: Auto ARIMA offers excellent balance for production systems
- **Learning**: Grid search reveals parameter landscape and sensitivity

The grid search methodology provides educational value and transparency, while auto_arima offers practical efficiency. Both approaches have merit and can complement each other in a robust model selection pipeline.
