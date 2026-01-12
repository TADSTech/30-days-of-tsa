# Day 5: ADF Test & Differencing

## Project Overview
Statistical testing for stationarity using the Augmented Dickey-Fuller test and practical implementation of differencing to transform non-stationary data into stationary data suitable for ARIMA modeling.

## Key Concepts

### Augmented Dickey-Fuller (ADF) Test
A statistical hypothesis test for stationarity that tests the null hypothesis of a unit root:

- **Null Hypothesis (H₀)**: Time series has a unit root (non-stationary)
- **Alternative Hypothesis (H₁)**: Time series is stationary

**Decision Rule:**
- **p-value < 0.05**: Reject H₀ → Series is **stationary**
- **p-value ≥ 0.05**: Fail to reject H₀ → Series is **non-stationary**

### First-Order Differencing
Transforms a time series by computing the change between consecutive observations:

```
Price_Diff(t) = Price(t) - Price(t-1)
```

This removes trends and often achieves stationarity, converting a non-stationary series into a stationary one.

## Dataset
- **Source**: Gold price historical data
- **Period**: 10 years of daily prices
- **Data File**: gold_prices.csv

## Methodology

### 1. ADF Test on Original Data
- Load and preprocess gold price data
- Apply Augmented Dickey-Fuller test to raw prices
- Interpret p-value and critical values
- Confirm non-stationarity

### 2. Apply First-Order Differencing
- Compute differences between consecutive days
- Calculate new mean and standard deviation
- Compare statistics between original and differenced data

### 3. Visual Comparison
- Create 2-panel plot showing:
  - Top panel: Original prices with upward trend
  - Bottom panel: Differenced prices fluctuating around zero
- Add horizontal reference line at y=0

### 4. ADF Test on Differenced Data
- Re-run ADF test on differenced series
- Compare p-values before and after differencing
- Confirm stationarity achievement

## Key Findings

### Original Gold Prices
- **p-value**: High (typically > 0.05)
- **Status**: Non-stationary
- **Characteristics**:
  - Strong upward trend over 10 years
  - Mean increases significantly over time
  - High autocorrelation (values depend on recent history)
  - Not suitable for ARIMA modeling without transformation

### Differenced Gold Prices
- **p-value**: Very low (typically < 0.01)
- **Status**: Stationary
- **Characteristics**:
  - Fluctuates around constant mean (near zero)
  - Stable variance
  - Trend component removed
  - Ready for ARIMA modeling

## Statistical Interpretation

**Why the p-value Changed:**
1. **Trend Removal**: Differencing eliminated the upward trend
2. **Constant Mean**: Differenced series has stable average
3. **Reduced Autocorrelation**: Daily changes are less dependent on history
4. **Unit Root Test**: ADF test now rejects the null hypothesis of non-stationarity

## ARIMA Implications

**ARIMA(p,d,q) Model Parameters:**
- **p**: Autoregressive order (AR)
- **d**: Differencing order (I) = **1** in our case
- **q**: Moving average order (MA)

Since we applied first-order differencing (d=1), future ARIMA models would use **ARIMA(p,1,q)** format.

## Technical Implementation

### Libraries Used
- **statsmodels**: `adfuller` function for ADF testing
- **pandas**: Data manipulation and differencing
- **plotly**: Visualization of original vs differenced data
- **numpy**: Numerical operations

### Key Functions
- `adfuller()`: Performs Augmented Dickey-Fuller test
- `.diff()`: Computes first-order differencing
- `make_subplots()`: Creates multi-panel visualizations

## Insights

1. **Statistical Validation**: ADF test provides rigorous statistical evidence for stationarity assessment
2. **Differencing Effectiveness**: First-order differencing successfully transformed non-stationary data
3. **Data Transformation**: Proper preprocessing is essential before applying forecasting models
4. **Ready for Modeling**: Differenced data meets assumptions for ARIMA models

## Applications

- **Hypothesis Testing**: Statistical confirmation of stationarity
- **Data Preprocessing**: Preparing data for time series models
- **Model Parameter Selection**: Determining the "d" parameter in ARIMA(p,d,q)
- **Forecasting Pipeline**: Establishing best practices for model building

## Next Steps

- Perform ACF/PACF analysis on differenced data to determine p and q parameters
- Fit ARIMA models with various parameter combinations
- Implement grid search for optimal parameter selection
- Validate model performance on test data
- Generate price forecasts

## Conclusion

This project demonstrates the complete workflow for stationarizing time series data:
1. Test for stationarity using ADF test
2. Apply differencing transformation
3. Verify stationarity achievement
4. Prepare data for forecasting models

The successful transformation of gold prices from non-stationary to stationary provides the foundation for building reliable ARIMA forecasting models. Understanding when and how to difference data is a critical skill in time series analysis.
