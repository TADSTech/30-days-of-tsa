# Day 4: Stationarity Concepts

## Project Overview
Introduction to the concept of stationarity in time series analysis. This project demonstrates why stationarity is critical for forecasting models like ARIMA and teaches practical methods to identify non-stationary data.

## Key Concepts

### What is Stationarity?
A time series is **weakly stationary** if it satisfies three conditions:

1. **Constant Mean**: The average value remains constant over time (no trend)
2. **Constant Variance**: The variability around the mean remains constant (homoscedasticity)
3. **Constant Autocorrelation**: The relationship between observations depends only on time lag, not absolute time

### Why Stationarity Matters
Forecasting models like **ARIMA** require stationary data because:
- They assume constant statistical properties over time
- Non-stationary data produces unreliable predictions
- Trends and changing variance violate model assumptions

## Dataset
- **Source**: Gold price historical data
- **Period**: 10 years of daily prices
- **Data File**: gold_prices.csv

## Methodology

### 1. Visual Inspection - Synthetic Data
Generated three types of synthetic time series:
- **Stationary**: White noise with constant mean and variance
- **Non-Stationary (Trend)**: Linear trend added to white noise
- **Non-Stationary (Heteroscedastic)**: Expanding variance over time

### 2. Visual Inspection - Rolling Statistics
Calculated moving window statistics for gold prices:
- **30-day Rolling Mean**: Shows overall trend direction
- **30-day Rolling Std**: Indicates changing volatility
- **Visual Comparison**: Overlaid on original prices to detect non-stationarity

### 3. Autocorrelation Function (ACF)
Used ACF plots to compare:
- **Stationary Series**: ACF decays quickly to zero
- **Non-Stationary Series**: ACF decays very slowly (strong persistence)

## Key Findings

### Synthetic Data Analysis
- Stationary white noise shows rapid, random fluctuations
- Trending series shows clear directional movement
- Heteroscedastic series shows increasing spread over time

### Gold Price Analysis
- **Status**: Non-Stationary
- **Evidence**: 
  - Strong upward trend in price level
  - Rolling mean shows clear trend component
  - Rolling standard deviation varies significantly
  - ACF remains high for many lags (slow decay)

### ACF Interpretation
- **Stationary ACF**: Values drop sharply within confidence bands
- **Non-Stationary ACF**: Values decrease very slowly, indicating trend-driven behavior

## Why We Must Stationarize Data

Before applying ARIMA or similar models, non-stationary data must be transformed:

**Common Techniques:**
1. **Differencing**: Compute price changes (today - yesterday)
2. **Detrending**: Remove trend component
3. **Deseasonalizing**: Remove seasonal patterns
4. **Log Transformation**: Stabilize variance

## Technical Implementation

### Libraries Used
- **pandas**: Data loading and manipulation
- **numpy**: Synthetic data generation
- **plotly**: Interactive visualizations
- **statsmodels**: ACF calculations and plotting
- **matplotlib**: Multiple ACF comparison plots

### Key Functions
- `rolling()`: Calculate moving statistics
- `plot_acf()`: Generate ACF plots
- `make_subplots()`: Create multi-panel visualizations

## Insights

1. **Visual Patterns Matter**: Plots reveal non-stationarity more intuitively than statistics
2. **Rolling Statistics Are Practical**: Easy-to-implement method for quick stationarity checks
3. **ACF Is Quantitative**: Provides rigorous statistical evidence
4. **Gold Prices Are Non-Stationary**: Clear evidence requiring transformation before modeling

## Applications

- Preprocessing data for time series forecasting
- Validating ARIMA model assumptions
- Detecting structural breaks and regime changes
- Assessing portfolio stability

## Future Work

- Implement differencing to stationarize gold prices
- Apply Augmented Dickey-Fuller (ADF) test for statistical confirmation
- Compare different differencing orders (1st, 2nd derivative)
- Develop ARIMA models on stationarized data
- Implement KPSS test as alternative stationarity test

## Conclusion

Understanding stationarity is fundamental to time series analysis. This project demonstrates that gold prices are non-stationary and establishes the foundation for transforming data into a form suitable for forecasting models. The combination of visual inspection, rolling statistics, and ACF analysis provides a comprehensive toolkit for stationarity assessment in any time series dataset.
