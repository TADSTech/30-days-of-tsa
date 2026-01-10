# Day 3: Visualizing Time Series Data

## Project Overview
Advanced visualization techniques for time series data using gold price data. This project demonstrates multiple approaches to visualizing financial time series including line plots, moving averages, and seasonal decomposition.

## Dataset
- **Source**: Historical gold price data
- **Ticker**: GLD (SPDR Gold Shares ETF)
- **Period**: Jan 2016 - Jan 2026 (10 years)
- **Data File**: gold_prices.csv

## Methodology

### 1. Data Preparation
- Load gold price data from CSV with date index
- Convert price to numeric format
- Apply scaling factor (10.8x) for USD representation
- Round values for clarity

### 2. Visualization Techniques

#### Line Plot Visualization
- Full time series display of gold prices
- Interactive Plotly charts for detailed exploration
- Grid overlays for easier reading
- Custom axis formatting ($200 increments, $1000-$5000 range)

#### Rolling Statistics (Moving Averages)
- **30-Day Moving Average**: Short-term trend indicator
- **90-Day Moving Average**: Medium-term trend indicator  
- **30-Day Standard Deviation**: Volatility measurement
- Overlaid on original price to show smoothing effect

#### Seasonal Decomposition
- Monthly resampling for clearer seasonal patterns
- Additive decomposition model
- 12-month seasonal period (annual patterns)
- Separation into: Observed, Trend, Seasonal, Residuals components

## Key Findings

### Trend Analysis
- **Long-term Uptrend**: Gold prices show consistent upward trend over 10 years
- **Recent Rally**: Strong acceleration in 2024-2026 period
- **Support Levels**: Identifiable price floors and resistance points

### Seasonal Patterns
- **Annual Seasonality**: Gold prices exhibit yearly cyclical patterns
- **Trend Component**: Strong positive trend masks seasonal variations
- **Residuals**: Relatively small residuals indicate good decomposition fit

### Volatility Insights
- **Short-term Volatility**: 30-day moving average captures daily fluctuations
- **Smoothing Effect**: 90-day MA reveals underlying trend
- **Risk Assessment**: Standard deviation quantifies price variability

## Technical Implementation

### Libraries Used
- **pandas**: Data loading and time series operations
- **plotly**: Interactive visualizations
- **statsmodels**: Seasonal decomposition
- **plotly.subplots**: Multi-panel visualizations

### Visualization Types
1. **Simple Line Plot**: Overview of entire time series
2. **Multi-line Plot**: Price vs moving averages comparison
3. **Subplot Decomposition**: Four-panel seasonal decomposition view

## Insights
This project demonstrates:
- Multiple approaches to time series visualization
- Importance of moving averages in trend identification
- Seasonal decomposition for understanding underlying components
- Interactive tools for time series exploration
- Practical visualization best practices (gridlines, axis formatting)

## Applications
- Investment decision making (trend and volatility analysis)
- Risk assessment through volatility measurement
- Pattern recognition for forecasting
- Portfolio analysis and rebalancing

## Future Enhancements
- Bollinger Bands for volatility bands
- Relative Strength Index (RSI) for momentum
- MACD (Moving Average Convergence Divergence)
- Autocorrelation and partial autocorrelation plots
- Comparative analysis with other assets
- Interactive dashboard with multiple indicators

## Conclusion
This project establishes a comprehensive toolkit for time series visualization. By combining multiple visualization techniques, we can gain deeper insights into price movements, trends, seasonality, and volatility. These visualizations form the foundation for more sophisticated time series analysis and forecasting.

- Real-time access to financial time series data via APIs
- Essential data cleaning techniques for real-world datasets
- Interactive visualization of financial time series
- Practical considerations for financial data analysis (market closures, data anomalies, etc.)

## Future Enhancements
- Moving average analysis (20-day, 50-day, 200-day)
- Volatility calculations (standard deviation, rolling volatility)
- Multi-asset comparison (gold vs other commodities)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Forecasting models using ARIMA or Prophet
- Correlation analysis with other financial assets

## Conclusion
Using yfinance provides a seamless way to access financial time series data for analysis. This project establishes a foundation for more complex financial time series analysis and demonstrates the power of combining Python's data science ecosystem with real-world API data.
