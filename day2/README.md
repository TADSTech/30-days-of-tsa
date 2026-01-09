# Day 2: Time Series Data from APIs (Yfinance)

## Project Overview
Analysis of gold prices using real-world financial data from Yahoo Finance via the yfinance library. This project demonstrates how to fetch, clean, and visualize time series data from external APIs.

## Dataset
- **Source**: Yahoo Finance (via yfinance)
- **Ticker**: GLD (SPDR Gold Shares ETF)
- **Period**: 10 years of daily historical data
- **Data File**: gold_data_10y.csv

## Methodology
1. **API Integration**: Used yfinance to fetch 10 years of GLD price history
2. **Data Cleaning**: 
   - Removed missing values (market closures)
   - Dropped corrupted header rows
   - Converted date columns to datetime format
3. **Data Processing**: 
   - Set date as index
   - Adjusted price data for proper scaling
   - Rounded values for clarity
4. **Visualization**: Created interactive Plotly line chart for Jan 2024 - Jan 2026 period

## Key Findings

### Gold Price Trends
- **Recent Period (2024-2026)**: Gold prices show strong upward momentum
- **Volatility**: Moderate volatility with periodic corrections
- **Range**: Price fluctuations within the $1000-$5000 USD range
- **Market Drivers**: Reflects global economic uncertainty, inflation, and geopolitical factors

## Technical Implementation

### Libraries Used
- **yfinance**: Fetching historical financial data
- **pandas**: Data manipulation and cleaning
- **plotly**: Interactive visualization
- **numpy**: Numerical operations

### Data Processing Steps
1. Download 10-year GLD history using `yf.download()`
2. Extract closing prices
3. Handle missing data points
4. Rename and clean columns
5. Convert to proper datetime index
6. Filter for desired date range
7. Scale and format for visualization

## Insights
This project demonstrates:
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
