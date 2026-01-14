# 30 Days of TSA( Time Series Analysis )

A personal challenge to work learn and practice Time Series Analysis (TSA) skills over 30 days through weekly projects.

## Overview

This project focuses on building and enhancing Time Series Analysis skills through a series of weekly projects. Each day will involve working on a different aspect of TSA, including data cleaning, exploratory data analysis, visualization, and modeling. The goal is to develop a comprehensive understanding of TSA techniques and best practices while creating a portfolio of work.

## weekly projects

- [Day 1: Time Series Fundamentals](./day1/)
- [Day 2: Time Series Data from APIs](./day2/)
- [Day 3: Visualizing Time Series Data](./day3/)

## Project Details

### Day 1: Time Series Fundamentals

**Status**: ✓ Complete

**Objective**: Analyze Daily Delhi Climate data to understand seasonal trends and time series patterns

**Key Deliverables**:
- Time series visualization of temperature, humidity, wind speed, and pressure
- Identification of seasonal trends and patterns
- Data preprocessing and exploratory analysis
- Interactive Plotly visualizations

**Outcomes**:
- Successfully identified clear seasonal patterns in temperature and humidity
- Demonstrated effective use of pandas for data manipulation
- Created reusable visualization pipeline
- Documented insights for practical applications

### Day 2: Time Series Data from APIs

**Status**: ✓ Complete

**Objective**: Fetch and analyze financial time series data from Yahoo Finance API

**Key Deliverables**:
- Integration with yfinance API for real-time financial data
- Gold (GLD) price data extraction and cleaning
- 10-year historical price analysis
- Interactive financial visualization with proper scaling
- Comprehensive data cleaning pipeline

**Outcomes**:
- Successfully fetched and processed 10 years of gold price data
- Implemented robust data cleaning for API-sourced datasets
- Created publication-quality visualizations for financial time series
- Demonstrated practical API integration in time series analysis

### Day 3: Visualizing Time Series Data

**Status**: ✓ Complete

**Objective**: Master advanced visualization techniques for time series data

**Key Deliverables**:
- Multiple time series visualization approaches (line plots, moving averages)
- Rolling statistics analysis (30-day and 90-day moving averages)
- Seasonal decomposition into trend, seasonal, and residual components
- Interactive multi-panel visualizations
- Volatility measurement and visualization

**Outcomes**:
- Successfully created comprehensive visualization toolkit
- Demonstrated seasonal decomposition for pattern recognition
- Implemented moving averages for trend identification
- Created publication-quality interactive charts with Plotly

### Day 4: Stationarity Concepts

**Status**: ✓ Complete

**Objective**: Understand and test for stationarity in time series data

**Key Deliverables**:
- Educational explanation of weak stationarity (constant mean, variance, autocorrelation)
- Visual comparison of stationary vs non-stationary synthetic data
- Rolling statistics visualization for stationarity detection
- Autocorrelation Function (ACF) analysis and interpretation
- Foundation for data preprocessing before ARIMA modeling

**Outcomes**:
- Successfully demonstrated stationarity concepts with synthetic and real data
- Identified gold prices as non-stationary with visual and statistical evidence
- Provided practical tools for stationarity assessment
- Established preprocessing requirements for forecasting models

### Day 5: ADF Test & Differencing

**Status**: ✓ Complete

**Objective**: Apply statistical testing and transformation techniques to achieve stationarity

**Key Deliverables**:
- Augmented Dickey-Fuller (ADF) test implementation for statistical hypothesis testing
- ADF helper function for reproducible stationarity testing
- First-order differencing transformation of gold prices
- 2-panel visualization comparing original vs differenced data
- Statistical interpretation of p-value changes

**Outcomes**:
- Successfully tested non-stationarity of original gold prices using ADF
- Applied differencing to transform data into stationary form
- Confirmed stationarity achievement with ADF test (p-value < 0.01)
- Established ARIMA model readiness with d=1 parameter
- Created reproducible pipeline for data preprocessing

### Day 6: Autocorrelation Analysis (ACF & PACF)

**Status**: ✓ Complete

**Objective**: Analyze autocorrelation patterns to identify optimal ARIMA parameters

**Key Deliverables**:
- ACF and PACF conceptual explanation (MA vs AR order identification)
- Side-by-side correlograms for 40 lags with confidence intervals
- Visual interpretation guide for cutoff vs decay patterns
- Systematic parameter identification methodology
- White noise detection and ARIMA(0,1,0) identification

**Outcomes**:
- Successfully generated ACF/PACF visualizations for differenced gold prices
- Discovered white noise behavior in differenced gold prices (no significant autocorrelations)
- Identified ARIMA(0,1,0) random walk model as optimal for gold price data
- Validated Efficient Market Hypothesis—gold price changes are random
- Ready for ARIMA fitting, forecasting, and model validation

### Day 7: Exploratory Data Analysis & Gold-Themed Dashboard

**Status**: ✓ Complete

**Objective**: Conduct comprehensive EDA and create professional visualizations

**Key Deliverables**:
- Dataset overview and statistical summary (2,515 daily observations)
- Daily returns and volatility metrics analysis
- Seasonal pattern identification by month, quarter, and year
- Interactive 6-panel gold-themed dashboard using Plotly
- Cumulative returns and moving average analysis

**Dashboard Components**:
1. Price trend over time (304% gain from 2016-2025)
2. Price distribution histogram (right-skewed, μ=$1,857)
3. Daily returns distribution (μ=+0.059%, σ=0.933%)
4. Monthly volatility tracking (14.82% annualized)
5. Cumulative returns visualization
6. 50-day and 200-day moving averages

**Key Findings**:
- Gold shows strong upward trend with acceleration post-2020
- Positive daily return bias: 52.6% of days are up (mean +0.0591%)
- Seasonal strength: Q4 highest ($1,957 avg), Q1 lowest ($1,761 avg)
- Yearly acceleration: 2025 showed +43% gain vs 2016-2019 stability
- Volatility spikes during market stress periods (2020, 2023)

**Outcomes**:
- Comprehensive visual understanding of gold price dynamics
- Professional gold-themed dashboard suitable for presentations
- Temporal pattern insights for business decisions
- Validation of data quality and preprocessing effectiveness
- Ready to build and evaluate forecasting models

## Goals

- Understand the basics of Time Series data
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis (EDA)
- Create meaningful visualizations
- Extract actionable insights from time series data
- Build a portfolio of TSA projects

## Tools Used

- Python (pandas, plotly, numpy)
- Jupyter Notebooks
- Scikit-learn (ML pipelines and ensemble methods)
- Statsmodels (Time Series Analysis)
- Time Series specific libraries (fbprophet, pmdarima)

## Progress

Current Day: 6/30