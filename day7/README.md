# Day 7: Exploratory Data Analysis (EDA) & Gold-Themed Dashboard

## Overview

On Day 7, we shift from statistical testing to **visual exploration** through a comprehensive Exploratory Data Analysis (EDA) and an interactive dashboard. This approach leverages the human brain's ability to recognize patterns, trends, and anomalies through visualization.

## Learning Objectives

- Understand dataset structure and distributions
- Identify seasonal patterns and trends
- Visualize volatility and return characteristics
- Create professional, theme-aligned dashboards
- Extract actionable insights from multi-dimensional data

## Key Concepts

### Why EDA Matters in Time Series

EDA is crucial for time series because it:
1. **Reveals Hidden Patterns**: Visualizations expose seasonality, trends, and cycles invisible in raw numbers
2. **Guides Model Selection**: Understanding data characteristics informs which models to use
3. **Identifies Anomalies**: Outliers and breaks can significantly impact forecasting
4. **Validates Preprocessing**: Confirms that transformations (differencing, scaling) worked correctly
5. **Tells the Data Story**: Communicates findings effectively to stakeholders

### Gold-Themed Design Choices

The dashboard uses a gold color scheme (#FFD700 primary, #DAA520 secondary) that:
- Matches the subject matter (gold prices)
- Creates visual cohesion and professionalism
- Improves readability with high contrast on dark backgrounds
- Symbolizes value and importance of the analysis

## Dashboard Components

### 1. Price Trend Over Time
- **Purpose**: Shows the long-term trajectory of gold prices
- **Insight**: Gold prices show strong upward trend especially post-2020
- **Finding**: Price increased from ~$1,113 (2016 low) to $4,501 (2025 high) — a 304% gain

### 2. Price Distribution
- **Purpose**: Reveals the shape and range of price values
- **Insight**: Right-skewed distribution indicating most prices cluster in lower range with some extreme highs
- **Metrics**: 
  - Skewness: 1.7198 (moderately right-skewed)
  - Kurtosis: 2.9658 (heavier tails than normal)

### 3. Daily Returns Distribution
- **Purpose**: Shows return characteristics and volatility
- **Insight**: Nearly normal distribution centered around 0.059% daily return
- **Key Metrics**:
  - Mean: +0.0591% per day
  - Std Dev: 0.9333% (volatility)
  - Annualized Volatility: 14.82% (moderate risk asset)
  - Min: -6.43% | Max: +4.93%

### 4. Monthly Volatility
- **Purpose**: Tracks how volatility changes over time
- **Insight**: Volatility spikes during market stress periods (2020 pandemic, 2023 banking crisis)
- **Pattern**: Generally stable ~0.8-1.0% monthly, with occasional 2-3% spikes

### 5. Cumulative Returns
- **Purpose**: Shows total wealth accumulation from passive holding
- **Insight**: Strong compound growth especially from 2020 onwards
- **Finding**: $100 invested in 2016 would grow to ~$400+ by 2026

### 6. Price Moving Averages
- **Purpose**: Smooths noise and reveals trend direction
- **Interpretation**:
  - Price above both MAs = Strong uptrend
  - Price between MAs = Mixed signals
  - Price below both MAs = Downtrend
  - MA50 crosses MA200 = Potential trend change

## Key Findings from EDA

### Market Direction & Volatility
- **Up Days**: 1,324 (54.2%) vs Down Days: 1,117 (45.8%)
- **Positive Return Days**: 1,324 (52.6%) vs Negative: 1,117 (44.4%)
- **Implication**: Positive skew in daily movements—gold tends to drift higher

### Seasonal Patterns
**Strongest by Quarter**: Q4 (Oct-Dec) with average $1,957 per troy oz
**Weakest by Quarter**: Q1 (Jan-Mar) with average $1,761 per troy oz
**Month Ranking** (highest to lowest):
1. December: $1,991
2. November: $1,927
3. October: $1,954
4. September: $1,913

**Implication**: Gold prices tend to strengthen in Q3-Q4, possibly due to seasonal demand and winter/year-end factors.

### Yearly Acceleration
- **2016-2019**: Relatively flat (~$1,290-$1,420 range)
- **2020 Jump**: +38% increase (pandemic safe-haven demand)
- **2024**: +31% increase year-over-year
- **2025**: Extraordinary +43% surge in first year of data
- **Trend**: Acceleration intensifies over time

## Statistical Summary
```
Price Statistics:
- Count: 2,515 observations (10 years of daily data)
- Mean: $1,856.81
- Std Dev: $662.43
- Min: $1,113.00
- 25th Percentile: $1,331.50
- Median: $1,778.00
- 75th Percentile: $1,977.50
- Max: $4,501.00
```

## Visualization Interpretation Tips

1. **Trend**: Look at the overall direction (upward, downward, flat)
2. **Seasonality**: Check if patterns repeat at regular intervals
3. **Volatility**: Notice periods of stable vs. erratic movement
4. **Anomalies**: Identify sudden spikes or crashes that need investigation
5. **Relationships**: See how different series move together (correlations)

## Code Structure

The analysis is organized in three parts:

1. **Data Loading & Preprocessing**: 
   - Reads CSV with date indexing
   - Applies price scaling factor
   - Computes derivatives (returns, changes)

2. **Analysis Functions**:
   - `print_dataset_overview()`: Basic info
   - `print_statistical_summary()`: Descriptive stats
   - `print_returns_analysis()`: Return metrics
   - `print_volatility_insights()`: Volatility & trend
   - `print_seasonal_patterns()`: Temporal patterns

3. **Dashboard Creation**:
   - 3×2 subplot grid with Plotly
   - Gold-themed color palette
   - Interactive hover information
   - Professional styling

## Next Steps (Day 8)

Having thoroughly understood our data through EDA, we're ready to:
1. **Fit ARIMA(0,1,0)** model (random walk with drift)
2. **Generate forecasts** with confidence intervals
3. **Validate model performance** on test set
4. **Visualize predictions** alongside actual data

## Conclusion

EDA transforms raw data into understanding. By visualizing distributions, trends, seasonality, and volatility, we gain intuition about gold prices that statistical tests alone cannot provide. The gold-themed dashboard presents these insights in an accessible, professional format suitable for presentations and reports.

The data tells a clear story: gold prices have experienced significant structural shifts post-2020, with increasing volatility and consistent upward bias. These patterns will inform our ARIMA modeling choices and help us build more accurate forecasts.
