"""
Day 7: Exploratory Data Analysis (EDA) & Gold-Themed Dashboard
=================================================================

This script performs comprehensive exploratory data analysis on gold price data
and generates a professional gold-themed dashboard with 6 key visualizations.

Key Analyses:
- Dataset overview and statistics
- Daily returns and volatility metrics
- Price distribution analysis
- Seasonal patterns and trends
- Moving average crossovers
- Cumulative returns tracking

Data: Historical gold prices (2016-2026, 2515 daily observations)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Load and preprocess data
gold = pd.read_csv("data/gold_prices.csv", parse_dates=["Date"], index_col="Date")
gold['Price'] = gold['Price'].astype(float)
gold['Price'] = gold['Price'] * 10.8
gold['Price'] = gold['Price'].round(0)

# Calculate derivatives
gold['Daily_Return'] = gold['Price'].pct_change() * 100
gold['Daily_Change'] = gold['Price'].diff()
gold['Year'] = gold.index.year
gold['Month'] = gold.index.month
gold['Quarter'] = gold.index.quarter
gold['DayOfWeek'] = gold.index.dayofweek


def print_dataset_overview():
    """Print dataset overview and basic statistics."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {gold.shape}")
    print(f"Date Range: {gold.index.min()} to {gold.index.max()}")
    print(f"Total Days: {len(gold)}")
    print(f"Data Types:\n{gold[['Price']].dtypes}")
    print(f"Missing Values: {gold['Price'].isnull().sum()}\n")


def print_statistical_summary():
    """Print statistical summary of gold prices."""
    print("=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    summary = gold['Price'].describe()
    print(summary)
    print(f"\nSkewness: {gold['Price'].skew():.4f}")
    print(f"Kurtosis: {gold['Price'].kurtosis():.4f}\n")


def print_returns_analysis():
    """Print daily returns and change analysis."""
    print("=" * 60)
    print("DAILY RETURNS & CHANGES")
    print("=" * 60)
    print(f"Mean Daily Return: {gold['Daily_Return'].mean():.4f}%")
    print(f"Std Dev Daily Return: {gold['Daily_Return'].std():.4f}%")
    print(f"Min Daily Return: {gold['Daily_Return'].min():.4f}%")
    print(f"Max Daily Return: {gold['Daily_Return'].max():.4f}%\n")


def print_volatility_insights():
    """Print volatility and trend insights."""
    price_range = gold['Price'].max() - gold['Price'].min()
    price_range_pct = (price_range / gold['Price'].min()) * 100

    print("=" * 60)
    print("VOLATILITY & TREND INSIGHTS")
    print("=" * 60)
    print(f"Price Range: ${gold['Price'].min():.2f} - ${gold['Price'].max():.2f}")
    print(f"Total Range: ${price_range:.2f} ({price_range_pct:.1f}% increase)")
    print(f"Current Price: ${gold['Price'].iloc[-1]:.2f}")
    print(f"\nVolatility Metrics:")
    print(f"  Daily Volatility (σ): {gold['Daily_Return'].std():.4f}%")
    print(f"  Annualized Volatility: {gold['Daily_Return'].std() * np.sqrt(252):.2f}%")

    up_days = (gold['Daily_Change'] > 0).sum()
    down_days = (gold['Daily_Change'] < 0).sum()
    total_trading_days = up_days + down_days

    print(f"\nMarket Direction:")
    print(f"  Up Days: {up_days} ({(up_days/total_trading_days)*100:.1f}%)")
    print(f"  Down Days: {down_days} ({(down_days/total_trading_days)*100:.1f}%)")

    winning_days_pct = (gold['Daily_Return'] > 0).sum()
    losing_days_pct = (gold['Daily_Return'] < 0).sum()

    print(f"\nReturn Distribution:")
    print(f"  Days with Positive Returns: {winning_days_pct} ({(winning_days_pct/len(gold))*100:.1f}%)")
    print(f"  Days with Negative Returns: {losing_days_pct} ({(losing_days_pct/len(gold))*100:.1f}%)\n")


def print_seasonal_patterns():
    """Print seasonal patterns and yearly performance."""
    print("=" * 60)
    print("SEASONAL PATTERNS")
    print("=" * 60)

    monthly_avg = gold.groupby('Month')['Price'].mean()
    quarterly_avg = gold.groupby('Quarter')['Price'].mean()

    print("\nAverage Price by Month:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, price in monthly_avg.items():
        print(f"  {months[month-1]}: ${price:.2f}")

    print("\nAverage Price by Quarter:")
    quarter_names = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
    for quarter, price in quarterly_avg.items():
        print(f"  {quarter_names[quarter-1]}: ${price:.2f}")

    yearly_avg = gold.groupby('Year')['Price'].agg(['mean', 'min', 'max'])
    print("\nYearly Performance:")
    print(yearly_avg.to_string())
    print()


def create_dashboard():
    """Create a comprehensive gold-themed dashboard with 6 visualizations."""
    # Gold-themed color palette
    gold_color_primary = "#FFD700"  # Gold
    gold_color_dark = "#DAA520"     # Goldenrod
    gold_color_light = "#FFF8DC"    # Cornsilk
    bg_color = "#1a1a1a"            # Dark background
    text_color = "#ffffff"          # White text

    # Create comprehensive dashboard with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Price Trend Over Time",
            "Price Distribution",
            "Daily Returns Distribution",
            "Monthly Volatility",
            "Cumulative Returns",
            "Price Moving Averages"
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # 1. Price Trend
    fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=gold['Price'],
            mode='lines',
            name='Gold Price',
            line=dict(color=gold_color_primary, width=2),
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. Price Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=gold['Price'],
            name='Price Distribution',
            marker=dict(color=gold_color_dark),
            nbinsx=40,
            hovertemplate='<b>Price Range:</b> $%{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Daily Returns Distribution
    fig.add_trace(
        go.Histogram(
            x=gold['Daily_Return'].dropna(),
            name='Daily Returns',
            marker=dict(color=gold_color_primary),
            nbinsx=50,
            hovertemplate='<b>Return:</b> %{x:.2f}%<br><b>Frequency:</b> %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. Monthly Volatility
    monthly_volatility = gold['Daily_Return'].resample('ME').std()
    fig.add_trace(
        go.Bar(
            x=monthly_volatility.index,
            y=monthly_volatility.values,
            name='Monthly Volatility',
            marker=dict(color=gold_color_dark),
            hovertemplate='<b>Month:</b> %{x|%B %Y}<br><b>Volatility:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=2
    )

    # 5. Cumulative Returns
    cumulative_returns = ((1 + gold['Daily_Return'] / 100).cumprod() - 1) * 100
    fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color=gold_color_primary, width=2),
            fill='tozeroy',
            fillcolor=f'rgba(255, 215, 0, 0.2)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )

    # 6. Moving Averages
    ma_50 = gold['Price'].rolling(window=50).mean()
    ma_200 = gold['Price'].rolling(window=200).mean()

    fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=gold['Price'],
            mode='lines',
            name='Price',
            line=dict(color=gold_color_primary, width=1),
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ),
        row=3, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=ma_50,
            mode='lines',
            name='50-Day MA',
            line=dict(color=gold_color_dark, width=2, dash='dash'),
            hovertemplate='<b>Date:</b> %{x}<br><b>MA50:</b> $%{y:.2f}<extra></extra>'
        ),
        row=3, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=gold.index,
            y=ma_200,
            mode='lines',
            name='200-Day MA',
            line=dict(color='#FFB6C1', width=2, dash='dot'),
            hovertemplate='<b>Date:</b> %{x}<br><b>MA200:</b> $%{y:.2f}<extra></extra>'
        ),
        row=3, col=2
    )

    # Update layout with gold theme
    fig.update_layout(
        title={
            'text': '<b>Gold Price Analysis Dashboard</b><br><sub>Comprehensive Exploratory Data Analysis</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': gold_color_primary}
        },
        height=1400,
        showlegend=True,
        template='plotly_dark',
        hovermode='x unified',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(0, 0, 0, 0.5)',
            bordercolor=gold_color_primary,
            borderwidth=1
        )
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 215, 0, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 215, 0, 0.1)')

    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Price (USD)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_xaxes(title_text="Daily Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_yaxes(title_text="Price (USD)", row=3, col=2)

    fig.show()
    return fig


if __name__ == "__main__":
    print("\n")
    print_dataset_overview()
    print_statistical_summary()
    print_returns_analysis()
    print_volatility_insights()
    print_seasonal_patterns()
    print("\nGenerating interactive dashboard...")
    create_dashboard()
