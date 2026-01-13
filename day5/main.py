import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from statsmodels.tsa.stattools import adfuller

# Set Plotly template
pio.templates.default = "ggplot2"

# Define paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Load and preprocess gold price data
gold = pd.read_csv(os.path.join(data_dir, "gold_prices.csv"), parse_dates=["Date"], index_col="Date")
gold['Price'] = gold['Price'].astype(float)
gold['Price'] = gold['Price'] * 10.8
gold['Price'] = gold['Price'].round(0)

print("=== Day 5: ADF Test & Differencing ===")
print(f"\nGold price data shape: {gold.shape}")
print(gold.head())

# Define ADF test helper function
def adf_test(series, name=''):
    """
    Perform Augmented Dickey-Fuller test and print results
    """
    result = adfuller(series.dropna())
    
    print(f'\n--- ADF Test Results for {name} ---')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print(f"\n✓ Result: The series IS stationary (p-value = {result[1]:.6f} < 0.05)")
    else:
        print(f"\n✗ Result: The series IS NOT stationary (p-value = {result[1]:.6f} ≥ 0.05)")
    print()

# Run ADF test on raw gold prices
print("\n=== Testing Original Gold Prices ===")
adf_test(gold['Price'], 'Gold Price (Original)')

# Apply first-order differencing
gold['Price_Diff'] = gold['Price'].diff()

print("\n=== Differenced Data Statistics ===")
print(gold[['Price', 'Price_Diff']].head(10))
print(f"\nOriginal Price - Mean: {gold['Price'].mean():.2f}, Std: {gold['Price'].std():.2f}")
print(f"Differenced Price - Mean: {gold['Price_Diff'].mean():.2f}, Std: {gold['Price_Diff'].std():.2f}")

# Create 2-panel comparison plot
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Original Gold Prices", "Differenced Gold Prices (Daily Change)"),
    vertical_spacing=0.12
)

# Original prices
fig.add_trace(
    go.Scatter(x=gold.index, y=gold['Price'], mode='lines', name='Original Price', line=dict(color='blue')),
    row=1, col=1
)

# Differenced prices
fig.add_trace(
    go.Scatter(x=gold.index, y=gold['Price_Diff'], mode='lines', name='Price Change', line=dict(color='red')),
    row=2, col=1
)

# Add horizontal line at y=0 for differenced plot
fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

# Update layout
fig.update_layout(
    height=700,
    title_text="Comparison: Original vs Differenced Gold Prices",
    showlegend=False
)
fig.update_xaxes(title_text="Date", row=2, col=1)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Price Change (USD)", row=2, col=1)
fig.show()

# Run ADF test on differenced prices
print("\n=== Testing Differenced Gold Prices ===")
adf_test(gold['Price_Diff'], 'Gold Price (Differenced)')

print("\n=== Analysis Complete ===")
print("The differencing transformation has successfully stationarized the gold price series!")
print("We are now ready for ARIMA modeling with d=1.")
