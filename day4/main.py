import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

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

print(gold.head())

# Generate synthetic data to visualize stationarity
np.random.seed(42)
n_samples = 200

# Stationary Series: White Noise
stationary_series = np.random.normal(loc=0, scale=1, size=n_samples)

# Non-Stationary Series: Trend
trend_series = np.linspace(0, 10, n_samples) + stationary_series

# Non-Stationary Series: Expanding Variance (Heteroscedasticity)
expanding_variance_series = stationary_series * np.linspace(1, 5, n_samples)

# Create 3-panel subplot for visualization
fig = make_subplots(
    rows=3, cols=1, 
    subplot_titles=("Stationary (White Noise)", "Non-Stationary (Trend)", "Non-Stationary (Expanding Variance)")
)

fig.add_trace(go.Scatter(y=stationary_series, mode='lines', name='Stationary'), row=1, col=1)
fig.add_trace(go.Scatter(y=trend_series, mode='lines', name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(y=expanding_variance_series, mode='lines', name='Expanding Variance'), row=3, col=1)

fig.update_layout(height=800, title_text="Visual Inspection of Stationarity vs Non-Stationarity", showlegend=False)
fig.show()

# Calculate rolling statistics for Gold Prices
window_size = 30
gold['roll_mean'] = gold['Price'].rolling(window=window_size).mean()
gold['roll_std'] = gold['Price'].rolling(window=window_size).std()

# Create figure with secondary y-axis for Standard Deviation
fig2 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig2.add_trace(
    go.Scatter(x=gold.index, y=gold['Price'], name='Original Price', opacity=0.5), 
    secondary_y=False
)
fig2.add_trace(
    go.Scatter(x=gold.index, y=gold['roll_mean'], name='30-Day Mean', line=dict(color='orange')), 
    secondary_y=False
)
fig2.add_trace(
    go.Scatter(x=gold.index, y=gold['roll_std'], name='30-Day Rolling Std', line=dict(color='green', dash='dot')), 
    secondary_y=True
)

# Layout adjustments
fig2.update_layout(title_text="Gold Prices: Visualizing Non-Stationarity (Rolling Stats)")
fig2.update_yaxes(title_text="Price (USD)", secondary_y=False)
fig2.update_yaxes(title_text="Standard Deviation", secondary_y=True)
fig2.show()

# Plot ACF for comparison
fig3, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot ACF for Synthetic Stationary Data
plot_acf(stationary_series, lags=40, ax=axes[0], title='ACF: Synthetic Stationary Data')

# Plot ACF for Gold Price (Non-Stationary)
plot_acf(gold['Price'].dropna(), lags=40, ax=axes[1], title='ACF: Gold Price (Non-Stationary)')

plt.show()

print("\n=== Stationarity Analysis Complete ===")
print(f"Gold Price Mean: {gold['Price'].mean():.2f}")
print(f"Gold Price Std: {gold['Price'].std():.2f}")
print(f"Rolling Mean Range: {gold['roll_mean'].min():.2f} - {gold['roll_mean'].max():.2f}")
print(f"Rolling Std Range: {gold['roll_std'].min():.2f} - {gold['roll_std'].max():.2f}")
