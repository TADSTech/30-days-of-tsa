import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

print("=== Day 6: Autocorrelation Analysis (ACF & PACF) ===")
print(f"\nGold price data shape: {gold.shape}")
print(gold.head())

# Apply first-order differencing
gold['Price_Diff'] = gold['Price'].diff()

# Remove NaN values for analysis
gold_diff_clean = gold['Price_Diff'].dropna()

print(f"\n=== Differenced Data Statistics ===")
print(f"Original data points: {len(gold)}") 
print(f"Differenced data points (after removing NaN): {len(gold_diff_clean)}")
print(f"\nFirst few differenced values:")
print(gold_diff_clean.head())

# Create side-by-side ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# ACF Plot
plot_acf(gold_diff_clean, lags=40, ax=axes[0], title='ACF: Differenced Gold Prices')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('Autocorrelation')

# PACF Plot
plot_pacf(gold_diff_clean, lags=40, ax=axes[1], title='PACF: Differenced Gold Prices')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.show()

# Autocorrelation Analysis for White Noise Detection
from statsmodels.tsa.stattools import acf, pacf

print("\n" + "="*60)
print("AUTOCORRELATION ANALYSIS FINDINGS")
print("="*60)

# Calculate ACF and PACF
acf_vals = acf(gold_diff_clean, nlags=40)
pacf_vals = pacf(gold_diff_clean, nlags=40)

print(f"\nACF at lag 1: {acf_vals[1]:.4f}")
print(f"PACF at lag 1: {pacf_vals[1]:.4f}")

# Check for significant correlations (threshold = 1.96/sqrt(n))
threshold = 1.96 / np.sqrt(len(gold_diff_clean))
significant_acf = sum(1 for val in acf_vals[1:11] if abs(val) > threshold)
significant_pacf = sum(1 for val in pacf_vals[1:11] if abs(val) > threshold)

print(f"\nSignificance threshold (95% CI): ±{threshold:.4f}")
print(f"Significant ACF lags (first 10): {significant_acf}")
print(f"Significant PACF lags (first 10): {significant_pacf}")

if significant_acf <= 1 and significant_pacf <= 1:
    print("\n✓ CONCLUSION: Differenced prices exhibit WHITE NOISE behavior")
    print("  → No temporal dependencies detected")
    print("  → Gold prices follow a RANDOM WALK (market efficient)")
    print("  → This validates the Efficient Market Hypothesis")
    print("\n✓ RECOMMENDED MODEL: ARIMA(0,1,0)")
    print("  → p = 0: No autoregressive terms needed")
    print("  → d = 1: One differencing applied")
    print("  → q = 0: No moving average terms needed")
    print("\n  This is a simple random walk model:")
    print("  Next Price = Current Price + Random Shock")
else:
    print("\n✗ Some correlation detected")
    print("  Consider ARIMA models with AR/MA terms")

print("\n" + "="*60)
print("KEY INSIGHT: White noise in price differences means past prices")
print("cannot reliably predict future prices. This is expected for")
print("efficient markets like gold commodities.")
print("="*60)
