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

# Analysis helper function
def analyze_acf_pacf(acf_series, pacf_series, threshold=0.05):
    """
    Analyze ACF and PACF to identify significant lags
    """
    print("\n=== ACF & PACF Analysis ===")
    
    # Find significant ACF lags
    acf_cutoff = np.where(np.abs(acf_series) < threshold)[0][0] if any(np.abs(acf_series) < threshold) else len(acf_series)
    print(f"\nACF: Significant lags appear to be up to lag {acf_cutoff - 1}")
    print(f"  Suggests MA order q ≈ {max(0, acf_cutoff - 1)}")
    
    # Find significant PACF lags
    pacf_cutoff = np.where(np.abs(pacf_series) < threshold)[0][0] if any(np.abs(pacf_series) < threshold) else len(pacf_series)
    print(f"\nPACF: Significant lags appear to be up to lag {pacf_cutoff - 1}")
    print(f"  Suggests AR order p ≈ {max(0, pacf_cutoff - 1)}")
    
    print("\n=== Recommended ARIMA Models to Test ===")
    print("1. ARIMA(1,1,1) - Common for financial data")
    print("2. ARIMA(2,1,2) - If patterns persist beyond lag 1")
    print("3. ARIMA(1,1,0) - If PACF shows clear AR pattern")
    print("4. ARIMA(0,1,1) - If ACF shows clear MA pattern")
    print("5. ARIMA(2,1,0) or ARIMA(0,1,2) - For stronger single-component models")
    print("\nNote: d = 1 (first-order differencing confirmed in Day 5)")

# Calculate approximate ACF and PACF for analysis
from statsmodels.tsa.stattools import acf, pacf
acf_vals = acf(gold_diff_clean, nlags=40)
pacf_vals = pacf(gold_diff_clean, nlags=40)

# Analyze the results
analyze_acf_pacf(acf_vals, pacf_vals)

print("\n=== Analysis Complete ===")
print("Next step: Fit multiple ARIMA models and compare using AIC/BIC criteria")
