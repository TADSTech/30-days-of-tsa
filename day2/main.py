import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import os

# Set Plotly template
pio.templates.default = "ggplot2"

# Define paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Since XAUUSD=X can be spotty with yfinance, GLD (SPDR Gold Shares) will be used
symbol = "GLD"
t = yf.Ticker(symbol)

# Daily time series
data = t.history(period="10y", interval="1d")
print(data.tail())

# Latest close only
last_close = data["Close"].iloc[-1]
print(f"Latest close for {symbol}: {last_close}")

# Download 10 years of data
data = yf.download("GLD", period="10y")

# Select only the 'Close' column
gold_data = data[['Close']].copy()

# Remove any days where the market was closed (NaNs)
gold_data.dropna(inplace=True)

print(gold_data.head())

# Save raw data
gold_data.to_csv(os.path.join(data_dir, "gold_data_10y.csv"))

# Load and clean data
gold_data = pd.read_csv(os.path.join(data_dir, "gold_data_10y.csv"))
print(gold_data.head())

# Rename columns
gold_data.rename(columns={"Price": "Date", "Close": "Price"}, inplace=True)

# Fix data issues - drop first 2 rows
rows_to_drop = gold_data.index[:2]
gold_data.drop(rows_to_drop, inplace=True)

# Fix indexing and date format
gold_data.reset_index(drop=True, inplace=True)
gold_data['Date'] = pd.to_datetime(gold_data['Date'])

# Set the index to date
gold_data.set_index('Date', inplace=True)

print(gold_data.tail())

# Filter for specific plot range: Jan 2024 to Jan 2026
plot_gold_data = gold_data.loc['2024-01-01':'2026-01-09'].copy()
plot_gold_data['Price'] = plot_gold_data['Price'].astype(float)
plot_gold_data['Price'] = plot_gold_data['Price'] * 10.8
plot_gold_data['Price'] = plot_gold_data['Price'].round(0)

# Create the plot
fig = px.line(
    plot_gold_data, 
    y='Price', 
    title='Gold (GLD) Price: Jan 2024 - Jan 2026',
    labels={'Price': 'Price (USD)', 'Date': 'Date'}
)

# Clean up the Y-Axis
fig.update_yaxes(
    tickformat=".0f", 
    dtick=200,          
    range=[1000, 5000]    
)

# Add a grid to make it easier to read
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

fig.show()

# Save clean data
gold_data.to_csv(os.path.join(data_dir, "cleaned_gold_data_10y.csv"))
