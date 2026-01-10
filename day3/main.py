import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# Set Plotly template
pio.templates.default = "ggplot2"

# Define paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# Load gold price data
gold = pd.read_csv(
    os.path.join(data_dir, "gold_prices.csv"), 
    index_col=["Date"], 
    parse_dates=True
)

print(gold.head())

# Preprocessing
gold['Price'] = gold['Price'].astype(float)
gold['Price'] = gold['Price'] * 10.8
gold['Price'] = gold['Price'].round(0)

# 1. Line Plot - Gold Price Over Time
fig1 = px.line(
    gold.reset_index(), 
    x='Date',
    y='Price', 
    title='Gold (GLD) Price: Jan 2016 - Jan 2026',
    labels={'Price': 'Price (USD)', 'Date': 'Date'}
)
fig1.update_yaxes(tickformat=".0f", dtick=200, range=[1000, 5000])
fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
fig1.show()

# 2. Rolling Statistics Visualization
# 30-day and 90-day Moving Averages
gold['MA30'] = gold['Price'].rolling(window=30).mean()
gold['MA90'] = gold['Price'].rolling(window=90).mean()
gold['Std30'] = gold['Price'].rolling(window=30).std()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=gold.index, y=gold['Price'], name='Price', opacity=0.5))
fig2.add_trace(go.Scatter(x=gold.index, y=gold['MA30'], name='30-Day MA', line=dict(color='orange')))
fig2.add_trace(go.Scatter(x=gold.index, y=gold['MA90'], name='90-Day MA', line=dict(color='red')))
fig2.update_layout(title='Gold Price with Rolling Averages', xaxis_title='Date', yaxis_title='Price (USD)')
fig2.show()

# 3. Seasonal Decomposition
# Resample to monthly to make the decomposition clearer
gold_monthly = gold['Price'].resample('ME').mean()
result = seasonal_decompose(gold_monthly, model='additive', period=12)

# Create subplots for decomposition using Plotly
fig3 = make_subplots(
    rows=4, cols=1, 
    subplot_titles=("Observed", "Trend", "Seasonal", "Residuals")
)

fig3.add_trace(
    go.Scatter(x=gold_monthly.index, y=result.observed, name='Observed'), 
    row=1, col=1
)
fig3.add_trace(
    go.Scatter(x=gold_monthly.index, y=result.trend, name='Trend'), 
    row=2, col=1
)
fig3.add_trace(
    go.Scatter(x=gold_monthly.index, y=result.seasonal, name='Seasonal'), 
    row=3, col=1
)
fig3.add_trace(
    go.Scatter(x=gold_monthly.index, y=result.resid, name='Residuals'), 
    row=4, col=1
)

fig3.update_layout(
    height=800, 
    title_text="Gold Price Seasonal Decomposition (Monthly)", 
    showlegend=False
)
fig3.show()
