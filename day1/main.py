import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# Set Plotly template
pio.templates.default = "ggplot2"

# Define paths to data files relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
train_df_path = os.path.join(script_dir, 'data', 'DailyDelhiClimateTrain.csv')
test_df_path = os.path.join(script_dir, 'data', 'DailyDelhiClimateTest.csv')

try:
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Data files not found. Please ensure the dataset is in the correct path.")
    exit(1)

# Display first few rows
print(train_df.head())

# Set 'date' as datetime index
try:
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df.set_index('date', inplace=True)
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df.set_index('date', inplace=True)
    print("Date column set as index successfully.")
except Exception as e:
    print(f"Error setting date as index: {e}")
    exit(1)

# Split data into singular time series
tstemp = train_df['meantemp']
tshum = train_df['humidity']
tswind = train_df['wind_speed']
tspress = train_df['meanpressure']

# Plot Time Series
tstemp_fig = px.line(tstemp, title='Mean Temperature over Time', labels={'date': 'Date', 'value': 'Mean Temperature (°C)'})
tstemp_fig.show()

tshum_fig = px.line(tshum, title='Humidity over Time', labels={'date': 'Date', 'value': 'Humidity (%)'})
tshum_fig.show()

tswind_fig = px.line(tswind, title='Wind Speed over Time', labels={'date': 'Date', 'value': 'Wind Speed (km/h)'})
tswind_fig.show()

tspress_fig = px.line(tspress, title='Mean Pressure over Time', labels={'date': 'Date', 'value': 'Mean Pressure (hPa)'})
tspress_fig.show()
