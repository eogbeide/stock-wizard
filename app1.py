import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Define the ticker list
ticker_list = sorted(['DOW', 'SPY', 'META', 'TSLA', 'AMZN', 'GOOG', 'UNH', 'SPCE', 'NVDA'])

# Function to generate random noise data
def generate_random_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    num_hours = len(date_range)
    noise_data = np.random.normal(0, 1, num_hours)
    return pd.DataFrame({'Date': date_range, 'Noise': noise_data})

# Function to generate forecast
def generate_forecast(stock_data):
    future_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=30, freq='H')
    forecast_data = pd.DataFrame({'Date': future_dates})
    forecast_data['Noise'] = np.random.normal(0, 1, len(forecast_data))
    return forecast_data.set_index('Date')

# Fetch stock data and generate forecast for each ticker
forecasts = {}
for ticker in ticker_list:
    # Fetch stock data for the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = generate_random_data(start_date, end_date)
    
    # Generate forecast
    forecast = generate_forecast(stock_data)
    forecasts[ticker] = forecast

# Display forecast plots using Streamlit
for ticker, forecast in forecasts.items():
    st.subheader(f"Forecast for {ticker}")
    st.write(forecast)
    st.line_chart(forecast)
