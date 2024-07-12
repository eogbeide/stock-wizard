import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from xgboost import XGBRegressor

# Define the ticker list
ticker_list = sorted(['DOW', 'SPY', 'META', 'TSLA', 'AMZN', 'GOOG', 'UNH', 'SPCE', 'NVDA'])

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock_data.reset_index()

# Function to preprocess stock data for XGBoost
def preprocess_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Day'] = stock_data['Date'].dt.day
    stock_data['Hour'] = stock_data['Date'].dt.hour
    stock_data['Minute'] = stock_data['Date'].dt.minute
    stock_data['Weekday'] = stock_data['Date'].dt.weekday
    return stock_data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Weekday', 'Close']]

# Function to train the XGBoost model and generate forecast
def generate_forecast(stock_data):
    X = stock_data.iloc[:, :-1]
    y = stock_data.iloc[:, -1]
    model = XGBRegressor()
    model.fit(X, y)
    
    # Generate 30-day forecast
    future_dates = pd.date_range(start=stock_data['Date'].iloc[-1], periods=30, freq='H')
    future_data = pd.DataFrame({'Year': future_dates.year,
                                'Month': future_dates.month,
                                'Day': future_dates.day,
                                'Hour': future_dates.hour,
                                'Minute': future_dates.minute,
                                'Weekday': future_dates.weekday})
    forecast = model.predict(future_data)
    
    # Create forecast dataframe
    forecast_data = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    return forecast_data.set_index('Date')

# Fetch stock data and generate forecast for each ticker
forecasts = {}
for ticker in ticker_list:
    # Fetch stock data for the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess data
    preprocessed_data = preprocess_data(stock_data)
    
    # Generate forecast
    forecast = generate_forecast(preprocessed_data)
    forecasts[ticker] = forecast

# Display forecast plots using Streamlit
for ticker, forecast in forecasts.items():
    st.subheader(f"Forecast for {ticker}")
    st.write(forecast)
    st.line_chart(forecast)
