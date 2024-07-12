import pandas as pd
from fbprophet import Prophet
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

# Define the ticker list
ticker_list = sorted(['DOW', 'SPY', 'META', 'TSLA', 'AMZN', 'GOOG', 'UNH', 'SPCE', 'NVDA'])

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock_data.reset_index()

# Function to train the Prophet model and generate forecast
def generate_forecast(stock_data):
    model = Prophet(daily_seasonality=True)
    model.fit(stock_data)
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    return forecast

# Fetch stock data and generate forecast for each ticker
forecasts = {}
for ticker in ticker_list:
    # Fetch stock data for the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Generate forecast
    forecast = generate_forecast(stock_data)
    forecasts[ticker] = forecast

# Display forecast plots using Streamlit
for ticker, forecast in forecasts.items():
    st.subheader(f"Forecast for {ticker}")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
