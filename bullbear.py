import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# Streamlit app title
st.title("Stock Price Forecasting with SARIMA")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", value=['AAPL', 'SPY', 'AMZN', 'TSLA'])

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    data = data['Close']  # Use the closing prices
    data = data.asfreq('D')  # Set frequency to daily
    data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)  # Example values
    seasonal_order = (1, 1, 1, 12)  # Example values for monthly seasonality

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Step 4: Forecast the next three months (90 days)
    forecast_steps = 90
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Get confidence intervals
    conf_int = forecast.conf_int()

    # Step 5: Plot historical data and forecast
    plt.figure(figsize=(14, 7))
    plt.plot(data[-180:], label='Last 6 Months Historical Data', color='blue')  # Last 6 months of historical data
    plt.plot(forecast_index, forecast_values, label='3 Months Forecast', color='orange')
    plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
    plt.title(f'{ticker} Price Forecast for Next 3 Months')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

    # Optionally, show the forecast data in a table
    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted Price': forecast_values})
    st.write(forecast_df)
