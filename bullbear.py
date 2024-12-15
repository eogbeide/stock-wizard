import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# Streamlit app title
st.title("Stock Price Forecasting with SARIMA and EMA")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA'])

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

    # Calculate 20-day and 200-day EMA
    ema_20 = data.ewm(span=20, adjust=False).mean()
    ema_200 = data.ewm(span=200, adjust=False).mean()

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

    # Step 5: Plot historical data, forecast, and EMA
    plt.figure(figsize=(14, 7))
    plt.plot(data[-180:], label='Last 6 Months Historical Data', color='blue')  # Last 6 months of historical data
    plt.plot(ema_20[-180:], label='20-Day EMA', color='red', linestyle='--')  # 20-day EMA
    plt.plot(ema_200[-180:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA
    plt.plot(forecast_index, forecast_values, label='3 Months Forecast', color='orange')
    plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

    # Adding the last value annotations
    last_forecast_value = forecast_values.iloc[-1]
    last_ema_20_value = ema_20.iloc[-1]
    last_ema_200_value = ema_200.iloc[-1]
    
    plt.annotate(f'Forecast: {last_forecast_value:.2f}', 
                 xy=(forecast_index[-1], last_forecast_value), 
                 xytext=(forecast_index[-1], last_forecast_value + 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.annotate(f'20-Day EMA: {last_ema_20_value:.2f}', 
                 xy=(data.index[-1], last_ema_20_value), 
                 xytext=(data.index[-1], last_ema_20_value + 5),
                 arrowprops=dict(facecolor='red', arrowstyle='->'))

    plt.annotate(f'200-Day EMA: {last_ema_200_value:.2f}', 
                 xy=(data.index[-1], last_ema_200_value), 
                 xytext=(data.index[-1], last_ema_200_value + 5),
                 arrowprops=dict(facecolor='green', arrowstyle='->'))

    plt.title(f'{ticker} Price Forecast for Next 3 Months')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Display the plot in Streamlit
    st.pyplot(plt)

    # Create a DataFrame for forecast data including confidence intervals
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })

    # Show the forecast data in a table
    st.write(forecast_df)
