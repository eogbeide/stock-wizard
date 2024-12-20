import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt

# Function to compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def compute_bollinger_bands(data, window=20, num_sd=2):
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_sd)
    lower_band = middle_band - (std_dev * num_sd)
    return lower_band, middle_band, upper_band

# Streamlit app title
st.title("Stock Price Forecasting using SARIMA with EMA, MA, and Variance")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA', 'JYD', 'META', 'SITM', 'MARA', 'GOOG', 'HOOD', 'UBER', 'DOW', 'AFRM', 'MSFT', 'TSM', 'NFLX'])

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Prepare the data
    prices = data['Close']  # Use the closing prices
    prices = prices.asfreq('D')  # Set frequency to daily
    prices.fillna(method='ffill', inplace=True)  # Forward fill to handle missing values

    # Calculate 200-day EMA
    ema_200 = prices.ewm(span=200, adjust=False).mean()

    # Calculate daily moving average (e.g., 30-day)
    moving_average = prices.rolling(window=30).mean()

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

    # Calculate rolling variance and standard deviation of Close prices
    rolling_variance = prices.rolling(window=30).var()
    rolling_std_dev = prices.rolling(window=30).std()

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)  # Example values
    seasonal_order = (1, 1, 1, 12)  # Example values for monthly seasonality

    model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Step 4: Forecast the next one month (30 days)
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
    forecast_values = forecast.predicted_mean

    # Get confidence intervals
    conf_int = forecast.conf_int()

    # Step 5: Plot historical data, forecast, EMA, daily moving average, and Bollinger Bands
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot price and 200-day EMA
    ax1.set_title(f'{ticker} Price Forecast, EMA, MA, and Bollinger Bands', fontsize=16)
    ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')  # Last 12 months of historical data
    ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA for the last 12 months
    ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
    ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)

    # Add daily moving average for the last 12 months
    ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')

    # Plot Bollinger Bands
    ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')
    #ax1.plot(middle_band[-360:], label='Bollinger Middle Band', color='black', linestyle='--')
    #ax1.plot(upper_band[-360:], label='Bollinger Upper Band', color='pink', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Step 6: Plot rolling variance and standard deviation in a separate figure
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.set_title(f'{ticker} 30-Day Rolling Variance and Standard Deviation', fontsize=16)
    ax2.plot(rolling_variance[-360:], label='30-Day Rolling Variance', color='purple', linestyle='--')
    ax2.plot(rolling_std_dev[-360:], label='30-Day Rolling Std Dev', color='orange', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc='upper left')

    # Display the variance and std dev plot in Streamlit
    st.pyplot(fig2)

    # Create a DataFrame for forecast data including confidence intervals
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })

    # Show the forecast data in a table
    st.write(forecast_df)
