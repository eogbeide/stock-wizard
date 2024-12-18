import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

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
st.title("Stock Price Forecasting with SARIMA, EMA, MACD, RSI, and Bollinger Bands")

# User input for stock ticker using a dropdown menu
ticker = st.selectbox("Select Stock Ticker:", options=['AAPL', 'SPY', 'AMZN', 'TSLA', 'PLTR', 'NVDA', 'JYD', 'META', 'SITM', 'MARA', 'GOOG', 'HOOD', 'UBER', 'DOW', 'AFRM', 'MSFT', 'TSM', 'NFLX'])

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

    # Calculate 200-day EMA
    ema_200 = data.ewm(span=200, adjust=False).mean()

    # Calculate MACD
    short_ema = data.ewm(span=12, adjust=False).mean()  # Short-term EMA
    long_ema = data.ewm(span=26, adjust=False).mean()  # Long-term EMA
    macd_line = short_ema - long_ema  # MACD Line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()  # Signal Line

    # Calculate RSI
    rsi = compute_rsi(data)

    # Calculate Bollinger Bands
    lower_band, middle_band, upper_band = compute_bollinger_bands(data)

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

    # Step 5: Plot historical data, forecast, EMA, MACD, and Bollinger Bands
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot price and 200-day EMA
    ax1.set_title(f'{ticker} Price Forecast, MACD, RSI, and Bollinger Bands', fontsize=16)
    ax1.plot(data[-180:], label='Last 6 Months Historical Data', color='blue')  # Last 6 months of historical data
    ax1.plot(ema_200[-180:], label='200-Day EMA', color='green', linestyle='--')  # 200-day EMA
    ax1.plot(forecast_index, forecast_values, label='3 Months Forecast', color='orange')
    ax1.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
    
    # Plot Bollinger Bands
    ax1.plot(lower_band[-180:], label='Bollinger Lower Band', color='purple', linestyle='--')
    ax1.plot(middle_band[-180:], label='Bollinger Middle Band', color='orange', linestyle='--')
    ax1.plot(upper_band[-180:], label='Bollinger Upper Band', color='red', linestyle='--')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for MACD
    ax2 = ax1.twinx()  
    ax2.plot(macd_line[-180:], label='MACD Line', color='purple')
    ax2.plot(signal_line[-180:], label='Signal Line', color='red', linestyle='--')
    ax2.set_ylabel('MACD', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Adding a horizontal line at 0

    # Add MACD legend
    ax2.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Create a DataFrame for forecast data including confidence intervals
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted Price': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })

    # Show the forecast data in a table
    st.write(forecast_df)
