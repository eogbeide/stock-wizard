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
st.title("Forex Price Forecasting using SARIMA with EMA, MA & Bollinger Bands")

# Information box at the top
st.info(
    "For currency pairs, the direction of the 200-Day EMA indicates trend direction. "
    "A favorable buying opportunity during an uptrend arises when the closing price is near or below the 200 EMA. "
    "It’s stronger if the Lower Bollinger Band is near or touches the 200 EMA. "
    "Additionally, a cross above the 30-day MA signals a potential buy."
)

# User input for forex pair using a dropdown menu
symbol = st.selectbox(
    "Select Forex Pair:",
    options=[
        'EURUSD=X', 'EURJPY=X', 'GBPUSD=X',
        'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X'
    ]
)

# Button to fetch and process data
if st.button("Forecast"):
    # Step 1: Download historical data from Yahoo Finance
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(symbol, start=start_date, end=end_date)

    # Step 2: Prepare the data
    prices = data['Close']
    prices = prices.asfreq('D')
    prices.fillna(method='ffill', inplace=True)

    # Calculate indicators
    ema_200 = prices.ewm(span=200, adjust=False).mean()
    ma_30 = prices.rolling(window=30).mean()
    lower_band, middle_band, upper_band = compute_bollinger_bands(prices)

    # Step 3: Fit the SARIMA model
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model = SARIMAX(prices, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # Step 4: Forecast next 30 days
    steps = 30
    forecast = model_fit.get_forecast(steps=steps)
    forecast_index = pd.date_range(
        start=prices.index[-1] + timedelta(days=1),
        periods=steps, freq='D'
    )
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Step 5: Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(prices[-360:], label='Last 12 Months', color='blue')
    ax.plot(ema_200[-360:], label='200-Day EMA', linestyle='--', color='green')
    ax.plot(ma_30[-360:], label='30-Day MA', linestyle='--', color='brown')
    ax.plot(forecast_index, forecast_values, label='30-Day Forecast', color='orange')
    ax.fill_between(
        forecast_index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3, label='Confidence Interval', color='orange'
    )
    ax.plot(lower_band[-360:], linestyle='--', label='Lower BB', color='red')
    ax.plot(upper_band[-360:], linestyle='--', label='Upper BB', color='purple')

    # Horizontal lines for current values
    current_vals = {
        'Close': float(prices.iloc[-1]),
        'EMA200': float(ema_200.iloc[-1]),
        'MA30': float(ma_30.iloc[-1]),
        'LowerBB': float(lower_band.iloc[-1]),
        'UpperBB': float(upper_band.iloc[-1])
    }
    for name, val in current_vals.items():
        ax.axhline(y=val, linestyle='-', label=f'Current {name}: {val:.4f}')

    ax.set_title(f'{symbol} Forecast & Indicators', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Exchange Rate')
    ax.legend(loc='lower right', fontsize='small', framealpha=0.5)
    st.pyplot(fig)

    # Display forecast table
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecast': forecast_values,
        'Lower': conf_int.iloc[:, 0],
        'Upper': conf_int.iloc[:, 1]
    }).set_index('Date')
    st.write(forecast_df)
