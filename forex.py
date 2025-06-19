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
    # Step 1: Download historical daily data
    start_date = '2018-01-01'
    end_date = pd.to_datetime("today")
    data = yf.download(symbol, start=start_date, end=end_date)

    # Prepare daily series
    prices = data['Close'].asfreq('D').fillna(method='ffill')
    ema_200 = prices.ewm(span=200, adjust=False).mean()
    ma_30 = prices.rolling(window=30).mean()
    lower_bb, mid_bb, upper_bb = compute_bollinger_bands(prices)

    # Fit SARIMA
    model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)

    # Forecast next 30 days
    fc = fit.get_forecast(steps=30)
    idx = pd.date_range(prices.index[-1] + timedelta(days=1), periods=30, freq='D')
    fc_vals = fc.predicted_mean
    ci = fc.conf_int()

    # Daily plot
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(prices[-360:], label='Last 12 Months', color='blue')
    ax.plot(ema_200[-360:], label='200-Day EMA', linestyle='--', color='green')
    ax.plot(ma_30[-360:], label='30-Day MA', linestyle='--', color='brown')
    ax.plot(idx, fc_vals, label='30-Day Forecast', color='orange')
    ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, color='orange')
    ax.plot(lower_bb[-360:], linestyle='--', label='Lower BB', color='red')
    ax.plot(upper_bb[-360:], linestyle='--', label='Upper BB', color='purple')
    for name, val in {
        'Close': prices.iloc[-1], 'EMA200': ema_200.iloc[-1],
        'MA30': ma_30.iloc[-1], 'LowerBB': lower_bb.iloc[-1], 'UpperBB': upper_bb.iloc[-1]
    }.items():
        ax.axhline(y=val, linestyle='-', label=f'Current {name}: {val:.4f}')
    ax.set_title(f'{symbol} Daily Forecast & Indicators')
    ax.set_xlabel('Date'); ax.set_ylabel('Exchange Rate')
    ax.legend(loc='lower right', fontsize='small', framealpha=0.5)
    st.pyplot(fig)

    # Fetch intraday hourly data (today)
    hourly = yf.download(symbol, period='1d', interval='60m')
    hourly_close = hourly['Close'].fillna(method='ffill')
    hourly_ema = hourly_close.ewm(span=20, adjust=False).mean()

    # Hourly plot
    fig2, ax2 = plt.subplots(figsize=(14,5))
    ax2.plot(hourly_close, label='Hourly Close', color='blue')
    ax2.plot(hourly_ema, label='20-Period EMA', linestyle='--', color='green')
    ax2.set_title(f'{symbol} Intraday Hourly Close & EMA')
    ax2.set_xlabel('Datetime'); ax2.set_ylabel('Exchange Rate')
    ax2.legend(loc='lower right', fontsize='small', framealpha=0.5)
    st.pyplot(fig2)

    # Display forecast table
    forecast_df = pd.DataFrame({'Date': idx, 'Forecast': fc_vals, 'Lower': ci.iloc[:,0], 'Upper': ci.iloc[:,1]}).set_index('Date')
    st.write(forecast_df)
