import streamlit as stg
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time
import datetime
from zoneinfo import ZoneInfo

# Auto-refresh logic: rerun every 5 minutes
REFRESH_INTERVAL = 300  # seconds
def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.experimental_rerun()
        except AttributeError:
            pass

auto_refresh()

# Display last update time in US Pacific Time
if 'last_refresh' in st.session_state:
    last_dt = datetime.datetime.fromtimestamp(
        st.session_state.last_refresh, tz=ZoneInfo('UTC')
    ).astimezone(ZoneInfo('America/Los_Angeles'))
    st.sidebar.write(f"Last updated: {last_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Indicator functions

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(data, window=20, num_sd=2):
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_sd)
    lower_band = middle_band - (std_dev * num_sd)
    return lower_band, middle_band, upper_band

# App selector
app_mode = st.sidebar.selectbox("Choose App:", ["Stock Forecast", "Forex Forecast"] )

if app_mode == "Stock Forecast":
    # Original Stock Forecast App
    st.title("Stock Price Forecasting using SARIMA with EMA, MA & Bollinger")
    st.info(
        "For certain stocks, the direction of the 200-Day EMA indicates whether we are experiencing an upward or downward trend. "
        "A favorable buying opportunity during an upward trend arises when the closing price is near, at, or below the 200 EMA line. "
        "It’s even better if the Lower Bollinger Band is close to or touches the 200 EMA. Additionally, when the price crosses above "
        "the 30-day moving average, it indicates an upward trend and a potential buy signal."
    )
    ticker = st.selectbox("Select Stock Ticker:", options=sorted([
        'AAPL', 'SPY', 'AMZN', 'DIA', 'TSLA', 'SPGI',
        'JPM', 'VTWG', 'PLTR', 'NVDA', 'META', 'SITM',
        'MARA', 'GOOG', 'HOOD', 'BABA', 'IBM','AVGO',
        'GUSH', 'VOO', 'MSFT', 'TSM', 'NFLX', 'MP', 'AAL',
        'URI', 'DAL'
    ]))
    if st.button("Forecast"):
        start_date = '2018-01-01'
        end_date = pd.to_datetime("today")
        data = yf.download(ticker, start=start_date, end=end_date)
        prices = data['Close'].asfreq('D').fillna(method='ffill')
        ema_200 = prices.ewm(span=200, adjust=False).mean()
        moving_average = prices.rolling(window=30).mean()
        lower_band, middle_band, upper_band = compute_bollinger_bands(prices)
        rsi = compute_rsi(prices)
        model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecast_steps = 30
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        fig, ax1 = plt.subplots(figsize=(14,7))
        ax1.set_title(f'{ticker} Price Forecast, EMA, MA, Bollinger Bands & RSI', fontsize=16)
        ax1.plot(prices[-360:], label='Last 12 Months', color='blue')
        ax1.plot(ema_200[-360:], label='200-Day EMA', linestyle='--')
        ax1.plot(moving_average[-360:], label='30-Day MA', linestyle='--')
        ax1.plot(forecast_index, forecast_values, label='30-Day Forecast', color='orange')
        ax1.fill_between(forecast_index, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.3)
        ax1.plot(lower_band[-360:], label='Lower BB', linestyle='--')
        ax1.plot(upper_band[-360:], label='Upper BB', linestyle='--')
        for name, val in {
            'Close': prices.iloc[-1], 'EMA200': ema_200.iloc[-1],
            'MA30': moving_average.iloc[-1], 'LowerBB': lower_band.iloc[-1], 'UpperBB': upper_band.iloc[-1]
        }.items():
            ax1.axhline(y=float(val), linestyle='-', label=f'Current {name}: {float(val):.2f}')
        ax1.set_xlabel('Date'); ax1.set_ylabel('Price')
        ax1.legend(loc='lower left', fontsize='small', framealpha=0.5)
        st.pyplot(fig)
        # RSI plot beneath
        fig2, ax2 = plt.subplots(figsize=(14,3))
        ax2.plot(rsi[-360:], label='RSI (14)', color='magenta')
        ax2.axhline(70, linestyle='--', color='grey')
        ax2.axhline(30, linestyle='--', color='grey')
        ax2.set_title('RSI (14)')
        st.pyplot(fig2)
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted Price': forecast_values,
            'Lower Bound': conf_int.iloc[:,0],
            'Upper Bound': conf_int.iloc[:,1]
        }).set_index('Date')
        st.write(forecast_df)
