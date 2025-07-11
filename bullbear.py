import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time

# Auto-refresh logic for Forex app: rerun every 5 minutes
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
        model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecast_steps = 30
        forecast = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=prices.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()
        fig, ax1 = plt.subplots(figsize=(14,7))
        ax1.set_title(f'{ticker} Price Forecast, EMA, MA, and Bollinger Bands', fontsize=16)
        ax1.plot(prices[-360:], label='Last 12 Months Historical Data', color='blue')
        ax1.plot(ema_200[-360:], label='200-Day EMA', color='green', linestyle='--')
        ax1.plot(forecast_index, forecast_values, label='1 Month Forecast', color='orange')
        ax1.fill_between(forecast_index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='orange', alpha=0.3)
        ax1.plot(moving_average[-360:], label='30-Day Moving Average', color='brown', linestyle='--')
        ax1.plot(lower_band[-360:], label='Bollinger Lower Band', color='red', linestyle='--')
        ax1.plot(upper_band[-360:], label='Bollinger Upper Band', color='purple', linestyle='--')
        current_vals = {
            'UpperBB': float(upper_band.iloc[-1]),
            '30MA': float(moving_average.iloc[-1]),
            'Close': float(prices.iloc[-1]),
            'LowerBB': float(lower_band.iloc[-1]),
            '200EMA': float(ema_200.iloc[-1])
        }
        for name, val in current_vals.items():
            ax1.axhline(y=val, linestyle='-', label=f'Current {name}: {val:.2f}')
        price_min = float(prices[-360:].min() if len(prices)>360 else prices.min())
        price_max = float(prices[-360:].max() if len(prices)>360 else prices.max())
        ax1.set_ylim(bottom=min(price_min, current_vals['LowerBB']) * 0.95,
                     top=max(price_max, current_vals['UpperBB']) * 1.05)
        ax1.set_xlabel('Date'); ax1.set_ylabel('Price')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='lower left', fontsize='x-small', fancybox=True, framealpha=0.5,
                   title='Legend', title_fontsize='small')
        st.pyplot(fig)
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted Price': forecast_values,
            'Lower Bound': conf_int.iloc[:,0],
            'Upper Bound': conf_int.iloc[:,1]
        }).set_index('Date')
        st.write(forecast_df)

else:
    # Original Forex Forecast App
    st.title("Forex Price Forecasting using SARIMA with EMA, MA & Bollinger Bands")
    st.info(
        "For currency pairs, the direction of the 200-Day EMA indicates trend direction. "
        "A favorable buying opportunity during an uptrend arises when the closing price is near or below the 200 EMA. "
        "It’s stronger if the Lower Bollinger Band is near or touches the 200 EMA. "
        "Additionally, a cross above the 30-day MA signals a potential buy."
    )
    auto_refresh()
    symbol = st.sidebar.selectbox(
        "Select Forex Pair:", ['EURUSD=X', 'EURJPY=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X', 'HKDJOY=X',
                               'USDCAD=X', 'USDCNY=X', 'USDCHF=X', 'EURGBP=X', 'USDHKD=X', 'EURHKD=X', 'GBPHKD=X']
    )
    chart_option = st.sidebar.radio("Choose chart to display:", ('Daily', 'Hourly', 'Both'))
    if st.sidebar.button("Generate Charts"):
        data = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'].asfreq('D').fillna(method='ffill')
        ema_200 = data.ewm(span=200, adjust=False).mean()
        ma_30 = data.rolling(window=30).mean()
        lower_bb, mid_bb, upper_bb = compute_bollinger_bands(data)
        model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc = model.get_forecast(steps=30)
        idx = pd.date_range(data.index[-1]+timedelta(days=1), periods=30, freq='D')
        vals = fc.predicted_mean; ci = fc.conf_int()
        if chart_option in ('Daily', 'Both'):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(data[-360:], label='Last 12 Months', color='blue')
            ax.plot(ema_200[-360:], linestyle='--', label='200-Day EMA', color='green')
            ax.plot(ma_30[-360:], linestyle='--', label='30-Day MA', color='brown')
            ax.plot(idx, vals, label='30-Day Forecast', color='orange')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, color='orange')
            ax.plot(lower_bb[-360:], linestyle='--', label='Lower BB', color='red')
            ax.plot(upper_bb[-360:], linestyle='--', label='Upper BB', color='purple')
            ax.legend(loc='lower left', fontsize='small', framealpha=0.5)
            st.pyplot(fig)
        if chart_option in ('Hourly', 'Both'):
            # only this line changed to 5-minute interval:
            hourly = yf.download(symbol, period='1d', interval='5m')
            if not hourly.empty:
                hourly_close = hourly['Close'].fillna(method='ffill')
                hourly_ema = hourly_close.ewm(span=20, adjust=False).mean()
                fig2, ax2 = plt.subplots(figsize=(14,5))
                ax2.plot(hourly_close, label='Hourly Close', color='blue')
                ax2.plot(hourly_ema, linestyle='--', label='20-Period EMA', color='green')
                ax2.set_title(f'{symbol} Intraday (Last 24h) Close & EMA (Auto-refresh every 5m)')
                ax2.set_xlabel('Datetime'); ax2.set_ylabel('Exchange Rate')
                ax2.legend(loc='lower left', fontsize='small', framealpha=0.5)
                st.pyplot(fig2)
            else:
                st.warning('No intraday data available for the selected symbol.')
        forecast_df = pd.DataFrame({
            'Date': idx,
            'Forecast': vals,
            'Lower': ci.iloc[:,0],
            'Upper': ci.iloc[:,1]
        }).set_index('Date')
        st.write(forecast_df)
