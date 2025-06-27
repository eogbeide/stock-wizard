import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time

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

# Indicator functions

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(data, window=20, num_sd=2):
    mid = data.rolling(window=window).mean()
    sd = data.rolling(window=window).std()
    return mid - num_sd*sd, mid, mid + num_sd*sd

# App selection
app_mode = st.sidebar.selectbox("Choose App:", ["Stock Forecast", "Forex Forecast"])

if app_mode == "Stock Forecast":
    st.title("Stock Price Forecasting using SARIMA with EMA, MA & Bollinger Bands")
    st.info(
        "The 200-Day EMA trend signals market direction; close prices near or below it during uptrends can indicate buy opportunities. "
        "Lower Bollinger touches strengthen signals, and crosses above the 30-day MA can signal bullish momentum."
    )
    # Stock inputs
    ticker = st.sidebar.selectbox("Select Stock Ticker:", sorted([
        'AAPL','AMZN','MSFT','GOOG','TSLA','NFLX','SPY','VOO','JPM','NVDA'
    ]))
    if st.sidebar.button("Run Stock Forecast"):
        # Fetch and prepare
        df = yf.download(ticker, start='2018-01-01', end=pd.to_datetime("today"))['Close'].asfreq('D').fillna(method='ffill')
        ema200 = df.ewm(span=200).mean()
        ma30 = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        # SARIMA
        model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12))
        fit = model.fit(disp=False)
        fc = fit.get_forecast(30)
        idx = pd.date_range(df.index[-1]+timedelta(1), periods=30, freq='D')
        vals = fc.predicted_mean; ci = fc.conf_int()
        # Plot
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df[-360:], label='Price')
        ax.plot(ema200[-360:], '--', label='200 EMA')
        ax.plot(ma30[-360:], '--', label='30 MA')
        ax.plot(idx, vals, label='Forecast')
        ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
        ax.plot(lb[-360:], '--', label='Lower BB')
        ax.plot(ub[-360:], '--', label='Upper BB')
        ax.legend(); st.pyplot(fig)
        # Table
        st.write(pd.DataFrame({'Forecast':vals, 'Lower':ci.iloc[:,0], 'Upper':ci.iloc[:,1]}, index=idx))

else:
    st.title("Forex Price Forecasting using SARIMA with EMA, MA & Bollinger Bands")
    st.info(
        "Use the 200-Day EMA and Bollinger Bands for trend signals; hourly EMA charts auto-refresh every 5m."
    )
    # Forex inputs
    symbol = st.sidebar.selectbox("Select Forex Pair:", ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X'])
    chart_option = st.sidebar.radio("Chart View:", ['Daily','Hourly','Both'])
    if st.sidebar.button("Run Forex Forecast"):
        # Daily data
        data = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'].asfreq('D').fillna(method='ffill')
        ema200 = data.ewm(span=200).mean()
        ma30 = data.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(data)
        model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc = model.get_forecast(30)
        idx = pd.date_range(data.index[-1]+timedelta(1), periods=30, freq='D')
        vals = fc.predicted_mean; ci = fc.conf_int()
        # Daily Chart
        if chart_option in ('Daily','Both'):
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(data[-360:], label='Price')
            ax.plot(ema200[-360:], '--', label='200 EMA')
            ax.plot(ma30[-360:], '--', label='30 MA')
            ax.plot(idx, vals, label='Forecast')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], '--', label='Lower BB')
            ax.plot(ub[-360:], '--', label='Upper BB')
            ax.legend(); st.pyplot(fig)
        # Hourly Chart
        if chart_option in ('Hourly','Both'):
            hourly = yf.download(symbol, period='1d', interval='60m')
            if not hourly.empty:
                close_h = hourly['Close'].fillna(method='ffill')
                ema20h = close_h.ewm(span=20).mean()
                fig2, ax2 = plt.subplots(figsize=(12,4))
                ax2.plot(close_h, label='Hourly Close')
                ax2.plot(ema20h, '--', label='20 EMA')
                ax2.legend(); st.pyplot(fig2)
            else:
                st.warning('No intraday data available.')
        # Table
        st.write(pd.DataFrame({'Forecast':vals, 'Lower':ci.iloc[:,0],'Upper':ci.iloc[:,1]}, index=idx))
