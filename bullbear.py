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
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_sd)
    lower_band = middle_band - (std_dev * num_sd)
    return lower_band, middle_band, upper_band

# App selector
app_mode = st.sidebar.selectbox("Choose App:", ["Stock Forecast", "Forex Forecast"] )

if app_mode == "Stock Forecast":
    # Stock Forecast App
    st.title("Stock Price Forecasting using SARIMA with EMA, MA, Bollinger & RSI")
    st.info(
        "200-Day EMA trend signals direction; prices near or below it during uptrends can indicate buys. "
        "Bollinger Bands highlight volatility and RSI shows overbought/oversold levels."
    )
    ticker = st.sidebar.selectbox(
        "Select Stock Ticker:",
        sorted([
            'AAPL','SPY','AMZN','DIA','TSLA','SPGI',
            'JPM','VTWG','PLTR','NVDA','META','SITM',
            'MARA','GOOG','HOOD','BABA','IBM','AVGO',
            'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL',
            'URI','DAL'
        ])
    )
    if st.sidebar.button("Run Stock Forecast"):
        df = yf.download(ticker, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
               .asfreq('D').fillna(method='ffill')
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        rsi = compute_rsi(df)

        # SARIMA forecast
        model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc    = model.get_forecast(steps=30)
        idx   = pd.date_range(df.index[-1]+timedelta(1), periods=30, freq='D')
        vals  = fc.predicted_mean
        ci    = fc.conf_int()

        # Price + indicators plot
        fig, ax1 = plt.subplots(figsize=(14,7))
        ax1.plot(df[-360:], label='Price')
        ax1.plot(ema200[-360:], '--', label='200-Day EMA')
        ax1.plot(ma30[-360:], '--', label='30-Day MA')
        ax1.plot(idx, vals, label='30-Day Forecast')
        ax1.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
        ax1.plot(lb[-360:], '--', label='Lower BB')
        ax1.plot(ub[-360:], '--', label='Upper BB')
        ax1.legend(loc='lower left', framealpha=0.5)
        ax1.set_title(f'{ticker} Forecast & Indicators')
        st.pyplot(fig)

        # RSI subplot
        fig2, ax2 = plt.subplots(figsize=(14,3))
        ax2.plot(rsi[-360:], label='RSI (14)', color='magenta')
        ax2.axhline(70, linestyle='--', color='grey')
        ax2.axhline(30, linestyle='--', color='grey')
        ax2.set_title('RSI (14)')
        st.pyplot(fig2)

        # Forecast DataFrame
        forecast_df = pd.DataFrame({'Forecast': vals, 'Lower': ci.iloc[:,0], 'Upper': ci.iloc[:,1]}, index=idx)
        st.write(forecast_df)

else:
    # Forex Forecast App
    st.title("Forex Price Forecasting using SARIMA with EMA, MA, Bollinger & RSI")
    st.info(
        "200-Day EMA shows trend; Bollinger Bands show volatility; RSI shows overbought/oversold."
    )
    auto_refresh()
    symbol = st.sidebar.selectbox("Select Forex Pair:", [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X'
    ])
    chart_option = st.sidebar.radio("Chart View:", ('Daily','Intraday'))
    if st.sidebar.button("Run Forex Forecast"):
        # Daily
        daily = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
        ema200 = daily.ewm(span=200).mean()
        ma30   = daily.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(daily)
        rsi = compute_rsi(daily)

        model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc    = model.get_forecast(steps=30)
        idx   = pd.date_range(daily.index[-1]+timedelta(1), periods=30, freq='D')
        vals  = fc.predicted_mean
        ci    = fc.conf_int()

        if chart_option in ('Daily','Both'):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(daily[-360:], label='Price')
            ax.plot(ema200[-360:], '--', label='200-Day EMA')
            ax.plot(ma30[-360:], '--', label='30-Day MA')
            ax.plot(idx, vals, label='30-Day Forecast')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], '--', label='Lower BB')
            ax.plot(ub[-360:], '--', label='Upper BB')
            # Fibonacci
            high, low = daily[-360:].max(), daily[-360:].min()
            diff = high - low
            for lev in [0, .236, .382, .5, .618, 1]:
                y = high - diff*lev
                ax.hlines(y, daily.index[-360], daily.index[-1], colors='grey', linestyles='dotted')
            ax.legend(loc='lower left', framealpha=0.5)
            ax.set_title(f'{symbol} Daily Forecast & Indicators')
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label='RSI (14)', color='magenta')
            ax2.axhline(70, linestyle='--', color='grey')
            ax2.axhline(30, linestyle='--', color='grey')
            ax2.set_title('RSI (14)')
            st.pyplot(fig2)

        # Intraday
        if chart_option in ('Intraday','Both'):
            intraday = yf.download(symbol, period='1d', interval='5m', progress=False)
            if not intraday.empty:
                ic = intraday['Close'].ffill()
                ie = ic.ewm(span=20).mean()
                lb2, mb2, ub2 = compute_bollinger_bands(ic)
                ri = compute_rsi(ic)
                fig3, ax3 = plt.subplots(figsize=(14,5))
                ax3.plot(ic, label='Close')
                ax3.plot(ie, '--', label='20-EMA')
                ax3.plot(lb2, '--', label='Lower BB')
                ax3.plot(ub2, '--', label='Upper BB')
                ax3.legend(loc='lower left', framealpha=0.5)
                ax3.set_title(f'{symbol} Intraday (5m) & Indicators')
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label='RSI (14)', color='magenta')
                ax4.axhline(70, linestyle='--', color='grey')
                ax4.axhline(30, linestyle='--', color='grey')
                ax4.set_title('Intraday RSI (14)')
                st.pyplot(fig4)

        # Forecast table
        df_fc = pd.DataFrame({'Forecast': vals, 'Lower': ci.iloc[:,0], 'Upper': ci.iloc[:,1]}, index=idx)
        st.write(df_fc)
