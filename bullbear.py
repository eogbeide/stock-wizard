import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time

# Auto‑refresh logic
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

# **<— ADD THIS CALL HERE —>**
auto_refresh()

# Indicator functions
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    middle = data.rolling(window=window).mean()
    sd = data.rolling(window=window).std()
    upper = middle + sd * num_sd
    lower = middle - sd * num_sd
    return lower, middle, upper

# Global sidebar: pick Stock or Forex
mode = st.sidebar.selectbox("Mode:", ["Stock", "Forex"], key="global_mode")

# Create two tabs
tab1, tab2 = st.tabs(["Original Forecast", "Enhanced Forecast"])

with tab1:
    if mode == "Stock":
        st.title("Stock Price Forecast (SARIMA + EMA, MA & Bollinger)")
        ticker = st.selectbox(
            "Select Stock Ticker:",
            sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI',
                'JPM','VTWG','PLTR','NVDA','META','SITM',
                'MARA','GOOG','HOOD','BABA','IBM','AVGO',
                'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL',
                'URI','DAL'
            ]),
            key="orig_stock_ticker"
        )
        if st.button("Run Forecast", key="orig_stock_btn"):
            df = yf.download(ticker, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(steps=30)
            idx  = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci   = fc.conf_int()

            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label='Price')
            ax.plot(ema200[-360:], '--', label='200 EMA')
            ax.plot(ma30[-360:], '--', label='30 MA')
            ax.plot(idx, vals, label='Forecast')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], '--', label='Lower BB')
            ax.plot(ub[-360:], '--', label='Upper BB')
            ax.set_title(f"{ticker} Forecast & Indicators")
            ax.legend(loc='lower left', framealpha=0.5)
            st.pyplot(fig)

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))

    else:  # Forex
        st.title("Forex Price Forecast (SARIMA + EMA, MA & Bollinger)")
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X',
             'AUDUSD=X','NZDUSD=X','HKDJPY=X','USDCAD=X',
             'USDCNY=X','USDCHF=X','EURGBP=X','USDHKD=X',
             'EURHKD=X','GBPHKD=X'],
            key="orig_forex_pair"
        )
        chart = st.radio(
            "Chart View:",
            ['Daily','Hourly','Both'],
            key="orig_forex_chart"
        )
        if st.button("Run Forecast", key="orig_forex_btn"):
            df = yf.download(pair, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(steps=30)
            idx  = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci   = fc.conf_int()

            if chart in ('Daily','Both'):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label='Price')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                ax.set_title(f"{pair} Daily Forecast")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

            if chart in ('Hourly','Both'):
                intraday = yf.download(pair, period='1d', interval='5m', progress=False)
                if not intraday.empty:
                    hc = intraday['Close'].ffill()
                    he = hc.ewm(span=20).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label='Intraday Close')
                    ax2.plot(he, '--', label='20 EMA')
                    ax2.set_title(f"{pair} Intraday (5m)")
                    ax2.legend(loc='lower left', framealpha=0.5)
                    st.pyplot(fig2)
                else:
                    st.warning("No intraday data.")

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))


with tab2:
    if mode == "Stock":
        st.title("Enhanced Stock Forecast (SARIMA + EMA, MA, Bollinger, RSI & Fibonacci)")
        ticker = st.selectbox(
            "Select Stock Ticker:",
            sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI',
                'JPM','VTWG','PLTR','NVDA','META','SITM',
                'MARA','GOOG','HOOD','BABA','IBM','AVGO',
                'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL',
                'URI','DAL'
            ]),
            key="enh_stock_ticker"
        )
        view = st.radio(
            "View:",
            ['Daily','Intraday','Both'],
            key="enh_stock_view"
        )
        if st.button("Run Forecast", key="enh_stock_btn"):
            daily = yf.download(ticker, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                        .asfreq('D').fillna(method='ffill')
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi = compute_rsi(daily)
            model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(steps=30)
            idx  = pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci   = fc.conf_int()

            if view in ('Daily','Both'):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label='Price')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                # fibonacci
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236,0.382,0.5,0.618):
                    y = high - diff * lev
                    ax.hlines(y, daily.index[-360], daily.index[-1], linestyles='dotted')
                ax.set_title(f"{ticker} Daily + Fib")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(rsi[-360:], label='RSI(14)')
                ax2.axhline(70, linestyle='--')
                ax2.axhline(30, linestyle='--')
                ax2.legend()
                st.pyplot(fig2)

            if view in ('Intraday','Both'):
                intraday = yf.download(ticker, period='1d', interval='5m', progress=False)
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    ic = intraday['Close'].ffill()
                    ie = ic.ewm(span=20).mean()
                    lb2, mb2, ub2 = compute_bollinger_bands(ic)
                    ri = compute_rsi(ic)

                    fig3, ax3 = plt.subplots(figsize=(14,5))
                    ax3.plot(ic, label='Intraday Close')
                    ax3.plot(ie, '--', label='20 EMA')
                    ax3.plot(lb2, '--', label='Lower BB')
                    ax3.plot(ub2, '--', label='Upper BB')
                    ax3.set_title(f"{ticker} Intraday + Fib")
                    ax3.legend(loc='lower left', framealpha=0.5)
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(ri, label='RSI(14)')
                    ax4.axhline(70, linestyle='--')
                    ax4.axhline(30, linestyle='--')
                    ax4.legend()
                    st.pyplot(fig4)

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))

    else:  # Forex
        st.title("Enhanced Forex Forecast (SARIMA + EMA, MA, Bollinger, RSI & Fibonacci)")
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X',
             'AUDUSD=X','NZDUSD=X','HKDJPY=X','USDCAD=X',
             'USDCNY=X','USDCHF=X','EURGBP=X','USDHKD=X',
             'EURHKD=X','GBPHKD=X'],
            key="enh_forex_pair"
        )
        view = st.radio(
            "View:",
            ['Daily','Intraday','Both'],
            key="enh_forex_view"
        )
        if st.button("Run Forecast", key="enh_forex_btn"):
            daily = yf.download(pair, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                        .asfreq('D').fillna(method='ffill')
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi = compute_rsi(daily)
            model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(steps=30)
            idx  = pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci   = fc.conf_int()

            if view in ('Daily','Both'):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label='Price')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                # fibonacci
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236,0.382,0.5,0.618):
                    y = high - diff * lev
                    ax.hlines(y, daily.index[-360], daily.index[-1], linestyles='dotted')
                ax.set_title(f"{pair} Daily + Fib")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(rsi[-360:], label='RSI(14)')
                ax2.axhline(70, linestyle='--')
                ax2.axhline(30, linestyle='--')
                ax2.legend()
                st.pyplot(fig2)

            if view in ('Intraday','Both'):
                intraday = yf.download(pair, period='1d', interval='5m', progress=False)
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    ic = intraday['Close'].ffill()
                    ie = ic.ewm(span=20).mean()
                    lb2, mb2, ub2 = compute_bollinger_bands(ic)
                    ri = compute_rsi(ic)

                    fig3, ax3 = plt.subplots(figsize=(14,5))
                    ax3.plot(ic, label='Intraday Close')
                    ax3.plot(ie, '--', label='20 EMA')
                    ax3.plot(lb2, '--', label='Lower BB')
                    ax3.plot(ub2, '--', label='Upper BB')
                    ax3.set_title(f"{pair} Intraday + Fib")
                    ax3.legend(loc='lower left', framealpha=0.5)
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(ri, label='RSI(14)')
                    ax4.axhline(70, linestyle='--')
                    ax4.axhline(30, linestyle='--')
                    ax4.legend()
                    st.pyplot(fig4)

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))
