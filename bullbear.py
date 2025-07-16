import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time

# Auto-refresh logic
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
    middle = data.rolling(window=window).mean()
    sd = data.rolling(window=window).std()
    upper = middle + sd * num_sd
    lower = middle - sd * num_sd
    return lower, middle, upper

# Create two tabs
tab1, tab2 = st.tabs([
    "Original Forecast (Daily/Hourly/Both)",
    "Enhanced Forecast w/ RSI & Fib (Daily/Intraday/Both)"
])

with tab1:
    auto_refresh()
    st.header("Original Forecast")
    mode1 = st.sidebar.selectbox(
        "Mode:", ["Stock", "Forex"], key="mode1"
    )
    chart1 = st.sidebar.radio(
        "Chart:", ["Daily", "Hourly", "Both"], key="chart1"
    )
    if mode1 == "Stock":
        ticker1 = st.sidebar.selectbox(
            "Select Stock Ticker:", sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI',
                'JPM','VTWG','PLTR','NVDA','META','SITM',
                'MARA','GOOG','HOOD','BABA','IBM','AVGO',
                'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL',
                'URI','DAL'
            ]), key="ticker1"
        )
        if st.sidebar.button("Run Forecast", key="btn1"):
            # Daily forecast data
            df = yf.download(ticker1, start='2018-01-01',
                             end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
            model = SARIMAX(df, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = model.get_forecast(steps=30)
            idx = pd.date_range(df.index[-1]+timedelta(1),
                                periods=30, freq='D')
            vals = fc.predicted_mean; ci = fc.conf_int()

            if chart1 in ("Daily", "Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label='Hist Close')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='30‑Day Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                ax.set_title(f"{ticker1} Daily Forecast")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

            if chart1 in ("Hourly", "Both"):
                hist = yf.download(ticker1, period='1d',
                                  interval='5m', progress=False)
                if not hist.empty:
                    hc = hist['Close'].ffill()
                    he = hc.ewm(span=20).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label='Intraday Close')
                    ax2.plot(he, '--', label='20 EMA')
                    ax2.set_title(f"{ticker1} Intraday (5m)")
                    ax2.legend(loc='lower left', framealpha=0.5)
                    st.pyplot(fig2)
                else:
                    st.warning("No intraday data available.")

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))

    else:  # Forex
        symbol1 = st.sidebar.selectbox(
            "Select Forex Pair:", [
                'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X',
                'AUDUSD=X','NZDUSD=X','HKDJPY=X','USDCAD=X',
                'USDCNY=X','USDCHF=X','EURGBP=X','USDHKD=X',
                'EURHKD=X','GBPHKD=X'
            ], key="symbol1"
        )
        if st.sidebar.button("Run Forecast", key="btn2"):
            # Daily
            df = yf.download(symbol1, start='2018-01-01',
                             end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
            model = SARIMAX(df, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = model.get_forecast(steps=30)
            idx = pd.date_range(df.index[-1]+timedelta(1),
                                periods=30, freq='D')
            vals = fc.predicted_mean; ci = fc.conf_int()

            if chart1 in ("Daily", "Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label='Hist Close')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='30‑Day Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                ax.set_title(f"{symbol1} Daily Forecast")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

            if chart1 in ("Hourly", "Both"):
                hist = yf.download(symbol1, period='1d',
                                  interval='5m', progress=False)
                if not hist.empty:
                    hc = hist['Close'].ffill()
                    he = hc.ewm(span=20).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label='Intraday Close')
                    ax2.plot(he, '--', label='20 EMA')
                    ax2.set_title(f"{symbol1} Intraday (5m)")
                    ax2.legend(loc='lower left', framealpha=0.5)
                    st.pyplot(fig2)
                else:
                    st.warning("No intraday data available.")

            st.write(pd.DataFrame({
                'Forecast': vals,
                'Lower':    ci.iloc[:,0],
                'Upper':    ci.iloc[:,1]
            }, index=idx))


with tab2:
    auto_refresh()
    st.header("Enhanced Forecast w/ RSI & Fibonacci")
    mode2 = st.sidebar.selectbox(
        "Mode:", ["Stock", "Forex"], key="mode2"
    )
    view2 = st.sidebar.radio(
        "View:", ["Daily", "Intraday", "Both"], key="chart2"
    )

    if mode2 == "Stock":
        ticker2 = st.sidebar.selectbox(
            "Select Stock Ticker:", sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI',
                'JPM','VTWG','PLTR','NVDA','META','SITM',
                'MARA','GOOG','HOOD','BABA','IBM','AVGO',
                'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL',
                'URI','DAL'
            ]), key="ticker2"
        )
        if st.sidebar.button("Run Forecast", key="btn3"):
            daily = yf.download(ticker2, start='2018-01-01',
                                end=pd.to_datetime("today"))['Close'] \
                       .asfreq('D').fillna(method='ffill')
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi = compute_rsi(daily)
            model = SARIMAX(daily, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = model.get_forecast(steps=30)
            idx = pd.date_range(daily.index[-1]+timedelta(1),
                                periods=30, freq='D')
            vals = fc.predicted_mean; ci = fc.conf_int()

            if view2 in ("Daily", "Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label='Hist Close')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                # Fibonacci levels
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236, 0.382, 0.5, 0.618):
                    y = high - diff * lev
                    ax.hlines(y, daily.index[-360], daily.index[-1], linestyles='dotted')
                ax.set_title(f"{ticker2} Daily Forecast + Fib")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(rsi[-360:], label='RSI(14)')
                ax2.axhline(70, linestyle='--')
                ax2.axhline(30, linestyle='--')
                ax2.legend()
                st.pyplot(fig2)

            if view2 in ("Intraday", "Both"):
                intraday = yf.download(ticker2, period='1d',
                                      interval='5m', progress=False)
                if intraday.empty:
                    st.warning("No intraday data available.")
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
                    ax3.set_title(f"{ticker2} Intraday + Fib")
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
        symbol2 = st.sidebar.selectbox(
            "Select Forex Pair:", [
                'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X',
                'AUDUSD=X','NZDUSD=X','HKDJPY=X','USDCAD=X',
                'USDCNY=X','USDCHF=X','EURGBP=X','USDHKD=X',
                'EURHKD=X','GBPHKD=X'
            ], key="symbol2"
        )
        if st.sidebar.button("Run Forecast", key="btn4"):
            daily = yf.download(symbol2, start='2018-01-01',
                                end=pd.to_datetime("today"))['Close'] \
                       .asfreq('D').fillna(method='ffill')
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi = compute_rsi(daily)
            model = SARIMAX(daily, order=(1,1,1),
                            seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = model.get_forecast(steps=30)
            idx = pd.date_range(daily.index[-1]+timedelta(1),
                                periods=30, freq='D')
            vals = fc.predicted_mean; ci = fc.conf_int()

            if view2 in ("Daily", "Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label='Hist Close')
                ax.plot(ema200[-360:], '--', label='200 EMA')
                ax.plot(ma30[-360:], '--', label='30 MA')
                ax.plot(idx, vals, label='Forecast')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], '--', label='Lower BB')
                ax.plot(ub[-360:], '--', label='Upper BB')
                # Fibonacci levels
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236, 0.382, 0.5, 0.618):
                    y = high - diff * lev
                    ax.hlines(y, daily.index[-360], daily.index[-1], linestyles='dotted')
                ax.set_title(f"{symbol2} Daily Forecast + Fib")
                ax.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(rsi[-360:], label='RSI(14)')
                ax2.axhline(70, linestyle='--')
                ax2.axhline(30, linestyle='--')
                ax2.legend()
                st.pyplot(fig2)

            if view2 in ("Intraday", "Both"):
                intraday = yf.download(symbol2, period='1d',
                                      interval='5m', progress=False)
                if intraday.empty:
                    st.warning("No intraday data available.")
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
                    ax3.set_title(f"{symbol2} Intraday + Fib")
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
