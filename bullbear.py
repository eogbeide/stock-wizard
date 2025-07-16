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
tab1, tab2 = st.tabs(["Original Stock/Forex Forecast", "Enhanced Forecast with RSI & Fib"])

with tab1:
    # --- Your Original App Code ---
    app_mode = st.sidebar.selectbox("Choose App:", ["Stock Forecast", "Forex Forecast"] )

    if app_mode == "Stock Forecast":
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
            ma30 = prices.rolling(window=30).mean()
            lb, mb, ub = compute_bollinger_bands(prices)
            model = SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,12))
            fit = model.fit(disp=False)
            fh = 30
            fc = fit.get_forecast(steps=fh)
            idx = pd.date_range(prices.index[-1] + timedelta(days=1), periods=fh, freq='D')
            vals = fc.predicted_mean
            ci = fc.conf_int()

            fig, ax = plt.subplots(figsize=(14,7))
            ax.set_title(f'{ticker} Price Forecast, EMA, MA, and Bollinger Bands', fontsize=16)
            ax.plot(prices[-360:], label='Last 12 Months', color='blue')
            ax.plot(ema_200[-360:], '--', label='200-Day EMA', color='green')
            ax.plot(idx, vals, label='1 Month Forecast', color='orange')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, color='orange')
            ax.plot(ma30[-360:], '--', label='30-Day MA', color='brown')
            ax.plot(lb[-360:], '--', label='Lower BB', color='red')
            ax.plot(ub[-360:], '--', label='Upper BB', color='purple')
            ax.legend(loc='lower left', fontsize='small', framealpha=0.5)
            st.pyplot(fig)

            df_fc = pd.DataFrame({
                'Date': idx,
                'Forecasted Price': vals,
                'Lower Bound': ci.iloc[:,0],
                'Upper Bound': ci.iloc[:,1]
            }).set_index('Date')
            st.write(df_fc)

    else:
        st.title("Forex Price Forecasting using SARIMA with EMA, MA & Bollinger Bands")
        st.info(
            "For currency pairs, the direction of the 200-Day EMA indicates trend direction. "
            "A favorable buying opportunity during an uptrend arises when the closing price is near or below the 200 EMA. "
            "It’s stronger if the Lower Bollinger Band is near or touches the 200 EMA. "
            "Additionally, a cross above the 30-day MA signals a potential buy."
        )
        auto_refresh()
        symbol = st.sidebar.selectbox(
            "Select Forex Pair:", ['EURUSD=X', 'EURJPY=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'NZDUSD=X',
                                   'HKDJPY=X', 'USDCAD=X', 'USDCNY=X', 'USDCHF=X', 'EURGBP=X', 'USDHKD=X',
                                   'EURHKD=X', 'GBPHKD=X']
        )
        chart_option = st.sidebar.radio("Choose chart to display:", ('Daily', 'Hourly', 'Both'))
        if st.sidebar.button("Generate Charts"):
            df = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                     .asfreq('D').fillna(method='ffill')
            ema200 = df.ewm(span=200, adjust=False).mean()
            ma30 = df.rolling(window=30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
            model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc = model.get_forecast(steps=30)
            idx = pd.date_range(df.index[-1] + timedelta(days=1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci = fc.conf_int()

            if chart_option in ('Daily', 'Both'):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label='Last 12 Months', color='blue')
                ax.plot(ema200[-360:], '--', label='200-Day EMA', color='green')
                ax.plot(ma30[-360:], '--', label='30-Day MA', color='brown')
                ax.plot(idx, vals, label='30-Day Forecast', color='orange')
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, color='orange')
                ax.plot(lb[-360:], '--', label='Lower BB', color='red')
                ax.plot(ub[-360:], '--', label='Upper BB', color='purple')
                ax.legend(loc='lower left', fontsize='small', framealpha=0.5)
                st.pyplot(fig)

            if chart_option in ('Hourly', 'Both'):
                hourly = yf.download(symbol, period='1d', interval='5m')
                if not hourly.empty:
                    hc = hourly['Close'].ffill()
                    he = hc.ewm(span=20, adjust=False).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label='Intraday Close', color='blue')
                    ax2.plot(he, '--', label='20-Period EMA', color='green')
                    ax2.set_title(f'{symbol} Intraday (Last 24h) Close & EMA (Auto-refresh every 5m)')
                    ax2.legend(loc='lower left', fontsize='small', framealpha=0.5)
                    st.pyplot(fig2)
                else:
                    st.warning('No intraday data available.')

            df_fc = pd.DataFrame({'Forecast': vals, 'Lower': ci.iloc[:,0], 'Upper': ci.iloc[:,1]}, index=idx)
            st.write(df_fc)

with tab2:
    # --- Enhanced Forex‑Only View ---
    auto_refresh()

    st.title("Forex Forecasting with SARIMA + EMA, MA, Bollinger, RSI & Fibonacci")
    st.info("Select your Forex pair and choose Daily, Intraday or Both charts.")

    symbol = st.sidebar.selectbox("Select Forex Pair:", [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X'
    ])
    view = st.sidebar.radio("View:", ('Daily', 'Intraday', 'Both'))
    if st.sidebar.button("Run Forex Forecast"):
        # Fetch daily data
        daily = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
        ema200 = daily.ewm(span=200).mean()
        ma30   = daily.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(daily)
        rsi = compute_rsi(daily)

        # SARIMA forecast
        model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc    = model.get_forecast(steps=30)
        idx   = pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq='D')
        vals  = fc.predicted_mean
        ci    = fc.conf_int()

        # Daily chart
        if view in ('Daily','Both'):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(daily[-360:], label='Price')
            ax.plot(ema200[-360:], '--', label='200 EMA')
            ax.plot(ma30[-360:], '--', label='30 MA')
            ax.plot(idx, vals, label='Forecast')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], '--', label='Lower BB')
            ax.plot(ub[-360:], '--', label='Upper BB')
            # Fibonacci levels
            high, low = daily[-360:].max(), daily[-360:].min()
            diff = high - low
            for lev in [0.236, 0.382, 0.5, 0.618]:
                y = high - diff * lev
                ax.hlines(y, daily.index[-360], daily.index[-1], linestyles='dotted')
            ax.legend(loc='lower left', framealpha=0.5)
            st.pyplot(fig)

            # RSI subplot
            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label='RSI(14)')
            ax2.axhline(70, linestyle='--')
            ax2.axhline(30, linestyle='--')
            ax2.legend()
            st.pyplot(fig2)

        # Intraday chart
        if view in ('Intraday','Both'):
            intraday = yf.download(symbol, period='1d', interval='5m', progress=False)
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
                ax3.legend(loc='lower left', framealpha=0.5)
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label='RSI(14)')
                ax4.axhline(70, linestyle='--')
                ax4.axhline(30, linestyle='--')
                ax4.legend()
                st.pyplot(fig4)

        # Forecast table
        forecast_df = pd.DataFrame({
            'Forecast': vals,
            'Lower':    ci.iloc[:,0],
            'Upper':    ci.iloc[:,1]
        }, index=idx)
        st.write(forecast_df)
