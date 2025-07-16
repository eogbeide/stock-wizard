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
    # (Stock Forecast code unchanged)
    # ...
    pass

else:
    # Forex Forecast App with RSI
    st.title("Forex Price Forecasting using SARIMA with EMA, MA, Bollinger & RSI")
    st.info(
        "For currency pairs, 200-Day EMA shows trend; Bollinger Bands and RSI give overbought/oversold signals. "
        "Intraday chart refreshes every 5 minutes."
    )
    auto_refresh()
    symbol = st.sidebar.selectbox(
        "Select Forex Pair:", [
            'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
            'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
            'USDHKD=X','EURHKD=X','GBPHKD=X'
        ]
    )
    chart_option = st.sidebar.radio("Chart View:", ('Daily','Hourly','Both'))
    if st.sidebar.button("Run Forex Forecast"):
        # Daily series
        daily = yf.download(symbol, start='2018-01-01', end=pd.to_datetime("today"))['Close'] \
                   .asfreq('D').fillna(method='ffill')
        ema200 = daily.ewm(span=200).mean()
        ma30   = daily.rolling(window=30).mean()
        lb, mb, ub = compute_bollinger_bands(daily)
        rsi = compute_rsi(daily)
        # SARIMA forecast
        model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc    = model.get_forecast(steps=30)
        idx   = pd.date_range(daily.index[-1]+timedelta(1), periods=30, freq='D')
        vals  = fc.predicted_mean
        ci    = fc.conf_int()

        if chart_option in ('Daily','Both'):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(daily[-360:], label='Price', color='blue')
            ax.plot(ema200[-360:], '--', label='200-Day EMA', color='green')
            ax.plot(ma30[-360:],   '--', label='30-Day MA',  color='brown')
            ax.plot(idx, vals,     label='30-Day Forecast', color='orange')
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, color='orange')
            ax.plot(lb[-360:], '--', label='Lower BB', color='red')
            ax.plot(ub[-360:], '--', label='Upper BB', color='purple')
            # Fibonacci
            high, low = daily[-360:].max(), daily[-360:].min()
            diff = high - low
            for lev in [0, .236, .382, .5, .618, 1]:
                y = high - diff * lev
                ax.hlines(y, daily.index[-360], daily.index[-1],
                          colors='grey', linestyles='dotted', 
                          label=f'Fib {int(lev*100)}%')
            ax.set_title(f'{symbol} Daily Forecast & Indicators')
            ax.legend(loc='lower left', fontsize='small', framealpha=0.5)
            st.pyplot(fig)
            # RSI plot
            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label='RSI (14)', color='magenta')
            ax2.axhline(70, linestyle='--', color='grey')
            ax2.axhline(30, linestyle='--', color='grey')
            ax2.set_title('RSI (14)')
            st.pyplot(fig2)

        if chart_option in ('Hourly','Both'):
            hourly = yf.download(symbol, period='1d', interval='5m', progress=False)
            if not hourly.empty:
                hc = hourly['Close'].fillna(method='ffill')
                he = hc.ewm(span=20).mean()
                hlb, hmb, hub = compute_bollinger_bands(hc)
                hrs = compute_rsi(hc)
                fig3, ax3 = plt.subplots(figsize=(14,5))
                ax3.plot(hc, label='Close', color='blue')
                ax3.plot(he, '--', label='20-Period EMA', color='green')
                ax3.plot(hmb, '--', label='Middle BB', color='grey')
                ax3.plot(hlb, '--', label='Lower BB', color='red')
                ax3.plot(hub, '--', label='Upper BB', color='purple')
                ax3.set_title(f'{symbol} Intraday (5m) Close, EMA & BB')
                ax3.legend(loc='lower left', fontsize='small', framealpha=0.5)
                st.pyplot(fig3)
                # Intraday RSI
                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(hrs, label='RSI (14)', color='magenta')
                ax4.axhline(70, linestyle='--', color='grey')
                ax4.axhline(30, linestyle='--', color='grey')
                ax4.set_title('Intraday RSI (14)')
                st.pyplot(fig4)
            else:
                st.warning("No intraday data available.")
        # Forecast table
        st.write(pd.DataFrame({'Forecast':vals,'Lower':ci.iloc[:,0],'Upper':ci.iloc[:,1]}, index=idx))
