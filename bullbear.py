import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import matplotlib.pyplot as plt
import time

# Auto-refresh logic (for Forex intraday)
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
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

# Create two tabs
tab1, tab2 = st.tabs(["Unified App", "Separate Apps"])

with tab1:
    st.header("Unified Stock & Forex Forecast")
    st.info("Toggle between Stock vs. Forex and Chart vs. Data views in one unified interface.")
    
    # Unified app controls
    app_mode = st.sidebar.selectbox("Select Market:", ["Stock Forecast", "Forex Forecast"])
    view_mode = st.sidebar.radio("View As:", ["Chart", "Data"])
    
    if app_mode == "Stock Forecast":
        st.title("📈 Stock Price Forecasting")
        ticker = st.sidebar.selectbox("Ticker:",
            sorted(['AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR',
                    'NVDA','META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO',
                    'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL','URI','DAL'])
        )
        if st.sidebar.button("Run Stock Forecast"):
            df = yf.download(ticker, start="2018-01-01", end=pd.Timestamp.today())['Close'].asfreq('D').ffill()
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
            rsi = compute_rsi(df)
            
            # SARIMA forecast
            model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(30)
            idx  = pd.date_range(df.index[-1]+timedelta(1), periods=30, freq='D')
            vals = fc.predicted_mean
            ci   = fc.conf_int()
            
            if view_mode == "Chart":
                fig, ax = plt.subplots(figsize=(14,6))
                ax.plot(df[-360:], label="Price")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="30d Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.legend(loc="lower left", framealpha=0.5)
                ax.set_title(f"{ticker} Forecast & Indicators")
                st.pyplot(fig)
                
                fig2, ax2 = plt.subplots(figsize=(14,2))
                ax2.plot(rsi[-360:], label="RSI (14)", color="magenta")
                ax2.axhline(70, "--", color="grey")
                ax2.axhline(30, "--", color="grey")
                ax2.set_title("RSI (14)")
                st.pyplot(fig2)
            else:
                forecast_df = pd.DataFrame({
                    "Forecast": vals,
                    "Lower CI": ci.iloc[:,0],
                    "Upper CI": ci.iloc[:,1]
                }, index=idx)
                st.write(forecast_df)
    
    else:
        st.title("💱 Forex Price Forecasting")
        symbol = st.sidebar.selectbox("Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X']
        )
        chart_option = st.sidebar.radio("Chart Type:", ["Daily","Intraday","Both"])
        if st.sidebar.button("Run Forex Forecast"):
            # Daily
            daily = yf.download(symbol, start="2018-01-01", end=pd.Timestamp.today())['Close'].asfreq('D').ffill()
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi = compute_rsi(daily)
            
            model = SARIMAX(daily, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            fc   = model.get_forecast(30)
            idx  = pd.date_range(daily.index[-1]+timedelta(1), periods=30, freq="D")
            vals = fc.predicted_mean
            ci   = fc.conf_int()
            
            if view_mode == "Chart" and chart_option in ("Daily","Both"):
                fig, ax = plt.subplots(figsize=(14,6))
                ax.plot(daily[-360:], label="Price")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="30d Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                # Fibonacci
                h, l = daily[-360:].max(), daily[-360:].min()
                diff = h - l
                for lev in [0, .236, .382, .5, .618, 1]:
                    y = h - diff*lev
                    ax.hlines(y, daily.index[-360], daily.index[-1],
                              linestyles="dotted", colors="grey",
                              label=f"Fib {int(lev*100)}%")
                ax.legend(loc="lower left", framealpha=0.5)
                ax.set_title(f"{symbol} Daily Forecast & Indicators")
                st.pyplot(fig)
                
                fig2, ax2 = plt.subplots(figsize=(14,2))
                ax2.plot(rsi[-360:], label="RSI (14)", color="magenta")
                ax2.axhline(70, "--", color="grey")
                ax2.axhline(30, "--", color="grey")
                ax2.set_title("RSI (14)")
                st.pyplot(fig2)
            
            if view_mode == "Chart" and chart_option in ("Intraday","Both"):
                intraday = yf.download(symbol, period="1d", interval="5m", progress=False)
                if not intraday.empty:
                    ic = intraday["Close"].ffill()
                    ie = ic.ewm(span=20).mean()
                    lb2, mb2, ub2 = compute_bollinger_bands(ic)
                    ri = compute_rsi(ic)
                    
                    fig3, ax3 = plt.subplots(figsize=(14,4))
                    ax3.plot(ic, label="Close")
                    ax3.plot(ie, "--", label="20 EMA")
                    ax3.plot(lb2, "--", label="Lower BB")
                    ax3.plot(ub2, "--", label="Upper BB")
                    ax3.legend(loc="lower left", framealpha=0.5)
                    ax3.set_title(f"{symbol} Intraday (5m) & Indicators")
                    st.pyplot(fig3)
                    
                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(ri, label="RSI (14)", color="magenta")
                    ax4.axhline(70, "--", color="grey")
                    ax4.axhline(30, "--", color="grey")
                    ax4.set_title("Intraday RSI (14)")
                    st.pyplot(fig4)
                else:
                    st.warning("No intraday data available.")
            
            if view_mode == "Data":
                forecast_df = pd.DataFrame({
                    "Forecast": vals,
                    "Lower CI": ci.iloc[:,0],
                    "Upper CI": ci.iloc[:,1]
                }, index=idx)
                st.write(forecast_df)

with tab2:
    st.header("Separate Stock & Forex Apps")
    
    # 1) Stock Forecast App
    st.subheader("1. Stock Forecast")
    st.info("SARIMA forecast + EMA, MA, Bollinger, RSI for stocks.")
    ticker2 = st.selectbox("Select Stock Ticker:", key="t2", options=sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR',
        'NVDA','META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO',
        'GUSH','VOO','MSFT','TSM','NFLX','MP','AAL','URI','DAL'
    ]))
    if st.button("Forecast Stock", key="btn2"):
        df2 = yf.download(ticker2, start="2018-01-01", end=pd.Timestamp.today())['Close'].asfreq('D').ffill()
        ema2 = df2.ewm(span=200).mean()
        ma2  = df2.rolling(30).mean()
        lb2, mb2, ub2 = compute_bollinger_bands(df2)
        rsi2 = compute_rsi(df2)
        model2 = SARIMAX(df2, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fc2   = model2.get_forecast(30)
        idx2  = pd.date_range(df2.index[-1]+timedelta(1), periods=30, freq='D')
        vals2 = fc2.predicted_mean
        ci2   = fc2.conf_int()
        fig5, ax5 = plt.subplots(figsize=(14,6))
        ax5.plot(df2[-360:], label="Price")
        ax5.plot(ema2[-360:], "--", label="200 EMA")
        ax5.plot(ma2[-360:], "--", label="30 MA")
        ax5.plot(idx2, vals2, label="30d Forecast")
        ax5.fill_between(idx2, ci2.iloc[:,0], ci2.iloc[:,1], alpha=0.3)
        ax5.plot(lb2[-360:], "--", label="Lower BB")
        ax5.plot(ub2[-360:], "--", label="Upper BB")
        ax5.legend(loc="lower left", framealpha=0.5)
        st.pyplot(fig5)
        fig6, ax6 = plt.subplots(figsize=(14,2))
        ax6.plot(rsi2[-360:], label="RSI (14)", color="magenta")
        ax6.axhline(70, "--", color="grey")
        ax6.axhline(30, "--", color="grey")
        ax6.set_title("RSI (14)")
        st.pyplot(fig6)
        st.write(pd.DataFrame({
            "Forecast": vals2,
            "Lower CI": ci2.iloc[:,0],
            "Upper CI": ci2.iloc[:,1]
        }, index=idx2))
    
    st.markdown("---")
    
    # 2) Forex Forecast App
    st.subheader("2. Forex Forecast")
    st.info("SARIMA + EMA, MA, Bollinger, RSI, Fibonacci + 5m intraday for forex.")
    symbol2 = st.selectbox("Select Forex Pair:", key="s2", options=[
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X'
    ])
    chart_opt2 = st.radio("View:", key="r2", options=["Daily","Intraday","Both"])
    if st.button("Forecast Forex", key="b2"):
        daily2 = yf.download(symbol2, start="2018-01-01", end=pd.Timestamp.today())['Close'] \
                   .asfreq('D').ffill()
        emaF = daily2.ewm(span=200).mean()
        maF  = daily2.rolling(30).mean()
        lbF, mbF, ubF = compute_bollinger_bands(daily2)
        rsiF = compute_rsi(daily2)
        modelF = SARIMAX(daily2, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        fcF   = modelF.get_forecast(30)
        idxF  = pd.date_range(daily2.index[-1]+timedelta(1), periods=30, freq="D")
        valsF = fcF.predicted_mean
        ciF   = fcF.conf_int()
        if chart_opt2 in ("Daily","Both"):
            fig7, ax7 = plt.subplots(figsize=(14,6))
            ax7.plot(daily2[-360:], label="Price")
            ax7.plot(emaF[-360:], "--", label="200 EMA")
            ax7.plot(maF[-360:], "--", label="30 MA")
            ax7.plot(idxF, valsF, label="30d Forecast")
            ax7.fill_between(idxF, ciF.iloc[:,0], ciF.iloc[:,1], alpha=0.3)
            ax7.plot(lbF[-360:], "--", label="Lower BB")
            ax7.plot(ubF[-360:], "--", label="Upper BB")
            h2, l2 = daily2[-360:].max(), daily2[-360:].min()
            d2 = h2 - l2
            for lev in [0, .236, .382, .5, .618, 1]:
                y2 = h2 - d2*lev
                ax7.hlines(y2, daily2.index[-360], daily2.index[-1],
                           linestyles="dotted", colors="grey")
            ax7.legend(loc="lower left", framealpha=0.5)
            ax7.set_title(f"{symbol2} Daily Forecast & Indicators")
            st.pyplot(fig7)
            fig8, ax8 = plt.subplots(figsize=(14,2))
            ax8.plot(rsiF[-360:], label="RSI (14)", color="magenta")
            ax8.axhline(70, "--", color="grey")
            ax8.axhline(30, "--", color="grey")
            ax8.set_title("RSI (14)")
            st.pyplot(fig8)
        if chart_opt2 in ("Intraday","Both"):
            intd = yf.download(symbol2, period="1d", interval="5m", progress=False)
            if not intd.empty:
                ic2 = intd["Close"].ffill()
                ie2 = ic2.ewm(span=20).mean()
                lb2b, mb2b, ub2b = compute_bollinger_bands(ic2)
                ri2 = compute_rsi(ic2)
                fig9, ax9 = plt.subplots(figsize=(14,4))
                ax9.plot(ic2, label="Close")
                ax9.plot(ie2, "--", label="20 EMA")
                ax9.plot(lb2b, "--", label="Lower BB")
                ax9.plot(ub2b, "--", label="Upper BB")
                ax9.legend(loc="lower left", framealpha=0.5)
                ax9.set_title(f"{symbol2} Intraday (5m) & Indicators")
                st.pyplot(fig9)
                fig10, ax10 = plt.subplots(figsize=(14,2))
                ax10.plot(ri2, label="RSI (14)", color="magenta")
                ax10.axhline(70, "--", color="grey")
                ax10.axhline(30, "--", color="grey")
                ax10.set_title("Intraday RSI (14)")
                st.pyplot(fig10)
            else:
                st.warning("No intraday data available.")
        # Forecast table
        st.write(pd.DataFrame({"Forecast": valsF, "Lower CI": ciF.iloc[:,0], "Upper CI": ciF.iloc[:,1]}, index=idxF))
