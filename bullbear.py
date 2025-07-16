import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time

# Auto‐refresh logic
REFRESH_INTERVAL = 120  # seconds
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
st.sidebar.markdown(
    f"**Last refresh:** {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')}"
)

# Indicator functions
def compute_rsi(data, window=14):
    delta = data.diff()
    gain  = delta.where(delta>0, 0).rolling(window).mean()
    loss  = -delta.where(delta<0, 0).rolling(window).mean()
    rs    = gain/loss
    return 100 - (100/(1+rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    mid = data.rolling(window).mean()
    sd  = data.rolling(window).std()
    return mid - num_sd*sd, mid, mid + num_sd*sd

# Safe SARIMAX helper
def safe_sarimax(endog, order, seasonal_order):
    try:
        return SARIMAX(endog, order=order, seasonal_order=seasonal_order)\
               .fit(disp=False)
    except np.linalg.LinAlgError:
        return SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)\
               .fit(disp=False)

# Sidebar mode selector
mode = st.sidebar.selectbox("Mode:", ["Stock", "Forex"], key="global_mode")

# Define four tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Original US Forecast",
    "Enhanced US Forecast",
    "🇳🇬 Nigeria Forecast",
    "🇬🇧 LSE Tech & Index"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    if mode == "Stock":
        ticker = st.selectbox(
            "Select Stock Ticker:",
            sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                'MSFT','TSM','NFLX','MP','AAL','URI','DAL'
            ]),
            key="orig_stock_ticker"
        )
        chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_stock_chart")
        if st.button("Run Stock Forecast", key="orig_stock_btn"):
            df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                   .asfreq("D").fillna(method="ffill")
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            model = safe_sarimax(df, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx   = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
            vals, ci = fc.predicted_mean, fc.conf_int()

            if chart in ("Daily","Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.set_title(f"{ticker} Daily Forecast")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

            if chart in ("Hourly","Both"):
                intraday = yf.download(ticker, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    hc = intraday["Close"].ffill()
                    he = hc.ewm(span=20).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label="Intraday")
                    ax2.plot(he, "--", label="20 EMA")
                    ax2.set_title(f"{ticker} Intraday (5m)")
                    ax2.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))

    else:  # Forex
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X'],
            key="orig_forex_pair"
        )
        chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_forex_chart")
        if st.button("Run Forex Forecast", key="orig_forex_btn"):
            df = yf.download(pair, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                   .asfreq("D").fillna(method="ffill")
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            model = safe_sarimax(df, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx   = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
            vals, ci = fc.predicted_mean, fc.conf_int()

            if chart in ("Daily","Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.set_title(f"{pair} Daily Forecast")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

            if chart in ("Hourly","Both"):
                intraday = yf.download(pair, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    hc = intraday["Close"].ffill()
                    he = hc.ewm(span=20).mean()
                    fig2, ax2 = plt.subplots(figsize=(14,5))
                    ax2.plot(hc, label="Intraday")
                    ax2.plot(he, "--", label="20 EMA")
                    ax2.set_title(f"{pair} Intraday (5m)")
                    ax2.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if mode == "Stock":
        ticker = st.selectbox(
            "Select Stock Ticker:",
            sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                'MSFT','TSM','NFLX','MP','AAL','URI','DAL'
            ]),
            key="enh_stock_ticker"
        )
        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_stock_view")
        if st.button("Run Enhanced Stock Forecast", key="enh_stock_btn"):
            daily = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                       .asfreq("D").fillna(method="ffill")
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            rsi    = compute_rsi(daily)

            model = safe_sarimax(daily, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx   = pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq="D")
            vals, ci = fc.predicted_mean, fc.conf_int()

            if view in ("Daily","Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236,0.382,0.5,0.618):
                    ax.hlines(high - diff*lev, daily.index[-360], daily.index[-1], linestyles="dotted")
                ax.set_title(f"{ticker} Daily + Fib")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(rsi[-360:], label="RSI(14)")
                ax2.axhline(70, linestyle="--")
                ax2.axhline(30, linestyle="--")
                ax2.legend()
                st.pyplot(fig2)

            if view in ("Intraday","Both"):
                intraday = yf.download(ticker, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    ic = intraday["Close"].ffill()
                    ie = ic.ewm(span=20).mean()
                    lb2, mb2, ub2 = compute_bollinger_bands(ic)
                    ri = compute_rsi(ic)

                    fig3, ax3 = plt.subplots(figsize=(14,5))
                    ax3.plot(ic, label="Intraday")
                    ax3.plot(ie, "--", label="20 EMA")
                    ax3.plot(lb2, "--", label="Lower BB")
                    ax3.plot(ub2, "--", label="Upper BB")
                    ax3.set_title(f"{ticker} Intraday + Fib")
                    ax3.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(ri, label="RSI(14)")
                    ax4.axhline(70, linestyle="--")
                    ax4.axhline(30, linestyle="--")
                    ax4.legend()
                    st.pyplot(fig4)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))

    else:  # Enhanced Forex
        pair = st.selectbox(
            "Select Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X'],
            key="enh_forex_pair"
        )
        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_forex_view")
        if st.button("Run Enhanced Forex Forecast", key="enh_forex_btn"):
            daily = yf.download(pair, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                       .asfreq("D").fillna(method="ffill")
            ema200 = daily.ewm(span=200).mean()
            ma30   = daily.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(daily)
            ri     = compute_rsi(daily)

            model = safe_sarimax(daily, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx   = pd.date_range(daily.index[-1] + timedelta(1), periods=30, freq="D")
            vals, ci = fc.predicted_mean, fc.conf_int()

            if view in ("Daily","Both"):
                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(daily[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                high, low = daily[-360:].max(), daily[-360:].min()
                diff = high - low
                for lev in (0.236,0.382,0.5,0.618):
                    ax.hlines(high - diff*lev, daily.index[-360], daily.index[-1], linestyles="dotted")
                ax.set_title(f"{pair} Daily + Fib")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14,3))
                ax2.plot(ri[-360:], label="RSI(14)")
                ax2.axhline(70, linestyle="--")
                ax2.axhline(30, linestyle="--")
                ax2.legend()
                st.pyplot(fig2)

            if view in ("Intraday","Both"):
                intraday = yf.download(pair, period="1d", interval="5m")
                if intraday.empty:
                    st.warning("No intraday data.")
                else:
                    ic = intraday["Close"].ffill()
                    ie = ic.ewm(span=20).mean()
                    lb2, mb2, ub2 = compute_bollinger_bands(ic)
                    ri = compute_rsi(ic)

                    fig3, ax3 = plt.subplots(figsize=(14,5))
                    ax3.plot(ic, label="Intraday")
                    ax3.plot(ie, "--", label="20 EMA")
                    ax3.plot(lb2, "--", label="Lower BB")
                    ax3.plot(ub2, "--", label="Upper BB")
                    ax3.set_title(f"{pair} Intraday + Fib")
                    ax3.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig3)

                    fig4, ax4 = plt.subplots(figsize=(14,2))
                    ax4.plot(ri, label="RSI(14)")
                    ax4.axhline(70, linestyle="--")
                    ax4.axhline(30, linestyle="--")
                    ax4.legend()
                    st.pyplot(fig4)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))

# --- Tab 3: Nigeria Forecast ---
with tab3:
    st.header("🇳🇬 Nigeria Stock & FX Forecast")
    nigeria_mode = st.radio(
        "Choose Nigeria App:",
        ["Stock Forecast", "Forex Forecast"],
        key="nigeria_mode"
    )

    nigeria_stocks = {
        "Dangote Cement (DANGCEM)":      "DANGCEM.LG",
        "GT Bank (GTCO)":                "GTCO.IL",
        "Zenith Bank (ZENITHBANK)":      "ZENB.L",
        "Access Holdings (ACCESSCORP)":  "ACCESSCORP.LG",
        "UBA (UBA)":                     "UBA.LG",
        "MTN Nigeria (MTNN)":            "MTN.JO",
        "Seplat Energy (SEPLAT)":        "SEPL.L",
        "BUA Cement (BUACEMENT)":        "BUACEMENT.LG",
        "Nestle Nigeria (NESTLE)":       "NESTLE.LG",
        "Nigerian Breweries (NB)":       "NB.LG"
    }
    nigeria_fx = {
        "USD/NGN": "USDNGN=X",
        "EUR/NGN": "EURNGN=X",
        "GBP/NGN": "GBPNGN=X",
        "CNY/NGN": "CNYNGN=X",
        "ZAR/NGN": "ZARNGN=X"
    }

    if nigeria_mode == "Stock Forecast":
        sel    = st.selectbox("Select NGX Stock:", list(nigeria_stocks.keys()), key="nig_stock")
        ticker = nigeria_stocks[sel]
        if st.button("Run Nigeria Stock Forecast", key="nig_stock_btn"):
            df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                   .asfreq("D").fillna(method="ffill")
            if df.empty:
                st.warning("No data available for this stock.")
            else:
                ema200 = df.ewm(span=200).mean()
                ma30   = df.rolling(30).mean()
                lb, mb, ub = compute_bollinger_bands(df)

                model = safe_sarimax(df, (1,1,1), (1,1,1,12))
                fc    = model.get_forecast(steps=30)
                idx   = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
                vals, ci = fc.predicted_mean, fc.conf_int()

                fig, ax = plt.subplots(figsize=(14,7))
                ax.plot(df[-360:], label="History")
                ax.plot(ema200[-360:], "--", label="200 EMA")
                ax.plot(ma30[-360:], "--", label="30 MA")
                ax.plot(idx, vals, label="Forecast")
                ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                ax.plot(lb[-360:], "--", label="Lower BB")
                ax.plot(ub[-360:], "--", label="Upper BB")
                ax.set_title(f"{sel} Nigeria Forecast")
                ax.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig)

                st.write(pd.DataFrame({
                    "Forecast": vals,
                    "Lower":    ci.iloc[:,0],
                    "Upper":    ci.iloc[:,1]
                }, index=idx))
    else:
        sel    = st.selectbox("Select NGN FX Pair:", list(nigeria_fx.keys()), key="nig_fx")
        ticker = nigeria_fx[sel]
        chart  = st.radio("Chart View:", ["Daily","Hourly","Both"], key="nig_fx_chart")
        if st.button("Run Nigeria FX Forecast", key="nig_fx_btn"):
            df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                   .asfreq("D").fillna(method="ffill")
            if df.empty:
                st.warning("No FX data available for this pair.")
            else:
                ema200 = df.ewm(span=200).mean()
                ma30   = df.rolling(30).mean()
                lb, mb, ub = compute_bollinger_bands(df)

                model = safe_sarimax(df, (1,1,1), (1,1,1,12))
                fc    = model.get_forecast(steps=30)
                idx   = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
                vals, ci = fc.predicted_mean, fc.conf_int()

                if chart in ("Daily","Both"):
                    fig, ax = plt.subplots(figsize=(14,7))
                    ax.plot(df[-360:], label="History")
                    ax.plot(ema200[-360:], "--", label="200 EMA")
                    ax.plot(ma30[-360:], "--", label="30 MA")
                    ax.plot(idx, vals, label="Forecast")
                    ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                    ax.plot(lb[-360:], "--", label="Lower BB")
                    ax.plot(ub[-360:], "--", label="Upper BB")
                    ax.set_title(f"{sel} Nigeria Daily FX Forecast")
                    ax.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig)

                if chart in ("Hourly","Both"):
                    intraday = yf.download(ticker, period="1d", interval="5m")
                    if intraday.empty:
                        st.warning("No intraday data available.")
                    else:
                        hc = intraday["Close"].ffill()
                        he = hc.ewm(span=20).mean()
                        fig2, ax2 = plt.subplots(figsize=(14,5))
                        ax2.plot(hc, label="Intraday")
                        ax2.plot(he, "--", label="20 EMA")
                        ax2.set_title(f"{sel} Nigeria Intraday (5m)")
                        ax2.legend(loc="lower left", framealpha=0.5)
                        st.pyplot(fig2)

                st.write(pd.DataFrame({
                    "Forecast": vals,
                    "Lower":    ci.iloc[:,0],
                    "Upper":    ci.iloc[:,1]
                }, index=idx))

# --- Tab 4: LSE Tech & Index ---
with tab4:
    st.header("🇬🇧 LSE Tech Stocks & Index ETFs")
    lse_mode = st.radio(
        "Choose Category:",
        ["Tech Stocks", "Index ETFs"],
        key="lse_mode"
    )

    lse_tech = {
        "ARM Holdings (ARM.L)":      "ARM.L",
        "Sage Group (SGE.L)":        "SGE.L",
        "AVEVA Group (AVV.L)":       "AVV.L",
        "Softcat plc (SCT.L)":       "SCT.L",
        "Darktrace plc (DARK.L)":    "DARK.L"
    }
    lse_etfs = {
        "iShares FTSE 100 UCITS ETF (ISF.L)":       "ISF.L",
        "Vanguard FTSE 100 UCITS ETF (VUKE.L)":     "VUKE.L",
        "iShares FTSE 250 UCITS ETF (MIDD.L)":      "MIDD.L",
        "iShares Core FTSE 100 ETF GBP Acc (CUKX.L)": "CUKX.L",
        "iShares Core MSCI World UCITS ETF (SWDA.L)": "SWDA.L"
    }

    if lse_mode == "Tech Stocks":
        sel    = st.selectbox("Select Tech Stock:", list(lse_tech.keys()), key="lse_tech")
        ticker = lse_tech[sel]
    else:
        sel    = st.selectbox("Select Index ETF:", list(lse_etfs.keys()), key="lse_etf")
        ticker = lse_etfs[sel]

    if st.button("Run Forecast", key="run_lse"):
        df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
               .asfreq("D").fillna(method="ffill")
        if df.empty:
            st.warning("No data available for this selection.")
        else:
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            model = safe_sarimax(df, (1,1,1), (1,1,1,12))
            fc    = model.get_forecast(steps=30)
            idx, vals, ci = (
                pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D"),
                fc.predicted_mean,
                fc.conf_int()
            )

            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_title(f"{sel} Forecast")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            st.write(pd.DataFrame({
                "Forecast": vals,
                "Lower":    ci.iloc[:,0],
                "Upper":    ci.iloc[:,1]
            }, index=idx))
