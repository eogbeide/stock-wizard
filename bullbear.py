import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import time
import pytz

# --- Page config ---
st.set_page_config(page_title="📊 Dashboard & Forecasts", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Auto-refresh ---
REFRESH_INTERVAL = 120
PACIFIC = pytz.timezone("US/Pacific")
def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        st.rerun()
auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# -------- Sidebar Controls --------
st.sidebar.subheader("Ichimoku")
show_ichi = st.sidebar.checkbox("Show Ichimoku", value=True)
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1)
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1)
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1)

# -------- Yahoo Finance --------
@st.cache_data(ttl=120)
def fetch_data(ticker: str, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    # ✅ FIX: check if index is tz-aware before localizing
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
    else:
        df.index = df.index.tz_convert(PACIFIC)
    return df

# -------- Ichimoku --------
def ichimoku_lines(h,l,c,conv=9,base=26,span_b=52):
    tenkan=(h.rolling(conv).max()+l.rolling(conv).min())/2
    kijun=(h.rolling(base).max()+l.rolling(base).min())/2
    span_a=(tenkan+kijun)/2
    span_b_=(h.rolling(span_b).max()+l.rolling(span_b).min())/2
    chikou=c.shift(-base)
    return tenkan,kijun,span_a,span_b_,chikou

# -------- Main App --------
st.title("📊 Stocks / Forex Dashboard + Forecasts")

ticker = st.sidebar.text_input("Enter ticker", "AAPL").upper()
if ticker:
    df_d = fetch_data(ticker, period="1y", interval="1d")
    df_h = fetch_data(ticker, period="5d", interval="1h")

    if not df_d.empty:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_d['Close'], label="Close")
        _,kijun,_,_,_=ichimoku_lines(df_d['High'],df_d['Low'],df_d['Close'],
                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
        if show_ichi:
            ax.plot(kijun, "-", color="black", label="Kijun")
        ax.legend(); st.pyplot(fig)

    if not df_h.empty:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_h['Close'], label="Close")
        _,kijun,_,_,_=ichimoku_lines(df_h['High'],df_h['Low'],df_h['Close'],
                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
        if show_ichi:
            ax.plot(kijun, "-", color="black", label="Kijun")
        ax.legend(); st.pyplot(fig)
