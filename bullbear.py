# bullbear.py — Stocks/Forex Dashboard + Forecasts

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory

# --- Page config ---
st.set_page_config(page_title="📊 Dashboard & Forecasts", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- Minimal CSS ---
st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 { transform: none !important; visibility: visible !important; width: 100% !important; position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

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

# ---------- Helpers ----------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None: return pd.Series(dtype=float)
    if isinstance(obj, pd.Series): return pd.to_numeric(obj, errors="coerce")
    if isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        return pd.to_numeric(obj[num_cols[0]], errors="coerce") if num_cols else pd.Series(dtype=float)
    try: return pd.Series(obj)
    except: return pd.Series(dtype=float)

def fmt_price_val(y: float) -> str:
    try: return f"{float(y):,.3f}"
    except: return "n/a"

# -------- Sidebar Controls --------
st.sidebar.subheader("Ichimoku (EW Panels & Price Kijun)")
show_ichi = st.sidebar.checkbox("Show Ichimoku", value=True)
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1)
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1)
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1)
ichi_norm_win = st.sidebar.slider("Normalization window", 30, 600, 240, 10)
ichi_price_weight = st.sidebar.slider("Weight: Price vs Cloud (EW)", 0.0, 1.0, 0.6, 0.05)

# -------- Yahoo Finance --------
@st.cache_data(ttl=120)
def fetch_data(ticker: str, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty: return df
    df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
    return df

# -------- Ichimoku --------
def ichimoku_lines(h,l,c,conv=9,base=26,span_b=52):
    tenkan=(h.rolling(conv).max()+l.rolling(conv).min())/2
    kijun=(h.rolling(base).max()+l.rolling(base).min())/2
    span_a=(tenkan+kijun)/2
    span_b_=(h.rolling(span_b).max()+l.rolling(span_b).min())/2
    chikou=c.shift(-base)
    return tenkan,kijun,span_a,span_b_,chikou

def compute_normalized_ichimoku(h,l,c,conv=9,base=26,span_b=52,norm_win=240,w=0.6):
    tenkan,kijun,sa,sb,_=ichimoku_lines(h,l,c,conv,base,span_b)
    cloud=((sa+sb)/2).reindex(c.index)
    vol=c.rolling(norm_win,min_periods=norm_win//10).std()
    z1=(c-cloud)/vol; z2=(tenkan-kijun)/vol
    return np.tanh((w*z1+(1-w)*z2)/2)

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
        st.pyplot(fig)

    if not df_h.empty:
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_h['Close'], label="Close")
        _,kijun,_,_,_=ichimoku_lines(df_h['High'],df_h['Low'],df_h['Close'],
                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
        if show_ichi:
            ax.plot(kijun, "-", color="black", label="Kijun")
        st.pyplot(fig)
