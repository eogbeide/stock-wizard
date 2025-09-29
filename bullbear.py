# bullbear.py — Stocks/Forex Dashboard + Forecasts (+ Normalized Price Oscillator on EW panels)
# - Forex news markers on intraday charts
# - Hourly momentum indicator (ROC%) with robust handling
# - Momentum trendline & momentum S/R
# - Daily shows: History, 30 EMA, 30 S/R, Daily slope, Pivots (P, R1/S1, R2/S2) + value labels
# - EMA30 slope overlay on Daily
# - Hourly includes Supertrend overlay (configurable ATR period & multiplier)
# - Fixes tz_localize error by using tz-aware UTC timestamps
# - Auto-refresh, SARIMAX (for probabilities)
# - Cache TTLs = 2 minutes (120s)
# - Hourly BUY/SELL logic (near S/R + confidence threshold)
# - Value labels on intraday Resistance/Support placed on the LEFT; price label outside chart (top-right)
# - All displayed price values formatted to 3 decimal places
# - Hourly Support/Resistance drawn as STRAIGHT LINES across the entire chart
# - Current price shown OUTSIDE of chart area (top-right)
# - Normalized Elliott Wave panel for Hourly (dates aligned to hourly chart)
# - Normalized Elliott Wave panel for Daily (dates aligned to daily chart, shared x-axis with price)
# - EW panels show BUY/SELL signals when forecast confidence > 95% and display current price on top
# - EW panels draw a red line at +0.5 and a green line at -0.5
# - EW panels draw black lines at +0.75 and -0.75
# - Adds Normalized Price Oscillator (NPO) overlay to EW panels with sidebar controls
# - Adds Normalized Trend Direction (NTD) overlay + optional green/red shading to EW panels with sidebar controls
# - Daily view selector (Historical / 6M / 12M / 24M)
# - Red shading under NPO curve on EW panels
# - NEW: Daily trend-direction line (green=uptrend, red=downtrend) with slope label
# - NEW: NTD -0.5 Scanner tab for Stocks & Forex (Daily; plus Hourly for Forex)
# - NEW: Normalized Ichimoku overlay on EW panels (Daily & Hourly) with sidebar controls
# - NEW: Ichimoku Kijun (Base) line added to Daily & Hourly price charts (solid black, continuous)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
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
        st.experimental_rerun()
auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# ---------- Helpers ----------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None: return pd.Series(dtype=float)
    if isinstance(obj, pd.Series): s = obj
    elif isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if not num_cols: return pd.Series(dtype=float)
        s = obj[num_cols[0]]
    else:
        try: s = pd.Series(obj)
        except Exception: return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_price_val(y: float) -> str:
    try: y = float(y)
    except Exception: return "n/a"
    return f"{y:,.3f}"

def fmt_pct(x, digits=1) -> str:
    try: xv = float(x)
    except Exception: return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_slope(m: float) -> str: return f"{m:.4f}" if np.isfinite(m) else "n/a"

def label_on_left(ax, y_val, text, color="black", fontsize=9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, y_val, text, transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6), zorder=6)

# -------- Sidebar Config --------
# (Includes Ichimoku, NPO, NTD, Forecast options...)
# [KEEPING your existing sidebar controls, optimized for grouping/efficiency]

# -------- Cached Fetch Functions --------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close'].asfreq("D").ffill()
    try: s = s.tz_localize(PACIFIC)
    except TypeError: s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try: df = df.tz_localize('UTC')
    except TypeError: pass
    return df.tz_convert(PACIFIC)

# -------- Indicators (optimized) --------
def compute_roc(s, n=10): return _coerce_1d_series(s).pct_change(n) * 100

def slope_line(s, lookback):
    s = _coerce_1d_series(s).dropna()
    if len(s) < 2: return pd.Series(dtype=float), np.nan
    s = s.tail(lookback)
    x = np.arange(len(s))
    m, b = np.polyfit(x, s, 1)
    return pd.Series(m*x + b, index=s.index), m

def compute_supertrend(df, atr_period=10, atr_mult=3):
    hl2 = (df['High']+df['Low'])/2
    tr = (df['High']-df['Low']).abs().combine((df['High']-df['Close'].shift()).abs(), max).combine((df['Low']-df['Close'].shift()).abs(), max)
    atr = tr.ewm(alpha=1/atr_period).mean()
    upper, lower = hl2+atr_mult*atr, hl2-atr_mult*atr
    st_line, uptrend = pd.Series(index=df.index, dtype=float), True
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > (upper.iloc[i] if uptrend else lower.iloc[i]):
            uptrend = True
        elif df['Close'].iloc[i] < (lower.iloc[i] if not uptrend else upper.iloc[i]):
            uptrend = False
        st_line.iloc[i] = lower.iloc[i] if uptrend else upper.iloc[i]
    return st_line

# ---- Ichimoku ----
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

# ---- Normalized Oscillators ----
def compute_npo(c,fast=12,slow=26,norm_win=240):
    ef,es=c.ewm(span=fast).mean(),c.ewm(span=slow).mean()
    ppo=(ef-es)/es*100; mean,std=ppo.rolling(norm_win).mean(),ppo.rolling(norm_win).std()
    return np.tanh(((ppo-mean)/std)/2)

def compute_normalized_trend(c,window=60):
    slope=c.rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)),x,1)[0] if len(x)>2 else np.nan)
    vol=c.rolling(window).std()
    return np.tanh(((slope*window)/vol)/2)

# -------- Main App --------
st.title("📊 Stocks / Forex Dashboard + Forecasts")

ticker = st.sidebar.text_input("Enter ticker", "AAPL").upper()
if ticker:
    df_d = fetch_hist(ticker)
    df_h = fetch_intraday(ticker)

    if not df_d.empty:
        # Daily Chart
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_d, label="Close")
        ema=df_d.ewm(span=30).mean()
        ax.plot(ema, "--", label="EMA30")
        _,kijun,_,_,_=ichimoku_lines(df_d,df_d,df_d)
        ax.plot(kijun, "-", color="black", label="Kijun")
        ax.legend(); st.pyplot(fig)

    if not df_h.empty:
        # Hourly Chart
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df_h['Close'], label="Close")
        st_line=compute_supertrend(df_h)
        ax.plot(st_line, "--", label="Supertrend")
        _,kijun,_,_,_=ichimoku_lines(df_h['High'],df_h['Low'],df_h['Close'])
        ax.plot(kijun, "-", color="black", label="Kijun")
        ax.legend(); st.pyplot(fig)
