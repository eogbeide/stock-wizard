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
from matplotlib.transforms import blended_transform_factory  # for left-side labels

# --- Page config (must be the first Streamlit call) ---
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.experimental_rerun()
        except Exception:
            pass

auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# ---------- Helpers ----------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if not num_cols:
            return pd.Series(dtype=float)
        s = obj[num_cols[0]]
    else:
        try:
            s = pd.Series(obj)
        except Exception:
            return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_price_val(y: float) -> str:
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.3f}"

def fmt_slope(m: float) -> str:
    return f"{m:.4f}" if np.isfinite(m) else "n/a"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    try:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.01, y_val, text,
            transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=6
        )
    except Exception:
        pass
# =====================================================
# Sidebar controls
# =====================================================

st.sidebar.subheader("Chart View")
view_choice = st.sidebar.radio("Select Daily view range", ["Historical", "6M", "12M", "24M"], index=1)

st.sidebar.subheader("Normalized Price Oscillator (NPO)")
show_npo = st.sidebar.checkbox("Show NPO", value=True)
npo_win = st.sidebar.slider("Window", 30, 600, 240, 10)

st.sidebar.subheader("Normalized Trend Direction (NTD)")
show_ntd = st.sidebar.checkbox("Show NTD", value=True)
ntd_win = st.sidebar.slider("Window", 30, 600, 240, 10)
shade_ntd = st.sidebar.checkbox("Shade background by NTD", value=False)

st.sidebar.subheader("Ichimoku (EW Panels & Price Kijun)")
show_ichi = st.sidebar.checkbox("Show Ichimoku", value=True)
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1)
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1)
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1)
ichi_norm_win = st.sidebar.slider("Normalization window", 30, 600, 240, 10)
ichi_price_weight = st.sidebar.slider("Weight: Price vs Cloud (EW)", 0.0, 1.0, 0.6, 0.05)

# =====================================================
# Caching for Yahoo Finance
# =====================================================

@st.cache_data(ttl=120)
def yf_download(ticker: str, period: str, interval: str):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        if df.empty: return df
        df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
        return df
    except Exception:
        return pd.DataFrame()

# =====================================================
# Indicators
# =====================================================

def supertrend(df, period=10, multiplier=3):
    high = df['High']
    low = df['Low']
    close = df['Close']
    hl2 = (high + low) / 2
    atr = truerange(df).rolling(period).mean()

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(df)):
        if close.iloc[i] > final_upperband.iloc[i-1]:
            final_upperband.iloc[i] = upperband.iloc[i]
        else:
            final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i-1])

        if close.iloc[i] < final_lowerband.iloc[i-1]:
            final_lowerband.iloc[i] = lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i-1])

    stx = pd.Series(index=close.index, dtype=float)
    for i in range(period, len(df)):
        if close.iloc[i] <= final_upperband.iloc[i]:
            stx.iloc[i] = final_upperband.iloc[i]
        elif close.iloc[i] >= final_lowerband.iloc[i]:
            stx.iloc[i] = final_lowerband.iloc[i]
    return stx

def truerange(df):
    h_l = df['High'] - df['Low']
    h_pc = abs(df['High'] - df['Close'].shift())
    l_pc = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr

def normalized_price_oscillator(close, window=240):
    c = _coerce_1d_series(close)
    ma = c.rolling(window).mean()
    sd = c.rolling(window).std().replace(0, np.nan)
    return ((c - ma) / sd).clip(-2, 2)

def normalized_trend_direction(close, window=240):
    c = _coerce_1d_series(close)
    returns = c.pct_change()
    ma = returns.rolling(window).mean()
    sd = returns.rolling(window).std().replace(0, np.nan)
    return (ma / sd).clip(-2, 2)

# =====================================================
# Ichimoku functions (classic + normalized)
# =====================================================

def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    H = _coerce_1d_series(high)
    L = _coerce_1d_series(low)
    C = _coerce_1d_series(close)
    if H.empty or L.empty or C.empty:
        idx = C.index if not C.empty else (H.index if not H.empty else L.index)
        return (pd.Series(index=idx, dtype=float),)*5

    tenkan = (H.rolling(conv).max() + L.rolling(conv).min()) / 2.0
    kijun  = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (H.rolling(span_b).max() + L.rolling(span_b).min()) / 2.0
    span_a = span_a_raw.shift(base) if shift_cloud else span_a_raw
    span_b = span_b_raw.shift(base) if shift_cloud else span_b_raw
    chikou = C.shift(-base)
    return tenkan, kijun, span_a, span_b, chikou

def compute_normalized_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                                conv: int = 9, base: int = 26, span_b: int = 52,
                                norm_win: int = 240, price_weight: float = 0.6) -> pd.Series:
    C = _coerce_1d_series(close).astype(float)
    H = _coerce_1d_series(high).astype(float)
    L = _coerce_1d_series(low).astype(float)
    if C.empty or H.empty or L.empty:
        return pd.Series(index=C.index, dtype=float)

    tenkan, kijun, span_a, span_b, _ = ichimoku_lines(H, L, C, conv=conv, base=base, span_b=span_b, shift_cloud=False)
    cloud_mid = ((span_a + span_b) / 2.0).reindex(C.index)

    minp = max(10, norm_win // 10)
    vol = C.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)

    z1 = (C - cloud_mid) / vol
    z2 = (tenkan - kijun) / vol

    blend = price_weight * z1 + (1.0 - price_weight) * z2
    n_ichi = np.tanh(blend / 2.0)
    return n_ichi.reindex(C.index)
# =====================================================
# Chart plotting logic
# =====================================================

def plot_daily_chart(df_daily):
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot price
    ax.plot(df_daily.index, df_daily['Close'], label="Close", color="blue")

    # EMA30
    ema30 = df_daily['Close'].ewm(span=30).mean()
    ax.plot(df_daily.index, ema30, label="EMA30", color="orange")

    # EMA slope overlay
    slope = (ema30 - ema30.shift(1))
    slope_color = np.where(slope > 0, "green", "red")
    ax.scatter(df_daily.index, ema30, c=slope_color, s=5, label="EMA30 slope")

    # Pivots
    piv = (df_daily['High'] + df_daily['Low'] + df_daily['Close']) / 3
    ax.plot(df_daily.index, piv, linestyle="--", color="purple", label="Pivot (P)")

    # --- Ichimoku Kijun (Base line) ---
    if show_ichi:
        _, kijun_d, _, _, _ = ichimoku_lines(df_daily['High'], df_daily['Low'], df_daily['Close'],
                                             conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
        ax.plot(kijun_d.index, kijun_d.values, "-", linewidth=1.8, color="black", label="Ichimoku Kijun")

    ax.set_title("Daily Price Chart + EMA30 + Pivots + Kijun")
    ax.legend(loc="upper left")
    st.pyplot(fig)

def plot_hourly_chart(df_hourly):
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot price
    ax.plot(df_hourly.index, df_hourly['Close'], label="Close", color="blue")

    # Supertrend
    stx = supertrend(df_hourly)
    ax.plot(stx.index, stx.values, linestyle="--", color="purple", label="Supertrend")

    # --- Ichimoku Kijun (Base line) ---
    if show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(df_hourly['High'], df_hourly['Low'], df_hourly['Close'],
                                             conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
        ax.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label="Ichimoku Kijun")

    ax.set_title("Hourly Price Chart + Supertrend + Kijun")
    ax.legend(loc="upper left")
    st.pyplot(fig)
# =====================================================
# Elliott Wave Panels
# =====================================================

def plot_daily_ew_panel(df_daily):
    fig, ax = plt.subplots(figsize=(12, 3))

    close = df_daily['Close']
    ew_norm = normalized_price_oscillator(close, window=npo_win)

    ax.plot(close.index, ew_norm, label="Normalized EW", color="blue")

    # --- Normalized Price Oscillator (NPO) ---
    if show_npo:
        npo = normalized_price_oscillator(close, window=npo_win)
        ax.plot(npo.index, npo.values, color="red", linewidth=1.2, label="NPO")
        ax.fill_between(npo.index, 0, npo.values, where=npo.values < 0, color="red", alpha=0.2)

    # --- Normalized Trend Direction (NTD) ---
    if show_ntd:
        ntd = normalized_trend_direction(close, window=ntd_win)
        ax.plot(ntd.index, ntd.values, color="green", linewidth=1.2, label="NTD")
        if shade_ntd:
            ax.fill_between(ntd.index, -1, 1, where=ntd.values > 0, color="green", alpha=0.1)
            ax.fill_between(ntd.index, -1, 1, where=ntd.values < 0, color="red", alpha=0.1)

    # --- Normalized Ichimoku ---
    if show_ichi:
        ichi_d_norm = compute_normalized_ichimoku(df_daily['High'], df_daily['Low'], df_daily['Close'],
                                                  conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                                                  norm_win=ichi_norm_win, price_weight=ichi_price_weight)
        ax.plot(ichi_d_norm.index, ichi_d_norm.values, "-", linewidth=1.4, color="black", label="IchimokuN")

    # Reference lines
    ax.axhline(0.5, color="red", linestyle="-", linewidth=1)
    ax.axhline(-0.5, color="green", linestyle="-", linewidth=1)
    ax.axhline(0.75, color="black", linestyle="-", linewidth=1)
    ax.axhline(-0.75, color="black", linestyle="-", linewidth=1)

    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Daily Elliott Wave + NPO + NTD + Ichimoku")
    ax.legend(loc="upper left")
    st.pyplot(fig)


def plot_hourly_ew_panel(df_hourly):
    fig, ax = plt.subplots(figsize=(12, 3))

    close = df_hourly['Close']
    ew_norm = normalized_price_oscillator(close, window=npo_win)

    ax.plot(close.index, ew_norm, label="Normalized EW", color="blue")

    # --- Normalized Price Oscillator (NPO) ---
    if show_npo:
        npo = normalized_price_oscillator(close, window=npo_win)
        ax.plot(npo.index, npo.values, color="red", linewidth=1.2, label="NPO")
        ax.fill_between(npo.index, 0, npo.values, where=npo.values < 0, color="red", alpha=0.2)

    # --- Normalized Trend Direction (NTD) ---
    if show_ntd:
        ntd = normalized_trend_direction(close, window=ntd_win)
        ax.plot(ntd.index, ntd.values, color="green", linewidth=1.2, label="NTD")
        if shade_ntd:
            ax.fill_between(ntd.index, -1, 1, where=ntd.values > 0, color="green", alpha=0.1)
            ax.fill_between(ntd.index, -1, 1, where=ntd.values < 0, color="red", alpha=0.1)

    # --- Normalized Ichimoku ---
    if show_ichi:
        ichi_h_norm = compute_normalized_ichimoku(df_hourly['High'], df_hourly['Low'], df_hourly['Close'],
                                                  conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                                                  norm_win=ichi_norm_win, price_weight=ichi_price_weight)
        ax.plot(ichi_h_norm.index, ichi_h_norm.values, "-", linewidth=1.4, color="black", label="IchimokuN")

    # Reference lines
    ax.axhline(0.5, color="red", linestyle="-", linewidth=1)
    ax.axhline(-0.5, color="green", linestyle="-", linewidth=1)
    ax.axhline(0.75, color="black", linestyle="-", linewidth=1)
    ax.axhline(-0.75, color="black", linestyle="-", linewidth=1)

    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Hourly Elliott Wave + NPO + NTD + Ichimoku")
    ax.legend(loc="upper left")
    st.pyplot(fig)
# =====================================================
# Main App Logic
# =====================================================

st.title("📊 Stocks / Forex Dashboard + Forecasts")

# --- Ticker selection ---
ticker = st.sidebar.text_input("Enter ticker symbol", value="AAPL").upper()

if ticker:
    df_daily = yf_download(ticker, period="1y", interval="1d")
    df_hourly = yf_download(ticker, period="5d", interval="1h")

    if not df_daily.empty:
        with st.expander("Daily Price Chart", expanded=True):
            plot_daily_chart(df_daily)
        with st.expander("Daily Elliott Wave Panel", expanded=True):
            plot_daily_ew_panel(df_daily)

    if not df_hourly.empty:
        with st.expander("Hourly Price Chart", expanded=True):
            plot_hourly_chart(df_hourly)
        with st.expander("Hourly Elliott Wave Panel", expanded=True):
            plot_hourly_ew_panel(df_hourly)
