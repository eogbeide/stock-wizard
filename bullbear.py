# bullbear.py — Stocks/Forex Dashboard + Forecasts
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
# - Daily Normalized Elliott Wave panel (dates aligned to daily chart, shared x-axis with price)
# - EW panels show BUY/SELL signals when forecast confidence > 95% and display current price on top
# - EW panels draw a red line at +0.5 and a green line at -0.5
# - EW panels draw black lines at +0.75 and -0.75
# - Adds Normalized Price Oscillator (NPO) overlay to EW panels with sidebar controls
# - Adds Normalized Trend Direction (NTD) overlay + optional green/red shading to EW panels with sidebar controls
# - Daily view selector (Historical / 6M / 12M / 24M)
# - Red shading under NPO curve on EW panels
# - Daily trend-direction line (green=uptrend, red=downtrend) with slope label
# - NTD -0.5 Scanner tab for Stocks & Forex (Daily; plus Hourly for Forex)
# - Normalized Ichimoku overlay on EW panels (Daily) with sidebar controls
# - Ichimoku Kijun (Base) line added to Daily & Hourly price charts (solid black, continuous)
# - UPDATED: Removed Hourly EW panel and added **Normalized RSI (NRSI)** panel below the hourly price chart
# - UPDATED: RSI panel shows **NRSI + NVol + NMACD(+signal)** with guides at 0 (dashed), ±0.5 (black), ±0.75 (thick red)
# - UPDATED: Added **NTD overlay & Trend direction with % certainty** to RSI panel
# - NEW: Refresh buttons on the **right side** of both **Daily** and **Intraday** charts
# - NEW: **20Y History** tab to visualize 20 years of historical close prices

import streamlit as sts
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory  # for left-side labels

# --- Page config ---
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
            st.rerun()  # FIX: experimental_rerun -> rerun
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

# Place text at the left edge (x in axes coords, y in data coords)
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

# Range helper for daily views
def subset_by_daily_view(obj, view_label: str):
    if obj is None or len(obj.index) == 0:
        return obj
    idx = obj.index
    end = idx.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        start = idx.min()
    else:
        start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return obj.loc[(idx >= start) & (idx <= end)]

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")

daily_view = st.sidebar.selectbox(
    "Daily view range:",
    ["Historical", "6M", "12M", "24M"],
    index=2,
    key="sb_daily_view"
)

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

# Hourly Momentum controls
st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=True, key="sb_show_mom_hourly")
mom_lb_hourly   = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

# Hourly NRSI controls
st.sidebar.subheader("Hourly Normalized RSI")
show_nrsi   = st.sidebar.checkbox("Show NRSI (hourly)", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (bars)", 5, 60, 14, 1, key="sb_nrsi_period")

# Hourly Supertrend controls
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# Signal logic controls
st.sidebar.subheader("Signal Logic (Hourly)")
signal_threshold = st.sidebar.slider("Signal confidence threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# Daily EW controls
st.sidebar.subheader("Normalized Elliott Wave (Daily)")
pivot_lookback_d = st.sidebar.slider("Pivot lookback (days)", 3, 31, 9, 2, key="sb_pivot_lb_d")
norm_window_d    = st.sidebar.slider("Normalization window (days)", 30, 1200, 360, 10, key="sb_norm_win_d")
waves_to_annotate_d = st.sidebar.slider("Annotate recent waves (daily)", 3, 12, 7, 1, key="sb_wave_ann_d")

# NPO overlay controls (for EW panels)
st.sidebar.subheader("Normalized Price Oscillator (overlay on EW panels)")
show_npo    = st.sidebar.checkbox("Show NPO overlay", value=True, key="sb_show_npo")
npo_fast    = st.sidebar.slider("NPO fast EMA", 5, 30, 12, 1, key="sb_npo_fast")
npo_slow    = st.sidebar.slider("NPO slow EMA", 10, 60, 26, 1, key="sb_npo_slow")
npo_norm_win= st.sidebar.slider("NPO normalization window", 30, 600, 240, 10, key="sb_npo_norm")

# NTD overlay controls (for EW panels AND RSI panel)
st.sidebar.subheader("Normalized Trend (EW/RSI panels)")
show_ntd  = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window= st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (EW only: green=up, red=down)", value=True, key="sb_ntd_shade")

# Ichimoku controls (Normalized for EW + Kijun on price)
st.sidebar.subheader("Normalized Ichimoku (EW panels) + Kijun on price")
show_ichi = st.sidebar.checkbox("Show Ichimoku", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")
ichi_norm_win = st.sidebar.slider("Ichimoku normalization window (EW)", 30, 600, 240, 10, key="sb_ichi_norm")
ichi_price_weight = st.sidebar.slider("Weight: Price vs Cloud (EW)", 0.0, 1.0, 0.6, 0.05, key="sb_ichi_w")

# Forex news controls (only shown in Forex mode)
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
else:
    show_fx_news = False
    news_window_days = 7

# Universe
if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'
    ]

# --- Cache helpers (TTL = 120 seconds) ---
@st.cache_data(ttl=120)
def fetch_hist(ticker: str, nonce: int = 0) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str, nonce: int = 0) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[['Open','High','Low','Close']].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

# NEW: 20-year history (close) for Stocks & Forex
@st.cache_data(ttl=3600)
def fetch_hist_20y(ticker: str, nonce: int = 0) -> pd.Series:
    # Use Yahoo period=20y to let YF choose best start date; fallback to 2004 if needed.
    try:
        df = yf.download(ticker, period="20y")['Close']
    except Exception:
        df = yf.download(ticker, start="2004-01-01", end=pd.to_datetime("today"))['Close']
    s = df.dropna().asfreq("D").fillna(method="ffill")
    try:
        s = s.tz_localize("UTC").tz_convert(PACIFIC)
    except Exception:
        try:
            s = s.tz_convert(PACIFIC)
        except Exception:
            # last resort: make it tz-aware in PACIFIC
            s.index = s.index.tz_localize(PACIFIC)
    return s

# Add 'nonce' so pressing refresh busts the cache
@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d", nonce: int = 0) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---- Indicators ----
def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return {}
    hi = float(s.max()); lo = float(s.min())
    diff = hi - lo
    if diff == 0:
        return {"100%": lo}
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5   * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100%": lo,
    }

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {'High','Low','Close'}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    try:
        H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    except Exception:
        return {}
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

def slope_line(series_like, lookback: int):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    s = s.iloc[-lookback:] if lookback > 0 else s
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    yhat = pd.Series(m * x + b, index=s.index)
    return yhat, float(m)

def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

# ---- RSI / Normalized RSI ----
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(s.index)

def compute_nrsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = compute_rsi(close, period=period)
    nrsi = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    return nrsi.reindex(rsi.index)

# ---- Normalized MACD (price-based) ----
def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
                  norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),)*3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig

    # Normalize MACD & signal to [-1,1] using rolling z then tanh
    minp = max(10, norm_win//10)
    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)
    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return nmacd.reindex(s.index), nsignal.reindex(s.index), nhist.reindex(s.index)

# ---- Normalized Volume (z-score → tanh in [-1,1]) ----
def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win//10)
    m = v.rolling(norm_win, min_periods=minp).mean()
    sd = v.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (v - m) / sd
    return np.tanh(z / 3.0)

# ---- Normalized Price Oscillator (for EW panel) ----
def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26, norm_win: int = 240) -> pd.Series:
    s = _coerce_1d_series(close)
    if s.empty or not np.isfinite(fast) or not np.isfinite(slow) or fast <= 0 or slow <= 0:
        return pd.Series(index=s.index, dtype=float)
    if fast >= slow:
        fast, slow = max(1, slow - 1), slow
        if fast >= slow:
            return pd.Series(index=s.index, dtype=float)
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean().replace(0, np.nan)
    ppo = (ema_fast - ema_slow) / ema_slow * 100.0
    minp = max(10, int(norm_win)//10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std  = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
    z = (ppo - mean) / std
    npo = np.tanh(z / 2.0)
    return npo.reindex(s.index)

# ---- Normalized Trend Direction (for EW & RSI panels) ----
def compute_normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)

    minp = max(5, window // 3)
    def _slope(y: pd.Series) -> float:
        y = pd.Series(y).dropna()
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        try:
            m, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
        except Exception:
            return np.nan
        return float(m)

    slope_roll = s.rolling(window, min_periods=minp).apply(_slope, raw=False)
    vol = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    ntd_raw = (slope_roll * window) / vol
    ntd = np.tanh(ntd_raw / 2.0)
    return ntd.reindex(s.index)

def shade_ntd_regions(ax, ntd: pd.Series):
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, step=None)
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, step=None)

# Red shading under NPO curve
def shade_npo_regions(ax, npo: pd.Series):
    if npo is None or npo.empty:
        return
    pos = npo.where(npo > 0)
    neg = npo.where(npo < 0)
    ax.fill_between(pos.index, 0, pos, alpha=0.15, color="tab:red")
    ax.fill_between(neg.index, 0, neg, alpha=0.08, color="tab:red")

# Daily trend-direction line helper
def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "tab:green" if m >= 0 else "tab:red"
    ax.plot(s.index, yhat, "-", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# ---- Supertrend helpers (hourly overlay) ----
def _true_range(df: pd.DataFrame):
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = _true_range(df[['High','Low','Close']])
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["ST","in_uptrend","upperband","lowerband"])
    ohlc = df[['High','Low','Close']].copy()
    hl2 = (ohlc['High'] + ohlc['Low']) / 2.0
    atr = compute_atr(ohlc, atr_period)
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr
    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_up   = pd.Series(index=ohlc.index, dtype=bool)
    st_line.iloc[0] = upperband.iloc[0]
    in_up.iloc[0]   = True
    for i in range(1, len(ohlc)):
        prev_st = st_line.iloc[i-1]
        prev_up = in_up.iloc[i-1]
        up_i = min(upperband.iloc[i], prev_st) if prev_up else upperband.iloc[i]
        dn_i = max(lowerband.iloc[i], prev_st) if not prev_up else lowerband.iloc[i]
        close_i = ohlc['Close'].iloc[i]
        if close_i > up_i:
            curr_up = True
        elif close_i < dn_i:
            curr_up = False
        else:
            curr_up = prev_up
        in_up.iloc[i]   = curr_up
        st_line.iloc[i] = dn_i if curr_up else up_i
    return pd.DataFrame({
        "ST": st_line, "in_uptrend": in_up,
        "upperband": upperband, "lowerband": lowerband
    })

# ---- Normalized Elliott Wave (for DAILY panel) ----
def compute_normalized_elliott_wave(close: pd.Series,
                                    pivot_lb: int = 7,
                                    norm_win: int = 240):
    s = _coerce_1d_series(close).dropna()
    if s.empty:
        return pd.Series(index=_coerce_1d_series(close).index, dtype=float), pd.DataFrame(columns=["time","price","type","wave"])

    minp = max(10, norm_win//10)
    mean = s.rolling(norm_win, min_periods=minp).mean()
    std  = s.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (s - mean) / std
    wave_norm = np.tanh(z / 2.0)
    wave_norm = wave_norm.reindex(close.index)

    if pivot_lb % 2 == 0:
        pivot_lb += 1
    roll_max = s.rolling(pivot_lb, center=True).max()
    roll_min = s.rolling(pivot_lb, center=True).min()

    pivots = []
    half = pivot_lb // 2
    for i in range(half, len(s)-half):
        if not np.isfinite(s.iloc[i]):
            continue
        if s.iloc[i] == roll_max.iloc[i]:
            pivots.append((s.index[i], float(s.iloc[i]), 'H'))
        elif s.iloc[i] == roll_min.iloc[i]:
            pivots.append((s.index[i], float(s.iloc[i]), 'L'))

    dedup = []
    for t, p, typ in pivots:
        if not dedup:
            dedup.append((t,p,typ))
        else:
            pt, pp, ptyp = dedup[-1]
            if typ == ptyp:
                if (typ == 'H' and p > pp) or (typ == 'L' and p < pp):
                    dedup[-1] = (t,p,typ)
            else:
                dedup.append((t,p,typ))

    waves = []
    wave_num = 1
    for t, p, typ in dedup:
        waves.append((t,p,typ,wave_num))
        wave_num += 1
        if wave_num > 5:
            wave_num = 1

    pivots_df = pd.DataFrame(waves, columns=["time","price","type","wave"])
    return wave_norm, pivots_df

# ---- Ichimoku (classic + normalized) ----
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

# ========= Signals & News helpers =========
EW_CONFIDENCE = 0.95

def elliott_conf_signal(price_now: float, fc_vals: pd.Series, conf: float = EW_CONFIDENCE):
    fc = _coerce_1d_series(fc_vals).dropna().to_numpy(dtype=float)
    if fc.size == 0 or not np.isfinite(price_now):
        return None
    p_up = float(np.mean(fc > price_now))
    p_dn = float(np.mean(fc < price_now))
    if p_up >= conf:
        return {"side": "BUY", "prob": p_up}
    if p_dn >= conf:
        return {"side": "SELL", "prob": p_dn}
    return None

def sr_proximity_signal(hc: pd.Series, res_h: pd.Series, sup_h: pd.Series,
                        fc_vals: pd.Series, threshold: float, prox: float):
    try:
        last_close = float(hc.iloc[-1])
        res = float(res_h.iloc[-1])
        sup = float(sup_h.iloc[-1])
    except Exception:
        return None

    if not np.all(np.isfinite([last_close, res, sup])) or res <= sup:
        return None

    near_support = last_close <= sup * (1.0 + prox)
    near_resist  = last_close >= res * (1.0 - prox)

    fc = np.asarray(_coerce_1d_series(fc_vals).dropna(), dtype=float)
    if fc.size == 0:
        return None
    p_up_from_here = float(np.mean(fc > last_close))
    p_dn_from_here = float(np.mean(fc < last_close))

    if near_support and p_up_from_here >= threshold:
        return {
            "side": "BUY",
            "prob": p_up_from_here,
            "level": sup,
            "reason": f"Near support {fmt_price_val(sup)} with {fmt_pct(p_up_from_here)} up-confidence ≥ {fmt_pct(threshold)}"
        }
    if near_resist and p_dn_from_here >= threshold:
        return {
            "side": "SELL",
            "prob": p_dn_from_here,
            "level": res,
            "reason": f"Near resistance {fmt_price_val(res)} with {fmt_pct(p_dn_from_here)} down-confidence ≥ {fmt_pct(threshold)}"
        }
    return None

@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_news(symbol: str, window_days: int = 7) -> pd.DataFrame:
    rows = []
    try:
        news_list = yf.Ticker(symbol).news or []
    except Exception:
        news_list = []
    for item in news_list:
        ts = item.get("providerPublishTime") or item.get("pubDate")
        if ts is None:
            continue
        try:
            dt_utc = pd.to_datetime(ts, unit="s", utc=True)
        except (ValueError, OverflowError, TypeError):
            try:
                dt_utc = pd.to_datetime(ts, utc=True)
            except Exception:
                continue
        dt_pst = dt_utc.tz_convert(PACIFIC)
        rows.append({
            "time": dt_pst,
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", "")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

def draw_news_markers(ax, times, ymin, ymax, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# ========= Cached last values for scanning =========
@st.cache_data(ttl=120)
def last_daily_ntd_value(symbol: str, ntd_win: int):
    try:
        s = fetch_hist(symbol)
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None

@st.cache_data(ttl=120)
def last_hourly_ntd_value(symbol: str, ntd_win: int, period: str = "1d"):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df:
            return np.nan, None
        s = df["Close"].ffill()
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'intraday_nonce' not in st.session_state:
    st.session_state.intraday_nonce = 0  # increments to bust intraday cache
if 'daily_nonce' not in st.session_state:
    st.session_state.daily_nonce = 0  # increments to bust daily caches
if 'long_nonce' not in st.session_state:
    st.session_state.long_nonce = 0   # for 20Y history cache busting

# Tabs (added 6th tab)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.5 Scanner",
    "20Y History"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    auto_run = (
        st.session_state.run_all and (
            sel != st.session_state.ticker or
            hour_range != st.session_state.get("hour_range")
        )
    )

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
        df_hist = fetch_hist(sel, nonce=st.session_state.daily_nonce)
        df_ohlc = fetch_hist_ohlc(sel, nonce=st.session_state.daily_nonce)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range], nonce=st.session_state.intraday_nonce)
        st.session_state.update({
            "df_hist": df_hist,
            "df_ohlc": df_ohlc,
            "fc_idx": fc_idx,
            "fc_vals": fc_vals,
            "fc_ci": fc_ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc

        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily (Price + EW) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            wave_norm_d, piv_df_d = compute_normalized_elliott_wave(df, pivot_lb=pivot_lookback_d, norm_win=norm_window_d)
            npo_d = compute_npo(df, fast=npo_fast, slow=npo_slow, norm_win=npo_norm_win) if show_npo else pd.Series(index=df.index, dtype=float)
            ntd_d = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            ichi_d = pd.Series(index=df.index, dtype=float)
            kijun_d = pd.Series(index=df.index, dtype=float)

            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                ichi_d = compute_normalized_ichimoku(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    norm_win=ichi_norm_win, price_weight=ichi_price_weight
                )
                _, kijun_d, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                )
                kijun_d = kijun_d.ffill().bfill()

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            wave_d_show = wave_norm_d.reindex(df_show.index)
            npo_d_show  = npo_d.reindex(df_show.index)
            ntd_d_show  = ntd_d.reindex(df_show.index)
            ichi_d_show = ichi_d.reindex(df_show.index).ffill().bfill()
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            piv_df_d_show = piv_df_d[(piv_df_d["time"] >= df_show.index.min()) & (piv_df_d["time"] <= df_show.index.max())] if not piv_df_d.empty else piv_df_d

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            ax.set_title(f"{sel} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema_show.empty:
                ax.plot(yhat_ema_show.index, yhat_ema_show.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            if len(res30_show) and len(sup30_show):
                r30_last = float(res30_show.iloc[-1]); s30_last = float(sup30_show.iloc[-1])
                ax.text(df_show.index[-1], r30_last, f"  30R = {fmt_price_val(r30_last)}", va="bottom")
                ax.text(df_show.index[-1], s30_last, f"  30S = {fmt_price_val(s30_last)}", va="top")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)

            axdw.set_title("Daily Normalized Elliott Wave + NPO + NTD + Ichimoku (normalized)")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_npo and not npo_d_show.dropna().empty:
                shade_npo_regions(axdw, npo_d_show)

            axdw.plot(wave_d_show.index, wave_d_show, label="Norm EW (Daily)", linewidth=1.8)
            if show_npo and not npo_d_show.dropna().empty:
                axdw.plot(npo_d_show.index, npo_d_show, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")
            if show_ichi and not ichi_d_show.dropna().empty:
                axdw.plot(ichi_d_show.index, ichi_d_show.values, "-", linewidth=1.4, color="black",
                          label=f"IchimokuN (c{ichi_conv},b{ichi_base},sb{ichi_spanb})")

            for yline, style, col, lbl in [
                (0.0, "--", None, "EW 0"),
                (0.5, "-", "tab:red", "EW +0.5"),
                (-0.5, "-", "tab:green", "EW -0.5"),
                (0.75, "-", "black", "EW +0.75"),
                (-0.75, "-", "black", "EW -0.75"),
            ]:
                axdw.axhline(yline, linestyle=style, linewidth=1, color=col, label=lbl)

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")

            if not piv_df_d_show.empty:
                show_df_d = piv_df_d_show.tail(int(waves_to_annotate_d))
                for _, r in show_df_d.iterrows():
                    t = r["time"]; w = r["wave"]; typ = r["type"]
                    ylab = 0.9 if typ == 'H' else -0.9
                    axdw.annotate(str(int(w)), (t, ylab),
                                  xytext=(0, 0), textcoords="offset points",
                                  ha="center", va="center",
                                  fontsize=9, fontweight="bold")

            px_daily = _safe_last_float(df)
            ew_sig_d = elliott_conf_signal(px_daily, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw = axdw.get_position()
            label_txt = f"Price: {fmt_price_val(px_daily)}"
            if ew_sig_d is not None:
                side = ew_sig_d['side']; prob = fmt_pct(ew_sig_d['prob'], digits=0)
                label_txt += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily)}  ({prob})"
            fig.text(posdw.x1, posdw.y1 + 0.01, label_txt, ha="right", va="bottom",
                     fontsize=10, fontweight="bold")

            axdw.legend(loc="lower left", framealpha=0.5)

            # ---- RIGHT-SIDE REFRESH BUTTON (Tab 1 DAILY) ----
            dl, dr = st.columns([0.92, 0.08])
            with dl:
                st.pyplot(fig)
            with dr:
                st.markdown("### ")
                if st.button("🔄 Refresh", key="refresh_daily_tab1"):
                    st.session_state.daily_nonce += 1
                    # Refetch daily data and forecast
                    st.session_state.df_hist = fetch_hist(sel, nonce=st.session_state.daily_nonce)
                    st.session_state.df_ohlc = fetch_hist_ohlc(sel, nonce=st.session_state.daily_nonce)
                    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(st.session_state.df_hist)
                    st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci = fc_idx, fc_vals, fc_ci
                    st.session_state.last_refresh = time.time()
                    st.rerun()
                st.caption(f"Updated {int(time.time()-st.session_state.last_refresh)}s ago")

        # ----- Hourly (price + NRSI/MACD/Vol + momentum) -----
        if chart in ("Hourly","Both"):
            intraday = st.session_state.intraday
            if intraday is None or intraday.empty or "Close" not in intraday:
                st.warning("No intraday data available.")
            else:
                hc = intraday["Close"].ffill()
                he = hc.ewm(span=20).mean()
                xh = np.arange(len(hc))
                slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
                trend_h = slope_h * xh + intercept_h
                res_h = hc.rolling(60, min_periods=1).max()
                sup_h = hc.rolling(60, min_periods=1).min()

                st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday else pd.Series(index=hc.index, dtype=float)

                # Ichimoku Kijun on price chart (solid black)
                kijun_h = pd.Series(index=hc.index, dtype=float)
                if {'High','Low','Close'}.issubset(intraday.columns) and show_ichi:
                    _, kijun_h, _, _, _ = ichimoku_lines(
                        intraday["High"], intraday["Low"], intraday["Close"],
                        conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                    )
                    kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

                yhat_h, m_h = slope_line(hc, slope_lb_hourly)

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)

                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black",
                             label=f"Ichimoku Kijun ({ichi_base})")

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val  = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                buy_sell_text = ""
                if np.isfinite(sup_val): buy_sell_text += f" — ▲ BUY @{fmt_price_val(sup_val)}"
                if np.isfinite(res_val): buy_sell_text += f"  ▼ SELL @{fmt_price_val(res_val)}"
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text}")

                if np.isfinite(px_val):
                    pos = ax2.get_position()
                    fig2.text(pos.x1, pos.y1 + 0.02, f"Current price: {fmt_price_val(px_val)}",
                              ha="right", va="bottom", fontsize=11, fontweight="bold")

                if not st_line_intr.dropna().empty:
                    ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center")

                # ---- RIGHT-SIDE REFRESH BUTTON (Tab 1 INTRADAY) ----
                left_col, right_col = st.columns([0.92, 0.08])
                with left_col:
                    st.pyplot(fig2)
                with right_col:
                    st.markdown("### ")  # top spacer
                    if st.button("🔄 Refresh", key="refresh_hourly_tab1"):
                        st.session_state.intraday_nonce += 1
                        st.session_state.intraday = fetch_intraday(
                            sel, period=period_map[hour_range], nonce=st.session_state.intraday_nonce
                        )
                        st.session_state.last_refresh = time.time()
                        st.rerun()
                    st.caption(f"Updated {int(time.time()-st.session_state.last_refresh)}s ago")

                signal = sr_proximity_signal(hc, res_h, sup_h, st.session_state.fc_vals,
                                             threshold=signal_threshold, prox=sr_prox_pct)
                if signal is not None and np.isfinite(px_val):
                    if signal["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal['level'])} — {signal['reason']}")
                    elif signal["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal['level'])} — {signal['reason']}")

                xlim_price = ax2.get_xlim()

                # === Normalized RSI Panel (Hourly) ===
                if show_nrsi:
                    nrsi_h = compute_nrsi(hc, period=nrsi_period)
                    nmacd_h, nmacd_sig_h, _ = compute_nmacd(hc)
                    nvol_h = compute_nvol(intraday.get("Volume", pd.Series(index=hc.index)).reindex(hc.index))
                    ntd_h_rsipanel = compute_normalized_trend(hc, window=ntd_window)

                    fig2r, ax2r = plt.subplots(figsize=(14,2.8))
                    ax2r.set_title(f"NRSI (p={nrsi_period}) + NVol + NMACD (+signal) + NTD")

                    posv = nvol_h.where(nvol_h > 0)
                    negv = nvol_h.where(nvol_h < 0)
                    ax2r.fill_between(posv.index, 0, posv, alpha=0.10, step=None, label="NVol(+)")
                    ax2r.fill_between(negv.index, 0, negv, alpha=0.10, step=None, label="NVol(-)")

                    ax2r.plot(nrsi_h.index, nrsi_h, "-", linewidth=1.4, label="NRSI")
                    ax2r.plot(nmacd_h.index, nmacd_h, "-", linewidth=1.4, label="NMACD")
                    ax2r.plot(nmacd_sig_h.index, nmacd_sig_h, "--", linewidth=1.2, label="NMACD signal")

                    if show_ntd and not ntd_h_rsipanel.dropna().empty:
                        ax2r.plot(ntd_h_rsipanel.index, ntd_h_rsipanel, ":", linewidth=1.6,
                                  label=f"NTD (win={ntd_window})")
                        last_ntd = float(ntd_h_rsipanel.dropna().iloc[-1])
                        t_dir = "UP" if last_ntd >= 0 else "DOWN"
                        color = "tab:green" if last_ntd >= 0 else "tab:red"
                        certainty = int(round(50 + 50*abs(last_ntd)))
                        ax2r.text(0.99, 0.92, f"Trend: {t_dir} ({certainty}%)",
                                  transform=ax2r.transAxes, ha="right", va="top",
                                  fontsize=10, fontweight="bold", color=color,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

                    ax2r.axhline(0, linestyle="--", linewidth=1, color="black", label="0")
                    ax2r.axhline(0.5, linestyle="-", linewidth=1.2, color="black", label="+0.5")
                    ax2r.axhline(-0.5, linestyle="-", linewidth=1.2, color="black", label="-0.5")
                    ax2r.axhline(0.75, linestyle="-", linewidth=3.0, color="red", label="+0.75")
                    ax2r.axhline(-0.75, linestyle="-", linewidth=3.0, color="red", label="-0.75")

                    ax2r.set_ylim(-1.1, 1.1)
                    ax2r.set_xlim(xlim_price)
                    ax2r.legend(loc="lower left", framealpha=0.5)
                    ax2r.set_xlabel("Time (PST)")
                    st.pyplot(fig2r)

                # Momentum panel (ROC%)
                if show_mom_hourly:
                    roc = compute_roc(hc, n=mom_lb_hourly)
                    res_m = roc.rolling(60, min_periods=1).max()
                    sup_m = roc.rolling(60, min_periods=1).min()
                    fig2m, ax2m = plt.subplots(figsize=(14,2.8))
                    ax2m.set_title(f"Momentum (ROC% over {mom_lb_hourly} bars)")
                    ax2m.plot(roc.index, roc, label=f"ROC%({mom_lb_hourly})")
                    yhat_m, m_m = slope_line(roc, slope_lb_hourly)
                    if not yhat_m.empty:
                        ax2m.plot(yhat_m.index, yhat_m.values, "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m)}%/bar)")
                    ax2m.plot(res_m.index, res_m, ":", label="Mom Resistance")
                    ax2m.plot(sup_m.index, sup_m, ":", label="Mom Support")
                    ax2m.axhline(0, linestyle="--", linewidth=1)
                    ax2m.set_xlabel("Time (PST)")
                    ax2m.legend(loc="lower left", framealpha=0.5)
                    ax2m.set_xlim(xlim_price)
                    st.pyplot(fig2m)

        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}** (change in 'Original Forecast' tab and rerun)")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        # ----- Daily -----
        if view in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            wave_norm_d2, piv_df_d2 = compute_normalized_elliott_wave(df, pivot_lb=pivot_lookback_d, norm_win=norm_window_d)
            npo_d2 = compute_npo(df, fast=npo_fast, slow=npo_slow, norm_win=npo_norm_win) if show_npo else pd.Series(index=df.index, dtype=float)
            ntd_d2 = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            ichi_d2 = pd.Series(index=df.index, dtype=float)
            kijun_d2 = pd.Series(index=df.index, dtype=float)

            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                ichi_d2 = compute_normalized_ichimoku(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    norm_win=ichi_norm_win, price_weight=ichi_price_weight
                )
                _, kijun_d2, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                )
                kijun_d2 = kijun_d2.ffill().bfill()

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            wave_d_show = wave_norm_d2.reindex(df_show.index)
            npo_d_show  = npo_d2.reindex(df_show.index)
            ntd_d_show  = ntd_d2.reindex(df_show.index)
            ichi_d2_show = ichi_d2.reindex(df_show.index).ffill().bfill()
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()
            piv_df_d_show = piv_df_d2[(piv_df_d2["time"] >= df_show.index.min()) & (piv_df_d2["time"] <= df_show.index.max())] if not piv_df_d2.empty else piv_df_d2

            fig, (ax, axdw2) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            ax.set_title(f"{st.session_state.ticker} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")
            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(kijun_d2_show.index, kijun_d2_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema_show.empty:
                ax.plot(yhat_ema_show.index, yhat_ema_show.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            if len(res30_show) and len(sup30_show):
                r30_last = float(res30_show.iloc[-1]); s30_last = float(sup30_show.iloc[-1])
                ax.text(df_show.index[-1], r30_last, f"  30R = {fmt_price_val(r30_last)}", va="bottom")
                ax.text(df_show.index[-1], s30_last, f"  30S = {fmt_price_val(s30_last)}", va="top")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)

            axdw2.set_title("Daily Normalized Elliott Wave + NPO + NTD + Ichimoku (normalized)")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw2, ntd_d_show)
            if show_npo and not npo_d_show.dropna().empty:
                shade_npo_regions(axdw2, npo_d_show)

            axdw2.plot(wave_d_show.index, wave_d_show, label="Norm EW (Daily)", linewidth=1.8)
            if show_npo and not npo_d_show.dropna().empty:
                axdw2.plot(npo_d_show.index, npo_d_show, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw2.plot(ntd_d_show.index, ntd_d_show, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")
            if show_ichi and not ichi_d2_show.dropna().empty:
                axdw2.plot(ichi_d2_show.index, ichi_d2_show.values, "-", linewidth=1.4, color="black",
                           label=f"IchimokuN (c{ichi_conv},b{ichi_base},sb{ichi_spanb})")

            for yline, style, col, lbl in [
                (0.0, "--", None, "EW 0"),
                (0.5, "-", "tab:red", "EW +0.5"),
                (-0.5, "-", "tab:green", "EW -0.5"),
                (0.75, "-", "black", "EW +0.75"),
                (-0.75, "-", "black", "EW -0.75"),
            ]:
                axdw2.axhline(yline, linestyle=style, linewidth=1, color=col, label=lbl)

            axdw2.set_ylim(-1.1, 1.1)
            axdw2.set_xlabel("Date (PST)")

            if not piv_df_d_show.empty:
                show_df_d2 = piv_df_d_show.tail(int(waves_to_annotate_d))
                for _, r in show_df_d2.iterrows():
                    t = r["time"]; w = r["wave"]; typ = r["type"]
                    ylab = 0.9 if typ == 'H' else -0.9
                    axdw2.annotate(str(int(w)), (t, ylab),
                                   xytext=(0, 0), textcoords="offset points",
                                   ha="center", va="center",
                                   fontsize=9, fontweight="bold")

            px_daily2 = _safe_last_float(df)
            ew_sig_d2 = elliott_conf_signal(px_daily2, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw2 = axdw2.get_position()
            label_txt_d2 = f"Price: {fmt_price_val(px_daily2)}"
            if ew_sig_d2 is not None:
                side = ew_sig_d2['side']; prob = fmt_pct(ew_sig_d2['prob'], digits=0)
                label_txt_d2 += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily2)}  ({prob})"
            fig.text(posdw2.x1, posdw2.y1 + 0.01, label_txt_d2, ha="right", va="bottom",
                     fontsize=10, fontweight="bold")

            axdw2.legend(loc="lower left", framealpha=0.5)

            # ---- RIGHT-SIDE REFRESH BUTTON (Tab 2 DAILY) ----
            dl2, dr2 = st.columns([0.92, 0.08])
            with dl2:
                st.pyplot(fig)
            with dr2:
                st.markdown("### ")
                if st.button("🔄 Refresh", key="refresh_daily_tab2"):
                    st.session_state.daily_nonce += 1
                    st.session_state.df_hist = fetch_hist(st.session_state.ticker, nonce=st.session_state.daily_nonce)
                    st.session_state.df_ohlc = fetch_hist_ohlc(st.session_state.ticker, nonce=st.session_state.daily_nonce)
                    idx2, vals2, ci2 = compute_sarimax_forecast(st.session_state.df_hist)
                    st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci = idx2, vals2, ci2
                    st.session_state.last_refresh = time.time()
                    st.rerun()
                st.caption(f"Updated {int(time.time()-st.session_state.last_refresh)}s ago")

        # ----- Intraday (Hourly) -----
        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                ic = intr["Close"].ffill()
                ie = ic.ewm(span=20).mean()
                xi = np.arange(len(ic))
                slope_i, intercept_i = np.polyfit(xi, ic.values, 1)
                trend_i = slope_i * xi + intercept_i
                res_i = ic.rolling(60, min_periods=1).max()
                sup_i = ic.rolling(60, min_periods=1).min()
                st_intraday = compute_supertrend(intr, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(ic.index) if "ST" in st_intraday else pd.Series(index=ic.index, dtype=float)

                kijun_i = pd.Series(index=ic.index, dtype=float)
                if {'High','Low','Close'}.issubset(intr.columns) and show_ichi:
                    _, kijun_i, _, _, _ = ichimoku_lines(
                        intr["High"], intr["Low"], intr["Close"],
                        conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                    )
                    kijun_i = kijun_i.reindex(ic.index).ffill().bfill()

                yhat_h, m_h = slope_line(ic, slope_lb_hourly)

                fig3, ax3 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax3.plot(ic.index, ic, label="Intraday")
                ax3.plot(ic.index, ie, "--", label="20 EMA")
                ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)
                if show_ichi and not kijun_i.dropna().empty:
                    ax3.plot(kijun_i.index, kijun_i.values, "-", linewidth=1.8, color="black",
                             label=f"Ichimoku Kijun ({ichi_base})")

                res_val2 = sup_val2 = px_val2 = np.nan
                try:
                    res_val2 = float(res_i.iloc[-1]); sup_val2 = float(sup_i.iloc[-1]); px_val2 = float(ic.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val2) and np.isfinite(sup_val2):
                    ax3.hlines(res_val2, xmin=ic.index[0], xmax=ic.index[-1],
                               colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax3.hlines(sup_val2, xmin=ic.index[0], xmax=ic.index[-1],
                               colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax3, res_val2, f"R {fmt_price_val(res_val2)}", color="tab:red")
                    label_on_left(ax3, sup_val2, f"S {fmt_price_val(sup_val2)}", color="tab:green")

                buy_sell_text2 = ""
                if np.isfinite(sup_val2): buy_sell_text2 += f" — ▲ BUY @{fmt_price_val(sup_val2)}"
                if np.isfinite(res_val2): buy_sell_text2 += f"  ▼ SELL @{fmt_price_val(res_val2)}"
                ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text2}")

                if np.isfinite(px_val2):
                    pos2 = ax3.get_position()
                    fig3.text(pos2.x1, pos2.y1 + 0.02, f"Current price: {fmt_price_val(px_val2)}",
                              ha="right", va="bottom", fontsize=11, fontweight="bold")

                if not st_line_intr.dropna().empty:
                    ax3.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax3.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                if show_fibs and not ic.empty:
                    fibs_h = fibonacci_levels(ic)
                    for lbl, y in fibs_h.items():
                        ax3.hlines(y, xmin=ic.index[0], xmax=ic.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax3.text(ic.index[-1], y, f" {lbl}", va="center")

                # ---- RIGHT-SIDE REFRESH BUTTON (Tab 2 INTRADAY) ----
                left2, right2 = st.columns([0.92, 0.08])
                with left2:
                    st.pyplot(fig3)
                with right2:
                    st.markdown("### ")
                    if st.button("🔄 Refresh", key="refresh_hourly_tab2"):
                        st.session_state.intraday_nonce += 1
                        st.session_state.intraday = fetch_intraday(
                            st.session_state.ticker,
                            period=period_map[st.session_state.hour_range],
                            nonce=st.session_state.intraday_nonce
                        )
                        st.session_state.last_refresh = time.time()
                        st.rerun()
                    st.caption(f"Updated {int(time.time()-st.session_state.last_refresh)}s ago")

                signal2 = sr_proximity_signal(ic, res_i, sup_i, st.session_state.fc_vals,
                                              threshold=signal_threshold, prox=sr_prox_pct)
                if signal2 is not None and np.isfinite(px_val2):
                    if signal2["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")
                    elif signal2["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")

                xlim_price2 = ax3.get_xlim()

                # === NRSI + NMACD(+signal) + NVol + NTD (Intraday view) ===
                if show_nrsi:
                    nrsi_i = compute_nrsi(ic, period=nrsi_period)
                    nmacd_i, nmacd_sig_i, _ = compute_nmacd(ic)
                    nvol_i = compute_nvol(intr.get("Volume", pd.Series(index=ic.index)).reindex(ic.index))
                    ntd_i_rsipanel = compute_normalized_trend(ic, window=ntd_window)

                    fig3r, ax3r = plt.subplots(figsize=(14,2.8))
                    ax3r.set_title(f"NRSI (p={nrsi_period}) + NVol + NMACD (+signal) + NTD")
                    posv = nvol_i.where(nvol_i > 0)
                    negv = nvol_i.where(nvol_i < 0)
                    ax3r.fill_between(posv.index, 0, posv, alpha=0.10, step=None, label="NVol(+)")
                    ax3r.fill_between(negv.index, 0, negv, alpha=0.10, label="NVol(-)")
                    ax3r.plot(nrsi_i.index, nrsi_i, "-", linewidth=1.4, label="NRSI")
                    ax3r.plot(nmacd_i.index, nmacd_i, "-", linewidth=1.4, label="NMACD")
                    ax3r.plot(nmacd_sig_i.index, nmacd_sig_i, "--", linewidth=1.2, label="NMACD signal")

                    if show_ntd and not ntd_i_rsipanel.dropna().empty:
                        ax3r.plot(ntd_i_rsipanel.index, ntd_i_rsipanel, ":", linewidth=1.6,
                                  label=f"NTD (win={ntd_window})")
                        last_ntd = float(ntd_i_rsipanel.dropna().iloc[-1])
                        t_dir = "UP" if last_ntd >= 0 else "DOWN"
                        color = "tab:green" if last_ntd >= 0 else "tab:red"
                        certainty = int(round(50 + 50*abs(last_ntd)))
                        ax3r.text(0.99, 0.92, f"Trend: {t_dir} ({certainty}%)",
                                  transform=ax3r.transAxes, ha="right", va="top",
                                  fontsize=10, fontweight="bold", color=color,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

                    ax3r.axhline(0, linestyle="--", linewidth=1, color="black", label="0")
                    ax3r.axhline(0.5, linestyle="-", linewidth=1.2, color="black", label="+0.5")
                    ax3r.axhline(-0.5, linestyle="-", linewidth=1.2, color="black", label="-0.5")
                    ax3r.axhline(0.75, linestyle="-", linewidth=3.0, color="red", label="+0.75")
                    ax3r.axhline(-0.75, linestyle="-", linewidth=3.0, color="red", label="-0.75")
                    ax3r.set_ylim(-1.1, 1.1)
                    ax3r.set_xlim(xlim_price2)
                    ax3r.legend(loc="lower left", framealpha=0.5)
                    ax3r.set_xlabel("Time (PST)")
                    st.pyplot(fig3r)

                # Momentum panel (ROC%)
                if show_mom_hourly:
                    roc_i = compute_roc(ic, n=mom_lb_hourly)
                    res_m2 = roc_i.rolling(60, min_periods=1).max()
                    sup_m2 = roc_i.rolling(60, min_periods=1).min()
                    fig3m, ax3m = plt.subplots(figsize=(14,2.8))
                    ax3m.set_title(f"Momentum (ROC% over {mom_lb_hourly} bars)")
                    ax3m.plot(roc_i.index, roc_i, label=f"ROC%({mom_lb_hourly})")
                    yhat_m2, m_m2 = slope_line(roc_i, slope_lb_hourly)
                    if not yhat_m2.empty:
                        ax3m.plot(yhat_m2.index, yhat_m2.values, "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m2)}%/bar)")
                    ax3m.plot(res_m2.index, res_m2, ":", label="Mom Resistance")
                    ax3m.plot(sup_m2.index, sup_m2, ":", label="Mom Support")
                    ax3m.axhline(0, linestyle="--", linewidth=1)
                    ax3m.set_xlabel("Time (PST)")
                    ax3m.legend(loc="lower left", framealpha=0.5)
                    ax3m.set_xlim(xlim_price2)
                    st.pyplot(fig3m)

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange'] > 0
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last_price = _safe_last_float(df_hist)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        x3m = np.arange(len(df3m))
        slope3m, intercept3m = np.polyfit(x3m, df3m.values, 1)
        trend3m = slope3m * x3m + intercept3m

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m, ":", label="Support")
        ax.plot(df3m.index, trend3m, "--", label="Trend")
        ax.set_xlabel("Date (PST)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        x0 = np.arange(len(df0))
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0 * x0 + intercept0
        res0 = df0.rolling(30, min_periods=1).max()
        sup0 = df0.rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(res0, ":", label="Resistance")
        ax0.plot(sup0, ":", label="Support")
        ax0.plot(df0.index, trend0, "--", label="Trend")
        ax0.set_xlabel("Date (PST)")
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.5 Scanner (Stocks & Forex) ---
with tab5:
    st.header("NTD -0.5 Scanner")
    st.caption("Shows **symbols with Normalized Trend Direction (NTD) < -0.5**. Daily for both Stocks & Forex; plus optional **Hourly** for Forex.")

    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox(
        "Hourly lookback for Forex (for Hourly scan below):",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
        key="ntd_scan_hour_range"
    )
    scan_period = period_map[scan_hour_range]

    c1, c2 = st.columns(2)
    with c1:
        thresh = st.slider("NTD threshold", -1.0, 0.0, -0.5, 0.05, key="ntd_thresh")
    with c2:
        run = st.button("Scan Universe", key="btn_ntd_scan")

    if run:
        # DAILY scan for both universes
        daily_rows = []
        for sym in universe:
            ntd_val, ts = last_daily_ntd_value(sym, ntd_window)
            daily_rows.append({"Symbol": sym, "NTD_Daily": ntd_val, "Timestamp": ts})
        df_daily = pd.DataFrame(daily_rows)
        below_daily = df_daily[np.isfinite(df_daily["NTD_Daily"]) & (df_daily["NTD_Daily"] < thresh)].sort_values("NTD_Daily")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(f"Daily NTD < {thresh:+.2f}", int(below_daily.shape[0]))

        st.subheader(f"Daily — NTD < {thresh:+.2f}")
        if below_daily.empty:
            st.info(f"No symbols with Daily NTD < {thresh:+.2f}.")
        else:
            show = below_daily.copy()
            show["NTD_Daily"] = show["NTD_Daily"].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "n/a")
            st.dataframe(show.reset_index(drop=True), use_container_width=True)

        # HOURLY scan for Forex only
        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — NTD < {thresh:+.2f}  ({scan_hour_range} lookback)")
            hourly_rows = []
            for sym in universe:
                v, ts = last_hourly_ntd_value(sym, ntd_window, period=scan_period)
                hourly_rows.append({"Symbol": sym, "NTD_Hourly": v, "Timestamp": ts})
            df_hour = pd.DataFrame(hourly_rows)
            below_hour = df_hour[np.isfinite(df_hour["NTD_Hourly"]) & (df_hour["NTD_Hourly"] < thresh)].sort_values("NTD_Hourly")

            c5, c6 = st.columns(2)
            c5.metric("FX Pairs Scanned", len(universe))
            c6.metric(f"Hourly NTD < {thresh:+.2f}", int(below_hour.shape[0]))

            if below_hour.empty:
                st.info(f"No Forex pairs with Hourly NTD < {thresh:+.2f}.")
            else:
                showh = below_hour.copy()
                showh["NTD_Hourly"] = showh["NTD_Hourly"].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "n/a")
                st.dataframe(showh.reset_index(drop=True), use_container_width=True)

# --- Tab 6: 20Y History ---
with tab6:
    st.header("20-Year Historical Prices")
    st.caption("View **20 years** of daily close prices for the selected ticker (Stock or Forex).")

    # default to last selected ticker if available, else first in universe
    default_symbol = st.session_state.get("ticker") or universe[0]
    sym_20 = st.selectbox("Ticker:", universe, index=universe.index(default_symbol) if default_symbol in universe else 0, key="long_hist_ticker")

    # Optional: simple controls
    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    with c1:
        logscale = st.checkbox("Log scale", value=False, key="long_log")
    with c2:
        smooth = st.checkbox("Add 200-day MA", value=True, key="long_ma")
    with c3:
        refresh = st.button("🔄 Refresh 20Y", key="btn_refresh_20y")

    if refresh:
        st.session_state.long_nonce += 1

    s20 = fetch_hist_20y(sym_20, nonce=st.session_state.long_nonce)

    if s20 is None or s20.empty:
        st.warning("No long-term data available for this symbol.")
    else:
        ma200 = s20.rolling(200, min_periods=1).mean() if smooth else None

        figl, axl = plt.subplots(figsize=(14,5))
        title = f"{sym_20} — 20Y Daily Close"
        axl.set_title(title)
        axl.plot(s20.index, s20.values, label="Close", linewidth=1.2)
        if ma200 is not None:
            axl.plot(ma200.index, ma200.values, "--", label="200D MA", linewidth=1.2)
        if logscale:
            axl.set_yscale("log")
        axl.set_xlabel("Date (PST)")
        axl.set_ylabel("Price")
        axl.legend(loc="lower left", framealpha=0.5)
        st.pyplot(figl)

        st.caption(f"Data points: {len(s20):,} | First: {s20.index.min().strftime('%Y-%m-%d')} | Last: {s20.index.max().strftime('%Y-%m-%d')}")
