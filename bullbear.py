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
# - UPDATED: Price chart shows trendline slope in legend + bottom-left; NRSI panel adds trendline + slope; NRSI Trend badge moved to bottom-right
# - NEW: Scanner lists **Price > Kijun(26)** symbols (Daily for Stocks & FX; Hourly for FX)
# - NEW: Hourly price chart badge also shows **R²** of the trendline (full lookback used on the chart)

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
    """Robust slope formatter (handles numpy scalars/arrays)."""
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.4f}" if np.isfinite(mv) else "n/a"

def fmt_r2(r2: float, digits: int = 3) -> str:
    try:
        v = float(r2)
    except Exception:
        return "n/a"
    return f"{v:.{digits}f}" if np.isfinite(v) else "n/a"

# Compute R² between a true series and a fitted/predicted series
def compute_r2(y_true, y_hat) -> float:
    y = _coerce_1d_series(y_true).astype(float)
    yh = _coerce_1d_series(y_hat).astype(float)
    df = pd.concat([y, yh], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    yt = df.iloc[:,0].to_numpy()
    yp = df.iloc[:,1].to_numpy()
    ss_res = float(np.sum((yt - yp)**2))
    ss_tot = float(np.sum((yt - yt.mean())**2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res/ss_tot)

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
def fetch_hist(ticker: str) -> pd.Series:
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
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[['Open','High','Low','Close']].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
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
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
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

# ---- Price > Kijun detection (Daily & Hourly) ----
def _price_above_kijun_from_df(df: pd.DataFrame, base: int = 26):
    """Returns (above_now, timestamp, close_now, kijun_now) for the latest bar."""
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return False, None, np.nan, np.nan
    ohlc = df[['High','Low','Close']].copy()
    _, kijun, _, _, _ = ichimoku_lines(ohlc['High'], ohlc['Low'], ohlc['Close'], base=base)
    kijun = kijun.ffill().bfill().reindex(ohlc.index)
    close = ohlc['Close'].astype(float).reindex(ohlc.index)

    mask = close.notna() & kijun.notna()
    if mask.sum() < 1:
        return False, None, np.nan, np.nan
    c_now = float(close[mask].iloc[-1])
    k_now = float(kijun[mask].iloc[-1])
    ts = close[mask].index[-1]
    above = np.isfinite(c_now) and np.isfinite(k_now) and (c_now > k_now)
    return above, ts if above else None, c_now, k_now

@st.cache_data(ttl=120)
def price_above_kijun_info_daily(symbol: str, base: int = 26):
    try:
        df = fetch_hist_ohlc(symbol)
        return _price_above_kijun_from_df(df, base=base)
    except Exception:
        return False, None, np.nan, np.nan

@st.cache_data(ttl=120)
def price_above_kijun_info_hourly(symbol: str, period: str = "1d", base: int = 26):
    try:
        df = fetch_intraday(symbol, period=period)
        return _price_above_kijun_from_df(df, base=base)
    except Exception:
        return False, None, np.nan, np.nan

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.5 Scanner"
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
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
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
            plt.subplots_adjust(hspace=0.05, top=0.92, right
