# bullbear.py — Stocks/Forex Dashboard + Forecasts
# (UPDATED) London & New York session Open/Close markers in PST on Forex intraday charts.

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

def fmt_r2(r2: float, digits: int = 1) -> str:
    """Format R² as a percentage string (coefficient of determination)."""
    try:
        rv = float(r2)
    except Exception:
        return "n/a"
    return fmt_pct(rv, digits=digits) if np.isfinite(rv) else "n/a"

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

# Hourly Normalized RSI controls (kept for layout; panel now shows only NTD + Trend)
st.sidebar.subheader("Hourly Normalized RSI (panel now shows NTD + Trend only)")
show_nrsi   = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

# Hourly Supertrend controls
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# NEW: Parabolic SAR controls
st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

# Signal logic controls
st.sidebar.subheader("Signal Logic (Hourly)")
signal_threshold = st.sidebar.slider("Signal confidence threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# Daily indicator panel controls
st.sidebar.subheader("NTD (Daily Indicator Panel)")
pivot_lookback_d = st.sidebar.slider("Pivot lookback (unused)", 3, 31, 9, 2, key="sb_pivot_lb_d")
norm_window_d    = st.sidebar.slider("Normalization window (unused)", 30, 1200, 360, 10, key="sb_norm_win_d")
waves_to_annotate_d = st.sidebar.slider("Annotate recent waves (unused)", 3, 12, 7, 1, key="sb_wave_ann_d")

# NPO overlay controls (unused now; kept for compatibility)
st.sidebar.subheader("Normalized Price Oscillator (unused on indicator panels)")
show_npo    = st.sidebar.checkbox("Show NPO overlay (unused)", value=False, key="sb_show_npo")
npo_fast    = st.sidebar.slider("NPO fast EMA", 5, 30, 12, 1, key="sb_npo_fast")
npo_slow    = st.sidebar.slider("NPO slow EMA", 10, 60, 26, 1, key="sb_npo_slow")
npo_norm_win= st.sidebar.slider("NPO normalization window", 30, 600, 240, 10, key="sb_npo_norm")

# NTD overlay controls (for NTD panels)
st.sidebar.subheader("Normalized Trend (NTD panels — Daily & Hourly)")
show_ntd  = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window= st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (Daily & Hourly: green=up, red=down)", value=True, key="sb_ntd_shade")

# Ichimoku controls (Normalized for EW panels + Kijun on price) — only Kijun used on price charts
st.sidebar.subheader("Normalized Ichimoku (EW panels) + Kijun on price")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")
ichi_norm_win = st.sidebar.slider("Ichimoku normalization window (unused)", 30, 600, 240, 10, key="sb_ichi_norm")
ichi_price_weight = st.sidebar.slider("Weight: Price vs Cloud (unused)", 0.0, 1.0, 0.6, 0.05, key="sb_ichi_w")

# NEW: Bollinger Bands controls
st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

# NEW: Probabilistic HMA Crossover controls
st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

# Forex-only controls
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
    # NEW: Session markers toggle
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_fx_news = False
    news_window_days = 7
    show_sessions_pst = False

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
def fetch_hist_max(ticker: str) -> pd.Series:
    """Full history (Yahoo 'max'), daily freq with ffill, tz-aware."""
    df = yf.download(ticker, period="max")[['Close']].dropna()
    s = df['Close'].asfreq("D").fillna(method="ffill")
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

def regression_r2(series_like, lookback: int):
    """R² for a simple linear fit over the last `lookback` bars of `series_like`."""
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
    if s.shape[0] < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res/ss_tot)

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

# ---- Normalized Trend Direction (for NTD panels) ----
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
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

# Red shading under NPO curve (unused now)
def shade_npo_regions(ax, npo: pd.Series):
    return

# Daily trend-direction line helper (PRICE chart)
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

# --- Parabolic SAR (NEW) ---
def compute_parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2):
    """Return PSAR series and a boolean series in_uptrend (True when dots below price)."""
    H = _coerce_1d_series(high).astype(float)
    L = _coerce_1d_series(low).astype(float)
    df = pd.concat([H.rename("H"), L.rename("L")], axis=1).dropna()
    if df.empty:
        idx = H.index if len(H) else L.index
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=bool)

    n = len(df)
    psar = np.zeros(n) * np.nan
    up = np.zeros(n, dtype=bool)

    # initialize
    uptrend = True
    af = float(step)
    ep = df["H"].iloc[0]     # extreme point
    psar[0] = df["L"].iloc[0]
    up[0] = True

    for i in range(1, n):
        prev_psar = psar[i-1]
        if uptrend:
            psar[i] = prev_psar + af * (ep - prev_psar)
            # limit
            lo1 = df["L"].iloc[i-1]
            lo2 = df["L"].iloc[i-2] if i >= 2 else lo1
            psar[i] = min(psar[i], lo1, lo2)
            # update EP/AF
            if df["H"].iloc[i] > ep:
                ep = df["H"].iloc[i]
                af = min(af + step, max_step)
            # flip?
            if df["L"].iloc[i] < psar[i]:
                uptrend = False
                psar[i] = ep  # on flip, PSAR equals EP
                ep = df["L"].iloc[i]
                af = step
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            hi1 = df["H"].iloc[i-1]
            hi2 = df["H"].iloc[i-2] if i >= 2 else hi1
            psar[i] = max(psar[i], hi1, hi2)
            if df["L"].iloc[i] < ep:
                ep = df["L"].iloc[i]
                af = min(af + step, max_step)
            if df["H"].iloc[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = df["H"].iloc[i]
                af = step

        up[i] = uptrend

    psar_s = pd.Series(psar, index=df.index, name="PSAR")
    up_s = pd.Series(up, index=df.index, name="in_uptrend")
    return psar_s, up_s

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["PSAR","in_uptrend"])
    ps, up = compute_parabolic_sar(df["High"], df["Low"], step=step, max_step=max_step)
    return pd.DataFrame({"PSAR": ps, "in_uptrend": up})

# ---- Normalized Elliott Wave (kept but unused) ----
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
    return wave_norm, pd.DataFrame(columns=["time","price","type","wave"])

# ---- Ichimoku (classic) ----
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
    # Kept for compatibility; not used on indicator panels
    C = _coerce_1d_series(close).astype(float)
    return pd.Series(index=C.index, dtype=float)

# ---- Bollinger Bands (SMA/EMA midline) + normalized %B (NBB) ----
def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    """
    Returns:
      mid, upper, lower, pctB (0..1), nbb (-1..1)
    """
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 2 or not np.isfinite(mult):
        idx = s.index
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty, empty

    minp = max(2, window // 2)
    mid = s.ewm(span=window, adjust=False).mean() if use_ema else s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std

    width = (upper - lower).replace(0, np.nan)
    pctb = ((s - lower) / width).clip(0.0, 1.0)
    nbb = pctb * 2.0 - 1.0
    return mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index)

# ---- HMA (Hull Moving Average) + crossover helpers (NEW) ----
def _wma(s: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(s).astype(float)
    if s.empty or window < 1:
        return pd.Series(index=s.index, dtype=float)
    w = np.arange(1, window + 1, dtype=float)
    return s.rolling(window, min_periods=window).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)
    half = max(1, int(period / 2))
    sqrtp = max(1, int(np.sqrt(period)))
    wma_half = _wma(s, half)
    wma_full = _wma(s, period)
    diff = 2 * wma_half - wma_full
    hma = _wma(diff, sqrtp)
    return hma.reindex(s.index)

def detect_last_crossover(price: pd.Series, line: pd.Series):
    """Return {'time': ts, 'side': 'BUY'|'SELL'} for the most-recent price↔line crossover (if any)."""
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    mask = p.notna() & l.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]; l = l[mask]
    above = p > l
    cross_up  = above & (~above.shift(1).fillna(False))
    cross_dn  = (~above) & (above.shift(1).fillna(False))
    t_up = cross_up[cross_up].index[-1] if cross_up.any() else None
    t_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None
    if t_up is None and t_dn is None:
        return None
    if t_dn is None or (t_up is not None and t_up > t_dn):
        return {"time": t_up, "side": "BUY"}
    else:
        return {"time": t_dn, "side": "SELL"}

def annotate_crossover(ax, ts, px, side: str, conf: float):
    """Put a ▲/▼ marker and a small label at the crossover point."""
    try:
        if side == "BUY":
            ax.scatter([ts], [px], marker="^", s=90, color="tab:green", zorder=7)
            ax.text(ts, px, f"  BUY {int(conf*100)}%", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
        else:
            ax.scatter([ts], [px], marker="v", s=90, color="tab:red", zorder=7)
            ax.text(ts, px, f"  SELL {int(conf*100)}%", va="top", fontsize=9, color="tab:red", fontweight="bold")
    except Exception:
        pass

# ========= Sessions (NEW) =========
NY_TZ   = pytz.timezone("America/New_York")
LDN_TZ  = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    """Return two lists (open_times_pst, close_times_pst) within idx range, converted to PST.
    Uses local session dates & pytz to handle DST correctly."""
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    # session-local date range covering the visible data
    start_d = idx[0].astimezone(session_tz).date()
    end_d   = idx[-1].astimezone(session_tz).date()
    # iterate inclusive
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        # localize session open/close at that session-local date
        try:
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            # if ambiguous/nonexistent times, let pytz resolve
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst  = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst  <= hi: opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi: closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    """Collect London & New York session open/close times in PST for the given intraday index."""
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)  # 8:00–17:00 London local
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ,  8, 17)  # 8:00–17:00 New York local
    return {
        "ldn_open": ldn_open, "ldn_close": ldn_close,
        "ny_open": ny_open,   "ny_close": ny_close
    }

def draw_session_lines(ax, lines: dict):
    """Draw vertical lines for session opens/closes with legend entries. Solid=open, dashed=close."""
    # Legend handles (one each)
    ax.plot([], [], linestyle="-",  color="tab:blue",   label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",   label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
    # Lines (light alpha)
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:orange", alpha=0.35)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=0.35)
    # Why: keeps the viewer aware that all session markers are PST
    ax.text(0.99, 0.98, "Session times in PST",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))

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

# --- Volume helpers (new) ---
def rolling_midline(series_like: pd.Series, window: int) -> pd.Series:
    """Mid-line as (rolling max + rolling min)/2 over `window`."""
    s = _coerce_1d_series(series_like).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll = s.rolling(window, min_periods=1)
    mid = (roll.max() + roll.min()) / 2.0
    return mid.reindex(s.index)

def _has_volume_to_plot(vol: pd.Series) -> bool:
    """Robust scalar check to decide if volume panel should render."""
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10  # default for Long-Term History tab

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.5 Scanner",
    "Long-Term History"
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

        # ----- Daily (Price + Indicator panel with NTD+Trend) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            # NTD for daily panel
            ntd_d = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            # Kijun for price chart (optional)
            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                )
                kijun_d = kijun_d.ffill().bfill()

            # BBands (Daily)
            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            ntd_d_show  = ntd_d.reindex(df_show.index)
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()

            # BB subset for the shown window
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show= bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            # HMA (Daily) for price chart + probabilistic crossover
            hma_d_full = compute_hma(df, period=hma_period) if show_hma else pd.Series(index=df.index, dtype=float)
            hma_d_show = hma_d_full.reindex(df_show.index)

            # PSAR (Daily)
            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            # --- PRICE CHART ---
            ax.set_title(f"{sel} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            # HMA line
            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            # Kijun (Daily)
            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            # Bollinger Bands (Daily)
            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)
                # NBB badge (bottom-right)
                try:
                    last_pct = float(bb_pctb_d_show.dropna().iloc[-1])
                    last_nbb = float(bb_nbb_d_show.dropna().iloc[-1])
                    ax.text(0.99, 0.02, f"NBB {last_nbb:+.2f}  |  %B {fmt_pct(last_pct, digits=0)}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=9, color="black",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                except Exception:
                    pass

            # PSAR overlay (Daily)
            if show_psar and not psar_d_df.empty:
                up_mask = psar_d_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask],
                               s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask],
                               s=15, color="tab:red", zorder=6)

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

            # Probabilistic HMA crossover (Daily)
            if show_hma and not hma_d_show.dropna().empty:
                cross_d = detect_last_crossover(df_show, hma_d_show)
                if cross_d is not None and cross_d["time"] is not None:
                    ts = cross_d["time"]; px_here = float(df_show.loc[ts])
                    if cross_d["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "BUY", hma_conf)
                        st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — last price crossed **up** HMA({hma_period}) with P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)}")
                    elif cross_d["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "SELL", hma_conf)
                        st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — last price crossed **down** HMA({hma_period}) with P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)}")

            # --- DAILY INDICATOR PANEL: NTD + Trend ONLY ---
            axdw.set_title("Daily Indicator Panel — NTD + Trend")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)

            # NTD line
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                # NTD trendline
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                              label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")

            # Reference lines
            axdw.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
            axdw.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
            axdw.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
            axdw.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
            axdw.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")
            axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly (price + NEW NTD panel + momentum + NEW Volume panel) -----
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

                # Bollinger Bands (Hourly)
                bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

                # HMA (Hourly) overlay + probabilistic crossover signal
                hma_h = compute_hma(hc, period=hma_period) if show_hma else pd.Series(index=hc.index, dtype=float)

                # PSAR (Hourly)
                psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                psar_h_df = psar_h_df.reindex(hc.index)

                # Lookback slope & R²
                yhat_h, m_h = slope_line(hc, slope_lb_hourly)
                r2_h = regression_r2(hc, slope_lb_hourly)

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", label=f"Trend (m={fmt_slope(slope_h)}/bar)", linewidth=2)

                # HMA line
                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black",
                             label=f"Ichimoku Kijun ({ichi_base})")

                # Plot BBands (Hourly)
                if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
                    ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                    ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1,
                             label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                    ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
                    ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

                # PSAR overlay (Hourly)
                if show_psar and not psar_h_df.dropna().empty:
                    up_mask = psar_h_df["in_uptrend"] == True
                    dn_mask = ~up_mask
                    if up_mask.any():
                        ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask],
                                    s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                    if dn_mask.any():
                        ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask],
                                    s=15, color="tab:red", zorder=6)

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

                # === Current price badge (bottom-right) ===
                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                    except Exception:
                        pass
                    ax2.text(0.99, 0.02, f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                             transform=ax2.transAxes, ha="right", va="bottom",
                             fontsize=11, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not st_line_intr.dropna().empty:
                    ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                # Bottom-left slope badge
                ax2.text(0.01, 0.02, f"Slope: {fmt_slope(slope_h)}/bar",
                         transform=ax2.transAxes, ha="left", va="bottom",
                         fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                # Bottom-center R² badge
                ax2.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
                         transform=ax2.transAxes, ha="center", va="bottom",
                         fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                # --- NEW: Session lines (PST) for Forex intraday ---
                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess = compute_session_lines(hc.index)
                    draw_session_lines(ax2, sess)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center")

                if mode == "Forex" and show_fx_news and not hc.empty:
                    fx_news = fetch_yf_news(sel, window_days=news_window_days)
                    if not fx_news.empty:
                        t0, t1 = hc.index[0], hc.index[-1]
                        times = [t for t in fx_news["time"] if t0 <= t <= t1]
                        if times:
                            draw_news_markers(ax2, times, float(hc.min()), float(hc.max()), label="News")

                signal = sr_proximity_signal(hc, res_h, sup_h, st.session_state.fc_vals,
                                             threshold=signal_threshold, prox=sr_prox_pct)
                if signal is not None and np.isfinite(px_val):
                    if signal["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal['level'])} — {signal['reason']}")
                    elif signal["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal['level'])} — {signal['reason']}")

                # Probabilistic HMA crossover (Hourly)
                if show_hma and not hma_h.dropna().empty:
                    cross_h = detect_last_crossover(hc, hma_h)
                    if cross_h is not None and cross_h["time"] is not None:
                        ts = cross_h["time"]; px_here = float(hc.loc[ts])
                        if cross_h["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                            annotate_crossover(ax2, ts, px_here, "BUY", hma_conf)
                            st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — price crossed **up** HMA({hma_period}) with P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)}")
                        elif cross_h["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                            annotate_crossover(ax2, ts, px_here, "SELL", hma_conf)
                            st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — price crossed **down** HMA({hma_period}) with P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)}")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                xlim_price = ax2.get_xlim()
                st.pyplot(fig2)

                # === NEW: Hourly Volume panel (separate chart) ===
                vol = _coerce_1d_series(intraday.get("Volume", pd.Series(index=hc.index))).reindex(hc.index).astype(float)
                if _has_volume_to_plot(vol):
                    v_mid = rolling_midline(vol, window=max(3, int(slope_lb_hourly)))
                    v_trend, v_m = slope_line(vol, slope_lb_hourly)
                    v_r2 = regression_r2(vol, slope_lb_hourly)

                    fig2v, ax2v = plt.subplots(figsize=(14, 2.8))
                    ax2v.set_title(f"Volume (Hourly) — Mid-line & Trend  |  Slope={fmt_slope(v_m)}/bar")
                    # BLUE volume area + line
                    ax2v.fill_between(vol.index, 0, vol, alpha=0.18, label="Volume", color="tab:blue")
                    ax2v.plot(vol.index, vol, linewidth=1.0, color="tab:blue")

                    # Rolling mid-line + label
                    ax2v.plot(v_mid.index, v_mid, ":", linewidth=1.6, label=f"Mid-line ({slope_lb_hourly}-roll)")
                    if v_mid.notna().any():
                        last_mid = float(v_mid.dropna().iloc[-1])
                        ax2v.hlines(last_mid, xmin=vol.index[0], xmax=vol.index[-1],
                                    linestyles="dotted", linewidth=1.0)
                        label_on_left(ax2v, last_mid, f"Mid {last_mid:,.0f}", color="black")

                    if not v_trend.empty:
                        ax2v.plot(v_trend.index, v_trend.values, "--", linewidth=2,
                                  label=f"Trend {slope_lb_hourly} ({fmt_slope(v_m)}/bar)")

                    ax2v.text(0.01, 0.02, f"Slope: {fmt_slope(v_m)}/bar",
                              transform=ax2v.transAxes, ha="left", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                    ax2v.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(v_r2)}",
                              transform=ax2v.transAxes, ha="center", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                    ax2v.set_xlim(xlim_price)
                    ax2v.set_xlabel("Time (PST)")
                    ax2v.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2v)

                # === Hourly Indicator Panel: NTD + Trend ONLY (SHADED) ===
                if show_nrsi:
                    ntd_h = compute_normalized_trend(hc, window=ntd_window)
                    ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)

                    fig2r, ax2r = plt.subplots(figsize=(14,2.8))
                    ax2r.set_title(f"Hourly Indicator Panel — NTD + Trend (win={ntd_window})")

                    # NEW: shading (green above 0, red below 0)
                    if shade_ntd and not ntd_h.dropna().empty:
                        shade_ntd_regions(ax2r, ntd_h)

                    # NTD line
                    ax2r.plot(ntd_h.index, ntd_h, "-", linewidth=1.6, label="NTD")

                    # NTD trendline (dashed) + slope in legend
                    if not ntd_trend_h.empty:
                        ax2r.plot(ntd_trend_h.index, ntd_trend_h.values, "--", linewidth=2,
                                  label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")

                    # Trend badge (bottom-center)
                    if not ntd_h.dropna().empty:
                        last_ntd = float(ntd_h.dropna().iloc[-1])
                        t_dir = "UP" if last_ntd >= 0 else "DOWN"
                        color = "tab:green" if last_ntd >= 0 else "tab:red"
                        certainty = int(round(50 + 50*abs(last_ntd)))  # map [-1,1] → [50,100]
                        ax2r.text(0.50, 0.06, f"Trend: {t_dir} ({certainty}%)",
                                  transform=ax2r.transAxes, ha="center", va="bottom",
                                  fontsize=10, fontweight="bold", color=color,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

                    # Reference lines
                    ax2r.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
                    ax2r.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
                    ax2r.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
                    ax2r.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
                    ax2r.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")

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

            # NTD for daily panel
            ntd_d2 = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
                )
                kijun_d2 = kijun_d2.ffill().bfill()

            # BBands (Daily, Enhanced)
            bb_mid_d2, bb_up_d2, bb_lo_d2, bb_pctb_d2, bb_nbb_d2 = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            ntd_d_show  = ntd_d2.reindex(df_show.index)
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()

            # BB subset (Enhanced)
            bb_mid_d2_show = bb_mid_d2.reindex(df_show.index)
            bb_up_d2_show  = bb_up_d2.reindex(df_show.index)
            bb_lo_d2_show  = bb_lo_d2.reindex(df_show.index)
            bb_pctb_d2_show= bb_pctb_d2.reindex(df_show.index)
            bb_nbb_d2_show = bb_nbb_d2.reindex(df_show.index)

            # HMA (Daily, Enhanced)
            hma_d2_full = compute_hma(df, period=hma_period) if show_hma else pd.Series(index=df.index, dtype=float)
            hma_d2_show = hma_d2_full.reindex(df_show.index)

            # PSAR (Daily, Enhanced)
            psar_d2_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d2_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d2_df = psar_d2_df.loc[(psar_d2_df.index >= x0) & (psar_d2_df.index <= x1)]

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
            if show_hma and not hma_d2_show.dropna().empty:
                ax.plot(hma_d2_show.index, hma_d2_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(kijun_d2_show.index, kijun_d2_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            # Plot BBands (Daily, Enhanced)
            if show_bbands and not bb_up_d2_show.dropna().empty and not bb_lo_d2_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d2_show, bb_up_d2_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d2_show.index, bb_mid_d2_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d2_show.index, bb_up_d2_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d2_show.index, bb_lo_d2_show.values, ":", linewidth=1.0)

            # PSAR overlay (Daily, Enhanced)
            if show_psar and not psar_d2_df.empty:
                up_mask = psar_d2_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d2_df.index[up_mask], psar_d2_df["PSAR"][up_mask],
                               s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d2_df.index[dn_mask], psar_d2_df["PSAR"][dn_mask],
                               s=15, color="tab:red", zorder=6)

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

            # Probabilistic HMA crossover (Daily, Enhanced)
            if show_hma and not hma_d2_show.dropna().empty:
                cross_d2 = detect_last_crossover(df_show, hma_d2_show)
                if cross_d2 is not None and cross_d2["time"] is not None:
                    ts = cross_d2["time"]; px_here = float(df_show.loc[ts])
                    if cross_d2["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "BUY", hma_conf)
                        st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — last price crossed **up** HMA({hma_period}) with P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)}")
                    elif cross_d2["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "SELL", hma_conf)
                        st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — last price crossed **down** HMA({hma_period}) with P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)}")

            # --- DAILY INDICATOR PANEL (Enhanced): NTD + Trend ONLY ---
            axdw2.set_title("Daily Indicator Panel — NTD + Trend")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw2, ntd_d_show)

            if show_ntd and not ntd_d_show.dropna().empty:
                axdw2.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d2, ntd_m_d2 = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d2.empty:
                    axdw2.plot(ntd_trend_d2.index, ntd_trend_d2.values, "--", linewidth=2,
                               label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d2)}/bar)")

            # Reference lines
            axdw2.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
            axdw2.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
            axdw2.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
            axdw2.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
            axdw2.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")

            axdw2.set_ylim(-1.1, 1.1)
            axdw2.set_xlabel("Date (PST)")
            axdw2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Intraday (Enhanced) -----
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

                # Bollinger Bands (Hourly, Enhanced)
                bb_mid_i, bb_up_i, bb_lo_i, bb_pctb_i, bb_nbb_i = compute_bbands(ic, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

                # HMA (Hourly, Enhanced)
                hma_i = compute_hma(ic, period=hma_period) if show_hma else pd.Series(index=ic.index, dtype=float)

                # PSAR (Hourly, Enhanced)
                psar_i_df = compute_psar_from_ohlc(intr, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                psar_i_df = psar_i_df.reindex(ic.index)

                # Lookback slope & R²
                yhat_h, m_h = slope_line(ic, slope_lb_hourly)
                r2_i = regression_r2(ic, slope_lb_hourly)

                fig3, ax3 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax3.plot(ic.index, ic, label="Intraday")
                ax3.plot(ic.index, ie, "--", label="20 EMA")
                ax3.plot(ic.index, trend_i, "--", label=f"Trend (m={fmt_slope(slope_i)}/bar)", linewidth=2)
                if show_hma and not hma_i.dropna().empty:
                    ax3.plot(hma_i.index, hma_i.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
                if show_ichi and not kijun_i.dropna().empty:
                    ax3.plot(kijun_i.index, kijun_i.values, "-", linewidth=1.8, color="black",
                             label=f"Ichimoku Kijun ({ichi_base})")

                # Plot BBands (Hourly, Enhanced)
                if show_bbands and not bb_up_i.dropna().empty and not bb_lo_i.dropna().empty:
                    ax3.fill_between(ic.index, bb_lo_i, bb_up_i, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                    ax3.plot(bb_mid_i.index, bb_mid_i.values, "-", linewidth=1.1,
                             label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                    ax3.plot(bb_up_i.index, bb_up_i.values, ":", linewidth=1.0)
                    ax3.plot(bb_lo_i.index, bb_lo_i.values, ":", linewidth=1.0)

                # PSAR overlay (Hourly, Enhanced)
                if show_psar and not psar_i_df.dropna().empty:
                    up_mask = psar_i_df["in_uptrend"] == True
                    dn_mask = ~up_mask
                    if up_mask.any():
                        ax3.scatter(psar_i_df.index[up_mask], psar_i_df["PSAR"][up_mask],
                                    s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                    if dn_mask.any():
                        ax3.scatter(psar_i_df.index[dn_mask], psar_i_df["PSAR"][dn_mask],
                                    s=15, color="tab:red", zorder=6)

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

                # === Current price badge (bottom-right) ===
                if np.isfinite(px_val2):
                    nbb_txt2 = ""
                    try:
                        last_pct2 = float(bb_pctb_i.dropna().iloc[-1]) if show_bbands else np.nan
                        last_nbb2 = float(bb_nbb_i.dropna().iloc[-1]) if show_bbands else np.nan
                        if np.isfinite(last_nbb2) and np.isfinite(last_pct2):
                            nbb_txt2 = f"  |  NBB {last_nbb2:+.2f}  •  %B {fmt_pct(last_pct2, digits=0)}"
                    except Exception:
                        pass
                    ax3.text(0.99, 0.02, f"Current price: {fmt_price_val(px_val2)}{nbb_txt2}",
                             transform=ax3.transAxes, ha="right", va="bottom",
                             fontsize=11, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not st_line_intr.dropna().empty:
                    ax3.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax3.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                # Bottom-left slope badge
                ax3.text(0.01, 0.02, f"Slope: {fmt_slope(slope_i)}/bar",
                         transform=ax3.transAxes, ha="left", va="bottom",
                         fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                # Bottom-center R² badge
                ax3.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_i)}",
                         transform=ax3.transAxes, ha="center", va="bottom",
                         fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                # --- NEW: Session lines (PST) for Forex intraday (Enhanced tab) ---
                if mode == "Forex" and show_sessions_pst and not ic.empty:
                    sess2 = compute_session_lines(ic.index)
                    draw_session_lines(ax3, sess2)

                if show_fibs and not ic.empty:
                    fibs_h = fibonacci_levels(ic)
                    for lbl, y in fibs_h.items():
                        ax3.hlines(y, xmin=ic.index[0], xmax=ic.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax3.text(ic.index[-1], y, f" {lbl}", va="center")

                signal2 = sr_proximity_signal(ic, res_i, sup_i, st.session_state.fc_vals,
                                              threshold=signal_threshold, prox=sr_prox_pct)
                if signal2 is not None and np.isfinite(px_val2):
                    if signal2["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")
                    elif signal2["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")

                # Probabilistic HMA crossover (Hourly, Enhanced)
                if show_hma and not hma_i.dropna().empty:
                    cross_i = detect_last_crossover(ic, hma_i)
                    if cross_i is not None and cross_i["time"] is not None:
                        ts = cross_i["time"]; px_here = float(ic.loc[ts])
                        if cross_i["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                            annotate_crossover(ax3, ts, px_here, "BUY", hma_conf)
                            st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — price crossed **up** HMA({hma_period}) with P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)}")
                        elif cross_i["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                            annotate_crossover(ax3, ts, px_here, "SELL", hma_conf)
                            st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — price crossed **down** HMA({hma_period}) with P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)}")

                ax3.set_xlabel("Time (PST)")
                ax3.legend(loc="lower left", framealpha=0.5)
                xlim_price2 = ax3.get_xlim()
                st.pyplot(fig3)

                # === NEW: Intraday Volume panel (separate chart in Enhanced tab) ===
                vol_i = _coerce_1d_series(intr.get("Volume", pd.Series(index=ic.index))).reindex(ic.index).astype(float)
                if _has_volume_to_plot(vol_i):
                    v_mid2 = rolling_midline(vol_i, window=max(3, int(slope_lb_hourly)))
                    v_trend2, v_m2 = slope_line(vol_i, slope_lb_hourly)
                    v_r2_2 = regression_r2(vol_i, slope_lb_hourly)

                    fig3v, ax3v = plt.subplots(figsize=(14, 2.8))
                    ax3v.set_title(f"Volume (Hourly) — Mid-line & Trend  |  Slope={fmt_slope(v_m2)}/bar")
                    # BLUE volume area + line
                    ax3v.fill_between(vol_i.index, 0, vol_i, alpha=0.18, label="Volume", color="tab:blue")
                    ax3v.plot(vol_i.index, vol_i, linewidth=1.0, color="tab:blue")
                    ax3v.plot(v_mid2.index, v_mid2, ":", linewidth=1.6, label=f"Mid-line ({slope_lb_hourly}-roll)")
                    if v_mid2.notna().any():
                        last_mid2 = float(v_mid2.dropna().iloc[-1])
                        ax3v.hlines(last_mid2, xmin=vol_i.index[0], xmax=vol_i.index[-1],
                                    linestyles="dotted", linewidth=1.0)
                        label_on_left(ax3v, last_mid2, f"Mid {last_mid2:,.0f}", color="black")
                    if not v_trend2.empty:
                        ax3v.plot(v_trend2.index, v_trend2.values, "--", linewidth=2,
                                  label=f"Trend {slope_lb_hourly} ({fmt_slope(v_m2)}/bar)")
                    ax3v.text(0.01, 0.02, f"Slope: {fmt_slope(v_m2)}/bar",
                              transform=ax3v.transAxes, ha="left", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                    ax3v.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(v_r2_2)}",
                              transform=ax3v.transAxes, ha="center", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                    ax3v.set_xlim(xlim_price2)
                    ax3v.set_xlabel("Time (PST)")
                    ax3v.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig3v)

                # === Hourly Indicator Panel (Enhanced): NTD + Trend ONLY (SHADED) ===
                if show_nrsi:
                    ntd_i = compute_normalized_trend(ic, window=ntd_window)
                    ntd_trend_i, ntd_m_i = slope_line(ntd_i, slope_lb_hourly)

                    fig3r, ax3r = plt.subplots(figsize=(14,2.8))
                    ax3r.set_title(f"Hourly Indicator Panel — NTD + Trend (win={ntd_window})")

                    # NEW: shading (green above 0, red below 0)
                    if shade_ntd and not ntd_i.dropna().empty:
                        shade_ntd_regions(ax3r, ntd_i)

                    ax3r.plot(ntd_i.index, ntd_i, "-", linewidth=1.6, label="NTD")
                    if not ntd_trend_i.empty:
                        ax3r.plot(ntd_trend_i.index, ntd_trend_i.values, "--", linewidth=2,
                                  label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_i)}/bar)")

                    if not ntd_i.dropna().empty:
                        last_ntd = float(ntd_i.dropna().iloc[-1])
                        t_dir = "UP" if last_ntd >= 0 else "DOWN"
                        color = "tab:green" if last_ntd >= 0 else "tab:red"
                        certainty = int(round(50 + 50*abs(last_ntd)))
                        ax3r.text(0.50, 0.06, f"Trend: {t_dir} ({certainty}%)",
                                  transform=ax3r.transAxes, ha="center", va="bottom",
                                  fontsize=10, fontweight="bold", color=color,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

                    ax3r.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
                    ax3r.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
                    ax3r.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
                    ax3r.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
                    ax3r.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")
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
    st.caption("Shows **symbols with Normalized Trend Direction (NTD) < -0.5** (Daily for Stocks & FX; Hourly for FX). Also lists **Price > Ichimoku Kijun(26)** symbols on the latest bar.")

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
        # DAILY scan for both universes — NTD
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

        # DAILY scan for both universes — Price > Kijun
        st.markdown("---")
        st.subheader(f"Daily — **Price > Ichimoku Kijun({ichi_base})** (latest bar)")
        above_rows = []
        for sym in universe:
            above, ts, close_now, kij_now = price_above_kijun_info_daily(sym, base=ichi_base)
            above_rows.append({
                "Symbol": sym,
                "AboveNow": above,
                "Timestamp": ts,
                "Close": close_now,
                "Kijun": kij_now
            })
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]

        c7, c8 = st.columns(2)
        c7.metric("Daily Price > Kijun", int(df_above_daily.shape[0]))
        c8.caption("Criteria: Latest close strictly greater than current Kijun (Base) value.")

        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            view_above = df_above_daily.copy()
            view_above["Close"] = view_above["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_above["Kijun"] = view_above["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view_above[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

        # HOURLY scan for Forex — NTD & Price > Kijun
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

            st.subheader(f"Forex Hourly — **Price > Ichimoku Kijun({ichi_base})** (latest bar, {scan_hour_range})")
            habove_rows = []
            for sym in universe:
                above_h, ts_h, close_h, kij_h = price_above_kijun_info_hourly(sym, period=scan_period, base=ichi_base)
                habove_rows.append({
                    "Symbol": sym,
                    "AboveNow": above_h,
                    "Timestamp": ts_h,
                    "Close": close_h,
                    "Kijun": kij_h
                })
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]

            c9, c10 = st.columns(2)
            c9.metric("Hourly Price > Kijun", int(df_above_hour.shape[0]))
            c10.caption("Criteria: Latest intraday close strictly greater than current Kijun value.")

            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                vch["Kijun"] = vch["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(vch[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

# --- Tab 6: Long-Term History (5/10/15/20y) ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    # Ticker selector (defaults to the one used in Tab 1, if set)
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox("Ticker:", universe, index=default_idx, key="hist_long_ticker")

    # Year buttons
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("5Y", key="btn_5y"):  st.session_state.hist_years = 5
    if c2.button("10Y", key="btn_10y"): st.session_state.hist_years = 10
    if c3.button("15Y", key="btn_15y"): st.session_state.hist_years = 15
    if c4.button("20Y", key="btn_20y"): st.session_state.hist_years = 20
    years = int(st.session_state.hist_years)
    st.caption(f"Showing last **{years} years**. Support/Resistance = rolling **252-day** extremes; trendline fits the shown window.")

    # Fetch full history once; slice to desired years
    s_full = fetch_hist_max(sym)
    if s_full is None or s_full.empty:
        st.warning("No historical data available.")
    else:
        end_ts = s_full.index.max()
        start_ts = end_ts - pd.DateOffset(years=years)
        s = s_full[s_full.index >= start_ts]
        if s.empty:
            st.warning(f"No data in the last {years} years for {sym}.")
        else:
            # 252-trading-day rolling extremes (approx 1y)
            res_roll = s.rolling(252, min_periods=1).max()
            sup_roll = s.rolling(252, min_periods=1).min()
            res_last = float(res_roll.iloc[-1]) if len(res_roll) else np.nan
            sup_last = float(sup_roll.iloc[-1]) if len(sup_roll) else np.nan

            # Overall trendline across the shown window
            yhat_all, m_all = slope_line(s, lookback=len(s))

            fig, ax = plt.subplots(figsize=(14,5))
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close")

            # Draw S/R as straight lines across the whole window (use last 252d extremes)
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1],
                          colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance (252d)")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1],
                          colors="tab:green", linestyles="-", linewidth=1.6, label="Support (252d)")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")

            # Trendline and slope badge
            if not yhat_all.empty:
                ax.plot(yhat_all.index, yhat_all.values, "--", linewidth=2,
                        label=f"Trend (m={fmt_slope(m_all)}/bar)")

                ax.text(0.01, 0.02, f"Slope: {fmt_slope(m_all)}/bar",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            # Current price badge (bottom-right)
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(0.99, 0.02, f"Current price: {fmt_price_val(px_now)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)
