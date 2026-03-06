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
# - Current price shown OUTSIDE of chart area (top-right above axes)
# - Normalized Elliott Wave panel for Hourly (dates aligned to hourly chart)
# - Normalized Elliott Wave panel for Daily (dates aligned to daily chart, shared x-axis with price)
# - EW panels show BUY/SELL signals when forecast confidence > 95% and display current price on top
# - EW panels draw a red line at +0.5 and a green line at -0.5
# - EW panels draw black lines at +0.75 and -0.25
# - Adds Normalized Price Oscillator (NPO) overlay to EW panels with sidebar controls
# - Adds Normalized Trend Direction (NTD) overlay + optional green/red shading to EW panels with sidebar controls
# - Daily view selector (Historical / 6M / 12M / 24M)
# - Red shading under NPO curve on EW panels
# - NEW: Daily trend-direction line (green=uptrend, red=downtrend) with slope label

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
    """Format price values to exactly 3 decimal places with thousands separators."""
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

# NEW: range helper for daily views
def subset_by_daily_view(obj, view_label: str):
    """Return series/df subset for Historical / 6M / 12M / 24M based on its own max date."""
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

# --- Sidebar config (explicit keys everywhere) ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")

# NEW: daily range selector
daily_view = st.sidebar.selectbox(
    "Daily view range:",
    ["Historical", "6M", "12M", "24M"],
    index=2,  # default 12M
    key="sb_daily_view"
)

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120, 6, key="sb_slope_lb_hourly")

# Hourly Momentum controls
st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=True, key="sb_show_mom_hourly")
mom_lb_hourly   = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

# Hourly Supertrend controls
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# Signal logic controls
st.sidebar.subheader("Signal Logic (Hourly)")
signal_threshold = st.sidebar.slider("Signal confidence threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# Elliott Wave controls (Hourly)
st.sidebar.subheader("Normalized Elliott Wave (Hourly)")
pivot_lookback = st.sidebar.slider("Pivot lookback (bars)", 3, 21, 7, 2, key="sb_pivot_lb")
norm_window    = st.sidebar.slider("Normalization window (bars)", 30, 600, 240, 10, key="sb_norm_win")
waves_to_annotate = st.sidebar.slider("Annotate recent waves", 3, 12, 7, 1, key="sb_wave_ann")

# Elliott Wave controls (Daily)
st.sidebar.subheader("Normalized Elliott Wave (Daily)")
pivot_lookback_d = st.sidebar.slider("Pivot lookback (days)", 3, 31, 9, 2, key="sb_pivot_lb_d")
norm_window_d    = st.sidebar.slider("Normalization window (days)", 30, 1200, 360, 10, key="sb_norm_win_d")
waves_to_annotate_d = st.sidebar.slider("Annotate recent waves (daily)", 3, 12, 7, 1, key="sb_wave_ann_d")

# --- Normalized Price Oscillator controls ---
st.sidebar.subheader("Normalized Price Oscillator (overlay on EW panels)")
show_npo = st.sidebar.checkbox("Show NPO overlay", value=True, key="sb_show_npo")
npo_fast = st.sidebar.slider("NPO fast EMA", 5, 30, 12, 1, key="sb_npo_fast")
npo_slow = st.sidebar.slider("NPO slow EMA", 10, 60, 26, 1, key="sb_npo_slow")
npo_norm_win = st.sidebar.slider("NPO normalization window", 30, 600, 240, 10, key="sb_npo_norm")

# --- Normalized Trend Direction (overlay on EW panels) ---
st.sidebar.subheader("Normalized Trend (EW panels)")
show_ntd = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window = st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")

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
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA','LYFT','QBTS',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO','QQQ','ETHA',
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

# ---- Indicators (no RSI) ----
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
    if ohlc is None or ohlc.empty:
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

# ---- NEW: Normalized Price Oscillator (PPO -> z-score -> tanh) ----
def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26, norm_win: int = 240) -> pd.Series:
    """
    Returns NPO in [-1,1]: tanh( zscore(PPO) / 2 ).
    PPO = (EMA_fast - EMA_slow) / EMA_slow * 100
    NOTE: If params invalid or insufficient data, returns empty/NaN series.
    """
    s = _coerce_1d_series(close)
    if s.empty or not np.isfinite(fast) or not np.isfinite(slow) or fast <= 0 or slow <= 0:
        return pd.Series(index=s.index, dtype=float)
    # Ensure fast < slow for stable PPO
    if fast >= slow:
        fast, slow = max(1, slow - 1), slow
        if fast >= slow:
            return pd.Series(index=s.index, dtype=float)  # cannot fix
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean().replace(0, np.nan)
    ppo = (ema_fast - ema_slow) / ema_slow * 100.0
    # Normalize PPO
    minp = max(10, int(norm_win)//10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std  = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
    z = (ppo - mean) / std
    npo = np.tanh(z / 2.0)
    return npo.reindex(s.index)

# ---- NEW: Normalized Trend Direction (rolling LR slope -> volatility-normalized -> tanh) ----
def compute_normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Normalized Trend Direction (NTD) in [-1,1].
    Steps:
      - Rolling linear-regression slope over 'window' bars
      - Scale by window and divide by rolling std (volatility)
      - Squash with tanh to bound and stabilize
    Interpretation:
      >0 uptrend bias; <0 downtrend bias; magnitude ~ strength
    """
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)

    # rolling slope via polyfit on each window
    minp = max(5, window // 3)
    def _slope(y: pd.Series) -> float:
        y = pd.Series(y).dropna()
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        try:
            m, b = np.polyfit(x, y.to_numpy(dtype=float), 1)
        except Exception:
            return np.nan
        return float(m)

    slope_roll = s.rolling(window, min_periods=minp).apply(_slope, raw=False)
    vol = s.rolling(window, min_periods=minp).std().replace(0, np.nan)

    ntd_raw = (slope_roll * window) / vol
    ntd = np.tanh(ntd_raw / 2.0)
    return ntd.reindex(s.index)

def shade_ntd_regions(ax, ntd: pd.Series):
    """Optional green/red shading under/over zero to emphasize direction."""
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, step=None)
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, step=None)

# NEW: Red shading under NPO curve
def shade_npo_regions(ax, npo: pd.Series):
    """Shade area between NPO and 0 in red (lighter when below zero)."""
    if npo is None or npo.empty:
        return
    pos = npo.where(npo > 0)
    neg = npo.where(npo < 0)
    ax.fill_between(pos.index, 0, pos, alpha=0.15, color="tab:red")
    ax.fill_between(neg.index, 0, neg, alpha=0.08, color="tab:red")

# --- NEW: Daily trend-direction line helper ---
def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    """
    Draws a regression line over the visible daily window and colors it by direction.
    Green = uptrend (positive slope), Red = downtrend (negative slope).
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "tab:green" if m >= 0 else "tab:red"
    ax.plot(s.index, yhat, "-", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# =========================
# NEW: Hot Cake helpers
# =========================
def _trend_slope_1d(series_like) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 3:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    try:
        m, _ = np.polyfit(x, s.to_numpy(dtype=float), 1)
    except Exception:
        return float("nan")
    return float(m)

def _cross_up_level(series_like, level: float) -> pd.Series:
    s = _coerce_1d_series(series_like)
    prev = s.shift(1)
    return ((s >= float(level)) & (prev < float(level))).fillna(False)

def _series_heading_up(series_like, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(series_like).dropna()
    confirm_bars = max(1, int(confirm_bars))
    if len(s) < confirm_bars + 1:
        return False
    d = s.diff().dropna()
    if len(d) < confirm_bars:
        return False
    return bool(np.all(d.iloc[-confirm_bars:] > 0))

def _bars_since(idx: pd.Index, t) -> int:
    try:
        if isinstance(idx, pd.DatetimeIndex):
            loc = int(idx.get_loc(t))
        else:
            loc = int(t)
        return int((len(idx) - 1) - loc)
    except Exception:
        return 10**9

@st.cache_data(ttl=120)
def hotcake_npo_cross_row_daily(symbol: str,
                                daily_view_label: str,
                                npo_level: float = -0.25,
                                npo_fast_: int = 12,
                                npo_slow_: int = 26,
                                npo_norm_: int = 240,
                                confirm_up_bars: int = 1,
                                max_bars_since: int = 10):
    """
    Daily: NPO crosses UP through npo_level recently AND NPO is going up.
    Returns slopes + last values; caller filters by trend direction (up/down).
    """
    try:
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None
        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if len(close_show) < 20:
            return None

        trend_m = _trend_slope_1d(close_show)

        npo_full = compute_npo(close_full, fast=int(npo_fast_), slow=int(npo_slow_), norm_win=int(npo_norm_))
        npo_show = _coerce_1d_series(npo_full).reindex(close_show.index).dropna()
        if len(npo_show) < 3:
            return None

        cross_up = _cross_up_level(npo_show, level=float(npo_level))
        if not cross_up.any():
            return None

        t_cross = cross_up[cross_up].index[-1]
        bars_since = _bars_since(npo_show.index, t_cross)
        if int(bars_since) > int(max_bars_since):
            return None

        # "going upward"
        if not _series_heading_up(npo_show, confirm_bars=int(confirm_up_bars)):
            return None
        npo_cross = float(npo_show.loc[t_cross]) if np.isfinite(npo_show.loc[t_cross]) else np.nan
        npo_last = float(npo_show.iloc[-1]) if np.isfinite(npo_show.iloc[-1]) else np.nan
        if not (np.isfinite(npo_cross) and np.isfinite(npo_last) and (npo_last > npo_cross)):
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time": t_cross,
            "NPO@Cross": npo_cross,
            "NPO(last)": npo_last,
            "Trend Slope": float(trend_m) if np.isfinite(trend_m) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def hotcake_support_reversal_row_daily(symbol: str,
                                       daily_view_label: str,
                                       confirm_up_bars: int = 1,
                                       max_bars_since: int = 10,
                                       prox: float = 0.0025):
    """
    Daily: Trend up AND price recently reversed from Support (30-day rolling min),
    and price is now going up.
    """
    try:
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None
        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if len(close_show) < 20:
            return None

        trend_m = _trend_slope_1d(close_show)
        if not (np.isfinite(trend_m) and trend_m > 0.0):
            return None

        sup_full = close_full.rolling(30, min_periods=1).min()
        sup = _coerce_1d_series(sup_full).reindex(close_show.index).ffill()

        p = _coerce_1d_series(close_show).astype(float)
        s = _coerce_1d_series(sup).reindex(p.index).ffill()

        touched = p.shift(1) <= s.shift(1) * (1.0 + float(prox))
        moved_up = p > p.shift(1)
        ev = (touched & moved_up).fillna(False)
        if not ev.any():
            return None

        t = ev[ev].index[-1]
        bars_since = _bars_since(p.index, t)
        if int(bars_since) > int(max_bars_since):
            return None

        if not _series_heading_up(p, confirm_bars=int(confirm_up_bars)):
            return None

        last_px = float(p.iloc[-1]) if np.isfinite(p.iloc[-1]) else np.nan
        sup_at = float(s.loc[t]) if np.isfinite(s.loc[t]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Reversal": int(bars_since),
            "Reversal Time": t,
            "Support": sup_at,
            "Last Price": last_px,
            "Trend Slope": float(trend_m),
        }
    except Exception:
        return None
# =========================
# NEW: Hot Cake helpers (Hourly)
# =========================
@st.cache_data(ttl=120)
def hotcake_npo_cross_row_hourly(symbol: str,
                                 period: str,
                                 npo_level: float = -0.25,
                                 npo_fast_: int = 12,
                                 npo_slow_: int = 26,
                                 npo_norm_: int = 240,
                                 confirm_up_bars: int = 1,
                                 max_bars_since: int = 30):
    """
    Hourly (5m bars): NPO crosses UP through npo_level recently AND NPO is going up.
    Returns slopes + last values; caller filters by trend direction (up/down).
    """
    try:
        intr = fetch_intraday(symbol, period=period)
        if intr is None or intr.empty or "Close" not in intr:
            return None

        hc = _coerce_1d_series(intr["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        trend_m = _trend_slope_1d(hc)

        npo = compute_npo(hc, fast=int(npo_fast_), slow=int(npo_slow_), norm_win=int(npo_norm_))
        npo = _coerce_1d_series(npo).dropna()
        if len(npo) < 3:
            return None

        cross_up = _cross_up_level(npo, level=float(npo_level))
        if not cross_up.any():
            return None

        bar = int(cross_up[cross_up].index[-1])
        bars_since = int((len(npo) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        if not _series_heading_up(npo, confirm_bars=int(confirm_up_bars)):
            return None

        npo_cross = float(npo.iloc[bar]) if np.isfinite(npo.iloc[bar]) else np.nan
        npo_last = float(npo.iloc[-1]) if np.isfinite(npo.iloc[-1]) else np.nan
        if not (np.isfinite(npo_cross) and np.isfinite(npo_last) and (npo_last > npo_cross)):
            return None

        last_px = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan

        cross_time = intr.index[bar] if isinstance(intr.index, pd.DatetimeIndex) and (0 <= bar < len(intr.index)) else bar

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since Cross": int(bars_since),
            "Cross Time": cross_time,
            "NPO@Cross": npo_cross,
            "NPO(last)": npo_last,
            "Trend Slope": float(trend_m) if np.isfinite(trend_m) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def hotcake_support_reversal_row_hourly(symbol: str,
                                        period: str,
                                        confirm_up_bars: int = 1,
                                        max_bars_since: int = 30,
                                        sr_lb: int = 60,
                                        prox: float = 0.0025):
    """
    Hourly (5m bars): Trend up AND price recently reversed from Support (rolling min),
    and price is now going up.
    """
    try:
        intr = fetch_intraday(symbol, period=period)
        if intr is None or intr.empty or "Close" not in intr:
            return None

        hc = _coerce_1d_series(intr["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        trend_m = _trend_slope_1d(hc)
        if not (np.isfinite(trend_m) and trend_m > 0.0):
            return None

        sup = hc.rolling(int(sr_lb), min_periods=1).min().ffill()
        p = hc.astype(float)
        s = _coerce_1d_series(sup).reindex(p.index).ffill()

        touched = p.shift(1) <= s.shift(1) * (1.0 + float(prox))
        moved_up = p > p.shift(1)
        ev = (touched & moved_up).fillna(False)
        if not ev.any():
            return None

        bar = int(ev[ev].index[-1])
        bars_since = int((len(p) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        if not _series_heading_up(p, confirm_bars=int(confirm_up_bars)):
            return None

        last_px = float(p.iloc[-1]) if np.isfinite(p.iloc[-1]) else np.nan
        sup_at = float(s.iloc[bar]) if np.isfinite(s.iloc[bar]) else np.nan
        rev_time = intr.index[bar] if isinstance(intr.index, pd.DatetimeIndex) and (0 <= bar < len(intr.index)) else bar

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since Reversal": int(bars_since),
            "Reversal Time": rev_time,
            "Support": sup_at,
            "Last Price": last_px,
            "Trend Slope": float(trend_m),
        }
    except Exception:
        return None

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
    "Hot Cake"
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

        # Pre-fetch Forex news (intraday only)
        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily (Price + EW in one figure with shared x) -----
        if chart in ("Daily","Both"):
            # Prepare series and overlays
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            # Compute EW + overlays (full history)
            wave_norm_d, piv_df_d = compute_normalized_elliott_wave(df, pivot_lb=pivot_lookback_d, norm_win=norm_window_d)
            npo_d = compute_npo(df, fast=npo_fast, slow=npo_slow, norm_win=npo_norm_win) if show_npo else pd.Series(index=df.index, dtype=float)
            ntd_d = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)

            # Subset by Daily view range (Historical/6M/12M/24M)
            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            wave_d_show = wave_norm_d.reindex(df_show.index)
            npo_d_show  = npo_d.reindex(df_show.index)
            ntd_d_show  = ntd_d.reindex(df_show.index)
            piv_df_d_show = piv_df_d[(piv_df_d["time"] >= df_show.index.min()) & (piv_df_d["time"] <= df_show.index.max())] if not piv_df_d.empty else piv_df_d

            # Create a single figure with shared x-axis (perfect alignment)
            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            # Top: Daily price panel
            ax.set_title(f"{sel} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema_show.empty:
                ax.plot(yhat_ema_show.index, yhat_ema_show.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            # NEW: Trend-direction line over the visible window
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

            # Bottom: Daily EW + overlays (perfectly aligned via sharex)
            axdw.set_title("Daily Normalized Elliott Wave + NPO + NTD")
            # NTD shading (optional)
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            # NPO red shading
            if show_npo and not npo_d_show.dropna().empty:
                shade_npo_regions(axdw, npo_d_show)

            # Lines on top of shading
            axdw.plot(wave_d_show.index, wave_d_show, label="Norm EW (Daily)", linewidth=1.8)
            if show_npo and not npo_d_show.dropna().empty:
                axdw.plot(npo_d_show.index, npo_d_show, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")

            for yline, style, col, lbl in [
                (0.0, "--", None, "EW 0"),
                (0.5, "-", "tab:red", "EW +0.5"),
                (-0.5, "-", "tab:green", "EW -0.5"),
                (0.75, "-", "black", "EW +0.75"),
                (-0.25, "-", "black", "EW -0.25"),
            ]:
                axdw.axhline(yline, linestyle=style, linewidth=1, color=col, label=lbl)

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")

            # Recent pivot wave numbers
            if not piv_df_d_show.empty:
                show_df_d = piv_df_d_show.tail(int(waves_to_annotate_d))
                for _, r in show_df_d.iterrows():
                    t = r["time"]; w = r["wave"]; typ = r["type"]
                    ylab = 0.9 if typ == 'H' else -0.9
                    axdw.annotate(str(int(w)), (t, ylab),
                                  xytext=(0, 0), textcoords="offset points",
                                  ha="center", va="center",
                                  fontsize=9, fontweight="bold")

            # Price + EW signal label (top-right, within bottom axes context)
            px_daily = _safe_last_float(df)
            ew_sig_d = elliott_conf_signal(px_daily, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw = axdw.get_position()
            label_txt = f"Price: {fmt_price_val(px_daily)}"
            if ew_sig_d is not None:
                side = ew_sig_d['side']
                prob = fmt_pct(ew_sig_d['prob'], digits=0)
                label_txt += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily)}  ({prob})"
            fig.text(posdw.x1, posdw.y1 + 0.01, label_txt, ha="right", va="bottom",
                     fontsize=10, fontweight="bold")

            axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly (unchanged core; EW has NPO red shading) -----
        if chart in ("Hourly","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                hc = intr["Close"].ffill()
                he = hc.ewm(span=20).mean()
                xh = np.arange(len(hc))
                slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
                trend_h = slope_h * xh + intercept_h
                res_h = hc.rolling(60, min_periods=1).max()
                sup_h = hc.rolling(60, min_periods=1).min()

                # Supertrend from intraday OHLC
                st_intraday = compute_supertrend(intr, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday else pd.Series(index=hc.index, dtype=float)

                # Slope on hourly close
                yhat_h, m_h = slope_line(hc, slope_lb_hourly)

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)

                # STRAIGHT Support/Resistance lines across entire chart
                res_val = np.nan
                sup_val = np.nan
                px_val  = np.nan
                try:
                    res_val = float(res_h.iloc[-1])
                    sup_val = float(sup_h.iloc[-1])
                    px_val  = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                # Dynamic title area
                buy_sell_text = ""
                if np.isfinite(sup_val):
                    buy_sell_text += f" — ▲ BUY @{fmt_price_val(sup_val)}"
                if np.isfinite(res_val):
                    buy_sell_text += f"  ▼ SELL @{fmt_price_val(res_val)}"
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text}")

                # Current price label OUTSIDE (top-right above axes)
                if np.isfinite(px_val):
                    pos = ax2.get_position()
                    fig2.text(
                        pos.x1, pos.y1 + 0.02,
                        f"Current price: {fmt_price_val(px_val)}",
                        ha="right", va="bottom",
                        fontsize=11, fontweight="bold"
                    )

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

                if mode == "Forex" and show_fx_news and not hc.empty and 'time' in fx_news:
                    t0, t1 = hc.index[0], hc.index[-1]
                    times = [t for t in fx_news["time"] if t0 <= t <= t1]
                    if times:
                        draw_news_markers(ax2, times, float(hc.min()), float(hc.max()), label="News")

                # Signal (text only)
                signal = sr_proximity_signal(hc, res_h, sup_h, st.session_state.fc_vals,
                                             threshold=signal_threshold, prox=sr_prox_pct)
                if signal is not None and np.isfinite(px_val):
                    if signal["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal['level'])} — {signal['reason']}")
                    elif signal["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal['level'])} — {signal['reason']}")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                xlim_price = ax2.get_xlim()
                st.pyplot(fig2)

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

                # --- Normalized Elliott Wave panel (Hourly) + signals ---
                wave_norm, piv_df = compute_normalized_elliott_wave(hc, pivot_lb=pivot_lookback, norm_win=norm_window)
                npo_h = compute_npo(hc, fast=npo_fast, slow=npo_slow, norm_win=npo_norm_win) if show_npo else pd.Series(index=hc.index, dtype=float)
                ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)

                fig2w, ax2w = plt.subplots(figsize=(14,2.8))
                plt.subplots_adjust(top=0.88, right=0.93)

                ax2w.set_title("Normalized Elliott Wave + NPO + NTD")
                if show_ntd and shade_ntd and not ntd_h.dropna().empty:
                    shade_ntd_regions(ax2w, ntd_h)
                # NPO red shading
                if show_npo and not npo_h.dropna().empty:
                    shade_npo_regions(ax2w, npo_h)

                ax2w.plot(wave_norm.index, wave_norm, label="Norm EW", linewidth=1.8)
                if show_npo and not npo_h.dropna().empty:
                    ax2w.plot(npo_h.index, npo_h, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
                if show_ntd and not ntd_h.dropna().empty:
                    ax2w.plot(ntd_h.index, ntd_h, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")

                ax2w.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
                ax2w.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
                ax2w.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")
                ax2w.axhline(0.75, color="black", linestyle="-", linewidth=1, label="EW +0.75")
                ax2w.axhline(-0.25, color="black", linestyle="-", linewidth=1, label="EW -0.25")

                ax2w.set_ylim(-1.1, 1.1)
                ax2w.set_xlabel("Time (PST)")
                ax2w.set_xlim(xlim_price)

                if not piv_df.empty:
                    show_df = piv_df.tail(int(waves_to_annotate))
                    for _, r in show_df.iterrows():
                        t = r["time"]; w = r["wave"]; typ = r["type"]
                        ylab = 0.9 if typ == 'H' else -0.9
                        ax2w.annotate(str(int(w)), (t, ylab),
                                      xytext=(0, 0), textcoords="offset points",
                                      ha="center", va="center",
                                      fontsize=9, fontweight="bold")

                px_intr = _safe_last_float(hc)
                ew_sig_h = elliott_conf_signal(px_intr, st.session_state.fc_vals, EW_CONFIDENCE)
                pos2w = ax2w.get_position()
                label_txt_h = f"Price: {fmt_price_val(px_intr)}"
                if ew_sig_h is not None:
                    side = ew_sig_h['side']
                    prob = fmt_pct(ew_sig_h['prob'], digits=0)
                    label_txt_h += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_intr)}  ({prob})"
                fig2w.text(pos2w.x1, pos2w.y1 + 0.01, label_txt_h, ha="right", va="bottom",
                           fontsize=10, fontweight="bold")

                ax2w.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2w)

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

        # ----- Daily (Price + EW in one figure with shared x) -----
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

            # Apply the same daily view range
            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            wave_d_show = wave_norm_d2.reindex(df_show.index)
            npo_d_show  = npo_d2.reindex(df_show.index)
            ntd_d_show  = ntd_d2.reindex(df_show.index)
            piv_df_d_show = piv_df_d2[(piv_df_d2["time"] >= df_show.index.min()) & (piv_df_d2["time"] <= df_show.index.max())] if not piv_df_d2.empty else piv_df_d2

            fig, (ax, axdw2) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            # Top: Daily panel
            ax.set_title(f"{st.session_state.ticker} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema_show.empty:
                ax.plot(yhat_ema_show.index, yhat_ema_show.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            # NEW: Trend-direction line over the visible window
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

            # Bottom: Daily EW + overlays
            axdw2.set_title("Daily Normalized Elliott Wave + NPO + NTD")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw2, ntd_d_show)
            if show_npo and not npo_d_show.dropna().empty:
                shade_npo_regions(axdw2, npo_d_show)

            axdw2.plot(wave_d_show.index, wave_d_show, label="Norm EW (Daily)", linewidth=1.8)
            if show_npo and not npo_d_show.dropna().empty:
                axdw2.plot(npo_d_show.index, npo_d_show, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw2.plot(ntd_d_show.index, ntd_d_show, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")

            for yline, style, col, lbl in [
                (0.0, "--", None, "EW 0"),
                (0.5, "-", "tab:red", "EW +0.5"),
                (-0.5, "-", "tab:green", "EW -0.5"),
                (0.75, "-", "black", "EW +0.75"),
                (-0.25, "-", "black", "EW -0.25"),
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

            # Price + EW signal label
            px_daily2 = _safe_last_float(df)
            ew_sig_d2 = elliott_conf_signal(px_daily2, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw2 = axdw2.get_position()
            label_txt_d2 = f"Price: {fmt_price_val(px_daily2)}"
            if ew_sig_d2 is not None:
                side = ew_sig_d2['side']
                prob = fmt_pct(ew_sig_d2['prob'], digits=0)
                label_txt_d2 += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily2)}  ({prob})"
            fig.text(posdw2.x1, posdw2.y1 + 0.01, label_txt_d2, ha="right", va="bottom",
                     fontsize=10, fontweight="bold")

            axdw2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

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
                yhat_h, m_h = slope_line(ic, slope_lb_hourly)

                fig3, ax3 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax3.plot(ic.index, ic, label="Intraday")
                ax3.plot(ic.index, ie, "--", label="20 EMA")
                ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)

                res_val2 = np.nan
                sup_val2 = np.nan
                px_val2  = np.nan
                try:
                    res_val2 = float(res_i.iloc[-1])
                    sup_val2 = float(sup_i.iloc[-1])
                    px_val2  = float(ic.iloc[-1])
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
                if np.isfinite(sup_val2):
                    buy_sell_text2 += f" — ▲ BUY @{fmt_price_val(sup_val2)}"
                if np.isfinite(res_val2):
                    buy_sell_text2 += f"  ▼ SELL @{fmt_price_val(res_val2)}"
                ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text2}")

                if np.isfinite(px_val2):
                    pos2 = ax3.get_position()
                    fig3.text(
                        pos2.x1, pos2.y1 + 0.02,
                        f"Current price: {fmt_price_val(px_val2)}",
                        ha="right", va="bottom",
                        fontsize=11, fontweight="bold"
                    )

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

                signal2 = sr_proximity_signal(ic, res_i, sup_i, st.session_state.fc_vals,
                                              threshold=signal_threshold, prox=sr_prox_pct)
                if signal2 is not None and np.isfinite(px_val2):
                    if signal2["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")
                    elif signal2["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")

                ax3.set_xlabel("Time (PST)")
                ax3.legend(loc="lower left", framealpha=0.5)
                xlim_price2 = ax3.get_xlim()
                st.pyplot(fig3)

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

                # --- Hourly EW panel + NPO + NTD + signals ---
                wave_norm2, piv_df2 = compute_normalized_elliott_wave(ic, pivot_lb=pivot_lookback, norm_win=norm_window)
                npo_h2 = compute_npo(ic, fast=npo_fast, slow=npo_slow, norm_win=npo_norm_win) if show_npo else pd.Series(index=ic.index, dtype=float)
                ntd_h2 = compute_normalized_trend(ic, window=ntd_window) if show_ntd else pd.Series(index=ic.index, dtype=float)

                fig3w, ax3w = plt.subplots(figsize=(14,2.8))
                plt.subplots_adjust(top=0.88, right=0.93)

                ax3w.set_title("Normalized Elliott Wave + NPO + NTD")
                if show_ntd and shade_ntd and not ntd_h2.dropna().empty:
                    shade_ntd_regions(ax3w, ntd_h2)
                if show_npo and not npo_h2.dropna().empty:
                    shade_npo_regions(ax3w, npo_h2)  # red shading

                ax3w.plot(wave_norm2.index, wave_norm2, label="Norm EW", linewidth=1.8)
                if show_npo and not npo_h2.dropna().empty:
                    ax3w.plot(npo_h2.index, npo_h2, "--", linewidth=1.2, label=f"NPO ({npo_fast},{npo_slow})")
                if show_ntd and not ntd_h2.dropna().empty:
                    ax3w.plot(ntd_h2.index, ntd_h2, ":", linewidth=1.2, label=f"NTD (win={ntd_window})")

                ax3w.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
                ax3w.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
                ax3w.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")
                ax3w.axhline(0.75, color="black", linestyle="-", linewidth=1, label="EW +0.75")
                ax3w.axhline(-0.25, color="black", linestyle="-", linewidth=1, label="EW -0.25")

                ax3w.set_ylim(-1.1, 1.1)
                ax3w.set_xlabel("Time (PST)")
                ax3w.set_xlim(xlim_price2)

                if not piv_df2.empty:
                    show_df2 = piv_df2.tail(int(waves_to_annotate))
                    for _, r in show_df2.iterrows():
                        t = r["time"]; w = r["wave"]; typ = r["type"]
                        ylab = 0.9 if typ == 'H' else -0.9
                        ax3w.annotate(str(int(w)), (t, ylab),
                                      xytext=(0, 0), textcoords="offset points",
                                      ha="center", va="center",
                                      fontsize=9, fontweight="bold")

                px_intr2 = _safe_last_float(ic)
                ew_sig_h2 = elliott_conf_signal(px_intr2, st.session_state.fc_vals, EW_CONFIDENCE)
                pos3w = ax3w.get_position()
                label_txt_h2 = f"Price: {fmt_price_val(px_intr2)}"
                if ew_sig_h2 is not None:
                    side = ew_sig_h2['side']
                    prob = fmt_pct(ew_sig_h2['prob'], digits=0)
                    label_txt_h2 += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_intr2)}  ({prob})"
                fig3w.text(pos3w.x1, pos3w.y1 + 0.01, label_txt_h2, ha="right", va="bottom",
                           fontsize=10, fontweight="bold")

                ax3w.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig3w)

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
        ax.plot(df3m.index, res3m, ":", label="Resistance")
        ax.plot(df3m.index, sup3m, ":", label="Support")
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
        ax0.plot(df0.index, res0, ":", label="Resistance")
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

# =========================
# NEW: Hot Cake core utilities + Daily helpers
# =========================
def _trend_slope_1d(series_like) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 3:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    try:
        m, _ = np.polyfit(x, s.to_numpy(dtype=float), 1)
    except Exception:
        return float("nan")
    return float(m)

def _cross_up_level(series_like: pd.Series, level: float) -> pd.Series:
    s = _coerce_1d_series(series_like)
    prev = s.shift(1)
    return ((s >= float(level)) & (prev < float(level))).fillna(False)

def _series_heading_up(series_like: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(series_like).dropna()
    confirm_bars = max(1, int(confirm_bars))
    if len(s) < confirm_bars + 1:
        return False
    d = s.diff().dropna()
    if len(d) < confirm_bars:
        return False
    return bool(np.all(d.iloc[-confirm_bars:] > 0))

@st.cache_data(ttl=120)
def hotcake_npo_cross_row_daily(symbol: str,
                                daily_view_label: str,
                                npo_level: float = -0.25,
                                npo_fast_: int = 12,
                                npo_slow_: int = 26,
                                npo_norm_: int = 240,
                                confirm_up_bars: int = 1,
                                max_bars_since: int = 20):
    """
    Daily: Trend slope computed over the visible daily view.
    Condition: NPO crosses UP through npo_level recently, and NPO is going up.
    """
    try:
        close = fetch_hist(symbol).dropna()
        if close.empty:
            return None

        close_show = subset_by_daily_view(close, daily_view_label).dropna()
        if len(close_show) < 30:
            return None

        trend_m = _trend_slope_1d(close_show)

        npo = compute_npo(close, fast=int(npo_fast_), slow=int(npo_slow_), norm_win=int(npo_norm_))
        npo_show = _coerce_1d_series(npo).reindex(close_show.index).dropna()
        if len(npo_show) < 3:
            return None

        cross_up = _cross_up_level(npo_show, level=float(npo_level))
        if not cross_up.any():
            return None

        t_cross = cross_up[cross_up].index[-1]
        bars_since = int((len(npo_show) - 1) - int(npo_show.index.get_loc(t_cross)))
        if int(bars_since) > int(max_bars_since):
            return None

        if not _series_heading_up(npo_show, confirm_bars=int(confirm_up_bars)):
            return None

        npo_cross = float(npo_show.loc[t_cross]) if np.isfinite(npo_show.loc[t_cross]) else np.nan
        npo_last = float(npo_show.iloc[-1]) if np.isfinite(npo_show.iloc[-1]) else np.nan
        if not (np.isfinite(npo_cross) and np.isfinite(npo_last) and (npo_last > npo_cross)):
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time": t_cross,
            "NPO@Cross": npo_cross,
            "NPO(last)": npo_last,
            "Trend Slope": float(trend_m) if np.isfinite(trend_m) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def hotcake_support_reversal_row_daily(symbol: str,
                                       daily_view_label: str,
                                       sr_lb: int = 30,
                                       prox: float = 0.0025,
                                       confirm_up_bars: int = 1,
                                       max_bars_since: int = 20):
    """
    Daily: Trend up AND price recently reversed from Support (rolling min), and price is now going up.
    """
    try:
        close = fetch_hist(symbol).dropna()
        if close.empty:
            return None

        close_show = subset_by_daily_view(close, daily_view_label).dropna()
        if len(close_show) < 30:
            return None

        trend_m = _trend_slope_1d(close_show)
        if not (np.isfinite(trend_m) and trend_m > 0.0):
            return None

        sup = close.rolling(int(sr_lb), min_periods=1).min().reindex(close_show.index).ffill()
        p = _coerce_1d_series(close_show).astype(float)
        s = _coerce_1d_series(sup).reindex(p.index).ffill()

        touched = p.shift(1) <= s.shift(1) * (1.0 + float(prox))
        moved_up = p > p.shift(1)
        ev = (touched & moved_up).fillna(False)
        if not ev.any():
            return None

        t = ev[ev].index[-1]
        bars_since = int((len(p) - 1) - int(p.index.get_loc(t)))
        if int(bars_since) > int(max_bars_since):
            return None

        if not _series_heading_up(p, confirm_bars=int(confirm_up_bars)):
            return None

        last_px = float(p.iloc[-1]) if np.isfinite(p.iloc[-1]) else np.nan
        sup_at = float(s.loc[t]) if np.isfinite(s.loc[t]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Reversal": int(bars_since),
            "Reversal Time": t,
            "Support": sup_at,
            "Last Price": last_px,
            "Trend Slope": float(trend_m),
        }
    except Exception:
        return None

# =========================
# TAB 5 — Hot Cake  ✅ NEW
# =========================
with tab5:
    st.header("Hot Cake")
    st.caption(
        "Daily + Hourly lists:\n"
        "(1) **Trend UP** and **NPO** recently crossed **UP** through **-0.25**, and is still going up.\n"
        "(2) **Trend DOWN** and **NPO** recently crossed **UP** through **-0.25**, and is still going up.\n"
        "(3) **Trend UP** and **Price** recently reversed from **Support**, and price is now going up."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key="hotcake_rows")
    within_daily = c2.selectbox("Daily: within N bars", [3, 5, 10, 15, 20, 30], index=2, key="hotcake_within_d")
    within_hourly = c3.selectbox("Hourly: within N bars", [3, 5, 10, 15, 20, 30, 60], index=5, key="hotcake_within_h")
    hours = c4.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key="hotcake_hr_win")
    confirm_up = c5.slider("Confirm 'going up' bars", 1, 5, 1, 1, key="hotcake_confirm")

    run_hot = st.button("Run Hot Cake Scan", key="btn_run_hotcake", use_container_width=True)

    if run_hot:
        period_map_hot = {"24h": "1d", "48h": "2d", "96h": "4d"}
        hr_period = period_map_hot.get(hours, "1d")

        # ---------- DAILY ----------
        daily_npo_rows = []
        daily_rev_rows = []

        for sym in universe:
            r = hotcake_npo_cross_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                npo_level=-0.25,
                npo_fast_=int(npo_fast),
                npo_slow_=int(npo_slow),
                npo_norm_=int(npo_norm_win),
                confirm_up_bars=int(confirm_up),
                max_bars_since=int(within_daily),
            )
            if r:
                daily_npo_rows.append(r)

            rr = hotcake_support_reversal_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=30,
                prox=float(sr_prox_pct),
                confirm_up_bars=int(confirm_up),
                max_bars_since=int(within_daily),
            )
            if rr:
                daily_rev_rows.append(rr)

        d_uptrend_npo = [r for r in daily_npo_rows if np.isfinite(r.get("Trend Slope", np.nan)) and float(r["Trend Slope"]) > 0.0]
        d_dntrend_npo = [r for r in daily_npo_rows if np.isfinite(r.get("Trend Slope", np.nan)) and float(r["Trend Slope"]) < 0.0]

        st.subheader("Daily Chart")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.subheader("1) Trend UP + NPO crossed UP through -0.25")
            if not d_uptrend_npo:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_uptrend_npo).sort_values(["Bars Since Cross", "NPO(last)", "Trend Slope"], ascending=[True, False, False])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        with d2:
            st.subheader("2) Trend DOWN + NPO crossed UP through -0.25")
            if not d_dntrend_npo:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_dntrend_npo).sort_values(["Bars Since Cross", "NPO(last)", "Trend Slope"], ascending=[True, False, True])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        with d3:
            st.subheader("3) Trend UP + Price reversed from Support and going up")
            if not daily_rev_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(daily_rev_rows).sort_values(["Bars Since Reversal", "Trend Slope"], ascending=[True, False])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        # ---------- HOURLY ----------
        hourly_npo_rows = []
        hourly_rev_rows = []

        for sym in universe:
            r = hotcake_npo_cross_row_hourly(
                symbol=sym,
                period=hr_period,
                npo_level=-0.25,
                npo_fast_=int(npo_fast),
                npo_slow_=int(npo_slow),
                npo_norm_=int(npo_norm_win),
                confirm_up_bars=int(confirm_up),
                max_bars_since=int(within_hourly),
            )
            if r:
                hourly_npo_rows.append(r)

            rr = hotcake_support_reversal_row_hourly(
                symbol=sym,
                period=hr_period,
                confirm_up_bars=int(confirm_up),
                max_bars_since=int(within_hourly),
                sr_lb=60,
                prox=float(sr_prox_pct),
            )
            if rr:
                hourly_rev_rows.append(rr)

        h_uptrend_npo = [r for r in hourly_npo_rows if np.isfinite(r.get("Trend Slope", np.nan)) and float(r["Trend Slope"]) > 0.0]
        h_dntrend_npo = [r for r in hourly_npo_rows if np.isfinite(r.get("Trend Slope", np.nan)) and float(r["Trend Slope"]) < 0.0]

        st.subheader(f"Hourly Chart ({hours})")
        h1, h2, h3 = st.columns(3)

        with h1:
            st.subheader("1) Trend UP + NPO crossed UP through -0.25")
            if not h_uptrend_npo:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_uptrend_npo).sort_values(["Bars Since Cross", "NPO(last)", "Trend Slope"], ascending=[True, False, False])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        with h2:
            st.subheader("2) Trend DOWN + NPO crossed UP through -0.25")
            if not h_dntrend_npo:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_dntrend_npo).sort_values(["Bars Since Cross", "NPO(last)", "Trend Slope"], ascending=[True, False, True])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        with h3:
            st.subheader("3) Trend UP + Price reversed from Support and going up")
            if not hourly_rev_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(hourly_rev_rows).sort_values(["Bars Since Reversal", "Trend Slope"], ascending=[True, False])
                st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)
