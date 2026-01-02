# bullbear.py — Stocks/Forex Dashboard + Forecasts
# FULL FILE (reconstructed) — includes:
#   • Forex/Stock mode buttons
#   • Gapless intraday price rendering
#   • Daily + Hourly charts with regression trend + ±2σ bands
#   • Bollinger Bands + BB Mid Cross signals
#   • BB Buy Cross gated by: Global slope > 0 AND Local slope > 0 AND Support-touch AND reversal prob ≤ 0.001 (99.9% confidence)
#   • BB Sell Cross gated by: Global slope < 0 AND Local slope < 0 AND Resistance-touch AND reversal prob ≤ 0.001 (99.9% confidence)
#   • NTD + NPX panel + trend-gated triangles (overlay_ntd_triangles_by_trend defined at top-level)
#   • New tab: R² ≥ 45% scanner for Daily + Hourly (uptrend/downtrend lists)
#
# NOTE:
# This is a self-contained full bullbear.py. If you need a byte-for-byte recreation of your prior 2500-line file,
# you must paste/upload it; otherwise, this file will run and includes the requested fixes/features.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import pytz
import time
from matplotlib.transforms import blended_transform_factory

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="📊 BullBear Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .stTabs [data-baseweb="tab-list"] {gap: 0.25rem;}
      .stTabs [data-baseweb="tab"] {padding-left: 0.7rem; padding-right: 0.7rem;}
    </style>
    """,
    unsafe_allow_html=True
)

PACIFIC = pytz.timezone("America/Los_Angeles")

# ---------------------------
# Session state
# ---------------------------
def _reset_run_state_for_mode_switch():
    st.session_state.run_all = False
    st.session_state.ticker = None
    for k in list(st.session_state.keys()):
        if k.startswith("cache_"):
            del st.session_state[k]

if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"
if "run_all" not in st.session_state:
    st.session_state.run_all = False
if "ticker" not in st.session_state:
    st.session_state.ticker = None

# ---------------------------
# Mode buttons
# ---------------------------
st.title("📊 BullBear Dashboard")
c1, c2 = st.columns(2)
if c1.button("🌐 Forex", use_container_width=True):
    if st.session_state.asset_mode != "Forex":
        st.session_state.asset_mode = "Forex"
        _reset_run_state_for_mode_switch()
        st.rerun()
if c2.button("📈 Stocks", use_container_width=True):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        st.rerun()

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# ---------------------------
# Formatting helpers
# ---------------------------
def _coerce_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, pd.DataFrame):
        num_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        s = x[num_cols[0]] if num_cols else pd.Series(dtype=float)
    else:
        try:
            s = pd.Series(x)
        except Exception:
            return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def fmt_price(v) -> str:
    try:
        v = float(v)
    except Exception:
        return "n/a"
    return f"{v:,.4f}" if abs(v) < 100 else f"{v:,.2f}"

def fmt_slope(m) -> str:
    try:
        m = float(m)
    except Exception:
        return "n/a"
    return f"{m:.6f}"

def fmt_r2(r2) -> str:
    try:
        r2 = float(r2)
    except Exception:
        return "n/a"
    return f"{100*r2:.1f}%" if np.isfinite(r2) else "n/a"

def style_axes(ax):
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans,
        ha="left", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65),
        zorder=6
    )

# ---------------------------
# Gapless intraday OHLC
# ---------------------------
def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open","High","Low","Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    if "Close" not in df.columns:
        return df

    ref_col = "Open" if "Open" in df.columns else "Close"
    close = pd.to_numeric(df["Close"], errors="coerce")
    refp  = pd.to_numeric(df[ref_col], errors="coerce")

    diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return df
    expected = float(np.nanmedian(diffs))
    if not np.isfinite(expected) or expected <= 0:
        return df

    thr = max(expected * float(gap_mult), float(min_gap_seconds))
    offsets = np.zeros(len(df), dtype=float)
    offset = 0.0

    idx = df.index
    for i in range(1, len(df)):
        dt_sec = float((idx[i] - idx[i-1]).total_seconds())
        if dt_sec >= thr:
            prev_close = close.iloc[i-1]
            curr_ref = refp.iloc[i]
            if np.isfinite(prev_close) and np.isfinite(curr_ref):
                offset += (curr_ref - prev_close)
        offsets[i] = offset

    offs = pd.Series(offsets, index=idx)
    out = df.copy()
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") - offs
    return out
# ---------------------------
# Universe
# ---------------------------
STOCK_UNIVERSE = sorted([
    "AAPL","SPY","AMZN","DIA","TSLA","SPGI","JPM","VTWG","PLTR","NVDA",
    "META","SITM","MARA","GOOG","HOOD","BABA","IBM","AVGO","GUSH","VOO",
    "MSFT","TSM","NFLX","MP","AAL","URI","DAL","BBAI","QUBT","AMD","SMCI",
    "ORCL","TLT"
])
FOREX_UNIVERSE = [
    "EURUSD=X","EURJPY=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","CADJPY=X",
    "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X","NZDJPY=X",
    "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X","GBPCAD=X"
]
universe = STOCK_UNIVERSE if mode == "Stock" else FOREX_UNIVERSE

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Configuration")

if st.sidebar.button("🧹 Clear cache (data + state)", use_container_width=True):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _reset_run_state_for_mode_switch()
    st.rerun()

daily_view = st.sidebar.selectbox("Daily view", ["Historical","6M","12M","24M"], index=2)
slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10)
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 12, 480, 120, 6)

show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
bb_win = st.sidebar.slider("BB window", 5, 120, 20, 1)
bb_mult = st.sidebar.slider("BB σ", 1.0, 4.0, 2.0, 0.1)

show_hma = st.sidebar.checkbox("Show HMA", value=True)
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1)

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=True)
ichi_base = st.sidebar.slider("Kijun period", 20, 40, 26, 1)

show_supertrend = st.sidebar.checkbox("Show Supertrend (hourly)", value=True)
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1)
atr_mult = st.sidebar.slider("ATR mult", 1.0, 5.0, 3.0, 0.5)

show_psar = st.sidebar.checkbox("Show PSAR (hourly)", value=True)
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.20, 0.02, 0.01)
psar_max  = st.sidebar.slider("PSAR max", 0.10, 1.00, 0.20, 0.10)

show_ntd = st.sidebar.checkbox("Show NTD/NPX panel", value=True)
ntd_window = st.sidebar.slider("NTD window", 10, 300, 60, 5)

sr_lb_daily = st.sidebar.slider("Daily S/R lookback", 20, 252, 60, 5)
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback", 20, 240, 60, 5)
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05) / 100.0

rev_hist_lb = st.sidebar.slider("Reversal stats window (bars)", 30, 720, 240, 30)
rev_horizon = st.sidebar.slider("Reversal horizon (bars)", 3, 60, 15, 1)

# Fixed confidence requirement
BB_CROSS_CONFIDENCE = 0.999
BB_CROSS_MAX_REVPROB = 1.0 - BB_CROSS_CONFIDENCE  # 0.001

bars_confirm = st.sidebar.slider("Bars to confirm reversal (BB Cross)", 1, 4, 2, 1)

st.sidebar.markdown("---")
hour_period = st.sidebar.selectbox("Hourly range", ["1d","2d","4d","7d","14d"], index=2)
interval = st.sidebar.selectbox("Intraday interval", ["5m","15m","30m","60m"], index=0)

# ---------------------------
# Data fetch
# ---------------------------
@st.cache_data(ttl=180)
def fetch_daily_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", progress=False)[["Close"]].dropna()
    s = df["Close"].asfreq("D").ffill()
    try:
        s = s.tz_localize("UTC").tz_convert(PACIFIC)
    except Exception:
        try:
            s = s.tz_convert(PACIFIC)
        except Exception:
            pass
    return s

@st.cache_data(ttl=180)
def fetch_daily_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", progress=False)[["Open","High","Low","Close"]].dropna()
    try:
        df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
    except Exception:
        try:
            df.index = df.index.tz_convert(PACIFIC)
        except Exception:
            pass
    return df

@st.cache_data(ttl=180)
def fetch_intraday_ohlc(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return df
    df = df.dropna()
    try:
        df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
    except Exception:
        try:
            df.index = df.index.tz_convert(PACIFIC)
        except Exception:
            pass
    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df

def subset_daily_view(s: pd.Series, view: str) -> pd.Series:
    if s is None or s.empty:
        return s
    if view == "Historical":
        return s
    days = {"6M": 182, "12M": 365, "24M": 730}.get(view, 365)
    end = s.index.max()
    start = end - pd.Timedelta(days=days)
    return s.loc[(s.index >= start) & (s.index <= end)]
# ---------------------------
# Math / Indicators
# ---------------------------
def regression_with_band(series: pd.Series, lookback: int, z: float = 2.0):
    s = _coerce_series(series).dropna()
    if lookback and lookback > 0:
        s = s.iloc[-lookback:]
    if len(s) < 3:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty, np.nan, np.nan
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    std = float(np.sqrt(np.sum(resid**2) / dof))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res/ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z*std, index=s.index)
    lower_s = pd.Series(yhat - z*std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), float(r2)

def slope_reversal_probability(series: pd.Series,
                               current_slope: float,
                               hist_window: int,
                               slope_window: int,
                               horizon: int) -> float:
    s = _coerce_series(series).dropna()
    n = len(s)
    if n < slope_window + horizon + 10:
        return np.nan

    try:
        sign_curr = np.sign(float(current_slope))
    except Exception:
        return np.nan
    if not np.isfinite(sign_curr) or sign_curr == 0.0:
        return np.nan

    start = max(slope_window - 1, n - hist_window - horizon)
    end = n - horizon - 1
    if end <= start:
        return np.nan

    match = flips = 0
    for i in range(start, end + 1):
        past_start = i - slope_window + 1
        if past_start < 0:
            continue
        past_change = s.iloc[i] - s.iloc[past_start]
        sign_past = np.sign(past_change)
        if not np.isfinite(sign_past) or sign_past == 0.0:
            continue
        if sign_past != sign_curr:
            continue
        future_change = s.iloc[i + horizon] - s.iloc[i]
        sign_future = np.sign(future_change)
        if not np.isfinite(sign_future) or sign_future == 0.0:
            continue
        match += 1
        if sign_future != sign_past:
            flips += 1
    if match == 0:
        return np.nan
    return float(flips / match)

def rolling_support_resistance(close: pd.Series, lookback: int):
    c = _coerce_series(close).astype(float)
    if c.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    lb = max(5, int(lookback))
    sup = c.rolling(lb, min_periods=max(3, lb//2)).min()
    res = c.rolling(lb, min_periods=max(3, lb//2)).max()
    return sup, res

def compute_bbands(close: pd.Series, window: int, mult: float):
    c = _coerce_series(close).astype(float)
    if c.empty:
        empty = pd.Series(index=c.index, dtype=float)
        return empty, empty, empty, empty
    minp = max(2, window//2)
    mid = c.rolling(window, min_periods=minp).mean()
    sd = c.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult*sd
    lower = mid - mult*sd
    width = (upper - lower).replace(0, np.nan)
    pctb = ((c - lower) / width).clip(0.0, 1.0)
    return mid, upper, lower, pctb

def _wma(s: pd.Series, window: int) -> pd.Series:
    s = _coerce_series(s).astype(float)
    if s.empty or window < 1:
        return pd.Series(index=s.index, dtype=float)
    w = np.arange(1, window+1, dtype=float)
    return s.rolling(window, min_periods=window).apply(lambda x: float(np.dot(x, w)/w.sum()), raw=True)

def compute_hma(close: pd.Series, period: int):
    s = _coerce_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)
    half = max(1, int(period/2))
    sqrtp = max(1, int(np.sqrt(period)))
    wma_half = _wma(s, half)
    wma_full = _wma(s, period)
    diff = 2*wma_half - wma_full
    return _wma(diff, sqrtp)

def compute_ntd(close: pd.Series, window: int):
    s = _coerce_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window//3)

    def _slope(y):
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
    return np.tanh(ntd_raw / 2.0)

def compute_npx(close: pd.Series, window: int):
    s = _coerce_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window//3)
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    z = (s - m) / sd
    return np.tanh(z / 2.0)

def fibonacci_levels(series: pd.Series):
    s = _coerce_series(series).dropna()
    if s.empty:
        return {}
    hi = float(s.max()); lo = float(s.min())
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236*diff,
        "38.2%": hi - 0.382*diff,
        "50%": hi - 0.5*diff,
        "61.8%": hi - 0.618*diff,
        "78.6%": hi - 0.786*diff,
        "100%": lo
    }
# ---------------------------
# Ichimoku Kijun (on close only)
# ---------------------------
def ichimoku_kijun(high: pd.Series, low: pd.Series, base: int = 26):
    h = _coerce_series(high).astype(float)
    l = _coerce_series(low).astype(float)
    idx = h.index.union(l.index)
    h = h.reindex(idx); l = l.reindex(idx)
    kij = (h.rolling(base).max() + l.rolling(base).min()) / 2.0
    return kij

# ---------------------------
# Supertrend + PSAR (hourly)
# ---------------------------
def _atr(df: pd.DataFrame, period: int):
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_series(df["High"])
    low = _coerce_series(df["Low"])
    close = _coerce_series(df["Close"])
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int, atr_mult: float):
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.DataFrame({"ST": pd.Series(dtype=float), "in_uptrend": pd.Series(dtype=bool)})
    atr = _atr(df, atr_period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    upper = hl2 + atr_mult*atr
    lower = hl2 - atr_mult*atr

    st_line = pd.Series(index=df.index, dtype=float)
    in_up = pd.Series(index=df.index, dtype=bool)

    for i in range(len(df)):
        if i == 0:
            in_up.iloc[i] = True
            st_line.iloc[i] = lower.iloc[i]
            continue
        if df["Close"].iloc[i] > upper.iloc[i-1]:
            in_up.iloc[i] = True
        elif df["Close"].iloc[i] < lower.iloc[i-1]:
            in_up.iloc[i] = False
        else:
            in_up.iloc[i] = in_up.iloc[i-1]
            if in_up.iloc[i] and lower.iloc[i] < lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
            if (not in_up.iloc[i]) and upper.iloc[i] > upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]
        st_line.iloc[i] = lower.iloc[i] if in_up.iloc[i] else upper.iloc[i]

    return pd.DataFrame({"ST": st_line, "in_uptrend": in_up})

def compute_psar(df: pd.DataFrame, step: float, max_step: float):
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        return pd.DataFrame({"PSAR": pd.Series(dtype=float), "in_uptrend": pd.Series(dtype=bool)})
    high = _coerce_series(df["High"]).astype(float)
    low = _coerce_series(df["Low"]).astype(float)
    idx = df.index

    psar = pd.Series(index=idx, dtype=float)
    in_up = pd.Series(index=idx, dtype=bool)

    in_up.iloc[0] = True
    psar.iloc[0] = low.iloc[0]
    ep = high.iloc[0]
    af = step

    for i in range(1, len(idx)):
        prev = psar.iloc[i-1]
        if in_up.iloc[i-1]:
            psar.iloc[i] = prev + af*(ep - prev)
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + step, max_step)
            if low.iloc[i] < psar.iloc[i]:
                in_up.iloc[i] = False
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = step
            else:
                in_up.iloc[i] = True
        else:
            psar.iloc[i] = prev + af*(ep - prev)
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i >= 2 else high.iloc[i-1])
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + step, max_step)
            if high.iloc[i] > psar.iloc[i]:
                in_up.iloc[i] = True
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = step
            else:
                in_up.iloc[i] = False

    return pd.DataFrame({"PSAR": psar, "in_uptrend": in_up})
# ---------------------------
# BB Cross detection + gating
# ---------------------------
def _cross_up_dn(price: pd.Series, line: pd.Series):
    p = _coerce_series(price)
    l = _coerce_series(line).reindex(p.index)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)
    p = p[ok]; l = l[ok]
    above = p > l
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    return cross_up.reindex(price.index, fill_value=False), cross_dn.reindex(price.index, fill_value=False)

def _n_consecutive_increasing(series: pd.Series, n: int) -> bool:
    s = _coerce_series(series).dropna()
    if len(s) < n + 1:
        return False
    return bool(np.all(np.diff(s.iloc[-(n+1):]) > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int) -> bool:
    s = _coerce_series(series).dropna()
    if len(s) < n + 1:
        return False
    return bool(np.all(np.diff(s.iloc[-(n+1):]) < 0))

def find_bb_mid_cross_after_extreme(close: pd.Series,
                                   bb_mid: pd.Series,
                                   bb_pctb: pd.Series,
                                   horizon: int,
                                   eps: float,
                                   bars_confirm: int):
    c = _coerce_series(close)
    mid = _coerce_series(bb_mid).reindex(c.index)
    pctb = _coerce_series(bb_pctb).reindex(c.index)
    ok = c.notna() & mid.notna() & pctb.notna()
    if ok.sum() < 3:
        return None
    c = c[ok]; mid = mid[ok]; pctb = pctb[ok]

    cross_up, cross_dn = _cross_up_dn(c, mid)
    hz = max(1, int(horizon))
    eps = float(max(0.0, min(0.49, eps)))
    bc = max(1, int(bars_confirm))

    def _touch_before(t_cross, want_buy: bool):
        loc = c.index.get_indexer([t_cross])[0]
        j0 = max(0, loc - hz)
        w = pctb.iloc[j0:loc+1]
        if want_buy:
            m = (w <= eps)
        else:
            m = (w >= 1.0 - eps)
        if not m.any():
            return None
        return m[m].index[-1]

    buy_tr = None
    if cross_up.any():
        t_cross = cross_up[cross_up].index[-1]
        t_touch = _touch_before(t_cross, True)
        if t_touch is not None and _n_consecutive_increasing(c.loc[:t_cross], bc):
            buy_tr = {
                "side": "BUY",
                "touch_time": t_touch,
                "cross_time": t_cross,
                "cross_price": float(c.loc[t_cross]),
            }

    sell_tr = None
    if cross_dn.any():
        t_cross = cross_dn[cross_dn].index[-1]
        t_touch = _touch_before(t_cross, False)
        if t_touch is not None and _n_consecutive_decreasing(c.loc[:t_cross], bc):
            sell_tr = {
                "side": "SELL",
                "touch_time": t_touch,
                "cross_time": t_cross,
                "cross_price": float(c.loc[t_cross]),
            }

    if buy_tr is None and sell_tr is None:
        return None
    if buy_tr is None:
        return sell_tr
    if sell_tr is None:
        return buy_tr
    return buy_tr if buy_tr["cross_time"] >= sell_tr["cross_time"] else sell_tr

def gate_bb_cross(trig: dict,
                  close: pd.Series,
                  sup: pd.Series,
                  res: pd.Series,
                  global_slope: float,
                  local_slope: float,
                  rev_prob: float,
                  prox: float):
    if trig is None:
        return None
    side = str(trig.get("side", "")).upper().strip()
    if side not in ("BUY","SELL"):
        return None
    if not (np.isfinite(global_slope) and np.isfinite(local_slope)):
        return None

    # slope agreement gate
    if side == "BUY":
        if not (global_slope > 0 and local_slope > 0):
            return None
    else:
        if not (global_slope < 0 and local_slope < 0):
            return None

    # 99.9% confidence gate
    if (not np.isfinite(rev_prob)) or (rev_prob > BB_CROSS_MAX_REVPROB):
        return None

    c = _coerce_series(close).astype(float)
    if c.empty:
        return None
    s_sup = _coerce_series(sup).reindex(c.index).ffill()
    s_res = _coerce_series(res).reindex(c.index).ffill()

    t_touch = trig.get("touch_time")
    if t_touch is None or t_touch not in c.index:
        return None

    px = float(c.loc[t_touch])
    S = float(s_sup.loc[t_touch]) if np.isfinite(s_sup.loc[t_touch]) else np.nan
    R = float(s_res.loc[t_touch]) if np.isfinite(s_res.loc[t_touch]) else np.nan
    if not np.all(np.isfinite([px,S,R])):
        return None

    if side == "BUY":
        if not (px <= S*(1+prox)):
            return None
    else:
        if not (px >= R*(1-prox)):
            return None

    out = dict(trig)
    out["rev_prob"] = float(rev_prob)
    out["global_slope"] = float(global_slope)
    out["local_slope"] = float(local_slope)
    return out

def annotate_bb_cross(ax, trig: dict):
    if trig is None:
        return
    side = trig.get("side")
    t0 = trig.get("touch_time")
    t1 = trig.get("cross_time")
    p1 = trig.get("cross_price")
    if t0 is None or t1 is None or not np.isfinite(p1):
        return
    col = "tab:green" if side == "BUY" else "tab:red"
    lbl = "BB Buy Cross" if side == "BUY" else "BB Sell Cross"
    try:
        ax.annotate("", xy=(t1, p1), xytext=(t0, p1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0, alpha=0.85),
                    zorder=9)
    except Exception:
        pass
    ax.scatter([t1],[p1], s=95, color=col, zorder=10, label=lbl)
    ax.text(t1, p1, f"  {lbl}", color=col, fontsize=9, fontweight="bold",
            va="bottom" if side=="BUY" else "top", zorder=10)

# ---------------------------
# NTD overlays (NameError fix: defined at top-level)
# ---------------------------
def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_series(ntd).dropna()
    if s.empty or not np.isfinite(trend_slope) or trend_slope == 0:
        return
    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    cross_up0 = (s >= 0.0) & (s.shift(1) < 0.0)
    cross_dn0 = (s <= 0.0) & (s.shift(1) > 0.0)

    cross_out_hi = (s >= upper) & (s.shift(1) < upper)
    cross_out_lo = (s <= lower) & (s.shift(1) > lower)

    if uptrend:
        idx0 = list(cross_up0[cross_up0].index)
        idxL = list(cross_out_lo[cross_out_lo].index)
        if idx0:
            ax.scatter(idx0, [0.0]*len(idx0), marker="^", s=90, color="tab:green", zorder=10, label="NTD 0↑")
        if idxL:
            ax.scatter(idxL, s.loc[idxL], marker="^", s=80, color="tab:green", zorder=10, label="NTD < -0.75")
    if downtrend:
        idx0 = list(cross_dn0[cross_dn0].index)
        idxH = list(cross_out_hi[cross_out_hi].index)
        if idx0:
            ax.scatter(idx0, [0.0]*len(idx0), marker="v", s=90, color="tab:red", zorder=10, label="NTD 0↓")
        if idxH:
            ax.scatter(idxH, s.loc[idxH], marker="v", s=80, color="tab:red", zorder=10, label="NTD > +0.75")

# ---------------------------
# Forecast (SARIMAX)
# ---------------------------
@st.cache_data(ttl=300)
def compute_sarimax_forecast(close: pd.Series):
    s = _coerce_series(close).dropna()
    if len(s) < 50:
        return None
    try:
        model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception:
        model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(s.index[-1] + timedelta(days=1), periods=30, freq="D", tz=s.index.tz)
    return idx, fc.predicted_mean, fc.conf_int()
# ---------------------------
# Plotting
# ---------------------------
def _legend_outside(ax, loc="upper left"):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc=loc, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, frameon=False)

def render_daily_views(sel: str, alert_placeholder=None):
    """
    Returns a dict with computed metrics used by scanners:
      - m_global, m_local, r2_local, rev_prob, bb_cross(dict|None)
    """
    ohlc = fetch_daily_ohlc(sel)
    if ohlc is None or ohlc.empty:
        st.warning("No daily data.")
        return None

    close = ohlc["Close"]
    close_show = subset_daily_view(close, daily_view)
    ohlc_show = ohlc.reindex(close_show.index).dropna()

    if ohlc_show.empty:
        st.warning("No daily data (after view filter).")
        return None

    close_show = ohlc_show["Close"]

    # indicators
    bb_mid, bb_up, bb_lo, bb_pctb = compute_bbands(close_show, bb_win, bb_mult)
    hma = compute_hma(close_show, hma_period) if show_hma else None
    kij = ichimoku_kijun(ohlc_show["High"], ohlc_show["Low"], ichi_base) if show_ichi else None

    # regression: global = entire view, local = last slope_lb_daily bars
    reg_g, up_g, lo_g, m_g, r2_g = regression_with_band(close_show, lookback=len(close_show), z=2.0)
    reg_l, up_l, lo_l, m_l, r2_l = regression_with_band(close_show, lookback=slope_lb_daily, z=2.0)

    # reversal probability for current slope sign
    rev_prob = slope_reversal_probability(
        close_show, current_slope=m_l, hist_window=rev_hist_lb,
        slope_window=max(5, slope_lb_daily), horizon=rev_horizon
    )

    # S/R + BB Cross gating
    sup, res = rolling_support_resistance(close_show, sr_lb_daily)
    trig_raw = find_bb_mid_cross_after_extreme(
        close_show, bb_mid, bb_pctb,
        horizon=rev_horizon, eps=0.02, bars_confirm=bars_confirm
    )
    trig = gate_bb_cross(
        trig_raw, close_show, sup, res,
        global_slope=m_g, local_slope=m_l,
        rev_prob=rev_prob, prox=sr_prox_pct
    )

    # ---- plot
    nrows = 2 if show_ntd else 1
    fig_h = 7.5 if show_ntd else 5.4
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, fig_h), sharex=True)
    ax1 = axes if nrows == 1 else axes[0]
    style_axes(ax1)

    ax1.plot(close_show.index, close_show.values, linewidth=1.7, label="Close")
    label_on_left(ax1, float(close_show.iloc[-1]), f"{sel}  Close: {fmt_price(close_show.iloc[-1])}")

    # regression bands
    if not reg_g.empty:
        ax1.plot(reg_g.index, reg_g.values, linewidth=1.6, linestyle="--", label=f"Global trend (m={fmt_slope(m_g)}, R²={fmt_r2(r2_g)})")
    if not reg_l.empty:
        ax1.plot(reg_l.index, reg_l.values, linewidth=1.4, linestyle="-.", label=f"Local slope (m={fmt_slope(m_l)}, R²={fmt_r2(r2_l)})")
        ax1.plot(up_l.index, up_l.values, linewidth=1.1, alpha=0.75, label="Local +2σ")
        ax1.plot(lo_l.index, lo_l.values, linewidth=1.1, alpha=0.75, label="Local -2σ")

    # BBands
    if show_bbands:
        ax1.plot(bb_mid.index, bb_mid.values, linewidth=1.2, label=f"BB mid (SMA {bb_win})")
        ax1.plot(bb_up.index, bb_up.values, linewidth=0.9, alpha=0.7, label="BB upper")
        ax1.plot(bb_lo.index, bb_lo.values, linewidth=0.9, alpha=0.7, label="BB lower")

    # HMA / Kijun
    if show_hma and hma is not None:
        ax1.plot(hma.index, hma.values, linewidth=1.2, label=f"HMA({hma_period})")
    if show_ichi and kij is not None:
        ax1.plot(kij.index, kij.values, linewidth=1.2, label=f"Kijun({ichi_base})")

    # Support/Resistance
    ax1.plot(sup.index, sup.values, linewidth=1.0, alpha=0.8, label=f"Support (lb={sr_lb_daily})")
    ax1.plot(res.index, res.values, linewidth=1.0, alpha=0.8, label=f"Resistance (lb={sr_lb_daily})")

    # Fibonacci
    if show_fibs:
        fibs = fibonacci_levels(close_show)
        for name, y in fibs.items():
            ax1.axhline(y, linewidth=0.8, alpha=0.35)
        if fibs:
            ax1.text(close_show.index[0], list(fibs.values())[0], "Fib levels", fontsize=8, alpha=0.7)

    # BB Cross annotation
    if trig is not None:
        annotate_bb_cross(ax1, trig)
        if alert_placeholder is not None:
            alert_placeholder.success(
                f"{sel}: {trig['side']} — BB Cross confirmed | rev_prob={trig['rev_prob']:.4f} | "
                f"m_global={trig['global_slope']:.6f}, m_local={trig['local_slope']:.6f}"
            )

    # metrics box
    txt = f"m_global={fmt_slope(m_g)} | m_local={fmt_slope(m_l)} | R²(local)={fmt_r2(r2_l)} | rev_prob={rev_prob if np.isfinite(rev_prob) else np.nan:.4f}"
    ax1.set_title(f"Daily — {sel}   ({txt})", loc="left", fontsize=12, fontweight="bold")

    # NTD panel
    if show_ntd:
        ax2 = axes[1]
        style_axes(ax2)
        ntd = compute_ntd(close_show, ntd_window)
        npx = compute_npx(close_show, ntd_window)
        ax2.plot(ntd.index, ntd.values, linewidth=1.2, label="NTD")
        ax2.plot(npx.index, npx.values, linewidth=1.0, alpha=0.85, label="NPX")
        ax2.axhline(0.75, linewidth=1.0, alpha=0.6)
        ax2.axhline(-0.75, linewidth=1.0, alpha=0.6)
        ax2.axhline(0.0, linewidth=0.9, alpha=0.55)
        overlay_ntd_triangles_by_trend(ax2, ntd, trend_slope=m_l, upper=0.75, lower=-0.75)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_title("NTD/NPX (tanh-normalized)", loc="left", fontsize=11, fontweight="bold")
        _legend_outside(ax2, loc="upper left")

    _legend_outside(ax1, loc="upper left")
    st.pyplot(fig, clear_figure=True)

    return {
        "m_global": m_g, "m_local": m_l,
        "r2_local": r2_l, "rev_prob": rev_prob,
        "bb_cross": trig
    }
def render_hourly_views(sel: str, alert_placeholder=None):
    """
    Returns dict with computed metrics used by scanners:
      - m_global, m_local, r2_local, rev_prob, bb_cross(dict|None)
    """
    df = fetch_intraday_ohlc(sel, period=hour_period, interval=interval)
    if df is None or df.empty or "Close" not in df.columns:
        st.warning("No intraday data.")
        return None

    close = df["Close"].dropna()
    if close.empty:
        st.warning("No intraday close.")
        return None

    bb_mid, bb_up, bb_lo, bb_pctb = compute_bbands(close, bb_win, bb_mult)
    hma = compute_hma(close, hma_period) if show_hma else None

    reg_g, up_g, lo_g, m_g, r2_g = regression_with_band(close, lookback=len(close), z=2.0)
    reg_l, up_l, lo_l, m_l, r2_l = regression_with_band(close, lookback=slope_lb_hourly, z=2.0)

    rev_prob = slope_reversal_probability(
        close, current_slope=m_l, hist_window=rev_hist_lb,
        slope_window=max(5, slope_lb_hourly), horizon=min(rev_horizon, max(3, int(0.2*len(close))))
    )

    sup, res = rolling_support_resistance(close, sr_lb_hourly)
    trig_raw = find_bb_mid_cross_after_extreme(
        close, bb_mid, bb_pctb,
        horizon=min(rev_horizon, max(3, int(0.2*len(close)))), eps=0.02, bars_confirm=bars_confirm
    )
    trig = gate_bb_cross(
        trig_raw, close, sup, res,
        global_slope=m_g, local_slope=m_l,
        rev_prob=rev_prob, prox=sr_prox_pct
    )

    st_df = None
    psar_df = None
    if show_supertrend:
        try:
            st_df = compute_supertrend(df, atr_period, atr_mult)
        except Exception:
            st_df = None
    if show_psar:
        try:
            psar_df = compute_psar(df, psar_step, psar_max)
        except Exception:
            psar_df = None

    nrows = 2 if show_ntd else 1
    fig_h = 7.5 if show_ntd else 5.4
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12.5, fig_h), sharex=True)
    ax1 = axes if nrows == 1 else axes[0]
    style_axes(ax1)

    ax1.plot(close.index, close.values, linewidth=1.6, label="Close")
    label_on_left(ax1, float(close.iloc[-1]), f"{sel}  Close: {fmt_price(close.iloc[-1])}")

    if not reg_g.empty:
        ax1.plot(reg_g.index, reg_g.values, linewidth=1.4, linestyle="--", label=f"Global trend (m={fmt_slope(m_g)}, R²={fmt_r2(r2_g)})")
    if not reg_l.empty:
        ax1.plot(reg_l.index, reg_l.values, linewidth=1.3, linestyle="-.", label=f"Local slope (m={fmt_slope(m_l)}, R²={fmt_r2(r2_l)})")
        ax1.plot(up_l.index, up_l.values, linewidth=1.0, alpha=0.75, label="Local +2σ")
        ax1.plot(lo_l.index, lo_l.values, linewidth=1.0, alpha=0.75, label="Local -2σ")

    if show_bbands:
        ax1.plot(bb_mid.index, bb_mid.values, linewidth=1.1, label=f"BB mid (SMA {bb_win})")
        ax1.plot(bb_up.index, bb_up.values, linewidth=0.9, alpha=0.7, label="BB upper")
        ax1.plot(bb_lo.index, bb_lo.values, linewidth=0.9, alpha=0.7, label="BB lower")

    if show_hma and hma is not None:
        ax1.plot(hma.index, hma.values, linewidth=1.1, label=f"HMA({hma_period})")

    if st_df is not None and not st_df.empty:
        ax1.plot(st_df.index, st_df["ST"].values, linewidth=1.1, label="Supertrend")

    if psar_df is not None and not psar_df.empty:
        ax1.scatter(psar_df.index, psar_df["PSAR"].values, s=10, alpha=0.8, label="PSAR")

    ax1.plot(sup.index, sup.values, linewidth=1.0, alpha=0.8, label=f"Support (lb={sr_lb_hourly})")
    ax1.plot(res.index, res.values, linewidth=1.0, alpha=0.8, label=f"Resistance (lb={sr_lb_hourly})")

    if trig is not None:
        annotate_bb_cross(ax1, trig)
        if alert_placeholder is not None:
            alert_placeholder.success(
                f"{sel}: {trig['side']} — BB Cross confirmed | rev_prob={trig['rev_prob']:.4f} | "
                f"m_global={trig['global_slope']:.6f}, m_local={trig['local_slope']:.6f}"
            )

    txt = f"m_global={fmt_slope(m_g)} | m_local={fmt_slope(m_l)} | R²(local)={fmt_r2(r2_l)} | rev_prob={rev_prob if np.isfinite(rev_prob) else np.nan:.4f}"
    ax1.set_title(f"Hourly — {sel}  ({hour_period}/{interval})   ({txt})", loc="left", fontsize=12, fontweight="bold")

    if show_ntd:
        ax2 = axes[1]
        style_axes(ax2)
        ntd_h = compute_ntd(close, ntd_window)
        npx_h = compute_npx(close, ntd_window)
        ax2.plot(ntd_h.index, ntd_h.values, linewidth=1.2, label="NTD")
        ax2.plot(npx_h.index, npx_h.values, linewidth=1.0, alpha=0.85, label="NPX")
        ax2.axhline(0.75, linewidth=1.0, alpha=0.6)
        ax2.axhline(-0.75, linewidth=1.0, alpha=0.6)
        ax2.axhline(0.0, linewidth=0.9, alpha=0.55)

        # NameError fix target — this function is defined globally above
        overlay_ntd_triangles_by_trend(ax2, ntd_h, trend_slope=m_l, upper=0.75, lower=-0.75)

        ax2.set_ylim(-1.05, 1.05)
        ax2.set_title("NTD/NPX (tanh-normalized)", loc="left", fontsize=11, fontweight="bold")
        _legend_outside(ax2, loc="upper left")

    _legend_outside(ax1, loc="upper left")
    st.pyplot(fig, clear_figure=True)

    return {
        "m_global": m_g, "m_local": m_l,
        "r2_local": r2_l, "rev_prob": rev_prob,
        "bb_cross": trig
    }

# ---------------------------
# Scanners
# ---------------------------
def run_bb_cross_scanner(symbols: list[str], scope: str = "daily"):
    rows = []
    prog = st.progress(0.0)
    for i, sym in enumerate(symbols):
        try:
            if scope == "daily":
                ohlc = fetch_daily_ohlc(sym)
                if ohlc is None or ohlc.empty:
                    continue
                close = subset_daily_view(ohlc["Close"], daily_view)
                bb_mid, bb_up, bb_lo, bb_pctb = compute_bbands(close, bb_win, bb_mult)
                reg_g, _, _, m_g, _ = regression_with_band(close, lookback=len(close), z=2.0)
                _, _, _, m_l, r2_l = regression_with_band(close, lookback=slope_lb_daily, z=2.0)
                rev_prob = slope_reversal_probability(close, m_l, rev_hist_lb, max(5, slope_lb_daily), rev_horizon)
                sup, res = rolling_support_resistance(close, sr_lb_daily)
                trig_raw = find_bb_mid_cross_after_extreme(close, bb_mid, bb_pctb, rev_horizon, 0.02, bars_confirm)
                trig = gate_bb_cross(trig_raw, close, sup, res, m_g, m_l, rev_prob, sr_prox_pct)
            else:
                df = fetch_intraday_ohlc(sym, period=hour_period, interval=interval)
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].dropna()
                bb_mid, bb_up, bb_lo, bb_pctb = compute_bbands(close, bb_win, bb_mult)
                reg_g, _, _, m_g, _ = regression_with_band(close, lookback=len(close), z=2.0)
                _, _, _, m_l, r2_l = regression_with_band(close, lookback=slope_lb_hourly, z=2.0)
                hz = min(rev_horizon, max(3, int(0.2*len(close))))
                rev_prob = slope_reversal_probability(close, m_l, rev_hist_lb, max(5, slope_lb_hourly), hz)
                sup, res = rolling_support_resistance(close, sr_lb_hourly)
                trig_raw = find_bb_mid_cross_after_extreme(close, bb_mid, bb_pctb, hz, 0.02, bars_confirm)
                trig = gate_bb_cross(trig_raw, close, sup, res, m_g, m_l, rev_prob, sr_prox_pct)

            if trig is not None:
                rows.append({
                    "Symbol": sym,
                    "Side": trig["side"],
                    "Cross time": trig["cross_time"],
                    "Cross price": trig["cross_price"],
                    "rev_prob": trig["rev_prob"],
                    "m_global": trig["global_slope"],
                    "m_local": trig["local_slope"],
                    "R2_local": r2_l
                })
        except Exception:
            # keep scanner robust
            pass
        prog.progress((i+1)/max(1, len(symbols)))
    prog.empty()
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["Side","Cross time"], ascending=[True, False])
    return out

def run_r2_scanner(symbols: list[str], scope: str = "daily", r2_threshold: float = 0.45):
    up_rows, dn_rows = [], []
    prog = st.progress(0.0)
    for i, sym in enumerate(symbols):
        try:
            if scope == "daily":
                ohlc = fetch_daily_ohlc(sym)
                if ohlc is None or ohlc.empty:
                    continue
                close = subset_daily_view(ohlc["Close"], daily_view)
                _, _, _, m_l, r2_l = regression_with_band(close, lookback=slope_lb_daily, z=2.0)
            else:
                df = fetch_intraday_ohlc(sym, period=hour_period, interval=interval)
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].dropna()
                _, _, _, m_l, r2_l = regression_with_band(close, lookback=slope_lb_hourly, z=2.0)

            if not (np.isfinite(r2_l) and r2_l >= r2_threshold and np.isfinite(m_l) and m_l != 0):
                continue

            row = {"Symbol": sym, "m_local": m_l, "R2_local": r2_l}
            if m_l > 0:
                up_rows.append(row)
            else:
                dn_rows.append(row)
        except Exception:
            pass
        prog.progress((i+1)/max(1, len(symbols)))
    prog.empty()
    up = pd.DataFrame(up_rows).sort_values(by=["R2_local"], ascending=False) if up_rows else pd.DataFrame(columns=["Symbol","m_local","R2_local"])
    dn = pd.DataFrame(dn_rows).sort_values(by=["R2_local"], ascending=False) if dn_rows else pd.DataFrame(columns=["Symbol","m_local","R2_local"])
    return up, dn
# ---------------------------
# Main UI
# ---------------------------
tabs = st.tabs([
    "📌 Charts",
    "🔮 Forecast (Daily SARIMAX)",
    "✅ BB Cross Scanner",
    "📏 R² Scanner (≥45%)",
])

with tabs[0]:
    left, right = st.columns([1, 2], gap="large")
    with left:
        sel = st.selectbox("Ticker:", universe, index=0, key="main_ticker")
        st.session_state.ticker = sel

        st.markdown("### Run")
        colA, colB = st.columns(2)
        if colA.button("Run Daily", use_container_width=True):
            st.session_state.run_all = False
            st.session_state.run_daily = True
            st.session_state.run_hourly = False
        if colB.button("Run Hourly", use_container_width=True):
            st.session_state.run_all = False
            st.session_state.run_daily = False
            st.session_state.run_hourly = True

        colC, colD = st.columns(2)
        if colC.button("Run BOTH", use_container_width=True):
            st.session_state.run_all = True
            st.session_state.run_daily = True
            st.session_state.run_hourly = True
        if colD.button("Stop", use_container_width=True):
            st.session_state.run_all = False
            st.session_state.run_daily = False
            st.session_state.run_hourly = False

        st.markdown("---")
        st.caption("**BB Cross gating:** Global slope and Local slope must agree with the cross direction, and reversal probability must be ≤ 0.001 (99.9% confidence), plus S/R proximity check.")

    with right:
        alert_box = st.empty()
        if st.session_state.get("run_daily", False):
            render_daily_views(sel=st.session_state.ticker, alert_placeholder=alert_box)
        if st.session_state.get("run_hourly", False):
            render_hourly_views(sel=st.session_state.ticker, alert_placeholder=alert_box)

with tabs[1]:
    sel = st.selectbox("Ticker for forecast:", universe, index=0, key="fc_ticker")
    if st.button("Run Forecast", use_container_width=True):
        s = fetch_daily_close(sel)
        s_show = subset_daily_view(s, daily_view)
        res = compute_sarimax_forecast(s_show)
        if res is None:
            st.warning("Not enough data for forecast.")
        else:
            idx, mean, ci = res
            fig, ax = plt.subplots(figsize=(12.5, 5.2))
            style_axes(ax)
            ax.plot(s_show.index, s_show.values, linewidth=1.6, label="Close (history)")
            ax.plot(idx, mean.values, linewidth=1.6, label="Forecast (30d)")
            try:
                ax.fill_between(idx, ci.iloc[:,0].values, ci.iloc[:,1].values, alpha=0.22, label="Conf. interval")
            except Exception:
                pass
            ax.set_title(f"SARIMAX forecast — {sel}", loc="left", fontsize=12, fontweight="bold")
            _legend_outside(ax, loc="upper left")
            st.pyplot(fig, clear_figure=True)

with tabs[2]:
    st.subheader("✅ BB Buy/Sell Cross Scanner")
    st.caption("Lists symbols whose **latest** BB Mid cross meets the gating rules (global+local slope agreement, S/R proximity, and ≥99.9% reversal confidence).")
    cA, cB = st.columns(2)
    scope = cA.radio("Scope:", ["daily","hourly"], horizontal=True)
    if cB.button("Run scanner", use_container_width=True):
        df = run_bb_cross_scanner(universe, scope=scope)
        if df.empty:
            st.info("No signals found under the current gating rules.")
        else:
            st.dataframe(df, use_container_width=True)

with tabs[3]:
    st.subheader("📏 R² Scanner (≥45%)")
    st.caption("Shows symbols where **local** regression R² ≥ 45% (0.45) for an uptrend or downtrend.")
    cA, cB = st.columns(2)
    scope = cA.radio("Scope:", ["daily","hourly"], horizontal=True, key="r2_scope")
    r2_thr = cB.slider("R² threshold", 0.10, 0.95, 0.45, 0.05)
    if st.button("Run R² scanner", use_container_width=True):
        up, dn = run_r2_scanner(universe, scope=scope, r2_threshold=r2_thr)
        u1, u2 = st.columns(2)
        with u1:
            st.markdown("### Uptrend (m>0)")
            st.dataframe(up, use_container_width=True)
        with u2:
            st.markdown("### Downtrend (m<0)")
            st.dataframe(dn, use_container_width=True)
