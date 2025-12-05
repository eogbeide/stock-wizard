# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATED — BUY/SELL only when: 2σ bounce → HMA(55) cross near S/R with slope alignment.
# Charts show entry marker, TAKE PROFIT line/label, and pips/Δ between them.

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
    .css-18e3th9 { transform: none !important; visibility: visible !important; width: 100% !important;
                   position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh (PST) ---
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
elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# ---------- Core helpers ----------
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
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.4f}" if np.isfinite(mv) else "n/a"

def fmt_r2(r2: float, digits: int = 1) -> str:
    try:
        rv = float(r2)
    except Exception:
        return "n/a"
    return fmt_pct(rv, digits=digits) if np.isfinite(rv) else "n/a"

# FX helpers
def pip_size_for_symbol(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return None
    return 0.01 if "JPY" in s else 0.0001

def _diff_text(a: float, b: float, symbol: str) -> str:
    try:
        av = float(a); bv = float(b)
    except Exception:
        return ""
    ps = pip_size_for_symbol(symbol)
    diff = abs(bv - av)
    if ps:
        return f"{diff/ps:.1f} pips"
    return f"Δ {diff:.3f}"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans,
        ha="left", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        zorder=6
    )

def label_on_right(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.99, y_val, text, transform=trans,
        ha="right", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        zorder=6
    )

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

# --- slope alignment helper ---
def slopes_aligned(local_slope: float, global_slope: float) -> bool:
    try:
        s1 = np.sign(float(local_slope))
        s2 = np.sign(float(global_slope))
    except Exception:
        return False
    return np.isfinite(s1) and np.isfinite(s2) and (s1 != 0.0) and (s1 == s2)
# ---------- Sidebar configuration ----------
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"],
                                  index=2, key="sb_daily_view")
show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

# Slope reversal probability controls
st.sidebar.subheader("Slope Reversal Probability (experimental)")
rev_hist_lb = st.sidebar.slider("History window for reversal stats (bars)", 30, 720, 240, 30, key="sb_rev_hist_lb")
rev_horizon = st.sidebar.slider("Forward horizon for reversal (bars)", 3, 60, 15, 1, key="sb_rev_horizon")

# Daily/Hourly S/R windows
st.sidebar.subheader("Daily Support/Resistance Window")
sr_lb_daily = st.sidebar.slider("Daily S/R lookback (bars)", 20, 252, 60, 5, key="sb_sr_lb_daily")
st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

# Hourly Indicator Panel (NTD)
st.sidebar.subheader("Hourly Indicator Panel")
show_nrsi = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")

st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox("Highlight when price is between S/R (S↔R) on NTD",
                                       value=True, key="sb_ntd_channel")

# Hourly Supertrend
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# Parabolic SAR
st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

# Signal Logic
st.sidebar.subheader("Signal Logic")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# NTD (Daily/Hourly)
st.sidebar.subheader("NTD (Daily/Hourly)")
show_ntd  = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window= st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
show_npx_ntd   = st.sidebar.checkbox("Overlay normalized price (NPX) on NTD", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses (dots)", value=True, key="sb_mark_npx_cross")

# Ichimoku
st.sidebar.subheader("Normalized Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

# Bollinger Bands
st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

# Probabilistic HMA Crossover
st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

# HMA(55) Reversal on NTD
st.sidebar.subheader("HMA(55) Reversal on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb       = st.sidebar.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

# Forex-only extras
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
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
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI',
        'ORCL','TLT'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
    ]

# ---------- Data fetchers ----------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
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
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {"0%": hi, "23.6%": hi - 0.236*diff, "38.2%": hi - 0.382*diff, "50%": hi - 0.5*diff,
            "61.8%": hi - 0.618*diff, "78.6%": hi - 0.786*diff, "100%": lo}

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {'High','Low','Close'}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L; S1 = 2 * P - H
    R2 = P + (H - L); S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

# ---------- Regression & ±2σ band ----------
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

def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
    if s.shape[0] < 3:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty, float("nan"), float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    std = float(np.sqrt(np.sum(resid**2) / dof))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res/ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

# --- Bollinger Bands helper (SMA/EMA midline) ---
def compute_bbands(price: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    p = _coerce_1d_series(price).astype(float)
    if p.empty or window < 1:
        idx = p.index
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty, empty

    if use_ema:
        mid = p.ewm(span=int(window), adjust=False).mean()
    else:
        mid = p.rolling(int(window), min_periods=1).mean()

    std = p.rolling(int(window), min_periods=1).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std

    rng = (upper - lower).replace(0, np.nan)
    pct_b = (p - lower) / rng
    nbb = (p - mid) / (mult * std.replace(0, np.nan))

    return mid.reindex(p.index), upper.reindex(p.index), lower.reindex(p.index), pct_b.reindex(p.index), nbb.reindex(p.index)

# --- empirical slope reversal probability helper ---
def slope_reversal_probability(series_like, current_slope: float, hist_window: int = 240,
                               slope_window: int = 60, horizon: int = 15) -> float:
    s = _coerce_1d_series(series_like).dropna()
    n = len(s)
    if n < slope_window + horizon + 5:
        return float("nan")
    try:
        sign_curr = np.sign(float(current_slope))
    except Exception:
        return float("nan")
    if not np.isfinite(sign_curr) or sign_curr == 0.0:
        return float("nan")
    start = max(slope_window - 1, n - hist_window - horizon)
    end = n - horizon - 1
    if end <= start:
        return float("nan")
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
        return float("nan")
    return float(flips / match)

# --- bounce helper using ±2σ band with alignment gating ---
def find_band_bounce_signal(price: pd.Series,
                            upper_band: pd.Series,
                            lower_band: pd.Series,
                            local_slope: float,
                            global_slope: float = None,
                            require_align: bool = True):
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    mask = p.notna() & u.notna() & l.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]; u = u.reindex(p.index); l = l.reindex(p.index)

    inside = (p <= u) & (p >= l)
    below  = p < l
    above  = p > u

    try:
        ls = float(local_slope)
    except Exception:
        ls = np.nan
    if not np.isfinite(ls) or ls == 0.0:
        return None

    if require_align and (global_slope is not None):
        if not slopes_aligned(ls, global_slope):
            return None

    if ls > 0:
        candidates = inside & below.shift(1, fill_value=False)
        idx = list(candidates[candidates].index)
        if not idx:
            return None
        t = idx[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    else:
        candidates = inside & above.shift(1, fill_value=False)
        idx = list(candidates[candidates].index)
        if not idx:
            return None
        t = idx[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL"}

# HMA helpers
def _wma(s: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(s).astype(float)
    if s.empty or window < 1:
        return pd.Series(index=s.index, dtype=float)
    w = np.arange(1, window + 1, dtype=float)
    return s.rolling(window, min_periods=window) \
            .apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

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

def _cross_series(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index if len(p) else l.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)
    p = p[ok]; l = l[ok]
    above = p > l
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    return cross_up.reindex(p.index, fill_value=False), cross_dn.reindex(p.index, fill_value=False)

# --- NEW: Combined signal — 2σ bounce → HMA(55) cross near S/R (aligned slopes) ---
def find_bounce_then_hma_sr_signal(price: pd.Series,
                                   hma: pd.Series,
                                   upper_band: pd.Series,
                                   lower_band: pd.Series,
                                   support: pd.Series,
                                   resistance: pd.Series,
                                   local_slope: float,
                                   global_slope: float,
                                   prox: float,
                                   within_bars: int = 8,
                                   require_align: bool = True):
    """
    Returns the most recent aligned trade where:
      - price bounces off ±2σ (re-enters the band),
      - shortly after (<= within_bars) price crosses HMA(55)
      - at the cross, price is near Support (BUY) or near Resistance (SELL)
    Output dict keys: time, price, side, tp_price
    """
    p = _coerce_1d_series(price)
    h = _coerce_1d_series(hma).reindex(p.index)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)
    sup = _coerce_1d_series(support).reindex(p.index)
    res = _coerce_1d_series(resistance).reindex(p.index)
    if p.empty or h.dropna().empty or u.dropna().empty or l.dropna().empty:
        return None

    try:
        ls = float(local_slope)
        gs = float(global_slope)
    except Exception:
        return None
    if not np.isfinite(ls) or ls == 0:
        return None
    if require_align and not slopes_aligned(ls, gs):
        return None

    inside = (p <= u) & (p >= l)
    below  = p < l
    above  = p > u
    cross_up, cross_dn = _cross_series(p, h)

    idx = list(p.index)
    pos = {t:i for i,t in enumerate(idx)}

    if ls > 0:
        # bounce from lower band
        bounce_mask = inside & below.shift(1, fill_value=False)
        b_idx = list(bounce_mask[bounce_mask].index)
        if not b_idx:
            return None
        t_b = b_idx[-1]
        i_b = pos.get(t_b, None)
        if i_b is None:
            return None
        # next cross up within window
        cu = list(cross_up[cross_up].index)
        if not cu:
            return None
        for t_cross in cu:
            i_c = pos.get(t_cross, -10**9)
            if i_c > i_b and (i_c - i_b) <= within_bars:
                px = float(p.loc[t_cross])
                S = sup.loc[t_cross] if t_cross in sup.index else np.nan
                if np.isfinite(S) and px <= S * (1.0 + prox):
                    tp = res.loc[t_cross] if t_cross in res.index else np.nan
                    return {"time": t_cross, "price": px, "side": "BUY", "tp_price": float(tp) if np.isfinite(tp) else np.nan}
        return None
    else:
        # bounce from upper band
        bounce_mask = inside & above.shift(1, fill_value=False)
        b_idx = list(bounce_mask[bounce_mask].index)
        if not b_idx:
            return None
        t_b = b_idx[-1]
        i_b = pos.get(t_b, None)
        if i_b is None:
            return None
        cd = list(cross_dn[cross_dn].index)
        if not cd:
            return None
        for t_cross in cd:
            i_c = pos.get(t_cross, -10**9)
            if i_c > i_b and (i_c - i_b) <= within_bars:
                px = float(p.loc[t_cross])
                R = res.loc[t_cross] if t_cross in res.index else np.nan
                if np.isfinite(R) and px >= R * (1.0 - prox):
                    tp = sup.loc[t_cross] if t_cross in sup.index else np.nan
                    return {"time": t_cross, "price": px, "side": "SELL", "tp_price": float(tp) if np.isfinite(tp) else np.nan}
        return None

# --- Plot helpers for entry + TP + pips text ---
def annotate_crossover(ax, ts, px, side: str, note: str = ""):
    if side == "BUY":
        ax.scatter([ts], [px], marker="P", s=90, color="tab:green", zorder=7)
        label = "BUY" if not note else f"BUY {note}"
        ax.text(ts, px, f"  {label}", va="bottom", fontsize=9,
                color="tab:green", fontweight="bold")
    else:
        ax.scatter([ts], [px], marker="X", s=90, color="tab:red", zorder=7)
        label = "SELL" if not note else f"SELL {note}"
        ax.text(ts, px, f"  {label}", va="top", fontsize=9,
                color="tab:red", fontweight="bold")

def annotate_trade_with_tp(ax, entry_ts, entry_px, side, tp_px, symbol: str):
    """Draw entry marker, TP dashed line + label on the right, and pips text near mid-point."""
    annotate_crossover(ax, entry_ts, entry_px, side, note="(HMA55+2σ+S/R)")
    if not np.isfinite(tp_px):
        return
    ax.axhline(tp_px, linestyle="--", linewidth=1.8,
               color=("tab:red" if side == "BUY" else "tab:green"), alpha=0.75)
    pip_text = _diff_text(entry_px, tp_px, symbol)
    label_on_right(ax, tp_px, f"TP {fmt_price_val(tp_px)}  •  {pip_text}",
                   color=("tab:red" if side == "BUY" else "tab:green"), fontsize=9)
    try:
        y_mid = (entry_px + tp_px) / 2.0
        ax.text(entry_ts, y_mid, f"  {pip_text}", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.65))
    except Exception:
        pass
# ---------- Other indicators ----------
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
    return ((rsi - 50.0) / 50.0).clip(-1.0, 1.0).reindex(rsi.index)

def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9, norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),)*3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    minp = max(10, norm_win//10)
    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)
    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return (nmacd.reindex(s.index),
            nsignal.reindex(s.index),
            nhist.reindex(s.index))

def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win//10)
    m = v.rolling(norm_win, min_periods=minp).mean()
    sd = v.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (v - m) / sd
    return np.tanh(z / 3.0)

def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26,
                norm_win: int = 240) -> pd.Series:
    s = _coerce_1d_series(close)
    if s.empty or fast <= 0 or slow <= 0:
        return pd.Series(index=s.index, dtype=float)
    if fast >= slow:
        fast = max(1, slow - 1)
        if fast >= slow:
            return pd.Series(index=s.index, dtype=float)
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean().replace(0, np.nan)
    ppo = (ema_fast - ema_slow) / ema_slow * 100.0
    minp = max(10, int(norm_win)//10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std  = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
    z = (ppo - mean) / std
    return np.tanh(z / 2.0).reindex(s.index)

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
    return np.tanh(ntd_raw / 2.0).reindex(s.index)

def compute_normalized_price(close: pd.Series, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window // 3)
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    z = (s - m) / sd
    return np.tanh(z / 2.0).reindex(s.index)

def shade_ntd_regions(ax, ntd: pd.Series):
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "tab:green" if m >= 0 else "tab:red"
    ax.plot(s.index, yhat, "--", linewidth=2.4, color=color,
            label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# ---------- Ichimoku, Supertrend, PSAR ----------
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26,
                   span_b: int = 52, shift_cloud: bool = True):
    h = _coerce_1d_series(high)
    l = _coerce_1d_series(low)
    c = _coerce_1d_series(close)
    idx = c.index.union(h.index).union(l.index)
    h = h.reindex(idx); l = l.reindex(idx); c = c.reindex(idx)
    tenkan = ((h.rolling(conv).max() + l.rolling(conv).min()) / 2.0)
    kijun  = ((h.rolling(base).max() + l.rolling(base).min()) / 2.0)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = ((h.rolling(span_b).max() + l.rolling(span_b).min()) / 2.0)
    if shift_cloud:
        senkou_a = senkou_a.shift(base)
        senkou_b = senkou_b.shift(base)
        chikou   = c.shift(-base)
    else:
        chikou   = c
    return (tenkan.reindex(idx), kijun.reindex(idx),
            senkou_a.reindex(idx), senkou_b.reindex(idx), chikou.reindex(idx))

def _compute_atr_from_ohlc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(df['High'])
    low  = _coerce_1d_series(df['Low'])
    close= _coerce_1d_series(df['Close'])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.reindex(df.index)

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return pd.DataFrame(columns=["ST","in_uptrend"])
    ohlc = df[['High','Low','Close']].copy()
    atr = _compute_atr_from_ohlc(ohlc, period=atr_period)
    hl2 = (ohlc['High'] + ohlc['Low']) / 2.0
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr

    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_uptrend = pd.Series(index=ohlc.index, dtype=bool)

    for i in range(len(ohlc)):
        if i == 0:
            in_uptrend.iloc[i] = True
            st_line.iloc[i] = lowerband.iloc[i]
            continue
        if ohlc['Close'].iloc[i] > upperband.iloc[i-1]:
            in_uptrend.iloc[i] = True
        elif ohlc['Close'].iloc[i] < lowerband.iloc[i-1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
            if in_uptrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if (not in_uptrend.iloc[i]) and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
        st_line.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]
    return pd.DataFrame({"ST": st_line, "in_uptrend": in_uptrend})

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {'High','Low'}.issubset(df.columns):
        return pd.DataFrame(columns=["PSAR","in_uptrend"])
    high = _coerce_1d_series(df['High'])
    low  = _coerce_1d_series(df['Low'])
    idx = high.index.union(low.index)
    high = high.reindex(idx); low = low.reindex(idx)
    psar = pd.Series(index=idx, dtype=float)
    in_uptrend = pd.Series(index=idx, dtype=bool)
    in_uptrend.iloc[0] = True
    psar.iloc[0] = float(low.iloc[0])
    ep = float(high.iloc[0])   # extreme point
    af = step                  # acceleration factor
    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i-1]
        if in_uptrend.iloc[i-1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i],
                               float(low.iloc[i-1]),
                               float(low.iloc[i-2]) if i >= 2 else float(low.iloc[i-1]))
            if high.iloc[i] > ep:
                ep = float(high.iloc[i]); af = min(af + step, max_step)
            if low.iloc[i] < psar.iloc[i]:
                in_uptrend.iloc[i] = False; psar.iloc[i] = ep
                ep = float(low.iloc[i]); af = step
            else:
                in_uptrend.iloc[i] = True
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = max(psar.iloc[i],
                               float(high.iloc[i-1]),
                               float(high.iloc[i-2]) if i >= 2 else float(high.iloc[i-1]))
            if low.iloc[i] < ep:
                ep = float(low.iloc[i]); af = min(af + step, max_step)
            if high.iloc[i] > psar.iloc[i]:
                in_uptrend.iloc[i] = True; psar.iloc[i] = ep
                ep = float(high.iloc[i]); af = step
            else:
                in_uptrend.iloc[i] = False
    return pd.DataFrame({"PSAR": psar, "in_uptrend": in_uptrend})

# ---------- Optional overlay helpers (no-ops to keep code complete) ----------
def overlay_hma_reversal_on_ntd(ax, ntd: pd.Series, hma: pd.Series, lookback: int = 3):
    # Placeholder: draw nothing (kept for compatibility)
    return

def overlay_npx_on_ntd(ax, npx: pd.Series, mark_cross: bool = True):
    if npx is None or npx.empty:
        return
    ax.plot(npx.index, npx.values, linewidth=1.0, label="NPX")
    if mark_cross:
        # mark NPX-0 crossings
        above = npx > 0
        cross_up = above & (~above.shift(1).fillna(False))
        cross_dn = (~above) & (above.shift(1).fillna(False))
        up_idx = list(cross_up[cross_up].index)
        dn_idx = list(cross_dn[cross_dn].index)
        if up_idx:
            ax.scatter(up_idx, npx.loc[up_idx], s=15, marker="o")
        if dn_idx:
            ax.scatter(dn_idx, npx.loc[dn_idx], s=15, marker="o")

def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series):
    # Placeholder: draw nothing (kept for compatibility)
    return

def overlay_ntd_sr_reversal_stars(ax, ntd: pd.Series, support: pd.Series, resistance: pd.Series):
    # Placeholder: draw nothing (kept for compatibility)
    return
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

def _price_above_kijun_from_df(df: pd.DataFrame, base: int = 26):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return False, None, np.nan, np.nan
    ohlc = df[['High','Low','Close']].copy()
    _, kijun, _, _, _ = ichimoku_lines(ohlc['High'], ohlc['Low'], ohlc['Close'], base=base)
    kijun = kijun.ffill().bfill().reindex(ohlc.index)
    close = ohlc['Close'].astype(float).reindex(ohlc.index)
    mask = close.notna() & kijun.notna()
    if mask.sum() < 1:
        return False, None, np.nan, np.nan
    c_now = float(close[mask].iloc[-1]); k_now = float(kijun[mask].iloc[-1])
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

# Instruction text helper
def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             use_take_profit: bool = True) -> str:
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    uptrend = False
    try:
        uptrend = float(trend_slope) >= 0.0
    except Exception:
        pass

    if use_take_profit:
        if uptrend:
            leg_a_val, leg_b_val = entry_buy, exit_sell
            text = f"▲ BUY @{fmt_price_val(leg_a_val)} → TAKE PROFIT @{fmt_price_val(leg_b_val)}"
        else:
            leg_a_val, leg_b_val = exit_sell, entry_buy
            text = f"▼ SELL @{fmt_price_val(leg_a_val)} → TAKE PROFIT @{fmt_price_val(leg_b_val)}"
    else:
        if uptrend:
            leg_a_val, leg_b_val = entry_buy, exit_sell
            text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        else:
            leg_a_val, leg_b_val = exit_sell, entry_buy
            text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"

    text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
    return text

# ---------- Session state init ----------
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# ---------- SHARED HOURLY RENDERER (uses combined signal) ----------
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    hc = intraday["Close"].ffill()
    he = hc.ewm(span=20).mean()
    xh = np.arange(len(hc))
    slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
    _ = slope_h * xh + intercept_h

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday.columns else pd.Series(dtype=float)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {'High','Low','Close'}.issubset(intraday.columns) and show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(
            intraday["High"], intraday["Low"], intraday["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
        )
        kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    hma_h = compute_hma(hc, period=hma_period)

    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    psar_h_df = psar_h_df.reindex(hc.index)

    # Local slope & regression band
    yhat_h, upper_h, lower_h, m_local_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    local_slope_h = m_local_h if np.isfinite(m_local_h) else slope_h

    # Figure
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.85, right=0.93)

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(hc.index, he, "--", label="20 EMA")
    m_global_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")
    if not np.isfinite(m_global_h):
        m_global_h = slope_h

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and not psar_h_df.dropna().empty:
        up_mask = psar_h_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red",   zorder=6)

    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    # Combined signal: 2σ bounce → HMA cross near S/R (aligned)
    combo_sig_h = find_bounce_then_hma_sr_signal(
        price=hc, hma=hma_h, upper_band=bb_up_h, lower_band=bb_lo_h,
        support=sup_h, resistance=res_h,
        local_slope=local_slope_h, global_slope=m_global_h,
        prox=sr_prox_pct, within_bars=8, require_align=True
    )
    if combo_sig_h is not None:
        annotate_trade_with_tp(ax2, combo_sig_h["time"], combo_sig_h["price"], combo_sig_h["side"],
                               combo_sig_h.get("tp_price", np.nan), sel)

    # Info / title
    rev_prob_h = slope_reversal_probability(
        hc, local_slope_h, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon
    )
    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_h}]"
    )

    # Current price bubble
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

    # Slope bands
    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_local_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

    if is_forex and show_sessions_pst and not hc.empty:
        from datetime import date  # local import ok
        # session line drawers
        try:
            NY_TZ   = pytz.timezone("America/New_York")
            LDN_TZ  = pytz.timezone("Europe/London")
            def session_markers_for_index(idx, session_tz, open_hr: int, close_hr: int):
                opens, closes = [], []
                if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
                    return opens, closes
                start_d = idx[0].astimezone(session_tz).date()
                end_d   = idx[-1].astimezone(session_tz).date()
                rng = pd.date_range(start=start_d, end=end_d, freq="D")
                lo, hi = idx.min(), idx.max()
                for d in rng:
                    try:
                        dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, 8, 0, 0), is_dst=None)
                        dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, 17,0, 0), is_dst=None)
                    except Exception:
                        dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, 8, 0, 0))
                        dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, 17,0, 0))
                    dt_open_pst  = dt_open_local.astimezone(PACIFIC)
                    dt_close_pst = dt_close_local.astimezone(PACIFIC)
                    if lo <= dt_open_pst  <= hi: opens.append(dt_open_pst)
                    if lo <= dt_close_pst <= hi: closes.append(dt_close_pst)
                return opens, closes
            ldn_open, ldn_close = session_markers_for_index(hc.index, LDN_TZ, 8, 17)
            ny_open, ny_close   = session_markers_for_index(hc.index, NY_TZ,  8, 17)
            for t in ldn_open:  ax2.axvline(t, linestyle="-",  linewidth=1.0, color="tab:blue",   alpha=0.35)
            for t in ldn_close: ax2.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue",   alpha=0.35)
            for t in ny_open:   ax2.axvline(t, linestyle="-",  linewidth=1.0, color="tab:orange", alpha=0.35)
            for t in ny_close:  ax2.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=0.35)
            ax2.text(0.99, 0.98, "Session times in PST", transform=ax2.transAxes,
                     ha="right", va="top", fontsize=8, color="black",
                     bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))
        except Exception:
            pass

    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    ax2.set_xlabel("Time (PST)")
    ax2.legend(loc="lower left", framealpha=0.5)
    st.pyplot(fig2)

    # ---- Hourly indicator panel ----
    if show_nrsi:
        ntd_h = compute_normalized_trend(hc, window=ntd_window)
        ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        fig2r, ax2r = plt.subplots(figsize=(14, 2.8))
        ax2r.set_title(f"Hourly Indicator Panel — NTD + NPX + Trend (win={ntd_window})")
        if shade_ntd and not ntd_h.dropna().empty:
            shade_ntd_regions(ax2r, ntd_h)
        if show_ntd_channel and np.isfinite(res_val) and np.isfinite(sup_val):
            state = (hc < sup_h) * -1 + (hc > res_h) * 1
            mask = (state == 0).fillna(False)
            spans = []
            start = None; prev = None
            for t,v in mask.items():
                if v and start is None: start = t
                if not v and start is not None:
                    if prev is not None: spans.append((start, prev))
                    start = None
                prev = t
            for a,b in spans:
                ax2r.axvspan(a, b, color="gold", alpha=0.15, zorder=1)
            ax2r.plot([], [], linewidth=8, color="gold", alpha=0.20, label="In Range (S↔R)")

        ax2r.plot(ntd_h.index, ntd_h, "-", linewidth=1.6, label="NTD")
        if not npx_h.dropna().empty:
            overlay_npx_on_ntd(ax2r, npx_h, mark_cross=mark_npx_cross)
        ax2r.axhline(0.0,  linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2r.axhline(0.75, linestyle="-",  linewidth=1.0, color="black", label="+0.75")
        ax2r.axhline(-0.75, linestyle="-",  linewidth=1.0, color="black", label="-0.75")
        ax2r.set_ylim(-1.1, 1.1); ax2r.set_xlabel("Time (PST)")
        ax2r.legend(loc="lower left", framealpha=0.5)
        st.pyplot(fig2r)
# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History"
])

# ==================== TAB 1: ORIGINAL FORECAST (uses combined signal) ====================
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for ~2 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:", ["24h", "48h", "96h"],
                              index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
                              key="hour_range_select")
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    auto_run = st.session_state.run_all

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist, "df_ohlc": df_ohlc,
            "fc_idx": fc_idx, "fc_vals": fc_vals, "fc_ci": fc_ci,
            "intraday": intraday, "ticker": sel, "chart": chart,
            "hour_range": hour_range, "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        # ----- DAILY (price + NTD panel) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res_d = df.rolling(sr_lb_daily, min_periods=1).max()
            sup_d = df.rolling(sr_lb_daily, min_periods=1).min()

            yhat_d, upper_d, lower_d, m_local_d, r2_d = regression_with_band(df, slope_lb_daily)
            rev_prob_d = slope_reversal_probability(df, m_local_d, hist_window=rev_hist_lb,
                                                    slope_window=slope_lb_daily, horizon=rev_horizon)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
            df_show = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res_d_show  = res_d.reindex(df_show.index)
            sup_d_show  = sup_d.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show= upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show= lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30

            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show= bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_full = compute_hma(df, period=hma_period)
            hma_d_show = hma_d_full.reindex(df_show.index)

            # Values
            px_val_d = _safe_last_float(df_show)
            try: res_val_d = float(res_d_show.iloc[-1])
            except Exception: res_val_d = np.nan
            try: sup_val_d = float(sup_d_show.iloc[-1])
            except Exception: sup_val_d = np.nan

            fig, (ax, axdw) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3.2, 1.3]})
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            m_global_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)
                try:
                    last_pct = float(bb_pctb_d_show.dropna().iloc[-1])
                    last_nbb = float(bb_nbb_d_show.dropna().iloc[-1])
                    ax.text(0.99, 0.02, f"NBB {last_nbb:+.2f}  |  %B {fmt_pct(last_pct, digits=0)}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=9, color="black",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                except Exception:
                    pass

            if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")

            # Daily slope bands
            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_local_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")

            # NEW: combined daily signal + annotate entry & TP & pips
            combo_sig_d = find_bounce_then_hma_sr_signal(
                price=df_show, hma=hma_d_show, upper_band=bb_up_d_show, lower_band=bb_lo_d_show,
                support=sup_d_show, resistance=res_d_show,
                local_slope=m_local_d, global_slope=m_global_d,
                prox=sr_prox_pct, within_bars=8, require_align=True
            )
            if combo_sig_d is not None:
                annotate_trade_with_tp(ax, combo_sig_d["time"], combo_sig_d["price"], combo_sig_d["side"],
                                       combo_sig_d.get("tp_price", np.nan), sel)

            # Title & extras
            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{sel} Daily — {daily_view} — ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
                f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
            )

            ax.set_ylabel("Price")
            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.legend(loc="lower left", framealpha=0.5)

            # DAILY NTD PANEL (short version)
            axdw.set_title(f"Daily Indicator Panel — NTD (win={ntd_window})")
            if show_ntd:
                ntd_d = compute_normalized_trend(df_show, window=ntd_window)
                if shade_ntd and not ntd_d.dropna().empty:
                    shade_ntd_regions(axdw, ntd_d)
                axdw.plot(ntd_d.index, ntd_d, "-", linewidth=1.6, label=f"NTD")
            axdw.axhline(0.0,  linestyle="--", linewidth=1.0, color="black", label="0.00")
            axdw.axhline(0.75, linestyle="-",  linewidth=1.0, color="black", label="+0.75")
            axdw.axhline(-0.75, linestyle="-",  linewidth=1.0, color="black", label="-0.75")
            axdw.set_ylim(-1.1, 1.1); axdw.set_xlabel("Date (PST)")
            axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- HOURLY (shared) -----
        if chart in ("Hourly","Both"):
            intraday = st.session_state.intraday
            render_hourly_views(sel=sel, intraday=intraday, p_up=p_up, p_dn=p_dn,
                                hour_range_label=st.session_state.hour_range,
                                is_forex=(mode == "Forex"))

# ==================== TAB 2: ENHANCED FORECAST ====================
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci)
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}**")
        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        # ENHANCED DAILY (reuses combined signal)
        if view in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res_d2 = df.rolling(sr_lb_daily, min_periods=1).max()
            sup_d2 = df.rolling(sr_lb_daily, min_periods=1).min()
            yhat_d, up_d, lo_d, m_local_d, r2_d = regression_with_band(df, slope_lb_daily)
            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                      conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d2 = kijun_d2.ffill().bfill()
            bb_mid_d2, bb_up_d2, bb_lo_d2 = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)[:3]
            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res_d2_show = res_d2.reindex(df_show.index)
            sup_d2_show = sup_d2.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            up_d_show = up_d.reindex(df_show.index) if not up_d.empty else up_d
            lo_d_show = lo_d.reindex(df_show.index) if not lo_d.empty else lo_d
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()
            bb_mid_d2_show = bb_mid_d2.reindex(df_show.index)
            bb_up_d2_show  = bb_up_d2.reindex(df_show.index)
            bb_lo_d2_show  = bb_lo_d2.reindex(df_show.index)
            hma_d2_full = compute_hma(df, period=hma_period)
            hma_d2_show = hma_d2_full.reindex(df_show.index)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            m_global_d2 = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            if show_hma and not hma_d2_show.dropna().empty:
                ax.plot(hma_d2_show.index, hma_d2_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(kijun_d2_show.index, kijun_d2_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d2_show.dropna().empty and not bb_lo_d2_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d2_show, bb_up_d2_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d2_show.index, bb_mid_d2_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")

            res_val_d2 = float(res_d2_show.iloc[-1]) if len(res_d2_show) else np.nan
            sup_val_d2 = float(sup_d2_show.iloc[-1]) if len(sup_d2_show) else np.nan
            if np.isfinite(res_val_d2) and np.isfinite(sup_val_d2):
                ax.hlines(res_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d2, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d2, f"R {fmt_price_val(res_val_d2)}", color="tab:red")
                label_on_left(ax, sup_val_d2, f"S {fmt_price_val(sup_val_d2)}", color="tab:green")

            combo_sig_d2 = find_bounce_then_hma_sr_signal(
                price=df_show, hma=hma_d2_show, upper_band=bb_up_d2_show, lower_band=bb_lo_d2_show,
                support=sup_d2_show, resistance=res_d2_show,
                local_slope=m_local_d, global_slope=m_global_d2,
                prox=sr_prox_pct, within_bars=8, require_align=True
            )
            if combo_sig_d2 is not None:
                annotate_trade_with_tp(ax, combo_sig_d2["time"], combo_sig_d2["price"], combo_sig_d2["side"],
                                       combo_sig_d2.get("tp_price", np.nan), st.session_state.ticker)

            ax.set_title(
                f"{st.session_state.ticker} Daily — {daily_view} (Enhanced)"
            )
            ax.legend(loc="lower left", framealpha=0.5)
            ax.set_xlabel("Date (PST)")
            st.pyplot(fig)

        # ENHANCED INTRADAY
        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                st.info("Using the same intraday layout as Tab 1.")
                render_hourly_views(sel=st.session_state.ticker, intraday=intr, p_up=p_up, p_dn=p_dn,
                                    hour_range_label=st.session_state.hour_range, is_forex=(mode == "Forex"))
# ==================== TAB 3: Bull vs Bear ====================
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        pct = df3['Close'].pct_change()
        df3 = df3.assign(PctChange=pct, Bull=pct > 0)
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
        c4.metric("Lookback", bb_period)

# ==================== TAB 4: Metrics ====================
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
        trend3m, up3m, lo3m, m3m, r2_3m = regression_with_band(df3m, lookback=len(df3m))

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(ma30_3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", label="Support")
        if not trend3m.empty:
            ax.plot(trend3m.index, trend3m.values, "--", label=f"Trend (m={fmt_slope(m3m)}/bar)")
        if not up3m.empty and not lo3m.empty:
            ax.plot(up3m.index, up3m.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
            ax.plot(lo3m.index, lo3m.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
        ax.set_xlabel("Date (PST)")
        ax.text(0.50, 0.02, f"R² (3M): {fmt_r2(r2_3m)}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax.legend(); st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0 = df0.assign(PctChange=df0['Close'].pct_change())
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        res0 = df0['Close'].rolling(30, min_periods=1).max()
        sup0 = df0['Close'].rolling(30, min_periods=1).min()
        trend0, up0, lo0, m0, r2_0 = regression_with_band(df0['Close'], lookback=len(df0))

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(res0.index, res0, ":", label="Resistance")
        ax0.plot(sup0.index, sup0, ":", label="Support")
        if not trend0.empty:
            ax0.plot(trend0.index, trend0.values, "--", label=f"Trend (m={fmt_slope(m0)}/bar)")
        if not up0.empty and not lo0.empty:
            ax0.plot(up0.index, up0.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
            ax0.plot(lo0.index, lo0.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
        ax0.set_xlabel("Date (PST)")
        ax0.text(0.50, 0.02, f"R² ({bb_period}): {fmt_r2(r2_0)}", transform=ax0.transAxes,
                 ha="center", va="bottom", fontsize=9, color="black",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax0.legend(); st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({"Type": ["Bull", "Bear"],
                             "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]}).set_index("Type")
        st.bar_chart(dist, use_container_width=True)

# ==================== TAB 5: NTD -0.75 Scanner ====================
with tab5:
    st.header("NTD -0.75 Scanner (NTD < -0.75)")
    st.caption("Scans the universe for symbols whose latest NTD value is below -0.75 on the Daily NTD line (and Hourly NTD for Forex).")
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox("Hourly lookback for Forex:", ["24h", "48h", "96h"],
                                   index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
                                   key="ntd_scan_hour_range")
    scan_period = period_map[scan_hour_range]
    thresh = -0.75
    run = st.button("Scan Universe", key="btn_ntd_scan")

    if run:
        # DAILY scan
        daily_rows = []
        for sym in universe:
            ntd_val, ts = last_daily_ntd_value(sym, ntd_window)
            try:
                s_close = fetch_hist(sym)
                close_val = _safe_last_float(s_close)
            except Exception:
                close_val = np.nan
            daily_rows.append({"Symbol": sym, "NTD_Last": ntd_val,
                               "BelowThresh": (np.isfinite(ntd_val) and ntd_val < thresh),
                               "Close": close_val, "Timestamp": ts})
        df_daily = pd.DataFrame(daily_rows)
        hits_daily = df_daily[df_daily["BelowThresh"] == True].copy().sort_values("NTD_Last")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(f"Daily NTD < {thresh:+.2f}", int(hits_daily.shape[0]))

        st.subheader(f"Daily — latest NTD < {thresh:+.2f}")
        if hits_daily.empty:
            st.info(f"No symbols where the latest daily NTD value is below {thresh:+.2f}.")
        else:
            view = hits_daily.copy()
            view["NTD_Last"] = view["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
            view["Close"] = view["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

        # DAILY: Price > Kijun
        st.markdown("---")
        st.subheader(f"Daily — Price > Ichimoku Kijun({ichi_base}) (latest bar)")
        above_rows = []
        for sym in universe:
            above, ts, close_now, kij_now = price_above_kijun_info_daily(sym, base=ichi_base)
            above_rows.append({"Symbol": sym, "AboveNow": above, "Timestamp": ts, "Close": close_now, "Kijun": kij_now})
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]
        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            view_above = df_above_daily.copy()
            view_above["Close"] = view_above["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_above["Kijun"] = view_above["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view_above[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

        # FOREX HOURLY scan
        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — latest NTD < {thresh:+.2f} ({scan_hour_range} lookback)")
            hourly_rows = []
            for sym in universe:
                ntd_val_h, ts_h = last_hourly_ntd_value(sym, ntd_window, period=scan_period)
                try:
                    intr = fetch_intraday(sym, period=scan_period)
                    if intr is not None and not intr.empty and "Close" in intr:
                        close_val_h = _safe_last_float(intr["Close"])
                    else:
                        close_val_h = np.nan
                except Exception:
                    close_val_h = np.nan
                hourly_rows.append({"Symbol": sym, "NTD_Last": ntd_val_h,
                                    "BelowThresh": (np.isfinite(ntd_val_h) and ntd_val_h < thresh),
                                    "Close": close_val_h, "Timestamp": ts_h})
            df_hour = pd.DataFrame(hourly_rows)
            hits_hour = df_hour[df_hour["BelowThresh"] == True].copy().sort_values("NTD_Last")

            if hits_hour.empty:
                st.info(f"No Forex pairs where the latest hourly NTD value is below {thresh:+.2f} within {scan_hour_range}.")
            else:
                showh = hits_hour.copy()
                showh["NTD_Last"] = showh["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
                showh["Close"] = showh["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(showh[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

            # FOREX HOURLY: Price > Kijun
            st.subheader(f"Forex Hourly — Price > Ichimoku Kijun({ichi_base}) (latest bar, {scan_hour_range})")
            habove_rows = []
            for sym in universe:
                above_h, ts_h, close_h, kij_h = price_above_kijun_info_hourly(sym, period=scan_period, base=ichi_base)
                habove_rows.append({"Symbol": sym, "AboveNow": above_h, "Timestamp": ts_h, "Close": close_h, "Kijun": kij_h})
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]
            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                vch["Kijun"] = vch["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(vch[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

# ==================== TAB 6: Long-Term History ====================
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox("Ticker:", universe, index=default_idx, key="hist_long_ticker")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("5Y", key="btn_5y"):  st.session_state.hist_years = 5
    if c2.button("10Y", key="btn_10y"): st.session_state.hist_years = 10
    if c3.button("15Y", key="btn_15y"): st.session_state.hist_years = 15
    if c4.button("20Y", key="btn_20y"): st.session_state.hist_years = 20

    years = int(st.session_state.hist_years)
    st.caption(f"Showing last **{years} years**. Support/Resistance = rolling **252-day** extremes; trendline fits the shown window.")

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
            res_roll = s.rolling(252, min_periods=1).max()
            sup_roll = s.rolling(252, min_periods=1).min()
            res_last = float(res_roll.iloc[-1]) if len(res_roll) else np.nan
            sup_last = float(sup_roll.iloc[-1]) if len(sup_roll) else np.nan
            yhat_all, up_all, lo_all, m_all, r2_all = regression_with_band(s, lookback=len(s))

            fig, ax = plt.subplots(figsize=(14,5))
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close")
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label="Resistance (252d)")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support (252d)")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")
            if not yhat_all.empty:
                ax.plot(yhat_all.index, yhat_all.values, "--", linewidth=2, label=f"Trend (m={fmt_slope(m_all)}/bar)")
            if not up_all.empty and not lo_all.empty:
                ax.plot(up_all.index, up_all.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
                ax.plot(lo_all.index, lo_all.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(0.99, 0.02, f"Current price: {fmt_price_val(px_now)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.text(0.01, 0.02, f"Slope: {fmt_slope(m_all)}/bar",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.text(0.50, 0.02, f"R² (trend): {fmt_r2(r2_all)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.set_xlabel("Date (PST)"); ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)
