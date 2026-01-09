# =========================
# Batch 1/10 — bullbear.py
# =========================
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# Statsmodels (SARIMAX)
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Stock Wizard — Bull/Bear Forecast",
    layout="wide",
)

# ---------------------------
# Timezones
# ---------------------------
PACIFIC = pytz.timezone("America/Los_Angeles")

# ---------------------------
# UI strings
# ---------------------------
FIB_ALERT_TEXT = (
    "⚠️ **Fibonacci Reversal Alerts**\n\n"
    "**BUY**: price touches near the **100%** Fib line (low) then prints consecutive higher closes.\n"
    "**SELL**: price touches near the **0%** Fib line (high) then prints consecutive lower closes.\n\n"
    "Also supported: **Fib + NPX 0.0 signal** (touch extreme + NPX crosses 0.0 in the reversal direction)."
)

# ---------------------------
# Formatting helpers
# ---------------------------
def fmt_price_val(x) -> str:
    try:
        if x is None:
            return "n/a"
        x = float(x)
        if not np.isfinite(x):
            return "n/a"
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        if abs(x) >= 100:
            return f"{x:,.3f}"
        if abs(x) >= 1:
            return f"{x:,.4f}"
        return f"{x:,.6f}"
    except Exception:
        return "n/a"

def fmt_pct(x, digits: int = 1) -> str:
    try:
        if x is None:
            return "n/a"
        x = float(x)
        if not np.isfinite(x):
            return "n/a"
        return f"{x*100:.{int(digits)}f}%"
    except Exception:
        return "n/a"

def fmt_slope(m) -> str:
    try:
        if m is None:
            return "n/a"
        m = float(m)
        if not np.isfinite(m):
            return "n/a"
        return f"{m:+.6f}"
    except Exception:
        return "n/a"

def fmt_r2(r2) -> str:
    try:
        if r2 is None:
            return "n/a"
        r2 = float(r2)
        if not np.isfinite(r2):
            return "n/a"
        return f"{r2:.4f}"
    except Exception:
        return "n/a"

# ---------------------------
# Series coercion / safety
# ---------------------------
def _coerce_1d_series(x) -> pd.Series:
    """
    Accepts Series or 1-col DataFrame and returns a float Series.
    """
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return pd.to_numeric(x.iloc[:, 0], errors="coerce").astype(float)
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").astype(float)
    try:
        return pd.Series(x, dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def _safe_last_float(x) -> float:
    s = _coerce_1d_series(x).dropna()
    if s.empty:
        return float("nan")
    try:
        v = float(s.iloc[-1])
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

# ---------------------------
# Plot styling
# ---------------------------
def style_axes(ax):
    try:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=9)
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
    except Exception:
        pass

def label_on_left(ax, y: float, text: str, color: str = "black"):
    """
    Draws a label anchored at the left edge of the current axes.
    """
    try:
        x0, x1 = ax.get_xlim()
        ax.text(
            x0, y, f" {text}",
            ha="left", va="center",
            fontsize=9, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.75),
            zorder=50
        )
    except Exception:
        pass

# ---------------------------
# Intraday time-axis helpers
# ---------------------------
def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    """
    If the plotted x-axis is a RangeIndex (0..N-1), this sets tick locations to evenly spaced bars
    and labels them with formatted real timestamps (PST).
    """
    try:
        if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
            return

        n = len(real_times)
        if n < 2:
            return

        n_ticks = int(max(2, n_ticks))
        idxs = np.linspace(0, n - 1, n_ticks).round().astype(int)
        idxs = np.unique(idxs)

        rt = real_times
        # Ensure PST for display
        try:
            if rt.tz is None:
                rt = rt.tz_localize("UTC").tz_convert(PACIFIC)
            else:
                rt = rt.tz_convert(PACIFIC)
        except Exception:
            pass

        labels = []
        for i in idxs:
            try:
                labels.append(rt[i].strftime("%m-%d %H:%M"))
            except Exception:
                labels.append(str(rt[i])[:16])

        ax.set_xticks(idxs.tolist())
        ax.set_xticklabels(labels, rotation=0)
    except Exception:
        pass

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, target_times):
    """
    Map target timestamps to nearest bar positions for RangeIndex plots.
    Returns a list of integer positions.
    """
    pos = []
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return pos
    if target_times is None:
        return pos

    rt = real_times
    try:
        if rt.tz is None:
            rt = rt.tz_localize("UTC").tz_convert(PACIFIC)
        else:
            rt = rt.tz_convert(PACIFIC)
    except Exception:
        pass

    # Convert to int64 ns for fast search
    rt_ns = rt.view("int64")

    for t in target_times:
        try:
            tt = pd.Timestamp(t)
            if tt.tz is None:
                tt = tt.tz_localize(PACIFIC)
            else:
                tt = tt.tz_convert(PACIFIC)
            t_ns = tt.value

            j = int(np.searchsorted(rt_ns, t_ns))
            if j <= 0:
                pos.append(0)
            elif j >= len(rt_ns):
                pos.append(len(rt_ns) - 1)
            else:
                # nearest of j-1 and j
                if abs(rt_ns[j] - t_ns) < abs(rt_ns[j - 1] - t_ns):
                    pos.append(j)
                else:
                    pos.append(j - 1)
        except Exception:
            continue
    return pos

# ---------------------------
# Cross helpers
# ---------------------------
def _cross_series(a: pd.Series, b: pd.Series):
    """
    Returns (cross_up, cross_down) masks where a crosses b.
    cross_up: a[t] >= b[t] and a[t-1] < b[t-1]
    cross_down: a[t] <= b[t] and a[t-1] > b[t-1]
    """
    a = _coerce_1d_series(a)
    b = _coerce_1d_series(b).reindex(a.index)
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    cross_up = (a >= b) & (prev_a < prev_b)
    cross_down = (a <= b) & (prev_a > prev_b)
    return cross_up.fillna(False), cross_down.fillna(False)

# ---------------------------
# Basic universe + sidebar
# ---------------------------
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "SPY", "QQQ", "IWM", "DIA"
]
DEFAULT_FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"
]

with st.sidebar:
    st.title("Stock Wizard")

    mode = st.radio("Mode", ["Stocks", "Forex"], index=0)
    universe = DEFAULT_FOREX if mode == "Forex" else DEFAULT_STOCKS

    st.subheader("Daily view")
    daily_view = st.selectbox(
        "Daily chart range",
        ["6mo", "1y", "2y", "5y", "10y", "max"],
        index=3
    )

    st.subheader("Core windows")
    sr_lb_daily = st.slider("Daily S/R window", 5, 240, 60, 5)
    sr_lb_hourly = st.slider("Hourly S/R window", 5, 240, 60, 5)
    slope_lb_daily = st.slider("Daily slope lookback", 10, 360, 90, 10)
    slope_lb_hourly = st.slider("Hourly slope lookback", 10, 360, 90, 10)

    st.subheader("Normalized indicators")
    ntd_window = st.slider("NTD / NPX window", 10, 400, 90, 10)

    st.subheader("Reversal model")
    rev_hist_lb = st.slider("Reversal hist window", 50, 800, 300, 25)
    rev_horizon = st.slider("Reversal horizon (bars)", 2, 60, 10, 1)
    rev_bars_confirm = st.slider("Reversal confirm bars", 1, 5, 2, 1)

    st.subheader("Proximity")
    sr_prox_pct = st.slider("S/R proximity (%)", 0.1, 10.0, 1.5, 0.1) / 100.0

    st.subheader("Overlays")
    show_hma = st.checkbox("Show HMA", value=True)
    hma_period = st.slider("HMA period", 10, 200, 55, 5)

    show_macd = st.checkbox("Show MACD panel", value=False)

    show_bbands = st.checkbox("Show Bollinger Bands", value=True)
    bb_win = st.slider("BB window", 10, 200, 55, 5)
    bb_mult = st.slider("BB sigma", 1.0, 4.0, 2.0, 0.1)
    bb_use_ema = st.checkbox("BB mid uses EMA", value=True)

    show_ntd = st.checkbox("Show NTD", value=True)
    shade_ntd = st.checkbox("Shade NTD regimes", value=True)
    show_npx_ntd = st.checkbox("Show NPX on NTD", value=True)
    mark_npx_cross = st.checkbox("Mark NPX crosses", value=True)

    show_ntd_channel = st.checkbox("Highlight in-range vs S/R on NTD", value=False)

    show_fibs = st.checkbox("Show Fibonacci", value=True)

    show_fx_news = st.checkbox("Show Yahoo Finance news (Forex)", value=False)
    news_window_days = st.slider("News window (days)", 1, 30, 7, 1)

    show_sessions_pst = st.checkbox("Show London/NY sessions (Forex)", value=True)

    # (present in later code; keep defaults here)
    show_psar = st.checkbox("Show PSAR", value=False)
    psar_step = st.slider("PSAR step", 0.01, 0.10, 0.02, 0.01)
    psar_max = st.slider("PSAR max", 0.10, 0.50, 0.20, 0.05)

    show_ichi = st.checkbox("Show Ichimoku Kijun", value=False)
    ichi_conv = st.slider("Ichimoku conversion", 5, 30, 9, 1)
    ichi_base = st.slider("Ichimoku base (Kijun)", 10, 120, 26, 1)
    ichi_spanb = st.slider("Ichimoku spanB", 20, 180, 52, 1)

    show_nrsi = st.checkbox("Use 2-panel layout (price + NTD)", value=True)

    show_hma_rev_ntd = st.checkbox("Show HMA reversal markers on NTD", value=False)
    hma_rev_lb = st.slider("HMA reversal lookback", 5, 120, 20, 5)

# =========================
# Batch 2/10 — bullbear.py
# =========================
# ---------------------------
# Daily view subsetting
# ---------------------------
def subset_by_daily_view(close: pd.Series, daily_view_label: str) -> pd.Series:
    s = _coerce_1d_series(close).dropna()
    if s.empty:
        return s

    lbl = str(daily_view_label).lower().strip()
    if lbl == "max":
        return s
    if lbl.endswith("mo"):
        n = int(lbl.replace("mo", "").strip())
        return s.iloc[-int(max(20, n * 21)):]  # ~21 trading days/mo
    if lbl.endswith("y"):
        n = int(lbl.replace("y", "").strip())
        return s.iloc[-int(max(60, n * 252)):]  # ~252 trading days/yr

    # fallback: 5y
    return s.iloc[-int(5 * 252):]

# ---------------------------
# Data fetchers (Yahoo Finance)
# ---------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist(symbol: str, period: str = "10y") -> pd.Series:
    """
    Daily close series (timezone-naive date index).
    """
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float)
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, period: str = "10y") -> pd.DataFrame:
    """
    Daily OHLC frame (timezone-naive date index).
    """
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d", interval: str = "60m") -> pd.DataFrame:
    """
    Intraday OHLC. Index is tz-aware in many cases; normalize to PST.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize time zone for display/logic
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
                else:
                    df.index = df.index.tz_convert(PACIFIC)
            except Exception:
                pass
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_hist_max(symbol: str) -> pd.Series:
    try:
        df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df["Close"], errors="coerce").dropna().astype(float)
    except Exception:
        return pd.Series(dtype=float)

# ---------------------------
# SARIMAX forecast (30d)
# ---------------------------
def compute_sarimax_forecast(close: pd.Series, steps: int = 30):
    """
    Returns (forecast_index, forecast_values, conf_int_df)
    """
    s = _coerce_1d_series(close).dropna()
    if s.empty or len(s) < 40:
        idx = pd.date_range(datetime.now().date(), periods=steps, freq="D")
        fc = pd.Series(index=idx, data=np.nan, dtype=float)
        ci = pd.DataFrame(index=idx, data={"lower": np.nan, "upper": np.nan})
        return idx, fc, ci

    # stabilize scale a bit
    y = s.astype(float)

    try:
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        f = res.get_forecast(steps=steps)
        mean = pd.Series(f.predicted_mean, index=f.predicted_mean.index).astype(float)
        ci = f.conf_int()
        ci = ci.rename(columns={ci.columns[0]: "lower", ci.columns[1]: "upper"})
    except Exception:
        # fallback: flat forecast
        last = float(y.iloc[-1])
        idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        mean = pd.Series(index=idx, data=last, dtype=float)
        ci = pd.DataFrame(index=idx, data={"lower": last, "upper": last})

    fc_idx = mean.index
    return fc_idx, mean, ci

# ---------------------------
# Indicators
# ---------------------------
def _wma(series: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    w = int(max(1, window))
    weights = np.arange(1, w + 1, dtype=float)

    def _calc(x):
        if np.any(~np.isfinite(x)):
            return np.nan
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(w, min_periods=w).apply(_calc, raw=True)

def compute_hma(series: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    p = int(max(2, period))
    half = max(1, p // 2)
    sqrtp = max(1, int(math.sqrt(p)))
    wma1 = _wma(s, half)
    wma2 = _wma(s, p)
    diff = 2 * wma1 - wma2
    return _wma(diff, sqrtp)

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(series).astype(float).ffill()
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=int(signal), adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def compute_bbands(series: pd.Series, window: int = 55, mult: float = 2.0, use_ema: bool = True):
    s = _coerce_1d_series(series).astype(float).ffill()
    w = int(max(2, window))
    if use_ema:
        mid = s.ewm(span=w, adjust=False).mean()
        # EMA-based stdev approximation via rolling stdev of residuals (stable enough for display)
        resid = s - mid
        sd = resid.rolling(w, min_periods=w).std(ddof=0)
    else:
        mid = s.rolling(w, min_periods=w).mean()
        sd = s.rolling(w, min_periods=w).std(ddof=0)

    up = mid + float(mult) * sd
    lo = mid - float(mult) * sd

    # %B in [0,1] (clip for stability)
    pctb = (s - lo) / (up - lo)
    pctb = pctb.replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)

    # Normalized BB position centered at 0 (approx)
    # -1 at lower band, +1 at upper band
    nbb = (pctb * 2.0) - 1.0
    return mid, up, lo, pctb, nbb

# ---------------------------
# Normalized Trend / Price
# ---------------------------
def compute_normalized_price(series: pd.Series, window: int = 90) -> pd.Series:
    """
    NPX in [-1, 1]:
      0 is the midpoint of rolling min/max range.
    """
    s = _coerce_1d_series(series).astype(float).ffill()
    w = int(max(2, window))
    rmin = s.rolling(w, min_periods=2).min()
    rmax = s.rolling(w, min_periods=2).max()
    denom = (rmax - rmin).replace(0.0, np.nan)
    n01 = (s - rmin) / denom
    n01 = n01.replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0)
    return (n01 * 2.0) - 1.0

def compute_normalized_trend(series: pd.Series, window: int = 90) -> pd.Series:
    """
    NTD in [-1, 1] based on rolling slope (polyfit) scaled by rolling price range.
    """
    s = _coerce_1d_series(series).astype(float).ffill()
    w = int(max(5, window))

    def _slope(x):
        if len(x) < 2 or np.any(~np.isfinite(x)):
            return np.nan
        t = np.arange(len(x), dtype=float)
        m, _b = np.polyfit(t, x, 1)
        return float(m)

    slope = s.rolling(w, min_periods=max(5, w // 4)).apply(_slope, raw=True)

    rmin = s.rolling(w, min_periods=2).min()
    rmax = s.rolling(w, min_periods=2).max()
    rng = (rmax - rmin).replace(0.0, np.nan)

    # scale slope by range/window so typical values sit inside [-1,1]
    scaled = (slope * w) / rng
    scaled = scaled.replace([np.inf, -np.inf], np.nan)

    # squash with tanh to bound in [-1,1]
    return np.tanh(scaled)

# ---------------------------
# Regression band (trend + ±2σ) — used throughout
# ---------------------------
def regression_with_band(price: pd.Series, lookback: int = 90, sigma: float = 2.0):
    p = _coerce_1d_series(price).astype(float).ffill()
    lb = int(max(5, lookback))
    if p.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), np.nan, np.nan

    # Work on last lb points but return aligned to full index
    seg = p.iloc[-lb:] if len(p) > lb else p.copy()
    seg = seg.dropna()
    if len(seg) < 5:
        return pd.Series(index=p.index, dtype=float), pd.Series(index=p.index, dtype=float), pd.Series(index=p.index, dtype=float), np.nan, np.nan

    x = np.arange(len(seg), dtype=float)
    y = seg.to_numpy(dtype=float)

    m, b = np.polyfit(x, y, 1)
    yhat_seg = m * x + b
    resid = y - yhat_seg
    sd = float(np.nanstd(resid, ddof=0))

    # R² on seg
    ss_res = float(np.nansum((y - yhat_seg) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    # build full-length yhat aligned to p.index: only last lb is meaningful; earlier NaN
    yhat = pd.Series(index=p.index, dtype=float)
    upper = pd.Series(index=p.index, dtype=float)
    lower = pd.Series(index=p.index, dtype=float)

    # map seg positions to the end of the full index
    start = len(p) - len(seg)
    xx_full = np.arange(len(seg), dtype=float)
    yhat_vals = m * xx_full + b
    yhat.iloc[start:] = yhat_vals
    upper.iloc[start:] = yhat_vals + float(sigma) * sd
    lower.iloc[start:] = yhat_vals - float(sigma) * sd

    return yhat, upper, lower, float(m), float(r2)

# ---------------------------
# Slope reversal probability (simple empirical)
# ---------------------------
def slope_reversal_probability(price: pd.Series,
                              current_slope: float,
                              hist_window: int = 300,
                              slope_window: int = 90,
                              horizon: int = 10) -> float:
    """
    Empirical estimate:
      in the last hist_window bars, how often does the slope over slope_window
      flip sign within the next 'horizon' bars?

    Returns probability in [0,1].
    """
    p = _coerce_1d_series(price).astype(float).ffill().dropna()
    if p.empty or len(p) < max(hist_window, slope_window + horizon + 5):
        return float("nan")

    hist_window = int(max(50, hist_window))
    slope_window = int(max(10, slope_window))
    horizon = int(max(1, horizon))

    tail = p.iloc[-hist_window:].copy()
    n = len(tail)
    if n < slope_window + horizon + 5:
        return float("nan")

    def _seg_slope(arr):
        t = np.arange(len(arr), dtype=float)
        m, _b = np.polyfit(t, arr, 1)
        return float(m)

    # compute rolling slopes on tail
    slopes = []
    for i in range(slope_window, n - horizon + 1):
        seg = tail.iloc[i - slope_window:i].to_numpy(dtype=float)
        slopes.append(_seg_slope(seg))
    slopes = np.array(slopes, dtype=float)

    # For each slope observation, check if within next horizon bars slope flips sign at least once
    flips = 0
    total = 0
    for i in range(len(slopes) - horizon):
        s0 = slopes[i]
        if not np.isfinite(s0) or s0 == 0.0:
            continue
        future = slopes[i + 1:i + 1 + horizon]
        if future.size == 0:
            continue
        total += 1
        if np.any(np.sign(future) != np.sign(s0)):
            flips += 1

    if total <= 0:
        return float("nan")

    base = flips / total

    # Optional: gently weight by whether current_slope matches typical slope magnitudes
    try:
        cs = float(current_slope)
        if np.isfinite(cs) and cs != 0.0:
            return float(np.clip(base, 0.0, 1.0))
        return float(np.clip(base, 0.0, 1.0))
    except Exception:
        return float(np.clip(base, 0.0, 1.0))

# ---------------------------
# Fibonacci levels (0% = high, 100% = low)
# ---------------------------
def fibonacci_levels(series: pd.Series):
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return {}
    hi = float(np.nanmax(s.to_numpy(dtype=float)))
    lo = float(np.nanmin(s.to_numpy(dtype=float)))
    if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
        return {}
    rng = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * rng,
        "38.2%": hi - 0.382 * rng,
        "50%": hi - 0.500 * rng,
        "61.8%": hi - 0.618 * rng,
        "78.6%": hi - 0.786 * rng,
        "100%": lo,
    }

# ============================================================
# FIX (THIS BUG): define _fib_npx_zero_signal_series
# ============================================================
def _confirm_npx_side(npx: pd.Series, t, side: str, confirm_bars: int = 1) -> bool:
    """
    After the cross bar t, require NPX to remain on the crossed side for confirm_bars bars (inclusive of t).
    side = "BUY" -> NPX >= 0.0; "SELL" -> NPX <= 0.0
    """
    try:
        confirm_bars = int(max(1, confirm_bars))
        i = int(npx.index.get_loc(t))
        j = min(len(npx) - 1, i + confirm_bars - 1)
        seg = _coerce_1d_series(npx.iloc[i:j + 1]).dropna()
        if seg.empty:
            return False
        if str(side).upper().startswith("B"):
            return bool((seg >= 0.0).all())
        return bool((seg <= 0.0).all())
    except Exception:
        return False

def _fib_npx_zero_signal_series(close: pd.Series,
                               npx: pd.Series,
                               prox: float = 0.015,
                               lookback_bars: int = 10,
                               slope_lb: int = 90,
                               npx_confirm_bars: int = 1):
    """
    Returns a dict signal (for annotation/scanning) or None.

    BUY signal:
      - price touched Fib 100% (low) within lookback window (t_touch)
      - NPX crossed UP through 0.0 within lookback window (t_cross)
      - touch happens at/before cross within the window
      - regression slope over slope_lb aligns with direction (slope > 0)

    SELL signal:
      - price touched Fib 0% (high)
      - NPX crossed DOWN through 0.0
      - slope < 0
    """
    c = _coerce_1d_series(close).astype(float).ffill().dropna()
    if c.empty or len(c) < 10:
        return None

    npx = _coerce_1d_series(npx).reindex(c.index).astype(float)
    if npx.dropna().empty:
        return None

    fibs = fibonacci_levels(c)
    if not fibs or ("0%" not in fibs) or ("100%" not in fibs):
        return None

    fib_hi = float(fibs["0%"])
    fib_lo = float(fibs["100%"])
    if not (np.isfinite(fib_hi) and np.isfinite(fib_lo)) or fib_hi == fib_lo:
        return None

    prox = float(max(0.0, prox))
    lb = int(max(2, lookback_bars))
    win = c.iloc[-lb:]
    win_npx = npx.reindex(win.index)

    # Touch tolerance relative to level
    def _near_level(price_ser: pd.Series, level: float) -> pd.Series:
        if not np.isfinite(level) or level == 0:
            # fallback to range-based tolerance
            rng = float(abs(fib_hi - fib_lo))
            if not np.isfinite(rng) or rng <= 0:
                return pd.Series(False, index=price_ser.index)
            thr = prox * rng
            return (price_ser - level).abs() <= thr
        return (price_ser - level).abs() <= (abs(level) * prox)

    touch_low = _near_level(win, fib_lo)   # near 100%
    touch_high = _near_level(win, fib_hi)  # near 0%

    # Cross of NPX through 0.0
    cross_up, cross_dn = _cross_series(win_npx, pd.Series(0.0, index=win_npx.index))

    # Confirm bars on crossed side
    cross_up = cross_up & cross_up.apply(lambda v: True)  # keep dtype
    cross_dn = cross_dn & cross_dn.apply(lambda v: True)

    # Find most recent BUY
    buy_cross_times = cross_up[cross_up].index.tolist()
    sell_cross_times = cross_dn[cross_dn].index.tolist()

    # slope alignment on current regression window (same logic used elsewhere)
    _yhat, _up, _lo, m_now, r2_now = regression_with_band(c, lookback=int(slope_lb))
    m_now = float(m_now) if np.isfinite(m_now) else np.nan

    best_sig = None

    # BUY: last cross_up, last touch_low at/before cross
    if buy_cross_times and np.isfinite(m_now) and m_now > 0.0:
        for t_cross in reversed(buy_cross_times):
            if not _confirm_npx_side(win_npx, t_cross, "BUY", confirm_bars=int(npx_confirm_bars)):
                continue
            # require a touch at or before the cross inside window
            touches = touch_low & (touch_low.index <= t_cross)
            if touches.any():
                t_touch = touches[touches].index[-1]
                best_sig = {
                    "side": "BUY",
                    "time": t_cross,
                    "price": float(c.loc[t_cross]) if t_cross in c.index else float("nan"),
                    "from_level": "100%",
                    "touch_time": t_touch,
                    "touch_price": float(c.loc[t_touch]) if t_touch in c.index else float("nan"),
                    "npx_at_cross": float(npx.loc[t_cross]) if t_cross in npx.index else float("nan"),
                    "slope": float(m_now),
                    "r2": float(r2_now) if np.isfinite(r2_now) else np.nan,
                }
                break

    # SELL: last cross_dn, last touch_high at/before cross
    if best_sig is None and sell_cross_times and np.isfinite(m_now) and m_now < 0.0:
        for t_cross in reversed(sell_cross_times):
            if not _confirm_npx_side(win_npx, t_cross, "SELL", confirm_bars=int(npx_confirm_bars)):
                continue
            touches = touch_high & (touch_high.index <= t_cross)
            if touches.any():
                t_touch = touches[touches].index[-1]
                best_sig = {
                    "side": "SELL",
                    "time": t_cross,
                    "price": float(c.loc[t_cross]) if t_cross in c.index else float("nan"),
                    "from_level": "0%",
                    "touch_time": t_touch,
                    "touch_price": float(c.loc[t_touch]) if t_touch in c.index else float("nan"),
                    "npx_at_cross": float(npx.loc[t_cross]) if t_cross in npx.index else float("nan"),
                    "slope": float(m_now),
                    "r2": float(r2_now) if np.isfinite(r2_now) else np.nan,
                }
                break

    return best_sig

# ---------------------------
# Scanner row helper used by Tab 9 (and anywhere else)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_fib_npx_zero_signal(symbol: str,
                                   daily_view_label: str,
                                   ntd_win: int,
                                   direction: str,
                                   prox: float,
                                   lookback_bars: int,
                                   slope_lb: int,
                                   npx_confirm_bars: int = 1):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 10:
            return None

        npx = compute_normalized_price(close_show, window=int(ntd_win))
        sig = _fib_npx_zero_signal_series(
            close=close_show,
            npx=npx,
            prox=float(prox),
            lookback_bars=int(lookback_bars),
            slope_lb=int(slope_lb),
            npx_confirm_bars=int(npx_confirm_bars),
        )
        if not isinstance(sig, dict):
            return None

        want = str(direction).upper()
        if want.startswith("B") and sig.get("side") != "BUY":
            return None
        if want.startswith("S") and sig.get("side") != "SELL":
            return None

        t_cross = sig.get("time")
        if t_cross is None or t_cross not in close_show.index:
            return None
        loc = int(close_show.index.get_loc(t_cross))
        bars_since = int((len(close_show) - 1) - loc)

        return {
            "Symbol": symbol,
            "Side": sig.get("side"),
            "From Level": sig.get("from_level"),
            "Cross Time": t_cross,
            "Bars Since Cross": bars_since,
            "Cross Price": sig.get("price"),
            "Touch Time": sig.get("touch_time"),
            "Touch Price": sig.get("touch_price"),
            "NPX@Cross": sig.get("npx_at_cross"),
            "Slope": sig.get("slope"),
            "R2": sig.get("r2"),
        }
    except Exception:
        return None

# ---------------------------
# Chart annotation helper (used by tabs that plot price)
# ---------------------------
def annotate_fib_npx_signal(ax, sig: dict):
    """
    Places a BUY/SELL marker at sig["time"] with a compact label.
    Works for DateTimeIndex or numeric RangeIndex.
    """
    if not isinstance(sig, dict):
        return
    try:
        t = sig.get("time")
        px = sig.get("price", np.nan)
        side = str(sig.get("side", "")).upper()
        frm = str(sig.get("from_level", ""))
        if t is None or not np.isfinite(float(px)):
            return

        if side.startswith("B"):
            marker = "^"
            edge = "tab:green"
            txt = f"Fib+NPX BUY ({frm})"
        else:
            marker = "v"
            edge = "tab:red"
            txt = f"Fib+NPX SELL ({frm})"

        ax.scatter([t], [px], marker=marker, s=90, color=edge, zorder=40)
        ax.annotate(
            txt,
            xy=(t, px),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge, alpha=0.85),
            arrowprops=dict(arrowstyle="->", color=edge, lw=1.2),
            zorder=41
        )
    except Exception:
        pass


# =========================
# Part 3/10 — bullbear.py
# =========================
# ---------------------------
# Regression & ±2σ band
# ---------------------------
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
    """
    Linear regression on last `lookback` bars with:
      • fitted trendline
      • symmetric ±z·σ band (σ = std of residuals)
      • R² of the fit
    """
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

def slope_reversal_probability(series_like,
                               current_slope: float,
                               hist_window: int = 240,
                               slope_window: int = 60,
                               horizon: int = 15) -> float:
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

    match = 0
    flips = 0
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

def find_band_bounce_signal(price: pd.Series,
                            upper_band: pd.Series,
                            lower_band: pd.Series,
                            slope_val: float):
    """
    Detect the most recent BUY/SELL signal based on a 'bounce' off the ±2σ band.
    """
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    mask = p.notna() & u.notna() & l.notna()
    if mask.sum() < 2:
        return None

    p = p[mask]
    u = u.reindex(p.index)
    l = l.reindex(p.index)

    inside = (p <= u) & (p >= l)
    below  = p < l
    above  = p > u

    try:
        slope = float(slope_val)
    except Exception:
        slope = np.nan
    if not np.isfinite(slope) or slope == 0.0:
        return None

    if slope > 0:
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

def _cross_series(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index if len(p) else l.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)
    p = p[ok]
    l = l[ok]
    above = p > l
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    return cross_up.reindex(p.index, fill_value=False), cross_dn.reindex(p.index, fill_value=False)

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

# ---------------------------
# NEW (THIS REQUEST): Fibonacci Buy/Sell markers (Price chart area)
# ---------------------------
def overlay_fib_npx_signals(ax,
                            price: pd.Series,
                            buy_mask: pd.Series,
                            sell_mask: pd.Series,
                            label_buy: str = "Fibonacci BUY",
                            label_sell: str = "Fibonacci SELL"):
    """
    Plot Fibonacci BUY/SELL markers on the PRICE chart.

    Uses buy_mask/sell_mask computed from:
      - price near Fib 100% (low) / 0% (high)
      - NPX crosses 0.0 upward/downward (recent)
    """
    p = _coerce_1d_series(price)
    bm = _coerce_1d_series(buy_mask).reindex(p.index).fillna(0).astype(bool) if buy_mask is not None else pd.Series(False, index=p.index)
    sm = _coerce_1d_series(sell_mask).reindex(p.index).fillna(0).astype(bool) if sell_mask is not None else pd.Series(False, index=p.index)

    buy_idx = list(bm[bm].index)
    sell_idx = list(sm[sm].index)

    if buy_idx:
        ax.scatter(buy_idx, p.loc[buy_idx], marker="^", s=120, color="tab:green", zorder=11, label=label_buy)
        for t in buy_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB BUY", va="bottom", fontsize=9, color="tab:green", fontweight="bold", zorder=12)
            except Exception:
                pass

    if sell_idx:
        ax.scatter(sell_idx, p.loc[sell_idx], marker="v", s=120, color="tab:red", zorder=11, label=label_sell)
        for t in sell_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB SELL", va="top", fontsize=9, color="tab:red", fontweight="bold", zorder=12)
            except Exception:
                pass

# ---------------------------
# Slope BUY/SELL Trigger (leaderline + legend)
# ---------------------------
def find_slope_trigger_after_band_reversal(price: pd.Series,
                                          yhat: pd.Series,
                                          upper_band: pd.Series,
                                          lower_band: pd.Series,
                                          horizon: int = 15):
    """
    BUY trigger:
      - price touches/breaches LOWER band, then crosses ABOVE the slope line (yhat)
    SELL trigger:
      - price touches/breaches UPPER band, then crosses BELOW the slope line (yhat)
    Returns the most recent trigger dict or None.
    """
    p = _coerce_1d_series(price)
    y = _coerce_1d_series(yhat).reindex(p.index)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & y.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None
    p = p[ok]; y = y[ok]; u = u[ok]; l = l[ok]

    cross_up, cross_dn = _cross_series(p, y)
    below = (p <= l)
    above = (p >= u)

    hz = max(1, int(horizon))

    def _last_touch_before(t_idx, touch_mask: pd.Series):
        try:
            loc = int(p.index.get_loc(t_idx))
        except Exception:
            return None
        j0 = max(0, loc - hz)
        window = touch_mask.iloc[j0:loc+1]
        if not window.any():
            return None
        return window[window].index[-1]

    last_buy_cross = cross_up[cross_up].index[-1] if cross_up.any() else None
    last_sell_cross = cross_dn[cross_dn].index[-1] if cross_dn.any() else None

    buy_tr = None
    if last_buy_cross is not None:
        t_touch = _last_touch_before(last_buy_cross, below)
        if t_touch is not None:
            buy_tr = {
                "side": "BUY",
                "touch_time": t_touch,
                "touch_price": float(p.loc[t_touch]),
                "cross_time": last_buy_cross,
                "cross_price": float(p.loc[last_buy_cross]),
            }

    sell_tr = None
    if last_sell_cross is not None:
        t_touch = _last_touch_before(last_sell_cross, above)
        if t_touch is not None:
            sell_tr = {
                "side": "SELL",
                "touch_time": t_touch,
                "touch_price": float(p.loc[t_touch]),
                "cross_time": last_sell_cross,
                "cross_price": float(p.loc[last_sell_cross]),
            }

    if buy_tr is None and sell_tr is None:
        return None
    if buy_tr is None:
        return sell_tr
    if sell_tr is None:
        return buy_tr

    return buy_tr if buy_tr["cross_time"] >= sell_tr["cross_time"] else sell_tr

def annotate_slope_trigger(ax, trig: dict):
    if trig is None:
        return
    side = trig.get("side", "")
    t0 = trig.get("touch_time")
    p0 = trig.get("touch_price")
    t1 = trig.get("cross_time")
    p1 = trig.get("cross_price")
    if t0 is None or t1 is None:
        return
    if not (np.isfinite(p0) and np.isfinite(p1)):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    lbl = f"Slope {side} Trigger"
    ax.annotate(
        "",
        xy=(t1, p1),
        xytext=(t0, p0),
        arrowprops=dict(arrowstyle="->", color=col, lw=2.0, alpha=0.85),
        zorder=9
    )
    ax.scatter([t1], [p1], marker="o", s=90, color=col, zorder=10, label=lbl)
    ax.text(
        t1, p1,
        f"  {lbl}",
        color=col,
        fontsize=9,
        fontweight="bold",
        va="bottom" if side == "BUY" else "top",
        zorder=10
    )


# =========================
# Part 4/10 — bullbear.py
# =========================
# ---------------------------
# Other indicators
# ---------------------------
def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

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

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    return macd.reindex(s.index), sig.reindex(s.index), hist.reindex(s.index)

def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9, norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),)*3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    minp = max(10, norm_win//10)

    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)

    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return (nmacd.reindex(s.index), nsignal.reindex(s.index), nhist.reindex(s.index))

def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win//10)
    m = v.rolling(norm_win, min_periods=minp).mean()
    sd = v.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (v - m) / sd
    return np.tanh(z / 3.0)

def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26, norm_win: int = 240) -> pd.Series:
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

# ---------------------------
# NEW (THIS REQUEST): Fib touch + NPX(0.0) cross logic for Fibonacci BUY/SELL signals
# ---------------------------
def npx_zero_cross_masks(npx: pd.Series, level: float = 0.0):
    """
    NPX cross of a constant level (default 0.0):
      - cross_up: npx goes from < level to >= level
      - cross_dn: npx goes from > level to <= level
    """
    s = _coerce_1d_series(npx)
    prev = s.shift(1)
    cross_up = (s >= float(level)) & (prev < float(level))
    cross_dn = (s <= float(level)) & (prev > float(level))
    return cross_up.fillna(False), cross_dn.fillna(False)

def fib_touch_masks(price: pd.Series, proximity_pct_of_range: float = 0.02):
    """
    Returns (near_hi_0pct, near_lo_100pct, fibs_dict).
    'near' uses a tolerance = proximity_pct_of_range * (fib_range).
    """
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        idx = _coerce_1d_series(price).index
        return (pd.Series(False, index=idx), pd.Series(False, index=idx), {})

    fibs = fibonacci_levels(p)
    if not fibs:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), {})

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    near_hi = (p >= (hi - tol)).reindex(p.index, fill_value=False)
    near_lo = (p <= (lo + tol)).reindex(p.index, fill_value=False)
    return near_hi, near_lo, fibs

def fib_npx_zero_cross_signal_masks(price: pd.Series,
                                   npx: pd.Series,
                                   horizon_bars: int = 15,
                                   proximity_pct_of_range: float = 0.02,
                                   npx_level: float = 0.0):
    """
    Fibonacci BUY mask:
      - NPX crosses UP through 0.0
      - AND price touched near Fib 100% (low) within last `horizon_bars` (including current)

    Fibonacci SELL mask:
      - NPX crosses DOWN through 0.0
      - AND price touched near Fib 0% (high) within last `horizon_bars` (including current)
    """
    p = _coerce_1d_series(price)
    x = _coerce_1d_series(npx).reindex(p.index)

    near_hi, near_lo, fibs = fib_touch_masks(p, proximity_pct_of_range=float(proximity_pct_of_range))
    up0, dn0 = npx_zero_cross_masks(x, level=float(npx_level))

    hz = max(1, int(horizon_bars))
    touched_lo_recent = near_lo.rolling(hz + 1, min_periods=1).max().astype(bool)
    touched_hi_recent = near_hi.rolling(hz + 1, min_periods=1).max().astype(bool)

    buy_mask = up0.reindex(p.index, fill_value=False) & touched_lo_recent.reindex(p.index, fill_value=False)
    sell_mask = dn0.reindex(p.index, fill_value=False) & touched_hi_recent.reindex(p.index, fill_value=False)

    return buy_mask.fillna(False), sell_mask.fillna(False), fibs

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
    color = "green" if m >= 0 else "red"
    ax.plot(s.index, yhat, linestyle="--", linewidth=2.4, color=color,
            label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return float(m)

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

def find_macd_hma_sr_signal(close: pd.Series,
                            hma: pd.Series,
                            macd: pd.Series,
                            sup: pd.Series,
                            res: pd.Series,
                            global_trend_slope: float,
                            prox: float = 0.0025):
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(hma).reindex(c.index)
    m = _coerce_1d_series(macd).reindex(c.index)
    s_sup = _coerce_1d_series(sup).reindex(c.index).ffill()
    s_res = _coerce_1d_series(res).reindex(c.index).ffill()

    ok = c.notna() & h.notna() & m.notna() & s_sup.notna() & s_res.notna()
    if ok.sum() < 3:
        return None

    c = c[ok]; h = h[ok]; m = m[ok]; s_sup = s_sup[ok]; s_res = s_res[ok]

    cross_up, cross_dn = _cross_series(c, h)
    cross_up = cross_up.reindex(c.index, fill_value=False)
    cross_dn = cross_dn.reindex(c.index, fill_value=False)

    near_support = c <= s_sup * (1.0 + prox)
    away_from_support = (c - s_sup) > (c.shift(1) - s_sup.shift(1))
    near_resist = c >= s_res * (1.0 - prox)
    away_from_resist = (s_res - c) > (s_res.shift(1) - c.shift(1))

    uptrend = np.isfinite(global_trend_slope) and float(global_trend_slope) > 0
    downtrend = np.isfinite(global_trend_slope) and float(global_trend_slope) < 0

    buy_mask = uptrend & (m < 0.0) & cross_up & near_support & away_from_support
    sell_mask = downtrend & (m > 0.0) & cross_dn & near_resist & away_from_resist

    last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
    last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None
    if last_buy is None and last_sell is None:
        return None

    if last_sell is None:
        t = last_buy; side = "BUY"
    elif last_buy is None:
        t = last_sell; side = "SELL"
    else:
        t = last_buy if last_buy >= last_sell else last_sell
        side = "BUY" if t == last_buy else "SELL"

    px = float(c.loc[t]) if np.isfinite(c.loc[t]) else np.nan
    note = "MACD/HMA55 + S/R"
    return {"time": t, "price": px, "side": side, "note": note}

def annotate_macd_signal(ax, ts, px, side: str):
    if side == "BUY":
        ax.scatter([ts], [px], marker="*", s=180, color="tab:green", zorder=10, label="MACD BUY (HMA55+S/R)")
    else:
        ax.scatter([ts], [px], marker="*", s=180, color="tab:red", zorder=10, label="MACD SELL (HMA55+S/R)")

def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty or window < 2 or not np.isfinite(mult):
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
    return (mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index))


# =========================
# Part 5/10 — bullbear.py
# =========================
# ---------------------------
# Ichimoku, Supertrend, PSAR
# ---------------------------
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = True):
    h = _coerce_1d_series(high)
    l = _coerce_1d_series(low)
    c = _coerce_1d_series(close)
    idx = c.index.union(h.index).union(l.index)
    h = h.reindex(idx)
    l = l.reindex(idx)
    c = c.reindex(idx)

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
    return (tenkan.reindex(idx), kijun.reindex(idx), senkou_a.reindex(idx), senkou_b.reindex(idx), chikou.reindex(idx))

def _compute_atr_from_ohlc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(df["High"])
    low  = _coerce_1d_series(df["Low"])
    close= _coerce_1d_series(df["Close"])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.reindex(df.index)

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.DataFrame(columns=["ST","in_uptrend"])
    ohlc = df[["High","Low","Close"]].copy()
    atr = _compute_atr_from_ohlc(ohlc, period=atr_period)
    hl2 = (ohlc["High"] + ohlc["Low"]) / 2.0
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr

    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_uptrend = pd.Series(index=ohlc.index, dtype=bool)

    for i in range(len(ohlc)):
        if i == 0:
            in_uptrend.iloc[i] = True
            st_line.iloc[i] = lowerband.iloc[i]
            continue

        if ohlc["Close"].iloc[i] > upperband.iloc[i-1]:
            in_uptrend.iloc[i] = True
        elif ohlc["Close"].iloc[i] < lowerband.iloc[i-1]:
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
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        return pd.DataFrame(columns=["PSAR","in_uptrend"])
    high = _coerce_1d_series(df["High"])
    low  = _coerce_1d_series(df["Low"])
    idx = high.index.union(low.index)
    high = high.reindex(idx)
    low  = low.reindex(idx)

    psar = pd.Series(index=idx, dtype=float)
    in_uptrend = pd.Series(index=idx, dtype=bool)

    in_uptrend.iloc[0] = True
    psar.iloc[0] = float(low.iloc[0])
    ep = float(high.iloc[0])
    af = step

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i-1]
        if in_uptrend.iloc[i-1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i],
                               float(low.iloc[i-1]),
                               float(low.iloc[i-2]) if i >= 2 else float(low.iloc[i-1]))
            if high.iloc[i] > ep:
                ep = float(high.iloc[i])
                af = min(af + step, max_step)
            if low.iloc[i] < psar.iloc[i]:
                in_uptrend.iloc[i] = False
                psar.iloc[i] = ep
                ep = float(low.iloc[i])
                af = step
            else:
                in_uptrend.iloc[i] = True
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = max(psar.iloc[i],
                               float(high.iloc[i-1]),
                               float(high.iloc[i-2]) if i >= 2 else float(high.iloc[i-1]))
            if low.iloc[i] < ep:
                ep = float(low.iloc[i])
                af = min(af + step, max_step)
            if high.iloc[i] > psar.iloc[i]:
                in_uptrend.iloc[i] = True
                psar.iloc[i] = ep
                ep = float(high.iloc[i])
                af = step
            else:
                in_uptrend.iloc[i] = False

    return pd.DataFrame({"PSAR": psar, "in_uptrend": in_uptrend})

def detect_hma_reversal_masks(price: pd.Series, hma: pd.Series, lookback: int = 3):
    h = _coerce_1d_series(hma)
    slope = h.diff().rolling(lookback, min_periods=1).mean()
    sign_now = np.sign(slope)
    sign_prev = np.sign(slope.shift(1))
    cross_up, cross_dn = _cross_series(price, hma)
    buy_rev  = cross_up & (sign_now > 0) & (sign_prev < 0)
    sell_rev = cross_dn & (sign_now < 0) & (sign_prev > 0)
    return buy_rev.fillna(False), sell_rev.fillna(False)

def overlay_hma_reversal_on_ntd(ax, price: pd.Series, hma: pd.Series,
                               lookback: int = 3, y_up: float = 0.95, y_dn: float = -0.95,
                               period: int = 55, ntd: pd.Series = None):
    buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
    idx_up = list(buy_rev[buy_rev].index)
    idx_dn = list(sell_rev[sell_rev].index)
    if len(idx_up):
        ax.scatter(idx_up, [y_up]*len(idx_up), marker="s", s=70, color="tab:green", zorder=8, label=f"HMA({period}) REV")
    if len(idx_dn):
        ax.scatter(idx_dn, [y_dn]*len(idx_dn), marker="D", s=70, color="tab:red", zorder=8, label=f"HMA({period}) REV")

def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx)
    ntd = ntd.reindex(idx)
    if npx.dropna().empty:
        return
    ax.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")
    if mark_crosses and not ntd.dropna().empty:
        up_mask, dn_mask = _cross_series(npx, ntd)
        up_idx = list(up_mask[up_mask].index)
        dn_idx = list(dn_mask[dn_mask].index)
        if len(up_idx):
            ax.scatter(up_idx, ntd.loc[up_idx], marker="o", s=40, color="tab:green", zorder=9, label="Price↑NTD")
        if len(dn_idx):
            ax.scatter(dn_idx, ntd.loc[dn_idx], marker="x", s=60, color="tab:red", zorder=9, label="Price↓NTD")

def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd).dropna()
    if s.empty or not np.isfinite(trend_slope):
        return
    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    cross_up0 = (s >= 0.0) & (s.shift(1) < 0.0)
    cross_dn0 = (s <= 0.0) & (s.shift(1) > 0.0)
    idx_up0 = list(cross_up0[cross_up0].index)
    idx_dn0 = list(cross_dn0[cross_dn0].index)

    cross_out_hi = (s >= upper) & (s.shift(1) < upper)
    cross_out_lo = (s <= lower) & (s.shift(1) > lower)
    idx_hi = list(cross_out_hi[cross_out_hi].index)
    idx_lo = list(cross_out_lo[cross_out_lo].index)

    if uptrend:
        if idx_up0:
            ax.scatter(idx_up0, [0.0]*len(idx_up0), marker="^", s=95, color="tab:green", zorder=10, label="NTD 0↑")
        if idx_lo:
            ax.scatter(idx_lo, s.loc[idx_lo], marker="^", s=85, color="tab:green", zorder=10, label="NTD < -0.75")
    if downtrend:
        if idx_dn0:
            ax.scatter(idx_dn0, [0.0]*len(idx_dn0), marker="v", s=95, color="tab:red", zorder=10, label="NTD 0↓")
        if idx_hi:
            ax.scatter(idx_hi, s.loc[idx_hi], marker="v", s=85, color="tab:red", zorder=10, label="NTD > +0.75")

def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas < 0))

def overlay_ntd_sr_reversal_stars(ax,
                                 price: pd.Series,
                                 sup: pd.Series,
                                 res: pd.Series,
                                 trend_slope: float,
                                 ntd: pd.Series,
                                 prox: float = 0.0025,
                                 bars_confirm: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        return
    s_sup = _coerce_1d_series(sup).reindex(p.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(p.index).ffill().bfill()
    s_ntd = _coerce_1d_series(ntd).reindex(p.index)

    t = p.index[-1]
    if not (t in s_sup.index and t in s_res.index and t in s_ntd.index):
        return
    c0 = float(p.iloc[-1])
    c1 = float(p.iloc[-2]) if len(p) >= 2 else np.nan
    S0 = float(s_sup.loc[t]) if pd.notna(s_sup.loc[t]) else np.nan
    R0 = float(s_res.loc[t]) if pd.notna(s_res.loc[t]) else np.nan
    ntd0 = float(s_ntd.loc[t]) if pd.notna(s_ntd.loc[t]) else np.nan
    if not np.all(np.isfinite([c0, S0, R0, ntd0])):
        return

    near_support = c0 <= S0 * (1.0 + prox)
    near_resist  = c0 >= R0 * (1.0 - prox)

    toward_res = toward_sup = False
    if np.isfinite(c1):
        toward_res = (R0 - c0) < (R0 - c1)
        toward_sup = (c0 - S0) < (c1 - S0)

    buy_cond  = (trend_slope > 0) and near_support and _n_consecutive_increasing(p, bars_confirm) and toward_res
    sell_cond = (trend_slope < 0) and near_resist  and _n_consecutive_decreasing(p, bars_confirm) and toward_sup

    if buy_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:green", zorder=12, label="BUY ★ (Support reversal)")
    if sell_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:red", zorder=12, label="SELL ★ (Resistance reversal)")

def regression_slope_reversal_at_fib_extremes(series_like,
                                              slope_lb: int,
                                              proximity_pct_of_range: float = 0.02,
                                              confirm_bars: int = 2,
                                              lookback_bars: int = 120):
    """
    Returns dict when BOTH are true:
      1) price touched near Fib 0% (high) or 100% (low)
      2) regression slope sign flipped after that touch
         + confirms reversal via consecutive closes
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return None

    lb = int(max(10, lookback_bars))
    s = s.iloc[-lb:] if len(s) > lb else s
    if len(s) < max(6, int(slope_lb) + 3):
        return None

    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return None

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return None

    near_hi = s >= (hi - tol)
    near_lo = s <= (lo + tol)
    last_hi_touch = near_hi[near_hi].index[-1] if near_hi.any() else None
    last_lo_touch = near_lo[near_lo].index[-1] if near_lo.any() else None

    _, _, _, m_curr, _ = regression_with_band(s, lookback=int(slope_lb))

    def _pre_slope_at(t_touch):
        seg = _coerce_1d_series(s.loc[:t_touch]).dropna().tail(int(slope_lb))
        if len(seg) < 3:
            return np.nan
        _, _, _, m_pre, _ = regression_with_band(seg, lookback=int(slope_lb))
        return float(m_pre) if np.isfinite(m_pre) else np.nan

    buy_rev = None
    if last_lo_touch is not None:
        m_pre = _pre_slope_at(last_lo_touch)
        seg_after = s.loc[last_lo_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) < 0.0) and (float(m_curr) > 0.0) and _n_consecutive_increasing(seg_after, int(confirm_bars)):
                buy_rev = {
                    "side": "BUY",
                    "from_level": "100%",
                    "touch_time": last_lo_touch,
                    "touch_price": float(s.loc[last_lo_touch]) if np.isfinite(s.loc[last_lo_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    sell_rev = None
    if last_hi_touch is not None:
        m_pre = _pre_slope_at(last_hi_touch)
        seg_after = s.loc[last_hi_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) > 0.0) and (float(m_curr) < 0.0) and _n_consecutive_decreasing(seg_after, int(confirm_bars)):
                sell_rev = {
                    "side": "SELL",
                    "from_level": "0%",
                    "touch_time": last_hi_touch,
                    "touch_price": float(s.loc[last_hi_touch]) if np.isfinite(s.loc[last_hi_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    if buy_rev is None and sell_rev is None:
        return None
    if buy_rev is None:
        return sell_rev
    if sell_rev is None:
        return buy_rev

    return buy_rev if buy_rev["touch_time"] >= sell_rev["touch_time"] else sell_rev

def annotate_reverse_possible(ax, rev_info: dict, text: str = "Reverse Possible"):
    if not isinstance(rev_info, dict):
        return
    t = rev_info.get("touch_time", None)
    y = rev_info.get("touch_price", np.nan)
    side = str(rev_info.get("side", "")).upper()
    if t is None or (not np.isfinite(y)):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    va = "bottom" if side == "BUY" else "top"
    ax.text(
        t, y,
        f"  {text}",
        color=col,
        fontsize=10,
        fontweight="bold",
        va=va,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.80),
        zorder=25
    )
# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Sessions (PST)
# ---------------------------
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    start_d = idx[0].astimezone(session_tz).date()
    end_d   = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst  = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ, 8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}

def draw_session_lines(ax, lines: dict, alpha: float = 0.35):
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:orange", alpha=alpha)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=alpha)

    handles = [
        Line2D([0], [0], color="tab:blue",   linestyle="-",  linewidth=1.6, label="London Open"),
        Line2D([0], [0], color="tab:blue",   linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], color="tab:orange", linestyle="-",  linewidth=1.6, label="New York Open"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels

# ---------------------------
# News (Yahoo Finance)
# ---------------------------
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

def draw_news_markers(ax, times, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# ---------------------------
# Channel-in-range helpers for NTD panel
# ---------------------------
def channel_state_series(price: pd.Series, sup: pd.Series, res: pd.Series, eps: float = 0.0) -> pd.Series:
    p = _coerce_1d_series(price)
    s_sup = _coerce_1d_series(sup).reindex(p.index)
    s_res = _coerce_1d_series(res).reindex(p.index)
    state = pd.Series(index=p.index, dtype=float)
    ok = p.notna() & s_sup.notna() & s_res.notna()
    if ok.any():
        below = p < (s_sup - eps)
        above = p > (s_res + eps)
        between = ~(below | above)
        state[ok & below] = -1
        state[ok & between] = 0
        state[ok & above] = 1
    return state

def _true_spans(mask: pd.Series):
    spans = []
    if mask is None or mask.empty:
        return spans
    s = mask.fillna(False).astype(bool)
    start = None
    prev_t = None
    for t, val in s.items():
        if val and start is None:
            start = t
        if not val and start is not None:
            if prev_t is not None:
                spans.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None and prev_t is not None:
        spans.append((start, prev_t))
    return spans

def overlay_inrange_on_ntd(ax, price: pd.Series, sup: pd.Series, res: pd.Series):
    state = channel_state_series(price, sup, res)
    in_mask = (state == 0)
    for a, b in _true_spans(in_mask):
        try:
            ax.axvspan(a, b, color="gold", alpha=0.15, zorder=1)
        except Exception:
            pass
    ax.plot([], [], linewidth=8, color="gold", alpha=0.20, label="In Range (S↔R)")
    enter_from_below = (state.shift(1) == -1) & (state == 0)
    enter_from_above = (state.shift(1) == 1) & (state == 0)
    if enter_from_below.any():
        ax.scatter(price.index[enter_from_below], [0.92]*int(enter_from_below.sum()),
                   marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above], [0.92]*int(enter_from_above.sum()),
                   marker="v", s=60, color="tab:orange", zorder=7, label="Enter from R")

    last = state.dropna().iloc[-1] if state.dropna().shape[0] else np.nan
    if np.isfinite(last):
        if last == 0:
            lbl, col = "IN RANGE (S↔R)", "black"
        elif last > 0:
            lbl, col = "Above R", "tab:orange"
        else:
            lbl, col = "Below S", "tab:red"
        ax.text(0.99, 0.94, lbl, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85))
    return last

def rolling_midline(series_like: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll = s.rolling(window, min_periods=1)
    mid = (roll.max() + roll.min()) / 2.0
    return mid.reindex(s.index)

def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

# ---------------------------
# Cached last values for scanning
# ---------------------------
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


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Recent BUY scanner helpers (uses SAME band-bounce logic as the chart)
# ---------------------------
@st.cache_data(ttl=120)
def last_band_bounce_signal_daily(symbol: str, slope_lb: int):
    try:
        s = fetch_hist(symbol)
        p_full = _coerce_1d_series(s).dropna()
        if p_full.empty:
            return None

        yhat, up, lo, m, r2 = regression_with_band(p_full, lookback=int(slope_lb))
        sig = find_band_bounce_signal(p_full, up, lo, m)
        if sig is None:
            return None

        t = sig.get("time", None)
        if t is None or t not in p_full.index:
            return None

        loc = int(p_full.index.get_loc(t))
        bars_since = int((len(p_full) - 1) - loc)

        curr = float(p_full.iloc[-1]) if np.isfinite(p_full.iloc[-1]) else np.nan
        spx = float(sig.get("price", np.nan))
        dlt = (curr / spx - 1.0) if np.isfinite(curr) and np.isfinite(spx) and spx != 0 else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Side": sig.get("side", ""),
            "Bars Since": bars_since,
            "Signal Time": t,
            "Signal Price": spx,
            "Current Price": curr,
            "DeltaPct": dlt,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_band_bounce_signal_hourly(symbol: str, period: str, slope_lb: int):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if hc.empty:
            return None

        yhat, up, lo, m, r2 = regression_with_band(hc, lookback=int(slope_lb))
        sig = find_band_bounce_signal(hc, up, lo, m)
        if sig is None:
            return None

        bar = sig.get("time", None)
        if bar is None:
            return None
        try:
            bar = int(bar)
        except Exception:
            return None

        n = len(hc)
        if bar < 0 or bar >= n:
            return None
        bars_since = int((n - 1) - bar)

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        curr = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan
        spx = float(sig.get("price", np.nan))
        dlt = (curr / spx - 1.0) if np.isfinite(curr) and np.isfinite(spx) and spx != 0 else np.nan

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Side": sig.get("side", ""),
            "Bars Since": bars_since,
            "Signal Time": ts,
            "Signal Price": spx,
            "Current Price": curr,
            "DeltaPct": dlt,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str):
    try:
        s_full = fetch_hist(symbol)
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 2:
            return None

        x = np.arange(len(close_show), dtype=float)
        m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) <= 0.0:
            return None

        ntd_full = compute_normalized_trend(close_full, window=ntd_win)
        npx_full = compute_normalized_price(close_full, window=ntd_win)

        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        cross_up, _ = _cross_series(npx_show, ntd_show)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)
        if not cross_up.any():
            return None

        t = cross_up[cross_up].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        ntd_at = float(ntd_show.loc[t]) if (t in ntd_show.index and np.isfinite(ntd_show.loc[t])) else np.nan
        npx_at = float(npx_show.loc[t]) if (t in npx_show.index and np.isfinite(npx_show.loc[t])) else np.nan

        ntd_last = float(ntd_show.dropna().iloc[-1]) if len(ntd_show.dropna()) else np.nan
        npx_last = float(npx_show.dropna().iloc[-1]) if len(npx_show.dropna()) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Signal": "NPX↑NTD (Uptrend)",
            "Bars Since": bars_since,
            "Cross Time": t,
            "Global Slope": float(m),
            "Current Price": curr_px,
            "NTD@Cross": ntd_at,
            "NPX@Cross": npx_at,
            "NTD (last)": ntd_last,
            "NPX (last)": npx_last,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_daily_npx_zero_cross_with_local_slope(symbol: str,
                                               ntd_win: int,
                                               daily_view_label: str,
                                               local_slope_lb: int,
                                               max_abs_npx_at_cross: float,
                                               direction: str = "up"):
    try:
        s_full = fetch_hist(symbol)
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        # NOTE: (prior request) uses 0.5 cross level instead of 0.0
        level = 0.5

        prev = npx_show.shift(1)
        if str(direction).lower().startswith("up"):
            cross_mask = (npx_show >= level) & (prev < level)
            sig_label = "NPX 0.5↑"
        else:
            cross_mask = (npx_show <= level) & (prev > level)
            sig_label = "NPX 0.5↓"

        cross_mask = cross_mask.fillna(False)
        if not cross_mask.any():
            return None

        eps = float(max_abs_npx_at_cross)
        near_level = ((npx_show - level).abs() <= eps) & ((prev - level).abs() <= eps)
        cross_mask = cross_mask & near_level.fillna(False)
        if not cross_mask.any():
            return None

        t = cross_mask[cross_mask].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        seg = close_show.loc[:t].tail(int(local_slope_lb))
        seg = _coerce_1d_series(seg).dropna()
        if len(seg) < 2:
            return None
        x = np.arange(len(seg), dtype=float)
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) == 0.0:
            return None

        if sig_label.endswith("↑") and float(m) <= 0.0:
            return None
        if sig_label.endswith("↓") and float(m) >= 0.0:
            return None

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        npx_at = float(npx_show.loc[t]) if (t in npx_show.index and np.isfinite(npx_show.loc[t])) else np.nan
        npx_prev = float(prev.loc[t]) if (t in prev.index and np.isfinite(prev.loc[t])) else np.nan
        npx_last = float(npx_show.dropna().iloc[-1]) if len(npx_show.dropna()) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Daily View": daily_view_label,
            "Signal": sig_label,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Local Slope": float(m),
            "Current Price": curr_px,
            "NPX@Cross": npx_at,
            "NPX(prev)": npx_prev,
            "NPX (last)": npx_last,
            "Zero-Eps": float(eps),
            "Slope LB": int(local_slope_lb),
        }
    except Exception:
        return None

# ---------------------------
# NEW (THIS REQUEST): Fib 0%/100% proximity + reversal-chance helper
# ---------------------------
@st.cache_data(ttl=120)
def fib_extreme_reversal_watch(symbol: str,
                               daily_view_label: str,
                               slope_lb: int,
                               hist_window: int,
                               slope_window: int,
                               horizon: int,
                               proximity_pct_of_range: float = 0.02,
                               confirm_bars: int = 2,
                               lookback_bars_for_trigger: int = 90):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 10:
            return None

        fibs = fibonacci_levels(close_show)
        if not fibs:
            return None

        hi = float(fibs.get("0%", np.nan))
        lo = float(fibs.get("100%", np.nan))
        if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
            return None
        rng = hi - lo
        if not np.isfinite(rng) or rng <= 0:
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        if not np.isfinite(last_px):
            return None

        dist0 = abs(last_px - hi)
        dist100 = abs(last_px - lo)
        thr = float(proximity_pct_of_range) * rng

        near0 = dist0 <= thr
        near100 = dist100 <= thr
        if not (near0 or near100):
            return None

        yhat, up, lo_band, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        rev_prob = slope_reversal_probability(
            close_show, m,
            hist_window=int(hist_window),
            slope_window=int(slope_window),
            horizon=int(horizon)
        )

        trig = fib_reversal_trigger_from_extremes(
            close_show,
            proximity_pct_of_range=float(proximity_pct_of_range),
            confirm_bars=int(confirm_bars),
            lookback_bars=int(lookback_bars_for_trigger),
        )

        near_level = "0%" if (near0 and (dist0 <= dist100)) else ("100%" if near100 else ("0%" if near0 else "100%"))
        dist_pct = (dist0 / rng) if near_level == "0%" else (dist100 / rng)

        return {
            "Symbol": symbol,
            "Near": near_level,
            "Last Price": last_px,
            "Fib 0%": hi,
            "Fib 100%": lo,
            "Dist (% of range)": float(dist_pct),
            f"P(slope rev≤{int(horizon)} bars)": float(rev_prob) if np.isfinite(rev_prob) else np.nan,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Confirmed Trigger": (trig.get("side") + f" from {trig.get('from_level')}") if isinstance(trig, dict) else "",
        }
    except Exception:
        return None

# ---------------------------
# NEW (THIS REQUEST): Daily slope direction helper for new tab
# ---------------------------
@st.cache_data(ttl=120)
def daily_global_slope(symbol: str, daily_view_label: str):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, np.nan, None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 2:
            return np.nan, np.nan, None
        x = np.arange(len(close_show), dtype=float)
        y = close_show.to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
        return float(m), r2, close_show.index[-1]
    except Exception:
        return np.nan, np.nan, None

# ---------------------------
# NEW (THIS REQUEST): Fib 0%/100% "99.9% confidence" (R²≥0.999) confirmed reversal list helper
# ---------------------------
@st.cache_data(ttl=120)
def fib_extreme_confirmed_reversal_999(symbol: str,
                                       daily_view_label: str,
                                       slope_lb: int,
                                       confirm_bars: int,
                                       lookback_bars_for_trigger: int,
                                       proximity_pct_of_range: float = 0.02,
                                       min_r2: float = 0.999):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(10, int(slope_lb)):
            return None

        trig = fib_reversal_trigger_from_extremes(
            close_show,
            proximity_pct_of_range=float(proximity_pct_of_range),
            confirm_bars=int(confirm_bars),
            lookback_bars=int(lookback_bars_for_trigger),
        )
        if not isinstance(trig, dict):
            return None

        yhat, up, lo, m_now, r2_now = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m_now) and np.isfinite(r2_now)):
            return None
        if float(r2_now) < float(min_r2):
            return None

        side = str(trig.get("side", "")).upper()
        want_up = side.startswith("B")
        if want_up and float(m_now) <= 0.0:
            return None
        if (not want_up) and float(m_now) >= 0.0:
            return None

        t_touch = trig.get("touch_time", None)
        if t_touch is None or t_touch not in close_show.index:
            return None
        seg_touch = _coerce_1d_series(close_show.loc[:t_touch]).dropna().tail(int(slope_lb))
        if len(seg_touch) < 2:
            return None
        x = np.arange(len(seg_touch), dtype=float)
        mt, bt = np.polyfit(x, seg_touch.to_numpy(dtype=float), 1)
        m_touch = float(mt) if np.isfinite(mt) else np.nan

        if not (np.isfinite(m_touch) and np.isfinite(m_now)):
            return None
        if np.sign(m_touch) == 0.0 or np.sign(m_now) == 0.0:
            return None
        if np.sign(m_touch) == np.sign(m_now):
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "From Level": str(trig.get("from_level", "")),
            "Side": side,
            "Touch Time": t_touch,
            "Touch Price": float(trig.get("touch_price", np.nan)),
            "Current Price": last_px,
            "Slope (now)": float(m_now),
            "R2 (now)": float(r2_now),
        }
    except Exception:
        return None

# ---------------------------
# NEW (THIS REQUEST): Enhanced Forecast Buy/Sell scanner helper (Daily)
# ---------------------------
@st.cache_data(ttl=120)
def enhanced_sr_buy_sell_candidate(symbol: str,
                                   daily_view_label: str,
                                   sr_lb: int,
                                   prox_pct: float,
                                   recent_bars: int = 3,
                                   sr_slope_lb: int = 20):
    """
    For the 'Enhanced Forecast Buy and Sell' tab (Daily):

    BUY LIST:
      - Support line slope is UP (support_slope > 0)
      - AND price is currently below/at/near Support OR crossed UP through Support within last 1-3 bars

    SELL LIST:
      - Resistance line slope is DOWN (resistance_slope < 0)
      - AND price is currently above/at/near Resistance OR crossed DOWN through Resistance within last 1-3 bars

    Returns dict with flags (Buy OK / Sell OK) or None if insufficient data.
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < max(5, int(sr_lb)):
            return None

        sup = close.rolling(int(sr_lb), min_periods=1).min()
        res = close.rolling(int(sr_lb), min_periods=1).max()

        c_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
        s_last = float(sup.iloc[-1]) if np.isfinite(sup.iloc[-1]) else np.nan
        r_last = float(res.iloc[-1]) if np.isfinite(res.iloc[-1]) else np.nan
        if not np.all(np.isfinite([c_last, s_last, r_last])):
            return None

        def _slope_last(series: pd.Series, lb: int) -> float:
            s = _coerce_1d_series(series).dropna()
            if len(s) < 2:
                return np.nan
            lb = max(2, int(lb))
            s = s.iloc[-lb:] if len(s) > lb else s
            if len(s) < 2:
                return np.nan
            x = np.arange(len(s), dtype=float)
            m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
            return float(m) if np.isfinite(m) else np.nan

        sup_slope = _slope_last(sup, sr_slope_lb)
        res_slope = _slope_last(res, sr_slope_lb)

        cross_up_sup, _ = _cross_series(close, sup)
        cross_up_sup = cross_up_sup.reindex(close.index, fill_value=False)

        _, cross_dn_res = _cross_series(close, res)
        cross_dn_res = cross_dn_res.reindex(close.index, fill_value=False)

        def _bars_since(t_cross):
            if t_cross is None:
                return None
            try:
                loc = int(close.index.get_loc(t_cross))
                return int((len(close) - 1) - loc)
            except Exception:
                return None

        t_cross_up_sup = cross_up_sup[cross_up_sup].index[-1] if cross_up_sup.any() else None
        bs_cross_up_sup = _bars_since(t_cross_up_sup)

        t_cross_dn_res = cross_dn_res[cross_dn_res].index[-1] if cross_dn_res.any() else None
        bs_cross_dn_res = _bars_since(t_cross_dn_res)

        prox = float(prox_pct)
        buy_now = c_last <= s_last * (1.0 + prox)
        sell_now = c_last >= r_last * (1.0 - prox)

        recent_bars = max(1, int(recent_bars))
        buy_cross_recent = (bs_cross_up_sup is not None) and (0 <= bs_cross_up_sup <= recent_bars)
        sell_cross_recent = (bs_cross_dn_res is not None) and (0 <= bs_cross_dn_res <= recent_bars)

        buy_ok = (np.isfinite(sup_slope) and sup_slope > 0.0) and (buy_now or buy_cross_recent)
        sell_ok = (np.isfinite(res_slope) and res_slope < 0.0) and (sell_now or sell_cross_recent)

        dist_sup_pct = (c_last / s_last - 1.0) if s_last != 0 else np.nan
        dist_res_pct = (c_last / r_last - 1.0) if r_last != 0 else np.nan

        return {
            "Symbol": symbol,
            "AsOf": close.index[-1],
            "Close": c_last,
            "Support": s_last,
            "Support Slope": sup_slope,
            "Dist vs Support": dist_sup_pct,
            "Buy Now": bool(buy_now),
            "Buy Cross Time": t_cross_up_sup,
            "Buy Bars Since Cross": bs_cross_up_sup,
            "Buy OK": bool(buy_ok),
            "Resistance": r_last,
            "Resistance Slope": res_slope,
            "Dist vs Resistance": dist_res_pct,
            "Sell Now": bool(sell_now),
            "Sell Cross Time": t_cross_dn_res,
            "Sell Bars Since Cross": bs_cross_dn_res,
            "Sell OK": bool(sell_ok),
        }
    except Exception:
        return None

# ---------------------------
# NEW (THIS REQUEST): Enhanced Support Crossed helper (Daily)
# ---------------------------
@st.cache_data(ttl=120)
def enhanced_sr_retreat_cross(symbol: str,
                              daily_view_label: str,
                              sr_lb: int):
    """
    For the 'Enhanced Support Crossed' tab (Daily), using the SAME S/R lines as Enhanced Forecast:

    Support Retreat (from below moving upward):
      - previous close < support(previous)
      - current close >= support(current)

    Resistance Retreat (from above moving downward):
      - previous close > resistance(previous)
      - current close <= resistance(current)

    Returns dict with the most recent retreat timestamps + bars-since for each side (may be None).
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < max(5, int(sr_lb) + 2):
            return None

        sup = close.rolling(int(sr_lb), min_periods=1).min()
        res = close.rolling(int(sr_lb), min_periods=1).max()

        c_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
        s_last = float(sup.iloc[-1]) if np.isfinite(sup.iloc[-1]) else np.nan
        r_last = float(res.iloc[-1]) if np.isfinite(res.iloc[-1]) else np.nan

        # Retreat masks (explicit "from below/above")
        prev_c = close.shift(1)
        prev_s = sup.shift(1)
        prev_r = res.shift(1)

        retreat_up_sup = (close >= sup) & (prev_c < prev_s)
        retreat_dn_res = (close <= res) & (prev_c > prev_r)

        def _last_event(mask: pd.Series):
            mask = mask.fillna(False)
            if not mask.any():
                return None, None, np.nan
            t = mask[mask].index[-1]
            try:
                loc = int(close.index.get_loc(t))
                bs = int((len(close) - 1) - loc)
            except Exception:
                bs = None
            line_at = np.nan
            try:
                line_at = float(sup.loc[t]) if (t in sup.index and np.isfinite(sup.loc[t])) else np.nan
            except Exception:
                pass
            return t, bs, line_at

        t_sup, bs_sup, sup_at = _last_event(retreat_up_sup)
        t_res, bs_res, res_at = _last_event(retreat_dn_res)

        return {
            "Symbol": symbol,
            "AsOf": close.index[-1],
            "Close": c_last,
            "Support (last)": s_last,
            "Resistance (last)": r_last,
            "Support Retreat Time": t_sup,
            "Support Bars Since": bs_sup,
            "Support@Retreat": sup_at,
            "Resistance Retreat Time": t_res,
            "Resistance Bars Since": bs_res,
            "Resistance@Retreat": res_at,
        }
    except Exception:
        return None


# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Session state init
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

# ---------------------------
# Shared hourly renderer (Stock & Forex)
# ---------------------------
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return None

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill()
    he = hc.ewm(span=20).mean()

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    hma_h = compute_hma(hc, period=hma_period)
    macd_h, macd_sig_h, macd_hist_h = compute_macd(hc)

    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday.columns else pd.Series(dtype=float)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {"High","Low","Close"}.issubset(intraday.columns) and show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(
            intraday["High"], intraday["Low"], intraday["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
            shift_cloud=False
        )
        kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    )

    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    if not psar_h_df.empty:
        psar_h_df = psar_h_df.reindex(hc.index)

    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    slope_sig_h = m_h

    rev_prob_h = slope_reversal_probability(
        hc,
        slope_sig_h,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_hourly,
        horizon=rev_horizon,
    )

    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    ax2w = None
    if show_nrsi:
        fig2, (ax2, ax2w) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        # UPDATED (THIS REQUEST): more bottom room for legend below chart
        plt.subplots_adjust(hspace=0.05, top=0.90, right=0.93, bottom=0.30)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        # UPDATED (THIS REQUEST): more bottom room for legend below chart
        plt.subplots_adjust(top=0.85, right=0.93, bottom=0.30)

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and (not psar_h_df.empty) and ("PSAR" in psar_h_df.columns):
        up_mask = psar_h_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, slope_sig_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax2, news_pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
    )

    macd_sig = find_macd_hma_sr_signal(
        close=hc, hma=hma_h, macd=macd_h, sup=sup_h, res=res_h,
        global_trend_slope=global_m_h, prox=sr_prox_pct
    )

    macd_instr_txt = "MACD/HMA55: n/a"
    if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
        side = macd_sig["side"]
        macd_instr_txt = f"MACD/HMA55: {side} @ {fmt_price_val(macd_sig['price'])}"
        annotate_macd_signal(ax2, macd_sig["time"], macd_sig["price"], side)

    ax2.text(
        0.01, 0.98, macd_instr_txt,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        zorder=20
    )

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_h}]"
    )

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        ax2.text(0.99, 0.02,
                 f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax2.text(0.01, 0.02,
             f"Slope: {fmt_slope(slope_sig_h)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_h)}",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
    ax2.text(0.50, 0.02,
             f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    session_handles = None
    session_labels = None
    if is_forex and show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and not real_times.empty:
        sess = compute_session_lines(real_times)
        sess_pos = {
            "ldn_open": _map_times_to_bar_positions(real_times, sess.get("ldn_open", [])),
            "ldn_close": _map_times_to_bar_positions(real_times, sess.get("ldn_close", [])),
            "ny_open": _map_times_to_bar_positions(real_times, sess.get("ny_open", [])),
            "ny_close": _map_times_to_bar_positions(real_times, sess.get("ny_close", [])),
        }
        session_handles, session_labels = draw_session_lines(ax2, sess_pos)

    # Fibonacci (applies to hourly; sidebar now says "Show Fibonacci")
    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    # UPDATED (THIS REQUEST): Fib BUY/SELL on PRICE chart now also requires slope reversal alignment + NPX direction
    npx_h_for_sig = compute_normalized_price(hc, window=ntd_window)
    fib_sig_h = _fib_npx_zero_signal_series(
        close=hc,
        npx=npx_h_for_sig,
        prox=sr_prox_pct,
        lookback_bars=int(max(3, rev_horizon)),
        slope_lb=int(slope_lb_hourly),
        npx_confirm_bars=1
    )
    if isinstance(fib_sig_h, dict):
        annotate_fib_npx_signal(ax2, fib_sig_h)

    # Write "Reverse Possible" when regression slope has successfully reversed at Fib 0%/100%
    fib_trig_chart = fib_reversal_trigger_from_extremes(
        hc,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_hourly)),
    )
    if isinstance(fib_trig_chart, dict):
        try:
            touch_bar = int(fib_trig_chart.get("touch_time"))
        except Exception:
            touch_bar = None

        m_touch = np.nan
        if touch_bar is not None and 0 <= touch_bar < len(hc):
            seg_touch = _coerce_1d_series(hc.iloc[:touch_bar+1]).dropna().tail(int(slope_lb_hourly))
            if len(seg_touch) >= 2:
                x = np.arange(len(seg_touch), dtype=float)
                mt, bt = np.polyfit(x, seg_touch.to_numpy(dtype=float), 1)
                m_touch = float(mt) if np.isfinite(mt) else np.nan

        m_now = float(m_h) if np.isfinite(m_h) else np.nan
        side_now = str(fib_trig_chart.get("side", "")).upper()
        want_up = side_now.startswith("B")
        slope_ok = (np.isfinite(m_now) and ((want_up and m_now > 0.0) or ((not want_up) and m_now < 0.0)))
        reversed_ok = (np.isfinite(m_touch) and np.isfinite(m_now)
                       and np.sign(m_touch) != 0.0 and np.sign(m_now) != 0.0
                       and np.sign(m_touch) != np.sign(m_now))

        if slope_ok and reversed_ok:
            edge = "tab:green" if want_up else "tab:red"
            ax2.text(
                0.99, 0.90, "Reverse Possible",
                transform=ax2.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=edge,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge, alpha=0.85),
                zorder=25
            )

    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_hourly})")
        ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
            ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
            if not ntd_trend_h.empty:
                ax2w.plot(ntd_trend_h.index, ntd_trend_h.values, "--", linewidth=2,
                          label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")

            overlay_ntd_triangles_by_trend(ax2w, ntd_h, trend_slope=m_h, upper=0.75, lower=-0.75)
            overlay_ntd_sr_reversal_stars(
                ax2w, price=hc, sup=sup_h, res=res_h,
                trend_slope=m_h, ntd=ntd_h, prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm
            )

        if show_ntd_channel:
            overlay_inrange_on_ntd(ax2w, price=hc, sup=sup_h, res=res_h)

        if show_npx_ntd and not _coerce_1d_series(npx_h).dropna().empty and not _coerce_1d_series(ntd_h).dropna().empty:
            overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        if show_hma_rev_ntd and not hma_h.dropna().empty and not hc.dropna().empty:
            overlay_hma_reversal_on_ntd(ax2w, hc, hma_h, lookback=hma_rev_lb,
                                        period=hma_period, ntd=ntd_h)

        ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2w.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax2w.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax2w.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax2w.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        ax2w.set_ylim(-1.1, 1.1)
        ax2w.set_xlabel("Time (PST)")
    else:
        ax2.set_xlabel("Time (PST)")

    # UPDATED (THIS REQUEST): move legends below chart (outside plot area) using a single figure legend
    handles, labels = [], []
    h1, l1 = ax2.get_legend_handles_labels()
    handles += h1; labels += l1
    if ax2w is not None:
        h2, l2 = ax2w.get_legend_handles_labels()
        handles += h2; labels += l2
    if session_handles and session_labels:
        handles += list(session_handles)
        labels += list(session_labels)

    seen = set()
    h_u, l_u = [], []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        h_u.append(h)
        l_u.append(l)

    fig2.legend(
        handles=h_u,
        labels=l_u,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
        fontsize=9,
        framealpha=0.5
    )

    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax2w if ax2w is not None else ax2, real_times, n_ticks=8)

    style_axes(ax2)
    if ax2w is not None:
        style_axes(ax2w)
    xlim_price = ax2.get_xlim()
    st.pyplot(fig2)

    if show_macd and not macd_h.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        # UPDATED (THIS REQUEST): more bottom room for legend below chart
        figm.subplots_adjust(top=0.88, bottom=0.35)
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.set_xlim(xlim_price)
        axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.5, fontsize=9)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)

    # return values so Tab 1 can show instructions below Run Forecast
    trig_disp = None
    if isinstance(fib_trig_chart, dict):
        trig_disp = dict(fib_trig_chart)
        if isinstance(real_times, pd.DatetimeIndex):
            for k in ["touch_time", "last_time"]:
                try:
                    bi = int(trig_disp.get(k))
                    if 0 <= bi < len(real_times):
                        trig_disp[k] = real_times[bi]
                except Exception:
                    pass

    return {
        "trade_instruction": instr_txt,
        "fib_trigger": trig_disp,
    }


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Tabs
# ---------------------------

# UPDATED (THIS REQUEST): force Streamlit tabs to wrap so ALL tabs show by default (no horizontal overflow)
st.markdown(
    """
    <style>
      div[data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        gap: 0.25rem !important;
      }
      div[data-baseweb="tab"] {
        flex: 0 0 auto !important;
      }
      div[data-baseweb="tab"] button {
        padding: 6px 10px !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# UPDATED (THIS REQUEST): added new tab "Enhanced Support Crossed" (all existing tabs unchanged)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "NPX 0.5-Cross Scanner",
    "Fib NPX 0.0 Signal Scanner",            # NEW (THIS REQUEST)
    "Daily Slope+BB Reversal Scanner",
    "Fib 0%/100% Reversal Watchlist",
    "Slope Direction Scan",
    "Fib 0%/100% 99.9% Reversal (R²≥0.999)",
    "Trendline Direction Lists",             # NEW (THIS REQUEST)
    "Enhanced Forecast Buy and Sell",        # NEW (THIS REQUEST)
    "Enhanced Support Crossed"               # NEW (THIS REQUEST)
])

# ---------------------------
# TAB 1: ORIGINAL FORECAST
# ---------------------------
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data is cached for ~2 minutes after first fetch. "
            "Charts stay on the last RUN ticker until you run again.")

    sel = st.selectbox("Ticker:", universe, key=f"orig_ticker_{mode}")
    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key=f"orig_chart_{mode}_v2")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key=f"hour_range_select_{mode}"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    run_clicked = st.button("Run Forecast", key=f"btn_run_forecast_{mode}")

    fib_instruction_box = st.empty()
    trade_instruction_box = st.empty()

    if run_clicked:
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
            "run_all": True,
            "mode_at_run": mode
        })

    if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
        disp_ticker = st.session_state.ticker
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"**Displayed (last run):** {disp_ticker}  •  "
                   f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(disp_ticker, window_days=news_window_days)

        with fib_instruction_box.container():
            st.warning(FIB_ALERT_TEXT)
            st.caption(
                "Fibonacci Reversal Trigger (confirmed): "
                "BUY when price touches near the **100%** line then prints consecutive higher closes; "
                "SELL when price touches near the **0%** line then prints consecutive lower closes."
            )
            st.caption(
                "Fibonacci NPX 0.0 Signal (THIS UPDATE): "
                "BUY when price touched **100%** and NPX crossed **up** through **0.0** recently; "
                "SELL when price touched **0%** and NPX crossed **down** through **0.0** recently."
            )

        daily_instr_txt = None
        hourly_instr_txt = None
        daily_fib_trig = None
        hourly_fib_trig = None

        if chart in ("Daily", "Both"):
            ema30 = df.ewm(span=30).mean()
            res_d = df.rolling(sr_lb_daily, min_periods=1).max()
            sup_d = df.rolling(sr_lb_daily, min_periods=1).min()

            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)
            rev_prob_d = slope_reversal_probability(df, m_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)
            piv = current_daily_pivots(df_ohlc)

            ntd_d = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            npx_d_full = compute_normalized_price(df, window=ntd_window) if show_npx_ntd else pd.Series(index=df.index, dtype=float)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    shift_cloud=False
                )
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res_d_show = res_d.reindex(df_show.index)
            sup_d_show = sup_d.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            ntd_d_show = ntd_d.reindex(df_show.index)
            npx_d_show = npx_d_full.reindex(df_show.index)
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show = bb_up_d.reindex(df_show.index)
            bb_lo_d_show = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show = bb_pctb.reindex(df_show.index) if 'bb_pctb' in globals() else bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_show = compute_hma(df, period=hma_period).reindex(df_show.index)
            macd_d, macd_sig_d, macd_hist_d = compute_macd(df_show)

            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            # UPDATED (THIS REQUEST): more bottom room for legend below chart
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.26)

            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{disp_ticker} Daily — {daily_view} — History, EMA, S/R (w={sr_lb_daily}), Slope, Pivots "
                f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
            )

            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")

            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)

            if show_psar and (not psar_d_df.empty) and ("PSAR" in psar_d_df.columns):
                up_mask = psar_d_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                               label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

            res_val_d = sup_val_d = np.nan
            try:
                res_val_d = float(res_d_show.iloc[-1])
                sup_val_d = float(sup_d_show.iloc[-1])
            except Exception:
                pass
            if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-", linewidth=1.6,
                          label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6,
                          label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
                bounce_sig_d = find_band_bounce_signal(df_show, upper_d_show, lower_d_show, m_d)
                if bounce_sig_d is not None:
                    annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

            macd_sig_d = find_macd_hma_sr_signal(
                close=df_show, hma=hma_d_show, macd=macd_d, sup=sup_d_show, res=res_d_show,
                global_trend_slope=global_m_d, prox=sr_prox_pct
            )
            macd_instr_txt_d = "MACD/HMA55: n/a"
            if macd_sig_d is not None and np.isfinite(macd_sig_d.get("price", np.nan)):
                macd_instr_txt_d = f"MACD/HMA55: {macd_sig_d['side']} @ {fmt_price_val(macd_sig_d['price'])}"
                annotate_macd_signal(ax, macd_sig_d["time"], macd_sig_d["price"], macd_sig_d["side"])

            ax.text(
                0.01, 0.98, macd_instr_txt_d,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
                zorder=30
            )

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            # UPDATED (THIS REQUEST): Fib BUY/SELL on PRICE chart now also requires slope reversal alignment + NPX direction
            fib_sig_d = _fib_npx_zero_signal_series(
                close=df_show,
                npx=npx_d_show,
                prox=sr_prox_pct,
                lookback_bars=int(max(3, rev_horizon)),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if isinstance(fib_sig_d, dict):
                annotate_fib_npx_signal(ax, fib_sig_d)

            # "Reverse Possible" (existing)
            daily_fib_trig = fib_reversal_trigger_from_extremes(
                df_show,
                proximity_pct_of_range=0.02,
                confirm_bars=int(rev_bars_confirm),
                lookback_bars=int(max(60, slope_lb_daily)),
            )
            if isinstance(daily_fib_trig, dict):
                t_touch = daily_fib_trig.get("touch_time", None)
                m_touch = np.nan
                if t_touch is not None and t_touch in df_show.index:
                    seg_touch = _coerce_1d_series(df_show.loc[:t_touch]).dropna().tail(int(slope_lb_daily))
                    if len(seg_touch) >= 2:
                        x = np.arange(len(seg_touch), dtype=float)
                        mt, bt = np.polyfit(x, seg_touch.to_numpy(dtype=float), 1)
                        m_touch = float(mt) if np.isfinite(mt) else np.nan

                m_now = float(m_d) if np.isfinite(m_d) else np.nan
                side_now = str(daily_fib_trig.get("side", "")).upper()
                want_up = side_now.startswith("B")
                slope_ok = (np.isfinite(m_now) and ((want_up and m_now > 0.0) or ((not want_up) and m_now < 0.0)))
                reversed_ok = (np.isfinite(m_touch) and np.isfinite(m_now)
                               and np.sign(m_touch) != 0.0 and np.sign(m_now) != 0.0
                               and np.sign(m_touch) != np.sign(m_now))

                if slope_ok and reversed_ok:
                    edge = "tab:green" if want_up else "tab:red"
                    ax.text(
                        0.99, 0.90, "Reverse Possible",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=10, fontweight="bold", color=edge,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge, alpha=0.85),
                        zorder=35
                    )

            last_px_show = _safe_last_float(df_show)
            if np.isfinite(last_px_show):
                nbb_txt = ""
                try:
                    last_pct = float(bb_pctb_d_show.dropna().iloc[-1]) if show_bbands else np.nan
                    last_nbb = float(bb_nbb_d_show.dropna().iloc[-1]) if show_bbands else np.nan
                    if np.isfinite(last_nbb) and np.isfinite(last_pct):
                        nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                except Exception:
                    pass
                ax.text(0.99, 0.02,
                        f"Current price: {fmt_price_val(last_px_show)}{nbb_txt}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            ax.set_ylabel("Price")
            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            axdw.set_title(f"Daily Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_daily})")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                              label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")
                overlay_ntd_triangles_by_trend(axdw, ntd_d_show, trend_slope=m_d, upper=0.75, lower=-0.75)
                overlay_ntd_sr_reversal_stars(axdw, price=df_show, sup=sup_d_show, res=res_d_show,
                                              trend_slope=m_d, ntd=ntd_d_show, prox=sr_prox_pct,
                                              bars_confirm=rev_bars_confirm)
            if show_npx_ntd and not npx_d_show.dropna().empty and not ntd_d_show.dropna().empty:
                overlay_npx_on_ntd(axdw, npx_d_show, ntd_d_show, mark_crosses=mark_npx_cross)
            if show_hma_rev_ntd and not hma_d_show.dropna().empty and not df_show.dropna().empty:
                overlay_hma_reversal_on_ntd(axdw, df_show, hma_d_show, lookback=hma_rev_lb,
                                            period=hma_period, ntd=ntd_d_show)

            axdw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
            axdw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
            axdw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
            axdw.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
            axdw.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")

            # UPDATED (THIS REQUEST): move legends below chart (outside plot area) using a single figure legend
            handles, labels = [], []
            h1, l1 = ax.get_legend_handles_labels()
            handles += h1; labels += l1
            h2, l2 = axdw.get_legend_handles_labels()
            handles += h2; labels += l2

            seen = set()
            h_u, l_u = [], []
            for h, l in zip(handles, labels):
                if not l or l in seen:
                    continue
                seen.add(l)
                h_u.append(h)
                l_u.append(l)

            fig.legend(
                handles=h_u,
                labels=l_u,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=4,
                frameon=True,
                fontsize=9,
                framealpha=0.5
            )

            style_axes(ax)
            style_axes(axdw)
            st.pyplot(fig)

            if show_macd and not macd_d.dropna().empty:
                figm, axm = plt.subplots(figsize=(14, 2.6))
                # UPDATED (THIS REQUEST): more bottom room for legend below chart
                figm.subplots_adjust(top=0.88, bottom=0.35)
                axm.set_title("MACD (optional)")
                axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
                axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
                axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.5, fontsize=9)
                style_axes(axm)
                st.pyplot(figm)

            daily_instr_txt = format_trade_instruction(
                trend_slope=m_d,
                buy_val=sup_val_d,
                sell_val=res_val_d,
                close_val=last_px_show,
                symbol=disp_ticker,
                global_trend_slope=global_m_d
            )

        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            out_h = render_hourly_views(
                sel=disp_ticker,
                intraday=intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.hour_range,
                is_forex=(mode == "Forex")
            )
            if isinstance(out_h, dict):
                hourly_instr_txt = out_h.get("trade_instruction", None)
                hourly_fib_trig = out_h.get("fib_trigger", None)

        # NEW (THIS REQUEST): Track the last time each instruction text changed (PST)
        if "instr_daily_text" not in st.session_state:
            st.session_state.instr_daily_text = None
        if "instr_daily_updated_at" not in st.session_state:
            st.session_state.instr_daily_updated_at = None
        if "instr_hourly_text" not in st.session_state:
            st.session_state.instr_hourly_text = None
        if "instr_hourly_updated_at" not in st.session_state:
            st.session_state.instr_hourly_updated_at = None

        now_pst = datetime.now(PACIFIC)

        if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
            if st.session_state.get("instr_daily_text") != daily_instr_txt:
                st.session_state.instr_daily_text = daily_instr_txt
                st.session_state.instr_daily_updated_at = now_pst

        if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
            if st.session_state.get("instr_hourly_text") != hourly_instr_txt:
                st.session_state.instr_hourly_text = hourly_instr_txt
                st.session_state.instr_hourly_updated_at = now_pst

        def _ts_str(dt_obj):
            if isinstance(dt_obj, datetime):
                try:
                    d = dt_obj.astimezone(PACIFIC)
                except Exception:
                    d = dt_obj
                return f"{d.strftime('%Y-%m-%d %H:%M:%S')} PST"
            return "n/a"

        with trade_instruction_box.container():
            if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
                daily_msg = f"Daily (updated {_ts_str(st.session_state.get('instr_daily_updated_at'))}): {daily_instr_txt}"
                if daily_instr_txt.startswith("ALERT:"):
                    st.error(daily_msg)
                else:
                    st.success(daily_msg)

            if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
                hourly_msg = f"Hourly (updated {_ts_str(st.session_state.get('instr_hourly_updated_at'))}): {hourly_instr_txt}"
                if hourly_instr_txt.startswith("ALERT:"):
                    st.error(hourly_msg)
                else:
                    st.success(hourly_msg)

            if isinstance(daily_fib_trig, dict):
                st.info(
                    f"Daily Fib Reversal Trigger: **{daily_fib_trig.get('side')}** "
                    f"(from {daily_fib_trig.get('from_level')}) • touch={daily_fib_trig.get('touch_time')} "
                    f"@ {fmt_price_val(daily_fib_trig.get('touch_price', np.nan))}"
                )
            else:
                st.caption("Daily Fib Reversal Trigger: none confirmed.")

            if isinstance(hourly_fib_trig, dict):
                st.info(
                    f"Hourly Fib Reversal Trigger: **{hourly_fib_trig.get('side')}** "
                    f"(from {hourly_fib_trig.get('from_level')}) • touch={hourly_fib_trig.get('touch_time')} "
                    f"@ {fmt_price_val(hourly_fib_trig.get('touch_price', np.nan))}"
                )
            elif chart in ("Hourly", "Both"):
                st.caption("Hourly Fib Reversal Trigger: none confirmed.")

        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# TAB 2: ENHANCED FORECAST
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        df = st.session_state.df_hist
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"Displayed ticker: **{st.session_state.ticker}**  •  Intraday lookback: **{st.session_state.get('hour_range','24h')}**")
        view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}")

        if view in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)
            res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
            sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()
            hma_d_show = compute_hma(df_show, period=hma_period)
            macd_d, macd_sig_d, _ = compute_macd(df_show)

            fig, ax = plt.subplots(figsize=(14, 5))
            # UPDATED (THIS REQUEST): more bottom room for legend below chart
            fig.subplots_adjust(bottom=0.25)
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            # UPDATED (THIS REQUEST): show Support/Resistance values on the Enhanced chart (same look/feel as originals)
            res_val_e = sup_val_e = np.nan
            try:
                res_val_e = float(res_d_show.iloc[-1]) if len(res_d_show) else np.nan
                sup_val_e = float(sup_d_show.iloc[-1]) if len(sup_d_show) else np.nan
            except Exception:
                pass

            if np.isfinite(res_val_e) and np.isfinite(sup_val_e) and len(df_show.index) > 0:
                ax.hlines(res_val_e, xmin=df_show.index[0], xmax=df_show.index[-1],
                          colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(sup_val_e, xmin=df_show.index[0], xmax=df_show.index[-1],
                          colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                label_on_left(ax, res_val_e, f"R {fmt_price_val(res_val_e)}", color="tab:red")
                label_on_left(ax, sup_val_e, f"S {fmt_price_val(sup_val_e)}", color="tab:green")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            # UPDATED (THIS REQUEST): Fib BUY/SELL on Enhanced DAILY price chart now requires slope reversal alignment + NPX direction
            npx_d_show = compute_normalized_price(df_show, window=ntd_window)
            fib_sig_d = _fib_npx_zero_signal_series(
                close=df_show,
                npx=npx_d_show,
                prox=sr_prox_pct,
                lookback_bars=int(max(3, rev_horizon)),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if isinstance(fib_sig_d, dict):
                annotate_fib_npx_signal(ax, fib_sig_d)

            macd_sig = find_macd_hma_sr_signal(df_show, hma_d_show, macd_d, sup_d_show, res_d_show, global_m_d, prox=sr_prox_pct)
            macd_txt = "MACD/HMA55: n/a"
            if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
                macd_txt = f"MACD/HMA55: {macd_sig['side']} @ {fmt_price_val(macd_sig['price'])}"
                annotate_macd_signal(ax, macd_sig["time"], macd_sig["price"], macd_sig["side"])
            ax.text(0.01, 0.98, macd_txt, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8))

            # UPDATED (THIS REQUEST): add Current Price to the Enhanced chart area
            last_px_show = _safe_last_float(df_show)
            if np.isfinite(last_px_show):
                ax.text(
                    0.99, 0.02,
                    f"Current price: {fmt_price_val(last_px_show)}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
                )

            # UPDATED (THIS REQUEST): legend below chart (outside plot area)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.5, fontsize=9)
            style_axes(ax)
            st.pyplot(fig)

            if show_macd and not macd_d.dropna().empty:
                figm, axm = plt.subplots(figsize=(14, 2.6))
                # UPDATED (THIS REQUEST): more bottom room for legend below chart
                figm.subplots_adjust(top=0.88, bottom=0.35)
                axm.set_title("MACD (optional)")
                axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
                axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
                axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, framealpha=0.5, fontsize=9)
                style_axes(axm)
                st.pyplot(figm)

        if view in ("Intraday", "Both"):
            render_hourly_views(
                sel=st.session_state.ticker,
                intraday=st.session_state.intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.get("hour_range","24h"),
                is_forex=(mode == "Forex")
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:, 0],
            "Upper":    ci.iloc[:, 1]
        }, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    st.caption("Simple lookback performance overview (based on Bull/Bear lookback selection).")

    sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
    try:
        dfp = yf.download(sel_bb, period=bb_period, interval="1d")[["Close"]].dropna()
    except Exception:
        dfp = pd.DataFrame()

    if dfp.empty:
        st.warning("No data available.")
    else:
        s = dfp["Close"].astype(float)
        ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if len(s) > 1 else np.nan
        st.metric(label=f"{sel_bb} return over {bb_period}", value=fmt_pct(ret))
        fig, ax = plt.subplots(figsize=(14, 4))
        # UPDATED (THIS REQUEST): more bottom room for legend below chart
        fig.subplots_adjust(bottom=0.25)
        ax.set_title(f"{sel_bb} — {bb_period} Close")
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
        # UPDATED (THIS REQUEST): legend below chart (outside plot area)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 4: METRICS
# ---------------------------
with tab4:
    st.header("Metrics")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        tkr = st.session_state.ticker
        df = st.session_state.df_hist
        intr = st.session_state.intraday

        st.subheader(f"Current ticker: {tkr}")

        yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)
        st.write({
            "Daily slope (reg band)": fmt_slope(m_d),
            "Daily R²": fmt_r2(r2_d),
            f"P(slope reverses ≤ {rev_horizon} bars)": fmt_pct(slope_reversal_probability(df, m_d, rev_hist_lb, slope_lb_daily, rev_horizon))
        })

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({
                "Hourly slope (reg band)": fmt_slope(m_h),
                "Hourly R²": fmt_r2(r2_h),
                f"P(slope reverses ≤ {rev_horizon} bars) hourly": fmt_pct(slope_reversal_probability(hc, m_h, rev_hist_lb, slope_lb_hourly, rev_horizon))
            })

# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Lists symbols where the latest NTD is below -0.75 (using latest intraday for hourly; daily uses daily close).")

    scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
    run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

    if run_scan:
        rows = []
        if scan_frame.startswith("Hourly"):
            period = "1d"
            for sym in universe:
                val, ts = last_hourly_ntd_value(sym, ntd_window, period=period)
                if np.isfinite(val) and val < -0.75:
                    rows.append({"Symbol": sym, "NTD": float(val), "Time": ts})
        else:
            for sym in universe:
                val, ts = last_daily_ntd_value(sym, ntd_window)
                if np.isfinite(val) and val < -0.75:
                    rows.append({"Symbol": sym, "NTD": float(val), "Time": ts})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values("NTD")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 6: LONG-TERM HISTORY
# ---------------------------
with tab6:
    st.header("Long-Term History")
    sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    try:
        smax = fetch_hist_max(sel_lt)
    except Exception:
        smax = pd.Series(dtype=float)

    if smax is None or smax.dropna().empty:
        st.warning("No long-term history available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        # UPDATED (THIS REQUEST): more bottom room for legend below chart
        fig.subplots_adjust(bottom=0.25)
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        # UPDATED (THIS REQUEST): legend below chart (outside plot area)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: RECENT BUY SCANNER
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner — Daily NPX↑NTD in Uptrend (Stocks + Forex)")
    st.caption(
        "Lists symbols (in the current mode's universe) where **NPX (normalized price)** most recently crossed "
        "**ABOVE** the **NTD** line (the green circle condition) **AND** the DAILY chart-area global trendline "
        "(in the selected Daily view range) is **upward**."
    )

    max_bars = st.slider("Max bars since NPX↑NTD cross", 0, 20, 2, 1, key="buy_scan_npx_max_bars")
    run_buy_scan = st.button("Run Recent BUY Scan", key="btn_run_recent_buy_scan_npx")

    if run_buy_scan:
        rows = []
        for sym in universe:
            r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view)
            if r is not None and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)

        if not rows:
            st.info("No recent NPX↑NTD crosses found in an upward daily global trend (within the selected bar window).")
        else:
            out = pd.DataFrame(rows)
            if "Bars Since" in out.columns:
                out["Bars Since"] = out["Bars Since"].astype(int)
            if "Global Slope" in out.columns:
                out["Global Slope"] = out["Global Slope"].astype(float)
            out = out.sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 8: NPX 0.5-CROSS SCANNER
# ---------------------------
with tab8:
    st.header("NPX 0.5-Cross Scanner — Local Slope Confirmed (Daily)")
    st.caption(
        "Scans the current universe for symbols where **NPX (normalized price)** has **recently crossed 0.5** "
        "(with NPX very close to 0.5 at the crossing) and the **local price slope** agrees:\n"
        "• **UP list:** NPX crosses **up** through 0.5 AND local price slope is **up**\n"
        "• **DOWN list:** NPX crosses **down** through 0.5 AND local price slope is **down**"
    )

    c1, c2, c3 = st.columns(3)
    max_bars0 = c1.slider("Max bars since NPX 0.5-cross", 0, 30, 2, 1, key="npx0_max_bars")
    eps0 = c2.slider("Max |NPX-0.5| at cross (near 0.5)", 0.01, 0.30, 0.08, 0.01, key="npx0_eps")
    lb_local = c3.slider("Local slope lookback (bars)", 10, 360, int(slope_lb_daily), 10, key="npx0_slope_lb")

    run0 = st.button("Run NPX 0.5-Cross Scan", key="btn_run_npx0_scan")

    if run0:
        rows_up, rows_dn = [], []
        for sym in universe:
            r_up = last_daily_npx_zero_cross_with_local_slope(
                sym, ntd_win=ntd_window, daily_view_label=daily_view,
                local_slope_lb=lb_local, max_abs_npx_at_cross=eps0, direction="up"
            )
            if r_up is not None and int(r_up.get("Bars Since", 9999)) <= int(max_bars0):
                rows_up.append(r_up)

            r_dn = last_daily_npx_zero_cross_with_local_slope(
                sym, ntd_win=ntd_window, daily_view_label=daily_view,
                local_slope_lb=lb_local, max_abs_npx_at_cross=eps0, direction="down"
            )
            if r_dn is not None and int(r_dn.get("Bars Since", 9999)) <= int(max_bars0):
                rows_dn.append(r_dn)

        left, right = st.columns(2)

        with left:
            st.subheader("NPX 0.5↑ with Local UP Slope")
            if not rows_up:
                st.info("No matches.")
            else:
                out_up = pd.DataFrame(rows_up)
                out_up["Bars Since"] = out_up["Bars Since"].astype(int)
                out_up["Local Slope"] = out_up["Local Slope"].astype(float)
                out_up = out_up.sort_values(["Bars Since", "Local Slope"], ascending=[True, False])
                st.dataframe(out_up.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("NPX 0.5↓ with Local DOWN Slope")
            if not rows_dn:
                st.info("No matches.")
            else:
                out_dn = pd.DataFrame(rows_dn)
                out_dn["Bars Since"] = out_dn["Bars Since"].astype(int)
                out_dn["Local Slope"] = out_dn["Local Slope"].astype(float)
                out_dn = out_dn.sort_values(["Bars Since", "Local Slope"], ascending=[True, True])
                st.dataframe(out_dn.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 9: Fib NPX 0.0 Signal Scanner (NEW)
# ---------------------------
with tab9:
    st.header("Fib NPX 0.0 Signal Scanner")
    st.caption(
        "Scans the current universe for **Fibonacci BUY/SELL** signals on the **Daily** chart:\n"
        "• **Fib BUY:** price touched **100%** (low) and NPX crossed **UP** through **0.0** recently\n"
        "• **Fib SELL:** price touched **0%** (high) and NPX crossed **DOWN** through **0.0** recently\n\n"
        "Uses the selected Daily view range and the existing S/R proximity setting for touch tolerance."
    )

    c1, c2 = st.columns(2)
    lb_sig = c1.slider("Lookback window (bars) for touch + NPX cross", 2, 90, int(max(3, rev_horizon)), 1, key="fibnpx0_lb")
    run_fibsig = c2.button("Run Fib NPX 0.0 Scan", key=f"btn_run_fibnpx0_{mode}")

    if run_fibsig:
        buy_rows, sell_rows = [], []
        for sym in universe:
            rb = last_daily_fib_npx_zero_signal(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=ntd_window,
                direction="BUY",
                prox=sr_prox_pct,
                lookback_bars=int(lb_sig),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if rb is not None:
                buy_rows.append(rb)

            rs = last_daily_fib_npx_zero_signal(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=ntd_window,
                direction="SELL",
                prox=sr_prox_pct,
                lookback_bars=int(lb_sig),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if rs is not None:
                sell_rows.append(rs)

        left, right = st.columns(2)
        with left:
            st.subheader("Fib BUY — 100% touch + NPX 0.0↑")
            if not buy_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(buy_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Fib SELL — 0% touch + NPX 0.0↓")
            if not sell_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(sell_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 10: Daily Slope + S/R reversal + BB mid cross scanner
# ---------------------------
with tab10:
    st.header("Daily Slope + S/R Reversal + BB Midline Scanner (R² ≥ 0.99)")
    st.caption(
        "BUY list:\n"
        "• Daily regression slope is UP and **R² ≥ 0.99** (99% confidence)\n"
        "• Price reversed from Support (touch within horizon + confirmed reversal)\n"
        "• Price crossed ABOVE BB midline\n\n"
        "SELL list:\n"
        "• Daily regression slope is DOWN and **R² ≥ 0.99** (99% confidence)\n"
        "• Price reversed from Resistance (touch within horizon + confirmed reversal)\n"
        "• Price crossed BELOW BB midline"
    )

    c1, c2 = st.columns(2)
    max_bars_since = c1.slider("Max bars since BB mid cross", 0, 60, 10, 1, key="srbb_max_bars_since")
    r2_thr = c2.slider("Min R² (confidence)", 0.80, 0.99, 0.99, 0.01, key="srbb_r2_thr")

    run_scan = st.button("Run Daily Slope+BB Scan", key="btn_run_daily_slope_bb_scan")

    if run_scan:
        buy_rows, sell_rows = [], []
        for sym in universe:
            rb = last_daily_sr_reversal_bbmid(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                sr_lb=sr_lb_daily,
                bb_window=bb_win,
                bb_sigma=bb_mult,
                bb_ema=bb_use_ema,
                prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm,
                horizon=rev_horizon,
                side="BUY",
                min_r2=float(r2_thr),
            )
            if rb is not None and int(rb.get("Bars Since Cross", 9999)) <= int(max_bars_since):
                buy_rows.append(rb)

            rs = last_daily_sr_reversal_bbmid(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                sr_lb=sr_lb_daily,
                bb_window=bb_win,
                bb_sigma=bb_mult,
                bb_ema=bb_use_ema,
                prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm,
                horizon=rev_horizon,
                side="SELL",
                min_r2=float(r2_thr),
            )
            if rs is not None and int(rs.get("Bars Since Cross", 9999)) <= int(max_bars_since):
                sell_rows.append(rs)

        left, right = st.columns(2)

        with left:
            st.subheader("BUY — Up Slope + Support Reversal + BB Mid Cross")
            if not buy_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(buy_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out["Slope"] = out["Slope"].astype(float)
                out["R2"] = out["R2"].astype(float)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("SELL — Down Slope + Resistance Reversal + BB Mid Cross")
            if not sell_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(sell_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out["Slope"] = out["Slope"].astype(float)
                out["R2"] = out["R2"].astype(float)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 11: Fib 0% / 100% proximity + reversal chance
# ---------------------------
with tab11:
    st.header("Fib 0%/100% Reversal Watchlist")
    st.caption(
        "Lists symbols close to the Fibonacci **0%** (high) or **100%** (low) lines, "
        "and includes a slope-reversal probability estimate + any confirmed Fib reversal trigger."
    )

    c1, c2, c3 = st.columns(3)
    prox_pct = c1.slider("Proximity to 0%/100% (as % of Fib range)", 0.005, 0.08, 0.02, 0.005, key="fibwatch_prox")
    min_rev = c2.slider(f"Min P(slope rev≤{rev_horizon} bars)", 0.00, 0.95, 0.25, 0.05, key="fibwatch_minrev")
    run_watch = c3.button("Run Fib Watchlist", key=f"btn_run_fib_watch_{mode}")

    if run_watch:
        rows = []
        for sym in universe:
            r = fib_extreme_reversal_watch(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                hist_window=rev_hist_lb,
                slope_window=slope_lb_daily,
                horizon=rev_horizon,
                proximity_pct_of_range=float(prox_pct),
                confirm_bars=int(rev_bars_confirm),
                lookback_bars_for_trigger=int(max(60, slope_lb_daily)),
            )
            if r is None:
                continue
            pcol = f"P(slope rev≤{int(rev_horizon)} bars)"
            pv = r.get(pcol, np.nan)
            if np.isfinite(pv) and float(pv) < float(min_rev):
                continue
            rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            pcol = f"P(slope rev≤{int(rev_horizon)} bars)"
            if "Dist (% of range)" in out.columns:
                out["Dist (% of range)"] = out["Dist (% of range)"].astype(float)
            if pcol in out.columns:
                out[pcol] = out[pcol].astype(float)

            left, right = st.columns(2)
            with left:
                st.subheader("Near 100% (Low) — potential BUY area")
                out100 = out[out["Near"] == "100%"].copy()
                if out100.empty:
                    st.info("No matches.")
                else:
                    out100 = out100.sort_values(["Dist (% of range)", pcol], ascending=[True, False])
                    st.dataframe(out100.reset_index(drop=True), use_container_width=True)

            with right:
                st.subheader("Near 0% (High) — potential SELL area")
                out0 = out[out["Near"] == "0%"].copy()
                if out0.empty:
                    st.info("No matches.")
                else:
                    out0 = out0.sort_values(["Dist (% of range)", pcol], ascending=[True, False])
                    st.dataframe(out0.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 12: Slope Direction Scan
# ---------------------------
with tab12:
    st.header("Slope Direction Scan")
    st.caption(
        "Lists symbols whose **current DAILY global trendline slope** is **up** vs **down** "
        "(based on the selected Daily view range)."
    )

    run_slope = st.button("Run Slope Direction Scan", key=f"btn_run_slope_dir_{mode}")

    if run_slope:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue
            rows.append({
                "Symbol": sym,
                "Slope": float(m),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "AsOf": ts
            })

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            up = out[out["Slope"] > 0].sort_values(["Slope"], ascending=False)
            dn = out[out["Slope"] < 0].sort_values(["Slope"], ascending=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Upward Slope")
                if up.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(up.reset_index(drop=True), use_container_width=True)

            with right:
                st.subheader("Downward Slope")
                if dn.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(dn.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 13: Fib 0%/100% 99.9% Reversal (R²≥0.999)
# ---------------------------
with tab13:
    st.header("Fib 0%/100% 99.9% Reversal (R² ≥ 0.999)")
    st.caption(
        "Lists symbols that have a **CONFIRMED Fib extreme reversal** (touch + consecutive closes) "
        "AND have **R² ≥ 0.999** on the current regression window, with the regression slope having "
        "successfully reversed sign from the touch-window slope."
    )

    run_fib999 = st.button("Run 99.9% Fib Reversal Scan", key=f"btn_run_fib999_{mode}")

    if run_fib999:
        rows = []
        for sym in universe:
            r = fib_extreme_confirmed_reversal_999(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                confirm_bars=int(rev_bars_confirm),
                lookback_bars_for_trigger=int(max(60, slope_lb_daily)),
                proximity_pct_of_range=0.02,
                min_r2=0.999,
            )
            if r is not None:
                rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            left, right = st.columns(2)

            with left:
                st.subheader("Touched 100% (Low) — reversal up (BUY)")
                out100 = out[out["From Level"] == "100%"].copy()
                if out100.empty:
                    st.info("No matches.")
                else:
                    out100 = out100.sort_values(["R2 (now)", "Slope (now)"], ascending=[False, False])
                    st.dataframe(out100.reset_index(drop=True), use_container_width=True)

            with right:
                st.subheader("Touched 0% (High) — reversal down (SELL)")
                out0 = out[out["From Level"] == "0%"].copy()
                if out0.empty:
                    st.info("No matches.")
                else:
                    out0 = out0.sort_values(["R2 (now)", "Slope (now)"], ascending=[False, True])
                    st.dataframe(out0.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 14: Trendline Direction Lists (NEW)
# ---------------------------
with tab14:
    st.header("Trendline Direction Lists")
    st.caption(
        "Displays symbols whose **current DAILY chart-area global trendline** is:\n"
        "• **Upward** (green dashed global trendline)\n"
        "• **Downward** (red dashed global trendline)\n\n"
        "Uses the selected Daily view range."
    )

    run_trend_lists = st.button("Run Trendline Direction Lists", key=f"btn_run_trendline_lists_{mode}")

    if run_trend_lists:
        up_syms, dn_syms = [], []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue
            if float(m) >= 0.0:
                up_syms.append(sym)
            else:
                dn_syms.append(sym)

        left, right = st.columns(2)

        with left:
            st.subheader("Upward Trend (Green dashed)")
            if not up_syms:
                st.info("No matches.")
            else:
                st.dataframe(pd.DataFrame({"Symbol": sorted(up_syms)}), use_container_width=True)

        with right:
            st.subheader("Downward Trend (Red dashed)")
            if not dn_syms:
                st.info("No matches.")
            else:
                st.dataframe(pd.DataFrame({"Symbol": sorted(dn_syms)}), use_container_width=True)
    else:
        st.info("Click **Run Trendline Direction Lists** to scan the current universe.")

# ---------------------------
# TAB 15: Enhanced Forecast Buy and Sell (NEW)
# ---------------------------
with tab15:
    st.header("Enhanced Forecast Buy and Sell")
    st.caption(
        "Daily scan based on the existing **Daily view range**, **Daily S/R window**, and **S/R proximity (%)**.\n\n"
        "✅ **Buy List**: symbols where price is currently **below/at/near Support** OR **crossed UP through Support** in the last **1–3** bars, "
        "**AND** the **Support line is sloping upward**.\n"
        "✅ **Sell List**: symbols where price is currently **above/at/near Resistance** OR **crossed DOWN through Resistance** in the last **1–3** bars, "
        "**AND** the **Resistance line is sloping downward**."
    )

    c1, c2 = st.columns(2)
    recent_bars = c1.slider("Recent cross window (bars)", 1, 3, 3, 1, key=f"enh_buysell_recent_{mode}")
    sr_slope_lb = c2.slider("S/R slope lookback (bars)", 5, 60, min(20, int(sr_lb_daily)), 5, key=f"enh_buysell_srslope_{mode}")

    run_bs = st.button("Run Enhanced Buy/Sell Scan", key=f"btn_run_enh_buysell_{mode}")

    if run_bs:
        buy_rows, sell_rows = [], []

        for sym in universe:
            r = enhanced_sr_buy_sell_candidate(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=sr_lb_daily,
                prox_pct=sr_prox_pct,
                recent_bars=int(recent_bars),
                sr_slope_lb=int(sr_slope_lb),
            )
            if not isinstance(r, dict):
                continue

            if bool(r.get("Buy OK", False)):
                trig = "At/Below Support" if bool(r.get("Buy Now", False)) else "Crossed UP (≤3 bars)"
                buy_rows.append({
                    "Symbol": r.get("Symbol"),
                    "AsOf": r.get("AsOf"),
                    "Close": r.get("Close"),
                    "Support": r.get("Support"),
                    "Support Slope": r.get("Support Slope"),
                    "Dist vs Support": r.get("Dist vs Support"),
                    "Trigger": trig,
                    "Bars Since Cross": r.get("Buy Bars Since Cross"),
                    "Cross Time": r.get("Buy Cross Time"),
                })

            if bool(r.get("Sell OK", False)):
                trig = "At/Above Resistance" if bool(r.get("Sell Now", False)) else "Crossed DOWN (≤3 bars)"
                sell_rows.append({
                    "Symbol": r.get("Symbol"),
                    "AsOf": r.get("AsOf"),
                    "Close": r.get("Close"),
                    "Resistance": r.get("Resistance"),
                    "Resistance Slope": r.get("Resistance Slope"),
                    "Dist vs Resistance": r.get("Dist vs Resistance"),
                    "Trigger": trig,
                    "Bars Since Cross": r.get("Sell Bars Since Cross"),
                    "Cross Time": r.get("Sell Cross Time"),
                })

        left, right = st.columns(2)

        with left:
            st.subheader("Buy List (Support slope UP)")
            if not buy_rows:
                st.info("No matches.")
            else:
                outb = pd.DataFrame(buy_rows)
                if "Bars Since Cross" in outb.columns:
                    outb["Bars Since Cross"] = pd.to_numeric(outb["Bars Since Cross"], errors="coerce")
                if "Support Slope" in outb.columns:
                    outb["Support Slope"] = pd.to_numeric(outb["Support Slope"], errors="coerce")
                if "Dist vs Support" in outb.columns:
                    outb["Dist vs Support"] = pd.to_numeric(outb["Dist vs Support"], errors="coerce")

                # Sort: best = currently at/below support first, then most recent crosses, then closest distance
                outb["_buy_now_rank"] = (outb["Trigger"] == "At/Below Support").astype(int)
                outb["_bars_rank"] = outb["Bars Since Cross"].fillna(9999)
                outb = outb.sort_values(["_buy_now_rank", "_bars_rank", "Dist vs Support"], ascending=[False, True, True]).drop(columns=["_buy_now_rank","_bars_rank"])
                st.dataframe(outb.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Sell List (Resistance slope DOWN)")
            if not sell_rows:
                st.info("No matches.")
            else:
                outs = pd.DataFrame(sell_rows)
                if "Bars Since Cross" in outs.columns:
                    outs["Bars Since Cross"] = pd.to_numeric(outs["Bars Since Cross"], errors="coerce")
                if "Resistance Slope" in outs.columns:
                    outs["Resistance Slope"] = pd.to_numeric(outs["Resistance Slope"], errors="coerce")
                if "Dist vs Resistance" in outs.columns:
                    outs["Dist vs Resistance"] = pd.to_numeric(outs["Dist vs Resistance"], errors="coerce")

                outs["_sell_now_rank"] = (outs["Trigger"] == "At/Above Resistance").astype(int)
                outs["_bars_rank"] = outs["Bars Since Cross"].fillna(9999)
                outs = outs.sort_values(["_sell_now_rank", "_bars_rank", "Dist vs Resistance"], ascending=[False, True, False]).drop(columns=["_sell_now_rank","_bars_rank"])
                st.dataframe(outs.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run Enhanced Buy/Sell Scan** to scan the current universe.")

# ---------------------------
# TAB 16: Enhanced Support Crossed (NEW)
# ---------------------------
with tab16:
    st.header("Enhanced Support Crossed")
    st.caption(
        "Daily scan using the same **Daily view range** and **Daily S/R window** as the Enhanced Forecast chart.\n\n"
        "✅ **Support Retreat (Up):** price was **below** Support, then moved **upward** to **at/above** Support.\n"
        "✅ **Resistance Retreat (Down):** price was **above** Resistance, then moved **downward** to **at/below** Resistance."
    )

    c1, c2 = st.columns(2)
    max_bars = c1.slider("Max bars since retreat event", 0, 60, 3, 1, key=f"enh_retreat_maxbars_{mode}")
    run_retreat = c2.button("Run Enhanced Support Crossed Scan", key=f"btn_run_enh_retreat_{mode}")

    if run_retreat:
        up_rows, dn_rows = [], []
        for sym in universe:
            r = enhanced_sr_retreat_cross(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=sr_lb_daily
            )
            if not isinstance(r, dict):
                continue

            bs_sup = r.get("Support Bars Since", None)
            bs_res = r.get("Resistance Bars Since", None)

            if bs_sup is not None and np.isfinite(bs_sup) and int(bs_sup) <= int(max_bars):
                up_rows.append({
                    "Symbol": r.get("Symbol"),
                    "AsOf": r.get("AsOf"),
                    "Close": r.get("Close"),
                    "Support (last)": r.get("Support (last)"),
                    "Support@Retreat": r.get("Support@Retreat"),
                    "Retreat Time": r.get("Support Retreat Time"),
                    "Bars Since": int(bs_sup),
                })

            if bs_res is not None and np.isfinite(bs_res) and int(bs_res) <= int(max_bars):
                dn_rows.append({
                    "Symbol": r.get("Symbol"),
                    "AsOf": r.get("AsOf"),
                    "Close": r.get("Close"),
                    "Resistance (last)": r.get("Resistance (last)"),
                    "Resistance@Retreat": r.get("Resistance@Retreat"),
                    "Retreat Time": r.get("Resistance Retreat Time"),
                    "Bars Since": int(bs_res),
                })

        left, right = st.columns(2)
        with left:
            st.subheader("Support Retreat (from below → up)")
            if not up_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(up_rows)
                out["Bars Since"] = pd.to_numeric(out["Bars Since"], errors="coerce")
                out = out.sort_values(["Bars Since"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Resistance Retreat (from above → down)")
            if not dn_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(dn_rows)
                out["Bars Since"] = pd.to_numeric(out["Bars Since"], errors="coerce")
                out = out.sort_values(["Bars Since"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run Enhanced Support Crossed Scan** to scan the current universe.")
