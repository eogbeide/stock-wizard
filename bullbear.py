# =========================
# Part 1/6 — bullbear.py
# =========================
# bullbear.py — Stocks/Forex Dashboard + Forecasts
#
# UPDATE (This request): FIX the visible intraday "time gap" (stretched empty region)
#   • Keep price gap adjustment (make_gapless_ohlc) AND remove time-axis gaps by plotting on
#     a gapless bar index (0..N-1), while labeling ticks with PST timestamps.
#   • Full code, no omissions (6 parts).
#
# Includes (core features):
#   • Market switch buttons: Forex / Stocks (Forex default on first open).
#   • Details page for selected market + ticker (persists after Run).
#   • Daily + Intraday charts
#   • Supertrend, PSAR, Bollinger Bands, HMA(55), Ichimoku Kijun(26)
#   • Regression slope line + ±2σ bands (thicker/darker)
#   • Band-bounce BUY/SELL markers (only when price re-enters from outside ±2σ bands)
#   • Forex-only volume panel (with mid-line + trendline)
#   • Momentum ROC% panel
#   • NTD + NPX panel with triangles/stars + in-range shading
#   • Session lines (London/NY in PST) + optional news markers (Forex)
#   • Sidebar show/hide + Clear Cache button
#
# Requirements:
#   pip install streamlit yfinance pandas numpy matplotlib

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# ---------------------------
# App config
# ---------------------------
st.set_page_config(
    page_title="BullBear — Stocks/Forex Dashboard",
    layout="wide",
)

PACIFIC = ZoneInfo("America/Los_Angeles")

# ---------------------------
# Session state defaults (applies ONLY on first open)
# ---------------------------
def _ss_init_once():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # Default market = Forex (first open only)
        st.session_state.market = "Forex"

        # Persist last run selection (so reruns do not reset)
        st.session_state.last_run_market = "Forex"
        st.session_state.last_run_symbol = "EURUSD=X"
        st.session_state.last_run_range = "48h"

        # UI toggles
        st.session_state.show_sidebar = True

_ss_init_once()

# ---------------------------
# Universes
# ---------------------------
FOREX_UNIVERSE = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X",
]

STOCK_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "INTC", "JPM", "BAC", "XOM", "CVX", "GLD", "SLV",
]

def universe_for(market: str) -> list[str]:
    return FOREX_UNIVERSE if market == "Forex" else STOCK_UNIVERSE

def is_forex_symbol(sym: str) -> bool:
    return sym.endswith("=X")

# ---------------------------
# Formatting helpers
# ---------------------------
def fmt_price_val(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 10:
        return f"{x:.3f}"
    return f"{x:.5f}"

def fmt_slope(m: float) -> str:
    if not np.isfinite(m):
        return "n/a"
    return f"{m:+.6f}"

def fmt_r2(r2: float) -> str:
    if not np.isfinite(r2):
        return "n/a"
    return f"{r2:.3f}"

def fmt_pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x*100:.{digits}f}%"

def label_on_left(ax, y: float, text: str, color: str = "black"):
    xmin, xmax = ax.get_xlim()
    ax.text(
        xmin, y, f" {text}",
        va="center", ha="left",
        fontsize=9, color=color,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.75),
        zorder=20
    )

def _coerce_1d_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.Index, np.ndarray, list)):
        return pd.Series(x)
    return pd.Series(dtype=float)

# ---------------------------
# Core data fetch
# ---------------------------
@st.cache_data(ttl=600)
def fetch_daily(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d")
    if df is None or df.empty:
        return df
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # yfinance daily often timezone-naive; treat as UTC midnight-ish then convert for consistent labels
    try:
        df.index = df.index.tz_localize("UTC")
    except Exception:
        pass
    try:
        df.index = df.index.tz_convert(PACIFIC)
    except Exception:
        pass
    return df

def make_gapless_ohlc(df: pd.DataFrame, gap_threshold: float = 0.02) -> pd.DataFrame:
    """
    Adjust OHLC so overnight/weekend gaps don't create price discontinuities.
    This does NOT remove time gaps. Time-axis gap removal is handled separately.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return df

    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    gap = (df["Open"].astype(float) - prev_close) / prev_close
    gap = gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    adj = np.zeros(len(df), dtype=float)
    # cumulative adjustment applied to all subsequent bars after a gap
    for i in range(1, len(df)):
        g = float(gap.iloc[i])
        if np.isfinite(g) and abs(g) >= gap_threshold:
            # shift so Open aligns with prev close (remove the gap)
            shift = float(df["Open"].iloc[i] - prev_close.iloc[i])
            adj[i:] -= shift

    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].astype(float) + adj
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return df

    # Ensure clean, monotonic time index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    try:
        df.index = df.index.tz_localize("UTC")
    except Exception:
        pass
    try:
        df.index = df.index.tz_convert(PACIFIC)
    except Exception:
        pass

    # Remove price gaps so intraday price is continuous (gapless OHLC)
    df = make_gapless_ohlc(df)

    # Re-sort after transforms
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

# ---------------------------
# Indicator utilities
# ---------------------------
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    """
    Returns DataFrame with columns: ST, in_uptrend
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame()

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    hl2 = (high + low) / 2.0
    _atr = atr(high, low, close, period=atr_period)

    upperband = hl2 + atr_mult * _atr
    lowerband = hl2 - atr_mult * _atr

    st = pd.Series(index=df.index, dtype=float)
    in_uptrend = pd.Series(index=df.index, dtype=bool)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
            in_uptrend.iloc[i] = True
            continue

        prev_st = st.iloc[i - 1]
        prev_up = in_uptrend.iloc[i - 1]

        if close.iloc[i] > upperband.iloc[i - 1]:
            in_uptrend.iloc[i] = True
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = prev_up
            if in_uptrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if (not in_uptrend.iloc[i]) and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        st.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]

    out = pd.DataFrame({"ST": st, "in_uptrend": in_uptrend})
    return out

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """
    Parabolic SAR (simple implementation).
    Returns DataFrame with columns: PSAR, in_uptrend
    """
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return pd.DataFrame()

    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()

    psar = np.zeros(len(df), dtype=float)
    uptrend = True
    af = step
    ep = high[0]
    psar[0] = low[0]

    for i in range(1, len(df)):
        prev_psar = psar[i - 1]

        if uptrend:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i])  # clamp
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = low[i]
                af = step
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i])  # clamp
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = high[i]
                af = step

    return pd.DataFrame({"PSAR": psar, "in_uptrend": uptrend}, index=df.index)

def wma(s: pd.Series, period: int) -> pd.Series:
    s = s.astype(float)
    w = np.arange(1, period + 1, dtype=float)
    return s.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    if period < 2:
        return close
    half = max(1, period // 2)
    sqrtp = max(1, int(math.sqrt(period)))
    return wma(2 * wma(close, half) - wma(close, period), sqrtp)

def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    c = close.astype(float)
    mid = c.ewm(span=window, adjust=False).mean() if use_ema else c.rolling(window).mean()
    sd = c.rolling(window).std(ddof=0)
    up = mid + mult * sd
    lo = mid - mult * sd
    pctb = (c - lo) / (up - lo)
    nbb = (c - mid) / (sd.replace(0, np.nan))
    return mid, up, lo, pctb, nbb

def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2.0
    kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0)
    span_bv = (high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0
    chikou = close.shift(-base)

    if shift_cloud:
        span_a = span_a.shift(base)
        span_bv = span_bv.shift(base)

    return tenkan, kijun, span_a, span_bv, chikou
# =========================
# Part 2/6 — bullbear.py
# =========================
# Regression, normalization, signals, and "gapless time-axis" helpers

def rolling_midline(s: pd.Series, window: int = 55) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()

def slope_line(s: pd.Series, window: int) -> tuple[pd.Series, float]:
    """
    Fit a line on the last `window` points and return yhat across full index + slope.
    """
    s = _coerce_1d_series(s).astype(float)
    s = s.copy()
    idx = s.index
    if len(s) < 2:
        return pd.Series(index=idx, dtype=float), np.nan
    w = min(window, len(s))
    y = s.iloc[-w:].to_numpy(dtype=float)
    x = np.arange(w, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return pd.Series(index=idx, dtype=float), np.nan
    m, b = np.polyfit(x[ok], y[ok], 1)
    xx_full = np.arange(len(s), dtype=float)
    # anchor b to last window x-space
    # line in "global bar space" for display:
    # compute line through last window end:
    # y_end = m*(w-1) + b; map that to full index end:
    y_end = m*(w-1) + b
    b_full = y_end - m*(len(s)-1)
    yhat_full = m*xx_full + b_full
    return pd.Series(yhat_full, index=idx), float(m)

def regression_r2(s: pd.Series, window: int) -> float:
    s = _coerce_1d_series(s).astype(float)
    if len(s) < 3:
        return np.nan
    w = min(window, len(s))
    y = s.iloc[-w:].to_numpy(dtype=float)
    x = np.arange(w, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    m, b = np.polyfit(x[ok], y[ok], 1)
    yhat = m*x + b
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)

def regression_with_band(close: pd.Series, window: int):
    """
    Fit on last `window` points; return yhat across full series plus ±2σ residual bands.
    """
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    yhat, m = slope_line(s, window)
    if yhat.dropna().empty:
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float), np.nan, np.nan

    w = min(window, len(s))
    y = s.iloc[-w:].to_numpy(dtype=float)
    x = np.arange(w, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return yhat, pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float), float(m), np.nan

    m2, b2 = np.polyfit(x[ok], y[ok], 1)
    resid = y - (m2*x + b2)
    sigma = float(np.nanstd(resid))
    if not np.isfinite(sigma) or sigma == 0:
        sigma = np.nan

    upper = yhat + 2.0 * sigma
    lower = yhat - 2.0 * sigma
    r2 = regression_r2(s, window)
    return yhat, upper, lower, float(m), float(r2)

def compute_roc(close: pd.Series, n: int = 12) -> pd.Series:
    c = close.astype(float)
    return 100.0 * (c / c.shift(n) - 1.0)

def compute_normalized_trend(close: pd.Series, window: int = 55) -> pd.Series:
    """
    NTD: roughly z-scored deviation from rolling mean, clipped to [-1, 1].
    """
    c = close.astype(float)
    mu = c.rolling(window, min_periods=max(5, window//4)).mean()
    sd = c.rolling(window, min_periods=max(5, window//4)).std(ddof=0)
    z = (c - mu) / sd.replace(0, np.nan)
    return z.clip(-1.0, 1.0)

def compute_normalized_price(close: pd.Series, window: int = 55) -> pd.Series:
    """
    NPX: z-scored price, clipped to [-1, 1].
    """
    c = close.astype(float)
    mu = c.rolling(window, min_periods=max(5, window//4)).mean()
    sd = c.rolling(window, min_periods=max(5, window//4)).std(ddof=0)
    z = (c - mu) / sd.replace(0, np.nan)
    return z.clip(-1.0, 1.0)

def _cross_series(a: pd.Series, b: pd.Series):
    a = _coerce_1d_series(a).astype(float)
    b = _coerce_1d_series(b).astype(float)
    up = (a > b) & (a.shift(1) <= b.shift(1))
    dn = (a < b) & (a.shift(1) >= b.shift(1))
    return up, dn

def _n_consecutive_increasing(s: pd.Series, n: int = 2) -> bool:
    if s is None or len(s) < n + 1:
        return False
    tail = s.dropna().iloc[-(n+1):].to_numpy(dtype=float)
    if len(tail) < n + 1:
        return False
    return bool(np.all(np.diff(tail) > 0))

def _n_consecutive_decreasing(s: pd.Series, n: int = 2) -> bool:
    if s is None or len(s) < n + 1:
        return False
    tail = s.dropna().iloc[-(n+1):].to_numpy(dtype=float)
    if len(tail) < n + 1:
        return False
    return bool(np.all(np.diff(tail) < 0))

def slope_reversal_probability(close: pd.Series,
                               slope_sig: float,
                               hist_window: int = 180,
                               slope_window: int = 60,
                               horizon: int = 12) -> float:
    """
    Empirical probability that slope sign flips within `horizon` bars,
    based on history of rolling slopes.
    """
    c = _coerce_1d_series(close).astype(float).dropna()
    if len(c) < max(hist_window, slope_window) + horizon + 5:
        return np.nan

    # compute slopes over the last hist_window bars
    c = c.iloc[-(hist_window + slope_window + horizon):]
    slopes = []
    for i in range(slope_window, len(c) - horizon):
        seg = c.iloc[i - slope_window:i]
        y = seg.to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 3:
            slopes.append(np.nan)
            continue
        m, _ = np.polyfit(x[ok], y[ok], 1)
        slopes.append(float(m))
    slopes = np.array(slopes, dtype=float)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) < 20 or not np.isfinite(slope_sig):
        return np.nan

    current_sign = np.sign(slope_sig)
    if current_sign == 0:
        return np.nan

    # Count how often sign flips within horizon using forward slopes
    flips = 0
    total = 0
    # approximate by looking at future window slopes (coarse but stable)
    for i in range(len(slopes) - horizon):
        s0 = np.sign(slopes[i])
        if s0 == 0:
            continue
        total += 1
        fut = slopes[i + 1:i + 1 + horizon]
        if np.any(np.sign(fut) == -s0):
            flips += 1
    if total <= 0:
        return np.nan
    return float(flips / total)

def fibonacci_levels(close: pd.Series) -> dict[str, float]:
    c = _coerce_1d_series(close).astype(float).dropna()
    if c.empty:
        return {}
    hi = float(c.max())
    lo = float(c.min())
    rng = hi - lo
    if rng <= 0:
        return {}
    return {
        "0.0": hi,
        "0.236": hi - 0.236 * rng,
        "0.382": hi - 0.382 * rng,
        "0.5": hi - 0.5 * rng,
        "0.618": hi - 0.618 * rng,
        "0.786": hi - 0.786 * rng,
        "1.0": lo,
    }

def _has_volume_to_plot(vol: pd.Series) -> bool:
    v = _coerce_1d_series(vol).dropna()
    if v.empty:
        return False
    return float(v.max()) > 0.0

# ---------------------------
# Band bounce signal (±2σ) — ONLY triggers on re-entry from outside
# ---------------------------
def find_band_bounce_signal(close: pd.Series, upper: pd.Series, lower: pd.Series, slope_sig: float):
    c = _coerce_1d_series(close).astype(float)
    u = _coerce_1d_series(upper).astype(float).reindex(c.index)
    l = _coerce_1d_series(lower).astype(float).reindex(c.index)
    if len(c) < 3:
        return None
    if not np.isfinite(slope_sig):
        return None

    # Use last bar and previous bar to detect "outside -> inside" bounce
    c0, c1 = float(c.iloc[-1]), float(c.iloc[-2])
    u0, u1 = float(u.iloc[-1]), float(u.iloc[-2])
    l0, l1 = float(l.iloc[-1]), float(l.iloc[-2])
    if not np.all(np.isfinite([c0, c1, u0, u1, l0, l1])):
        return None

    # BUY: slope > 0 AND price was below lower band then re-enters above it
    if slope_sig > 0 and (c1 < l1) and (c0 >= l0):
        return {"time": c.index[-1], "price": c0, "side": "BUY"}

    # SELL: slope < 0 AND price was above upper band then re-enters below it
    if slope_sig < 0 and (c1 > u1) and (c0 <= u0):
        return {"time": c.index[-1], "price": c0, "side": "SELL"}

    return None

# ---------------------------
# Trade instruction banner text (slope-aware)
# ---------------------------
def format_trade_instruction(trend_slope: float, buy_val: float, sell_val: float, close_val: float, symbol: str) -> str:
    if not np.isfinite(trend_slope) or not np.isfinite(close_val) or not np.isfinite(buy_val) or not np.isfinite(sell_val):
        return "No instruction"
    if trend_slope > 0:
        return f"Uptrend → BUY near Support {fmt_price_val(buy_val)}"
    if trend_slope < 0:
        return f"Downtrend → SELL near Resistance {fmt_price_val(sell_val)}"
    return "Flat trend"

# ---------------------------
# Gapless TIME axis helpers (intraday)
# ---------------------------
def _gapless_time_x(idx: pd.DatetimeIndex):
    n = 0 if idx is None else len(idx)
    return np.arange(n, dtype=float)

def _set_gapless_time_ticks(ax, idx: pd.DatetimeIndex, max_ticks: int = 9, fmt: str = "%m-%d %H:%M"):
    if idx is None or len(idx) == 0:
        return
    k = min(max_ticks, len(idx))
    pos = np.linspace(0, len(idx) - 1, num=k, dtype=int)
    labels = [idx[i].strftime(fmt) for i in pos]
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)

def _nearest_xpos(idx: pd.DatetimeIndex, t) -> int | None:
    if idx is None or len(idx) == 0 or t is None:
        return None
    try:
        loc = idx.get_indexer([t], method="nearest")[0]
        loc = int(loc)
        return loc if loc >= 0 else None
    except Exception:
        return None

def draw_news_markers_gapless(ax, base_idx: pd.DatetimeIndex, times, label="News"):
    for t in times:
        pos = _nearest_xpos(base_idx, t)
        if pos is None:
            continue
        ax.axvline(pos, color="tab:red", alpha=0.18, linewidth=1)
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

def compute_session_lines(base_idx: pd.DatetimeIndex) -> dict:
    """
    London/NY open/close in PST for each day in base_idx.
    (Approximations; consistent visual markers.)
    """
    if base_idx is None or len(base_idx) == 0:
        return {}

    # PST-based times (approx):
    # London open ~ 00:00 PST, close ~ 09:00 PST
    # NY open ~ 06:30 PST, close ~ 13:00 PST
    days = pd.to_datetime(pd.Series(base_idx.date)).drop_duplicates().to_list()
    ldn_open, ldn_close, ny_open, ny_close = [], [], [], []

    for d in days:
        # d is date
        dt0 = datetime(d.year, d.month, d.day, tzinfo=PACIFIC)
        ldn_open.append(dt0.replace(hour=0, minute=0))
        ldn_close.append(dt0.replace(hour=9, minute=0))
        ny_open.append(dt0.replace(hour=6, minute=30))
        ny_close.append(dt0.replace(hour=13, minute=0))

    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}

def draw_session_lines_gapless(ax, base_idx: pd.DatetimeIndex, lines: dict):
    ax.plot([], [], linestyle="-",  color="tab:blue",   label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",   label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")

    def _vline_list(ts_list, ls, col):
        for t in ts_list:
            pos = _nearest_xpos(base_idx, t)
            if pos is None:
                continue
            ax.axvline(pos, linestyle=ls, linewidth=1.0, color=col, alpha=0.35)

    _vline_list(lines.get("ldn_open",  []), "-",  "tab:blue")
    _vline_list(lines.get("ldn_close", []), "--", "tab:blue")
    _vline_list(lines.get("ny_open",   []), "-",  "tab:orange")
    _vline_list(lines.get("ny_close",  []), "--", "tab:orange")

    ax.text(0.99, 0.98, "Session times in PST", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))

# ---------------------------
# Forex news (optional)
# ---------------------------
@st.cache_data(ttl=600)
def fetch_yf_news(symbol: str, window_days: int = 3) -> pd.DataFrame:
    """
    yfinance news is heuristic. We use it only for vertical markers.
    """
    try:
        t = yf.Ticker(symbol)
        items = t.news or []
    except Exception:
        items = []

    rows = []
    now = datetime.now(tz=PACIFIC)
    cutoff = now - timedelta(days=window_days)

    for it in items:
        try:
            # often 'providerPublishTime' in epoch seconds
            ts = it.get("providerPublishTime", None)
            if ts is None:
                continue
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(PACIFIC)
            if dt < cutoff:
                continue
            rows.append({"time": dt, "title": it.get("title", "")})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time")
    return df
# =========================
# Part 3/6 — bullbear.py
# =========================
# NTD overlays (gapless) + Support/Resistance reversal star logic

def detect_hma_reversal_masks(price: pd.Series, hma: pd.Series, lookback: int = 3):
    """
    Detect small reversals around HMA: price crosses above HMA after being below (buy_rev),
    and crosses below after being above (sell_rev). Confirm via `lookback`.
    """
    p = _coerce_1d_series(price).astype(float)
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)

    above = p > h
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))

    # light confirmation: in last lookback bars, slope direction consistent
    buy_rev = cross_up & p.diff().rolling(lookback, min_periods=1).mean() > 0
    sell_rev = cross_dn & p.diff().rolling(lookback, min_periods=1).mean() < 0
    return buy_rev.fillna(False), sell_rev.fillna(False)

def shade_ntd_regions_gapless(ax, x, ntd: pd.Series):
    n = _coerce_1d_series(ntd)
    if n.empty:
        return
    vals = n.to_numpy(dtype=float)
    pos = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
    neg = np.where(np.isfinite(vals) & (vals < 0), vals, 0.0)
    ax.fill_between(x, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(x, 0, neg, alpha=0.12, color="tab:red")

def overlay_inrange_on_ntd_gapless(ax, x, price: pd.Series, sup: pd.Series, res: pd.Series):
    p = _coerce_1d_series(price)
    s_sup = _coerce_1d_series(sup).reindex(p.index)
    s_res = _coerce_1d_series(res).reindex(p.index)

    ok = p.notna() & s_sup.notna() & s_res.notna()
    if ok.sum() < 2:
        return np.nan

    below = p < s_sup
    above = p > s_res
    state = pd.Series(index=p.index, dtype=float)
    state[ok & below] = -1
    state[ok & (~below) & (~above)] = 0
    state[ok & above] = 1

    in_mask = (state == 0).fillna(False).to_numpy(dtype=bool)
    if in_mask.any():
        starts = np.where((~in_mask[:-1]) & (in_mask[1:]))[0] + 1
        if in_mask[0]:
            starts = np.r_[0, starts]
        ends = np.where((in_mask[:-1]) & (~in_mask[1:]))[0]
        if in_mask[-1]:
            ends = np.r_[ends, len(in_mask) - 1]
        for a, b in zip(starts, ends):
            ax.axvspan(x[a], x[b], color="gold", alpha=0.15, zorder=1)

    ax.plot([], [], linewidth=8, color="gold", alpha=0.20, label="In Range (S↔R)")

    enter_from_below = ((state.shift(1) == -1) & (state == 0)).fillna(False).to_numpy(bool)
    enter_from_above = ((state.shift(1) == 1) & (state == 0)).fillna(False).to_numpy(bool)

    if enter_from_below.any():
        xi = np.where(enter_from_below)[0]
        ax.scatter(x[xi], [0.92]*len(xi), marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        xi = np.where(enter_from_above)[0]
        ax.scatter(x[xi], [0.92]*len(xi), marker="v", s=60, color="tab:orange", zorder=7, label="Enter from R")

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

def overlay_ntd_triangles_by_trend_gapless(ax, x, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd)
    if s.dropna().empty or not np.isfinite(trend_slope):
        return
    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    cross_up0 = ((s >= 0.0) & (s.shift(1) < 0.0)).fillna(False).to_numpy(bool)
    cross_dn0 = ((s <= 0.0) & (s.shift(1) > 0.0)).fillna(False).to_numpy(bool)
    cross_out_hi = ((s >= upper) & (s.shift(1) < upper)).fillna(False).to_numpy(bool)
    cross_out_lo = ((s <= lower) & (s.shift(1) > lower)).fillna(False).to_numpy(bool)

    if uptrend:
        xi = np.where(cross_up0)[0]
        if len(xi):
            ax.scatter(x[xi], [0.0]*len(xi), marker="^", s=95, color="tab:green", zorder=10, label="NTD 0↑")
        xi = np.where(cross_out_lo)[0]
        if len(xi):
            ax.scatter(x[xi], s.to_numpy(float)[xi], marker="^", s=85, color="tab:green", zorder=10, label="NTD < -0.75")

    if downtrend:
        xi = np.where(cross_dn0)[0]
        if len(xi):
            ax.scatter(x[xi], [0.0]*len(xi), marker="v", s=95, color="tab:red", zorder=10, label="NTD 0↓")
        xi = np.where(cross_out_hi)[0]
        if len(xi):
            ax.scatter(x[xi], s.to_numpy(float)[xi], marker="v", s=85, color="tab:red", zorder=10, label="NTD > +0.75")

def overlay_npx_on_ntd_gapless(ax, x, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    if npx.dropna().empty:
        return
    ax.plot(x, npx.to_numpy(dtype=float), "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")

    if mark_crosses and not ntd.dropna().empty:
        up_mask, dn_mask = _cross_series(npx, ntd)
        up = up_mask.reindex(npx.index, fill_value=False).to_numpy(bool)
        dn = dn_mask.reindex(npx.index, fill_value=False).to_numpy(bool)
        if up.any():
            xi = np.where(up)[0]
            ax.scatter(x[xi], ntd.to_numpy(float)[xi], marker="o", s=40, color="tab:green", zorder=9, label="Price↑NTD")
        if dn.any():
            xi = np.where(dn)[0]
            ax.scatter(x[xi], ntd.to_numpy(float)[xi], marker="x", s=60, color="tab:red", zorder=9, label="Price↓NTD")

def overlay_hma_reversal_on_ntd_gapless(ax, x, price: pd.Series, hma: pd.Series, lookback: int = 3, y_up: float = 0.95, y_dn: float = -0.95, period: int = 55):
    buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
    buy = buy_rev.reindex(price.index, fill_value=False).to_numpy(bool)
    sell = sell_rev.reindex(price.index, fill_value=False).to_numpy(bool)
    if buy.any():
        xi = np.where(buy)[0]
        ax.scatter(x[xi], [y_up]*len(xi), marker="s", s=70, color="tab:green", zorder=8, label=f"HMA({period}) REV")
    if sell.any():
        xi = np.where(sell)[0]
        ax.scatter(x[xi], [y_dn]*len(xi), marker="D", s=70, color="tab:red", zorder=8, label=f"HMA({period}) REV")

def overlay_ntd_sr_reversal_stars_gapless(ax,
                                         x,
                                         price: pd.Series,
                                         sup: pd.Series,
                                         res: pd.Series,
                                         trend_slope: float,
                                         ntd: pd.Series,
                                         prox: float = 0.0025,
                                         bars_confirm: int = 2):
    """
    Trend-aware star:
      • Uptrend: near support + last bars rising + moving toward resistance -> BUY ★
      • Downtrend: near resistance + last bars falling + moving toward support -> SELL ★
    """
    p = _coerce_1d_series(price).astype(float)
    if p.dropna().empty or len(p) < 3:
        return

    s_sup = _coerce_1d_series(sup).reindex(p.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(p.index).ffill().bfill()
    s_ntd = _coerce_1d_series(ntd).reindex(p.index)

    c0 = float(p.iloc[-1])
    c1 = float(p.iloc[-2])
    S0 = float(s_sup.iloc[-1])
    R0 = float(s_res.iloc[-1])
    ntd0 = float(s_ntd.iloc[-1]) if pd.notna(s_ntd.iloc[-1]) else np.nan
    if not np.all(np.isfinite([c0, c1, S0, R0, ntd0])) or not np.isfinite(trend_slope):
        return

    near_support = c0 <= S0 * (1.0 + prox)
    near_resist  = c0 >= R0 * (1.0 - prox)

    toward_res = (R0 - c0) < (R0 - c1)
    toward_sup = (c0 - S0) < (c1 - S0)

    buy_cond  = (trend_slope > 0) and near_support and _n_consecutive_increasing(p, bars_confirm) and toward_res
    sell_cond = (trend_slope < 0) and near_resist  and _n_consecutive_decreasing(p, bars_confirm) and toward_sup

    if buy_cond:
        ax.scatter([x[-1]], [ntd0], marker="*", s=170, color="tab:green", zorder=12, label="BUY ★ (Support reversal)")
    if sell_cond:
        ax.scatter([x[-1]], [ntd0], marker="*", s=170, color="tab:red", zorder=12, label="SELL ★ (Resistance reversal)")
# =========================
# Part 4/6 — bullbear.py
# =========================
# Rendering: Daily view + Intraday view (gapless time-axis)

def render_daily_view(symbol: str,
                      daily: pd.DataFrame,
                      show_hma: bool,
                      show_bbands: bool,
                      show_ichi: bool,
                      hma_period: int,
                      bb_win: int,
                      bb_mult: float,
                      bb_use_ema: bool,
                      ichi_base: int,
                      atr_period: int,
                      atr_mult: float,
                      show_psar: bool,
                      psar_step: float,
                      psar_max: float,
                      slope_lb_daily: int):
    if daily is None or daily.empty or "Close" not in daily:
        st.warning("No daily data available.")
        return

    daily = daily.sort_index()
    c = daily["Close"].astype(float).ffill()

    # Indicators
    yhat, upper, lower, m, r2 = regression_with_band(c, slope_lb_daily)

    hma = compute_hma(c, period=hma_period) if show_hma else pd.Series(index=c.index, dtype=float)
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(c, window=bb_win, mult=bb_mult, use_ema=bb_use_ema) if show_bbands else (None, None, None, None, None)

    kijun = pd.Series(index=c.index, dtype=float)
    if show_ichi and {"High", "Low", "Close"}.issubset(daily.columns):
        _, kijun, _, _, _ = ichimoku_lines(daily["High"], daily["Low"], daily["Close"], base=ichi_base, shift_cloud=False)
        kijun = kijun.reindex(c.index).ffill().bfill()

    st_df = compute_supertrend(daily, atr_period=atr_period, atr_mult=atr_mult)
    st_line = st_df["ST"].reindex(c.index) if not st_df.empty else pd.Series(index=c.index, dtype=float)

    psar_df = compute_psar_from_ohlc(daily, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    if not psar_df.empty:
        psar_df = psar_df.reindex(c.index)

    # Plot (daily keeps real time axis; time gaps not the issue here)
    fig, ax = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.85, right=0.93)

    ax.plot(c.index, c.to_numpy(), label="Close")

    # Global trendline (green/red)
    y = c.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    if ok.sum() >= 2:
        m0, b0 = np.polyfit(x[ok], y[ok], 1)
        yhat0 = m0*x + b0
        ax.plot(c.index, yhat0, "--", linewidth=2.4, color=("tab:green" if m0 >= 0 else "tab:red"),
                label=f"Trend (global) ({fmt_slope(m0)}/bar)")

    if show_hma and not hma.dropna().empty:
        ax.plot(c.index, hma.to_numpy(dtype=float), "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun.dropna().empty:
        ax.plot(c.index, kijun.to_numpy(dtype=float), "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and bb_up is not None and bb_lo is not None:
        ax.fill_between(c.index, bb_lo.to_numpy(dtype=float), bb_up.to_numpy(dtype=float), alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(c.index, bb_mid.to_numpy(dtype=float), "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(c.index, bb_up.to_numpy(dtype=float), ":", linewidth=1.0)
        ax.plot(c.index, bb_lo.to_numpy(dtype=float), ":", linewidth=1.0)

    if not st_line.dropna().empty:
        ax.plot(c.index, st_line.to_numpy(dtype=float), "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        # show PSAR points
        ps = psar_df["PSAR"].astype(float)
        ax.scatter(c.index, ps.to_numpy(dtype=float), s=14, alpha=0.7, label="PSAR")

    # Regression slope + ±2σ
    if not yhat.dropna().empty:
        ax.plot(c.index, yhat.to_numpy(dtype=float), "-", linewidth=2, label=f"Slope {slope_lb_daily} bars ({fmt_slope(m)}/bar)")
    if not upper.dropna().empty and not lower.dropna().empty:
        ax.plot(c.index, upper.to_numpy(dtype=float), "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(c.index, lower.to_numpy(dtype=float), "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

    # Current price box (daily)
    px = float(c.iloc[-1]) if len(c) else np.nan
    nbb_txt = ""
    try:
        if show_bbands and bb_nbb is not None and not bb_nbb.dropna().empty:
            last_nbb = float(bb_nbb.dropna().iloc[-1])
            last_pct = float(bb_pctb.dropna().iloc[-1])
            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
    except Exception:
        pass

    ax.text(0.99, 0.02,
            f"Current price: {fmt_price_val(px)}{nbb_txt}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax.text(0.01, 0.02,
            f"Slope: {fmt_slope(m)}/bar  |  R²({slope_lb_daily}): {fmt_r2(r2)}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax.set_title(f"{symbol} — Daily")
    ax.legend(loc="lower left", framealpha=0.5)
    ax.set_xlabel("Date (PST)")
    st.pyplot(fig)

def render_hourly_views(symbol: str,
                        intraday: pd.DataFrame,
                        hour_range_label: str,
                        is_forex: bool,
                        # options
                        sr_lb_hourly: int,
                        atr_period: int,
                        atr_mult: float,
                        show_psar: bool,
                        psar_step: float,
                        psar_max: float,
                        show_bbands: bool,
                        bb_win: int,
                        bb_mult: float,
                        bb_use_ema: bool,
                        show_hma: bool,
                        hma_period: int,
                        show_ichi: bool,
                        ichi_base: int,
                        slope_lb_hourly: int,
                        show_fibs: bool,
                        # panels
                        show_fx_news: bool,
                        news_window_days: int,
                        show_sessions_pst: bool,
                        show_mom_hourly: bool,
                        mom_lb_hourly: int,
                        show_ntd_panel: bool,
                        ntd_window: int,
                        shade_ntd: bool,
                        show_ntd_channel: bool,
                        show_npx_ntd: bool,
                        mark_npx_cross: bool,
                        show_hma_rev_ntd: bool,
                        hma_rev_lb: int,
                        # reversal star
                        sr_prox_pct: float,
                        rev_bars_confirm: int,
                        # slope reversal probability
                        rev_hist_lb: int,
                        rev_horizon: int):
    """
    FIXED: Removes visible time gaps by plotting on a gapless bar index (0..N-1),
           while keeping PST timestamps as tick labels.
    """
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    intraday = intraday.sort_index()
    intraday = intraday[~intraday.index.duplicated(keep="last")]

    hc = intraday["Close"].astype(float).ffill()
    he = hc.ewm(span=20, adjust=False).mean()

    base_idx = hc.index
    x = _gapless_time_x(base_idx)

    def _vals(slike):
        return _coerce_1d_series(slike).reindex(base_idx).to_numpy(dtype=float)

    # Support / resistance
    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    # Supertrend
    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    st_line_intr = st_intraday["ST"].reindex(base_idx) if not st_intraday.empty and "ST" in st_intraday.columns else pd.Series(dtype=float)

    # Ichimoku Kijun (unshifted)
    kijun_h = pd.Series(index=base_idx, dtype=float)
    if show_ichi and {"High", "Low", "Close"}.issubset(intraday.columns):
        _, kijun_h, _, _, _ = ichimoku_lines(
            intraday["High"], intraday["Low"], intraday["Close"],
            base=ichi_base,
            shift_cloud=False
        )
        kijun_h = kijun_h.reindex(base_idx).ffill().bfill()

    # Bollinger
    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    ) if show_bbands else (pd.Series(index=base_idx, dtype=float),
                           pd.Series(index=base_idx, dtype=float),
                           pd.Series(index=base_idx, dtype=float),
                           pd.Series(index=base_idx, dtype=float),
                           pd.Series(index=base_idx, dtype=float))

    # HMA
    hma_h = compute_hma(hc, period=hma_period) if show_hma else pd.Series(index=base_idx, dtype=float)

    # PSAR
    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    if not psar_h_df.empty:
        psar_h_df = psar_h_df.reindex(base_idx)

    # Regression band on price
    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    slope_sig_h = m_h

    # Slope reversal probability
    rev_prob_h = slope_reversal_probability(
        hc,
        slope_sig_h,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_hourly,
        horizon=rev_horizon,
    )

    # News (forex)
    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(symbol, window_days=news_window_days)

    # ---------------- Price chart (gapless x) ----------------
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.85, right=0.93)

    ax2.plot(x, _vals(hc), label="Intraday")
    ax2.plot(x, _vals(he), "--", label="20 EMA")

    # Global trendline (green/red) — compute on bar index
    s = _coerce_1d_series(hc).astype(float)
    ok = np.isfinite(s.to_numpy())
    if ok.sum() >= 2:
        xx = np.arange(len(s), dtype=float)
        m0, b0 = np.polyfit(xx[ok], s.to_numpy(dtype=float)[ok], 1)
        yhat0 = m0 * xx + b0
        color = "tab:green" if m0 >= 0 else "tab:red"
        ax2.plot(x, yhat0, "--", linewidth=2.4, color=color, label=f"Trend (global) ({fmt_slope(m0)}/bar)")

    # Indicators
    if show_hma and not hma_h.dropna().empty:
        ax2.plot(x, _vals(hma_h), "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(x, _vals(kijun_h), "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and (not bb_up_h.dropna().empty) and (not bb_lo_h.dropna().empty):
        ax2.fill_between(x, _vals(bb_lo_h), _vals(bb_up_h), alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(x, _vals(bb_mid_h), "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(x, _vals(bb_up_h), ":", linewidth=1.0)
        ax2.plot(x, _vals(bb_lo_h), ":", linewidth=1.0)

    if show_psar and (not psar_h_df.empty) and ("PSAR" in psar_h_df.columns):
        ps = psar_h_df["PSAR"].astype(float)
        ax2.scatter(x, _vals(ps), s=15, alpha=0.7, label="PSAR")

    # S/R lines
    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=x[0], xmax=x[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=x[0], xmax=x[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    # Supertrend
    if not st_line_intr.dropna().empty:
        ax2.plot(x, _vals(st_line_intr), "-", label=f"Supertrend ({atr_period},{atr_mult})")

    # Regression slope + bands (thicker/darker)
    if not yhat_h.dropna().empty:
        ax2.plot(x, _vals(yhat_h), "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if (not upper_h.dropna().empty) and (not lower_h.dropna().empty):
        ax2.plot(x, _vals(upper_h), "--", linewidth=2.6, color="black", alpha=0.88, label="Slope +2σ")
        ax2.plot(x, _vals(lower_h), "--", linewidth=2.6, color="black", alpha=0.88, label="Slope -2σ")

        # Bounce signal on ±2σ
        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, slope_sig_h)
        if bounce_sig_h is not None:
            pos = _nearest_xpos(base_idx, bounce_sig_h["time"])
            if pos is not None:
                px = float(bounce_sig_h["price"])
                side = bounce_sig_h["side"]
                if side == "BUY":
                    ax2.scatter([pos], [px], marker="P", s=90, color="tab:green", zorder=7)
                    ax2.text(pos, px, "  BUY", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
                else:
                    ax2.scatter([pos], [px], marker="X", s=90, color="tab:red", zorder=7)
                    ax2.text(pos, px, "  SELL", va="top", fontsize=9, color="tab:red", fontweight="bold")

    # News markers (gapless)
    if is_forex and show_fx_news and (not fx_news.empty):
        draw_news_markers_gapless(ax2, base_idx, fx_news["time"].tolist(), label="News")

    # Instruction text
    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=symbol
    )

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{symbol} Intraday ({hour_range_label}) — {instr_txt}  "
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_h}]"
    )

    # Current price box (intraday)
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

    # Slope/R² badges
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

    # Session lines (gapless)
    if is_forex and show_sessions_pst and len(base_idx) > 0:
        sess = compute_session_lines(base_idx)
        draw_session_lines_gapless(ax2, base_idx, sess)

    # Fibonacci (intraday)
    if show_fibs and len(hc) > 0:
        fibs_h = fibonacci_levels(hc)
        for lbl, yv in fibs_h.items():
            ax2.hlines(yv, xmin=x[0], xmax=x[-1], linestyles="dotted", linewidth=1)
        for lbl, yv in fibs_h.items():
            ax2.text(x[-1], yv, f" {lbl}", va="center", fontsize=8)

    ax2.set_xlim(x[0], x[-1])
    _set_gapless_time_ticks(ax2, base_idx, max_ticks=9)
    ax2.set_xlabel("Time (PST)")
    ax2.legend(loc="lower left", framealpha=0.5)
    st.pyplot(fig2)

    xlim_price = ax2.get_xlim()

    # ---------------- Forex-only volume panel (gapless x) ----------------
    vol = _coerce_1d_series(intraday.get("Volume", pd.Series(index=base_idx))).reindex(base_idx).astype(float)
    if is_forex and _has_volume_to_plot(vol):
        v_mid = rolling_midline(vol, window=max(3, int(slope_lb_hourly)))
        v_trend, v_m = slope_line(vol, slope_lb_hourly)
        v_r2 = regression_r2(vol, slope_lb_hourly)

        fig2v, ax2v = plt.subplots(figsize=(14, 2.8))
        ax2v.set_title(f"Volume (Intraday) — Mid-line & Trend  |  Slope={fmt_slope(v_m)}/bar")
        ax2v.fill_between(x, 0, _vals(vol), alpha=0.18, label="Volume")
        ax2v.plot(x, _vals(vol), linewidth=1.0)

        ax2v.plot(x, _vals(v_mid), ":", linewidth=1.6, label=f"Mid-line ({slope_lb_hourly}-roll)")
        if not v_trend.dropna().empty:
            ax2v.plot(x, _vals(v_trend), "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(v_m)}/bar)")

        ax2v.text(0.01, 0.02, f"Slope: {fmt_slope(v_m)}/bar",
                  transform=ax2v.transAxes, ha="left", va="bottom",
                  fontsize=9, color="black",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax2v.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(v_r2)}",
                  transform=ax2v.transAxes, ha="center", va="bottom",
                  fontsize=9, color="black",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

        ax2v.set_xlim(xlim_price)
        _set_gapless_time_ticks(ax2v, base_idx, max_ticks=9)
        ax2v.set_xlabel("Time (PST)")
        ax2v.legend(loc="lower left", framealpha=0.5)
        st.pyplot(fig2v)

    # ---------------- Optional momentum panel (gapless x) ----------------
    if show_mom_hourly:
        roc = compute_roc(hc, n=mom_lb_hourly).reindex(base_idx)

        fig2m, ax2m = plt.subplots(figsize=(14, 2.8))
        ax2m.set_title(f"Momentum (ROC% over {mom_lb_hourly} bars)")
        ax2m.plot(x, _vals(roc), label=f"ROC%({mom_lb_hourly})")

        yhat_m, m_m = slope_line(roc, slope_lb_hourly)
        if not yhat_m.dropna().empty:
            ax2m.plot(x, _vals(yhat_m), "--", linewidth=2,
                      label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m)}%/bar)")

        ax2m.axhline(0, linestyle="--", linewidth=1)
        ax2m.set_xlim(xlim_price)
        _set_gapless_time_ticks(ax2m, base_idx, max_ticks=9)
        ax2m.set_xlabel("Time (PST)")
        ax2m.legend(loc="lower left", framealpha=0.5)
        st.pyplot(fig2m)

    # ---------------- NTD + NPX panel (gapless x) ----------------
    if show_ntd_panel:
        ntd_h = compute_normalized_trend(hc, window=ntd_window).reindex(base_idx)
        ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
        npx_h = compute_normalized_price(hc, window=ntd_window).reindex(base_idx) if show_npx_ntd else pd.Series(index=base_idx, dtype=float)

        fig2r, ax2r = plt.subplots(figsize=(14, 2.8))
        ax2r.set_title(f"Indicator Panel — NTD + NPX (win={ntd_window})")

        if shade_ntd and not ntd_h.dropna().empty:
            shade_ntd_regions_gapless(ax2r, x, ntd_h)

        if show_ntd_channel and np.isfinite(res_val) and np.isfinite(sup_val):
            overlay_inrange_on_ntd_gapless(ax2r, x, hc, sup_h, res_h)

        ax2r.plot(x, _vals(ntd_h), "-", linewidth=1.6, label="NTD")

        overlay_ntd_triangles_by_trend_gapless(ax2r, x, ntd_h, trend_slope=m_h, upper=0.75, lower=-0.75)
        overlay_ntd_sr_reversal_stars_gapless(ax2r, x, price=hc, sup=sup_h, res=res_h, trend_slope=m_h, ntd=ntd_h,
                                              prox=sr_prox_pct, bars_confirm=rev_bars_confirm)

        if show_npx_ntd and not npx_h.dropna().empty and not ntd_h.dropna().empty:
            overlay_npx_on_ntd_gapless(ax2r, x, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        if not ntd_trend_h.dropna().empty:
            ax2r.plot(x, _vals(ntd_trend_h), "--", linewidth=2,
                      label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")

        if show_hma_rev_ntd and show_hma and not hma_h.dropna().empty and not hc.dropna().empty:
            overlay_hma_reversal_on_ntd_gapless(ax2r, x, hc, hma_h, lookback=hma_rev_lb, period=hma_period)

        ax2r.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2r.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax2r.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")

        ax2r.set_ylim(-1.1, 1.1)
        ax2r.set_xlim(xlim_price)
        _set_gapless_time_ticks(ax2r, base_idx, max_ticks=9)
        ax2r.legend(loc="lower left", framealpha=0.5)
        ax2r.set_xlabel("Time (PST)")
        st.pyplot(fig2r)
# =========================
# Part 5/6 — bullbear.py
# =========================
# Scanner tab + UI controls (sidebar show/hide, cache, market switch buttons, persistence)

def _market_switch_buttons():
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("Forex", use_container_width=True):
            st.session_state.market = "Forex"
    with c2:
        if st.button("Stocks", use_container_width=True):
            st.session_state.market = "Stocks"
    with c3:
        st.write(f"**Selected market:** {st.session_state.market}")

def _sidebar_controls():
    # Sidebar show/hide is controlled via session_state.show_sidebar
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.markdown("## Options")
            if st.button("Hide Sidebar"):
                st.session_state.show_sidebar = False
                st.rerun()

            if st.button("Clear Cache (st.cache_data)"):
                st.cache_data.clear()
                st.success("Cache cleared.")
                st.rerun()
    else:
        top = st.columns([1, 8])
        with top[0]:
            if st.button("Show Sidebar"):
                st.session_state.show_sidebar = True
                st.rerun()
        with top[1]:
            st.caption("Sidebar hidden — click to show options.")

def render_scanner_tab(market: str,
                       ntd_threshold: float,
                       ntd_window: int,
                       intraday_period: str):
    """
    Simple NTD scanner: list symbols whose latest intraday NTD < threshold.
    """
    st.subheader("Scanner — NTD Threshold")
    st.caption("Scans the selected market universe and lists symbols with latest NTD below the threshold.")

    uni = universe_for(market)
    rows = []
    prog = st.progress(0.0)

    for i, sym in enumerate(uni, start=1):
        try:
            df = fetch_intraday(sym, period=intraday_period)
            if df is None or df.empty or "Close" not in df:
                continue
            c = df["Close"].astype(float).ffill()
            ntd = compute_normalized_trend(c, window=ntd_window)
            last = float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan
            last_px = float(c.iloc[-1]) if len(c) else np.nan
            if np.isfinite(last) and last < ntd_threshold:
                rows.append({"Symbol": sym, "Last NTD": last, "Last Price": last_px})
        except Exception:
            continue
        prog.progress(i / len(uni))

    prog.empty()

    if not rows:
        st.info("No symbols matched.")
        return

    out = pd.DataFrame(rows).sort_values("Last NTD")
    st.dataframe(out, use_container_width=True)

def render_details_page(market: str,
                        symbol: str,
                        range_label: str,
                        daily_period: str,
                        intraday_period: str,
                        # toggles
                        show_hma: bool,
                        show_bbands: bool,
                        show_ichi: bool,
                        show_supertrend: bool,
                        show_psar: bool,
                        show_fibs: bool,
                        show_fx_news: bool,
                        show_sessions_pst: bool,
                        show_mom_hourly: bool,
                        show_ntd_panel: bool,
                        shade_ntd: bool,
                        show_ntd_channel: bool,
                        show_npx_ntd: bool,
                        mark_npx_cross: bool,
                        show_hma_rev_ntd: bool,
                        # params
                        hma_period: int,
                        bb_win: int,
                        bb_mult: float,
                        bb_use_ema: bool,
                        ichi_base: int,
                        atr_period: int,
                        atr_mult: float,
                        psar_step: float,
                        psar_max: float,
                        sr_lb_hourly: int,
                        slope_lb_hourly: int,
                        slope_lb_daily: int,
                        mom_lb_hourly: int,
                        ntd_window: int,
                        hma_rev_lb: int,
                        sr_prox_pct: float,
                        rev_bars_confirm: int,
                        news_window_days: int,
                        rev_hist_lb: int,
                        rev_horizon: int):
    is_fx = (market == "Forex")

    st.markdown(f"## {market} — {symbol}")

    tab1, tab2, tab3 = st.tabs(["Daily", "Intraday", "Scanner"])

    with tab1:
        daily = fetch_daily(symbol, period=daily_period)
        render_daily_view(
            symbol=symbol,
            daily=daily,
            show_hma=show_hma,
            show_bbands=show_bbands,
            show_ichi=show_ichi,
            hma_period=hma_period,
            bb_win=bb_win,
            bb_mult=bb_mult,
            bb_use_ema=bb_use_ema,
            ichi_base=ichi_base,
            atr_period=atr_period,
            atr_mult=atr_mult,
            show_psar=show_psar,
            psar_step=psar_step,
            psar_max=psar_max,
            slope_lb_daily=slope_lb_daily,
        )

    with tab2:
        intraday = fetch_intraday(symbol, period=intraday_period)
        render_hourly_views(
            symbol=symbol,
            intraday=intraday,
            hour_range_label=range_label,
            is_forex=is_fx,
            sr_lb_hourly=sr_lb_hourly,
            atr_period=atr_period,
            atr_mult=atr_mult,
            show_psar=show_psar,
            psar_step=psar_step,
            psar_max=psar_max,
            show_bbands=show_bbands,
            bb_win=bb_win,
            bb_mult=bb_mult,
            bb_use_ema=bb_use_ema,
            show_hma=show_hma,
            hma_period=hma_period,
            show_ichi=show_ichi,
            ichi_base=ichi_base,
            slope_lb_hourly=slope_lb_hourly,
            show_fibs=show_fibs,
            show_fx_news=show_fx_news,
            news_window_days=news_window_days,
            show_sessions_pst=show_sessions_pst,
            show_mom_hourly=show_mom_hourly,
            mom_lb_hourly=mom_lb_hourly,
            show_ntd_panel=show_ntd_panel,
            ntd_window=ntd_window,
            shade_ntd=shade_ntd,
            show_ntd_channel=show_ntd_channel,
            show_npx_ntd=show_npx_ntd,
            mark_npx_cross=mark_npx_cross,
            show_hma_rev_ntd=show_hma_rev_ntd,
            hma_rev_lb=hma_rev_lb,
            sr_prox_pct=sr_prox_pct,
            rev_bars_confirm=rev_bars_confirm,
            rev_hist_lb=rev_hist_lb,
            rev_horizon=rev_horizon,
        )

    with tab3:
        # Scanner threshold differs slightly by default for forex vs stocks
        default_thr = -0.50
        ntd_threshold = st.slider("NTD threshold (latest < value)", -1.0, 1.0, float(default_thr), 0.05)
        render_scanner_tab(
            market=market,
            ntd_threshold=ntd_threshold,
            ntd_window=ntd_window,
            intraday_period=intraday_period
        )

def _range_to_period(range_label: str) -> tuple[str, str]:
    """
    Map UI range to yfinance intraday period and label.
    5m interval supports up to ~60d depending on symbol; keep short for stability.
    """
    if range_label == "24h":
        return "1d", "24h"
    if range_label == "48h":
        return "2d", "48h"
    if range_label == "96h":
        return "5d", "96h"
    return "2d", "48h"
# =========================
# Part 6/6 — bullbear.py
# =========================
# Main app layout

def main():
    st.title("BullBear — Stocks / Forex Dashboard")

    _sidebar_controls()

    # Market switch buttons (Forex default on first open)
    _market_switch_buttons()

    market = st.session_state.market
    uni = universe_for(market)

    # ---- Options (sidebar if visible, else inline minimal defaults) ----
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.markdown("### Symbol & Run")
            default_sym = st.session_state.last_run_symbol if st.session_state.last_run_market == market else uni[0]
            symbol = st.selectbox("Ticker / Pair", uni, index=uni.index(default_sym) if default_sym in uni else 0)

            range_label = st.selectbox("Intraday range", ["24h", "48h", "96h"],
                                       index=["24h", "48h", "96h"].index(st.session_state.last_run_range)
                                       if st.session_state.last_run_range in ["24h", "48h", "96h"] else 1)

            run = st.button("Run", type="primary", use_container_width=True)

            st.markdown("---")
            st.markdown("### Indicators")
            show_hma = st.checkbox("HMA", value=True)
            hma_period = st.number_input("HMA period", min_value=5, max_value=300, value=55, step=1)

            show_bbands = st.checkbox("Bollinger Bands", value=True)
            bb_win = st.number_input("BB window", min_value=5, max_value=200, value=20, step=1)
            bb_mult = st.number_input("BB mult", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            bb_use_ema = st.checkbox("BB midline uses EMA", value=False)

            show_ichi = st.checkbox("Ichimoku Kijun", value=True)
            ichi_base = st.number_input("Kijun (base) period", min_value=5, max_value=200, value=26, step=1)

            show_supertrend = st.checkbox("Supertrend", value=True)
            atr_period = st.number_input("ATR period", min_value=3, max_value=100, value=10, step=1)
            atr_mult = st.number_input("ATR mult", min_value=0.5, max_value=10.0, value=3.0, step=0.1)

            show_psar = st.checkbox("PSAR", value=True)
            psar_step = st.number_input("PSAR step", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")
            psar_max = st.number_input("PSAR max", min_value=0.01, max_value=1.0, value=0.2, step=0.01, format="%.2f")

            show_fibs = st.checkbox("Fibonacci (Intraday)", value=True)

            st.markdown("---")
            st.markdown("### Panels")
            show_mom_hourly = st.checkbox("Momentum ROC% panel", value=True)
            mom_lb_hourly = st.number_input("ROC lookback (bars)", min_value=2, max_value=300, value=12, step=1)

            show_ntd_panel = st.checkbox("NTD + NPX panel", value=True)
            ntd_window = st.number_input("NTD/NPX window", min_value=10, max_value=300, value=55, step=1)
            shade_ntd = st.checkbox("Shade NTD regions", value=True)
            show_ntd_channel = st.checkbox("In-range shading (S↔R) on NTD", value=True)
            show_npx_ntd = st.checkbox("Overlay NPX on NTD", value=True)
            mark_npx_cross = st.checkbox("Mark NPX↔NTD crosses", value=True)
            show_hma_rev_ntd = st.checkbox("HMA reversal markers on NTD", value=True)
            hma_rev_lb = st.number_input("HMA reversal lookback", min_value=1, max_value=20, value=3, step=1)

            st.markdown("---")
            st.markdown("### Support/Resistance + Slope")
            sr_lb_hourly = st.number_input("S/R lookback (bars)", min_value=10, max_value=600, value=120, step=10)
            slope_lb_hourly = st.number_input("Slope window (intraday bars)", min_value=10, max_value=600, value=120, step=10)
            slope_lb_daily = st.number_input("Slope window (daily bars)", min_value=10, max_value=300, value=60, step=5)

            sr_prox_pct = st.number_input("S/R proximity (fraction)", min_value=0.0005, max_value=0.05, value=0.0025, step=0.0005, format="%.4f")
            rev_bars_confirm = st.number_input("Reversal confirm bars", min_value=1, max_value=10, value=2, step=1)

            st.markdown("---")
            st.markdown("### Forex extras")
            show_fx_news = st.checkbox("Show news markers (Forex)", value=True if market == "Forex" else False)
            news_window_days = st.number_input("News window (days)", min_value=1, max_value=14, value=3, step=1)
            show_sessions_pst = st.checkbox("Show session lines (PST)", value=True if market == "Forex" else False)

            st.markdown("---")
            st.markdown("### Slope reversal probability")
            rev_hist_lb = st.number_input("History window (bars)", min_value=60, max_value=2000, value=180, step=20)
            rev_horizon = st.number_input("Horizon (bars)", min_value=3, max_value=200, value=12, step=1)

            # Daily/intraday periods
            daily_period = st.selectbox("Daily history", ["3mo", "6mo", "1y", "2y"], index=1)

    else:
        # Minimal inline defaults if sidebar hidden
        default_sym = st.session_state.last_run_symbol if st.session_state.last_run_market == market else uni[0]
        symbol = st.selectbox("Ticker / Pair", uni, index=uni.index(default_sym) if default_sym in uni else 0)
        range_label = st.selectbox("Intraday range", ["24h", "48h", "96h"],
                                   index=["24h", "48h", "96h"].index(st.session_state.last_run_range)
                                   if st.session_state.last_run_range in ["24h", "48h", "96h"] else 1)
        run = st.button("Run", type="primary")

        show_hma = True
        show_bbands = True
        show_ichi = True
        show_supertrend = True
        show_psar = True
        show_fibs = True

        show_fx_news = True if market == "Forex" else False
        show_sessions_pst = True if market == "Forex" else False
        show_mom_hourly = True
        show_ntd_panel = True
        shade_ntd = True
        show_ntd_channel = True
        show_npx_ntd = True
        mark_npx_cross = True
        show_hma_rev_ntd = True

        hma_period = 55
        bb_win = 20
        bb_mult = 2.0
        bb_use_ema = False
        ichi_base = 26
        atr_period = 10
        atr_mult = 3.0
        psar_step = 0.02
        psar_max = 0.2
        sr_lb_hourly = 120
        slope_lb_hourly = 120
        slope_lb_daily = 60
        mom_lb_hourly = 12
        ntd_window = 55
        hma_rev_lb = 3
        sr_prox_pct = 0.0025
        rev_bars_confirm = 2
        news_window_days = 3
        rev_hist_lb = 180
        rev_horizon = 12
        daily_period = "6mo"

    # ---- Persist selection: chart last-run symbol until a new Run ----
    if run:
        st.session_state.last_run_market = market
        st.session_state.last_run_symbol = symbol
        st.session_state.last_run_range = range_label

    # Use last-run if current rerun didn’t click Run
    active_market = st.session_state.last_run_market
    active_symbol = st.session_state.last_run_symbol
    active_range = st.session_state.last_run_range

    # If user changed market but didn't press Run, keep charting last-run
    # (Per request: any symbol run should stay charted until another run)
    intraday_period, hour_range_label = _range_to_period(active_range)

    render_details_page(
        market=active_market,
        symbol=active_symbol,
        range_label=hour_range_label,
        daily_period=daily_period,
        intraday_period=intraday_period,
        show_hma=show_hma,
        show_bbands=show_bbands,
        show_ichi=show_ichi,
        show_supertrend=show_supertrend,
        show_psar=show_psar,
        show_fibs=show_fibs,
        show_fx_news=show_fx_news,
        show_sessions_pst=show_sessions_pst,
        show_mom_hourly=show_mom_hourly,
        show_ntd_panel=show_ntd_panel,
        shade_ntd=shade_ntd,
        show_ntd_channel=show_ntd_channel,
        show_npx_ntd=show_npx_ntd,
        mark_npx_cross=mark_npx_cross,
        show_hma_rev_ntd=show_hma_rev_ntd,
        hma_period=hma_period,
        bb_win=bb_win,
        bb_mult=bb_mult,
        bb_use_ema=bb_use_ema,
        ichi_base=ichi_base,
        atr_period=atr_period,
        atr_mult=atr_mult,
        psar_step=psar_step,
        psar_max=psar_max,
        sr_lb_hourly=sr_lb_hourly,
        slope_lb_hourly=slope_lb_hourly,
        slope_lb_daily=slope_lb_daily,
        mom_lb_hourly=mom_lb_hourly,
        ntd_window=ntd_window,
        hma_rev_lb=hma_rev_lb,
        sr_prox_pct=sr_prox_pct,
        rev_bars_confirm=rev_bars_confirm,
        news_window_days=news_window_days,
        rev_hist_lb=rev_hist_lb,
        rev_horizon=rev_horizon
    )

if __name__ == "__main__":
    main()
