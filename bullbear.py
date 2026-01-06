# app.py
# Streamlit multi-tab technical dashboard (Stocks + Forex)
# UPDATED: Fib confirmation trigger now ALSO requires crossing back above/below the 23.6% Fib level
#          (while keeping the same UI look/feel and tab structure).

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import streamlit as st
import yfinance as yf

# Optional (forecast)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# =========================
# Part 1/10
# =========================
st.set_page_config(page_title="Forecast Dashboard", layout="wide")

PACIFIC = pytz.timezone("America/Los_Angeles")

# Confirmation threshold requested in earlier changes
FIB_CONFIRM_R2 = 0.999

# ---------------------------
# Small utilities
# ---------------------------
def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        s = x.iloc[:, 0].copy()
    else:
        try:
            s = pd.Series(x).copy()
        except Exception:
            return pd.Series(dtype=float)
    try:
        s = s.astype(float)
    except Exception:
        pass
    return s

def _safe_last_float(series_like) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return np.nan
    v = s.iloc[-1]
    return float(v) if np.isfinite(v) else np.nan

def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"

def fmt_price_val(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    ax = abs(float(x))
    if ax >= 1000:
        return f"{float(x):,.2f}"
    if ax >= 1:
        return f"{float(x):.4f}"
    return f"{float(x):.6f}"

def fmt_slope(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{float(x):+.6f}"

def fmt_r2(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{float(x):.4f}"

def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def label_on_left(ax, y, text, color="black"):
    try:
        x0 = ax.get_xlim()[0]
        ax.text(x0, y, f" {text}", va="center", ha="left", fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85))
    except Exception:
        pass

def _to_pacific_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return idx
    if idx.tz is None:
        # yfinance can return naive timestamps for some markets; treat as UTC for stability
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(PACIFIC)

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return
    n = len(real_times)
    if n <= n_ticks:
        ticks = np.arange(n)
    else:
        ticks = np.linspace(0, n - 1, n_ticks).round().astype(int)
    labels = [real_times[i].astimezone(PACIFIC).strftime("%m-%d %H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha="right")

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, event_times: list):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return []
    rt = pd.DatetimeIndex(real_times)
    out = []
    for t in event_times:
        try:
            tt = pd.Timestamp(t)
            if tt.tzinfo is None:
                tt = tt.tz_localize(PACIFIC)
            else:
                tt = tt.tz_convert(PACIFIC)
            # find nearest index
            pos = int(np.searchsorted(rt.values, tt.to_datetime64(), side="left"))
            pos = max(0, min(pos, len(rt) - 1))
            out.append(pos)
        except Exception:
            continue
    return out

def _cross_series(a: pd.Series, b: pd.Series):
    a = _coerce_1d_series(a)
    b = _coerce_1d_series(b).reindex(a.index)
    prev_a = a.shift(1)
    prev_b = b.shift(1)
    cross_up = (a >= b) & (prev_a < prev_b)
    cross_dn = (a <= b) & (prev_a > prev_b)
    return cross_up.fillna(False), cross_dn.fillna(False)


# =========================
# Part 2/10
# =========================
# ---------------------------
# Data fetchers
# ---------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist(symbol: str, years: int = 10) -> pd.Series:
    # daily close
    period = f"{int(max(1, years))}y"
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if "Close" in df.columns:
            s = df["Close"].dropna().astype(float)
        else:
            s = df.iloc[:, 0].dropna().astype(float)
        # keep naive daily dates (UI already assumes "PST" label)
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, years: int = 10) -> pd.DataFrame:
    period = f"{int(max(1, years))}y"
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[keep].dropna(how="all").copy()
        # daily index is date-like; keep as-is (naive)
        return out
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_max(symbol: str) -> pd.Series:
    try:
        df = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df["Close"].dropna().astype(float) if "Close" in df.columns else df.iloc[:, 0].dropna().astype(float)
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d", interval: str = "60m") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Ensure column case/availability
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = _to_pacific_index(df.index)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# =========================
# Part 3/10
# =========================
# ---------------------------
# Indicators: HMA, MACD, BBands, NTD/NPX
# ---------------------------
def wma(series: pd.Series, period: int) -> pd.Series:
    s = _coerce_1d_series(series)
    if period <= 1:
        return s
    w = np.arange(1, period + 1, dtype=float)
    denom = w.sum()

    def _wavg(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, w) / denom)

    return s.rolling(period, min_periods=period).apply(_wavg, raw=True)

def compute_hma(series: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(series)
    p = int(max(2, period))
    half = int(max(1, p // 2))
    sqrtp = int(max(1, round(math.sqrt(p))))
    w1 = wma(s, half)
    w2 = wma(s, p)
    raw = 2.0 * w1 - w2
    return wma(raw, sqrtp)

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(series)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bbands(series: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(series)
    w = int(max(2, window))
    mid = s.ewm(span=w, adjust=False).mean() if use_ema else s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=0)
    up = mid + float(mult) * sd
    lo = mid - float(mult) * sd
    # %B and normalized band width (NBB)
    pctb = (s - lo) / (up - lo)
    nbb = (up - lo) / mid
    return mid, up, lo, pctb, nbb

def compute_normalized_price(series: pd.Series, window: int = 55) -> pd.Series:
    s = _coerce_1d_series(series)
    w = int(max(2, window))
    rmin = s.rolling(w, min_periods=1).min()
    rmax = s.rolling(w, min_periods=1).max()
    denom = (rmax - rmin).replace(0.0, np.nan)
    npx = (s - rmin) / denom
    return npx.clip(0.0, 1.0)

def _rolling_linreg_slope(series: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series)
    w = int(max(2, window))
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y):
        y = np.asarray(y, dtype=float)
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        return float(cov / x_var) if x_var != 0 else np.nan

    return s.rolling(w, min_periods=w).apply(_slope, raw=True)

def compute_normalized_trend(series: pd.Series, window: int = 55) -> pd.Series:
    s = _coerce_1d_series(series)
    w = int(max(5, window))
    slope = _rolling_linreg_slope(s, w)
    mu = slope.rolling(w, min_periods=5).mean()
    sd = slope.rolling(w, min_periods=5).std(ddof=0).replace(0.0, np.nan)
    z = (slope - mu) / sd
    ntd = np.tanh(z / 2.0)
    return pd.Series(ntd, index=s.index)

def slope_line(series: pd.Series, lookback: int = 90):
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return pd.Series(dtype=float), np.nan
    lb = int(min(max(2, lookback), len(s)))
    seg = s.iloc[-lb:]
    x = np.arange(lb, dtype=float)
    try:
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        yhat = m * x + b
        out = pd.Series(index=seg.index, data=yhat)
        return out.reindex(series.index), float(m)
    except Exception:
        return pd.Series(dtype=float), np.nan


# =========================
# Part 4/10
# =========================
# ---------------------------
# Regression band + signals
# ---------------------------
def regression_with_band(series: pd.Series, lookback: int = 90):
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), np.nan, np.nan)

    lb = int(min(max(5, lookback), len(s)))
    seg = s.iloc[-lb:]
    x = np.arange(lb, dtype=float)
    y = seg.to_numpy(dtype=float)

    try:
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        resid = y - yhat
        sigma = float(np.nanstd(resid, ddof=0))
        upper = yhat + 2.0 * sigma
        lower = yhat - 2.0 * sigma

        # R²
        ss_res = float(np.nansum((y - yhat) ** 2))
        ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        yhat_s = pd.Series(index=seg.index, data=yhat).reindex(series.index)
        up_s = pd.Series(index=seg.index, data=upper).reindex(series.index)
        lo_s = pd.Series(index=seg.index, data=lower).reindex(series.index)
        return yhat_s, up_s, lo_s, float(m), float(r2)
    except Exception:
        return (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), np.nan, np.nan)

def draw_trend_direction_line(ax, series: pd.Series, label_prefix="Trend (global)"):
    s = _coerce_1d_series(series).dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    try:
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        ax.plot(s.index, yhat, "-", linewidth=1.8, label=f"{label_prefix}: {fmt_slope(m)}/bar")
        return float(m)
    except Exception:
        return np.nan

def find_band_bounce_signal(price: pd.Series, upper: pd.Series, lower: pd.Series, slope: float):
    p = _coerce_1d_series(price)
    up = _coerce_1d_series(upper).reindex(p.index)
    lo = _coerce_1d_series(lower).reindex(p.index)
    if p.dropna().empty or up.dropna().empty or lo.dropna().empty:
        return None

    # Simple bounce logic:
    # Up-slope: look for cross UP above lower band after being below it.
    # Down-slope: look for cross DOWN below upper band after being above it.
    if np.isfinite(slope) and slope > 0:
        cross_up = (p >= lo) & (p.shift(1) < lo.shift(1))
        if cross_up.any():
            t = cross_up[cross_up].index[-1]
            return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    if np.isfinite(slope) and slope < 0:
        cross_dn = (p <= up) & (p.shift(1) > up.shift(1))
        if cross_dn.any():
            t = cross_dn[cross_dn].index[-1]
            return {"time": t, "price": float(p.loc[t]), "side": "SELL"}
    return None

def annotate_crossover(ax, t, y, side: str):
    if t is None or y is None or not np.isfinite(y):
        return
    if str(side).upper().startswith("B"):
        ax.scatter([t], [y], marker="^", s=120, zorder=10, label="Band Bounce BUY")
    else:
        ax.scatter([t], [y], marker="v", s=120, zorder=10, label="Band Bounce SELL")

def find_slope_trigger_after_band_reversal(price: pd.Series,
                                          yhat: pd.Series,
                                          upper: pd.Series,
                                          lower: pd.Series,
                                          horizon: int = 10):
    # A lightweight "trigger" heuristic:
    # after a bounce signal, wait for price to cross the regression line.
    p = _coerce_1d_series(price)
    yh = _coerce_1d_series(yhat).reindex(p.index)
    if p.dropna().empty or yh.dropna().empty:
        return None
    hz = int(max(1, horizon))
    seg = p.iloc[-(hz + 2):]
    seg_y = yh.reindex(seg.index)

    cross_up, cross_dn = _cross_series(seg, seg_y)
    if cross_up.any():
        t = cross_up[cross_up].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY", "label": "Slope trigger"}
    if cross_dn.any():
        t = cross_dn[cross_dn].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL", "label": "Slope trigger"}
    return None

def annotate_slope_trigger(ax, trig: dict):
    if not trig:
        return
    t = trig.get("time")
    y = trig.get("price")
    side = str(trig.get("side", "")).upper()
    if t is None or y is None or not np.isfinite(y):
        return
    mk = "P" if side.startswith("B") else "X"
    ax.scatter([t], [y], marker=mk, s=140, zorder=12, label=f"{trig.get('label','Trigger')} ({side})")

def slope_reversal_probability(price: pd.Series,
                               slope_now: float,
                               hist_window: int = 250,
                               slope_window: int = 90,
                               horizon: int = 10) -> float:
    # Empirical: over history, how often does slope sign flip within 'horizon' bars?
    p = _coerce_1d_series(price).dropna()
    if len(p) < max(hist_window, slope_window) + horizon + 5:
        return np.nan
    slopes = _rolling_linreg_slope(p, int(max(5, slope_window))).dropna()
    if slopes.empty:
        return np.nan

    # Use last hist_window slopes
    slopes = slopes.iloc[-int(hist_window):]
    sign = np.sign(slopes)
    # For each time i, did a sign flip occur in next horizon?
    hz = int(max(1, horizon))
    flips = []
    for i in range(len(sign) - hz):
        s0 = sign.iloc[i]
        if s0 == 0:
            continue
        nxt = sign.iloc[i + 1:i + 1 + hz]
        flips.append(bool((nxt * s0 < 0).any()))
    if not flips:
        return np.nan
    return float(np.mean(flips))

def current_daily_pivots(df_ohlc: pd.DataFrame):
    if df_ohlc is None or df_ohlc.empty or not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        return {}
    d = df_ohlc.dropna().iloc[-1]
    H, L, C = float(d["High"]), float(d["Low"]), float(d["Close"])
    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}


# =========================
# Part 5/10
# =========================
# ---------------------------
# Fibonacci helpers + UPDATED Confirmation trigger logic
# ---------------------------
def fibonacci_levels(series: pd.Series):
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return {}
    lo = float(np.nanmin(s.to_numpy(dtype=float)))
    hi = float(np.nanmax(s.to_numpy(dtype=float)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi == lo:
        return {}
    rng = hi - lo
    levels = {
        "0%": lo,
        "23.6%": lo + rng * 0.236,
        "38.2%": lo + rng * 0.382,
        "50%": lo + rng * 0.500,
        "61.8%": lo + rng * 0.618,
        "78.6%": lo + rng * 0.786,
        "100%": hi,
    }
    return levels

def fib_position_percent(price: float, fibs: dict):
    if not fibs or "0%" not in fibs or "100%" not in fibs:
        return np.nan
    lo = float(fibs["0%"])
    hi = float(fibs["100%"])
    if not (np.isfinite(price) and np.isfinite(lo) and np.isfinite(hi)) or hi == lo:
        return np.nan
    return float((price - lo) / (hi - lo) * 100.0)

def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
        return False
    deltas = np.diff(s.iloc[-(n + 1):])
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
        return False
    deltas = np.diff(s.iloc[-(n + 1):])
    return bool(np.all(deltas < 0))

def find_fib_confirmation_trigger(price: pd.Series,
                                  fibs: dict,
                                  r2: float,
                                  horizon: int = 10,
                                  prox: float = 0.0025,
                                  bars_confirm: int = 2,
                                  r2_thr: float = FIB_CONFIRM_R2):
    """
    Confirmation Trigger (R² >= r2_thr):
      BUY: price recently touched near Fib 0% (LOW) AND THEN crossed back ABOVE Fib 23.6%,
           with bars_confirm consecutive rising closes into the cross.
      SELL: price recently touched near Fib 100% (HIGH) AND THEN crossed back BELOW Fib 23.6%,
            with bars_confirm consecutive falling closes into the cross.

    NOTE: The 23.6% cross requirement is exactly as requested (extra confirmation).
    """
    p = _coerce_1d_series(price).dropna()
    if p.empty or not fibs or "0%" not in fibs or "100%" not in fibs or "23.6%" not in fibs:
        return None
    if not (np.isfinite(r2) and float(r2) >= float(r2_thr)):
        return None

    fib0 = float(fibs["0%"])
    fib100 = float(fibs["100%"])
    fib236 = float(fibs["23.6%"])
    if not np.all(np.isfinite([fib0, fib100, fib236])):
        return None

    hz = int(max(1, horizon))
    # Work on last hz+something bars so crosses can be detected after touch
    look = p.iloc[-(hz + 5):] if len(p) > (hz + 5) else p.copy()

    # --- BUY path ---
    touch0 = look <= fib0 * (1.0 + float(prox))
    buy_event = None
    if touch0.any():
        t_touch = touch0[touch0].index[-1]
        after = look.loc[t_touch:]
        # Cross above 23.6 from below
        cross_up_236 = (after >= fib236) & (after.shift(1) < fib236)
        if cross_up_236.any():
            t_cross = cross_up_236[cross_up_236].index[-1]
            seg = p.loc[:t_cross]
            if _n_consecutive_increasing(seg, int(bars_confirm)):
                buy_event = {"time": t_cross, "price": float(p.loc[t_cross]), "side": "BUY",
                             "label": "FIB Confirm (0%→23.6%)"}

    # --- SELL path ---
    touch100 = look >= fib100 * (1.0 - float(prox))
    sell_event = None
    if touch100.any():
        t_touch = touch100[touch100].index[-1]
        after = look.loc[t_touch:]
        # Cross below 23.6 from above (exactly as requested)
        cross_dn_236 = (after <= fib236) & (after.shift(1) > fib236)
        if cross_dn_236.any():
            t_cross = cross_dn_236[cross_dn_236].index[-1]
            seg = p.loc[:t_cross]
            if _n_consecutive_decreasing(seg, int(bars_confirm)):
                sell_event = {"time": t_cross, "price": float(p.loc[t_cross]), "side": "SELL",
                              "label": "FIB Confirm (100%→23.6%)"}

    # If both exist, pick the most recent
    if buy_event and sell_event:
        return buy_event if buy_event["time"] >= sell_event["time"] else sell_event
    return buy_event or sell_event

def annotate_fib_confirmation_trigger(ax, trig: dict):
    if not trig:
        return
    t = trig.get("time")
    y = trig.get("price")
    side = str(trig.get("side", "")).upper()
    lbl = trig.get("label", "FIB Confirm")
    if t is None or y is None or not np.isfinite(y):
        return
    if side.startswith("B"):
        ax.scatter([t], [y], marker="*", s=220, zorder=13, label=lbl)
    else:
        ax.scatter([t], [y], marker="*", s=220, zorder=13, label=lbl)

def shade_ntd_regions(ax, ntd: pd.Series):
    s = _coerce_1d_series(ntd).dropna()
    if s.empty:
        return
    ax.axhspan(0.75, 1.1, alpha=0.08)
    ax.axhspan(-1.1, -0.75, alpha=0.08)


# =========================
# Part 6/10
# =========================
# ---------------------------
# Ichimoku, Supertrend, PSAR + NTD overlays
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


# =========================
# Part 7/10
# =========================
# ---------------------------
# Sessions (PST) + News (Yahoo Finance) + NTD channel shading + scan caches + fib proximity caches
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

def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

@st.cache_data(ttl=120)
def last_daily_ntd_value(symbol: str, ntd_win: int, years: int):
    try:
        s = fetch_hist(symbol, years=years)
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

@st.cache_data(ttl=120)
def last_daily_fib_position(symbol: str, daily_view_label: str, years: int):
    try:
        s_full = fetch_hist(symbol, years=years)
        s_show = subset_by_daily_view(s_full, daily_view_label)
        s_show = _coerce_1d_series(s_show).dropna()
        if s_show.empty:
            return None
        fibs = fibonacci_levels(s_show)
        if not fibs:
            return None
        px = float(s_show.iloc[-1]) if np.isfinite(s_show.iloc[-1]) else np.nan
        pct = fib_position_percent(px, fibs)
        if not np.isfinite(pct):
            return None
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Fib%": float(pct),
            "Price": float(px),
            "Fib0": float(fibs.get("0%", np.nan)),
            "Fib100": float(fibs.get("100%", np.nan)),
            "Time": s_show.index[-1],
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_hourly_fib_position(symbol: str, period: str = "1d"):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        s = _coerce_1d_series(df["Close"]).ffill().dropna()
        if s.empty:
            return None
        fibs = fibonacci_levels(s)
        if not fibs:
            return None
        px = float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan
        pct = fib_position_percent(px, fibs)
        if not np.isfinite(pct):
            return None
        t = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df.index) else None
        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Fib%": float(pct),
            "Price": float(px),
            "Fib0": float(fibs.get("0%", np.nan)),
            "Fib100": float(fibs.get("100%", np.nan)),
            "Time": t,
        }
    except Exception:
        return None


# =========================
# Part 8/10
# =========================
# ---------------------------
# Scanners (recent BUY, NPX 0.5 cross, SR+BB mid) + forecast
# ---------------------------
def compute_sarimax_forecast(series: pd.Series, steps: int = 30):
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        idx = pd.date_range(pd.Timestamp.today(), periods=steps, freq="D")
        vals = pd.Series(index=idx, data=np.nan)
        ci = pd.DataFrame(index=idx, data={"lower": np.nan, "upper": np.nan})
        return idx, vals, ci

    # fallback naive if statsmodels missing or fails
    if not _HAS_SM or len(s) < 60:
        last = float(s.iloc[-1])
        idx = pd.date_range(s.index[-1] if hasattr(s.index, "freq") else pd.Timestamp.today(), periods=steps + 1, freq="D")[1:]
        vals = pd.Series(index=idx, data=np.full(len(idx), last, dtype=float))
        ci = pd.DataFrame(index=idx, data={"lower": vals * 0.98, "upper": vals * 1.02})
        return idx, vals, ci

    try:
        model = SARIMAX(s, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps)
        mean = fc.predicted_mean
        conf = fc.conf_int(alpha=0.05)
        idx = mean.index
        ci = pd.DataFrame(index=idx, data={
            "lower": conf.iloc[:, 0].astype(float),
            "upper": conf.iloc[:, 1].astype(float),
        })
        return idx, mean.astype(float), ci
    except Exception:
        last = float(s.iloc[-1])
        idx = pd.date_range(pd.Timestamp.today(), periods=steps, freq="D")
        vals = pd.Series(index=idx, data=np.full(len(idx), last, dtype=float))
        ci = pd.DataFrame(index=idx, data={"lower": vals * 0.98, "upper": vals * 1.02})
        return idx, vals, ci

def subset_by_daily_view(series: pd.Series, daily_view_label: str) -> pd.Series:
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return s
    lab = str(daily_view_label)
    if lab == "1Y":
        return s.tail(252)
    if lab == "2Y":
        return s.tail(252 * 2)
    if lab == "5Y":
        return s.tail(252 * 5)
    if lab == "10Y":
        return s.tail(252 * 10)
    if lab == "Max":
        return s
    return s

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float):
    if not np.isfinite(close_val):
        return "n/a"
    # Basic instruction; keep same surface behavior (string returned, sometimes prefixed with ALERT:)
    if np.isfinite(trend_slope) and np.isfinite(global_trend_slope):
        if (trend_slope > 0 and global_trend_slope < 0) or (trend_slope < 0 and global_trend_slope > 0):
            return "ALERT: Local vs Global trend conflict"

    if np.isfinite(trend_slope) and trend_slope > 0 and np.isfinite(buy_val):
        return f"BUY zone near Support {fmt_price_val(buy_val)}"
    if np.isfinite(trend_slope) and trend_slope < 0 and np.isfinite(sell_val):
        return f"SELL zone near Resistance {fmt_price_val(sell_val)}"
    return "Hold / Wait"

def find_macd_hma_sr_signal(close: pd.Series,
                            hma: pd.Series,
                            macd: pd.Series,
                            sup: pd.Series,
                            res: pd.Series,
                            global_trend_slope: float,
                            prox: float = 0.0025):
    c = _coerce_1d_series(close).dropna()
    if c.empty:
        return None
    h = _coerce_1d_series(hma).reindex(c.index)
    m = _coerce_1d_series(macd).reindex(c.index)
    s_sup = _coerce_1d_series(sup).reindex(c.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(c.index).ffill().bfill()

    t = c.index[-1]
    if not np.all(np.isfinite([c.iloc[-1], s_sup.iloc[-1], s_res.iloc[-1]])):
        return None

    # Simple: if global uptrend and near support and MACD rising and price above HMA -> BUY
    if np.isfinite(global_trend_slope) and global_trend_slope > 0:
        near_sup = float(c.iloc[-1]) <= float(s_sup.iloc[-1]) * (1.0 + prox)
        macd_rise = (m.iloc[-1] > m.iloc[-2]) if len(m.dropna()) >= 2 else False
        above_hma = (c.iloc[-1] > h.iloc[-1]) if np.isfinite(h.iloc[-1]) else False
        if near_sup and macd_rise and above_hma:
            return {"time": t, "price": float(c.iloc[-1]), "side": "BUY"}

    # If global downtrend and near resistance and MACD falling and price below HMA -> SELL
    if np.isfinite(global_trend_slope) and global_trend_slope < 0:
        near_res = float(c.iloc[-1]) >= float(s_res.iloc[-1]) * (1.0 - prox)
        macd_fall = (m.iloc[-1] < m.iloc[-2]) if len(m.dropna()) >= 2 else False
        below_hma = (c.iloc[-1] < h.iloc[-1]) if np.isfinite(h.iloc[-1]) else False
        if near_res and macd_fall and below_hma:
            return {"time": t, "price": float(c.iloc[-1]), "side": "SELL"}
    return None

def annotate_macd_signal(ax, t, y, side: str):
    if t is None or y is None or not np.isfinite(y):
        return
    if str(side).upper().startswith("B"):
        ax.scatter([t], [y], marker="^", s=140, zorder=14, label="MACD/HMA BUY")
    else:
        ax.scatter([t], [y], marker="v", s=140, zorder=14, label="MACD/HMA SELL")

@st.cache_data(ttl=120)
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str, years: int):
    try:
        s_full = fetch_hist(symbol, years=years)
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
                                               direction: str = "up",
                                               years: int = 10):
    try:
        s_full = fetch_hist(symbol, years=years)
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

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

@st.cache_data(ttl=120)
def last_daily_sr_reversal_bbmid(symbol: str,
                                 daily_view_label: str,
                                 slope_lb: int,
                                 sr_lb: int,
                                 bb_window: int,
                                 bb_sigma: float,
                                 bb_ema: bool,
                                 prox: float,
                                 bars_confirm: int,
                                 horizon: int,
                                 side: str = "BUY",
                                 min_r2: float = 0.99,
                                 years: int = 10):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol, years=years)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(10, int(slope_lb), int(sr_lb)):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None
        if float(r2) < float(min_r2):
            return None

        want_buy = str(side).upper().startswith("B")
        if want_buy and float(m) <= 0.0:
            return None
        if (not want_buy) and float(m) >= 0.0:
            return None

        res = close_show.rolling(int(sr_lb), min_periods=1).max()
        sup = close_show.rolling(int(sr_lb), min_periods=1).min()

        bb_mid, bb_up, bb_lo, _, _ = compute_bbands(close_show, window=int(bb_window), mult=float(bb_sigma), use_ema=bool(bb_ema))
        if bb_mid.dropna().empty:
            return None

        cross_up, cross_dn = _cross_series(close_show, bb_mid)
        hz = max(1, int(horizon))

        if want_buy:
            if not cross_up.any():
                return None
            t_cross = cross_up[cross_up].index[-1]
            loc = int(close_show.index.get_loc(t_cross))
            j0 = max(0, loc - hz)
            touch_mask = close_show.iloc[j0:loc+1] <= (sup.iloc[j0:loc+1] * (1.0 + float(prox)))
            if not touch_mask.any():
                return None
            t_touch = touch_mask[touch_mask].index[-1]
            if not _n_consecutive_increasing(close_show.loc[:t_cross], int(bars_confirm)):
                return None
        else:
            if not cross_dn.any():
                return None
            t_cross = cross_dn[cross_dn].index[-1]
            loc = int(close_show.index.get_loc(t_cross))
            j0 = max(0, loc - hz)
            touch_mask = close_show.iloc[j0:loc+1] >= (res.iloc[j0:loc+1] * (1.0 - float(prox)))
            if not touch_mask.any():
                return None
            t_touch = touch_mask[touch_mask].index[-1]
            if not _n_consecutive_decreasing(close_show.loc[:t_cross], int(bars_confirm)):
                return None

        bars_since_cross = int((len(close_show) - 1) - int(close_show.index.get_loc(t_cross)))

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        cross_px = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        mid_px = float(bb_mid.loc[t_cross]) if (t_cross in bb_mid.index and np.isfinite(bb_mid.loc[t_cross])) else np.nan
        sup_px = float(sup.loc[t_touch]) if (t_touch in sup.index and np.isfinite(sup.loc[t_touch])) else np.nan
        res_px = float(res.loc[t_touch]) if (t_touch in res.index and np.isfinite(res.loc[t_touch])) else np.nan

        return {
            "Symbol": symbol,
            "Side": "BUY" if want_buy else "SELL",
            "Daily View": daily_view_label,
            "Bars Since Cross": bars_since_cross,
            "Touch Time": t_touch,
            "Cross Time": t_cross,
            "Slope": float(m),
            "R2": float(r2),
            "Support@Touch": sup_px,
            "Resistance@Touch": res_px,
            "BB Mid@Cross": mid_px,
            "Price@Cross": cross_px,
            "Current Price": curr_px,
        }
    except Exception:
        return None


# =========================
# Part 9/10
# =========================
# ---------------------------
# UI sidebar (keeps look/feel: toggles/labels/behavior)
# ---------------------------
st.sidebar.title("Controls")

mode = st.sidebar.radio("Mode:", ["Stocks", "Forex"], index=0)

if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

st.session_state.hist_years = st.sidebar.slider("History years (daily)", 1, 25, int(st.session_state.hist_years), 1)

daily_view = st.sidebar.selectbox("Daily view range:", ["1Y", "2Y", "5Y", "10Y", "Max"], index=3)

# Core lookback settings
sr_lb_daily = st.sidebar.slider("S/R lookback (daily)", 10, 300, 60, 5)
sr_lb_hourly = st.sidebar.slider("S/R lookback (hourly)", 10, 300, 60, 5)
slope_lb_daily = st.sidebar.slider("Slope lookback (daily)", 10, 360, 90, 5)
slope_lb_hourly = st.sidebar.slider("Slope lookback (hourly)", 10, 360, 90, 5)

# Reversal probability / confirmation settings
rev_hist_lb = st.sidebar.slider("Reversal prob hist window", 50, 500, 250, 10)
rev_horizon = st.sidebar.slider("Reversal horizon (bars)", 2, 60, 10, 1)
rev_bars_confirm = st.sidebar.slider("Bars confirm (reversal)", 1, 5, 2, 1)

# Proximity
sr_prox_pct = st.sidebar.slider("Proximity (S/R & Fib) %", 0.05, 2.00, 0.25, 0.05) / 100.0

# Indicators toggles
show_hma = st.sidebar.checkbox("Show HMA", True)
hma_period = st.sidebar.slider("HMA period", 10, 200, 55, 1)

show_macd = st.sidebar.checkbox("Show MACD panel", False)

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", True)
bb_win = st.sidebar.slider("BB window", 10, 120, 20, 1)
bb_mult = st.sidebar.slider("BB sigma", 1.0, 4.0, 2.0, 0.1)
bb_use_ema = st.sidebar.checkbox("BB mid uses EMA", False)

show_fibs = st.sidebar.checkbox("Show Fibonacci levels", True)

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", False)
ichi_conv = st.sidebar.slider("Ichimoku conv", 5, 20, 9, 1)
ichi_base = st.sidebar.slider("Ichimoku base", 10, 60, 26, 1)
ichi_spanb = st.sidebar.slider("Ichimoku spanB", 20, 120, 52, 1)

show_psar = st.sidebar.checkbox("Show PSAR", False)
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.10, 0.02, 0.01)
psar_max = st.sidebar.slider("PSAR max", 0.10, 0.50, 0.20, 0.01)

# Supertrend parameters (daily enabled by default in chart logic)
atr_period = st.sidebar.slider("ATR period (Supertrend)", 5, 30, 10, 1)
atr_mult = st.sidebar.slider("ATR mult (Supertrend)", 1.0, 6.0, 3.0, 0.1)

# NTD/NPX
show_nrsi = st.sidebar.checkbox("Show NTD indicator panel (2nd panel)", True)
show_ntd = st.sidebar.checkbox("Show NTD", True)
shade_ntd = st.sidebar.checkbox("Shade NTD extremes", True)
show_npx_ntd = st.sidebar.checkbox("Show NPX on NTD panel", True)
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses", True)
show_hma_rev_ntd = st.sidebar.checkbox("Show HMA reversal markers (NTD panel)", True)
hma_rev_lb = st.sidebar.slider("HMA reversal lookback", 2, 12, 3, 1)
show_ntd_channel = st.sidebar.checkbox("Shade 'in range' (S↔R) on NTD panel", True)
ntd_window = st.sidebar.slider("NTD/NPX window", 10, 250, 55, 1)

# Forex extras
show_sessions_pst = st.sidebar.checkbox("Show London/NY sessions (PST) [Forex]", True)
show_fx_news = st.sidebar.checkbox("Show Forex news (Yahoo) [Forex]", False)
news_window_days = st.sidebar.slider("News window (days)", 1, 14, 7, 1)

# Bull/Bear tab lookback
bb_period = st.sidebar.selectbox("Bull/Bear lookback", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# Universe lists (keep them simple; you can replace with your original lists if needed)
stock_universe = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "SPY", "QQQ", "IWM", "GLD", "TLT"
]
forex_universe = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"
]
universe = stock_universe if mode == "Stocks" else forex_universe

# NOTE (THIS REQUEST)
st.sidebar.caption("📝 Note: Place Buy Trade Closer to 0% Fibonnaci and Sell trade closer to 100% Fibonacci.")


# =========================
# Part 9.5/10
# =========================
# ---------------------------
# Shared hourly renderer (Stock & Forex)
# ---------------------------
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool,
                        alert_placeholder=None):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

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
        plt.subplots_adjust(hspace=0.05, top=0.90, right=0.93, bottom=0.22)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.85, right=0.93, bottom=0.24)

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

        trig_h = find_slope_trigger_after_band_reversal(hc, yhat_h, upper_h, lower_h, horizon=rev_horizon)
        annotate_slope_trigger(ax2, trig_h)

    # Fibonacci levels + UPDATED Confirmation Trigger (requires 23.6% cross)
    fibs_h = fibonacci_levels(hc)
    if show_fibs and fibs_h:
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    fib_conf_h = find_fib_confirmation_trigger(
        price=hc, fibs=fibs_h, r2=r2_h,
        horizon=rev_horizon, prox=sr_prox_pct, bars_confirm=rev_bars_confirm,
        r2_thr=FIB_CONFIRM_R2
    )
    annotate_fib_confirmation_trigger(ax2, fib_conf_h)

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

    is_alert = isinstance(instr_txt, str) and instr_txt.startswith("ALERT:")
    if alert_placeholder is not None:
        if is_alert:
            alert_placeholder.error(instr_txt)
        else:
            alert_placeholder.empty()

    title_instr = instr_txt
    if alert_placeholder is not None and is_alert:
        title_instr = ""

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
    instr_part = f" — {title_instr} " if isinstance(title_instr, str) and title_instr.strip() else " "
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{instr_part}"
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

    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_hourly})")
        ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
            ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
            if not ntd_trend_h.dropna().empty:
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
        ax2w.legend(loc="lower left", framealpha=0.5, fontsize=9)
        ax2w.set_xlabel("Time (PST)")
    else:
        ax2.set_xlabel("Time (PST)")

    ax2.legend(loc="lower left", framealpha=0.5, fontsize=9)

    if session_handles and session_labels:
        fig2.legend(
            handles=session_handles,
            labels=session_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=True,
            fontsize=9,
            title="Sessions (PST)",
            title_fontsize=9
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
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.set_xlim(xlim_price)
        axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)


# =========================
# Part 10/10
# =========================
# ---------------------------
# Tabs (unchanged order / names)
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "NPX 0.5-Cross Scanner",
    "Daily Slope+BB Reversal Scanner",
    "Fibonacci Proximity Scanner"
])

# ---------------------------
# TAB 1: ORIGINAL FORECAST
# ---------------------------
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data is cached for ~2 minutes after first fetch. "
            "Charts stay on the last RUN ticker until you run again.")

    st.caption("📝 Place Buy Trade Closer to 0% Fibonnaci and Sell trade closer to 100% Fibonacci.")

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
    alert_box = st.empty()

    if run_clicked:
        df_hist = fetch_hist(sel, years=st.session_state.hist_years)
        df_ohlc = fetch_hist_ohlc(sel, years=st.session_state.hist_years)
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
        try:
            alert_box.empty()
        except Exception:
            pass

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
            bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_show = compute_hma(df, period=hma_period).reindex(df_show.index)
            macd_d, macd_sig_d, macd_hist_d = compute_macd(df_show)

            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            # Daily Supertrend ON by default
            st_d_line = pd.Series(index=df_show.index, dtype=float)
            try:
                if df_ohlc is not None and not df_ohlc.empty and {"High","Low","Close"}.issubset(df_ohlc.columns):
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    ohlc_show = df_ohlc.loc[(df_ohlc.index >= x0) & (df_ohlc.index <= x1)]
                    st_d = compute_supertrend(ohlc_show, atr_period=atr_period, atr_mult=atr_mult)
                    if "ST" in st_d.columns:
                        st_d_line = _coerce_1d_series(st_d["ST"]).reindex(df_show.index).ffill().bfill()
            except Exception:
                pass

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

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

            if not _coerce_1d_series(st_d_line).dropna().empty:
                ax.plot(st_d_line.index, st_d_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
                bounce_sig_d = find_band_bounce_signal(df_show, upper_d_show, lower_d_show, m_d)
                if bounce_sig_d is not None:
                    annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

                trig_d = find_slope_trigger_after_band_reversal(df_show, yhat_d_show, upper_d_show, lower_d_show, horizon=rev_horizon)
                annotate_slope_trigger(ax, trig_d)

            # Daily Fibonacci + UPDATED Confirmation Trigger (requires 23.6% cross)
            fibs_d = fibonacci_levels(df_show)
            if show_fibs and fibs_d:
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="dotted", linewidth=1)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")

            fib_conf_d = find_fib_confirmation_trigger(
                price=df_show, fibs=fibs_d, r2=r2_d,
                horizon=rev_horizon, prox=sr_prox_pct, bars_confirm=rev_bars_confirm,
                r2_thr=FIB_CONFIRM_R2
            )
            annotate_fib_confirmation_trigger(ax, fib_conf_d)

            macd_sig_d2 = find_macd_hma_sr_signal(
                close=df_show, hma=hma_d_show, macd=macd_d, sup=sup_d_show, res=res_d_show,
                global_trend_slope=global_m_d, prox=sr_prox_pct
            )
            macd_instr_txt_d = "MACD/HMA55: n/a"
            if macd_sig_d2 is not None and np.isfinite(macd_sig_d2.get("price", np.nan)):
                macd_instr_txt_d = f"MACD/HMA55: {macd_sig_d2['side']} @ {fmt_price_val(macd_sig_d2['price'])}"
                annotate_macd_signal(ax, macd_sig_d2["time"], macd_sig_d2["price"], macd_sig_d2["side"])

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
            ax.legend(loc="lower left", framealpha=0.5, fontsize=9)

            axdw.set_title(f"Daily Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_daily})")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.dropna().empty:
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
            axdw.legend(loc="lower left", framealpha=0.5, fontsize=9)

            style_axes(ax)
            style_axes(axdw)
            st.pyplot(fig)

            if show_macd and not macd_d.dropna().empty:
                figm, axm = plt.subplots(figsize=(14, 2.6))
                axm.set_title("MACD (optional)")
                axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
                axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
                axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
                style_axes(axm)
                st.pyplot(figm)

        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            render_hourly_views(
                sel=disp_ticker,
                intraday=intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.hour_range,
                is_forex=(mode == "Forex"),
                alert_placeholder=alert_box
            )
        else:
            try:
                alert_box.empty()
            except Exception:
                pass

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
            "Lower":    st.session_state.fc_ci["lower"],
            "Upper":    st.session_state.fc_ci["upper"]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")

# ---------------------------
# TAB 2: ENHANCED FORECAST
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
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

            # Supertrend ON by default in Enhanced Daily
            st_d_line = pd.Series(index=df_show.index, dtype=float)
            try:
                if df_ohlc is not None and not df_ohlc.empty and len(df_show.index) > 0:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    ohlc_show = df_ohlc.loc[(df_ohlc.index >= x0) & (df_ohlc.index <= x1)]
                    st_d = compute_supertrend(ohlc_show, atr_period=atr_period, atr_mult=atr_mult)
                    if "ST" in st_d.columns:
                        st_d_line = _coerce_1d_series(st_d["ST"]).reindex(df_show.index).ffill().bfill()
            except Exception:
                pass

            # compute R² for confirmation trigger on enhanced daily chart
            _, _, _, _, r2_enh = regression_with_band(df_show, lookback=int(slope_lb_daily))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if not _coerce_1d_series(st_d_line).dropna().empty:
                ax.plot(st_d_line.index, st_d_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

            if not res_d_show.empty and not sup_d_show.empty:
                ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")

            # Daily Fibonacci + UPDATED Confirmation Trigger
            fibs_d = fibonacci_levels(df_show)
            if show_fibs and fibs_d:
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="dotted", linewidth=1)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")

            fib_conf = find_fib_confirmation_trigger(
                price=df_show, fibs=fibs_d, r2=r2_enh,
                horizon=rev_horizon, prox=sr_prox_pct, bars_confirm=rev_bars_confirm,
                r2_thr=FIB_CONFIRM_R2
            )
            annotate_fib_confirmation_trigger(ax, fib_conf)

            macd_sig2 = find_macd_hma_sr_signal(df_show, hma_d_show, macd_d, sup_d_show, res_d_show, global_m_d, prox=sr_prox_pct)
            macd_txt = "MACD/HMA55: n/a"
            if macd_sig2 is not None and np.isfinite(macd_sig2.get("price", np.nan)):
                macd_txt = f"MACD/HMA55: {macd_sig2['side']} @ {fmt_price_val(macd_sig2['price'])}"
                annotate_macd_signal(ax, macd_sig2["time"], macd_sig2["price"], macd_sig2["side"])
            ax.text(0.01, 0.98, macd_txt, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8))

            ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
            style_axes(ax)
            st.pyplot(fig)

            if show_macd and not macd_d.dropna().empty:
                figm, axm = plt.subplots(figsize=(14, 2.6))
                axm.set_title("MACD (optional)")
                axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
                axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
                axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
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
            "Lower":    ci["lower"],
            "Upper":    ci["upper"]
        }, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    st.caption("Simple lookback performance overview (based on Bull/Bear lookback selection).")

    sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
    try:
        dfp = yf.download(sel_bb, period=bb_period, interval="1d", auto_adjust=True, progress=False)[["Close"]].dropna()
    except Exception:
        dfp = pd.DataFrame()

    if dfp.empty:
        st.warning("No data available.")
    else:
        s = dfp["Close"].astype(float)
        ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if len(s) > 1 else np.nan
        st.metric(label=f"{sel_bb} return over {bb_period}", value=fmt_pct(ret))
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title(f"{sel_bb} — {bb_period} Close")
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
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
                val, ts = last_daily_ntd_value(sym, ntd_window, years=st.session_state.hist_years)
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
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
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
            r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view, years=st.session_state.hist_years)
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
                local_slope_lb=lb_local, max_abs_npx_at_cross=eps0, direction="up",
                years=st.session_state.hist_years
            )
            if r_up is not None and int(r_up.get("Bars Since", 9999)) <= int(max_bars0):
                rows_up.append(r_up)

            r_dn = last_daily_npx_zero_cross_with_local_slope(
                sym, ntd_win=ntd_window, daily_view_label=daily_view,
                local_slope_lb=lb_local, max_abs_npx_at_cross=eps0, direction="down",
                years=st.session_state.hist_years
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
# TAB 9: Daily Slope + S/R reversal + BB mid cross scanner
# ---------------------------
with tab9:
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
                years=st.session_state.hist_years
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
                years=st.session_state.hist_years
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
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 10: Fibonacci Proximity Scanner (NEW)
# ---------------------------
with tab10:
    st.header("Fibonacci Proximity Scanner — Closest to 0% and 100%")
    st.caption("Shows symbols closest to the Fibonacci edges. 0% = LOW, 100% = HIGH.")

    frame = st.radio("Frame:", ["Daily", "Hourly (intraday)", "Both"], index=2, key=f"fib_scan_frame_{mode}")
    top_n = st.slider("Top N per edge", 3, 30, 10, 1, key=f"fib_topn_{mode}")

    period = st.selectbox("Hourly lookback (scanner):", ["1d", "2d", "4d"], index=0, key=f"fib_scan_period_{mode}")

    run_fib_scan = st.button("Run Fibonacci Proximity Scan", key=f"btn_run_fib_scan_{mode}")

    if run_fib_scan:
        rows = []
        if frame in ("Daily", "Both"):
            for sym in universe:
                r = last_daily_fib_position(sym, daily_view_label=daily_view, years=st.session_state.hist_years)
                if r is not None and np.isfinite(r.get("Fib%", np.nan)):
                    rows.append(r)

        if frame in ("Hourly (intraday)", "Both"):
            for sym in universe:
                r = last_hourly_fib_position(sym, period=period)
                if r is not None and np.isfinite(r.get("Fib%", np.nan)):
                    rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out["Fib%"] = out["Fib%"].astype(float)
            out["Dist to 0%"] = out["Fib%"].astype(float)
            out["Dist to 100%"] = (100.0 - out["Fib%"].astype(float)).astype(float)

            left, right = st.columns(2)

            with left:
                st.subheader("Closest to 0% Fibonacci (LOW)")
                o0 = out.sort_values(["Dist to 0%","Fib%"], ascending=[True, True]).head(int(top_n))
                st.dataframe(o0.reset_index(drop=True), use_container_width=True)

            with right:
                st.subheader("Closest to 100% Fibonacci (HIGH)")
                o1 = out.sort_values(["Dist to 100%","Fib%"], ascending=[True, False]).head(int(top_n))
                st.dataframe(o1.reset_index(drop=True), use_container_width=True)
