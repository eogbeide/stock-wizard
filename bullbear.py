# bullbear.py
# ============================================================
# Batch 1: Parts 1/10 to 5/10
# ============================================================

# =========================
# Part 1/10 — Imports, config, constants
# =========================
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None


st.set_page_config(page_title="Stock Wizard — BullBear", layout="wide")

PACIFIC = pytz.timezone("America/Los_Angeles")


# =========================
# Part 2/10 — Formatting + small utilities
# =========================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100.0 * float(x):.{digits}f}%"
    except Exception:
        return "n/a"


def fmt_slope(x: Any, digits: int = 6) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def fmt_r2(x: Any, digits: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "n/a"


def fmt_price_val(x: Any, digits: int = 4) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        v = float(x)
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 100:
            return f"{v:.2f}"
        if abs(v) >= 1:
            return f"{v:.4f}"
        return f"{v:.6f}"
    except Exception:
        return "n/a"


def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame):
        if "Close" in x.columns:
            return x["Close"].copy()
        if x.shape[1] == 1:
            return x.iloc[:, 0].copy()
        return pd.Series(dtype=float)
    try:
        return pd.Series(x).copy()
    except Exception:
        return pd.Series(dtype=float)


def _safe_last_float(s: pd.Series) -> float:
    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return float("nan")
    try:
        v = float(s.iloc[-1])
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def label_on_left(ax, y: float, text: str, color: str = "black"):
    try:
        xmin, _ = ax.get_xlim()
        ax.text(
            xmin,
            y,
            f" {text}",
            va="center",
            ha="left",
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7),
            zorder=10,
        )
    except Exception:
        pass


# =========================
# Part 3/10 — Data fetching + time index normalization
# =========================
def _ensure_pacific_dtindex(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return idx
    try:
        if idx.tz is None:
            # yfinance often returns tz-naive indexes. Assume UTC then convert.
            idx = idx.tz_localize("UTC")
        return idx.tz_convert(PACIFIC)
    except Exception:
        return idx


def _ensure_pacific_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = _ensure_pacific_dtindex(out.index)
    return out


@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist(symbol: str, years: int = 10) -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    try:
        period = f"{int(years)}y"
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        df = _ensure_pacific_df(df)
        if df.empty:
            return pd.Series(dtype=float)
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        return df["Close"].astype(float).dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, years: int = 10) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        period = f"{int(years)}y"
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        df = _ensure_pacific_df(df)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_max(symbol: str) -> pd.Series:
    if yf is None:
        return pd.Series(dtype=float)
    try:
        df = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        df = _ensure_pacific_df(df)
        if df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        return df["Close"].astype(float).dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        df = _ensure_pacific_df(df)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna(how="all")
    except Exception:
        return pd.DataFrame()


def subset_by_daily_view(series: pd.Series, label: str) -> pd.Series:
    s = _coerce_1d_series(series).dropna()
    if s.empty:
        return s

    label = str(label).strip().upper()
    end = s.index[-1]

    def _cut(days: int) -> pd.Series:
        start = end - pd.Timedelta(days=int(days))
        return s[s.index >= start]

    if label in ("ALL", "MAX"):
        return s
    if label in ("10Y", "10 Y", "10YR"):
        return _cut(365 * 10 + 10)
    if label in ("5Y", "5 Y", "5YR"):
        return _cut(365 * 5 + 5)
    if label in ("3Y", "3 Y", "3YR"):
        return _cut(365 * 3 + 3)
    if label in ("2Y", "2 Y", "2YR"):
        return _cut(365 * 2 + 2)
    if label in ("1Y", "1 Y", "1YR"):
        return _cut(365 + 1)
    if label in ("6M", "6 M"):
        return _cut(183)
    if label in ("3M", "3 M"):
        return _cut(92)
    if label in ("1M", "1 M"):
        return _cut(31)

    # fallback
    return s


# =========================
# Part 4/10 — Indicators (HMA, MACD, BBands, ATR/Supertrend, Ichimoku, PSAR)
# =========================
def wma(series: pd.Series, period: int) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    n = int(period)
    if n <= 1:
        return s
    weights = np.arange(1, n + 1, dtype=float)

    def _calc(x):
        if np.any(~np.isfinite(x)):
            return np.nan
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(n, min_periods=n).apply(_calc, raw=True)


def compute_hma(series: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    n = int(period)
    if n <= 1:
        return s
    half = max(1, n // 2)
    sqrt_n = max(1, int(math.sqrt(n)))
    h = 2 * wma(s, half) - wma(s, n)
    return wma(h, sqrt_n)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    c = _coerce_1d_series(close).astype(float)
    ema_fast = c.ewm(span=int(fast), adjust=False).mean()
    ema_slow = c.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    c = _coerce_1d_series(close).astype(float)
    n = int(window)
    if n <= 1:
        mid = c
        std = c * np.nan
    else:
        if use_ema:
            mid = c.ewm(span=n, adjust=False).mean()
            # approximate EMA std using rolling std
            std = c.rolling(n, min_periods=max(2, n // 2)).std()
        else:
            mid = c.rolling(n, min_periods=max(2, n // 2)).mean()
            std = c.rolling(n, min_periods=max(2, n // 2)).std()

    upper = mid + float(mult) * std
    lower = mid - float(mult) * std

    # %B (0=lower, 1=upper)
    pctb = (c - lower) / (upper - lower)
    # Normalized Band Width (dimensionless)
    nbb = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, pctb, nbb


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    c = _coerce_1d_series(close).astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(int(period), min_periods=max(2, int(period) // 2)).mean()


def compute_supertrend(df_ohlc: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    df = df_ohlc.copy()
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame(index=df_ohlc.index)

    atr = average_true_range(df["High"], df["Low"], df["Close"], period=int(atr_period))
    hl2 = (df["High"].astype(float) + df["Low"].astype(float)) / 2.0

    upperband = hl2 + float(atr_mult) * atr
    lowerband = hl2 - float(atr_mult) * atr

    st_line = pd.Series(index=df.index, dtype=float)
    in_uptrend = pd.Series(index=df.index, dtype=bool)

    for i in range(len(df)):
        if i == 0:
            st_line.iloc[i] = upperband.iloc[i]
            in_uptrend.iloc[i] = True
            continue

        prev_st = st_line.iloc[i - 1]
        prev_up = in_uptrend.iloc[i - 1]
        c = float(df["Close"].iloc[i])

        ub = float(upperband.iloc[i]) if np.isfinite(upperband.iloc[i]) else np.nan
        lb = float(lowerband.iloc[i]) if np.isfinite(lowerband.iloc[i]) else np.nan

        # determine trend
        if prev_up:
            if np.isfinite(lb) and c < lb:
                in_uptrend.iloc[i] = False
                st_line.iloc[i] = ub
            else:
                in_uptrend.iloc[i] = True
                st_line.iloc[i] = max(lb, prev_st) if np.isfinite(lb) and np.isfinite(prev_st) else lb
        else:
            if np.isfinite(ub) and c > ub:
                in_uptrend.iloc[i] = True
                st_line.iloc[i] = lb
            else:
                in_uptrend.iloc[i] = False
                st_line.iloc[i] = min(ub, prev_st) if np.isfinite(ub) and np.isfinite(prev_st) else ub

    return pd.DataFrame({"ST": st_line, "in_uptrend": in_uptrend}, index=df.index)


def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    c = _coerce_1d_series(close).astype(float)

    conv_line = (h.rolling(conv).max() + l.rolling(conv).min()) / 2.0
    base_line = (h.rolling(base).max() + l.rolling(base).min()) / 2.0
    span_a = (conv_line + base_line) / 2.0
    span_b_line = (h.rolling(span_b).max() + l.rolling(span_b).min()) / 2.0

    if shift_cloud:
        span_a = span_a.shift(base)
        span_b_line = span_b_line.shift(base)

    lagging = c.shift(-base) if shift_cloud else c.shift(base)
    return conv_line, base_line, span_a, span_b_line, lagging


def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return pd.DataFrame()

    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    n = len(df)

    psar = np.zeros(n, dtype=float)
    uptrend = True
    af = float(step)
    ep = high[0]
    psar[0] = low[0]

    for i in range(1, n):
        prev_psar = psar[i - 1]

        psar[i] = prev_psar + af * (ep - prev_psar)

        if uptrend:
            psar[i] = min(psar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = low[i]
                af = step
        else:
            psar[i] = max(psar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = high[i]
                af = step

    return pd.DataFrame({"PSAR": psar, "in_uptrend": uptrend}, index=df.index)


# =========================
# Part 5/10 — Regression band, NTD/NPX, crossings, forecast
# =========================
def regression_with_band(series: pd.Series, lookback: int = 90):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.shape[0] < 3:
        idx = s.index
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, float("nan"), float("nan")

    n = len(s)
    lb = max(3, min(int(lookback), n))
    seg = s.iloc[-lb:]
    x_local = np.arange(lb, dtype=float)

    try:
        m, b = np.polyfit(x_local, seg.to_numpy(dtype=float), 1)
    except Exception:
        idx = s.index
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, float("nan"), float("nan")

    x_start = n - lb
    x_full = np.arange(n, dtype=float)
    yhat_full = m * (x_full - x_start) + b
    yhat = pd.Series(yhat_full, index=s.index)

    # residual sigma on segment
    yhat_seg = m * x_local + b
    resid = seg.to_numpy(dtype=float) - yhat_seg
    sigma = float(np.nanstd(resid, ddof=1)) if lb >= 3 else float(np.nanstd(resid))

    upper = yhat + 2.0 * sigma
    lower = yhat - 2.0 * sigma

    # r2 on segment
    y = seg.to_numpy(dtype=float)
    ss_res = float(np.nansum((y - yhat_seg) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return yhat, upper, lower, float(m), float(r2)


def draw_trend_direction_line(ax, series: pd.Series, label_prefix: str = "Trend (global)") -> float:
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.shape[0] < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    try:
        m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
        y = m * x + b
        ax.plot(s.index, y, linewidth=2.0, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
        return float(m)
    except Exception:
        return float("nan")


def slope_line(series: pd.Series, lookback: int = 90) -> Tuple[pd.Series, float]:
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.shape[0] < 3:
        return pd.Series(index=s.index, dtype=float), float("nan")
    lb = max(3, min(int(lookback), len(s)))
    seg = s.iloc[-lb:]
    x = np.arange(lb, dtype=float)
    try:
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
    except Exception:
        return pd.Series(index=s.index, dtype=float), float("nan")

    y_seg = m * x + b
    line = pd.Series(y_seg, index=seg.index)
    return line, float(m)


def compute_normalized_trend(price: pd.Series, window: int = 63) -> pd.Series:
    p = _coerce_1d_series(price).astype(float)
    w = max(5, int(window))
    roll = p.rolling(w, min_periods=max(5, w // 3))
    hi = roll.max()
    lo = roll.min()
    mid = (hi + lo) / 2.0
    half_rng = (hi - lo) / 2.0
    ntd = (p - mid) / half_rng.replace(0, np.nan)
    return ntd.clip(-1.2, 1.2)


def compute_normalized_price(price: pd.Series, window: int = 63) -> pd.Series:
    p = _coerce_1d_series(price).astype(float)
    w = max(5, int(window))
    roll = p.rolling(w, min_periods=max(5, w // 3))
    hi = roll.max()
    lo = roll.min()
    npx = (p - lo) / (hi - lo).replace(0, np.nan)
    return npx.clip(-0.2, 1.2)


def shade_ntd_regions(ax, ntd: pd.Series):
    s = _coerce_1d_series(ntd).astype(float)
    ax.axhspan(0.75, 1.2, alpha=0.06)
    ax.axhspan(-1.2, -0.75, alpha=0.06)


def _cross_series(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    A = _coerce_1d_series(a).astype(float)
    B = _coerce_1d_series(b).astype(float).reindex(A.index)
    prevA, prevB = A.shift(1), B.shift(1)
    cross_up = (A >= B) & (prevA < prevB)
    cross_dn = (A <= B) & (prevA > prevB)
    return cross_up.fillna(False), cross_dn.fillna(False)


def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    x = _coerce_1d_series(npx).astype(float)
    y = _coerce_1d_series(ntd).astype(float).reindex(x.index)
    ax.plot(x.index, (x - 0.5) * 2.0, linewidth=1.2, label="NPX (scaled to [-1,1])")
    if mark_crosses:
        cu, cd = _cross_series((x - 0.5) * 2.0, y)
        if cu.any():
            ax.scatter(x.index[cu], y.loc[cu], marker="o", s=35, zorder=5, label="NPX↑NTD")
        if cd.any():
            ax.scatter(x.index[cd], y.loc[cd], marker="x", s=35, zorder=5, label="NPX↓NTD")


def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd).astype(float)
    if s.dropna().empty:
        return
    if not np.isfinite(trend_slope):
        return
    if trend_slope > 0:
        mask = s <= lower
        if mask.any():
            ax.scatter(s.index[mask], s[mask], marker="^", s=45, zorder=6, label="Oversold (Uptrend)")
    else:
        mask = s >= upper
        if mask.any():
            ax.scatter(s.index[mask], s[mask], marker="v", s=45, zorder=6, label="Overbought (Downtrend)")


def overlay_hma_reversal_on_ntd(ax, close: pd.Series, hma: pd.Series, lookback: int = 10, period: int = 55, ntd: pd.Series | None = None):
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(hma).astype(float).reindex(c.index)
    if c.shape[0] < max(lookback, 5):
        return
    d = h.diff()
    # Mark HMA slope flips
    flip = (d.shift(1) < 0) & (d > 0)
    if flip.any():
        ax.scatter(c.index[flip], 0.0, marker="*", s=90, zorder=7, label=f"HMA({period}) slope flip")


def _n_consecutive_increasing(s: pd.Series, n: int) -> bool:
    s = _coerce_1d_series(s).astype(float).dropna()
    if len(s) < n + 1:
        return False
    last = s.iloc[-(n + 1):]
    dif = last.diff().dropna()
    return bool((dif > 0).all())


def _n_consecutive_decreasing(s: pd.Series, n: int) -> bool:
    s = _coerce_1d_series(s).astype(float).dropna()
    if len(s) < n + 1:
        return False
    last = s.iloc[-(n + 1):]
    dif = last.diff().dropna()
    return bool((dif < 0).all())


def slope_reversal_probability(price: pd.Series,
                               current_slope: float,
                               hist_window: int = 600,
                               slope_window: int = 90,
                               horizon: int = 10) -> float:
    """
    Empirical estimate:
      Over the last hist_window bars, compute rolling slope over slope_window.
      Count how often the slope sign flips within the next `horizon` bars.
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    if p.shape[0] < max(50, slope_window + horizon + 5):
        return float("nan")
    hw = min(int(hist_window), len(p) - (slope_window + horizon + 1))
    if hw < 50:
        return float("nan")

    start = len(p) - hw - (slope_window + horizon)
    end = len(p) - (slope_window + horizon)
    if start < 0:
        start = 0

    idx = p.index
    slopes = np.full(len(p), np.nan, dtype=float)

    for i in range(slope_window, len(p)):
        seg = p.iloc[i - slope_window:i]
        if seg.dropna().shape[0] < slope_window * 0.8:
            continue
        x = np.arange(len(seg), dtype=float)
        try:
            m, _ = np.polyfit(x, seg.to_numpy(dtype=float), 1)
            slopes[i] = float(m)
        except Exception:
            continue

    base = slopes[start:end]
    if np.sum(np.isfinite(base)) < 30:
        return float("nan")

    flips = 0
    total = 0
    for i in range(start, end):
        m0 = slopes[i]
        if not np.isfinite(m0) or m0 == 0:
            continue
        future = slopes[i + 1:i + 1 + horizon]
        if np.sum(np.isfinite(future)) < max(1, horizon // 2):
            continue
        total += 1
        # sign flip means exists future with opposite sign
        if np.any(np.sign(future[np.isfinite(future)]) == -np.sign(m0)):
            flips += 1

    if total == 0:
        return float("nan")

    return float(flips / total)


def find_band_bounce_signal(price: pd.Series, upper: pd.Series, lower: pd.Series, slope: float):
    """
    Very simple "bounce" detector:
      - If slope > 0: look for last bar where price dipped below lower band and then closed back above it.
      - If slope < 0: look for last bar where price popped above upper band and then closed back below it.
    """
    p = _coerce_1d_series(price).astype(float)
    up = _coerce_1d_series(upper).astype(float).reindex(p.index)
    lo = _coerce_1d_series(lower).astype(float).reindex(p.index)
    if p.dropna().empty or up.dropna().empty or lo.dropna().empty:
        return None

    lb = 8
    seg = p.tail(lb)
    seg_up = up.reindex(seg.index)
    seg_lo = lo.reindex(seg.index)

    if not np.isfinite(slope):
        return None

    if slope > 0:
        # bounce up from lower band
        dipped = (seg.shift(1) < seg_lo.shift(1)) & (seg >= seg_lo)
        if dipped.any():
            t = dipped[dipped].index[-1]
            return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    if slope < 0:
        # bounce down from upper band
        popped = (seg.shift(1) > seg_up.shift(1)) & (seg <= seg_up)
        if popped.any():
            t = popped[popped].index[-1]
            return {"time": t, "price": float(p.loc[t]), "side": "SELL"}

    return None


def annotate_crossover(ax, t, y, side: str):
    try:
        ax.scatter([t], [y], s=90, marker="o", zorder=9)
        ax.text(t, y, f" {side}", fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8), zorder=10)
    except Exception:
        pass


def find_slope_trigger_after_band_reversal(price: pd.Series, yhat: pd.Series, upper: pd.Series, lower: pd.Series, horizon: int = 10):
    """
    Optional trigger: after a band bounce signal, look for crossing above/below the regression line within `horizon`.
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    yh = _coerce_1d_series(yhat).astype(float).reindex(p.index)
    if p.empty or yh.dropna().empty:
        return None
    cu, cd = _cross_series(p, yh)
    if cu.any():
        t = cu[cu].index[-1]
        return {"time": t, "side": "TRIGGER↑", "price": float(p.loc[t])}
    if cd.any():
        t = cd[cd].index[-1]
        return {"time": t, "side": "TRIGGER↓", "price": float(p.loc[t])}
    return None


def annotate_slope_trigger(ax, trig: dict | None):
    if not isinstance(trig, dict):
        return
    try:
        t = trig.get("time")
        y = trig.get("price")
        side = trig.get("side", "")
        ax.scatter([t], [y], s=110, marker="D", zorder=9)
        ax.text(t, y, f" {side}", fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8), zorder=10)
    except Exception:
        pass


def annotate_macd_signal(ax, t, y, side: str):
    try:
        ax.scatter([t], [y], s=140, marker="P", zorder=10)
        ax.text(t, y, f" {side}", fontsize=9, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.8), zorder=11)
    except Exception:
        pass


def find_macd_hma_sr_signal(close: pd.Series,
                            hma: pd.Series,
                            macd: pd.Series,
                            sup: pd.Series,
                            res: pd.Series,
                            global_trend_slope: float,
                            prox: float = 0.01):
    """
    Lightweight signal:
      BUY  if MACD crosses above 0, close near support, and global trend is up.
      SELL if MACD crosses below 0, close near resistance, and global trend is down.
    """
    c = _coerce_1d_series(close).astype(float).dropna()
    m = _coerce_1d_series(macd).astype(float).reindex(c.index)
    s_sup = _coerce_1d_series(sup).astype(float).reindex(c.index)
    s_res = _coerce_1d_series(res).astype(float).reindex(c.index)
    if c.shape[0] < 5:
        return None

    cross_up = (m >= 0) & (m.shift(1) < 0)
    cross_dn = (m <= 0) & (m.shift(1) > 0)

    last_i = c.index[-1]
    px = float(c.loc[last_i])
    sup_v = float(s_sup.loc[last_i]) if np.isfinite(s_sup.loc[last_i]) else np.nan
    res_v = float(s_res.loc[last_i]) if np.isfinite(s_res.loc[last_i]) else np.nan

    near_sup = np.isfinite(sup_v) and px <= sup_v * (1.0 + float(prox))
    near_res = np.isfinite(res_v) and px >= res_v * (1.0 - float(prox))

    if np.isfinite(global_trend_slope) and global_trend_slope > 0 and near_sup and cross_up.any():
        t = cross_up[cross_up].index[-1]
        return {"time": t, "price": float(c.loc[t]), "side": "BUY"}
    if np.isfinite(global_trend_slope) and global_trend_slope < 0 and near_res and cross_dn.any():
        t = cross_dn[cross_dn].index[-1]
        return {"time": t, "price": float(c.loc[t]), "side": "SELL"}
    return None


def fibonacci_levels(series: pd.Series) -> Dict[str, float]:
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty:
        return {}
    hi = float(s.max())
    lo = float(s.min())
    if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
        return {}
    # 0% = High, 100% = Low (as requested)
    ratios = [
        ("0%", 0.0),
        ("23.6%", 0.236),
        ("38.2%", 0.382),
        ("50%", 0.5),
        ("61.8%", 0.618),
        ("78.6%", 0.786),
        ("100%", 1.0),
    ]
    out = {}
    for lbl, r in ratios:
        out[lbl] = hi - (hi - lo) * r
    return out


def fib_reversal_trigger_from_extremes(series: pd.Series,
                                       proximity_pct_of_range: float = 0.02,
                                       confirm_bars: int = 2,
                                       lookback_bars: int = 90):
    """
    Confirmed Fib reversal trigger:
      BUY  when price touches near 100% (low) and prints consecutive higher closes.
      SELL when price touches near 0% (high) and prints consecutive lower closes.
    Returns dict or None.
    """
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.shape[0] < max(10, confirm_bars + 3):
        return None

    s = s.tail(int(lookback_bars)).copy()
    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs["0%"])
    lo = float(fibs["100%"])
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return None

    thr = float(proximity_pct_of_range) * rng

    touch_high = (s >= hi - thr)
    touch_low = (s <= lo + thr)

    # BUY from low: find most recent low-touch then confirm higher closes
    if touch_low.any():
        t_touch = touch_low[touch_low].index[-1]
        seg = s.loc[t_touch:]
        if _n_consecutive_increasing(seg, int(confirm_bars)):
            return {
                "side": "BUY",
                "from_level": "100%",
                "touch_time": t_touch,
                "touch_price": float(s.loc[t_touch]),
                "last_time": s.index[-1],
                "last_price": float(s.iloc[-1]),
            }

    # SELL from high
    if touch_high.any():
        t_touch = touch_high[touch_high].index[-1]
        seg = s.loc[t_touch:]
        if _n_consecutive_decreasing(seg, int(confirm_bars)):
            return {
                "side": "SELL",
                "from_level": "0%",
                "touch_time": t_touch,
                "touch_price": float(s.loc[t_touch]),
                "last_time": s.index[-1],
                "last_price": float(s.iloc[-1]),
            }

    return None


def current_daily_pivots(df_ohlc: pd.DataFrame) -> Dict[str, float]:
    if df_ohlc is None or df_ohlc.empty or not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        return {}
    d = df_ohlc.dropna().copy()
    if d.shape[0] < 2:
        return {}
    # use previous day
    prev = d.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    p = (h + l + c) / 3.0
    r1 = 2 * p - l
    s1 = 2 * p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2}


def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float | None = None) -> str:
    if not np.isfinite(close_val) or not np.isfinite(buy_val) or not np.isfinite(sell_val):
        return "ALERT: missing price/S/R values."

    bias = "UP" if np.isfinite(trend_slope) and trend_slope > 0 else ("DOWN" if np.isfinite(trend_slope) and trend_slope < 0 else "FLAT")

    if close_val <= buy_val:
        action = f"BUY zone (≤ Support {fmt_price_val(buy_val)})"
    elif close_val >= sell_val:
        action = f"SELL zone (≥ Resistance {fmt_price_val(sell_val)})"
    else:
        action = f"IN RANGE (Support {fmt_price_val(buy_val)} ↔ Resistance {fmt_price_val(sell_val)})"

    extra = ""
    if global_trend_slope is not None and np.isfinite(global_trend_slope):
        g = "UP" if global_trend_slope > 0 else ("DOWN" if global_trend_slope < 0 else "FLAT")
        extra = f" | Global trend: {g}"

    return f"{symbol}: {action} | Local slope bias: {bias}{extra}"


@st.cache_data(ttl=120, show_spinner=False)
def compute_sarimax_forecast(close: pd.Series, horizon_days: int = 30):
    s = _coerce_1d_series(close).astype(float).dropna()
    if s.shape[0] < 50:
        # naive fallback
        idx = pd.date_range(start=(s.index[-1] + pd.Timedelta(days=1)) if not s.empty else pd.Timestamp.now(tz=PACIFIC),
                            periods=horizon_days, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1]) if not s.empty else np.nan] * horizon_days, index=idx)
        ci = pd.DataFrame({"lower": vals * np.nan, "upper": vals * np.nan}, index=idx)
        return idx, vals, ci

    if SARIMAX is None:
        # naive fallback
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1])] * horizon_days, index=idx)
        ci = pd.DataFrame({"lower": vals * np.nan, "upper": vals * np.nan}, index=idx)
        return idx, vals, ci

    # Build SARIMAX on log price for stability
    y = np.log(s.replace(0, np.nan)).dropna()
    if y.shape[0] < 50:
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1])] * horizon_days, index=idx)
        ci = pd.DataFrame({"lower": vals * np.nan, "upper": vals * np.nan}, index=idx)
        return idx, vals, ci

    try:
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)

        fc = res.get_forecast(steps=int(horizon_days))
        mean = fc.predicted_mean
        conf = fc.conf_int()

        # convert back to price
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D", tz=PACIFIC)
        vals = pd.Series(np.exp(mean.values), index=idx)
        lower = pd.Series(np.exp(conf.iloc[:, 0].values), index=idx)
        upper = pd.Series(np.exp(conf.iloc[:, 1].values), index=idx)
        ci = pd.DataFrame({0: lower, 1: upper}, index=idx)
        return idx, vals, ci
    except Exception:
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1])] * horizon_days, index=idx)
        ci = pd.DataFrame({"lower": vals * np.nan, "upper": vals * np.nan}, index=idx)
        return idx, vals, ci


def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return
    n = len(real_times)
    n_ticks = max(3, int(n_ticks))
    locs = np.linspace(0, max(0, n - 1), n_ticks).astype(int)
    locs = np.unique(locs)
    ax.set_xticks(locs)
    labels = [real_times[i].strftime("%m-%d %H:%M") for i in locs]
    ax.set_xticklabels(labels, rotation=0, ha="center")


FIB_ALERT_TEXT = (
    "FIB ALERT: Fibonacci 0% = High and 100% = Low.\n"
    "Watch near 0% for SELL reversals and near 100% for BUY reversals (with confirmation)."
)
# ============================================================
# Batch 2: Parts 6/10 to 10/10
# ============================================================

# =========================
# Part 6/10 — Sessions (PST), Yahoo news, channel helpers
# =========================
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")


def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes

    start_d = idx[0].astimezone(session_tz).date()
    end_d = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")

    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))

        dt_open_pst = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)

        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)

    return opens, closes


def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close = session_markers_for_index(idx, NY_TZ, 8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}


def draw_session_lines(ax, lines: dict, alpha: float = 0.35):
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, alpha=alpha)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, alpha=alpha)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, alpha=alpha)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, alpha=alpha)

    handles = [
        Line2D([0], [0], linestyle="-", linewidth=1.6, label="London Open"),
        Line2D([0], [0], linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], linestyle="-", linewidth=1.6, label="New York Open"),
        Line2D([0], [0], linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels


@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_news(symbol: str, window_days: int = 7) -> pd.DataFrame:
    rows = []
    if yf is None:
        return pd.DataFrame()

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
        except Exception:
            try:
                dt_utc = pd.to_datetime(ts, utc=True)
            except Exception:
                continue

        dt_pst = dt_utc.tz_convert(PACIFIC)
        rows.append(
            {
                "time": dt_pst,
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")


def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, times: List[pd.Timestamp]) -> List[int]:
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return []
    out = []
    for t in times:
        try:
            # map to nearest bar index
            pos = int(np.searchsorted(real_times.values, pd.Timestamp(t).to_datetime64()))
            pos = max(0, min(pos, len(real_times) - 1))
            out.append(pos)
        except Exception:
            continue
    return out


def draw_news_markers(ax, positions: List[int], label="News"):
    for p in positions:
        try:
            ax.axvline(p, alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], alpha=0.5, linewidth=2, label=label)


def channel_state_series(price: pd.Series, sup: pd.Series, res: pd.Series, eps: float = 0.0) -> pd.Series:
    p = _coerce_1d_series(price).astype(float)
    s_sup = _coerce_1d_series(sup).astype(float).reindex(p.index)
    s_res = _coerce_1d_series(res).astype(float).reindex(p.index)
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
            ax.axvspan(a, b, alpha=0.15, zorder=1)
        except Exception:
            pass
    ax.plot([], [], linewidth=8, alpha=0.20, label="In Range (S↔R)")
    return state


# =========================
# Part 7/10 — Scanner helpers
# =========================
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


@st.cache_data(ttl=120)
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str):
    try:
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_show.empty or len(close_show) < 30:
            return None

        # uptrend check: global slope on displayed window
        x = np.arange(len(close_show), dtype=float)
        m, _ = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) <= 0:
            return None

        ntd_full = compute_normalized_trend(close_full, window=ntd_win)
        npx_full = compute_normalized_price(close_full, window=ntd_win)
        ntd_show = ntd_full.reindex(close_show.index)
        npx_show = npx_full.reindex(close_show.index)

        cross_up, _ = _cross_series((npx_show - 0.5) * 2.0, ntd_show)
        if not cross_up.any():
            return None

        t = cross_up[cross_up].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Signal": "NPX↑NTD (Uptrend)",
            "Bars Since": bars_since,
            "Cross Time": t,
            "Global Slope": float(m),
            "Current Price": float(close_show.iloc[-1]),
            "NTD (last)": float(ntd_show.dropna().iloc[-1]) if not ntd_show.dropna().empty else np.nan,
            "NPX (last)": float(npx_show.dropna().iloc[-1]) if not npx_show.dropna().empty else np.nan,
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
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_show.empty or len(close_show) < max(30, int(local_slope_lb) + 5):
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = npx_full.reindex(close_show.index)
        level = 0.5

        prev = npx_show.shift(1)
        direction = str(direction).lower()
        if direction.startswith("up"):
            cross_mask = (npx_show >= level) & (prev < level)
            sig_label = "NPX 0.5↑"
            slope_sign_ok = lambda m: m > 0
        else:
            cross_mask = (npx_show <= level) & (prev > level)
            sig_label = "NPX 0.5↓"
            slope_sign_ok = lambda m: m < 0

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

        seg = close_show.loc[:t].tail(int(local_slope_lb)).dropna()
        if len(seg) < 2:
            return None
        x = np.arange(len(seg), dtype=float)
        m, _ = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or not slope_sign_ok(float(m)):
            return None

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Daily View": daily_view_label,
            "Signal": sig_label,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Local Slope": float(m),
            "Current Price": float(close_show.iloc[-1]),
            "NPX@Cross": float(npx_show.loc[t]) if np.isfinite(npx_show.loc[t]) else np.nan,
            "NPX (last)": float(npx_show.dropna().iloc[-1]) if not npx_show.dropna().empty else np.nan,
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
                                 min_r2: float = 0.99):
    try:
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if len(close_show) < max(30, slope_lb, sr_lb, bb_window):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)) or float(r2) < float(min_r2):
            return None

        want_buy = str(side).upper().startswith("B")
        if want_buy and float(m) <= 0:
            return None
        if (not want_buy) and float(m) >= 0:
            return None

        res = close_show.rolling(int(sr_lb), min_periods=1).max()
        sup = close_show.rolling(int(sr_lb), min_periods=1).min()

        bb_mid, _, _, _, _ = compute_bbands(close_show, window=int(bb_window), mult=float(bb_sigma), use_ema=bool(bb_ema))
        cross_up, cross_dn = _cross_series(close_show, bb_mid)

        hz = max(1, int(horizon))
        if want_buy:
            if not cross_up.any():
                return None
            t_cross = cross_up[cross_up].index[-1]
            loc = int(close_show.index.get_loc(t_cross))
            j0 = max(0, loc - hz)
            touch_mask = close_show.iloc[j0:loc + 1] <= (sup.iloc[j0:loc + 1] * (1.0 + float(prox)))
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
            touch_mask = close_show.iloc[j0:loc + 1] >= (res.iloc[j0:loc + 1] * (1.0 - float(prox)))
            if not touch_mask.any():
                return None
            t_touch = touch_mask[touch_mask].index[-1]
            if not _n_consecutive_decreasing(close_show.loc[:t_cross], int(bars_confirm)):
                return None

        bars_since_cross = int((len(close_show) - 1) - int(close_show.index.get_loc(t_cross)))

        return {
            "Symbol": symbol,
            "Side": "BUY" if want_buy else "SELL",
            "Daily View": daily_view_label,
            "Bars Since Cross": bars_since_cross,
            "Touch Time": t_touch,
            "Cross Time": t_cross,
            "Slope": float(m),
            "R2": float(r2),
            "Support@Touch": float(sup.loc[t_touch]) if np.isfinite(sup.loc[t_touch]) else np.nan,
            "Resistance@Touch": float(res.loc[t_touch]) if np.isfinite(res.loc[t_touch]) else np.nan,
            "BB Mid@Cross": float(bb_mid.loc[t_cross]) if np.isfinite(bb_mid.loc[t_cross]) else np.nan,
            "Price@Cross": float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan,
            "Current Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
    except Exception:
        return None


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
        close_full = fetch_hist(symbol).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_show.empty or len(close_show) < 30:
            return None

        fibs = fibonacci_levels(close_show)
        if not fibs:
            return None

        hi = float(fibs.get("0%", np.nan))
        lo = float(fibs.get("100%", np.nan))
        if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
            return None
        rng = hi - lo
        thr = float(proximity_pct_of_range) * rng

        last_px = float(close_show.iloc[-1])
        dist0 = abs(last_px - hi)
        dist100 = abs(last_px - lo)

        near0 = dist0 <= thr
        near100 = dist100 <= thr
        if not (near0 or near100):
            return None

        yhat, up, low_band, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        rev_prob = slope_reversal_probability(
            close_show, m,
            hist_window=int(hist_window),
            slope_window=int(slope_window),
            horizon=int(horizon),
        )
        trig = fib_reversal_trigger_from_extremes(
            close_show,
            proximity_pct_of_range=float(proximity_pct_of_range),
            confirm_bars=int(confirm_bars),
            lookback_bars=int(lookback_bars_for_trigger),
        )

        near_level = "0%" if (near0 and dist0 <= dist100) else "100%"
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


# =========================
# Part 8/10 — Session state + FIX duplicate key for sidebar toggle
# =========================
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

if "show_global_trendline" not in st.session_state:
    st.session_state.show_global_trendline = True  # default ON


# =========================
# Part 9/10 — Sidebar controls + shared hourly renderer
# =========================
st.sidebar.header("Controls")

mode = st.sidebar.radio("Mode", ["Stocks", "Forex"], index=0, key="mode_select_v1")

DEFAULT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY", "QQQ"]
DEFAULT_FX = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]

universe_default = DEFAULT_STOCKS if mode == "Stocks" else DEFAULT_FX

custom_universe = st.sidebar.text_area(
    "Universe (comma-separated)",
    value=",".join(universe_default),
    height=80,
    key=f"universe_text_{mode}",
)
universe = [x.strip() for x in custom_universe.split(",") if x.strip()]

daily_view = st.sidebar.selectbox(
    "Daily view range",
    ["All", "5Y", "3Y", "2Y", "1Y", "6M", "3M"],
    index=3,
    key=f"daily_view_{mode}",
)

hist_years = st.sidebar.slider("Daily history (years)", 1, 20, int(st.session_state.hist_years), 1, key=f"hist_years_{mode}")
st.session_state.hist_years = int(hist_years)

st.sidebar.subheader("Overlays")
show_hma = st.sidebar.checkbox("Show HMA(55)", value=True, key=f"show_hma_{mode}")
hma_period = 55

show_macd = st.sidebar.checkbox("Show MACD panel", value=False, key=f"show_macd_{mode}")
show_psar = st.sidebar.checkbox("Show PSAR", value=False, key=f"show_psar_{mode}")
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.05, 0.02, 0.01, key=f"psar_step_{mode}")
psar_max = st.sidebar.slider("PSAR max", 0.10, 0.30, 0.20, 0.01, key=f"psar_max_{mode}")

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=False, key=f"show_ichi_{mode}")
ichi_conv = 9
ichi_base = 26
ichi_spanb = 52

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key=f"show_bbands_{mode}")
bb_win = st.sidebar.slider("BB window", 10, 60, 20, 1, key=f"bb_win_{mode}")
bb_mult = st.sidebar.slider("BB sigma", 1.0, 3.5, 2.0, 0.1, key=f"bb_mult_{mode}")
bb_use_ema = st.sidebar.checkbox("BB mid uses EMA", value=False, key=f"bb_ema_{mode}")

show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True, key=f"show_fibs_{mode}")

st.sidebar.subheader("S/R + Slope")
sr_lb_daily = st.sidebar.slider("Daily S/R lookback", 10, 200, 90, 5, key=f"sr_lb_daily_{mode}")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback", 10, 400, 120, 10, key=f"sr_lb_hourly_{mode}")
sr_prox_pct = st.sidebar.slider("S/R proximity %", 0.0, 0.05, 0.01, 0.0025, key=f"sr_prox_{mode}")

slope_lb_daily = st.sidebar.slider("Daily slope lookback", 20, 240, 90, 5, key=f"slope_lb_d_{mode}")
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback", 20, 400, 120, 10, key=f"slope_lb_h_{mode}")

st.sidebar.subheader("NTD / NPX")
show_ntd = st.sidebar.checkbox("Show NTD panel", value=True, key=f"show_ntd_{mode}")
shade_ntd = st.sidebar.checkbox("Shade NTD ±0.75", value=True, key=f"shade_ntd_{mode}")
show_npx_ntd = st.sidebar.checkbox("Overlay NPX on NTD", value=True, key=f"show_npx_{mode}")
mark_npx_cross = st.sidebar.checkbox("Mark NPX/NTD crosses", value=True, key=f"mark_npx_{mode}")
ntd_window = st.sidebar.slider("NTD/NPX window", 20, 200, 63, 1, key=f"ntd_win_{mode}")

st.sidebar.subheader("Reversal probability")
rev_hist_lb = st.sidebar.slider("Rev hist window", 200, 2000, 600, 50, key=f"rev_hist_{mode}")
rev_horizon = st.sidebar.slider("Rev horizon (bars)", 3, 30, 10, 1, key=f"rev_hz_{mode}")
rev_bars_confirm = st.sidebar.slider("Reversal confirm bars", 1, 5, 2, 1, key=f"rev_confirm_{mode}")

st.sidebar.subheader("Forex extras")
show_sessions_pst = st.sidebar.checkbox("Show London/NY session lines", value=(mode == "Forex"), key=f"show_sess_{mode}")
show_fx_news = st.sidebar.checkbox("Show FX news (Yahoo)", value=(mode == "Forex"), key=f"show_fx_news_{mode}")
news_window_days = st.sidebar.slider("News window (days)", 1, 30, 7, 1, key=f"news_win_{mode}")


# ---- FIX: unique key for the sidebar toggle button (prevents StreamlitDuplicateElementKey) ----
_toggle_lbl = "Hide global trendline" if st.session_state.show_global_trendline else "Show global trendline"
if st.sidebar.button(_toggle_lbl, use_container_width=True, key=f"sidebar_toggle_global_trendline_v2_{mode}"):
    st.session_state.show_global_trendline = not st.session_state.show_global_trendline
    try:
        st.experimental_rerun()
    except Exception:
        pass


def _global_slope_only(series_like) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    try:
        m, _ = np.polyfit(x, s.to_numpy(dtype=float), 1)
        return float(m)
    except Exception:
        return float("nan")


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

    # plot with RangeIndex but keep real_times for tick labels
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill()
    he = hc.ewm(span=20, adjust=False).mean()

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    hma_h = compute_hma(hc, period=hma_period)
    macd_h, macd_sig_h, macd_hist_h = compute_macd(hc)

    st_intraday = compute_supertrend(intraday, atr_period=10, atr_mult=3.0)
    st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday.columns else pd.Series(dtype=float)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {"High", "Low", "Close"}.issubset(intraday.columns) and show_ichi:
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
    rev_prob_h = slope_reversal_probability(hc, m_h, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon)

    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    fig2, (ax2, ax2w) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3.2, 1.3]}
    )
    plt.subplots_adjust(hspace=0.05, top=0.90, right=0.93, bottom=0.22)

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    # global trendline toggle
    if st.session_state.get("show_global_trendline", True):
        global_m_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")
    else:
        global_m_h = _global_slope_only(hc)

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1,
                 label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and (not psar_h_df.empty) and ("PSAR" in psar_h_df.columns):
        up_mask = psar_h_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, zorder=6)

    res_val = float(res_h.iloc[-1]) if np.isfinite(res_h.iloc[-1]) else np.nan
    sup_val = float(sup_h.iloc[-1]) if np.isfinite(sup_h.iloc[-1]) else np.nan
    px_val = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}")

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label="Supertrend")

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, m_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

        trig_h = find_slope_trigger_after_band_reversal(hc, yhat_h, upper_h, lower_h, horizon=rev_horizon)
        annotate_slope_trigger(ax2, trig_h)

    # news markers (convert to bar positions)
    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if pos:
            draw_news_markers(ax2, pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=m_h,
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
        macd_instr_txt = f"MACD/HMA55: {macd_sig['side']} @ {fmt_price_val(macd_sig['price'])}"
        annotate_macd_signal(ax2, macd_sig["time"], macd_sig["price"], macd_sig["side"])

    ax2.text(
        0.01, 0.98, macd_instr_txt,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        zorder=20
    )

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
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
        ax2.text(
            0.99, 0.02, f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
            transform=ax2.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7),
        )

    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    # NTD panel
    ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX (w={ntd_window})")
    ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
    npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

    if show_ntd and shade_ntd and not ntd_h.dropna().empty:
        shade_ntd_regions(ax2w, ntd_h)
    if show_ntd and not ntd_h.dropna().empty:
        ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
        overlay_ntd_triangles_by_trend(ax2w, ntd_h, trend_slope=m_h, upper=0.75, lower=-0.75)
        overlay_inrange_on_ntd(ax2w, price=hc, sup=sup_h, res=res_h)

    if show_npx_ntd and not npx_h.dropna().empty and not ntd_h.dropna().empty:
        overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross)

    ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
    ax2w.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
    ax2w.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
    ax2w.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
    ax2w.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
    ax2w.set_ylim(-1.2, 1.2)
    ax2w.legend(loc="lower left", framealpha=0.5, fontsize=9)

    ax2.legend(loc="lower left", framealpha=0.5, fontsize=9)

    # session lines (forex only) — map real_times to positions
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

    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax2w, real_times, n_ticks=8)
        _apply_compact_time_ticks(ax2, real_times, n_ticks=8)

    style_axes(ax2)
    style_axes(ax2w)
    st.pyplot(fig2)

    if show_macd and not macd_h.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)

    trig_raw = fib_reversal_trigger_from_extremes(
        hc,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_hourly)),
    )

    trig_disp = None
    if isinstance(trig_raw, dict):
        trig_disp = dict(trig_raw)
        if isinstance(real_times, pd.DatetimeIndex):
            # convert integer bar index? (we store timestamps already in this build)
            pass

    return {
        "trade_instruction": instr_txt,
        "fib_trigger": trig_disp,
    }


# =========================
# Part 10/10 — Tabs + app UI
# =========================
st.markdown(
    """
<style>
  div[data-baseweb="tab-list"] { flex-wrap: wrap; }
</style>
""",
    unsafe_allow_html=True,
)

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
    "Fib 0%/100% Reversal Watchlist",
])

# ---- TAB 1 ----
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data is cached ~2 minutes. Charts show the last RUN ticker until you run again.")

    sel = st.selectbox("Ticker:", universe, key=f"orig_ticker_{mode}")
    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key=f"orig_chart_{mode}_v2")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key=f"hour_range_select_{mode}",
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    run_clicked = st.button("Run Forecast", key=f"btn_run_forecast_{mode}")

    fib_instruction_box = st.empty()
    trade_instruction_box = st.empty()

    if run_clicked:
        df_hist = fetch_hist(sel, years=hist_years)
        df_ohlc = fetch_hist_ohlc(sel, years=hist_years)
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
            "mode_at_run": mode,
        })

    if st.session_state.get("run_all", False) and st.session_state.get("ticker") is not None and st.session_state.get("mode_at_run") == mode:
        disp_ticker = st.session_state.ticker
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"**Displayed (last run):** {disp_ticker} • Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

        with fib_instruction_box.container():
            st.warning(FIB_ALERT_TEXT)
            st.caption(
                "Confirmed Fib Trigger:\n"
                "• **BUY** when price touches near **100%** then prints consecutive higher closes\n"
                "• **SELL** when price touches near **0%** then prints consecutive lower closes"
            )

        daily_instr_txt = None
        hourly_instr_txt = None
        daily_fib_trig = None
        hourly_fib_trig = None

        # Daily chart
        if chart in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = df_show.ewm(span=30, adjust=False).mean()
            res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
            sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()

            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df_show, slope_lb_daily)
            rev_prob_d = slope_reversal_probability(df_show, m_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)
            piv = current_daily_pivots(df_ohlc)

            hma_d_show = compute_hma(df_show, period=hma_period)
            macd_d, macd_sig_d, _ = compute_macd(df_show)

            kijun_d_show = pd.Series(index=df_show.index, dtype=float)
            if show_ichi and df_ohlc is not None and not df_ohlc.empty and {"High","Low","Close"}.issubset(df_ohlc.columns):
                ohlc_show = df_ohlc.loc[df_show.index.min():df_show.index.max()].copy()
                _, kijun_d, _, _, _ = ichimoku_lines(ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"])
                kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()

            bb_mid_d_show, bb_up_d_show, bb_lo_d_show, bb_pctb_d_show, bb_nbb_d_show = compute_bbands(
                df_show, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
            )

            psar_d_df = pd.DataFrame()
            if show_psar and df_ohlc is not None and not df_ohlc.empty:
                ohlc_show = df_ohlc.loc[df_show.index.min():df_show.index.max()].copy()
                psar_d_df = compute_psar_from_ohlc(ohlc_show, step=psar_step, max_step=psar_max)

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]},
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{disp_ticker} Daily — {daily_view} — EMA, S/R (w={sr_lb_daily}), Slope, Pivots "
                f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
            )

            ax.plot(df_show.index, df_show.values, label="History")
            ax.plot(ema30_show.index, ema30_show.values, "--", label="30 EMA")

            if st.session_state.get("show_global_trendline", True):
                global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            else:
                global_m_d = _global_slope_only(df_show)

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
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=15, zorder=6,
                               label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=15, zorder=6)

            res_val_d = float(res_d_show.iloc[-1]) if np.isfinite(res_d_show.iloc[-1]) else np.nan
            sup_val_d = float(sup_d_show.iloc[-1]) if np.isfinite(sup_d_show.iloc[-1]) else np.nan

            if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6,
                          label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6,
                          label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}")
                label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}")

            if not yhat_d.empty:
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2, label=f"Daily slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d.empty and not lower_d.empty:
                ax.plot(upper_d.index, upper_d.values, "--", linewidth=2.2, alpha=0.85, label="Daily +2σ")
                ax.plot(lower_d.index, lower_d.values, "--", linewidth=2.2, alpha=0.85, label="Daily -2σ")

                bounce_sig_d = find_band_bounce_signal(df_show, upper_d, lower_d, m_d)
                if bounce_sig_d is not None:
                    annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

                trig_d = find_slope_trigger_after_band_reversal(df_show, yhat_d, upper_d, lower_d, horizon=rev_horizon)
                annotate_slope_trigger(ax, trig_d)

            # pivots
            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            # fibs on daily
            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

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
                ax.text(
                    0.99, 0.02, f"Current price: {fmt_price_val(last_px_show)}{nbb_txt}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7),
                )

            macd_sig_d2 = find_macd_hma_sr_signal(
                close=df_show, hma=hma_d_show, macd=macd_d, sup=sup_d_show, res=res_d_show,
                global_trend_slope=global_m_d, prox=sr_prox_pct
            )
            macd_txt_d = "MACD/HMA55: n/a"
            if macd_sig_d2 is not None and np.isfinite(macd_sig_d2.get("price", np.nan)):
                macd_txt_d = f"MACD/HMA55: {macd_sig_d2['side']} @ {fmt_price_val(macd_sig_d2['price'])}"
                annotate_macd_signal(ax, macd_sig_d2["time"], macd_sig_d2["price"], macd_sig_d2["side"])

            ax.text(
                0.01, 0.98, macd_txt_d,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
                zorder=30
            )

            ax.legend(loc="lower left", framealpha=0.5, fontsize=9)

            # daily indicator panel
            axdw.set_title(f"Daily Indicator Panel — NTD + NPX (w={ntd_window})")
            ntd_d = compute_normalized_trend(df_show, window=ntd_window) if show_ntd else pd.Series(index=df_show.index, dtype=float)
            npx_d = compute_normalized_price(df_show, window=ntd_window) if show_npx_ntd else pd.Series(index=df_show.index, dtype=float)

            if show_ntd and shade_ntd and not ntd_d.dropna().empty:
                shade_ntd_regions(axdw, ntd_d)
            if show_ntd and not ntd_d.dropna().empty:
                axdw.plot(ntd_d.index, ntd_d.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                overlay_ntd_triangles_by_trend(axdw, ntd_d, trend_slope=m_d, upper=0.75, lower=-0.75)
                overlay_inrange_on_ntd(axdw, price=df_show, sup=sup_d_show, res=res_d_show)

            if show_npx_ntd and not npx_d.dropna().empty and not ntd_d.dropna().empty:
                overlay_npx_on_ntd(axdw, npx_d, ntd_d, mark_crosses=mark_npx_cross)

            axdw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
            axdw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
            axdw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
            axdw.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
            axdw.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
            axdw.set_ylim(-1.2, 1.2)
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

            daily_instr_txt = format_trade_instruction(
                trend_slope=m_d, buy_val=sup_val_d, sell_val=res_val_d, close_val=last_px_show,
                symbol=disp_ticker, global_trend_slope=global_m_d
            )
            daily_fib_trig = fib_reversal_trigger_from_extremes(
                df_show,
                proximity_pct_of_range=0.02,
                confirm_bars=int(rev_bars_confirm),
                lookback_bars=int(max(60, slope_lb_daily)),
            )

        # Hourly chart
        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            out_h = render_hourly_views(
                sel=disp_ticker,
                intraday=intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.hour_range,
                is_forex=(mode == "Forex"),
            )
            if isinstance(out_h, dict):
                hourly_instr_txt = out_h.get("trade_instruction")
                hourly_fib_trig = out_h.get("fib_trigger")

        # Instructions BELOW the Run Forecast button
        with trade_instruction_box.container():
            if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
                st.success(f"Daily: {daily_instr_txt}")
            if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
                st.success(f"Hourly: {hourly_instr_txt}")

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
            fx_news = fetch_yf_news(disp_ticker, window_days=news_window_days)
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time", "publisher", "title", "link"]].reset_index(drop=True), use_container_width=True)

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower": st.session_state.fc_ci.iloc[:, 0] if st.session_state.fc_ci.shape[1] > 0 else np.nan,
            "Upper": st.session_state.fc_ci.iloc[:, 1] if st.session_state.fc_ci.shape[1] > 1 else np.nan,
        }, index=st.session_state.fc_idx))

    else:
        st.info("Click **Run Forecast** to display charts and forecast.")


# ---- TAB 2 ----
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

        st.caption(f"Displayed ticker: **{st.session_state.ticker}** • Intraday lookback: **{st.session_state.get('hour_range','24h')}**")
        view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}")

        if view in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)
            res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
            sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()
            hma_d_show = compute_hma(df_show, period=hma_period)
            macd_d, macd_sig_d, _ = compute_macd(df_show)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")

            if st.session_state.get("show_global_trendline", True):
                global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            else:
                global_m_d = _global_slope_only(df_show)

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if not res_d_show.empty and not sup_d_show.empty:
                ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6, label="Support")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            macd_sig = find_macd_hma_sr_signal(df_show, hma_d_show, macd_d, sup_d_show, res_d_show, global_m_d, prox=sr_prox_pct)
            macd_txt = "MACD/HMA55: n/a"
            if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
                macd_txt = f"MACD/HMA55: {macd_sig['side']} @ {fmt_price_val(macd_sig['price'])}"
                annotate_macd_signal(ax, macd_sig["time"], macd_sig["price"], macd_sig["side"])
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
                hour_range_label=st.session_state.get("hour_range", "24h"),
                is_forex=(mode == "Forex"),
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower": ci.iloc[:, 0] if ci.shape[1] > 0 else np.nan,
            "Upper": ci.iloc[:, 1] if ci.shape[1] > 1 else np.nan,
        }, index=idx))


# ---- TAB 3 ----
with tab3:
    st.header("Bull vs Bear")
    sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")

    dfp = pd.DataFrame()
    if yf is not None:
        try:
            dfp = yf.download(sel_bb, period="6mo", interval="1d", auto_adjust=True, progress=False)[["Close"]].dropna()
            dfp = _ensure_pacific_df(dfp)
        except Exception:
            dfp = pd.DataFrame()

    if dfp.empty:
        st.warning("No data available.")
    else:
        s = dfp["Close"].astype(float)
        ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if len(s) > 1 else np.nan
        st.metric(label=f"{sel_bb} return (6mo)", value=fmt_pct(ret))
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title(f"{sel_bb} — 6mo Close")
        ax.plot(s.index, s.values, label="Close")
        if st.session_state.get("show_global_trendline", True):
            draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)


# ---- TAB 4 ----
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
            f"P(slope reverses ≤ {rev_horizon} bars)": fmt_pct(slope_reversal_probability(df, m_d, rev_hist_lb, slope_lb_daily, rev_horizon)),
        })

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({
                "Hourly slope (reg band)": fmt_slope(m_h),
                "Hourly R²": fmt_r2(r2_h),
                f"P(slope reverses ≤ {rev_horizon} bars) hourly": fmt_pct(slope_reversal_probability(hc, m_h, rev_hist_lb, slope_lb_hourly, rev_horizon)),
            })


# ---- TAB 5 ----
with tab5:
    st.header("NTD -0.75 Scanner")
    scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
    run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

    if run_scan:
        rows = []
        if scan_frame.startswith("Hourly"):
            for sym in universe:
                val, ts = last_hourly_ntd_value(sym, ntd_window, period="1d")
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


# ---- TAB 6 ----
with tab6:
    st.header("Long-Term History")
    sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    smax = fetch_hist_max(sel_lt)

    if smax is None or smax.dropna().empty:
        st.warning("No long-term history available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        if st.session_state.get("show_global_trendline", True):
            draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)


# ---- TAB 7 ----
with tab7:
    st.header("Recent BUY Scanner — Daily NPX↑NTD in Uptrend")
    max_bars = st.slider("Max bars since NPX↑NTD cross", 0, 20, 2, 1, key=f"buy_scan_npx_max_bars_{mode}")
    run_buy_scan = st.button("Run Recent BUY Scan", key=f"btn_run_recent_buy_scan_npx_{mode}")

    if run_buy_scan:
        rows = []
        for sym in universe:
            r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view)
            if r is not None and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---- TAB 8 ----
with tab8:
    st.header("NPX 0.5-Cross Scanner — Local Slope Confirmed (Daily)")
    c1, c2, c3 = st.columns(3)
    max_bars0 = c1.slider("Max bars since NPX 0.5-cross", 0, 30, 2, 1, key=f"npx0_max_bars_{mode}")
    eps0 = c2.slider("Max |NPX-0.5| at cross", 0.01, 0.30, 0.08, 0.01, key=f"npx0_eps_{mode}")
    lb_local = c3.slider("Local slope lookback (bars)", 10, 360, int(slope_lb_daily), 10, key=f"npx0_slope_lb_{mode}")

    run0 = st.button("Run NPX 0.5-Cross Scan", key=f"btn_run_npx0_scan_{mode}")

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
            st.dataframe(pd.DataFrame(rows_up).reset_index(drop=True), use_container_width=True) if rows_up else st.info("No matches.")
        with right:
            st.subheader("NPX 0.5↓ with Local DOWN Slope")
            st.dataframe(pd.DataFrame(rows_dn).reset_index(drop=True), use_container_width=True) if rows_dn else st.info("No matches.")


# ---- TAB 9 ----
with tab9:
    st.header("Daily Slope + S/R Reversal + BB Midline Scanner (R² ≥ threshold)")
    c1, c2 = st.columns(2)
    max_bars_since = c1.slider("Max bars since BB mid cross", 0, 60, 10, 1, key=f"srbb_max_bars_since_{mode}")
    r2_thr = c2.slider("Min R² (confidence)", 0.80, 0.99, 0.99, 0.01, key=f"srbb_r2_thr_{mode}")
    run_scan = st.button("Run Daily Slope+BB Scan", key=f"btn_run_daily_slope_bb_scan_{mode}")

    if run_scan:
        buy_rows, sell_rows = [], []
        for sym in universe:
            rb = last_daily_sr_reversal_bbmid(
                symbol=sym, daily_view_label=daily_view,
                slope_lb=slope_lb_daily, sr_lb=sr_lb_daily,
                bb_window=bb_win, bb_sigma=bb_mult, bb_ema=bb_use_ema,
                prox=sr_prox_pct, bars_confirm=rev_bars_confirm, horizon=rev_horizon,
                side="BUY", min_r2=float(r2_thr),
            )
            if rb is not None and int(rb.get("Bars Since Cross", 9999)) <= int(max_bars_since):
                buy_rows.append(rb)

            rs = last_daily_sr_reversal_bbmid(
                symbol=sym, daily_view_label=daily_view,
                slope_lb=slope_lb_daily, sr_lb=sr_lb_daily,
                bb_window=bb_win, bb_sigma=bb_mult, bb_ema=bb_use_ema,
                prox=sr_prox_pct, bars_confirm=rev_bars_confirm, horizon=rev_horizon,
                side="SELL", min_r2=float(r2_thr),
            )
            if rs is not None and int(rs.get("Bars Since Cross", 9999)) <= int(max_bars_since):
                sell_rows.append(rs)

        left, right = st.columns(2)
        with left:
            st.subheader("BUY — Up Slope + Support Reversal + BB Mid Cross")
            st.dataframe(pd.DataFrame(buy_rows).reset_index(drop=True), use_container_width=True) if buy_rows else st.info("No matches.")
        with right:
            st.subheader("SELL — Down Slope + Resistance Reversal + BB Mid Cross")
            st.dataframe(pd.DataFrame(sell_rows).reset_index(drop=True), use_container_width=True) if sell_rows else st.info("No matches.")


# ---- TAB 10 ----
with tab10:
    st.header("Fib 0%/100% Reversal Watchlist")
    c1, c2, c3 = st.columns(3)
    prox_pct = c1.slider("Proximity (as % of Fib range)", 0.005, 0.08, 0.02, 0.005, key=f"fibwatch_prox_{mode}")
    min_rev = c2.slider(f"Min P(slope rev≤{rev_horizon} bars)", 0.00, 0.95, 0.25, 0.05, key=f"fibwatch_minrev_{mode}")
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
