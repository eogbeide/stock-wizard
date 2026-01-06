# ============================================================
# bullbear.py  — COMPLETE UPDATED CODE (BATCH 1/2: Parts 1–5)
# Changes in this update (per your (a) and (b)):
#   - Tab 10 Fib Extremes Scanner now supports Hourly (intraday) AND Daily
#   - Tab 10 results auto-sort "closest to extreme first"
# UI look & feel preserved (same tab layout; minimal, consistent controls).
# ============================================================

# =========================
# Part 1/10 — Imports & Setup
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
from matplotlib.lines import Line2D

from statsmodels.tsa.statespace.sarimax import SARIMAX


PACIFIC = pytz.timezone("America/Los_Angeles")

st.set_page_config(page_title="BullBear", layout="wide")


# =========================
# Part 2/10 — Formatting & Small Helpers
# =========================
def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].copy()
        raise ValueError("Expected a 1D series-like object.")
    return pd.Series(x).copy()

def _safe_last_float(s: pd.Series) -> float:
    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return np.nan
    try:
        return float(s.iloc[-1])
    except Exception:
        return np.nan

def fmt_price_val(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    ax = abs(float(x))
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:,.4f}"
    return f"{x:,.6f}"

def fmt_pct(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"

def fmt_slope(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{float(x):+.6f}"

def fmt_r2(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{float(x):.4f}"

def style_axes(ax):
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_alpha(0.25)
    ax.spines["right"].set_alpha(0.25)
    ax.tick_params(axis="x", labelrotation=0)

def label_on_left(ax, y: float, text: str, color="black"):
    try:
        xmin, xmax = ax.get_xlim()
        ax.text(xmin, y, f" {text}", va="center", ha="left",
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color, alpha=0.75))
    except Exception:
        pass

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    try:
        if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
            return
        n = len(real_times)
        if n <= 2:
            return
        xs = np.linspace(0, n - 1, num=min(n_ticks, n), dtype=int)
        xs = np.unique(xs)
        ax.set_xticks(xs)
        labels = []
        for i in xs:
            t = real_times[int(i)]
            if t.tzinfo is None:
                t = pytz.UTC.localize(t).astimezone(PACIFIC)
            else:
                t = t.astimezone(PACIFIC)
            labels.append(t.strftime("%m-%d %H:%M"))
        ax.set_xticklabels(labels, fontsize=9)
    except Exception:
        pass

def _cross_series(a: pd.Series, b: pd.Series):
    a = _coerce_1d_series(a)
    b = _coerce_1d_series(b)
    idx = a.index.union(b.index)
    a = a.reindex(idx)
    b = b.reindex(idx)
    prev = (a - b).shift(1)
    now = (a - b)
    cross_up = (now >= 0) & (prev < 0)
    cross_dn = (now <= 0) & (prev > 0)
    return cross_up.fillna(False), cross_dn.fillna(False)


# =========================
# Part 3/10 — Data Fetching (Cached)
# =========================
@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist(symbol: str, years: int = 10) -> pd.Series:
    try:
        end = datetime.now(tz=PACIFIC).date()
        start = (datetime.now(tz=PACIFIC) - timedelta(days=int(365 * years))).date()
        df = yf.download(symbol, start=str(start), end=str(end), interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].astype(float).dropna()
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
            s.index = s.index.tz_localize("UTC").tz_convert(PACIFIC)
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, years: int = 10) -> pd.DataFrame:
    try:
        end = datetime.now(tz=PACIFIC).date()
        start = (datetime.now(tz=PACIFIC) - timedelta(days=int(365 * years))).date()
        df = yf.download(symbol, start=str(start), end=str(end), interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        need = {"Open", "High", "Low", "Close"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        out = df[list(need)].copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(how="all")
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None:
            out.index = out.index.tz_localize("UTC").tz_convert(PACIFIC)
        return out
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_hist_max(symbol: str) -> pd.Series:
    try:
        df = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty or "Close" not in df.columns:
            return pd.Series(dtype=float)
        s = df["Close"].astype(float).dropna()
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
            s.index = s.index.tz_localize("UTC").tz_convert(PACIFIC)
        return s
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
            else:
                df.index = df.index.tz_convert(PACIFIC)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# =========================
# Part 4/10 — Indicators & Core Math
# =========================
def compute_sarimax_forecast(series: pd.Series, steps: int = 30):
    s = _coerce_1d_series(series).dropna()
    if len(s) < 60:
        # fallback: naive forward
        idx = pd.date_range(start=(s.index[-1] + pd.Timedelta(days=1)).date(), periods=steps, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1])] * steps, index=idx)
        ci = pd.DataFrame({"lower": vals * 0.98, "upper": vals * 1.02}, index=idx)
        return idx, vals, ci

    try:
        y = s.astype(float)
        model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps)
        fc_mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        # coerce index to PST daily
        start = (y.index[-1] + pd.Timedelta(days=1)).date()
        idx = pd.date_range(start=start, periods=steps, freq="D", tz=PACIFIC)
        vals = pd.Series(fc_mean.to_numpy(dtype=float), index=idx)
        if ci.shape[1] >= 2:
            ci_out = pd.DataFrame({ "Lower": ci.iloc[:, 0].to_numpy(dtype=float), "Upper": ci.iloc[:, 1].to_numpy(dtype=float) }, index=idx)
        else:
            ci_out = pd.DataFrame({ "Lower": vals * 0.98, "Upper": vals * 1.02 }, index=idx)
        return idx, vals, ci_out
    except Exception:
        idx = pd.date_range(start=(s.index[-1] + pd.Timedelta(days=1)).date(), periods=steps, freq="D", tz=PACIFIC)
        vals = pd.Series([float(s.iloc[-1])] * steps, index=idx)
        ci = pd.DataFrame({"Lower": vals * 0.98, "Upper": vals * 1.02}, index=idx)
        return idx, vals, ci

def compute_hma(series: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)

    def wma(x: pd.Series, n: int) -> pd.Series:
        if n <= 1:
            return x
        w = np.arange(1, n + 1, dtype=float)
        return x.rolling(n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)

    n = int(period)
    half = max(1, n // 2)
    root = max(1, int(round(math.sqrt(n))))
    return wma(2 * wma(s, half) - wma(s, n), root)

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(series).astype(float)
    if s.empty:
        z = pd.Series(index=s.index, dtype=float)
        return z, z, z
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bbands(series: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(series).astype(float)
    if s.empty:
        z = pd.Series(index=s.index, dtype=float)
        return z, z, z, z, z
    w = int(window)
    if use_ema:
        mid = s.ewm(span=w, adjust=False).mean()
        var = (s - mid).pow(2).ewm(span=w, adjust=False).mean()
        std = np.sqrt(var)
    else:
        mid = s.rolling(w).mean()
        std = s.rolling(w).std(ddof=0)

    up = mid + float(mult) * std
    lo = mid - float(mult) * std

    pctb = (s - lo) / (up - lo)
    nbb = (s - mid) / (float(mult) * std.replace(0.0, np.nan))
    return mid, up, lo, pctb, nbb

def regression_with_band(series: pd.Series, lookback: int = 120):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty or len(s) < 5:
        z = pd.Series(index=_coerce_1d_series(series).index, dtype=float)
        return z, z, z, np.nan, np.nan

    n = len(s)
    lb = int(min(max(5, lookback), n))
    y = s.iloc[-lb:].to_numpy(dtype=float)
    x = np.arange(lb, dtype=float)

    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        z = pd.Series(index=s.index, dtype=float)
        return z, z, z, np.nan, np.nan

    yhat_lb = m * x + b
    resid = y - yhat_lb
    sigma = float(np.nanstd(resid)) if np.isfinite(np.nanstd(resid)) else 0.0

    # extend line across entire s index using positions
    x_all = np.arange(n, dtype=float)
    # anchor intercept so that last lb segment aligns:
    # yhat_all = m*(x_all-(n-lb)) + (b)  => aligns segment start
    yhat_all = m * (x_all - (n - lb)) + b

    yhat = pd.Series(yhat_all, index=s.index, dtype=float)
    upper = yhat + 2.0 * sigma
    lower = yhat - 2.0 * sigma

    # R² on last lb window
    ss_res = float(np.nansum(resid**2))
    ss_tot = float(np.nansum((y - np.nanmean(y))**2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # reindex to original (keep full)
    full_idx = _coerce_1d_series(series).index
    return yhat.reindex(full_idx), upper.reindex(full_idx), lower.reindex(full_idx), float(m), float(r2)

def draw_trend_direction_line(ax, series: pd.Series, label_prefix: str = "Trend (global)"):
    s = _coerce_1d_series(series).astype(float).dropna()
    if len(s) < 5:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    y = m * x + b
    ax.plot(s.index, y, "-", linewidth=1.7, label=f"{label_prefix} {fmt_slope(m)}/bar")
    return float(m)

def annotate_crossover(ax, t, y, side: str):
    if t is None or not np.isfinite(y):
        return
    txt = "BUY" if str(side).upper().startswith("B") else "SELL"
    color = "tab:green" if txt == "BUY" else "tab:red"
    try:
        ax.scatter([t], [y], marker="o", s=70, color=color, zorder=10)
        ax.text(t, y, f" {txt}", color=color, fontsize=10, fontweight="bold", va="center")
    except Exception:
        pass

def find_band_bounce_signal(price: pd.Series, upper: pd.Series, lower: pd.Series, slope: float):
    p = _coerce_1d_series(price).astype(float)
    up = _coerce_1d_series(upper).astype(float).reindex(p.index)
    lo = _coerce_1d_series(lower).astype(float).reindex(p.index)
    if p.dropna().empty or up.dropna().empty or lo.dropna().empty or not np.isfinite(slope):
        return None

    # simple recent bounce:
    # uptrend: touch lower then move up
    # downtrend: touch upper then move down
    if float(slope) > 0:
        touch = p <= lo
        if touch.any():
            t0 = touch[touch].index[-1]
            # confirm move away
            if t0 in p.index:
                loc = p.index.get_loc(t0)
                if isinstance(loc, slice):
                    return None
                if loc < len(p) - 1:
                    if float(p.iloc[-1]) > float(p.iloc[loc]):
                        return {"time": t0, "price": float(p.loc[t0]), "side": "BUY"}
    elif float(slope) < 0:
        touch = p >= up
        if touch.any():
            t0 = touch[touch].index[-1]
            if t0 in p.index:
                loc = p.index.get_loc(t0)
                if isinstance(loc, slice):
                    return None
                if loc < len(p) - 1:
                    if float(p.iloc[-1]) < float(p.iloc[loc]):
                        return {"time": t0, "price": float(p.loc[t0]), "side": "SELL"}
    return None

def slope_line(series: pd.Series, lookback: int = 120):
    s = _coerce_1d_series(series).astype(float).dropna()
    if len(s) < 5:
        z = pd.Series(index=_coerce_1d_series(series).index, dtype=float)
        return z, np.nan
    lb = int(min(max(5, lookback), len(s)))
    seg = s.iloc[-lb:]
    x = np.arange(len(seg), dtype=float)
    m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
    yhat = m * x + b
    out = pd.Series(index=seg.index, data=yhat, dtype=float)
    return out.reindex(_coerce_1d_series(series).index), float(m)

def slope_reversal_probability(series: pd.Series,
                              current_slope: float,
                              hist_window: int = 240,
                              slope_window: int = 120,
                              horizon: int = 10):
    """
    Heuristic probability: among historical points, how often slope sign flips within `horizon` bars
    after being in similar slope quantile. Returns [0,1].
    """
    s = _coerce_1d_series(series).astype(float).dropna()
    if len(s) < max(50, hist_window, slope_window + horizon + 5) or not np.isfinite(current_slope):
        return np.nan

    w = int(slope_window)
    h = int(horizon)
    hw = int(hist_window)

    # compute rolling slopes (simple polyfit) over window w
    slopes = pd.Series(index=s.index, dtype=float)
    arr = s.to_numpy(dtype=float)
    for i in range(w, len(arr)):
        y = arr[i - w:i]
        x = np.arange(w, dtype=float)
        m, _ = np.polyfit(x, y, 1)
        slopes.iloc[i] = m

    slopes = slopes.dropna()
    if slopes.empty:
        return np.nan

    # focus last hist_window
    slopes = slopes.iloc[-min(hw, len(slopes)):]
    cur = float(current_slope)
    if not np.isfinite(cur):
        return np.nan

    # pick "similar" by quantile bin
    qlo, qhi = np.nanquantile(slopes.to_numpy(dtype=float), [0.45, 0.55])
    if cur < qlo:
        mask = slopes <= qlo
    elif cur > qhi:
        mask = slopes >= qhi
    else:
        mask = (slopes >= qlo) & (slopes <= qhi)

    idxs = np.where(mask.to_numpy(dtype=bool))[0]
    if len(idxs) < 8:
        return np.nan

    # count sign flip within horizon
    flip = 0
    total = 0
    sgn = np.sign(slopes.to_numpy(dtype=float))
    for j in idxs:
        if j + h >= len(sgn):
            continue
        total += 1
        now = sgn[j]
        fut = sgn[j + 1:j + h + 1]
        if now == 0:
            continue
        if np.any(fut == -now):
            flip += 1
    return float(flip / total) if total > 0 else np.nan

def compute_normalized_trend(series: pd.Series, window: int = 120) -> pd.Series:
    """
    NTD: normalized slope of price over rolling window, squashed into [-1,1] by tanh.
    """
    s = _coerce_1d_series(series).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    w = int(max(5, window))
    out = pd.Series(index=s.index, dtype=float)
    arr = s.to_numpy(dtype=float)

    for i in range(w, len(arr) + 1):
        seg = arr[i - w:i]
        if np.all(~np.isfinite(seg)):
            continue
        x = np.arange(w, dtype=float)
        m, _ = np.polyfit(x, seg, 1)
        # normalize by average price scale in segment
        scale = float(np.nanmean(np.abs(seg))) if np.isfinite(np.nanmean(np.abs(seg))) else 1.0
        if scale <= 0:
            scale = 1.0
        val = (m * w) / scale  # approximate % move over window
        out.iloc[i - 1] = np.tanh(val)
    return out

def compute_normalized_price(series: pd.Series, window: int = 120) -> pd.Series:
    """
    NPX: normalized price position within rolling min/max => [-1,1]
    """
    s = _coerce_1d_series(series).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    w = int(max(5, window))
    roll = s.rolling(w, min_periods=1)
    lo = roll.min()
    hi = roll.max()
    denom = (hi - lo).replace(0.0, np.nan)
    n01 = (s - lo) / denom
    return (2.0 * n01 - 1.0)

def fibonacci_levels(series: pd.Series):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty:
        return {}
    hi = float(np.nanmax(s.to_numpy(dtype=float)))
    lo = float(np.nanmin(s.to_numpy(dtype=float)))
    if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
        return {}
    rng = hi - lo
    # 0% at high, 100% at low
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * rng,
        "38.2%": hi - 0.382 * rng,
        "50%": hi - 0.500 * rng,
        "61.8%": hi - 0.618 * rng,
        "78.8%": hi - 0.788 * rng,
        "100%": lo
    }

def subset_by_daily_view(series: pd.Series, daily_view_label: str) -> pd.Series:
    s = _coerce_1d_series(series)
    if s.dropna().empty:
        return s
    label = str(daily_view_label)
    if label == "1Y":
        return s.iloc[-252:]
    if label == "2Y":
        return s.iloc[-504:]
    if label == "5Y":
        return s.iloc[-1260:]
    if label == "10Y":
        return s  # already max by session setting
    if label == "MAX":
        return s
    # default
    return s.iloc[-504:]


# =========================
# Part 5/10 — NEW/UPDATED: Fib Extremes Row Helper (Daily + Hourly)
# =========================
def fib_extremes_row(symbol: str,
                     daily_view_label: str,
                     near_pct: float,
                     touch_lookback: int,
                     frame: str = "Daily",
                     hour_range_label: str = "24h",
                     intraday_interval: str = "5m"):
    """
    Returns a dict row for Tab 10.
    - Daily: uses subset_by_daily_view over fetch_hist
    - Hourly: uses fetch_intraday(period=map[hour_range_label]) over intraday Close
    near_pct: fraction of range (e.g., 0.05 means 5% of (high-low))
    touch_lookback: bars lookback for touch detection
    """

    frame_s = str(frame)
    if frame_s.startswith("Hourly"):
        period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
        period = period_map.get(str(hour_range_label), "1d")
        df = fetch_intraday(symbol, period=period, interval=intraday_interval)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = _coerce_1d_series(df["Close"]).astype(float).dropna()
        if close.empty:
            return None
        view_label = f"Hourly ({hour_range_label})"
    else:
        close_full = _coerce_1d_series(fetch_hist(symbol, years=int(st.session_state.get("hist_years", 10)))).dropna()
        if close_full.empty:
            return None
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).astype(float).dropna()
        if close.empty:
            return None
        view_label = str(daily_view_label)

    fibs = fibonacci_levels(close)
    if not fibs:
        return None

    hi = float(fibs["0%"])
    lo = float(fibs["100%"])
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return None

    last_close = float(close.iloc[-1])
    if not np.isfinite(last_close):
        return None

    # threshold distance in price units (as % of range)
    thr = float(near_pct) * rng

    dist_to_high = float(abs(hi - last_close))
    dist_to_low = float(abs(last_close - lo))

    near0 = dist_to_high <= thr
    near100 = dist_to_low <= thr

    # last touch within touch_lookback bars
    lb = max(1, int(touch_lookback))
    tail = close.tail(lb)
    if tail.empty:
        return None

    touch0_mask = (abs(tail - hi) <= thr)
    touch100_mask = (abs(tail - lo) <= thr)

    last_touch_0 = touch0_mask[touch0_mask].index[-1] if touch0_mask.any() else None
    last_touch_100 = touch100_mask[touch100_mask].index[-1] if touch100_mask.any() else None

    # confirmation rules (as you specified):
    # Near 0% confirms if moved to/under 23.6% after last touch of 0%
    # Near 100% confirms if moved to/over 78.8% after last touch of 100%
    confirm_0 = False
    confirm_100 = False

    lvl_236 = float(fibs["23.6%"])
    lvl_788 = float(fibs["78.8%"])

    if last_touch_0 is not None and last_touch_0 in close.index:
        after = close.loc[last_touch_0:]
        if not after.empty:
            confirm_0 = bool((after <= lvl_236).any())

    if last_touch_100 is not None and last_touch_100 in close.index:
        after = close.loc[last_touch_100:]
        if not after.empty:
            confirm_100 = bool((after >= lvl_788).any())

    # % distances for sorting/display
    dist_to_high_pct = (dist_to_high / rng) * 100.0
    dist_to_low_pct = (dist_to_low / rng) * 100.0

    return {
        "Symbol": symbol,
        "Daily View": view_label,  # keep original column name to preserve UI layout
        "Last Close": float(last_close),

        "Near 0%": bool(near0),
        "Confirm 0%→23.6%": bool(confirm_0),
        "Fib 23.6%": float(lvl_236),

        "Near 100%": bool(near100),
        "Confirm 100%→78.8%": bool(confirm_100),
        "Fib 78.8%": float(lvl_788),

        "High": float(hi),
        "Low": float(lo),

        "Last Touch 0% Time": last_touch_0,
        "Last Touch 100% Time": last_touch_100,

        "DistToHigh%": float(dist_to_high_pct),
        "DistToLow%": float(dist_to_low_pct),
    }
# ============================================================
# bullbear.py — COMPLETE UPDATED CODE (BATCH 2/2: Parts 6–10)
# ============================================================

# =========================
# Part 6/10 — Ichimoku, Supertrend, PSAR, NTD overlays
# =========================
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


# =========================
# Part 7/10 — Sessions, News, In-range shading, Cached last values
# =========================
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

@st.cache_data(ttl=120)
def last_daily_ntd_value(symbol: str, ntd_win: int):
    try:
        s = fetch_hist(symbol, years=int(st.session_state.get("hist_years", 10)))
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
# Part 8/10 — Scanners (recent buy, NPX 0.5-cross, SR+BB reversal)
# =========================
@st.cache_data(ttl=120)
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str):
    try:
        s_full = fetch_hist(symbol, years=int(st.session_state.get("hist_years", 10)))
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
        s_full = fetch_hist(symbol, years=int(st.session_state.get("hist_years", 10)))
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        # UPDATED (prior request): use 0.5 cross level instead of 0.0
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
                                 min_r2: float = 0.99):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol, years=int(st.session_state.get("hist_years", 10)))).dropna()
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
            seg = close_show.loc[:t_cross]
            if not _n_consecutive_increasing(seg, int(bars_confirm)):
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
            seg = close_show.loc[:t_cross]
            if not _n_consecutive_decreasing(seg, int(bars_confirm)):
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
# Part 9/10 — App Controls + Shared Hourly Renderer
# =========================
# Session state init
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

st.sidebar.title("Controls")

mode = st.sidebar.selectbox("Mode", ["Stocks", "Forex"], index=0)
hist_years = st.sidebar.slider("Daily history years", 2, 20, int(st.session_state.get("hist_years", 10)), 1)
st.session_state.hist_years = int(hist_years)

daily_view = st.sidebar.selectbox("Daily View", ["1Y", "2Y", "5Y", "10Y", "MAX"], index=2)

st.sidebar.subheader("Indicators")
show_hma = st.sidebar.checkbox("Show HMA", True)
hma_period = st.sidebar.slider("HMA period", 10, 200, 55, 1)

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", True)
bb_win = st.sidebar.slider("BB window", 5, 200, 20, 1)
bb_mult = st.sidebar.slider("BB sigma", 0.5, 4.0, 2.0, 0.1)
bb_use_ema = st.sidebar.checkbox("BB mid uses EMA", False)

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", False)
ichi_conv = st.sidebar.slider("Ichimoku conv", 5, 20, 9, 1)
ichi_base = st.sidebar.slider("Ichimoku base", 10, 60, 26, 1)
ichi_spanb = st.sidebar.slider("Ichimoku spanB", 20, 120, 52, 1)

show_psar = st.sidebar.checkbox("Show PSAR", False)
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.10, 0.02, 0.01)
psar_max = st.sidebar.slider("PSAR max", 0.10, 0.50, 0.20, 0.01)

show_ntd = st.sidebar.checkbox("Show NTD panel", True)
shade_ntd = st.sidebar.checkbox("Shade NTD regimes", True)
show_npx_ntd = st.sidebar.checkbox("Show NPX on NTD panel", True)
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses", True)
show_ntd_channel = st.sidebar.checkbox("Shade in-range (S↔R)", False)
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA reversal on NTD", True)
hma_rev_lb = st.sidebar.slider("HMA reversal slope lookback", 2, 10, 3, 1)

ntd_window = st.sidebar.slider("NTD/NPX window", 20, 400, 120, 5)

st.sidebar.subheader("Trend & S/R")
sr_lb_daily = st.sidebar.slider("Daily S/R lookback", 10, 400, 120, 5)
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback", 10, 600, 180, 10)
sr_prox_pct = st.sidebar.slider("S/R proximity %", 0.05, 2.00, 0.25, 0.05) / 100.0

slope_lb_daily = st.sidebar.slider("Daily slope lookback", 20, 400, 120, 5)
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback", 20, 600, 180, 10)

rev_hist_lb = st.sidebar.slider("Reversal prob history window", 60, 600, 240, 10)
rev_horizon = st.sidebar.slider("Reversal horizon (bars)", 3, 30, 10, 1)
rev_bars_confirm = st.sidebar.slider("Reversal confirm bars", 1, 5, 2, 1)

st.sidebar.subheader("Supertrend")
atr_period = st.sidebar.slider("ATR period", 3, 30, 10, 1)
atr_mult = st.sidebar.slider("ATR mult", 1.0, 6.0, 3.0, 0.1)

show_macd = st.sidebar.checkbox("Show MACD panel (optional)", False)

show_fibs = st.sidebar.checkbox("Show Fibonacci levels", True)

st.sidebar.subheader("Forex extras")
show_sessions_pst = st.sidebar.checkbox("Show London/NY sessions (intraday)", True)
show_fx_news = st.sidebar.checkbox("Show Yahoo Finance news markers (intraday)", False)
news_window_days = st.sidebar.slider("News window (days)", 1, 30, 7, 1)

st.sidebar.subheader("Bull/Bear")
bb_period = st.sidebar.selectbox("Bull/Bear lookback", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

if mode == "Stocks":
    universe = st.sidebar.text_area(
        "Universe (comma-separated tickers)",
        "SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA"
    )
else:
    universe = st.sidebar.text_area(
        "Universe (comma-separated tickers)",
        "EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X"
    )
universe = [x.strip() for x in universe.split(",") if x.strip()]

def format_trade_instruction(trend_slope: float, buy_val: float, sell_val: float, close_val: float, symbol: str, global_trend_slope: float = np.nan):
    if not np.isfinite(trend_slope) or not np.isfinite(buy_val) or not np.isfinite(sell_val) or not np.isfinite(close_val):
        return "Instruction: n/a"
    prox = float(sr_prox_pct)
    near_buy = close_val <= buy_val * (1.0 + prox)
    near_sell = close_val >= sell_val * (1.0 - prox)

    if trend_slope > 0 and near_buy:
        return f"BUY zone near Support ({fmt_price_val(buy_val)})"
    if trend_slope < 0 and near_sell:
        return f"SELL zone near Resistance ({fmt_price_val(sell_val)})"
    if (trend_slope > 0) and (close_val < buy_val):
        return f"ALERT: Below Support ({fmt_price_val(buy_val)}) in uptrend?"
    if (trend_slope < 0) and (close_val > sell_val):
        return f"ALERT: Above Resistance ({fmt_price_val(sell_val)}) in downtrend?"
    return "Hold / watch"

def find_macd_hma_sr_signal(close: pd.Series,
                           hma: pd.Series,
                           macd: pd.Series,
                           sup: pd.Series,
                           res: pd.Series,
                           global_trend_slope: float,
                           prox: float = 0.0025):
    c = _coerce_1d_series(close).astype(float).dropna()
    if len(c) < 5:
        return None
    h = _coerce_1d_series(hma).reindex(c.index)
    m = _coerce_1d_series(macd).reindex(c.index)
    s_sup = _coerce_1d_series(sup).reindex(c.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(c.index).ffill().bfill()

    # buy: macd crosses up 0 + price near support + price above HMA
    macd_up, macd_dn = _cross_series(m, pd.Series(0.0, index=m.index))
    near_support = c <= s_sup * (1.0 + prox)
    near_resist  = c >= s_res * (1.0 - prox)

    if macd_up.any():
        t = macd_up[macd_up].index[-1]
        if (t in c.index) and bool(near_support.loc[t]) and (c.loc[t] >= h.loc[t] if np.isfinite(h.loc[t]) else True):
            if np.isfinite(global_trend_slope) and global_trend_slope < 0:
                return None
            return {"time": t, "price": float(c.loc[t]), "side": "BUY"}

    if macd_dn.any():
        t = macd_dn[macd_dn].index[-1]
        if (t in c.index) and bool(near_resist.loc[t]) and (c.loc[t] <= h.loc[t] if np.isfinite(h.loc[t]) else True):
            if np.isfinite(global_trend_slope) and global_trend_slope > 0:
                return None
            return {"time": t, "price": float(c.loc[t]), "side": "SELL"}

    return None

def annotate_macd_signal(ax, t, y, side: str):
    if t is None or not np.isfinite(y):
        return
    col = "tab:green" if str(side).upper().startswith("B") else "tab:red"
    try:
        ax.scatter([t], [y], marker="P", s=140, color=col, zorder=11)
        ax.text(t, y, f" {side}", color=col, fontsize=10, fontweight="bold", va="center")
    except Exception:
        pass

def shade_ntd_regions(ax, ntd: pd.Series, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd).dropna()
    if s.empty:
        return
    try:
        ax.fill_between(s.index, lower, upper, alpha=0.06)
        ax.fill_between(s.index, upper, 1.1, alpha=0.06)
        ax.fill_between(s.index, -1.1, lower, alpha=0.06)
    except Exception:
        pass

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, times_list):
    """
    Map datetime stamps onto integer bar positions for plots that use RangeIndex.
    """
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return []
    if not times_list:
        return []
    rt = real_times
    out = []
    for t in times_list:
        try:
            if t.tzinfo is None:
                t = pytz.UTC.localize(t).astimezone(PACIFIC)
            else:
                t = t.astimezone(PACIFIC)
            # nearest index
            j = int(np.argmin(np.abs((rt - t).to_numpy(dtype="timedelta64[ns]").astype("int64"))))
            out.append(j)
        except Exception:
            continue
    return out

def annotate_slope_trigger(ax, trig: dict):
    if not trig:
        return
    try:
        t = trig.get("time")
        y = trig.get("price")
        side = trig.get("side", "")
        annotate_crossover(ax, t, y, side)
    except Exception:
        pass

def find_slope_trigger_after_band_reversal(price: pd.Series, trend_line: pd.Series, upper: pd.Series, lower: pd.Series, horizon: int = 10):
    p = _coerce_1d_series(price).astype(float).dropna()
    if len(p) < 10:
        return None
    # heuristic: if last touch band within horizon and then crosses trend line in direction
    tl = _coerce_1d_series(trend_line).reindex(p.index)
    up = _coerce_1d_series(upper).reindex(p.index)
    lo = _coerce_1d_series(lower).reindex(p.index)

    hz = max(2, int(horizon))
    tail = p.tail(hz)
    if tail.empty:
        return None
    # buy trigger: touched lower then crosses above trend line
    touched_lo = (tail <= lo.reindex(tail.index)).any()
    crossed_up, crossed_dn = _cross_series(p, tl)
    if touched_lo and crossed_up.any():
        t = crossed_up[crossed_up].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY"}

    touched_up = (tail >= up.reindex(tail.index)).any()
    if touched_up and crossed_dn.any():
        t = crossed_dn[crossed_dn].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL"}
    return None

def current_daily_pivots(df_ohlc: pd.DataFrame):
    if df_ohlc is None or df_ohlc.empty or not {"High","Low","Close"}.issubset(df_ohlc.columns):
        return {}
    last = df_ohlc.dropna().iloc[-1]
    H = float(last["High"])
    L = float(last["Low"])
    C = float(last["Close"])
    if not np.all(np.isfinite([H, L, C])):
        return {}
    P = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

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
    if show_ntd:
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

    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

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
# Part 10/10 — Tabs (1..10)  — includes UPDATED Tab 10 (Hourly + Sorting)
# =========================
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
    "Fib 0%/100% Extremes Scanner"
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
    alert_box = st.empty()

    if run_clicked:
        df_hist = fetch_hist(sel, years=int(st.session_state.get("hist_years", 10)))
        df_ohlc = fetch_hist_ohlc(sel, years=int(st.session_state.get("hist_years", 10)))
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

            # Supertrend on Daily by default (same params as Hourly)
            st_daily = compute_supertrend(df_ohlc, atr_period=atr_period, atr_mult=atr_mult) if (df_ohlc is not None and not df_ohlc.empty) else pd.DataFrame()
            st_line_d = st_daily["ST"] if (not st_daily.empty and "ST" in st_daily.columns) else pd.Series(dtype=float)

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
            st_line_d_show = st_line_d.reindex(df_show.index) if not st_line_d.empty else st_line_d

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

            # show Daily Supertrend by default
            if not st_line_d_show.empty:
                ax.plot(st_line_d_show.index, st_line_d_show.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

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

            # Fibonacci on Daily by default
            if show_fibs and not df_show.empty:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="dotted", linewidth=1)
                    x1 = df_show.index[-1]
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
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
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
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if not res_d_show.empty and not sup_d_show.empty:
                ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")

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
# TAB 10: Fib 0%/100% Extremes Scanner (UPDATED: Hourly + Sorting)
# ---------------------------
with tab10:
    st.header("Fib 0%/100% Extremes Scanner")
    st.caption(
        "Scans the current universe for symbols whose **latest close** is near the Fib **0% (high)** or **100% (low)** "
        "within the selected Daily view range.\n\n"
        "Reversal confirmation:\n"
        "• Near **0%** (high) confirms if price has moved to/under **23.6%**\n"
        "• Near **100%** (low) confirms if price has moved to/over **78.8%**"
    )

    # UPDATED (THIS REQUEST): add Hourly support while keeping UI style consistent
    frame = st.radio("Frame:", ["Daily", "Hourly (intraday)"], index=0, key="fibext_frame")

    c1, c2, c3 = st.columns(3)
    near_thr_pct = c1.slider("Near extreme threshold (%)", 1.0, 10.0, 5.0, 0.5, key="fibext_near_thr")  # default 5%
    touch_lb = c2.slider("Touch lookback (bars)", 10, 240, 60, 10, key="fibext_touch_lb")
    run_fib_scan = c3.button("Run Fib Extremes Scan", key="btn_run_fib_extremes_scan")

    hour_range_local = st.session_state.get("hour_range", "24h")
    if frame.startswith("Hourly"):
        hour_range_local = st.selectbox(
            "Hourly lookback:",
            ["24h", "48h", "96h"],
            index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
            key="fibext_hour_range"
        )

    if run_fib_scan:
        rows0, rows100 = [], []
        near_frac = float(near_thr_pct) / 100.0

        for sym in universe:
            r = fib_extremes_row(
                sym,
                daily_view_label=daily_view,
                near_pct=near_frac,
                touch_lookback=int(touch_lb),
                frame=frame,
                hour_range_label=hour_range_local,
                intraday_interval="5m",
            )
            if not r:
                continue

            last_touch_0 = r.get("Last Touch 0% Time", None)
            last_touch_100 = r.get("Last Touch 100% Time", None)

            # Require a touch within lookback for "confirmation" to count
            r["Confirm 0%→23.6%"] = bool((last_touch_0 is not None) and bool(r.get("Confirm 0%→23.6%", False)))
            r["Confirm 100%→78.8%"] = bool((last_touch_100 is not None) and bool(r.get("Confirm 100%→78.8%", False)))

            if bool(r.get("Near 0%", False)):
                rows0.append(r)
            if bool(r.get("Near 100%", False)):
                rows100.append(r)

        left, right = st.columns(2)

        show_cols = [
            "Symbol", "Daily View", "Last Close",
            "Near 0%", "Confirm 0%→23.6%", "Fib 23.6%",
            "Near 100%", "Confirm 100%→78.8%", "Fib 78.8%",
            "High", "Low",
            "Last Touch 0% Time", "Last Touch 100% Time",
            "DistToHigh%", "DistToLow%"
        ]

        with left:
            st.subheader("Near Fib 0% (High)")
            if not rows0:
                st.info("No matches.")
            else:
                out0 = pd.DataFrame(rows0)
                for c in show_cols:
                    if c not in out0.columns:
                        out0[c] = np.nan
                out0 = out0[show_cols]

                # UPDATED (THIS REQUEST): closest to extreme first (0% => DistToHigh%)
                if "DistToHigh%" in out0.columns:
                    out0["DistToHigh%"] = pd.to_numeric(out0["DistToHigh%"], errors="coerce")
                    out0 = out0.sort_values(["DistToHigh%"], ascending=[True])

                st.dataframe(out0.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Near Fib 100% (Low)")
            if not rows100:
                st.info("No matches.")
            else:
                out1 = pd.DataFrame(rows100)
                for c in show_cols:
                    if c not in out1.columns:
                        out1[c] = np.nan
                out1 = out1[show_cols]

                # UPDATED (THIS REQUEST): closest to extreme first (100% => DistToLow%)
                if "DistToLow%" in out1.columns:
                    out1["DistToLow%"] = pd.to_numeric(out1["DistToLow%"], errors="coerce")
                    out1 = out1.sort_values(["DistToLow%"], ascending=[True])

                st.dataframe(out1.reset_index(drop=True), use_container_width=True)
