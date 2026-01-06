# app.py
# Streamlit Trading Dashboard — Stocks + Forex
# ✅ Includes: 10 tabs (with scanners), Fibonacci levels + Confirmation Trigger
# ✅ UPDATED per request:
#    Fib confirmation trigger NOW ALSO REQUIRES:
#      • BUY: price crosses back ABOVE the 23.6% Fib level
#      • SELL: price crosses back BELOW the 76.4% Fib level (mirror of 23.6 from the top)

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import pytz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import streamlit as st
import yfinance as yf

# Optional SARIMAX forecasting
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_SARIMAX = True
except Exception:
    _HAS_SARIMAX = False


# =========================
# Config
# =========================
st.set_page_config(page_title="Trend + Fib Dashboard", layout="wide")

PACIFIC = pytz.timezone("America/Los_Angeles")
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

FIB_CONFIRM_R2 = 0.999  # 99.9% confidence trigger threshold


# =========================
# Formatting helpers
# =========================
def fmt_pct(x, digits=1):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100.0*float(x):.{digits}f}%"
    except Exception:
        return "n/a"

def fmt_price_val(x):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        x = float(x)
        # show more decimals for FX
        if abs(x) < 10:
            return f"{x:.5f}"
        return f"{x:,.2f}"
    except Exception:
        return "n/a"

def fmt_slope(m):
    try:
        if m is None or not np.isfinite(m):
            return "n/a"
        return f"{float(m):+.6g}"
    except Exception:
        return "n/a"

def fmt_r2(r2):
    try:
        if r2 is None or not np.isfinite(r2):
            return "n/a"
        return f"{float(r2):.4f}"
    except Exception:
        return "n/a"

def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame) and x.shape[1] >= 1:
        return x.iloc[:, 0].copy()
    try:
        return pd.Series(x).copy()
    except Exception:
        return pd.Series(dtype=float)

def _safe_last_float(s: pd.Series):
    try:
        s = _coerce_1d_series(s).dropna()
        if s.empty:
            return np.nan
        v = float(s.iloc[-1])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


# =========================
# Plot styling
# =========================
def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def label_on_left(ax, y, txt, color="black"):
    try:
        ax.text(0.01, y, txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8))
    except Exception:
        pass


# =========================
# Data fetching
# =========================
def _ensure_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist(symbol: str, years: int = 10) -> pd.Series:
    # daily close for last N years
    period = f"{int(years)}y"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = _ensure_tz_index(df)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].astype(float).dropna()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, years: int = 10) -> pd.DataFrame:
    period = f"{int(years)}y"
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _ensure_tz_index(df)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].copy().dropna(how="all")

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_max(symbol: str) -> pd.Series:
    df = yf.download(symbol, period="max", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = _ensure_tz_index(df)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].astype(float).dropna()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d") -> pd.DataFrame:
    # 60m bars for intraday lookback (yfinance supports 60m)
    df = yf.download(symbol, period=period, interval="60m", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _ensure_tz_index(df)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].copy().dropna(how="all")


# =========================
# Technical indicators
# =========================
def compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)

    def wma(x, n):
        x = _coerce_1d_series(x).astype(float)
        if x.empty:
            return pd.Series(index=x.index, dtype=float)
        w = np.arange(1, n + 1, dtype=float)
        return x.rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)

    half = max(1, period // 2)
    sqrtp = max(1, int(math.sqrt(period)))
    w1 = wma(s, half)
    w2 = wma(s, period)
    raw = 2.0 * w1 - w2
    return wma(raw, sqrtp)

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        idx = s.index
        return (pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float))
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        idx = s.index
        return (pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float))
    if use_ema:
        mid = s.ewm(span=window, adjust=False).mean()
        var = (s - mid).pow(2).ewm(span=window, adjust=False).mean()
        sd = np.sqrt(var)
    else:
        mid = s.rolling(window).mean()
        sd = s.rolling(window).std()
    up = mid + mult * sd
    lo = mid - mult * sd

    rng = (up - lo).replace(0.0, np.nan)
    pctb = (s - lo) / rng
    # "normalized BB width" (relative width)
    nbb = (up - lo) / mid.replace(0.0, np.nan)
    return mid, up, lo, pctb, nbb

def regression_with_band(series: pd.Series, lookback: int = 200):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty or len(s) < 3:
        idx = _coerce_1d_series(series).index
        return (pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                pd.Series(index=idx, dtype=float),
                np.nan, np.nan)

    lb = int(min(max(3, lookback), len(s)))
    seg = s.iloc[-lb:]
    x = np.arange(lb, dtype=float)
    y = seg.to_numpy(dtype=float)

    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b

    # R^2
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    resid = y - yhat
    sd = float(np.nanstd(resid)) if np.isfinite(np.nanstd(resid)) else np.nan

    idx_full = _coerce_1d_series(series).index
    yhat_s = pd.Series(index=idx_full, dtype=float)
    up_s   = pd.Series(index=idx_full, dtype=float)
    lo_s   = pd.Series(index=idx_full, dtype=float)

    seg_idx = seg.index
    yhat_s.loc[seg_idx] = yhat
    if np.isfinite(sd):
        up_s.loc[seg_idx] = yhat + 2.0 * sd
        lo_s.loc[seg_idx] = yhat - 2.0 * sd

    return yhat_s, up_s, lo_s, float(m), float(r2) if np.isfinite(r2) else np.nan

def current_daily_pivots(df_ohlc: pd.DataFrame):
    if df_ohlc is None or df_ohlc.empty or not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        return {}
    last = df_ohlc.dropna().iloc[-1]
    H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
    P = (H + L + C) / 3.0
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

def compute_normalized_price(close: pd.Series, window: int = 200) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll_min = s.rolling(window, min_periods=1).min()
    roll_max = s.rolling(window, min_periods=1).max()
    denom = (roll_max - roll_min).replace(0.0, np.nan)
    npx = (s - roll_min) / denom
    return npx.clip(0.0, 1.0)

def compute_normalized_trend(close: pd.Series, window: int = 200) -> pd.Series:
    # maps "distance from rolling mid" into [-1, +1]
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll_min = s.rolling(window, min_periods=1).min()
    roll_max = s.rolling(window, min_periods=1).max()
    mid = (roll_max + roll_min) / 2.0
    half = ((roll_max - roll_min) / 2.0).replace(0.0, np.nan)
    ntd = (s - mid) / half
    return ntd.clip(-1.0, 1.0)

def slope_line(series: pd.Series, lookback: int = 200):
    yhat, _, _, m, _ = regression_with_band(series, lookback=lookback)
    return yhat, m

def draw_trend_direction_line(ax, series: pd.Series, label_prefix="Trend (global)"):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty or len(s) < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    y = m * x + b
    ax.plot(s.index, y, linewidth=2.0, alpha=0.8, label=f"{label_prefix}: {fmt_slope(m)}/bar")
    return float(m)

def shade_ntd_regions(ax, ntd: pd.Series):
    s = _coerce_1d_series(ntd).astype(float)
    if s.dropna().empty:
        return
    # light shading of extremes
    ax.fill_between(s.index, 0.75, 1.10, alpha=0.08)
    ax.fill_between(s.index, -1.10, -0.75, alpha=0.08)


# =========================
# Cross helpers & signals
# =========================
def _cross_series(a: pd.Series, b: pd.Series):
    a = _coerce_1d_series(a)
    b = _coerce_1d_series(b).reindex(a.index)
    up = (a >= b) & (a.shift(1) < b.shift(1))
    dn = (a <= b) & (a.shift(1) > b.shift(1))
    return up.fillna(False), dn.fillna(False)

def annotate_crossover(ax, t, y, side: str):
    if t is None or y is None or not np.isfinite(y):
        return
    side_u = str(side).upper()
    if "BUY" in side_u:
        ax.scatter([t], [y], marker="^", s=130, zorder=10)
        ax.text(t, y, " BUY", va="center", fontsize=9, fontweight="bold")
    else:
        ax.scatter([t], [y], marker="v", s=130, zorder=10)
        ax.text(t, y, " SELL", va="center", fontsize=9, fontweight="bold")

def annotate_slope_trigger(ax, trig):
    if trig is None:
        return
    t = trig.get("time")
    y = trig.get("price")
    side = trig.get("side", "")
    if t is None or y is None or not np.isfinite(y):
        return
    ax.scatter([t], [y], marker="P", s=140, zorder=11)
    ax.text(t, y, f" SLOPE TRIG {side}", va="center", fontsize=8, fontweight="bold")

def find_band_bounce_signal(close: pd.Series, upper: pd.Series, lower: pd.Series, slope: float):
    s = _coerce_1d_series(close).astype(float)
    up = _coerce_1d_series(upper).reindex(s.index)
    lo = _coerce_1d_series(lower).reindex(s.index)
    if s.dropna().empty or up.dropna().empty or lo.dropna().empty:
        return None

    # simple "touch then rebound" signal on last bar
    # BUY if uptrend slope>0 and last bar crossed back above lower band
    # SELL if downtrend slope<0 and last bar crossed back below upper band
    i = s.index[-1]
    prev = s.shift(1)
    if np.isfinite(slope) and slope > 0:
        cond = (s >= lo) & (prev < lo.shift(1))
        if cond.fillna(False).any():
            t = cond[cond].index[-1]
            return {"time": t, "price": float(s.loc[t]), "side": "BUY"}
    if np.isfinite(slope) and slope < 0:
        cond = (s <= up) & (prev > up.shift(1))
        if cond.fillna(False).any():
            t = cond[cond].index[-1]
            return {"time": t, "price": float(s.loc[t]), "side": "SELL"}
    return None

def find_slope_trigger_after_band_reversal(close: pd.Series, yhat: pd.Series, upper: pd.Series, lower: pd.Series, horizon: int = 10):
    s = _coerce_1d_series(close).astype(float).dropna()
    if s.empty:
        return None
    yh = _coerce_1d_series(yhat).reindex(s.index)
    if yh.dropna().empty:
        return None

    hz = max(1, int(horizon))
    seg = s.iloc[-(hz+2):]
    seg_y = yh.reindex(seg.index)

    up, dn = _cross_series(seg, seg_y)
    if up.any():
        t = up[up].index[-1]
        return {"time": t, "price": float(seg.loc[t]), "side": "BUY"}
    if dn.any():
        t = dn[dn].index[-1]
        return {"time": t, "price": float(seg.loc[t]), "side": "SELL"}
    return None

def find_macd_hma_sr_signal(close: pd.Series, hma: pd.Series, macd: pd.Series, sup: pd.Series, res: pd.Series,
                           global_trend_slope: float, prox: float = 0.0025):
    s = _coerce_1d_series(close).astype(float).dropna()
    if s.empty:
        return None
    h = _coerce_1d_series(hma).reindex(s.index)
    m = _coerce_1d_series(macd).reindex(s.index)
    su = _coerce_1d_series(sup).reindex(s.index).ffill()
    re = _coerce_1d_series(res).reindex(s.index).ffill()

    t = s.index[-1]
    c0 = float(s.iloc[-1])
    if not np.isfinite(c0):
        return None
    S0 = float(su.iloc[-1]) if len(su) else np.nan
    R0 = float(re.iloc[-1]) if len(re) else np.nan

    near_support = np.isfinite(S0) and (c0 <= S0 * (1.0 + float(prox)))
    near_resist  = np.isfinite(R0) and (c0 >= R0 * (1.0 - float(prox)))

    # MACD direction
    macd_dir = np.sign(m.diff().iloc[-1]) if len(m.dropna()) >= 2 else 0.0
    above_hma = np.isfinite(float(h.iloc[-1])) and (c0 >= float(h.iloc[-1]))

    if np.isfinite(global_trend_slope) and global_trend_slope > 0 and near_support and macd_dir >= 0 and above_hma:
        return {"time": t, "price": c0, "side": "BUY"}
    if np.isfinite(global_trend_slope) and global_trend_slope < 0 and near_resist and macd_dir <= 0 and (not above_hma):
        return {"time": t, "price": c0, "side": "SELL"}
    return None

def annotate_macd_signal(ax, t, y, side: str):
    if t is None or y is None or not np.isfinite(y):
        return
    if str(side).upper().startswith("B"):
        ax.scatter([t], [y], marker="o", s=110, zorder=12)
        ax.text(t, y, " MACD/HMA BUY", va="center", fontsize=8, fontweight="bold")
    else:
        ax.scatter([t], [y], marker="x", s=150, zorder=12)
        ax.text(t, y, " MACD/HMA SELL", va="center", fontsize=8, fontweight="bold")


# =========================
# Slope reversal probability (simple empirical)
# =========================
def slope_reversal_probability(series: pd.Series, current_slope: float, hist_window: int = 800, slope_window: int = 200, horizon: int = 10):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty or len(s) < max(10, slope_window + horizon + 5):
        return np.nan

    hw = int(min(hist_window, len(s)))
    sw = int(max(3, slope_window))
    hz = int(max(1, horizon))

    s = s.iloc[-hw:]
    slopes = []
    idxs = []
    for i in range(sw, len(s) - hz):
        seg = s.iloc[i - sw:i]
        x = np.arange(len(seg), dtype=float)
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        slopes.append(float(m))
        idxs.append(seg.index[-1])

    if len(slopes) < 20 or not np.isfinite(current_slope):
        return np.nan

    slopes = np.array(slopes, dtype=float)
    curr_sign = np.sign(current_slope)
    if curr_sign == 0:
        return np.nan

    # reversal if sign flips within next hz bars
    reversals = 0
    total = 0
    for j in range(len(slopes) - 1):
        m0 = slopes[j]
        sign0 = np.sign(m0)
        if sign0 == 0:
            continue
        total += 1
        # approximate: treat next slope estimates as "future" for reversal probability
        k1 = min(len(slopes), j + max(2, hz // 2))
        future = slopes[j+1:k1]
        if future.size and np.any(np.sign(future) == -sign0):
            reversals += 1

    if total <= 0:
        return np.nan
    return float(reversals / total)


# =========================
# Fibonacci utilities + UPDATED confirmation trigger
# =========================
def fibonacci_levels(series: pd.Series):
    s = _coerce_1d_series(series).astype(float).dropna()
    if s.empty:
        return {}
    lo = float(np.nanmin(s.to_numpy()))
    hi = float(np.nanmax(s.to_numpy()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return {}

    def lvl(p):  # p in [0,1]
        return lo + p * (hi - lo)

    fibs = {
        "0%": lo,
        "23.6%": lvl(0.236),
        "38.2%": lvl(0.382),
        "50%": lvl(0.5),
        "61.8%": lvl(0.618),
        "76.4%": lvl(0.764),   # mirror of 23.6 from the top
        "78.6%": lvl(0.786),
        "100%": hi,
    }
    return fibs

def fib_position_percent(price: float, fibs: dict):
    try:
        if fibs is None or not fibs:
            return np.nan
        lo = float(fibs.get("0%", np.nan))
        hi = float(fibs.get("100%", np.nan))
        if not np.isfinite(price) or not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return np.nan
        pct = 100.0 * (float(price) - lo) / (hi - lo)
        return float(np.clip(pct, 0.0, 100.0))
    except Exception:
        return np.nan

def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
        return False
    deltas = np.diff(s.iloc[-(n+1):].to_numpy(dtype=float))
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
        return False
    deltas = np.diff(s.iloc[-(n+1):].to_numpy(dtype=float))
    return bool(np.all(deltas < 0))

def find_fib_confirmation_trigger(price: pd.Series,
                                  fibs: dict,
                                  r2: float,
                                  horizon: int = 10,
                                  prox: float = 0.0025,
                                  bars_confirm: int = 2,
                                  r2_thr: float = FIB_CONFIRM_R2):
    """
    UPDATED (per request):
      BUY trigger requires:
        1) R² >= r2_thr
        2) price touched near 0% within 'horizon' bars BEFORE trigger
        3) price crosses UP through 23.6% level (extra confirmation)
        4) last 'bars_confirm' bars into trigger are increasing

      SELL trigger requires:
        1) R² >= r2_thr
        2) price touched near 100% within 'horizon' bars BEFORE trigger
        3) price crosses DOWN through 76.4% level (mirror of 23.6 from top)
        4) last 'bars_confirm' bars into trigger are decreasing
    """
    s = _coerce_1d_series(price).astype(float).dropna()
    if s.empty or not fibs:
        return None
    if not (np.isfinite(r2) and float(r2) >= float(r2_thr)):
        return None

    lo = float(fibs.get("0%", np.nan))
    hi = float(fibs.get("100%", np.nan))
    lvl23 = float(fibs.get("23.6%", np.nan))
    lvl764 = float(fibs.get("76.4%", np.nan))
    if not np.all(np.isfinite([lo, hi, lvl23, lvl764])):
        return None

    hz = max(1, int(horizon))

    # proximity check uses relative-to-price; keep consistent with SR prox usage
    near_low = s <= lo * (1.0 + float(prox))
    near_hi  = s >= hi * (1.0 - float(prox))

    # BUY: cross back above 23.6%
    prev = s.shift(1)
    cross_up_23 = (s >= lvl23) & (prev < lvl23)

    # SELL: cross back below 76.4% (mirror threshold)
    cross_dn_764 = (s <= lvl764) & (prev > lvl764)

    # Find latest BUY trigger
    if cross_up_23.any():
        t_cross = cross_up_23[cross_up_23].index[-1]
        # confirm a near-low touch occurred within horizon bars prior to t_cross
        loc = int(s.index.get_loc(t_cross))
        j0 = max(0, loc - hz)
        touched = near_low.iloc[j0:loc+1]
        if touched.any():
            t_touch = touched[touched].index[-1]
            seg = s.loc[:t_cross]
            if _n_consecutive_increasing(seg, int(bars_confirm)):
                return {
                    "side": "BUY",
                    "time": t_cross,
                    "price": float(s.loc[t_cross]),
                    "touch_time": t_touch,
                    "touch_price": float(s.loc[t_touch]),
                    "level": lvl23,
                    "level_label": "23.6%",
                    "r2": float(r2),
                }

    # Find latest SELL trigger
    if cross_dn_764.any():
        t_cross = cross_dn_764[cross_dn_764].index[-1]
        loc = int(s.index.get_loc(t_cross))
        j0 = max(0, loc - hz)
        touched = near_hi.iloc[j0:loc+1]
        if touched.any():
            t_touch = touched[touched].index[-1]
            seg = s.loc[:t_cross]
            if _n_consecutive_decreasing(seg, int(bars_confirm)):
                return {
                    "side": "SELL",
                    "time": t_cross,
                    "price": float(s.loc[t_cross]),
                    "touch_time": t_touch,
                    "touch_price": float(s.loc[t_touch]),
                    "level": lvl764,
                    "level_label": "76.4%",
                    "r2": float(r2),
                }

    return None

def annotate_fib_confirmation_trigger(ax, trig):
    if trig is None:
        return
    t = trig.get("time")
    y = trig.get("price")
    side = str(trig.get("side", "")).upper()
    lvl_lbl = trig.get("level_label", "")
    if t is None or y is None or not np.isfinite(y):
        return

    ax.scatter([t], [y], marker="*", s=220, zorder=15)
    ax.text(t, y, f" FIB CONF {side} ({lvl_lbl})", va="center",
            fontsize=9, fontweight="bold")


# =========================
# Ichimoku, Supertrend, PSAR
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


# =========================
# Sessions + News
# =========================
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

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, times):
    if real_times is None or not isinstance(real_times, pd.DatetimeIndex) or len(real_times) == 0:
        return []
    out = []
    rt = real_times
    for t in times:
        try:
            if isinstance(t, (pd.Timestamp, datetime)):
                tt = pd.Timestamp(t)
                if tt.tz is None:
                    tt = tt.tz_localize(PACIFIC)
                # nearest
                pos = int(np.searchsorted(rt.values, tt.value))
                pos = max(0, min(len(rt)-1, pos))
                out.append(pos)
        except Exception:
            pass
    return out

def draw_session_lines(ax, lines: dict, alpha: float = 0.35):
    for x in lines.get("ldn_open", []):
        ax.axvline(x, linestyle="-", linewidth=1.0, alpha=alpha)
    for x in lines.get("ldn_close", []):
        ax.axvline(x, linestyle="--", linewidth=1.0, alpha=alpha)
    for x in lines.get("ny_open", []):
        ax.axvline(x, linestyle="-", linewidth=1.0, alpha=alpha)
    for x in lines.get("ny_close", []):
        ax.axvline(x, linestyle="--", linewidth=1.0, alpha=alpha)

    handles = [
        Line2D([0], [0], linestyle="-",  linewidth=1.6, label="London Open"),
        Line2D([0], [0], linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], linestyle="-",  linewidth=1.6, label="New York Open"),
        Line2D([0], [0], linestyle="--", linewidth=1.6, label="New York Close"),
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

def draw_news_markers(ax, positions, label="News"):
    for x in positions:
        try:
            ax.axvline(x, alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], alpha=0.5, linewidth=2, label=label)


# =========================
# Range view helper
# =========================
def subset_by_daily_view(series: pd.Series, daily_view_label: str) -> pd.Series:
    s = _coerce_1d_series(series)
    if s.empty:
        return s
    lab = str(daily_view_label).strip().lower()
    if lab == "max":
        return s
    # approximate trading days
    mapping = {
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
        "5y": 1260,
        "10y": 2520,
    }
    n = mapping.get(lab, 252)
    return s.tail(int(n))

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if ax is None or real_times is None or len(real_times) == 0:
        return
    n = len(real_times)
    k = max(2, int(n_ticks))
    pos = np.linspace(0, n-1, k).astype(int)
    lbl = [real_times[i].strftime("%m-%d %H:%M") for i in pos]
    ax.set_xticks(pos)
    ax.set_xticklabels(lbl, rotation=0, ha="center", fontsize=9)


# =========================
# Forecasting
# =========================
def compute_sarimax_forecast(close: pd.Series, steps: int = 30):
    s = _coerce_1d_series(close).astype(float).dropna()
    if len(s) < 30:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize(), periods=steps, freq="D")
        vals = pd.Series(index=idx, dtype=float)
        ci = pd.DataFrame(index=idx, data={"lower": np.nan, "upper": np.nan})
        return idx, vals, ci

    # daily frequency index for forecast
    last_date = s.index[-1]
    if isinstance(last_date, (pd.Timestamp, datetime)):
        start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1)
    fc_idx = pd.date_range(start, periods=steps, freq="D", tz=PACIFIC)

    if _HAS_SARIMAX:
        try:
            model = SARIMAX(s, order=(1, 1, 1), trend="c",
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fc = res.get_forecast(steps=steps)
            mean = fc.predicted_mean
            conf = fc.conf_int(alpha=0.20)  # ~80% interval
            # align index
            mean.index = fc_idx
            conf.index = fc_idx
            ci = pd.DataFrame(index=fc_idx, data={"lower": conf.iloc[:, 0], "upper": conf.iloc[:, 1]})
            return fc_idx, mean, ci
        except Exception:
            pass

    # fallback: drift
    y = s.to_numpy(dtype=float)
    drift = (y[-1] - y[0]) / max(1, len(y) - 1)
    vals = pd.Series([y[-1] + drift*(i+1) for i in range(steps)], index=fc_idx)
    ci = pd.DataFrame(index=fc_idx, data={"lower": np.nan, "upper": np.nan})
    return fc_idx, vals, ci


# =========================
# Trade instruction (simple)
# =========================
def format_trade_instruction(trend_slope: float, buy_val: float, sell_val: float, close_val: float,
                             symbol: str, global_trend_slope: float):
    if not (np.isfinite(buy_val) and np.isfinite(sell_val) and np.isfinite(close_val)):
        return "Instruction: n/a"

    prox = 0.0025
    near_s = close_val <= buy_val * (1.0 + prox)
    near_r = close_val >= sell_val * (1.0 - prox)

    # ALERT if local slope contradicts global slope
    if np.isfinite(trend_slope) and np.isfinite(global_trend_slope):
        if np.sign(trend_slope) != 0 and np.sign(global_trend_slope) != 0 and np.sign(trend_slope) != np.sign(global_trend_slope):
            return f"ALERT: mixed trend (local {fmt_slope(trend_slope)} vs global {fmt_slope(global_trend_slope)})"

    if np.isfinite(trend_slope) and trend_slope > 0:
        if near_s:
            return f"BUY zone: near Support ({fmt_price_val(buy_val)})"
        return f"Uptrend: prefer BUY dips toward Support"
    if np.isfinite(trend_slope) and trend_slope < 0:
        if near_r:
            return f"SELL zone: near Resistance ({fmt_price_val(sell_val)})"
        return f"Downtrend: prefer SELL rallies toward Resistance"
    return "Sideways: wait / range trade S↔R"


# =========================
# Scanners (cached last values)
# =========================
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


@st.cache_data(ttl=120)
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str, years: int):
    try:
        close_full = fetch_hist(symbol, years=years).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_show.empty or len(close_show) < 2:
            return None

        # global slope on shown window
        x = np.arange(len(close_show), dtype=float)
        m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) <= 0.0:
            return None

        ntd_full = compute_normalized_trend(close_full, window=ntd_win)
        npx_full = compute_normalized_price(close_full, window=ntd_win)

        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        cross_up, _ = _cross_series(npx_show, ntd_show)
        if not cross_up.any():
            return None

        t = cross_up[cross_up].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Signal": "NPX↑NTD (Uptrend)",
            "Bars Since": bars_since,
            "Cross Time": t,
            "Global Slope": float(m),
            "Current Price": curr_px,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_daily_npx_zero_cross_with_local_slope(symbol: str,
                                               ntd_win: int,
                                               daily_view_label: str,
                                               local_slope_lb: int,
                                               max_abs_npx_at_cross: float,
                                               years: int,
                                               direction: str = "up"):
    try:
        close_full = fetch_hist(symbol, years=years).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
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

        seg = close_show.loc[:t].tail(int(local_slope_lb)).dropna()
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
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Daily View": daily_view_label,
            "Signal": sig_label,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Local Slope": float(m),
            "Current Price": curr_px,
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
                                 years: int = 10,
                                 min_r2: float = 0.99):
    try:
        close_full = fetch_hist(symbol, years=years).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
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
        return {
            "Symbol": symbol,
            "Side": "BUY" if want_buy else "SELL",
            "Daily View": daily_view_label,
            "Bars Since Cross": bars_since_cross,
            "Touch Time": t_touch,
            "Cross Time": t_cross,
            "Slope": float(m),
            "R2": float(r2),
            "Support@Touch": float(sup.loc[t_touch]) if (t_touch in sup.index and np.isfinite(sup.loc[t_touch])) else np.nan,
            "Resistance@Touch": float(res.loc[t_touch]) if (t_touch in res.index and np.isfinite(res.loc[t_touch])) else np.nan,
            "BB Mid@Cross": float(bb_mid.loc[t_cross]) if (t_cross in bb_mid.index and np.isfinite(bb_mid.loc[t_cross])) else np.nan,
            "Price@Cross": float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan,
            "Current Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
    except Exception:
        return None


# =========================
# Hourly renderer
# =========================
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool,
                        alert_placeholder=None,
                        *,
                        sr_lb_hourly: int,
                        slope_lb_hourly: int,
                        sr_prox_pct: float,
                        ntd_window: int,
                        show_ntd_panel: bool,
                        show_ntd: bool,
                        shade_ntd: bool,
                        show_npx_ntd: bool,
                        mark_npx_cross: bool,
                        show_bbands: bool,
                        bb_win: int,
                        bb_mult: float,
                        bb_use_ema: bool,
                        show_hma: bool,
                        hma_period: int,
                        show_macd: bool,
                        show_fibs: bool,
                        show_ichi: bool,
                        ichi_conv: int,
                        ichi_base: int,
                        ichi_spanb: int,
                        show_psar: bool,
                        psar_step: float,
                        psar_max: float,
                        atr_period: int,
                        atr_mult: float,
                        show_sessions_pst: bool,
                        show_fx_news: bool,
                        news_window_days: int,
                        rev_hist_lb: int,
                        rev_horizon: int,
                        rev_bars_confirm: int):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None

    # use numeric x for plotting while preserving timestamps for tick labels
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill().astype(float)
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
    rev_prob_h = slope_reversal_probability(
        hc, m_h, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon
    )

    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    ax2w = None
    if show_ntd_panel:
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
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, zorder=6)

    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}")

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.dropna().empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} ({fmt_slope(m_h)}/bar)")
    if not upper_h.dropna().empty and not lower_h.dropna().empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, m_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

        trig_h = find_slope_trigger_after_band_reversal(hc, yhat_h, upper_h, lower_h, horizon=rev_horizon)
        annotate_slope_trigger(ax2, trig_h)

    # Fib levels + UPDATED confirmation trigger
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

    # Forex: news + sessions
    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax2, news_pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=m_h,
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

    title_instr = "" if (alert_placeholder is not None and is_alert) else instr_txt
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
             f"Slope: {fmt_slope(m_h)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_h)}",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=9,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
    ax2.text(0.50, 0.02,
             f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9,
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

    # Indicator panel
    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX (win={ntd_window})")
        ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label="NTD")

        if show_npx_ntd and not _coerce_1d_series(npx_h).dropna().empty and show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(npx_h.index, npx_h.values*2 - 1.0, "-", linewidth=1.2, alpha=0.7, label="NPX (mapped to [-1,1])")
            if mark_npx_cross:
                up_mask, dn_mask = _cross_series(npx_h, (ntd_h+1)/2.0)  # compare in [0,1]
                if up_mask.any():
                    ax2w.scatter(up_mask[up_mask].index, [0.9]*int(up_mask.sum()), marker="o", s=40, zorder=9, label="NPX↑NTD")
                if dn_mask.any():
                    ax2w.scatter(dn_mask[dn_mask].index, [-0.9]*int(dn_mask.sum()), marker="x", s=60, zorder=9, label="NPX↓NTD")

        ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
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

    # optional MACD chart
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
# Sidebar + mode + settings
# =========================
st.title("Trend + Fibonacci Dashboard (Stocks + Forex)")

mode = st.sidebar.selectbox("Mode:", ["Stocks", "Forex"], index=0)

st.sidebar.caption("📝 Note: Place Buy Trade Closer to 0% Fibonacci and Sell trade closer to 100% Fibonacci.")

# Editable universe
default_stocks = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","SPY","QQQ","IWM"]
default_fx = ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","NZDUSD=X","USDCHF=X"]

universe_text = st.sidebar.text_area(
    "Universe (comma-separated tickers):",
    value=",".join(default_fx if mode == "Forex" else default_stocks),
    height=80
)
universe = [t.strip() for t in universe_text.split(",") if t.strip()]

# History window for daily data
hist_years = st.sidebar.slider("Daily history years", 1, 20, 10, 1)

daily_view = st.sidebar.selectbox("Daily view (chart/scanners):", ["3mo","6mo","1y","2y","5y","10y","Max"], index=2)

# Core windows
ntd_window = st.sidebar.slider("NTD/NPX window", 20, 500, 200, 10)
slope_lb_daily = st.sidebar.slider("Daily slope lookback (bars)", 20, 600, 200, 10)
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 10, 400, 120, 10)

sr_lb_daily = st.sidebar.slider("Daily S/R window", 5, 260, 60, 5)
sr_lb_hourly = st.sidebar.slider("Hourly S/R window", 5, 200, 30, 5)
sr_prox_pct = st.sidebar.slider("S/R proximity (pct)", 0.0005, 0.02, 0.0025, 0.0005)

# Reversal probability settings
rev_hist_lb = st.sidebar.slider("Reversal probability history window", 200, 2000, 800, 50)
rev_horizon = st.sidebar.slider("Reversal horizon (bars)", 2, 50, 10, 1)
rev_bars_confirm = st.sidebar.slider("Reversal confirm bars", 1, 5, 2, 1)

# Indicators toggles
show_ntd_panel = st.sidebar.checkbox("Show NTD/NPX panel", value=True)
show_ntd = st.sidebar.checkbox("Show NTD", value=True)
shade_ntd = st.sidebar.checkbox("Shade NTD extremes", value=True)
show_npx_ntd = st.sidebar.checkbox("Show NPX overlay", value=True)
mark_npx_cross = st.sidebar.checkbox("Mark NPX crosses", value=True)

show_hma = st.sidebar.checkbox("Show HMA", value=True)
hma_period = st.sidebar.slider("HMA period", 10, 200, 55, 1)

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
bb_win = st.sidebar.slider("BB window", 10, 200, 20, 1)
bb_mult = st.sidebar.slider("BB sigma", 1.0, 4.0, 2.0, 0.1)
bb_use_ema = st.sidebar.checkbox("BB mid uses EMA", value=False)

show_macd = st.sidebar.checkbox("Show MACD chart", value=False)

show_fibs = st.sidebar.checkbox("Show Fibonacci levels", value=True)

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=False)
ichi_conv = st.sidebar.slider("Ichimoku conv", 5, 30, 9, 1)
ichi_base = st.sidebar.slider("Ichimoku base", 10, 60, 26, 1)
ichi_spanb = st.sidebar.slider("Ichimoku spanB", 20, 120, 52, 1)

show_psar = st.sidebar.checkbox("Show PSAR", value=False)
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.10, 0.02, 0.01)
psar_max = st.sidebar.slider("PSAR max", 0.10, 0.50, 0.20, 0.01)

# Supertrend defaults ON in charts (as you requested earlier)
atr_period = st.sidebar.slider("Supertrend ATR period", 5, 30, 10, 1)
atr_mult = st.sidebar.slider("Supertrend ATR mult", 1.0, 6.0, 3.0, 0.25)

# Forex extras
show_sessions_pst = st.sidebar.checkbox("Show FX sessions (PST)", value=(mode=="Forex"))
show_fx_news = st.sidebar.checkbox("Show FX news markers", value=(mode=="Forex"))
news_window_days = st.sidebar.slider("News window (days)", 1, 30, 7, 1)

# Bull/Bear tab setting
bb_period = st.sidebar.selectbox("Bull/Bear lookback", ["1mo","3mo","6mo","1y","2y","5y"], index=2)

# Session init
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
    st.session_state.mode_at_run = None


# =========================
# Tabs
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
    "Fibonacci Proximity Scanner"
])

# ---------------------------
# TAB 1: ORIGINAL FORECAST
# ---------------------------
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data is cached for ~2 minutes. Charts stay on the last RUN ticker until you run again.")

    st.caption("📝 Place Buy Trade Closer to 0% Fibonacci and Sell trade closer to 100% Fibonacci.")

    if not universe:
        st.warning("Universe is empty. Add tickers in the sidebar.")
    else:
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
            df_hist = fetch_hist(sel, years=hist_years)
            df_ohlc = fetch_hist_ohlc(sel, years=hist_years)
            fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist, steps=30)
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

            # DAILY
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
                if df_ohlc is not None and not df_ohlc.empty and show_ichi and {"High","Low","Close"}.issubset(df_ohlc.columns):
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
                yhat_d_show = yhat_d.reindex(df_show.index)
                upper_d_show = upper_d.reindex(df_show.index)
                lower_d_show = lower_d.reindex(df_show.index)
                ntd_d_show = ntd_d.reindex(df_show.index)
                npx_d_show = npx_d_full.reindex(df_show.index)
                kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
                bb_mid_d_show = bb_mid_d.reindex(df_show.index)
                bb_up_d_show = bb_up_d.reindex(df_show.index)
                bb_lo_d_show = bb_lo_d.reindex(df_show.index)
                bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
                bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

                hma_d_show = compute_hma(df_show, period=hma_period)
                macd_d, macd_sig_d, macd_hist_d = compute_macd(df_show)

                psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                if not psar_d_df.empty and len(df_show.index) > 0:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

                # Daily Supertrend (ON by default)
                st_d_line = pd.Series(index=df_show.index, dtype=float)
                try:
                    if df_ohlc is not None and not df_ohlc.empty and {"High","Low","Close"}.issubset(df_ohlc.columns) and len(df_show.index) > 0:
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
                    f"{disp_ticker} Daily — {daily_view} — EMA, S/R (w={sr_lb_daily}), Slope, Pivots "
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
                        ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=15, zorder=6,
                                   label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                    if dn_mask.any():
                        ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=15, zorder=6)

                res_val_d = sup_val_d = np.nan
                try:
                    res_val_d = float(res_d_show.iloc[-1])
                    sup_val_d = float(sup_d_show.iloc[-1])
                except Exception:
                    pass
                if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                    ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6,
                              label=f"Resistance (w={sr_lb_daily})")
                    ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6,
                              label=f"Support (w={sr_lb_daily})")
                    label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}")
                    label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}")

                # Daily Supertrend plot
                if not _coerce_1d_series(st_d_line).dropna().empty:
                    ax.plot(st_d_line.index, st_d_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

                if not yhat_d_show.dropna().empty:
                    ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                            label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
                if not upper_d_show.dropna().empty and not lower_d_show.dropna().empty:
                    ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, alpha=0.85, label="Daily Trend +2σ")
                    ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, alpha=0.85, label="Daily Trend -2σ")

                    bounce_sig_d = find_band_bounce_signal(df_show, upper_d_show, lower_d_show, m_d)
                    if bounce_sig_d is not None:
                        annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

                    trig_d = find_slope_trigger_after_band_reversal(df_show, yhat_d_show, upper_d_show, lower_d_show, horizon=rev_horizon)
                    annotate_slope_trigger(ax, trig_d)

                # Fib levels + UPDATED confirmation trigger
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

                # MACD/HMA signal
                macd_sig_d = find_macd_hma_sr_signal(
                    close=df_show, hma=hma_d_show, macd=macd_d, sup=sup_d_show, res=res_d_show,
                    global_trend_slope=global_m_d, prox=sr_prox_pct
                )
                macd_txt = "MACD/HMA: n/a"
                if macd_sig_d is not None and np.isfinite(macd_sig_d.get("price", np.nan)):
                    macd_txt = f"MACD/HMA: {macd_sig_d['side']} @ {fmt_price_val(macd_sig_d['price'])}"
                    annotate_macd_signal(ax, macd_sig_d["time"], macd_sig_d["price"], macd_sig_d["side"])

                ax.text(0.01, 0.98, macd_txt,
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8))

                if piv and len(df_show) > 0:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in piv.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
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

                ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                ax.legend(loc="lower left", framealpha=0.5, fontsize=9)

                axdw.set_title(f"Daily Indicator Panel — NTD + NPX (win={ntd_window})")
                if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                    shade_ntd_regions(axdw, ntd_d_show)
                if show_ntd and not ntd_d_show.dropna().empty:
                    axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label="NTD")
                if show_npx_ntd and not npx_d_show.dropna().empty:
                    axdw.plot(npx_d_show.index, npx_d_show*2 - 1.0, "-", linewidth=1.2, alpha=0.7, label="NPX (mapped)")
                axdw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
                axdw.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
                axdw.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
                axdw.set_ylim(-1.1, 1.1)
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

            # HOURLY
            if chart in ("Hourly", "Both"):
                intraday = st.session_state.intraday
                render_hourly_views(
                    sel=disp_ticker,
                    intraday=intraday,
                    p_up=p_up,
                    p_dn=p_dn,
                    hour_range_label=st.session_state.hour_range,
                    is_forex=(mode == "Forex"),
                    alert_placeholder=alert_box,
                    sr_lb_hourly=sr_lb_hourly,
                    slope_lb_hourly=slope_lb_hourly,
                    sr_prox_pct=sr_prox_pct,
                    ntd_window=ntd_window,
                    show_ntd_panel=show_ntd_panel,
                    show_ntd=show_ntd,
                    shade_ntd=shade_ntd,
                    show_npx_ntd=show_npx_ntd,
                    mark_npx_cross=mark_npx_cross,
                    show_bbands=show_bbands,
                    bb_win=bb_win,
                    bb_mult=bb_mult,
                    bb_use_ema=bb_use_ema,
                    show_hma=show_hma,
                    hma_period=hma_period,
                    show_macd=show_macd,
                    show_fibs=show_fibs,
                    show_ichi=show_ichi,
                    ichi_conv=ichi_conv,
                    ichi_base=ichi_base,
                    ichi_spanb=ichi_spanb,
                    show_psar=show_psar,
                    psar_step=psar_step,
                    psar_max=psar_max,
                    atr_period=atr_period,
                    atr_mult=atr_mult,
                    show_sessions_pst=show_sessions_pst,
                    show_fx_news=show_fx_news,
                    news_window_days=news_window_days,
                    rev_hist_lb=rev_hist_lb,
                    rev_horizon=rev_horizon,
                    rev_bars_confirm=rev_bars_confirm
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

            st.subheader("Forecast (30d)")
            st.write(pd.DataFrame({
                "Forecast": st.session_state.fc_vals,
                "Lower":    st.session_state.fc_ci["lower"] if "lower" in st.session_state.fc_ci.columns else st.session_state.fc_ci.iloc[:, 0],
                "Upper":    st.session_state.fc_ci["upper"] if "upper" in st.session_state.fc_ci.columns else st.session_state.fc_ci.iloc[:, 1],
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

            # Daily supertrend
            st_d_line = pd.Series(index=df_show.index, dtype=float)
            try:
                if df_ohlc is not None and not df_ohlc.empty and len(df_show.index) > 0 and {"High","Low","Close"}.issubset(df_ohlc.columns):
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    ohlc_show = df_ohlc.loc[(df_ohlc.index >= x0) & (df_ohlc.index <= x1)]
                    st_d = compute_supertrend(ohlc_show, atr_period=atr_period, atr_mult=atr_mult)
                    if "ST" in st_d.columns:
                        st_d_line = _coerce_1d_series(st_d["ST"]).reindex(df_show.index).ffill().bfill()
            except Exception:
                pass

            # R² for fib confirmation trigger
            _, _, _, m_enh, r2_enh = regression_with_band(df_show, lookback=int(slope_lb_daily))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if not _coerce_1d_series(st_d_line).dropna().empty:
                ax.plot(st_d_line.index, st_d_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

            if not res_d_show.empty and not sup_d_show.empty:
                ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="-", linewidth=1.6, label="Support")

            # Fib levels + UPDATED confirmation trigger
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

            macd_sig = find_macd_hma_sr_signal(df_show, hma_d_show, macd_d, sup_d_show, res_d_show, global_m_d, prox=sr_prox_pct)
            macd_txt = "MACD/HMA: n/a"
            if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
                macd_txt = f"MACD/HMA: {macd_sig['side']} @ {fmt_price_val(macd_sig['price'])}"
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
                is_forex=(mode == "Forex"),
                sr_lb_hourly=sr_lb_hourly,
                slope_lb_hourly=slope_lb_hourly,
                sr_prox_pct=sr_prox_pct,
                ntd_window=ntd_window,
                show_ntd_panel=show_ntd_panel,
                show_ntd=show_ntd,
                shade_ntd=shade_ntd,
                show_npx_ntd=show_npx_ntd,
                mark_npx_cross=mark_npx_cross,
                show_bbands=show_bbands,
                bb_win=bb_win,
                bb_mult=bb_mult,
                bb_use_ema=bb_use_ema,
                show_hma=show_hma,
                hma_period=hma_period,
                show_macd=show_macd,
                show_fibs=show_fibs,
                show_ichi=show_ichi,
                ichi_conv=ichi_conv,
                ichi_base=ichi_base,
                ichi_spanb=ichi_spanb,
                show_psar=show_psar,
                psar_step=psar_step,
                psar_max=psar_max,
                atr_period=atr_period,
                atr_mult=atr_mult,
                show_sessions_pst=show_sessions_pst,
                show_fx_news=show_fx_news,
                news_window_days=news_window_days,
                rev_hist_lb=rev_hist_lb,
                rev_horizon=rev_horizon,
                rev_bars_confirm=rev_bars_confirm
            )

        st.subheader("Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci["lower"] if "lower" in ci.columns else ci.iloc[:, 0],
            "Upper":    ci["upper"] if "upper" in ci.columns else ci.iloc[:, 1]
        }, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    st.caption("Simple lookback performance overview (based on Bull/Bear lookback selection).")

    if not universe:
        st.warning("Universe is empty.")
    else:
        sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
        try:
            dfp = yf.download(sel_bb, period=bb_period, interval="1d", auto_adjust=False, progress=False)[["Close"]].dropna()
            dfp = _ensure_tz_index(dfp)
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

        fibs = fibonacci_levels(subset_by_daily_view(df, daily_view))
        px = _safe_last_float(df)
        st.write({
            "Fib position (daily view)": f"{fib_position_percent(px, fibs):.2f}%" if np.isfinite(fib_position_percent(px, fibs)) else "n/a"
        })

        if intr is not None and not intr.empty and "Close" in intr:
            # numeric index version
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
    st.caption("Lists symbols where the latest NTD is below -0.75 (Hourly uses latest intraday; Daily uses daily close).")

    if not universe:
        st.warning("Universe is empty.")
    else:
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
                    val, ts = last_daily_ntd_value(sym, ntd_window, years=hist_years)
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
    if not universe:
        st.warning("Universe is empty.")
    else:
        sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
        smax = fetch_hist_max(sel_lt)

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
    st.header("Recent BUY Scanner — Daily NPX↑NTD in Uptrend")
    st.caption(
        "Lists symbols where **NPX** most recently crossed **ABOVE** **NTD** (on Daily view), "
        "AND the daily global trend (in the selected Daily view range) is upward."
    )

    if not universe:
        st.warning("Universe is empty.")
    else:
        max_bars = st.slider("Max bars since NPX↑NTD cross", 0, 20, 2, 1, key="buy_scan_npx_max_bars")
        run_buy_scan = st.button("Run Recent BUY Scan", key="btn_run_recent_buy_scan_npx")

        if run_buy_scan:
            rows = []
            for sym in universe:
                r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view, years=hist_years)
                if r is not None and int(r.get("Bars Since", 9999)) <= int(max_bars):
                    rows.append(r)

            if not rows:
                st.info("No recent NPX↑NTD crosses found in an upward daily global trend (within selected bars).")
            else:
                out = pd.DataFrame(rows)
                out["Bars Since"] = out["Bars Since"].astype(int)
                out["Global Slope"] = out["Global Slope"].astype(float)
                out = out.sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 8: NPX 0.5-CROSS SCANNER
# ---------------------------
with tab8:
    st.header("NPX 0.5-Cross Scanner — Local Slope Confirmed (Daily)")
    st.caption(
        "Scans for symbols where NPX crossed 0.5 and the local price slope confirms.\n"
        "• UP list: NPX crosses up through 0.5 AND local slope up\n"
        "• DOWN list: NPX crosses down through 0.5 AND local slope down"
    )

    if not universe:
        st.warning("Universe is empty.")
    else:
        c1, c2, c3 = st.columns(3)
        max_bars0 = c1.slider("Max bars since NPX 0.5-cross", 0, 30, 2, 1, key="npx0_max_bars")
        eps0 = c2.slider("Max |NPX-0.5| at cross", 0.01, 0.30, 0.08, 0.01, key="npx0_eps")
        lb_local = c3.slider("Local slope lookback (bars)", 10, 360, int(slope_lb_daily), 10, key="npx0_slope_lb")

        run0 = st.button("Run NPX 0.5-Cross Scan", key="btn_run_npx0_scan")

        if run0:
            rows_up, rows_dn = [], []
            for sym in universe:
                r_up = last_daily_npx_zero_cross_with_local_slope(
                    sym, ntd_win=ntd_window, daily_view_label=daily_view,
                    local_slope_lb=lb_local, max_abs_npx_at_cross=eps0,
                    years=hist_years, direction="up"
                )
                if r_up is not None and int(r_up.get("Bars Since", 9999)) <= int(max_bars0):
                    rows_up.append(r_up)

                r_dn = last_daily_npx_zero_cross_with_local_slope(
                    sym, ntd_win=ntd_window, daily_view_label=daily_view,
                    local_slope_lb=lb_local, max_abs_npx_at_cross=eps0,
                    years=hist_years, direction="down"
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
    st.header("Daily Slope + S/R Reversal + BB Midline Scanner (R² threshold)")
    st.caption(
        "BUY list:\n"
        "• Daily regression slope UP and R² >= threshold\n"
        "• Price reversed from Support (touch within horizon + confirmed reversal)\n"
        "• Price crossed ABOVE BB midline\n\n"
        "SELL list:\n"
        "• Daily regression slope DOWN and R² >= threshold\n"
        "• Price reversed from Resistance\n"
        "• Price crossed BELOW BB midline"
    )

    if not universe:
        st.warning("Universe is empty.")
    else:
        c1, c2 = st.columns(2)
        max_bars_since = c1.slider("Max bars since BB mid cross", 0, 60, 10, 1, key="srbb_max_bars_since")
        r2_thr = c2.slider("Min R² (confidence)", 0.80, 0.999, 0.99, 0.001, key="srbb_r2_thr")

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
                    years=hist_years,
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
                    years=hist_years,
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
# TAB 10: Fibonacci Proximity Scanner (Daily / Hourly / Both)
# ---------------------------
with tab10:
    st.header("Fibonacci Proximity Scanner — Closest to 0% and 100%")
    st.caption("Shows symbols closest to Fibonacci edges. 0% = LOW, 100% = HIGH.")

    if not universe:
        st.warning("Universe is empty.")
    else:
        frame = st.radio("Frame:", ["Daily", "Hourly (intraday)", "Both"], index=2, key=f"fib_scan_frame_{mode}")
        top_n = st.slider("Top N per edge", 3, 30, 10, 1, key=f"fib_topn_{mode}")
        period = st.selectbox("Hourly lookback (scanner):", ["1d", "2d", "4d"], index=0, key=f"fib_scan_period_{mode}")

        run_fib_scan = st.button("Run Fibonacci Proximity Scan", key=f"btn_run_fib_scan_{mode}")

        if run_fib_scan:
            rows = []
            if frame in ("Daily", "Both"):
                for sym in universe:
                    r = last_daily_fib_position(sym, daily_view_label=daily_view, years=hist_years)
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
