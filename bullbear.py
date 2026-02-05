# ============================================================
# bullbear.py  (FULL APP — BATCH 1/3)
# ============================================================

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional dependency: yfinance for data
try:
    import yfinance as yf
except Exception as e:
    yf = None


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Stock Wizard — Bull/Bear",
    layout="wide",
)


# ---------------------------
# Utilities
# ---------------------------
def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def _coerce_1d_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x)
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)


def _series_heading_up(s: pd.Series, confirm_bars: int = 1) -> bool:
    """True if last `confirm_bars` consecutive closes are rising."""
    s = _coerce_1d_series(s).dropna()
    if len(s) < confirm_bars + 1:
        return False
    tail = s.iloc[-(confirm_bars + 1):].values
    diffs = np.diff(tail)
    return bool(np.all(diffs > 0))


def _parse_universe(text: str) -> List[str]:
    """Parse comma/space/newline separated tickers."""
    if not isinstance(text, str):
        return []
    raw = (
        text.replace("\n", ",")
        .replace(" ", ",")
        .replace(";", ",")
        .replace("|", ",")
        .split(",")
    )
    out = []
    for t in raw:
        t = t.strip().upper()
        if not t:
            continue
        # Basic sanitize:
        t = "".join(ch for ch in t if ch.isalnum() or ch in ".-_^")
        if t:
            out.append(t)
    # De-duplicate preserving order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _require_yfinance():
    if yf is None:
        st.error(
            "This app requires `yfinance` but it isn't available in this environment. "
            "Install it with `pip install yfinance`."
        )
        st.stop()


# ---------------------------
# Sidebar — Inputs & Settings
# ---------------------------
st.sidebar.title("Stock Wizard Settings")

mode = st.sidebar.radio(
    "Mode",
    ["Single Symbol", "Universe Scan"],
    index=0,
    help="Single Symbol shows charts; Universe Scan runs scanners across a ticker list."
)

default_universe = "AAPL, MSFT, NVDA, AMZN, META, TSLA, SPY, QQQ"
universe_text = st.sidebar.text_area(
    "Universe tickers (for scans)",
    value=default_universe,
    height=100
)

universe = _parse_universe(universe_text)
if not universe:
    universe = ["AAPL"]

symbol = st.sidebar.text_input(
    "Symbol (Single Symbol mode)",
    value=universe[0] if universe else "AAPL"
).strip().upper() or (universe[0] if universe else "AAPL")

daily_view = st.sidebar.selectbox(
    "Daily view range",
    ["3M", "6M", "1Y", "2Y", "5Y", "MAX"],
    index=2
)

# Regression lookbacks
slope_lb_daily = st.sidebar.slider("Daily regression lookback (bars)", 20, 400, 120, 5)
slope_lb_hourly = st.sidebar.slider("Hourly regression lookback (bars)", 20, 400, 120, 5)

# NTD window for "Norm Price" (NPX)
ntd_window = st.sidebar.slider("NTD window (bars)", 20, 400, 120, 5)

# S/R scan parameters
sr_lb_daily = st.sidebar.slider("Daily S/R lookback (bars)", 10, 400, 120, 5)
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 5.00, 0.75, 0.05) / 100.0
rev_bars_confirm = st.sidebar.slider("Reversal confirm bars", 1, 10, 2, 1)
rev_horizon = st.sidebar.slider("Reversal search horizon (bars)", 3, 60, 15, 1)

# Ichimoku parameters (IMPORTANT: must exist, fixes NameError at kijun scanner callsite)
st.sidebar.subheader("Ichimoku (Daily)")
ichi_conv = st.sidebar.slider("Tenkan (conversion) period", 5, 30, 9, 1)
ichi_base = st.sidebar.slider("Kijun (base) period", 10, 60, 26, 1)
ichi_spanb = st.sidebar.slider("Senkou Span B period", 30, 120, 52, 1)

# Basic performance knobs
st.sidebar.subheader("Performance")
max_scan_symbols = st.sidebar.slider("Max symbols per scan", 10, 500, 150, 10)
scan_delay_ms = st.sidebar.slider("Per-symbol delay (ms)", 0, 200, 0, 10)

# Cap universe size for scans
if len(universe) > int(max_scan_symbols):
    universe = universe[: int(max_scan_symbols)]


# ---------------------------
# Data Fetching & View Windows
# ---------------------------
DAILY_VIEW_MAP = {
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "2Y": 365 * 2,
    "5Y": 365 * 5,
    "MAX": None,
}


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV using yfinance. Returns DataFrame with columns: Open, High, Low, Close, Volume."""
    _require_yfinance()
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance may return multi-index columns if multiple tickers; handle single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=0)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # Standardize column names
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c not in df.columns:
            continue
    # Keep just primary OHLCV if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=["Close"]) if "Close" in df.columns else df.dropna()
    return df


def daily_df_full(symbol: str) -> pd.DataFrame:
    # Long-ish period so we can slice view windows
    return fetch_ohlc(symbol, interval="1d", period="10y")


def daily_df_in_view(symbol: str, daily_view_label: str) -> pd.DataFrame:
    df = daily_df_full(symbol)
    if df.empty:
        return df
    days = DAILY_VIEW_MAP.get(daily_view_label, None)
    if days is None:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=int(days))
    return df[df.index >= cutoff]


def hourly_df_period(symbol: str, period: str) -> pd.DataFrame:
    # yfinance supports e.g. period="1d","2d","5d","1mo" etc with interval="60m"
    return fetch_ohlc(symbol, interval="60m", period=period)


# ---------------------------
# Regression with Bands
# ---------------------------
@dataclass
class RegResult:
    slope: float
    intercept: float
    r2: float
    yhat: pd.Series
    resid: pd.Series
    sigma: float
    upper_2s: pd.Series
    lower_2s: pd.Series


def regression_with_band(y: pd.Series, lookback: int) -> Optional[RegResult]:
    """
    Linear regression y ~ x on the last `lookback` points of y.
    Returns fitted line, residuals, sigma (std of residuals), and ±2σ bands.
    """
    y = _coerce_1d_series(y).dropna()
    if y.empty:
        return None
    if len(y) < max(10, int(lookback)):
        lookback = len(y)

    y_lb = y.iloc[-int(lookback):].astype(float)
    n = len(y_lb)
    if n < 10:
        return None

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y_lb.values.mean()

    # Compute slope/intercept
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return None
    slope = float(np.sum((x - x_mean) * (y_lb.values - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)

    yhat_vals = intercept + slope * x
    yhat = pd.Series(yhat_vals, index=y_lb.index, name="yhat")
    resid = (y_lb - yhat).rename("resid")

    # R^2
    ss_res = float(np.sum((resid.values) ** 2))
    ss_tot = float(np.sum((y_lb.values - y_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    sigma = float(np.std(resid.values, ddof=1)) if n > 2 else float("nan")
    if not np.isfinite(sigma) or sigma <= 0:
        # Avoid division by zero; still return a result but bands equal yhat
        sigma = float("nan")

    upper_2s = (yhat + 2.0 * sigma).rename("upper_2s") if np.isfinite(sigma) else yhat.rename("upper_2s")
    lower_2s = (yhat - 2.0 * sigma).rename("lower_2s") if np.isfinite(sigma) else yhat.rename("lower_2s")

    return RegResult(
        slope=slope,
        intercept=intercept,
        r2=r2,
        yhat=yhat,
        resid=resid,
        sigma=sigma,
        upper_2s=upper_2s,
        lower_2s=lower_2s,
    )


# ---------------------------
# Daily slope / R2 helpers
# ---------------------------
def daily_global_slope(symbol: str, daily_view_label: str) -> Tuple[float, float, str]:
    """Slope + R2 on daily close for the current view window (or as much as possible)."""
    df = daily_df_in_view(symbol, daily_view_label)
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, _now_ts()

    rr = regression_with_band(df["Close"], lookback=min(len(df), int(slope_lb_daily)))
    if rr is None:
        return np.nan, np.nan, _now_ts()
    return float(rr.slope), float(rr.r2), _now_ts()


def daily_regression_r2(symbol: str, slope_lb: int) -> Tuple[float, float, str]:
    df = daily_df_full(symbol)
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, _now_ts()
    rr = regression_with_band(df["Close"], lookback=int(slope_lb))
    if rr is None:
        return np.nan, np.nan, _now_ts()
    return float(rr.r2), float(rr.slope), _now_ts()


def hourly_regression_r2(symbol: str, period: str, slope_lb: int) -> Tuple[float, float, str]:
    df = hourly_df_period(symbol, period=period)
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, _now_ts()
    rr = regression_with_band(df["Close"], lookback=int(slope_lb))
    if rr is None:
        return np.nan, np.nan, _now_ts()
    return float(rr.r2), float(rr.slope), _now_ts()


# ---------------------------
# NPX (Normalized Price) within a view
#   NPX = (Close - yhat) / sigma   where yhat/sigma from regression
# ---------------------------
def daily_npx_series_in_view(symbol: str, daily_view_label: str, ntd_win: int) -> pd.Series:
    df = daily_df_in_view(symbol, daily_view_label)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    rr = regression_with_band(df["Close"], lookback=min(int(ntd_win), len(df)))
    if rr is None or (not np.isfinite(rr.sigma)) or rr.sigma == 0:
        return pd.Series(dtype=float)

    # Align close to regression lookback window index
    close_lb = df["Close"].iloc[-len(rr.yhat):]
    npx = ((close_lb - rr.yhat) / rr.sigma).rename("NPX")
    return npx


def daily_last_npx_in_view(symbol: str, daily_view_label: str, ntd_win: int) -> Tuple[float, str]:
    s = daily_npx_series_in_view(symbol, daily_view_label, ntd_win)
    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return np.nan, _now_ts()
    return float(s.iloc[-1]), str(s.index[-1])


# ---------------------------
# Support/Resistance + Reversal logic
# ---------------------------
def _support_resistance(df: pd.DataFrame, lb: int) -> Tuple[pd.Series, pd.Series]:
    """
    Simple rolling support/resistance:
      support = rolling min of Low
      resistance = rolling max of High
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    low = df["Low"] if "Low" in df.columns else df["Close"]
    high = df["High"] if "High" in df.columns else df["Close"]
    support = low.rolling(int(lb), min_periods=max(5, int(lb) // 3)).min().rename("Support")
    resistance = high.rolling(int(lb), min_periods=max(5, int(lb) // 3)).max().rename("Resistance")
    return support, resistance


def daily_support_reversal_heading_up(
    symbol: str,
    daily_view_label: str,
    sr_lb: int,
    prox: float,
    bars_confirm: int,
    horizon: int,
) -> Optional[Dict]:
    """
    Find most recent support "touch" within last `horizon` bars where price was near support,
    followed by `bars_confirm` consecutive rising closes.

    Returns dict with metadata if found.
    """
    df = daily_df_in_view(symbol, daily_view_label)
    if df.empty or "Close" not in df.columns:
        return None

    support, _ = _support_resistance(df, lb=sr_lb)
    close = df["Close"]

    # work on last horizon window
    if len(df) < max(10, horizon + bars_confirm + 2):
        horizon = min(horizon, max(3, len(df) - (bars_confirm + 2)))

    tail_idx = df.index[-int(horizon):]
    # Identify touch points: close <= support*(1+prox)
    touch = (close.loc[tail_idx] <= support.loc[tail_idx] * (1.0 + prox))

    if not touch.any():
        return None

    # Last touch time
    last_touch_ts = touch[touch].index[-1]
    touch_pos = df.index.get_loc(last_touch_ts)

    # Need enough bars after touch
    after = close.iloc[touch_pos:touch_pos + bars_confirm + 1]
    if len(after) < bars_confirm + 1:
        return None

    if not _series_heading_up(after, confirm_bars=bars_confirm):
        return None

    bars_since_touch = len(df) - 1 - touch_pos
    return {
        "Symbol": symbol,
        "Touch Date": str(last_touch_ts.date()),
        "Close@Touch": float(close.loc[last_touch_ts]),
        "Support@Touch": float(support.loc[last_touch_ts]) if np.isfinite(support.loc[last_touch_ts]) else np.nan,
        "Bars Since Touch": int(bars_since_touch),
        "Confirm Bars": int(bars_confirm),
    }


# ---------------------------
# Ichimoku helpers
# ---------------------------
def ichimoku_lines(df: pd.DataFrame, conv: int, base: int, span_b: int) -> pd.DataFrame:
    """
    Compute Ichimoku lines on OHLC df:
      Tenkan-sen (conversion) = (9-period high + 9-period low)/2
      Kijun-sen (base) = (26-period high + 26-period low)/2
      Senkou Span B = (52-period high + 52-period low)/2
    """
    if df.empty:
        return pd.DataFrame()

    high = df["High"] if "High" in df.columns else df["Close"]
    low = df["Low"] if "Low" in df.columns else df["Close"]

    tenkan = ((high.rolling(conv).max() + low.rolling(conv).min()) / 2.0).rename("Tenkan")
    kijun = ((high.rolling(base).max() + low.rolling(base).min()) / 2.0).rename("Kijun")
    span_b_line = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0).rename("SpanB")

    out = pd.concat([tenkan, kijun, span_b_line], axis=1)
    return out


def last_daily_kijun_cross_up(
    symbol: str,
    daily_view_label: str,
    slope_lb: int,
    conv: int,
    base: int,
    span_b: int,
    within_last_n_bars: int = 5,
) -> Optional[Dict]:
    """
    Find the most recent daily cross where Close crosses from below Kijun to above Kijun.
    Return details + slope + r2 computed on daily view.
    """
    df = daily_df_in_view(symbol, daily_view_label)
    if df.empty or "Close" not in df.columns:
        return None

    ichi = ichimoku_lines(df, conv=conv, base=base, span_b=span_b)
    if ichi.empty or "Kijun" not in ichi.columns:
        return None

    close = df["Close"].astype(float)
    kijun = ichi["Kijun"].astype(float)

    # Need overlap where kijun is not NaN
    m = pd.concat([close, kijun], axis=1).dropna()
    if m.empty or len(m) < 5:
        return None

    c = m["Close"]
    k = m["Kijun"]

    # Cross up: yesterday close <= kijun AND today close > kijun
    cross = (c.shift(1) <= k.shift(1)) & (c > k)
    if not cross.any():
        return None

    cross_times = cross[cross].index
    last_cross_ts = cross_times[-1]

    # Enforce within last N bars in *aligned* data index
    if within_last_n_bars is not None and int(within_last_n_bars) > 0:
        bars_since = len(m) - 1 - m.index.get_loc(last_cross_ts)
        if bars_since > int(within_last_n_bars):
            return None
    else:
        bars_since = len(m) - 1 - m.index.get_loc(last_cross_ts)

    # "Heading up" confirmation: last two closes rising (simple)
    if len(c) < 2 or not _series_heading_up(c.dropna(), confirm_bars=1):
        return None

    # slope/r2 on view (same logic used by scanners)
    rr = regression_with_band(df["Close"], lookback=min(len(df), int(slope_lb)))
    if rr is None:
        slope = np.nan
        r2 = np.nan
    else:
        slope = float(rr.slope)
        r2 = float(rr.r2)

    return {
        "Symbol": symbol,
        "Cross Date": str(last_cross_ts.date()),
        "Bars Since Cross": int(bars_since),
        "Price@Cross": float(c.loc[last_cross_ts]),
        "Kijun@Cross": float(k.loc[last_cross_ts]),
        "Slope": slope,
        "R2": r2,
    }


# ---------------------------
# Signed R2 band proximity (Daily)
#   SignedR2 = R2 * sign(slope)
# ---------------------------
def daily_r2_band_proximity(
    symbol: str,
    daily_view_label: str,
    slope_lb: int,
    prox: float,
) -> Optional[Dict]:
    df = daily_df_in_view(symbol, daily_view_label)
    if df.empty or "Close" not in df.columns:
        return None

    rr = regression_with_band(df["Close"], lookback=min(len(df), int(slope_lb)))
    if rr is None:
        return None

    close_lb = df["Close"].iloc[-len(rr.yhat):]
    last_ts = close_lb.index[-1]
    last_close = float(close_lb.iloc[-1])

    # Bands at last timestamp
    up = rr.upper_2s.loc[last_ts] if last_ts in rr.upper_2s.index else np.nan
    lo = rr.lower_2s.loc[last_ts] if last_ts in rr.lower_2s.index else np.nan

    if not (np.isfinite(up) and np.isfinite(lo)):
        return None

    # Distance as % of price
    absdist_up = abs(last_close - float(up)) / last_close if last_close != 0 else np.nan
    absdist_lo = abs(last_close - float(lo)) / last_close if last_close != 0 else np.nan

    near_upper = bool(np.isfinite(absdist_up) and absdist_up <= prox)
    near_lower = bool(np.isfinite(absdist_lo) and absdist_lo <= prox)

    slope = float(rr.slope) if np.isfinite(rr.slope) else np.nan
    r2 = float(rr.r2) if np.isfinite(rr.r2) else np.nan
    signed_r2 = (r2 * (1.0 if slope >= 0 else -1.0)) if np.isfinite(r2) and np.isfinite(slope) else np.nan

    return {
        "Symbol": symbol,
        "Close": last_close,
        "Upper(+2σ)": float(up),
        "Lower(-2σ)": float(lo),
        "AbsDist Upper (%)": float(absdist_up * 100.0) if np.isfinite(absdist_up) else np.nan,
        "AbsDist Lower (%)": float(absdist_lo * 100.0) if np.isfinite(absdist_lo) else np.nan,
        "Slope": slope,
        "R2": r2,
        "SignedR2": signed_r2,
        "Near Upper": near_upper,
        "Near Lower": near_lower,
        "AsOf": _now_ts(),
    }
# ============================================================
# bullbear.py  (FULL APP — BATCH 2/3)
# ============================================================

import matplotlib.pyplot as plt

st.title("Stock Wizard — Bull/Bear Dashboard")

# Top status
cA, cB, cC = st.columns([2, 2, 3])
cA.metric("Mode", mode)
cB.metric("Universe size", len(universe))
cC.caption(f"Last refresh: `{_now_ts()}`")

if yf is None:
    st.warning("`yfinance` is not installed. Install it to fetch market data.")

tabs = st.tabs([
    "1) Overview",
    "2) Daily Price Chart",
    "3) Hourly Price Chart",
    "4) Daily Regression Stats",
    "5) Daily Support/Resistance",
    "6) Support Reversal (Single)",
    "7) Quick Scans (Small)",
    "8) Diagnostics",
    # scanners tabs continue in Batch 3
    "9) NTD Buy Signal",
    "10) Slope Direction Scan",
    "11) Trendline Direction Lists",
    "12) NTD Hot List",
    "13) NPX 0.0–0.2 Scanner",
    "14) Uptrend vs Downtrend",
    "15) Ichimoku Kijun Scanner",
    "16) R² > Threshold Daily/Hourly",
    "17) R² < Threshold Daily/Hourly",
    "18) SignedR² ±2σ Proximity",
])

(tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,
 tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18) = tabs


# ---------------------------
# Plot helpers
# ---------------------------
def _plot_close_with_bands(df: pd.DataFrame, rr: RegResult, title: str):
    close_lb = df["Close"].iloc[-len(rr.yhat):]
    fig, ax = plt.subplots()
    ax.plot(close_lb.index, close_lb.values, label="Close")
    ax.plot(rr.yhat.index, rr.yhat.values, linestyle="--", label="Regression")
    ax.plot(rr.upper_2s.index, rr.upper_2s.values, linestyle=":", label="+2σ")
    ax.plot(rr.lower_2s.index, rr.lower_2s.values, linestyle=":", label="-2σ")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def _plot_support_resistance(df: pd.DataFrame, support: pd.Series, resistance: pd.Series, title: str):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["Close"].values, label="Close")
    if not support.dropna().empty:
        ax.plot(support.index, support.values, linestyle="--", label="Support")
    if not resistance.dropna().empty:
        ax.plot(resistance.index, resistance.values, linestyle="--", label="Resistance")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# ---------------------------
# TAB 1: Overview
# ---------------------------
with tab1:
    st.header("Overview")
    st.write(
        "This app provides daily/hourly charts, regression bands, support/resistance, "
        "NPX (normalized price vs regression), and multiple scanners across a universe."
    )

    st.subheader("Current Inputs")
    st.write(
        {
            "Mode": mode,
            "Single symbol": symbol,
            "Universe (first 25)": universe[:25],
            "Daily view": daily_view,
            "Daily reg lookback": slope_lb_daily,
            "Hourly reg lookback": slope_lb_hourly,
            "NTD window": ntd_window,
            "S/R lookback": sr_lb_daily,
            "S/R proximity %": sr_prox_pct * 100.0,
            "Reversal confirm bars": rev_bars_confirm,
            "Reversal horizon": rev_horizon,
            "Ichimoku conv/base/spanB": (ichi_conv, ichi_base, ichi_spanb),
        }
    )

    st.info(
        "Tip: Use **Universe Scan** mode for the scanner tabs (9–18). "
        "Single Symbol mode is best for tabs 2–6."
    )


# ---------------------------
# TAB 2: Daily Price Chart
# ---------------------------
with tab2:
    st.header("Daily Price Chart")
    if mode != "Single Symbol":
        st.warning("Switch to **Single Symbol** mode to use this tab.")

    _require_yfinance()
    df = daily_df_in_view(symbol, daily_view)
    if df.empty:
        st.error("No daily data returned.")
    else:
        rr = regression_with_band(df["Close"], lookback=min(len(df), int(slope_lb_daily)))
        if rr is None:
            st.error("Not enough data to compute regression.")
        else:
            st.caption(f"Slope: {rr.slope:.6f} | R²: {rr.r2:.3f} | σ: {rr.sigma:.4f}")
            _plot_close_with_bands(df, rr, title=f"{symbol} — Daily Close with Regression Bands ({daily_view})")

        st.dataframe(df.tail(20), use_container_width=True)


# ---------------------------
# TAB 3: Hourly Price Chart
# ---------------------------
with tab3:
    st.header("Hourly Price Chart")
    if mode != "Single Symbol":
        st.warning("Switch to **Single Symbol** mode to use this tab.")

    _require_yfinance()
    period = st.selectbox("Hourly period", ["1d", "2d", "5d", "1mo"], index=1, key="hour_period_chart")
    hdf = hourly_df_period(symbol, period=period)
    if hdf.empty:
        st.error("No hourly data returned.")
    else:
        rr = regression_with_band(hdf["Close"], lookback=min(len(hdf), int(slope_lb_hourly)))
        if rr is None:
            st.error("Not enough data to compute hourly regression.")
        else:
            st.caption(f"Slope: {rr.slope:.6f} | R²: {rr.r2:.3f} | σ: {rr.sigma:.4f}")
            _plot_close_with_bands(hdf, rr, title=f"{symbol} — Hourly Close with Regression Bands (period={period})")

        st.dataframe(hdf.tail(30), use_container_width=True)


# ---------------------------
# TAB 4: Daily Regression Stats
# ---------------------------
with tab4:
    st.header("Daily Regression Stats (View Window)")
    _require_yfinance()

    df = daily_df_in_view(symbol, daily_view)
    if df.empty:
        st.error("No daily data.")
    else:
        rr = regression_with_band(df["Close"], lookback=min(len(df), int(slope_lb_daily)))
        if rr is None:
            st.error("Not enough data.")
        else:
            last_close = float(df["Close"].iloc[-1])
            last_hat = float(rr.yhat.iloc[-1])
            npx = (last_close - last_hat) / rr.sigma if np.isfinite(rr.sigma) and rr.sigma != 0 else np.nan

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Slope", f"{rr.slope:.6f}")
            c2.metric("R²", f"{rr.r2:.3f}")
            c3.metric("NPX (last)", f"{npx:.3f}" if np.isfinite(npx) else "n/a")
            c4.metric("Last Close", f"{last_close:.2f}")

            st.subheader("NPX series (last 60)")
            npx_series = daily_npx_series_in_view(symbol, daily_view, ntd_window)
            st.line_chart(npx_series.tail(60))


# ---------------------------
# TAB 5: Daily Support/Resistance
# ---------------------------
with tab5:
    st.header("Daily Support/Resistance")
    _require_yfinance()

    df = daily_df_in_view(symbol, daily_view)
    if df.empty:
        st.error("No daily data.")
    else:
        support, resistance = _support_resistance(df, lb=int(sr_lb_daily))
        _plot_support_resistance(df, support, resistance, title=f"{symbol} — Support/Resistance ({daily_view})")

        last = df.index[-1]
        sup = support.loc[last] if last in support.index else np.nan
        res = resistance.loc[last] if last in resistance.index else np.nan
        close = float(df["Close"].iloc[-1])

        c1, c2, c3 = st.columns(3)
        c1.metric("Close", f"{close:.2f}")
        c2.metric("Support", f"{sup:.2f}" if np.isfinite(sup) else "n/a")
        c3.metric("Resistance", f"{res:.2f}" if np.isfinite(res) else "n/a")


# ---------------------------
# TAB 6: Support Reversal (Single)
# ---------------------------
with tab6:
    st.header("Support Reversal (Single Symbol)")
    _require_yfinance()

    rev = daily_support_reversal_heading_up(
        symbol=symbol,
        daily_view_label=daily_view,
        sr_lb=int(sr_lb_daily),
        prox=float(sr_prox_pct),
        bars_confirm=int(rev_bars_confirm),
        horizon=int(rev_horizon),
    )

    if rev is None:
        st.info("No recent confirmed support reversal (heading up) found within the horizon.")
    else:
        st.success("Support reversal detected!")
        st.write(rev)


# ---------------------------
# TAB 7: Quick Scans (Small)
# ---------------------------
with tab7:
    st.header("Quick Scans (Small)")
    st.caption("Lightweight examples on a small subset of the universe.")

    _require_yfinance()
    sample_n = st.slider("Sample size", 5, min(50, len(universe)), min(10, len(universe)), 1)

    run_small = st.button("Run sample scan", key="btn_run_small")
    if run_small:
        rows = []
        for sym in universe[:sample_n]:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                npx_last, _ = daily_last_npx_in_view(sym, daily_view, ntd_window)
                rows.append({"Symbol": sym, "Slope": m, "R2": r2, "NPX": npx_last, "AsOf": ts})
                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)
            except Exception as e:
                rows.append({"Symbol": sym, "Slope": np.nan, "R2": np.nan, "NPX": np.nan, "AsOf": _now_ts()})

        out = pd.DataFrame(rows).sort_values(["Slope"], ascending=False)
        st.dataframe(out, use_container_width=True)


# ---------------------------
# TAB 8: Diagnostics
# ---------------------------
with tab8:
    st.header("Diagnostics")
    st.write("Use this to quickly validate data availability and indicator readiness.")

    _require_yfinance()

    test_sym = st.selectbox("Test symbol", universe, index=0, key="diag_sym")
    ddf = daily_df_in_view(test_sym, daily_view)
    hdf = hourly_df_period(test_sym, period="2d")

    st.subheader("Daily")
    st.write({"rows": int(len(ddf)), "cols": list(ddf.columns)})
    st.dataframe(ddf.tail(5), use_container_width=True)

    st.subheader("Hourly (2d)")
    st.write({"rows": int(len(hdf)), "cols": list(hdf.columns)})
    st.dataframe(hdf.tail(5), use_container_width=True)

    st.subheader("Ichimoku readiness")
    ichi = ichimoku_lines(ddf, conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
    st.write({"ichimoku_rows": int(len(ichi.dropna()))})
    st.dataframe(ichi.tail(5), use_container_width=True)
# ============================================================
# bullbear.py  (FULL APP — BATCH 3/3)
# ============================================================

# NOTE:
# Tabs 9–18 assume `universe` is defined and yfinance is available.

# ---------------------------
# TAB 9: NTD BUY SIGNAL
#   - Regression line slope > 0
#   - NPX between +0.5 and +0.6
#   - NPX heading up
# ---------------------------
with tab9:
    st.header("NTD Buy Signal")
    st.caption(
        "Shows symbols where:\n"
        "• Regression slope **> 0** (Daily, selected Daily view range)\n"
        "• **NPX** between **+0.5 and +0.6**\n"
        "• NPX is **heading up**"
    )

    _require_yfinance()

    c1, c2 = st.columns(2)
    npx_up_bars_056 = c1.slider("NPX heading-up confirmation (consecutive bars)", 1, 5, 1, 1, key="npx_056_up_bars")
    run_ntd_buy = c2.button("Run NTD Buy Signal Scan", key=f"btn_run_ntd_buy_{mode}")

    if run_ntd_buy:
        rows = []
        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m) or float(m) <= 0.0:
                    continue

                npx_s = daily_npx_series_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
                npx_s = _coerce_1d_series(npx_s).dropna()
                if npx_s.empty or len(npx_s) < 2:
                    continue

                npx_last = float(npx_s.iloc[-1]) if np.isfinite(npx_s.iloc[-1]) else np.nan
                npx_prev = float(npx_s.iloc[-2]) if np.isfinite(npx_s.iloc[-2]) else np.nan
                if not np.isfinite(npx_last):
                    continue

                if not (0.5 <= float(npx_last) <= 0.6):
                    continue

                if not _series_heading_up(npx_s, confirm_bars=int(npx_up_bars_056)):
                    continue

                rows.append({
                    "Symbol": sym,
                    "Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "NPX (last)": float(npx_last),
                    "NPX (prev)": float(npx_prev) if np.isfinite(npx_prev) else np.nan,
                    "AsOf": ts
                })

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out = out.sort_values(["NPX (last)", "Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 10: Slope Direction Scan
# ---------------------------
with tab10:
    st.header("Slope Direction Scan")
    st.caption(
        "Lists symbols whose **current DAILY trendline slope** is **up** vs **down** "
        "(based on the selected Daily view range)."
    )

    _require_yfinance()

    run_slope = st.button("Run Slope Direction Scan", key=f"btn_run_slope_dir_{mode}")

    if run_slope:
        rows = []
        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue
                rows.append({
                    "Symbol": sym,
                    "Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "AsOf": ts
                })
                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)
            except Exception:
                continue

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            up = out[out["Slope"] > 0].sort_values(["Slope"], ascending=False)
            dn = out[out["Slope"] < 0].sort_values(["Slope"], ascending=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Upward Slope")
                st.dataframe(up.reset_index(drop=True), use_container_width=True) if not up.empty else st.info("No matches.")
            with right:
                st.subheader("Downward Slope")
                st.dataframe(dn.reset_index(drop=True), use_container_width=True) if not dn.empty else st.info("No matches.")


# ---------------------------
# TAB 11: Trendline Direction Lists
# ---------------------------
with tab11:
    st.header("Trendline Direction Lists")
    st.caption(
        "Displays symbols whose DAILY global trendline is:\n"
        "• **Upward** (slope >= 0) and **NPX < 0.0**\n"
        "• **Downward** (slope < 0) and **NPX > 0.5**\n\n"
        "Uses the selected Daily view range."
    )

    _require_yfinance()

    run_trend_lists = st.button("Run Trendline Direction Lists", key=f"btn_run_trendline_lists_{mode}")

    if run_trend_lists:
        up_rows, dn_rows = [], []
        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
                if not np.isfinite(npx_last):
                    continue

                if float(m) >= 0.0 and float(npx_last) < 0.0:
                    up_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})
                elif float(m) < 0.0 and float(npx_last) > 0.5:
                    dn_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("Upward Trend")
            if not up_rows:
                st.info("No matches.")
            else:
                st.dataframe(pd.DataFrame(up_rows).sort_values("Symbol").reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Downward Trend")
            if not dn_rows:
                st.info("No matches.")
            else:
                st.dataframe(pd.DataFrame(dn_rows).sort_values("Symbol").reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run Trendline Direction Lists** to scan the current universe.")


# ---------------------------
# TAB 12: NTD Hot List
# ---------------------------
with tab12:
    st.header("NTD Hot List")
    st.caption(
        "Lists symbols where:\n"
        "• Daily regression slope **> 0**\n"
        "• NPX between **0.0** and **0.5** (inclusive)\n"
        "Uses the selected Daily view range."
    )

    _require_yfinance()

    run_hot = st.button("Run NTD Hot List", key=f"btn_run_ntd_hot_{mode}")

    if run_hot:
        rows = []
        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m) or float(m) <= 0.0:
                    continue

                npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
                if not np.isfinite(npx_last):
                    continue

                if 0.0 <= float(npx_last) <= 0.5:
                    rows.append({
                        "Symbol": sym,
                        "Slope": float(m),
                        "NPX (Norm Price)": float(npx_last),
                        "R2": float(r2) if np.isfinite(r2) else np.nan,
                        "AsOf": ts
                    })

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Slope", "NPX (Norm Price)"], ascending=[False, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run NTD Hot List** to scan the current universe.")


# ---------------------------
# TAB 13: NTD NPX 0.0–0.2 Scanner
# ---------------------------
with tab13:
    st.header("NTD NPX 0.0–0.2 Scanner")
    st.caption(
        "Scans for symbols where **NPX is between 0.0 and 0.2** and **heading up**.\n"
        "Split into two lists:\n"
        "• List 1: slope > 0\n"
        "• List 2: slope < 0\n"
        "Includes NPX(last) and R²."
    )

    _require_yfinance()

    c1, c2 = st.columns(2)
    npx_up_bars = c1.slider("NPX heading-up confirmation (consecutive bars)", 1, 5, 1, 1, key="npx_02_up_bars")
    run_npx02 = c2.button("Run NTD NPX 0.0–0.2 Scan", key=f"btn_run_npx02_{mode}")

    if run_npx02:
        rows_up_slope, rows_dn_slope = [], []

        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                npx_s = daily_npx_series_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
                npx_s = _coerce_1d_series(npx_s).dropna()
                if npx_s.empty or len(npx_s) < 2:
                    continue

                npx_last = float(npx_s.iloc[-1]) if np.isfinite(npx_s.iloc[-1]) else np.nan
                if not np.isfinite(npx_last):
                    continue

                if not (0.0 <= float(npx_last) <= 0.2):
                    continue

                if not _series_heading_up(npx_s, confirm_bars=int(npx_up_bars)):
                    continue

                row = {
                    "Symbol": sym,
                    "Slope": float(m),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "NPX (Norm Price)": float(npx_last),
                    "AsOf": ts
                }

                if float(m) > 0.0:
                    rows_up_slope.append(row)
                elif float(m) < 0.0:
                    rows_dn_slope.append(row)

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)

        with left:
            st.subheader("List 1 — Slope > 0 and NPX 0.0–0.2 heading up")
            if not rows_up_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_up_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("List 2 — Slope < 0 and NPX 0.0–0.2 heading up")
            if not rows_dn_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_dn_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 14: Uptrend vs Downtrend
# ---------------------------
with tab14:
    st.header("Uptrend vs Downtrend")
    st.caption(
        "Lists symbols where price **reversed from support heading up**, split into:\n"
        "• (a) Uptrend: slope > 0\n"
        "• (b) Downtrend: slope < 0\n"
        "Uses the same Daily S/R proximity and confirmation bars."
    )

    _require_yfinance()

    c1, c2 = st.columns(2)
    hz_sr = c1.slider(
        "Support-touch lookback window (bars)",
        3, 60, int(max(3, rev_horizon)), 1,
        key="ud_sr_hz"
    )
    run_ud = c2.button("Run Uptrend vs Downtrend Scan", key=f"btn_run_ud_{mode}")

    if run_ud:
        rows_uptrend, rows_downtrend = [], []

        for sym in universe:
            try:
                m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
                if not np.isfinite(m):
                    continue

                rev = daily_support_reversal_heading_up(
                    symbol=sym,
                    daily_view_label=daily_view,
                    sr_lb=int(sr_lb_daily),
                    prox=float(sr_prox_pct),
                    bars_confirm=int(rev_bars_confirm),
                    horizon=int(hz_sr)
                )
                if rev is None:
                    continue

                row = dict(rev)
                row["Slope"] = float(m)
                row["R2"] = float(r2) if np.isfinite(r2) else np.nan
                row["AsOf"] = ts

                if float(m) > 0.0:
                    rows_uptrend.append(row)
                elif float(m) < 0.0:
                    rows_downtrend.append(row)

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("(a) Uptrend — Slope > 0 and Support Reversal heading up")
            if not rows_uptrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_uptrend)
                if "Bars Since Touch" in out.columns:
                    out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Downtrend — Slope < 0 and Support Reversal heading up")
            if not rows_downtrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_downtrend)
                if "Bars Since Touch" in out.columns:
                    out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 15: Ichimoku Kijun Scanner  (FIXED NameError causes)
# ---------------------------
with tab15:
    st.header("Ichimoku Kijun Scanner")
    st.caption(
        "Daily-only scanner:\n"
        "• List 1: slope > 0 AND price crossed above Kijun, heading up\n"
        "• List 2: slope < 0 AND price crossed above Kijun, heading up\n"
        "Includes Price@Cross, Kijun@Cross, Bars Since Cross, and R²."
    )

    _require_yfinance()

    c1, c2 = st.columns(2)
    kijun_within = c1.slider("Cross must be within last N bars", 0, 60, 5, 1, key="kijun_within_n")
    run_kijun = c2.button("Run Ichimoku Kijun Scan", key=f"btn_run_kijun_scan_{mode}")

    if run_kijun:
        rows_list1, rows_list2 = [], []
        for sym in universe:
            try:
                r = last_daily_kijun_cross_up(
                    symbol=sym,
                    daily_view_label=daily_view,
                    slope_lb=int(slope_lb_daily),
                    conv=int(ichi_conv),
                    base=int(ichi_base),
                    span_b=int(ichi_spanb),
                    within_last_n_bars=int(kijun_within),
                )
                if r is None:
                    continue

                m = _safe_float(r.get("Slope", np.nan), np.nan)
                if not np.isfinite(m):
                    continue

                if m > 0.0:
                    rows_list1.append(r)
                elif m < 0.0:
                    rows_list2.append(r)

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("List 1 — Slope > 0 and Kijun Cross-Up (heading up)")
            if not rows_list1:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list1)
                if "Bars Since Cross" in out.columns:
                    out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("List 2 — Slope < 0 and Kijun Cross-Up (heading up)")
            if not rows_list2:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list2)
                if "Bars Since Cross" in out.columns:
                    out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 16: R² > threshold Daily/Hourly
# ---------------------------
with tab16:
    st.header("R² > Threshold Daily/Hourly")
    st.caption(
        "Shows symbols where R² is above a threshold for:\n"
        "• Daily regression (lookback = Daily regression lookback)\n"
        "• Hourly regression (lookback = Hourly regression lookback)\n"
    )

    _require_yfinance()

    c1, c2, c3 = st.columns(3)
    r2_thr = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan")
    hour_period = c2.selectbox("Hourly period", ["1d", "2d", "4d"], index=0, key="r2_hour_period")
    run_r2 = c3.button("Run R² Scan", key=f"btn_run_r2_scan_{mode}")

    if run_r2:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            try:
                r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
                if np.isfinite(r2_d) and float(r2_d) > float(r2_thr):
                    daily_rows.append({
                        "Symbol": sym,
                        "R2": float(r2_d),
                        "Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                        "AsOf": ts_d
                    })

                r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period), slope_lb=int(slope_lb_hourly))
                if np.isfinite(r2_h) and float(r2_h) > float(r2_thr):
                    hourly_rows.append({
                        "Symbol": sym,
                        "R2": float(r2_h),
                        "Slope": float(m_h) if np.isfinite(m_h) else np.nan,
                        "AsOf": ts_h,
                        "Period": str(hour_period)
                    })

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("Daily — R² > threshold")
            if not daily_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly ({hour_period}) — R² > threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 17: R² < threshold Daily/Hourly
# ---------------------------
with tab17:
    st.header("R² < Threshold Daily/Hourly")
    st.caption(
        "Shows symbols where R² is below a threshold for:\n"
        "• Daily regression\n"
        "• Hourly regression\n"
    )

    _require_yfinance()

    c1, c2, c3 = st.columns(3)
    r2_thr_lo = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan_lo")
    hour_period_lo = c2.selectbox("Hourly period", ["1d", "2d", "4d"], index=0, key="r2_hour_period_lo")
    run_r2_lo = c3.button("Run R² Low Scan", key=f"btn_run_r2_scan_lo_{mode}")

    if run_r2_lo:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            try:
                r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
                if np.isfinite(r2_d) and float(r2_d) < float(r2_thr_lo):
                    daily_rows.append({
                        "Symbol": sym,
                        "R2": float(r2_d),
                        "Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                        "AsOf": ts_d
                    })

                r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period_lo), slope_lb=int(slope_lb_hourly))
                if np.isfinite(r2_h) and float(r2_h) < float(r2_thr_lo):
                    hourly_rows.append({
                        "Symbol": sym,
                        "R2": float(r2_h),
                        "Slope": float(m_h) if np.isfinite(m_h) else np.nan,
                        "AsOf": ts_h,
                        "Period": str(hour_period_lo)
                    })

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("Daily — R² < threshold")
            if not daily_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly ({hour_period_lo}) — R² < threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 18: SignedR² ±2σ Proximity (Daily)
# ---------------------------
with tab18:
    st.header("SignedR² ±2σ Proximity (Daily)")
    st.caption(
        "Daily-only scan. Creates four lists using **SignedR² = R² * sign(slope)**:\n"
        "1) SignedR² > 0 and Close near Lower -2σ\n"
        "2) SignedR² > 0 and Close near Upper +2σ\n"
        "3) SignedR² < 0 and Close near Lower -2σ\n"
        "4) SignedR² < 0 and Close near Upper +2σ\n"
        "“Near” uses sidebar S/R proximity (%)."
    )

    _require_yfinance()

    run_band_scan = st.button("Run SignedR² ±2σ Proximity Scan (Daily)", key=f"btn_run_r2_sign_band_scan_{mode}")

    if run_band_scan:
        rows_pos_lower, rows_pos_upper = [], []
        rows_neg_lower, rows_neg_upper = [], []

        for sym in universe:
            try:
                r = daily_r2_band_proximity(
                    symbol=sym,
                    daily_view_label=daily_view,
                    slope_lb=int(slope_lb_daily),
                    prox=float(sr_prox_pct)
                )
                if r is None:
                    continue

                signed_r2 = r.get("SignedR2", np.nan)
                if not np.isfinite(signed_r2):
                    continue

                near_lo = bool(r.get("Near Lower", False))
                near_up = bool(r.get("Near Upper", False))

                row = {k: v for k, v in r.items() if k not in ("Near Lower", "Near Upper")}

                if float(signed_r2) > 0.0:
                    if near_lo:
                        rows_pos_lower.append(row)
                    if near_up:
                        rows_pos_upper.append(row)
                elif float(signed_r2) < 0.0:
                    if near_lo:
                        rows_neg_lower.append(row)
                    if near_up:
                        rows_neg_upper.append(row)

                if scan_delay_ms:
                    time.sleep(scan_delay_ms / 1000.0)

            except Exception:
                continue

        st.info(f"Near threshold = ±{sr_prox_pct*100:.3f}% (from sidebar S/R proximity %)")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.subheader("SignedR² > 0  •  Near Lower -2σ")
            if not rows_pos_lower:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos_lower)
                if "AbsDist Lower (%)" in out.columns:
                    out = out.sort_values(["AbsDist Lower (%)", "SignedR2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with r1c2:
            st.subheader("SignedR² > 0  •  Near Upper +2σ")
            if not rows_pos_upper:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos_upper)
                if "AbsDist Upper (%)" in out.columns:
                    out = out.sort_values(["AbsDist Upper (%)", "SignedR2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.subheader("SignedR² < 0  •  Near Lower -2σ")
            if not rows_neg_lower:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg_lower)
                if "AbsDist Lower (%)" in out.columns:
                    out = out.sort_values(["AbsDist Lower (%)", "SignedR2"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with r2c2:
            st.subheader("SignedR² < 0  •  Near Upper +2σ")
            if not rows_neg_upper:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg_upper)
                if "AbsDist Upper (%)" in out.columns:
                    out = out.sort_values(["AbsDist Upper (%)", "SignedR2"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
