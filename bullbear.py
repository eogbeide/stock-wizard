# bullbear.py — Stocks/Forex Dashboard (Matplotlib) — GAPLESS FIX
# =============================================================================
# FIX (this update):
#   1) Removes fake flat "gap" segments caused by forward-filling missing bars.
#   2) Plots on a gapless x-axis (bar index) so time gaps (holiday/weekend/missing)
#      don't create long empty stretches.
#   3) Keeps readable time labels by formatting xticks with original timestamps.
#
# If you previously had: df = df.resample(...).ffill()
# That is the root cause of the flat "gap" you still see.
# =============================================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
st.set_page_config(page_title="BullBear — Gapless", layout="wide")

PST_TZ = "America/Los_Angeles"

FOREX_UNIVERSE = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
]
STOCK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY", "QQQ"]


# =========================
# Helpers
# =========================
def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df = df.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in df.columns])
    return df


def _to_pst_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance index can be tz-naive or tz-aware depending on interval.
    Normalize to tz-aware PST.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        # yfinance usually returns UTC-ish naive for intraday; localize as UTC then convert
        df.index = df.index.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward").tz_convert(PST_TZ)
    else:
        df.index = df.index.tz_convert(PST_TZ)
    return df


def wma(series: pd.Series, length: int) -> pd.Series:
    if length <= 1:
        return series.copy()
    w = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(series: pd.Series, length: int = 55) -> pd.Series:
    """
    Hull Moving Average:
      HMA(n) = WMA( 2*WMA(price, n/2) - WMA(price, n), sqrt(n) )
    """
    n = max(2, int(length))
    half = max(1, n // 2)
    root = max(1, int(round(math.sqrt(n))))
    w1 = wma(series, half)
    w2 = wma(series, n)
    raw = 2 * w1 - w2
    return wma(raw, root)


def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return lower, mid, upper


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def linreg_channel(close: pd.Series, window: int = 120, n_sigma: float = 2.0) -> Dict[str, float]:
    """
    Regression on last `window` bars; returns slope, intercept, sigma, r2.
    Values are in price units per bar.
    """
    s = close.dropna()
    if len(s) < window:
        window = max(10, len(s))
    y = s.iloc[-window:].values.astype(float)
    x = np.arange(len(y), dtype=float)

    # linear fit
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    resid = y - y_hat
    sigma = float(np.std(resid, ddof=0))

    # r2
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)

    return {
        "window": int(window),
        "slope": float(slope),
        "intercept": float(intercept),
        "sigma": float(sigma),
        "r2": float(r2),
        "n_sigma": float(n_sigma),
    }


def support_resistance(df: pd.DataFrame, window: int = 120) -> Tuple[float, float]:
    d = df.dropna()
    if len(d) < window:
        window = max(10, len(d))
    lo = float(d["Low"].iloc[-window:].min())
    hi = float(d["High"].iloc[-window:].max())
    return lo, hi


def ntd(close: pd.Series, win: int = 60) -> pd.Series:
    """
    Normalized Trend Direction (simple, stable):
      - compute rolling regression slope
      - normalize by rolling std of close
      - squash to [-1, 1] with tanh
    """
    close = close.astype(float)
    n = max(10, int(win))

    def _slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        y = arr.astype(float)
        if np.all(np.isfinite(y)) and len(y) >= 2:
            m, _b = np.polyfit(x, y, 1)
            return float(m)
        return np.nan

    slope = close.rolling(n).apply(lambda x: _slope(np.array(x)), raw=False)
    scale = close.rolling(n).std(ddof=0).replace(0, np.nan)
    z = slope / scale
    return np.tanh(z.fillna(0.0))


def npx(close: pd.Series, sup: pd.Series, res: pd.Series) -> pd.Series:
    """
    Normalized price within S/R: map to [-1, 1] where -1 ~ support, +1 ~ resistance.
    """
    rng = (res - sup).replace(0, np.nan)
    x = (close - sup) / rng
    x = (2 * x) - 1
    return x.clip(-1.25, 1.25).fillna(0.0)


def slope_reversal_probability(close: pd.Series, win: int = 120, lookahead: int = 15, lookback: int = 500) -> float:
    """
    Estimate P(slope reverses within <=lookahead bars) using past data.
    - compute rolling slope (win)
    - for each historical point, check if slope sign flips within next lookahead bars
    """
    s = close.dropna().astype(float)
    if len(s) < (win + lookahead + 20):
        return float("nan")

    def _roll_slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        m, _b = np.polyfit(x, arr.astype(float), 1)
        return float(m)

    slopes = s.rolling(win).apply(lambda x: _roll_slope(np.array(x)), raw=False).dropna()
    if len(slopes) < (lookahead + 20):
        return float("nan")

    slopes = slopes.iloc[-min(lookback, len(slopes)):]
    vals = slopes.values
    signs = np.sign(vals)
    # treat 0 as previous non-zero sign
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    hits = 0
    trials = 0
    for i in range(0, len(signs) - lookahead):
        s0 = signs[i]
        if s0 == 0:
            continue
        future = signs[i + 1 : i + 1 + lookahead]
        trials += 1
        if np.any(future == -s0):
            hits += 1

    if trials == 0:
        return float("nan")
    return hits / trials


# -------------------------
# GAPLESS FIX CORE
# -------------------------
def drop_empty_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    If you must resample, DO NOT forward-fill.
    This creates empty bins -> NaNs -> we DROP them.
    """
    df = _ensure_ohlcv(df)
    if df.empty:
        return df

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum" if "Volume" in df.columns else "last",
    }
    out = df.resample(rule).agg(agg)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])  # <<< NO FFILL
    return out


def add_gapless_x(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a gapless x-axis coordinate so plots have no time-gaps (holiday/weekend/missing bars).
    """
    df = df.copy()
    df["_x"] = np.arange(len(df), dtype=int)
    return df


def set_gapless_time_ticks(ax: plt.Axes, df: pd.DataFrame, n_ticks: int = 8) -> None:
    if df.empty:
        return
    n = len(df)
    n_ticks = int(np.clip(n_ticks, 4, 12))
    idx = np.linspace(0, n - 1, n_ticks).astype(int)
    xs = df["_x"].iloc[idx].values

    labels = []
    for i in idx:
        ts = df.index[i]
        # ts is tz-aware; display compact
        labels.append(ts.strftime("%m-%d %H:%M"))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)


def vline_at_time(ax: plt.Axes, df: pd.DataFrame, hh: int, mm: int, label: str) -> None:
    """
    Draw a vertical line at the nearest bar for each day at hh:mm PST.
    Uses gapless x.
    """
    if df.empty:
        return

    # group by date
    days = pd.Series(df.index.date).unique()
    first = True
    for d in days:
        day_mask = (df.index.date == d)
        sub = df.loc[day_mask]
        if sub.empty:
            continue

        target = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hh, minute=mm, tz=PST_TZ)
        # nearest bar
        deltas = (sub.index - target).to_series().abs()
        i = deltas.idxmin()
        x = int(sub.loc[i, "_x"])
        ax.axvline(x=x, linewidth=1, alpha=0.35, label=(label if first else None))
        first = False


# =========================
# Data
# =========================
@st.cache_data(ttl=60 * 5)
def fetch_ohlc(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = _ensure_ohlcv(df)
    df = _to_pst_index(df)
    df = df.sort_index()
    return df


def get_intraday(symbol: str) -> pd.DataFrame:
    """
    48h-ish intraday
    NOTE: we do NOT forward-fill any missing bars.
    """
    df = fetch_ohlc(symbol, interval="15m", period="5d")
    # If you want hourly plotting, resample WITHOUT FFILL:
    # df = drop_empty_resample(df, "1H")
    return df


def get_hourly(symbol: str) -> pd.DataFrame:
    df = fetch_ohlc(symbol, interval="60m", period="30d")
    return df


# =========================
# Plotting
# =========================
@dataclass
class PlotPack:
    fig: plt.Figure
    ax_price: plt.Axes
    ax_ind: plt.Axes


def build_figure() -> PlotPack:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.0, 2.0], hspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    return PlotPack(fig=fig, ax_price=ax1, ax_ind=ax2)


def plot_price_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> Dict[str, float]:
    """
    df must contain _x for gapless plotting.
    """
    close = df["Close"].astype(float)
    x = df["_x"].values

    # Indicators
    ema20 = ema(close, 20)
    hma55 = hma(close, 55)
    bb_l, bb_m, bb_u = bollinger(close, 20, 2.0)
    sup, res = support_resistance(df, 120)
    lr = linreg_channel(close, 120, 2.0)

    # Regression line and channel (over full plot using last-window params)
    n = len(df)
    line = lr["intercept"] + lr["slope"] * np.arange(n)
    ch_hi = line + lr["n_sigma"] * lr["sigma"]
    ch_lo = line - lr["n_sigma"] * lr["sigma"]

    # Plot
    ax.plot(x, close.values, linewidth=1.5, label="Intraday")
    ax.plot(x, ema20.values, linestyle="--", linewidth=1.2, label="EMA(20)")
    ax.plot(x, hma55.values, linewidth=1.4, label="HMA(55)")

    ax.plot(x, bb_u.values, linewidth=1.0, alpha=0.9, label="BB (×2.0)")
    ax.plot(x, bb_l.values, linewidth=1.0, alpha=0.9)
    ax.plot(x, bb_m.values, linewidth=1.0, alpha=0.9, label="BB mid (SMA, w=20)")

    ax.axhline(res, linewidth=1.6, label="Resistance")
    ax.axhline(sup, linewidth=1.6, label="Support")

    ax.plot(x, line, linewidth=2.0, label=f"Trend (global) ({lr['slope']:+.4f}/bar)")
    ax.plot(x, ch_hi, linestyle="--", linewidth=2.2, label="Slope +2σ")
    ax.plot(x, ch_lo, linestyle="--", linewidth=2.2, label="Slope -2σ")

    # Sessions (approx; will skip if data missing)
    vline_at_time(ax, df, 0, 0, "London Open (PST)")
    vline_at_time(ax, df, 8, 0, "London Close (PST)")
    vline_at_time(ax, df, 6, 30, "New York Open (PST)")
    vline_at_time(ax, df, 13, 0, "New York Close (PST)")

    # Current metrics
    cur = float(close.iloc[-1])
    nbb = float((cur - bb_m.iloc[-1]) / (bb_u.iloc[-1] - bb_m.iloc[-1] + 1e-12))
    pb = float((cur - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1] + 1e-12)) * 100.0

    # Reversal probability
    p_rev = slope_reversal_probability(close, win=120, lookahead=15, lookback=500)
    p_rev_txt = "nan" if not np.isfinite(p_rev) else f"{100*p_rev:.1f}%"

    trend_up = lr["slope"] >= 0
    up_pct = 100.0 if trend_up else 0.0
    dn_pct = 0.0 if trend_up else 100.0

    # Simple BUY/SELL reference (just for banner)
    # (Your real signal logic can stay; this doesn't affect the gap fix.)
    buy_ref = sup + 0.25 * (res - sup)
    sell_ref = res - 0.25 * (res - sup)
    pip = abs(sell_ref - buy_ref)
    pip_txt = f"{pip*10000:.1f} pips" if "USD" in title else f"{pip:.3f}"

    ax.set_title(f"{title}  ↑{up_pct:.1f}%  ↓{dn_pct:.1f}% — "
                 f"{'▲ BUY' if trend_up else '▼ SELL'} @{buy_ref:.3f} → "
                 f"{'▼ SELL' if trend_up else '▲ BUY'} @{sell_ref:.3f} • {pip_txt}  "
                 f"[P(slope rev≤15 bars)={p_rev_txt}]")

    ax.text(0.99, 0.02,
            f"Current price: {cur:.4f}  |  NBB {nbb:+.2f}  •  %B {pb:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.85))

    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("Price")

    return {
        "support": sup,
        "resistance": res,
        "slope": lr["slope"],
        "r2": lr["r2"],
        "nbb": nbb,
        "pb": pb,
    }


def plot_indicator_panel(ax: plt.Axes, df: pd.DataFrame, sr: Dict[str, float]) -> None:
    close = df["Close"].astype(float)
    x = df["_x"].values

    # rolling S/R series for NPX
    win = 60
    sup = df["Low"].rolling(win).min()
    res = df["High"].rolling(win).max()

    _ntd = ntd(close, win=60)
    _npx = npx(close, sup, res)

    # In-range shading (close between rolling S and R)
    in_range = (close >= sup) & (close <= res)
    ax.fill_between(x, -1.2, 1.2, where=in_range.fillna(False).values, alpha=0.12, label="In Range (S↔R)")

    ax.plot(x, _ntd.values, linewidth=2.0, label="NTD")
    ax.plot(x, _npx.values, linewidth=1.6, alpha=0.7, label="NPX (Norm Price)")

    # thresholds
    ax.axhline(0.0, linestyle="--", linewidth=1.2, label="0.00")
    ax.axhline(0.75, linewidth=1.2, alpha=0.85, label="+0.75")
    ax.axhline(-0.75, linewidth=1.2, alpha=0.85, label="-0.75")

    # Mark NTD threshold events
    ntd_up = (_ntd.shift(1) < 0.75) & (_ntd >= 0.75)
    ntd_dn = (_ntd.shift(1) > -0.75) & (_ntd <= -0.75)
    ax.scatter(df.loc[ntd_up.fillna(False), "_x"], _ntd.loc[ntd_up.fillna(False)], marker="v", s=90, label="NTD > +0.75")
    ax.scatter(df.loc[ntd_dn.fillna(False), "_x"], _ntd.loc[ntd_dn.fillna(False)], marker="^", s=90, label="NTD < -0.75")

    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Hourly Indicator Panel — NTD + NPX + Trend (win=60)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("Time (PST) — gapless bars")
    ax.set_ylabel("Norm")

    # gapless tick labels
    set_gapless_time_ticks(ax, df, n_ticks=8)


# =========================
# App UI
# =========================
st.title("BullBear — Gapless Charts (removes forward-fill gaps)")

colA, colB, colC = st.columns([1.2, 1.2, 1.0])

with colA:
    market = st.radio("Market", ["Forex", "Stocks"], horizontal=True, index=0)
with colB:
    universe = FOREX_UNIVERSE if market == "Forex" else STOCK_UNIVERSE
    symbol = st.selectbox("Symbol", universe, index=0)
with colC:
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = symbol
if "active_market" not in st.session_state:
    st.session_state.active_market = market

run = st.button("Run / Refresh", type="primary")

# Persist last-run selection (so it doesn't snap back)
if run:
    st.session_state.active_symbol = symbol
    st.session_state.active_market = market

active_symbol = st.session_state.active_symbol
active_market = st.session_state.active_market

st.caption(f"Active: **{active_market} — {active_symbol}**")

# =========================
# Compute + Plot
# =========================
df = get_intraday(active_symbol)
if df.empty or len(df) < 50:
    st.error("Not enough data returned. Try another symbol / timeframe.")
    st.stop()

# IMPORTANT: no resample ffill; if you resample, use drop_empty_resample(df, "1H") instead.
# df = drop_empty_resample(df, "1H")

# GAPLESS X for plotting (removes time gaps and prevents long empty/flat sections being shown as time)
df = add_gapless_x(df)

pack = build_figure()
sr = plot_price_panel(pack.ax_price, df, title=f"{active_symbol} — Intraday (gapless)")
plot_indicator_panel(pack.ax_ind, df, sr)

# small R2 badge similar to your screenshot
pack.ax_price.text(0.50, 0.02, f"R² (120 bars): {100*sr['r2']:.1f}%",
                   transform=pack.ax_price.transAxes, ha="center", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.25", alpha=0.75))

# show
st.pyplot(pack.fig, clear_figure=True)
