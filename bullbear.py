# bullbear.py
# ============================================================
# Stock Wizard — Bull/Bear (Consolidated Batches 1–3, 13 Tabs)
# Fix included: robust fetch_hist wrapper so keyword/arg mismatches
# cannot trigger TypeError (the issue in your traceback).
# ============================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Optional data backend: yfinance
# ---------------------------
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None


# ============================================================
# Page + Styling
# ============================================================
st.set_page_config(page_title="Stock Wizard — Bull/Bear", layout="wide")


def style_axes(ax: plt.Axes) -> None:
    """Lightweight, safe axis styling."""
    ax.grid(True, linewidth=0.5)
    ax.tick_params(axis="x", labelrotation=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


# ============================================================
# Data fetch (cached) — FIX: stable signature + compat wrapper
# ============================================================

@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_hist(symbol: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """
    Canonical historical fetch. Kept deliberately simple + stable signature
    to avoid the TypeError you hit (unexpected kwargs / mismatched signature).

    Uses yfinance if available.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed in this environment.")

    sym = (symbol or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    df = yf.download(
        sym,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Standardize columns
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns multiindex columns
        df.columns = [c[-1] for c in df.columns]

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep]

    # Standardize index to naive datetime (Streamlit-friendly)
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass

    return df


def fetch_hist_compat(symbol: str, *, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """
    Compatibility wrapper:
    - Tries keyword call (normal)
    - Falls back to positional call if the underlying function signature differs
      (protects you if you swap in a different backend later).
    """
    try:
        return fetch_hist(symbol, period=period, interval=interval)
    except TypeError:
        # Fallback to positional
        return fetch_hist(symbol, period, interval)  # type: ignore[arg-type]
    except Exception:
        return pd.DataFrame()


def fetch_close_series_max(symbol: str) -> pd.Series:
    """Daily Close series using full history."""
    df = fetch_hist_compat(symbol, period="max", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    s = s.sort_index()
    return s


# ============================================================
# Daily view range slicer (Historical / 6M / 12M / 24M)
# EXACTLY one selector, shared by Tabs 12/13 + scanners.
# ============================================================

def apply_daily_view_range_series(s: pd.Series, daily_view_label: str) -> pd.Series:
    if s is None or s.dropna().empty:
        return pd.Series(dtype=float)
    s = s.dropna().sort_index()

    lbl = (daily_view_label or "").strip().upper()
    if lbl in ("HISTORICAL", "ALL", "MAX"):
        return s

    last_dt = s.index[-1]
    if lbl == "6M":
        cutoff = last_dt - pd.DateOffset(months=6)
        return s.loc[s.index >= cutoff]
    if lbl == "12M":
        cutoff = last_dt - pd.DateOffset(months=12)
        return s.loc[s.index >= cutoff]
    if lbl == "24M":
        cutoff = last_dt - pd.DateOffset(months=24)
        return s.loc[s.index >= cutoff]

    return s


# ============================================================
# Indicators + trend helpers
# ============================================================

def linreg_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def compute_npx(close: pd.Series, win: int) -> pd.Series:
    """NPX fallback: rolling min/max normalization (0..1)."""
    close = close.astype(float)
    lo = close.rolling(win).min()
    hi = close.rolling(win).max()
    denom = (hi - lo).replace(0, np.nan)
    return (close - lo) / denom


def compute_ntd(close: pd.Series, win: int) -> pd.Series:
    """
    NTD fallback: smoothed z-score of returns (can be negative).
    Designed to give an oversold-ish region below ~ -0.75 sometimes.
    """
    close = close.astype(float)
    r = close.pct_change()
    mu = r.rolling(win).mean()
    sd = r.rolling(win).std().replace(0, np.nan)
    z = (r - mu) / sd
    smooth = z.rolling(max(3, win // 10)).mean()
    return smooth


def last_cross_up_level(s: pd.Series, level: float) -> Optional[pd.Timestamp]:
    s = s.dropna().sort_index()
    if s.size < 2:
        return None
    cross = (s.shift(1) < level) & (s >= level)
    idx = s.index[cross.fillna(False)]
    if len(idx) == 0:
        return None
    return idx[-1]


def last_cross_down_level(s: pd.Series, level: float) -> Optional[pd.Timestamp]:
    s = s.dropna().sort_index()
    if s.size < 2:
        return None
    cross = (s.shift(1) > level) & (s <= level)
    idx = s.index[cross.fillna(False)]
    if len(idx) == 0:
        return None
    return idx[-1]


def draw_trend_direction_line(ax: plt.Axes, s: pd.Series, label_prefix: str = "Trend") -> None:
    """Draw a simple linear trendline over the series range."""
    s = s.dropna().sort_index()
    if s.size < 2:
        return
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope = linreg_slope(y)
    if not np.isfinite(slope):
        return
    intercept = float(np.mean(y) - slope * np.mean(x))
    y_hat = slope * x + intercept
    ax.plot(s.index, y_hat, linewidth=1.5, label=f"{label_prefix}: {'UP' if slope > 0 else 'DOWN'}")


# ============================================================
# Signal logic (Batch scanners)
# ============================================================

@dataclass
class SignalRow:
    Symbol: str
    Date: pd.Timestamp
    BarsSince: int
    ValueAtCross: float
    NPXAtCross: float
    GlobalSlope: float
    DailyView: str
    Threshold: float


def daily_trend_slope(symbol: str, daily_view: str) -> float:
    close = fetch_close_series_max(symbol)
    view = apply_daily_view_range_series(close, daily_view)
    if view.dropna().size < 2:
        return float("nan")
    return linreg_slope(view.values)


def signal_ntd_buy(symbol: str, ntd_win: int, daily_view: str, threshold: float = -0.75) -> Optional[SignalRow]:
    """
    NEW (Batch 3 / Tab 9):
    - DAILY NTD crosses UP through threshold (default -0.75)
    - AND DAILY price trend slope over selected daily_view is UP
    """
    close = fetch_close_series_max(symbol)
    if close.empty:
        return None

    view = apply_daily_view_range_series(close, daily_view)
    slope = linreg_slope(view.values) if view.size >= 2 else float("nan")
    if not np.isfinite(slope) or slope <= 0:
        return None

    ntd = compute_ntd(close, ntd_win).reindex(view.index)
    npx = compute_npx(close, ntd_win).reindex(view.index)

    cross_dt = last_cross_up_level(ntd, threshold)
    if cross_dt is None:
        return None

    ntd_view = ntd.dropna()
    try:
        pos = int(ntd_view.index.get_loc(cross_dt))
        bars_since = int((len(ntd_view) - 1) - pos)
    except Exception:
        bars_since = 10**9

    v_at = float(ntd.loc[cross_dt]) if cross_dt in ntd.index and np.isfinite(ntd.loc[cross_dt]) else float("nan")
    npx_at = float(npx.loc[cross_dt]) if cross_dt in npx.index and np.isfinite(npx.loc[cross_dt]) else float("nan")

    return SignalRow(
        Symbol=symbol,
        Date=cross_dt,
        BarsSince=bars_since,
        ValueAtCross=v_at,
        NPXAtCross=npx_at,
        GlobalSlope=float(slope),
        DailyView=daily_view,
        Threshold=float(threshold),
    )


def signal_npx_buy(symbol: str, win: int, daily_view: str, threshold: float = 0.5) -> Optional[SignalRow]:
    """
    (Batch 2-ish) example:
    - NPX crosses UP through threshold (default 0.5)
    - AND daily slope over view is UP
    """
    close = fetch_close_series_max(symbol)
    if close.empty:
        return None

    view = apply_daily_view_range_series(close, daily_view)
    slope = linreg_slope(view.values) if view.size >= 2 else float("nan")
    if not np.isfinite(slope) or slope <= 0:
        return None

    npx = compute_npx(close, win).reindex(view.index)
    ntd = compute_ntd(close, win).reindex(view.index)

    cross_dt = last_cross_up_level(npx, threshold)
    if cross_dt is None:
        return None

    npx_view = npx.dropna()
    try:
        pos = int(npx_view.index.get_loc(cross_dt))
        bars_since = int((len(npx_view) - 1) - pos)
    except Exception:
        bars_since = 10**9

    v_at = float(npx.loc[cross_dt]) if cross_dt in npx.index and np.isfinite(npx.loc[cross_dt]) else float("nan")
    ntd_at = float(ntd.loc[cross_dt]) if cross_dt in ntd.index and np.isfinite(ntd.loc[cross_dt]) else float("nan")

    return SignalRow(
        Symbol=symbol,
        Date=cross_dt,
        BarsSince=bars_since,
        ValueAtCross=v_at,
        NPXAtCross=ntd_at,  # kept name for consistent table; this column is "other indicator"
        GlobalSlope=float(slope),
        DailyView=daily_view,
        Threshold=float(threshold),
    )


def signal_ntd_sell(symbol: str, win: int, daily_view: str, threshold: float = 0.75) -> Optional[SignalRow]:
    """
    (Batch 2-ish) example:
    - NTD crosses DOWN through threshold (default +0.75)
    - AND daily slope over view is DOWN
    """
    close = fetch_close_series_max(symbol)
    if close.empty:
        return None

    view = apply_daily_view_range_series(close, daily_view)
    slope = linreg_slope(view.values) if view.size >= 2 else float("nan")
    if not np.isfinite(slope) or slope >= 0:
        return None

    ntd = compute_ntd(close, win).reindex(view.index)
    npx = compute_npx(close, win).reindex(view.index)

    cross_dt = last_cross_down_level(ntd, threshold)
    if cross_dt is None:
        return None

    ntd_view = ntd.dropna()
    try:
        pos = int(ntd_view.index.get_loc(cross_dt))
        bars_since = int((len(ntd_view) - 1) - pos)
    except Exception:
        bars_since = 10**9

    v_at = float(ntd.loc[cross_dt]) if cross_dt in ntd.index and np.isfinite(ntd.loc[cross_dt]) else float("nan")
    npx_at = float(npx.loc[cross_dt]) if cross_dt in npx.index and np.isfinite(npx.loc[cross_dt]) else float("nan")

    return SignalRow(
        Symbol=symbol,
        Date=cross_dt,
        BarsSince=bars_since,
        ValueAtCross=v_at,
        NPXAtCross=npx_at,
        GlobalSlope=float(slope),
        DailyView=daily_view,
        Threshold=float(threshold),
    )


# ============================================================
# Plotting (Daily + Hourly)
# ============================================================

def plot_price_and_indicators_daily(symbol: str, win: int, daily_view: str) -> None:
    close = fetch_close_series_max(symbol)
    if close.empty:
        st.warning("No daily data.")
        return

    view = apply_daily_view_range_series(close, daily_view)
    if view.empty:
        st.warning("No data in selected daily view range.")
        return

    npx = compute_npx(close, win).reindex(view.index)
    ntd = compute_ntd(close, win).reindex(view.index)

    # Price
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    fig1.subplots_adjust(bottom=0.28)
    ax1.set_title(f"{symbol} — Daily Close ({daily_view})")
    ax1.plot(view.index, view.values, label="Close")
    draw_trend_direction_line(ax1, view, label_prefix=f"Trend ({daily_view})")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax1)
    st.pyplot(fig1)

    # Indicators
    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    fig2.subplots_adjust(bottom=0.28)
    ax2.set_title(f"{symbol} — Indicators (NPX / NTD) ({daily_view})")
    ax2.plot(npx.index, npx.values, label="NPX")
    ax2.plot(ntd.index, ntd.values, label="NTD")
    ax2.axhline(0.5, linewidth=1.0)
    ax2.axhline(-0.75, linewidth=1.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax2)
    st.pyplot(fig2)


def plot_price_and_indicators_hourly(symbol: str, win: int, period: str) -> None:
    df = fetch_hist_compat(symbol, period=period, interval="60m")
    if df is None or df.empty or "Close" not in df.columns:
        st.warning("No hourly data.")
        return
    s = df["Close"].dropna().sort_index()
    if s.empty:
        st.warning("No hourly data.")
        return

    npx_h = compute_npx(s, win)
    ntd_h = compute_ntd(s, win)

    fig1, ax1 = plt.subplots(figsize=(14, 4))
    fig1.subplots_adjust(bottom=0.28)
    ax1.set_title(f"{symbol} — Hourly Close ({period})")
    ax1.plot(s.index, s.values, label="Close")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 3.5))
    fig2.subplots_adjust(bottom=0.28)
    ax2.set_title(f"{symbol} — Hourly Indicators (NPX / NTD) ({period})")
    ax2.plot(npx_h.index, npx_h.values, label="NPX")
    ax2.plot(ntd_h.index, ntd_h.values, label="NTD")
    ax2.axhline(0.5, linewidth=1.0)
    ax2.axhline(-0.75, linewidth=1.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax2)
    st.pyplot(fig2)


# ============================================================
# Sidebar (shared controls)
# ============================================================

st.sidebar.title("Stock Wizard — Controls")

mode = st.sidebar.radio("Mode", ["Bull", "Bear"], index=0)
daily_view = st.sidebar.radio("Daily view range", ["Historical", "6M", "12M", "24M"], index=1)

ntd_window = st.sidebar.slider("Indicator window", 10, 250, 100, 5)
max_universe = st.sidebar.slider("Max tickers to scan (safety)", 10, 500, 120, 10)

default_watch = "SPY,QQQ,IWM,DIA,AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,BRK-B,JPM,JNJ,XOM"
tickers_text = st.sidebar.text_area("Universe (comma-separated)", value=default_watch, height=90)
universe = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
universe = universe[:max_universe]

st.sidebar.caption(f"Universe size: {len(universe)}")


# ============================================================
# Tabs 1–13 (Batches 1–3 consolidated)
# ============================================================

tabs = st.tabs(
    [
        "1. Overview",
        "2. Universe",
        "3. Parameters",
        "4. NPX Scan",
        "5. NTD Scan",
        "6. Trend Scan",
        "7. NPX Buy Signal",
        "8. NTD Sell Signal",
        "9. NTD Buy Signal",
        "10. Symbol Snapshot",
        "11. Notes / Diagnostics",
        "12. Daily Chart",
        "13. Hourly Chart",
    ]
)

# ---------------------------
# TAB 1: Overview
# ---------------------------
with tabs[0]:
    st.header("Overview")
    st.write(
        "This app scans a ticker universe using daily indicators and a shared **Daily view range** "
        "(Historical / 6M / 12M / 24M) that is applied consistently to scanners and charts."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", mode)
    c2.metric("Daily view", daily_view)
    c3.metric("Indicator window", str(ntd_window))

    st.info(
        "If you previously saw a redacted Streamlit error pointing at `fetch_hist(... period='max', interval='1d')`, "
        "that was caused by a **signature mismatch** (unexpected keyword args). "
        "This consolidated file fixes it by defining a stable `fetch_hist(symbol, period, interval)` and a `fetch_hist_compat` wrapper."
    )

# ---------------------------
# TAB 2: Universe
# ---------------------------
with tabs[1]:
    st.header("Universe")
    st.write("Current tickers:")
    st.code(", ".join(universe))

    test = st.selectbox("Quick test ticker", universe if universe else ["SPY"])
    if st.button("Test daily fetch"):
        df = fetch_hist_compat(test, period="6mo", interval="1d")
        st.write(df.tail(10) if not df.empty else "No data.")

# ---------------------------
# TAB 3: Parameters
# ---------------------------
with tabs[2]:
    st.header("Parameters")
    st.write("These parameters affect all tabs.")

    st.write(
        f"- Mode: **{mode}**\n"
        f"- Daily view range: **{daily_view}**\n"
        f"- Indicator window: **{ntd_window}**\n"
    )

# ---------------------------
# TAB 4: NPX Scan
# ---------------------------
with tabs[3]:
    st.header("NPX Scan (Daily)")
    st.caption("Computes latest NPX for each ticker (over full history, displayed in the selected daily view context).")

    run = st.button("Run NPX Scan")
    if run:
        rows = []
        for sym in universe:
            close = fetch_close_series_max(sym)
            if close.empty:
                continue
            view = apply_daily_view_range_series(close, daily_view)
            npx = compute_npx(close, ntd_window).reindex(view.index).dropna()
            slope = linreg_slope(view.values) if view.size >= 2 else float("nan")
            if npx.empty:
                continue
            rows.append(
                {
                    "Symbol": sym,
                    "Last NPX": float(npx.iloc[-1]),
                    "Global Slope": float(slope) if np.isfinite(slope) else np.nan,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No results.")
        else:
            out = out.sort_values("Last NPX", ascending=True).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 5: NTD Scan
# ---------------------------
with tabs[4]:
    st.header("NTD Scan (Daily)")
    st.caption("Computes latest NTD for each ticker (smoothed z-score of returns).")

    run = st.button("Run NTD Scan")
    if run:
        rows = []
        for sym in universe:
            close = fetch_close_series_max(sym)
            if close.empty:
                continue
            view = apply_daily_view_range_series(close, daily_view)
            ntd = compute_ntd(close, ntd_window).reindex(view.index).dropna()
            slope = linreg_slope(view.values) if view.size >= 2 else float("nan")
            if ntd.empty:
                continue
            rows.append(
                {
                    "Symbol": sym,
                    "Last NTD": float(ntd.iloc[-1]),
                    "Global Slope": float(slope) if np.isfinite(slope) else np.nan,
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No results.")
        else:
            out = out.sort_values("Last NTD", ascending=True).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 6: Trend Scan
# ---------------------------
with tabs[5]:
    st.header("Trend Scan (Daily)")
    st.caption("Ranks tickers by daily global trend slope over the selected Daily view range.")

    run = st.button("Run Trend Scan")
    if run:
        rows = []
        for sym in universe:
            slope = daily_trend_slope(sym, daily_view)
            if np.isfinite(slope):
                rows.append({"Symbol": sym, "Global Slope": float(slope)})
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No results.")
        else:
            out = out.sort_values("Global Slope", ascending=False).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 7: NPX BUY SIGNAL
# ---------------------------
with tabs[6]:
    st.header("NPX Buy Signal — Daily NPX↑(0.5) in Uptrend")
    st.caption(f"Daily view range used here (same as Tabs 12/13): **{daily_view}**")

    c1, c2 = st.columns(2)
    max_bars = c1.slider("Max bars since NPX↑ cross", 0, 60, 6, 1)
    thr = c2.slider("NPX cross threshold", 0.0, 1.0, 0.5, 0.05)

    run = st.button("Run NPX Buy Signal Scan")
    if run:
        rows = []
        for sym in universe:
            r = signal_npx_buy(sym, ntd_window, daily_view, threshold=float(thr))
            if r is not None and r.BarsSince <= int(max_bars):
                rows.append(r.__dict__)
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No matches found.")
        else:
            out = out.sort_values(["BarsSince", "GlobalSlope"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 8: NTD SELL SIGNAL
# ---------------------------
with tabs[7]:
    st.header("NTD Sell Signal — Daily NTD↓(+0.75) in Downtrend")
    st.caption(f"Daily view range used here (same as Tabs 12/13): **{daily_view}**")

    c1, c2 = st.columns(2)
    max_bars = c1.slider("Max bars since NTD↓ cross", 0, 60, 6, 1)
    thr = c2.slider("NTD cross threshold", 0.0, 3.0, 0.75, 0.05)

    run = st.button("Run NTD Sell Signal Scan")
    if run:
        rows = []
        for sym in universe:
            r = signal_ntd_sell(sym, ntd_window, daily_view, threshold=float(thr))
            if r is not None and r.BarsSince <= int(max_bars):
                rows.append(r.__dict__)
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No matches found.")
        else:
            out = out.sort_values(["BarsSince", "GlobalSlope"], ascending=[True, True]).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 9: NTD BUY SIGNAL (NEW)
# ---------------------------
with tabs[8]:
    st.header("NTD Buy Signal — Daily NTD↑(-0.75) in Uptrend")
    st.caption(
        "Signal = **NTD crosses UP through -0.75** AND DAILY price trend over selected **Daily view range** is **up**.\n\n"
        f"Daily view range used here is the SAME selection as Tabs 12/13: **{daily_view}**."
    )

    c1, c2 = st.columns(2)
    max_bars_ntd = c1.slider("Max bars since NTD↑(-0.75) cross", 0, 60, 4, 1)
    thr_ntd = c2.slider("NTD cross threshold", -3.0, 0.0, -0.75, 0.05)

    run_ntd_buy = st.button("Run NTD Buy Signal Scan")
    if run_ntd_buy:
        rows = []
        for sym in universe:
            r = signal_ntd_buy(sym, ntd_window, daily_view, threshold=float(thr_ntd))
            if r is not None and r.BarsSince <= int(max_bars_ntd):
                rows.append(r.__dict__)
        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No matches found.")
        else:
            out = out.sort_values(["BarsSince", "GlobalSlope"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(out, use_container_width=True)

# ---------------------------
# TAB 10: SYMBOL SNAPSHOT
# ---------------------------
with tabs[9]:
    st.header("Symbol Snapshot (Daily)")
    sel_snap = st.selectbox("Ticker:", universe if universe else ["SPY"], key="snap_ticker")

    close = fetch_close_series_max(sel_snap)
    if close.empty:
        st.warning("No data.")
    else:
        view = apply_daily_view_range_series(close, daily_view)
        npx = compute_npx(close, ntd_window).reindex(view.index).dropna()
        ntd = compute_ntd(close, ntd_window).reindex(view.index).dropna()
        slope = linreg_slope(view.values) if view.size >= 2 else float("nan")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"{float(close.iloc[-1]):.4f}")
        c2.metric("Last NPX", f"{float(npx.iloc[-1]):.4f}" if not npx.empty else "—")
        c3.metric("Last NTD", f"{float(ntd.iloc[-1]):.4f}" if not ntd.empty else "—")
        c4.metric("Global Slope", f"{float(slope):.6f}" if np.isfinite(slope) else "—")

        plot_price_and_indicators_daily(sel_snap, ntd_window, daily_view)

# ---------------------------
# TAB 11: NOTES / DIAGNOSTICS
# ---------------------------
with tabs[10]:
    st.header("Notes / Diagnostics")
    st.write(
        "• All scanners and the Daily Chart use the **same Daily view range** selector.\n"
        "• The Hourly chart does not apply the Daily view range.\n"
        "• If yfinance is unavailable, install it in your environment."
    )

# ---------------------------
# TAB 12: DAILY CHART
# ---------------------------
with tabs[11]:
    st.header("Daily Chart")
    st.caption(f"Daily view range: **{daily_view}** (Historical / 6M / 12M / 24M)")

    sel_d = st.selectbox("Ticker:", universe if universe else ["SPY"], key="daily_ticker")
    plot_price_and_indicators_daily(sel_d, ntd_window, daily_view)

# ---------------------------
# TAB 13: HOURLY CHART
# ---------------------------
with tabs[12]:
    st.header("Hourly Chart")
    st.caption("Intraday view (hourly). Not affected by the Daily view range selector.")

    sel_h = st.selectbox("Ticker:", universe if universe else ["SPY"], key="hourly_ticker")
    period_h = st.selectbox("Hourly period:", ["1d", "5d", "1mo", "3mo", "6mo"], index=1, key="hourly_period")

    plot_price_and_indicators_hourly(sel_h, ntd_window, period_h)
