# =========================
# Part 1/6 — bullbear.py
# =========================
# bullbear.py — Stocks/Forex Dashboard + Forecasts
# =========================
# UPDATE: Single latest band-reversal signal for trading
#   • Uptrend  → BUY when price reverses up from lower ±2σ band and is near it
#   • Downtrend→ SELL when price reverses down from upper ±2σ band and is near it
# Only the latest signal is shown at any time.
#
# FIX (This update):
#   • Prevent widgets from snapping back to defaults after reruns/auto-refresh.
#   • Fix NameError: render_daily_price_macd / render_intraday_price_macd are defined before use.
#
# Notes:
#   - This script is self-contained. Copy/paste into bullbear.py.
#   - Uses matplotlib for charts (works well on Streamlit Cloud).

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------
# Page / CSS
# -------------------------
st.set_page_config(page_title="📊 BullBear Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
<style>
#MainMenu, header, footer {visibility: hidden;}
div[data-testid="stSidebar"] { min-width: 330px; }
</style>
""",
    unsafe_allow_html=True,
)

PACIFIC = pytz.timezone("US/Pacific")
REFRESH_INTERVAL = 120  # seconds

# -------------------------
# Session-state defaults (ONE TIME ONLY)
# -------------------------
def init_defaults_once():
    """
    Streamlit reruns the script frequently (button clicks, widget changes, auto-refresh).
    If you assign defaults on every rerun, widgets can jump back to their defaults.
    This initializer sets defaults exactly once per user session.
    """
    if st.session_state.get("_defaults_inited", False):
        return

    defaults = dict(
        mode="Forex",
        daily_view="12M",
        news_window_days=7,
        show_fx_news=False,

        show_hma=True,
        hma_period=55,

        show_bbands=True,
        bb_win=20,
        bb_mult=2.0,
        bb_use_ema=False,

        show_ichi=False,
        ichi_base=26,

        show_psar=True,
        psar_step=0.02,
        psar_max=0.2,

        atr_period=10,
        atr_mult=3.0,

        slope_lb_daily=252,
        slope_lb_hourly=200,
        sr_lb_hourly=120,
        sr_prox_pct=0.25,       # as a percent in UI; convert to fraction in code
        rev_bars_confirm=2,

        show_fibs=True,
        show_sessions_pst=True,

        bb_period="1y",

        orig_chart="Daily",
        hour_range_select="24h",

        # scanners
        macd_hot_recent_bars=30,
        macd_hma_recent_bars=7,
        ntd_scan_hour_range="24h",

        # last symbols per mode
        last_symbol_stocks="AAPL",
        last_symbol_forex="EURUSD=X",
    )

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.session_state["_defaults_inited"] = True

def ensure_key_in_options(key: str, options: list[str], fallback):
    """If widget key doesn't exist or its value isn't valid anymore, reset it safely."""
    if key not in st.session_state or st.session_state.get(key) not in options:
        st.session_state[key] = fallback

init_defaults_once()

# -------------------------
# Auto-refresh
# -------------------------
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        safe_rerun()

auto_refresh()

elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# -------------------------
# Universe
# -------------------------
def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

mode = st.sidebar.radio("Mode", ["Stocks", "Forex"], index=(0 if st.session_state.mode == "Stocks" else 1), key="mode")

if mode == "Stocks":
    universe = sorted(dedup_keep_order([
        "AAPL","SPY","AMZN","DIA","TSLA","SPGI","JPM","VTWG","PLTR","NVDA",
        "META","SITM","MARA","GOOG","HOOD","BABA","IBM","AVGO","GUSH","VOO",
        "MSFT","TSM","NFLX","MP","AAL","URI","DAL","BBAI","QUBT","AMD","SMCI","ORCL","TLT"
    ]))
else:
    universe = dedup_keep_order([
        "EURUSD=X","EURJPY=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","NZDJPY=X",
        "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X",
        "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X"
    ])

# -------------------------
# Sidebar settings (keys persist via session_state)
# -------------------------
st.sidebar.markdown("---")
daily_view = st.sidebar.selectbox("Daily view window", ["6M", "12M", "24M", "Historical"], key="daily_view")
news_window_days = int(st.sidebar.slider("News window (days)", 1, 21, int(st.session_state.news_window_days), 1, key="news_window_days"))
show_fx_news = st.sidebar.checkbox("Show Yahoo Finance news (Forex only)", value=bool(st.session_state.show_fx_news), key="show_fx_news")

st.sidebar.markdown("### Indicators")
show_hma = st.sidebar.checkbox("Show HMA(55)", value=bool(st.session_state.show_hma), key="show_hma")
hma_period = int(st.sidebar.number_input("HMA period", min_value=10, max_value=200, value=int(st.session_state.hma_period), step=1, key="hma_period"))

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=bool(st.session_state.show_bbands), key="show_bbands")
bb_win = int(st.sidebar.number_input("BB window", min_value=10, max_value=200, value=int(st.session_state.bb_win), step=1, key="bb_win"))
bb_mult = float(st.sidebar.number_input("BB multiplier", min_value=1.0, max_value=5.0, value=float(st.session_state.bb_mult), step=0.1, key="bb_mult"))
bb_use_ema = st.sidebar.checkbox("BB midline uses EMA", value=bool(st.session_state.bb_use_ema), key="bb_use_ema")

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=bool(st.session_state.show_ichi), key="show_ichi")
ichi_base = int(st.sidebar.number_input("Ichimoku base (Kijun)", 10, 60, value=int(st.session_state.ichi_base), step=1, key="ichi_base"))

show_psar = st.sidebar.checkbox("Show PSAR", value=bool(st.session_state.show_psar), key="show_psar")
psar_step = float(st.sidebar.number_input("PSAR step", min_value=0.001, max_value=0.1, value=float(st.session_state.psar_step), step=0.005, key="psar_step"))
psar_max = float(st.sidebar.number_input("PSAR max step", min_value=0.05, max_value=0.5, value=float(st.session_state.psar_max), step=0.01, key="psar_max"))

st.sidebar.markdown("### SuperTrend")
atr_period = int(st.sidebar.number_input("ATR period", min_value=5, max_value=50, value=int(st.session_state.atr_period), step=1, key="atr_period"))
atr_mult = float(st.sidebar.number_input("ATR multiplier", min_value=1.0, max_value=10.0, value=float(st.session_state.atr_mult), step=0.25, key="atr_mult"))

st.sidebar.markdown("### Regression / Signals")
slope_lb_daily = int(st.sidebar.number_input("Daily regression lookback (bars)", 30, 600, value=int(st.session_state.slope_lb_daily), step=1, key="slope_lb_daily"))
slope_lb_hourly = int(st.sidebar.number_input("Hourly regression lookback (bars)", 30, 800, value=int(st.session_state.slope_lb_hourly), step=1, key="slope_lb_hourly"))
sr_lb_hourly = int(st.sidebar.number_input("Hourly S/R lookback (bars)", 20, 600, value=int(st.session_state.sr_lb_hourly), step=1, key="sr_lb_hourly"))
sr_prox_pct = float(st.sidebar.slider("S/R / Band proximity (%)", 0.0, 2.0, float(st.session_state.sr_prox_pct), 0.05, key="sr_prox_pct")) / 100.0
rev_bars_confirm = int(st.sidebar.slider("Consecutive bars to confirm", 1, 5, int(st.session_state.rev_bars_confirm), 1, key="rev_bars_confirm"))

st.sidebar.markdown("### Extras")
show_fibs = st.sidebar.checkbox("Show Fibonacci (Intraday)", value=bool(st.session_state.show_fibs), key="show_fibs")
show_sessions_pst = st.sidebar.checkbox("Show session lines (Forex)", value=bool(st.session_state.show_sessions_pst), key="show_sessions_pst")

st.sidebar.markdown("### Bull/Bear / Metrics lookback")
bb_period = st.sidebar.selectbox("Lookback period (yfinance)", ["6mo", "1y", "2y", "5y", "max"], key="bb_period")

# -------------------------
# Finance utilities
# -------------------------
def ensure_tz_index(idx: pd.DatetimeIndex, tz) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return idx
    if idx.tz is None:
        return idx.tz_localize(tz)
    return idx.tz_convert(tz)
# =========================
# Part 2/6 — bullbear.py
# =========================
@st.cache_data(ttl=120)
def fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = ensure_tz_index(df.index, PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "7d", interval: str = "60m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = ensure_tz_index(df.index, PACIFIC)
    return df

def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma(s: pd.Series, n: int = 55) -> pd.Series:
    n = int(n)
    if n < 2:
        return s
    return wma(2 * wma(s, n // 2) - wma(s, n), int(np.sqrt(n)))

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=int(span), adjust=False).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def linear_regression_channel(y: pd.Series, lookback: int):
    y = y.dropna()
    if len(y) < max(20, lookback):
        lookback = max(20, min(len(y), lookback))
    y2 = y.iloc[-lookback:]
    x = np.arange(len(y2), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y2.values, rcond=None)[0]
    yhat = m * x + b
    resid = y2.values - yhat
    sigma = float(np.nanstd(resid)) if len(resid) else np.nan
    ss_res = float(np.nansum(resid**2))
    ss_tot = float(np.nansum((y2.values - np.nanmean(y2.values))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    yhat_s = pd.Series(yhat, index=y2.index)
    upper = yhat_s + 2*sigma
    lower = yhat_s - 2*sigma
    return dict(m=m, b=b, r2=r2, mid=yhat_s, upper=upper, lower=lower, sigma=sigma, window=lookback)

def atr(df: pd.DataFrame, n: int = 10) -> pd.Series:
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(int(n)).mean()

def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    a = atr(df, n)
    hl2 = (df["High"] + df["Low"]) / 2.0
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    trend = pd.Series(index=df.index, dtype=int)
    st_line = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = 1
    st_line.iloc[0] = lower.iloc[0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif df["Close"].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
            if trend.iloc[i] == 1:
                lower.iloc[i] = max(lower.iloc[i], lower.iloc[i-1])
            else:
                upper.iloc[i] = min(upper.iloc[i], upper.iloc[i-1])
        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    return st_line, trend

def psar(high: pd.Series, low: pd.Series, step=0.02, max_step=0.2) -> pd.Series:
    # Basic PSAR implementation
    h = high.values
    l = low.values
    out = np.zeros_like(h, dtype=float)
    bull = True
    af = step
    ep = l[0]
    sar = h[0]
    out[0] = sar
    for i in range(1, len(h)):
        prev_sar = sar
        if bull:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, l[i-1], l[i])
            if h[i] > ep:
                ep = h[i]
                af = min(af + step, max_step)
            if l[i] < sar:
                bull = False
                sar = ep
                ep = l[i]
                af = step
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, h[i-1], h[i])
            if l[i] < ep:
                ep = l[i]
                af = min(af + step, max_step)
            if h[i] > sar:
                bull = True
                sar = ep
                ep = h[i]
                af = step
        out[i] = sar
    return pd.Series(out, index=high.index)

def bollinger(close: pd.Series, win=20, mult=2.0, use_ema=False):
    if use_ema:
        mid = close.ewm(span=int(win), adjust=False).mean()
    else:
        mid = close.rolling(int(win)).mean()
    sd = close.rolling(int(win)).std()
    upper = mid + mult * sd
    lower = mid - mult * sd
    return mid, upper, lower
# =========================
# Part 3/6 — bullbear.py
# =========================
def fmt_price(y: float) -> str:
    try:
        return f"{float(y):,.3f}"
    except Exception:
        return "n/a"

def pip_size_for_symbol(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return None
    return 0.01 if "JPY" in s else 0.0001

def diff_text(a: float, b: float, symbol: str) -> str:
    try:
        av = float(a); bv = float(b)
    except Exception:
        return ""
    ps = pip_size_for_symbol(symbol)
    d = abs(bv - av)
    if ps:
        return f"{d/ps:.1f} pips"
    return f"Δ {d:.3f}"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans, ha="left", va="center",
        color=color, fontsize=fontsize, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        zorder=6
    )

def pad_right_xaxis(ax, frac: float = 0.06):
    try:
        left, right = ax.get_xlim()
        span = right - left
        ax.set_xlim(left, right + span * float(frac))
    except Exception:
        pass

def subset_by_daily_view(df: pd.DataFrame, view_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    end = df.index.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        start = df.index.min()
    else:
        start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return df.loc[(df.index >= start) & (df.index <= end)]

def find_support_resistance(series: pd.Series, lookback: int):
    s = series.dropna()
    if len(s) < 10:
        return (np.nan, np.nan)
    w = s.iloc[-lookback:]
    sup = float(w.min())
    res = float(w.max())
    return (sup, res)

def band_reversal_signal(close: pd.Series, ch: dict, prox_frac: float, confirm_bars: int):
    """Return latest (side, idx, price) or (None, None, None)."""
    if close is None or close.empty:
        return (None, None, None)
    lb = ch["lower"].reindex(close.index).ffill()
    ub = ch["upper"].reindex(close.index).ffill()
    slope = float(ch["m"])
    px = close

    rng = (ub - lb).replace(0, np.nan)
    prox = prox_frac * rng

    side = None
    ix = None
    price = None

    if np.isfinite(slope) and slope > 0:
        cond = (px.shift(1) < (lb.shift(1))) & (px >= lb) & ((px - lb).abs() <= prox)
        if confirm_bars > 1:
            ok = px.rolling(confirm_bars).apply(lambda x: int(np.all(np.diff(x) >= 0)), raw=True).astype(bool)
            cond = cond & ok
        hits = np.where(cond.fillna(False).values)[0]
        if len(hits):
            k = hits[-1]
            side, ix, price = "BUY", px.index[k], float(px.iloc[k])

    if (side is None) and (np.isfinite(slope) and slope < 0):
        cond = (px.shift(1) > (ub.shift(1))) & (px <= ub) & ((px - ub).abs() <= prox)
        if confirm_bars > 1:
            ok = px.rolling(confirm_bars).apply(lambda x: int(np.all(np.diff(x) <= 0)), raw=True).astype(bool)
            cond = cond & ok
        hits = np.where(cond.fillna(False).values)[0]
        if len(hits):
            k = hits[-1]
            side, ix, price = "SELL", px.index[k], float(px.iloc[k])

    return (side, ix, price)

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             confirm_side: str | None = None) -> str:
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    cs = (confirm_side or "").upper()
    buy_lbl  = "▲ BUY"  + (" (CONFIRMED)" if cs == "BUY"  else "")
    sell_lbl = "▼ SELL" + (" (CONFIRMED)" if cs == "SELL" else "")

    buy_txt  = f"{buy_lbl} @{fmt_price(entry_buy)}"
    sell_txt = f"{sell_lbl} @{fmt_price(exit_sell)}"
    pips_txt = f" • Value of PIPS: {diff_text(exit_sell, entry_buy, symbol)}"

    try:
        tslope = float(trend_slope)
    except Exception:
        tslope = 0.0

    if np.isfinite(tslope) and tslope > 0:
        return f"{buy_txt} → {sell_txt}{pips_txt}"
    else:
        return f"{sell_txt} → {buy_txt}{pips_txt}"

def draw_instruction_banner(ax,
                            trend_slope: float,
                            support: float,
                            resistance: float,
                            close_val: float,
                            symbol: str,
                            confirm_side: str | None = None,
                            global_slope: float | None = None,
                            extra_note: str | None = None):
    aligned = True
    if global_slope is not None and np.isfinite(global_slope) and np.isfinite(trend_slope):
        aligned = (global_slope * trend_slope) > 0

    if not aligned:
        ax.text(
            0.5, 1.08,
            "ALERT: Global Trendline and Local Slope are opposing — no trade instruction.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.95)
        )
        return

    color = "tab:green" if (np.isfinite(trend_slope) and trend_slope > 0) else "tab:red"
    instr = format_trade_instruction(trend_slope, support, resistance, close_val, symbol, confirm_side=confirm_side)
    if extra_note:
        instr = f"{instr}\n{extra_note}"

    ax.text(
        0.5, 1.08, instr,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95)
    )
# =========================
# Part 4/6 — bullbear.py
# =========================
# -------------------------
# Plot functions (FIX: defined before use)
# -------------------------
def render_daily_price_macd(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No daily data.")
        return

    d = subset_by_daily_view(df, daily_view)
    close = d["Close"]

    ch = linear_regression_channel(close, slope_lb_daily)
    support, resistance = find_support_resistance(close, slope_lb_daily)

    macd_line, sig_line, hist = macd(close)
    mid, bb_u, bb_l = bollinger(close, bb_win, bb_mult, bb_use_ema) if show_bbands else (None, None, None)
    hma_line = hma(close, hma_period) if show_hma else None

    side, sig_ix, sig_px = band_reversal_signal(close, ch, sr_prox_pct, rev_bars_confirm)

    extra_note = None
    if (np.isfinite(ch["m"]) and ch["m"] < 0) and show_hma:
        macd_cross_dn = (macd_line.shift(1) > 0) & (macd_line <= 0)
        hma_cross_dn = (close.shift(1) > hma_line.shift(1)) & (close <= hma_line)
        if bool((macd_cross_dn & hma_cross_dn).iloc[-1]):
            extra_note = "MACD Sell — HMA55 Cross"

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(close.index, close.values, linewidth=1.3, label="Close")
    ax.plot(ch["mid"].index, ch["mid"].values, linestyle="--", linewidth=1.6, label=f"Trend (m={ch['m']:.4f}, R²={ch['r2']:.2f})")
    ax.plot(ch["upper"].index, ch["upper"].values, linewidth=1.2, alpha=0.8)
    ax.plot(ch["lower"].index, ch["lower"].values, linewidth=1.2, alpha=0.8)

    if show_bbands and mid is not None:
        ax.plot(mid.index, mid.values, linewidth=1.0, alpha=0.9)
        ax.plot(bb_u.index, bb_u.values, linewidth=0.9, alpha=0.7)
        ax.plot(bb_l.index, bb_l.values, linewidth=0.9, alpha=0.7)

    if show_hma and hma_line is not None:
        ax.plot(hma_line.index, hma_line.values, linewidth=1.1, alpha=0.95, label=f"HMA({hma_period})")

    if show_ichi:
        kijun = (d["High"].rolling(ichi_base).max() + d["Low"].rolling(ichi_base).min()) / 2.0
        ax.plot(kijun.index, kijun.values, linewidth=1.0, alpha=0.9, label=f"Kijun({ichi_base})")

    if show_psar:
        ps = psar(d["High"], d["Low"], step=psar_step, max_step=psar_max)
        ax.scatter(ps.index, ps.values, s=8, alpha=0.7, label="PSAR")

    st_line, st_trend = supertrend(d, n=atr_period, mult=atr_mult)
    ax.plot(st_line.index, st_line.values, linewidth=1.2, alpha=0.9, label="SuperTrend")

    if np.isfinite(support):
        ax.axhline(support, linestyle=":", linewidth=1.2, alpha=0.9)
        label_on_left(ax, support, f"SUP {fmt_price(support)}")
    if np.isfinite(resistance):
        ax.axhline(resistance, linestyle=":", linewidth=1.2, alpha=0.9)
        label_on_left(ax, resistance, f"RES {fmt_price(resistance)}")

    if sig_ix is not None and sig_ix in close.index:
        ax.scatter([sig_ix], [sig_px], marker="^" if side == "BUY" else "v", s=90, zorder=10)
        ax.annotate(f"{side} Band REV", xy=(sig_ix, sig_px),
                    xytext=(15, -30), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", alpha=0.7))

    draw_instruction_banner(ax, float(ch["m"]), support, resistance, float(close.iloc[-1]), symbol, confirm_side=side, extra_note=extra_note)

    ax.set_title(f"{symbol} — Daily")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    pad_right_xaxis(ax)

    ax2.axhline(0, linewidth=1.0, alpha=0.8)
    ax2.plot(macd_line.index, macd_line.values, linewidth=1.2, label="MACD")
    ax2.plot(sig_line.index, sig_line.values, linewidth=1.0, alpha=0.9, label="Signal")
    ax2.bar(hist.index, hist.values, width=1.0, alpha=0.3, label="Hist")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylabel("MACD")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def render_intraday_price_macd(symbol: str, df_ohlc: pd.DataFrame):
    if df_ohlc is None or df_ohlc.empty:
        st.warning("No intraday data.")
        return

    close = df_ohlc["Close"]
    ch = linear_regression_channel(close, slope_lb_hourly)
    support, resistance = find_support_resistance(close, sr_lb_hourly)

    macd_line, sig_line, hist = macd(close)
    hma_line = hma(close, hma_period) if show_hma else None
    mid, bb_u, bb_l = bollinger(close, bb_win, bb_mult, bb_use_ema) if show_bbands else (None, None, None)

    side, sig_ix, sig_px = band_reversal_signal(close, ch, sr_prox_pct, rev_bars_confirm)

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(close.index, close.values, linewidth=1.3, label="Close")
    ax.plot(ch["mid"].index, ch["mid"].values, linestyle="--", linewidth=1.6, label=f"Trend (m={ch['m']:.4f}, R²={ch['r2']:.2f})")
    ax.plot(ch["upper"].index, ch["upper"].values, linewidth=1.2, alpha=0.8)
    ax.plot(ch["lower"].index, ch["lower"].values, linewidth=1.2, alpha=0.8)

    if show_bbands and mid is not None:
        ax.plot(mid.index, mid.values, linewidth=1.0, alpha=0.9)
        ax.plot(bb_u.index, bb_u.values, linewidth=0.9, alpha=0.7)
        ax.plot(bb_l.index, bb_l.values, linewidth=0.9, alpha=0.7)

    if show_hma and hma_line is not None:
        ax.plot(hma_line.index, hma_line.values, linewidth=1.1, alpha=0.95, label=f"HMA({hma_period})")

    if np.isfinite(support):
        ax.axhline(support, linestyle=":", linewidth=1.2, alpha=0.9)
        label_on_left(ax, support, f"SUP {fmt_price(support)}")
    if np.isfinite(resistance):
        ax.axhline(resistance, linestyle=":", linewidth=1.2, alpha=0.9)
        label_on_left(ax, resistance, f"RES {fmt_price(resistance)}")

    if show_fibs and np.isfinite(support) and np.isfinite(resistance):
        lo, hi = support, resistance
        if hi > lo:
            fibs = [0.236, 0.382, 0.5, 0.618, 0.786]
            for f in fibs:
                y = hi - (hi - lo) * f
                ax.axhline(y, linewidth=0.8, alpha=0.25)
                ax.text(1.005, y, f"{int(f*100)}%", transform=blended_transform_factory(ax.transAxes, ax.transData),
                        fontsize=8, va="center")

    if mode == "Forex" and show_sessions_pst:
        sess_hours = [0, 6, 16]  # London, NY, Tokyo (approx PST anchors)
        for h in sess_hours:
            try:
                xs = [t for t in close.index if getattr(t, "hour", None) == h and getattr(t, "minute", None) == 0]
                if xs:
                    ax.axvline(xs[-1], linewidth=1.0, alpha=0.25)
            except Exception:
                pass

    if sig_ix is not None and sig_ix in close.index:
        ax.scatter([sig_ix], [sig_px], marker="^" if side == "BUY" else "v", s=90, zorder=10)
        ax.annotate(f"{side} Band REV", xy=(sig_ix, sig_px),
                    xytext=(15, -30), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", alpha=0.7))

    draw_instruction_banner(ax, float(ch["m"]), support, resistance, float(close.iloc[-1]), symbol, confirm_side=side)

    ax.set_title(f"{symbol} — Intraday (60m)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.2)
    pad_right_xaxis(ax)

    ax2.axhline(0, linewidth=1.0, alpha=0.8)
    ax2.plot(macd_line.index, macd_line.values, linewidth=1.2, label="MACD")
    ax2.plot(sig_line.index, sig_line.values, linewidth=1.0, alpha=0.9, label="Signal")
    ax2.bar(hist.index, hist.values, width=0.03, alpha=0.3, label="Hist")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylabel("MACD")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
# =========================
# Part 5/6 — bullbear.py
# =========================
# -------------------------
# Forecasts
# -------------------------
def forecast_sarimax(series: pd.Series, steps: int = 14):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 60:
        return None
    try:
        model = SARIMAX(s, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        fc = pred.predicted_mean
        conf = pred.conf_int()
        return fc, conf
    except Exception:
        return None

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Chart & Forecast",
    "Breakout / Breakdown",
    "Trend / Metrics",
    "News",
    "MACD + HMA Scanner",
    "Long-Term Trend",
    "NTD Scanner"
])

# -------------------------
# Tab 1 — Chart & Forecast
# -------------------------
with tab1:
    st.subheader("Chart & Forecast")

    last_for_mode = st.session_state.last_symbol_stocks if mode == "Stocks" else st.session_state.last_symbol_forex
    if "ticker" in st.session_state and st.session_state.get("ticker") in universe:
        fallback = st.session_state.get("ticker")
    else:
        fallback = last_for_mode if last_for_mode in universe else universe[0]

    ensure_key_in_options("tab1_ticker", universe, fallback)
    sel = st.selectbox("Ticker:", universe, key="tab1_ticker")

    if mode == "Stocks":
        st.session_state.last_symbol_stocks = sel
    else:
        st.session_state.last_symbol_forex = sel

    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly range", ["24h", "3d", "7d", "30d", "90d"], key="hour_range_select")

    run = st.button("Run Forecast", type="primary")
    auto_run = bool(st.session_state.get("run_all", False))

    if run or auto_run:
        st.session_state.run_all = True
        st.session_state.ticker = sel
        st.session_state.chart = chart
        st.session_state.hour_range = hour_range

    if bool(st.session_state.get("run_all", False)) and st.session_state.get("ticker") == sel:
        df_daily = fetch_daily(sel)
        df_hourly = fetch_intraday(sel, period=hour_range, interval="60m")

        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown("#### Daily")
            render_daily_price_macd(sel, df_daily)

        with colB:
            st.markdown("#### Hourly")
            render_intraday_price_macd(sel, df_hourly)

        st.markdown("---")
        st.markdown("#### Forecast (Daily Close)")
        if df_daily is not None and not df_daily.empty:
            fc = forecast_sarimax(df_daily["Close"], steps=14)
            if fc is None:
                st.info("Not enough history for SARIMAX forecast.")
            else:
                mean, conf = fc
                out = pd.DataFrame({"Forecast": mean})
                out["Lower"] = conf.iloc[:, 0]
                out["Upper"] = conf.iloc[:, 1]
                st.dataframe(out.tail(14), use_container_width=True)
        else:
            st.warning("No daily data for forecast.")

# -------------------------
# Tab 2 — Breakout / Breakdown
# -------------------------
with tab2:
    st.subheader("Breakout / Breakdown")
    ensure_key_in_options("tab2_ticker", universe, fallback if "fallback" in locals() else universe[0])
    t2 = st.selectbox("Ticker:", universe, key="tab2_ticker")

    df = fetch_daily(t2)
    if df is None or df.empty:
        st.warning("No data.")
    else:
        d = subset_by_daily_view(df, daily_view)
        close = d["Close"]
        sup, res = find_support_resistance(close, slope_lb_daily)
        last = float(close.iloc[-1])
        st.metric("Last Close", fmt_price(last))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Support", fmt_price(sup))
        with c2:
            st.metric("Resistance", fmt_price(res))

        if np.isfinite(res) and last > res:
            st.success("Breakout: price is above resistance.")
        elif np.isfinite(sup) and last < sup:
            st.error("Breakdown: price is below support.")
        else:
            st.info("Price is within range.")
# =========================
# Part 6/6 — bullbear.py
# =========================
# -------------------------
# Tab 3 — Trend / Metrics
# -------------------------
with tab3:
    st.subheader("Trend / Metrics")
    ensure_key_in_options("tab3_ticker", universe, fallback if "fallback" in locals() else universe[0])
    t3 = st.selectbox("Ticker:", universe, key="tab3_ticker")

    df = fetch_daily(t3)
    if df is None or df.empty:
        st.warning("No data.")
    else:
        d = subset_by_daily_view(df, daily_view)
        close = d["Close"]
        ch = linear_regression_channel(close, slope_lb_daily)
        st.write(
            f"**Global slope (daily, last {ch['window']} bars):** {ch['m']:.5f}  \n"
            f"**R²:** {ch['r2']:.3f}"
        )
        if np.isfinite(ch["r2"]) and ch["r2"] >= 0.6:
            st.success("Trendline fit is strong (R² ≥ 0.60). Prefer the global trendline.")
        elif np.isfinite(ch["r2"]) and ch["r2"] <= 0.3:
            st.warning("Trendline fit is weak (R² ≤ 0.30). Prefer local S/R + mean-reversion signals.")
        else:
            st.info("Moderate trendline fit. Use confluence (trend + S/R + indicators).")

        out = pd.DataFrame({
            "Close": close.tail(20),
            "TrendMid": ch["mid"].reindex(close.index).ffill().tail(20),
            "Upper2σ": ch["upper"].reindex(close.index).ffill().tail(20),
            "Lower2σ": ch["lower"].reindex(close.index).ffill().tail(20),
        })
        st.dataframe(out, use_container_width=True)

# -------------------------
# Tab 4 — News
# -------------------------
with tab4:
    st.subheader("News")
    ensure_key_in_options("tab4_ticker", universe, fallback if "fallback" in locals() else universe[0])
    t4 = st.selectbox("Ticker:", universe, key="tab4_ticker")

    if mode == "Forex" and (not show_fx_news):
        st.info("Enable “Show Yahoo Finance news (Forex only)” in the sidebar to see Forex news.")
    else:
        try:
            tk = yf.Ticker(t4)
            news = tk.news or []
        except Exception:
            news = []

        if not news:
            st.info("No news returned by Yahoo Finance.")
        else:
            cutoff = datetime.utcnow() - timedelta(days=int(news_window_days))
            for item in news[:30]:
                title = item.get("title", "Untitled")
                publisher = item.get("publisher", "")
                ts = item.get("providerPublishTime", None)
                when = datetime.utcfromtimestamp(ts) if ts else None
                if when and when < cutoff:
                    continue
                st.markdown(f"**{title}**  \n{publisher}  \n{when.strftime('%Y-%m-%d %H:%M UTC') if when else ''}")
                st.markdown("---")

# -------------------------
# Tab 5 — MACD + HMA Scanner
# -------------------------
with tab5:
    st.subheader("MACD + HMA Scanner")

    recent_bars = int(st.number_input("MACD cross recency (bars)", 3, 120, int(st.session_state.macd_hot_recent_bars), 1, key="macd_hot_recent_bars"))
    hma_cross_recent = int(st.number_input("HMA cross recency (bars)", 1, 30, int(st.session_state.macd_hma_recent_bars), 1, key="macd_hma_recent_bars"))
    tol = st.slider("MACD near-zero tolerance", 0.0000, 0.0200, 0.0020, 0.0005)

    scan = st.button("Run Scanner")
    if scan:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            prog.progress((i + 1) / max(1, len(universe)))
            df = fetch_daily(sym)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna()
            if len(close) < 80:
                continue

            ch = linear_regression_channel(close, slope_lb_daily)
            trend_up = np.isfinite(ch["m"]) and ch["m"] > 0
            trend_dn = np.isfinite(ch["m"]) and ch["m"] < 0

            m, s, _ = macd(close)
            h = hma(close, hma_period)

            cross_up = (m.shift(1) < 0) & (m >= 0) & (m.abs() <= tol)
            cross_dn = (m.shift(1) > 0) & (m <= 0) & (m.abs() <= tol)

            px_up = (close.shift(1) < h.shift(1)) & (close >= h)
            px_dn = (close.shift(1) > h.shift(1)) & (close <= h)

            window = close.index[-recent_bars:] if len(close) >= recent_bars else close.index

            hit_buy = trend_up and bool((cross_up & px_up).reindex(window).fillna(False).any())
            hit_sell = trend_dn and bool((cross_dn & px_dn).reindex(window).fillna(False).any())

            if hit_buy or hit_sell:
                rows.append({
                    "Symbol": sym,
                    "Signal": "BUY" if hit_buy else "SELL",
                    "Slope": ch["m"],
                    "R²": ch["r2"],
                    "Last Close": float(close.iloc[-1]),
                    "MACD": float(m.iloc[-1]),
                })

        prog.empty()
        if not rows:
            st.info("No matches found.")
        else:
            out = pd.DataFrame(rows).sort_values(["Signal", "R²"], ascending=[True, False])
            st.dataframe(out, use_container_width=True)

# -------------------------
# Tab 6 — Long-Term Trend
# -------------------------
with tab6:
    st.subheader("Long-Term Trend & Extremes")
    ensure_key_in_options("hist_long_ticker", universe, fallback if "fallback" in locals() else universe[0])
    t6 = st.selectbox("Ticker:", universe, key="hist_long_ticker")

    df = fetch_daily(t6)
    if df is None or df.empty:
        st.warning("No data.")
    else:
        close = df["Close"]
        ch = linear_regression_channel(close, min(600, max(80, slope_lb_daily)))
        sup, res = find_support_resistance(close, min(600, max(80, slope_lb_daily)))
        st.write(f"**Slope:** {ch['m']:.6f}  •  **R²:** {ch['r2']:.3f}")
        st.write(f"**Range:** {fmt_price(sup)} → {fmt_price(res)}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(close.index, close.values, linewidth=1.0, label="Close")
        ax.plot(ch["mid"].index, ch["mid"].values, linestyle="--", linewidth=1.4, label="Trend")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# -------------------------
# Tab 7 — NTD Scanner (simple z-slope)
# -------------------------
with tab7:
    st.subheader("NTD Scanner (simple)")
    hr = st.selectbox("Hourly range", ["24h", "3d", "7d", "30d", "90d"], key="ntd_scan_hour_range")
    thr = st.slider("Threshold", 0.1, 2.0, 0.5, 0.1)

    run_ntd = st.button("Run NTD Scan")
    if run_ntd:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            prog.progress((i + 1) / max(1, len(universe)))
            dfh = fetch_intraday(sym, period=hr, interval="60m")
            if dfh is None or dfh.empty or "Close" not in dfh:
                continue
            c = dfh["Close"].dropna()
            if len(c) < 50:
                continue
            d = c.diff()
            z = (d - d.rolling(60).mean()) / (d.rolling(60).std() + 1e-9)
            last = float(z.iloc[-1])
            if abs(last) >= thr:
                rows.append({"Symbol": sym, "NTD(zΔ)": last, "Last Close": float(c.iloc[-1])})

        prog.empty()
        if not rows:
            st.info("No symbols exceeded the threshold.")
        else:
            out = pd.DataFrame(rows).sort_values("NTD(zΔ)", ascending=False)
            st.dataframe(out, use_container_width=True)
