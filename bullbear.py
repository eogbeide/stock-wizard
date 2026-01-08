# =========================
# Part 1/10 — bullbear.py
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D

# ---------------------------
# Page config + UI CSS
# ---------------------------
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 {
      transform: none !important;
      visibility: visible !important;
      width: 100% !important;
      position: relative !important;
      margin-bottom: 1rem;
    }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Auto-refresh (PST)
# ---------------------------
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.experimental_rerun()
        except Exception:
            pass

auto_refresh()
elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# ---------------------------
# Mode buttons (Forex / Stocks)
# ---------------------------
def _reset_run_state_for_mode_switch():
    """
    When switching modes, reset run state so:
      • selectbox keys don't crash due to old values not in new universe
      • charts/forecast don't show stale data
    """
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.pop("df_hist", None)
    st.session_state.pop("df_ohlc", None)
    st.session_state.pop("fc_idx", None)
    st.session_state.pop("fc_vals", None)
    st.session_state.pop("fc_ci", None)
    st.session_state.pop("intraday", None)
    st.session_state.pop("chart", None)
    st.session_state.pop("hour_range", None)
    st.session_state.pop("mode_at_run", None)

if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"  # default

st.title("📊 Dashboard & Forecasts")
mcol1, mcol2 = st.columns(2)

if mcol1.button("🌐 Forex", use_container_width=True, key="btn_mode_forex"):
    if st.session_state.asset_mode != "Forex":
        st.session_state.asset_mode = "Forex"
        _reset_run_state_for_mode_switch()
        try:
            st.experimental_rerun()
        except Exception:
            pass

if mcol2.button("📈 Stocks", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        try:
            st.experimental_rerun()
        except Exception:
            pass

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# ---------------------------
# Aesthetic helper (no logic change)
# ---------------------------
def style_axes(ax):
    """Simple, consistent, user-friendly chart styling."""
    try:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    except Exception:
        pass

# ---------------------------
# Core helpers
# ---------------------------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if not num_cols:
            return pd.Series(dtype=float)
        s = obj[num_cols[0]]
    else:
        try:
            s = pd.Series(obj)
        except Exception:
            return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_price_val(y: float) -> str:
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.3f}"

def fmt_slope(m: float) -> str:
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.4f}" if np.isfinite(mv) else "n/a"

def fmt_r2(r2: float, digits: int = 1) -> str:
    try:
        rv = float(r2)
    except Exception:
        return "n/a"
    return fmt_pct(rv, digits=digits) if np.isfinite(rv) else "n/a"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans,
        ha="left", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        zorder=6
    )

def subset_by_daily_view(obj, view_label: str):
    if obj is None or len(obj.index) == 0:
        return obj
    idx = obj.index
    end = idx.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        start = idx.min()
    else:
        start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return obj.loc[(idx >= start) & (idx <= end)]

# FX helpers
def pip_size_for_symbol(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return None
    return 0.01 if "JPY" in s else 0.0001

def _diff_text(a: float, b: float, symbol: str) -> str:
    try:
        av = float(a); bv = float(b)
    except Exception:
        return ""
    ps = pip_size_for_symbol(symbol)
    diff = abs(bv - av)
    if ps:
        return f"{diff/ps:.1f} pips"
    return f"Δ {diff:.3f}"

ALERT_TEXT = "ALERT: Trend may be changing - Open trade position with caution while still following the signals on the chat."

# NEW (THIS REQUEST): Fibonacci-specific alert instruction
FIB_ALERT_TEXT = "ALERT: Fibonacci guidance — BUY close to the 100% line and SELL close to the 0% line."

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float = None,
                             # NEW (THIS REQUEST): ADX(+DI/-DI) gate (optional)
                             use_adx_filter: bool = False,
                             adx_last: float = None,
                             di_plus_last: float = None,
                             di_minus_last: float = None,
                             adx_threshold: float = 20.0) -> str:
    """
    UPDATED (prior request):
      - Show BUY instruction only when Global Trendline slope and Local Slope agree (both UP)
      - Show SELL instruction only when Global Trendline slope and Local Slope agree (both DOWN)
      - Otherwise show an alert message.

    NEW (THIS REQUEST):
      - Optional ADX(+DI/-DI) filter gate:
          • Allow BUY only if ADX >= adx_threshold and +DI > -DI
          • Allow SELL only if ADX >= adx_threshold and -DI > +DI
        If enabled and the gate fails, returns ALERT.

    Backward-compatibility:
      - If global_trend_slope is None, falls back to the prior behavior (uses only trend_slope).
      - If use_adx_filter is False (default), ADX gate is ignored.
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    def _adx_gate_ok(side: str) -> (bool, str):
        """Return (ok, reason)."""
        if not bool(use_adx_filter):
            return True, ""
        if not (_finite(adx_last) and _finite(di_plus_last) and _finite(di_minus_last) and _finite(adx_threshold)):
            return False, "ADX/DMI unavailable"
        a = float(adx_last)
        p = float(di_plus_last)
        n = float(di_minus_last)
        thr = float(adx_threshold)
        if not np.isfinite(a) or a < thr:
            return False, f"ADX<{thr:.0f}"
        side = str(side).upper()
        if side.startswith("B"):
            return (p > n), "DMI not aligned"
        return (n > p), "DMI not aligned"

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    if global_trend_slope is None:
        uptrend = False
        try:
            uptrend = float(trend_slope) >= 0.0
        except Exception:
            pass

        init_side = "BUY" if uptrend else "SELL"
        ok, reason = _adx_gate_ok(init_side)
        if not ok:
            return f"{ALERT_TEXT} ({reason})"

        if uptrend:
            leg_a_val, leg_b_val = entry_buy, exit_sell
            text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        else:
            leg_a_val, leg_b_val = exit_sell, entry_buy
            text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"

        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    try:
        g = float(global_trend_slope)
        l = float(trend_slope)
    except Exception:
        g = np.nan
        l = np.nan

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return ALERT_TEXT

    sg = float(np.sign(g))
    sl = float(np.sign(l))

    if sg == 0.0 or sl == 0.0:
        return ALERT_TEXT

    if sg > 0 and sl > 0:
        ok, reason = _adx_gate_ok("BUY")
        if not ok:
            return f"{ALERT_TEXT} ({reason})"
        leg_a_val, leg_b_val = entry_buy, exit_sell
        text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    if sg < 0 and sl < 0:
        ok, reason = _adx_gate_ok("SELL")
        if not ok:
            return f"{ALERT_TEXT} ({reason})"
        leg_a_val, leg_b_val = exit_sell, entry_buy
        text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    return ALERT_TEXT
# =========================
# Part 2/10 — bullbear.py
# =========================
# ---------------------------
# Gapless (continuous) intraday prices
# ---------------------------
def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open", "High", "Low", "Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
    """
    Remove *price gaps* at session breaks by applying a cumulative offset so that
    the first bar after a large time-gap STARTS (Open) at the previous bar's Close.
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if "Close" not in df.columns:
        return df

    ref_col = "Open" if "Open" in df.columns else "Close"

    close = pd.to_numeric(df["Close"], errors="coerce")
    refp  = pd.to_numeric(df[ref_col], errors="coerce")

    idx = close.index
    diffs = idx.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return df
    expected = float(np.nanmedian(diffs))
    if not np.isfinite(expected) or expected <= 0:
        return df

    thr = max(expected * float(gap_mult), float(min_gap_seconds))
    offsets = np.zeros(len(close), dtype=float)
    offset = 0.0

    for i in range(1, len(close)):
        try:
            dt_sec = float((idx[i] - idx[i-1]).total_seconds())
        except Exception:
            dt_sec = 0.0

        if dt_sec >= thr:
            prev_close = float(close.iloc[i-1]) if np.isfinite(close.iloc[i-1]) else np.nan
            curr_ref   = float(refp.iloc[i])    if np.isfinite(refp.iloc[i])    else np.nan
            if np.isfinite(prev_close) and np.isfinite(curr_ref):
                offset += (curr_ref - prev_close)

        offsets[i] = offset

    offs = pd.Series(offsets, index=idx)
    out = df.copy()
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") - offs
    return out

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return
    n = len(real_times)
    n_ticks = int(max(2, min(n_ticks, n)))
    pos = np.linspace(0, n - 1, n_ticks, dtype=int)
    labels = []
    for i in pos:
        try:
            labels.append(real_times[i].strftime("%m-%d %H:%M"))
        except Exception:
            labels.append(str(real_times[i]))
    ax.set_xticks(pos.tolist())
    ax.set_xticklabels(labels, rotation=0, fontsize=8)

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, times_list):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return []
    if times_list is None:
        return []
    try:
        t = pd.to_datetime(list(times_list))
    except Exception:
        return []
    if len(t) == 0:
        return []
    try:
        idxer = real_times.get_indexer(t, method="nearest")
    except Exception:
        return []
    pos = [int(i) for i in idxer if int(i) >= 0]
    return pos

# ---------------------------
# Sidebar configuration
# ---------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

if st.sidebar.button("🧹 Clear cache (data + run state)", use_container_width=True, key="btn_clear_cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _reset_run_state_for_mode_switch()
    for k in ["sb_show_fibs", "sb_show_mom_hourly", "sb_show_macd"]:
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass
    try:
        st.experimental_rerun()
    except Exception:
        pass

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

# UPDATED (THIS REQUEST): Fibonacci applies to Daily + Hourly, default ON
show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True, key="sb_show_fibs")

slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 12, 480, 120, 6, key="sb_slope_lb_hourly")

st.sidebar.subheader("MACD")
show_macd = st.sidebar.checkbox("Show MACD chart", value=False, key="sb_show_macd")

st.sidebar.subheader("Slope Reversal Probability (experimental)")
rev_hist_lb = st.sidebar.slider("History window for reversal stats (bars)", 30, 720, 240, 30, key="sb_rev_hist_lb")
rev_horizon = st.sidebar.slider("Forward horizon for reversal (bars)", 3, 60, 15, 1, key="sb_rev_horizon")

st.sidebar.subheader("Daily Support/Resistance Window")
sr_lb_daily = st.sidebar.slider("Daily S/R lookback (bars)", 20, 252, 60, 5, key="sb_sr_lb_daily")

st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=False, key="sb_show_mom_hourly")
mom_lb_hourly = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

st.sidebar.subheader("Hourly Indicator Panel")
show_nrsi = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox(
    "Highlight when price is between S/R (S↔R) on NTD",
    value=True, key="sb_ntd_channel"
)

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signal Logic")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# NEW (THIS REQUEST): ADX(+DI/-DI) filter settings (added; no other UI changed)
st.sidebar.subheader("ADX (Trend Strength Filter)")
use_adx_filter = st.sidebar.checkbox(
    "Use ADX(+DI/-DI) filter for BUY/SELL text",
    value=True,
    key="sb_use_adx_filter"
)
adx_period = st.sidebar.slider("ADX period", 5, 50, 14, 1, key="sb_adx_period")
adx_threshold = st.sidebar.slider("ADX threshold (trend strength)", 5.0, 50.0, 20.0, 1.0, key="sb_adx_threshold")

st.sidebar.subheader("NTD (Daily/Hourly)")
# UPDATED (THIS REQUEST): use a new key so Daily NTD displays ON by default again
show_ntd = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd_v2")
ntd_window = st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
show_npx_ntd = st.sidebar.checkbox("Overlay normalized price (NPX) on NTD", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses (dots)", value=True, key="sb_mark_npx_cross")

st.sidebar.subheader("Normalized Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf = st.sidebar.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

st.sidebar.subheader("HMA(55) Reversal on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb = st.sidebar.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

st.sidebar.subheader("Reversal Stars (on NTD panel)")
rev_bars_confirm = st.sidebar.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_fx_news = False
    news_window_days = 7
    show_sessions_pst = False

if mode == "Stock":
    universe = sorted([
        "AAPL","SPY","AMZN","DIA","TSLA","SPGI","JPM","VTWG","PLTR","NVDA",
        "META","SITM","MARA","GOOG","HOOD","BABA","IBM","AVGO","GUSH","VOO",
        "MSFT","TSM","NFLX","MP","AAL","URI","DAL","BBAI","QUBT","AMD","SMCI",
        "ORCL","TLT"
    ])
else:
    universe = [
        "EURUSD=X","EURJPY=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","CADJPY=X","USDCHF=X",
        "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
        "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X","GBPCAD=X"
    ]

# ---------------------------
# Data fetchers
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))["Close"]
         .asfreq("D").ffill())
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")[["Close"]].dropna()
    s = df["Close"].asfreq("D").ffill()
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[
        ["Open","High","Low","Close"]
    ].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return df
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)

    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)

    return df

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236*diff,
        "38.2%": hi - 0.382*diff,
        "50%": hi - 0.5*diff,
        "61.8%": hi - 0.618*diff,
        "78.6%": hi - 0.786*diff,
        "100%": lo
    }

# NEW (THIS REQUEST): Trigger that confirms reversal from Fib 0% / 100%
def fib_reversal_trigger_from_extremes(series_like,
                                      proximity_pct_of_range: float = 0.02,
                                      confirm_bars: int = 2,
                                      lookback_bars: int = 60):
    """
    CONFIRMED BUY:
      - price touched near Fib 100% (low) within lookback
      - then prints `confirm_bars` consecutive higher closes (reversal up from low)
    CONFIRMED SELL:
      - price touched near Fib 0% (high) within lookback
      - then prints `confirm_bars` consecutive lower closes (reversal down from high)

    Returns dict or None.
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.empty or len(s) < max(4, int(confirm_bars) + 2):
        return None

    lb = max(10, int(lookback_bars))
    s = s.iloc[-lb:] if len(s) > lb else s

    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 0:
        return None

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return None

    near_hi = s >= (hi - tol)
    near_lo = s <= (lo + tol)

    last_hi_touch = near_hi[near_hi].index[-1] if near_hi.any() else None
    last_lo_touch = near_lo[near_lo].index[-1] if near_lo.any() else None

    def _confirmed_up(from_time):
        seg = s.loc[from_time:]
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars)+1):]) > 0))

    def _confirmed_down(from_time):
        seg = s.loc[from_time:]
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars)+1):]) < 0))

    buy_tr = None
    if last_lo_touch is not None and _confirmed_up(last_lo_touch):
        buy_tr = {
            "side": "BUY",
            "from_level": "100%",
            "touch_time": last_lo_touch,
            "touch_price": float(s.loc[last_lo_touch]) if np.isfinite(s.loc[last_lo_touch]) else np.nan,
            "last_time": s.index[-1],
            "last_price": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan
        }

    sell_tr = None
    if last_hi_touch is not None and _confirmed_down(last_hi_touch):
        sell_tr = {
            "side": "SELL",
            "from_level": "0%",
            "touch_time": last_hi_touch,
            "touch_price": float(s.loc[last_hi_touch]) if np.isfinite(s.loc[last_hi_touch]) else np.nan,
            "last_time": s.index[-1],
            "last_price": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan
        }

    if buy_tr is None and sell_tr is None:
        return None
    if buy_tr is None:
        return sell_tr
    if sell_tr is None:
        return buy_tr
    return buy_tr if buy_tr["touch_time"] >= sell_tr["touch_time"] else sell_tr

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L; S1 = 2 * P - H
    R2 = P + (H - L); S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}
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

# NEW (THIS REQUEST): ADX (+DI / -DI) trend-strength filter
def compute_adx_dmi(df: pd.DataFrame, period: int = 14):
    """
    Returns (ADX, +DI, -DI) as Series indexed like df.
    Uses Wilder's smoothing (RMA) via ewm(alpha=1/period, adjust=False).
    """
    if df is None or (not isinstance(df, pd.DataFrame)) or df.empty or (not {"High","Low","Close"}.issubset(df.columns)):
        idx = df.index if isinstance(df, pd.DataFrame) else pd.Index([])
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty

    h = _coerce_1d_series(df["High"]).astype(float)
    l = _coerce_1d_series(df["Low"]).astype(float)
    c = _coerce_1d_series(df["Close"]).astype(float)

    idx = h.index.union(l.index).union(c.index)
    h = h.reindex(idx)
    l = l.reindex(idx)
    c = c.reindex(idx)

    up_move = h.diff()
    down_move = l.shift(1) - l

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=idx, dtype=float)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=idx, dtype=float)

    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    p = max(2, int(period))
    alpha = 1.0 / float(p)

    tr_s = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    plus_s = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_s = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = (100.0 * (plus_s / tr_s)).replace([np.inf, -np.inf], np.nan)
    minus_di = (100.0 * (minus_s / tr_s)).replace([np.inf, -np.inf], np.nan)

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100.0 * ((plus_di - minus_di).abs() / denom)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.reindex(idx), plus_di.reindex(idx), minus_di.reindex(idx)

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

# ---------------------------
# NEW (THIS REQUEST): "Reverse Possible" when regression slope successfully reverses at Fib 0% / 100%
# ---------------------------
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
# Utilities: S/R, bull-bear, resample, FX sessions
# ---------------------------
def compute_support_resistance_from_close(close: pd.Series, lookback: int = 60):
    c = _coerce_1d_series(close).astype(float)
    if c.empty:
        empty = pd.Series(index=c.index, dtype=float)
        return empty, empty
    lb = max(5, int(lookback))
    sup = c.rolling(lb, min_periods=max(3, lb//3)).min()
    res = c.rolling(lb, min_periods=max(3, lb//3)).max()
    return sup.reindex(c.index), res.reindex(c.index)

def compute_support_resistance_from_ohlc(df: pd.DataFrame, lookback: int = 60):
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        idx = df.index if isinstance(df, pd.DataFrame) else pd.Index([])
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    lb = max(5, int(lookback))
    low = _coerce_1d_series(df["Low"]).astype(float)
    high = _coerce_1d_series(df["High"]).astype(float)
    sup = low.rolling(lb, min_periods=max(3, lb//3)).min()
    res = high.rolling(lb, min_periods=max(3, lb//3)).max()
    return sup.reindex(df.index), res.reindex(df.index)

def bull_bear_score(close: pd.Series, period: str = "6mo"):
    """
    Simple bull/bear: count up vs down daily closes over a lookback.
    period in {"1mo","3mo","6mo","1y"}.
    """
    c = _coerce_1d_series(close).dropna()
    if c.empty or len(c) < 5:
        return {"bull": 0, "bear": 0, "ratio": np.nan}
    days = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}.get(period, 132)
    seg = c.iloc[-days:] if len(c) > days else c
    ret = seg.diff()
    bull = int((ret > 0).sum())
    bear = int((ret < 0).sum())
    denom = bull + bear
    ratio = (bull / denom) if denom > 0 else np.nan
    return {"bull": bull, "bear": bear, "ratio": ratio}

def resample_to_hourly(df5m: pd.DataFrame) -> pd.DataFrame:
    if df5m is None or df5m.empty:
        return df5m
    if not isinstance(df5m.index, pd.DatetimeIndex):
        return df5m
    cols = df5m.columns
    has_ohlc = {"Open","High","Low","Close"}.issubset(cols)
    if not has_ohlc:
        return df5m
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in cols:
        agg["Volume"] = "sum"
    out = df5m.resample("1H").agg(agg).dropna()
    return out

def add_fx_sessions_pst(ax, idx: pd.DatetimeIndex):
    """
    Approx session windows in PST:
      - London: 00:00–08:00 PST
      - New York: 05:00–13:00 PST
    """
    if not show_sessions_pst or idx is None or len(idx) == 0:
        return
    if not isinstance(idx, pd.DatetimeIndex):
        return

    # For each day in view, shade these windows lightly (no specific colors requested; keep minimal)
    days = pd.to_datetime(pd.Series(idx.date).unique())
    for d in days:
        start_l = PACIFIC.localize(datetime(d.year, d.month, d.day, 0, 0))
        end_l   = PACIFIC.localize(datetime(d.year, d.month, d.day, 8, 0))
        start_ny = PACIFIC.localize(datetime(d.year, d.month, d.day, 5, 0))
        end_ny   = PACIFIC.localize(datetime(d.year, d.month, d.day, 13, 0))
        ax.axvspan(start_l, end_l, alpha=0.06)
        ax.axvspan(start_ny, end_ny, alpha=0.06)

# ---------------------------
# Plot: Daily chart
# ---------------------------
def plot_daily_chart(df_daily: pd.DataFrame,
                     df_daily_all: pd.DataFrame,
                     ticker: str):
    """
    Daily price chart + regression + S/R + optional Fib + optional BBands + optional Ichimoku Kijun.
    Also computes and displays:
      - Global slope + R² (all history)
      - Local slope + R² (view / lookback)
      - BUY/SELL instruction gated by slope agreement + ADX filter
      - Reverse Possible label when regression slope reverses at Fib 0%/100%
    """
    if df_daily is None or df_daily.empty or "Close" not in df_daily.columns:
        st.warning("No daily data returned.")
        return

    # View subset
    dfv = subset_by_daily_view(df_daily, daily_view)
    if dfv is None or dfv.empty:
        st.warning("Daily view window has no data.")
        return

    close_v = _coerce_1d_series(dfv["Close"]).dropna()
    if close_v.empty:
        st.warning("Daily close series is empty.")
        return

    # Global history series for global regression
    if df_daily_all is None or df_daily_all.empty or "Close" not in df_daily_all.columns:
        df_daily_all = df_daily
    close_all = _coerce_1d_series(df_daily_all["Close"]).dropna()

    # Regression w/ band (local)
    yhat_l, up_l, lo_l, m_l, r2_l = regression_with_band(close_v, lookback=int(slope_lb_daily), z=2.0)

    # Regression w/ band (global)
    yhat_g, up_g, lo_g, m_g, r2_g = regression_with_band(close_all, lookback=0, z=2.0)

    # Align global line to view for plotting (recompute on view index to avoid jump)
    # (Keeps global slope displayed, but draws a global trendline over current view)
    yhat_g_view, _, _, m_g_view, r2_g_view = regression_with_band(close_v, lookback=0, z=2.0)
    m_g_to_show = m_g if np.isfinite(m_g) else m_g_view
    r2_g_to_show = r2_g if np.isfinite(r2_g) else r2_g_view

    # Support/Resistance (daily)
    sup, res = compute_support_resistance_from_ohlc(dfv, lookback=int(sr_lb_daily))

    # ADX filter (daily)
    adx_s, di_p, di_m = compute_adx_dmi(df_daily_all[["High","Low","Close"]].dropna(), period=int(adx_period))
    adx_last = _safe_last_float(adx_s)
    di_p_last = _safe_last_float(di_p)
    di_m_last = _safe_last_float(di_m)

    # Fibonacci (daily view)
    fibs = fibonacci_levels(close_v) if show_fibs else {}

    # Reverse Possible (daily) based on fib + regression slope flip
    rev_info = regression_slope_reversal_at_fib_extremes(
        close_v,
        slope_lb=int(slope_lb_daily),
        proximity_pct_of_range=0.02,
        confirm_bars=2,
        lookback_bars=max(120, int(slope_lb_daily) * 2)
    ) if show_fibs else None

    # BBands (price chart)
    mid, bb_u, bb_l, pctb, nbb = compute_bbands(close_v, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    # Ichimoku kijun on price
    kijun = None
    if show_ichi and {"High","Low","Close"}.issubset(dfv.columns):
        tenkan, kijun, sa, sb, chik = ichimoku_lines(dfv["High"], dfv["Low"], dfv["Close"],
                                                     conv=int(ichi_conv), base=int(ichi_base),
                                                     span_b=int(ichi_spanb), shift_cloud=False)

    # Build figure
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(close_v.index, close_v.values, linewidth=1.4, label="Close")
    ax.plot(yhat_l.index, yhat_l.values, linestyle="--", linewidth=2.2,
            label=f"Local Trend ({fmt_slope(m_l)}/bar, R² {fmt_r2(r2_l)})")

    # ±2σ band (local)
    ax.plot(up_l.index, up_l.values, linewidth=1.0, alpha=0.9, label="+2σ Band")
    ax.plot(lo_l.index, lo_l.values, linewidth=1.0, alpha=0.9, label="-2σ Band")

    # Labels on left (band + local trend)
    if len(up_l.dropna()) and len(lo_l.dropna()) and len(yhat_l.dropna()):
        label_on_left(ax, float(up_l.dropna().iloc[-1]), "+2σ")
        label_on_left(ax, float(yhat_l.dropna().iloc[-1]), "Local Trend")
        label_on_left(ax, float(lo_l.dropna().iloc[-1]), "-2σ")

    # Global slope shown (draw global line over view)
    ax.plot(yhat_g_view.index, yhat_g_view.values, linestyle=":", linewidth=2.2,
            label=f"Global Trend ({fmt_slope(m_g_to_show)}/bar, R² {fmt_r2(r2_g_to_show)})")

    # S/R
    if sup.notna().any():
        ax.plot(sup.index, sup.values, linewidth=1.1, alpha=0.9, label="Support")
    if res.notna().any():
        ax.plot(res.index, res.values, linewidth=1.1, alpha=0.9, label="Resistance")

    # BBands
    if show_bbands and bb_u.notna().any() and bb_l.notna().any():
        ax.plot(bb_u.index, bb_u.values, linewidth=1.0, alpha=0.85, label="BB Upper")
        ax.plot(mid.index, mid.values, linewidth=1.0, alpha=0.85, label="BB Mid")
        ax.plot(bb_l.index, bb_l.values, linewidth=1.0, alpha=0.85, label="BB Lower")

    # Kijun on price
    if show_ichi and kijun is not None and _coerce_1d_series(kijun).notna().any():
        ax.plot(kijun.index, kijun.values, linewidth=1.2, alpha=0.95, label="Kijun (Ichimoku)")

    # Fibonacci lines (daily)
    if show_fibs and fibs:
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=0.9, alpha=0.65)
                ax.text(close_v.index[0], v, f"  Fib {k}", va="center", fontsize=8, alpha=0.85)

    # Annotate Reverse Possible
    if show_fibs and rev_info is not None:
        annotate_reverse_possible(ax, rev_info, text="Reverse Possible")

    # Slope trigger (band touch -> cross slope line)
    trig = find_slope_trigger_after_band_reversal(close_v, yhat_l, up_l, lo_l, horizon=int(rev_horizon))
    if trig is not None:
        annotate_slope_trigger(ax, trig)

    # Compose instruction text (Daily)
    buy_val = float(sup.dropna().iloc[-1]) if sup.notna().any() else float(close_v.iloc[-1])
    sell_val = float(res.dropna().iloc[-1]) if res.notna().any() else float(close_v.iloc[-1])
    instr = format_trade_instruction(
        trend_slope=m_l,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=float(close_v.iloc[-1]),
        symbol=ticker,
        global_trend_slope=m_g_to_show,
        use_adx_filter=bool(use_adx_filter),
        adx_last=adx_last,
        di_plus_last=di_p_last,
        di_minus_last=di_m_last,
        adx_threshold=float(adx_threshold)
    )

    ax.set_title(f"{ticker} — Daily")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    style_axes(ax)

    # Compact legend
    ax.legend(loc="best", fontsize=8, frameon=False)

    st.pyplot(fig)

    # Info blocks (Daily)
    c_last = float(close_v.iloc[-1])
    st.markdown(f"**Daily Instruction:** {instr}")
    if show_fibs:
        st.info(FIB_ALERT_TEXT)

    # Show fib-confirmed reversal trigger (separate from regression-slope reversal)
    if show_fibs:
        fib_tr = fib_reversal_trigger_from_extremes(close_v, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=120)
        if fib_tr is not None:
            st.warning(f"**Fib Reversal Confirmed:** {fib_tr['side']} from Fib {fib_tr['from_level']} "
                       f"(touch {fib_tr['touch_time']}, last {fib_tr['last_time']}).")

    # Return components used by other panels
    return {
        "close_view": close_v,
        "sup": sup,
        "res": res,
        "m_local": m_l,
        "r2_local": r2_l,
        "m_global": m_g_to_show,
        "r2_global": r2_g_to_show,
        "adx_last": adx_last,
        "di_p_last": di_p_last,
        "di_m_last": di_m_last
    }
# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Plot: NTD / NPX panel (daily)
# ---------------------------
def plot_daily_ntd_panel(close_view: pd.Series,
                         sup: pd.Series,
                         res: pd.Series,
                         trend_slope_for_triangles: float,
                         title: str = "NTD (Daily)"):
    if close_view is None or close_view.empty:
        return
    ntd = compute_normalized_trend(close_view, window=int(ntd_window))
    npx = compute_normalized_price(close_view, window=int(ntd_window))

    fig = plt.figure(figsize=(12, 3.3))
    ax = fig.add_subplot(111)
    ax.plot(ntd.index, ntd.values, linewidth=1.4, label="NTD")

    if shade_ntd:
        shade_ntd_regions(ax, ntd)

    if show_npx_ntd:
        overlay_npx_on_ntd(ax, npx, ntd, mark_crosses=bool(mark_npx_cross))

    # Optional triangles based on price-trend slope
    overlay_ntd_triangles_by_trend(ax, ntd, trend_slope_for_triangles)

    # HMA reversal markers on NTD
    if show_hma_rev_ntd:
        hma = compute_hma(close_view, period=int(hma_period))
        overlay_hma_reversal_on_ntd(ax, close_view, hma, lookback=int(hma_rev_lb),
                                    period=int(hma_period), ntd=ntd)

    # Reversal stars near S/R on NTD (daily)
    if sup is not None and res is not None:
        overlay_ntd_sr_reversal_stars(ax,
                                     price=close_view,
                                     sup=sup,
                                     res=res,
                                     trend_slope=trend_slope_for_triangles,
                                     ntd=ntd,
                                     prox=float(sr_prox_pct),
                                     bars_confirm=int(rev_bars_confirm))

    ax.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax.axhline(0.75, linewidth=0.8, alpha=0.5)
    ax.axhline(-0.75, linewidth=0.8, alpha=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    style_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False)
    st.pyplot(fig)

# ---------------------------
# Plot: MACD panel (optional)
# ---------------------------
def plot_macd_panel(close_view: pd.Series, title: str = "MACD"):
    if close_view is None or close_view.empty:
        return
    macd, sig, hist = compute_macd(close_view)
    fig = plt.figure(figsize=(12, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(macd.index, macd.values, linewidth=1.3, label="MACD")
    ax.plot(sig.index, sig.values, linewidth=1.1, label="Signal")
    ax.bar(hist.index, hist.values, width=1.0, alpha=0.35, label="Hist")
    ax.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    style_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False)
    st.pyplot(fig)

# ---------------------------
# Plot: Momentum panel (hourly optional)
# ---------------------------
def plot_momentum_panel(close_series: pd.Series, lookback: int = 12, title: str = "Momentum (ROC%)"):
    if close_series is None or close_series.empty:
        return
    roc = compute_roc(close_series, n=int(lookback))
    fig = plt.figure(figsize=(12, 2.8))
    ax = fig.add_subplot(111)
    ax.plot(roc.index, roc.values, linewidth=1.4, label=f"ROC {lookback}")
    ax.axhline(0.0, linewidth=1.0, alpha=0.7)
    ax.set_title(title)
    style_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False)
    st.pyplot(fig)
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Plot: Hourly chart (+ ADX filter + Fib + Reverse Possible)
# ---------------------------
def plot_hourly_chart(df_hourly: pd.DataFrame,
                      ticker: str):
    if df_hourly is None or df_hourly.empty or "Close" not in df_hourly.columns:
        st.warning("No hourly data returned.")
        return

    # Local view window (use last N bars derived from slope lookback and SR lookback)
    # Keep it reasonably wide while respecting data size.
    n_view = int(max(240, slope_lb_hourly * 3, sr_lb_hourly * 3))
    dfv = df_hourly.tail(n_view).copy()
    close_v = _coerce_1d_series(dfv["Close"]).dropna()
    if close_v.empty:
        st.warning("Hourly close series is empty.")
        return

    # Global (hourly) and local regression
    yhat_l, up_l, lo_l, m_l, r2_l = regression_with_band(close_v, lookback=int(slope_lb_hourly), z=2.0)
    yhat_g, _, _, m_g, r2_g = regression_with_band(close_v, lookback=0, z=2.0)

    # S/R (hourly)
    sup, res = compute_support_resistance_from_ohlc(dfv, lookback=int(sr_lb_hourly))

    # Fibonacci (hourly) — UPDATED (THIS REQUEST): display on hourly chart too
    fibs = fibonacci_levels(close_v) if show_fibs else {}

    # Reverse Possible (hourly)
    rev_info = regression_slope_reversal_at_fib_extremes(
        close_v,
        slope_lb=int(slope_lb_hourly),
        proximity_pct_of_range=0.02,
        confirm_bars=2,
        lookback_bars=max(120, int(slope_lb_hourly) * 2)
    ) if show_fibs else None

    # ADX / DMI (hourly) — UPDATED (THIS REQUEST): used as BUY/SELL gate
    adx_s, di_p, di_m = compute_adx_dmi(dfv[["High","Low","Close"]].dropna(), period=int(adx_period))
    adx_last = _safe_last_float(adx_s)
    di_p_last = _safe_last_float(di_p)
    di_m_last = _safe_last_float(di_m)

    # Supertrend + PSAR (hourly)
    st_df = compute_supertrend(dfv[["High","Low","Close"]], atr_period=int(atr_period), atr_mult=float(atr_mult))
    ps_df = compute_psar_from_ohlc(dfv[["High","Low"]], step=float(psar_step), max_step=float(psar_max)) if show_psar else None

    # Optional HMA & MACD/HMA/SR signal
    hma = compute_hma(close_v, period=int(hma_period)) if show_hma else None
    macd, sig, hist = compute_macd(close_v)
    macd_hma_sig = None
    if show_hma and hma is not None and sup is not None and res is not None:
        macd_hma_sig = find_macd_hma_sr_signal(
            close=close_v,
            hma=hma,
            macd=macd,
            sup=sup,
            res=res,
            global_trend_slope=m_g,
            prox=float(sr_prox_pct)
        )

    # Plot hourly price
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(close_v.index, close_v.values, linewidth=1.4, label="Close")
    ax.plot(yhat_l.index, yhat_l.values, linestyle="--", linewidth=2.2,
            label=f"Local Trend ({fmt_slope(m_l)}/bar, R² {fmt_r2(r2_l)})")

    ax.plot(up_l.index, up_l.values, linewidth=1.0, alpha=0.9, label="+2σ Band")
    ax.plot(lo_l.index, lo_l.values, linewidth=1.0, alpha=0.9, label="-2σ Band")

    if len(up_l.dropna()) and len(lo_l.dropna()) and len(yhat_l.dropna()):
        label_on_left(ax, float(up_l.dropna().iloc[-1]), "+2σ")
        label_on_left(ax, float(yhat_l.dropna().iloc[-1]), "Local Trend")
        label_on_left(ax, float(lo_l.dropna().iloc[-1]), "-2σ")

    # Global slope shown
    ax.plot(yhat_g.index, yhat_g.values, linestyle=":", linewidth=2.2,
            label=f"Global Trend ({fmt_slope(m_g)}/bar, R² {fmt_r2(r2_g)})")

    # S/R
    if sup.notna().any():
        ax.plot(sup.index, sup.values, linewidth=1.1, alpha=0.9, label="Support")
    if res.notna().any():
        ax.plot(res.index, res.values, linewidth=1.1, alpha=0.9, label="Resistance")

    # Supertrend
    if st_df is not None and not st_df.empty and "ST" in st_df.columns:
        st_line = _coerce_1d_series(st_df["ST"]).reindex(close_v.index)
        if st_line.notna().any():
            ax.plot(st_line.index, st_line.values, linewidth=1.2, alpha=0.9, label="Supertrend")

    # PSAR
    if show_psar and ps_df is not None and not ps_df.empty and "PSAR" in ps_df.columns:
        psar_line = _coerce_1d_series(ps_df["PSAR"]).reindex(close_v.index)
        if psar_line.notna().any():
            ax.scatter(psar_line.index, psar_line.values, s=18, alpha=0.85, label="PSAR")

    # HMA
    if show_hma and hma is not None and _coerce_1d_series(hma).notna().any():
        ax.plot(hma.index, hma.values, linewidth=1.2, alpha=0.9, label=f"HMA({hma_period})")

    # Fibonacci (hourly)
    if show_fibs and fibs:
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=0.9, alpha=0.65)
                ax.text(close_v.index[0], v, f"  Fib {k}", va="center", fontsize=8, alpha=0.85)

    # Reverse Possible label
    if show_fibs and rev_info is not None:
        annotate_reverse_possible(ax, rev_info, text="Reverse Possible")

    # Slope trigger (band touch -> cross slope line)
    trig = find_slope_trigger_after_band_reversal(close_v, yhat_l, up_l, lo_l, horizon=int(rev_horizon))
    if trig is not None:
        annotate_slope_trigger(ax, trig)

    # MACD/HMA/SR star signal
    if macd_hma_sig is not None:
        annotate_macd_signal(ax, macd_hma_sig["time"], macd_hma_sig["price"], macd_hma_sig["side"])

    # Sessions (PST) for forex intraday/hourly
    add_fx_sessions_pst(ax, close_v.index)

    ax.set_title(f"{ticker} — Hourly")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False)

    st.pyplot(fig)

    # Instruction text (Hourly) — gated by slope alignment + ADX/DMI
    buy_val = float(sup.dropna().iloc[-1]) if sup.notna().any() else float(close_v.iloc[-1])
    sell_val = float(res.dropna().iloc[-1]) if res.notna().any() else float(close_v.iloc[-1])

    instr = format_trade_instruction(
        trend_slope=m_l,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=float(close_v.iloc[-1]),
        symbol=ticker,
        global_trend_slope=m_g,
        use_adx_filter=bool(use_adx_filter),
        adx_last=adx_last,
        di_plus_last=di_p_last,
        di_minus_last=di_m_last,
        adx_threshold=float(adx_threshold)
    )
    st.markdown(f"**Hourly Instruction:** {instr}")

    # ADX status line (explicit)
    if bool(use_adx_filter):
        st.caption(f"ADX gate: ADX={adx_last:.1f} • +DI={di_p_last:.1f} • -DI={di_m_last:.1f} "
                   f"(threshold {adx_threshold:.0f})")

    if show_fibs:
        st.info(FIB_ALERT_TEXT)
        fib_tr = fib_reversal_trigger_from_extremes(close_v, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=120)
        if fib_tr is not None:
            st.warning(f"**Fib Reversal Confirmed (Hourly):** {fib_tr['side']} from Fib {fib_tr['from_level']} "
                       f"(touch {fib_tr['touch_time']}, last {fib_tr['last_time']}).")

    # Hourly NTD panel (uses toggle show_nrsi)
    if show_nrsi:
        ntd = compute_normalized_trend(close_v, window=int(ntd_window))
        npx = compute_normalized_price(close_v, window=int(ntd_window))

        fig2 = plt.figure(figsize=(12, 3.3))
        ax2 = fig2.add_subplot(111)
        ax2.plot(ntd.index, ntd.values, linewidth=1.4, label="NTD (Hourly)")
        if shade_ntd:
            shade_ntd_regions(ax2, ntd)
        if show_npx_ntd:
            overlay_npx_on_ntd(ax2, npx, ntd, mark_crosses=bool(mark_npx_cross))

        # Highlight channel when price is between S and R (Hourly)
        if show_ntd_channel and (sup is not None) and (res is not None):
            s_sup = _coerce_1d_series(sup).reindex(close_v.index).ffill()
            s_res = _coerce_1d_series(res).reindex(close_v.index).ffill()
            ch = (close_v >= s_sup) & (close_v <= s_res)
            if ch.any():
                ax2.fill_between(ntd.index, -1.0, 1.0, where=ch.reindex(ntd.index, fill_value=False),
                                 alpha=0.06)

        overlay_ntd_triangles_by_trend(ax2, ntd, m_g)

        if show_hma_rev_ntd:
            hma2 = compute_hma(close_v, period=int(hma_period))
            overlay_hma_reversal_on_ntd(ax2, close_v, hma2, lookback=int(hma_rev_lb),
                                        period=int(hma_period), ntd=ntd)

        if sup is not None and res is not None:
            overlay_ntd_sr_reversal_stars(ax2,
                                         price=close_v,
                                         sup=sup,
                                         res=res,
                                         trend_slope=m_g,
                                         ntd=ntd,
                                         prox=float(sr_prox_pct),
                                         bars_confirm=int(rev_bars_confirm))

        ax2.axhline(0.0, linewidth=1.0, alpha=0.7)
        ax2.axhline(0.75, linewidth=0.8, alpha=0.5)
        ax2.axhline(-0.75, linewidth=0.8, alpha=0.5)
        ax2.set_ylim(-1.05, 1.05)
        ax2.set_title("NTD (Hourly)")
        style_axes(ax2)
        ax2.legend(loc="best", fontsize=8, frameon=False)
        st.pyplot(fig2)

    # Optional hourly momentum
    if show_mom_hourly:
        plot_momentum_panel(close_v, lookback=int(mom_lb_hourly), title="Momentum (Hourly ROC%)")

    # Optional MACD
    if show_macd:
        plot_macd_panel(close_v, title="MACD (Hourly)")

    return {
        "close_view": close_v,
        "sup": sup,
        "res": res,
        "m_local": m_l,
        "r2_local": r2_l,
        "m_global": m_g,
        "r2_global": r2_g,
        "adx_last": adx_last,
        "di_p_last": di_p_last,
        "di_m_last": di_m_last
    }
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Main app controls
# ---------------------------
st.sidebar.subheader("Ticker")
default_ticker = universe[0] if universe else None
if "ticker" not in st.session_state or st.session_state.ticker not in universe:
    st.session_state.ticker = default_ticker

ticker = st.sidebar.selectbox(
    "Select symbol:",
    universe,
    index=universe.index(st.session_state.ticker) if st.session_state.ticker in universe else 0,
    key="sb_ticker"
)
st.session_state.ticker = ticker

if mode == "Forex":
    intraday_period = st.sidebar.selectbox("Intraday period (5m bars):", ["1d", "5d", "1mo"], index=1, key="sb_intra_period")
else:
    intraday_period = st.sidebar.selectbox("Intraday period (5m bars):", ["1d", "5d", "1mo"], index=0, key="sb_intra_period")

# Run button
run_all = st.sidebar.button("▶️ Run", use_container_width=True, key="btn_run_all")
if run_all:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# If mode changed since last run, suppress stale plots until next run
if st.session_state.get("mode_at_run", mode) != mode:
    st.session_state.run_all = False

# ---------------------------
# Data load + cache into session (only when run)
# ---------------------------
if st.session_state.get("run_all", False):
    with st.spinner("Fetching market data..."):
        # Daily
        df_daily_ohlc = fetch_hist_ohlc(ticker)
        df_daily_all  = fetch_hist_ohlc(ticker)  # kept separate for clarity; cached anyway

        # Intraday 5m + hourly
        df_5m = fetch_intraday(ticker, period=intraday_period)
        df_1h = resample_to_hourly(df_5m) if (df_5m is not None and not df_5m.empty) else pd.DataFrame()

        # Store
        st.session_state.df_daily_ohlc = df_daily_ohlc
        st.session_state.df_daily_all = df_daily_all
        st.session_state.df_5m = df_5m
        st.session_state.df_1h = df_1h

        # Forecast uses daily close
        try:
            idx_fc, fc_vals, fc_ci = compute_sarimax_forecast(df_daily_ohlc["Close"])
            st.session_state.fc_idx = idx_fc
            st.session_state.fc_vals = fc_vals
            st.session_state.fc_ci = fc_ci
        except Exception:
            st.session_state.fc_idx = None
            st.session_state.fc_vals = None
            st.session_state.fc_ci = None

# ---------------------------
# Read from session for display
# ---------------------------
df_daily_ohlc = st.session_state.get("df_daily_ohlc", None)
df_daily_all  = st.session_state.get("df_daily_all", None)
df_5m = st.session_state.get("df_5m", None)
df_1h = st.session_state.get("df_1h", None)

fc_idx = st.session_state.get("fc_idx", None)
fc_vals = st.session_state.get("fc_vals", None)
fc_ci = st.session_state.get("fc_ci", None)

# ---------------------------
# Top-level summary metrics
# ---------------------------
if df_daily_ohlc is not None and isinstance(df_daily_ohlc, pd.DataFrame) and (not df_daily_ohlc.empty) and "Close" in df_daily_ohlc.columns:
    close_daily = _coerce_1d_series(df_daily_ohlc["Close"]).dropna()
    bb = bull_bear_score(close_daily, period=bb_period)
    last_px = float(close_daily.iloc[-1]) if len(close_daily) else np.nan
    chg = float(close_daily.iloc[-1] - close_daily.iloc[-2]) if len(close_daily) >= 2 else np.nan
    chg_pct = float(close_daily.pct_change().iloc[-1]) if len(close_daily) >= 2 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"{last_px:,.4f}" if np.isfinite(last_px) else "n/a",
              f"{chg:+.4f}" if np.isfinite(chg) else None)
    c2.metric("Daily %", f"{chg_pct*100:.2f}%" if np.isfinite(chg_pct) else "n/a")
    c3.metric("Bull/Bear (lookback)", f"{bb['bull']} / {bb['bear']}")
    c4.metric("Bull Ratio", f"{bb['ratio']*100:.1f}%" if np.isfinite(bb["ratio"]) else "n/a")

    if mode == "Forex":
        piv = current_daily_pivots(df_daily_ohlc)
        if piv:
            st.caption(f"Daily pivots: P={piv['P']:.4f} • R1={piv['R1']:.4f} • S1={piv['S1']:.4f} • R2={piv['R2']:.4f} • S2={piv['S2']:.4f}")
# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Tabs: Daily / Hourly / Forecast / Intraday (5m)
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily", "🕐 Hourly", "🔮 Forecast", "⏱️ Intraday (5m)"])

with tab1:
    if df_daily_ohlc is None or df_daily_ohlc.empty:
        st.info("Click **Run** to load data.")
    else:
        out_daily = plot_daily_chart(df_daily_ohlc, df_daily_all, ticker)

        # Daily NTD — UPDATED (THIS REQUEST): on by default (show_ntd default True)
        if show_ntd and isinstance(out_daily, dict):
            plot_daily_ntd_panel(
                close_view=out_daily["close_view"],
                sup=out_daily["sup"],
                res=out_daily["res"],
                trend_slope_for_triangles=out_daily["m_global"],
                title="NTD (Daily)"
            )

        # Optional MACD (daily)
        if show_macd and isinstance(out_daily, dict):
            plot_macd_panel(out_daily["close_view"], title="MACD (Daily)")

with tab2:
    if df_1h is None or (not isinstance(df_1h, pd.DataFrame)) or df_1h.empty:
        st.info("Click **Run** to load intraday data and build the hourly series.")
    else:
        plot_hourly_chart(df_1h, ticker)

with tab3:
    if fc_idx is None or fc_vals is None or fc_ci is None:
        st.info("Forecast will appear after clicking **Run** (SARIMAX on daily closes).")
    else:
        # Forecast plot
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        # history for context (last 180 days)
        if df_daily_ohlc is not None and (not df_daily_ohlc.empty) and "Close" in df_daily_ohlc.columns:
            hist = _coerce_1d_series(df_daily_ohlc["Close"]).dropna()
            hist = hist.iloc[-180:] if len(hist) > 180 else hist
            ax.plot(hist.index, hist.values, linewidth=1.4, label="History")

        ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, linestyle="--", linewidth=2.0, label="Forecast")
        ci = fc_ci
        try:
            lo = _coerce_1d_series(ci.iloc[:, 0]).values
            hi = _coerce_1d_series(ci.iloc[:, 1]).values
            ax.fill_between(fc_idx, lo, hi, alpha=0.15, label="Confidence")
        except Exception:
            pass

        ax.set_title(f"{ticker} — 30D Forecast (SARIMAX)")
        style_axes(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)
        st.pyplot(fig)

with tab4:
    if df_5m is None or (not isinstance(df_5m, pd.DataFrame)) or df_5m.empty:
        st.info("Click **Run** to load intraday 5-minute data.")
    else:
        # Intraday chart (gapless already applied inside fetch_intraday)
        dfv = df_5m.copy()
        if {"Close","High","Low"}.issubset(dfv.columns):
            close = _coerce_1d_series(dfv["Close"]).dropna()
            if close.empty:
                st.warning("Intraday close series is empty.")
            else:
                # Local regression (intraday)
                yhat, up, lo, m, r2 = regression_with_band(close, lookback=min(int(slope_lb_hourly), len(close)), z=2.0)

                # S/R intraday (use hourly SR window scaled)
                sup, res = compute_support_resistance_from_ohlc(dfv, lookback=max(20, int(sr_lb_hourly)))

                # Fib on intraday is optional; request only said Daily + Hourly, so keep OFF here unless user wants.
                # (Still allow: show_fibs applies, but can be noisy.)
                fibs = fibonacci_levels(close) if show_fibs else {}

                fig = plt.figure(figsize=(12, 5))
                ax = fig.add_subplot(111)

                ax.plot(close.index, close.values, linewidth=1.3, label="Close (5m)")
                ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=2.0,
                        label=f"Trend ({fmt_slope(m)}/bar, R² {fmt_r2(r2)})")
                ax.plot(up.index, up.values, linewidth=1.0, alpha=0.9, label="+2σ")
                ax.plot(lo.index, lo.values, linewidth=1.0, alpha=0.9, label="-2σ")

                if sup.notna().any():
                    ax.plot(sup.index, sup.values, linewidth=1.0, alpha=0.85, label="Support")
                if res.notna().any():
                    ax.plot(res.index, res.values, linewidth=1.0, alpha=0.85, label="Resistance")

                if show_fx_news:
                    st.caption("Forex news markers are currently not sourced (placeholder).")

                add_fx_sessions_pst(ax, close.index)

                # Optional: Fib overlay (intraday)
                if show_fibs and fibs:
                    for k, v in fibs.items():
                        if np.isfinite(v):
                            ax.axhline(v, linewidth=0.8, alpha=0.55)
                            ax.text(close.index[0], v, f"  Fib {k}", va="center", fontsize=8, alpha=0.75)

                # Slope trigger & Reverse Possible on intraday (best-effort)
                if show_fibs:
                    rev_info = regression_slope_reversal_at_fib_extremes(
                        close, slope_lb=max(30, min(180, int(slope_lb_hourly))),
                        proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=240
                    )
                    if rev_info is not None:
                        annotate_reverse_possible(ax, rev_info, text="Reverse Possible")

                trig = find_slope_trigger_after_band_reversal(close, yhat, up, lo, horizon=int(rev_horizon))
                if trig is not None:
                    annotate_slope_trigger(ax, trig)

                ax.set_title(f"{ticker} — Intraday (5m, gapless)")
                style_axes(ax)
                ax.legend(loc="best", fontsize=8, frameon=False)
                st.pyplot(fig)

                if show_fibs:
                    st.info(FIB_ALERT_TEXT)

# ---------------------------
# Footer note
# ---------------------------
st.caption("Educational dashboard only. Not financial advice.")
