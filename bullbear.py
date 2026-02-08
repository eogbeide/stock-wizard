# ==========================================================
# bullbear.py (COMPLETE UPDATED CODE) — BATCH 1 / 3
# ==========================================================
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
# Rerun helper (deprecation-proof)
# ---------------------------
def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---------------------------
# Matplotlib theme (STYLE ONLY — no logic changes)
# ---------------------------
def _apply_mpl_theme():
    """A clean, modern look for matplotlib output rendered in Streamlit (no data/logic changes)."""
    try:
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "legend.fontsize": 9,
            "legend.framealpha": 0.70,
            "legend.fancybox": True,
            "lines.linewidth": 1.6,
        })
    except Exception:
        pass

_apply_mpl_theme()

# ---------------------------
# Page config + UI CSS (SINGLE canonical block)
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

  /* =========================
     Ribbon Tabs (BaseWeb) — rectangular
     ========================= */
  div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    gap: 0.40rem !important;
    padding: 0.25rem 0.25rem 0.40rem 0.25rem !important;
    border-bottom: 1px solid rgba(49, 51, 63, 0.14) !important;
  }
  div[data-baseweb="tab"] { flex: 0 0 auto !important; }
  div[data-baseweb="tab"] > button,
  div[data-baseweb="tab"] button {
    padding: 0.40rem 0.85rem !important;
    border: 1px solid rgba(49, 51, 63, 0.20) !important;
    border-radius: 6px !important; /* rectangular ribbon feel */
    background: rgba(248, 250, 252, 0.96) !important;
    box-shadow: 0 1px 0 rgba(0,0,0,0.03) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    white-space: nowrap !important;
    transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background 120ms ease !important;
  }
  div[data-baseweb="tab"] > button:hover,
  div[data-baseweb="tab"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,0.10) !important;
    border-color: rgba(49, 51, 63, 0.32) !important;
    background: rgba(241, 245, 249, 1.0) !important;
  }
  div[data-baseweb="tab"] > button[aria-selected="true"],
  div[data-baseweb="tab"] button[aria-selected="true"] {
    background: linear-gradient(90deg, rgba(59,130,246,0.96), rgba(99,102,241,0.96)) !important;
    border-color: rgba(59,130,246,0.95) !important;
    box-shadow: 0 12px 24px rgba(59,130,246,0.24) !important;
  }
  div[data-baseweb="tab"] > button[aria-selected="true"] p,
  div[data-baseweb="tab"] button[aria-selected="true"] p {
    color: white !important;
  }
  div[data-baseweb="tab"] p {
    margin: 0 !important;
    font-size: 0.90rem !important;
  }
  div[data-baseweb="tab"] > button:focus,
  div[data-baseweb="tab"] button:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(49, 51, 63, 0.18) !important;
  }

  /* =========================
     Chart container styling (Streamlit React UI wrappers)
     ========================= */
  div[data-testid="stImage"] {
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 14px;
    background: rgba(255,255,255,0.65);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    padding: 0.35rem 0.35rem 0.15rem 0.35rem;
    overflow: hidden;
  }
  div[data-testid="stImage"] img {
    border-radius: 14px !important;
    background: white !important;
    padding: 6px !important;
    box-shadow: 0 12px 28px rgba(0,0,0,0.10) !important;
  }
  @media (max-width: 600px) {
    div[data-testid="stImage"] img {
      border-radius: 10px !important;
      padding: 4px !important;
      box-shadow: 0 10px 22px rgba(0,0,0,0.10) !important;
    }
  }

  /* Mobile: keep sidebar usable */
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
        _rerun()

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
        _rerun()

if mcol2.button("📈 Stocks", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        _rerun()

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# ---------------------------
# Aesthetic helper (STYLE ONLY — no logic change)
# ---------------------------
def style_axes(ax):
    """Simple, consistent, user-friendly chart styling (no data/logic changes)."""
    try:
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.07, linewidth=0.6)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_alpha(0.25)

        ax.tick_params(axis="both", which="major", length=4, width=0.9, colors="0.25")
        ax.tick_params(axis="both", which="minor", length=2, width=0.6, colors="0.35")
        ax.margins(x=0.01)
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

FIB_ALERT_TEXT = "ALERT: Fibonacci Guidance — Prices often reverse at the 100% and 0% lines. It's essential to implement risk management when trading near these Fibonacci levels."

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float = None) -> str:
    """
    UPDATED (prior request):
      - Show BUY instruction only when Global Trendline slope and Local Slope agree (both UP)
      - Show SELL instruction only when Global Trendline slope and Local Slope agree (both DOWN)
      - Otherwise show an alert message.

    Backward-compatibility:
      - If global_trend_slope is None, falls back to the prior behavior (uses only trend_slope).
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    if global_trend_slope is None:
        uptrend = False
        try:
            uptrend = float(trend_slope) >= 0.0
        except Exception:
            pass

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

    alert_txt = ALERT_TEXT

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return alert_txt

    sg = float(np.sign(g))
    sl = float(np.sign(l))

    if sg == 0.0 or sl == 0.0:
        return alert_txt

    if sg > 0 and sl > 0:
        leg_a_val, leg_b_val = entry_buy, exit_sell
        text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    if sg < 0 and sl < 0:
        leg_a_val, leg_b_val = exit_sell, entry_buy
        text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    return alert_txt


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
    _rerun()

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

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

st.sidebar.subheader("NTD (Daily/Hourly)")
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
        "HKDJPY=X","USDCAD=X","USDCNY=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
        "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X","GBPCAD=X"
    ]

# ---------------------------
# Timezone helper (keeps prior behavior: naive -> PACIFIC)
# ---------------------------
def _ensure_pacific_index(obj):
    if isinstance(obj, (pd.Series, pd.DataFrame)) and isinstance(obj.index, pd.DatetimeIndex):
        try:
            if obj.index.tz is None:
                obj = obj.copy()
                obj.index = obj.index.tz_localize(PACIFIC)
            else:
                obj = obj.copy()
                obj.index = obj.index.tz_convert(PACIFIC)
        except Exception:
            pass
    return obj

# ---------------------------
# Data fetchers
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))["Close"]
         .asfreq("D").ffill())
    s = _ensure_pacific_index(s)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")[["Close"]].dropna()
    s = df["Close"].asfreq("D").ffill()
    s = _ensure_pacific_index(s)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[
        ["Open","High","Low","Close"]
    ].dropna()
    df = _ensure_pacific_index(df)
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
# NEW (THIS REQUEST): Fibonacci Buy/Sell markers (Price chart area)
# ---------------------------
def overlay_fib_npx_signals(ax,
                            price: pd.Series,
                            buy_mask: pd.Series,
                            sell_mask: pd.Series,
                            label_buy: str = "Fibonacci BUY",
                            label_sell: str = "Fibonacci SELL"):
    """
    Plot Fibonacci BUY/SELL markers on the PRICE chart.

    Uses buy_mask/sell_mask computed from:
      - price near Fib 100% (low) / 0% (high)
      - NPX crosses 0.0 upward/downward (recent)
    """
    p = _coerce_1d_series(price)
    bm = _coerce_1d_series(buy_mask).reindex(p.index).fillna(0).astype(bool) if buy_mask is not None else pd.Series(False, index=p.index)
    sm = _coerce_1d_series(sell_mask).reindex(p.index).fillna(0).astype(bool) if sell_mask is not None else pd.Series(False, index=p.index)

    buy_idx = list(bm[bm].index)
    sell_idx = list(sm[sm].index)

    if buy_idx:
        ax.scatter(buy_idx, p.loc[buy_idx], marker="^", s=120, color="tab:green", zorder=11, label=label_buy)
        for t in buy_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB BUY", va="bottom", fontsize=9, color="tab:green", fontweight="bold", zorder=12)
            except Exception:
                pass

    if sell_idx:
        ax.scatter(sell_idx, p.loc[sell_idx], marker="v", s=120, color="tab:red", zorder=11, label=label_sell)
        for t in sell_idx:
            try:
                ax.text(t, float(p.loc[t]), "  FIB SELL", va="top", fontsize=9, color="tab:red", fontweight="bold", zorder=12)
            except Exception:
                pass

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

# ---------------------------
# NEW (THIS REQUEST): Fib touch + NPX(0.0) cross logic for Fibonacci BUY/SELL signals
# ---------------------------
def npx_zero_cross_masks(npx: pd.Series, level: float = 0.0):
    """
    NPX cross of a constant level (default 0.0):
      - cross_up: npx goes from < level to >= level
      - cross_dn: npx goes from > level to <= level
    """
    s = _coerce_1d_series(npx)
    prev = s.shift(1)
    cross_up = (s >= float(level)) & (prev < float(level))
    cross_dn = (s <= float(level)) & (prev > float(level))
    return cross_up.fillna(False), cross_dn.fillna(False)

def fib_touch_masks(price: pd.Series, proximity_pct_of_range: float = 0.02):
    """
    Returns (near_hi_0pct, near_lo_100pct, fibs_dict).
    'near' uses a tolerance = proximity_pct_of_range * (fib_range).
    """
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        idx = _coerce_1d_series(price).index
        return (pd.Series(False, index=idx), pd.Series(False, index=idx), {})

    fibs = fibonacci_levels(p)
    if not fibs:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), {})

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return (pd.Series(False, index=p.index), pd.Series(False, index=p.index), fibs)

    near_hi = (p >= (hi - tol)).reindex(p.index, fill_value=False)
    near_lo = (p <= (lo + tol)).reindex(p.index, fill_value=False)
    return near_hi, near_lo, fibs

def fib_npx_zero_cross_signal_masks(price: pd.Series,
                                   npx: pd.Series,
                                   horizon_bars: int = 15,
                                   proximity_pct_of_range: float = 0.02,
                                   npx_level: float = 0.0):
    """
    Fibonacci BUY mask:
      - NPX crosses UP through 0.0
      - AND price touched near Fib 100% (low) within last `horizon_bars` (including current)

    Fibonacci SELL mask:
      - NPX crosses DOWN through 0.0
      - AND price touched near Fib 0% (high) within last `horizon_bars` (including current)
    """
    p = _coerce_1d_series(price)
    x = _coerce_1d_series(npx).reindex(p.index)

    near_hi, near_lo, fibs = fib_touch_masks(p, proximity_pct_of_range=float(proximity_pct_of_range))
    up0, dn0 = npx_zero_cross_masks(x, level=float(npx_level))

    hz = max(1, int(horizon_bars))
    touched_lo_recent = near_lo.rolling(hz + 1, min_periods=1).max().astype(bool)
    touched_hi_recent = near_hi.rolling(hz + 1, min_periods=1).max().astype(bool)

    buy_mask = up0.reindex(p.index, fill_value=False) & touched_lo_recent.reindex(p.index, fill_value=False)
    sell_mask = dn0.reindex(p.index, fill_value=False) & touched_hi_recent.reindex(p.index, fill_value=False)

    return buy_mask.fillna(False), sell_mask.fillna(False), fibs

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
# ==========================================================
# bullbear.py (COMPLETE UPDATED CODE) — BATCH 2 / 3
# Continues immediately from BATCH 1 (Part 5/10 onward)
# ==========================================================

import requests

# =========================
# Part 5/10 — bullbear.py
# =========================
# ---------------------------
# Support / Resistance
# ---------------------------
def rolling_support_resistance_from_ohlc(ohlc: pd.DataFrame, lookback: int = 60):
    """
    Support = rolling min of Low
    Resistance = rolling max of High
    Returns (support_series, resistance_series)
    """
    if ohlc is None or ohlc.empty:
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.Index([])
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    df = ohlc.copy()
    for c in ["High", "Low"]:
        if c not in df.columns:
            return pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)

    lb = max(2, int(lookback))
    sup = pd.to_numeric(df["Low"], errors="coerce").rolling(lb, min_periods=max(2, lb // 3)).min()
    res = pd.to_numeric(df["High"], errors="coerce").rolling(lb, min_periods=max(2, lb // 3)).max()
    return sup.reindex(df.index), res.reindex(df.index)

def rolling_support_resistance_from_close(close: pd.Series, lookback: int = 60):
    s = _coerce_1d_series(close)
    lb = max(2, int(lookback))
    sup = s.rolling(lb, min_periods=max(2, lb // 3)).min()
    res = s.rolling(lb, min_periods=max(2, lb // 3)).max()
    return sup.reindex(s.index), res.reindex(s.index)

# ---------------------------
# ATR / Supertrend (Hourly)
# ---------------------------
def compute_atr(ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype=float)
    if not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.Series(index=ohlc.index, dtype=float)
    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    c = pd.to_numeric(ohlc["Close"], errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / max(2, int(period)), adjust=False).mean()
    return atr.reindex(ohlc.index)

def compute_supertrend(ohlc: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """
    Returns (supertrend_line, direction_series)
      direction: +1 = uptrend, -1 = downtrend
    """
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.Index([])
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    df = ohlc.copy()
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    atr = compute_atr(df, period=period)

    hl2 = (h + l) / 2.0
    upper_basic = hl2 + float(multiplier) * atr
    lower_basic = hl2 - float(multiplier) * atr

    upper_final = upper_basic.copy()
    lower_final = lower_basic.copy()

    for i in range(1, len(df)):
        if np.isfinite(upper_final.iloc[i-1]) and np.isfinite(upper_basic.iloc[i]) and np.isfinite(c.iloc[i-1]):
            if upper_basic.iloc[i] < upper_final.iloc[i-1] or c.iloc[i-1] > upper_final.iloc[i-1]:
                upper_final.iloc[i] = upper_basic.iloc[i]
            else:
                upper_final.iloc[i] = upper_final.iloc[i-1]
        if np.isfinite(lower_final.iloc[i-1]) and np.isfinite(lower_basic.iloc[i]) and np.isfinite(c.iloc[i-1]):
            if lower_basic.iloc[i] > lower_final.iloc[i-1] or c.iloc[i-1] < lower_final.iloc[i-1]:
                lower_final.iloc[i] = lower_basic.iloc[i]
            else:
                lower_final.iloc[i] = lower_final.iloc[i-1]

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    # Initialize
    st_line.iloc[0] = np.nan
    direction.iloc[0] = np.nan

    for i in range(1, len(df)):
        prev_st = st_line.iloc[i-1]
        prev_dir = direction.iloc[i-1]

        cu = upper_final.iloc[i]
        cl = lower_final.iloc[i]
        cc = c.iloc[i]

        if not np.isfinite(cc) or (not np.isfinite(cu)) or (not np.isfinite(cl)):
            st_line.iloc[i] = prev_st
            direction.iloc[i] = prev_dir
            continue

        if not np.isfinite(prev_dir):
            # first decision
            if cc >= cl:
                direction.iloc[i] = 1.0
                st_line.iloc[i] = cl
            else:
                direction.iloc[i] = -1.0
                st_line.iloc[i] = cu
            continue

        if prev_dir > 0:
            if cc < cl:
                direction.iloc[i] = -1.0
                st_line.iloc[i] = cu
            else:
                direction.iloc[i] = 1.0
                st_line.iloc[i] = cl
        else:
            if cc > cu:
                direction.iloc[i] = 1.0
                st_line.iloc[i] = cl
            else:
                direction.iloc[i] = -1.0
                st_line.iloc[i] = cu

    return st_line.reindex(df.index), direction.reindex(df.index)

# ---------------------------
# Parabolic SAR
# ---------------------------
def compute_psar(ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.2):
    """
    Returns psar series aligned to ohlc.index.
    Standard implementation using High/Low.
    """
    if ohlc is None or ohlc.empty or not {"High", "Low"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.Index([])
        return pd.Series(index=idx, dtype=float)

    df = ohlc.copy()
    high = pd.to_numeric(df["High"], errors="coerce").to_numpy()
    low = pd.to_numeric(df["Low"], errors="coerce").to_numpy()
    n = len(df)

    psar = np.full(n, np.nan, dtype=float)
    bull = True  # start bullish
    af = float(step)
    ep = high[0] if np.isfinite(high[0]) else np.nan
    psar[0] = low[0] if np.isfinite(low[0]) else np.nan

    for i in range(1, n):
        prev = psar[i-1]
        if not np.isfinite(prev):
            prev = low[i-1] if bull else high[i-1]

        ps = prev + af * (ep - prev) if np.isfinite(ep) else prev

        if bull:
            # PSAR can't be above prior 2 lows
            if i >= 2:
                ps = min(ps, low[i-1], low[i-2])
            else:
                ps = min(ps, low[i-1])
            # Flip?
            if np.isfinite(low[i]) and low[i] < ps:
                bull = False
                ps = ep  # on flip, PSAR set to EP
                af = float(step)
                ep = low[i]
            else:
                # Update EP and AF
                if np.isfinite(high[i]) and (not np.isfinite(ep) or high[i] > ep):
                    ep = high[i]
                    af = min(af + float(step), float(max_step))
        else:
            # PSAR can't be below prior 2 highs
            if i >= 2:
                ps = max(ps, high[i-1], high[i-2])
            else:
                ps = max(ps, high[i-1])
            # Flip?
            if np.isfinite(high[i]) and high[i] > ps:
                bull = True
                ps = ep
                af = float(step)
                ep = high[i]
            else:
                if np.isfinite(low[i]) and (not np.isfinite(ep) or low[i] < ep):
                    ep = low[i]
                    af = min(af + float(step), float(max_step))

        psar[i] = ps

    return pd.Series(psar, index=df.index)

# ---------------------------
# Ichimoku (Kijun on price)
# ---------------------------
def compute_ichimoku_kijun(ohlc: pd.DataFrame, conv: int = 9, base: int = 26, spanb: int = 52):
    """
    Returns tenkan, kijun, span_a, span_b (all aligned to ohlc.index)
    Only kijun is required for overlay, but we compute the standard set.
    """
    if ohlc is None or ohlc.empty or not {"High", "Low"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.Index([])
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty

    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    conv = max(2, int(conv))
    base = max(2, int(base))
    spanb = max(2, int(spanb))

    tenkan = (h.rolling(conv, min_periods=max(2, conv // 2)).max() + l.rolling(conv, min_periods=max(2, conv // 2)).min()) / 2.0
    kijun  = (h.rolling(base, min_periods=max(2, base // 2)).max() + l.rolling(base, min_periods=max(2, base // 2)).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(base)
    span_b_s = (h.rolling(spanb, min_periods=max(2, spanb // 2)).max() + l.rolling(spanb, min_periods=max(2, spanb // 2)).min()) / 2.0
    span_b_s = span_b_s.shift(base)
    return tenkan.reindex(ohlc.index), kijun.reindex(ohlc.index), span_a.reindex(ohlc.index), span_b_s.reindex(ohlc.index)

# ---------------------------
# HMA-based reversal markers on NTD (panel)
# ---------------------------
def hma_reversal_events_on_series(x: pd.Series, hma_period: int = 55, slope_lb: int = 3):
    """
    For a given series (e.g., NTD), compute HMA(period) and detect points where:
      - HMA slope changes sign (using slope_lb bars)
    Returns dict with 'hma', 'rev_up_mask', 'rev_dn_mask'
    """
    s = _coerce_1d_series(x).astype(float)
    h = compute_hma(s, period=int(hma_period))
    lb = max(2, int(slope_lb))

    # approximate slope sign via finite difference over lb bars
    dh = h - h.shift(lb)
    sign = np.sign(dh)
    sign_prev = sign.shift(1)
    rev_up = (sign > 0) & (sign_prev < 0)
    rev_dn = (sign < 0) & (sign_prev > 0)

    return {
        "hma": h.reindex(s.index),
        "rev_up_mask": rev_up.fillna(False).reindex(s.index, fill_value=False),
        "rev_dn_mask": rev_dn.fillna(False).reindex(s.index, fill_value=False),
    }

def consecutive_reversal_stars(ntd: pd.Series, confirm_bars: int = 2):
    """
    Mark reversal when NTD changes sign and stays on the new side for `confirm_bars` consecutive bars.
    Returns (rev_up_mask, rev_dn_mask).
    """
    x = _coerce_1d_series(ntd)
    if x.empty:
        return pd.Series(index=x.index, dtype=bool), pd.Series(index=x.index, dtype=bool)

    cb = max(1, int(confirm_bars))
    sign = np.sign(x)
    # Up reversal: previously <=0, now >0 for cb bars
    pos = sign > 0
    neg = sign < 0

    pos_run = pos.rolling(cb, min_periods=cb).sum() == cb
    neg_run = neg.rolling(cb, min_periods=cb).sum() == cb

    # reversal moment = first bar of confirmed run AND previous bar was opposite/zero
    rev_up = pos_run & (~pos_run.shift(1).fillna(False)) & (sign.shift(cb).fillna(0) <= 0)
    rev_dn = neg_run & (~neg_run.shift(1).fillna(False)) & (sign.shift(cb).fillna(0) >= 0)

    return rev_up.fillna(False).reindex(x.index, fill_value=False), rev_dn.fillna(False).reindex(x.index, fill_value=False)

# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# FX news markers (optional, intraday)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_forex_factory_calendar_json():
    """
    Uses the commonly-used ForexFactory JSON feed endpoint.
    Returns list[dict]. If the feed is unavailable, returns [].
    """
    url = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def _pair_currencies(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return set()
    base = s[:3]
    quote = s[3:6]
    out = {base, quote}
    # Yahoo sometimes uses CNH, etc; keep as-is
    return out

def _parse_ff_time_to_utc(item: dict):
    """
    Feed usually provides 'date' and 'time' fields, sometimes 'timestamp' (seconds).
    We'll try multiple formats and return UTC datetime (aware), else None.
    """
    try:
        ts = item.get("timestamp", None)
        if ts is not None:
            return datetime.fromtimestamp(int(ts), tz=pytz.UTC)
    except Exception:
        pass

    date_s = str(item.get("date", "")).strip()
    time_s = str(item.get("time", "")).strip()

    # The feed can include "All Day" or empty time; skip those
    if (not date_s) or (not time_s) or ("all" in time_s.lower()):
        return None

    # Try a few known formats
    cand = [
        f"{date_s} {time_s}",
        f"{date_s} {time_s} UTC",
    ]
    for s in cand:
        try:
            dt = pd.to_datetime(s, utc=True)
            if pd.notna(dt):
                return dt.to_pydatetime()
        except Exception:
            continue
    return None

def get_fx_news_markers(symbol: str, window_days: int = 7):
    """
    Returns list of dict: {"time_pst": datetime, "impact": str, "title": str, "currency": str}
    Filtered by symbol currencies and last `window_days` days.
    """
    if "=X" not in str(symbol).upper():
        return []

    data = fetch_forex_factory_calendar_json()
    if not data:
        return []

    cur_set = _pair_currencies(symbol)
    now_utc = datetime.now(tz=pytz.UTC)
    cutoff = now_utc - timedelta(days=max(1, int(window_days)))

    out = []
    for item in data:
        try:
            ccy = str(item.get("currency", "")).upper().strip()
            if ccy not in cur_set:
                continue
            dt_utc = _parse_ff_time_to_utc(item)
            if dt_utc is None:
                continue
            if dt_utc < cutoff or dt_utc > (now_utc + timedelta(days=1)):
                continue
            impact = str(item.get("impact", "")).strip()
            title = str(item.get("title", item.get("event", ""))).strip()
            out.append({
                "time_pst": dt_utc.astimezone(PACIFIC),
                "impact": impact,
                "title": title,
                "currency": ccy
            })
        except Exception:
            continue

    # Sort by time
    out.sort(key=lambda x: x["time_pst"])
    return out

def _impact_to_marker(impact: str):
    s = str(impact).lower()
    if "high" in s:
        return ("D", 80, "tab:red")
    if "medium" in s:
        return ("o", 55, "tab:orange")
    if "low" in s:
        return (".", 35, "0.35")
    return ("o", 40, "0.45")

# ---------------------------
# Session windows (PST) for Forex intraday chart
# ---------------------------
LONDON_TZ = pytz.timezone("Europe/London")
NY_TZ = pytz.timezone("America/New_York")

def _session_bounds_for_date_pst(d_pst: datetime):
    """
    For a PST date (midnight), return London and NY session open/close in PST.
    London: 08:00–17:00 London local
    NY:     08:00–17:00 New York local
    """
    # Make a date in each local TZ
    y, m, d = d_pst.year, d_pst.month, d_pst.day

    lon_open = LONDON_TZ.localize(datetime(y, m, d, 8, 0, 0))
    lon_close = LONDON_TZ.localize(datetime(y, m, d, 17, 0, 0))
    ny_open = NY_TZ.localize(datetime(y, m, d, 8, 0, 0))
    ny_close = NY_TZ.localize(datetime(y, m, d, 17, 0, 0))

    return (
        lon_open.astimezone(PACIFIC),
        lon_close.astimezone(PACIFIC),
        ny_open.astimezone(PACIFIC),
        ny_close.astimezone(PACIFIC),
    )

def overlay_sessions_on_gapless_axis(ax, real_times: pd.DatetimeIndex):
    """
    On a gapless-axis plot where x = bar positions, draw vertical spans for London/NY sessions.
    """
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return

    # Determine unique PST dates in view
    rt = pd.DatetimeIndex(real_times).tz_convert(PACIFIC)
    dates = sorted({datetime(t.year, t.month, t.day, tzinfo=PACIFIC) for t in rt.to_pydatetime()})

    for d0 in dates:
        lon_o, lon_c, ny_o, ny_c = _session_bounds_for_date_pst(d0)
        # Map to nearest bar positions
        lon_pos = _map_times_to_bar_positions(rt, [lon_o, lon_c])
        ny_pos = _map_times_to_bar_positions(rt, [ny_o, ny_c])
        if len(lon_pos) == 2:
            ax.axvspan(lon_pos[0], lon_pos[1], alpha=0.06, color="tab:blue", lw=0)
        if len(ny_pos) == 2:
            ax.axvspan(ny_pos[0], ny_pos[1], alpha=0.05, color="tab:purple", lw=0)

# ---------------------------
# Hourly data fetch
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hourly_ohlc(ticker: str, period: str = "60d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="60m")
    if df is None or df.empty:
        return df
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)
    # Keep standard columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="all")
    return df

# =========================
# Part 7/10 — bullbear.py
# ---------------------------
# Chart builders
# ---------------------------
def _add_bbands(ax, close: pd.Series, win: int, mult: float, use_ema: bool):
    mid, up, lo, pctb, nbb = compute_bbands(close, window=int(win), mult=float(mult), use_ema=bool(use_ema))
    if mid.notna().sum() >= 2:
        ax.plot(mid.index, mid.values, linestyle="--", alpha=0.9, label="BB Mid")
    if up.notna().sum() >= 2 and lo.notna().sum() >= 2:
        ax.plot(up.index, up.values, alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, alpha=0.85, label="-2σ")
        try:
            ax.fill_between(up.index, lo.values, up.values, alpha=0.05)
        except Exception:
            pass
    return mid, up, lo, pctb, nbb

def build_daily_chart(ticker: str,
                      ohlc: pd.DataFrame,
                      daily_close: pd.Series,
                      view_label: str,
                      slope_lb: int,
                      sr_lb: int):
    """
    Daily price chart: close + trendline + ±2σ + fib + S/R + kijun + bbands + triggers.
    Returns (fig, summary_dict)
    """
    df = ohlc.copy() if isinstance(ohlc, pd.DataFrame) else pd.DataFrame()
    close = _coerce_1d_series(daily_close)

    if df.empty and close.empty:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No daily data", ha="center", va="center")
        return fig, {}

    # Apply view slice consistently
    if not df.empty:
        df_view = subset_by_daily_view(df, view_label)
        close_view = df_view["Close"] if "Close" in df_view.columns else close.reindex(df_view.index)
    else:
        close_view = subset_by_daily_view(close.to_frame("Close"), view_label)["Close"]
        df_view = close_view.to_frame("Close")

    close_view = _coerce_1d_series(close_view).dropna()
    if close_view.empty:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No daily data in range", ha="center", va="center")
        return fig, {}

    # Regression and band on *lookback* within view
    yhat, up2, lo2, m, r2 = regression_with_band(close_view, lookback=int(slope_lb), z=2.0)

    # S/R
    if not df_view.empty and {"High", "Low"}.issubset(df_view.columns):
        sup, res = rolling_support_resistance_from_ohlc(df_view, lookback=int(sr_lb))
    else:
        sup, res = rolling_support_resistance_from_close(close_view, lookback=int(sr_lb))

    # Kijun
    kijun = pd.Series(index=close_view.index, dtype=float)
    if show_ichi and (not df_view.empty) and {"High", "Low"}.issubset(df_view.columns):
        _, kijun, _, _ = compute_ichimoku_kijun(df_view, conv=ichi_conv, base=ichi_base, spanb=ichi_spanb)

    # BBands
    mid, bb_up, bb_lo, _, _ = (pd.Series(index=close_view.index, dtype=float),)*5
    if show_bbands:
        mid, bb_up, bb_lo, _, _ = _add_bbands(ax=None, close=close_view, win=bb_win, mult=bb_mult, use_ema=bb_use_ema)  # placeholder

    # Build figure
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close_view.index, close_view.values, label="Close")

    # Regression band
    if yhat.notna().sum() >= 2:
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trendline (m={fmt_slope(m)}, R²={fmt_r2(r2)})")
    if up2.notna().sum() >= 2 and lo2.notna().sum() >= 2:
        ax.plot(up2.index, up2.values, alpha=0.85, label="+2σ (Trend)")
        ax.plot(lo2.index, lo2.values, alpha=0.85, label="-2σ (Trend)")
        try:
            ax.fill_between(up2.index, lo2.values, up2.values, alpha=0.06)
        except Exception:
            pass

    # S/R
    if sup.notna().sum() >= 2:
        ax.plot(sup.index, sup.values, linestyle=":", alpha=0.9, label=f"Support ({sr_lb})")
    if res.notna().sum() >= 2:
        ax.plot(res.index, res.values, linestyle=":", alpha=0.9, label=f"Resistance ({sr_lb})")

    # Kijun
    if show_ichi and kijun.notna().sum() >= 2:
        ax.plot(kijun.index, kijun.values, linestyle="-.", alpha=0.9, label=f"Kijun ({ichi_base})")

    # BBands (recompute with axis now)
    if show_bbands:
        mid, bb_up, bb_lo, pctb, nbb = compute_bbands(close_view, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
        if mid.notna().sum() >= 2:
            ax.plot(mid.index, mid.values, linestyle="--", alpha=0.85, label="BB Mid")
        if bb_up.notna().sum() >= 2 and bb_lo.notna().sum() >= 2:
            ax.plot(bb_up.index, bb_up.values, alpha=0.85, label="BB +σ")
            ax.plot(bb_lo.index, bb_lo.values, alpha=0.85, label="BB -σ")
            try:
                ax.fill_between(bb_up.index, bb_lo.values, bb_up.values, alpha=0.04)
            except Exception:
                pass

    # Fibonacci levels (within view)
    fibs = {}
    if show_fibs:
        fibs = fibonacci_levels(close_view)
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linestyle="--", alpha=0.35)
                label_on_left(ax, v, f"Fib {k} {fmt_price_val(v)}", fontsize=8)

    # Band bounce signal
    band_sig = find_band_bounce_signal(close_view, up2, lo2, m)
    if band_sig:
        annotate_crossover(ax, band_sig["time"], band_sig["price"], band_sig["side"], note="(Band bounce)")

    # Slope trigger after band reversal
    slope_trig = find_slope_trigger_after_band_reversal(close_view, yhat, up2, lo2, horizon=rev_horizon)
    annotate_slope_trigger(ax, slope_trig)

    # Fib reversal trigger from extremes (confirm bars)
    if show_fibs:
        fib_trig = fib_reversal_trigger_from_extremes(close_view, proximity_pct_of_range=0.02,
                                                      confirm_bars=max(1, int(rev_bars_confirm)),
                                                      lookback_bars=max(30, int(slope_lb)))
        if fib_trig:
            side = fib_trig["side"]
            t = fib_trig["last_time"]
            px = fib_trig["last_price"]
            if np.isfinite(px):
                ax.scatter([t], [px], marker="s", s=120,
                           color=("tab:green" if side == "BUY" else "tab:red"),
                           zorder=10, label=f"Fib Reversal {side}")
                ax.text(t, px, f"  Fib Rev {side}", fontsize=9, fontweight="bold",
                        color=("tab:green" if side == "BUY" else "tab:red"),
                        va=("bottom" if side == "BUY" else "top"), zorder=11)

    # Titles & style
    ax.set_title(f"{ticker} — Daily ({view_label})")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)

    summary = {
        "daily_slope": float(m) if np.isfinite(m) else np.nan,
        "daily_r2": float(r2) if np.isfinite(r2) else np.nan,
        "daily_last_close": float(close_view.iloc[-1]) if len(close_view) else np.nan,
    }
    return fig, summary

def build_hourly_price_chart(ticker: str,
                             ohlc_h: pd.DataFrame,
                             slope_lb: int,
                             sr_lb: int):
    """
    Hourly price chart: close + trendline + ±2σ + SR + supertrend + psar + bbands + kijun + signals.
    Returns (fig, details_dict)
    """
    if ohlc_h is None or ohlc_h.empty or "Close" not in ohlc_h.columns:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No hourly data", ha="center", va="center")
        return fig, {}

    df = ohlc_h.copy().dropna(subset=["Close"])
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No hourly close", ha="center", va="center")
        return fig, {}

    # Regression & band (local)
    yhat, up2, lo2, m_local, r2_local = regression_with_band(close, lookback=int(slope_lb), z=2.0)

    # S/R
    sup, res = rolling_support_resistance_from_ohlc(df, lookback=int(sr_lb))

    # Supertrend
    st_line, st_dir = compute_supertrend(df, period=int(atr_period), multiplier=float(atr_mult))

    # PSAR
    psar = compute_psar(df, step=float(psar_step), max_step=float(psar_max)) if show_psar else pd.Series(index=df.index, dtype=float)

    # Kijun
    kijun = pd.Series(index=df.index, dtype=float)
    if show_ichi and {"High", "Low"}.issubset(df.columns):
        _, kijun, _, _ = compute_ichimoku_kijun(df, conv=ichi_conv, base=ichi_base, spanb=ichi_spanb)

    # BBands
    mid, bb_up, bb_lo, pctb, nbb = (pd.Series(index=df.index, dtype=float),)*5
    if show_bbands:
        mid, bb_up, bb_lo, pctb, nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close.index, close.values, label="Close")

    # Trendline and band
    if yhat.notna().sum() >= 2:
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trendline (m={fmt_slope(m_local)}, R²={fmt_r2(r2_local)})")
    if up2.notna().sum() >= 2 and lo2.notna().sum() >= 2:
        ax.plot(up2.index, up2.values, alpha=0.85, label="+2σ (Trend)")
        ax.plot(lo2.index, lo2.values, alpha=0.85, label="-2σ (Trend)")
        try:
            ax.fill_between(up2.index, lo2.values, up2.values, alpha=0.06)
        except Exception:
            pass

    # S/R
    if sup.notna().sum() >= 2:
        ax.plot(sup.index, sup.values, linestyle=":", alpha=0.95, label=f"Support ({sr_lb})")
    if res.notna().sum() >= 2:
        ax.plot(res.index, res.values, linestyle=":", alpha=0.95, label=f"Resistance ({sr_lb})")

    # Supertrend
    if st_line.notna().sum() >= 2:
        ax.plot(st_line.index, st_line.values, linestyle="-", alpha=0.85, label="Supertrend")

    # PSAR
    if show_psar and psar.notna().sum() >= 2:
        ax.scatter(psar.index, psar.values, s=12, alpha=0.75, label="PSAR")

    # Kijun
    if show_ichi and kijun.notna().sum() >= 2:
        ax.plot(kijun.index, kijun.values, linestyle="-.", alpha=0.85, label=f"Kijun ({ichi_base})")

    # BBands
    if show_bbands and mid.notna().sum() >= 2:
        ax.plot(mid.index, mid.values, linestyle="--", alpha=0.8, label="BB Mid")
    if show_bbands and bb_up.notna().sum() >= 2 and bb_lo.notna().sum() >= 2:
        ax.plot(bb_up.index, bb_up.values, alpha=0.8, label="BB +σ")
        ax.plot(bb_lo.index, bb_lo.values, alpha=0.8, label="BB -σ")
        try:
            ax.fill_between(bb_up.index, bb_lo.values, bb_up.values, alpha=0.04)
        except Exception:
            pass

    # Band bounce & slope triggers
    band_sig = find_band_bounce_signal(close, up2, lo2, m_local)
    if band_sig:
        annotate_crossover(ax, band_sig["time"], band_sig["price"], band_sig["side"], note="(Band bounce)")
    slope_trig = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=rev_horizon)
    annotate_slope_trigger(ax, slope_trig)

    ax.set_title(f"{ticker} — Hourly Price")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)

    details = {
        "hourly_slope": float(m_local) if np.isfinite(m_local) else np.nan,
        "hourly_r2": float(r2_local) if np.isfinite(r2_local) else np.nan,
        "hourly_last_close": float(close.iloc[-1]) if len(close) else np.nan,
        "sup": float(_safe_last_float(sup)),
        "res": float(_safe_last_float(res)),
        "supertrend_dir": float(_safe_last_float(st_dir)),
    }
    return fig, details

def build_intraday_gapless_chart(ticker: str,
                                 intraday_df: pd.DataFrame,
                                 show_news: bool,
                                 window_days: int,
                                 show_sessions: bool):
    """
    Intraday chart plotted on a gapless x-axis (bar index) but retains *true* timestamps for tick labels.
    Shows optional FX news markers and session spans (PST).
    """
    if intraday_df is None or intraday_df.empty or "Close" not in intraday_df.columns:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No intraday data", ha="center", va="center")
        return fig

    df = intraday_df.copy()
    df = df.dropna(subset=["Close"])
    if df.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No intraday close", ha="center", va="center")
        return fig

    real_times = pd.DatetimeIndex(df.index)
    x = np.arange(len(df), dtype=int)
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, close, label="Close")

    # Sessions (Forex only; safe to call—if not Forex, show_sessions False)
    if show_sessions:
        overlay_sessions_on_gapless_axis(ax, real_times)

    # News markers
    if show_news:
        try:
            markers = get_fx_news_markers(ticker, window_days=int(window_days))
        except Exception:
            markers = []
        if markers:
            times = [m["time_pst"] for m in markers]
            pos = _map_times_to_bar_positions(real_times.tz_convert(PACIFIC), times)
            for p_i, item in zip(pos, markers):
                mk, sz, col = _impact_to_marker(item.get("impact", ""))
                ax.scatter([p_i], [close[p_i]], marker=mk, s=sz, color=col, alpha=0.9, zorder=9)
            # Legend keys (dedup)
            leg = [
                Line2D([0], [0], marker="D", color="w", markerfacecolor="tab:red", markersize=8, label="High impact"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=7, label="Medium impact"),
                Line2D([0], [0], marker=".", color="w", markerfacecolor="0.35", markersize=10, label="Low impact"),
            ]
            ax.legend(handles=leg, loc="best")

    _apply_compact_time_ticks(ax, real_times, n_ticks=8)
    ax.set_title(f"{ticker} — Intraday (gapless price)")
    ax.set_xlabel("Time (PST labels)")
    ax.set_ylabel("Price")
    style_axes(ax)
    return fig
# ==========================================================
# bullbear.py (COMPLETE UPDATED CODE) — BATCH 3 / 3
# Finishes the file: indicator panels, forecast, Streamlit UI,
# run logic, and app wiring. No omissions.
# ==========================================================

# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# (Fix) BBands helper: allow ax=None
# ---------------------------
def _add_bbands(ax, close: pd.Series, win: int, mult: float, use_ema: bool):
    """
    If ax is None: compute and return BB components only.
    If ax provided: also plot.
    """
    mid, up, lo, pctb, nbb = compute_bbands(close, window=int(win), mult=float(mult), use_ema=bool(use_ema))
    if ax is not None:
        if mid.notna().sum() >= 2:
            ax.plot(mid.index, mid.values, linestyle="--", alpha=0.9, label="BB Mid")
        if up.notna().sum() >= 2 and lo.notna().sum() >= 2:
            ax.plot(up.index, up.values, alpha=0.85, label="+2σ")
            ax.plot(lo.index, lo.values, alpha=0.85, label="-2σ")
            try:
                ax.fill_between(up.index, lo.values, up.values, alpha=0.05)
            except Exception:
                pass
    return mid, up, lo, pctb, nbb

# ---------------------------
# NTD / NPX (panels)
# ---------------------------
def compute_ntd_npx(close: pd.Series,
                    ntd_ema_span: int = 55,
                    npx_z_win: int = 55):
    """
    NTD: % deviation from EMA (normalized trend deviation)
    NPX: rolling z-score of price vs rolling mean/std (normalized price extension)
    """
    c = _coerce_1d_series(close).astype(float)
    if c.empty:
        idx = c.index
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    ema = c.ewm(span=max(2, int(ntd_ema_span)), adjust=False).mean()
    ntd = 100.0 * (c / ema - 1.0)

    w = max(10, int(npx_z_win))
    mu = c.rolling(w, min_periods=max(5, w // 3)).mean()
    sd = c.rolling(w, min_periods=max(5, w // 3)).std(ddof=0)
    npx = (c - mu) / sd
    return ntd.reindex(c.index), npx.reindex(c.index)

def fibonacci_levels_series(x: pd.Series):
    """
    Fib levels across min/max of the series in-view.
    Returns dict of level_name -> y_value.
    """
    s = _coerce_1d_series(x).dropna().astype(float)
    if s.empty:
        return {}
    lo = float(np.nanmin(s.values))
    hi = float(np.nanmax(s.values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return {}
    rng = hi - lo
    # classic retracements measured from high toward low
    levels = {
        "0.0": hi,
        "23.6": hi - 0.236 * rng,
        "38.2": hi - 0.382 * rng,
        "50.0": hi - 0.500 * rng,
        "61.8": hi - 0.618 * rng,
        "78.6": hi - 0.786 * rng,
        "100.0": lo,
    }
    return levels

def plot_fib_levels(ax, levels: dict, prefix: str = "Fib", alpha: float = 0.22):
    if not levels:
        return
    for k, v in levels.items():
        if np.isfinite(v):
            ax.axhline(v, linestyle="--", alpha=alpha)
            try:
                label_on_left(ax, v, f"{prefix} {k} {v:,.3f}", fontsize=8)
            except Exception:
                pass

def build_ntd_panel(ticker: str,
                    close: pd.Series,
                    view_label: str,
                    ntd_ema_span: int,
                    npx_z_win: int,
                    hma_period: int,
                    hma_slope_lb: int,
                    confirm_bars: int,
                    show_fibs_local: bool = True):
    """
    One figure: NTD with HMA and reversal markers + optional fib levels + BUY/SELL fib reversal marker.
    """
    c = _coerce_1d_series(close).dropna()
    if c.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data for NTD/NPX", ha="center", va="center")
        return fig, {}

    # Apply same daily view slicing approach for indicator panels
    c_view = subset_by_daily_view(c.to_frame("Close"), view_label)["Close"].dropna()
    if c_view.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data in view", ha="center", va="center")
        return fig, {}

    ntd, npx = compute_ntd_npx(c_view, ntd_ema_span=ntd_ema_span, npx_z_win=npx_z_win)

    events = hma_reversal_events_on_series(ntd, hma_period=hma_period, slope_lb=hma_slope_lb)
    hma = events["hma"]
    rev_up = events["rev_up_mask"]
    rev_dn = events["rev_dn_mask"]

    stars_up, stars_dn = consecutive_reversal_stars(ntd, confirm_bars=confirm_bars)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axhline(0, linestyle="--", alpha=0.35)

    ax.plot(ntd.index, ntd.values, label="NTD")
    if hma.notna().sum() >= 2:
        ax.plot(hma.index, hma.values, linestyle="-", alpha=0.85, label=f"HMA({hma_period})")

    # Markers: HMA slope flips
    if rev_up.any():
        ax.scatter(ntd.index[rev_up], ntd[rev_up], marker="^", s=90, alpha=0.9, label="HMA Rev Up")
    if rev_dn.any():
        ax.scatter(ntd.index[rev_dn], ntd[rev_dn], marker="v", s=90, alpha=0.9, label="HMA Rev Down")

    # Markers: consecutive sign-confirm reversals (stars)
    if stars_up.any():
        ax.scatter(ntd.index[stars_up], ntd[stars_up], marker="*", s=140, alpha=0.9, label=f"Confirm Up ({confirm_bars})")
    if stars_dn.any():
        ax.scatter(ntd.index[stars_dn], ntd[stars_dn], marker="*", s=140, alpha=0.9, label=f"Confirm Down ({confirm_bars})")

    # Fib levels on NTD (in-view)
    fib_levels = fibonacci_levels_series(ntd) if show_fibs_local else {}
    if show_fibs_local and fib_levels:
        plot_fib_levels(ax, fib_levels, prefix="NTD Fib", alpha=0.20)

    # Fib reversal trigger marker (BUY/SELL)
    fib_trig = None
    try:
        fib_trig = fib_reversal_trigger_from_extremes(
            ntd,
            proximity_pct_of_range=0.06,
            confirm_bars=max(1, int(confirm_bars)),
            lookback_bars=max(40, int(hma_period) * 2),
        )
    except Exception:
        fib_trig = None

    if fib_trig:
        side = fib_trig["side"]
        t = fib_trig["last_time"]
        px = fib_trig["last_price"]
        if np.isfinite(px):
            ax.scatter([t], [px], marker="s", s=160,
                       color=("tab:green" if side == "BUY" else "tab:red"),
                       zorder=10, label=f"NTD Fib Reversal {side}")
            ax.text(t, px, f"  {side}", fontsize=10, fontweight="bold",
                    color=("tab:green" if side == "BUY" else "tab:red"),
                    va=("bottom" if side == "BUY" else "top"), zorder=11)

    ax.set_title(f"{ticker} — NTD Panel ({view_label})")
    ax.set_ylabel("NTD (% vs EMA)")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)

    details = {
        "ntd_last": float(ntd.iloc[-1]) if len(ntd) else np.nan,
        "npx_last": float(npx.iloc[-1]) if len(npx) else np.nan,
    }
    return fig, details

def build_npx_panel(ticker: str,
                    close: pd.Series,
                    view_label: str,
                    ntd_ema_span: int,
                    npx_z_win: int,
                    show_fibs_local: bool = True):
    """
    NPX-only panel with optional fib levels + fib reversal marker.
    """
    c = _coerce_1d_series(close).dropna()
    if c.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data for NPX", ha="center", va="center")
        return fig, {}

    c_view = subset_by_daily_view(c.to_frame("Close"), view_label)["Close"].dropna()
    if c_view.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data in view", ha="center", va="center")
        return fig, {}

    _, npx = compute_ntd_npx(c_view, ntd_ema_span=ntd_ema_span, npx_z_win=npx_z_win)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0, linestyle="--", alpha=0.35)
    ax.axhline(2, linestyle=":", alpha=0.25)
    ax.axhline(-2, linestyle=":", alpha=0.25)

    ax.plot(npx.index, npx.values, label="NPX (z-score)")

    fib_levels = fibonacci_levels_series(npx) if show_fibs_local else {}
    if show_fibs_local and fib_levels:
        plot_fib_levels(ax, fib_levels, prefix="NPX Fib", alpha=0.18)

    fib_trig = None
    try:
        fib_trig = fib_reversal_trigger_from_extremes(
            npx,
            proximity_pct_of_range=0.08,
            confirm_bars=max(1, int(rev_bars_confirm)),
            lookback_bars=max(50, int(npx_z_win) * 2),
        )
    except Exception:
        fib_trig = None

    if fib_trig:
        side = fib_trig["side"]
        t = fib_trig["last_time"]
        px = fib_trig["last_price"]
        if np.isfinite(px):
            ax.scatter([t], [px], marker="s", s=160,
                       color=("tab:green" if side == "BUY" else "tab:red"),
                       zorder=10, label=f"NPX Fib Reversal {side}")
            ax.text(t, px, f"  {side}", fontsize=10, fontweight="bold",
                    color=("tab:green" if side == "BUY" else "tab:red"),
                    va=("bottom" if side == "BUY" else "top"), zorder=11)

    ax.set_title(f"{ticker} — NPX Panel ({view_label})")
    ax.set_ylabel("NPX (rolling z)")
    style_axes(ax)
    ax.legend(loc="best")
    details = {"npx_last": float(npx.iloc[-1]) if len(npx) else np.nan}
    return fig, details

# ---------------------------
# MACD panel
# ---------------------------
def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        idx = s.index
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty

    f = max(2, int(fast))
    sl = max(f + 1, int(slow))
    sg = max(2, int(signal))

    ema_fast = s.ewm(span=f, adjust=False).mean()
    ema_slow = s.ewm(span=sl, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=sg, adjust=False).mean()
    hist = macd - sig
    return macd.reindex(s.index), sig.reindex(s.index), hist.reindex(s.index)

def build_macd_panel(ticker: str, close: pd.Series, view_label: str):
    c = _coerce_1d_series(close).dropna()
    if c.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data for MACD", ha="center", va="center")
        return fig, {}

    c_view = subset_by_daily_view(c.to_frame("Close"), view_label)["Close"].dropna()
    if c_view.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data in view", ha="center", va="center")
        return fig, {}

    macd, sig, hist = compute_macd(c_view, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axhline(0, linestyle="--", alpha=0.35)
    ax.plot(macd.index, macd.values, label="MACD")
    ax.plot(sig.index, sig.values, linestyle="--", alpha=0.85, label="Signal")
    # histogram as thin bars
    try:
        ax.bar(hist.index, hist.values, width=0.8, alpha=0.25, label="Hist")
    except Exception:
        pass

    # cross markers
    cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    cross_dn = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    if cross_up.any():
        ax.scatter(macd.index[cross_up], macd[cross_up], marker="^", s=80, alpha=0.9, label="Cross Up")
    if cross_dn.any():
        ax.scatter(macd.index[cross_dn], macd[cross_dn], marker="v", s=80, alpha=0.9, label="Cross Down")

    ax.set_title(f"{ticker} — MACD Panel ({view_label})")
    ax.set_ylabel("MACD")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)
    details = {
        "macd_last": float(macd.iloc[-1]) if len(macd) else np.nan,
        "macd_sig_last": float(sig.iloc[-1]) if len(sig) else np.nan,
        "macd_hist_last": float(hist.iloc[-1]) if len(hist) else np.nan,
    }
    return fig, details

# ---------------------------
# Forecast panel (lightweight, deterministic)
# ---------------------------
def linear_forecast_with_band(close: pd.Series,
                              lookback: int = 90,
                              horizon: int = 20):
    """
    Linear regression on last `lookback` points; projects `horizon` steps forward.
    Confidence band is ±2*RMSE of residuals (simple heuristic).
    Returns forecast_index, yhat, upper, lower.
    """
    s = _coerce_1d_series(close).dropna().astype(float)
    if s.empty:
        return pd.DatetimeIndex([]), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    lb = max(20, int(lookback))
    y = s.iloc[-lb:] if len(s) > lb else s
    x = np.arange(len(y), dtype=float)
    if len(y) < 3:
        return pd.DatetimeIndex([]), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    # Fit line
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    m, b = float(coef[0]), float(coef[1])
    y_fit = m * x + b

    resid = y.values - y_fit
    rmse = float(np.sqrt(np.nanmean(resid ** 2))) if np.isfinite(np.nanmean(resid ** 2)) else 0.0

    h = max(1, int(horizon))
    x_f = np.arange(len(y), len(y) + h, dtype=float)
    y_f = m * x_f + b

    # Build index forward: preserve frequency if possible; else use business days.
    idx = y.index
    if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
        f_idx = pd.date_range(start=idx[-1] + idx.freq, periods=h, freq=idx.freq)
    else:
        # infer step from last two points
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
            step = idx[-1] - idx[-2]
            if step <= pd.Timedelta(0):
                step = pd.Timedelta(days=1)
            f_idx = pd.DatetimeIndex([idx[-1] + step * (i + 1) for i in range(h)])
        else:
            f_idx = pd.date_range(start=pd.Timestamp.utcnow(), periods=h, freq="D")

    yhat = pd.Series(y_f, index=f_idx)
    upper = yhat + 2.0 * rmse
    lower = yhat - 2.0 * rmse
    return f_idx, yhat, upper, lower

def build_forecast_panel(ticker: str,
                         close: pd.Series,
                         view_label: str,
                         lookback: int,
                         horizon: int):
    c = _coerce_1d_series(close).dropna()
    if c.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data for forecast", ha="center", va="center")
        return fig, {}

    c_view = subset_by_daily_view(c.to_frame("Close"), view_label)["Close"].dropna()
    if c_view.empty:
        fig = plt.figure(figsize=(12, 4))
        plt.text(0.5, 0.5, "No data in view", ha="center", va="center")
        return fig, {}

    f_idx, yhat, up, lo = linear_forecast_with_band(c_view, lookback=int(lookback), horizon=int(horizon))

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(c_view.index, c_view.values, label="Close")
    if len(yhat):
        ax.plot(yhat.index, yhat.values, linestyle="--", label="Forecast (linear)")
        ax.plot(up.index, up.values, alpha=0.6, label="Forecast +2RMSE")
        ax.plot(lo.index, lo.values, alpha=0.6, label="Forecast -2RMSE")
        try:
            ax.fill_between(up.index, lo.values, up.values, alpha=0.06)
        except Exception:
            pass

    ax.set_title(f"{ticker} — Forecast ({view_label})")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)

    details = {
        "forecast_last": float(yhat.iloc[-1]) if len(yhat) else np.nan,
        "forecast_upper_last": float(up.iloc[-1]) if len(up) else np.nan,
        "forecast_lower_last": float(lo.iloc[-1]) if len(lo) else np.nan,
    }
    return fig, details

# =========================
# Part 9/10 — bullbear.py
# ---------------------------
# Intraday data fetch (gapless chart uses df.index as real time)
# ---------------------------
@st.cache_data(ttl=120)
def fetch_intraday_ohlc(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    yfinance intraday. For FX, 5m/15m is often okay, but availability varies.
    """
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return df
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(how="all")
    return df

def _is_fx_symbol(ticker: str) -> bool:
    return "=X" in str(ticker).upper()

def _summary_cards(daily_sum: dict, hourly_sum: dict, ntd_sum: dict, macd_sum: dict, forecast_sum: dict):
    cols = st.columns(5)
    with cols[0]:
        st.metric("Daily slope", fmt_slope(daily_sum.get("daily_slope", np.nan)))
        st.metric("Daily R²", fmt_r2(daily_sum.get("daily_r2", np.nan)))
    with cols[1]:
        st.metric("Hourly slope", fmt_slope(hourly_sum.get("hourly_slope", np.nan)))
        st.metric("Hourly R²", fmt_r2(hourly_sum.get("hourly_r2", np.nan)))
    with cols[2]:
        st.metric("NTD last", f"{ntd_sum.get('ntd_last', np.nan):.2f}" if np.isfinite(ntd_sum.get("ntd_last", np.nan)) else "—")
        st.metric("NPX last", f"{ntd_sum.get('npx_last', np.nan):.2f}" if np.isfinite(ntd_sum.get("npx_last", np.nan)) else "—")
    with cols[3]:
        st.metric("MACD", f"{macd_sum.get('macd_last', np.nan):.3f}" if np.isfinite(macd_sum.get("macd_last", np.nan)) else "—")
        st.metric("Hist", f"{macd_sum.get('macd_hist_last', np.nan):.3f}" if np.isfinite(macd_sum.get("macd_hist_last", np.nan)) else "—")
    with cols[4]:
        st.metric("Forecast", fmt_price_val(forecast_sum.get("forecast_last", np.nan)))
        st.metric("Band (±)", f"{abs(forecast_sum.get('forecast_upper_last', np.nan) - forecast_sum.get('forecast_last', np.nan)):.2f}"
                  if np.isfinite(forecast_sum.get("forecast_upper_last", np.nan)) and np.isfinite(forecast_sum.get("forecast_last", np.nan))
                  else "—")

# =========================
# Part 10/10 — bullbear.py
# ---------------------------
# Streamlit UI + app wiring
# ---------------------------
def run_app():
    st.set_page_config(page_title="BullBear", layout="wide")

    st.title("BullBear — Multi-timeframe Trend + Reversal Dashboard")

    # ---- Sidebar controls ----
    with st.sidebar:
        st.header("Controls")

        ticker = st.text_input("Ticker (Yahoo)", value="SPY").strip().upper()
        if not ticker:
            st.stop()

        # Views
        view_label = st.selectbox("Daily view", options=DAILY_VIEW_CHOICES, index=0)

        # Daily / Hourly lookbacks
        slope_lb = st.slider("Trendline lookback (bars)", min_value=30, max_value=260, value=120, step=5)
        sr_lb = st.slider("S/R lookback (bars)", min_value=20, max_value=200, value=60, step=5)

        st.divider()
        st.subheader("Overlays")

        # Set globals used by chart builders from Batch B
        global show_fibs, show_bbands, bb_win, bb_mult, bb_use_ema
        global show_ichi, ichi_conv, ichi_base, ichi_spanb
        global atr_period, atr_mult
        global show_psar, psar_step, psar_max
        global rev_horizon, rev_bars_confirm

        show_fibs = st.checkbox("Show Fibonacci (price panels)", value=True)
        show_bbands = st.checkbox("Show Bollinger Bands", value=True)
        bb_win = st.slider("BB window", 10, 120, 20, 1)
        bb_mult = st.slider("BB mult (σ)", 1.0, 4.0, 2.0, 0.1)
        bb_use_ema = st.checkbox("BB midline uses EMA", value=False)

        show_ichi = st.checkbox("Show Ichimoku Kijun", value=False)
        ichi_conv = st.slider("Ichimoku Tenkan (conv)", 5, 20, 9, 1)
        ichi_base = st.slider("Ichimoku Kijun (base)", 10, 80, 26, 1)
        ichi_spanb = st.slider("Ichimoku SpanB", 26, 120, 52, 1)

        st.divider()
        st.subheader("Hourly Indicators")
        atr_period = st.slider("Supertrend ATR period", 5, 30, 10, 1)
        atr_mult = st.slider("Supertrend multiplier", 1.0, 6.0, 3.0, 0.1)

        show_psar = st.checkbox("Show PSAR", value=False)
        psar_step = st.slider("PSAR step", 0.01, 0.10, 0.02, 0.01)
        psar_max = st.slider("PSAR max", 0.10, 0.50, 0.20, 0.01)

        st.divider()
        st.subheader("Reversal Logic")
        rev_horizon = st.slider("Reversal horizon (bars)", 3, 30, 10, 1)
        rev_bars_confirm = st.slider("Confirm bars", 1, 6, 2, 1)

        st.divider()
        st.subheader("NTD / NPX")
        ntd_ema_span = st.slider("NTD EMA span", 10, 200, 55, 1)
        npx_z_win = st.slider("NPX z-score window", 20, 200, 55, 1)
        ntd_hma_period = st.slider("NTD HMA period", 10, 200, 55, 1)
        ntd_hma_slope_lb = st.slider("HMA slope lookback", 2, 10, 3, 1)

        st.divider()
        st.subheader("MACD")
        global macd_fast, macd_slow, macd_signal
        macd_fast = st.slider("MACD fast", 5, 20, 12, 1)
        macd_slow = st.slider("MACD slow", 10, 60, 26, 1)
        macd_signal = st.slider("MACD signal", 3, 20, 9, 1)

        st.divider()
        st.subheader("Forecast")
        fc_lookback = st.slider("Forecast lookback", 30, 260, 120, 5)
        fc_horizon = st.slider("Forecast horizon (steps)", 5, 90, 20, 1)

        st.divider()
        st.subheader("Intraday (FX gapless)")
        intraday_interval = st.selectbox("Intraday interval", options=["5m", "15m", "30m", "60m"], index=0)
        intraday_period = st.selectbox("Intraday period", options=["5d", "10d", "30d", "60d"], index=0)
        show_news = st.checkbox("Show FX news markers", value=_is_fx_symbol(ticker))
        show_sessions = st.checkbox("Show London/NY sessions", value=_is_fx_symbol(ticker))
        news_window_days = st.slider("News window (days)", 1, 14, 7, 1)

    # ---- Data fetch ----
    with st.spinner("Fetching data..."):
        daily_df = fetch_daily_ohlc(ticker)  # defined earlier (Batch A)
        hourly_df = fetch_hourly_ohlc(ticker, period="60d")
        intraday_df = fetch_intraday_ohlc(ticker, period=intraday_period, interval=intraday_interval)

    # Make sure we have a daily close series for panels
    daily_close = None
    if isinstance(daily_df, pd.DataFrame) and (not daily_df.empty) and ("Close" in daily_df.columns):
        daily_close = pd.to_numeric(daily_df["Close"], errors="coerce")
    else:
        daily_close = pd.Series(dtype=float)

    # ---- Build figures ----
    daily_fig, daily_sum = build_daily_chart(
        ticker=ticker,
        ohlc=daily_df,
        daily_close=daily_close,
        view_label=view_label,
        slope_lb=int(slope_lb),
        sr_lb=int(sr_lb),
    )

    hourly_fig, hourly_sum = build_hourly_price_chart(
        ticker=ticker,
        ohlc_h=hourly_df,
        slope_lb=int(slope_lb),
        sr_lb=int(sr_lb),
    )

    intraday_fig = build_intraday_gapless_chart(
        ticker=ticker,
        intraday_df=intraday_df,
        show_news=bool(show_news) and _is_fx_symbol(ticker),
        window_days=int(news_window_days),
        show_sessions=bool(show_sessions) and _is_fx_symbol(ticker),
    )

    ntd_fig, ntd_sum = build_ntd_panel(
        ticker=ticker,
        close=daily_close,
        view_label=view_label,
        ntd_ema_span=int(ntd_ema_span),
        npx_z_win=int(npx_z_win),
        hma_period=int(ntd_hma_period),
        hma_slope_lb=int(ntd_hma_slope_lb),
        confirm_bars=int(rev_bars_confirm),
        show_fibs_local=True,
    )

    npx_fig, _ = build_npx_panel(
        ticker=ticker,
        close=daily_close,
        view_label=view_label,
        ntd_ema_span=int(ntd_ema_span),
        npx_z_win=int(npx_z_win),
        show_fibs_local=True,
    )

    macd_fig, macd_sum = build_macd_panel(
        ticker=ticker,
        close=daily_close,
        view_label=view_label,
    )

    forecast_fig, forecast_sum = build_forecast_panel(
        ticker=ticker,
        close=daily_close,
        view_label=view_label,
        lookback=int(fc_lookback),
        horizon=int(fc_horizon),
    )

    # ---- Summary row ----
    st.subheader("Snapshot")
    _summary_cards(daily_sum, hourly_sum, ntd_sum, macd_sum, forecast_sum)

    # ---- Tabs ----
    tab_names = ["Daily", "Hourly", "Intraday", "NTD", "NPX", "MACD", "Forecast", "Data"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.pyplot(daily_fig, clear_figure=True, use_container_width=True)
        st.caption("Daily: close + trendline ±2σ band + optional BBands, Kijun, support/resistance, fib, and reversal markers.")

    with tabs[1]:
        st.pyplot(hourly_fig, clear_figure=True, use_container_width=True)
        st.caption("Hourly: close + trendline ±2σ band + supertrend + optional PSAR/BBands/Kijun + reversal markers.")

    with tabs[2]:
        st.pyplot(intraday_fig, clear_figure=True, use_container_width=True)
        if _is_fx_symbol(ticker):
            st.caption("Intraday (FX): gapless x-axis chart with optional London/NY sessions and ForexFactory news markers.")
        else:
            st.caption("Intraday: gapless x-axis chart (news/session overlays are FX-only).")

    with tabs[3]:
        st.pyplot(ntd_fig, clear_figure=True, use_container_width=True)
        st.caption("NTD: % deviation from EMA with HMA reversal markers, confirm-stars, and fib reversal BUY/SELL marker.")

    with tabs[4]:
        st.pyplot(npx_fig, clear_figure=True, use_container_width=True)
        st.caption("NPX: rolling z-score extension with fib reversal BUY/SELL marker.")

    with tabs[5]:
        st.pyplot(macd_fig, clear_figure=True, use_container_width=True)
        st.caption("MACD: MACD line, signal, histogram, and cross markers.")

    with tabs[6]:
        st.pyplot(forecast_fig, clear_figure=True, use_container_width=True)
        st.caption("Forecast: simple linear projection with ±2×RMSE band (deterministic heuristic).")

    with tabs[7]:
        st.write("Daily (tail):")
        st.dataframe(daily_df.tail(20) if isinstance(daily_df, pd.DataFrame) else pd.DataFrame())
        st.write("Hourly (tail):")
        st.dataframe(hourly_df.tail(40) if isinstance(hourly_df, pd.DataFrame) else pd.DataFrame())
        st.write("Intraday (tail):")
        st.dataframe(intraday_df.tail(60) if isinstance(intraday_df, pd.DataFrame) else pd.DataFrame())

    st.divider()
    st.caption("Note: This app is informational only and not financial advice.")

if __name__ == "__main__":
    run_app()
