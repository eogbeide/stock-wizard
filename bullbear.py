# /mount/src/stock-wizard/bullbear.py
# =========================
# Batch 1/3 — bullbear.py  (UPDATED: 15 tabs + HMA Buy)
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

  /* =========================
     Beautiful rectangular ribbon tabs (BaseWeb tabs)
     ========================= */
  div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    gap: 0.45rem !important;
    padding: 0.35rem 0.35rem 0.25rem 0.35rem !important;
    border-bottom: 1px solid rgba(49, 51, 63, 0.18) !important;
  }
  div[data-baseweb="tab"] { flex: 0 0 auto !important; }

  div[data-baseweb="tab"] > button,
  div[data-baseweb="tab"] button {
    border: 1px solid rgba(49, 51, 63, 0.22) !important;
    background: rgba(255,255,255,0.92) !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 6px !important;       /* rectangular ribbon (not pill) */
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease !important;
    white-space: nowrap !important;
  }
  div[data-baseweb="tab"] > button:hover,
  div[data-baseweb="tab"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.10) !important;
  }
  div[data-baseweb="tab"] > button[aria-selected="true"],
  div[data-baseweb="tab"] button[aria-selected="true"] {
    background: rgba(49, 51, 63, 0.92) !important;
    color: white !important;
    border-color: rgba(49, 51, 63, 0.92) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,0.16) !important;
  }
  div[data-baseweb="tab"] > button:focus,
  div[data-baseweb="tab"] button:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(49, 51, 63, 0.18) !important;
  }

  /* =========================
     Beautiful chart container styling
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
    border-radius: 12px;
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
    Show BUY only when Global slope and Local slope both UP.
    Show SELL only when Global slope and Local slope both DOWN.
    Otherwise show alert message.
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

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return ALERT_TEXT

    sg = float(np.sign(g))
    sl = float(np.sign(l))

    if sg == 0.0 or sl == 0.0:
        return ALERT_TEXT

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

    return ALERT_TEXT

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
    """
    Compact, readable x-axis labeling (no overlap) for intraday bar-index charts.
    """
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
    try:
        st.experimental_rerun()
    except Exception:
        pass

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

# ---------------------------
# Universe selection
# ---------------------------
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
# Symbol picker (keeps original behavior; avoids duplicate keys)
# ---------------------------
def _symbol_picker():
    if mode == "Forex":
        sel = st.sidebar.selectbox("Forex Symbol", universe, index=0, key="dd_forex_symbol")
    else:
        sel = st.sidebar.selectbox("Stock Symbol", universe, index=0, key="dd_stock_symbol")
    return sel

sel_symbol = _symbol_picker()

run_btn = st.sidebar.button("🚀 Run / Refresh Charts", use_container_width=True, key="btn_run_all")

if "run_all" not in st.session_state:
    st.session_state.run_all = False
if run_btn:
    st.session_state.run_all = True
    st.session_state.ticker = sel_symbol
    st.session_state.mode_at_run = mode

# Always keep ticker synced for display (no UI change)
if "ticker" not in st.session_state or st.session_state.ticker is None:
    st.session_state.ticker = sel_symbol

ticker = st.session_state.ticker

# =========================
# Batch 2/3 continues below…
# =========================
# =========================
# Batch 2/3 — bullbear.py
# =========================

# ---------------------------
# Data fetchers
# ---------------------------
def _tz_fix_index(idx, tz):
    if not isinstance(idx, pd.DatetimeIndex):
        return idx
    try:
        if idx.tz is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)
    except Exception:
        return idx

@st.cache_data(ttl=120)
def fetch_hist(tkr: str) -> pd.Series:
    df = yf.download(tkr, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns multiindex columns
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            pass
    s = _coerce_1d_series(df.get("Close", pd.Series(dtype=float))).asfreq("D").ffill()
    s.index = _tz_fix_index(s.index, PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(tkr: str) -> pd.DataFrame:
    df = yf.download(tkr, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            pass
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    out = df[keep].dropna()
    out.index = _tz_fix_index(out.index, PACIFIC)
    return out

@st.cache_data(ttl=120)
def fetch_intraday(tkr: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(tkr, period=period, interval="5m", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = [c[0] for c in df.columns]
        except Exception:
            pass
    try:
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    except Exception:
        pass
    try:
        df.index = df.index.tz_convert(PACIFIC)
    except Exception:
        pass

    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)

    return df

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        return pd.DatetimeIndex([]), pd.Series(dtype=float), pd.DataFrame()
    if isinstance(series.index, pd.DatetimeIndex):
        series.index = _tz_fix_index(series.index, PACIFIC)
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception:
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

# ---------------------------
# Regression & ±2σ band
# ---------------------------
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
    return yhat_s, upper_s, lower_s, float(m), float(r2)

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

# ---------------------------
# Indicators
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

# ---------------------------
# Ichimoku, Supertrend, PSAR (kept for chart overlays)
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

# ---------------------------
# Scanner helpers (kept)
# ---------------------------
@st.cache_data(ttl=120)
def last_band_bounce_signal_daily(symbol: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        yhat, up, lo, m, r2 = regression_with_band(close_full, lookback=int(slope_lb))
        sig = find_band_bounce_signal(close_full, up, lo, m)
        if sig is None:
            return None
        t = sig.get("time", None)
        if t is None or t not in close_full.index:
            return None
        loc = int(close_full.index.get_loc(t))
        bars_since = int((len(close_full) - 1) - loc)

        curr = float(close_full.iloc[-1]) if np.isfinite(close_full.iloc[-1]) else np.nan
        spx = float(sig.get("price", np.nan))
        dlt = (curr / spx - 1.0) if np.isfinite(curr) and np.isfinite(spx) and spx != 0 else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Side": sig.get("side", ""),
            "Bars Since": bars_since,
            "Signal Time": t,
            "Signal Price": spx,
            "Current Price": curr,
            "DeltaPct": dlt,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_band_bounce_signal_hourly(symbol: str, period: str, slope_lb: int):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if hc.empty:
            return None

        yhat, up, lo, m, r2 = regression_with_band(hc, lookback=int(slope_lb))
        sig = find_band_bounce_signal(hc, up, lo, m)
        if sig is None:
            return None

        bar = sig.get("time", None)
        try:
            bar = int(bar)
        except Exception:
            return None
        n = len(hc)
        if bar < 0 or bar >= n:
            return None
        bars_since = int((n - 1) - bar)

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        curr = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan
        spx = float(sig.get("price", np.nan))
        dlt = (curr / spx - 1.0) if np.isfinite(curr) and np.isfinite(spx) and spx != 0 else np.nan

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Side": sig.get("side", ""),
            "Bars Since": bars_since,
            "Signal Time": ts,
            "Signal Price": spx,
            "Current Price": curr,
            "DeltaPct": dlt,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def daily_regression_r2(symbol: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, np.nan, None
        _, _, _, m, r2 = regression_with_band(close_full, lookback=int(slope_lb))
        ts = close_full.index[-1] if isinstance(close_full.index, pd.DatetimeIndex) and len(close_full.index) else None
        return float(r2) if np.isfinite(r2) else np.nan, float(m) if np.isfinite(m) else np.nan, ts
    except Exception:
        return np.nan, np.nan, None

@st.cache_data(ttl=120)
def hourly_regression_r2(symbol: str, period: str, slope_lb: int):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return np.nan, np.nan, None

        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if hc.empty:
            ts = real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None
            return np.nan, np.nan, ts

        _, _, _, m, r2 = regression_with_band(hc, lookback=int(slope_lb))
        ts = real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None
        return float(r2) if np.isfinite(r2) else np.nan, float(m) if np.isfinite(m) else np.nan, ts
    except Exception:
        return np.nan, np.nan, None

@st.cache_data(ttl=120)
def daily_r2_band_proximity(symbol: str,
                            daily_view_label: str,
                            slope_lb: int,
                            prox: float,
                            z: float = 2.0):
    """
    Daily-only:
      - uses the selected Daily view range (subset_by_daily_view)
      - computes regression_with_band over slope_lb
      - checks proximity to the last ±zσ band values
      - 'Near' is abs(distance %) <= prox (where prox is sr_prox_pct, already a fraction)
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty:
            return None

        slope_lb = int(max(2, slope_lb))
        if len(close_show) < max(6, slope_lb + 2):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=slope_lb, z=float(z))
        if lo is None or up is None or lo.dropna().empty or up.dropna().empty:
            return None

        px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        lo_last = float(lo.iloc[-1]) if np.isfinite(lo.iloc[-1]) else np.nan
        up_last = float(up.iloc[-1]) if np.isfinite(up.iloc[-1]) else np.nan

        if not np.all(np.isfinite([px, lo_last, up_last])):
            return None
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None

        dist_lo = (px / lo_last - 1.0) if lo_last != 0 else np.nan
        dist_up = (px / up_last - 1.0) if up_last != 0 else np.nan

        abs_lo = abs(dist_lo) if np.isfinite(dist_lo) else np.nan
        abs_up = abs(dist_up) if np.isfinite(dist_up) else np.nan

        prox = abs(float(prox))
        near_lo = bool(np.isfinite(abs_lo) and abs_lo <= prox)
        near_up = bool(np.isfinite(abs_up) and abs_up <= prox)

        return {
            "Symbol": symbol,
            "Daily View": daily_view_label,
            "AsOf": close_show.index[-1] if isinstance(close_show.index, pd.DatetimeIndex) and len(close_show.index) else None,
            "Price": px,
            "Lower -2σ": lo_last,
            "Upper +2σ": up_last,
            "Dist Lower (%)": dist_lo,
            "Dist Upper (%)": dist_up,
            "AbsDist Lower (%)": abs_lo,
            "AbsDist Upper (%)": abs_up,
            "Slope": float(m),
            "R2": float(r2),
            "Near Lower": near_lo,
            "Near Upper": near_up,
            "Slope LB": int(slope_lb),
        }
    except Exception:
        return None

# ---------------------------
# REMOVALS (THIS REQUEST)
#   - Removed: NPX 0.5-Cross Scanner tab + logic
#   - Removed: Fib NPX 0.0 Signal Scanner tab + logic
#   - Removed: News tab + intraday news markers + sidebar controls
#   - Removed: Ichimoku Kijun Scanner tab + logic
# (Chart overlays for Ichimoku/HMA/BBands/etc remain unchanged)
# ---------------------------

# ---------------------------
# NEW (THIS REQUEST): HMA Buy scanner helper
#   - price crosses ABOVE HMA(55) on DAILY, within last 1..N bars (N slider: 1–3)
#   - then bucket by regression slope sign (Regression > 0 / Regression < 0)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_hma_cross_up(symbol: str,
                            daily_view_label: str,
                            hma_len: int,
                            within_last_n_bars: int,
                            slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(5, int(hma_len) + 3, int(slope_lb) + 3):
            return None

        hma = compute_hma(close_show, period=int(hma_len))
        hma = _coerce_1d_series(hma).reindex(close_show.index)

        cross_up, _ = _cross_series(close_show, hma)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)

        if not cross_up.any():
            return None

        t_cross = cross_up[cross_up].index[-1]
        loc = int(close_show.index.get_loc(t_cross))
        bars_since = int((len(close_show) - 1) - loc)

        # Requirement: "recently crossed ... by 1-3 bars"
        within_last_n_bars = int(max(1, within_last_n_bars))
        if not (1 <= bars_since <= within_last_n_bars):
            return None

        px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        hma_cross = float(hma.loc[t_cross]) if (t_cross in hma.index and np.isfinite(hma.loc[t_cross])) else np.nan
        px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        hma_last = float(hma.dropna().iloc[-1]) if len(hma.dropna()) else np.nan

        # Regression slope over the same daily view window (matches charts)
        _, _, _, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not np.isfinite(m):
            return None

        return {
            "Symbol": symbol,
            "Bars Since Cross": bars_since,
            "Cross Time": t_cross,
            "Price@Cross": px_cross,
            f"HMA({hma_len})@Cross": hma_cross,
            "Current Price": px_last,
            f"HMA({hma_len}) (last)": hma_last,
            "Regression Slope": float(m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

# =========================
# Batch 3/3 continues below…
# =========================
# =========================
# Batch 3/3 — bullbear.py
# =========================

# ---------------------------
# Renderers
# ---------------------------
def render_daily_price_chart(symbol: str, daily_view_label: str):
    close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
    if close_full.empty:
        st.warning("No daily data available.")
        return

    close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
    if close.empty:
        st.warning("No daily data in selected range.")
        return

    ohlc = fetch_hist_ohlc(symbol)
    if ohlc is None or ohlc.empty:
        ohlc = pd.DataFrame(index=close.index, data={"High": close, "Low": close, "Close": close})

    ohlc = ohlc.sort_index()
    ohlc = ohlc.loc[(ohlc.index >= close.index[0]) & (ohlc.index <= close.index[-1])]

    # Indicators
    yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb_daily))
    sup = close.rolling(int(sr_lb_daily), min_periods=1).min()
    res = close.rolling(int(sr_lb_daily), min_periods=1).max()

    ntd = compute_normalized_trend(close, window=int(ntd_window)) if show_ntd else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close, window=int(ntd_window)) if show_ntd else pd.Series(index=close.index, dtype=float)

    hma = compute_hma(close, period=int(hma_period))
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    kijun = pd.Series(index=close.index, dtype=float)
    if show_ichi and {"High","Low","Close"}.issubset(ohlc.columns):
        _, kijun, _, _, _ = ichimoku_lines(
            ohlc["High"], ohlc["Low"], ohlc["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
            shift_cloud=False
        )
        kijun = _coerce_1d_series(kijun).reindex(close.index).ffill().bfill()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax.plot(close.index, close.values, "-", label="Close")

    global_m = draw_trend_direction_line(ax, close, label_prefix="Trend (global)")

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and (not bb_up.dropna().empty) and (not bb_lo.dropna().empty):
        ax.fill_between(close.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Slope {slope_lb_daily} bars ({fmt_slope(m)}/bar)")
    if not up.empty and not lo.empty:
        ax.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")
        sig = find_band_bounce_signal(close, up, lo, m)
        if sig is not None:
            annotate_crossover(ax, sig["time"], sig["price"], sig["side"])

    # Support/Resistance
    if len(sup.dropna()) and len(res.dropna()):
        s_val = float(sup.iloc[-1]) if np.isfinite(sup.iloc[-1]) else np.nan
        r_val = float(res.iloc[-1]) if np.isfinite(res.iloc[-1]) else np.nan
        if np.isfinite(s_val) and np.isfinite(r_val):
            ax.hlines(r_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
            ax.hlines(s_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
            label_on_left(ax, r_val, f"R {fmt_price_val(r_val)}", color="tab:red")
            label_on_left(ax, s_val, f"S {fmt_price_val(s_val)}", color="tab:green")

    # Fibonacci
    if show_fibs:
        fibs = fibonacci_levels(close)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs.items():
            ax.text(close.index[-1], y, f" {lbl}", va="center")

    # Title + callouts
    px_val = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
    ax.set_title(f"{symbol} Daily Price ({daily_view_label})  |  R²: {fmt_r2(r2)}  |  Slope: {fmt_slope(m)}/bar")

    if np.isfinite(px_val):
        try:
            last_nbb = float(bb_nbb.dropna().iloc[-1]) if show_bbands else np.nan
        except Exception:
            last_nbb = np.nan
        ax.text(0.99, 0.02,
                f"Current price: {fmt_price_val(px_val)}" + (f"  |  NBB {last_nbb:+.2f}" if np.isfinite(last_nbb) else ""),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    style_axes(ax)
    ax.legend(loc="upper left", ncols=3)
    st.pyplot(fig, clear_figure=True)

    # NTD panel (below)
    if show_ntd and show_nrsi:
        fig2, axn = plt.subplots(figsize=(14, 3.2))
        fig2.patch.set_facecolor("white")
        axn.axhline(0, color="black", alpha=0.25, linewidth=1)

        if shade_ntd:
            pos = ntd.where(ntd > 0)
            neg = ntd.where(ntd < 0)
            axn.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
            axn.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

        axn.plot(ntd.index, ntd.values, "-", linewidth=1.8, label="NTD")
        if show_npx_ntd:
            axn.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX")
        axn.set_title(f"{symbol} NTD (Daily) — window={ntd_window}")
        style_axes(axn)
        axn.legend(loc="upper left", ncols=4)
        st.pyplot(fig2, clear_figure=True)

def render_intraday(symbol: str, hour_range: str):
    period_map = {"24h": "1d", "5d": "5d", "1mo": "1mo"}
    period = period_map.get(hour_range, "1d")
    df = fetch_intraday(symbol, period=period)
    if df is None or df.empty or "Close" not in df.columns:
        st.warning("No intraday data available.")
        return

    real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

    intr_plot = df.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    hc = _coerce_1d_series(intr_plot["Close"]).ffill()

    he = hc.ewm(span=20).mean()

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    hma_h = compute_hma(hc, period=int(hma_period))
    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if show_ichi and {"High","Low","Close"}.issubset(intr_plot.columns):
        _, kijun_h, _, _, _ = ichimoku_lines(
            intr_plot["High"], intr_plot["Low"], intr_plot["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
            shift_cloud=False
        )
        kijun_h = _coerce_1d_series(kijun_h).reindex(hc.index).ffill().bfill()

    st_line_intr = pd.Series(index=hc.index, dtype=float)
    try:
        st_df = compute_supertrend(intr_plot, atr_period=int(atr_period), atr_mult=float(atr_mult))
        if "ST" in st_df.columns:
            st_line_intr = _coerce_1d_series(st_df["ST"]).reindex(hc.index)
    except Exception:
        pass

    psar_df = pd.DataFrame()
    if show_psar:
        try:
            psar_df = compute_psar_from_ohlc(intr_plot, step=float(psar_step), max_step=float(psar_max)).reindex(hc.index)
        except Exception:
            psar_df = pd.DataFrame()

    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, lookback=int(slope_lb_hourly))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")

    ax.plot(hc.index, hc, label="Intraday")
    ax.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax, hc, label_prefix="Trend (global)")

    if show_hma and not hma_h.dropna().empty:
        ax.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    # Support/Resistance
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
    except Exception:
        res_val, sup_val = np.nan, np.nan

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not st_line_intr.dropna().empty:
        ax.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.empty:
        ax.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")
        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, m_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

    # Fibonacci
    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax.text(hc.index[-1], y, f" {lbl}", va="center")

    # Labels
    px_val = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan
    ax.set_title(f"{symbol} Intraday ({hour_range})  |  R²: {fmt_r2(r2_h)}  |  Slope: {fmt_slope(m_h)}/bar")
    if np.isfinite(px_val):
        ax.text(0.99, 0.02,
                f"Current price: {fmt_price_val(px_val)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    style_axes(ax)
    ax.legend(loc="upper left", ncols=3)

    # Use compact, readable x-axis ticks for intraday
    if isinstance(real_times, pd.DatetimeIndex) and len(real_times):
        _apply_compact_time_ticks(ax, real_times, n_ticks=8)

    st.pyplot(fig, clear_figure=True)

# ---------------------------
# Tab Layout (15 tabs)
#   Removed tabs:
#     - NPX 0.5-Cross Scanner
#     - Fib NPX 0.0 Signal Scanner
#     - News
#     - Ichimoku Kijun Scanner
#   Added tab:
#     - HMA Buy
# ---------------------------
tabs = st.tabs([
    "Forecast",
    "Enhanced Forecast",
    "Daily Price",
    "Intraday",
    "Bull/Bear",
    "Metrics",
    "Recent Band Bounce",
    "R² (Daily)",
    "R² (Hourly)",
    "R² + Band Proximity",
    "NTD (Daily)",
    "Support/Resistance (Daily)",
    "Support/Resistance (Hourly)",
    "Signals",
    "HMA Buy",  # NEW
])

# ---------------------------
# Shared prep
# ---------------------------
close_full = _coerce_1d_series(fetch_hist(ticker)).dropna()
ohlc_full = fetch_hist_ohlc(ticker)

# ---------------------------
# Tab 1: Forecast
# ---------------------------
with tabs[0]:
    st.subheader("Forecast (SARIMAX)")
    if close_full.empty:
        st.warning("No data.")
    else:
        with st.spinner("Computing forecast..."):
            idx, fc, ci = compute_sarimax_forecast(close_full)
        if len(idx) == 0 or fc is None or len(fc) == 0:
            st.warning("Forecast unavailable.")
        else:
            fig, ax = plt.subplots(figsize=(14, 5))
            fig.patch.set_facecolor("white")
            hist_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
            ax.plot(hist_show.index, hist_show.values, label="History")
            ax.plot(idx, fc.values, "--", label="Forecast")
            try:
                ci2 = ci.copy()
                ci2.index = idx
                ax.fill_between(idx, ci2.iloc[:, 0].values, ci2.iloc[:, 1].values, alpha=0.12, label="CI")
            except Exception:
                pass
            ax.set_title(f"{ticker} — 30D Forecast")
            style_axes(ax)
            ax.legend(loc="upper left")
            st.pyplot(fig, clear_figure=True)

# ---------------------------
# Tab 2: Enhanced Forecast (adds regression + ±2σ on daily)
# ---------------------------
with tabs[1]:
    st.subheader("Enhanced Forecast (Daily regression + ±2σ)")
    if close_full.empty:
        st.warning("No data.")
    else:
        hist_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(hist_show, lookback=int(slope_lb_daily))
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("white")

        ax.plot(hist_show.index, hist_show.values, label="Close")
        if not yhat.empty:
            ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Regression ({fmt_slope(m)}/bar, R²={fmt_r2(r2)})")
        if not up.empty and not lo.empty:
            ax.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
            ax.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")
            sig = find_band_bounce_signal(hist_show, up, lo, m)
            if sig is not None:
                annotate_crossover(ax, sig["time"], sig["price"], sig["side"])

        ax.set_title(f"{ticker} — Enhanced (Daily)")
        style_axes(ax)
        ax.legend(loc="upper left", ncols=3)
        st.pyplot(fig, clear_figure=True)

# ---------------------------
# Tab 3: Daily Price
# ---------------------------
with tabs[2]:
    st.subheader("Daily Price Chart")
    render_daily_price_chart(ticker, daily_view)

# ---------------------------
# Tab 4: Intraday
# ---------------------------
with tabs[3]:
    st.subheader("Intraday Chart (5m)")
    if "hour_range" not in st.session_state:
        st.session_state.hour_range = "24h"
    st.session_state.hour_range = st.selectbox("Range:", ["24h", "5d", "1mo"], index=["24h","5d","1mo"].index(st.session_state.hour_range), key="dd_intraday_range")
    render_intraday(ticker, st.session_state.hour_range)

# ---------------------------
# Tab 5: Bull/Bear
# ---------------------------
with tabs[4]:
    st.subheader("Bull / Bear Snapshot")
    if close_full.empty:
        st.warning("No data.")
    else:
        lb_map = {"1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}
        lb = lb_map.get(bb_period, 132)
        s = close_full.tail(lb)
        if len(s) < 5:
            st.warning("Not enough data for Bull/Bear window.")
        else:
            ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if np.isfinite(s.iloc[-1]) and np.isfinite(s.iloc[0]) and float(s.iloc[0]) != 0 else np.nan
            up_days = int((s.diff() > 0).sum())
            dn_days = int((s.diff() < 0).sum())
            st.metric("Return", fmt_pct(ret))
            st.write(pd.DataFrame([{
                "Window": bb_period,
                "Start": s.index[0],
                "End": s.index[-1],
                "Up Days": up_days,
                "Down Days": dn_days,
            }]))

# ---------------------------
# Tab 6: Metrics
# ---------------------------
with tabs[5]:
    st.subheader("Metrics")
    if close_full.empty:
        st.warning("No data.")
    else:
        s = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(slope_lb_daily))
        h = compute_hma(s, period=int(hma_period))
        bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(s, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
        ntd = compute_normalized_trend(s, window=int(ntd_window))
        npx = compute_normalized_price(s, window=int(ntd_window))
        out = {
            "Symbol": ticker,
            "Daily View": daily_view,
            "Current Price": float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan,
            f"HMA({hma_period})": float(h.dropna().iloc[-1]) if len(h.dropna()) else np.nan,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NTD (last)": float(ntd.dropna().iloc[-1]) if len(ntd.dropna()) else np.nan,
            "NPX (last)": float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan,
            "NBB (last)": float(bb_nbb.dropna().iloc[-1]) if len(bb_nbb.dropna()) else np.nan,
        }
        st.dataframe(pd.DataFrame([out]), use_container_width=True)

# ---------------------------
# Tab 7: Recent Band Bounce Scanner
# ---------------------------
with tabs[6]:
    st.subheader("Recent Band Bounce (Scanner)")
    max_bars = st.slider("Show signals within last N bars:", 1, 30, 10, 1, key="sl_bb_recent_n")
    rows = []
    with st.spinner("Scanning..."):
        for sym in universe:
            r = last_band_bounce_signal_daily(sym, slope_lb=int(slope_lb_daily))
            if r is None:
                continue
            if int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)
    if not rows:
        st.info("No signals found.")
    else:
        df = pd.DataFrame(rows).sort_values(["Bars Since", "Symbol"])
        st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 8: R² (Daily)
# ---------------------------
with tabs[7]:
    st.subheader("R² Scanner (Daily)")
    r2_min = st.slider("Min R² (Daily):", 0.0, 1.0, 0.45, 0.01, key="sl_r2_min_daily")
    rows = []
    with st.spinner("Scanning..."):
        for sym in universe:
            r2v, mv, ts = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2v) and float(r2v) >= float(r2_min):
                rows.append({"Symbol": sym, "R2": float(r2v), "Slope": float(mv) if np.isfinite(mv) else np.nan, "AsOf": ts})
    if not rows:
        st.info("No symbols found.")
    else:
        df = pd.DataFrame(rows).sort_values(["R2","Symbol"], ascending=[False, True])
        st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 9: R² (Hourly)
# ---------------------------
with tabs[8]:
    st.subheader("R² Scanner (Hourly)")
    hr_period = st.selectbox("Hourly window:", ["1d", "5d", "1mo"], index=0, key="dd_hr_r2_period")
    r2_min = st.slider("Min R² (Hourly):", 0.0, 1.0, 0.45, 0.01, key="sl_r2_min_hourly")
    rows = []
    with st.spinner("Scanning..."):
        for sym in universe:
            r2v, mv, ts = hourly_regression_r2(sym, period=hr_period, slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2v) and float(r2v) >= float(r2_min):
                rows.append({"Symbol": sym, "R2": float(r2v), "Slope": float(mv) if np.isfinite(mv) else np.nan, "AsOf": ts})
    if not rows:
        st.info("No symbols found.")
    else:
        df = pd.DataFrame(rows).sort_values(["R2","Symbol"], ascending=[False, True])
        st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 10: R² + Band Proximity
# ---------------------------
with tabs[9]:
    st.subheader("R² + Band Proximity (Daily)")
    prox = float(sr_prox_pct)
    rows = []
    with st.spinner("Scanning..."):
        for sym in universe:
            r = daily_r2_band_proximity(sym, daily_view_label=daily_view, slope_lb=int(slope_lb_daily), prox=prox, z=2.0)
            if r is None:
                continue
            # keep ones near either band
            if bool(r.get("Near Lower", False)) or bool(r.get("Near Upper", False)):
                rows.append(r)
    if not rows:
        st.info("No symbols near ±2σ bands (within selected proximity).")
    else:
        df = pd.DataFrame(rows).sort_values(["R2","Symbol"], ascending=[False, True])
        st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 11: NTD (Daily)
# ---------------------------
with tabs[10]:
    st.subheader("NTD (Daily)")
    if close_full.empty:
        st.warning("No data.")
    else:
        s = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        ntd = compute_normalized_trend(s, window=int(ntd_window))
        npx = compute_normalized_price(s, window=int(ntd_window))
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor("white")
        ax.axhline(0, color="black", alpha=0.25, linewidth=1)
        if shade_ntd:
            pos = ntd.where(ntd > 0)
            neg = ntd.where(ntd < 0)
            ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
            ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")
        ax.plot(ntd.index, ntd.values, "-", linewidth=1.8, label="NTD")
        if show_npx_ntd:
            ax.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX")
        ax.set_title(f"{ticker} — NTD (window={ntd_window})")
        style_axes(ax)
        ax.legend(loc="upper left", ncols=4)
        st.pyplot(fig, clear_figure=True)

# ---------------------------
# Tab 12: Support/Resistance (Daily)
# ---------------------------
with tabs[11]:
    st.subheader("Support / Resistance (Daily)")
    if close_full.empty:
        st.warning("No data.")
    else:
        s = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        sup = s.rolling(int(sr_lb_daily), min_periods=1).min()
        res = s.rolling(int(sr_lb_daily), min_periods=1).max()
        df = pd.DataFrame({
            "Close": s,
            "Support": sup,
            "Resistance": res,
            "Dist vs Support (%)": (s / sup - 1.0),
            "Dist vs Resistance (%)": (res / s - 1.0),
        })
        st.dataframe(df.tail(120), use_container_width=True)

# ---------------------------
# Tab 13: Support/Resistance (Hourly)
# ---------------------------
with tabs[12]:
    st.subheader("Support / Resistance (Hourly)")
    hr_period = st.selectbox("Window:", ["1d", "5d", "1mo"], index=0, key="dd_hr_sr_period")
    df = fetch_intraday(ticker, period=hr_period)
    if df is None or df.empty or "Close" not in df.columns:
        st.warning("No intraday data.")
    else:
        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill()
        sup = hc.rolling(int(sr_lb_hourly), min_periods=1).min()
        res = hc.rolling(int(sr_lb_hourly), min_periods=1).max()
        out = pd.DataFrame({"Close": hc, "Support": sup, "Resistance": res})
        st.dataframe(out.tail(240), use_container_width=True)

# ---------------------------
# Tab 14: Signals (summary)
# ---------------------------
with tabs[13]:
    st.subheader("Signals (Summary)")
    if close_full.empty:
        st.warning("No data.")
    else:
        s = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(slope_lb_daily))
        sup = s.rolling(int(sr_lb_daily), min_periods=1).min()
        res = s.rolling(int(sr_lb_daily), min_periods=1).max()
        sup_val = float(sup.iloc[-1]) if len(sup) and np.isfinite(sup.iloc[-1]) else np.nan
        res_val = float(res.iloc[-1]) if len(res) and np.isfinite(res.iloc[-1]) else np.nan
        px_val = float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan

        instr = format_trade_instruction(
            trend_slope=float(m) if np.isfinite(m) else np.nan,
            buy_val=sup_val,
            sell_val=res_val,
            close_val=px_val,
            symbol=ticker,
            global_trend_slope=float(m) if np.isfinite(m) else np.nan
        )
        st.info(instr)

# ---------------------------
# Tab 15: HMA Buy (NEW)
# ---------------------------
with tabs[14]:
    st.subheader("HMA Buy (Daily) — Recent Close↑HMA(55) Cross")

    # Slider required by your request: 1–3 bars
    bars_window = st.slider("Cross must be within last N bars (1–3):", 1, 3, 3, 1, key="sl_hma_buy_bars")

    # Use HMA(55) for this tab regardless of sidebar HMA period
    fixed_hma_len = 55

    rows_pos = []
    rows_neg = []

    with st.spinner("Scanning universe for recent HMA(55) cross-ups..."):
        for sym in universe:
            r = last_daily_hma_cross_up(
                sym,
                daily_view_label=daily_view,
                hma_len=fixed_hma_len,
                within_last_n_bars=int(bars_window),
                slope_lb=int(slope_lb_daily),
            )
            if r is None:
                continue
            m = float(r.get("Regression Slope", np.nan))
            if np.isfinite(m) and m > 0:
                rows_pos.append(r)
            elif np.isfinite(m) and m < 0:
                rows_neg.append(r)

    st.markdown("### (a) Regression > 0")
    if not rows_pos:
        st.info("No symbols found for Regression > 0.")
    else:
        dfp = pd.DataFrame(rows_pos).sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
        st.dataframe(dfp, use_container_width=True)

    st.markdown("### (b) Regression < 0")
    if not rows_neg:
        st.info("No symbols found for Regression < 0.")
    else:
        dfn = pd.DataFrame(rows_neg).sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
        st.dataframe(dfn, use_container_width=True)
