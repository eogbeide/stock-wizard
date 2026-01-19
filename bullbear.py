# /bullbear.py

# =========================
# Part 1/10 — bullbear.py  (UPDATED: Ribbon Tabs + Beautiful Chart Styling)
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
# NEW (THIS REQUEST): optional scipy for slope p-values (99.9% confidence gating)
# ---------------------------
try:
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:
    _scipy_stats = None

GLOBAL_TREND_CONFIDENCE = 0.999
GLOBAL_TREND_ALPHA = 1.0 - GLOBAL_TREND_CONFIDENCE  # 0.001

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
     UPDATED (THIS REQUEST):
     (1) Beautiful rectangular ribbon tabs (BaseWeb tabs)
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
     UPDATED (THIS REQUEST):
     (2) Beautiful chart container styling (Streamlit React UI wrappers)
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

# ---------------------------
# NEW (THIS REQUEST): slope p-value / confidence helpers for "global trendline 99.9% confidence"
# ---------------------------
def linear_regression_slope_pvalue(series_like):
    """
    Returns (slope, intercept, r2, p_value) for y ~ a + b*x, two-sided p-value for slope!=0.
    Uses scipy if available; falls back to normal approximation for large n.
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    y = s.to_numpy(dtype=float)
    n = int(len(y))
    x = np.arange(n, dtype=float)

    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")

    yhat = m * x + b
    resid = y - yhat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    dof = max(n - 2, 1)
    mse = ss_res / float(dof)

    xbar = float(np.mean(x))
    sxx = float(np.sum((x - xbar) ** 2))
    if not np.isfinite(sxx) or sxx <= 0:
        return float(m), float(b), float(r2), float("nan")

    se_m = float(np.sqrt(mse / sxx)) if np.isfinite(mse) and mse >= 0 else float("nan")
    if not np.isfinite(se_m) or se_m == 0.0:
        return float(m), float(b), float(r2), float("nan")

    t_stat = float(m / se_m)
    if not np.isfinite(t_stat):
        return float(m), float(b), float(r2), float("nan")

    if _scipy_stats is not None:
        try:
            p = float(2.0 * (1.0 - _scipy_stats.t.cdf(abs(t_stat), df=dof)))
        except Exception:
            p = float("nan")
    else:
        # Normal approximation (ok when n is reasonably large)
        try:
            z = abs(t_stat)
            p = float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))))
        except Exception:
            p = float("nan")

    return float(m), float(b), float(r2), float(p)

def slope_is_confident(p_value: float, confidence: float = GLOBAL_TREND_CONFIDENCE) -> bool:
    """
    True when slope!=0 at the requested confidence (default 99.9% => p < 0.001).
    """
    try:
        pv = float(p_value)
        conf = float(confidence)
    except Exception:
        return False
    if not (np.isfinite(pv) and np.isfinite(conf)):
        return False
    alpha = 1.0 - conf
    return bool(pv < alpha)

def global_trend_stats(series_like, confidence: float = GLOBAL_TREND_CONFIDENCE):
    """
    Returns dict for global trend gating:
      - slope, r2, p_value
      - ok_up: slope>0 and p<alpha
      - ok_dn: slope<0 and p<alpha
    """
    m, b, r2, p = linear_regression_slope_pvalue(series_like)
    ok = slope_is_confident(p, confidence=confidence)
    ok_up = bool(ok and np.isfinite(m) and float(m) > 0.0)
    ok_dn = bool(ok and np.isfinite(m) and float(m) < 0.0)
    return {"slope": m, "r2": r2, "p_value": p, "ok_up": ok_up, "ok_dn": ok_dn}

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

# ---------------------------
# NEW (THIS REQUEST): "recently reversed from ±2σ band" (within last `rev_horizon` bars)
#   Used later by:
#     • Daily chart BUY/SELL signals (Req 1/2)
#     • New tab list (Req 3)
#     • Slope Direction Scan subsets (Req 4)
# ---------------------------
def recent_band_reversal_signal(price: pd.Series,
                                upper_band: pd.Series,
                                lower_band: pd.Series,
                                side: str,
                                horizon_bars: int,
                                confirm_bars: int = 2):
    """
    BUY (side="BUY"):
      - touched/breached LOWER band within last `horizon_bars`
      - then crossed back ABOVE the LOWER band (back inside)
      - and price is currently "going upward" (last `confirm_bars` diffs > 0)

    SELL (side="SELL"):
      - touched/breached UPPER band within last `horizon_bars`
      - then crossed back BELOW the UPPER band (back inside)
      - and price is currently "going downward" (last `confirm_bars` diffs < 0)

    Returns most recent qualifying reversal dict or None.
    """
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None

    p = p[ok]; u = u[ok]; l = l[ok]
    hz = max(1, int(horizon_bars))
    cb = max(1, int(confirm_bars))

    def _last_n_deltas(seg: pd.Series, n: int):
        s = _coerce_1d_series(seg).dropna()
        if len(s) < n + 1:
            return None
        return np.diff(s.iloc[-(n + 1):].to_numpy(dtype=float))

    tail_idx = p.index[-(hz + 1):] if len(p) > (hz + 1) else p.index

    side_u = str(side).upper().strip()
    if side_u.startswith("B"):
        touch = (p <= l)
        cross_back = (p >= l) & (p.shift(1) < l.shift(1))

        touch_t = touch.loc[tail_idx]
        cross_t = cross_back.loc[tail_idx]

        if not (touch_t.any() and cross_t.any()):
            return None

        t_cross = cross_t[cross_t].index[-1]
        touch_before = touch.loc[:t_cross]
        if not touch_before.any():
            return None
        t_touch = touch_before[touch_before].index[-1]

        deltas = _last_n_deltas(p, cb)
        if deltas is None or not bool(np.all(deltas > 0)):
            return None

        px_cross = float(p.loc[t_cross]) if np.isfinite(p.loc[t_cross]) else np.nan
        px_now = float(p.iloc[-1]) if np.isfinite(p.iloc[-1]) else np.nan
        if not (np.isfinite(px_cross) and np.isfinite(px_now) and px_now >= px_cross):
            return None

        return {
            "side": "BUY",
            "touch_time": t_touch,
            "touch_price": float(p.loc[t_touch]) if np.isfinite(p.loc[t_touch]) else np.nan,
            "signal_time": t_cross,
            "signal_price": px_cross,
        }

    if side_u.startswith("S"):
        touch = (p >= u)
        cross_back = (p <= u) & (p.shift(1) > u.shift(1))

        touch_t = touch.loc[tail_idx]
        cross_t = cross_back.loc[tail_idx]

        if not (touch_t.any() and cross_t.any()):
            return None

        t_cross = cross_t[cross_t].index[-1]
        touch_before = touch.loc[:t_cross]
        if not touch_before.any():
            return None
        t_touch = touch_before[touch_before].index[-1]

        deltas = _last_n_deltas(p, cb)
        if deltas is None or not bool(np.all(deltas < 0)):
            return None

        px_cross = float(p.loc[t_cross]) if np.isfinite(p.loc[t_cross]) else np.nan
        px_now = float(p.iloc[-1]) if np.isfinite(p.iloc[-1]) else np.nan
        if not (np.isfinite(px_cross) and np.isfinite(px_now) and px_now <= px_cross):
            return None

        return {
            "side": "SELL",
            "touch_time": t_touch,
            "touch_price": float(p.loc[t_touch]) if np.isfinite(p.loc[t_touch]) else np.nan,
            "signal_time": t_cross,
            "signal_price": px_cross,
        }

    return None

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
# =========================
# Part 5/10 — bullbear.py  (UPDATED: BBands/HMA/Ichimoku + signal helpers)
# =========================
def compute_wma(series_like, period: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    p = int(period)
    if s.empty or p < 2:
        return pd.Series(index=s.index, dtype=float)
    weights = np.arange(1, p + 1, dtype=float)

    def _w(x):
        x = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(x)):
            return np.nan
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(p, min_periods=p).apply(_w, raw=True).reindex(s.index)

def compute_hma(series_like, period: int = 55) -> pd.Series:
    """
    HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    s = _coerce_1d_series(series_like).astype(float)
    n = int(period)
    if s.empty or n < 2:
        return pd.Series(index=s.index, dtype=float)
    n2 = max(1, n // 2)
    ns = max(1, int(np.sqrt(n)))
    w1 = compute_wma(s, n2)
    w2 = compute_wma(s, n)
    raw = 2.0 * w1 - w2
    return compute_wma(raw, ns).reindex(s.index)

def compute_bollinger(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    w = int(window)
    if s.empty or w < 2:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    mid = s.ewm(span=w, adjust=False).mean() if use_ema else s.rolling(w, min_periods=max(3, w // 3)).mean()
    sd = s.rolling(w, min_periods=max(3, w // 3)).std().replace(0, np.nan)
    upper = mid + float(mult) * sd
    lower = mid - float(mult) * sd
    return mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index)

def compute_kijun(high: pd.Series, low: pd.Series, period: int = 26) -> pd.Series:
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    p = int(period)
    if h.empty or l.empty or p < 2:
        return pd.Series(index=h.index if len(h) else l.index, dtype=float)
    hh = h.rolling(p, min_periods=max(3, p // 3)).max()
    ll = l.rolling(p, min_periods=max(3, p // 3)).min()
    return ((hh + ll) / 2.0).reindex(h.index)

def _near_pct(x: float, y: float, pct: float) -> bool:
    try:
        xv = float(x); yv = float(y); pv = float(pct)
    except Exception:
        return False
    if not (np.isfinite(xv) and np.isfinite(yv) and np.isfinite(pv)):
        return False
    if yv == 0:
        return False
    return abs(xv - yv) / abs(yv) <= pv

def strict_daily_2sigma_signal(close_view: pd.Series,
                              local_lookback: int,
                              rev_horizon_bars: int,
                              confirm_bars: int,
                              global_close_full: pd.Series):
    """
    Returns dict signal or None:
      - BUY: global ok_up(99.9%) AND local slope>0 AND recent lower-band reversal (within rev_horizon) AND rising
      - SELL: global ok_dn(99.9%) AND local slope<0 AND recent upper-band reversal (within rev_horizon) AND falling
    """
    p_view = _coerce_1d_series(close_view).dropna()
    if len(p_view) < max(30, int(local_lookback) + 5):
        return None

    yhat, upper, lower, m_local, _r2_local = regression_with_band(p_view, lookback=int(local_lookback), z=2.0)
    if yhat.empty or upper.empty or lower.empty or not np.isfinite(m_local):
        return None

    g = global_trend_stats(global_close_full, confidence=GLOBAL_TREND_CONFIDENCE)
    ok_up = bool(g.get("ok_up", False))
    ok_dn = bool(g.get("ok_dn", False))

    cb = max(1, int(confirm_bars))
    hz = max(1, int(rev_horizon_bars))

    buy = None
    if ok_up and float(m_local) > 0.0:
        buy = recent_band_reversal_signal(p_view, upper, lower, side="BUY", horizon_bars=hz, confirm_bars=cb)
        if buy:
            buy["global_slope"] = float(g.get("slope", np.nan))
            buy["global_r2"] = float(g.get("r2", np.nan))
            buy["global_p"] = float(g.get("p_value", np.nan))
            buy["local_slope"] = float(m_local)
            buy["label"] = "BUY (2σ, 99.9%)"
            return buy

    sell = None
    if ok_dn and float(m_local) < 0.0:
        sell = recent_band_reversal_signal(p_view, upper, lower, side="SELL", horizon_bars=hz, confirm_bars=cb)
        if sell:
            sell["global_slope"] = float(g.get("slope", np.nan))
            sell["global_r2"] = float(g.get("r2", np.nan))
            sell["global_p"] = float(g.get("p_value", np.nan))
            sell["local_slope"] = float(m_local)
            sell["label"] = "SELL (2σ, 99.9%)"
            return sell

    return None

def fib_signal_masks_for_price(close_view: pd.Series,
                              npx: pd.Series,
                              horizon_bars: int,
                              proximity_pct_of_range: float = 0.02):
    """
    Convenience wrapper for fib+NPX signals (uses rev_horizon bars for recency).
    """
    return fib_npx_zero_cross_signal_masks(
        price=_coerce_1d_series(close_view),
        npx=_coerce_1d_series(npx).reindex(_coerce_1d_series(close_view).index),
        horizon_bars=int(horizon_bars),
        proximity_pct_of_range=float(proximity_pct_of_range),
        npx_level=0.0,
    )


# =========================
# Part 6/10 — bullbear.py  (UPDATED: Daily & Hourly chart builders)
# =========================
def _plot_global_trendline(ax, price_full: pd.Series, price_view: pd.Series, label: str = "Global Trend"):
    """
    Draw global trendline over the *view* x-range, computed from full series.
    """
    full = _coerce_1d_series(price_full).dropna()
    view = _coerce_1d_series(price_view).dropna()
    if full.shape[0] < 3 or view.shape[0] < 3:
        return None

    m, b, r2, p = linear_regression_slope_pvalue(full)
    if not (np.isfinite(m) and np.isfinite(b)):
        return None

    n_view = len(view)
    x_view = np.arange(n_view, dtype=float)
    yhat_view = m * (np.linspace(len(full) - n_view, len(full) - 1, n_view, dtype=float)) + b
    col = "tab:green" if m > 0 else "tab:red"
    conf_ok = slope_is_confident(p, confidence=GLOBAL_TREND_CONFIDENCE)
    suffix = "✓" if conf_ok else "✗"
    ax.plot(view.index, yhat_view, linestyle="--", linewidth=2.4, color=col,
            label=f"{label} {suffix} (m={fmt_slope(m)}, p={p:.2g})")
    return {"slope": float(m), "r2": float(r2), "p": float(p), "ok": bool(conf_ok)}

def plot_daily_price_chart(ticker: str,
                           ohlc: pd.DataFrame,
                           close_full: pd.Series,
                           daily_view_label: str,
                           local_lookback: int,
                           rev_horizon_bars: int,
                           confirm_bars: int,
                           show_fibs_flag: bool,
                           show_bbands_flag: bool,
                           bb_window: int,
                           bb_mult_val: float,
                           bb_use_ema_flag: bool,
                           show_ichi_flag: bool,
                           ichi_base_period: int,
                           show_hma_flag: bool,
                           hma_period_val: int,
                           ntd_window_val: int):
    """
    Daily price chart with:
      - Global trendline w/ 99.9% confidence marker
      - Local regression + ±2σ band
      - Strict BUY/SELL markers (2σ reversal + global/local slope agreement + 99.9%)
      - Fibonacci BUY/SELL markers (Fib touch + NPX crosses 0.0 within rev_horizon)
    """
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty:
        return None

    close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_view.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close_view.index, close_view.values, label="Close", alpha=0.95)
    style_axes(ax)

    # Global trend
    _plot_global_trendline(ax, close_full, close_view, label="Global Trend")

    # Local regression + 2σ band (computed on view)
    yhat, upper, lower, m_local, r2_local = regression_with_band(close_view, lookback=int(local_lookback), z=2.0)
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="-", linewidth=2.0, alpha=0.90,
                label=f"Local Slope (m={fmt_slope(m_local)}, R²={fmt_r2(r2_local, digits=1)})")
        ax.plot(upper.index, upper.values, linestyle=":", alpha=0.9, label="+2σ")
        ax.plot(lower.index, lower.values, linestyle=":", alpha=0.9, label="-2σ")
        ax.fill_between(upper.index, lower.values, upper.values, alpha=0.06)

    # Bollinger (optional)
    if show_bbands_flag:
        mid, bu, bl = compute_bollinger(close_view, window=int(bb_window), mult=float(bb_mult_val), use_ema=bool(bb_use_ema_flag))
        if mid.notna().any():
            ax.plot(mid.index, mid.values, linewidth=1.4, alpha=0.9, label="BB Mid")
        if bu.notna().any() and bl.notna().any():
            ax.plot(bu.index, bu.values, linewidth=1.0, alpha=0.7, label="BB Upper")
            ax.plot(bl.index, bl.values, linewidth=1.0, alpha=0.7, label="BB Lower")

    # Ichimoku Kijun (optional)
    if show_ichi_flag and isinstance(ohlc, pd.DataFrame) and {"High", "Low"}.issubset(ohlc.columns):
        view_ohlc = subset_by_daily_view(ohlc, daily_view_label)
        kij = compute_kijun(view_ohlc["High"], view_ohlc["Low"], period=int(ichi_base_period))
        if kij.notna().any():
            ax.plot(kij.index, kij.values, linewidth=1.6, alpha=0.95, label="Kijun")

    # HMA (optional)
    if show_hma_flag:
        hma = compute_hma(close_view, period=int(hma_period_val))
        if hma.notna().any():
            ax.plot(hma.index, hma.values, linewidth=1.6, alpha=0.95, label=f"HMA({int(hma_period_val)})")

    # Strict 2σ reversal marker (latest qualifying)
    strict_sig = None
    if (not upper.empty) and (not lower.empty) and np.isfinite(m_local):
        strict_sig = strict_daily_2sigma_signal(
            close_view=close_view,
            local_lookback=int(local_lookback),
            rev_horizon_bars=int(rev_horizon_bars),
            confirm_bars=int(confirm_bars),
            global_close_full=close_full,
        )

    if strict_sig:
        t = strict_sig.get("signal_time")
        px = strict_sig.get("signal_price")
        side = str(strict_sig.get("side", "")).upper()
        lab = str(strict_sig.get("label", side))
        if t is not None and np.isfinite(px):
            if side == "BUY":
                ax.scatter([t], [px], marker="^", s=160, color="tab:green", zorder=20, label=lab)
            elif side == "SELL":
                ax.scatter([t], [px], marker="v", s=160, color="tab:red", zorder=20, label=lab)

    # Fibonacci + NPX signals (on price)
    if show_fibs_flag:
        npx = compute_normalized_price(close_view, window=int(ntd_window_val))
        buy_mask, sell_mask, _fibs = fib_signal_masks_for_price(
            close_view=close_view,
            npx=npx,
            horizon_bars=int(rev_horizon_bars),
            proximity_pct_of_range=0.02,
        )
        overlay_fib_npx_signals(ax, close_view, buy_mask, sell_mask)

    ax.set_title(f"{ticker} — Daily Price")
    ax.set_ylabel("Price")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig, {"strict": strict_sig}

def plot_hourly_price_chart(ticker: str,
                            intraday_df: pd.DataFrame,
                            local_lookback: int,
                            rev_horizon_bars: int,
                            confirm_bars: int,
                            show_fibs_flag: bool,
                            ntd_window_val: int):
    """
    Hourly/Intraday price chart with fib+NPX signals (optional).
    """
    if intraday_df is None or intraday_df.empty or "Close" not in intraday_df.columns:
        return None

    close = _coerce_1d_series(intraday_df["Close"]).dropna()
    if close.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(close)), close.values, label="Close", alpha=0.95)
    style_axes(ax)
    ax.set_title(f"{ticker} — Intraday (gapless)")
    ax.set_ylabel("Price")

    # Local regression + 2σ band
    yhat, upper, lower, m_local, r2_local = regression_with_band(close, lookback=int(local_lookback), z=2.0)
    if not yhat.empty:
        ax.plot(np.arange(len(yhat)), yhat.values, linewidth=2.0,
                label=f"Local Slope (m={fmt_slope(m_local)}, R²={fmt_r2(r2_local, digits=1)})")
        ax.plot(np.arange(len(upper)), upper.values, linestyle=":", alpha=0.9, label="+2σ")
        ax.plot(np.arange(len(lower)), lower.values, linestyle=":", alpha=0.9, label="-2σ")
        ax.fill_between(np.arange(len(upper)), lower.values, upper.values, alpha=0.06)

    # Fibonacci + NPX signals (optional)
    if show_fibs_flag:
        npx = compute_normalized_price(close, window=int(ntd_window_val))
        buy_mask, sell_mask, _fibs = fib_signal_masks_for_price(
            close_view=close,
            npx=npx,
            horizon_bars=int(rev_horizon_bars),
            proximity_pct_of_range=0.02,
        )
        # For intraday, overlay on positional x-axis
        bm = buy_mask.reindex(close.index, fill_value=False)
        sm = sell_mask.reindex(close.index, fill_value=False)
        buy_pos = np.where(bm.to_numpy(dtype=bool))[0].tolist()
        sell_pos = np.where(sm.to_numpy(dtype=bool))[0].tolist()
        if buy_pos:
            ax.scatter(buy_pos, close.iloc[buy_pos].values, marker="^", s=140, color="tab:green", zorder=20, label="Fibonacci BUY")
        if sell_pos:
            ax.scatter(sell_pos, close.iloc[sell_pos].values, marker="v", s=140, color="tab:red", zorder=20, label="Fibonacci SELL")

    _apply_compact_time_ticks(ax, close.index, n_ticks=8)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


# =========================
# Part 7/10 — bullbear.py  (UPDATED: Scanner helpers for new tabs)
# =========================
@st.cache_data(ttl=300)
def scan_daily_strict_signals(universe_list,
                              daily_view_label: str,
                              local_lookback: int,
                              rev_horizon_bars: int,
                              confirm_bars: int):
    """
    Produces a DataFrame for:
      - New tab: "Daily 99.9% Confidence Reversal"
      - Slope Direction Scan subsets (later)
    """
    rows = []
    for tkr in list(universe_list):
        try:
            close_full = fetch_hist(tkr)
        except Exception:
            continue
        close_full = _coerce_1d_series(close_full).dropna()
        if close_full.empty:
            continue

        close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_view.empty:
            continue

        sig = strict_daily_2sigma_signal(
            close_view=close_view,
            local_lookback=int(local_lookback),
            rev_horizon_bars=int(rev_horizon_bars),
            confirm_bars=int(confirm_bars),
            global_close_full=close_full,
        )
        if not sig:
            continue

        last_px = float(close_full.iloc[-1]) if np.isfinite(close_full.iloc[-1]) else np.nan
        rows.append({
            "ticker": tkr,
            "side": sig.get("side"),
            "last_price": last_px,
            "signal_time": sig.get("signal_time"),
            "signal_price": sig.get("signal_price"),
            "global_slope": sig.get("global_slope"),
            "global_r2": sig.get("global_r2"),
            "global_p": sig.get("global_p"),
            "local_slope": sig.get("local_slope"),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "ticker","side","last_price","signal_time","signal_price",
            "global_slope","global_r2","global_p","local_slope"
        ])

    df = pd.DataFrame(rows).sort_values(["side", "ticker"]).reset_index(drop=True)
    return df

def scan_daily_reversal_near_band(universe_list,
                                 daily_view_label: str,
                                 local_lookback: int,
                                 rev_horizon_bars: int,
                                 confirm_bars: int,
                                 near_pct: float):
    """
    For Slope Direction Scan tab:
      - subset tickers that recently reversed (BUY/SELL strict) within rev_horizon
      - and classify whether current price is near +2σ/-2σ
    """
    out = {"BUY_near_lower": [], "BUY_not_near": [], "SELL_near_upper": [], "SELL_not_near": []}

    for tkr in list(universe_list):
        try:
            close_full = fetch_hist(tkr)
        except Exception:
            continue
        close_full = _coerce_1d_series(close_full).dropna()
        if close_full.empty:
            continue

        close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
        if close_view.empty:
            continue

        yhat, upper, lower, m_local, _ = regression_with_band(close_view, lookback=int(local_lookback), z=2.0)
        if upper.empty or lower.empty:
            continue

        sig = strict_daily_2sigma_signal(
            close_view=close_view,
            local_lookback=int(local_lookback),
            rev_horizon_bars=int(rev_horizon_bars),
            confirm_bars=int(confirm_bars),
            global_close_full=close_full,
        )
        if not sig:
            continue

        last_px = float(close_view.iloc[-1]) if np.isfinite(close_view.iloc[-1]) else np.nan
        up = float(upper.iloc[-1]) if np.isfinite(upper.iloc[-1]) else np.nan
        lo = float(lower.iloc[-1]) if np.isfinite(lower.iloc[-1]) else np.nan
        if not (np.isfinite(last_px) and np.isfinite(up) and np.isfinite(lo)):
            continue

        side = str(sig.get("side", "")).upper()
        if side == "BUY":
            near = _near_pct(last_px, lo, near_pct)
            out["BUY_near_lower" if near else "BUY_not_near"].append(tkr)
        elif side == "SELL":
            near = _near_pct(last_px, up, near_pct)
            out["SELL_near_upper" if near else "SELL_not_near"].append(tkr)

    for k in out:
        out[k] = sorted(set(out[k]))
    return out
# =========================
# Part 8/10 — bullbear.py (UPDATED: fib horizon slider + plot overrides)
# =========================

# --- NEW: Fibonacci horizon slider (separate from rev_horizon) ---
fib_horizon_bars = st.sidebar.slider(
    "Fibonacci signal horizon (bars)",
    1, 120,
    int(min(30, max(1, rev_horizon))),
    1,
    key="sb_fib_horizon_bars",
)

# --- Override daily plot to use fib_horizon_bars (redefines prior function safely) ---
def plot_daily_price_chart(ticker: str,
                           ohlc: pd.DataFrame,
                           close_full: pd.Series,
                           daily_view_label: str,
                           local_lookback: int,
                           rev_horizon_bars: int,
                           confirm_bars: int,
                           show_fibs_flag: bool,
                           fib_horizon_bars: int,
                           show_bbands_flag: bool,
                           bb_window: int,
                           bb_mult_val: float,
                           bb_use_ema_flag: bool,
                           show_ichi_flag: bool,
                           ichi_base_period: int,
                           show_hma_flag: bool,
                           hma_period_val: int,
                           ntd_window_val: int):
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty:
        return None

    close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_view.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close_view.index, close_view.values, label="Close", alpha=0.95)
    style_axes(ax)

    _plot_global_trendline(ax, close_full, close_view, label="Global Trend")

    yhat, upper, lower, m_local, r2_local = regression_with_band(close_view, lookback=int(local_lookback), z=2.0)
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="-", linewidth=2.0, alpha=0.90,
                label=f"Local Slope (m={fmt_slope(m_local)}, R²={fmt_r2(r2_local, digits=1)})")
        ax.plot(upper.index, upper.values, linestyle=":", alpha=0.9, label="+2σ")
        ax.plot(lower.index, lower.values, linestyle=":", alpha=0.9, label="-2σ")
        ax.fill_between(upper.index, lower.values, upper.values, alpha=0.06)

    if show_bbands_flag:
        mid, bu, bl = compute_bollinger(close_view, window=int(bb_window), mult=float(bb_mult_val), use_ema=bool(bb_use_ema_flag))
        if mid.notna().any():
            ax.plot(mid.index, mid.values, linewidth=1.4, alpha=0.9, label="BB Mid")
        if bu.notna().any() and bl.notna().any():
            ax.plot(bu.index, bu.values, linewidth=1.0, alpha=0.7, label="BB Upper")
            ax.plot(bl.index, bl.values, linewidth=1.0, alpha=0.7, label="BB Lower")

    if show_ichi_flag and isinstance(ohlc, pd.DataFrame) and {"High", "Low"}.issubset(ohlc.columns):
        view_ohlc = subset_by_daily_view(ohlc, daily_view_label)
        kij = compute_kijun(view_ohlc["High"], view_ohlc["Low"], period=int(ichi_base_period))
        if kij.notna().any():
            ax.plot(kij.index, kij.values, linewidth=1.6, alpha=0.95, label="Kijun")

    if show_hma_flag:
        hma = compute_hma(close_view, period=int(hma_period_val))
        if hma.notna().any():
            ax.plot(hma.index, hma.values, linewidth=1.6, alpha=0.95, label=f"HMA({int(hma_period_val)})")

    strict_sig = None
    if (not upper.empty) and (not lower.empty) and np.isfinite(m_local):
        strict_sig = strict_daily_2sigma_signal(
            close_view=close_view,
            local_lookback=int(local_lookback),
            rev_horizon_bars=int(rev_horizon_bars),
            confirm_bars=int(confirm_bars),
            global_close_full=close_full,
        )

    if strict_sig:
        t = strict_sig.get("signal_time")
        px = strict_sig.get("signal_price")
        side = str(strict_sig.get("side", "")).upper()
        lab = str(strict_sig.get("label", side))
        if t is not None and np.isfinite(px):
            if side == "BUY":
                ax.scatter([t], [px], marker="^", s=160, color="tab:green", zorder=20, label=lab)
            elif side == "SELL":
                ax.scatter([t], [px], marker="v", s=160, color="tab:red", zorder=20, label=lab)

    if show_fibs_flag:
        npx = compute_normalized_price(close_view, window=int(ntd_window_val))
        buy_mask, sell_mask, _ = fib_signal_masks_for_price(
            close_view=close_view,
            npx=npx,
            horizon_bars=int(fib_horizon_bars),
            proximity_pct_of_range=0.02,
        )
        overlay_fib_npx_signals(ax, close_view, buy_mask, sell_mask)

    ax.set_title(f"{ticker} — Daily Price")
    ax.set_ylabel("Price")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig, {"strict": strict_sig}

# --- Override intraday plot to use fib_horizon_bars (redefines prior function safely) ---
def plot_hourly_price_chart(ticker: str,
                            intraday_df: pd.DataFrame,
                            local_lookback: int,
                            rev_horizon_bars: int,
                            confirm_bars: int,
                            show_fibs_flag: bool,
                            fib_horizon_bars: int,
                            ntd_window_val: int):
    if intraday_df is None or intraday_df.empty or "Close" not in intraday_df.columns:
        return None

    close = _coerce_1d_series(intraday_df["Close"]).dropna()
    if close.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.arange(len(close)), close.values, label="Close", alpha=0.95)
    style_axes(ax)
    ax.set_title(f"{ticker} — Intraday (gapless)")
    ax.set_ylabel("Price")

    yhat, upper, lower, m_local, r2_local = regression_with_band(close, lookback=int(local_lookback), z=2.0)
    if not yhat.empty:
        ax.plot(np.arange(len(yhat)), yhat.values, linewidth=2.0,
                label=f"Local Slope (m={fmt_slope(m_local)}, R²={fmt_r2(r2_local, digits=1)})")
        ax.plot(np.arange(len(upper)), upper.values, linestyle=":", alpha=0.9, label="+2σ")
        ax.plot(np.arange(len(lower)), lower.values, linestyle=":", alpha=0.9, label="-2σ")
        ax.fill_between(np.arange(len(upper)), lower.values, upper.values, alpha=0.06)

    if show_fibs_flag:
        npx = compute_normalized_price(close, window=int(ntd_window_val))
        buy_mask, sell_mask, _ = fib_signal_masks_for_price(
            close_view=close,
            npx=npx,
            horizon_bars=int(fib_horizon_bars),
            proximity_pct_of_range=0.02,
        )
        bm = buy_mask.reindex(close.index, fill_value=False)
        sm = sell_mask.reindex(close.index, fill_value=False)
        buy_pos = np.where(bm.to_numpy(dtype=bool))[0].tolist()
        sell_pos = np.where(sm.to_numpy(dtype=bool))[0].tolist()
        if buy_pos:
            ax.scatter(buy_pos, close.iloc[buy_pos].values, marker="^", s=140, color="tab:green", zorder=20, label="Fibonacci BUY")
        if sell_pos:
            ax.scatter(sell_pos, close.iloc[sell_pos].values, marker="v", s=140, color="tab:red", zorder=20, label="Fibonacci SELL")

    _apply_compact_time_ticks(ax, close.index, n_ticks=8)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return fig


# =========================
# Part 9/10 — bullbear.py (UPDATED: main tabs + new 99.9 tab + slope scan buckets)
# =========================
if "run_all" not in st.session_state:
    st.session_state.run_all = False

ticker = st.selectbox(
    "Select ticker:",
    universe,
    index=0,
    key=f"sb_ticker_{mode}",
)

run = st.button("▶ Run", use_container_width=True, key="btn_run_all")
if run:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode
    st.session_state.ticker = ticker
    st.session_state.df_hist = fetch_hist(ticker)
    st.session_state.df_ohlc = fetch_hist_ohlc(ticker)
    st.session_state.intraday = fetch_intraday(ticker, period="1d")
    try:
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(st.session_state.df_hist)
    except Exception:
        fc_idx, fc_vals, fc_ci = None, None, None
    st.session_state.fc_idx = fc_idx
    st.session_state.fc_vals = fc_vals
    st.session_state.fc_ci = fc_ci

tabs = st.tabs([
    "Overview",
    "Daily Price",
    "Intraday",
    "Forecast",
    "Daily 99.9% Confidence Reversal",
    "Slope Direction Scan",
])

with tabs[0]:
    st.subheader("Overview")
    if not st.session_state.run_all:
        st.info("Click **Run** to load data.")
    else:
        s = _coerce_1d_series(st.session_state.df_hist).dropna()
        last = float(s.iloc[-1]) if len(s) else np.nan
        r1 = float(s.pct_change(1).iloc[-1]) if len(s) > 2 else np.nan
        r5 = float(s.pct_change(5).iloc[-1]) if len(s) > 6 else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Last Close", fmt_price_val(last))
        c2.metric("1D %", fmt_pct(r1, 2))
        c3.metric("5D %", fmt_pct(r5, 2))

with tabs[1]:
    st.subheader("Daily Price")
    if not st.session_state.run_all:
        st.info("Click **Run** to load data.")
    else:
        fig_out = plot_daily_price_chart(
            ticker=st.session_state.ticker,
            ohlc=st.session_state.df_ohlc,
            close_full=st.session_state.df_hist,
            daily_view_label=daily_view,
            local_lookback=slope_lb_daily,
            rev_horizon_bars=rev_horizon,
            confirm_bars=rev_bars_confirm,
            show_fibs_flag=show_fibs,
            fib_horizon_bars=fib_horizon_bars,
            show_bbands_flag=show_bbands,
            bb_window=bb_win,
            bb_mult_val=bb_mult,
            bb_use_ema_flag=bb_use_ema,
            show_ichi_flag=show_ichi,
            ichi_base_period=ichi_base,
            show_hma_flag=show_hma,
            hma_period_val=hma_period,
            ntd_window_val=ntd_window,
        )
        if fig_out:
            fig, meta = fig_out
            st.pyplot(fig, use_container_width=True)
            sig = (meta or {}).get("strict")
            if sig:
                st.success(f"Strict signal: {sig.get('label')} @ {fmt_price_val(sig.get('signal_price'))} on {sig.get('signal_time')}")
            else:
                st.caption("Strict signal: none (conditions not met).")

with tabs[2]:
    st.subheader("Intraday")
    if not st.session_state.run_all:
        st.info("Click **Run** to load data.")
    else:
        fig = plot_hourly_price_chart(
            ticker=st.session_state.ticker,
            intraday_df=st.session_state.intraday,
            local_lookback=slope_lb_hourly,
            rev_horizon_bars=rev_horizon,
            confirm_bars=rev_bars_confirm,
            show_fibs_flag=show_fibs,
            fib_horizon_bars=fib_horizon_bars,
            ntd_window_val=ntd_window,
        )
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("No intraday data.")

with tabs[3]:
    st.subheader("Forecast (SARIMAX)")
    if not st.session_state.run_all:
        st.info("Click **Run** to load data.")
    else:
        fc_idx = st.session_state.get("fc_idx")
        fc_vals = st.session_state.get("fc_vals")
        fc_ci = st.session_state.get("fc_ci")
        hist = _coerce_1d_series(st.session_state.df_hist).dropna()

        if fc_idx is None or fc_vals is None or fc_ci is None or hist.empty:
            st.warning("Forecast unavailable.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(hist.index[-365:], hist.values[-365:], label="History (1y)")
            ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, label="Forecast")
            ci = fc_ci
            try:
                lo = _coerce_1d_series(ci.iloc[:, 0]).values
                hi = _coerce_1d_series(ci.iloc[:, 1]).values
                ax.fill_between(fc_idx, lo, hi, alpha=0.15)
            except Exception:
                pass
            style_axes(ax)
            ax.set_title(f"{st.session_state.ticker} — 30D Forecast")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Daily 99.9% Confidence Reversal")
    st.caption("Strict signals: reversed from ±2σ within last rev_horizon bars + local/global slope agreement + global slope 99.9% confidence.")
    run_scan = st.button("🔎 Run Daily 99.9 Scan", use_container_width=True, key="btn_scan_daily_999")

    if run_scan:
        st.session_state.daily_999_scan = scan_daily_strict_signals(
            universe_list=universe,
            daily_view_label=daily_view,
            local_lookback=slope_lb_daily,
            rev_horizon_bars=rev_horizon,
            confirm_bars=rev_bars_confirm,
        )

    df_scan = st.session_state.get("daily_999_scan")
    if df_scan is None or df_scan.empty:
        st.info("Click **Run Daily 99.9 Scan** to populate results.")
    else:
        df = df_scan.copy()
        buy_df = df[df["side"].astype(str).str.upper() == "BUY"].copy()
        sell_df = df[df["side"].astype(str).str.upper() == "SELL"].copy()

        st.markdown("#### BUY")
        st.dataframe(buy_df.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("#### SELL")
        st.dataframe(sell_df.reset_index(drop=True), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Slope Direction Scan")
    st.caption("Strict reversal tickers (within rev_horizon bars) bucketed by proximity to ±2σ.")

    run_buckets = st.button("📌 Refresh Reversal Buckets", use_container_width=True, key="btn_scan_buckets")
    if run_buckets:
        st.session_state.rev_buckets = scan_daily_reversal_near_band(
            universe_list=universe,
            daily_view_label=daily_view,
            local_lookback=slope_lb_daily,
            rev_horizon_bars=rev_horizon,
            confirm_bars=rev_bars_confirm,
            near_pct=float(sr_prox_pct),
        )

    buckets = st.session_state.get("rev_buckets")
    if not buckets:
        st.info("Click **Refresh Reversal Buckets** to populate.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### BUY reversed")
            st.markdown(f"**Near -2σ** ({len(buckets.get('BUY_near_lower', []))}):")
            st.write(", ".join(buckets.get("BUY_near_lower", [])) or "—")
            st.markdown(f"**Not near -2σ** ({len(buckets.get('BUY_not_near', []))}):")
            st.write(", ".join(buckets.get("BUY_not_near", [])) or "—")

        with c2:
            st.markdown("#### SELL reversed")
            st.markdown(f"**Near +2σ** ({len(buckets.get('SELL_near_upper', []))}):")
            st.write(", ".join(buckets.get("SELL_near_upper", [])) or "—")
            st.markdown(f"**Not near +2σ** ({len(buckets.get('SELL_not_near', []))}):")
            st.write(", ".join(buckets.get("SELL_not_near", [])) or "—")


# =========================
# Part 10/10 — tests/test_strict_2sigma_harness.py (NEW: pure pandas pytest harness)
# =========================
# File: tests/test_strict_2sigma_harness.py
import numpy as np
import pandas as pd

GLOBAL_TREND_CONFIDENCE = 0.999

def _coerce_1d_series(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        s = obj[num_cols[0]] if num_cols else pd.Series(dtype=float)
    else:
        s = pd.Series(obj)
    return pd.to_numeric(s, errors="coerce")

def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
    if len(s) < 3:
        e = pd.Series(index=s.index, dtype=float)
        return e, e, e, float("nan"), float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    std = float(np.sqrt(np.sum(resid ** 2) / dof))
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return yhat_s, upper_s, lower_s, float(m), float(r2)

def recent_band_reversal_signal(price: pd.Series,
                                upper_band: pd.Series,
                                lower_band: pd.Series,
                                side: str,
                                horizon_bars: int,
                                confirm_bars: int = 2):
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None

    p = p[ok]; u = u[ok]; l = l[ok]
    hz = max(1, int(horizon_bars))
    cb = max(1, int(confirm_bars))

    def _last_n_deltas(seg: pd.Series, n: int):
        s = _coerce_1d_series(seg).dropna()
        if len(s) < n + 1:
            return None
        return np.diff(s.iloc[-(n + 1):].to_numpy(dtype=float))

    tail_idx = p.index[-(hz + 1):] if len(p) > (hz + 1) else p.index

    side_u = str(side).upper().strip()
    if side_u.startswith("B"):
        touch = (p <= l)
        cross_back = (p >= l) & (p.shift(1) < l.shift(1))

        touch_t = touch.loc[tail_idx]
        cross_t = cross_back.loc[tail_idx]
        if not (touch_t.any() and cross_t.any()):
            return None

        t_cross = cross_t[cross_t].index[-1]
        touch_before = touch.loc[:t_cross]
        if not touch_before.any():
            return None
        t_touch = touch_before[touch_before].index[-1]

        deltas = _last_n_deltas(p, cb)
        if deltas is None or not bool(np.all(deltas > 0)):
            return None

        px_cross = float(p.loc[t_cross])
        px_now = float(p.iloc[-1])
        if not (np.isfinite(px_cross) and np.isfinite(px_now) and px_now >= px_cross):
            return None

        return {"side": "BUY", "touch_time": t_touch, "signal_time": t_cross, "signal_price": px_cross}

    if side_u.startswith("S"):
        touch = (p >= u)
        cross_back = (p <= u) & (p.shift(1) > u.shift(1))

        touch_t = touch.loc[tail_idx]
        cross_t = cross_back.loc[tail_idx]
        if not (touch_t.any() and cross_t.any()):
            return None

        t_cross = cross_t[cross_t].index[-1]
        touch_before = touch.loc[:t_cross]
        if not touch_before.any():
            return None
        t_touch = touch_before[touch_before].index[-1]

        deltas = _last_n_deltas(p, cb)
        if deltas is None or not bool(np.all(deltas < 0)):
            return None

        px_cross = float(p.loc[t_cross])
        px_now = float(p.iloc[-1])
        if not (np.isfinite(px_cross) and np.isfinite(px_now) and px_now <= px_cross):
            return None

        return {"side": "SELL", "touch_time": t_touch, "signal_time": t_cross, "signal_price": px_cross}

    return None

def test_recent_band_reversal_buy_basic():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    p = pd.Series([10, 9, 8, 7, 8, 9], index=idx, dtype=float)
    lower = pd.Series([7.5] * 6, index=idx, dtype=float)
    upper = pd.Series([12.0] * 6, index=idx, dtype=float)

    sig = recent_band_reversal_signal(p, upper, lower, side="BUY", horizon_bars=5, confirm_bars=2)
    assert sig is not None
    assert sig["side"] == "BUY"
    assert sig["signal_time"] == idx[4]

def test_recent_band_reversal_sell_basic():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    p = pd.Series([10, 11, 12.5, 13, 12.5, 12.0], index=idx, dtype=float)
    upper = pd.Series([12.6] * 6, index=idx, dtype=float)
    lower = pd.Series([8.0] * 6, index=idx, dtype=float)

    sig = recent_band_reversal_signal(p, upper, lower, side="SELL", horizon_bars=5, confirm_bars=2)
    assert sig is not None
    assert sig["side"] == "SELL"
    assert sig["signal_time"] == idx[5]

def test_regression_with_band_shapes():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    x = np.arange(120, dtype=float)
    y = 100 + 0.2 * x + 0.01 * np.sin(x)
    s = pd.Series(y, index=idx)
    yhat, up, lo, m, r2 = regression_with_band(s, lookback=90, z=2.0)
    assert len(yhat) == len(up) == len(lo) == 90
    assert np.isfinite(m)
    assert np.isfinite(r2)


# =========================
# How to run tests (doc):
#   pytest -q
# =========================
