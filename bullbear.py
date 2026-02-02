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

# =========================
# NEW (Feb 2026 request): Daily regression guidance under Forecast button
# ---------------------------
def daily_regression_guidance_message(close_show: pd.Series,
                                      slope_lb: int,
                                      z: float = 2.0):
    """
    Implements requested rules (Tab 1 under Forecast Button):
      If regression slope > 0 AND
        - reversed from lower 2σ and heading toward upper 2σ -> "Daily Slope is positive, which favors a Buy position."
        - reversed from upper 2σ and heading toward lower 2σ -> caution message
    Returns string or None.
    """
    c = _coerce_1d_series(close_show).dropna()
    if c.empty or len(c) < max(6, int(slope_lb) + 2):
        return None

    yhat, up, lo, m, r2 = regression_with_band(c, lookback=int(slope_lb), z=float(z))
    if not (np.isfinite(m) and float(m) > 0) or up.dropna().empty or lo.dropna().empty:
        return None

    if len(c) < 3 or len(up) < 3 or len(lo) < 3:
        return None

    c0, c1 = float(c.iloc[-1]), float(c.iloc[-2])
    up0, up1 = float(up.iloc[-1]), float(up.iloc[-2])
    lo0, lo1 = float(lo.iloc[-1]), float(lo.iloc[-2])
    if not np.all(np.isfinite([c0, c1, up0, up1, lo0, lo1])):
        return None

    reversed_from_lower = (c1 <= lo1) and (c0 > lo0)
    reversed_from_upper = (c1 >= up1) and (c0 < up0)

    heading_toward_upper = (c0 > c1) and ((up0 - c0) < (up1 - c1))
    heading_toward_lower = (c0 < c1) and ((c0 - lo0) < (c1 - lo1))

    if reversed_from_lower and heading_toward_upper:
        return "Daily Slope is positive, which favors a Buy position."

    if reversed_from_upper and heading_toward_lower:
        return ("Daily Slope is positive, which favors a Buy position but be cautious when placing a Buy position "
                "because the price has reached the reversal points.")

    return None


# =========================
# Part 4/10 — bullbear.py
# =========================
# ---------------------------
# Other indicators (HMA, BBands, Ichimoku, etc.)
# ---------------------------
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

# =========================
# NEW (Feb 2026 request): Scanner helpers for
#   (A) Regression>0 + recent Kijun cross up/down
#   (B) Recent HMA(55) cross up/down
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_kijun_cross(symbol: str,
                           daily_view_label: str,
                           slope_lb: int,
                           conv: int,
                           base: int,
                           span_b: int,
                           within_last_n_bars: int = 5,
                           direction: str = "up",
                           require_positive_regression: bool = True):
    """
    Returns dict for last Kijun cross in selected daily view.
    direction:
      - "up": price crosses ABOVE kijun
      - "down": price crosses BELOW kijun
    Filter:
      - within_last_n_bars
      - optionally require regression slope > 0 (on the same view)
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(3, int(base) + 2, int(slope_lb)):
            return None

        ohlc = fetch_hist_ohlc(symbol)
        if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
            return None

        x0, x1 = close_show.index[0], close_show.index[-1]
        ohlc_show = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)]
        if ohlc_show.empty or len(ohlc_show) < max(3, int(base) + 2):
            return None

        _, kijun, _, _, _ = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=int(conv), base=int(base), span_b=int(span_b),
            shift_cloud=False
        )
        kijun = _coerce_1d_series(kijun).reindex(close_show.index).ffill().bfill()
        if kijun.dropna().empty:
            return None

        cross_up, cross_dn = _cross_series(close_show, kijun)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)
        cross_dn = cross_dn.reindex(close_show.index, fill_value=False)

        want_up = str(direction).lower().startswith("u")
        cross_mask = cross_up if want_up else cross_dn
        if not cross_mask.any():
            return None

        t_cross = cross_mask[cross_mask].index[-1]
        try:
            loc = int(close_show.index.get_loc(t_cross))
        except Exception:
            return None
        bars_since = int((len(close_show) - 1) - loc)

        within_last_n_bars = max(0, int(within_last_n_bars))
        if bars_since > within_last_n_bars:
            return None

        # Regression over same daily-view subset (global)
        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None
        if require_positive_regression and not (float(m) > 0.0):
            return None

        px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        kij_cross = float(kijun.loc[t_cross]) if (t_cross in kijun.index and np.isfinite(kijun.loc[t_cross])) else np.nan
        px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        # direction confirmation (simple)
        px_prev = float(close_show.shift(1).loc[t_cross]) if (t_cross in close_show.index and np.isfinite(close_show.shift(1).loc[t_cross])) else np.nan
        if want_up:
            heading_ok = (np.isfinite(px_cross) and np.isfinite(px_prev) and (px_cross > px_prev) and np.isfinite(px_last) and (px_last >= px_cross))
            sig_label = "Kijun Cross ↑"
        else:
            heading_ok = (np.isfinite(px_cross) and np.isfinite(px_prev) and (px_cross < px_prev) and np.isfinite(px_last) and (px_last <= px_cross))
            sig_label = "Kijun Cross ↓"
        if not heading_ok:
            return None

        return {
            "Symbol": symbol,
            "Signal": sig_label,
            "Bars Since Cross": int(bars_since),
            "Cross Time": t_cross,
            "Price@Cross": float(px_cross) if np.isfinite(px_cross) else np.nan,
            "Kijun@Cross": float(kij_cross) if np.isfinite(kij_cross) else np.nan,
            "Current Price": float(px_last) if np.isfinite(px_last) else np.nan,
            "Slope": float(m),
            "R2": float(r2),
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_daily_hma55_cross(symbol: str,
                           daily_view_label: str,
                           hma_len: int = 55,
                           within_last_n_bars: int = 5):
    """
    NEW tab (Feb 2026 request):
      (a) Price recently crossed HMA(55) going up
      (b) Price recently crossed HMA(55) going down
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(10, int(hma_len) + 2):
            return None

        hma = compute_hma(close_show, period=int(hma_len)).reindex(close_show.index)
        if hma.dropna().empty:
            return None

        cross_up, cross_dn = _cross_series(close_show, hma)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)
        cross_dn = cross_dn.reindex(close_show.index, fill_value=False)

        if (not cross_up.any()) and (not cross_dn.any()):
            return None

        last_up = cross_up[cross_up].index[-1] if cross_up.any() else None
        last_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None

        # choose most recent
        if last_up is None:
            t_cross, direction = last_dn, "down"
        elif last_dn is None:
            t_cross, direction = last_up, "up"
        else:
            t_cross, direction = (last_up, "up") if last_up >= last_dn else (last_dn, "down")

        if t_cross is None:
            return None

        try:
            loc = int(close_show.index.get_loc(t_cross))
        except Exception:
            return None
        bars_since = int((len(close_show) - 1) - loc)

        within_last_n_bars = max(0, int(within_last_n_bars))
        if bars_since > within_last_n_bars:
            return None

        px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        hma_cross = float(hma.loc[t_cross]) if (t_cross in hma.index and np.isfinite(hma.loc[t_cross])) else np.nan
        px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        px_prev = float(close_show.shift(1).loc[t_cross]) if (t_cross in close_show.index and np.isfinite(close_show.shift(1).loc[t_cross])) else np.nan
        if direction == "up":
            heading_ok = (np.isfinite(px_cross) and np.isfinite(px_prev) and (px_cross > px_prev) and np.isfinite(px_last) and (px_last >= px_cross))
            sig = "HMA55 Cross ↑"
        else:
            heading_ok = (np.isfinite(px_cross) and np.isfinite(px_prev) and (px_cross < px_prev) and np.isfinite(px_last) and (px_last <= px_cross))
            sig = "HMA55 Cross ↓"
        if not heading_ok:
            return None

        return {
            "Symbol": symbol,
            "Signal": sig,
            "Bars Since Cross": int(bars_since),
            "Cross Time": t_cross,
            "Price@Cross": float(px_cross) if np.isfinite(px_cross) else np.nan,
            "HMA@Cross": float(hma_cross) if np.isfinite(hma_cross) else np.nan,
            "Current Price": float(px_last) if np.isfinite(px_last) else np.nan,
        }
    except Exception:
        return None


# =========================
# Part 5/10 — bullbear.py  (Batch 1 continues up to Tab 4)
# =========================
# ---------------------------
# Session state init
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

# ---------------------------
# Tabs (KEEP original 18; append 2 new tabs at the end — no changes to original ordering)
# ---------------------------
TAB_LABELS = [
    "Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "NPX 0.5-Cross Scanner",
    "Fib NPX 0.0 Signal Scanner",
    "Slope Direction Scan",
    "Trendline Direction Lists",
    "NTD Hot List",
    "NTD NPX 0.0-0.2 Scanner",
    "Uptrend vs Downtrend",
    "Ichimoku Kijun Scanner",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)",
    # -------------------------
    # NEW (Feb 2026 request) — appended (does not alter existing tabs)
    # -------------------------
    "Regression + Kijun Cross (NEW)",
    "HMA55 Cross Scanner (NEW)",
]
tabs = st.tabs(TAB_LABELS)
(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8,
 tab9, tab10, tab11, tab12,
 tab13, tab14, tab15, tab16,
 tab17, tab18, tab19, tab20) = tabs

# ---------------------------
# Symbol dropdowns (keep present + stable)
# ---------------------------
def _symbol_picker():
    if mode == "Forex":
        sel = st.selectbox("Forex Symbol", universe, index=0, key="dd_forex_symbol")
    else:
        sel = st.selectbox("Stock Symbol", universe, index=0, key="dd_stock_symbol")
    return sel

# ---------------------------
# TAB 1: Forecast
# ---------------------------
with tab1:
    st.header("Forecast")

    sel = _symbol_picker()

    # Hour range selection (kept simple; original layout preserved conceptually)
    hr_col1, hr_col2, hr_col3 = st.columns([1, 1, 2])
    with hr_col1:
        hour_range_label = st.selectbox("Hourly range", ["24h", "5d", "1mo"], index=0, key="dd_hour_range")
    with hr_col2:
        run_btn = st.button("Forecast", use_container_width=True, key=f"btn_forecast_{mode}")
    with hr_col3:
        st.caption("Uses SARIMAX for 30D forecast + regression bands + hourly view.")

    if run_btn:
        st.session_state.run_all = True
        st.session_state.ticker = sel
        st.session_state.mode_at_run = mode
        st.session_state.hour_range = hour_range_label

        # Cache data for other tabs
        try:
            df_close = fetch_hist(sel)
            df_ohlc = fetch_hist_ohlc(sel)
            st.session_state.df_hist = df_close
            st.session_state.df_ohlc = df_ohlc
        except Exception:
            st.session_state.df_hist = None
            st.session_state.df_ohlc = None

        try:
            fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(st.session_state.df_hist)
            st.session_state.fc_idx = fc_idx
            st.session_state.fc_vals = fc_vals
            st.session_state.fc_ci = fc_ci
        except Exception:
            st.session_state.fc_idx = None
            st.session_state.fc_vals = None
            st.session_state.fc_ci = None

        try:
            period_map = {"24h": "1d", "5d": "5d", "1mo": "1mo"}
            intr = fetch_intraday(sel, period=period_map.get(hour_range_label, "1d"))
            st.session_state.intraday = intr
        except Exception:
            st.session_state.intraday = None

    if not st.session_state.run_all or st.session_state.ticker != sel:
        st.info("Click **Forecast** to run charts/forecast.")
    else:
        close_full = _coerce_1d_series(st.session_state.df_hist).dropna() if st.session_state.df_hist is not None else pd.Series(dtype=float)
        if close_full.empty:
            st.warning("No daily data available.")
        else:
            close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
            if close_show.empty:
                close_show = close_full

            # --- Requested (2): statement under Forecast Button reflecting current daily regression ---
            msg = daily_regression_guidance_message(close_show, slope_lb=slope_lb_daily, z=2.0)
            if msg:
                st.success(msg)

            # ---- Forecast chart ----
            fc_idx = st.session_state.fc_idx
            fc_vals = st.session_state.fc_vals
            fc_ci = st.session_state.fc_ci
            last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

            if fc_idx is not None and fc_vals is not None and len(fc_vals) > 0:
                fmean = _coerce_1d_series(fc_vals)
                # Simple up/down probability proxy
                try:
                    p_up = float((fmean > last_px).mean()) if np.isfinite(last_px) else np.nan
                    p_dn = 1.0 - p_up if np.isfinite(p_up) else np.nan
                except Exception:
                    p_up = np.nan
                    p_dn = np.nan

                fig_fc, ax_fc = plt.subplots(figsize=(14, 4))
                ax_fc.plot(close_show.index, close_show.values, label="Close")
                ax_fc.plot(fc_idx, fmean.values, label="SARIMAX Forecast")
                try:
                    ci = fc_ci
                    if ci is not None and hasattr(ci, "iloc"):
                        lo = ci.iloc[:, 0].values
                        hi = ci.iloc[:, 1].values
                        ax_fc.fill_between(fc_idx, lo, hi, alpha=0.12, label="Conf. Int.")
                except Exception:
                    pass
                ax_fc.set_title(f"{sel} — 30D Forecast  |  P↑={fmt_pct(p_up)}  P↓={fmt_pct(p_dn)}")
                style_axes(ax_fc)
                ax_fc.legend(loc="upper left")
                st.pyplot(fig_fc, use_container_width=True)
            else:
                st.warning("Forecast not available (SARIMAX).")

            # ---- Daily price chart with regression bands + selected overlays ----
            fig_d, ax_d = plt.subplots(figsize=(14, 5))
            ax_d.plot(close_show.index, close_show.values, label="Close")

            # Regression bands
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)
            if not yhat_d.dropna().empty:
                ax_d.plot(yhat_d.index, yhat_d.values, "-", linewidth=2.0, label=f"Slope ({fmt_slope(m_d)}/bar)")
            if not up_d.dropna().empty and not lo_d.dropna().empty:
                ax_d.plot(up_d.index, up_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
                ax_d.plot(lo_d.index, lo_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

            # HMA
            if show_hma:
                hma_d = compute_hma(close_show, period=int(hma_period))
                if not hma_d.dropna().empty:
                    ax_d.plot(hma_d.index, hma_d.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            # Ichimoku Kijun on daily
            if show_ichi:
                ohlc = st.session_state.df_ohlc
                if ohlc is not None and isinstance(ohlc, pd.DataFrame) and not ohlc.empty:
                    x0, x1 = close_show.index[0], close_show.index[-1]
                    ohlc_show = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)]
                    if not ohlc_show.empty:
                        _, kijun_d, _, _, _ = ichimoku_lines(
                            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
                            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
                            shift_cloud=False
                        )
                        kijun_d = _coerce_1d_series(kijun_d).reindex(close_show.index).ffill().bfill()
                        if not kijun_d.dropna().empty:
                            ax_d.plot(kijun_d.index, kijun_d.values, "-", linewidth=1.8, color="black",
                                      label=f"Ichimoku Kijun ({ichi_base})")

            # Bollinger
            if show_bbands:
                bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close_show, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
                if not bb_up.dropna().empty and not bb_lo.dropna().empty:
                    ax_d.fill_between(close_show.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                    ax_d.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'})")
                    ax_d.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
                    ax_d.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

            # Fibonacci (optional)
            if show_fibs:
                fibs = fibonacci_levels(close_show)
                for lbl, y in fibs.items():
                    ax_d.hlines(y, xmin=close_show.index[0], xmax=close_show.index[-1],
                                linestyles="dotted", linewidth=1.0)
                for lbl, y in fibs.items():
                    ax_d.text(close_show.index[-1], y, f" {lbl}", va="center")

            ax_d.set_title(f"{sel} Daily  |  R²={fmt_r2(r2_d)}")
            style_axes(ax_d)
            ax_d.legend(loc="upper left")
            st.pyplot(fig_d, use_container_width=True)

        # ---- Hourly view (basic renderer for Tab1) ----
        intr = st.session_state.intraday
        if intr is not None and isinstance(intr, pd.DataFrame) and not intr.empty:
            # Re-use probabilities from forecast plot if available
            try:
                fmean = _coerce_1d_series(st.session_state.fc_vals)
                last_px = float(close_full.iloc[-1]) if close_full is not None and len(close_full) else np.nan
                p_up = float((fmean > last_px).mean()) if np.isfinite(last_px) and len(fmean) else np.nan
                p_dn = 1.0 - p_up if np.isfinite(p_up) else np.nan
            except Exception:
                p_up = np.nan
                p_dn = np.nan

            # Simple hourly chart
            real_times = intr.index if isinstance(intr.index, pd.DatetimeIndex) else None
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))

            hc = _coerce_1d_series(intr_plot["Close"]).ffill()
            fig_h, ax_h = plt.subplots(figsize=(14, 4))
            ax_h.plot(hc.index, hc.values, label="Intraday Close")

            # Regression bands hourly
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, lookback=int(slope_lb_hourly), z=2.0)
            if not yhat_h.dropna().empty:
                ax_h.plot(yhat_h.index, yhat_h.values, "-", linewidth=2.0, label=f"Slope ({fmt_slope(m_h)}/bar)")
            if not up_h.dropna().empty and not lo_h.dropna().empty:
                ax_h.plot(up_h.index, up_h.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
                ax_h.plot(lo_h.index, lo_h.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

            ax_h.set_title(f"{sel} Intraday ({st.session_state.hour_range})  |  P↑={fmt_pct(p_up)}  P↓={fmt_pct(p_dn)}  |  R²={fmt_r2(r2_h)}")
            style_axes(ax_h)
            ax_h.legend(loc="upper left")

            if isinstance(real_times, pd.DatetimeIndex):
                _apply_compact_time_ticks(ax_h, real_times)

            st.pyplot(fig_h, use_container_width=True)
        else:
            st.warning("No intraday data available for hourly view.")

# ---------------------------
# TAB 2: Enhanced Forecast
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    sel2 = _symbol_picker()

    if not st.session_state.run_all or st.session_state.ticker != sel2:
        st.info("Run **Forecast** in Tab 1 to populate cached data, then return here.")
    else:
        close_full = _coerce_1d_series(st.session_state.df_hist).dropna() if st.session_state.df_hist is not None else pd.Series(dtype=float)
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()

        fig_e, ax_e = plt.subplots(figsize=(14, 5))
        ax_e.plot(close_show.index, close_show.values, label="Close")

        # HMA + BB + Kijun overlays (matches your sidebar toggles)
        if show_hma:
            hma_d = compute_hma(close_show, period=int(hma_period))
            if not hma_d.dropna().empty:
                ax_e.plot(hma_d.index, hma_d.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

        if show_bbands:
            bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close_show, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
            if not bb_up.dropna().empty and not bb_lo.dropna().empty:
                ax_e.fill_between(close_show.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax_e.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'})")

        if show_ichi:
            ohlc = st.session_state.df_ohlc
            if ohlc is not None and isinstance(ohlc, pd.DataFrame) and not ohlc.empty:
                x0, x1 = close_show.index[0], close_show.index[-1]
                ohlc_show = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)]
                if not ohlc_show.empty:
                    _, kijun_d, _, _, _ = ichimoku_lines(
                        ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
                        conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
                        shift_cloud=False
                    )
                    kijun_d = _coerce_1d_series(kijun_d).reindex(close_show.index).ffill().bfill()
                    if not kijun_d.dropna().empty:
                        ax_e.plot(kijun_d.index, kijun_d.values, "-", linewidth=1.8, color="black",
                                  label=f"Ichimoku Kijun ({ichi_base})")

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)
        if not yhat.dropna().empty:
            ax_e.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Regression ({fmt_slope(m)}/bar)")
        if not up.dropna().empty and not lo.dropna().empty:
            ax_e.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
            ax_e.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")

        ax_e.set_title(f"{sel2} Enhanced Daily  |  R²={fmt_r2(r2)}")
        style_axes(ax_e)
        ax_e.legend(loc="upper left")
        st.pyplot(fig_e, use_container_width=True)

# ---------------------------
# TAB 3: Bull vs Bear
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    sel3 = _symbol_picker()

    try:
        s = _coerce_1d_series(fetch_hist(sel3)).dropna()
        if s.empty:
            st.warning("No data.")
        else:
            per = str(bb_period)
            # quick period slice using pandas offsets
            if per == "1mo":
                s2 = s.iloc[-22:] if len(s) > 22 else s
            elif per == "3mo":
                s2 = s.iloc[-66:] if len(s) > 66 else s
            elif per == "6mo":
                s2 = s.iloc[-132:] if len(s) > 132 else s
            else:
                s2 = s.iloc[-252:] if len(s) > 252 else s

            ret = (float(s2.iloc[-1]) / float(s2.iloc[0]) - 1.0) if len(s2) >= 2 else np.nan
            up_days = float((s2.diff() > 0).sum())
            dn_days = float((s2.diff() < 0).sum())
            tot = max(1.0, up_days + dn_days)
            bull = up_days / tot
            bear = dn_days / tot

            c1, c2, c3 = st.columns(3)
            c1.metric("Return", fmt_pct(ret))
            c2.metric("Bull days %", fmt_pct(bull))
            c3.metric("Bear days %", fmt_pct(bear))

            fig_bb, ax_bb = plt.subplots(figsize=(14, 4))
            ax_bb.plot(s2.index, s2.values, label="Close")
            ax_bb.set_title(f"{sel3} — {bb_period} Bull/Bear")
            style_axes(ax_bb)
            ax_bb.legend(loc="upper left")
            st.pyplot(fig_bb, use_container_width=True)
    except Exception:
        st.error("Bull/Bear computation failed.")

# ---------------------------
# TAB 4: Metrics
# ---------------------------
with tab4:
    st.header("Metrics")
    sel4 = _symbol_picker()

    try:
        close_full = _coerce_1d_series(fetch_hist(sel4)).dropna()
        ohlc = fetch_hist_ohlc(sel4)

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)

        piv = current_daily_pivots(ohlc)

        rows = []
        rows.append(["Mode", mode])
        rows.append(["Daily View", daily_view])
        rows.append(["Daily Regression Slope", fmt_slope(m)])
        rows.append(["Daily Regression R²", fmt_r2(r2)])
        if piv:
            rows.append(["Pivot P", fmt_price_val(piv.get("P", np.nan))])
            rows.append(["Pivot R1", fmt_price_val(piv.get("R1", np.nan))])
            rows.append(["Pivot S1", fmt_price_val(piv.get("S1", np.nan))])

        dfm = pd.DataFrame(rows, columns=["Metric", "Value"])
        st.dataframe(dfm, use_container_width=True)

    except Exception:
        st.error("Metrics tab failed to compute.")

# =========================
# END OF BATCH 1 (Tabs 1–4)
# Batch 2 will continue with Tabs 5–8.
# =========================
# =========================
# Batch 2/3 — Tabs 5 through 8 (NO UI changes to existing tabs; unchanged content style)
# =========================

# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner (Hourly)")
    st.caption(
        "Hourly scan of **Normalized Trend Direction (NTD)**.\n"
        "Lists symbols where the latest Hourly NTD is **≤ -0.75** (deep negative zone)."
    )

    run_ntd075 = st.button("Run NTD -0.75 Scan (Hourly)", key=f"btn_run_ntd075_{mode}")

    if run_ntd075:
        rows = []
        prog = st.progress(0)
        total = max(1, len(universe))

        # Use default hourly period unless your app stores a specific selection elsewhere
        scan_period = "1d"

        for i, sym in enumerate(universe):
            v, ts = last_hourly_ntd_value(sym, ntd_window, period=scan_period)
            if np.isfinite(v) and float(v) <= -0.75:
                rows.append({
                    "Symbol": sym,
                    "NTD (last)": float(v),
                    "AsOf": ts
                })
            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        if not rows:
            st.info("No symbols currently meet NTD ≤ -0.75 on the Hourly scan.")
        else:
            df = pd.DataFrame(rows).sort_values("NTD (last)", ascending=True).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)


# ---------------------------
# TAB 6: Long-Term History
# ---------------------------
with tab6:
    st.header("Long-Term History")
    st.caption(
        "Long-term price history view using maximum available data from Yahoo Finance.\n"
        "This tab uses the **selected symbol** from the main UI (Forecast run state)."
    )

    # keep existing behavior: rely on the run selection
    sel = st.session_state.get("ticker", None)

    # allow user to choose long-term window without changing the global UI
    years = st.slider(
        "Years to display (max history subset)",
        min_value=1, max_value=25,
        value=int(st.session_state.get("hist_years", 10)),
        step=1,
        key=f"lt_years_{mode}"
    )
    st.session_state.hist_years = int(years)

    if not sel:
        st.info("Run **Forecast** (or select a symbol) to populate the long-term history view.")
    else:
        smax = _coerce_1d_series(fetch_hist_max(sel)).dropna()
        if smax.empty:
            st.warning("No max-history data available.")
        else:
            end = smax.index.max()
            start = end - pd.Timedelta(days=int(years) * 365)
            s = smax.loc[(smax.index >= start) & (smax.index <= end)].dropna()

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(s.index, s.values, label="Close")
            _ = draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
            ax.set_title(f"{sel} — Long-Term History (last {years} years)")
            style_axes(ax)
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)


# ---------------------------
# TAB 7: Recent BUY Scanner
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner")
    st.caption(
        "Finds symbols with a **recent slope-band bounce BUY** signal.\n"
        "Uses the same logic as the charts: regression ±2σ band bounce detection."
    )

    max_bars = st.slider(
        "Max bars since BUY signal (Daily)",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"recent_buy_max_bars_daily_{mode}"
    )

    include_hourly = st.checkbox(
        "Also include Hourly BUY signals (uses current Hourly settings)",
        value=False,
        key=f"recent_buy_include_hourly_{mode}"
    )

    run_recent_buy = st.button("Run Recent BUY Scan", key=f"btn_run_recent_buy_{mode}")

    if run_recent_buy:
        buy_rows_daily = []
        buy_rows_hourly = []

        prog = st.progress(0)
        total = max(1, len(universe))

        # default hourly period; the hourly renderer uses period selections elsewhere
        scan_period = "1d"

        for i, sym in enumerate(universe):
            # Daily
            r = last_band_bounce_signal_daily(sym, slope_lb_daily)
            if isinstance(r, dict) and r.get("Side") == "BUY":
                try:
                    if int(r.get("Bars Since", 10**9)) <= int(max_bars):
                        buy_rows_daily.append(r)
                except Exception:
                    pass

            # Hourly (optional)
            if include_hourly:
                rh = last_band_bounce_signal_hourly(sym, period=scan_period, slope_lb=slope_lb_hourly)
                if isinstance(rh, dict) and rh.get("Side") == "BUY":
                    # Hourly bars-since can be larger; still respect daily max_bars as a user convenience
                    try:
                        if int(rh.get("Bars Since", 10**9)) <= int(max_bars):
                            buy_rows_hourly.append(rh)
                    except Exception:
                        pass

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        st.subheader("Daily BUY signals")
        if not buy_rows_daily:
            st.info("No Daily BUY signals found within the selected bars-since window.")
        else:
            dfb = pd.DataFrame(buy_rows_daily).sort_values(["Bars Since", "R2"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(dfb, use_container_width=True)

        if include_hourly:
            st.subheader("Hourly BUY signals")
            if not buy_rows_hourly:
                st.info("No Hourly BUY signals found within the selected bars-since window.")
            else:
                dfh = pd.DataFrame(buy_rows_hourly).sort_values(["Bars Since", "R2"], ascending=[True, False]).reset_index(drop=True)
                st.dataframe(dfh, use_container_width=True)


# ---------------------------
# TAB 8: NPX 0.5-Cross Scanner
# ---------------------------
with tab8:
    st.header("NPX 0.5-Cross Scanner")
    st.caption(
        "Scans the selected universe for a **recent NPX (Normalized Price) cross of +0.5**.\n"
        "Uses the selected **Daily view range** and **NTD window** settings."
    )

    bars_since_max = st.slider(
        "Max bars since NPX +0.5 cross",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"npx05_bars_since_max_{mode}"
    )

    # Keep existing behavior: the helper already implements the +0.5 level and additional checks.
    # This tab is preserved (no UI changes), simply rendered cleanly.

    run_npx05 = st.button("Run NPX +0.5 Cross Scan", key=f"btn_run_npx05_scan_{mode}")

    if run_npx05:
        rows_up = []
        rows_dn = []

        prog = st.progress(0)
        total = max(1, len(universe))

        for i, sym in enumerate(universe):
            # Up cross (+0.5↑)
            r_up = last_daily_npx_zero_cross_with_local_slope(
                symbol=sym,
                ntd_win=ntd_window,
                daily_view_label=daily_view,
                local_slope_lb=max(2, int(slope_lb_daily)),
                max_abs_npx_at_cross=0.20,
                direction="up"
            )
            if isinstance(r_up, dict):
                try:
                    if int(r_up.get("Bars Since", 10**9)) <= int(bars_since_max):
                        rows_up.append(r_up)
                except Exception:
                    pass

            # Down cross (+0.5↓)
            r_dn = last_daily_npx_zero_cross_with_local_slope(
                symbol=sym,
                ntd_win=ntd_window,
                daily_view_label=daily_view,
                local_slope_lb=max(2, int(slope_lb_daily)),
                max_abs_npx_at_cross=0.20,
                direction="down"
            )
            if isinstance(r_dn, dict):
                try:
                    if int(r_dn.get("Bars Since", 10**9)) <= int(bars_since_max):
                        rows_dn.append(r_dn)
                except Exception:
                    pass

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("NPX +0.5 Cross UP (recent)")
            if not rows_up:
                st.info("No recent NPX +0.5 cross-up signals found.")
            else:
                dfu = pd.DataFrame(rows_up).sort_values(["Bars Since", "Local Slope"], ascending=[True, False]).reset_index(drop=True)
                st.dataframe(dfu, use_container_width=True)

        with c2:
            st.subheader("NPX +0.5 Cross DOWN (recent)")
            if not rows_dn:
                st.info("No recent NPX +0.5 cross-down signals found.")
            else:
                dfd = pd.DataFrame(rows_dn).sort_values(["Bars Since", "Local Slope"], ascending=[True, True]).reset_index(drop=True)
                st.dataframe(dfd, use_container_width=True)
# =========================
# Batch 3/3 — Tabs 9 through END + (NEW) Regression>0 + Ichimoku Cross tab
# + Patch to prevent NameError for tab variables (tab15, etc.)
# =========================

# -------------------------------------------------------------------
# SAFE TAB BINDING PATCH (prevents NameError: tab15 not defined)
# Place this *after* your st.tabs([...]) creation if you have `tabs = st.tabs(...)`.
# It will auto-bind tab1..tab18 (and tab19 if present) if they weren't unpacked.
# -------------------------------------------------------------------
try:
    tabs  # noqa
except Exception:
    tabs = None

if tabs is not None:
    _auto_tab_names = [f"tab{i}" for i in range(1, 25)]
    for _i, _nm in enumerate(_auto_tab_names):
        if _i < len(tabs) and _nm not in globals():
            globals()[_nm] = tabs[_i]

# -------------------------------------------------------------------
# FALLBACK HELPERS (only defined if missing)
# -------------------------------------------------------------------
try:
    _map_daily_view_to_period  # noqa
except Exception:
    def _map_daily_view_to_period(daily_view_label: str) -> str:
        """Map your Daily view selector into a yfinance-compatible period string."""
        m = {
            "1mo": "1mo",
            "3mo": "3mo",
            "6mo": "6mo",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y",
            "10y": "10y",
            "max": "max",
        }
        if daily_view_label in m:
            return m[daily_view_label]
        # common UI labels
        if "month" in str(daily_view_label).lower():
            return "6mo"
        if "year" in str(daily_view_label).lower():
            return "1y"
        return "6mo"

try:
    _fetch_ohlc  # noqa
except Exception:
    def _fetch_ohlc(symbol: str, interval: str = "1d", period: str = "6mo") -> pd.DataFrame:
        """Fallback OHLC fetch (uses yfinance) if your app's fetcher isn't available."""
        try:
            df = fetch_price_df(symbol, interval=interval, period=period)  # type: ignore
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df.copy()
        except Exception:
            pass

        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # yfinance sometimes returns multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
            "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume",
        })
        return df

try:
    ichimoku_kijun  # noqa
except Exception:
    def ichimoku_kijun(df: pd.DataFrame, period: int = 26) -> pd.Series:
        """Kijun-sen = (rolling high + rolling low)/2."""
        if df is None or df.empty:
            return pd.Series(dtype=float)
        h = df["High"].rolling(period).max()
        l = df["Low"].rolling(period).min()
        return (h + l) / 2.0

try:
    _linear_regression_slope  # noqa
except Exception:
    def _linear_regression_slope(y: pd.Series, lookback: int) -> float:
        """Simple slope via polyfit over last `lookback` bars."""
        if y is None or len(y) < max(3, lookback):
            return np.nan
        ys = y.dropna().astype(float)
        if len(ys) < max(3, lookback):
            return np.nan
        ys = ys.iloc[-lookback:]
        x = np.arange(len(ys), dtype=float)
        try:
            m, _b = np.polyfit(x, ys.values, 1)
            return float(m)
        except Exception:
            return np.nan

try:
    _bars_since_cross  # noqa
except Exception:
    def _bars_since_cross(diff: pd.Series, direction: str) -> int:
        """
        diff = close - line (e.g., close - kijun).
        direction: 'up' => last cross from <=0 to >0
                   'down' => last cross from >=0 to <0
        returns: bars since last cross; large int if none.
        """
        if diff is None or diff.dropna().empty:
            return 10**9
        d = diff.dropna()
        s = np.sign(d.values)
        # treat zeros as previous sign for stability
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i - 1]
        idx = None
        if direction == "up":
            for i in range(1, len(s)):
                if s[i - 1] <= 0 and s[i] > 0:
                    idx = i
        else:
            for i in range(1, len(s)):
                if s[i - 1] >= 0 and s[i] < 0:
                    idx = i
        if idx is None:
            return 10**9
        return int((len(s) - 1) - idx)

try:
    _r2_of_fit  # noqa
except Exception:
    def _r2_of_fit(y: pd.Series, lookback: int) -> float:
        if y is None or len(y) < max(5, lookback):
            return np.nan
        ys = y.dropna().astype(float)
        if len(ys) < max(5, lookback):
            return np.nan
        ys = ys.iloc[-lookback:]
        x = np.arange(len(ys), dtype=float)
        try:
            m, b = np.polyfit(x, ys.values, 1)
            yhat = m * x + b
            ss_res = float(np.sum((ys.values - yhat) ** 2))
            ss_tot = float(np.sum((ys.values - np.mean(ys.values)) ** 2))
            if ss_tot == 0:
                return np.nan
            return float(1 - ss_res / ss_tot)
        except Exception:
            return np.nan

# -------------------------------------------------------------------
# TAB 9: Fib
# -------------------------------------------------------------------
with tab9:
    st.header("Fib")
    st.caption(
        "Fibonacci context for the selected symbol.\n"
        "Keeps existing UI behavior: uses the currently selected symbol/state."
    )

    sel = st.session_state.get("ticker", None)
    if not sel:
        st.info("Run **Forecast** (or select a symbol) to populate Fib context.")
    else:
        period = _map_daily_view_to_period(st.session_state.get("daily_view", "6mo"))
        df = _fetch_ohlc(sel, interval="1d", period=period)
        if df.empty:
            st.warning("No data available.")
        else:
            closes = df["Close"].dropna()
            if closes.empty:
                st.warning("No Close series available.")
            else:
                lo = float(closes.min())
                hi = float(closes.max())
                rng = hi - lo

                if rng <= 0:
                    st.info("Insufficient range for Fib levels.")
                else:
                    fibs = {
                        "0.0%": hi,
                        "23.6%": hi - 0.236 * rng,
                        "38.2%": hi - 0.382 * rng,
                        "50.0%": hi - 0.500 * rng,
                        "61.8%": hi - 0.618 * rng,
                        "78.6%": hi - 0.786 * rng,
                        "100%": lo,
                    }

                    st.subheader(f"{sel} — Fib Levels (Daily, {period})")
                    st.dataframe(
                        pd.DataFrame([{"Level": k, "Price": v} for k, v in fibs.items()]),
                        use_container_width=True
                    )

                    # Optional lightweight chart, styled using your existing helpers if present
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(df.index, df["Close"].values, label="Close")
                    for k, v in fibs.items():
                        ax.axhline(v, linewidth=0.8, alpha=0.65, label=f"Fib {k}")
                    ax.set_title(f"{sel} — Close with Fib Levels")
                    try:
                        style_axes(ax)
                    except Exception:
                        pass
                    ax.legend(loc="upper left", ncol=2)
                    st.pyplot(fig, use_container_width=True)

# -------------------------------------------------------------------
# TAB 10: Fib NPX 0.0 Signal Scanner
# -------------------------------------------------------------------
with tab10:
    st.header("Fib NPX 0.0 Signal Scanner")
    st.caption(
        "Scans symbols for a **recent NPX cross of 0.0** (proxy for mean reversion / neutrality).\n"
        "Uses your existing NPX helper if present; otherwise falls back to a simple normalized-price approximation."
    )

    bars_since_max = st.slider(
        "Max bars since NPX 0.0 cross",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"npx00_bars_since_{mode}"
    )

    run_npx00 = st.button("Run NPX 0.0 Cross Scan", key=f"btn_run_npx00_{mode}")

    if run_npx00:
        rows_up, rows_dn = [], []
        prog = st.progress(0)
        total = max(1, len(universe))

        for i, sym in enumerate(universe):
            # Prefer your existing helper if you have it
            try:
                r_up = last_daily_npx_zero_cross_with_local_slope(
                    symbol=sym,
                    ntd_win=ntd_window,
                    daily_view_label=daily_view,
                    local_slope_lb=max(2, int(slope_lb_daily)),
                    max_abs_npx_at_cross=0.20,
                    direction="up"
                )
                r_dn = last_daily_npx_zero_cross_with_local_slope(
                    symbol=sym,
                    ntd_win=ntd_window,
                    daily_view_label=daily_view,
                    local_slope_lb=max(2, int(slope_lb_daily)),
                    max_abs_npx_at_cross=0.20,
                    direction="down"
                )
            except Exception:
                # Fallback: approximate NPX with z-score of close over window
                period = _map_daily_view_to_period(daily_view)
                df = _fetch_ohlc(sym, interval="1d", period=period)
                r_up, r_dn = None, None
                if not df.empty and "Close" in df:
                    c = df["Close"].astype(float).dropna()
                    if len(c) >= max(30, int(ntd_window)):
                        win = int(ntd_window)
                        mu = c.rolling(win).mean()
                        sd = c.rolling(win).std()
                        npx = (c - mu) / sd.replace(0, np.nan)
                        diff = npx  # cross 0
                        bs_up = _bars_since_cross(diff, "up")
                        bs_dn = _bars_since_cross(diff, "down")
                        if bs_up <= bars_since_max:
                            r_up = {"Symbol": sym, "Bars Since": bs_up, "Local Slope": _linear_regression_slope(c, int(slope_lb_daily))}
                        if bs_dn <= bars_since_max:
                            r_dn = {"Symbol": sym, "Bars Since": bs_dn, "Local Slope": _linear_regression_slope(c, int(slope_lb_daily))}

            if isinstance(r_up, dict):
                if int(r_up.get("Bars Since", 10**9)) <= int(bars_since_max):
                    rows_up.append(r_up)
            if isinstance(r_dn, dict):
                if int(r_dn.get("Bars Since", 10**9)) <= int(bars_since_max):
                    rows_dn.append(r_dn)

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("NPX 0.0 Cross UP")
            if not rows_up:
                st.info("No recent NPX 0.0 cross-up signals.")
            else:
                st.dataframe(
                    pd.DataFrame(rows_up).sort_values(["Bars Since"], ascending=True).reset_index(drop=True),
                    use_container_width=True
                )
        with c2:
            st.subheader("NPX 0.0 Cross DOWN")
            if not rows_dn:
                st.info("No recent NPX 0.0 cross-down signals.")
            else:
                st.dataframe(
                    pd.DataFrame(rows_dn).sort_values(["Bars Since"], ascending=True).reset_index(drop=True),
                    use_container_width=True
                )

# -------------------------------------------------------------------
# TAB 11: Slope Direction Scan
# -------------------------------------------------------------------
with tab11:
    st.header("Slope Direction Scan")
    st.caption(
        "Scans for **positive vs negative** Daily slope over the selected Daily slope lookback."
    )

    run_slope_scan = st.button("Run Slope Direction Scan (Daily)", key=f"btn_run_slope_scan_{mode}")

    if run_slope_scan:
        up, dn = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                m = _linear_regression_slope(c, int(slope_lb_daily))
                if np.isfinite(m):
                    row = {"Symbol": sym, "Daily Slope": float(m)}
                    if m > 0:
                        up.append(row)
                    elif m < 0:
                        dn.append(row)
            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Daily slope > 0")
            if not up:
                st.info("No symbols with positive daily slope.")
            else:
                st.dataframe(pd.DataFrame(up).sort_values("Daily Slope", ascending=False), use_container_width=True)
        with c2:
            st.subheader("Daily slope < 0")
            if not dn:
                st.info("No symbols with negative daily slope.")
            else:
                st.dataframe(pd.DataFrame(dn).sort_values("Daily Slope", ascending=True), use_container_width=True)

# -------------------------------------------------------------------
# TAB 12: Trendline Direction Lists
# -------------------------------------------------------------------
with tab12:
    st.header("Trendline Direction Lists")
    st.caption(
        "Uses **regression slope** on Daily close over the selected lookback.\n"
        "Creates lists for regression slope > 0 vs < 0."
    )

    run_trendline_dir = st.button("Run Trendline Direction Lists (Daily)", key=f"btn_run_trendline_dir_{mode}")

    if run_trendline_dir:
        pos, neg = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                m = _linear_regression_slope(c, int(slope_lb_daily))
                r2 = _r2_of_fit(c, int(slope_lb_daily))
                if np.isfinite(m):
                    row = {"Symbol": sym, "Reg Slope": float(m), "R2": float(r2) if np.isfinite(r2) else np.nan}
                    if m > 0:
                        pos.append(row)
                    elif m < 0:
                        neg.append(row)
            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Regression slope > 0")
            if not pos:
                st.info("No symbols with positive regression slope.")
            else:
                st.dataframe(pd.DataFrame(pos).sort_values(["R2", "Reg Slope"], ascending=[False, False]), use_container_width=True)
        with c2:
            st.subheader("Regression slope < 0")
            if not neg:
                st.info("No symbols with negative regression slope.")
            else:
                st.dataframe(pd.DataFrame(neg).sort_values(["R2", "Reg Slope"], ascending=[False, True]), use_container_width=True)

# -------------------------------------------------------------------
# TAB 13: NTD Hot List
# -------------------------------------------------------------------
with tab13:
    st.header("NTD Hot List")
    st.caption(
        "Lists symbols with **high absolute Hourly NTD** (default threshold 0.75).\n"
        "Uses your existing Hourly NTD helper if available."
    )

    hot_thr = st.slider(
        "NTD threshold (abs)",
        min_value=0.10, max_value=0.99,
        value=0.75, step=0.01,
        key=f"ntd_hot_thr_{mode}"
    )

    run_ntd_hot = st.button("Run NTD Hot List (Hourly)", key=f"btn_run_ntd_hot_{mode}")

    if run_ntd_hot:
        hot_pos, hot_neg = [], []
        prog = st.progress(0)
        total = max(1, len(universe))

        for i, sym in enumerate(universe):
            try:
                v, ts = last_hourly_ntd_value(sym, ntd_window, period="1d")
            except Exception:
                v, ts = (np.nan, None)

            if np.isfinite(v):
                if float(v) >= float(hot_thr):
                    hot_pos.append({"Symbol": sym, "NTD (last)": float(v), "AsOf": ts})
                elif float(v) <= -float(hot_thr):
                    hot_neg.append({"Symbol": sym, "NTD (last)": float(v), "AsOf": ts})

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"NTD ≥ +{hot_thr:.2f}")
            if not hot_pos:
                st.info("No symbols above the positive threshold.")
            else:
                st.dataframe(pd.DataFrame(hot_pos).sort_values("NTD (last)", ascending=False), use_container_width=True)

        with c2:
            st.subheader(f"NTD ≤ -{hot_thr:.2f}")
            if not hot_neg:
                st.info("No symbols below the negative threshold.")
            else:
                st.dataframe(pd.DataFrame(hot_neg).sort_values("NTD (last)", ascending=True), use_container_width=True)

# -------------------------------------------------------------------
# TAB 14: NTD NPX 0.0-0.2 Scanner
# -------------------------------------------------------------------
with tab14:
    st.header("NTD NPX 0.0-0.2 Scanner")
    st.caption(
        "Finds symbols where Daily NPX is within **[0.0, 0.2]** (or [-0.2, 0.0]) "
        "and Daily slope confirms direction."
    )

    npx_lo, npx_hi = st.slider(
        "NPX band (absolute)",
        min_value=0.0, max_value=1.0,
        value=(0.0, 0.2), step=0.01,
        key=f"ntd_npx_band_{mode}"
    )

    run_npx_band = st.button("Run NTD NPX Band Scan (Daily)", key=f"btn_run_ntd_npx_band_{mode}")

    if run_npx_band:
        rows_pos, rows_neg = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                if len(c) >= max(30, int(ntd_window)):
                    win = int(ntd_window)
                    mu = c.rolling(win).mean()
                    sd = c.rolling(win).std().replace(0, np.nan)
                    npx = (c - mu) / sd
                    last_npx = float(npx.dropna().iloc[-1]) if not npx.dropna().empty else np.nan
                    slope = _linear_regression_slope(c, int(slope_lb_daily))

                    if np.isfinite(last_npx) and np.isfinite(slope):
                        if (last_npx >= npx_lo and last_npx <= npx_hi):
                            rows_pos.append({"Symbol": sym, "NPX": last_npx, "Daily Slope": float(slope)})
                        if (last_npx <= -npx_lo and last_npx >= -npx_hi):
                            rows_neg.append({"Symbol": sym, "NPX": last_npx, "Daily Slope": float(slope)})

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"NPX in [{npx_lo:.2f}, {npx_hi:.2f}]")
            if not rows_pos:
                st.info("No symbols in the positive NPX band.")
            else:
                st.dataframe(pd.DataFrame(rows_pos).sort_values(["NPX"], ascending=True), use_container_width=True)

        with c2:
            st.subheader(f"NPX in [-{npx_hi:.2f}, -{npx_lo:.2f}]")
            if not rows_neg:
                st.info("No symbols in the negative NPX band.")
            else:
                st.dataframe(pd.DataFrame(rows_neg).sort_values(["NPX"], ascending=False), use_container_width=True)

# -------------------------------------------------------------------
# TAB 15: Uptrend vs Downtrend
# -------------------------------------------------------------------
with tab15:
    st.header("Uptrend vs Downtrend")
    st.caption(
        "Summary buckets based on Daily regression slope and R² quality.\n"
        "Uptrend = slope > 0; Downtrend = slope < 0."
    )

    r2_thr = st.slider(
        "R² threshold",
        min_value=0.00, max_value=0.95,
        value=0.45, step=0.01,
        key=f"updn_r2_thr_{mode}"
    )

    run_updn = st.button("Run Uptrend vs Downtrend (Daily)", key=f"btn_run_updn_{mode}")

    if run_updn:
        up_good, up_weak, dn_good, dn_weak = [], [], [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                m = _linear_regression_slope(c, int(slope_lb_daily))
                r2 = _r2_of_fit(c, int(slope_lb_daily))
                if np.isfinite(m) and np.isfinite(r2):
                    row = {"Symbol": sym, "Reg Slope": float(m), "R2": float(r2)}
                    if m > 0 and r2 >= r2_thr:
                        up_good.append(row)
                    elif m > 0:
                        up_weak.append(row)
                    elif m < 0 and r2 >= r2_thr:
                        dn_good.append(row)
                    elif m < 0:
                        dn_weak.append(row)

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        st.subheader("Uptrend (slope > 0)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**R² ≥ {r2_thr:.2f}**")
            st.dataframe(pd.DataFrame(up_good).sort_values(["R2", "Reg Slope"], ascending=[False, False]) if up_good else pd.DataFrame(),
                         use_container_width=True)
        with c2:
            st.markdown(f"**R² < {r2_thr:.2f}**")
            st.dataframe(pd.DataFrame(up_weak).sort_values(["R2", "Reg Slope"], ascending=[False, False]) if up_weak else pd.DataFrame(),
                         use_container_width=True)

        st.subheader("Downtrend (slope < 0)")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"**R² ≥ {r2_thr:.2f}**")
            st.dataframe(pd.DataFrame(dn_good).sort_values(["R2", "Reg Slope"], ascending=[False, True]) if dn_good else pd.DataFrame(),
                         use_container_width=True)
        with c4:
            st.markdown(f"**R² < {r2_thr:.2f}**")
            st.dataframe(pd.DataFrame(dn_weak).sort_values(["R2", "Reg Slope"], ascending=[False, True]) if dn_weak else pd.DataFrame(),
                         use_container_width=True)

# -------------------------------------------------------------------
# TAB 16: Ichimoku Kijun Scanner
# -------------------------------------------------------------------
with tab16:
    st.header("Ichimoku Kijun Scanner")
    st.caption(
        "Daily scan: price above/below Kijun-sen and recent cross detection."
    )

    kijun_period = st.slider(
        "Kijun period",
        min_value=9, max_value=52,
        value=26, step=1,
        key=f"kijun_period_{mode}"
    )
    max_bars_cross = st.slider(
        "Max bars since Kijun cross",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"kijun_cross_bars_{mode}"
    )

    run_kijun_scan = st.button("Run Ichimoku Kijun Scan (Daily)", key=f"btn_run_kijun_{mode}")

    if run_kijun_scan:
        above, below, cross_up, cross_dn = [], [], [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and {"High", "Low", "Close"}.issubset(df.columns):
                kij = ichimoku_kijun(df, period=int(kijun_period))
                c = df["Close"].astype(float)
                diff = (c - kij).dropna()

                if not diff.empty:
                    last = float(diff.iloc[-1])
                    bs_up = _bars_since_cross(diff, "up")
                    bs_dn = _bars_since_cross(diff, "down")

                    row = {"Symbol": sym, "Close-Kijun": last, "BarsSinceUpCross": bs_up, "BarsSinceDownCross": bs_dn}
                    if last > 0:
                        above.append(row)
                    elif last < 0:
                        below.append(row)
                    if bs_up <= max_bars_cross:
                        cross_up.append(row)
                    if bs_dn <= max_bars_cross:
                        cross_dn.append(row)

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Recent Kijun Cross UP")
            if not cross_up:
                st.info("No recent cross-up signals.")
            else:
                st.dataframe(pd.DataFrame(cross_up).sort_values("BarsSinceUpCross", ascending=True), use_container_width=True)
        with c2:
            st.subheader("Recent Kijun Cross DOWN")
            if not cross_dn:
                st.info("No recent cross-down signals.")
            else:
                st.dataframe(pd.DataFrame(cross_dn).sort_values("BarsSinceDownCross", ascending=True), use_container_width=True)

# -------------------------------------------------------------------
# TAB 17: R² > 45% Daily/Hourly
# -------------------------------------------------------------------
with tab17:
    st.header("R² > 45% Daily/Hourly")
    st.caption("Lists symbols whose regression fit quality (R²) is above the selected threshold.")

    r2_thr = st.slider(
        "R² threshold (good fit)",
        min_value=0.10, max_value=0.95,
        value=0.45, step=0.01,
        key=f"r2_good_thr_{mode}"
    )

    run_r2_good = st.button("Run R² > threshold Scan", key=f"btn_run_r2_good_{mode}")

    if run_r2_good:
        rows_daily, rows_hourly = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            # Daily
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                r2 = _r2_of_fit(c, int(slope_lb_daily))
                m = _linear_regression_slope(c, int(slope_lb_daily))
                if np.isfinite(r2) and float(r2) >= float(r2_thr):
                    rows_daily.append({"Symbol": sym, "R2": float(r2), "Reg Slope": float(m) if np.isfinite(m) else np.nan})

            # Hourly (best-effort)
            try:
                dfh = _fetch_ohlc(sym, interval="1h", period="1mo")
                if not dfh.empty and "Close" in dfh:
                    ch = dfh["Close"].astype(float).dropna()
                    r2h = _r2_of_fit(ch, int(slope_lb_hourly))
                    mh = _linear_regression_slope(ch, int(slope_lb_hourly))
                    if np.isfinite(r2h) and float(r2h) >= float(r2_thr):
                        rows_hourly.append({"Symbol": sym, "R2": float(r2h), "Reg Slope": float(mh) if np.isfinite(mh) else np.nan})
            except Exception:
                pass

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Daily")
            if not rows_daily:
                st.info("No Daily symbols above the threshold.")
            else:
                st.dataframe(pd.DataFrame(rows_daily).sort_values("R2", ascending=False), use_container_width=True)

        with c2:
            st.subheader("Hourly")
            if not rows_hourly:
                st.info("No Hourly symbols above the threshold.")
            else:
                st.dataframe(pd.DataFrame(rows_hourly).sort_values("R2", ascending=False), use_container_width=True)

# -------------------------------------------------------------------
# TAB 18: R² < 45% Daily/Hourly
# -------------------------------------------------------------------
with tab18:
    st.header("R² < 45% Daily/Hourly")
    st.caption("Lists symbols whose regression fit quality (R²) is below the selected threshold.")

    r2_thr = st.slider(
        "R² threshold (weak fit)",
        min_value=0.10, max_value=0.95,
        value=0.45, step=0.01,
        key=f"r2_weak_thr_{mode}"
    )

    run_r2_weak = st.button("Run R² < threshold Scan", key=f"btn_run_r2_weak_{mode}")

    if run_r2_weak:
        rows_daily, rows_hourly = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            # Daily
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if not df.empty and "Close" in df:
                c = df["Close"].astype(float).dropna()
                r2 = _r2_of_fit(c, int(slope_lb_daily))
                m = _linear_regression_slope(c, int(slope_lb_daily))
                if np.isfinite(r2) and float(r2) < float(r2_thr):
                    rows_daily.append({"Symbol": sym, "R2": float(r2), "Reg Slope": float(m) if np.isfinite(m) else np.nan})

            # Hourly (best-effort)
            try:
                dfh = _fetch_ohlc(sym, interval="1h", period="1mo")
                if not dfh.empty and "Close" in dfh:
                    ch = dfh["Close"].astype(float).dropna()
                    r2h = _r2_of_fit(ch, int(slope_lb_hourly))
                    mh = _linear_regression_slope(ch, int(slope_lb_hourly))
                    if np.isfinite(r2h) and float(r2h) < float(r2_thr):
                        rows_hourly.append({"Symbol": sym, "R2": float(r2h), "Reg Slope": float(mh) if np.isfinite(mh) else np.nan})
            except Exception:
                pass

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Daily")
            if not rows_daily:
                st.info("No Daily symbols below the threshold.")
            else:
                st.dataframe(pd.DataFrame(rows_daily).sort_values("R2", ascending=True), use_container_width=True)

        with c2:
            st.subheader("Hourly")
            if not rows_hourly:
                st.info("No Hourly symbols below the threshold.")
            else:
                st.dataframe(pd.DataFrame(rows_hourly).sort_values("R2", ascending=True), use_container_width=True)

# -------------------------------------------------------------------
# (NEW) TAB 19: Regression>0 + Ichimoku Cross Scanner (Daily)
# -------------------------------------------------------------------
# NOTE: Add a 19th tab label in your st.tabs([...]) list:
# e.g., "...", "R² < 45% Daily/Hourly", "Regression>0 + Ichimoku Cross (NEW)"
# -------------------------------------------------------------------
try:
    tab19  # noqa
except Exception:
    # If you haven't added the 19th tab label yet, this will safely render in-page without breaking
    tab19 = st.container()

with tab19:
    st.header("Regression>0 + Ichimoku Cross (NEW)")
    st.caption(
        "Daily scan:\n"
        "- Regression slope **> 0** (over Daily slope lookback)\n"
        "- Price recently crossed **Kijun** (Ichimoku line) **UP** or **DOWN**\n"
        "Use sliders to limit **bars since cross**."
    )

    kijun_period = st.slider(
        "Kijun period (Ichimoku line)",
        min_value=9, max_value=52,
        value=26, step=1,
        key=f"new_kijun_period_{mode}"
    )

    max_bars_up = st.slider(
        "Max bars since CROSS UP (price > Kijun)",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"new_kijun_maxbars_up_{mode}"
    )

    max_bars_dn = st.slider(
        "Max bars since CROSS DOWN (price < Kijun)",
        min_value=0, max_value=120,
        value=10, step=1,
        key=f"new_kijun_maxbars_dn_{mode}"
    )

    run_new_scan = st.button("Run Regression>0 + Ichimoku Cross Scan", key=f"btn_run_reg_ich_cross_{mode}")

    if run_new_scan:
        rows_up, rows_dn = [], []
        prog = st.progress(0)
        total = max(1, len(universe))
        period = _map_daily_view_to_period(daily_view)

        for i, sym in enumerate(universe):
            df = _fetch_ohlc(sym, interval="1d", period=period)
            if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
                prog.progress(int(((i + 1) / total) * 100))
                continue

            c = df["Close"].astype(float).dropna()
            if len(c) < max(30, int(slope_lb_daily), int(kijun_period) + 5):
                prog.progress(int(((i + 1) / total) * 100))
                continue

            # Condition 1: regression slope > 0
            reg_slope = _linear_regression_slope(c, int(slope_lb_daily))
            if not (np.isfinite(reg_slope) and float(reg_slope) > 0):
                prog.progress(int(((i + 1) / total) * 100))
                continue

            # Kijun line + cross detection
            kij = ichimoku_kijun(df, period=int(kijun_period))
            diff = (df["Close"].astype(float) - kij).dropna()

            bs_up = _bars_since_cross(diff, "up")
            bs_dn = _bars_since_cross(diff, "down")

            # keep same symbol row shape for tables
            base = {
                "Symbol": sym,
                "Reg Slope": float(reg_slope),
                "BarsSinceUpCross": int(bs_up),
                "BarsSinceDownCross": int(bs_dn),
            }

            if int(bs_up) <= int(max_bars_up):
                rows_up.append(base)
            if int(bs_dn) <= int(max_bars_dn):
                rows_dn.append(base)

            prog.progress(int(((i + 1) / total) * 100))

        prog.empty()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Regression>0 + Recent CROSS UP")
            if not rows_up:
                st.info("No symbols meet Regression>0 + Cross-UP within the selected bars.")
            else:
                st.dataframe(
                    pd.DataFrame(rows_up).sort_values(["BarsSinceUpCross", "Reg Slope"], ascending=[True, False]).reset_index(drop=True),
                    use_container_width=True
                )
        with c2:
            st.subheader("Regression>0 + Recent CROSS DOWN")
            if not rows_dn:
                st.info("No symbols meet Regression>0 + Cross-DOWN within the selected bars.")
            else:
                st.dataframe(
                    pd.DataFrame(rows_dn).sort_values(["BarsSinceDownCross", "Reg Slope"], ascending=[True, False]).reset_index(drop=True),
                    use_container_width=True
                )

# -------------------------------------------------------------------
# INSERTION SNIPPET — Under the Forecast Button (Tab 1)
# -------------------------------------------------------------------
# Add this *directly under* your Forecast button block where `symbol` is the selected ticker.
# It prints the current Daily trend + slope direction (uses current lookbacks).
#
# Example usage (inside Tab 1, after you have `symbol` and have fetched daily/h1 data):
#
#   st.markdown(render_daily_state_statement(symbol, daily_view, slope_lb_daily))
#
# And for Hourly chart section:
#   st.caption(render_daily_state_compact(symbol, daily_view, slope_lb_daily))
# -------------------------------------------------------------------
try:
    render_daily_state_statement  # noqa
except Exception:
    def render_daily_state_statement(symbol: str, daily_view_label: str, slope_lb: int) -> str:
        period = _map_daily_view_to_period(daily_view_label)
        df = _fetch_ohlc(symbol, interval="1d", period=period)
        if df.empty or "Close" not in df:
            return "Daily Trend/Slope: (unavailable — no daily data)"
        c = df["Close"].astype(float).dropna()
        m = _linear_regression_slope(c, int(slope_lb))
        trend = "UP" if np.isfinite(m) and m > 0 else ("DOWN" if np.isfinite(m) and m < 0 else "FLAT")
        return f"**Daily Trend:** {trend}  |  **Daily Slope:** {m:.6f}" if np.isfinite(m) else f"**Daily Trend:** {trend}  |  **Daily Slope:** n/a"

    def render_daily_state_compact(symbol: str, daily_view_label: str, slope_lb: int) -> str:
        period = _map_daily_view_to_period(daily_view_label)
        df = _fetch_ohlc(symbol, interval="1d", period=period)
        if df.empty or "Close" not in df:
            return "Daily: n/a"
        c = df["Close"].astype(float).dropna()
        m = _linear_regression_slope(c, int(slope_lb))
        if not np.isfinite(m):
            return "Daily: n/a"
        trend = "UP" if m > 0 else ("DOWN" if m < 0 else "FLAT")
        return f"Daily Trend={trend}, Daily Slope={m:.6f}"
