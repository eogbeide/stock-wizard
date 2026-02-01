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

def daily_regression_guidance_message(price_series: pd.Series,
                                      slope_lb: int,
                                      recent_bars: int = 12) -> str | None:
    """
    UPDATED (THIS REQUEST):
      Under the Forecast button, display a statement reflecting the current DAILY regression.

    Rules:
      If slope > 0 AND price reversed from LOWER -2σ heading upward -> show BUY-favor message.
      If slope > 0 AND price reversed from UPPER +2σ heading downward -> show BUY-favor but cautious message.
    """
    s = _coerce_1d_series(price_series).dropna()
    if s.empty or len(s) < 5:
        return None

    slope_lb = int(max(3, slope_lb))
    yhat, up, lo, m, r2 = regression_with_band(s, lookback=slope_lb, z=2.0)
    if yhat is None or up is None or lo is None:
        return None
    if not np.isfinite(m) or float(m) <= 0.0:
        return None

    # Align on the regression window index (last slope_lb bars)
    p = _coerce_1d_series(s).reindex(lo.index)
    u = _coerce_1d_series(up).reindex(p.index)
    l = _coerce_1d_series(lo).reindex(p.index)

    ok = p.notna() & u.notna() & l.notna()
    p = p[ok]; u = u[ok]; l = l[ok]
    if len(p) < 3:
        return None

    rb = int(max(3, min(int(recent_bars), len(p) - 1)))
    pw = p.iloc[-(rb+1):]
    uw = u.reindex(pw.index)
    lw = l.reindex(pw.index)

    # Determine the most recent meaningful "touch" and confirm direction by last bar move.
    touch_lo = pw <= lw
    touch_hi = pw >= uw

    last_lo = touch_lo[touch_lo].index[-1] if touch_lo.any() else None
    last_hi = touch_hi[touch_hi].index[-1] if touch_hi.any() else None

    def _is_heading_up() -> bool:
        try:
            return float(p.iloc[-1]) > float(p.iloc[-2])
        except Exception:
            return False

    def _is_heading_down() -> bool:
        try:
            return float(p.iloc[-1]) < float(p.iloc[-2])
        except Exception:
            return False

    inside_now = bool((p.iloc[-1] <= u.iloc[-1]) and (p.iloc[-1] >= l.iloc[-1]))

    # Lower reversal: touched/breached lower recently AND now inside AND heading up
    lower_ok = False
    if last_lo is not None and inside_now and _is_heading_up():
        try:
            lower_ok = bool(float(p.loc[last_lo]) <= float(l.reindex(p.index).loc[last_lo]))
        except Exception:
            lower_ok = False

    # Upper reversal: touched/breached upper recently AND now inside AND heading down
    upper_ok = False
    if last_hi is not None and inside_now and _is_heading_down():
        try:
            upper_ok = bool(float(p.loc[last_hi]) >= float(u.reindex(p.index).loc[last_hi]))
        except Exception:
            upper_ok = False

    # If both, prefer the most recent touch
    pick = None
    if lower_ok and upper_ok:
        pick = "upper" if last_hi >= last_lo else "lower"
    elif lower_ok:
        pick = "lower"
    elif upper_ok:
        pick = "upper"

    if pick == "lower":
        return "Daily Slope is positive, which favors a Buy position."
    if pick == "upper":
        return ("Daily Slope is positive, which favors a Buy position but be cautious when placing a Buy position "
                "because the price has reached the reversal points.")
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
# Sessions (PST)
# ---------------------------
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

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

def draw_session_lines(ax, lines: dict, alpha: float = 0.35):
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:orange", alpha=alpha)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=alpha)

    handles = [
        Line2D([0], [0], color="tab:blue",   linestyle="-",  linewidth=1.6, label="London Open"),
        Line2D([0], [0], color="tab:blue",   linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], color="tab:orange", linestyle="-",  linewidth=1.6, label="New York Open"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels

# ---------------------------
# News (Yahoo Finance)
# ---------------------------
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

def draw_news_markers(ax, times, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# ---------------------------
# Channel-in-range helpers for NTD panel
# ---------------------------
def channel_state_series(price: pd.Series, sup: pd.Series, res: pd.Series, eps: float = 0.0) -> pd.Series:
    p = _coerce_1d_series(price)
    s_sup = _coerce_1d_series(sup).reindex(p.index)
    s_res = _coerce_1d_series(res).reindex(p.index)
    state = pd.Series(index=p.index, dtype=float)
    ok = p.notna() & s_sup.notna() & s_res.notna()
    if ok.any():
        below = p < (s_sup - eps)
        above = p > (s_res + eps)
        between = ~(below | above)
        state[ok & below] = -1
        state[ok & between] = 0
        state[ok & above] = 1
    return state

def _true_spans(mask: pd.Series):
    spans = []
    if mask is None or mask.empty:
        return spans
    s = mask.fillna(False).astype(bool)
    start = None
    prev_t = None
    for t, val in s.items():
        if val and start is None:
            start = t
        if not val and start is not None:
            if prev_t is not None:
                spans.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None and prev_t is not None:
        spans.append((start, prev_t))
    return spans

def overlay_inrange_on_ntd(ax, price: pd.Series, sup: pd.Series, res: pd.Series):
    state = channel_state_series(price, sup, res)
    in_mask = (state == 0)
    for a, b in _true_spans(in_mask):
        try:
            ax.axvspan(a, b, color="gold", alpha=0.15, zorder=1)
        except Exception:
            pass
    ax.plot([], [], linewidth=8, color="gold", alpha=0.20, label="In Range (S↔R)")
    enter_from_below = (state.shift(1) == -1) & (state == 0)
    enter_from_above = (state.shift(1) == 1) & (state == 0)
    if enter_from_below.any():
        ax.scatter(price.index[enter_from_below], [0.92]*int(enter_from_below.sum()),
                   marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above], [0.92]*int(enter_from_above.sum()),
                   marker="v", s=60, color="tab:orange", zorder=7, label="Enter from R")

    last = state.dropna().iloc[-1] if state.dropna().shape[0] else np.nan
    if np.isfinite(last):
        if last == 0:
            lbl, col = "IN RANGE (S↔R)", "black"
        elif last > 0:
            lbl, col = "Above R", "tab:orange"
        else:
            lbl, col = "Below S", "tab:red"
        ax.text(0.99, 0.94, lbl, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85))
    return last

def rolling_midline(series_like: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll = s.rolling(window, min_periods=1)
    mid = (roll.max() + roll.min()) / 2.0
    return mid.reindex(s.index)

def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

# =========================
# Part 8/10 — bullbear.py
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
# Shared hourly renderer (Stock & Forex)
# ---------------------------
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return None

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill()
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
    slope_sig_h = m_h

    rev_prob_h = slope_reversal_probability(
        hc,
        slope_sig_h,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_hourly,
        horizon=rev_horizon,
    )

    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    ax2w = None
    if show_nrsi:
        fig2, (ax2, ax2w) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.34)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)

    try:
        fig2.patch.set_facecolor("white")
    except Exception:
        pass

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
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, slope_sig_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax2, news_pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
    )

    macd_sig = find_macd_hma_sr_signal(
        close=hc, hma=hma_h, macd=macd_h, sup=sup_h, res=res_h,
        global_trend_slope=global_m_h, prox=sr_prox_pct
    )

    macd_instr_txt = "MACD/HMA55: n/a"
    if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
        side = macd_sig["side"]
        macd_instr_txt = f"MACD/HMA55: {side} @ {fmt_price_val(macd_sig['price'])}"
        annotate_macd_signal(ax2, macd_sig["time"], macd_sig["price"], side)

    ax2.text(
        0.01, 0.98, macd_instr_txt,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        zorder=20
    )

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
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
             f"Slope: {fmt_slope(slope_sig_h)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_h)}",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
    ax2.text(0.50, 0.02,
             f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, color="black",
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

    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    npx_h_for_sig = compute_normalized_price(hc, window=ntd_window)
    fib_sig_h = None
    fib_trig_chart = None

    trig_disp = None
    return {
        "trade_instruction": instr_txt,
        "fib_trigger": trig_disp,
    }


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Tabs
# ---------------------------

# ---------------------------
# Matplotlib + Streamlit chart UI polish (THIS REQUEST)
# ---------------------------
try:
    plt.rcParams.update({
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.65,
        "axes.titleweight": "bold",
    })
except Exception:
    pass

st.markdown(
    """
    <style>
      /* ==========================
         Tabs: rectangular ribbons
         ========================== */
      div[data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        gap: 0.40rem !important;
        padding: 0.25rem 0.25rem 0.40rem 0.25rem !important;
        border-bottom: 1px solid rgba(49, 51, 63, 0.14) !important;
      }

      div[data-baseweb="tab"] {
        flex: 0 0 auto !important;
      }

      div[data-baseweb="tab"] > button {
        padding: 0.40rem 0.85rem !important;
        border: 1px solid rgba(49, 51, 63, 0.20) !important;
        border-radius: 6px !important; /* rectangular ribbon feel */
        background: rgba(248, 250, 252, 0.96) !important;
        box-shadow: 0 1px 0 rgba(0,0,0,0.03) !important;
        font-weight: 700 !important;
        line-height: 1.05 !important;
        transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background 120ms ease !important;
      }

      div[data-baseweb="tab"] > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 10px 22px rgba(0,0,0,0.10) !important;
        border-color: rgba(49, 51, 63, 0.32) !important;
        background: rgba(241, 245, 249, 1.0) !important;
      }

      div[data-baseweb="tab"] > button[aria-selected="true"] {
        background: linear-gradient(90deg, rgba(59,130,246,0.96), rgba(99,102,241,0.96)) !important;
        border-color: rgba(59,130,246,0.95) !important;
        box-shadow: 0 12px 24px rgba(59,130,246,0.24) !important;
      }

      div[data-baseweb="tab"] > button[aria-selected="true"] p {
        color: white !important;
      }

      div[data-baseweb="tab"] p {
        margin: 0 !important;
        font-size: 0.90rem !important;
      }

      /* ==========================
         Charts: "React UI" polish
         (Streamlit renders matplotlib via <img>)
         ========================== */
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
    </style>
    """,
    unsafe_allow_html=True
)

# UPDATED (THIS REQUEST): add a NEW tab (keeping existing 18 unchanged)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = st.tabs([
    "Original Forecast",
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
    "Ichimoku Cross Scanner (Slope>0)"
])



# ---------------------------
# TAB 1: ORIGINAL FORECAST
# ---------------------------
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data is cached for ~2 minutes after first fetch. "
            "Charts stay on the last RUN ticker until you run again.")

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

    # UPDATED (THIS REQUEST): Daily regression statement under the Forecast button
    daily_reg_msg_box = st.empty()

    fib_instruction_box = st.empty()
    trade_instruction_box = st.empty()

    if run_clicked:
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
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
        disp_ticker = st.session_state.ticker
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"**Displayed (last run):** {disp_ticker}  •  "
                   f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

        # UPDATED (THIS REQUEST): show daily regression guidance under Forecast button
        with daily_reg_msg_box.container():
            msg = daily_regression_guidance_message(df, slope_lb=slope_lb_daily, recent_bars=12)
            if isinstance(msg, str) and msg.strip():
                st.caption(msg)

        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(disp_ticker, window_days=news_window_days)

        with fib_instruction_box.container():
            st.warning(FIB_ALERT_TEXT)
            st.caption(
                "Fibonacci Reversal Trigger (confirmed): "
                "BUY when price touches near the **100%** line then prints consecutive higher closes; "
                "SELL when price touches near the **0%** line then prints consecutive lower closes."
            )
            st.caption(
                "Fibonacci NPX 0.0 Signal (THIS UPDATE): "
                "BUY when price touched **100%** and NPX crossed **up** through **0.0** recently; "
                "SELL when price touched **0%** and NPX crossed **down** through **0.0** recently."
            )

        daily_instr_txt = None
        hourly_instr_txt = None
        daily_fib_trig = None
        hourly_fib_trig = None

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
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
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
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            ntd_d_show = ntd_d.reindex(df_show.index)
            npx_d_show = npx_d_full.reindex(df_show.index)
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show = bb_up_d.reindex(df_show.index)
            bb_lo_d_show = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_show = compute_hma(df, period=hma_period).reindex(df_show.index)
            macd_d, macd_sig_d, macd_hist_d = compute_macd(df_show)

            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.33)

            try:
                fig.patch.set_facecolor("white")
            except Exception:
                pass

            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{disp_ticker} Daily — {daily_view} — History, EMA, S/R (w={sr_lb_daily}), Slope, Pivots "
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
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                               label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

            res_val_d = sup_val_d = np.nan
            try:
                res_val_d = float(res_d_show.iloc[-1])
                sup_val_d = float(sup_d_show.iloc[-1])
            except Exception:
                pass
            if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-", linewidth=1.6,
                          label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6,
                          label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
                bounce_sig_d = find_band_bounce_signal(df_show, upper_d_show, lower_d_show, m_d)
                if bounce_sig_d is not None:
                    annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

            macd_sig_d = find_macd_hma_sr_signal(
                close=df_show, hma=hma_d_show, macd=macd_d, sup=sup_d_show, res=res_d_show,
                global_trend_slope=global_m_d, prox=sr_prox_pct
            )
            macd_instr_txt_d = "MACD/HMA55: n/a"
            if macd_sig_d is not None and np.isfinite(macd_sig_d.get("price", np.nan)):
                macd_instr_txt_d = f"MACD/HMA55: {macd_sig_d['side']} @ {fmt_price_val(macd_sig_d['price'])}"
                annotate_macd_signal(ax, macd_sig_d["time"], macd_sig_d["price"], macd_sig_d["side"])

            ax.text(
                0.01, 0.98, macd_instr_txt_d,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
                zorder=30
            )

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            fib_sig_d = None

            daily_fib_trig = fib_reversal_trigger_from_extremes(
                df_show,
                proximity_pct_of_range=0.02,
                confirm_bars=int(rev_bars_confirm),
                lookback_bars=int(max(60, slope_lb_daily)),
            )

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

            ax.set_ylabel("Price")
            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

            axdw.set_title(f"Daily Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_daily})")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                              label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")
                overlay_ntd_triangles_by_trend(axdw, ntd_d_show, trend_slope=m_d, upper=0.75, lower=-0.75)
                overlay_ntd_sr_reversal_stars(axdw, price=df_show, sup=sup_d_show, res=res_d_show,
                                              trend_slope=m_d, ntd=ntd_d_show, prox=sr_prox_pct,
                                              bars_confirm=rev_bars_confirm)

            if show_npx_ntd and not npx_d_show.dropna().empty and not ntd_d_show.dropna().empty:
                overlay_npx_on_ntd(axdw, npx_d_show, ntd_d_show, mark_crosses=mark_npx_cross)
            if show_hma_rev_ntd and not hma_d_show.dropna().empty and not df_show.dropna().empty:
                overlay_hma_reversal_on_ntd(axdw, df_show, hma_d_show, lookback=hma_rev_lb,
                                            period=hma_period, ntd=ntd_d_show)

            axdw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
            axdw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
            axdw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
            axdw.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
            axdw.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")

            handles, labels = [], []
            h1, l1 = ax.get_legend_handles_labels()
            handles += h1; labels += l1
            h2, l2 = axdw.get_legend_handles_labels()
            handles += h2; labels += l2

            seen = set()
            h_u, l_u = [], []
            for h, l in zip(handles, labels):
                if not l or l in seen:
                    continue
                seen.add(l)
                h_u.append(h)
                l_u.append(l)

            fig.legend(
                handles=h_u,
                labels=l_u,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=4,
                frameon=True,
                fontsize=9,
                framealpha=0.65,
                fancybox=True,
                borderpad=0.6,
                handlelength=2.0
            )

            style_axes(ax)
            style_axes(axdw)
            st.pyplot(fig)

            daily_instr_txt = format_trade_instruction(
                trend_slope=m_d,
                buy_val=sup_val_d,
                sell_val=res_val_d,
                close_val=last_px_show,
                symbol=disp_ticker,
                global_trend_slope=global_m_d
            )

        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            out_h = render_hourly_views(
                sel=disp_ticker,
                intraday=intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.hour_range,
                is_forex=(mode == "Forex")
            )
            if isinstance(out_h, dict):
                hourly_instr_txt = out_h.get("trade_instruction", None)
                hourly_fib_trig = out_h.get("fib_trigger", None)

        if "instr_daily_text" not in st.session_state:
            st.session_state.instr_daily_text = None
        if "instr_daily_updated_at" not in st.session_state:
            st.session_state.instr_daily_updated_at = None
        if "instr_hourly_text" not in st.session_state:
            st.session_state.instr_hourly_text = None
        if "instr_hourly_updated_at" not in st.session_state:
            st.session_state.instr_hourly_updated_at = None

        now_pst = datetime.now(PACIFIC)

        if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
            if st.session_state.get("instr_daily_text") != daily_instr_txt:
                st.session_state.instr_daily_text = daily_instr_txt
                st.session_state.instr_daily_updated_at = now_pst

        if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
            if st.session_state.get("instr_hourly_text") != hourly_instr_txt:
                st.session_state.instr_hourly_text = hourly_instr_txt
                st.session_state.instr_hourly_updated_at = now_pst

        def _ts_str(dt_obj):
            if isinstance(dt_obj, datetime):
                try:
                    d = dt_obj.astimezone(PACIFIC)
                except Exception:
                    d = dt_obj
                return f"{d.strftime('%Y-%m-%d %H:%M:%S')} PST"
            return "n/a"

        with trade_instruction_box.container():
            if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
                daily_msg = f"Daily (updated {_ts_str(st.session_state.get('instr_daily_updated_at'))}): {daily_instr_txt}"
                if daily_instr_txt.startswith("ALERT:"):
                    st.error(daily_msg)
                else:
                    st.success(daily_msg)

            if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
                hourly_msg = f"Hourly (updated {_ts_str(st.session_state.get('instr_hourly_updated_at'))}): {hourly_instr_txt}"
                if hourly_instr_txt.startswith("ALERT:"):
                    st.error(hourly_msg)
                else:
                    st.success(hourly_msg)

            if isinstance(daily_fib_trig, dict):
                st.info(
                    f"Daily Fib Reversal Trigger: **{daily_fib_trig.get('side')}** "
                    f"(from {daily_fib_trig.get('from_level')}) • touch={daily_fib_trig.get('touch_time')} "
                    f"@ {fmt_price_val(daily_fib_trig.get('touch_price', np.nan))}"
                )
            else:
                st.caption("Daily Fib Reversal Trigger: none confirmed.")

            if isinstance(hourly_fib_trig, dict):
                st.info(
                    f"Hourly Fib Reversal Trigger: **{hourly_fib_trig.get('side')}** "
                    f"(from {hourly_fib_trig.get('from_level')}) • touch={hourly_fib_trig.get('touch_time')} "
                    f"@ {fmt_price_val(hourly_fib_trig.get('touch_price', np.nan))}"
                )
            elif chart in ("Hourly", "Both"):
                st.caption("Hourly Fib Reversal Trigger: none confirmed.")

        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# TAB 2: ENHANCED FORECAST
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        df = st.session_state.df_hist
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

            fig, ax = plt.subplots(figsize=(14, 5))
            fig.subplots_adjust(bottom=0.30)
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if not res_d_show.empty and not sup_d_show.empty:
                ax.hlines(float(res_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1],
                          colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                ax.hlines(float(sup_d_show.iloc[-1]), xmin=df_show.index[0], xmax=df_show.index[-1],
                          colors="tab:green", linestyles="-", linewidth=1.6, label="Support")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
            style_axes(ax)
            st.pyplot(fig)

        if view in ("Intraday", "Both"):
            render_hourly_views(
                sel=st.session_state.ticker,
                intraday=st.session_state.intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.get("hour_range","24h"),
                is_forex=(mode == "Forex")
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:, 0],
            "Upper":    ci.iloc[:, 1]
        }, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR
# ---------------------------
with tab3:
    st.header("Bull vs Bear")
    st.caption("Simple lookback performance overview (based on Bull/Bear lookback selection).")

    sel_bb = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
    try:
        dfp = yf.download(sel_bb, period=bb_period, interval="1d")[["Close"]].dropna()
    except Exception:
        dfp = pd.DataFrame()

    if dfp.empty:
        st.warning("No data available.")
    else:
        s = dfp["Close"].astype(float)
        ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1.0) if len(s) > 1 else np.nan
        st.metric(label=f"{sel_bb} return over {bb_period}", value=fmt_pct(ret))
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{sel_bb} — {bb_period} Close")
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
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

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({
                "Hourly slope (reg band)": fmt_slope(m_h),
                "Hourly R²": fmt_r2(r2_h),
                f"P(slope reverses ≤ {rev_horizon} bars) hourly": fmt_pct(slope_reversal_probability(hc, m_h, rev_hist_lb, slope_lb_hourly, rev_horizon))
            })
# =========================
# BATCH 2/3 — bullbear.py (CONTINUATION)
# Tabs 5–8 + scanner helpers
# =========================

# ---------------------------
# Scanner helpers (Daily)
# ---------------------------
@st.cache_data(ttl=300, show_spinner=False)
def daily_snapshot(ticker: str,
                   slope_lb: int,
                   ntd_win: int,
                   sr_lb: int) -> dict:
    """
    Fast daily snapshot used by scanner tabs.
    Returns dict with robust NaN handling.
    """
    try:
        s = fetch_hist(ticker)
    except Exception:
        return {"ticker": ticker}

    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return {"ticker": ticker}

    last_px = float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else np.nan

    # Daily NTD / NPX
    try:
        ntd = compute_normalized_trend(s, window=ntd_win)
        ntd_last = float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan
    except Exception:
        ntd_last = np.nan

    try:
        npx = compute_normalized_price(s, window=ntd_win)
        npx_last = float(npx.dropna().iloc[-1]) if not npx.dropna().empty else np.nan
    except Exception:
        npx_last = np.nan

    # Daily regression slope & R²
    try:
        _, _, _, m, r2 = regression_with_band(s, lookback=int(max(3, slope_lb)), z=2.0)
        m = float(m) if np.isfinite(m) else np.nan
        r2 = float(r2) if np.isfinite(r2) else np.nan
    except Exception:
        m, r2 = np.nan, np.nan

    # Daily S/R
    try:
        res = float(s.rolling(int(sr_lb), min_periods=1).max().iloc[-1])
        sup = float(s.rolling(int(sr_lb), min_periods=1).min().iloc[-1])
    except Exception:
        res, sup = np.nan, np.nan

    return {
        "ticker": ticker,
        "last": last_px,
        "ntd": ntd_last,
        "npx": npx_last,
        "slope": m,
        "r2": r2,
        "res": res,
        "sup": sup,
        "time": s.index[-1],
    }

def _last_level_cross(series_like: pd.Series, level: float):
    """
    Returns (last_cross_up_time, last_cross_dn_time).
    """
    s = _coerce_1d_series(series_like)
    if s.dropna().shape[0] < 2:
        return None, None
    s = s.dropna()
    above = s >= float(level)
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    t_up = cross_up[cross_up].index[-1] if cross_up.any() else None
    t_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None
    return t_up, t_dn

def _within_last_bars(idx: pd.DatetimeIndex, t, bars: int) -> bool:
    if t is None or not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return False
    try:
        loc = idx.get_loc(t)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        return (len(idx) - 1 - int(loc)) <= int(bars)
    except Exception:
        # If exact match fails, try nearest
        try:
            pos = idx.get_indexer([pd.Timestamp(t)], method="nearest")[0]
            return (len(idx) - 1 - int(pos)) <= int(bars)
        except Exception:
            return False

@st.cache_data(ttl=300, show_spinner=False)
def daily_npx_series(ticker: str, ntd_win: int) -> pd.Series:
    try:
        s = fetch_hist(ticker)
    except Exception:
        return pd.Series(dtype=float)
    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return compute_normalized_price(s, window=ntd_win)

@st.cache_data(ttl=300, show_spinner=False)
def daily_ntd_series(ticker: str, ntd_win: int) -> pd.Series:
    try:
        s = fetch_hist(ticker)
    except Exception:
        return pd.Series(dtype=float)
    s = _coerce_1d_series(s).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return compute_normalized_trend(s, window=ntd_win)

@st.cache_data(ttl=300, show_spinner=False)
def daily_buy_bounce_signal_time(ticker: str, slope_lb: int):
    """
    Uses regression ±2σ band bounce logic on DAILY close.
    Returns last BUY bounce time (or None).
    """
    try:
        s = fetch_hist(ticker)
    except Exception:
        return None
    s = _coerce_1d_series(s).dropna()
    if len(s) < max(20, int(slope_lb) + 5):
        return None
    yhat, up, lo, m, _ = regression_with_band(s, lookback=int(max(3, slope_lb)), z=2.0)
    sig = find_band_bounce_signal(s.reindex(yhat.index), up, lo, m)
    if sig and sig.get("side") == "BUY":
        return sig.get("time")
    return None

# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Scans the current universe for tickers whose **daily NTD** is at/below a threshold (default -0.75).")

    thr = st.slider("NTD threshold (daily)", -1.0, 0.0, -0.75, 0.01, key=f"ntd_thr_{mode}")
    include_near = st.checkbox("Include near-threshold (within 0.05)", value=True, key=f"ntd_thr_near_{mode}")
    near_eps = 0.05

    run_scan = st.button("Run NTD Scanner", key=f"btn_scan_ntd_{mode}")

    if run_scan:
        rows = []
        prog = st.progress(0)
        total = len(universe)
        for i, tkr in enumerate(universe, start=1):
            snap = daily_snapshot(tkr, slope_lb=slope_lb_daily, ntd_win=ntd_window, sr_lb=sr_lb_daily)
            ntdv = snap.get("ntd", np.nan)
            if np.isfinite(ntdv):
                ok = (ntdv <= thr) or (include_near and (ntdv <= thr + near_eps))
                if ok:
                    rows.append({
                        "Ticker": tkr,
                        "Last": snap.get("last", np.nan),
                        "NTD": ntdv,
                        "NPX": snap.get("npx", np.nan),
                        "Slope": snap.get("slope", np.nan),
                        "R²": snap.get("r2", np.nan),
                        "Support": snap.get("sup", np.nan),
                        "Resistance": snap.get("res", np.nan),
                        "AsOf": snap.get("time", None),
                    })
            prog.progress(int(i / total * 100))
        prog.empty()

        if not rows:
            st.info("No tickers matched the filter.")
        else:
            df_out = pd.DataFrame(rows)
            df_out = df_out.sort_values(["NTD", "R²"], ascending=[True, False]).reset_index(drop=True)
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"ntd_scanner_{mode.lower()}.csv",
                mime="text/csv",
                key=f"dl_ntd_{mode}",
            )

# ---------------------------
# TAB 6: Long-Term History
# ---------------------------
with tab6:
    st.header("Long-Term History")
    st.caption("Uses Yahoo 'max' history. Helpful for long-range context and trend sanity-checks.")

    lt_tkr = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    years = st.slider("Show last N years (max history is still fetched)", 1, 25, int(st.session_state.get("hist_years", 10)), 1,
                      key=f"lt_years_{mode}")
    st.session_state.hist_years = years

    show_log = st.checkbox("Log scale", value=False, key=f"lt_log_{mode}")
    show_reg = st.checkbox("Show regression + ±2σ band", value=True, key=f"lt_reg_{mode}")
    reg_lb = st.slider("Regression lookback (bars)", 30, 800, 252, 10, key=f"lt_reg_lb_{mode}")

    try:
        smax = fetch_hist_max(lt_tkr)
    except Exception:
        smax = pd.Series(dtype=float)

    smax = _coerce_1d_series(smax).dropna()
    if smax.empty:
        st.warning("No long-term data available.")
    else:
        # Keep last N years
        try:
            end = smax.index.max()
            start = end - pd.Timedelta(days=int(years * 365.25))
            sshow = smax.loc[(smax.index >= start) & (smax.index <= end)]
        except Exception:
            sshow = smax

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{lt_tkr} Long-Term (last {years}y)")
        ax.plot(sshow.index, sshow.values, label="Close")

        global_m = draw_trend_direction_line(ax, sshow, label_prefix="Trend (global)")

        if show_reg and len(sshow) >= 10:
            yhat, up, lo, m, r2 = regression_with_band(sshow, lookback=int(reg_lb), z=2.0)
            if not yhat.empty:
                ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Reg ({reg_lb}) slope={fmt_slope(m)}/bar")
            if not up.empty and not lo.empty:
                ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.80, label="+2σ")
                ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.80, label="-2σ")
            ax.text(0.50, 0.02, f"R² ({reg_lb} bars): {fmt_r2(r2)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

        if show_log:
            ax.set_yscale("log")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: Recent BUY Scanner
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner")
    st.caption(
        "Finds tickers with a **recent DAILY BUY bounce** from the -2σ regression band, "
        "and (optionally) requires global & local trend agreement."
    )

    recent_days = st.slider("How recent (days)", 1, 30, 7, 1, key=f"buy_recent_days_{mode}")
    require_agree = st.checkbox("Require global trend slope > 0 AND local slope > 0", value=True, key=f"buy_req_agree_{mode}")
    run_buy_scan = st.button("Run Recent BUY Scan", key=f"btn_scan_buy_{mode}")

    if run_buy_scan:
        rows = []
        prog = st.progress(0)
        total = len(universe)

        for i, tkr in enumerate(universe, start=1):
            try:
                s = fetch_hist(tkr)
            except Exception:
                prog.progress(int(i / total * 100))
                continue

            s = _coerce_1d_series(s).dropna()
            if len(s) < max(30, slope_lb_daily + 5):
                prog.progress(int(i / total * 100))
                continue

            # global slope: use a longer window to avoid noise (1y if available)
            glob_lb = int(min(252, len(s)))
            _, g_m = slope_line(s, glob_lb)
            yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(max(3, slope_lb_daily)), z=2.0)

            sig = None
            if not yhat.empty and not up.empty and not lo.empty:
                sig = find_band_bounce_signal(s.reindex(yhat.index), up, lo, m)

            if not sig or sig.get("side") != "BUY":
                prog.progress(int(i / total * 100))
                continue

            t_sig = sig.get("time")
            if t_sig is None:
                prog.progress(int(i / total * 100))
                continue

            # recency filter
            is_recent = False
            try:
                is_recent = (s.index.max() - pd.Timestamp(t_sig)).days <= int(recent_days)
            except Exception:
                is_recent = False
            if not is_recent:
                prog.progress(int(i / total * 100))
                continue

            # agreement filter
            agree_ok = True
            if require_agree:
                agree_ok = (np.isfinite(g_m) and np.isfinite(m) and float(g_m) > 0 and float(m) > 0)
            if not agree_ok:
                prog.progress(int(i / total * 100))
                continue

            snap = daily_snapshot(tkr, slope_lb=slope_lb_daily, ntd_win=ntd_window, sr_lb=sr_lb_daily)
            rows.append({
                "Ticker": tkr,
                "SignalTime": t_sig,
                "Last": snap.get("last", np.nan),
                "NTD": snap.get("ntd", np.nan),
                "NPX": snap.get("npx", np.nan),
                "GlobalSlope": g_m,
                "LocalSlope": m,
                "R²": r2,
                "Support": snap.get("sup", np.nan),
                "Resistance": snap.get("res", np.nan),
            })

            prog.progress(int(i / total * 100))

        prog.empty()

        if not rows:
            st.info("No tickers matched.")
        else:
            df_out = pd.DataFrame(rows)
            df_out = df_out.sort_values(["SignalTime", "R²"], ascending=[False, False]).reset_index(drop=True)
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"recent_buy_scanner_{mode.lower()}.csv",
                mime="text/csv",
                key=f"dl_buy_{mode}",
            )

# ---------------------------
# TAB 8: NPX 0.5-Cross Scanner
# ---------------------------
with tab8:
    st.header("NPX 0.5-Cross Scanner")
    st.caption(
        "Scans for recent daily **NPX** crossings of a chosen level (default +0.50). "
        "Cross up can signal strengthening momentum; cross down can signal weakening momentum."
    )

    level = st.slider("NPX level", -1.0, 1.0, 0.50, 0.01, key=f"npx_level_{mode}")
    last_bars = st.slider("Lookback bars for last crossing", 5, 120, 30, 1, key=f"npx_lastbars_{mode}")
    include_up = st.checkbox("Include Cross UP", value=True, key=f"npx_inc_up_{mode}")
    include_dn = st.checkbox("Include Cross DOWN", value=True, key=f"npx_inc_dn_{mode}")
    run_npx_scan = st.button("Run NPX Cross Scan", key=f"btn_scan_npx_{mode}")

    if run_npx_scan:
        rows = []
        prog = st.progress(0)
        total = len(universe)

        for i, tkr in enumerate(universe, start=1):
            npx = daily_npx_series(tkr, ntd_win=ntd_window)
            npx = _coerce_1d_series(npx).dropna()
            if npx.empty:
                prog.progress(int(i / total * 100))
                continue

            t_up, t_dn = _last_level_cross(npx, level=float(level))
            recent_up = _within_last_bars(npx.index, t_up, bars=int(last_bars))
            recent_dn = _within_last_bars(npx.index, t_dn, bars=int(last_bars))

            if (include_up and recent_up) or (include_dn and recent_dn):
                snap = daily_snapshot(tkr, slope_lb=slope_lb_daily, ntd_win=ntd_window, sr_lb=sr_lb_daily)

                # Determine which was most recent (if both)
                which = None
                t_last = None
                if recent_up and recent_dn:
                    which = "UP" if t_up >= t_dn else "DOWN"
                    t_last = t_up if which == "UP" else t_dn
                elif recent_up:
                    which = "UP"
                    t_last = t_up
                elif recent_dn:
                    which = "DOWN"
                    t_last = t_dn

                rows.append({
                    "Ticker": tkr,
                    "Cross": which,
                    "CrossTime": t_last,
                    "Last": snap.get("last", np.nan),
                    "NTD": snap.get("ntd", np.nan),
                    "NPX": snap.get("npx", np.nan),
                    "Slope": snap.get("slope", np.nan),
                    "R²": snap.get("r2", np.nan),
                })

            prog.progress(int(i / total * 100))

        prog.empty()

        if not rows:
            st.info("No tickers matched.")
        else:
            df_out = pd.DataFrame(rows)
            df_out = df_out.sort_values(["CrossTime", "R²"], ascending=[False, False]).reset_index(drop=True)
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download CSV",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name=f"npx_cross_scanner_{mode.lower()}.csv",
                mime="text/csv",
                key=f"dl_npx_{mode}",
            )

# End of BATCH 2/3.
# Type "continue" for BATCH 3/3 (Tabs 9–19 + remaining app code).
# =========================
# BATCH 3/3 — bullbear.py (CONTINUATION)
# Tabs 9–19 + remaining app code
# =========================

# -------------------------------------------------
# (Optional) Intraday fetch helper (safe fallback)
# -------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday(ticker: str, interval: str = "5m", period: str = "5d") -> pd.Series:
    """
    Returns 1D intraday close series (DatetimeIndex -> float).
    Uses yfinance if available; returns empty series on failure.
    """
    try:
        import yfinance as yf  # local import to avoid hard dependency earlier
        df = yf.Ticker(ticker).history(interval=interval, period=period)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if "Close" in df.columns:
            s = df["Close"].copy()
        else:
            # best-effort: first numeric column
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            s = df[numcols[0]].copy() if numcols else pd.Series(dtype=float)
        s.index = pd.to_datetime(s.index)
        s = _coerce_1d_series(s).dropna()
        s.name = "Close"
        return s
    except Exception:
        return pd.Series(dtype=float)

# -------------------------------------------------
# TAB 9: Intraday View
# -------------------------------------------------
with tab9:
    st.header("Intraday View")
    st.caption("Intraday close + quick regression band. Intended for quick timing/context, not as a primary signal.")

    it_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"it_ticker_{mode}")
    colA, colB, colC = st.columns(3)
    with colA:
        it_interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
                                   index=2, key=f"it_interval_{mode}")
    with colB:
        it_period = st.selectbox("Period", ["1d", "2d", "5d", "7d", "1mo"],
                                 index=2, key=f"it_period_{mode}")
    with colC:
        it_reg_lb = st.slider("Regression lookback (bars)", 20, 400, 120, 10, key=f"it_reg_lb_{mode}")

    show_band = st.checkbox("Show ±2σ band", value=True, key=f"it_show_band_{mode}")
    show_trend = st.checkbox("Show global trend line", value=True, key=f"it_show_trend_{mode}")

    s_intra = fetch_intraday(it_tkr, interval=it_interval, period=it_period)
    s_intra = _coerce_1d_series(s_intra).dropna()

    if s_intra.empty or len(s_intra) < 10:
        st.warning("No intraday data available (or too few bars). Try a different interval/period.")
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{it_tkr} Intraday ({it_interval}, {it_period})")

        ax.plot(s_intra.index, s_intra.values, label="Close")

        if show_trend:
            _ = draw_trend_direction_line(ax, s_intra, label_prefix="Trend")

        yhat, up, lo, m, r2 = regression_with_band(s_intra, lookback=int(it_reg_lb), z=2.0)
        if not yhat.empty:
            ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Reg slope={fmt_slope(m)}/bar")

        if show_band and (not up.empty) and (not lo.empty):
            ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.80, label="+2σ")
            ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.80, label="-2σ")

        ax.text(
            0.50, 0.02,
            f"R²: {fmt_r2(r2)} | bars: {len(s_intra)} | last: {s_intra.iloc[-1]:.4g}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# -------------------------------------------------
# TAB 10: Support/Resistance (Daily/Weekly)
# -------------------------------------------------
with tab10:
    st.header("Support / Resistance")
    st.caption("Simple rolling S/R levels and distance-to-levels.")

    sr_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"sr_ticker_{mode}")
    sr_win = st.slider("Rolling window (days)", 10, 400, int(sr_lb_daily), 5, key=f"sr_win_{mode}")

    try:
        s = fetch_hist(sr_tkr)
    except Exception:
        s = pd.Series(dtype=float)

    s = _coerce_1d_series(s).dropna()
    if s.empty:
        st.warning("No price history available.")
    else:
        res = s.rolling(int(sr_win), min_periods=1).max()
        sup = s.rolling(int(sr_win), min_periods=1).min()

        last = float(s.iloc[-1])
        last_res = float(res.iloc[-1])
        last_sup = float(sup.iloc[-1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Last", f"{last:.4g}")
        col2.metric("Resistance", f"{last_res:.4g}", f"{((last / last_res) - 1) * 100:+.2f}%")
        col3.metric("Support", f"{last_sup:.4g}", f"{((last / last_sup) - 1) * 100:+.2f}%")

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{sr_tkr} Daily Close + Rolling S/R ({sr_win})")
        ax.plot(s.index, s.values, label="Close")
        ax.plot(res.index, res.values, "--", linewidth=2.0, color="black", alpha=0.80, label="Rolling Resistance")
        ax.plot(sup.index, sup.values, "--", linewidth=2.0, color="black", alpha=0.80, label="Rolling Support")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# -------------------------------------------------
# TAB 11: Normalized Metrics (Daily)
# -------------------------------------------------
with tab11:
    st.header("Normalized Metrics")
    st.caption("Daily NPX and NTD series with quick cross-level markers.")

    nm_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"nm_ticker_{mode}")

    try:
        s = fetch_hist(nm_tkr)
    except Exception:
        s = pd.Series(dtype=float)

    s = _coerce_1d_series(s).dropna()
    if s.empty:
        st.warning("No history available.")
    else:
        npx = compute_normalized_price(s, window=ntd_window)
        ntd = compute_normalized_trend(s, window=ntd_window)

        lvl = st.slider("Marker level", -1.0, 1.0, 0.50, 0.01, key=f"nm_lvl_{mode}")
        show_marks = st.checkbox("Show last cross times", value=True, key=f"nm_marks_{mode}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Last", f"{float(s.iloc[-1]):.4g}")
        col2.metric("NPX", f"{float(npx.dropna().iloc[-1]) if not npx.dropna().empty else np.nan:.3f}")
        col3.metric("NTD", f"{float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan:.3f}")

        if show_marks:
            t_up, t_dn = _last_level_cross(npx.dropna(), float(lvl))
            c1, c2 = st.columns(2)
            c1.write(f"**Last NPX cross UP {lvl:+.2f}:** {t_up if t_up else '—'}")
            c2.write(f"**Last NPX cross DOWN {lvl:+.2f}:** {t_dn if t_dn else '—'}")

        fig1, ax1 = plt.subplots(figsize=(14, 4))
        fig1.subplots_adjust(bottom=0.25)
        ax1.set_title(f"{nm_tkr} NPX (window={ntd_window})")
        ax1.plot(npx.index, npx.values, label="NPX")
        ax1.axhline(float(lvl), linestyle="--", color="black", alpha=0.75)
        ax1.axhline(0.0, linestyle=":", color="black", alpha=0.6)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax1)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(14, 4))
        fig2.subplots_adjust(bottom=0.25)
        ax2.set_title(f"{nm_tkr} NTD (window={ntd_window})")
        ax2.plot(ntd.index, ntd.values, label="NTD")
        ax2.axhline(-0.75, linestyle="--", color="black", alpha=0.75)
        ax2.axhline(0.0, linestyle=":", color="black", alpha=0.6)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax2)
        st.pyplot(fig2)

# -------------------------------------------------
# TAB 12: Regression Diagnostics (Daily)
# -------------------------------------------------
with tab12:
    st.header("Regression Diagnostics")
    st.caption("Inspect the regression fit (R²) across a range of lookbacks.")

    rd_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"rd_ticker_{mode}")
    lb_min = st.slider("Min lookback", 20, 300, 60, 5, key=f"rd_lb_min_{mode}")
    lb_max = st.slider("Max lookback", 30, 800, 252, 10, key=f"rd_lb_max_{mode}")
    step = st.slider("Step", 5, 50, 10, 1, key=f"rd_step_{mode}")

    try:
        s = fetch_hist(rd_tkr)
    except Exception:
        s = pd.Series(dtype=float)

    s = _coerce_1d_series(s).dropna()
    if s.empty or len(s) < lb_min + 5:
        st.warning("Not enough history for diagnostics.")
    else:
        lbs = list(range(int(lb_min), int(lb_max) + 1, int(step)))
        rows = []
        for lb in lbs:
            try:
                _, _, _, m, r2 = regression_with_band(s, lookback=int(lb), z=2.0)
                rows.append({"lookback": lb, "slope": float(m), "r2": float(r2)})
            except Exception:
                rows.append({"lookback": lb, "slope": np.nan, "r2": np.nan})
        df = pd.DataFrame(rows)

        st.dataframe(df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(bottom=0.25)
        ax.set_title(f"{rd_tkr} R² vs Lookback")
        ax.plot(df["lookback"], df["r2"], label="R²")
        ax.axhline(0.0, linestyle=":", color="black", alpha=0.6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, framealpha=0.65, fontsize=9, fancybox=True)
        ax.set_xlabel("Lookback (bars)")
        ax.set_ylabel("R²")
        style_axes(ax)
        st.pyplot(fig)

# -------------------------------------------------
# TAB 13: News (FIXED — uses fetch_yf_news wrapper)
# -------------------------------------------------
with tab13:
    st.header("News")
    st.caption("Yahoo Finance news via yfinance (best-effort). If unavailable, shows an empty set gracefully.")

    news_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                            key=f"news_ticker_{mode}")
    news_window_days = st.slider("News lookback (days)", 1, 30, int(st.session_state.get("news_window_days", 7)), 1,
                                 key=f"news_window_{mode}")
    st.session_state.news_window_days = news_window_days

    # ---- FIX: fetch_yf_news is defined earlier in the file (Batch 1)
    # and returns a list[dict] with time filtering. This prevents NameError.
    fx_news = fetch_yf_news(news_tkr, window_days=news_window_days)

    if not fx_news:
        st.info("No recent news items found (or news provider unavailable).")
    else:
        # Render a compact list
        for i, item in enumerate(fx_news[:50], start=1):
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            pub = item.get("publisher", item.get("source", ""))
            t = item.get("providerPublishTime", item.get("time", None))

            # Normalize time display
            t_disp = "—"
            try:
                if t is not None:
                    t_disp = pd.to_datetime(t, unit="s", utc=True).tz_convert(None).strftime("%Y-%m-%d %H:%M")
            except Exception:
                try:
                    t_disp = pd.to_datetime(t).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    t_disp = str(t) if t is not None else "—"

            with st.expander(f"{i}. {title}", expanded=(i <= 3)):
                st.write(f"**Publisher:** {pub if pub else '—'}")
                st.write(f"**Time:** {t_disp}")
                if link:
                    st.write(link)
                summ = item.get("summary") or item.get("content", "")
                if summ:
                    st.write(summ)

# -------------------------------------------------
# TAB 14: Fundamentals (Best-effort from yfinance.info)
# -------------------------------------------------
with tab14:
    st.header("Fundamentals")
    st.caption("Best-effort fundamentals from Yahoo via yfinance. Availability varies by ticker.")

    f_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                         key=f"fund_ticker_{mode}")

    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_fundamentals(ticker: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            # Reduce noise: keep common keys if present
            keys = [
                "shortName", "longName", "sector", "industry",
                "marketCap", "currency", "exchange", "quoteType",
                "trailingPE", "forwardPE", "pegRatio",
                "priceToBook", "beta",
                "dividendYield", "payoutRatio",
                "profitMargins", "operatingMargins",
                "revenueGrowth", "earningsGrowth",
                "totalRevenue", "grossProfits",
                "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
                "regularMarketPrice",
            ]
            out = {k: info.get(k, None) for k in keys if k in info}
            # Add lastUpdateTime if exists
            if "regularMarketTime" in info:
                out["regularMarketTime"] = info.get("regularMarketTime")
            return out
        except Exception:
            return {}

    f = fetch_fundamentals(f_tkr)
    if not f:
        st.info("Fundamentals unavailable for this ticker.")
    else:
        # Pretty print with light formatting
        df_f = pd.DataFrame([f]).T.reset_index()
        df_f.columns = ["Field", "Value"]
        st.dataframe(df_f, use_container_width=True)

# -------------------------------------------------
# TAB 15: Watchlist / Universe Tools
# -------------------------------------------------
with tab15:
    st.header("Watchlist / Universe Tools")
    st.caption("Quick tools for the current universe list. Export, search, and quick snapshots.")

    st.write(f"**Universe size:** {len(universe)}")
    q = st.text_input("Search tickers (substring)", value="", key=f"uni_search_{mode}")
    filtered = [t for t in universe if q.upper() in t.upper()] if q else universe

    st.write("**Tickers:**")
    st.code(", ".join(filtered[:200]) + (" ..." if len(filtered) > 200 else ""))

    if st.button("Export Universe (CSV)", key=f"btn_uni_csv_{mode}"):
        dfu = pd.DataFrame({"Ticker": universe})
        st.download_button(
            "Download universe.csv",
            data=dfu.to_csv(index=False).encode("utf-8"),
            file_name="universe.csv",
            mime="text/csv",
            key=f"dl_uni_{mode}",
        )

    st.divider()
    st.subheader("Quick Snapshot Table")
    snap_n = st.slider("How many tickers (top of list)", 5, min(200, len(universe)), min(50, len(universe)), 5,
                       key=f"uni_snap_n_{mode}")
    if st.button("Build Snapshot", key=f"btn_uni_snap_{mode}"):
        rows = []
        prog = st.progress(0)
        for i, tkr in enumerate(universe[:snap_n], start=1):
            snap = daily_snapshot(tkr, slope_lb=slope_lb_daily, ntd_win=ntd_window, sr_lb=sr_lb_daily)
            rows.append({
                "Ticker": tkr,
                "Last": snap.get("last", np.nan),
                "NTD": snap.get("ntd", np.nan),
                "NPX": snap.get("npx", np.nan),
                "Slope": snap.get("slope", np.nan),
                "R²": snap.get("r2", np.nan),
            })
            prog.progress(int(i / snap_n * 100))
        prog.empty()
        df = pd.DataFrame(rows).sort_values(["NTD", "R²"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

# -------------------------------------------------
# TAB 16: Simple Backtest (Daily)
# -------------------------------------------------
with tab16:
    st.header("Simple Backtest")
    st.caption(
        "A minimal, transparent backtest to sanity-check a single signal idea. "
        "Not intended as financial advice."
    )

    bt_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"bt_ticker_{mode}")
    bt_lb = st.slider("Regression lookback (bars)", 30, 500, int(slope_lb_daily), 5, key=f"bt_lb_{mode}")
    bt_z = st.slider("Band z (σ)", 1.0, 3.0, 2.0, 0.1, key=f"bt_z_{mode}")
    hold_days = st.slider("Hold days (after BUY)", 1, 60, 10, 1, key=f"bt_hold_{mode}")

    bt_run = st.button("Run Backtest", key=f"btn_bt_{mode}")

    if bt_run:
        try:
            s = fetch_hist(bt_tkr)
        except Exception:
            s = pd.Series(dtype=float)

        s = _coerce_1d_series(s).dropna()
        if len(s) < bt_lb + hold_days + 5:
            st.warning("Not enough history to run the backtest with these settings.")
        else:
            yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(bt_lb), z=float(bt_z))
            if yhat.empty or up.empty or lo.empty:
                st.warning("Regression computation failed (insufficient data).")
            else:
                # Align
                ss = s.reindex(yhat.index).dropna()
                yhat = yhat.reindex(ss.index)
                up = up.reindex(ss.index)
                lo = lo.reindex(ss.index)

                # BUY signal = touches/breaches lower band and slope > 0, then "bounce" (next bar up)
                # We'll implement a conservative version:
                # condition at t-1: close <= lower band AND slope>0, and at t: close > close[t-1]
                slope_ok = float(m) > 0 if np.isfinite(m) else False

                if not slope_ok:
                    st.info("Slope <= 0: this backtest requires positive slope. Adjust settings or interpret accordingly.")

                close = ss
                cond_prev = (close.shift(1) <= lo.shift(1))
                bounce = (close > close.shift(1))
                buy = cond_prev & bounce & slope_ok

                buy_times = buy[buy].index.tolist()

                # Compute forward returns over hold_days
                rets = []
                for t in buy_times:
                    try:
                        i = close.index.get_loc(t)
                        j = i + int(hold_days)
                        if j < len(close):
                            entry = float(close.iloc[i])
                            exitp = float(close.iloc[j])
                            rets.append({
                                "time": t,
                                "entry": entry,
                                "exit": exitp,
                                "ret": (exitp / entry) - 1.0
                            })
                    except Exception:
                        continue

                if not rets:
                    st.info("No trades found for these settings.")
                else:
                    df = pd.DataFrame(rets)
                    st.dataframe(df, use_container_width=True)

                    avg = float(df["ret"].mean())
                    med = float(df["ret"].median())
                    win = float((df["ret"] > 0).mean()) * 100.0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trades", f"{len(df)}")
                    c2.metric("Win rate", f"{win:.1f}%")
                    c3.metric("Avg return", f"{avg*100:.2f}%")

                    fig, ax = plt.subplots(figsize=(14, 4))
                    fig.subplots_adjust(bottom=0.25)
                    ax.set_title(f"{bt_tkr} Backtest Returns (hold={hold_days}d)")
                    ax.plot(df["time"], df["ret"].values, label="Trade return")
                    ax.axhline(0.0, linestyle=":", color="black", alpha=0.6)
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, framealpha=0.65, fontsize=9, fancybox=True)
                    style_axes(ax)
                    st.pyplot(fig)

                    st.write(f"Median return: **{med*100:.2f}%** | R² (fit): **{fmt_r2(r2)}**")

# -------------------------------------------------
# TAB 17: Alerts (UI-only placeholders)
# -------------------------------------------------
with tab17:
    st.header("Alerts")
    st.caption(
        "Streamlit apps can’t run background alerts reliably without external scheduling. "
        "This tab provides exportable watch conditions you can run elsewhere (or via Streamlit Cloud + cron)."
    )

    al_tkr = st.selectbox("Ticker", universe, index=(universe.index(disp_ticker) if disp_ticker in universe else 0),
                          key=f"al_ticker_{mode}")
    al_npx = st.slider("NPX cross level", -1.0, 1.0, 0.50, 0.01, key=f"al_npx_level_{mode}")
    al_ntd = st.slider("NTD threshold", -1.0, 0.0, -0.75, 0.01, key=f"al_ntd_level_{mode}")

    st.write("Suggested alert conditions:")
    st.code(
        f"""
Ticker: {al_tkr}
Condition A (momentum up): NPX crosses above {al_npx:+.2f}
Condition B (deep pullback): NTD <= {al_ntd:+.2f}
Condition C (buy bounce): touches lower 2σ band AND slope>0 AND next bar up
""".strip()
    )

# -------------------------------------------------
# TAB 18: Settings / Diagnostics
# -------------------------------------------------
with tab18:
    st.header("Settings / Diagnostics")
    st.caption("Cache, environment, and quick sanity checks.")

    st.write("**Mode:**", mode)
    st.write("**Selected ticker:**", disp_ticker)
    st.write("**Universe size:**", len(universe))

    st.subheader("Cache")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear st.cache_data", key=f"btn_clear_cache_data_{mode}"):
            st.cache_data.clear()
            st.success("st.cache_data cleared.")
    with c2:
        if st.button("Clear st.cache_resource", key=f"btn_clear_cache_res_{mode}"):
            st.cache_resource.clear()
            st.success("st.cache_resource cleared.")

    st.subheader("Quick checks")
    try:
        _ = fetch_hist(disp_ticker)
        st.success("fetch_hist OK")
    except Exception as e:
        st.error(f"fetch_hist failed: {e}")

    try:
        _ = fetch_yf_news(disp_ticker, window_days=1)
        st.success("fetch_yf_news OK")
    except Exception as e:
        st.error(f"fetch_yf_news failed: {e}")

# -------------------------------------------------
# TAB 19: About / Help
# -------------------------------------------------
with tab19:
    st.header("About / Help")
    st.markdown(
        """
**BullBear / Stock Wizard**

This app visualizes:
- Daily close + trend direction (simple line slope)
- Regression fit and ±2σ bands
- Normalized metrics (NPX/NTD)
- Simple scanners (NTD threshold, recent BUY bounces, NPX cross)

**Important**
- This is research tooling, not financial advice.
- Data availability depends on Yahoo Finance / yfinance and may be delayed or incomplete.
- Intraday data is best-effort and can fail for some tickers/intervals.

**Troubleshooting**
- If you see a generic Streamlit Cloud error, open app logs.
- Use the **Settings / Diagnostics** tab to clear caches.
"""
    )

st.caption("© BullBear — research tooling. Use at your own risk.")

# End of BATCH 3/3 (end of file).
