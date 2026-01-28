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

# ---------------------------
# Cached last values for scanning
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_ntd_value(symbol: str, ntd_win: int):
    try:
        s = fetch_hist(symbol)
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None

@st.cache_data(ttl=120)
def last_hourly_ntd_value(symbol: str, ntd_win: int, period: str = "1d"):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df:
            return np.nan, None
        s = df["Close"].ffill()
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None

# ---------------------------
# Cached last NPX values for scanning (Daily/Hourly)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_npx_value(symbol: str, ntd_win: int):
    try:
        s = fetch_hist(symbol)
        npx = compute_normalized_price(s, window=ntd_win).dropna()
        if npx.empty:
            return np.nan, None
        return float(npx.iloc[-1]), npx.index[-1]
    except Exception:
        return np.nan, None

@st.cache_data(ttl=120)
def last_hourly_npx_value(symbol: str, ntd_win: int, period: str = "1d"):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df:
            return np.nan, None
        s = df["Close"].ffill()
        npx = compute_normalized_price(s, window=ntd_win).dropna()
        if npx.empty:
            return np.nan, None
        return float(npx.iloc[-1]), npx.index[-1]
    except Exception:
        return np.nan, None


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Recent BUY scanner helpers (uses SAME band-bounce logic as the chart)
# ---------------------------
@st.cache_data(ttl=120)
def last_band_bounce_signal_daily(symbol: str, slope_lb: int):
    try:
        s = fetch_hist(symbol)
        p_full = _coerce_1d_series(s).dropna()
        if p_full.empty:
            return None

        yhat, up, lo, m, r2 = regression_with_band(p_full, lookback=int(slope_lb))
        sig = find_band_bounce_signal(p_full, up, lo, m)
        if sig is None:
            return None

        t = sig.get("time", None)
        if t is None or t not in p_full.index:
            return None

        loc = int(p_full.index.get_loc(t))
        bars_since = int((len(p_full) - 1) - loc)

        curr = float(p_full.iloc[-1]) if np.isfinite(p_full.iloc[-1]) else np.nan
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
        if bar is None:
            return None
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
def last_daily_npx_cross_up_in_uptrend(symbol: str, ntd_win: int, daily_view_label: str):
    try:
        s_full = fetch_hist(symbol)
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 2:
            return None

        x = np.arange(len(close_show), dtype=float)
        m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) <= 0.0:
            return None

        ntd_full = compute_normalized_trend(close_full, window=ntd_win)
        npx_full = compute_normalized_price(close_full, window=ntd_win)

        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        cross_up, _ = _cross_series(npx_show, ntd_show)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)
        if not cross_up.any():
            return None

        t = cross_up[cross_up].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        ntd_at = float(ntd_show.loc[t]) if (t in ntd_show.index and np.isfinite(ntd_show.loc[t])) else np.nan
        npx_at = float(npx_show.loc[t]) if (t in npx_show.index and np.isfinite(npx_show.loc[t])) else np.nan

        ntd_last = float(ntd_show.dropna().iloc[-1]) if len(ntd_show.dropna()) else np.nan
        npx_last = float(npx_show.dropna().iloc[-1]) if len(npx_show.dropna()) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Signal": "NPX↑NTD (Uptrend)",
            "Bars Since": bars_since,
            "Cross Time": t,
            "Global Slope": float(m),
            "Current Price": curr_px,
            "NTD@Cross": ntd_at,
            "NPX@Cross": npx_at,
            "NTD (last)": ntd_last,
            "NPX (last)": npx_last,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_daily_npx_zero_cross_with_local_slope(symbol: str,
                                               ntd_win: int,
                                               daily_view_label: str,
                                               local_slope_lb: int,
                                               max_abs_npx_at_cross: float,
                                               direction: str = "up"):
    try:
        s_full = fetch_hist(symbol)
        close_full = _coerce_1d_series(s_full).dropna()
        if close_full.empty:
            return None

        close_show = subset_by_daily_view(close_full, daily_view_label)
        close_show = _coerce_1d_series(close_show).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        # NOTE: uses 0.5 cross level
        level = 0.5

        prev = npx_show.shift(1)
        if str(direction).lower().startswith("up"):
            cross_mask = (npx_show >= level) & (prev < level)
            sig_label = "NPX 0.5↑"
        else:
            cross_mask = (npx_show <= level) & (prev > level)
            sig_label = "NPX 0.5↓"

        cross_mask = cross_mask.fillna(False)
        if not cross_mask.any():
            return None

        eps = float(max_abs_npx_at_cross)
        near_level = ((npx_show - level).abs() <= eps) & ((prev - level).abs() <= eps)
        cross_mask = cross_mask & near_level.fillna(False)
        if not cross_mask.any():
            return None

        t = cross_mask[cross_mask].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        seg = close_show.loc[:t].tail(int(local_slope_lb))
        seg = _coerce_1d_series(seg).dropna()
        if len(seg) < 2:
            return None
        x = np.arange(len(seg), dtype=float)
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        if not np.isfinite(m) or float(m) == 0.0:
            return None

        if sig_label.endswith("↑") and float(m) <= 0.0:
            return None
        if sig_label.endswith("↓") and float(m) >= 0.0:
            return None

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        npx_at = float(npx_show.loc[t]) if (t in npx_show.index and np.isfinite(npx_show.loc[t])) else np.nan
        npx_prev = float(prev.loc[t]) if (t in prev.index and np.isfinite(prev.loc[t])) else np.nan
        npx_last = float(npx_show.dropna().iloc[-1]) if len(npx_show.dropna()) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Daily View": daily_view_label,
            "Signal": sig_label,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Local Slope": float(m),
            "Current Price": curr_px,
            "NPX@Cross": npx_at,
            "NPX(prev)": npx_prev,
            "NPX (last)": npx_last,
            "Zero-Eps": float(eps),
            "Slope LB": int(local_slope_lb),
        }
    except Exception:
        return None

# ---------------------------
# Fib 0%/100% + NPX 0.0-cross (Up/Down) helpers
# ---------------------------
def _npx_zero_cross_masks(npx: pd.Series):
    s = _coerce_1d_series(npx)
    prev = s.shift(1)
    cross_up0 = (s >= 0.0) & (prev < 0.0)
    cross_dn0 = (s <= 0.0) & (prev > 0.0)
    return cross_up0.fillna(False), cross_dn0.fillna(False)

def _fib_npx_zero_signal_series(close: pd.Series,
                                npx: pd.Series,
                                prox: float,
                                lookback_bars: int,
                                slope_lb: int,
                                npx_confirm_bars: int = 1):
    c = _coerce_1d_series(close).dropna()
    if c.empty or len(c) < 3:
        return None

    fibs = fibonacci_levels(c)
    if not fibs:
        return None

    fib0 = float(fibs.get("0%", np.nan))
    fib100 = float(fibs.get("100%", np.nan))
    if not (np.isfinite(fib0) and np.isfinite(fib100)):
        return None

    npx_s = _coerce_1d_series(npx).reindex(c.index)
    if npx_s.dropna().empty:
        return None

    lb = max(2, int(lookback_bars))
    c_lb = c.iloc[-lb:] if len(c) > lb else c
    npx_lb = npx_s.reindex(c_lb.index)

    cross_up0, cross_dn0 = _npx_zero_cross_masks(npx_lb)

    touch_lo = c_lb <= (fib100 * (1.0 + float(prox)))
    touch_hi = c_lb >= (fib0   * (1.0 - float(prox)))

    slope_lb = max(2, int(slope_lb))
    npx_confirm_bars = max(1, int(npx_confirm_bars))

    def _slope(seg: pd.Series) -> float:
        seg = _coerce_1d_series(seg).dropna()
        if len(seg) < 2:
            return np.nan
        x = np.arange(len(seg), dtype=float)
        m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
        return float(m) if np.isfinite(m) else np.nan

    m_now = _slope(c.tail(slope_lb))

    def _npx_direction_ok(t_cross, want_up: bool) -> bool:
        if t_cross is None or t_cross not in npx_s.index:
            return False
        post = npx_s.loc[t_cross:]
        post = _coerce_1d_series(post).dropna()
        if len(post) < (npx_confirm_bars + 1):
            return False
        deltas = post.diff().dropna()
        if deltas.empty:
            return False
        last_d = deltas.iloc[-npx_confirm_bars:]
        if want_up:
            return bool(np.all(last_d > 0) and (float(post.iloc[-1]) > float(post.iloc[0])))
        return bool(np.all(last_d < 0) and (float(post.iloc[-1]) < float(post.iloc[0])))

    buy = None
    if cross_up0.any() and touch_lo.any():
        t_cross = cross_up0[cross_up0].index[-1]
        touch_before = touch_lo.loc[:t_cross]
        if touch_before.any():
            t_touch = touch_before[touch_before].index[-1]
            m_touch = _slope(c.loc[:t_touch].tail(slope_lb))

            slope_ok = (np.isfinite(m_touch) and np.isfinite(m_now) and (float(m_touch) < 0.0) and (float(m_now) > 0.0))
            npx_ok = _npx_direction_ok(t_cross, want_up=True)

            if slope_ok and npx_ok:
                px = float(c.loc[t_cross]) if (t_cross in c.index and np.isfinite(c.loc[t_cross])) else np.nan
                buy = {
                    "side": "BUY",
                    "time": t_cross,
                    "price": px,
                    "touch_time": t_touch,
                    "fib_level": "100%",
                    "fib_price": fib100,
                    "npx_at_cross": float(npx_s.loc[t_cross]) if (t_cross in npx_s.index and np.isfinite(npx_s.loc[t_cross])) else np.nan,
                    "slope_touch": float(m_touch),
                    "slope_now": float(m_now),
                }

    sell = None
    if cross_dn0.any() and touch_hi.any():
        t_cross = cross_dn0[cross_dn0].index[-1]
        touch_before = touch_hi.loc[:t_cross]
        if touch_before.any():
            t_touch = touch_before[touch_before].index[-1]
            m_touch = _slope(c.loc[:t_touch].tail(slope_lb))

            slope_ok = (np.isfinite(m_touch) and np.isfinite(m_now) and (float(m_touch) > 0.0) and (float(m_now) < 0.0))
            npx_ok = _npx_direction_ok(t_cross, want_up=False)

            if slope_ok and npx_ok:
                px = float(c.loc[t_cross]) if (t_cross in c.index and np.isfinite(c.loc[t_cross])) else np.nan
                sell = {
                    "side": "SELL",
                    "time": t_cross,
                    "price": px,
                    "touch_time": t_touch,
                    "fib_level": "0%",
                    "fib_price": fib0,
                    "npx_at_cross": float(npx_s.loc[t_cross]) if (t_cross in npx_s.index and np.isfinite(npx_s.loc[t_cross])) else np.nan,
                    "slope_touch": float(m_touch),
                    "slope_now": float(m_now),
                }

    if buy is None and sell is None:
        return None
    if buy is None:
        return sell
    if sell is None:
        return buy
    return buy if buy["time"] >= sell["time"] else sell

def annotate_fib_npx_signal(ax, sig: dict):
    if not isinstance(sig, dict):
        return
    side = str(sig.get("side", "")).upper()
    t = sig.get("time", None)
    px = sig.get("price", np.nan)
    if t is None or (not np.isfinite(px)):
        return

    col = "tab:green" if side.startswith("B") else "tab:red"
    marker = "^" if side.startswith("B") else "v"
    label = "Fib BUY" if side.startswith("B") else "Fib SELL"

    ax.scatter([t], [px], marker=marker, s=110, color=col, zorder=12)
    ax.text(
        t, px,
        f"  {label}",
        color=col,
        fontsize=9,
        fontweight="bold",
        va="bottom" if side.startswith("B") else "top",
        zorder=12
    )

@st.cache_data(ttl=120)
def last_daily_fib_npx_zero_signal(symbol: str,
                                  daily_view_label: str,
                                  ntd_win: int,
                                  direction: str,
                                  prox: float,
                                  lookback_bars: int,
                                  slope_lb: int,
                                  npx_confirm_bars: int = 1):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        fibs = fibonacci_levels(close_show)
        if not fibs:
            return None
        fib0 = float(fibs.get("0%", np.nan))
        fib100 = float(fibs.get("100%", np.nan))
        if not (np.isfinite(fib0) and np.isfinite(fib100)):
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)
        if npx_show.dropna().empty:
            return None

        lb = max(2, int(lookback_bars))
        c_lb = close_show.iloc[-lb:] if len(close_show) > lb else close_show
        npx_lb = npx_show.reindex(c_lb.index)

        cross_up0, cross_dn0 = _npx_zero_cross_masks(npx_lb)

        slope_lb = max(2, int(slope_lb))
        npx_confirm_bars = max(1, int(npx_confirm_bars))

        def _slope(seg: pd.Series) -> float:
            seg = _coerce_1d_series(seg).dropna()
            if len(seg) < 2:
                return np.nan
            x = np.arange(len(seg), dtype=float)
            m, b = np.polyfit(x, seg.to_numpy(dtype=float), 1)
            return float(m) if np.isfinite(m) else np.nan

        m_now = _slope(close_show.tail(slope_lb))

        def _npx_dir_ok(t_cross, want_up: bool) -> bool:
            if t_cross is None or t_cross not in npx_show.index:
                return False
            post = _coerce_1d_series(npx_show.loc[t_cross:]).dropna()
            if len(post) < (npx_confirm_bars + 1):
                return False
            d = post.diff().dropna()
            if d.empty:
                return False
            last_d = d.iloc[-npx_confirm_bars:]
            if want_up:
                return bool(np.all(last_d > 0) and (float(post.iloc[-1]) > float(post.iloc[0])))
            return bool(np.all(last_d < 0) and (float(post.iloc[-1]) < float(post.iloc[0])))

        want_buy = str(direction).lower().startswith(("b", "u"))
        if want_buy:
            if not cross_up0.any():
                return None
            t_cross = cross_up0[cross_up0].index[-1]
            touch_mask = c_lb <= (fib100 * (1.0 + float(prox)))
            touch_before = touch_mask.loc[:t_cross]
            if not touch_before.any():
                return None
            t_touch = touch_before[touch_before].index[-1]
            m_touch = _slope(close_show.loc[:t_touch].tail(slope_lb))

            slope_ok = (np.isfinite(m_touch) and np.isfinite(m_now) and (float(m_touch) < 0.0) and (float(m_now) > 0.0))
            npx_ok = _npx_dir_ok(t_cross, want_up=True)
            if not (slope_ok and npx_ok):
                return None

            side = "BUY"
            fib_level = "100%"
            fib_price = fib100
        else:
            if not cross_dn0.any():
                return None
            t_cross = cross_dn0[cross_dn0].index[-1]
            touch_mask = c_lb >= (fib0 * (1.0 - float(prox)))
            touch_before = touch_mask.loc[:t_cross]
            if not touch_before.any():
                return None
            t_touch = touch_before[touch_before].index[-1]
            m_touch = _slope(close_show.loc[:t_touch].tail(slope_lb))

            slope_ok = (np.isfinite(m_touch) and np.isfinite(m_now) and (float(m_touch) > 0.0) and (float(m_now) < 0.0))
            npx_ok = _npx_dir_ok(t_cross, want_up=False)
            if not (slope_ok and npx_ok):
                return None

            side = "SELL"
            fib_level = "0%"
            fib_price = fib0

        loc = int(close_show.index.get_loc(t_cross))
        bars_since = int((len(close_show) - 1) - loc)

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        cross_px = float(close_show.loc[t_cross]) if (t_cross in close_show.index and np.isfinite(close_show.loc[t_cross])) else np.nan
        npx_at = float(npx_show.loc[t_cross]) if (t_cross in npx_show.index and np.isfinite(npx_show.loc[t_cross])) else np.nan

        return {
            "Symbol": symbol,
            "Side": side,
            "Daily View": daily_view_label,
            "Bars Since Cross": bars_since,
            "Touch Time": t_touch,
            "Cross Time": t_cross,
            "Price@Cross": cross_px,
            "Current Price": curr_px,
            "Fib Level": fib_level,
            "Fib Price": fib_price,
            "NPX@Cross": npx_at,
            "Slope@Touch": float(m_touch) if np.isfinite(m_touch) else np.nan,
            "Slope (now)": float(m_now) if np.isfinite(m_now) else np.nan,
        }
    except Exception:
        return None

# ---------------------------
# Daily slope direction helper
# ---------------------------
@st.cache_data(ttl=120)
def daily_global_slope(symbol: str, daily_view_label: str):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, np.nan, None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 2:
            return np.nan, np.nan, None
        x = np.arange(len(close_show), dtype=float)
        y = close_show.to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
        return float(m), r2, close_show.index[-1]
    except Exception:
        return np.nan, np.nan, None

# ---------------------------
# NEW (THIS REQUEST): Daily regression slope (band slope) in selected Daily view
#   Used to print "daily slope direction" on the HOURLY chart.
# ---------------------------
@st.cache_data(ttl=120)
def daily_regression_slope_in_view(symbol: str, daily_view_label: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, np.nan, None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(3, int(slope_lb) + 2):
            return np.nan, np.nan, (close_show.index[-1] if len(close_show) else None)
        _, _, _, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        return float(m) if np.isfinite(m) else np.nan, float(r2) if np.isfinite(r2) else np.nan, close_show.index[-1]
    except Exception:
        return np.nan, np.nan, None

# ---------------------------
# Daily last NPX in selected Daily view range
# ---------------------------
@st.cache_data(ttl=120)
def daily_last_npx_in_view(symbol: str, daily_view_label: str, ntd_win: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty:
            return np.nan, None

        npx_full = compute_normalized_price(close_full, window=int(ntd_win))
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index).dropna()
        if npx_show.empty:
            return np.nan, close_show.index[-1]

        return float(npx_show.iloc[-1]), npx_show.index[-1]
    except Exception:
        return np.nan, None

# ---------------------------
# Daily NPX series in selected Daily view range (for scanners)
# ---------------------------
@st.cache_data(ttl=120)
def daily_npx_series_in_view(symbol: str, daily_view_label: str, ntd_win: int) -> pd.Series:
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return pd.Series(dtype=float)

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty:
            return pd.Series(index=close_show.index, dtype=float)

        npx_full = compute_normalized_price(close_full, window=int(ntd_win))
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)
        return npx_show
    except Exception:
        return pd.Series(dtype=float)

def _series_heading_up(series_like: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(series_like).dropna()
    confirm_bars = max(1, int(confirm_bars))
    if len(s) < confirm_bars + 1:
        return False
    d = s.diff().dropna()
    if len(d) < confirm_bars:
        return False
    last_d = d.iloc[-confirm_bars:]
    return bool(np.all(last_d > 0))

# ---------------------------
# NEW (THIS REQUEST): NPX 0.5 Marker helper (Daily)
#   - NPX recently crossed UP through +0.5
#   - Current NPX is between +0.5 and +0.7
#   - NPX slope is positive (heading up)
#   - Returns row including DAILY global trend slope (for split lists slope>0 vs slope<0)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_npx_05_marker(symbol: str,
                             daily_view_label: str,
                             ntd_win: int,
                             max_bars_since_cross: int = 5,
                             band_low: float = 0.5,
                             band_high: float = 0.7,
                             npx_slope_lb: int = 20):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        npx_full = compute_normalized_price(close_full, window=int(ntd_win))
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index).dropna()
        if npx_show.empty or len(npx_show) < 3:
            return None

        level = float(band_low)
        prev = npx_show.shift(1)
        cross_up = (npx_show >= level) & (prev < level)
        cross_up = cross_up.fillna(False)
        if not cross_up.any():
            return None

        t_cross = cross_up[cross_up].index[-1]
        loc = int(npx_show.index.get_loc(t_cross))
        bars_since = int((len(npx_show) - 1) - loc)
        if int(bars_since) > int(max_bars_since_cross):
            return None

        npx_last = float(npx_show.iloc[-1]) if np.isfinite(npx_show.iloc[-1]) else np.nan
        if not (np.isfinite(npx_last) and float(band_low) <= npx_last <= float(band_high)):
            return None

        npx_seg = npx_show.tail(int(max(3, npx_slope_lb)))
        if len(npx_seg) < 3:
            return None
        x = np.arange(len(npx_seg), dtype=float)
        m_npx, b_npx = np.polyfit(x, npx_seg.to_numpy(dtype=float), 1)
        if not (np.isfinite(m_npx) and float(m_npx) > 0.0):
            return None

        # Daily global trend slope (price)
        x2 = np.arange(len(close_show), dtype=float)
        m_px, b_px = np.polyfit(x2, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m_px):
            return None

        npx_at = float(npx_show.loc[t_cross]) if (t_cross in npx_show.index and np.isfinite(npx_show.loc[t_cross])) else np.nan
        px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Cross Time": t_cross,
            "Bars Since Cross": int(bars_since),
            "NPX@Cross": npx_at,
            "NPX (last)": float(npx_last),
            "NPX Slope": float(m_npx),
            "Daily Trend Slope": float(m_px),
            "Current Price": px_last,
            "Daily View": daily_view_label,
        }
    except Exception:
        return None

# ---------------------------
# Support reversal heading up (Daily) helper
# ---------------------------
@st.cache_data(ttl=120)
def daily_support_reversal_heading_up(symbol: str,
                                      daily_view_label: str,
                                      sr_lb: int,
                                      prox: float,
                                      bars_confirm: int,
                                      horizon: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < max(5, int(sr_lb)):
            return None

        sup = close.rolling(int(sr_lb), min_periods=1).min()

        hz = max(1, int(horizon))
        win = min(len(close), hz + 1)
        near_support = close <= (sup * (1.0 + float(prox)))

        recent_mask = near_support.iloc[-win:]
        if not recent_mask.any():
            return None

        t_touch = recent_mask[recent_mask].index[-1]
        try:
            loc_touch = int(close.index.get_loc(t_touch))
        except Exception:
            return None

        bars_since_touch = int((len(close) - 1) - loc_touch)

        seg = close.loc[t_touch:]
        seg = _coerce_1d_series(seg).dropna()
        if len(seg) < int(bars_confirm) + 1:
            return None
        if not _n_consecutive_increasing(seg, int(bars_confirm)):
            return None

        sup_seg = _coerce_1d_series(sup).reindex(seg.index).ffill()
        dist = (seg - sup_seg).iloc[-(int(bars_confirm) + 1):]
        if dist.isna().any():
            return None
        if not bool(np.all(np.diff(dist.to_numpy(dtype=float)) > 0)):
            return None

        c_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
        s_last = float(sup.iloc[-1]) if np.isfinite(sup.iloc[-1]) else np.nan
        dist_sup_pct = (c_last / s_last - 1.0) if np.isfinite(c_last) and np.isfinite(s_last) and s_last != 0 else np.nan

        return {
            "Symbol": symbol,
            "Touch Time": t_touch,
            "Bars Since Touch": bars_since_touch,
            "Close": c_last,
            "Support": s_last,
            "Dist vs Support": dist_sup_pct,
        }
    except Exception:
        return None

# ---------------------------
# NEW: Ichimoku Kijun Daily Cross-Up Scanner helper (Daily only / matches Price Chart)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_kijun_cross_up(symbol: str,
                              daily_view_label: str,
                              slope_lb: int,
                              conv: int,
                              base: int,
                              span_b: int,
                              within_last_n_bars: int = 5):
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
        ohlc = ohlc.sort_index()
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

        cross_up, _ = _cross_series(close_show, kijun)
        cross_up = cross_up.reindex(close_show.index, fill_value=False)
        if not cross_up.any():
            return None

        t_cross = cross_up[cross_up].index[-1]
        try:
            loc = int(close_show.index.get_loc(t_cross))
        except Exception:
            return None
        bars_since = int((len(close_show) - 1) - loc)

        within_last_n_bars = max(0, int(within_last_n_bars))
        if bars_since > within_last_n_bars:
            return None

        px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        px_prev = float(close_show.shift(1).loc[t_cross]) if (t_cross in close_show.index and np.isfinite(close_show.shift(1).loc[t_cross])) else np.nan
        px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        heading_up = (np.isfinite(px_cross) and np.isfinite(px_prev) and (px_cross > px_prev) and np.isfinite(px_last) and (px_last >= px_cross))
        if not heading_up:
            return None

        kij_cross = float(kijun.loc[t_cross]) if (t_cross in kijun.index and np.isfinite(kijun.loc[t_cross])) else np.nan
        if not np.isfinite(kij_cross):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None

        return {
            "Symbol": symbol,
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

# ---------------------------
# NEW: R² scanners (Daily/Hourly)
# ---------------------------
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
            return np.nan, np.nan, (real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None)

        _, _, _, m, r2 = regression_with_band(hc, lookback=int(slope_lb))
        ts = real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None
        return float(r2) if np.isfinite(r2) else np.nan, float(m) if np.isfinite(m) else np.nan, ts
    except Exception:
        return np.nan, np.nan, None

# ---------------------------
# Daily R² + ±2σ band proximity helper (for Tab 18)
# ---------------------------
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
    Returns dict (or None).
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

    # =========================
    # NEW (THIS REQUEST): Hourly chart statement showing DAILY trend + DAILY slope directions
    #   - Daily Trend = daily_global_slope() sign
    #   - Daily Slope = daily_regression_slope_in_view() sign (slope band slope)
    #   - Uses the currently-selected daily_view & slope_lb_daily (from sidebar)
    # =========================
    d_trend_m, d_trend_r2, _ = daily_global_slope(sel, daily_view_label=daily_view)
    d_slope_m, d_slope_r2, _ = daily_regression_slope_in_view(sel, daily_view_label=daily_view, slope_lb=int(slope_lb_daily))

    def _dir(mv):
        try:
            mv = float(mv)
        except Exception:
            return "n/a"
        if not np.isfinite(mv) or mv == 0.0:
            return "FLAT"
        return "UP" if mv > 0.0 else "DOWN"

    daily_note = (
        f"Daily Trend: {_dir(d_trend_m)} ({fmt_slope(d_trend_m)}) | R² {fmt_r2(d_trend_r2)}\n"
        f"Daily Slope: {_dir(d_slope_m)} ({fmt_slope(d_slope_m)}) | R² {fmt_r2(d_slope_r2)}"
    )
    ax2.text(
        0.99, 0.98, daily_note,
        transform=ax2.transAxes, ha="right", va="top",
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.80),
        zorder=40
    )

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
    fib_sig_h = _fib_npx_zero_signal_series(
        close=hc,
        npx=npx_h_for_sig,
        prox=sr_prox_pct,
        lookback_bars=int(max(3, rev_horizon)),
        slope_lb=int(slope_lb_hourly),
        npx_confirm_bars=1
    )
    if isinstance(fib_sig_h, dict):
        annotate_fib_npx_signal(ax2, fib_sig_h)

    fib_trig_chart = fib_reversal_trigger_from_extremes(
        hc,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_hourly)),
    )
    if isinstance(fib_trig_chart, dict):
        try:
            touch_bar = int(fib_trig_chart.get("touch_time"))
        except Exception:
            touch_bar = None

        m_touch = np.nan
        if touch_bar is not None and 0 <= touch_bar < len(hc):
            seg_touch = _coerce_1d_series(hc.iloc[:touch_bar+1]).dropna().tail(int(slope_lb_hourly))
            if len(seg_touch) >= 2:
                x = np.arange(len(seg_touch), dtype=float)
                mt, bt = np.polyfit(x, seg_touch.to_numpy(dtype=float), 1)
                m_touch = float(mt) if np.isfinite(mt) else np.nan

        m_now = float(m_h) if np.isfinite(m_h) else np.nan
        side_now = str(fib_trig_chart.get("side", "")).upper()
        want_up = side_now.startswith("B")
        slope_ok = (np.isfinite(m_now) and ((want_up and m_now > 0.0) or ((not want_up) and m_now < 0.0)))
        reversed_ok = (np.isfinite(m_touch) and np.isfinite(m_now)
                       and np.sign(m_touch) != 0.0 and np.sign(m_now) != 0.0
                       and np.sign(m_touch) != np.sign(m_now))

        if slope_ok and reversed_ok:
            edge = "tab:green" if want_up else "tab:red"
            ax2.text(
                0.99, 0.90, "Reverse Possible",
                transform=ax2.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=edge,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge, alpha=0.85),
                zorder=25
            )

    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_hourly})")
        ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
            ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
            if not ntd_trend_h.empty:
                ax2w.plot(ntd_trend_h.index, ntd_trend_h.values, "--", linewidth=2,
                          label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")

            overlay_ntd_triangles_by_trend(ax2w, ntd_h, trend_slope=m_h, upper=0.75, lower=-0.75)
            overlay_ntd_sr_reversal_stars(
                ax2w, price=hc, sup=sup_h, res=res_h,
                trend_slope=m_h, ntd=ntd_h, prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm
            )

        if show_ntd_channel:
            overlay_inrange_on_ntd(ax2w, price=hc, sup=sup_h, res=res_h)

        if show_npx_ntd and not _coerce_1d_series(npx_h).dropna().empty and not _coerce_1d_series(ntd_h).dropna().empty:
            overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        if show_hma_rev_ntd and not hma_h.dropna().empty and not hc.dropna().empty:
            overlay_hma_reversal_on_ntd(ax2w, hc, hma_h, lookback=hma_rev_lb,
                                        period=hma_period, ntd=ntd_h)

        ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2w.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax2w.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax2w.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax2w.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        ax2w.set_ylim(-1.1, 1.1)
        ax2w.set_xlabel("Time (PST)")
    else:
        ax2.set_xlabel("Time (PST)")

    handles, labels = [], []
    h1, l1 = ax2.get_legend_handles_labels()
    handles += h1; labels += l1
    if ax2w is not None:
        h2, l2 = ax2w.get_legend_handles_labels()
        handles += h2; labels += l2
    if session_handles and session_labels:
        handles += list(session_handles)
        labels += list(session_labels)

    seen = set()
    h_u, l_u = [], []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        h_u.append(h)
        l_u.append(l)

    fig2.legend(
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

    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax2w if ax2w is not None else ax2, real_times, n_ticks=8)

    style_axes(ax2)
    if ax2w is not None:
        style_axes(ax2w)
    xlim_price = ax2.get_xlim()
    st.pyplot(fig2)

    if show_macd and not macd_h.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        figm.subplots_adjust(top=0.88, bottom=0.45)
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.set_xlim(xlim_price)
        axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)

    trig_disp = None
    if isinstance(fib_trig_chart, dict):
        trig_disp = dict(fib_trig_chart)
        if isinstance(real_times, pd.DatetimeIndex):
            for k in ["touch_time", "last_time"]:
                try:
                    bi = int(trig_disp.get(k))
                    if 0 <= bi < len(real_times):
                        trig_disp[k] = real_times[bi]
                except Exception:
                    pass

    return {
        "trade_instruction": instr_txt,
        "fib_trigger": trig_disp,
    }
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Shared daily renderer (Stock & Forex)
# ---------------------------
def render_daily_views(sel: str, is_forex: bool):
    close_full = _coerce_1d_series(fetch_hist(sel)).dropna()
    if close_full.empty:
        st.warning("No daily data available.")
        return None

    close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
    if close.empty:
        st.warning("No daily data in selected Daily View.")
        return None

    # OHLC for Ichimoku / PSAR / Supertrend (if available)
    ohlc = fetch_hist_ohlc(sel)
    ohlc_show = None
    if isinstance(ohlc, pd.DataFrame) and (not ohlc.empty) and {"High", "Low", "Close"}.issubset(ohlc.columns):
        x0, x1 = close.index[0], close.index[-1]
        ohlc_show = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)].copy()
        if ohlc_show.empty:
            ohlc_show = None

    ema20 = close.ewm(span=20).mean()
    hma_d = compute_hma(close, period=hma_period) if show_hma else pd.Series(index=close.index, dtype=float)
    macd_d, macd_sig_d, macd_hist_d = compute_macd(close)

    res_d = close.rolling(sr_lb_daily, min_periods=1).max()
    sup_d = close.rolling(sr_lb_daily, min_periods=1).min()

    kijun_d = pd.Series(index=close.index, dtype=float)
    if show_ichi and ohlc_show is not None:
        _, kijun_d, _, _, _ = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
            shift_cloud=False
        )
        kijun_d = _coerce_1d_series(kijun_d).reindex(close.index).ffill().bfill()

    bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(
        close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    )

    st_line_d = pd.Series(index=close.index, dtype=float)
    if show_supertrend and ohlc_show is not None:
        st_df = compute_supertrend(ohlc_show, atr_period=atr_period, atr_mult=atr_mult)
        if not st_df.empty and "ST" in st_df.columns:
            st_line_d = _coerce_1d_series(st_df["ST"]).reindex(close.index).ffill().bfill()

    psar_d_df = pd.DataFrame()
    if show_psar and ohlc_show is not None:
        psar_d_df = compute_psar_from_ohlc(ohlc_show, step=psar_step, max_step=psar_max).reindex(close.index)

    yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(close, slope_lb_daily)
    slope_sig_d = m_d

    rev_prob_d = slope_reversal_probability(
        close,
        slope_sig_d,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_daily,
        horizon=rev_horizon,
    )

    # News (optional)
    news_df = pd.DataFrame()
    if show_fx_news and is_forex:
        news_df = fetch_yf_news(sel, window_days=news_window_days)

    axw = None
    if show_nrsi:
        fig, (ax, axw) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.34)
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)

    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    ax.plot(close.index, close.values, label="Daily Close")
    ax.plot(ema20.index, ema20.values, "--", label="20 EMA")

    global_m_d = draw_trend_direction_line(ax, close, label_prefix="Trend (global)")

    if show_hma and not hma_d.dropna().empty:
        ax.plot(hma_d.index, hma_d.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_d.dropna().empty:
        ax.plot(kijun_d.index, kijun_d.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_d.dropna().empty and not bb_lo_d.dropna().empty:
        ax.fill_between(close.index, bb_lo_d, bb_up_d, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid_d.index, bb_mid_d.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up_d.index, bb_up_d.values, ":", linewidth=1.0)
        ax.plot(bb_lo_d.index, bb_lo_d.values, ":", linewidth=1.0)

    if show_psar and (not psar_d_df.empty) and ("PSAR" in psar_d_df.columns):
        up_mask = psar_d_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=18, color="tab:green", zorder=6,
                       label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=18, color="tab:red", zorder=6)

    # Support/Resistance
    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_d.iloc[-1])
        sup_val = float(sup_d.iloc[-1])
        px_val  = float(close.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax.hlines(res_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax.hlines(sup_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if show_supertrend and not st_line_d.dropna().empty:
        ax.plot(st_line_d.index, st_line_d.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    # Regression band
    if not yhat_d.empty:
        ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2, label=f"Slope {slope_lb_daily} bars ({fmt_slope(m_d)}/bar)")
    if not upper_d.empty and not lower_d.empty:
        ax.plot(upper_d.index, upper_d.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax.plot(lower_d.index, lower_d.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")
        bounce_sig_d = find_band_bounce_signal(close, upper_d, lower_d, slope_sig_d)
        if bounce_sig_d is not None:
            annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

    # MACD/HMA/SR combined signal
    macd_sig = find_macd_hma_sr_signal(
        close=close, hma=hma_d, macd=macd_d, sup=sup_d, res=res_d,
        global_trend_slope=global_m_d, prox=sr_prox_pct
    )
    macd_instr_txt = "MACD/HMA55: n/a"
    if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
        side = macd_sig["side"]
        macd_instr_txt = f"MACD/HMA55: {side} @ {fmt_price_val(macd_sig['price'])}"
        annotate_macd_signal(ax, macd_sig["time"], macd_sig["price"], side)

    ax.text(
        0.01, 0.98, macd_instr_txt,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        zorder=20
    )

    # Fib overlays (optional)
    if show_fibs and not close.empty:
        fibs_d = fibonacci_levels(close)
        for lbl, y in fibs_d.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_d.items():
            ax.text(close.index[-1], y, f" {lbl}", va="center")

    # Fib + NPX zero-cross signal (chart)
    npx_d_for_sig = compute_normalized_price(close, window=ntd_window)
    fib_sig_d = _fib_npx_zero_signal_series(
        close=close,
        npx=npx_d_for_sig,
        prox=sr_prox_pct,
        lookback_bars=int(max(3, rev_horizon)),
        slope_lb=int(slope_lb_daily),
        npx_confirm_bars=1
    )
    if isinstance(fib_sig_d, dict):
        annotate_fib_npx_signal(ax, fib_sig_d)

    # Reverse Possible (Fib extremes + slope flip)
    fib_trig_d = fib_reversal_trigger_from_extremes(
        close,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_daily)),
    )
    if isinstance(fib_trig_d, dict):
        m_touch = float(fib_trig_d.get("slope_touch", np.nan))
        m_now = float(fib_trig_d.get("slope_now", np.nan))
        side_now = str(fib_trig_d.get("side", "")).upper()
        want_up = side_now.startswith("B")
        slope_ok = (np.isfinite(m_now) and ((want_up and m_now > 0.0) or ((not want_up) and m_now < 0.0)))
        reversed_ok = (np.isfinite(m_touch) and np.isfinite(m_now)
                       and np.sign(m_touch) != 0.0 and np.sign(m_now) != 0.0
                       and np.sign(m_touch) != np.sign(m_now))
        if slope_ok and reversed_ok:
            edge = "tab:green" if want_up else "tab:red"
            ax.text(
                0.99, 0.90, "Reverse Possible",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold", color=edge,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=edge, alpha=0.85),
                zorder=25
            )

    # Title + footer stats
    rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
    ax.set_title(
        f"{sel} Daily ({daily_view})  "
        f"↑{fmt_pct(float((close.pct_change().dropna() > 0).mean()) if len(close) > 5 else np.nan)}  "
        f"↓{fmt_pct(float((close.pct_change().dropna() < 0).mean()) if len(close) > 5 else np.nan)}  "
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
    )

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb_d.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb_d.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        ax.text(0.99, 0.02,
                f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax.text(0.01, 0.02,
            f"Slope: {fmt_slope(slope_sig_d)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_d)}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
    ax.text(0.50, 0.02,
            f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    # Indicator panel (NTD/NPX etc)
    if axw is not None:
        axw.set_title(f"Daily Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_daily})")

        ntd_d = compute_normalized_trend(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)
        npx_d = compute_normalized_price(close, window=ntd_window) if show_npx_ntd else pd.Series(index=close.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_d).dropna().empty:
            shade_ntd_regions(axw, ntd_d)

        if show_ntd and not _coerce_1d_series(ntd_d).dropna().empty:
            axw.plot(ntd_d.index, ntd_d.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
            ntd_trend_d, ntd_m_d = slope_line(ntd_d, slope_lb_daily)
            if not ntd_trend_d.empty:
                axw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                         label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")

            overlay_ntd_triangles_by_trend(axw, ntd_d, trend_slope=m_d, upper=0.75, lower=-0.75)
            overlay_ntd_sr_reversal_stars(
                axw, price=close, sup=sup_d, res=res_d,
                trend_slope=m_d, ntd=ntd_d, prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm
            )

        if show_ntd_channel:
            overlay_inrange_on_ntd(axw, price=close, sup=sup_d, res=res_d)

        if show_npx_ntd and not _coerce_1d_series(npx_d).dropna().empty and not _coerce_1d_series(ntd_d).dropna().empty:
            overlay_npx_on_ntd(axw, npx_d, ntd_d, mark_crosses=mark_npx_cross)

        if show_hma_rev_ntd and show_hma and not hma_d.dropna().empty:
            overlay_hma_reversal_on_ntd(axw, close, hma_d, lookback=hma_rev_lb, period=hma_period, ntd=ntd_d)

        axw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        axw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        axw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        axw.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        axw.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        axw.set_ylim(-1.1, 1.1)

    handles, labels = ax.get_legend_handles_labels()
    if axw is not None:
        h2, l2 = axw.get_legend_handles_labels()
        handles += h2
        labels += l2

    # de-dup legend entries
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
    if axw is not None:
        style_axes(axw)

    st.pyplot(fig)

    # Optional MACD plot
    if show_macd and not macd_d.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        figm.subplots_adjust(top=0.88, bottom=0.45)
        axm.set_title("MACD (optional)")
        axm.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_d.index, macd_sig_d.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(axm)
        st.pyplot(figm)

    # Text trade instruction (same function used elsewhere)
    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_d,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_d
    )
    st.info(instr_txt)

    return {
        "close": close,
        "instr": instr_txt,
        "fib_trigger": fib_trig_d if isinstance(fib_trig_d, dict) else None,
    }

# ---------------------------
# Watchlist parsing + scanner runner
# ---------------------------
def parse_symbols(text: str):
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    parts = re.split(r"[\s,;]+", raw)
    out = []
    seen = set()
    for p in parts:
        p = p.strip().upper()
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out

def run_scan(symbols, fn, *args, **kwargs) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        r = fn(sym, *args, **kwargs)
        if isinstance(r, dict) and r:
            rows.append(r)
    return pd.DataFrame(rows)

def show_df(df: pd.DataFrame, sort_cols=None, ascending=True, max_rows: int = 250):
    if df is None or df.empty:
        st.write("No results.")
        return
    d = df.copy()
    if sort_cols:
        try:
            d = d.sort_values(sort_cols, ascending=ascending)
        except Exception:
            pass
    if max_rows and len(d) > int(max_rows):
        d = d.head(int(max_rows))
    st.dataframe(d, use_container_width=True)

# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# App UI
# ---------------------------
st.title("Bull / Bear Dashboard")

with st.sidebar:
    st.subheader("Chart")
    default_symbol = st.session_state.ticker or "SPY"
    sel = st.text_input("Symbol", value=default_symbol).strip().upper()
    st.session_state.ticker = sel

    is_forex = st.checkbox("Forex mode", value=bool(sel.endswith("=X")))
    st.session_state.hour_range = st.selectbox("Hourly range", ["6h", "12h", "24h", "3d", "5d"], index=2)

    hour_period_map = {"6h": "1d", "12h": "1d", "24h": "1d", "3d": "5d", "5d": "5d"}
    hour_period = hour_period_map.get(st.session_state.hour_range, "1d")

    st.subheader("Daily View")
    daily_view = st.selectbox("Daily View window", ["3mo", "6mo", "1y", "2y", "5y", "10y", "MAX"], index=2)

    st.subheader("Watchlist / Scan Symbols")
    wl_text = st.text_area("Symbols (comma/space/newline separated)", value="SPY QQQ IWM DIA AAPL MSFT NVDA AMZN TSLA", height=90)
    symbols = parse_symbols(wl_text)

    st.session_state.run_all = st.button("Run all scanners")

# Tabs (kept at 18 to match the earlier layout)
tab_names = [
    "1) Daily Chart",
    "2) Hourly Chart",
    "3) Daily Band-Bounce",
    "4) Hourly Band-Bounce",
    "5) Daily NPX↑NTD (Uptrend)",
    "6) Daily NPX 0.5 Cross (Local Slope Up)",
    "7) Daily NPX 0.5 Cross (Local Slope Down)",
    "8) Daily Fib+NPX Zero BUY",
    "9) Daily Fib+NPX Zero SELL",
    "10) Daily Support Reversal (Heading Up)",
    "11) Daily Kijun Cross-Up",
    "12) Daily R² (Regression)",
    "13) Hourly R² (Regression)",
    "14) Daily NTD Last",
    "15) Hourly NTD Last",
    "16) Daily NPX Last",
    "17) Daily NPX 0.5 Marker",
    "18) Daily R² + ±2σ Proximity",
]
tabs = st.tabs(tab_names)

# --- Tab 1: Daily chart
with tabs[0]:
    render_daily_views(sel, is_forex=is_forex)

# --- Tab 2: Hourly chart
with tabs[1]:
    intr = fetch_intraday(sel, period=hour_period)
    if intr is None or intr.empty or "Close" not in intr.columns:
        st.warning("No intraday data.")
    else:
        up_prob, dn_prob = estimate_intraday_updown_probs(intr["Close"])
        render_hourly_views(
            sel=sel,
            intraday=intr,
            p_up=up_prob,
            p_dn=dn_prob,
            hour_range_label=st.session_state.hour_range,
            is_forex=is_forex
        )

# --- Tab 3: Daily Band-Bounce scan
with tabs[2]:
    st.caption("Find most recent regression-band bounce signals (daily).")
    if st.session_state.run_all or st.button("Run daily band-bounce scan"):
        df = run_scan(symbols, last_band_bounce_signal_daily, slope_lb=int(slope_lb_daily))
        show_df(df, sort_cols=["Bars Since", "Symbol"], ascending=True)

# --- Tab 4: Hourly Band-Bounce scan
with tabs[3]:
    st.caption("Find most recent regression-band bounce signals (hourly).")
    if st.session_state.run_all or st.button("Run hourly band-bounce scan"):
        df = run_scan(symbols, last_band_bounce_signal_hourly, period=hour_period, slope_lb=int(slope_lb_hourly))
        show_df(df, sort_cols=["Bars Since", "Symbol"], ascending=True)

# --- Tab 5: Daily NPX↑NTD (Uptrend)
with tabs[4]:
    st.caption("NPX crossed up through NTD while global daily trend slope is UP (in selected Daily View).")
    if st.session_state.run_all or st.button("Run NPX↑NTD scan (daily)"):
        df = run_scan(symbols, last_daily_npx_cross_up_in_uptrend, ntd_win=int(ntd_window), daily_view_label=daily_view)
        show_df(df, sort_cols=["Bars Since", "Symbol"], ascending=True)

# --- Tab 6: Daily NPX 0.5 Cross Up with Local Slope Up
with tabs[5]:
    st.caption("NPX crosses up through +0.5 (near-level constraint) AND local price slope is UP.")
    if st.session_state.run_all or st.button("Run NPX 0.5↑ scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_npx_zero_cross_with_local_slope,
            ntd_win=int(ntd_window),
            daily_view_label=daily_view,
            local_slope_lb=int(npx_local_slope_lb),
            max_abs_npx_at_cross=float(npx_zero_eps),
            direction="up"
        )
        show_df(df, sort_cols=["Bars Since", "Symbol"], ascending=True)

# --- Tab 7: Daily NPX 0.5 Cross Down with Local Slope Down
with tabs[6]:
    st.caption("NPX crosses down through +0.5 (near-level constraint) AND local price slope is DOWN.")
    if st.session_state.run_all or st.button("Run NPX 0.5↓ scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_npx_zero_cross_with_local_slope,
            ntd_win=int(ntd_window),
            daily_view_label=daily_view,
            local_slope_lb=int(npx_local_slope_lb),
            max_abs_npx_at_cross=float(npx_zero_eps),
            direction="down"
        )
        show_df(df, sort_cols=["Bars Since", "Symbol"], ascending=True)

# --- Tab 8: Daily Fib+NPX Zero BUY
with tabs[7]:
    st.caption("BUY when near Fib 100% (low) AND NPX crosses up through 0 AND regression slope flips negative→positive.")
    if st.session_state.run_all or st.button("Run Fib+NPX BUY scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_fib_npx_zero_signal,
            daily_view_label=daily_view,
            ntd_win=int(ntd_window),
            direction="buy",
            prox=float(sr_prox_pct),
            lookback_bars=int(fib_npx_lookback_bars),
            slope_lb=int(slope_lb_daily),
            npx_confirm_bars=int(fib_npx_confirm_bars)
        )
        show_df(df, sort_cols=["Bars Since Cross", "Symbol"], ascending=True)

# --- Tab 9: Daily Fib+NPX Zero SELL
with tabs[8]:
    st.caption("SELL when near Fib 0% (high) AND NPX crosses down through 0 AND regression slope flips positive→negative.")
    if st.session_state.run_all or st.button("Run Fib+NPX SELL scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_fib_npx_zero_signal,
            daily_view_label=daily_view,
            ntd_win=int(ntd_window),
            direction="sell",
            prox=float(sr_prox_pct),
            lookback_bars=int(fib_npx_lookback_bars),
            slope_lb=int(slope_lb_daily),
            npx_confirm_bars=int(fib_npx_confirm_bars)
        )
        show_df(df, sort_cols=["Bars Since Cross", "Symbol"], ascending=True)

# --- Tab 10: Daily Support reversal heading up
with tabs[9]:
    st.caption("Touched near rolling support recently, then closes rising (and distance from support widening).")
    if st.session_state.run_all or st.button("Run support-reversal scan (daily)"):
        df = run_scan(
            symbols,
            daily_support_reversal_heading_up,
            daily_view_label=daily_view,
            sr_lb=int(sr_lb_daily),
            prox=float(sr_prox_pct),
            bars_confirm=int(rev_bars_confirm),
            horizon=int(rev_horizon),
        )
        show_df(df, sort_cols=["Bars Since Touch", "Symbol"], ascending=True)

# --- Tab 11: Daily Kijun cross-up
with tabs[10]:
    st.caption("Close crosses above Ichimoku Kijun (daily) recently, heading up; includes slope + R².")
    if st.session_state.run_all or st.button("Run Kijun cross-up scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_kijun_cross_up,
            daily_view_label=daily_view,
            slope_lb=int(slope_lb_daily),
            conv=int(ichi_conv),
            base=int(ichi_base),
            span_b=int(ichi_spanb),
            within_last_n_bars=int(kijun_within_bars),
        )
        show_df(df, sort_cols=["Bars Since Cross", "Symbol"], ascending=True)

# --- Tab 12: Daily R²
with tabs[11]:
    st.caption("Daily regression R² (full daily history used in fetch_hist).")
    if st.session_state.run_all or st.button("Run daily R² scan"):
        rows = []
        for sym in symbols:
            r2v, mv, ts = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2v):
                rows.append({"Symbol": sym, "R2": r2v, "Slope": mv, "AsOf": ts})
        df = pd.DataFrame(rows)
        show_df(df, sort_cols=["R2", "Symbol"], ascending=False)

# --- Tab 13: Hourly R²
with tabs[12]:
    st.caption("Hourly regression R² (intraday period selection).")
    if st.session_state.run_all or st.button("Run hourly R² scan"):
        rows = []
        for sym in symbols:
            r2v, mv, ts = hourly_regression_r2(sym, period=hour_period, slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2v):
                rows.append({"Symbol": sym, "R2": r2v, "Slope": mv, "AsOf": ts})
        df = pd.DataFrame(rows)
        show_df(df, sort_cols=["R2", "Symbol"], ascending=False)

# --- Tab 14: Daily NTD last
with tabs[13]:
    st.caption("Last Daily NTD value for each symbol (window = NTD).")
    if st.session_state.run_all or st.button("Run daily NTD last scan"):
        rows = []
        for sym in symbols:
            v, ts = last_daily_ntd_value(sym, ntd_win=int(ntd_window))
            if np.isfinite(v):
                rows.append({"Symbol": sym, "NTD (last)": v, "AsOf": ts})
        df = pd.DataFrame(rows)
        show_df(df, sort_cols=["NTD (last)", "Symbol"], ascending=False)

# --- Tab 15: Hourly NTD last
with tabs[14]:
    st.caption("Last Hourly NTD value for each symbol (window = NTD).")
    if st.session_state.run_all or st.button("Run hourly NTD last scan"):
        rows = []
        for sym in symbols:
            v, ts = last_hourly_ntd_value(sym, ntd_win=int(ntd_window), period=hour_period)
            if np.isfinite(v):
                rows.append({"Symbol": sym, "NTD (last)": v, "AsOf": ts})
        df = pd.DataFrame(rows)
        show_df(df, sort_cols=["NTD (last)", "Symbol"], ascending=False)

# --- Tab 16: Daily NPX last
with tabs[15]:
    st.caption("Last Daily NPX value for each symbol (window = NTD).")
    if st.session_state.run_all or st.button("Run daily NPX last scan"):
        rows = []
        for sym in symbols:
            v, ts = last_daily_npx_value(sym, ntd_win=int(ntd_window))
            if np.isfinite(v):
                rows.append({"Symbol": sym, "NPX (last)": v, "AsOf": ts})
        df = pd.DataFrame(rows)
        show_df(df, sort_cols=["NPX (last)", "Symbol"], ascending=False)

# --- Tab 17: Daily NPX 0.5 Marker (per your (a) No, (b) No → single combined table)
with tabs[16]:
    st.caption("NPX recently crossed UP through +0.5; current NPX is between +0.5 and +0.7; NPX slope is positive.")
    if st.session_state.run_all or st.button("Run NPX 0.5 marker scan (daily)"):
        df = run_scan(
            symbols,
            last_daily_npx_05_marker,
            daily_view_label=daily_view,
            ntd_win=int(ntd_window),
            max_bars_since_cross=int(npx05_max_bars_since),
            band_low=float(npx05_band_low),
            band_high=float(npx05_band_high),
            npx_slope_lb=int(npx05_slope_lb),
        )
        show_df(df, sort_cols=["Bars Since Cross", "Symbol"], ascending=True)

# --- Tab 18: Daily R² + ±2σ band proximity (Tab 18 completion)
with tabs[17]:
    st.caption("Daily view-range regression band (±2σ) proximity + R²; useful for mean-reversion setups.")
    if st.session_state.run_all or st.button("Run R² + ±2σ proximity scan (daily)"):
        df = run_scan(
            symbols,
            daily_r2_band_proximity,
            daily_view_label=daily_view,
            slope_lb=int(slope_lb_daily),
            prox=float(sr_prox_pct),
            z=2.0
        )
        # Show those near either band first
        if df is not None and not df.empty:
            try:
                df["Near Either"] = (df["Near Lower"].astype(bool) | df["Near Upper"].astype(bool))
                df = df.sort_values(["Near Either", "R2", "Symbol"], ascending=[False, False, True])
            except Exception:
                pass
        show_df(df)
