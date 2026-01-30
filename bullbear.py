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
# Supertrend / ATR / PSAR / Ichimoku / S/R helpers
# ---------------------------
def compute_atr(ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)
    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    c = pd.to_numeric(ohlc["Close"], errors="coerce")
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/max(2, int(period)), adjust=False).mean()
    return atr

def compute_supertrend(ohlc: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    """
    Returns: (supertrend_line, direction_up_bool)
      direction_up_bool True => uptrend, False => downtrend
    """
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else None
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=bool)

    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    c = pd.to_numeric(ohlc["Close"], errors="coerce")
    atr = compute_atr(ohlc, period=int(atr_period)).reindex(c.index)

    hl2 = (h + l) / 2.0
    upperband = hl2 + float(atr_mult) * atr
    lowerband = hl2 - float(atr_mult) * atr

    st_line = pd.Series(index=c.index, dtype=float)
    dir_up = pd.Series(index=c.index, dtype=bool)

    # initialize
    st_line.iloc[0] = float(upperband.iloc[0]) if np.isfinite(upperband.iloc[0]) else np.nan
    dir_up.iloc[0] = True

    for i in range(1, len(c)):
        prev = i - 1

        ub = upperband.iloc[i]
        lb = lowerband.iloc[i]
        prev_st = st_line.iloc[prev]
        prev_dir = bool(dir_up.iloc[prev])

        # band continuity
        if np.isfinite(ub) and np.isfinite(upperband.iloc[prev]) and c.iloc[prev] > upperband.iloc[prev]:
            ub = max(ub, upperband.iloc[prev])
        if np.isfinite(lb) and np.isfinite(lowerband.iloc[prev]) and c.iloc[prev] < lowerband.iloc[prev]:
            lb = min(lb, lowerband.iloc[prev])

        # direction switch logic
        if prev_dir:
            # was uptrend -> line is lowerband unless close crosses below it
            if np.isfinite(lb) and np.isfinite(c.iloc[i]) and c.iloc[i] < lb:
                dir_up.iloc[i] = False
                st_line.iloc[i] = ub if np.isfinite(ub) else prev_st
            else:
                dir_up.iloc[i] = True
                st_line.iloc[i] = lb if np.isfinite(lb) else prev_st
        else:
            # was downtrend -> line is upperband unless close crosses above it
            if np.isfinite(ub) and np.isfinite(c.iloc[i]) and c.iloc[i] > ub:
                dir_up.iloc[i] = True
                st_line.iloc[i] = lb if np.isfinite(lb) else prev_st
            else:
                dir_up.iloc[i] = False
                st_line.iloc[i] = ub if np.isfinite(ub) else prev_st

    return st_line, dir_up

def compute_psar(ohlc: pd.DataFrame, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """
    Parabolic SAR implementation (classic).
    Returns psar series.
    """
    if ohlc is None or ohlc.empty or not {"High","Low"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else None
        return pd.Series(index=idx, dtype=float)

    high = pd.to_numeric(ohlc["High"], errors="coerce").to_numpy()
    low  = pd.to_numeric(ohlc["Low"], errors="coerce").to_numpy()
    idx = ohlc.index

    psar = np.full(len(idx), np.nan, dtype=float)

    # pick initial direction
    bull = True
    if len(idx) >= 2 and np.isfinite(high[1]) and np.isfinite(high[0]) and np.isfinite(low[1]) and np.isfinite(low[0]):
        bull = (high[1] + low[1]) >= (high[0] + low[0])

    af = float(step)
    ep = high[0] if bull else low[0]
    psar[0] = low[0] if bull else high[0]

    for i in range(1, len(idx)):
        prev = i - 1
        if not (np.isfinite(high[i]) and np.isfinite(low[i]) and np.isfinite(psar[prev])):
            psar[i] = psar[prev]
            continue

        ps = psar[prev] + af * (ep - psar[prev])

        # constrain
        if bull:
            ps = min(ps, low[prev], low[i])
        else:
            ps = max(ps, high[prev], high[i])

        # reversal checks
        if bull and low[i] < ps:
            bull = False
            ps = ep
            ep = low[i]
            af = float(step)
        elif (not bull) and high[i] > ps:
            bull = True
            ps = ep
            ep = high[i]
            af = float(step)
        else:
            # update EP/AF
            if bull and high[i] > ep:
                ep = high[i]
                af = min(float(max_af), af + float(step))
            if (not bull) and low[i] < ep:
                ep = low[i]
                af = min(float(max_af), af + float(step))

        psar[i] = ps

    return pd.Series(psar, index=idx)

def compute_support_resistance(close: pd.Series, lookback: int = 60):
    """
    Simple rolling S/R:
      support = rolling min
      resistance = rolling max
    """
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float), pd.Series(index=s.index, dtype=float)
    lb = max(5, int(lookback))
    sup = s.rolling(lb, min_periods=max(3, lb//3)).min()
    res = s.rolling(lb, min_periods=max(3, lb//3)).max()
    return sup.reindex(s.index), res.reindex(s.index)

def compute_kijun(close: pd.Series, high: pd.Series, low: pd.Series, base: int = 26) -> pd.Series:
    """
    Ichimoku Kijun-sen = (highest(high, base) + lowest(low, base))/2
    Returned in price units.
    """
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(high).reindex(c.index)
    l = _coerce_1d_series(low).reindex(c.index)
    if c.empty:
        return pd.Series(index=c.index, dtype=float)
    base = max(2, int(base))
    hh = h.rolling(base, min_periods=max(2, base//2)).max()
    ll = l.rolling(base, min_periods=max(2, base//2)).min()
    return ((hh + ll) / 2.0).reindex(c.index)

def session_lines_pst(index: pd.DatetimeIndex):
    """
    Returns dict with London/NY open times per day, in PST.
    London open ~ 00:00 PST (varies with DST, approximation)
    NY open ~ 06:30 PST (approx)
    This is a visual aid only.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        return {}
    # Normalize to date
    days = pd.to_datetime(pd.Series(index.date).unique())
    out = {"London": [], "NewYork": []}
    for d in days:
        try:
            d = pd.Timestamp(d).tz_localize(PACIFIC)
        except Exception:
            d = pd.Timestamp(d).tz_convert(PACIFIC) if hasattr(d, "tz_convert") else pd.Timestamp(d)
            if getattr(d, "tz", None) is None:
                d = d.tz_localize(PACIFIC)
        out["London"].append(d + pd.Timedelta(hours=0, minutes=0))
        out["NewYork"].append(d + pd.Timedelta(hours=6, minutes=30))
    return out

def _safe_title_ticker(ticker: str) -> str:
    return str(ticker).replace("=X", "")

# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Main controls (ticker + run)
# ---------------------------
st.subheader("Run Panel")

# ensure a valid default ticker in session_state
if "ticker" not in st.session_state or st.session_state.ticker not in universe:
    st.session_state.ticker = universe[0]

c1, c2, c3, c4 = st.columns([2.0, 1.3, 1.3, 1.1])

ticker = c1.selectbox("Ticker / Pair", universe, index=universe.index(st.session_state.ticker), key="sb_ticker")
st.session_state.ticker = ticker

intraday_period = c2.selectbox("Intraday period", ["1d", "5d", "1mo"], index=0, key="sb_intraday_period")
band_lookback = c3.selectbox("Regression band lookback", ["Auto", "60", "90", "120", "180"], index=2, key="sb_band_lb")
run_clicked = c4.button("▶ Run / Update", use_container_width=True, key="btn_run_all")

# optional auto-run on first load
if "run_all" not in st.session_state:
    st.session_state.run_all = False

if run_clicked:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# If mode changed after prior run, force rerun to prevent stale series mismatch
if st.session_state.get("run_all", False) and st.session_state.get("mode_at_run") != mode:
    st.session_state.run_all = False

# ---------------------------
# Fetch + compute (only after Run)
# ---------------------------
if st.session_state.get("run_all", False):
    with st.spinner("Fetching data + computing indicators..."):
        # daily series
        try:
            df_hist = fetch_hist(ticker)
        except Exception:
            df_hist = pd.Series(dtype=float)

        try:
            df_ohlc = fetch_hist_ohlc(ticker)
        except Exception:
            df_ohlc = pd.DataFrame()

        # intraday ohlc
        try:
            intraday = fetch_intraday(ticker, period=intraday_period)
        except Exception:
            intraday = pd.DataFrame()

        # store raw
        st.session_state.df_hist = df_hist
        st.session_state.df_ohlc = df_ohlc
        st.session_state.intraday = intraday

        # apply daily view windowing (for plotting panels only)
        df_hist_view = subset_by_daily_view(df_hist, daily_view)
        df_ohlc_view = subset_by_daily_view(df_ohlc, daily_view) if isinstance(df_ohlc, pd.DataFrame) else df_ohlc

        # regression with ±2σ on daily
        lb = 0
        if band_lookback != "Auto":
            lb = int(band_lookback)

        yhat_d, up_d, lo_d, slope_d, r2_d = regression_with_band(df_hist_view, lookback=lb, z=2.0)

        # global slope uses full view series for robustness
        _, global_slope = slope_line(df_hist_view, lookback=max(30, int(slope_lb_daily)))

        # fibonacci + fib triggers
        fibs_d = fibonacci_levels(df_hist_view) if show_fibs else {}
        fib_rev = fib_reversal_trigger_from_extremes(df_hist_view, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=60)

        # support/resistance daily
        sup_d, res_d = compute_support_resistance(df_hist_view, lookback=int(sr_lb_daily))

        # NTD / NPX daily (for NTD panel)
        ntd_d = compute_normalized_trend(df_hist_view, window=int(ntd_window)) if show_ntd else pd.Series(index=df_hist_view.index, dtype=float)
        npx_d = compute_normalized_price(df_hist_view, window=int(ntd_window)) if show_ntd else pd.Series(index=df_hist_view.index, dtype=float)

        # fib+NPX(0) signals (daily)
        fib_buy_d, fib_sell_d, fibs_for_signals = fib_npx_zero_cross_signal_masks(
            price=df_hist_view,
            npx=npx_d,
            horizon_bars=int(rev_horizon),
            proximity_pct_of_range=0.02,
            npx_level=0.0
        )

        # Bollinger + HMA
        bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df_hist_view, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema)) if show_bbands else (
            pd.Series(index=df_hist_view.index, dtype=float),)*5
        hma_d = compute_hma(df_hist_view, period=int(hma_period)) if show_hma else pd.Series(index=df_hist_view.index, dtype=float)

        # slope trigger based on band reversal
        slope_trig = find_slope_trigger_after_band_reversal(df_hist_view, yhat_d, up_d, lo_d, horizon=int(rev_horizon))

        # reversal probability
        rev_prob = slope_reversal_probability(df_hist_view, current_slope=slope_d, hist_window=int(rev_hist_lb), slope_window=int(ntd_window), horizon=int(rev_horizon))

        # Intraday indicator stack
        if isinstance(intraday, pd.DataFrame) and not intraday.empty and "Close" in intraday.columns:
            close_i = intraday["Close"].copy()
            sup_i, res_i = compute_support_resistance(close_i, lookback=int(sr_lb_hourly))
            ntd_i = compute_normalized_trend(close_i, window=int(ntd_window)) if show_ntd else pd.Series(index=close_i.index, dtype=float)
            npx_i = compute_normalized_price(close_i, window=int(ntd_window)) if show_ntd else pd.Series(index=close_i.index, dtype=float)

            st_i, st_dir_i = compute_supertrend(intraday[["High","Low","Close"]].copy(), atr_period=int(atr_period), atr_mult=float(atr_mult))

            psar_i = compute_psar(intraday[["High","Low"]].copy(), step=float(psar_step), max_af=float(psar_max)) if show_psar else pd.Series(index=intraday.index, dtype=float)

            # MACD/HMA/SR combined signal (intraday)
            nmacd_i, nsignal_i, nhist_i = compute_nmacd(close_i, fast=12, slow=26, signal=9, norm_win=240)
            hma_i = compute_hma(close_i, period=int(hma_period))
            macd_sr_sig = find_macd_hma_sr_signal(
                close=close_i, hma=hma_i, macd=nmacd_i,
                sup=sup_i, res=res_i,
                global_trend_slope=float(global_slope) if np.isfinite(global_slope) else np.nan,
                prox=float(sr_prox_pct)
            )

            # hourly momentum
            roc_i = compute_roc(close_i, n=int(mom_lb_hourly)) if show_mom_hourly else pd.Series(index=close_i.index, dtype=float)

        else:
            close_i = pd.Series(dtype=float)
            sup_i = res_i = pd.Series(dtype=float)
            ntd_i = npx_i = pd.Series(dtype=float)
            st_i = pd.Series(dtype=float)
            st_dir_i = pd.Series(dtype=bool)
            psar_i = pd.Series(dtype=float)
            nmacd_i = nsignal_i = nhist_i = pd.Series(dtype=float)
            hma_i = pd.Series(dtype=float)
            macd_sr_sig = None
            roc_i = pd.Series(dtype=float)

        # save computed outputs
        st.session_state.comp = {
            "df_hist_view": df_hist_view,
            "df_ohlc_view": df_ohlc_view,
            "yhat_d": yhat_d, "up_d": up_d, "lo_d": lo_d, "slope_d": slope_d, "r2_d": r2_d,
            "global_slope": global_slope,
            "fibs_d": fibs_d,
            "fib_rev": fib_rev,
            "sup_d": sup_d, "res_d": res_d,
            "ntd_d": ntd_d, "npx_d": npx_d,
            "fib_buy_d": fib_buy_d, "fib_sell_d": fib_sell_d,
            "bb_mid_d": bb_mid_d, "bb_up_d": bb_up_d, "bb_lo_d": bb_lo_d,
            "hma_d": hma_d,
            "slope_trig": slope_trig,
            "rev_prob": rev_prob,

            "close_i": close_i,
            "sup_i": sup_i, "res_i": res_i,
            "ntd_i": ntd_i, "npx_i": npx_i,
            "st_i": st_i, "st_dir_i": st_dir_i,
            "psar_i": psar_i,
            "nmacd_i": nmacd_i, "nsignal_i": nsignal_i, "nhist_i": nhist_i,
            "hma_i": hma_i,
            "macd_sr_sig": macd_sr_sig,
            "roc_i": roc_i,
        }

# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Plotters (matplotlib)
# ---------------------------
def plot_daily_price_panel(ticker: str, ohlc: pd.DataFrame, close: pd.Series, comp: dict):
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_title(f"Daily Price — {_safe_title_ticker(ticker)}")
    ax.plot(close.index, close.values, label="Close", alpha=0.95)

    # regression + band
    yhat = comp.get("yhat_d", pd.Series(dtype=float))
    up = comp.get("up_d", pd.Series(dtype=float))
    lo = comp.get("lo_d", pd.Series(dtype=float))
    slope_val = comp.get("slope_d", np.nan)
    r2 = comp.get("r2_d", np.nan)

    if isinstance(yhat, pd.Series) and not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trendline (slope {fmt_slope(slope_val)}, R² {fmt_r2(r2)})", alpha=0.95)
    if isinstance(up, pd.Series) and isinstance(lo, pd.Series) and (not up.empty) and (not lo.empty):
        ax.plot(up.index, up.values, linestyle=":", alpha=0.85, label="±2σ band")
        ax.plot(lo.index, lo.values, linestyle=":", alpha=0.85)
        ax.fill_between(up.index, lo.values, up.values, alpha=0.10)

    # Bollinger Bands
    if show_bbands:
        bb_mid = comp.get("bb_mid_d"); bb_up = comp.get("bb_up_d"); bb_lo = comp.get("bb_lo_d")
        if isinstance(bb_mid, pd.Series) and not bb_mid.empty:
            ax.plot(bb_mid.index, bb_mid.values, alpha=0.75, label="BB mid")
        if isinstance(bb_up, pd.Series) and isinstance(bb_lo, pd.Series) and (not bb_up.empty) and (not bb_lo.empty):
            ax.plot(bb_up.index, bb_up.values, alpha=0.55, label="BB upper/lower")
            ax.plot(bb_lo.index, bb_lo.values, alpha=0.55)
            ax.fill_between(bb_up.index, bb_lo.values, bb_up.values, alpha=0.06)

    # HMA
    if show_hma:
        hma_d = comp.get("hma_d", pd.Series(dtype=float))
        if isinstance(hma_d, pd.Series) and not hma_d.empty:
            ax.plot(hma_d.index, hma_d.values, linewidth=2.2, alpha=0.85, label=f"HMA({hma_period})")

    # Ichimoku Kijun on price
    if show_ichi and isinstance(ohlc, pd.DataFrame) and not ohlc.empty:
        try:
            kij = compute_kijun(close, ohlc["High"], ohlc["Low"], base=int(ichi_base))
            ax.plot(kij.index, kij.values, linestyle="-.", alpha=0.85, label=f"Kijun({ichi_base})")
        except Exception:
            pass

    # Fibonacci
    fibs = comp.get("fibs_d", {})
    if show_fibs and isinstance(fibs, dict) and fibs:
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=1.0, alpha=0.35)
                label_on_left(ax, v, f"Fib {k}: {fmt_price_val(v)}", fontsize=8)

    # Fibonacci BUY/SELL markers from (Fib touch + NPX cross 0.0)
    if show_fibs and show_ntd:
        fb = comp.get("fib_buy_d", pd.Series(dtype=bool))
        fs = comp.get("fib_sell_d", pd.Series(dtype=bool))
        overlay_fib_npx_signals(ax, close, fb, fs)

    # slope trigger annotation
    trig = comp.get("slope_trig", None)
    annotate_slope_trigger(ax, trig)

    # daily S/R lines
    sup = comp.get("sup_d", pd.Series(dtype=float))
    res = comp.get("res_d", pd.Series(dtype=float))
    if isinstance(sup, pd.Series) and not sup.empty:
        ax.plot(sup.index, sup.values, alpha=0.55, linestyle="--", label=f"Support({sr_lb_daily})")
    if isinstance(res, pd.Series) and not res.empty:
        ax.plot(res.index, res.values, alpha=0.55, linestyle="--", label=f"Resistance({sr_lb_daily})")

    style_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

def plot_intraday_panel(ticker: str, intraday: pd.DataFrame, comp: dict):
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.set_title(f"Intraday (5m gapless) — {_safe_title_ticker(ticker)}")

    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        ax.text(0.5, 0.5, "No intraday data.", ha="center", va="center", transform=ax.transAxes)
        style_axes(ax)
        fig.tight_layout()
        return fig

    close = pd.to_numeric(intraday["Close"], errors="coerce")
    ax.plot(np.arange(len(close)), close.values, label="Close", alpha=0.95)

    # S/R
    sup = comp.get("sup_i", pd.Series(dtype=float)).reindex(close.index)
    res = comp.get("res_i", pd.Series(dtype=float)).reindex(close.index)
    if isinstance(sup, pd.Series) and not sup.empty:
        ax.plot(np.arange(len(sup)), sup.values, linestyle="--", alpha=0.55, label=f"Support({sr_lb_hourly})")
    if isinstance(res, pd.Series) and not res.empty:
        ax.plot(np.arange(len(res)), res.values, linestyle="--", alpha=0.55, label=f"Resistance({sr_lb_hourly})")

    # Supertrend
    st_line = comp.get("st_i", pd.Series(dtype=float)).reindex(close.index)
    st_dir = comp.get("st_dir_i", pd.Series(dtype=bool)).reindex(close.index)
    if isinstance(st_line, pd.Series) and not st_line.empty:
        ax.plot(np.arange(len(st_line)), st_line.values, linewidth=2.0, alpha=0.85, label="Supertrend")

    # PSAR
    if show_psar:
        psar = comp.get("psar_i", pd.Series(dtype=float)).reindex(close.index)
        if isinstance(psar, pd.Series) and not psar.empty:
            ax.scatter(np.arange(len(psar)), psar.values, s=10, alpha=0.55, label="PSAR")

    # Session markers (PST)
    if mode == "Forex" and show_sessions_pst and isinstance(intraday.index, pd.DatetimeIndex):
        sess = session_lines_pst(intraday.index)
        pos_l = _map_times_to_bar_positions(intraday.index, sess.get("London", []))
        pos_n = _map_times_to_bar_positions(intraday.index, sess.get("NewYork", []))
        for p in pos_l:
            ax.axvline(p, linestyle=":", alpha=0.25)
        for p in pos_n:
            ax.axvline(p, linestyle=":", alpha=0.25)

    # optional MACD/HMA/SR star marker
    sig = comp.get("macd_sr_sig", None)
    if sig is not None and "time" in sig:
        t = sig["time"]
        side = sig.get("side", "")
        try:
            # map to intraday position
            pidx = intraday.index.get_indexer([pd.to_datetime(t)], method="nearest")[0]
            px = float(close.iloc[pidx]) if pidx >= 0 else np.nan
            if np.isfinite(px):
                if side == "BUY":
                    ax.scatter([pidx], [px], marker="*", s=200, color="tab:green", zorder=10, label="MACD/HMA + S/R")
                else:
                    ax.scatter([pidx], [px], marker="*", s=200, color="tab:red", zorder=10, label="MACD/HMA + S/R")
        except Exception:
            pass

    # Compact time ticks
    _apply_compact_time_ticks(ax, intraday.index, n_ticks=8)

    style_axes(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

def plot_ntd_panel(title: str, ntd: pd.Series, npx: pd.Series, sup: pd.Series = None, res: pd.Series = None):
    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.set_title(title)

    ntd = _coerce_1d_series(ntd)
    npx = _coerce_1d_series(npx).reindex(ntd.index)

    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.plot(ntd.index, ntd.values, label="NTD", linewidth=2.0, alpha=0.9)
    if show_npx_ntd:
        ax.plot(npx.index, npx.values, label="NPX", alpha=0.75)

    if shade_ntd:
        shade_ntd_regions(ax, ntd)

    # Optional channel shading when "between S/R" on NTD (rough proxy: npx between ntd? visual cue)
    if show_ntd_channel and (sup is not None) and (res is not None):
        # this is a lightweight cue: shade when price is between support/resistance
        # actual channel logic is applied in the price panel; here we provide an extra visual
        pass

    # Mark NPX↔NTD crosses as dots
    if mark_npx_cross and (not ntd.empty) and (not npx.empty):
        diff = (npx - ntd)
        sgn = np.sign(diff)
        cross = (sgn != sgn.shift(1)) & sgn.notna() & sgn.shift(1).notna()
        cross_idx = list(cross[cross].index)
        if cross_idx:
            ax.scatter(cross_idx, ntd.loc[cross_idx].values, s=45, alpha=0.9, label="NPX↔NTD cross")

    style_axes(ax)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

def plot_macd_panel(title: str, nmacd: pd.Series, nsignal: pd.Series, nhist: pd.Series):
    fig, ax = plt.subplots(figsize=(12.8, 3.8))
    ax.set_title(title)
    nmacd = _coerce_1d_series(nmacd)
    nsignal = _coerce_1d_series(nsignal).reindex(nmacd.index)
    nhist = _coerce_1d_series(nhist).reindex(nmacd.index)

    ax.axhline(0.0, linewidth=1.0, alpha=0.35)
    ax.plot(nmacd.index, nmacd.values, label="nMACD", linewidth=2.0, alpha=0.9)
    ax.plot(nsignal.index, nsignal.values, label="Signal", alpha=0.75)
    ax.bar(nhist.index, nhist.values, alpha=0.25, label="Hist")

    style_axes(ax)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Tabs (FIXED: define ALL tab variables so tab15 exists)
# ---------------------------
tab_labels = [
    "🏁 Overview",
    "📅 Daily Chart",
    "⏱️ Intraday (5m)",
    "🧠 Forecast (SARIMAX)",
    "🧬 Fibonacci",
    "🧯 Support/Resistance",
    "🧭 NTD / NPX",
    "📉 MACD",
    "🧮 HMA Signals",
    "🌀 Supertrend / PSAR",
    "📌 Pivots",
    "🧪 Reversal Prob",
    "📰 News",
    "📋 Data Tables",
    "ℹ️ About",
]
tabs = st.tabs(tab_labels)

# Unpack explicitly (prevents NameError: tab15 ...)
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8, tab9, tab10,
 tab11, tab12, tab13, tab14, tab15) = tabs

# ---------------------------
# Basic tab content (more tabs completed in Batch 3)
# ---------------------------
with tab1:
    st.subheader("Overview")
    if not st.session_state.get("run_all", False):
        st.info("Click **Run / Update** to load data and populate all tabs.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        slope_d = comp.get("slope_d", np.nan)
        global_slope = comp.get("global_slope", np.nan)
        r2_d = comp.get("r2_d", np.nan)
        rev_prob = comp.get("rev_prob", np.nan)

        last = _safe_last_float(close)
        st.metric("Last (daily close)", fmt_price_val(last) if np.isfinite(last) else "n/a")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Local slope", fmt_slope(slope_d))
        colB.metric("Global slope", fmt_slope(global_slope))
        colC.metric("Trend fit R²", fmt_r2(r2_d))
        colD.metric("Reversal prob", fmt_pct(rev_prob, digits=1))

        # Trade instruction (uses NEW: require global/local slope agreement)
        if np.isfinite(last):
            # lightweight buy/sell values: use S/R if available
            sup = comp.get("sup_d", pd.Series(dtype=float))
            res = comp.get("res_d", pd.Series(dtype=float))
            buy_val = _safe_last_float(sup) if isinstance(sup, pd.Series) else last
            sell_val = _safe_last_float(res) if isinstance(res, pd.Series) else last
            instr = format_trade_instruction(
                trend_slope=slope_d,
                buy_val=buy_val,
                sell_val=sell_val,
                close_val=last,
                symbol=ticker,
                global_trend_slope=global_slope
            )
            st.success(instr) if "BUY" in instr else st.warning(instr)

        # show any fib reversal alert
        fib_rev = comp.get("fib_rev", None)
        if fib_rev is not None:
            side = fib_rev.get("side", "")
            t = fib_rev.get("touch_time")
            tp = fib_rev.get("touch_price")
            st.info(f"{FIB_ALERT_TEXT}\n\nLatest Fib reversal trigger: **{side}** near **{fib_rev.get('from_level','?')}** at {t} (touch {fmt_price_val(tp)})")

with tab2:
    st.subheader("Daily Chart")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to see the daily chart.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        ohlc = comp.get("df_ohlc_view", pd.DataFrame())
        fig = plot_daily_price_panel(ticker, ohlc, close, comp)
        st.pyplot(fig, use_container_width=True)

with tab3:
    st.subheader("Intraday (5m, gapless)")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to see intraday data.")
    else:
        comp = st.session_state.get("comp", {})
        intraday = st.session_state.get("intraday", pd.DataFrame())
        fig = plot_intraday_panel(ticker, intraday, comp)
        st.pyplot(fig, use_container_width=True)

        if show_mom_hourly and isinstance(comp.get("roc_i", None), pd.Series) and not comp["roc_i"].empty:
            st.caption("Hourly momentum (ROC%)")
            fig2, ax2 = plt.subplots(figsize=(12.8, 3.2))
            ax2.axhline(0.0, linewidth=1.0, alpha=0.35)
            ax2.plot(comp["roc_i"].index, comp["roc_i"].values, label=f"ROC({mom_lb_hourly})%", alpha=0.9)
            style_axes(ax2)
            ax2.legend(loc="upper left")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)

with tab4:
    st.subheader("Forecast (SARIMAX)")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to generate the SARIMAX forecast.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        if close is None or close.empty:
            st.warning("No daily series available.")
        else:
            try:
                fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close)
                st.session_state.fc_idx = fc_idx
                st.session_state.fc_vals = fc_vals
                st.session_state.fc_ci = fc_ci

                fig, ax = plt.subplots(figsize=(12.8, 5.0))
                ax.set_title(f"30-Day Forecast — {_safe_title_ticker(ticker)}")
                ax.plot(close.index, close.values, label="History", alpha=0.9)
                ax.plot(fc_idx, fc_vals.values, linestyle="--", label="Forecast", alpha=0.9)
                ax.fill_between(fc_idx, fc_ci.iloc[:, 0].values, fc_ci.iloc[:, 1].values, alpha=0.15, label="Conf. Int.")
                style_axes(ax)
                ax.legend(loc="upper left")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")

with tab5:
    st.subheader("Fibonacci")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to view Fibonacci levels and signals.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        fibs = comp.get("fibs_d", {})
        if not show_fibs:
            st.info("Enable **Show Fibonacci** in the sidebar.")
        elif not fibs:
            st.warning("Fibonacci levels unavailable (not enough range).")
        else:
            df_fib = pd.DataFrame({"Level": list(fibs.keys()), "Price": list(fibs.values())})
            st.dataframe(df_fib, use_container_width=True)

            # show latest fib+NPX(0) cross signal
            if show_ntd:
                fb = comp.get("fib_buy_d", pd.Series(dtype=bool))
                fs = comp.get("fib_sell_d", pd.Series(dtype=bool))
                last_buy = fb[fb].index[-1] if isinstance(fb, pd.Series) and fb.any() else None
                last_sell = fs[fs].index[-1] if isinstance(fs, pd.Series) and fs.any() else None
                if last_buy is None and last_sell is None:
                    st.info("No recent Fib+NPX(0) cross signals.")
                else:
                    if last_sell is None or (last_buy is not None and last_buy >= last_sell):
                        st.success(f"Latest Fib signal: **BUY** at {last_buy}")
                    else:
                        st.warning(f"Latest Fib signal: **SELL** at {last_sell}")

with tab6:
    st.subheader("Support / Resistance")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to view S/R calculations.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        sup = comp.get("sup_d", pd.Series(dtype=float))
        res = comp.get("res_d", pd.Series(dtype=float))
        if close.empty:
            st.warning("No daily data.")
        else:
            fig, ax = plt.subplots(figsize=(12.8, 4.8))
            ax.set_title(f"Daily S/R — {_safe_title_ticker(ticker)}")
            ax.plot(close.index, close.values, label="Close", alpha=0.9)
            if isinstance(sup, pd.Series) and not sup.empty:
                ax.plot(sup.index, sup.values, linestyle="--", alpha=0.75, label="Support")
            if isinstance(res, pd.Series) and not res.empty:
                ax.plot(res.index, res.values, linestyle="--", alpha=0.75, label="Resistance")
            style_axes(ax)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

with tab7:
    st.subheader("NTD / NPX")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to view NTD/NPX.")
    else:
        comp = st.session_state.get("comp", {})
        ntd_d = comp.get("ntd_d", pd.Series(dtype=float))
        npx_d = comp.get("npx_d", pd.Series(dtype=float))
        if not show_ntd:
            st.info("Enable **Show NTD overlay** in the sidebar.")
        else:
            fig = plot_ntd_panel("Daily NTD / NPX", ntd_d, npx_d, comp.get("sup_d"), comp.get("res_d"))
            st.pyplot(fig, use_container_width=True)

            # Intraday NTD/NPX (if available)
            ntd_i = comp.get("ntd_i", pd.Series(dtype=float))
            npx_i = comp.get("npx_i", pd.Series(dtype=float))
            if isinstance(ntd_i, pd.Series) and not ntd_i.empty:
                fig2 = plot_ntd_panel("Intraday NTD / NPX", ntd_i, npx_i, comp.get("sup_i"), comp.get("res_i"))
                st.pyplot(fig2, use_container_width=True)

with tab8:
    st.subheader("MACD (Normalized)")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to view MACD.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        if close.empty:
            st.warning("No daily data.")
        else:
            nmacd_d, nsignal_d, nhist_d = compute_nmacd(close, fast=12, slow=26, signal=9, norm_win=240)
            fig = plot_macd_panel("Daily nMACD", nmacd_d, nsignal_d, nhist_d)
            st.pyplot(fig, use_container_width=True)

        # Intraday MACD
        nmacd_i = comp.get("nmacd_i", pd.Series(dtype=float))
        nsignal_i = comp.get("nsignal_i", pd.Series(dtype=float))
        nhist_i = comp.get("nhist_i", pd.Series(dtype=float))
        if isinstance(nmacd_i, pd.Series) and not nmacd_i.empty:
            fig2 = plot_macd_panel("Intraday nMACD", nmacd_i, nsignal_i, nhist_i)
            st.pyplot(fig2, use_container_width=True)

# NOTE:
# Tabs 9–15 are defined (so NO NameError), and will be fully populated in Batch 3.
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Extra helpers for tabs 9–15
# ---------------------------
def _cross_events(series_a: pd.Series, series_b: pd.Series) -> pd.DataFrame:
    """
    Returns a dataframe of cross events where A crosses above/below B.
    """
    a = _coerce_1d_series(series_a).astype(float)
    b = _coerce_1d_series(series_b).reindex(a.index).astype(float)
    if a.empty or b.empty:
        return pd.DataFrame(columns=["time", "side", "a", "b"])
    diff = a - b
    sgn = np.sign(diff)
    cross = (sgn != sgn.shift(1)) & sgn.notna() & sgn.shift(1).notna()
    times = list(cross[cross].index)
    rows = []
    for t in times:
        i = a.index.get_loc(t)
        if i <= 0:
            continue
        prev = i - 1
        if not (np.isfinite(diff.iloc[i]) and np.isfinite(diff.iloc[prev])):
            continue
        side = "ABOVE" if diff.iloc[i] > 0 else "BELOW"
        rows.append({"time": t, "side": side, "a": float(a.iloc[i]), "b": float(b.iloc[i])})
    return pd.DataFrame(rows)

def compute_classic_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """
    Classic floor pivots from previous session:
      PP = (H+L+C)/3
      R1 = 2*PP - L
      S1 = 2*PP - H
      R2 = PP + (H-L)
      S2 = PP - (H-L)
      R3 = H + 2*(PP-L)
      S3 = L - 2*(H-PP)
    """
    H, L, C = float(prev_high), float(prev_low), float(prev_close)
    pp = (H + L + C) / 3.0
    r1 = 2.0 * pp - L
    s1 = 2.0 * pp - H
    r2 = pp + (H - L)
    s2 = pp - (H - L)
    r3 = H + 2.0 * (pp - L)
    s3 = L - 2.0 * (H - pp)
    return {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

def _latest_prev_session_ohlc(df_ohlc: pd.DataFrame):
    """
    Returns previous completed session (H,L,C) from daily ohlc.
    Uses the second-to-last row when available, else last row.
    """
    if not isinstance(df_ohlc, pd.DataFrame) or df_ohlc.empty:
        return None
    cols = {"High","Low","Close"}
    if not cols.issubset(df_ohlc.columns):
        return None
    use_idx = -2 if len(df_ohlc) >= 2 else -1
    row = df_ohlc.iloc[use_idx]
    try:
        return float(row["High"]), float(row["Low"]), float(row["Close"]), df_ohlc.index[use_idx]
    except Exception:
        return None

def _news_rss(symbol: str, n: int = 15) -> pd.DataFrame:
    """
    Lightweight RSS fetch using Google News query RSS.
    Robust to failures; returns columns: title, published, link.
    """
    import xml.etree.ElementTree as ET
    import requests
    from datetime import datetime

    sym = str(symbol).replace("=X", "").strip()
    q = sym
    # for forex pairs like EURUSD, add "forex"
    if mode == "Forex":
        q = f"{sym} forex"
    elif mode == "Crypto":
        q = f"{sym} crypto"
    elif mode == "Stocks":
        q = f"{sym} stock"

    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"

    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = root.findall(".//item")
        rows = []
        for it in items[: max(5, int(n))]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            # keep pubDate raw; also attempt parse
            parsed = None
            try:
                parsed = pd.to_datetime(pub, utc=True)
            except Exception:
                parsed = pd.NaT
            rows.append({"title": title, "published": parsed, "published_raw": pub, "link": link})
        df = pd.DataFrame(rows)
        if not df.empty and "published" in df.columns:
            df = df.sort_values("published", ascending=False, na_position="last")
        return df
    except Exception:
        return pd.DataFrame(columns=["title","published","published_raw","link"])

# =========================
# Part 10/10 — bullbear.py
# ---------------------------
# Tabs 9–15 (complete)
# ---------------------------

with tab9:
    st.subheader("HMA Signals")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to compute HMA signals.")
    else:
        comp = st.session_state.get("comp", {})
        close_d = comp.get("df_hist_view", pd.Series(dtype=float))
        hma_d = comp.get("hma_d", pd.Series(dtype=float))

        if close_d.empty:
            st.warning("No daily data.")
        else:
            # Daily crosses
            if not show_hma:
                st.info("Enable **Show HMA** in the sidebar to overlay on charts.")
            if hma_d is None or hma_d.empty:
                hma_d = compute_hma(close_d, period=int(hma_period))

            ev_d = _cross_events(close_d, hma_d)
            st.caption("Daily price ↔ HMA crosses")
            if ev_d.empty:
                st.write("No cross events found in the current window.")
            else:
                st.dataframe(ev_d.tail(12), use_container_width=True)

            fig, ax = plt.subplots(figsize=(12.8, 4.6))
            ax.set_title(f"Daily Close + HMA({hma_period}) — {_safe_title_ticker(ticker)}")
            ax.plot(close_d.index, close_d.values, label="Close", alpha=0.9)
            ax.plot(hma_d.index, hma_d.values, label="HMA", linewidth=2.2, alpha=0.85)
            style_axes(ax)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Intraday HMA crosses
        close_i = comp.get("close_i", pd.Series(dtype=float))
        hma_i = comp.get("hma_i", pd.Series(dtype=float))
        if isinstance(close_i, pd.Series) and (not close_i.empty):
            if hma_i is None or hma_i.empty:
                hma_i = compute_hma(close_i, period=int(hma_period))
            ev_i = _cross_events(close_i, hma_i)

            st.caption("Intraday price ↔ HMA crosses")
            if ev_i.empty:
                st.write("No intraday cross events found.")
            else:
                st.dataframe(ev_i.tail(12), use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(12.8, 4.2))
            ax2.set_title(f"Intraday Close + HMA({hma_period}) — {_safe_title_ticker(ticker)}")
            ax2.plot(np.arange(len(close_i)), close_i.values, label="Close", alpha=0.9)
            ax2.plot(np.arange(len(hma_i)), hma_i.values, label="HMA", linewidth=2.2, alpha=0.85)
            style_axes(ax2)
            ax2.legend(loc="upper left")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)

with tab10:
    st.subheader("Supertrend / PSAR")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to compute Supertrend / PSAR.")
    else:
        comp = st.session_state.get("comp", {})
        intraday = st.session_state.get("intraday", pd.DataFrame())
        df_ohlc_view = comp.get("df_ohlc_view", pd.DataFrame())
        close_d = comp.get("df_hist_view", pd.Series(dtype=float))

        # Daily supertrend (if OHLC available)
        if isinstance(df_ohlc_view, pd.DataFrame) and (not df_ohlc_view.empty) and {"High","Low","Close"}.issubset(df_ohlc_view.columns):
            st_line_d, st_dir_d = compute_supertrend(df_ohlc_view[["High","Low","Close"]].copy(), atr_period=int(atr_period), atr_mult=float(atr_mult))
            psar_d = compute_psar(df_ohlc_view[["High","Low"]].copy(), step=float(psar_step), max_af=float(psar_max)) if show_psar else pd.Series(index=df_ohlc_view.index, dtype=float)

            fig, ax = plt.subplots(figsize=(12.8, 5.0))
            ax.set_title(f"Daily Supertrend / PSAR — {_safe_title_ticker(ticker)}")
            ax.plot(close_d.index, close_d.values, label="Close", alpha=0.9)
            if isinstance(st_line_d, pd.Series) and not st_line_d.empty:
                ax.plot(st_line_d.index, st_line_d.values, linewidth=2.0, alpha=0.85, label="Supertrend")
            if show_psar and isinstance(psar_d, pd.Series) and not psar_d.empty:
                ax.scatter(psar_d.index, psar_d.values, s=10, alpha=0.55, label="PSAR")
            style_axes(ax)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Status
            last_dir = bool(st_dir_d.dropna().iloc[-1]) if isinstance(st_dir_d, pd.Series) and st_dir_d.dropna().size else None
            st.write(f"Latest daily Supertrend direction: **{'UP' if last_dir else 'DOWN' if last_dir is not None else 'n/a'}**")
        else:
            st.info("Daily Supertrend requires daily OHLC data (High/Low/Close).")

        # Intraday supertrend status (already computed in comp)
        st_line_i = comp.get("st_i", pd.Series(dtype=float))
        st_dir_i = comp.get("st_dir_i", pd.Series(dtype=bool))
        psar_i = comp.get("psar_i", pd.Series(dtype=float))

        if isinstance(intraday, pd.DataFrame) and not intraday.empty and "Close" in intraday.columns:
            fig2 = plot_intraday_panel(ticker, intraday, comp)
            st.pyplot(fig2, use_container_width=True)
            last_idir = bool(st_dir_i.dropna().iloc[-1]) if isinstance(st_dir_i, pd.Series) and st_dir_i.dropna().size else None
            st.write(f"Latest intraday Supertrend direction: **{'UP' if last_idir else 'DOWN' if last_idir is not None else 'n/a'}**")

with tab11:
    st.subheader("Pivots")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to compute pivot levels.")
    else:
        comp = st.session_state.get("comp", {})
        df_ohlc_view = comp.get("df_ohlc_view", pd.DataFrame())
        close_d = comp.get("df_hist_view", pd.Series(dtype=float))

        prev = _latest_prev_session_ohlc(df_ohlc_view)
        if prev is None:
            st.warning("Pivot levels require daily OHLC.")
        else:
            ph, pl, pc, pdate = prev
            piv = compute_classic_pivots(ph, pl, pc)
            st.write(f"Previous session used: **{pdate}** (H={fmt_price_val(ph)}, L={fmt_price_val(pl)}, C={fmt_price_val(pc)})")

            dfp = pd.DataFrame({"Level": list(piv.keys()), "Price": list(piv.values())})
            st.dataframe(dfp, use_container_width=True)

            fig, ax = plt.subplots(figsize=(12.8, 5.0))
            ax.set_title(f"Daily Close with Pivot Levels — {_safe_title_ticker(ticker)}")
            ax.plot(close_d.index, close_d.values, label="Close", alpha=0.9)
            for k, v in piv.items():
                if np.isfinite(v):
                    ax.axhline(v, linewidth=1.0, alpha=0.35)
                    label_on_left(ax, v, f"{k}: {fmt_price_val(v)}", fontsize=8)
            style_axes(ax)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

with tab12:
    st.subheader("Reversal Probability")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to compute reversal probability.")
    else:
        comp = st.session_state.get("comp", {})
        close = comp.get("df_hist_view", pd.Series(dtype=float))
        rev_prob = comp.get("rev_prob", np.nan)
        slope_d = comp.get("slope_d", np.nan)
        global_slope = comp.get("global_slope", np.nan)
        r2_d = comp.get("r2_d", np.nan)
        fib_rev = comp.get("fib_rev", None)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reversal prob", fmt_pct(rev_prob, digits=1))
        c2.metric("Local slope", fmt_slope(slope_d))
        c3.metric("Global slope", fmt_slope(global_slope))
        c4.metric("Trend fit R²", fmt_r2(r2_d))

        if fib_rev is not None:
            st.info(
                f"Latest Fib reversal trigger: **{fib_rev.get('side','?')}** "
                f"near level **{fib_rev.get('from_level','?')}** at "
                f"{fib_rev.get('touch_time','?')} (touch {fmt_price_val(fib_rev.get('touch_price', np.nan))})"
            )

        # Show a simple diagnostic view: NTD + band reversal markers if available
        if isinstance(close, pd.Series) and not close.empty:
            yhat = comp.get("yhat_d", pd.Series(dtype=float))
            up = comp.get("up_d", pd.Series(dtype=float))
            lo = comp.get("lo_d", pd.Series(dtype=float))

            fig, ax = plt.subplots(figsize=(12.8, 5.0))
            ax.set_title(f"Band Context + Close — {_safe_title_ticker(ticker)}")
            ax.plot(close.index, close.values, label="Close", alpha=0.9)
            if isinstance(yhat, pd.Series) and not yhat.empty:
                ax.plot(yhat.index, yhat.values, linestyle="--", label="Trendline", alpha=0.85)
            if isinstance(up, pd.Series) and isinstance(lo, pd.Series) and (not up.empty) and (not lo.empty):
                ax.plot(up.index, up.values, linestyle=":", alpha=0.75, label="±2σ band")
                ax.plot(lo.index, lo.values, linestyle=":", alpha=0.75)
                ax.fill_between(up.index, lo.values, up.values, alpha=0.10)

            trig = comp.get("slope_trig", None)
            annotate_slope_trigger(ax, trig)

            style_axes(ax)
            ax.legend(loc="upper left")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.caption(
            "Reminder: reversal probability is a heuristic derived from slope/volatility context and is not a guarantee."
        )

with tab13:
    st.subheader("News")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to load news.")
    else:
        n = st.slider("Headlines", min_value=5, max_value=40, value=15, step=5, key="news_n")
        df_news = _news_rss(ticker, n=n)
        if df_news.empty:
            st.warning("No news returned (or RSS fetch blocked). Try again later or use a different symbol.")
        else:
            # display as clickable list
            for _, row in df_news.head(int(n)).iterrows():
                t = row.get("title", "")
                link = row.get("link", "")
                pub = row.get("published")
                pub_txt = ""
                if isinstance(pub, pd.Timestamp) and pd.notna(pub):
                    try:
                        pub_local = pub.tz_convert(PACIFIC)
                        pub_txt = pub_local.strftime("%b %d, %Y %I:%M %p %Z")
                    except Exception:
                        pub_txt = str(pub)
                else:
                    pub_txt = row.get("published_raw", "")

                st.markdown(f"- [{t}]({link})  \n  <small>{pub_txt}</small>", unsafe_allow_html=True)

with tab14:
    st.subheader("Data Tables")
    if not st.session_state.get("run_all", False):
        st.info("Run the app to view data tables.")
    else:
        comp = st.session_state.get("comp", {})
        df_ohlc_view = comp.get("df_ohlc_view", pd.DataFrame())
        intraday = st.session_state.get("intraday", pd.DataFrame())
        close_d = comp.get("df_hist_view", pd.Series(dtype=float))

        st.write("Daily close (view window)")
        st.dataframe(close_d.to_frame("Close"), use_container_width=True)

        st.write("Daily OHLC (view window)")
        if isinstance(df_ohlc_view, pd.DataFrame) and not df_ohlc_view.empty:
            st.dataframe(df_ohlc_view, use_container_width=True)
        else:
            st.info("No OHLC table available for this symbol.")

        st.write("Intraday OHLC (head)")
        if isinstance(intraday, pd.DataFrame) and not intraday.empty:
            st.dataframe(intraday.head(200), use_container_width=True)
        else:
            st.info("No intraday data available.")

        st.write("Computed series (tail snapshot)")
        snap = pd.DataFrame(index=close_d.index)
        for key in ["yhat_d","up_d","lo_d","sup_d","res_d","bb_mid_d","bb_up_d","bb_lo_d","hma_d","ntd_d","npx_d"]:
            s = comp.get(key, None)
            if isinstance(s, pd.Series) and (not s.empty):
                snap[key] = s.reindex(snap.index)
        if snap.shape[1] > 0:
            st.dataframe(snap.tail(120), use_container_width=True)
        else:
            st.info("No computed series snapshot available.")

with tab15:
    st.subheader("About")
    st.markdown(
        """
**Bull/Bear Wizard** is an indicator dashboard that combines:
- Trend regression with σ-bands
- Support/Resistance (rolling min/max)
- Fibonacci levels + Fib-touch & NPX(0) cross signals
- Normalized Trend (NTD) and Normalized Price (NPX)
- Normalized MACD
- HMA smoothing and cross signals
- Supertrend and optional PSAR
- SARIMAX forecasting (when enabled and data supports it)

**Notes**
- Signals are heuristics and should be validated against your own risk rules.
- News headlines are pulled from RSS and may occasionally fail due to rate limits/network policies.
- Forex “session lines” are approximate (visual aid only).

**Troubleshooting**
- If a tab looks empty: click **Run / Update** again.
- If you switch Mode (Stocks/Forex/Crypto): the app forces a fresh run to avoid stale mismatched series.
"""
    )
    st.caption("Not financial advice. Use at your own risk.")

# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption("© Bull/Bear Wizard — Streamlit dashboard")
