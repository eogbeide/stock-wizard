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
# Support / Resistance + Supertrend + PSAR
# ---------------------------
def rolling_support_resistance(close: pd.Series, lookback: int = 60):
    c = _coerce_1d_series(close).astype(float)
    if c.empty:
        empty = pd.Series(index=c.index, dtype=float)
        return empty, empty
    lb = max(5, int(lookback))
    support = c.rolling(lb, min_periods=max(3, lb // 3)).min()
    resist  = c.rolling(lb, min_periods=max(3, lb // 3)).max()
    return support.reindex(c.index), resist.reindex(c.index)

def true_range(ohlc: pd.DataFrame) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)
    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    c = pd.to_numeric(ohlc["Close"], errors="coerce")
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = true_range(ohlc)
    if tr.empty:
        return pd.Series(index=ohlc.index if ohlc is not None else None, dtype=float)
    p = max(2, int(period))
    return tr.ewm(alpha=1/p, adjust=False).mean()

def compute_supertrend(ohlc: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """
    Returns:
      supertrend line (float series),
      direction (+1 uptrend, -1 downtrend)
    """
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        idx = ohlc.index if ohlc is not None else None
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)

    h = pd.to_numeric(ohlc["High"], errors="coerce")
    l = pd.to_numeric(ohlc["Low"], errors="coerce")
    c = pd.to_numeric(ohlc["Close"], errors="coerce")

    a = atr(ohlc, period=period)
    hl2 = (h + l) / 2.0
    upper = hl2 + float(mult) * a
    lower = hl2 - float(mult) * a

    st_line = pd.Series(index=ohlc.index, dtype=float)
    dirn = pd.Series(index=ohlc.index, dtype=float)

    # initialize
    st_line.iloc[0] = upper.iloc[0]
    dirn.iloc[0] = -1.0

    for i in range(1, len(ohlc.index)):
        prev_st = st_line.iloc[i-1]
        prev_dir = dirn.iloc[i-1]
        cu = upper.iloc[i]
        cl = lower.iloc[i]

        if prev_dir > 0:
            st = max(cl, prev_st) if np.isfinite(prev_st) else cl
        else:
            st = min(cu, prev_st) if np.isfinite(prev_st) else cu

        # trend switch rules
        if np.isfinite(c.iloc[i]) and np.isfinite(st):
            if prev_dir < 0 and c.iloc[i] > st:
                prev_dir = 1.0
                st = cl
            elif prev_dir > 0 and c.iloc[i] < st:
                prev_dir = -1.0
                st = cu

        st_line.iloc[i] = st
        dirn.iloc[i] = prev_dir

    return st_line, dirn

def compute_psar(ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.2):
    """
    Basic Parabolic SAR implementation.
    Returns psar series.
    """
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        idx = ohlc.index if ohlc is not None else None
        return pd.Series(index=idx, dtype=float)

    h = pd.to_numeric(ohlc["High"], errors="coerce").to_numpy()
    l = pd.to_numeric(ohlc["Low"], errors="coerce").to_numpy()
    c = pd.to_numeric(ohlc["Close"], errors="coerce").to_numpy()
    idx = ohlc.index

    psar = np.full(len(idx), np.nan, dtype=float)

    # start direction from first movement
    bull = True
    if len(c) >= 2 and np.isfinite(c[0]) and np.isfinite(c[1]) and c[1] < c[0]:
        bull = False

    af = float(step)
    ep = h[0] if bull else l[0]
    psar[0] = l[0] if bull else h[0]

    for i in range(1, len(idx)):
        prev_psar = psar[i-1]
        if not np.isfinite(prev_psar):
            prev_psar = l[i-1] if bull else h[i-1]

        # compute next psar
        psar_i = prev_psar + af * (ep - prev_psar)

        # clamp
        if bull:
            psar_i = min(psar_i, l[i-1])
            if i >= 2:
                psar_i = min(psar_i, l[i-2])
        else:
            psar_i = max(psar_i, h[i-1])
            if i >= 2:
                psar_i = max(psar_i, h[i-2])

        # switch?
        if bull:
            if np.isfinite(l[i]) and l[i] < psar_i:
                bull = False
                psar_i = ep
                af = float(step)
                ep = l[i]
            else:
                if np.isfinite(h[i]) and h[i] > ep:
                    ep = h[i]
                    af = min(float(max_step), af + float(step))
        else:
            if np.isfinite(h[i]) and h[i] > psar_i:
                bull = True
                psar_i = ep
                af = float(step)
                ep = h[i]
            else:
                if np.isfinite(l[i]) and l[i] < ep:
                    ep = l[i]
                    af = min(float(max_step), af + float(step))

        psar[i] = psar_i

    return pd.Series(psar, index=idx)

def add_session_markers(ax, idx: pd.DatetimeIndex):
    """
    London & New York session OPEN markers in PST.
    Kept lightweight + approximate; no external calls.
    """
    if not show_sessions_pst or not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return

    # approximate opens in PST (varies with DST; keep simple for usability)
    # London open ~ 00:00 PST, NY open ~ 05:00 PST, NY close ~ 14:00 PST
    london_open_h = 0
    ny_open_h = 5
    ny_close_h = 14

    seen_days = set()
    for t in idx:
        d = t.date()
        if d in seen_days:
            continue
        seen_days.add(d)
        try:
            t0 = datetime(d.year, d.month, d.day, london_open_h, 0, tzinfo=PACIFIC)
            t1 = datetime(d.year, d.month, d.day, ny_open_h, 0, tzinfo=PACIFIC)
            t2 = datetime(d.year, d.month, d.day, ny_close_h, 0, tzinfo=PACIFIC)
        except Exception:
            continue

        for tt, lab in [(t0, "London"), (t1, "NY Open"), (t2, "NY Close")]:
            # only draw if within window
            if tt < idx.min() or tt > idx.max():
                continue
            ax.axvline(tt, linestyle="--", linewidth=1.0, alpha=0.35)
            ax.text(tt, ax.get_ylim()[1], f" {lab}", va="top", fontsize=8, alpha=0.6)

# ---------------------------
# Hourly fetcher
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hourly_ohlc(ticker: str, period: str = "730d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="60m")
    if df is None or df.empty:
        return df
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)
    df = df.dropna(subset=["Close"])
    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df

# ---------------------------
# (Optional) FX news markers — best-effort stub (won't break if unavailable)
# ---------------------------
@st.cache_data(ttl=3600)
def fetch_fx_news_markers(_symbol: str, days: int = 7):
    """
    Placeholder to keep UI consistent.
    Returns [] (no markers) by default.
    You can later wire this to a calendar API; app won't crash meanwhile.
    """
    return []


# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Plotting helpers
# ---------------------------
def _legend_dedupe(ax):
    h, l = ax.get_legend_handles_labels()
    seen = set()
    hh, ll = [], []
    for hi, li in zip(h, l):
        if li in seen or li.strip() == "":
            continue
        seen.add(li)
        hh.append(hi)
        ll.append(li)
    if hh:
        ax.legend(hh, ll, loc="best")

def plot_daily_panel(symbol: str,
                     ohlc: pd.DataFrame,
                     slope_lb: int,
                     sr_lb: int,
                     show_fibs_flag: bool,
                     show_ntd_flag: bool):
    if ohlc is None or ohlc.empty:
        st.warning("No daily data available.")
        return

    close = ohlc["Close"].copy()
    close = pd.to_numeric(close, errors="coerce").dropna()

    # Global trend (full history) + Local regression (lookback)
    yhat_full, global_slope = slope_line(close, lookback=len(close))
    yhat, upper, lower, local_slope, r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)

    # Support / resistance
    sup, res = rolling_support_resistance(close, lookback=int(sr_lb))

    # BBands
    bb_mid, bb_u, bb_l, bb_pctb, bb_nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    # HMA
    hma = compute_hma(close, period=int(hma_period))

    # Ichimoku (Kijun on price)
    kijun = None
    if show_ichi:
        hh = ohlc["High"].rolling(int(ichi_base), min_periods=max(5, int(ichi_base)//3)).max()
        ll = ohlc["Low"].rolling(int(ichi_base), min_periods=max(5, int(ichi_base)//3)).min()
        kijun = (hh + ll) / 2.0

    # NTD/NPX
    ntd = compute_normalized_trend(close, window=int(ntd_window)) if show_ntd_flag else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close, window=int(ntd_window)) if (show_ntd_flag and show_npx_ntd) else pd.Series(index=close.index, dtype=float)

    # Fibonacci + NPX-cross signals (NEW)
    fib_buy_mask = fib_sell_mask = pd.Series(False, index=close.index)
    fibs = {}
    if show_fibs_flag and show_ntd_flag and show_npx_ntd:
        fib_buy_mask, fib_sell_mask, fibs = fib_npx_zero_cross_signal_masks(
            price=close, npx=npx,
            horizon_bars=int(rev_horizon),
            proximity_pct_of_range=float(sr_prox_pct),
            npx_level=0.0
        )

    # Band bounce signal (legacy)
    band_sig = find_band_bounce_signal(close, upper, lower, local_slope)

    # Slope trigger after band touch (NEW)
    slope_trig = find_slope_trigger_after_band_reversal(close, yhat, upper, lower, horizon=int(rev_horizon))

    # MACD / HMA / S-R signal (hourly-oriented but useful daily too)
    macd_line, macd_sig, macd_hist = compute_macd(close)
    macd_sr_sig = find_macd_hma_sr_signal(
        close=close, hma=hma, macd=macd_line,
        sup=sup, res=res,
        global_trend_slope=float(global_slope),
        prox=float(sr_prox_pct)
    )

    # Reversal probability (experimental)
    revp = slope_reversal_probability(
        close,
        current_slope=float(local_slope),
        hist_window=int(rev_hist_lb),
        slope_window=int(slope_lb),
        horizon=int(rev_horizon)
    )

    # Trade instruction uses both slopes (updated behavior)
    trade_text = format_trade_instruction(
        trend_slope=float(local_slope),
        buy_val=float(sup.iloc[-1]) if len(sup) else float(close.iloc[-1]),
        sell_val=float(res.iloc[-1]) if len(res) else float(close.iloc[-1]),
        close_val=float(close.iloc[-1]),
        symbol=symbol,
        global_trend_slope=float(global_slope)
    )

    # --------- PRICE CHART ----------
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_title(f"{symbol} — Daily Price (view: {daily_view})")

    # plot series
    ax.plot(close.index, close.values, label="Close", alpha=0.95)

    if len(yhat):
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trend (slope {fmt_slope(local_slope)} | R² {fmt_r2(r2)})")
        ax.plot(upper.index, upper.values, linestyle=":", alpha=0.9, label="+2σ band")
        ax.plot(lower.index, lower.values, linestyle=":", alpha=0.9, label="-2σ band")

    # Support/resistance
    ax.plot(sup.index, sup.values, linestyle="--", alpha=0.75, label=f"Support ({sr_lb} bars)")
    ax.plot(res.index, res.values, linestyle="--", alpha=0.75, label=f"Resistance ({sr_lb} bars)")

    # BBands
    if show_bbands and len(bb_mid.dropna()):
        ax.plot(bb_mid.index, bb_mid.values, alpha=0.85, label="BB mid")
        ax.plot(bb_u.index, bb_u.values, alpha=0.65, label="BB upper")
        ax.plot(bb_l.index, bb_l.values, alpha=0.65, label="BB lower")

    # HMA
    if show_hma and len(hma.dropna()):
        ax.plot(hma.index, hma.values, alpha=0.9, label=f"HMA({hma_period})")

    # Kijun
    if show_ichi and kijun is not None and len(pd.Series(kijun).dropna()):
        ax.plot(kijun.index, kijun.values, alpha=0.9, label=f"Kijun({ichi_base})")

    # Fibonacci levels
    if show_fibs_flag:
        fibs_here = fibonacci_levels(close if daily_view == "Historical" else subset_by_daily_view(close, daily_view))
        if fibs_here:
            for k, v in fibs_here.items():
                ax.axhline(v, linestyle=":", linewidth=1.0, alpha=0.35)
            ax.text(close.index[-1], float(list(fibs_here.values())[0]), " Fib", fontsize=8, alpha=0.55)

    # Markers: band bounce, slope trigger, fib signals, macd signal
    if band_sig is not None:
        annotate_crossover(ax, band_sig["time"], band_sig["price"], band_sig["side"], note="(band bounce)")

    if slope_trig is not None:
        annotate_slope_trigger(ax, slope_trig)

    if show_fibs_flag and (fib_buy_mask.any() or fib_sell_mask.any()):
        overlay_fib_npx_signals(ax, close, fib_buy_mask, fib_sell_mask)

    if macd_sr_sig is not None:
        annotate_macd_signal(ax, macd_sr_sig["time"], macd_sr_sig["price"], macd_sr_sig["side"])

    # Labels on left: last price + support/resistance
    last_px = float(close.iloc[-1]) if len(close) else np.nan
    if np.isfinite(last_px):
        label_on_left(ax, last_px, f"Last {fmt_price_val(last_px)}", fontsize=9)

    if len(sup) and np.isfinite(float(sup.iloc[-1])):
        label_on_left(ax, float(sup.iloc[-1]), f"S {fmt_price_val(float(sup.iloc[-1]))}", fontsize=8)
    if len(res) and np.isfinite(float(res.iloc[-1])):
        label_on_left(ax, float(res.iloc[-1]), f"R {fmt_price_val(float(res.iloc[-1]))}", fontsize=8)

    ax.set_xlabel("")
    ax.set_ylabel("Price")
    style_axes(ax)
    _legend_dedupe(ax)

    st.pyplot(fig, use_container_width=True)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Local slope", fmt_slope(local_slope))
    m2.metric("Global slope", fmt_slope(global_slope))
    m3.metric("R² (local fit)", fmt_r2(r2))
    m4.metric("Reversal prob (exp.)", fmt_pct(revp, digits=1))

    st.info(trade_text)

    # Fib reversal trigger banner (existing)
    if show_fibs_flag:
        fib_tr = fib_reversal_trigger_from_extremes(
            close,
            proximity_pct_of_range=float(sr_prox_pct),
            confirm_bars=2,
            lookback_bars=max(60, int(sr_lb))
        )
        if fib_tr is not None:
            st.warning(
                f"{FIB_ALERT_TEXT}\n\n"
                f"**{fib_tr['side']} confirmed** from Fib {fib_tr['from_level']} "
                f"(touch {fib_tr['touch_time']:%Y-%m-%d}, last {fib_tr['last_time']:%Y-%m-%d})."
            )

    # --------- NTD PANEL ----------
    if show_ntd_flag:
        fig2, ax2 = plt.subplots(figsize=(12, 3.3))
        ax2.set_title(f"{symbol} — Daily NTD (window {ntd_window})")
        ax2.axhline(0.0, linewidth=1.0, alpha=0.35)
        ax2.plot(ntd.index, ntd.values, label="NTD", alpha=0.9)

        if shade_ntd:
            shade_ntd_regions(ax2, ntd)

        if show_npx_ntd:
            ax2.plot(npx.index, npx.values, label="NPX", alpha=0.75)
            if mark_npx_cross:
                up0, dn0 = npx_zero_cross_masks(npx, level=0.0)
                ax2.scatter(npx.index[up0.fillna(False)], npx[up0.fillna(False)], marker="o", s=30, alpha=0.8, label="NPX cross up")
                ax2.scatter(npx.index[dn0.fillna(False)], npx[dn0.fillna(False)], marker="o", s=30, alpha=0.8, label="NPX cross dn")

        ax2.set_ylim(-1.05, 1.05)
        style_axes(ax2)
        _legend_dedupe(ax2)
        st.pyplot(fig2, use_container_width=True)

def plot_hourly_panel(symbol: str,
                      ohlc_h: pd.DataFrame,
                      slope_lb: int,
                      sr_lb: int):
    if ohlc_h is None or ohlc_h.empty:
        st.warning("No hourly data available.")
        return

    close = pd.to_numeric(ohlc_h["Close"], errors="coerce").dropna()
    if close.empty:
        st.warning("Hourly close series is empty.")
        return

    # Local regression with band
    yhat, upper, lower, local_slope, r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)
    sup, res = rolling_support_resistance(close, lookback=int(sr_lb))

    # NTD / NPX
    ntd = compute_normalized_trend(close, window=int(ntd_window))
    npx = compute_normalized_price(close, window=int(ntd_window))

    # Supertrend + PSAR
    st_line, st_dir = compute_supertrend(ohlc_h, period=int(atr_period), mult=float(atr_mult))
    psar = compute_psar(ohlc_h, step=float(psar_step), max_step=float(psar_max)) if show_psar else pd.Series(index=ohlc_h.index, dtype=float)

    # MACD/HMA
    hma = compute_hma(close, period=int(hma_period))
    nmacd, nsignal, nhist = compute_nmacd(close)
    macd_sr_sig = find_macd_hma_sr_signal(
        close=close, hma=hma, macd=(nmacd if len(nmacd.dropna()) else pd.Series(index=close.index, dtype=float)),
        sup=sup, res=res,
        global_trend_slope=float(local_slope),  # for hourly we use local as “global” proxy
        prox=float(sr_prox_pct)
    )

    # Momentum
    roc = compute_roc(close, n=int(mom_lb_hourly))

    # PRICE CHART
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_title(f"{symbol} — Hourly Price")

    ax.plot(close.index, close.values, label="Close", alpha=0.95)

    if len(yhat):
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trend (slope {fmt_slope(local_slope)} | R² {fmt_r2(r2)})")
        ax.plot(upper.index, upper.values, linestyle=":", alpha=0.85, label="+2σ band")
        ax.plot(lower.index, lower.values, linestyle=":", alpha=0.85, label="-2σ band")

    ax.plot(sup.index, sup.values, linestyle="--", alpha=0.75, label=f"Support ({sr_lb} bars)")
    ax.plot(res.index, res.values, linestyle="--", alpha=0.75, label=f"Resistance ({sr_lb} bars)")

    if show_hma and len(hma.dropna()):
        ax.plot(hma.index, hma.values, alpha=0.9, label=f"HMA({hma_period})")

    if len(st_line.dropna()):
        ax.plot(st_line.index, st_line.values, alpha=0.9, label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and len(psar.dropna()):
        ax.scatter(psar.index, psar.values, s=10, alpha=0.7, label="PSAR")

    if macd_sr_sig is not None:
        annotate_macd_signal(ax, macd_sr_sig["time"], macd_sr_sig["price"], macd_sr_sig["side"])

    style_axes(ax)
    _legend_dedupe(ax)
    st.pyplot(fig, use_container_width=True)

    # NTD / Indicators
    c1, c2 = st.columns(2)

    with c1:
        fig2, ax2 = plt.subplots(figsize=(12, 3.2))
        ax2.set_title(f"{symbol} — Hourly NTD (window {ntd_window})")
        ax2.axhline(0.0, linewidth=1.0, alpha=0.35)
        ax2.plot(ntd.index, ntd.values, label="NTD", alpha=0.9)
        if shade_ntd:
            shade_ntd_regions(ax2, ntd)

        if show_npx_ntd:
            ax2.plot(npx.index, npx.values, label="NPX", alpha=0.75)
            if mark_npx_cross:
                up0, dn0 = npx_zero_cross_masks(npx, level=0.0)
                ax2.scatter(npx.index[up0.fillna(False)], npx[up0.fillna(False)], marker="o", s=25, alpha=0.85, label="NPX up")
                ax2.scatter(npx.index[dn0.fillna(False)], npx[dn0.fillna(False)], marker="o", s=25, alpha=0.85, label="NPX dn")

        # NTD channel highlight: when NTD between support/resist zone in NTD-space
        if show_ntd_channel:
            # convert S/R to NTD-space by mapping price zscore to tanh(z/2)
            sup_npx = compute_normalized_price(sup, window=int(ntd_window))
            res_npx = compute_normalized_price(res, window=int(ntd_window))
            ok = sup_npx.notna() & res_npx.notna() & ntd.notna()
            if ok.any():
                lo = np.minimum(sup_npx[ok], res_npx[ok])
                hi = np.maximum(sup_npx[ok], res_npx[ok])
                inside = (npx[ok] >= lo) & (npx[ok] <= hi)
                ax2.fill_between(npx[ok].index, -1.0, 1.0, where=inside.to_numpy(), alpha=0.06)

        ax2.set_ylim(-1.05, 1.05)
        style_axes(ax2)
        _legend_dedupe(ax2)
        st.pyplot(fig2, use_container_width=True)

    with c2:
        if show_mom_hourly:
            fig3, ax3 = plt.subplots(figsize=(12, 3.2))
            ax3.set_title(f"{symbol} — Hourly Momentum (ROC%, {mom_lb_hourly} bars)")
            ax3.axhline(0.0, linewidth=1.0, alpha=0.35)
            ax3.plot(roc.index, roc.values, label="ROC%", alpha=0.9)
            style_axes(ax3)
            _legend_dedupe(ax3)
            st.pyplot(fig3, use_container_width=True)
        else:
            fig3, ax3 = plt.subplots(figsize=(12, 3.2))
            ax3.set_title(f"{symbol} — Hourly Normalized MACD")
            ax3.axhline(0.0, linewidth=1.0, alpha=0.35)
            ax3.plot(nmacd.index, nmacd.values, label="nMACD", alpha=0.9)
            ax3.plot(nsignal.index, nsignal.values, label="nSignal", alpha=0.8)
            style_axes(ax3)
            _legend_dedupe(ax3)
            st.pyplot(fig3, use_container_width=True)

    # quick hourly metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Hourly slope", fmt_slope(local_slope))
    m2.metric("Hourly R²", fmt_r2(r2))
    m3.metric("Last close", fmt_price_val(float(close.iloc[-1])) if len(close) else "n/a")


def plot_intraday_panel(symbol: str, intraday: pd.DataFrame):
    if intraday is None or intraday.empty:
        st.warning("No intraday data available.")
        return

    ohlc = intraday.copy()
    if not {"Open","High","Low","Close"}.issubset(ohlc.columns):
        st.warning("Intraday data missing OHLC columns.")
        return

    close = pd.to_numeric(ohlc["Close"], errors="coerce").dropna()
    if close.empty:
        st.warning("Intraday close is empty.")
        return

    # regression on intraday
    yhat, upper, lower, local_slope, r2 = regression_with_band(close, lookback=min(len(close), int(slope_lb_hourly)), z=2.0)

    # S/R
    sup, res = rolling_support_resistance(close, lookback=int(sr_lb_hourly))

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_title(f"{symbol} — Intraday (gapless)")

    # use bar positions for nicer ticks on dense series
    # (we keep true time on x-axis, but formatting compact)
    ax.plot(close.index, close.values, label="Close", alpha=0.95)
    if len(yhat):
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Trend (slope {fmt_slope(local_slope)} | R² {fmt_r2(r2)})")
        ax.plot(upper.index, upper.values, linestyle=":", alpha=0.85, label="+2σ")
        ax.plot(lower.index, lower.values, linestyle=":", alpha=0.85, label="-2σ")

    ax.plot(sup.index, sup.values, linestyle="--", alpha=0.75, label=f"Support ({sr_lb_hourly})")
    ax.plot(res.index, res.values, linestyle="--", alpha=0.75, label=f"Resistance ({sr_lb_hourly})")

    # sessions (PST)
    add_session_markers(ax, close.index)

    # FX news markers (optional)
    if mode == "Forex" and show_fx_news:
        markers = fetch_fx_news_markers(symbol, days=int(news_window_days))
        # expected format: [{"time": datetime, "label": "CPI"}, ...]
        for ev in markers:
            try:
                t = pd.to_datetime(ev.get("time")).tz_convert(PACIFIC)
                lab = str(ev.get("label", "News"))
                if close.index.min() <= t <= close.index.max():
                    ax.axvline(t, linestyle=":", linewidth=1.0, alpha=0.35)
                    ax.text(t, ax.get_ylim()[0], f" {lab}", va="bottom", fontsize=8, alpha=0.6)
            except Exception:
                continue

    style_axes(ax)
    _legend_dedupe(ax)
    st.pyplot(fig, use_container_width=True)

    # trade instruction (intraday uses local slope only)
    trade_text = format_trade_instruction(
        trend_slope=float(local_slope),
        buy_val=float(sup.iloc[-1]) if len(sup) else float(close.iloc[-1]),
        sell_val=float(res.iloc[-1]) if len(res) else float(close.iloc[-1]),
        close_val=float(close.iloc[-1]),
        symbol=symbol,
        global_trend_slope=None
    )
    st.info(trade_text)


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Main selection + Run button
# ---------------------------
ticker = st.sidebar.selectbox("Select instrument:", universe, key="sb_ticker")
st.session_state.ticker = ticker

# Intraday range choice
if mode == "Forex":
    intraday_period = st.sidebar.selectbox("Intraday range:", ["1d", "5d"], index=1, key="sb_intraday_period")
else:
    intraday_period = st.sidebar.selectbox("Intraday range:", ["1d", "5d"], index=0, key="sb_intraday_period")

run = st.sidebar.button("▶ Run analysis", use_container_width=True, key="btn_run")

if run:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# If mode switched after run, don't reuse stale results
if st.session_state.get("run_all", False) and st.session_state.get("mode_at_run") != mode:
    _reset_run_state_for_mode_switch()

# ---------------------------
# Compute & cache run outputs in session_state
# ---------------------------
if st.session_state.get("run_all", False):
    with st.spinner("Fetching data + computing indicators..."):
        # daily
        df_ohlc = fetch_hist_ohlc(ticker)
        df_ohlc = subset_by_daily_view(df_ohlc, daily_view)
        st.session_state.df_ohlc = df_ohlc

        df_hist = df_ohlc["Close"] if df_ohlc is not None and not df_ohlc.empty else fetch_hist(ticker)
        st.session_state.df_hist = df_hist

        # forecast
        try:
            fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        except Exception:
            fc_idx, fc_vals, fc_ci = (pd.DatetimeIndex([]), pd.Series(dtype=float), pd.DataFrame())
        st.session_state.fc_idx = fc_idx
        st.session_state.fc_vals = fc_vals
        st.session_state.fc_ci = fc_ci

        # hourly
        ohlc_h = fetch_hourly_ohlc(ticker, period="730d")
        st.session_state.ohlc_h = ohlc_h

        # intraday
        intraday = fetch_intraday(ticker, period=intraday_period)
        st.session_state.intraday = intraday


# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Tabs layout (ribbon styled via CSS above)
# ---------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📅 Daily",
    "🕐 Hourly",
    "⚡ Intraday",
    "🔮 Forecast",
    "📦 Raw Data"
])

with tabs[0]:
    st.subheader("Overview")
    if not st.session_state.get("run_all", False):
        st.info("Choose an instrument on the left and click **Run analysis**.")
    else:
        df_hist = st.session_state.get("df_hist")
        df_ohlc = st.session_state.get("df_ohlc")

        last_px = _safe_last_float(df_hist)
        prev_px = float(df_hist.dropna().iloc[-2]) if df_hist is not None and df_hist.dropna().shape[0] >= 2 else np.nan
        chg = (last_px / prev_px - 1.0) if np.isfinite(last_px) and np.isfinite(prev_px) and prev_px != 0 else np.nan

        # Bull/Bear (simple): last close vs 200D SMA + recent up-days
        s = _coerce_1d_series(df_hist).dropna()
        sma200 = s.rolling(200, min_periods=50).mean()
        bull = bool(len(sma200.dropna()) and np.isfinite(last_px) and last_px >= float(sma200.dropna().iloc[-1]))
        up_days = (s.pct_change().dropna() > 0).tail(60).mean() if len(s) > 60 else (s.pct_change().dropna() > 0).mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last", fmt_price_val(last_px) if np.isfinite(last_px) else "n/a")
        m2.metric("1D change", fmt_pct(chg, digits=2))
        m3.metric("Bias (SMA200)", "Bullish" if bull else "Bearish")
        m4.metric("Up-days (last 60)", fmt_pct(up_days, digits=1))

        st.caption("Use **Daily / Hourly / Intraday** tabs for full indicator panels.")

with tabs[1]:
    st.subheader("Daily")
    if not st.session_state.get("run_all", False):
        st.info("Run analysis to view charts.")
    else:
        df_ohlc = st.session_state.get("df_ohlc")
        plot_daily_panel(
            symbol=ticker,
            ohlc=df_ohlc,
            slope_lb=int(slope_lb_daily),
            sr_lb=int(sr_lb_daily),
            show_fibs_flag=bool(show_fibs),
            show_ntd_flag=bool(show_ntd)
        )

with tabs[2]:
    st.subheader("Hourly")
    if not st.session_state.get("run_all", False):
        st.info("Run analysis to view charts.")
    else:
        ohlc_h = st.session_state.get("ohlc_h")
        plot_hourly_panel(
            symbol=ticker,
            ohlc_h=ohlc_h,
            slope_lb=int(slope_lb_hourly),
            sr_lb=int(sr_lb_hourly)
        )

with tabs[3]:
    st.subheader("Intraday")
    if not st.session_state.get("run_all", False):
        st.info("Run analysis to view charts.")
    else:
        intraday = st.session_state.get("intraday")
        plot_intraday_panel(ticker, intraday)

with tabs[4]:
    st.subheader("Forecast (SARIMAX)")
    if not st.session_state.get("run_all", False):
        st.info("Run analysis to view forecast.")
    else:
        hist = _coerce_1d_series(st.session_state.get("df_hist")).dropna()
        fc_idx = st.session_state.get("fc_idx")
        fc_vals = _coerce_1d_series(st.session_state.get("fc_vals"))
        fc_ci = st.session_state.get("fc_ci")

        if hist.empty or fc_vals.empty or fc_idx is None or len(fc_idx) == 0:
            st.warning("Forecast not available for this instrument at the moment.")
        else:
            fig, ax = plt.subplots(figsize=(12, 4.6))
            ax.set_title(f"{ticker} — 30D Forecast")
            ax.plot(hist.index, hist.values, label="History", alpha=0.9)
            ax.plot(fc_idx, fc_vals.values, linestyle="--", label="Forecast", alpha=0.95)

            # conf int
            if isinstance(fc_ci, pd.DataFrame) and fc_ci.shape[1] >= 2:
                lo = pd.to_numeric(fc_ci.iloc[:, 0], errors="coerce").to_numpy()
                hi = pd.to_numeric(fc_ci.iloc[:, 1], errors="coerce").to_numpy()
                ax.fill_between(fc_idx, lo, hi, alpha=0.12, label="CI")

            style_axes(ax)
            _legend_dedupe(ax)
            st.pyplot(fig, use_container_width=True)

            fc_tbl = pd.DataFrame({
                "Date": pd.to_datetime(fc_idx).tz_convert(PACIFIC).strftime("%Y-%m-%d"),
                "Forecast": np.round(fc_vals.values.astype(float), 4)
            })
            st.dataframe(fc_tbl, use_container_width=True, height=260)

with tabs[5]:
    st.subheader("Raw Data")
    if not st.session_state.get("run_all", False):
        st.info("Run analysis to view data.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Daily OHLC")
            st.dataframe(st.session_state.get("df_ohlc"), use_container_width=True, height=320)
        with c2:
            st.caption("Hourly OHLC")
            st.dataframe(st.session_state.get("ohlc_h"), use_container_width=True, height=320)

        st.caption("Intraday (gapless OHLC)")
        st.dataframe(st.session_state.get("intraday"), use_container_width=True, height=240)
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Small cleanup utilities + safety guards
# ---------------------------
def _safe_last_float(x) -> float:
    try:
        s = _coerce_1d_series(x).dropna()
        if len(s) == 0:
            return float("nan")
        v = float(s.iloc[-1])
        return v
    except Exception:
        return float("nan")

def subset_by_daily_view(obj, view: str):
    """
    Supports both Series and DataFrame with a DatetimeIndex.
    view examples: "Historical", "Last 15 years", "Last 10 years", "Last 5 years",
                   "Last 3 years", "Last 1 year"
    """
    if obj is None:
        return obj
    if not isinstance(getattr(obj, "index", None), pd.DatetimeIndex):
        return obj
    if obj.empty:
        return obj

    v = str(view or "").strip().lower()
    if v in ("historical", "all", ""):
        return obj

    years_map = {
        "last 15 years": 15,
        "last 10 years": 10,
        "last 5 years": 5,
        "last 3 years": 3,
        "last 1 year": 1,
        "last year": 1,
    }
    years = years_map.get(v, None)
    if years is None:
        return obj

    last_ts = obj.index.max()
    try:
        cutoff = last_ts - pd.DateOffset(years=int(years))
        return obj.loc[obj.index >= cutoff]
    except Exception:
        return obj

def _reset_run_state_for_mode_switch():
    # Clear cached results so charts don't mismatch when switching mode.
    st.session_state.run_all = False
    for k in ["df_ohlc", "df_hist", "fc_idx", "fc_vals", "fc_ci", "ohlc_h", "intraday", "mode_at_run"]:
        if k in st.session_state:
            del st.session_state[k]

# Default fib alert text if not defined earlier
if "FIB_ALERT_TEXT" not in globals():
    FIB_ALERT_TEXT = "⚠️ Fibonacci reversal alert (experimental). Use risk controls."

# Ensure PACIFIC is defined
try:
    PACIFIC
except NameError:
    PACIFIC = ZoneInfo("America/Los_Angeles")


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Footer + guardrails (non-breaking)
# ---------------------------
st.markdown("---")
st.caption(
    "Disclaimer: This app is for educational purposes only and is **not** financial advice. "
    "Indicators are heuristic/experimental and may be wrong. Always use position sizing, stops, and verify with multiple sources."
)

# Quick sanity note when the user hasn't run analysis yet
if not st.session_state.get("run_all", False):
    st.caption("Tip: Start with **SPY** (Index) or **EURUSD=X** (Forex) and click **Run analysis**.")

# If Yahoo sometimes returns duplicated columns or a MultiIndex, try to normalize (best effort)
def _normalize_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # flatten MultiIndex columns like ('Close', 'SPY')
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) and len(c) else str(c) for c in df.columns]
    # remove duplicate columns
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df

# Apply normalization to cached frames if present
if st.session_state.get("run_all", False):
    for key in ["df_ohlc", "ohlc_h", "intraday"]:
        df = st.session_state.get(key, None)
        if isinstance(df, pd.DataFrame):
            st.session_state[key] = _normalize_yf_frame(df)
