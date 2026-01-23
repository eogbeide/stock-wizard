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

# NEW (REQUESTED CHANGE #1 support helper): daily Supertrend series aligned to a given index (no UI change)
def supertrend_line_aligned(ohlc_df: pd.DataFrame,
                            align_index: pd.Index,
                            atr_period: int,
                            atr_mult: float) -> pd.Series:
    """
    Compute Supertrend on OHLC df and return the ST line reindexed to align_index.
    """
    if ohlc_df is None or ohlc_df.empty or not {"High","Low","Close"}.issubset(ohlc_df.columns):
        return pd.Series(index=align_index, dtype=float)
    st_df = compute_supertrend(ohlc_df, atr_period=int(atr_period), atr_mult=float(atr_mult))
    if st_df is None or st_df.empty or "ST" not in st_df.columns:
        return pd.Series(index=align_index, dtype=float)
    st_line = _coerce_1d_series(st_df["ST"]).reindex(align_index)
    return st_line

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

        # NOTE: (prior request) uses 0.5 cross level instead of 0.0
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
# NEW (REQUESTED CHANGE #2 helper): Regression Trend bounce scanner (Daily)
#   Conditions:
#     (a) Global trendline > 0 AND regression slope > 0 AND recent bounce from LOWER -2σ (up)
#     (b) Global trendline < 0 AND regression slope < 0 AND recent bounce from UPPER +2σ (down)
#   "Recent" uses existing sidebar setting via max_bars_since passed from caller (we will pass rev_horizon).
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_regression_trend_bounce(symbol: str,
                                       daily_view_label: str,
                                       slope_lb: int,
                                       max_bars_since: int = 15):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(5, int(slope_lb) + 2):
            return None

        # Global trendline slope on the same displayed range
        xg = np.arange(len(close_show), dtype=float)
        mg, bg = np.polyfit(xg, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(mg) or float(mg) == 0.0:
            return None

        # Regression band slope on slope_lb
        yhat, up, lo, mr, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not np.isfinite(mr) or float(mr) == 0.0:
            return None

        # Bounce from ±2σ (same logic as charts)
        sig = find_band_bounce_signal(close_show, up, lo, mr)
        if sig is None:
            return None

        t = sig.get("time", None)
        if t is None or t not in close_show.index:
            return None

        try:
            loc = int(close_show.index.get_loc(t))
        except Exception:
            return None
        bars_since = int((len(close_show) - 1) - loc)
        if int(bars_since) > int(max_bars_since):
            return None

        side = str(sig.get("side", "")).upper()
        if side not in ("BUY", "SELL"):
            return None

        # "going up / going down" confirmation at the signal bar
        prev_idx_ok = (loc - 1) >= 0
        if not prev_idx_ok:
            return None
        px_sig = float(close_show.iloc[loc]) if np.isfinite(close_show.iloc[loc]) else np.nan
        px_prev = float(close_show.iloc[loc - 1]) if np.isfinite(close_show.iloc[loc - 1]) else np.nan
        if not (np.isfinite(px_sig) and np.isfinite(px_prev)):
            return None

        if side == "BUY" and not (px_sig > px_prev):
            return None
        if side == "SELL" and not (px_sig < px_prev):
            return None

        # Apply the requested slope-sign filters
        mg = float(mg)
        mr = float(mr)

        if side == "BUY":
            if not (mg > 0.0 and mr > 0.0):
                return None
        else:
            if not (mg < 0.0 and mr < 0.0):
                return None

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Side": side,
            "Bars Since": int(bars_since),
            "Signal Time": t,
            "Signal Price": float(px_sig),
            "Current Price": float(curr_px) if np.isfinite(curr_px) else np.nan,
            "Global Slope": float(mg),
            "Regression Slope": float(mr),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Daily View": str(daily_view_label),
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

    # Touch masks in the same lookback window
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

    # "current" slope (now)
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
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Plotting helpers (matplotlib)
# ---------------------------
FAST_ST_ATR_PERIOD = 7
FAST_ST_ATR_MULT = 2.0
SMA8_PERIOD = 8

def sma(series_like: pd.Series, period: int = 20) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    return s.rolling(int(period), min_periods=1).mean()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_hist_ohlc(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Daily OHLC for indicators that need High/Low.
    Returns tz-aware index in Pacific time when possible.
    """
    try:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])

    # Standardize cols
    keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].copy()

    # Ensure DatetimeIndex and localize/convert
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            # yfinance daily often comes as naive (interpretable as UTC)
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass
        try:
            df.index = df.index.tz_convert(PACIFIC)
        except Exception:
            pass

    return df

def _plot_regression_channel(ax, close_show: pd.Series, slope_lb: int, show_band: bool = True):
    yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
    if yhat is None or yhat.dropna().empty:
        return None
    ax.plot(yhat.index, yhat.values, linewidth=1.6, label=f"Reg ({slope_lb})")
    if show_band and (up is not None) and (lo is not None):
        ax.plot(up.index, up.values, linewidth=1.1, alpha=0.9, label="+2σ")
        ax.plot(lo.index, lo.values, linewidth=1.1, alpha=0.9, label="-2σ")
    return {"yhat": yhat, "up": up, "lo": lo, "m": m, "r2": r2}

def _annotate_band_bounce(ax, close_show: pd.Series, band: dict):
    if not band:
        return None
    sig = find_band_bounce_signal(close_show, band.get("up"), band.get("lo"), band.get("m", np.nan))
    if not sig:
        return None
    t = sig.get("time", None)
    px = sig.get("price", np.nan)
    side = str(sig.get("side", "")).upper()
    if t is None or (not np.isfinite(px)):
        return None
    col = "tab:green" if side == "BUY" else "tab:red"
    marker = "^" if side == "BUY" else "v"
    ax.scatter([t], [px], s=140, marker=marker, color=col, zorder=15)
    ax.text(t, px, f"  {side}", color=col, fontsize=10, fontweight="bold",
            va="bottom" if side == "BUY" else "top", zorder=15)
    return sig

def _overlay_sma8(ax, close_show: pd.Series):
    sma8 = sma(close_show, SMA8_PERIOD)
    ax.plot(sma8.index, sma8.values, linewidth=1.4, alpha=0.95, label=f"SMA({SMA8_PERIOD})")

def _overlay_fast_supertrend_daily(ax, daily_ohlc: pd.DataFrame, close_show: pd.Series):
    """
    REQUESTED CHANGE:
      Fast Supertrend computed on DAILY OHLC (never intraday), then aligned to displayed close_show index.
    """
    if daily_ohlc is None or daily_ohlc.empty:
        return
    st_line = supertrend_line_aligned(
        daily_ohlc[["High","Low","Close"]],
        align_index=close_show.index,
        atr_period=FAST_ST_ATR_PERIOD,
        atr_mult=FAST_ST_ATR_MULT,
    )
    if st_line.dropna().empty:
        return
    ax.plot(st_line.index, st_line.values, linewidth=1.4, alpha=0.95, label=f"Fast ST ({FAST_ST_ATR_PERIOD},{FAST_ST_ATR_MULT:g})")

def _format_price_axis(ax, title: str, ylabel: str = "Price"):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

def _format_ntd_axis(ax):
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axhline(0.75, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.axhline(-0.75, linestyle="--", linewidth=1.0, alpha=0.45)
    ax.set_ylabel("NTD / NPX")
    ax.grid(True, alpha=0.25)

def build_daily_figure(symbol: str,
                       daily_view_label: str,
                       ntd_win: int,
                       slope_lb: int,
                       show_sessions: bool,
                       show_news: bool,
                       news_days: int,
                       show_inrange: bool,
                       show_npx: bool,
                       show_hma: bool,
                       hma_period: int,
                       hma_rev_lb: int,
                       show_fib: bool,
                       show_fast_st: bool,
                       show_sma8: bool):
    # Daily OHLC for any OHLC-based overlays (fast ST, PSAR if you add it later)
    ohlc = fetch_hist_ohlc(symbol, period="5y", interval="1d")
    if ohlc is None or ohlc.empty or "Close" not in ohlc.columns:
        return None, None

    close_full = _coerce_1d_series(ohlc["Close"]).dropna()
    close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
    if close_show.empty or len(close_show) < 5:
        return None, None

    # Indicators on full, then align to shown range
    ntd_full = compute_normalized_trend(close_full, window=int(ntd_win))
    npx_full = compute_normalized_price(close_full, window=int(ntd_win))
    ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
    npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

    # Prepare figure
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.07)
    ax_p = fig.add_subplot(gs[0])
    ax_n = fig.add_subplot(gs[1], sharex=ax_p)

    # Price
    ax_p.plot(close_show.index, close_show.values, linewidth=1.8, label="Close")

    # Regression channel + bounce markers
    band = _plot_regression_channel(ax_p, close_show, slope_lb=int(slope_lb), show_band=True)
    sig = _annotate_band_bounce(ax_p, close_show, band if band else None)

    # REQUESTED CHANGE: SMA(8)
    if bool(show_sma8):
        _overlay_sma8(ax_p, close_show)

    # REQUESTED CHANGE: fast daily Supertrend line
    if bool(show_fast_st):
        ohlc_show = ohlc.reindex(close_show.index)
        _overlay_fast_supertrend_daily(ax_p, ohlc_show, close_show)

    # Fib levels (optional)
    if bool(show_fib):
        fibs = fibonacci_levels(close_show)
        if fibs:
            for k, v in fibs.items():
                if np.isfinite(v):
                    ax_p.axhline(float(v), linestyle="--", linewidth=1.0, alpha=0.45, label=f"Fib {k}")

    # Sessions (only makes sense if index is tz-aware; daily still can show verticals but sparse)
    if bool(show_sessions) and isinstance(close_show.index, pd.DatetimeIndex) and close_show.index.tz is not None:
        lines = compute_session_lines(close_show.index)
        handles, labels = draw_session_lines(ax_p, lines, alpha=0.18)
        # add session legend handles without duplicating
        for h in handles:
            ax_p.plot([], [], linestyle=h.get_linestyle(), linewidth=h.get_linewidth(),
                      color=h.get_color(), alpha=0.65, label=h.get_label())

    # News markers (optional)
    if bool(show_news):
        news_df = fetch_yf_news(symbol, window_days=int(news_days))
        if news_df is not None and not news_df.empty:
            draw_news_markers(ax_p, list(news_df["time"].values), label=f"News (last {news_days}d)")

    _format_price_axis(ax_p, f"{symbol} — Daily ({daily_view_label})")

    # NTD panel
    ax_n.plot(ntd_show.index, ntd_show.values, linewidth=1.6, label=f"NTD ({ntd_win})")

    # In-range shading uses support/resistance from regression band
    if bool(show_inrange) and band and band.get("up") is not None and band.get("lo") is not None:
        overlay_inrange_on_ntd(ax_n, close_show, band["lo"], band["up"])

    # NPX overlay
    if bool(show_npx):
        overlay_npx_on_ntd(ax_n, npx_show, ntd_show, mark_crosses=True)

    # HMA + reversals
    if bool(show_hma):
        h = hma(close_show, period=int(hma_period))
        # plot HMA on price
        if h is not None and not _coerce_1d_series(h).dropna().empty:
            ax_p.plot(h.index, h.values, linewidth=1.4, alpha=0.9, label=f"HMA({hma_period})")
            overlay_hma_reversal_on_ntd(ax_n, close_show, h, lookback=int(hma_rev_lb),
                                       period=int(hma_period), ntd=ntd_show)

    _format_ntd_axis(ax_n)

    # Legend
    ax_p.legend(loc="upper left", fontsize=9, ncol=2)
    ax_n.legend(loc="upper left", fontsize=9, ncol=2)

    plt.setp(ax_p.get_xticklabels(), visible=False)
    return fig, {"close_show": close_show, "ntd_show": ntd_show, "npx_show": npx_show, "band": band, "sig": sig}


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Scanners / tables
# ---------------------------
def _symbols_from_text(raw: str):
    if raw is None:
        return []
    txt = str(raw).replace("\n", ",").replace(";", ",").replace("|", ",")
    syms = [s.strip().upper() for s in txt.split(",") if s.strip()]
    # keep order, unique
    out, seen = [], set()
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _df_from_rows(rows: list):
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def run_recent_buy_scans(symbols: list,
                         slope_lb: int,
                         ntd_win: int,
                         daily_view_label: str,
                         hourly_period: str,
                         max_bars_since: int):
    """
    Collects:
      - band bounce daily/hourly
      - NPX↑NTD uptrend daily
      - NPX 0.5 cross w/ local slope daily
    """
    out = {}

    # Band bounce (daily)
    rows = []
    for sym in symbols:
        r = last_band_bounce_signal_daily(sym, slope_lb=int(slope_lb))
        if r and int(r.get("Bars Since", 9999)) <= int(max_bars_since) and str(r.get("Side","")).upper() == "BUY":
            rows.append(r)
    df = _df_from_rows(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since","R2"], ascending=[True, False])
    out["Band Bounce BUY (Daily)"] = df

    # Band bounce (hourly)
    rows = []
    for sym in symbols:
        r = last_band_bounce_signal_hourly(sym, period=str(hourly_period), slope_lb=int(slope_lb))
        if r and int(r.get("Bars Since", 9999)) <= int(max_bars_since) and str(r.get("Side","")).upper() == "BUY":
            rows.append(r)
    df = _df_from_rows(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since","R2"], ascending=[True, False])
    out[f"Band Bounce BUY (Hourly {hourly_period})"] = df

    # Daily NPX↑NTD in uptrend
    rows = []
    for sym in symbols:
        r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=int(ntd_win), daily_view_label=str(daily_view_label))
        if r and int(r.get("Bars Since", 9999)) <= int(max_bars_since):
            rows.append(r)
    df = _df_from_rows(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since","Global Slope"], ascending=[True, False])
    out["NPX↑NTD (Daily Uptrend)"] = df

    return out

def run_regression_trend_bounce_scan(symbols: list,
                                     daily_view_label: str,
                                     slope_lb: int,
                                     max_bars_since: int):
    """
    REQUESTED CHANGE #2:
      - Uses existing sidebar 'recent horizon' (max_bars_since) as "recent".
      - (a) BUY: global slope>0 AND regression slope>0 AND bounce from -2σ with bar going up
      - (b) SELL: global slope<0 AND regression slope<0 AND bounce from +2σ with bar going down
    """
    rows = []
    for sym in symbols:
        r = last_daily_regression_trend_bounce(
            sym,
            daily_view_label=str(daily_view_label),
            slope_lb=int(slope_lb),
            max_bars_since=int(max_bars_since),
        )
        if r:
            rows.append(r)
    df = _df_from_rows(rows)
    if not df.empty:
        df = df.sort_values(["Side", "Bars Since", "R2"], ascending=[True, True, False])
    return df


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Streamlit app
# ---------------------------
def main():
    st.set_page_config(page_title="BullBear", layout="wide")

    st.title("BullBear — Trend, Bands, NTD/NPX")

    # Sidebar
    st.sidebar.header("Controls")

    symbol = st.sidebar.text_input("Symbol", value="SPY").strip().upper()

    daily_view_label = st.sidebar.selectbox(
        "Daily view",
        options=["6M", "1Y", "2Y", "5Y", "MAX"],
        index=1
    )

    ntd_win = st.sidebar.slider("NTD/NPX window", min_value=10, max_value=200, value=60, step=5)
    slope_lb = st.sidebar.slider("Regression lookback (bars)", min_value=20, max_value=260, value=80, step=5)

    # Reuse this as the “recent” horizon everywhere (per your note)
    rev_horizon = st.sidebar.slider("Recent horizon (bars)", min_value=1, max_value=60, value=15, step=1)

    st.sidebar.subheader("Overlays")
    show_sma8 = st.sidebar.checkbox("SMA(8)", value=True)
    show_fast_st = st.sidebar.checkbox("Fast Supertrend (Daily)", value=True)
    show_inrange = st.sidebar.checkbox("Shade In-Range (S↔R) on NTD", value=True)
    show_npx = st.sidebar.checkbox("Overlay NPX on NTD", value=True)

    st.sidebar.subheader("HMA")
    show_hma = st.sidebar.checkbox("Show HMA + reversals", value=False)
    hma_period = st.sidebar.slider("HMA period", min_value=10, max_value=200, value=55, step=5)
    hma_rev_lb = st.sidebar.slider("HMA reversal lookback", min_value=1, max_value=10, value=3, step=1)

    st.sidebar.subheader("Fib")
    show_fib = st.sidebar.checkbox("Show Fibonacci levels", value=False)

    st.sidebar.subheader("Markers")
    show_sessions = st.sidebar.checkbox("Session lines (NY/LDN)", value=False)
    show_news = st.sidebar.checkbox("News markers (Yahoo)", value=False)
    news_days = st.sidebar.slider("News lookback (days)", min_value=1, max_value=30, value=7, step=1)

    # Scanners
    st.sidebar.header("Scanner universe")
    default_universe = "SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, AMZN, META, TSLA"
    universe_text = st.sidebar.text_area("Symbols (comma/newline separated)", value=default_universe, height=120)
    universe = _symbols_from_text(universe_text)

    hourly_period = st.sidebar.selectbox("Hourly period", options=["1d", "5d", "1mo"], index=1)

    tab1, tab2 = st.tabs(["Chart", "Scanners"])

    with tab1:
        if not symbol:
            st.info("Enter a symbol.")
        else:
            fig, meta = build_daily_figure(
                symbol=symbol,
                daily_view_label=daily_view_label,
                ntd_win=int(ntd_win),
                slope_lb=int(slope_lb),
                show_sessions=bool(show_sessions),
                show_news=bool(show_news),
                news_days=int(news_days),
                show_inrange=bool(show_inrange),
                show_npx=bool(show_npx),
                show_hma=bool(show_hma),
                hma_period=int(hma_period),
                hma_rev_lb=int(hma_rev_lb),
                show_fib=bool(show_fib),
                show_fast_st=bool(show_fast_st),
                show_sma8=bool(show_sma8),
            )
            if fig is None:
                st.error("No data returned. Try a different symbol.")
            else:
                st.pyplot(fig, clear_figure=False)

                # Optional: show recent news table
                if show_news:
                    news_df = fetch_yf_news(symbol, window_days=int(news_days))
                    if news_df is not None and not news_df.empty:
                        with st.expander("Recent news"):
                            st.dataframe(news_df[["time","publisher","title","link"]], use_container_width=True)

    with tab2:
        st.subheader("Scanners")

        if not universe:
            st.info("Add symbols to the scanner universe in the sidebar.")
        else:
            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Regression Trend Bounce (Daily)")
                st.caption("BUY: uptrend + bounce from -2σ (bar up). SELL: downtrend + bounce from +2σ (bar down). Uses your **Recent horizon (bars)**.")
                df_rtb = run_regression_trend_bounce_scan(
                    symbols=universe,
                    daily_view_label=daily_view_label,
                    slope_lb=int(slope_lb),
                    max_bars_since=int(rev_horizon),  # reuse existing setting
                )
                st.dataframe(df_rtb, use_container_width=True)

            with colB:
                st.markdown("### Recent BUY Signals")
                st.caption("Band bounce + NPX↑NTD scans filtered by **Recent horizon (bars)**.")
                results = run_recent_buy_scans(
                    symbols=universe,
                    slope_lb=int(slope_lb),
                    ntd_win=int(ntd_win),
                    daily_view_label=daily_view_label,
                    hourly_period=str(hourly_period),
                    max_bars_since=int(rev_horizon),
                )
                for title, df in results.items():
                    with st.expander(title, expanded=False):
                        st.dataframe(df, use_container_width=True)


main()
