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

# ---------------------------
# NEW (THIS REQUEST): Band reversal detectors for Regression Trend tab
#   - Works regardless of slope sign
#   - Enforces "going up" / "going downward" via close-to-close direction
# ---------------------------
def find_band_reversal_up_from_lower(price: pd.Series,
                                     upper_band: pd.Series,
                                     lower_band: pd.Series):
    """
    'Just reversed' UP from lower band:
      - previous bar was below lower band
      - current bar is back inside the band
      - and current close > previous close
    Returns dict or None.
    """
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & u.notna() & l.notna()
    if ok.sum() < 2:
        return None
    p = p[ok]; u = u[ok]; l = l[ok]

    inside = (p <= u) & (p >= l)
    was_below = p.shift(1) < l.shift(1)
    going_up = p > p.shift(1)

    cand = inside & was_below & going_up
    if not cand.any():
        return None
    t = cand[cand].index[-1]
    return {"time": t, "price": float(p.loc[t]), "side": "UP_FROM_LOWER"}

def find_band_reversal_down_from_upper(price: pd.Series,
                                       upper_band: pd.Series,
                                       lower_band: pd.Series):
    """
    'Just reversed' DOWN from upper band:
      - previous bar was above upper band
      - current bar is back inside the band
      - and current close < previous close
    Returns dict or None.
    """
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & u.notna() & l.notna()
    if ok.sum() < 2:
        return None
    p = p[ok]; u = u[ok]; l = l[ok]

    inside = (p <= u) & (p >= l)
    was_above = p.shift(1) > u.shift(1)
    going_down = p < p.shift(1)

    cand = inside & was_above & going_down
    if not cand.any():
        return None
    t = cand[cand].index[-1]
    return {"time": t, "price": float(p.loc[t]), "side": "DOWN_FROM_UPPER"}

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
# Supertrend (REQUESTED CHANGE #1)
# - Used on BOTH Daily & Hourly price charts
# - Length=10, Factor=3
# - Shown by default (always plotted on price charts)
# ---------------------------
def compute_atr(ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)
    high = pd.to_numeric(ohlc["High"], errors="coerce")
    low = pd.to_numeric(ohlc["Low"], errors="coerce")
    close = pd.to_numeric(ohlc["Close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing (EMA alpha=1/period)
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False, min_periods=max(2, period // 2)).mean()
    return atr

def compute_supertrend(ohlc: pd.DataFrame, period: int = 10, factor: float = 3.0):
    """
    Returns:
      supertrend_line (pd.Series)
      direction_up (pd.Series bool)  True=uptrend, False=downtrend
      final_upper (pd.Series)
      final_lower (pd.Series)
    """
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else None
        empty = pd.Series(index=idx, dtype=float)
        return empty, pd.Series(index=idx, dtype=bool), empty, empty

    df = ohlc.copy()
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")

    atr = compute_atr(df, period=int(period))
    hl2 = (h + l) / 2.0

    basic_ub = hl2 + float(factor) * atr
    basic_lb = hl2 - float(factor) * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    # Final bands
    for i in range(1, len(df)):
        if pd.isna(final_ub.iloc[i - 1]) or pd.isna(c.iloc[i - 1]):
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            if (basic_ub.iloc[i] < final_ub.iloc[i - 1]) or (c.iloc[i - 1] > final_ub.iloc[i - 1]):
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i - 1]

        if pd.isna(final_lb.iloc[i - 1]) or pd.isna(c.iloc[i - 1]):
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            if (basic_lb.iloc[i] > final_lb.iloc[i - 1]) or (c.iloc[i - 1] < final_lb.iloc[i - 1]):
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i - 1]

    direction_up = pd.Series(True, index=df.index, dtype=bool)
    st_line = pd.Series(index=df.index, dtype=float)

    # Initialize
    if len(df) > 0:
        st_line.iloc[0] = final_lb.iloc[0] if np.isfinite(final_lb.iloc[0]) else np.nan

    for i in range(1, len(df)):
        prev_dir = bool(direction_up.iloc[i - 1])

        # Flip conditions
        if prev_dir and np.isfinite(c.iloc[i]) and np.isfinite(final_lb.iloc[i]) and c.iloc[i] < final_lb.iloc[i]:
            direction_up.iloc[i] = False
        elif (not prev_dir) and np.isfinite(c.iloc[i]) and np.isfinite(final_ub.iloc[i]) and c.iloc[i] > final_ub.iloc[i]:
            direction_up.iloc[i] = True
        else:
            direction_up.iloc[i] = prev_dir

        st_line.iloc[i] = final_lb.iloc[i] if direction_up.iloc[i] else final_ub.iloc[i]

    return st_line, direction_up, final_ub, final_lb

# ---------------------------
# Parabolic SAR (already in UI; keep existing toggles)
# ---------------------------
def compute_psar(ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.20) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)

    high = pd.to_numeric(ohlc["High"], errors="coerce")
    low = pd.to_numeric(ohlc["Low"], errors="coerce")
    close = pd.to_numeric(ohlc["Close"], errors="coerce")
    idx = ohlc.index

    psar = pd.Series(index=idx, dtype=float)

    # Init: assume uptrend initially
    uptrend = True
    af = float(step)
    ep = high.iloc[0] if np.isfinite(high.iloc[0]) else close.iloc[0]
    psar.iloc[0] = low.iloc[0] if np.isfinite(low.iloc[0]) else close.iloc[0]

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i - 1]
        if not np.isfinite(prev_psar):
            prev_psar = close.iloc[i - 1]

        ps = prev_psar + af * (ep - prev_psar)

        # Clamp
        if uptrend:
            ps = min(ps, low.iloc[i - 1] if np.isfinite(low.iloc[i - 1]) else ps)
            if i >= 2:
                ps = min(ps, low.iloc[i - 2] if np.isfinite(low.iloc[i - 2]) else ps)
        else:
            ps = max(ps, high.iloc[i - 1] if np.isfinite(high.iloc[i - 1]) else ps)
            if i >= 2:
                ps = max(ps, high.iloc[i - 2] if np.isfinite(high.iloc[i - 2]) else ps)

        # Reversal
        if uptrend and np.isfinite(low.iloc[i]) and low.iloc[i] < ps:
            uptrend = False
            ps = ep
            ep = low.iloc[i] if np.isfinite(low.iloc[i]) else ep
            af = float(step)
        elif (not uptrend) and np.isfinite(high.iloc[i]) and high.iloc[i] > ps:
            uptrend = True
            ps = ep
            ep = high.iloc[i] if np.isfinite(high.iloc[i]) else ep
            af = float(step)
        else:
            # Update EP & AF
            if uptrend:
                if np.isfinite(high.iloc[i]) and (not np.isfinite(ep) or high.iloc[i] > ep):
                    ep = high.iloc[i]
                    af = min(float(max_step), af + float(step))
            else:
                if np.isfinite(low.iloc[i]) and (not np.isfinite(ep) or low.iloc[i] < ep):
                    ep = low.iloc[i]
                    af = min(float(max_step), af + float(step))

        psar.iloc[i] = ps

    return psar

# ---------------------------
# Ichimoku Kijun (on price)
# ---------------------------
def compute_kijun(ohlc: pd.DataFrame, base_period: int = 26) -> pd.Series:
    if ohlc is None or ohlc.empty or not {"High", "Low"}.issubset(ohlc.columns):
        return pd.Series(dtype=float)
    high = pd.to_numeric(ohlc["High"], errors="coerce")
    low = pd.to_numeric(ohlc["Low"], errors="coerce")
    hh = high.rolling(int(base_period), min_periods=max(2, int(base_period)//2)).max()
    ll = low.rolling(int(base_period), min_periods=max(2, int(base_period)//2)).min()
    return (hh + ll) / 2.0


# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Plotting helpers (keep charts/views; add Supertrend overlay by default)
# ---------------------------
def plot_price_chart_with_overlays(
    ohlc: pd.DataFrame,
    title: str,
    lookback_for_reg: int,
    show_fibs_flag: bool,
    show_bbands_flag: bool,
    bb_window: int,
    bb_sigma: float,
    bb_ema: bool,
    show_ichi_flag: bool,
    ichi_base_period: int,
    show_psar_flag: bool,
    psar_step_val: float,
    psar_max_val: float,
):
    if ohlc is None or ohlc.empty:
        st.info("No data to plot.")
        return

    close = pd.to_numeric(ohlc["Close"], errors="coerce")

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(close.index, close.values, label="Close")

    # Regression trendline + ±2σ (already used across app; keep)
    yhat, ub, lb, slope_val, r2_val = regression_with_band(close, lookback=int(lookback_for_reg), z=2.0)
    if yhat is not None and len(yhat.dropna()) > 2:
        ax.plot(yhat.index, yhat.values, linestyle="--", label=f"Regression (slope={fmt_slope(slope_val)}, R²={fmt_r2(r2_val)})")
        ax.plot(ub.index, ub.values, linestyle=":", label="+2σ")
        ax.plot(lb.index, lb.values, linestyle=":", label="-2σ")

    # Fibonacci levels (if enabled)
    if show_fibs_flag:
        fibs = fibonacci_levels(close.dropna())
        if fibs:
            for k, v in fibs.items():
                ax.axhline(float(v), linestyle="--", linewidth=1.0, alpha=0.35)
                # keep labels subtle
                ax.text(close.index[-1], float(v), f" {k}", va="center", fontsize=8, alpha=0.6)

    # Bollinger Bands (if enabled)
    if show_bbands_flag:
        mid, upper, lower, _, _ = compute_bbands(close, window=int(bb_window), mult=float(bb_sigma), use_ema=bool(bb_ema))
        if mid is not None and len(mid.dropna()) > 2:
            ax.plot(mid.index, mid.values, linestyle="-.", label="BB Mid")
            ax.plot(upper.index, upper.values, linestyle=":", label="BB Upper")
            ax.plot(lower.index, lower.values, linestyle=":", label="BB Lower")

    # Ichimoku Kijun (if enabled)
    if show_ichi_flag:
        kijun = compute_kijun(ohlc, base_period=int(ichi_base_period))
        if kijun is not None and len(kijun.dropna()) > 2:
            ax.plot(kijun.index, kijun.values, linestyle="-.", label=f"Kijun({ichi_base_period})")

    # Parabolic SAR (if enabled)
    if show_psar_flag:
        psar = compute_psar(ohlc, step=float(psar_step_val), max_step=float(psar_max_val))
        if psar is not None and len(psar.dropna()) > 2:
            ax.scatter(psar.index, psar.values, s=12, label="PSAR")

    # ---------------------------
    # Supertrend overlay (REQUESTED CHANGE #1)
    # Always plotted; length=10, factor=3
    # ---------------------------
    st_line, st_dir_up, _, _ = compute_supertrend(ohlc, period=10, factor=3.0)
    if st_line is not None and len(st_line.dropna()) > 2:
        # plot as one line, and (optionally) direction markers
        ax.plot(st_line.index, st_line.values, linewidth=2.0, label="Supertrend(10,3)")

    ax.set_title(title)
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best", ncols=2)

    st.pyplot(fig, use_container_width=True)

def plot_ntd_panel(close: pd.Series, window: int, shade: bool, show_npx: bool, mark_cross: bool):
    if close is None or len(close) == 0:
        st.info("No NTD data to plot.")
        return

    ntd = compute_normalized_trend(close, window=int(window))
    npx = compute_normalized_price(close, window=int(window))

    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axhline(0, linewidth=1.0, alpha=0.35)
    ax.plot(ntd.index, ntd.values, label="NTD")

    if shade:
        shade_ntd_regions(ax, ntd)

    if show_npx:
        ax.plot(npx.index, npx.values, linestyle="--", label="NPX")

    if show_npx and mark_cross:
        cross_up, cross_dn = _cross_series(npx, ntd)
        up_idx = list(cross_up[cross_up].index) if cross_up is not None else []
        dn_idx = list(cross_dn[cross_dn].index) if cross_dn is not None else []
        if up_idx:
            ax.scatter(up_idx, ntd.loc[up_idx], s=25, label="Cross Up")
        if dn_idx:
            ax.scatter(dn_idx, ntd.loc[dn_idx], s=25, label="Cross Down")

    ax.set_title("NTD Panel")
    ax.set_ylabel("Normalized")
    style_axes(ax)
    ax.legend(loc="best", ncols=3)
    st.pyplot(fig, use_container_width=True)


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Main selection + run pipeline (keep run button; compute once; tabs rendered in Parts 8–10)
# ---------------------------
# Safe default index for selectbox
default_idx = 0
if "ticker_select" in st.session_state and st.session_state.ticker_select in universe:
    default_idx = universe.index(st.session_state.ticker_select)

ticker = st.sidebar.selectbox("Symbol", universe, index=default_idx, key="ticker_select")

run_clicked = st.sidebar.button("▶ Run", use_container_width=True, key="btn_run_all")
if run_clicked:
    st.session_state.run_all = True
    st.session_state.ticker = ticker
    st.session_state.mode_at_run = mode

# If mode changed after run, force rerun-required state off
if st.session_state.get("mode_at_run") is not None and st.session_state.mode_at_run != mode:
    st.session_state.run_all = False

if st.session_state.get("run_all", False) and st.session_state.get("ticker"):

    sym = st.session_state.ticker

    # Fetch data
    df_hist = fetch_hist(sym)               # Daily close (Series)
    df_ohlc = fetch_hist_ohlc(sym)          # Daily OHLC
    intraday = fetch_intraday(sym, "5d")    # Hourly-ish (5m bars, gapless)

    st.session_state.df_hist = df_hist
    st.session_state.df_ohlc = df_ohlc
    st.session_state.intraday = intraday

    # Forecast (kept)
    try:
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
    except Exception:
        fc_idx, fc_vals, fc_ci = None, None, None
    st.session_state.fc_idx = fc_idx
    st.session_state.fc_vals = fc_vals
    st.session_state.fc_ci = fc_ci

    # Prepared views
    daily_close = subset_by_daily_view(df_hist, daily_view)
    st.session_state.daily_close_view = daily_close

    # Hourly close from intraday bars (still 5m bars but used in "hourly" logic)
    if intraday is not None and not intraday.empty and "Close" in intraday.columns:
        st.session_state.hourly_close = pd.to_numeric(intraday["Close"], errors="coerce")
    else:
        st.session_state.hourly_close = pd.Series(dtype=float)

    # Quick top metrics display (non-invasive)
    last_close = _safe_last_float(daily_close)
    st.metric("Last Daily Close", fmt_price_val(last_close) if np.isfinite(last_close) else "n/a")

else:
    st.info("Select a symbol and click ▶ Run to load data.")
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Tab render helpers (keep same UI views; Supertrend already added in price plot)
# ---------------------------
def plot_forecast_panel(close: pd.Series, fc_idx, fc_vals, fc_ci, title: str = "Forecast"):
    if close is None or len(close) == 0:
        st.info("No data for forecast.")
        return

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.plot(close.index, close.values, label="Close")

    if fc_idx is not None and fc_vals is not None and len(fc_vals) > 0:
        ax.plot(fc_idx, fc_vals, linestyle="--", label="Forecast")
        if fc_ci is not None and isinstance(fc_ci, (pd.DataFrame, pd.Series)):
            try:
                if isinstance(fc_ci, pd.DataFrame) and fc_ci.shape[1] >= 2:
                    lo = fc_ci.iloc[:, 0]
                    hi = fc_ci.iloc[:, 1]
                    ax.fill_between(fc_idx, lo.values, hi.values, alpha=0.2, label="CI")
            except Exception:
                pass

    ax.set_title(title)
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="best")
    st.pyplot(fig, use_container_width=True)

@st.cache_data(show_spinner=False)
def _cached_daily_close(sym: str) -> pd.Series:
    return fetch_hist(sym)

@st.cache_data(show_spinner=False)
def _cached_daily_ohlc(sym: str) -> pd.DataFrame:
    return fetch_hist_ohlc(sym)

@st.cache_data(show_spinner=False)
def _cached_intraday(sym: str) -> pd.DataFrame:
    return fetch_intraday(sym, "5d")

def _make_hourly_ohlc(intraday: pd.DataFrame) -> pd.DataFrame:
    if intraday is None or intraday.empty:
        return pd.DataFrame()
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(intraday.columns):
        return pd.DataFrame()
    df = intraday.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample("60min").agg(agg)
    out = out.dropna(subset=["Close"])
    return out

def _scan_universe_rows(rows_fn, symbols: list, label: str = "Scanning...") -> pd.DataFrame:
    out_rows = []
    prog = st.progress(0, text=label)
    n = max(1, len(symbols))
    for i, s in enumerate(symbols):
        try:
            row = rows_fn(s)
            if row is not None:
                out_rows.append(row)
        except Exception:
            pass
        prog.progress(int(((i + 1) / n) * 100), text=label)
    prog.empty()
    return pd.DataFrame(out_rows)

def _safe_last2(series: pd.Series):
    if series is None or len(series.dropna()) < 2:
        return None, None, None
    s = series.dropna()
    return s.index[-1], float(s.iloc[-1]), float(s.iloc[-2])

def _fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "n/a"


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Tabs (keep originals; add new "Regression Trend" tab)
# ---------------------------
if st.session_state.get("run_all", False) and st.session_state.get("ticker"):

    sym = st.session_state.ticker
    df_hist = st.session_state.get("df_hist", pd.Series(dtype=float))
    df_ohlc = st.session_state.get("df_ohlc", pd.DataFrame())
    intraday = st.session_state.get("intraday", pd.DataFrame())
    fc_idx = st.session_state.get("fc_idx", None)
    fc_vals = st.session_state.get("fc_vals", None)
    fc_ci = st.session_state.get("fc_ci", None)

    # Read UI controls (defined in Parts 1–4)
    daily_view = st.session_state.get("daily_view", "6M")
    lookback_reg = int(st.session_state.get("lookback_reg", 200))
    show_fibs_flag = bool(st.session_state.get("show_fibs", False))
    show_bbands_flag = bool(st.session_state.get("show_bbands", True))
    bb_window = int(st.session_state.get("bb_window", 20))
    bb_sigma = float(st.session_state.get("bb_sigma", 2.0))
    bb_ema = bool(st.session_state.get("bb_ema", False))
    show_ichi_flag = bool(st.session_state.get("show_ichi", False))
    ichi_base_period = int(st.session_state.get("ichi_base_period", 26))
    show_psar_flag = bool(st.session_state.get("show_psar", False))
    psar_step_val = float(st.session_state.get("psar_step", 0.02))
    psar_max_val = float(st.session_state.get("psar_max", 0.20))
    ntd_window = int(st.session_state.get("ntd_window", 20))
    shade_ntd = bool(st.session_state.get("shade_ntd", True))
    show_npx = bool(st.session_state.get("show_npx", True))
    mark_cross = bool(st.session_state.get("mark_cross", True))

    # Views
    daily_close_view = subset_by_daily_view(df_hist, daily_view)
    hourly_ohlc = _make_hourly_ohlc(intraday)

    tab_names = [
        "Original Forecast",
        "Enhanced Forecast",
        "Bull vs Bear Metrics",
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
        # Requested change #2:
        "Regression Trend",
    ]

    tabs = st.tabs(tab_names)

    # ---- Tab 1: Original Forecast ----
    with tabs[0]:
        plot_forecast_panel(daily_close_view, fc_idx, fc_vals, fc_ci, title=f"{sym} — Original Forecast")
        if df_ohlc is not None and not df_ohlc.empty:
            plot_price_chart_with_overlays(
                df_ohlc.loc[daily_close_view.index.intersection(df_ohlc.index)] if isinstance(df_ohlc.index, pd.DatetimeIndex) else df_ohlc,
                title=f"{sym} — Daily Price",
                lookback_for_reg=lookback_reg,
                show_fibs_flag=False,
                show_bbands_flag=False,
                bb_window=bb_window,
                bb_sigma=bb_sigma,
                bb_ema=bb_ema,
                show_ichi_flag=False,
                ichi_base_period=ichi_base_period,
                show_psar_flag=False,
                psar_step_val=psar_step_val,
                psar_max_val=psar_max_val,
            )

    # ---- Tab 2: Enhanced Forecast ----
    with tabs[1]:
        plot_forecast_panel(daily_close_view, fc_idx, fc_vals, fc_ci, title=f"{sym} — Enhanced Forecast")
        if df_ohlc is not None and not df_ohlc.empty:
            plot_price_chart_with_overlays(
                df_ohlc.loc[daily_close_view.index.intersection(df_ohlc.index)] if isinstance(df_ohlc.index, pd.DatetimeIndex) else df_ohlc,
                title=f"{sym} — Daily Price (Overlays)",
                lookback_for_reg=lookback_reg,
                show_fibs_flag=show_fibs_flag,
                show_bbands_flag=show_bbands_flag,
                bb_window=bb_window,
                bb_sigma=bb_sigma,
                bb_ema=bb_ema,
                show_ichi_flag=show_ichi_flag,
                ichi_base_period=ichi_base_period,
                show_psar_flag=show_psar_flag,
                psar_step_val=psar_step_val,
                psar_max_val=psar_max_val,
            )
        plot_ntd_panel(daily_close_view, window=ntd_window, shade=shade_ntd, show_npx=show_npx, mark_cross=mark_cross)

    # ---- Tab 3: Bull vs Bear Metrics ----
    with tabs[2]:
        c_last = _safe_last_float(daily_close_view)
        c_prev = float(daily_close_view.dropna().iloc[-2]) if len(daily_close_view.dropna()) >= 2 else np.nan
        st.metric("Daily Close", fmt_price_val(c_last) if np.isfinite(c_last) else "n/a",
                  delta=(fmt_price_val(c_last - c_prev) if np.isfinite(c_last) and np.isfinite(c_prev) else None))

        if df_ohlc is not None and not df_ohlc.empty:
            plot_price_chart_with_overlays(
                df_ohlc.loc[daily_close_view.index.intersection(df_ohlc.index)] if isinstance(df_ohlc.index, pd.DatetimeIndex) else df_ohlc,
                title=f"{sym} — Daily Price",
                lookback_for_reg=lookback_reg,
                show_fibs_flag=show_fibs_flag,
                show_bbands_flag=show_bbands_flag,
                bb_window=bb_window,
                bb_sigma=bb_sigma,
                bb_ema=bb_ema,
                show_ichi_flag=show_ichi_flag,
                ichi_base_period=ichi_base_period,
                show_psar_flag=show_psar_flag,
                psar_step_val=psar_step_val,
                psar_max_val=psar_max_val,
            )

        if hourly_ohlc is not None and not hourly_ohlc.empty:
            plot_price_chart_with_overlays(
                hourly_ohlc,
                title=f"{sym} — Hourly Price",
                lookback_for_reg=max(60, lookback_reg // 4),
                show_fibs_flag=False,
                show_bbands_flag=False,
                bb_window=bb_window,
                bb_sigma=bb_sigma,
                bb_ema=bb_ema,
                show_ichi_flag=False,
                ichi_base_period=ichi_base_period,
                show_psar_flag=False,
                psar_step_val=psar_step_val,
                psar_max_val=psar_max_val,
            )

        plot_ntd_panel(daily_close_view, window=ntd_window, shade=shade_ntd, show_npx=show_npx, mark_cross=mark_cross)

    # ---- Tab 4: NTD -0.75 Scanner ----
    with tabs[3]:
        thresh = -0.75

        def _row_ntd(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 5:
                return None
            ntd = compute_normalized_trend(close.dropna(), window=ntd_window)
            if ntd is None or len(ntd.dropna()) == 0:
                return None
            v = float(ntd.dropna().iloc[-1])
            if v <= thresh:
                d, last, prev = _safe_last2(close)
                return {
                    "Symbol": s,
                    "NTD": round(v, 4),
                    "Last Close": fmt_price_val(last),
                    "Prev Close": fmt_price_val(prev),
                    "Date": str(d.date()) if hasattr(d, "date") else str(d),
                }
            return None

        df_scan = _scan_universe_rows(_row_ntd, universe, label="Scanning NTD ≤ -0.75 ...")
        if df_scan.empty:
            st.info("No matches.")
        else:
            df_scan = df_scan.sort_values(["NTD", "Symbol"], ascending=[True, True]).reset_index(drop=True)
            st.dataframe(df_scan, use_container_width=True)

    # ---- Tab 5: Long-Term History ----
    with tabs[4]:
        close = df_hist
        if close is None or len(close) == 0:
            st.info("No history.")
        else:
            fig, ax = plt.subplots(figsize=(11.5, 4.2))
            ax.plot(close.index, close.values, label="Close")
            ax.set_title(f"{sym} — Long-Term History")
            ax.set_ylabel("Price")
            style_axes(ax)
            ax.legend(loc="best")
            st.pyplot(fig, use_container_width=True)


# =========================
# Part 10/10 — bullbear.py
# ---------------------------
# Remaining tabs + NEW Regression Trend tab (Requested change #2)
# ---------------------------
if st.session_state.get("run_all", False) and st.session_state.get("ticker"):

    sym = st.session_state.ticker
    df_hist = st.session_state.get("df_hist", pd.Series(dtype=float))
    df_ohlc = st.session_state.get("df_ohlc", pd.DataFrame())
    intraday = st.session_state.get("intraday", pd.DataFrame())
    daily_view = st.session_state.get("daily_view", "6M")
    lookback_reg = int(st.session_state.get("lookback_reg", 200))
    ntd_window = int(st.session_state.get("ntd_window", 20))
    daily_close_view = subset_by_daily_view(df_hist, daily_view)

    # NOTE: tabs[] created in Part 9; in Streamlit this code runs top-to-bottom,
    # so we re-create the same tabs handle here only if it isn't in session_state.
    # This preserves UI without changing tab names/order.
    if "tabs_handle" not in st.session_state:
        tab_names = [
            "Original Forecast",
            "Enhanced Forecast",
            "Bull vs Bear Metrics",
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
            "Regression Trend",
        ]
        st.session_state.tabs_handle = st.tabs(tab_names)

    tabs = st.session_state.tabs_handle

    # Pull shared UI controls
    show_fibs_flag = bool(st.session_state.get("show_fibs", False))
    show_bbands_flag = bool(st.session_state.get("show_bbands", True))
    bb_window = int(st.session_state.get("bb_window", 20))
    bb_sigma = float(st.session_state.get("bb_sigma", 2.0))
    bb_ema = bool(st.session_state.get("bb_ema", False))
    show_ichi_flag = bool(st.session_state.get("show_ichi", False))
    ichi_base_period = int(st.session_state.get("ichi_base_period", 26))
    show_psar_flag = bool(st.session_state.get("show_psar", False))
    psar_step_val = float(st.session_state.get("psar_step", 0.02))
    psar_max_val = float(st.session_state.get("psar_max", 0.20))

    # Hourly OHLC for R² scans
    hourly_ohlc = _make_hourly_ohlc(intraday)

    # ---- Tab 6: Recent BUY Scanner ----
    with tabs[5]:
        # Conservative "BUY": NPX crosses above NTD (as used elsewhere), and NTD rising
        def _row_buy(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 5:
                return None
            close = close.dropna()
            ntd = compute_normalized_trend(close, window=ntd_window)
            npx = compute_normalized_price(close, window=ntd_window)
            if ntd is None or npx is None or len(ntd.dropna()) < 3 or len(npx.dropna()) < 3:
                return None
            cross_up, _ = _cross_series(npx, ntd)
            if cross_up is None or len(cross_up) == 0:
                return None
            if bool(cross_up.iloc[-1]) and float(ntd.iloc[-1]) > float(ntd.iloc[-2]):
                d, last, prev = _safe_last2(close)
                return {
                    "Symbol": s,
                    "Date": str(d.date()) if hasattr(d, "date") else str(d),
                    "Last Close": fmt_price_val(last),
                    "Prev Close": fmt_price_val(prev),
                    "NTD": round(float(ntd.iloc[-1]), 4),
                    "NPX": round(float(npx.iloc[-1]), 4),
                }
            return None

        df_buy = _scan_universe_rows(_row_buy, universe, label="Scanning Recent BUY ...")
        if df_buy.empty:
            st.info("No matches.")
        else:
            df_buy = df_buy.sort_values(["Date", "Symbol"], ascending=[False, True]).reset_index(drop=True)
            st.dataframe(df_buy, use_container_width=True)

    # ---- Tab 7: NPX 0.5-Cross Scanner ----
    with tabs[6]:
        def _row_npx05(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 5:
                return None
            close = close.dropna()
            npx = compute_normalized_price(close, window=ntd_window)
            if npx is None or len(npx.dropna()) < 3:
                return None
            # Cross above 0.5 in last bar
            if float(npx.iloc[-2]) < 0.5 <= float(npx.iloc[-1]):
                d, last, prev = _safe_last2(close)
                return {
                    "Symbol": s,
                    "Date": str(d.date()) if hasattr(d, "date") else str(d),
                    "NPX": round(float(npx.iloc[-1]), 4),
                    "Last Close": fmt_price_val(last),
                }
            return None

        df_npx = _scan_universe_rows(_row_npx05, universe, label="Scanning NPX 0.5 Cross ...")
        if df_npx.empty:
            st.info("No matches.")
        else:
            df_npx = df_npx.sort_values(["Date", "Symbol"], ascending=[False, True]).reset_index(drop=True)
            st.dataframe(df_npx, use_container_width=True)

    # ---- Tab 8: Fib NPX 0.0 Signal Scanner ----
    with tabs[7]:
        def _row_fib_npx0(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 30:
                return None
            close = close.dropna()
            npx = compute_normalized_price(close, window=ntd_window)
            if npx is None or len(npx.dropna()) < 3:
                return None
            # "Signal": NPX crosses up through 0.0
            if float(npx.iloc[-2]) < 0.0 <= float(npx.iloc[-1]):
                fibs = fibonacci_levels(close.tail(252))
                d, last, _ = _safe_last2(close)
                return {
                    "Symbol": s,
                    "Date": str(d.date()) if hasattr(d, "date") else str(d),
                    "NPX": round(float(npx.iloc[-1]), 4),
                    "Last Close": fmt_price_val(last),
                    "Fib 0.618": fmt_price_val(fibs.get("0.618", np.nan)) if fibs else "n/a",
                    "Fib 0.500": fmt_price_val(fibs.get("0.500", np.nan)) if fibs else "n/a",
                }
            return None

        df_fib = _scan_universe_rows(_row_fib_npx0, universe, label="Scanning Fib + NPX 0.0 Signal ...")
        if df_fib.empty:
            st.info("No matches.")
        else:
            df_fib = df_fib.sort_values(["Date", "Symbol"], ascending=[False, True]).reset_index(drop=True)
            st.dataframe(df_fib, use_container_width=True)

    # ---- Tab 9: Slope Direction Scan ----
    with tabs[8]:
        def _row_slope(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < lookback_reg + 5:
                return None
            close = close.dropna()
            yhat, ub, lb, slope, r2 = regression_with_band(close, lookback=lookback_reg, z=2.0)
            if yhat is None or slope is None:
                return None
            d, last, _ = _safe_last2(close)
            return {
                "Symbol": s,
                "Date": str(d.date()) if hasattr(d, "date") else str(d),
                "Slope": float(slope),
                "R²": float(r2) if r2 is not None else np.nan,
                "Last Close": fmt_price_val(last),
                "Direction": "Up" if float(slope) > 0 else "Down" if float(slope) < 0 else "Flat",
            }

        df_slope = _scan_universe_rows(_row_slope, universe, label="Scanning Regression Slope ...")
        if df_slope.empty:
            st.info("No results.")
        else:
            df_slope["Slope"] = df_slope["Slope"].map(lambda x: round(float(x), 6) if np.isfinite(x) else np.nan)
            df_slope["R²"] = df_slope["R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            st.dataframe(df_slope.sort_values(["Slope", "Symbol"], ascending=[False, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 10: Trendline Direction Lists ----
    with tabs[9]:
        def _row_trendlist(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < lookback_reg + 5:
                return None
            close = close.dropna()
            _, _, _, slope, r2 = regression_with_band(close, lookback=lookback_reg, z=2.0)
            if slope is None:
                return None
            return {"Symbol": s, "Slope": float(slope), "R²": float(r2) if r2 is not None else np.nan}

        df_t = _scan_universe_rows(_row_trendlist, universe, label="Building Trendline Lists ...")
        if df_t.empty:
            st.info("No results.")
        else:
            up = df_t[df_t["Slope"] > 0].copy().sort_values(["Slope", "Symbol"], ascending=[False, True])
            dn = df_t[df_t["Slope"] < 0].copy().sort_values(["Slope", "Symbol"], ascending=[True, True])

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Uptrend")
                st.dataframe(up.reset_index(drop=True), use_container_width=True, height=420)
            with c2:
                st.subheader("Downtrend")
                st.dataframe(dn.reset_index(drop=True), use_container_width=True, height=420)

    # ---- Tab 11: NTD Hot List ----
    with tabs[10]:
        def _row_hot(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 5:
                return None
            close = close.dropna()
            ntd = compute_normalized_trend(close, window=ntd_window)
            if ntd is None or len(ntd.dropna()) == 0:
                return None
            v = float(ntd.iloc[-1])
            d, last, _ = _safe_last2(close)
            return {"Symbol": s, "Date": str(d.date()) if hasattr(d, "date") else str(d), "NTD": v, "Last Close": fmt_price_val(last)}

        df_hot = _scan_universe_rows(_row_hot, universe, label="Building NTD Hot List ...")
        if df_hot.empty:
            st.info("No results.")
        else:
            df_hot["NTD"] = df_hot["NTD"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            st.dataframe(df_hot.sort_values(["NTD", "Symbol"], ascending=[False, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 12: NTD NPX 0.0-0.2 Scanner ----
    with tabs[11]:
        lo, hi = 0.0, 0.2

        def _row_ntd_npx(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < ntd_window + 5:
                return None
            close = close.dropna()
            ntd = compute_normalized_trend(close, window=ntd_window)
            npx = compute_normalized_price(close, window=ntd_window)
            if ntd is None or npx is None or len(npx.dropna()) == 0:
                return None
            v = float(npx.iloc[-1])
            if lo <= v <= hi:
                d, last, _ = _safe_last2(close)
                return {
                    "Symbol": s,
                    "Date": str(d.date()) if hasattr(d, "date") else str(d),
                    "NPX": round(v, 4),
                    "NTD": round(float(ntd.iloc[-1]), 4) if ntd is not None else np.nan,
                    "Last Close": fmt_price_val(last),
                }
            return None

        df_rng = _scan_universe_rows(_row_ntd_npx, universe, label="Scanning NPX 0.0–0.2 ...")
        if df_rng.empty:
            st.info("No matches.")
        else:
            st.dataframe(df_rng.sort_values(["NPX", "Symbol"], ascending=[True, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 13: Uptrend vs Downtrend ----
    with tabs[12]:
        def _row_ud(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < lookback_reg + 5:
                return None
            close = close.dropna()
            _, _, _, slope, r2 = regression_with_band(close, lookback=lookback_reg, z=2.0)
            if slope is None:
                return None
            return {"Symbol": s, "Uptrend": float(slope) > 0, "Slope": float(slope), "R²": float(r2) if r2 is not None else np.nan}

        df_ud = _scan_universe_rows(_row_ud, universe, label="Scanning Uptrend vs Downtrend ...")
        if df_ud.empty:
            st.info("No results.")
        else:
            up_cnt = int(df_ud["Uptrend"].sum())
            dn_cnt = int((~df_ud["Uptrend"]).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Uptrend", up_cnt)
            c2.metric("Downtrend", dn_cnt)
            c3.metric("Universe", len(df_ud))

            st.dataframe(
                df_ud.sort_values(["Uptrend", "Slope", "Symbol"], ascending=[False, False, True]).reset_index(drop=True),
                use_container_width=True
            )

    # ---- Tab 14: Ichimoku Kijun Scanner ----
    with tabs[13]:
        base_period = int(st.session_state.get("ichi_base_period", 26))

        def _row_kijun(s: str):
            ohlc = _cached_daily_ohlc(s)
            if ohlc is None or ohlc.empty or len(ohlc.dropna(subset=["Close"])) < base_period + 5:
                return None
            kijun = compute_kijun(ohlc, base_period=base_period)
            close = pd.to_numeric(ohlc["Close"], errors="coerce")
            if kijun is None or len(kijun.dropna()) == 0 or len(close.dropna()) == 0:
                return None
            last_close = float(close.dropna().iloc[-1])
            last_kijun = float(kijun.dropna().iloc[-1])
            return {
                "Symbol": s,
                "Close": fmt_price_val(last_close),
                "Kijun": fmt_price_val(last_kijun),
                "Above Kijun": last_close > last_kijun,
                "Diff": fmt_price_val(last_close - last_kijun),
            }

        df_k = _scan_universe_rows(_row_kijun, universe, label="Scanning Ichimoku Kijun ...")
        if df_k.empty:
            st.info("No results.")
        else:
            st.dataframe(df_k.sort_values(["Above Kijun", "Symbol"], ascending=[False, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 15: R² > 45% Daily/Hourly ----
    with tabs[14]:
        thr = 0.45

        def _row_r2_hi(s: str):
            # Daily
            close = _cached_daily_close(s)
            d_r2 = np.nan
            if close is not None and len(close.dropna()) >= lookback_reg + 5:
                _, _, _, _, r2 = regression_with_band(close.dropna(), lookback=lookback_reg, z=2.0)
                d_r2 = float(r2) if r2 is not None else np.nan

            # Hourly
            intr = _cached_intraday(s)
            h_r2 = np.nan
            h = _make_hourly_ohlc(intr)
            if h is not None and not h.empty and "Close" in h.columns:
                hc = pd.to_numeric(h["Close"], errors="coerce").dropna()
                if len(hc) >= max(60, lookback_reg // 4):
                    _, _, _, _, r2h = regression_with_band(hc, lookback=max(60, lookback_reg // 4), z=2.0)
                    h_r2 = float(r2h) if r2h is not None else np.nan

            if (np.isfinite(d_r2) and d_r2 >= thr) or (np.isfinite(h_r2) and h_r2 >= thr):
                return {"Symbol": s, "Daily R²": d_r2, "Hourly R²": h_r2}
            return None

        df_hi = _scan_universe_rows(_row_r2_hi, universe, label="Scanning R² ≥ 45% ...")
        if df_hi.empty:
            st.info("No matches.")
        else:
            df_hi["Daily R²"] = df_hi["Daily R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            df_hi["Hourly R²"] = df_hi["Hourly R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            st.dataframe(df_hi.sort_values(["Daily R²", "Hourly R²", "Symbol"], ascending=[False, False, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 16: R² < 45% Daily/Hourly ----
    with tabs[15]:
        thr = 0.45

        def _row_r2_lo(s: str):
            close = _cached_daily_close(s)
            d_r2 = np.nan
            if close is not None and len(close.dropna()) >= lookback_reg + 5:
                _, _, _, _, r2 = regression_with_band(close.dropna(), lookback=lookback_reg, z=2.0)
                d_r2 = float(r2) if r2 is not None else np.nan

            intr = _cached_intraday(s)
            h_r2 = np.nan
            h = _make_hourly_ohlc(intr)
            if h is not None and not h.empty and "Close" in h.columns:
                hc = pd.to_numeric(h["Close"], errors="coerce").dropna()
                if len(hc) >= max(60, lookback_reg // 4):
                    _, _, _, _, r2h = regression_with_band(hc, lookback=max(60, lookback_reg // 4), z=2.0)
                    h_r2 = float(r2h) if r2h is not None else np.nan

            # Include if either is known and below threshold
            cond = (np.isfinite(d_r2) and d_r2 < thr) or (np.isfinite(h_r2) and h_r2 < thr)
            if cond:
                return {"Symbol": s, "Daily R²": d_r2, "Hourly R²": h_r2}
            return None

        df_lo = _scan_universe_rows(_row_r2_lo, universe, label="Scanning R² < 45% ...")
        if df_lo.empty:
            st.info("No matches.")
        else:
            df_lo["Daily R²"] = df_lo["Daily R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            df_lo["Hourly R²"] = df_lo["Hourly R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)
            st.dataframe(df_lo.sort_values(["Daily R²", "Hourly R²", "Symbol"], ascending=[True, True, True]).reset_index(drop=True),
                         use_container_width=True)

    # ---- Tab 17: Regression Trend (NEW) ----
    with tabs[16]:
        # Requested change #2:
        # (a) slope > 0 and price just reversed from LOWER 2σ line going UP
        # (b) slope > 0 and price just reversed from UPPER 2σ line going DOWN
        def _reg_reversal_row(s: str):
            close = _cached_daily_close(s)
            if close is None or len(close.dropna()) < lookback_reg + 5:
                return None
            close = close.dropna()
            yhat, ub, lb, slope, r2 = regression_with_band(close, lookback=lookback_reg, z=2.0)
            if yhat is None or ub is None or lb is None or slope is None:
                return None
            if len(close) < 3 or len(lb.dropna()) < 3 or len(ub.dropna()) < 3:
                return None

            # Align indices
            common = close.index.intersection(lb.index).intersection(ub.index)
            if len(common) < 3:
                return None
            close2 = close.loc[common]
            lb2 = lb.loc[common]
            ub2 = ub.loc[common]

            last_i = close2.index[-1]
            prev_i = close2.index[-2]

            last_close = float(close2.loc[last_i])
            prev_close = float(close2.loc[prev_i])
            prev_lb = float(lb2.loc[prev_i])
            prev_ub = float(ub2.loc[prev_i])

            slope_v = float(slope)
            r2_v = float(r2) if r2 is not None else np.nan

            if not np.isfinite(slope_v) or slope_v <= 0:
                return None

            # Reversal definitions (most recent two bars):
            # A) "reversed from lower band going up": previous close at/below prev lower band, then close increased
            hit_lower_then_up = (np.isfinite(prev_lb) and prev_close <= prev_lb and last_close > prev_close)

            # B) "reversed from upper band going down": previous close at/above prev upper band, then close decreased
            hit_upper_then_down = (np.isfinite(prev_ub) and prev_close >= prev_ub and last_close < prev_close)

            if not (hit_lower_then_up or hit_upper_then_down):
                return None

            return {
                "Symbol": s,
                "Date": str(last_i.date()) if hasattr(last_i, "date") else str(last_i),
                "Last Close": fmt_price_val(last_close),
                "Prev Close": fmt_price_val(prev_close),
                "Slope": slope_v,
                "R²": r2_v,
                "Signal": ("Lower 2σ Reversal Up" if hit_lower_then_up else "Upper 2σ Reversal Down"),
            }

        df_reg = _scan_universe_rows(_reg_reversal_row, universe, label="Scanning Regression Trend Reversals ...")
        if df_reg.empty:
            st.info("No matches.")
        else:
            df_reg["Slope"] = df_reg["Slope"].map(lambda x: round(float(x), 6) if np.isfinite(x) else np.nan)
            df_reg["R²"] = df_reg["R²"].map(lambda x: round(float(x), 4) if np.isfinite(x) else np.nan)

            df_a = df_reg[df_reg["Signal"] == "Lower 2σ Reversal Up"].copy()
            df_b = df_reg[df_reg["Signal"] == "Upper 2σ Reversal Down"].copy()

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("A) slope > 0 AND reversed from LOWER 2σ going UP")
                if df_a.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(
                        df_a.sort_values(["Date", "Slope", "Symbol"], ascending=[False, False, True]).reset_index(drop=True),
                        use_container_width=True,
                        height=520,
                    )
            with c2:
                st.subheader("B) slope > 0 AND reversed from UPPER 2σ going DOWN")
                if df_b.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(
                        df_b.sort_values(["Date", "Slope", "Symbol"], ascending=[False, False, True]).reset_index(drop=True),
                        use_container_width=True,
                        height=520,
                    )
