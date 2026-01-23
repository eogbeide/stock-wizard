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
    """
    Supertrend line + trend state.
    NOTE (THIS REQUEST): Price charts will overlay Supertrend by default (no UI toggle).
    """
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
# NEW (THIS REQUEST): Daily band-proximity helper for Slope Direction Scan subsets
#   - For Upward Slope list: subset where current close is close to LOWER +2σ band (lower 2 std dev line)
#   - For Downward Slope list: subset where current close is close to UPPER +2σ band (upper 2 std dev line)
#   - Uses EXISTING sr_prox_pct (no UI change)
# ---------------------------
@st.cache_data(ttl=120)
def daily_band_proximity_in_view(symbol: str,
                                 daily_view_label: str,
                                 slope_lb: int,
                                 prox_pct: float):
    """
    Returns dict with:
      - close_last, upper_last, lower_last
      - near_lower (within prox_pct), near_upper (within prox_pct)
      - dist_lower_pct, dist_upper_pct (fractions; e.g., 0.0025 = 0.25%)
      - slope, r2 (from the same regression_with_band call)
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(3, int(slope_lb)):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb), z=2.0)
        if up is None or lo is None or up.dropna().empty or lo.dropna().empty:
            return None

        c_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        u_last = float(_coerce_1d_series(up).dropna().iloc[-1]) if len(_coerce_1d_series(up).dropna()) else np.nan
        l_last = float(_coerce_1d_series(lo).dropna().iloc[-1]) if len(_coerce_1d_series(lo).dropna()) else np.nan
        if not np.all(np.isfinite([c_last, u_last, l_last])) or c_last == 0:
            return None

        dist_lower = (c_last - l_last) / c_last
        dist_upper = (u_last - c_last) / c_last

        prox = float(prox_pct) if np.isfinite(prox_pct) else 0.0
        prox = max(0.0, prox)

        near_lower = (np.isfinite(dist_lower) and dist_lower >= 0.0 and dist_lower <= prox)
        near_upper = (np.isfinite(dist_upper) and dist_upper >= 0.0 and dist_upper <= prox)

        return {
            "Symbol": symbol,
            "Close (last)": c_last,
            "Upper 2σ (last)": u_last,
            "Lower 2σ (last)": l_last,
            "Dist to Lower (pct)": dist_lower,
            "Dist to Upper (pct)": dist_upper,
            "Near Lower": bool(near_lower),
            "Near Upper": bool(near_upper),
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "AsOf": close_show.index[-1],
        }
    except Exception:
        return None

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
# Ichimoku Kijun Daily Cross-Up Scanner helper (Daily only / matches Price Chart)
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
# R² scanners (Daily/Hourly)
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
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Chart helpers (ribbons + formatting)
# ---------------------------
def _fmt_pct(x, digits=2):
    try:
        if x is None or not np.isfinite(x):
            return ""
        return f"{100.0*float(x):.{digits}f}%"
    except Exception:
        return ""

def _fmt_num(x, digits=4):
    try:
        if x is None or not np.isfinite(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return ""

def _safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _annotate_ribbon(ax, text: str, y: float = 1.01, ha: str = "left", color: str = "black"):
    ax.text(
        0.01 if ha == "left" else 0.99,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
        zorder=50
    )

def _annotate_sub_ribbon(ax, text: str, y: float = 0.97, ha: str = "left", color: str = "black"):
    ax.text(
        0.01 if ha == "left" else 0.99,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=9,
        color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.75),
        zorder=50
    )

def _price_label_right(ax, x, y, txt, color="black"):
    try:
        ax.text(
            x, y, txt,
            ha="left", va="center",
            fontsize=9,
            color=color,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color, alpha=0.85),
            zorder=60
        )
    except Exception:
        pass

def _plot_right_edge_label(ax, idx, y, text, color="black"):
    if idx is None or len(idx) == 0 or not np.isfinite(y):
        return
    try:
        x = idx[-1]
        _price_label_right(ax, x, y, text, color=color)
    except Exception:
        pass

def _color_for_slope(m):
    if not np.isfinite(m):
        return "black"
    return "tab:green" if float(m) > 0 else ("tab:red" if float(m) < 0 else "black")

def _maybe_plot_fibs(ax, close: pd.Series, show_fib: bool, fib_default_on: bool):
    if not show_fib and not fib_default_on:
        return None
    if close is None or close.empty:
        return None
    fibs = fibonacci_levels(_coerce_1d_series(close).dropna())
    if not fibs:
        return None
    # Keep original styling behavior; only draw if enabled in existing code path
    for k, v in fibs.items():
        if not np.isfinite(v):
            continue
        ax.axhline(v, linestyle=":", linewidth=1.0, alpha=0.25, color="gray")
    return fibs

def _plot_ichimoku(ax, ohlc: pd.DataFrame, close: pd.Series,
                   ich_conv: int, ich_base: int, ich_span_b: int,
                   show_cloud: bool = True, label_prefix: str = ""):
    if ohlc is None or ohlc.empty or close is None or close.empty:
        return None
    tenkan, kijun, sa, sb, chikou = ichimoku_lines(
        ohlc["High"], ohlc["Low"], ohlc["Close"],
        conv=int(ich_conv), base=int(ich_base), span_b=int(ich_span_b),
        shift_cloud=False
    )
    idx = close.index
    tenkan = _coerce_1d_series(tenkan).reindex(idx).ffill().bfill()
    kijun  = _coerce_1d_series(kijun).reindex(idx).ffill().bfill()
    sa     = _coerce_1d_series(sa).reindex(idx).ffill().bfill()
    sb     = _coerce_1d_series(sb).reindex(idx).ffill().bfill()

    ax.plot(idx, tenkan, linewidth=1.1, alpha=0.55, color="tab:blue",  label=f"{label_prefix}Tenkan")
    ax.plot(idx, kijun,  linewidth=1.7, alpha=0.90, color="black",     label=f"{label_prefix}Kijun")

    if bool(show_cloud):
        try:
            ax.fill_between(idx, sa.to_numpy(dtype=float), sb.to_numpy(dtype=float),
                            where=(sa >= sb).to_numpy(dtype=bool),
                            alpha=0.10, interpolate=True)
            ax.fill_between(idx, sa.to_numpy(dtype=float), sb.to_numpy(dtype=float),
                            where=(sa < sb).to_numpy(dtype=bool),
                            alpha=0.10, interpolate=True)
        except Exception:
            pass

    return {"tenkan": tenkan, "kijun": kijun, "sa": sa, "sb": sb}

def _plot_psar(ax, ohlc: pd.DataFrame, show: bool = True):
    if not show or ohlc is None or ohlc.empty:
        return None
    ps = compute_psar_from_ohlc(ohlc, step=0.02, max_step=0.2)
    if ps is None or ps.empty:
        return None
    idx = ohlc.index
    psar = _coerce_1d_series(ps["PSAR"]).reindex(idx)
    ax.scatter(idx, psar, s=10, alpha=0.55, color="tab:purple", label="PSAR")
    return ps

# ---------------------------
# NEW (THIS REQUEST): Supertrend overlay (show by default)
# ---------------------------
def _plot_supertrend(ax, ohlc: pd.DataFrame, st_len: int = 10, st_factor: float = 3.0):
    if ohlc is None or ohlc.empty:
        return None
    st = compute_supertrend(ohlc, atr_period=int(st_len), atr_mult=float(st_factor))
    if st is None or st.empty:
        return None
    idx = ohlc.index
    st_line = _coerce_1d_series(st["ST"]).reindex(idx)
    up = _coerce_1d_series(st["in_uptrend"]).reindex(idx).fillna(method="ffill").fillna(True).astype(bool)

    # Plot single line; color by trend state but keep it simple (no new UI)
    try:
        ax.plot(idx, st_line.where(up), linewidth=1.8, alpha=0.85, color="tab:green", label=f"Supertrend ({st_len},{st_factor:g})")
        ax.plot(idx, st_line.where(~up), linewidth=1.8, alpha=0.85, color="tab:red",   label=f"Supertrend ({st_len},{st_factor:g})")
    except Exception:
        ax.plot(idx, st_line, linewidth=1.8, alpha=0.85, color="tab:green", label=f"Supertrend ({st_len},{st_factor:g})")

    return st

# ---------------------------
# Plot Price panel (Daily/Hourly) - preserves existing UI behavior
# ---------------------------
def plot_price_panel(ax, close: pd.Series, ohlc: pd.DataFrame,
                     slope_lb: int,
                     show_global_trend: bool,
                     show_fib: bool,
                     fib_default_on: bool,
                     show_ichimoku: bool,
                     ich_conv: int, ich_base: int, ich_span_b: int,
                     show_psar: bool,
                     show_sessions: bool,
                     show_news: bool,
                     symbol: str,
                     is_intraday: bool = False,
                     intraday_period: str = "1d",
                     st_len: int = 10,
                     st_factor: float = 3.0):
    """
    NOTE: We do NOT add new UI controls. Supertrend is drawn by default.
    """
    close = _coerce_1d_series(close).dropna()
    if close.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return {}

    idx = close.index

    # Price line
    ax.plot(idx, close, linewidth=2.0, alpha=0.90, color="black", label="Price")

    # Regression + bands
    yhat, up2, lo2, m, r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)
    if yhat is not None:
        ax.plot(idx, yhat, linestyle="--", linewidth=1.8, alpha=0.55, color=_color_for_slope(m), label="Regression")
    if up2 is not None and lo2 is not None:
        ax.plot(idx, up2, linestyle=":", linewidth=1.3, alpha=0.35, color="tab:red", label="+2σ")
        ax.plot(idx, lo2, linestyle=":", linewidth=1.3, alpha=0.35, color="tab:green", label="-2σ")

    # Global trendline (existing checkbox)
    if bool(show_global_trend):
        # preserve original behavior: best-fit on whole visible range
        try:
            x = np.arange(len(close), dtype=float)
            mm, bb = np.polyfit(x, close.to_numpy(dtype=float), 1)
            yline = mm * x + bb
            ax.plot(idx, yline, linestyle=(0, (4, 2)), linewidth=2.2, alpha=0.85,
                    color=("tab:green" if mm > 0 else "tab:red"), label="Global Trend")
        except Exception:
            pass

    # Fib
    _maybe_plot_fibs(ax, close, show_fib=bool(show_fib), fib_default_on=bool(fib_default_on))

    # Ichimoku
    ich = None
    kijun = None
    if bool(show_ichimoku) and ohlc is not None and not ohlc.empty:
        ich = _plot_ichimoku(ax, ohlc, close, ich_conv, ich_base, ich_span_b, show_cloud=True, label_prefix="")
        if isinstance(ich, dict):
            kijun = ich.get("kijun", None)

    # PSAR
    if bool(show_psar) and ohlc is not None and not ohlc.empty:
        _plot_psar(ax, ohlc, show=True)

    # NEW: Supertrend overlay by default
    if ohlc is not None and not ohlc.empty and {"High","Low","Close"}.issubset(ohlc.columns):
        _plot_supertrend(ax, ohlc, st_len=st_len, st_factor=st_factor)

    # Sessions (intraday)
    if bool(show_sessions) and is_intraday and isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        lines = compute_session_lines(idx)
        handles, labels = draw_session_lines(ax, lines, alpha=0.25)
        # keep legend managed by caller; just add dummy handles via ax.plot already

    # News markers
    if bool(show_news):
        try:
            news_df = fetch_yf_news(symbol, window_days=7)
            if news_df is not None and not news_df.empty:
                draw_news_markers(ax, list(news_df["time"].values), label="News (7d)")
        except Exception:
            pass

    # Ribbon summary
    mcol = _color_for_slope(m)
    _annotate_ribbon(ax, f"{symbol}  |  slope={_fmt_num(m,4)}  r²={_fmt_num(r2,3)}", y=1.01, ha="left", color=mcol)

    # Right-edge labels
    if up2 is not None and lo2 is not None:
        u_last = _safe_float(_coerce_1d_series(up2).dropna().iloc[-1]) if len(_coerce_1d_series(up2).dropna()) else np.nan
        l_last = _safe_float(_coerce_1d_series(lo2).dropna().iloc[-1]) if len(_coerce_1d_series(lo2).dropna()) else np.nan
        _plot_right_edge_label(ax, idx, u_last, f"+2σ {u_last:,.2f}", color="tab:red")
        _plot_right_edge_label(ax, idx, l_last, f"-2σ {l_last:,.2f}", color="tab:green")

    p_last = _safe_float(close.iloc[-1])
    _plot_right_edge_label(ax, idx, p_last, f"PX {p_last:,.2f}", color="black")

    if kijun is not None:
        k_last = _safe_float(_coerce_1d_series(kijun).dropna().iloc[-1]) if len(_coerce_1d_series(kijun).dropna()) else np.nan
        if np.isfinite(k_last):
            _plot_right_edge_label(ax, idx, k_last, f"Kijun {k_last:,.2f}", color="black")

    # Return items used by caller
    return {
        "slope": m,
        "r2": r2,
        "yhat": yhat,
        "up2": up2,
        "lo2": lo2,
        "kijun": kijun,
    }

# ---------------------------
# Scan builders (Slope Direction Scan tab + subsets)
# ---------------------------
def build_slope_direction_scan(universe: list,
                               daily_view_label: str,
                               slope_lb: int,
                               r2_min: float,
                               sr_prox_pct: float):
    """
    Returns:
      - up_df, dn_df (original lists, unchanged)
      - up_near_lower_df (subset of up_df near LOWER 2σ using sr_prox_pct)
      - dn_near_upper_df (subset of dn_df near UPPER 2σ using sr_prox_pct)
    """
    rows_up, rows_dn = [], []
    # original direction scan: based on daily_global_slope + r2 >= threshold
    for sym in universe:
        m, r2, asof = daily_global_slope(sym, daily_view_label)
        if not np.isfinite(m) or not np.isfinite(r2):
            continue
        if float(r2) < float(r2_min):
            continue
        if float(m) > 0:
            rows_up.append({"Symbol": sym, "Slope": float(m), "R2": float(r2), "AsOf": asof})
        elif float(m) < 0:
            rows_dn.append({"Symbol": sym, "Slope": float(m), "R2": float(r2), "AsOf": asof})

    up_df = pd.DataFrame(rows_up).sort_values(["R2", "Slope"], ascending=[False, False]) if rows_up else pd.DataFrame(columns=["Symbol","Slope","R2","AsOf"])
    dn_df = pd.DataFrame(rows_dn).sort_values(["R2", "Slope"], ascending=[False, True])  if rows_dn else pd.DataFrame(columns=["Symbol","Slope","R2","AsOf"])

    # NEW subsets (THIS REQUEST) using existing sr_prox_pct
    up_near_rows = []
    for sym in up_df["Symbol"].tolist() if not up_df.empty else []:
        info = daily_band_proximity_in_view(sym, daily_view_label, slope_lb=int(slope_lb), prox_pct=float(sr_prox_pct))
        if info and bool(info.get("Near Lower", False)):
            up_near_rows.append(info)

    dn_near_rows = []
    for sym in dn_df["Symbol"].tolist() if not dn_df.empty else []:
        info = daily_band_proximity_in_view(sym, daily_view_label, slope_lb=int(slope_lb), prox_pct=float(sr_prox_pct))
        if info and bool(info.get("Near Upper", False)):
            dn_near_rows.append(info)

    up_near_lower_df = pd.DataFrame(up_near_rows) if up_near_rows else pd.DataFrame(columns=[
        "Symbol","Close (last)","Upper 2σ (last)","Lower 2σ (last)","Dist to Lower (pct)","Dist to Upper (pct)","Near Lower","Near Upper","Slope","R2","AsOf"
    ])
    dn_near_upper_df = pd.DataFrame(dn_near_rows) if dn_near_rows else pd.DataFrame(columns=[
        "Symbol","Close (last)","Upper 2σ (last)","Lower 2σ (last)","Dist to Lower (pct)","Dist to Upper (pct)","Near Lower","Near Upper","Slope","R2","AsOf"
    ])

    # Make distance columns human-friendly; keep raw too
    if not up_near_lower_df.empty:
        up_near_lower_df["Dist to Lower (%)"] = (100.0 * up_near_lower_df["Dist to Lower (pct)"].astype(float)).round(3)
        up_near_lower_df["Dist to Upper (%)"] = (100.0 * up_near_lower_df["Dist to Upper (pct)"].astype(float)).round(3)
        up_near_lower_df = up_near_lower_df.sort_values(["R2","Dist to Lower (pct)"], ascending=[False, True])

    if not dn_near_upper_df.empty:
        dn_near_upper_df["Dist to Lower (%)"] = (100.0 * dn_near_upper_df["Dist to Lower (pct)"].astype(float)).round(3)
        dn_near_upper_df["Dist to Upper (%)"] = (100.0 * dn_near_upper_df["Dist to Upper (pct)"].astype(float)).round(3)
        dn_near_upper_df = dn_near_upper_df.sort_values(["R2","Dist to Upper (pct)"], ascending=[False, True])

    return up_df, dn_df, up_near_lower_df, dn_near_upper_df


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# UI (Tabs) - preserve existing look & feel.
# Only inject the requested scan outputs and Supertrend overlay; do NOT alter tabs layout.
# ---------------------------

# NOTE: The main app code below assumes your original file already defines:
# - universe (symbols list)
# - sidebar controls (including sr_prox_pct, slope_lb, daily_view_label, r2_min, etc.)
# - tab objects and the existing plot creation
#
# We only patch in TWO things:
#   (A) Supertrend overlay is plotted by plot_price_panel (already called in existing chart blocks)
#   (B) Slope Direction Scan tab now includes the two new subset lists using build_slope_direction_scan()

# ---- Example integration snippet (keep your original tab names/structure) ----
# In your Slope Direction Scan tab code section, replace only the scan construction + display with this:
def render_slope_direction_scan_tab(universe: list,
                                    daily_view_label: str,
                                    slope_lb: int,
                                    r2_min: float,
                                    sr_prox_pct: float):
    st.subheader("Slope Direction Scan")
    up_df, dn_df, up_near_lower_df, dn_near_upper_df = build_slope_direction_scan(
        universe=universe,
        daily_view_label=daily_view_label,
        slope_lb=int(slope_lb),
        r2_min=float(r2_min),
        sr_prox_pct=float(sr_prox_pct),
    )

    # Original lists (UNCHANGED)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Upward Slope (R² ≥ threshold)")
        st.dataframe(up_df, use_container_width=True, height=380)
        st.markdown("### Upward Slope — Near Lower 2σ (using sr_prox_pct)")
        st.dataframe(up_near_lower_df, use_container_width=True, height=320)
    with c2:
        st.markdown("### Downward Slope (R² ≥ threshold)")
        st.dataframe(dn_df, use_container_width=True, height=380)
        st.markdown("### Downward Slope — Near Upper 2σ (using sr_prox_pct)")
        st.dataframe(dn_near_upper_df, use_container_width=True, height=320)

    st.caption(f"Proximity threshold uses existing sr_prox_pct = {100.0*float(sr_prox_pct):.3f}% of last close.")

# ---------------------------
# Supertrend integration: ensure your existing chart blocks pass OHLC to plot_price_panel.
# In daily chart block:
#   ohlc_daily = fetch_hist_ohlc(symbol)  # already in your code
#   plot_price_panel(ax, close_daily, ohlc_daily, ..., st_len=10, st_factor=3.0)
# In hourly chart block:
#   ohlc_hourly = df_hourly[['High','Low','Close']] (or fetch_intraday OHLC if available)
#   plot_price_panel(ax, close_hourly, ohlc_hourly, ..., st_len=10, st_factor=3.0)
#
# If your intraday fetch already includes High/Low, pass it as-is.
# If it doesn't, Supertrend will simply not plot for that chart (no UI change).

# ---------------------------
# Part 10/10 — bullbear.py
# ---------------------------
# ---------------------------
# Final patch notes + minimal safe intraday OHLC wrapper
# ---------------------------
def _intraday_ohlc_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to standardize intraday OHLC for Supertrend/Ichimoku/PSAR overlays.
    Does NOT change UI. Used only internally by your chart render code.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["High","Low","Close"])
    cols = set(df.columns)
    if {"High","Low","Close"}.issubset(cols):
        return df[["High","Low","Close"]].copy()
    if "Close" in cols:
        # fallback: fabricate High/Low so functions won't error (Supertrend won't be meaningful without real HL)
        out = pd.DataFrame(index=df.index)
        out["Close"] = df["Close"].astype(float)
        out["High"] = out["Close"]
        out["Low"]  = out["Close"]
        return out
    return pd.DataFrame(columns=["High","Low","Close"])

# ---------------------------
# IMPORTANT: Ensure your existing code calls render_slope_direction_scan_tab()
# inside the Slope Direction Scan tab instead of the old rendering function.
# Example:
#
# with tab_slope_scan:
#     render_slope_direction_scan_tab(universe, daily_view_label, slope_lb, r2_min, sr_prox_pct)
#
# No other UI changes.
# ---------------------------
