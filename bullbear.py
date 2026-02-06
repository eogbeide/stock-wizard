# bullbear.py
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
        av = float(a)
        bv = float(b)
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
    refp = pd.to_numeric(df[ref_col], errors="coerce")

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
            dt_sec = float((idx[i] - idx[i - 1]).total_seconds())
        except Exception:
            dt_sec = 0.0

        if dt_sec >= thr:
            prev_close = float(close.iloc[i - 1]) if np.isfinite(close.iloc[i - 1]) else np.nan
            curr_ref = float(refp.iloc[i]) if np.isfinite(refp.iloc[i]) else np.nan
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

slope_lb_daily = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
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
psar_max = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

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
        "AAPL", "SPY", "AMZN", "DIA", "TSLA", "SPGI", "JPM", "VTWG", "PLTR", "NVDA",
        "META", "SITM", "MARA", "GOOG", "HOOD", "BABA", "IBM", "AVGO", "GUSH", "VOO",
        "MSFT", "TSM", "NFLX", "MP", "AAL", "URI", "DAL", "BBAI", "QUBT", "AMD", "SMCI",
        "ORCL", "TLT"
    ])
else:
    universe = [
        "EURUSD=X", "EURJPY=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "CADJPY=X", "USDCHF=X",
        "HKDJPY=X", "USDCAD=X", "USDCNY=X", "EURGBP=X", "EURCAD=X", "NZDJPY=X", "USDKRW=X",
        "USDHKD=X", "EURHKD=X", "GBPHKD=X", "GBPJPY=X", "CNHJPY=X", "AUDJPY=X", "GBPCAD=X"
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
        ["Open", "High", "Low", "Close"]
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

    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
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
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
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
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5 * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
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
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars) + 1):]) > 0))

    def _confirmed_down(from_time):
        seg = s.loc[from_time:]
        return bool(len(seg) >= int(confirm_bars) + 1 and np.all(np.diff(seg.iloc[-(int(confirm_bars) + 1):]) < 0))

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
    if ohlc is None or ohlc.empty or not {"High", "Low", "Close"}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
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
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


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
    std = float(np.sqrt(np.sum(resid ** 2) / dof))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
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
    below = p < l
    above = p > u

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
    p = p[ok]
    y = y[ok]
    u = u[ok]
    l = l[ok]

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
        window = touch_mask.iloc[j0:loc + 1]
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
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
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
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    return macd.reindex(s.index), sig.reindex(s.index), hist.reindex(s.index)


def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9, norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),) * 3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=int(signal), adjust=False).mean()
    minp = max(10, norm_win // 10)

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
    minp = max(10, norm_win // 10)
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
    minp = max(10, int(norm_win) // 10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
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

# =========================
# Part 5/10 — bullbear.py
# (unchanged in this batch)
# =========================
# NOTE: The rest of the original code (Parts 5–8 helpers) remains unchanged and continues in Batch 2/3.

# =========================
# Part 9/10 — bullbear.py
# ---------------------------
# Tabs
# =========================

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
      div[data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        gap: 0.40rem !important;
        padding: 0.25rem 0.25rem 0.40rem 0.25rem !important;
        border-bottom: 1px solid rgba(49, 51, 63, 0.14) !important;
      }
      div[data-baseweb="tab"] { flex: 0 0 auto !important; }
      div[data-baseweb="tab"] > button {
        padding: 0.40rem 0.85rem !important;
        border: 1px solid rgba(49, 51, 63, 0.20) !important;
        border-radius: 6px !important;
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
      div[data-baseweb="tab"] > button[aria-selected="true"] p { color: white !important; }
      div[data-baseweb="tab"] p { margin: 0 !important; font-size: 0.90rem !important; }

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

# UPDATED (THIS REQUEST):
# - Removed tabs: "NPX 0.5-Cross Scanner", "Fib NPX 0.0 Signal Scanner"
# - Added tab: "Buy Candidates"
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "Buy Candidates",
    "Slope Direction Scan",
    "Trendline Direction Lists",
    "NTD Hot List",
    "NTD NPX 0.0-0.2 Scanner",
    "Uptrend vs Downtrend",
    "Ichimoku Kijun Scanner",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)"
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

        # NOTE: Tab 1 charts are unchanged in this batch and continue in Batch 2/3.

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower": st.session_state.fc_ci.iloc[:, 0],
            "Upper": st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")


# =========================
# Part 10/10 — bullbear.py
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

            # UPDATED (THIS REQUEST): regression line + ±2σ lines on enhanced daily chart
            yhat_e, up_e, lo_e, m_e, r2_e = regression_with_band(df_show, lookback=int(slope_lb_daily), z=2.0)
            if not yhat_e.empty:
                ax.plot(yhat_e.index, yhat_e.values, "-", linewidth=2.0,
                        label=f"Regression ({slope_lb_daily} bars, {fmt_slope(m_e)}/bar)")
            if (not up_e.empty) and (not lo_e.empty):
                ax.plot(up_e.index, up_e.values, "--", linewidth=2.0, color="black", alpha=0.85, label="Reg +2σ")
                ax.plot(lo_e.index, lo_e.values, "--", linewidth=2.0, color="black", alpha=0.85, label="Reg -2σ")

            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                if fibs_d:
                    x0, x1 = df_show.index[0], df_show.index[-1]
                    for lbl, y in fibs_d.items():
                        ax.hlines(y, xmin=x0, xmax=x1, linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_d.items():
                        ax.text(x1, y, f" {lbl}", va="center")

            npx_d_show = compute_normalized_price(df_show, window=ntd_window)
            fib_sig_d = _fib_npx_zero_signal_series(
                close=df_show,
                npx=npx_d_show,
                prox=sr_prox_pct,
                lookback_bars=int(max(3, rev_horizon)),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if isinstance(fib_sig_d, dict):
                annotate_fib_npx_signal(ax, fib_sig_d)

            macd_sig = find_macd_hma_sr_signal(df_show, hma_d_show, macd_d, sup_d_show, res_d_show, global_m_d, prox=sr_prox_pct)
            macd_txt = "MACD/HMA55: n/a"
            if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
                macd_txt = f"MACD/HMA55: {macd_sig['side']} @ {fmt_price_val(macd_sig['price'])}"
                annotate_macd_signal(ax, macd_sig["time"], macd_sig["price"], macd_sig["side"])
            ax.text(0.01, 0.98, macd_txt, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8))

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
            style_axes(ax)
            st.pyplot(fig)

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

        if view in ("Intraday", "Both"):
            render_hourly_views(
                sel=st.session_state.ticker,
                intraday=st.session_state.intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.get("hour_range", "24h"),
                is_forex=(mode == "Forex")
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower": ci.iloc[:, 0],
            "Upper": ci.iloc[:, 1]
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
# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Lists symbols where the latest NTD is below -0.75 (using latest intraday for hourly; daily uses daily close).")

    scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
    run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

    if run_scan:
        rows = []
        if scan_frame.startswith("Hourly"):
            period = "1d"
            for sym in universe:
                val, ts = last_hourly_ntd_value(sym, ntd_window, period=period)
                if np.isfinite(val) and val < -0.75:
                    npx_val, _ = last_hourly_npx_value(sym, ntd_window, period=period)
                    rows.append({
                        "Symbol": sym,
                        "NTD": float(val),
                        "NPX (Norm Price)": float(npx_val) if np.isfinite(npx_val) else np.nan,
                        "Time": ts
                    })
        else:
            for sym in universe:
                val, ts = last_daily_ntd_value(sym, ntd_window)
                if np.isfinite(val) and val < -0.75:
                    npx_val, _ = last_daily_npx_value(sym, ntd_window)
                    rows.append({
                        "Symbol": sym,
                        "NTD": float(val),
                        "NPX (Norm Price)": float(npx_val) if np.isfinite(npx_val) else np.nan,
                        "Time": ts
                    })

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values("NTD")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 6: LONG-TERM HISTORY
# ---------------------------
with tab6:
    st.header("Long-Term History")
    sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    try:
        smax = fetch_hist_max(sel_lt)
    except Exception:
        smax = pd.Series(dtype=float)

    if smax is None or smax.dropna().empty:
        st.warning("No long-term history available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: RECENT BUY SCANNER
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner — Daily NPX↑NTD in Uptrend (Stocks + Forex)")
    st.caption(
        "Lists symbols (in the current mode's universe) where **NPX (normalized price)** most recently crossed "
        "**ABOVE** the **NTD** line (the green circle condition) **AND** the DAILY chart-area global trendline "
        "(in the selected Daily view range) is **upward**."
    )

    max_bars = st.slider("Max bars since NPX↑NTD cross", 0, 20, 2, 1, key="buy_scan_npx_max_bars")
    run_buy_scan = st.button("Run Recent BUY Scan", key="btn_run_recent_buy_scan_npx")

    if run_buy_scan:
        rows = []
        for sym in universe:
            r = last_daily_npx_cross_up_in_uptrend(sym, ntd_win=ntd_window, daily_view_label=daily_view)
            if r is not None and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)

        if not rows:
            st.info("No recent NPX↑NTD crosses found in an upward daily global trend (within the selected bar window).")
        else:
            out = pd.DataFrame(rows)
            if "Bars Since" in out.columns:
                out["Bars Since"] = out["Bars Since"].astype(int)
            if "Global Slope" in out.columns:
                out["Global Slope"] = out["Global Slope"].astype(float)
            out = out.sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 8: BUY CANDIDATES (NEW — THIS REQUEST)
#   - replaces removed NPX 0.5-Cross Scanner tab
#   - list symbols where price crossed ABOVE Ichimoku Kijun recently (1–2 bars)
#   - AND Regression line slope > 0
# ---------------------------
with tab8:
    st.header("Buy Candidates")
    st.caption(
        "Lists symbols where price **recently crossed above** the **Ichimoku Kijun** line "
        "within the last **N bars** (excluding 0 bars), and the **daily regression slope > 0** "
        "(uses selected Daily view range + your Ichimoku settings)."
    )

    c1, c2 = st.columns(2)
    max_bars_kijun = c1.slider("Cross must be within last N bars (default 2)", 1, 20, 2, 1, key="buycand_within_n")
    run_buycand = c2.button("Run Buy Candidates Scan", key=f"btn_run_buycand_{mode}")

    if run_buycand:
        rows = []
        for sym in universe:
            r = last_daily_kijun_cross_up(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                conv=int(ichi_conv),
                base=int(ichi_base),
                span_b=int(ichi_spanb),
                within_last_n_bars=int(max_bars_kijun),
            )
            if r is None:
                continue

            # Enforce "crossed by 1–N bars" (exclude same-bar/0)
            try:
                bars_since = int(r.get("Bars Since Cross", 9999))
            except Exception:
                bars_since = 9999
            if bars_since < 1 or bars_since > int(max_bars_kijun):
                continue

            # Enforce Regression line > 0
            try:
                m = float(r.get("Slope", np.nan))
            except Exception:
                m = np.nan
            if not (np.isfinite(m) and m > 0.0):
                continue

            rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            if "Bars Since Cross" in out.columns:
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
            if "Slope" in out.columns:
                out["Slope"] = out["Slope"].astype(float)
            if "R2" in out.columns:
                out["R2"] = out["R2"].astype(float)

            out = out.sort_values(["Bars Since Cross", "R2", "Slope"], ascending=[True, False, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
# ---------------------------
# TABS (UPDATED — 19 total)
#   - Replaced: "NPX 0.5-Cross Scanner"  →  "Buy Candidates"
#   - Replaced: "Fib NPX 0.0 Signal Scanner"  →  "Fib Reversal Scanner"
#   - Added: "Band Bounce Scanner" (new 19th tab)
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "Buy Candidates",
    "Fib Reversal Scanner",
    "Slope Direction Scan",
    "Trendline Direction Lists",
    "NTD Hot List",
    "NTD NPX 0.0-0.2 Scanner",
    "Uptrend vs Downtrend",
    "Ichimoku Kijun Scanner",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)",
    "Band Bounce Scanner"
])


# =========================
# PATCH (THIS REQUEST #2):
# In TAB 2 "Enhanced Forecast" DAILY section, add regression + ±2σ lines.
# Insert inside: `if view in ("Daily", "Both"):` after you define `df_show`
# and after `ax.plot(df_show.index, df_show.values, label="History")`.
# =========================
# yhat_reg_d, up_reg_d, lo_reg_d, m_reg_d, r2_reg_d = regression_with_band(df_show, lookback=int(slope_lb_daily), z=2.0)
# if not yhat_reg_d.dropna().empty:
#     ax.plot(yhat_reg_d.index, yhat_reg_d.values, "-", linewidth=2.0,
#             label=f"Regression (w={slope_lb_daily}) ({fmt_slope(m_reg_d)}/bar)")
# if not up_reg_d.dropna().empty and not lo_reg_d.dropna().empty:
#     ax.plot(up_reg_d.index, up_reg_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="Regression +2σ")
#     ax.plot(lo_reg_d.index, lo_reg_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="Regression -2σ")


# ---------------------------
# TAB 9: FIB REVERSAL SCANNER (REPLACED TAB — removed "Fib NPX 0.0 Signal Scanner")
# ---------------------------
with tab9:
    st.header("Fib Reversal Scanner")
    st.caption(
        "Daily-only scan for **confirmed Fibonacci reversals** (same logic used in charts):\n"
        "• **BUY:** price touched near Fib **100%** (low) then printed consecutive higher closes\n"
        "• **SELL:** price touched near Fib **0%** (high) then printed consecutive lower closes\n\n"
        "Uses the selected Daily view range."
    )

    c1, c2, c3 = st.columns(3)
    fib_confirm = c1.slider("Confirm bars (default 2)", 1, 5, 2, 1, key="fibrev_confirm_bars")
    fib_lb = c2.slider("Lookback window (bars)", 30, 365, 120, 15, key="fibrev_lookback_bars")
    run_fibrev = c3.button("Run Fib Reversal Scan", key=f"btn_run_fibrev_{mode}")

    if run_fibrev:
        rows_buy, rows_sell = [], []

        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue
                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if close_show.empty or len(close_show) < 6:
                    continue

                sig = fib_reversal_trigger_from_extremes(
                    close_show,
                    proximity_pct_of_range=0.02,
                    confirm_bars=int(fib_confirm),
                    lookback_bars=int(fib_lb),
                )
                if not isinstance(sig, dict):
                    continue

                row = {
                    "Symbol": sym,
                    "Side": sig.get("side", ""),
                    "From Level": sig.get("from_level", ""),
                    "Touch Time": sig.get("touch_time", None),
                    "Touch Price": sig.get("touch_price", np.nan),
                    "Last Time": sig.get("last_time", None),
                    "Last Price": sig.get("last_price", np.nan),
                }

                if str(row["Side"]).upper().startswith("B"):
                    rows_buy.append(row)
                else:
                    rows_sell.append(row)
            except Exception:
                continue

        left, right = st.columns(2)
        with left:
            st.subheader("Fib BUY — 100% touch + confirmed reversal up")
            if not rows_buy:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_buy).sort_values(["Touch Time"], ascending=False)
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Fib SELL — 0% touch + confirmed reversal down")
            if not rows_sell:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_sell).sort_values(["Touch Time"], ascending=False)
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 10: Slope Direction Scan
# ---------------------------
with tab10:
    st.header("Slope Direction Scan")
    st.caption(
        "Lists symbols whose **current DAILY global trendline slope** is **up** vs **down** "
        "(based on the selected Daily view range)."
    )

    run_slope = st.button("Run Slope Direction Scan", key=f"btn_run_slope_dir_{mode}")

    if run_slope:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue
            rows.append({
                "Symbol": sym,
                "Slope": float(m),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "AsOf": ts
            })

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            up = out[out["Slope"] > 0].sort_values(["Slope"], ascending=False)
            dn = out[out["Slope"] < 0].sort_values(["Slope"], ascending=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Upward Slope")
                if up.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(up.reset_index(drop=True), use_container_width=True)

            with right:
                st.subheader("Downward Slope")
                if dn.empty:
                    st.info("No matches.")
                else:
                    st.dataframe(dn.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 11: Trendline Direction Lists
# ---------------------------
with tab11:
    st.header("Trendline Direction Lists")
    st.caption(
        "Displays symbols whose **current DAILY chart-area global trendline** is:\n"
        "• **Upward** (green dashed global trendline)\n"
        "• **Downward** (red dashed global trendline)\n\n"
        "Uses the selected Daily view range."
    )

    run_trend_lists = st.button("Run Trendline Direction Lists", key=f"btn_run_trendline_lists_{mode}")

    if run_trend_lists:
        up_rows, dn_rows = [], []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            if not np.isfinite(npx_last):
                continue

            if float(m) >= 0.0 and float(npx_last) < 0.0:
                up_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})
            elif float(m) < 0.0 and float(npx_last) > 0.5:
                dn_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})

        left, right = st.columns(2)

        with left:
            st.subheader("Upward Trend (Green dashed)")
            if not up_rows:
                st.info("No matches.")
            else:
                out_up = pd.DataFrame(up_rows).sort_values(["Symbol"], ascending=True)
                st.dataframe(out_up.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Downward Trend (Red dashed)")
            if not dn_rows:
                st.info("No matches.")
            else:
                out_dn = pd.DataFrame(dn_rows).sort_values(["Symbol"], ascending=True)
                st.dataframe(out_dn.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run Trendline Direction Lists** to scan the current universe.")

# ---------------------------
# TAB 12: NTD Hot List
# ---------------------------
with tab12:
    st.header("NTD Hot List")
    st.caption(
        "Lists symbols where the **daily regression slope > 0** (upward trend) "
        "and **NPX (Norm Price)** is between **0.0** and **0.5** (inclusive), "
        "using the selected Daily view range."
    )

    run_hot = st.button("Run NTD Hot List", key=f"btn_run_ntd_hot_{mode}")

    if run_hot:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m) or float(m) <= 0.0:
                continue

            npx_last, npx_ts = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            if not np.isfinite(npx_last):
                continue

            if 0.0 <= float(npx_last) <= 0.5:
                rows.append({
                    "Symbol": sym,
                    "Slope": float(m),
                    "NPX (Norm Price)": float(npx_last),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "AsOf": ts
                })

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out = out.sort_values(["Slope", "NPX (Norm Price)"], ascending=[False, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run NTD Hot List** to scan the current universe.")

# ---------------------------
# TAB 13: NTD NPX 0.0–0.2 Scanner
# ---------------------------
with tab13:
    st.header("NTD NPX 0.0-0.2 Scanner")
    st.caption(
        "Scans the current universe for symbols where **NPX (Norm Price)** is between **0.0** and **0.2** "
        "and is **heading up**, split into two lists:\n"
        "• **List 1:** regression slope **> 0**\n"
        "• **List 2:** regression slope **< 0**\n\n"
        "Includes **NPX (last)** and **R²** of the regression slope line."
    )

    c1, c2 = st.columns(2)
    npx_up_bars = c1.slider("NPX heading-up confirmation (consecutive bars)", 1, 5, 1, 1, key="npx_02_up_bars")
    run_npx02 = c2.button("Run NTD NPX 0.0-0.2 Scan", key=f"btn_run_npx02_{mode}")

    if run_npx02:
        rows_up_slope, rows_dn_slope = [], []

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            npx_s = daily_npx_series_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            npx_s = _coerce_1d_series(npx_s).dropna()
            if npx_s.empty or len(npx_s) < 2:
                continue

            npx_last = float(npx_s.iloc[-1]) if np.isfinite(npx_s.iloc[-1]) else np.nan
            if not np.isfinite(npx_last):
                continue

            if not (0.0 <= float(npx_last) <= 0.2):
                continue

            if not _series_heading_up(npx_s, confirm_bars=int(npx_up_bars)):
                continue

            row = {
                "Symbol": sym,
                "Slope": float(m),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "NPX (Norm Price)": float(npx_last),
                "AsOf": ts
            }

            if float(m) > 0.0:
                rows_up_slope.append(row)
            elif float(m) < 0.0:
                rows_dn_slope.append(row)

        left, right = st.columns(2)

        with left:
            st.subheader("List 1 — Slope > 0 and NPX 0.0–0.2 heading up")
            if not rows_up_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_up_slope)
                out = out.sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("List 2 — Slope < 0 and NPX 0.0–0.2 heading up")
            if not rows_dn_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_dn_slope)
                out = out.sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 14: Uptrend vs Downtrend
# ---------------------------
with tab14:
    st.header("Uptrend vs Downtrend")
    st.caption(
        "Lists symbols where the price has **reversed from support heading up** (Daily view), split into:\n"
        "• **(a) Uptrend:** Slope > 0 and price reversed from support heading up\n"
        "• **(b) Downtrend:** Slope < 0 and price reversed from support heading up\n\n"
        "Support reversal uses the same Daily S/R proximity logic and confirmation bars."
    )

    c1, c2 = st.columns(2)
    hz_sr = c1.slider("Support-touch lookback window (bars)", 1, 60, int(max(3, rev_horizon)), 1, key="ud_sr_hz")
    run_ud = c2.button("Run Uptrend vs Downtrend Scan", key=f"btn_run_ud_{mode}")

    if run_ud:
        rows_uptrend, rows_downtrend = [], []

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            rev = daily_support_reversal_heading_up(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=int(sr_lb_daily),
                prox=float(sr_prox_pct),
                bars_confirm=int(rev_bars_confirm),
                horizon=int(hz_sr)
            )
            if rev is None:
                continue

            row = dict(rev)
            row["Slope"] = float(m)
            row["R2"] = float(r2) if np.isfinite(r2) else np.nan
            row["AsOf"] = ts

            if float(m) > 0.0:
                rows_uptrend.append(row)
            elif float(m) < 0.0:
                rows_downtrend.append(row)

        left, right = st.columns(2)

        with left:
            st.subheader("(a) Uptrend — Slope > 0 and Support Reversal heading up")
            if not rows_uptrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_uptrend)
                if "Bars Since Touch" in out.columns:
                    out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Downtrend — Slope < 0 and Support Reversal heading up")
            if not rows_downtrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_downtrend)
                if "Bars Since Touch" in out.columns:
                    out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 15: Ichimoku Kijun Scanner
# ---------------------------
with tab15:
    st.header("Ichimoku Kijun Scanner")
    st.caption(
        "Daily-only scanner (matches the **Price Chart**):\n"
        "• **List 1:** regression line slope **> 0** AND price **crossed above** the **Ichimoku Kijun** line, **heading up**\n"
        "• **List 2:** regression line slope **< 0** AND price **crossed above** the **Ichimoku Kijun** line, **heading up**\n\n"
        "Includes **Price@Cross**, **Kijun@Cross**, and **R²**."
    )

    c1, c2 = st.columns(2)
    kijun_within = c1.slider("Cross must be within last N bars", 0, 60, 5, 1, key="kijun_within_n")
    run_kijun = c2.button("Run Ichimoku Kijun Scan", key=f"btn_run_kijun_scan_{mode}")

    if run_kijun:
        rows_list1, rows_list2 = [], []
        for sym in universe:
            r = last_daily_kijun_cross_up(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                conv=int(ichi_conv),
                base=int(ichi_base),
                span_b=int(ichi_spanb),
                within_last_n_bars=int(kijun_within),
            )
            if r is None:
                continue

            try:
                m = float(r.get("Slope", np.nan))
            except Exception:
                m = np.nan
            if not np.isfinite(m):
                continue

            if m > 0.0:
                rows_list1.append(r)
            elif m < 0.0:
                rows_list2.append(r)

        left, right = st.columns(2)
        with left:
            st.subheader("Ichimoku Kijun List 1 — Slope > 0 and Kijun Cross-Up (heading up)")
            if not rows_list1:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list1)
                if "Bars Since Cross" in out.columns:
                    out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Ichimoku Kijun List 2 — Slope < 0 and Kijun Cross-Up (heading up)")
            if not rows_list2:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list2)
                if "Bars Since Cross" in out.columns:
                    out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 16: R² > 45% Daily/Hourly
# ---------------------------
with tab16:
    st.header("R² > 45% Daily/Hourly")
    st.caption(
        "Shows symbols where the **R²** (regression fit quality) is **> 45%** using:\n"
        "• **Daily:** regression_with_band on daily close (lookback = Daily slope lookback)\n"
        "• **Hourly (intraday):** regression_with_band on intraday close (lookback = Hourly slope lookback)\n\n"
        "No changes to any existing tabs/views/buttons — this is an added scanner tab only."
    )

    c1, c2, c3 = st.columns(3)
    r2_thr = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan")
    hour_period = c2.selectbox("Hourly intraday period", ["1d", "2d", "4d"], index=0, key="r2_hour_period")
    run_r2 = c3.button("Run R² Scan", key=f"btn_run_r2_scan_{mode}")

    if run_r2:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2_d) and float(r2_d) > float(r2_thr):
                daily_rows.append({
                    "Symbol": sym,
                    "R2": float(r2_d),
                    "Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                    "AsOf": ts_d
                })

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) > float(r2_thr):
                hourly_rows.append({
                    "Symbol": sym,
                    "R2": float(r2_h),
                    "Slope": float(m_h) if np.isfinite(m_h) else np.nan,
                    "AsOf": ts_h,
                    "Period": str(hour_period)
                })

        left, right = st.columns(2)

        with left:
            st.subheader("Daily — R² > threshold")
            if not daily_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(daily_rows)
                out = out.sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly (intraday {hour_period}) — R² > threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows)
                out = out.sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 17: R² < 45% Daily/Hourly
# ---------------------------
with tab17:
    st.header("R² < 45% Daily/Hourly")
    st.caption(
        "Shows symbols where the **R²** (regression fit quality) is **< threshold** using:\n"
        "• **Daily:** regression_with_band on daily close (lookback = Daily slope lookback)\n"
        "• **Hourly (intraday):** regression_with_band on intraday close (lookback = Hourly slope lookback)\n\n"
        "This is a scanner tab only; it does not change any charts or logic elsewhere."
    )

    c1, c2, c3 = st.columns(3)
    r2_thr_lo = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan_lo")
    hour_period_lo = c2.selectbox("Hourly intraday period", ["1d", "2d", "4d"], index=0, key="r2_hour_period_lo")
    run_r2_lo = c3.button("Run R² Low Scan", key=f"btn_run_r2_scan_lo_{mode}")

    if run_r2_lo:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2_d) and float(r2_d) < float(r2_thr_lo):
                daily_rows.append({
                    "Symbol": sym,
                    "R2": float(r2_d),
                    "Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                    "AsOf": ts_d
                })

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period_lo), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) < float(r2_thr_lo):
                hourly_rows.append({
                    "Symbol": sym,
                    "R2": float(r2_h),
                    "Slope": float(m_h) if np.isfinite(m_h) else np.nan,
                    "AsOf": ts_h,
                    "Period": str(hour_period_lo)
                })

        left, right = st.columns(2)

        with left:
            st.subheader("Daily — R² < threshold")
            if not daily_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(daily_rows)
                out = out.sort_values(["R2", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly (intraday {hour_period_lo}) — R² < threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows)
                out = out.sort_values(["R2", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 18: R² Sign ±2σ Proximity (Daily)
# ---------------------------
with tab18:
    st.header("R² Sign ±2σ Proximity (Daily)")
    st.caption(
        "Daily-only scan (uses the selected **Daily view range** and **Daily slope lookback**):\n"
        "Creates four lists:\n"
        "1) **R² > 0** and price **near Lower -2σ**\n"
        "2) **R² > 0** and price **near Upper +2σ**\n"
        "3) **R² < 0** and price **near Lower -2σ**\n"
        "4) **R² < 0** and price **near Upper +2σ**\n\n"
        "“Near” uses the existing sidebar **S/R proximity (%)** value."
    )

    run_band_scan = st.button("Run R² Sign ±2σ Proximity Scan (Daily)", key=f"btn_run_r2_sign_band_scan_{mode}")

    if run_band_scan:
        rows_pos_lower, rows_pos_upper = [], []
        rows_neg_lower, rows_neg_upper = [], []

        for sym in universe:
            r = daily_r2_band_proximity(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                prox=float(sr_prox_pct)
            )
            if r is None:
                continue

            r2v = r.get("R2", np.nan)
            if not np.isfinite(r2v):
                continue

            near_lo = bool(r.get("Near Lower", False))
            near_up = bool(r.get("Near Upper", False))

            row = {k: v for k, v in r.items() if k not in ("Near Lower", "Near Upper")}

            if float(r2v) > 0.0:
                if near_lo:
                    rows_pos_lower.append(row)
                if near_up:
                    rows_pos_upper.append(row)
            elif float(r2v) < 0.0:
                if near_lo:
                    rows_neg_lower.append(row)
                if near_up:
                    rows_neg_upper.append(row)

        st.info(f"Near threshold = ±{sr_prox_pct*100:.3f}% (from sidebar S/R proximity %)")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.subheader("R² > 0  •  Near Lower -2σ")
            if not rows_pos_lower:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos_lower)
                if "AbsDist Lower (%)" in out.columns:
                    out = out.sort_values(["AbsDist Lower (%)", "R2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with r1c2:
            st.subheader("R² > 0  •  Near Upper +2σ")
            if not rows_pos_upper:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos_upper)
                if "AbsDist Upper (%)" in out.columns:
                    out = out.sort_values(["AbsDist Upper (%)", "R2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.subheader("R² < 0  •  Near Lower -2σ")
            if not rows_neg_lower:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg_lower)
                if "AbsDist Lower (%)" in out.columns:
                    out = out.sort_values(["AbsDist Lower (%)", "R2"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with r2c2:
            st.subheader("R² < 0  •  Near Upper +2σ")
            if not rows_neg_upper:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg_upper)
                if "AbsDist Upper (%)" in out.columns:
                    out = out.sort_values(["AbsDist Upper (%)", "R2"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 19: Band Bounce Scanner (NEW)
# ---------------------------
with tab19:
    st.header("Band Bounce Scanner")
    st.caption(
        "Scans for the most recent **band-bounce** signal (same ±2σ bounce logic used on charts):\n"
        "• **Daily:** last_band_bounce_signal_daily\n"
        "• **Hourly:** last_band_bounce_signal_hourly\n"
    )

    c1, c2, c3 = st.columns(3)
    max_bars_since = c1.slider("Max bars since signal", 0, 60, 5, 1, key="bbounce_max_bars")
    hour_period_bb = c2.selectbox("Hourly intraday period", ["1d", "2d", "4d"], index=0, key="bbounce_hour_period")
    run_bbounce = c3.button("Run Band Bounce Scan", key=f"btn_run_bbounce_{mode}")

    if run_bbounce:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            r_d = last_band_bounce_signal_daily(sym, slope_lb=int(slope_lb_daily))
            if isinstance(r_d, dict) and int(r_d.get("Bars Since", 9999)) <= int(max_bars_since):
                daily_rows.append(r_d)

            r_h = last_band_bounce_signal_hourly(sym, period=str(hour_period_bb), slope_lb=int(slope_lb_hourly))
            if isinstance(r_h, dict) and int(r_h.get("Bars Since", 9999)) <= int(max_bars_since):
                hourly_rows.append(r_h)

        left, right = st.columns(2)
        with left:
            st.subheader("Daily band-bounce signals")
            if not daily_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(daily_rows)
                out = out.sort_values(["Bars Since", "R2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly band-bounce signals ({hour_period_bb})")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows)
                out = out.sort_values(["Bars Since", "R2"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)
