# /mount/src/stock-wizard/bullbear.py
# =========================
# Batch 1/3 — Beginning through Tabs 1–4 (UPDATED)
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
     UPDATED: rectangular ribbon tabs
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
    border-radius: 6px !important;
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
     UPDATED: chart container styling
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
REFRESH_INTERVAL = 120
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
    st.session_state.asset_mode = "Forex"

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


ALERT_TEXT = (
    "ALERT: Trend may be changing - Open trade position with caution while still following the signals on the chat."
)

FIB_ALERT_TEXT = (
    "ALERT: Fibonacci Guidance — Prices often reverse at the 100% and 0% lines. It's essential to implement risk "
    "management when trading near these Fibonacci levels."
)


def format_trade_instruction(
    trend_slope: float,
    buy_val: float,
    sell_val: float,
    close_val: float,
    symbol: str,
    global_trend_slope: float = None
) -> str:
    """
    Shows BUY/SELL only when Global Trendline slope and Local Slope agree.
    Falls back to prior behavior if global_trend_slope is None.
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


# ---------------------------
# Gapless (continuous) intraday prices
# ---------------------------
def make_gapless_ohlc(
    df: pd.DataFrame,
    price_cols=("Open", "High", "Low", "Close"),
    gap_mult: float = 12.0,
    min_gap_seconds: float = 3600.0
) -> pd.DataFrame:
    """Remove price gaps at session breaks by applying a cumulative offset."""
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
    return [int(i) for i in idxer if int(i) >= 0]


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

# UPDATED (THIS REQUEST): remove News entirely (sidebar + markers + tables)
# Keep sessions option only.
if mode == "Forex":
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_sessions_pst = False

# ---------------------------
# Universe
# ---------------------------
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


def fib_reversal_trigger_from_extremes(
    series_like,
    proximity_pct_of_range: float = 0.02,
    confirm_bars: int = 2,
    lookback_bars: int = 60
):
    """Confirmed fib reversal trigger from extremes."""
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


def slope_reversal_probability(
    series_like,
    current_slope: float,
    hist_window: int = 240,
    slope_window: int = 60,
    horizon: int = 15
) -> float:
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


def find_band_bounce_signal(price: pd.Series, upper_band: pd.Series, lower_band: pd.Series, slope_val: float):
    """Detect most recent BUY/SELL based on a 'bounce' off ±2σ band."""
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
        ax.text(ts, px, f"  {label}", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
    else:
        ax.scatter([ts], [px], marker="X", s=90, color="tab:red", zorder=7)
        label = "SELL" if not note else f"SELL {note}"
        ax.text(ts, px, f"  {label}", va="top", fontsize=9, color="tab:red", fontweight="bold")


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


# ---------------------------
# Ichimoku, Supertrend, PSAR (needed for charts)
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
    kijun = ((h.rolling(base).max() + l.rolling(base).min()) / 2.0)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = ((h.rolling(span_b).max() + l.rolling(span_b).min()) / 2.0)
    if shift_cloud:
        senkou_a = senkou_a.shift(base)
        senkou_b = senkou_b.shift(base)
        chikou = c.shift(-base)
    else:
        chikou = c
    return (tenkan.reindex(idx), kijun.reindex(idx), senkou_a.reindex(idx), senkou_b.reindex(idx), chikou.reindex(idx))


def _compute_atr_from_ohlc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(df["High"])
    low = _coerce_1d_series(df["Low"])
    close = _coerce_1d_series(df["Close"])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr.reindex(df.index)


def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return pd.DataFrame(columns=["ST", "in_uptrend"])
    ohlc = df[["High", "Low", "Close"]].copy()
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

        if ohlc["Close"].iloc[i] > upperband.iloc[i - 1]:
            in_uptrend.iloc[i] = True
        elif ohlc["Close"].iloc[i] < lowerband.iloc[i - 1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i - 1]
            if in_uptrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if (not in_uptrend.iloc[i]) and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]

        st_line.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]

    return pd.DataFrame({"ST": st_line, "in_uptrend": in_uptrend})


def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        return pd.DataFrame(columns=["PSAR", "in_uptrend"])
    high = _coerce_1d_series(df["High"])
    low = _coerce_1d_series(df["Low"])
    idx = high.index.union(low.index)
    high = high.reindex(idx)
    low = low.reindex(idx)

    psar = pd.Series(index=idx, dtype=float)
    in_uptrend = pd.Series(index=idx, dtype=bool)

    in_uptrend.iloc[0] = True
    psar.iloc[0] = float(low.iloc[0])
    ep = float(high.iloc[0])
    af = step

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i - 1]
        if in_uptrend.iloc[i - 1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i],
                               float(low.iloc[i - 1]),
                               float(low.iloc[i - 2]) if i >= 2 else float(low.iloc[i - 1]))
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
                               float(high.iloc[i - 1]),
                               float(high.iloc[i - 2]) if i >= 2 else float(high.iloc[i - 1]))
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
    end_d = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes


def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close = session_markers_for_index(idx, NY_TZ, 8, 17)
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
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=1.6, label="London Open"),
        Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], color="tab:orange", linestyle="-", linewidth=1.6, label="New York Open"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels


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
        ax.scatter(price.index[enter_from_below], [0.92] * int(enter_from_below.sum()),
                   marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above], [0.92] * int(enter_from_above.sum()),
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


def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)


# =========================
# Session state init
# =========================
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

# ---------------------------
# Shared hourly renderer (Stock & Forex)
# ---------------------------
def render_hourly_views(
    sel: str,
    intraday: pd.DataFrame,
    p_up: float,
    p_dn: float,
    hour_range_label: str,
    is_forex: bool
):
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
    if {"High", "Low", "Close"}.issubset(intraday.columns) and show_ichi:
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
        px_val = float(hc.iloc[-1])
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

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
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
        ax2.text(
            0.99, 0.02,
            f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
            transform=ax2.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
        )

    ax2.text(
        0.01, 0.02,
        f"Slope: {fmt_slope(slope_sig_h)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_h)}",
        transform=ax2.transAxes, ha="left", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )
    ax2.text(
        0.50, 0.02,
        f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
        transform=ax2.transAxes, ha="center", va="bottom",
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )

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

    handles, labels = ax2.get_legend_handles_labels()
    if ax2w is not None:
        h2, l2 = ax2w.get_legend_handles_labels()
        handles += h2
        labels += l2
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
    st.pyplot(fig2)

    return {"trade_instruction": instr_txt}


# =========================
# Tabs (UPDATED list)
#   - Removed: NPX 0.5-Cross Scanner, Fib NPX 0.0 Signal Scanner
#   - Added: Buy Candidates, Regression>0/Regression<0 with NBB [-1.0,-0.8], NPX 0.0–0.2 in NTD chart
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = st.tabs([
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
    "Regression & NBB (-1.0 to -0.8)",
    "NPX 0.0-0.2 in NTD Chart",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)"
])

# ---------------------------
# TAB 1: ORIGINAL FORECAST
# ---------------------------
with tab1:
    st.header("Original Forecast")
    st.info(
        "Pick a ticker; data is cached for ~2 minutes after first fetch. "
        "Charts stay on the last RUN ticker until you run again."
    )

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

        st.caption(
            f"**Displayed (last run):** {disp_ticker}  •  "
            f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}"
        )

        with fib_instruction_box.container():
            st.warning(FIB_ALERT_TEXT)
            st.caption(
                "Fibonacci Reversal Trigger (confirmed): "
                "BUY when price touches near the **100%** line then prints consecutive higher closes; "
                "SELL when price touches near the **0%** line then prints consecutive lower closes."
            )

        daily_instr_txt = None
        hourly_instr_txt = None
        daily_fib_trig = None

        if chart in ("Daily", "Both"):
            ema30 = df.ewm(span=30).mean()
            res_d = df.rolling(sr_lb_daily, min_periods=1).max()
            sup_d = df.rolling(sr_lb_daily, min_periods=1).min()

            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)
            rev_prob_d = slope_reversal_probability(df, m_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)
            piv = current_daily_pivots(df_ohlc)

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
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show = bb_up_d.reindex(df_show.index)
            bb_lo_d_show = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_show = compute_hma(df, period=hma_period).reindex(df_show.index)
            macd_d, macd_sig_d, _ = compute_macd(df_show)

            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, ax = plt.subplots(figsize=(14, 5))
            fig.subplots_adjust(bottom=0.30)

            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{disp_ticker} Daily — {daily_view} — "
                f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
            )

            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")

            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

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
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-",
                          linewidth=1.6, label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-",
                          linewidth=1.6, label=f"Support (w={sr_lb_daily})")
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
                ax.text(
                    0.99, 0.02,
                    f"Current price: {fmt_price_val(last_px_show)}{nbb_txt}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
                )

            ax.text(
                0.50, 0.02,
                f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
            )

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
            style_axes(ax)
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

        now_pst = datetime.now(PACIFIC)

        if "instr_daily_text" not in st.session_state:
            st.session_state.instr_daily_text = None
            st.session_state.instr_daily_updated_at = None
        if "instr_hourly_text" not in st.session_state:
            st.session_state.instr_hourly_text = None
            st.session_state.instr_hourly_updated_at = None

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
                st.error(daily_msg) if daily_instr_txt.startswith("ALERT:") else st.success(daily_msg)

            if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
                hourly_msg = f"Hourly (updated {_ts_str(st.session_state.get('instr_hourly_updated_at'))}): {hourly_instr_txt}"
                st.error(hourly_msg) if hourly_instr_txt.startswith("ALERT:") else st.success(hourly_msg)

            if isinstance(daily_fib_trig, dict):
                st.info(
                    f"Daily Fib Reversal Trigger: **{daily_fib_trig.get('side')}** "
                    f"(from {daily_fib_trig.get('from_level')}) • touch={daily_fib_trig.get('touch_time')} "
                    f"@ {fmt_price_val(daily_fib_trig.get('touch_price', np.nan))}"
                )
            else:
                st.caption("Daily Fib Reversal Trigger: none confirmed.")

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower": st.session_state.fc_ci.iloc[:, 0],
            "Upper": st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")

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

        # FIX (THIS REQUEST): unique key to avoid StreamlitDuplicateElementKey
        view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}_tab2_v3")

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

            # ADD (THIS REQUEST): regression line + ±2σ bands on enhanced daily chart
            yhat_e, up_e, lo_e, m_e, r2_e = regression_with_band(df_show, lookback=int(slope_lb_daily), z=2.0)
            if not yhat_e.dropna().empty:
                ax.plot(yhat_e.index, yhat_e.values, "-", linewidth=2.0,
                        label=f"Regression ({slope_lb_daily}) {fmt_slope(m_e)}/bar")
            if not up_e.dropna().empty and not lo_e.dropna().empty:
                ax.plot(up_e.index, up_e.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
                ax.plot(lo_e.index, lo_e.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")
                bounce_sig = find_band_bounce_signal(df_show, up_e, lo_e, m_e)
                if bounce_sig is not None:
                    annotate_crossover(ax, bounce_sig["time"], bounce_sig["price"], bounce_sig["side"])

            ax.text(
                0.50, 0.02,
                f"R² ({slope_lb_daily} bars): {fmt_r2(r2_e)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
            )

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
            f"P(slope reverses ≤ {rev_horizon} bars)": fmt_pct(
                slope_reversal_probability(df, m_d, rev_hist_lb, slope_lb_daily, rev_horizon)
            )
        })

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({
                "Hourly slope (reg band)": fmt_slope(m_h),
                "Hourly R²": fmt_r2(r2_h),
                f"P(slope reverses ≤ {rev_horizon} bars) hourly": fmt_pct(
                    slope_reversal_probability(hc, m_h, rev_hist_lb, slope_lb_hourly, rev_horizon)
                )
            })

# =========================
# Batch 1 ends here.
# Batch 2 will continue with Tabs 5–8.
# =========================
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
#   - Price crossed ABOVE Ichimoku Kijun within last 1–2 bars
#   - Regression slope > 0
# ---------------------------
with tab8:
    st.header("Buy Candidates")
    st.caption(
        "Shows symbols where price **recently crossed above Ichimoku Kijun** by **1–2 bars** "
        "and the **regression slope > 0** (Daily chart, selected Daily view range)."
    )

    run_buy_candidates = st.button("Run Buy Candidates Scan", key=f"btn_run_buy_candidates_{mode}")

    if run_buy_candidates:
        rows = []
        for sym in universe:
            r = last_daily_kijun_cross_up(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                conv=int(ichi_conv),
                base=int(ichi_base),
                span_b=int(ichi_spanb),
                within_last_n_bars=2,   # enforce “recent”
            )
            if r is None:
                continue

            try:
                b = int(r.get("Bars Since Cross", 9999))
            except Exception:
                b = 9999
            if b not in (1, 2):
                continue

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
# =========================
# PATCH 1 (THIS REQUEST): Remove NEWS UI (no news markers / no news tables)
#   Replace ONLY the existing "if mode == 'Forex': ... show_fx_news ..." block with this.
# =========================
if mode == "Forex":
    show_fx_news = False
    news_window_days = 7
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_fx_news = False
    news_window_days = 7
    show_sessions_pst = False


# =========================
# PATCH 2 (THIS REQUEST): Tabs list update
#   Replace ONLY the existing st.tabs([...]) assignment block with this.
#   (Removes: NPX 0.5-Cross Scanner, Fib NPX 0.0 Signal Scanner)
#   (Adds: Buy Candidates, NBB -0.8~-1.0 Lists)
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18 = st.tabs([
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
    "NBB -0.8~-1.0 Lists",
    "NTD NPX 0.0-0.2 Scanner",
    "Uptrend vs Downtrend",
    "Ichimoku Kijun Scanner",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)"
])


# =========================
# PATCH 3 (THIS REQUEST): TAB 2 fix
#   Replace ONLY the entire existing "with tab2:" block with this updated version.
#   Fixes StreamlitDuplicateElementKey by changing the radio key
#   Adds regression line + ±2σ lines on the Enhanced DAILY chart
# =========================
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
        view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}_v2")  # KEY FIX

        if view in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)
            res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
            sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()
            hma_d_show = compute_hma(df_show, period=hma_period)
            macd_d, macd_sig_d, _ = compute_macd(df_show)

            # NEW: regression + ±2σ on Enhanced Daily chart
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df_show, lookback=int(slope_lb_daily), z=2.0)

            fig, ax = plt.subplots(figsize=(14, 5))
            fig.subplots_adjust(bottom=0.30)
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")

            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if (yhat_d is not None) and (not _coerce_1d_series(yhat_d).dropna().empty):
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2.0,
                        label=f"Regression ({slope_lb_daily}) {fmt_slope(m_d)}/bar  •  R² {fmt_r2(r2_d)}")
            if (up_d is not None) and (lo_d is not None) and (not _coerce_1d_series(up_d).dropna().empty) and (not _coerce_1d_series(lo_d).dropna().empty):
                ax.plot(up_d.index, up_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
                ax.plot(lo_d.index, lo_d.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

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
                hour_range_label=st.session_state.get("hour_range","24h"),
                is_forex=(mode == "Forex")
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:, 0],
            "Upper":    ci.iloc[:, 1]
        }, index=idx))


# =========================
# BATCH 3: REMAINDER OF TABS (Tabs 9 through 18)
# =========================

# ---------------------------
# Helper for NEW Tab 12 (NBB lists)
# ---------------------------
@st.cache_data(ttl=120)
def daily_last_nbb_in_view(symbol: str, daily_view_label: str, bb_window: int, bb_sigma: float, use_ema_mid: bool):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return np.nan, None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty:
            return np.nan, None
        _, _, _, _, nbb = compute_bbands(close_show, window=int(bb_window), mult=float(bb_sigma), use_ema=bool(use_ema_mid))
        nbb = _coerce_1d_series(nbb).dropna()
        if nbb.empty:
            return np.nan, close_show.index[-1]
        return float(nbb.iloc[-1]), nbb.index[-1]
    except Exception:
        return np.nan, None

# ---------------------------
# TAB 9: Slope Direction Scan
# ---------------------------
with tab9:
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
# TAB 10: Trendline Direction Lists
# ---------------------------
with tab10:
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
# TAB 11: NTD Hot List
# ---------------------------
with tab11:
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
# TAB 12: NBB -0.8~-1.0 Lists (NEW — THIS REQUEST)
# ---------------------------
with tab12:
    st.header("NBB -0.8~-1.0 Lists")
    st.caption(
        "Creates two lists (Daily, selected Daily view range):\n"
        "• **(a)** Regression slope **> 0** and **NBB** between **-1.0 and -0.8**\n"
        "• **(b)** Regression slope **< 0** and **NBB** between **-1.0 and -0.8**\n\n"
        "NBB is derived from the existing Bollinger settings in the sidebar."
    )

    run_nbb = st.button("Run NBB -0.8~-1.0 Scan", key=f"btn_run_nbb_scan_{mode}")

    if run_nbb:
        rows_pos, rows_neg = [], []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            nbb_last, nbb_ts = daily_last_nbb_in_view(
                symbol=sym,
                daily_view_label=daily_view,
                bb_window=int(bb_win),
                bb_sigma=float(bb_mult),
                use_ema_mid=bool(bb_use_ema),
            )
            if not np.isfinite(nbb_last):
                continue

            if not (-1.0 <= float(nbb_last) <= -0.8):
                continue

            row = {
                "Symbol": sym,
                "Slope": float(m),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "NBB": float(nbb_last),
                "AsOf": ts
            }
            if float(m) > 0.0:
                rows_pos.append(row)
            elif float(m) < 0.0:
                rows_neg.append(row)

        left, right = st.columns(2)

        with left:
            st.subheader("(a) Regression > 0 and NBB -1.0 to -0.8")
            if not rows_pos:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos).sort_values(["NBB", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Regression < 0 and NBB -1.0 to -0.8")
            if not rows_neg:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg).sort_values(["NBB", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

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
    kijun_within = c1.slider("Cross must be within last N bars", 0, 60, 5, 1, key=f"kijun_within_n_{mode}")
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
        "• **Hourly (intraday):** regression_with_band on intraday close (lookback = Hourly slope lookback)"
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
                out = pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly (intraday {hour_period}) — R² > threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[False, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 17: R² < 45% Daily/Hourly
# ---------------------------
with tab17:
    st.header("R² < 45% Daily/Hourly")
    st.caption(
        "Shows symbols where the **R²** (regression fit quality) is **< threshold** using:\n"
        "• **Daily:** regression_with_band on daily close (lookback = Daily slope lookback)\n"
        "• **Hourly (intraday):** regression_with_band on intraday close (lookback = Hourly slope lookback)"
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
                out = pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader(f"Hourly (intraday {hour_period_lo}) — R² < threshold")
            if not hourly_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[True, True])
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
