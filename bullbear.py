# =========================
# bullbear.py  (UPDATED — Batch 1/3)
# Changes implemented (additions only):
# (1) Global trend line is NOT shown by default (plot disabled by default; slope still computed)
# (2) New tab "Reversals" added (content in Batch 3)
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

  /* Ribbon tabs */
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

  /* Chart container styling */
  div[data-testid="stImage"] {
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 14px;
    background: rgba(255,255,255,0.65);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    padding: 0.35rem 0.35rem 0.15rem 0.35rem;
    overflow: hidden;
  }
  div[data-testid="stImage"] img { border-radius: 12px; }

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
# Aesthetic helper (STYLE ONLY)
# ---------------------------
def style_axes(ax):
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
# Gapless intraday prices
# =========================
def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open", "High", "Low", "Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
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

# ---------------------------
# NEWS REMOVED (as in current code)
# ---------------------------
show_fx_news = False
news_window_days = 7

# Sessions still allowed
if mode == "Forex":
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
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

# =========================
# Regression & bands
# =========================
def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
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

# ---------------------------
# (1) Global trend line NOT shown by default (plot disabled by default)
#     NOTE: slope is still computed and returned; only the line is hidden by default.
# ---------------------------
SHOW_GLOBAL_TREND_LINE = False  # default: do not draw

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    if SHOW_GLOBAL_TREND_LINE:
        color = "green" if m >= 0 else "red"
        ax.plot(
            s.index, yhat,
            linestyle="--",
            linewidth=2.4,
            color=color,
            label=f"{label_prefix} ({fmt_slope(m)}/bar)"
        )
    return float(m)

# ---------------------------
# FIX FOR NameError: daily_global_slope MUST exist before Tabs
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

# =========================
# (2) New "Reversals" tab helper (used in Batch 3)
# =========================
@st.cache_data(ttl=120)
def daily_band_reversal(symbol: str,
                        daily_view_label: str,
                        slope_lb: int,
                        direction: str,
                        confirm_bars: int = 2,
                        z: float = 2.0):
    """
    direction:
      - "up_from_lower": price moved from below lower -2σ to inside/above, and is heading up
      - "down_from_upper": price moved from above upper +2σ to inside/below, and is heading down
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(6, int(slope_lb) + 2):
            return None

        yhat, upper, lower, m, r2 = regression_with_band(close_show, lookback=int(slope_lb), z=float(z))
        if yhat.empty or upper.empty or lower.empty:
            return None

        upper = _coerce_1d_series(upper).reindex(close_show.index)
        lower = _coerce_1d_series(lower).reindex(close_show.index)

        c = close_show
        confirm_bars = max(1, int(confirm_bars))

        # find last cross event
        if direction == "up_from_lower":
            cross = (c >= lower) & (c.shift(1) < lower.shift(1))
            cross = cross.fillna(False)
            if not cross.any():
                return None
            t = cross[cross].index[-1]
            loc = int(c.index.get_loc(t))
            bars_since = int((len(c) - 1) - loc)

            # heading up now
            if len(c) < confirm_bars + 1:
                return None
            d = c.diff().dropna()
            if len(d) < confirm_bars or not bool(np.all(d.iloc[-confirm_bars:] > 0)):
                return None

            return {
                "Symbol": symbol,
                "Event": "Reversal ↑ from Lower -2σ",
                "Bars Since": bars_since,
                "Event Time": t,
                "Close@Event": float(c.loc[t]) if np.isfinite(c.loc[t]) else np.nan,
                "Lower@Event": float(lower.loc[t]) if (t in lower.index and np.isfinite(lower.loc[t])) else np.nan,
                "Close (last)": float(c.iloc[-1]) if np.isfinite(c.iloc[-1]) else np.nan,
                "Regression Slope": float(m) if np.isfinite(m) else np.nan,
                "R2": float(r2) if np.isfinite(r2) else np.nan,
            }

        if direction == "down_from_upper":
            cross = (c <= upper) & (c.shift(1) > upper.shift(1))
            cross = cross.fillna(False)
            if not cross.any():
                return None
            t = cross[cross].index[-1]
            loc = int(c.index.get_loc(t))
            bars_since = int((len(c) - 1) - loc)

            # heading down now
            if len(c) < confirm_bars + 1:
                return None
            d = c.diff().dropna()
            if len(d) < confirm_bars or not bool(np.all(d.iloc[-confirm_bars:] < 0)):
                return None

            return {
                "Symbol": symbol,
                "Event": "Reversal ↓ from Upper +2σ",
                "Bars Since": bars_since,
                "Event Time": t,
                "Close@Event": float(c.loc[t]) if np.isfinite(c.loc[t]) else np.nan,
                "Upper@Event": float(upper.loc[t]) if (t in upper.index and np.isfinite(upper.loc[t])) else np.nan,
                "Close (last)": float(c.iloc[-1]) if np.isfinite(c.iloc[-1]) else np.nan,
                "Regression Slope": float(m) if np.isfinite(m) else np.nan,
                "R2": float(r2) if np.isfinite(r2) else np.nan,
            }

        return None
    except Exception:
        return None

# =========================
# Sessions (PST) — kept
# =========================
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

# =========================
# Session state init
# =========================
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

# =========================
# Shared hourly renderer (kept minimal — unchanged here)
# =========================
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

    # Simple band regression
    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    slope_sig_h = m_h

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")

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

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
    )

    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}"
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
        _apply_compact_time_ticks(ax2, real_times, n_ticks=8)

    style_axes(ax2)
    st.pyplot(fig2)

    return {"trade_instruction": instr_txt}

# =========================
# Tabs (Batch 1: tabs 1–4)
# NOTE: "Reversals" tab is added (content in Batch 3)
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Recent BUY Scanner",
    "HMA Buy",
    "NPX Cross Up (0.0 / 0.5)",
    "Slope Direction Scan",
    "Trendline Direction Lists",
    "NTD Hot List",
    "NTD NPX 0.0-0.2 Scanner",
    "Uptrend vs Downtrend",
    "R² > 45% Daily/Hourly",
    "R² < 45% Daily/Hourly",
    "R² Sign ±2σ Proximity (Daily)",
    "Reversals"
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
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"**Displayed (last run):** {disp_ticker}  •  "
                   f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

        with fib_instruction_box.container():
            st.warning(FIB_ALERT_TEXT)

        daily_instr_txt = None
        hourly_instr_txt = None

        # Daily view (kept minimal in this version)
        if chart in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)

            fig, ax = plt.subplots(figsize=(14, 5))
            fig.subplots_adjust(bottom=0.30)
            ax.set_title(f"{disp_ticker} Daily — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")

            # Global trend line computed but hidden by default via SHOW_GLOBAL_TREND_LINE
            draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
            style_axes(ax)
            st.pyplot(fig)

            daily_instr_txt = "Daily: (kept per original logic)"

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

        with trade_instruction_box.container():
            if isinstance(daily_instr_txt, str) and daily_instr_txt.strip():
                st.success(daily_instr_txt)
            if isinstance(hourly_instr_txt, str) and hourly_instr_txt.strip():
                if hourly_instr_txt.startswith("ALERT:"):
                    st.error(hourly_instr_txt)
                else:
                    st.success(hourly_instr_txt)

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")

# ---------------------------
# TAB 2: ENHANCED FORECAST (kept)
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci
        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({"Forecast": vals, "Lower": ci.iloc[:, 0], "Upper": ci.iloc[:, 1]}, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR (kept)
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

        # Global trend line computed but hidden by default via SHOW_GLOBAL_TREND_LINE
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 4: METRICS (kept minimal)
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
        st.write({"Daily slope": fmt_slope(m_d), "Daily R²": fmt_r2(r2_d)})

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            _, _, _, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({"Hourly slope": fmt_slope(m_h), "Hourly R²": fmt_r2(r2_h)})

# =========================
# END Batch 1/3
# =========================
# =========================
# bullbear.py  (UPDATED — Batch 2/3)
# Tabs 5 through 8
# =========================

# ---------------------------
# NTD / NPX / HMA helpers (additive; used by Tabs 5–8 and later tabs)
# ---------------------------
def npx_minmax(close_series_like, window: int = 60) -> pd.Series:
    c = _coerce_1d_series(close_series_like).astype(float).dropna()
    if c.empty:
        return pd.Series(dtype=float)
    w = max(2, int(window))
    rmin = c.rolling(w, min_periods=2).min()
    rmax = c.rolling(w, min_periods=2).max()
    denom = (rmax - rmin).replace(0.0, np.nan)
    npx = (c - rmin) / denom
    return npx.clip(lower=0.0, upper=1.0)

def compute_ntd(close_series_like, window: int = 60) -> pd.Series:
    """
    NTD (Normalized Trend Direction):
      - Build NPX as rolling min-max normalized price in [0,1]
      - Take slope over `window` bars and squash with tanh into [-1,1]
    """
    npx = npx_minmax(close_series_like, window=window).dropna()
    if npx.empty:
        return pd.Series(dtype=float)
    w = max(2, int(window))
    slope = (npx - npx.shift(w)) / float(w)
    ntd = np.tanh(12.0 * slope)  # squashes to roughly [-1,1]
    return ntd

def wma(series_like, period: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    n = int(max(1, period))
    w = np.arange(1, n + 1, dtype=float)
    def _wma(x):
        x = np.asarray(x, dtype=float)
        if np.all(~np.isfinite(x)):
            return np.nan
        return np.nansum(x * w) / np.nansum(w[np.isfinite(x)])
    return s.rolling(n, min_periods=n).apply(_wma, raw=True)

def hma(series_like, period: int) -> pd.Series:
    p = int(max(2, period))
    half = max(1, p // 2)
    sqrtp = max(1, int(np.sqrt(p)))
    s = _coerce_1d_series(series_like).astype(float)
    w1 = wma(s, half)
    w2 = wma(s, p)
    raw = 2.0 * w1 - w2
    return wma(raw, sqrtp)

# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Scans for symbols where the latest Daily NTD is at or below -0.75 (strong down-trend zone).")

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    scan_view = c1.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=["Historical","6M","12M","24M"].index(daily_view), key=f"tab5_view_{mode}")
    ntd_thr = c2.slider("NTD threshold", -0.99, -0.10, -0.75, 0.01, key=f"tab5_ntd_thr_{mode}")
    max_rows = c3.slider("Max rows", 10, 200, 50, 10, key=f"tab5_max_rows_{mode}")

    run_scan = st.button("Scan NTD", key=f"tab5_scan_btn_{mode}")

    if run_scan:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                s = _coerce_1d_series(fetch_hist(sym)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                if len(s) < max(10, ntd_window + 5):
                    continue
                ntd = compute_ntd(s, window=ntd_window).dropna()
                if ntd.empty:
                    continue
                last_ntd = float(ntd.iloc[-1])
                if not np.isfinite(last_ntd) or last_ntd > float(ntd_thr):
                    continue

                # add quick regression info for context
                _, _, _, m_d, r2_d = regression_with_band(s, lookback=slope_lb_daily, z=2.0)
                rows.append({
                    "Symbol": sym,
                    "NTD (last)": last_ntd,
                    "Close (last)": float(s.iloc[-1]),
                    "Regression Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                    "R²": float(r2_d) if np.isfinite(r2_d) else np.nan
                })
            except Exception:
                continue
            finally:
                prog.progress((i + 1) / max(1, len(universe)))

        prog.empty()

        if not rows:
            st.info("No matches found with current settings.")
        else:
            df_scan = pd.DataFrame(rows).sort_values("NTD (last)", ascending=True).head(int(max_rows))
            st.dataframe(df_scan, use_container_width=True)

            pick = st.selectbox("Plot a symbol:", df_scan["Symbol"].tolist(), key=f"tab5_pick_{mode}")
            if pick:
                s = _coerce_1d_series(fetch_hist(pick)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                ntd = compute_ntd(s, window=ntd_window)

                fig, ax = plt.subplots(figsize=(14, 4))
                fig.subplots_adjust(bottom=0.28)
                ax.set_title(f"{pick} — Daily Close ({scan_view})")
                ax.plot(s.index, s.values, label="Close")
                draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
                style_axes(ax)
                st.pyplot(fig)

                fig2, ax2 = plt.subplots(figsize=(14, 3))
                fig2.subplots_adjust(bottom=0.28)
                ax2.set_title(f"{pick} — NTD (window={ntd_window})")
                ax2.plot(ntd.index, ntd.values, label="NTD")
                ax2.axhline(float(ntd_thr), linestyle="--", linewidth=1.4, alpha=0.8, label=f"Threshold {ntd_thr:.2f}")
                ax2.axhline(-0.75, linestyle=":", linewidth=1.2, alpha=0.8, label="-0.75")
                ax2.axhline(0.0, linestyle="-", linewidth=1.0, alpha=0.25)
                ax2.set_ylim(-1.05, 1.05)
                ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
                style_axes(ax2)
                st.pyplot(fig2)

# ---------------------------
# TAB 6: Long-Term History
# ---------------------------
with tab6:
    st.header("Long-Term History")
    st.caption("Max-available daily history. Uses Yahoo Finance 'max' period.")

    sym = st.selectbox("Ticker:", universe, key=f"tab6_ticker_{mode}")
    years_back = st.slider("Years to display (approx)", 1, 25, int(st.session_state.get("hist_years", 10)), 1, key=f"tab6_years_{mode}")
    show_reg_band = st.checkbox("Show regression midline and ±2σ band (lookback = Daily slope lookback)", value=True, key=f"tab6_show_band_{mode}")

    try:
        smax = _coerce_1d_series(fetch_hist_max(sym)).dropna()
    except Exception:
        smax = pd.Series(dtype=float)

    if smax.empty:
        st.warning("No history available.")
    else:
        st.session_state.hist_years = int(years_back)
        end = smax.index.max()
        start = end - pd.Timedelta(days=int(years_back) * 365)
        s = smax.loc[smax.index >= start]

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.subplots_adjust(bottom=0.30)
        ax.set_title(f"{sym} — Long-Term History (~{years_back}y)")
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")

        if show_reg_band:
            yhat, up, lo, m, r2 = regression_with_band(s, lookback=slope_lb_daily, z=2.0)
            if not yhat.empty:
                ax.plot(yhat.index, yhat.values, linewidth=2.0, label=f"Slope {slope_lb_daily} bars ({fmt_slope(m)}/bar, R² {fmt_r2(r2)})")
            if not up.empty and not lo.empty:
                ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.75, label="+2σ")
                ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.75, label="-2σ")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: Recent BUY Scanner
# ---------------------------
with tab7:
    st.header("Recent BUY Scanner")
    st.caption("Heuristic: daily regression slope > 0 AND last close is within the S/R proximity threshold of support (daily rolling min).")

    c1, c2, c3 = st.columns([1.3, 1.0, 1.0])
    scan_view = c1.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=["Historical","6M","12M","24M"].index(daily_view), key=f"tab7_view_{mode}")
    max_rows = c2.slider("Max rows", 10, 200, 50, 10, key=f"tab7_max_rows_{mode}")
    require_r2 = c3.checkbox("Require R² ≥ 0.45", value=False, key=f"tab7_req_r2_{mode}")

    run_scan = st.button("Scan BUY", key=f"tab7_scan_btn_{mode}")

    if run_scan:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                s = _coerce_1d_series(fetch_hist(sym)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                if len(s) < max(20, slope_lb_daily + 5, sr_lb_daily + 5):
                    continue

                # support / resistance
                sup = s.rolling(int(sr_lb_daily), min_periods=2).min()
                if sup.empty or not np.isfinite(sup.iloc[-1]) or not np.isfinite(s.iloc[-1]):
                    continue

                last = float(s.iloc[-1])
                sup_last = float(sup.iloc[-1])

                # within proximity of support
                if last > sup_last * (1.0 + float(sr_prox_pct)):
                    continue

                # regression slope filter
                _, _, _, m_d, r2_d = regression_with_band(s, lookback=int(slope_lb_daily), z=2.0)
                if not np.isfinite(m_d) or float(m_d) <= 0:
                    continue
                if require_r2 and (not np.isfinite(r2_d) or float(r2_d) < 0.45):
                    continue

                dist_pct = (last / sup_last - 1.0) if sup_last != 0 else np.nan

                rows.append({
                    "Symbol": sym,
                    "Close (last)": last,
                    "Support": sup_last,
                    "Dist to Support": fmt_pct(dist_pct, digits=2),
                    "Regression Slope": float(m_d),
                    "R²": float(r2_d) if np.isfinite(r2_d) else np.nan
                })
            except Exception:
                continue
            finally:
                prog.progress((i + 1) / max(1, len(universe)))

        prog.empty()

        if not rows:
            st.info("No matches found with current settings.")
        else:
            df_buy = pd.DataFrame(rows)
            # sort by distance to support (closest first)
            def _parse_pct(x):
                try:
                    return float(str(x).replace("%","")) / 100.0
                except Exception:
                    return np.nan
            df_buy["_dist"] = df_buy["Dist to Support"].apply(_parse_pct)
            df_buy = df_buy.sort_values(["_dist", "R²"], ascending=[True, False]).drop(columns=["_dist"])
            df_buy = df_buy.head(int(max_rows))
            st.dataframe(df_buy, use_container_width=True)

# ---------------------------
# TAB 8: HMA Buy
# ---------------------------
with tab8:
    st.header("HMA Buy")
    st.caption("Signals when price crosses above HMA(period) on the daily chart (recent crossover).")

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    scan_view = c1.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=["Historical","6M","12M","24M"].index(daily_view), key=f"tab8_view_{mode}")
    period = c2.slider("HMA period", 5, 120, int(hma_period), 1, key=f"tab8_hma_period_{mode}")
    max_rows = c3.slider("Max rows", 10, 200, 50, 10, key=f"tab8_max_rows_{mode}")

    run_scan = st.button("Scan HMA Buys", key=f"tab8_scan_btn_{mode}")

    if run_scan:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                s = _coerce_1d_series(fetch_hist(sym)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                if len(s) < max(30, int(period) + 10):
                    continue

                h = hma(s, int(period))
                if h.empty:
                    continue

                c = s.reindex(h.index).astype(float)
                ok = c.notna() & h.notna()
                c = c[ok]; h = h[ok]
                if len(c) < 3:
                    continue

                cross_up = (c > h) & (c.shift(1) <= h.shift(1))
                if not bool(cross_up.iloc[-1]):
                    continue

                # keep context metrics
                _, _, _, m_d, r2_d = regression_with_band(s, lookback=int(slope_lb_daily), z=2.0)

                rows.append({
                    "Symbol": sym,
                    "Close (last)": float(c.iloc[-1]),
                    "HMA (last)": float(h.iloc[-1]),
                    "Regression Slope": float(m_d) if np.isfinite(m_d) else np.nan,
                    "R²": float(r2_d) if np.isfinite(r2_d) else np.nan
                })
            except Exception:
                continue
            finally:
                prog.progress((i + 1) / max(1, len(universe)))

        prog.empty()

        if not rows:
            st.info("No HMA buy crossovers found with current settings.")
        else:
            df_hma = pd.DataFrame(rows).sort_values(["R²", "Regression Slope"], ascending=[False, False]).head(int(max_rows))
            st.dataframe(df_hma, use_container_width=True)

            pick = st.selectbox("Plot a symbol:", df_hma["Symbol"].tolist(), key=f"tab8_pick_{mode}")
            if pick:
                s = _coerce_1d_series(fetch_hist(pick)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                h = hma(s, int(period))

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.subplots_adjust(bottom=0.30)
                ax.set_title(f"{pick} — Close with HMA({period}) ({scan_view})")
                ax.plot(s.index, s.values, label="Close")
                ax.plot(h.index, h.values, linewidth=2.0, label=f"HMA({period})")
                draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
                style_axes(ax)
                st.pyplot(fig)

# =========================
# END Batch 2/3
# =========================
# =========================
# bullbear.py  (UPDATED — Batch 3/3)
# Remainder of the file + NEW “Reversals” tab
# =========================

# ---------------------------
# FIX: Provide missing daily_global_slope (prevents NameError)
# ---------------------------
def _linreg_slope_r2(y: np.ndarray) -> tuple[float, float]:
    """
    Linear regression of y on x=0..n-1.
    Returns (slope_per_bar, r2). Assumes y is finite and length>=2.
    """
    n = int(len(y))
    if n < 2:
        return (np.nan, np.nan)
    x = np.arange(n, dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x0 = x - x_mean
    y0 = y - y_mean
    denom = float(np.dot(x0, x0))
    if denom == 0.0:
        return (np.nan, np.nan)
    slope = float(np.dot(x0, y0) / denom)
    intercept = y_mean - slope * x_mean
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return (slope, r2)

def rolling_slope_r2(series_like, window: int) -> tuple[pd.Series, pd.Series]:
    s = _coerce_1d_series(series_like).astype(float)
    w = int(max(2, window))
    vals = s.values
    slope_out = np.full(len(vals), np.nan, dtype=float)
    r2_out = np.full(len(vals), np.nan, dtype=float)

    for i in range(w - 1, len(vals)):
        y = vals[i - w + 1 : i + 1]
        if not np.all(np.isfinite(y)):
            continue
        m, r2 = _linreg_slope_r2(y)
        slope_out[i] = m
        r2_out[i] = r2

    return (pd.Series(slope_out, index=s.index), pd.Series(r2_out, index=s.index))

def daily_global_slope(sym, daily_view_label: str = "Historical", lookback: int | None = None):
    """
    Compatibility helper used elsewhere in the file.
    Returns: (m, r2, ts) where:
      - m = latest rolling regression slope over `lookback` bars
      - r2 = latest rolling regression R² over `lookback` bars
      - ts = rolling slope time-series
    """
    lb = int(lookback or slope_lb_daily)
    s = _coerce_1d_series(fetch_hist(sym)).dropna()
    s = _coerce_1d_series(subset_by_daily_view(s, daily_view_label)).dropna()
    if len(s) < lb + 2:
        return (np.nan, np.nan, pd.Series(dtype=float))

    slope_ts, r2_ts = rolling_slope_r2(s, lb)
    slope_clean = slope_ts.dropna()
    r2_clean = r2_ts.dropna()
    m = float(slope_clean.iloc[-1]) if not slope_clean.empty else np.nan
    r2 = float(r2_clean.iloc[-1]) if not r2_clean.empty else np.nan
    return (m, r2, slope_ts)

# ---------------------------
# NEW TAB: Reversals
# ---------------------------
def _recent_reversal_from_lower(close: pd.Series, lower: pd.Series, days: int = 10) -> tuple[bool, int | None]:
    """
    True if: within last `days` bars price touched/breached lower band AND now it's moving up off it.
    Returns (match, days_since_touch).
    """
    df = pd.DataFrame({"c": close.astype(float), "lo": lower.astype(float)}).dropna()
    if len(df) < 3:
        return (False, None)

    d = int(max(2, days))
    tail = df.iloc[-(d + 2) :] if len(df) > (d + 2) else df.copy()

    touched = tail["c"] <= tail["lo"]
    if not bool(touched.any()):
        return (False, None)

    # "Heading up": last close > prior close and last close above lower band
    c_last = float(tail["c"].iloc[-1])
    c_prev = float(tail["c"].iloc[-2])
    lo_last = float(tail["lo"].iloc[-1])

    reversed_now = (c_last > lo_last) and (c_last > c_prev)

    # days since last touch
    idxs = np.where(touched.values)[0]
    days_since = int(len(tail) - 1 - idxs[-1]) if len(idxs) else None

    return (bool(reversed_now), days_since)

def _recent_reversal_from_upper(close: pd.Series, upper: pd.Series, days: int = 10) -> tuple[bool, int | None]:
    """
    True if: within last `days` bars price touched/breached upper band AND now it's moving down off it.
    Returns (match, days_since_touch).
    """
    df = pd.DataFrame({"c": close.astype(float), "up": upper.astype(float)}).dropna()
    if len(df) < 3:
        return (False, None)

    d = int(max(2, days))
    tail = df.iloc[-(d + 2) :] if len(df) > (d + 2) else df.copy()

    touched = tail["c"] >= tail["up"]
    if not bool(touched.any()):
        return (False, None)

    # "Heading down": last close < prior close and last close below upper band
    c_last = float(tail["c"].iloc[-1])
    c_prev = float(tail["c"].iloc[-2])
    up_last = float(tail["up"].iloc[-1])

    reversed_now = (c_last < up_last) and (c_last < c_prev)

    idxs = np.where(touched.values)[0]
    days_since = int(len(tail) - 1 - idxs[-1]) if len(idxs) else None

    return (bool(reversed_now), days_since)

# NOTE: This assumes Batch 1 added the new tab and assigned it to `tab9`
#       with label "Reversals" (keeping all other tabs unchanged).
with tab9:
    st.header("Reversals")
    st.caption(
        "Finds symbols that recently reversed from the ±2σ regression bands.\n\n"
        "• Uptrend list: Regression slope > 0 AND price reversed up off the lower 2σ band.\n"
        "• Downtrend list: Regression slope < 0 AND price reversed down off the upper 2σ band."
    )

    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])
    scan_view = c1.selectbox(
        "Daily view range:",
        ["Historical", "6M", "12M", "24M"],
        index=["Historical", "6M", "12M", "24M"].index(daily_view),
        key=f"rev_view_{mode}",
    )
    reg_lb = c2.slider("Regression lookback (bars)", 20, 400, int(slope_lb_daily), 5, key=f"rev_lb_{mode}")
    rev_days = c3.slider("Reversal window (days)", 2, 30, 10, 1, key=f"rev_days_{mode}")
    max_rows = c4.slider("Max rows", 10, 200, 50, 10, key=f"rev_max_{mode}")

    run_scan = st.button("Scan Reversals", key=f"rev_scan_btn_{mode}")

    if run_scan:
        up_rows = []
        down_rows = []
        prog = st.progress(0)

        for i, sym in enumerate(universe):
            try:
                s = _coerce_1d_series(fetch_hist(sym)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()
                if len(s) < int(reg_lb) + 5:
                    continue

                yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(reg_lb), z=2.0)
                if yhat.empty or up.empty or lo.empty or (not np.isfinite(m)):
                    continue

                # align series
                close = s.reindex(yhat.index).astype(float)
                up2 = up.reindex(yhat.index).astype(float)
                lo2 = lo.reindex(yhat.index).astype(float)

                if float(m) > 0:
                    ok, since = _recent_reversal_from_lower(close, lo2, days=int(rev_days))
                    if ok:
                        up_rows.append({
                            "Symbol": sym,
                            "Close (last)": float(close.dropna().iloc[-1]) if not close.dropna().empty else np.nan,
                            "Lower 2σ (last)": float(lo2.dropna().iloc[-1]) if not lo2.dropna().empty else np.nan,
                            "Days since touch": since,
                            "Regression Slope": float(m),
                            "R²": float(r2) if np.isfinite(r2) else np.nan,
                        })
                elif float(m) < 0:
                    ok, since = _recent_reversal_from_upper(close, up2, days=int(rev_days))
                    if ok:
                        down_rows.append({
                            "Symbol": sym,
                            "Close (last)": float(close.dropna().iloc[-1]) if not close.dropna().empty else np.nan,
                            "Upper 2σ (last)": float(up2.dropna().iloc[-1]) if not up2.dropna().empty else np.nan,
                            "Days since touch": since,
                            "Regression Slope": float(m),
                            "R²": float(r2) if np.isfinite(r2) else np.nan,
                        })

            except Exception:
                continue
            finally:
                prog.progress((i + 1) / max(1, len(universe)))

        prog.empty()

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Uptrend reversals (slope > 0): from lower 2σ heading up")
            if not up_rows:
                st.info("No matches found.")
                df_up = pd.DataFrame(columns=["Symbol"])
            else:
                df_up = pd.DataFrame(up_rows).sort_values(
                    ["Days since touch", "R²"], ascending=[True, False]
                ).head(int(max_rows))
                st.dataframe(df_up, use_container_width=True)

        with colB:
            st.subheader("Downtrend reversals (slope < 0): from upper 2σ heading down")
            if not down_rows:
                st.info("No matches found.")
                df_dn = pd.DataFrame(columns=["Symbol"])
            else:
                df_dn = pd.DataFrame(down_rows).sort_values(
                    ["Days since touch", "R²"], ascending=[True, False]
                ).head(int(max_rows))
                st.dataframe(df_dn, use_container_width=True)

        # Optional plot chooser
        all_syms = []
        if "Symbol" in df_up.columns and not df_up.empty:
            all_syms += df_up["Symbol"].tolist()
        if "Symbol" in df_dn.columns and not df_dn.empty:
            all_syms += df_dn["Symbol"].tolist()

        if all_syms:
            st.markdown("---")
            pick = st.selectbox("Plot a reversal candidate:", all_syms, key=f"rev_plot_pick_{mode}")
            if pick:
                s = _coerce_1d_series(fetch_hist(pick)).dropna()
                s = _coerce_1d_series(subset_by_daily_view(s, scan_view)).dropna()

                yhat, up, lo, m, r2 = regression_with_band(s, lookback=int(reg_lb), z=2.0)
                close = s.reindex(yhat.index).astype(float)

                fig, ax = plt.subplots(figsize=(14, 5))
                fig.subplots_adjust(bottom=0.30)
                ax.set_title(f"{pick} — Reversal context ({scan_view}) | slope={fmt_slope(m)}/bar, R²={fmt_r2(r2)}")
                ax.plot(close.index, close.values, label="Close")
                if not yhat.empty:
                    ax.plot(yhat.index, yhat.values, linewidth=2.0, label="Regression midline")
                if not up.empty and not lo.empty:
                    ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.75, label="+2σ")
                    ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.75, label="-2σ")

                # IMPORTANT: no global trend line is added here (keeps it off by default)
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
                style_axes(ax)
                st.pyplot(fig)

# =========================
# END Batch 3/3
# =========================
