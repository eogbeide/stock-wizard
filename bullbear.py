# bullbear.py — Complete updated Streamlit app (Batch 1/3)
# -------------------------------------------------------
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

# =========================
# Styling / Page
# =========================
def _apply_mpl_theme():
    """Clean matplotlib look (STYLE ONLY)."""
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

st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ribbon tabs + nicer chart container
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

  /* Chart container */
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
    div[data-baseweb="tab-list"] { gap: 0.30rem !important; }
  }
</style>
""", unsafe_allow_html=True)

# =========================
# Time / Auto-refresh (PST)
# =========================
PACIFIC = pytz.timezone("US/Pacific")
REFRESH_INTERVAL = 120

def _rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        _rerun()

auto_refresh()
elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# =========================
# Shared helpers
# =========================
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
    BUY only when Global slope and Local slope agree UP.
    SELL only when Global slope and Local slope agree DOWN.
    Otherwise show ALERT.
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    if global_trend_slope is None:
        try:
            uptrend = float(trend_slope) >= 0.0
        except Exception:
            uptrend = False
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
        return ALERT_TEXT

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return ALERT_TEXT

    sg = float(np.sign(g))
    sl = float(np.sign(l))
    if sg == 0.0 or sl == 0.0:
        return ALERT_TEXT

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

    return ALERT_TEXT

# =========================
# Gapless intraday OHLC
# =========================
def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open", "High", "Low", "Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
    """Remove price gaps across large time-gaps by applying cumulative offset."""
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

# =========================
# Mode switching
# =========================
def _reset_run_state_for_mode_switch():
    st.session_state.run_all = False
    st.session_state.ticker = None
    for k in [
        "df_hist","df_ohlc","fc_idx","fc_vals","fc_ci",
        "intraday","chart","hour_range","mode_at_run"
    ]:
        st.session_state.pop(k, None)

if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"

st.title("📊 Dashboard & Forecasts")
mcol1, mcol2 = st.columns(2)
if mcol1.button("🌐 Forex", use_container_width=True, key="btn_mode_forex"):
    if st.session_state.asset_mode != "Forex":
        st.session_state.asset_mode = "Forex"
        _reset_run_state_for_mode_switch()
        _rerun()

if mcol2.button("📈 Stocks", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        _rerun()

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# =========================
# Sidebar config
# =========================
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

if st.sidebar.button("🧹 Clear cache (data + run state)", use_container_width=True, key="btn_clear_cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _reset_run_state_for_mode_switch()
    _rerun()

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
hma_conf = st.sidebar.slider("Crossover confidence (label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

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

# Universe
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

# =========================
# Data fetchers
# =========================
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].asfreq("D").ffill()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
        s.index = s.index.tz_localize(PACIFIC)
    elif isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max", progress=False)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].dropna().asfreq("D").ffill()
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is None:
        s.index = s.index.tz_localize(PACIFIC)
    elif isinstance(s.index, pd.DatetimeIndex):
        s.index = s.index.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    need = ["Open","High","Low","Close"]
    have = [c for c in need if c in df.columns]
    if len(have) < 4:
        return pd.DataFrame()
    out = df[need].dropna()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None:
        out.index = out.index.tz_localize(PACIFIC)
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_convert(PACIFIC)
    return out

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    except Exception:
        pass
    try:
        df = df.tz_convert(PACIFIC)
    except Exception:
        pass
    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize() + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower","upper"])
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    ci = fc.conf_int()
    ci.index = idx
    pm = fc.predicted_mean
    pm.index = idx
    return idx, pm, ci
# bullbear.py — Complete updated Streamlit app (Batch 2/3)
# -------------------------------------------------------

# =========================
# Indicators / Math helpers
# =========================
def sma(s: pd.Series, n: int) -> pd.Series:
    s = _coerce_1d_series(s)
    return s.rolling(n, min_periods=max(2, n//3)).mean()

def ema(s: pd.Series, n: int) -> pd.Series:
    s = _coerce_1d_series(s)
    return s.ewm(span=n, adjust=False, min_periods=max(2, n//3)).mean()

def roc_pct(s: pd.Series, n: int) -> pd.Series:
    s = _coerce_1d_series(s)
    return (s / s.shift(n) - 1.0) * 100.0

def rolling_linreg_slope_r2(y: pd.Series, lookback: int = 90):
    """
    Returns last slope and R^2 from OLS over the most recent 'lookback' bars.
    Uses x = 0..(n-1).
    """
    y = _coerce_1d_series(y).dropna()
    if len(y) < max(5, int(lookback)):
        return np.nan, np.nan
    yv = y.iloc[-lookback:].values.astype(float)
    x = np.arange(len(yv), dtype=float)
    # OLS closed-form
    x_mean = x.mean()
    y_mean = yv.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx <= 0:
        return np.nan, np.nan
    m = np.sum((x - x_mean) * (yv - y_mean)) / ss_xx
    b = y_mean - m * x_mean
    y_hat = m * x + b
    ss_res = np.sum((yv - y_hat) ** 2)
    ss_tot = np.sum((yv - y_mean) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(m), float(r2)

def bollinger(s: pd.Series, win: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(s)
    mid = ema(s, win) if use_ema else sma(s, win)
    std = s.rolling(win, min_periods=max(2, win//3)).std()
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower

def wma(s: pd.Series, n: int) -> pd.Series:
    s = _coerce_1d_series(s)
    if n <= 1:
        return s
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=max(2, n//3)).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

def hma(s: pd.Series, n: int) -> pd.Series:
    s = _coerce_1d_series(s)
    n = int(max(2, n))
    half = max(1, n // 2)
    sqrt_n = max(1, int(np.sqrt(n)))
    return wma(2 * wma(s, half) - wma(s, n), sqrt_n)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    s = _coerce_1d_series(series)
    m = ema(s, fast) - ema(s, slow)
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def true_range(df: pd.DataFrame) -> pd.Series:
    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 10) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0/n, adjust=False, min_periods=max(2, n//3)).mean()

def supertrend(df: pd.DataFrame, atr_period_: int = 10, atr_mult_: float = 3.0):
    """
    Returns: supertrend line (float), direction (+1 up, -1 down)
    """
    if df is None or df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    h = pd.to_numeric(df["High"], errors="coerce")
    l = pd.to_numeric(df["Low"], errors="coerce")
    c = pd.to_numeric(df["Close"], errors="coerce")

    hl2 = (h + l) / 2.0
    _atr = atr(df, atr_period_)
    upperband = hl2 + atr_mult_ * _atr
    lowerband = hl2 - atr_mult_ * _atr

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            st_line.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = 1.0
            continue

        prev_st = st_line.iloc[i-1]
        prev_dir = direction.iloc[i-1]

        # final bands
        ub = upperband.iloc[i]
        lb = lowerband.iloc[i]
        pub = upperband.iloc[i-1]
        plb = lowerband.iloc[i-1]

        if ub > pub and c.iloc[i-1] <= pub:
            ub = pub
        if lb < plb and c.iloc[i-1] >= plb:
            lb = plb

        # direction
        dir_now = prev_dir
        st_now = prev_st

        if prev_st == pub:
            if c.iloc[i] > ub:
                dir_now = 1.0
            else:
                dir_now = -1.0
        elif prev_st == plb:
            if c.iloc[i] < lb:
                dir_now = -1.0
            else:
                dir_now = 1.0

        st_now = lb if dir_now > 0 else ub
        st_line.iloc[i] = st_now
        direction.iloc[i] = dir_now

    return st_line, direction

def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Simple PSAR implementation.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    high = pd.to_numeric(df["High"], errors="coerce").values
    low = pd.to_numeric(df["Low"], errors="coerce").values
    idx = df.index

    psar = np.full(len(df), np.nan, dtype=float)
    bull = True
    af = step
    ep = high[0]
    psar[0] = low[0]

    for i in range(1, len(df)):
        prev = psar[i-1]

        if bull:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], low[i-1], low[i] if i >= 2 else low[i-1])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(max_step, af + step)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], high[i-1], high[i] if i >= 2 else high[i-1])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(max_step, af + step)

    return pd.Series(psar, index=idx)

def normalized_trend_direction(series: pd.Series, window: int = 60):
    """
    NTD: normalize price to 0..1 over rolling window,
    then compute slope over the same window.
    Returns: ntd (0..1), ntd_slope
    """
    s = _coerce_1d_series(series)
    rmin = s.rolling(window, min_periods=max(5, window//3)).min()
    rmax = s.rolling(window, min_periods=max(5, window//3)).max()
    denom = (rmax - rmin).replace(0, np.nan)
    ntd = (s - rmin) / denom
    ntd_slope = ntd.diff().rolling(window, min_periods=max(5, window//3)).mean()
    return ntd, ntd_slope

def ichimoku_kijun(series: pd.Series, base: int = 26):
    s = _coerce_1d_series(series)
    hh = s.rolling(base, min_periods=max(5, base//3)).max()
    ll = s.rolling(base, min_periods=max(5, base//3)).min()
    kijun = (hh + ll) / 2.0
    return kijun

def support_resistance(series: pd.Series, lookback: int = 60):
    s = _coerce_1d_series(series).dropna()
    if len(s) < max(10, lookback):
        return np.nan, np.nan
    w = s.iloc[-lookback:]
    return float(w.min()), float(w.max())

def fib_levels(low: float, high: float):
    if not (np.isfinite(low) and np.isfinite(high)):
        return {}
    if high == low:
        return {}
    a, b = float(low), float(high)
    lo, hi = min(a, b), max(a, b)
    rng = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * rng,
        "38.2%": hi - 0.382 * rng,
        "50%": hi - 0.5 * rng,
        "61.8%": hi - 0.618 * rng,
        "78.6%": hi - 0.786 * rng,
        "100%": lo,
    }

def slope_reversal_probability(series: pd.Series, lookback_hist: int = 240, horizon: int = 15, slope_win: int = 30):
    """
    Experimental: Estimate P(reversal within horizon) given current slope sign.
    - Compute slope sign over rolling window slope_win.
    - Define reversal event if slope sign flips at least once in next 'horizon' bars.
    """
    s = _coerce_1d_series(series).dropna()
    if len(s) < (lookback_hist + horizon + slope_win + 5):
        return np.nan

    s = s.iloc[-(lookback_hist + horizon + slope_win + 5):].copy()
    # slope sign series
    slopes = s.diff().rolling(slope_win, min_periods=max(5, slope_win//3)).mean()
    sign = np.sign(slopes).replace(0, np.nan).ffill()

    # build samples
    events = []
    conds = []
    for i in range(len(s) - horizon - 1):
        cond = sign.iloc[i]
        if not np.isfinite(cond):
            continue
        future = sign.iloc[i+1:i+1+horizon]
        ev = 1.0 if (future.dropna().ne(cond).any()) else 0.0
        events.append(ev)
        conds.append(cond)

    if len(events) < 30:
        return np.nan

    events = np.array(events, dtype=float)
    conds = np.array(conds, dtype=float)

    # conditional probability for current sign
    curr_sign = float(conds[-1])
    mask = conds == curr_sign
    if mask.sum() < 10:
        return float(events.mean())
    return float(events[mask].mean())

# =========================
# Scanner logic
# =========================
def compute_asset_metrics(ticker: str):
    """
    Fetch daily close + intraday OHLC and compute metrics used by scanner and tabs.
    Returns dict.
    """
    # Daily close
    close_daily = fetch_hist(ticker)
    if close_daily is None or close_daily.empty:
        return {"ticker": ticker, "ok": False}

    # Daily OHLC for BBANDS/PSAR overlay (daily chart)
    ohlc_daily = fetch_hist_ohlc(ticker)

    # Intraday (5m)
    intraday = fetch_intraday(ticker, "1d") if ticker else pd.DataFrame()

    # Slope/R2 (local daily)
    slope_d, r2_d = rolling_linreg_slope_r2(close_daily, lookback=slope_lb_daily)

    # "Global" slope/R2 (max history)
    close_global = fetch_hist_max(ticker)
    slope_g, r2_g = rolling_linreg_slope_r2(close_global, lookback=max(180, slope_lb_daily * 2))

    # Support / resistance
    s_d, r_d = support_resistance(close_daily, lookback=sr_lb_daily)

    # Trend label (daily)
    trend = "UP" if np.isfinite(slope_d) and slope_d > 0 else ("DOWN" if np.isfinite(slope_d) and slope_d < 0 else "FLAT")

    # Trend alignment (GLOBAL vs LOCAL)  ✅ updated logic used by scanner column
    if np.isfinite(slope_g) and np.isfinite(slope_d):
        sg = np.sign(slope_g)
        sl = np.sign(slope_d)
        if sg != 0 and sl != 0 and sg == sl:
            trend_slope_align = "ALIGNED"
        else:
            trend_slope_align = "MISALIGNED"
    else:
        trend_slope_align = "UNKNOWN"

    # Reversal probability (daily)
    rev_prob = slope_reversal_probability(close_daily, lookback_hist=rev_hist_lb, horizon=rev_horizon, slope_win=max(10, slope_lb_daily//3))

    # Close (last)
    last_close = _safe_last_float(close_daily)

    # Daily BB
    bb_mid, bb_u, bb_l = (pd.Series(dtype=float),)*3
    if len(ohlc_daily.index) and "Close" in ohlc_daily.columns:
        bb_mid, bb_u, bb_l = bollinger(ohlc_daily["Close"], win=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # Hourly series from intraday (resample)
    hourly_close = pd.Series(dtype=float)
    hourly_ohlc = pd.DataFrame()
    if intraday is not None and not intraday.empty and {"Open","High","Low","Close"}.issubset(intraday.columns):
        hourly_ohlc = intraday[["Open","High","Low","Close"]].resample("1H").agg({
            "Open":"first","High":"max","Low":"min","Close":"last"
        }).dropna()
        hourly_close = hourly_ohlc["Close"]

    slope_h, r2_h = rolling_linreg_slope_r2(hourly_close, lookback=slope_lb_hourly) if len(hourly_close) else (np.nan, np.nan)
    s_h, r_h = support_resistance(hourly_close, lookback=sr_lb_hourly) if len(hourly_close) else (np.nan, np.nan)

    # Hourly indicators
    st_line, st_dir = supertrend(hourly_ohlc, atr_period_=atr_period, atr_mult_=atr_mult) if len(hourly_ohlc) else (pd.Series(dtype=float), pd.Series(dtype=float))
    psar = parabolic_sar(hourly_ohlc, step=psar_step, max_step=psar_max) if (show_psar and len(hourly_ohlc)) else pd.Series(dtype=float)

    # NTD
    ntd_d, ntd_slope_d = normalized_trend_direction(close_daily, window=ntd_window) if show_ntd else (pd.Series(dtype=float), pd.Series(dtype=float))
    ntd_h, ntd_slope_h = normalized_trend_direction(hourly_close, window=max(10, ntd_window//2)) if (show_ntd and len(hourly_close)) else (pd.Series(dtype=float), pd.Series(dtype=float))

    # HMA
    hma_d = hma(close_daily, hma_period) if show_hma else pd.Series(dtype=float)
    hma_h = hma(hourly_close, hma_period) if (show_hma and len(hourly_close)) else pd.Series(dtype=float)

    # Kijun (normalized ichimoku on price)
    kijun_d = ichimoku_kijun(close_daily, base=ichi_base) if show_ichi else pd.Series(dtype=float)
    kijun_h = ichimoku_kijun(hourly_close, base=max(10, ichi_base//2)) if (show_ichi and len(hourly_close)) else pd.Series(dtype=float)

    # MACD daily (optional)
    macd_line, macd_sig, macd_hist = (pd.Series(dtype=float),)*3
    if show_macd:
        macd_line, macd_sig, macd_hist = macd(close_daily)

    # Fib levels from last SR window (daily)
    fibs = fib_levels(s_d, r_d) if show_fibs and np.isfinite(s_d) and np.isfinite(r_d) else {}

    # Proximity signals (daily) near S/R
    near_support = False
    near_resist = False
    if np.isfinite(last_close) and np.isfinite(s_d) and np.isfinite(r_d) and r_d != s_d:
        near_support = (last_close - s_d) / max(1e-12, (r_d - s_d)) <= sr_prox_pct
        near_resist  = (r_d - last_close) / max(1e-12, (r_d - s_d)) <= sr_prox_pct

    return {
        "ticker": ticker,
        "ok": True,
        "close_daily": close_daily,
        "ohlc_daily": ohlc_daily,
        "intraday": intraday,
        "hourly_ohlc": hourly_ohlc,
        "hourly_close": hourly_close,
        "last_close": last_close,

        "trend": trend,
        "slope_d": slope_d,
        "r2_d": r2_d,
        "slope_g": slope_g,
        "r2_g": r2_g,
        "trend_slope_align": trend_slope_align,
        "rev_prob": rev_prob,

        "support_d": s_d,
        "resist_d": r_d,
        "support_h": s_h,
        "resist_h": r_h,

        "bb_mid": bb_mid,
        "bb_u": bb_u,
        "bb_l": bb_l,

        "st_line": st_line,
        "st_dir": st_dir,
        "psar": psar,

        "ntd_d": ntd_d,
        "ntd_slope_d": ntd_slope_d,
        "ntd_h": ntd_h,
        "ntd_slope_h": ntd_slope_h,

        "hma_d": hma_d,
        "hma_h": hma_h,

        "kijun_d": kijun_d,
        "kijun_h": kijun_h,

        "macd_line": macd_line,
        "macd_sig": macd_sig,
        "macd_hist": macd_hist,

        "fibs": fibs,
        "near_support": near_support,
        "near_resist": near_resist,
    }

def build_scanner_table(tickers):
    rows = []
    for t in tickers:
        m = compute_asset_metrics(t)
        if not m.get("ok"):
            continue

        # ✅ UPDATED: Trend and Slope Align column now reflects the same gating logic
        # used by the trade instruction (Global slope + Local slope).
        slope_g = m.get("slope_g", np.nan)
        slope_d = m.get("slope_d", np.nan)
        if np.isfinite(slope_g) and np.isfinite(slope_d):
            sg = np.sign(slope_g)
            sl = np.sign(slope_d)
            if sg > 0 and sl > 0:
                tsa = "BUY (Global+Local UP)"
            elif sg < 0 and sl < 0:
                tsa = "SELL (Global+Local DOWN)"
            else:
                tsa = "ALERT (Mismatch)"
        else:
            tsa = "ALERT (Unknown)"

        rows.append({
            "Ticker": m["ticker"],
            "Trend (Daily)": m["trend"],
            "Slope (Daily)": m["slope_d"],
            "R² (Daily)": m["r2_d"],
            "Slope (Global)": m["slope_g"],
            "R² (Global)": m["r2_g"],
            "Trend and Slope Align": tsa,  # ✅ updated output
            "Reversal Prob (Daily)": m["rev_prob"],
            "Close": m["last_close"],
            "Daily Support": m["support_d"],
            "Daily Resistance": m["resist_d"],
            "Near Support": bool(m["near_support"]),
            "Near Resistance": bool(m["near_resist"]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Trend and Slope Align", "Ticker"], ascending=[True, True]).reset_index(drop=True)
    return df

# =========================
# UI: run button + state
# =========================
st.sidebar.markdown("---")
ticker = st.sidebar.selectbox("Select asset:", universe, index=0, key="sb_ticker_select")

run_all = st.sidebar.button("▶ Run / Refresh selected", use_container_width=True, key="btn_run_sel")

if "run_all" not in st.session_state:
    st.session_state.run_all = False
if "ticker" not in st.session_state:
    st.session_state.ticker = None

if run_all:
    st.session_state.run_all = True
    st.session_state.ticker = ticker
    st.session_state.mode_at_run = mode

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "Scanner",
    "Daily",
    "Hourly",
    "Intraday",
    "Forecast",
    "NTD",
    "Notes",
    "Regression Reversal",  # ✅ NEW TAB (implemented in Batch 3/3)
])

# =========================
# Tab: Scanner
# =========================
with tabs[0]:
    st.subheader("Scanner")
    st.caption("Signals are **gated** by Global slope vs Local slope alignment.")
    scan_df = build_scanner_table(universe)

    if scan_df.empty:
        st.info("No data returned (try again / change ticker / clear cache).")
    else:
        show_cols = [
            "Ticker",
            "Trend and Slope Align",
            "Trend (Daily)",
            "Slope (Daily)",
            "R² (Daily)",
            "Slope (Global)",
            "R² (Global)",
            "Reversal Prob (Daily)",
            "Close",
            "Daily Support",
            "Daily Resistance",
            "Near Support",
            "Near Resistance",
        ]
        st.dataframe(
            scan_df[show_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**How to read ‘Trend and Slope Align’**")
        st.markdown(
            "- **BUY (Global+Local UP)**: Both slopes are positive.\n"
            "- **SELL (Global+Local DOWN)**: Both slopes are negative.\n"
            "- **ALERT**: Slopes disagree or are unavailable."
        )

# =========================
# Prepare selected metrics
# =========================
selected = st.session_state.ticker if st.session_state.run_all else ticker
metrics = compute_asset_metrics(selected)

if not metrics.get("ok"):
    st.error("Could not fetch data for the selected symbol.")
    st.stop()

close_daily = metrics["close_daily"]
ohlc_daily = metrics["ohlc_daily"]
intraday = metrics["intraday"]
hourly_ohlc = metrics["hourly_ohlc"]
hourly_close = metrics["hourly_close"]

# Trade instruction gating (Global + Local)
trade_text_daily = format_trade_instruction(
    trend_slope=metrics["slope_d"],
    buy_val=metrics.get("support_d", np.nan),
    sell_val=metrics.get("resist_d", np.nan),
    close_val=metrics.get("last_close", np.nan),
    symbol=metrics["ticker"],
    global_trend_slope=metrics.get("slope_g", np.nan),
)

# =========================
# Tab: Daily
# =========================
with tabs[1]:
    st.subheader(f"Daily • {metrics['ticker']}")
    st.write(trade_text_daily)

    # Slice range for display
    disp_daily = subset_by_daily_view(ohlc_daily if not ohlc_daily.empty else close_daily.to_frame("Close"), daily_view)
    if isinstance(disp_daily, pd.Series):
        disp_close = disp_daily
    else:
        disp_close = disp_daily["Close"] if "Close" in disp_daily.columns else _coerce_1d_series(disp_daily)

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.plot(disp_close.values, label="Close")

    # Bollinger
    if show_bbands and not ohlc_daily.empty and "Close" in ohlc_daily.columns:
        bb_mid, bb_u, bb_l = bollinger(ohlc_daily["Close"], win=bb_win, mult=bb_mult, use_ema=bb_use_ema)
        bb_mid = subset_by_daily_view(bb_mid, daily_view)
        bb_u = subset_by_daily_view(bb_u, daily_view)
        bb_l = subset_by_daily_view(bb_l, daily_view)
        ax.plot(bb_mid.values, label="BB Mid")
        ax.plot(bb_u.values, label="BB Upper")
        ax.plot(bb_l.values, label="BB Lower")

    # Ichimoku Kijun
    if show_ichi:
        kij = subset_by_daily_view(metrics["kijun_d"], daily_view)
        if len(kij.dropna()):
            ax.plot(kij.values, label="Kijun")

    # Support / Resistance
    if np.isfinite(metrics["support_d"]):
        ax.axhline(metrics["support_d"], linestyle="--", linewidth=1.0, alpha=0.8, label="Support")
    if np.isfinite(metrics["resist_d"]):
        ax.axhline(metrics["resist_d"], linestyle="--", linewidth=1.0, alpha=0.8, label="Resistance")

    # Fibonacci
    if show_fibs and metrics["fibs"]:
        for k, v in metrics["fibs"].items():
            ax.axhline(v, linestyle=":", linewidth=0.9, alpha=0.7)
            label_on_left(ax, v, f"Fib {k}")

        st.warning(FIB_ALERT_TEXT)

    ax.set_title(f"Daily Chart • slope={fmt_slope(metrics['slope_d'])} • R²={fmt_r2(metrics['r2_d'])} • Global slope={fmt_slope(metrics['slope_g'])}")
    style_axes(ax)
    ax.legend(loc="upper left", ncol=3)
    st.pyplot(fig, use_container_width=True)

    # Optional MACD
    if show_macd and len(metrics["macd_line"]):
        fig2, ax2 = plt.subplots(figsize=(11.5, 3.1))
        md = subset_by_daily_view(metrics["macd_line"], daily_view)
        ms = subset_by_daily_view(metrics["macd_sig"], daily_view)
        mh = subset_by_daily_view(metrics["macd_hist"], daily_view)
        ax2.plot(md.values, label="MACD")
        ax2.plot(ms.values, label="Signal")
        ax2.bar(np.arange(len(mh)), mh.values, alpha=0.3, label="Hist")
        ax2.axhline(0, linewidth=0.9, alpha=0.5)
        style_axes(ax2)
        ax2.set_title("MACD (Daily)")
        ax2.legend(loc="upper left", ncol=3)
        st.pyplot(fig2, use_container_width=True)

# =========================
# Tab: Hourly
# =========================
with tabs[2]:
    st.subheader(f"Hourly • {metrics['ticker']}")
    if hourly_ohlc is None or hourly_ohlc.empty:
        st.info("No intraday data to build hourly bars (market closed or symbol unsupported).")
    else:
        st.write(
            format_trade_instruction(
                trend_slope=metrics["slope_h"],
                buy_val=metrics.get("support_h", np.nan),
                sell_val=metrics.get("resist_h", np.nan),
                close_val=_safe_last_float(hourly_close),
                symbol=metrics["ticker"],
                global_trend_slope=metrics.get("slope_g", np.nan),
            )
        )

        fig, ax = plt.subplots(figsize=(11.5, 4.4))
        ax.plot(hourly_close.values, label="Hourly Close")

        # Supertrend
        if len(metrics["st_line"]):
            ax.plot(metrics["st_line"].values, label="Supertrend")

        # PSAR
        if show_psar and len(metrics["psar"]):
            ax.scatter(np.arange(len(metrics["psar"])), metrics["psar"].values, s=10, label="PSAR", alpha=0.8)

        # Hourly support/resistance
        if np.isfinite(metrics["support_h"]):
            ax.axhline(metrics["support_h"], linestyle="--", linewidth=1.0, alpha=0.8, label="Support (H)")
        if np.isfinite(metrics["resist_h"]):
            ax.axhline(metrics["resist_h"], linestyle="--", linewidth=1.0, alpha=0.8, label="Resistance (H)")

        # Hourly Kijun
        if show_ichi and len(metrics["kijun_h"]):
            ax.plot(metrics["kijun_h"].values, label="Kijun (H)")

        ax.set_title(f"Hourly Chart • slope={fmt_slope(metrics['slope_h'])} • R²={fmt_r2(metrics['r2_h'])}")
        style_axes(ax)
        ax.legend(loc="upper left", ncol=3)
        st.pyplot(fig, use_container_width=True)

        # Optional hourly momentum
        if show_mom_hourly:
            figm, axm = plt.subplots(figsize=(11.5, 3.0))
            mom = roc_pct(hourly_close, mom_lb_hourly)
            axm.plot(mom.values, label=f"ROC% ({mom_lb_hourly})")
            axm.axhline(0, linewidth=0.9, alpha=0.5)
            style_axes(axm)
            axm.set_title("Hourly Momentum (ROC%)")
            axm.legend(loc="upper left")
            st.pyplot(figm, use_container_width=True)

# =========================
# Tab: Intraday
# =========================
with tabs[3]:
    st.subheader(f"Intraday (5m gapless) • {metrics['ticker']}")
    if intraday is None or intraday.empty:
        st.info("No intraday data returned.")
    else:
        # allow hour-range selection
        max_hours = int(max(1, (intraday.index.max() - intraday.index.min()).total_seconds() // 3600))
        hour_range = st.slider("Show last N hours", 1, max(1, min(48, max_hours)), min(12, max_hours), 1, key="sl_intraday_hours")

        end = intraday.index.max()
        start = end - pd.Timedelta(hours=hour_range)
        intr = intraday.loc[(intraday.index >= start) & (intraday.index <= end)].copy()

        # Plot as line (close) using gapless bar positions (0..n-1)
        close_i = pd.to_numeric(intr["Close"], errors="coerce").dropna()
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.plot(np.arange(len(close_i)), close_i.values, label="Close (5m)")

        # Session markers (PST)
        if show_sessions_pst:
            # London 00:00 UTC -> PST varies; approximate by marking 01:00 PST and NY 06:30 PST as visual guides
            ax.axvline(max(0, len(close_i) - int(len(close_i) * 0.65)), linestyle=":", linewidth=1.0, alpha=0.5)
            ax.axvline(max(0, len(close_i) - int(len(close_i) * 0.35)), linestyle=":", linewidth=1.0, alpha=0.5)

        _apply_compact_time_ticks(ax, close_i.index, n_ticks=9)
        style_axes(ax)
        ax.set_title("Intraday (gapless positions; real time labels)")
        ax.legend(loc="upper left")
        st.pyplot(fig, use_container_width=True)

# =========================
# Tab: Forecast (SARIMAX)
# =========================
with tabs[4]:
    st.subheader(f"30-Day Forecast (SARIMAX) • {metrics['ticker']}")
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close_daily)

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.plot(close_daily.values[-240:], label="History (last ~240D)")
    ax.plot(np.arange(len(close_daily.values[-240:]), len(close_daily.values[-240:]) + len(fc_vals)), fc_vals.values, label="Forecast")
    try:
        lower = fc_ci.iloc[:, 0].values
        upper = fc_ci.iloc[:, 1].values
        xs = np.arange(len(close_daily.values[-240:]), len(close_daily.values[-240:]) + len(fc_vals))
        ax.fill_between(xs, lower, upper, alpha=0.18, label="CI")
    except Exception:
        pass
    style_axes(ax)
    ax.set_title("Forecast (next 30 days)")
    ax.legend(loc="upper left", ncol=3)
    st.pyplot(fig, use_container_width=True)

# =========================
# Tab: NTD (Daily + Hourly)
# =========================
with tabs[5]:
    st.subheader(f"NTD • {metrics['ticker']}")
    if not show_ntd:
        st.info("Enable NTD in the sidebar to view this panel.")
    else:
        ntd_d = metrics["ntd_d"]
        ntd_slope_d = metrics["ntd_slope_d"]

        fig, ax = plt.subplots(figsize=(11.5, 3.6))
        dd = subset_by_daily_view(ntd_d, daily_view)
        ax.plot(dd.values, label="NTD (Daily)")
        ax.axhline(0.0, linewidth=0.8, alpha=0.4)
        ax.axhline(1.0, linewidth=0.8, alpha=0.4)

        # Shade by slope
        if shade_ntd and len(ntd_slope_d.dropna()):
            sd = subset_by_daily_view(ntd_slope_d, daily_view).values
            xs = np.arange(len(dd))
            ax.fill_between(xs, 0, 1, where=(sd >= 0), alpha=0.08, interpolate=True)
            ax.fill_between(xs, 0, 1, where=(sd < 0), alpha=0.08, interpolate=True)

        style_axes(ax)
        ax.set_title("Normalized Trend Direction (Daily)")
        ax.legend(loc="upper left")
        st.pyplot(fig, use_container_width=True)

        # Hourly NTD
        if len(metrics["ntd_h"]):
            fig2, ax2 = plt.subplots(figsize=(11.5, 3.4))
            ax2.plot(metrics["ntd_h"].values, label="NTD (Hourly)")
            ax2.axhline(0.0, linewidth=0.8, alpha=0.4)
            ax2.axhline(1.0, linewidth=0.8, alpha=0.4)

            # NTD channel highlight (between hourly S/R)
            if show_ntd_channel and np.isfinite(metrics["support_h"]) and np.isfinite(metrics["resist_h"]):
                # highlight bars where hourly close is between support/resistance
                hc = hourly_close.values
                mask = (hc >= metrics["support_h"]) & (hc <= metrics["resist_h"])
                ax2.fill_between(np.arange(len(hc)), 0, 1, where=mask, alpha=0.08, interpolate=True)

            style_axes(ax2)
            ax2.set_title("Normalized Trend Direction (Hourly)")
            ax2.legend(loc="upper left")
            st.pyplot(fig2, use_container_width=True)

# =========================
# Tab: Notes
# =========================
with tabs[6]:
    st.subheader("Notes")
    st.markdown(
        "- **Global slope** uses a longer history window (max available) to approximate macro bias.\n"
        "- **Local slope** uses the configured daily/hourly lookbacks.\n"
        "- Trade text is **gated**: BUY only when Global+Local slopes are both up; SELL only when both down; otherwise **ALERT**."
    )
    if "ALERT" in trade_text_daily:
        st.warning(ALERT_TEXT)
# bullbear.py — Complete updated Streamlit app (Batch 3/3)
# -------------------------------------------------------

def render_hourly_chart(symbol: str, period: str, hour_range_label: str):
    intraday = fetch_intraday(symbol, period=period)
    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        st.warning("No intraday data available.")
        return None

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None

    # Use RangeIndex for clean spacing (gapless), but keep real_times for ticks/markers
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    df = intr_plot
    hc = _coerce_1d_series(df["Close"]).ffill()
    he = hc.ewm(span=20).mean()

    # S/R
    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    # HMA + MACD
    hma_h = compute_hma(hc, period=hma_period)
    macd_h, macd_sig_h, macd_hist_h = compute_macd(hc)

    # Supertrend + PSAR (needs OHLC)
    st_df = compute_supertrend(df, atr_period=atr_period, atr_mult=atr_mult) if {"High","Low","Close"}.issubset(df.columns) else pd.DataFrame()
    st_line = st_df["ST"].reindex(hc.index) if (not st_df.empty and "ST" in st_df.columns) else pd.Series(index=hc.index, dtype=float)

    psar_df = compute_psar_from_ohlc(df, step=psar_step, max_step=psar_max) if (show_psar and {"High","Low"}.issubset(df.columns)) else pd.DataFrame()
    if not psar_df.empty:
        psar_df = psar_df.reindex(hc.index)

    # Kijun (no cloud shift)
    kijun_h = pd.Series(index=hc.index, dtype=float)
    if show_ichi and {"High","Low","Close"}.issubset(df.columns):
        _, kijun_calc, _, _, _ = ichimoku_lines(df["High"], df["Low"], df["Close"],
                                               conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
        kijun_h = _coerce_1d_series(kijun_calc).reindex(hc.index).ffill().bfill()

    # Bollinger
    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # Local regression
    yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    rev_prob_h = slope_reversal_probability(hc, m_h, hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon)

    # News + sessions (forex only)
    fx_news = pd.DataFrame()
    if (mode == "Forex") and show_fx_news:
        fx_news = fetch_yf_news(symbol, window_days=news_window_days)

    # Plot: price + (optional) NTD panel
    if show_nrsi:
        fig, (ax, axw) = plt.subplots(2, 1, sharex=True, figsize=(14, 7),
                                      gridspec_kw={"height_ratios": [3.2, 1.3]})
        plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.34)
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)
        axw = None

    ax.set_title(f"{symbol} Intraday ({hour_range_label})  |  P(slope rev≤{rev_horizon})={fmt_pct(rev_prob_h)}")
    ax.plot(hc.index, hc.values, label="Intraday")
    ax.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax, hc, label_prefix="Trend (global)")

    if show_hma and not hma_h.dropna().empty:
        ax.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
    if show_ichi and not kijun_h.dropna().empty:
        ax.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")
    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    # PSAR points
    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                       label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    # S/R lines
    res_val = float(res_h.iloc[-1]) if len(res_h.dropna()) else np.nan
    sup_val = float(sup_h.iloc[-1]) if len(sup_h.dropna()) else np.nan
    px_val  = float(hc.iloc[-1]) if len(hc.dropna()) else np.nan
    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linewidth=1.6, label="Resistance")
        ax.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linewidth=1.6, label="Support")
        label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    # Supertrend
    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    # Local regression + bands + bounce
    if not yhat_h.dropna().empty:
        ax.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} ({fmt_slope(m_h)}/bar)")
    if not up_h.dropna().empty and not lo_h.dropna().empty:
        ax.plot(up_h.index, up_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo_h.index, lo_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")
        bounce_sig = find_band_bounce_signal(hc, up_h, lo_h, m_h)
        if bounce_sig is not None:
            annotate_crossover(ax, bounce_sig["time"], bounce_sig["price"], bounce_sig["side"])

    # MACD/HMA/SR star
    macd_sig_obj = find_macd_hma_sr_signal(
        close=hc, hma=hma_h, macd=macd_h, sup=sup_h, res=res_h,
        global_trend_slope=global_m_h, prox=sr_prox_pct
    )
    macd_instr = "MACD/HMA55: n/a"
    if macd_sig_obj is not None and np.isfinite(macd_sig_obj.get("price", np.nan)):
        macd_instr = f"MACD/HMA55: {macd_sig_obj['side']} @ {fmt_price_val(macd_sig_obj['price'])}"
        annotate_macd_signal(ax, macd_sig_obj["time"], macd_sig_obj["price"], macd_sig_obj["side"])
    ax.text(0.01, 0.98, macd_instr, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8), zorder=20)

    # Forex extras: news + sessions
    session_handles = session_labels = None
    if (mode == "Forex") and show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and not real_times.empty:
        sess = compute_session_lines(real_times)
        sess_pos = {
            "ldn_open": _map_times_to_bar_positions(real_times, sess.get("ldn_open", [])),
            "ldn_close": _map_times_to_bar_positions(real_times, sess.get("ldn_close", [])),
            "ny_open": _map_times_to_bar_positions(real_times, sess.get("ny_open", [])),
            "ny_close": _map_times_to_bar_positions(real_times, sess.get("ny_close", [])),
        }
        session_handles, session_labels = draw_session_lines(ax, sess_pos)

    if (mode == "Forex") and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax, news_pos, label="News")

    # Fibonacci
    if show_fibs and not hc.dropna().empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
            ax.text(hc.index[-1], y, f" {lbl}", va="center")

    # Indicator panel (NTD + NPX + optional in-range shading)
    if axw is not None:
        axw.set_title(f"Hourly Indicator Panel — NTD/NPX (win={ntd_window})")
        ntd = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not ntd.dropna().empty:
            shade_ntd_regions(axw, ntd)
        if show_ntd and not ntd.dropna().empty:
            axw.plot(ntd.index, ntd.values, "-", linewidth=1.6, label="NTD")
        if show_ntd_channel:
            overlay_inrange_on_ntd(axw, price=hc, sup=sup_h, res=res_h)
        if show_npx_ntd and not npx.dropna().empty and show_ntd and not ntd.dropna().empty:
            overlay_npx_on_ntd(axw, npx, ntd, mark_crosses=mark_npx_cross)

        axw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        axw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        axw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        axw.set_ylim(-1.1, 1.1)

    # Legend combine
    handles, labels = [], []
    h1, l1 = ax.get_legend_handles_labels()
    handles += h1; labels += l1
    if axw is not None:
        h2, l2 = axw.get_legend_handles_labels()
        handles += h2; labels += l2
    if session_handles and session_labels:
        handles += list(session_handles)
        labels += list(session_labels)

    seen = set()
    hu, lu = [], []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        hu.append(h); lu.append(l)

    fig.legend(hu, lu, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=True, fontsize=9, framealpha=0.65, fancybox=True)

    # Real time ticks if available
    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(axw if axw is not None else ax, real_times, n_ticks=8)

    style_axes(ax)
    if axw is not None:
        style_axes(axw)
    st.pyplot(fig)

    # Optional MACD panel
    if show_macd and not macd_h.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        figm.subplots_adjust(top=0.88, bottom=0.45)
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)

    trade_txt = format_trade_instruction(
        trend_slope=m_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=symbol,
        global_trend_slope=global_m_h
    )

    return {
        "intraday": intraday,
        "global_slope": global_m_h,
        "local_slope": m_h,
        "r2": r2_h,
        "rev_prob": rev_prob_h,
        "trade_instruction": trade_txt,
        "last_price": px_val,
    }

# =========================
# Session state init
# =========================
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
    st.session_state.chart = "Both"
    st.session_state.mode_at_run = mode

# =========================
# NEW: Trend/Slope Align cross metadata helpers (for all table outputs)
# =========================
@st.cache_data(ttl=120)
def trend_slope_align_cross_meta_daily(symbol: str, daily_view_label: str, ntd_win: int = 60):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 3:
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": np.nan,
                "NTD(last)": np.nan,
            }

        ntd = _coerce_1d_series(compute_normalized_trend(close_full, window=int(ntd_win))).reindex(close_show.index)
        npx = _coerce_1d_series(compute_normalized_price(close_full, window=int(ntd_win))).reindex(close_show.index)

        ok = ntd.notna() & npx.notna()
        if ok.sum() < 2:
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": np.nan,
                "NTD(last)": np.nan,
            }

        ntd = ntd[ok]
        npx = npx[ok]

        up_mask, dn_mask = _cross_series(npx, ntd)
        cross_mask = (up_mask | dn_mask).reindex(ntd.index, fill_value=False)

        npx_last = float(npx.iloc[-1]) if np.isfinite(npx.iloc[-1]) else np.nan
        ntd_last = float(ntd.iloc[-1]) if np.isfinite(ntd.iloc[-1]) else np.nan

        if not cross_mask.any():
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": npx_last,
                "NTD(last)": ntd_last,
            }

        t_cross = cross_mask[cross_mask].index[-1]
        bars_since = int((len(ntd) - 1) - int(ntd.index.get_loc(t_cross)))
        cross_dir = "Up" if bool(up_mask.reindex(ntd.index, fill_value=False).loc[t_cross]) else "Down"

        return {
            "Bars Since Cross": int(bars_since),
            "Cross Direction": cross_dir,
            "Time Crossed": t_cross,
            "NPX(last)": npx_last,
            "NTD(last)": ntd_last,
        }
    except Exception:
        return {
            "Bars Since Cross": np.nan,
            "Cross Direction": None,
            "Time Crossed": None,
            "NPX(last)": np.nan,
            "NTD(last)": np.nan,
        }

@st.cache_data(ttl=120)
def trend_slope_align_cross_meta_hourly(symbol: str, period: str, ntd_win: int = 60):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": np.nan,
                "NTD(last)": np.nan,
            }

        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 3:
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": np.nan,
                "NTD(last)": np.nan,
            }

        ntd = _coerce_1d_series(compute_normalized_trend(hc, window=int(ntd_win)))
        npx = _coerce_1d_series(compute_normalized_price(hc, window=int(ntd_win)))

        ok = ntd.notna() & npx.notna()
        if ok.sum() < 2:
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": np.nan,
                "NTD(last)": np.nan,
            }

        ntd = ntd[ok]
        npx = npx[ok]
        up_mask, dn_mask = _cross_series(npx, ntd)
        cross_mask = (up_mask | dn_mask).reindex(ntd.index, fill_value=False)

        npx_last = float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan
        ntd_last = float(ntd.dropna().iloc[-1]) if len(ntd.dropna()) else np.nan

        if not cross_mask.any():
            return {
                "Bars Since Cross": np.nan,
                "Cross Direction": None,
                "Time Crossed": None,
                "NPX(last)": npx_last,
                "NTD(last)": ntd_last,
            }

        bar = int(cross_mask[cross_mask].index[-1])
        bars_since = int((len(hc) - 1) - bar)
        cross_dir = "Up" if bool(up_mask.reindex(ntd.index, fill_value=False).loc[bar]) else "Down"

        cross_time = bar
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            cross_time = real_times[bar]

        return {
            "Bars Since Cross": int(bars_since),
            "Cross Direction": cross_dir,
            "Time Crossed": cross_time,
            "NPX(last)": npx_last,
            "NTD(last)": ntd_last,
        }
    except Exception:
        return {
            "Bars Since Cross": np.nan,
            "Cross Direction": None,
            "Time Crossed": None,
            "NPX(last)": np.nan,
            "NTD(last)": np.nan,
        }

# =========================
# Tabs
# =========================
(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,
    tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21, tab22, tab23, tab24, tab25, tab26, tab27, tab28
) = st.tabs([
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
    "Zero cross",
    "Slope Candidates",
    "Reversal Candidates",
    "NTD -0.5 Cross",
    "Trend and Slope Align",
    "NPX 0.0 Cross Trend/PTD",
    "Magic Cross",
    "Max History Marker",
    "Daily Bet",
    "Regression Reversal",
])

period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

# =========================
# TAB 1 — Original Forecast
# =========================
with tab1:
    st.header("Original Forecast")
    st.info("Charts persist for the last **Run** ticker. Change ticker and click **Run** to refresh the display.")

    sel = st.selectbox("Ticker:", universe, key=f"orig_ticker_{mode}")
    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key=f"orig_chart_{mode}")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key=f"orig_hour_range_{mode}"
    )

    run_clicked = st.button("Run Forecast", key=f"btn_run_forecast_{mode}", use_container_width=True)
    fib_box = st.empty()
    trade_box = st.empty()

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
            "mode_at_run": mode,
        })

    if st.session_state.get("run_all") and st.session_state.get("mode_at_run") == mode and st.session_state.get("ticker"):
        disp_ticker = st.session_state.ticker
        st.caption(f"**Displayed (last run):** {disp_ticker}")

        close_series = _coerce_1d_series(st.session_state.df_hist).dropna()
        last_price = float(close_series.iloc[-1]) if len(close_series) else np.nan
        fc_vals = _coerce_1d_series(st.session_state.fc_vals)
        p_up = float(np.mean(fc_vals.to_numpy() > last_price)) if (len(fc_vals) and np.isfinite(last_price)) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        with fib_box.container():
            st.warning(FIB_ALERT_TEXT)
            st.caption("Fib+NPX: BUY when touched 100% then NPX crosses up through 0.0 recently; SELL when touched 0% then NPX crosses down through 0.0.")

        daily_instr = None
        hourly_instr = None

        if st.session_state.chart in ("Daily", "Both"):
            out_d = render_daily_chart(disp_ticker, daily_view)
            if isinstance(out_d, dict):
                daily_instr = out_d.get("trade_instruction", None)

        if st.session_state.chart in ("Hourly", "Both"):
            out_h = render_hourly_chart(disp_ticker, period_map[st.session_state.hour_range], st.session_state.hour_range)
            if isinstance(out_h, dict):
                hourly_instr = out_h.get("trade_instruction", None)

        with trade_box.container():
            if isinstance(daily_instr, str) and daily_instr.strip():
                (st.error if daily_instr.startswith("ALERT:") else st.success)(f"Daily: {daily_instr}")
            if isinstance(hourly_instr, str) and hourly_instr.strip():
                (st.error if hourly_instr.startswith("ALERT:") else st.success)(f"Hourly: {hourly_instr}")

        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(disp_ticker, window_days=news_window_days)
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.subheader("SARIMAX Forecast (30d)")
        ci = st.session_state.fc_ci.copy()
        if ci.shape[1] >= 2:
            ci.columns = ["Lower", "Upper"] + list(ci.columns[2:])
        st.dataframe(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    ci.iloc[:, 0] if ci.shape[1] else np.nan,
            "Upper":    ci.iloc[:, 1] if ci.shape[1] > 1 else np.nan
        }, index=st.session_state.fc_idx), use_container_width=True)

    else:
        st.info("Click **Run Forecast** to display charts and forecast.")

# =========================
# TAB 2 — Enhanced Forecast
# =========================
with tab2:
    st.header("Enhanced Forecast")
    st.caption("Same SARIMAX forecast, with an additional chart view for forecast fan and historical residual overview.")

    sel2 = st.selectbox("Ticker:", universe, key=f"enh_ticker_{mode}")
    run2 = st.button("Run Enhanced Forecast", key=f"btn_run_enh_{mode}", use_container_width=True)

    if run2:
        s = fetch_hist(sel2)
        s = _coerce_1d_series(s).dropna()
        idx, fc, ci = compute_sarimax_forecast(s)

        st.subheader("Forecast Fan (30d)")
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(s.index, s.values, label="History")
        ax.plot(idx, fc.values, "--", linewidth=2, label="Forecast")
        if ci is not None and isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
            ax.fill_between(idx, ci.iloc[:, 0].values, ci.iloc[:, 1].values, alpha=0.15, label="Conf. Interval")
        global_m = draw_trend_direction_line(ax, s.iloc[-min(len(s), 250):], label_prefix="Trend (recent)")
        ax.set_title(f"{sel2} — Forecast Fan  |  Recent trend slope={fmt_slope(global_m)}")
        ax.legend(loc="upper left")
        style_axes(ax)
        st.pyplot(fig)

        st.subheader("Forecast Table")
        out = pd.DataFrame({"Forecast": fc, "Lower": ci.iloc[:, 0], "Upper": ci.iloc[:, 1]}, index=idx) if (ci is not None and ci.shape[1] >= 2) else pd.DataFrame({"Forecast": fc}, index=idx)
        st.dataframe(out, use_container_width=True)

# =========================
# TAB 3 — Bull vs Bear
# =========================
with tab3:
    st.header("Bull vs Bear")
    st.caption("Bull/Bear is computed over the chosen lookback using daily closes.")

    sel3 = st.selectbox("Ticker:", universe, key=f"bb_ticker_{mode}")
    run3 = st.button("Run Bull/Bear", key=f"btn_run_bb_{mode}", use_container_width=True)

    if run3:
        s = fetch_hist(sel3).dropna()
        if s.empty:
            st.warning("No data.")
        else:
            days_map = {"1mo": 30, "3mo": 90, "6mo": 182, "1y": 365}
            look_days = days_map.get(bb_period, 182)
            s_lb = s[s.index >= (s.index.max() - pd.Timedelta(days=look_days))]
            rets = s_lb.pct_change().dropna()
            bulls = int((rets > 0).sum())
            bears = int((rets < 0).sum())
            flat = int((rets == 0).sum())
            total = len(rets)

            st.metric("Total days", total)
            c1, c2, c3c = st.columns(3)
            c1.metric("Bull days", bulls)
            c2.metric("Bear days", bears)
            c3c.metric("Flat days", flat)

            fig, ax = plt.subplots(figsize=(10, 3.6))
            ax.bar(["Bull", "Bear", "Flat"], [bulls, bears, flat])
            ax.set_title(f"{sel3} — Bull vs Bear ({bb_period})")
            style_axes(ax)
            st.pyplot(fig)

# =========================
# TAB 4 — Metrics
# =========================
with tab4:
    st.header("Metrics")
    st.caption("Quick stats based on the selected daily view + latest intraday snapshot.")

    sel4 = st.selectbox("Ticker:", universe, key=f"metrics_ticker_{mode}")
    run4 = st.button("Compute Metrics", key=f"btn_run_metrics_{mode}", use_container_width=True)

    if run4:
        close = fetch_hist(sel4).dropna()
        close_show = subset_by_daily_view(close, daily_view).dropna()
        intr = fetch_intraday(sel4, period="1d")
        last_intr = float(intr["Close"].dropna().iloc[-1]) if (intr is not None and not intr.empty and "Close" in intr) else np.nan

        if close_show.empty:
            st.warning("No data.")
        else:
            yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=min(len(close_show), slope_lb_daily))
            vol = float(close_show.pct_change().dropna().std() * np.sqrt(252)) if len(close_show) > 3 else np.nan
            ret_1m = float(close_show.iloc[-1] / close_show.iloc[max(0, len(close_show)-30)] - 1.0) if len(close_show) > 30 else np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Last (daily)", fmt_price_val(float(close_show.iloc[-1])))
            c2.metric("Last (intraday)", fmt_price_val(last_intr) if np.isfinite(last_intr) else "n/a")
            c3.metric("Ann. Vol (σ)", fmt_pct(vol, 1))

            c4, c5, c6 = st.columns(3)
            c4.metric("Slope (local)", fmt_slope(m))
            c5.metric("R²", fmt_r2(r2))
            c6.metric("1M Return", fmt_pct(ret_1m, 1))

            st.subheader("Daily view chart")
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(close_show.index, close_show.values, label="Close")
            if not yhat.dropna().empty:
                ax.plot(yhat.index, yhat.values, "--", linewidth=2, label="Regression")
            ax.set_title(f"{sel4} — Metrics Chart ({daily_view})")
            ax.legend(loc="upper left")
            style_axes(ax)
            st.pyplot(fig)

# =========================
# TAB 5 — NTD -0.75 Scanner  ✅ UPDATED (adds hourly list where global slope > 0)
# =========================
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption(
        "Daily list: NTD(last) <= -0.75 AND global trendline slope (daily view) > 0 AND regression slope > 0.\n"
        "Hourly list: NTD(last) <= -0.75 AND global trendline slope (intraday window) > 0."
    )

    c1, c2 = st.columns(2)
    max_rows = c1.slider("Max rows", 10, 200, 50, 10, key=f"ntdneg_rows_{mode}")
    hr_win = c2.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"ntdneg_hrwin_{mode}")

    run5 = st.button("Run NTD -0.75 Scan", key=f"btn_run_ntdneg_{mode}", use_container_width=True)

    if run5:
        # ---------- DAILY ----------
        daily_rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            if s.empty:
                continue

            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 10:
                continue

            # Robust scalar global slope
            g_slope = _global_slope_1d(s_show)
            if not (np.isfinite(g_slope) and (g_slope > 0.0)):
                continue

            # Regression slope over daily view (local)
            _, _, _, r_slope, r2 = regression_with_band(
                s_show,
                lookback=min(len(s_show), int(slope_lb_daily))
            )
            if not (np.isfinite(r_slope) and (float(r_slope) > 0.0)):
                continue

            ntd = compute_normalized_trend(s, window=60).reindex(s_show.index).dropna()
            if ntd.empty:
                continue

            last_ntd = float(ntd.iloc[-1])
            if last_ntd <= -0.75:
                daily_rows.append({
                    "Symbol": sym,
                    "Frame": "Daily",
                    "NTD(last)": last_ntd,
                    "Global Slope": float(g_slope),
                    "Regression Slope": float(r_slope),
                    "R2": float(r2) if np.isfinite(r2) else np.nan,
                    "Last Price": float(s_show.iloc[-1]),
                    "AsOf": s_show.index[-1],
                })

        st.subheader("Daily matches")
        if not daily_rows:
            st.write("No daily matches.")
        else:
            df_d = pd.DataFrame(daily_rows).sort_values(
                ["NTD(last)", "Regression Slope", "Global Slope"],
                ascending=[True, False, False]
            )
            st.dataframe(df_d.head(max_rows).reset_index(drop=True), use_container_width=True)

        # ---------- HOURLY (intraday 5m bars over selected window) ----------
        hourly_rows = []
        hr_period = period_map.get(hr_win, "1d")

        for sym in universe:
            df = fetch_intraday(sym, period=hr_period)
            if df is None or df.empty or "Close" not in df.columns:
                continue

            hc = _coerce_1d_series(df["Close"]).ffill().dropna()
            if len(hc) < 10:
                continue

            g_slope_h = _global_slope_1d(hc)
            if not (np.isfinite(g_slope_h) and (g_slope_h > 0.0)):
                continue

            ntd_h = compute_normalized_trend(hc, window=60).dropna()
            if ntd_h.empty:
                continue

            last_ntd_h = float(ntd_h.iloc[-1])
            if last_ntd_h <= -0.75:
                asof = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) and len(df.index) else None
                hourly_rows.append({
                    "Symbol": sym,
                    "Frame": f"Hourly({hr_win})",
                    "NTD(last)": last_ntd_h,
                    "Global Slope": float(g_slope_h),
                    "Last Price": float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan,
                    "AsOf": asof,
                })

        st.subheader(f"Hourly matches ({hr_win})")
        if not hourly_rows:
            st.write("No hourly matches.")
        else:
            df_h = pd.DataFrame(hourly_rows).sort_values(
                ["NTD(last)", "Global Slope"],
                ascending=[True, False]
            )
            st.dataframe(df_h.head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 6 — Long-Term History
# =========================
with tab6:
    st.header("Long-Term History")
    st.caption("Max history with global trendline (recent slice) and optional BB/NTD overlay.")

    sel6 = st.selectbox("Ticker:", universe, key=f"long_ticker_{mode}")
    run6 = st.button("Run Long-Term", key=f"btn_run_long_{mode}", use_container_width=True)

    if run6:
        s = fetch_hist_max(sel6).dropna()
        if s.empty:
            st.warning("No data.")
        else:
            fig, ax = plt.subplots(figsize=(14, 4.8))
            ax.plot(s.index, s.values, label="Close (max history)")
            recent = s.iloc[-min(len(s), 600):]
            gm = draw_trend_direction_line(ax, recent, label_prefix="Trend (recent)")
            ax.set_title(f"{sel6} — Max History  |  Recent slope={fmt_slope(gm)}")
            ax.legend(loc="upper left")
            style_axes(ax)
            st.pyplot(fig)

            if show_ntd:
                ntd = compute_normalized_trend(s, window=ntd_window).dropna()
                fig2, ax2 = plt.subplots(figsize=(14, 2.8))
                if shade_ntd:
                    shade_ntd_regions(ax2, ntd)
                ax2.plot(ntd.index, ntd.values, label="NTD")
                ax2.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
                ax2.set_ylim(-1.1, 1.1)
                ax2.set_title("NTD (max history)")
                ax2.legend(loc="upper left")
                style_axes(ax2)
                st.pyplot(fig2)

# =========================
# TAB 7 — Recent BUY Scanner
# =========================
with tab7:
    st.header("Recent BUY Scanner")
    st.caption("Finds the most recent band-bounce BUY signals (daily + intraday).")

    c1, c2, c3 = st.columns(3)
    max_bars = c1.slider("Max bars since signal", 0, 200, 10, 1, key=f"buy_maxbars_{mode}")
    hours = c2.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"buy_hr_{mode}")
    run7 = c3.button("Run Recent BUY Scan", key=f"btn_run_buy_scan_{mode}", use_container_width=True)

    if run7:
        rows = []
        for sym in universe:
            r = last_band_bounce_signal_daily(sym, slope_lb_daily)
            if r and r.get("Side") == "BUY" and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)
        df_daily = pd.DataFrame(rows).sort_values(["Bars Since", "DeltaPct"], ascending=[True, False]) if rows else pd.DataFrame()
        st.subheader("Daily BUY signals")
        st.dataframe(df_daily.reset_index(drop=True), use_container_width=True) if not df_daily.empty else st.write("No matches.")

        rows = []
        for sym in universe:
            r = last_band_bounce_signal_hourly(sym, period_map[hours], slope_lb_hourly)
            if r and r.get("Side") == "BUY" and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)
        df_h = pd.DataFrame(rows).sort_values(["Bars Since", "DeltaPct"], ascending=[True, False]) if rows else pd.DataFrame()
        st.subheader(f"Hourly BUY signals ({hours})")
        st.dataframe(df_h.reset_index(drop=True), use_container_width=True) if not df_h.empty else st.write("No matches.")

# =========================
# TAB 8 — NPX 0.5-Cross Scanner
# =========================
with tab8:
    st.header("NPX 0.5-Cross Scanner")
    st.caption("Lists symbols where NPX crossed up through +0.5 recently (daily view).")

    c1, c2 = st.columns(2)
    max_bars = c1.slider("Max bars since NPX +0.5 cross-up", 0, 60, 5, 1, key=f"npx05_max_{mode}")
    run8 = c2.button("Run NPX 0.5 Scan", key=f"btn_run_npx05_{mode}", use_container_width=True)

    if run8:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 3:
                continue
            npx = compute_normalized_price(s, window=ntd_window).reindex(s_show.index).dropna()
            if len(npx) < 3:
                continue
            cross_up, _ = npx_zero_cross_masks(npx, level=0.5)
            if not cross_up.any():
                continue
            t = cross_up[cross_up].index[-1]
            bars_since = int((len(npx) - 1) - int(npx.index.get_loc(t)))
            if bars_since <= int(max_bars):
                rows.append({
                    "Symbol": sym,
                    "Bars Since": bars_since,
                    "Cross Time": t,
                    "NPX@Cross": float(npx.loc[t]),
                    "NPX(last)": float(npx.iloc[-1]),
                    "Last Price": float(s_show.iloc[-1]),
                })
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values(["Bars Since", "NPX(last)"], ascending=[True, False])
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 9 — Fib NPX 0.0 Signal Scanner
# =========================
with tab9:
    st.header("Fib NPX 0.0 Signal Scanner")
    st.caption("BUY: price touched Fib 100% and NPX crossed up through 0.0 recently. SELL: touched Fib 0% and NPX crossed down through 0.0.")

    c1, c2, c3 = st.columns(3)
    max_bars = c1.slider("Max bars since signal", 0, 90, 10, 1, key=f"fibsig_max_{mode}")
    hz = c2.slider("Touch horizon (bars)", 3, 60, 15, 1, key=f"fibsig_hz_{mode}")
    run9 = c3.button("Run Fib+NPX Scan", key=f"btn_run_fibsig_{mode}", use_container_width=True)

    if run9:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 20:
                continue
            npx = compute_normalized_price(s, window=ntd_window).reindex(s_show.index)
            buy_mask, sell_mask, fibs = fib_npx_zero_cross_signal_masks(s_show, npx, horizon_bars=int(hz), proximity_pct_of_range=0.02, npx_level=0.0)

            last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
            last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None
            if last_buy is None and last_sell is None:
                continue

            if last_sell is None or (last_buy is not None and last_buy >= last_sell):
                t, side = last_buy, "BUY"
            else:
                t, side = last_sell, "SELL"

            bars_since = int((len(s_show) - 1) - int(s_show.index.get_loc(t)))
            if bars_since <= int(max_bars):
                rows.append({
                    "Symbol": sym,
                    "Side": side,
                    "Bars Since": bars_since,
                    "Time": t,
                    "Last Price": float(s_show.iloc[-1]),
                    "NPX(last)": float(_coerce_1d_series(npx).dropna().iloc[-1]) if len(_coerce_1d_series(npx).dropna()) else np.nan,
                })

        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values(["Bars Since", "Side"], ascending=[True, True])
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 10 — Slope Direction Scan
# =========================
with tab10:
    st.header("Slope Direction Scan")
    st.caption("Lists symbols by local regression slope sign (daily view).")

    c1, c2 = st.columns(2)
    want = c1.selectbox("Slope sign", ["Up (m>0)", "Down (m<0)"], index=0, key=f"slope_sign_{mode}")
    run10 = c2.button("Run Slope Scan", key=f"btn_run_slope_{mode}", use_container_width=True)

    if run10:
        want_up = want.startswith("Up")
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 20:
                continue
            _, _, _, m, r2 = regression_with_band(s_show, lookback=min(len(s_show), slope_lb_daily))
            if not np.isfinite(m):
                continue
            if want_up and m <= 0:
                continue
            if (not want_up) and m >= 0:
                continue
            rows.append({"Symbol": sym, "Slope": float(m), "R2": float(r2) if np.isfinite(r2) else np.nan, "Last": float(s_show.iloc[-1])})
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values("Slope", ascending=not want_up)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 11 — Trendline Direction Lists
# =========================
with tab11:
    st.header("Trendline Direction Lists")
    st.caption("Uses the global trendline slope (in daily view) to split into Up vs Down.")

    run11 = st.button("Compute Trendline Lists", key=f"btn_run_trendlists_{mode}", use_container_width=True)
    if run11:
        ups, dns = [], []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 10:
                continue
            x = np.arange(len(s_show), dtype=float)
            m, _ = np.polyfit(x, s_show.to_numpy(dtype=float), 1)
            (ups if m > 0 else dns).append((sym, float(m)))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Uptrend (global slope > 0)")
            df = pd.DataFrame(ups, columns=["Symbol","Global Slope"]).sort_values("Global Slope", ascending=False)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)
        with c2:
            st.subheader("Downtrend (global slope < 0)")
            df = pd.DataFrame(dns, columns=["Symbol","Global Slope"]).sort_values("Global Slope", ascending=True)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 12 — NTD Hot List
# =========================
with tab12:
    st.header("NTD Hot List")
    st.caption("Sorts symbols by |NTD(last)| descending (daily view).")

    run12 = st.button("Run Hot List", key=f"btn_run_ntdhot_{mode}", use_container_width=True)
    if run12:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 10:
                continue
            ntd = compute_normalized_trend(s, window=ntd_window).reindex(s_show.index).dropna()
            if ntd.empty:
                continue
            rows.append({
                "Symbol": sym,
                "NTD(last)": float(ntd.iloc[-1]),
                "|NTD|": abs(float(ntd.iloc[-1])),
                "Last Price": float(s_show.iloc[-1]),
            })
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values("|NTD|", ascending=False)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 13 — NTD NPX 0.0-0.2 Scanner
# =========================
with tab13:
    st.header("NTD NPX 0.0-0.2 Scanner")
    st.caption("Finds symbols where NTD(last) is between 0.0 and 0.2 and NPX(last) is positive (daily view).")

    c1, c2 = st.columns(2)
    max_rows = c1.slider("Max rows", 10, 200, 50, 10, key=f"ntd002_rows_{mode}")
    run13 = c2.button("Run NTD/NPX Range Scan", key=f"btn_run_ntd002_{mode}", use_container_width=True)

    if run13:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 20:
                continue
            ntd = compute_normalized_trend(s, window=ntd_window).reindex(s_show.index).dropna()
            npx = compute_normalized_price(s, window=ntd_window).reindex(s_show.index).dropna()
            if ntd.empty or npx.empty:
                continue
            ntd_last = float(ntd.iloc[-1]); npx_last = float(npx.iloc[-1])
            if 0.0 <= ntd_last <= 0.2 and npx_last > 0:
                rows.append({"Symbol": sym, "NTD(last)": ntd_last, "NPX(last)": npx_last, "Last Price": float(s_show.iloc[-1])})
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values(["NTD(last)", "NPX(last)"], ascending=[True, False]).head(max_rows)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 14 — Uptrend vs Downtrend
# =========================
with tab14:
    st.header("Uptrend vs Downtrend")
    st.caption("Counts symbols by global slope sign (daily view).")

    run14 = st.button("Run Up/Down Summary", key=f"btn_run_updown_{mode}", use_container_width=True)
    if run14:
        up, dn, flat = [], [], []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 10:
                continue
            x = np.arange(len(s_show), dtype=float)
            m, _ = np.polyfit(x, s_show.to_numpy(dtype=float), 1)
            if abs(m) < 1e-12:
                flat.append(sym)
            elif m > 0:
                up.append(sym)
            else:
                dn.append(sym)

        c1, c2, c3 = st.columns(3)
        c1.metric("Uptrend", len(up))
        c2.metric("Downtrend", len(dn))
        c3.metric("Flat", len(flat))

        st.write("Uptrend:", ", ".join(up) if up else "—")
        st.write("Downtrend:", ", ".join(dn) if dn else "—")
        st.write("Flat:", ", ".join(flat) if flat else "—")

# =========================
# TAB 15 — Ichimoku Kijun Scanner
# =========================
with tab15:
    st.header("Ichimoku Kijun Scanner")
    st.caption("Lists symbols where price is above/below Kijun (daily view).")

    c1, c2 = st.columns(2)
    side = c1.selectbox("Condition", ["Price > Kijun", "Price < Kijun"], index=0, key=f"kij_cond_{mode}")
    run15 = c2.button("Run Kijun Scan", key=f"btn_run_kij_{mode}", use_container_width=True)

    if run15:
        want_above = side.startswith("Price >")
        rows = []
        for sym in universe:
            ohlc = fetch_hist_ohlc(sym)
            if ohlc is None or ohlc.empty:
                continue
            close = _coerce_1d_series(ohlc["Close"]).dropna()
            close_show = subset_by_daily_view(close, daily_view).dropna()
            if len(close_show) < 30:
                continue
            _, kijun, _, _, _ = ichimoku_lines(ohlc["High"], ohlc["Low"], ohlc["Close"],
                                               conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
            kijun = _coerce_1d_series(kijun).reindex(close_show.index).ffill().bfill()
            if kijun.dropna().empty:
                continue
            px = float(close_show.iloc[-1]); kj = float(kijun.iloc[-1])
            cond = (px > kj) if want_above else (px < kj)
            if cond:
                rows.append({"Symbol": sym, "Last Price": px, "Kijun": kj, "Diff%": (px/kj - 1.0) if kj else np.nan})
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values("Diff%", ascending=not want_above)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 16 — R² > 45% Daily/Hourly
# =========================
with tab16:
    st.header("R² > 45% Daily/Hourly")
    st.caption("Filters symbols where regression R² exceeds 0.45 (daily view + optional intraday).")

    c1, c2, c3 = st.columns(3)
    check_hourly = c1.checkbox("Also check hourly", value=False, key=f"r2hi_hr_{mode}")
    hours = c2.selectbox("Hourly window", ["24h", "48h", "96h"], index=0, key=f"r2hi_win_{mode}")
    run16 = c3.button("Run R² High Scan", key=f"btn_run_r2hi_{mode}", use_container_width=True)

    if run16:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 20:
                continue
            _, _, _, m, r2 = regression_with_band(s_show, lookback=min(len(s_show), slope_lb_daily))
            if np.isfinite(r2) and r2 >= 0.45:
                rows.append({"Symbol": sym, "Frame": "Daily", "R2": float(r2), "Slope": float(m) if np.isfinite(m) else np.nan})
        df_d = pd.DataFrame(rows).sort_values("R2", ascending=False) if rows else pd.DataFrame()
        st.subheader("Daily")
        st.dataframe(df_d.reset_index(drop=True), use_container_width=True) if not df_d.empty else st.write("No matches.")

        if check_hourly:
            rows = []
            for sym in universe:
                df = fetch_intraday(sym, period=period_map[hours])
                if df is None or df.empty or "Close" not in df:
                    continue
                df2 = df.copy(); df2.index = pd.RangeIndex(len(df2))
                hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
                if len(hc) < 40:
                    continue
                _, _, _, m, r2 = regression_with_band(hc, lookback=min(len(hc), slope_lb_hourly))
                if np.isfinite(r2) and r2 >= 0.45:
                    rows.append({"Symbol": sym, "Frame": f"Hourly({hours})", "R2": float(r2), "Slope": float(m) if np.isfinite(m) else np.nan})
            df_h = pd.DataFrame(rows).sort_values("R2", ascending=False) if rows else pd.DataFrame()
            st.subheader("Hourly")
            st.dataframe(df_h.reset_index(drop=True), use_container_width=True) if not df_h.empty else st.write("No matches.")

# =========================
# TAB 17 — R² < 45% Daily/Hourly
# =========================
with tab17:
    st.header("R² < 45% Daily/Hourly")
    st.caption("Filters symbols where regression R² is below 0.45 (daily view + optional intraday).")

    c1, c2, c3 = st.columns(3)
    check_hourly = c1.checkbox("Also check hourly", value=False, key=f"r2lo_hr_{mode}")
    hours = c2.selectbox("Hourly window", ["24h", "48h", "96h"], index=0, key=f"r2lo_win_{mode}")
    run17 = c3.button("Run R² Low Scan", key=f"btn_run_r2lo_{mode}", use_container_width=True)

    if run17:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 20:
                continue
            _, _, _, m, r2 = regression_with_band(s_show, lookback=min(len(s_show), slope_lb_daily))
            if np.isfinite(r2) and r2 < 0.45:
                rows.append({"Symbol": sym, "Frame": "Daily", "R2": float(r2), "Slope": float(m) if np.isfinite(m) else np.nan})
        df_d = pd.DataFrame(rows).sort_values("R2", ascending=True) if rows else pd.DataFrame()
        st.subheader("Daily")
        st.dataframe(df_d.reset_index(drop=True), use_container_width=True) if not df_d.empty else st.write("No matches.")

        if check_hourly:
            rows = []
            for sym in universe:
                df = fetch_intraday(sym, period=period_map[hours])
                if df is None or df.empty or "Close" not in df:
                    continue
                df2 = df.copy(); df2.index = pd.RangeIndex(len(df2))
                hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
                if len(hc) < 40:
                    continue
                _, _, _, m, r2 = regression_with_band(hc, lookback=min(len(hc), slope_lb_hourly))
                if np.isfinite(r2) and r2 < 0.45:
                    rows.append({"Symbol": sym, "Frame": f"Hourly({hours})", "R2": float(r2), "Slope": float(m) if np.isfinite(m) else np.nan})
            df_h = pd.DataFrame(rows).sort_values("R2", ascending=True) if rows else pd.DataFrame()
            st.subheader("Hourly")
            st.dataframe(df_h.reset_index(drop=True), use_container_width=True) if not df_h.empty else st.write("No matches.")

# =========================
# TAB 18 — R² Sign ±2σ Proximity (Daily)
# =========================
with tab18:
    st.header("R² Sign ±2σ Proximity (Daily)")
    st.caption("Shows symbols where price is near +2σ or -2σ, with R² info.")

    c1, c2, c3 = st.columns(3)
    near_pct = c1.slider("Near band threshold (% of price)", 0.05, 2.0, 0.25, 0.05, key=f"band_near_{mode}") / 100.0
    min_r2 = c2.slider("Min R²", 0.00, 0.90, 0.45, 0.05, key=f"band_minr2_{mode}")
    run18 = c3.button("Run ±2σ Proximity Scan", key=f"btn_run_bandprox_{mode}", use_container_width=True)

    if run18:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            s_show = subset_by_daily_view(s, daily_view).dropna()
            if len(s_show) < 30:
                continue
            yhat, up, lo, m, r2 = regression_with_band(s_show, lookback=min(len(s_show), slope_lb_daily))
            if not (np.isfinite(r2) and r2 >= float(min_r2)):
                continue
            px = float(s_show.iloc[-1])
            upv = float(up.iloc[-1]) if len(up.dropna()) else np.nan
            lov = float(lo.iloc[-1]) if len(lo.dropna()) else np.nan
            if not (np.isfinite(upv) and np.isfinite(lov) and np.isfinite(px) and px != 0):
                continue
            d_up = abs(upv - px) / px
            d_lo = abs(px - lov) / px
            prox = None
            if d_up <= near_pct:
                prox = "Near +2σ"
            if d_lo <= near_pct:
                prox = "Near -2σ" if prox is None else (prox + " & -2σ")
            if prox is None:
                continue
            rows.append({
                "Symbol": sym,
                "Proximity": prox,
                "Last": px,
                "Slope": float(m),
                "R2": float(r2),
                "DistTo+2σ%": d_up,
                "DistTo-2σ%": d_lo
            })
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values(["R2", "Slope"], ascending=[False, False])
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 19 — Zero cross
# =========================
with tab19:
    st.header("Zero cross")
    st.caption(
        "Lists symbols where **NTD (win=60)** has **recently crossed UP through 0.0**, "
        "**NTD is heading upward**, and the **Daily chart global trendline is UP**.\n\n"
        "Scan uses the selected **Daily view range**."
    )

    c1, c2, c3 = st.columns(3)
    max_bars = c1.slider("Max bars since NTD 0.0 cross-up", 0, 60, 3, 1, key=f"zc_max_bars_{mode}")
    confirm_bars = c2.slider("NTD heading-up confirmation (bars)", 1, 5, 1, 1, key=f"zc_confirm_{mode}")
    run_zero = c3.button("Run Zero cross Scan", key=f"btn_run_zero_cross_{mode}", use_container_width=True)

    if run_zero:
        rows = []
        for sym in universe:
            r = last_daily_ntd_zero_cross_up_in_uptrend(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=60,
                confirm_bars=int(confirm_bars),
            )
            if r is None:
                continue
            if int(r.get("Bars Since Cross", 9999)) <= int(max_bars):
                rows.append(r)

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out = out.sort_values(["Bars Since Cross", "NTD (last)", "Global Slope"], ascending=[True, False, False])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# =========================
# TAB 20 — Slope Candidates
# =========================
with tab20:
    st.header("Slope Candidates")
    st.caption(
        "Buy Candidates: regression slope > 0 and current price is **below** the regression line.\n"
        "Sell Candidates: regression slope < 0 and current price is **above** the regression line.\n\n"
        "Tables show: Symbol, Current Slope Price, Current Regression line price, Distance to regression line (ascending)."
    )

    c1, c2 = st.columns(2)
    max_rows = c1.slider("Max rows per list", 10, 200, 50, 10, key=f"slopecand_rows_{mode}")
    run20 = c2.button("Run Slope Candidates Scan", key=f"btn_run_slopecand_{mode}", use_container_width=True)

    if run20:
        buy_rows, sell_rows = [], []
        for sym in universe:
            r = slope_candidate_row(sym, daily_view_label=daily_view, slope_lb=slope_lb_daily)
            if r is None:
                continue

            m = float(r.get("_slope", np.nan))
            signed = float(r.get("_signed", np.nan))

            if np.isfinite(m) and np.isfinite(signed) and (m > 0.0) and (signed < 0.0):
                buy_rows.append(r)
            if np.isfinite(m) and np.isfinite(signed) and (m < 0.0) and (signed > 0.0):
                sell_rows.append(r)

        show_cols = ["Symbol", "Current Slope Price", "Current Regression line price", "Distance to regression line"]

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Buy Candidates")
            if not buy_rows:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(buy_rows).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("Sell Candidates")
            if not sell_rows:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(sell_rows).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfs[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 21 — Reversal Candidates
# =========================
with tab21:
    st.header("Reversal Candidates")
    st.caption(
        "Buy: Price bounced up from **-2σ**, with **local slope > 0** and **global trendline > 0**.\n"
        "Sell: Price bounced down from **+2σ**, with **local slope < 0** and **global trendline < 0**.\n\n"
        "Distance is **abs(price − regression)**, sorted ascending."
    )

    c1, c2, c3, c4 = st.columns(4)
    within_daily = c1.selectbox("Daily: within N bars", [3, 5, 10], index=1, key=f"rev_within_d_{mode}")
    within_hourly = c2.selectbox("Hourly: within N bars", [3, 5, 10], index=2, key=f"rev_within_h_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"rev_hr_win_{mode}")
    max_rows = c4.slider("Max rows per list", 10, 200, 50, 10, key=f"rev_rows_{mode}")

    run21 = st.button("Run Reversal Candidates Scan", key=f"btn_run_reversal_{mode}", use_container_width=True)

    if run21:
        show_cols = ["Symbol", "Bars Since", "Current Slope Price", "Current Regression line price", "Distance to regression line"]

        st.subheader("Daily")
        d_buy, d_sell = [], []
        for sym in universe:
            r = reversal_candidate_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                max_bars_since=int(within_daily),
            )
            if not r:
                continue
            if r.get("Side") == "BUY":
                d_buy.append(r)
            elif r.get("Side") == "SELL":
                d_sell.append(r)

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Daily Buy Candidates")
            if not d_buy:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(d_buy).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("Daily Sell Candidates")
            if not d_sell:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(d_sell).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfs[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        st.subheader(f"Hourly ({hours})")
        h_buy, h_sell = [], []
        for sym in universe:
            r = reversal_candidate_row_hourly(
                symbol=sym,
                period=period_map[hours],
                slope_lb=slope_lb_hourly,
                max_bars_since=int(within_hourly),
            )
            if not r:
                continue
            if r.get("Side") == "BUY":
                h_buy.append(r)
            elif r.get("Side") == "SELL":
                h_sell.append(r)

        cL2, cR2 = st.columns(2)
        with cL2:
            st.subheader("Hourly Buy Candidates")
            if not h_buy:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(h_buy).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR2:
            st.subheader("Hourly Sell Candidates")
            if not h_sell:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(h_sell).sort_values("Distance to regression line", ascending=True)
                st.dataframe(dfs[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 22 — NTD -0.5 Cross
# =========================
with tab22:
    st.header("NTD -0.5 Cross")
    st.caption(
        "Lists symbols where **regression slope > 0** and **NTD crossed up into [-0.5, -0.4]**.\n"
        "Daily uses selected Daily view range; Hourly uses 24/48/96 lookback."
    )

    c1, c2, c3, c4 = st.columns(4)
    within_daily = c1.selectbox("Daily: within N bars", [3, 5, 10], index=1, key=f"ntd05_within_d_{mode}")
    within_hourly = c2.selectbox("Hourly: within N bars", [3, 5, 10], index=2, key=f"ntd05_within_h_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"ntd05_hr_win_{mode}")
    max_rows = c4.slider("Max rows", 10, 200, 50, 10, key=f"ntd05_rows_{mode}")

    run22 = st.button("Run NTD -0.5 Cross Scan", key=f"btn_run_ntd05_{mode}", use_container_width=True)

    if run22:
        st.subheader("Daily")
        d_rows = []
        for sym in universe:
            r = ntd_minus05_cross_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                ntd_win=60,
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        if not d_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(d_rows).sort_values(["Bars Since", "NTD(last)"], ascending=[True, False])
            st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

        st.subheader(f"Hourly ({hours})")
        h_rows = []
        for sym in universe:
            r = ntd_minus05_cross_row_hourly(
                symbol=sym,
                period=period_map[hours],
                slope_lb=slope_lb_hourly,
                ntd_win=60,
                max_bars_since=int(within_hourly),
            )
            if r:
                h_rows.append(r)

        if not h_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(h_rows).sort_values(["Bars Since", "NTD(last)"], ascending=[True, False])
            st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 23 — Trend and Slope Align  ✅ UPDATED (cross metadata shown in ALL tables)
# =========================
with tab23:
    st.header("Trend and Slope Align")
    st.caption(
        "Buy Opportunities: **Trendline > 0 AND Regression Slope > 0**.\n"
        "Sell Opportunities: **Trendline < 0 AND Regression Slope < 0**.\n\n"
        "Shows both **Daily** and **Hourly** results, including a focused Buy list where "
        "**NPX (Norm Price) recently crossed the NTD line**.\n\n"
        "Trendline slope = global slope of the selected frame.\n"
        "Regression slope = local regression slope over the configured lookback.\n"
        "NPX↔NTD cross = NPX crosses above/below NTD on the indicator panel scale."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"tsa_rows_{mode}")
    min_abs_slope = c2.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"tsa_minabs_{mode}")
    max_cross_bars = c3.slider("Max bars since NPX↔NTD cross", 0, 60, 5, 1, key=f"tsa_cross_bars_{mode}")
    tsa_hours = c4.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"tsa_hr_win_{mode}")
    run23 = c5.button("Run Trend/Slope Align Scan", key=f"btn_run_tsa_{mode}", use_container_width=True)

    if run23:
        # ---------- DAILY ----------
        buys, sells, buys_cross = [], [], []

        for sym in universe:
            r = trend_slope_align_row(sym, daily_view_label=daily_view, slope_lb=slope_lb_daily)
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if not (np.isfinite(tm) and np.isfinite(rm)):
                continue

            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue

            # Add latest NPX↔NTD cross metadata to ALL rows/tables
            cross_meta = trend_slope_align_cross_meta_daily(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=int(ntd_window),
            )
            r = r.copy()
            r["Bars Since Cross"] = cross_meta.get("Bars Since Cross", np.nan)
            r["Cross Direction"] = cross_meta.get("Cross Direction", None)
            r["Time Crossed"] = cross_meta.get("Time Crossed", None)
            r["NPX(last)"] = cross_meta.get("NPX(last)", np.nan)
            r["NTD(last)"] = cross_meta.get("NTD(last)", np.nan)

            # BUY / SELL classification (daily)
            if (tm > 0.0) and (rm > 0.0):
                buys.append(r)
            elif (tm < 0.0) and (rm < 0.0):
                sells.append(r)

            # Daily Buy + NPX recently crossed NTD (fresh cross)
            bsc = r.get("Bars Since Cross", np.nan)
            if (tm > 0.0) and (rm > 0.0) and np.isfinite(bsc) and int(bsc) <= int(max_cross_bars):
                buys_cross.append(r.copy())

        # ---------- HOURLY ----------
        h_buys, h_sells, h_buys_cross = [], [], []
        tsa_period = period_map[tsa_hours]

        for sym in universe:
            r = trend_slope_align_row_hourly(sym, period=tsa_period, slope_lb=slope_lb_hourly)
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if not (np.isfinite(tm) and np.isfinite(rm)):
                continue

            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue

            # Add latest NPX↔NTD cross metadata to ALL rows/tables
            cross_meta = trend_slope_align_cross_meta_hourly(
                symbol=sym,
                period=tsa_period,
                ntd_win=int(ntd_window),
            )
            r = r.copy()
            r["Bars Since Cross"] = cross_meta.get("Bars Since Cross", np.nan)
            r["Cross Direction"] = cross_meta.get("Cross Direction", None)
            r["Time Crossed"] = cross_meta.get("Time Crossed", None)
            r["NPX(last)"] = cross_meta.get("NPX(last)", np.nan)
            r["NTD(last)"] = cross_meta.get("NTD(last)", np.nan)

            if (tm > 0.0) and (rm > 0.0):
                h_buys.append(r)
            elif (tm < 0.0) and (rm < 0.0):
                h_sells.append(r)

            bsc = r.get("Bars Since Cross", np.nan)
            if (tm > 0.0) and (rm > 0.0) and np.isfinite(bsc) and int(bsc) <= int(max_cross_bars):
                h_buys_cross.append(r.copy())

        # ---------- DISPLAY: DAILY ----------
        show_cols_buy = [
            "Symbol", "Trendline Slope", "Regression Slope", "R2", "Last Price",
            "Bars Since Cross", "Cross Direction", "Time Crossed"
        ]

        st.subheader("Daily")
        cL, cR = st.columns(2)
        with cL:
            st.subheader("Buy Opportunities (Trendline > 0 & Regression Slope > 0)")
            if not buys:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(buys)
                dfb["_score"] = dfb["Trendline Slope"].astype(float) + dfb["Regression Slope"].astype(float)
                dfb = dfb.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(dfb[show_cols_buy].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("Buy + NPX recently crossed NTD")
            if not buys_cross:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(buys_cross)
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                show_cols_cross = [
                    "Symbol", "Trendline Slope", "Regression Slope", "R2", "Last Price",
                    "Bars Since Cross", "Cross Direction", "Time Crossed", "NPX(last)", "NTD(last)"
                ]
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfc[show_cols_cross].head(max_rows).reset_index(drop=True), use_container_width=True)

        st.subheader("Sell Opportunities (Trendline < 0 & Regression Slope < 0)")
        if not sells:
            st.write("No matches.")
        else:
            dfs = pd.DataFrame(sells)
            dfs["_score"] = dfs["Trendline Slope"].astype(float) + dfs["Regression Slope"].astype(float)
            dfs = dfs.sort_values(["_score", "R2"], ascending=[True, False])
            st.dataframe(dfs[show_cols_buy].head(max_rows).reset_index(drop=True), use_container_width=True)

        # ---------- DISPLAY: HOURLY ----------
        st.subheader(f"Hourly ({tsa_hours})")
        cLh, cRh = st.columns(2)
        with cLh:
            st.subheader("Buy Opportunities (Trendline > 0 & Regression Slope > 0)")
            if not h_buys:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(h_buys)
                dfb["_score"] = dfb["Trendline Slope"].astype(float) + dfb["Regression Slope"].astype(float)
                show_cols_buy_h = [
                    "Symbol", "Frame", "Trendline Slope", "Regression Slope", "R2", "Last Price",
                    "Bars Since Cross", "Cross Direction", "Time Crossed"
                ]
                dfb = dfb.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(dfb[show_cols_buy_h].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cRh:
            st.subheader("Buy + NPX recently crossed NTD")
            if not h_buys_cross:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(h_buys_cross)
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                show_cols_cross_h = [
                    "Symbol", "Frame", "Trendline Slope", "Regression Slope", "R2", "Last Price",
                    "Bars Since Cross", "Cross Direction", "Time Crossed", "NPX(last)", "NTD(last)"
                ]
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfc[show_cols_cross_h].head(max_rows).reset_index(drop=True), use_container_width=True)

        st.subheader("Sell Opportunities (Trendline < 0 & Regression Slope < 0)")
        if not h_sells:
            st.write("No matches.")
        else:
            dfs = pd.DataFrame(h_sells)
            dfs["_score"] = dfs["Trendline Slope"].astype(float) + dfs["Regression Slope"].astype(float)
            show_cols_sell_h = [
                "Symbol", "Frame", "Trendline Slope", "Regression Slope", "R2", "Last Price",
                "Bars Since Cross", "Cross Direction", "Time Crossed"
            ]
            dfs = dfs.sort_values(["_score", "R2"], ascending=[True, False])
            st.dataframe(dfs[show_cols_sell_h].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 24 — NPX 0.0 Cross Trend/PTD  ✅ NEW
# =========================
with tab24:
    st.header("NPX 0.0 Cross Trend/PTD")
    st.caption(
        "(1) Lists symbols where **Trendline > 0** and **NPX (Norm Price)** recently crossed the **0.0** line "
        "on the indicator panel (Daily + Hourly).\n"
        "(2) Lists symbols where **Trendline > 0 AND Regression Slope > 0** and **NPX** recently crossed **0.0** "
        "(interpreting your PTD request as the stricter Trend+Slope filter).\n\n"
        "Cross direction can be **Up** or **Down** unless you later want an up-only version."
    )

    c1, c2, c3, c4 = st.columns(4)
    within_daily = c1.selectbox("Daily: within N bars", [3, 5, 10, 15, 20], index=1, key=f"npx0trend_d_{mode}")
    within_hourly = c2.selectbox("Hourly: within N bars", [3, 5, 10, 15, 20], index=2, key=f"npx0trend_h_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"npx0trend_hrwin_{mode}")
    max_rows = c4.slider("Max rows per table", 10, 300, 60, 10, key=f"npx0trend_rows_{mode}")

    run24 = st.button("Run NPX 0.0 Cross Trend/PTD Scan", key=f"btn_run_npx0trend_{mode}", use_container_width=True)

    if run24:
        show_cols = [
            "Symbol", "Bars Since", "Cross Dir", "NPX@Cross", "NPX(last)",
            "Trendline Slope", "Regression Slope", "R2", "Last Price", "Cross Time"
        ]

        # ---------- DAILY ----------
        daily_all, daily_trend_slope = [], []
        for sym in universe:
            r = npx_zero_cross_trend_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_daily),
            )
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))

            if np.isfinite(tm) and (tm > 0.0):
                daily_all.append(r)
                if np.isfinite(rm) and (rm > 0.0):
                    daily_trend_slope.append(r)

        st.subheader("Daily")
        cL1, cR1 = st.columns(2)

        with cL1:
            st.subheader("Trendline > 0 + NPX recently crossed 0.0")
            if not daily_all:
                st.write("No matches.")
            else:
                dfa = pd.DataFrame(daily_all)
                dfa["_score"] = dfa["Trendline Slope"].astype(float).fillna(0) + dfa["Regression Slope"].astype(float).fillna(0)
                dfa = dfa.sort_values(["Bars Since", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfa[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR1:
            st.subheader("Trendline > 0 & Slope > 0 + NPX recently crossed 0.0 (PTD)")
            if not daily_trend_slope:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(daily_trend_slope)
                dfb["_score"] = dfb["Trendline Slope"].astype(float).fillna(0) + dfb["Regression Slope"].astype(float).fillna(0)
                dfb = dfb.sort_values(["Bars Since", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        # ---------- HOURLY ----------
        hourly_all, hourly_trend_slope = [], []
        for sym in universe:
            r = npx_zero_cross_trend_row_hourly(
                symbol=sym,
                period=period_map[hours],
                slope_lb=slope_lb_hourly,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_hourly),
            )
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))

            if np.isfinite(tm) and (tm > 0.0):
                hourly_all.append(r)
                if np.isfinite(rm) and (rm > 0.0):
                    hourly_trend_slope.append(r)

        st.subheader(f"Hourly ({hours})")
        cL2, cR2 = st.columns(2)

        with cL2:
            st.subheader("Trendline > 0 + NPX recently crossed 0.0")
            if not hourly_all:
                st.write("No matches.")
            else:
                dfa = pd.DataFrame(hourly_all)
                dfa["_score"] = dfa["Trendline Slope"].astype(float).fillna(0) + dfa["Regression Slope"].astype(float).fillna(0)
                dfa = dfa.sort_values(["Bars Since", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfa[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR2:
            st.subheader("Trendline > 0 & Slope > 0 + NPX recently crossed 0.0 (PTD)")
            if not hourly_trend_slope:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(hourly_trend_slope)
                dfb["_score"] = dfb["Trendline Slope"].astype(float).fillna(0) + dfb["Regression Slope"].astype(float).fillna(0)
                dfb = dfb.sort_values(["Bars Since", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 25 — Magic Cross  ✅ NEW
# =========================
with tab25:
    st.header("Magic Cross")
    st.caption(
        "Shows symbols where **NPX (Norm Price)** recently crossed **UP** above the **NTD** line **in the positive zone** "
        "(cross occurs with NPX and NTD between **0.0 and 1.0** on the indicator scale).\n\n"
        "For each frame, lists:\n"
        "1) **Regression Slope > 0** + Magic Cross Up\n"
        "2) **Global Trendline > 0** + Magic Cross Up\n"
        "3) **Global Trendline > 0 AND Regression Slope > 0** + Magic Cross Up"
    )

    c1, c2, c3 = st.columns(3)
    within_daily = c1.selectbox("Daily: within N bars", [3, 5, 10, 15, 20], index=1, key=f"magic_within_d_{mode}")
    within_hourly = c2.selectbox("Hourly (24h/48h): within N bars", [3, 5, 10, 15, 20], index=2, key=f"magic_within_h_{mode}")
    max_rows = c3.slider("Max rows per table", 10, 300, 60, 10, key=f"magic_rows_{mode}")

    run_magic = st.button("Run Magic Cross Scan", key=f"btn_run_magic_cross_{mode}", use_container_width=True)

    if run_magic:
        show_cols_daily = [
            "Symbol", "Bars Since Cross", "Cross Time",
            "NPX@Cross", "NTD@Cross", "NPX(last)", "NTD(last)",
            "Trendline Slope", "Regression Slope", "R2", "Last Price"
        ]
        show_cols_hourly = [
            "Symbol", "Frame", "Bars Since Cross", "Cross Time",
            "NPX@Cross", "NTD@Cross", "NPX(last)", "NTD(last)",
            "Trendline Slope", "Regression Slope", "R2", "Last Price"
        ]

        def _sort_magic(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            dfx = df.copy()
            dfx["_score"] = dfx["Trendline Slope"].astype(float).fillna(0) + dfx["Regression Slope"].astype(float).fillna(0)
            return dfx.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])

        # -------------------------
        # DAILY rows
        # -------------------------
        daily_rows = []
        for sym in universe:
            try:
                # Slopes / R2 / last price
                base = trend_slope_align_row(sym, daily_view_label=daily_view, slope_lb=slope_lb_daily)
                if not base:
                    continue

                    # (kept original indentation/logic style)
                tm = float(base.get("Trendline Slope", np.nan))
                rm = float(base.get("Regression Slope", np.nan))
                r2v = float(base.get("R2", np.nan)) if np.isfinite(base.get("R2", np.nan)) else np.nan
                last_px = float(base.get("Last Price", np.nan)) if np.isfinite(base.get("Last Price", np.nan)) else np.nan

                # NTD/NPX (daily view slice)
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if len(close_show) < 20:
                    continue

                ntd = _coerce_1d_series(compute_normalized_trend(close_full, window=int(ntd_window))).reindex(close_show.index)
                npx = _coerce_1d_series(compute_normalized_price(close_full, window=int(ntd_window))).reindex(close_show.index)

                ok = ntd.notna() & npx.notna()
                if ok.sum() < 2:
                    continue
                ntd = ntd[ok]
                npx = npx[ok]

                up_mask, _ = _cross_series(npx, ntd)  # only UP cross requested
                up_mask = up_mask.reindex(ntd.index, fill_value=False)

                # Positive-zone Magic Cross (0.0 to 1.0 for both series at cross)
                zone_mask = (
                    (npx >= 0.0) & (npx <= 1.0) &
                    (ntd >= 0.0) & (ntd <= 1.0)
                ).reindex(ntd.index, fill_value=False)

                magic_mask = up_mask & zone_mask
                if not magic_mask.any():
                    continue

                t_cross = magic_mask[magic_mask].index[-1]
                bars_since = int((len(ntd) - 1) - int(ntd.index.get_loc(t_cross)))
                if bars_since > int(within_daily):
                    continue

                row = {
                    "Symbol": sym,
                    "Bars Since Cross": int(bars_since),
                    "Cross Time": t_cross,
                    "NPX@Cross": float(npx.loc[t_cross]) if np.isfinite(npx.loc[t_cross]) else np.nan,
                    "NTD@Cross": float(ntd.loc[t_cross]) if np.isfinite(ntd.loc[t_cross]) else np.nan,
                    "NPX(last)": float(npx.iloc[-1]) if np.isfinite(npx.iloc[-1]) else np.nan,
                    "NTD(last)": float(ntd.iloc[-1]) if np.isfinite(ntd.iloc[-1]) else np.nan,
                    "Trendline Slope": tm,
                    "Regression Slope": rm,
                    "R2": r2v,
                    "Last Price": last_px,
                }
                daily_rows.append(row)
            except Exception:
                continue

        d_reg_up = [r for r in daily_rows if np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Regression Slope"]) > 0.0]
        d_trend_up = [r for r in daily_rows if np.isfinite(r.get("Trendline Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0]
        d_both_up = [r for r in daily_rows if (
            np.isfinite(r.get("Trendline Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0 and
            np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Regression Slope"]) > 0.0
        )]

        st.subheader("Daily Chart")
        d1, d2, d3 = st.columns(3)

        with d1:
            st.subheader("1) Regression Slope > 0 + Magic Cross Up")
            if not d_reg_up:
                st.write("No matches.")
            else:
                df = _sort_magic(pd.DataFrame(d_reg_up))
                st.dataframe(df[show_cols_daily].head(max_rows).reset_index(drop=True), use_container_width=True)

        with d2:
            st.subheader("2) Global Trendline > 0 + Magic Cross Up")
            if not d_trend_up:
                st.write("No matches.")
            else:
                df = _sort_magic(pd.DataFrame(d_trend_up))
                st.dataframe(df[show_cols_daily].head(max_rows).reset_index(drop=True), use_container_width=True)

        with d3:
            st.subheader("3) Trendline > 0 & Regression Slope > 0 + Magic Cross Up")
            if not d_both_up:
                st.write("No matches.")
            else:
                df = _sort_magic(pd.DataFrame(d_both_up))
                st.dataframe(df[show_cols_daily].head(max_rows).reset_index(drop=True), use_container_width=True)

        # -------------------------
        # HOURLY rows builder (24h, 48h)
        # -------------------------
        def _magic_rows_hourly(window_label: str):
            rows = []
            period = period_map[window_label]
            for sym in universe:
                try:
                    # slope metrics
                    base = trend_slope_align_row_hourly(sym, period=period, slope_lb=slope_lb_hourly)
                    if not base:
                        continue

                    tm = float(base.get("Trendline Slope", np.nan))
                    rm = float(base.get("Regression Slope", np.nan))
                    r2v = float(base.get("R2", np.nan)) if np.isfinite(base.get("R2", np.nan)) else np.nan
                    last_px = float(base.get("Last Price", np.nan)) if np.isfinite(base.get("Last Price", np.nan)) else np.nan

                    df = fetch_intraday(sym, period=period)
                    if df is None or df.empty or "Close" not in df.columns:
                        continue
                    real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

                    df2 = df.copy()
                    df2.index = pd.RangeIndex(len(df2))
                    hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
                    if len(hc) < 20:
                        continue

                    ntd = _coerce_1d_series(compute_normalized_trend(hc, window=int(ntd_window)))
                    npx = _coerce_1d_series(compute_normalized_price(hc, window=int(ntd_window)))
                    ok = ntd.notna() & npx.notna()
                    if ok.sum() < 2:
                        continue
                    ntd = ntd[ok]
                    npx = npx[ok]

                    up_mask, _ = _cross_series(npx, ntd)
                    up_mask = up_mask.reindex(ntd.index, fill_value=False)

                    zone_mask = (
                        (npx >= 0.0) & (npx <= 1.0) &
                        (ntd >= 0.0) & (ntd <= 1.0)
                    ).reindex(ntd.index, fill_value=False)

                    magic_mask = up_mask & zone_mask
                    if not magic_mask.any():
                        continue

                    bar = int(magic_mask[magic_mask].index[-1])
                    bars_since = int((len(hc) - 1) - bar)
                    if bars_since > int(within_hourly):
                        continue

                    cross_time = None
                    if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
                        cross_time = real_times[bar]

                    rows.append({
                        "Symbol": sym,
                        "Frame": f"Hourly ({window_label})",
                        "Bars Since Cross": int(bars_since),
                        "Cross Time": cross_time if cross_time is not None else bar,
                        "NPX@Cross": float(npx.loc[bar]) if np.isfinite(npx.loc[bar]) else np.nan,
                        "NTD@Cross": float(ntd.loc[bar]) if np.isfinite(ntd.loc[bar]) else np.nan,
                        "NPX(last)": float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan,
                        "NTD(last)": float(ntd.dropna().iloc[-1]) if len(ntd.dropna()) else np.nan,
                        "Trendline Slope": tm,
                        "Regression Slope": rm,
                        "R2": r2v,
                        "Last Price": last_px,
                    })
                except Exception:
                    continue
            return rows

        def _show_magic_hourly_section(window_label: str, rows: list):
            h_reg_up = [r for r in rows if np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Regression Slope"]) > 0.0]
            h_trend_up = [r for r in rows if np.isfinite(r.get("Trendline Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0]
            h_both_up = [r for r in rows if (
                np.isfinite(r.get("Trendline Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0 and
                np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Regression Slope"]) > 0.0
            )]

            st.subheader(f"Hourly Chart ({window_label})")
            c1h, c2h, c3h = st.columns(3)

            with c1h:
                st.subheader("1) Regression Slope > 0 + Magic Cross Up")
                if not h_reg_up:
                    st.write("No matches.")
                else:
                    df = _sort_magic(pd.DataFrame(h_reg_up))
                    st.dataframe(df[show_cols_hourly].head(max_rows).reset_index(drop=True), use_container_width=True)

            with c2h:
                st.subheader("2) Global Trendline > 0 + Magic Cross Up")
                if not h_trend_up:
                    st.write("No matches.")
                else:
                    df = _sort_magic(pd.DataFrame(h_trend_up))
                    st.dataframe(df[show_cols_hourly].head(max_rows).reset_index(drop=True), use_container_width=True)

            with c3h:
                st.subheader("3) Trendline > 0 & Regression Slope > 0 + Magic Cross Up")
                if not h_both_up:
                    st.write("No matches.")
                else:
                    df = _sort_magic(pd.DataFrame(h_both_up))
                    st.dataframe(df[show_cols_hourly].head(max_rows).reset_index(drop=True), use_container_width=True)

        rows_24 = _magic_rows_hourly("24h")
        _show_magic_hourly_section("24h", rows_24)

        rows_48 = _magic_rows_hourly("48h")
        _show_magic_hourly_section("48h", rows_48)

# =========================
# TAB 26 — Max History Marker ✅ NEW
# =========================
with tab26:
    st.header("Max History Marker")
    st.caption(
        "Scans the **Long-Term History (max-history)** NTD chart and lists symbols where the **NTD line** "
        "recently crossed the **0.00** line:\n"
        "• **Going Up** (crossed up through 0.00)\n"
        "• **Going Down** (crossed down through 0.00)\n\n"
        "This uses the same max-history NTD logic shown in the **Long-Term History** tab."
    )

    c1, c2, c3 = st.columns(3)
    within_bars = c1.selectbox("Within N bars (max-history NTD)", [3, 5, 10, 20, 30, 60], index=3, key=f"maxhist_marker_within_{mode}")
    max_rows = c2.slider("Max rows per list", 10, 300, 60, 10, key=f"maxhist_marker_rows_{mode}")
    run26 = c3.button("Run Max History Marker Scan", key=f"btn_run_max_history_marker_{mode}", use_container_width=True)

    if run26:
        up_rows, down_rows = [], []

        for sym in universe:
            r_up = max_history_ntd_zero_cross_row(
                symbol=sym,
                ntd_win=int(ntd_window),
                direction="up",
                max_bars_since=int(within_bars),
            )
            if r_up:
                up_rows.append(r_up)

            r_dn = max_history_ntd_zero_cross_row(
                symbol=sym,
                ntd_win=int(ntd_window),
                direction="down",
                max_bars_since=int(within_bars),
            )
            if r_dn:
                down_rows.append(r_dn)

        show_cols = [
            "Symbol",
            "Bars Since Cross",
            "Cross Time",
            "NTD@Cross",
            "NTD(last)",
            "Recent Global Slope",
            "Last Price",
        ]

        cL, cR = st.columns(2)

        with cL:
            st.subheader("NTD crossed 0.00 going UP (Max History)")
            if not up_rows:
                st.write("No matches.")
            else:
                dfu = pd.DataFrame(up_rows)
                # Prefer freshest crosses first, then stronger current NTD
                dfu = dfu.sort_values(["Bars Since Cross", "NTD(last)", "Recent Global Slope"], ascending=[True, False, False])
                st.dataframe(dfu[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("NTD crossed 0.00 going DOWN (Max History)")
            if not down_rows:
                st.write("No matches.")
            else:
                dfd = pd.DataFrame(down_rows)
                # Prefer freshest crosses first, then more negative current NTD
                dfd = dfd.sort_values(["Bars Since Cross", "NTD(last)", "Recent Global Slope"], ascending=[True, True, True])
                st.dataframe(dfd[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 27 — Daily Bet ✅ NEW
# =========================
with tab27:
    st.header("Daily Bet")
    st.caption(
        "Hourly-only scanner based on **Fibonacci reversal + NPX 0.0 cross** on the NTD panel.\n\n"
        "1) **Sell-side reversal list**: price recently touched **Fib 0%** and then **NPX crossed DOWN through 0.0**.\n"
        "2) **Buy-side reversal list**: price recently touched **Fib 100%** and then **NPX crossed UP through 0.0**.\n\n"
        "This uses the same Fib+NPX signal logic already used in the charts/scanners."
    )

    c1, c2, c3, c4 = st.columns(4)
    hours = c1.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"dailybet_hr_win_{mode}")
    within_bars = c2.selectbox("Within N bars", [3, 5, 10, 15, 20], index=2, key=f"dailybet_within_{mode}")
    touch_hz = c3.slider("Fib touch horizon (bars)", 3, 60, 15, 1, key=f"dailybet_touch_hz_{mode}")
    max_rows = c4.slider("Max rows per list", 10, 300, 60, 10, key=f"dailybet_rows_{mode}")

    run27 = st.button("Run Daily Bet Scan", key=f"btn_run_daily_bet_{mode}", use_container_width=True)

    if run27:
        sell_rows = []
        buy_rows = []

        hr_period = period_map[hours]

        for sym in universe:
            r_sell = daily_bet_signal_row_hourly(
                symbol=sym,
                period=hr_period,
                side="SELL",
                ntd_win=int(ntd_window),
                touch_horizon_bars=int(touch_hz),
                max_bars_since=int(within_bars),
            )
            if r_sell:
                sell_rows.append(r_sell)

            r_buy = daily_bet_signal_row_hourly(
                symbol=sym,
                period=hr_period,
                side="BUY",
                ntd_win=int(ntd_window),
                touch_horizon_bars=int(touch_hz),
                max_bars_since=int(within_bars),
            )
            if r_buy:
                buy_rows.append(r_buy)

        show_cols = [
            "Symbol", "Frame", "Touched Fib", "Cross Dir", "Bars Since",
            "Signal Time", "Signal Price", "Last Price", "NPX@Cross", "NPX(last)"
        ]

        cL, cR = st.columns(2)

        with cL:
            st.subheader("Hourly reversals from Fib 0% + NPX crossed 0.0 downward")
            if not sell_rows:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(sell_rows)
                dfs = dfs.sort_values(["Bars Since", "NPX(last)"], ascending=[True, True])
                st.dataframe(dfs[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("Hourly reversals from Fib 100% + NPX crossed 0.0 upward")
            if not buy_rows:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(buy_rows)
                dfb = dfb.sort_values(["Bars Since", "NPX(last)"], ascending=[True, False])
                st.dataframe(dfb[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 28 — Regression Reversal ✅ NEW
# =========================
with tab28:
    st.header("Regression Reversal")
    st.caption(
        "Daily and Hourly regression-reversal scanner.\n\n"
        "**BUY list**: Trendline and Regression slopes are upward (**> 0**) and price has recently reversed up "
        "from the **lower 2σ regression line** / support-side regression bounce.\n"
        "**SELL list**: Trendline and Regression slopes are downward (**< 0**) and price has recently reversed down "
        "from the **upper 2σ regression line** / resistance-side regression bounce."
    )

    c1, c2, c3, c4 = st.columns(4)
    within_daily = c1.selectbox("Daily: within N bars", [3, 5, 10], index=1, key=f"regrev_within_d_{mode}")
    within_hourly = c2.selectbox("Hourly: within N bars", [3, 5, 10], index=2, key=f"regrev_within_h_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"regrev_hr_win_{mode}")
    max_rows = c4.slider("Max rows per list", 10, 200, 50, 10, key=f"regrev_rows_{mode}")

    run_regrev = st.button("Run Regression Reversal Scan", key=f"btn_run_regrev_{mode}", use_container_width=True)

    if run_regrev:
        # DAILY
        d_buy, d_sell = [], []
        for sym in universe:
            r = reversal_candidate_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                max_bars_since=int(within_daily),
            )
            if not r:
                continue
            # expose hidden fields for display
            r = r.copy()
            r["Trendline Slope"] = float(r.get("_global_slope", np.nan)) if np.isfinite(r.get("_global_slope", np.nan)) else np.nan
            r["Regression Slope"] = float(r.get("_local_slope", np.nan)) if np.isfinite(r.get("_local_slope", np.nan)) else np.nan
            r["R2"] = float(r.get("_r2", np.nan)) if np.isfinite(r.get("_r2", np.nan)) else np.nan
            if r.get("Side") == "BUY":
                d_buy.append(r)
            elif r.get("Side") == "SELL":
                d_sell.append(r)

        st.subheader("Daily Chart")
        cL, cR = st.columns(2)

        daily_cols = [
            "Symbol", "Side", "Bars Since",
            "Current Slope Price", "Current Regression line price", "Distance to regression line",
            "Trendline Slope", "Regression Slope", "R2"
        ]

        with cL:
            st.subheader("Daily BUY Reversals (uptrend + lower 2σ reversal up)")
            if not d_buy:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(d_buy).sort_values(["Bars Since", "Distance to regression line"], ascending=[True, True])
                st.dataframe(dfb[daily_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("Daily SELL Reversals (downtrend + upper 2σ reversal down)")
            if not d_sell:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(d_sell).sort_values(["Bars Since", "Distance to regression line"], ascending=[True, True])
                st.dataframe(dfs[daily_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        # HOURLY
        h_buy, h_sell = [], []
        for sym in universe:
            r = reversal_candidate_row_hourly(
                symbol=sym,
                period=period_map[hours],
                slope_lb=slope_lb_hourly,
                max_bars_since=int(within_hourly),
            )
            if not r:
                continue
            r = r.copy()
            r["Trendline Slope"] = float(r.get("_global_slope", np.nan)) if np.isfinite(r.get("_global_slope", np.nan)) else np.nan
            r["Regression Slope"] = float(r.get("_local_slope", np.nan)) if np.isfinite(r.get("_local_slope", np.nan)) else np.nan
            r["R2"] = float(r.get("_r2", np.nan)) if np.isfinite(r.get("_r2", np.nan)) else np.nan
            if r.get("Side") == "BUY":
                h_buy.append(r)
            elif r.get("Side") == "SELL":
                h_sell.append(r)

        st.subheader(f"Hourly Chart ({hours})")
        cL2, cR2 = st.columns(2)

        hourly_cols = [
            "Symbol", "Frame", "Side", "Bars Since", "Signal Time",
            "Current Slope Price", "Current Regression line price", "Distance to regression line",
            "Trendline Slope", "Regression Slope", "R2"
        ]

        with cL2:
            st.subheader("Hourly BUY Reversals (uptrend + lower 2σ reversal up)")
            if not h_buy:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(h_buy).sort_values(["Bars Since", "Distance to regression line"], ascending=[True, True])
                st.dataframe(dfb[hourly_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR2:
            st.subheader("Hourly SELL Reversals (downtrend + upper 2σ reversal down)")
            if not h_sell:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(h_sell).sort_values(["Bars Since", "Distance to regression line"], ascending=[True, True])
                st.dataframe(dfs[hourly_cols].head(max_rows).reset_index(drop=True), use_container_width=True)
