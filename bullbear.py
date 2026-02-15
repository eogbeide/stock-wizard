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
# Regression / bands / triggers
# =========================
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
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    mask = p.notna() & u.notna() & l.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]; u = u[mask]; l = l[mask]

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
        if not candidates.any():
            return None
        t = candidates[candidates].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    else:
        candidates = inside & above.shift(1, fill_value=False)
        if not candidates.any():
            return None
        t = candidates[candidates].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL"}

def _cross_series(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index if len(p) else l.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)
    p = p[ok]; l = l[ok]
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

# =========================
# Fibonacci
# =========================
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

def fib_touch_masks(price: pd.Series, proximity_pct_of_range: float = 0.02):
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

def npx_zero_cross_masks(npx: pd.Series, level: float = 0.0):
    s = _coerce_1d_series(npx)
    prev = s.shift(1)
    cross_up = (s >= float(level)) & (prev < float(level))
    cross_dn = (s <= float(level)) & (prev > float(level))
    return cross_up.fillna(False), cross_dn.fillna(False)

def fib_npx_zero_cross_signal_masks(price: pd.Series,
                                   npx: pd.Series,
                                   horizon_bars: int = 15,
                                   proximity_pct_of_range: float = 0.02,
                                   npx_level: float = 0.0):
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

def overlay_fib_npx_signals(ax,
                            price: pd.Series,
                            buy_mask: pd.Series,
                            sell_mask: pd.Series,
                            label_buy: str = "Fibonacci BUY",
                            label_sell: str = "Fibonacci SELL"):
    p = _coerce_1d_series(price)
    bm = _coerce_1d_series(buy_mask).reindex(p.index).fillna(0).astype(bool) if buy_mask is not None else pd.Series(False, index=p.index)
    sm = _coerce_1d_series(sell_mask).reindex(p.index).fillna(0).astype(bool) if sell_mask is not None else pd.Series(False, index=p.index)
    buy_idx = list(bm[bm].index)
    sell_idx = list(sm[sm].index)
    if buy_idx:
        ax.scatter(buy_idx, p.loc[buy_idx], marker="^", s=120, color="tab:green", zorder=11, label=label_buy)
    if sell_idx:
        ax.scatter(sell_idx, p.loc[sell_idx], marker="v", s=120, color="tab:red", zorder=11, label=label_sell)

# =========================
# Indicators: MACD / HMA / BB / NTD / NPX / ROC / RSI
# =========================
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
        if up_idx:
            ax.scatter(up_idx, ntd.loc[up_idx], marker="o", s=40, color="tab:green", zorder=9, label="Price↑NTD")
        if dn_idx:
            ax.scatter(dn_idx, ntd.loc[dn_idx], marker="x", s=60, color="tab:red", zorder=9, label="Price↓NTD")

# =========================
# Ichimoku / ATR / Supertrend / PSAR
# =========================
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = True):
    h = _coerce_1d_series(high)
    l = _coerce_1d_series(low)
    c = _coerce_1d_series(close)
    idx = c.index.union(h.index).union(l.index)
    h = h.reindex(idx); l = l.reindex(idx); c = c.reindex(idx)

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
    return tenkan.reindex(idx), kijun.reindex(idx), senkou_a.reindex(idx), senkou_b.reindex(idx), chikou.reindex(idx)

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
    high = high.reindex(idx); low = low.reindex(idx)

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

# =========================
# Pivots
# =========================
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
# Forex sessions + Yahoo news
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
        except Exception:
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

# =========================
# NTD channel-in-range (hourly panel)
# =========================
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

# =========================
# MACD/HMA/SR combined signal (used on charts + scanners)
# =========================
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
    near_resist  = c >= s_res * (1.0 - prox)

    uptrend = np.isfinite(global_trend_slope) and float(global_trend_slope) > 0
    downtrend = np.isfinite(global_trend_slope) and float(global_trend_slope) < 0

    buy_mask  = uptrend   & (m < 0.0) & cross_up & near_support
    sell_mask = downtrend & (m > 0.0) & cross_dn & near_resist

    last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
    last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None
    if last_buy is None and last_sell is None:
        return None

    if last_sell is None:
        t, side = last_buy, "BUY"
    elif last_buy is None:
        t, side = last_sell, "SELL"
    else:
        t = last_buy if last_buy >= last_sell else last_sell
        side = "BUY" if t == last_buy else "SELL"

    px = float(c.loc[t]) if np.isfinite(c.loc[t]) else np.nan
    return {"time": t, "price": px, "side": side, "note": "MACD/HMA55 + S/R"}

def annotate_macd_signal(ax, ts, px, side: str):
    if side == "BUY":
        ax.scatter([ts], [px], marker="*", s=180, color="tab:green", zorder=10, label="MACD BUY (HMA55+S/R)")
    else:
        ax.scatter([ts], [px], marker="*", s=180, color="tab:red", zorder=10, label="MACD SELL (HMA55+S/R)")

# =========================
# Scanners: cached small computations
# =========================
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
        bar = int(bar)
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

def _series_heading_up(series_like: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(series_like).dropna()
    confirm_bars = max(1, int(confirm_bars))
    if len(s) < confirm_bars + 1:
        return False
    d = s.diff().dropna()
    if len(d) < confirm_bars:
        return False
    return bool(np.all(d.iloc[-confirm_bars:] > 0))

@st.cache_data(ttl=120)
def last_daily_ntd_zero_cross_up_in_uptrend(symbol: str,
                                            daily_view_label: str,
                                            ntd_win: int = 60,
                                            confirm_bars: int = 1):
    """
    NEW tab support:
      - Daily global trendline slope (in chosen daily view) must be UP
      - NTD(win=60) crossed UP through 0.0 recently
      - NTD is heading UP and higher than at cross
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 3:
            return None

        x = np.arange(len(close_show), dtype=float)
        y = close_show.to_numpy(dtype=float)
        m, b = np.polyfit(x, y, 1)
        if not np.isfinite(m) or float(m) <= 0.0:
            return None

        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

        ntd_full = compute_normalized_trend(close_full, window=int(ntd_win))
        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        if ntd_show.dropna().shape[0] < 2:
            return None

        cross_up0 = (ntd_show >= 0.0) & (ntd_show.shift(1) < 0.0)
        cross_up0 = cross_up0.fillna(False)
        if not cross_up0.any():
            return None

        t_cross = cross_up0[cross_up0].index[-1]
        loc = int(close_show.index.get_loc(t_cross))
        bars_since = int((len(close_show) - 1) - loc)

        ntd_cross = float(ntd_show.loc[t_cross]) if np.isfinite(ntd_show.loc[t_cross]) else np.nan
        ntd_last = float(ntd_show.dropna().iloc[-1]) if ntd_show.dropna().shape[0] else np.nan
        if not (np.isfinite(ntd_cross) and np.isfinite(ntd_last)):
            return None
        if float(ntd_last) <= float(ntd_cross):
            return None
        if not _series_heading_up(ntd_show, confirm_bars=int(confirm_bars)):
            return None

        npx_full = compute_normalized_price(close_full, window=int(ntd_win))
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)
        npx_last = float(npx_show.dropna().iloc[-1]) if npx_show.dropna().shape[0] else np.nan

        return {
            "Symbol": symbol,
            "Bars Since Cross": int(bars_since),
            "Cross Time": t_cross,
            "NTD@Cross": float(ntd_cross),
            "NTD (last)": float(ntd_last),
            "NPX (last)": float(npx_last) if np.isfinite(npx_last) else np.nan,
            "Global Slope": float(m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NTD win": int(ntd_win),
            "Confirm Bars": int(confirm_bars),
        }
    except Exception:
        return None

# =========================
# Chart renderers
# =========================
def render_daily_chart(symbol: str, daily_view_label: str):
    close_full = fetch_hist(symbol)
    ohlc = fetch_hist_ohlc(symbol)
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty:
        st.warning("No daily data.")
        return None

    close = subset_by_daily_view(close_full, daily_view_label)
    close = _coerce_1d_series(close).dropna()
    if close.empty:
        st.warning("No daily data in chosen view.")
        return None

    ema30 = close_full.ewm(span=30).mean().reindex(close.index)
    res_d = close_full.rolling(sr_lb_daily, min_periods=1).max().reindex(close.index)
    sup_d = close_full.rolling(sr_lb_daily, min_periods=1).min().reindex(close.index)

    # Local regression band
    yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(close_full, slope_lb_daily)
    yhat_d = yhat_d.reindex(close.index)
    upper_d = upper_d.reindex(close.index)
    lower_d = lower_d.reindex(close.index)

    rev_prob = slope_reversal_probability(close_full, m_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)

    # NTD/NPX
    ntd = compute_normalized_trend(close_full, window=ntd_window).reindex(close.index) if show_ntd else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close_full, window=ntd_window).reindex(close.index) if show_npx_ntd else pd.Series(index=close.index, dtype=float)

    # HMA, MACD (on shown close)
    hma = compute_hma(close, period=hma_period)
    macd, macd_sig, macd_hist = compute_macd(close)

    # BB
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # Kijun
    kijun = pd.Series(index=close.index, dtype=float)
    if show_ichi and (ohlc is not None) and (not ohlc.empty):
        _, kijun_full, _, _, _ = ichimoku_lines(ohlc["High"], ohlc["Low"], ohlc["Close"],
                                               conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
        kijun = _coerce_1d_series(kijun_full).reindex(close.index).ffill().bfill()

    # Pivots
    piv = current_daily_pivots(ohlc)

    # Figure with NTD panel
    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3.2, 1.3]}
    )
    plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.33)

    ax.set_title(
        f"{symbol} Daily — {daily_view_label}  |  P(slope rev≤{rev_horizon})={fmt_pct(rev_prob)}"
    )
    ax.plot(close.index, close.values, label="Price")
    ax.plot(ema30.index, ema30.values, "--", label="30 EMA")

    global_m = draw_trend_direction_line(ax, close, label_prefix="Trend (global)")

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up.dropna().empty and not bb_lo.dropna().empty:
        ax.fill_between(close.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    # S/R
    res_val = float(res_d.iloc[-1]) if len(res_d.dropna()) else np.nan
    sup_val = float(sup_d.iloc[-1]) if len(sup_d.dropna()) else np.nan
    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax.hlines(res_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linewidth=1.6, label=f"R (w={sr_lb_daily})")
        ax.hlines(sup_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linewidth=1.6, label=f"S (w={sr_lb_daily})")
        label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    # Local regression + bands
    if not yhat_d.dropna().empty:
        ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2, label=f"Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
    if not upper_d.dropna().empty and not lower_d.dropna().empty:
        ax.plot(upper_d.index, upper_d.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax.plot(lower_d.index, lower_d.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")
        bounce_sig = find_band_bounce_signal(close, upper_d, lower_d, m_d)
        if bounce_sig is not None:
            annotate_crossover(ax, bounce_sig["time"], bounce_sig["price"], bounce_sig["side"])

    # MACD/HMA/SR star
    macd_sig_obj = find_macd_hma_sr_signal(
        close=close, hma=hma, macd=macd, sup=sup_d, res=res_d,
        global_trend_slope=global_m, prox=sr_prox_pct
    )
    macd_instr = "MACD/HMA55: n/a"
    if macd_sig_obj is not None and np.isfinite(macd_sig_obj.get("price", np.nan)):
        macd_instr = f"MACD/HMA55: {macd_sig_obj['side']} @ {fmt_price_val(macd_sig_obj['price'])}"
        annotate_macd_signal(ax, macd_sig_obj["time"], macd_sig_obj["price"], macd_sig_obj["side"])
    ax.text(0.01, 0.98, macd_instr, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8), zorder=30)

    # Pivots
    if piv:
        for lbl, y in piv.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dashed", linewidth=1.0)
            ax.text(close.index[-1], y, f" {lbl}={fmt_price_val(y)}", va="center")

    # Fibonacci (levels + fib-npx signal markers)
    fibs = {}
    buy_mask = sell_mask = None
    if show_fibs:
        fibs = fibonacci_levels(close)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
            ax.text(close.index[-1], y, f" {lbl}", va="center")
        if show_npx_ntd and not npx.dropna().empty:
            buy_mask, sell_mask, _ = fib_npx_zero_cross_signal_masks(close, npx, horizon_bars=rev_horizon, proximity_pct_of_range=0.02, npx_level=0.0)
            overlay_fib_npx_signals(ax, close, buy_mask, sell_mask)

    # NTD panel
    ax2.set_title(f"Daily Indicator Panel — NTD/NPX (win={ntd_window})")
    if show_ntd and shade_ntd and not ntd.dropna().empty:
        shade_ntd_regions(ax2, ntd)
    if show_ntd and not ntd.dropna().empty:
        ax2.plot(ntd.index, ntd.values, "-", linewidth=1.6, label="NTD")
        ntd_tr, ntd_m = slope_line(ntd, slope_lb_daily)
        if not ntd_tr.dropna().empty:
            ax2.plot(ntd_tr.index, ntd_tr.values, "--", linewidth=2, label=f"NTD trend ({fmt_slope(ntd_m)}/bar)")
    if show_npx_ntd and not npx.dropna().empty and show_ntd and not ntd.dropna().empty:
        overlay_npx_on_ntd(ax2, npx, ntd, mark_crosses=mark_npx_cross)

    ax2.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
    ax2.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
    ax2.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
    ax2.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
    ax2.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
    ax2.set_ylim(-1.1, 1.1)

    # Legend (combined)
    handles, labels = [], []
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h1 + h2
    labels += l1 + l2
    seen = set()
    hu, lu = [], []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        hu.append(h); lu.append(l)
    fig.legend(hu, lu, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4,
               frameon=True, fontsize=9, framealpha=0.65, fancybox=True)

    style_axes(ax); style_axes(ax2)
    st.pyplot(fig)

    last_px = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
    trade_txt = format_trade_instruction(
        trend_slope=m_d,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=last_px,
        symbol=symbol,
        global_trend_slope=global_m
    )

    return {
        "close_full": close_full,
        "close_show": close,
        "ohlc": ohlc,
        "global_slope": global_m,
        "local_slope": m_d,
        "r2": r2_d,
        "rev_prob": rev_prob,
        "trade_instruction": trade_txt,
        "fibs": fibs,
        "fib_buy_mask": buy_mask,
        "fib_sell_mask": sell_mask,
        "last_price": last_px,
    }

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
# bullbear.py — Complete updated Streamlit app (Batch 3/3)
# -------------------------------------------------------

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
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = st.tabs([
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
    "Zero cross"
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

        # Forecast probabilities
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
            # Using yfinance period mapping: interpret bb_period in days
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
# TAB 5 — NTD -0.75 Scanner
# =========================
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Shows symbols where Daily NTD (win=60) is below -0.75 (oversold / strong down normalized trend).")

    max_rows = st.slider("Max rows", 10, 200, 50, 10, key=f"ntdneg_rows_{mode}")
    run5 = st.button("Run NTD -0.75 Scan", key=f"btn_run_ntdneg_{mode}", use_container_width=True)

    if run5:
        rows = []
        for sym in universe:
            s = fetch_hist(sym).dropna()
            if s.empty:
                continue
            s_show = subset_by_daily_view(s, daily_view).dropna()
            ntd = compute_normalized_trend(s, window=60).reindex(s_show.index).dropna()
            if ntd.empty:
                continue
            last_ntd = float(ntd.iloc[-1])
            if last_ntd <= -0.75:
                rows.append({
                    "Symbol": sym,
                    "NTD(last)": last_ntd,
                    "Last Price": float(s_show.iloc[-1]),
                    "AsOf": s_show.index[-1],
                })
        if not rows:
            st.info("No matches.")
        else:
            df = pd.DataFrame(rows).sort_values("NTD(last)")
            st.dataframe(df.head(max_rows).reset_index(drop=True), use_container_width=True)

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
        if df_daily.empty:
            st.write("No matches.")
        else:
            st.dataframe(df_daily.reset_index(drop=True), use_container_width=True)

        rows = []
        for sym in universe:
            r = last_band_bounce_signal_hourly(sym, period_map[hours], slope_lb_hourly)
            if r and r.get("Side") == "BUY" and int(r.get("Bars Since", 9999)) <= int(max_bars):
                rows.append(r)
        df_h = pd.DataFrame(rows).sort_values(["Bars Since", "DeltaPct"], ascending=[True, False]) if rows else pd.DataFrame()
        st.subheader(f"Hourly BUY signals ({hours})")
        if df_h.empty:
            st.write("No matches.")
        else:
            st.dataframe(df_h.reset_index(drop=True), use_container_width=True)

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

            # Most recent signal
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
    st.caption("Shows symbols where price is near +2σ (potential SELL in downtrend) or -2σ (potential BUY in uptrend), with R² info.")

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
# TAB 19 — Zero cross (NEW)
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
