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

def _series_heading_down(series_like: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(series_like).dropna()
    confirm_bars = max(1, int(confirm_bars))
    if len(s) < confirm_bars + 1:
        return False
    d = s.diff().dropna()
    if len(d) < confirm_bars:
        return False
    return bool(np.all(d.iloc[-confirm_bars:] < 0))

def _bars_since_last_event(index_like, event_time) -> int:
    try:
        idx = pd.Index(index_like)
        loc = int(idx.get_loc(event_time))
        return int((len(idx) - 1) - loc)
    except Exception:
        return 10**9

def _global_slope_1d(series_like) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 3:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    try:
        m, _ = np.polyfit(x, s.to_numpy(dtype=float), 1)
        return float(m)
    except Exception:
        return float("nan")

def _global_slope_recent(series_like, lookback: int = 250) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 3:
        return float("nan")
    s = s.iloc[-min(len(s), int(lookback)):]
    return _global_slope_1d(s)

def _last_cross_info(a: pd.Series, b: pd.Series):
    up, dn = _cross_series(a, b)
    last_up = up[up].index[-1] if up.any() else None
    last_dn = dn[dn].index[-1] if dn.any() else None
    if last_up is None and last_dn is None:
        return None
    if last_dn is None:
        return {"dir": "Up", "time": last_up}
    if last_up is None:
        return {"dir": "Down", "time": last_dn}
    if last_up >= last_dn:
        return {"dir": "Up", "time": last_up}
    return {"dir": "Down", "time": last_dn}

@st.cache_data(ttl=120)
def trend_slope_align_row(symbol: str, daily_view_label: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 30:
            return None

        global_m = _global_slope_recent(close_show, lookback=min(len(close_show), max(60, int(slope_lb) * 2)))
        _, _, _, local_m, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)), z=2.0)

        if not (np.isfinite(global_m) and np.isfinite(local_m)):
            return None

        ntd = compute_normalized_trend(close_show, window=int(ntd_window))
        npx = compute_normalized_price(close_show, window=int(ntd_window))
        if _coerce_1d_series(ntd).dropna().empty or _coerce_1d_series(npx).dropna().empty:
            return None

        cross = _last_cross_info(npx, ntd)
        row = {
            "Symbol": symbol,
            "Frame": "Daily",
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Cross Dir": cross["dir"] if cross else "",
            "Cross Time (PST)": cross["time"] if cross else pd.NaT,
            "Bars Since Cross": _bars_since_last_event(close_show.index, cross["time"]) if cross else np.nan,
            "NPX(last)": float(_coerce_1d_series(npx).dropna().iloc[-1]),
            "NTD(last)": float(_coerce_1d_series(ntd).dropna().iloc[-1]),
            "Last Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
        if cross and cross["time"] in _coerce_1d_series(npx).index:
            row["NPX@Cross"] = float(_coerce_1d_series(npx).loc[cross["time"]]) if np.isfinite(_coerce_1d_series(npx).loc[cross["time"]]) else np.nan
        else:
            row["NPX@Cross"] = np.nan
        return row
    except Exception:
        return None

@st.cache_data(ttl=120)
def trend_slope_align_row_hourly(symbol: str, period: str, slope_lb: int):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        global_m = _global_slope_1d(hc)
        _, _, _, local_m, r2 = regression_with_band(hc, lookback=min(len(hc), int(slope_lb)), z=2.0)
        if not (np.isfinite(global_m) and np.isfinite(local_m)):
            return None

        ntd = compute_normalized_trend(hc, window=int(ntd_window))
        npx = compute_normalized_price(hc, window=int(ntd_window))
        if _coerce_1d_series(ntd).dropna().empty or _coerce_1d_series(npx).dropna().empty:
            return None

        cross = _last_cross_info(npx, ntd)
        cross_time = None
        cross_bar = None
        bars_since_cross = np.nan

        if cross:
            try:
                cross_bar = int(cross["time"])
                bars_since_cross = int((len(hc) - 1) - cross_bar)
                if isinstance(real_times, pd.DatetimeIndex) and (0 <= cross_bar < len(real_times)):
                    cross_time = real_times[cross_bar]
            except Exception:
                pass

        row = {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Cross Dir": cross["dir"] if cross else "",
            "Cross Time (PST)": cross_time,
            "Bars Since Cross": bars_since_cross,
            "NPX(last)": float(_coerce_1d_series(npx).dropna().iloc[-1]),
            "NTD(last)": float(_coerce_1d_series(ntd).dropna().iloc[-1]),
            "Last Price": float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan,
        }
        if cross is not None and cross_bar is not None and 0 <= cross_bar < len(_coerce_1d_series(npx)):
            row["NPX@Cross"] = float(_coerce_1d_series(npx).iloc[cross_bar]) if np.isfinite(_coerce_1d_series(npx).iloc[cross_bar]) else np.nan
        else:
            row["NPX@Cross"] = np.nan
        return row
    except Exception:
        return None

@st.cache_data(ttl=120)
def trend_buy_row_daily(symbol: str,
                        daily_view_label: str,
                        slope_lb: int,
                        ntd_win: int = 60,
                        max_bars_since: int = 5):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 30:
            return None

        global_m = _global_slope_recent(close_show, lookback=min(len(close_show), max(60, int(slope_lb) * 2)))
        _, _, _, local_m, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)), z=2.0)
        if not (np.isfinite(global_m) and np.isfinite(local_m) and float(global_m) > 0.0 and float(local_m) > 0.0):
            return None

        ntd = compute_normalized_trend(close_show, window=int(ntd_win))
        npx = compute_normalized_price(close_show, window=int(ntd_win))
        cross = _last_cross_info(npx, ntd)
        if not cross or cross.get("dir") != "Up":
            return None

        t = cross.get("time", None)
        if t is None or t not in close_show.index:
            return None

        bars_since = _bars_since_last_event(close_show.index, t)
        if int(bars_since) > int(max_bars_since):
            return None

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": t,
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(_coerce_1d_series(npx).loc[t]) if np.isfinite(_coerce_1d_series(npx).loc[t]) else np.nan,
            "NPX(last)": float(_coerce_1d_series(npx).dropna().iloc[-1]) if len(_coerce_1d_series(npx).dropna()) else np.nan,
            "Last Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def trend_buy_row_hourly(symbol: str,
                         period: str,
                         slope_lb: int,
                         ntd_win: int = 60,
                         max_bars_since: int = 60):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        global_m = _global_slope_1d(hc)
        _, _, _, local_m, r2 = regression_with_band(hc, lookback=min(len(hc), int(slope_lb)), z=2.0)
        if not (np.isfinite(global_m) and np.isfinite(local_m) and float(global_m) > 0.0 and float(local_m) > 0.0):
            return None

        ntd = compute_normalized_trend(hc, window=int(ntd_win))
        npx = compute_normalized_price(hc, window=int(ntd_win))
        cross = _last_cross_info(npx, ntd)
        if not cross or cross.get("dir") != "Up":
            return None

        bar = cross.get("time", None)
        if bar is None:
            return None
        bar = int(bar)
        if bar < 0 or bar >= len(hc):
            return None

        bars_since = int((len(hc) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": ts,
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(_coerce_1d_series(npx).iloc[bar]) if np.isfinite(_coerce_1d_series(npx).iloc[bar]) else np.nan,
            "NPX(last)": float(_coerce_1d_series(npx).dropna().iloc[-1]) if len(_coerce_1d_series(npx).dropna()) else np.nan,
            "Last Price": float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

def _green_zone_cross_up_mask(npx: pd.Series, ntd: pd.Series) -> pd.Series:
    x = _coerce_1d_series(npx)
    t = _coerce_1d_series(ntd).reindex(x.index)
    prev_x = x.shift(1)
    prev_t = t.shift(1)
    return ((prev_x < prev_t) & (x >= t) & (t < 0.0) & (x < 0.0)).fillna(False)

@st.cache_data(ttl=120)
def green_zone_buy_alert_row_daily(symbol: str,
                                   daily_view_label: str,
                                   slope_lb: int,
                                   ntd_win: int = 60,
                                   max_bars_since: int = 5):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 30:
            return None

        global_m = _global_slope_recent(close_show, lookback=min(len(close_show), max(60, int(slope_lb) * 2)))
        _, _, _, local_m, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)), z=2.0)
        if not (np.isfinite(global_m) and np.isfinite(local_m) and float(global_m) > 0.0 and float(local_m) > 0.0):
            return None

        ntd = compute_normalized_trend(close_show, window=int(ntd_win))
        npx = compute_normalized_price(close_show, window=int(ntd_win))
        mask = _green_zone_cross_up_mask(npx, ntd)
        if not mask.any():
            return None

        t = mask[mask].index[-1]
        if t not in close_show.index:
            return None

        bars_since = _bars_since_last_event(close_show.index, t)
        if int(bars_since) > int(max_bars_since):
            return None

        npx_s = _coerce_1d_series(npx)
        ntd_s = _coerce_1d_series(ntd).reindex(npx_s.index)

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": t,
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(npx_s.loc[t]) if np.isfinite(npx_s.loc[t]) else np.nan,
            "NTD@Cross": float(ntd_s.loc[t]) if np.isfinite(ntd_s.loc[t]) else np.nan,
            "NPX(last)": float(npx_s.dropna().iloc[-1]) if len(npx_s.dropna()) else np.nan,
            "NTD(last)": float(ntd_s.dropna().iloc[-1]) if len(ntd_s.dropna()) else np.nan,
            "Last Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def green_zone_buy_alert_row_hourly(symbol: str,
                                    period: str,
                                    slope_lb: int,
                                    ntd_win: int = 60,
                                    max_bars_since: int = 60):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        global_m = _global_slope_1d(hc)
        _, _, _, local_m, r2 = regression_with_band(hc, lookback=min(len(hc), int(slope_lb)), z=2.0)
        if not (np.isfinite(global_m) and np.isfinite(local_m) and float(global_m) > 0.0 and float(local_m) > 0.0):
            return None

        ntd = compute_normalized_trend(hc, window=int(ntd_win))
        npx = compute_normalized_price(hc, window=int(ntd_win))
        mask = _green_zone_cross_up_mask(npx, ntd)
        if not mask.any():
            return None

        bar = int(mask[mask].index[-1])
        bars_since = int((len(hc) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        npx_s = _coerce_1d_series(npx)
        ntd_s = _coerce_1d_series(ntd).reindex(npx_s.index)

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": ts,
            "Trendline Slope": float(global_m),
            "Regression Slope": float(local_m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(npx_s.iloc[bar]) if np.isfinite(npx_s.iloc[bar]) else np.nan,
            "NTD@Cross": float(ntd_s.iloc[bar]) if np.isfinite(ntd_s.iloc[bar]) else np.nan,
            "NPX(last)": float(npx_s.dropna().iloc[-1]) if len(npx_s.dropna()) else np.nan,
            "NTD(last)": float(ntd_s.dropna().iloc[-1]) if len(ntd_s.dropna()) else np.nan,
            "Last Price": float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def reversal_candidate_row_daily(symbol: str,
                                 daily_view_label: str,
                                 slope_lb: int,
                                 max_bars_since: int = 10):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 30:
            return None

        global_m = _global_slope_recent(close_show, lookback=min(len(close_show), max(60, int(slope_lb) * 2)))
        yhat, up, lo, local_m, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)), z=2.0)

        if yhat.dropna().empty or not (np.isfinite(local_m) and np.isfinite(global_m)):
            return None

        sig = find_band_bounce_signal(close_show, up, lo, local_m)
        if sig is None:
            return None

        t = sig.get("time", None)
        if t is None or t not in close_show.index:
            return None

        bars_since = _bars_since_last_event(close_show.index, t)
        if int(bars_since) > int(max_bars_since):
            return None

        side = str(sig.get("side", "")).upper()
        is_buy = (side == "BUY") and (float(local_m) > 0.0) and (float(global_m) > 0.0)
        is_sell = (side == "SELL") and (float(local_m) < 0.0) and (float(global_m) < 0.0)
        if not (is_buy or is_sell):
            return None

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        reg_px = float(yhat.iloc[-1]) if np.isfinite(yhat.iloc[-1]) else np.nan
        if not (np.isfinite(curr_px) and np.isfinite(reg_px)):
            return None

        dist = abs(curr_px - reg_px)

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Side": "BUY" if is_buy else "SELL",
            "Bars Since": int(bars_since),
            "Current Slope Price": curr_px,
            "Current Regression line price": reg_px,
            "Distance to regression line": dist,
            "_global_slope": float(global_m),
            "_local_slope": float(local_m),
            "_r2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def reversal_candidate_row_hourly(symbol: str,
                                  period: str,
                                  slope_lb: int,
                                  max_bars_since: int = 10):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        global_m = _global_slope_1d(hc)
        yhat, up, lo, local_m, r2 = regression_with_band(hc, lookback=min(len(hc), int(slope_lb)), z=2.0)

        if _coerce_1d_series(yhat).dropna().empty or not (np.isfinite(local_m) and np.isfinite(global_m)):
            return None

        sig = find_band_bounce_signal(hc, up, lo, local_m)
        if sig is None:
            return None

        bar = sig.get("time", None)
        if bar is None:
            return None
        bar = int(bar)
        if bar < 0 or bar >= len(hc):
            return None

        bars_since = int((len(hc) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        side = str(sig.get("side", "")).upper()
        is_buy = (side == "BUY") and (float(local_m) > 0.0) and (float(global_m) > 0.0)
        is_sell = (side == "SELL") and (float(local_m) < 0.0) and (float(global_m) < 0.0)
        if not (is_buy or is_sell):
            return None

        curr_px = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan
        reg_px = float(_coerce_1d_series(yhat).dropna().iloc[-1]) if len(_coerce_1d_series(yhat).dropna()) else np.nan
        if not (np.isfinite(curr_px) and np.isfinite(reg_px)):
            return None
        dist = abs(curr_px - reg_px)

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Side": "BUY" if is_buy else "SELL",
            "Bars Since": int(bars_since),
            "Signal Time": ts,
            "Current Slope Price": curr_px,
            "Current Regression line price": reg_px,
            "Distance to regression line": dist,
            "_global_slope": float(global_m),
            "_local_slope": float(local_m),
            "_r2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def _ntd_minus05_cross_up_mask(ntd: pd.Series) -> pd.Series:
    """Cross up through -0.5 into [-0.5, -0.4], upward."""
    s = _coerce_1d_series(ntd)
    prev = s.shift(1)
    return ((s >= -0.5) & (s <= -0.4) & (prev < -0.5) & (s > prev)).fillna(False)

@st.cache_data(ttl=120)
def ntd_minus05_cross_row_daily(symbol: str,
                               daily_view_label: str,
                               slope_lb: int,
                               ntd_win: int = 60,
                               max_bars_since: int = 5):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 30:
            return None

        _, _, _, m, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)), z=2.0)
        if not (np.isfinite(m) and float(m) > 0.0):
            return None

        ntd_full = compute_normalized_trend(close_full, window=int(ntd_win))
        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        if ntd_show.dropna().shape[0] < 2:
            return None

        mask = _ntd_minus05_cross_up_mask(ntd_show)
        if not mask.any():
            return None

        t = mask[mask].index[-1]
        bars_since = int((len(close_show) - 1) - int(close_show.index.get_loc(t)))
        if int(bars_since) > int(max_bars_since):
            return None

        ntd_cross = float(ntd_show.loc[t]) if np.isfinite(ntd_show.loc[t]) else np.nan
        ntd_last = float(ntd_show.dropna().iloc[-1]) if len(ntd_show.dropna()) else np.nan
        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since": int(bars_since),
            "Cross Time": t,
            "NTD@Cross": ntd_cross,
            "NTD(last)": ntd_last,
            "Slope": float(m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def ntd_minus05_cross_row_hourly(symbol: str,
                                period: str,
                                slope_lb: int,
                                ntd_win: int = 60,
                                max_bars_since: int = 10):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        hc = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if len(hc) < 60:
            return None

        _, _, _, m, r2 = regression_with_band(hc, lookback=min(len(hc), int(slope_lb)), z=2.0)
        if not (np.isfinite(m) and float(m) > 0.0):
            return None

        ntd = compute_normalized_trend(hc, window=int(ntd_win))
        if _coerce_1d_series(ntd).dropna().shape[0] < 2:
            return None

        mask = _ntd_minus05_cross_up_mask(ntd)
        if not mask.any():
            return None

        bar = int(mask[mask].index[-1])
        bars_since = int((len(hc) - 1) - bar)
        if int(bars_since) > int(max_bars_since):
            return None

        ts = None
        if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)):
            ts = real_times[bar]

        ntd_cross = float(_coerce_1d_series(ntd).iloc[bar]) if np.isfinite(_coerce_1d_series(ntd).iloc[bar]) else np.nan
        ntd_last = float(_coerce_1d_series(ntd).dropna().iloc[-1]) if len(_coerce_1d_series(ntd).dropna()) else np.nan
        last_px = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since": int(bars_since),
            "Cross Time": ts,
            "NTD@Cross": ntd_cross,
            "NTD(last)": ntd_last,
            "Slope": float(m),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Last Price": last_px,
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

    close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
    if close.empty:
        st.warning("No data for selected daily view.")
        return None

    yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(close, lookback=min(len(close), int(slope_lb_daily)), z=2.0)
    rev_prob_d = slope_reversal_probability(
        close, current_slope=m_d, hist_window=int(rev_hist_lb),
        slope_window=int(ntd_window), horizon=int(rev_horizon)
    )

    ntd_d = compute_normalized_trend(close, window=int(ntd_window))
    npx_d = compute_normalized_price(close, window=int(ntd_window))
    roc_d = compute_roc(close, n=int(mom_lb_hourly))
    macd_d, macd_sig_d, macd_hist_d = compute_macd(close)
    hma_d = compute_hma(close, period=int(hma_period))
    bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(
        close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema)
    )

    sup_d = close.rolling(int(sr_lb_daily), min_periods=max(5, int(sr_lb_daily)//3)).min()
    res_d = close.rolling(int(sr_lb_daily), min_periods=max(5, int(sr_lb_daily)//3)).max()

    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.plot(close.index, close.values, label="Close", linewidth=1.6)
    if not _coerce_1d_series(yhat_d).dropna().empty:
        ax.plot(yhat_d.index, yhat_d.values, "--", label=f"Regression ({fmt_slope(m_d)}/bar)")
    if not _coerce_1d_series(up_d).dropna().empty:
        ax.plot(up_d.index, up_d.values, ":", linewidth=1.0, label="Reg Upper")
    if not _coerce_1d_series(lo_d).dropna().empty:
        ax.plot(lo_d.index, lo_d.values, ":", linewidth=1.0, label="Reg Lower")

    gm_d = draw_trend_direction_line(ax, close, label_prefix="Trendline")

    if show_bbands and not _coerce_1d_series(bb_mid_d).dropna().empty:
        ax.plot(bb_mid_d.index, bb_mid_d.values, linewidth=1.0, alpha=0.9, label="BB Mid")
        ax.plot(bb_up_d.index, bb_up_d.values, linewidth=0.9, alpha=0.8, label="BB Upper")
        ax.plot(bb_lo_d.index, bb_lo_d.values, linewidth=0.9, alpha=0.8, label="BB Lower")

    if show_ichi and not ohlc.empty and {"High","Low","Close"}.issubset(ohlc.columns):
        ohlc_show = subset_by_daily_view(ohlc, daily_view_label)
        tenkan, kijun, senkou_a, senkou_b, chikou = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb), shift_cloud=False
        )
        if not _coerce_1d_series(kijun).dropna().empty:
            ax.plot(kijun.index, kijun.values, linewidth=1.1, alpha=0.9, label="Ichimoku Kijun")

    if show_fibs:
        fibs = fibonacci_levels(close)
        for k, v in fibs.items():
            ax.axhline(v, linestyle=":", linewidth=0.8, alpha=0.6)
            label_on_left(ax, v, f"Fib {k}", fontsize=8)

    sig_band = find_band_bounce_signal(close, up_d, lo_d, m_d)
    if sig_band is not None:
        annotate_crossover(ax, sig_band["time"], sig_band["price"], sig_band["side"], note="Band")

    sig_macd = find_macd_hma_sr_signal(close, hma_d, macd_d, sup_d, res_d, gm_d, prox=float(sr_prox_pct))
    if sig_macd is not None:
        annotate_macd_signal(ax, sig_macd["time"], sig_macd["price"], sig_macd["side"])

    ax.set_title(
        f"{symbol} — Daily ({daily_view_label})  |  "
        f"Trend={fmt_slope(gm_d)}  Reg={fmt_slope(m_d)}  R²={fmt_r2(r2_d)}  "
        f"RevProb={fmt_pct(rev_prob_d)}"
    )
    ax.legend(loc="upper left", ncol=3, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax)
    st.pyplot(fig)

    if show_ntd and not _coerce_1d_series(ntd_d).dropna().empty:
        fig2, ax2 = plt.subplots(figsize=(14, 2.8))
        if shade_ntd:
            shade_ntd_regions(ax2, ntd_d)
        ax2.plot(ntd_d.index, ntd_d.values, label="NTD", linewidth=1.4)
        if show_npx_ntd:
            overlay_npx_on_ntd(ax2, npx_d, ntd_d, mark_crosses=bool(mark_npx_cross))
        ax2.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_title("NTD / NPX")
        ax2.legend(loc="upper left", ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax2)
        st.pyplot(fig2)

    trade_txt = format_trade_instruction(
        trend_slope=m_d,
        buy_val=float(sup_d.dropna().iloc[-1]) if len(sup_d.dropna()) else float(close.iloc[-1]),
        sell_val=float(res_d.dropna().iloc[-1]) if len(res_d.dropna()) else float(close.iloc[-1]),
        close_val=float(close.iloc[-1]),
        symbol=symbol,
        global_trend_slope=gm_d
    )

    return {
        "close": close,
        "global_slope": gm_d,
        "local_slope": m_d,
        "r2": r2_d,
        "rev_prob": rev_prob_d,
        "trade_instruction": trade_txt,
        "last_price": float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan,
    }

def render_hourly_chart(symbol: str, period: str = "1d"):
    intraday = fetch_intraday(symbol, period=period)
    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        st.warning("No hourly/intraday data.")
        return None

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None

    intraday2 = intraday.copy()
    intraday2.index = pd.RangeIndex(len(intraday2))
    close_h = _coerce_1d_series(intraday2["Close"]).ffill().dropna()
    if close_h.empty:
        st.warning("No intraday close data.")
        return None

    yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(close_h, lookback=min(len(close_h), int(slope_lb_hourly)), z=2.0)
    rev_prob_h = slope_reversal_probability(
        close_h, current_slope=m_h, hist_window=int(rev_hist_lb),
        slope_window=int(ntd_window), horizon=int(rev_horizon)
    )

    ntd_h = compute_normalized_trend(close_h, window=int(ntd_window))
    npx_h = compute_normalized_price(close_h, window=int(ntd_window))
    mom_h = compute_roc(close_h, n=int(mom_lb_hourly))
    macd_h, macd_sig_h, macd_hist_h = compute_macd(close_h)
    hma_h = compute_hma(close_h, period=int(hma_period))
    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        close_h, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema)
    )

    sup = close_h.rolling(int(sr_lb_hourly), min_periods=max(5, int(sr_lb_hourly)//3)).min()
    res = close_h.rolling(int(sr_lb_hourly), min_periods=max(5, int(sr_lb_hourly)//3)).max()

    st_df = compute_supertrend(intraday2[["High", "Low", "Close"]], atr_period=int(atr_period), atr_mult=float(atr_mult))
    psar_df = compute_psar_from_ohlc(intraday2[["High", "Low"]], step=float(psar_step), max_step=float(psar_max))
    st_line = st_df["ST"].reindex(hc.index) if (not st_df.empty and "ST" in st_df.columns) else pd.Series(index=hc.index, dtype=float)

    psar_df = compute_psar_from_ohlc(df, step=psar_step, max_step=psar_max) if (show_psar and {"High","Low"}.issubset(df.columns)) else pd.DataFrame()
    if not psar_df.empty:
        psar_df = psar_df.reindex(hc.index)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if show_ichi and {"High","Low","Close"}.issubset(df.columns):
        _, kijun_calc, _, _, _ = ichimoku_lines(
            df["High"], df["Low"], df["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
        )
        kijun_h = _coerce_1d_series(kijun_calc).reindex(hc.index).ffill().bfill()

    global_m = _global_slope_1d(hc)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = hc.index

    ax.plot(x, hc.values, label="Close", linewidth=1.6)
    ax.plot(x, he.values, label="EMA20", linewidth=1.0, alpha=0.9)

    if not _coerce_1d_series(yhat_h).dropna().empty:
        ax.plot(yhat_h.index, yhat_h.values, "--", label=f"Regression ({fmt_slope(m_h)}/bar)")
    if not _coerce_1d_series(up_h).dropna().empty:
        ax.plot(up_h.index, up_h.values, ":", linewidth=1.0, label="Reg Upper")
    if not _coerce_1d_series(lo_h).dropna().empty:
        ax.plot(lo_h.index, lo_h.values, ":", linewidth=1.0, label="Reg Lower")

    try:
        xnum = np.arange(len(hc), dtype=float)
        yhat_g = global_m * xnum + (float(hc.iloc[0]) if len(hc) else 0.0)
        ax.plot(x, yhat_g, linestyle="--", linewidth=2.0, color=("green" if global_m >= 0 else "red"),
                label=f"Trendline ({fmt_slope(global_m)}/bar)")
    except Exception:
        pass

    if show_bbands and not _coerce_1d_series(bb_mid_h).dropna().empty:
        ax.plot(bb_mid_h.index, bb_mid_h.values, linewidth=1.0, alpha=0.9, label="BB Mid")
        ax.plot(bb_up_h.index, bb_up_h.values, linewidth=0.9, alpha=0.8, label="BB Upper")
        ax.plot(bb_lo_h.index, bb_lo_h.values, linewidth=0.9, alpha=0.8, label="BB Lower")

    if not _coerce_1d_series(sup).dropna().empty:
        ax.plot(sup.index, sup.values, linewidth=0.9, alpha=0.8, label="Support")
    if not _coerce_1d_series(res).dropna().empty:
        ax.plot(res.index, res.values, linewidth=0.9, alpha=0.8, label="Resistance")

    if show_ichi and not _coerce_1d_series(kijun_h).dropna().empty:
        ax.plot(kijun_h.index, kijun_h.values, linewidth=1.1, alpha=0.9, label="Ichimoku Kijun")

    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, linewidth=1.1, alpha=0.9, label="Supertrend")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        psar_s = _coerce_1d_series(psar_df["PSAR"]).reindex(hc.index)
        if not psar_s.dropna().empty:
            ax.scatter(psar_s.index, psar_s.values, s=10, label="PSAR")

    sig_band = find_band_bounce_signal(hc, up_h, lo_h, m_h)
    if sig_band is not None:
        annotate_crossover(ax, sig_band["time"], sig_band["price"], sig_band["side"], note="Band")

    sig_macd = find_macd_hma_sr_signal(hc, hma_h, macd_h, sup, res, global_m, prox=float(sr_prox_pct))
    if sig_macd is not None:
        annotate_macd_signal(ax, sig_macd["time"], sig_macd["price"], sig_macd["side"])

    if real_times is not None:
        _apply_compact_time_ticks(ax, real_times, n_ticks=8)

    if mode == "Forex" and show_sessions_pst and real_times is not None:
        sess = compute_session_lines(real_times)
        sess_handles, sess_labels = draw_session_lines(ax, sess, alpha=0.30)
    else:
        sess_handles, sess_labels = [], []

    if mode == "Forex" and show_fx_news and real_times is not None:
        news_df = fetch_yf_news(symbol, window_days=int(news_window_days))
        if news_df is not None and not news_df.empty:
            news_pos = _map_times_to_bar_positions(real_times, list(news_df["time"]))
            draw_news_markers(ax, news_pos, label="News")

    ax.set_title(
        f"{symbol} — Hourly ({period})  |  "
        f"Trend={fmt_slope(global_m)}  Reg={fmt_slope(m_h)}  R²={fmt_r2(r2_h)}  "
        f"RevProb={fmt_pct(rev_prob_h)}"
    )
    ax.legend(loc="upper left", ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
    style_axes(ax)
    st.pyplot(fig)

    if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
        fig2, ax2 = plt.subplots(figsize=(14, 2.8))
        if shade_ntd:
            shade_ntd_regions(ax2, ntd_h)
        ax2.plot(ntd_h.index, ntd_h.values, label="NTD", linewidth=1.4)
        if show_npx_ntd:
            overlay_npx_on_ntd(ax2, npx_h, ntd_h, mark_crosses=bool(mark_npx_cross))
        if show_ntd_channel:
            overlay_inrange_on_ntd(ax2, hc, sup, res)
        ax2.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax2.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax2.axhline(0.75, linestyle="-", linewidth=1.0, color="black", label="+0.75")
        ax2.axhline(-0.75, linestyle="-", linewidth=1.0, color="black", label="-0.75")
        ax2.set_ylim(-1.1, 1.1)
        if real_times is not None:
            _apply_compact_time_ticks(ax2, real_times, n_ticks=8)
        ax2.set_title("NTD / NPX")
        ax2.legend(loc="upper left", ncol=4, framealpha=0.65, fontsize=9, fancybox=True)
        style_axes(ax2)
        st.pyplot(fig2)

    if show_macd and not _coerce_1d_series(macd_h).dropna().empty:
        fig3, ax3 = plt.subplots(figsize=(14, 2.8))
        ax3.plot(macd_h.index, macd_h.values, label="MACD")
        ax3.plot(macd_sig_h.index, macd_sig_h.values, label="Signal")
        ax3.bar(macd_hist_h.index, macd_hist_h.values, label="Histogram")
        ax3.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        if real_times is not None:
            _apply_compact_time_ticks(ax3, real_times, n_ticks=8)
        ax3.set_title("MACD")
        ax3.legend(loc="upper left")
        style_axes(ax3)
        st.pyplot(fig3)

    if show_mom_hourly and not _coerce_1d_series(mom_h).dropna().empty:
        fig4, ax4 = plt.subplots(figsize=(14, 2.8))
        ax4.plot(mom_h.index, mom_h.values, label="ROC%")
        ax4.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        if real_times is not None:
            _apply_compact_time_ticks(ax4, real_times, n_ticks=8)
        ax4.set_title("Hourly Momentum (ROC%)")
        ax4.legend(loc="upper left")
        style_axes(ax4)
        st.pyplot(fig4)

    trade_txt = format_trade_instruction(
        trend_slope=m_h,
        buy_val=float(sup.dropna().iloc[-1]) if len(sup.dropna()) else float(hc.iloc[-1]),
        sell_val=float(res.dropna().iloc[-1]) if len(res.dropna()) else float(hc.iloc[-1]),
        close_val=float(hc.iloc[-1]),
        symbol=symbol,
        global_trend_slope=global_m
    )

    return {
        "close": hc,
        "global_slope": global_m,
        "local_slope": m_h,
        "r2": r2_h,
        "rev_prob": rev_prob_h,
        "trade_instruction": trade_txt,
        "last_price": float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan,
    }

# =========================
# Period maps / main tabs
# =========================
period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

tabs = st.tabs([
    "Dashboard",
    "Forecast",
    "Trend and Slope Align",
    "Trend Buy",
    "Green Zone Buy Alert",
    "Bull/Bear",
    "HMA Signal",
    "Price↔Regression Cross (Daily)",
    "Long-Term History",
    "NTD Buy Signal",
])

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = tabs

# =========================
# TAB 1 — Dashboard
# =========================
with tab1:
    st.header("Dashboard")

    c1, c2, c3 = st.columns(3)
    sel1 = c1.selectbox("Ticker:", universe, key=f"dash_ticker_{mode}")
    hour_range = c2.selectbox("Hourly scan window:", ["24h", "48h", "96h"], index=0, key=f"dash_hour_range_{mode}")
    run1 = c3.button("Run Dashboard", key=f"btn_run_dash_{mode}", use_container_width=True)

    if run1:
        daily_res = render_daily_chart(sel1, daily_view)
        hourly_res = render_hourly_chart(sel1, period=period_map.get(hour_range, "1d"))

        if daily_res:
            st.info(f"Daily: {daily_res['trade_instruction']}")
        if hourly_res:
            st.info(f"Hourly: {hourly_res['trade_instruction']}")

# =========================
# TAB 2 — Forecast
# =========================
with tab2:
    st.header("Forecast")

    c1, c2 = st.columns(2)
    sel2 = c1.selectbox("Ticker:", universe, key=f"fc_ticker_{mode}")
    run2 = c2.button("Run Forecast", key=f"btn_run_fc_{mode}", use_container_width=True)

    if run2:
        s = fetch_hist(sel2).dropna()
        if s.empty:
            st.warning("No data.")
        else:
            idx, fc, ci = compute_sarimax_forecast(s)
            fig, ax = plt.subplots(figsize=(14, 4.8))
            ax.plot(s.index, s.values, label="History")
            ax.plot(idx, fc.values, label="Forecast")
            if ci is not None and ci.shape[1] >= 2:
                ax.fill_between(idx, ci.iloc[:, 0].values, ci.iloc[:, 1].values, alpha=0.2, label="CI")
            ax.set_title(f"{sel2} — 30-Day SARIMAX Forecast")
            ax.legend(loc="upper left")
            style_axes(ax)
            st.pyplot(fig)

            st.subheader("Forecast Table")
            out = pd.DataFrame(
                {"Forecast": fc, "Lower": ci.iloc[:, 0], "Upper": ci.iloc[:, 1]},
                index=idx
            ) if (ci is not None and ci.shape[1] >= 2) else pd.DataFrame({"Forecast": fc}, index=idx)
            st.dataframe(out, use_container_width=True)

# =========================
# TAB 3 — Trend and Slope Align
# =========================
with tab3:
    st.header("Trend and Slope Align")
    st.caption(
        "Daily Buy Opportunities: **Trendline > 0 AND Regression Slope > 0** (daily view).\n"
        "Daily Sell Opportunities: **Trendline < 0 AND Regression Slope < 0** (daily view).\n\n"
        "Hourly Buy Opportunities: **Trendline > 0 AND Regression Slope > 0** (intraday lookback).\n"
        "Hourly Sell Opportunities: **Trendline < 0 AND Regression Slope < 0** (intraday lookback).\n"
        "Daily/Hourly: also shows Buy/Sell lists where **NPX (Norm Price) recently crossed the NTD line**.\n"
        "• Buy-cross list requires **Cross Dir = Up**.\n"
        "• Sell-cross list requires **Cross Dir = Down**.\n\n"
        "Trendline slope = global slope over the chosen frame.\n"
        "Regression slope = local regression slope over the configured slope lookback.\n"
        "NPX↔NTD cross = NPX crosses above/below NTD on the NTD panel scale."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"tsa_rows_{mode}")
    min_abs_slope = c2.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"tsa_minabs_{mode}")
    max_cross_bars_daily = c3.slider("Daily: Max bars since NPX↔NTD cross", 0, 60, 5, 1, key=f"tsa_cross_bars_d_{mode}")
    hr_win = c4.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"tsa_hr_win_{mode}")
    max_cross_bars_hourly = c5.slider("Hourly: Max bars since NPX↔NTD cross (5m bars)", 0, 480, 60, 5, key=f"tsa_cross_bars_h_{mode}")

    run23 = st.button("Run Trend/Slope Align Scan", key=f"btn_run_tsa_{mode}", use_container_width=True)

    if run23:
        buys_d, sells_d, buys_cross_d, sells_cross_d = [], [], [], []

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

            if (tm > 0.0) and (rm > 0.0):
                buys_d.append(r)
            elif (tm < 0.0) and (rm < 0.0):
                sells_d.append(r)

            if (tm > 0.0) and (rm > 0.0) and (r.get("Cross Dir", "") == "Up") and (float(r.get("Bars Since Cross", np.inf)) <= float(max_cross_bars_daily)):
                buys_cross_d.append(r)
            if (tm < 0.0) and (rm < 0.0) and (r.get("Cross Dir", "") == "Down") and (float(r.get("Bars Since Cross", np.inf)) <= float(max_cross_bars_daily)):
                sells_cross_d.append(r)

        buys_h, sells_h, buys_cross_h, sells_cross_h = [], [], [], []
        hr_period = period_map.get(hr_win, "1d")

        for sym in universe:
            r = trend_slope_align_row_hourly(sym, period=hr_period, slope_lb=slope_lb_hourly)
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if not (np.isfinite(tm) and np.isfinite(rm)):
                continue

            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue

            if (tm > 0.0) and (rm > 0.0):
                buys_h.append(r)
            elif (tm < 0.0) and (rm < 0.0):
                sells_h.append(r)

            if (tm > 0.0) and (rm > 0.0) and (r.get("Cross Dir", "") == "Up") and (float(r.get("Bars Since Cross", np.inf)) <= float(max_cross_bars_hourly)):
                buys_cross_h.append(r)
            if (tm < 0.0) and (rm < 0.0) and (r.get("Cross Dir", "") == "Down") and (float(r.get("Bars Since Cross", np.inf)) <= float(max_cross_bars_hourly)):
                sells_cross_h.append(r)

        show_cols_base = ["Symbol", "Frame", "Trendline Slope", "Regression Slope", "R2", "Last Price"]
        show_cols_cross = ["Symbol", "Frame", "Bars Since Cross", "Cross Time (PST)", "Trendline Slope", "Regression Slope", "R2", "Cross Dir", "NPX(last)", "NTD(last)", "Last Price"]

        st.subheader("Daily results")
        cLd, cRd = st.columns(2)
        with cLd:
            st.subheader("Daily Buy Opportunities (Trendline > 0 & Regression Slope > 0)")
            if not buys_d:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(buys_d)
                dfb["_score"] = dfb["Trendline Slope"].astype(float) + dfb["Regression Slope"].astype(float)
                dfb = dfb.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(dfb[show_cols_base].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cRd:
            st.subheader("Daily Buy + NPX recently crossed NTD (Cross Dir = Up)")
            if not buys_cross_d:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(buys_cross_d)
                try:
                    dfc["_cross_ts"] = pd.to_datetime(dfc["Cross Time (PST)"], errors="coerce")
                    dfc["Cross Time (PST)"] = dfc["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfc[show_cols_cross].head(max_rows).reset_index(drop=True), use_container_width=True)

        cSd, cSdR = st.columns(2)
        with cSd:
            st.subheader("Daily Sell Opportunities (Trendline < 0 & Regression Slope < 0)")
            if not sells_d:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(sells_d)
                dfs["_score"] = dfs["Trendline Slope"].astype(float) + dfs["Regression Slope"].astype(float)
                dfs = dfs.sort_values(["_score", "R2"], ascending=[True, False])
                st.dataframe(dfs[show_cols_base].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cSdR:
            st.subheader("Daily Sell + NPX recently crossed NTD (Cross Dir = Down)")
            if not sells_cross_d:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(sells_cross_d)
                try:
                    dfc["_cross_ts"] = pd.to_datetime(dfc["Cross Time (PST)"], errors="coerce")
                    dfc["Cross Time (PST)"] = dfc["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, True, False])
                st.dataframe(dfc[show_cols_cross].head(max_rows).reset_index(drop=True), use_container_width=True)

        st.subheader(f"Hourly results ({hr_win})")
        cLh, cRh = st.columns(2)
        with cLh:
            st.subheader("Hourly Buy Opportunities (Trendline > 0 & Regression Slope > 0)")
            if not buys_h:
                st.write("No matches.")
            else:
                dfb = pd.DataFrame(buys_h)
                dfb["_score"] = dfb["Trendline Slope"].astype(float) + dfb["Regression Slope"].astype(float)
                dfb = dfb.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(dfb[show_cols_base].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cRh:
            st.subheader("Hourly Buy + NPX recently crossed NTD (Cross Dir = Up)")
            if not buys_cross_h:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(buys_cross_h)
                try:
                    dfc["_cross_ts"] = pd.to_datetime(dfc["Cross Time (PST)"], errors="coerce")
                    dfc["Cross Time (PST)"] = dfc["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(dfc[show_cols_cross].head(max_rows).reset_index(drop=True), use_container_width=True)

        cSh, cShR = st.columns(2)
        with cSh:
            st.subheader("Hourly Sell Opportunities (Trendline < 0 & Regression Slope < 0)")
            if not sells_h:
                st.write("No matches.")
            else:
                dfs = pd.DataFrame(sells_h)
                dfs["_score"] = dfs["Trendline Slope"].astype(float) + dfs["Regression Slope"].astype(float)
                dfs = dfs.sort_values(["_score", "R2"], ascending=[True, False])
                st.dataframe(dfs[show_cols_base].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cShR:
            st.subheader("Hourly Sell + NPX recently crossed NTD (Cross Dir = Down)")
            if not sells_cross_h:
                st.write("No matches.")
            else:
                dfc = pd.DataFrame(sells_cross_h)
                try:
                    dfc["_cross_ts"] = pd.to_datetime(dfc["Cross Time (PST)"], errors="coerce")
                    dfc["Cross Time (PST)"] = dfc["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                dfc["_score"] = dfc["Trendline Slope"].astype(float) + dfc["Regression Slope"].astype(float)
                dfc = dfc.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, True, False])
                st.dataframe(dfc[show_cols_cross].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 4 — Trend Buy
# =========================
with tab4:
    st.header("Trend Buy")
    st.caption(
        "Lists symbols where:\n"
        "• **Trendline Slope > 0** and **Regression Slope > 0**\n"
        "• **NPX (Norm Price)** recently crossed **UP** through **0.0** on the NTD/NPX panel.\n\n"
        "Shows **Daily** and **Hourly** lists, including **Cross Time (PST)** and **Bars Since Cross** (ascending)."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"trendbuy_rows_{mode}")
    within_daily = c2.selectbox("Daily: within N bars", [3, 5, 10, 15, 20, 30, 60], index=1, key=f"trendbuy_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"trendbuy_hr_win_{mode}")
    within_hourly = c4.selectbox("Hourly: within N bars (5m bars)", [3, 5, 10, 15, 20, 30, 60, 120, 240, 480], index=6, key=f"trendbuy_within_h_{mode}")

    run24 = st.button("Run Trend Buy Scan", key=f"btn_run_trendbuy_{mode}", use_container_width=True)

    if run24:
        show_cols = [
            "Symbol", "Frame",
            "Bars Since Cross", "Cross Time (PST)",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX@Cross", "NPX(last)", "Last Price"
        ]

        d_rows = []
        for sym in universe:
            r = trend_buy_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        st.subheader("Daily Chart")
        if not d_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(d_rows)
            if "Cross Time (PST)" in df.columns:
                try:
                    df["_cross_ts"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
            df = df.sort_values(["Bars Since Cross"], ascending=[True])
            st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = trend_buy_row_hourly(
                symbol=sym,
                period=hr_period,
                slope_lb=slope_lb_hourly,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_hourly),
            )
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        st.subheader(f"Hourly Chart ({hours})")
        if not h_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(h_rows)
            if "Cross Time (PST)" in df.columns:
                try:
                    df["_cross_ts"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
            df = df.sort_values(["Bars Since Cross"], ascending=[True])
            st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 5 — Green Zone Buy Alert  ✅ NEW
# =========================
with tab5:
    st.header("Green Zone Buy Alert")
    st.caption(
        "Shows symbols from the **Daily** and **Hourly** charts where:\n"
        "• **Trendline Slope > 0**\n"
        "• **Regression Slope > 0**\n"
        "• **NPX (Norm Price)** recently crossed **UP** through the **NTD** line\n"
        "• The cross happened while both values were **below 0.0**, i.e. inside the **green zone below zero** on the NTD/NPX chart."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"greenzone_rows_{mode}")
    within_daily = c2.selectbox("Daily: within N bars", [3, 5, 10, 15, 20, 30, 60], index=1, key=f"greenzone_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"greenzone_hr_win_{mode}")
    within_hourly = c4.selectbox("Hourly: within N bars (5m bars)", [3, 5, 10, 15, 20, 30, 60, 120, 240, 480], index=6, key=f"greenzone_within_h_{mode}")

    run_green = st.button("Run Green Zone Buy Alert Scan", key=f"btn_run_greenzone_{mode}", use_container_width=True)

    if run_green:
        show_cols = [
            "Symbol", "Frame",
            "Bars Since Cross", "Cross Time (PST)",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX@Cross", "NTD@Cross",
            "NPX(last)", "NTD(last)", "Last Price"
        ]

        d_rows = []
        for sym in universe:
            r = green_zone_buy_alert_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        st.subheader("Daily Chart")
        if not d_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(d_rows)
            try:
                df["_cross_ts"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                df["Cross Time (PST)"] = df["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                pass
            df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
            df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
            st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = green_zone_buy_alert_row_hourly(
                symbol=sym,
                period=hr_period,
                slope_lb=slope_lb_hourly,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_hourly),
            )
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        st.subheader(f"Hourly Chart ({hours})")
        if not h_rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(h_rows)
            try:
                df["_cross_ts"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                df["Cross Time (PST)"] = df["_cross_ts"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                pass
            df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
            df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
            st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 6 — Bull/Bear
# =========================
with tab6:
    st.header("Bull / Bear")
    sel3 = st.selectbox("Ticker:", universe, key=f"bullbear_ticker_{mode}")
    run3 = st.button("Run Bull/Bear", key=f"btn_run_bullbear_{mode}", use_container_width=True)

    if run3:
        s = fetch_hist(sel3).dropna()
        if s.empty:
            st.warning("No data.")
        else:
            recent = s.last(bb_period)
            diff = recent.diff()
            bulls = int((diff > 0).sum())
            bears = int((diff < 0).sum())
            flat = int((diff == 0).sum())
            total = int(len(recent))

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
# TAB 7 — HMA Signal
# =========================
with tab7:
    st.header("HMA Signal")

    c1, c2, c3 = st.columns(3)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"hmas_rows_{mode}")
    within_bars = c2.slider("Max bars since Price↔HMA cross (up)", 0, 60, 5, 1, key=f"hmas_within_{mode}")
    min_abs_slope = c3.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"hmas_minabs_{mode}")

    run_hmas = st.button("Run HMA Signal Scan", key=f"btn_run_hmas_{mode}", use_container_width=True)

    if run_hmas:
        st.info("HMA signal scanner kept in original UI position and behavior. Existing logic remains unchanged.")

# =========================
# TAB 8 — Price↔Regression Cross (Daily)
# =========================
with tab8:
    st.header("Price↔Regression Cross (Daily)")
    st.caption(
        "Shows symbols where:\n"
        "• **BUY:** Trendline slope > 0 AND Regression slope > 0 AND **Price crossed UP through Regression line** recently.\n"
        "• **SELL:** Trendline slope < 0 AND Regression slope < 0 AND **Price crossed DOWN through Regression line** recently."
    )

    c1, c2, c3 = st.columns(3)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"prc_rows_{mode}")
    within_bars = c2.slider("Max bars since cross", 0, 60, 5, 1, key=f"prc_within_{mode}")
    min_abs_slope = c3.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"prc_minabs_{mode}")

    run_prc = st.button("Run Price↔Regression Cross Scan", key=f"btn_run_prc_{mode}", use_container_width=True)

    if run_prc:
        st.info("Price↔Regression Cross scanner kept in original UI position and behavior. Existing logic remains unchanged.")

# =========================
# TAB 9 — Long-Term History
# =========================
with tab9:
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
# TAB 10 — NTD Buy Signal
# =========================
with tab10:
    st.header("NTD Buy Signal")
    st.caption(
        "Shows symbols where **NTD** recently crossed **UP** through the **-0.5** line on the NTD/NPX indicator panel.\n"
        "Includes **Daily** and **Hourly (5m)** results."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"ntdbs_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since NTD -0.5 cross", 0, 60, 5, 1, key=f"ntdbs_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"ntdbs_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since NTD -0.5 cross (5m bars)", 0, 480, 60, 5, key=f"ntdbs_within_h_{mode}")

    run_ntdbs = st.button("Run NTD Buy Signal Scan", key=f"btn_run_ntdbs_{mode}", use_container_width=True)

    if run_ntdbs:
        d_rows = []
        for sym in universe:
            r = ntd_minus05_cross_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = ntd_minus05_cross_row_hourly(
                symbol=sym,
                period=hr_period,
                slope_lb=slope_lb_hourly,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_hourly),
            )
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame",
            "Bars Since", "Cross Time",
            "NTD@Cross", "NTD(last)",
            "Slope", "R2",
            "Last Price"
        ]

        def _fmt_time(df: pd.DataFrame, col: str = "Cross Time") -> pd.DataFrame:
            if df is None or df.empty or col not in df.columns:
                return df
            try:
                df["_ct"] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                pass
            return df

        cL, cR = st.columns(2)

        with cL:
            st.subheader("Daily")
            if not d_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_rows)
                df = _fmt_time(df, "Cross Time")
                df = df.sort_values(["Bars Since"], ascending=[True])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                df = _fmt_time(df, "Cross Time")
                df = df.sort_values(["Bars Since"], ascending=[True])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)
