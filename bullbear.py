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

def label_on_right(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.99, y_val, text, transform=trans,
        ha="right", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
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
        return f"{diff / ps:.1f} pips"
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

# =========================
# Mode switching
# =========================
def _reset_run_state_for_mode_switch():
    st.session_state.run_all = False
    st.session_state.ticker = None
    for k in [
        "df_hist", "df_ohlc", "fc_idx", "fc_vals", "fc_ci",
        "intraday", "chart", "hour_range", "mode_at_run"
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

# =========================
# Data fetchers
# =========================
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

def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"), progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    need = ["Open", "High", "Low", "Close"]
    have = [c for c in need if c in df.columns]
    if len(have) < 4:
        return pd.DataFrame()
    out = df[need].dropna()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is None:
        out.index = out.index.tz_localize(PACIFIC)
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_convert(PACIFIC)
    return out

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
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)
    return df

def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize() + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower", "upper"])
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    except Exception:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
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
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    mask = p.notna() & u.notna() & l.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]
    u = u[mask]
    l = l[mask]

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
    p = p[ok]
    l = l[ok]
    above = p > l
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    return cross_up.reindex(p.index, fill_value=False), cross_dn.reindex(p.index, fill_value=False)

def _strict_cross_series(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index if len(p) else l.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)

    p = p[ok]
    l = l[ok]
    diff = p - l
    prev_diff = diff.shift(1)

    cross_up = (diff > 0.0) & (prev_diff <= 0.0)
    cross_dn = (diff < 0.0) & (prev_diff >= 0.0)

    if len(cross_up):
        cross_up.iloc[0] = False
        cross_dn.iloc[0] = False

    return cross_up.fillna(False).reindex(p.index, fill_value=False), cross_dn.fillna(False).reindex(p.index, fill_value=False)
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

def trend_slope_align_row(symbol: str, daily_view_label: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 20:
            return None
        tm = _global_slope_1d(close_show)
        _, _, _, rm, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)))
        if not (np.isfinite(tm) and np.isfinite(rm)):
            return None
        return {
            "Symbol": symbol,
            "Trendline Slope": float(tm),
            "Regression Slope": float(rm),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "Last Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
        }
    except Exception:
        return None

def _current_regression_cross_side(price: pd.Series, reg_line: pd.Series):
    p = _coerce_1d_series(price)
    r = _coerce_1d_series(reg_line).reindex(p.index)
    ok = p.notna() & r.notna()
    if ok.sum() < 2:
        return None
    p = p[ok]; r = r[ok]
    above = bool(p.iloc[-1] > r.iloc[-1])
    return "Above" if above else "Below"

def last_regression_cross_row(symbol: str, daily_view_label: str, slope_lb: int):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 20:
            return None

        reg, _, _, slope_val, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)))
        if reg.empty:
            return None

        cross_up, cross_dn = _strict_cross_series(close_show, reg.reindex(close_show.index))
        last_up = cross_up[cross_up].index[-1] if cross_up.any() else None
        last_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None

        if last_up is None and last_dn is None:
            return None

        if last_dn is None or (last_up is not None and last_up >= last_dn):
            cross_time = last_up
            side = "Crossed Above"
        else:
            cross_time = last_dn
            side = "Crossed Below"

        bars_since = _bars_since_event(close_show.index, cross_time)
        curr = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        reg_now = float(reg.reindex(close_show.index).iloc[-1]) if np.isfinite(reg.reindex(close_show.index).iloc[-1]) else np.nan
        dlt = (curr / reg_now - 1.0) if np.isfinite(curr) and np.isfinite(reg_now) and reg_now != 0 else np.nan

        return {
            "Symbol": symbol,
            "Side": side,
            "Bars Since": int(bars_since),
            "Signal Time": cross_time,
            "Last Price": curr,
            "Regression": reg_now,
            "DeltaPct": dlt,
            "Slope": float(slope_val) if np.isfinite(slope_val) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def _global_slope_1d(close_like: pd.Series) -> float:
    s = _coerce_1d_series(close_like).dropna()
    if len(s) < 20:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, _ = np.polyfit(x, s.to_numpy(dtype=float), 1)
    return float(m)

def trend_buy_row_daily(symbol: str,
                        daily_view_label: str,
                        slope_lb_daily_val: int,
                        sr_lb_daily_val: int,
                        prox_pct: float):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < max(30, int(sr_lb_daily_val)):
            return None

        global_slope = _global_slope_1d(close_show)
        _, _, _, reg_slope, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb_daily_val)))
        if not (np.isfinite(global_slope) and np.isfinite(reg_slope)):
            return None
        if not (global_slope > 0 and reg_slope > 0):
            return None

        support = close_show.rolling(int(sr_lb_daily_val)).min()
        last_close = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        last_support = float(support.iloc[-1]) if np.isfinite(support.iloc[-1]) else np.nan
        if not (np.isfinite(last_close) and np.isfinite(last_support) and last_support > 0):
            return None

        near_support = last_close <= last_support * (1.0 + float(prox_pct))
        if not near_support:
            return None

        dist_pct = (last_close / last_support - 1.0)
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Last Price": last_close,
            "Support": last_support,
            "Dist to Support": dist_pct,
            "Global Slope": float(global_slope),
            "Regression Slope": float(reg_slope),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def trend_buy_row_hourly(symbol: str,
                         period: str,
                         slope_lb_hourly_val: int,
                         sr_lb_hourly_val: int,
                         prox_pct: float):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close_h = _coerce_1d_series(df["Close"]).dropna()
        if len(close_h) < max(30, int(sr_lb_hourly_val)):
            return None

        daily_close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        global_slope = _global_slope_1d(daily_close_full)

        _, _, _, reg_slope, r2 = regression_with_band(close_h, lookback=min(len(close_h), int(slope_lb_hourly_val)))
        if not (np.isfinite(global_slope) and np.isfinite(reg_slope)):
            return None
        if not (global_slope > 0 and reg_slope > 0):
            return None

        support = close_h.rolling(int(sr_lb_hourly_val)).min()
        last_close = float(close_h.iloc[-1]) if np.isfinite(close_h.iloc[-1]) else np.nan
        last_support = float(support.iloc[-1]) if np.isfinite(support.iloc[-1]) else np.nan
        if not (np.isfinite(last_close) and np.isfinite(last_support) and last_support > 0):
            return None

        near_support = last_close <= last_support * (1.0 + float(prox_pct))
        if not near_support:
            return None

        dist_pct = (last_close / last_support - 1.0)
        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Last Price": last_close,
            "Support": last_support,
            "Dist to Support": dist_pct,
            "Global Slope": float(global_slope),
            "Regression Slope": float(reg_slope),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def hot_buy_row_daily(symbol: str,
                      daily_view_label: str,
                      slope_lb_daily_val: int,
                      sr_lb_daily_val: int,
                      prox_pct: float):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < max(30, int(sr_lb_daily_val), int(hma_period) + 5):
            return None

        global_slope = _global_slope_1d(close_show)
        _, _, _, reg_slope, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb_daily_val)))
        if not (np.isfinite(global_slope) and np.isfinite(reg_slope)):
            return None
        if not (global_slope > 0 and reg_slope > 0):
            return None

        support = close_show.rolling(int(sr_lb_daily_val)).min()
        last_close = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        last_support = float(support.iloc[-1]) if np.isfinite(support.iloc[-1]) else np.nan
        if not (np.isfinite(last_close) and np.isfinite(last_support) and last_support > 0):
            return None
        if not (last_close <= last_support * (1.0 + float(prox_pct))):
            return None

        hma = compute_hma(close_show, period=int(hma_period))
        if hma.dropna().shape[0] < 3:
            return None
        cross_up, _ = _strict_cross_series(close_show, hma.reindex(close_show.index))
        last_up = cross_up[cross_up].index[-1] if cross_up.any() else None
        if last_up is None:
            return None
        bars_since_cross = _bars_since_event(close_show.index, last_up)

        dist_pct = (last_close / last_support - 1.0)
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Last Price": last_close,
            "Support": last_support,
            "Dist to Support": dist_pct,
            "Bars Since HMA↑": int(bars_since_cross),
            "Global Slope": float(global_slope),
            "Regression Slope": float(reg_slope),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def hot_buy_row_hourly(symbol: str,
                       period: str,
                       slope_lb_hourly_val: int,
                       sr_lb_hourly_val: int,
                       prox_pct: float):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close_h = _coerce_1d_series(df["Close"]).dropna()
        if len(close_h) < max(30, int(sr_lb_hourly_val), int(hma_period) + 5):
            return None

        daily_close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        global_slope = _global_slope_1d(daily_close_full)
        _, _, _, reg_slope, r2 = regression_with_band(close_h, lookback=min(len(close_h), int(slope_lb_hourly_val)))
        if not (np.isfinite(global_slope) and np.isfinite(reg_slope)):
            return None
        if not (global_slope > 0 and reg_slope > 0):
            return None

        support = close_h.rolling(int(sr_lb_hourly_val)).min()
        last_close = float(close_h.iloc[-1]) if np.isfinite(close_h.iloc[-1]) else np.nan
        last_support = float(support.iloc[-1]) if np.isfinite(support.iloc[-1]) else np.nan
        if not (np.isfinite(last_close) and np.isfinite(last_support) and last_support > 0):
            return None
        if not (last_close <= last_support * (1.0 + float(prox_pct))):
            return None

        hma = compute_hma(close_h, period=int(hma_period))
        if hma.dropna().shape[0] < 3:
            return None
        cross_up, _ = _strict_cross_series(close_h, hma.reindex(close_h.index))
        last_up = cross_up[cross_up].index[-1] if cross_up.any() else None
        if last_up is None:
            return None
        bars_since_cross = _bars_since_event(close_h.index, last_up)

        dist_pct = (last_close / last_support - 1.0)
        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Last Price": last_close,
            "Support": last_support,
            "Dist to Support": dist_pct,
            "Bars Since HMA↑": int(bars_since_cross),
            "Global Slope": float(global_slope),
            "Regression Slope": float(reg_slope),
            "R2": float(r2) if np.isfinite(r2) else np.nan,
        }
    except Exception:
        return None

def green_zone_buy_alert_row_daily(symbol: str,
                                   daily_view_label: str,
                                   slope_lb: int,
                                   ntd_win: int = 60,
                                   max_bars_since: int = 5):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 20:
            return None

        tm = _global_slope_1d(close_show)
        _, _, _, rm, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)))

        if not (np.isfinite(tm) and np.isfinite(rm)):
            return None
        if not (float(tm) > 0.0 and float(rm) > 0.0):
            return None

        ntd_full = compute_normalized_trend(close_full, window=int(ntd_win))
        npx_full = compute_normalized_price(close_full, window=int(ntd_win))

        ntd_show = _coerce_1d_series(ntd_full).reindex(close_show.index)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        ok = ntd_show.notna() & npx_show.notna()
        if ok.sum() < 2:
            return None
        ntd_show = ntd_show[ok]
        npx_show = npx_show[ok]

        up_mask, _ = _strict_cross_series(npx_show, ntd_show)
        up_mask = up_mask.reindex(ntd_show.index, fill_value=False)

        valid = up_mask & (npx_show < 0.0) & (ntd_show < 0.0)
        if not valid.any():
            return None

        t_cross = valid[valid].index[-1]
        bars_since = _bars_since_event(close_show.index, t_cross)
        if bars_since > int(max_bars_since):
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": t_cross,
            "Trendline Slope": float(tm) if np.isfinite(tm) else np.nan,
            "Regression Slope": float(rm) if np.isfinite(rm) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(npx_show.loc[t_cross]) if np.isfinite(npx_show.loc[t_cross]) else np.nan,
            "NPX(last)": float(npx_show.iloc[-1]) if np.isfinite(npx_show.iloc[-1]) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

def green_zone_buy_alert_row_hourly(symbol: str,
                                    period: str,
                                    slope_lb: int,
                                    ntd_win: int = 60,
                                    max_bars_since: int = 60):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = _coerce_1d_series(df["Close"]).ffill().dropna()
        if len(close) < 40:
            return None

        tm = _global_slope_1d(close)
        _, _, _, rm, r2 = regression_with_band(close, lookback=min(len(close), int(slope_lb)))

        if not (np.isfinite(tm) and np.isfinite(rm)):
            return None
        if not (float(tm) > 0.0 and float(rm) > 0.0):
            return None

        ntd = _coerce_1d_series(compute_normalized_trend(close, window=int(ntd_win)))
        npx = _coerce_1d_series(compute_normalized_price(close, window=int(ntd_win)))

        ok = ntd.notna() & npx.notna()
        if ok.sum() < 2:
            return None
        ntd = ntd[ok]
        npx = npx[ok]

        up_mask, _ = _strict_cross_series(npx, ntd)
        up_mask = up_mask.reindex(ntd.index, fill_value=False)

        valid = up_mask & (npx < 0.0) & (ntd < 0.0)
        if not valid.any():
            return None

        t_cross = valid[valid].index[-1]
        bars_since = _bars_since_event(close.index, t_cross)
        if bars_since > int(max_bars_since):
            return None

        last_px = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": t_cross,
            "Trendline Slope": float(tm) if np.isfinite(tm) else np.nan,
            "Regression Slope": float(rm) if np.isfinite(rm) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX@Cross": float(npx.loc[t_cross]) if np.isfinite(npx.loc[t_cross]) else np.nan,
            "NPX(last)": float(npx.iloc[-1]) if np.isfinite(npx.iloc[-1]) else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None

def zero_cross_row_daily(symbol: str,
                         daily_view_label: str,
                         slope_lb: int,
                         ntd_win: int = 60,
                         max_bars_since: int = 5):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if len(close_show) < 20:
            return None

        tm = _global_slope_1d(close_show)
        _, _, _, rm, r2 = regression_with_band(close_show, lookback=min(len(close_show), int(slope_lb)))
        if not (np.isfinite(tm) and float(tm) > 0.0):
            return None

        npx_full = compute_normalized_price(close_full, window=int(ntd_win))
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)
        if npx_show.dropna().shape[0] < 2:
            return None

        up_mask, _ = npx_zero_cross_masks(npx_show, level=0.0)
        if not up_mask.any():
            return None

        t_cross = up_mask[up_mask].index[-1]
        bars_since = _bars_since_event(close_show.index, t_cross)
        if bars_since > int(max_bars_since):
            return None

        last_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Bars Since Cross": int(bars_since),
            "Cross Time (PST)": t_cross,
            "Trendline Slope": float(tm) if np.isfinite(tm) else np.nan,
            "Regression Slope": float(rm) if np.isfinite(rm) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NPX(last)": float(npx_show.dropna().iloc[-1]) if npx_show.dropna().shape[0] else np.nan,
            "Last Price": last_px,
        }
    except Exception:
        return None
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
(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15
) = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Trend and Slope Align",
    "Trend Buy",
    "Hot Buy",
    "Green Zone Buy Alert",
    "Zero Cross",
    "Green Cross",
    "HMA Signal",
    "New HMA Cross",
    "NEW NTD Cross",
    "Price↔Regression Cross",
    "Bull vs Bear",
    "Long-Term History",
    "NTD Buy Signal",
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

        if st.session_state.chart in ("Daily", "Both"):
            daily_res = render_daily_chart(disp_ticker, daily_view)
            if daily_res:
                trade_box.info(f"Daily: {daily_res['trade_instruction']}")

        if st.session_state.chart in ("Hourly", "Both"):
            hourly_res = render_hourly_chart(disp_ticker, period=period_map.get(st.session_state.hour_range, "1d"))
            if hourly_res:
                trade_box.info(f"Hourly: {hourly_res['trade_instruction']}")

        st.subheader("30-Day Forecast")
        render_forecast_chart(
            st.session_state.get("df_hist"),
            st.session_state.get("fc_idx"),
            st.session_state.get("fc_vals"),
            st.session_state.get("fc_ci"),
            disp_ticker,
        )

# =========================
# TAB 2 — Enhanced Forecast
# =========================
with tab2:
    st.header("Enhanced Forecast")
    st.caption("Daily + Hourly charts with active overlays.")

    sel2 = st.selectbox("Ticker:", universe, key=f"enh_ticker_{mode}")
    chart2 = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key=f"enh_chart_{mode}")
    hour_range2 = st.selectbox("Hourly lookback:", ["24h", "48h", "96h"], key=f"enh_hour_range_{mode}")

    run2 = st.button("Run Enhanced Forecast", key=f"btn_run_enh_{mode}", use_container_width=True)
    if run2:
        if chart2 in ("Daily", "Both"):
            render_daily_chart(sel2, daily_view)
        if chart2 in ("Hourly", "Both"):
            render_hourly_chart(sel2, period=period_map.get(hour_range2, "1d"))

# =========================
# TAB 3 — Trend and Slope Align
# =========================
with tab3:
    st.header("Trend and Slope Align")
    st.caption(
        "Shows symbols where:\n"
        "• **Trendline slope > 0** AND **Regression slope > 0** on the selected Daily view."
    )

    c1, c2 = st.columns(2)
    max_rows = c1.slider("Max rows", 10, 300, 60, 10, key=f"tsa_rows_{mode}")
    min_abs_slope = c2.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"tsa_minabs_{mode}")

    run_tsa = st.button("Run Trend and Slope Align Scan", key=f"btn_run_tsa_{mode}", use_container_width=True)

    if run_tsa:
        rows = []
        for sym in universe:
            r = trend_slope_align_row(sym, daily_view, slope_lb_daily)
            if not r:
                continue
            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if not (np.isfinite(tm) and np.isfinite(rm) and tm > 0 and rm > 0):
                continue
            if float(min_abs_slope) > 0.0 and (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                continue
            rows.append(r)

        if not rows:
            st.write("No matches.")
        else:
            df = pd.DataFrame(rows)
            df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
            df = df.sort_values(["_score", "R2"], ascending=[False, False])
            st.dataframe(
                df[["Symbol", "Trendline Slope", "Regression Slope", "R2", "Last Price"]].head(max_rows).reset_index(drop=True),
                use_container_width=True
            )

# =========================
# TAB 4 — Trend Buy
# =========================
with tab4:
    st.header("Trend Buy")
    st.caption(
        "Shows symbols where both **Trendline slope > 0** and **Regression slope > 0**.\n"
        "Daily and Hourly are shown separately."
    )

    c1, c2, c3 = st.columns(3)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"tb_rows_{mode}")
    hours = c2.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"tb_hr_win_{mode}")
    min_abs_slope = c3.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"tb_minabs_{mode}")

    run_tb = st.button("Run Trend Buy Scan", key=f"btn_run_tb_{mode}", use_container_width=True)

    if run_tb:
        d_rows = []
        for sym in universe:
            r = trend_buy_row_daily(sym, daily_view, slope_lb_daily)
            if not r:
                continue
            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue
            d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = trend_buy_row_hourly(sym, hr_period, slope_lb_hourly)
            if not r:
                continue
            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue
            h_rows.append(r)

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Daily")
            if not d_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_rows)
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(df[["Symbol", "Trendline Slope", "Regression Slope", "R2", "Last Price", "Regression", "GapPct"]].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["_score", "R2"], ascending=[False, False])
                st.dataframe(df[["Symbol", "Trendline Slope", "Regression Slope", "R2", "Last Price", "Regression", "GapPct"]].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 5 — Hot Buy
# =========================
with tab5:
    st.header("Hot Buy")
    st.caption(
        "Shows symbols where **Price recently crossed UP through HMA** and:\n"
        "• **Trendline slope > 0** AND **Regression slope > 0**."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"hot_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since HMA cross (up)", 0, 60, 5, 1, key=f"hot_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"hot_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since HMA cross (up)", 0, 480, 60, 5, key=f"hot_within_h_{mode}")

    run_hot = st.button("Run Hot Buy Scan", key=f"btn_run_hot_{mode}", use_container_width=True)

    if run_hot:
        d_rows = []
        for sym in universe:
            r = hot_buy_row_daily(sym, daily_view, slope_lb_daily, hma_len=int(hma_period), max_bars_since=int(within_daily))
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = hot_buy_row_hourly(sym, hr_period, slope_lb_hourly, hma_len=int(hma_period), max_bars_since=int(within_hourly))
            if r:
                h_rows.append(r)

        show_cols = ["Symbol", "Bars Since Cross", "Cross Time", "Trendline Slope", "Regression Slope", "R2", "Last Price", "HMA(last)"]

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
                df = _fmt_time(df)
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                df = _fmt_time(df)
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 6 — Green Zone Buy Alert
# =========================
with tab6:
    st.header("Green Zone Buy Alert")
    st.caption(
        "Shows symbols where **NPX recently crossed UP through NTD** and:\n"
        "• **Trendline slope > 0** AND **Regression slope > 0**\n"
        "• Both NPX and NTD were **below 0.0** at the cross"
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"gz_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since NPX↔NTD cross", 0, 60, 5, 1, key=f"gz_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"gz_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since NPX↔NTD cross", 0, 480, 60, 5, key=f"gz_within_h_{mode}")

    run_gz = st.button("Run Green Zone Buy Alert Scan", key=f"btn_run_gz_{mode}", use_container_width=True)

    if run_gz:
        d_rows = []
        for sym in universe:
            r = green_zone_buy_alert_row_daily(sym, daily_view, slope_lb_daily, ntd_win=int(ntd_window), max_bars_since=int(within_daily))
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = green_zone_buy_alert_row_hourly(sym, hr_period, slope_lb_hourly, ntd_win=int(ntd_window), max_bars_since=int(within_hourly))
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame", "Bars Since Cross", "Cross Time (PST)",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX@Cross", "NPX(last)", "Last Price"
        ]

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Daily")
            if not d_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "Regression Slope", "R2"], ascending=[True, False, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "Regression Slope", "R2"], ascending=[True, False, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 7 — Zero Cross
# =========================
with tab7:
    st.header("Zero Cross")
    st.caption(
        "Shows symbols where **NPX recently crossed UP through 0.0** and the **global trendline slope > 0**.\n"
        "Includes Daily and Hourly results."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"zc_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since NPX zero cross", 0, 60, 5, 1, key=f"zc_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"zc_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since NPX zero cross", 0, 480, 60, 5, key=f"zc_within_h_{mode}")

    run_zc = st.button("Run Zero Cross Scan", key=f"btn_run_zc_{mode}", use_container_width=True)

    if run_zc:
        d_rows = []
        for sym in universe:
            r = zero_cross_row_daily(sym, daily_view, slope_lb_daily, ntd_win=int(ntd_window), max_bars_since=int(within_daily))
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = zero_cross_row_hourly(sym, hr_period, slope_lb_hourly, ntd_win=int(ntd_window), max_bars_since=int(within_hourly))
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame", "Bars Since Cross", "Cross Time (PST)",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX(last)", "Last Price"
        ]

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Daily")
            if not d_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 8 — Green Cross
# =========================
with tab8:
    st.header("Green Cross")
    st.caption(
        "Shows symbols where **NPX recently crossed UP through -0.5** and the **global trendline slope > 0**.\n"
        "Includes Daily and Hourly results."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"gc_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since NPX green cross", 0, 60, 5, 1, key=f"gc_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"gc_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since NPX green cross", 0, 480, 60, 5, key=f"gc_within_h_{mode}")

    run_gc = st.button("Run Green Cross Scan", key=f"btn_run_gc_{mode}", use_container_width=True)

    if run_gc:
        d_rows = []
        for sym in universe:
            r = green_cross_row_daily(sym, daily_view, slope_lb_daily, ntd_win=int(ntd_window), max_bars_since=int(within_daily))
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = green_cross_row_hourly(sym, hr_period, slope_lb_hourly, ntd_win=int(ntd_window), max_bars_since=int(within_hourly))
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame", "Bars Since Cross", "Cross Time (PST)",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX(last)", "Last Price"
        ]

        cL, cR = st.columns(2)
        with cL:
            st.subheader("Daily")
            if not d_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(d_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time (PST)"], errors="coerce")
                    df["Cross Time (PST)"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df = df.sort_values(["Bars Since Cross", "Trendline Slope", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 9 — HMA Signal (Daily)
# =========================
with tab9:
    st.header("HMA Signal (Daily)")
    st.caption(
        "Shows symbols where **Price recently crossed UP through HMA** on the **Daily** price chart.\n\n"
        "Lists:\n"
        "1) **Trendline > 0 AND Regression > 0** + Cross Up\n"
        "2) **Trendline > 0** + Cross Up\n"
        "3) **Regression > 0** + Cross Up\n"
    )

    c1, c2, c3 = st.columns(3)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"hmas_rows_{mode}")
    within_bars = c2.slider("Max bars since Price↔HMA cross (up)", 0, 60, 5, 1, key=f"hmas_within_{mode}")
    min_abs_slope = c3.slider("Min |slope| filter (optional)", 0.0, 1.0, 0.0, 0.01, key=f"hmas_minabs_{mode}")

    run_hmas = st.button("Run HMA Signal Scan", key=f"btn_run_hmas_{mode}", use_container_width=True)

    if run_hmas:
        base_rows = []
        for sym in universe:
            r = hma_cross_up_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                hma_len=int(hma_period),
                max_bars_since=int(within_bars),
            )
            if not r:
                continue

            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))

            if float(min_abs_slope) > 0.0:
                keep = (np.isfinite(tm) and abs(tm) >= float(min_abs_slope)) or (np.isfinite(rm) and abs(rm) >= float(min_abs_slope))
                if not keep:
                    continue

            base_rows.append(r)

        rows_1 = [r for r in base_rows if np.isfinite(r.get("Trendline Slope", np.nan)) and np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0 and float(r["Regression Slope"]) > 0.0]
        rows_2 = [r for r in base_rows if np.isfinite(r.get("Trendline Slope", np.nan)) and float(r["Trendline Slope"]) > 0.0]
        rows_3 = [r for r in base_rows if np.isfinite(r.get("Regression Slope", np.nan)) and float(r["Regression Slope"]) > 0.0]

        show_cols = [
            "Symbol", "Bars Since Cross", "Cross Time",
            "Trendline Slope", "Regression Slope", "R2",
            "Last Price", "HMA(last)"
        ]

        def _fmt_cross_time(df: pd.DataFrame, col: str = "Cross Time") -> pd.DataFrame:
            if df is None or df.empty or col not in df.columns:
                return df
            try:
                df["_ct"] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                pass
            return df

        cA, cB, cC = st.columns(3)

        with cA:
            st.subheader("1) Trendline > 0 AND Regression > 0 + Cross Up")
            if not rows_1:
                st.write("No matches.")
            else:
                df = pd.DataFrame(rows_1)
                df = _fmt_cross_time(df, "Cross Time")
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cB:
            st.subheader("2) Trendline > 0 + Cross Up")
            if not rows_2:
                st.write("No matches.")
            else:
                df = pd.DataFrame(rows_2)
                df = _fmt_cross_time(df, "Cross Time")
                df["_score"] = df["Trendline Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cC:
            st.subheader("3) Regression > 0 + Cross Up")
            if not rows_3:
                st.write("No matches.")
            else:
                df = pd.DataFrame(rows_3)
                df = _fmt_cross_time(df, "Cross Time")
                df["_score"] = df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 10 — New HMA Cross
# =========================
with tab10:
    st.header("New HMA Cross")
    st.caption(
        "Shows symbols where the **global trend is upward** and **Price recently crossed UP through the BB Mid line**.\n"
        "Includes **Daily** and **Hourly (5m)** results."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"bbmid_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since Price↔BB Mid cross (up)", 0, 60, 5, 1, key=f"bbmid_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"bbmid_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since Price↔BB Mid cross (up)", 0, 480, 60, 5, key=f"bbmid_within_h_{mode}")

    run_bbmid = st.button("Run New HMA Cross Scan", key=f"btn_run_bbmid_{mode}", use_container_width=True)

    if run_bbmid:
        d_rows = []
        for sym in universe:
            r = bb_mid_cross_up_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                bb_window=int(bb_win),
                bb_sigma=float(bb_mult),
                bb_ema=bool(bb_use_ema),
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = bb_mid_cross_up_row_hourly(
                symbol=sym,
                period=hr_period,
                bb_window=int(bb_win),
                bb_sigma=float(bb_mult),
                bb_ema=bool(bb_use_ema),
                max_bars_since=int(within_hourly),
            )
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame",
            "Bars Since Cross", "Cross Time",
            "Trendline Slope", "Regression Slope", "R2",
            "Last Price", "BB Mid(last)"
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
                df["_score"] = df["Trendline Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                df = _fmt_time(df, "Cross Time")
                df["_score"] = df["Trendline Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 11 — NEW NTD Cross
# =========================
with tab11:
    st.header("NEW NTD Cross")
    st.caption(
        "Shows symbols from the **Daily** and **Hourly** charts where the **global trend is upward** and "
        "**NPX (Norm Price) recently crossed UP through the NTD line**."
    )

    c1, c2, c3, c4 = st.columns(4)
    max_rows = c1.slider("Max rows per list", 10, 300, 60, 10, key=f"newntdx_rows_{mode}")
    within_daily = c2.slider("Daily: Max bars since NPX↔NTD cross (up)", 0, 60, 5, 1, key=f"newntdx_within_d_{mode}")
    hours = c3.selectbox("Hourly scan window", ["24h", "48h", "96h"], index=0, key=f"newntdx_hr_win_{mode}")
    within_hourly = c4.slider("Hourly: Max bars since NPX↔NTD cross (up)", 0, 480, 60, 5, key=f"newntdx_within_h_{mode}")

    run_newntdx = st.button("Run NEW NTD Cross Scan", key=f"btn_run_newntdx_{mode}", use_container_width=True)

    if run_newntdx:
        d_rows = []
        for sym in universe:
            r = npx_ntd_cross_up_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_daily),
            )
            if r:
                d_rows.append(r)

        h_rows = []
        hr_period = period_map.get(hours, "1d")
        for sym in universe:
            r = npx_ntd_cross_up_row_hourly(
                symbol=sym,
                period=hr_period,
                ntd_win=int(ntd_window),
                max_bars_since=int(within_hourly),
            )
            if r:
                r2 = dict(r)
                r2["Frame"] = f"Hourly({hours})"
                h_rows.append(r2)

        show_cols = [
            "Symbol", "Frame",
            "Bars Since Cross", "Cross Time",
            "Trendline Slope", "Regression Slope", "R2",
            "NPX@Cross", "NTD@Cross", "NPX(last)", "NTD(last)",
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
                df["_score"] = df["Trendline Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader(f"Hourly ({hours})")
            if not h_rows:
                st.write("No matches.")
            else:
                df = pd.DataFrame(h_rows)
                df = _fmt_time(df, "Cross Time")
                df["_score"] = df["Trendline Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 12 — Price↔Regression Cross (Daily)
# =========================
with tab12:
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
        rows = []
        for sym in universe:
            r = price_regression_cross_row_daily(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=slope_lb_daily,
                max_bars_since=int(within_bars),
            )
            if not r:
                continue
            tm = float(r.get("Trendline Slope", np.nan))
            rm = float(r.get("Regression Slope", np.nan))
            if float(min_abs_slope) > 0.0:
                if (abs(tm) < float(min_abs_slope)) and (abs(rm) < float(min_abs_slope)):
                    continue
            rows.append(r)

        buys = [r for r in rows if str(r.get("Side", "")).upper() == "BUY"]
        sells = [r for r in rows if str(r.get("Side", "")).upper() == "SELL"]

        show_cols = [
            "Symbol", "Side",
            "Bars Since Cross", "Cross Time",
            "Trendline Slope", "Regression Slope", "R2",
            "Last Price", "Regression Line (last)"
        ]

        cL, cR = st.columns(2)
        with cL:
            st.subheader("BUY: Price crossed UP through regression line")
            if not buys:
                st.write("No matches.")
            else:
                df = pd.DataFrame(buys)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time"], errors="coerce")
                    df["Cross Time"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, False, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

        with cR:
            st.subheader("SELL: Price crossed DOWN through regression line")
            if not sells:
                st.write("No matches.")
            else:
                df = pd.DataFrame(sells)
                try:
                    df["_ct"] = pd.to_datetime(df["Cross Time"], errors="coerce")
                    df["Cross Time"] = df["_ct"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except Exception:
                    pass
                df["_score"] = df["Trendline Slope"].astype(float) + df["Regression Slope"].astype(float)
                df = df.sort_values(["Bars Since Cross", "_score", "R2"], ascending=[True, True, False])
                st.dataframe(df[show_cols].head(max_rows).reset_index(drop=True), use_container_width=True)

# =========================
# TAB 13 — Bull vs Bear
# =========================
with tab13:
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
# TAB 14 — Long-Term History
# =========================
with tab14:
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
# TAB 15 — NTD Buy Signal
# =========================
with tab15:
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
