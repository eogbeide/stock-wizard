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
     UPDATED:
     Beautiful rectangular ribbon tabs (BaseWeb tabs)
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
     UPDATED:
     Beautiful chart container styling
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
            try:
                st.rerun()
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
    for k in [
        "df_hist", "df_ohlc", "fc_idx", "fc_vals", "fc_ci",
        "intraday", "chart", "hour_range", "mode_at_run"
    ]:
        st.session_state.pop(k, None)

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
            try:
                st.rerun()
            except Exception:
                pass

if mcol2.button("📈 Stocks", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        try:
            st.experimental_rerun()
        except Exception:
            try:
                st.rerun()
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

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float = None) -> str:
    """
    Shows BUY/SELL only when Global Trendline slope and Local Slope agree.
    Backward-compatible: if global_trend_slope is None, uses trend_slope only.
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
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
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

st.sidebar.subheader("HMA (Price Charts)")
show_hma = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")

# ---------------------------
# Universe
# ---------------------------
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
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        try:
            s = s.tz_convert(PACIFIC)
        except Exception:
            pass
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[
        ["Open","High","Low","Close"]
    ].dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        try:
            df = df.tz_convert(PACIFIC)
        except Exception:
            pass
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.tz_localize("UTC")
        except TypeError:
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
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC), periods=30, freq="D", tz=PACIFIC)
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower Close", "upper Close"])
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)

    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception:
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

# =========================
# Part 3/10 — bullbear.py
# =========================
# ---------------------------
# Regression & ±2σ band
# ---------------------------
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

def find_band_bounce_signal(price: pd.Series,
                            upper_band: pd.Series,
                            lower_band: pd.Series,
                            slope_val: float):
    """
    Detect most recent BUY/SELL signal based on a 'bounce' back inside the ±2σ band.
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
    ntd = _coerce_1d_series(ntd)
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

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

# =========================
# Part 5/10 — bullbear.py
# =========================
# ---------------------------
# Ichimoku, Supertrend, PSAR (kept for overlays; scanner tab removed per request)
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

# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Scanners (kept):
#   - Recent BUY (band bounce)
#   - R² Daily / Hourly
#   - Band Proximity
#   - Support Reversal
#   - NEW: HMA Buy (Daily HMA55 cross-up within N bars)
# Removed per request:
#   - NPX 0.5-Cross Scanner tab
#   - Fib NPX 0.0 Signal Scanner tab
#   - News tab
#   - Ichimoku Kijun Scanner tab
# ---------------------------
@st.cache_data(ttl=120)
def last_band_bounce_signal_daily(symbol: str, slope_lb: int, daily_view_label: str):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < 5:
            return None

        yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb))
        sig = find_band_bounce_signal(close, up, lo, m)
        if sig is None:
            return None

        t = sig.get("time", None)
        if t is None or t not in close.index:
            return None

        loc = int(close.index.get_loc(t))
        bars_since = int((len(close) - 1) - loc)

        curr = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
        spx = float(sig.get("price", np.nan))
        dlt = (curr / spx - 1.0) if np.isfinite(curr) and np.isfinite(spx) and spx != 0 else np.nan

        return {
            "Symbol": symbol,
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
            ts = real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None
            return np.nan, np.nan, ts
        _, _, _, m, r2 = regression_with_band(hc, lookback=int(slope_lb))
        ts = real_times[-1] if isinstance(real_times, pd.DatetimeIndex) and len(real_times) else None
        return float(r2) if np.isfinite(r2) else np.nan, float(m) if np.isfinite(m) else np.nan, ts
    except Exception:
        return np.nan, np.nan, None

@st.cache_data(ttl=120)
def daily_r2_band_proximity(symbol: str,
                            daily_view_label: str,
                            slope_lb: int,
                            prox: float,
                            z: float = 2.0):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty:
            return None
        slope_lb = int(max(2, slope_lb))
        if len(close_show) < max(6, slope_lb + 2):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=slope_lb, z=float(z))
        if lo is None or up is None or lo.dropna().empty or up.dropna().empty:
            return None

        px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        lo_last = float(lo.iloc[-1]) if np.isfinite(lo.iloc[-1]) else np.nan
        up_last = float(up.iloc[-1]) if np.isfinite(up.iloc[-1]) else np.nan
        if not np.all(np.isfinite([px, lo_last, up_last, m, r2])):
            return None

        dist_lo = (px / lo_last - 1.0) if lo_last != 0 else np.nan
        dist_up = (px / up_last - 1.0) if up_last != 0 else np.nan
        abs_lo = abs(dist_lo) if np.isfinite(dist_lo) else np.nan
        abs_up = abs(dist_up) if np.isfinite(dist_up) else np.nan

        prox = abs(float(prox))
        near_lo = bool(np.isfinite(abs_lo) and abs_lo <= prox)
        near_up = bool(np.isfinite(abs_up) and abs_up <= prox)

        return {
            "Symbol": symbol,
            "AsOf": close_show.index[-1] if isinstance(close_show.index, pd.DatetimeIndex) and len(close_show.index) else None,
            "Price": px,
            "Lower -2σ": lo_last,
            "Upper +2σ": up_last,
            "Dist Lower (%)": dist_lo,
            "Dist Upper (%)": dist_up,
            "Slope": float(m),
            "R2": float(r2),
            "Near Lower": near_lo,
            "Near Upper": near_up,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def daily_support_reversal_heading_up(symbol: str,
                                      daily_view_label: str,
                                      sr_lb: int,
                                      prox: float,
                                      confirm_bars: int = 2):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < max(10, int(sr_lb)):
            return None

        sup = close.rolling(int(sr_lb), min_periods=1).min()
        near_support = close <= (sup * (1.0 + float(prox)))

        if not near_support.any():
            return None
        t_touch = near_support[near_support].index[-1]
        loc_touch = int(close.index.get_loc(t_touch))
        bars_since_touch = int((len(close) - 1) - loc_touch)

        seg = close.loc[t_touch:].dropna()
        if len(seg) < int(confirm_bars) + 1:
            return None
        deltas = np.diff(seg.iloc[-(int(confirm_bars)+1):])
        if not bool(np.all(deltas > 0)):
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
# NEW (THIS REQUEST): HMA Buy scanner helper (Daily HMA(55) cross-up within last N bars)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_hma_cross_up(symbol: str,
                            daily_view_label: str,
                            hma_len: int,
                            slope_lb: int,
                            within_last_n_bars: int):
    """
    Returns dict when:
      - Close crosses ABOVE HMA(hma_len)
      - Cross occurred within last N bars (N from slider 1–3)
    Also returns regression slope sign (for grouping into Regression >0 vs <0).
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close.empty or len(close) < max(10, int(hma_len) + 5):
            return None

        hma = compute_hma(close, period=int(hma_len)).reindex(close.index)
        if hma.dropna().empty:
            return None

        cross_up, _ = _cross_series(close, hma)
        if not cross_up.any():
            return None
        t_cross = cross_up[cross_up].index[-1]
        loc = int(close.index.get_loc(t_cross))
        bars_since = int((len(close) - 1) - loc)

        within_last_n_bars = int(max(1, within_last_n_bars))
        if bars_since > within_last_n_bars:
            return None

        px_cross = float(close.loc[t_cross]) if np.isfinite(close.loc[t_cross]) else np.nan
        hma_cross = float(hma.loc[t_cross]) if np.isfinite(hma.loc[t_cross]) else np.nan
        px_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan

        yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None

        return {
            "Symbol": symbol,
            "Bars Since Cross": bars_since,
            "Cross Time": t_cross,
            "Price@Cross": px_cross,
            "HMA@Cross": hma_cross,
            "Current Price": px_last,
            "Slope": float(m),
            "R2": float(r2),
        }
    except Exception:
        return None
# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Session state init
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "1d"
    st.session_state.mode_at_run = None

# ---------------------------
# Symbol picker (kept stable keys to avoid duplicate-key issues)
# ---------------------------
def _symbol_picker(universe_list):
    if mode == "Forex":
        label = "Forex Symbol"
        key = "dd_forex_symbol"
    else:
        label = "Stock Symbol"
        key = "dd_stock_symbol"

    # Ensure existing value remains valid
    if key in st.session_state:
        if st.session_state[key] not in universe_list:
            st.session_state[key] = universe_list[0] if universe_list else None

    sel = st.sidebar.selectbox(label, universe_list, index=0, key=key)
    return sel

# ---------------------------
# Run controls (kept: Forecast button + dropdown behaviors)
# ---------------------------
st.sidebar.subheader("Run")
hour_range = st.sidebar.selectbox(
    "Intraday Range:",
    ["1d", "5d", "1mo", "3mo"],
    index=0,
    key="sb_hour_range"
)

run_btn = st.sidebar.button("🔮 Forecast + Charts", use_container_width=True, key="btn_run_forecast")

sel = _symbol_picker(universe)
st.session_state.ticker = sel

if run_btn:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# ---------------------------
# Data load (only when run)
# ---------------------------
def _load_all(symbol: str, hour_range: str):
    close = fetch_hist(symbol)
    ohlc = fetch_hist_ohlc(symbol)
    intr = fetch_intraday(symbol, period=hour_range)
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close)
    return close, ohlc, intr, fc_idx, fc_vals, fc_ci

if st.session_state.run_all and st.session_state.ticker:
    with st.spinner("Loading data + building forecasts…"):
        try:
            df_hist, df_ohlc, intraday, fc_idx, fc_vals, fc_ci = _load_all(st.session_state.ticker, hour_range)
            st.session_state.df_hist = df_hist
            st.session_state.df_ohlc = df_ohlc
            st.session_state.intraday = intraday
            st.session_state.fc_idx = fc_idx
            st.session_state.fc_vals = fc_vals
            st.session_state.fc_ci = fc_ci
            st.session_state.hour_range = hour_range
        except Exception as e:
            st.session_state.run_all = False
            st.error(f"Run failed: {e}")

# ---------------------------
# Renderers
# ---------------------------
def render_forecast(symbol: str, close_full: pd.Series, fc_idx, fc_vals, fc_ci):
    close = _coerce_1d_series(close_full).dropna()
    if close.empty:
        st.warning("No daily history available.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")

    ax.plot(close.index, close.values, label="Close")
    ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, label="Forecast")

    if isinstance(fc_ci, pd.DataFrame) and len(fc_ci.columns) >= 2:
        lo = _coerce_1d_series(fc_ci.iloc[:, 0]).reindex(fc_idx)
        hi = _coerce_1d_series(fc_ci.iloc[:, 1]).reindex(fc_idx)
        ax.fill_between(fc_idx, lo.values, hi.values, alpha=0.12, label="Forecast CI")

    ax.set_title(f"{symbol} — SARIMAX Forecast (Daily)")
    ax.set_xlabel("Date (PST)")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def render_enhanced_forecast(symbol: str, close_full: pd.Series, slope_lb: int):
    close = _coerce_1d_series(close_full).dropna()
    close = _coerce_1d_series(subset_by_daily_view(close, daily_view)).dropna()
    if close.empty or len(close) < 5:
        st.warning("Not enough history for enhanced view.")
        return

    yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb))

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")

    ax.plot(close.index, close.values, label="Close")
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.2, label=f"Regression ({fmt_slope(m)}/bar)")
    if (not up.empty) and (not lo.empty):
        ax.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")

        sig = find_band_bounce_signal(close, up, lo, m)
        if sig is not None:
            annotate_crossover(ax, sig["time"], sig["price"], sig["side"])

    ax.text(
        0.99, 0.02,
        f"R² ({slope_lb} bars): {fmt_r2(r2)}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )

    if show_fibs:
        fibs = fibonacci_levels(close)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
            ax.text(close.index[-1], y, f" {lbl}", va="center")

    ax.set_title(f"{symbol} — Enhanced Daily (Regression + ±2σ)")
    ax.set_xlabel("Date (PST)")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def render_daily_price(symbol: str, close_full: pd.Series, ohlc_full: pd.DataFrame):
    close_full = _coerce_1d_series(close_full).dropna()
    close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
    if close.empty:
        st.warning("No daily data.")
        return

    # Align OHLC
    ohlc = ohlc_full.copy() if isinstance(ohlc_full, pd.DataFrame) else pd.DataFrame()
    if not ohlc.empty and isinstance(ohlc.index, pd.DatetimeIndex):
        ohlc = ohlc.sort_index()
        x0, x1 = close.index[0], close.index[-1]
        ohlc = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)]

    ema20 = close.ewm(span=20, adjust=False).mean()
    sup = close.rolling(int(sr_lb_daily), min_periods=1).min()
    res = close.rolling(int(sr_lb_daily), min_periods=1).max()

    hma = compute_hma(close, period=int(hma_period)) if show_hma else pd.Series(index=close.index, dtype=float)
    kijun = pd.Series(index=close.index, dtype=float)
    if show_ichi and (not ohlc.empty) and {"High","Low","Close"}.issubset(ohlc.columns):
        _, kijun, _, _, _ = ichimoku_lines(
            ohlc["High"], ohlc["Low"], ohlc["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
            shift_cloud=False
        )
        kijun = _coerce_1d_series(kijun).reindex(close.index).ffill().bfill()

    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    yhat, up2, lo2, m, r2 = regression_with_band(close, lookback=int(slope_lb_daily))

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("white")

    ax.plot(close.index, close.values, label="Close")
    ax.plot(ema20.index, ema20.values, "--", label="EMA 20")

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.8, label=f"HMA({hma_period})")

    if show_ichi and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up.dropna().empty and not bb_lo.dropna().empty:
        ax.fill_between(close.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    # Support/Resistance
    if not sup.dropna().empty and not res.dropna().empty:
        s_last = float(sup.iloc[-1]) if np.isfinite(sup.iloc[-1]) else np.nan
        r_last = float(res.iloc[-1]) if np.isfinite(res.iloc[-1]) else np.nan
        if np.isfinite(s_last) and np.isfinite(r_last):
            ax.hlines(r_last, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linewidth=1.6, label="Resistance")
            ax.hlines(s_last, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linewidth=1.6, label="Support")
            label_on_left(ax, r_last, f"R {fmt_price_val(r_last)}", color="tab:red")
            label_on_left(ax, s_last, f"S {fmt_price_val(s_last)}", color="tab:green")

    # Regression + bands + bounce signal
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.2, label=f"Regression ({fmt_slope(m)}/bar)")
    if not up2.empty and not lo2.empty:
        ax.plot(up2.index, up2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo2.index, lo2.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")
        sig = find_band_bounce_signal(close, up2, lo2, m)
        if sig is not None:
            annotate_crossover(ax, sig["time"], sig["price"], sig["side"])

    # Title + footer stats
    px_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
    nbb_txt = ""
    try:
        last_nbb = float(_coerce_1d_series(bb_nbb).dropna().iloc[-1]) if show_bbands else np.nan
        if np.isfinite(last_nbb):
            nbb_txt = f" | NBB {last_nbb:+.2f}"
    except Exception:
        pass

    ax.text(
        0.99, 0.02,
        f"Current price: {fmt_price_val(px_last)}{nbb_txt}  •  R²({slope_lb_daily})={fmt_r2(r2)}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )

    ax.set_title(f"{symbol} — Daily Price Chart")
    ax.set_xlabel("Date (PST)")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def render_hourly_price(symbol: str, intraday_df: pd.DataFrame):
    if intraday_df is None or intraday_df.empty or "Close" not in intraday_df.columns:
        st.warning("No intraday data available.")
        return

    real_times = intraday_df.index if isinstance(intraday_df.index, pd.DatetimeIndex) else None
    intr_plot = intraday_df.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))

    hc = _coerce_1d_series(intr_plot["Close"]).ffill()
    he = hc.ewm(span=20).mean()

    sup_h = hc.rolling(int(sr_lb_hourly), min_periods=1).min()
    res_h = hc.rolling(int(sr_lb_hourly), min_periods=1).max()

    hma_h = compute_hma(hc, period=int(hma_period)) if show_hma else pd.Series(index=hc.index, dtype=float)

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    )

    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, lookback=int(slope_lb_hourly))

    fig2, axes = None, None
    if show_nrsi:
        fig2, (ax2, ax2w) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.34)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        ax2w = None
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)

    fig2.patch.set_facecolor("white")

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    # S/R
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        if np.isfinite(res_val) and np.isfinite(sup_val):
            ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linewidth=1.6, label="Resistance")
            ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linewidth=1.6, label="Support")
            label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
            label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")
    except Exception:
        res_val = np.nan
        sup_val = np.nan

    # Regression bands + bounce
    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")
        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, m_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

    # Title + stats
    px_val = float(hc.iloc[-1]) if np.isfinite(hc.iloc[-1]) else np.nan
    instr_txt = format_trade_instruction(
        trend_slope=m_h, buy_val=sup_val, sell_val=res_val, close_val=px_val,
        symbol=symbol, global_trend_slope=m_h  # local=global in the compressed intraday axis
    )
    ax2.set_title(f"{symbol} Intraday ({st.session_state.get('hour_range','')})  •  {instr_txt}")

    ax2.text(
        0.50, 0.02,
        f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
        transform=ax2.transAxes, ha="center", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )

    # NTD panel
    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX (win={ntd_window})")
        ntd_h = compute_normalized_trend(hc, window=int(ntd_window)) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=int(ntd_window)) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label="NTD")

        if show_npx_ntd and not _coerce_1d_series(npx_h).dropna().empty and not _coerce_1d_series(ntd_h).dropna().empty:
            overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2w.set_ylim(-1.1, 1.1)
        ax2w.set_xlabel("Time (PST)")

    # X ticks readable using real timestamps
    if isinstance(real_times, pd.DatetimeIndex) and len(real_times):
        _apply_compact_time_ticks(ax2, real_times, n_ticks=8)
        if ax2w is not None:
            _apply_compact_time_ticks(ax2w, real_times, n_ticks=8)

    style_axes(ax2)
    if ax2w is not None:
        style_axes(ax2w)

    handles, labels = ax2.get_legend_handles_labels()
    if ax2w is not None:
        h2, l2 = ax2w.get_legend_handles_labels()
        handles += h2; labels += l2

    # De-dupe legend
    seen = set()
    h_u, l_u = [], []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        h_u.append(h)
        l_u.append(l)

    fig2.legend(h_u, l_u, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.01))
    st.pyplot(fig2, use_container_width=True)

    # Optional momentum
    if show_mom_hourly:
        roc = compute_roc(hc, n=int(mom_lb_hourly))
        figm, axm = plt.subplots(figsize=(14, 2.4))
        figm.patch.set_facecolor("white")
        axm.plot(roc.index, roc.values, label=f"ROC% ({mom_lb_hourly})")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.set_title("Hourly Momentum (ROC%)")
        style_axes(axm)
        axm.legend(loc="upper left")
        if isinstance(real_times, pd.DatetimeIndex) and len(real_times):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        st.pyplot(figm, use_container_width=True)
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Tabs (FINAL = 15 tabs)
# Removed:
#   - NPX 0.5-Cross Scanner
#   - Fib NPX 0.0 Signal Scanner
#   - News
#   - Ichimoku Kijun Scanner
# Added:
#   - HMA Buy
# ---------------------------

TAB_NAMES = [
    "Forecast",
    "Enhanced Forecast",
    "Daily Price",
    "Hourly Price",
    "NTD (Daily)",
    "NTD (Hourly)",
    "Bull/Bear",
    "Metrics",
    "Recent BUY",
    "R² Daily",
    "R² Hourly",
    "Band Proximity",
    "Support Reversals",
    "Stickers",
    "HMA Buy",
]

tabs = st.tabs(TAB_NAMES)

# ---------------------------
# Pull session data if available
# ---------------------------
symbol = st.session_state.get("ticker", sel)
df_hist = st.session_state.get("df_hist", None)
df_ohlc = st.session_state.get("df_ohlc", None)
intraday = st.session_state.get("intraday", None)
fc_idx = st.session_state.get("fc_idx", None)
fc_vals = st.session_state.get("fc_vals", None)
fc_ci = st.session_state.get("fc_ci", None)

def _need_run_warn():
    st.info("Click **🔮 Forecast + Charts** in the sidebar to load data and populate tabs.")

# ---------------------------
# Tab 1: Forecast
# ---------------------------
with tabs[0]:
    if not st.session_state.get("run_all", False) or df_hist is None or fc_idx is None:
        _need_run_warn()
    else:
        render_forecast(symbol, df_hist, fc_idx, fc_vals, fc_ci)

# ---------------------------
# Tab 2: Enhanced Forecast (Daily regression + ±2σ)
# ---------------------------
with tabs[1]:
    if not st.session_state.get("run_all", False) or df_hist is None:
        _need_run_warn()
    else:
        render_enhanced_forecast(symbol, df_hist, slope_lb=slope_lb_daily)

# ---------------------------
# Tab 3: Daily Price Chart
# ---------------------------
with tabs[2]:
    if not st.session_state.get("run_all", False) or df_hist is None or df_ohlc is None:
        _need_run_warn()
    else:
        render_daily_price(symbol, df_hist, df_ohlc)

# ---------------------------
# Tab 4: Hourly Price Chart
# ---------------------------
with tabs[3]:
    if not st.session_state.get("run_all", False) or intraday is None:
        _need_run_warn()
    else:
        render_hourly_price(symbol, intraday)

# ---------------------------
# Tab 5: NTD (Daily)
# ---------------------------
with tabs[4]:
    if not st.session_state.get("run_all", False) or df_hist is None:
        _need_run_warn()
    else:
        close_full = _coerce_1d_series(df_hist).dropna()
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        if close.empty:
            st.warning("No daily data.")
        else:
            ntd = compute_normalized_trend(close, window=int(ntd_window))
            npx = compute_normalized_price(close, window=int(ntd_window))
            fig, ax = plt.subplots(figsize=(14, 3.2))
            fig.patch.set_facecolor("white")
            if shade_ntd:
                shade_ntd_regions(ax, ntd)
            ax.plot(ntd.index, ntd.values, "-", linewidth=1.8, label="NTD")
            if show_npx_ntd:
                overlay_npx_on_ntd(ax, npx, ntd, mark_crosses=mark_npx_cross)
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
            ax.set_ylim(-1.1, 1.1)
            ax.set_title(f"{symbol} — NTD (Daily)")
            style_axes(ax)
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)

# ---------------------------
# Tab 6: NTD (Hourly)
# ---------------------------
with tabs[5]:
    if not st.session_state.get("run_all", False) or intraday is None or intraday.empty or "Close" not in intraday.columns:
        _need_run_warn()
    else:
        real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
        intr_plot = intraday.copy()
        intr_plot.index = pd.RangeIndex(len(intr_plot))
        hc = _coerce_1d_series(intr_plot["Close"]).ffill()
        ntd = compute_normalized_trend(hc, window=int(ntd_window))
        npx = compute_normalized_price(hc, window=int(ntd_window))

        fig, ax = plt.subplots(figsize=(14, 3.2))
        fig.patch.set_facecolor("white")
        if shade_ntd:
            shade_ntd_regions(ax, ntd)
        ax.plot(ntd.index, ntd.values, "-", linewidth=1.8, label="NTD (Hourly)")
        if show_npx_ntd:
            overlay_npx_on_ntd(ax, npx, ntd, mark_crosses=mark_npx_cross)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"{symbol} — NTD (Hourly)")
        style_axes(ax)
        ax.legend(loc="upper left")
        if isinstance(real_times, pd.DatetimeIndex) and len(real_times):
            _apply_compact_time_ticks(ax, real_times, n_ticks=8)
        st.pyplot(fig, use_container_width=True)

# ---------------------------
# Tab 7: Bull/Bear
# ---------------------------
with tabs[6]:
    if not st.session_state.get("run_all", False) or df_hist is None:
        _need_run_warn()
    else:
        close = _coerce_1d_series(df_hist).dropna()
        if close.empty:
            st.warning("No history.")
        else:
            try:
                lookback_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}.get(bb_period, 180)
                cut = close.index.max() - pd.Timedelta(days=int(lookback_days))
                seg = close.loc[close.index >= cut]
                if len(seg) < 2:
                    seg = close.tail(60)
            except Exception:
                seg = close.tail(60)

            pct = (float(seg.iloc[-1]) / float(seg.iloc[0]) - 1.0) if np.isfinite(seg.iloc[-1]) and np.isfinite(seg.iloc[0]) and float(seg.iloc[0]) != 0 else np.nan
            st.metric("Bull/Bear Change", fmt_pct(pct, digits=2), help=f"Lookback: {bb_period}")

            fig, ax = plt.subplots(figsize=(14, 3))
            fig.patch.set_facecolor("white")
            ax.plot(seg.index, seg.values, label="Close")
            ax.set_title(f"{symbol} — Bull/Bear ({bb_period})")
            style_axes(ax)
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)

# ---------------------------
# Tab 8: Metrics
# ---------------------------
with tabs[7]:
    if not st.session_state.get("run_all", False) or df_hist is None:
        _need_run_warn()
    else:
        close_full = _coerce_1d_series(df_hist).dropna()
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb_daily))
        st.write("### Daily Metrics")
        c_last = float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan
        st.write(pd.DataFrame([{
            "Symbol": symbol,
            "AsOf": close.index[-1],
            "Close": c_last,
            "Slope": m,
            "R²": r2,
        }]))
        if show_macd:
            macd, sig, hist = compute_macd(close)
            st.write("### MACD (Daily)")
            fig, ax = plt.subplots(figsize=(14, 3))
            fig.patch.set_facecolor("white")
            ax.plot(macd.index, macd.values, label="MACD")
            ax.plot(sig.index, sig.values, label="Signal")
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
            ax.set_title("MACD (Daily)")
            style_axes(ax)
            ax.legend(loc="upper left")
            st.pyplot(fig, use_container_width=True)

# ---------------------------
# Tab 9: Recent BUY (Band Bounce Scanner)
# ---------------------------
with tabs[8]:
    st.write("### Recent BUY/SELL — Band Bounce (Daily)")
    scan_btn = st.button("Scan Universe (Band Bounce)", use_container_width=True, key="btn_scan_band_bounce")
    if not scan_btn:
        st.caption("Click the scan button to compute signals for the current universe.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            row = last_band_bounce_signal_daily(sym, slope_lb=int(slope_lb_daily), daily_view_label=daily_view)
            if row is not None:
                rows.append(row)
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()
        if not rows:
            st.info("No recent band-bounce signals found.")
        else:
            df = pd.DataFrame(rows).sort_values(["Side","Bars Since","R2"], ascending=[True, True, False])
            st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 10: R² Daily
# ---------------------------
with tabs[9]:
    st.write("### R² (Daily) — Regression Fit Strength")
    scan_btn = st.button("Scan Universe (R² Daily)", use_container_width=True, key="btn_scan_r2_daily")
    if not scan_btn:
        st.caption("Click the scan button to compute daily R² across the universe.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            r2, m, ts = daily_regression_r2(sym, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2):
                rows.append({"Symbol": sym, "R2": r2, "Slope": m, "AsOf": ts})
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()
        if not rows:
            st.info("No results.")
        else:
            df = pd.DataFrame(rows).sort_values("R2", ascending=False)
            st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 11: R² Hourly
# ---------------------------
with tabs[10]:
    st.write("### R² (Hourly) — Regression Fit Strength")
    scan_btn = st.button("Scan Universe (R² Hourly)", use_container_width=True, key="btn_scan_r2_hourly")
    if not scan_btn:
        st.caption("Click the scan button to compute hourly R² across the universe.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            r2, m, ts = hourly_regression_r2(sym, period=st.session_state.get("hour_range","1d"), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2):
                rows.append({"Symbol": sym, "R2": r2, "Slope": m, "AsOf": ts})
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()
        if not rows:
            st.info("No results.")
        else:
            df = pd.DataFrame(rows).sort_values("R2", ascending=False)
            st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 12: Band Proximity (Daily ±2σ proximity)
# ---------------------------
with tabs[11]:
    st.write("### Daily ±2σ Band Proximity (Regression Bands)")
    scan_btn = st.button("Scan Universe (Band Proximity)", use_container_width=True, key="btn_scan_band_prox")
    if not scan_btn:
        st.caption("Find symbols near the daily regression ±2σ bands.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            row = daily_r2_band_proximity(sym, daily_view_label=daily_view, slope_lb=int(slope_lb_daily), prox=float(sr_prox_pct))
            if row is not None and (row.get("Near Lower") or row.get("Near Upper")):
                rows.append(row)
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()
        if not rows:
            st.info("No symbols near ±2σ bands within the selected proximity.")
        else:
            df = pd.DataFrame(rows).sort_values(["Near Lower","Near Upper","R2"], ascending=[False, False, False])
            st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 13: Support Reversals (Daily)
# ---------------------------
with tabs[12]:
    st.write("### Support Reversal (Daily) — Heading Up Confirmation")
    confirm_bars = st.slider("Confirm bars (consecutive higher closes)", 1, 4, 2, 1, key="sr_confirm_bars")
    scan_btn = st.button("Scan Universe (Support Reversal)", use_container_width=True, key="btn_scan_support_rev")
    if not scan_btn:
        st.caption("Find symbols that recently touched support and are now reversing upward.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            row = daily_support_reversal_heading_up(
                sym, daily_view_label=daily_view, sr_lb=int(sr_lb_daily), prox=float(sr_prox_pct), confirm_bars=int(confirm_bars)
            )
            if row is not None:
                rows.append(row)
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()
        if not rows:
            st.info("No confirmed support reversals found.")
        else:
            df = pd.DataFrame(rows).sort_values(["Bars Since Touch"], ascending=True)
            st.dataframe(df, use_container_width=True)

# ---------------------------
# Tab 14: Stickers (FIXED)
# ---------------------------
with tabs[13]:
    st.write("### Stickers (Quick Snapshot)")
    st.caption("A compact snapshot for the selected symbol (no extra scanners added here).")
    if not st.session_state.get("run_all", False) or df_hist is None:
        _need_run_warn()
    else:
        close_full = _coerce_1d_series(df_hist).dropna()
        close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
        yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb_daily))
        ntd = compute_normalized_trend(close, window=int(ntd_window))
        npx = compute_normalized_price(close, window=int(ntd_window))
        hma = compute_hma(close, period=int(hma_period))

        # ✅ FIX: st.write(...) does NOT accept use_container_width
        # Use st.dataframe for width control.
        df_stickers = pd.DataFrame([{
            "Symbol": symbol,
            "AsOf": close.index[-1],
            "Close": float(close.iloc[-1]) if np.isfinite(close.iloc[-1]) else np.nan,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R²": float(r2) if np.isfinite(r2) else np.nan,
            "NTD": float(_coerce_1d_series(ntd).dropna().iloc[-1]) if len(_coerce_1d_series(ntd).dropna()) else np.nan,
            "NPX": float(_coerce_1d_series(npx).dropna().iloc[-1]) if len(_coerce_1d_series(npx).dropna()) else np.nan,
            f"HMA({hma_period})": float(_coerce_1d_series(hma).dropna().iloc[-1]) if len(_coerce_1d_series(hma).dropna()) else np.nan,
        }])
        st.dataframe(df_stickers, use_container_width=True)

# ---------------------------
# Tab 15: HMA Buy (NEW)
# ---------------------------
with tabs[14]:
    st.write("## HMA Buy (Daily HMA(55) Cross-Up)")
    st.caption("Lists symbols where price **recently crossed ABOVE HMA(55)** on the Daily chart within **1–3 bars**.")
    bars_since = st.slider("Bars since HMA cross (1–3)", 1, 3, 3, 1, key="hma_buy_bars_since")
    scan_btn = st.button("Scan Universe (HMA Buy)", use_container_width=True, key="btn_scan_hma_buy")

    if not scan_btn:
        st.caption("Click scan to build the HMA Buy lists.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            row = last_daily_hma_cross_up(
                sym,
                daily_view_label=daily_view,
                hma_len=55,                 # fixed to HMA(55) per request
                slope_lb=int(slope_lb_daily),
                within_last_n_bars=int(bars_since)
            )
            if row is not None:
                rows.append(row)
            prog.progress(int(((i+1)/max(1,len(universe)))*100))
        prog.empty()

        if not rows:
            st.info("No symbols found with an HMA(55) cross-up within the selected bars.")
        else:
            df = pd.DataFrame(rows)
            df_pos = df[df["Slope"] > 0].copy()
            df_neg = df[df["Slope"] < 0].copy()

            st.write("### (a) Regression > 0")
            if df_pos.empty:
                st.caption("No matches with Regression > 0.")
            else:
                df_pos = df_pos.sort_values(["Bars Since Cross","R2"], ascending=[True, False])
                st.dataframe(df_pos, use_container_width=True)

            st.write("### (b) Regression < 0")
            if df_neg.empty:
                st.caption("No matches with Regression < 0.")
            else:
                df_neg = df_neg.sort_values(["Bars Since Cross","R2"], ascending=[True, False])
                st.dataframe(df_neg, use_container_width=True)
