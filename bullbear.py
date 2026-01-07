# =========================
# Part 1/10 — bullbear.py
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
# Aesthetic helper (no logic change)
# ---------------------------
def style_axes(ax):
    """Simple, consistent, user-friendly chart styling."""
    try:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
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

# NEW (THIS REQUEST): Fibonacci-specific alert instruction
FIB_ALERT_TEXT = "ALERT: Fibonacci guidance — BUY close to the 100% line and SELL close to the 0% line."

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

# UPDATED (THIS REQUEST): global trendline show/hide toggle button (default ON)
if "show_global_trendline" not in st.session_state:
    st.session_state.show_global_trendline = True

st.sidebar.subheader("Global Trendline")
if st.sidebar.button("Toggle global trendline", use_container_width=True):
    st.session_state.show_global_trendline = not bool(st.session_state.show_global_trendline)
st.sidebar.caption(f"Global trendline: **{'ON' if st.session_state.show_global_trendline else 'OFF'}**")

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

# UPDATED (THIS REQUEST): Fibonacci applies to Daily + Hourly, default ON
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
# UPDATED (THIS REQUEST): use a new key so Daily NTD displays ON by default again
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
        "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
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

# NEW (THIS REQUEST): Trigger that confirms reversal from Fib 0% / 100%
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

def shade_ntd_regions(ax, ntd: pd.Series):
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

# UPDATED (THIS REQUEST): optional show/hide rendering for the global trendline
def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend", show_line: bool = True):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "green" if m >= 0 else "red"
    if bool(show_line):
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

def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd).dropna()
    if s.empty or not np.isfinite(trend_slope):
        return
    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    cross_up0 = (s >= 0.0) & (s.shift(1) < 0.0)
    cross_dn0 = (s <= 0.0) & (s.shift(1) > 0.0)
    idx_up0 = list(cross_up0[cross_up0].index)
    idx_dn0 = list(cross_dn0[cross_dn0].index)

    cross_out_hi = (s >= upper) & (s.shift(1) < upper)
    cross_out_lo = (s <= lower) & (s.shift(1) > lower)
    idx_hi = list(cross_out_hi[cross_out_hi].index)
    idx_lo = list(cross_out_lo[cross_out_lo].index)

    if uptrend:
        if idx_up0:
            ax.scatter(idx_up0, [0.0]*len(idx_up0), marker="^", s=95, color="tab:green", zorder=10, label="NTD 0↑")
        if idx_lo:
            ax.scatter(idx_lo, s.loc[idx_lo], marker="^", s=85, color="tab:green", zorder=10, label="NTD < -0.75")
    if downtrend:
        if idx_dn0:
            ax.scatter(idx_dn0, [0.0]*len(idx_dn0), marker="v", s=95, color="tab:red", zorder=10, label="NTD 0↓")
        if idx_hi:
            ax.scatter(idx_hi, s.loc[idx_hi], marker="v", s=85, color="tab:red", zorder=10, label="NTD > +0.75")

def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas < 0))

def overlay_ntd_sr_reversal_stars(ax,
                                 price: pd.Series,
                                 sup: pd.Series,
                                 res: pd.Series,
                                 trend_slope: float,
                                 ntd: pd.Series,
                                 prox: float = 0.0025,
                                 bars_confirm: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        return
    s_sup = _coerce_1d_series(sup).reindex(p.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(p.index).ffill().bfill()
    s_ntd = _coerce_1d_series(ntd).reindex(p.index)

    t = p.index[-1]
    if not (t in s_sup.index and t in s_res.index and t in s_ntd.index):
        return
    c0 = float(p.iloc[-1])
    c1 = float(p.iloc[-2]) if len(p) >= 2 else np.nan
    S0 = float(s_sup.loc[t]) if pd.notna(s_sup.loc[t]) else np.nan
    R0 = float(s_res.loc[t]) if pd.notna(s_res.loc[t]) else np.nan
    ntd0 = float(s_ntd.loc[t]) if pd.notna(s_ntd.loc[t]) else np.nan
    if not np.all(np.isfinite([c0, S0, R0, ntd0])):
        return

    near_support = c0 <= S0 * (1.0 + prox)
    near_resist  = c0 >= R0 * (1.0 - prox)

    toward_res = toward_sup = False
    if np.isfinite(c1):
        toward_res = (R0 - c0) < (R0 - c1)
        toward_sup = (c0 - S0) < (c1 - S0)

    buy_cond  = (trend_slope > 0) and near_support and _n_consecutive_increasing(p, bars_confirm) and toward_res
    sell_cond = (trend_slope < 0) and near_resist  and _n_consecutive_decreasing(p, bars_confirm) and toward_sup

    if buy_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:green", zorder=12, label="BUY ★ (Support reversal)")
    if sell_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:red", zorder=12, label="SELL ★ (Resistance reversal)")
# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Support / Resistance series
# ---------------------------
def compute_support_resistance(close: pd.Series, lookback: int = 60):
    c = _coerce_1d_series(close).astype(float)
    if c.empty or lookback < 2:
        empty = pd.Series(index=c.index, dtype=float)
        return empty, empty
    lb = int(max(2, lookback))
    sup = c.rolling(lb, min_periods=max(2, lb//3)).min()
    res = c.rolling(lb, min_periods=max(2, lb//3)).max()
    return sup.reindex(c.index), res.reindex(c.index)

def plot_sr_lines(ax, sup: pd.Series, res: pd.Series, price: pd.Series, label_prefix: str = ""):
    s_sup = _coerce_1d_series(sup).reindex(price.index)
    s_res = _coerce_1d_series(res).reindex(price.index)
    if s_sup.dropna().empty or s_res.dropna().empty:
        return np.nan, np.nan
    try:
        S0 = float(s_sup.dropna().iloc[-1])
        R0 = float(s_res.dropna().iloc[-1])
    except Exception:
        return np.nan, np.nan
    if np.isfinite(S0):
        ax.axhline(S0, linestyle=":", linewidth=1.5, color="tab:green",
                   alpha=0.85, label=f"{label_prefix}Support")
        label_on_left(ax, S0, f"S {fmt_price_val(S0)}", color="tab:green")
    if np.isfinite(R0):
        ax.axhline(R0, linestyle=":", linewidth=1.5, color="tab:red",
                   alpha=0.85, label=f"{label_prefix}Resistance")
        label_on_left(ax, R0, f"R {fmt_price_val(R0)}", color="tab:red")
    return S0, R0

# ---------------------------
# Fibonacci plotting helper
# ---------------------------
def plot_fibonacci(ax, fibs: dict, x0=None, x1=None, alpha: float = 0.55):
    if not fibs:
        return
    keys = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
    for k in keys:
        if k not in fibs:
            continue
        y = fibs[k]
        if not np.isfinite(y):
            continue
        ax.axhline(y, linestyle="--", linewidth=1.0, alpha=alpha, color="tab:blue")
        try:
            if x0 is not None and x1 is not None:
                ax.text(x1, y, f" {k}", va="center", fontsize=8, alpha=0.85)
        except Exception:
            pass

# ---------------------------
# Sessions (PST) for Forex intraday chart
# ---------------------------
def _session_times_utc(day_utc: pd.Timestamp):
    """
    Approximate major FX session windows in UTC.
    (These are broad/typical windows and can vary by market conventions.)
    """
    d = pd.Timestamp(day_utc.date(), tz="UTC")
    london_open = d + pd.Timedelta(hours=7)
    london_close = d + pd.Timedelta(hours=16)
    ny_open = d + pd.Timedelta(hours=13)
    ny_close = d + pd.Timedelta(hours=22)
    return (london_open, london_close, ny_open, ny_close)

def overlay_fx_sessions_pst(ax, idx_pst: pd.DatetimeIndex):
    if idx_pst is None or len(idx_pst) == 0:
        return
    if idx_pst.tz is None:
        try:
            idx_pst = idx_pst.tz_localize(PACIFIC)
        except Exception:
            pass

    start = idx_pst.min()
    end = idx_pst.max()
    start_utc = start.tz_convert("UTC")
    end_utc = end.tz_convert("UTC")

    days = pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D", tz="UTC")
    for d in days:
        lo, lc, no, nc = _session_times_utc(d)
        lo_pst = lo.tz_convert(PACIFIC); lc_pst = lc.tz_convert(PACIFIC)
        no_pst = no.tz_convert(PACIFIC); nc_pst = nc.tz_convert(PACIFIC)

        for t, lab in [(lo_pst, "London Open"), (lc_pst, "London Close"),
                       (no_pst, "NY Open"), (nc_pst, "NY Close")]:
            if t < start or t > end:
                continue
            ax.axvline(t, linestyle=":", linewidth=1.0, alpha=0.35, color="tab:gray")
    ax.text(0.01, 0.98, "Sessions shown in PST", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, alpha=0.75)

# ---------------------------
# UI: Ticker selection + run button
# ---------------------------
def _on_ticker_change():
    st.session_state.run_all = False
    st.session_state.pop("df_hist", None)
    st.session_state.pop("df_ohlc", None)
    st.session_state.pop("intraday", None)
    st.session_state.pop("fc_idx", None)
    st.session_state.pop("fc_vals", None)
    st.session_state.pop("fc_ci", None)
    st.session_state.pop("chart", None)
    st.session_state.pop("mode_at_run", None)

default_ticker = universe[0] if len(universe) else ""
curr_ticker = st.session_state.get("ticker", default_ticker)
if curr_ticker not in universe:
    curr_ticker = default_ticker

st.session_state.ticker = st.selectbox(
    "Select symbol",
    options=universe,
    index=int(universe.index(curr_ticker)) if curr_ticker in universe else 0,
    key="ticker_select",
    on_change=_on_ticker_change,
)

b1, b2, b3 = st.columns([1, 1, 2])
if b1.button("🚀 Run / Update", use_container_width=True, key="btn_run_all"):
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode
if b2.button("🧹 Reset view", use_container_width=True, key="btn_reset_view"):
    _reset_run_state_for_mode_switch()
    st.session_state.ticker = default_ticker
    try:
        st.experimental_rerun()
    except Exception:
        pass
b3.caption("Tip: switching modes resets run state automatically.")

ticker = st.session_state.ticker


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Main execution
# ---------------------------
def _period_to_days(p: str) -> int:
    p = str(p).lower().strip()
    if p == "1mo":
        return 21
    if p == "3mo":
        return 63
    if p == "6mo":
        return 126
    if p == "1y":
        return 252
    return 126

if not st.session_state.get("run_all", False):
    st.info("Select a symbol and click **Run / Update** to load charts and signals.")
else:
    with st.spinner("Fetching data..."):
        try:
            close_daily = fetch_hist(ticker)
            ohlc_daily = fetch_hist_ohlc(ticker)
            close_max = fetch_hist_max(ticker)
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            close_daily = pd.Series(dtype=float)
            ohlc_daily = pd.DataFrame()

        intraday_period = "5d"
        try:
            intraday = fetch_intraday(ticker, period=intraday_period)
        except Exception:
            intraday = pd.DataFrame()

    st.session_state.df_hist = close_daily
    st.session_state.df_ohlc = ohlc_daily
    st.session_state.intraday = intraday

    # ---------------------------
    # Quick bull/bear summary
    # ---------------------------
    st.subheader("Snapshot")

    if close_daily is None or close_daily.dropna().empty:
        st.warning("No daily price data available for this symbol.")
    else:
        last_close = float(close_daily.dropna().iloc[-1])
        prev_close = float(close_daily.dropna().iloc[-2]) if len(close_daily.dropna()) >= 2 else np.nan
        chg = (last_close / prev_close - 1.0) if np.isfinite(prev_close) and prev_close != 0 else np.nan

        d_look = _period_to_days(bb_period)
        s_for_bb = close_daily.dropna()
        if len(s_for_bb) > d_look:
            bb_ret = s_for_bb.iloc[-1] / s_for_bb.iloc[-(d_look+1)] - 1.0
        else:
            bb_ret = np.nan
        bb_state = "BULL" if np.isfinite(bb_ret) and bb_ret >= 0 else "BEAR"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", fmt_price_val(last_close))
        c2.metric("Daily Change", fmt_pct(chg, 2))
        c3.metric(f"Return ({bb_period})", fmt_pct(bb_ret, 2))
        c4.metric("Bull/Bear", bb_state)

    st.divider()

    # ---------------------------
    # Compute global trend slope (from max history)
    # ---------------------------
    global_trend_slope = np.nan
    try:
        if close_max is not None and close_max.dropna().shape[0] >= 3:
            # just compute the slope; plotting is optional per toggle
            _, global_trend_slope = slope_line(close_max.dropna(), lookback=len(close_max.dropna()))
    except Exception:
        global_trend_slope = np.nan

    # ---------------------------
    # Daily view subset
    # ---------------------------
    close_view = close_daily
    ohlc_view = ohlc_daily
    if close_view is not None and not close_view.empty:
        close_view = subset_by_daily_view(close_view, daily_view)
    if ohlc_view is not None and not ohlc_view.empty:
        ohlc_view = subset_by_daily_view(ohlc_view, daily_view)

    if close_view is None or close_view.dropna().empty:
        st.warning("Daily view is empty after filtering.")
        close_view = close_daily

    # ---------------------------
    # Support / Resistance (Daily)
    # ---------------------------
    sup_d, res_d = compute_support_resistance(close_view, lookback=sr_lb_daily)
    S_d = _safe_last_float(sup_d)
    R_d = _safe_last_float(res_d)

    # ---------------------------
    # Local regression (Daily)
    # ---------------------------
    yhat_d, upper_d, lower_d, slope_d, r2_d = regression_with_band(close_view, lookback=slope_lb_daily, z=2.0)
    rev_prob_d = slope_reversal_probability(
        close_view, current_slope=slope_d,
        hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon
    )

    # ---------------------------
    # Daily indicators
    # ---------------------------
    macd_d, sig_d, hist_d = compute_macd(close_view)
    hma_d = compute_hma(close_view, period=hma_period) if show_hma else pd.Series(index=close_view.index, dtype=float)

    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(
        close_view, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    ) if show_bbands else (pd.Series(index=close_view.index, dtype=float),)*5

    if ohlc_view is not None and not ohlc_view.empty and show_ichi:
        tenkan_d, kijun_d, senA_d, senB_d, chikou_d = ichimoku_lines(
            ohlc_view["High"], ohlc_view["Low"], ohlc_view["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
        )
    else:
        kijun_d = pd.Series(index=close_view.index, dtype=float)

    # PSAR (Daily) if possible
    psar_d = pd.Series(index=close_view.index, dtype=float)
    if show_psar and ohlc_view is not None and not ohlc_view.empty:
        try:
            ps = compute_psar_from_ohlc(ohlc_view, step=psar_step, max_step=psar_max)
            psar_d = ps["PSAR"].reindex(close_view.index)
        except Exception:
            psar_d = pd.Series(index=close_view.index, dtype=float)

    # Fibonacci (Daily)
    fibs_d = fibonacci_levels(close_view) if show_fibs else {}
    fib_trig_d = fib_reversal_trigger_from_extremes(
        close_view, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=max(60, slope_lb_daily)
    ) if show_fibs else None

    # Daily Signals
    band_sig_d = find_band_bounce_signal(close_view, upper_d, lower_d, slope_d)
    slope_trig_d = find_slope_trigger_after_band_reversal(close_view, yhat_d, upper_d, lower_d, horizon=rev_horizon)

    macd_hma_sig_d = None
    if show_hma and close_view is not None and not close_view.empty:
        macd_hma_sig_d = find_macd_hma_sr_signal(
            close=close_view, hma=hma_d, macd=macd_d, sup=sup_d, res=res_d,
            global_trend_slope=global_trend_slope, prox=sr_prox_pct
        )

    # Trade instruction (Daily)
    try:
        close_last = float(close_view.dropna().iloc[-1])
    except Exception:
        close_last = float("nan")
    trade_text_daily = format_trade_instruction(
        trend_slope=slope_d,
        buy_val=S_d,
        sell_val=R_d,
        close_val=close_last,
        symbol=ticker,
        global_trend_slope=global_trend_slope
    )


# =========================
# Part 8/10 — bullbear.py
# =========================
    # ---------------------------
    # DAILY PRICE CHART
    # ---------------------------
    st.subheader("Daily Price (with Local ±2σ Band, S/R, and Optional Fib)")
    if show_fibs:
        st.caption(FIB_ALERT_TEXT)

    fig_d, ax_d = plt.subplots(figsize=(12, 5))
    style_axes(ax_d)

    # Price
    ax_d.plot(close_view.index, close_view.values, linewidth=1.6, label="Close")

    # Optional global trendline
    try:
        if close_max is not None and close_max.dropna().shape[0] >= 3:
            _ = draw_trend_direction_line(ax_d, close_max.dropna(), label_prefix="Global", show_line=st.session_state.show_global_trendline)
    except Exception:
        pass

    # Local regression band (on last lookback bars)
    if yhat_d is not None and not yhat_d.dropna().empty:
        ax_d.plot(yhat_d.index, yhat_d.values, linestyle="--", linewidth=2.2, label=f"Local trend ({fmt_slope(slope_d)}/bar, R² {fmt_r2(r2_d,1)})")
        ax_d.plot(upper_d.index, upper_d.values, linestyle="--", linewidth=1.1, alpha=0.9, label="+2σ")
        ax_d.plot(lower_d.index, lower_d.values, linestyle="--", linewidth=1.1, alpha=0.9, label="-2σ")

    # Bollinger bands
    if show_bbands and bb_up is not None and not bb_up.dropna().empty:
        ax_d.plot(bb_up.index, bb_up.values, linewidth=1.0, alpha=0.8, label="BB Upper")
        ax_d.plot(bb_lo.index, bb_lo.values, linewidth=1.0, alpha=0.8, label="BB Lower")
        ax_d.plot(bb_mid.index, bb_mid.values, linewidth=1.0, alpha=0.8, label="BB Mid")

    # Kijun
    if show_ichi and kijun_d is not None and not kijun_d.dropna().empty:
        ax_d.plot(kijun_d.index, kijun_d.values, linewidth=1.2, alpha=0.9, label="Kijun")

    # PSAR dots
    if show_psar and psar_d is not None and not psar_d.dropna().empty:
        ax_d.scatter(psar_d.index, psar_d.values, s=14, alpha=0.75, label="PSAR")

    # Support / Resistance
    plot_sr_lines(ax_d, sup_d, res_d, close_view, label_prefix="Daily ")

    # Fibonacci levels (Daily)
    if show_fibs and fibs_d:
        try:
            x0 = close_view.index.min()
            x1 = close_view.index.max()
        except Exception:
            x0 = x1 = None
        plot_fibonacci(ax_d, fibs_d, x0=x0, x1=x1, alpha=0.45)

    # Band bounce signal marker
    if band_sig_d is not None:
        annotate_crossover(ax_d, band_sig_d["time"], band_sig_d["price"], band_sig_d["side"], note="(Band bounce)")

    # Slope trigger marker + leaderline
    annotate_slope_trigger(ax_d, slope_trig_d)

    # MACD/HMA/SR star
    if macd_hma_sig_d is not None:
        annotate_macd_signal(ax_d, macd_hma_sig_d["time"], macd_hma_sig_d["price"], macd_hma_sig_d["side"])

    # Fibonacci reversal trigger marker
    if show_fibs and fib_trig_d is not None:
        side = fib_trig_d.get("side", "")
        t = fib_trig_d.get("last_time")
        px = fib_trig_d.get("last_price", np.nan)
        if side == "BUY":
            ax_d.scatter([t], [px], marker="^", s=110, color="tab:green", zorder=11, label="Fib Trigger (BUY)")
            ax_d.text(t, px, "  Fib BUY", color="tab:green", fontsize=9, fontweight="bold", va="bottom")
        elif side == "SELL":
            ax_d.scatter([t], [px], marker="v", s=110, color="tab:red", zorder=11, label="Fib Trigger (SELL)")
            ax_d.text(t, px, "  Fib SELL", color="tab:red", fontsize=9, fontweight="bold", va="top")

    ax_d.set_title(f"{ticker} — Daily")
    ax_d.set_ylabel("Price")
    ax_d.legend(loc="best", fontsize=8)

    st.pyplot(fig_d, use_container_width=True)

    # ---------------------------
    # Daily Summary + Instruction
    # ---------------------------
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Global slope (max history)", fmt_slope(global_trend_slope))
    s2.metric("Local slope (daily)", fmt_slope(slope_d))
    s3.metric("Local fit R² (daily)", fmt_r2(r2_d, 1))
    s4.metric("Slope reversal prob (daily)", fmt_pct(rev_prob_d, 1))

    if trade_text_daily.strip().startswith("ALERT"):
        st.warning(trade_text_daily)
    else:
        st.success(trade_text_daily)

    if show_fibs and fib_trig_d is not None:
        lvl = fib_trig_d.get("from_level", "")
        side = fib_trig_d.get("side", "")
        tt = fib_trig_d.get("touch_time")
        st.info(f"Trigger 1 (Daily) — Confirmed reversal from Fibonacci **{lvl}**: **{side}** (last touch: {tt})")
    elif show_fibs:
        st.caption("Trigger 1 (Daily) — No confirmed Fibonacci 0%/100% reversal yet.")

    # ---------------------------
    # DAILY NTD PANEL
    # ---------------------------
    if show_ntd:
        st.subheader("NTD (Daily) + Signals")

        ntd_d = compute_normalized_trend(close_view, window=ntd_window)
        npx_d = compute_normalized_price(close_view, window=ntd_window)

        fig_ntd, ax_ntd = plt.subplots(figsize=(12, 3))
        style_axes(ax_ntd)

        ax_ntd.plot(ntd_d.index, ntd_d.values, linewidth=1.6, label="NTD")
        ax_ntd.axhline(0.0, linewidth=1.0, alpha=0.5)
        ax_ntd.axhline(0.75, linestyle="--", linewidth=0.9, alpha=0.6)
        ax_ntd.axhline(-0.75, linestyle="--", linewidth=0.9, alpha=0.6)

        if shade_ntd:
            shade_ntd_regions(ax_ntd, ntd_d)

        # Triangles based on DAILY local slope
        overlay_ntd_triangles_by_trend(ax_ntd, ntd_d, trend_slope=slope_d)

        # Stars based on S/R reversal confirmation
        overlay_ntd_sr_reversal_stars(
            ax_ntd,
            price=close_view,
            sup=sup_d,
            res=res_d,
            trend_slope=slope_d,
            ntd=ntd_d,
            prox=sr_prox_pct,
            bars_confirm=rev_bars_confirm
        )

        # HMA reversal markers on NTD
        if show_hma_rev_ntd and show_hma:
            overlay_hma_reversal_on_ntd(
                ax_ntd, price=close_view, hma=hma_d,
                lookback=hma_rev_lb, period=hma_period, ntd=ntd_d
            )

        # NPX overlay and cross dots
        if show_npx_ntd:
            overlay_npx_on_ntd(ax_ntd, npx=npx_d, ntd=ntd_d, mark_crosses=mark_npx_cross)

        ax_ntd.set_ylim(-1.05, 1.05)
        ax_ntd.set_title(f"{ticker} — NTD (Daily)")
        ax_ntd.legend(loc="best", fontsize=8)
        st.pyplot(fig_ntd, use_container_width=True)

    # ---------------------------
    # Optional MACD panel (Daily)
    # ---------------------------
    if show_macd:
        st.subheader("MACD (Daily)")
        fig_m, ax_m = plt.subplots(figsize=(12, 3))
        style_axes(ax_m)
        ax_m.plot(macd_d.index, macd_d.values, linewidth=1.4, label="MACD")
        ax_m.plot(sig_d.index, sig_d.values, linewidth=1.2, label="Signal")
        ax_m.axhline(0.0, linewidth=1.0, alpha=0.5)
        ax_m.set_title(f"{ticker} — MACD (Daily)")
        ax_m.legend(loc="best", fontsize=8)
        st.pyplot(fig_m, use_container_width=True)


# =========================
# Part 9/10 — bullbear.py
# =========================
    # ---------------------------
    # HOURLY (from intraday 5m resampled)
    # ---------------------------
    st.subheader("Hourly (Resampled from 5m Intraday)")

    if intraday is None or intraday.empty or "Close" not in intraday.columns:
        st.warning("No intraday data available for hourly analysis.")
    else:
        # Hour range selector
        hr_options = [6, 12, 24, 48, 72, 120]
        default_hr = 48
        if "hour_range" not in st.session_state:
            st.session_state.hour_range = default_hr

        st.session_state.hour_range = st.selectbox(
            "Show last N hours",
            options=hr_options,
            index=int(hr_options.index(st.session_state.hour_range)) if st.session_state.hour_range in hr_options else hr_options.index(default_hr),
            key="hour_range_select"
        )
        hour_range = int(st.session_state.hour_range)

        # Resample to hourly OHLC
        df = intraday.copy().dropna(subset=["Close"])
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize(PACIFIC)
            except Exception:
                pass

        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"

        hourly = df.resample("1H").agg(agg).dropna(subset=["Close"])
        if hourly.empty:
            st.warning("Hourly resample produced no data.")
        else:
            hourly = hourly.iloc[-hour_range:] if len(hourly) > hour_range else hourly
            close_h = hourly["Close"]

            # Hourly S/R, regression, probabilities
            sup_h, res_h = compute_support_resistance(close_h, lookback=sr_lb_hourly)
            S_h = _safe_last_float(sup_h)
            R_h = _safe_last_float(res_h)

            yhat_h, upper_h, lower_h, slope_h, r2_h = regression_with_band(close_h, lookback=slope_lb_hourly, z=2.0)
            rev_prob_h = slope_reversal_probability(
                close_h, current_slope=slope_h,
                hist_window=rev_hist_lb, slope_window=slope_lb_hourly, horizon=rev_horizon
            )

            # Hourly fibs + trigger
            fibs_h = fibonacci_levels(close_h) if show_fibs else {}
            fib_trig_h = fib_reversal_trigger_from_extremes(
                close_h, proximity_pct_of_range=0.02, confirm_bars=rev_bars_confirm, lookback_bars=max(60, slope_lb_hourly)
            ) if show_fibs else None

            # Hourly supertrend + PSAR
            st_h = pd.DataFrame()
            if {"High", "Low", "Close"}.issubset(hourly.columns):
                if atr_period and atr_mult:
                    try:
                        st_h = compute_supertrend(hourly[["High", "Low", "Close"]], atr_period=atr_period, atr_mult=atr_mult)
                    except Exception:
                        st_h = pd.DataFrame()
            psar_h = pd.Series(index=close_h.index, dtype=float)
            if show_psar and {"High", "Low"}.issubset(hourly.columns):
                try:
                    ps = compute_psar_from_ohlc(hourly[["High", "Low"]].join(hourly[["Close"]], how="left"), step=psar_step, max_step=psar_max)
                    psar_h = ps["PSAR"].reindex(close_h.index)
                except Exception:
                    psar_h = pd.Series(index=close_h.index, dtype=float)

            # Hourly price chart
            fig_h, ax_h = plt.subplots(figsize=(12, 5))
            style_axes(ax_h)
            ax_h.plot(close_h.index, close_h.values, linewidth=1.6, label="Close (Hourly)")

            # Local regression band
            if yhat_h is not None and not yhat_h.dropna().empty:
                ax_h.plot(yhat_h.index, yhat_h.values, linestyle="--", linewidth=2.0,
                          label=f"Local trend ({fmt_slope(slope_h)}/bar, R² {fmt_r2(r2_h,1)})")
                ax_h.plot(upper_h.index, upper_h.values, linestyle="--", linewidth=1.1, alpha=0.9, label="+2σ")
                ax_h.plot(lower_h.index, lower_h.values, linestyle="--", linewidth=1.1, alpha=0.9, label="-2σ")

            # S/R
            plot_sr_lines(ax_h, sup_h, res_h, close_h, label_prefix="Hourly ")

            # Supertrend
            if not st_h.empty and "ST" in st_h.columns:
                ax_h.plot(st_h.index, st_h["ST"].values, linewidth=1.3, alpha=0.9, label="Supertrend")

            # PSAR
            if show_psar and psar_h is not None and not psar_h.dropna().empty:
                ax_h.scatter(psar_h.index, psar_h.values, s=14, alpha=0.75, label="PSAR")

            # Fibonacci levels (Hourly)
            if show_fibs and fibs_h:
                try:
                    x0 = close_h.index.min()
                    x1 = close_h.index.max()
                except Exception:
                    x0 = x1 = None
                plot_fibonacci(ax_h, fibs_h, x0=x0, x1=x1, alpha=0.45)

            # Fibonacci reversal trigger marker (Hourly)
            if show_fibs and fib_trig_h is not None:
                side = fib_trig_h.get("side", "")
                t = fib_trig_h.get("last_time")
                px = fib_trig_h.get("last_price", np.nan)
                if side == "BUY":
                    ax_h.scatter([t], [px], marker="^", s=110, color="tab:green", zorder=11, label="Fib Trigger (BUY)")
                    ax_h.text(t, px, "  Fib BUY", color="tab:green", fontsize=9, fontweight="bold", va="bottom")
                elif side == "SELL":
                    ax_h.scatter([t], [px], marker="v", s=110, color="tab:red", zorder=11, label="Fib Trigger (SELL)")
                    ax_h.text(t, px, "  Fib SELL", color="tab:red", fontsize=9, fontweight="bold", va="top")

            # Forex sessions overlay (PST)
            if mode == "Forex" and show_sessions_pst:
                overlay_fx_sessions_pst(ax_h, close_h.index)

            ax_h.set_title(f"{ticker} — Hourly")
            ax_h.set_ylabel("Price")
            ax_h.legend(loc="best", fontsize=8)
            st.pyplot(fig_h, use_container_width=True)

            # Hourly summary + instruction
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Global slope (max history)", fmt_slope(global_trend_slope))
            h2.metric("Local slope (hourly)", fmt_slope(slope_h))
            h3.metric("Local fit R² (hourly)", fmt_r2(r2_h, 1))
            h4.metric("Slope reversal prob (hourly)", fmt_pct(rev_prob_h, 1))

            trade_text_hourly = format_trade_instruction(
                trend_slope=slope_h,
                buy_val=S_h,
                sell_val=R_h,
                close_val=float(close_h.dropna().iloc[-1]) if close_h.dropna().shape[0] else np.nan,
                symbol=ticker,
                global_trend_slope=global_trend_slope
            )
            if trade_text_hourly.strip().startswith("ALERT"):
                st.warning(trade_text_hourly)
            else:
                st.success(trade_text_hourly)

            if show_fibs and fib_trig_h is not None:
                lvl = fib_trig_h.get("from_level", "")
                side = fib_trig_h.get("side", "")
                tt = fib_trig_h.get("touch_time")
                st.info(f"Trigger 1 (Hourly) — Confirmed reversal from Fibonacci **{lvl}**: **{side}** (last touch: {tt})")
            elif show_fibs:
                st.caption("Trigger 1 (Hourly) — No confirmed Fibonacci 0%/100% reversal yet.")

            # Hourly momentum
            if show_mom_hourly:
                st.subheader("Hourly Momentum (ROC%)")
                roc_h = compute_roc(close_h, n=mom_lb_hourly)
                fig_roc, ax_roc = plt.subplots(figsize=(12, 3))
                style_axes(ax_roc)
                ax_roc.plot(roc_h.index, roc_h.values, linewidth=1.5, label=f"ROC% ({mom_lb_hourly})")
                ax_roc.axhline(0.0, linewidth=1.0, alpha=0.5)
                ax_roc.set_title(f"{ticker} — ROC% (Hourly)")
                ax_roc.legend(loc="best", fontsize=8)
                st.pyplot(fig_roc, use_container_width=True)

            # Hourly NTD panel
            if show_nrsi:
                st.subheader("NTD (Hourly) + Channel Highlight")
                ntd_h = compute_normalized_trend(close_h, window=ntd_window)
                npx_h = compute_normalized_price(close_h, window=ntd_window)

                fig_ntdh, ax_ntdh = plt.subplots(figsize=(12, 3))
                style_axes(ax_ntdh)
                ax_ntdh.plot(ntd_h.index, ntd_h.values, linewidth=1.6, label="NTD (Hourly)")
                ax_ntdh.axhline(0.0, linewidth=1.0, alpha=0.5)
                ax_ntdh.axhline(0.75, linestyle="--", linewidth=0.9, alpha=0.6)
                ax_ntdh.axhline(-0.75, linestyle="--", linewidth=0.9, alpha=0.6)

                if shade_ntd:
                    shade_ntd_regions(ax_ntdh, ntd_h)

                # Highlight channel when price is between S/R on NTD panel
                if show_ntd_channel:
                    sh = _coerce_1d_series(sup_h).reindex(close_h.index).ffill()
                    rh = _coerce_1d_series(res_h).reindex(close_h.index).ffill()
                    inside = (close_h >= sh) & (close_h <= rh)
                    if inside.any():
                        ax_ntdh.fill_between(ntd_h.index, -1.05, 1.05, where=inside.reindex(ntd_h.index, fill_value=False),
                                             alpha=0.07)

                overlay_ntd_triangles_by_trend(ax_ntdh, ntd_h, trend_slope=slope_h)

                if show_npx_ntd:
                    overlay_npx_on_ntd(ax_ntdh, npx=npx_h, ntd=ntd_h, mark_crosses=mark_npx_cross)

                ax_ntdh.set_ylim(-1.05, 1.05)
                ax_ntdh.set_title(f"{ticker} — NTD (Hourly)")
                ax_ntdh.legend(loc="best", fontsize=8)
                st.pyplot(fig_ntdh, use_container_width=True)


# =========================
# Part 10/10 — bullbear.py
# =========================
    # ---------------------------
    # SARIMAX Forecast (Daily)
    # ---------------------------
    st.subheader("30-Day Forecast (SARIMAX)")

    if close_daily is None or close_daily.dropna().shape[0] < 60:
        st.warning("Not enough daily history for SARIMAX forecast (need ~60+ data points).")
    else:
        with st.spinner("Computing forecast..."):
            try:
                fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close_daily.dropna())
                st.session_state.fc_idx = fc_idx
                st.session_state.fc_vals = fc_vals
                st.session_state.fc_ci = fc_ci
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                fc_idx = pd.DatetimeIndex([])
                fc_vals = pd.Series(dtype=float)
                fc_ci = pd.DataFrame()

        if len(fc_idx) and fc_vals is not None and len(fc_vals):
            fig_fc, ax_fc = plt.subplots(figsize=(12, 4))
            style_axes(ax_fc)

            # Plot last ~180 days of history for context
            hist_ctx = close_daily.dropna()
            hist_ctx = hist_ctx.iloc[-180:] if len(hist_ctx) > 180 else hist_ctx
            ax_fc.plot(hist_ctx.index, hist_ctx.values, linewidth=1.6, label="History")

            # Forecast
            ax_fc.plot(fc_idx, fc_vals.values, linestyle="--", linewidth=2.0, label="Forecast")

            # Confidence interval
            if isinstance(fc_ci, pd.DataFrame) and fc_ci.shape[1] >= 2:
                lo = fc_ci.iloc[:, 0].to_numpy(dtype=float)
                hi = fc_ci.iloc[:, 1].to_numpy(dtype=float)
                ax_fc.fill_between(fc_idx, lo, hi, alpha=0.15, label="Conf. Interval")

            ax_fc.set_title(f"{ticker} — 30-Day SARIMAX Forecast")
            ax_fc.set_ylabel("Price")
            ax_fc.legend(loc="best", fontsize=8)
            st.pyplot(fig_fc, use_container_width=True)

            # Forecast table
            try:
                out = pd.DataFrame({
                    "date": fc_idx,
                    "forecast": pd.to_numeric(fc_vals, errors="coerce").values
                })
                if isinstance(fc_ci, pd.DataFrame) and fc_ci.shape[1] >= 2:
                    out["low"] = pd.to_numeric(fc_ci.iloc[:, 0], errors="coerce").values
                    out["high"] = pd.to_numeric(fc_ci.iloc[:, 1], errors="coerce").values
                out = out.set_index("date")
                st.dataframe(out.tail(30), use_container_width=True)
            except Exception:
                pass

    st.divider()
    st.caption(
        "Disclaimer: This dashboard is for informational/educational purposes only and is not financial advice. "
        "Signals are heuristic and may be wrong, especially during regime changes or illiquid conditions."
    )
