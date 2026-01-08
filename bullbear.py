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

# ---------------------------
# NEW (THIS REQUEST): "Reverse Possible" when regression slope successfully reverses at Fib 0% / 100%
# ---------------------------
def regression_slope_reversal_at_fib_extremes(series_like,
                                              slope_lb: int,
                                              proximity_pct_of_range: float = 0.02,
                                              confirm_bars: int = 2,
                                              lookback_bars: int = 120):
    """
    Returns dict when BOTH are true:
      1) price touched near Fib 0% (high) or 100% (low)
      2) regression slope sign flipped after that touch
         + confirms reversal via consecutive closes
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return None

    lb = int(max(10, lookback_bars))
    s = s.iloc[-lb:] if len(s) > lb else s
    if len(s) < max(6, int(slope_lb) + 3):
        return None

    fibs = fibonacci_levels(s)
    if not fibs:
        return None

    hi = float(fibs.get("0%", np.nan))
    lo = float(fibs.get("100%", np.nan))
    rng = hi - lo
    if not (np.isfinite(hi) and np.isfinite(lo) and np.isfinite(rng)) or rng <= 0:
        return None

    tol = float(proximity_pct_of_range) * rng
    if not np.isfinite(tol) or tol <= 0:
        return None

    near_hi = s >= (hi - tol)
    near_lo = s <= (lo + tol)
    last_hi_touch = near_hi[near_hi].index[-1] if near_hi.any() else None
    last_lo_touch = near_lo[near_lo].index[-1] if near_lo.any() else None

    _, _, _, m_curr, _ = regression_with_band(s, lookback=int(slope_lb))

    def _pre_slope_at(t_touch):
        seg = _coerce_1d_series(s.loc[:t_touch]).dropna().tail(int(slope_lb))
        if len(seg) < 3:
            return np.nan
        _, _, _, m_pre, _ = regression_with_band(seg, lookback=int(slope_lb))
        return float(m_pre) if np.isfinite(m_pre) else np.nan

    buy_rev = None
    if last_lo_touch is not None:
        m_pre = _pre_slope_at(last_lo_touch)
        seg_after = s.loc[last_lo_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) < 0.0) and (float(m_curr) > 0.0) and _n_consecutive_increasing(seg_after, int(confirm_bars)):
                buy_rev = {
                    "side": "BUY",
                    "from_level": "100%",
                    "touch_time": last_lo_touch,
                    "touch_price": float(s.loc[last_lo_touch]) if np.isfinite(s.loc[last_lo_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    sell_rev = None
    if last_hi_touch is not None:
        m_pre = _pre_slope_at(last_hi_touch)
        seg_after = s.loc[last_hi_touch:]
        if np.isfinite(m_pre) and np.isfinite(m_curr):
            if (float(m_pre) > 0.0) and (float(m_curr) < 0.0) and _n_consecutive_decreasing(seg_after, int(confirm_bars)):
                sell_rev = {
                    "side": "SELL",
                    "from_level": "0%",
                    "touch_time": last_hi_touch,
                    "touch_price": float(s.loc[last_hi_touch]) if np.isfinite(s.loc[last_hi_touch]) else np.nan,
                    "pre_slope": float(m_pre),
                    "curr_slope": float(m_curr),
                }

    if buy_rev is None and sell_rev is None:
        return None
    if buy_rev is None:
        return sell_rev
    if sell_rev is None:
        return buy_rev

    return buy_rev if buy_rev["touch_time"] >= sell_rev["touch_time"] else sell_rev

def annotate_reverse_possible(ax, rev_info: dict, text: str = "Reverse Possible"):
    if not isinstance(rev_info, dict):
        return
    t = rev_info.get("touch_time", None)
    y = rev_info.get("touch_price", np.nan)
    side = str(rev_info.get("side", "")).upper()
    if t is None or (not np.isfinite(y)):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    va = "bottom" if side == "BUY" else "top"
    ax.text(
        t, y,
        f"  {text}",
        color=col,
        fontsize=10,
        fontweight="bold",
        va=va,
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.80),
        zorder=25
    )
# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Support / Resistance + helpers
# ---------------------------
def rolling_support_resistance_from_ohlc(ohlc: pd.DataFrame, lookback: int = 60):
    """
    Simple rolling S/R:
      Support = rolling min of Low
      Resistance = rolling max of High
    Returns (support_series, resistance_series)
    """
    if ohlc is None or ohlc.empty or not {"High", "Low"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else pd.DatetimeIndex([])
        return (pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float))
    lb = max(2, int(lookback))
    low = _coerce_1d_series(ohlc["Low"])
    high = _coerce_1d_series(ohlc["High"])
    sup = low.rolling(lb, min_periods=max(2, lb // 3)).min()
    res = high.rolling(lb, min_periods=max(2, lb // 3)).max()
    return sup.reindex(ohlc.index), res.reindex(ohlc.index)

def last_support_resistance(ohlc: pd.DataFrame, lookback: int = 60):
    sup, res = rolling_support_resistance_from_ohlc(ohlc, lookback=lookback)
    if sup.dropna().empty or res.dropna().empty:
        return (np.nan, np.nan, sup, res)
    return float(sup.dropna().iloc[-1]), float(res.dropna().iloc[-1]), sup, res

def band_touch_proximity(price: pd.Series, upper: pd.Series, lower: pd.Series, side: str = "lower", tol: float = 0.0):
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper).reindex(p.index)
    l = _coerce_1d_series(lower).reindex(p.index)
    ok = p.notna() & u.notna() & l.notna()
    if ok.sum() == 0:
        return pd.Series(False, index=p.index)
    p = p[ok]; u = u[ok]; l = l[ok]
    if side == "lower":
        return (p <= (l * (1.0 + tol))).reindex(price.index, fill_value=False)
    return (p >= (u * (1.0 - tol))).reindex(price.index, fill_value=False)

def compute_bull_bear_pct(close: pd.Series, period: str = "6mo"):
    """
    Return:
      - pct_up: % days up
      - pct_down: % days down
      - pct_flat: remaining
    """
    s = _coerce_1d_series(close).dropna()
    if s.empty:
        return np.nan, np.nan, np.nan
    df = s.to_frame("Close").copy()
    df["ret"] = df["Close"].pct_change()
    df = df.dropna()

    if period == "1mo":
        df = df.tail(22)
    elif period == "3mo":
        df = df.tail(66)
    elif period == "6mo":
        df = df.tail(132)
    else:  # 1y
        df = df.tail(252)

    if df.empty:
        return np.nan, np.nan, np.nan

    up = (df["ret"] > 0).mean()
    dn = (df["ret"] < 0).mean()
    fl = max(0.0, 1.0 - up - dn)
    return float(up), float(dn), float(fl)

def _safe_title_symbol(symbol: str) -> str:
    return str(symbol).replace("=X", "").upper()

# ---------------------------
# Forex news markers (best-effort)
# ---------------------------
def _try_fetch_forex_factory_calendar(days: int = 7):
    """
    Best-effort fetch of ForexFactory calendar.
    If request fails (network blocked), returns empty DataFrame.
    """
    import pandas as _pd
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        return _pd.DataFrame()

    url = "https://www.forexfactory.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception:
        return _pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"class": "calendar__table"})
    if table is None:
        return _pd.DataFrame()

    rows = table.find_all("tr", {"class": lambda x: x and "calendar__row" in x})
    events = []
    curr_date = None

    for tr in rows:
        date_cell = tr.find("td", {"class": "calendar__date"})
        time_cell = tr.find("td", {"class": "calendar__time"})
        cur_cell  = tr.find("td", {"class": "calendar__currency"})
        imp_cell  = tr.find("td", {"class": "calendar__impact"})
        evt_cell  = tr.find("td", {"class": "calendar__event"})
        if date_cell and date_cell.get_text(strip=True):
            curr_date = date_cell.get_text(strip=True)

        ttxt = time_cell.get_text(" ", strip=True) if time_cell else ""
        ccy  = cur_cell.get_text(strip=True) if cur_cell else ""
        evt  = evt_cell.get_text(" ", strip=True) if evt_cell else ""
        imp  = imp_cell.get_text(" ", strip=True) if imp_cell else ""
        if not evt:
            continue
        events.append({"date_raw": curr_date, "time_raw": ttxt, "ccy": ccy, "impact": imp, "event": evt})

    if not events:
        return _pd.DataFrame()

    df = _pd.DataFrame(events)

    # Parse date/time roughly; ForexFactory uses ET; we convert to Pacific approximately
    def _parse_dt(row):
        try:
            # date_raw like "MonJan 6" or "MonJan 6" depending on FF
            dr = str(row["date_raw"]).strip()
            tr = str(row["time_raw"]).strip()
            if dr in ("", "None", "nan") or tr in ("", "All Day", "Tentative", "None", "nan"):
                return pd.NaT
            # Make a best effort: extract month/day tokens
            # Example: "MonJan 6" -> "Jan 6 2026"
            tokens = dr.replace("\n", " ").replace("\t", " ").split()
            # If dr merges weekday+month, split letters
            if len(tokens) == 1:
                s = tokens[0]
                # find first capital sequence month
                # crude: locate month abbreviations
                months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                mpos = None
                mname = None
                for m in months:
                    p = s.find(m)
                    if p != -1:
                        mpos = p
                        mname = m
                        break
                if mpos is None:
                    return pd.NaT
                tail = s[mpos:]
                # tail like "Jan 6" or "Jan6"
                tail = tail.replace(mname, f"{mname} ")
                tail_tokens = tail.split()
                if len(tail_tokens) >= 2:
                    md = f"{tail_tokens[0]} {tail_tokens[1]}"
                else:
                    return pd.NaT
            else:
                # tokens like ["Mon", "Jan", "6"]
                if len(tokens) >= 3:
                    md = f"{tokens[-2]} {tokens[-1]}"
                else:
                    return pd.NaT

            year = datetime.now(PACIFIC).year
            # time like "8:30am"
            ts = f"{md} {year} {tr}"
            dt = pd.to_datetime(ts, errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            # Assume ET -> convert to Pacific (approx)
            dt = dt.tz_localize("US/Eastern").tz_convert(PACIFIC)
            return dt
        except Exception:
            return pd.NaT

    df["dt_pacific"] = df.apply(_parse_dt, axis=1)
    df = df.dropna(subset=["dt_pacific"]).sort_values("dt_pacific")
    # filter last `days`
    cutoff = datetime.now(PACIFIC) - pd.Timedelta(days=int(days))
    df = df[df["dt_pacific"] >= cutoff]
    return df.reset_index(drop=True)

def get_forex_news_markers(days: int = 7):
    """
    Returns DataFrame with columns: dt_pacific, ccy, impact, event
    """
    try:
        df = _try_fetch_forex_factory_calendar(days=days)
        return df
    except Exception:
        return pd.DataFrame(columns=["dt_pacific", "ccy", "impact", "event"])


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Plot builders — Daily
# ---------------------------
def plot_daily_panel(symbol: str,
                     ohlc_daily: pd.DataFrame,
                     daily_view_label: str,
                     slope_lb: int,
                     sr_lb: int,
                     ntd_window: int,
                     show_fibs: bool,
                     show_ntd: bool,
                     shade_ntd: bool,
                     show_npx_ntd: bool,
                     mark_npx_cross: bool,
                     show_ichi: bool,
                     ichi_conv: int,
                     ichi_base: int,
                     ichi_spanb: int,
                     show_bbands: bool,
                     bb_win: int,
                     bb_mult: float,
                     bb_use_ema: bool,
                     show_hma: bool,
                     hma_period: int,
                     show_hma_rev_ntd: bool,
                     hma_rev_lb: int,
                     rev_bars_confirm: int,
                     show_macd: bool):
    """
    Returns:
      fig_price, fig_ntd (or None)
      plus computed slopes and last values for instruction block.
    """
    if ohlc_daily is None or ohlc_daily.empty:
        return None, None, {}

    # subset view
    ohlc = subset_by_daily_view(ohlc_daily, daily_view_label)
    if ohlc is None or ohlc.empty:
        return None, None, {}

    close = _coerce_1d_series(ohlc["Close"]).dropna()
    if close.empty:
        return None, None, {}

    # Global trend: use max history close (from cached fetch_hist_max)
    try:
        close_max = fetch_hist_max(symbol)
    except Exception:
        close_max = close

    # slopes
    _, local_slope = slope_line(close, lookback=int(slope_lb))
    _, global_slope = slope_line(close_max.dropna(), lookback=int(max(252, slope_lb)))

    # Regression band on local window
    yhat, upper, lower, reg_slope, reg_r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)

    # S/R
    s_last, r_last, sup, res = last_support_resistance(ohlc, lookback=int(sr_lb))

    # Bollinger
    mid, bb_u, bb_l, bb_pctb, bb_n = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    # Ichimoku
    tenkan = kijun = sen_a = sen_b = chikou = None
    if show_ichi and {"High","Low","Close"}.issubset(ohlc.columns):
        tenkan, kijun, sen_a, sen_b, chikou = ichimoku_lines(
            ohlc["High"], ohlc["Low"], ohlc["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
            shift_cloud=True
        )

    # HMA + MACD
    hma = compute_hma(close, period=int(hma_period)) if show_hma else pd.Series(index=close.index, dtype=float)
    macd, macd_sig, macd_hist = compute_macd(close) if show_macd else (pd.Series(index=close.index, dtype=float),)*3

    # Signals
    band_sig = find_band_bounce_signal(close, upper, lower, slope_val=reg_slope)
    slope_trig = find_slope_trigger_after_band_reversal(close, yhat, upper, lower, horizon=int(rev_horizon))

    # MACD/HMA/SR signal (uses global slope)
    macd_sr_sig = None
    if show_hma and show_macd:
        macd_sr_sig = find_macd_hma_sr_signal(
            close=close,
            hma=hma,
            macd=macd,
            sup=sup,
            res=res,
            global_trend_slope=global_slope,
            prox=float(sr_prox_pct)
        )

    # Fibonacci
    fibs = fibonacci_levels(close) if show_fibs else {}

    # NEW: Fibonacci reversal trigger
    fib_trigger = fib_reversal_trigger_from_extremes(
        close,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb))
    ) if show_fibs else None

    # NEW: Reverse Possible from regression slope reversal at fib extremes
    rev_possible = regression_slope_reversal_at_fib_extremes(
        close,
        slope_lb=int(slope_lb),
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(120, slope_lb * 2))
    ) if show_fibs else None

    # Build price fig
    fig_price, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(close.index, close.values, linewidth=1.6, label="Close")
    style_axes(ax)
    ax.set_title(f"{_safe_title_symbol(symbol)} — Daily Price")
    ax.set_ylabel("Price")

    # Regression trendline + band
    if not yhat.dropna().empty:
        ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=2.0, label=f"Local Regression ({fmt_slope(reg_slope)}/bar, R² {fmt_r2(reg_r2)})")
    if not upper.dropna().empty and not lower.dropna().empty:
        ax.plot(upper.index, upper.values, linewidth=1.0, alpha=0.65, label="+2σ band")
        ax.plot(lower.index, lower.values, linewidth=1.0, alpha=0.65, label="-2σ band")

    # Support / Resistance (rolling)
    if sup.dropna().any():
        ax.plot(sup.index, sup.values, linewidth=1.2, alpha=0.8, label=f"Support ({sr_lb} bars)")
    if res.dropna().any():
        ax.plot(res.index, res.values, linewidth=1.2, alpha=0.8, label=f"Resistance ({sr_lb} bars)")

    # Bollinger
    if show_bbands and mid.dropna().any():
        ax.plot(mid.index, mid.values, linewidth=1.1, alpha=0.9, label=f"BB mid ({bb_win})")
        ax.plot(bb_u.index, bb_u.values, linewidth=1.0, alpha=0.75, label=f"BB upper ({bb_mult}σ)")
        ax.plot(bb_l.index, bb_l.values, linewidth=1.0, alpha=0.75, label=f"BB lower ({bb_mult}σ)")

    # Ichimoku Kijun on price
    if show_ichi and kijun is not None and kijun.dropna().any():
        ax.plot(kijun.index, kijun.values, linewidth=1.6, alpha=0.9, label=f"Kijun ({ichi_base})")

    # HMA on price
    if show_hma and hma.dropna().any():
        ax.plot(hma.index, hma.values, linewidth=1.8, alpha=0.9, label=f"HMA({hma_period})")

    # Fibonacci lines + labels
    if show_fibs and fibs:
        for k, lvl in fibs.items():
            ax.axhline(lvl, linestyle=":", linewidth=1.0, alpha=0.7)
            label_on_left(ax, lvl, f"Fib {k} {fmt_price_val(lvl)}", fontsize=8)

        # NEW: show Fib instruction text (requested)
        ax.text(
            0.01, 0.98,
            FIB_ALERT_TEXT,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            zorder=30
        )

    # Mark signals on price
    if band_sig is not None:
        annotate_crossover(ax, band_sig["time"], band_sig["price"], band_sig["side"], note="Band Bounce")

    if slope_trig is not None:
        annotate_slope_trigger(ax, slope_trig)

    if macd_sr_sig is not None:
        annotate_macd_signal(ax, macd_sr_sig["time"], macd_sr_sig["price"], macd_sr_sig["side"])
        ax.text(macd_sr_sig["time"], macd_sr_sig["price"], f"  {macd_sr_sig.get('note','')}",
                fontsize=9, fontweight="bold",
                va="bottom" if macd_sr_sig["side"] == "BUY" else "top")

    # NEW: Fib reversal trigger label
    if isinstance(fib_trigger, dict) and fib_trigger.get("touch_time") is not None and np.isfinite(fib_trigger.get("touch_price", np.nan)):
        col = "tab:green" if fib_trigger["side"] == "BUY" else "tab:red"
        ax.scatter([fib_trigger["touch_time"]], [fib_trigger["touch_price"]], marker="o", s=120, color=col, zorder=20)
        ax.text(
            fib_trigger["touch_time"], fib_trigger["touch_price"],
            f"  CONFIRMED {fib_trigger['side']} (Fib {fib_trigger['from_level']})",
            color=col, fontsize=10, fontweight="bold",
            va="bottom" if fib_trigger["side"] == "BUY" else "top",
            zorder=21
        )

    # NEW: Reverse Possible label (from regression slope reversal at fib extremes)
    if isinstance(rev_possible, dict):
        annotate_reverse_possible(ax, rev_possible, text="Reverse Possible")

    # Legend
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Instruction block (global + local agreement required)
    last_close = float(close.iloc[-1])
    buy_val = s_last if np.isfinite(s_last) else last_close
    sell_val = r_last if np.isfinite(r_last) else last_close

    instruction = format_trade_instruction(
        trend_slope=local_slope,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=last_close,
        symbol=symbol,
        global_trend_slope=global_slope
    )

    # reversal probability on daily
    rev_prob = slope_reversal_probability(
        close,
        current_slope=local_slope,
        hist_window=int(rev_hist_lb),
        slope_window=int(slope_lb),
        horizon=int(rev_horizon)
    )

    # NTD fig (optional)
    fig_ntd = None
    ntd = compute_normalized_trend(close, window=int(ntd_window))
    npx = compute_normalized_price(close, window=int(ntd_window))

    if show_ntd:
        fig_ntd, ax2 = plt.subplots(figsize=(12.5, 3.4))
        ax2.plot(ntd.index, ntd.values, linewidth=2.0, label="NTD (Norm Trend)")
        if shade_ntd:
            shade_ntd_regions(ax2, ntd)

        # triangles based on global trend
        overlay_ntd_triangles_by_trend(ax2, ntd, trend_slope=global_slope)

        # NPX overlay
        if show_npx_ntd:
            overlay_npx_on_ntd(ax2, npx, ntd, mark_crosses=mark_npx_cross)

        # HMA reversal markers on NTD
        if show_hma and show_hma_rev_ntd and hma.dropna().any():
            overlay_hma_reversal_on_ntd(
                ax2, price=close, hma=hma,
                lookback=int(hma_rev_lb),
                period=int(hma_period),
                ntd=ntd
            )

        # NTD reversal stars using S/R logic
        overlay_ntd_sr_reversal_stars(
            ax2,
            price=close,
            sup=sup,
            res=res,
            trend_slope=global_slope,
            ntd=ntd,
            prox=float(sr_prox_pct),
            bars_confirm=int(rev_bars_confirm)
        )

        style_axes(ax2)
        ax2.set_title(f"{_safe_title_symbol(symbol)} — NTD (Daily)")
        ax2.set_ylim(-1.05, 1.05)
        ax2.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax2.axhline(0.75, linestyle=":", linewidth=1.0, alpha=0.5)
        ax2.axhline(-0.75, linestyle=":", linewidth=1.0, alpha=0.5)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

    meta = {
        "local_slope": float(local_slope) if np.isfinite(local_slope) else np.nan,
        "global_slope": float(global_slope) if np.isfinite(global_slope) else np.nan,
        "reg_slope": float(reg_slope) if np.isfinite(reg_slope) else np.nan,
        "reg_r2": float(reg_r2) if np.isfinite(reg_r2) else np.nan,
        "last_close": float(last_close),
        "support": float(s_last) if np.isfinite(s_last) else np.nan,
        "resistance": float(r_last) if np.isfinite(r_last) else np.nan,
        "instruction": instruction,
        "rev_prob": float(rev_prob) if np.isfinite(rev_prob) else np.nan,
        "fibs": fibs,
        "fib_trigger": fib_trigger,
        "reverse_possible": rev_possible,
    }

    return fig_price, fig_ntd, meta


# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Plot builders — Intraday (5m gapless) + Hourly indicators
# ---------------------------
def resample_to_hourly(ohlc_5m: pd.DataFrame) -> pd.DataFrame:
    if ohlc_5m is None or ohlc_5m.empty or not {"Open","High","Low","Close"}.issubset(ohlc_5m.columns):
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = ohlc_5m.copy().sort_index()
    rule = "1H"
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    h = df.resample(rule).agg(agg).dropna()
    return h

def plot_intraday_price(symbol: str,
                        intraday_5m: pd.DataFrame,
                        hour_range: str,
                        show_fibs: bool,
                        slope_lb_hourly: int,
                        sr_lb_hourly: int,
                        show_sessions_pst: bool,
                        show_fx_news: bool,
                        news_window_days: int,
                        show_bbands: bool,
                        bb_win: int,
                        bb_mult: float,
                        bb_use_ema: bool,
                        show_ichi: bool,
                        ichi_conv: int,
                        ichi_base: int,
                        ichi_spanb: int,
                        show_psar: bool,
                        psar_step: float,
                        psar_max: float,
                        atr_period: int,
                        atr_mult: float):
    """
    Intraday display uses 5m gapless OHLC with x-axis in *bar positions*
    (continuous), plus optional markers for sessions + news.
    Also computes Hourly-based signals from resampled hourly OHLC for slopes/SR.
    """
    if intraday_5m is None or intraday_5m.empty:
        return None, None, {}

    df = intraday_5m.copy().sort_index()

    # hour range selection
    if hour_range == "6h":
        cutoff = df.index.max() - pd.Timedelta(hours=6)
    elif hour_range == "12h":
        cutoff = df.index.max() - pd.Timedelta(hours=12)
    elif hour_range == "24h":
        cutoff = df.index.max() - pd.Timedelta(hours=24)
    else:
        cutoff = df.index.min()
    df = df[df.index >= cutoff]
    if df.empty:
        return None, None, {}

    # store real times, and use integer x for gapless
    real_times = df.index
    x = np.arange(len(df), dtype=int)

    close_5m = _coerce_1d_series(df["Close"]).reindex(df.index)
    open_5m  = _coerce_1d_series(df["Open"]).reindex(df.index)
    high_5m  = _coerce_1d_series(df["High"]).reindex(df.index)
    low_5m   = _coerce_1d_series(df["Low"]).reindex(df.index)

    # Hourly resample for indicators
    hourly = resample_to_hourly(df)
    close_h = _coerce_1d_series(hourly["Close"]).dropna()

    # slopes hourly
    _, slope_h = slope_line(close_h, lookback=int(slope_lb_hourly))
    # use global slope as daily max trend if possible
    try:
        close_max = fetch_hist_max(symbol).dropna()
    except Exception:
        close_max = close_h
    _, slope_global = slope_line(close_max, lookback=int(max(252, slope_lb_hourly)))

    # regression band on hourly
    yhat_h, upper_h, lower_h, reg_slope_h, reg_r2_h = regression_with_band(close_h, lookback=int(slope_lb_hourly), z=2.0)

    # S/R hourly
    s_last_h, r_last_h, sup_h, res_h = last_support_resistance(hourly, lookback=int(sr_lb_hourly))

    # indicators on 5m price chart (BB/Ichimoku/PSAR/Supertrend computed on 5m)
    mid, bb_u, bb_l, bb_pctb, bb_n = compute_bbands(close_5m, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    tenkan = kijun = sen_a = sen_b = chikou = None
    if show_ichi:
        tenkan, kijun, sen_a, sen_b, chikou = ichimoku_lines(high_5m, low_5m, close_5m,
                                                             conv=int(ichi_conv),
                                                             base=int(ichi_base),
                                                             span_b=int(ichi_spanb),
                                                             shift_cloud=False)

    st_df = compute_supertrend(df[["High","Low","Close"]], atr_period=int(atr_period), atr_mult=float(atr_mult))
    psar_df = compute_psar_from_ohlc(df[["High","Low"]], step=float(psar_step), max_step=float(psar_max)) if show_psar else pd.DataFrame()

    fibs_5m = fibonacci_levels(close_5m) if show_fibs else {}

    # NEW: fib triggers & reverse possible on hourly close (since intraday guidance wants hourly too)
    fib_trigger_h = fib_reversal_trigger_from_extremes(
        close_h,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_hourly))
    ) if show_fibs else None

    rev_possible_h = regression_slope_reversal_at_fib_extremes(
        close_h,
        slope_lb=int(slope_lb_hourly),
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(120, slope_lb_hourly * 2))
    ) if show_fibs else None

    # Signals on hourly series
    band_sig_h = find_band_bounce_signal(close_h, upper_h, lower_h, slope_val=reg_slope_h)
    slope_trig_h = find_slope_trigger_after_band_reversal(close_h, yhat_h, upper_h, lower_h, horizon=int(rev_horizon))

    # Plot
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(x, close_5m.values, linewidth=1.6, label="Close (5m gapless)")
    style_axes(ax)
    ax.set_title(f"{_safe_title_symbol(symbol)} — Intraday (Gapless)")
    ax.set_ylabel("Price")

    # BB on 5m
    if show_bbands and mid.dropna().any():
        ax.plot(x, mid.values, linewidth=1.0, alpha=0.85, label=f"BB mid ({bb_win})")
        ax.plot(x, bb_u.values, linewidth=0.9, alpha=0.75, label=f"BB upper ({bb_mult}σ)")
        ax.plot(x, bb_l.values, linewidth=0.9, alpha=0.75, label=f"BB lower ({bb_mult}σ)")

    # Ichimoku Kijun on 5m
    if show_ichi and kijun is not None and kijun.dropna().any():
        ax.plot(x, kijun.values, linewidth=1.4, alpha=0.9, label=f"Kijun ({ichi_base})")

    # Supertrend on 5m
    if st_df is not None and not st_df.empty and "ST" in st_df.columns:
        st_line = _coerce_1d_series(st_df["ST"]).reindex(df.index)
        ax.plot(x, st_line.values, linewidth=1.4, alpha=0.85, label=f"Supertrend (ATR {atr_period}×{atr_mult})")

    # PSAR
    if show_psar and psar_df is not None and not psar_df.empty and "PSAR" in psar_df.columns:
        ps = _coerce_1d_series(psar_df["PSAR"]).reindex(df.index)
        ax.scatter(x, ps.values, s=8, alpha=0.7, label="PSAR")

    # Fibonacci on 5m
    if show_fibs and fibs_5m:
        for k, lvl in fibs_5m.items():
            ax.axhline(lvl, linestyle=":", linewidth=1.0, alpha=0.65)
            # small labels to reduce clutter
        ax.text(
            0.01, 0.98,
            FIB_ALERT_TEXT,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            zorder=30
        )

    # Sessions (approx, PST) — show vertical lines at 00:00, 08:00, 13:30 PST
    if show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and len(real_times) > 10:
        # London open ~ 00:00 PST (approx), NY open ~ 05:30 PST (approx, can drift), we mark common anchors
        anchors = []
        start = real_times.min().floor("D")
        end = real_times.max().ceil("D")
        days = pd.date_range(start, end, freq="D", tz=PACIFIC)
        for d in days:
            anchors.append(d + pd.Timedelta(hours=0))   # 00:00
            anchors.append(d + pd.Timedelta(hours=5.5)) # 05:30
            anchors.append(d + pd.Timedelta(hours=13.5))# 13:30

        xs = _map_times_to_bar_positions(real_times, anchors)
        for xv in xs:
            ax.axvline(xv, linestyle="--", linewidth=0.8, alpha=0.28)

    # Forex news markers
    if show_fx_news and mode == "Forex":
        news_df = get_forex_news_markers(days=int(news_window_days))
        if news_df is not None and not news_df.empty and "dt_pacific" in news_df.columns:
            xs = _map_times_to_bar_positions(real_times, news_df["dt_pacific"].tolist())
            for xv in xs:
                ax.axvline(xv, linestyle=":", linewidth=1.2, alpha=0.35)
            ax.text(
                0.01, 0.02,
                "Forex news markers (best-effort)",
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.65),
                zorder=30
            )

    _apply_compact_time_ticks(ax, real_times, n_ticks=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # Build Hourly NTD panel
    fig_ntd = None
    meta = {}

    if not close_h.empty:
        ntd_h = compute_normalized_trend(close_h, window=int(ntd_window))
        npx_h = compute_normalized_price(close_h, window=int(ntd_window))

        fig_ntd, ax2 = plt.subplots(figsize=(12.5, 3.4))
        ax2.plot(ntd_h.index, ntd_h.values, linewidth=2.0, label="NTD (Hourly)")
        if shade_ntd:
            shade_ntd_regions(ax2, ntd_h)

        # NTD channel highlight (between S/R) — plotted on NTD: shade region when last close between S/R
        if show_ntd_channel:
            # Create boolean series on hourly timestamps
            sup_h2 = _coerce_1d_series(sup_h).reindex(close_h.index).ffill()
            res_h2 = _coerce_1d_series(res_h).reindex(close_h.index).ffill()
            between = (close_h >= sup_h2) & (close_h <= res_h2)
            # Shade background where between True
            # Do contiguous regions
            if between.any():
                in_seg = False
                seg_start = None
                for t, val in between.items():
                    if val and not in_seg:
                        in_seg = True
                        seg_start = t
                    elif (not val) and in_seg:
                        in_seg = False
                        seg_end = t
                        ax2.axvspan(seg_start, seg_end, alpha=0.10)
                if in_seg and seg_start is not None:
                    ax2.axvspan(seg_start, close_h.index[-1], alpha=0.10)

        overlay_ntd_triangles_by_trend(ax2, ntd_h, trend_slope=slope_global)
        if show_npx_ntd:
            overlay_npx_on_ntd(ax2, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        # Mark fib confirmed trigger on hourly NTD (at ±0.95)
        if isinstance(fib_trigger_h, dict):
            side = fib_trigger_h.get("side", "")
            col = "tab:green" if side == "BUY" else "tab:red"
            yv = 0.95 if side == "BUY" else -0.95
            ax2.scatter([fib_trigger_h.get("last_time", close_h.index[-1])], [yv], marker="o", s=120, color=col, zorder=12,
                        label=f"Fib CONF {side} (Hourly)")

        # Mark Reverse Possible on hourly NTD
        if isinstance(rev_possible_h, dict):
            side = str(rev_possible_h.get("side", "")).upper()
            col = "tab:green" if side == "BUY" else "tab:red"
            yv = 0.80 if side == "BUY" else -0.80
            ax2.scatter([rev_possible_h.get("touch_time", close_h.index[-1])], [yv], marker="s", s=110, color=col, zorder=12,
                        label="Reverse Possible")

        style_axes(ax2)
        ax2.set_title(f"{_safe_title_symbol(symbol)} — NTD (Hourly)")
        ax2.set_ylim(-1.05, 1.05)
        ax2.axhline(0.0, linewidth=1.0, alpha=0.6)
        ax2.axhline(0.75, linestyle=":", linewidth=1.0, alpha=0.5)
        ax2.axhline(-0.75, linestyle=":", linewidth=1.0, alpha=0.5)
        ax2.legend(loc="upper right", fontsize=8, frameon=False)

        # build hourly instruction using global + local agreement rule
        last_close_h = float(close_h.iloc[-1])
        buy_val = s_last_h if np.isfinite(s_last_h) else last_close_h
        sell_val = r_last_h if np.isfinite(r_last_h) else last_close_h

        instruction_h = format_trade_instruction(
            trend_slope=slope_h,
            buy_val=buy_val,
            sell_val=sell_val,
            close_val=last_close_h,
            symbol=symbol,
            global_trend_slope=slope_global
        )

        rev_prob_h = slope_reversal_probability(
            close_h,
            current_slope=slope_h,
            hist_window=int(rev_hist_lb),
            slope_window=int(slope_lb_hourly),
            horizon=int(rev_horizon)
        )

        meta = {
            "hourly_slope": float(slope_h) if np.isfinite(slope_h) else np.nan,
            "global_slope": float(slope_global) if np.isfinite(slope_global) else np.nan,
            "reg_slope_hourly": float(reg_slope_h) if np.isfinite(reg_slope_h) else np.nan,
            "reg_r2_hourly": float(reg_r2_h) if np.isfinite(reg_r2_h) else np.nan,
            "last_close_hourly": float(last_close_h),
            "support_hourly": float(s_last_h) if np.isfinite(s_last_h) else np.nan,
            "resistance_hourly": float(r_last_h) if np.isfinite(r_last_h) else np.nan,
            "instruction_hourly": instruction_h,
            "rev_prob_hourly": float(rev_prob_h) if np.isfinite(rev_prob_h) else np.nan,
            "band_sig_hourly": band_sig_h,
            "slope_trig_hourly": slope_trig_h,
            "fib_trigger_hourly": fib_trigger_h,
            "reverse_possible_hourly": rev_possible_h,
        }

    return fig, fig_ntd, meta


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Main UI: Selection + Run
# ---------------------------
st.subheader("Select Asset")
# Protect selectbox from stale state after mode switch
default_idx = 0
if "ticker" in st.session_state and st.session_state.ticker in universe:
    default_idx = universe.index(st.session_state.ticker)

ticker = st.selectbox("Ticker / Symbol", options=universe, index=default_idx, key="ticker_selectbox")
st.session_state.ticker = ticker

st.sidebar.subheader("Intraday Range")
hour_range = st.sidebar.selectbox("Intraday range", ["6h", "12h", "24h", "All"], index=2, key="sb_hour_range")
st.session_state.hour_range = hour_range

run_all = st.button("▶ Run Dashboard", type="primary", use_container_width=True, key="btn_run_all")
if run_all:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

if not st.session_state.get("run_all", False):
    st.info("Choose a ticker and click **Run Dashboard**.")
    st.stop()

# Safety: if mode changed after run, re-run
if st.session_state.get("mode_at_run") != mode:
    _reset_run_state_for_mode_switch()
    st.stop()

with st.spinner("Fetching data..."):
    # Daily
    df_ohlc = fetch_hist_ohlc(ticker)
    df_hist = fetch_hist(ticker)
    st.session_state.df_ohlc = df_ohlc
    st.session_state.df_hist = df_hist

    # Intraday
    intraday = fetch_intraday(ticker, period="5d" if mode == "Forex" else "5d")
    st.session_state.intraday = intraday

# ---------------------------
# Bull/Bear summary
# ---------------------------
st.subheader("Market Bias")
pct_up, pct_dn, pct_fl = compute_bull_bear_pct(df_hist, period=bb_period)

c1, c2, c3 = st.columns(3)
c1.metric("Bull days", fmt_pct(pct_up, 1))
c2.metric("Bear days", fmt_pct(pct_dn, 1))
c3.metric("Flat/Other", fmt_pct(pct_fl, 1))

# ---------------------------
# Forecast (SARIMAX)
# ---------------------------
st.subheader("30-Day Forecast (Daily Close)")
try:
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
    st.session_state.fc_idx = fc_idx
    st.session_state.fc_vals = fc_vals
    st.session_state.fc_ci = fc_ci

    fig_fc, axfc = plt.subplots(figsize=(12.5, 4.0))
    hist_tail = _coerce_1d_series(df_hist).dropna().tail(180)
    axfc.plot(hist_tail.index, hist_tail.values, linewidth=1.6, label="History (last 180d)")
    axfc.plot(fc_idx, fc_vals.values, linewidth=2.2, label="Forecast")
    try:
        lo = fc_ci.iloc[:, 0]
        hi = fc_ci.iloc[:, 1]
        axfc.fill_between(fc_idx, lo.values, hi.values, alpha=0.2, label="Confidence interval")
    except Exception:
        pass
    style_axes(axfc)
    axfc.set_title(f"{_safe_title_symbol(ticker)} — SARIMAX Forecast")
    axfc.legend(loc="upper left", fontsize=8, frameon=False)
    st.pyplot(fig_fc, use_container_width=True)
except Exception as e:
    st.warning(f"Forecast unavailable: {e}")

# ---------------------------
# Daily charts
# ---------------------------
st.subheader("Daily Analysis")
fig_price_d, fig_ntd_d, meta_d = plot_daily_panel(
    symbol=ticker,
    ohlc_daily=df_ohlc,
    daily_view_label=daily_view,
    slope_lb=int(slope_lb_daily),
    sr_lb=int(sr_lb_daily),
    ntd_window=int(ntd_window),
    show_fibs=bool(show_fibs),
    show_ntd=bool(show_ntd),
    shade_ntd=bool(shade_ntd),
    show_npx_ntd=bool(show_npx_ntd),
    mark_npx_cross=bool(mark_npx_cross),
    show_ichi=bool(show_ichi),
    ichi_conv=int(ichi_conv),
    ichi_base=int(ichi_base),
    ichi_spanb=int(ichi_spanb),
    show_bbands=bool(show_bbands),
    bb_win=int(bb_win),
    bb_mult=float(bb_mult),
    bb_use_ema=bool(bb_use_ema),
    show_hma=bool(show_hma),
    hma_period=int(hma_period),
    show_hma_rev_ntd=bool(show_hma_rev_ntd),
    hma_rev_lb=int(hma_rev_lb),
    rev_bars_confirm=int(rev_bars_confirm),
    show_macd=bool(show_macd)
)

if fig_price_d is not None:
    st.pyplot(fig_price_d, use_container_width=True)

# Daily instruction + reversal probability
if isinstance(meta_d, dict) and meta_d:
    m1, m2, m3 = st.columns([2, 1, 1])
    m1.markdown(f"**Daily Instruction:** {meta_d.get('instruction', '')}")
    m2.metric("Daily Local Slope", fmt_slope(meta_d.get("local_slope", np.nan)))
    m3.metric("Daily Reversal Prob", fmt_pct(meta_d.get("rev_prob", np.nan), 0))

    # NEW: show Reverse Possible + CONFIRMED Fib triggers in text
    fib_tr = meta_d.get("fib_trigger", None)
    rev_ps = meta_d.get("reverse_possible", None)
    if isinstance(fib_tr, dict):
        st.success(
            f"CONFIRMED {fib_tr.get('side','')} from Fib {fib_tr.get('from_level','')} "
            f"(touched {fib_tr.get('touch_time')} @ {fmt_price_val(fib_tr.get('touch_price', np.nan))})"
        )
    if isinstance(rev_ps, dict):
        st.warning(
            f"Reverse Possible ({rev_ps.get('side','')}) — Fib {rev_ps.get('from_level','')} "
            f"with regression slope flip ({fmt_slope(rev_ps.get('pre_slope', np.nan))} → {fmt_slope(rev_ps.get('curr_slope', np.nan))})"
        )

if fig_ntd_d is not None:
    st.pyplot(fig_ntd_d, use_container_width=True)

# ---------------------------
# Intraday charts
# ---------------------------
st.subheader("Intraday Analysis (Gapless 5m) + Hourly Indicators")
fig_intra, fig_ntd_h, meta_h = plot_intraday_price(
    symbol=ticker,
    intraday_5m=intraday,
    hour_range=hour_range,
    show_fibs=bool(show_fibs),
    slope_lb_hourly=int(slope_lb_hourly),
    sr_lb_hourly=int(sr_lb_hourly),
    show_sessions_pst=bool(show_sessions_pst),
    show_fx_news=bool(show_fx_news),
    news_window_days=int(news_window_days),
    show_bbands=bool(show_bbands),
    bb_win=int(bb_win),
    bb_mult=float(bb_mult),
    bb_use_ema=bool(bb_use_ema),
    show_ichi=bool(show_ichi),
    ichi_conv=int(ichi_conv),
    ichi_base=int(ichi_base),
    ichi_spanb=int(ichi_spanb),
    show_psar=bool(show_psar),
    psar_step=float(psar_step),
    psar_max=float(psar_max),
    atr_period=int(atr_period),
    atr_mult=float(atr_mult)
)

if fig_intra is not None:
    st.pyplot(fig_intra, use_container_width=True)

if isinstance(meta_h, dict) and meta_h:
    h1, h2, h3 = st.columns([2, 1, 1])
    h1.markdown(f"**Hourly Instruction:** {meta_h.get('instruction_hourly', '')}")
    h2.metric("Hourly Local Slope", fmt_slope(meta_h.get("hourly_slope", np.nan)))
    h3.metric("Hourly Reversal Prob", fmt_pct(meta_h.get("rev_prob_hourly", np.nan), 0))

    # show hourly band/slope triggers info (text)
    bs = meta_h.get("band_sig_hourly", None)
    stg = meta_h.get("slope_trig_hourly", None)
    if isinstance(bs, dict):
        st.info(f"Hourly band bounce signal: {bs.get('side','')} at {bs.get('time')} @ {fmt_price_val(bs.get('price', np.nan))}")
    if isinstance(stg, dict):
        st.info(f"Hourly slope trigger: {stg.get('side','')} (touch {stg.get('touch_time')} → cross {stg.get('cross_time')})")

    # NEW: show Reverse Possible + CONFIRMED Fib triggers in text (hourly)
    fib_h = meta_h.get("fib_trigger_hourly", None)
    rev_h = meta_h.get("reverse_possible_hourly", None)
    if isinstance(fib_h, dict):
        st.success(
            f"Hourly CONFIRMED {fib_h.get('side','')} from Fib {fib_h.get('from_level','')} "
            f"(touched {fib_h.get('touch_time')} @ {fmt_price_val(fib_h.get('touch_price', np.nan))})"
        )
    if isinstance(rev_h, dict):
        st.warning(
            f"Hourly Reverse Possible ({rev_h.get('side','')}) — Fib {rev_h.get('from_level','')} "
            f"with regression slope flip ({fmt_slope(rev_h.get('pre_slope', np.nan))} → {fmt_slope(rev_h.get('curr_slope', np.nan))})"
        )

if fig_ntd_h is not None and show_nrsi:
    st.pyplot(fig_ntd_h, use_container_width=True)

# Optional hourly momentum (ROC)
if show_mom_hourly and intraday is not None and not intraday.empty:
    hourly = resample_to_hourly(intraday)
    close_h = _coerce_1d_series(hourly["Close"]).dropna()
    roc = compute_roc(close_h, n=int(mom_lb_hourly))
    fig_mom, axm = plt.subplots(figsize=(12.5, 3.2))
    axm.plot(roc.index, roc.values, linewidth=1.8, label=f"ROC% ({mom_lb_hourly} bars)")
    style_axes(axm)
    axm.axhline(0.0, linewidth=1.0, alpha=0.6)
    axm.set_title(f"{_safe_title_symbol(ticker)} — Hourly Momentum (ROC%)")
    axm.legend(loc="upper right", fontsize=8, frameon=False)
    st.pyplot(fig_mom, use_container_width=True)

# Optional MACD chart (daily)
if show_macd:
    close_d = _coerce_1d_series(df_hist).dropna()
    macd_d, sig_d, hist_d = compute_macd(close_d)
    fig_m, axm = plt.subplots(figsize=(12.5, 3.4))
    axm.plot(macd_d.index, macd_d.values, linewidth=1.8, label="MACD")
    axm.plot(sig_d.index, sig_d.values, linewidth=1.4, label="Signal")
    axm.bar(hist_d.index, hist_d.values, alpha=0.35, label="Hist")
    style_axes(axm)
    axm.axhline(0.0, linewidth=1.0, alpha=0.6)
    axm.set_title(f"{_safe_title_symbol(ticker)} — MACD (Daily)")
    axm.legend(loc="upper right", fontsize=8, frameon=False)
    st.pyplot(fig_m, use_container_width=True)


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# NTX Scanner (updated, no omissions)
# ---------------------------
st.subheader("NTX Scanner (Universe Sweep)")

st.caption(
    "Scans the current universe for **Global+Local slope agreement**, "
    "**NTD channel**, **Fib confirmed triggers**, and **Reverse Possible** markers."
)

scan_enabled = st.checkbox("Enable NTX Scanner", value=True, key="sb_ntx_enable")
scan_limit = st.slider("Max tickers to scan", 5, min(60, len(universe)), min(20, len(universe)), 1, key="sb_ntx_limit")
scan_use_hourly = st.checkbox("Scan using Hourly (intraday) signals", value=True, key="sb_ntx_hourly")
scan_use_daily = st.checkbox("Scan using Daily signals", value=True, key="sb_ntx_daily")
scan_sort = st.selectbox("Sort by", ["Signal Strength", "Reversal Prob", "R²", "NTD"], index=0, key="sb_ntx_sort")

@st.cache_data(ttl=120)
def _scanner_fetch_daily(symbol: str):
    df_ohlc = fetch_hist_ohlc(symbol)
    close = _coerce_1d_series(df_ohlc["Close"]).dropna()
    return df_ohlc, close

@st.cache_data(ttl=120)
def _scanner_fetch_hourly(symbol: str):
    intra = fetch_intraday(symbol, period="5d")
    hourly = resample_to_hourly(intra)
    close_h = _coerce_1d_series(hourly["Close"]).dropna()
    return intra, hourly, close_h

def _signal_strength_from_instruction(text: str) -> float:
    """
    Heuristic strength:
      BUY/SELL instruction => 1.0
      ALERT => 0.0
    """
    t = str(text or "")
    if "ALERT" in t:
        return 0.0
    if "BUY" in t or "SELL" in t:
        return 1.0
    return 0.0

def _scanner_row(symbol: str):
    out = {
        "Symbol": symbol,
        "DailyInstruction": "",
        "HourlyInstruction": "",
        "DailySlope": np.nan,
        "HourlySlope": np.nan,
        "GlobalSlope": np.nan,
        "DailyR2": np.nan,
        "HourlyR2": np.nan,
        "DailyRevProb": np.nan,
        "HourlyRevProb": np.nan,
        "DailyNTD": np.nan,
        "HourlyNTD": np.nan,
        "DailyFibConfirmed": "",
        "HourlyFibConfirmed": "",
        "DailyReversePossible": "",
        "HourlyReversePossible": "",
        "NTD_Channel_Hourly": "",
        "SignalStrength": 0.0,
    }

    # Global slope from max history
    try:
        close_max = fetch_hist_max(symbol).dropna()
        _, g_slope = slope_line(close_max, lookback=252)
    except Exception:
        g_slope = np.nan
    out["GlobalSlope"] = float(g_slope) if np.isfinite(g_slope) else np.nan

    if scan_use_daily:
        try:
            ohlc_d, close_d = _scanner_fetch_daily(symbol)
            if not close_d.empty:
                _, l_slope = slope_line(close_d, lookback=int(slope_lb_daily))
                yhat, u, l, m, r2 = regression_with_band(close_d, lookback=int(slope_lb_daily), z=2.0)
                out["DailySlope"] = float(l_slope) if np.isfinite(l_slope) else np.nan
                out["DailyR2"] = float(r2) if np.isfinite(r2) else np.nan

                # S/R for instruction pricing
                s_last, r_last, sup, res = last_support_resistance(ohlc_d, lookback=int(sr_lb_daily))
                last_close = float(close_d.iloc[-1])
                instr = format_trade_instruction(
                    trend_slope=l_slope,
                    buy_val=s_last if np.isfinite(s_last) else last_close,
                    sell_val=r_last if np.isfinite(r_last) else last_close,
                    close_val=last_close,
                    symbol=symbol,
                    global_trend_slope=g_slope
                )
                out["DailyInstruction"] = instr

                rp = slope_reversal_probability(
                    close_d,
                    current_slope=l_slope,
                    hist_window=int(rev_hist_lb),
                    slope_window=int(slope_lb_daily),
                    horizon=int(rev_horizon)
                )
                out["DailyRevProb"] = float(rp) if np.isfinite(rp) else np.nan

                ntd_d = compute_normalized_trend(close_d, window=int(ntd_window))
                if not ntd_d.dropna().empty:
                    out["DailyNTD"] = float(ntd_d.dropna().iloc[-1])

                # Fib confirmed + Reverse Possible (daily)
                if show_fibs:
                    fib_tr = fib_reversal_trigger_from_extremes(
                        close_d,
                        proximity_pct_of_range=0.02,
                        confirm_bars=int(rev_bars_confirm),
                        lookback_bars=int(max(60, slope_lb_daily))
                    )
                    if isinstance(fib_tr, dict):
                        out["DailyFibConfirmed"] = f"{fib_tr.get('side','')} from {fib_tr.get('from_level','')}"

                    rev_ps = regression_slope_reversal_at_fib_extremes(
                        close_d,
                        slope_lb=int(slope_lb_daily),
                        proximity_pct_of_range=0.02,
                        confirm_bars=int(rev_bars_confirm),
                        lookback_bars=int(max(120, slope_lb_daily * 2))
                    )
                    if isinstance(rev_ps, dict):
                        out["DailyReversePossible"] = f"{rev_ps.get('side','')} at {rev_ps.get('from_level','')}"
        except Exception:
            pass

    if scan_use_hourly:
        try:
            intra, ohlc_h, close_h = _scanner_fetch_hourly(symbol)
            if not close_h.empty:
                _, l_slope_h = slope_line(close_h, lookback=int(slope_lb_hourly))
                yhat_h, u_h, l_h, m_h, r2_h = regression_with_band(close_h, lookback=int(slope_lb_hourly), z=2.0)
                out["HourlySlope"] = float(l_slope_h) if np.isfinite(l_slope_h) else np.nan
                out["HourlyR2"] = float(r2_h) if np.isfinite(r2_h) else np.nan

                s_last_h, r_last_h, sup_h, res_h = last_support_resistance(ohlc_h, lookback=int(sr_lb_hourly))
                last_close_h = float(close_h.iloc[-1])
                instr_h = format_trade_instruction(
                    trend_slope=l_slope_h,
                    buy_val=s_last_h if np.isfinite(s_last_h) else last_close_h,
                    sell_val=r_last_h if np.isfinite(r_last_h) else last_close_h,
                    close_val=last_close_h,
                    symbol=symbol,
                    global_trend_slope=g_slope
                )
                out["HourlyInstruction"] = instr_h

                rp_h = slope_reversal_probability(
                    close_h,
                    current_slope=l_slope_h,
                    hist_window=int(rev_hist_lb),
                    slope_window=int(slope_lb_hourly),
                    horizon=int(rev_horizon)
                )
                out["HourlyRevProb"] = float(rp_h) if np.isfinite(rp_h) else np.nan

                ntd_h = compute_normalized_trend(close_h, window=int(ntd_window))
                if not ntd_h.dropna().empty:
                    out["HourlyNTD"] = float(ntd_h.dropna().iloc[-1])

                # NTD channel: last close between hourly S/R
                if show_ntd_channel:
                    sup_last = float(_coerce_1d_series(sup_h).dropna().iloc[-1]) if _coerce_1d_series(sup_h).dropna().any() else np.nan
                    res_last = float(_coerce_1d_series(res_h).dropna().iloc[-1]) if _coerce_1d_series(res_h).dropna().any() else np.nan
                    if np.isfinite(sup_last) and np.isfinite(res_last):
                        out["NTD_Channel_Hourly"] = "YES" if (last_close_h >= sup_last and last_close_h <= res_last) else "NO"

                # Fib confirmed + Reverse Possible (hourly)
                if show_fibs:
                    fib_tr_h = fib_reversal_trigger_from_extremes(
                        close_h,
                        proximity_pct_of_range=0.02,
                        confirm_bars=int(rev_bars_confirm),
                        lookback_bars=int(max(60, slope_lb_hourly))
                    )
                    if isinstance(fib_tr_h, dict):
                        out["HourlyFibConfirmed"] = f"{fib_tr_h.get('side','')} from {fib_tr_h.get('from_level','')}"

                    rev_ps_h = regression_slope_reversal_at_fib_extremes(
                        close_h,
                        slope_lb=int(slope_lb_hourly),
                        proximity_pct_of_range=0.02,
                        confirm_bars=int(rev_bars_confirm),
                        lookback_bars=int(max(120, slope_lb_hourly * 2))
                    )
                    if isinstance(rev_ps_h, dict):
                        out["HourlyReversePossible"] = f"{rev_ps_h.get('side','')} at {rev_ps_h.get('from_level','')}"
        except Exception:
            pass

    # Strength: prefer hourly instruction if present, else daily
    strength = 0.0
    if out["HourlyInstruction"]:
        strength = _signal_strength_from_instruction(out["HourlyInstruction"])
    elif out["DailyInstruction"]:
        strength = _signal_strength_from_instruction(out["DailyInstruction"])
    out["SignalStrength"] = float(strength)

    return out

if scan_enabled:
    symbols_to_scan = universe[: int(scan_limit)]
    rows = []
    with st.spinner("Scanning universe..."):
        for sym in symbols_to_scan:
            rows.append(_scanner_row(sym))

    scan_df = pd.DataFrame(rows)

    # Sorting
    if scan_sort == "Signal Strength":
        scan_df = scan_df.sort_values(["SignalStrength", "HourlyFibConfirmed", "DailyFibConfirmed"], ascending=[False, False, False])
    elif scan_sort == "Reversal Prob":
        # prioritize hourly if enabled, else daily
        col = "HourlyRevProb" if scan_use_hourly else "DailyRevProb"
        scan_df = scan_df.sort_values(col, ascending=True)  # lower reversal prob = more stable
    elif scan_sort == "R²":
        col = "HourlyR2" if scan_use_hourly else "DailyR2"
        scan_df = scan_df.sort_values(col, ascending=False)
    else:  # NTD
        col = "HourlyNTD" if scan_use_hourly else "DailyNTD"
        scan_df = scan_df.sort_values(col, ascending=False)

    # Display with helpful formatting
    show_cols = [
        "Symbol",
        "SignalStrength",
        "GlobalSlope",
        "DailySlope", "DailyR2", "DailyRevProb", "DailyNTD", "DailyInstruction", "DailyFibConfirmed", "DailyReversePossible",
        "HourlySlope", "HourlyR2", "HourlyRevProb", "HourlyNTD", "HourlyInstruction", "HourlyFibConfirmed", "HourlyReversePossible",
        "NTD_Channel_Hourly"
    ]
    show_cols = [c for c in show_cols if c in scan_df.columns]
    st.dataframe(scan_df[show_cols], use_container_width=True, height=420)

    # Quick highlights
    top_hits = scan_df[(scan_df["SignalStrength"] >= 1.0)]
    if not top_hits.empty:
        st.success(f"Scanner: {len(top_hits)} tickers have an actionable BUY/SELL instruction (Global+Local agreement).")
    else:
        st.info("Scanner: No tickers met the Global+Local agreement rule right now (instructions are ALERT).")

else:
    st.info("NTX Scanner is disabled.")

st.caption("Done.")
