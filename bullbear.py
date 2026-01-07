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

# Fibonacci-specific alert instruction
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

# Fibonacci applies to Daily + Hourly, default ON
show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True, key="sb_show_fibs")

# NEW (THIS REQUEST): show/hide global trendline (default ON)
show_global_trend = st.sidebar.checkbox("Show Global Trendline", value=True, key="sb_show_global_trend")

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

def trend_slope_only(series_like: pd.Series) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    m, _ = np.polyfit(x, s.values, 1)
    return float(m)

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

# FIX (THIS REQUEST): ensure this exists before any usage (prevents NameError)
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
# Support/Resistance utilities + plot overlays
# ---------------------------
def compute_support_resistance_from_ohlc(ohlc: pd.DataFrame, lookback: int = 60):
    """
    Rolling Support/Resistance using rolling Low(min) and High(max).
    Returns (support_series, resistance_series) aligned to ohlc index.
    """
    if ohlc is None or ohlc.empty:
        return (pd.Series(dtype=float), pd.Series(dtype=float))
    if not {"High", "Low"}.issubset(ohlc.columns):
        return (pd.Series(dtype=float), pd.Series(dtype=float))
    lb = max(5, int(lookback))
    low = _coerce_1d_series(ohlc["Low"])
    high = _coerce_1d_series(ohlc["High"])
    sup = low.rolling(lb, min_periods=max(3, lb // 3)).min()
    res = high.rolling(lb, min_periods=max(3, lb // 3)).max()
    return sup.reindex(ohlc.index), res.reindex(ohlc.index)

def plot_fibonacci(ax, series_like: pd.Series, label_prefix: str = "Fib"):
    """
    Draw Fibonacci levels as horizontal lines on the given axis.
    """
    fibs = fibonacci_levels(series_like)
    if not fibs:
        return
    for k, v in fibs.items():
        if not np.isfinite(v):
            continue
        ax.axhline(v, linewidth=0.9, alpha=0.45, linestyle=":")
        label_on_left(ax, v, f"{label_prefix} {k}: {fmt_price_val(v)}", fontsize=8)

def _safe_ylim(ax, y: pd.Series, pad_frac: float = 0.06):
    y = _coerce_1d_series(y).dropna()
    if y.empty:
        return
    lo = float(y.min())
    hi = float(y.max())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if hi == lo:
        hi = lo + 1e-6
    pad = (hi - lo) * pad_frac
    ax.set_ylim(lo - pad, hi + pad)

def plot_sr(ax, sup: pd.Series, res: pd.Series, show_labels: bool = True):
    """
    Plot support/resistance lines.
    """
    s = _coerce_1d_series(sup)
    r = _coerce_1d_series(res)
    if not s.dropna().empty:
        ax.plot(s.index, s.values, linewidth=1.4, alpha=0.85, linestyle="-", label="Support")
        if show_labels:
            v = _safe_last_float(s)
            if np.isfinite(v):
                label_on_left(ax, v, f"SUP {fmt_price_val(v)}", fontsize=8)
    if not r.dropna().empty:
        ax.plot(r.index, r.values, linewidth=1.4, alpha=0.85, linestyle="-", label="Resistance")
        if show_labels:
            v = _safe_last_float(r)
            if np.isfinite(v):
                label_on_left(ax, v, f"RES {fmt_price_val(v)}", fontsize=8)

def compute_global_trendline(price: pd.Series):
    """
    Global trendline fit on FULL available series (in-sample).
    Returns: (yhat_series, slope)
    """
    s = _coerce_1d_series(price).dropna()
    if len(s) < 3:
        return pd.Series(index=price.index if isinstance(price, pd.Series) else None, dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    yhat = pd.Series(m * x + b, index=s.index)
    return yhat.reindex(price.index), float(m)

def plot_global_trend(ax, yhat: pd.Series, slope: float, show: bool = True):
    if not show:
        return
    y = _coerce_1d_series(yhat).dropna()
    if y.empty:
        return
    col = "green" if np.isfinite(slope) and slope >= 0 else "red"
    ax.plot(y.index, y.values, linestyle="--", linewidth=2.2, alpha=0.75,
            color=col, label=f"Global Trend ({fmt_slope(slope)}/bar)")

def plot_bbands(ax, close: pd.Series):
    mid, upper, lower, pctb, nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    if mid.dropna().empty:
        return
    ax.plot(mid.index, mid.values, linewidth=1.1, alpha=0.9, linestyle="-", label="BB Mid")
    ax.plot(upper.index, upper.values, linewidth=1.0, alpha=0.8, linestyle="--", label=f"BB +{bb_mult}σ")
    ax.plot(lower.index, lower.values, linewidth=1.0, alpha=0.8, linestyle="--", label=f"BB -{bb_mult}σ")
    return mid, upper, lower, pctb, nbb

def _session_markers_pst(times: pd.DatetimeIndex):
    """
    Returns session marker times (PST) for intraday axis.
    London open ~ 00:00 PST (08:00 UTC winter), NY open ~ 06:30 PST (14:30 UTC).
    These are approximations; used for visual cues.
    """
    if not isinstance(times, pd.DatetimeIndex) or times.empty:
        return []
    t0 = times.min().floor("D")
    days = pd.date_range(t0, times.max().ceil("D"), freq="D", tz=PACIFIC)
    marks = []
    for d in days:
        marks.append(d + pd.Timedelta(hours=0))               # London approx
        marks.append(d + pd.Timedelta(hours=6, minutes=30))   # NY approx
    return [m for m in marks if (m >= times.min() and m <= times.max())]

def plot_sessions_pst(ax, times: pd.DatetimeIndex):
    if not show_sessions_pst:
        return
    marks = _session_markers_pst(times)
    if not marks:
        return
    for m in marks:
        ax.axvline(m, alpha=0.22, linewidth=1.0, linestyle=":")

def _make_legend_compact(ax):
    try:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        # de-duplicate while preserving order
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h); l2.append(l)
        ax.legend(h2, l2, loc="best", fontsize=8, frameon=False)
    except Exception:
        pass


# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Daily views renderer
# ---------------------------
def render_daily_views(sel: str,
                       ohlc: pd.DataFrame,
                       close: pd.Series,
                       daily_range: str,
                       slope_lb: int,
                       sr_lb: int):
    """
    Produces:
      • Daily Price chart (with local slope, optional global trendline, S/R, optional fibs, optional BBands, optional Ichimoku)
      • Daily NTD chart (with overlays and stars)
      • Daily summary metrics dict (slopes, r2, etc.)
    """
    out = {}
    if close is None or _coerce_1d_series(close).dropna().empty:
        st.warning("No daily data.")
        return out

    # Subset
    close_show = subset_by_daily_view(close, daily_range)
    ohlc_show = None
    if ohlc is not None and not ohlc.empty:
        ohlc_show = ohlc.reindex(close.index).dropna()
        ohlc_show = subset_by_daily_view(ohlc_show, daily_range)

    # S/R
    sup_d, res_d = compute_support_resistance_from_ohlc(ohlc_show if ohlc_show is not None else ohlc, lookback=sr_lb)

    # Local slope line + R²
    yhat_local, slope_local = slope_line(close_show, lookback=int(slope_lb))
    r2_local = regression_r2(close_show, lookback=int(slope_lb))

    # Global trendline (fit on full daily close, plot only if toggle ON)
    yhat_global_full, slope_global = compute_global_trendline(close)
    yhat_global = yhat_global_full.reindex(close_show.index)

    # Trade instruction values (use S/R as entry/exit anchors when available)
    buy_val = _safe_last_float(sup_d) if not sup_d.empty else _safe_last_float(close_show)
    sell_val = _safe_last_float(res_d) if not res_d.empty else _safe_last_float(close_show)
    close_val = _safe_last_float(close_show)

    # ---------------------------
    # Daily Price chart
    # ---------------------------
    figp = plt.figure(figsize=(11, 5.2))
    axp = figp.add_subplot(111)
    axp.plot(close_show.index, close_show.values, linewidth=1.6, label=f"{sel} Close")
    style_axes(axp)

    if yhat_local.dropna().shape[0] > 0:
        col = "green" if np.isfinite(slope_local) and slope_local >= 0 else "red"
        axp.plot(yhat_local.index, yhat_local.values, linestyle="--", linewidth=2.0, alpha=0.85,
                 color=col, label=f"Local Slope ({fmt_slope(slope_local)}/bar)")

    plot_global_trend(axp, yhat_global, slope_global, show=bool(show_global_trend))

    if show_bbands:
        plot_bbands(axp, close_show)

    plot_sr(axp, sup_d, res_d, show_labels=True)

    if show_fibs:
        plot_fibonacci(axp, close_show, label_prefix="Fib")

    if show_ichi and ohlc_show is not None and not ohlc_show.empty:
        tenkan, kijun, sa, sb, chikou = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
        )
        if kijun.dropna().shape[0] > 0:
            axp.plot(kijun.index, kijun.values, linewidth=1.2, alpha=0.9, label=f"Kijun({ichi_base})")

    axp.set_title(f"Daily Close — {sel}")
    _safe_ylim(axp, close_show)
    _make_legend_compact(axp)

    # instruction (daily)
    instr_daily = format_trade_instruction(
        trend_slope=slope_local,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=close_val,
        symbol=sel,
        global_trend_slope=slope_global
    )
    st.markdown(f"**Daily Instruction:** {instr_daily}")

    st.pyplot(figp, clear_figure=True)
    plt.close(figp)

    # ---------------------------
    # Daily NTD chart
    # ---------------------------
    ntd_d = compute_normalized_trend(close_show, window=ntd_window)
    npx_d = compute_normalized_price(close_show, window=ntd_window)

    fign = plt.figure(figsize=(11, 4.1))
    axn = fign.add_subplot(111)
    axn.plot(ntd_d.index, ntd_d.values, linewidth=1.6, label="NTD (Daily)")
    axn.axhline(0.0, linewidth=1.0, alpha=0.4)
    axn.axhline(0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    axn.axhline(-0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    style_axes(axn)

    if shade_ntd:
        shade_ntd_regions(axn, ntd_d)

    # trend triangles on NTD based on LOCAL slope
    overlay_ntd_triangles_by_trend(axn, ntd_d, trend_slope=slope_local, upper=0.75, lower=-0.75)

    # optional NPX overlay on NTD
    if show_npx_ntd:
        overlay_npx_on_ntd(axn, npx_d, ntd_d, mark_crosses=mark_npx_cross)

    # HMA reversal markers on NTD
    if show_hma_rev_ntd:
        hma_d = compute_hma(close_show, period=hma_period)
        overlay_hma_reversal_on_ntd(axn, close_show, hma_d, lookback=hma_rev_lb, period=hma_period, ntd=ntd_d)

    # NTD reversal stars based on S/R (daily)
    overlay_ntd_sr_reversal_stars(
        axn,
        price=close_show,
        sup=sup_d,
        res=res_d,
        trend_slope=slope_local,
        ntd=ntd_d,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm
    )

    axn.set_title(f"NTD (Daily) — {sel}")
    axn.set_ylim(-1.05, 1.05)
    _make_legend_compact(axn)

    st.pyplot(fign, clear_figure=True)
    plt.close(fign)

    out["slope_local_daily"] = slope_local
    out["r2_local_daily"] = r2_local
    out["slope_global_daily"] = slope_global
    out["sup_daily"] = sup_d
    out["res_daily"] = res_d
    out["close_show"] = close_show
    out["ntd_daily"] = ntd_d
    return out


# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Hourly views renderer (intraday 5m aggregated to hourly)
# ---------------------------
def _resample_to_hourly(ohlc_5m: pd.DataFrame) -> pd.DataFrame:
    if ohlc_5m is None or ohlc_5m.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    df = ohlc_5m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    o = df["Open"].resample("1H").first() if "Open" in df.columns else None
    h = df["High"].resample("1H").max() if "High" in df.columns else None
    l = df["Low"].resample("1H").min() if "Low" in df.columns else None
    c = df["Close"].resample("1H").last() if "Close" in df.columns else None
    v = df["Volume"].resample("1H").sum() if "Volume" in df.columns else None
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    if v is not None:
        out["Volume"] = v
    out = out.dropna(subset=["Close"])
    return out

def render_hourly_views(sel: str,
                        intraday_5m: pd.DataFrame,
                        slope_lb: int,
                        sr_lb: int,
                        is_forex: bool = False):
    """
    Produces:
      • Hourly Close chart (with local slope, optional global trendline, S/R, optional fibs, BBands, supertrend, psar)
      • Hourly NTD chart (+ overlays, including the fixed overlay_ntd_sr_reversal_stars)
      • Optional hourly momentum chart
      • Optional MACD chart (normalized)
      • Metrics dict
    """
    out = {}
    if intraday_5m is None or intraday_5m.empty or "Close" not in intraday_5m.columns:
        st.warning("No intraday data.")
        return out

    # Convert to hourly bars
    ohlc_h = _resample_to_hourly(intraday_5m)
    if ohlc_h.empty:
        st.warning("No hourly bars after resample.")
        return out

    close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
    if close_h.empty:
        st.warning("Hourly close is empty.")
        return out

    # S/R (hourly)
    sup_h, res_h = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb)

    # Local slope
    yhat_local, slope_local = slope_line(close_h, lookback=int(slope_lb))
    r2_local = regression_r2(close_h, lookback=int(slope_lb))

    # Global trendline computed on hourly visible series for stability (intraday only)
    yhat_global, slope_global = compute_global_trendline(close_h)

    buy_val = _safe_last_float(sup_h) if not sup_h.empty else _safe_last_float(close_h)
    sell_val = _safe_last_float(res_h) if not res_h.empty else _safe_last_float(close_h)
    close_val = _safe_last_float(close_h)

    # ---------------------------
    # Hourly Price chart
    # ---------------------------
    figp = plt.figure(figsize=(11, 5.2))
    axp = figp.add_subplot(111)
    axp.plot(close_h.index, close_h.values, linewidth=1.6, label=f"{sel} Hourly Close")
    style_axes(axp)
    plot_sessions_pst(axp, close_h.index)

    if yhat_local.dropna().shape[0] > 0:
        col = "green" if np.isfinite(slope_local) and slope_local >= 0 else "red"
        axp.plot(yhat_local.index, yhat_local.values, linestyle="--", linewidth=2.0, alpha=0.85,
                 color=col, label=f"Local Slope ({fmt_slope(slope_local)}/bar)")

    plot_global_trend(axp, yhat_global, slope_global, show=bool(show_global_trend))

    if show_bbands:
        plot_bbands(axp, close_h)

    plot_sr(axp, sup_h, res_h, show_labels=True)

    if show_fibs:
        plot_fibonacci(axp, close_h, label_prefix="Fib")

    # Supertrend
    try:
        st_df = compute_supertrend(ohlc_h, atr_period=atr_period, atr_mult=atr_mult)
        if not st_df.empty and "ST" in st_df.columns:
            axp.plot(st_df.index, st_df["ST"].values, linewidth=1.3, alpha=0.85, label="Supertrend")
    except Exception:
        pass

    # PSAR
    if show_psar:
        try:
            ps_df = compute_psar_from_ohlc(ohlc_h, step=psar_step, max_step=psar_max)
            if not ps_df.empty and "PSAR" in ps_df.columns:
                axp.scatter(ps_df.index, ps_df["PSAR"].values, s=12, alpha=0.65, label="PSAR")
        except Exception:
            pass

    axp.set_title(f"Hourly Close — {sel}")
    _safe_ylim(axp, close_h)
    _make_legend_compact(axp)

    instr_hourly = format_trade_instruction(
        trend_slope=slope_local,
        buy_val=buy_val,
        sell_val=sell_val,
        close_val=close_val,
        symbol=sel,
        global_trend_slope=slope_global
    )
    st.markdown(f"**Hourly Instruction:** {instr_hourly}")

    st.pyplot(figp, clear_figure=True)
    plt.close(figp)

    # ---------------------------
    # Hourly NTD chart
    # ---------------------------
    ntd_h = compute_normalized_trend(close_h, window=ntd_window)
    npx_h = compute_normalized_price(close_h, window=ntd_window)

    fign = plt.figure(figsize=(11, 4.1))
    axn = fign.add_subplot(111)
    axn.plot(ntd_h.index, ntd_h.values, linewidth=1.6, label="NTD (Hourly)")
    axn.axhline(0.0, linewidth=1.0, alpha=0.4)
    axn.axhline(0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    axn.axhline(-0.75, linewidth=1.0, alpha=0.25, linestyle="--")
    style_axes(axn)
    plot_sessions_pst(axn, ntd_h.index)

    if shade_ntd:
        shade_ntd_regions(axn, ntd_h)

    overlay_ntd_triangles_by_trend(axn, ntd_h, trend_slope=slope_local, upper=0.75, lower=-0.75)

    if show_npx_ntd:
        overlay_npx_on_ntd(axn, npx_h, ntd_h, mark_crosses=mark_npx_cross)

    if show_hma_rev_ntd:
        hma_h = compute_hma(close_h, period=hma_period)
        overlay_hma_reversal_on_ntd(axn, close_h, hma_h, lookback=hma_rev_lb, period=hma_period, ntd=ntd_h)

    # FIXED: this function now exists (prevents NameError)
    overlay_ntd_sr_reversal_stars(
        axn,
        price=close_h,
        sup=sup_h,
        res=res_h,
        trend_slope=slope_local,
        ntd=ntd_h,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm
    )

    axn.set_title(f"NTD (Hourly) — {sel}")
    axn.set_ylim(-1.05, 1.05)
    _make_legend_compact(axn)

    st.pyplot(fign, clear_figure=True)
    plt.close(fign)

    # ---------------------------
    # Optional: Hourly Momentum (ROC%)
    # ---------------------------
    if show_mom_hourly:
        roc = compute_roc(close_h, n=mom_lb_hourly)
        figm = plt.figure(figsize=(11, 3.4))
        axm = figm.add_subplot(111)
        axm.plot(roc.index, roc.values, linewidth=1.4, label=f"ROC% ({mom_lb_hourly})")
        axm.axhline(0.0, linewidth=1.0, alpha=0.4)
        style_axes(axm)
        plot_sessions_pst(axm, roc.index)
        axm.set_title(f"Hourly Momentum — {sel}")
        _make_legend_compact(axm)
        st.pyplot(figm, clear_figure=True)
        plt.close(figm)

    # ---------------------------
    # Optional: MACD (Normalized)
    # ---------------------------
    if show_macd:
        nmacd, nsig, nhist = compute_nmacd(close_h, fast=12, slow=26, signal=9, norm_win=240)
        figc = plt.figure(figsize=(11, 3.6))
        axc = figc.add_subplot(111)
        axc.plot(nmacd.index, nmacd.values, linewidth=1.3, label="N-MACD")
        axc.plot(nsig.index, nsig.values, linewidth=1.1, alpha=0.9, label="N-Signal")
        axc.axhline(0.0, linewidth=1.0, alpha=0.35)
        style_axes(axc)
        plot_sessions_pst(axc, nmacd.index)
        axc.set_ylim(-1.05, 1.05)
        axc.set_title(f"Normalized MACD — {sel}")
        _make_legend_compact(axc)
        st.pyplot(figc, clear_figure=True)
        plt.close(figc)

    out["slope_local_hourly"] = slope_local
    out["r2_local_hourly"] = r2_local
    out["slope_global_hourly"] = slope_global
    out["sup_hourly"] = sup_h
    out["res_hourly"] = res_h
    out["close_hourly"] = close_h
    out["ntd_hourly"] = ntd_h
    out["ohlc_hourly"] = ohlc_h
    return out


# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Scanners (simple + fast; keeps UI consistent)
# ---------------------------
@st.cache_data(ttl=120)
def _scan_one_ticker_daily(ticker: str):
    try:
        ohlc = fetch_hist_ohlc(ticker)
        close = ohlc["Close"].asfreq("D").ffill()
        close = close.tz_convert(PACIFIC) if close.index.tz is not None else close.tz_localize(PACIFIC)
    except Exception:
        return None

    close_show = subset_by_daily_view(close, daily_view)
    yhat_local, slope_local = slope_line(close_show, lookback=int(slope_lb_daily))
    r2 = regression_r2(close_show, lookback=int(slope_lb_daily))
    yhat_g, slope_g = compute_global_trendline(close)

    ntd = compute_normalized_trend(close_show, window=ntd_window)
    ntd_last = _safe_last_float(ntd)

    sup, res = compute_support_resistance_from_ohlc(ohlc, lookback=sr_lb_daily)
    sup_last = _safe_last_float(sup)
    res_last = _safe_last_float(res)
    px_last = _safe_last_float(close_show)

    fib_trig = fib_reversal_trigger_from_extremes(close_show, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=60)

    return {
        "Ticker": ticker,
        "Last": px_last,
        "LocalSlope": slope_local,
        "GlobalSlope": slope_g,
        "R2": r2,
        "NTD": ntd_last,
        "SUP": sup_last,
        "RES": res_last,
        "FibTrig": (fib_trig["side"] if fib_trig else ""),
    }

@st.cache_data(ttl=120)
def _scan_one_ticker_intraday(ticker: str):
    try:
        intra = fetch_intraday(ticker, period="5d")
        if intra is None or intra.empty or "Close" not in intra.columns:
            return None
        ohlc_h = _resample_to_hourly(intra)
        if ohlc_h.empty:
            return None
        close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
    except Exception:
        return None

    yhat_local, slope_local = slope_line(close_h, lookback=int(slope_lb_hourly))
    r2 = regression_r2(close_h, lookback=int(slope_lb_hourly))
    yhat_g, slope_g = compute_global_trendline(close_h)

    ntd = compute_normalized_trend(close_h, window=ntd_window)
    ntd_last = _safe_last_float(ntd)

    sup, res = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb_hourly)
    sup_last = _safe_last_float(sup)
    res_last = _safe_last_float(res)
    px_last = _safe_last_float(close_h)

    fib_trig = fib_reversal_trigger_from_extremes(close_h, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=80)

    return {
        "Ticker": ticker,
        "Last": px_last,
        "LocalSlope": slope_local,
        "GlobalSlope": slope_g,
        "R2": r2,
        "NTD": ntd_last,
        "SUP": sup_last,
        "RES": res_last,
        "FibTrig": (fib_trig["side"] if fib_trig else ""),
    }

def render_scanners(universe_list):
    st.subheader("Scanners")

    s_tabs = st.tabs(["Daily Trend", "Hourly Trend", "Fib Reversal", "MACD/HMA + S/R"])

    with s_tabs[0]:
        st.caption("Daily snapshot across the current universe.")
        rows = []
        for t in universe_list:
            r = _scan_one_ticker_daily(t)
            if r:
                rows.append(r)
        if not rows:
            st.info("No scan results.")
        else:
            df = pd.DataFrame(rows)
            # light ranking: prefer agreement of local+global slope and strong fit
            df["Agree"] = np.sign(df["LocalSlope"].astype(float)) == np.sign(df["GlobalSlope"].astype(float))
            df = df.sort_values(["Agree", "R2"], ascending=[False, False])
            st.dataframe(df, use_container_width=True, height=420)

    with s_tabs[1]:
        st.caption("Hourly snapshot (uses 5-day intraday data).")
        rows = []
        for t in universe_list:
            r = _scan_one_ticker_intraday(t)
            if r:
                rows.append(r)
        if not rows:
            st.info("No scan results.")
        else:
            df = pd.DataFrame(rows)
            df["Agree"] = np.sign(df["LocalSlope"].astype(float)) == np.sign(df["GlobalSlope"].astype(float))
            df = df.sort_values(["Agree", "R2"], ascending=[False, False])
            st.dataframe(df, use_container_width=True, height=420)

    with s_tabs[2]:
        st.caption("Confirmed fib reversal signals (0% / 100%) using the confirmation rule.")
        st.info(FIB_ALERT_TEXT)
        rows_d, rows_h = [], []
        for t in universe_list:
            rd = _scan_one_ticker_daily(t)
            rh = _scan_one_ticker_intraday(t)
            if rd and rd.get("FibTrig"):
                rows_d.append(rd)
            if rh and rh.get("FibTrig"):
                rows_h.append(rh)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Daily Confirmed**")
            if rows_d:
                st.dataframe(pd.DataFrame(rows_d).sort_values(["FibTrig","R2"], ascending=[True, False]),
                             use_container_width=True, height=360)
            else:
                st.write("None.")
        with c2:
            st.markdown("**Hourly Confirmed**")
            if rows_h:
                st.dataframe(pd.DataFrame(rows_h).sort_values(["FibTrig","R2"], ascending=[True, False]),
                             use_container_width=True, height=360)
            else:
                st.write("None.")

    with s_tabs[3]:
        st.caption("Signal occurs when MACD/HMA55 cross aligns with S/R proximity and Global Trendline direction.")
        rows = []
        for t in universe_list:
            try:
                intra = fetch_intraday(t, period="5d")
                ohlc_h = _resample_to_hourly(intra)
                if ohlc_h.empty:
                    continue
                close_h = _coerce_1d_series(ohlc_h["Close"]).dropna()
                sup, res = compute_support_resistance_from_ohlc(ohlc_h, lookback=sr_lb_hourly)
                hma = compute_hma(close_h, period=hma_period)
                macd, sig, hist = compute_macd(close_h)
                yhat_g, slope_g = compute_global_trendline(close_h)
                sigd = find_macd_hma_sr_signal(close_h, hma, macd, sup, res, global_trend_slope=slope_g, prox=sr_prox_pct)
                if sigd:
                    rows.append({
                        "Ticker": t,
                        "Side": sigd.get("side", ""),
                        "Time": sigd.get("time", ""),
                        "Price": sigd.get("price", np.nan),
                        "GlobalSlope": slope_g
                    })
            except Exception:
                continue

        if not rows:
            st.write("No signals found.")
        else:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=420)


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Main app execution
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False

# Ticker picker (robust to mode switches)
default_ticker = universe[0] if universe else None
if st.session_state.get("ticker") not in universe:
    st.session_state.ticker = default_ticker

disp_ticker = st.selectbox(
    "Select ticker:",
    options=universe,
    index=universe.index(st.session_state.ticker) if st.session_state.ticker in universe else 0,
    key="ticker_selectbox"
)
st.session_state.ticker = disp_ticker

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    if st.button("▶ Run", use_container_width=True, key="btn_run"):
        st.session_state.run_all = True
        st.session_state.mode_at_run = mode

with run_col2:
    st.caption("Tip: Switching Forex/Stocks resets run state to prevent stale selects.")

if not st.session_state.run_all:
    st.info("Select a ticker and click **Run**.")
    st.stop()

# Guard: mode changed after running
if st.session_state.get("mode_at_run") != mode:
    st.warning("Mode changed since last run. Please click **Run** again.")
    st.stop()

# ---------------------------
# Load data
# ---------------------------
with st.spinner("Loading data…"):
    df_ohlc = fetch_hist_ohlc(disp_ticker)
    df_close = df_ohlc["Close"].asfreq("D").ffill()
    try:
        df_close = df_close.tz_convert(PACIFIC) if df_close.index.tz is not None else df_close.tz_localize(PACIFIC)
    except Exception:
        pass

    # Intraday (5d gives enough hourly bars)
    df_intra = fetch_intraday(disp_ticker, period="5d")

# ---------------------------
# Forecast (Daily SARIMAX)
# ---------------------------
fc_idx, fc_vals, fc_ci = (None, None, None)
try:
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_close)
except Exception:
    pass

# ---------------------------
# Tabs (10 total)
# ---------------------------
tabs = st.tabs([
    "1) Daily Close",
    "2) Hourly Close",
    "3) NTD Daily",
    "4) NTD Hourly",
    "5) Forecast",
    "6) MACD",
    "7) Momentum",
    "8) 2σ Reversal",
    "9) Fib 0/100 Reversal",
    "10) Scanners"
])

# Precompute render outputs (so each tab is fast + consistent)
daily_out = {}
hourly_out = {}

with tabs[0]:
    st.subheader("Daily Close")
    daily_out = render_daily_views(
        sel=disp_ticker,
        ohlc=df_ohlc,
        close=df_close,
        daily_range=daily_view,
        slope_lb=slope_lb_daily,
        sr_lb=sr_lb_daily
    )

with tabs[1]:
    st.subheader("Hourly Close")
    hourly_out = render_hourly_views(
        sel=disp_ticker,
        intraday_5m=df_intra,
        slope_lb=slope_lb_hourly,
        sr_lb=sr_lb_hourly,
        is_forex=(mode == "Forex")
    )

with tabs[2]:
    st.subheader("NTD Daily")
    if daily_out and "ntd_daily" in daily_out:
        # Daily NTD already shown in render_daily_views, but keep this tab as a dedicated display (no UI change)
        st.info("Daily NTD is shown in the Daily section above. This tab is kept for consistent navigation.")
    else:
        st.warning("Run Daily view first.")

with tabs[3]:
    st.subheader("NTD Hourly")
    if hourly_out and "ntd_hourly" in hourly_out:
        st.info("Hourly NTD is shown in the Hourly section above. This tab is kept for consistent navigation.")
    else:
        st.warning("Run Hourly view first.")

with tabs[4]:
    st.subheader("Forecast (Daily SARIMAX)")
    if fc_idx is None or fc_vals is None or len(fc_vals) == 0:
        st.warning("Forecast unavailable.")
    else:
        figf = plt.figure(figsize=(11, 4.8))
        axf = figf.add_subplot(111)
        show = subset_by_daily_view(df_close, daily_view)
        axf.plot(show.index, show.values, linewidth=1.6, label="History")
        axf.plot(fc_idx, fc_vals.values, linewidth=1.6, linestyle="--", label="Forecast")
        try:
            lo = fc_ci.iloc[:, 0].values
            hi = fc_ci.iloc[:, 1].values
            axf.fill_between(fc_idx, lo, hi, alpha=0.18)
        except Exception:
            pass
        style_axes(axf)
        axf.set_title(f"30-Day Forecast — {disp_ticker}")
        _make_legend_compact(axf)
        st.pyplot(figf, clear_figure=True)
        plt.close(figf)

with tabs[5]:
    st.subheader("MACD (Daily + Hourly if available)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Daily MACD**")
        close_show = subset_by_daily_view(df_close, daily_view)
        macd_d, sig_d, hist_d = compute_macd(close_show)
        fig = plt.figure(figsize=(9, 3.6))
        ax = fig.add_subplot(111)
        ax.plot(macd_d.index, macd_d.values, linewidth=1.3, label="MACD")
        ax.plot(sig_d.index, sig_d.values, linewidth=1.1, alpha=0.9, label="Signal")
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)
        style_axes(ax)
        ax.set_title("Daily MACD")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly MACD**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            macd_h, sig_h, hist_h = compute_macd(close_h)
            fig = plt.figure(figsize=(9, 3.6))
            ax = fig.add_subplot(111)
            ax.plot(macd_h.index, macd_h.values, linewidth=1.3, label="MACD")
            ax.plot(sig_h.index, sig_h.values, linewidth=1.1, alpha=0.9, label="Signal")
            ax.axhline(0.0, linewidth=1.0, alpha=0.35)
            style_axes(ax)
            plot_sessions_pst(ax, macd_h.index)
            ax.set_title("Hourly MACD")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[6]:
    st.subheader("Momentum (ROC%)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Daily ROC%**")
        close_show = subset_by_daily_view(df_close, daily_view)
        roc_d = compute_roc(close_show, n=10)
        fig = plt.figure(figsize=(9, 3.4))
        ax = fig.add_subplot(111)
        ax.plot(roc_d.index, roc_d.values, linewidth=1.4, label="ROC% (10)")
        ax.axhline(0.0, linewidth=1.0, alpha=0.4)
        style_axes(ax)
        ax.set_title("Daily Momentum")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly ROC%**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            roc_h = compute_roc(close_h, n=mom_lb_hourly)
            fig = plt.figure(figsize=(9, 3.4))
            ax = fig.add_subplot(111)
            ax.plot(roc_h.index, roc_h.values, linewidth=1.4, label=f"ROC% ({mom_lb_hourly})")
            ax.axhline(0.0, linewidth=1.0, alpha=0.4)
            style_axes(ax)
            plot_sessions_pst(ax, roc_h.index)
            ax.set_title("Hourly Momentum")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[7]:
    st.subheader("2σ Reversal (Bands + Slope Trigger)")
    st.caption("This tab uses linear regression ±2σ bands and slope-cross triggers.")

    close_show = subset_by_daily_view(df_close, daily_view)
    yhat, upper, lower, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)

    # Reversal probability (experimental)
    prob = slope_reversal_probability(
        close_show,
        current_slope=m,
        hist_window=rev_hist_lb,
        slope_window=ntd_window,
        horizon=rev_horizon
    )

    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(111)
    ax.plot(close_show.index, close_show.values, linewidth=1.6, label="Price")
    if yhat.dropna().shape[0] > 0:
        ax.plot(yhat.index, yhat.values, linewidth=2.0, linestyle="--", alpha=0.9, label="Slope Line")
    if upper.dropna().shape[0] > 0 and lower.dropna().shape[0] > 0:
        ax.plot(upper.index, upper.values, linewidth=1.0, linestyle="--", alpha=0.7, label="+2σ")
        ax.plot(lower.index, lower.values, linewidth=1.0, linestyle="--", alpha=0.7, label="-2σ")

    style_axes(ax)
    ax.set_title(f"2σ Bands + Slope Trigger — {disp_ticker}")
    _safe_ylim(ax, close_show)

    # Band bounce signal
    bounce = find_band_bounce_signal(close_show, upper, lower, slope_val=m)
    if bounce:
        annotate_crossover(ax, bounce["time"], bounce["price"], bounce["side"], note="(2σ bounce)")

    # Slope trigger after band reversal
    trig = find_slope_trigger_after_band_reversal(close_show, yhat, upper, lower, horizon=rev_horizon)
    annotate_slope_trigger(ax, trig)

    _make_legend_compact(ax)

    st.metric("Reversal probability (experimental)", fmt_pct(prob, digits=1) if np.isfinite(prob) else "n/a")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

with tabs[8]:
    st.subheader("Fib 0/100 Reversal (Confirmed)")
    st.info(FIB_ALERT_TEXT)

    close_show = subset_by_daily_view(df_close, daily_view)
    trig_d = fib_reversal_trigger_from_extremes(close_show, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=60)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Daily Confirmation**")
        if trig_d is None:
            st.write("No confirmed fib reversal (daily).")
        else:
            st.success(f"Confirmed **{trig_d['side']}** from **{trig_d['from_level']}** "
                       f"(touch: {trig_d['touch_time']}, last: {trig_d['last_time']})")

        fig = plt.figure(figsize=(9.2, 4.6))
        ax = fig.add_subplot(111)
        ax.plot(close_show.index, close_show.values, linewidth=1.6, label="Daily Close")
        style_axes(ax)
        if show_fibs:
            plot_fibonacci(ax, close_show, label_prefix="Fib")
        ax.set_title(f"Daily Fib Levels — {disp_ticker}")
        _make_legend_compact(ax)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with c2:
        st.markdown("**Hourly Confirmation**")
        if hourly_out and "close_hourly" in hourly_out:
            close_h = hourly_out["close_hourly"]
            trig_h = fib_reversal_trigger_from_extremes(close_h, proximity_pct_of_range=0.02, confirm_bars=2, lookback_bars=80)
            if trig_h is None:
                st.write("No confirmed fib reversal (hourly).")
            else:
                st.success(f"Confirmed **{trig_h['side']}** from **{trig_h['from_level']}** "
                           f"(touch: {trig_h['touch_time']}, last: {trig_h['last_time']})")

            fig = plt.figure(figsize=(9.2, 4.6))
            ax = fig.add_subplot(111)
            ax.plot(close_h.index, close_h.values, linewidth=1.6, label="Hourly Close")
            style_axes(ax)
            plot_sessions_pst(ax, close_h.index)
            if show_fibs:
                plot_fibonacci(ax, close_h, label_prefix="Fib")
            ax.set_title(f"Hourly Fib Levels — {disp_ticker}")
            _make_legend_compact(ax)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        else:
            st.write("Run Hourly view first.")

with tabs[9]:
    render_scanners(universe)

# Footer note
st.caption("Signals are informational only. Use risk management and confirm with your own analysis.")
