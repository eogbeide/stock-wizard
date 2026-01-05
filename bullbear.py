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

# UPDATED (THIS REQUEST): keep hourly fib toggle + add daily fib toggle
show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly)", value=True, key="sb_show_fibs")
show_fibs_daily = st.sidebar.checkbox("Show Fibonacci (daily)", value=True, key="sb_show_fibs_daily")

# UPDATED (THIS REQUEST): user note
st.sidebar.info('Place **BUY** trade closer to **0% Fibonacci** and **SELL** trade closer to **100% Fibonacci**.')

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
# NEW (THIS REQUEST): Slope BUY/SELL Trigger (leaderline + legend)
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
# NEW (THIS REQUEST): Fibonacci Confirmation Trigger (0%/100% + reversal + R²≥0.999)
# ---------------------------
def find_fib_confirmation_trigger(price: pd.Series,
                                 fibs: dict,
                                 r2_val: float,
                                 prox: float = 0.0025,
                                 bars_confirm: int = 2,
                                 min_r2: float = 0.999,
                                 horizon: int = 15):
    """
    CONFIRMATION TRIGGER:
      - Price touches 0% Fibonacci (high) OR 100% Fibonacci (low)
      - Then reverses with confirmation (consecutive bars)
      - AND regression R² >= 0.999 (treated as 99.9% confidence)
    Returns most recent confirmation dict or None.
    """
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        return None

    try:
        r2v = float(r2_val)
    except Exception:
        r2v = np.nan

    if not (np.isfinite(r2v) and r2v >= float(min_r2)):
        return None

    hi = fibs.get("0%", np.nan)
    lo = fibs.get("100%", np.nan)
    try:
        hi = float(hi); lo = float(lo)
    except Exception:
        hi = np.nan; lo = np.nan
    if not (np.isfinite(hi) and np.isfinite(lo)) or hi == lo:
        return None

    hz = max(1, int(horizon))
    tail = p.iloc[-min(len(p), max(hz * 4, 25)):]  # keep a reasonable recent window

    # Touch definitions (near extremes)
    touch_hi = tail >= (hi * (1.0 - float(prox)))
    touch_lo = tail <= (lo * (1.0 + float(prox)))

    def _last_confirm_after_touch(touch_mask: pd.Series, side: str):
        if not touch_mask.any():
            return None
        t_touch = touch_mask[touch_mask].index[-1]
        try:
            loc0 = int(tail.index.get_loc(t_touch))
        except Exception:
            return None

        # Search for a confirmed reversal after touch
        last_conf = None
        for j in range(loc0 + bars_confirm, len(tail)):
            seg = tail.iloc[:j+1]
            if side == "BUY":
                if _n_consecutive_increasing(seg, bars_confirm):
                    last_conf = seg.index[-1]
            else:
                if _n_consecutive_decreasing(seg, bars_confirm):
                    last_conf = seg.index[-1]
        if last_conf is None:
            # If touch is very recent, allow last bar check
            if side == "BUY" and _n_consecutive_increasing(tail, bars_confirm):
                last_conf = tail.index[-1]
            if side == "SELL" and _n_consecutive_decreasing(tail, bars_confirm):
                last_conf = tail.index[-1]

        if last_conf is None:
            return None

        px = float(tail.loc[last_conf]) if np.isfinite(tail.loc[last_conf]) else np.nan
        if not np.isfinite(px):
            return None

        return {
            "side": side,
            "time": last_conf,
            "price": px,
            "r2": r2v,
            "touched": "100%" if side == "BUY" else "0%",
            "touch_time": t_touch
        }

    buy_conf = _last_confirm_after_touch(touch_lo, side="BUY")
    sell_conf = _last_confirm_after_touch(touch_hi, side="SELL")

    if buy_conf is None and sell_conf is None:
        return None
    if buy_conf is None:
        return sell_conf
    if sell_conf is None:
        return buy_conf
    return buy_conf if buy_conf["time"] >= sell_conf["time"] else sell_conf

def annotate_fib_confirmation(ax, trig: dict):
    if trig is None:
        return
    t = trig.get("time", None)
    px = trig.get("price", np.nan)
    side = trig.get("side", "")
    touched = trig.get("touched", "")
    if t is None or not np.isfinite(px):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    mkr = "D" if side == "BUY" else "X"
    lbl = f"CONFIRM {side} (99.9%) @ {touched}"
    ax.scatter([t], [px], marker=mkr, s=140, color=col, zorder=13, label=lbl)
    ax.text(t, px, f"  {lbl}", color=col, fontsize=9, fontweight="bold",
            va="bottom" if side == "BUY" else "top", zorder=13)
# =========================
# Part 6/10 — bullbear.py
# =========================
# ---------------------------
# Support/Resistance + Signal helpers
# ---------------------------
def compute_support_resistance(ohlc: pd.DataFrame, lookback: int = 60):
    """
    Support = rolling LOW min
    Resistance = rolling HIGH max
    """
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        idx = ohlc.index if isinstance(ohlc, pd.DataFrame) else None
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    lb = max(2, int(lookback))
    sup = _coerce_1d_series(ohlc["Low"]).rolling(lb, min_periods=max(2, lb//3)).min()
    res = _coerce_1d_series(ohlc["High"]).rolling(lb, min_periods=max(2, lb//3)).max()
    return sup.reindex(ohlc.index), res.reindex(ohlc.index)

def compute_bb_cross_signal(close: pd.Series,
                            bb_mid: pd.Series,
                            fibs: dict,
                            r2_val: float,
                            global_slope: float,
                            local_slope: float,
                            prox: float = 0.0025,
                            bars_confirm: int = 2,
                            min_r2: float = 0.999):
    """
    BB Buy Cross:
      - Touch (near) 100% fib (low) then reverse upward (confirmed bars)
      - Cross ABOVE BB midline
      - Global & Local slopes UP
      - R² >= 0.999 (99.9% confidence gate)

    BB Sell Cross:
      - Touch (near) 0% fib (high) then reverse downward (confirmed bars)
      - Cross BELOW BB midline
      - Global & Local slopes DOWN
      - R² >= 0.999
    """
    c = _coerce_1d_series(close).dropna()
    m = _coerce_1d_series(bb_mid).reindex(c.index)

    ok = c.notna() & m.notna()
    if ok.sum() < 5:
        return None
    c = c[ok]; m = m[ok]

    try:
        r2v = float(r2_val)
    except Exception:
        r2v = np.nan
    if not (np.isfinite(r2v) and r2v >= float(min_r2)):
        return None

    try:
        g = float(global_slope); l = float(local_slope)
    except Exception:
        return None
    if not (np.isfinite(g) and np.isfinite(l)):
        return None

    fib_hi = fibs.get("0%", np.nan)
    fib_lo = fibs.get("100%", np.nan)
    try:
        fib_hi = float(fib_hi); fib_lo = float(fib_lo)
    except Exception:
        fib_hi = np.nan; fib_lo = np.nan
    if not (np.isfinite(fib_hi) and np.isfinite(fib_lo)) or fib_hi == fib_lo:
        return None

    # Cross masks
    cross_up, cross_dn = _cross_series(c, m)
    cross_up = cross_up.reindex(c.index, fill_value=False)
    cross_dn = cross_dn.reindex(c.index, fill_value=False)

    # Touch masks
    touch_hi = c >= (fib_hi * (1.0 - prox))  # near 0% fib
    touch_lo = c <= (fib_lo * (1.0 + prox))  # near 100% fib

    # Reversal confirmation at the end
    rev_up_ok = _n_consecutive_increasing(c, bars_confirm)
    rev_dn_ok = _n_consecutive_decreasing(c, bars_confirm)

    # Require that touch happened recently (within last ~horizon)
    hz = max(10, bars_confirm * 5)
    tail_idx = c.index[-hz:] if len(c) >= hz else c.index

    touched_lo_recent = bool(touch_lo.reindex(tail_idx, fill_value=False).any())
    touched_hi_recent = bool(touch_hi.reindex(tail_idx, fill_value=False).any())

    # Last cross times
    last_up = cross_up[cross_up].index[-1] if cross_up.any() else None
    last_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None

    buy_ok = (g > 0 and l > 0 and touched_lo_recent and rev_up_ok and (last_up is not None))
    sell_ok = (g < 0 and l < 0 and touched_hi_recent and rev_dn_ok and (last_dn is not None))

    buy_sig = None
    if buy_ok:
        t = last_up
        buy_sig = {"time": t, "price": float(c.loc[t]), "side": "BUY", "note": "BB Buy Cross (99.9%)"}

    sell_sig = None
    if sell_ok:
        t = last_dn
        sell_sig = {"time": t, "price": float(c.loc[t]), "side": "SELL", "note": "BB Sell Cross (99.9%)"}

    if buy_sig is None and sell_sig is None:
        return None
    if buy_sig is None:
        return sell_sig
    if sell_sig is None:
        return buy_sig
    return buy_sig if buy_sig["time"] >= sell_sig["time"] else sell_sig

def annotate_bb_cross(ax, sig: dict):
    if sig is None:
        return
    t = sig.get("time", None)
    px = sig.get("price", np.nan)
    side = sig.get("side", "")
    note = sig.get("note", "")
    if t is None or not np.isfinite(px):
        return
    if side == "BUY":
        ax.scatter([t], [px], marker="^", s=140, color="tab:green", zorder=14, label="BB Buy Cross")
        ax.text(t, px, f"  {note}", color="tab:green", fontsize=9, fontweight="bold", va="bottom", zorder=14)
    else:
        ax.scatter([t], [px], marker="v", s=140, color="tab:red", zorder=14, label="BB Sell Cross")
        ax.text(t, px, f"  {note}", color="tab:red", fontsize=9, fontweight="bold", va="top", zorder=14)

def fib_proximity_metrics(close_val: float, fibs: dict):
    """
    Returns normalized distances to fib extremes (0% and 100%).
    """
    try:
        c = float(close_val)
    except Exception:
        return {"dist_to_0": np.nan, "dist_to_100": np.nan, "range": np.nan}

    try:
        f0 = float(fibs.get("0%", np.nan))
        f100 = float(fibs.get("100%", np.nan))
    except Exception:
        f0 = np.nan; f100 = np.nan

    if not (np.isfinite(c) and np.isfinite(f0) and np.isfinite(f100)) or f0 == f100:
        return {"dist_to_0": np.nan, "dist_to_100": np.nan, "range": np.nan}

    rng = abs(f0 - f100)
    dist0 = abs(c - f0) / rng
    dist100 = abs(c - f100) / rng
    return {"dist_to_0": float(dist0), "dist_to_100": float(dist100), "range": float(rng)}
# =========================
# Part 7/10 — bullbear.py
# =========================
# ---------------------------
# Plotting helpers (price + NTD + MACD)
# ---------------------------
def _plot_fibs(ax, fibs: dict, label_prefix: str = "Fib"):
    if not fibs:
        return
    # Keep same clean look: horizontal lines + left labels
    for k, v in fibs.items():
        try:
            y = float(v)
        except Exception:
            continue
        if not np.isfinite(y):
            continue
        ax.axhline(y, linewidth=1.0, alpha=0.55, linestyle="--")
        label_on_left(ax, y, f"{label_prefix} {k}: {fmt_price_val(y)}", fontsize=8)

def _plot_supertrend(ax, ohlc: pd.DataFrame, st_df: pd.DataFrame):
    if st_df is None or st_df.empty or "ST" not in st_df.columns:
        return
    st_line = _coerce_1d_series(st_df["ST"]).reindex(ohlc.index)
    up = st_df.get("in_uptrend", pd.Series(index=ohlc.index, dtype=bool)).reindex(ohlc.index).fillna(True)

    # Draw as a continuous line; color by trend (simple + consistent)
    # Keep look/feel: solid line, modest thickness
    if not st_line.dropna().empty:
        # Split segments
        idx = st_line.index
        stv = st_line.values
        upv = up.values.astype(bool)
        # Plot as two masked lines
        ax.plot(idx, np.where(upv, stv, np.nan), linewidth=2.0, alpha=0.9, label="Supertrend (UP)")
        ax.plot(idx, np.where(~upv, stv, np.nan), linewidth=2.0, alpha=0.9, label="Supertrend (DN)")

def _plot_psar(ax, psar_df: pd.DataFrame, idx_ref: pd.DatetimeIndex):
    if psar_df is None or psar_df.empty or "PSAR" not in psar_df.columns:
        return
    ps = _coerce_1d_series(psar_df["PSAR"]).reindex(idx_ref)
    up = psar_df.get("in_uptrend", pd.Series(index=idx_ref, dtype=bool)).reindex(idx_ref).fillna(True)
    if ps.dropna().empty:
        return
    ax.scatter(ps.index, ps.values, s=14, alpha=0.85, marker=".", label="PSAR")

def _plot_ichimoku(ax, ohlc: pd.DataFrame):
    if ohlc is None or ohlc.empty:
        return
    if not {"High","Low","Close"}.issubset(ohlc.columns):
        return
    tenkan, kijun, sen_a, sen_b, chikou = ichimoku_lines(
        ohlc["High"], ohlc["Low"], ohlc["Close"],
        conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False
    )
    # Only draw Kijun on price (per sidebar label)
    if kijun is not None and not _coerce_1d_series(kijun).dropna().empty:
        ax.plot(kijun.index, kijun.values, linewidth=1.6, alpha=0.85, label="Kijun(26)")

def build_price_ntd_figure(ohlc: pd.DataFrame,
                           title: str,
                           slope_lb: int,
                           sr_lb: int,
                           show_fibs_flag: bool,
                           fib_label: str,
                           show_sessions: bool = False,
                           is_hourly: bool = False):
    """
    Build a 2-row figure:
      row1: price + overlays
      row2: NTD (+ optional overlays)
    """
    if ohlc is None or ohlc.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(f"{title} (no data)")
        style_axes(ax)
        return fig

    close = _coerce_1d_series(ohlc["Close"])
    high = _coerce_1d_series(ohlc["High"])
    low  = _coerce_1d_series(ohlc["Low"])

    # Core computations
    yhat, upper2, lower2, local_slope, r2_local = regression_with_band(close, lookback=int(slope_lb), z=2.0)

    # Global trendline slope computed on full visible slice (same look/feel: dashed line)
    yhat_g, global_slope = slope_line(close, lookback=len(close))

    # S/R
    sup, res = compute_support_resistance(ohlc, lookback=int(sr_lb))

    # Bollinger
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    # NTD + NPX
    ntd = compute_normalized_trend(close, window=int(ntd_window))
    npx = compute_normalized_price(close, window=int(ntd_window))

    # Supertrend (ON by default)
    st_df = compute_supertrend(ohlc[["High","Low","Close"]], atr_period=int(atr_period), atr_mult=float(atr_mult))

    # PSAR
    psar_df = compute_psar_from_ohlc(ohlc[["High","Low"]], step=float(psar_step), max_step=float(psar_max)) if show_psar else None

    # Fibonacci (ON per requested toggles)
    fibs = fibonacci_levels(close)

    # Signals: BB Cross + Slope Trigger + Fib Confirmation
    bb_sig = compute_bb_cross_signal(
        close=close, bb_mid=bb_mid, fibs=fibs, r2_val=r2_local,
        global_slope=global_slope, local_slope=local_slope,
        prox=float(sr_prox_pct), bars_confirm=int(rev_bars_confirm), min_r2=0.999
    )

    slope_trig = find_slope_trigger_after_band_reversal(
        price=close, yhat=yhat, upper_band=upper2, lower_band=lower2, horizon=int(rev_horizon)
    )

    fib_conf = find_fib_confirmation_trigger(
        price=close, fibs=fibs, r2_val=r2_local, prox=float(sr_prox_pct),
        bars_confirm=int(rev_bars_confirm), min_r2=0.999, horizon=int(rev_horizon)
    )

    # Figure layout (keep simple + consistent)
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.35], hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # --- PRICE PANEL ---
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.plot(close.index, close.values, linewidth=1.8, label="Close")

    # Regression trend + ±2σ
    if not yhat.dropna().empty:
        ax1.plot(yhat.index, yhat.values, linestyle="-", linewidth=2.0, alpha=0.9, label=f"Local Trend ({fmt_slope(local_slope)})")
    if not upper2.dropna().empty and not lower2.dropna().empty:
        ax1.plot(upper2.index, upper2.values, linestyle="--", linewidth=1.4, alpha=0.75, label="+2σ")
        ax1.plot(lower2.index, lower2.values, linestyle="--", linewidth=1.4, alpha=0.75, label="-2σ")

    # Global dashed trendline (keep the feel: dashed, bold)
    if not yhat_g.dropna().empty:
        gcol = "green" if global_slope >= 0 else "red"
        ax1.plot(yhat_g.index, yhat_g.values, linestyle="--", linewidth=2.4, alpha=0.9, color=gcol, label=f"Global Trend ({fmt_slope(global_slope)})")

    # S/R lines
    if sup is not None and not _coerce_1d_series(sup).dropna().empty:
        ax1.plot(sup.index, sup.values, linewidth=1.2, alpha=0.75, label="Support")
    if res is not None and not _coerce_1d_series(res).dropna().empty:
        ax1.plot(res.index, res.values, linewidth=1.2, alpha=0.75, label="Resistance")

    # Bollinger Bands
    if show_bbands and not bb_mid.dropna().empty:
        ax1.plot(bb_mid.index, bb_mid.values, linewidth=1.4, alpha=0.85, label="BB Mid")
        ax1.plot(bb_up.index, bb_up.values, linewidth=1.0, alpha=0.6, label="BB Upper")
        ax1.plot(bb_lo.index, bb_lo.values, linewidth=1.0, alpha=0.6, label="BB Lower")

    # Ichimoku Kijun
    if show_ichi:
        _plot_ichimoku(ax1, ohlc)

    # Supertrend
    _plot_supertrend(ax1, ohlc, st_df)

    # PSAR
    if show_psar and psar_df is not None:
        _plot_psar(ax1, psar_df, close.index)

    # Fibonacci
    if show_fibs_flag:
        _plot_fibs(ax1, fibs, label_prefix=fib_label)

    # Trade instruction (uses both global + local gating)
    try:
        close_last = float(close.dropna().iloc[-1])
    except Exception:
        close_last = np.nan
    if np.isfinite(close_last):
        # Use fib extremes as "reference" (buy closer to low, sell closer to high)
        buy_ref = fibs.get("100%", close_last)
        sell_ref = fibs.get("0%", close_last)
        txt = format_trade_instruction(
            trend_slope=float(local_slope),
            buy_val=float(buy_ref) if np.isfinite(float(buy_ref)) else close_last,
            sell_val=float(sell_ref) if np.isfinite(float(sell_ref)) else close_last,
            close_val=close_last,
            symbol=title,
            global_trend_slope=float(global_slope)
        )
        ax1.text(0.01, 0.98, txt, transform=ax1.transAxes, ha="left", va="top",
                 fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.65))

    # R² badge
    ax1.text(0.99, 0.98, f"R²: {fmt_r2(r2_local, 1)}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.65))

    # Triggers
    annotate_bb_cross(ax1, bb_sig)
    annotate_slope_trigger(ax1, slope_trig)
    annotate_fib_confirmation(ax1, fib_conf)

    # Sessions (Forex optional)
    if show_sessions and isinstance(close.index, pd.DatetimeIndex) and len(close.index) > 5:
        # Keep minimal and consistent: faint vertical spans
        # Approx sessions in PST (not DST-precise; matches prior "simple markers" feel)
        # London: 00:00–08:00 PST, NY: 05:00–13:00 PST
        idx = close.index
        days = pd.to_datetime(pd.Series(idx.date).unique())
        for d in days:
            try:
                d0 = pd.Timestamp(d).tz_localize(PACIFIC)
            except Exception:
                try:
                    d0 = pd.Timestamp(d).tz_convert(PACIFIC)
                except Exception:
                    continue
            lon0 = d0 + pd.Timedelta(hours=0)
            lon1 = d0 + pd.Timedelta(hours=8)
            ny0  = d0 + pd.Timedelta(hours=5)
            ny1  = d0 + pd.Timedelta(hours=13)
            ax1.axvspan(lon0, lon1, alpha=0.04)
            ax1.axvspan(ny0, ny1, alpha=0.04)

    style_axes(ax1)

    # --- NTD PANEL ---
    ax2.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax2.axhline(0.75, linewidth=1.0, alpha=0.45, linestyle="--")
    ax2.axhline(-0.75, linewidth=1.0, alpha=0.45, linestyle="--")
    ax2.plot(ntd.index, ntd.values, linewidth=1.8, label="NTD")

    if shade_ntd:
        shade_ntd_regions(ax2, ntd)

    if show_npx_ntd:
        overlay_npx_on_ntd(ax2, npx, ntd, mark_crosses=mark_npx_cross)

    # Reversal markers on NTD (stars)
    overlay_ntd_sr_reversal_stars(
        ax2, price=close, sup=sup, res=res, trend_slope=float(global_slope),
        ntd=ntd, prox=float(sr_prox_pct), bars_confirm=int(rev_bars_confirm)
    )

    # HMA reversal markers on NTD
    if show_hma_rev_ntd and show_hma:
        hma = compute_hma(close, period=int(hma_period))
        overlay_hma_reversal_on_ntd(ax2, close, hma, lookback=int(hma_rev_lb), period=int(hma_period))

    # Triangles by trend
    overlay_ntd_triangles_by_trend(ax2, ntd, trend_slope=float(global_slope), upper=0.75, lower=-0.75)

    ax2.set_ylim(-1.05, 1.05)
    style_axes(ax2)
    ax2.legend(loc="upper left", fontsize=8, frameon=True)

    # Legend (price)
    ax1.legend(loc="lower left", fontsize=8, frameon=True)

    return fig

def build_macd_figure(close: pd.Series, title: str):
    macd, sig, hist = compute_macd(close)
    fig, ax = plt.subplots(figsize=(13.5, 2.7))
    ax.set_title(f"{title} — MACD", fontsize=11, fontweight="bold")
    ax.axhline(0.0, linewidth=1.0, alpha=0.6)
    ax.plot(macd.index, macd.values, linewidth=1.5, label="MACD")
    ax.plot(sig.index, sig.values, linewidth=1.5, label="Signal")
    # histogram as line for consistent minimal style
    ax.plot(hist.index, hist.values, linewidth=1.0, alpha=0.7, label="Hist")
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    return fig
# =========================
# Part 8/10 — bullbear.py
# =========================
# ---------------------------
# Hourly data + builders
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hourly_ohlc(ticker: str, period: str = "60d") -> pd.DataFrame:
    """
    Hourly OHLC for charts + hourly scanner.
    """
    df = yf.download(ticker, period=period, interval="60m")
    if df is None or df.empty:
        return df
    # Ensure timezone PST
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)
    # Gapless adjustment (optional but keeps feel consistent with intraday)
    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df, price_cols=("Open","High","Low","Close"))
    return df[["Open","High","Low","Close","Volume"]].dropna(how="all") if "Volume" in df.columns else df[["Open","High","Low","Close"]].dropna(how="all")

def subset_hourly(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        return df
    label = str(label)
    if label == "1W":
        return df.iloc[-24*7:]
    if label == "2W":
        return df.iloc[-24*14:]
    if label == "1M":
        return df.iloc[-24*30:]
    if label == "3M":
        return df.iloc[-24*90:]
    return df  # "All"

def render_daily_section(symbol: str, daily_ohlc: pd.DataFrame):
    dfv = subset_by_daily_view(daily_ohlc, daily_view)
    fig = build_price_ntd_figure(
        ohlc=dfv,
        title=f"{symbol} — Daily",
        slope_lb=int(slope_lb_daily),
        sr_lb=int(sr_lb_daily),
        show_fibs_flag=bool(show_fibs_daily),
        fib_label="Daily",
        show_sessions=False,
        is_hourly=False
    )
    st.pyplot(fig, clear_figure=True)
    if show_macd:
        close = _coerce_1d_series(dfv["Close"])
        st.pyplot(build_macd_figure(close, f"{symbol} — Daily"), clear_figure=True)

def render_hourly_section(symbol: str, hourly_ohlc: pd.DataFrame, hour_range: str):
    dfv = subset_hourly(hourly_ohlc, hour_range)
    fig = build_price_ntd_figure(
        ohlc=dfv,
        title=f"{symbol} — Hourly ({hour_range})",
        slope_lb=int(slope_lb_hourly),
        sr_lb=int(sr_lb_hourly),
        show_fibs_flag=bool(show_fibs),
        fib_label="Hourly",
        show_sessions=bool(show_sessions_pst and mode == "Forex"),
        is_hourly=True
    )
    st.pyplot(fig, clear_figure=True)
    if show_macd:
        close = _coerce_1d_series(dfv["Close"])
        st.pyplot(build_macd_figure(close, f"{symbol} — Hourly"), clear_figure=True)

# ---------------------------
# Scanner helpers (R² and Fib proximity)
# ---------------------------
@st.cache_data(ttl=180)
def scan_r2_trends_daily(symbols, view_label: str, slope_lb: int, r2_min: float = 0.45):
    rows_up = []
    rows_dn = []
    for sym in symbols:
        try:
            ohlc = fetch_hist_ohlc(sym)
            ohlc = subset_by_daily_view(ohlc, view_label)
            close = _coerce_1d_series(ohlc["Close"]).dropna()
            if len(close) < max(30, slope_lb):
                continue
            _, _, _, m, r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)
            if not (np.isfinite(r2) and r2 >= float(r2_min) and np.isfinite(m) and m != 0.0):
                continue
            last_close = float(close.iloc[-1])
            row = {
                "Symbol": sym,
                "Slope": float(m),
                "R²": float(r2),
                "Close": float(last_close),
            }
            if m > 0:
                rows_up.append(row)
            else:
                rows_dn.append(row)
        except Exception:
            continue
    df_up = pd.DataFrame(rows_up).sort_values(["R²","Slope"], ascending=[False, False]) if rows_up else pd.DataFrame(columns=["Symbol","Slope","R²","Close"])
    df_dn = pd.DataFrame(rows_dn).sort_values(["R²","Slope"], ascending=[False, True]) if rows_dn else pd.DataFrame(columns=["Symbol","Slope","R²","Close"])
    return df_up, df_dn

@st.cache_data(ttl=180)
def scan_r2_trends_hourly(symbols, hour_range_label: str, slope_lb: int, r2_min: float = 0.45):
    rows_up = []
    rows_dn = []
    for sym in symbols:
        try:
            ohlc = fetch_hourly_ohlc(sym, period="90d" if hour_range_label in ("3M","All") else "60d")
            ohlc = subset_hourly(ohlc, hour_range_label)
            if ohlc is None or ohlc.empty:
                continue
            close = _coerce_1d_series(ohlc["Close"]).dropna()
            if len(close) < max(40, slope_lb):
                continue
            _, _, _, m, r2 = regression_with_band(close, lookback=int(slope_lb), z=2.0)
            if not (np.isfinite(r2) and r2 >= float(r2_min) and np.isfinite(m) and m != 0.0):
                continue
            last_close = float(close.iloc[-1])
            row = {
                "Symbol": sym,
                "Slope": float(m),
                "R²": float(r2),
                "Close": float(last_close),
            }
            if m > 0:
                rows_up.append(row)
            else:
                rows_dn.append(row)
        except Exception:
            continue
    df_up = pd.DataFrame(rows_up).sort_values(["R²","Slope"], ascending=[False, False]) if rows_up else pd.DataFrame(columns=["Symbol","Slope","R²","Close"])
    df_dn = pd.DataFrame(rows_dn).sort_values(["R²","Slope"], ascending=[False, True]) if rows_dn else pd.DataFrame(columns=["Symbol","Slope","R²","Close"])
    return df_up, df_dn

@st.cache_data(ttl=180)
def scan_fib_proximity_daily(symbols, view_label: str):
    """
    NEW (THIS REQUEST): symbols closest to 0% fib and 100% fib on DAILY.
    """
    rows = []
    for sym in symbols:
        try:
            ohlc = fetch_hist_ohlc(sym)
            ohlc = subset_by_daily_view(ohlc, view_label)
            if ohlc is None or ohlc.empty:
                continue
            close = _coerce_1d_series(ohlc["Close"]).dropna()
            if len(close) < 30:
                continue
            fibs = fibonacci_levels(close)
            if not fibs:
                continue
            last_close = float(close.iloc[-1])
            proxm = fib_proximity_metrics(last_close, fibs)
            rows.append({
                "Symbol": sym,
                "Close": last_close,
                "Fib 0%": float(fibs.get("0%", np.nan)),
                "Fib 100%": float(fibs.get("100%", np.nan)),
                "Dist→0% (norm)": proxm["dist_to_0"],
                "Dist→100% (norm)": proxm["dist_to_100"],
            })
        except Exception:
            continue

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Symbol","Close","Fib 0%","Fib 100%","Dist→0% (norm)","Dist→100% (norm)"])
    if df.empty:
        return df, df
    df0 = df.sort_values("Dist→0% (norm)", ascending=True).reset_index(drop=True)
    df100 = df.sort_values("Dist→100% (norm)", ascending=True).reset_index(drop=True)
    return df0, df100
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Main UI: selector + tabs
# ---------------------------
st.sidebar.subheader("Symbol")
default_symbol = universe[0] if universe else ""
ticker = st.sidebar.selectbox("Choose symbol", universe, index=0, key="sb_ticker")
st.session_state.ticker = ticker

st.sidebar.subheader("Hourly range")
hour_range = st.sidebar.selectbox("Hourly range", ["1W","2W","1M","3M","All"], index=2, key="sb_hour_range")
st.session_state.hour_range = hour_range

run_btn = st.sidebar.button("▶ Run / Refresh Analysis", use_container_width=True, key="btn_run")
if run_btn:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# If mode changed after run, don't show stale charts
if st.session_state.get("mode_at_run", mode) != mode:
    st.session_state.run_all = False

# Tabs (keep consistent, add new requested tab)
tab_overview, tab_daily, tab_hourly, tab_forecast, tab_r2scan, tab_fibscan = st.tabs([
    "📌 Overview",
    "📅 Daily",
    "⏱ Hourly",
    "🔮 Forecast",
    "📈 R² Trend Scanner",
    "🧲 Fibonacci Proximity"
])

with tab_overview:
    st.subheader("Overview")
    st.write(f"**Mode:** {mode}")
    st.write(f"**Symbol:** `{ticker}`")

    st.markdown("""
**Notes**
- Supertrend is shown by default on Daily and Hourly charts.
- Fibonacci is shown on Hourly by default; Daily Fibonacci can be toggled on the left.
- **Place BUY trade closer to 0% Fibonacci and SELL trade closer to 100% Fibonacci.**
""")

    if not st.session_state.get("run_all", False):
        st.info("Click **Run / Refresh Analysis** in the sidebar to load charts and scanners.")

with tab_daily:
    st.subheader("Daily Chart")
    if st.session_state.get("run_all", False):
        try:
            df_daily = fetch_hist_ohlc(ticker)
            st.session_state.df_ohlc = df_daily
            render_daily_section(ticker, df_daily)
        except Exception as e:
            st.error(f"Daily chart error: {e}")
    else:
        st.info("Run analysis to display the daily chart.")

with tab_hourly:
    st.subheader("Hourly Chart")
    if st.session_state.get("run_all", False):
        try:
            df_hourly = fetch_hourly_ohlc(ticker, period="90d" if hour_range in ("3M","All") else "60d")
            st.session_state.intraday = df_hourly
            render_hourly_section(ticker, df_hourly, hour_range)
        except Exception as e:
            st.error(f"Hourly chart error: {e}")
    else:
        st.info("Run analysis to display the hourly chart.")

with tab_forecast:
    st.subheader("30-Day SARIMAX Forecast (Daily Close)")
    if st.session_state.get("run_all", False):
        try:
            s = fetch_hist(ticker)
            idx, fc, ci = compute_sarimax_forecast(s)
            fig, ax = plt.subplots(figsize=(13.5, 4.5))
            ax.set_title(f"{ticker} — Forecast", fontsize=12, fontweight="bold")
            ax.plot(s.index, s.values, linewidth=1.6, label="History")
            ax.plot(idx, fc.values, linewidth=2.0, label="Forecast")
            try:
                ax.fill_between(idx, ci.iloc[:, 0].values, ci.iloc[:, 1].values, alpha=0.18)
            except Exception:
                pass
            style_axes(ax)
            ax.legend(loc="upper left", fontsize=8, frameon=True)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.error(f"Forecast error: {e}")
    else:
        st.info("Run analysis to compute the forecast.")

with tab_r2scan:
    st.subheader("R² Trend Scanner (R² > 45%) — Uptrend vs Downtrend")
    st.caption("Daily uses your Daily view range and Daily slope lookback. Hourly uses your Hourly range and Hourly slope lookback.")

    if st.session_state.get("run_all", False):
        with st.spinner("Scanning symbols for R² trend strength..."):
            try:
                d_up, d_dn = scan_r2_trends_daily(universe, daily_view, int(slope_lb_daily), r2_min=0.45)
                h_up, h_dn = scan_r2_trends_hourly(universe, hour_range, int(slope_lb_hourly), r2_min=0.45)
            except Exception as e:
                st.error(f"Scanner error: {e}")
                d_up = d_dn = h_up = h_dn = pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Daily — Uptrend")
            st.dataframe(d_up, use_container_width=True, height=340)
        with c2:
            st.markdown("### Daily — Downtrend")
            st.dataframe(d_dn, use_container_width=True, height=340)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("### Hourly — Uptrend")
            st.dataframe(h_up, use_container_width=True, height=340)
        with c4:
            st.markdown("### Hourly — Downtrend")
            st.dataframe(h_dn, use_container_width=True, height=340)
    else:
        st.info("Run analysis to populate the R² scanner.")

with tab_fibscan:
    st.subheader("Fibonacci Proximity — Closest to 0% and 100% (Daily)")
    st.caption("NEW: This tab lists symbols that are **closest to 0% Fibonacci** (near the high) and **closest to 100% Fibonacci** (near the low).")

    if st.session_state.get("run_all", False):
        with st.spinner("Scanning symbols for Fibonacci proximity..."):
            try:
                df0, df100 = scan_fib_proximity_daily(universe, daily_view)
            except Exception as e:
                st.error(f"Fib proximity scan error: {e}")
                df0 = df100 = pd.DataFrame()

        left, right = st.columns(2)
        with left:
            st.markdown("### Closest to **0% Fibonacci** (near high)")
            st.dataframe(df0.head(30), use_container_width=True, height=420)
        with right:
            st.markdown("### Closest to **100% Fibonacci** (near low)")
            st.dataframe(df100.head(30), use_container_width=True, height=420)
    else:
        st.info("Run analysis to populate the Fibonacci proximity tab.")
# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# Footer / Small runtime info (kept minimal)
# ---------------------------
st.markdown("---")
st.caption(
    "© bullbear.py • Charts: Matplotlib • Data: Yahoo Finance (yfinance) • "
    "Signals: Regression ±2σ, NTD, BB Cross, Supertrend, Fibonacci Proximity"
)

# Ensure app does not crash on first load without run
if "run_all" not in st.session_state:
    st.session_state.run_all = False

# Keep session tidy (no functional change)
for _k in ["chart"]:
    if _k not in st.session_state:
        st.session_state[_k] = None
