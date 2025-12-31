# =========================
# Part 1/8 — bullbear.py
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
# Part 2/8 — bullbear.py
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

# UPDATED (THIS REQUEST): Fibonacci should show on Daily + Hourly by default
show_fibs = st.sidebar.checkbox("Show Fibonacci (daily + hourly)", value=True, key="sb_show_fibs")

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

# UPDATED (THIS REQUEST): NPX markers are now NPX 0.0-cross markers (trend-filtered)
mark_npx_cross = st.sidebar.checkbox("Mark NPX 0.0-cross markers", value=True, key="sb_mark_npx_cross")

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
        "EURUSD=X","EURJPY=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","CADJPY=X",
        "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X","NZDJPY=X",
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
# Part 3/8 — bullbear.py
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
# Part 4/8 — bullbear.py
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

# ---------------------------
# UPDATED (THIS REQUEST): HMA55 + NPX star signal masks + plotting
# ---------------------------
def hma55_npx_star_masks(close: pd.Series,
                         hma55: pd.Series,
                         npx: pd.Series,
                         max_bar_gap: int = 2):
    """
    Blue ★:
      - Price crosses ABOVE HMA55
      - NPX crosses ABOVE -0.5 within ±{max_bar_gap} bars
      - Price diff > 0 AND NPX diff > 0 at the price-cross bar

    Purple ★:
      - Price crosses BELOW HMA55
      - NPX crosses BELOW +0.5 within ±{max_bar_gap} bars
      - Price diff < 0 AND NPX diff < 0 at the price-cross bar
    """
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(hma55).reindex(c.index)
    x = _coerce_1d_series(npx).reindex(c.index)

    ok = c.notna() & h.notna() & x.notna()
    if ok.sum() < 3:
        idx = c.index if len(c.index) else h.index
        empty = pd.Series(False, index=idx)
        return empty, empty

    c = c[ok]; h = h[ok]; x = x[ok]
    idx = c.index
    n = len(idx)
    if n < 3:
        empty = pd.Series(False, index=idx)
        return empty, empty

    price_cross_up, price_cross_dn = _cross_series(c, h)

    npx_cross_up_m05 = (x >= -0.5) & (x.shift(1) < -0.5)
    npx_cross_dn_p05 = (x <= 0.5) & (x.shift(1) > 0.5)

    dc = c.diff()
    dx = x.diff()

    blue = pd.Series(False, index=idx)
    purple = pd.Series(False, index=idx)

    mg = max(0, int(max_bar_gap))

    pu_pos = np.where(price_cross_up.values)[0]
    for i in pu_pos:
        j0 = max(0, i - mg)
        j1 = min(n - 1, i + mg)
        if bool(npx_cross_up_m05.iloc[j0:j1+1].any()):
            if np.isfinite(dc.iloc[i]) and np.isfinite(dx.iloc[i]) and (dc.iloc[i] > 0) and (dx.iloc[i] > 0):
                blue.iloc[i] = True

    pd_pos = np.where(price_cross_dn.values)[0]
    for i in pd_pos:
        j0 = max(0, i - mg)
        j1 = min(n - 1, i + mg)
        if bool(npx_cross_dn_p05.iloc[j0:j1+1].any()):
            if np.isfinite(dc.iloc[i]) and np.isfinite(dx.iloc[i]) and (dc.iloc[i] < 0) and (dx.iloc[i] < 0):
                purple.iloc[i] = True

    return blue.reindex(close.index, fill_value=False), purple.reindex(close.index, fill_value=False)

def annotate_hma55_npx_stars(ax,
                            close: pd.Series,
                            blue_mask: pd.Series,
                            purple_mask: pd.Series):
    c = _coerce_1d_series(close).astype(float)
    blue_mask = pd.Series(blue_mask).reindex(c.index).fillna(False).astype(bool)
    purple_mask = pd.Series(purple_mask).reindex(c.index).fillna(False).astype(bool)

    if blue_mask.any():
        t = c.index[blue_mask]
        ax.scatter(t, c.loc[t], marker="*", s=220, color="blue", zorder=13,
                   label="Blue ★ (HMA55↑ & NPX -0.5↑)")
    if purple_mask.any():
        t = c.index[purple_mask]
        ax.scatter(t, c.loc[t], marker="*", s=220, color="purple", zorder=13,
                   label="Purple ★ (HMA55↓ & NPX 0.5↓)")

# ---------------------------
# UPDATED (THIS REQUEST): Fibonacci 0%/100% reversal marker @ 99% confidence (R² ≥ 0.99)
# ---------------------------
def overlay_fib_extreme_reversal_marker(ax,
                                       price: pd.Series,
                                       fibs: dict,
                                       r2_val: float,
                                       r2_thr: float = 0.99,
                                       tol_frac: float = 0.001):
    if fibs is None or not isinstance(fibs, dict) or not fibs:
        return
    if not (np.isfinite(r2_val) and float(r2_val) >= float(r2_thr)):
        return

    p = _coerce_1d_series(price).astype(float).dropna()
    if p.empty:
        return

    hi = fibs.get("0%", np.nan)
    lo = fibs.get("100%", np.nan)
    if not (np.isfinite(hi) and np.isfinite(lo)):
        return

    tol_hi = (p - hi).abs() <= (abs(hi) * float(tol_frac))
    tol_lo = (p - lo).abs() <= (abs(lo) * float(tol_frac))
    hit = tol_hi | tol_lo
    if not hit.any():
        return

    t_last = hit[hit].index[-1]
    y_last = float(p.loc[t_last]) if np.isfinite(p.loc[t_last]) else np.nan
    if not np.isfinite(y_last):
        return

    ax.scatter([t_last], [y_last], marker="D", s=120, color="tab:orange", zorder=14,
               label="Fib 0/100 Reversal (R²≥0.99)")
    ax.text(t_last, y_last, "  Possible reversal (Fib 0/100, 99%)",
            fontsize=9, fontweight="bold", color="tab:orange",
            va="bottom", zorder=14)
# =========================
# Part 5/8 — bullbear.py
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

# ---------------------------
# UPDATED (THIS REQUEST): NPX markers = NPX 0.0-cross markers filtered by PRICE trend direction
#   - Price trend UP:   green circle (NPX 0↑) + green triangle (NPX 0↓)
#   - Price trend DOWN: red cross  (NPX 0↓) + red triangle  (NPX 0↑)
# ---------------------------
def overlay_npx_on_ntd(ax,
                       npx: pd.Series,
                       ntd: pd.Series,
                       mark_crosses: bool = True,
                       price_trend_slope: float = np.nan):
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx)
    ntd = ntd.reindex(idx)
    if npx.dropna().empty:
        return

    ax.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")

    if not mark_crosses:
        return

    uptrend = np.isfinite(price_trend_slope) and float(price_trend_slope) > 0.0
    downtrend = np.isfinite(price_trend_slope) and float(price_trend_slope) < 0.0

    cross_up0 = (npx >= 0.0) & (npx.shift(1) < 0.0)
    cross_dn0 = (npx <= 0.0) & (npx.shift(1) > 0.0)

    up_idx = list(cross_up0[cross_up0.fillna(False)].index)
    dn_idx = list(cross_dn0[cross_dn0.fillna(False)].index)

    if uptrend:
        if up_idx:
            ax.scatter(up_idx, [0.0]*len(up_idx), marker="o", s=55, color="tab:green", zorder=9,
                       label="NPX 0↑ (Price↑)")
        if dn_idx:
            ax.scatter(dn_idx, [0.0]*len(dn_idx), marker="^", s=75, color="tab:green", zorder=9,
                       label="NPX 0↓ (Price↑)")
    elif downtrend:
        if dn_idx:
            ax.scatter(dn_idx, [0.0]*len(dn_idx), marker="x", s=70, color="tab:red", zorder=9,
                       label="NPX 0↓ (Price↓)")
        if up_idx:
            ax.scatter(up_idx, [0.0]*len(up_idx), marker="v", s=75, color="tab:red", zorder=9,
                       label="NPX 0↑ (Price↓)")
    else:
        all_idx = sorted(set(up_idx + dn_idx))
        if all_idx:
            ax.scatter(all_idx, [0.0]*len(all_idx), marker=".", s=50, color="tab:gray", zorder=9,
                       label="NPX 0-cross (trend n/a)")

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
# Part 6/8 — bullbear.py
# =========================
# ---------------------------
# Sessions (PST)
# ---------------------------
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

# ---------------------------
# News (Yahoo Finance)
# ---------------------------
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
        except (ValueError, OverflowError, TypeError):
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

# ---------------------------
# Channel-in-range helpers for NTD panel
# ---------------------------
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

def rolling_midline(series_like: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll = s.rolling(window, min_periods=1)
    mid = (roll.max() + roll.min()) / 2.0
    return mid.reindex(s.index)

def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

# ---------------------------
# Cached last values for scanning
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_ntd_value(symbol: str, ntd_win: int):
    try:
        s = fetch_hist(symbol)
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None

@st.cache_data(ttl=120)
def last_hourly_ntd_value(symbol: str, ntd_win: int, period: str = "1d"):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df:
            return np.nan, None
        s = df["Close"].ffill()
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty:
            return np.nan, None
        return float(ntd.iloc[-1]), ntd.index[-1]
    except Exception:
        return np.nan, None
# =========================
# Part 7/8 — bullbear.py
# =========================
# ---------------------------
# Recent BUY scanner helpers
# ---------------------------
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
        try:
            bar = int(bar)
        except Exception:
            return None

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

# ---------------------------
# UPDATED (THIS REQUEST): Daily NPX 0.0-cross scanner (trend-filtered)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_npx_zero_cross_in_trend(symbol: str,
                                       ntd_win: int,
                                       daily_view_label: str,
                                       direction: str = "up"):
    """
    direction="up":
      - Price global trend slope (in selected Daily view) > 0
      - NPX crosses up through 0.0
    direction="down":
      - Price global trend slope < 0
      - NPX crosses down through 0.0
    """
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 5:
            return None

        x = np.arange(len(close_show), dtype=float)
        m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
        if not np.isfinite(m):
            return None

        want_up = str(direction).lower().startswith("u")
        if want_up and float(m) <= 0.0:
            return None
        if (not want_up) and float(m) >= 0.0:
            return None

        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        prev = npx_show.shift(1)
        if want_up:
            cross = (npx_show >= 0.0) & (prev < 0.0)
            label = "NPX 0↑ (Price↑)"
        else:
            cross = (npx_show <= 0.0) & (prev > 0.0)
            label = "NPX 0↓ (Price↓)"

        cross = cross.fillna(False)
        if not cross.any():
            return None

        t = cross[cross].index[-1]
        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        npx_at = float(npx_show.loc[t]) if (t in npx_show.index and np.isfinite(npx_show.loc[t])) else np.nan
        npx_last = float(npx_show.dropna().iloc[-1]) if len(npx_show.dropna()) else np.nan

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Daily View": daily_view_label,
            "Signal": label,
            "Bars Since": bars_since,
            "Cross Time": t,
            "Global Slope": float(m),
            "Current Price": curr_px,
            "NPX@Cross": npx_at,
            "NPX (last)": npx_last,
        }
    except Exception:
        return None

# ---------------------------
# UPDATED (THIS REQUEST): Blue/Purple star scanners
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_blue_purple_star(symbol: str,
                                daily_view_label: str,
                                ntd_win: int,
                                max_gap: int = 2):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < 10:
            return None

        hma55 = compute_hma(close_show, period=55)
        npx_full = compute_normalized_price(close_full, window=ntd_win)
        npx_show = _coerce_1d_series(npx_full).reindex(close_show.index)

        blue, purple = hma55_npx_star_masks(close_show, hma55, npx_show, max_bar_gap=max_gap)

        if (not blue.any()) and (not purple.any()):
            return None

        if blue.any():
            t_blue = blue[blue].index[-1]
        else:
            t_blue = None

        if purple.any():
            t_purple = purple[purple].index[-1]
        else:
            t_purple = None

        if t_blue is None:
            t = t_purple
            kind = "Purple"
        elif t_purple is None:
            t = t_blue
            kind = "Blue"
        else:
            t = t_blue if t_blue >= t_purple else t_purple
            kind = "Blue" if t == t_blue else "Purple"

        loc = int(close_show.index.get_loc(t))
        bars_since = int((len(close_show) - 1) - loc)

        return {
            "Symbol": symbol,
            "Frame": "Daily",
            "Star": kind,
            "Bars Since": bars_since,
            "Time": t,
            "Price": float(close_show.loc[t]) if np.isfinite(close_show.loc[t]) else np.nan,
        }
    except Exception:
        return None

@st.cache_data(ttl=120)
def last_hourly_blue_purple_star(symbol: str,
                                 period: str,
                                 ntd_win: int,
                                 max_gap: int = 2):
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        real_times = df.index if isinstance(df.index, pd.DatetimeIndex) else None

        df2 = df.copy()
        df2.index = pd.RangeIndex(len(df2))
        close = _coerce_1d_series(df2["Close"]).ffill().dropna()
        if close.empty or len(close) < 20:
            return None

        hma55 = compute_hma(close, period=55)
        npx = compute_normalized_price(close, window=ntd_win)

        blue, purple = hma55_npx_star_masks(close, hma55, npx, max_bar_gap=max_gap)
        if (not blue.any()) and (not purple.any()):
            return None

        t_blue = int(blue[blue].index[-1]) if blue.any() else None
        t_purple = int(purple[purple].index[-1]) if purple.any() else None

        if t_blue is None:
            bar = t_purple
            kind = "Purple"
        elif t_purple is None:
            bar = t_blue
            kind = "Blue"
        else:
            bar = t_blue if t_blue >= t_purple else t_purple
            kind = "Blue" if bar == t_blue else "Purple"

        bars_since = int((len(close) - 1) - bar)
        ts = real_times[bar] if isinstance(real_times, pd.DatetimeIndex) and (0 <= bar < len(real_times)) else None

        return {
            "Symbol": symbol,
            "Frame": f"Hourly ({period})",
            "Star": kind,
            "Bars Since": bars_since,
            "Time": ts,
            "Price": float(close.iloc[bar]) if np.isfinite(close.iloc[bar]) else np.nan,
        }
    except Exception:
        return None

# ---------------------------
# Daily Slope+BB Reversal Scanner (existing)
# ---------------------------
@st.cache_data(ttl=120)
def last_daily_sr_reversal_bbmid(symbol: str,
                                 daily_view_label: str,
                                 slope_lb: int,
                                 sr_lb: int,
                                 bb_window: int,
                                 bb_sigma: float,
                                 bb_ema: bool,
                                 prox: float,
                                 bars_confirm: int,
                                 horizon: int,
                                 side: str = "BUY",
                                 min_r2: float = 0.99):
    try:
        close_full = _coerce_1d_series(fetch_hist(symbol)).dropna()
        if close_full.empty:
            return None

        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(10, int(slope_lb), int(sr_lb)):
            return None

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb))
        if not (np.isfinite(m) and np.isfinite(r2)):
            return None
        if float(r2) < float(min_r2):
            return None

        want_buy = str(side).upper().startswith("B")
        if want_buy and float(m) <= 0.0:
            return None
        if (not want_buy) and float(m) >= 0.0:
            return None

        res = close_show.rolling(int(sr_lb), min_periods=1).max()
        sup = close_show.rolling(int(sr_lb), min_periods=1).min()

        bb_mid, bb_up, bb_lo, _, _ = compute_bbands(close_show, window=int(bb_window), mult=float(bb_sigma), use_ema=bool(bb_ema))
        if bb_mid.dropna().empty:
            return None

        cross_up, cross_dn = _cross_series(close_show, bb_mid)
        hz = max(1, int(horizon))

        if want_buy:
            if not cross_up.any():
                return None
            t_cross = cross_up[cross_up].index[-1]

            loc = int(close_show.index.get_loc(t_cross))
            j0 = max(0, loc - hz)
            touch_mask = close_show.iloc[j0:loc+1] <= (sup.iloc[j0:loc+1] * (1.0 + float(prox)))
            if not touch_mask.any():
                return None
            t_touch = touch_mask[touch_mask].index[-1]

            seg = close_show.loc[:t_cross]
            if not _n_consecutive_increasing(seg, int(bars_confirm)):
                return None

        else:
            if not cross_dn.any():
                return None
            t_cross = cross_dn[cross_dn].index[-1]

            loc = int(close_show.index.get_loc(t_cross))
            j0 = max(0, loc - hz)
            touch_mask = close_show.iloc[j0:loc+1] >= (res.iloc[j0:loc+1] * (1.0 - float(prox)))
            if not touch_mask.any():
                return None
            t_touch = touch_mask[touch_mask].index[-1]

            seg = close_show.loc[:t_cross]
            if not _n_consecutive_decreasing(seg, int(bars_confirm)):
                return None

        bars_since_cross = int((len(close_show) - 1) - int(close_show.index.get_loc(t_cross)))

        curr_px = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
        cross_px = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
        mid_px = float(bb_mid.loc[t_cross]) if (t_cross in bb_mid.index and np.isfinite(bb_mid.loc[t_cross])) else np.nan
        sup_px = float(sup.loc[t_touch]) if (t_touch in sup.index and np.isfinite(sup.loc[t_touch])) else np.nan
        res_px = float(res.loc[t_touch]) if (t_touch in res.index and np.isfinite(res.loc[t_touch])) else np.nan

        return {
            "Symbol": symbol,
            "Side": "BUY" if want_buy else "SELL",
            "Daily View": daily_view_label,
            "Bars Since Cross": bars_since_cross,
            "Touch Time": t_touch,
            "Cross Time": t_cross,
            "Slope": float(m),
            "R2": float(r2),
            "Support@Touch": sup_px,
            "Resistance@Touch": res_px,
            "BB Mid@Cross": mid_px,
            "Price@Cross": cross_px,
            "Current Price": curr_px,
        }
    except Exception:
        return None
# =========================
# Part 8/8 — bullbear.py
# =========================
# ---------------------------
# Session state init
# ---------------------------
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

# ---------------------------
# Shared hourly renderer (Stock & Forex)
# ---------------------------
def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool,
                        alert_placeholder=None):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill()
    he = hc.ewm(span=20).mean()

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    hma_h = compute_hma(hc, period=hma_period)
    hma55_h = compute_hma(hc, period=55)

    macd_h, macd_sig_h, macd_hist_h = compute_macd(hc)

    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday.columns else pd.Series(dtype=float)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {"High","Low","Close"}.issubset(intraday.columns) and show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(
            intraday["High"], intraday["Low"], intraday["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
            shift_cloud=False
        )
        kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    )

    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    if not psar_h_df.empty:
        psar_h_df = psar_h_df.reindex(hc.index)

    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    slope_sig_h = m_h

    rev_prob_h = slope_reversal_probability(
        hc,
        slope_sig_h,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_hourly,
        horizon=rev_horizon,
    )

    fx_news = pd.DataFrame()
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    ax2w = None
    if show_nrsi:
        fig2, (ax2, ax2w) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        plt.subplots_adjust(hspace=0.05, top=0.90, right=0.80, bottom=0.22)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.85, right=0.80, bottom=0.24)

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and (not psar_h_df.empty) and ("PSAR" in psar_h_df.columns):
        up_mask = psar_h_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

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

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, slope_sig_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

        trig_h = find_slope_trigger_after_band_reversal(hc, yhat_h, upper_h, lower_h, horizon=rev_horizon)
        annotate_slope_trigger(ax2, trig_h)

    # UPDATED: Blue/Purple stars on price chart (HMA55 + NPX thresholds)
    npx_h_for_star = compute_normalized_price(hc, window=ntd_window)
    blue_h, purple_h = hma55_npx_star_masks(hc, hma55_h, npx_h_for_star, max_bar_gap=2)
    annotate_hma55_npx_stars(ax2, hc, blue_h, purple_h)

    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax2, news_pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
    )

    is_alert = isinstance(instr_txt, str) and instr_txt.startswith("ALERT:")
    if alert_placeholder is not None:
        if is_alert:
            alert_placeholder.error(instr_txt)
        else:
            alert_placeholder.empty()

    title_instr = instr_txt
    if alert_placeholder is not None and is_alert:
        title_instr = ""

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    instr_part = f" — {title_instr} " if isinstance(title_instr, str) and title_instr.strip() else " "
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{instr_part}"
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_h}]"
    )

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        ax2.text(0.99, 0.02,
                 f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    # UPDATED: Fibonacci on Hourly (and reversal marker when R²≥0.99)
    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")
        overlay_fib_extreme_reversal_marker(ax2, hc, fibs_h, r2_val=r2_h, r2_thr=0.99, tol_frac=0.001)

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

    # UPDATED: Hourly NTD panel + NPX 0-cross markers filtered by price slope direction
    if ax2w is not None:
        ax2w.set_title(f"Hourly Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_hourly})")
        ntd_h = compute_normalized_trend(hc, window=ntd_window) if show_ntd else pd.Series(index=hc.index, dtype=float)
        npx_h = compute_normalized_price(hc, window=ntd_window) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

        if show_ntd and shade_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            shade_ntd_regions(ax2w, ntd_h)

        if show_ntd and not _coerce_1d_series(ntd_h).dropna().empty:
            ax2w.plot(ntd_h.index, ntd_h.values, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
            ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
            if not ntd_trend_h.empty:
                ax2w.plot(ntd_trend_h.index, ntd_trend_h.values, "--", linewidth=2,
                          label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")

            overlay_ntd_sr_reversal_stars(
                ax2w, price=hc, sup=sup_h, res=res_h,
                trend_slope=m_h, ntd=ntd_h, prox=sr_prox_pct,
                bars_confirm=rev_bars_confirm
            )

        if show_ntd_channel:
            overlay_inrange_on_ntd(ax2w, price=hc, sup=sup_h, res=res_h)

        if show_npx_ntd and not _coerce_1d_series(npx_h).dropna().empty:
            overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross, price_trend_slope=m_h)

        if show_hma_rev_ntd and not hma_h.dropna().empty and not hc.dropna().empty:
            overlay_hma_reversal_on_ntd(ax2w, hc, hma_h, lookback=hma_rev_lb,
                                        period=hma_period, ntd=ntd_h)

        ax2w.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
        ax2w.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
        ax2w.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")
        ax2w.set_ylim(-1.1, 1.1)
        ax2w.legend(loc="lower left", framealpha=0.5, fontsize=9)
        ax2w.set_xlabel("Time (PST)")
    else:
        ax2.set_xlabel("Time (PST)")

    # UPDATED (THIS REQUEST): legend outside chart area (price chart)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.5, fontsize=9)

    if session_handles and session_labels:
        fig2.legend(
            handles=session_handles,
            labels=session_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=True,
            fontsize=9,
            title="Sessions (PST)",
            title_fontsize=9
        )

    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax2w if ax2w is not None else ax2, real_times, n_ticks=8)

    style_axes(ax2)
    if ax2w is not None:
        style_axes(ax2w)
    xlim_price = ax2.get_xlim()
    st.pyplot(fig2)

    if show_macd and not macd_h.dropna().empty:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        axm.set_title("MACD (optional)")
        axm.plot(macd_h.index, macd_h.values, linewidth=1.4, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, linewidth=1.2, label="Signal")
        axm.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        axm.set_xlim(xlim_price)
        axm.legend(loc="lower left", framealpha=0.5, fontsize=9)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        style_axes(axm)
        st.pyplot(figm)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "NPX 0.0-Cross Scanner",
    "NPX 0.5-Cross Scanner",
    "Daily Slope+BB Reversal Scanner",
    "Blue/Purple Star Scanner"
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
    alert_box = st.empty()

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
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"**Displayed (last run):** {disp_ticker}  •  "
                   f"Selection now: {sel}{' (run to switch)' if sel != disp_ticker else ''}")

        if chart in ("Daily", "Both"):
            ema30 = df.ewm(span=30).mean()
            res_d = df.rolling(sr_lb_daily, min_periods=1).max()
            sup_d = df.rolling(sr_lb_daily, min_periods=1).min()

            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)
            rev_prob_d = slope_reversal_probability(df, m_d, hist_window=rev_hist_lb, slope_window=slope_lb_daily, horizon=rev_horizon)
            piv = current_daily_pivots(df_ohlc)

            ntd_d = compute_normalized_trend(df, window=ntd_window) if show_ntd else pd.Series(index=df.index, dtype=float)
            npx_d_full = compute_normalized_price(df, window=ntd_window) if show_npx_ntd else pd.Series(index=df.index, dtype=float)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    shift_cloud=False
                )
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res_d_show = res_d.reindex(df_show.index)
            sup_d_show = sup_d.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            ntd_d_show = ntd_d.reindex(df_show.index)
            npx_d_show = npx_d_full.reindex(df_show.index)
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show = bb_up_d.reindex(df_show.index)
            bb_lo_d_show = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_show = compute_hma(df, period=hma_period).reindex(df_show.index)
            hma55_d_show = compute_hma(df_show, period=55).reindex(df_show.index)

            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.80)

            rev_txt_d = fmt_pct(rev_prob_d) if np.isfinite(rev_prob_d) else "n/a"
            ax.set_title(
                f"{disp_ticker} Daily — {daily_view} — History, EMA, S/R (w={sr_lb_daily}), Slope, Pivots "
                f"[P(slope rev≤{rev_horizon} bars)={rev_txt_d}]"
            )

            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")

            global_m_d = draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)

            if show_psar and (not psar_d_df.empty) and ("PSAR" in psar_d_df.columns):
                up_mask = psar_d_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                               label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

            res_val_d = sup_val_d = np.nan
            try:
                res_val_d = float(res_d_show.iloc[-1])
                sup_val_d = float(sup_d_show.iloc[-1])
            except Exception:
                pass
            if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
                ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:red", linestyles="-", linewidth=1.6,
                          label=f"Resistance (w={sr_lb_daily})")
                ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1], colors="tab:green", linestyles="-", linewidth=1.6,
                          label=f"Support (w={sr_lb_daily})")
                label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
                label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")

            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
                bounce_sig_d = find_band_bounce_signal(df_show, upper_d_show, lower_d_show, m_d)
                if bounce_sig_d is not None:
                    annotate_crossover(ax, bounce_sig_d["time"], bounce_sig_d["price"], bounce_sig_d["side"])

                trig_d = find_slope_trigger_after_band_reversal(df_show, yhat_d_show, upper_d_show, lower_d_show, horizon=rev_horizon)
                annotate_slope_trigger(ax, trig_d)

            # UPDATED: Blue/Purple stars on daily price chart
            npx_d_for_star = compute_normalized_price(df_show, window=ntd_window)
            blue_d, purple_d = hma55_npx_star_masks(df_show, hma55_d_show, npx_d_for_star, max_bar_gap=2)
            annotate_hma55_npx_stars(ax, df_show, blue_d, purple_d)

            # UPDATED: Daily Fibonacci (default ON) + reversal marker at Fib 0/100 when R²≥0.99
            if show_fibs and not df_show.empty:
                fibs_d = fibonacci_levels(df_show)
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="dotted", linewidth=1)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")
                overlay_fib_extreme_reversal_marker(ax, df_show, fibs_d, r2_val=r2_d, r2_thr=0.99, tol_frac=0.001)

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            ax.set_ylabel("Price")

            # UPDATED: legend outside chart area (price chart)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.5, fontsize=9)

            axdw.set_title(f"Daily Indicator Panel — NTD + NPX + Trend (S/R w={sr_lb_daily})")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                              label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")
                overlay_ntd_sr_reversal_stars(axdw, price=df_show, sup=sup_d_show, res=res_d_show,
                                              trend_slope=m_d, ntd=ntd_d_show, prox=sr_prox_pct,
                                              bars_confirm=rev_bars_confirm)

            if show_npx_ntd and not npx_d_show.dropna().empty:
                overlay_npx_on_ntd(axdw, npx_d_show, ntd_d_show, mark_crosses=mark_npx_cross, price_trend_slope=m_d)

            if show_hma_rev_ntd and not hma_d_show.dropna().empty and not df_show.dropna().empty:
                overlay_hma_reversal_on_ntd(axdw, df_show, hma_d_show, lookback=hma_rev_lb,
                                            period=hma_period, ntd=ntd_d_show)

            axdw.axhline(0.0, linestyle="--", linewidth=1.0, color="black", label="0.00")
            axdw.axhline(0.5, linestyle="-", linewidth=1.2, color="red", label="+0.50")
            axdw.axhline(-0.5, linestyle="-", linewidth=1.2, color="red", label="-0.50")

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")
            axdw.legend(loc="lower left", framealpha=0.5, fontsize=9)

            style_axes(ax)
            style_axes(axdw)
            st.pyplot(fig)

        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            render_hourly_views(
                sel=disp_ticker,
                intraday=intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.hour_range,
                is_forex=(mode == "Forex"),
                alert_placeholder=alert_box
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Click **Run Forecast** to display charts and forecast.")

# ---------------------------
# TAB 2: ENHANCED FORECAST
# ---------------------------
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.get("run_all", False) or st.session_state.get("ticker") is None or st.session_state.get("mode_at_run") != mode:
        st.info("Run Tab 1 first (in the current mode).")
    else:
        df = st.session_state.df_hist
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"Displayed ticker: **{st.session_state.ticker}**  •  Intraday lookback: **{st.session_state.get('hour_range','24h')}**")
        view = st.radio("View:", ["Daily", "Intraday", "Both"], key=f"enh_view_{mode}")

        if view in ("Daily", "Both"):
            df_show = subset_by_daily_view(df, daily_view)
            res_d_show = df_show.rolling(sr_lb_daily, min_periods=1).max()
            sup_d_show = df_show.rolling(sr_lb_daily, min_periods=1).min()

            hma55 = compute_hma(df_show, period=55)
            npx = compute_normalized_price(df_show, window=ntd_window)
            blue, purple = hma55_npx_star_masks(df_show, hma55, npx, max_bar_gap=2)

            fig, ax = plt.subplots(figsize=(14, 5))
            plt.subplots_adjust(right=0.80)
            ax.set_title(f"{st.session_state.ticker} Daily (Enhanced) — {daily_view}")
            ax.plot(df_show.index, df_show.values, label="History")
            draw_trend_direction_line(ax, df_show, label_prefix="Trend (global)")
            annotate_hma55_npx_stars(ax, df_show, blue, purple)

            if show_fibs and not df_show.empty:
                fibs = fibonacci_levels(df_show)
                for lbl, y in fibs.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1], linestyles="dotted", linewidth=1)
                for lbl, y in fibs.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")
                _, _, _, _, r2 = regression_with_band(df_show, lookback=int(slope_lb_daily))
                overlay_fib_extreme_reversal_marker(ax, df_show, fibs, r2_val=r2, r2_thr=0.99, tol_frac=0.001)

            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.5, fontsize=9)
            style_axes(ax)
            st.pyplot(fig)

        if view in ("Intraday", "Both"):
            render_hourly_views(
                sel=st.session_state.ticker,
                intraday=st.session_state.intraday,
                p_up=p_up,
                p_dn=p_dn,
                hour_range_label=st.session_state.get("hour_range","24h"),
                is_forex=(mode == "Forex")
            )

        st.subheader("SARIMAX Forecast (30d)")
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:, 0],
            "Upper":    ci.iloc[:, 1]
        }, index=idx))

# ---------------------------
# TAB 3: BULL vs BEAR
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
        ax.set_title(f"{sel_bb} — {bb_period} Close")
        ax.plot(s.index, s.values, label="Close")
        draw_trend_direction_line(ax, s, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 4: METRICS
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
        st.write({
            "Daily slope (reg band)": fmt_slope(m_d),
            "Daily R²": fmt_r2(r2_d),
            f"P(slope reverses ≤ {rev_horizon} bars)": fmt_pct(slope_reversal_probability(df, m_d, rev_hist_lb, slope_lb_daily, rev_horizon))
        })

        if intr is not None and not intr.empty and "Close" in intr:
            intr_plot = intr.copy()
            intr_plot.index = pd.RangeIndex(len(intr_plot))
            hc = intr_plot["Close"].ffill()
            yhat_h, up_h, lo_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
            st.write({
                "Hourly slope (reg band)": fmt_slope(m_h),
                "Hourly R²": fmt_r2(r2_h),
                f"P(slope reverses ≤ {rev_horizon} bars) hourly": fmt_pct(slope_reversal_probability(hc, m_h, rev_hist_lb, slope_lb_hourly, rev_horizon))
            })

# ---------------------------
# TAB 5: NTD -0.75 Scanner
# ---------------------------
with tab5:
    st.header("NTD -0.75 Scanner")
    st.caption("Lists symbols where the latest NTD is below -0.75 (using latest intraday for hourly; daily uses daily close).")

    scan_frame = st.radio("Frame:", ["Hourly (intraday)", "Daily"], index=0, key=f"ntd_scan_frame_{mode}")
    run_scan = st.button("Run Scanner", key=f"btn_run_ntd_scan_{mode}")

    if run_scan:
        rows = []
        if scan_frame.startswith("Hourly"):
            period = "1d"
            for sym in universe:
                val, ts = last_hourly_ntd_value(sym, ntd_window, period=period)
                if np.isfinite(val) and val < -0.75:
                    rows.append({"Symbol": sym, "NTD": float(val), "Time": ts})
        else:
            for sym in universe:
                val, ts = last_daily_ntd_value(sym, ntd_window)
                if np.isfinite(val) and val < -0.75:
                    rows.append({"Symbol": sym, "NTD": float(val), "Time": ts})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values("NTD")
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 6: LONG-TERM HISTORY
# ---------------------------
with tab6:
    st.header("Long-Term History")
    sel_lt = st.selectbox("Ticker:", universe, key=f"lt_ticker_{mode}")
    try:
        smax = fetch_hist_max(sel_lt)
    except Exception:
        smax = pd.Series(dtype=float)

    if smax is None or smax.dropna().empty:
        st.warning("No long-term history available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.set_title(f"{sel_lt} — Max History")
        ax.plot(smax.index, smax.values, label="Close")
        draw_trend_direction_line(ax, smax, label_prefix="Trend (global)")
        ax.legend(loc="lower left", framealpha=0.5, fontsize=9)
        style_axes(ax)
        st.pyplot(fig)

# ---------------------------
# TAB 7: NPX 0.0-CROSS SCANNER
# ---------------------------
with tab7:
    st.header("NPX 0.0-Cross Scanner (Daily, trend-filtered)")
    st.caption(
        "Lists symbols where **NPX crosses 0.0** on the NTD panel, filtered by the **price chart trend**:\n"
        "• UP list: Price trend UP + NPX 0↑\n"
        "• DOWN list: Price trend DOWN + NPX 0↓"
    )

    max_bars = st.slider("Max bars since NPX 0-cross", 0, 30, 2, 1, key="npx0_scan_max_bars")
    run_scan0 = st.button("Run NPX 0.0-Cross Scan", key="btn_run_npx0_scan")

    if run_scan0:
        up_rows, dn_rows = [], []
        for sym in universe:
            r_up = last_daily_npx_zero_cross_in_trend(sym, ntd_win=ntd_window, daily_view_label=daily_view, direction="up")
            if r_up is not None and int(r_up.get("Bars Since", 9999)) <= int(max_bars):
                up_rows.append(r_up)

            r_dn = last_daily_npx_zero_cross_in_trend(sym, ntd_win=ntd_window, daily_view_label=daily_view, direction="down")
            if r_dn is not None and int(r_dn.get("Bars Since", 9999)) <= int(max_bars):
                dn_rows.append(r_dn)

        left, right = st.columns(2)
        with left:
            st.subheader("UP: Price↑ + NPX 0↑ (green circle)")
            if not up_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(up_rows).sort_values(["Bars Since", "Global Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("DOWN: Price↓ + NPX 0↓ (red cross)")
            if
