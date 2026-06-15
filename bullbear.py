# bullbear.py — Stocks/Forex Dashboard + Forecasts
# (UPDATED) London & New York session Open/Close markers in PST on Forex intraday charts.
# (UPDATED) Removed MACD from NTD panels; NTD panels now use a smoothed NPX price overlay.
# (UPDATED) NTD panels are less noisy: green/red triangles now appear only after confirmed S/R reversals.
# (NEW) BB Divergence Signals (price trend vs. Bollinger band drift) with confidence gate
# (NEW) ADX filter (period/threshold) + confluence gating for HMA, BB Divergence, and Near S/R signals
# (UPDATED) Removed hourly Momentum chart, red/green directional PSAR price overlays, and NTD-cross triangles.
# (UPDATED) Removed Ichimoku Kijun and Supertrend lines from price charts.
# (UPDATED) Fibonacci lines are shown on price charts by default.
# (UPDATED) BUY near-support green ribbon now starts with Profit Alert.
# (UPDATED) NTD-panel green/red triangles now appear only for NTD buy/sell opportunities.
# (NEW) S/R Reversal Crosses tab separates Daily/Hourly and Upward/Downward zero-line crosses.
# (NEW) -0.75 SR Crossers tab finds daily S/R Reversal line support-zone upward reversals and current below-threshold symbols grouped by trend.
# (NEW) Green Triangle Pick tab scans daily/hourly NTD support-reversal BUY triangles aligned with upward trend.
# (NEW) S/R Cross tab scans daily S/R Reversal Index below -0.75 and recent upward 0.0 crosses.
# (UPDATED) S/R Reversal Crosses tab includes a daily -0.75 upward-cross table after support-zone reversal.
# (UPDATED) Trade State gating: buy/profit alerts require recent NTD S/R support-reversal turn from the lower zone.
# (NEW) S/R -0.5 Cross tab scans daily upward crosses through -0.5 and 0.0 on the S/R Reversal Index.
# (UPDATED) S/R Cross tab adds an Actionable S/R Long Picks ranking table with trade status, setup quality, support/resistance, and reward/risk.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory  # for left-side labels

# --- Page config ---
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal CSS ---
st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 { transform: none !important; visibility: visible !important; width: 100% !important; position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)


# --- Wrapped tabs CSS for better UX when many tabs are present ---
st.markdown("""
<style>
  div[data-testid="stTabs"] div[role="tablist"] {
    flex-wrap: wrap;
    gap: 0.15rem 0.35rem;
  }
  div[data-testid="stTabs"] button[role="tab"] {
    flex: 0 0 auto;
  }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if 'last_refresh' not in st.session_state:
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

# ---------- Helpers ----------
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

def trend_color_for_slope(slope: float) -> str:
    """Green for upward/flat trendlines, red for downward trendlines."""
    try:
        sv = float(np.squeeze(slope))
    except Exception:
        return "tab:gray"
    if not np.isfinite(sv):
        return "tab:gray"
    return "tab:green" if sv >= 0 else "tab:red"

# === NEW: FX pip helpers + ordered instruction text ===
def pip_size_for_symbol(symbol: str):
    try:
        s = str(symbol).upper()
    except Exception:
        return None
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
    if ps is not None and ps > 0:
        return f"{diff/ps:.1f} pips"
    return f"Δ {diff:.3f}"

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str) -> str:
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False
    entry_buy  = float(buy_val)  if _finite(buy_val)  else float(close_val)
    exit_sell  = float(sell_val) if _finite(sell_val) else float(close_val)
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

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    try:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.01, y_val, text,
            transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=6
        )
    except Exception:
        pass

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

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")

daily_view = st.sidebar.selectbox(
    "Daily view range:",
    ["Historical", "6M", "12M", "24M"],
    index=2,
    key="sb_daily_view"
)

show_fibs = st.sidebar.checkbox("Show Fibonacci lines on price charts", value=True, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

# NEW: Hourly S/R window
st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

# Hourly Indicator Panel
st.sidebar.subheader("Hourly Indicator Panel")
show_nrsi   = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

# NTD Channel on Indicator Panel
st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox("Highlight when price is between S/R (S↔R) on NTD", value=True, key="sb_ntd_channel")

# Hourly Supertrend
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# Parabolic SAR price overlay removed to reduce chart noise.
show_psar = False
psar_step = 0.02
psar_max = 0.20

# Signal logic
st.sidebar.subheader("Signal Logic (Hourly)")
signal_threshold = st.sidebar.slider("Signal confidence threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# Daily indicator panel controls
st.sidebar.subheader("NTD (Daily Indicator Panel)")
pivot_lookback_d = st.sidebar.slider("Pivot lookback (unused)", 3, 31, 9, 2, key="sb_pivot_lb_d")
norm_window_d    = st.sidebar.slider("Normalization window (unused)", 30, 1200, 360, 10, key="sb_norm_win_d")
waves_to_annotate_d = st.sidebar.slider("Annotate recent waves (unused)", 3, 12, 7, 1, key="sb_wave_ann_d")

# NPO overlay (unused)
st.sidebar.subheader("Normalized Price Oscillator (unused on indicator panels)")
show_npo    = st.sidebar.checkbox("Show NPO overlay (unused)", value=False, key="sb_show_npo")
npo_fast    = st.sidebar.slider("NPO fast EMA", 5, 30, 12, 1, key="sb_npo_fast")
npo_slow    = st.sidebar.slider("NPO slow EMA", 10, 60, 26, 1, key="sb_npo_slow")
npo_norm_win= st.sidebar.slider("NPO normalization window", 30, 600, 240, 10, key="sb_npo_norm")

# NTD overlays
st.sidebar.subheader("Normalized Trend (NTD panels — Daily & Hourly)")
show_ntd  = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window= st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (Daily & Hourly: green=up, red=down)", value=True, key="sb_ntd_shade")

# Smoothed NPX overlay for NTD panels
st.sidebar.subheader("Smoothed NPX Overlay (NTD Panels)")
show_npx_ntd   = st.sidebar.checkbox("Show smoothed NPX overlay on NTD panels", value=True, key="sb_show_npx_ntd")
npx_smooth_span = st.sidebar.slider("NPX smoothing EMA span", 2, 60, 9, 1, key="sb_npx_smooth_span")
mark_npx_cross = st.sidebar.checkbox("Legacy NPX triangle markers (disabled; S/R reversals control triangles)", value=False, key="sb_mark_npx_cross")

# S/R Reversal Indicator on NTD panels
st.sidebar.subheader("S/R Reversal Indicator (NTD)")
show_sr_reversal_ntd = st.sidebar.checkbox(
    "Show support/resistance reversal indicator",
    value=True,
    key="sb_sr_rev_show"
)
sr_rev_zone = st.sidebar.slider(
    "S/R reversal zone strength",
    0.40, 0.95, 0.80, 0.05,
    key="sb_sr_rev_zone"
)
sr_rev_lookback = st.sidebar.slider(
    "S/R reversal touch lookback (bars)",
    2, 30, 8, 1,
    key="sb_sr_rev_lookback"
)
sr_rev_confirm = st.sidebar.slider(
    "S/R reversal confirmation bars",
    1, 6, 3, 1,
    key="sb_sr_rev_confirm"
)
sr_rev_smooth = st.sidebar.slider(
    "S/R reversal index smoothing",
    1, 30, 8, 1,
    key="sb_sr_rev_smooth"
)

# Ichimoku / Kijun
st.sidebar.subheader("Normalized Ichimoku (EW panels) + Kijun on price")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")
ichi_norm_win = st.sidebar.slider("Ichimoku normalization window (unused)", 30, 600, 240, 10, key="sb_ichi_norm")
ichi_price_weight = st.sidebar.slider("Weight: Price vs Cloud (unused)", 0.0, 1.0, 0.6, 0.05, key="sb_ichi_w")

# Bollinger
st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

# HMA
st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

# HMA reversal markers on NTD
st.sidebar.subheader("HMA(55) Reversal on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb       = st.sidebar.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

# BB Divergence
st.sidebar.subheader("BB Divergence Signals")
show_bb_div = st.sidebar.checkbox("Show BB divergence signals", value=True, key="sb_bbdiv_show")
bb_conf     = st.sidebar.slider("BB divergence confidence", 0.50, 0.99, 0.95, 0.01, key="sb_bbdiv_conf")

# 🔶 NEW: ADX filter controls
st.sidebar.subheader("Trend Strength Filter (ADX)")
use_adx_filter = st.sidebar.checkbox("Require ADX ≥ threshold for signals", value=True, key="sb_adx_use")
adx_period     = st.sidebar.slider("ADX period", 5, 50, 14, 1, key="sb_adx_period")
adx_min        = st.sidebar.slider("Min ADX to allow signals", 5, 50, 20, 1, key="sb_adx_min")

# Forex-only
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
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','AUDJPY=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','EURCAD=X'
    ]

# --- Cache helpers (TTL = 120 seconds) ---
def _flatten_yf_columns(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """
    yfinance can return either flat columns (Close, High, ...)
    or MultiIndex columns (Price/Ticker). This normalizes the result
    so the rest of the app always sees ordinary OHLCV column names.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if ticker is not None:
            try:
                lvl_values = [str(x).upper() for x in out.columns.get_level_values(-1)]
                if str(ticker).upper() in lvl_values:
                    out = out.xs(ticker, axis=1, level=-1, drop_level=True)
            except Exception:
                pass
        if isinstance(out.columns, pd.MultiIndex):
            # Prefer the level containing OHLCV names, then drop any ticker level.
            for level in range(out.columns.nlevels):
                vals = set(map(str, out.columns.get_level_values(level)))
                if any(c in vals for c in wanted):
                    try:
                        if out.columns.nlevels == 2:
                            other_level = 1 - level
                            first_other = out.columns.get_level_values(other_level)[0]
                            out = out.xs(first_other, axis=1, level=other_level, drop_level=True)
                        else:
                            out.columns = out.columns.get_level_values(level)
                        break
                    except Exception:
                        out.columns = ["_".join(map(str, c)).strip("_") for c in out.columns.to_flat_index()]
                        break
    out.columns = [str(c) for c in out.columns]
    return out

def _ensure_pacific_index(obj):
    if obj is None or obj.empty:
        return obj
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[out.index.notna()]
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize(PACIFIC)
        else:
            out.index = out.index.tz_convert(PACIFIC)
    except TypeError:
        out.index = out.index.tz_convert(PACIFIC)
    return out

def _close_series_from_yf(df: pd.DataFrame, ticker: str = None) -> pd.Series:
    df = _flatten_yf_columns(df, ticker=ticker)
    if df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        s = df["Close"]
    elif "Adj Close" in df.columns:
        s = df["Adj Close"]
    else:
        s = _coerce_1d_series(df)
    s = _coerce_1d_series(s).dropna()
    s = _ensure_pacific_index(s)
    return s.sort_index()

def _ohlc_from_yf(df: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    df = _flatten_yf_columns(df, ticker=ticker)
    needed = ["Open", "High", "Low", "Close"]
    if df.empty:
        return pd.DataFrame(columns=needed)
    missing = [c for c in needed if c not in df.columns]
    if missing and "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
        missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=needed)
    out = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    out = _ensure_pacific_index(out)
    return out.sort_index()

@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"),
                     progress=False, auto_adjust=False)
    s = _close_series_from_yf(df, ticker=ticker)
    if s.empty:
        return s
    return s.asfreq("D").ffill().dropna()

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
    s = _close_series_from_yf(df, ticker=ticker)
    if s.empty:
        return s
    return s.asfreq("D").ffill().dropna()

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"),
                     progress=False, auto_adjust=False)
    return _ohlc_from_yf(df, ticker=ticker)

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m",
                     progress=False, auto_adjust=False)
    out = _flatten_yf_columns(df, ticker=ticker)
    if out.empty:
        return pd.DataFrame()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[out.index.notna()]
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
    except TypeError:
        pass
    return out.tz_convert(PACIFIC).sort_index()

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
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---- Indicators ----
def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0%": hi, "23.6%": hi - 0.236 * diff, "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5 * diff, "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff, "100%": lo,
    }

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {'High','Low','Close'}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    try:
        H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    except Exception:
        return {}
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

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

def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

# ---- RSI / Normalized RSI ----
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
    nrsi = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    return nrsi.reindex(rsi.index)

# ---- Normalized Volume (z-score → tanh) ----
def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win//10)
    m = v.rolling(norm_win, min_periods=minp).mean()
    sd = v.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (v - m) / sd
    return np.tanh(z / 3.0)

# ---- Normalized Price Oscillator (EW, unused) ----
def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26, norm_win: int = 240) -> pd.Series:
    s = _coerce_1d_series(close)
    if s.empty or not np.isfinite(fast) or not np.isfinite(slow) or fast <= 0 or slow <= 0:
        return pd.Series(index=s.index, dtype=float)
    if fast >= slow:
        fast, slow = max(1, slow - 1), slow
        if fast >= slow:
            return pd.Series(index=s.index, dtype=float)
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean().replace(0, np.nan)
    ppo = (ema_fast - ema_slow) / ema_slow * 100.0
    minp = max(10, int(norm_win)//10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std  = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
    z = (ppo - mean) / std
    npo = np.tanh(z / 2.0)
    return npo.reindex(s.index)

# ---- Normalized Trend Direction (NTD) ----
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
    ntd = np.tanh(ntd_raw / 2.0)
    return ntd.reindex(s.index)

# ---- NEW: Normalized Price (NPX) for NTD panel ----
def compute_normalized_price(close: pd.Series, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window // 3)
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    z = (s - m) / sd
    npx = np.tanh(z / 2.0)
    return npx.reindex(s.index)


def smooth_npx(npx: pd.Series, span: int = 9) -> pd.Series:
    """Smooth NPX for cleaner intraday reading on the NTD panel."""
    s = _coerce_1d_series(npx).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    span = int(max(1, span))
    return s.ewm(span=span, adjust=False, min_periods=1).mean().clip(-1.0, 1.0).reindex(s.index)

def shade_ntd_regions(ax, ntd: pd.Series):
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

def shade_npo_regions(ax, npo: pd.Series):
    return

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = trend_color_for_slope(m)
    ax.plot(s.index, yhat, "-", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# ---- Supertrend (hourly overlay) ----
def _true_range(df: pd.DataFrame):
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = _true_range(df[['High','Low','Close']])
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["ST","in_uptrend","upperband","lowerband"])
    ohlc = df[['High','Low','Close']].copy()
    hl2 = (ohlc['High'] + ohlc['Low']) / 2.0
    atr = compute_atr(ohlc, atr_period)
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr
    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_up   = pd.Series(index=ohlc.index, dtype=bool)
    st_line.iloc[0] = upperband.iloc[0]
    in_up.iloc[0]   = True
    for i in range(1, len(ohlc)):
        prev_st = st_line.iloc[i-1]
        prev_up = in_up.iloc[i-1]
        up_i = min(upperband.iloc[i], prev_st) if prev_up else upperband.iloc[i]
        dn_i = max(lowerband.iloc[i], prev_st) if not prev_up else lowerband.iloc[i]
        close_i = ohlc['Close'].iloc[i]
        if close_i > up_i:
            curr_up = True
        elif close_i < dn_i:
            curr_up = False
        else:
            curr_up = prev_up
        in_up.iloc[i]   = curr_up
        st_line.iloc[i] = dn_i if curr_up else up_i
    return pd.DataFrame({
        "ST": st_line, "in_uptrend": in_up,
        "upperband": upperband, "lowerband": lowerband
    })

# --- Parabolic SAR ---
def compute_parabolic_sar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2):
    H = _coerce_1d_series(high).astype(float)
    L = _coerce_1d_series(low).astype(float)
    df = pd.concat([H.rename("H"), L.rename("L")], axis=1).dropna()
    if df.empty:
        idx = H.index if len(H) else L.index
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=bool)
    n = len(df)
    psar = np.zeros(n) * np.nan
    up = np.zeros(n, dtype=bool)
    uptrend = True
    af = float(step)
    ep = df["H"].iloc[0]
    psar[0] = df["L"].iloc[0]
    up[0] = True
    for i in range(1, n):
        prev_psar = psar[i-1]
        if uptrend:
            psar[i] = prev_psar + af * (ep - prev_psar)
            lo1 = df["L"].iloc[i-1]
            lo2 = df["L"].iloc[i-2] if i >= 2 else lo1
            psar[i] = min(psar[i], lo1, lo2)
            if df["H"].iloc[i] > ep:
                ep = df["H"].iloc[i]; af = min(af + step, max_step)
            if df["L"].iloc[i] < psar[i]:
                uptrend = False; psar[i] = ep; ep = df["L"].iloc[i]; af = step
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            hi1 = df["H"].iloc[i-1]
            hi2 = df["H"].iloc[i-2] if i >= 2 else hi1
            psar[i] = max(psar[i], hi1, hi2)
            if df["L"].iloc[i] < ep:
                ep = df["L"].iloc[i]; af = min(af + step, max_step)
            if df["H"].iloc[i] > psar[i]:
                uptrend = True; psar[i] = ep; ep = df["H"].iloc[i]; af = step
        up[i] = uptrend
    psar_s = pd.Series(psar, index=df.index, name="PSAR")
    up_s = pd.Series(up, index=df.index, name="in_uptrend")
    return psar_s, up_s

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["PSAR","in_uptrend"])
    ps, up = compute_parabolic_sar(df["High"], df["Low"], step=step, max_step=max_step)
    return pd.DataFrame({"PSAR": ps, "in_uptrend": up})

# ---- Ichimoku (classic) ----
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = False):
    H = _coerce_1d_series(high)
    L = _coerce_1d_series(low)
    C = _coerce_1d_series(close)
    if H.empty or L.empty or C.empty:
        idx = C.index if not C.empty else (H.index if not H.empty else L.index)
        return (pd.Series(index=idx, dtype=float),)*5
    tenkan = (H.rolling(conv).max() + L.rolling(conv).min()) / 2.0
    kijun  = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (H.rolling(span_b).max() + L.rolling(span_b).min()) / 2.0
    span_a = span_a_raw.shift(base) if shift_cloud else span_a_raw
    span_b = span_b_raw.shift(base) if shift_cloud else span_b_raw
    chikou = C.shift(-base)
    return tenkan, kijun, span_a, span_b, chikou

def compute_normalized_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
                                conv: int = 9, base: int = 26, span_b: int = 52,
                                norm_win: int = 240, price_weight: float = 0.6) -> pd.Series:
    C = _coerce_1d_series(close).astype(float)
    return pd.Series(index=C.index, dtype=float)

# ---- Bollinger + %B + NBB ----
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
    return mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index)

# ---- HMA ----
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

def detect_last_crossover(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    mask = p.notna() & l.notna()
    if mask.sum() < 2:
        return None
    p = p[mask]; l = l[mask]
    above = p > l
    cross_up  = above & (~above.shift(1).fillna(False))
    cross_dn  = (~above) & (above.shift(1).fillna(False))
    t_up = cross_up[cross_up].index[-1] if cross_up.any() else None
    t_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None
    if t_up is None and t_dn is None:
        return None
    if t_dn is None or (t_up is not None and t_up > t_dn):
        return {"time": t_up, "side": "BUY"}
    else:
        return {"time": t_dn, "side": "SELL"}

def annotate_crossover(ax, ts, px, side: str, conf: float):
    try:
        if side == "BUY":
            ax.scatter([ts], [px], marker="^", s=90, color="tab:green", zorder=7)
            ax.text(ts, px, f"  BUY {int(conf*100)}%", va="bottom", fontsize=9, color="tab:green", fontweight="bold")
        else:
            ax.scatter([ts], [px], marker="v", s=90, color="tab:red", zorder=7)
            ax.text(ts, px, f"  SELL {int(conf*100)}%", va="top", fontsize=9, color="tab:red", fontweight="bold")
    except Exception:
        pass

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

def detect_hma_reversal_masks(price: pd.Series, hma: pd.Series, lookback: int = 3):
    h = _coerce_1d_series(hma)
    slope = h.diff().rolling(lookback, min_periods=1).mean()
    sign_now = np.sign(slope)
    sign_prev = np.sign(slope.shift(1))
    cross_up, cross_dn = _cross_series(price, hma)
    buy_rev = cross_up & (sign_now > 0) & (sign_prev < 0)
    sell_rev = cross_dn & (sign_now < 0) & (sign_prev > 0)
    buy_rev = buy_rev.fillna(False)
    sell_rev = sell_rev.fillna(False)
    return buy_rev, sell_rev

def overlay_hma_reversal_on_ntd(ax, price: pd.Series, hma: pd.Series, lookback: int = 3,
                                y_up: float = 0.95, y_dn: float = -0.95,
                                label_prefix: str = "HMA REV", period: int = 55):
    """
    HMA reversal markers are intentionally NOT triangles so the only green/red
    triangles on the NTD panel are the actual NTD buy/sell opportunity markers.
    """
    try:
        buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
        idx_up = list(buy_rev[buy_rev].index)
        idx_dn = list(sell_rev[sell_rev].index)
        if len(idx_up):
            ax.scatter(idx_up, [y_up]*len(idx_up), marker="s", s=60, color="tab:green",
                       zorder=8, label=f"HMA({period}) ↑ REV")
        if len(idx_dn):
            ax.scatter(idx_dn, [y_dn]*len(idx_dn), marker="D", s=60, color="tab:red",
                       zorder=8, label=f"HMA({period}) ↓ REV")
    except Exception:
        pass

def _apply_signal_cooldown(mask: pd.Series, cooldown_bars: int = 12) -> pd.Series:
    """
    Keep only the first signal in each cooldown window. This prevents a single
    reversal from printing repeated triangles on several follow-through bars.
    """
    s = pd.Series(mask, copy=True).fillna(False).astype(bool)
    if s.empty:
        return s
    try:
        cd = max(0, int(cooldown_bars))
    except Exception:
        cd = 12
    if cd <= 0:
        return s

    out = pd.Series(False, index=s.index)
    last_i = -10**9
    for i, val in enumerate(s.to_numpy(dtype=bool)):
        if val and (i - last_i > cd):
            out.iloc[i] = True
            last_i = i
    return out


def _ntd_buy_sell_opportunity_masks(ntd: pd.Series,
                                    low_thr: float = -0.75,
                                    high_thr: float = 0.75,
                                    lookback: int = 10,
                                    confirm_bars: int = 3,
                                    cooldown_bars: int = 14):
    """
    Less-noisy NTD reversal rules:
      - BUY: NTD recently made a lower-zone extreme, then turns up for the
        confirmation window.
      - SELL: NTD recently made an upper-zone extreme, then turns down for the
        confirmation window.

    These masks are kept for compatibility, but the displayed green/red
    triangles are now drawn by the S/R reversal indicator so triangles only
    appear when price actually reverses from support/resistance.
    """
    n = _coerce_1d_series(ntd).astype(float)
    if n.dropna().shape[0] < max(4, confirm_bars + 2):
        idx = n.index if len(n) else pd.Index([])
        return pd.Series(False, index=idx), pd.Series(False, index=idx)

    try:
        lb = max(4, int(lookback))
    except Exception:
        lb = 10
    try:
        cb = max(2, int(confirm_bars))
    except Exception:
        cb = 3

    buy = pd.Series(False, index=n.index)
    sell = pd.Series(False, index=n.index)
    vals = n.to_numpy(dtype=float)

    for i in range(len(vals)):
        if i < max(lb, cb + 1):
            continue
        if not np.isfinite(vals[i]):
            continue

        recent = vals[max(0, i - lb):i + 1]
        if not np.isfinite(recent).any():
            continue

        # Confirmation requires a real turn, not just a threshold touch.
        confirm_slice = vals[i - cb:i + 1]
        if len(confirm_slice) < cb + 1 or not np.all(np.isfinite(confirm_slice)):
            continue
        deltas = np.diff(confirm_slice)

        recent_min = float(np.nanmin(recent))
        recent_max = float(np.nanmax(recent))
        prev_val = vals[i - cb]

        buy_turn = (recent_min <= low_thr) and np.all(deltas > 0) and (vals[i] > prev_val)
        sell_turn = (recent_max >= high_thr) and np.all(deltas < 0) and (vals[i] < prev_val)

        # Prefer the first bar after the extreme begins to reverse.
        if buy_turn:
            buy.iloc[i] = True
        if sell_turn:
            sell.iloc[i] = True

    buy = _apply_signal_cooldown(buy, cooldown_bars=cooldown_bars)
    sell = _apply_signal_cooldown(sell, cooldown_bars=cooldown_bars)
    return buy.fillna(False), sell.fillna(False)


# ========= NPX ↔ NTD overlay =========
def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    """
    Plot smoothed NPX on the NTD panel without printing ordinary cross triangles.
    Green/red triangles on the NTD panel are now reserved for confirmed
    support/resistance reversals only.
    """
    npx = _coerce_1d_series(npx).astype(float)
    ntd = _coerce_1d_series(ntd).astype(float)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx)
    ntd = ntd.reindex(idx)

    if npx.dropna().empty:
        return

    ax.plot(npx.index, npx.values, "-", linewidth=1.6, color="tab:gray", alpha=0.80,
            label="Smoothed NPX")


# ========= NTD price-chart triangle overlays remain removed; NTD-panel triangles are opportunity-only =========

# ========= Support/Resistance Reversal Index for NTD panels =========
def compute_sr_reversal_index(price: pd.Series,
                              support: pd.Series,
                              resistance: pd.Series,
                              smooth_span: int = 4) -> pd.Series:
    """
    Channel-position indicator used on the NTD panel:
      -1.0 means price is at/near support
      +1.0 means price is at/near resistance

    This makes support/resistance reversals visible on the same normalized
    scale as NTD, which is better suited for day-trading than raw NPX crosses.
    """
    p = _coerce_1d_series(price).astype(float)
    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()

    if p.empty:
        return pd.Series(index=p.index, dtype=float)

    width = (res - sup).replace(0, np.nan)
    sri = ((p - sup) / width) * 2.0 - 1.0
    sri = sri.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)

    try:
        span = int(smooth_span)
    except Exception:
        span = 1
    if span > 1:
        sri = sri.ewm(span=span, adjust=False).mean()

    return sri.reindex(p.index)


def sr_reversal_opportunity_masks(price: pd.Series,
                                  support: pd.Series,
                                  resistance: pd.Series,
                                  trend_slope: float,
                                  prox: float = 0.0025,
                                  zone: float = 0.65,
                                  lookback: int = 5,
                                  confirm_bars: int = 2,
                                  smooth_span: int = 4):
    """
    Confirmed, lower-noise support/resistance reversal rules.

    BUY appears only when:
      - the price-chart trendline is upward,
      - price recently touched the support zone,
      - price formed a local low and then closed higher for the confirmation
        window,
      - the S/R Reversal Index also turned upward from the support zone.

    SELL appears only when:
      - the price-chart trendline is downward,
      - price recently touched the resistance zone,
      - price formed a local high and then closed lower for the confirmation
        window,
      - the S/R Reversal Index also turned downward from the resistance zone.

    The function applies a cooldown so one reversal prints one triangle instead
    of repeated triangles on each follow-through bar.
    """
    p = _coerce_1d_series(price).astype(float)
    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    sri = compute_sr_reversal_index(p, sup, res, smooth_span=smooth_span)

    idx = p.index
    buy = pd.Series(False, index=idx)
    sell = pd.Series(False, index=idx)

    try:
        slope = float(trend_slope)
    except Exception:
        slope = np.nan

    if p.dropna().shape[0] < 6 or not np.isfinite(slope):
        return buy, sell, sri

    try:
        lb = max(5, int(lookback))
    except Exception:
        lb = 6
    try:
        cb = max(2, int(confirm_bars))
    except Exception:
        cb = 2
    try:
        z = float(zone)
    except Exception:
        z = 0.65
    try:
        px = max(0.0, float(prox))
    except Exception:
        px = 0.0025

    uptrend = slope >= 0.0
    downtrend = slope < 0.0

    pvals = p.to_numpy(dtype=float)
    supvals = sup.to_numpy(dtype=float)
    resvals = res.to_numpy(dtype=float)
    srivals = sri.to_numpy(dtype=float)

    # Minimum bounce/rejection as a fraction of the recent S/R channel width.
    # This avoids plotting triangles for tiny wiggles at the level.
    min_channel_move = 0.10

    for i in range(len(idx)):
        if i < max(lb, cb + 2):
            continue
        if not np.all(np.isfinite([pvals[i], srivals[i], supvals[i], resvals[i]])):
            continue

        start_recent = max(0, i - lb)
        recent_p = pvals[start_recent:i + 1]
        recent_sup = supvals[start_recent:i + 1]
        recent_res = resvals[start_recent:i + 1]
        recent_sri = srivals[start_recent:i + 1]

        if not (np.isfinite(recent_p).any() and np.isfinite(recent_sup).any() and np.isfinite(recent_res).any()):
            continue

        channel_width = float(np.nanmedian(recent_res - recent_sup))
        if not np.isfinite(channel_width) or channel_width <= 0:
            channel_width = max(abs(float(pvals[i])) * 0.0005, 1e-9)
        min_bounce = channel_width * min_channel_move

        low_pos = int(np.nanargmin(recent_p))
        high_pos = int(np.nanargmax(recent_p))
        low_val = float(recent_p[low_pos])
        high_val = float(recent_p[high_pos])
        sup_at_low = float(recent_sup[low_pos]) if np.isfinite(recent_sup[low_pos]) else np.nan
        res_at_high = float(recent_res[high_pos]) if np.isfinite(recent_res[high_pos]) else np.nan

        # The touch must happen before the current bar so the marker represents
        # a reversal after the touch, not the touch itself.
        support_was_touched = (
            np.isfinite(low_val) and np.isfinite(sup_at_low)
            and low_pos < len(recent_p) - 1
            and low_val <= sup_at_low * (1.0 + px)
        )
        resistance_was_touched = (
            np.isfinite(high_val) and np.isfinite(res_at_high)
            and high_pos < len(recent_p) - 1
            and high_val >= res_at_high * (1.0 - px)
        )

        # Confirmation by price action.
        confirm_prices = pvals[i - cb:i + 1]
        confirm_sri = srivals[i - cb:i + 1]
        if len(confirm_prices) < cb + 1 or len(confirm_sri) < cb + 1:
            continue
        if not (np.all(np.isfinite(confirm_prices)) and np.all(np.isfinite(confirm_sri))):
            continue

        price_deltas = np.diff(confirm_prices)
        sri_deltas = np.diff(confirm_sri)

        price_reversed_up = np.all(price_deltas > 0) and (pvals[i] - low_val >= min_bounce)
        price_reversed_down = np.all(price_deltas < 0) and (high_val - pvals[i] >= min_bounce)

        sri_recent_min = float(np.nanmin(recent_sri)) if np.isfinite(recent_sri).any() else np.nan
        sri_recent_max = float(np.nanmax(recent_sri)) if np.isfinite(recent_sri).any() else np.nan

        sri_reversed_up = (
            np.isfinite(sri_recent_min)
            and sri_recent_min <= -z
            and np.all(sri_deltas > 0)
            and srivals[i] > srivals[i - cb]
        )
        sri_reversed_down = (
            np.isfinite(sri_recent_max)
            and sri_recent_max >= z
            and np.all(sri_deltas < 0)
            and srivals[i] < srivals[i - cb]
        )

        raw_buy = uptrend and support_was_touched and price_reversed_up and sri_reversed_up
        raw_sell = downtrend and resistance_was_touched and price_reversed_down and sri_reversed_down

        if raw_buy:
            buy.iloc[i] = True
        if raw_sell:
            sell.iloc[i] = True

    cooldown = max(lb * 2, cb * 4, 12)
    buy = _apply_signal_cooldown(buy, cooldown_bars=cooldown)
    sell = _apply_signal_cooldown(sell, cooldown_bars=cooldown)

    return buy.fillna(False), sell.fillna(False), sri


def overlay_sr_reversal_indicator_on_ntd(ax,
                                         price: pd.Series,
                                         support: pd.Series,
                                         resistance: pd.Series,
                                         trend_slope: float,
                                         ntd: pd.Series = None,
                                         prox: float = 0.0025,
                                         zone: float = 0.65,
                                         lookback: int = 5,
                                         confirm_bars: int = 2,
                                         smooth_span: int = 4):
    """
    Adds a day-trading S/R reversal indicator to an NTD panel.

    Green ▲ = price reversed upward from support while the price trendline is up.
    Red ▼   = price reversed downward from resistance while the price trendline is down.
    """
    buy, sell, sri = sr_reversal_opportunity_masks(
        price=price,
        support=support,
        resistance=resistance,
        trend_slope=trend_slope,
        prox=prox,
        zone=zone,
        lookback=lookback,
        confirm_bars=confirm_bars,
        smooth_span=smooth_span,
    )

    if not sri.dropna().empty:
        ax.plot(sri.index, sri.values, "-", linewidth=1.4, color="tab:purple", alpha=0.85,
                label="S/R Reversal Index")

    if ntd is None:
        yref = sri
    else:
        yref = _coerce_1d_series(ntd).reindex(sri.index)
        if yref.dropna().empty:
            yref = sri

    if buy.any():
        idx_buy = list(buy[buy].index)
        y_buy = yref.reindex(idx_buy).fillna(sri.reindex(idx_buy)).fillna(-zone)
        ax.scatter(idx_buy, y_buy.values, marker="^", s=120, color="tab:green", zorder=12,
                   label="BUY: support reversal")
    if sell.any():
        idx_sell = list(sell[sell].index)
        y_sell = yref.reindex(idx_sell).fillna(sri.reindex(idx_sell)).fillna(zone)
        ax.scatter(idx_sell, y_sell.values, marker="v", s=120, color="tab:red", zorder=12,
                   label="SELL: resistance reversal")


# ========= S/R Reversal zero-line scanner helpers =========
def _last_zero_cross_info(line_like: pd.Series, max_bars_since: int = 20):
    """
    Return the latest zero-line cross for a normalized indicator.

    Direction:
      - Upward: previous value < 0 and current value >= 0
      - Downward: previous value > 0 and current value <= 0
    """
    s = _coerce_1d_series(line_like).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return None
    try:
        max_bars = max(0, int(max_bars_since))
    except Exception:
        max_bars = 20

    vals = s.to_numpy(dtype=float)
    idx = s.index
    events = []
    for i in range(1, len(vals)):
        prev_v = vals[i - 1]
        curr_v = vals[i]
        if not np.all(np.isfinite([prev_v, curr_v])):
            continue
        if prev_v < 0.0 and curr_v >= 0.0:
            events.append({
                "direction": "Upward",
                "bar_index": i,
                "time": idx[i],
                "value_at_cross": float(curr_v),
            })
        elif prev_v > 0.0 and curr_v <= 0.0:
            events.append({
                "direction": "Downward",
                "bar_index": i,
                "time": idx[i],
                "value_at_cross": float(curr_v),
            })

    if not events:
        return None

    ev = events[-1]
    bars_since = (len(s) - 1) - int(ev["bar_index"])
    if bars_since > max_bars:
        return None

    ev["bars_since"] = int(bars_since)
    ev["current_value"] = float(vals[-1])
    return ev


@st.cache_data(ttl=120, show_spinner=False)
def sr_reversal_cross_info_daily(symbol: str, smooth_span: int = 8, recent_bars: int = 20):
    """
    Scan one symbol's Daily S/R Reversal line for a recent zero-line cross.
    Daily S/R uses the same 30-bar support/resistance window used in the
    daily price chart.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < 5:
            return None
        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        )
        ev = _last_zero_cross_info(sri, max_bars_since=int(recent_bars))
        if ev is None:
            return None
        return {
            "Symbol": symbol,
            "Chart": "Daily",
            "Direction": ev["direction"],
            "Bars Since Cross": ev["bars_since"],
            "Cross Time": ev["time"],
            "Value at Cross": ev["value_at_cross"],
            "Current S/R Reversal": ev["current_value"],
            "Last Close": _safe_last_float(close),
        }
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def sr_reversal_cross_info_hourly(symbol: str, period: str = "1d",
                                  sr_window: int = 60,
                                  smooth_span: int = 8,
                                  recent_bars: int = 24):
    """
    Scan one symbol's Hourly/5-minute intraday S/R Reversal line for a recent
    zero-line cross. The app's intraday chart uses 5-minute bars, so "Hourly"
    here follows the existing hourly/intraday chart data source.
    """
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = _coerce_1d_series(df["Close"]).ffill().dropna()
        if close.shape[0] < 5:
            return None
        win = max(2, int(sr_window))
        support = close.rolling(win, min_periods=1).min()
        resistance = close.rolling(win, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        )
        ev = _last_zero_cross_info(sri, max_bars_since=int(recent_bars))
        if ev is None:
            return None
        return {
            "Symbol": symbol,
            "Chart": "Hourly",
            "Direction": ev["direction"],
            "Bars Since Cross": ev["bars_since"],
            "Cross Time": ev["time"],
            "Value at Cross": ev["value_at_cross"],
            "Current S/R Reversal": ev["current_value"],
            "Last Close": _safe_last_float(close),
        }
    except Exception:
        return None


def _render_sr_cross_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No recent crosses found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
        for col in ["Value at Cross", "Current S/R Reversal", "Last Close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Cross Time" in df.columns:
            df["Cross Time"] = df["Cross Time"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _last_threshold_up_cross_after_reversal_info(line_like: pd.Series,
                                                threshold: float = -0.75,
                                                recent_bars: int = 20,
                                                confirm_bars: int = 2):
    """
    Return the latest upward cross through a support-side threshold after the
    line previously traded below that threshold and is now rising.

    Used for S/R Reversal Index cases where the line was below -0.75, reversed,
    crossed back above -0.75, and is still moving upward.
    """
    s = _coerce_1d_series(line_like).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < max(4, int(confirm_bars) + 2):
        return None

    try:
        th = float(threshold)
    except Exception:
        th = -0.75
    try:
        rb = max(1, int(recent_bars))
    except Exception:
        rb = 20
    try:
        cb = max(1, int(confirm_bars))
    except Exception:
        cb = 2

    vals = s.to_numpy(dtype=float)
    idx = s.index
    events = []

    for i in range(1, len(vals)):
        prev_v = vals[i - 1]
        curr_v = vals[i]
        if not np.all(np.isfinite([prev_v, curr_v])):
            continue
        if prev_v < th and curr_v >= th:
            lookback_start = max(0, i - max(rb, cb + 2))
            prior_vals = vals[lookback_start:i + 1]
            if not np.isfinite(prior_vals).any():
                continue
            recent_min = float(np.nanmin(prior_vals))
            if not np.isfinite(recent_min) or recent_min >= th:
                continue
            events.append({
                "bar_index": i,
                "time": idx[i],
                "value_at_cross": float(curr_v),
                "recent_min": recent_min,
            })

    if not events:
        return None

    ev = events[-1]
    bars_since = (len(s) - 1) - int(ev["bar_index"])
    if bars_since > rb:
        return None

    current_value = float(vals[-1])
    if not np.isfinite(current_value) or current_value < th:
        return None

    confirm_start = max(0, len(vals) - cb - 1)
    confirm_slice = vals[confirm_start:]
    if confirm_slice.size < cb + 1 or not np.all(np.isfinite(confirm_slice)):
        return None
    deltas = np.diff(confirm_slice)
    if not np.all(deltas > 0):
        return None

    ev["bars_since"] = int(bars_since)
    ev["current_value"] = current_value
    ev["change_since_cross"] = current_value - float(ev["value_at_cross"])
    return ev


@st.cache_data(ttl=120, show_spinner=False)
def sr_reversal_minus075_up_cross_info_daily(symbol: str,
                                             smooth_span: int = 8,
                                             recent_bars: int = 20,
                                             confirm_bars: int = 2,
                                             threshold: float = -0.75,
                                             slope_lookback: int = 90):
    """
    Daily scanner row for symbols where the S/R Reversal Index was below the
    support-side threshold, recently crossed upward through it, and is still
    rising.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(35, int(slope_lookback) // 2):
            return None

        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        )

        ev = _last_threshold_up_cross_after_reversal_info(
            sri,
            threshold=float(threshold),
            recent_bars=int(recent_bars),
            confirm_bars=int(confirm_bars),
        )
        if ev is None:
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_direction = "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward"

        return {
            "Symbol": symbol,
            "Bars Since -0.75 Cross": ev["bars_since"],
            "Cross Date": ev["time"],
            "Value at Cross": ev["value_at_cross"],
            "Current S/R Reversal": ev["current_value"],
            "Recent Minimum Below -0.75": ev["recent_min"],
            "Change Since Cross": ev["change_since_cross"],
            "Trend Direction": trend_direction,
            "Trend Slope": trend_slope,
            "Last Close": _safe_last_float(close),
        }
    except Exception:
        return None


def _render_minus075_up_cross_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No recent upward -0.75 crosses found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since -0.75 Cross", "Symbol"], ascending=[True, True])
        for col in [
            "Value at Cross",
            "Current S/R Reversal",
            "Recent Minimum Below -0.75",
            "Change Since Cross",
            "Trend Slope",
            "Last Close",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Cross Date" in df.columns:
            df["Cross Date"] = df["Cross Date"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _last_minus075_support_reversal_info(sri: pd.Series,
                                         threshold: float = -0.75,
                                         recent_bars: int = 20,
                                         confirm_bars: int = 3):
    """
    Find the most recent daily S/R Reversal line support-zone reversal.

    A match requires:
      - current S/R Reversal value is still below the threshold,
      - the line recently traded at/below the threshold,
      - the recent low occurred before the current bar,
      - the line has been rising through the confirmation window.
    """
    s = _coerce_1d_series(sri).dropna()
    if s.shape[0] < max(5, int(confirm_bars) + 2):
        return None

    try:
        th = float(threshold)
    except Exception:
        th = -0.75
    try:
        rb = max(2, int(recent_bars))
    except Exception:
        rb = 20
    try:
        cb = max(1, int(confirm_bars))
    except Exception:
        cb = 3

    vals = s.to_numpy(dtype=float)
    idx = s.index

    current_value = float(vals[-1])
    if not np.isfinite(current_value) or current_value > th:
        return None

    start = max(0, len(vals) - rb - 1)
    recent_vals = vals[start:]
    if recent_vals.size < max(3, cb + 1) or not np.isfinite(recent_vals).any():
        return None

    recent_min_pos = int(np.nanargmin(recent_vals))
    recent_min_abs = start + recent_min_pos
    recent_min = float(vals[recent_min_abs])

    # It must have visited the -0.75 support zone and started reversing after that low.
    if not np.isfinite(recent_min) or recent_min > th:
        return None
    if recent_min_abs >= len(vals) - 1:
        return None

    bars_since_reversal = (len(vals) - 1) - recent_min_abs
    if bars_since_reversal > rb:
        return None

    # Confirmation: the latest values are rising and above the recent low.
    confirm_start = max(0, len(vals) - cb - 1)
    confirm_slice = vals[confirm_start:]
    if confirm_slice.size < cb + 1 or not np.all(np.isfinite(confirm_slice)):
        return None
    deltas = np.diff(confirm_slice)
    if not np.all(deltas > 0):
        return None
    if current_value <= recent_min:
        return None

    return {
        "bars_since_reversal": int(bars_since_reversal),
        "reversal_time": idx[recent_min_abs],
        "current_value": current_value,
        "recent_min": recent_min,
        "value_change_since_reversal": current_value - recent_min,
    }


@st.cache_data(ttl=120, show_spinner=False)
def minus075_sr_crosser_info_daily(symbol: str,
                                   smooth_span: int = 8,
                                   recent_bars: int = 20,
                                   confirm_bars: int = 3,
                                   threshold: float = -0.75,
                                   slope_lookback: int = 90):
    """
    Daily-only scanner row for symbols where the S/R Reversal line is below
    -0.75 and has recently turned upward from that support-side zone.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(35, int(slope_lookback) // 2):
            return None

        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        )

        ev = _last_minus075_support_reversal_info(
            sri,
            threshold=float(threshold),
            recent_bars=int(recent_bars),
            confirm_bars=int(confirm_bars),
        )
        if ev is None:
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_group = "Upward Trend" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward Trend"

        return {
            "Symbol": symbol,
            "Trend Group": trend_group,
            "Bars Since Reversal": ev["bars_since_reversal"],
            "Reversal Date": ev["reversal_time"],
            "Current S/R Reversal": ev["current_value"],
            "Recent Minimum": ev["recent_min"],
            "Change Since Reversal": ev["value_change_since_reversal"],
            "Last Close": _safe_last_float(close),
            "Trend Slope": trend_slope,
        }
    except Exception:
        return None


def _render_minus075_sr_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No symbols found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since Reversal", "Symbol"], ascending=[True, True])
        for col in ["Current S/R Reversal", "Recent Minimum", "Change Since Reversal", "Last Close", "Trend Slope"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Reversal Date" in df.columns:
            df["Reversal Date"] = df["Reversal Date"].astype(str)
        if "Trend Group" in df.columns:
            df = df.drop(columns=["Trend Group"])
    st.dataframe(df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=120, show_spinner=False)
def minus075_sr_below_info_daily(symbol: str,
                                 smooth_span: int = 8,
                                 threshold: float = -0.75,
                                 slope_lookback: int = 90):
    """
    Daily-only scanner row for symbols where the S/R Reversal line is currently
    below the selected support-side threshold, regardless of whether a confirmed
    upward reversal has already happened.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(35, int(slope_lookback) // 2):
            return None

        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        ).dropna()

        if sri.empty:
            return None

        current_value = _safe_last_float(sri)
        try:
            th = float(threshold)
        except Exception:
            th = -0.75

        if not np.isfinite(current_value) or current_value > th:
            return None

        below_mask = sri <= th
        bars_below = 0
        for val in reversed(below_mask.to_numpy(dtype=bool)):
            if val:
                bars_below += 1
            else:
                break

        first_below_time = None
        if bars_below > 0:
            first_below_time = sri.index[-bars_below]

        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_group = "Upward Trend" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward Trend"

        recent_window = sri.iloc[-min(len(sri), 20):]
        recent_min = _safe_last_float(recent_window.min()) if len(recent_window) else np.nan

        return {
            "Symbol": symbol,
            "Trend Group": trend_group,
            "Bars Below Threshold": int(bars_below),
            "First Below Date": first_below_time,
            "Current S/R Reversal": current_value,
            "Recent Minimum": recent_min,
            "Last Close": _safe_last_float(close),
            "Trend Slope": trend_slope,
        }
    except Exception:
        return None


def _render_minus075_below_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No symbols found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Current S/R Reversal", "Symbol"], ascending=[True, True])
        for col in ["Current S/R Reversal", "Recent Minimum", "Last Close", "Trend Slope"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "First Below Date" in df.columns:
            df["First Below Date"] = df["First Below Date"].astype(str)
        if "Trend Group" in df.columns:
            df = df.drop(columns=["Trend Group"])
    st.dataframe(df, use_container_width=True, hide_index=True)



@st.cache_data(ttl=120, show_spinner=False)
def sr_cross_daily_below_threshold_info(symbol: str,
                                        smooth_span: int = 8,
                                        threshold: float = -0.75,
                                        slope_lookback: int = 90):
    """
    Daily S/R Cross scanner row for symbols where the S/R Reversal Index is
    currently below the selected support-side threshold.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < 35:
            return None

        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        ).dropna()

        if sri.empty:
            return None

        current_value = _safe_last_float(sri)
        th = float(threshold)
        if not np.isfinite(current_value) or current_value > th:
            return None

        below_mask = (sri <= th)
        bars_below = 0
        for val in reversed(below_mask.to_numpy(dtype=bool)):
            if val:
                bars_below += 1
            else:
                break

        first_below_time = sri.index[-bars_below] if bars_below > 0 else None
        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_direction = "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward"

        return {
            "Symbol": symbol,
            "Current S/R Reversal": current_value,
            "Bars Below -0.75": int(bars_below),
            "First Below Date": first_below_time,
            "Last Close": _safe_last_float(close),
            "Trend Direction": trend_direction,
            "Trend Slope": trend_slope,
        }
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def sr_cross_daily_upward_zero_info(symbol: str,
                                    smooth_span: int = 8,
                                    recent_bars: int = 30,
                                    slope_lookback: int = 90):
    """
    Daily S/R Cross scanner row for symbols where the S/R Reversal Index has
    recently crossed the 0.0 line upward.
    """
    try:
        row = sr_reversal_cross_info_daily(
            symbol,
            smooth_span=int(smooth_span),
            recent_bars=int(recent_bars),
        )
        if row is None or row.get("Direction") != "Upward":
            return None

        close = fetch_hist(symbol)
        close = _coerce_1d_series(close).dropna()
        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_direction = "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward"

        return {
            "Symbol": symbol,
            "Bars Since Cross": int(row.get("Bars Since Cross", 0)),
            "Cross Time": row.get("Cross Time"),
            "Value at Cross": row.get("Value at Cross"),
            "Current S/R Reversal": row.get("Current S/R Reversal"),
            "Last Close": row.get("Last Close"),
            "Trend Direction": trend_direction,
            "Trend Slope": trend_slope,
        }
    except Exception:
        return None


def _render_sr_cross_daily_threshold_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No symbols found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Current S/R Reversal", "Symbol"], ascending=[True, True])
        for col in ["Current S/R Reversal", "Last Close", "Trend Slope"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "First Below Date" in df.columns:
            df["First Below Date"] = df["First Below Date"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_sr_cross_daily_upward_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No recent upward 0.0 crosses found.")
        return
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
        for col in ["Value at Cross", "Current S/R Reversal", "Last Close", "Trend Slope"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Cross Time" in df.columns:
            df["Cross Time"] = df["Cross Time"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _last_level_up_cross_info(line_like: pd.Series,
                              level: float = -0.5,
                              max_bars_since: int = 30,
                              require_current_above: bool = True):
    """
    Return the latest upward cross through an arbitrary S/R Reversal level.

    Upward cross definition:
      previous value < level and current value >= level.
    """
    s = _coerce_1d_series(line_like).replace([np.inf, -np.inf], np.nan).dropna()
    if s.shape[0] < 2:
        return None

    try:
        lvl = float(level)
    except Exception:
        lvl = -0.5
    try:
        max_bars = max(0, int(max_bars_since))
    except Exception:
        max_bars = 30

    vals = s.to_numpy(dtype=float)
    idx = s.index
    events = []

    for i in range(1, len(vals)):
        prev_v = vals[i - 1]
        curr_v = vals[i]
        if not np.all(np.isfinite([prev_v, curr_v])):
            continue
        if prev_v < lvl and curr_v >= lvl:
            events.append({
                "bar_index": i,
                "time": idx[i],
                "value_at_cross": float(curr_v),
                "previous_value": float(prev_v),
            })

    if not events:
        return None

    ev = events[-1]
    bars_since = (len(s) - 1) - int(ev["bar_index"])
    if bars_since > max_bars:
        return None

    current_value = float(vals[-1])
    if require_current_above and (not np.isfinite(current_value) or current_value < lvl):
        return None

    current_slope = np.nan
    try:
        recent = s.iloc[-min(8, len(s)):].dropna()
        if len(recent) >= 2:
            x = np.arange(len(recent), dtype=float)
            current_slope = float(np.polyfit(x, recent.to_numpy(dtype=float), 1)[0])
    except Exception:
        current_slope = np.nan

    ev["bars_since"] = int(bars_since)
    ev["current_value"] = current_value
    ev["current_slope"] = current_slope
    ev["current_direction"] = "Upward" if np.isfinite(current_slope) and current_slope >= 0 else "Downward"
    ev["level"] = lvl
    return ev


@st.cache_data(ttl=120, show_spinner=False)
def sr_reversal_level_up_cross_info_daily(symbol: str,
                                          level: float = -0.5,
                                          smooth_span: int = 8,
                                          recent_bars: int = 30,
                                          slope_lookback: int = 90):
    """
    Daily scanner row for symbols where the S/R Reversal Index recently crossed
    upward through the requested level on the NTD chart.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(35, int(slope_lookback) // 2):
            return None

        support = close.rolling(30, min_periods=1).min()
        resistance = close.rolling(30, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        ).dropna()

        if sri.empty:
            return None

        ev = _last_level_up_cross_info(
            sri,
            level=float(level),
            max_bars_since=int(recent_bars),
            require_current_above=True,
        )
        if ev is None:
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_direction = "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward"
        level_label = f"{float(level):.2f}"

        return {
            "Symbol": symbol,
            "Cross Level": level_label,
            "Bars Since Cross": ev["bars_since"],
            "Cross Time": ev["time"],
            "Previous S/R Reversal": ev["previous_value"],
            "Value at Cross": ev["value_at_cross"],
            "Current S/R Reversal": ev["current_value"],
            "Current S/R Direction": ev["current_direction"],
            "Current S/R Slope": ev["current_slope"],
            "Trend Direction": trend_direction,
            "Trend Slope": trend_slope,
            "Last Close": _safe_last_float(close),
        }
    except Exception:
        return None


def _render_sr_level_up_cross_table(title: str, rows: list, empty_text: str):
    st.subheader(title)
    if not rows:
        st.info(empty_text)
        return

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
        for col in [
            "Previous S/R Reversal",
            "Value at Cross",
            "Current S/R Reversal",
            "Current S/R Slope",
            "Trend Slope",
            "Last Close",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Cross Time" in df.columns:
            df["Cross Time"] = df["Cross Time"].astype(str)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _sr_reversal_zone_label(value: float) -> str:
    """Human-readable location of the S/R Reversal Index."""
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if v < -0.75:
        return "Below -0.75"
    if v < -0.50:
        return "-0.75 to -0.50"
    if v < 0.0:
        return "-0.50 to 0.00"
    return "Above 0.00"


def _safe_pct_distance(numerator: float, denominator: float) -> float:
    try:
        n = float(numerator)
        d = float(denominator)
    except Exception:
        return float("nan")
    if not np.isfinite(n) or not np.isfinite(d) or d == 0:
        return float("nan")
    return float(n / abs(d) * 100.0)


def _trade_status_sort_rank(status: str) -> int:
    order = {
        "BUY CONFIRMED": 0,
        "BUY SETUP": 1,
        "EARLY BUY SETUP": 2,
        "BUY CONFIRMATION PENDING": 3,
        "LATE BUY / WAIT PULLBACK": 4,
        "WATCH": 5,
        "WEAK CONFIRMATION": 6,
        "AVOID": 7,
    }
    return order.get(str(status), 99)


def _first_valid_bars_since(*values) -> float:
    valid = []
    for value in values:
        try:
            v = float(value)
            if np.isfinite(v):
                valid.append(v)
        except Exception:
            pass
    return min(valid) if valid else float("nan")


@st.cache_data(ttl=120, show_spinner=False)
def actionable_sr_long_pick_daily(symbol: str,
                                  smooth_span: int = 8,
                                  recent_bars: int = 30,
                                  slope_lookback: int = 90,
                                  sr_window: int = 30):
    """
    Build one actionable daily long-candidate row from the S/R Reversal Index.

    A symbol is returned when it is either:
      - currently below -0.75, or
      - recently crossed upward through -0.75, -0.50, or 0.00.

    The row includes trade status, setup quality, support/resistance distances,
    reward/risk, and a suggested action.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(40, int(sr_window) + 5, int(slope_lookback) // 2):
            return None

        sr_window = int(max(5, sr_window))
        support = close.rolling(sr_window, min_periods=1).min()
        resistance = close.rolling(sr_window, min_periods=1).max()
        sri = compute_sr_reversal_index(
            price=close,
            support=support,
            resistance=resistance,
            smooth_span=int(smooth_span),
        ).dropna()
        if sri.empty:
            return None

        current_sri = _safe_last_float(sri)
        last_close = _safe_last_float(close)
        current_support = _safe_last_float(support)
        current_resistance = _safe_last_float(resistance)
        if not np.isfinite(current_sri) or not np.isfinite(last_close):
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        trend_direction = "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward"

        sri_recent = sri.iloc[-min(8, len(sri)):].dropna()
        sri_slope = float("nan")
        if len(sri_recent) >= 2:
            x = np.arange(len(sri_recent), dtype=float)
            sri_slope = float(np.polyfit(x, sri_recent.to_numpy(dtype=float), 1)[0])
        reversal_direction = "Rising" if np.isfinite(sri_slope) and sri_slope >= 0 else "Falling"

        ev_m075 = _last_level_up_cross_info(sri, level=-0.75, max_bars_since=int(recent_bars), require_current_above=True)
        ev_m050 = _last_level_up_cross_info(sri, level=-0.50, max_bars_since=int(recent_bars), require_current_above=True)
        ev_000 = _last_level_up_cross_info(sri, level=0.00, max_bars_since=int(recent_bars), require_current_above=True)

        currently_below_m075 = bool(current_sri < -0.75)
        has_recent_cross = ev_m075 is not None or ev_m050 is not None or ev_000 is not None
        if not currently_below_m075 and not has_recent_cross:
            return None

        bars_m075 = ev_m075["bars_since"] if ev_m075 is not None else np.nan
        bars_m050 = ev_m050["bars_since"] if ev_m050 is not None else np.nan
        bars_000 = ev_000["bars_since"] if ev_000 is not None else np.nan
        nearest_bars = _first_valid_bars_since(bars_000, bars_m050, bars_m075)

        price_above_support = bool(np.isfinite(current_support) and last_close >= current_support)
        price_below_resistance = bool(np.isfinite(current_resistance) and last_close <= current_resistance)

        risk = last_close - current_support if np.all(np.isfinite([last_close, current_support])) else np.nan
        reward = current_resistance - last_close if np.all(np.isfinite([last_close, current_resistance])) else np.nan
        if np.isfinite(risk) and risk > 0 and np.isfinite(reward):
            reward_risk = float(reward / risk)
        else:
            reward_risk = np.nan

        dist_support_pct = _safe_pct_distance(last_close - current_support, last_close)
        dist_resist_pct = _safe_pct_distance(current_resistance - last_close, last_close)

        if trend_direction != "Upward":
            if ev_000 is not None or ev_m050 is not None or ev_m075 is not None:
                trade_status = "WEAK CONFIRMATION"
                suggested_action = "Avoid until trend turns upward"
            else:
                trade_status = "AVOID"
                suggested_action = "Avoid"
        elif ev_000 is not None:
            if ev_000["bars_since"] <= max(10, int(recent_bars) // 3) and reversal_direction == "Rising" and price_above_support:
                trade_status = "BUY CONFIRMED"
                suggested_action = "Buy pullback/hold above support"
            else:
                trade_status = "LATE BUY / WAIT PULLBACK"
                suggested_action = "Wait for pullback or fresh support hold"
        elif ev_m050 is not None and reversal_direction == "Rising":
            trade_status = "BUY SETUP"
            suggested_action = "Watch for 0.0 cross confirmation"
        elif ev_m075 is not None and reversal_direction == "Rising":
            trade_status = "EARLY BUY SETUP"
            suggested_action = "Early watch; wait for -0.5/0.0 confirmation"
        elif currently_below_m075 and reversal_direction == "Rising":
            trade_status = "BUY CONFIRMATION PENDING"
            suggested_action = "Watch for -0.75 upward cross"
        elif currently_below_m075:
            trade_status = "WATCH"
            suggested_action = "Oversold watch; wait for turn upward"
        else:
            trade_status = "WATCH"
            suggested_action = "Monitor"

        score = 0.0
        if trend_direction == "Upward":
            score += 25
        if reversal_direction == "Rising":
            score += 20
        if ev_000 is not None:
            score += 25
        elif ev_m050 is not None:
            score += 18
        elif ev_m075 is not None:
            score += 12
        elif currently_below_m075:
            score += 8
        if price_above_support:
            score += 10
        if np.isfinite(reward_risk) and reward_risk >= 1.5:
            score += 10
        elif np.isfinite(reward_risk) and reward_risk >= 1.0:
            score += 5
        if np.isfinite(nearest_bars):
            if nearest_bars <= 5:
                score += 10
            elif nearest_bars <= 15:
                score += 5
            elif nearest_bars > max(20, int(recent_bars) * 0.75):
                score -= 10
        if trade_status in ("WEAK CONFIRMATION", "AVOID"):
            score = min(score, 35)
        setup_quality = int(max(0, min(100, round(score))))

        return {
            "Symbol": symbol,
            "Trade Status": trade_status,
            "Setup Quality": setup_quality,
            "Current Zone": _sr_reversal_zone_label(current_sri),
            "Reversal Direction": reversal_direction,
            "Current S/R Reversal": current_sri,
            "Bars Since -0.75 Cross": bars_m075,
            "Bars Since -0.50 Cross": bars_m050,
            "Bars Since 0.00 Cross": bars_000,
            "Trend Direction": trend_direction,
            "Trend Slope": trend_slope,
            "Last Close": last_close,
            "Support": current_support,
            "Resistance": current_resistance,
            "Price Above Support": price_above_support,
            "Distance to Support %": dist_support_pct,
            "Distance to Resistance %": dist_resist_pct,
            "Reward/Risk": reward_risk,
            "Suggested Action": suggested_action,
            "Sort Rank": _trade_status_sort_rank(trade_status),
            "Nearest Bars Since Cross": nearest_bars,
        }
    except Exception:
        return None


def _render_actionable_sr_long_picks_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No actionable S/R long candidates found.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No actionable S/R long candidates found.")
        return

    sort_cols = ["Sort Rank", "Setup Quality", "Nearest Bars Since Cross", "Symbol"]
    ascending = [True, False, True, True]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    existing_ascending = [ascending[sort_cols.index(c)] for c in existing_sort_cols]
    if existing_sort_cols:
        df = df.sort_values(existing_sort_cols, ascending=existing_ascending, na_position="last")

    for col in [
        "Current S/R Reversal",
        "Trend Slope",
        "Last Close",
        "Support",
        "Resistance",
        "Distance to Support %",
        "Distance to Resistance %",
        "Reward/Risk",
        "Nearest Bars Since Cross",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Bars Since -0.75 Cross", "Bars Since -0.50 Cross", "Bars Since 0.00 Cross"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    display_cols = [
        "Symbol",
        "Trade Status",
        "Setup Quality",
        "Current Zone",
        "Reversal Direction",
        "Current S/R Reversal",
        "Bars Since -0.75 Cross",
        "Bars Since -0.50 Cross",
        "Bars Since 0.00 Cross",
        "Trend Direction",
        "Trend Slope",
        "Last Close",
        "Support",
        "Resistance",
        "Price Above Support",
        "Distance to Support %",
        "Distance to Resistance %",
        "Reward/Risk",
        "Suggested Action",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].reset_index(drop=True)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Setup Quality": st.column_config.ProgressColumn(
                "Setup Quality",
                help="0–100 ranking based on trend alignment, S/R reversal timing, support hold, and reward/risk.",
                min_value=0,
                max_value=100,
                format="%d",
            ),
            "Distance to Support %": st.column_config.NumberColumn("Distance to Support %", format="%.2f%%"),
            "Distance to Resistance %": st.column_config.NumberColumn("Distance to Resistance %", format="%.2f%%"),
            "Reward/Risk": st.column_config.NumberColumn("Reward/Risk", format="%.2f"),
            "Current S/R Reversal": st.column_config.NumberColumn("Current S/R Reversal", format="%.4f"),
            "Trend Slope": st.column_config.NumberColumn("Trend Slope", format="%.5f"),
            "Last Close": st.column_config.NumberColumn("Last Close", format="%.5f"),
            "Support": st.column_config.NumberColumn("Support", format="%.5f"),
            "Resistance": st.column_config.NumberColumn("Resistance", format="%.5f"),
        },
    )


# ========= Sessions =========
NY_TZ   = pytz.timezone("America/New_York")
LDN_TZ  = pytz.timezone("Europe/London")

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
        if lo <= dt_open_pst  <= hi: opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi: closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ,  8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}

def draw_session_lines(ax, lines: dict):
    ax.plot([], [], linestyle="-",  color="tab:blue",   label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",   label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:orange", alpha=0.35)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=0.35)
    ax.text(0.99, 0.98, "Session times in PST",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))

# ========= Signals & News helpers =========
EW_CONFIDENCE = 0.95

def elliott_conf_signal(price_now: float, fc_vals: pd.Series, conf: float = EW_CONFIDENCE):
    fc = _coerce_1d_series(fc_vals).dropna().to_numpy(dtype=float)
    if fc.size == 0 or not np.isfinite(price_now):
        return None
    p_up = float(np.mean(fc > price_now))
    p_dn = float(np.mean(fc < price_now))
    if p_up >= conf:
        return {"side": "BUY", "prob": p_up}
    if p_dn >= conf:
        return {"side": "SELL", "prob": p_dn}
    return None

def sr_proximity_signal(hc: pd.Series, res_h: pd.Series, sup_h: pd.Series,
                        fc_vals: pd.Series, threshold: float, prox: float):
    try:
        last_close = _safe_last_float(hc)
        res = _safe_last_float(res_h)
        sup = _safe_last_float(sup_h)
    except Exception:
        return None
    if not np.all(np.isfinite([last_close, res, sup])) or res <= sup:
        return None
    near_support = last_close <= sup * (1.0 + prox)
    near_resist  = last_close >= res * (1.0 - prox)
    fc = np.asarray(_coerce_1d_series(fc_vals).dropna(), dtype=float)
    if fc.size == 0:
        return None
    p_up_from_here = float(np.mean(fc > last_close))
    p_dn_from_here = float(np.mean(fc < last_close))
    if near_support and p_up_from_here >= threshold:
        return {"side": "BUY", "prob": p_up_from_here, "level": sup}
    if near_resist and p_dn_from_here >= threshold:
        return {"side": "SELL", "prob": p_dn_from_here, "level": res}
    return None

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
        rows.append({"time": dt_pst, "title": item.get("title",""), "publisher": item.get("publisher",""), "link": item.get("link","")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

def draw_news_markers(ax, times, ymin, ymax, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# ========= NTD channel =========
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
    start = None; prev_t = None
    for t, val in s.items():
        if val and start is None: start = t
        if not val and start is not None:
            if prev_t is not None: spans.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None and prev_t is not None: spans.append((start, prev_t))
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
    enter_from_above = (state.shift(1) ==  1) & (state == 0)
    if enter_from_below.any():
        ax.scatter(price.index[enter_from_below], [0.92]*int(enter_from_below.sum()),
                   marker="o", s=45, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above], [0.92]*int(enter_from_above.sum()),
                   marker="o", s=45, color="tab:orange", zorder=7, label="Enter from R")
    lbl = None; col = "black"
    last = state.dropna().iloc[-1] if state.dropna().shape[0] else np.nan
    if np.isfinite(last):
        if last == 0:
            lbl, col = "IN RANGE (S↔R)", "black"
        elif last > 0:
            lbl, col = "Above R", "tab:orange"
        else:
            lbl, col = "Below S", "tab:red"
        ax.text(0.99, 0.94, lbl, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.85))
    return last

# ========= BB Divergence =========
def _last_delta_sign(series_like: pd.Series) -> float:
    s = _coerce_1d_series(series_like).dropna()
    if len(s) < 2:
        return np.nan
    d = float(s.iloc[-1] - s.iloc[-2])
    return np.sign(d) if np.isfinite(d) else np.nan

def bb_divergence_signals(ax, price: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series,
                          lookback: int, conf_up: float, conf_dn: float, conf_level: float = 0.95):
    p = _coerce_1d_series(price).astype(float)
    up = _coerce_1d_series(bb_upper).reindex(p.index).astype(float)
    lo = _coerce_1d_series(bb_lower).reindex(p.index).astype(float)
    if p.dropna().shape[0] < max(3, lookback) or up.dropna().empty or lo.dropna().empty:
        return
    p = p.dropna().iloc[-lookback:]
    up = up.reindex(p.index).ffill().bfill()
    lo = lo.reindex(p.index).ffill().bfill()
    _, m_price = slope_line(p, lookback)
    _, m_upper = slope_line(up, lookback)
    _, m_lower = slope_line(lo, lookback)
    _, m_dist_buy  = slope_line(p - lo, lookback)
    _, m_dist_sell = slope_line(up - p, lookback)
    last_sign = _last_delta_sign(p)
    ts = p.index[-1]
    px = float(p.iloc[-1]) if len(p) else np.nan
    buy_cond  = (m_price > 0) and (m_lower < 0) and (m_dist_buy > 0) and (last_sign > 0) and (conf_up >= conf_level)
    sell_cond = (m_price < 0) and (m_upper > 0) and (m_dist_sell > 0) and (last_sign < 0) and (conf_dn >= conf_level)
    try:
        if buy_cond and np.isfinite(px):
            ax.scatter([ts], [px], marker="^", s=120, color="tab:green", zorder=9)
            ax.text(ts, px, f"  BB BUY {int(conf_level*100)}%", va="bottom", fontsize=9,
                    color="tab:green", fontweight="bold")
            st.success(
                f"**BB Divergence BUY** @ {fmt_price_val(px)} — trend↑ ({fmt_slope(m_price)}), "
                f"lowerBB↓ ({fmt_slope(m_lower)}), Δ(price−lower)↑ ({fmt_slope(m_dist_buy)}), P(up)≥{int(conf_level*100)}%"
            )
        if sell_cond and np.isfinite(px):
            ax.scatter([ts], [px], marker="v", s=120, color="tab:red", zorder=9)
            ax.text(ts, px, f"  BB SELL {int(conf_level*100)}%", va="top", fontsize=9,
                    color="tab:red", fontweight="bold")
            st.error(
                f"**BB Divergence SELL** @ {fmt_price_val(px)} — trend↓ ({fmt_slope(m_price)}), "
                f"upperBB↑ ({fmt_slope(m_upper)}), Δ(upper−price)↑ ({fmt_slope(m_dist_sell)}), P(down)≥{int(conf_level*100)}%"
            )
    except Exception:
        pass

# ========= NEW: DMI / ADX =========
def _rma(x: pd.Series, period: int) -> pd.Series:
    return x.ewm(alpha=1/period, adjust=False).mean()

def compute_dmi_adx(df: pd.DataFrame, period: int = 14):
    """
    Returns (+DI, -DI, ADX) as % values.
    Works on any OHLC dataframe with High/Low/Close.
    """
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty

    high = _coerce_1d_series(df['High']).astype(float)
    low  = _coerce_1d_series(df['Low']).astype(float)
    close= _coerce_1d_series(df['Close']).astype(float)

    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(df[['High','Low','Close']])
    tr_rma     = _rma(tr, period)
    plus_dm_rma  = _rma(plus_dm, period)
    minus_dm_rma = _rma(minus_dm, period)

    plus_di  = 100.0 * (plus_dm_rma / tr_rma).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (minus_dm_rma / tr_rma).replace([np.inf, -np.inf], np.nan)

    dx = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = _rma(dx, period)

    return plus_di.reindex(close.index), minus_di.reindex(close.index), adx.reindex(close.index)

def _adx_pass(series: pd.Series, thresh: float) -> (bool, float):
    last = _safe_last_float(series)
    return (np.isfinite(last) and last >= float(thresh)), last

# ========= Cached last values for scanning =========
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

# ---- Price > Kijun detection ----
def _price_above_kijun_from_df(df: pd.DataFrame, base: int = 26):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return False, None, np.nan, np.nan
    ohlc = df[['High','Low','Close']].copy()
    _, kijun, _, _, _ = ichimoku_lines(ohlc['High'], ohlc['Low'], ohlc['Close'], base=base)
    kijun = kijun.ffill().bfill().reindex(ohlc.index)
    close = ohlc['Close'].astype(float).reindex(ohlc.index)
    mask = close.notna() & kijun.notna()
    if mask.sum() < 1:
        return False, None, np.nan, np.nan
    c_now = float(close[mask].iloc[-1])
    k_now = float(kijun[mask].iloc[-1])
    ts = close[mask].index[-1]
    above = np.isfinite(c_now) and np.isfinite(k_now) and (c_now > k_now)
    return above, ts if above else None, c_now, k_now

@st.cache_data(ttl=120)
def price_above_kijun_info_daily(symbol: str, base: int = 26):
    try:
        df = fetch_hist_ohlc(symbol)
        return _price_above_kijun_from_df(df, base=base)
    except Exception:
        return False, None, np.nan, np.nan

@st.cache_data(ttl=120)
def price_above_kijun_info_hourly(symbol: str, period: str = "1d", base: int = 26):
    try:
        df = fetch_intraday(symbol, period=period)
        return _price_above_kijun_from_df(df, base=base)
    except Exception:
        return False, None, np.nan, np.nan

# --- Volume helpers ---
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
    vmax = float(np.nanmax(arr)); vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)


# ========= HMA Cross scanner helpers =========
def _last_hma_up_cross_info(price: pd.Series, hma: pd.Series, max_bars_since: int = 20):
    """
    Return the most recent upward price-vs-HMA cross.

    Upward cross:
      - previous close <= previous HMA
      - current close > current HMA
    """
    p = _coerce_1d_series(price).astype(float)
    h = _coerce_1d_series(hma).reindex(p.index).astype(float)
    ok = p.notna() & h.notna()
    if ok.sum() < 2:
        return None

    p = p[ok]
    h = h[ok]
    prev_p = p.shift(1)
    prev_h = h.shift(1)
    cross_up = (prev_p <= prev_h) & (p > h)
    cross_up = cross_up.fillna(False)

    if not cross_up.any():
        return None

    cross_indices = list(cross_up[cross_up].index)
    cross_time = cross_indices[-1]
    cross_pos = p.index.get_loc(cross_time)
    bars_since = (len(p) - 1) - int(cross_pos)

    try:
        max_bars = max(0, int(max_bars_since))
    except Exception:
        max_bars = 20

    if bars_since > max_bars:
        return None

    return {
        "cross_time": cross_time,
        "cross_bar_index": int(cross_pos),
        "bars_since": int(bars_since),
        "price_at_cross": float(p.loc[cross_time]),
        "hma_at_cross": float(h.loc[cross_time]),
        "current_price": float(p.iloc[-1]),
        "current_hma": float(h.iloc[-1]),
    }


def _support_reversal_before_or_near_cross(price: pd.Series,
                                           support: pd.Series,
                                           cross_time,
                                           lookback: int = 20,
                                           prox: float = 0.0025,
                                           confirm_bars: int = 2):
    """
    Confirm price recently reversed upward from the 30-support line before or
    near the HMA cross.

    A match requires:
      - a support touch within the lookback window ending at the HMA cross,
      - the touch happens before the cross/current confirmation point,
      - price is above the touched support after the reversal,
      - recent closes into the cross are rising for the confirmation window.
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    if p.empty or sup.empty or cross_time not in p.index:
        return None

    try:
        lb = max(3, int(lookback))
    except Exception:
        lb = 20
    try:
        cb = max(1, int(confirm_bars))
    except Exception:
        cb = 2
    try:
        px = max(0.0, float(prox))
    except Exception:
        px = 0.0025

    cross_pos = int(p.index.get_loc(cross_time))
    start = max(0, cross_pos - lb)
    end = cross_pos + 1

    win_p = p.iloc[start:end]
    win_sup = sup.iloc[start:end]
    if win_p.shape[0] < max(3, cb + 1):
        return None

    touch_mask = win_p <= win_sup * (1.0 + px)
    if not touch_mask.any():
        return None

    touch_times = list(touch_mask[touch_mask].index)
    touch_time = touch_times[-1]
    touch_pos_full = int(p.index.get_loc(touch_time))

    # The support touch must precede the cross; otherwise this is a touch, not a reversal.
    if touch_pos_full >= cross_pos:
        return None

    touch_price = float(p.loc[touch_time])
    touch_support = float(sup.loc[touch_time])
    cross_price = float(p.loc[cross_time])

    if not np.all(np.isfinite([touch_price, touch_support, cross_price])):
        return None

    recent_confirm = p.iloc[max(0, cross_pos - cb):cross_pos + 1]
    if recent_confirm.shape[0] < cb + 1:
        return None
    deltas = recent_confirm.diff().dropna()
    if deltas.empty or not (deltas > 0).all():
        return None

    channel_move = max(abs(touch_support) * 0.0005, 1e-9)
    if cross_price <= touch_price + channel_move:
        return None

    return {
        "support_touch_time": touch_time,
        "bars_from_touch_to_cross": int(cross_pos - touch_pos_full),
        "support_at_touch": touch_support,
        "price_at_touch": touch_price,
        "bounce_amount": float(cross_price - touch_price),
    }


@st.cache_data(ttl=120, show_spinner=False)
def hma_cross_info_daily(symbol: str,
                         hma_period_fixed: int = 55,
                         slope_lookback: int = 90,
                         recent_cross_bars: int = 20,
                         support_lookback: int = 30,
                         support_reversal_lookback: int = 20,
                         support_prox: float = 0.0025,
                         support_confirm_bars: int = 2):
    """
    Daily scanner row for symbols where:
      - trendline slope is upward,
      - price recently reversed upward from the 30-support line,
      - price recently crossed upward through HMA(55).
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None
        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(int(hma_period_fixed) + 5, int(slope_lookback) // 2, int(support_lookback) + 5):
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        if not np.isfinite(trend_slope) or trend_slope < 0:
            return None

        hma = compute_hma(close, period=int(hma_period_fixed))
        cross = _last_hma_up_cross_info(hma=hma, price=close, max_bars_since=int(recent_cross_bars))
        if cross is None:
            return None

        support = close.rolling(int(max(2, support_lookback)), min_periods=1).min()
        reversal = _support_reversal_before_or_near_cross(
            price=close,
            support=support,
            cross_time=cross["cross_time"],
            lookback=int(support_reversal_lookback),
            prox=float(support_prox),
            confirm_bars=int(support_confirm_bars),
        )
        if reversal is None:
            return None

        current_support = _safe_last_float(support)
        return {
            "Symbol": symbol,
            "Bars Since HMA Cross": int(cross["bars_since"]),
            "HMA Cross Time": cross["cross_time"],
            "Support Touch Time": reversal["support_touch_time"],
            "Bars Touch → Cross": int(reversal["bars_from_touch_to_cross"]),
            "Price at Cross": float(cross["price_at_cross"]),
            "HMA at Cross": float(cross["hma_at_cross"]),
            "Current Price": float(cross["current_price"]),
            "Current HMA": float(cross["current_hma"]),
            "Current 30 Support": float(current_support) if np.isfinite(current_support) else np.nan,
            "Support at Touch": float(reversal["support_at_touch"]),
            "Bounce from Touch": float(reversal["bounce_amount"]),
            "Trend Slope": float(trend_slope),
        }
    except Exception:
        return None


def _render_hma_cross_table(rows: list):
    if not rows:
        st.info("No symbols matched the HMA Cross setup.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Bars Since HMA Cross", "Symbol"], ascending=[True, True])

    for col in [
        "Price at Cross", "HMA at Cross", "Current Price", "Current HMA",
        "Current 30 Support", "Support at Touch", "Bounce from Touch", "Trend Slope"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["HMA Cross Time", "Support Touch Time"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)

# ========= Green Triangle Pick scanner helpers =========
def _last_true_signal_info(mask_like: pd.Series, max_bars_since: int = 20):
    """
    Return the latest True signal from a boolean mask if it is recent enough.
    Used to find the most recent green NTD-panel BUY/support-reversal triangle.
    """
    s = pd.Series(mask_like).fillna(False).astype(bool)
    if s.empty or not s.any():
        return None
    try:
        max_bars = max(0, int(max_bars_since))
    except Exception:
        max_bars = 20

    true_positions = np.flatnonzero(s.to_numpy(dtype=bool))
    if len(true_positions) == 0:
        return None

    pos = int(true_positions[-1])
    bars_since = (len(s) - 1) - pos
    if bars_since > max_bars:
        return None

    return {
        "signal_time": s.index[pos],
        "bar_index": pos,
        "bars_since": int(bars_since),
    }



def _recent_sr_signal_context(sri: pd.Series,
                              signal_mask: pd.Series,
                              max_bars_since: int = 12,
                              zone_threshold: float = -0.50,
                              lookback: int = 10):
    """
    Context for the latest S/R Reversal signal on the NTD panel.

    For BUY/support reversals, ok=True means the latest green triangle is recent
    and the S/R Reversal line recently turned up from the lower zone
    (default: at/below -0.50; also records whether it reached -0.75).
    """
    sig = _last_true_signal_info(signal_mask, max_bars_since=max_bars_since)
    result = {
        "ok": False,
        "signal": sig,
        "bars_since": np.nan,
        "signal_time": None,
        "value_at_signal": np.nan,
        "recent_extreme": np.nan,
        "from_below_minus_050": False,
        "from_below_minus_075": False,
        "zone_label": "No recent signal",
    }
    if sig is None:
        return result

    s = _coerce_1d_series(sri).replace([np.inf, -np.inf], np.nan)
    if s.empty:
        return result

    pos = int(sig.get("bar_index", len(s) - 1))
    pos = max(0, min(pos, len(s) - 1))
    try:
        lb = max(1, int(lookback))
    except Exception:
        lb = 10

    window = s.iloc[max(0, pos - lb):pos + 1].dropna()
    value_at_signal = _safe_last_float(pd.Series([s.iloc[pos]], index=[s.index[pos]]))
    recent_extreme = float(window.min()) if len(window) else np.nan

    below_050 = np.isfinite(recent_extreme) and recent_extreme <= -0.50
    below_075 = np.isfinite(recent_extreme) and recent_extreme <= -0.75
    try:
        th = float(zone_threshold)
    except Exception:
        th = -0.50

    if below_075:
        zone_label = "Turned up from below -0.75"
    elif below_050:
        zone_label = "Turned up from below -0.50"
    elif np.isfinite(value_at_signal) and value_at_signal < 0:
        zone_label = "Turned up below 0.00"
    elif np.isfinite(value_at_signal):
        zone_label = "Turned up above 0.00"
    else:
        zone_label = "Recent signal"

    result.update({
        "ok": bool(np.isfinite(recent_extreme) and recent_extreme <= th),
        "bars_since": int(sig["bars_since"]),
        "signal_time": sig["signal_time"],
        "value_at_signal": float(value_at_signal) if np.isfinite(value_at_signal) else np.nan,
        "recent_extreme": float(recent_extreme) if np.isfinite(recent_extreme) else np.nan,
        "from_below_minus_050": bool(below_050),
        "from_below_minus_075": bool(below_075),
        "zone_label": zone_label,
    })
    return result


def _trade_state_from_sr_context(price: pd.Series,
                                 support: pd.Series,
                                 resistance: pd.Series,
                                 trend_slope: float,
                                 ntd: pd.Series,
                                 sri: pd.Series,
                                 buy_context: dict,
                                 sell_context: dict = None,
                                 prox: float = 0.0025):
    """
    Classify the current chart into WAIT / BUY SETUP / BUY CONFIRMED /
    SELL SETUP / SELL CONFIRMED using price trend plus the NTD-panel
    S/R Reversal line.
    """
    sell_context = sell_context or {}
    px = _safe_last_float(price)
    sup = _safe_last_float(support)
    res = _safe_last_float(resistance)
    ntd_last = _safe_last_float(ntd)
    sri_last = _safe_last_float(sri)

    try:
        slope = float(trend_slope)
    except Exception:
        slope = np.nan

    try:
        p = max(0.0, float(prox))
    except Exception:
        p = 0.0025

    trend_up = np.isfinite(slope) and slope >= 0.0
    trend_down = np.isfinite(slope) and slope < 0.0
    near_support = np.all(np.isfinite([px, sup])) and px <= sup * (1.0 + p)
    near_resistance = np.all(np.isfinite([px, res])) and px >= res * (1.0 - p)

    if trend_up and bool(buy_context.get("ok", False)):
        state = "BUY CONFIRMED"
        detail = (
            f"Upward price trend and recent green NTD support-reversal triangle "
            f"({buy_context.get('zone_label', 'lower-zone turn')}; "
            f"{buy_context.get('bars_since', 'n/a')} bars ago)."
        )
        level = "success"
    elif trend_up and near_support and (not np.isfinite(sri_last) or sri_last <= 0.0):
        state = "BUY SETUP"
        detail = "Upward price trend and price is near support, but a fresh green NTD support-reversal triangle is still needed."
        level = "warning"
    elif trend_down and bool(sell_context.get("ok", False)):
        state = "SELL CONFIRMED"
        detail = (
            f"Downward price trend and recent red NTD resistance-reversal triangle "
            f"({sell_context.get('bars_since', 'n/a')} bars ago)."
        )
        level = "error"
    elif trend_down and near_resistance and (not np.isfinite(sri_last) or sri_last >= 0.0):
        state = "SELL SETUP"
        detail = "Downward price trend and price is near resistance, but a fresh red NTD resistance-reversal triangle is still needed."
        level = "warning"
    else:
        state = "WAIT"
        if trend_up:
            detail = "Price trend is upward, but NTD/S/R reversal confirmation is not aligned yet."
        elif trend_down:
            detail = "Price trend is downward, but price/S/R reversal confirmation is not aligned yet."
        else:
            detail = "Trend direction is unclear."
        level = "info"

    return {
        "state": state,
        "detail": detail,
        "level": level,
        "trend_direction": "Upward" if trend_up else ("Downward" if trend_down else "Unclear"),
        "near_support": bool(near_support),
        "near_resistance": bool(near_resistance),
        "ntd_last": float(ntd_last) if np.isfinite(ntd_last) else np.nan,
        "sri_last": float(sri_last) if np.isfinite(sri_last) else np.nan,
    }


def render_trade_state_banner(trade_state: dict):
    if not isinstance(trade_state, dict):
        return
    msg = (
        f"**Trade State: {trade_state.get('state', 'WAIT')}** — "
        f"{trade_state.get('detail', '')} "
        f"| Trend: {trade_state.get('trend_direction', 'n/a')} "
        f"| NTD {fmt_slope(trade_state.get('ntd_last', np.nan))} "
        f"| S/R Rev {fmt_slope(trade_state.get('sri_last', np.nan))}"
    )
    level = trade_state.get("level", "info")
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    elif level == "error":
        st.error(msg)
    else:
        st.info(msg)


def _green_triangle_context(price: pd.Series,
                            support: pd.Series,
                            resistance: pd.Series,
                            sri: pd.Series,
                            sig: dict) -> dict:
    """
    Extra scanner columns for green NTD support-reversal triangles.
    Shows where the S/R Reversal line was when the triangle printed and
    whether price remains above support.
    """
    p = _coerce_1d_series(price).astype(float)
    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    sr = _coerce_1d_series(sri).reindex(p.index)

    out = {
        "Triangle S/R Reversal": np.nan,
        "Triangle Zone": "n/a",
        "Triangle Below 0.00": False,
        "Triangle Below -0.50": False,
        "Triangle Below -0.75": False,
        "Recent S/R Low Before Triangle": np.nan,
        "Price Above Support": False,
        "Distance Above Support": np.nan,
        "Support at Triangle": np.nan,
        "Price at Triangle": np.nan,
    }
    if not isinstance(sig, dict) or p.empty:
        return out

    pos = int(sig.get("bar_index", len(p) - 1))
    pos = max(0, min(pos, len(p) - 1))
    ts = p.index[pos]

    tri_sri = _safe_last_float(pd.Series([sr.iloc[pos]], index=[ts])) if len(sr) else np.nan
    tri_price = _safe_last_float(pd.Series([p.iloc[pos]], index=[ts])) if len(p) else np.nan
    tri_sup = _safe_last_float(pd.Series([sup.iloc[pos]], index=[ts])) if len(sup) else np.nan

    sr_window = sr.iloc[max(0, pos - 10):pos + 1].dropna()
    recent_low = float(sr_window.min()) if len(sr_window) else np.nan

    last_price = _safe_last_float(p)
    last_support = _safe_last_float(sup)
    price_above_support = np.all(np.isfinite([last_price, last_support])) and last_price >= last_support
    dist_above_support = last_price - last_support if price_above_support else np.nan

    basis = recent_low if np.isfinite(recent_low) else tri_sri
    if np.isfinite(basis) and basis <= -0.75:
        zone = "Below -0.75"
    elif np.isfinite(basis) and basis <= -0.50:
        zone = "Below -0.50"
    elif np.isfinite(tri_sri) and tri_sri < 0:
        zone = "Below 0.00"
    elif np.isfinite(tri_sri):
        zone = "Above 0.00"
    else:
        zone = "n/a"

    out.update({
        "Triangle S/R Reversal": float(tri_sri) if np.isfinite(tri_sri) else np.nan,
        "Triangle Zone": zone,
        "Triangle Below 0.00": bool(np.isfinite(basis) and basis < 0.0),
        "Triangle Below -0.50": bool(np.isfinite(basis) and basis <= -0.50),
        "Triangle Below -0.75": bool(np.isfinite(basis) and basis <= -0.75),
        "Recent S/R Low Before Triangle": float(recent_low) if np.isfinite(recent_low) else np.nan,
        "Price Above Support": bool(price_above_support),
        "Distance Above Support": float(dist_above_support) if np.isfinite(dist_above_support) else np.nan,
        "Support at Triangle": float(tri_sup) if np.isfinite(tri_sup) else np.nan,
        "Price at Triangle": float(tri_price) if np.isfinite(tri_price) else np.nan,
    })
    return out


@st.cache_data(ttl=120, show_spinner=False)
def green_triangle_pick_info_daily(symbol: str,
                                   slope_lookback: int = 90,
                                   recent_bars: int = 20,
                                   support_window: int = 30,
                                   sr_prox: float = 0.0025,
                                   sr_zone: float = 0.80,
                                   sr_lookback: int = 8,
                                   sr_confirm: int = 3,
                                   sr_smooth: int = 8):
    """
    Daily scanner row for symbols where the price-chart trendline is upward
    and the NTD chart recently printed a green BUY/support-reversal triangle.
    """
    try:
        close = fetch_hist(symbol)
        if close is None or close.empty:
            return None

        close = _coerce_1d_series(close).dropna()
        if close.shape[0] < max(10, int(support_window) + 2, int(sr_lookback) + int(sr_confirm) + 3):
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        if not np.isfinite(trend_slope) or trend_slope < 0.0:
            return None

        win = max(2, int(support_window))
        support = close.rolling(win, min_periods=1).min()
        resistance = close.rolling(win, min_periods=1).max()

        buy_mask, _, sri = sr_reversal_opportunity_masks(
            price=close,
            support=support,
            resistance=resistance,
            trend_slope=float(trend_slope),
            prox=float(sr_prox),
            zone=float(sr_zone),
            lookback=int(sr_lookback),
            confirm_bars=int(sr_confirm),
            smooth_span=int(sr_smooth),
        )

        sig = _last_true_signal_info(buy_mask, max_bars_since=int(recent_bars))
        if sig is None:
            return None

        last_price = _safe_last_float(close)
        last_support = _safe_last_float(support)
        last_resistance = _safe_last_float(resistance)
        last_sri = _safe_last_float(sri)
        triangle_context = _green_triangle_context(close, support, resistance, sri, sig)

        return {
            "Symbol": symbol,
            "Chart": "Daily",
            "Bars Since Triangle": int(sig["bars_since"]),
            "Triangle Time": sig["signal_time"],
            "Trend Direction": "Upward",
            "Trend Slope": float(trend_slope),
            "Current Price": float(last_price) if np.isfinite(last_price) else np.nan,
            "Current Support": float(last_support) if np.isfinite(last_support) else np.nan,
            "Current Resistance": float(last_resistance) if np.isfinite(last_resistance) else np.nan,
            "Current S/R Reversal": float(last_sri) if np.isfinite(last_sri) else np.nan,
            **triangle_context,
        }
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def green_triangle_pick_info_hourly(symbol: str,
                                    period: str = "2d",
                                    slope_lookback: int = 120,
                                    recent_bars: int = 48,
                                    support_window: int = 60,
                                    sr_prox: float = 0.0025,
                                    sr_zone: float = 0.80,
                                    sr_lookback: int = 8,
                                    sr_confirm: int = 3,
                                    sr_smooth: int = 8):
    """
    Hourly/intraday scanner row for symbols where the intraday price-chart
    trendline is upward and the NTD chart recently printed a green
    BUY/support-reversal triangle.
    """
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = _coerce_1d_series(df["Close"]).ffill().dropna()
        if close.shape[0] < max(10, int(support_window) + 2, int(sr_lookback) + int(sr_confirm) + 3):
            return None

        _, trend_slope = slope_line(close, int(slope_lookback))
        if not np.isfinite(trend_slope) or trend_slope < 0.0:
            return None

        win = max(2, int(support_window))
        support = close.rolling(win, min_periods=1).min()
        resistance = close.rolling(win, min_periods=1).max()

        buy_mask, _, sri = sr_reversal_opportunity_masks(
            price=close,
            support=support,
            resistance=resistance,
            trend_slope=float(trend_slope),
            prox=float(sr_prox),
            zone=float(sr_zone),
            lookback=int(sr_lookback),
            confirm_bars=int(sr_confirm),
            smooth_span=int(sr_smooth),
        )

        sig = _last_true_signal_info(buy_mask, max_bars_since=int(recent_bars))
        if sig is None:
            return None

        last_price = _safe_last_float(close)
        last_support = _safe_last_float(support)
        last_resistance = _safe_last_float(resistance)
        last_sri = _safe_last_float(sri)
        triangle_context = _green_triangle_context(close, support, resistance, sri, sig)

        return {
            "Symbol": symbol,
            "Chart": "Hourly",
            "Bars Since Triangle": int(sig["bars_since"]),
            "Triangle Time": sig["signal_time"],
            "Trend Direction": "Upward",
            "Trend Slope": float(trend_slope),
            "Current Price": float(last_price) if np.isfinite(last_price) else np.nan,
            "Current Support": float(last_support) if np.isfinite(last_support) else np.nan,
            "Current Resistance": float(last_resistance) if np.isfinite(last_resistance) else np.nan,
            "Current S/R Reversal": float(last_sri) if np.isfinite(last_sri) else np.nan,
            **triangle_context,
        }
    except Exception:
        return None


def _render_green_triangle_pick_table(title: str, rows: list):
    st.subheader(title)
    if not rows:
        st.info("No recent green BUY/support-reversal triangles found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["Bars Since Triangle", "Symbol"], ascending=[True, True])

    for col in [
        "Trend Slope", "Current Price", "Current Support",
        "Current Resistance", "Current S/R Reversal",
        "Triangle S/R Reversal", "Recent S/R Low Before Triangle",
        "Distance Above Support", "Support at Triangle", "Price at Triangle"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Triangle Time" in df.columns:
        df["Triangle Time"] = df["Triangle Time"].astype(str)

    st.dataframe(df.reset_index(drop=True), use_container_width=True, hide_index=True)

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.5 Scanner",
    "Long-Term History",
    "S/R Reversal Crosses",
    "-0.75 SR Crossers",
    "HMA Cross",
    "Green Triangle Pick",
    "S/R Cross",
    "S/R -0.5 Cross"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    auto_run = st.session_state.run_all

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
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
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc

        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            # NTD + smoothed NPX overlay
            ntd_d = compute_normalized_trend(df, window=ntd_window)
            npx_d_raw = compute_normalized_price(df, window=ntd_window)
            npx_d_full = smooth_npx(npx_d_raw, span=npx_smooth_span) if show_npx_ntd else pd.Series(index=df.index, dtype=float)

            # Ichimoku Kijun price-chart line removed by request.
            kijun_d = pd.Series(index=df.index, dtype=float)

            # BBands (Daily)
            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            # 🔶 NEW: ADX (Daily)
            plus_d, minus_d, adx_d = compute_dmi_adx(df_ohlc, period=adx_period) if df_ohlc is not None else (pd.Series(dtype=float),)*3
            adx_d_show = adx_d.reindex(df.index)
            adx_ok_d, adx_last_d = _adx_pass(adx_d_show, adx_min) if use_adx_filter else (True, _safe_last_float(adx_d_show))

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            yhat_ema_show = yhat_ema30.reindex(df_show.index) if not yhat_ema30.empty else yhat_ema30
            ntd_d_show  = ntd_d.reindex(df_show.index)
            npx_d_show  = npx_d_full.reindex(df_show.index)
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show= bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)
            hma_d_full = compute_hma(df, period=hma_period)
            hma_d_show = hma_d_full.reindex(df_show.index)
            psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[(psar_d_df.index >= x0) & (psar_d_df.index <= x1)]

            fig, (ax, axdw) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), gridspec_kw={"height_ratios": [3.2, 1.3]})
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            ax.set_title(f"{sel} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots  |  ADX {adx_last_d:.1f}")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")
            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)
                try:
                    last_pct = _safe_last_float(bb_pctb_d_show.dropna())
                    last_nbb = _safe_last_float(bb_nbb_d_show.dropna())
                    ax.text(0.99, 0.02, f"NBB {last_nbb:+.2f}  |  %B {fmt_pct(last_pct, digits=0)}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=9, color="black",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                except Exception:
                    pass
            if show_psar and not psar_d_df.empty:
                up_mask = psar_d_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d_df.index[up_mask], psar_d_df["PSAR"][up_mask],
                               s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask], psar_d_df["PSAR"][dn_mask],
                               s=15, color="tab:red", zorder=6)
            if not yhat_d_show.empty:
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2, color=trend_color_for_slope(m_d), label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema_show.empty:
                ax.plot(yhat_ema_show.index, yhat_ema_show.values, "-", linewidth=2, color=trend_color_for_slope(m_ema30), label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")
            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")
            if show_fibs and len(df_show) > 0:
                fibs_d = fibonacci_levels(df_show)
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1],
                              linestyles="dotted", linewidth=1.0, alpha=0.9)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" Fib {lbl}", va="center", fontsize=8)
            if len(res30_show) and len(sup30_show):
                r30_last = _safe_last_float(res30_show); s30_last = _safe_last_float(sup30_show)
                ax.text(df_show.index[-1], r30_last, f"  30R = {fmt_price_val(r30_last)}", va="bottom")
                ax.text(df_show.index[-1], s30_last, f"  30S = {fmt_price_val(s30_last)}", va="top")

            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)

            # HMA crossover (Daily) — ADX gate
            if show_hma and not hma_d_show.dropna().empty:
                cross_d = detect_last_crossover(df_show, hma_d_show)
                if cross_d is not None and cross_d["time"] is not None and (not use_adx_filter or adx_ok_d):
                    ts = cross_d["time"]; px_here = _safe_last_float(pd.Series([df_show.loc[ts]]))
                    if cross_d["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "BUY", hma_conf)
                        st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)} | ADX {adx_last_d:.1f}")
                    elif cross_d["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                        annotate_crossover(ax, ts, px_here, "SELL", hma_conf)
                        st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)} | ADX {adx_last_d:.1f}")
                elif cross_d is not None and cross_d["time"] is not None and use_adx_filter and not adx_ok_d:
                    st.info(f"HMA signal gated off: ADX {adx_last_d:.1f} < {adx_min}")

            # BB Divergence (Daily) — ADX gate
            if show_bb_div and (not use_adx_filter or adx_ok_d):
                bb_divergence_signals(ax, df_show, bb_up_d_show, bb_lo_d_show,
                                      lookback=slope_lb_daily, conf_up=p_up, conf_dn=p_dn, conf_level=bb_conf)
            elif show_bb_div and use_adx_filter and not adx_ok_d:
                st.info(f"BB Divergence gated off: ADX {adx_last_d:.1f} < {adx_min}")

            # --- DAILY INDICATOR PANEL ---
            axdw.set_title("Daily Indicator Panel — NTD + Smoothed NPX + S/R Reversal + Trend")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(ntd_d_show.index, ntd_d_show, "-", linewidth=1.6, label=f"NTD (win={ntd_window})")
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(ntd_trend_d.index, ntd_trend_d.values, "--", linewidth=2,
                              label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)")

            if show_npx_ntd and not npx_d_show.dropna().empty and not ntd_d_show.dropna().empty:
                overlay_npx_on_ntd(axdw, npx_d_show, ntd_d_show, mark_crosses=mark_npx_cross)
            if show_sr_reversal_ntd and not df_show.dropna().empty and not ntd_d_show.dropna().empty:
                overlay_sr_reversal_indicator_on_ntd(
                    axdw,
                    price=df_show,
                    support=sup30_show,
                    resistance=res30_show,
                    trend_slope=m_d,
                    ntd=ntd_d_show,
                    prox=sr_prox_pct,
                    zone=sr_rev_zone,
                    lookback=sr_rev_lookback,
                    confirm_bars=sr_rev_confirm,
                    smooth_span=sr_rev_smooth,
                )
            if show_hma_rev_ntd and not hma_d_show.dropna().empty and not df_show.dropna().empty:
                overlay_hma_reversal_on_ntd(axdw, df_show, hma_d_show, lookback=hma_rev_lb, period=hma_period)
            axdw.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
            axdw.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
            axdw.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
            axdw.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
            axdw.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")
            axdw.set_ylim(-1.1, 1.1); axdw.set_xlabel("Date (PST)"); axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly -----
        if chart in ("Hourly","Both"):
            intraday = st.session_state.intraday
            if intraday is None or intraday.empty or "Close" not in intraday:
                st.warning("No intraday data available.")
            else:
                hc = intraday["Close"].ffill()
                he = hc.ewm(span=20).mean()
                xh = np.arange(len(hc))
                slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
                trend_h = slope_h * xh + intercept_h
                res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
                sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

                # Supertrend and Ichimoku Kijun price-chart lines removed by request.
                st_line_intr = pd.Series(index=hc.index, dtype=float)
                kj_h = pd.Series(index=hc.index, dtype=float)

                bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
                hma_h = compute_hma(hc, period=hma_period)
                psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                psar_h_df = psar_h_df.reindex(hc.index)

                # NTD + smoothed NPX for the hourly indicator panel
                ntd_h = compute_normalized_trend(hc, window=ntd_window)
                npx_h_raw = compute_normalized_price(hc, window=ntd_window)
                npx_h = smooth_npx(npx_h_raw, span=npx_smooth_span) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)

                yhat_h, m_h = slope_line(hc, slope_lb_hourly)
                r2_h = regression_r2(hc, slope_lb_hourly)

                # 🔶 NEW: ADX (Hourly)
                plus_h, minus_h, adx_h = compute_dmi_adx(intraday, period=adx_period)
                adx_ok_h, adx_last_h = _adx_pass(adx_h.reindex(hc.index), adx_min) if use_adx_filter else (True, _safe_last_float(adx_h))

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", color=trend_color_for_slope(slope_h), label=f"Trend (m={fmt_slope(slope_h)}/bar)", linewidth=2)
                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
                if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
                    ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                    ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                    ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
                    ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)
                if show_psar and not psar_h_df.dropna().empty:
                    up_mask = psar_h_df["in_uptrend"] == True
                    dn_mask = ~up_mask
                    if up_mask.any():
                        ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask],
                                    s=15, color="tab:green", zorder=6, label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                    if dn_mask.any():
                        ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = _safe_last_float(res_h); sup_val = _safe_last_float(sup_h); px_val  = _safe_last_float(hc)
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                trade_buy_mask = pd.Series(False, index=hc.index)
                trade_sell_mask = pd.Series(False, index=hc.index)
                trade_sri = pd.Series(index=hc.index, dtype=float)
                trade_buy_context = {"ok": False}
                trade_sell_context = {"ok": False}
                try:
                    trade_buy_mask, trade_sell_mask, trade_sri = sr_reversal_opportunity_masks(
                        price=hc,
                        support=sup_h,
                        resistance=res_h,
                        trend_slope=float(m_h),
                        prox=float(sr_prox_pct),
                        zone=float(sr_rev_zone),
                        lookback=int(sr_rev_lookback),
                        confirm_bars=int(sr_rev_confirm),
                        smooth_span=int(sr_rev_smooth),
                    )
                    trade_buy_context = _recent_sr_signal_context(
                        trade_sri,
                        trade_buy_mask,
                        max_bars_since=max(6, int(sr_rev_lookback) * 2),
                        zone_threshold=-0.50,
                        lookback=max(6, int(sr_rev_lookback)),
                    )
                    trade_sell_context = _recent_sr_signal_context(
                        -trade_sri,
                        trade_sell_mask,
                        max_bars_since=max(6, int(sr_rev_lookback) * 2),
                        zone_threshold=-0.50,
                        lookback=max(6, int(sr_rev_lookback)),
                    )
                    trade_state = _trade_state_from_sr_context(
                        price=hc,
                        support=sup_h,
                        resistance=res_h,
                        trend_slope=float(m_h),
                        ntd=ntd_h,
                        sri=trade_sri,
                        buy_context=trade_buy_context,
                        sell_context=trade_sell_context,
                        prox=float(sr_prox_pct),
                    )
                    render_trade_state_banner(trade_state)
                except Exception:
                    trade_state = {"state": "WAIT", "detail": "Trade-state calculation unavailable.", "level": "info"}

                instr_txt = format_trade_instruction(slope_h, sup_val, res_val, px_val, sel)
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)} — {instr_txt}  |  ADX {adx_last_h:.1f}")

                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = _safe_last_float(bb_pctb_h.dropna()) if show_bbands else np.nan
                        last_nbb = _safe_last_float(bb_nbb_h.dropna()) if show_bbands else np.nan
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                    except Exception:
                        pass
                    ax2.text(0.99, 0.02, f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                             transform=ax2.transAxes, ha="right", va="bottom",
                             fontsize=11, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not yhat_h.empty:
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, color=trend_color_for_slope(m_h), label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                ax2.text(0.01, 0.02, f"Slope: {fmt_slope(slope_h)}/bar",
                         transform=ax2.transAxes, ha="left", va="bottom", fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                ax2.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
                         transform=ax2.transAxes, ha="center", va="bottom", fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                # Sessions
                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess = compute_session_lines(hc.index); draw_session_lines(ax2, sess)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1],
                                   linestyles="dotted", linewidth=1.0, alpha=0.9)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" Fib {lbl}", va="center", fontsize=8)

                if mode == "Forex" and show_fx_news and not hc.empty:
                    fx_news = fetch_yf_news(sel, window_days=news_window_days)
                    if not fx_news.empty:
                        t0, t1 = hc.index[0], hc.index[-1]
                        times = [t for t in fx_news["time"] if t0 <= t <= t1]
                        if times: draw_news_markers(ax2, times, float(hc.min()), float(hc.max()), label="News")

                # ===== Near S/R signal — ADX gate =====
                signal = sr_proximity_signal(hc, res_h, sup_h, st.session_state.fc_vals,
                                             threshold=signal_threshold, prox=sr_prox_pct)
                if signal is not None and np.isfinite(px_val):
                    if not use_adx_filter or adx_ok_h:
                        pips_txt = _diff_text(sup_val, res_val, sel) if np.isfinite(sup_val) and np.isfinite(res_val) else ""
                        try:
                            green_trend_up = float(slope_h) >= 0.0
                        except Exception:
                            green_trend_up = False

                        if green_trend_up:
                            close_price = res_val
                            close_label = "Sell Price"
                            conf_tag = f"↑{fmt_pct(p_up)}" if np.isfinite(p_up) else "↑n/a"
                        else:
                            close_price = sup_val
                            close_label = "Buy Price"
                            conf_tag = f"↓{fmt_pct(p_dn)}" if np.isfinite(p_dn) else "↓n/a"

                        if signal["side"] == "BUY":
                            near_txt = f"Near support {fmt_price_val(sup_val)}"
                            buy_alert_confirmed = bool(green_trend_up and trade_buy_context.get("ok", False))
                            if buy_alert_confirmed:
                                st.success(
                                    f"**Profit Alert:** **CLOSE** @ {fmt_price_val(close_price)} — "
                                    f"{close_label} {fmt_price_val(close_price)}; {near_txt} with {conf_tag} "
                                    f"• {pips_txt} | ADX {adx_last_h:.1f} "
                                    f"| NTD S/R turn: {trade_buy_context.get('zone_label', 'confirmed')}"
                                )
                            else:
                                st.info(
                                    f"**BUY SETUP / WAIT:** {near_txt}, but Profit Alert is gated until "
                                    f"the price trendline is upward **and** the NTD S/R Reversal line prints "
                                    f"a recent green support-reversal triangle from below -0.50/-0.75."
                                )
                        else:
                            near_txt = f"Near resistance {fmt_price_val(res_val)}"
                            st.error(
                                f"**CLOSE** @ {fmt_price_val(close_price)} — "
                                f"{close_label} {fmt_price_val(close_price)}; {near_txt} with {conf_tag} "
                                f"• {pips_txt} | ADX {adx_last_h:.1f}"
                            )
                    else:
                        st.info(f"Near S/R signal gated off: ADX {adx_last_h:.1f} < {adx_min}")

                # HMA crossover — ADX gate
                if show_hma and not hma_h.dropna().empty:
                    cross_h = detect_last_crossover(hc, hma_h)
                    if cross_h is not None and cross_h["time"] is not None and (not use_adx_filter or adx_ok_h):
                        ts = cross_h["time"]; px_here = _safe_last_float(pd.Series([hc.loc[ts]]))
                        if cross_h["side"] == "BUY" and np.isfinite(p_up) and p_up >= hma_conf:
                            annotate_crossover(ax2, ts, px_here, "BUY", hma_conf)
                            st.success(f"**HMA BUY** @ {fmt_price_val(px_here)} — P(up)={fmt_pct(p_up)} ≥ {fmt_pct(hma_conf)} | ADX {adx_last_h:.1f}")
                        elif cross_h["side"] == "SELL" and np.isfinite(p_dn) and p_dn >= hma_conf:
                            annotate_crossover(ax2, ts, px_here, "SELL", hma_conf)
                            st.error(f"**HMA SELL** @ {fmt_price_val(px_here)} — P(down)={fmt_pct(p_dn)} ≥ {fmt_pct(hma_conf)} | ADX {adx_last_h:.1f}")
                    elif cross_h is not None and cross_h["time"] is not None and use_adx_filter and not adx_ok_h:
                        st.info(f"HMA signal gated off: ADX {adx_last_h:.1f} < {adx_min}")

                # BB Divergence — ADX gate
                if show_bb_div and (not use_adx_filter or adx_ok_h):
                    bb_divergence_signals(ax2, hc, bb_up_h, bb_lo_h,
                                          lookback=slope_lb_hourly, conf_up=p_up, conf_dn=p_dn, conf_level=bb_conf)
                elif show_bb_div and use_adx_filter and not adx_ok_h:
                    st.info(f"BB Divergence gated off: ADX {adx_last_h:.1f} < {adx_min}")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                xlim_price = ax2.get_xlim()
                st.pyplot(fig2)

                # === Hourly Volume panel ===
                vol = _coerce_1d_series(intraday.get("Volume", pd.Series(index=hc.index))).reindex(hc.index).astype(float)
                if _has_volume_to_plot(vol):
                    v_mid = rolling_midline(vol, window=max(3, int(slope_lb_hourly)))
                    v_trend, v_m = slope_line(vol, slope_lb_hourly)
                    v_r2 = regression_r2(vol, slope_lb_hourly)
                    fig2v, ax2v = plt.subplots(figsize=(14, 2.8))
                    ax2v.set_title(f"Volume (Hourly) — Mid-line & Trend  |  Slope={fmt_slope(v_m)}/bar")
                    ax2v.fill_between(vol.index, 0, vol, alpha=0.18, label="Volume", color="tab:blue")
                    ax2v.plot(vol.index, vol, linewidth=1.0, color="tab:blue")
                    ax2v.plot(v_mid.index, v_mid, ":", linewidth=1.6, label=f"Mid-line ({slope_lb_hourly}-roll)")
                    if v_mid.notna().any():
                        last_mid = _safe_last_float(v_mid.dropna())
                        ax2v.hlines(last_mid, xmin=vol.index[0], xmax=vol.index[-1], linestyles="dotted", linewidth=1.0)
                        label_on_left(ax2v, last_mid, f"Mid {last_mid:,.0f}", color="black")
                    if not v_trend.empty:
                        ax2v.plot(v_trend.index, v_trend.values, "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(v_m)}/bar)")
                    ax2v.text(0.01, 0.02, f"Slope: {fmt_slope(v_m)}/bar",
                              transform=ax2v.transAxes, ha="left", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                    ax2v.text(0.50, 0.02, f"R² ({slope_lb_hourly} bars): {fmt_r2(v_r2)}",
                              transform=ax2v.transAxes, ha="center", va="bottom",
                              fontsize=9, color="black",
                              bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
                    ax2v.set_xlim(xlim_price)
                    ax2v.set_xlabel("Time (PST)")
                    ax2v.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2v)

                # === Hourly Indicator Panel: NTD + Smoothed NPX + S↔R channel ===
                if show_nrsi:
                    ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
                    npx_h = npx_h.reindex(hc.index) if show_npx_ntd else pd.Series(index=hc.index, dtype=float)
                    fig2r, ax2r = plt.subplots(figsize=(14,2.8))
                    ax2r.set_title(f"Hourly Indicator Panel — NTD + Smoothed NPX + S/R Reversal + Trend (win={ntd_window})")
                    if shade_ntd and not ntd_h.dropna().empty:
                        shade_ntd_regions(ax2r, ntd_h)
                    if show_ntd_channel and np.isfinite(res_val) and np.isfinite(sup_val):
                        overlay_inrange_on_ntd(ax2r, hc, sup_h, res_h)
                    ax2r.plot(ntd_h.index, ntd_h, "-", linewidth=1.6, label="NTD")
                    if show_npx_ntd and not npx_h.dropna().empty and not ntd_h.dropna().empty:
                        overlay_npx_on_ntd(ax2r, npx_h, ntd_h, mark_crosses=mark_npx_cross)
                    if show_sr_reversal_ntd and not hc.dropna().empty and not ntd_h.dropna().empty:
                        overlay_sr_reversal_indicator_on_ntd(
                            ax2r,
                            price=hc,
                            support=sup_h,
                            resistance=res_h,
                            trend_slope=m_h,
                            ntd=ntd_h,
                            prox=sr_prox_pct,
                            zone=sr_rev_zone,
                            lookback=sr_rev_lookback,
                            confirm_bars=sr_rev_confirm,
                            smooth_span=sr_rev_smooth,
                        )
                    if not ntd_trend_h.empty:
                        ax2r.plot(ntd_trend_h.index, ntd_trend_h.values, "--", linewidth=2,
                                  label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)")
                    if show_hma_rev_ntd and not hma_h.dropna().empty and not hc.dropna().empty:
                        overlay_hma_reversal_on_ntd(ax2r, hc, hma_h, lookback=hma_rev_lb, period=hma_period)

                    ax2r.axhline(0.0,  linestyle="--", linewidth=1.0, color="black",    label="0.00")
                    ax2r.axhline(0.5,  linestyle="-",  linewidth=1.2, color="black",    label="+0.50")
                    ax2r.axhline(-0.5, linestyle="-",  linewidth=1.2, color="black",    label="-0.50")
                    ax2r.axhline(0.75, linestyle="-",  linewidth=3.0, color="tab:green", label="+0.75")
                    ax2r.axhline(-0.75, linestyle="-", linewidth=3.0, color="tab:red",   label="-0.75")
                    ax2r.set_ylim(-1.1, 1.1); ax2r.set_xlim(xlim_price)
                    ax2r.legend(loc="lower left", framealpha=0.5)
                    ax2r.set_xlabel("Time (PST)")
                    st.pyplot(fig2r)

        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
# (UNCHANGED)

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange'] > 0
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last_price = _safe_last_float(df_hist)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        x3m = np.arange(len(df3m))
        slope3m, intercept3m = np.polyfit(x3m, df3m.values, 1)
        trend3m = slope3m * x3m + intercept3m

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", label="Support")
        ax.plot(df3m.index, trend3m, "--", color=trend_color_for_slope(slope3m), label="Trend")
        ax.set_xlabel("Date (PST)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        x0 = np.arange(len(df0))
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0 * x0 + intercept0
        res0 = df0.rolling(30, min_periods=1).max()
        sup0 = df0.rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(res0, ":", label="Resistance")
        ax0.plot(sup0, ":", label="Support")
        ax0.plot(df0.index, trend0, "--", label="Trend")
        ax0.set_xlabel("Date (PST)")
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({"Type": ["Bull", "Bear"], "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]}).set_index("Type")
        st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.5 Scanner (unchanged, uses helpers above) ---
with tab5:
    st.header("NTD -0.5 Scanner")
    st.caption("Shows **symbols with NTD < -0.5** (Daily for Stocks & FX; Hourly for FX). Also lists **Price > Ichimoku Kijun(26)**.")
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox(
        "Hourly lookback for Forex (for Hourly scan below):",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
        key="ntd_scan_hour_range"
    )
    scan_period = period_map[scan_hour_range]
    c1, c2 = st.columns(2)
    with c1:
        thresh = st.slider("NTD threshold", -1.0, 0.0, -0.5, 0.05, key="ntd_thresh")
    with c2:
        run = st.button("Scan Universe", key="btn_ntd_scan")

    if run:
        daily_rows = []
        for sym in universe:
            ntd_val, ts = last_daily_ntd_value(sym, ntd_window)
            daily_rows.append({"Symbol": sym, "NTD_Daily": ntd_val, "Timestamp": ts})
        df_daily = pd.DataFrame(daily_rows)
        below_daily = df_daily[np.isfinite(df_daily["NTD_Daily"]) & (df_daily["NTD_Daily"] < thresh)].sort_values("NTD_Daily")
        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(f"Daily NTD < {thresh:+.2f}", int(below_daily.shape[0]))
        st.subheader(f"Daily — NTD < {thresh:+.2f}")
        if below_daily.empty:
            st.info(f"No symbols with Daily NTD < {thresh:+.2f}.")
        else:
            show = below_daily.copy()
            show["NTD_Daily"] = show["NTD_Daily"].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "n/a")
            st.dataframe(show.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.subheader(f"Daily — **Price > Ichimoku Kijun({ichi_base})** (latest bar)")
        above_rows = []
        for sym in universe:
            above, ts, close_now, kij_now = price_above_kijun_info_daily(sym, base=ichi_base)
            above_rows.append({"Symbol": sym, "AboveNow": above, "Timestamp": ts, "Close": close_now, "Kijun": kij_now})
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]
        c7, c8 = st.columns(2)
        c7.metric("Daily Price > Kijun", int(df_above_daily.shape[0]))
        c8.caption("Latest close strictly greater than current Kijun value.")
        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            view_above = df_above_daily.copy()
            view_above["Close"] = view_above["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_above["Kijun"] = view_above["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view_above[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — NTD < {thresh:+.2f}  ({scan_hour_range} lookback)")
            hourly_rows = []
            for sym in universe:
                v, ts = last_hourly_ntd_value(sym, ntd_window, period=scan_period)
                hourly_rows.append({"Symbol": sym, "NTD_Hourly": v, "Timestamp": ts})
            df_hour = pd.DataFrame(hourly_rows)
            below_hour = df_hour[np.isfinite(df_hour["NTD_Hourly"]) & (df_hour["NTD_Hourly"] < thresh)].sort_values("NTD_Hourly")
            c5, c6 = st.columns(2)
            c5.metric("FX Pairs Scanned", len(universe))
            c6.metric(f"Hourly NTD < {thresh:+.2f}", int(below_hour.shape[0]))
            if below_hour.empty:
                st.info(f"No Forex pairs with Hourly NTD < {thresh:+.2f}.")
            else:
                showh = below_hour.copy()
                showh["NTD_Hourly"] = showh["NTD_Hourly"].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "n/a")
                st.dataframe(showh.reset_index(drop=True), use_container_width=True)

            st.subheader(f"Forex Hourly — **Price > Ichimoku Kijun({ichi_base})** (latest bar, {scan_hour_range})")
            habove_rows = []
            for sym in universe:
                above_h, ts_h, close_h, kij_h = price_above_kijun_info_hourly(sym, period=scan_period, base=ichi_base)
                habove_rows.append({"Symbol": sym, "AboveNow": above_h, "Timestamp": ts_h, "Close": close_h, "Kijun": kij_h})
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]
            c9, c10 = st.columns(2)
            c9.metric("Hourly Price > Kijun", int(df_above_hour.shape[0]))
            c10.caption("Latest intraday close strictly greater than current Kijun value.")
            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                vch["Kijun"] = vch["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(vch[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

# --- Tab 6: Long-Term History (unchanged) ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox("Ticker:", universe, index=default_idx, key="hist_long_ticker")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("5Y", key="btn_5y"):  st.session_state.hist_years = 5
    if c2.button("10Y", key="btn_10y"): st.session_state.hist_years = 10
    if c3.button("15Y", key="btn_15y"): st.session_state.hist_years = 15
    if c4.button("20Y", key="btn_20y"): st.session_state.hist_years = 20
    years = int(st.session_state.hist_years)
    st.caption(f"Showing last **{years} years**. Support/Resistance = rolling **252-day** extremes; trendline fits the shown window.")
    s_full = fetch_hist_max(sym)
    if s_full is None or s_full.empty:
        st.warning("No historical data available.")
    else:
        end_ts = s_full.index.max()
        start_ts = end_ts - pd.DateOffset(years=years)
        s = s_full[s_full.index >= start_ts]
        if s.empty:
            st.warning(f"No data in the last {years} years for {sym}.")
        else:
            res_roll = s.rolling(252, min_periods=1).max()
            sup_roll = s.rolling(252, min_periods=1).min()
            res_last = _safe_last_float(res_roll) if len(res_roll) else np.nan
            sup_last = _safe_last_float(sup_roll) if len(sup_roll) else np.nan
            yhat_all, m_all = slope_line(s, lookback=len(s))
            fig, ax = plt.subplots(figsize=(14,5))
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close")
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance (252d)")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support (252d)")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")
            if not yhat_all.empty:
                ax.plot(yhat_all.index, yhat_all.values, "--", linewidth=2, color=trend_color_for_slope(m_all), label=f"Trend (m={fmt_slope(m_all)}/bar)")
                ax.text(0.01, 0.02, f"Slope: {fmt_slope(m_all)}/bar",
                        transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(0.99, 0.02, f"Current price: {fmt_price_val(px_now)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.set_xlabel("Date (PST)"); ax.set_ylabel("Price"); ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

# --- Tab 7: S/R Reversal zero-line cross scanner ---
with tab7:
    st.header("S/R Reversal Crosses")
    st.caption(
        "Lists symbols where the **S/R Reversal line** on the NTD panel has recently crossed the **0.0** line. "
        "Daily and Hourly results are separated, with Upward crosses shown above Downward crosses. "
        "A third Daily table also shows symbols that were below **-0.75**, crossed back above **-0.75**, and are still rising."
    )

    period_map_sr = {"24h": "1d", "48h": "2d", "96h": "4d"}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sr_cross_daily_recent = st.slider(
            "Daily recent window (bars)",
            1, 60, 20, 1,
            key="sr_cross_daily_recent"
        )
    with c2:
        sr_cross_hourly_recent = st.slider(
            "Hourly recent window (bars)",
            1, 120, 36, 1,
            key="sr_cross_hourly_recent"
        )
    with c3:
        sr_cross_hour_range = st.selectbox(
            "Hourly lookback:",
            ["24h", "48h", "96h"],
            index=0,
            key="sr_cross_hour_range"
        )
    with c4:
        sr_cross_smooth = st.slider(
            "S/R line smoothing",
            1, 30, int(sr_rev_smooth), 1,
            key="sr_cross_smooth"
        )

    run_sr_cross_scan = st.button("Scan S/R Reversal Crosses", key="btn_sr_cross_scan")

    if run_sr_cross_scan:
        scan_period = period_map_sr[sr_cross_hour_range]
        daily_up_rows = []
        daily_down_rows = []
        daily_minus075_up_rows = []
        hourly_up_rows = []
        hourly_down_rows = []

        progress = st.progress(0, text="Scanning S/R Reversal crosses...")
        total_steps = max(1, len(universe) * 2)
        step_count = 0

        for sym in universe:
            row = sr_reversal_cross_info_daily(
                sym,
                smooth_span=int(sr_cross_smooth),
                recent_bars=int(sr_cross_daily_recent),
            )
            if row is not None:
                if row.get("Direction") == "Upward":
                    daily_up_rows.append(row)
                elif row.get("Direction") == "Downward":
                    daily_down_rows.append(row)

            minus075_row = sr_reversal_minus075_up_cross_info_daily(
                sym,
                smooth_span=int(sr_cross_smooth),
                recent_bars=int(sr_cross_daily_recent),
                confirm_bars=2,
                threshold=-0.75,
                slope_lookback=int(slope_lb_daily),
            )
            if minus075_row is not None:
                daily_minus075_up_rows.append(minus075_row)

            step_count += 1
            progress.progress(min(1.0, step_count / total_steps), text=f"Scanning daily: {sym}")

            row = sr_reversal_cross_info_hourly(
                sym,
                period=scan_period,
                sr_window=int(sr_lb_hourly),
                smooth_span=int(sr_cross_smooth),
                recent_bars=int(sr_cross_hourly_recent),
            )
            if row is not None:
                if row.get("Direction") == "Upward":
                    hourly_up_rows.append(row)
                elif row.get("Direction") == "Downward":
                    hourly_down_rows.append(row)

            step_count += 1
            progress.progress(min(1.0, step_count / total_steps), text=f"Scanning hourly: {sym}")

        progress.empty()

        st.markdown("### Daily")
        _render_sr_cross_table("Daily — Upward 0.0 crosses", daily_up_rows)
        _render_sr_cross_table("Daily — Downward 0.0 crosses", daily_down_rows)
        _render_minus075_up_cross_table("Daily — Reversed from below -0.75 and crossed upward", daily_minus075_up_rows)

        st.markdown("### Hourly")
        _render_sr_cross_table("Hourly — Upward crosses", hourly_up_rows)
        _render_sr_cross_table("Hourly — Downward crosses", hourly_down_rows)

        total_found = (
            len(daily_up_rows)
            + len(daily_down_rows)
            + len(daily_minus075_up_rows)
            + len(hourly_up_rows)
            + len(hourly_down_rows)
        )
        st.caption(f"Total recent crosses found: {total_found}")
    else:
        st.info("Click **Scan S/R Reversal Crosses** to build the Daily/Hourly cross tables.")



# --- Tab 8: -0.75 SR Crossers ---
with tab8:
    st.header("-0.75 SR Crossers")
    st.caption(
        "Daily-only scan for the **S/R Reversal line** on the NTD panel. "
        "The first section shows confirmed upward reversals from the support-side zone. "
        "The second section shows all symbols whose S/R Reversal line is currently below the selected threshold."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        minus075_recent = st.slider(
            "Recent reversal window (daily bars)",
            2, 90, 20, 1,
            key="minus075_sr_recent"
        )
    with c2:
        minus075_confirm = st.slider(
            "Rising confirmation bars",
            1, 10, 3, 1,
            key="minus075_sr_confirm"
        )
    with c3:
        minus075_smooth = st.slider(
            "S/R line smoothing",
            1, 30, int(sr_rev_smooth), 1,
            key="minus075_sr_smooth"
        )
    with c4:
        minus075_threshold = st.slider(
            "S/R threshold",
            -0.95, -0.50, -0.75, 0.05,
            key="minus075_sr_threshold"
        )

    run_minus075_scan = st.button("Scan -0.75 SR Crossers", key="btn_minus075_sr_scan")

    if run_minus075_scan:
        reversal_upward_trend_rows = []
        reversal_downward_trend_rows = []
        below_upward_trend_rows = []
        below_downward_trend_rows = []

        progress = st.progress(0, text="Scanning daily -0.75 S/R reversals and below-threshold symbols...")
        total_steps = max(1, len(universe))

        for i, sym in enumerate(universe, start=1):
            reversal_row = minus075_sr_crosser_info_daily(
                sym,
                smooth_span=int(minus075_smooth),
                recent_bars=int(minus075_recent),
                confirm_bars=int(minus075_confirm),
                threshold=float(minus075_threshold),
                slope_lookback=int(slope_lb_daily),
            )
            if reversal_row is not None:
                if reversal_row.get("Trend Group") == "Upward Trend":
                    reversal_upward_trend_rows.append(reversal_row)
                else:
                    reversal_downward_trend_rows.append(reversal_row)

            below_row = minus075_sr_below_info_daily(
                sym,
                smooth_span=int(minus075_smooth),
                threshold=float(minus075_threshold),
                slope_lookback=int(slope_lb_daily),
            )
            if below_row is not None:
                if below_row.get("Trend Group") == "Upward Trend":
                    below_upward_trend_rows.append(below_row)
                else:
                    below_downward_trend_rows.append(below_row)

            progress.progress(min(1.0, i / total_steps), text=f"Scanning daily: {sym}")

        progress.empty()

        st.markdown("### Recently Reversed Upward from the -0.75 Zone")
        _render_minus075_sr_table("Upward Trend", reversal_upward_trend_rows)
        _render_minus075_sr_table("Downward Trend", reversal_downward_trend_rows)

        st.markdown("### Currently Below the -0.75 Zone")
        _render_minus075_below_table("Upward Trend", below_upward_trend_rows)
        _render_minus075_below_table("Downward Trend", below_downward_trend_rows)

        total_reversal = len(reversal_upward_trend_rows) + len(reversal_downward_trend_rows)
        total_below = len(below_upward_trend_rows) + len(below_downward_trend_rows)
        st.caption(
            f"Total recent upward reversals found: {total_reversal} • "
            f"Total currently below threshold found: {total_below}"
        )
    else:
        st.info("Click **Scan -0.75 SR Crossers** to build the grouped daily tables.")


# --- Tab 9: HMA Cross ---
with tab9:
    st.header("HMA Cross")
    st.caption(
        "Daily-only scan for symbols where the trendline is upward, price has reversed upward from the "
        "30-support line, and price recently crossed upward through the HMA(55). Results are sorted by "
        "the lowest number of bars since the HMA cross."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        hma_cross_recent = st.slider(
            "Recent HMA cross window (daily bars)",
            1, 90, 20, 1,
            key="hma_cross_recent_bars"
        )
    with c2:
        hma_support_reversal_lookback = st.slider(
            "Support reversal lookback (daily bars)",
            3, 90, 20, 1,
            key="hma_support_reversal_lookback"
        )
    with c3:
        hma_support_confirm = st.slider(
            "Rising confirmation bars",
            1, 6, 2, 1,
            key="hma_support_confirm"
        )
    with c4:
        hma_support_prox = st.slider(
            "30-support proximity (%)",
            0.05, 2.00, 0.35, 0.05,
            key="hma_support_prox_pct"
        ) / 100.0

    run_hma_cross_scan = st.button("Scan HMA Cross", key="btn_hma_cross_scan")

    if run_hma_cross_scan:
        hma_rows = []
        progress = st.progress(0, text="Scanning daily HMA Cross setups...")
        total_steps = max(1, len(universe))

        for i, sym in enumerate(universe, start=1):
            row = hma_cross_info_daily(
                symbol=sym,
                hma_period_fixed=55,
                slope_lookback=int(slope_lb_daily),
                recent_cross_bars=int(hma_cross_recent),
                support_lookback=30,
                support_reversal_lookback=int(hma_support_reversal_lookback),
                support_prox=float(hma_support_prox),
                support_confirm_bars=int(hma_support_confirm),
            )
            if row is not None:
                hma_rows.append(row)

            progress.progress(min(1.0, i / total_steps), text=f"Scanning daily: {sym}")

        progress.empty()

        st.subheader("Daily HMA(55) Upward Cross after 30-Support Reversal")
        _render_hma_cross_table(hma_rows)
        st.caption(f"Total HMA Cross setups found: {len(hma_rows)}")
    else:
        st.info("Click **Scan HMA Cross** to build the daily setup table.")

# --- Tab 10: Green Triangle Pick ---
with tab10:
    st.header("Green Triangle Pick")
    st.caption(
        "Scans daily and hourly/intraday charts separately for symbols where the NTD chart recently "
        "printed a green BUY/support-reversal triangle while the price trendline is upward. Results "
        "are sorted by the lowest number of bars since the triangle appeared and include the S/R "
        "Reversal zone where the triangle formed plus whether price remains above support."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        green_daily_recent = st.slider(
            "Recent daily triangle window (bars)",
            1, 120, 30, 1,
            key="green_triangle_daily_recent_bars"
        )
    with c2:
        green_hourly_recent = st.slider(
            "Recent hourly/intraday triangle window (bars)",
            1, 240, 60, 1,
            key="green_triangle_hourly_recent_bars"
        )
    with c3:
        green_hourly_period_label = st.selectbox(
            "Hourly lookback",
            ["24h", "48h", "96h"],
            index=1,
            key="green_triangle_hourly_period"
        )
    with c4:
        green_support_window_daily = st.slider(
            "Daily support window",
            10, 120, 30, 5,
            key="green_triangle_daily_support_window"
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        green_support_window_hourly = st.slider(
            "Hourly support window",
            20, 240, int(sr_lb_hourly), 5,
            key="green_triangle_hourly_support_window"
        )
    with c6:
        green_sr_zone = st.slider(
            "S/R reversal zone strength",
            0.40, 0.95, float(sr_rev_zone), 0.05,
            key="green_triangle_sr_zone"
        )
    with c7:
        green_sr_confirm = st.slider(
            "S/R reversal confirmation bars",
            1, 6, int(sr_rev_confirm), 1,
            key="green_triangle_sr_confirm"
        )
    with c8:
        green_sr_lookback = st.slider(
            "S/R reversal touch lookback",
            2, 30, int(sr_rev_lookback), 1,
            key="green_triangle_sr_lookback"
        )

    run_green_triangle_scan = st.button("Scan Green Triangle Pick", key="btn_green_triangle_pick_scan")

    if run_green_triangle_scan:
        hourly_period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
        hourly_period = hourly_period_map.get(green_hourly_period_label, "2d")

        daily_rows = []
        hourly_rows = []

        total_steps = max(1, len(universe) * 2)
        done_steps = 0
        progress = st.progress(0, text="Scanning daily green NTD triangles...")

        for sym in universe:
            row = green_triangle_pick_info_daily(
                symbol=sym,
                slope_lookback=int(slope_lb_daily),
                recent_bars=int(green_daily_recent),
                support_window=int(green_support_window_daily),
                sr_prox=float(sr_prox_pct),
                sr_zone=float(green_sr_zone),
                sr_lookback=int(green_sr_lookback),
                sr_confirm=int(green_sr_confirm),
                sr_smooth=int(sr_rev_smooth),
            )
            if row is not None:
                daily_rows.append(row)

            done_steps += 1
            progress.progress(min(1.0, done_steps / total_steps), text=f"Scanning daily: {sym}")

        for sym in universe:
            row = green_triangle_pick_info_hourly(
                symbol=sym,
                period=hourly_period,
                slope_lookback=int(slope_lb_hourly),
                recent_bars=int(green_hourly_recent),
                support_window=int(green_support_window_hourly),
                sr_prox=float(sr_prox_pct),
                sr_zone=float(green_sr_zone),
                sr_lookback=int(green_sr_lookback),
                sr_confirm=int(green_sr_confirm),
                sr_smooth=int(sr_rev_smooth),
            )
            if row is not None:
                hourly_rows.append(row)

            done_steps += 1
            progress.progress(min(1.0, done_steps / total_steps), text=f"Scanning hourly: {sym}")

        progress.empty()

        _render_green_triangle_pick_table("Daily — Upward Trend with Recent Green BUY/Support-Reversal Triangle", daily_rows)
        _render_green_triangle_pick_table("Hourly — Upward Trend with Recent Green BUY/Support-Reversal Triangle", hourly_rows)
        st.caption(
            f"Daily green triangle picks found: {len(daily_rows)} • "
            f"Hourly green triangle picks found: {len(hourly_rows)}"
        )
    else:
        st.info("Click **Scan Green Triangle Pick** to build the daily and hourly green-triangle tables.")


# --- Tab 11: S/R Cross ---
with tab11:
    st.header("S/R Cross")
    st.caption(
        "Daily-chart scan for symbols where the **S/R Reversal Index** on the NTD chart is either "
        "below the selected support-side threshold or has recently crossed the **0.0** line upward. "
        "The Actionable S/R Long Picks table ranks the same scan into trade-ready candidates with "
        "status, setup quality, support/resistance, and reward/risk."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sr_cross_daily_only_threshold = st.slider(
            "Below-threshold level",
            -0.95, -0.25, -0.75, 0.05,
            key="sr_cross_daily_only_threshold"
        )
    with c2:
        sr_cross_daily_only_recent = st.slider(
            "Recent upward cross window (daily bars)",
            1, 120, 30, 1,
            key="sr_cross_daily_only_recent"
        )
    with c3:
        sr_cross_daily_only_smooth = st.slider(
            "S/R Reversal smoothing",
            1, 30, int(sr_rev_smooth), 1,
            key="sr_cross_daily_only_smooth"
        )
    with c4:
        sr_cross_daily_only_sr_window = st.slider(
            "Support/Resistance lookback",
            10, 120, 30, 5,
            key="sr_cross_daily_only_sr_window"
        )

    run_sr_cross_daily_only = st.button("Scan S/R Cross", key="btn_sr_cross_daily_only_scan")

    if run_sr_cross_daily_only:
        actionable_rows = []
        below_rows = []
        upward_rows = []

        progress = st.progress(0, text="Scanning daily S/R Cross setup...")
        total_steps = max(1, len(universe) * 3)
        step_count = 0

        for sym in universe:
            action_row = actionable_sr_long_pick_daily(
                sym,
                smooth_span=int(sr_cross_daily_only_smooth),
                recent_bars=int(sr_cross_daily_only_recent),
                slope_lookback=int(slope_lb_daily),
                sr_window=int(sr_cross_daily_only_sr_window),
            )
            if action_row is not None:
                actionable_rows.append(action_row)

            step_count += 1
            progress.progress(
                min(1.0, step_count / total_steps),
                text=f"Building actionable S/R long candidate: {sym}"
            )

            below_row = sr_cross_daily_below_threshold_info(
                sym,
                smooth_span=int(sr_cross_daily_only_smooth),
                threshold=float(sr_cross_daily_only_threshold),
                slope_lookback=int(slope_lb_daily),
            )
            if below_row is not None:
                below_rows.append(below_row)

            step_count += 1
            progress.progress(
                min(1.0, step_count / total_steps),
                text=f"Scanning below-threshold S/R Reversal: {sym}"
            )

            up_row = sr_cross_daily_upward_zero_info(
                sym,
                smooth_span=int(sr_cross_daily_only_smooth),
                recent_bars=int(sr_cross_daily_only_recent),
                slope_lookback=int(slope_lb_daily),
            )
            if up_row is not None:
                upward_rows.append(up_row)

            step_count += 1
            progress.progress(
                min(1.0, step_count / total_steps),
                text=f"Scanning upward 0.0 S/R Reversal crosses: {sym}"
            )

        progress.empty()

        st.markdown("### Actionable S/R Long Picks")
        st.caption(
            "Ranks daily long candidates by trade status, setup quality, cross timing, trend alignment, "
            "support hold, and reward/risk. Use this as a candidate list, then confirm on the chart."
        )
        _render_actionable_sr_long_picks_table(
            "Actionable S/R Long Picks — ranked by status and setup quality",
            actionable_rows
        )

        st.markdown("### Daily S/R Reversal Index Below Threshold")
        _render_sr_cross_daily_threshold_table(
            f"Current S/R Reversal Index below {float(sr_cross_daily_only_threshold):.2f}",
            below_rows
        )

        st.markdown("### Daily S/R Reversal Index Recently Crossed 0.0 Upward")
        _render_sr_cross_daily_upward_table(
            "Recent upward 0.0 crosses — sorted by bars since cross",
            upward_rows
        )

        st.caption(
            f"Actionable candidates found: {len(actionable_rows)} • "
            f"Below-threshold symbols found: {len(below_rows)} • "
            f"Recent upward 0.0 crosses found: {len(upward_rows)}"
        )
    else:
        st.info("Click **Scan S/R Cross** to build the actionable daily S/R Cross tables.")

# --- Tab 12: S/R -0.5 Cross ---
with tab12:
    st.header("S/R -0.5 Cross")
    st.caption(
        "Daily-chart scan for symbols where the **S/R Reversal Index** on the NTD chart recently crossed "
        "**-0.5 upward** and where it recently crossed **0.0 upward**. Results are sorted by the number "
        "of bars since the cross, lowest to highest."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sr_minus05_cross_recent = st.slider(
            "Recent -0.5 upward cross window (daily bars)",
            1, 120, 30, 1,
            key="sr_minus05_cross_recent"
        )
    with c2:
        sr_minus05_zero_cross_recent = st.slider(
            "Recent 0.0 upward cross window (daily bars)",
            1, 120, 30, 1,
            key="sr_minus05_zero_cross_recent"
        )
    with c3:
        sr_minus05_cross_smooth = st.slider(
            "S/R Reversal smoothing",
            1, 30, int(sr_rev_smooth), 1,
            key="sr_minus05_cross_smooth"
        )
    with c4:
        sr_minus05_show_only_current_up = st.checkbox(
            "Only show rows still sloping upward",
            value=False,
            key="sr_minus05_show_only_current_up"
        )

    run_sr_minus05_cross = st.button("Scan S/R -0.5 Cross", key="btn_sr_minus05_cross_scan")

    if run_sr_minus05_cross:
        minus05_rows = []
        zero_rows = []

        progress = st.progress(0, text="Scanning daily S/R -0.5 Cross setup...")
        total_steps = max(1, len(universe) * 2)
        step_count = 0

        for sym in universe:
            minus05_row = sr_reversal_level_up_cross_info_daily(
                sym,
                level=-0.5,
                smooth_span=int(sr_minus05_cross_smooth),
                recent_bars=int(sr_minus05_cross_recent),
                slope_lookback=int(slope_lb_daily),
            )
            if minus05_row is not None:
                if (not sr_minus05_show_only_current_up) or minus05_row.get("Current S/R Direction") == "Upward":
                    minus05_rows.append(minus05_row)

            step_count += 1
            progress.progress(
                min(1.0, step_count / total_steps),
                text=f"Scanning upward -0.5 S/R Reversal crosses: {sym}"
            )

            zero_row = sr_reversal_level_up_cross_info_daily(
                sym,
                level=0.0,
                smooth_span=int(sr_minus05_cross_smooth),
                recent_bars=int(sr_minus05_zero_cross_recent),
                slope_lookback=int(slope_lb_daily),
            )
            if zero_row is not None:
                if (not sr_minus05_show_only_current_up) or zero_row.get("Current S/R Direction") == "Upward":
                    zero_rows.append(zero_row)

            step_count += 1
            progress.progress(
                min(1.0, step_count / total_steps),
                text=f"Scanning upward 0.0 S/R Reversal crosses: {sym}"
            )

        progress.empty()

        st.markdown("### Daily S/R Reversal Index Recently Crossed -0.5 Upward")
        _render_sr_level_up_cross_table(
            "Recent upward -0.5 crosses — sorted by bars since cross",
            minus05_rows,
            "No recent upward -0.5 crosses found."
        )

        st.markdown("### Daily S/R Reversal Index Recently Crossed 0.0 Upward")
        _render_sr_level_up_cross_table(
            "Recent upward 0.0 crosses — sorted by bars since cross",
            zero_rows,
            "No recent upward 0.0 crosses found."
        )

        st.caption(
            f"Recent upward -0.5 crosses found: {len(minus05_rows)} • "
            f"Recent upward 0.0 crosses found: {len(zero_rows)}"
        )
    else:
        st.info("Click **Scan S/R -0.5 Cross** to build the daily upward-cross tables.")
