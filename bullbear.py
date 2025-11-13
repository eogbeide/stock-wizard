# bullbear.py — Stocks/Forex Dashboard + Forecasts
# (UPDATED) London & New York session Open/Close markers in PST on Forex intraday charts.
# (NEW) Normalized Price (NPX) plotted on NTD panels + crossing markers
# (NEW) BB Divergence Signals (price trend vs. Bollinger band drift) with confidence gate
# (NEW) ADX filter (period/threshold) + confluence gating for HMA, BB Divergence, and Near S/R signals
# (NEW) Purple triangle BUY/SELL markers on NTD panel when NTD crosses −0.75 upward (BUY) or +0.75 downward (SELL)

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

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

# NEW: Hourly S/R window
st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

# Hourly Momentum
st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=True, key="sb_show_mom_hourly")
mom_lb_hourly   = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

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

# NEW: Parabolic SAR
st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

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
show_npx_ntd   = st.sidebar.checkbox("Overlay normalized price (NPX) on NTD panels", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses (▲/▼)", value=True, key="sb_mark_npx_cross")

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
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','EURCAD=X'
    ]

# --- Cache helpers (TTL = 120 seconds) ---
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")[['Close']].dropna()
    s = df['Close'].asfreq("D").fillna(method="ffill")
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[['Open','High','Low','Close']].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

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

# ---- Normalized MACD (price-based) ----
def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
                  norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),)*3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    minp = max(10, norm_win//10)
    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)
    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return nmacd.reindex(s.index), nsignal.reindex(s.index), nhist.reindex(s.index)

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
    color = "tab:green" if m >= 0 else "tab:red"
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
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
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
                                y_up: float = 0.95, y_dn: float = -0.95, label_prefix: str = "HMA REV", period: int = 55):
    try:
        buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
        idx_up = list(buy_rev[buy_rev].index)
        idx_dn = list(sell_rev[sell_rev].index)
        if len(idx_up):
            ax.scatter(idx_up, [y_up]*len(idx_up), marker="^", s=70, color="tab:green",
                       zorder=8, label=f"HMA({period}) ↑ REV")
        if len(idx_dn):
            ax.scatter(idx_dn, [y_dn]*len(idx_dn), marker="v", s=70, color="tab:red",
                       zorder=8, label=f"HMA({period}) ↓ REV")
    except Exception:
        pass

# ========= NPX ↔ NTD overlay =========
def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx); ntd = ntd.reindex(idx)
    if npx.dropna().empty:
        return
    ax.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")
    if mark_crosses and not ntd.dropna().empty:
        up_mask, dn_mask = _cross_series(npx, ntd)
        try:
            up_idx = list(up_mask[up_mask].index)
            dn_idx = list(dn_mask[dn_mask].index)
            if len(up_idx):
                ax.scatter(up_idx, ntd.loc[up_idx], marker="^", s=65, color="tab:green", zorder=9, label="Price↑NTD")
            if len(dn_idx):
                ax.scatter(dn_idx, ntd.loc[dn_idx], marker="v", s=65, color="tab:red", zorder=9, label="Price↓NTD")
        except Exception:
            pass

# ========= NEW: NTD threshold crossing (purple triangles) =========
def overlay_ntd_threshold_triangles(ax, ntd: pd.Series, buy_level: float = -0.75, sell_level: float = 0.75,
                                    size: int = 85):
    """
    Plot purple triangles on the NTD panel:
      - BUY (^, purple) when NTD crosses up through buy_level (default -0.75)
      - SELL (v, purple) when NTD crosses down through sell_level (default +0.75)
    Markers are placed exactly on the threshold levels for readability.
    """
    s = _coerce_1d_series(ntd)
    if s.dropna().shape[0] < 2:
        return
    # Upward cross through buy_level
    cross_up = (s >= buy_level) & (s.shift(1) < buy_level)
    # Downward cross through sell_level
    cross_dn = (s <= sell_level) & (s.shift(1) > sell_level)

    idx_up = list(cross_up[cross_up.fillna(False)].index)
    idx_dn = list(cross_dn[cross_dn.fillna(False)].index)

    try:
        if idx_up:
            ax.scatter(idx_up, [buy_level]*len(idx_up), marker="^", s=size, color="purple",
                       zorder=10, label="NTD BUY (−0.75↑)")
        if idx_dn:
            ax.scatter(idx_dn, [sell_level]*len(idx_dn), marker="v", s=size, color="purple",
                       zorder=10, label="NTD SELL (+0.75↓)")
    except Exception:
        pass

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
        last_close = float(hc.iloc[-1])
        res = float(res_h.iloc[-1])
        sup = float(sup_h.iloc[-1])
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
                   marker="^", s=60, color="tab:green", zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above], [0.92]*int(enter_from_above.sum()),
                   marker="v", s=60, color="tab:orange", zorder=7, label="Enter from R")
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

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if 'hist_years' not in st.session_state:
    st.session_state.hist_years = 10

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.5 Scanner",
    "Long-Term History"
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
            yhat_ema30, m_ema30
