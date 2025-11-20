# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATED — per request:
#   On the NTD panel, show **green triangles only when the PRICE trendline is upward**
#   and **red triangles only when the PRICE trendline is downward**.
#   (We keep the triangle events the same as before: NTD 0-line crosses and first
#    entries outside the ±0.75 band — but we gate their display by the price trend sign.)
#
#   Other indicators (stars for S/R reversals, NPX overlay, etc.) remain unchanged.
#
#   NEW (this update):
#   • regression_with_band(): linear trendline + ±2σ band + R²
#   • Daily price chart: trendline ±2σ + R²
#   • Intraday price chart: local slope trendline ±2σ + R²
#   • Long-Term History: trendline ±2σ + R²
#   • Metrics price charts: trendline ±2σ + R²
#
#   NEW (scanner update):
#   • Tab 5 now scans for the same event that plots **green dots** on the NTD panel:
#       Price↑NTD (NPX crosses above NTD) on the **latest bar**, with NTD below a filter level.

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

# --- Page config ---
st.set_page_config(page_title="📊 Dashboard & Forecasts", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

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

# --- Auto-refresh ---
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

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str) -> str:
    """
    IMPORTANT: trade direction is keyed to `trend_slope` (the **regression slope** we use
    for the ±2σ bands), not the global long-range dashed trendline.
    For upward slope: BUY first, then SELL. For downward slope: SELL first, then BUY.
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

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

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, y_val, text, transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=6)

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
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")
show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=True, key="sb_show_mom_hourly")
mom_lb_hourly   = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

st.sidebar.subheader("Hourly Indicator Panel")
show_nrsi   = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox("Highlight when price is between S/R (S↔R) on NTD", value=True, key="sb_ntd_channel")

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signal Logic")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

st.sidebar.subheader("NTD (Daily/Hourly)")
show_ntd  = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd")
ntd_window= st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
show_npx_ntd   = st.sidebar.checkbox("Overlay normalized price (NPX) on NTD", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses (dots)", value=True, key="sb_mark_npx_cross")

st.sidebar.subheader("Normalized Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

st.sidebar.subheader("HMA(55) Reversal on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb       = st.sidebar.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

st.sidebar.subheader("Reversal Stars (on NTD panel)")
rev_bars_confirm = st.sidebar.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

# Forex-only controls
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
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X'
    ]

# Cache TTL = 120s
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
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
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
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
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5 * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100%": lo
    }

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {'High','Low','Close'}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
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
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    """
    Linear regression on last `lookback` bars of `series_like` with:
      • fitted trendline
      • symmetric ±z·σ band (σ = std of residuals, z≈2 → ~95% coverage)
      • R² of the fit
    Returns (trend, upper, lower, slope, r2).
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
    std = float(np.sqrt(np.sum(resid ** 2) / dof))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

# RSI / Normalized RSI
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

# Normalized MACD (kept)
def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),) * 3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig

    minp = max(10, norm_win // 10)

    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)

    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return nmacd.reindex(s.index), nsignal.reindex(s.index), nhist.reindex(s.index)

def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win // 10)
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
    minp = max(10, int(norm_win) // 10)
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
    color = "tab:green" if m >= 0 else "tab:red"
    ax.plot(s.index, yhat, "-", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m
# Supertrend
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
    in_up.iloc[0] = True
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
    return pd.DataFrame(
        {"ST": st_line, "in_uptrend": in_up,
         "upperband": upperband, "lowerband": lowerband}
    )

# Parabolic SAR
def compute_parabolic_sar(high: pd.Series, low: pd.Series,
                          step: float = 0.02, max_step: float = 0.2):
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
                ep = df["H"].iloc[i]
                af = min(af + step, max_step)
            if df["L"].iloc[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = df["L"].iloc[i]
                af = step
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            hi1 = df["H"].iloc[i-1]
            hi2 = df["H"].iloc[i-2] if i >= 2 else hi1
            psar[i] = max(psar[i], hi1, hi2)
            if df["L"].iloc[i] < ep:
                ep = df["L"].iloc[i]
                af = min(af + step, max_step)
            if df["H"].iloc[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = df["H"].iloc[i]
                af = step
        up[i] = uptrend

    return pd.Series(psar, index=df.index, name="PSAR"), pd.Series(up, index=df.index, name="in_uptrend")

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["PSAR","in_uptrend"])
    ps, up = compute_parabolic_sar(df["High"], df["Low"], step=step, max_step=max_step)
    return pd.DataFrame({"PSAR": ps, "in_uptrend": up})

# Ichimoku (classic)
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52,
                   shift_cloud: bool = False):
    H = _coerce_1d_series(high)
    L = _coerce_1d_series(low)
    C = _coerce_1d_series(close)
    if H.empty or L.empty or C.empty:
        idx = C.index if not C.empty else (H.index if not H.empty else L.index)
        return (pd.Series(index=idx, dtype=float),) * 5

    tenkan = (H.rolling(conv).max() + L.rolling(conv).min()) / 2.0
    kijun  = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (H.rolling(span_b).max() + L.rolling(span_b).min()) / 2.0
    span_a = span_a_raw.shift(base) if shift_cloud else span_a_raw
    span_b = span_b_raw.shift(base) if shift_cloud else span_b_raw
    chikou = C.shift(-base)
    return tenkan, kijun, span_a, span_b, chikou

# Bollinger Bands + normalized %B
def compute_bbands(close: pd.Series, window: int = 20,
                   mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty or window < 2 or not np.isfinite(mult):
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty, empty
    minp = max(2, window // 2)
    mid = (s.ewm(span=window, adjust=False).mean()
           if use_ema else s.rolling(window, min_periods=minp).mean())
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0, np.nan)
    pctb = ((s - lower) / width).clip(0.0, 1.0)
    nbb = pctb * 2.0 - 1.0
    return (mid.reindex(s.index), upper.reindex(s.index),
            lower.reindex(s.index), pctb.reindex(s.index),
            nbb.reindex(s.index))

# HMA + crossover helpers (for price chart)
def _wma(s: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(s).astype(float)
    if s.empty or window < 1:
        return pd.Series(index=s.index, dtype=float)
    w = np.arange(1, window + 1, dtype=float)
    return s.rolling(window, min_periods=window).apply(
        lambda x: float(np.dot(x, w) / w.sum()),
        raw=True
    )

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
    p = p[mask]
    l = l[mask]
    above = p > l
    cross_up  = above & (~above.shift(1).fillna(False))   # up-cross
    cross_dn  = (~above) & (above.shift(1).fillna(False)) # down-cross
    t_up = cross_up[cross_up].index[-1] if cross_up.any() else None
    t_dn = cross_dn[cross_dn].index[-1] if cross_dn.any() else None
    if t_up is None and t_dn is None:
        return None
    if t_dn is None or (t_up is not None and t_up > t_dn):
        return {"time": t_up, "side": "BUY"}
    else:
        return {"time": t_dn, "side": "SELL"}

def annotate_crossover(ax, ts, px, side: str, conf: float):
    if side == "BUY":
        ax.scatter([ts], [px], marker="P", s=90, color="tab:green", zorder=7)
        ax.text(ts, px, f"  BUY {int(conf*100)}%", va="bottom",
                fontsize=9, color="tab:green", fontweight="bold")
    else:
        ax.scatter([ts], [px], marker="X", s=90, color="tab:red", zorder=7)
        ax.text(ts, px, f"  SELL {int(conf*100)}%", va="top",
                fontsize=9, color="tab:red", fontweight="bold")

# HMA reversal markers on NTD (kept – shapes are squares/diamonds)
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
    return (cross_up.reindex(p.index, fill_value=False),
            cross_dn.reindex(p.index, fill_value=False))

def detect_hma_reversal_masks(price: pd.Series, hma: pd.Series,
                              lookback: int = 3):
    h = _coerce_1d_series(hma)
    slope = h.diff().rolling(lookback, min_periods=1).mean()
    sign_now = np.sign(slope)
    sign_prev = np.sign(slope.shift(1))
    cross_up, cross_dn = _cross_series(price, hma)
    buy_rev  = cross_up & (sign_now > 0) & (sign_prev < 0)
    sell_rev = cross_dn & (sign_now < 0) & (sign_prev > 0)
    return buy_rev.fillna(False), sell_rev.fillna(False)

def overlay_hma_reversal_on_ntd(ax, price: pd.Series, hma: pd.Series,
                                lookback: int = 3,
                                y_up: float = 0.95, y_dn: float = -0.95,
                                label_prefix: str = "HMA REV",
                                period: int = 55,
                                ntd: pd.Series = None,
                                upper_thr: float = 0.75,
                                lower_thr: float = -0.75):
    buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
    idx_up = list(buy_rev[buy_rev].index)
    idx_dn = list(sell_rev[sell_rev].index)
    if ntd is not None:
        ntd = _coerce_1d_series(ntd).reindex(hma.index)

    if len(idx_up):
        ax.scatter(idx_up, [y_up] * len(idx_up), marker="s", s=70,
                   color="tab:green", zorder=8, label=f"HMA({period}) REV")
    if len(idx_dn):
        ax.scatter(idx_dn, [y_dn] * len(idx_dn), marker="D", s=70,
                   color="tab:red",   zorder=8, label=f"HMA({period}) REV")

# NPX ↔ NTD overlay/helpers (markers dots/x)
def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series,
                       mark_crosses: bool = True):
    """
    This is the source of the **green dots** on the NTD panel:
      • green 'o' = Price↑NTD (NPX crosses above NTD)
      • red 'x'   = Price↓NTD
    The scanner in Tab 5 now uses the **same event logic** (Price↑NTD on the latest bar).
    """
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx)
    ntd = ntd.reindex(idx)
    if npx.dropna().empty:
        return
    ax.plot(npx.index, npx.values, "-", linewidth=1.2,
            color="tab:gray", alpha=0.9, label="NPX (Norm Price)")
    if mark_crosses and not ntd.dropna().empty:
        up_mask, dn_mask = _cross_series(npx, ntd)
        up_idx = list(up_mask[up_mask].index)
        dn_idx = list(dn_mask[dn_mask].index)
        if len(up_idx):
            ax.scatter(up_idx, ntd.loc[up_idx], marker="o", s=40,
                       color="tab:green", zorder=9, label="Price↑NTD")
        if len(dn_idx):
            ax.scatter(dn_idx, ntd.loc[dn_idx], marker="x", s=60,
                       color="tab:red",   zorder=9, label="Price↓NTD")

# --- NEW: NTD triangles gated by PRICE trend sign ---
def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series,
                                   trend_slope: float,
                                   upper: float = 0.75,
                                   lower: float = -0.75):
    """
    Show triangles on NTD panel, but:
      • GREEN triangles only if price trend slope > 0
      • RED triangles   only if price trend slope < 0
    Triangle events:
      • Cross 0 upward (green) / downward (red) — plotted at y=0
      • First entry outside band: above +0.75 (red), below -0.75 (green)
    """
    s = _coerce_1d_series(ntd).dropna()
    if s.empty or not np.isfinite(trend_slope):
        return

    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    # Zero crossings
    cross_up0 = (s >= 0.0) & (s.shift(1) < 0.0)
    cross_dn0 = (s <= 0.0) & (s.shift(1) > 0.0)
    idx_up0 = list(cross_up0[cross_up0].index)
    idx_dn0 = list(cross_dn0[cross_dn0].index)

    # Outside band entries
    cross_out_hi = (s >= upper) & (s.shift(1) < upper)   # entering > +0.75
    cross_out_lo = (s <= lower) & (s.shift(1) > lower)   # entering < -0.75
    idx_hi = list(cross_out_hi[cross_out_hi].index)
    idx_lo = list(cross_out_lo[cross_out_lo].index)

    if uptrend:
        if idx_up0:
            ax.scatter(idx_up0, [0.0] * len(idx_up0),
                       marker="^", s=95, color="tab:green",
                       zorder=10, label="NTD 0↑")
        if idx_lo:
            ax.scatter(idx_lo, s.loc[idx_lo],
                       marker="^", s=85, color="tab:green",
                       zorder=10, label="NTD < -0.75")
    if downtrend:
        if idx_dn0:
            ax.scatter(idx_dn0, [0.0] * len(idx_dn0),
                       marker="v", s=95, color="tab:red",
                       zorder=10, label="NTD 0↓")
        if idx_hi:
            ax.scatter(idx_hi, s.loc[idx_hi],
                       marker="v", s=85, color="tab:red",
                       zorder=10, label="NTD > +0.75")

# --- Reversal Stars (kept) ---
def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n + 1:
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
        gap_res_0 = R0 - c0
        gap_res_1 = R0 - c1
        toward_res = gap_res_0 < gap_res_1

        gap_sup_0 = c0 - S0
        gap_sup_1 = c1 - S0
        toward_sup = gap_sup_0 < gap_sup_1

    buy_cond  = (trend_slope > 0) and near_support \
        and _n_consecutive_increasing(p, bars_confirm) and toward_res
    sell_cond = (trend_slope < 0) and near_resist \
        and _n_consecutive_decreasing(p, bars_confirm) and toward_sup

    if buy_cond:
        ax.scatter([t], [ntd0], marker="*",
                   s=170, color="tab:green", zorder=12,
                   label="BUY ★ (Support reversal)")
    if sell_cond:
        ax.scatter([t], [ntd0], marker="*",
                   s=170, color="tab:red",   zorder=12,
                   label="SELL ★ (Resistance reversal)")

# Sessions
NY_TZ   = pytz.timezone("America/New_York")
LDN_TZ  = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz,
                              open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    start_d = idx[0].astimezone(session_tz).date()
    end_d   = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local  = session_tz.localize(
                datetime(d.year, d.month, d.day, open_hr, 0, 0),
                is_dst=None
            )
            dt_close_local = session_tz.localize(
                datetime(d.year, d.month, d.day, close_hr, 0, 0),
                is_dst=None
            )
        except Exception:
            dt_open_local  = session_tz.localize(
                datetime(d.year, d.month, d.day, open_hr, 0, 0)
            )
            dt_close_local = session_tz.localize(
                datetime(d.year, d.month, d.day, close_hr, 0, 0)
            )
        dt_open_pst  = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst  <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ,  8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close,
            "ny_open": ny_open,   "ny_close": ny_close}

def draw_session_lines(ax, lines: dict):
    ax.plot([], [], linestyle="-",  color="tab:blue",
            label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",
            label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange",
            label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange",
            label="New York Close (PST)")
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0,
                   color="tab:blue", alpha=0.35)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0,
                   color="tab:blue", alpha=0.35)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-",  linewidth=1.0,
                   color="tab:orange", alpha=0.35)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0,
                   color="tab:orange", alpha=0.35)
    ax.text(0.99, 0.98, "Session times in PST",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22",
                      fc="white", ec="grey", alpha=0.7))

# News (Yahoo Finance)
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

def draw_news_markers(ax, times, ymin, ymax, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# Channel-in-range helpers for NTD panel
def channel_state_series(price: pd.Series,
                         sup: pd.Series,
                         res: pd.Series,
                         eps: float = 0.0) -> pd.Series:
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

def overlay_inrange_on_ntd(ax, price: pd.Series,
                           sup: pd.Series,
                           res: pd.Series):
    state = channel_state_series(price, sup, res)
    in_mask = (state == 0)
    for a, b in _true_spans(in_mask):
        try:
            ax.axvspan(a, b, color="gold", alpha=0.15, zorder=1)
        except Exception:
            pass
    ax.plot([], [], linewidth=8, color="gold", alpha=0.20,
            label="In Range (S↔R)")
    enter_from_below = (state.shift(1) == -1) & (state == 0)
    enter_from_above = (state.shift(1) ==  1) & (state == 0)
    if enter_from_below.any():
        ax.scatter(price.index[enter_from_below],
                   [0.92] * int(enter_from_below.sum()),
                   marker="^", s=60, color="tab:green",
                   zorder=7, label="Enter from S")
    if enter_from_above.any():
        ax.scatter(price.index[enter_from_above],
                   [0.92] * int(enter_from_above.sum()),
                   marker="v", s=60, color="tab:orange",
                   zorder=7, label="Enter from R")

    lbl = None
    col = "black"
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
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=col, alpha=0.85))
    return last
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

def _price_above_kijun_from_df(df: pd.DataFrame, base: int = 26):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        return False, None, np.nan, np.nan
    ohlc = df[['High','Low','Close']].copy()
    _, kijun, _, _, _ = ichimoku_lines(
        ohlc['High'], ohlc['Low'], ohlc['Close'], base=base
    )
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
def price_above_kijun_info_hourly(symbol: str, period: str = "1d",
                                  base: int = 26):
    try:
        df = fetch_intraday(symbol, period=period)
        return _price_above_kijun_from_df(df, base=base)
    except Exception:
        return False, None, np.nan, np.nan

def rolling_midline(series_like: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(series_like).astype(float)
    if s.empty:
        return pd.Series(index=s.index, dtype=float)
    roll = s.rolling(window, min_periods=1)
    mid = (roll.max() + roll.min()) / 2.0
    return mid.reindex(s.index)

def _has_volume_to_plot(vol: pd.Series) -> bool:
    s = _coerce_1d_series(vol).astype(float).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if s.shape[0] < 2:
        return False
    arr = s.to_numpy(dtype=float)
    vmax = float(np.nanmax(arr))
    vmin = float(np.nanmin(arr))
    return (np.isfinite(vmax) and vmax > 0.0) or (np.isfinite(vmin) and vmin < 0.0)

# --- NEW: helpers for green-dot (Price↑NTD) scanner ---
@st.cache_data(ttl=120)
def last_green_dot_daily(symbol: str, ntd_win: int, level: float = None):
    """
    Daily green-dot condition (Price↑NTD on latest bar) with optional NTD level filter.
    """
    try:
        s = fetch_hist(symbol)
        if s is None or s.empty:
            return False, np.nan, None, np.nan, np.nan

        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty or len(ntd) < 2:
            return False, np.nan, None, np.nan, np.nan

        npx = compute_normalized_price(s, window=ntd_win).reindex(ntd.index)
        df = pd.DataFrame({"ntd": ntd, "npx": npx}).dropna()
        if df.shape[0] < 2:
            return False, np.nan, None, np.nan, np.nan

        above = df["npx"] > df["ntd"]
        above_now = bool(above.iloc[-1])
        above_prev = bool(above.iloc[-2])
        is_signal = (not above_prev) and above_now

        ntd_last = float(df["ntd"].iloc[-1])
        npx_last = float(df["npx"].iloc[-1])
        idx_last = df.index[-1]

        s_aligned = _coerce_1d_series(s).reindex(df.index)
        try:
            close_last = float(s_aligned.iloc[-1])
        except Exception:
            close_last = _safe_last_float(s)

        if level is not None and np.isfinite(ntd_last):
            if not (ntd_last < level):
                is_signal = False

        return is_signal, ntd_last, idx_last, npx_last, close_last
    except Exception:
        return False, np.nan, None, np.nan, np.nan

@st.cache_data(ttl=120)
def last_green_dot_hourly(symbol: str, ntd_win: int,
                          period: str = "1d",
                          level: float = None):
    """
    Hourly green-dot condition (Price↑NTD on latest bar) with optional NTD level filter.
    """
    try:
        df = fetch_intraday(symbol, period=period)
        if df is None or df.empty or "Close" not in df:
            return False, np.nan, None, np.nan, np.nan

        s = df["Close"].ffill()
        ntd = compute_normalized_trend(s, window=ntd_win).dropna()
        if ntd.empty or len(ntd) < 2:
            return False, np.nan, None, np.nan, np.nan

        npx = compute_normalized_price(s, window=ntd_win).reindex(ntd.index)
        df2 = pd.DataFrame({"ntd": ntd, "npx": npx}).dropna()
        if df2.shape[0] < 2:
            return False, np.nan, None, np.nan, np.nan

        above = df2["npx"] > df2["ntd"]
        above_now = bool(above.iloc[-1])
        above_prev = bool(above.iloc[-2])
        is_signal = (not above_prev) and above_now

        ntd_last = float(df2["ntd"].iloc[-1])
        npx_last = float(df2["npx"].iloc[-1])
        idx_last = df2.index[-1]

        s_aligned = s.reindex(df2.index)
        try:
            close_last = float(s_aligned.iloc[-1])
        except Exception:
            close_last = _safe_last_float(s)

        if level is not None and np.isfinite(ntd_last):
            if not (ntd_last < level):
                is_signal = False

        return is_signal, ntd_last, idx_last, npx_last, close_last
    except Exception:
        return False, np.nan, None, np.nan, np.nan

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
        index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
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
        p_up = (np.mean(st.session_state.fc_vals.to_numpy() > last_price)
                if np.isfinite(last_price) else np.nan)
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily (Price + Indicator panel) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()

            # NEW: trendline with ±2σ band and R² (Daily)
            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(
                df, slope_lb_daily
            )

            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)
            piv = current_daily_pivots(df_ohlc)

            ntd_d = (compute_normalized_trend(df, window=ntd_window)
                     if show_ntd else pd.Series(index=df.index, dtype=float))
            npx_d_full = (compute_normalized_price(df, window=ntd_window)
                          if show_npx_ntd else pd.Series(index=df.index, dtype=float))

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    shift_cloud=False
                )
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(
                df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
            )

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            yhat_ema_show = (yhat_ema30.reindex(df_show.index)
                             if not yhat_ema30.empty else yhat_ema30)
            ntd_d_show  = ntd_d.reindex(df_show.index)
            npx_d_show  = npx_d_full.reindex(df_show.index)
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()

            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)
            bb_pctb_d_show= bb_pctb_d.reindex(df_show.index)
            bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

            hma_d_full = compute_hma(df, period=hma_period)
            hma_d_show = hma_d_full.reindex(df_show.index)

            psar_d_df = (compute_psar_from_ohlc(
                df_ohlc, step=psar_step, max_step=psar_max
            ) if show_psar else pd.DataFrame())
            if not psar_d_df.empty and len(df_show.index) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                psar_d_df = psar_d_df.loc[
                    (psar_d_df.index >= x0) & (psar_d_df.index <= x1)
                ]

            fig, (ax, axdw) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)

            # PRICE CHART (Daily)
            ax.set_title(
                f"{sel} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope, Pivots"
            )
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-",
                        linewidth=1.6, label=f"HMA({hma_period})")

            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-",
                        linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and (not bb_up_d_show.dropna().empty
                                and not bb_lo_d_show.dropna().empty):
                ax.fill_between(
                    df_show.index, bb_lo_d_show, bb_up_d_show,
                    alpha=0.06, label=f"BB (×{bb_mult:.1f})"
                )
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-",
                        linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":",
                        linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":",
                        linewidth=1.0)
                try:
                    last_pct = float(bb_pctb_d_show.dropna().iloc[-1])
                    last_nbb = float(bb_nbb_d_show.dropna().iloc[-1])
                    ax.text(
                        0.99, 0.02,
                        f"NBB {last_nbb:+.2f}  |  %B {fmt_pct(last_pct, digits=0)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="grey", alpha=0.7)
                    )
                except Exception:
                    pass

            if show_psar and not psar_d_df.empty:
                up_mask = psar_d_df["in_uptrend"] == True
                dn_mask = ~up_mask
                if up_mask.any():
                    ax.scatter(psar_d_df.index[up_mask],
                               psar_d_df["PSAR"][up_mask],
                               s=15, color="tab:green", zorder=6,
                               label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
                if dn_mask.any():
                    ax.scatter(psar_d_df.index[dn_mask],
                               psar_d_df["PSAR"][dn_mask],
                               s=15, color="tab:red",   zorder=6)

            if not yhat_d_show.empty:
                ax.plot(
                    yhat_d_show.index, yhat_d_show.values, "-",
                    linewidth=2,
                    label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)"
                )
            if not upper_d_show.empty and not lower_d_show.empty:
                # thicker ±2σ band lines
                ax.plot(
                    upper_d_show.index, upper_d_show.values, "--",
                    linewidth=2.0, label="Daily Trend +2σ"
                )
                ax.plot(
                    lower_d_show.index, lower_d_show.values, "--",
                    linewidth=2.0, label="Daily Trend -2σ"
                )

            if not yhat_ema_show.empty:
                ax.plot(
                    yhat_ema_show.index, yhat_ema_show.values, "-",
                    linewidth=2,
                    label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)"
                )

            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")

            if piv and len(df_show) > 0:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1,
                              linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}",
                            va="center")

            if len(res30_show) and len(sup30_show):
                r30_last = float(res30_show.iloc[-1])
                s30_last = float(sup30_show.iloc[-1])
                ax.text(
                    df_show.index[-1], r30_last,
                    f"  30R = {fmt_price_val(r30_last)}", va="bottom"
                )
                ax.text(
                    df_show.index[-1], s30_last,
                    f"  30S = {fmt_price_val(s30_last)}", va="top"
                )

            ax.set_ylabel("Price")
            # NEW: R² annotation for Daily price trend
            ax.text(
                0.50, 0.02,
                f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="grey", alpha=0.7)
            )
            ax.legend(loc="lower left", framealpha=0.5)

            # DAILY INDICATOR PANEL — NTD + NPX + Triangles by Trend + Stars
            axdw.set_title("Daily Indicator Panel — NTD + NPX + Trend")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw.plot(
                    ntd_d_show.index, ntd_d_show, "-",
                    linewidth=1.6, label=f"NTD (win={ntd_window})"
                )
                ntd_trend_d, ntd_m_d = slope_line(ntd_d_show, slope_lb_daily)
                if not ntd_trend_d.empty:
                    axdw.plot(
                        ntd_trend_d.index, ntd_trend_d.values, "--",
                        linewidth=2,
                        label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d)}/bar)"
                    )

                # NEW — triangles gated by price trend sign
                overlay_ntd_triangles_by_trend(
                    axdw, ntd_d_show, trend_slope=m_d,
                    upper=0.75, lower=-0.75
                )

                # Stars for S/R reversals
                overlay_ntd_sr_reversal_stars(
                    axdw, price=df_show, sup=sup30_show, res=res30_show,
                    trend_slope=m_d, ntd=ntd_d_show,
                    prox=sr_prox_pct, bars_confirm=rev_bars_confirm
                )

            if show_npx_ntd and (not npx_d_show.dropna().empty
                                 and not ntd_d_show.dropna().empty):
                overlay_npx_on_ntd(
                    axdw, npx_d_show, ntd_d_show, mark_crosses=mark_npx_cross
                )
            if show_hma_rev_ntd and (not hma_d_show.dropna().empty
                                     and not df_show.dropna().empty):
                overlay_hma_reversal_on_ntd(
                    axdw, df_show, hma_d_show, lookback=hma_rev_lb,
                    period=hma_period, ntd=ntd_d_show
                )

            axdw.axhline(0.0,  linestyle="--", linewidth=1.0,
                         color="black", label="0.00")
            axdw.axhline(0.75, linestyle="-",  linewidth=1.0,
                         color="black", label="+0.75")
            axdw.axhline(-0.75, linestyle="-",  linewidth=1.0,
                         color="black", label="-0.75")
            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")
            axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly (price + NTD panel + momentum + Volume) -----
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
                st_intraday = compute_supertrend(
                    intraday, atr_period=atr_period, atr_mult=atr_mult
                )
                st_line_intr = (st_intraday["ST"].reindex(hc.index)
                                if "ST" in st_intraday
                                else pd.Series(index=hc.index, dtype=float))

                kijun_h = pd.Series(index=hc.index, dtype=float)
                if {'High','Low','Close'}.issubset(intraday.columns) and show_ichi:
                    _, kijun_h, _, _, _ = ichimoku_lines(
                        intraday["High"], intraday["Low"], intraday["Close"],
                        conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                        shift_cloud=False
                    )
                    kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

                bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
                    hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
                )
                hma_h = compute_hma(hc, period=hma_period)
                psar_h_df = (compute_psar_from_ohlc(
                    intraday, step=psar_step, max_step=psar_max
                ) if show_psar else pd.DataFrame())
                psar_h_df = psar_h_df.reindex(hc.index)

                # NEW: local regression trend with ±2σ bands and R² (Hourly)
                yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(
                    hc, slope_lb_hourly
                )
                # slope used for trade instructions & HMA direction
                slope_for_sign = m_h if np.isfinite(m_h) else slope_h

                fig2, ax2 = plt.subplots(figsize=(14, 4))
                plt.subplots_adjust(top=0.85, right=0.93)
                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(
                    hc.index, trend_h, "--",
                    label=f"Trend (m={fmt_slope(slope_h)}/bar)", linewidth=2
                )

                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(
                        hma_h.index, hma_h.values, "-",
                        linewidth=1.6, label=f"HMA({hma_period})"
                    )
                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(
                        kijun_h.index, kijun_h.values, "-",
                        linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})"
                    )

                if show_bbands and (not bb_up_h.dropna().empty
                                    and not bb_lo_h.dropna().empty):
                    ax2.fill_between(
                        hc.index, bb_lo_h, bb_up_h,
                        alpha=0.06, label=f"BB (×{bb_mult:.1f})"
                    )
                    ax2.plot(
                        bb_mid_h.index, bb_mid_h.values, "-",
                        linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})"
                    )
                    ax2.plot(bb_up_h.index, bb_up_h.values, ":",
                             linewidth=1.0)
                    ax2.plot(bb_lo_h.index, bb_lo_h.values, ":",
                             linewidth=1.0)

                if show_psar and not psar_h_df.dropna().empty:
                    up_mask = psar_h_df["in_uptrend"] == True
                    dn_mask = ~up_mask
                    if up_mask.any():
                        ax2.scatter(
                            psar_h_df.index[up_mask],
                            psar_h_df["PSAR"][up_mask],
                            s=15, color="tab:green", zorder=6,
                            label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})"
                        )
                    if dn_mask.any():
                        ax2.scatter(
                            psar_h_df.index[dn_mask],
                            psar_h_df["PSAR"][dn_mask],
                            s=15, color="tab:red",   zorder=6
                        )

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = float(res_h.iloc[-1])
                    sup_val = float(sup_h.iloc[-1])
                    px_val  = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(
                        res_val, xmin=hc.index[0], xmax=hc.index[-1],
                        colors="tab:red", linestyles="-", linewidth=1.6,
                        label="Resistance"
                    )
                    ax2.hlines(
                        sup_val, xmin=hc.index[0], xmax=hc.index[-1],
                        colors="tab:green", linestyles="-", linewidth=1.6,
                        label="Support"
                    )
                    label_on_left(ax2, res_val,
                                  f"R {fmt_price_val(res_val)}",
                                  color="tab:red")
                    label_on_left(ax2, sup_val,
                                  f"S {fmt_price_val(sup_val)}",
                                  color="tab:green")

                # --- NEW: trade instruction keyed to regression slope (m_h) ---
                instr_txt = format_trade_instruction(
                    trend_slope=slope_for_sign,
                    buy_val=sup_val,
                    sell_val=res_val,
                    close_val=px_val,
                    symbol=sel
                )

                # --- NEW: HMA BUY/SELL signal aligned with slope direction ---
                hma_signal_text = ""
                if (show_hma and not hma_h.dropna().empty
                        and np.isfinite(slope_for_sign)):
                    last_cross = detect_last_crossover(hc, hma_h)
                    if last_cross is not None:
                        ts_cross = last_cross.get("time")
                        if ts_cross is not None and ts_cross in hc.index:
                            price_cross = float(hc.loc[ts_cross])
                            # Direction from slope, not raw cross side
                            side_from_slope = "BUY" if slope_for_sign >= 0 else "SELL"
                            conf_val = 1.0  # treat as 100% for display; gated by hma_conf
                            if conf_val >= hma_conf:
                                annotate_crossover(
                                    ax2, ts_cross, price_cross,
                                    side_from_slope, conf_val
                                )
                                dir_word = "up" if side_from_slope == "BUY" else "down"
                                hma_signal_text = (
                                    f"HMA {side_from_slope} @{fmt_price_val(price_cross)} — "
                                    f"price crossed HMA({hma_period}) with "
                                    f"P({dir_word})={conf_val*100:.1f}% ≥ {hma_conf*100:.1f}%"
                                )

                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = (float(bb_pctb_h.dropna().iloc[-1])
                                    if show_bbands else np.nan)
                        last_nbb = (float(bb_nbb_h.dropna().iloc[-1])
                                    if show_bbands else np.nan)
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = (
                                f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                            )
                    except Exception:
                        pass
                    ax2.text(
                        0.99, 0.02,
                        f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                        transform=ax2.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="grey", alpha=0.7)
                    )

                if not st_line_intr.dropna().empty:
                    ax2.plot(
                        st_line_intr.index, st_line_intr.values, "-",
                        label=f"Supertrend ({atr_period},{atr_mult})"
                    )
                if not yhat_h.empty:
                    ax2.plot(
                        yhat_h.index, yhat_h.values, "-",
                        linewidth=2,
                        label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)"
                    )
                if not upper_h.empty and not lower_h.empty:
                    # thicker ±2σ band lines
                    ax2.plot(
                        upper_h.index, upper_h.values, "--",
                        linewidth=2.0, label="Slope +2σ"
                    )
                    ax2.plot(
                        lower_h.index, lower_h.values, "--",
                        linewidth=2.0, label="Slope -2σ"
                    )

                ax2.text(
                    0.01, 0.02,
                    f"Slope: {fmt_slope(slope_for_sign)}/bar",
                    transform=ax2.transAxes, ha="left", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="grey", alpha=0.7)
                )
                ax2.text(
                    0.50, 0.02,
                    f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
                    transform=ax2.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="grey", alpha=0.7)
                )

                # NEW: title that includes slope-aligned trade instruction and HMA signal (if any)
                title_txt = (
                    f"{sel} Intraday ({st.session_state.hour_range})  "
                    f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)} — {instr_txt}"
                )
                if hma_signal_text:
                    title_txt += f"\n{hma_signal_text}"
                ax2.set_title(title_txt)

                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess = compute_session_lines(hc.index)
                    draw_session_lines(ax2, sess)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(
                            y, xmin=hc.index[0], xmax=hc.index[-1],
                            linestyles="dotted", linewidth=1
                        )
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                xlim_price = ax2.get_xlim()
                st.pyplot(fig2)
                # Volume panel
                vol = _coerce_1d_series(
                    intraday.get("Volume", pd.Series(index=hc.index))
                ).reindex(hc.index).astype(float)
                if _has_volume_to_plot(vol):
                    v_mid = rolling_midline(
                        vol, window=max(3, int(slope_lb_hourly))
                    )
                    v_trend, v_m = slope_line(vol, slope_lb_hourly)
                    v_r2 = regression_r2(vol, slope_lb_hourly)

                    fig2v, ax2v = plt.subplots(figsize=(14, 2.8))
                    ax2v.set_title(
                        f"Volume (Hourly) — Mid-line & Trend  |  Slope={fmt_slope(v_m)}/bar"
                    )
                    ax2v.fill_between(
                        vol.index, 0, vol,
                        alpha=0.18, label="Volume", color="tab:blue"
                    )
                    ax2v.plot(vol.index, vol, linewidth=1.0, color="tab:blue")
                    ax2v.plot(
                        v_mid.index, v_mid, ":", linewidth=1.6,
                        label=f"Mid-line ({slope_lb_hourly}-roll)"
                    )
                    if not v_trend.empty:
                        ax2v.plot(
                            v_trend.index, v_trend.values, "--",
                            linewidth=2,
                            label=f"Trend {slope_lb_hourly} ({fmt_slope(v_m)}/bar)"
                        )
                    ax2v.text(
                        0.01, 0.02,
                        f"Slope: {fmt_slope(v_m)}/bar",
                        transform=ax2v.transAxes, ha="left", va="bottom",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="grey", alpha=0.7)
                    )
                    ax2v.text(
                        0.50, 0.02,
                        f"R² ({slope_lb_hourly} bars): {fmt_r2(v_r2)}",
                        transform=ax2v.transAxes, ha="center", va="bottom",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="grey", alpha=0.7)
                    )
                    ax2v.set_xlim(xlim_price)
                    ax2v.set_xlabel("Time (PST)")
                    ax2v.legend(loc="lower left", framealpha=0.5)
                    st.pyplot(fig2v)

                # Hourly Indicator Panel — NTD + NPX + Triangles by Trend + Stars
                if show_nrsi:
                    ntd_h = compute_normalized_trend(hc, window=ntd_window)
                    ntd_trend_h, ntd_m_h = slope_line(ntd_h, slope_lb_hourly)
                    npx_h = (compute_normalized_price(hc, window=ntd_window)
                             if show_npx_ntd
                             else pd.Series(index=hc.index, dtype=float))

                    fig2r, ax2r = plt.subplots(figsize=(14, 2.8))
                    ax2r.set_title(
                        f"Hourly Indicator Panel — NTD + NPX + Trend (win={ntd_window})"
                    )
                    if shade_ntd and not ntd_h.dropna().empty:
                        shade_ntd_regions(ax2r, ntd_h)
                    if show_ntd_channel and np.isfinite(res_val) and np.isfinite(sup_val):
                        overlay_inrange_on_ntd(ax2r, hc, sup_h, res_h)

                    ax2r.plot(
                        ntd_h.index, ntd_h, "-",
                        linewidth=1.6, label="NTD"
                    )

                    # Triangles gated by price slope sign
                    overlay_ntd_triangles_by_trend(
                        ax2r, ntd_h, trend_slope=m_h,
                        upper=0.75, lower=-0.75
                    )

                    # Stars for S/R reversals
                    overlay_ntd_sr_reversal_stars(
                        ax2r, price=hc, sup=sup_h, res=res_h,
                        trend_slope=m_h, ntd=ntd_h,
                        prox=sr_prox_pct, bars_confirm=rev_bars_confirm
                    )

                    if show_npx_ntd and (not npx_h.dropna().empty
                                         and not ntd_h.dropna().empty):
                        overlay_npx_on_ntd(
                            ax2r, npx_h, ntd_h, mark_crosses=mark_npx_cross
                        )
                    if not ntd_trend_h.empty:
                        ax2r.plot(
                            ntd_trend_h.index, ntd_trend_h.values, "--",
                            linewidth=2,
                            label=f"NTD Trend {slope_lb_hourly} ({fmt_slope(ntd_m_h)}/bar)"
                        )
                    if show_hma_rev_ntd and (not hma_h.dropna().empty
                                             and not hc.dropna().empty):
                        overlay_hma_reversal_on_ntd(
                            ax2r, hc, hma_h, lookback=hma_rev_lb,
                            period=hma_period, ntd=ntd_h
                        )

                    for yv, lab, lw, col in [
                        (0.0,  "0.00", 1.0, "black"),
                        (0.75, "+0.75",1.0, "black"),
                        (-0.75,"-0.75",1.0, "black")
                    ]:
                        ax2r.axhline(
                            yv,
                            linestyle="--" if yv == 0.0 else "-",
                            linewidth=lw, color=col, label=lab
                        )
                    ax2r.set_ylim(-1.1, 1.1)
                    ax2r.set_xlim(xlim_price)
                    ax2r.legend(loc="lower left", framealpha=0.5)
                    ax2r.set_xlabel("Time (PST)")
                    st.pyplot(fig2r)

                # Momentum panel (ROC%)
                if show_mom_hourly:
                    roc = compute_roc(hc, n=mom_lb_hourly)
                    res_m = roc.rolling(60, min_periods=1).max()
                    sup_m = roc.rolling(60, min_periods=1).min()

                    fig2m, ax2m = plt.subplots(figsize=(14, 2.8))
                    ax2m.set_title(
                        f"Momentum (ROC% over {mom_lb_hourly} bars)"
                    )
                    ax2m.plot(roc.index, roc,
                              label=f"ROC%({mom_lb_hourly})")
                    yhat_m, m_m = slope_line(roc, slope_lb_hourly)
                    if not yhat_m.empty:
                        ax2m.plot(
                            yhat_m.index, yhat_m.values, "--",
                            linewidth=2,
                            label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m)}%/bar)"
                        )
                    ax2m.plot(res_m.index, res_m, ":",
                              label="Mom Resistance")
                    ax2m.plot(sup_m.index, sup_m, ":",
                              label="Mom Support")
                    ax2m.axhline(0, linestyle="--", linewidth=1)
                    ax2m.set_xlabel("Time (PST)")
                    ax2m.legend(loc="lower left", framealpha=0.5)
                    ax2m.set_xlim(xlim_price)
                    st.pyplot(fig2m)

        # News table
        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(
                    show_cols[["time","publisher","title","link"]]
                    .reset_index(drop=True),
                    use_container_width=True
                )

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:, 0],
            "Upper":    st.session_state.fc_ci.iloc[:, 1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = _safe_last_float(df)
        p_up = (np.mean(vals.to_numpy() > last_price)
                if np.isfinite(last_price) else np.nan)
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        st.caption(
            f"Intraday lookback: **{st.session_state.get('hour_range','24h')}**"
        )

        view = st.radio("View:", ["Daily","Intraday","Both"],
                        key="enh_view")

        if view in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()

            # NEW: regression-based daily trend with band + R² (Enhanced tab)
            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(
                df, slope_lb_daily
            )

            ntd_d2 = (compute_normalized_trend(df, window=ntd_window)
                      if show_ntd else pd.Series(index=df.index, dtype=float))
            npx_d2_full = (compute_normalized_price(df, window=ntd_window)
                           if show_npx_ntd else pd.Series(index=df.index, dtype=float))

            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(
                    df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                    conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
                    shift_cloud=False
                )
                kijun_d2 = kijun_d2.ffill().bfill()

            bb_mid_d2, bb_up_d2, bb_lo_d2 = compute_bbands(
                df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
            )[:3]

            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res30_show = res30.reindex(df_show.index)
            sup30_show = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            ntd_d_show = ntd_d2.reindex(df_show.index)
            npx_d2_show = npx_d2_full.reindex(df_show.index)
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()
            bb_mid_d2_show = bb_mid_d2.reindex(df_show.index)
            bb_up_d2_show  = bb_up_d2.reindex(df_show.index)
            bb_lo_d2_show  = bb_lo_d2.reindex(df_show.index)

            hma_d2_full = compute_hma(df, period=hma_period)
            hma_d2_show = hma_d2_full.reindex(df_show.index)

            fig, (ax, axdw2) = plt.subplots(
                2, 1, sharex=True, figsize=(14, 8),
                gridspec_kw={"height_ratios": [3.2, 1.3]}
            )
            plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93)
            ax.set_title(
                f"{st.session_state.ticker} Daily — {daily_view} — History, 30 EMA, 30 S/R, Slope"
            )
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if show_hma and not hma_d2_show.dropna().empty:
                ax.plot(
                    hma_d2_show.index, hma_d2_show.values, "-",
                    linewidth=1.6, label=f"HMA({hma_period})"
                )
            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(
                    kijun_d2_show.index, kijun_d2_show.values, "-",
                    linewidth=1.8, color="black",
                    label=f"Ichimoku Kijun ({ichi_base})"
                )
            if show_bbands and (not bb_up_d2_show.dropna().empty
                                and not bb_lo_d2_show.dropna().empty):
                ax.fill_between(
                    df_show.index, bb_lo_d2_show, bb_up_d2_show,
                    alpha=0.06, label=f"BB (×{bb_mult:.1f})"
                )
                ax.plot(
                    bb_mid_d2_show.index, bb_mid_d2_show.values, "-",
                    linewidth=1.1,
                    label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})"
                )
                ax.plot(bb_up_d2_show.index, bb_up_d2_show.values, ":",
                        linewidth=1.0)
                ax.plot(bb_lo_d2_show.index, bb_lo_d2_show.values, ":",
                        linewidth=1.0)
            if not yhat_d_show.empty:
                ax.plot(
                    yhat_d_show.index, yhat_d_show.values, "-",
                    linewidth=2,
                    label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)"
                )
            if not upper_d_show.empty and not lower_d_show.empty:
                # thicker ±2σ band lines
                ax.plot(
                    upper_d_show.index, upper_d_show.values, "--",
                    linewidth=2.0, label="Daily Trend +2σ"
                )
                ax.plot(
                    lower_d_show.index, lower_d_show.values, "--",
                    linewidth=2.0, label="Daily Trend -2σ"
                )
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")
            ax.set_ylabel("Price")
            ax.text(
                0.50, 0.02,
                f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="grey", alpha=0.7)
            )
            ax.legend(loc="lower left", framealpha=0.5)

            # Indicator panel (Daily)
            axdw2.set_title("Daily Indicator Panel — NTD + NPX + Trend")
            if show_ntd and shade_ntd and not ntd_d_show.dropna().empty:
                shade_ntd_regions(axdw2, ntd_d_show)
            if show_ntd and not ntd_d_show.dropna().empty:
                axdw2.plot(
                    ntd_d_show.index, ntd_d_show, "-",
                    linewidth=1.6, label=f"NTD (win={ntd_window})"
                )
                ntd_trend_d2, ntd_m_d2 = slope_line(
                    ntd_d_show, slope_lb_daily
                )
                if not ntd_trend_d2.empty:
                    axdw2.plot(
                        ntd_trend_d2.index, ntd_trend_d2.values, "--",
                        linewidth=2,
                        label=f"NTD Trend {slope_lb_daily} ({fmt_slope(ntd_m_d2)}/bar)"
                    )
                overlay_ntd_triangles_by_trend(
                    axdw2, ntd_d_show, trend_slope=m_d,
                    upper=0.75, lower=-0.75
                )
                overlay_ntd_sr_reversal_stars(
                    axdw2, price=df_show, sup=sup30_show, res=res30_show,
                    trend_slope=m_d, ntd=ntd_d_show,
                    prox=sr_prox_pct, bars_confirm=rev_bars_confirm
                )
            if show_npx_ntd and (not npx_d2_show.dropna().empty
                                 and not ntd_d_show.dropna().empty):
                overlay_npx_on_ntd(
                    axdw2, npx_d2_show, ntd_d_show, mark_crosses=mark_npx_cross
                )
            if show_hma_rev_ntd and (not hma_d2_show.dropna().empty
                                     and not df_show.dropna().empty):
                overlay_hma_reversal_on_ntd(
                    axdw2, df_show, hma_d2_show, lookback=hma_rev_lb,
                    period=hma_period, ntd=ntd_d_show
                )
            axdw2.axhline(0.0,  linestyle="--", linewidth=1.0,
                          color="black", label="0.00")
            axdw2.axhline(0.75, linestyle="-",  linewidth=1.0,
                          color="black", label="+0.75")
            axdw2.axhline(-0.75, linestyle="-",  linewidth=1.0,
                          color="black", label="-0.75")
            axdw2.set_ylim(-1.1, 1.1)
            axdw2.set_xlabel("Date (PST)")
            axdw2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                # Already rendered in Tab 1 intraday; logic unchanged here.
                st.info("Intraday view is rendered in Tab 1. (Logic unchanged here.)")

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
        p_up = (np.mean(vals.to_numpy() > last_price)
                if np.isfinite(last_price) else np.nan)
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        # NEW: 3M trend with band + R²
        trend3m, up3m, lo3m, m3m, r2_3m = regression_with_band(
            df3m, lookback=len(df3m)
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", label="Support")
        if not trend3m.empty:
            ax.plot(
                trend3m.index, trend3m.values, "--",
                label=f"Trend (m={fmt_slope(m3m)}/bar)"
            )
        if not up3m.empty and not lo3m.empty:
            # thicker ±2σ band lines
            ax.plot(
                up3m.index, up3m.values, ":",
                linewidth=2.0, label="Trend +2σ"
            )
            ax.plot(
                lo3m.index, lo3m.values, ":",
                linewidth=2.0, label="Trend -2σ"
            )
        ax.set_xlabel("Date (PST)")
        ax.text(
            0.50, 0.02,
            f"R² (3M): {fmt_r2(r2_3m)}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec="grey", alpha=0.7)
        )
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        res0 = df0['Close'].rolling(30, min_periods=1).max()
        sup0 = df0['Close'].rolling(30, min_periods=1).min()
        # NEW: full-period trend with band + R²
        trend0, up0, lo0, m0, r2_0 = regression_with_band(
            df0['Close'], lookback=len(df0)
        )

        fig0, ax0 = plt.subplots(figsize=(14, 5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(res0.index, res0, ":", label="Resistance")
        ax0.plot(sup0.index, sup0, ":", label="Support")
        if not trend0.empty:
            ax0.plot(
                trend0.index, trend0.values, "--",
                label=f"Trend (m={fmt_slope(m0)}/bar)"
            )
        if not up0.empty and not lo0.empty:
            # thicker ±2σ band lines
            ax0.plot(
                up0.index, up0.values, ":",
                linewidth=2.0, label="Trend +2σ"
            )
            ax0.plot(
                lo0.index, lo0.values, ":",
                linewidth=2.0, label="Trend -2σ"
            )
        ax0.set_xlabel("Date (PST)")
        ax0.text(
            0.50, 0.02,
            f"R² ({bb_period}): {fmt_r2(r2_0)}",
            transform=ax0.transAxes, ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec="grey", alpha=0.7)
        )
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()),
                     int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.5 Scanner (Price↑NTD Green Dot) ---
with tab5:
    st.header("NTD Green-Dot Scanner (Price↑NTD)")
    st.caption(
        "Scans the universe for symbols whose **latest NTD bar** prints a "
        "**green dot** on the NTD panel (NPX just crossed above NTD = Price↑NTD). "
        "You can also require that NTD on that bar is below a chosen level "
        "(e.g., -0.5). For Forex, the same logic is applied on hourly data."
    )

    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox(
        "Hourly lookback for Forex:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
        key="ntd_scan_hour_range"
    )
    scan_period = period_map[scan_hour_range]

    c1, c2 = st.columns(2)
    with c1:
        thresh = st.slider(
            "NTD filter level (signal NTD must be below this)",
            -1.0, 1.0, -0.5, 0.05, key="ntd_thresh"
        )
    with c2:
        run = st.button("Scan Universe", key="btn_ntd_scan")

    if run:
        # ---- DAILY: GREEN DOT (Price↑NTD) on latest bar ----
        daily_rows = []
        for sym in universe:
            has_sig, ntd_val, ts, npx_val, close_val = last_green_dot_daily(
                sym, ntd_window, level=thresh
            )
            daily_rows.append({
                "Symbol": sym,
                "HasSignal": has_sig,
                "NTD_Last": ntd_val,
                "NPX_Last": npx_val,
                "Close": close_val,
                "Timestamp": ts
            })
        df_daily = pd.DataFrame(daily_rows)
        hits_daily = df_daily[df_daily["HasSignal"] == True].copy()
        hits_daily = hits_daily.sort_values("NTD_Last")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(
            f"Daily green-dot signals (NTD < {thresh:+.2f})",
            int(hits_daily.shape[0])
        )

        st.subheader(
            f"Daily — 'Price↑NTD' green dot on latest bar (NTD < {thresh:+.2f})"
        )
        if hits_daily.empty:
            st.info(
                "No symbols where the latest **daily** NTD bar has a green dot "
                f"(Price↑NTD) with NTD below {thresh:+.2f}."
            )
        else:
            view = hits_daily.copy()
            view["NTD_Last"] = view["NTD_Last"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a"
            )
            view["NPX_Last"] = view["NPX_Last"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a"
            )
            view["Close"]    = view["Close"].map(
                lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
            )
            st.dataframe(
                view[["Symbol","Timestamp","Close","NTD_Last","NPX_Last"]]
                .reset_index(drop=True),
                use_container_width=True
            )

        # ---- DAILY: PRICE > KIJUN ----
        st.markdown("---")
        st.subheader(
            f"Daily — Price > Ichimoku Kijun({ichi_base}) (latest bar)"
        )
        above_rows = []
        for sym in universe:
            above, ts, close_now, kij_now = price_above_kijun_info_daily(
                sym, base=ichi_base
            )
            above_rows.append({
                "Symbol": sym,
                "AboveNow": above,
                "Timestamp": ts,
                "Close": close_now,
                "Kijun": kij_now
            })
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]
        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            view_above = df_above_daily.copy()
            view_above["Close"] = view_above["Close"].map(
                lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
            )
            view_above["Kijun"] = view_above["Kijun"].map(
                lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
            )
            st.dataframe(
                view_above[["Symbol","Timestamp","Close","Kijun"]]
                .reset_index(drop=True),
                use_container_width=True
            )

        # ---- FOREX HOURLY: GREEN DOT (Price↑NTD) on latest bar ----
        if mode == "Forex":
            st.markdown("---")
            st.subheader(
                f"Forex Hourly — 'Price↑NTD' green dot on latest bar "
                f"({scan_hour_range} lookback, NTD < {thresh:+.2f})"
            )
            hourly_rows = []
            for sym in universe:
                has_sig_h, ntd_val_h, ts_h, npx_val_h, close_val_h = (
                    last_green_dot_hourly(
                        sym, ntd_window, period=scan_period, level=thresh
                    )
                )
                hourly_rows.append({
                    "Symbol": sym,
                    "HasSignal": has_sig_h,
                    "NTD_Last": ntd_val_h,
                    "NPX_Last": npx_val_h,
                    "Close": close_val_h,
                    "Timestamp": ts_h
                })
            df_hour = pd.DataFrame(hourly_rows)
            hits_hour = df_hour[df_hour["HasSignal"] == True].copy()
            hits_hour = hits_hour.sort_values("NTD_Last")

            if hits_hour.empty:
                st.info(
                    f"No Forex pairs where the latest **hourly** NTD bar "
                    f"has a green dot (Price↑NTD) within {scan_hour_range} "
                    f"lookback and NTD below {thresh:+.2f}."
                )
            else:
                showh = hits_hour.copy()
                showh["NTD_Last"] = showh["NTD_Last"].map(
                    lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a"
                )
                showh["NPX_Last"] = showh["NPX_Last"].map(
                    lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a"
                )
                showh["Close"]    = showh["Close"].map(
                    lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
                )
                st.dataframe(
                    showh[["Symbol","Timestamp","Close","NTD_Last","NPX_Last"]]
                    .reset_index(drop=True),
                    use_container_width=True
                )

            # ---- FOREX HOURLY: PRICE > KIJUN ----
            st.subheader(
                f"Forex Hourly — Price > Ichimoku Kijun({ichi_base}) "
                f"(latest bar, {scan_hour_range})"
            )
            habove_rows = []
            for sym in universe:
                above_h, ts_h, close_h, kij_h = price_above_kijun_info_hourly(
                    sym, period=scan_period, base=ichi_base
                )
                habove_rows.append({
                    "Symbol": sym,
                    "AboveNow": above_h,
                    "Timestamp": ts_h,
                    "Close": close_h,
                    "Kijun": kij_h
                })
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]
            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(
                    lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
                )
                vch["Kijun"] = vch["Kijun"].map(
                    lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a"
                )
                st.dataframe(
                    vch[["Symbol","Timestamp","Close","Kijun"]]
                    .reset_index(drop=True),
                    use_container_width=True
                )

# --- Tab 6: Long-Term History ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox(
        "Ticker:", universe, index=default_idx, key="hist_long_ticker"
    )
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("5Y", key="btn_5y"):
        st.session_state.hist_years = 5
    if c2.button("10Y", key="btn_10y"):
        st.session_state.hist_years = 10
    if c3.button("15Y", key="btn_15y"):
        st.session_state.hist_years = 15
    if c4.button("20Y", key="btn_20y"):
        st.session_state.hist_years = 20

    years = int(st.session_state.hist_years)
    st.caption(
        f"Showing last **{years} years**. "
        "Support/Resistance = rolling **252-day** extremes; "
        "trendline fits the shown window."
    )

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
            res_last = float(res_roll.iloc[-1]) if len(res_roll) else np.nan
            sup_last = float(sup_roll.iloc[-1]) if len(sup_roll) else np.nan
            # NEW: long-term trend with ±2σ band + R²
            yhat_all, upper_all, lower_all, m_all, r2_all = regression_with_band(
                s, lookback=len(s)
            )

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.set_title(
                f"{sym} — Last {years} Years — Price + 252d S/R + Trend"
            )
            ax.plot(s.index, s.values, label="Close")
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(
                    res_last, xmin=s.index[0], xmax=s.index[-1],
                    colors="tab:red",   linestyles="-",
                    linewidth=1.6, label="Resistance (252d)"
                )
                ax.hlines(
                    sup_last, xmin=s.index[0], xmax=s.index[-1],
                    colors="tab:green", linestyles="-",
                    linewidth=1.6, label="Support (252d)"
                )
                label_on_left(ax, res_last,
                              f"R {fmt_price_val(res_last)}",
                              color="tab:red")
                label_on_left(ax, sup_last,
                              f"S {fmt_price_val(sup_last)}",
                              color="tab:green")
            if not yhat_all.empty:
                ax.plot(
                    yhat_all.index, yhat_all.values, "--",
                    linewidth=2,
                    label=f"Trend (m={fmt_slope(m_all)}/bar)"
                )
            if not upper_all.empty and not lower_all.empty:
                # thicker ±2σ band lines
                ax.plot(
                    upper_all.index, upper_all.values, ":",
                    linewidth=2.0, label="Trend +2σ"
                )
                ax.plot(
                    lower_all.index, lower_all.values, ":",
                    linewidth=2.0, label="Trend -2σ"
                )
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(
                    0.99, 0.02,
                    f"Current price: {fmt_price_val(px_now)}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec="grey", alpha=0.7)
                )
            ax.text(
                0.01, 0.02,
                f"Slope: {fmt_slope(m_all)}/bar",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="grey", alpha=0.7)
            )
            ax.text(
                0.50, 0.02,
                f"R² (trend): {fmt_r2(r2_all)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="grey", alpha=0.7)
            )
            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)
