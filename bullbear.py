# bullbear.py — Stocks/Forex Dashboard + Forecasts (Complete, all tabs)
# FIXED: NameError on pytz — imports come FIRST, then timezone constants.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.transforms import blended_transform_factory

# SARIMAX (optional; app falls back if it errors)
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SARIMAX = True
except Exception:
    HAS_SARIMAX = False

# ---------------- Timezones (defined AFTER importing pytz) ----------------
PACIFIC = pytz.timezone("US/Pacific")
NY_TZ   = pytz.timezone("America/New_York")
LDN_TZ  = pytz.timezone("Europe/London")

# ---------------- Streamlit config ----------------
st.set_page_config(page_title="📊 BullBear Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  .stPlotlyChart, .stMarkdown { z-index: 0 !important; }
  .legend { z-index: 1 !important; }
  @media (max-width: 600px) {
    .css-18e3th9 { transform: none !important; visibility: visible !important;
                   width: 100% !important; position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

# ---------------- Auto refresh ----------------
REFRESH_INTERVAL = 120  # seconds
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

top_warn = st.empty()

# ---------------- Sidebar ----------------
st.sidebar.title("Configuration")

mode = st.sidebar.selectbox("Mode", ["Stock", "Forex"], index=0)

bb_period = st.sidebar.selectbox("Bull/Bear Lookback", ["1mo", "3mo", "6mo", "1y"], index=2)
daily_view = st.sidebar.selectbox("Daily view range", ["Historical", "6M", "12M", "24M"], index=2)

st.sidebar.subheader("Local slope lookbacks")
slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10)
slope_lb_hourly = st.sidebar.slider("Intraday slope lookback (bars)", 12, 480, 120, 6)

st.sidebar.subheader("Intraday window")
hour_range = st.sidebar.selectbox("Intraday range", ["6h", "24h", "5d"], index=1)

st.sidebar.subheader("Support/Resistance")
sr_lb_hourly = st.sidebar.slider("Intraday S/R lookback (bars)", 20, 240, 60, 5)
sr_lb_daily  = st.sidebar.slider("Daily S/R lookback (bars)", 10, 200, 40, 5)
sr_prox_pct  = st.sidebar.slider("Proximity (%)", 0.05, 1.00, 0.25, 0.05) / 100.0

st.sidebar.subheader("Bollinger (Price)")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
bb_win      = st.sidebar.slider("BB window", 5, 120, 20, 1)
bb_mult     = st.sidebar.slider("BB σ multiplier", 1.0, 4.0, 2.0, 0.1)
bb_use_ema  = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False)

st.sidebar.subheader("HMA (Price)")
show_hma   = st.sidebar.checkbox("Show HMA", value=True)
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1)

st.sidebar.subheader("Ichimoku (Kijun)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=True)
ichi_base = st.sidebar.slider("Kijun period", 20, 40, 26, 1)

st.sidebar.subheader("Supertrend (Intraday)")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1)
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5)

st.sidebar.subheader("Parabolic SAR (Intraday)")
show_psar = st.sidebar.checkbox("Show PSAR", value=True)
psar_step = st.sidebar.slider("PSAR step", 0.01, 0.20, 0.02, 0.01)
psar_max  = st.sidebar.slider("PSAR max", 0.10, 1.00, 0.20, 0.10)

st.sidebar.subheader("MACD (Normalized overlay)")
show_macd   = st.sidebar.checkbox("Show MACD overlay", value=True)
macd_fast   = int(st.sidebar.number_input("MACD fast", 2, 100, 12, 1))
macd_slow   = int(st.sidebar.number_input("MACD slow", 3, 200, 26, 1))
macd_signal = int(st.sidebar.number_input("MACD signal", 1, 100, 9, 1))

st.sidebar.subheader("Signals")
rev_bars_confirm = st.sidebar.slider("Confirm bars (reversal)", 1, 4, 2, 1)

st.sidebar.subheader("Forex extras")
show_sessions_pst = False
show_fx_news = False
news_window_days = 7
if mode == "Forex":
    show_sessions_pst = st.sidebar.checkbox("Show London/NY sessions (PST)", value=True)
    show_fx_news = st.sidebar.checkbox("Show news markers", value=True)
    news_window_days = st.sidebar.slider("News window (days)", 1, 14, 7)

st.sidebar.subheader("Scanners")
ntd_window = st.sidebar.slider("NTD window (bars)", 10, 300, 60, 5)
run_scans = st.sidebar.button("Run scanners now")

# ---------------- Universe ----------------
if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','JPM','PLTR','NVDA','META','GOOG',
        'MSFT','TSM','NFLX','AMD','SMCI','ORCL','TLT','VOO','AVGO','BABA'
    ])
else:
    universe = [
        'EURUSD=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'USDCAD=X','USDCHF=X','EURJPY=X','GBPJPY=X','AUDJPY=X'
    ]

ticker = st.sidebar.selectbox("Symbol", universe, index=0)

# ---------------- Formatting helpers ----------------
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

def fmt_price_val(y: float) -> str:
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.4f}" if np.isfinite(y) else "n/a"

def fmt_slope(m: float) -> str:
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.6f}" if np.isfinite(mv) else "n/a"

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
    return f"Δ {diff:.4f}"

# ---------------- Data fetch (cached) ----------------
def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
    # yfinance can return multi-index columns for some tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    out = df[cols].copy()
    out = out.dropna(subset=["Close"])
    return out

@st.cache_data(ttl=180, show_spinner=False)
def fetch_daily_ohlc(symbol: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    df = _normalize_yf_df(df)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # localize/convert to PACIFIC
    if df.index.tz is None:
        df.index = df.index.tz_localize(PACIFIC)
    else:
        df.index = df.index.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=180, show_spinner=False)
def fetch_daily_close(symbol: str, period: str = "5y") -> pd.Series:
    df = fetch_daily_ohlc(symbol, period=period)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].copy()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_intraday_ohlc(symbol: str, window: str) -> pd.DataFrame:
    if window == "6h":
        period, interval = "1d", "5m"
    elif window == "24h":
        period, interval = "2d", "5m"
    else:
        period, interval = "5d", "15m"
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    df = _normalize_yf_df(df)
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # yfinance intraday usually UTC-naive; assume UTC then convert
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(PACIFIC)
    else:
        df.index = df.index.tz_convert(PACIFIC)
    return df

# ---------------- Regression helpers ----------------
def subset_by_daily_view(df: pd.DataFrame, view_label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    end = df.index.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        return df
    start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return df.loc[(df.index >= start) & (df.index <= end)]

def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
    if s.shape[0] < 3:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty, float("nan"), float("nan"), float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    std = float(np.sqrt(np.sum(resid**2) / dof))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), float(r2), float(std)

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

# ---------------- Plot helpers ----------------
def _simplify_axes(ax):
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=9)

def draw_top_badges(ax, badges: list):
    if not badges:
        return
    y = 1.02
    for text, color in badges:
        ax.text(0.01, y, text,
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=9, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95))
        y += 0.055

def draw_instruction_ribbon(ax, local_slope: float, sup: float, res: float, px: float, symbol: str,
                           global_slope: float = None):
    def _finite(x):
        try: return np.isfinite(float(x))
        except Exception: return False

    aligned = True
    if global_slope is not None and _finite(local_slope) and _finite(global_slope):
        aligned = (float(local_slope) * float(global_slope)) > 0
    elif global_slope is not None:
        aligned = False

    if not aligned:
        ax.text(0.5, 1.08,
                "ALERT: Global Trendline and Local Slope are opposing — no trade instruction.",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.95))
        return

    color = "tab:green" if (_finite(local_slope) and float(local_slope) > 0) else "tab:red"
    buy_txt  = f"▲ BUY @{fmt_price_val(sup)}"
    sell_txt = f"▼ SELL @{fmt_price_val(res)}"
    pips_txt = f" • Value of PIPS: {_diff_text(res, sup, symbol)}"
    if _finite(local_slope) and float(local_slope) > 0:
        msg = f"{buy_txt} → {sell_txt}{pips_txt}"
    else:
        msg = f"{sell_txt} → {buy_txt}{pips_txt}"

    ax.text(0.5, 1.08, msg,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95))

# ---------------- Intraday compressed x-axis ----------------
def make_market_time_formatter(index: pd.DatetimeIndex) -> FuncFormatter:
    def _fmt(x, _pos=None):
        i = int(round(x))
        if 0 <= i < len(index):
            ts = index[i]
            return ts.strftime("%m-%d %H:%M")
        return ""
    return FuncFormatter(_fmt)

def market_time_axis(ax, index: pd.DatetimeIndex):
    ax.set_xlim(0, max(0, len(index) - 1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(make_market_time_formatter(index))

# ---------------- Forecast ----------------
@st.cache_data(ttl=180, show_spinner=False)
def compute_forecast_sarimax(close: pd.Series, steps: int = 30):
    s = _coerce_1d_series(close).dropna()
    if s.empty or len(s) < 10 or not HAS_SARIMAX:
        # fallback: naive drift
        if s.empty:
            idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1),
                                periods=steps, freq="D", tz=PACIFIC)
            return idx, pd.Series(np.nan, index=idx), pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=idx)
        drift = (s.iloc[-1] - s.iloc[0]) / max(len(s)-1, 1)
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D", tz=PACIFIC)
        mean = pd.Series(s.iloc[-1] + drift*np.arange(1, steps+1), index=idx)
        ci = pd.DataFrame({"lower": mean*0.98, "upper": mean*1.02}, index=idx)
        return idx, mean, ci

    try:
        model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = model.get_forecast(steps=steps)
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D", tz=PACIFIC)
        mean = fc.predicted_mean
        mean.index = idx
        ci = fc.conf_int()
        ci.index = idx
        ci.columns = ["lower", "upper"] if len(ci.columns) >= 2 else ci.columns
        if "lower" not in ci.columns or "upper" not in ci.columns:
            # best effort
            ci = pd.DataFrame({"lower": mean*0.98, "upper": mean*1.02}, index=idx)
        return idx, mean, ci[["lower","upper"]]
    except Exception:
        # fallback drift
        drift = (s.iloc[-1] - s.iloc[0]) / max(len(s)-1, 1)
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D", tz=PACIFIC)
        mean = pd.Series(s.iloc[-1] + drift*np.arange(1, steps+1), index=idx)
        ci = pd.DataFrame({"lower": mean*0.98, "upper": mean*1.02}, index=idx)
        return idx, mean, ci
# ---------------- Indicators ----------------
def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty or window < 2 or not np.isfinite(mult):
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty
    minp = max(2, window // 2)
    mid = s.rolling(window, min_periods=minp).mean() if not use_ema else s.ewm(span=window, adjust=False).mean()
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower

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
    diff = 2*wma_half - wma_full
    hma = _wma(diff, sqrtp)
    return hma

def ichimoku_kijun(high: pd.Series, low: pd.Series, base: int = 26) -> pd.Series:
    H = _coerce_1d_series(high)
    L = _coerce_1d_series(low)
    if H.empty or L.empty:
        idx = H.index if not H.empty else L.index
        return pd.Series(index=idx, dtype=float)
    kijun = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
    return kijun

# --- ATR / Supertrend ---
def _true_range(df: pd.DataFrame):
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = _true_range(df[["High","Low","Close"]])
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["ST","in_uptrend","upperband","lowerband"])
    ohlc = df[["High","Low","Close"]].copy()
    hl2 = (ohlc["High"] + ohlc["Low"]) / 2.0
    atr = compute_atr(ohlc, atr_period)
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr
    st = pd.Series(index=ohlc.index, dtype=float)
    in_up = pd.Series(index=ohlc.index, dtype=bool)

    st.iloc[0] = upperband.iloc[0]
    in_up.iloc[0] = True
    for i in range(1, len(ohlc)):
        prev_st = st.iloc[i-1]
        prev_up = in_up.iloc[i-1]
        up_i = min(upperband.iloc[i], prev_st) if prev_up else upperband.iloc[i]
        dn_i = max(lowerband.iloc[i], prev_st) if (not prev_up) else lowerband.iloc[i]
        close_i = ohlc["Close"].iloc[i]

        if close_i > up_i:
            curr_up = True
        elif close_i < dn_i:
            curr_up = False
        else:
            curr_up = prev_up

        in_up.iloc[i] = curr_up
        st.iloc[i] = dn_i if curr_up else up_i

    return pd.DataFrame({"ST": st, "in_uptrend": in_up, "upperband": upperband, "lowerband": lowerband})

# --- PSAR ---
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
            psar[i] = prev_psar + af*(ep - prev_psar)
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
            psar[i] = prev_psar + af*(ep - prev_psar)
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

# --- MACD (normalized overlay) ---
def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty:
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty
    fast = max(2, int(fast))
    slow = max(3, int(slow))
    signal = max(1, int(signal))
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def macd_to_price_overlay(close: pd.Series, macd: pd.Series, scale_frac: float = 0.12):
    """
    Map MACD into the price axis (non-blocking overlay).
    """
    c = _coerce_1d_series(close).astype(float)
    m = _coerce_1d_series(macd).astype(float).reindex(c.index)
    mask = c.notna() & m.notna()
    if mask.sum() < 10:
        return pd.Series(index=c.index, dtype=float)
    c2 = c[mask]
    m2 = m[mask]
    pr = float(c2.max() - c2.min())
    pr = pr if np.isfinite(pr) and pr > 0 else 1.0
    mr = float(m2.max() - m2.min())
    mr = mr if np.isfinite(mr) and mr > 0 else 1.0
    scaled = (m - float(m2.mean())) * (pr * scale_frac / mr) + float(c2.mean())
    return scaled

# ---------------- Signals ----------------
def last_band_reversal_signal(price: pd.Series,
                              band_upper: pd.Series,
                              band_lower: pd.Series,
                              local_slope: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 2):
    """
    Only the latest signal:
      • Uptrend  → BUY when recent bar touched lower band and next bars rise
      • Downtrend→ SELL when recent bar touched upper band and next bars fall
    """
    p = _coerce_1d_series(price).dropna()
    if len(p) < confirm_bars + 3 or not np.isfinite(local_slope) or local_slope == 0:
        return None
    u = _coerce_1d_series(band_upper).reindex(p.index)
    l = _coerce_1d_series(band_lower).reindex(p.index)

    def _inc_after(i):
        if i + confirm_bars >= len(p): return False
        seg = p.iloc[i:(i+confirm_bars+1)]
        return bool(np.all(np.diff(seg) > 0))

    def _dec_after(i):
        if i + confirm_bars >= len(p): return False
        seg = p.iloc[i:(i+confirm_bars+1)]
        return bool(np.all(np.diff(seg) < 0))

    for i in range(len(p) - confirm_bars - 2, -1, -1):
        pc = float(p.iloc[i])
        if local_slope > 0:
            lo = float(l.iloc[i]) if np.isfinite(l.iloc[i]) else np.nan
            if np.isfinite(lo) and pc <= lo*(1+prox) and _inc_after(i):
                t = p.index[i+confirm_bars]
                return {"time": t, "price": float(p.loc[t]), "side": "BUY", "note": "Band REV"}
        else:
            up = float(u.iloc[i]) if np.isfinite(u.iloc[i]) else np.nan
            if np.isfinite(up) and pc >= up*(1-prox) and _dec_after(i):
                t = p.index[i+confirm_bars]
                return {"time": t, "price": float(p.loc[t]), "side": "SELL", "note": "Band REV"}
    return None

def last_hma_cross_signal(price: pd.Series, hma: pd.Series, local_slope: float, lookback: int = 60):
    """
    Latest bar crossover aligned with slope:
      • Up slope: BUY when price crosses ABOVE HMA
      • Down slope: SELL when price crosses BELOW HMA
    """
    p = _coerce_1d_series(price).astype(float)
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)
    mask = p.notna() & h.notna()
    if mask.sum() < 3 or not np.isfinite(local_slope) or local_slope == 0:
        return None
    p = p[mask]; h = h[mask]
    if len(p) > lookback:
        p = p.iloc[-lookback:]; h = h.iloc[-lookback:]
    up_cross = (p.shift(1) < h.shift(1)) & (p >= h)
    dn_cross = (p.shift(1) > h.shift(1)) & (p <= h)
    if local_slope > 0 and up_cross.any():
        t = up_cross[up_cross].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY", "note": "HMA Cross"}
    if local_slope < 0 and dn_cross.any():
        t = dn_cross[dn_cross].index[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL", "note": "HMA Cross"}
    return None

# ---------------- Sessions (Forex) ----------------
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

def map_times_to_positions(index: pd.DatetimeIndex, times: list):
    pos = []
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return pos
    for t in times:
        try:
            j = index.get_indexer([pd.Timestamp(t).tz_convert(index.tz)], method="nearest")[0]
        except Exception:
            j = index.get_indexer([pd.Timestamp(t)], method="nearest")[0]
        if j != -1:
            pos.append(j)
    return pos

def map_session_lines_to_positions(lines: dict, index: pd.DatetimeIndex):
    return {k: map_times_to_positions(index, v) for k, v in lines.items()}

# ---------------- News (best-effort) ----------------
@st.cache_data(ttl=180, show_spinner=False)
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
        except Exception:
            try:
                dt_utc = pd.to_datetime(ts, utc=True)
            except Exception:
                continue
        dt_pst = dt_utc.tz_convert(PACIFIC)
        rows.append({"time": dt_pst, "title": item.get("title",""), "publisher": item.get("publisher","")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cutoff = (pd.Timestamp.now(tz=PACIFIC) - pd.Timedelta(days=window_days))
    return df[df["time"] >= cutoff].sort_values("time")
# ---------------- Support / resistance ----------------
def rolling_support_resistance(close: pd.Series, lookback: int):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    sup = s.rolling(lookback, min_periods=max(3, lookback//3)).min()
    res = s.rolling(lookback, min_periods=max(3, lookback//3)).max()
    return sup, res

# ---------------- NTD (simple normalized trend direction) ----------------
def compute_ntd(close: pd.Series, window: int = 60) -> float:
    """
    Returns value in [-1, 1] (tanh scaled).
    Uses regression slope over `window`, normalized by price std.
    """
    s = _coerce_1d_series(close).dropna().astype(float)
    if len(s) < max(10, window//2):
        return float("nan")
    s = s.iloc[-window:]
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    vol = float(np.std(y)) if np.std(y) > 0 else 1.0
    raw = float(m / vol)  # unitless
    return float(np.tanh(raw * 6.0))

# ---------------- Daily chart ----------------
Z_FOR_99 = 2.576

def plot_daily(symbol: str, df_daily: pd.DataFrame):
    """
    Daily chart:
      - global trendline (thick)
      - local slope line + 99% band (for reversal tab) optional
      - BB, HMA, Kijun, MACD overlay
      - HMA cross marker (buy=black, sell=blue) aligned with local slope
    """
    if df_daily is None or df_daily.empty:
        st.warning("No daily data.")
        return

    close = df_daily["Close"].astype(float)
    high  = df_daily["High"].astype(float) if "High" in df_daily.columns else close
    low   = df_daily["Low"].astype(float) if "Low" in df_daily.columns else close

    # Global trendline slope over full visible range
    yhat_g, _, _, m_g, r2_g, _ = regression_with_band(close, lookback=0, z=2.0)

    # Local slope (dashed) over slope_lb_daily
    yhat_l, upper2, lower2, m_l, r2_l, _ = regression_with_band(close, lookback=slope_lb_daily, z=2.0)

    # Support/Resistance
    sup, res = rolling_support_resistance(close, sr_lb_daily)
    sup_v = float(sup.dropna().iloc[-1]) if sup.dropna().size else float(close.iloc[-1])
    res_v = float(res.dropna().iloc[-1]) if res.dropna().size else float(close.iloc[-1])

    # Indicators
    mid, bb_u, bb_l = compute_bbands(close, bb_win, bb_mult, bb_use_ema) if show_bbands else (None, None, None)
    hma = compute_hma(close, hma_period) if show_hma else None
    kijun = ichimoku_kijun(high, low, ichi_base) if show_ichi else None

    macd_line, macd_sig, macd_hist = compute_macd(close, macd_fast, macd_slow, macd_signal)
    macd_overlay = macd_to_price_overlay(close, macd_line) if show_macd else None

    # Signals
    hma_sig = last_hma_cross_signal(close, hma, m_l, lookback=90) if show_hma else None

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(close.index, close.values, linewidth=1.7, label="Close")

    # Global trend line (solid)
    if not yhat_g.empty and np.isfinite(m_g):
        ax.plot(yhat_g.index, yhat_g.values,
                linewidth=3.2,
                color=("tab:green" if m_g >= 0 else "tab:red"),
                label=f"Global Trend ({fmt_slope(m_g)}/bar)")

    # Local slope line (dashed)
    if not yhat_l.empty and np.isfinite(m_l):
        ax.plot(yhat_l.index, yhat_l.values,
                linewidth=2.3,
                linestyle="--",
                color=("tab:green" if m_l >= 0 else "tab:red"),
                label=f"Local Slope ({fmt_slope(m_l)}/bar)")

    # Bollinger
    if show_bbands and mid is not None and not mid.empty:
        ax.plot(mid.index, mid.values, linewidth=1.2, alpha=0.7, label="BB mid")
        ax.plot(bb_u.index, bb_u.values, linewidth=1.0, alpha=0.6, label="+BB")
        ax.plot(bb_l.index, bb_l.values, linewidth=1.0, alpha=0.6, label="-BB")

    # HMA
    if show_hma and hma is not None and not hma.empty:
        ax.plot(hma.index, hma.values, linewidth=1.5, alpha=0.85, label=f"HMA({hma_period})")

    # Kijun
    if show_ichi and kijun is not None and not kijun.empty:
        ax.plot(kijun.index, kijun.values, linewidth=1.3, alpha=0.8, label=f"Kijun({ichi_base})")

    # MACD overlay
    if show_macd and macd_overlay is not None and not macd_overlay.empty:
        ax.plot(macd_overlay.index, macd_overlay.values, linewidth=1.2, alpha=0.65, label="MACD overlay")

    # HMA cross marker (daily): Buy=black, Sell=blue
    badges = [
        (f"Daily local slope: {fmt_slope(m_l)} (R² {r2_l:.2f})", "black"),
        (f"Daily global slope: {fmt_slope(m_g)} (R² {r2_g:.2f})", "black"),
    ]
    if hma_sig:
        if hma_sig["side"] == "BUY":
            ax.scatter([hma_sig["time"]], [hma_sig["price"]], marker="*", s=180, c="black", zorder=10)
            badges.insert(0, ("★ Daily HMA BUY (black)", "black"))
        else:
            ax.scatter([hma_sig["time"]], [hma_sig["price"]], marker="*", s=180, c="tab:blue", zorder=10)
            badges.insert(0, ("★ Daily HMA SELL (blue)", "tab:blue"))

    draw_top_badges(ax, badges)

    # Instruction ribbon uses local slope, but only if aligned with global trend
    draw_instruction_ribbon(ax, m_l, sup_v, res_v, float(close.iloc[-1]), symbol, global_slope=m_g)

    _simplify_axes(ax)
    ax.set_title(f"{symbol} — Daily")
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)

# ---------------- Intraday chart (compressed axis) ----------------
def plot_intraday(symbol: str, df_i: pd.DataFrame, df_daily_for_global: pd.DataFrame):
    if df_i is None or df_i.empty:
        st.warning("No intraday data.")
        return

    close = df_i["Close"].astype(float)
    high  = df_i["High"].astype(float) if "High" in df_i.columns else close
    low   = df_i["Low"].astype(float) if "Low" in df_i.columns else close

    # Global slope from daily (sign only)
    m_g = float("nan")
    if df_daily_for_global is not None and not df_daily_for_global.empty:
        yhat_g, _, _, m_g, _, _ = regression_with_band(df_daily_for_global["Close"].astype(float), lookback=0, z=2.0)

    # Local slope band on intraday close (±2σ)
    yhat_l, up2, lo2, m_l, r2_l, _ = regression_with_band(close, lookback=min(slope_lb_hourly, len(close)), z=2.0)

    # Support/Resistance rolling
    sup, res = rolling_support_resistance(close, sr_lb_hourly)
    sup_v = float(sup.dropna().iloc[-1]) if sup.dropna().size else float(close.iloc[-1])
    res_v = float(res.dropna().iloc[-1]) if res.dropna().size else float(close.iloc[-1])

    # Indicators
    mid, bb_u, bb_l = compute_bbands(close, bb_win, bb_mult, bb_use_ema) if show_bbands else (None, None, None)
    hma = compute_hma(close, hma_period) if show_hma else None
    kijun = ichimoku_kijun(high, low, ichi_base) if show_ichi else None

    st_df = compute_supertrend(df_i, atr_period, atr_mult)
    psar_s, psar_up = (compute_parabolic_sar(high, low, psar_step, psar_max) if show_psar else (None, None))

    macd_line, _, _ = compute_macd(close, macd_fast, macd_slow, macd_signal)
    macd_overlay = macd_to_price_overlay(close, macd_line) if show_macd else None

    # Band reversal signal (latest only)
    band_sig = last_band_reversal_signal(close, up2, lo2, m_l, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)

    # HMA cross signal (latest)
    hma_sig = last_hma_cross_signal(close, hma, m_l, lookback=240) if show_hma else None

    # Compressed x
    x = np.arange(len(df_i), dtype=float)
    idx = df_i.index

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(x, close.values, linewidth=1.6, label="Close")

    # Local slope line & band
    if not yhat_l.empty and np.isfinite(m_l):
        # yhat_l is only last lookback; map to x positions for that slice
        lb = len(yhat_l)
        x_l = x[-lb:]
        ax.plot(x_l, yhat_l.values, linestyle="--",
                linewidth=2.3, color=("tab:green" if m_l >= 0 else "tab:red"),
                label=f"Local slope ({fmt_slope(m_l)}/bar)")
        ax.plot(x_l, up2.values, linewidth=1.0, alpha=0.55, label="+2σ")
        ax.plot(x_l, lo2.values, linewidth=1.0, alpha=0.55, label="-2σ")

    # Bollinger
    if show_bbands and mid is not None and not mid.empty:
        ax.plot(x, mid.values, linewidth=1.1, alpha=0.75, label="BB mid")
        ax.plot(x, bb_u.values, linewidth=1.0, alpha=0.6, label="+BB")
        ax.plot(x, bb_l.values, linewidth=1.0, alpha=0.6, label="-BB")

    # HMA
    if show_hma and hma is not None and not hma.empty:
        ax.plot(x, hma.values, linewidth=1.5, alpha=0.85, label=f"HMA({hma_period})")

    # Kijun
    if show_ichi and kijun is not None and not kijun.empty:
        ax.plot(x, kijun.values, linewidth=1.2, alpha=0.8, label=f"Kijun({ichi_base})")

    # Supertrend
    if st_df is not None and not st_df.empty and "ST" in st_df.columns:
        ax.plot(x, st_df["ST"].values, linewidth=1.2, alpha=0.7, label="Supertrend")

    # PSAR
    if show_psar and psar_s is not None and not psar_s.empty:
        ax.scatter(x, psar_s.reindex(idx).values, s=8, alpha=0.7, label="PSAR")

    # MACD overlay
    if show_macd and macd_overlay is not None and not macd_overlay.empty:
        ax.plot(x, macd_overlay.reindex(idx).values, linewidth=1.2, alpha=0.65, label="MACD overlay")

    # Session lines (Forex)
    if mode == "Forex" and show_sessions_pst:
        lines = compute_session_lines(idx)
        pos_lines = map_session_lines_to_positions(lines, idx)
        for p in pos_lines.get("ldn_open", []): ax.axvline(p, color="tab:blue", alpha=0.25, linewidth=1.0)
        for p in pos_lines.get("ldn_close", []): ax.axvline(p, color="tab:blue", alpha=0.25, linewidth=1.0, linestyle="--")
        for p in pos_lines.get("ny_open", []): ax.axvline(p, color="tab:orange", alpha=0.25, linewidth=1.0)
        for p in pos_lines.get("ny_close", []): ax.axvline(p, color="tab:orange", alpha=0.25, linewidth=1.0, linestyle="--")

    # News markers (best-effort)
    if mode == "Forex" and show_fx_news:
        news = fetch_yf_news(symbol, news_window_days)
        if not news.empty:
            for t in news["time"].tolist():
                p = map_times_to_positions(idx, [t])
                if p:
                    ax.axvline(p[0], color="grey", alpha=0.18, linewidth=1.0)

    badges = [
        (f"Intraday local slope: {fmt_slope(m_l)} (R² {r2_l:.2f})", "black"),
        (f"Daily global slope sign: {'UP' if (np.isfinite(m_g) and m_g>0) else 'DOWN'}", "black"),
    ]

    # Band reversal badge + in-chart marker
    if band_sig:
        if band_sig["side"] == "BUY":
            badges.insert(0, ("▲ BUY Band REV", "tab:green"))
            # triangle marker inside chart
            try:
                p = idx.get_loc(band_sig["time"])
                ax.scatter([p], [band_sig["price"]], marker="^", s=160, c="tab:green", zorder=12, edgecolors="none")
            except Exception:
                pass
        else:
            badges.insert(0, ("▼ SELL Band REV", "tab:red"))
            try:
                p = idx.get_loc(band_sig["time"])
                ax.scatter([p], [band_sig["price"]], marker="v", s=160, c="tab:red", zorder=12, edgecolors="none")
            except Exception:
                pass

    # HMA cross badge
    if hma_sig:
        if hma_sig["side"] == "BUY":
            badges.insert(0, ("★ HMA BUY (intraday)", "black"))
        else:
            badges.insert(0, ("★ HMA SELL (intraday)", "tab:blue"))

    draw_top_badges(ax, badges)

    # Instruction ribbon: local slope + global alignment
    draw_instruction_ribbon(ax, m_l, sup_v, res_v, float(close.iloc[-1]), symbol, global_slope=m_g)

    ax.set_title(f"{symbol} — Intraday ({hour_range})")
    market_time_axis(ax, idx)
    _simplify_axes(ax)
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)
# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "Upward Slope Stickers",
    "Daily Support Reversals"
])

# Shared data (load once)
df_daily_full = fetch_daily_ohlc(ticker, period="5y")
df_daily = subset_by_daily_view(df_daily_full, daily_view) if not df_daily_full.empty else df_daily_full
df_intraday = fetch_intraday_ohlc(ticker, hour_range)

# ---------------- Tab 1: Original Forecast ----------------
with tab1:
    st.subheader(f"{ticker} — Forecast (SARIMAX if available, else drift fallback)")
    if df_daily is None or df_daily.empty:
        st.warning("No daily data available.")
    else:
        close = df_daily["Close"].astype(float)
        idx_fc, mean_fc, ci_fc = compute_forecast_sarimax(close, steps=30)

        fig, ax = plt.subplots(figsize=(12, 5.0))
        ax.plot(close.index, close.values, label="Close", linewidth=1.7)
        ax.plot(mean_fc.index, mean_fc.values, label="Forecast", linewidth=2.2)
        if ci_fc is not None and not ci_fc.empty:
            ax.fill_between(ci_fc.index, ci_fc["lower"].values, ci_fc["upper"].values, alpha=0.2, label="CI")
        _simplify_axes(ax)
        ax.set_title("30-Day Forecast")
        ax.legend(loc="best", fontsize=8)
        st.pyplot(fig)

        st.caption("Note: SARIMAX may fall back to a drift model if SARIMAX is unavailable or errors.")

# ---------------- Tab 2: Enhanced Forecast ----------------
with tab2:
    st.subheader(f"{ticker} — Enhanced View (Daily + Intraday)")
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Daily")
        plot_daily(ticker, df_daily)
    with colB:
        st.markdown("### Intraday (compressed axis)")
        plot_intraday(ticker, df_intraday, df_daily)

# ---------------- Tab 3: Bull vs Bear ----------------
with tab3:
    st.subheader(f"{ticker} — Bull vs Bear ({bb_period})")
    df_bb = fetch_daily_ohlc(ticker, period=bb_period)
    if df_bb.empty:
        st.warning("No data for bull/bear.")
    else:
        close = df_bb["Close"].astype(float)
        ret = close.pct_change().dropna()
        bull = int((ret > 0).sum())
        bear = int((ret <= 0).sum())

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.pie([bull, bear], labels=[f"Bull days ({bull})", f"Bear days ({bear})"], autopct="%1.1f%%")
        ax.set_title("Bull vs Bear (Daily returns)")
        st.pyplot(fig)

        st.write(pd.DataFrame({
            "Metric": ["Bull days", "Bear days", "Total", "Bull %"],
            "Value": [bull, bear, bull + bear, (bull / max(bull + bear, 1))*100.0]
        }))

# ---------------- Tab 4: Metrics ----------------
with tab4:
    st.subheader(f"{ticker} — Metrics")
    if df_daily.empty:
        st.warning("No data.")
    else:
        close = df_daily["Close"].astype(float)
        yhat_g, _, _, m_g, r2_g, std_g = regression_with_band(close, lookback=0, z=2.0)
        yhat_l, _, _, m_l, r2_l, std_l = regression_with_band(close, lookback=slope_lb_daily, z=2.0)
        sup, res = rolling_support_resistance(close, sr_lb_daily)

        last_close = float(close.dropna().iloc[-1]) if close.dropna().size else float("nan")
        d1 = float(close.dropna().iloc[-2]) if close.dropna().size >= 2 else float("nan")
        chg = (last_close - d1) / d1 if np.isfinite(last_close) and np.isfinite(d1) and d1 != 0 else float("nan")

        vol = float(close.pct_change().dropna().std()) if close.pct_change().dropna().size else float("nan")
        ntd_val = compute_ntd(close, ntd_window)

        metrics = [
            ("Last close", last_close),
            ("1-day %", chg*100 if np.isfinite(chg) else np.nan),
            ("Daily vol (std of returns)", vol),
            ("Global slope", m_g),
            ("Global R²", r2_g),
            ("Local slope", m_l),
            ("Local R²", r2_l),
            ("NTD (tanh scaled)", ntd_val),
            ("Support (rolling)", float(sup.dropna().iloc[-1]) if sup.dropna().size else np.nan),
            ("Resistance (rolling)", float(res.dropna().iloc[-1]) if res.dropna().size else np.nan),
        ]
        dfm = pd.DataFrame(metrics, columns=["Metric", "Value"])
        st.dataframe(dfm, use_container_width=True)

# ---------------- Tab 5: NTD -0.75 Scanner ----------------
with tab5:
    st.subheader("NTD Scanner (simple normalized trend direction)")
    st.caption("Lists symbols where NTD <= -0.75 (bearish) or >= +0.75 (bullish).")

    if not run_scans:
        st.info("Click **Run scanners now** in the sidebar to refresh scanner tables.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                ddf = fetch_daily_ohlc(sym, period="1y")
                if ddf.empty:
                    continue
                close = ddf["Close"].astype(float)
                ntd = compute_ntd(close, ntd_window)
                yhat, _, _, m, r2, _ = regression_with_band(close, lookback=min(slope_lb_daily, len(close)), z=2.0)
                rows.append({
                    "Symbol": sym,
                    "NTD": ntd,
                    "LocalSlope": m,
                    "R2": r2,
                    "LastClose": float(close.dropna().iloc[-1]) if close.dropna().size else np.nan
                })
            except Exception:
                continue
            prog.progress((i+1)/max(len(universe),1))

        scan = pd.DataFrame(rows)
        if scan.empty:
            st.warning("Scanner produced no results.")
        else:
            bear = scan[scan["NTD"] <= -0.75].sort_values("NTD")
            bull = scan[scan["NTD"] >= +0.75].sort_values("NTD", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Bearish (NTD ≤ -0.75)")
                st.dataframe(bear.reset_index(drop=True), use_container_width=True)
            with c2:
                st.markdown("### Bullish (NTD ≥ +0.75)")
                st.dataframe(bull.reset_index(drop=True), use_container_width=True)

# ---------------- Tab 6: Long-Term History ----------------
with tab6:
    st.subheader(f"{ticker} — Long-Term History (max available)")
    df_max = fetch_daily_ohlc(ticker, period="max")
    if df_max.empty:
        st.warning("No max-history data.")
    else:
        close = df_max["Close"].astype(float)
        yhat_g, _, _, m_g, r2_g, _ = regression_with_band(close, lookback=0, z=2.0)

        fig, ax = plt.subplots(figsize=(12, 5.0))
        ax.plot(close.index, close.values, linewidth=1.4, label="Close")
        if not yhat_g.empty and np.isfinite(m_g):
            ax.plot(yhat_g.index, yhat_g.values, linewidth=3.0,
                    color=("tab:green" if m_g >= 0 else "tab:red"),
                    label=f"Global Trend ({fmt_slope(m_g)}/bar)")
        _simplify_axes(ax)
        ax.set_title(f"Max History — slope {fmt_slope(m_g)} | R² {r2_g:.2f}")
        ax.legend(loc="best", fontsize=8)
        st.pyplot(fig)

# ---------------- Tab 7: Upward Slope Stickers ----------------
with tab7:
    st.subheader("Upward Slope Stickers — Up daily slope AND price below slope (latest bar)")
    st.caption("Useful for spotting upward-trend pullbacks (price under regression slope line).")

    if not run_scans:
        st.info("Click **Run scanners now** in the sidebar to refresh.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                ddf = fetch_daily_ohlc(sym, period="1y")
                if ddf.empty:
                    continue
                close = ddf["Close"].astype(float)
                yhat, _, _, m, r2, _ = regression_with_band(close, lookback=min(slope_lb_daily, len(close)), z=2.0)
                if not yhat.empty and np.isfinite(m) and m > 0:
                    last_close = float(close.dropna().iloc[-1]) if close.dropna().size else np.nan
                    last_yhat = float(yhat.dropna().iloc[-1]) if yhat.dropna().size else np.nan
                    if np.isfinite(last_close) and np.isfinite(last_yhat) and last_close < last_yhat:
                        rows.append({
                            "Symbol": sym,
                            "LocalSlope": m,
                            "R2": r2,
                            "LastClose": last_close,
                            "SlopeLine": last_yhat,
                            "BelowBy": (last_yhat - last_close)
                        })
            except Exception:
                continue
            prog.progress((i+1)/max(len(universe),1))

        out = pd.DataFrame(rows).sort_values(["R2","BelowBy"], ascending=[False, False])
        st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------- Tab 8: Daily Support Reversals (99% validation) ----------------
def _rel_near(a: float, b: float, tol: float) -> bool:
    try:
        a = float(a); b = float(b)
    except Exception:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= float(tol)

def daily_sr_99_reversal_signal(price: pd.Series,
                                support: pd.Series,
                                resistance: pd.Series,
                                upper99: pd.Series,
                                lower99: pd.Series,
                                local_slope: float,
                                prox: float = 0.0025,
                                confirm_bars: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(3, confirm_bars + 1) or not np.isfinite(local_slope) or local_slope == 0:
        return None

    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    up99 = _coerce_1d_series(upper99).reindex(p.index)
    lo99 = _coerce_1d_series(lower99).reindex(p.index)

    if sup.dropna().empty or res.dropna().empty or up99.dropna().empty or lo99.dropna().empty:
        return None

    def _inc_ok(n: int) -> bool:
        s = p.dropna()
        if len(s) < n+1: return False
        return bool(np.all(np.diff(s.iloc[-(n+1):]) > 0))

    def _dec_ok(n: int) -> bool:
        s = p.dropna()
        if len(s) < n+1: return False
        return bool(np.all(np.diff(s.iloc[-(n+1):]) < 0))

    t0 = p.index[-1]
    c0, c1 = float(p.iloc[-1]), float(p.iloc[-2])
    s1 = float(sup.iloc[-2])
    r1 = float(res.iloc[-2])
    u1 = float(up99.iloc[-2]) if np.isfinite(up99.iloc[-2]) else np.nan
    l1 = float(lo99.iloc[-2]) if np.isfinite(lo99.iloc[-2]) else np.nan

    if local_slope > 0:
        prev_near_support = (c1 <= s1 * (1.0 + prox))
        support_near_99   = _rel_near(s1, l1, prox)
        going_up          = _inc_ok(confirm_bars)
        if prev_near_support and support_near_99 and going_up:
            return {"time": t0, "price": c0, "side": "BUY", "note": "ALERT 99% SR REV"}
    else:
        prev_near_res = (c1 >= r1 * (1.0 - prox))
        res_near_99   = _rel_near(r1, u1, prox)
        going_down    = _dec_ok(confirm_bars)
        if prev_near_res and res_near_99 and going_down:
            return {"time": t0, "price": c0, "side": "SELL", "note": "ALERT 99% SR REV"}
    return None

with tab8:
    st.subheader("Daily Support Reversals — validated vs 99% regression band")
    st.caption("Up-slope: price reverses up from support near the 99% lower band. Down-slope: reverses down from resistance near the 99% upper band.")

    if not run_scans:
        st.info("Click **Run scanners now** in the sidebar to refresh.")
    else:
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(universe):
            try:
                ddf = fetch_daily_ohlc(sym, period="1y")
                if ddf.empty:
                    continue
                close = ddf["Close"].astype(float)
                sup, res = rolling_support_resistance(close, sr_lb_daily)
                yhat99, up99, lo99, m, r2, std = regression_with_band(close, lookback=min(slope_lb_daily, len(close)), z=Z_FOR_99)
                sig = daily_sr_99_reversal_signal(close, sup, res, up99, lo99, m, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)
                if sig:
                    rows.append({
                        "Symbol": sym,
                        "Side": sig["side"],
                        "Time": sig["time"],
                        "Price": sig["price"],
                        "LocalSlope": m,
                        "R2": r2
                    })
            except Exception:
                continue
            prog.progress((i+1)/max(len(universe),1))

        out = pd.DataFrame(rows).sort_values("Time", ascending=False)
        st.dataframe(out.reset_index(drop=True), use_container_width=True)

        st.markdown("---")
        st.markdown("### Detail chart (selected symbol)")
        sel = st.selectbox("Pick a symbol from results (or any)", universe, index=universe.index(ticker) if ticker in universe else 0)
        ddf = fetch_daily_ohlc(sel, period="1y")
        ddf = subset_by_daily_view(ddf, daily_view) if not ddf.empty else ddf
        if ddf.empty:
            st.warning("No data.")
        else:
            close = ddf["Close"].astype(float)
            sup, res = rolling_support_resistance(close, sr_lb_daily)
            yhat99, up99, lo99, m, r2, std = regression_with_band(close, lookback=min(slope_lb_daily, len(close)), z=Z_FOR_99)
            sig = daily_sr_99_reversal_signal(close, sup, res, up99, lo99, m, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)

            fig, ax = plt.subplots(figsize=(12, 5.2))
            ax.plot(close.index, close.values, linewidth=1.7, label="Close")
            ax.plot(sup.index, sup.values, linewidth=1.0, alpha=0.7, label="Support (rolling)")
            ax.plot(res.index, res.values, linewidth=1.0, alpha=0.7, label="Resistance (rolling)")

            if not yhat99.empty:
                ax.plot(yhat99.index, yhat99.values, linestyle="--", linewidth=2.2,
                        color=("tab:green" if m >= 0 else "tab:red"),
                        label=f"Slope line ({fmt_slope(m)}/bar)")
                ax.plot(up99.index, up99.values, linewidth=1.2, alpha=0.7, label="Upper 99%")
                ax.plot(lo99.index, lo99.values, linewidth=1.2, alpha=0.7, label="Lower 99%")

            badges = [
                (f"Local slope: {fmt_slope(m)} (R² {r2:.2f})", "black"),
                ("99% band validation active", "black"),
            ]
            if sig:
                if sig["side"] == "BUY":
                    ax.scatter([sig["time"]], [sig["price"]], marker="^", s=170, c="tab:green", zorder=12)
                    badges.insert(0, ("▲ BUY ALERT 99% SR REV", "tab:green"))
                else:
                    ax.scatter([sig["time"]], [sig["price"]], marker="v", s=170, c="tab:red", zorder=12)
                    badges.insert(0, ("▼ SELL ALERT 99% SR REV", "tab:red"))

            draw_top_badges(ax, badges)
            _simplify_axes(ax)
            ax.set_title(f"{sel} — Daily Support Reversal Detail")
            ax.legend(loc="best", fontsize=8)
            st.pyplot(fig)
# ---------------- Top banner / status ----------------
try:
    if df_daily is not None and not df_daily.empty:
        last_px = float(df_daily["Close"].dropna().iloc[-1])
        yhat_g, _, _, m_g, r2_g, _ = regression_with_band(df_daily["Close"].astype(float), lookback=0, z=2.0)
        msg = f"{ticker} last: {fmt_price_val(last_px)} | Global trend: {'UP' if (np.isfinite(m_g) and m_g>0) else 'DOWN'} (slope {fmt_slope(m_g)})"
        top_warn.info(msg)
    else:
        top_warn.warning("No daily data loaded.")
except Exception:
    pass
