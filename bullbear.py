# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATE: Single latest band-reversal signal for trading
#   • Uptrend  → BUY when price reverses up from lower ±2σ band and is near it
#   • Downtrend→ SELL when price reverses down from upper ±2σ band and is near it
# Only the latest signal is shown at any time.
# CHANGELOG:
#   • All downward trend lines are red (upward = green) throughout.
#   • Instruction text fixed to: "SELL → BUY • Value of Pips: …" (always this order).
#   • Removed Momentum & NTD/NPX charts (scanner tab remains).
#   • HMA(55) alert added to price charts; signals gated by trend + reversal confirmation.
#   • Fibonacci default = ON (hourly only).
#   • Fixed SARIMAX crash when history is empty/too short.
#   • Fixed NameError: added back Supertrend/PSAR/Ichimoku/BB helpers.

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
    .css-18e3th9 { transform: none !important; visibility: visible !important;
                   width: 100% !important; position: relative !important; margin-bottom: 1rem; }
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
    Instruction format (fixed order):
      ▼ SELL first, then ▲ BUY, then explicit Value of Pips.
      Uses the provided `sell_val` (resistance) and `buy_val` (support).
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    text = (
        f"▼ SELL @{fmt_price_val(exit_sell)} → ▲ BUY @{fmt_price_val(entry_buy)}"
        f" • Value of Pips: {_diff_text(exit_sell, entry_buy, symbol)}"
    )
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
show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=True, key="sb_show_fibs")  # default ON

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120,  6, key="sb_slope_lb_hourly")

st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signals")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R / Band proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0
rev_bars_confirm = st.sidebar.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

st.sidebar.subheader("Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb= st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands   = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_bb_show")
bb_win        = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult       = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema    = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("HMA (Price Charts)")
show_hma    = st.sidebar.checkbox("Show HMA", value=True, key="sb_hma_show")
hma_period  = st.sidebar.slider("HMA period (plotted)", 5, 120, 55, 1, key="sb_hma_period")
hma_conf    = st.sidebar.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

st.sidebar.subheader("Scanner Settings")
ntd_window= st.sidebar.slider("NTD slope window (for scanner)", 10, 300, 60, 5, key="sb_ntd_win")

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
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
    ]

# ---------------- Data fetch & core calcs ----------------
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
    # Normalize timezone
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    # Guard: empty or too-short series → safe NaN forecast
    if series.shape[0] < 5:
        start = (pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1))
        idx = pd.date_range(start, periods=30, freq="D", tz=PACIFIC)
        vals = pd.Series(np.nan, index=idx)
        ci = pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=idx)
        return idx, vals, ci
    # Model
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=30)
    last_idx = series.index[-1]
    idx = pd.date_range(last_idx + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    mean = fc.predicted_mean; mean.index = idx
    ci = fc.conf_int();       ci.index   = idx
    return idx, mean, ci

def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {"0%": hi, "23.6%": hi - 0.236*diff, "38.2%": hi - 0.382*diff,
            "50%": hi - 0.5*diff, "61.8%": hi - 0.618*diff, "78.6%": hi - 0.786*diff, "100%": lo}

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
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "tab:green" if m >= 0 else "tab:red"   # downward = red
    ax.plot(s.index, yhat, "-", linewidth=2.4, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# --- Supertrend / ATR ---
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
    return pd.DataFrame({"ST": st_line, "in_uptrend": in_up,
                         "upperband": upperband, "lowerband": lowerband})

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

# --- Ichimoku (classic) ---
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

# --- Bollinger Bands + normalized %B / NBB ---
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

# --- HMA & signal helpers ---
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

def _inc_ok_at(series: pd.Series, t, n: int) -> bool:
    s = _coerce_1d_series(series)
    if t not in s.index or n < 1:
        return False
    seg = s.loc[:t].dropna().iloc[-(n+1):]
    if len(seg) < n+1:
        return False
    d = np.diff(seg)
    return bool(np.all(d > 0))

def _dec_ok_at(series: pd.Series, t, n: int) -> bool:
    s = _coerce_1d_series(series)
    if t not in s.index or n < 1:
        return False
    seg = s.loc[:t].dropna().iloc[-(n+1):]
    if len(seg) < n+1:
        return False
    d = np.diff(seg)
    return bool(np.all(d < 0))

def detect_hma_trend_reversal(price: pd.Series, hma: pd.Series, trend_slope: float, confirm_bars: int = 2):
    """
    BUY when trend_slope > 0 AND price crosses UP through HMA AND last N bars into the cross were rising.
    SELL when trend_slope < 0 AND price crosses DOWN through HMA AND last N bars into the cross were falling.
    Returns the most recent valid event (time, price, side) or None.
    """
    p = _coerce_1d_series(price)
    h = _coerce_1d_series(hma).reindex(p.index)
    mask = p.notna() & h.notna()
    if mask.sum() < max(3, confirm_bars + 1):
        return None
    p = p[mask]; h = h[mask]
    cross_up, cross_dn = _cross_series(p, h)
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    if trend_slope > 0:
        idxs = list(cross_up[cross_up].index)
        for t in reversed(idxs):
            if _inc_ok_at(p, t, confirm_bars):
                return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    else:  # trend_slope < 0
        idxs = list(cross_dn[cross_dn].index)
        for t in reversed(idxs):
            if _dec_ok_at(p, t, confirm_bars):
                return {"time": t, "price": float(p.loc[t]), "side": "SELL"}
    return None

def annotate_crossover(ax, ts, px, side: str, note: str = ""):
    if side == "BUY":
        ax.scatter([ts], [px], marker="P", s=100, color="tab:green", zorder=7)
        label = "BUY" if not note else f"BUY {note}"
        ax.text(ts, px, f"  {label}", va="bottom", fontsize=10,
                color="tab:green", fontweight="bold")
    else:
        ax.scatter([ts], [px], marker="X", s=100, color="tab:red", zorder=7)
        label = "SELL" if not note else f"SELL {note}"
        ax.text(ts, px, f"  {label}", va="top", fontsize=10,
                color="tab:red", fontweight="bold")

# --- Bands + single latest band-reversal trading signal ---
def last_band_reversal_signal(price: pd.Series,
                              band_upper: pd.Series,
                              band_lower: pd.Series,
                              trend_slope: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 1):
    p = _coerce_1d_series(price).dropna()
    up = _coerce_1d_series(band_upper).reindex(p.index)
    lo = _coerce_1d_series(band_lower).reindex(p.index)

    if p.shape[0] < 2 or up.dropna().empty or lo.dropna().empty:
        return None
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    mask = p.notna() & up.notna() & lo.notna()
    p = p[mask]; up = up[mask]; lo = lo[mask]
    if p.shape[0] < 2:
        return None

    t0 = p.index[-1]
    c0, c1 = float(p.iloc[-1]), float(p.iloc[-2])
    u0, u1 = float(up.iloc[-1]), float(up.iloc[-2])
    l0, l1 = float(lo.iloc[-1]), float(lo.iloc[-2])

    def _inc_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n+1:
            return False
        d = np.diff(s.iloc[-(n+1):])
        return bool(np.all(d > 0))

    def _dec_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n+1:
            return False
        d = np.diff(s.iloc[-(n+1):])
        return bool(np.all(d < 0))

    if trend_slope > 0:
        prev_near_lower = (c1 <= l1 * (1.0 + prox))
        bounced_above   = (c0 >= l0)
        going_up        = _inc_ok(p, confirm_bars)
        if prev_near_lower and bounced_above and going_up:
            return {"time": t0, "price": c0, "side": "BUY", "note": "Band REV"}
    else:
        prev_near_upper = (c1 >= u1 * (1.0 - prox))
        rolled_below    = (c0 <= u0)
        going_down      = _dec_ok(p, confirm_bars)
        if prev_near_upper and rolled_below and going_down:
            return {"time": t0, "price": c0, "side": "SELL", "note": "Band REV"}
    return None

# --- Sessions & News ---
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
    return {"ldn_open": ldn_open, "ldn_close": ldn_close,
            "ny_open": ny_open,   "ny_close": ny_close}

def draw_session_lines(ax, lines: dict):
    ax.plot([], [], linestyle="-",  color="tab:blue",   label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue",   label="London Close (PST)")
    ax.plot([], [], linestyle="-",  color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
    for t in lines.get("ldn_open", []):  ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ldn_close", []): ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue",   alpha=0.35)
    for t in lines.get("ny_open", []):   ax.axvline(t, linestyle="-",  linewidth=1.0, color="tab:orange", alpha=0.35)
    for t in lines.get("ny_close", []):  ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=0.35)
    ax.text(0.99, 0.98, "Session times in PST", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="black",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7))

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
            "title": item.get("title",""),
            "publisher": item.get("publisher",""),
            "link": item.get("link","")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")
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
    "NTD -0.75 Scanner",
    "Long-Term History"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")
    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:", ["24h", "48h", "96h"],
                              index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
                              key="hour_range_select")
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    auto_run = st.session_state.run_all

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist, "df_ohlc": df_ohlc,
            "fc_idx": fc_idx, "fc_vals": fc_vals, "fc_ci": fc_ci,
            "intraday": intraday, "ticker": sel, "chart": chart,
            "hour_range": hour_range, "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        fx_news = fetch_yf_news(sel, window_days=news_window_days) if (mode == "Forex" and show_fx_news) else pd.DataFrame()

        # ----- Daily (Price only) -----
        if chart in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()

            # Trendline with ±2σ band and R² (Daily)
            yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)

            kijun_d = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                     conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d = kijun_d.ffill().bfill()

            bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

            df_show     = subset_by_daily_view(df, daily_view)
            ema30_show  = ema30.reindex(df_show.index)
            res30_show  = res30.reindex(df_show.index)
            sup30_show  = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
            lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
            kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
            bb_mid_d_show = bb_mid_d.reindex(df_show.index)
            bb_up_d_show  = bb_up_d.reindex(df_show.index)
            bb_lo_d_show  = bb_lo_d.reindex(df_show.index)

            # HMA lines & HMA(55) signal
            hma_d_full = compute_hma(df, period=hma_period)
            hma_d_show = hma_d_full.reindex(df_show.index)
            hma55_d = compute_hma(df, period=55).reindex(df_show.index)
            hma55_evt_d = detect_hma_trend_reversal(df_show, hma55_d, trend_slope=m_d, confirm_bars=rev_bars_confirm)

            fig, ax = plt.subplots(figsize=(14, 6))
            plt.subplots_adjust(top=0.92, right=0.93)

            ax.set_title(f"{sel} Daily — {daily_view} — Price, 30 EMA, 30 S/R, Slope")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if show_hma and not hma_d_show.dropna().empty:
                ax.plot(hma_d_show.index, hma_d_show.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_ichi and not kijun_d_show.dropna().empty:
                ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.8, color="black",
                        label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=1.1,
                        label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d_show.index, bb_up_d_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d_show.index, bb_lo_d_show.values, ":", linewidth=1.0)

            if not yhat_d_show.empty:
                slope_col_d = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2, color=slope_col_d,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not upper_d_show.empty and not lower_d_show.empty:
                ax.plot(upper_d_show.index, upper_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lower_d_show.index, lower_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")

            # Single latest band-reversal signal (Daily)
            band_sig_d = last_band_reversal_signal(
                price=df_show, band_upper=upper_d_show, band_lower=lower_d_show,
                trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
            )
            if band_sig_d is not None:
                annotate_crossover(ax, band_sig_d["time"], band_sig_d["price"], band_sig_d["side"], note=band_sig_d.get("note",""))

            # HMA(55) alert annotation (trend-valid)
            if hma55_evt_d is not None:
                annotate_crossover(ax, hma55_evt_d["time"], hma55_evt_d["price"], hma55_evt_d["side"], note="HMA55")

            ax.set_ylabel("Price")
            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly (Price only) -----
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
                st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday else pd.Series(index=hc.index, dtype=float)

                kijun_h = pd.Series(index=hc.index, dtype=float)
                if {'High','Low','Close'}.issubset(intraday.columns) and show_ichi:
                    _, kijun_h, _, _, _ = ichimoku_lines(intraday["High"], intraday["Low"], intraday["Close"],
                                                         conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                    kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

                bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
                hma_h = compute_hma(hc, period=hma_period)
                hma55_h = compute_hma(hc, period=55)

                psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
                psar_h_df = psar_h_df.reindex(hc.index)

                # Hourly regression slope & bands
                yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
                slope_sig_h = m_h if np.isfinite(m_h) else slope_h

                # GLOBAL (DAILY) slope for instruction only
                try:
                    df_global = st.session_state.df_hist
                    _, _, _, m_global, _ = regression_with_band(df_global, slope_lb_daily)
                except Exception:
                    m_global = slope_sig_h

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                trend_color = "tab:green" if slope_h >= 0 else "tab:red"
                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", label=f"Trend (m={fmt_slope(slope_h)}/bar)", linewidth=2, color=trend_color)

                if show_hma and not hma_h.dropna().empty:
                    ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

                if show_ichi and not kijun_h.dropna().empty:
                    ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

                if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
                    ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                    ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                    ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
                    ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

                res_val = sup_val = px_val = np.nan
                try:
                    res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label="Resistance")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                # Single latest band-reversal signal (Hourly)
                band_sig_h = last_band_reversal_signal(
                    price=hc, band_upper=upper_h, band_lower=lower_h,
                    trend_slope=m_h, prox=sr_prox_pct, confirm_bars=rev_bars_confirm
                )
                if band_sig_h is not None:
                    annotate_crossover(ax2, band_sig_h["time"], band_sig_h["price"], band_sig_h["side"], note=band_sig_h.get("note",""))

                # HMA(55) alert annotation (trend-valid)
                hma55_evt_h = detect_hma_trend_reversal(hc, hma55_h, trend_slope=m_h, confirm_bars=rev_bars_confirm)
                if hma55_evt_h is not None:
                    annotate_crossover(ax2, hma55_evt_h["time"], hma55_evt_h["price"], hma55_evt_h["side"], note="HMA55")

                # BUY/SELL instruction text (fixed order)
                instr_txt = format_trade_instruction(trend_slope=m_global, buy_val=sup_val, sell_val=res_val, close_val=px_val, symbol=sel)

                title_sig = ""
                if 'band_sig_h' in locals() and band_sig_h is not None:
                    title_sig = f" — Latest Signal: {band_sig_h['side']} @ {fmt_price_val(band_sig_h['price'])}"

                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)} — {instr_txt}{title_sig}")

                ax2.text(0.01, 0.965, f"Instruction uses GLOBAL (Daily) slope: {fmt_slope(m_global)}/bar",
                         transform=ax2.transAxes, ha="left", va="top",
                         fontsize=9, color="black",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if np.isfinite(px_val):
                    nbb_txt = ""
                    try:
                        last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
                        if np.isfinite(last_nbb) and np.isfinite(last_pct):
                            nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
                    except Exception:
                        pass
                    ax2.text(0.99, 0.02, f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                             transform=ax2.transAxes, ha="right", va="bottom",
                             fontsize=11, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

                if not st_line_intr.dropna().empty:
                    ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    slope_col_h = "tab:green" if m_h >= 0 else "tab:red"
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, color=slope_col_h, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
                if not upper_h.empty and not lower_h.empty:
                    ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
                    ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

                if mode == "Forex" and show_sessions_pst and not hc.empty:
                    sess = compute_session_lines(hc.index)
                    draw_session_lines(ax2, sess)

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2)

        # News table
        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        # Forecast table
        st.write(pd.DataFrame({"Forecast": st.session_state.fc_vals,
                               "Lower":    st.session_state.fc_ci.iloc[:,0],
                               "Upper":    st.session_state.fc_ci.iloc[:,1]}, index=st.session_state.fc_idx))
# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci)
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan
        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}**")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        if view in ("Daily","Both"):
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, up_d, lo_d, m_d, r2_d = regression_with_band(df, slope_lb_daily)

            kijun_d2 = pd.Series(index=df.index, dtype=float)
            if df_ohlc is not None and not df_ohlc.empty and show_ichi:
                _, kijun_d2, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                                      conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
                kijun_d2 = kijun_d2.ffill().bfill()

            bb_mid_d2, bb_up_d2, bb_lo_d2 = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)[:3]

            df_show = subset_by_daily_view(df, daily_view)
            ema30_show = ema30.reindex(df_show.index)
            res30_show = res30.reindex(df_show.index)
            sup30_show = sup30.reindex(df_show.index)
            yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
            up_d_show   = up_d.reindex(df_show.index) if not up_d.empty else up_d
            lo_d_show   = lo_d.reindex(df_show.index) if not lo_d.empty else lo_d
            kijun_d2_show = kijun_d2.reindex(df_show.index).ffill().bfill()
            bb_mid_d2_show = bb_mid_d2.reindex(df_show.index)
            bb_up_d2_show  = bb_up_d2.reindex(df_show.index)
            bb_lo_d2_show  = bb_lo_d2.reindex(df_show.index)

            # HMA plotting + HMA(55) alert (trend-valid)
            hma_d2_full = compute_hma(df, period=hma_period).reindex(df_show.index)
            hma55_d2 = compute_hma(df, period=55).reindex(df_show.index)
            hma55_evt_d2 = detect_hma_trend_reversal(df_show, hma55_d2, trend_slope=m_d, confirm_bars=rev_bars_confirm)

            fig, ax = plt.subplots(figsize=(14, 6))
            plt.subplots_adjust(top=0.92, right=0.93)
            ax.set_title(f"{st.session_state.ticker} Daily — {daily_view} — Price, 30 EMA, 30 S/R, Slope")
            ax.plot(df_show, label="History")
            ax.plot(ema30_show, "--", label="30 EMA")
            ax.plot(res30_show, ":", label="30 Resistance")
            ax.plot(sup30_show, ":", label="30 Support")

            if show_hma and not hma_d2_full.dropna().empty:
                ax.plot(hma_d2_full.index, hma_d2_full.values, "-", linewidth=1.6, label=f"HMA({hma_period})")
            if show_ichi and not kijun_d2_show.dropna().empty:
                ax.plot(kijun_d2_show.index, kijun_d2_show.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

            if show_bbands and not bb_up_d2_show.dropna().empty and not bb_lo_d2_show.dropna().empty:
                ax.fill_between(df_show.index, bb_lo_d2_show, bb_up_d2_show, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                ax.plot(bb_mid_d2_show.index, bb_mid_d2_show.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                ax.plot(bb_up_d2_show.index, bb_up_d2_show.values, ":", linewidth=1.0)
                ax.plot(bb_lo_d2_show.index, bb_lo_d2_show.values, ":", linewidth=1.0)

            if not yhat_d_show.empty:
                slope_col_d2 = "tab:green" if m_d >= 0 else "tab:red"
                ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=2, color=slope_col_d2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not up_d_show.empty and not lo_d_show.empty:
                ax.plot(up_d_show.index, up_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend +2σ")
                ax.plot(lo_d_show.index, lo_d_show.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Daily Trend -2σ")
            if len(df_show) > 1:
                draw_trend_direction_line(ax, df_show, label_prefix="Trend")

            # Band signal (Daily)
            band_sig_d2 = last_band_reversal_signal(price=df_show, band_upper=up_d_show, band_lower=lo_d_show,
                                                    trend_slope=m_d, prox=sr_prox_pct, confirm_bars=rev_bars_confirm)
            if band_sig_d2 is not None:
                annotate_crossover(ax, band_sig_d2["time"], band_sig_d2["price"], band_sig_d2["side"], note=band_sig_d2.get("note",""))

            # HMA55 trend-valid alert
            if hma55_evt_d2 is not None:
                annotate_crossover(ax, hma55_evt_d2["time"], hma55_evt_d2["price"], hma55_evt_d2["side"], note="HMA55")

            ax.set_ylabel("Price")
            ax.text(0.50, 0.02, f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                st.info("Intraday view is rendered fully in Tab 1 (same logic).")

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
        trend3m, up3m, lo3m, m3m, r2_3m = regression_with_band(df3m, lookback=len(df3m))

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", label="Support")
        if not trend3m.empty:
            col3 = "tab:green" if m3m >= 0 else "tab:red"
            ax.plot(trend3m.index, trend3m.values, "--", color=col3, label=f"Trend (m={fmt_slope(m3m)}/bar)")
        if not up3m.empty and not lo3m.empty:
            ax.plot(up3m.index, up3m.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
            ax.plot(lo3m.index, lo3m.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
        ax.set_xlabel("Date (PST)")
        ax.text(0.50, 0.02, f"R² (3M): {fmt_r2(r2_3m)}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
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
        trend0, up0, lo0, m0, r2_0 = regression_with_band(df0['Close'], lookback=len(df0))

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(res0.index, res0, ":", label="Resistance")
        ax0.plot(sup0.index, sup0, ":", label="Support")
        if not trend0.empty:
            col0 = "tab:green" if m0 >= 0 else "tab:red"
            ax0.plot(trend0.index, trend0.values, "--", color=col0, label=f"Trend (m={fmt_slope(m0)}/bar)")
        if not up0.empty and not lo0.empty:
            ax0.plot(up0.index, up0.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
            ax0.plot(lo0.index, lo0.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
        ax0.set_xlabel("Date (PST)")
        ax0.text(0.50, 0.02, f"R² ({bb_period}): {fmt_r2(r2_0)}", transform=ax0.transAxes, ha="center", va="bottom",
                 fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({"Type": ["Bull", "Bear"],
                             "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]}).set_index("Type")
        st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.75 Scanner (Latest NTD < -0.75) ---
with tab5:
    st.header("NTD -0.75 Scanner (NTD < -0.75)")
    st.caption("Scans the universe for symbols whose **latest NTD value** is below **-0.75** "
               "on the Daily NTD line (and on the Hourly NTD line for Forex).")

    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox("Hourly lookback for Forex:",
                                   ["24h", "48h", "96h"],
                                   index=["24h","48h","96h"].index(st.session_state.get("hour_range", "24h")),
                                   key="ntd_scan_hour_range")
    scan_period = period_map[scan_hour_range]
    thresh = -0.75
    run = st.button("Scan Universe", key="btn_ntd_scan")

    # Local NTD (kept for scanner only)
    def compute_normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
        s = _coerce_1d_series(close).astype(float)
        if s.empty or window < 3:
            return pd.Series(index=s.index, dtype=float)
        minp = max(5, window // 3)
        def _slope(y: pd.Series) -> float:
            y = pd.Series(y).dropna()
            if len(y) < 3: return np.nan
            x = np.arange(len(y), dtype=float)
            try: m, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
            except Exception: return np.nan
            return float(m)
        slope_roll = s.rolling(window, min_periods=minp).apply(_slope, raw=False)
        vol = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
        ntd_raw = (slope_roll * window) / vol
        return np.tanh(ntd_raw / 2.0).reindex(s.index)

    if run:
        # ---- DAILY: latest NTD < -0.75 ----
        daily_rows = []
        for sym in universe:
            try:
                s = fetch_hist(sym)
                ntd = compute_normalized_trend(s, window=ntd_window).dropna()
                ntd_val = float(ntd.iloc[-1]) if not ntd.empty else np.nan
                ts = ntd.index[-1] if not ntd.empty else None
                close_val = _safe_last_float(s)
            except Exception:
                ntd_val, ts, close_val = np.nan, None, np.nan
            daily_rows.append({"Symbol": sym, "NTD_Last": ntd_val,
                               "BelowThresh": (np.isfinite(ntd_val) and ntd_val < thresh),
                               "Close": close_val, "Timestamp": ts})
        df_daily = pd.DataFrame(daily_rows)
        hits_daily = df_daily[df_daily["BelowThresh"] == True].copy().sort_values("NTD_Last")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric(f"Daily NTD < {thresh:+.2f}", int(hits_daily.shape[0]))

        st.subheader(f"Daily — latest NTD < {thresh:+.2f}")
        if hits_daily.empty:
            st.info(f"No symbols where the latest **daily** NTD value is below {thresh:+.2f}.")
        else:
            view = hits_daily.copy()
            view["NTD_Last"] = view["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
            view["Close"] = view["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

        # ---- DAILY: PRICE > KIJUN ----
        st.markdown("---")
        st.subheader(f"Daily — Price > Ichimoku Kijun({ichi_base}) (latest bar)")
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
            c_now = float(close[mask].iloc[-1]); k_now = float(kijun[mask].iloc[-1])
            ts = close[mask].index[-1]
            above = np.isfinite(c_now) and np.isfinite(k_now) and (c_now > k_now)
            return above, ts if above else None, c_now, k_now

        above_rows = []
        for sym in universe:
            try:
                df_ohlc = fetch_hist_ohlc(sym)
                above, ts, cnow, know = _price_above_kijun_from_df(df_ohlc, base=ichi_base)
            except Exception:
                above, ts, cnow, know = False, None, np.nan, np.nan
            above_rows.append({"Symbol": sym, "AboveNow": above, "Timestamp": ts, "Close": cnow, "Kijun": know})
        df_above_daily = pd.DataFrame(above_rows)
        df_above_daily = df_above_daily[df_above_daily["AboveNow"] == True]
        if df_above_daily.empty:
            st.info("No Daily symbols with Price > Kijun on the latest bar.")
        else:
            v = df_above_daily.copy()
            v["Close"] = v["Close"].map(lambda x: fmt_price_val(x) if np.isfinite(x) else "n/a")
            v["Kijun"] = v["Kijun"].map(lambda x: fmt_price_val(x) if np.isfinite(x) else "n/a")
            st.dataframe(v[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

        # ---- FOREX HOURLY: latest NTD < -0.75 ----
        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — latest NTD < {thresh:+.2f} ({scan_hour_range} lookback)")
            hourly_rows = []
            for sym in universe:
                try:
                    intr = fetch_intraday(sym, period=scan_period)
                    close_val_h = _safe_last_float(intr["Close"]) if intr is not None and "Close" in intr else np.nan
                    ntd = compute_normalized_trend(intr["Close"], window=ntd_window) if (intr is not None and "Close" in intr) else pd.Series(dtype=float)
                    ntd_val_h = float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan
                    ts_h = ntd.dropna().index[-1] if not ntd.dropna().empty else None
                except Exception:
                    close_val_h, ntd_val_h, ts_h = np.nan, np.nan, None
                hourly_rows.append({"Symbol": sym, "NTD_Last": ntd_val_h, "BelowThresh": (np.isfinite(ntd_val_h) and ntd_val_h < thresh),
                                    "Close": close_val_h, "Timestamp": ts_h})
            df_hour = pd.DataFrame(hourly_rows)
            hits_hour = df_hour[df_hour["BelowThresh"] == True].copy().sort_values("NTD_Last")

            if hits_hour.empty:
                st.info(f"No Forex pairs where the latest **hourly** NTD value is below {thresh:+.2f} within {scan_hour_range} lookback.")
            else:
                showh = hits_hour.copy()
                showh["NTD_Last"] = showh["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
                showh["Close"] = showh["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(showh[["Symbol","Timestamp","Close","NTD_Last"]].reset_index(drop=True), use_container_width=True)

            # ---- FOREX HOURLY: PRICE > KIJUN ----
            st.subheader(f"Forex Hourly — Price > Ichimoku Kijun({ichi_base}) (latest bar, {scan_hour_range})")
            habove_rows = []
            for sym in universe:
                try:
                    intr = fetch_intraday(sym, period=scan_period)
                    if intr is None or intr.empty or not {'High','Low','Close'}.issubset(intr.columns):
                        above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                    else:
                        _, kij, _, _, _ = ichimoku_lines(intr["High"], intr["Low"], intr["Close"], base=ichi_base)
                        kij = kij.ffill().bfill().reindex(intr.index)
                        close = intr["Close"].astype(float).reindex(intr.index)
                        mask = close.notna() & kij.notna()
                        if mask.sum() < 1:
                            above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                        else:
                            close_h = float(close[mask].iloc[-1]); kij_h = float(kij[mask].iloc[-1])
                            ts_h = close[mask].index[-1]; above_h = (close_h > kij_h)
                except Exception:
                    above_h, ts_h, close_h, kij_h = False, None, np.nan, np.nan
                habove_rows.append({"Symbol": sym, "AboveNow": above_h, "Timestamp": ts_h, "Close": close_h, "Kijun": kij_h})
            df_above_hour = pd.DataFrame(habove_rows)
            df_above_hour = df_above_hour[df_above_hour["AboveNow"] == True]
            if df_above_hour.empty:
                st.info("No Forex pairs with Price > Kijun on the latest bar.")
            else:
                vch = df_above_hour.copy()
                vch["Close"] = vch["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                vch["Kijun"] = vch["Kijun"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(vch[["Symbol","Timestamp","Close","Kijun"]].reset_index(drop=True), use_container_width=True)

# --- Tab 6: Long-Term History ---
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
            res_last = float(res_roll.iloc[-1]) if len(res_roll) else np.nan
            sup_last = float(sup_roll.iloc[-1]) if len(sup_roll) else np.nan
            yhat_all, upper_all, lower_all, m_all, r2_all = regression_with_band(s, lookback=len(s))

            fig, ax = plt.subplots(figsize=(14,5))
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close")
            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red",   linestyles="-", linewidth=1.6, label="Resistance (252d)")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support (252d)")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")
            if not yhat_all.empty:
                col_all = "tab:green" if m_all >= 0 else "tab:red"
                ax.plot(yhat_all.index, yhat_all.values, "--", linewidth=2, color=col_all, label=f"Trend (m={fmt_slope(m_all)}/bar)")
            if not upper_all.empty and not lower_all.empty:
                ax.plot(upper_all.index, upper_all.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend +2σ")
                ax.plot(lower_all.index, lower_all.values, ":", linewidth=2.0, color="black", alpha=0.85, label="Trend -2σ")
            px_now = _safe_last_float(s)
            if np.isfinite(px_now):
                ax.text(0.99, 0.02, f"Current price: {fmt_price_val(px_now)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.text(0.01, 0.02, f"Slope: {fmt_slope(m_all)}/bar", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.text(0.50, 0.02, f"R² (trend): {fmt_r2(r2_all)}", transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="black", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)
