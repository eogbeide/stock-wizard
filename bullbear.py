# =========================
# Part 1/6 — bullbear.py
# =========================
# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATE: Single latest MACD + S/R reversal signal for trading
#   • Uptrend  → BUY when price successfully reverses up from Support AND MACD crosses 0.0 UP
#   • Downtrend→ SELL when price successfully reverses down from Resistance AND MACD crosses 0.0 DOWN
# Signals are plotted at the MACD 0-cross bar (no HMA55 alignment required).
#
# FIX (prior): NameError crash at Tab 1 selectbox:
#   sel = st.selectbox("Ticker:", universe, key="tab1_ticker")
# → universe is now defined BEFORE tabs are created.
#
# Includes:
#   • Daily + Intraday charts (price + MACD)
#   • SuperTrend line + PSAR line on price chart
#   • MACD 0-cross triangles (trend + S/R reversal filtered)
#   • Tabs: Forecast, Enhanced, Bull/Bear, Metrics, Scanners, Long-Term, Stickers, Support Reversals

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time
import pytz
from matplotlib.transforms import blended_transform_factory

# --- Page config ---
# Sidebar shown by default (button below controls visibility via CSS)
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# NEW (This request): Sidebar show/hide toggle button (no other behavior changes)
# =============================================================================
if "hide_sidebar" not in st.session_state:
    st.session_state.hide_sidebar = False

toggle_label = "Hide Sidebar" if not st.session_state.hide_sidebar else "Show Sidebar"
if st.button(toggle_label, key="btn_toggle_sidebar"):
    st.session_state.hide_sidebar = not st.session_state.hide_sidebar

# Hide the sidebar via CSS when toggled off (button remains in main area to restore it)
if st.session_state.hide_sidebar:
    st.markdown(
        """
<style>
  section[data-testid="stSidebar"] { display: none !important; }
  div[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""",
        unsafe_allow_html=True
    )

# --- Minimal CSS (keep plots readable) ---
st.markdown(
    """
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
""",
    unsafe_allow_html=True
)

# --- Auto-refresh ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def _safe_rerun():
    # Streamlit renamed experimental_rerun → rerun in newer versions
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        _safe_rerun()

auto_refresh()
elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)

st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# =============================================================================
# FIX: Define universe + all “globals” used later BEFORE tabs are created
# =============================================================================

st.sidebar.markdown("---")

# =============================================================================
# NEW (This request): Show Forex/Stocks selector in MAIN AREA (and mirror in sidebar)
#   - Keeps working even when sidebar is hidden.
# =============================================================================
if "mode" not in st.session_state:
    st.session_state.mode = "Forex"  # default

# keep widget states synced across main + sidebar
if "mode_main" not in st.session_state:
    st.session_state.mode_main = st.session_state.mode
if "mode_sidebar" not in st.session_state:
    st.session_state.mode_sidebar = st.session_state.mode

def _sync_mode_from_main():
    st.session_state.mode = st.session_state.mode_main
    st.session_state.mode_sidebar = st.session_state.mode_main

def _sync_mode_from_sidebar():
    st.session_state.mode = st.session_state.mode_sidebar
    st.session_state.mode_main = st.session_state.mode_sidebar

st.markdown("### Market")
st.radio(
    "Choose Forex or Stocks:",
    ["Stocks", "Forex"],
    horizontal=True,
    key="mode_main",
    on_change=_sync_mode_from_main
)

# Sidebar mirror (still useful when sidebar is visible)
st.sidebar.radio(
    "Mode",
    ["Stocks", "Forex"],
    key="mode_sidebar",
    on_change=_sync_mode_from_sidebar
)

mode = st.session_state.mode

def _dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

# Universe (restored full lists; custom tickers removed)
if mode == "Stocks":
    universe = sorted(_dedup_keep_order([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
    ]))
else:
    universe = _dedup_keep_order([
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X','NZDJPY=X','NZDJPY=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
    ])

# When switching Mode, ensure selectbox states remain valid (prevents option mismatch issues).
for _k in ("tab1_ticker", "hist_long_ticker"):
    if _k in st.session_state and st.session_state[_k] not in universe:
        st.session_state[_k] = universe[0]

# Sidebar chart/settings
st.sidebar.markdown("---")
daily_view = st.sidebar.selectbox("Daily view window", ["6M", "12M", "24M", "Historical"], index=1, key="daily_view")
news_window_days = int(st.sidebar.slider("News window (days)", 1, 21, 7, 1, key="news_window_days"))
show_fx_news = st.sidebar.checkbox("Show Yahoo Finance news (Forex only)", value=False, key="show_fx_news")

st.sidebar.markdown("### Indicators")
show_hma = st.sidebar.checkbox("Show HMA(55)", value=True, key="show_hma")
hma_period = int(st.sidebar.number_input("HMA period", min_value=10, max_value=200, value=55, step=1, key="hma_period"))

show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="show_bbands")
bb_win = int(st.sidebar.number_input("BB window", min_value=10, max_value=200, value=20, step=1, key="bb_win"))
bb_mult = float(st.sidebar.number_input("BB multiplier", min_value=1.0, max_value=5.0, value=2.0, step=0.1, key="bb_mult"))
bb_use_ema = st.sidebar.checkbox("BB midline uses EMA", value=False, key="bb_use_ema")

show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun", value=False, key="show_ichi")
ichi_conv = int(st.sidebar.number_input("Ichimoku conversion", 5, 30, 9, 1, key="ichi_conv"))
ichi_base = int(st.sidebar.number_input("Ichimoku base (Kijun)", 10, 60, 26, 1, key="ichi_base"))
ichi_spanb = int(st.sidebar.number_input("Ichimoku span B", 20, 120, 52, 1, key="ichi_spanb"))

show_psar = st.sidebar.checkbox("Show PSAR", value=True, key="show_psar")
psar_step = float(st.sidebar.number_input("PSAR step", min_value=0.001, max_value=0.1, value=0.02, step=0.005, key="psar_step"))
psar_max = float(st.sidebar.number_input("PSAR max step", min_value=0.05, max_value=0.5, value=0.2, step=0.01, key="psar_max"))

st.sidebar.markdown("### SuperTrend")
atr_period = int(st.sidebar.number_input("ATR period", min_value=5, max_value=50, value=10, step=1, key="atr_period"))
atr_mult = float(st.sidebar.number_input("ATR multiplier", min_value=1.0, max_value=10.0, value=3.0, step=0.25, key="atr_mult"))

st.sidebar.markdown("### Regression / Signals")
slope_lb_daily = int(st.sidebar.number_input("Daily regression lookback (bars)", 30, 600, 252, 1, key="slope_lb_daily"))
slope_lb_hourly = int(st.sidebar.number_input("Hourly regression lookback (bars)", 30, 800, 200, 1, key="slope_lb_hourly"))
sr_lb_hourly = int(st.sidebar.number_input("Hourly S/R lookback (bars)", 20, 600, 120, 1, key="sr_lb_hourly"))
sr_prox_pct = float(st.sidebar.slider("S/R / Band proximity (%)", 0.0, 2.0, 0.25, 0.05, key="sr_prox_pct")) / 100.0
rev_bars_confirm = int(st.sidebar.slider("Consecutive bars to confirm", 1, 5, 2, 1, key="rev_bars_confirm"))

st.sidebar.markdown("### Extras")
show_fibs = st.sidebar.checkbox("Show Fibonacci (Intraday)", value=True, key="show_fibs")
show_sessions_pst = st.sidebar.checkbox("Show session lines (Forex)", value=True, key="show_sessions_pst")
ntd_window = int(st.sidebar.number_input("NTD window", 20, 200, 60, 1, key="ntd_window"))

st.sidebar.markdown("### Bull/Bear / Metrics lookback")
bb_period = st.sidebar.selectbox("Lookback period (yfinance)", ["6mo", "1y", "2y", "5y", "max"], index=1, key="bb_period")

# --- Top-of-page caution banner placeholder ---
top_warn = st.empty()

# =========================
# Part 2/6 — bullbear.py
# =========================

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

def is_forex_symbol(symbol: str) -> bool:
    return "=X" in str(symbol).upper()

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
                             symbol: str,
                             confirm_side: str = None) -> str:
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    cs = (confirm_side or "").upper()
    buy_lbl  = "▲ BUY"  + (" (CONFIRMED)" if cs == "BUY"  else "")
    sell_lbl = "▼ SELL" + (" (CONFIRMED)" if cs == "SELL" else "")

    buy_txt  = f"{buy_lbl} @{fmt_price_val(entry_buy)}"
    sell_txt = f"{sell_lbl} @{fmt_price_val(exit_sell)}"
    pips_txt = f" • Value of PIPS: {_diff_text(exit_sell, entry_buy, symbol)}"

    try:
        tslope = float(trend_slope)
    except Exception:
        tslope = 0.0

    if np.isfinite(tslope) and tslope > 0:
        return f"{buy_txt} → {sell_txt}{pips_txt}"
    else:
        return f"{sell_txt} → {buy_txt}{pips_txt}"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans, ha="left", va="center",
        color=color, fontsize=fontsize, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
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

def _simplify_axes(ax):
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass
    ax.tick_params(axis="both", which="both", labelsize=9)
    ax.grid(True, alpha=0.2)

def pad_right_xaxis(ax, frac: float = 0.06):
    try:
        left, right = ax.get_xlim()
        span = right - left
        ax.set_xlim(left, right + span * float(frac))
    except Exception:
        pass

def draw_top_badges(ax, badges: list):
    if not badges:
        return
    y = 1.02
    for text, color in badges:
        ax.text(
            0.01, y, text,
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=9, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95),
        )
        y += 0.055

def draw_instruction_ribbons(ax,
                            trend_slope: float,
                            sup_val: float,
                            res_val: float,
                            px_val: float,
                            symbol: str,
                            confirm_side: str = None,
                            global_slope: float = None,
                            extra_note: str = None):
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    slope_ok = _finite(trend_slope)
    color = "tab:green" if (slope_ok and float(trend_slope) > 0) else "tab:red"

    aligned = True
    if global_slope is not None:
        if not (_finite(trend_slope) and _finite(global_slope)):
            aligned = False
        else:
            aligned = (float(trend_slope) * float(global_slope)) > 0

    if not aligned:
        ax.text(
            0.5, 1.08,
            "ALERT: Global Trendline and Local Slope are opposing — no trade instruction.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.95)
        )
        return

    instr = format_trade_instruction(trend_slope, sup_val, res_val, px_val, symbol, confirm_side=confirm_side)
    if extra_note:
        instr = f"{instr}\n{extra_note}"

    ax.text(
        0.5, 1.08, instr,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95)
    )

# ---------------- Data fetch & core calcs ----------------
def _ensure_tz_index(idx: pd.DatetimeIndex, tz) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return idx
    if idx.tz is None:
        return idx.tz_localize(tz)
    return idx.tz_convert(tz)

@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].asfreq("D").ffill()
    s.index = _ensure_tz_index(s.index, PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].asfreq("D").ffill()
    s.index = _ensure_tz_index(s.index, PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    need = ["Open", "High", "Low", "Close"]
    df = df[[c for c in need if c in df.columns]].dropna()
    df.index = _ensure_tz_index(df.index, PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    # yfinance intraday is typically UTC-naive; treat as UTC then convert
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_close_df_period(ticker: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period)
    except Exception:
        return pd.DataFrame(columns=["Close"])
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        s = raw["Close"].dropna()
    else:
        s = _coerce_1d_series(raw).dropna()
    if isinstance(s, pd.Series) and not s.empty:
        return pd.DataFrame({"Close": s})
    return pd.DataFrame(columns=["Close"])

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        start = (pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1))
        idx = pd.date_range(start=start, periods=30, freq="D", tz=PACIFIC)
        vals = pd.Series(np.nan, index=idx, name="Forecast")
        ci = pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=idx)
        return idx, vals, ci

    if isinstance(series.index, pd.DatetimeIndex):
        series.index = _ensure_tz_index(series.index, PACIFIC)

    if series.shape[0] < 5:
        start = (pd.Timestamp.now(tz=PACIFIC).normalize() + pd.Timedelta(days=1))
        idx = pd.date_range(start=start, periods=30, freq="D", tz=PACIFIC)
        vals = pd.Series(np.nan, index=idx, name="Forecast")
        ci = pd.DataFrame({"lower": np.nan, "upper": np.nan}, index=idx)
        return idx, vals, ci

    try:
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    except Exception:
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

    fc = model.get_forecast(steps=30)
    last_idx = series.index[-1]
    idx = pd.date_range(last_idx + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    mean = fc.predicted_mean
    mean.index = idx
    ci = fc.conf_int()
    ci.index = idx
    return idx, mean, ci

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
    std = float(np.sqrt(np.sum(resid ** 2) / dof))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
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
    color = "tab:green" if m >= 0 else "tab:red"
    ax.plot(s.index, yhat, "-", linewidth=3.2, color=color, label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return m

# --- Supertrend / ATR ---
def _true_range(df: pd.DataFrame):
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = _true_range(df[["High", "Low", "Close"]])
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["ST", "in_uptrend", "upperband", "lowerband"])

    ohlc = df[["High", "Low", "Close"]].copy()
    hl2 = (ohlc["High"] + ohlc["Low"]) / 2.0
    atr = compute_atr(ohlc, atr_period)
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr

    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_up = pd.Series(index=ohlc.index, dtype=bool)

    st_line.iloc[0] = upperband.iloc[0]
    in_up.iloc[0] = True

    for i in range(1, len(ohlc)):
        prev_st = st_line.iloc[i - 1]
        prev_up = bool(in_up.iloc[i - 1])

        up_i = min(upperband.iloc[i], prev_st) if prev_up else upperband.iloc[i]
        dn_i = max(lowerband.iloc[i], prev_st) if (not prev_up) else lowerband.iloc[i]
        close_i = float(ohlc["Close"].iloc[i])

        if close_i > up_i:
            curr_up = True
        elif close_i < dn_i:
            curr_up = False
        else:
            curr_up = prev_up

        in_up.iloc[i] = curr_up
        st_line.iloc[i] = dn_i if curr_up else up_i

    return pd.DataFrame(
        {"ST": st_line, "in_uptrend": in_up, "upperband": upperband, "lowerband": lowerband}
    )

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
    ep = float(df["H"].iloc[0])
    psar[0] = float(df["L"].iloc[0])
    up[0] = True

    for i in range(1, n):
        prev_psar = psar[i - 1]
        if uptrend:
            psar[i] = prev_psar + af * (ep - prev_psar)
            lo1 = float(df["L"].iloc[i - 1])
            lo2 = float(df["L"].iloc[i - 2]) if i >= 2 else lo1
            psar[i] = min(psar[i], lo1, lo2)

            if float(df["H"].iloc[i]) > ep:
                ep = float(df["H"].iloc[i])
                af = min(af + step, max_step)

            if float(df["L"].iloc[i]) < psar[i]:
                uptrend = False
                psar[i] = ep
                ep = float(df["L"].iloc[i])
                af = float(step)
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            hi1 = float(df["H"].iloc[i - 1])
            hi2 = float(df["H"].iloc[i - 2]) if i >= 2 else hi1
            psar[i] = max(psar[i], hi1, hi2)

            if float(df["L"].iloc[i]) < ep:
                ep = float(df["L"].iloc[i])
                af = min(af + step, max_step)

            if float(df["H"].iloc[i]) > psar[i]:
                uptrend = True
                psar[i] = ep
                ep = float(df["H"].iloc[i])
                af = float(step)

        up[i] = uptrend

    return pd.Series(psar, index=df.index, name="PSAR"), pd.Series(up, index=df.index, name="in_uptrend")

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High", "Low"}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["PSAR", "in_uptrend"])
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
        return (pd.Series(index=idx, dtype=float),) * 5

    tenkan = (H.rolling(conv).max() + L.rolling(conv).min()) / 2.0
    kijun = (H.rolling(base).max() + L.rolling(base).min()) / 2.0
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
    mid = s.rolling(window, min_periods=minp).mean() if not use_ema else s.ewm(span=window, adjust=False).mean()
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0, np.nan)
    pctb = ((s - lower) / width).clip(0.0, 1.0)
    nbb = pctb * 2.0 - 1.0
    return mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index)

# --- HMA ---
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

# --- MACD helper ---
def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = (ema_fast - ema_slow).reindex(s.index)
    sig = macd.ewm(span=int(signal), adjust=False).mean().reindex(s.index)
    hist = (macd - sig).reindex(s.index)
    return macd, sig, hist

# --- Reversal detection helpers ---
def _after_all_increasing(series: pd.Series, start_ts, n: int) -> bool:
    s = _coerce_1d_series(series).dropna()
    if start_ts not in s.index or n < 1:
        return False
    seg = s.loc[start_ts:].iloc[: n + 1]
    if len(seg) < n + 1:
        return False
    return bool(np.all(np.diff(seg) > 0))

def _after_all_decreasing(series: pd.Series, start_ts, n: int) -> bool:
    s = _coerce_1d_series(series).dropna()
    if start_ts not in s.index or n < 1:
        return False
    seg = s.loc[start_ts:].iloc[: n + 1]
    if len(seg) < n + 1:
        return False
    return bool(np.all(np.diff(seg) < 0))
