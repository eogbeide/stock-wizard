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
# UPDATE (3): Sidebar shown by default
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
  [data-testid="stSidebar"] { display: none !important; }
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }
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
# UPDATE (1): Select Forex by default
mode = st.sidebar.radio("Mode", ["Stocks", "Forex"], index=1, key="mode")

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

# UPDATE (1): When switching Mode, ensure selectbox states remain valid (prevents option mismatch issues).
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
# =========================
# Part 3/6 — bullbear.py
# =========================

# --- Core signals used by the UI (these were referenced in your snippet) ---
def last_band_reversal_signal(price: pd.Series,
                              band_upper: pd.Series,
                              band_lower: pd.Series,
                              trend_slope: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 2,
                              lookback: int = 120):
    """
    Trend-aware band reversal.
      Uptrend: BUY when price was below lower band and returns back above it (near it).
      Downtrend: SELL when price was above upper band and returns back below it (near it).
    Returns latest signal in lookback window.
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    if p.shape[0] < max(3, confirm_bars + 1) or not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    up = _coerce_1d_series(band_upper).reindex(p.index)
    lo = _coerce_1d_series(band_lower).reindex(p.index)
    m = p.notna() & up.notna() & lo.notna()
    p, up, lo = p[m], up[m], lo[m]
    if len(p) < max(3, confirm_bars + 1):
        return None

    if lookback and len(p) > lookback:
        p = p.iloc[-lookback:]
        up = up.iloc[-lookback:]
        lo = lo.iloc[-lookback:]

    for i in range(len(p) - 1, 1, -1):
        p_prev, p_now = float(p.iloc[i - 1]), float(p.iloc[i])
        up_prev, up_now = float(up.iloc[i - 1]), float(up.iloc[i])
        lo_prev, lo_now = float(lo.iloc[i - 1]), float(lo.iloc[i])

        if trend_slope > 0:
            was_below = p_prev < lo_prev * (1.0 + prox)
            now_above = p_now >= lo_now * (1.0 + prox)
            if was_below and now_above:
                t = p.index[i]
                if _after_all_increasing(p, t, confirm_bars):
                    return {"time": t, "price": p_now, "side": "BUY"}
        else:
            was_above = p_prev > up_prev * (1.0 - prox)
            now_below = p_now <= up_now * (1.0 - prox)
            if was_above and now_below:
                t = p.index[i]
                if _after_all_decreasing(p, t, confirm_bars):
                    return {"time": t, "price": p_now, "side": "SELL"}

    return None

def last_reversal_star(price: pd.Series,
                       trend_slope: float,
                       lookback: int = 20,
                       confirm_bars: int = 2):
    """
    Trend-aware star:
      Uptrend  -> prefer trough reversals
      Downtrend-> prefer peak reversals
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    if p.shape[0] < max(5, confirm_bars + 2) or not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    tail = p.iloc[-min(int(lookback), len(p)):]
    if tail.shape[0] < max(5, confirm_bars + 2):
        return None

    # scan backward for pivot
    for i in range(len(tail) - confirm_bars - 1, 1, -1):
        t = tail.index[i]
        v = float(tail.iloc[i])
        left = tail.iloc[max(0, i - 2): i]
        right = tail.iloc[i + 1: i + 1 + confirm_bars]
        if len(right) < confirm_bars:
            continue

        if trend_slope > 0:
            # trough
            if (v <= float(left.min())) and _after_all_increasing(tail, t, confirm_bars):
                return {"time": t, "price": v, "kind": "trough"}
        else:
            # peak
            if (v >= float(left.max())) and _after_all_decreasing(tail, t, confirm_bars):
                return {"time": t, "price": v, "kind": "peak"}
    return None

def last_hma_cross_star(price: pd.Series,
                        hma: pd.Series,
                        trend_slope: float,
                        lookback: int = 30):
    """
    Daily rule in your changelog:
      Up slope  -> Buy ★ when price crosses ABOVE HMA
      Down slope-> Sell ★ when price crosses BELOW HMA
    Returns {"time","price","kind"} where kind trough=buy, peak=sell.
    """
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    p = _coerce_1d_series(price).astype(float).dropna()
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)
    m = p.notna() & h.notna()
    p, h = p[m], h[m]
    if len(p) < 2:
        return None
    if lookback and len(p) > lookback:
        p = p.iloc[-lookback:]
        h = h.iloc[-lookback:]

    for i in range(len(p) - 1, 0, -1):
        p_prev, p_now = float(p.iloc[i - 1]), float(p.iloc[i])
        h_prev, h_now = float(h.iloc[i - 1]), float(h.iloc[i])
        if trend_slope > 0:
            if (p_prev < h_prev) and (p_now >= h_now):
                return {"time": p.index[i], "price": p_now, "kind": "trough"}
        else:
            if (p_prev > h_prev) and (p_now <= h_now):
                return {"time": p.index[i], "price": p_now, "kind": "peak"}
    return None

# --- Annotation helpers ---
def annotate_signal_box(ax, ts, px, side: str, note: str = "", ypad_frac: float = 0.045):
    text = f"{'▲ BUY' if side=='BUY' else '▼ SELL'}" + (f" {note}" if note else "")
    try:
        ymin, ymax = ax.get_ylim()
        yr = (ymax - ymin) if np.isfinite(ymax) and np.isfinite(ymin) else 1.0
        yoff = yr * ypad_frac * (1 if side == "BUY" else -1)
        ax.annotate(
            text,
            xy=(ts, px),
            xytext=(ts, px + yoff),
            textcoords="data",
            ha="left",
            va="bottom" if side == "BUY" else "top",
            fontsize=10,
            fontweight="bold",
            color="tab:green" if side == "BUY" else "tab:red",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=("tab:green" if side == "BUY" else "tab:red"), alpha=0.9),
            arrowprops=dict(arrowstyle="->", color=("tab:green" if side == "BUY" else "tab:red"), lw=1.5),
            zorder=9,
        )
        ax.scatter([ts], [px], s=60, c=("tab:green" if side == "BUY" else "tab:red"), zorder=10)
    except Exception:
        ax.text(ts, px, f" {text}", color=("tab:green" if side == "BUY" else "tab:red"),
                fontsize=10, fontweight="bold")

def annotate_star(ax, ts, px, kind: str, show_text: bool = False, color_override: str = None):
    color = color_override if color_override else ("tab:red" if kind == "peak" else "tab:green")
    label = "★ Peak REV" if kind == "peak" else "★ Trough REV"
    try:
        ax.scatter([ts], [px], marker="*", s=160, c=color, zorder=12, edgecolors="none")
        if show_text:
            ymin, ymax = ax.get_ylim()
            yoff = (ymax - ymin) * (0.02 if kind == "trough" else -0.02)
            ax.text(ts, px + yoff, label, color=color, fontsize=9, fontweight="bold",
                    ha="left", va="bottom" if kind == "trough" else "top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
                    zorder=12)
    except Exception:
        ax.text(ts, px, "★", color=color, fontsize=12, fontweight="bold")

def annotate_buy_triangle(ax, ts, px, size: int = 140):
    try:
        ax.scatter([ts], [px], marker="^", s=size, c="tab:green", edgecolors="none", zorder=12)
    except Exception:
        ax.text(ts, px, "▲", color="tab:green", fontsize=12, fontweight="bold", zorder=12)

def annotate_sell_triangle(ax, ts, px, size: int = 140):
    try:
        ax.scatter([ts], [px], marker="v", s=size, c="tab:red", edgecolors="none", zorder=12)
    except Exception:
        ax.text(ts, px, "▼", color="tab:red", fontsize=12, fontweight="bold", zorder=12)

def annotate_macd_star_callout(ax, ts_or_x, px, side: str, hma_period: int = 55, y_frac: float = 0.10):
    """
    Star at (x, price) + instruction box anchored at the bottom of the chart
    with a leader line pointing to the star.
    """
    try:
        color = "tab:green" if side == "BUY" else "tab:red"
        ax.scatter([ts_or_x], [px], marker="*", s=170, c=color, edgecolors="none", zorder=14)

        label = f"★ MACD {'Buy' if side=='BUY' else 'Sell'} — HMA{hma_period} Cross @{fmt_price_val(px)}"

        # Place the box near the bottom of the chart (x in data coords, y in axes fraction)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate(
            label,
            xy=(ts_or_x, px),
            xycoords="data",
            xytext=(ts_or_x, float(y_frac)),
            textcoords=trans,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=color, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            zorder=15,
            annotation_clip=False,
        )
    except Exception:
        ax.text(ts_or_x, px, "★", color=("tab:green" if side == "BUY" else "tab:red"),
                fontsize=12, fontweight="bold", zorder=14)

def last_macd_hma_cross_star(price: pd.Series,
                             hma: pd.Series,
                             macd: pd.Series,
                             trend_slope: float,
                             lookback: int = 120):
    """
    UPDATE (1) & (2) — Requested logic:

    Uptrend:  MACD Buy only when:
      - trend_slope > 0
      - price crosses UP through HMA
      - MACD is still < 0 at that cross bar

    Downtrend: MACD Sell only when:
      - trend_slope < 0
      - price crosses DOWN through HMA
      - MACD is still > 0 at that cross bar
    """
    if not np.isfinite(trend_slope) or trend_slope == 0:
        return None
    p = _coerce_1d_series(price).astype(float).dropna()
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)
    m = _coerce_1d_series(macd).astype(float).reindex(p.index)
    mask = p.notna() & h.notna() & m.notna()
    p, h, m = p[mask], h[mask], m[mask]
    if len(p) < 2:
        return None
    if lookback and len(p) > lookback:
        p = p.iloc[-lookback:]; h = h.iloc[-lookback:]; m = m.iloc[-lookback:]

    for i in range(len(p) - 1, 0, -1):
        p_prev, p_cur = float(p.iloc[i - 1]), float(p.iloc[i])
        h_prev, h_cur = float(h.iloc[i - 1]), float(h.iloc[i])
        mac_cur = float(m.iloc[i])
        t = p.index[i]

        if float(trend_slope) > 0:
            up_cross = (p_prev < h_prev) and (p_cur >= h_cur)
            if up_cross and (mac_cur < 0.0):
                return {"time": t, "price": p_cur, "side": "BUY"}
        else:
            dn_cross = (p_prev > h_prev) and (p_cur <= h_cur)
            if dn_cross and (mac_cur > 0.0):
                return {"time": t, "price": p_cur, "side": "SELL"}
    return None

# --- alignment helper ---
def _align_series_to_index(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = _coerce_1d_series(s)
    if s.empty:
        return pd.Series(index=idx, dtype=float)
    out = s.reindex(idx)
    try:
        out = out.interpolate(method="time").ffill().bfill()
    except Exception:
        out = out.ffill().bfill()
    return out.reindex(idx)

def plot_supertrend_line(ax, x_vals, st_df: pd.DataFrame, idx: pd.DatetimeIndex, use_positions: bool = False,
                         lw: float = 1.8, alpha: float = 0.75):
    if st_df is None or st_df.empty or "ST" not in st_df.columns:
        return

    st_line = _align_series_to_index(st_df["ST"], idx)

    in_up = None
    if "in_uptrend" in st_df.columns:
        in_up = st_df["in_uptrend"].reindex(idx)
        try:
            in_up = in_up.ffill().bfill().astype(bool)
        except Exception:
            in_up = pd.Series(index=idx, dtype=bool)

    if in_up is None or in_up.dropna().empty:
        if use_positions:
            ax.plot(x_vals, st_line.values, "-", linewidth=lw, alpha=alpha, label="SuperTrend")
        else:
            ax.plot(idx, st_line.values, "-", linewidth=lw, alpha=alpha, label="SuperTrend")
        return

    up_line = st_line.where(in_up == True)
    dn_line = st_line.where(in_up == False)

    if use_positions:
        ax.plot(x_vals, up_line.values, "-", linewidth=lw, alpha=alpha, color="tab:green", label="SuperTrend")
        ax.plot(x_vals, dn_line.values, "-", linewidth=lw, alpha=alpha, color="tab:red", label="_nolegend_")
    else:
        ax.plot(idx, up_line.values, "-", linewidth=lw, alpha=alpha, color="tab:green", label="SuperTrend")
        ax.plot(idx, dn_line.values, "-", linewidth=lw, alpha=alpha, color="tab:red", label="_nolegend_")
# =========================
# Part 4/6 — bullbear.py
# =========================

# --- Breakout detection (used in your snippet) ---
def last_breakout_signal(price: pd.Series,
                         resistance: pd.Series,
                         support: pd.Series,
                         prox: float = 0.0,
                         confirm_bars: int = 1):
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(3, confirm_bars + 1):
        return None
    r_prev = _coerce_1d_series(resistance).reindex(p.index).shift(1)
    s_prev = _coerce_1d_series(support).reindex(p.index).shift(1)

    if not (np.isfinite(p.iloc[-1]) and np.isfinite(p.iloc[-2])):
        return None

    def _confirmed_up():
        k = max(1, int(confirm_bars))
        p_tail = p.iloc[-k:]
        r_tail = r_prev.iloc[-k:]
        mask = p_tail.notna() & r_tail.notna()
        if mask.sum() < k:
            return False
        return bool(np.all(p_tail[mask] > (r_tail[mask] * (1.0 + prox))))

    def _confirmed_down():
        k = max(1, int(confirm_bars))
        p_tail = p.iloc[-k:]
        s_tail = s_prev.iloc[-k:]
        mask = p_tail.notna() & s_tail.notna()
        if mask.sum() < k:
            return False
        return bool(np.all(p_tail[mask] < (s_tail[mask] * (1.0 - prox))))

    up_ok = (
        np.isfinite(r_prev.iloc[-1]) and np.isfinite(r_prev.iloc[-2])
        and (p.iloc[-2] <= r_prev.iloc[-2] * (1.0 + prox))
        and (p.iloc[-1] > r_prev.iloc[-1] * (1.0 + prox))
        and _confirmed_up()
    )

    dn_ok = (
        np.isfinite(s_prev.iloc[-1]) and np.isfinite(s_prev.iloc[-2])
        and (p.iloc[-2] >= s_prev.iloc[-2] * (1.0 - prox))
        and (p.iloc[-1] < s_prev.iloc[-1] * (1.0 - prox))
        and _confirmed_down()
    )

    if up_ok:
        return {"time": p.index[-1], "price": float(p.iloc[-1]), "dir": "UP"}
    if dn_ok:
        return {"time": p.index[-1], "price": float(p.iloc[-1]), "dir": "DOWN"}
    return None

def annotate_breakout(ax, ts, px, direction: str):
    try:
        if direction == "UP":
            ax.scatter([ts], [px], marker="D", s=110, c="tab:green", edgecolors="none", zorder=13)
        else:
            ax.scatter([ts], [px], marker="v", s=110, c="tab:red", edgecolors="none", zorder=13)
    except Exception:
        ax.text(ts, px, "B/O" if direction == "UP" else "B/D",
                color=("tab:green" if direction == "UP" else "tab:red"),
                fontsize=9, fontweight="bold", zorder=13)

# --- Fibonacci extreme reversal detection & marker ---
def last_fib_extreme_reversal(price: pd.Series,
                              slope: float,
                              fib0_level: float,
                              fib100_level: float,
                              prox: float = 0.0025,
                              confirm_bars: int = 2,
                              lookback: int = 40):
    if not np.isfinite(slope):
        return None
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(5, confirm_bars + 2):
        return None

    tail = p.iloc[-min(lookback, len(p)):]
    last_time = p.index[-1]
    sig = None

    if np.isfinite(fib0_level) and slope < 0:
        near0 = (np.abs(tail - fib0_level) / max(1e-12, abs(fib0_level))) <= prox
        if near0.any():
            t_touch0 = tail[near0].index[-1]
            if _after_all_decreasing(p, t_touch0, confirm_bars):
                if float(p.iloc[-1]) < float(p.loc[t_touch0]):
                    sig = {"dir": "DOWN", "time": last_time, "level": float(fib0_level), "t_touch": t_touch0}

    if np.isfinite(fib100_level) and slope > 0:
        near100 = (np.abs(tail - fib100_level) / max(1e-12, abs(fib100_level))) <= prox
        if near100.any():
            t_touch100 = tail[near100].index[-1]
            if _after_all_increasing(p, t_touch100, confirm_bars):
                if float(p.iloc[-1]) > float(p.loc[t_touch100]):
                    cand = {"dir": "UP", "time": last_time, "level": float(fib100_level), "t_touch": t_touch100}
                    if sig is None or cand["t_touch"] > sig["t_touch"]:
                        sig = cand
    return sig

def annotate_fib_reversal(ax, ts, y_level: float, direction: str, label: str = ""):
    try:
        color = "tab:red" if direction == "DOWN" else "tab:green"
        marker = "v" if direction == "DOWN" else "^"
        ax.scatter([ts], [y_level], marker=marker, s=130, c=color, edgecolors="none", zorder=14)
        ymin, ymax = ax.get_ylim()
        yoff = (ymax - ymin) * (0.025 if direction == "UP" else -0.025)
        if label:
            ax.text(ts, y_level + yoff, label,
                    ha="left", va="bottom" if direction == "UP" else "top",
                    fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.95),
                    zorder=14)
    except Exception:
        ax.text(ts, y_level, "Fib REV", color=("tab:red" if direction == "DOWN" else "tab:green"),
                fontsize=9, fontweight="bold", zorder=14)

# --- Daily-only 99% SR reversal logic ---
Z_FOR_99 = 2.576  # ≈ 99% two-sided (~2.58σ)

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
                                trend_slope: float,
                                prox: float = 0.0025,
                                confirm_bars: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.shape[0] < max(3, confirm_bars + 1) or not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    sup = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    res = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    up99 = _coerce_1d_series(upper99).reindex(p.index)
    lo99 = _coerce_1d_series(lower99).reindex(p.index)

    if sup.dropna().empty or res.dropna().empty or up99.dropna().empty or lo99.dropna().empty:
        return None
    if len(up99) < 2 or len(lo99) < 2:
        return None

    def _inc_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n + 1:
            return False
        return bool(np.all(np.diff(s.iloc[-(n + 1):]) > 0))

    def _dec_ok(series: pd.Series, n: int) -> bool:
        s = _coerce_1d_series(series).dropna()
        if len(s) < n + 1:
            return False
        return bool(np.all(np.diff(s.iloc[-(n + 1):]) < 0))

    t0 = p.index[-1]
    c0, c1 = float(p.iloc[-1]), float(p.iloc[-2])
    s0, s1 = float(sup.iloc[-1]), float(sup.iloc[-2])
    r0, r1 = float(res.iloc[-1]), float(res.iloc[-2])
    u1 = float(up99.iloc[-2]) if np.isfinite(up99.iloc[-2]) else np.nan
    l1 = float(lo99.iloc[-2]) if np.isfinite(lo99.iloc[-2]) else np.nan

    if trend_slope > 0:
        prev_near_support = (c1 <= s1 * (1.0 + prox))
        support_near_99 = _rel_near(s1, l1, prox)
        going_up = _inc_ok(p, confirm_bars)
        if prev_near_support and support_near_99 and going_up:
            return {"time": t0, "price": c0, "side": "BUY", "note": "ALERT 99% SR REV"}
    else:
        prev_near_resistance = (c1 >= r1 * (1.0 - prox))
        resistance_near_99 = _rel_near(r1, u1, prox)
        going_down = _dec_ok(p, confirm_bars)
        if prev_near_resistance and resistance_near_99 and going_down:
            return {"time": t0, "price": c0, "side": "SELL", "note": "ALERT 99% SR REV"}
    return None

def last_macd_zero_cross_confident(macd: pd.Series,
                                   trend_slope: float,
                                   z: float = Z_FOR_99,
                                   vol_lookback: int = 60,
                                   scan_back: int = 160):
    """
    Latest MACD 0-line cross that is '99% confident' (|MACD| >= z*rolling_std),
    filtered by trend_slope sign:
      Uptrend: BUY when crosses up through 0 and MACD >= z*σ
      Downtrend: SELL when crosses down through 0 and -MACD >= z*σ
    """
    m = _coerce_1d_series(macd).dropna()
    if m.shape[0] < 3 or not np.isfinite(trend_slope) or trend_slope == 0:
        return None

    k = int(max(10, vol_lookback // 3))
    sig = m.rolling(int(vol_lookback), min_periods=k).std().replace(0, np.nan)

    start = max(1, len(m) - int(scan_back))
    for i in range(len(m) - 1, start - 1, -1):
        prev = float(m.iloc[i - 1]) if np.isfinite(m.iloc[i - 1]) else np.nan
        curr = float(m.iloc[i]) if np.isfinite(m.iloc[i]) else np.nan
        sgm = float(sig.iloc[i]) if np.isfinite(sig.iloc[i]) else np.nan
        if not (np.isfinite(prev) and np.isfinite(curr) and np.isfinite(sgm) and sgm > 0):
            continue

        if float(trend_slope) > 0:
            crossed = (prev <= 0.0) and (curr > 0.0)
            confident = (curr >= float(z) * sgm)
            if crossed and confident:
                return {"time": m.index[i], "value": curr, "side": "BUY"}
        else:
            crossed = (prev >= 0.0) and (curr < 0.0)
            confident = ((-curr) >= float(z) * sgm)
            if crossed and confident:
                return {"time": m.index[i], "value": curr, "side": "SELL"}
    return None

# --- Support-touch scanner helper ---
def find_support_touch_confirmed_up(price: pd.Series,
                                    support: pd.Series,
                                    prox: float = 0.0025,
                                    confirm_bars: int = 2,
                                    lookback_bars: int = 30):
    p = _coerce_1d_series(price).dropna()
    s = _coerce_1d_series(support).reindex(p.index).ffill().bfill()
    if p.shape[0] < confirm_bars + 2 or s.dropna().empty:
        return None

    tail = p.iloc[-(lookback_bars + confirm_bars + 1):]
    for t in reversed(tail.iloc[:-confirm_bars].index.tolist()):
        try:
            pc = float(p.loc[t]); sc = float(s.loc[t])
        except Exception:
            continue
        if not (np.isfinite(pc) and np.isfinite(sc)):
            continue
        touched = (pc <= sc * (1.0 + prox))
        if not touched:
            continue
        if _after_all_increasing(p, t, confirm_bars):
            now_close = float(p.iloc[-1])
            gain_pct = (now_close - pc) / pc if pc != 0 else np.nan
            try:
                loc = p.index.get_loc(t)
                if isinstance(loc, slice):
                    loc = loc.start
                bars_since = int((len(p) - 1) - loc)
            except Exception:
                bars_since = np.nan
            return {
                "t_touch": t,
                "touch_close": pc,
                "support": sc,
                "now_close": now_close,
                "gain_pct": gain_pct,
                "bars_since_touch": bars_since
            }
    return None

# NEW (This request): resistance-touch scanner helper (mirror of support-touch)
def find_resistance_touch_confirmed_down(price: pd.Series,
                                        resistance: pd.Series,
                                        prox: float = 0.0025,
                                        confirm_bars: int = 2,
                                        lookback_bars: int = 30):
    p = _coerce_1d_series(price).dropna()
    r = _coerce_1d_series(resistance).reindex(p.index).ffill().bfill()
    if p.shape[0] < confirm_bars + 2 or r.dropna().empty:
        return None

    tail = p.iloc[-(lookback_bars + confirm_bars + 1):]
    for t in reversed(tail.iloc[:-confirm_bars].index.tolist()):
        try:
            pc = float(p.loc[t]); rc = float(r.loc[t])
        except Exception:
            continue
        if not (np.isfinite(pc) and np.isfinite(rc)):
            continue
        touched = (pc >= rc * (1.0 - prox))
        if not touched:
            continue
        if _after_all_decreasing(p, t, confirm_bars):
            now_close = float(p.iloc[-1])
            drop_pct = (pc - now_close) / pc if pc != 0 else np.nan
            try:
                loc = p.index.get_loc(t)
                if isinstance(loc, slice):
                    loc = loc.start
                bars_since = int((len(p) - 1) - loc)
            except Exception:
                bars_since = np.nan
            return {
                "t_touch": t,
                "touch_close": pc,
                "resistance": rc,
                "now_close": now_close,
                "drop_pct": drop_pct,
                "bars_since_touch": bars_since
            }
    return None

# NEW (This request): MACD 0-cross + S/R reversal (trend-filtered) signal
# (kept for Tab 7 optional scan; chart signals were updated per request)
def last_macd_zero_cross_with_sr_reversal(price: pd.Series,
                                         support: pd.Series,
                                         resistance: pd.Series,
                                         macd: pd.Series,
                                         trend_slope: float,
                                         prox: float = 0.0025,
                                         confirm_bars: int = 2,
                                         lookback_macd: int = 160,
                                         sr_lookback: int = 60):
    """
    BUY when:
      - trend_slope > 0 (global uptrend)
      - MACD crosses 0.0 UP  (prev <= 0, curr > 0)
      - Prior to the cross, price touched Support (within prox) AND reversal was confirmed

    SELL when:
      - trend_slope < 0 (global downtrend)
      - MACD crosses 0.0 DOWN (prev >= 0, curr < 0)
      - Prior to the cross, price touched Resistance (within prox) AND reversal was confirmed

    Returns the latest matching signal within lookback_macd.
    """
    if not np.isfinite(trend_slope) or float(trend_slope) == 0.0:
        return None

    p = _coerce_1d_series(price).astype(float).dropna()
    if p.shape[0] < max(3, confirm_bars + 2):
        return None

    sup = _coerce_1d_series(support).astype(float).reindex(p.index)
    res = _coerce_1d_series(resistance).astype(float).reindex(p.index)
    m = _coerce_1d_series(macd).astype(float).reindex(p.index)

    mask = p.notna() & sup.notna() & res.notna() & m.notna()
    p, sup, res, m = p[mask], sup[mask], res[mask], m[mask]
    if len(p) < max(3, confirm_bars + 2):
        return None

    if lookback_macd and len(p) > int(lookback_macd):
        p = p.iloc[-int(lookback_macd):]
        sup = sup.iloc[-int(lookback_macd):]
        res = res.iloc[-int(lookback_macd):]
        m = m.iloc[-int(lookback_macd):]

    sr_lb = max(5, int(sr_lookback))
    cb = max(1, int(confirm_bars))

    for i in range(len(p) - 1, 0, -1):
        m_prev, m_cur = float(m.iloc[i - 1]), float(m.iloc[i])
        t_cross = p.index[i]
        p_cross = float(p.iloc[i])

        if float(trend_slope) > 0:
            macd_cross = (m_prev <= 0.0) and (m_cur > 0.0)
            if not macd_cross:
                continue

            j0 = max(0, i - sr_lb)
            for j in range(i - 1, j0 - 1, -1):
                t_touch = p.index[j]
                p_touch = float(p.iloc[j])
                s_touch = float(sup.iloc[j])

                if not (np.isfinite(p_touch) and np.isfinite(s_touch)):
                    continue

                touched_support = (p_touch <= s_touch * (1.0 + prox))
                if not touched_support:
                    continue

                # Ensure confirmation window completes on/before MACD cross bar
                if (j + cb) > i:
                    continue

                if _after_all_increasing(p, t_touch, cb):
                    bars_since = (len(p) - 1) - i
                    return {
                        "time": t_cross,
                        "price": p_cross,
                        "side": "BUY",
                        "macd_value": float(m_cur),
                        "t_touch": t_touch,
                        "sr_level": s_touch,
                        "bars_since": int(bars_since),
                    }

        else:
            macd_cross = (m_prev >= 0.0) and (m_cur < 0.0)
            if not macd_cross:
                continue

            j0 = max(0, i - sr_lb)
            for j in range(i - 1, j0 - 1, -1):
                t_touch = p.index[j]
                p_touch = float(p.iloc[j])
                r_touch = float(res.iloc[j])

                if not (np.isfinite(p_touch) and np.isfinite(r_touch)):
                    continue

                touched_res = (p_touch >= r_touch * (1.0 - prox))
                if not touched_res:
                    continue

                if (j + cb) > i:
                    continue

                if _after_all_decreasing(p, t_touch, cb):
                    bars_since = (len(p) - 1) - i
                    return {
                        "time": t_cross,
                        "price": p_cross,
                        "side": "SELL",
                        "macd_value": float(m_cur),
                        "t_touch": t_touch,
                        "sr_level": r_touch,
                        "bars_since": int(bars_since),
                    }

    return None

# --- Sessions & News ---
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    start_d = idx[0].astimezone(session_tz).date()
    end_d = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            dt_open_local = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close = session_markers_for_index(idx, NY_TZ, 8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}

def draw_session_lines(ax, lines: dict):
    ax.plot([], [], linestyle="-", color="tab:blue", label="London Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:blue", label="London Close (PST)")
    ax.plot([], [], linestyle="-", color="tab:orange", label="New York Open (PST)")
    ax.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:blue", alpha=0.35)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", color="tab:blue", alpha=0.35)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", color="tab:orange", linewidth=1.0, alpha=0.35)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", color="tab:orange", linewidth=1.0, alpha=0.35)
    ax.text(
        0.99, 0.98, "Session times in PST",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="grey", alpha=0.7)
    )

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
        rows.append({"time": dt_pst, "title": item.get("title", ""), "publisher": item.get("publisher", ""), "link": item.get("link", "")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

# --- Market-time compressed axis (intraday plotted as numeric positions) ---
def make_market_time_formatter(index: pd.DatetimeIndex) -> FuncFormatter:
    def _fmt(x, _pos=None):
        i = int(round(x))
        if 0 <= i < len(index):
            ts = index[i]
            return ts.strftime("%m-%d %H:%M")
        return ""
    return FuncFormatter(_fmt)

def map_times_to_positions(index: pd.DatetimeIndex, times: list):
    pos = []
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return pos
    for t in times:
        try:
            tt = pd.Timestamp(t)
            if tt.tzinfo is None and index.tz is not None:
                tt = tt.tz_localize(index.tz)
            elif index.tz is not None:
                tt = tt.tz_convert(index.tz)
            j = index.get_indexer([tt], method="nearest")[0]
        except Exception:
            j = -1
        if j != -1:
            pos.append(j)
    return pos

def map_session_lines_to_positions(lines: dict, index: pd.DatetimeIndex):
    return {k: map_times_to_positions(index, v) for k, v in lines.items()}

def market_time_axis(ax, index: pd.DatetimeIndex):
    ax.set_xlim(0, max(0, len(index) - 1))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_major_formatter(make_market_time_formatter(index))

def render_daily_price_macd(sel: str, df: pd.Series, df_ohlc: pd.DataFrame):
    """
    Daily: price + MACD panel.
    Includes: trendline + 99% band, S/R, HMA, BB, Kijun, PSAR, SuperTrend,
              reversal star, breakout marker, instruction banner,
              MACD HMA-cross star callout (per requested rules).
    """
    df = _coerce_1d_series(df).dropna()
    if df.empty:
        st.warning("No daily data available.")
        return

    ema30 = df.ewm(span=30).mean()
    res30 = df.rolling(30, min_periods=1).max()
    sup30 = df.rolling(30, min_periods=1).min()

    yhat_d, upper_d, lower_d, m_d, r2_d = regression_with_band(df, slope_lb_daily, z=Z_FOR_99)

    kijun_d = pd.Series(index=df.index, dtype=float)
    if df_ohlc is not None and not df_ohlc.empty and show_ichi and {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        _, kijun_d, _, _, _ = ichimoku_lines(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"],
                                             conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
        kijun_d = kijun_d.ffill().bfill()

    bb_mid_d, bb_up_d, bb_lo_d, bb_pctb_d, bb_nbb_d = compute_bbands(df, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    psar_d_df = compute_psar_from_ohlc(df_ohlc, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    st_d_df = compute_supertrend(df_ohlc, atr_period=atr_period, atr_mult=atr_mult) if (df_ohlc is not None and not df_ohlc.empty) else pd.DataFrame()

    df_show = subset_by_daily_view(df, daily_view)

    psar_d_show = _align_series_to_index(psar_d_df["PSAR"], df_show.index) if (show_psar and not psar_d_df.empty and "PSAR" in psar_d_df) else pd.Series(index=df_show.index, dtype=float)
    ema30_show = ema30.reindex(df_show.index)
    res30_show = res30.reindex(df_show.index)
    sup30_show = sup30.reindex(df_show.index)
    yhat_d_show = yhat_d.reindex(df_show.index) if not yhat_d.empty else yhat_d
    upper_d_show = upper_d.reindex(df_show.index) if not upper_d.empty else upper_d
    lower_d_show = lower_d.reindex(df_show.index) if not lower_d.empty else lower_d
    kijun_d_show = kijun_d.reindex(df_show.index).ffill().bfill()
    bb_mid_d_show = bb_mid_d.reindex(df_show.index)
    bb_up_d_show = bb_up_d.reindex(df_show.index)
    bb_lo_d_show = bb_lo_d.reindex(df_show.index)
    bb_pctb_d_show = bb_pctb_d.reindex(df_show.index)
    bb_nbb_d_show = bb_nbb_d.reindex(df_show.index)

    hma_d_full = compute_hma(df, period=hma_period).reindex(df_show.index)

    macd_d, _sig_unused, macd_hist_d = compute_macd(df_show)

    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]}
    )
    plt.subplots_adjust(top=0.88, right=0.995, left=0.06, hspace=0.05)

    ax.set_title(f"{sel} Daily — {daily_view}")
    ax.plot(df_show.index, df_show.values, label="Price", linewidth=1.4)
    ax.plot(ema30_show.index, ema30_show.values, "--", alpha=0.4, linewidth=1.0, label="_nolegend_")

    if show_bbands and not bb_up_d_show.dropna().empty and not bb_lo_d_show.dropna().empty:
        ax.fill_between(df_show.index, bb_lo_d_show, bb_up_d_show, alpha=0.04, label="_nolegend_")
        ax.plot(bb_mid_d_show.index, bb_mid_d_show.values, "-", linewidth=0.9, alpha=0.35, label="_nolegend_")

    if show_ichi and not kijun_d_show.dropna().empty:
        ax.plot(kijun_d_show.index, kijun_d_show.values, "-", linewidth=1.2, color="black", alpha=0.55, label="Kijun")

    if show_hma and not hma_d_full.dropna().empty:
        ax.plot(hma_d_full.index, hma_d_full.values, "-", linewidth=1.3, alpha=0.9, label="HMA")

    if show_psar and not psar_d_show.dropna().empty:
        ax.plot(psar_d_show.index, psar_d_show.values, "-", linewidth=1.8, color="purple", alpha=0.95, label="PSAR", zorder=6)

    if st_d_df is not None and not st_d_df.empty:
        plot_supertrend_line(ax, None, st_d_df, df_show.index, use_positions=False, lw=1.8, alpha=0.75)

    if not yhat_d_show.empty:
        slope_col_d = "tab:green" if m_d >= 0 else "tab:red"
        ax.plot(yhat_d_show.index, yhat_d_show.values, "-", linewidth=3.2, color=slope_col_d, label="Trend")
    if not upper_d_show.empty and not lower_d_show.empty:
        ax.plot(upper_d_show.index, upper_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
        ax.plot(lower_d_show.index, lower_d_show.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")

    if len(df_show) > 1:
        draw_trend_direction_line(ax, df_show, label_prefix="")

    # S/R lines
    try:
        res_val_d = float(res30_show.iloc[-1])
        sup_val_d = float(sup30_show.iloc[-1])
        if np.isfinite(res_val_d) and np.isfinite(sup_val_d):
            ax.hlines(res_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                      colors="tab:red", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
            ax.hlines(sup_val_d, xmin=df_show.index[0], xmax=df_show.index[-1],
                      colors="tab:green", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
            label_on_left(ax, res_val_d, f"R {fmt_price_val(res_val_d)}", color="tab:red")
            label_on_left(ax, sup_val_d, f"S {fmt_price_val(sup_val_d)}", color="tab:green")
    except Exception:
        res_val_d = sup_val_d = np.nan

    # UPDATE (2): Fibonacci on the DAILY chart (uses the same show_fibs toggle)
    if show_fibs and not df_show.empty:
        fibs_d = fibonacci_levels(df_show)
        for _lbl, y in fibs_d.items():
            ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1],
                      linestyles="dotted", linewidth=0.6, alpha=0.35)
        for lbl, y in fibs_d.items():
            ax.text(df_show.index[-1], y, f" {lbl}", va="center",
                    fontsize=8, alpha=0.6, fontweight="bold")

        # Also show Fib 0%/100% extreme reversal marker + label (same as intraday)
        try:
            fib0 = fibs_d.get("0%")
            fib100 = fibs_d.get("100%")
            if np.isfinite(fib0) and np.isfinite(fib100):
                fib_sig_d = last_fib_extreme_reversal(
                    price=df_show,
                    slope=m_d,  # daily uses global regression slope
                    fib0_level=float(fib0),
                    fib100_level=float(fib100),
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback=max(40, int(slope_lb_daily))
                )
                if fib_sig_d is not None:
                    if fib_sig_d["dir"] == "DOWN":
                        annotate_fib_reversal(ax, ts=df_show.index[-1], y_level=float(fib0),
                                              direction="DOWN", label="Fib 0% REV → 100%")
                    elif fib_sig_d["dir"] == "UP":
                        annotate_fib_reversal(ax, ts=df_show.index[-1], y_level=float(fib100),
                                              direction="UP", label="Fib 100% REV → 0%")
        except Exception:
            pass

    badges_top = []

    # UPDATE (1) & (2): MACD Buy/Sell ONLY on HMA55 cross with MACD sign condition (trend-filtered by global slope m_d)
    macd_hma_sig_d = last_macd_hma_cross_star(
        price=df_show,
        hma=hma_d_full,
        macd=macd_d.reindex(df_show.index),
        trend_slope=m_d,
        lookback=160
    )
    if macd_hma_sig_d is not None:
        annotate_macd_star_callout(ax, macd_hma_sig_d["time"], macd_hma_sig_d["price"],
                                   side=macd_hma_sig_d["side"], hma_period=hma_period, y_frac=0.06)
        if macd_hma_sig_d["side"] == "BUY":
            badges_top.append((f"★ MACD Buy — HMA{hma_period} Cross @{fmt_price_val(macd_hma_sig_d['price'])}", "tab:green"))
        else:
            badges_top.append((f"★ MACD Sell — HMA{hma_period} Cross @{fmt_price_val(macd_hma_sig_d['price'])}", "tab:red"))

    star_d = last_reversal_star(df_show, trend_slope=m_d, lookback=20, confirm_bars=rev_bars_confirm)
    if star_d is not None:
        annotate_star(ax, star_d["time"], star_d["price"], star_d["kind"], show_text=False)
        if star_d.get("kind") == "trough":
            badges_top.append((f"★ Trough REV @{fmt_price_val(star_d['price'])}", "tab:green"))
        elif star_d.get("kind") == "peak":
            badges_top.append((f"★ Peak REV @{fmt_price_val(star_d['price'])}", "tab:red"))

    breakout_d = last_breakout_signal(
        price=df_show, resistance=res30_show, support=sup30_show,
        prox=sr_prox_pct, confirm_bars=rev_bars_confirm
    )
    if breakout_d is not None:
        if breakout_d["dir"] == "UP":
            annotate_breakout(ax, breakout_d["time"], breakout_d["price"], "UP")
            badges_top.append((f"▲ BREAKOUT @{fmt_price_val(breakout_d['price'])}", "tab:green"))
        else:
            annotate_breakout(ax, breakout_d["time"], breakout_d["price"], "DOWN")
            badges_top.append((f"▼ BREAKDOWN @{fmt_price_val(breakout_d['price'])}", "tab:red"))

    draw_top_badges(ax, badges_top)

    # instruction banner (daily local==global)
    try:
        px_val_d = float(df_show.iloc[-1])
        confirm_side = macd_hma_sig_d["side"] if macd_hma_sig_d is not None else None
        draw_instruction_ribbons(ax, m_d, sup_val_d, res_val_d, px_val_d, sel,
                                 confirm_side=confirm_side,
                                 global_slope=m_d,
                                 extra_note=None)
    except Exception:
        pass

    # Daily chart footer now includes Current Price (similar to intraday)
    px_now_d = float(df_show.iloc[-1]) if len(df_show) else np.nan
    nbb_txt_d = ""
    try:
        last_pct_d = float(bb_pctb_d_show.dropna().iloc[-1]) if show_bbands else np.nan
        last_nbb_d = float(bb_nbb_d_show.dropna().iloc[-1]) if show_bbands else np.nan
        if np.isfinite(last_nbb_d) and np.isfinite(last_pct_d):
            nbb_txt_d = f"  |  NBB {last_nbb_d:+.2f}  •  %B {fmt_pct(last_pct_d, digits=0)}"
    except Exception:
        pass

    footer_d = (
        (f"Current price: {fmt_price_val(px_now_d)}{nbb_txt_d}\n" if np.isfinite(px_now_d) else "")
        + f"R² ({slope_lb_daily} bars): {fmt_r2(r2_d)}  •  Slope: {fmt_slope(m_d)}/bar"
    )
    ax.text(
        0.99, 0.02,
        footer_d,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold", color="black",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
    )

    # MACD panel: NO signal line; show triangle ONLY for the MACD-HMA cross signal (if present)
    if not macd_d.dropna().empty:
        axm.plot(df_show.index, macd_d.reindex(df_show.index).values, linewidth=1.1, label="MACD")
        try:
            axm.bar(df_show.index, macd_hist_d.reindex(df_show.index).values, width=1.0, alpha=0.25, label="Hist")
        except Exception:
            pass
        axm.axhline(0, linewidth=0.9, alpha=0.6)

        if macd_hma_sig_d is not None:
            try:
                mac_at = float(macd_d.reindex(df_show.index).loc[macd_hma_sig_d["time"]])
            except Exception:
                mac_at = np.nan
            if np.isfinite(mac_at):
                if macd_hma_sig_d["side"] == "BUY":
                    annotate_buy_triangle(axm, macd_hma_sig_d["time"], mac_at, size=120)
                else:
                    annotate_sell_triangle(axm, macd_hma_sig_d["time"], mac_at, size=120)

    _simplify_axes(ax)
    _simplify_axes(axm)
    ax.set_ylabel("Price")
    axm.set_ylabel("MACD")
    axm.set_xlabel("Date (PST)")
    ax.legend(loc="lower left", framealpha=0.4)
    axm.legend(loc="lower left", framealpha=0.4)
    pad_right_xaxis(ax, frac=0.06)

    st.pyplot(fig)

def render_intraday_price_macd(sel: str, intraday: pd.DataFrame, p_up: float, p_dn: float):
    """
    Intraday: price plotted on compressed numeric axis, MACD panel beneath.
    Includes: local dashed slope trend_h (LOCAL), regression fit m_h (GLOBAL),
              S/R, HMA, BB, Kijun, PSAR, SuperTrend, fibs, session lines,
              reversal star, breakout, instruction banner only when local & global slopes aligned,
              MACD HMA-cross star callout (per requested rules).
    """
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    hc = intraday["Close"].astype(float).ffill()
    idx_mt = hc.index
    x_mt = np.arange(len(idx_mt), dtype=float)

    # LOCAL dashed slope based on raw intraday close
    xh = np.arange(len(hc), dtype=float)
    if len(hc.dropna()) >= 2:
        try:
            coef = np.polyfit(xh, hc.values.astype(float), 1)
            slope_h = float(coef[0]); intercept_h = float(coef[1])
            trend_h = slope_h * xh + intercept_h
        except Exception:
            slope_h = 0.0
            intercept_h = float(hc.iloc[-1]) if len(hc) else 0.0
            trend_h = np.full_like(xh, intercept_h, dtype=float)
    else:
        slope_h = 0.0
        intercept_h = float(hc.iloc[-1]) if len(hc) else 0.0
        trend_h = np.full_like(xh, intercept_h, dtype=float)

    he = hc.ewm(span=20).mean()
    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {"High", "Low", "Close"}.issubset(intraday.columns) and show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(intraday["High"], intraday["Low"], intraday["Close"],
                                             conv=ichi_conv, base=ichi_base, span_b=ichi_spanb, shift_cloud=False)
        kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    hma_h = compute_hma(hc, period=hma_period)

    macd_h, _sig_unused, macd_hist_h = compute_macd(hc)

    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    psar_h_aligned = _align_series_to_index(psar_h_df["PSAR"], hc.index) if (show_psar and not psar_h_df.empty and "PSAR" in psar_h_df) else pd.Series(index=hc.index, dtype=float)

    # GLOBAL regression fit m_h (used by instructions + ±2σ band)
    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)

    fig2, (ax2, ax2m) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]}
    )
    plt.subplots_adjust(top=0.88, right=0.995, left=0.06, hspace=0.05)

    trend_color = "tab:green" if slope_h >= 0 else "tab:red"
    ax2.set_title(f"{sel} Intraday (5m)  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")

    ax2.plot(x_mt, hc.values, label="Price", linewidth=1.2)
    ax2.plot(x_mt, he.reindex(idx_mt).values, "--", alpha=0.45, linewidth=0.9, label="_nolegend_")
    ax2.plot(x_mt, trend_h, "--", label="Trend (LOCAL)", linewidth=2.4, color=trend_color, alpha=0.95)

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(x_mt, hma_h.reindex(idx_mt).values, "-", linewidth=1.3, alpha=0.9, label="HMA")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(x_mt, kijun_h.reindex(idx_mt).values, "-", linewidth=1.1, color="black", alpha=0.55, label="Kijun")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(x_mt, bb_lo_h.reindex(idx_mt).values, bb_up_h.reindex(idx_mt).values, alpha=0.04, label="_nolegend_")
        ax2.plot(x_mt, bb_mid_h.reindex(idx_mt).values, "-", linewidth=0.8, alpha=0.3, label="_nolegend_")

    if st_intraday is not None and not st_intraday.empty:
        plot_supertrend_line(ax2, x_mt, st_intraday, idx_mt, use_positions=True, lw=1.8, alpha=0.75)

    if show_psar and not psar_h_aligned.dropna().empty:
        ax2.plot(x_mt, psar_h_aligned.reindex(idx_mt).values, "-", linewidth=1.8, color="purple", alpha=0.95, label="PSAR", zorder=6)

    # S/R levels
    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1]); sup_val = float(sup_h.iloc[-1]); px_val = float(hc.iloc[-1])
    except Exception:
        pass
    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=0, xmax=len(x_mt) - 1, colors="tab:red", linestyles="-", linewidth=2.0, alpha=0.95, label="_nolegend_")
        ax2.hlines(sup_val, xmin=0, xmax=len(x_mt) - 1, colors="tab:green", linestyles="-", linewidth=2.0, alpha=0.95, label="_nolegend_")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    def _pos(ts):
        try:
            ix = idx_mt.get_indexer([ts], method="nearest")[0]
            return float(ix) if ix != -1 else np.nan
        except Exception:
            return np.nan

    badges_top_h = []

    # UPDATE (1) & (2): MACD Buy/Sell ONLY on HMA55 cross with MACD sign condition (trend-filtered by GLOBAL m_h)
    macd_hma_sig_h = last_macd_hma_cross_star(
        price=hc,
        hma=hma_h,
        macd=macd_h.reindex(idx_mt),
        trend_slope=m_h,
        lookback=240
    )
    if macd_hma_sig_h is not None:
        x_cross = _pos(macd_hma_sig_h["time"])
        if np.isfinite(x_cross):
            annotate_macd_star_callout(ax2, x_cross, macd_hma_sig_h["price"],
                                       side=macd_hma_sig_h["side"], hma_period=hma_period, y_frac=0.06)
            if macd_hma_sig_h["side"] == "BUY":
                badges_top_h.append((f"★ MACD Buy — HMA{hma_period} Cross @{fmt_price_val(macd_hma_sig_h['price'])}", "tab:green"))
            else:
                badges_top_h.append((f"★ MACD Sell — HMA{hma_period} Cross @{fmt_price_val(macd_hma_sig_h['price'])}", "tab:red"))

    star_h = last_reversal_star(hc, trend_slope=m_h, lookback=20, confirm_bars=rev_bars_confirm)
    if star_h is not None:
        tpos_star = _pos(star_h["time"])
        if np.isfinite(tpos_star):
            annotate_star(ax2, tpos_star, star_h["price"], star_h["kind"], show_text=False)
            if star_h.get("kind") == "trough":
                badges_top_h.append((f"★ Trough REV @{fmt_price_val(star_h['price'])}", "tab:green"))
            elif star_h.get("kind") == "peak":
                badges_top_h.append((f"★ Peak REV @{fmt_price_val(star_h['price'])}", "tab:red"))

    breakout_h = last_breakout_signal(
        price=hc, resistance=res_h, support=sup_h,
        prox=sr_prox_pct, confirm_bars=rev_bars_confirm
    )
    if breakout_h is not None:
        tpos_bo = _pos(breakout_h["time"])
        if np.isfinite(tpos_bo):
            if breakout_h["dir"] == "UP":
                annotate_breakout(ax2, tpos_bo, breakout_h["price"], "UP")
                badges_top_h.append((f"▲ BREAKOUT @{fmt_price_val(breakout_h['price'])}", "tab:green"))
            else:
                annotate_breakout(ax2, tpos_bo, breakout_h["price"], "DOWN")
                badges_top_h.append((f"▼ BREAKDOWN @{fmt_price_val(breakout_h['price'])}", "tab:red"))

    draw_top_badges(ax2, badges_top_h)

    # instruction banner: LOCAL = slope_h, GLOBAL = m_h
    confirm_side_h = macd_hma_sig_h["side"] if macd_hma_sig_h is not None else None
    draw_instruction_ribbons(ax2, slope_h, sup_val, res_val, px_val, sel,
                             confirm_side=confirm_side_h,
                             global_slope=m_h,
                             extra_note=None)

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        footer_txt = (
            f"Current price: {fmt_price_val(px_val)}{nbb_txt}\n"
            f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}  •  Slope: {fmt_slope(m_h)}/bar"
        )
        ax2.text(
            0.5, 0.02, footer_txt,
            transform=ax2.transAxes, ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
        )

    # regression fit + ±2σ band (GLOBAL)
    if not yhat_h.empty:
        slope_col_h = "tab:green" if m_h >= 0 else "tab:red"
        ax2.plot(x_mt, yhat_h.reindex(idx_mt).values, "-", linewidth=2.6, color=slope_col_h, alpha=0.95, label="Slope Fit (GLOBAL)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(x_mt, upper_h.reindex(idx_mt).values, ":", linewidth=2.5, color="black", alpha=1.0, label="_nolegend_")
        ax2.plot(x_mt, lower_h.reindex(idx_mt).values, ":", linewidth=2.5, color="black", alpha=1.0, label="_nolegend_")

    # sessions (Forex) — Tie to the *symbol*
    if is_forex_symbol(sel) and show_sessions_pst and not hc.empty:
        sess_dt = compute_session_lines(idx_mt)
        sess_pos = map_session_lines_to_positions(sess_dt, idx_mt)
        ax2.plot([], [], linestyle="-", color="tab:blue", label="London Open (PST)")
        ax2.plot([], [], linestyle="--", color="tab:blue", label="London Close (PST)")
        ax2.plot([], [], linestyle="-", color="tab:orange", label="New York Open (PST)")
        ax2.plot([], [], linestyle="--", color="tab:orange", label="New York Close (PST)")
        for t in sess_pos.get("ldn_open", []):
            ax2.axvline(t, linestyle="-", linewidth=1.0, color="tab:blue", alpha=0.35)
        for t in sess_pos.get("ldn_close", []):
            ax2.axvline(t, linestyle="--", color="tab:blue", alpha=0.35)
        for t in sess_pos.get("ny_open", []):
            ax2.axvline(t, linestyle="-", color="tab:orange", linewidth=1.0, alpha=0.35)
        for t in sess_pos.get("ny_close", []):
            ax2.axvline(t, linestyle="--", color="tab:orange", linewidth=1.0, alpha=0.35)

    # fibs (intraday)
    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for _lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=0, xmax=len(x_mt) - 1, linestyles="dotted", linewidth=0.6, alpha=0.35)
        for lbl, y in fibs_h.items():
            ax2.text(len(x_mt) - 1, y, f" {lbl}", va="center", fontsize=8, alpha=0.6, fontweight="bold")
        try:
            fib0 = fibs_h.get("0%")
            fib100 = fibs_h.get("100%")
            if np.isfinite(fib0) and np.isfinite(fib100):
                fib_sig = last_fib_extreme_reversal(
                    price=hc,
                    slope=slope_h,
                    fib0_level=float(fib0),
                    fib100_level=float(fib100),
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback=max(20, int(sr_lb_hourly))
                )
                if fib_sig is not None:
                    if fib_sig["dir"] == "DOWN":
                        annotate_fib_reversal(ax2, ts=len(x_mt) - 1, y_level=float(fib0), direction="DOWN", label="Fib 0% REV → 100%")
                    elif fib_sig["dir"] == "UP":
                        annotate_fib_reversal(ax2, ts=len(x_mt) - 1, y_level=float(fib100), direction="UP", label="Fib 100% REV → 0%")
        except Exception:
            pass

    # MACD panel (intraday): triangle ONLY for MACD-HMA cross signal (if present)
    if not macd_h.dropna().empty:
        ax2m.plot(x_mt, macd_h.reindex(idx_mt).values, linewidth=1.1, label="MACD")
        try:
            ax2m.bar(x_mt, macd_hist_h.reindex(idx_mt).values, width=0.8, alpha=0.25, label="Hist")
        except Exception:
            pass
        ax2m.axhline(0, linewidth=0.9, alpha=0.6)

        if macd_hma_sig_h is not None:
            x_cross = _pos(macd_hma_sig_h["time"])
            if np.isfinite(x_cross):
                try:
                    mac_at = float(macd_h.reindex(idx_mt).loc[macd_hma_sig_h["time"]])
                except Exception:
                    mac_at = np.nan
                if np.isfinite(mac_at):
                    if macd_hma_sig_h["side"] == "BUY":
                        annotate_buy_triangle(ax2m, x_cross, mac_at, size=110)
                    else:
                        annotate_sell_triangle(ax2m, x_cross, mac_at, size=110)

    # axis formatting
    market_time_axis(ax2m, idx_mt)
    _simplify_axes(ax2)
    _simplify_axes(ax2m)
    ax2.set_xlabel("")
    ax2m.set_xlabel("Time (PST)")
    ax2.set_ylabel("Price")
    ax2m.set_ylabel("MACD")
    ax2.legend(loc="lower left", framealpha=0.4)
    ax2m.legend(loc="lower left", framealpha=0.4)
    pad_right_xaxis(ax2, frac=0.06)

    st.pyplot(fig2)

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

# --- Session init ---
if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"
if "hist_years" not in st.session_state:
    st.session_state.hist_years = 10

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics",
    "NTD -0.75 Scanner",
    "Long-Term History",
    "MACD Hot List",
    "Daily Support Reversals"
])
# =========================
# Part 5/6 — bullbear.py
# =========================

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")

    # ✅ FIXED: universe is defined above tabs now
    sel = st.selectbox("Ticker:", universe, key="tab1_ticker")

    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], index=(1 if mode == "Forex" else 0), key="orig_chart")
    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    auto_run = st.session_state.run_all

    # FIX (This request): keep the last RUN symbol sticky during auto-reruns until the user clicks Run Forecast again.
    btn_run = st.button("Run Forecast", key="btn_run_forecast")
    if btn_run or auto_run:
        run_ticker = sel if btn_run else st.session_state.get("ticker", sel)

        df_hist = fetch_hist(run_ticker)
        df_ohlc = fetch_hist_ohlc(run_ticker)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(run_ticker, period=period_map[hour_range])

        st.session_state.update({
            "df_hist": df_hist,
            "df_ohlc": df_ohlc,
            "fc_idx": fc_idx,
            "fc_vals": fc_vals,
            "fc_ci": fc_ci,
            "intraday": intraday,
            "ticker": run_ticker,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

    st.caption(
        "The Slope Line serves as an informational tool that signals potential trend changes and should be used for risk management rather than trading decisions. "
        "Trading based on the slope should only occur when it aligns with the trend line."
    )

    caution_below_btn = st.empty()

    # UPDATE (1): Once a symbol is RUN, keep charting that last-run symbol until another Run occurs.
    if st.session_state.run_all and st.session_state.get("ticker") is not None:
        active_sel = st.session_state.ticker  # last-run symbol (sticky)
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = (1 - p_up) if np.isfinite(p_up) else np.nan

        # Forex news should follow the last-run symbol type (not the sidebar Mode toggle)
        fx_news = fetch_yf_news(active_sel, window_days=news_window_days) if (is_forex_symbol(active_sel) and show_fx_news) else pd.DataFrame()

        # Daily
        if chart in ("Daily", "Both"):
            render_daily_price_macd(active_sel, df, df_ohlc)

        # Intraday
        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            if intraday is None or intraday.empty or "Close" not in intraday:
                st.warning("No intraday data available.")
            else:
                # show caution if LOCAL vs GLOBAL disagree (same logic as your snippet)
                try:
                    hc = intraday["Close"].astype(float).ffill()
                    xh = np.arange(len(hc), dtype=float)
                    coef = np.polyfit(xh, hc.values.astype(float), 1) if len(hc.dropna()) >= 2 else [0.0, float(hc.iloc[-1])]
                    slope_local = float(coef[0])

                    _, _, _, slope_global, _ = regression_with_band(hc, slope_lb_hourly)
                    if np.isfinite(slope_local) and np.isfinite(slope_global) and (slope_local * slope_global < 0):
                        caution_below_btn.warning(
                            "ALERT: Please exercise caution while trading at this moment, as the current slope indicates that the dash trendline may be reversing. "
                            "A reversal occurs near the 100% or 0% Fibonacci retracement levels. Once the reversal is confirmed, the trendline changes direction"
                        )
                except Exception:
                    pass

                render_intraday_price_macd(active_sel, intraday, p_up, p_dn)

        if is_forex_symbol(active_sel):
            st.subheader("Forex Session Overlaps (PST)")
            st.markdown(
                """
| Overlap | Time (PST) | Applies To |
|---|---|---|
| **New York & London** | **5:00 AM – 8:00 AM** | Any pair including **EUR**, **USD**, **GBP** |
| **Tokyo & New York** | **4:00 PM – 7:00 PM** | Any pair including **USD**, **JPY** |
| **London & Tokyo** | **12:00 AM – 1:00 AM** | Any pair including **EUR**, **GBP**, **JPY** |
"""
            )
            st.caption("These windows often see higher liquidity and volatility.")

        # Forecast table
        try:
            st.write(pd.DataFrame({
                "Forecast": st.session_state.fc_vals,
                "Lower": st.session_state.fc_ci.iloc[:, 0],
                "Upper": st.session_state.fc_ci.iloc[:, 1]
            }, index=st.session_state.fc_idx))
        except Exception:
            st.info("Forecast output not available.")

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci)
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = (1 - p_up) if np.isfinite(p_up) else np.nan

        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range', '24h')}**")
        view = st.radio("View:", ["Daily", "Intraday", "Both"], index=(1 if is_forex_symbol(st.session_state.ticker) else 0), key="enh_view")

        if view in ("Daily", "Both"):
            render_daily_price_macd(st.session_state.ticker, df, df_ohlc)

        if view in ("Intraday", "Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                st.info("Intraday view uses the same logic as Tab 1.")
                render_intraday_price_macd(st.session_state.ticker, intr, p_up, p_dn)

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = fetch_close_df_period(st.session_state.ticker, bb_period)
        if df3.empty or "Close" not in df3:
            st.warning("Not enough historical data to compute Bull vs Bear summary.")
        else:
            df3["PctChange"] = df3["Close"].pct_change()
            df3["Bull"] = df3["PctChange"] > 0
            bull = int(df3["Bull"].sum())
            bear = int((~df3["Bull"]).sum())
            total = bull + bear if (bull + bear) > 0 else 1
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Days", bull + bear)
            c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
            c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
            c4.metric("Lookback", bb_period)
# =========================
# Part 6/6 — bullbear.py
# =========================

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
        p_dn = (1 - p_up) if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        trend3m, up3m, lo3m, m3m, r2_3m = regression_with_band(df3m, lookback=len(df3m))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(res3m.index, res3m, ":", linewidth=2.0, color="tab:red", alpha=0.9, label="Resistance")
        ax.plot(sup3m.index, sup3m, ":", linewidth=2.0, color="tab:green", alpha=0.9, label="Support")
        if not trend3m.empty:
            col3 = "tab:green" if m3m >= 0 else "tab:red"
            ax.plot(trend3m.index, trend3m.values, "--", color=col3, linewidth=3.0,
                    label=f"Trend (m={fmt_slope(m3m)}/bar)")
        if not up3m.empty and not lo3m.empty:
            ax.plot(up3m.index, up3m.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend +2σ")
            ax.plot(lo3m.index, lo3m.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend -2σ")
        ax.set_xlabel("Date (PST)")
        ax.text(
            0.99, 0.02,
            f"R² (3M): {fmt_r2(r2_3m)}  •  Slope: {fmt_slope(m3m)}/bar",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
        )
        ax.legend()
        pad_right_xaxis(ax, frac=0.06)
        st.pyplot(fig)

        st.markdown("---")
        df0 = fetch_close_df_period(st.session_state.ticker, bb_period)
        if df0.empty or "Close" not in df0:
            st.warning("Not enough data to compute metrics for the selected lookback.")
        else:
            df0["PctChange"] = df0["Close"].pct_change()
            df0["Bull"] = df0["PctChange"] > 0
            df0["MA30"] = df0["Close"].rolling(30, min_periods=1).mean()

            st.subheader("Close + 30-day MA + Trend")
            res0 = df0["Close"].rolling(30, min_periods=1).max()
            sup0 = df0["Close"].rolling(30, min_periods=1).min()
            trend0, up0, lo0, m0, r2_0 = regression_with_band(df0["Close"], lookback=len(df0))

            fig0, ax0 = plt.subplots(figsize=(14, 5))
            ax0.plot(df0.index, df0["Close"], label="Close")
            ax0.plot(df0.index, df0["MA30"], label="30 MA")
            ax0.plot(res0.index, res0, ":", linewidth=2.0, color="tab:red", alpha=0.9, label="Resistance")
            ax0.plot(sup0.index, sup0, ":", linewidth=2.0, color="tab:green", alpha=0.9, label="Support")
            if not trend0.empty:
                col0 = "tab:green" if m0 >= 0 else "tab:red"
                ax0.plot(trend0.index, trend0.values, "--", color=col0, linewidth=3.0,
                         label=f"Trend (m={fmt_slope(m0)}/bar)")
            if not up0.empty and not lo0.empty:
                ax0.plot(up0.index, up0.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend +2σ")
                ax0.plot(lo0.index, lo0.values, ":", linewidth=3.0, color="black", alpha=1.0, label="Trend -2σ")
            ax0.set_xlabel("Date (PST)")
            ax0.text(
                0.99, 0.02,
                f"R² ({bb_period}): {fmt_r2(r2_0)}  •  Slope: {fmt_slope(m0)}/bar",
                transform=ax0.transAxes, ha="right", va="bottom",
                fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
            )
            ax0.legend()
            pad_right_xaxis(ax0, frac=0.06)
            st.pyplot(fig0)

            st.markdown("---")
            st.subheader("Daily % Change")
            st.line_chart(df0["PctChange"], use_container_width=True)

            st.subheader("Bull/Bear Distribution")
            dist = pd.DataFrame({"Type": ["Bull", "Bear"], "Days": [int(df0["Bull"].sum()), int((~df0["Bull"]).sum())]}).set_index("Type")
            st.bar_chart(dist, use_container_width=True)

# --- Tab 5: NTD -0.75 Scanner ---
with tab5:
    st.header("NTD -0.75 Scanner (NTD < -0.75)")
    st.caption("Scans the universe for symbols whose latest NTD value is below -0.75.")

    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    scan_hour_range = st.selectbox(
        "Hourly lookback for Forex:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
        key="ntd_scan_hour_range"
    )
    scan_period = period_map[scan_hour_range]
    thresh = -0.75
    run = st.button("Scan Universe", key="btn_ntd_scan")

    if run:
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
            daily_rows.append({"Symbol": sym, "NTD_Last": ntd_val, "BelowThresh": (np.isfinite(ntd_val) and ntd_val < thresh),
                               "Close": close_val, "Timestamp": ts})

        df_daily = pd.DataFrame(daily_rows)
        hits_daily = df_daily[df_daily["BelowThresh"] == True].copy().sort_values("NTD_Last")

        c3, c4 = st.columns(2)
        c3.metric("Universe Size", len(universe))
        c4.metric("Daily NTD < -0.75", int(hits_daily.shape[0]))

        st.subheader("Daily — latest NTD < -0.75")
        if hits_daily.empty:
            st.info("No symbols where the latest daily NTD value is below -0.75.")
        else:
            view = hits_daily.copy()
            view["NTD_Last"] = view["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
            view["Close"] = view["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(view[["Symbol", "Timestamp", "Close", "NTD_Last"]].reset_index(drop=True), use_container_width=True)

        if mode == "Forex":
            st.markdown("---")
            st.subheader(f"Forex Hourly — latest NTD < -0.75 ({scan_hour_range} lookback)")
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
                st.info("No Forex pairs where the latest hourly NTD value is below -0.75.")
            else:
                showh = hits_hour.copy()
                showh["NTD_Last"] = showh["NTD_Last"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
                showh["Close"] = showh["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
                st.dataframe(showh[["Symbol", "Timestamp", "Close", "NTD_Last"]].reset_index(drop=True), use_container_width=True)

# --- Tab 6: Long-Term History ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")
    default_idx = 0
    if st.session_state.get("ticker") in universe:
        default_idx = universe.index(st.session_state["ticker"])
    sym = st.selectbox("Ticker:", universe, index=default_idx, key="hist_long_ticker")

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
    st.caption(f"Showing last **{years} years**. Support/Resistance = rolling 252-day extremes; trendline fits the shown window.")

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

            fig, ax = plt.subplots(figsize=(14, 5))
            plt.subplots_adjust(right=0.995, left=0.06, top=0.92)
            ax.set_title(f"{sym} — Last {years} Years — Price + 252d S/R + Trend")
            ax.plot(s.index, s.values, label="Close", linewidth=1.4)

            if np.isfinite(res_last) and np.isfinite(sup_last):
                ax.hlines(res_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:red", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                ax.hlines(sup_last, xmin=s.index[0], xmax=s.index[-1], colors="tab:green", linestyles="-", linewidth=2.2, alpha=0.95, label="_nolegend_")
                label_on_left(ax, res_last, f"R {fmt_price_val(res_last)}", color="tab:red")
                label_on_left(ax, sup_last, f"S {fmt_price_val(sup_last)}", color="tab:green")

            if not yhat_all.empty:
                col_all = "tab:green" if m_all >= 0 else "tab:red"
                ax.plot(yhat_all.index, yhat_all.values, "--", linewidth=3.2, color=col_all, label="Trend")
            if not upper_all.empty and not lower_all.empty:
                ax.plot(upper_all.index, upper_all.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")
                ax.plot(lower_all.index, lower_all.values, ":", linewidth=3.0, color="black", alpha=1.0, label="_nolegend_")

            px_now = _safe_last_float(s)
            price_line = f"Current price: {fmt_price_val(px_now)}" if np.isfinite(px_now) else ""
            footer = (price_line + ("\n" if price_line else "") + f"R² (trend): {fmt_r2(r2_all)}  •  Slope: {fmt_slope(m_all)}/bar")
            ax.text(
                0.99, 0.02, footer,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7)
            )

            _simplify_axes(ax)
            ax.set_xlabel("Date (PST)")
            ax.set_ylabel("Price")
            ax.legend(loc="lower left", framealpha=0.4)
            pad_right_xaxis(ax, frac=0.06)
            st.pyplot(fig)

# --- Tab 7: MACD Hot List ---
with tab7:
    st.header("MACD Hot List")
    st.caption(
        "Shows symbols where the **MACD line** crossed the **0.0** line in the direction of the "
        "**Global Daily Trendline** (regression slope).  \n"
        "• Global Uptrend → MACD 0-cross **UP**  \n"
        "• Global Downtrend → MACD 0-cross **DOWN**"
    )

    # MACD-only hot list (trend-filtered)
    recent_bars_macd = int(st.slider(
        "Max bars since MACD 0-cross (daily bars)",
        1, 252, 30, 1,
        key="macd_hot_recent_bars"
    ))
    run_hot = st.button("Scan Universe for MACD 0-Cross (Trend-filtered)", key="btn_scan_macd_hot")

    if run_hot:
        up_rows = []
        dn_rows = []

        for sym in universe:
            try:
                s = fetch_hist(sym)
                s = _coerce_1d_series(s).dropna()
                if s is None or s.empty or s.shape[0] < max(30, slope_lb_daily):
                    continue

                # Global daily trendline slope
                _yhat, _u, _l, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                if not np.isfinite(m_sym) or float(m_sym) == 0.0:
                    continue

                macd_s, _sig_unused, _hist_unused = compute_macd(s)

                # Align and clean
                p = _coerce_1d_series(s).astype(float).dropna()
                m = _coerce_1d_series(macd_s).astype(float).reindex(p.index)
                mask = p.notna() & m.notna()
                p, m = p[mask], m[mask]
                if len(m) < 3:
                    continue

                direction = "UP" if float(m_sym) > 0 else "DOWN"

                # Find most recent MACD 0-cross in the direction of the global trend
                ix_macd = None
                for i in range(len(m) - 1, 0, -1):
                    prev = float(m.iloc[i - 1]); cur = float(m.iloc[i])
                    if direction == "UP":
                        if (prev <= 0.0) and (cur > 0.0):
                            ix_macd = i
                            break
                    else:
                        if (prev >= 0.0) and (cur < 0.0):
                            ix_macd = i
                            break

                if ix_macd is None:
                    continue

                bars_since_macd = (len(m) - 1) - int(ix_macd)
                if bars_since_macd > recent_bars_macd:
                    continue

                last_close = float(p.iloc[-1])
                macd_at_cross = float(m.iloc[ix_macd])
                macd_last = float(m.iloc[-1])

                row = {
                    "Symbol": sym,
                    "Trend": ("Up" if direction == "UP" else "Down"),
                    "Slope": float(m_sym),
                    "R2": float(r2_sym),
                    "Close": last_close,
                    "MACD_Cross_Time": p.index[ix_macd],
                    "MACD_At_Cross": macd_at_cross,
                    "MACD_Bars_Since": int(bars_since_macd),
                    "MACD_Last": macd_last,
                }

                if direction == "UP":
                    up_rows.append(row)
                else:
                    dn_rows.append(row)

            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        c1.metric("Universe Size", len(universe))
        c2.metric("Global Uptrend MACD 0↑ Hits", int(len(up_rows)))
        c3.metric("Global Downtrend MACD 0↓ Hits", int(len(dn_rows)))

        st.markdown("---")

        st.subheader("Global Upward Trendline — MACD 0-Cross Up")
        if not up_rows:
            st.info("No symbols matched the Uptrend MACD 0-cross criteria in the selected recency window.")
        else:
            out_up = pd.DataFrame(up_rows).sort_values(
                ["MACD_Bars_Since", "Symbol"],
                ascending=[True, True]
            )
            view_up = out_up.copy()
            view_up["Close"] = view_up["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_up["Slope"] = view_up["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view_up["R2"] = view_up["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            view_up["MACD_At_Cross"] = view_up["MACD_At_Cross"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            view_up["MACD_Last"] = view_up["MACD_Last"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            st.dataframe(
                view_up[[
                    "Symbol", "MACD_Cross_Time", "MACD_At_Cross", "MACD_Bars_Since",
                    "MACD_Last", "Close", "Slope", "R2"
                ]].reset_index(drop=True),
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Global Downward Trendline — MACD 0-Cross Down")
        if not dn_rows:
            st.info("No symbols matched the Downtrend MACD 0-cross criteria in the selected recency window.")
        else:
            out_dn = pd.DataFrame(dn_rows).sort_values(
                ["MACD_Bars_Since", "Symbol"],
                ascending=[True, True]
            )
            view_dn = out_dn.copy()
            view_dn["Close"] = view_dn["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_dn["Slope"] = view_dn["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view_dn["R2"] = view_dn["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            view_dn["MACD_At_Cross"] = view_dn["MACD_At_Cross"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            view_dn["MACD_Last"] = view_dn["MACD_Last"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            st.dataframe(
                view_dn[[
                    "Symbol", "MACD_Cross_Time", "MACD_At_Cross", "MACD_Bars_Since",
                    "MACD_Last", "Close", "Slope", "R2"
                ]].reset_index(drop=True),
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("Optional — MACD 0-Cross + S/R Reversal (AND)")
    st.caption(
        "This combined scan requires **BOTH** conditions (AND):  \n"
        "• MACD crosses 0.0 in the direction of the global trend  \n"
        "• Price has successfully reversed from **Support** (uptrend) or **Resistance** (downtrend) before the cross  \n"
        "No HMA55 alignment is used."
    )

    recent_bars = int(st.slider("How recent is 'recent' (daily bars)?", 1, 252, 7, 1, key="macd_sr_recent_bars"))
    run_combo = st.button("Scan Universe for MACD 0-Cross + S/R Reversal (AND)", key="btn_scan_macd_sr_hot")

    if run_combo:
        up_rows = []
        dn_rows = []

        for sym in universe:
            try:
                s = fetch_hist(sym)
                s = _coerce_1d_series(s).dropna()
                if s is None or s.empty or s.shape[0] < max(30, slope_lb_daily):
                    continue

                _yhat, _u, _l, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                if not np.isfinite(m_sym) or float(m_sym) == 0.0:
                    continue

                macd_s, _sig_unused, _hist_unused = compute_macd(s)
                sup30 = s.rolling(30, min_periods=1).min()
                res30 = s.rolling(30, min_periods=1).max()

                sig = last_macd_zero_cross_with_sr_reversal(
                    price=s,
                    support=sup30,
                    resistance=res30,
                    macd=macd_s,
                    trend_slope=m_sym,
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback_macd=max(160, recent_bars + 10),
                    sr_lookback=60
                )
                if sig is None:
                    continue

                if int(sig.get("bars_since", 999999)) > int(recent_bars):
                    continue

                last_close = float(s.iloc[-1])
                row = {
                    "Symbol": sym,
                    "Trend": ("Up" if float(m_sym) > 0 else "Down"),
                    "Slope": float(m_sym),
                    "R2": float(r2_sym),
                    "Close": last_close,
                    "MACD_Cross_Time": sig["time"],
                    "MACD_At_Cross": float(sig["macd_value"]),
                    "MACD_Bars_Since": int(sig["bars_since"]),
                    "SR_Touch_Time": sig.get("t_touch"),
                    "SR_Level_At_Touch": float(sig.get("sr_level")) if sig.get("sr_level") is not None else np.nan,
                }

                if row["Trend"] == "Up":
                    up_rows.append(row)
                else:
                    dn_rows.append(row)

            except Exception:
                pass

        c1, c2, c3 = st.columns(3)
        c1.metric("Universe Size", len(universe))
        c2.metric("Global Uptrend (MACD+S/R) Hits", int(len(up_rows)))
        c3.metric("Global Downtrend (MACD+S/R) Hits", int(len(dn_rows)))

        st.markdown("---")

        st.subheader("Global Uptrend — MACD 0-Cross Up + Support Reversal Confirmed")
        if not up_rows:
            st.info("No symbols matched the Uptrend MACD+S/R criteria in the selected recency window.")
        else:
            out_up = pd.DataFrame(up_rows).sort_values(
                ["MACD_Bars_Since", "Symbol"],
                ascending=[True, True]
            )
            view_up = out_up.copy()
            view_up["Close"] = view_up["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_up["Slope"] = view_up["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view_up["R2"] = view_up["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            view_up["MACD_At_Cross"] = view_up["MACD_At_Cross"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            view_up["SR_Level_At_Touch"] = view_up["SR_Level_At_Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(
                view_up[[
                    "Symbol", "MACD_Cross_Time", "MACD_At_Cross", "MACD_Bars_Since",
                    "SR_Touch_Time", "SR_Level_At_Touch",
                    "Close", "Slope", "R2"
                ]].reset_index(drop=True),
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Global Downtrend — MACD 0-Cross Down + Resistance Reversal Confirmed")
        if not dn_rows:
            st.info("No symbols matched the Downtrend MACD+S/R criteria in the selected recency window.")
        else:
            out_dn = pd.DataFrame(dn_rows).sort_values(
                ["MACD_Bars_Since", "Symbol"],
                ascending=[True, True]
            )
            view_dn = out_dn.copy()
            view_dn["Close"] = view_dn["Close"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view_dn["Slope"] = view_dn["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view_dn["R2"] = view_dn["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")
            view_dn["MACD_At_Cross"] = view_dn["MACD_At_Cross"].map(lambda v: f"{v:+.4f}" if np.isfinite(v) else "n/a")
            view_dn["SR_Level_At_Touch"] = view_dn["SR_Level_At_Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            st.dataframe(
                view_dn[[
                    "Symbol", "MACD_Cross_Time", "MACD_At_Cross", "MACD_Bars_Since",
                    "SR_Touch_Time", "SR_Level_At_Touch",
                    "Close", "Slope", "R2"
                ]].reset_index(drop=True),
                use_container_width=True
            )

# --- Tab 8: Daily Support Reversals ---
with tab8:
    st.header("Daily Support Reversals")
    st.caption(
        "Scans for symbols that touched daily Support (rolling 30-bar Close min) within proximity and then printed consecutive higher closes."
    )

    if st.button("Scan Universe for Daily Support Reversals", key="btn_scan_support_rev"):
        rows = []
        for sym in universe:
            try:
                s = fetch_hist(sym)
                if s is None or s.dropna().shape[0] < max(10, slope_lb_daily):
                    continue
                sup30 = s.rolling(30, min_periods=1).min()
                sig = find_support_touch_confirmed_up(
                    price=s,
                    support=sup30,
                    prox=sr_prox_pct,
                    confirm_bars=rev_bars_confirm,
                    lookback_bars=30
                )
                if sig is None:
                    continue
                _, _, _, m_sym, r2_sym = regression_with_band(s, lookback=slope_lb_daily, z=Z_FOR_99)
                rows.append({
                    "Symbol": sym,
                    "Touched": sig["t_touch"],
                    "Close@Touch": sig["touch_close"],
                    "Support@Touch": sig["support"],
                    "Now": s.dropna().index[-1],
                    "NowClose": sig["now_close"],
                    "RisePct": sig["gain_pct"],
                    "BarsSince": sig["bars_since_touch"],
                    "Slope": m_sym,
                    "R2": r2_sym
                })
            except Exception:
                pass

        if not rows:
            st.info("No symbols met the support-touch → confirmed up criteria at this time.")
        else:
            df = pd.DataFrame(rows).sort_values(["RisePct", "BarsSince"], ascending=[False, True])
            view = df.copy()
            view["Close@Touch"] = view["Close@Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["Support@Touch"] = view["Support@Touch"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["NowClose"] = view["NowClose"].map(lambda v: fmt_price_val(v) if np.isfinite(v) else "n/a")
            view["RisePct"] = view["RisePct"].map(lambda v: fmt_pct(v, 2) if np.isfinite(v) else "n/a")
            view["Slope"] = view["Slope"].map(lambda v: f"{v:+.5f}" if np.isfinite(v) else "n/a")
            view["R2"] = view["R2"].map(lambda v: fmt_pct(v, 1) if np.isfinite(v) else "n/a")

            st.dataframe(
                view[["Symbol", "Touched", "Close@Touch", "Support@Touch", "Now", "NowClose", "RisePct", "BarsSince", "Slope", "R2"]].reset_index(drop=True),
                use_container_width=True
            )
