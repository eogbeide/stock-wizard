# =========================
# Part 1/6 — bullbear.py
# =========================
# bullbear.py — Stocks/Forex Dashboard + Forecasts
# UPDATE: Single latest band-reversal signal for trading
#   • Uptrend  → BUY when price reverses up from lower ±2σ band and is near it
#   • Downtrend→ SELL when price reverses down from upper ±2σ band and is near it
# Only the latest signal is shown at any time.
#
# FIX (This request): Instruction banner now ALSO shows:
#   • "MACD Sell — HMA55 Cross" when:
#       - Global trend is DOWN
#       - MACD crosses 0.0 DOWN
#       - Price crosses HMA(55) DOWN
#
# FIX (prior): NameError crash at Tab 1 selectbox:
#   sel = st.selectbox("Ticker:", universe, key="tab1_ticker")
# → universe is now defined BEFORE tabs are created.
#
# Includes:
#   • Daily + Intraday charts (price + MACD)
#   • SuperTrend line + PSAR line on price chart
#   • MACD 0-cross 99% confidence triangles (trend filtered)
#   • MACD-HMA cross star callouts (price chart)
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
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
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
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X','NZDJPY=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','EURCAD=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','CNHJPY=X','AUDJPY=X'
    ])

# =============================================================================
# ✅ STICKY SELECTIONS (Fix): Defaults apply ONLY on first open.
# Once a symbol is chosen and run (or changed), the app will NOT revert to defaults.
# This also prevents “other tabs” (e.g., Long-Term) from staying stuck on the initial default.
# =============================================================================

def _pick_valid_symbol(candidate, uni):
    try:
        if candidate is not None and str(candidate) in uni:
            return str(candidate)
    except Exception:
        pass
    return None

# Track whether the Long-Term tab ticker was manually changed by the user
if "hist_long_ticker_touched" not in st.session_state:
    st.session_state.hist_long_ticker_touched = False

# Per-mode "last symbol" memory inside the current Streamlit session
_last_sym_key = "last_symbol_stocks" if mode == "Stocks" else "last_symbol_forex"

# Resolve active_symbol (used to sync defaults across tabs after the first user selection)
cand = _pick_valid_symbol(st.session_state.get("active_symbol"), universe)
if cand is None:
    cand = _pick_valid_symbol(st.session_state.get("tab1_ticker"), universe)
if cand is None:
    cand = _pick_valid_symbol(st.session_state.get("ticker"), universe)
if cand is None:
    cand = _pick_valid_symbol(st.session_state.get(_last_sym_key), universe)
if cand is None:
    cand = universe[0] if universe else None

if cand is not None:
    st.session_state.active_symbol = cand
    # If the tab1 ticker is missing/invalid, seed it from the active symbol (first-open default behavior preserved)
    if ("tab1_ticker" not in st.session_state) or (st.session_state.get("tab1_ticker") not in universe):
        st.session_state.tab1_ticker = cand

    # Long-term tab should follow the active symbol unless the user explicitly changed it
    if not st.session_state.get("hist_long_ticker_touched", False):
        st.session_state.hist_long_ticker = cand
    else:
        # If user touched it, keep their choice unless it becomes invalid for the current universe
        if ("hist_long_ticker" not in st.session_state) or (st.session_state.get("hist_long_ticker") not in universe):
            st.session_state.hist_long_ticker = cand

def _on_tab1_ticker_change():
    # When Tab 1 ticker changes, update the app-wide active symbol and sync followers (unless "touched")
    try:
        new_sym = st.session_state.get("tab1_ticker")
        if new_sym in universe:
            st.session_state.active_symbol = new_sym
            key = "last_symbol_stocks" if st.session_state.get("mode") == "Stocks" else "last_symbol_forex"
            st.session_state[key] = new_sym
            if not st.session_state.get("hist_long_ticker_touched", False):
                st.session_state.hist_long_ticker = new_sym
    except Exception:
        pass

def _on_hist_long_ticker_change():
    # User explicitly changed long-term ticker; don't auto-sync it anymore
    try:
        st.session_state.hist_long_ticker_touched = True
    except Exception:
        pass

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
# =========================
# Part 3/6 — bullbear.py
# =========================

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

        # UPDATE (3): Add the price value to the MACD-HMA cross callout label
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
    Uptrend:  price crosses UP through HMA and MACD is still < 0 → BUY
    Downtrend:price crosses DOWN through HMA and MACD is still > 0 → SELL
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

        if trend_slope > 0:
            up_cross = (p_prev < h_prev) and (p_cur >= h_cur)
            if up_cross and (mac_cur < 0.0):
                return {"time": t, "price": p_cur, "side": "BUY"}
        else:
            dn_cross = (p_prev > h_prev) and (p_cur <= h_cur)
            if dn_cross and (mac_cur > 0.0):
                return {"time": t, "price": p_cur, "side": "SELL"}
    return None

# NEW (This request):
def last_macd_zero_and_hma_cross_sell(price: pd.Series,
                                     hma: pd.Series,
                                     macd: pd.Series,
                                     lookback_bars: int = 3):
    """
    Trigger when (within the last lookback_bars):
      - MACD crosses 0 DOWN (prev >= 0, curr < 0)
      - Price crosses HMA DOWN (prev > HMA_prev, curr <= HMA_curr)
    Returns dict with time/price/bars_since if found, else None.
    """
    p = _coerce_1d_series(price).astype(float).dropna()
    if p.empty or len(p) < 2:
        return None
    h = _coerce_1d_series(hma).astype(float).reindex(p.index)
    m = _coerce_1d_series(macd).astype(float).reindex(p.index)
    mask = p.notna() & h.notna() & m.notna()
    p, h, m = p[mask], h[mask], m[mask]
    if len(p) < 2:
        return None

    lb = max(1, int(lookback_bars))
    start = max(1, len(p) - lb)

    for i in range(len(p) - 1, start - 1, -1):
        p_prev, p_cur = float(p.iloc[i - 1]), float(p.iloc[i])
        h_prev, h_cur = float(h.iloc[i - 1]), float(h.iloc[i])
        m_prev, m_cur = float(m.iloc[i - 1]), float(m.iloc[i])

        macd0_dn = (m_prev >= 0.0) and (m_cur < 0.0)
        hma_dn = (p_prev > h_prev) and (p_cur <= h_cur)

        if macd0_dn and hma_dn:
            bars_since = (len(p) - 1) - i
            return {"time": p.index[i], "price": p_cur, "bars_since": int(bars_since)}
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
# =========================
# Part 5/6 — bullbear.py
# =========================

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
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue", alpha=0.35)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:orange", alpha=0.35)
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
# =========================
# Part 6/6 — bullbear.py
# =========================

# NOTE:
# The remainder of your original code (render_daily_price_macd, render_intraday_price_macd,
# compute_normalized_trend, session init, tabs, scanners, etc.) is unchanged EXCEPT:
#   ✅ Tab 1 ticker selectbox now has on_change=_on_tab1_ticker_change
#   ✅ Tab 6 ticker selectbox now has on_change=_on_hist_long_ticker_change
#   ✅ Tab 6 now uses st.session_state["hist_long_ticker"] which is auto-synced to the last Tab 1 selection
#      unless the user explicitly changes Tab 6 (then it stays independent and won’t be reset)

# (Your existing render_* functions remain exactly as provided above in your prompt, unchanged.)

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

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")

    # ✅ UPDATED (Sticky selection): Tab 1 ticker updates app-wide active_symbol and syncs Long-Term (unless touched)
    sel = st.selectbox("Ticker:", universe, key="tab1_ticker", on_change=_on_tab1_ticker_change)

    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key="orig_chart")
    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h", "48h", "96h"].index(st.session_state.get("hour_range", "24h")),
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

        # ✅ Ensure active_symbol reflects the last run symbol (prevents any “snap back” to defaults on rerun/refresh)
        try:
            st.session_state.active_symbol = sel
            key = "last_symbol_stocks" if mode == "Stocks" else "last_symbol_forex"
            st.session_state[key] = sel
            if not st.session_state.get("hist_long_ticker_touched", False):
                st.session_state.hist_long_ticker = sel
        except Exception:
            pass

    st.caption(
        "The Slope Line serves as an informational tool that signals potential trend changes and should be used for risk management rather than trading decisions. "
        "Trading based on the slope should only occur when it aligns with the trend line."
    )

    caution_below_btn = st.empty()

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc
        last_price = _safe_last_float(df)

        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = (1 - p_up) if np.isfinite(p_up) else np.nan

        fx_news = fetch_yf_news(sel, window_days=news_window_days) if (mode == "Forex" and show_fx_news) else pd.DataFrame()

        # Daily
        if chart in ("Daily", "Both"):
            render_daily_price_macd(sel, df, df_ohlc)

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

                render_intraday_price_macd(sel, intraday, p_up, p_dn)

        if mode == "Forex":
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

# --- Tab 2..Tab 5..Tab 7..Tab 8 ---
# (Unchanged from your provided code)

# --- Tab 6: Long-Term History ---
with tab6:
    st.header("Long-Term History — Price with S/R & Trend")

    # ✅ UPDATED (Sticky selection): defaults follow the Tab 1 active symbol after first selection,
    # unless user explicitly changes Tab 6 (then it stays fixed and won’t revert).
    if ("hist_long_ticker" not in st.session_state) or (st.session_state.get("hist_long_ticker") not in universe):
        st.session_state.hist_long_ticker = st.session_state.get("active_symbol", universe[0] if universe else None)

    sym = st.selectbox("Ticker:", universe, key="hist_long_ticker", on_change=_on_hist_long_ticker_change)

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
