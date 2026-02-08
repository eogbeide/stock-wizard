# /mount/src/stock-wizard/bullbear.py
# =========================
# bullbear.py — COMPLETE FILE (3 BATCHES)
# FIX: NameError for fibonacci_levels() by defining it BEFORE render_daily_price_chart()
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import pytz
import time

from matplotlib.dates import DateFormatter, AutoDateLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="bullbear.py — Stocks/Forex Dashboard + Forecasts",
    page_icon="📈",
    layout="wide",
)

TZ = pytz.timezone("America/Los_Angeles")


# ---------------------------
# Matplotlib theme (STYLE ONLY)
# ---------------------------
def _apply_mpl_theme():
    try:
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "axes.grid": True,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        })
    except Exception:
        pass


_apply_mpl_theme()


# ---------------------------
# Minimal "ribbon-ish" tab styling
# ---------------------------
st.markdown(
    """
    <style>
      /* Keep it clean and readable */
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      /* Slightly tighten widgets */
      div[data-baseweb="select"] > div { border-radius: 10px; }
      /* Make tabs feel more "pill/ribbon" */
      button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 8px 14px !important;
        margin-right: 6px !important;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        font-weight: 700 !important;
        border: 1px solid rgba(0,0,0,0.2) !important;
      }
      /* Tables */
      .stDataFrame { border-radius: 14px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# Universes (edit freely)
# ---------------------------
DEFAULT_FOREX = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X"
]

DEFAULT_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "JPM", "XOM", "UNH", "AVGO", "COST"
]


# ---------------------------
# Data helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def _download_ohlc(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV via yfinance.
    Returns a normalized DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # yfinance sometimes returns multiindex columns for multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        # take the first (only) ticker
        df.columns = df.columns.get_level_values(-1)

    cols = ["Open", "High", "Low", "Close"]
    for c in cols:
        if c not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])
    return df


def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (np.ndarray, list, tuple)):
        return pd.Series(x)
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------
# Indicators
# ---------------------------
def wma(series: pd.Series, period: int) -> pd.Series:
    s = _coerce_1d_series(series).astype(float)
    if period <= 1 or len(s) < period:
        return pd.Series(index=s.index, dtype=float)
    weights = np.arange(1, period + 1, dtype=float)

    def _calc(x):
        return np.dot(x, weights) / weights.sum()

    return s.rolling(period).apply(_calc, raw=True)


def hma(series: pd.Series, period: int = 55) -> pd.Series:
    """
    Hull Moving Average:
      HMA(n) = WMA( 2*WMA(price, n/2) - WMA(price, n), sqrt(n) )
    """
    s = _coerce_1d_series(series).astype(float)
    if period <= 1 or len(s) < period:
        return pd.Series(index=s.index, dtype=float)

    half = max(int(period / 2), 1)
    sqrt_n = max(int(np.sqrt(period)), 1)
    w1 = wma(s, half)
    w2 = wma(s, period)
    raw = 2 * w1 - w2
    return wma(raw, sqrt_n)


def rolling_regression(y: pd.Series, window: int = 120):
    """
    Returns:
      reg_line (Series), slope (float of last window), r2 (float of last window),
      resid_std (float of last window)
    """
    y = _coerce_1d_series(y).astype(float).dropna()
    if len(y) < max(window, 20):
        idx = y.index
        return pd.Series(index=idx, dtype=float), np.nan, np.nan, np.nan

    # full-length regression line by rolling window (fast enough for typical use)
    reg = pd.Series(index=y.index, dtype=float)

    last_slope = np.nan
    last_r2 = np.nan
    last_std = np.nan

    for i in range(window - 1, len(y)):
        yy = y.iloc[i - window + 1:i + 1].values
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        y_mean = yy.mean()
        cov = np.mean((x - x_mean) * (yy - y_mean))
        var = np.mean((x - x_mean) ** 2)
        slope = cov / var if var != 0 else 0.0
        intercept = y_mean - slope * x_mean
        yhat = intercept + slope * x

        # assign the last point's fitted value to reg line at i
        reg.iloc[i] = yhat[-1]

        # compute goodness for the last window (keep last)
        ss_res = np.sum((yy - yhat) ** 2)
        ss_tot = np.sum((yy - y_mean) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        std = np.std(yy - yhat)

        last_slope = slope
        last_r2 = r2
        last_std = std

    # forward-fill early NaNs for plotting aesthetics
    reg = reg.ffill()
    return reg, float(last_slope), float(last_r2), float(last_std)


def normalized_price_channel(close: pd.Series, window: int = 120) -> pd.Series:
    """
    Simple normalized distance from rolling regression line in std-dev units.
    Equivalent vibe to "NPX" style: (close - reg) / std
    """
    close = _coerce_1d_series(close).astype(float)
    reg, _, _, std = rolling_regression(close, window=window)
    if not np.isfinite(std) or std == 0:
        return pd.Series(index=close.index, dtype=float)
    return (close - reg) / std


# ---------------------------
# ✅ Fibonacci helper (FIX for NameError)
# ---------------------------
def fibonacci_levels(close: pd.Series, lookback: int = 240) -> dict:
    """
    Returns Fibonacci retracement levels based on the highest high / lowest low
    observed over the last `lookback` bars of the CLOSE series.

    Output dict:
      {"0.0%": ..., "23.6%": ..., "38.2%": ..., "50.0%": ..., "61.8%": ..., "78.6%": ..., "100.0%": ...}
    """
    s = _coerce_1d_series(close).astype(float).dropna()
    if len(s) < 20:
        return {}

    s2 = s.iloc[-lookback:] if len(s) > lookback else s
    hi = float(np.nanmax(s2.values))
    lo = float(np.nanmin(s2.values))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}

    diff = hi - lo
    # standard retracements (from hi down to lo)
    levels = {
        "0.0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50.0%": hi - 0.500 * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100.0%": lo,
    }
    return levels


def _crossed_above(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _coerce_1d_series(a).astype(float)
    b = _coerce_1d_series(b).astype(float)
    if len(a) == 0 or len(b) == 0:
        return pd.Series(dtype=bool)
    return (a.shift(1) <= b.shift(1)) & (a > b)


def _crossed_below(a: pd.Series, b: pd.Series) -> pd.Series:
    a = _coerce_1d_series(a).astype(float)
    b = _coerce_1d_series(b).astype(float)
    if len(a) == 0 or len(b) == 0:
        return pd.Series(dtype=bool)
    return (a.shift(1) >= b.shift(1)) & (a < b)


# ---------------------------
# Chart rendering
# ---------------------------
def render_daily_price_chart(ticker: str, df: pd.DataFrame, daily_view: dict):
    """
    daily_view keys:
      - show_hma (bool)
      - show_reg (bool)
      - show_bands (bool)
      - show_fib (bool)
      - reg_window (int)
      - hma_period (int)
    """
    if df is None or df.empty:
        st.warning("No daily data available.")
        return

    close = df["Close"].astype(float)
    hma_period = int(daily_view.get("hma_period", 55))
    reg_window = int(daily_view.get("reg_window", 120))

    h = hma(close, hma_period) if daily_view.get("show_hma", True) else None
    reg, slope, r2, std = rolling_regression(close, window=reg_window) if daily_view.get("show_reg", True) else (None, np.nan, np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(df.index, close.values, linewidth=1.25, label="Close")

    if h is not None and len(h.dropna()):
        ax.plot(df.index, h.values, linewidth=1.15, label=f"HMA({hma_period})")

    if reg is not None and len(reg.dropna()):
        ax.plot(df.index, reg.values, linewidth=1.1, linestyle="--", label=f"Regression({reg_window})")
        if daily_view.get("show_bands", True) and np.isfinite(std) and std > 0:
            upper = reg + 2.0 * std
            lower = reg - 2.0 * std
            ax.plot(df.index, upper.values, linewidth=1.0, linestyle=":", label="+2σ")
            ax.plot(df.index, lower.values, linewidth=1.0, linestyle=":", label="-2σ")

    # ✅ Fibonacci (this was crashing when fibonacci_levels was undefined)
    if daily_view.get("show_fib", True):
        fibs = fibonacci_levels(close)
        if fibs:
            for k, v in fibs.items():
                ax.axhline(float(v), linewidth=0.9, alpha=0.55, linestyle="-")
            # add a small legend-like text block
            txt = "Fibs: " + ", ".join([f"{k}:{v:.2f}" for k, v in fibs.items()])
            ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=8, alpha=0.85)

    ax.set_title(f"{ticker} — Daily Price")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")

    # readable x-axis
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")

    # inline stats
    stat = f"Slope: {slope:.4f} | R²: {r2:.3f}" if np.isfinite(slope) and np.isfinite(r2) else "Slope/R²: n/a"
    ax.text(0.99, 0.02, stat, transform=ax.transAxes, fontsize=9, ha="right", va="bottom", alpha=0.9)

    st.pyplot(fig, use_container_width=True)


def render_hourly_price_chart(ticker: str, df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No hourly data available.")
        return

    close = df["Close"].astype(float)

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(df.index, close.values, linewidth=1.15, label="Close")
    ax.set_title(f"{ticker} — Hourly Price")
    ax.legend(loc="upper left")

    # readability fix: rotate, fewer ticks, auto locator
    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    st.pyplot(fig, use_container_width=True)


# ---------------------------
# UI: symbol picker + sidebar controls
# ---------------------------
def _symbol_picker(universe: list[str], label: str, key: str) -> str:
    if not universe:
        return ""
    # ensure stable index
    return st.selectbox(label, universe, index=0, key=key)


def _sidebar_controls():
    st.sidebar.markdown("### Controls")

    mode = st.sidebar.radio("Mode", ["Forex", "Stocks"], index=0, key="mode_radio")
    universe = DEFAULT_FOREX if mode == "Forex" else DEFAULT_STOCKS

    custom = st.sidebar.text_area(
        "Optional: Custom symbols (comma-separated)",
        value="",
        help="Example: AAPL, MSFT, NVDA  or  EURUSD=X, USDJPY=X",
        key="custom_symbols"
    ).strip()

    if custom:
        uni = [x.strip() for x in custom.split(",") if x.strip()]
        if uni:
            universe = uni

    ticker = _symbol_picker(universe, f"{mode} Symbol", key=f"symbol_{mode.lower()}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Daily View")
    show_hma = st.sidebar.checkbox("Show HMA", value=True, key="dv_show_hma")
    show_reg = st.sidebar.checkbox("Show Regression", value=True, key="dv_show_reg")
    show_bands = st.sidebar.checkbox("Show ±2σ Bands", value=True, key="dv_show_bands")
    show_fib = st.sidebar.checkbox("Show Fibonacci Levels", value=True, key="dv_show_fib")

    hma_period = st.sidebar.number_input("HMA Period", min_value=5, max_value=200, value=55, step=1, key="dv_hma_period")
    reg_window = st.sidebar.number_input("Regression Window", min_value=30, max_value=400, value=120, step=5, key="dv_reg_window")

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("Refresh Data", key="btn_refresh")

    daily_view = {
        "show_hma": bool(show_hma),
        "show_reg": bool(show_reg),
        "show_bands": bool(show_bands),
        "show_fib": bool(show_fib),
        "hma_period": int(hma_period),
        "reg_window": int(reg_window),
    }

    return mode, ticker, universe, daily_view, refresh


# ---------------------------
# Title / header
# ---------------------------
st.markdown("## 📈 bullbear.py — Stocks/Forex Dashboard + Forecasts")
st.caption("Daily + Hourly charts, regression bands, scanners (including HMA Buy), and quick metrics.")


mode, ticker, universe, daily_view, refresh = _sidebar_controls()

# basic guard
if not ticker:
    st.stop()

# data pulls (daily + hourly)
daily_df = _download_ohlc(ticker, period="2y", interval="1d")
hourly_df = _download_ohlc(ticker, period="60d", interval="1h")

# ---------------------------
# Additional analytics helpers
# ---------------------------
def _pct_returns(close: pd.Series) -> pd.Series:
    c = _coerce_1d_series(close).astype(float)
    return c.pct_change()


def _max_drawdown(close: pd.Series) -> float:
    c = _coerce_1d_series(close).astype(float).dropna()
    if len(c) < 5:
        return np.nan
    peak = c.cummax()
    dd = (c / peak) - 1.0
    return float(dd.min())


def _annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _coerce_1d_series(returns).astype(float).dropna()
    if len(r) < 10:
        return np.nan
    return float(r.std() * np.sqrt(periods_per_year))


def _annualized_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = _coerce_1d_series(returns).astype(float).dropna()
    if len(r) < 10:
        return np.nan
    excess = r - (rf / periods_per_year)
    sd = excess.std()
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def _coerce_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


# ---------------------------
# Simple forecast (fast + no extra deps)
# ---------------------------
def simple_regression_forecast(close: pd.Series, window: int = 120, steps: int = 14):
    """
    Forecast next `steps` points using last-window linear regression slope/intercept.
    Returns:
      forecast_index, forecast_values, reg_line_last_window, std_last_window
    """
    c = _coerce_1d_series(close).astype(float).dropna()
    if len(c) < max(window, 30):
        return None, None, None, np.nan

    y = c.iloc[-window:].values
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = np.mean((x - x_mean) * (y - y_mean))
    var = np.mean((x - x_mean) ** 2)
    slope = cov / var if var != 0 else 0.0
    intercept = y_mean - slope * x_mean
    yhat = intercept + slope * x
    resid = y - yhat
    std = float(np.std(resid))

    # forecast forward
    xf = np.arange(window, window + steps, dtype=float)
    yf = intercept + slope * xf

    last_date = c.index[-1]
    # assume business days for daily data; for simplicity add calendar days and rely on index labels only
    idx = [last_date + timedelta(days=i) for i in range(1, steps + 1)]

    return idx, yf, pd.Series(yhat, index=c.index[-window:]), std


def render_forecast_chart(ticker: str, df: pd.DataFrame, window: int = 120, steps: int = 14):
    if df is None or df.empty:
        st.warning("No daily data available.")
        return

    close = df["Close"].astype(float)
    out = simple_regression_forecast(close, window=window, steps=steps)
    if out[0] is None:
        st.warning("Not enough data to forecast.")
        return

    idx_f, yf, yhat, std = out

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.plot(df.index, close.values, linewidth=1.25, label="Close")

    # regression fit (last window)
    ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=1.1, label=f"Reg fit ({window})")

    # forecast line
    ax.plot(idx_f, yf, linewidth=1.25, label=f"Forecast ({steps}d)")

    # ±2σ around forecast (carry std)
    if np.isfinite(std) and std > 0:
        upper = yf + 2.0 * std
        lower = yf - 2.0 * std
        ax.plot(idx_f, upper, linestyle=":", linewidth=1.0, label="+2σ (forecast)")
        ax.plot(idx_f, lower, linestyle=":", linewidth=1.0, label="-2σ (forecast)")

    ax.set_title(f"{ticker} — Simple Regression Forecast")
    ax.legend(loc="upper left")

    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")

    st.pyplot(fig, use_container_width=True)


def render_enhanced_forecast_chart(ticker: str, df: pd.DataFrame, reg_window: int = 120):
    """
    Enhanced daily chart:
      - Close
      - Regression line + ±2σ bands
      - NPX (normalized distance) shown as a secondary panel-like overlay text/markers (kept simple in one axis)
    """
    if df is None or df.empty:
        st.warning("No daily data available.")
        return

    close = df["Close"].astype(float)
    reg, slope, r2, std = rolling_regression(close, window=reg_window)
    npx = normalized_price_channel(close, window=reg_window)

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(df.index, close.values, linewidth=1.25, label="Close")

    if len(reg.dropna()):
        ax.plot(df.index, reg.values, linestyle="--", linewidth=1.1, label=f"Regression({reg_window})")
        if np.isfinite(std) and std > 0:
            upper = reg + 2.0 * std
            lower = reg - 2.0 * std
            ax.plot(df.index, upper.values, linestyle=":", linewidth=1.0, label="+2σ")
            ax.plot(df.index, lower.values, linestyle=":", linewidth=1.0, label="-2σ")

    # annotate last NPX
    npx_last = _coerce_float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan
    stat = f"Slope: {slope:.4f} | R²: {r2:.3f} | NPX: {npx_last:.2f}" if np.isfinite(npx_last) else f"Slope: {slope:.4f} | R²: {r2:.3f}"
    ax.text(0.99, 0.02, stat, transform=ax.transAxes, fontsize=9, ha="right", va="bottom", alpha=0.9)

    ax.set_title(f"{ticker} — Enhanced Daily (Reg + Bands + NPX)")
    ax.legend(loc="upper left")

    ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=9))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")

    st.pyplot(fig, use_container_width=True)


# ---------------------------
# Scanners (multi-symbol)
# ---------------------------
@st.cache_data(show_spinner=False)
def _download_daily_for_scan(symbol: str) -> pd.DataFrame:
    return _download_ohlc(symbol, period="1y", interval="1d")


def scan_hma_buy(universe: list[str], hma_period: int = 55, reg_window: int = 120, bars_back: int = 3) -> pd.DataFrame:
    """
    HMA Buy scanner:
      - Price crossed ABOVE HMA within last `bars_back` bars
      - Categorize by regression slope >0 vs <0
    """
    rows = []
    bars_back = max(int(bars_back), 1)

    for sym in universe:
        df = _download_daily_for_scan(sym)
        if df is None or df.empty or len(df) < max(reg_window, hma_period, 60):
            continue

        close = df["Close"].astype(float)
        h = hma(close, hma_period)
        if len(h.dropna()) < 5:
            continue

        cross_up = _crossed_above(close, h)
        # find most recent cross-up index
        cross_idx = cross_up[cross_up.fillna(False)].index
        if len(cross_idx) == 0:
            continue

        last_cross_date = cross_idx[-1]
        # bars since last cross
        try:
            pos_cross = df.index.get_loc(last_cross_date)
            bars_since = (len(df) - 1) - pos_cross
        except Exception:
            continue

        if bars_since < 0 or bars_since > bars_back:
            continue

        reg, slope, r2, std = rolling_regression(close, window=reg_window)
        npx = normalized_price_channel(close, window=reg_window)
        npx_last = _coerce_float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan

        rows.append({
            "Symbol": sym,
            "BarsSinceCross": int(bars_since),
            "CrossDate": str(pd.to_datetime(last_cross_date).date()),
            "Close": float(close.dropna().iloc[-1]),
            f"HMA({hma_period})": float(h.dropna().iloc[-1]),
            "RegSlope": float(slope),
            "R2": float(r2),
            "NPX": float(npx_last),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["BarsSinceCross", "RegSlope"], ascending=[True, False]).reset_index(drop=True)
    return out


def scan_r2_trend(universe: list[str], reg_window: int = 120, r2_min: float = 0.45):
    rows_up, rows_down = [], []
    for sym in universe:
        df = _download_daily_for_scan(sym)
        if df is None or df.empty or len(df) < max(reg_window, 80):
            continue
        close = df["Close"].astype(float)
        reg, slope, r2, std = rolling_regression(close, window=reg_window)
        if not np.isfinite(r2) or not np.isfinite(slope):
            continue
        if r2 < float(r2_min):
            continue

        row = {
            "Symbol": sym,
            "Close": float(close.dropna().iloc[-1]),
            "RegSlope": float(slope),
            "R2": float(r2),
        }
        if slope > 0:
            rows_up.append(row)
        elif slope < 0:
            rows_down.append(row)

    up = pd.DataFrame(rows_up).sort_values(["R2", "RegSlope"], ascending=[False, False]).reset_index(drop=True) if rows_up else pd.DataFrame()
    down = pd.DataFrame(rows_down).sort_values(["R2", "RegSlope"], ascending=[False, True]).reset_index(drop=True) if rows_down else pd.DataFrame()
    return up, down


def scan_npx_extremes(universe: list[str], reg_window: int = 120, low_thr: float = -1.0, high_thr: float = 1.0):
    rows_low, rows_high = [], []
    for sym in universe:
        df = _download_daily_for_scan(sym)
        if df is None or df.empty or len(df) < max(reg_window, 80):
            continue
        close = df["Close"].astype(float)
        reg, slope, r2, std = rolling_regression(close, window=reg_window)
        npx = normalized_price_channel(close, window=reg_window)
        if len(npx.dropna()) == 0:
            continue
        n = float(npx.dropna().iloc[-1])
        row = {
            "Symbol": sym,
            "Close": float(close.dropna().iloc[-1]),
            "NPX": n,
            "RegSlope": float(slope),
            "R2": float(r2),
        }
        if n <= float(low_thr):
            rows_low.append(row)
        if n >= float(high_thr):
            rows_high.append(row)

    low = pd.DataFrame(rows_low).sort_values(["NPX", "R2"], ascending=[True, False]).reset_index(drop=True) if rows_low else pd.DataFrame()
    high = pd.DataFrame(rows_high).sort_values(["NPX", "R2"], ascending=[False, False]).reset_index(drop=True) if rows_high else pd.DataFrame()
    return low, high


# ---------------------------
# Tabs (15 total)
# ---------------------------
tabs = st.tabs([
    "Overview",
    "Daily",
    "Hourly",
    "Forecast",
    "Enhanced",
    "Metrics",
    "HMA Buy",
    "R² Scanner",
    "NPX Scanner",
    "Volatility",
    "Watchlist",
    "Correlation",
    "Notes",
    "Help",
    "About",
])

# TAB 1 — Overview
with tabs[0]:
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)

    if daily_df is not None and not daily_df.empty:
        close = daily_df["Close"].astype(float).dropna()
        last = float(close.iloc[-1]) if len(close) else np.nan
        ret_1d = float(close.pct_change().iloc[-1]) if len(close) > 1 else np.nan
        ret_5d = float(close.pct_change(5).iloc[-1]) if len(close) > 6 else np.nan
        dd = _max_drawdown(close)

        col1.metric("Last Close", f"{last:,.4f}" if np.isfinite(last) else "n/a")
        col2.metric("1D Return", f"{ret_1d*100:.2f}%" if np.isfinite(ret_1d) else "n/a")
        col3.metric("5D Return", f"{ret_5d*100:.2f}%" if np.isfinite(ret_5d) else "n/a")
        col4.metric("Max Drawdown (2y)", f"{dd*100:.2f}%" if np.isfinite(dd) else "n/a")

        st.markdown("#### Recent daily bars")
        st.dataframe(daily_df.tail(12), use_container_width=True)
    else:
        st.info("Daily data not available for this symbol.")

# TAB 2 — Daily
with tabs[1]:
    st.subheader("Daily Chart")
    render_daily_price_chart(ticker, daily_df, daily_view)

# TAB 3 — Hourly
with tabs[2]:
    st.subheader("Hourly Chart")
    render_hourly_price_chart(ticker, hourly_df)

# TAB 4 — Forecast
with tabs[3]:
    st.subheader("Forecast")
    c1, c2 = st.columns([1, 1])
    with c1:
        fc_window = st.number_input("Regression window", min_value=30, max_value=400, value=int(daily_view["reg_window"]), step=5, key="fc_window")
    with c2:
        fc_steps = st.number_input("Forecast steps (days)", min_value=5, max_value=60, value=14, step=1, key="fc_steps")

    render_forecast_chart(ticker, daily_df, window=int(fc_window), steps=int(fc_steps))

# TAB 5 — Enhanced
with tabs[4]:
    st.subheader("Enhanced Daily")
    reg_window_enh = st.number_input("Enhanced regression window", min_value=30, max_value=400, value=int(daily_view["reg_window"]), step=5, key="enh_reg_window")
    render_enhanced_forecast_chart(ticker, daily_df, reg_window=int(reg_window_enh))

# TAB 6 — Metrics
with tabs[5]:
    st.subheader("Metrics")
    if daily_df is None or daily_df.empty:
        st.info("No daily data.")
    else:
        close = daily_df["Close"].astype(float).dropna()
        r = _pct_returns(close)

        vol = _annualized_vol(r, periods_per_year=252)
        shrp = _annualized_sharpe(r, rf=0.0, periods_per_year=252)
        dd = _max_drawdown(close)

        reg, slope, r2, std = rolling_regression(close, window=int(daily_view["reg_window"]))
        npx = normalized_price_channel(close, window=int(daily_view["reg_window"]))
        npx_last = _coerce_float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Ann. Vol", f"{vol*100:.2f}%" if np.isfinite(vol) else "n/a")
        m2.metric("Sharpe", f"{shrp:.2f}" if np.isfinite(shrp) else "n/a")
        m3.metric("Max Drawdown", f"{dd*100:.2f}%" if np.isfinite(dd) else "n/a")
        m4.metric("Reg Slope", f"{slope:.4f}" if np.isfinite(slope) else "n/a")
        m5.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "n/a")

        st.markdown("#### NPX (normalized distance from regression)")
        st.write(f"Latest NPX: **{npx_last:.2f}**" if np.isfinite(npx_last) else "Latest NPX: n/a")
        npx_df = pd.DataFrame({"NPX": npx}).dropna().tail(60)
        st.line_chart(npx_df, use_container_width=True)

# TAB 7 — HMA Buy (Scanner)
with tabs[6]:
    st.subheader("HMA Buy Scanner")

    st.markdown(
        "Find symbols where **price recently crossed ABOVE HMA(55)** on the daily chart within **N bars**.\n"
        "Results are split into **Regression slope > 0** vs **Regression slope < 0**."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scan_hma_period = st.number_input("HMA period", min_value=5, max_value=200, value=55, step=1, key="scan_hma_period")
    with c2:
        scan_reg_window = st.number_input("Regression window", min_value=30, max_value=400, value=120, step=5, key="scan_reg_window")
    with c3:
        scan_bars = st.slider("Bars since cross", min_value=1, max_value=10, value=3, step=1, key="scan_bars")

    run_scan = st.button("Run HMA Buy Scan", key="btn_run_hma_scan")

    if run_scan:
        with st.spinner("Scanning universe…"):
            res = scan_hma_buy(
                universe=universe,
                hma_period=int(scan_hma_period),
                reg_window=int(scan_reg_window),
                bars_back=int(scan_bars),
            )

        if res is None or res.empty:
            st.info("No matches found.")
        else:
            pos = res[res["RegSlope"] > 0].copy()
            neg = res[res["RegSlope"] < 0].copy()

            st.markdown("### Regression > 0")
            if pos.empty:
                st.write("None")
            else:
                st.dataframe(pos, use_container_width=True)

            st.markdown("### Regression < 0")
            if neg.empty:
                st.write("None")
            else:
                st.dataframe(neg, use_container_width=True)

            st.markdown("### All Matches")
            st.dataframe(res, use_container_width=True)

# TAB 8 — R² Scanner
with tabs[7]:
    st.subheader("R² Trend Scanner")

    c1, c2 = st.columns([1, 1])
    with c1:
        r2_window = st.number_input("Regression window", min_value=30, max_value=400, value=120, step=5, key="r2_scan_window")
    with c2:
        r2_min = st.slider("Min R²", min_value=0.05, max_value=0.95, value=0.45, step=0.05, key="r2_min")

    run_r2 = st.button("Run R² Scan", key="btn_run_r2_scan")

    if run_r2:
        with st.spinner("Scanning universe…"):
            up, down = scan_r2_trend(universe, reg_window=int(r2_window), r2_min=float(r2_min))

        st.markdown("### Trend Up (Slope > 0)")
        if up is None or up.empty:
            st.write("None")
        else:
            st.dataframe(up, use_container_width=True)

        st.markdown("### Trend Down (Slope < 0)")
        if down is None or down.empty:
            st.write("None")
        else:
            st.dataframe(down, use_container_width=True)

# TAB 9 — NPX Scanner
with tabs[8]:
    st.subheader("NPX Scanner")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        npx_window = st.number_input("Regression window", min_value=30, max_value=400, value=120, step=5, key="npx_scan_window")
    with c2:
        npx_low = st.slider("Low NPX threshold", min_value=-3.0, max_value=0.0, value=-1.0, step=0.1, key="npx_low_thr")
    with c3:
        npx_high = st.slider("High NPX threshold", min_value=0.0, max_value=3.0, value=1.0, step=0.1, key="npx_high_thr")

    run_npx = st.button("Run NPX Scan", key="btn_run_npx_scan")

    if run_npx:
        with st.spinner("Scanning universe…"):
            low_df, high_df = scan_npx_extremes(
                universe,
                reg_window=int(npx_window),
                low_thr=float(npx_low),
                high_thr=float(npx_high),
            )

        st.markdown(f"### NPX ≤ {float(npx_low):.2f}")
        if low_df is None or low_df.empty:
            st.write("None")
        else:
            st.dataframe(low_df, use_container_width=True)

        st.markdown(f"### NPX ≥ {float(npx_high):.2f}")
        if high_df is None or high_df.empty:
            st.write("None")
        else:
            st.dataframe(high_df, use_container_width=True)

# TAB 10 — Volatility
with tabs[9]:
    st.subheader("Volatility")

    if daily_df is None or daily_df.empty:
        st.info("No daily data.")
    else:
        close = daily_df["Close"].astype(float).dropna()
        r = _pct_returns(close).dropna()

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            vol_window = st.number_input("Rolling window (days)", min_value=5, max_value=120, value=20, step=5, key="vol_roll_window")
        with c2:
            ann = st.selectbox("Annualization", ["252 (stocks)", "365 (crypto/forex)"], index=0, key="vol_ann")
        with c3:
            show_hist = st.checkbox("Show returns histogram", value=False, key="vol_show_hist")

        periods = 252 if ann.startswith("252") else 365
        roll = r.rolling(int(vol_window)).std() * np.sqrt(periods)
        vol_last = float(roll.dropna().iloc[-1]) if len(roll.dropna()) else np.nan

        st.metric("Latest Rolling Ann. Vol", f"{vol_last*100:.2f}%" if np.isfinite(vol_last) else "n/a")

        fig, ax = plt.subplots(figsize=(12.5, 4.8))
        ax.plot(roll.index, roll.values, linewidth=1.25, label=f"Rolling vol ({int(vol_window)}d)")
        ax.set_title(f"{ticker} — Rolling Annualized Volatility")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(AutoDateLocator(minticks=4, maxticks=9))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")
        st.pyplot(fig, use_container_width=True)

        if show_hist and len(r) >= 30:
            fig2, ax2 = plt.subplots(figsize=(12.5, 4.0))
            ax2.hist(r.values, bins=40)
            ax2.set_title(f"{ticker} — Daily Returns Histogram")
            st.pyplot(fig2, use_container_width=True)

# TAB 11 — Watchlist
with tabs[10]:
    st.subheader("Watchlist")

    st.caption("Enter a comma-separated list of symbols (forex like EURUSD=X, stocks like AAPL).")
    wl_default = st.session_state.get("watchlist_text", "")
    wl_text = st.text_input("Symbols", value=wl_default, key="watchlist_text_input")
    st.session_state["watchlist_text"] = wl_text

    wl = [s.strip() for s in wl_text.split(",") if s.strip()]
    if len(wl) == 0:
        st.info("Add symbols to build a watchlist.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            wl_reg_window = st.number_input("Regression window", min_value=30, max_value=400, value=120, step=5, key="wl_reg_window")
        with c2:
            wl_hma_period = st.number_input("HMA period", min_value=5, max_value=200, value=55, step=1, key="wl_hma_period")
        with c3:
            wl_period = st.selectbox("Download period", ["6mo", "1y", "2y"], index=1, key="wl_dl_period")

        rows = []
        for sym in wl:
            dfw = _download_ohlc(sym, period=wl_period, interval="1d")
            if dfw is None or dfw.empty or len(dfw) < max(int(wl_reg_window), int(wl_hma_period), 60):
                rows.append({"Symbol": sym, "Status": "No/insufficient data"})
                continue

            close = dfw["Close"].astype(float).dropna()
            reg, slope, r2, std = rolling_regression(close, window=int(wl_reg_window))
            npx = normalized_price_channel(close, window=int(wl_reg_window))
            npx_last = _coerce_float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan
            h = hma(close, int(wl_hma_period))
            h_last = float(h.dropna().iloc[-1]) if len(h.dropna()) else np.nan
            last = float(close.iloc[-1]) if len(close) else np.nan
            ret1 = float(close.pct_change().iloc[-1]) if len(close) > 1 else np.nan

            rows.append({
                "Symbol": sym,
                "Last": last,
                "1D%": ret1 * 100.0 if np.isfinite(ret1) else np.nan,
                f"HMA({int(wl_hma_period)})": h_last,
                "RegSlope": float(slope) if np.isfinite(slope) else np.nan,
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "NPX": float(npx_last) if np.isfinite(npx_last) else np.nan,
                "Status": "OK",
            })

        wldf = pd.DataFrame(rows)
        st.dataframe(wldf, use_container_width=True)

        csv = wldf.to_csv(index=False).encode("utf-8")
        st.download_button("Download Watchlist CSV", csv, file_name="watchlist.csv", mime="text/csv", key="dl_watchlist_csv")

# TAB 12 — Correlation
with tabs[11]:
    st.subheader("Correlation")

    wl_text2 = st.session_state.get("watchlist_text", "")
    wl2 = [s.strip() for s in wl_text2.split(",") if s.strip()]

    if len(wl2) < 2:
        st.info("Add at least 2 symbols in the Watchlist to view correlations.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            corr_period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="corr_period")
        with c2:
            corr_method = st.selectbox("Method", ["pearson", "spearman"], index=0, key="corr_method")

        rets = {}
        for sym in wl2:
            dfc = _download_ohlc(sym, period=corr_period, interval="1d")
            if dfc is None or dfc.empty:
                continue
            close = dfc["Close"].astype(float).dropna()
            r = close.pct_change().dropna()
            if len(r) < 30:
                continue
            rets[sym] = r

        if len(rets) < 2:
            st.warning("Not enough overlapping data to compute correlations.")
        else:
            ret_df = pd.DataFrame(rets).dropna(how="any")
            if ret_df.shape[0] < 30:
                st.warning("Overlap after alignment is small; correlation may be noisy.")
            corr = ret_df.corr(method=corr_method)

            st.dataframe(corr.round(3), use_container_width=True)

            # Heatmap using matplotlib (no seaborn)
            fig, ax = plt.subplots(figsize=(10.5, 7.5))
            im = ax.imshow(corr.values, aspect="auto")

            ax.set_xticks(np.arange(len(corr.columns)))
            ax.set_yticks(np.arange(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.index)

            ax.set_title(f"Correlation Heatmap ({corr_method})")
            fig.colorbar(im, ax=ax, shrink=0.85)

            # annotate values
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

            st.pyplot(fig, use_container_width=True)

# TAB 13 — Notes
with tabs[12]:
    st.subheader("Notes")
    st.caption("Local notes stored in your current Streamlit session (not persisted server-side).")
    if "notes_text" not in st.session_state:
        st.session_state["notes_text"] = ""

    st.session_state["notes_text"] = st.text_area(
        "Your notes",
        value=st.session_state["notes_text"],
        height=260,
        key="notes_textarea",
    )

# TAB 14 — Help
with tabs[13]:
    st.subheader("Help / Troubleshooting")

    st.markdown(
        """
**Common issues**
- **Empty chart / missing data**: Yahoo sometimes returns sparse history for some tickers or forex pairs.
- **Slow scans**: Scanners download daily data per symbol. Reduce universe size or scan less frequently.
- **NameError: fibonacci_levels**: This app requires a `fibonacci_levels(series)` helper. It is included in this build
  so the daily chart can compute Fibonacci levels safely (only when enough data exists).
- **StreamlitDuplicateElementKey**: If you copy/paste widgets, ensure `key=` values are unique.

**Tips**
- Use **Daily View** toggles (regression window, show fibs, etc.) to reduce chart clutter.
- For scanners, start with a smaller universe for faster iteration.
        """
    )

# TAB 15 — About
with tabs[14]:
    st.subheader("About")
    st.markdown(
        """
**bullbear.py — Stocks/Forex Dashboard + Forecasts**

- Multi-tab Streamlit dashboard with:
  - Daily and hourly charts
  - Regression + ±2σ bands
  - NPX normalization (distance from regression in σ units)
  - Scanners (HMA Buy, R², NPX)
  - Watchlist and correlation heatmap

If you want additional scanners (Supertrend flips, PSAR, MACD, etc.) we can layer them in without changing the current UI layout.
        """
    )

# ---- End of file ----
