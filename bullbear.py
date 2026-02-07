# bullbear.py
# Streamlit single-file app: Daily + Intraday charts + scanners
# Fix: Ensure last_daily_kijun_cross_up() is defined before it is used.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception as e:
    yf = None


# -----------------------------
# Utilities
# -----------------------------
def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (pd.DataFrame,)):
        if x.shape[1] == 0:
            return pd.Series(dtype=float)
        return x.iloc[:, 0]
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype=float)


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _as_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df.sort_index()


def _series_heading_up(s: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(s).dropna()
    if len(s) < max(2, confirm_bars + 1):
        return False
    diffs = np.diff(s.values.astype(float))
    if confirm_bars <= 1:
        return diffs[-1] > 0
    return np.all(diffs[-confirm_bars:] > 0)


# -----------------------------
# Data Fetch
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_hist(symbol: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV using yfinance. Returns columns: Open/High/Low/Close/Volume.
    Uses auto_adjust=True (prices adjusted).
    """
    if yf is None:
        return pd.DataFrame()

    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = _as_date_index(df)

        # Normalize column names
        df.columns = [str(c).title() for c in df.columns]
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                return pd.DataFrame()
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(symbol: str, period: str = "1d", interval: str = "60m") -> pd.DataFrame:
    """
    Intraday OHLCV using yfinance. period like '1d','2d','5d','1mo' etc.
    interval '60m' for hourly.
    """
    if yf is None:
        return pd.DataFrame()

    symbol = (symbol or "").strip().upper()
    if not symbol:
        return pd.DataFrame()

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = _as_date_index(df)
        df.columns = [str(c).title() for c in df.columns]
        if "Close" not in df.columns:
            return pd.DataFrame()
        for col in ["Open", "High", "Low"]:
            if col not in df.columns:
                df[col] = np.nan
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def subset_by_daily_view(close: pd.Series, daily_view_label: str) -> pd.Series:
    """
    Applies a view-range to a daily close series.
    """
    close = _coerce_1d_series(close).dropna()
    if close.empty:
        return close

    label = (daily_view_label or "6M").strip().upper()

    # Approx trading days mapping
    mapping = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "9M": 189,
        "1Y": 252,
        "2Y": 504,
        "3Y": 756,
        "5Y": 1260,
        "10Y": 2520,
        "MAX": None,
    }
    n = mapping.get(label, 126)
    return close if n is None else close.tail(int(n))


# -----------------------------
# Indicators
# -----------------------------
def compute_bbands(
    close: pd.Series,
    window: int = 20,
    mult: float = 2.0,
    use_ema: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (mid, upper, lower, pctb, nbb)
    pctb = (close - lower)/(upper-lower)
    nbb  = pctb - 1  -> [-1 at lower, 0 at upper], so -0.8..-1.0 means near lower band.
    """
    close = _coerce_1d_series(close).astype(float).dropna()
    if close.empty or len(close) < max(5, window):
        na = pd.Series(index=close.index, dtype=float)
        return na, na, na, na, na

    if use_ema:
        mid = close.ewm(span=window, adjust=False).mean()
        std = close.ewm(span=window, adjust=False).std(bias=False)
    else:
        mid = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)

    upper = mid + mult * std
    lower = mid - mult * std
    denom = (upper - lower).replace(0, np.nan)
    pctb = (close - lower) / denom
    nbb = pctb - 1.0
    return mid, upper, lower, pctb, nbb


def compute_normalized_price(close: pd.Series, window: int = 50) -> pd.Series:
    """
    NPX in [0..1]: position of price within rolling min/max.
    """
    close = _coerce_1d_series(close).astype(float).dropna()
    if close.empty or len(close) < max(5, window):
        return pd.Series(index=close.index, dtype=float)

    lo = close.rolling(window).min()
    hi = close.rolling(window).max()
    denom = (hi - lo).replace(0, np.nan)
    npx = (close - lo) / denom
    return npx


def compute_normalized_trend(close: pd.Series, window: int = 50) -> pd.Series:
    """
    NTD: rolling regression slope normalized by price range to be roughly comparable.
    Not "the" canonical formula; used as a stable intraday trend signal for this app.
    """
    close = _coerce_1d_series(close).astype(float).dropna()
    if close.empty or len(close) < max(10, window):
        return pd.Series(index=close.index, dtype=float)

    x = np.arange(window, dtype=float)

    out = pd.Series(index=close.index, dtype=float)
    vals = close.values.astype(float)
    for i in range(window - 1, len(vals)):
        y = vals[i - window + 1 : i + 1]
        if np.any(~np.isfinite(y)):
            out.iloc[i] = np.nan
            continue
        slope, intercept = np.polyfit(x, y, 1)
        rng = np.nanmax(y) - np.nanmin(y)
        out.iloc[i] = slope / (rng + 1e-9)
    return out


def regression_with_band(y: pd.Series, band_mult: float = 2.0) -> Dict[str, pd.Series | float]:
    """
    Fits y ~ a*x + b on the full y series and computes ±band_mult*std(resid).
    Returns dict with slope, intercept, r2, fit, upper, lower, sigma.
    """
    y = _coerce_1d_series(y).astype(float).dropna()
    if y.empty or len(y) < 5:
        na = pd.Series(index=y.index, dtype=float)
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "fit": na,
            "upper": na,
            "lower": na,
            "sigma": np.nan,
        }

    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y.values.astype(float), 1)
    fit = slope * x + intercept
    resid = y.values.astype(float) - fit
    sigma = float(np.nanstd(resid))
    upper = fit + band_mult * sigma
    lower = fit - band_mult * sigma

    ss_res = float(np.nansum((resid) ** 2))
    ss_tot = float(np.nansum((y.values.astype(float) - np.nanmean(y.values.astype(float))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fit_s = pd.Series(fit, index=y.index)
    upper_s = pd.Series(upper, index=y.index)
    lower_s = pd.Series(lower, index=y.index)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "fit": fit_s,
        "upper": upper_s,
        "lower": lower_s,
        "sigma": float(sigma),
    }


def signed_r2_from_slope(r2: float, slope: float) -> float:
    if not np.isfinite(r2) or not np.isfinite(slope):
        return np.nan
    return float(r2) * (1.0 if slope >= 0 else -1.0)


# -----------------------------
# Daily helpers used by tabs
# -----------------------------
def daily_global_slope(symbol: str, daily_view_label: str) -> Tuple[float, float, Optional[pd.Timestamp]]:
    df = fetch_hist(symbol)
    if df.empty:
        return np.nan, np.nan, None
    close_full = _coerce_1d_series(df["Close"]).dropna()
    close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_show.empty or len(close_show) < 5:
        return np.nan, np.nan, close_show.index[-1] if len(close_show) else None
    reg = regression_with_band(close_show, band_mult=2.0)
    return _safe_float(reg["slope"]), _safe_float(reg["r2"]), close_show.index[-1]


def daily_regression_r2(symbol: str, daily_view_label: str, slope_lb: int) -> Tuple[float, float, Optional[pd.Timestamp]]:
    df = fetch_hist(symbol)
    if df.empty:
        return np.nan, np.nan, None
    close_full = _coerce_1d_series(df["Close"]).dropna()
    close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_show.empty:
        return np.nan, np.nan, None
    y = close_show.tail(int(max(5, slope_lb)))
    if len(y) < 5:
        return np.nan, np.nan, y.index[-1]
    reg = regression_with_band(y, band_mult=2.0)
    return _safe_float(reg["r2"]), _safe_float(reg["slope"]), y.index[-1]


def hourly_regression_r2(symbol: str, period: str, slope_lb: int) -> Tuple[float, float, Optional[pd.Timestamp]]:
    df = fetch_intraday(symbol, period=period, interval="60m")
    if df.empty:
        return np.nan, np.nan, None
    close = _coerce_1d_series(df["Close"]).dropna()
    y = close.tail(int(max(5, slope_lb)))
    if len(y) < 5:
        return np.nan, np.nan, y.index[-1] if len(y) else None
    reg = regression_with_band(y, band_mult=2.0)
    return _safe_float(reg["r2"]), _safe_float(reg["slope"]), y.index[-1]


def daily_npx_series_in_view(symbol: str, daily_view_label: str, ntd_win: int) -> pd.Series:
    df = fetch_hist(symbol)
    if df.empty:
        return pd.Series(dtype=float)
    close_full = _coerce_1d_series(df["Close"]).dropna()
    close_show = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_show.empty:
        return pd.Series(dtype=float)
    return compute_normalized_price(close_show, window=int(ntd_win))


def daily_last_npx_in_view(symbol: str, daily_view_label: str, ntd_win: int) -> Tuple[float, Optional[pd.Timestamp]]:
    npx = daily_npx_series_in_view(symbol, daily_view_label, ntd_win).dropna()
    if npx.empty:
        return np.nan, None
    return _safe_float(npx.iloc[-1]), npx.index[-1]


def sr_levels(close: pd.Series, lookback: int = 50) -> Tuple[pd.Series, pd.Series]:
    close = _coerce_1d_series(close).astype(float).dropna()
    if close.empty or len(close) < max(5, lookback):
        na = pd.Series(index=close.index, dtype=float)
        return na, na
    support = close.rolling(lookback).min()
    resistance = close.rolling(lookback).max()
    return support, resistance


def daily_support_reversal_heading_up(
    symbol: str,
    daily_view_label: str,
    sr_lb: int,
    prox: float,
    bars_confirm: int,
    horizon: int,
) -> Optional[Dict[str, object]]:
    """
    Detects: within the last `horizon` bars, price "touched" support (within prox)
    and then headed up (last `bars_confirm` diffs > 0).
    Returns a dict with touch info and current values.
    """
    df = fetch_hist(symbol)
    if df.empty:
        return None
    close_full = _coerce_1d_series(df["Close"]).dropna()
    close = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close.empty or len(close) < max(10, sr_lb + bars_confirm + 2):
        return None

    support, _ = sr_levels(close, lookback=int(sr_lb))
    support = support.dropna()
    if support.empty:
        return None

    close2 = close.loc[support.index]
    if close2.empty:
        return None

    hz = int(max(3, horizon))
    segment = close2.tail(hz)
    supp_seg = support.loc[segment.index]

    # Touch: close within prox of support (relative)
    rel = (segment - supp_seg).abs() / (supp_seg.replace(0, np.nan)).abs()
    touch_mask = rel <= float(prox)

    if not touch_mask.any():
        return None

    touch_idx = touch_mask[touch_mask].index[-1]
    touch_pos = segment.index.get_loc(touch_idx)

    # Need bars after touch for confirmation
    if touch_pos >= len(segment) - (bars_confirm + 1):
        return None

    after = segment.iloc[touch_pos:]
    if not _series_heading_up(after, confirm_bars=int(bars_confirm)):
        return None

    bars_since = len(segment) - 1 - touch_pos
    return {
        "Symbol": symbol,
        "Touch Date": touch_idx,
        "Touch Close": _safe_float(segment.loc[touch_idx]),
        "Support@Touch": _safe_float(supp_seg.loc[touch_idx]),
        "Bars Since Touch": int(bars_since),
        "Last Close": _safe_float(segment.iloc[-1]),
    }


def daily_r2_band_proximity(
    symbol: str,
    daily_view_label: str,
    slope_lb: int,
    prox: float,
    band_mult: float = 2.0,
) -> Optional[Dict[str, object]]:
    """
    Computes regression band on last `slope_lb` bars of the daily view.
    Returns signed R2 (r2 * sign(slope)), distance to upper/lower in %,
    and flags whether last close is near upper/lower band (within prox).
    """
    df = fetch_hist(symbol)
    if df.empty:
        return None

    close_full = _coerce_1d_series(df["Close"]).dropna()
    close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_view.empty:
        return None

    y = close_view.tail(int(max(10, slope_lb)))
    if len(y) < 10:
        return None

    reg = regression_with_band(y, band_mult=float(band_mult))
    slope = _safe_float(reg["slope"])
    r2 = _safe_float(reg["r2"])
    sr2 = signed_r2_from_slope(r2, slope)

    upper = _coerce_1d_series(reg["upper"]).dropna()
    lower = _coerce_1d_series(reg["lower"]).dropna()
    fit = _coerce_1d_series(reg["fit"]).dropna()
    if upper.empty or lower.empty or fit.empty:
        return None

    px = _safe_float(y.iloc[-1])
    up = _safe_float(upper.iloc[-1])
    lo = _safe_float(lower.iloc[-1])

    # distance in percent of price
    absdist_upper = abs(px - up) / (abs(px) + 1e-9)
    absdist_lower = abs(px - lo) / (abs(px) + 1e-9)

    near_upper = absdist_upper <= float(prox)
    near_lower = absdist_lower <= float(prox)

    return {
        "Symbol": symbol,
        "R2": sr2,
        "Slope": slope,
        "Price": px,
        "Upper(+2σ)": up,
        "Lower(-2σ)": lo,
        "AbsDist Upper (%)": float(absdist_upper) * 100.0,
        "AbsDist Lower (%)": float(absdist_lower) * 100.0,
        "Near Upper": bool(near_upper),
        "Near Lower": bool(near_lower),
        "AsOf": y.index[-1],
    }


# -----------------------------
# Ichimoku (Daily)
# -----------------------------
def ichimoku_lines(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52) -> Dict[str, pd.Series]:
    """
    Computes Ichimoku lines (Tenkan, Kijun, Senkou A/B) on df with High/Low/Close.
    """
    if df is None or df.empty:
        return {"tenkan": pd.Series(dtype=float), "kijun": pd.Series(dtype=float),
                "senkou_a": pd.Series(dtype=float), "senkou_b": pd.Series(dtype=float)}
    high = _coerce_1d_series(df.get("High", pd.Series(dtype=float))).astype(float)
    low = _coerce_1d_series(df.get("Low", pd.Series(dtype=float))).astype(float)

    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2.0
    kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(base)
    senkou_b = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0).shift(base)

    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b}


def last_daily_kijun_cross_up(
    symbol: str,
    daily_view_label: str,
    slope_lb: int,
    conv: int,
    base: int,
    span_b: int,
    within_last_n_bars: int = 5,
    heading_confirm_bars: int = 1,
) -> Optional[Dict[str, object]]:
    """
    Finds the most recent daily cross where Close crosses ABOVE Kijun, and is heading up.
    Returns None if no qualifying cross in range / within_last_n_bars.

    Output fields include: Symbol, Slope, R2, Cross Date, Bars Since Cross,
    Price@Cross, Kijun@Cross, Last Close.
    """
    df = fetch_hist(symbol)
    if df.empty:
        return None

    df = df.copy()
    close_full = _coerce_1d_series(df["Close"]).dropna()
    close_view = subset_by_daily_view(close_full, daily_view_label).dropna()
    if close_view.empty or len(close_view) < max(30, base + 5):
        return None

    # Align df to view index
    df_view = df.loc[close_view.index].copy()
    lines = ichimoku_lines(df_view, conv=int(conv), base=int(base), span_b=int(span_b))
    kijun = _coerce_1d_series(lines["kijun"]).dropna()
    if kijun.empty:
        return None

    close = close_view.loc[kijun.index].dropna()
    kijun = kijun.loc[close.index].dropna()
    if close.empty or len(close) < 5:
        return None

    cross = (close.shift(1) <= kijun.shift(1)) & (close > kijun)
    if not cross.any():
        return None

    cross_dates = cross[cross].index
    cross_date = cross_dates[-1]
    bars_since = int(len(close) - 1 - close.index.get_loc(cross_date))

    if int(within_last_n_bars) > 0 and bars_since > int(within_last_n_bars):
        return None

    # Heading up confirmation: after cross, last N diffs > 0
    post = close.loc[cross_date:]
    if not _series_heading_up(post, confirm_bars=int(heading_confirm_bars)):
        return None

    # Slope + R2 on last slope_lb bars (within view)
    y = close.tail(int(max(10, slope_lb)))
    reg = regression_with_band(y, band_mult=2.0)
    slope = _safe_float(reg["slope"])
    r2 = _safe_float(reg["r2"])

    return {
        "Symbol": symbol,
        "Slope": slope,
        "R2": r2,
        "Cross Date": cross_date,
        "Bars Since Cross": int(bars_since),
        "Price@Cross": _safe_float(close.loc[cross_date]),
        "Kijun@Cross": _safe_float(kijun.loc[cross_date]),
        "Last Close": _safe_float(close.iloc[-1]),
        "AsOf": close.index[-1],
    }
# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Stock Wizard - Bull/Bear", layout="wide")

st.title("Stock Wizard — Bull/Bear Dashboard")

if yf is None:
    st.error("Missing dependency: yfinance. Install it (pip install yfinance) and restart.")
    st.stop()

mode = "main"  # used to keep widget keys stable if you later add modes

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

symbols_text = st.sidebar.text_area(
    "Universe (comma/space separated tickers)",
    value="SPY, QQQ, AAPL, MSFT, NVDA, TSLA, AMZN, META, AMD, NFLX",
    height=110,
    key=f"universe_{mode}",
)

def parse_universe(txt: str) -> List[str]:
    if not txt:
        return []
    raw = txt.replace("\n", " ").replace(",", " ").split()
    out = []
    for s in raw:
        s = s.strip().upper()
        if s and s not in out:
            out.append(s)
    return out

universe = parse_universe(symbols_text)
if not universe:
    st.warning("Universe is empty — add at least one symbol in the sidebar.")
    st.stop()

daily_view = st.sidebar.selectbox(
    "Daily view range",
    ["1M", "3M", "6M", "1Y", "2Y", "5Y", "MAX"],
    index=2,
    key=f"daily_view_{mode}",
)

# Indicators settings
st.sidebar.subheader("Bollinger Bands")
bb_win = st.sidebar.number_input("BB window", min_value=5, max_value=200, value=20, step=1, key=f"bb_win_{mode}")
bb_mult = st.sidebar.number_input("BB σ multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.1, key=f"bb_mult_{mode}")
bb_use_ema = st.sidebar.checkbox("Use EMA for BB mid/std", value=False, key=f"bb_ema_{mode}")

st.sidebar.subheader("Regression / Fit")
slope_lb_daily = st.sidebar.number_input("Daily slope lookback (bars)", min_value=10, max_value=2000, value=126, step=1, key=f"slope_lb_daily_{mode}")
slope_lb_hourly = st.sidebar.number_input("Hourly slope lookback (bars)", min_value=10, max_value=400, value=60, step=1, key=f"slope_lb_hourly_{mode}")

st.sidebar.subheader("Support/Resistance")
sr_lb_daily = st.sidebar.number_input("Daily S/R lookback (bars)", min_value=5, max_value=500, value=50, step=1, key=f"sr_lb_daily_{mode}")
sr_prox_pct = st.sidebar.number_input("S/R proximity (%)", min_value=0.05, max_value=10.0, value=0.50, step=0.05, key=f"sr_prox_{mode}") / 100.0

st.sidebar.subheader("Reversal Confirmation")
rev_bars_confirm = st.sidebar.number_input("Heading-up confirmation bars", min_value=1, max_value=10, value=2, step=1, key=f"rev_confirm_{mode}")
rev_horizon = st.sidebar.number_input("Support-touch horizon (bars)", min_value=3, max_value=200, value=30, step=1, key=f"rev_hz_{mode}")

st.sidebar.subheader("NTD / NPX")
ntd_window = st.sidebar.number_input("NTD/NPX window", min_value=10, max_value=300, value=50, step=1, key=f"ntd_win_{mode}")

st.sidebar.subheader("Ichimoku")
ichi_conv = st.sidebar.number_input("Tenkan (conv) period", min_value=3, max_value=50, value=9, step=1, key=f"ichi_conv_{mode}")
ichi_base = st.sidebar.number_input("Kijun (base) period", min_value=5, max_value=100, value=26, step=1, key=f"ichi_base_{mode}")
ichi_spanb = st.sidebar.number_input("Senkou B period", min_value=10, max_value=200, value=52, step=1, key=f"ichi_spanb_{mode}")

# -----------------------------
# Tabs
# -----------------------------
tab_names = [
    "1) Price Chart (Daily)",
    "2) NTD Chart (Intraday)",
    "3) Quick Stats",
    "4) Universe Table",
    "5) BB Summary",
    "6) Regression Summary",
    "7) S/R Touches",
    "8) Support Reversals",
    "9) NBB -0.8 to -1.0 Lists",
    "10) Slope Direction Scan",
    "11) Trendline Direction Lists",
    "12) NTD Hot List",
    "13) NTD NPX 0.0–0.2 Scanner",
    "14) Uptrend vs Downtrend",
    "15) Ichimoku Kijun Scanner",
    "16) R² > 45% Daily/Hourly",
    "17) R² < 45% Daily/Hourly",
    "18) R² Sign ±2σ Proximity (Daily)",
    "19) NTD Chart — NPX 0.0–0.2 (Intraday)",
]
tabs = st.tabs(tab_names)

(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10,
    tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19
) = tabs

# -----------------------------
# TAB 1: Daily Price Chart
# -----------------------------
with tab1:
    st.header("Daily Price Chart")
    sym = st.selectbox("Symbol", universe, index=0, key=f"sym_daily_{mode}")

    df = fetch_hist(sym)
    if df.empty:
        st.error("No daily data returned.")
    else:
        close_full = _coerce_1d_series(df["Close"]).dropna()
        close = subset_by_daily_view(close_full, daily_view).dropna()
        df_view = df.loc[close.index].copy()

        mid, upper, lower, pctb, nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
        support, resistance = sr_levels(close, lookback=int(sr_lb_daily))
        reg = regression_with_band(close.tail(int(max(10, slope_lb_daily))), band_mult=2.0)

        # Simple table preview (fast and dependency-free)
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Last Close", f"{_safe_float(close.iloc[-1]):.2f}")
        colB.metric("NBB (last)", f"{_safe_float(nbb.dropna().iloc[-1]) if not nbb.dropna().empty else np.nan:.3f}")
        colC.metric("Slope (view)", f"{daily_global_slope(sym, daily_view)[0]:.6f}")
        colD.metric("R² (tail)", f"{_safe_float(reg['r2']):.3f}")

        st.subheader("Latest Rows")
        st.dataframe(df_view.tail(20), use_container_width=True)

# -----------------------------
# TAB 2: Intraday NTD Chart
# -----------------------------
with tab2:
    st.header("Intraday NTD / NPX")
    sym = st.selectbox("Symbol", universe, index=0, key=f"sym_intraday_{mode}")
    period = st.selectbox("Intraday period", ["1d", "2d", "5d", "1mo"], index=0, key=f"intra_period_{mode}")

    df_i = fetch_intraday(sym, period=period, interval="60m")
    if df_i.empty:
        st.error("No intraday data returned.")
    else:
        close = _coerce_1d_series(df_i["Close"]).dropna()
        npx = compute_normalized_price(close, window=int(ntd_window))
        ntd = compute_normalized_trend(close, window=int(ntd_window))

        c1, c2, c3 = st.columns(3)
        c1.metric("Last Close", f"{_safe_float(close.iloc[-1]):.2f}")
        c2.metric("NPX (last)", f"{_safe_float(npx.dropna().iloc[-1]) if not npx.dropna().empty else np.nan:.3f}")
        c3.metric("NTD (last)", f"{_safe_float(ntd.dropna().iloc[-1]) if not ntd.dropna().empty else np.nan:.6f}")

        st.subheader("Latest Rows")
        out = df_i.copy()
        out["NPX"] = npx
        out["NTD"] = ntd
        st.dataframe(out.tail(50), use_container_width=True)

# -----------------------------
# TAB 3: Quick Stats
# -----------------------------
with tab3:
    st.header("Quick Stats")
    sym = st.selectbox("Symbol", universe, index=0, key=f"sym_stats_{mode}")
    df = fetch_hist(sym)
    if df.empty:
        st.error("No data.")
    else:
        close = subset_by_daily_view(df["Close"], daily_view).dropna()
        mid, upper, lower, pctb, nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
        r2, slope, ts = daily_regression_r2(sym, daily_view, slope_lb=int(slope_lb_daily))

        st.write(
            {
                "Symbol": sym,
                "AsOf": str(close.index[-1]) if not close.empty else None,
                "Close": _safe_float(close.iloc[-1]) if not close.empty else np.nan,
                "Slope (tail)": slope,
                "R2 (tail)": r2,
                "NBB (last)": _safe_float(nbb.dropna().iloc[-1]) if not nbb.dropna().empty else np.nan,
                "PctB (last)": _safe_float(pctb.dropna().iloc[-1]) if not pctb.dropna().empty else np.nan,
            }
        )

# -----------------------------
# TAB 4: Universe Table
# -----------------------------
with tab4:
    st.header("Universe Table (Daily view)")
    rows = []
    for sym in universe:
        m, r2, ts = daily_global_slope(sym, daily_view)
        npx_last, _ = daily_last_npx_in_view(sym, daily_view, ntd_win=int(ntd_window))
        rows.append({"Symbol": sym, "Slope(view)": m, "R2(view)": r2, "NPX(last)": npx_last, "AsOf": ts})
    out = pd.DataFrame(rows)
    st.dataframe(out.sort_values(["Slope(view)"], ascending=False).reset_index(drop=True), use_container_width=True)

# -----------------------------
# TAB 5: BB Summary
# -----------------------------
with tab5:
    st.header("Bollinger Summary")
    rows = []
    for sym in universe:
        df = fetch_hist(sym)
        if df.empty:
            continue
        close = subset_by_daily_view(df["Close"], daily_view).dropna()
        if close.empty:
            continue
        _, _, _, pctb, nbb = compute_bbands(close, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
        if nbb.dropna().empty:
            continue
        rows.append({"Symbol": sym, "NBB": _safe_float(nbb.dropna().iloc[-1]), "%B": _safe_float(pctb.dropna().iloc[-1])})
    st.dataframe(pd.DataFrame(rows).sort_values(["NBB"]).reset_index(drop=True), use_container_width=True)

# -----------------------------
# TAB 6: Regression Summary
# -----------------------------
with tab6:
    st.header("Regression Summary (Daily)")
    rows = []
    for sym in universe:
        r2, slope, ts = daily_regression_r2(sym, daily_view, slope_lb=int(slope_lb_daily))
        rows.append({"Symbol": sym, "Slope": slope, "R2": r2, "AsOf": ts})
    st.dataframe(pd.DataFrame(rows).sort_values(["R2"], ascending=False).reset_index(drop=True), use_container_width=True)

# -----------------------------
# TAB 7: S/R Touches
# -----------------------------
with tab7:
    st.header("Support/Resistance Snapshot")
    sym = st.selectbox("Symbol", universe, index=0, key=f"sym_sr_{mode}")
    df = fetch_hist(sym)
    if df.empty:
        st.error("No data.")
    else:
        close = subset_by_daily_view(df["Close"], daily_view).dropna()
        support, resistance = sr_levels(close, lookback=int(sr_lb_daily))
        st.write(
            {
                "Last Close": _safe_float(close.iloc[-1]),
                "Support (rolling min)": _safe_float(support.dropna().iloc[-1]) if not support.dropna().empty else np.nan,
                "Resistance (rolling max)": _safe_float(resistance.dropna().iloc[-1]) if not resistance.dropna().empty else np.nan,
                "Proximity %": float(sr_prox_pct) * 100.0,
            }
        )

# -----------------------------
# TAB 8: Support Reversals
# -----------------------------
with tab8:
    st.header("Support Reversal Scanner (Daily)")
    run_rev = st.button("Run Support Reversal Scan", key=f"tab8_run_rev_{mode}")
    if run_rev:
        rows = []
        for sym in universe:
            r = daily_support_reversal_heading_up(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=int(sr_lb_daily),
                prox=float(sr_prox_pct),
                bars_confirm=int(rev_bars_confirm),
                horizon=int(rev_horizon),
            )
            if r is not None:
                rows.append(r)
        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Bars Since Touch"], ascending=True)
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 9: NBB -0.8 to -1.0 Lists
# ---------------------------
with tab9:
    st.header("NBB -0.8 to -1.0 Lists")
    st.caption(
        "Creates two lists (Daily, using selected Daily view range):\n"
        "(a) Regression > 0 AND NBB between -0.8 and -1.0\n"
        "(b) Regression < 0 AND NBB between -0.8 and -1.0"
    )

    c1, c2, c3 = st.columns(3)
    nbb_lo = c1.number_input("NBB lower bound", value=-1.0, step=0.05, format="%.2f", key=f"tab9_nbb_lo_{mode}")
    nbb_hi = c2.number_input("NBB upper bound", value=-0.8, step=0.05, format="%.2f", key=f"tab9_nbb_hi_{mode}")
    run_nbb = c3.button("Run NBB Scan", key=f"tab9_btn_run_nbb_{mode}")

    @st.cache_data(ttl=120, show_spinner=False)
    def _daily_last_nbb_in_view(symbol: str, daily_view_label: str, bb_window: int, bb_mult_: float, bb_use_ema_: bool):
        close_full = _coerce_1d_series(fetch_hist(symbol).get("Close", pd.Series(dtype=float))).dropna()
        if close_full.empty:
            return np.nan, np.nan, np.nan, None
        close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
        if close_show.empty or len(close_show) < max(5, int(bb_window)):
            return np.nan, np.nan, np.nan, close_show.index[-1] if len(close_show) else None

        _, _, _, pctb, nbb = compute_bbands(close_show, window=int(bb_window), mult=float(bb_mult_), use_ema=bool(bb_use_ema_))
        nbb_s = _coerce_1d_series(nbb).dropna()
        pctb_s = _coerce_1d_series(pctb).dropna()
        if nbb_s.empty:
            return np.nan, np.nan, _safe_float(close_show.iloc[-1]), close_show.index[-1]

        nbb_last = _safe_float(nbb_s.iloc[-1])
        pctb_last = _safe_float(pctb_s.iloc[-1]) if not pctb_s.empty else np.nan
        px_last = _safe_float(close_show.iloc[-1])
        return nbb_last, pctb_last, px_last, close_show.index[-1]

    if run_nbb:
        rows_pos, rows_neg = [], []
        lo_b = float(min(nbb_lo, nbb_hi))
        hi_b = float(max(nbb_lo, nbb_hi))

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            nbb_last, pctb_last, px_last, asof = _daily_last_nbb_in_view(
                sym, daily_view_label=daily_view,
                bb_window=int(bb_win),
                bb_mult_=float(bb_mult),
                bb_use_ema_=bool(bb_use_ema)
            )
            if not np.isfinite(nbb_last):
                continue

            if not (lo_b <= float(nbb_last) <= hi_b):
                continue

            row = {
                "Symbol": sym,
                "Slope": float(m),
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "NBB": float(nbb_last),
                "%B": float(pctb_last) if np.isfinite(pctb_last) else np.nan,
                "Price": float(px_last) if np.isfinite(px_last) else np.nan,
                "AsOf": asof if asof is not None else ts
            }

            if float(m) > 0.0:
                rows_pos.append(row)
            elif float(m) < 0.0:
                rows_neg.append(row)

        left, right = st.columns(2)
        with left:
            st.subheader("(a) Regression > 0 and NBB -0.8 to -1.0")
            if not rows_pos:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_pos).sort_values(["NBB", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Regression < 0 and NBB -0.8 to -1.0")
            if not rows_neg:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_neg).sort_values(["NBB", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 10: Slope Direction Scan
# ---------------------------
with tab10:
    st.header("Slope Direction Scan")
    st.caption("Lists symbols whose current DAILY global slope is up vs down (based on selected Daily view).")

    run_slope = st.button("Run Slope Direction Scan", key=f"tab10_btn_run_slope_dir_{mode}")

    if run_slope:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue
            rows.append({"Symbol": sym, "Slope": float(m), "R2": float(r2) if np.isfinite(r2) else np.nan, "AsOf": ts})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            up = out[out["Slope"] > 0].sort_values(["Slope"], ascending=False)
            dn = out[out["Slope"] < 0].sort_values(["Slope"], ascending=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Upward Slope")
                st.dataframe(up.reset_index(drop=True), use_container_width=True) if not up.empty else st.info("No matches.")
            with right:
                st.subheader("Downward Slope")
                st.dataframe(dn.reset_index(drop=True), use_container_width=True) if not dn.empty else st.info("No matches.")
# ---------------------------
# TAB 11: Trendline Direction Lists
# ---------------------------
with tab11:
    st.header("Trendline Direction Lists")
    st.caption(
        "Uses the selected Daily view range.\n"
        "Upward Trend list: Slope >= 0 and NPX < 0.0\n"
        "Downward Trend list: Slope < 0 and NPX > 0.5"
    )

    run_trend_lists = st.button("Run Trendline Direction Lists", key=f"tab11_btn_run_trendline_lists_{mode}")

    if run_trend_lists:
        up_rows, dn_rows = [], []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=int(ntd_window))
            if not np.isfinite(npx_last):
                continue

            if float(m) >= 0.0 and float(npx_last) < 0.0:
                up_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})
            elif float(m) < 0.0 and float(npx_last) > 0.5:
                dn_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})

        left, right = st.columns(2)
        with left:
            st.subheader("Upward Trend List")
            if not up_rows:
                st.info("No matches.")
            else:
                out_up = pd.DataFrame(up_rows).sort_values(["Symbol"], ascending=True)
                st.dataframe(out_up.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Downward Trend List")
            if not dn_rows:
                st.info("No matches.")
            else:
                out_dn = pd.DataFrame(dn_rows).sort_values(["Symbol"], ascending=True)
                st.dataframe(out_dn.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run Trendline Direction Lists** to scan the current universe.")

# ---------------------------
# TAB 12: NTD Hot List
# ---------------------------
with tab12:
    st.header("NTD Hot List")
    st.caption("Slope > 0 and NPX between 0.0 and 0.5 (inclusive) using selected Daily view.")

    run_hot = st.button("Run NTD Hot List", key=f"tab12_btn_run_ntd_hot_{mode}")

    if run_hot:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m) or float(m) <= 0.0:
                continue

            npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=int(ntd_window))
            if not np.isfinite(npx_last):
                continue

            if 0.0 <= float(npx_last) <= 0.5:
                rows.append({"Symbol": sym, "Slope": float(m), "NPX (Norm Price)": float(npx_last), "R2": float(r2), "AsOf": ts})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Slope", "NPX (Norm Price)"], ascending=[False, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Click **Run NTD Hot List** to scan the current universe.")

# ---------------------------
# TAB 13: NTD NPX 0.0–0.2 Scanner
# ---------------------------
with tab13:
    st.header("NTD NPX 0.0–0.2 Scanner")
    st.caption(
        "Scans for symbols where DAILY NPX is between 0.0 and 0.2 and is heading up.\n"
        "Split by slope sign (List 1 slope>0, List 2 slope<0)."
    )

    c1, c2 = st.columns(2)
    npx_up_bars = c1.slider("Heading-up confirmation (consecutive bars)", 1, 5, 1, 1, key=f"tab13_npx_up_{mode}")
    run_npx02 = c2.button("Run NTD NPX 0.0–0.2 Scan", key=f"tab13_btn_run_npx02_{mode}")

    if run_npx02:
        rows_up_slope, rows_dn_slope = [], []

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            npx_s = daily_npx_series_in_view(sym, daily_view_label=daily_view, ntd_win=int(ntd_window))
            npx_s = _coerce_1d_series(npx_s).dropna()
            if npx_s.empty or len(npx_s) < 2:
                continue

            npx_last = _safe_float(npx_s.iloc[-1])
            if not np.isfinite(npx_last):
                continue

            if not (0.0 <= float(npx_last) <= 0.2):
                continue

            if not _series_heading_up(npx_s, confirm_bars=int(npx_up_bars)):
                continue

            row = {"Symbol": sym, "Slope": float(m), "R2": float(r2), "NPX (Norm Price)": float(npx_last), "AsOf": ts}

            if float(m) > 0.0:
                rows_up_slope.append(row)
            elif float(m) < 0.0:
                rows_dn_slope.append(row)

        left, right = st.columns(2)
        with left:
            st.subheader("List 1 — Slope > 0 and NPX 0.0–0.2 heading up")
            if not rows_up_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_up_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("List 2 — Slope < 0 and NPX 0.0–0.2 heading up")
            if not rows_dn_slope:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_dn_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 14: Uptrend vs Downtrend
# ---------------------------
with tab14:
    st.header("Uptrend vs Downtrend")
    st.caption("Support reversal heading up (Daily) split by slope sign.")

    c1, c2 = st.columns(2)
    hz_sr = c1.slider("Support-touch horizon (bars)", 3, 200, int(max(3, rev_horizon)), 1, key=f"tab14_hz_{mode}")
    run_ud = c2.button("Run Uptrend vs Downtrend Scan", key=f"tab14_btn_run_ud_{mode}")

    if run_ud:
        rows_uptrend, rows_downtrend = [], []

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            rev = daily_support_reversal_heading_up(
                symbol=sym,
                daily_view_label=daily_view,
                sr_lb=int(sr_lb_daily),
                prox=float(sr_prox_pct),
                bars_confirm=int(rev_bars_confirm),
                horizon=int(hz_sr),
            )
            if rev is None:
                continue

            row = dict(rev)
            row["Slope"] = float(m)
            row["R2"] = float(r2) if np.isfinite(r2) else np.nan
            row["AsOf"] = ts

            if float(m) > 0.0:
                rows_uptrend.append(row)
            elif float(m) < 0.0:
                rows_downtrend.append(row)

        left, right = st.columns(2)
        with left:
            st.subheader("(a) Uptrend — Slope > 0 and Support Reversal heading up")
            if not rows_uptrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_uptrend).sort_values(["Bars Since Touch", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Downtrend — Slope < 0 and Support Reversal heading up")
            if not rows_downtrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_downtrend).sort_values(["Bars Since Touch", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 15: Ichimoku Kijun Scanner (FIXED: function exists in Batch 1)
# ---------------------------
with tab15:
    st.header("Ichimoku Kijun Scanner")
    st.caption(
        "Daily-only scanner:\n"
        "List 1: Slope > 0 AND crossed above Kijun heading up\n"
        "List 2: Slope < 0 AND crossed above Kijun heading up\n"
        "Includes Price@Cross, Kijun@Cross, Bars Since Cross, R²."
    )

    c1, c2 = st.columns(2)
    kijun_within = c1.slider("Cross must be within last N bars", 0, 60, 5, 1, key=f"tab15_kijun_within_{mode}")
    run_kijun = c2.button("Run Ichimoku Kijun Scan", key=f"tab15_btn_run_kijun_{mode}")

    if run_kijun:
        rows_list1, rows_list2 = [], []
        for sym in universe:
            r = last_daily_kijun_cross_up(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                conv=int(ichi_conv),
                base=int(ichi_base),
                span_b=int(ichi_spanb),
                within_last_n_bars=int(kijun_within),
                heading_confirm_bars=int(rev_bars_confirm),
            )
            if r is None:
                continue

            m = _safe_float(r.get("Slope", np.nan))
            if not np.isfinite(m):
                continue

            if m > 0.0:
                rows_list1.append(r)
            elif m < 0.0:
                rows_list2.append(r)

        left, right = st.columns(2)
        with left:
            st.subheader("List 1 — Slope > 0 and Kijun Cross-Up")
            if not rows_list1:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list1)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("List 2 — Slope < 0 and Kijun Cross-Up")
            if not rows_list2:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_list2)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

# ---------------------------
# TAB 16: R² > 45% Daily/Hourly
# ---------------------------
with tab16:
    st.header("R² > 45% Daily/Hourly")
    st.caption("Uses plain R² (0..1). Daily uses selected Daily view. Hourly uses intraday period.")

    c1, c2, c3 = st.columns(3)
    r2_thr = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key=f"tab16_r2_thr_{mode}")
    hour_period = c2.selectbox("Hourly intraday period", ["1d", "2d", "5d", "1mo"], index=0, key=f"tab16_hour_period_{mode}")
    run_r2 = c3.button("Run R² Scan", key=f"tab16_run_{mode}")

    if run_r2:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, daily_view_label=daily_view, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2_d) and float(r2_d) > float(r2_thr):
                daily_rows.append({"Symbol": sym, "R2": float(r2_d), "Slope": float(m_d), "AsOf": ts_d})

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) > float(r2_thr):
                hourly_rows.append({"Symbol": sym, "R2": float(r2_h), "Slope": float(m_h), "AsOf": ts_h, "Period": str(hour_period)})

        left, right = st.columns(2)
        with left:
            st.subheader("Daily — R² > threshold")
            st.dataframe(pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[False, False]).reset_index(drop=True), use_container_width=True) if daily_rows else st.info("No matches.")
        with right:
            st.subheader(f"Hourly ({hour_period}) — R² > threshold")
            st.dataframe(pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[False, False]).reset_index(drop=True), use_container_width=True) if hourly_rows else st.info("No matches.")

# ---------------------------
# TAB 17: R² < 45% Daily/Hourly
# ---------------------------
with tab17:
    st.header("R² < 45% Daily/Hourly")
    st.caption("Uses plain R² (0..1). Daily uses selected Daily view. Hourly uses intraday period.")

    c1, c2, c3 = st.columns(3)
    r2_thr_lo = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key=f"tab17_r2_thr_{mode}")
    hour_period_lo = c2.selectbox("Hourly intraday period", ["1d", "2d", "5d", "1mo"], index=0, key=f"tab17_hour_period_{mode}")
    run_r2_lo = c3.button("Run R² Low Scan", key=f"tab17_run_{mode}")

    if run_r2_lo:
        daily_rows, hourly_rows = [], []

        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, daily_view_label=daily_view, slope_lb=int(slope_lb_daily))
            if np.isfinite(r2_d) and float(r2_d) < float(r2_thr_lo):
                daily_rows.append({"Symbol": sym, "R2": float(r2_d), "Slope": float(m_d), "AsOf": ts_d})

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period_lo), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) < float(r2_thr_lo):
                hourly_rows.append({"Symbol": sym, "R2": float(r2_h), "Slope": float(m_h), "AsOf": ts_h, "Period": str(hour_period_lo)})

        left, right = st.columns(2)
        with left:
            st.subheader("Daily — R² < threshold")
            st.dataframe(pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[True, True]).reset_index(drop=True), use_container_width=True) if daily_rows else st.info("No matches.")
        with right:
            st.subheader(f"Hourly ({hour_period_lo}) — R² < threshold")
            st.dataframe(pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[True, True]).reset_index(drop=True), use_container_width=True) if hourly_rows else st.info("No matches.")

# ---------------------------
# TAB 18: R² Sign ±2σ Proximity (Daily)
# ---------------------------
with tab18:
    st.header("R² Sign ±2σ Proximity (Daily)")
    st.caption(
        "Uses SIGNED R² = (R² * sign(slope)) so it can be positive or negative.\n"
        "Creates 4 lists by sign and proximity to ±2σ regression band."
    )

    run_band_scan = st.button("Run R² Sign ±2σ Proximity Scan (Daily)", key=f"tab18_run_{mode}")

    if run_band_scan:
        rows_pos_lower, rows_pos_upper = [], []
        rows_neg_lower, rows_neg_upper = [], []

        for sym in universe:
            r = daily_r2_band_proximity(
                symbol=sym,
                daily_view_label=daily_view,
                slope_lb=int(slope_lb_daily),
                prox=float(sr_prox_pct),
                band_mult=2.0
            )
            if r is None:
                continue

            r2v = r.get("R2", np.nan)
            if not np.isfinite(r2v):
                continue

            near_lo = bool(r.get("Near Lower", False))
            near_up = bool(r.get("Near Upper", False))
            row = {k: v for k, v in r.items() if k not in ("Near Lower", "Near Upper")}

            if float(r2v) > 0.0:
                if near_lo:
                    rows_pos_lower.append(row)
                if near_up:
                    rows_pos_upper.append(row)
            elif float(r2v) < 0.0:
                if near_lo:
                    rows_neg_lower.append(row)
                if near_up:
                    rows_neg_upper.append(row)

        st.info(f"Near threshold = ±{sr_prox_pct*100:.3f}% (from sidebar S/R proximity %)")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Signed R² > 0  •  Near Lower -2σ")
            st.dataframe(pd.DataFrame(rows_pos_lower).sort_values(["AbsDist Lower (%)"], ascending=True).reset_index(drop=True), use_container_width=True) if rows_pos_lower else st.info("No matches.")
        with c2:
            st.subheader("Signed R² > 0  •  Near Upper +2σ")
            st.dataframe(pd.DataFrame(rows_pos_upper).sort_values(["AbsDist Upper (%)"], ascending=True).reset_index(drop=True), use_container_width=True) if rows_pos_upper else st.info("No matches.")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Signed R² < 0  •  Near Lower -2σ")
            st.dataframe(pd.DataFrame(rows_neg_lower).sort_values(["AbsDist Lower (%)"], ascending=True).reset_index(drop=True), use_container_width=True) if rows_neg_lower else st.info("No matches.")
        with c4:
            st.subheader("Signed R² < 0  •  Near Upper +2σ")
            st.dataframe(pd.DataFrame(rows_neg_upper).sort_values(["AbsDist Upper (%)"], ascending=True).reset_index(drop=True), use_container_width=True) if rows_neg_upper else st.info("No matches.")

# ---------------------------
# TAB 19: Intraday NPX 0.0–0.2 Scanner
# ---------------------------
with tab19:
    st.header("NTD Chart — NPX 0.0–0.2 (Intraday)")
    st.caption("Lists symbols where the latest intraday NPX is between 0.0 and 0.2 using hourly intraday data.")

    c1, c2, c3 = st.columns(3)
    hour_period_npx = c1.selectbox("Intraday period", ["1d", "2d", "5d", "1mo"], index=0, key=f"tab19_period_{mode}")
    npx_lo_i = c2.number_input("NPX lower bound", value=0.0, step=0.05, format="%.2f", key=f"tab19_npx_lo_{mode}")
    npx_hi_i = c3.number_input("NPX upper bound", value=0.2, step=0.05, format="%.2f", key=f"tab19_npx_hi_{mode}")

    run_tab19 = st.button("Run NPX 0.0–0.2 (Intraday) Scan", key=f"tab19_run_{mode}")

    if run_tab19:
        rows = []
        lo_b = float(min(npx_lo_i, npx_hi_i))
        hi_b = float(max(npx_lo_i, npx_hi_i))

        for sym in universe:
            df_i = fetch_intraday(sym, period=str(hour_period_npx), interval="60m")
            if df_i.empty or "Close" not in df_i.columns:
                continue

            close = _coerce_1d_series(df_i["Close"]).ffill().dropna()
            if close.empty:
                continue

            npx = compute_normalized_price(close, window=int(ntd_window)).dropna()
            ntd = compute_normalized_trend(close, window=int(ntd_window)).dropna()

            if npx.empty:
                continue

            npx_last = _safe_float(npx.iloc[-1])
            if not np.isfinite(npx_last):
                continue

            if not (lo_b <= npx_last <= hi_b):
                continue

            ntd_last = _safe_float(ntd.iloc[-1]) if not ntd.empty else np.nan

            rows.append({"Symbol": sym, "NPX (last)": npx_last, "NTD (last)": ntd_last, "Period": str(hour_period_npx)})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["NPX (last)"], ascending=True)
            st.dataframe(out.reset_index(drop=True), use_container_width=True)

st.caption("Done.")
