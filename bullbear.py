# bullbear.py
# Stock Wizard (Streamlit) — NEWS REMOVED (no yfinance news / no fetch_yf_news calls)

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

try:
    import pytz
    PACIFIC = pytz.timezone("US/Pacific")
except Exception:
    PACIFIC = None


# -----------------------------
# Utilities
# -----------------------------
def _coerce_1d_series(x) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0].copy()
    return pd.Series(x)

def _ensure_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Handle yfinance returning multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        # if one ticker, take first level
        if len(df.columns.levels) >= 2:
            df.columns = df.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[cols].copy()
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    return df

def _to_pacific_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if PACIFIC is None:
        return df
    idx = df.index
    try:
        if idx.tz is None:
            df.index = idx.tz_localize(PACIFIC)
        else:
            df.index = idx.tz_convert(PACIFIC)
    except Exception:
        # best effort
        pass
    return df

def fmt_price_val(x) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):,.2f}"
    except Exception:
        return "n/a"

def _has_volume_to_plot(vol: pd.Series) -> bool:
    v = _coerce_1d_series(vol).dropna()
    if v.empty:
        return False
    try:
        return float(v.sum()) > 0.0
    except Exception:
        return False

def style_axes(ax):
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=0)
    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def annotate_crossover(ax, t, y, side: str, note: str = ""):
    txt = f"{side}"
    if note:
        txt += f"\n{note}"
    ax.annotate(
        txt,
        xy=(t, y),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        arrowprops=dict(arrowstyle="->", alpha=0.55),
    )


# -----------------------------
# Data fetchers (cached)
# -----------------------------
@st.cache_data(ttl=300)
def fetch_hist_ohlcv(ticker: str, start: str = "2018-01-01") -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=pd.to_datetime("today"),
        progress=False,
        auto_adjust=False,
    )
    df = _ensure_ohlcv_cols(df).dropna()
    df = _to_pacific_index(df)
    return df

@st.cache_data(ttl=120)
def fetch_daily_range(ticker: str, start: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=pd.to_datetime("today"),
        progress=False,
        auto_adjust=False,
    )
    df = _ensure_ohlcv_cols(df).dropna()
    df = _to_pacific_index(df)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d", interval: str = "60m") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
    )
    df = _ensure_ohlcv_cols(df).dropna()
    df = _to_pacific_index(df)
    return df


# -----------------------------
# Indicators
# -----------------------------
def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    c = _coerce_1d_series(close).astype(float)
    if c.dropna().empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty, empty
    window = int(max(2, window))
    minp = max(2, window // 2)

    if use_ema:
        mid = c.ewm(span=window, adjust=False).mean()
        # approximate rolling std using rolling std (still ok)
        sd = c.rolling(window, min_periods=minp).std()
    else:
        mid = c.rolling(window, min_periods=minp).mean()
        sd = c.rolling(window, min_periods=minp).std()

    up = mid + float(mult) * sd
    lo = mid - float(mult) * sd
    pctb = (c - lo) / (up - lo)
    nbb = (c - mid) / (sd.replace(0, np.nan))
    return mid, up, lo, pctb, nbb


def regression_with_band_lastn(close: pd.Series, lookback: int = 120, mult: float = 2.0):
    """
    Regression on the last N points, return:
      slope, intercept, r2, std, last_fit, last_upper, last_lower
    """
    c = _coerce_1d_series(close).dropna().astype(float)
    if c.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    lookback = int(max(5, lookback))
    c = c.iloc[-lookback:].copy()
    n = len(c)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    x = np.arange(n, dtype=float)
    y = c.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    std = float(np.nanstd(resid, ddof=1)) if n > 2 else np.nan

    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    last_fit = float(yhat[-1])
    last_up = last_fit + float(mult) * std if np.isfinite(std) else np.nan
    last_lo = last_fit - float(mult) * std if np.isfinite(std) else np.nan

    return float(slope), float(intercept), float(r2), float(std), last_fit, last_up, last_lo


def regression_with_band_series(close: pd.Series, mult: float = 2.0):
    """
    Regression on entire series, returns (fit, upper, lower, slope, r2, std).
    """
    c = _coerce_1d_series(close).dropna().astype(float)
    if c.empty or len(c) < 5:
        empty = pd.Series(index=_coerce_1d_series(close).index, dtype=float)
        return empty, empty, empty, np.nan, np.nan, np.nan

    idx = c.index
    n = len(c)
    x = np.arange(n, dtype=float)
    y = c.to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    resid = y - yhat
    std = float(np.nanstd(resid, ddof=1)) if n > 2 else np.nan

    ss_res = float(np.nansum((y - yhat) ** 2))
    ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    fit = pd.Series(yhat, index=idx)
    up = fit + float(mult) * std if np.isfinite(std) else pd.Series(index=idx, dtype=float)
    lo = fit - float(mult) * std if np.isfinite(std) else pd.Series(index=idx, dtype=float)
    return fit, up, lo, float(slope), float(r2), float(std)


def compute_ichimoku(df_ohlc: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52):
    """
    Returns dict of Ichimoku lines (Tenkan, Kijun, SpanA, SpanB).
    """
    if df_ohlc is None or df_ohlc.empty or not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        return {"tenkan": pd.Series(dtype=float), "kijun": pd.Series(dtype=float),
                "spana": pd.Series(dtype=float), "spanb": pd.Series(dtype=float)}

    high = _coerce_1d_series(df_ohlc["High"]).astype(float)
    low = _coerce_1d_series(df_ohlc["Low"]).astype(float)

    conv = int(max(2, conv))
    base = int(max(2, base))
    span_b = int(max(2, span_b))

    tenkan = (high.rolling(conv).max() + low.rolling(conv).min()) / 2.0
    kijun = (high.rolling(base).max() + low.rolling(base).min()) / 2.0
    spana = ((tenkan + kijun) / 2.0).shift(base)
    spanb = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2.0).shift(base)

    return {"tenkan": tenkan, "kijun": kijun, "spana": spana, "spanb": spanb}


def compute_adx(df_ohlc: pd.DataFrame, period: int = 14):
    """
    ADX(14) +DI/-DI using Wilder smoothing (EWMA alpha=1/period).
    Returns (adx, pdi, ndi).
    """
    if df_ohlc is None or df_ohlc.empty or not {"High", "Low", "Close"}.issubset(df_ohlc.columns):
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    high = _coerce_1d_series(df_ohlc["High"]).astype(float)
    low  = _coerce_1d_series(df_ohlc["Low"]).astype(float)
    close= _coerce_1d_series(df_ohlc["Close"]).astype(float)
    idx = close.index

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=idx)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=idx)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    p = max(2, int(period))
    atr = tr.ewm(alpha=1/p, adjust=False).mean().replace(0, np.nan)

    plus_dm_sm  = plus_dm.ewm(alpha=1/p, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1/p, adjust=False).mean()

    pdi = (100.0 * (plus_dm_sm / atr)).replace([np.inf, -np.inf], np.nan)
    ndi = (100.0 * (minus_dm_sm / atr)).replace([np.inf, -np.inf], np.nan)

    den = (pdi + ndi).replace(0, np.nan)
    dx = (100.0 * (pdi - ndi).abs() / den).replace([np.inf, -np.inf], np.nan)

    adx = dx.ewm(alpha=1/p, adjust=False).mean()
    return adx.reindex(idx), pdi.reindex(idx), ndi.reindex(idx)


def volume_confirmation_mask(volume: pd.Series, window: int = 20, mult: float = 1.0):
    v = _coerce_1d_series(volume).astype(float)
    if v.dropna().empty:
        return pd.Series(False, index=v.index), pd.Series(index=v.index, dtype=float)
    window = int(max(2, window))
    minp = max(2, window // 2)
    vma = v.rolling(window, min_periods=minp).mean()
    conf = v > (vma * float(mult))
    return conf.fillna(False).reindex(v.index), vma.reindex(v.index)
def _last_pivot_time(df_ohlc: pd.DataFrame, span: int = 5, kind: str = "low"):
    """
    Finds the most recent pivot low/high using a symmetric window:
      pivot low: Low[i] == min(Low[i-span : i+span])
      pivot high: High[i] == max(High[i-span : i+span])
    Returns timestamp (index value) or None.
    """
    if df_ohlc is None or df_ohlc.empty:
        return None
    span = int(max(1, span))

    if kind.lower().startswith("h"):
        s = _coerce_1d_series(df_ohlc["High"] if "High" in df_ohlc.columns else df_ohlc.get("Close", pd.Series(dtype=float))).dropna()
        if s.empty or len(s) < (2 * span + 1):
            return None
        arr = s.to_numpy(dtype=float)
        pivots = []
        for i in range(span, len(arr) - span):
            w = arr[i - span:i + span + 1]
            if not np.isfinite(arr[i]):
                continue
            if np.isfinite(np.nanmax(w)) and arr[i] == np.nanmax(w):
                pivots.append(s.index[i])
        return pivots[-1] if pivots else None

    s = _coerce_1d_series(df_ohlc["Low"] if "Low" in df_ohlc.columns else df_ohlc.get("Close", pd.Series(dtype=float))).dropna()
    if s.empty or len(s) < (2 * span + 1):
        return None
    arr = s.to_numpy(dtype=float)
    pivots = []
    for i in range(span, len(arr) - span):
        w = arr[i - span:i + span + 1]
        if not np.isfinite(arr[i]):
            continue
        if np.isfinite(np.nanmin(w)) and arr[i] == np.nanmin(w):
            pivots.append(s.index[i])
    return pivots[-1] if pivots else None


def anchored_vwap(df_ohlcv: pd.DataFrame, anchor_time, use_typical_price: bool = True) -> pd.Series:
    """
    Anchored VWAP from anchor_time to end:
      AVWAP = cumsum(price * volume) / cumsum(volume)
    """
    if df_ohlcv is None or df_ohlcv.empty or "Close" not in df_ohlcv.columns:
        return pd.Series(dtype=float)
    if "Volume" not in df_ohlcv.columns:
        return pd.Series(index=df_ohlcv.index, dtype=float)

    vol = _coerce_1d_series(df_ohlcv["Volume"]).astype(float)
    if vol.dropna().empty or float(vol.fillna(0).sum()) == 0.0:
        return pd.Series(index=df_ohlcv.index, dtype=float)

    if use_typical_price and {"High", "Low", "Close"}.issubset(df_ohlcv.columns):
        price = (_coerce_1d_series(df_ohlcv["High"]) + _coerce_1d_series(df_ohlcv["Low"]) + _coerce_1d_series(df_ohlcv["Close"])) / 3.0
    else:
        price = _coerce_1d_series(df_ohlcv["Close"]).astype(float)

    idx = df_ohlcv.index
    if anchor_time is None:
        return pd.Series(index=idx, dtype=float)
    try:
        anchor_loc = int(idx.get_indexer([anchor_time], method="nearest")[0])
    except Exception:
        try:
            anchor_loc = int(idx.get_loc(anchor_time))
        except Exception:
            return pd.Series(index=idx, dtype=float)

    pv = (price * vol).astype(float)
    pv_seg = pv.iloc[anchor_loc:].copy()
    v_seg = vol.iloc[anchor_loc:].copy()

    cum_pv = pv_seg.cumsum()
    cum_v = v_seg.cumsum().replace(0, np.nan)
    av = (cum_pv / cum_v).reindex(idx)

    if anchor_loc > 0:
        av.iloc[:anchor_loc] = np.nan
    return av


def bb_fade_last_signal(close: pd.Series, upper: pd.Series, lower: pd.Series):
    """
    Mean-reversion 'band fade':
      BUY when price was below lower band on prior bar and closes back inside.
      SELL when price was above upper band on prior bar and closes back inside.
    Returns most recent dict or None.
    """
    c = _coerce_1d_series(close)
    u = _coerce_1d_series(upper).reindex(c.index)
    l = _coerce_1d_series(lower).reindex(c.index)
    ok = c.notna() & u.notna() & l.notna()
    if ok.sum() < 2:
        return None
    c = c[ok]; u = u[ok]; l = l[ok]
    inside = (c <= u) & (c >= l)
    buy = inside & (c.shift(1) < l.shift(1))
    sell = inside & (c.shift(1) > u.shift(1))

    tb = buy[buy].index[-1] if buy.any() else None
    ts = sell[sell].index[-1] if sell.any() else None
    if tb is None and ts is None:
        return None
    if ts is None or (tb is not None and tb >= ts):
        return {"side": "BUY", "time": tb, "price": float(c.loc[tb]) if np.isfinite(c.loc[tb]) else np.nan, "note": "BB Fade (mean-rev)"}
    return {"side": "SELL", "time": ts, "price": float(c.loc[ts]) if np.isfinite(c.loc[ts]) else np.nan, "note": "BB Fade (mean-rev)"}


def _series_heading_up(s: pd.Series, confirm_bars: int = 1) -> bool:
    s = _coerce_1d_series(s).dropna().astype(float)
    if s.empty or len(s) < (confirm_bars + 1):
        return False
    confirm_bars = int(max(1, confirm_bars))
    tail = s.iloc[-(confirm_bars + 1):]
    diffs = tail.diff().iloc[1:]
    return bool((diffs > 0).all())


def daily_view_to_start(label: str) -> datetime:
    now = datetime.now(tz=timezone.utc)
    if label == "6M":
        return now - timedelta(days=183)
    if label == "1Y":
        return now - timedelta(days=365)
    if label == "2Y":
        return now - timedelta(days=365 * 2)
    if label == "5Y":
        return now - timedelta(days=365 * 5)
    if label == "YTD":
        return datetime(now.year, 1, 1, tzinfo=timezone.utc)
    # Max
    return now - timedelta(days=365 * 15)


def compute_npx_from_bands(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    c = _coerce_1d_series(close).astype(float)
    u = _coerce_1d_series(upper).reindex(c.index).astype(float)
    l = _coerce_1d_series(lower).reindex(c.index).astype(float)
    den = (u - l).replace(0, np.nan)
    npx = (c - l) / den
    return npx


def daily_global_slope(symbol: str, daily_view_label: str = "1Y") -> Tuple[float, float, Optional[pd.Timestamp]]:
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, None
    close = df["Close"].dropna()
    fit, up, lo, slope, r2, std = regression_with_band_series(close, mult=2.0)
    ts = close.index[-1] if not close.empty else None
    return float(slope), float(r2), ts


def daily_regression_r2(symbol: str, slope_lb: int = 120, daily_view_label: str = "1Y") -> Tuple[float, float, Optional[pd.Timestamp]]:
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, None
    close = df["Close"].dropna()
    m, b, r2, std, fit, up, lo = regression_with_band_lastn(close, lookback=int(slope_lb), mult=2.0)
    ts = close.index[-1] if not close.empty else None
    return float(r2), float(m), ts


def hourly_regression_r2(symbol: str, period: str = "1d", slope_lb: int = 60) -> Tuple[float, float, Optional[pd.Timestamp]]:
    df = fetch_intraday(symbol, period=period, interval="60m")
    if df.empty or "Close" not in df.columns:
        return np.nan, np.nan, None
    close = df["Close"].dropna()
    m, b, r2, std, fit, up, lo = regression_with_band_lastn(close, lookback=int(slope_lb), mult=2.0)
    ts = close.index[-1] if not close.empty else None
    return float(r2), float(m), ts


def daily_npx_series_in_view(symbol: str, daily_view_label: str = "1Y", ntd_win: int = 120) -> pd.Series:
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["Close"].dropna()

    # Use entire view regression bands (fast + stable)
    fit, up, lo, slope, r2, std = regression_with_band_series(close, mult=2.0)
    npx = compute_npx_from_bands(close, up, lo)
    return npx


def daily_last_npx_in_view(symbol: str, daily_view_label: str = "1Y", ntd_win: int = 120) -> Tuple[float, Optional[pd.Timestamp]]:
    npx = daily_npx_series_in_view(symbol, daily_view_label=daily_view_label, ntd_win=ntd_win).dropna()
    if npx.empty:
        return np.nan, None
    return float(npx.iloc[-1]), npx.index[-1]


def daily_r2_band_proximity(symbol: str, daily_view_label: str, slope_lb: int, prox: float) -> Optional[Dict]:
    """
    Computes last-bar proximity of Close to regression ±2σ (based on last slope_lb bars).
    prox is decimal (e.g., 0.01 means 1%).
    """
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None

    m, b, r2, std, fit, up, lo = regression_with_band_lastn(close, lookback=int(slope_lb), mult=2.0)
    last_close = float(close.iloc[-1])
    if not (np.isfinite(up) and np.isfinite(lo)):
        return None

    absdist_up = abs(last_close - up) / max(1e-12, abs(up))
    absdist_lo = abs(last_close - lo) / max(1e-12, abs(lo))

    near_up = absdist_up <= float(prox)
    near_lo = absdist_lo <= float(prox)

    return {
        "Symbol": symbol,
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "Slope": float(m) if np.isfinite(m) else np.nan,
        "Close": float(last_close),
        "Upper +2σ": float(up),
        "Lower -2σ": float(lo),
        "AbsDist Upper (%)": float(absdist_up * 100.0),
        "AbsDist Lower (%)": float(absdist_lo * 100.0),
        "Near Upper": bool(near_up),
        "Near Lower": bool(near_lo),
        "AsOf": close.index[-1],
    }


def daily_support_reversal_heading_up(
    symbol: str,
    daily_view_label: str,
    sr_lb: int,
    prox: float,
    bars_confirm: int,
    horizon: int
) -> Optional[Dict]:
    """
    Approximate support = rolling min(Low, sr_lb).
    Touch if Low <= support*(1+prox).
    Reversal heading up if last `bars_confirm` closes are increasing and after last touch.
    """
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or not {"Low", "Close"}.issubset(df.columns):
        return None

    df = df.dropna(subset=["Low", "Close"]).copy()
    if len(df) < max(10, sr_lb + 2):
        return None

    sr_lb = int(max(5, sr_lb))
    horizon = int(max(1, horizon))
    prox = float(max(0.0, prox))
    bars_confirm = int(max(1, bars_confirm))

    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    support = low.rolling(sr_lb, min_periods=max(3, sr_lb // 2)).min()

    # Evaluate last horizon bars for touch
    tail = df.iloc[-horizon:].copy()
    tail_support = support.reindex(tail.index)
    touch = (tail["Low"] <= (tail_support * (1.0 + prox)))

    if not touch.any():
        return None

    last_touch_time = touch[touch].index[-1]
    bars_since = int((len(df) - 1) - df.index.get_loc(last_touch_time))

    # confirmation: last bars_confirm closes increasing
    if not _series_heading_up(close, confirm_bars=bars_confirm):
        return None

    return {
        "Symbol": symbol,
        "Touch Time": last_touch_time,
        "Bars Since Touch": bars_since,
        "Support (approx)": float(tail_support.loc[last_touch_time]) if np.isfinite(tail_support.loc[last_touch_time]) else np.nan,
        "Low@Touch": float(df.loc[last_touch_time, "Low"]),
        "Close Now": float(close.iloc[-1]),
    }


def last_daily_fib_npx_zero_signal(
    symbol: str,
    daily_view_label: str,
    ntd_win: int,
    direction: str,
    prox: float,
    lookback_bars: int,
    slope_lb: int,
    npx_confirm_bars: int = 1
) -> Optional[Dict]:
    """
    BUY: touched fib 100% (swing low) and NPX crossed UP through 0.0 recently.
    SELL: touched fib 0% (swing high) and NPX crossed DOWN through 0.0 recently.
    """
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return None

    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(df) < 30:
        return None

    lookback_bars = int(max(2, lookback_bars))
    prox = float(max(0.0, prox))

    # fib swing from view range
    swing_high = float(df["High"].max())
    swing_low = float(df["Low"].min())
    fib0 = swing_high
    fib100 = swing_low

    close = df["Close"].astype(float)

    # NPX series
    fit, up, lo, slope, r2, std = regression_with_band_series(close, mult=2.0)
    npx = compute_npx_from_bands(close, up, lo).dropna()
    if npx.empty:
        return None

    # lookback window
    w = df.iloc[-lookback_bars:].copy()
    w_npx = npx.reindex(w.index).dropna()
    if w_npx.empty or len(w_npx) < 2:
        return None

    # touch tests
    touch_buy = (w["Low"] <= fib100 * (1.0 + prox)) | (w["Close"] <= fib100 * (1.0 + prox))
    touch_sell = (w["High"] >= fib0 * (1.0 - prox)) | (w["Close"] >= fib0 * (1.0 - prox))

    # NPX cross 0.0
    cross_up = (w_npx.shift(1) < 0.0) & (w_npx >= 0.0)
    cross_dn = (w_npx.shift(1) > 0.0) & (w_npx <= 0.0)

    direction = str(direction).upper().strip()
    if direction == "BUY":
        if not touch_buy.any() or not cross_up.any():
            return None
        t_cross = cross_up[cross_up].index[-1]
    else:
        if not touch_sell.any() or not cross_dn.any():
            return None
        t_cross = cross_dn[cross_dn].index[-1]

    bars_since = int((len(df) - 1) - df.index.get_loc(t_cross))
    return {
        "Symbol": symbol,
        "Direction": direction,
        "Bars Since Cross": bars_since,
        "Cross Time": t_cross,
        "Close@Cross": float(close.loc[t_cross]) if t_cross in close.index else np.nan,
        "Swing High (Fib 0%)": float(fib0),
        "Swing Low (Fib 100%)": float(fib100),
        "Slope": float(slope) if np.isfinite(slope) else np.nan,
        "R2": float(r2) if np.isfinite(r2) else np.nan,
    }


def last_daily_kijun_cross_up(
    symbol: str,
    daily_view_label: str,
    slope_lb: int,
    conv: int,
    base: int,
    span_b: int,
    within_last_n_bars: int = 5
) -> Optional[Dict]:
    start = daily_view_to_start(daily_view_label)
    df = fetch_daily_range(symbol, start=start)
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        return None

    df = df.dropna(subset=["High", "Low", "Close"]).copy()
    if len(df) < max(60, base + 5):
        return None

    ichi = compute_ichimoku(df, conv=conv, base=base, span_b=span_b)
    kijun = _coerce_1d_series(ichi["kijun"]).reindex(df.index).astype(float)
    close = df["Close"].astype(float)

    ok = kijun.notna() & close.notna()
    if ok.sum() < 5:
        return None

    kijun = kijun[ok]
    close = close[ok]

    crossed = (close.shift(1) <= kijun.shift(1)) & (close > kijun)
    if not crossed.any():
        return None

    t_cross = crossed[crossed].index[-1]
    bars_since = int((len(close) - 1) - close.index.get_loc(t_cross))
    if within_last_n_bars and bars_since > int(within_last_n_bars):
        return None

    # heading up (1 bar confirm) after cross
    if len(close) < 3 or not _series_heading_up(close, confirm_bars=1):
        return None

    # regression stats (last slope_lb bars)
    m, b, r2, std, fit, up, lo = regression_with_band_lastn(close, lookback=int(slope_lb), mult=2.0)

    return {
        "Symbol": symbol,
        "Bars Since Cross": bars_since,
        "Cross Time": t_cross,
        "Price@Cross": float(close.loc[t_cross]),
        "Kijun@Cross": float(kijun.loc[t_cross]) if np.isfinite(kijun.loc[t_cross]) else np.nan,
        "Slope": float(m) if np.isfinite(m) else np.nan,
        "R2": float(r2) if np.isfinite(r2) else np.nan,
    }


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Stock Wizard (News Removed)", layout="wide")

st.title("Stock Wizard — Bull/Bear Dashboard (News Removed)")
st.caption("This build removes all news fetching/rendering to prevent NameError and reduce API churn.")

with st.sidebar:
    st.header("Settings")

    mode = st.radio("Mode", ["Single Ticker", "Universe Scan"], index=0, key="mode_sel")
    default_universe = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD"
    tickers_text = st.text_input("Universe tickers (comma-separated)", value=default_universe, key="universe_text")
    universe = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    if not universe:
        universe = ["AAPL"]

    if mode == "Single Ticker":
        disp_ticker = st.selectbox("Display ticker", universe, index=0, key="disp_ticker")
    else:
        disp_ticker = st.selectbox("Display ticker (for charts)", universe, index=0, key="disp_ticker_scan")

    daily_view = st.selectbox("Daily view range", ["6M", "1Y", "2Y", "5Y", "YTD", "Max"], index=1, key="daily_view")
    hour_period = st.selectbox("Hourly period", ["1d", "2d", "4d"], index=0, key="hour_period")

    st.subheader("Regression / Bands")
    slope_lb_daily = st.slider("Daily slope lookback (bars)", 30, 300, 120, 5, key="slope_lb_daily")
    slope_lb_hourly = st.slider("Hourly slope lookback (bars)", 20, 200, 60, 5, key="slope_lb_hourly")

    band_mult = st.slider("Regression band σ-mult", 1.0, 4.0, 2.0, 0.1, key="band_mult")

    st.subheader("Bollinger Bands")
    show_bbands = st.checkbox("Show Bollinger Bands", value=True, key="show_bbands")
    bb_win = st.slider("BB window", 10, 80, 20, 1, key="bb_win")
    bb_mult = st.slider("BB mult", 1.0, 4.0, 2.0, 0.1, key="bb_mult")
    bb_use_ema = st.checkbox("BB midline uses EMA", value=False, key="bb_use_ema")

    st.subheader("Ichimoku")
    show_ichimoku = st.checkbox("Show Ichimoku (Kijun/Tenkan)", value=False, key="show_ichimoku")
    ichi_conv = st.slider("Ichimoku Tenkan (conv)", 5, 20, 9, 1, key="ichi_conv")
    ichi_base = st.slider("Ichimoku Kijun (base)", 10, 60, 26, 1, key="ichi_base")
    ichi_spanb = st.slider("Ichimoku SpanB", 20, 120, 52, 1, key="ichi_spanb")

    st.subheader("S/R + Reversal Scans")
    sr_prox_pct_ui = st.slider("S/R proximity (%)", 0.1, 5.0, 1.0, 0.1, key="sr_prox_ui")
    sr_prox_pct = float(sr_prox_pct_ui / 100.0)

    sr_lb_daily = st.slider("Support rolling window (bars)", 10, 120, 40, 5, key="sr_lb_daily")
    rev_bars_confirm = st.slider("Reversal confirm bars", 1, 5, 1, 1, key="rev_confirm_bars")
    rev_horizon = st.slider("Support-touch horizon (bars)", 3, 60, 10, 1, key="rev_horizon")

    st.subheader("NTD / NPX")
    ntd_window = st.slider("NTD window (NPX calc)", 30, 300, 120, 5, key="ntd_window")
# -----------------------------
# Tabs
# -----------------------------
tab_labels = [
    "1) Daily Chart",
    "2) Hourly Chart",
    "3) Daily Stats",
    "4) Hourly Stats",
    "5) Fibonacci",
    "6) S/R (Approx)",
    "7) Notes",
    "8) About",
    "9) Fib NPX 0.0 Signal Scanner",
    "10) Slope Direction Scan",
    "11) Trendline Direction Lists",
    "12) NTD Hot List",
    "13) NTD NPX 0.0–0.2 Scanner",
    "14) Uptrend vs Downtrend",
    "15) Ichimoku Kijun Scanner",
    "16) R² > 45% Daily/Hourly",
    "17) R² < 45% Daily/Hourly",
    "18) R² Sign ±2σ Proximity (Daily)",
    "19) AVWAP + ADX Regime Gate",
]
tabs = st.tabs(tab_labels)
(
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,
    tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19
) = tabs


# -----------------------------
# Tab 1: Daily Chart
# -----------------------------
with tab1:
    st.header("Daily Chart")
    start = daily_view_to_start(daily_view)
    df = fetch_daily_range(disp_ticker, start=start)

    if df.empty:
        st.warning("No daily data.")
    else:
        close = df["Close"].astype(float)
        fit, up, lo, slope, r2, std = regression_with_band_series(close, mult=float(band_mult))
        npx = compute_npx_from_bands(close, up, lo)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title(f"{disp_ticker} — Daily — View {daily_view}")
        ax.plot(close.index, close.values, label="Close")
        if not fit.dropna().empty:
            ax.plot(fit.index, fit.values, linewidth=1.5, label="Regression Fit")
        if not up.dropna().empty and not lo.dropna().empty:
            ax.fill_between(close.index, lo.reindex(close.index), up.reindex(close.index), alpha=0.08, label=f"±{band_mult:.1f}σ band")

        if show_bbands:
            bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
            if not bb_up.dropna().empty and not bb_lo.dropna().empty:
                ax.plot(bb_mid.index, bb_mid.values, linewidth=1.0, label="BB mid")
                ax.plot(bb_up.index, bb_up.values, linestyle=":", linewidth=1.0, label="BB upper")
                ax.plot(bb_lo.index, bb_lo.values, linestyle=":", linewidth=1.0, label="BB lower")

        if show_ichimoku:
            ichi = compute_ichimoku(df, conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
            tenkan = ichi["tenkan"].reindex(close.index)
            kijun = ichi["kijun"].reindex(close.index)
            ax.plot(tenkan.index, tenkan.values, linewidth=1.0, label="Tenkan")
            ax.plot(kijun.index, kijun.values, linewidth=1.0, label="Kijun")

        style_axes(ax)
        ax.legend(loc="best", framealpha=0.6)
        st.pyplot(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slope (view)", f"{slope:.4f}" if np.isfinite(slope) else "n/a")
        c2.metric("R² (view)", f"{r2:.3f}" if np.isfinite(r2) else "n/a")
        c3.metric("NPX (last)", f"{float(npx.dropna().iloc[-1]):.3f}" if not npx.dropna().empty else "n/a")
        c4.metric("Close (last)", fmt_price_val(close.iloc[-1]))


# -----------------------------
# Tab 2: Hourly Chart
# -----------------------------
with tab2:
    st.header("Hourly Chart (60m)")
    df = fetch_intraday(disp_ticker, period=hour_period, interval="60m")
    if df.empty:
        st.warning("No hourly data.")
    else:
        close = df["Close"].astype(float)
        fit, up, lo, slope, r2, std = regression_with_band_series(close, mult=float(band_mult))
        npx = compute_npx_from_bands(close, up, lo)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title(f"{disp_ticker} — Hourly (60m) — Period {hour_period}")
        ax.plot(close.index, close.values, label="Close")
        if not fit.dropna().empty:
            ax.plot(fit.index, fit.values, linewidth=1.5, label="Regression Fit")
        if not up.dropna().empty and not lo.dropna().empty:
            ax.fill_between(close.index, lo.reindex(close.index), up.reindex(close.index), alpha=0.08, label=f"±{band_mult:.1f}σ band")

        if show_bbands:
            bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
            if not bb_up.dropna().empty and not bb_lo.dropna().empty:
                ax.plot(bb_mid.index, bb_mid.values, linewidth=1.0, label="BB mid")
                ax.plot(bb_up.index, bb_up.values, linestyle=":", linewidth=1.0, label="BB upper")
                ax.plot(bb_lo.index, bb_lo.values, linestyle=":", linewidth=1.0, label="BB lower")

        style_axes(ax)
        ax.legend(loc="best", framealpha=0.6)
        st.pyplot(fig)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slope (period)", f"{slope:.4f}" if np.isfinite(slope) else "n/a")
        c2.metric("R² (period)", f"{r2:.3f}" if np.isfinite(r2) else "n/a")
        c3.metric("NPX (last)", f"{float(npx.dropna().iloc[-1]):.3f}" if not npx.dropna().empty else "n/a")
        c4.metric("Close (last)", fmt_price_val(close.iloc[-1]))


# -----------------------------
# Tab 3: Daily Stats
# -----------------------------
with tab3:
    st.header("Daily Stats")
    r2_d, m_d, ts_d = daily_regression_r2(disp_ticker, slope_lb=int(slope_lb_daily), daily_view_label=daily_view)
    npx_last, npx_ts = daily_last_npx_in_view(disp_ticker, daily_view_label=daily_view, ntd_win=ntd_window)
    st.write(
        {
            "Ticker": disp_ticker,
            "Daily View": daily_view,
            "R2 (last N bars)": r2_d,
            "Slope (last N bars)": m_d,
            "AsOf": ts_d,
            "NPX (view regression, last)": npx_last,
            "NPX AsOf": npx_ts,
        }
    )


# -----------------------------
# Tab 4: Hourly Stats
# -----------------------------
with tab4:
    st.header("Hourly Stats")
    r2_h, m_h, ts_h = hourly_regression_r2(disp_ticker, period=hour_period, slope_lb=int(slope_lb_hourly))
    st.write(
        {
            "Ticker": disp_ticker,
            "Period": hour_period,
            "R2 (last N bars)": r2_h,
            "Slope (last N bars)": m_h,
            "AsOf": ts_h,
        }
    )


# -----------------------------
# Tab 5: Fibonacci (simple swing)
# -----------------------------
with tab5:
    st.header("Fibonacci (Swing High/Low on Daily View)")
    start = daily_view_to_start(daily_view)
    df = fetch_daily_range(disp_ticker, start=start)
    if df.empty:
        st.warning("No data.")
    else:
        swing_high = float(df["High"].max())
        swing_low = float(df["Low"].min())
        st.write({"Swing High (0%)": swing_high, "Swing Low (100%)": swing_low})

        levels = {
            "0%": swing_high,
            "23.6%": swing_high - 0.236 * (swing_high - swing_low),
            "38.2%": swing_high - 0.382 * (swing_high - swing_low),
            "50.0%": swing_high - 0.5 * (swing_high - swing_low),
            "61.8%": swing_high - 0.618 * (swing_high - swing_low),
            "78.6%": swing_high - 0.786 * (swing_high - swing_low),
            "100%": swing_low,
        }
        st.dataframe(pd.DataFrame({"Level": list(levels.keys()), "Price": list(levels.values())}))


# -----------------------------
# Tab 6: S/R (Approx)
# -----------------------------
with tab6:
    st.header("Support/Resistance (Approx) — Rolling Low/High")
    start = daily_view_to_start(daily_view)
    df = fetch_daily_range(disp_ticker, start=start)
    if df.empty:
        st.warning("No data.")
    else:
        sr_win = int(sr_lb_daily)
        support = df["Low"].rolling(sr_win, min_periods=max(3, sr_win // 2)).min()
        resist = df["High"].rolling(sr_win, min_periods=max(3, sr_win // 2)).max()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_title(f"{disp_ticker} — Daily — Approx S/R (window={sr_win})")
        ax.plot(df.index, df["Close"].values, label="Close")
        ax.plot(support.index, support.values, linewidth=1.2, label="Support (roll min)")
        ax.plot(resist.index, resist.values, linewidth=1.2, label="Resistance (roll max)")
        style_axes(ax)
        ax.legend(loc="best", framealpha=0.6)
        st.pyplot(fig)

        st.write(
            {
                "Close (last)": float(df["Close"].iloc[-1]),
                "Support (last)": float(support.dropna().iloc[-1]) if not support.dropna().empty else np.nan,
                "Resistance (last)": float(resist.dropna().iloc[-1]) if not resist.dropna().empty else np.nan,
                "Proximity setting (%)": sr_prox_pct_ui,
            }
        )


# -----------------------------
# Tab 7: Notes
# -----------------------------
with tab7:
    st.header("Notes")
    st.markdown(
        "- This file intentionally **removes all news functionality**.\n"
        "- If you previously had a News tab or `fetch_yf_news` calls, they are not present here.\n"
        "- Use Universe Scan tabs (9–19) for screening."
    )


# -----------------------------
# Tab 8: About
# -----------------------------
with tab8:
    st.header("About")
    st.markdown(
        "This Streamlit app provides:\n"
        "- Daily/Hourly regression bands + BB + optional Ichimoku\n"
        "- Multiple universe scanners\n"
        "- AVWAP + ADX regime gate\n\n"
        "**News has been removed** to prevent `NameError: fetch_yf_news` and to reduce external calls."
    )


# ---------------------------
# TAB 9: Fib NPX 0.0 Signal Scanner
# ---------------------------
with tab9:
    st.header("Fib NPX 0.0 Signal Scanner")
    st.caption(
        "Scans the current universe for **Fibonacci BUY/SELL** signals on the **Daily** chart:\n"
        "• **Fib BUY:** price touched **100%** (low) and NPX crossed **UP** through **0.0** recently\n"
        "• **Fib SELL:** price touched **0%** (high) and NPX crossed **DOWN** through **0.0** recently\n\n"
        "Uses the selected Daily view range and the sidebar S/R proximity (%) for touch tolerance."
    )

    c1, c2 = st.columns(2)
    lb_sig = c1.slider("Lookback window (bars) for touch + NPX cross", 2, 90, int(max(3, rev_horizon)), 1, key="fibnpx0_lb")
    run_fibsig = c2.button("Run Fib NPX 0.0 Scan", key=f"btn_run_fibnpx0_{mode}")

    if run_fibsig:
        buy_rows, sell_rows = [], []
        for sym in universe:
            rb = last_daily_fib_npx_zero_signal(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=ntd_window,
                direction="BUY",
                prox=sr_prox_pct,
                lookback_bars=int(lb_sig),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if rb is not None:
                buy_rows.append(rb)

            rs = last_daily_fib_npx_zero_signal(
                symbol=sym,
                daily_view_label=daily_view,
                ntd_win=ntd_window,
                direction="SELL",
                prox=sr_prox_pct,
                lookback_bars=int(lb_sig),
                slope_lb=int(slope_lb_daily),
                npx_confirm_bars=1
            )
            if rs is not None:
                sell_rows.append(rs)

        left, right = st.columns(2)
        with left:
            st.subheader("Fib BUY — 100% touch + NPX 0.0↑")
            if not buy_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(buy_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("Fib SELL — 0% touch + NPX 0.0↓")
            if not sell_rows:
                st.info("No matches.")
            else:
                out = pd.DataFrame(sell_rows)
                out["Bars Since Cross"] = out["Bars Since Cross"].astype(int)
                out = out.sort_values(["Bars Since Cross"], ascending=[True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 10: Slope Direction Scan
# ---------------------------
with tab10:
    st.header("Slope Direction Scan")
    st.caption("Lists symbols whose **current DAILY global trendline slope** is **up** vs **down** (selected Daily view range).")

    run_slope = st.button("Run Slope Direction Scan", key=f"btn_run_slope_dir_{mode}")

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
        "Displays symbols whose **current DAILY global trendline** is:\n"
        "• **Upward** (slope ≥ 0) AND NPX(last) < 0.0\n"
        "• **Downward** (slope < 0) AND NPX(last) > 0.5\n"
    )

    run_trend_lists = st.button("Run Trendline Direction Lists", key=f"btn_run_trendline_lists_{mode}")

    if run_trend_lists:
        up_rows, dn_rows = [], []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue
            npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            if not np.isfinite(npx_last):
                continue

            if float(m) >= 0.0 and float(npx_last) < 0.0:
                up_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})
            elif float(m) < 0.0 and float(npx_last) > 0.5:
                dn_rows.append({"Symbol": sym, "NPX (Norm Price)": float(npx_last)})

        left, right = st.columns(2)
        with left:
            st.subheader("Upward Trend (Slope ≥ 0) + NPX < 0")
            st.dataframe(pd.DataFrame(up_rows).sort_values("Symbol").reset_index(drop=True), use_container_width=True) if up_rows else st.info("No matches.")
        with right:
            st.subheader("Downward Trend (Slope < 0) + NPX > 0.5")
            st.dataframe(pd.DataFrame(dn_rows).sort_values("Symbol").reset_index(drop=True), use_container_width=True) if dn_rows else st.info("No matches.")


# ---------------------------
# TAB 12: NTD Hot List
# ---------------------------
with tab12:
    st.header("NTD Hot List")
    st.caption("Slope > 0 and NPX between 0.0 and 0.5 (inclusive) using selected Daily view range.")

    run_hot = st.button("Run NTD Hot List", key=f"btn_run_ntd_hot_{mode}")

    if run_hot:
        rows = []
        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m) or float(m) <= 0.0:
                continue
            npx_last, _ = daily_last_npx_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            if not np.isfinite(npx_last):
                continue
            if 0.0 <= float(npx_last) <= 0.5:
                rows.append({"Symbol": sym, "Slope": float(m), "NPX (Norm Price)": float(npx_last), "R2": float(r2) if np.isfinite(r2) else np.nan, "AsOf": ts})

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["Slope", "NPX (Norm Price)"], ascending=[False, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 13: NTD NPX 0.0–0.2 Scanner
# ---------------------------
with tab13:
    st.header("NTD NPX 0.0–0.2 Scanner")
    st.caption("Scans for NPX in [0.0, 0.2] and **heading up**, split by slope sign.")

    c1, c2 = st.columns(2)
    npx_up_bars = c1.slider("NPX heading-up confirmation (consecutive bars)", 1, 5, 1, 1, key="npx_02_up_bars")
    run_npx02 = c2.button("Run NTD NPX 0.0-0.2 Scan", key=f"btn_run_npx02_{mode}")

    if run_npx02:
        rows_up_slope, rows_dn_slope = [], []

        for sym in universe:
            m, r2, ts = daily_global_slope(sym, daily_view_label=daily_view)
            if not np.isfinite(m):
                continue

            npx_s = daily_npx_series_in_view(sym, daily_view_label=daily_view, ntd_win=ntd_window)
            npx_s = _coerce_1d_series(npx_s).dropna()
            if npx_s.empty or len(npx_s) < 2:
                continue

            npx_last = float(npx_s.iloc[-1]) if np.isfinite(npx_s.iloc[-1]) else np.nan
            if not np.isfinite(npx_last):
                continue

            if not (0.0 <= float(npx_last) <= 0.2):
                continue

            if not _series_heading_up(npx_s, confirm_bars=int(npx_up_bars)):
                continue

            row = {"Symbol": sym, "Slope": float(m), "R2": float(r2) if np.isfinite(r2) else np.nan, "NPX (Norm Price)": float(npx_last), "AsOf": ts}
            if float(m) > 0.0:
                rows_up_slope.append(row)
            elif float(m) < 0.0:
                rows_dn_slope.append(row)

        left, right = st.columns(2)
        with left:
            st.subheader("List 1 — Slope > 0 and NPX 0.0–0.2 heading up")
            st.dataframe(pd.DataFrame(rows_up_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, False]).reset_index(drop=True), use_container_width=True) if rows_up_slope else st.info("No matches.")
        with right:
            st.subheader("List 2 — Slope < 0 and NPX 0.0–0.2 heading up")
            st.dataframe(pd.DataFrame(rows_dn_slope).sort_values(["NPX (Norm Price)", "Slope"], ascending=[True, True]).reset_index(drop=True), use_container_width=True) if rows_dn_slope else st.info("No matches.")


# ---------------------------
# TAB 14: Uptrend vs Downtrend
# ---------------------------
with tab14:
    st.header("Uptrend vs Downtrend")
    st.caption(
        "Lists symbols where price reversed from support heading up, split into:\n"
        "• (a) Uptrend: Slope > 0\n"
        "• (b) Downtrend: Slope < 0"
    )

    c1, c2 = st.columns(2)
    hz_sr = c1.slider("Support-touch lookback window (bars)", 1, 60, int(max(3, rev_horizon)), 1, key="ud_sr_hz")
    run_ud = c2.button("Run Uptrend vs Downtrend Scan", key=f"btn_run_ud_{mode}")

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
                horizon=int(hz_sr)
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
                out = pd.DataFrame(rows_uptrend)
                out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, False])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)

        with right:
            st.subheader("(b) Downtrend — Slope < 0 and Support Reversal heading up")
            if not rows_downtrend:
                st.info("No matches.")
            else:
                out = pd.DataFrame(rows_downtrend)
                out["Bars Since Touch"] = out["Bars Since Touch"].astype(int)
                out = out.sort_values(["Bars Since Touch", "Slope"], ascending=[True, True])
                st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 15: Ichimoku Kijun Scanner
# ---------------------------
with tab15:
    st.header("Ichimoku Kijun Scanner")
    st.caption(
        "Daily-only:\n"
        "• List 1: slope > 0 AND crossed above Kijun (heading up)\n"
        "• List 2: slope < 0 AND crossed above Kijun (heading up)"
    )

    c1, c2 = st.columns(2)
    kijun_within = c1.slider("Cross must be within last N bars", 0, 60, 5, 1, key="kijun_within_n")
    run_kijun = c2.button("Run Ichimoku Kijun Scan", key=f"btn_run_kijun_scan_{mode}")

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
            )
            if r is None:
                continue

            m = float(r.get("Slope", np.nan))
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
    c1, c2, c3 = st.columns(3)
    r2_thr = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan")
    hour_period_scan = c2.selectbox("Hourly intraday period", ["1d", "2d", "4d"], index=0, key="r2_hour_period")
    run_r2 = c3.button("Run R² Scan", key=f"btn_run_r2_scan_{mode}")

    if run_r2:
        daily_rows, hourly_rows = [], []
        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily), daily_view_label=daily_view)
            if np.isfinite(r2_d) and float(r2_d) > float(r2_thr):
                daily_rows.append({"Symbol": sym, "R2": float(r2_d), "Slope": float(m_d) if np.isfinite(m_d) else np.nan, "AsOf": ts_d})

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period_scan), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) > float(r2_thr):
                hourly_rows.append({"Symbol": sym, "R2": float(r2_h), "Slope": float(m_h) if np.isfinite(m_h) else np.nan, "AsOf": ts_h, "Period": str(hour_period_scan)})

        left, right = st.columns(2)
        with left:
            st.subheader("Daily — R² > threshold")
            st.dataframe(pd.DataFrame(daily_rows).sort_values(["R2", "Slope"], ascending=[False, False]).reset_index(drop=True), use_container_width=True) if daily_rows else st.info("No matches.")
        with right:
            st.subheader(f"Hourly ({hour_period_scan}) — R² > threshold")
            st.dataframe(pd.DataFrame(hourly_rows).sort_values(["R2", "Slope"], ascending=[False, False]).reset_index(drop=True), use_container_width=True) if hourly_rows else st.info("No matches.")


# ---------------------------
# TAB 17: R² < 45% Daily/Hourly
# ---------------------------
with tab17:
    st.header("R² < 45% Daily/Hourly")
    c1, c2, c3 = st.columns(3)
    r2_thr_lo = c1.slider("R² threshold", 0.00, 1.00, 0.45, 0.01, key="r2_thr_scan_lo")
    hour_period_lo = c2.selectbox("Hourly intraday period", ["1d", "2d", "4d"], index=0, key="r2_hour_period_lo")
    run_r2_lo = c3.button("Run R² Low Scan", key=f"btn_run_r2_scan_lo_{mode}")

    if run_r2_lo:
        daily_rows, hourly_rows = [], []
        for sym in universe:
            r2_d, m_d, ts_d = daily_regression_r2(sym, slope_lb=int(slope_lb_daily), daily_view_label=daily_view)
            if np.isfinite(r2_d) and float(r2_d) < float(r2_thr_lo):
                daily_rows.append({"Symbol": sym, "R2": float(r2_d), "Slope": float(m_d) if np.isfinite(m_d) else np.nan, "AsOf": ts_d})

            r2_h, m_h, ts_h = hourly_regression_r2(sym, period=str(hour_period_lo), slope_lb=int(slope_lb_hourly))
            if np.isfinite(r2_h) and float(r2_h) < float(r2_thr_lo):
                hourly_rows.append({"Symbol": sym, "R2": float(r2_h), "Slope": float(m_h) if np.isfinite(m_h) else np.nan, "AsOf": ts_h, "Period": str(hour_period_lo)})

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
    st.caption("Four lists based on sign of R² and proximity to ±2σ bands (daily). Uses S/R proximity (%) as 'near' threshold.")

    run_band_scan = st.button("Run R² Sign ±2σ Proximity Scan (Daily)", key=f"btn_run_r2_sign_band_scan_{mode}")

    if run_band_scan:
        rows_pos_lower, rows_pos_upper = [], []
        rows_neg_lower, rows_neg_upper = [], []

        for sym in universe:
            r = daily_r2_band_proximity(symbol=sym, daily_view_label=daily_view, slope_lb=int(slope_lb_daily), prox=float(sr_prox_pct))
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

        st.info(f"Near threshold = ±{sr_prox_pct*100:.3f}%")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.subheader("R² > 0  •  Near Lower -2σ")
            st.dataframe(pd.DataFrame(rows_pos_lower).sort_values(["AbsDist Lower (%)", "R2"], ascending=[True, False]).reset_index(drop=True), use_container_width=True) if rows_pos_lower else st.info("No matches.")
        with r1c2:
            st.subheader("R² > 0  •  Near Upper +2σ")
            st.dataframe(pd.DataFrame(rows_pos_upper).sort_values(["AbsDist Upper (%)", "R2"], ascending=[True, False]).reset_index(drop=True), use_container_width=True) if rows_pos_upper else st.info("No matches.")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.subheader("R² < 0  •  Near Lower -2σ")
            st.dataframe(pd.DataFrame(rows_neg_lower).sort_values(["AbsDist Lower (%)", "R2"], ascending=[True, True]).reset_index(drop=True), use_container_width=True) if rows_neg_lower else st.info("No matches.")
        with r2c2:
            st.subheader("R² < 0  •  Near Upper +2σ")
            st.dataframe(pd.DataFrame(rows_neg_upper).sort_values(["AbsDist Upper (%)", "R2"], ascending=[True, True]).reset_index(drop=True), use_container_width=True) if rows_neg_upper else st.info("No matches.")


# ---------------------------
# TAB 19: AVWAP + ADX Regime Gate
# ---------------------------
with tab19:
    st.header("Anchored VWAP (AVWAP) + ADX Regime Gate")
    st.caption(
        "• AVWAP anchors: last pivot low (long bias) & last pivot high (short bias)\n"
        "• Regime gate: TREND when ADX(14) > threshold; else RANGE\n"
        "• Range mode: BB fade (mean-reversion)\n"
        "• Trend mode: AVWAP cross + DI confirmation (optional volume confirm)"
    )

    cA, cB, cC, cD = st.columns(4)
    sel_av = cA.selectbox("Ticker:", universe, key=f"avwap_ticker_{mode}")
    tf = cB.radio("Timeframe:", ["Daily", "Hourly"], index=0, key=f"avwap_tf_{mode}")
    pivot_span = cC.slider("Pivot span (bars)", 2, 15, 5, 1, key=f"avwap_pivot_span_{mode}")
    adx_thr = cD.slider("ADX(14) threshold", 10.0, 50.0, 22.5, 0.5, key=f"avwap_adx_thr_{mode}")

    cE, cF, cG = st.columns(3)
    use_vol_confirm = cE.checkbox("Use volume confirmation", value=False, key=f"avwap_use_vol_{mode}")
    vol_win = cF.slider("Volume MA window", 5, 60, 20, 1, key=f"avwap_vol_win_{mode}")
    show_vol_panel = cG.checkbox("Show volume panel", value=False, key=f"avwap_show_vol_{mode}")

    hour_range_av = "24h"
    period_map_av = {"24h": "1d", "48h": "2d", "96h": "4d"}
    if tf == "Hourly":
        hour_range_av = st.selectbox("Hourly lookback:", ["24h", "48h", "96h"], index=0, key=f"avwap_hour_range_{mode}")

    run_av = st.button("Run AVWAP + ADX", key=f"btn_run_avwap_{mode}")

    if run_av:
        if tf == "Daily":
            ohlcv = fetch_hist_ohlcv(sel_av)
        else:
            ohlcv = fetch_intraday(sel_av, period=period_map_av.get(hour_range_av, "1d"), interval="60m")

        if ohlcv is None or ohlcv.empty or not {"High", "Low", "Close"}.issubset(ohlcv.columns):
            st.warning("No OHLC data available.")
        else:
            ohlcv = ohlcv.sort_index()

            t_piv_low = _last_pivot_time(ohlcv, span=int(pivot_span), kind="low")
            t_piv_high = _last_pivot_time(ohlcv, span=int(pivot_span), kind="high")

            avwap_long = anchored_vwap(ohlcv, t_piv_low, use_typical_price=True)
            avwap_short = anchored_vwap(ohlcv, t_piv_high, use_typical_price=True)

            adx, pdi, ndi = compute_adx(ohlcv, period=14)
            adx_last = float(adx.dropna().iloc[-1]) if len(adx.dropna()) else np.nan
            regime = "TREND" if (np.isfinite(adx_last) and float(adx_last) > float(adx_thr)) else "RANGE"

            close = _coerce_1d_series(ohlcv["Close"]).astype(float)

            # Volume confirmation
            vol_ok = pd.Series(False, index=ohlcv.index)
            vol_ma = pd.Series(index=ohlcv.index, dtype=float)
            has_vol = ("Volume" in ohlcv.columns) and _has_volume_to_plot(ohlcv["Volume"])
            if has_vol:
                vol_ok, vol_ma = volume_confirmation_mask(ohlcv["Volume"], window=int(vol_win), mult=1.0)
            if not use_vol_confirm:
                vol_ok = pd.Series(True, index=ohlcv.index)

            # Trend-following candidates (TREND)
            trend_buy = pd.Series(False, index=close.index)
            trend_sell = pd.Series(False, index=close.index)

            if not avwap_long.dropna().empty:
                above_long = close > avwap_long
                trend_buy = above_long & (~above_long.shift(1).fillna(False)) & (pdi > ndi) & vol_ok

            if not avwap_short.dropna().empty:
                below_short = close < avwap_short
                trend_sell = below_short & (~below_short.shift(1).fillna(False)) & (ndi > pdi) & vol_ok

            t_buy = trend_buy[trend_buy].index[-1] if trend_buy.any() else None
            t_sell = trend_sell[trend_sell].index[-1] if trend_sell.any() else None

            # Range candidates (RANGE)
            bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
            bb_sig = bb_fade_last_signal(close, bb_up, bb_lo)

            st.subheader("Regime Gate")
            if regime == "TREND":
                st.success(f"Regime: TREND MODE • ADX(14)={adx_last:.2f} > {adx_thr:.1f}")
            else:
                st.info(f"Regime: RANGE MODE • ADX(14)={adx_last:.2f} ≤ {adx_thr:.1f}" if np.isfinite(adx_last) else "Regime: RANGE MODE • ADX(14)=n/a")

            st.subheader("Latest Candidate Signal (Regime-Gated)")
            if regime == "TREND":
                last_sig = None
                if t_buy is None and t_sell is None:
                    last_sig = None
                elif t_sell is None or (t_buy is not None and t_buy >= t_sell):
                    last_sig = {"side": "BUY", "time": t_buy, "price": float(close.loc[t_buy]) if t_buy in close.index else np.nan, "note": "Trend: AVWAP↑ +DI>-DI"}
                else:
                    last_sig = {"side": "SELL", "time": t_sell, "price": float(close.loc[t_sell]) if t_sell in close.index else np.nan, "note": "Trend: AVWAP↓ -DI>+DI"}

                if last_sig is None:
                    st.write("No trend-following candidate found (AVWAP cross + DI + volume gate).")
                else:
                    st.write(f"**{last_sig['side']}** @ {fmt_price_val(last_sig['price'])} • {last_sig['time']} • {last_sig['note']}")
            else:
                if bb_sig is None:
                    st.write("No mean-reversion candidate found (BB fade).")
                else:
                    st.write(f"**{bb_sig['side']}** @ {fmt_price_val(bb_sig['price'])} • {bb_sig['time']} • {bb_sig['note']}")

            rows = 3 if show_vol_panel else 2
            heights = [3.0, 1.2, 1.2] if show_vol_panel else [3.2, 1.3]
            fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(14, 7.5), gridspec_kw={"height_ratios": heights})
            plt.subplots_adjust(hspace=0.06, top=0.92, right=0.93, bottom=0.28)

            axp = axes[0]
            axd = axes[1] if rows >= 2 else axes

            axp.set_title(f"{sel_av} — {tf} — AVWAP (Pivot Low/High) + ADX Regime Gate")
            axp.plot(close.index, close.values, label="Close")

            if not avwap_long.dropna().empty:
                axp.plot(avwap_long.index, avwap_long.values, linewidth=1.8, label="AVWAP (anchor: pivot LOW)")
            if not avwap_short.dropna().empty:
                axp.plot(avwap_short.index, avwap_short.values, linewidth=1.8, label="AVWAP (anchor: pivot HIGH)")

            if show_bbands and not bb_up.dropna().empty and not bb_lo.dropna().empty:
                axp.fill_between(close.index, bb_lo.reindex(close.index), bb_up.reindex(close.index), alpha=0.06, label=f"BB (×{bb_mult:.1f})")
                axp.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
                axp.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
                axp.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

            try:
                if t_piv_low is not None and t_piv_low in ohlcv.index:
                    axp.scatter([t_piv_low], [float(ohlcv.loc[t_piv_low, "Low"])], marker="^", s=90, zorder=8, label="Pivot LOW (anchor)")
            except Exception:
                pass
            try:
                if t_piv_high is not None and t_piv_high in ohlcv.index:
                    axp.scatter([t_piv_high], [float(ohlcv.loc[t_piv_high, "High"])], marker="v", s=90, zorder=8, label="Pivot HIGH (anchor)")
            except Exception:
                pass

            if regime == "TREND":
                if t_buy is not None:
                    annotate_crossover(axp, t_buy, float(close.loc[t_buy]), "BUY", note="AVWAP")
                if t_sell is not None:
                    annotate_crossover(axp, t_sell, float(close.loc[t_sell]), "SELL", note="AVWAP")
            else:
                if isinstance(bb_sig, dict) and bb_sig.get("time", None) is not None and np.isfinite(bb_sig.get("price", np.nan)):
                    annotate_crossover(axp, bb_sig["time"], bb_sig["price"], bb_sig["side"], note="BB Fade")

            axp.text(
                0.99, 0.02,
                f"Regime: {regime}  |  ADX(14): {adx_last:.2f}  |  Thr: {adx_thr:.1f}",
                transform=axp.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.75)
            )

            axd.set_title("ADX(14) +DI/-DI")
            if not adx.dropna().empty:
                axd.plot(adx.index, adx.values, linewidth=1.6, label="ADX(14)")
                axd.axhline(float(adx_thr), linestyle="--", linewidth=1.2, label=f"Threshold ({adx_thr:.1f})")
            if not pdi.dropna().empty:
                axd.plot(pdi.index, pdi.values, linewidth=1.1, label="+DI")
            if not ndi.dropna().empty:
                axd.plot(ndi.index, ndi.values, linewidth=1.1, label="-DI")
            axd.set_ylim(0, 60)

            if show_vol_panel:
                axv = axes[2]
                axv.set_title("Volume (optional)")
                if has_vol:
                    v = _coerce_1d_series(ohlcv["Volume"]).astype(float)
                    axv.plot(v.index, v.values, linewidth=1.0, label="Volume")
                    if not vol_ma.dropna().empty:
                        axv.plot(vol_ma.index, vol_ma.values, linewidth=1.2, label=f"Vol MA({int(vol_win)})")
                else:
                    axv.text(0.5, 0.5, "No usable volume available.", transform=axv.transAxes, ha="center", va="center", fontsize=10)

            handles, labels = [], []
            h, l = axp.get_legend_handles_labels()
            handles += h; labels += l
            h, l = axd.get_legend_handles_labels()
            handles += h; labels += l
            if show_vol_panel:
                h, l = axv.get_legend_handles_labels()
                handles += h; labels += l

            seen = set()
            h_u, l_u = [], []
            for hh, ll in zip(handles, labels):
                if not ll or ll in seen:
                    continue
                seen.add(ll)
                h_u.append(hh)
                l_u.append(ll)

            fig.legend(
                handles=h_u,
                labels=l_u,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=4,
                frameon=True,
                fontsize=9,
                framealpha=0.65,
                fancybox=True,
                borderpad=0.6,
                handlelength=2.0
            )

            style_axes(axp)
            style_axes(axd)
            if show_vol_panel:
                style_axes(axv)

            st.pyplot(fig)
