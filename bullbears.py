# forex_hourly_trading_indicator.py
# Clean Forex-only Streamlit dashboard with hourly trading indicators.
# Focus: compressed intraday chart (no weekend/closure gaps), trend-aligned support/resistance reversals,
# 30 EMA crosses, NTD/S/R reversal confirmation, and forex scanner.

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pytz
from matplotlib.lines import Line2D


# =========================
# App / Styling
# =========================
PACIFIC = pytz.timezone("US/Pacific")
REFRESH_INTERVAL = 120

st.set_page_config(
    page_title="Forex Hourly Trading Indicator",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
#MainMenu, header, footer {visibility: hidden;}
div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    gap: .45rem !important;
}
div[data-baseweb="tab"] { flex: 0 0 auto !important; }
.small-muted { color: #888; font-size: 0.90rem; }
.signal-buy {
    padding: 0.75rem 1rem; border-radius: 0.75rem;
    background: rgba(0, 150, 80, 0.15); border: 1px solid rgba(0,150,80,0.35);
    font-weight: 700;
}
.signal-sell {
    padding: 0.75rem 1rem; border-radius: 0.75rem;
    background: rgba(220, 40, 40, 0.15); border: 1px solid rgba(220,40,40,0.35);
    font-weight: 700;
}
.signal-wait {
    padding: 0.75rem 1rem; border-radius: 0.75rem;
    background: rgba(160, 160, 160, 0.15); border: 1px solid rgba(160,160,160,0.35);
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
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
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.70,
        "legend.fancybox": True,
        "lines.linewidth": 1.55,
    }
)


# =========================
# Universe / helpers
# =========================
FOREX_UNIVERSE = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X", "USDCHF=X",
    "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
    "EURGBP=X", "EURCAD=X", "EURAUD=X", "EURNZD=X", "GBPCAD=X", "GBPAUD=X", "GBPNZD=X",
    "AUDCAD=X", "AUDNZD=X", "NZDCAD=X", "EURCHF=X", "GBPCHF=X", "AUDCHF=X", "CADCHF=X",
    "USDHKD=X", "EURHKD=X", "GBPHKD=X", "HKDJPY=X",
]


def auto_refresh() -> None:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass


def pip_size(symbol: str) -> float:
    s = str(symbol).upper()
    return 0.01 if "JPY" in s else 0.0001


def fmt_price(symbol: str, value) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.3f}" if "JPY" in symbol.upper() else f"{v:.5f}"


def fmt_pips(symbol: str, value) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    ps = pip_size(symbol)
    return f"{v / ps:.1f} pips"


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(np.squeeze(value))
    except Exception:
        return default
    return out if np.isfinite(out) else default


def coerce_series(obj, index=None) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float, index=index)
    if isinstance(obj, pd.Series):
        s = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        numeric_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        s = obj[numeric_cols[0]].copy() if numeric_cols else pd.Series(dtype=float, index=index)
    else:
        s = pd.Series(obj, index=index)
    return pd.to_numeric(s, errors="coerce")


def flatten_yf_columns(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if ticker:
            for level in range(out.columns.nlevels):
                values = [str(v).upper() for v in out.columns.get_level_values(level)]
                if str(ticker).upper() in values:
                    try:
                        out = out.xs(ticker, axis=1, level=level, drop_level=True)
                        break
                    except Exception:
                        pass
        if isinstance(out.columns, pd.MultiIndex):
            for level in range(out.columns.nlevels):
                values = set(map(str, out.columns.get_level_values(level)))
                if wanted.intersection(values):
                    out.columns = out.columns.get_level_values(level)
                    break
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = ["_".join(map(str, c)).strip("_") for c in out.columns.to_flat_index()]
    out.columns = [str(c) for c in out.columns]
    return out


def ensure_pacific_index(df_or_series):
    if df_or_series is None or len(df_or_series) == 0:
        return df_or_series
    out = df_or_series.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[out.index.notna()]
    if len(out) == 0:
        return out
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        out.index = out.index.tz_convert(PACIFIC)
    except Exception:
        try:
            out.index = out.index.tz_localize(PACIFIC)
        except Exception:
            pass
    return out.sort_index()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_forex_ohlc(symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
    """Fetch real trading bars only. Do not forward-fill weekend/closure gaps."""
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    df = flatten_yf_columns(df, ticker=symbol)
    needed = ["Open", "High", "Low", "Close"]
    if df.empty or any(c not in df.columns for c in needed):
        return pd.DataFrame(columns=needed + ["Volume"])
    cols = needed + (["Volume"] if "Volume" in df.columns else [])
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=needed)
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = ensure_pacific_index(out)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# =========================
# Indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    s = coerce_series(series)
    return s.ewm(span=int(span), adjust=False, min_periods=1).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    s = coerce_series(series)
    n = int(max(1, length))
    weights = np.arange(1, n + 1, dtype=float)

    def _calc(x):
        x = np.asarray(x, dtype=float)
        if len(x) != n or np.any(~np.isfinite(x)):
            return np.nan
        return float(np.dot(x, weights) / weights.sum())

    return s.rolling(n, min_periods=n).apply(_calc, raw=True)


def hma(series: pd.Series, length: int = 55) -> pd.Series:
    n = int(max(2, length))
    half = max(1, n // 2)
    root = max(1, int(math.sqrt(n)))
    return wma(2 * wma(series, half) - wma(series, n), root)


def true_range(df: pd.DataFrame) -> pd.Series:
    high = coerce_series(df["High"])
    low = coerce_series(df["Low"])
    close = coerce_series(df["Close"])
    return pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / int(period), adjust=False, min_periods=1).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = coerce_series(df["High"])
    low = coerce_series(df["Low"])
    close = coerce_series(df["Close"])

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr_val = tr.ewm(alpha=1 / int(period), adjust=False, min_periods=1).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / int(period), adjust=False, min_periods=1).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / int(period), adjust=False, min_periods=1).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / int(period), adjust=False, min_periods=1).mean()


def bollinger(series: pd.Series, window: int = 20, mult: float = 2.0):
    s = coerce_series(series)
    mid = s.rolling(window, min_periods=max(2, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(2, window // 3)).std()
    return mid, mid + mult * sd, mid - mult * sd


def regression_line(series: pd.Series, lookback: int):
    s = coerce_series(series).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float), np.nan, np.nan
    lb = int(max(2, lookback))
    ss = s.iloc[-lb:]
    x = np.arange(len(ss), dtype=float)
    y = ss.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = pd.Series(m * x + b, index=ss.index)
    ss_res = np.sum((y - yhat.to_numpy()) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return yhat, float(m), float(r2)


def rolling_regression_slope(series: pd.Series, window: int) -> pd.Series:
    s = coerce_series(series).astype(float)
    win = int(max(3, window))

    def _slope(y):
        y = pd.Series(y).dropna()
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        try:
            m, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
            return float(m)
        except Exception:
            return np.nan

    return s.rolling(win, min_periods=max(3, win // 3)).apply(_slope, raw=False)


def normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
    s = coerce_series(close).astype(float)
    slope = rolling_regression_slope(s, window)
    vol = s.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return pd.Series(np.tanh((slope * window / vol) / 2.0), index=s.index).clip(-1, 1)


def normalized_price(close: pd.Series, window: int = 60, smooth_span: int = 9) -> pd.Series:
    s = coerce_series(close).astype(float)
    mean = s.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    raw = pd.Series(np.tanh(((s - mean) / sd) / 2.0), index=s.index).clip(-1, 1)
    return raw.ewm(span=max(1, int(smooth_span)), adjust=False, min_periods=1).mean().clip(-1, 1)


def sr_reversal_index(df: pd.DataFrame, lookback: int = 60, smooth_span: int = 8) -> pd.Series:
    close = coerce_series(df["Close"])
    low = coerce_series(df["Low"])
    high = coerce_series(df["High"])
    lb = int(max(10, lookback))
    support = low.rolling(lb, min_periods=max(5, lb // 3)).min()
    resistance = high.rolling(lb, min_periods=max(5, lb // 3)).max()
    rng = (resistance - support).replace(0, np.nan)
    position = ((close - support) / rng * 2.0 - 1.0).clip(-1, 1)
    impulse = close.diff().ewm(span=max(2, smooth_span), adjust=False, min_periods=1).mean()
    impulse_scale = close.diff().abs().rolling(lb, min_periods=max(5, lb // 3)).mean().replace(0, np.nan)
    impulse_norm = np.tanh((impulse / impulse_scale) / 2.0)
    idx = (0.75 * position + 0.25 * impulse_norm).clip(-1, 1)
    return idx.ewm(span=max(1, int(smooth_span)), adjust=False, min_periods=1).mean().clip(-1, 1)


def support_resistance(df: pd.DataFrame, lookback: int = 120):
    lb = int(max(10, min(lookback, len(df))))
    if len(df) == 0:
        return np.nan, np.nan
    low = safe_float(df["Low"].iloc[-lb:].min())
    high = safe_float(df["High"].iloc[-lb:].max())
    return low, high


def recent_cross_up(a: pd.Series, b: pd.Series, lookback: int):
    aa = coerce_series(a)
    bb = coerce_series(b)
    cross = (aa.shift(1) <= bb.shift(1)) & (aa > bb)
    recent = cross.tail(int(max(1, lookback)))
    if recent.any():
        pos = np.where(recent.to_numpy())[0][-1]
        bars_since = len(recent) - 1 - pos
        ts = recent.index[pos]
        return True, int(bars_since), ts
    return False, None, None


def recent_cross_down(a: pd.Series, b: pd.Series, lookback: int):
    aa = coerce_series(a)
    bb = coerce_series(b)
    cross = (aa.shift(1) >= bb.shift(1)) & (aa < bb)
    recent = cross.tail(int(max(1, lookback)))
    if recent.any():
        pos = np.where(recent.to_numpy())[0][-1]
        bars_since = len(recent) - 1 - pos
        ts = recent.index[pos]
        return True, int(bars_since), ts
    return False, None, None


def trend_sr_reversal_signals(
    df: pd.DataFrame,
    trend_slope: float,
    sr_lookback: int,
    touch_lookback: int,
    confirm_bars: int,
    touch_tolerance_pips: float,
    cooldown: int,
    symbol: str,
):
    """Return boolean Series for trend-aligned bullish/bearish S/R reversals."""
    idx = df.index
    if df is None or df.empty:
        empty = pd.Series(False, index=idx)
        return empty, empty

    close = coerce_series(df["Close"])
    high = coerce_series(df["High"])
    low = coerce_series(df["Low"])

    lb = int(max(10, sr_lookback))
    support = low.rolling(lb, min_periods=max(5, lb // 3)).min()
    resistance = high.rolling(lb, min_periods=max(5, lb // 3)).max()
    tol = max(0.0, float(touch_tolerance_pips)) * pip_size(symbol)

    touched_support = (low <= support + tol).rolling(int(max(1, touch_lookback)), min_periods=1).max().astype(bool)
    touched_resistance = (high >= resistance - tol).rolling(int(max(1, touch_lookback)), min_periods=1).max().astype(bool)

    cb = int(max(1, confirm_bars))
    ema30 = ema(close, 30)
    hma55 = hma(close, 55)
    sr_idx = sr_reversal_index(df, sr_lookback, smooth_span=8)

    price_turn_up = (close > close.shift(cb)) & (close > support + tol) & ((close > ema30) | (close > hma55))
    price_turn_down = (close < close.shift(cb)) & (close < resistance - tol) & ((close < ema30) | (close < hma55))
    sr_turn_up = (sr_idx > sr_idx.shift(cb)) & (sr_idx.shift(cb) <= -0.50)
    sr_turn_down = (sr_idx < sr_idx.shift(cb)) & (sr_idx.shift(cb) >= 0.50)

    bullish_raw = (trend_slope >= 0) & touched_support & price_turn_up & sr_turn_up
    bearish_raw = (trend_slope < 0) & touched_resistance & price_turn_down & sr_turn_down

    bullish = apply_cooldown(bullish_raw.fillna(False), cooldown)
    bearish = apply_cooldown(bearish_raw.fillna(False), cooldown)
    return bullish, bearish


def apply_cooldown(signal: pd.Series, cooldown: int) -> pd.Series:
    cooldown = int(max(0, cooldown))
    out = pd.Series(False, index=signal.index)
    last_i = -10**9
    vals = signal.fillna(False).to_numpy(dtype=bool)
    for i, val in enumerate(vals):
        if val and i - last_i > cooldown:
            out.iloc[i] = True
            last_i = i
    return out


def bars_since_last(signal: pd.Series):
    sig = signal.fillna(False)
    if not sig.any():
        return None, None
    arr = sig.to_numpy(dtype=bool)
    pos = np.where(arr)[0][-1]
    return int(len(sig) - 1 - pos), sig.index[pos]


def prepare_indicators(df: pd.DataFrame, cfg: dict, symbol: str) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA30"] = ema(out["Close"], 30)
    out["HMA55"] = hma(out["Close"], 55)
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    out["BB_MID"], out["BB_UPPER"], out["BB_LOWER"] = bollinger(out["Close"], 20, 2.0)
    out["NTD"] = normalized_trend(out["Close"], cfg["ntd_window"])
    out["NPX"] = normalized_price(out["Close"], cfg["ntd_window"], cfg["npx_smooth"])
    out["SR_REV"] = sr_reversal_index(out, cfg["sr_lookback"], cfg["sr_smooth"])
    return out


def classify_trade(symbol: str, df: pd.DataFrame, trend_slope: float, cfg: dict) -> dict:
    if df.empty:
        return {"State": "NO DATA", "Bias": "None", "Reason": "No bars returned."}

    close = coerce_series(df["Close"])
    last = safe_float(close.iloc[-1])
    ema30_last = safe_float(df["EMA30"].iloc[-1])
    hma_last = safe_float(df["HMA55"].iloc[-1])
    sr_last = safe_float(df["SR_REV"].iloc[-1])
    ntd_last = safe_float(df["NTD"].iloc[-1])
    adx_last = safe_float(df["ADX14"].iloc[-1])
    support, resistance = support_resistance(df, cfg["sr_lookback"])

    cross_up, cross_up_bars, cross_up_time = recent_cross_up(close, df["EMA30"], cfg["cross_lookback"])
    cross_down, cross_down_bars, cross_down_time = recent_cross_down(close, df["EMA30"], cfg["cross_lookback"])

    bull_sig, bear_sig = trend_sr_reversal_signals(
        df,
        trend_slope=trend_slope,
        sr_lookback=cfg["sr_lookback"],
        touch_lookback=cfg["touch_lookback"],
        confirm_bars=cfg["confirm_bars"],
        touch_tolerance_pips=cfg["touch_tol_pips"],
        cooldown=cfg["cooldown"],
        symbol=symbol,
    )
    bull_bars, bull_time = bars_since_last(bull_sig.tail(cfg["signal_lookback"]))
    bear_bars, bear_time = bars_since_last(bear_sig.tail(cfg["signal_lookback"]))

    dist_support = last - support if np.isfinite(support) else np.nan
    dist_resistance = resistance - last if np.isfinite(resistance) else np.nan
    near_support = np.isfinite(dist_support) and dist_support <= cfg["pullback_tol_pips"] * pip_size(symbol)
    near_resistance = np.isfinite(dist_resistance) and dist_resistance <= cfg["pullback_tol_pips"] * pip_size(symbol)
    near_ema30 = abs(last - ema30_last) <= cfg["pullback_tol_pips"] * pip_size(symbol) if np.isfinite(ema30_last) else False
    near_hma55 = abs(last - hma_last) <= cfg["pullback_tol_pips"] * pip_size(symbol) if np.isfinite(hma_last) else False

    trend_up = trend_slope >= 0
    trend_down = trend_slope < 0

    if trend_up and bull_bars is not None and cross_up:
        state = "BUY CONFIRMED"
        bias = "Long"
        reason = "Uptrend + support reversal + recent 30 EMA cross upward."
    elif trend_up and (near_support or near_ema30 or near_hma55) and sr_last > -0.75:
        state = "BUY SETUP"
        bias = "Long"
        reason = "Uptrend pullback near support/EMA/HMA; wait for upward confirmation."
    elif trend_down and bear_bars is not None and cross_down:
        state = "SELL CONFIRMED"
        bias = "Short"
        reason = "Downtrend + resistance rejection + recent 30 EMA cross downward."
    elif trend_down and (near_resistance or near_ema30 or near_hma55) and sr_last < 0.75:
        state = "SELL SETUP"
        bias = "Short"
        reason = "Downtrend pullback near resistance/EMA/HMA; wait for downward confirmation."
    else:
        state = "WAIT"
        bias = "Neutral"
        reason = "No clean trend-aligned pullback/reversal confirmation."

    atr_last = safe_float(df["ATR14"].iloc[-1])
    if bias == "Long":
        entry_zone = f"{fmt_price(symbol, min(last, ema30_last))}–{fmt_price(symbol, max(last, ema30_last))}"
        stop_zone = fmt_price(symbol, min(support, last - 1.2 * atr_last) if np.isfinite(atr_last) and np.isfinite(support) else support)
        target1 = fmt_price(symbol, min(resistance, last + 1.5 * atr_last) if np.isfinite(atr_last) and np.isfinite(resistance) else resistance)
    elif bias == "Short":
        entry_zone = f"{fmt_price(symbol, min(last, ema30_last))}–{fmt_price(symbol, max(last, ema30_last))}"
        stop_zone = fmt_price(symbol, max(resistance, last + 1.2 * atr_last) if np.isfinite(atr_last) and np.isfinite(resistance) else resistance)
        target1 = fmt_price(symbol, max(support, last - 1.5 * atr_last) if np.isfinite(atr_last) and np.isfinite(support) else support)
    else:
        entry_zone = "Wait"
        stop_zone = "n/a"
        target1 = "n/a"

    return {
        "State": state,
        "Bias": bias,
        "Reason": reason,
        "Last Close": last,
        "Trend Direction": "Upward" if trend_up else "Downward",
        "Trend Slope": trend_slope,
        "S/R Reversal": sr_last,
        "NTD": ntd_last,
        "ADX": adx_last,
        "Support": support,
        "Resistance": resistance,
        "Near Support": bool(near_support),
        "Near Resistance": bool(near_resistance),
        "Near 30 EMA": bool(near_ema30),
        "Near HMA55": bool(near_hma55),
        "30 EMA Cross Up": bool(cross_up),
        "Bars Since 30 EMA Cross Up": cross_up_bars,
        "30 EMA Cross Down": bool(cross_down),
        "Bars Since 30 EMA Cross Down": cross_down_bars,
        "Bull S/R Bars": bull_bars,
        "Bear S/R Bars": bear_bars,
        "Entry Zone": entry_zone,
        "Stop / Invalidation": stop_zone,
        "Target 1": target1,
    }


# =========================
# Plotting
# =========================
def compressed_x(df: pd.DataFrame) -> np.ndarray:
    return np.arange(len(df), dtype=float)


def set_time_ticks(ax, df: pd.DataFrame, max_ticks: int = 9, right_padding: int = 8) -> None:
    if df.empty:
        return
    n = len(df)
    tick_count = min(max_ticks, n)
    if tick_count <= 1:
        ticks = [0]
    else:
        ticks = np.linspace(0, n - 1, tick_count).astype(int)
    labels = [df.index[i].strftime("%m-%d %H:%M") for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlim(-1, n - 1 + max(2, int(right_padding)))


def draw_hline_label(ax, x_last: float, y: float, text: str, color: str):
    if not np.isfinite(y):
        return
    ax.axhline(y, color=color, linewidth=1.7, alpha=0.75)
    ax.text(
        0.012,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color=color,
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
        zorder=8,
    )


def plot_forex_chart(symbol: str, df: pd.DataFrame, cfg: dict, trade: dict):
    if df.empty:
        st.warning("No chart data available.")
        return

    x = compressed_x(df)
    close = df["Close"]
    support, resistance = support_resistance(df, cfg["sr_lookback"])
    trend_line, trend_slope, trend_r2 = regression_line(close, cfg["trend_lookback"])

    bull_sig, bear_sig = trend_sr_reversal_signals(
        df,
        trend_slope=trend_slope,
        sr_lookback=cfg["sr_lookback"],
        touch_lookback=cfg["touch_lookback"],
        confirm_bars=cfg["confirm_bars"],
        touch_tolerance_pips=cfg["touch_tol_pips"],
        cooldown=cfg["cooldown"],
        symbol=symbol,
    )

    bull_recent = bull_sig.tail(cfg["signal_lookback"])
    bear_recent = bear_sig.tail(cfg["signal_lookback"])

    fig = plt.figure(figsize=(15.5, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.15], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    # Price panel
    ax.plot(x, close.to_numpy(), color="tab:blue", label="Price", linewidth=1.4)
    ax.plot(x, df["EMA20"].to_numpy(), color="tab:orange", linestyle="--", label="20 EMA", linewidth=1.35)
    ax.plot(x, df["EMA30"].to_numpy(), color="darkorange", label="30 EMA", linewidth=1.65)
    ax.plot(x, df["HMA55"].to_numpy(), color="tab:green", label="HMA(55)", linewidth=1.65)
    ax.plot(x, df["BB_MID"].to_numpy(), color="tab:red", alpha=0.8, label="BB mid", linewidth=1.15)
    ax.plot(x, df["BB_UPPER"].to_numpy(), color="tab:purple", alpha=0.35, linestyle=":", label="BB upper/lower")
    ax.plot(x, df["BB_LOWER"].to_numpy(), color="tab:brown", alpha=0.35, linestyle=":")

    if not trend_line.empty:
        trend_x = np.arange(len(df) - len(trend_line), len(df), dtype=float)
        ax.plot(
            trend_x,
            trend_line.to_numpy(),
            color="tab:green" if trend_slope >= 0 else "tab:red",
            linestyle="--",
            linewidth=2.0,
            label=f"Trend {cfg['trend_lookback']} ({trend_slope:.6f}/bar)",
        )

    draw_hline_label(ax, x[-1], resistance, f"R {fmt_price(symbol, resistance)}", "tab:red")
    draw_hline_label(ax, x[-1], support, f"S {fmt_price(symbol, support)}", "tab:green")

    # 30 EMA crosses
    cross_up = (close.shift(1) <= df["EMA30"].shift(1)) & (close > df["EMA30"])
    cross_down = (close.shift(1) >= df["EMA30"].shift(1)) & (close < df["EMA30"])
    recent_window = int(max(1, cfg["cross_lookback"]))
    up_idx = np.where(cross_up.tail(recent_window).fillna(False).to_numpy())[0]
    down_idx = np.where(cross_down.tail(recent_window).fillna(False).to_numpy())[0]
    start = len(df) - recent_window
    if len(up_idx):
        pos = start + up_idx
        ax.scatter(pos, close.iloc[pos], marker="^", s=65, color="tab:green", label="30 EMA ↑", zorder=6)
    if len(down_idx):
        pos = start + down_idx
        ax.scatter(pos, close.iloc[pos], marker="v", s=65, color="tab:red", label="30 EMA ↓", zorder=6)

    # Trend-aligned support/resistance reversal markers
    bpos = np.where(bull_recent.fillna(False).to_numpy())[0]
    rpos = np.where(bear_recent.fillna(False).to_numpy())[0]
    bstart = len(df) - len(bull_recent)
    rstart = len(df) - len(bear_recent)
    if len(bpos):
        pos = bstart + bpos
        ax.scatter(pos, df["Low"].iloc[pos] - 0.8 * pip_size(symbol), marker="^", s=120, color="limegreen",
                   edgecolor="white", linewidth=0.7, label="Trend S/R BUY", zorder=7)
    if len(rpos):
        pos = rstart + rpos
        ax.scatter(pos, df["High"].iloc[pos] + 0.8 * pip_size(symbol), marker="v", s=120, color="red",
                   edgecolor="white", linewidth=0.7, label="Trend S/R SELL", zorder=7)

    last = safe_float(close.iloc[-1])
    ema30_last = safe_float(df["EMA30"].iloc[-1])
    adx_last = safe_float(df["ADX14"].iloc[-1])
    sr_last = safe_float(df["SR_REV"].iloc[-1])
    ntd_last = safe_float(df["NTD"].iloc[-1])
    state = trade.get("State", "WAIT")
    state_color = "tab:green" if "BUY" in state else ("tab:red" if "SELL" in state else "0.35")

    title = (
        f"{symbol} Forex Trading Chart ({cfg['period']}, {cfg['interval']})  |  "
        f"{state}  |  Price {fmt_price(symbol, last)}  |  30EMA {fmt_price(symbol, ema30_last)}  |  "
        f"S/R {sr_last:+.2f}  NTD {ntd_last:+.2f}  ADX {adx_last:.1f}"
    )
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper left", ncol=2)
    ax.text(
        0.985,
        0.04,
        f"Entry: {trade.get('Entry Zone', 'Wait')} | Stop: {trade.get('Stop / Invalidation', 'n/a')} | Target: {trade.get('Target 1', 'n/a')}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=state_color,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=state_color, alpha=0.8),
    )
    ax.text(
        0.50,
        0.04,
        f"R² {trend_r2:.1%}" if np.isfinite(trend_r2) else "R² n/a",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.75),
    )

    # Indicator panel
    ax2.axhline(0, color="0.25", linestyle="--", linewidth=1.0)
    ax2.axhline(0.75, color="tab:green", linewidth=2.3)
    ax2.axhline(-0.75, color="tab:red", linewidth=2.3)
    ax2.axhline(0.50, color="0.25", linewidth=1.0)
    ax2.axhline(-0.50, color="0.25", linewidth=1.0)

    ax2.fill_between(x, 0, df["NTD"].clip(lower=0).to_numpy(), color="tab:green", alpha=0.10)
    ax2.fill_between(x, 0, df["NTD"].clip(upper=0).to_numpy(), color="tab:red", alpha=0.10)
    ax2.plot(x, df["NTD"].to_numpy(), color="tab:blue", label="NTD")
    ax2.plot(x, df["NPX"].to_numpy(), color="0.45", alpha=0.70, label="Smoothed NPX")
    ax2.plot(x, df["SR_REV"].to_numpy(), color="tab:purple", alpha=0.85, label="S/R Reversal Index")

    if len(bpos):
        pos = bstart + bpos
        ax2.scatter(pos, df["SR_REV"].iloc[pos], marker="s", s=75, color="tab:green", label="Trend S/R BUY", zorder=6)
    if len(rpos):
        pos = rstart + rpos
        ax2.scatter(pos, df["SR_REV"].iloc[pos], marker="v", s=85, color="tab:red", label="Trend S/R SELL", zorder=6)

    ntd_trend_line, ntd_slope, _ = regression_line(df["NTD"], min(cfg["trend_lookback"], max(5, len(df)//2)))
    if not ntd_trend_line.empty:
        tx = np.arange(len(df) - len(ntd_trend_line), len(df), dtype=float)
        ax2.plot(tx, ntd_trend_line.to_numpy(), color="darkorange", linestyle="--",
                 linewidth=2.0, label=f"NTD trend ({ntd_slope:.4f}/bar)")

    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title("Hourly Indicator Panel — NTD + Smoothed NPX + S/R Reversal")
    ax2.grid(True)
    ax2.legend(loc="upper left", ncol=2)
    ax2.set_xlabel("Compressed real trading bars — no weekend/closure gap")

    set_time_ticks(ax2, df, max_ticks=9, right_padding=cfg["right_padding"])
    set_time_ticks(ax, df, max_ticks=9, right_padding=cfg["right_padding"])
    plt.setp(ax.get_xticklabels(), visible=False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def format_trade_row(symbol: str, trade: dict) -> dict:
    return {
        "Symbol": symbol,
        "State": trade.get("State"),
        "Bias": trade.get("Bias"),
        "Trend": trade.get("Trend Direction"),
        "Trend Slope": round(safe_float(trade.get("Trend Slope")), 7),
        "S/R Rev": round(safe_float(trade.get("S/R Reversal")), 3),
        "NTD": round(safe_float(trade.get("NTD")), 3),
        "ADX": round(safe_float(trade.get("ADX")), 1),
        "Last Close": fmt_price(symbol, trade.get("Last Close")),
        "Support": fmt_price(symbol, trade.get("Support")),
        "Resistance": fmt_price(symbol, trade.get("Resistance")),
        "Near Support": trade.get("Near Support"),
        "Near Resistance": trade.get("Near Resistance"),
        "Near 30 EMA": trade.get("Near 30 EMA"),
        "Near HMA55": trade.get("Near HMA55"),
        "Bars Since 30 EMA Up": trade.get("Bars Since 30 EMA Cross Up"),
        "Bars Since 30 EMA Down": trade.get("Bars Since 30 EMA Cross Down"),
        "Bull S/R Bars": trade.get("Bull S/R Bars"),
        "Bear S/R Bars": trade.get("Bear S/R Bars"),
        "Entry Zone": trade.get("Entry Zone"),
        "Stop / Invalidation": trade.get("Stop / Invalidation"),
        "Target 1": trade.get("Target 1"),
        "Reason": trade.get("Reason"),
    }


def scan_symbol(symbol: str, cfg: dict):
    df = fetch_forex_ohlc(symbol, cfg["period"], cfg["interval"])
    if df.empty or len(df) < max(60, cfg["trend_lookback"] // 2):
        return None
    df = prepare_indicators(df, cfg, symbol)
    _, trend_slope, _ = regression_line(df["Close"], cfg["trend_lookback"])
    trade = classify_trade(symbol, df, trend_slope, cfg)
    return format_trade_row(symbol, trade)


# =========================
# Sidebar
# =========================
auto_refresh()
last_refresh = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
remaining = int(max(0, REFRESH_INTERVAL - (time.time() - st.session_state.last_refresh)))
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {last_refresh.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next refresh:** ~{remaining}s"
)

st.sidebar.title("Forex Hourly Indicator")
symbol = st.sidebar.selectbox("Forex pair", FOREX_UNIVERSE, index=FOREX_UNIVERSE.index("EURUSD=X"))
period = st.sidebar.selectbox("Chart period", ["1d", "2d", "5d", "7d", "10d", "30d"], index=2)
interval = st.sidebar.selectbox("Bar interval", ["5m", "15m", "30m", "60m"], index=1)

st.sidebar.subheader("Trend / Reversal Logic")
trend_lookback = st.sidebar.slider("Trendline lookback (bars)", 30, 360, 120, 5)
sr_lookback = st.sidebar.slider("Support/resistance lookback (bars)", 30, 360, 120, 5)
ntd_window = st.sidebar.slider("NTD window", 20, 180, 60, 5)
touch_lookback = st.sidebar.slider("Touch lookback (bars)", 1, 30, 8, 1)
confirm_bars = st.sidebar.slider("Confirmation bars", 1, 8, 3, 1)
signal_lookback = st.sidebar.slider("Recent signal window (bars)", 5, 120, 40, 5)
cross_lookback = st.sidebar.slider("Recent 30 EMA cross window (bars)", 3, 120, 30, 1)
touch_tol_pips = st.sidebar.slider("S/R touch tolerance (pips)", 0.0, 20.0, 3.0, 0.5)
pullback_tol_pips = st.sidebar.slider("Pullback proximity tolerance (pips)", 1.0, 40.0, 8.0, 0.5)
cooldown = st.sidebar.slider("Marker cooldown (bars)", 0, 60, 10, 1)

st.sidebar.subheader("Display")
npx_smooth = st.sidebar.slider("NPX smoothing", 1, 30, 9, 1)
sr_smooth = st.sidebar.slider("S/R index smoothing", 1, 30, 8, 1)
right_padding = st.sidebar.slider("Right-edge chart padding", 2, 40, 12, 1)

cfg = {
    "period": period,
    "interval": interval,
    "trend_lookback": trend_lookback,
    "sr_lookback": sr_lookback,
    "ntd_window": ntd_window,
    "npx_smooth": npx_smooth,
    "sr_smooth": sr_smooth,
    "touch_lookback": touch_lookback,
    "confirm_bars": confirm_bars,
    "signal_lookback": signal_lookback,
    "cross_lookback": cross_lookback,
    "touch_tol_pips": touch_tol_pips,
    "pullback_tol_pips": pullback_tol_pips,
    "cooldown": cooldown,
    "right_padding": right_padding,
}


# =========================
# Main app
# =========================
st.title("💱 Forex Hourly Trading Indicator")
st.caption(
    "Forex-only trading chart using real trading bars only. Weekend and market-closure gaps are removed visually by plotting compressed bar positions."
)

tab_chart, tab_scanner, tab_rules = st.tabs(["Hourly Trading Chart", "Forex Scanner", "Trading Rules"])

with tab_chart:
    data = fetch_forex_ohlc(symbol, period, interval)
    if data.empty:
        st.error("No data returned by yfinance for this pair/period/interval.")
    else:
        data = prepare_indicators(data, cfg, symbol)
        trend_line, trend_slope, trend_r2 = regression_line(data["Close"], trend_lookback)
        trade = classify_trade(symbol, data, trend_slope, cfg)

        state = trade.get("State", "WAIT")
        css = "signal-buy" if "BUY" in state else ("signal-sell" if "SELL" in state else "signal-wait")
        st.markdown(
            f"""
<div class="{css}">
{state} — {trade.get('Reason', '')}<br>
Entry Zone: {trade.get('Entry Zone', 'Wait')} &nbsp; | &nbsp;
Stop / Invalidation: {trade.get('Stop / Invalidation', 'n/a')} &nbsp; | &nbsp;
Target 1: {trade.get('Target 1', 'n/a')}
</div>
""",
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(6)
        metric_cols[0].metric("Last Close", fmt_price(symbol, trade.get("Last Close")))
        metric_cols[1].metric("Trend", trade.get("Trend Direction"), f"{safe_float(trade.get('Trend Slope')):.7f}/bar")
        metric_cols[2].metric("S/R Reversal", f"{safe_float(trade.get('S/R Reversal')):+.2f}")
        metric_cols[3].metric("NTD", f"{safe_float(trade.get('NTD')):+.2f}")
        metric_cols[4].metric("ADX", f"{safe_float(trade.get('ADX')):.1f}")
        metric_cols[5].metric("30 EMA Pullback", "Yes" if trade.get("Near 30 EMA") else "No")

        plot_forex_chart(symbol, data, cfg, trade)

        with st.expander("Current setup details", expanded=False):
            st.dataframe(pd.DataFrame([format_trade_row(symbol, trade)]), use_container_width=True, hide_index=True)

with tab_scanner:
    st.subheader("Forex Scanner")
    st.caption(
        "Scans the forex universe for trend-aligned support/resistance reversals, 30 EMA crosses, and pullbacks. "
        "BUY/SELL CONFIRMED rows are strongest; SETUP rows require confirmation."
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    scan_max = int(c1.number_input("Max rows per table", min_value=5, max_value=200, value=100, step=5))
    scan_universe = c2.multiselect("Pairs to scan", FOREX_UNIVERSE, default=FOREX_UNIVERSE)
    show_all = c3.checkbox("Also show WAIT rows", value=False)

    if st.button("Run Forex Scan", use_container_width=True):
        rows = []
        progress = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(scan_universe):
            status.write(f"Scanning {sym}...")
            try:
                row = scan_symbol(sym, cfg)
                if row is not None:
                    rows.append(row)
            except Exception as exc:
                rows.append({"Symbol": sym, "State": "ERROR", "Reason": str(exc)})
            progress.progress((i + 1) / max(1, len(scan_universe)))
        status.empty()

        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No scanner results.")
        else:
            state_order = {
                "BUY CONFIRMED": 0,
                "SELL CONFIRMED": 1,
                "BUY SETUP": 2,
                "SELL SETUP": 3,
                "WAIT": 4,
                "ERROR": 5,
            }
            out["_StateOrder"] = out["State"].map(state_order).fillna(9)
            out["_BullSort"] = pd.to_numeric(out.get("Bull S/R Bars"), errors="coerce").fillna(9999)
            out["_BearSort"] = pd.to_numeric(out.get("Bear S/R Bars"), errors="coerce").fillna(9999)
            out["_EmaUpSort"] = pd.to_numeric(out.get("Bars Since 30 EMA Up"), errors="coerce").fillna(9999)
            out["_EmaDownSort"] = pd.to_numeric(out.get("Bars Since 30 EMA Down"), errors="coerce").fillna(9999)

            buy_df = out[out["State"].isin(["BUY CONFIRMED", "BUY SETUP"])].sort_values(
                ["_StateOrder", "_BullSort", "_EmaUpSort", "ADX", "Symbol"],
                ascending=[True, True, True, False, True],
            ).drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").head(scan_max)

            sell_df = out[out["State"].isin(["SELL CONFIRMED", "SELL SETUP"])].sort_values(
                ["_StateOrder", "_BearSort", "_EmaDownSort", "ADX", "Symbol"],
                ascending=[True, True, True, False, True],
            ).drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").head(scan_max)

            ema_up_df = out[out["Bars Since 30 EMA Up"].notna()].sort_values(
                ["_EmaUpSort", "ADX", "Symbol"], ascending=[True, False, True]
            ).drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").head(scan_max)

            ema_down_df = out[out["Bars Since 30 EMA Down"].notna()].sort_values(
                ["_EmaDownSort", "ADX", "Symbol"], ascending=[True, False, True]
            ).drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore").head(scan_max)

            st.subheader("Long Watchlist — trend-aligned support/EMA pullbacks")
            if buy_df.empty:
                st.info("No long setups found.")
            else:
                st.dataframe(buy_df, use_container_width=True, hide_index=True)

            st.subheader("Short Watchlist — trend-aligned resistance/EMA pullbacks")
            if sell_df.empty:
                st.info("No short setups found.")
            else:
                st.dataframe(sell_df, use_container_width=True, hide_index=True)

            st.subheader("Recent 30 EMA Cross Up")
            if ema_up_df.empty:
                st.info("No recent 30 EMA upward crosses found.")
            else:
                st.dataframe(ema_up_df, use_container_width=True, hide_index=True)

            st.subheader("Recent 30 EMA Cross Down")
            if ema_down_df.empty:
                st.info("No recent 30 EMA downward crosses found.")
            else:
                st.dataframe(ema_down_df, use_container_width=True, hide_index=True)

            if show_all:
                st.subheader("All scan rows")
                all_df = out.sort_values(["_StateOrder", "Symbol"]).drop(
                    columns=[c for c in out.columns if c.startswith("_")], errors="ignore"
                )
                st.dataframe(all_df, use_container_width=True, hide_index=True)

with tab_rules:
    st.subheader("How to use this chart")
    st.markdown(
        """
**BUY CONFIRMED** means the chart has an upward trendline, price recently reversed from support,
and price recently crossed above the 30 EMA. This is the cleanest long condition.

**BUY SETUP** means price is pulling back near support, 30 EMA, or HMA while the trendline is upward.
Wait for a bounce, a green marker, or a 30 EMA reclaim before entry.

**SELL CONFIRMED** means the chart has a downward trendline, price recently rejected resistance,
and price recently crossed below the 30 EMA. This is the cleanest short condition.

**SELL SETUP** means price is pulling back near resistance, 30 EMA, or HMA while the trendline is downward.
Wait for rejection or a 30 EMA loss before entry.

**WAIT** means the chart is not aligned enough. Do not force a trade.
"""
    )
    st.info(
        "Risk rule: use the support/resistance stop zone as invalidation. Do not take a signal if the reward to the next target is smaller than the stop distance."
    )
