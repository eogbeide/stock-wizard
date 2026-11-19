# forex_hourly_trading_indicator.py
# Clean Stocks/FX Streamlit dashboard with hourly and daily trading indicators.
# Focus: compressed intraday chart (no weekend/closure gaps), trend-aligned support/resistance reversals,
# 30 EMA crosses, NTD/S/R reversal confirmation, Stocks/FX scanner,
# and chart-level trade plans for easier BUY/SELL decision-making.
# (UPDATED) Scores are no longer shown as probabilities: each signal shows its measured past record
# (bullbears_history.csv, built by bullbears_history_build.py), targets and stops sit 1.5 ATR either
# side of price, and a Forex Stochastic Dip tab shows the tested daily Stochastic <= 10 buy rule.
# (UPDATED) Structure metric now uses directional red/green trend badges and action/trend alignment text.
# (UPDATED) Chart trade instruction box is moved outside the plot area so it does not block candles/indicators.
# (UPDATED) Symbol lists and scanner tables are alphabetized; Buy/Sell List is split into Daily and Hourly tables.

import math
import time
from datetime import datetime
from pathlib import Path

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
# Target and stop both sit this many ATRs from price, so a hit rate above 50% means a real edge.
PLAN_ATR = 1.5
# Forex dip rule tested in Stock Wizard: buy when daily Stochastic %K(14) <= 10,
# sell when RSI(14) reaches 50 or after 60 days.
FX_STOCH_LOW = 10
FX_STOCH_EXIT_RSI = 50
FX_STOCH_MAX_DAYS = 60
HISTORY_FILE = Path(__file__).with_name("bullbears_history.csv")

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

.prob-guide {
    margin: 0.35rem 0 0.75rem 0;
    padding: 0.75rem 0.85rem;
    border-radius: 0.75rem;
    border: 1px solid rgba(90, 90, 90, 0.22);
    background: rgba(245, 245, 245, 0.70);
    font-size: 0.94rem;
    line-height: 1.35;
}
.prob-buy {
    border-left: 6px solid rgba(0, 150, 80, 0.80);
}
.prob-sell {
    border-left: 6px solid rgba(220, 40, 40, 0.80);
}
.prob-watch {
    border-left: 6px solid rgba(230, 170, 20, 0.90);
}
.prob-wait {
    border-left: 6px solid rgba(130, 130, 130, 0.90);
}
.prob-mini {
    font-size: 0.84rem;
    color: #555;
}


.trend-badge {
    display: inline-block;
    margin-top: 0.25rem;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    line-height: 1.15;
    white-space: nowrap;
}
.trend-badge-up {
    color: #0f7a3a;
    background: rgba(0, 150, 80, 0.13);
    border: 1px solid rgba(0, 150, 80, 0.30);
}
.trend-badge-down {
    color: #b42318;
    background: rgba(220, 40, 40, 0.13);
    border: 1px solid rgba(220, 40, 40, 0.30);
}
.trend-badge-flat {
    color: #6b7280;
    background: rgba(120, 120, 120, 0.13);
    border: 1px solid rgba(120, 120, 120, 0.30);
}
.alignment-badge {
    display: inline-block;
    margin-top: 0.18rem;
    padding: 0.20rem 0.50rem;
    border-radius: 0.55rem;
    font-size: 0.74rem;
    font-weight: 650;
}
.alignment-ok {
    color: #0f7a3a;
    background: rgba(0, 150, 80, 0.10);
}
.alignment-caution {
    color: #b26a00;
    background: rgba(230, 170, 20, 0.12);
}
.alignment-wait {
    color: #6b7280;
    background: rgba(120, 120, 120, 0.12);
}


/* Multi-row navigation: all sections stay visible without horizontal tab scrolling. */
div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
    display: flex !important;
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    white-space: normal !important;
    gap: 0.25rem 0.45rem !important;
    height: auto !important;
    max-width: 100% !important;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    padding: 0.45rem 0.65rem !important;
    white-space: nowrap !important;
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

STOCK_UNIVERSE = sorted([
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AMD", "AVGO",
    "NFLX", "PLTR", "HOOD", "MARA", "SMCI", "ORCL", "IBM", "JPM", "BAC", "GS", "SITM",
    "SPY", "QQQ", "DIA", "IWM", "VOO", "TLT", "GLD", "SLV", "USO", "GUSH",
    "TSM", "BABA", "AAL", "DAL", "URI", "MP", "QUBT", "BBAI", "SPGI", "VTWG",
])

ASSET_UNIVERSES = {
    "Forex": sorted(FOREX_UNIVERSE),
    "Stocks": sorted(STOCK_UNIVERSE),
}

def _sort_symbols_alpha(symbols) -> list[str]:
    """Return symbols sorted alphabetically, ignoring None/blank values."""
    try:
        return sorted([str(s) for s in symbols if str(s).strip()])
    except Exception:
        return []


def _sort_table_alphabetically(df: pd.DataFrame, timeframe_first: bool = True) -> pd.DataFrame:
    """Sort display tables by symbol A-Z, optionally keeping Daily rows before Hourly rows."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    sort_cols = []
    ascending = []
    if timeframe_first and "Timeframe" in out.columns:
        timeframe_order = {"Daily": 0, "Hourly": 1}
        out["_TimeframeAlphaOrder"] = out["Timeframe"].map(timeframe_order).fillna(9)
        sort_cols.append("_TimeframeAlphaOrder")
        ascending.append(True)
    if "Symbol" in out.columns:
        sort_cols.append("Symbol")
        ascending.append(True)
    if "State" in out.columns:
        state_order = {
            "BUY CONFIRMED": 0, "SELL CONFIRMED": 0,
            "BUY SETUP": 1, "SELL SETUP": 1,
            "WAIT": 4, "ERROR": 9,
        }
        out["_StateAlphaOrder"] = out["State"].map(state_order).fillna(8)
        sort_cols.append("_StateAlphaOrder")
        ascending.append(True)
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return out.drop(columns=[c for c in out.columns if str(c).startswith("_")], errors="ignore")


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
    """Small display/trading increment.

    FX pairs use conventional pip size; stocks/ETFs use cents. The app uses
    this as a default tolerance unit for support/resistance and pullback checks.
    """
    s = str(symbol).upper()
    if "=X" not in s:
        return 0.01
    return 0.01 if "JPY" in s else 0.0001


def fmt_price(symbol: str, value) -> str:
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    s = str(symbol).upper()
    if "=X" in s:
        return f"{v:.3f}" if "JPY" in s else f"{v:.5f}"
    return f"${v:,.2f}"


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
def fetch_market_ohlc(symbol: str, period: str = "5d", interval: str = "15m") -> pd.DataFrame:
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


def compute_trade_probabilities_and_instruction(symbol: str,
                                                state: str,
                                                bias: str,
                                                last: float,
                                                ema20_last: float,
                                                ema30_last: float,
                                                hma_last: float,
                                                sr_last: float,
                                                ntd_last: float,
                                                adx_last: float,
                                                trend_slope: float,
                                                support: float,
                                                resistance: float,
                                                near_support: bool,
                                                near_resistance: bool,
                                                near_ema30: bool,
                                                near_hma55: bool,
                                                cross_up: bool,
                                                cross_down: bool,
                                                entry_zone: str,
                                                stop_zone: str,
                                                target1: str,
                                                current_structure: str = "Mixed",
                                                sr_delta: float = np.nan,
                                                ntd_delta: float = np.nan) -> dict:
    """
    Convert chart conditions into actionable long/short probability estimates.

    Important: the probability model separates the long-term regression slope from
    the current trade structure. A slightly positive slope should not make a chart
    look bullish when price is trading below the 20 EMA, 30 EMA, and HMA.
    """
    trend_up = bool(np.isfinite(trend_slope) and trend_slope >= 0)
    trend_down = bool(np.isfinite(trend_slope) and trend_slope < 0)

    price_above_ema20 = bool(np.isfinite(last) and np.isfinite(ema20_last) and last >= ema20_last)
    price_below_ema20 = bool(np.isfinite(last) and np.isfinite(ema20_last) and last < ema20_last)
    price_above_ema30 = bool(np.isfinite(last) and np.isfinite(ema30_last) and last >= ema30_last)
    price_below_ema30 = bool(np.isfinite(last) and np.isfinite(ema30_last) and last < ema30_last)
    price_above_hma = bool(np.isfinite(last) and np.isfinite(hma_last) and last >= hma_last)
    price_below_hma = bool(np.isfinite(last) and np.isfinite(hma_last) and last < hma_last)

    bullish_structure = bool(str(current_structure).startswith("Bullish"))
    bearish_structure = bool(str(current_structure).startswith("Bearish"))
    sr_rising = bool(np.isfinite(sr_delta) and sr_delta > 0)
    sr_falling = bool(np.isfinite(sr_delta) and sr_delta < 0)
    ntd_rising = bool(np.isfinite(ntd_delta) and ntd_delta > 0)
    ntd_falling = bool(np.isfinite(ntd_delta) and ntd_delta < 0)

    strong_adx = bool(np.isfinite(adx_last) and adx_last >= 20)
    very_strong_adx = bool(np.isfinite(adx_last) and adx_last >= 25)

    long_score = 10.0
    short_score = 10.0

    # Long-term slope is useful, but it is intentionally a smaller input than
    # current price structure.
    if trend_up:
        long_score += 10
    elif trend_down:
        short_score += 10

    # Current structure is the most important part of the trade read.
    if bullish_structure:
        long_score += 24
        short_score -= 4
    elif bearish_structure:
        short_score += 24
        long_score -= 4
    else:
        long_score += 4
        short_score += 4

    if "BUY CONFIRMED" in str(state):
        long_score += 24
    elif "BUY SETUP" in str(state):
        long_score += 14
    elif "BUY WATCH" in str(state):
        long_score += 8

    if "SELL CONFIRMED" in str(state):
        short_score += 24
    elif "SELL SETUP" in str(state):
        short_score += 14
    elif "SELL WATCH" in str(state):
        short_score += 8

    # EMA/HMA structure. Being below all three should meaningfully reduce long
    # confidence even if the regression slope is slightly positive.
    if price_above_ema20:
        long_score += 6
    if price_below_ema20:
        short_score += 6
        long_score -= 2
    if price_above_ema30:
        long_score += 9
    if price_below_ema30:
        short_score += 9
        long_score -= 3
    if price_above_hma:
        long_score += 8
    if price_below_hma:
        short_score += 8
        long_score -= 3

    if bool(cross_up):
        long_score += 8
    if bool(cross_down):
        short_score += 8

    if np.isfinite(sr_last):
        if sr_last > 0:
            long_score += 7
        if sr_last < 0:
            short_score += 7
        if sr_rising:
            long_score += 5
        if sr_falling:
            short_score += 5
        if sr_last <= -0.50 and (trend_up or bullish_structure):
            long_score += 4  # dip in an otherwise constructive setup
        if sr_last >= 0.50 and (trend_down or bearish_structure):
            short_score += 4  # bounce into resistance in a weak setup

    if np.isfinite(ntd_last):
        if ntd_last > 0:
            long_score += 5
        if ntd_last < 0:
            short_score += 5
        if ntd_rising:
            long_score += 5
        if ntd_falling:
            short_score += 5
        # Penalize longs when NTD is rolling down from an elevated area.
        if ntd_last > 0.25 and ntd_falling and bearish_structure:
            short_score += 4
            long_score -= 4

    if near_support:
        long_score += 5
    if near_resistance:
        short_score += 5
    if near_ema30 or near_hma55:
        long_score += 2
        short_score += 2

    if strong_adx:
        if bullish_structure:
            long_score += 5
        elif bearish_structure:
            short_score += 5
        elif trend_up:
            long_score += 3
        elif trend_down:
            short_score += 3
    if very_strong_adx:
        if bullish_structure:
            long_score += 3
        elif bearish_structure:
            short_score += 3

    # Final safety gates: avoid showing a confident BUY when price is below the
    # main moving-average cluster; avoid showing a confident SELL when price is
    # above it.
    if bearish_structure:
        long_score = min(long_score, 62)
    if bullish_structure:
        short_score = min(short_score, 62)

    long_prob = float(np.clip(long_score, 5, 95))
    short_prob = float(np.clip(short_score, 5, 95))
    confidence = abs(long_prob - short_prob)

    if long_prob >= short_prob + 10 and long_prob >= 70 and bullish_structure:
        action = "BUY"
        instruction = (
            f"BUY/Long: enter near {entry_zone}; stop/invalidate below {stop_zone}; "
            f"first target {target1}. Best when price holds above 30 EMA/HMA after confirmation."
        )
    elif short_prob >= long_prob + 10 and short_prob >= 70 and bearish_structure:
        action = "SELL"
        instruction = (
            f"SELL/Short: enter near {entry_zone}; stop/invalidate above {stop_zone}; "
            f"first target {target1}. Best after rejection from resistance or failure below 30 EMA/HMA."
        )
    elif long_prob >= short_prob + 6 and long_prob >= 58 and not bearish_structure:
        action = "BUY WATCH"
        instruction = (
            f"BUY WATCH: wait for price to reclaim/hold 30 EMA and HMA. Ideal entry zone {entry_zone}; "
            f"invalidation {stop_zone}; target {target1}."
        )
    elif short_prob >= long_prob + 6 and short_prob >= 58 and not bullish_structure:
        action = "SELL WATCH"
        instruction = (
            f"SELL WATCH: current structure is weak. Look for rejection/failure near 30 EMA/HMA or resistance. "
            f"Ideal entry zone {entry_zone}; invalidation {stop_zone}; target {target1}."
        )
    elif bearish_structure and long_prob >= short_prob:
        action = "WAIT / SELL WATCH"
        instruction = (
            "WAIT / SELL WATCH: price is below the EMA/HMA structure, so avoid buying until price reclaims "
            "30 EMA and HMA. Watch for a rejection continuation lower."
        )
    elif bullish_structure and short_prob > long_prob:
        action = "WAIT / BUY WATCH"
        instruction = (
            "WAIT / BUY WATCH: price is above the EMA/HMA structure, so avoid shorting until price loses "
            "30 EMA and HMA. Watch for a pullback hold."
        )
    else:
        action = "WAIT"
        instruction = (
            "WAIT: long/short scores are not sufficiently separated or current structure is mixed. "
            "Avoid forcing a trade until price confirms direction."
        )

    return {
        "Trade Action": action,
        "Trade Instruction": instruction,
        "Long Score": long_prob,
        "Short Score": short_prob,
        "Score Edge": confidence,
    }


def probability_trade_read(trade: dict) -> dict:
    """Translate long/short probabilities into a simple trade plan users can read quickly."""
    long_prob = safe_float(trade.get("Long Score"))
    short_prob = safe_float(trade.get("Short Score"))
    edge = safe_float(trade.get("Score Edge"))
    action = str(trade.get("Trade Action", "WAIT") or "WAIT").upper()
    state = str(trade.get("State", "WAIT") or "WAIT").upper()

    if long_prob >= 75 and edge >= 12 and long_prob > short_prob:
        symbol_icon = "🟢▲"
        css = "prob-guide prob-buy"
        read = "BUY / Long favored"
        usage = "Long score is clearly ahead. Check the past record below before relying on it."
    elif short_prob >= 75 and edge >= 12 and short_prob > long_prob:
        symbol_icon = "🔴▼"
        css = "prob-guide prob-sell"
        read = "SELL / Short favored"
        usage = "Short score is clearly ahead. Check the past record below before relying on it."
    elif long_prob >= 60 and long_prob >= short_prob and edge >= 5:
        symbol_icon = "🟡▲"
        css = "prob-guide prob-watch"
        read = "BUY Watch"
        usage = "Long side is improving, but confirmation is still needed. Avoid chasing; wait for support/EMA hold."
    elif short_prob >= 60 and short_prob > long_prob and edge >= 5:
        symbol_icon = "🟡▼"
        css = "prob-guide prob-watch"
        read = "SELL Watch"
        usage = "Short side is improving, but confirmation is still needed. Avoid chasing; wait for resistance/EMA rejection."
    else:
        symbol_icon = "⏳"
        css = "prob-guide prob-wait"
        read = "WAIT"
        usage = "Probabilities are mixed or too close. Do not force a trade until one side reaches a clearer edge."

    if edge < 8:
        risk_note = "Small score gap: watchlist only."
    elif edge < 15:
        risk_note = "Moderate score gap: wait for price confirmation."
    else:
        risk_note = "Use the listed stop."

    if "CONFIRMED" in state:
        confirmation = "Confirmed setup: the score agrees with a completed chart signal."
    elif "SETUP" in state or "WATCH" in action:
        confirmation = "Setup/watch: wait for the next candle confirmation before entry."
    else:
        confirmation = "No confirmed setup yet."

    return {
        "Icon": symbol_icon,
        "Read": read,
        "CSS": css,
        "Usage": usage,
        "Risk Note": risk_note,
        "Confirmation": confirmation,
        "Long Score": long_prob,
        "Short Score": short_prob,
        "Score Edge": edge,
    }


def render_probability_action_panel(trade: dict, view_label: str = "Chart") -> None:
    """Compact trade plan shown above each chart: past record first, rule score second."""
    read = probability_trade_read(trade)
    plan_note = ("WATCH-ONLY hypothetical levels; not a trade signal. "
                 if trade.get("Plan Indicative", False) else "")
    st.markdown(
        f"""
<div class="{read['CSS']}">
  <b>{read['Icon']} {view_label} Trade Plan: {read['Read']}</b><br>
  <b>{trade.get('History Text', 'No measured history for this signal yet.')}</b><br>
  <span class="prob-mini">Rule score (points out of 95, not a probability): Long {read['Long Score']:.0f}
  &nbsp;|&nbsp; Short {read['Short Score']:.0f} &nbsp;|&nbsp; Gap {read['Score Edge']:.0f}</span><br>
  <span class="prob-mini">{read['Usage']} {read['Confirmation']} {read['Risk Note']}</span><br>
  <span class="prob-mini">{plan_note}Entry: <b>{trade.get('Entry Zone', 'Wait')}</b> &nbsp; | &nbsp;
  Stop/Invalidation: <b>{trade.get('Stop / Invalidation', 'n/a')}</b> &nbsp; | &nbsp;
  Target: <b>{trade.get('Target 1', 'n/a')}</b></span>
</div>
""",
        unsafe_allow_html=True,
    )


def probability_bar_values(trade: dict) -> tuple[float, float, float]:
    """Return sanitized long/short/edge values for chart annotation."""
    long_prob = float(np.clip(safe_float(trade.get("Long Score")), 0, 100))
    short_prob = float(np.clip(safe_float(trade.get("Short Score")), 0, 100))
    edge = float(abs(long_prob - short_prob))
    return long_prob, short_prob, edge


def trade_trend_alignment(trade: dict) -> dict:
    """Describe whether the trade action and current trend direction agree.

    This keeps the UI from showing a bearish structure with a green/up-looking
    trend badge, which can make SELL setups look bullish.
    """
    action = str(trade.get("Trade Action", "WAIT") or "WAIT").upper()
    trend_dir = str(trade.get("Trend Direction", "Mixed") or "Mixed")
    slope = safe_float(trade.get("Trend Slope"))
    structure = str(trade.get("Current Structure", "") or "")

    if np.isfinite(slope):
        if slope > 0:
            icon = "↑"
            trend_class = "trend-badge-up"
            slope_dir = "Upward"
        elif slope < 0:
            icon = "↓"
            trend_class = "trend-badge-down"
            slope_dir = "Downward"
        else:
            icon = "→"
            trend_class = "trend-badge-flat"
            slope_dir = "Flat"
    else:
        icon = "→"
        trend_class = "trend-badge-flat"
        slope_dir = "Mixed"

    # Prefer the numeric slope direction over a stale text value if they differ.
    if slope_dir in ("Upward", "Downward", "Flat"):
        trend_dir_display = slope_dir
    else:
        trend_dir_display = trend_dir

    if "BUY" in action and trend_dir_display == "Upward":
        align_text = "BUY aligned with upward trend"
        align_class = "alignment-ok"
    elif "SELL" in action and trend_dir_display == "Downward":
        align_text = "SELL aligned with downward trend"
        align_class = "alignment-ok"
    elif action == "WAIT" or "WAIT" in action:
        align_text = "WAIT until trend and structure align"
        align_class = "alignment-wait"
    elif "BUY" in action and trend_dir_display == "Downward":
        align_text = "Caution: BUY conflicts with downward trend"
        align_class = "alignment-caution"
    elif "SELL" in action and trend_dir_display == "Upward":
        align_text = "Caution: SELL conflicts with upward trend"
        align_class = "alignment-caution"
    else:
        align_text = "Check chart confirmation"
        align_class = "alignment-caution"

    return {
        "Icon": icon,
        "Trend Direction": trend_dir_display,
        "Trend Class": trend_class,
        "Alignment Text": align_text,
        "Alignment Class": align_class,
        "Trend Slope": slope,
        "Structure": structure,
    }


def render_structure_trend_badge(container, trade: dict) -> None:
    """Render trend/structure alignment without Streamlit's green metric delta arrow."""
    info = trade_trend_alignment(trade)
    slope = info["Trend Slope"]
    slope_txt = f"{slope:.7f}/bar" if np.isfinite(slope) else "n/a"
    container.markdown(
        f'''
<div class="trend-badge {info['Trend Class']}">
  {info['Icon']} Trend {info['Trend Direction']} • {slope_txt}
</div><br>
<div class="alignment-badge {info['Alignment Class']}">
  {info['Alignment Text']}
</div>
''',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_signal_history() -> pd.DataFrame:
    """Measured past results for each signal (see bullbears_history_build.py)."""
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()


def history_timeframe(cfg: dict) -> str:
    return "Daily" if str(cfg.get("interval", "")).lower() in ("1d", "1wk", "1mo") else "Intraday"


def signal_history(symbol: str, cfg: dict, state: str) -> dict | None:
    """Past record of this state for the symbol's market and timeframe, with the any-bar baseline."""
    hist = load_signal_history()
    if hist.empty:
        return None
    market = "Forex" if "=X" in str(symbol).upper() else "Stocks"
    tf = history_timeframe(cfg)
    pick = hist[(hist["Market"] == market) & (hist["Timeframe"] == tf)]
    row = pick[pick["State"] == state]
    if row.empty:
        return None
    out = row.iloc[0].to_dict()
    side = "Long" if "BUY" in state else "Short"
    base = pick[pick["State"] == f"ANY {side.upper()}"]
    if not base.empty:
        out["Baseline Target First %"] = float(base.iloc[0]["Target First %"])
        out["Baseline Avg Result %"] = float(base.iloc[0]["Avg Result %"])
    return out


def history_text(hist: dict | None, cfg: dict | None = None) -> str:
    """One-line plain description of a signal's past record."""
    if not hist:
        return "No measured history for this signal yet."
    text = (
        f"Past {hist['State']} signals ({hist['Market']}, {hist['Timeframe'].lower()}, "
        f"{int(hist['Signals']):,} since {hist['Since']}): target reached first "
        f"{hist['Target First %']:.0f}%, stop first {hist['Stop First %']:.0f}%, "
        f"average {hist['Avg Result %']:+.2f}% after costs."
    )
    base = hist.get("Baseline Target First %")
    if base is not None and np.isfinite(base):
        diff = hist["Target First %"] - base
        if diff >= 3 and hist["Avg Result %"] > 0:
            text += f" Better than trading any {hist['Timeframe'].lower()} bar ({base:.0f}%)."
        else:
            text += f" No better than trading any bar ({base:.0f}%), so treat it as no edge."
    if cfg is not None and history_timeframe(cfg) == "Intraday" and str(cfg.get("interval")) not in ("60m", "1h"):
        text += " Intraday history is measured on 1-hour bars."
    return text


def classify_trade(symbol: str, df: pd.DataFrame, trend_slope: float, cfg: dict) -> dict:
    if df.empty:
        return {"State": "NO DATA", "Bias": "None", "Reason": "No bars returned."}

    close = coerce_series(df["Close"])
    last = safe_float(close.iloc[-1])
    ema20_last = safe_float(df["EMA20"].iloc[-1])
    ema30_last = safe_float(df["EMA30"].iloc[-1])
    hma_last = safe_float(df["HMA55"].iloc[-1])
    sr_last = safe_float(df["SR_REV"].iloc[-1])
    ntd_last = safe_float(df["NTD"].iloc[-1])
    adx_last = safe_float(df["ADX14"].iloc[-1])
    support, resistance = support_resistance(df, cfg["sr_lookback"])

    # Short recent deltas help detect whether S/R and NTD are improving or rolling over.
    sr_series = coerce_series(df["SR_REV"]).dropna()
    ntd_series = coerce_series(df["NTD"]).dropna()
    sr_delta = safe_float(sr_series.iloc[-1] - sr_series.iloc[-4]) if len(sr_series) >= 4 else np.nan
    ntd_delta = safe_float(ntd_series.iloc[-1] - ntd_series.iloc[-4]) if len(ntd_series) >= 4 else np.nan
    sr_rising = bool(np.isfinite(sr_delta) and sr_delta > 0)
    sr_falling = bool(np.isfinite(sr_delta) and sr_delta < 0)
    ntd_rising = bool(np.isfinite(ntd_delta) and ntd_delta > 0)
    ntd_falling = bool(np.isfinite(ntd_delta) and ntd_delta < 0)

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

    trend_up = bool(np.isfinite(trend_slope) and trend_slope >= 0)
    trend_down = bool(np.isfinite(trend_slope) and trend_slope < 0)

    price_above_ema20 = bool(np.isfinite(last) and np.isfinite(ema20_last) and last >= ema20_last)
    price_below_ema20 = bool(np.isfinite(last) and np.isfinite(ema20_last) and last < ema20_last)
    price_above_ema30 = bool(np.isfinite(last) and np.isfinite(ema30_last) and last >= ema30_last)
    price_below_ema30 = bool(np.isfinite(last) and np.isfinite(ema30_last) and last < ema30_last)
    price_above_hma = bool(np.isfinite(last) and np.isfinite(hma_last) and last >= hma_last)
    price_below_hma = bool(np.isfinite(last) and np.isfinite(hma_last) and last < hma_last)

    above_count = int(price_above_ema20) + int(price_above_ema30) + int(price_above_hma)
    below_count = int(price_below_ema20) + int(price_below_ema30) + int(price_below_hma)

    if below_count >= 3:
        current_structure = "Bearish — below 20/30 EMA and HMA"
    elif above_count >= 3:
        current_structure = "Bullish — above 20/30 EMA and HMA"
    elif below_count >= 2:
        current_structure = "Weak / Bearish pullback"
    elif above_count >= 2:
        current_structure = "Constructive / Bullish pullback"
    else:
        current_structure = "Mixed"

    bearish_structure = current_structure.startswith("Bearish") or current_structure.startswith("Weak")
    bullish_structure = current_structure.startswith("Bullish") or current_structure.startswith("Constructive")

    # Trade state is based on current structure first and regression slope second.
    # This prevents a slightly positive daily slope from overriding a clearly weak
    # price structure below EMA/HMA.
    if bullish_structure and trend_up and bull_bars is not None and cross_up and sr_last >= 0:
        state = "BUY CONFIRMED"
        bias = "Long"
        reason = "Bullish structure + upward trend + support reversal + recent 30 EMA cross upward."
    elif bullish_structure and (near_support or near_ema30 or near_hma55 or sr_rising or ntd_rising) and not ntd_falling:
        state = "BUY SETUP"
        bias = "Long"
        reason = "Constructive structure; wait for support/EMA/HMA hold or upward confirmation."
    elif bearish_structure and (bear_bars is not None or cross_down) and (sr_last <= 0 or sr_falling or ntd_falling):
        state = "SELL CONFIRMED" if trend_down else "SELL SETUP"
        bias = "Short"
        reason = "Bearish structure below EMA/HMA with rejection/weakening confirmation."
    elif bearish_structure and (near_resistance or near_ema30 or near_hma55 or sr_falling or ntd_falling):
        state = "SELL SETUP"
        bias = "Short"
        reason = "Weak structure below EMA/HMA; watch for resistance/EMA/HMA rejection."
    elif trend_up and not bearish_structure and (near_support or near_ema30 or near_hma55) and sr_last > -0.75:
        state = "BUY SETUP"
        bias = "Long"
        reason = "Uptrend pullback near support/EMA/HMA; wait for upward confirmation."
    elif trend_down and not bullish_structure and (near_resistance or near_ema30 or near_hma55) and sr_last < 0.75:
        state = "SELL SETUP"
        bias = "Short"
        reason = "Downtrend pullback near resistance/EMA/HMA; wait for downward confirmation."
    else:
        state = "WAIT"
        bias = "Neutral"
        if bearish_structure and trend_up:
            reason = "Long-term slope is upward, but current price structure is weak below EMA/HMA; avoid buying until reclaim."
        elif bullish_structure and trend_down:
            reason = "Long-term slope is downward, but current price structure is constructive above EMA/HMA; avoid shorting until breakdown."
        else:
            reason = "No clean trend-aligned pullback/reversal confirmation."

    atr_last = safe_float(df["ATR14"].iloc[-1])
    # Target and stop sit the same distance (PLAN_ATR x ATR) either side of price,
    # so the measured hit rate shows whether the signal has an edge.
    plan_atr = atr_last if np.isfinite(atr_last) and atr_last > 0 else max(abs(last) * 0.002, pip_size(symbol) * 10)
    if bias in ("Long", "Short"):
        plan_direction = bias
    else:
        # Neutral / WAIT charts still get an *indicative* planning scenario.
        # This is not an entry recommendation or a confirmed trade.
        plan_direction = (
            "Long" if above_count > below_count else
            "Short" if below_count > above_count else
            "Long" if trend_up else "Short"
        )
    # Entry is the current price: that is how the past record was measured.
    entry_zone = fmt_price(symbol, last)
    sign = 1 if plan_direction == "Long" else -1
    stop_zone = fmt_price(symbol, last - sign * PLAN_ATR * plan_atr)
    target1 = fmt_price(symbol, last + sign * PLAN_ATR * plan_atr)

    # Keep direction metadata separate from the original trade signal.
    # A WAIT chart stays WAIT; any levels shown are for planning only.
    plan_direction = bias if bias in ("Long", "Short") else plan_direction
    plan_indicative = (bias == "Neutral")

    probability_info = compute_trade_probabilities_and_instruction(
        symbol=symbol,
        state=state,
        bias=bias,
        last=last,
        ema20_last=ema20_last,
        ema30_last=ema30_last,
        hma_last=hma_last,
        sr_last=sr_last,
        ntd_last=ntd_last,
        adx_last=adx_last,
        trend_slope=trend_slope,
        support=support,
        resistance=resistance,
        near_support=near_support,
        near_resistance=near_resistance,
        near_ema30=near_ema30,
        near_hma55=near_hma55,
        cross_up=cross_up,
        cross_down=cross_down,
        entry_zone=entry_zone,
        stop_zone=stop_zone,
        target1=target1,
        current_structure=current_structure,
        sr_delta=sr_delta,
        ntd_delta=ntd_delta,
    )

    # An individual snapshot cannot establish the exact historical moment the
    # entire multi-condition classification first became confirmed. Preserve
    # the most recent underlying reversal/crossover trigger instead; the UI
    # labels this clearly as a trigger estimate, not a backtested confirmation.
    if state == "BUY CONFIRMED":
        event_times = [t for t in (bull_time, cross_up_time) if t is not None]
    elif state == "SELL CONFIRMED":
        event_times = [t for t in (bear_time, cross_down_time) if t is not None]
    else:
        event_times = []
    confirmation_trigger_time = max(event_times) if event_times else None

    return {
        **probability_info,
        "State": state,
        "Bias": bias,
        "Reason": reason,
        "Confirmation Trigger Time": confirmation_trigger_time,
        "Last Bar Timestamp": df.index[-1],
        "Last Close": last,
        "Current Structure": current_structure,
        "Trend Direction": "Upward" if trend_up else "Downward",
        "Trend Slope": trend_slope,
        "S/R Reversal": sr_last,
        "S/R Delta": sr_delta,
        "NTD": ntd_last,
        "NTD Delta": ntd_delta,
        "ADX": adx_last,
        "Support": support,
        "Resistance": resistance,
        "Near Support": bool(near_support),
        "Near Resistance": bool(near_resistance),
        "Near 30 EMA": bool(near_ema30),
        "Near HMA55": bool(near_hma55),
        "Price Above 20 EMA": bool(price_above_ema20),
        "Price Above 30 EMA": bool(price_above_ema30),
        "Price Above HMA55": bool(price_above_hma),
        "30 EMA Cross Up": bool(cross_up),
        "Bars Since 30 EMA Cross Up": cross_up_bars,
        "30 EMA Cross Down": bool(cross_down),
        "Bars Since 30 EMA Cross Down": cross_down_bars,
        "Bull S/R Bars": bull_bars,
        "Bear S/R Bars": bear_bars,
        "Entry Zone": entry_zone,
        "Stop / Invalidation": stop_zone,
        "Target 1": target1,
        "Plan Direction": plan_direction,
        "Plan Indicative": plan_indicative,
        "History": signal_history(symbol, cfg, state),
        "History Text": history_text(signal_history(symbol, cfg, state), cfg),
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


def _trade_level_numbers(value) -> list[float]:
    """Parse formatted prices, including FX entry ranges separated by an en dash."""
    import re
    matches = re.findall(r"\$?\d[\d,]*(?:\.\d+)?", str(value))
    result = []
    for match in matches:
        try:
            price = float(match.replace("$", "").replace(",", ""))
            if np.isfinite(price):
                result.append(price)
        except ValueError:
            pass
    return result


def trade_pip_distances(symbol: str, trade: dict) -> dict | None:
    """Estimate reward/risk from the entry midpoint; include entry-zone variability."""
    entries = _trade_level_numbers(trade.get("Entry Zone", ""))
    targets = _trade_level_numbers(trade.get("Target 1", ""))
    stops = _trade_level_numbers(trade.get("Stop / Invalidation", ""))
    if not entries:
        return None

    low, high = min(entries), max(entries)
    mid = (low + high) / 2.0
    action = str(trade.get("Trade Action", "")).upper()
    bias = str(trade.get("Plan Direction", trade.get("Bias", ""))).upper()
    if bias == "SHORT" or ("SELL" in action and "BUY" not in action):
        direction = -1
    elif bias == "LONG" or ("BUY" in action and "SELL" not in action):
        direction = 1
    else:
        return None

    pip = pip_size(symbol)
    target = targets[0] if targets else np.nan
    stop = stops[0] if stops else np.nan
    # Signed returns indicate whether the level is correctly placed for this trade.
    reward = direction * (target - mid) / pip if np.isfinite(target) else np.nan
    risk = -direction * (stop - mid) / pip if np.isfinite(stop) else np.nan
    reward_range = sorted([direction * (target - e) / pip for e in (low, high)]) if np.isfinite(target) else []
    risk_range = sorted([-direction * (stop - e) / pip for e in (low, high)]) if np.isfinite(stop) else []
    rr = reward / risk if np.isfinite(reward) and np.isfinite(risk) and reward > 0 and risk > 0 else np.nan
    return dict(low=low, high=high, midpoint=mid, target=target, stop=stop,
                reward=reward, risk=risk, reward_range=reward_range,
                risk_range=risk_range, rr=rr)


def draw_trade_levels(ax, symbol: str, trade: dict) -> dict | None:
    """Keep price data unobstructed: draw only levels in axes; text goes in sidebar."""
    levels = trade_pip_distances(symbol, trade)
    if levels is None:
        return None

    low, high = levels["low"], levels["high"]
    if high > low:
        ax.axhspan(low, high, color="royalblue", alpha=0.075, zorder=0)
    for level in sorted(set((low, high))):
        ax.axhline(level, color="royalblue", linestyle="--",
                   linewidth=1.4, alpha=0.85, zorder=2)
    if np.isfinite(levels["target"]):
        ax.axhline(levels["target"], color="forestgreen",
                   linestyle="-.", linewidth=1.8, alpha=0.9, zorder=2)
    if np.isfinite(levels["stop"]):
        ax.axhline(levels["stop"], color="firebrick",
                   linestyle=":", linewidth=1.8, alpha=0.9, zorder=2)
    return levels


def draw_trade_sidebar(fig, symbol: str, levels: dict | None) -> None:
    """Dedicated right rail uses fixed positions to prevent overlapping chart labels."""
    if levels is None:
        return
    def place(y, content, color):
        fig.text(0.795, y, content, ha="left", va="top", fontsize=9,
                 fontweight="semibold", color=color,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white",
                           ec=color, alpha=0.95))

    low, high = levels["low"], levels["high"]
    if levels.get("indicative"):
        fig.text(0.795, 0.805,
                 f"WATCH ONLY · {levels['direction'].upper()} scenario\\nNot a confirmed entry",
                 fontsize=8.5, ha="left", va="top", color="0.35",
                 fontweight="bold")
    entry = fmt_price(symbol, low)
    if high > low:
        entry += " – " + fmt_price(symbol, high)
    place(0.735, "ENTRY ZONE\n" + entry, "royalblue")

    def pip_label(value, values):
        if not np.isfinite(value):
            return "n/a"
        result = f"{value:+.1f} pips"
        if len(values) == 2 and abs(values[1] - values[0]) > 0.05:
            result += f"\nRange {values[0]:+.1f} to {values[1]:+.1f}"
        return result

    place(0.645, "TARGET  " + fmt_price(symbol, levels["target"]) +
          "\n" + pip_label(levels["reward"], levels["reward_range"]), "forestgreen")
    place(0.545, "STOP  " + fmt_price(symbol, levels["stop"]) +
          "\n" + pip_label(levels["risk"], levels["risk_range"]), "firebrick")
    rr = levels["rr"]
    fig.text(0.795, 0.442,
             f"Estimated R:R  1 : {rr:.2f}" if np.isfinite(rr) else
             "R:R  n/a (check level placement)",
             fontsize=9, ha="left", va="top", fontweight="bold", color="0.25")
    fig.text(0.795, 0.403, "Pips from the current price",
             fontsize=8, ha="left", va="top", color="0.4")

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
    # Reserve a right-side margin for the probability/instruction box so it never
    # covers candles, EMA/HMA lines, markers, or support/resistance labels.
    fig.subplots_adjust(left=0.075, right=0.765, top=0.91, bottom=0.09)

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

    # Trade-plan overlays (entry band, target and stop) on the price chart.
    trade_levels = draw_trade_levels(ax, symbol, trade)

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

    view_name = "Daily" if str(cfg.get("interval", "")).lower() in ("1d", "1wk", "1mo") else "Hourly"
    title = (
        f"{symbol} {view_name} {cfg.get('asset_class', 'Market')} Trading Chart ({cfg['period']}, {cfg['interval']})  |  "
        f"{state} / {trade.get('Trade Action', 'WAIT')}  |  "
        f"Score L{safe_float(trade.get('Long Score')):.0f}/S{safe_float(trade.get('Short Score')):.0f}  |  "
        f"Price {fmt_price(symbol, last)}  |  30EMA {fmt_price(symbol, ema30_last)}  |  "
        f"S/R {sr_last:+.2f}  NTD {ntd_last:+.2f}  ADX {adx_last:.1f}"
    )
    ax.set_title(title, fontsize=10, loc="left", pad=10)
    ax.grid(True)
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    long_prob, short_prob, prob_edge = probability_bar_values(trade)
    prob_read = probability_trade_read(trade)
    prob_color = "tab:green" if "BUY" in prob_read["Read"] else ("tab:red" if "SELL" in prob_read["Read"] else "0.35")
    # Put the trade instruction box outside the plotting area. This keeps the
    # chart actionable without hiding candles/lines near the right edge.
    fig.text(
        0.795,
        0.875,
        (
            f"{prob_read['Icon']} {prob_read['Read']}\n"
            f"Score L{long_prob:.0f} vs S{short_prob:.0f}\n"
            + (f"Past: target first {trade['History']['Target First %']:.0f}%\n"
               f"avg {trade['History']['Avg Result %']:+.2f}%"
               if trade.get("History") else "No past record")
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=prob_color,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=prob_color, alpha=0.88),
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

    draw_trade_sidebar(fig, symbol, trade_levels)

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
    ax2.set_title(f"{view_name} Indicator Panel — NTD + Smoothed NPX + S/R Reversal")
    ax2.grid(True)
    ax2.legend(loc="upper left", ncol=2)
    ax2.set_xlabel(f"Compressed real {view_name.lower()} trading bars — no weekend/closure gap")

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
        "Trade Action": trade.get("Trade Action"),
        "Confirmation Trigger Time": trade.get("Confirmation Trigger Time"),
        "Last Bar Timestamp": trade.get("Last Bar Timestamp"),
        "Long Score": f"{safe_float(trade.get('Long Score')):.0f}",
        "Short Score": f"{safe_float(trade.get('Short Score')):.0f}",
        "Score Edge": f"{safe_float(trade.get('Score Edge')):.0f}",
        "Past Target First": (f"{trade['History']['Target First %']:.0f}%" if trade.get("History") else "n/a"),
        "Past Avg Result": (f"{trade['History']['Avg Result %']:+.2f}%" if trade.get("History") else "n/a"),
        "Past Signals": (int(trade["History"]["Signals"]) if trade.get("History") else None),
        "Trend": trade_trend_alignment(trade).get("Trend Direction"),
        "Action / Trend Alignment": trade_trend_alignment(trade).get("Alignment Text"),
        "Current Structure": trade.get("Current Structure"),
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
        "Trade Instruction": trade.get("Trade Instruction"),
        "Reason": trade.get("Reason"),
    }


def scan_symbol(symbol: str, cfg: dict):
    df = fetch_market_ohlc(symbol, cfg["period"], cfg["interval"])
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

# --- Persistent sidebar selections ---
# Streamlit widget state usually survives normal reruns, but a browser refresh,
# app restart, or auto-refresh can otherwise fall back to defaults.
# We store the selected market, symbol, periods, and interval in both
# session_state and URL query parameters so choices persist across refreshes.
ASSET_OPTIONS = ["Forex", "Stocks"]
PERIOD_OPTIONS = ["1d", "2d", "5d", "7d", "10d", "30d"]
INTERVAL_OPTIONS = ["5m", "15m", "30m", "60m"]
DAILY_PERIOD_OPTIONS = ["3mo", "6mo", "1y", "2y", "5y"]
DEFAULT_ASSET_CLASS = "Forex"
DEFAULT_SYMBOL_BY_ASSET = {"Forex": "EURUSD=X", "Stocks": "SPY"}
DEFAULT_PERIOD = "5d"
DEFAULT_INTERVAL = "15m"
DEFAULT_DAILY_PERIOD = "1y"

def _query_param_first(name: str, default: str) -> str:
    try:
        val = st.query_params.get(name, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val if val is not None else default
    except Exception:
        return default

def _init_persistent_choice(key: str, query_name: str, default: str, valid_options: list[str]) -> None:
    if key not in st.session_state:
        candidate = _query_param_first(query_name, default)
        if candidate not in valid_options:
            candidate = default
        st.session_state[key] = candidate

def _persist_sidebar_choices() -> None:
    try:
        selected_asset = st.session_state.get("persist_asset_class", DEFAULT_ASSET_CLASS)
        default_symbol_for_asset = DEFAULT_SYMBOL_BY_ASSET.get(selected_asset, "EURUSD=X")
        st.query_params["asset_class"] = selected_asset
        st.query_params["symbol"] = st.session_state.get("persist_symbol", default_symbol_for_asset)
        st.query_params["period"] = st.session_state.get("persist_period", DEFAULT_PERIOD)
        st.query_params["interval"] = st.session_state.get("persist_interval", DEFAULT_INTERVAL)
        st.query_params["daily_period"] = st.session_state.get("persist_daily_period", DEFAULT_DAILY_PERIOD)
    except Exception:
        # Older/hosted Streamlit environments can restrict query param writes.
        # session_state persistence still protects ordinary reruns.
        pass

_init_persistent_choice("persist_asset_class", "asset_class", DEFAULT_ASSET_CLASS, ASSET_OPTIONS)

st.sidebar.title("Trading Indicator")
asset_class = st.sidebar.selectbox(
    "Market",
    ASSET_OPTIONS,
    index=ASSET_OPTIONS.index(st.session_state["persist_asset_class"]),
    key="persist_asset_class",
    on_change=_persist_sidebar_choices,
)
current_universe = _sort_symbols_alpha(ASSET_UNIVERSES[asset_class])
default_symbol = DEFAULT_SYMBOL_BY_ASSET[asset_class]

# If the user switches market type, keep persistence but reset the symbol only
# when the previously selected symbol does not belong to the selected universe.
if "persist_symbol" not in st.session_state:
    candidate_symbol = _query_param_first("symbol", default_symbol)
    st.session_state["persist_symbol"] = candidate_symbol if candidate_symbol in current_universe else default_symbol
elif st.session_state["persist_symbol"] not in current_universe:
    st.session_state["persist_symbol"] = default_symbol

_init_persistent_choice("persist_period", "period", DEFAULT_PERIOD, PERIOD_OPTIONS)
_init_persistent_choice("persist_interval", "interval", DEFAULT_INTERVAL, INTERVAL_OPTIONS)
_init_persistent_choice("persist_daily_period", "daily_period", DEFAULT_DAILY_PERIOD, DAILY_PERIOD_OPTIONS)

symbol_label = "Forex pair" if asset_class == "Forex" else "Stock / ETF"
symbol = st.sidebar.selectbox(
    symbol_label,
    current_universe,
    index=current_universe.index(st.session_state["persist_symbol"]),
    key="persist_symbol",
    on_change=_persist_sidebar_choices,
)
period = st.sidebar.selectbox(
    "Intraday chart period",
    PERIOD_OPTIONS,
    index=PERIOD_OPTIONS.index(st.session_state["persist_period"]),
    key="persist_period",
    on_change=_persist_sidebar_choices,
)
interval = st.sidebar.selectbox(
    "Intraday bar interval",
    INTERVAL_OPTIONS,
    index=INTERVAL_OPTIONS.index(st.session_state["persist_interval"]),
    key="persist_interval",
    on_change=_persist_sidebar_choices,
)
daily_period = st.sidebar.selectbox(
    "Daily chart period",
    DAILY_PERIOD_OPTIONS,
    index=DAILY_PERIOD_OPTIONS.index(st.session_state["persist_daily_period"]),
    key="persist_daily_period",
    on_change=_persist_sidebar_choices,
)

_persist_sidebar_choices()

# Engine settings kept at the values the signal history was measured with
# (bullbears_history_build.py). The tabs that used other values were removed.
trend_lookback, sr_lookback, ntd_window, touch_lookback, confirm_bars = 120, 120, 60, 8, 3
signal_lookback, cross_lookback, touch_tol_pips, pullback_tol_pips, cooldown = 40, 30, 3.0, 8.0, 10
npx_smooth, sr_smooth, right_padding = 9, 8, 12

cfg = {
    "period": period,
    "interval": interval,
    "daily_period": daily_period,
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
    "asset_class": asset_class,
}


# =========================
# Daily NTD Dip Scanner helpers
# =========================
def _trend_direction_from_slope(value) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "Unknown"
    return "Upward" if v >= 0 else "Downward"


def _recent_level_cross_down(series: pd.Series, level: float, lookback: int):
    """Most recent downward cross through a level within the tail lookback window."""
    s = coerce_series(series).dropna()
    if s.empty:
        return False, None, None
    cross = (s.shift(1) >= float(level)) & (s < float(level))
    recent = cross.tail(int(max(1, lookback)))
    if recent.any():
        pos = np.where(recent.to_numpy(dtype=bool))[0][-1]
        bars_since = len(recent) - 1 - pos
        ts = recent.index[pos]
        return True, int(bars_since), ts
    return False, None, None


def _last_level_cross_down(series: pd.Series, level: float):
    """Most recent downward cross through a level across the available history."""
    s = coerce_series(series).dropna()
    if s.empty:
        return None, None
    cross = (s.shift(1) >= float(level)) & (s < float(level))
    if cross.fillna(False).any():
        arr = cross.to_numpy(dtype=bool)
        pos = np.where(arr)[0][-1]
        return int(len(s) - 1 - pos), s.index[pos]
    return None, None


def _format_cross_time(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def daily_ntd_dip_scan_row(symbol: str, threshold: float, cfg: dict, lookback: int = 180):
    """Return a row when daily NTD is currently below the chosen dip threshold.

    The row includes the most recent downward drop through the threshold when
    available, plus price move since that dip in pips for FX or points for stocks.
    """
    try:
        scan_cfg = cfg.copy()
        scan_cfg["period"] = "5y"
        scan_cfg["interval"] = "1d"
        df = fetch_market_ohlc(symbol, "5y", "1d")
        if df.empty or len(df) < max(80, int(scan_cfg.get("ntd_window", 60)) + 10):
            return None
        df = prepare_indicators(df, scan_cfg, symbol)
        ntd = coerce_series(df["NTD"]).dropna()
        close = coerce_series(df["Close"]).dropna()
        if ntd.empty or close.empty:
            return None

        current_ntd = safe_float(ntd.iloc[-1])
        if not np.isfinite(current_ntd) or current_ntd > float(threshold):
            return None

        trend_line, trend_slope, trend_r2 = regression_line(df["Close"], int(cfg.get("trend_lookback", 120)))
        trend_direction = _trend_direction_from_slope(trend_slope)

        recent_cross, recent_bars, recent_ts = _recent_level_cross_down(ntd, threshold, lookback)
        bars_since, cross_ts = _last_level_cross_down(ntd, threshold)
        if recent_cross:
            bars_since, cross_ts = recent_bars, recent_ts

        current_price = safe_float(close.iloc[-1])
        cross_price = np.nan
        move_since_cross = np.nan
        if cross_ts is not None:
            try:
                cross_price = safe_float(close.loc[:cross_ts].iloc[-1])
            except Exception:
                try:
                    cross_price = safe_float(close.loc[cross_ts])
                except Exception:
                    cross_price = np.nan
            if np.isfinite(current_price) and np.isfinite(cross_price):
                move_since_cross = current_price - cross_price

        ntd_slope_line, ntd_slope, ntd_slope_r2 = regression_line(ntd, min(30, max(5, len(ntd))))
        ntd_direction = _trend_direction_from_slope(ntd_slope)

        support, resistance = support_resistance(df, int(cfg.get("sr_lookback", 120)))
        sr_current = safe_float(df["SR_REV"].iloc[-1]) if "SR_REV" in df.columns and len(df) else np.nan

        return {
            "Symbol": symbol,
            "Threshold": f"{threshold:+.2f}",
            "Price Trend Direction": trend_direction,
            "NTD Direction": ntd_direction,
            "Current NTD": current_ntd,
            "NTD Slope": safe_float(ntd_slope),
            "Current S/R Reversal": sr_current,
            "Current Price": current_price,
            "Cross Time": cross_ts,
            "Bars Since Dip": bars_since if bars_since is not None else np.nan,
            "Price at Dip": cross_price,
            "Move Since Dip": move_since_cross,
            "Move Since Dip (pips/pts)": (move_since_cross / pip_size(symbol)) if np.isfinite(move_since_cross) else np.nan,
            "Support": support,
            "Resistance": resistance,
            "Trend Slope": safe_float(trend_slope),
            "Trend R²": safe_float(trend_r2),
            "Recent Dip": "Yes" if recent_cross else "No",
            "Explanation": (
                f"Daily NTD is below {threshold:+.2f}. "
                f"Price trend is {trend_direction.lower()}; NTD direction is {ntd_direction.lower()}."
            ),
        }
    except Exception as exc:
        return {
            "Symbol": symbol,
            "Threshold": f"{threshold:+.2f}",
            "Price Trend Direction": "Error",
            "NTD Direction": "Error",
            "Explanation": f"Scan error: {exc}",
        }


def hourly_ntd_dip_scan_row(symbol: str, threshold: float, cfg: dict, lookback: int = 120):
    """Return a row when the hourly NTD is currently below the chosen dip threshold.

    This mirrors the daily NTD dip scanner, but uses the active hourly period/interval
    so Top 3 daily dip tables can be paired with short-term hourly symbols.
    """
    try:
        scan_cfg = cfg.copy()
        scan_cfg["period"] = cfg.get("period", "10d")
        scan_cfg["interval"] = cfg.get("interval", "1h")
        df = fetch_market_ohlc(symbol, scan_cfg["period"], scan_cfg["interval"])
        min_bars = max(40, min(120, int(scan_cfg.get("ntd_window", 60)) + 10))
        if df.empty or len(df) < min_bars:
            return None
        df = prepare_indicators(df, scan_cfg, symbol)
        ntd = coerce_series(df["NTD"]).dropna()
        close = coerce_series(df["Close"]).dropna()
        if ntd.empty or close.empty:
            return None

        current_ntd = safe_float(ntd.iloc[-1])
        if not np.isfinite(current_ntd) or current_ntd > float(threshold):
            return None

        trend_lb = min(int(cfg.get("trend_lookback", 120)), max(10, len(close)))
        _, trend_slope, trend_r2 = regression_line(close, trend_lb)
        trend_direction = _trend_direction_from_slope(trend_slope)

        recent_cross, recent_bars, recent_ts = _recent_level_cross_down(ntd, threshold, lookback)
        bars_since, cross_ts = _last_level_cross_down(ntd, threshold)
        if recent_cross:
            bars_since, cross_ts = recent_bars, recent_ts

        current_price = safe_float(close.iloc[-1])
        cross_price = np.nan
        move_since_cross = np.nan
        if cross_ts is not None:
            try:
                cross_price = safe_float(close.loc[:cross_ts].iloc[-1])
            except Exception:
                try:
                    cross_price = safe_float(close.loc[cross_ts])
                except Exception:
                    cross_price = np.nan
            if np.isfinite(current_price) and np.isfinite(cross_price):
                move_since_cross = current_price - cross_price

        ntd_lb = min(30, max(5, len(ntd)))
        _, ntd_slope, _ = regression_line(ntd, ntd_lb)
        ntd_direction = _trend_direction_from_slope(ntd_slope)

        support, resistance = support_resistance(df, int(cfg.get("sr_lookback", 120)))
        sr_current = safe_float(df["SR_REV"].iloc[-1]) if "SR_REV" in df.columns and len(df) else np.nan

        return {
            "Symbol": symbol,
            "Timeframe": f"Hourly ({scan_cfg.get('interval', '1h')})",
            "Threshold": f"{threshold:+.2f}",
            "Price Trend Direction": trend_direction,
            "NTD Direction": ntd_direction,
            "Current NTD": current_ntd,
            "NTD Slope": safe_float(ntd_slope),
            "Current S/R Reversal": sr_current,
            "Current Price": current_price,
            "Cross Time": cross_ts,
            "Bars Since Dip": bars_since if bars_since is not None else np.nan,
            "Price at Dip": cross_price,
            "Move Since Dip": move_since_cross,
            "Move Since Dip (pips/pts)": (move_since_cross / pip_size(symbol)) if np.isfinite(move_since_cross) else np.nan,
            "Support": support,
            "Resistance": resistance,
            "Trend Slope": safe_float(trend_slope),
            "Trend R²": safe_float(trend_r2),
            "Recent Dip": "Yes" if recent_cross else "No",
            "Explanation": (
                f"Hourly NTD is below {threshold:+.2f}. "
                f"Price trend is {trend_direction.lower()}; NTD direction is {ntd_direction.lower()}."
            ),
        }
    except Exception as exc:
        return {
            "Symbol": symbol,
            "Timeframe": "Hourly",
            "Threshold": f"{threshold:+.2f}",
            "Price Trend Direction": "Error",
            "NTD Direction": "Error",
            "Explanation": f"Hourly scan error: {exc}",
        }


def _format_daily_ntd_dip_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "Cross Time" in out.columns:
        out["Cross Time"] = out["Cross Time"].map(_format_cross_time)
    for col in ["Current NTD", "NTD Slope", "Current S/R Reversal", "Trend Slope", "Trend R²"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if not np.isfinite(safe_float(x)) else f"{safe_float(x):+.4f}")
    for col in ["Current Price", "Price at Dip", "Support", "Resistance"]:
        if col in out.columns:
            out[col] = [
                fmt_price(sym, val) for sym, val in zip(out.get("Symbol", pd.Series("", index=out.index)), out[col])
            ]
    if "Move Since Dip (pips/pts)" in out.columns:
        labels = []
        for sym, val in zip(out.get("Symbol", pd.Series("", index=out.index)), out["Move Since Dip (pips/pts)"]):
            v = safe_float(val)
            if not np.isfinite(v):
                labels.append("")
            elif "=X" in str(sym).upper():
                labels.append(f"{v:+.1f} pips")
            else:
                labels.append(f"{v:+.2f} pts")
        out["Move Since Dip"] = labels
        out = out.drop(columns=["Move Since Dip (pips/pts)"], errors="ignore")
    if "Bars Since Dip" in out.columns:
        out["Bars Since Dip"] = out["Bars Since Dip"].map(lambda x: "" if not np.isfinite(safe_float(x)) else int(safe_float(x)))
    preferred = [
        "Symbol", "Timeframe", "Threshold", "Price Trend Direction", "NTD Direction", "Current NTD", "Recent Dip",
        "Cross Time", "Bars Since Dip", "Current Price", "Price at Dip", "Move Since Dip",
        "Current S/R Reversal", "Support", "Resistance", "NTD Slope", "Trend Slope", "Trend R²", "Explanation"
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def _df_col_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Return a 1D Series for a DataFrame column, even when the column is missing or duplicated."""
    if df is None or df.empty:
        return pd.Series(dtype=object)
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.Series(obj, index=df.index)


def _sort_daily_ntd_dips(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Most actionable dip list: deeper NTD first, recent dips before stale dips,
    # then closest price move since dip.
    out["_RecentOrder"] = _df_col_series(out, "Recent Dip", "").astype(str).map({"Yes": 0, "No": 1}).fillna(2)
    out["_NTDOrder"] = pd.to_numeric(_df_col_series(out, "Current NTD", np.nan), errors="coerce")
    out["_BarsOrder"] = pd.to_numeric(_df_col_series(out, "Bars Since Dip", np.nan), errors="coerce").fillna(9999)
    out["_MoveAbsOrder"] = pd.to_numeric(_df_col_series(out, "Move Since Dip (pips/pts)", np.nan), errors="coerce").abs().fillna(999999)
    if "Symbol" not in out.columns:
        out["Symbol"] = ""
    out = out.sort_values(["_RecentOrder", "_NTDOrder", "_BarsOrder", "_MoveAbsOrder", "Symbol"], ascending=[True, True, True, True, True], kind="mergesort")
    return out.drop(columns=[c for c in out.columns if str(c).startswith("_")], errors="ignore")


def _render_dip_section(title: str, df: pd.DataFrame, top_n: int = 3):
    st.markdown(f"### {title}")
    if df is None or df.empty:
        st.info("No symbols found.")
        return
    sorted_df = _sort_daily_ntd_dips(df)
    st.markdown(f"**Top {top_n}**")
    st.dataframe(_format_daily_ntd_dip_table(sorted_df.head(top_n)), use_container_width=True, hide_index=True)
    with st.expander(f"Show all {title}", expanded=False):
        st.dataframe(_format_daily_ntd_dip_table(sorted_df), use_container_width=True, hide_index=True)

# =========================
# Daily S/R Cross Scanner Helpers
# =========================
def _direction_from_slope(value) -> str:
    v = safe_float(value)
    if not np.isfinite(v):
        return "Neutral"
    return "Upward" if v >= 0 else "Downward"


def _last_value(series_like):
    s = coerce_series(series_like).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _latest_recent_cross_event(a: pd.Series, b: pd.Series, lookback: int, upward_only: bool = False):
    """Return the most recent crossing event between a and b inside the lookback window.

    If upward_only=True, only returns events where a crossed upward through b.
    Otherwise returns the most recent upward or downward cross.
    """
    aa = coerce_series(a)
    bb = coerce_series(b)
    idx = aa.index.intersection(bb.index)
    aa = aa.reindex(idx)
    bb = bb.reindex(idx)
    if len(idx) < 3:
        return None

    cross_up = (aa.shift(1) <= bb.shift(1)) & (aa > bb)
    cross_down = (aa.shift(1) >= bb.shift(1)) & (aa < bb)
    events = cross_up if upward_only else (cross_up | cross_down)
    recent = events.tail(int(max(1, lookback))).fillna(False)

    if not bool(recent.any()):
        return None

    rel_pos = np.where(recent.to_numpy(dtype=bool))[0][-1]
    pos = len(events) - len(recent) + int(rel_pos)
    ts = events.index[pos]
    direction = "Upward" if bool(cross_up.iloc[pos]) else "Downward"
    bars_since = len(events) - 1 - pos
    return {
        "pos": int(pos),
        "timestamp": ts,
        "direction": direction,
        "bars_since": int(bars_since),
    }


def _move_since_cross(symbol: str, current_price: float, cross_price: float) -> tuple[float, float]:
    cur = safe_float(current_price)
    cp = safe_float(cross_price)
    unit = pip_size(symbol)
    if not np.isfinite(cur) or not np.isfinite(cp) or unit <= 0:
        return float("nan"), float("nan")
    signed = (cur - cp) / unit
    return float(signed), float(abs(signed))


def _daily_sr_cross_row(symbol: str, cfg: dict, lookback: int):
    daily_cfg = cfg.copy()
    daily_cfg["period"] = cfg.get("daily_period", "1y")
    daily_cfg["interval"] = "1d"

    df = fetch_market_ohlc(symbol, daily_cfg["period"], "1d")
    if df.empty or len(df) < max(80, int(cfg.get("trend_lookback", 120)) // 2):
        return None

    df = prepare_indicators(df, daily_cfg, symbol)
    if df.empty or not {"Close", "SR_REV", "NTD"}.issubset(df.columns):
        return None

    close = coerce_series(df["Close"]).dropna()
    sr = coerce_series(df["SR_REV"]).reindex(df.index)
    ntd = coerce_series(df["NTD"]).reindex(df.index)
    if close.empty or sr.dropna().empty or ntd.dropna().empty:
        return None

    _, price_trend_slope, _ = regression_line(close, int(cfg.get("trend_lookback", 120)))
    ntd_lb = min(int(cfg.get("trend_lookback", 120)), max(10, len(ntd.dropna())))
    _, ntd_trend_slope, _ = regression_line(ntd.dropna(), ntd_lb)

    current_price = float(close.iloc[-1])
    sr_last = _last_value(sr)
    ntd_last = _last_value(ntd)

    base = {
        "Symbol": symbol,
        "Last Close": fmt_price(symbol, current_price),
        "Last Close Raw": current_price,
        "Current S/R Reversal": round(sr_last, 4) if np.isfinite(sr_last) else np.nan,
        "Current NTD": round(ntd_last, 4) if np.isfinite(ntd_last) else np.nan,
        "S/R Direction": "Upward" if len(sr.dropna()) >= 2 and sr.dropna().iloc[-1] >= sr.dropna().iloc[-2] else "Downward",
        "Price Trend Direction": _direction_from_slope(price_trend_slope),
        "NTD Trend Direction": _direction_from_slope(ntd_trend_slope),
        "Price Trend Slope": round(safe_float(price_trend_slope), 7),
        "NTD Trend Slope": round(safe_float(ntd_trend_slope), 7),
        "Last Bar Time": _format_cross_time(df.index[-1]),
    }

    out = {
        "sr_ntd_cross": None,
        "minus05_up": None,
        "zero_up": None,
    }

    # (a) S/R Reversal line crossing the NTD line, either direction.
    ev = _latest_recent_cross_event(sr, ntd, lookback, upward_only=False)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["sr_ntd_cross"] = {
            **base,
            "Cross Type": "S/R crossed NTD",
            "Cross Direction": ev["direction"],
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"S/R Reversal line crossed the NTD line {ev['direction'].lower()} {ev['bars_since']} daily bars ago.",
        }

    # (b1) S/R Reversal line crossing -0.5 upward.
    minus05_line = pd.Series(-0.5, index=df.index)
    ev = _latest_recent_cross_event(sr, minus05_line, lookback, upward_only=True)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["minus05_up"] = {
            **base,
            "Cross Type": "S/R crossed -0.5 upward",
            "Cross Direction": "Upward",
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"S/R Reversal line recently crossed upward through -0.5 {ev['bars_since']} daily bars ago.",
        }

    # (b2) S/R Reversal line crossing 0.0 upward.
    zero_line = pd.Series(0.0, index=df.index)
    ev = _latest_recent_cross_event(sr, zero_line, lookback, upward_only=True)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["zero_up"] = {
            **base,
            "Cross Type": "S/R crossed 0.0 upward",
            "Cross Direction": "Upward",
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"S/R Reversal line recently crossed upward through 0.0 {ev['bars_since']} daily bars ago.",
        }

    return out


def _hourly_sr_cross_row(symbol: str, cfg: dict, lookback: int):
    """Hourly version of the S/R cross scanner used for Top 3 quick-pick tables."""
    hourly_cfg = cfg.copy()
    hourly_cfg["period"] = cfg.get("period", "10d")
    hourly_cfg["interval"] = cfg.get("interval", "1h")

    df = fetch_market_ohlc(symbol, hourly_cfg["period"], hourly_cfg["interval"])
    min_bars = max(40, min(120, int(cfg.get("trend_lookback", 120)) // 2))
    if df.empty or len(df) < min_bars:
        return None

    df = prepare_indicators(df, hourly_cfg, symbol)
    if df.empty or not {"Close", "SR_REV", "NTD"}.issubset(df.columns):
        return None

    close = coerce_series(df["Close"]).dropna()
    sr = coerce_series(df["SR_REV"]).reindex(df.index)
    ntd = coerce_series(df["NTD"]).reindex(df.index)
    if close.empty or sr.dropna().empty or ntd.dropna().empty:
        return None

    price_lb = min(int(cfg.get("trend_lookback", 120)), max(10, len(close)))
    _, price_trend_slope, _ = regression_line(close, price_lb)
    ntd_lb = min(int(cfg.get("trend_lookback", 120)), max(10, len(ntd.dropna())))
    _, ntd_trend_slope, _ = regression_line(ntd.dropna(), ntd_lb)

    current_price = float(close.iloc[-1])
    sr_last = _last_value(sr)
    ntd_last = _last_value(ntd)
    interval_label = str(hourly_cfg.get("interval", "1h"))

    base = {
        "Symbol": symbol,
        "Timeframe": f"Hourly ({interval_label})",
        "Last Close": fmt_price(symbol, current_price),
        "Last Close Raw": current_price,
        "Current S/R Reversal": round(sr_last, 4) if np.isfinite(sr_last) else np.nan,
        "Current NTD": round(ntd_last, 4) if np.isfinite(ntd_last) else np.nan,
        "S/R Direction": "Upward" if len(sr.dropna()) >= 2 and sr.dropna().iloc[-1] >= sr.dropna().iloc[-2] else "Downward",
        "Price Trend Direction": _direction_from_slope(price_trend_slope),
        "NTD Trend Direction": _direction_from_slope(ntd_trend_slope),
        "Price Trend Slope": round(safe_float(price_trend_slope), 7),
        "NTD Trend Slope": round(safe_float(ntd_trend_slope), 7),
        "Last Bar Time": _format_cross_time(df.index[-1]),
    }

    out = {"sr_ntd_cross": None, "minus05_up": None, "zero_up": None}

    ev = _latest_recent_cross_event(sr, ntd, lookback, upward_only=False)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["sr_ntd_cross"] = {
            **base,
            "Cross Type": "S/R crossed NTD",
            "Cross Direction": ev["direction"],
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"Hourly S/R Reversal line crossed the NTD line {ev['direction'].lower()} {ev['bars_since']} bars ago.",
        }

    minus05_line = pd.Series(-0.5, index=df.index)
    ev = _latest_recent_cross_event(sr, minus05_line, lookback, upward_only=True)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["minus05_up"] = {
            **base,
            "Cross Type": "S/R crossed -0.5 upward",
            "Cross Direction": "Upward",
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"Hourly S/R Reversal line recently crossed upward through -0.5 {ev['bars_since']} bars ago.",
        }

    zero_line = pd.Series(0.0, index=df.index)
    ev = _latest_recent_cross_event(sr, zero_line, lookback, upward_only=True)
    if ev is not None:
        pos = ev["pos"]
        cross_price = safe_float(close.iloc[pos])
        signed_units, abs_units = _move_since_cross(symbol, current_price, cross_price)
        out["zero_up"] = {
            **base,
            "Cross Type": "S/R crossed 0.0 upward",
            "Cross Direction": "Upward",
            "Cross Time": _format_cross_time(ev["timestamp"]),
            "Bars Since Cross": ev["bars_since"],
            "Price at Cross": fmt_price(symbol, cross_price),
            "Price at Cross Raw": cross_price,
            "Pips/Points Since Cross": round(signed_units, 1) if np.isfinite(signed_units) else np.nan,
            "Abs Pips/Points Since Cross": round(abs_units, 1) if np.isfinite(abs_units) else np.nan,
            "Explanation": f"Hourly S/R Reversal line recently crossed upward through 0.0 {ev['bars_since']} bars ago.",
        }

    return out


def _sort_daily_sr_cross_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["Abs Pips/Points Since Cross", "Bars Since Cross"]:
        if col not in out.columns:
            out[col] = np.nan
    out["_TrendOrder"] = _df_col_series(out, "Price Trend Direction", "").astype(str).map({"Upward": 0, "Neutral": 1, "Downward": 2}).fillna(3)
    out["_SRDirOrder"] = _df_col_series(out, "S/R Direction", "").astype(str).map({"Upward": 0, "Downward": 1}).fillna(2)
    out = out.sort_values(
        by=["_TrendOrder", "_SRDirOrder", "Abs Pips/Points Since Cross", "Bars Since Cross", "Symbol"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    return out.drop(columns=["_TrendOrder", "_SRDirOrder"], errors="ignore")


def _display_daily_sr_cross_table(title: str, df: pd.DataFrame, empty_text: str) -> None:
    st.markdown(f"#### {title}")
    if df is None or df.empty:
        st.info(empty_text)
        return
    preferred = [
        "Symbol",
        "S/R Direction",
        "Price Trend Direction",
        "NTD Trend Direction",
        "Cross Direction",
        "Bars Since Cross",
        "Pips/Points Since Cross",
        "Abs Pips/Points Since Cross",
        "Cross Time",
        "Price at Cross",
        "Last Close",
        "Current S/R Reversal",
        "Current NTD",
        "Price Trend Slope",
        "NTD Trend Slope",
        "Last Bar Time",
        "Explanation",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred and not c.endswith(" Raw")]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def _top_daily_sr_buy_opportunities(*dfs: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Rank the strongest Daily S/R Cross rows as quick buy opportunities.

    This is intentionally conservative: it prefers upward S/R crosses, upward current
    S/R direction, upward price trend, and upward/improving NTD trend.  Rows are then
    ordered by higher setup score and fewer pips/points since the cross so users see
    the freshest actionable opportunities first.
    """
    frames = [df.copy() for df in dfs if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    for col in [
        "Cross Direction",
        "S/R Direction",
        "Price Trend Direction",
        "NTD Trend Direction",
        "Current S/R Reversal",
        "Current NTD",
        "Abs Pips/Points Since Cross",
        "Bars Since Cross",
        "Cross Type",
    ]:
        if col not in combined.columns:
            combined[col] = np.nan

    cross_type = combined["Cross Type"].astype(str)
    cross_dir = combined["Cross Direction"].astype(str)
    sr_dir = combined["S/R Direction"].astype(str)
    trend_dir = combined["Price Trend Direction"].astype(str)
    ntd_trend_dir = combined["NTD Trend Direction"].astype(str)

    upward_cross = cross_dir.eq("Upward") | cross_type.str.contains("upward", case=False, na=False)
    bullish_structure = trend_dir.eq("Upward")
    improving_sr = sr_dir.eq("Upward")

    candidates = combined[upward_cross & bullish_structure & improving_sr].copy()
    if candidates.empty:
        # If the strict filter is empty, still show useful early buy candidates:
        # upward crosses with an upward price trend.
        candidates = combined[upward_cross & bullish_structure].copy()
    if candidates.empty:
        return pd.DataFrame()

    sr_val = pd.to_numeric(candidates["Current S/R Reversal"], errors="coerce")
    ntd_val = pd.to_numeric(candidates["Current NTD"], errors="coerce")
    abs_units = pd.to_numeric(candidates["Abs Pips/Points Since Cross"], errors="coerce")
    bars = pd.to_numeric(candidates["Bars Since Cross"], errors="coerce")

    type_score = pd.Series(0, index=candidates.index, dtype=float)
    type_text = candidates["Cross Type"].astype(str)
    type_score += np.where(type_text.str.contains("0.0", case=False, na=False), 3.0, 0.0)
    type_score += np.where(type_text.str.contains("-0.5", case=False, na=False), 2.0, 0.0)
    type_score += np.where(type_text.str.contains("NTD", case=False, na=False), 1.0, 0.0)

    setup_score = (
        3.0 * candidates["Price Trend Direction"].astype(str).eq("Upward").astype(float)
        + 2.0 * candidates["NTD Trend Direction"].astype(str).eq("Upward").astype(float)
        + 2.0 * candidates["S/R Direction"].astype(str).eq("Upward").astype(float)
        + 1.5 * (sr_val > 0.0).astype(float)
        + 1.0 * (ntd_val > 0.0).astype(float)
        + type_score
    )

    candidates["Buy Opportunity Score"] = setup_score.round(1)
    candidates["Buy Opportunity Explanation"] = candidates.apply(
        lambda r: (
            f"{r.get('Cross Type', 'S/R cross')} with {r.get('S/R Direction', 'n/a')} S/R direction, "
            f"{r.get('Price Trend Direction', 'n/a')} price trend, and "
            f"{r.get('NTD Trend Direction', 'n/a')} NTD trend. "
            f"Cross occurred {r.get('Bars Since Cross', 'n/a')} daily bars ago."
        ),
        axis=1,
    )

    # Keep the best row per symbol so the top-three table is not dominated by one symbol.
    candidates["_AbsSort"] = abs_units.fillna(np.inf)
    candidates["_BarsSort"] = bars.fillna(np.inf)
    candidates = candidates.sort_values(
        by=["Buy Opportunity Score", "_AbsSort", "_BarsSort", "Symbol"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    candidates = candidates.drop_duplicates(subset=["Symbol"], keep="first").head(int(max(1, limit)))

    preferred = [
        "Symbol",
        "Buy Opportunity Score",
        "Cross Type",
        "S/R Direction",
        "Price Trend Direction",
        "NTD Trend Direction",
        "Bars Since Cross",
        "Pips/Points Since Cross",
        "Abs Pips/Points Since Cross",
        "Cross Time",
        "Price at Cross",
        "Last Close",
        "Current S/R Reversal",
        "Current NTD",
        "Buy Opportunity Explanation",
    ]
    cols = [c for c in preferred if c in candidates.columns] + [
        c for c in candidates.columns
        if c not in preferred and not c.startswith("_") and not c.endswith(" Raw")
    ]
    return candidates[cols].reset_index(drop=True)


def _top_downtrend_sr_ntd_upward_reversal_candidates(sr_ntd_df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Top rows from the Price Trend Downward S/R-vs-NTD table where both
    the current S/R direction and the NTD trend direction are upward.

    These are possible early reversal / bounce candidates inside a still-downward
    price trend, so they are shown separately from the main top-buy table.
    """
    if sr_ntd_df is None or sr_ntd_df.empty:
        return pd.DataFrame()

    df = sr_ntd_df.copy()
    for col in [
        "Price Trend Direction",
        "S/R Direction",
        "NTD Trend Direction",
        "Current S/R Reversal",
        "Current NTD",
        "Abs Pips/Points Since Cross",
        "Bars Since Cross",
        "Symbol",
        "Cross Type",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    candidates = df[
        df["Price Trend Direction"].astype(str).eq("Downward")
        & df["S/R Direction"].astype(str).eq("Upward")
        & df["NTD Trend Direction"].astype(str).eq("Upward")
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    sr_val = pd.to_numeric(candidates["Current S/R Reversal"], errors="coerce")
    ntd_val = pd.to_numeric(candidates["Current NTD"], errors="coerce")
    abs_units = pd.to_numeric(candidates["Abs Pips/Points Since Cross"], errors="coerce")
    bars = pd.to_numeric(candidates["Bars Since Cross"], errors="coerce")

    # This score rewards improving S/R and NTD conditions while still treating
    # these as countertrend/early reversal candidates because the price trend is downward.
    candidates["Reversal Watch Score"] = (
        3.0
        + 1.5 * candidates["S/R Direction"].astype(str).eq("Upward").astype(float)
        + 1.5 * candidates["NTD Trend Direction"].astype(str).eq("Upward").astype(float)
        + 1.0 * (sr_val > 0.0).astype(float)
        + 0.75 * (ntd_val > -0.25).astype(float)
    ).round(1)

    candidates["Reversal Watch Explanation"] = candidates.apply(
        lambda r: (
            "Price trend is still downward, but S/R Direction and NTD Trend Direction are both upward. "
            f"{r.get('Cross Type', 'S/R crossed NTD')} occurred {r.get('Bars Since Cross', 'n/a')} daily bars ago. "
            "Treat as an early reversal/bounce watch candidate, not a confirmed trend-following buy."
        ),
        axis=1,
    )

    candidates["_AbsSort"] = abs_units.fillna(np.inf)
    candidates["_BarsSort"] = bars.fillna(np.inf)
    candidates = candidates.sort_values(
        by=["Reversal Watch Score", "_AbsSort", "_BarsSort", "Symbol"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    candidates = candidates.drop_duplicates(subset=["Symbol"], keep="first").head(int(max(1, limit)))

    preferred = [
        "Symbol",
        "Reversal Watch Score",
        "Cross Type",
        "S/R Direction",
        "Price Trend Direction",
        "NTD Trend Direction",
        "Bars Since Cross",
        "Pips/Points Since Cross",
        "Abs Pips/Points Since Cross",
        "Cross Time",
        "Price at Cross",
        "Last Close",
        "Current S/R Reversal",
        "Current NTD",
        "Reversal Watch Explanation",
    ]
    cols = [c for c in preferred if c in candidates.columns] + [
        c for c in candidates.columns
        if c not in preferred and not c.startswith("_") and not c.endswith(" Raw")
    ]
    return candidates[cols].reset_index(drop=True)


# =========================
# Main app
# =========================
st.title("📈 Stocks / FX: tested signals only")
st.caption(
     "Only what held up in testing: a likely price range (not a direction), the tested dip buy signals with each "
     "symbol's own past record, and an intraday view for timing. Tabs whose signals did no better than chance were removed."
)


# =========================
# NTD zero-line upward cross scanner
# =========================
def _scan_ntd_zero_up(symbol: str, timeframe: str, settings: dict, recent_bars: int):
    """Return the most recent upward NTD zero cross within the requested real bars."""
    is_daily = timeframe == "Daily"
    df = fetch_market_ohlc(
        symbol,
        period="1y" if is_daily else "1mo",
        interval="1d" if is_daily else "1h",
    )
    if df.empty:
        return None
    close = coerce_series(df["Close"]).dropna()
    if len(close) < max(12, int(settings["ntd_window"]) // 3):
        return None
    ntd = normalized_trend(close, int(settings["ntd_window"]))
    # A zero cross requires a known, nonpositive previous observation.
    crossed = (ntd.shift(1) <= 0.0) & (ntd > 0.0)
    crossed = crossed.fillna(False)
    event_positions = np.flatnonzero(crossed.to_numpy(dtype=bool))
    if len(event_positions) == 0:
        return None
    pos = int(event_positions[-1])
    bars_since = len(ntd) - 1 - pos
    if bars_since > int(recent_bars):
        return None

    _, trend_slope, _ = regression_line(close, int(settings["trend_lookback"]))
    ntd_recent = ntd.dropna().iloc[-min(12, len(ntd.dropna())):]
    _, ntd_slope, _ = regression_line(ntd_recent, len(ntd_recent))
    cross_price = float(close.iloc[pos])
    last_price = float(close.iloc[-1])
    unit = "pips" if "=X" in symbol else "points"
    pip_size = (0.01 if "JPY" in symbol.upper() else 0.0001) if unit == "pips" else 1.0
    crossed_at = ntd.index[pos]
    return {
        "Symbol": symbol,
        "Bars Since Cross": bars_since,
        "Cross Time (Pacific)": pd.Timestamp(crossed_at).strftime("%Y-%m-%d %H:%M"),
        "NTD at Cross": round(float(ntd.iloc[pos]), 4),
        "Current NTD": round(float(ntd.iloc[-1]), 4),
        "NTD Direction": "Upward" if np.isfinite(ntd_slope) and ntd_slope >= 0 else "Downward",
        "Price Trend Direction": "Upward" if np.isfinite(trend_slope) and trend_slope >= 0 else "Downward",
        "Price at Cross": round(cross_price, 5),
        "Last Price": round(last_price, 5),
        "Move Since Cross": round((last_price - cross_price) / pip_size, 1),
        "Move Unit": unit,
    }


# =========================
# HMA(55) upward price-cross scanner
# =========================
def _scan_hma_price_cross_up(symbol: str, timeframe: str, settings: dict, recent_bars: int, hma_length: int = 55):
    """Find the most recent close-price upward HMA cross, with price trend at latest bar."""
    is_daily = timeframe == "Daily"
    df = fetch_market_ohlc(symbol, period="1y" if is_daily else "1mo",
                           interval="1d" if is_daily else "1h")
    if df.empty:
        return None
    close = coerce_series(df["Close"]).dropna()
    if len(close) < max(int(hma_length) + int(np.sqrt(hma_length)) + 3, 70):
        return None
    hma_line = hma(close, int(hma_length)).reindex(close.index)
    prev_price = close.shift(1)
    prev_hma = hma_line.shift(1)
    cross_mask = (prev_price <= prev_hma) & (close > hma_line)
    cross_mask = cross_mask.fillna(False)
    positions = np.flatnonzero(cross_mask.to_numpy(dtype=bool))
    if len(positions) == 0:
        return None
    pos = int(positions[-1])
    bars_since = len(close) - pos - 1
    if bars_since > int(recent_bars):
        return None
    _, trend_slope, _ = regression_line(close, int(settings["trend_lookback"]))
    if not np.isfinite(trend_slope):
        return None
    current_price = float(close.iloc[-1])
    price_at_cross = float(close.iloc[pos])
    current_hma = float(hma_line.iloc[-1])
    if not np.isfinite(current_hma):
        return None
    unit = "pips" if "=X" in symbol.upper() else "points"
    unit_size = (0.01 if "JPY" in symbol.upper() else 0.0001) if unit == "pips" else 1.0
    crossed_at = pd.Timestamp(close.index[pos])
    return {
        "Symbol": symbol,
        "Price Trend Direction": "Upward" if trend_slope >= 0 else "Downward",
        "Bars Since Cross": int(bars_since),
        "Cross Time (Pacific)": crossed_at.strftime("%Y-%m-%d %H:%M"),
        "Price at Cross": round(price_at_cross, 5),
        "Last Price": round(current_price, 5),
        "Current HMA": round(current_hma, 5),
        "Price Above HMA": bool(current_price > current_hma),
        "Move Since Cross": round((current_price - price_at_cross) / unit_size, 1),
        "Move Unit": unit,
        "Trend Slope": round(float(trend_slope), 7),
    }


# =========================
# Green / Red Zone onset scanner
# =========================
def scan_fresh_color_zone(symbol: str, timeframe: str, settings: dict, recent_bars: int):
    """Scan transitions of the same NTD sign that colors the chart's lower panel.

    A green zone starts when NTD crosses from <= 0 to > 0; a red zone starts
    when it crosses from >= 0 to < 0. Only report the current, fresh zone if
    NTD is still moving in that direction on the last bar.
    """
    is_daily = timeframe == "Daily"
    period = settings.get("daily_period", "1y") if is_daily else "1mo"
    interval = "1d" if is_daily else "1h"
    raw = fetch_market_ohlc(symbol, period=period, interval=interval)
    if raw.empty or len(raw) < max(20, int(settings["ntd_window"]) // 3):
        return None

    close = coerce_series(raw["Close"])
    ntd = normalized_trend(close, int(settings["ntd_window"]))
    npx = normalized_price(close, int(settings["ntd_window"]), int(settings["npx_smooth"]))

    # Match the regression trendline shown on the selected price chart.
    _, price_slope, price_r2 = regression_line(close, int(settings["trend_lookback"]))
    if np.isfinite(price_slope):
        price_trend = "Upward ↑" if price_slope > 0 else ("Downward ↓" if price_slope < 0 else "Flat →")
    else:
        price_trend = "Unknown"

    valid = ntd.dropna()
    if len(valid) < 3:
        return None

    last = float(ntd.iloc[-1])
    previous = float(ntd.iloc[-2])
    if not (np.isfinite(last) and np.isfinite(previous)):
        return None
    if last > 0 and last > previous:
        zone, event = "🟢 Green Zone — Going Up", (ntd.shift(1) <= 0) & (ntd > 0)
    elif last < 0 and last < previous:
        zone, event = "🔴 Red Zone — Going Down", (ntd.shift(1) >= 0) & (ntd < 0)
    else:
        return None

    positions = np.flatnonzero(event.fillna(False).to_numpy(dtype=bool))
    if len(positions) == 0:
        return None
    pos = int(positions[-1])
    bars_since = len(ntd) - 1 - pos
    if bars_since > int(recent_bars):
        return None

    return {
        "Symbol": symbol,
        "Zone": zone,
        "Price Trend": price_trend,
        "Trend Alignment": (
            "Aligned ✓"
            if (("Green Zone" in zone and price_trend == "Upward ↑")
                or ("Red Zone" in zone and price_trend == "Downward ↓"))
            else "Opposing ⚠"
            if price_trend in ("Upward ↑", "Downward ↓")
            else "Neutral / Unknown"
        ),
        "Price Trend Slope (pips/bar)" if "=X" in str(symbol).upper() else "Price Trend Slope (points/bar)": (
            round(price_slope / pip_size(symbol), 3) if np.isfinite(price_slope) else np.nan
        ),
        "Price Trend R²": round(price_r2, 3) if np.isfinite(price_r2) else np.nan,
        "Bars Since Start": bars_since,
        "Zone Started (Pacific)": pd.Timestamp(ntd.index[pos]).strftime("%Y-%m-%d %H:%M"),
        "Current NTD": round(last, 3),
        "NTD Change (1 bar)": round(last - previous, 3),
        "Smoothed NPX": round(safe_float(npx.iloc[-1]), 3),
        "Last Price": fmt_price(symbol, close.iloc[-1]),
    }


# =========================
# Fresh upward Smoothed NPX zero-line crossings
# =========================
def scan_smoothed_npx_up_cross(symbol: str, timeframe: str, settings: dict, recent_bars: int):
    """Find an upward NPX zero cross that remains positive on the latest bar.

    The price-trend category is based on the same closing-price linear
    regression used by the chart, not on the NPX slope.
    """
    is_daily = timeframe == "Daily"
    period = settings.get("daily_period", "1y") if is_daily else "1mo"
    interval = "1d" if is_daily else "1h"
    raw = fetch_market_ohlc(symbol, period=period, interval=interval)
    if raw.empty or len(raw) < max(20, int(settings["ntd_window"]) // 3):
        return None

    close = coerce_series(raw["Close"])
    npx = normalized_price(close, int(settings["ntd_window"]), int(settings["npx_smooth"]))
    if len(npx.dropna()) < 3:
        return None

    # A cross occurs on the first bar strictly above zero, after a bar at/below it.
    crossed = (npx.shift(1) <= 0.0) & (npx > 0.0)
    indices = np.flatnonzero(crossed.fillna(False).to_numpy(dtype=bool))
    if len(indices) == 0:
        return None

    cross_pos = int(indices[-1])
    bars_since = len(npx) - 1 - cross_pos
    now = safe_float(npx.iloc[-1])
    prev = safe_float(npx.iloc[-2])
    if (bars_since > int(recent_bars)
            or not (np.isfinite(now) and np.isfinite(prev))
            or now <= 0.0 or now <= prev):
        return None

    _, slope, r2 = regression_line(close, int(settings["trend_lookback"]))
    if not np.isfinite(slope):
        return None
    if slope > 0:
        price_trend = "Upward ↑"
        alignment = "Aligned ✓"
    elif slope < 0:
        price_trend = "Downward ↓"
        alignment = "Opposing ⚠"
    else:
        price_trend = "Flat →"
        alignment = "Neutral"

    unit = "pips/bar" if "=X" in str(symbol).upper() else "points/bar"
    slope_in_units = slope / pip_size(symbol) if "=X" in str(symbol).upper() else slope
    return {
        "Symbol": symbol,
        "Price Trend": price_trend,
        "Trend Alignment": alignment,
        "Bars Since Cross": int(bars_since),
        "Cross Time (Pacific)": pd.Timestamp(npx.index[cross_pos]).strftime("%Y-%m-%d %H:%M"),
        "NPX at Cross": round(safe_float(npx.iloc[cross_pos]), 3),
        "Current Smoothed NPX": round(now, 3),
        "NPX Change (1 bar)": round(now - prev, 3),
        f"Price Trend Slope ({unit})": round(slope_in_units, 3),
        "Price Trend R²": round(r2, 3) if np.isfinite(r2) else np.nan,
        "Last Price": fmt_price(symbol, close.iloc[-1]),
    }


# =========================
# 30-period EMA upward price-cross scanner
# =========================
def scan_ema30_price_cross_up(symbol: str, timeframe: str, settings: dict, recent_bars: int):
    """Latest close-price upward crossing of 30 EMA, grouped by chart trend slope."""
    is_daily = timeframe == "Daily"
    period = settings.get("daily_period", "1y") if is_daily else "1mo"
    raw = fetch_market_ohlc(symbol, period=period, interval="1d" if is_daily else "1h")
    if raw.empty:
        return None
    close = coerce_series(raw["Close"]).dropna()
    if len(close) < max(12, int(settings["trend_lookback"]) // 3):
        return None
    ema30_cross = ema(close, 30)
    events = ((close.shift(1) <= ema30_cross.shift(1)) & (close > ema30_cross)).fillna(False)
    positions = np.flatnonzero(events.to_numpy(dtype=bool))
    if len(positions) == 0:
        return None
    pos = int(positions[-1])
    bars_since = len(close) - 1 - pos
    if bars_since > int(recent_bars):
        return None
    # Keep active upward crosses only, rather than showing stale crosses
    # where the price has subsequently moved back below the 30 EMA.
    if not (np.isfinite(close.iloc[-1]) and np.isfinite(ema30_cross.iloc[-1])
            and close.iloc[-1] > ema30_cross.iloc[-1]):
        return None
    _, slope, r2 = regression_line(close, int(settings["trend_lookback"]))
    if not np.isfinite(slope):
        return None
    direction = "Upward ↑" if slope > 0 else ("Downward ↓" if slope < 0 else "Flat →")
    fx = "=X" in symbol.upper()
    unit = "pips" if fx else "points"
    unit_size = pip_size(symbol) if fx else 1.0
    return {
        "Symbol": symbol,
        "Price Trend": direction,
        "Trend Alignment": "Aligned ✓" if slope > 0 else ("Opposing ⚠" if slope < 0 else "Neutral"),
        "Bars Since Cross": bars_since,
        "Cross Time (Pacific)": pd.Timestamp(close.index[pos]).strftime("%Y-%m-%d %H:%M"),
        "Price at Cross": fmt_price(symbol, close.iloc[pos]),
        "Last Price": fmt_price(symbol, close.iloc[-1]),
        "Current 30 EMA": fmt_price(symbol, ema30_cross.iloc[-1]),
        f"Price Above EMA ({unit})": round((close.iloc[-1] - ema30_cross.iloc[-1]) / unit_size, 2),
        f"Move Since Cross ({unit})": round((close.iloc[-1] - close.iloc[pos]) / unit_size, 2),
        f"Price Trend Slope ({unit}/bar)": round(slope / unit_size, 3),
        "Price Trend R²": round(r2, 3) if np.isfinite(r2) else np.nan,
    }


def calculate_chart_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI shared by the TradingView chart and RSI scanner."""
    close = coerce_series(close)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    ratio = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + ratio)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)


def scan_rsi_oversold(symbol: str, timeframe: str, settings: dict) -> dict | None:
    """Return current RSI <= 30 symbols with closing-price regression trend."""
    daily = timeframe == "Daily"
    period = settings.get("daily_period", "1y") if daily else "1mo"
    interval = "1d" if daily else "1h"
    raw = fetch_market_ohlc(symbol, period=period, interval=interval)
    if raw.empty or len(raw) < max(35, int(settings["trend_lookback"]) // 2):
        return None
    close = coerce_series(raw["Close"]).dropna()
    rsi = calculate_chart_rsi(close, 14)
    current_rsi = safe_float(rsi.iloc[-1])
    if not np.isfinite(current_rsi) or current_rsi > 30.0:
        return None

    _, slope, r2 = regression_line(close, int(settings["trend_lookback"]))
    if not np.isfinite(slope):
        return None
    direction = "Upward ↑" if slope > 0 else ("Downward ↓" if slope < 0 else "Flat →")
    size = pip_size(symbol) if "=X" in symbol.upper() else 1.0
    unit = "pips/bar" if "=X" in symbol.upper() else "points/bar"
    return {
        "Symbol": symbol,
        "RSI (14)": round(current_rsi, 2),
        "Price Trend": direction,
        "Last Price": fmt_price(symbol, close.iloc[-1]),
        "Trend Slope": round(slope / size, 3),
        "Slope Unit": unit,
        "Trend R²": round(r2, 3) if np.isfinite(r2) else np.nan,
        "RSI Change (1 bar)": round(current_rsi - safe_float(rsi.iloc[-2]), 2),
        "Last Bar (Pacific)": close.index[-1].strftime("%Y-%m-%d %H:%M"),
    }


# =========================
# TradingView-inspired chart (independent of trade-plan chart)
# =========================
def _supertrend_10_3(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Standard SuperTrend: Wilder ATR(10), multiplier 3, trailing final bands."""
    high = coerce_series(df["High"])
    low = coerce_series(df["Low"])
    close = coerce_series(df["Close"])
    mid = (high + low) / 2.0
    vol = atr(df, 10)
    upper = (mid + 3.0 * vol).to_numpy(dtype=float)
    lower = (mid - 3.0 * vol).to_numpy(dtype=float)
    prices = close.to_numpy(dtype=float)
    n = len(df)
    final_up, final_low = upper.copy(), lower.copy()
    bullish = np.ones(n, dtype=bool)
    for i in range(1, n):
        if upper[i] < final_up[i-1] or prices[i-1] > final_up[i-1]:
            final_up[i] = upper[i]
        else:
            final_up[i] = final_up[i-1]
        if lower[i] > final_low[i-1] or prices[i-1] < final_low[i-1]:
            final_low[i] = lower[i]
        else:
            final_low[i] = final_low[i-1]
        if prices[i] > final_up[i-1]:
            bullish[i] = True
        elif prices[i] < final_low[i-1]:
            bullish[i] = False
        else:
            bullish[i] = bullish[i-1]
    line = np.where(bullish, final_low, final_up)
    return pd.Series(line, index=df.index), pd.Series(bullish, index=df.index)


def render_tradingview_style_chart(symbol: str, df: pd.DataFrame, timeframe: str) -> None:
    """Candlesticks with a single, dual-axis RSI/MACD overlay panel."""
    from matplotlib.patches import Rectangle
    if df.empty or len(df) < 35:
        st.info("At least 35 price bars are needed to display this chart.")
        return

    display_df = df.tail(180 if timeframe == "Hourly" else 160).copy()
    x = np.arange(len(display_df), dtype=float)
    full_close = coerce_series(df["Close"])
    fast = ema(full_close, 9).loc[display_df.index]
    slow = full_close.rolling(35, min_periods=35).mean().loc[display_df.index]
    _, full_bb_up, full_bb_low = bollinger(full_close, 20, 2)
    bb_up = full_bb_up.loc[display_df.index]
    bb_low = full_bb_low.loc[display_df.index]
    superline, superup = _supertrend_10_3(df)
    superline = superline.loc[display_df.index]
    superup = superup.loc[display_df.index]
    full_macd = ema(full_close, 12) - ema(full_close, 26)
    full_signal = ema(full_macd, 9)
    macd = full_macd.loc[display_df.index]
    signal = full_signal.loc[display_df.index]
    histogram = macd - signal

    # Share RSI computation with the RSI <= 30 scanner to prevent divergences.
    full_rsi = calculate_chart_rsi(full_close, 14)
    rsi = full_rsi.loc[display_df.index]
    rsi_signal = full_rsi.ewm(span=9, adjust=False, min_periods=1).mean().loc[display_df.index]

    # One shared lower panel: RSI is on the right axis (0-100), while MACD
    # keeps its true units on a left-side secondary axis.
    fig = plt.figure(figsize=(15.2, 8.7), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.45], hspace=0.13)
    ax = fig.add_subplot(gs[0])
    rsi_ax = fig.add_subplot(gs[1], sharex=ax)
    fig.subplots_adjust(left=0.09, right=0.865, top=0.93, bottom=0.105)
    fig.suptitle(f"{symbol.replace('=X', '').replace('USD', '/USD') if symbol == 'GBPUSD=X' else symbol} · {('1h' if timeframe == 'Hourly' else '1D')}  |  Candlesticks & Trend Indicators",
                 fontsize=14, fontweight="bold", y=0.975)

    up_color, down_color = "#0f9d82", "#e04451"
    candle_width = 0.65
    o = display_df["Open"].to_numpy(dtype=float)
    h = display_df["High"].to_numpy(dtype=float)
    l = display_df["Low"].to_numpy(dtype=float)
    c = display_df["Close"].to_numpy(dtype=float)
    for i in range(len(display_df)):
        rising = c[i] >= o[i]
        color = up_color if rising else down_color
        ax.vlines(x[i], l[i], h[i], color=color, linewidth=0.85, zorder=4)
        body_bottom = min(o[i], c[i])
        body_height = max(abs(c[i]-o[i]), 1e-8)
        ax.add_patch(Rectangle((x[i]-candle_width/2, body_bottom),
                               candle_width, body_height, facecolor=color,
                               edgecolor=color, lw=0.5, zorder=5))
    ax.fill_between(x, bb_low.to_numpy(dtype=float), bb_up.to_numpy(dtype=float),
                    color="#84bdea", alpha=0.12, label="BB (20, 2)", zorder=0)
    ax.plot(x, bb_up.to_numpy(dtype=float), color="#acc6dc", alpha=.55, linewidth=.75)
    ax.plot(x, bb_low.to_numpy(dtype=float), color="#acc6dc", alpha=.55, linewidth=.75)
    ax.plot(x, fast.to_numpy(dtype=float), color="#6034ae", linewidth=2.0, label="EMA 9", zorder=7)
    ax.plot(x, slow.to_numpy(dtype=float), color="#1686c4", linewidth=2.2, label="SMA 35", zorder=6)
    su = superup.to_numpy(dtype=bool)
    stline = superline.to_numpy(dtype=float)
    # Use separate series to avoid drawing vertical connections across trend flips.
    ax.plot(x, np.where(su, stline, np.nan), color="#237e52", linewidth=2.4,
            label="SuperTrend Bull", drawstyle="steps-post", zorder=6)
    ax.plot(x, np.where(~su, stline, np.nan), color="#f5a53b", linewidth=2.4,
            label="SuperTrend Bear", drawstyle="steps-post", zorder=6)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(True, alpha=.13)
    ax.legend(loc="upper left", fontsize=8, ncol=3, framealpha=.85)
    ax.tick_params(axis="x", labelbottom=False)

    # Screenshot-style COMBINED RSI + MACD panel.
    # RSI reference levels and the true RSI curve use the right-hand scale.
    # MACD and its signal use a separate, zero-centered left scale; they are
    # overlaid in the same panel, NOT misrepresented as RSI numbers.
    rsi_ax.axhspan(30, 70, color="#8e72bc", alpha=0.10, zorder=0)
    rsi_ax.axhline(70, color="#404040", linewidth=2.0,
                   linestyle=(0, (7, 7)), zorder=1)
    rsi_ax.axhline(50, color="#5178ef", linewidth=1.25,
                   linestyle="--", zorder=1)
    rsi_ax.axhline(30, color="#777777", linewidth=1.8, zorder=1)
    rsi_ax.plot(x, rsi.to_numpy(dtype=float), color="#9774b5",
                linewidth=1.5, alpha=.72, label="RSI 14", zorder=3)
    rsi_ax.plot(x, rsi_signal.to_numpy(dtype=float), color="#b9a2cf",
                linewidth=1.35, linestyle=":", alpha=.85,
                label="RSI EMA 9", zorder=3)
    rsi_ax.set_ylim(15, 85)
    rsi_ax.set_yticks([30, 40, 50, 60, 70])
    rsi_ax.yaxis.tick_right()
    rsi_ax.yaxis.set_label_position("right")
    rsi_ax.grid(True, alpha=.12)

    # Make the left axis transparent so it does not obscure the RSI shading.
    macd_ax = rsi_ax.twinx()
    macd_ax.yaxis.tick_left()
    macd_ax.yaxis.set_label_position("left")
    macd_ax.patch.set_visible(False)
    macd_ax.spines["right"].set_visible(False)
    all_values = np.r_[macd.to_numpy(dtype=float),
                       signal.to_numpy(dtype=float),
                       histogram.to_numpy(dtype=float)]
    finite_values = all_values[np.isfinite(all_values)]
    limit = max(float(np.max(np.abs(finite_values))) * 1.16, 1e-8) if finite_values.size else 1.0
    # Zero MACD aligns precisely with the RSI 50 reference line.
    macd_ax.set_ylim(-limit * 1.75, limit * 1.75)
    macd_ax.axhline(0, color="#777777", linewidth=.7, alpha=.5, zorder=1)
    macd_ax.bar(x, histogram.to_numpy(dtype=float), width=.65,
                color=np.where(histogram.to_numpy(dtype=float) >= 0,
                               "#0f9d82", "#e04451"),
                alpha=.16, label="MACD histogram", zorder=2)
    macd_ax.plot(x, macd.to_numpy(dtype=float), color="#248ceb",
                 linewidth=2.9, label="MACD 12,26", zorder=5)
    macd_ax.plot(x, signal.to_numpy(dtype=float), color="#ee7c29",
                 linewidth=2.9, label="MACD signal 9", zorder=5)
    macd_ax.tick_params(axis="y", labelsize=8, colors="#555555")
    rsi_ax.set_title("RSI (right scale) + MACD (left scale) — combined",
                     loc="left", fontsize=10)
    # One compact legend outside the data-heavy portion of the plot.
    lines1, labels1 = rsi_ax.get_legend_handles_labels()
    lines2, labels2 = macd_ax.get_legend_handles_labels()
    rsi_ax.legend(lines2 + lines1, labels2 + labels1,
                  loc="upper left", fontsize=7, ncol=3,
                  framealpha=.82, borderpad=.35)

    tick_count = min(8, len(display_df))
    indices = np.linspace(0, len(display_df)-1, tick_count, dtype=int)
    rsi_ax.set_xticks(indices)
    rsi_ax.set_xticklabels([
        display_df.index[i].strftime(
            "%m-%d %H:%M" if timeframe == "Hourly" else "%Y-%m-%d"
        ) for i in indices
    ], fontsize=8)
    rsi_ax.set_xlim(-1, len(display_df) + 3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =========================
# Range Chart: only what held up in testing
# =========================
# Likely range over the next 1-5 trading days: quantiles of each market's past
# moves, measured in units of the symbol's last 60 days of volatility (log
# returns / (vol * sqrt(days))). Fitted on this app's 31 pairs and 41 stocks,
# 2013-2026. Checked year by year (2018-2026), each year using only earlier
# years: the real close landed inside the 80% band 77-84% of the time (Forex)
# and 76-84% (Stocks), every pair between 77% and 83%. Direction is not forecast.
RANGE_VOL_DAYS = 60
RANGE_DAYS = 5
RANGE_Q = {  # 10%, 25%, 50%, 75%, 90%
    "Forex": {1: (-1.224, -0.5988, 0.0163, 0.6085, 1.2255), 2: (-1.2273, -0.5953, 0.0176, 0.6223, 1.2176),
              3: (-1.237, -0.5992, 0.0175, 0.6283, 1.229), 4: (-1.2376, -0.6037, 0.0171, 0.6306, 1.2332),
              5: (-1.2305, -0.6047, 0.0165, 0.6321, 1.226)},
    "Stocks": {1: (-1.1523, -0.5099, 0.035, 0.6086, 1.2044), 2: (-1.1672, -0.5182, 0.0621, 0.6437, 1.2248),
               3: (-1.1775, -0.5174, 0.0751, 0.6605, 1.2519), 4: (-1.1848, -0.5161, 0.0882, 0.6786, 1.2674),
               5: (-1.1749, -0.511, 0.0986, 0.6918, 1.278)},
}
RANGE_RECORD = {
    "Forex": "Checked 2018–2026 on all 31 pairs, each year using only earlier years: the real close landed inside the 80% band 77–84% of the time and inside the 50% band 47–54%.",
    "Stocks": "Checked 2018–2026 on all 41 stocks and funds, each year using only earlier years: the real close landed inside the 80% band 76–84% of the time and inside the 50% band 44–56%.",
}
# Stocks: daily RSI(14) at 30 or below while the 120-day trend line slopes down,
# held 20 trading days. Measured with dip_events on this app's 41 stocks since
# mid-2017: 323 signals, 64% rose against 57% for any stock on any day.
STOCK_DIP_RSI = 30
STOCK_DIP_HOLD = 20
STOCK_DIP_RECORD = ("Since mid-2017 on this app's 41 stocks: 323 signals, 64% rose over 20 trading days (average +5.8% "
                    "after costs), against 57% for any stock on any day (+3.4%). Weaker in 2020, 2024 and 2026 (54–58%). "
                    "A few big rebounds lift the average, so buy small, equal amounts.")
FX_RSI_DOWN_RECORD = "Over Nov 2021–Nov 2026 on this app's pairs: rose within 20 trading days 74% of the time, against 54% for any pair on any day. The Stochastic rule (its own tab) did better."
COST_RATE = {"Forex": 0.0003, "Stocks": 0.0005}
SESSIONS = (("Tokyo", "Asia/Tokyo", 9, 18, "#f2c14e"), ("London", "Europe/London", 8, 17, "#5b8def"),
            ("New York", "America/New_York", 8, 17, "#e86f5c"))


def bar_days(index) -> pd.DatetimeIndex:
    """Trading date of each daily bar. Yahoo stamps them at midnight in the market's own
    time zone, so after converting to Pacific a bar can show the previous day."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC") + pd.Timedelta(hours=12)
        idx = idx.tz_localize(None)
    return idx.normalize()


def weekday_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Yahoo's weekend FX bars (they hold Friday's late trading)."""
    return df[bar_days(df.index).day_of_week < 5]


def trend_slope_series(close: pd.Series, window: int = 120) -> pd.Series:
    """Slope of a least-squares line through each bar's last `window` closes."""
    y = coerce_series(close).astype(float)
    x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    n = window
    sx, sy = x.rolling(n).sum(), y.rolling(n).sum()
    sxy, sxx = (x * y).rolling(n).sum(), (x * x).rolling(n).sum()
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)


def range_forecast(daily: pd.DataFrame, market: str) -> dict | None:
    close = coerce_series(weekday_bars(daily)["Close"]).dropna().astype(float)
    if len(close) < RANGE_VOL_DAYS + 1:
        return None
    vol = float(np.log(close).diff().tail(RANGE_VOL_DAYS).std())
    if not np.isfinite(vol) or vol <= 0:
        return None
    last = float(close.iloc[-1])
    days = pd.bdate_range(bar_days(close.index)[-1] + pd.Timedelta(days=1), periods=RANGE_DAYS)
    rows = []
    for h, day in zip(range(1, RANGE_DAYS + 1), days):
        lo, lo50, _, hi50, hi = (last * math.exp(q * vol * math.sqrt(h)) for q in RANGE_Q[market][h])
        rows.append({"day": day, "low": lo, "low50": lo50, "high50": hi50, "high": hi})
    return {"last": last, "vol": vol, "days": rows}


def dip_events(daily: pd.DataFrame, market: str, since: str = "2017-06-01") -> pd.DataFrame:
    """Each past tested dip buy for one symbol: start, exit and result after costs.

    Forex: daily Stochastic %K(14) <= FX_STOCH_LOW, sold when RSI(14) reaches
    FX_STOCH_EXIT_RSI or after FX_STOCH_MAX_DAYS (one position at a time).
    Stocks: the first day RSI(14) is at STOCK_DIP_RSI or below with the 120-day
    trend line sloping down, held STOCK_DIP_HOLD trading days (one at a time).
    """
    df = weekday_bars(daily)
    close = coerce_series(df["Close"]).astype(float).to_numpy()
    rsi = calculate_chart_rsi(df["Close"], 14).to_numpy(float)
    days = bar_days(df.index)
    first = int(np.searchsorted(days, pd.Timestamp(since)))
    cost = COST_RATE[market]
    out = []
    if market == "Forex":
        k = stochastic_k(df, 14).to_numpy(float)
        i = max(first, 14)
        while i < len(df):
            if not (np.isfinite(k[i]) and k[i] <= FX_STOCH_LOW):
                i += 1
                continue
            j = i + 1
            while j < len(df) and j - i < FX_STOCH_MAX_DAYS and not rsi[j] >= FX_STOCH_EXIT_RSI:
                j += 1
            done = j < len(df)
            j = min(j, len(df) - 1)
            out.append({"start": i, "end": j, "open": not done,
                        "result": close[j] / close[i] - 1 - cost})
            i = j + 1
    else:
        slope = trend_slope_series(df["Close"], 120).to_numpy(float)
        on = (rsi <= STOCK_DIP_RSI) & (slope < 0)
        i = max(first, 1)
        while i < len(df):
            if not (on[i] and not on[i - 1]):
                i += 1
                continue
            j = min(i + STOCK_DIP_HOLD, len(df) - 1)
            out.append({"start": i, "end": j, "open": i + STOCK_DIP_HOLD >= len(df),
                        "result": close[j] / close[i] - 1 - cost})
            i = j + 1
    ev = pd.DataFrame(out, columns=["start", "end", "open", "result"])
    ev["date"] = [days[i] for i in ev["start"]]
    return ev


def dip_record_text(symbol: str, ev: pd.DataFrame, market: str) -> str:
    done = ev[~ev["open"]]
    name = symbol.replace("=X", "")
    if done.empty:
        return f"{name} has had no finished dip signal since 2017."
    rose = int((done["result"] > 0).sum())
    txt = (f"{name}: {len(done)} past dip signals since 2017, {rose} rose ({100 * rose / len(done):.0f}%), "
           f"average {100 * done['result'].mean():+.2f}% after costs.")
    if len(done) < 15:
        txt += " Too few to judge on their own; the all-symbols record is the better guide."
    return txt


def dip_now(daily: pd.DataFrame, market: str) -> dict:
    df = weekday_bars(daily)
    rsi = calculate_chart_rsi(df["Close"], 14)
    if market == "Forex":
        k = safe_float(stochastic_k(df, 14).iloc[-1])
        return {"on": k <= FX_STOCH_LOW, "text": f"Daily Stochastic %K is {k:.1f} (buy level {FX_STOCH_LOW} or lower); RSI {safe_float(rsi.iloc[-1]):.1f} (sell level {FX_STOCH_EXIT_RSI})."}
    r = safe_float(rsi.iloc[-1])
    down = safe_float(trend_slope_series(df["Close"], 120).iloc[-1]) < 0
    return {"on": r <= STOCK_DIP_RSI and down,
            "text": f"Daily RSI is {r:.1f} (buy level {STOCK_DIP_RSI} or lower); 120-day trend is {'down' if down else 'up'} (needs down)."}


def quote_to_usd(symbol: str) -> float:
    """USD value of one unit of the pair's quote currency (1.0 for stocks)."""
    if "=X" not in symbol:
        return 1.0
    quote = symbol.replace("=X", "")[3:6]
    if quote == "USD":
        return 1.0
    for sym, invert in ((f"USD{quote}=X", True), (f"{quote}USD=X", False)):
        d = fetch_market_ohlc(sym, "5d", "1d")
        if not d.empty:
            px = float(d["Close"].iloc[-1])
            return 1.0 / px if invert else px
    return float("nan")


def _candles(ax, x, o, h, l, c, width=0.65):
    up = c >= o
    color = np.where(up, "#0f9d82", "#e04451")
    ax.vlines(x, l, h, colors=color, linewidth=0.8, zorder=3)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-12), bottom=np.minimum(o, c), width=width,
           color=color, edgecolor=color, linewidth=0.3, zorder=4)


def plot_range_chart(symbol: str, market: str, daily: pd.DataFrame, hist: pd.DataFrame,
                     fc: dict | None, ev: pd.DataFrame, show_untested: bool, position: dict | None):
    d = weekday_bars(daily)
    n = len(d)
    x = np.arange(n, dtype=float)
    fig = plt.figure(figsize=(15.2, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.3], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    osc = fig.add_subplot(gs[1], sharex=ax)
    fig.subplots_adjust(left=0.04, right=0.86, top=0.94, bottom=0.07)
    name = symbol.replace("=X", "")
    fig.suptitle(f"{name} · Daily  |  likely range for the next {RANGE_DAYS} trading days", fontsize=14,
                 fontweight="bold", y=0.985)
    _candles(ax, x, *(d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")))
    if show_untested:
        full = coerce_series(hist["Close"])
        hm = hma(full, 55).reindex(d.index)
        _, bu, bl = bollinger(full, 20, 2)
        ax.plot(x, hm.to_numpy(float), color="#8a63d2", lw=1.2, alpha=.8, label="HMA 55 (untested)")
        ax.fill_between(x, bl.reindex(d.index).to_numpy(float), bu.reindex(d.index).to_numpy(float),
                        color="#84bdea", alpha=.10, label="Bollinger 20, 2 (untested)")
    if fc:
        fx_ = np.r_[n - 1, n - 1 + np.arange(1, RANGE_DAYS + 1)]
        col = lambda k: np.r_[fc["last"], [r[k] for r in fc["days"]]]  # noqa: E731
        ax.fill_between(fx_, col("low"), col("high"), color="#3b7dd8", alpha=.13, lw=0, label="80% likely range")
        ax.fill_between(fx_, col("low50"), col("high50"), color="#3b7dd8", alpha=.28, lw=0, label="50% likely range")
        last_day = fc["days"][-1]
        for key, va in (("high", "bottom"), ("high50", "bottom"), ("low50", "top"), ("low", "top")):
            ax.annotate(fmt_price(symbol, last_day[key]), (fx_[-1], last_day[key]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color="#1f4f99", fontweight="bold" if key in ("high", "low") else "normal")
    if position and position.get("entry"):
        colour = "#0f9d82" if position["side"] == "Long" else "#e04451"
        ax.axhline(position["entry"], color=colour, ls="--", lw=1.2, zorder=2)
        ax.annotate(f"Your {position['side'].lower()} {fmt_price(symbol, position['entry'])}", (0, position["entry"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8, color=colour)
    ax.yaxis.tick_right()
    ax.grid(True, alpha=.13)
    ax.legend(loc="upper left", fontsize=8, ncol=4, framealpha=.85)
    ax.set_xlim(-1, n + RANGE_DAYS + max(8, n // 18))
    ax.tick_params(axis="x", labelbottom=False)

    # Lower panel: the tested buy signal, with every past signal in view marked by how it ended.
    hd = weekday_bars(hist)
    rsi = calculate_chart_rsi(hd["Close"], 14).reindex(d.index)
    if market == "Forex":
        k = stochastic_k(hd, 14).reindex(d.index)
        osc.plot(x, k.to_numpy(float), color="#1686c4", lw=1.3, label="Stochastic %K 14")
        osc.plot(x, rsi.to_numpy(float), color="#9774b5", lw=1.1, alpha=.8, label="RSI 14")
        osc.axhline(FX_STOCH_LOW, color="#0f9d82", lw=1.4)
        osc.axhline(FX_STOCH_EXIT_RSI, color="#9774b5", lw=1.0, ls="--")
        osc.text(n + 0.5, FX_STOCH_LOW, f" buy ≤ {FX_STOCH_LOW}", va="center", fontsize=8, color="#0f9d82")
        osc.text(n + 0.5, FX_STOCH_EXIT_RSI, f" sell at RSI {FX_STOCH_EXIT_RSI}", va="center", fontsize=8, color="#9774b5")
        title = "Tested buy signal: Stochastic ≤ 10, sold when RSI reaches 50"
    else:
        osc.plot(x, rsi.to_numpy(float), color="#9774b5", lw=1.3, label="RSI 14")
        osc.axhline(STOCK_DIP_RSI, color="#0f9d82", lw=1.4)
        osc.text(n + 0.5, STOCK_DIP_RSI, f" buy ≤ {STOCK_DIP_RSI} in a downtrend", va="center", fontsize=8, color="#0f9d82")
        title = f"Tested buy signal: RSI ≤ {STOCK_DIP_RSI} with the 120-day trend down, held {STOCK_DIP_HOLD} trading days"
    pos = {day: i for i, day in enumerate(bar_days(d.index))}
    hdays = bar_days(hd.index)
    for _, e in ev.iterrows():
        i = pos.get(hdays[int(e["start"])])
        if i is None:
            continue
        colour = "#999999" if e["open"] else ("#0f9d82" if e["result"] > 0 else "#e04451")
        osc.scatter([i], [4], marker="^", s=55, color=colour, zorder=5)
        ax.scatter([i], [d["Low"].iloc[i]], marker="^", s=55, color=colour, zorder=6)
    osc.scatter([], [], marker="^", color="#0f9d82", label="past signal: rose")
    osc.scatter([], [], marker="^", color="#e04451", label="fell")
    osc.scatter([], [], marker="^", color="#999999", label="still open")
    osc.set_ylim(0, 100)
    osc.yaxis.tick_right()
    osc.grid(True, alpha=.12)
    osc.set_title(title, loc="left", fontsize=10)
    osc.legend(loc="upper left", fontsize=7, ncol=5, framealpha=.85)
    ticks = np.linspace(0, n - 1, min(8, n), dtype=int)
    osc.set_xticks(ticks)
    osc.set_xticklabels([bar_days(d.index)[i].strftime("%Y-%m-%d") for i in ticks], fontsize=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_intraday_timing(symbol: str, market: str, intra: pd.DataFrame, daily: pd.DataFrame, interval: str) -> str:
    """Intraday bars with today's range against a typical day's range and the trading sessions.
    No buy or sell labels: nothing on intraday bars tested better than chance."""
    d = weekday_bars(daily)
    rng = (d["High"] - d["Low"]).astype(float)
    typical = float(rng.iloc[-21:-1].median())
    utc = intra.index.tz_convert("UTC")
    if market == "Forex":  # Yahoo's daily FX bars run midnight to midnight London time
        day_key = intra.index.tz_convert("Europe/London").normalize()
    else:
        day_key = intra.index.tz_convert("America/New_York").normalize()
    today = intra[day_key == day_key[-1]]
    lo, hi = float(today["Low"].min()), float(today["High"].max())
    unit = pip_size(symbol) if market == "Forex" else 1.0
    unit_name = "pips" if market == "Forex" else "$"
    fmt_u = (lambda v: f"{v / unit:,.0f} pips") if market == "Forex" else (lambda v: f"${v:,.2f}")  # noqa: E731
    used = 100 * (hi - lo) / typical if typical > 0 else float("nan")

    n = len(intra)
    x = np.arange(n, dtype=float)
    fig, ax = plt.subplots(figsize=(15.2, 4.6), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.14)
    _candles(ax, x, *(intra[k].to_numpy(float) for k in ("Open", "High", "Low", "Close")), width=0.6)
    t0 = int(np.argmax(day_key == day_key[-1]))
    for y, label, colour in ((lo + typical, "low + typical range", "#0f9d82"),
                             (hi - typical, "high − typical range", "#e04451")):
        ax.hlines(y, t0, n + 2, colors=colour, linestyles="--", lw=1.1)
        ax.annotate(f"{fmt_price(symbol, y)}  {label}", (n + 2, y), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=colour)
    ax.axvline(t0 - 0.5, color="#888888", lw=0.8, ls=":")
    ymin, ymax = ax.get_ylim()
    strip = (ymax - ymin) * 0.035
    for k, (label, tz, start, end, colour) in enumerate(SESSIONS):
        local = intra.index.tz_convert(tz)
        inside = (local.hour >= start) & (local.hour < end) & (local.day_of_week < 5)
        base = ymin - strip * (k + 1)
        ax.fill_between(x, base, base + strip * 0.8, where=inside, color=colour, alpha=.55, lw=0, step="mid")
        ax.text(-2, base + strip * 0.4, label, fontsize=7, va="center", ha="right", color=colour)
    ax.set_ylim(ymin - strip * (len(SESSIONS) + 0.3), ymax)
    ax.yaxis.tick_right()
    ax.grid(True, alpha=.12)
    ax.set_xlim(-1, n + 3)
    ticks = np.linspace(0, n - 1, min(9, n), dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([intra.index[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    ax.set_title(f"{symbol.replace('=X', '')} · {interval} (Pacific time) — for timing only, not a signal",
                 loc="left", fontsize=11, fontweight="bold")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    return (f"Today's range so far: {fmt_u(hi - lo)}, {used:.0f}% of a typical day's {fmt_u(typical)} "
            f"(median of the last 20 days). On about half of days the full range ends up bigger than the typical one. "
            f"Near today's low with plenty of the typical range left is a calmer moment to buy than after a stretched run; "
            f"this is description, not a tested signal.")


def _position_from_query(symbol: str) -> dict:
    raw = st.query_params.get("pos", "")
    parts = str(raw).split(",")
    if len(parts) == 6 and parts[0] == symbol:
        try:
            return {"side": parts[1], "entry": float(parts[2]), "size": float(parts[3]),
                    "swap": float(parts[4]), "opened": pd.Timestamp(parts[5]).date()}
        except Exception:
            pass
    return {"side": "None", "entry": 0.0, "size": 0.0, "swap": 0.0, "opened": datetime.now(PACIFIC).date()}


def render_range_chart(symbol: str, market: str, cfg: dict) -> None:
    hist = fetch_market_ohlc(symbol, "10y", "1d")
    if hist.empty or len(hist) < 200:
        st.warning(f"Not enough daily history for {symbol}.")
        return
    show_days = {"1mo": 22, "3mo": 64, "6mo": 128, "1y": 254, "2y": 506, "5y": 1262, "10y": 2520}
    daily = hist.tail(show_days.get(cfg.get("daily_period", "1y"), 254))
    fc = range_forecast(hist, market)
    ev = dip_events(hist, market)
    now = dip_now(hist, market)

    with st.expander("Your position (optional)", expanded=False):
        saved = _position_from_query(symbol)
        c1, c2, c3, c4, c5 = st.columns(5)
        side = c1.selectbox("Side", ["None", "Long", "Short"], index=["None", "Long", "Short"].index(saved["side"]),
                            key=f"pos_side_{symbol}")
        entry = c2.number_input("Entry price", value=float(saved["entry"]), min_value=0.0, format="%.5f",
                                key=f"pos_entry_{symbol}")
        size = c3.number_input("Size (units or shares)", value=float(saved["size"]), min_value=0.0, step=1000.0,
                               key=f"pos_size_{symbol}")
        swap = c4.number_input("Swap per night, USD (from your broker; minus if you pay)", value=float(saved["swap"]),
                               step=1.0, key=f"pos_swap_{symbol}")
        opened = c5.date_input("Opened on", value=saved["opened"], key=f"pos_open_{symbol}")
        position = None
        if side != "None" and entry > 0 and size > 0:
            position = {"side": side, "entry": entry, "size": size, "swap": swap, "opened": opened}
            try:
                st.query_params["pos"] = f"{symbol},{side},{entry},{size},{swap},{opened.isoformat()}"
            except Exception:
                pass
        elif side == "None":
            try:
                if "pos" in st.query_params:
                    del st.query_params["pos"]
            except Exception:
                pass
        st.caption("Saved in this page's link only, so bookmark it to keep it. Nothing is sent anywhere.")

    if position and fc:
        sign = 1 if position["side"] == "Long" else -1
        to_usd = quote_to_usd(symbol)
        pl = sign * (fc["last"] - position["entry"]) * position["size"] * to_usd
        nights = max(0, (datetime.now(PACIFIC).date() - position["opened"]).days)
        carry = position["swap"] * nights
        d5 = fc["days"][-1]
        pl_a = sign * (d5["low"] - position["entry"]) * position["size"] * to_usd
        pl_b = sign * (d5["high"] - position["entry"]) * position["size"] * to_usd
        m = st.columns(4)
        m[0].metric("Profit / loss now", f"${pl:,.0f}")
        m[1].metric(f"Carry so far ({nights} nights)", f"${carry:,.0f}" if position["swap"] else "enter swap",
                    help="Swap per night × nights since opened. Brokers charge three nights on Wednesdays, so this is close, not exact.")
        m[2].metric("Carry per 30 days", f"${30 * position['swap']:,.0f}" if position["swap"] else "—")
        m[3].metric(f"P/L range by {d5['day']:%a %d %b} (80%)", f"${min(pl_a, pl_b):,.0f} to ${max(pl_a, pl_b):,.0f}")
        if now["on"] and position["side"] == "Short":
            st.warning("The tested dip buy signal is on for this symbol, which works against your short: "
                       "past signals like this rose most of the time. " + now["text"])
        elif now["on"] and position["side"] == "Long":
            st.success("The tested dip buy signal is on, which is in favour of your long. " + now["text"])

    if now["on"]:
        st.success(f"Tested dip buy signal is ON. {now['text']}")
    else:
        st.caption(f"Tested dip buy signal is off. {now['text']}")

    show_untested = st.checkbox("Show untested indicators (HMA 55, Bollinger Bands)", value=False,
                                key="range_untested",
                                help="In a 5-year backtest of this app's tabs, HMA and other indicator crosses did no better than chance.")
    plot_range_chart(symbol, market, daily, hist, fc, ev, show_untested, position)

    rec_all = STOCK_DIP_RECORD if market == "Stocks" else fx_stochastic_history_text()
    st.markdown(f"**Dip record — {dip_record_text(symbol, ev, market)}** All {market.lower()}: {rec_all}")
    if fc:
        tbl = pd.DataFrame([{"Day": f"{r['day']:%a %d %b}", "80% range": f"{fmt_price(symbol, r['low'])} – {fmt_price(symbol, r['high'])}",
                             "50% range": f"{fmt_price(symbol, r['low50'])} – {fmt_price(symbol, r['high50'])}"} for r in fc["days"]])
        with st.expander("Likely range by day", expanded=False):
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.caption(f"{RANGE_RECORD[market]} The range says how far the price is likely to move, not which way: "
                   "no direction forecast tested better than chance.")

    st.markdown("#### Intraday, for timing")
    intra = fetch_market_ohlc(symbol, cfg["period"], cfg["interval"])
    if intra.empty or len(intra) < 10:
        st.info("No intraday data right now.")
    else:
        st.caption(plot_intraday_timing(symbol, market, intra, hist, cfg["interval"]))
    st.caption("Not financial advice. Past results do not guarantee future results.")


# =========================
# Indicator Chart (TradingView-style layout)
# =========================
IND_TIMEFRAMES = {"15m": ("60d", "15m", None), "1h": ("60d", "1h", None), "4h": ("730d", "1h", "4h"),
                  "1d": ("2y", "1d", None)}
IND_RECORD = (
    "Tested on 16 Forex pairs, hourly bars, Nov 2024–Nov 2026: when all six readings below were bearish, the price "
    "was lower 24 hours later 47% of the time and lower 5 days later 47.5% (AUD/JPY: 46% and 42.5%). When all six "
    "were bullish, it was higher 51% and 53% of the time. Neither beat the pair's own drift by more than 0.05%. "
    "Earlier tests found trading SuperTrend and EMA/SMA crosses on hourly bars lost money. "
    "These lines describe the move so far; they did not forecast the next one."
)


def ind_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """SuperTrend as TradingView draws it (Wilder ATR). Returns (line, is_up)."""
    h, l, c = (df[k].to_numpy(float) for k in ("High", "Low", "Close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    mid = (h + l) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    fu, fl = ub.copy(), lb.copy()
    line, up = np.full(len(c), np.nan), np.ones(len(c), bool)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        up[i] = c[i] >= fl[i] if up[i - 1] else c[i] > fu[i]
        line[i] = fl[i] if up[i] else fu[i]
    line[:period] = np.nan
    return pd.Series(line, df.index), pd.Series(up, df.index)


def ind_frame(df: pd.DataFrame) -> pd.DataFrame:
    c = coerce_series(df["Close"])
    out = pd.DataFrame(index=df.index)
    out["st"], out["st_up"] = ind_supertrend(df)
    out["bb_mid"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    out["bb_up"], out["bb_lo"] = out["bb_mid"] + 2 * sd, out["bb_mid"] - 2 * sd
    out["ema9"] = c.ewm(span=9, adjust=False).mean()
    out["sma35"] = c.rolling(35).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    out["roc"] = 100 * (c / c.shift(9) - 1)
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd"], out["signal"] = m, m.ewm(span=9, adjust=False).mean()
    return out


def ind_readings(c: float, row: pd.Series) -> list[tuple[str, bool]]:
    """(label, is_bearish) for the six readings in the record."""
    return [
        ("SuperTrend down" if not row["st_up"] else "SuperTrend up", not bool(row["st_up"])),
        ("below BB middle" if c < row["bb_mid"] else "above BB middle", c < row["bb_mid"]),
        ("EMA 9 under SMA 35" if row["ema9"] < row["sma35"] else "EMA 9 over SMA 35", row["ema9"] < row["sma35"]),
        ("MACD under signal" if row["macd"] < row["signal"] else "MACD over signal", row["macd"] < row["signal"]),
        (f"RSI {row['rsi']:.1f}", row["rsi"] < 50),
        (f"ROC 9 {row['roc']:+.2f}%", row["roc"] < 0),
    ]


def plot_indicator_chart(symbol: str, tf: str, bars: pd.DataFrame, ind: pd.DataFrame, position: dict | None):
    n = len(bars)
    x = np.arange(n, dtype=float)
    fig = plt.figure(figsize=(15.2, 9.6), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.4, 1.2, 1.2], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax)
    ax_m = fig.add_subplot(gs[2], sharex=ax)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.06)
    name = symbol.replace("=X", "")
    o, h, l, c = (bars[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    fig.suptitle(f"{name} · {tf}   O {o[-1]:.5g}  H {h[-1]:.5g}  L {l[-1]:.5g}  C {c[-1]:.5g}",
                 fontsize=14, fontweight="bold", y=0.99)

    ax.fill_between(x, ind["bb_lo"].to_numpy(float), ind["bb_up"].to_numpy(float), color="#84bdea", alpha=.12,
                    label="BB 20, 2")
    ax.plot(x, ind["bb_mid"].to_numpy(float), color="#84bdea", lw=1.0, alpha=.8)
    st_line = ind["st"].to_numpy(float)
    st_up = ind["st_up"].to_numpy(bool)
    ax.plot(x, np.where(st_up, st_line, np.nan), color="#0f9d82", lw=2.6, alpha=.75, label="SuperTrend 10, 3 (up)")
    ax.plot(x, np.where(~st_up, st_line, np.nan), color="#f0a030", lw=2.6, alpha=.85, label="SuperTrend 10, 3 (down)")
    ax.plot(x, ind["sma35"].to_numpy(float), color="#3b82d6", lw=2.0, label="SMA 35")
    ax.plot(x, ind["ema9"].to_numpy(float), color="#4b2fa8", lw=2.0, label="EMA 9")
    _candles(ax, x, o, h, l, c, width=0.6)
    ax.axhline(c[-1], color="#e04451", lw=0.8, ls=":")
    tags = [(c[-1], f"{c[-1]:.5g}", "#e04451"), (ind["st"].iloc[-1], f"{ind['st'].iloc[-1]:.5g}",
            "#0f9d82" if st_up[-1] else "#f0a030"), (ind["sma35"].iloc[-1], f"{ind['sma35'].iloc[-1]:.5g}", "#3b82d6"),
            (ind["ema9"].iloc[-1], f"{ind['ema9'].iloc[-1]:.5g}", "#4b2fa8")]
    if position:
        e = float(position["entry"])
        ax.axhline(e, color="#111", lw=1.1, ls="--")
        tags.append((e, f"{position['side']} {e:.5g}", "#111"))
        lo, hi = np.nanmin(l), np.nanmax(h)
        ax.set_ylim(min(lo, e) - 0.03 * (hi - lo), max(hi, e) + 0.03 * (hi - lo))
    for yv, txt, col in tags:
        if np.isfinite(yv):
            ax.annotate(txt, xy=(1.0, yv), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=10, color="white",
                        bbox=dict(boxstyle="square,pad=0.25", fc=col, ec=col))
    ax.yaxis.tick_right()
    ax.grid(alpha=.15)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, frameon=False, ncol=5, borderaxespad=0.2)
    plt.setp(ax.get_xticklabels(), visible=False)

    r = ind["rsi"].to_numpy(float)
    ax_rsi.set_facecolor("#f4f1fb")
    ax_rsi.plot(x, r, color="#2f80ed", lw=1.8, label="RSI 14")
    for lv, ls_, lw_ in ((70, "--", 1.6), (60, "-", .9), (50, "-", .9), (40, "-", .9), (30, "--", 1.6)):
        ax_rsi.axhline(lv, color="#111" if lv in (30, 70) else "#3355ee", ls=ls_, lw=lw_, alpha=.6)
    ax_rsi.set_ylim(20, 85)
    ax_rsi.yaxis.tick_right()
    ax_rsi.annotate(f"{r[-1]:.2f}", xy=(1.0, r[-1]), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=10, color="white",
                    bbox=dict(boxstyle="square,pad=0.25", fc="#2f80ed", ec="#2f80ed"))
    ax_rsi.legend(loc="upper left", fontsize=9, frameon=False)
    plt.setp(ax_rsi.get_xticklabels(), visible=False)

    mac, sig = ind["macd"].to_numpy(float), ind["signal"].to_numpy(float)
    hist_ = mac - sig
    ax_m.bar(x, hist_, color=np.where(hist_ >= 0, "#0f9d82", "#e04451"), alpha=.35, width=0.7)
    ax_m.plot(x, mac, color="#2f80ed", lw=1.6, label=f"MACD 12 26 9  {mac[-1]:.4g}")
    ax_m.plot(x, sig, color="#f07a2a", lw=1.6, label=f"signal {sig[-1]:.4g}")
    ax_m.axhline(0, color="#888", lw=.8)
    roc_ax = ax_m.twinx()
    roc_ax.yaxis.tick_left()
    ax_m.yaxis.tick_right()
    roc_ax.plot(x, ind["roc"].to_numpy(float), color="#999", lw=1.0, alpha=.8, label="ROC 9 (left, %)")
    roc_ax.yaxis.set_label_position("left")
    h1, l1 = ax_m.get_legend_handles_labels()
    h2, l2 = roc_ax.get_legend_handles_labels()
    ax_m.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False, ncol=3)
    ax_m.grid(alpha=.15)

    idx = bars.index
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    fmt = "%b %d" if tf == "1d" else "%b %d %H:%M"
    ax_m.set_xticks(ticks)
    ax_m.set_xticklabels([idx[i].strftime(fmt) for i in ticks], fontsize=9)
    ax_m.set_xlim(-1, n + 1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_indicator_chart(symbol: str) -> None:
    c1, c2 = st.columns(2)
    tf = c1.selectbox("Timeframe", list(IND_TIMEFRAMES), index=1, key="ind_tf")
    show = c2.slider("Bars shown", 60, 400, 150, 10, key="ind_bars")
    period, interval, resample = IND_TIMEFRAMES[tf]
    raw = fetch_market_ohlc(symbol, period, interval)
    if resample and not raw.empty:
        u = raw.tz_convert("UTC")
        u = u.resample(resample, origin="epoch").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(
            subset=["Open", "High", "Low", "Close"])
        raw = u.tz_convert(PACIFIC)
    if raw.empty or len(raw) < 60:
        st.warning(f"Not enough {tf} data for {symbol}.")
        return
    ind = ind_frame(raw)
    bars, ind_v = raw.tail(show), ind.tail(show)
    saved = _position_from_query(symbol)
    position = saved if saved["side"] != "None" and saved["entry"] > 0 else None
    plot_indicator_chart(symbol, tf, bars, ind_v, position)
    reads = ind_readings(float(raw["Close"].iloc[-1]), ind.iloc[-1])
    bear = sum(b for _, b in reads)
    st.markdown(f"**Now: {bear} of 6 readings bearish** — " + " · ".join(t for t, _ in reads))
    st.caption(IND_RECORD)
    st.caption(f"Times are Pacific. {'4h bars are built from hourly bars, so they can differ slightly from your broker. ' if resample else ''}"
               "Not financial advice. Past results do not guarantee future results.")


tab_range, tab_ind, tab_rsi_oversold, tab_fx_stoch, tab_rules = st.tabs([
    "Range Chart",
    "Indicator Chart",
    "RSI ≤ 30",
    "FX Stochastic Dip",
    "Rules",
])


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic %K: where the close sits in the last `period` bars' range (0-100)."""
    close = coerce_series(df["Close"])
    lo = coerce_series(df["Low"]).rolling(period, min_periods=period).min()
    hi = coerce_series(df["High"]).rolling(period, min_periods=period).max()
    return 100.0 * (close - lo) / (hi - lo).replace(0, np.nan)


def fx_stochastic_reading(symbol: str) -> dict | None:
    daily = fetch_market_ohlc(symbol, "1y", "1d")
    if daily.empty or len(daily) < 30:
        return None
    k = stochastic_k(daily, 14)
    r = calculate_chart_rsi(daily["Close"], 14)
    return {
        "Symbol": symbol,
        "Stochastic %K": round(safe_float(k.iloc[-1]), 1),
        "RSI (14)": round(safe_float(r.iloc[-1]), 1),
        "Last Close": fmt_price(symbol, daily["Close"].iloc[-1]),
        "Dip Signal": bool(safe_float(k.iloc[-1]) <= FX_STOCH_LOW),
    }


def fx_stochastic_history_text() -> str:
    hist = load_signal_history()
    row = hist[hist["State"] == "FX STOCH DIP"] if not hist.empty else hist
    if row is None or row.empty:
        return "No measured history yet."
    h = row.iloc[0]
    return (
        f"Past signals ({int(h['Signals']):,} since {h['Since']}): rose by the exit "
        f"{h['Target First %']:.0f}% of the time, average {h['Avg Result %']:+.2f}% after costs, "
        f"typically held {h['Median Bars']:.0f} days."
    )


def render_fx_stochastic_note(symbol: str) -> None:
    reading = fx_stochastic_reading(symbol)
    if reading is None:
        return
    if reading["Dip Signal"]:
        st.success(
            f"Forex Stochastic Dip: daily %K is {reading['Stochastic %K']:.1f} (≤ {FX_STOCH_LOW}). "
            f"Tested rule: buy, sell when RSI reaches {FX_STOCH_EXIT_RSI} or after {FX_STOCH_MAX_DAYS} days. "
            + fx_stochastic_history_text()
        )
    else:
        st.caption(
            f"Forex Stochastic Dip: daily %K is {reading['Stochastic %K']:.1f}; "
            f"the tested buy rule needs {FX_STOCH_LOW} or lower."
        )


def _chart_timeframe_config(base_cfg: dict, timeframe: str) -> dict:
    """Use current dashboard settings for hourly; full daily bars for Daily."""
    selected = base_cfg.copy()
    if timeframe == "Daily":
        selected["period"] = base_cfg.get("daily_period", "1y")
        selected["interval"] = "1d"
    selected["right_padding"] = max(4, int(base_cfg.get("right_padding", 12)))
    return selected


with tab_range:
    render_range_chart(symbol, asset_class, cfg)


with tab_ind:
    render_indicator_chart(symbol)


with tab_rsi_oversold:
    st.subheader(f"{asset_class} — RSI at or below 30")
    st.caption(
        "Daily RSI (14) at or below 30, split by the 120-day trend line. Only the downtrend list was tested "
        "with an edge: " + (STOCK_DIP_RECORD if asset_class == "Stocks" else FX_RSI_DOWN_RECORD)
        + " Hourly RSI was removed: nothing on hourly bars tested better than chance."
    )
    rsi_timeframe = "Daily"
    rsi_symbols = st.multiselect(
        "Symbols to scan", sorted(current_universe),
        default=sorted(current_universe),
        key=f"rsi_oversold_symbols_{asset_class}",
    )
    rsi_signature = (
        rsi_timeframe, tuple(sorted(rsi_symbols)),
        int(cfg["trend_lookback"]),
        str(cfg.get("daily_period", "1y")),
    )
    if st.button(
        "Scan RSI ≤ 30", key=f"rsi_oversold_run_{asset_class}",
        use_container_width=True, disabled=not bool(rsi_symbols),
    ):
        found, failures = [], []
        progress = st.progress(0)
        for i, sym in enumerate(rsi_symbols):
            try:
                result = scan_rsi_oversold(sym, rsi_timeframe, dict(cfg, daily_period="1y"))
                if result is not None:
                    found.append(result)
            except Exception as exc:
                failures.append(f"{sym}: {exc}")
            progress.progress((i + 1) / max(1, len(rsi_symbols)))
        progress.empty()
        st.session_state[f"rsi_oversold_results_{asset_class}"] = {
            "signature": rsi_signature, "rows": found, "errors": failures,
        }

    saved_rsi = st.session_state.get(f"rsi_oversold_results_{asset_class}")
    if saved_rsi and saved_rsi["signature"] == rsi_signature:
        rsi_results = pd.DataFrame(saved_rsi["rows"])
        if rsi_results.empty:
            st.info("No symbols currently have RSI (14) at or below 30 for this timeframe.")
        else:
            rising = rsi_results[rsi_results["Price Trend"] == "Upward ↑"].sort_values(
                ["RSI (14)", "Symbol"]
            )
            falling = rsi_results[rsi_results["Price Trend"] == "Downward ↓"].sort_values(
                ["RSI (14)", "Symbol"]
            )
            flat = rsi_results[rsi_results["Price Trend"] == "Flat →"].sort_values(
                ["RSI (14)", "Symbol"]
            )
            summary_down, summary_up = st.columns(2)
            summary_down.metric("RSI ≤ 30 · Downtrend (tested)", len(falling))
            summary_up.metric("RSI ≤ 30 · Uptrend", len(rising))
            st.markdown("#### RSI ≤ 30 in a downtrend — the tested buy signal")
            if falling.empty:
                st.info("No oversold symbols in a downtrend right now.")
            else:
                st.dataframe(falling, use_container_width=True, hide_index=True)
            st.markdown("#### RSI ≤ 30 in an uptrend — no reliable edge")
            st.caption("Stocks did about the same as any stock (54% rose within 20 days); Forex looked better "
                       "(61%) but on too few signals (115) to rely on.")
            if rising.empty:
                st.info("No oversold symbols in an uptrend.")
            else:
                st.dataframe(rising, use_container_width=True, hide_index=True)
            if not flat.empty:
                with st.expander(f"Flat price trends ({len(flat)})"):
                    st.dataframe(flat, use_container_width=True, hide_index=True)
        if saved_rsi["errors"]:
            with st.expander(f"Symbols that could not be scanned ({len(saved_rsi['errors'])})"):
                st.write(saved_rsi["errors"])
    else:
        st.info("Click **Scan RSI ≤ 30** to view results.")


# =========================
# Confirmed trade-action scanner — uses the same engine as chart and existing scanner.
# =========================


with tab_fx_stoch:
    st.subheader("Forex Stochastic Dip (daily)")
    st.markdown(
        f"""
A tested Forex buy rule (not a forecast): **buy** a pair when its daily Stochastic %K(14) is
**{FX_STOCH_LOW} or lower**, **sell** when daily RSI(14) reaches **{FX_STOCH_EXIT_RSI}**, or after
**{FX_STOCH_MAX_DAYS} days**. Currency pairs tend to bounce back after sharp falls, which is why buying
dips has worked better than selling weakness.

**Past record:** {fx_stochastic_history_text()}
"""
    )
    if st.button("Scan Forex pairs", use_container_width=True, key="fx_stoch_scan"):
        stoch_rows = []
        stoch_progress = st.progress(0)
        for n, sym in enumerate(ASSET_UNIVERSES["Forex"]):
            try:
                reading = fx_stochastic_reading(sym)
                if reading is not None:
                    stoch_rows.append(reading)
            except Exception:
                pass
            stoch_progress.progress((n + 1) / len(ASSET_UNIVERSES["Forex"]))
        stoch_progress.empty()
        if stoch_rows:
            stoch_df = pd.DataFrame(stoch_rows).sort_values("Stochastic %K")
            hits = stoch_df[stoch_df["Dip Signal"]]
            if hits.empty:
                st.info(f"No pair is at daily Stochastic {FX_STOCH_LOW} or lower right now.")
            else:
                st.success("Pairs at the buy level: " + ", ".join(hits["Symbol"]))
            st.dataframe(stoch_df, use_container_width=True, hide_index=True)
    st.caption("Not financial advice. Past results do not guarantee future results.")


with tab_rules:
    st.subheader("What this app shows, and why")
    st.markdown(
        f"""
Everything here except the Indicator Chart held up when tested on past data. Signals that did no better than chance were removed.

**Range Chart**
- **Likely range (blue cone)**: where the price is likely to be over the next {RANGE_DAYS} trading days,
  with 80% and 50% bands. It is built from how far this market's symbols have moved in the past, scaled by the
  symbol's last {RANGE_VOL_DAYS} days of volatility. Checked year by year (2018–2026), each year using only
  earlier years: the real close landed inside the 80% band 77–84% of the time. It does **not** say which way
  the price will go: no direction forecast tested better than chance.
- **Dip buy signal (lower panel)**: Forex pairs when daily Stochastic %K(14) is {FX_STOCH_LOW} or lower, sold when RSI
  reaches {FX_STOCH_EXIT_RSI} (at most {FX_STOCH_MAX_DAYS} days); stocks when daily RSI falls to {STOCK_DIP_RSI} or
  below with the 120-day trend down, held {STOCK_DIP_HOLD} trading days. Every past signal on the chart is marked
  green (rose) or red (fell), with the symbol's own record below the chart.
- **Your position**: optional. Shows profit or loss, carry (enter the swap your broker shows) and the profit or
  loss range implied by the likely range. Warns when the dip signal works against you.
- **Intraday**: for timing only. Today's range against a typical day's range, and the Tokyo, London and
  New York sessions. No buy or sell labels.

**Indicator Chart**: the TradingView layout (SuperTrend 10 3, Bollinger 20 2, EMA 9, SMA 35, RSI 14,
ROC 9, MACD 12 26 9), shown because you use it, not because it tested well. {IND_RECORD}

**RSI ≤ 30**: daily oversold scan. Only the downtrend list had an edge.

**FX Stochastic Dip**: scans all pairs for the Forex dip buy signal. Past record: {fx_stochastic_history_text()}

**What was removed, and why.** A 5-year daily backtest of every tab (Nov 2021–Nov 2026) and a replay of every
BUY/SELL call since 2017 found that these did no better than buying or selling at random, either as given or
reversed: the Trading Chart's BUY/SELL states and scores, Buy/Sell, Trade Action, Scanner, Green Zone, NPX Cross,
30EMA Cross, S/R Cross, NTD Cross, HMA Cross and the NTD Dips lists. The TradingView-style chart's SuperTrend and
EMA/SMA crosses lost money (hourly −23% to −37% over 2 years; daily about −5%). The functions are still in the
code, so the history can be rebuilt with `bullbears_history_build.py`.
"""
    )
    st.info("Size each trade so that a bad outcome costs a small, fixed part of the account. "
            "Not financial advice. Past results do not guarantee future results.")
