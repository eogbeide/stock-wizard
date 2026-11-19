"""Tested-only charts shared by the Streamlit apps (bullbear.py).

Copied from bullbears.py, where these were built and tested: the Range Chart
(5-day likely range checked year by year on the apps' own symbols, the tested
dip buy signals with each symbol's own past record, an optional position panel
and an intraday view for timing), plus the daily RSI <= 30 and Forex
Stochastic dip scans. Nothing here gives a BUY/SELL call that tested no better
than chance.
"""
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

# =========================
# App / Styling
# =========================
PACIFIC = pytz.timezone("US/Pacific")

# Forex dip rule tested in Stock Wizard: buy when daily Stochastic %K(14) <= 10,
# sell when RSI(14) reaches 50 or after 60 days.
FX_STOCH_LOW = 10

FX_STOCH_EXIT_RSI = 50

FX_STOCH_MAX_DAYS = 60

HISTORY_FILE = Path(__file__).with_name("bullbears_history.csv")

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


def safe_float(value, default=np.nan) -> float:
    try:
        out = float(np.squeeze(value))
    except Exception:
        return default
    return out if np.isfinite(out) else default


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


def bollinger(series: pd.Series, window: int = 20, mult: float = 2.0):
    s = coerce_series(series)
    mid = s.rolling(window, min_periods=max(2, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(2, window // 3)).std()
    return mid, mid + mult * sd, mid - mult * sd


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic %K: where the close sits in the last `period` bars' range (0-100)."""
    close = coerce_series(df["Close"])
    lo = coerce_series(df["Low"]).rolling(period, min_periods=period).min()
    hi = coerce_series(df["High"]).rolling(period, min_periods=period).max()
    return 100.0 * (close - lo) / (hi - lo).replace(0, np.nan)


@st.cache_data(ttl=3600, show_spinner=False)
def load_signal_history() -> pd.DataFrame:
    """Measured past results for each signal (see bullbears_history_build.py)."""
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()


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


def rsi_scan_row(symbol: str) -> dict | None:
    """Daily RSI(14) at or below STOCK_DIP_RSI, with the 120-day trend line's direction."""
    daily = weekday_bars(fetch_market_ohlc(symbol, "1y", "1d"))
    if daily.empty or len(daily) < 130:
        return None
    rsi = calculate_chart_rsi(daily["Close"], 14)
    r = safe_float(rsi.iloc[-1])
    if not np.isfinite(r) or r > STOCK_DIP_RSI:
        return None
    slope = safe_float(trend_slope_series(daily["Close"], 120).iloc[-1])
    return {"Symbol": symbol, "RSI (14)": round(r, 1), "Trend (120 days)": "Down" if slope < 0 else "Up",
            "Last Close": fmt_price(symbol, daily["Close"].iloc[-1]),
            "Last Bar": f"{bar_days(daily.index)[-1]:%Y-%m-%d}"}


def render_rsi_scan(universe: list[str], market: str) -> None:
    st.caption(
        "Daily RSI (14) at or below 30, split by the 120-day trend line. Only the downtrend list was tested "
        "with an edge: " + (STOCK_DIP_RECORD if market == "Stocks" else FX_RSI_DOWN_RECORD)
    )
    if not st.button("Scan RSI ≤ 30", use_container_width=True, key=f"tc_rsi_scan_{market}"):
        st.info("Click **Scan RSI ≤ 30** to view results.")
        return
    rows, bar = [], st.progress(0)
    for n, sym in enumerate(universe):
        try:
            row = rsi_scan_row(sym)
            if row:
                rows.append(row)
        except Exception:
            pass
        bar.progress((n + 1) / len(universe))
    bar.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Nothing is at daily RSI 30 or below right now.")
        return
    st.markdown("#### RSI ≤ 30 in a downtrend — the tested buy signal")
    down = df[df["Trend (120 days)"] == "Down"].sort_values("RSI (14)")
    st.dataframe(down, use_container_width=True, hide_index=True) if not down.empty else st.info("None right now.")
    st.markdown("#### RSI ≤ 30 in an uptrend — no reliable edge")
    up = df[df["Trend (120 days)"] == "Up"].sort_values("RSI (14)")
    st.dataframe(up, use_container_width=True, hide_index=True) if not up.empty else st.info("None right now.")


def render_fx_stochastic_scan(universe: list[str]) -> None:
    st.markdown(
        f"A tested Forex buy rule: **buy** a pair when its daily Stochastic %K(14) is **{FX_STOCH_LOW} or lower**, "
        f"**sell** when daily RSI(14) reaches **{FX_STOCH_EXIT_RSI}**, or after **{FX_STOCH_MAX_DAYS} days**.\n\n"
        f"**Past record:** {fx_stochastic_history_text()}"
    )
    if not st.button("Scan Forex pairs", use_container_width=True, key="tc_fx_stoch_scan"):
        return
    rows = []
    for sym in universe:
        try:
            r = fx_stochastic_reading(sym)
            if r:
                rows.append(r)
        except Exception:
            pass
    if not rows:
        st.info("No data right now.")
        return
    df = pd.DataFrame(rows).sort_values("Stochastic %K")
    hits = df[df["Dip Signal"]]
    if hits.empty:
        st.info(f"No pair is at daily Stochastic {FX_STOCH_LOW} or lower right now.")
    else:
        st.success("Pairs at the buy level: " + ", ".join(hits["Symbol"]))
    st.dataframe(df, use_container_width=True, hide_index=True)
