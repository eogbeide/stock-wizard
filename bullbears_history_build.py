"""Builds bullbears_history.csv: the measured past record of each bullbears.py signal.

Runs the app's own classify_trade on every past bar (daily since 2017, 1-hour bars for the
last two years) for every Forex pair and stock in the app, then follows the trade plan:
target and stop both PLAN_ATR x ATR from the close, closed after MAX_BARS bars if neither
is reached (a bar touching both counts as the stop). Costs: 0.03% Forex, 0.05% Stocks.
Prices are split-adjusted so stock splits do not look like price moves.
Also records the Forex Stochastic Dip rule and an "ANY LONG"/"ANY SHORT" baseline (every bar).

Run from the repo folder:  python bullbears_history_build.py
Takes about 10-20 minutes. Rerun every few months to refresh the record.
"""
import ast
import sys
import types
from multiprocessing import Pool
from unittest import mock

import numpy as np
import pandas as pd
import yfinance as yf

START = "2017-06-01"
MAX_BARS = 20
COST = {"Forex": 0.0003, "Stocks": 0.0005}
TAIL = 400  # bars passed to classify_trade (indicators are computed on full history)
DEFAULT_CFG = dict(trend_lookback=120, sr_lookback=120, ntd_window=60, npx_smooth=9, sr_smooth=8,
                   touch_lookback=8, confirm_bars=3, signal_lookback=40, cross_lookback=30,
                   touch_tol_pips=3.0, pullback_tol_pips=8.0, cooldown=10, right_padding=12)


def load_app():
    """Load bullbears.py's functions and constants without running the Streamlit page."""
    src = open("bullbears.py", encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets))]
    fake_st = mock.MagicMock()
    fake_st.cache_data = lambda *a, **k: (lambda f: f)
    sys.modules["streamlit"] = fake_st
    ns = {"__name__": "bullbears", "__file__": "bullbears.py"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "bullbears.py", "exec"), ns)
    ns["signal_history"] = lambda *a, **k: None  # history is what we're building

    def fast_wma(series, length):
        # Same result as the app's wma, vectorised for speed.
        s = ns["coerce_series"](series).astype(float)
        n = int(max(1, length))
        w = np.arange(1, n + 1, dtype=float)
        v = s.to_numpy()
        out = np.full(len(v), np.nan)
        if len(v) >= n:
            win = np.lib.stride_tricks.sliding_window_view(v, n)
            ok = np.isfinite(win).all(axis=1)
            out[n - 1:][ok] = win[ok] @ w / w.sum()
        return pd.Series(out, index=s.index)

    ns["wma"] = fast_wma
    return ns


APP = None


def fetch(symbol, timeframe):
    if timeframe == "Daily":
        df = yf.download(symbol, start="2016-01-01", interval="1d", progress=False, auto_adjust=True)
    else:
        df = yf.download(symbol, period="730d", interval="1h", progress=False, auto_adjust=True)
    df = APP["flatten_yf_columns"](df, ticker=symbol)
    need = ["Open", "High", "Low", "Close"]
    if df.empty or any(c not in df.columns for c in need):
        return pd.DataFrame()
    df = df[need].apply(pd.to_numeric, errors="coerce").dropna()
    df["Volume"] = 0.0
    return df[~df.index.duplicated(keep="last")].sort_index()


def run_symbol(job):
    global APP
    if APP is None:
        APP = load_app()
    symbol, timeframe = job
    market = "Forex" if "=X" in symbol else "Stocks"
    cfg = dict(DEFAULT_CFG, interval="1d" if timeframe == "Daily" else "60m", period="", asset_class=market)
    try:
        raw = fetch(symbol, timeframe)
    except Exception:
        return []
    if len(raw) < 300:
        return []
    df = APP["prepare_indicators"](raw, cfg, symbol)
    close = df["Close"].to_numpy(float)
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    atr = df["ATR14"].to_numpy(float)
    lb = cfg["trend_lookback"]
    x = np.arange(lb, dtype=float)
    first = max(TAIL, lb)
    if timeframe == "Daily":
        first = max(first, int(np.searchsorted(df.index, pd.Timestamp(START))))
    rows = []
    cost = COST[market]
    for i in range(first, len(df) - 1):
        slope = float(np.polyfit(x, close[i - lb + 1:i + 1], 1)[0])
        trade = APP["classify_trade"](symbol, df.iloc[i - TAIL + 1:i + 1], slope, cfg)
        state = trade["State"]
        end = min(len(df) - 1, i + MAX_BARS)
        if end - i < MAX_BARS:
            continue
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else close[i] * 0.002
        for side in ("Long", "Short"):
            tag = f"ANY {side.upper()}"
            sign = 1 if side == "Long" else -1
            tgt = close[i] + sign * APP["PLAN_ATR"] * a
            stp = close[i] - sign * APP["PLAN_ATR"] * a
            out, res = "time", sign * (close[end] / close[i] - 1)
            for j in range(i + 1, end + 1):
                hit_stop = low[j] <= stp if sign > 0 else high[j] >= stp
                hit_tgt = high[j] >= tgt if sign > 0 else low[j] <= tgt
                if hit_stop:
                    out, res = "stop", sign * (stp / close[i] - 1)
                    break
                if hit_tgt:
                    out, res = "target", sign * (tgt / close[i] - 1)
                    break
            rec = dict(Market=market, Timeframe="Daily" if timeframe == "Daily" else "Intraday",
                       out=out, ret=res - cost, bars=j - i if out != "time" else MAX_BARS, date=df.index[i])
            rows.append(dict(rec, State=tag))
            if state != "WAIT" and (("BUY" in state) == (side == "Long")):
                rows.append(dict(rec, State=state))
    if market == "Forex" and timeframe == "Daily":
        k = APP["stochastic_k"](df, 14).to_numpy(float)
        rsi = APP["calculate_chart_rsi"](df["Close"], 14).to_numpy(float)
        i = first
        while i < len(df) - 1:
            if np.isfinite(k[i]) and k[i] <= APP["FX_STOCH_LOW"]:
                j = i + 1
                while j < len(df) - 1 and j - i < APP["FX_STOCH_MAX_DAYS"] and not (rsi[j] >= APP["FX_STOCH_EXIT_RSI"]):
                    j += 1
                if j - i >= APP["FX_STOCH_MAX_DAYS"] or rsi[j] >= APP["FX_STOCH_EXIT_RSI"]:
                    r = close[j] / close[i] - 1 - cost
                    rows.append(dict(Market=market, Timeframe="Daily", State="FX STOCH DIP",
                                     out="target" if r > 0 else "stop", ret=r, bars=j - i, date=df.index[i]))
                i = j + 1  # one position per pair at a time
            else:
                i += 1
    return rows


def main():
    global APP
    APP = load_app()
    jobs = [(s, tf) for tf in ("Daily", "Intraday")
            for s in APP["FOREX_UNIVERSE"] + APP["STOCK_UNIVERSE"]]
    rows = []
    with Pool() as pool:
        for n, part in enumerate(pool.imap_unordered(run_symbol, jobs), 1):
            rows.extend(part)
            print(f"{n}/{len(jobs)}", end="\r", flush=True)
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    g = d.groupby(["Market", "Timeframe", "State"])
    out = pd.DataFrame({
        "Signals": g.size(),
        "Target First %": g["out"].apply(lambda s: (s == "target").mean() * 100).round(1),
        "Stop First %": g["out"].apply(lambda s: (s == "stop").mean() * 100).round(1),
        "Avg Result %": g["ret"].mean().mul(100).round(3),
        "Median Bars": g["bars"].median(),
        "Since": g["date"].min().dt.strftime("%b %Y"),
    }).reset_index()
    out.to_csv("bullbears_history.csv", index=False)
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
