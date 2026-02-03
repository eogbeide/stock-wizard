# bullbear.py
# Streamlit app: Stock Wizard (Bull/Bear) — includes NPX/NTD scanners + Yahoo Finance news
# FIX for NameError: define fetch_yf_news() before it is used.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Stock Wizard | Bull/Bear",
    layout="wide",
)

st.title("Stock Wizard — Bull/Bear Dashboard")


# ----------------------------
# Utility: robust datetime handling
# ----------------------------
def _to_utc_dt(ts_like) -> Optional[datetime]:
    if ts_like is None or (isinstance(ts_like, float) and math.isnan(ts_like)):
        return None
    if isinstance(ts_like, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_like), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(ts_like, pd.Timestamp):
        if ts_like.tzinfo is None:
            return ts_like.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts_like.to_pydatetime().astimezone(timezone.utc)
    if isinstance(ts_like, datetime):
        if ts_like.tzinfo is None:
            return ts_like.replace(tzinfo=timezone.utc)
        return ts_like.astimezone(timezone.utc)
    try:
        # string
        return pd.to_datetime(ts_like, utc=True).to_pydatetime()
    except Exception:
        return None


# ----------------------------
# FIX: Yahoo Finance news fetcher
# ----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_yf_news(ticker: str, window_days: int = 7) -> pd.DataFrame:
    """
    Fetches Yahoo Finance news for a ticker via yfinance.Ticker(...).news
    and filters to articles within window_days.

    Returns columns:
      - published_utc
      - title
      - publisher
      - link
      - relatedTickers (if provided)
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.DataFrame(columns=["published_utc", "title", "publisher", "link", "relatedTickers"])

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(window_days))
    rows: List[Dict] = []

    try:
        t = yf.Ticker(ticker)
        items = t.news or []
    except Exception:
        items = []

    for it in items:
        # yfinance typically returns providerPublishTime (epoch seconds)
        pub = _to_utc_dt(it.get("providerPublishTime"))
        if pub is None:
            # sometimes publishTime or something else exists
            pub = _to_utc_dt(it.get("publishTime"))
        if pub is None:
            continue
        if pub < cutoff:
            continue

        rows.append(
            {
                "published_utc": pub,
                "title": it.get("title", ""),
                "publisher": it.get("publisher", it.get("provider", "")),
                "link": it.get("link", it.get("url", "")),
                "relatedTickers": ", ".join(it.get("relatedTickers", [])) if isinstance(it.get("relatedTickers"), list) else it.get("relatedTickers", ""),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["published_utc", "title", "publisher", "link", "relatedTickers"])
    df = df.sort_values("published_utc", ascending=False).reset_index(drop=True)
    return df


# ----------------------------
# Data fetching
# ----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch OHLCV history using yfinance.
    interval examples: '1d', '60m'
    period examples: '1y', '6mo', '1mo', '7d', '1d'
    """
    symbol = symbol.strip().upper()
    try:
        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Standardize columns (yfinance sometimes returns multiindex columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # Ensure expected cols exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


# ----------------------------
# Metrics: NPX / NTD
# ----------------------------
def compute_npx(close: pd.Series, win: int) -> float:
    """
    NPX: normalized price position in [0,1] over last `win` closes:
        (last - min) / (max - min)
    """
    if close is None or close.empty:
        return float("nan")
    s = close.dropna()
    if s.empty:
        return float("nan")
    w = int(win)
    if w <= 1:
        return float("nan")
    if len(s) < w:
        w = len(s)
    chunk = s.iloc[-w:]
    lo = float(chunk.min())
    hi = float(chunk.max())
    last = float(chunk.iloc[-1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not np.isfinite(last):
        return float("nan")
    if hi - lo == 0:
        return 0.5
    return (last - lo) / (hi - lo)


def compute_ntd(close: pd.Series, win: int) -> float:
    """
    NTD: trend direction score in [-1, 1], based on slope of log-price
    over last `win` points, normalized by volatility, then squashed via tanh.
    """
    if close is None or close.empty:
        return float("nan")
    s = close.dropna()
    if s.empty:
        return float("nan")
    w = int(win)
    if w <= 2:
        return float("nan")
    if len(s) < w:
        w = len(s)

    y = np.log(s.iloc[-w:].astype(float).values)
    if np.any(~np.isfinite(y)):
        return float("nan")

    x = np.arange(w, dtype=float)
    # slope via polyfit
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return float("nan")

    # normalize by residual std (vol proxy)
    yhat = slope * x + intercept
    resid = y - yhat
    vol = float(np.std(resid))  # residual volatility
    if not np.isfinite(vol) or vol == 0:
        # fallback: std of returns
        rets = np.diff(y)
        vol = float(np.std(rets)) if len(rets) else 0.0

    if not np.isfinite(vol) or vol == 0:
        # if everything is flat, trend is 0
        return 0.0

    score = slope / vol
    # squash to [-1, 1]
    return float(np.tanh(score))


def last_daily_npx_value(symbol: str, win: int) -> Tuple[float, Optional[pd.Timestamp]]:
    df = fetch_history(symbol, interval="1d", period="1y")
    if df.empty:
        return float("nan"), None
    return compute_npx(df["Close"], win), df.index[-1]


def last_daily_ntd_value(symbol: str, win: int) -> Tuple[float, Optional[pd.Timestamp]]:
    df = fetch_history(symbol, interval="1d", period="1y")
    if df.empty:
        return float("nan"), None
    return compute_ntd(df["Close"], win), df.index[-1]


def last_hourly_npx_value(symbol: str, win: int, period: str = "7d") -> Tuple[float, Optional[pd.Timestamp]]:
    df = fetch_history(symbol, interval="60m", period=period)
    if df.empty:
        return float("nan"), None
    return compute_npx(df["Close"], win), df.index[-1]


def last_hourly_ntd_value(symbol: str, win: int, period: str = "7d") -> Tuple[float, Optional[pd.Timestamp]]:
    df = fetch_history(symbol, interval="60m", period=period)
    if df.empty:
        return float("nan"), None
    return compute_ntd(df["Close"], win), df.index[-1]


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _latest_ntd_npx(sym: str, frame_label: str, ntd_win: int) -> Tuple[float, float, Optional[pd.Timestamp]]:
    """Returns (ntd, npx, timestamp) for the requested frame."""
    if frame_label.startswith("Hourly"):
        period = "7d"
        ntd, ts = last_hourly_ntd_value(sym, ntd_win, period=period)
        npx, _ = last_hourly_npx_value(sym, ntd_win, period=period)
        return ntd, npx, ts
    else:
        ntd, ts = last_daily_ntd_value(sym, ntd_win)
        npx, _ = last_daily_npx_value(sym, ntd_win)
        return ntd, npx, ts


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.subheader("Controls")

    mode = st.radio("Mode", ["Bull", "Bear"], index=0)
    disp_ticker = st.text_input("Display ticker", value="SPY").strip().upper()

    default_universe = "SPY, QQQ, DIA, IWM, XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLRE, XLU, TLT, GLD"
    universe_txt = st.text_area("Universe (comma-separated tickers)", value=default_universe, height=120)
    universe = [t.strip().upper() for t in universe_txt.split(",") if t.strip()]

    ntd_window = st.number_input("NTD/NPX window (bars)", min_value=5, max_value=400, value=60, step=5)
    news_window_days = st.number_input("News window (days)", min_value=1, max_value=60, value=7, step=1)

    st.caption("Tip: If hourly data looks sparse, increase period in code (fetch_history).")


# ----------------------------
# Tabs
# ----------------------------
tab_labels = [
    "1) Chart",
    "2) NTD & NPX Series",
    "3) Snapshot",
    "4) News",
    "5) Daily Movers",
    "6) Correlation",
    "7) Drawdown",
    "8) Signals",
    "9) NTD Buy Signal",
    "10) NTD Sell Signal",
    "11) NPX Low Scanner",
    "12) NPX High Scanner",
    "13) NTD Range Filter",
    "14) Universe Snapshot Table",
    "15) Watchlist Metrics",
    "16) Export Symbol Lists",
    "17) Quick Stats",
    "18) Cache / Troubleshooting",
    "19) Help / Notes",
]
(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
    tab10,
    tab11,
    tab12,
    tab13,
    tab14,
    tab15,
    tab16,
    tab17,
    tab18,
    tab19,
) = st.tabs(tab_labels)


# ---------------------------
# TAB 1: Chart
# ---------------------------
with tab1:
    st.header("Price Chart")

    c1, c2, c3 = st.columns(3)
    frame = c1.selectbox("Frame", ["Daily", "Hourly (intraday)"], index=0)
    if frame == "Daily":
        interval = "1d"
        period = c2.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    else:
        interval = "60m"
        period = c2.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo"], index=2)

    show_volume = c3.checkbox("Show Volume", value=True)

    df = fetch_history(disp_ticker, interval=interval, period=period)
    if df.empty:
        st.error("No data returned. Check ticker or try another period.")
    else:
        st.caption(f"{disp_ticker} — interval={interval}, period={period} (rows: {len(df)})")

        if go is None:
            st.line_chart(df["Close"])
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                )
            )
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        if show_volume:
            st.bar_chart(df["Volume"])


# ---------------------------
# TAB 2: NTD & NPX Series
# ---------------------------
with tab2:
    st.header("NTD & NPX Series")

    c1, c2 = st.columns(2)
    series_frame = c1.selectbox("Frame", ["Daily", "Hourly (intraday)"], index=0)
    win = c2.number_input("Rolling window (bars)", min_value=5, max_value=400, value=int(ntd_window), step=5)

    if series_frame == "Daily":
        df = fetch_history(disp_ticker, interval="1d", period="1y")
    else:
        df = fetch_history(disp_ticker, interval="60m", period="7d")

    if df.empty:
        st.error("No data returned.")
    else:
        close = df["Close"].dropna()
        if close.empty:
            st.error("Close series empty.")
        else:
            # rolling NPX/NTD (simple rolling apply, not super fast but OK for typical sizes)
            w = int(win)
            if w > len(close):
                w = len(close)

            def _roll_npx(x):
                return compute_npx(pd.Series(x), len(x))

            def _roll_ntd(x):
                return compute_ntd(pd.Series(x), len(x))

            npx_s = close.rolling(w).apply(_roll_npx, raw=True)
            ntd_s = close.rolling(w).apply(_roll_ntd, raw=True)

            out = pd.DataFrame({"Close": close, "NPX": npx_s, "NTD": ntd_s}).dropna()

            cA, cB, cC = st.columns(3)
            cA.metric("Latest Close", f"{out['Close'].iloc[-1]:.2f}")
            cB.metric("Latest NPX", f"{out['NPX'].iloc[-1]:.3f}")
            cC.metric("Latest NTD", f"{out['NTD'].iloc[-1]:.3f}")

            st.line_chart(out[["NPX", "NTD"]])


# ---------------------------
# TAB 3: Snapshot
# ---------------------------
with tab3:
    st.header("Quick Snapshot (selected ticker)")
    c1, c2 = st.columns(2)
    ntd_d, ts_d = last_daily_ntd_value(disp_ticker, int(ntd_window))
    npx_d, _ = last_daily_npx_value(disp_ticker, int(ntd_window))

    ntd_h, ts_h = last_hourly_ntd_value(disp_ticker, int(ntd_window), period="7d")
    npx_h, _ = last_hourly_npx_value(disp_ticker, int(ntd_window), period="7d")

    c1.subheader("Daily")
    c1.write({"NTD": ntd_d, "NPX": npx_d, "Time": ts_d})

    c2.subheader("Hourly (60m)")
    c2.write({"NTD": ntd_h, "NPX": npx_h, "Time": ts_h})


# ---------------------------
# TAB 4: News  (this is where your NameError was coming from)
# ---------------------------
with tab4:
    st.header("News (Yahoo Finance)")

    st.caption(
        "Uses yfinance.Ticker(ticker).news and filters by the selected window. "
        "If nothing appears, Yahoo may not provide news for that ticker."
    )

    # This line is the one from your traceback; it now works because fetch_yf_news is defined above.
    fx_news = fetch_yf_news(disp_ticker, window_days=int(news_window_days))

    if fx_news.empty:
        st.info("No news found in this window.")
    else:
        for i, row in fx_news.iterrows():
            title = row.get("title", "")
            publisher = row.get("publisher", "")
            link = row.get("link", "")
            pub = row.get("published_utc", None)
            pub_str = pub.strftime("%Y-%m-%d %H:%M UTC") if isinstance(pub, datetime) else str(pub)

            with st.expander(f"{pub_str} — {title}", expanded=(i == 0)):
                if publisher:
                    st.write(f"**Publisher:** {publisher}")
                if row.get("relatedTickers"):
                    st.write(f"**Related:** {row.get('relatedTickers')}")
                if link:
                    st.markdown(f"[Open article]({link})")


# ---------------------------
# TAB 5: Daily Movers (simple)
# ---------------------------
with tab5:
    st.header("Daily Movers (Universe)")
    st.caption("Computes 1-day % change for tickers in the universe (best effort).")

    run = st.button("Compute Movers")
    if run:
        rows = []
        for sym in universe:
            df = fetch_history(sym, interval="1d", period="10d")
            if df.empty or df["Close"].dropna().shape[0] < 2:
                continue
            c = df["Close"].dropna()
            pct = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1.0) * 100.0
            rows.append({"Symbol": sym, "Close": float(c.iloc[-1]), "1D %": pct})
        if not rows:
            st.info("No movers computed.")
        else:
            out = pd.DataFrame(rows).sort_values("1D %", ascending=False)
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 6: Correlation (daily closes)
# ---------------------------
with tab6:
    st.header("Correlation (Daily Returns)")
    st.caption("Builds a correlation matrix of daily returns for the current universe (may take time).")

    run = st.button("Build Correlation Matrix")
    if run:
        closes = {}
        for sym in universe:
            df = fetch_history(sym, interval="1d", period="6mo")
            if df.empty:
                continue
            s = df["Close"].dropna()
            if len(s) < 30:
                continue
            closes[sym] = s
        if len(closes) < 2:
            st.info("Not enough symbols with data to compute correlation.")
        else:
            wide = pd.DataFrame(closes).dropna()
            rets = wide.pct_change().dropna()
            corr = rets.corr()
            st.dataframe(corr, use_container_width=True)


# ---------------------------
# TAB 7: Drawdown (selected ticker)
# ---------------------------
with tab7:
    st.header("Drawdown (Selected Ticker)")

    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
    df = fetch_history(disp_ticker, interval="1d", period=period)
    if df.empty:
        st.error("No data returned.")
    else:
        close = df["Close"].dropna()
        peak = close.cummax()
        dd = (close / peak) - 1.0
        st.metric("Max drawdown", f"{dd.min() * 100:.2f}%")
        st.line_chart(dd)


# ---------------------------
# TAB 8: Signals (simple)
# ---------------------------
with tab8:
    st.header("Signals (Simple Heuristics)")
    st.caption("Example rules: NTD extreme + NPX extreme.")

    frame = st.radio("Frame", ["Hourly (intraday)", "Daily"], index=0)
    ntd_buy = st.number_input("Buy if NTD <", value=-0.75, step=0.05, format="%.2f")
    npx_buy = st.slider("...and NPX <=", 0.0, 1.0, 0.25, 0.01)
    ntd_sell = st.number_input("Sell if NTD >", value=0.75, step=0.05, format="%.2f")
    npx_sell = st.slider("...and NPX >=", 0.0, 1.0, 0.75, 0.01)

    if st.button("Evaluate"):
        ntd, npx, ts = _latest_ntd_npx(disp_ticker, frame, int(ntd_window))
        st.write({"ticker": disp_ticker, "frame": frame, "time": ts, "NTD": ntd, "NPX": npx})

        buy = np.isfinite(ntd) and np.isfinite(npx) and (ntd < float(ntd_buy)) and (npx <= float(npx_buy))
        sell = np.isfinite(ntd) and np.isfinite(npx) and (ntd > float(ntd_sell)) and (npx >= float(npx_sell))

        if buy and sell:
            st.warning("Both buy and sell triggered (check thresholds).")
        elif buy:
            st.success("BUY signal triggered.")
        elif sell:
            st.error("SELL signal triggered.")
        else:
            st.info("No signal triggered.")


# ============================================================
# BATCH 3 (Tabs 9–19) — includes NTD Buy Signal (NPX ascending)
# ============================================================

# ---------------------------
# TAB 9: NTD BUY SIGNAL (sorted by NPX ascending)
# ---------------------------
with tab9:
    st.header("NTD Buy Signal — Oversold Candidates (sorted by NPX ↑)")
    st.caption(
        "Scans for symbols where the latest **NTD** is below a threshold (default -0.75). "
        "Results are sorted by **NPX ascending** (lower NPX = closer to the bottom of its window)."
    )

    frame9 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=0,
        key=f"ntd_buy_frame_{mode}",
    )

    c1, c2, c3 = st.columns(3)
    thr_buy = c1.number_input(
        "NTD threshold (buy if NTD < threshold)",
        value=-0.75,
        step=0.05,
        format="%.2f",
        key=f"ntd_buy_thr_{mode}",
    )
    npx_max_buy = c2.slider(
        "Max NPX (optional filter)",
        0.0,
        1.0,
        1.00,
        0.01,
        key=f"ntd_buy_npx_max_{mode}",
    )
    require_npx = c3.checkbox(
        "Require NPX <= Max NPX",
        value=False,
        key=f"ntd_buy_req_npx_{mode}",
    )

    run_buy = st.button("Run NTD Buy Signal Scan", key=f"btn_run_ntd_buy_{mode}")

    if run_buy:
        rows = []
        for sym in universe:
            ntd, npx, ts = _latest_ntd_npx(sym, frame9, int(ntd_window))
            if not np.isfinite(ntd):
                continue
            if ntd < float(thr_buy):
                if require_npx and (not np.isfinite(npx) or float(npx) > float(npx_max_buy)):
                    continue
                rows.append(
                    {
                        "Symbol": sym,
                        "NTD": float(ntd),
                        "NPX (Norm Price)": float(npx) if np.isfinite(npx) else np.nan,
                        "Time": ts,
                        "Frame": "Hourly" if frame9.startswith("Hourly") else "Daily",
                    }
                )

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            out = out.sort_values(
                ["NPX (Norm Price)", "NTD", "Symbol"],
                ascending=[True, True, True],
                na_position="last",
            )
            st.dataframe(out.reset_index(drop=True), use_container_width=True)
            st.download_button(
                "Download CSV",
                data=_df_to_csv_bytes(out),
                file_name=f"ntd_buy_signal_{'hourly' if frame9.startswith('Hourly') else 'daily'}_{mode}.csv",
                mime="text/csv",
                key=f"dl_ntd_buy_{mode}",
            )


# ---------------------------
# TAB 10: NTD SELL SIGNAL (simple)
# ---------------------------
with tab10:
    st.header("NTD Sell Signal — Overbought Candidates")
    st.caption("Scans for symbols where the latest NTD is above a threshold (default +0.75).")

    frame10 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=0,
        key=f"ntd_sell_frame_{mode}",
    )

    c1, c2 = st.columns(2)
    thr_sell = c1.number_input(
        "NTD threshold (sell if NTD > threshold)",
        value=0.75,
        step=0.05,
        format="%.2f",
        key=f"ntd_sell_thr_{mode}",
    )
    sort10 = c2.selectbox(
        "Sort by:",
        ["NTD (desc)", "NPX (desc)", "NPX (asc)"],
        index=0,
        key=f"ntd_sell_sort_{mode}",
    )

    run_sell = st.button("Run NTD Sell Signal Scan", key=f"btn_run_ntd_sell_{mode}")

    if run_sell:
        rows = []
        for sym in universe:
            ntd, npx, ts = _latest_ntd_npx(sym, frame10, int(ntd_window))
            if np.isfinite(ntd) and ntd > float(thr_sell):
                rows.append(
                    {
                        "Symbol": sym,
                        "NTD": float(ntd),
                        "NPX (Norm Price)": float(npx) if np.isfinite(npx) else np.nan,
                        "Time": ts,
                        "Frame": "Hourly" if frame10.startswith("Hourly") else "Daily",
                    }
                )

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            if sort10 == "NTD (desc)":
                out = out.sort_values(["NTD", "NPX (Norm Price)", "Symbol"], ascending=[False, False, True])
            elif sort10 == "NPX (desc)":
                out = out.sort_values(["NPX (Norm Price)", "NTD", "Symbol"], ascending=[False, False, True])
            else:
                out = out.sort_values(["NPX (Norm Price)", "NTD", "Symbol"], ascending=[True, False, True])

            st.dataframe(out.reset_index(drop=True), use_container_width=True)
            st.download_button(
                "Download CSV",
                data=_df_to_csv_bytes(out),
                file_name=f"ntd_sell_signal_{'hourly' if frame10.startswith('Hourly') else 'daily'}_{mode}.csv",
                mime="text/csv",
                key=f"dl_ntd_sell_{mode}",
            )


# ---------------------------
# TAB 11: NPX LOW SCANNER
# ---------------------------
with tab11:
    st.header("NPX Low Scanner")
    st.caption("Lists symbols with latest NPX below a threshold (sorted by NPX ascending).")

    frame11 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=0,
        key=f"npx_low_frame_{mode}",
    )

    npx_low_thr = st.slider(
        "NPX max",
        0.0,
        1.0,
        0.20,
        0.01,
        key=f"npx_low_thr_{mode}",
    )

    run11 = st.button("Run NPX Low Scan", key=f"btn_run_npx_low_{mode}")

    if run11:
        rows = []
        for sym in universe:
            ntd, npx, ts = _latest_ntd_npx(sym, frame11, int(ntd_window))
            if np.isfinite(npx) and float(npx) <= float(npx_low_thr):
                rows.append(
                    {
                        "Symbol": sym,
                        "NPX (Norm Price)": float(npx),
                        "NTD": float(ntd) if np.isfinite(ntd) else np.nan,
                        "Time": ts,
                        "Frame": "Hourly" if frame11.startswith("Hourly") else "Daily",
                    }
                )

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["NPX (Norm Price)", "Symbol"], ascending=[True, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 12: NPX HIGH SCANNER
# ---------------------------
with tab12:
    st.header("NPX High Scanner")
    st.caption("Lists symbols with latest NPX above a threshold (sorted by NPX descending).")

    frame12 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=0,
        key=f"npx_high_frame_{mode}",
    )

    npx_high_thr = st.slider(
        "NPX min",
        0.0,
        1.0,
        0.80,
        0.01,
        key=f"npx_high_thr_{mode}",
    )

    run12 = st.button("Run NPX High Scan", key=f"btn_run_npx_high_{mode}")

    if run12:
        rows = []
        for sym in universe:
            ntd, npx, ts = _latest_ntd_npx(sym, frame12, int(ntd_window))
            if np.isfinite(npx) and float(npx) >= float(npx_high_thr):
                rows.append(
                    {
                        "Symbol": sym,
                        "NPX (Norm Price)": float(npx),
                        "NTD": float(ntd) if np.isfinite(ntd) else np.nan,
                        "Time": ts,
                        "Frame": "Hourly" if frame12.startswith("Hourly") else "Daily",
                    }
                )

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows).sort_values(["NPX (Norm Price)", "Symbol"], ascending=[False, True])
            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 13: NTD RANGE FILTER
# ---------------------------
with tab13:
    st.header("NTD Range Filter")
    st.caption("Filter symbols by a minimum and maximum NTD value (then sort).")

    frame13 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=0,
        key=f"ntd_range_frame_{mode}",
    )

    c1, c2, c3 = st.columns(3)
    ntd_min = c1.number_input("NTD min", value=-0.25, step=0.05, format="%.2f", key=f"ntd_rng_min_{mode}")
    ntd_max = c2.number_input("NTD max", value=0.25, step=0.05, format="%.2f", key=f"ntd_rng_max_{mode}")
    sort13 = c3.selectbox(
        "Sort by",
        ["NPX (asc)", "NPX (desc)", "NTD (asc)", "NTD (desc)"],
        index=0,
        key=f"ntd_rng_sort_{mode}",
    )

    run13 = st.button("Run Range Filter", key=f"btn_run_ntd_range_{mode}")

    if run13:
        rows = []
        lo, hi = float(ntd_min), float(ntd_max)
        if lo > hi:
            lo, hi = hi, lo

        for sym in universe:
            ntd, npx, ts = _latest_ntd_npx(sym, frame13, int(ntd_window))
            if np.isfinite(ntd) and lo <= float(ntd) <= hi:
                rows.append(
                    {
                        "Symbol": sym,
                        "NTD": float(ntd),
                        "NPX (Norm Price)": float(npx) if np.isfinite(npx) else np.nan,
                        "Time": ts,
                        "Frame": "Hourly" if frame13.startswith("Hourly") else "Daily",
                    }
                )

        if not rows:
            st.info("No matches.")
        else:
            out = pd.DataFrame(rows)
            if sort13 == "NPX (asc)":
                out = out.sort_values(["NPX (Norm Price)", "NTD", "Symbol"], ascending=[True, True, True])
            elif sort13 == "NPX (desc)":
                out = out.sort_values(["NPX (Norm Price)", "NTD", "Symbol"], ascending=[False, False, True])
            elif sort13 == "NTD (asc)":
                out = out.sort_values(["NTD", "NPX (Norm Price)", "Symbol"], ascending=[True, True, True])
            else:
                out = out.sort_values(["NTD", "NPX (Norm Price)", "Symbol"], ascending=[False, False, True])

            st.dataframe(out.reset_index(drop=True), use_container_width=True)


# ---------------------------
# TAB 14: UNIVERSE SNAPSHOT TABLE
# ---------------------------
with tab14:
    st.header("Universe Snapshot — Latest NPX & NTD")
    st.caption("Builds a full table for the current universe (frame-selectable) with quick filtering.")

    frame14 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=1,
        key=f"snap_frame_{mode}",
    )

    search_txt = st.text_input("Filter symbols (contains):", value="", key=f"snap_filter_{mode}")
    sort14 = st.selectbox(
        "Sort by",
        ["Symbol", "NPX (asc)", "NPX (desc)", "NTD (asc)", "NTD (desc)"],
        index=1,
        key=f"snap_sort_{mode}",
    )

    run14 = st.button("Build Snapshot", key=f"btn_build_snapshot_{mode}")

    if run14:
        rows = []
        for sym in universe:
            if search_txt and search_txt.upper() not in sym.upper():
                continue
            ntd, npx, ts = _latest_ntd_npx(sym, frame14, int(ntd_window))
            rows.append(
                {
                    "Symbol": sym,
                    "NPX (Norm Price)": float(npx) if np.isfinite(npx) else np.nan,
                    "NTD": float(ntd) if np.isfinite(ntd) else np.nan,
                    "Time": ts,
                    "Frame": "Hourly" if frame14.startswith("Hourly") else "Daily",
                }
            )

        out = pd.DataFrame(rows)
        if out.empty:
            st.info("No rows.")
        else:
            if sort14 == "Symbol":
                out = out.sort_values(["Symbol"], ascending=[True])
            elif sort14 == "NPX (asc)":
                out = out.sort_values(["NPX (Norm Price)", "Symbol"], ascending=[True, True], na_position="last")
            elif sort14 == "NPX (desc)":
                out = out.sort_values(["NPX (Norm Price)", "Symbol"], ascending=[False, True], na_position="last")
            elif sort14 == "NTD (asc)":
                out = out.sort_values(["NTD", "Symbol"], ascending=[True, True], na_position="last")
            else:
                out = out.sort_values(["NTD", "Symbol"], ascending=[False, True], na_position="last")

            st.dataframe(out.reset_index(drop=True), use_container_width=True)
            st.download_button(
                "Download CSV",
                data=_df_to_csv_bytes(out),
                file_name=f"snapshot_{'hourly' if frame14.startswith('Hourly') else 'daily'}_{mode}.csv",
                mime="text/csv",
                key=f"dl_snapshot_{mode}",
            )


# ---------------------------
# TAB 15: WATCHLIST METRICS
# ---------------------------
with tab15:
    st.header("Watchlist Metrics")
    st.caption("Pick a custom list of tickers and view their latest NPX/NTD metrics.")

    frame15 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=1,
        key=f"wl_frame_{mode}",
    )

    picks = st.multiselect(
        "Select symbols:",
        options=universe,
        default=[],
        key=f"wl_picks_{mode}",
    )

    if picks:
        rows = []
        for sym in picks:
            ntd, npx, ts = _latest_ntd_npx(sym, frame15, int(ntd_window))
            rows.append(
                {
                    "Symbol": sym,
                    "NPX (Norm Price)": float(npx) if np.isfinite(npx) else np.nan,
                    "NTD": float(ntd) if np.isfinite(ntd) else np.nan,
                    "Time": ts,
                }
            )

        out = pd.DataFrame(rows).sort_values(["NPX (Norm Price)", "Symbol"], ascending=[True, True], na_position="last")
        st.dataframe(out.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Select one or more symbols.")


# ---------------------------
# TAB 16: EXPORT SYMBOL LISTS
# ---------------------------
with tab16:
    st.header("Export Symbol Lists")
    st.caption("Generate newline / comma-separated lists from quick filters (useful for other tools).")

    frame16 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=1,
        key=f"exp_frame_{mode}",
    )

    c1, c2, c3 = st.columns(3)
    ntd_cut = c1.number_input("NTD < ", value=-0.75, step=0.05, format="%.2f", key=f"exp_ntd_cut_{mode}")
    npx_cut = c2.slider("NPX <= ", 0.0, 1.0, 0.50, 0.01, key=f"exp_npx_cut_{mode}")
    use_both = c3.checkbox("Require BOTH", value=True, key=f"exp_both_{mode}")

    run16 = st.button("Build List", key=f"btn_build_list_{mode}")

    if run16:
        syms = []
        for sym in universe:
            ntd, npx, _ = _latest_ntd_npx(sym, frame16, int(ntd_window))
            cond_ntd = np.isfinite(ntd) and float(ntd) < float(ntd_cut)
            cond_npx = np.isfinite(npx) and float(npx) <= float(npx_cut)
            if use_both:
                if cond_ntd and cond_npx:
                    syms.append(sym)
            else:
                if cond_ntd or cond_npx:
                    syms.append(sym)

        if not syms:
            st.info("No matches.")
        else:
            st.success(f"{len(syms)} symbols")
            st.text_area("Newline list", value="\n".join(syms), height=180, key=f"exp_newline_{mode}")
            st.text_area("Comma list", value=", ".join(syms), height=120, key=f"exp_comma_{mode}")


# ---------------------------
# TAB 17: QUICK STATS
# ---------------------------
with tab17:
    st.header("Quick Stats")
    st.caption("Simple distribution stats for latest NPX/NTD across the universe.")

    frame17 = st.radio(
        "Frame:",
        ["Hourly (intraday)", "Daily"],
        index=1,
        key=f"stats_frame_{mode}",
    )

    run17 = st.button("Compute Stats", key=f"btn_stats_{mode}")

    if run17:
        vals_npx, vals_ntd = [], []
        for sym in universe:
            ntd, npx, _ = _latest_ntd_npx(sym, frame17, int(ntd_window))
            if np.isfinite(npx):
                vals_npx.append(float(npx))
            if np.isfinite(ntd):
                vals_ntd.append(float(ntd))

        if not vals_npx and not vals_ntd:
            st.info("No numeric values found.")
        else:
            colA, colB = st.columns(2)
            if vals_npx:
                s = pd.Series(vals_npx, name="NPX")
                colA.metric("NPX count", int(s.count()))
                colA.write(s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame())
            if vals_ntd:
                s = pd.Series(vals_ntd, name="NTD")
                colB.metric("NTD count", int(s.count()))
                colB.write(s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame())


# ---------------------------
# TAB 18: CACHE / TROUBLESHOOTING
# ---------------------------
with tab18:
    st.header("Cache / Troubleshooting")
    st.caption("Basic utilities to help if data looks stale.")

    c1, c2 = st.columns(2)
    if c1.button("Clear st.cache_data", key=f"clr_cache_data_{mode}"):
        try:
            st.cache_data.clear()
            st.success("st.cache_data cleared.")
        except Exception as e:
            st.error(f"Could not clear st.cache_data: {e}")

    if c2.button("Clear st.cache_resource", key=f"clr_cache_res_{mode}"):
        try:
            st.cache_resource.clear()
            st.success("st.cache_resource cleared.")
        except Exception as e:
            st.error(f"Could not clear st.cache_resource: {e}")

    st.info("If you use custom caches elsewhere, add their clear() calls here too.")


# ---------------------------
# TAB 19: HELP
# ---------------------------
with tab19:
    st.header("Help / Notes")
    st.markdown(
        """
**NPX (Norm Price)** is a 0–1 normalized price position within the selected window.  
**NTD** is a bounded trend-direction score in [-1, 1] based on normalized log-price slope.

Typical workflow:
1) Use **NTD Buy Signal** to find deeply negative NTD names, then prioritize low NPX (already sorted).  
2) Confirm with the chart tabs.  
3) Use exports to build watchlists.

If you want the Buy Signal to be a *true reversal/cross* (not just “NTD below threshold”),
change the condition in Tab 9 to detect a cross on the last two computed NTD points.
        """
    )
