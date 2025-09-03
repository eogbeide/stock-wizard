# bullbear.py — Stocks/Forex Dashboard + Forecasts
# - Adds Forex news markers on daily & intraday charts
# - Fixes tz_localize error by using tz-aware UTC timestamps
# - Adds Momentum (ROC) indicator to hourly charts
# - Keeps auto-refresh, SARIMAX, RSI/BB/Fibs, slopes, etc.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz

# --- Page config & CSS ---
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 {
      transform: none !important;
      visibility: visible !important;
      width: 100% !important;
      position: relative !important;
      margin-bottom: 1rem;
    }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.experimental_rerun()
        except Exception:
            pass

auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly & daily)", value=True)
slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10)
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120, 6)
mom_lb_hourly    = st.sidebar.slider("Hourly momentum lookback (bars)", 5, 60, 10, 1)  # NEW

# Forex news controls (only shown in Forex mode)
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers", value=True)
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7)
else:
    show_fx_news = False
    news_window_days = 7  # unused in stock mode

# Universe
if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'
    ]

# --- Cache helpers ---
@st.cache_data(ttl=900)
def fetch_hist(ticker: str) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    return s.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def compute_sarimax_forecast(series: pd.Series):
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---- Indicators ----
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def fibonacci_levels(series: pd.Series):
    if series is None or len(series) == 0:
        return {}
    # Ensure 1D numeric series
    if isinstance(series, pd.DataFrame):
        num_cols = [c for c in series.columns if pd.api.types.is_numeric_dtype(series[c])]
        if not num_cols:
            return {}
        s = series[num_cols[0]].dropna()
    elif isinstance(series, pd.Series):
        s = pd.to_numeric(series, errors="coerce").dropna()
    else:
        s = pd.Series(series).dropna()
    if s.empty:
        return {}
    hi = float(s.max()); lo = float(s.min())
    diff = hi - lo
    if diff == 0:
        return {"100%": lo}
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5   * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100%": lo,
    }

def compute_roc(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of Change (percentage): 100 * (Close/Close_n - 1)"""
    s = pd.to_numeric(series, errors="coerce")
    return (s / s.shift(n) - 1.0) * 100.0

# ---- Robust slope helpers ----
def _coerce_1d_series(obj) -> pd.Series:
    """Return a numeric Series from Series/DataFrame/array-like; empty Series if impossible."""
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

def slope_line(series_like, lookback: int):
    """
    Fit y = m*x + b over the last `lookback` points of a 1D numeric series.
    Accepts Series, DataFrame, list, ndarray. Returns (yhat Series, slope float).
    """
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    s = s.iloc[-lookback:] if lookback > 0 else s
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    yhat = pd.Series(m * x + b, index=s.index)
    return yhat, float(m)

def fmt_slope(m: float) -> str:
    return f"{m:.4f}" if np.isfinite(m) else "n/a"

# ---- Forex News (Yahoo Finance) ----
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yf_news(symbol: str, window_days: int = 7) -> pd.DataFrame:
    """
    Fetch recent Yahoo Finance news for a symbol.
    Returns a DataFrame with PST timestamps, titles, and links, limited to window_days.
    Gracefully returns empty DataFrame if none/failed.
    """
    rows = []
    try:
        news_list = yf.Ticker(symbol).news or []
    except Exception:
        news_list = []

    for item in news_list:
        ts = item.get("providerPublishTime")
        if ts is None:
            ts = item.get("pubDate")
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
        rows.append({
            "time": dt_pst,
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", "")
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

def draw_news_markers(ax, times, ymin, ymax, label="News"):
    """Plot faint vertical lines at given times."""
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 15 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    auto_run = (
        st.session_state.run_all and (
            sel != st.session_state.ticker or
            hour_range != st.session_state.get("hour_range")
        )
    )

    if st.button("Run Forecast") or auto_run:
        df_hist = fetch_hist(sel)                       # Series
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])  # DataFrame
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": fc_idx,
            "fc_vals": fc_vals,
            "fc_ci": fc_ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist                  # Series
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )

        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        x_fc = np.arange(len(vals))
        slope_fc, intercept_fc = np.polyfit(x_fc, vals.to_numpy(), 1)
        trend_fc = slope_fc * x_fc + intercept_fc

        # Pre-fetch Forex news if applicable
        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily -----
        if chart in ("Daily","Both"):
            df_show = df[-360:]
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            res = df.rolling(30, min_periods=1).max()
            sup = df.rolling(30, min_periods=1).min()

            yhat_d, m_d = slope_line(df, slope_lb_daily)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df_show, label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="30 Resistance")
            ax.plot(sup[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.plot(idx, trend_fc, "--", label="Forecast Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")

            if not yhat_d.empty:
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2,
                        label=f"Slope {slope_lb_daily} bars ({fmt_slope(m_d)}/bar)")

            if show_fibs:
                fibs_d = fibonacci_levels(df_show)
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1],
                              linestyles="dotted", linewidth=1)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")

            # Forex news markers on daily
            if not fx_news.empty:
                t0, t1 = df_show.index[0], df_show.index[-1]
                times = [t for t in fx_news["time"] if t0 <= t <= t1]
                if times:
                    ymin, ymax = float(df_show.min()), float(df_show.max())
                    draw_news_markers(ax, times, ymin, ymax, label="News")

            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # ----- Hourly -----
        if chart in ("Hourly","Both"):
            intr = st.session_state.intraday              # DataFrame
            hc = intr["Close"].ffill()
            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
            trend_h = slope_h * xh + intercept_h
            res_h = hc.rolling(60, min_periods=1).max()
            sup_h = hc.rolling(60, min_periods=1).min()

            yhat_h, m_h = slope_line(hc, slope_lb_hourly)

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, res_h, ":", label="Resistance")
            ax2.plot(hc.index, sup_h, ":", label="Support")
            ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)

            if not yhat_h.empty:
                ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                         label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

            if show_fibs and not hc.empty:
                fibs_h = fibonacci_levels(hc)
                for lbl, y in fibs_h.items():
                    ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1],
                               linestyles="dotted", linewidth=1)
                for lbl, y in fibs_h.items():
                    ax2.text(hc.index[-1], y, f" {lbl}", va="center")

            # Forex news markers on intraday
            if not hc.empty and not fx_news.empty:
                t0, t1 = hc.index[0], hc.index[-1]
                times = [t for t in fx_news["time"] if t0 <= t <= t1]
                if times:
                    ymin, ymax = float(hc.min()), float(hc.max())
                    draw_news_markers(ax2, times, ymin, ymax, label="News")

            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

            # --- NEW: Hourly Momentum (ROC) panel ---
            if not hc.empty:
                roc = compute_roc(hc, n=mom_lb_hourly)
                fig2m, ax2m = plt.subplots(figsize=(14,2.6))
                ax2m.set_title(f"Hourly Momentum (ROC {mom_lb_hourly})")
                ax2m.plot(roc.index, roc, label=f"ROC({mom_lb_hourly})")
                ax2m.axhline(0, linestyle="--", linewidth=1)
                ax2m.set_xlabel("Time (PST)")
                ax2m.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2m)

        # Optional: small table of recent FX news
        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi    = compute_rsi(df)
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up
        res = df.rolling(30, min_periods=1).max()
        sup = df.rolling(30, min_periods=1).min()

        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}** "
                   "(change in 'Original Forecast' tab and rerun)")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
        if view in ("Daily","Both"):
            df_show = df[-360:]
            yhat_d, m_d = slope_line(df, slope_lb_daily)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df_show, label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="Resistance")
            ax.plot(sup[-360:], ":", label="Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)

            if not yhat_d.empty:
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2,
                        label=f"Slope {slope_lb_daily} bars ({fmt_slope(m_d)}/bar)")

            if show_fibs:
                fibs_d = fibonacci_levels(df_show)
                for lbl, y in fibs_d.items():
                    ax.hlines(y, xmin=df_show.index[0], xmax=df_show.index[-1],
                              linestyles="dotted", linewidth=1)
                for lbl, y in fibs_d.items():
                    ax.text(df_show.index[-1], y, f" {lbl}", va="center")

            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--"); ax2.axhline(30, linestyle="--")
            ax2.set_xlabel("Date (PST)")
            ax2.legend()
            st.pyplot(fig2)

        if view in ("Intraday","Both"):
            ic = st.session_state.intraday["Close"].ffill()
            ie = ic.ewm(span=20).mean()
            xi = np.arange(len(ic))
            slope_i, intercept_i = np.polyfit(xi, ic.values, 1)
            trend_i = slope_i * xi + intercept_i
            res_i = ic.rolling(60, min_periods=1).max()
            sup_i = ic.rolling(60, min_periods=1).min()

            yhat_h, m_h = slope_line(ic, slope_lb_hourly)

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range})  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax3.plot(ic.index, ic, label="Intraday")
            ax3.plot(ic.index, ie, "--", label="20 EMA")
            ax3.plot(ic.index, res_i, ":", label="Resistance")
            ax3.plot(ic.index, sup_i, ":", label="Support")
            ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)

            if not yhat_h.empty:
                ax3.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                         label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

            if show_fibs and not ic.empty:
                fibs_h = fibonacci_levels(ic)
                for lbl, y in fibs_h.items():
                    ax3.hlines(y, xmin=ic.index[0], xmax=ic.index[-1],
                               linestyles="dotted", linewidth=1)
                for lbl, y in fibs_h.items():
                    ax3.text(ic.index[-1], y, f" {lbl}", va="center")

            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(14,3))
            ri = compute_rsi(ic)
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)")
            ax4.legend()
            st.pyplot(fig4)

            # --- NEW: Hourly Momentum (ROC) panel in Enhanced view ---
            if not ic.empty:
                roc_i = compute_roc(ic, n=mom_lb_hourly)
                fig5, ax5 = plt.subplots(figsize=(14,2.6))
                ax5.set_title(f"Hourly Momentum (ROC {mom_lb_hourly})")
                ax5.plot(roc_i.index, roc_i, label=f"ROC({mom_lb_hourly})")
                ax5.axhline(0, linestyle="--", linewidth=1)
                ax5.set_xlabel("Time (PST)")
                ax5.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig5)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange'] > 0
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last_price = float(df_hist.iloc[-1])
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        st.subheader(f"Last 3 Months  ↑{p_up:.1%}  ↓{p_dn:.1%}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        x3m = np.arange(len(df3m))
        slope3m, intercept3m = np.polyfit(x3m, df3m.values, 1)
        trend3m = slope3m * x3m + intercept3m

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(df3m.index, res3m, ":", label="Resistance")
        ax.plot(df3m.index, sup3m, ":", label="Support")
        ax.plot(df3m.index, trend3m, "--", label="Trend")
        ax.set_xlabel("Date (PST)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        x0 = np.arange(len(df0))
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0 * x0 + intercept0
        res0 = df0.rolling(30, min_periods=1).max()
        sup0 = df0.rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(df0.index, res0, ":", label="Resistance")
        ax0.plot(df0.index, sup0, ":", label="Support")
        ax0.plot(df0.index, trend0, "--", label="Trend")
        ax0.set_xlabel("Date (PST)")
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)
