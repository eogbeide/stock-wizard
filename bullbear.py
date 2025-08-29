# bullbear.py
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
            if hasattr(st, "rerun"):
                st.rerun()
            else:
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

# Universe for selection
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

# --- Caching helpers (refresh every 15 minutes) ---
@st.cache_data(ttl=900)
def fetch_hist(ticker: str) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    return s.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    # period options: "1d" (~24h), "2d" (~48h), "4d" (~96h) with 5m interval
    df = yf.download(ticker, period=period, interval="5m")
    if df.empty:
        return df
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

# -------- Indicator helpers --------
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

def _coerce_volume_series(df: pd.DataFrame):
    """Return numeric Series for 'Volume' or None if not usable."""
    if "Volume" not in df.columns:
        return None
    vol = df["Volume"]
    # If Volume accidentally comes as DataFrame (duplicate column names), squeeze:
    if isinstance(vol, pd.DataFrame):
        if vol.shape[1] == 1:
            vol = vol.iloc[:, 0]
        else:
            # take the first numeric column
            for col in vol.columns:
                s = pd.to_numeric(vol[col], errors="coerce")
                if s.notna().any():
                    vol = s
                    break
    vol = pd.to_numeric(vol, errors="coerce")
    vol = vol.dropna()
    if vol.empty:
        return None
    return vol

def normalized_volume_1h(df_5m: pd.DataFrame, window:int = 20, method:str = "zscore"):
    """
    From 5-minute data, build 1-hour volume and normalize.
    Returns (vol_1h, norm) or (None, None) if volume isn't usable.
    """
    if df_5m is None or df_5m.empty:
        return None, None

    vol = _coerce_volume_series(df_5m)
    if vol is None:
        return None, None

    # Hourly aggregation; keep NaN if none in the hour (min_count=1)
    vol_h = vol.resample("1H").sum(min_count=1)

    # If still all NaN or (almost) all zeros
    if vol_h.dropna().empty:
        return None, None
    if np.nansum(vol_h.to_numpy()) == 0:
        return None, None

    # Rolling stats with sensible minimums to avoid all-NaN windows
    minp = max(3, window // 2)
    mean_h = vol_h.rolling(window, min_periods=minp).mean()
    if method.lower() == "ratio":
        norm = vol_h / mean_h
    else:
        std_h = vol_h.rolling(window, min_periods=minp).std(ddof=0)
        norm = (vol_h - mean_h) / std_h
    norm = norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return vol_h, norm

# Session state init
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

    # Hourly lookback selector
    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}
    vol_window = st.number_input("Normalized Volume lookback (hours)", 10, 60, 20, 1)

    auto_run = (
        st.session_state.run_all and (
            sel != st.session_state.ticker or
            hour_range != st.session_state.get("hour_range")
        )
    )

    if st.button("Run Forecast") or auto_run:
        df_hist = fetch_hist(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": idx,
            "fc_vals": vals,
            "fc_ci": ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
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

        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            res = df.rolling(30, min_periods=1).max()
            sup = df.rolling(30, min_periods=1).min()

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="30 Resistance")
            ax.plot(sup[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.plot(idx, trend_fc, "--", label="Forecast Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if chart in ("Hourly","Both"):
            intr = st.session_state.intraday[['Open','High','Low','Close','Volume']].dropna(how="all")
            if intr.empty:
                st.warning("No intraday data available.")
            else:
                hc = intr["Close"].ffill()
                he = hc.ewm(span=20).mean()
                xh = np.arange(len(hc))
                slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
                trend_h = slope_h * xh + intercept_h
                res_h = hc.rolling(60, min_periods=1).max()
                sup_h = hc.rolling(60, min_periods=1).min()

                # ---- Normalized Volume (1H) ----
                vol_1h, vol_norm = normalized_volume_1h(intr, window=int(vol_window), method="zscore")

                # Price + normalized volume stacked chart
                fig2, (ax2, axv) = plt.subplots(
                    2, 1, figsize=(14,6), sharex=True,
                    gridspec_kw={'height_ratios':[3,1]}
                )
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{p_up:.1%}  ↓{p_dn:.1%}")
                ax2.plot(hc.index, hc, label="5-min Close")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, res_h, ":", label="Resistance")
                ax2.plot(hc.index, sup_h, ":", label="Support")
                ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)
                ax2.set_ylabel("Price")
                ax2.legend(loc="lower left", framealpha=0.5)

                if vol_1h is None:
                    axv.text(0.5, 0.5, "No usable volume for this ticker (common for FX).",
                             transform=axv.transAxes, ha="center", va="center")
                else:
                    axv.bar(vol_norm.index, vol_norm.values, width=0.03)
                    axv.axhline(0, linestyle="--", linewidth=1)
                    axv.axhline(2, linestyle="--", linewidth=1)  # 2σ spike guide
                    axv.set_ylabel("Norm Vol (z)")
                axv.set_xlabel("Time (PST)")
                st.pyplot(fig2)

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
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="30 Resistance")
            ax.plot(sup[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(
                    df[-360:].max() - (df[-360:].max() - df[-360:].min())*lev,
                    df.index[-360], df.index[-1], linestyles="dotted"
                )
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
            intr = st.session_state.intraday[['Open','High','Low','Close','Volume']].dropna(how="all")
            if intr.empty:
                st.warning("No intraday data available.")
            else:
                ic = intr['Close'].ffill()
                ie = ic.ewm(span=20).mean()
                xi = np.arange(len(ic))
                slope_i, intercept_i = np.polyfit(xi, ic.values, 1)
                trend_i = slope_i * xi + intercept_i
                res_i = ic.rolling(60, min_periods=1).max()
                sup_i = ic.rolling(60, min_periods=1).min()

                vol_1h, vol_norm = normalized_volume_1h(intr, window=20, method="zscore")

                fig3, (ax3, axv3) = plt.subplots(2, 1, figsize=(14,6), sharex=True,
                                                 gridspec_kw={'height_ratios':[3,1]})
                ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range})")
                ax3.plot(ic.index, ic, label="5-min Close")
                ax3.plot(ic.index, ie, "--", label="20 EMA")
                ax3.plot(ic.index, res_i, ":", label="Resistance")
                ax3.plot(ic.index, sup_i, ":", label="Support")
                ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)
                ax3.legend(loc="lower left", framealpha=0.5)

                if vol_1h is None:
                    axv3.text(0.5, 0.5, "No usable volume for this ticker (common for FX).",
                              transform=axv3.transAxes, ha="center", va="center")
                else:
                    axv3.bar(vol_norm.index, vol_norm.values, width=0.03)
                    axv3.axhline(0, linestyle="--", linewidth=1)
                    axv3.axhline(2, linestyle="--", linewidth=1)
                    axv3.set_ylabel("Norm Vol (z)")
                axv3.set_xlabel("Time (PST)")
                st.pyplot(fig3)

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
