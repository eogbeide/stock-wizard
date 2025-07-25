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
    layout="wide"
)
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- Auto‑refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try:
            st.experimental_rerun()
        except:
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
    series = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    return series.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="5m")
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
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# Indicator helpers
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d > 0, 0).rolling(window).mean()
    loss = -d.where(d < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

# Session state init
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None

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
    st.info("Pick a ticker; initial run fetches live data, then cached for 15 minutes.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    auto_run = st.session_state.run_all and (sel != st.session_state.ticker)
    if st.button("Run Forecast", key="run") or auto_run:
        df_hist = fetch_hist(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel)
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": idx,
            "fc_vals": vals,
            "fc_ci": ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df   = st.session_state.df_hist
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)

        # Daily forecast & indicators
        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            resistance = df.rolling(30, min_periods=1).max()
            support    = df.rolling(30, min_periods=1).min()
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x, df.values, 1)
            trend = slope*x + intercept

            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(resistance[-360:], ":", label="30 Resist")
            ax.plot(support[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.plot(df.index[-360:], trend[-360:], "--", label="Trend")
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # Intraday price & EMA & trend
        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
            trend_h = slope_h*xh + intercept_h
            resistance_h = hc.rolling(60, min_periods=1).max()
            support_h    = hc.rolling(60, min_periods=1).min()

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, resistance_h, ":", label="Resist")
            ax2.plot(hc.index, support_h, ":", label="Support")
            ax2.plot(hc.index, trend_h, "--", label="Trend")
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        sel = st.session_state.ticker
        view = st.radio("View:", ["Daily","Hourly","Both"], key="enh_view")
        df = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi = compute_rsi(df)

        # Enhanced daily view + trend
        if view in ("Daily","Both"):
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x, df.values, 1)
            trend = slope*x + intercept

            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(trend[-360:], "--", label="Trend")
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            high, low = df[-360:].max(), df[-360:].min()
            diff = high - low
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(high - diff*lev, df.index[-360], df.index[-1], linestyles="dotted")
            ax.set_title(f"{sel} Daily + Indicators + Trend")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            # RSI subplot
            fig2, ax2 = plt.subplots(figsize=(14,2))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--")
            ax2.axhline(30, linestyle="--")
            ax2.legend(loc="lower left")
            st.pyplot(fig2)

        # Enhanced intraday view
        if view in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            ie = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
            trend_h = slope_h*xh + intercept_h
            lb2, mb2, ub2 = compute_bb(hc)

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.plot(hc.index, hc, label="Intraday")
            ax3.plot(hc.index, ie, "--", label="20 EMA")
            ax3.plot(hc.index, trend_h, "--", label="Trend")
            ax3.plot(hc.index, lb2, "--", label="Lower BB")
            ax3.plot(hc.index, ub2, "--", label="Upper BB")
            ax3.set_title(f"{sel} Intraday + Indicators + Trend")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        bull = int(df0['Bull'].sum())
        bear = int((~df0['Bull']).sum())
        total = bull + bear
        bp = bull/total*100 if total else 0
        brp = bear/total*100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bp:.1f}%")
        c3.metric("Bear Days", bear, f"{brp:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Metrics")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        df0 = (
            yf.download(st.session_state.ticker, period=bb_period)[['Close']]
            .dropna().tz_localize(PACIFIC)
        )
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        df0['MA30']      = df0['Close'].rolling(window=30, min_periods=1).mean()

        # Daily Price + MA + Trend (short lookback)
        st.subheader("Price Chart → Close + 30‑day MA + Trend")
        x = np.arange(len(df0))
        slope, intercept = np.polyfit(x, df0['Close'], 1)
        trend = slope * x + intercept

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df0.index, df0['Close'], label='Close')
        ax.plot(df0.index, df0['MA30'],  label='30‑day MA')
        ax.plot(df0.index, trend, "--",   label='Trend')
        ax.set_title(f"{st.session_state.ticker} Price + MA + Trend ({bb_period})")
        ax.legend(loc="lower left")
        st.pyplot(fig)

        # Full-history daily Price + MA + Trend
        st.subheader("Daily Chart → Full History Close + 30‑day MA + Trend")
        df_full = st.session_state.df_hist
        ma30_full = df_full.rolling(30, min_periods=1).mean()
        x_full = np.arange(len(df_full))
        slope_f, intercept_f = np.polyfit(x_full, df_full.values, 1)
        trend_full = slope_f * x_full + intercept_f

        fig2, ax2 = plt.subplots(figsize=(14,5))
        ax2.plot(df_full.index, df_full, label='Close')
        ax2.plot(df_full.index, ma30_full, "--", label='30‑day MA')
        ax2.plot(df_full.index, trend_full, "--", label='Trend')
        ax2.set_title(f"{st.session_state.ticker} Full History + MA + Trend")
        ax2.legend(loc="lower left")
        st.pyplot(fig2)

        st.subheader("Daily Percentage Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist_df = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        })
        st.bar_chart(dist_df.set_index("Type"), use_container_width=True)
