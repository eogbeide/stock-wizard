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

        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            resistance = df.rolling(30, min_periods=1).max()
            support    = df.rolling(30, min_periods=1).min()

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
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope, intercept = np.polyfit(xh, hc.values, 1)
            trend = slope*xh + intercept
            resistance_h = hc.rolling(60, min_periods=1).max()
            support_h    = hc.rolling(60, min_periods=1).min()

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, resistance_h, ":", label="Resist")
            ax2.plot(hc.index, support_h, ":", label="Support")
            ax2.plot(hc.index, trend, "--", label="Trend")
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# Tabs 2–4 follow identically, leveraging the same rolling support/resistance logic
