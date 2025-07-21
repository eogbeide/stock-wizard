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

# Universe
if mode == "Stock":
    universe = sorted([...])  # same list as before
else:
    universe = [...]        # same list as before

# --- Caching helpers ---
@st.cache_data
def fetch_hist(ticker: str) -> pd.Series:
    return (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    ).tz_localize(PACIFIC)

@st.cache_data
def fetch_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="5m")
    df = df.tz_localize('UTC').tz_convert(PACIFIC)
    return df

@st.cache_data
def compute_sarimax_forecast(series: pd.Series):
    # same as before, returns idx with tz=PACIFIC
    ...

# indicator helpers as before...

if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None

tab1, tab2, tab3, tab4 = st.tabs([..., ..., ..., ...])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Select a ticker to run (or auto‑run on change).")

    selected = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    # Determine whether to auto-run:
    auto_run = st.session_state.run_all and (selected != st.session_state.ticker)

    if st.button("Run Forecast", key="run") or auto_run:
        # Fetch from cache or live
        df_hist = fetch_hist(selected)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(selected)

        # Store in state
        st.session_state.df_hist = df_hist
        st.session_state.fc_idx = idx
        st.session_state.fc_vals = vals
        st.session_state.fc_ci = ci
        st.session_state.intraday = intraday
        st.session_state.ticker = selected
        st.session_state.chart = chart
        st.session_state.run_all = True

    if st.session_state.run_all and st.session_state.ticker == selected:
        df = st.session_state.df_hist
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci

        if chart in ("Daily","Both"):
            # plot daily (same as before, x-axis labeled PST)
            ...

        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            xh = np.arange(len(hc))
            slope, intercept = np.polyfit(xh, hc.values, 1)
            trend = slope*xh + intercept
            he = hc.ewm(span=20).mean()

            fig, ax = plt.subplots(figsize=(14,4))
            ax.plot(hc.index, hc, label="Intraday")
            ax.plot(hc.index, trend, "--", label="Trend")
            ax.plot(hc.index, he, "--", label="20 EMA")
            ax.set_xlabel("Time (PST)")
            ax.legend()
            st.pyplot(fig)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tabs 2–4 unchanged, pulling from st.session_state ---
