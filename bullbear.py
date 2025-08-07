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
        except:
            pass

auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

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

# --- Data & model helpers ---
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    return s.tz_localize(PACIFIC)

def fetch_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

def compute_sarimax_forecast(series: pd.Series):
    try:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = m.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

def compute_rsi(data, window=14):
    d    = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs   = gain/loss
    return 100 - (100/(1+rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m-num_sd*s, m, m+num_sd*s

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast","Enhanced Forecast","Bull vs Bear","Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; initial run fetches live data, then caches for 15 minutes.")

    selected = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart    = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    auto_run = st.session_state.run_all and (selected != st.session_state.ticker)
    if st.button("Run Forecast") or auto_run:
        df_hist = fetch_hist(selected)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(selected)
        st.session_state.update({
            "df_hist":  df_hist,
            "fc_idx":   idx,
            "fc_vals":  vals,
            "fc_ci":    ci,
            "intraday": intraday,
            "ticker":   selected,
            "chart":    chart,
            "run_all":  True
        })

    if st.session_state.run_all and st.session_state.ticker == selected:
        df, idx, vals, ci = (
            st.session_state.df_hist,
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )

        # Daily plot
        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend()
            st.pyplot(fig)

        # Hourly + trend inversion
        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()

            # rolling slope
            rolling_slope = hc.rolling(12).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
                raw=True
            ).dropna()

            # detect sign changes
            rolling_sign = rolling_slope.apply(np.sign)
            sign_change  = rolling_sign.diff().fillna(0) != 0

            # get inversion timestamps
            inv_times  = sign_change[sign_change].index
            inv_values = hc.loc[inv_times]

            # long-run trend & 20-EMA
            xh = np.arange(len(hc))
            slope, inter = np.polyfit(xh, hc.values, 1)
            trend = slope * xh + inter
            ema20 = hc.ewm(span=20).mean()

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, trend, "--", label="Trend")
            ax2.plot(hc.index, ema20, "--", label="20 EMA")
            ax2.scatter(inv_times, inv_values,
                        color="red", marker="o", s=50,
                        label="Trend Inversion")
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # Forecast summary
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast (unchanged) ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df   = st.session_state.df_hist
        rsi  = compute_rsi(df)
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df[-360:], label="History")
            ax.plot(df.ewm(span=200).mean()[-360:], "--", label="200 EMA")
            ax.plot(df.rolling(30).mean()[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(
                    df[-360:].max() - (df[-360:].max()-df[-360:].min())*lev,
                    df.index[-360], df.index[-1], linestyles="dotted"
                )
            ax.set_xlabel("Date (PST)")
            ax.legend()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--"); ax2.axhline(30, linestyle="--")
            ax2.set_xlabel("Date (PST)")
            ax2.legend()
            st.pyplot(fig2)

        if view in ("Intraday","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            rolling_slope = hc.rolling(12).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
                raw=True
            ).dropna()
            rolling_sign = rolling_slope.apply(np.sign)
            sign_change  = rolling_sign.diff().fillna(0) != 0

            inv_times  = sign_change[sign_change].index
            inv_values = hc.loc[inv_times]

            xh = np.arange(len(hc))
            slope, inter = np.polyfit(xh, hc.values, 1)
            trend = slope * xh + inter
            ema20 = hc.ewm(span=20).mean()

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.plot(hc.index, hc, label="Intraday")
            ax3.plot(hc.index, trend, "--", label="Trend")
            ax3.plot(hc.index, ema20, "--", label="20 EMA")
            ax3.scatter(inv_times, inv_values,
                        color="red", marker="o", s=50,
                        label="Trend Inversion")
            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

        # summary table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=idx))

# --- Tabs 3 & 4 unchanged ----------------------------------------------
