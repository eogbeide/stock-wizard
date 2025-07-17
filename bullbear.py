import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time

# --- Page config & CSS ---
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide"
)
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- Auto‐refresh logic ---
REFRESH_INTERVAL = 120  # seconds
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
st.sidebar.markdown(
    f"**Last refresh:** {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')}"
)

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="global_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

# --- Prepare state ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False

# --- Indicator & forecasting helpers ---
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def safe_sarimax(endog, order, seasonal_order):
    try:
        return SARIMAX(endog, order=order, seasonal_order=seasonal_order).fit(disp=False)
    except np.linalg.LinAlgError:
        return SARIMAX(
            endog, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🇺🇸 Original US Forecast",
    "🇺🇸 Enhanced US Forecast",
    "🐂 Bull vs Bear Summary",
    "📊 Detailed Metrics"
])

# --- Tab 1: Original US Forecast ---
with tab1:
    st.header("🇺🇸 Original US Forecast")
    st.info("Use this tab and click **Run Forecast** to initialize all data—then visit the other tabs.")

    # 1) Select ticker
    if mode == "Stock":
        ticker = st.selectbox(
            "Stock Ticker:",
            sorted(['AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                    'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                    'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI']),
            key="orig_stock_ticker"
        )
    else:
        ticker = st.selectbox(
            "Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'],
            key="orig_forex_pair"
        )

    # 2) Chart view
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    # 3) Run forecast button + progress
    if st.button("Run Forecast", key="orig_btn"):
        progress = st.progress(0)
        # Step 1: store selection
        st.session_state.ticker = ticker
        st.session_state.chart = chart
        st.session_state.mode = mode
        progress.progress(10)

        # Step 2: historic daily series
        df_hist = (
            yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
            .asfreq("D").fillna(method="ffill")
        )
        st.session_state.df_hist = df_hist
        progress.progress(40)

        # Step 3: SARIMAX fit & forecast
        model = safe_sarimax(df_hist, (1,1,1), (1,1,1,12))
        fc = model.get_forecast(steps=30)
        idx = pd.date_range(df_hist.index[-1] + timedelta(1), periods=30, freq="D")
        st.session_state.fc_idx = idx
        st.session_state.fc_vals = fc.predicted_mean
        st.session_state.fc_ci = fc.conf_int()
        progress.progress(70)

        # Step 4: intraday if stock
        if mode=="Stock":
            intraday = yf.download(ticker, period="1d", interval="5m")
            st.session_state.intraday = intraday
        st.session_state.run_all = True
        progress.progress(100)
        st.success("Forecast complete!")

    # 4) Display results
    if st.session_state.run_all:
        df = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30 = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        idx = st.session_state.fc_idx
        vals = st.session_state.fc_vals
        ci = st.session_state.fc_ci

        if chart in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_title(f"{ticker} Daily Forecast")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if chart in ("Hourly","Both") and mode=="Stock":
            intraday = st.session_state.intraday
            if intraday.empty:
                st.warning("No intraday data.")
            else:
                hc = intraday["Close"].ffill()
                he = hc.ewm(span=20).mean()
                fig2, ax2 = plt.subplots(figsize=(14,5))
                ax2.plot(hc, label="Intraday")
                ax2.plot(he, "--", label="20 EMA")
                ax2.set_title(f"{ticker} Intraday (5m)")
                ax2.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

    else:
        st.info("Press **Run Forecast** above to see results here and in other tabs.")

# --- Tab 2: Enhanced US Forecast ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        # ... (same as before) ...
        # [No changes here; content appears once run_all=True]

# --- Tab 3: Bull vs Bear Summary ---
with tab3:
    st.header("🐂 Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        # ... (same as before) ...

# --- Tab 4: Detailed Metrics ---
with tab4:
    st.header("📊 Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        # ... (same as before) ...
