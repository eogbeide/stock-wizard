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

# --- Auto‑refresh logic ---
REFRESH_INTERVAL = 120
def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try: st.experimental_rerun()
        except: pass
auto_refresh()
st.sidebar.markdown(f"**Last refresh:** {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')}")

# --- Sidebar config ---
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock","Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo","3mo","6mo","1y"], index=2)
ticker    = st.sidebar.selectbox(
    "Ticker:",
    sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI'
    ]) if mode=="Stock" else [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'
    ]
)
chart_view = st.sidebar.radio("Chart View:", ["Daily","Hourly","Both"])

# --- Caching helpers ---
@st.cache_data
def load_history(tkr):
    return (
        yf.download(tkr, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )

@st.cache_data
def fit_forecast(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    fc    = model.get_forecast(steps=30)
    idx   = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D")
    return idx, fc.predicted_mean, fc.conf_int()

@st.cache_data
def load_intraday(tkr):
    return yf.download(tkr, period="1d", interval="5m")

# --- Utilities ---
def compute_rsi(data, window=14):
    d    = data.diff()
    gain = d.where(d>0, 0).rolling(window).mean()
    loss = -d.where(d<0, 0).rolling(window).mean()
    rs   = gain/loss
    return 100 - (100/(1+rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def slice_lookback(series: pd.Series, lookback: str) -> pd.Series:
    if lookback.endswith("mo"):
        months = int(lookback[:-2])
        start  = series.index[-1] - pd.DateOffset(months=months)
    else:
        years  = int(lookback[:-1])
        start  = series.index[-1] - pd.DateOffset(years=years)
    return series.loc[start:]

# --- Load data ---
df_hist      = load_history(ticker)
idx, fc_vals, fc_ci = fit_forecast(df_hist)
intraday     = load_intraday(ticker)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Detailed Metrics"
])

# --- Tab 1 ---
with tab1:
    st.header(f"Original Forecast for {ticker}")
    ema200 = df_hist.ewm(span=200).mean()
    ma30   = df_hist.rolling(30).mean()
    lb, mb, ub = compute_bollinger_bands(df_hist)

    if chart_view in ("Daily","Both"):
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(df_hist[-360:], label="History")
        ax.plot(ema200[-360:], "--", label="200 EMA")
        ax.plot(ma30[-360:], "--", label="30 MA")
        ax.plot(idx, fc_vals, label="Forecast")
        ax.fill_between(idx, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.3)
        ax.plot(lb[-360:], "--", label="Lower BB")
        ax.plot(ub[-360:], "--", label="Upper BB")
        ax.set_title(f"{ticker} Daily Forecast")
        ax.legend()
        st.pyplot(fig)

    if chart_view in ("Hourly","Both"):
        hc = intraday["Close"].ffill()
        he = hc.ewm(span=20).mean()
        fig2, ax2 = plt.subplots(figsize=(14,4))
        ax2.plot(hc, label="Intraday")
        ax2.plot(he, "--", label="20 EMA")
        ax2.set_title(f"{ticker} Intraday (5m)")
        ax2.legend()
        st.pyplot(fig2)

    st.write(pd.DataFrame({
        "Forecast": fc_vals,
        "Lower":    fc_ci.iloc[:,0],
        "Upper":    fc_ci.iloc[:,1]
    }, index=idx))

# --- Tab 2 ---
with tab2:
    st.header(f"Enhanced Forecast for {ticker}")
    rsi = compute_rsi(df_hist)

    if chart_view in ("Daily","Both"):
        fig, ax = plt.subplots(figsize=(14,6))
        ax.plot(df_hist[-360:], label="History")
        ax.plot(ema200[-360:], "--", label="200 EMA")
        ax.plot(ma30[-360:], "--", label="30 MA")
        ax.plot(idx, fc_vals, label="Forecast")
        ax.fill_between(idx, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.3)
        high, low = df_hist[-360:].max(), df_hist[-360:].min()
        diff = high - low
        for lev in (0.236,0.382,0.5,0.618):
            ax.hlines(high - diff*lev, df_hist.index[-360], df_hist.index[-1], linestyles="dotted")
        ax.set_title(f"{ticker} Daily + Fib")
        ax.legend()
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(14,3))
        ax2.plot(rsi[-360:], label="RSI(14)")
        ax2.axhline(70, linestyle="--")
        ax2.axhline(30, linestyle="--")
        ax2.legend()
        st.pyplot(fig2)

    if chart_view in ("Hourly","Both"):
        ic = intraday["Close"].ffill()
        ie = ic.ewm(span=20).mean()
        lb2, mb2, ub2 = compute_bollinger_bands(ic)
        ri = compute_rsi(ic)
        fig3, ax3 = plt.subplots(figsize=(14,4))
        ax3.plot(ic, label="Intraday")
        ax3.plot(ie, "--", label="20 EMA")
        ax3.plot(lb2, "--", label="Lower BB")
        ax3.plot(ub2, "--", label="Upper BB")
        ax3.set_title(f"{ticker} Intraday + Fib")
        ax3.legend()
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(14,2))
        ax4.plot(ri, label="RSI(14)")
        ax4.axhline(70, linestyle="--")
        ax4.axhline(30, linestyle="--")
        ax4.legend()
        st.pyplot(fig4)

    st.write(pd.DataFrame({
        "Forecast": fc_vals,
        "Lower":    fc_ci.iloc[:,0],
        "Upper":    fc_ci.iloc[:,1]
    }, index=idx))

# --- Tab 3 ---
with tab3:
    st.header(f"Bull vs Bear Summary for {ticker}")
    slice_series = slice_lookback(df_hist, bb_period).dropna()
    df0 = slice_series.to_frame(name="Close")
    df0['Close'] = pd.to_numeric(df0['Close'], errors='coerce')
    df0 = df0.dropna()
    df0['PctChange'] = df0['Close'].pct_change()
    df0['Bull']      = df0['PctChange'] > 0
    bull = int(df0['Bull'].sum())
    bear = int((~df0['Bull']).sum())
    total = bull + bear

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Days", total)
    c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
    c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")

# --- Tab 4 ---
with tab4:
    st.header(f"Detailed Metrics for {ticker}")
    slice_series = slice_lookback(df_hist, bb_period).dropna()
    df0 = slice_series.to_frame(name="Close")
    df0['Close'] = pd.to_numeric(df0['Close'], errors='coerce')
    df0 = df0.dropna()
    df0['PctChange'] = df0['Close'].pct_change()
    df0['Bull']      = df0['PctChange'] > 0
    bull = int(df0['Bull'].sum())
    bear = int((~df0['Bull']).sum())

    # ensure numeric before rolling
    df0['MA30'] = df0['Close'].rolling(window=30, min_periods=1).mean()

    # Price + MA + Trend
    x = np.arange(len(df0))
    slope, intercept = np.polyfit(x, df0['Close'], 1)
    trend = slope * x + intercept
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df0.index, df0['Close'], label='Close')
    ax.plot(df0.index, df0['MA30'],  label='30‑day MA')
    ax.plot(df0.index, trend, "--",   label='Trend')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Daily % Change")
    st.line_chart(df0['PctChange'], use_container_width=True)

    st.subheader("Bull/Bear Distribution")
    dist_df = pd.DataFrame({
        "Type": ["Bull","Bear"],
        "Days": [bull,bear]
    }).set_index("Type")
    st.bar_chart(dist_df)
