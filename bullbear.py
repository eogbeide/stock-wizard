import os
import time
import pytz
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

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
    .css-18e3th9 {transform:none!important;visibility:visible!important;width:100%!important;position:relative!important;margin-bottom:1rem;}
    .css-1v3fvcr {margin-left:0!important;}
  }
</style>
""", unsafe_allow_html=True)

# --- Utility ---
def safe_trend(x: np.ndarray, y: np.ndarray):
    try:
        coeff = np.polyfit(x, y, 1)
        return coeff[0]*x + coeff[1], coeff
    except:
        m = np.nanmean(y)
        return np.full_like(x, m, dtype=float), (0.0, m)

# --- News fetcher (requires NEWSAPI_KEY in Streamlit secrets) ---
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
def fetch_bloomberg_news(ticker: str):
    if not NEWSAPI_KEY:
        return []
    resp = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": f"{ticker} Bloomberg",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": NEWSAPI_KEY,
            "pageSize": 5
        }
    ).json()
    items = []
    for art in resp.get("articles", []):
        ts = pd.to_datetime(art["publishedAt"]).tz_convert(PACIFIC)
        items.append({"time": ts, "title": art["title"], "url": art["url"]})
    return items

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        rerun = getattr(st, "experimental_rerun", None)
        if callable(rerun):
            rerun()

auto_refresh()
pst = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst.strftime('%Y-%m-%d %H:%M:%S')} PST")

# --- Sidebar ---
st.sidebar.title("Configuration")
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock","Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo","3mo","6mo","1y"], index=2)

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

# --- Data fetchers (cached) ---
@st.cache_data(ttl=900)
def fetch_hist(ticker: str) -> pd.Series:
    close = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))["Close"]
    return close.asfreq("D").ffill().tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize("UTC")
    except:
        pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def sarimax(series: pd.Series):
    try:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        m = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = m.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1]+timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicators ---
def compute_rsi(d, w=14):
    diff = d.diff()
    gain = diff.where(diff>0,0).rolling(w).mean()
    loss = -diff.where(diff<0,0).rolling(w).mean()
    return 100 - 100/(1+gain/loss)

def compute_bb(d, w=20, sd=2):
    m = d.rolling(w).mean()
    s = d.rolling(w).std()
    return m-sd*s, m, m+sd*s

def compute_macd(s, f=12, sl=26, sig=9):
    e_fast = s.ewm(span=f,adjust=False).mean()
    e_slow = s.ewm(span=sl,adjust=False).mean()
    macd   = e_fast - e_slow
    sigl   = macd.ewm(span=sig,adjust=False).mean()
    hist   = macd - sigl
    return macd, sigl, hist

# --- Session defaults ---
st.session_state.setdefault("run_all", False)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast","Enhanced Forecast","Bull vs Bear","Metrics"
])

with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data caches for 15 minutes.")

    sel    = st.selectbox("Ticker:", universe, key="t1_ticker")
    chart  = st.radio("Chart View:", ["Daily","Hourly","Both"], key="t1_chart")
    hrange = st.selectbox("Hourly lookback:", ["24h","48h","72h","96h","120h"], key="t1_hrange")
    run    = st.button("Run Forecast") or (st.session_state.run_all and sel!=st.session_state.ticker)

    if run or not st.session_state.run_all:
        dfh = fetch_hist(sel)
        df1 = fetch_intraday(sel, period={"24h":"1d","48h":"2d","72h":"3d","96h":"4d","120h":"5d"}[hrange])
        idx,vals,ci = sarimax(dfh)
        news = fetch_bloomberg_news(sel)
        st.session_state.update({
            "df_hist": dfh, "df_intr": df1,
            "fc_idx": idx,  "fc_vals": vals, "fc_ci": ci,
            "news": news,   "ticker": sel,
            "chart": chart, "hrange": hrange,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker==sel:
        dfh  = st.session_state.df_hist
        df1  = st.session_state.df_intr
        idx,vals,ci = (st.session_state.fc_idx,
                       st.session_state.fc_vals,
                       st.session_state.fc_ci)
        news = st.session_state.news

        last = float(dfh.iloc[-1])
        p_up, p_dn = np.mean(vals>last), 1-np.mean(vals>last)
        trend = ((float(vals.mean())-last)/last*100) if last else 0
        t_lbl = f"{trend:+.2f}%"

        # Intraday w/ SMA
        if chart in ("Hourly","Both"):
            hc = df1["Close"].ffill()
            ema20 = hc.ewm(span=20).mean()
            sma12 = hc.rolling(12, min_periods=1).mean()
            xh = np.arange(len(hc))
            tr_h, cf = safe_trend(xh, hc.values)
            base = float(hc.iloc[0]) or 1
            slope = cf[0]*(len(hc)-1)/base*100

            fig,ax = plt.subplots(figsize=(14,4))
            ax.set_title(f"{sel} Intraday ({hrange})  ↑{p_up:.1%}  ↓{p_dn:.1%}  Trend: {slope:.2f}%")
            ax.plot(hc.index, hc,      label="Close")
            ax.plot(hc.index, sma12,   label="12-period SMA")
            ax.plot(hc.index, ema20, "--", label="20-period EMA")
            ax.plot(hc.index, tr_h, "--",   label="Trend")
            ax.set_xlabel("Time (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # Daily + MACD unchanged…
        # … (the rest of your daily & other tabs code)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))
