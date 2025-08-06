import streamlit as sts
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
  /* mobile sidebar override */
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
        trend = coeff[0] * x + coeff[1]
        return trend, coeff
    except Exception:
        m = np.nanmean(y)
        return np.full_like(x, m, dtype=float), (0.0, m)

# --- Auto‐refresh logic ---
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

# --- Data fetchers (no caching) ---
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
    return df.asfreq("D").fillna(method="ffill").tz_localize(PACIFIC)

def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize("UTC")
    except:
        pass
    return df.tz_convert(PACIFIC)

def compute_sarimax_forecast(series: pd.Series):
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicator helpers ---
def compute_rsi(data, window=14):
    d    = data.diff()
    gain = d.where(d > 0, 0).rolling(window).mean()
    loss = -d.where(d < 0, 0).rolling(window).mean()
    rs   = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    hist       = macd_line - signal_line
    return macd_line, signal_line, hist

# --- Session init ---
st.session_state.setdefault("run_all", False)
st.session_state.setdefault("ticker", None)
st.session_state.setdefault("hour_range", "24h")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast", "Enhanced Forecast", "Bull vs Bear", "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be fetched on every run.")

    sel        = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart      = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:", ["24h","48h"], key="hour_range_select")
    auto_run   = st.session_state.run_all and (sel != st.session_state.ticker)

    if st.button("Run Forecast") or auto_run or not st.session_state.run_all:
        df_hist = fetch_hist(sel)
        intraday_period = "2d" if hour_range=="48h" else "1d"
        df_intr = fetch_intraday(sel, period=intraday_period)
        idx, vals, ci = compute_sarimax_forecast(df_hist)

        st.session_state.update({
            "df_hist": df_hist,
            "df_intr": df_intr,
            "fc_idx":  idx,
            "fc_vals": vals,
            "fc_ci":   ci,
            "ticker":  sel,
            "chart":   chart,
            "hour_range": hour_range,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker==sel:
        df    = st.session_state.df_hist
        dfint = st.session_state.df_intr
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci

        last_price = float(df.iloc[-1])
        p_up  = np.mean(vals>last_price)
        p_dn  = 1 - p_up
        trend_pct = ((float(vals.mean()) - last_price) / last_price)*100 if last_price else 0.0
        trend_lbl = f"{trend_pct:+.2f}%"

        # Hourly
        if chart in ("Hourly","Both"):
            hc = dfint["Close"].ffill()
            sma = hc.rolling(12).mean()
            ema = hc.ewm(span=20).mean()
            xh, yh = np.arange(len(hc)), hc.values
            tr_h, coeff_h = safe_trend(xh, yh)
            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday ({hour_range})  ↑{p_up:.1%}  ↓{p_dn:.1%}  Trend: {trend_lbl}")
            ax2.plot(hc.index, hc, label="Close")
            ax2.plot(hc.index, sma, "--", label="12‐pt SMA")
            ax2.plot(hc.index, ema, "--", label="20‐pt EMA")
            ax2.plot(hc.index, tr_h, "--", label="Trend")
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # Daily (unchanged)...

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))
