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
  /* hide Streamlit menu, header, footer */
  #MainMenu, header, footer {visibility: hidden;}
  /* on small screens, keep sidebar visible */
  @media (max-width: 600px) {
    .css-18e3th9 {transform: none !important; visibility: visible !important; width: 100% !important; position: relative !important; margin-bottom: 1rem;}
    .css-1v3fvcr {margin-left: 0 !important;}
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
def fetch_hist(ticker: str) -> pd.DataFrame:
    # historical OHLC daily
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))
    df = df[['Open','High','Low','Close']].asfreq('D').fillna(method='ffill')
    return df.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str) -> pd.DataFrame:
    # last 2 days of 5‑min bars for 48h window
    df = yf.download(ticker, period="2d", interval="5m")
    try:
        df = df.tz_localize('UTC')
    except:
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

def compute_atr(df: pd.DataFrame, window=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

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
    st.info("Pick a ticker; data will be cached for 15 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    auto_run = st.session_state.run_all and (sel != st.session_state.ticker)

    if st.button("Run Forecast") or auto_run:
        df_hist   = fetch_hist(sel)
        intraday  = fetch_intraday(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist['Close'])
        st.session_state.update({
            "df_hist": df_hist,
            "intraday": intraday,
            "fc_idx": idx,
            "fc_vals": vals,
            "fc_ci": ci,
            "ticker": sel,
            "chart": chart,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df_hist  = st.session_state.df_hist
        intraday = st.session_state.intraday
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = float(df_hist['Close'].iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        # --- Daily chart: historical daily ---
        if chart in ("Daily","Both"):
            price = df_hist['Close']
            ema200 = price.ewm(span=200).mean()
            ma30   = price.rolling(30).mean()
            lb, mb, ub = compute_bb(price)
            res = price.rolling(30, min_periods=1).max()
            sup = price.rolling(30, min_periods=1).min()

            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,8), sharex=True,
                                           gridspec_kw={'height_ratios':[3,1]})
            ax1.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax1.plot(price[-360:], label="Close")
            ax1.plot(ema200[-360:], "--", label="200 EMA")
            ax1.plot(ma30[-360:], "--", label="30 MA")
            ax1.plot(res[-360:], ":", label="Resistance")
            ax1.plot(sup[-360:], ":", label="Support")
            ax1.plot(idx, vals, label="Forecast")
            trend = np.polyfit(np.arange(len(vals)), vals.to_numpy(), 1)
            ax1.plot(idx, trend[0]*np.arange(len(vals)) + trend[1], "--", label="Trend")
            ax1.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax1.plot(lb[-360:], "--", label="Lower BB")
            ax1.plot(ub[-360:], "--", label="Upper BB")
            ax1.legend(loc="lower left", framealpha=0.5)

            atr = compute_atr(df_hist)[-360:]
            ax2.plot(atr.index, atr, label="ATR(14)")
            ax2.set_ylabel("ATR")
            ax2.legend(loc="upper left", framealpha=0.5)

            ax2.set_xlabel("Date (PST)")
            st.pyplot(fig)

        # --- Hourly chart: last 48 hours of 5‑min bars ---
        if chart in ("Hourly","Both"):
            hc = intraday['Close'].ffill()
            last48 = hc[-576:]  # 576 five‑minute bars = 48h
            he = last48.ewm(span=20).mean()
            xh = np.arange(len(last48))
            slope_h, intercept_h = np.polyfit(xh, last48.values, 1)
            trend_h = slope_h * xh + intercept_h
            res_h = last48.rolling(60, min_periods=1).max()
            sup_h = last48.rolling(60, min_periods=1).min()

            tr_intraday = compute_atr(intraday)[-576:]

            fig2, (ax3, ax4) = plt.subplots(2,1, figsize=(14,6), sharex=True,
                                           gridspec_kw={'height_ratios':[3,1]})
            ax3.set_title(f"{sel} Last 48 Hours  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax3.plot(last48.index, last48, label="Price")
            ax3.plot(last48.index, he, "--", label="20 EMA")
            ax3.plot(last48.index, res_h, ":", label="Resistance")
            ax3.plot(last48.index, sup_h, ":", label="Support")
            ax3.plot(last48.index, trend_h, "--", label="Trend")
            ax3.legend(loc="lower left", framealpha=0.5)

            ax4.plot(tr_intraday.index, tr_intraday, label="ATR(14)")
            ax4.set_ylabel("ATR")
            ax4.legend(loc="upper left", framealpha=0.5)
            ax4.set_xlabel("Time (PST)")

            st.pyplot(fig2)

        # --- Forecast table ---
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 2: Enhanced Forecast (unchanged) ---
# --- Tab 3: Bull vs Bear (unchanged) ---
# --- Tab 4: Metrics (unchanged) ---
