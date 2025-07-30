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
  /* mobile sidebar override */
  @media (max-width:600px) {
    .css-18e3th9 {transform:none!important;visibility:visible!important;width:100%!important;
                  position:relative!important;margin-bottom:1rem;}
    .css-1v3fvcr {margin-left:0!important;}
  }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh ---
REFRESH_INTERVAL = 120
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try: st.experimental_rerun()
        except: pass

auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt:%Y-%m-%d %H:%M:%S} PST")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock","Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo","3mo","6mo","1y"], index=2)

# Universe
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

# --- Data & Forecast Caches ---
@st.cache_data(ttl=900)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    return s.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="1d", interval="5m")
    try: df = df.tz_localize('UTC')
    except: pass
    return df['Close'].tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def compute_sarimax_forecast(series: pd.Series):
    try:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc  = m.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicator Caches ---
@st.cache_data(ttl=900)
def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    d    = data.diff()
    gain = d.where(d > 0, 0).rolling(window).mean()
    loss = -d.where(d < 0, 0).rolling(window).mean()
    rs   = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900)
def compute_bb(data: pd.Series, window: int = 20, num_sd: int = 2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

@st.cache_data(ttl=900)
def compute_momentum(data: pd.Series, window: int = 10) -> pd.Series:
    return data - data.shift(window)

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.update({'run_all': False, 'ticker': None})

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast", "Enhanced Forecast", "Bull vs Bear", "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Select ticker and click Run Forecast")

    sel   = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    auto  = st.session_state.run_all and sel != st.session_state.ticker

    if st.button("Run Forecast") or auto:
        h, intr = fetch_hist(sel), fetch_intraday(sel)
        idx, vals, ci = compute_sarimax_forecast(h)
        st.session_state.update({
            "df_hist": h, "fc_idx": idx, "fc_vals": vals, "fc_ci": ci,
            "intraday": intr, "ticker": sel, "chart": chart, "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df, idx, vals, ci = (
            st.session_state.df_hist,
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last = float(df.iloc[-1])
        p_up = np.mean(vals > last)
        p_dn = 1 - p_up

        # Daily
        if chart in ("Daily", "Both"):
            df360 = df[-360:]
            ema200 = df360.ewm(span=200).mean()
            ma30   = df360.rolling(30).mean()
            lb, mb, ub = compute_bb(df360)
            res = df360.rolling(30, min_periods=1).max()
            sup = df360.rolling(30, min_periods=1).min()
            x = np.arange(len(vals))
            slope, intercept = np.polyfit(x, vals, 1)
            trend_fc = slope*x + intercept

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df360, label="History")
            ax.plot(ema200, "--", label="200 EMA")
            ax.plot(ma30, "--", label="30 MA")
            ax.plot(res, ":", label="Resistance")
            ax.plot(sup, ":", label="Support")
            ax.plot(idx, vals, label="Forecast")
            ax.plot(idx, trend_fc, "--", label="Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb, "--", label="Lower BB")
            ax.plot(ub, "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(framealpha=0.5, loc="lower left")
            st.pyplot(fig)

        # Intraday (fixed trend calc)
        if chart in ("Hourly", "Both"):
            hc = st.session_state.intraday[-360:].ffill()
            he = hc.ewm(span=20).mean()
            res_h = hc.rolling(60, min_periods=1).max()
            sup_h = hc.rolling(60, min_periods=1).min()

            xh = np.arange(len(hc))
            # use poly1d to ensure proper 1d coefficients
            p_h = np.poly1d(np.polyfit(xh, hc.values, 1))
            trend_h = p_h(xh)

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, res_h, ":", label="Resistance")
            ax2.plot(hc.index, sup_h, ":", label="Support")
            ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)
            ax2.set_xlabel("Time (PST)")
            ax2.legend(framealpha=0.5, loc="lower left")
            st.pyplot(fig2)

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tabs 2–4 unchanged (Enhanced Forecast, Bull vs Bear, Metrics) ---
# (They remain the same as in your existing code.)
