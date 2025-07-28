import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import time
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# --- Universe ---
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

# --- Data functions ---
@st.cache_data(ttl=900)
def fetch_hist(ticker):
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))
    df = df[['Open','High','Low','Close']].asfreq('D').fillna(method='ffill')
    return df.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker):
    df = yf.download(ticker, period="2d", interval="5m")
    try:
        df = df.tz_localize('UTC')
    except:
        pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def compute_sarimax(series: pd.Series):
    try:
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        m = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    f = m.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, f.predicted_mean, f.conf_int()

def compute_bb(s: pd.Series, window=20, num_sd=2):
    m = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return m - num_sd*sd, m, m + num_sd*sd

def compute_atr(df: pd.DataFrame, window=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.update(run_all=False, ticker=None)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast",
    "Enhanced Forecast",
    "Bull vs Bear",
    "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key="orig_chart")
    auto = st.session_state.run_all and sel != st.session_state.ticker

    if st.button("Run Forecast") or auto:
        dfh = fetch_hist(sel)
        dfi = fetch_intraday(sel)
        idx, vals, ci = compute_sarimax(dfh['Close'])
        st.session_state.update(
            df_hist=dfh, intraday=dfi,
            fc_idx=idx, fc_vals=vals, fc_ci=ci,
            ticker=sel, chart=chart, run_all=True
        )

    if st.session_state.run_all and st.session_state.ticker == sel:
        dfh = st.session_state.df_hist
        dfi = st.session_state.intraday
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = float(dfh['Close'].iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        # Daily interactive
        if chart in ("Daily", "Both"):
            price = dfh['Close'][-360:]
            ema200 = price.ewm(span=200).mean()
            ma30 = price.rolling(30).mean()
            lb, mb, ub = compute_bb(dfh['Close'])
            lb, ub = lb[-360:], ub[-360:]
            res = dfh['Close'].rolling(30).max()[-360:]
            sup = dfh['Close'].rolling(30).min()[-360:]
            atr = compute_atr(dfh)[-360:]

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.05
            )
            fig.add_trace(go.Scatter(x=price.index, y=price, name="Close"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ema200.index, y=ema200, name="200 EMA", line=dict(dash="dash")),1,1)
            fig.add_trace(go.Scatter(x=ma30.index, y=ma30, name="30 MA", line=dict(dash="dash")),1,1)
            fig.add_trace(go.Scatter(x=res.index, y=res, name="Resistance", line=dict(dash="dot")),1,1)
            fig.add_trace(go.Scatter(x=sup.index, y=sup, name="Support", line=dict(dash="dot")),1,1)
            fig.add_trace(go.Scatter(x=idx, y=vals, name="Forecast"),1,1)
            trend = np.polyval(np.polyfit(np.arange(len(vals)), vals.to_numpy(), 1), np.arange(len(vals)))
            fig.add_trace(go.Scatter(x=idx, y=trend, name="Trend", line=dict(dash="dash")),1,1)
            fig.add_trace(go.Scatter(x=lb.index, y=lb, name="Lower BB", line=dict(dash="dash")),1,1)
            fig.add_trace(go.Scatter(x=ub.index, y=ub, name="Upper BB", line=dict(dash="dash")),1,1)

            fig.add_trace(go.Scatter(x=atr.index, y=atr, name="ATR(14)"), row=2, col=1)

            fig.update_layout(
                height=700,
                title_text=f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Hourly interactive
        if chart in ("Hourly", "Both"):
            hc = dfi['Close'].ffill()[-576:]
            ema20 = hc.ewm(span=20).mean()
            atr5 = compute_atr(dfi)[-576:]

            fig2 = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.05
            )
            fig2.add_trace(go.Scatter(x=hc.index, y=hc, name="Price"),1,1)
            fig2.add_trace(go.Scatter(x=ema20.index, y=ema20, name="20 EMA", line=dict(dash="dash")),1,1)
            fig2.add_trace(go.Scatter(x=atr5.index, y=atr5, name="ATR(14)"),2,1)
            fig2.update_layout(
                height=600,
                title_text=f"{sel} Last 48 Hours  ↑{p_up:.1%}  ↓{p_dn:.1%}"
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.write(pd.DataFrame({"Forecast": vals, "Lower": ci.iloc[:,0], "Upper": ci.iloc[:,1]}, index=idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        dfh = st.session_state df_hist
