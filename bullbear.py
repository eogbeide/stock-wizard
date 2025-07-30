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

  /* on small screens, override Streamlit's default so sidebar stays visible */
  @media (max-width: 600px) {
    .css-18e3th9 {  /* sidebar */
      transform: none !important;
      visibility: visible !important;
      width: 100% !important;
      position: relative !important;
      margin-bottom: 1rem;
    }
    .css-1v3fvcr {  /* main content */
      margin-left: 0 !important;
    }
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
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo","3mo","6mo","1y"], index=2)

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

# --- Caching helpers ---
@st.cache_data(ttl=900)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    return s.tz_localize(PACIFIC)

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
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# Indicator helpers
def compute_rsi(data, window=14):
    d    = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs   = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def compute_momentum(data, window=10):
    return data - data.shift(window)

# Session state init
if 'run_all' not in st.session_state:
    st.session_state.update({
        'run_all': False,
        'ticker': None
    })

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast", "Enhanced Forecast", "Bull vs Bear", "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 15 minutes after first fetch.")

    sel   = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    auto_run = st.session_state.run_all and (sel != st.session_state.ticker)

    if st.button("Run Forecast") or auto_run:
        df_hist = fetch_hist(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel)
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": idx, "fc_vals": vals, "fc_ci": ci,
            "intraday": intraday, "ticker": sel,
            "chart": chart, "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df    = st.session_state.df_hist
        idx, vals, ci = (st.session_state.fc_idx,
                        st.session_state.fc_vals,
                        st.session_state.fc_ci)
        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        # Daily
        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            res = df.rolling(30, min_periods=1).max()
            sup = df.rolling(30, min_periods=1).min()

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="30 Resistance")
            ax.plot(sup[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.plot(idx, np.poly1d(np.polyfit(range(len(vals)), vals, 1))(range(len(vals))),
                    "--", label="Forecast Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # Intraday
        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            he = hc.ewm(span=20).mean()
            res_h = hc.rolling(60, min_periods=1).max()
            sup_h = hc.rolling(60, min_periods=1).min()

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, res_h, ":", label="Resistance")
            ax2.plot(hc.index, sup_h, ":", label="Support")
            ax2.plot(hc.index,
                     np.poly1d(np.polyfit(range(len(hc)), hc.values, 1))(range(len(hc))),
                     "--", label="Trend", linewidth=2)
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df      = st.session_state.df_hist
        ema200  = df.ewm(span=200).mean()
        ma30    = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi     = compute_rsi(df)
        mom     = compute_momentum(df, window=10)
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)
        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        # Daily + Indicators
        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily with Forecast")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            # RSI panel
            fig_rsi, ax_rsi = plt.subplots(figsize=(14,3))
            ax_rsi.plot(rsi[-360:], label="RSI(14)")
            ax_rsi.axhline(70, linestyle="--")
            ax_rsi.axhline(30, linestyle="--")
            ax_rsi.set_xlabel("Date (PST)")
            ax_rsi.legend()
            st.pyplot(fig_rsi)

            # Momentum panel
            fig_mom, ax_mom = plt.subplots(figsize=(14,3))
            ax_mom.plot(mom[-360:], label="Momentum(10)")
            ax_mom.axhline(0, linestyle="--")
            ax_mom.set_xlabel("Date (PST)")
            ax_mom.legend()
            st.pyplot(fig_mom)

        # Intraday + Indicators
        if view in ("Intraday","Both"):
            ic  = st.session_state.intraday["Close"].ffill()
            ie  = ic.ewm(span=20).mean()
            mom_i = compute_momentum(ic, window=10)

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(f"{st.session_state.ticker} Intraday")
            ax3.plot(ic.index, ic, label="Intraday")
            ax3.plot(ic.index, ie, "--", label="20 EMA")
            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

            # RSI panel
            ri = compute_rsi(ic)
            fig4, ax4 = plt.subplots(figsize=(14,3))
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--")
            ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)")
            ax4.legend()
            st.pyplot(fig4)

            # Momentum panel
            fig5, ax5 = plt.subplots(figsize=(14,3))
            ax5.plot(mom_i, label="Momentum(10)")
            ax5.axhline(0, linestyle="--")
            ax5.set_xlabel("Time (PST)")
            ax5.legend()
            st.pyplot(fig5)

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df3 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull']      = df3['PctChange'] > 0
        bull = int(df3['Bull'].sum())
        bear = int((~df3['Bull']).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull/total*100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear/total*100:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Metrics ---
with tab4:
    st.header("Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last_price = float(df_hist.iloc[-1])
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up

        st.subheader(f"Last 3 Months  ↑{p_up:.1%}  ↓{p_dn:.1%}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m   = df3m.rolling(30, min_periods=1).max()
        sup3m   = df3m.rolling(30, min_periods=1).min()
        x3m     = np.arange(len(df3m))
        trend3m = np.poly1d(np.polyfit(x3m, df3m.values, 1))(x3m)

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(df3m.index, res3m, ":", label="Resistance")
        ax.plot(df3m.index, sup3m, ":", label="Support")
        ax.plot(df3m.index, trend3m, "--", label="Trend")
        ax.set_xlabel("Date (PST)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        df0['MA30']      = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30‑day MA + Trend")
        x0     = np.arange(len(df0))
        trend0 = np.poly1d(np.polyfit(x0, df0['Close'], 1))(x0)
        res0   = df0['Close'].rolling(30, min_periods=1).max()
        sup0   = df0['Close'].rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(df0.index, res0, ":", label="Resistance")
        ax0.plot(df0.index, sup0, ":", label="Support")
        ax0.plot(df0.index, trend0, "--", label="Trend")
        ax0.set_xlabel("Date (PST)")
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull","Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)
