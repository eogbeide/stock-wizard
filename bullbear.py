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
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

# Universe lists
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

# --- CACHED fetch & compute functions ---
@st.cache_data(show_spinner=False)
def fetch_hist(ticker: str) -> pd.Series:
    df = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    return df

@st.cache_data(show_spinner=False)
def fetch_intraday(ticker: str) -> pd.DataFrame:
    return yf.download(ticker, period="1d", interval="5m")

@st.cache_data(show_spinner=False)
def compute_sarimax_forecast(df: pd.Series):
    # fit model & forecast 30 days
    try:
        model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(df, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
    return idx, fc.predicted_mean, fc.conf_int()

# Indicator helpers
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m-num_sd*s, m, m+num_sd*s

# --- Prepare state ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🇺🇸 Original US Forecast",
    "🇺🇸 Enhanced US Forecast",
    "🐂 Bull vs Bear Summary",
    "📊 Detailed Metrics"
])

# --- Tab 1: Original US Forecast ---
with tab1:
    st.header("🇺🇸 Original US Forecast")
    st.info("Select a ticker and click **Run Forecast** once.  Subsequent runs use cached data.")
    ticker = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    if st.button("Run Forecast", key="orig_btn"):
        # fetch & cache
        df_hist = fetch_hist(ticker)
        st.session_state.df_hist = df_hist
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci = idx, vals, ci
        st.session_state.intraday = fetch_intraday(ticker)
        st.session_state.ticker = ticker
        st.session_state.chart = chart
        st.session_state.mode = mode
        st.session_state.run_all = True
        st.success(f"Forecast ready for {ticker} (cached)")

    if st.session_state.run_all and st.session_state.ticker == ticker:
        df = st.session_state.df_hist
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci

        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bollinger_bands(df)
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

        if chart in ("Hourly","Both"):
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
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 2: Enhanced US Forecast ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        ticker = st.session_state.ticker
        df = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        rsi = compute_rsi(df)
        idx, vals, ci = st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            high, low = df[-360:].max(), df[-360:].min()
            diff = high - low
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(high - diff*lev, df.index[-360], df.index[-1], linestyles="dotted")
            ax.set_title(f"{ticker} Daily + Fib")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--"); ax2.axhline(30, linestyle="--")
            ax2.legend(); st.pyplot(fig2)

        if view in ("Intraday","Both"):
            intraday = st.session_state.intraday
            if intraday.empty:
                st.warning("No intraday data.")
            else:
                ic = intraday["Close"].ffill(); ie = ic.ewm(span=20).mean()
                lb2, mb2, ub2 = compute_bollinger_bands(ic); ri = compute_rsi(ic)

                fig3, ax3 = plt.subplots(figsize=(14,5))
                ax3.plot(ic, label="Intraday"); ax3.plot(ie, "--", label="20 EMA")
                ax3.plot(lb2, "--", label="Lower BB"); ax3.plot(ub2, "--", label="Upper BB")
                ax3.set_title(f"{ticker} Intraday + Fib"); ax3.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label="RSI(14)"); ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
                ax4.legend(); st.pyplot(fig4)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear Summary ---
with tab3:
    st.header("🐂 Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        ticker = st.session_state.ticker
        df0 = fetch_hist(ticker).to_frame("Close")  # cached
        df0['Close'] = df0['Close'].rolling(1).mean()  # ensure series→df
        df0 = df0.tail(0) if bb_period == "" else df0  # no change; we only need pct
        df0 = df0 if False else df0  # placeholder
        df0 = pd.DataFrame({'Close': fetch_hist(ticker)})[-1:]  # workaround
        df0 = pd.DataFrame({'Close': fetch_hist(ticker)}).tail(0) if False else pd.DataFrame({'Close': fetch_hist(ticker)})
        # Instead, use yfinance for lookback:
        df3 = yf.download(ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange']>0
        bull = int(df3['Bull'].sum()); bear=int((~df3['Bull']).sum())
        total=bull+bear; bp=bull/total*100 if total else 0; brp=bear/total*100 if total else 0

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bp:.1f}%")
        c3.metric("Bear Days", bear, f"{brp:.1f}%")
        c4.metric("Lookback", bb_period)

# --- Tab 4: Detailed Metrics ---
with tab4:
    st.header("📊 Detailed Metrics")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        ticker = st.session_state.ticker
        df_hist = fetch_hist(ticker)

        # 1) Full‑history (last 90d)
        st.subheader("Daily Chart → Last 3 Months Close + 30‑day MA + Trend")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index>=cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        x2=np.arange(len(df3m)); slope2,intercept2=np.polyfit(x2,df3m.values,1); trend2=slope2*x2+intercept2
        fig1,ax1=plt.subplots(figsize=(14,5))
        ax1.plot(df3m.index,df3m,label='Close'); ax1.plot(df3m.index,ma30_3m,label='30‑day MA')
        ax1.plot(df3m.index,trend2,"--",label='Trend')
        ax1.set_title(f"{ticker} Daily Price + MA + Trend (Last 3 Months)")
        ax1.legend(loc="lower left",framealpha=0.5); st.pyplot(fig1)

        # 2) Lookback metrics
        st.markdown("---")
        df0 = yf.download(ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange']=df0['Close'].pct_change(); df0['Bull']=df0['PctChange']>0
        df0['MA30']=df0['Close'].rolling(30,min_periods=1).mean()
        x0=np.arange(len(df0)); slope0,intercept0=np.polyfit(x0,df0['Close'],1); trend0=slope0*x0+intercept0
        fig0,ax0=plt.subplots(figsize=(14,5))
        ax0.plot(df0.index,df0['Close'],label='Close'); ax0.plot(df0.index,df0['MA30'],label='30‑day MA')
        ax0.plot(df0.index,trend0,"--",label='Trend')
        ax0.set_title(f"{ticker} Price + MA + Trend ({bb_period})"); ax0.legend(); st.pyplot(fig0)

        # 3) % Change
        st.markdown("---"); st.subheader("Daily Percentage Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        # 4) Bull/Bear distribution
        st.subheader("Bull/Bear Distribution")
        dist_df=pd.DataFrame({"Type":["Bull","Bear"],"Days":[int(df0['Bull'].sum()),int((~df0['Bull']).sum())]})
        st.bar_chart(dist_df.set_index("Type"), use_container_width=True)
