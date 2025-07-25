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
def fetch_hist(ticker: str) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
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
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
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
    st.info("Pick a ticker; initial run fetches live data, then cached for 15 minutes.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    auto_run = st.session_state.run_all and (sel != st.session_state.ticker)
    if st.button("Run Forecast", key="run") or auto_run:
        df_hist = fetch_hist(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel)
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": idx,
            "fc_vals": vals,
            "fc_ci": ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)
        last = df.iloc[-1]
        p_up   = np.mean(vals.values > last)
        p_down = 1 - p_up

        # Daily + forecast
        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            resistance = df.rolling(30, min_periods=1).max()
            support    = df.rolling(30, min_periods=1).min()

            title = f"{sel} Daily  ↑{p_up:.1%}  ↓{p_down:.1%}"
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(title)
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(resistance[-360:], ":", label="30 Resist")
            ax.plot(support[-360:], ":", label="30 Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        # Intraday + trend
        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope, intercept = np.polyfit(xh, hc.values, 1)
            trend = slope*xh + intercept
            resistance_h = hc.rolling(60, min_periods=1).max()
            support_h    = hc.rolling(60, min_periods=1).min()

            title_h = f"{sel} Intraday  ↑{p_up:.1%}  ↓{p_down:.1%}"
            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(title_h)
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, resistance_h, ":", label="Resist")
            ax2.plot(hc.index, support_h, ":", label="Support")
            ax2.plot(hc.index, trend, "--", label="Trend")
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
        st.info("Run Tab 1 first.")
    else:
        df     = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi    = compute_rsi(df)
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)
        last = df.iloc[-1]
        p_up   = np.mean(vals.values > last)
        p_down = 1 - p_up
        resistance = df.rolling(30, min_periods=1).max()
        support    = df.rolling(30, min_periods=1).min()

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
        if view in ("Daily","Both"):
            title = f"{st.session_state.ticker} Daily  ↑{p_up:.1%}  ↓{p_down:.1%}"
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(title)
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(resistance[-360:], ":", label="Resist")
            ax.plot(support[-360:], ":", label="Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(
                    df[-360:].max() - (df[-360:].max()-df[-360:].min())*lev,
                    df.index[-360], df.index[-1],
                    linestyles="dotted"
                )
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--"); ax2.axhline(30, linestyle="--")
            ax2.set_xlabel("Date (PST)")
            ax2.legend(); st.pyplot(fig2)

        if view in ("Intraday","Both"):
            ic = st.session_state.intraday["Close"].ffill()
            ie = ic.ewm(span=20).mean()
            xi = np.arange(len(ic))
            slope, intercept = np.polyfit(xi, ic.values, 1)
            trend = slope*xi + intercept
            resistance_h = ic.rolling(60, min_periods=1).max()
            support_h    = ic.rolling(60, min_periods=1).min()

            title_h = f"{st.session_state.ticker} Intraday  ↑{p_up:.1%}  ↓{p_down:.1%}"
            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(title_h)
            ax3.plot(ic.index, ic, label="Intraday")
            ax3.plot(ic.index, ie, "--", label="20 EMA")
            ax3.plot(ic.index, resistance_h, ":", label="Resist")
            ax3.plot(ic.index, support_h, ":", label="Support")
            ax3.plot(ic.index, trend, "--", label="Trend")
            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(14,3))
            ri = compute_rsi(ic)
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)")
            ax4.legend(); st.pyplot(fig4)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear ---
with tab3:
    st.header("Bull vs Bear Summary")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        ticker = st.session_state.ticker
        df3 = yf.download(ticker, period=bb_period)[['Close']].dropna()
        df3['PctChange'] = df3['Close'].pct_change()
        df3['Bull'] = df3['PctChange'] > 0
        bull, bear = int(df3['Bull'].sum()), int((~df3['Bull']).sum())
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
        st.info("Run Tab 1 first.")
    else:
        df_hist = fetch_hist(st.session_state.ticker)
        last = df_hist.iloc[-1]
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up   = np.mean(vals.values > last)
        p_down = 1 - p_up

        st.subheader(f"Last 3 Months  ↑{p_up:.1%}  ↓{p_down:.1%}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m   = df3m.rolling(30, min_periods=1).mean()
        resistance = df3m.rolling(30, min_periods=1).max()
        support    = df3m.rolling(30, min_periods=1).min()
        x = np.arange(len(df3m))
        slope, intercept = np.polyfit(x, df3m.values, 1)
        trend = slope*x + intercept

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m, label="Close")
        ax.plot(df3m.index, ma30_3m, label="30 MA")
        ax.plot(df3m.index, resistance, ":", label="Resistance")
        ax.plot(df3m.index, support,    ":", label="Support")
        ax.plot(df3m.index, trend, "--", label="Trend")
        ax.set_xlabel("Date (PST)")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        x0 = np.arange(len(df0))
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0*x0 + intercept0
        resistance0 = df0.rolling(30, min_periods=1).max()
        support0    = df0.rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label="Close")
        ax0.plot(df0.index, df0['MA30'], label="30 MA")
        ax0.plot(df0.index, resistance0, ":", label="Resistance")
        ax0.plot(df0.index, support0,    ":", label="Support")
        ax0.plot(df0.index, trend0, "--", label="Trend")
        ax0.set_xlabel("Date (PST)")
        ax0.legend()
        st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)
