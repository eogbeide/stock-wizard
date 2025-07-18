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
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="global_mode")

# universe for multi-select
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

tickers = st.sidebar.multiselect(
    "Choose up to two tickers:",
    universe,
    default=universe[:2],
    max_selections=2,
    key="multi_tickers"
)

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

# --- Prepare state ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.data = {}
    st.session_state.intraday = {}
    st.session_state.forecasts = {}

# --- Helpers ---
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d > 0, 0).rolling(window).mean()
    loss = -d.where(d < 0, 0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def safe_sarimax(endog, order, seasonal_order):
    try:
        return SARIMAX(endog, order=order, seasonal_order=seasonal_order).fit(disp=False)
    except np.linalg.LinAlgError:
        return SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🇺🇸 Original US Forecast",
    "🇺🇸 Enhanced US Forecast",
    "🐂 Bull vs Bear Summary",
    "📊 Detailed Metrics"
])

# --- Tab 1: Original US Forecast (two-up) ---
with tab1:
    st.header("🇺🇸 Original US Forecast (up to 2 tickers)")
    if not tickers:
        st.warning("Select at least one ticker in the sidebar.")
    else:
        if st.button("Run Forecast for Selected", key="run_multi"):
            st.session_state.data.clear()
            st.session_state.intraday.clear()
            st.session_state.forecasts.clear()
            prog = st.progress(0)
            for i, tk in enumerate(tickers):
                df_hist = (
                    yf.download(tk, start="2018-01-01", end=pd.to_datetime("today"))['Close']
                    .asfreq("D")
                    .fillna(method="ffill")
                )
                st.session_state.data[tk] = df_hist

                model = safe_sarimax(df_hist, (1,1,1), (1,1,1,12))
                fc = model.get_forecast(steps=30)
                idx = pd.date_range(df_hist.index[-1] + timedelta(1), periods=30, freq="D")
                st.session_state.forecasts[tk] = (idx, fc.predicted_mean, fc.conf_int())

                st.session_state.intraday[tk] = yf.download(tk, period="1d", interval="5m")
                prog.progress(int((i+1)/len(tickers)*100))

            st.session_state.run_all = True
            st.success("Forecasts complete!")

        if st.session_state.run_all:
            cols = st.columns(len(tickers))
            for col, tk in zip(cols, tickers):
                with col:
                    df = st.session_state.data[tk]
                    ema200 = df.ewm(span=200).mean()
                    ma30   = df.rolling(30).mean()
                    lb, mb, ub = compute_bollinger_bands(df)
                    idx, vals, ci = st.session_state.forecasts[tk]

                    st.subheader(f"{tk} — Daily Forecast")
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(df[-360:], label="History")
                    ax.plot(ema200[-360:], "--", label="200 EMA")
                    ax.plot(ma30[-360:], "--", label="30 MA")
                    ax.plot(idx, vals, label="Forecast")
                    ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
                    ax.plot(lb[-360:], "--", label="Lower BB")
                    ax.plot(ub[-360:], "--", label="Upper BB")
                    ax.legend(loc="lower left", framealpha=0.3)
                    st.pyplot(fig)

# --- Tab 2: Enhanced US Forecast (first ticker) ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if not st.session_state.run_all or not tickers:
        st.info("Run a forecast in Tab 1 first.")
    else:
        tk = tickers[0]
        df = st.session_state.data[tk]
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        rsi = compute_rsi(df)
        idx, vals, ci = st.session_state.forecasts[tk]

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
            for lev in (0.236, 0.382, 0.5, 0.618):
                ax.hlines(high - diff*lev, df.index[-360], df.index[-1], linestyles="dotted")
            ax.set_title(f"{tk} Daily + Fib")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--")
            ax2.axhline(30, linestyle="--")
            ax2.legend()
            st.pyplot(fig2)

        if view in ("Intraday","Both"):
            intraday = st.session_state.intraday[tk]
            if intraday.empty:
                st.warning("No intraday data available.")
            else:
                ic = intraday["Close"].ffill()
                ie = ic.ewm(span=20).mean()
                lb2, mb2, ub2 = compute_bollinger_bands(ic)
                ri = compute_rsi(ic)

                fig3, ax3 = plt.subplots(figsize=(14,5))
                ax3.plot(ic, label="Intraday")
                ax3.plot(ie, "--", label="20 EMA")
                ax3.plot(lb2, "--", label="Lower BB")
                ax3.plot(ub2, "--", label="Upper BB")
                ax3.set_title(f"{tk} Intraday + Fib")
                ax3.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label="RSI(14)")
                ax4.axhline(70, linestyle="--")
                ax4.axhline(30, linestyle="--")
                ax4.legend()
                st.pyplot(fig4)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear Summary (first ticker) ---
with tab3:
    st.header("🐂 Bull vs Bear Summary")
    if not st.session_state.run_all or not tickers:
        st.info("Run a forecast in Tab 1 first.")
    else:
        tk = tickers[0]
        df0 = yf.download(tk, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        bull = int(df0['Bull'].sum())
        bear = int((~df0['Bull']).sum())
        total = bull + bear
        bp = bull / total * 100 if total else 0
        brp = bear / total * 100 if total else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bp:.1f}%")
        c3.metric("Bear Days", bear, f"{brp:.1f}%")
        c4.metric("Lookback", bb_period)

        st.markdown("---")
        st.write(f"Data for **{tk}** over the past **{bb_period}**")

# --- Tab 4: Detailed Metrics (first ticker) ---
with tab4:
    st.header("📊 Detailed Metrics")
    if not st.session_state.run_all or not tickers:
        st.info("Run a forecast in Tab 1 first.")
    else:
        tk = tickers[0]
        # 1) Full‑history chart (last 3 months)
        st.subheader("Daily Chart → Last 3 Months Close + 30‑day MA + Trend")
        df_hist = st.session_state.data[tk]
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(window=30, min_periods=1).mean()
        x2 = np.arange(len(df3m))
        slope2, intercept2 = np.polyfit(x2, df3m.values, 1)
        trend2 = slope2 * x2 + intercept2

        fig1, ax1 = plt.subplots(figsize=(14,5))
        ax1.plot(df3m.index, df3m, label='Close')
        ax1.plot(df3m.index, ma30_3m, label='30‑day MA')
        ax1.plot(df3m.index, trend2, "--", label='Trend')
        ax1.set_title(f"{tk} Daily Price + MA + Trend (Last 3 Months)")
        ax1.legend(loc="lower left", framealpha=0.5)
        st.pyplot(fig1)

        # 2) Lookback‑period metrics
        st.markdown("---")
        df0 = yf.download(tk, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        df0['MA30']      = df0['Close'].rolling(window=30, min_periods=1).mean()

        st.subheader("Price Chart → Close + 30‑day MA + Trend")
        x0 = np.arange(len(df0))
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0 * x0 + intercept0

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'], label='Close')
        ax0.plot(df0.index, df0['MA30'], label='30‑day MA')
        ax0.plot(df0.index, trend0, "--", label='Trend')
        ax0.set_title(f"{tk} Price + MA + Trend ({bb_period} lookback)")
        ax0.legend()
        st.pyplot(fig0)

        # 3) Daily Percentage Change
        st.markdown("---")
        st.subheader("Daily Percentage Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        # 4) Bull/Bear Distribution
        st.subheader("Bull/Bear Distribution")
        dist_df = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        })
        st.bar_chart(dist_df.set_index("Type"), use_container_width=True)
