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

# --- Auto‐refresh logic ---
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

# Changed: ticker selector for Bull vs Bear moved to dropdown
bb_tickers = sorted([
    'SPY','AAPL','MSFT','GOOGL','AMZN','TSLA','NVDA','META','JPM','V',
    'JNJ','WMT','PG','UNH','MA','DIS','BAC','KO','XOM','BRK-B'
])
symbol_bb = st.sidebar.selectbox("Bull / Bear Ticker:", bb_tickers, index=bb_tickers.index('SPY'))
period_bb = st.sidebar.selectbox("Bull / Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="global_mode")

# --- Bull/Bear data loader ---
@st.cache_data
def load_bullbear(sym, per):
    df0 = yf.download(sym, period=per)[['Close']].dropna()
    df0['PctChange'] = df0['Close'].pct_change()
    df0['Bull'] = df0['PctChange'] > 0
    return df0

df_bb = load_bullbear(symbol_bb, period_bb)
bull_days = int(df_bb['Bull'].sum())
bear_days = int((~df_bb['Bull']).sum())
total_days = bull_days + bear_days
bull_pct = bull_days / total_days * 100 if total_days else 0
bear_pct = bear_days / total_days * 100 if total_days else 0

# --- Indicator & forecasting helpers ---
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
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
        return SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

# --- Define four tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🇺🇸 Original US Forecast",
    "🇺🇸 Enhanced US Forecast",
    "🐂 Bull vs Bear Summary",
    "📊 Detailed Metrics"
])

# --- Tab 1: Original US Forecast ---
with tab1:
    st.header("🇺🇸 Original US Forecast")
    if mode == "Stock":
        ticker = st.selectbox(
            "Stock Ticker:",
            sorted(['AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                    'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                    'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI']),
            key="orig_stock_ticker"
        )
    else:
        ticker = st.selectbox(
            "Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'],
            key="orig_forex_pair"
        )
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    if st.button("Run Forecast", key="orig_btn"):
        df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
               .asfreq("D").fillna(method="ffill")
        ema200 = df.ewm(span=200).mean()
        ma30 = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        model = safe_sarimax(df, (1,1,1), (1,1,1,12))
        fc = model.get_forecast(steps=30)
        idx = pd.date_range(df.index[-1]+timedelta(1), periods=30, freq="D")
        vals, ci = fc.predicted_mean, fc.conf_int()

        if chart in ("Daily","Both"):
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
            intraday = yf.download(ticker, period="1d", interval="5m")
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

# --- Tab 2: Enhanced US Forecast ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if mode == "Stock":
        ticker = st.selectbox(
            "Stock Ticker:",
            sorted(['AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                    'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                    'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI']),
            key="enh_stock_ticker"
        )
    else:
        ticker = st.selectbox(
            "Forex Pair:",
            ['EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
             'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
             'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'],
            key="enh_forex_pair"
        )
    view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
    if st.button("Run Enhanced Forecast", key="enh_btn"):
        series = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                    .asfreq("D").fillna(method="ffill")
        ema200 = series.ewm(span=200).mean()
        ma30 = series.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(series)
        rsi = compute_rsi(series)
        model = safe_sarimax(series, (1,1,1), (1,1,1,12))
        fc = model.get_forecast(steps=30)
        idx, vals, ci = (
            pd.date_range(series.index[-1]+timedelta(1), periods=30, freq="D"),
            fc.predicted_mean,
            fc.conf_int()
        )

        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(series[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            hi, lo = series[-360:].max(), series[-360:].min()
            diff = hi - lo
            for lvl in (0.236,0.382,0.5,0.618):
                ax.hlines(hi - diff*lvl, series.index[-360], series.index[-1], linestyles="dotted")
            ax.set_title(f"{ticker} Daily + Fib")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig_rsi, ax_rsi = plt.subplots(figsize=(14,2))
            ax_rsi.plot(rsi[-360:], label="RSI(14)")
            ax_rsi.axhline(70, linestyle="--")
            ax_rsi.axhline(30, linestyle="--")
            ax_rsi.set_title("RSI (14)")
            ax_rsi.legend(loc="lower left")
            st.pyplot(fig_rsi)

        if view in ("Intraday","Both"):
            intraday = yf.download(ticker, period="1d", interval="5m")
            if intraday.empty:
                st.warning("No intraday data.")
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
                ax3.set_title(f"{ticker} Intraday + Fib")
                ax3.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig3)

                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label="RSI(14)")
                ax4.axhline(70, linestyle="--")
                ax4.axhline(30, linestyle="--")
                ax4.set_title("Intraday RSI (14)")
                ax4.legend(loc="lower left")
                st.pyplot(fig4)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Bull vs Bear Summary ---
with tab3:
    st.header("🐂 Bull vs Bear Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Days", total_days)
    c2.metric("Bull Days", bull_days, f"{bull_pct:.1f}%")
    c3.metric("Bear Days", bear_days, f"{bear_pct:.1f}%")
    c4.metric("Period", period_bb)
    st.markdown("---")
    st.write(f"Data for **{symbol_bb}** over the past **{period_bb}** as of {df_bb.index[-1].date()}")

# --- Tab 4: Detailed Metrics ---
with tab4:
    st.header("📊 Detailed Metrics")
    st.subheader("Price Chart")
    st.line_chart(df_bb['Close'], use_container_width=True)

    st.subheader("Daily Percentage Change")
    st.line_chart(df_bb['PctChange'], use_container_width=True)

    st.subheader("Bull/Bear Distribution")
    dist_df = pd.DataFrame({"Type":["Bull","Bear"], "Days":[bull_days,bear_days]})
    st.bar_chart(dist_df.set_index("Type"), use_container_width=True)
