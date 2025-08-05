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
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 {transform:none!important;visibility:visible!important;
                  width:100%!important;position:relative!important;
                  margin-bottom:1rem;}
    .css-1v3fvcr {margin-left:0!important;}
  }
</style>
""", unsafe_allow_html=True)

# --- Utility ---
def safe_trend(x: np.ndarray, y: np.ndarray):
    try:
        coeff = np.polyfit(x, y, 1)
        return coeff[0] * x + coeff[1], coeff
    except:
        m = np.nanmean(y)
        return np.full_like(x, m, dtype=float), (0.0, m)

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120
PACIFIC = pytz.timezone("US/Pacific")
def auto_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    elif time.time() - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        try: st.experimental_rerun()
        except: pass

auto_refresh()
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo","3mo","6mo","1y"], index=2)
if mode=="Stock":
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
    s = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
    return s.asfreq("D").fillna(method="ffill").tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker: str, period="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try: df = df.tz_localize("UTC")
    except: pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def compute_sarimax_forecast(series: pd.Series):
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1]+timedelta(1), periods=30,
                        freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicators ---
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m-num_sd*s, m, m+num_sd*s

def compute_macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

# --- Session defaults ---
st.session_state.setdefault("run_all", False)
st.session_state.setdefault("ticker", None)
st.session_state.setdefault("hour_range", "24h")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast","Enhanced Forecast","Bull vs Bear","Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data caches for 15 minutes.")

    sel        = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart      = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:",
                              ["24h","48h","72h","96h","120h"],
                              key="hour_range_select")
    auto_run   = st.session_state.run_all and sel!=st.session_state.ticker

    if st.button("Run Forecast") or auto_run or not st.session_state.run_all:
        df_hist = fetch_hist(sel)
        days    = {"24h":"1d","48h":"2d","72h":"3d","96h":"4d","120h":"5d"}[hour_range]
        intr    = fetch_intraday(sel, period=days)
        idx, vals, ci = compute_sarimax_forecast(df_hist)

        st.session_state.update({
            "df_hist":    df_hist,
            "intraday":   intr,
            "fc_idx":     idx,
            "fc_vals":    vals,
            "fc_ci":      ci,
            "ticker":     sel,
            "chart":      chart,
            "hour_range": hour_range,
            "run_all":    True
        })

    if st.session_state.run_all and st.session_state.ticker==sel:
        df   = st.session_state.df_hist
        intr = st.session_state.intraday
        idx, vals, ci = (st.session_state.fc_idx,
                         st.session_state.fc_vals,
                         st.session_state.fc_ci)

        last     = float(df.iloc[-1])
        p_up     = np.mean(vals>last)
        p_dn     = 1-p_up
        mean_fc  = float(vals.mean())
        pct_trnd = ((mean_fc-last)/last*100) if last else 0.0
        tlb      = f"+{pct_trnd:.2f}%" if pct_trnd>=0 else f"{pct_trnd:.2f}%"

        # — Intraday w/ Price Action & Std Dev —
        if chart in ("Hourly","Both"):
            hc = intr["Close"].ffill()
            he = hc.ewm(span=20).mean()
            lo = intr["Low"].ffill().values
            hi = intr["High"].ffill().values
            xh = np.arange(len(hc))
            trend_h, coeff_h = safe_trend(xh, hc.values)

            # explicit float conversion for base
            try:
                base = float(hc.iloc[0])
                if base==0: base=1.0
            except:
                base = 1.0
            slope_pct = coeff_h[0]*(len(hc)-1)/base*100

            window = int(st.session_state.hour_range[:-1]) * 12
            var_h  = hc.rolling(window, min_periods=1).var().values
            std_h  = hc.rolling(window, min_periods=1).std().values

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(
              f"{sel} Intraday ({st.session_state.hour_range})  ↑{p_up:.1%}"
              f"  ↓{p_dn:.1%}  Trend: {slope_pct:.2f}%"
            )
            ax2.fill_between(    xh, lo, hi,   color="silver", alpha=0.3, label="Price Action")
            ax2.plot(    xh, hc.values,      label="Close")
            ax2.plot(    xh, he.values,  "--", label="20 EMA")
            ax2.plot(    xh, trend_h,    "--", label="Trend")
            ax2.plot(    xh, var_h,     "-.",  label="Variance")
            ax2.plot(    xh, std_h,      ":",  label="Std Dev")

            # label by timestamp
            ticks  = xh[::12]
            labels = hc.index.strftime("%m-%d %H:%M")[::12]
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(labels, rotation=45, ha="right")

            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # — Daily + MACD —
        if chart in ("Daily","Both"):
            ema200 = df.ewm(span=200).mean()
            ma30   = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            res     = df.rolling(30, min_periods=1).max()
            sup     = df.rolling(30, min_periods=1).min()
            xfc     = np.arange(len(vals))
            tr_fc,_ = safe_trend(xfc, vals.values)
            macd_l, sig_l, hist = compute_macd(df)
            h_arr   = pd.Series(hist).fillna(0).values

            fig, (ax0, ax1) = plt.subplots(2,1, figsize=(14,8))
            ax0.set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}  Trend: {tlb}")
            ax0.plot(df[-360:],               label="History")
            ax0.plot(ema200[-360:], "--",     label="200 EMA")
            ax0.plot(ma30[-360:], "--",       label="30 MA")
            ax0.plot(res[-360:],  ":",        label="Resistance")
            ax0.plot(sup[-360:],  ":",        label="Support")
            ax0.plot(idx,           vals,     label="Forecast")
            ax0.plot(idx,           tr_fc,    "--", label="Forecast Trend")
            ax0.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax0.plot(lb[-360:], "--", label="Lower BB")
            ax0.plot(ub[-360:], "--", label="Upper BB")
            ax0.set_xlabel("Date (PST)")
            ax0.legend(loc="lower left", framealpha=0.5)

            ax1.plot(df.index, macd_l,        label="MACD Line")
            ax1.plot(df.index, sig_l,   "--",  label="Signal Line")
            ax1.bar(df.index, h_arr, alpha=0.5, label="Histogram")
            ax1.axhline(0, color="black", linewidth=0.5)
            ax1.set_ylabel("MACD"); ax1.set_xlabel("Date (PST)")
            ax1.legend(loc="lower left", framealpha=0.5)

            st.pyplot(fig)

        # Forecast table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# Tab 2, Tab 3 & Tab 4 remain unchanged (Enhanced Forecast, Bull vs Bear, Metrics).
