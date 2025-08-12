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

# Sidebar timing info
elapsed = time.time() - st.session_state.last_refresh
next_in = max(0, REFRESH_INTERVAL - int(elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")
st.sidebar.caption(f"Next auto-refresh in: {next_in}s")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

# Status placeholder (sidebar)
status_box = st.sidebar.empty()
def set_status(running: bool, msg: str = ""):
    if running:
        status_box.info(f"🟡 Running… {msg}")
    else:
        status_box.success("🟢 Idle")

set_status(False)

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

# --- Data & model helpers (no caching) ---
def fetch_hist(ticker: str) -> pd.Series:
    series = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    return series.tz_localize(PACIFIC)

def fetch_intraday(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1d", interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

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
    except Exception:
        model = SARIMAX(series, order=(1,1,0)).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

def compute_rsi(data, window=14):
    d    = data.diff()
    gain = d.where(d > 0, 0).rolling(window).mean()
    loss = -d.where(d < 0, 0).rolling(window).mean()
    rs   = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

# --- Session state init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None

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
    st.info("Pick a ticker; data fetches fresh on every run.")

    selected = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart    = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    auto_run = st.session_state.run_all and (selected != st.session_state.ticker)

    run_clicked = st.button("Run Forecast", key="run")

    if run_clicked or auto_run:
        t0 = time.perf_counter()
        prog = st.sidebar.progress(0, text="Starting…")
        set_status(True, "initializing")
        with st.spinner("⏳ Fetching data & running forecast…"):
            # Step 1: daily history
            prog.progress(10, text="Fetching daily history…")
            df_hist = fetch_hist(selected)

            # Step 2: model
            prog.progress(55, text="Computing SARIMAX forecast…")
            idx, vals, ci = compute_sarimax_forecast(df_hist)

            # Step 3: intraday
            prog.progress(85, text="Fetching intraday (5m)…")
            intraday = fetch_intraday(selected)

        # Save results
        st.session_state.df_hist  = df_hist
        st.session_state.fc_idx   = idx
        st.session_state.fc_vals  = vals
        st.session_state.fc_ci    = ci
        st.session_state.intraday = intraday
        st.session_state.ticker   = selected
        st.session_state.chart    = chart
        st.session_state.run_all  = True
        st.session_state.last_refresh = time.time()

        runtime = time.perf_counter() - t0
        prog.progress(100, text=f"Done in {runtime:.2f}s")
        time.sleep(0.15)
        prog.empty()
        set_status(False)
        st.toast(f"✅ Run finished in {runtime:.2f}s", icon="✅")

    if st.session_state.run_all and st.session_state.ticker == selected:
        df, idx, vals, ci = (
            st.session_state.df_hist,
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )

        # --- Hourly FIRST: inversions + mid-trend line + crossing markers ---
        if chart in ("Hourly","Both"):
            hc = st.session_state.intraday["Close"].ffill()
            if len(hc) >= 6:
                # Linear trend (regression)
                xh = np.arange(len(hc))
                slope, intercept = np.polyfit(xh, hc.values, 1)
                trend = slope * xh + intercept

                # Mid-trend line (horizontal at median of trend values)
                mid_val  = float(np.median(trend))

                # Where the trend line crosses the mid-trend line
                diff_sign = np.sign(trend - mid_val)
                cross_idx = np.where(np.diff(diff_sign) != 0)[0] + 1
                cross_times  = hc.index[cross_idx] if len(cross_idx) else []
                cross_prices = hc.iloc[cross_idx] if len(cross_idx) else pd.Series([], dtype=float)

                # Trend inversion (rolling slope sign changes)
                rolling_slope = hc.rolling(12).apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
                    raw=True
                ).dropna()
                rolling_sign = np.sign(rolling_slope)
                sign_change  = (rolling_sign.diff().fillna(0) != 0)
                if not sign_change.empty:
                    sign_change.iloc[0] = False
                inv_times  = sign_change[sign_change].index
                inv_values = hc.reindex(inv_times)

                ema20 = hc.ewm(span=20).mean()

                fig2, ax2 = plt.subplots(figsize=(14,4))
                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, trend, "--", label="Trend")
                ax2.plot(hc.index, ema20, "--", label="20 EMA")
                ax2.hlines(mid_val, xmin=hc.index[0], xmax=hc.index[-1],
                           linestyles="dashdot", label="Mid-Trend Line")
                if len(cross_idx):
                    ax2.scatter(cross_times, cross_prices, marker="x", s=70,
                                label="Trend ↔ Mid-line Cross")
                    for ct in cross_times:
                        ax2.axvline(ct, linestyle=":", alpha=0.35)
