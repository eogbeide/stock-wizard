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
  /* mobile override */
  @media (max-width: 600px) {
    .css-18e3th9 {transform:none!important;visibility:visible!important;width:100%!important;position:relative!important;margin-bottom:1rem;}
    .css-1v3fvcr {margin-left:0!important;}
  }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh logic ---
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
now_pst = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {now_pst:%Y-%m-%d %H:%M:%S} PST")

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode      = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2)

# --- Universe ---
if mode == "Stock":
    universe = sorted([...])  # same list as before
else:
    universe = [...]          # same FX list

# --- Data fetching & caching ---
@st.cache_data(ttl=900)
def fetch_hist(ticker):
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
         .asfreq("D").fillna(method="ffill"))
    return s.tz_localize(PACIFIC)

@st.cache_data(ttl=900)
def fetch_intraday(ticker):
    df = yf.download(ticker, period="1d", interval="5m")
    try: df = df.tz_localize('UTC')
    except: pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=900)
def sarimax_forecast(series):
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc  = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicator helpers ---
def compute_rsi(s, window=14):
    d    = s.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs   = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(s, window=20, num_sd=2):
    m = s.rolling(window).mean()
    sd= s.rolling(window).std()
    return m - num_sd*sd, m, m + num_sd*sd

# --- Session init ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker  = None

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast", "Enhanced Forecast", "Bull vs Bear", "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Select a ticker and click **Run Forecast**")
    sel   = st.selectbox("Ticker:", universe, key="orig_ticker")
    view  = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")
    auto  = st.session_state.run_all and sel!=st.session_state.ticker

    if st.button("Run Forecast") or auto:
        hist = fetch_hist(sel)
        idx, vals, ci = sarimax_forecast(hist)
        intr = fetch_intraday(sel)
        st.session_state.update({
            "df_hist": hist, "fc_idx": idx, "fc_vals": vals,
            "fc_ci": ci, "intraday": intr,
            "ticker": sel, "chart": view, "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker==sel:
        df, idx, vals, ci = (
            st.session_state.df_hist,
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last = float(df.iloc[-1])
        up_p = np.mean(vals>last)
        dn_p = 1-up_p

        # Daily
        if view in ("Daily","Both"):
            recent = df[-360:]
            ema200, ma30 = recent.ewm(span=200).mean(), recent.rolling(30).mean()
            lb, mb, ub  = compute_bb(recent)
            res, sup    = recent.rolling(30,min_periods=1).max(), recent.rolling(30,min_periods=1).min()
            x = np.arange(len(vals))
            y = vals.to_numpy().flatten()
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope*x + intercept

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily   ↑{up_p:.1%}  ↓{dn_p:.1%}")
            ax.plot(recent, label="History")
            ax.plot(ema200, "--", label="200 EMA")
            ax.plot(ma30, "--", label="30 MA")
            ax.plot(res, ":", label="Resistance")
            ax.plot(sup, ":", label="Support")
            ax.plot(idx, vals, label="Forecast")
            ax.plot(idx, trend, "--", label="Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb, "--", label="Lower BB")
            ax.plot(ub, "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(framealpha=0.5, loc="lower left")
            st.pyplot(fig)

        # Hourly
        if view in ("Hourly","Both"):
            rng = st.selectbox("Intraday Range:", ["Last 24 Hours","Last 48 Hours"], key="hourly_range")
            hrs = 24 if rng=="Last 24 Hours" else 48

            intr = st.session_state.intraday['Close'].ffill()
            cutoff = intr.index.max() - pd.Timedelta(hours=hrs)
            hc = intr[intr.index>=cutoff]
            he = hc.ewm(span=20).mean()
            res_h = hc.rolling(60,min_periods=1).max()
            sup_h = hc.rolling(60,min_periods=1).min()

            xh = np.arange(len(hc))
            yh = hc.values.flatten()
            slope_h, intercept_h = np.polyfit(xh, yh, 1)
            trend_h = slope_h*xh + intercept_h

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday ({rng})   ↑{up_p:.1%}  ↓{dn_p:.1%}")
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
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df360 = st.session_state.df_hist[-360:]
        ema200, ma30 = df360.ewm(span=200).mean(), df360.rolling(30).mean()
        lb, mb, ub   = compute_bb(df360)
        rsi = compute_rsi(df360)
        mom = df360.diff(10)
        idx, vals, ci = (st.session_state.fc_idx, st.session_state.fc_vals, st.session_state.fc_ci)
        last = float(st.session_state.df_hist.iloc[-1])
        up_p, dn_p = np.mean(vals>last), 1-np.mean(vals>last)

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        if view in ("Daily","Both"):
            x = np.arange(len(vals))
            y = vals.to_numpy().flatten()
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope*x + intercept

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily + Forecast")
            ax.plot(df360,       label="History")
            ax.plot(ema200,      "--", label="200 EMA")
            ax.plot(ma30,        "--", label="30 MA")
            ax.plot(idx, vals,   label="Forecast")
            ax.plot(idx, trend,  "--", label="Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.set_xlabel("Date (PST)")
            ax.legend(framealpha=0.5, loc="lower left")
            st.pyplot(fig)

            fig_r, ax_r = plt.subplots(figsize=(14,3))
            ax_r.plot(rsi, label="RSI(14)")
            ax_r.axhline(70, linestyle="--"); ax_r.axhline(30, linestyle="--")
            ax_r.set_xlabel("Date (PST)"); ax_r.legend()
            st.pyplot(fig_r)

            fig_m, ax_m = plt.subplots(figsize=(14,3))
            ax_m.plot(mom, label="Momentum(10)")
            ax_m.axhline(0, linestyle="--")
            ax_m.set_xlabel("Date (PST)"); ax_m.legend()
            st.pyplot(fig_m)

        if view in ("Intraday","Both"):
            ic = st.session_state.intraday['Close'].ffill()[-360:]
            ie = ic.ewm(span=20).mean()
            xi, yi = np.arange(len(ic)), ic.values.flatten()
            slope_i, intercept_i = np.polyfit(xi, yi, 1)
            trend_i = slope_i*xi + intercept_i

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(f"{st.session_state.ticker} Intraday")
            ax3.plot(ic.index, ic, label="Intraday")
            ax3.plot(ic.index, ie, "--", label="20 EMA")
            ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)
            ax3.set_xlabel("Time (PST)"); ax3.legend(framealpha=0.5, loc="lower left")
            st.pyplot(fig3)

            ri = compute_rsi(ic)
            fig4, ax4 = plt.subplots(figsize=(14,3))
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)"); ax4.legend()
            st.pyplot(fig4)

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
        df3['Bull'] = df3['PctChange'] > 0
        bull, bear = int(df3['Bull'].sum()), int((~df3['Bull']).sum())
        total      = bull + bear
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
        last     = float(df_hist.iloc[-1])
        idx, vals, ci = sarimax_forecast(df_hist)
        up_p, dn_p    = np.mean(vals>last), 1-np.mean(vals>last)

        st.subheader(f"Last 3 Months   ↑{up_p:.1%}  ↓{dn_p:.1%}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m   = df_hist[df_hist.index >= cutoff]

        ma30m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()

        x3m = np.arange(len(df3m))
        y3m = df3m.values.flatten()
        slope3, intercept3 = np.polyfit(x3m, y3m, 1)
        trend3 = slope3*x3m + intercept3

        fig, ax = plt.subplots(figsize=(14,5))
        ax.plot(df3m.index, df3m,        label="Close")
        ax.plot(df3m.index, ma30m,       label="30 MA")
        ax.plot(df3m.index, res3m, ":",  label="Resistance")
        ax.plot(df3m.index, sup3m, ":",  label="Support")
        ax.plot(df3m.index, trend3, "--",label="Trend")
        ax.set_xlabel("Date (PST)"); ax.legend(); st.pyplot(fig)

        st.markdown("---")
        df0 = yf.download(st.session_state.ticker, period=bb_period)[['Close']].dropna()
        df0['PctChange'] = df0['Close'].pct_change()
        df0['Bull']      = df0['PctChange'] > 0
        df0['MA30']      = df0['Close'].rolling(30, min_periods=1).mean()

        x0 = np.arange(len(df0))
        y0 = df0['Close'].values.flatten()
        slope0, intercept0 = np.polyfit(x0, y0, 1)
        trend0 = slope0*x0 + intercept0
        res0, sup0 = df0['Close'].rolling(30, min_periods=1).max(), df0['Close'].rolling(30, min_periods=1).min()

        fig0, ax0 = plt.subplots(figsize=(14,5))
        ax0.plot(df0.index, df0['Close'],      label="Close")
        ax0.plot(df0.index, df0['MA30'],       label="30 MA")
        ax0.plot(df0.index, res0, ":",         label="Resistance")
        ax0.plot(df0.index, sup0, ":",         label="Support")
        ax0.plot(df0.index, trend0, "--",      label="Trend")
        ax0.set_xlabel("Date (PST)"); ax0.legend(); st.pyplot(fig0)

        st.markdown("---")
        st.subheader("Daily % Change")
        st.line_chart(df0['PctChange'], use_container_width=True)

        st.subheader("Bull/Bear Distribution")
        dist = pd.DataFrame({
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)
