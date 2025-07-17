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
    page_title="📊 Multi‑Tab Forecast Dashboard",
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

# --- Helpers ---
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain/loss
    return 100 - (100/(1+rs))

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

# --- Sidebar config ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Mode:", ["Stock", "Forex"], key="global_mode")

# --- Session state ---
if 'run_all' not in st.session_state:
    st.session_state.run_all = False

# --- Tab definitions ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🇺🇸 Original US Forecast",
    "🇺🇸 Enhanced US Forecast",
    "🇳🇬 Nigeria Forecast",
    "🇬🇧 LSE Tech & Index"
])

# --- Tab 1: Original US Forecast ---
with tab1:
    st.header("🇺🇸 Original US Forecast")

    if mode == "Stock":
        ticker = st.selectbox(
            "Select Stock Ticker:",
            sorted([
                'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
                'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
                'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI'
            ]),
            key="orig_stock_ticker"
        )
    else:
        ticker = st.selectbox(
            "Select Forex Pair:",
            [
                'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
                'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
                'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'
            ],
            key="orig_forex_pair"
        )

    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    if st.button("Run Forecast", key="orig_btn"):
        # fetch historical
        df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
               .asfreq("D").fillna(method="ffill")
        st.session_state.df = df

        # SARIMAX forecast
        model = safe_sarimax(df, (1,1,1), (1,1,1,12))
        fc = model.get_forecast(steps=30)
        idx = pd.date_range(df.index[-1] + timedelta(1), periods=30, freq="D")
        st.session_state.fc_idx = idx
        st.session_state.fc_vals = fc.predicted_mean
        st.session_state.fc_ci = fc.conf_int()

        # intraday if stock
        if mode == "Stock":
            intraday = yf.download(ticker, period="1d", interval="5m")
            st.session_state.intraday = intraday

        # save selection
        st.session_state.ticker = ticker
        st.session_state.chart = chart
        st.session_state.mode = mode
        st.session_state.run_all = True

    # display (either just run or from state)
    if st.session_state.run_all:
        df = st.session_state.df
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        idx = st.session_state.fc_idx
        vals = st.session_state.fc_vals
        ci   = st.session_state.fc_ci

        if chart in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_title(f"{st.session_state.ticker} Daily Forecast")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if chart in ("Hourly","Both") and mode=="Stock":
            intraday = st.session_state.intraday
            if intraday.empty:
                st.warning("No intraday data.")
            else:
                hc = intraday["Close"].ffill()
                he = hc.ewm(span=20).mean()
                fig2, ax2 = plt.subplots(figsize=(14,5))
                ax2.plot(hc, label="Intraday")
                ax2.plot(he, "--", label="20 EMA")
                ax2.set_title(f"{st.session_state.ticker} Intraday (5m)")
                ax2.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2)

        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower":    st.session_state.fc_ci.iloc[:,0],
            "Upper":    st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))
    else:
        st.info("Press **Run Forecast** to populate all tabs.")

# --- Tab 2: Enhanced US Forecast ---
with tab2:
    st.header("🇺🇸 Enhanced US Forecast")
    if not st.session_state.run_all:
        st.info("Run the forecast in Tab 1 first.")
    else:
        df = st.session_state.df
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bollinger_bands(df)
        rsi    = compute_rsi(df)
        idx = st.session_state.fc_idx
        vals = st.session_state.fc_vals
        ci   = st.session_state.fc_ci

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            # Fibonacci
            hi, lo = df[-360:].max(), df[-360:].min()
            diff = hi-lo
            for lvl in (0.236,0.382,0.5,0.618):
                ax.hlines(hi-diff*lvl, df.index[-360], df.index[-1], linestyles="dotted")
            ax.set_title(f"{st.session_state.ticker} Daily + Fib")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig_rsi, ax_rsi = plt.subplots(figsize=(14,2))
            ax_rsi.plot(rsi[-360:], label="RSI(14)")
            ax_rsi.axhline(70, linestyle="--"); ax_rsi.axhline(30, linestyle="--")
            ax_rsi.set_title("RSI (14)"); ax_rsi.legend(loc="lower left")
            st.pyplot(fig_rsi)

        if view in ("Intraday","Both") and st.session_state.mode=="Stock":
            intraday = st.session_state.intraday
            if intraday.empty:
                st.warning("No intraday data.")
            else:
                ic = intraday["Close"].ffill(); ie = ic.ewm(span=20).mean()
                lb2, mb2, ub2 = compute_bollinger_bands(ic); ri = compute_rsi(ic)
                fig3, ax3 = plt.subplots(figsize=(14,5))
                ax3.plot(ic, label="Intraday"); ax3.plot(ie, "--", label="20 EMA")
                ax3.plot(lb2, "--", label="Lower BB"); ax3.plot(ub2, "--", label="Upper BB")
                ax3.set_title(f"{st.session_state.ticker} Intraday + Fib")
                ax3.legend(loc="lower left", framealpha=0.5); st.pyplot(fig3)
                fig4, ax4 = plt.subplots(figsize=(14,2))
                ax4.plot(ri, label="RSI(14)"); ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
                ax4.set_title("Intraday RSI (14)"); ax4.legend(loc="lower left"); st.pyplot(fig4)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower":    ci.iloc[:,0],
            "Upper":    ci.iloc[:,1]
        }, index=idx))

# --- Tab 3: Nigeria Forecast ---
with tab3:
    st.header("🇳🇬 Nigeria Forecast")
    if not st.session_state.run_all:
        st.info("Run the US forecast in Tab 1 first.")
    else:
        nigeria_mode = st.radio("Nigeria App:", ["Stock","Forex"], key="nig_mode")
        nigeria_stocks = {
            "Dangote Cement":"DANGCEM.LG","GT Bank":"GTCO.LG","Zenith Bank":"ZENL.L",
            "Access Holdings":"ACCESSCORP.LG","UBA":"UBA.LG","MTN Nigeria":"MTN.JO",
            "Seplat Energy":"SEPL.L","BUA Cement":"BUACEMENT.LG","Nestle Nigeria":"NESTLE.LG",
            "Nigerian Breweries":"NB.LG"
        }
        nigeria_fx = {
            "USD/NGN":"USDNGN=X","EUR/NGN":"EURNGN=X",
            "GBP/NGN":"GBPNGN=X","CNY/NGN":"CNYNGN=X","ZAR/NGN":"ZARNGN=X"
        }
        if nigeria_mode=="Stock":
            sel = st.selectbox("Select NG Stock:", list(nigeria_stocks.keys()), key="nig_stock")
            t_n = nigeria_stocks[sel]
        else:
            sel = st.selectbox("Select NG FX:", list(nigeria_fx.keys()), key="nig_fx")
            t_n = nigeria_fx[sel]

        df_n = yf.download(t_n, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                 .asfreq("D").fillna(method="ffill")
        if df_n.empty:
            st.warning("No data available.")
        else:
            ema2 = df_n.ewm(span=200).mean(); ma3 = df_n.rolling(30).mean()
            lb_n, mb_n, ub_n = compute_bollinger_bands(df_n)
            model_n = safe_sarimax(df_n, (1,1,1), (1,1,1,12))
            fc_n = model_n.get_forecast(steps=30)
            idx_n = pd.date_range(df_n.index[-1]+timedelta(1),30,freq="D")
            vals_n, ci_n = fc_n.predicted_mean, fc_n.conf_int()

            fign, axn = plt.subplots(figsize=(14,7))
            axn.plot(df_n[-360:], label="History")
            axn.plot(ema2[-360:], "--", label="200 EMA")
            axn.plot(ma3[-360:], "--", label="30 MA")
            axn.plot(idx_n, vals_n, label="Forecast")
            axn.fill_between(idx_n, ci_n.iloc[:,0], ci_n.iloc[:,1], alpha=0.3)
            axn.plot(lb_n[-360:], "--", label="Lower BB")
            axn.plot(ub_n[-360:], "--", label="Upper BB")
            axn.set_title(f"{sel} Nigeria Forecast")
            axn.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fign)

            st.write(pd.DataFrame({
                "Forecast": vals_n,
                "Lower":    ci_n.iloc[:,0],
                "Upper":    ci_n.iloc[:,1]
            }, index=idx_n))

# --- Tab 4: LSE Tech & Index ---
with tab4:
    st.header("🇬🇧 LSE Tech & Index")
    if not st.session_state.run_all:
        st.info("Run the US forecast in Tab 1 first.")
    else:
        lse_mode = st.radio("Category:", ["Tech Stocks","Index ETFs"], key="lse_mode")
        lse_tech = {
            "ARM Holdings":"ARM.L","Sage Group":"SGE.L","AVEVA":"AVV.L",
            "Softcat":"SCT.L","Darktrace":"DARK.L"
        }
        lse_etfs = {
            "iShares FTSE 100":"ISF.L","Vanguard FTSE 100":"VUKE.L",
            "FTSE 250":"MIDD.L","Core FTSE 100 Acc":"CUKX.L","MSCI World":"SWDA.L"
        }
        if lse_mode=="Tech Stocks":
            sel = st.selectbox("Select Tech Stock:", list(lse_tech.keys()), key="lse_tech")
            t_l = lse_tech[sel]
        else:
            sel = st.selectbox("Select Index ETF:", list(lse_etfs.keys()), key="lse_etf")
            t_l = lse_etfs[sel]

        df_l = yf.download(t_l, start="2018-01-01", end=pd.to_datetime("today"))['Close']\
                 .asfreq("D").fillna(method="ffill")
        if df_l.empty:
            st.warning("No data available.")
        else:
            ema2l = df_l.ewm(span=200).mean(); ma3l = df_l.rolling(30).mean()
            lb_l, mb_l, ub_l = compute_bollinger_bands(df_l)
            model_l = safe_sarimax(df_l, (1,1,1), (1,1,1,12))
            fc_l = model_l.get_forecast(steps=30)
            idx_l = pd.date_range(df_l.index[-1]+timedelta(1),30,freq="D")
            vals_l, ci_l = fc_l.predicted_mean, fc_l.conf_int()

            figl, axl = plt.subplots(figsize=(14,7))
            axl.plot(df_l[-360:], label="History")
            axl.plot(ema2l[-360:], "--", label="200 EMA")
            axl.plot(ma3l[-360:], "--", label="30 MA")
            axl.plot(idx_l, vals_l, label="Forecast")
            axl.fill_between(idx_l, ci_l.iloc[:,0], ci_l.iloc[:,1], alpha=0.3)
            axl.plot(lb_l[-360:], "--", label="Lower BB")
            axl.plot(ub_l[-360:], "--", label="Upper BB")
            axl.set_title(f"{sel} LSE Forecast")
            axl.legend(loc="lower left", framealpha=0.5)
            st.pyplot(figl)

            st.write(pd.DataFrame({
                "Forecast": vals_l,
                "Lower":    ci_l.iloc[:,0],
                "Upper":    ci_l.iloc[:,1]
            }, index=idx_l))
