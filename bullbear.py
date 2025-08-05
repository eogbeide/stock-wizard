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
  /* mobile sidebar override */
  @media (max-width: 600px) {
    .css-18e3th9 {transform:none!important;visibility:visible!important;width:100%!important;position:relative!important;margin-bottom:1rem;}
    .css-1v3fvcr {margin-left:0!important;}
  }
</style>
""", unsafe_allow_html=True)

# --- Utility ---
def safe_trend(x: np.ndarray, y: np.ndarray):
    try:
        coeff = np.polyfit(x, y, 1)
        trend = coeff[0] * x + coeff[1]
        return trend, coeff  # (trend array, (slope, intercept))
    except (np.linalg.LinAlgError, ValueError):
        m = np.nanmean(y)
        return np.full_like(x, m, dtype=float), (0.0, m)

# --- Auto-refresh logic ---
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if "last_refresh" not in st.session_state:
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

# Universe
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
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize("UTC")
    except Exception:
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
    idx = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# --- Indicator helpers ---
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

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- Session state init ---
st.session_state.setdefault("run_all", False)
st.session_state.setdefault("ticker", None)
st.session_state.setdefault("hour_range", "24h")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Original Forecast", "Enhanced Forecast", "Bull vs Bear", "Metrics"
])

# --- Tab 1: Original Forecast ---
with tab1:
    st.header("Original Forecast")
    st.info("Pick a ticker; data will be cached for 15 minutes after fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily", "Hourly", "Both"], key="orig_chart")
    hour_range = st.selectbox("Hourly lookback:", ["24h", "48h"], key="hour_range_select")
    auto_run = st.session_state.run_all and (sel != st.session_state.ticker)

    if st.button("Run Forecast") or auto_run or (not st.session_state.run_all):
        df_hist = fetch_hist(sel)
        intraday_period = "2d" if hour_range == "48h" else "1d"
        intraday = fetch_intraday(sel, period=intraday_period)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        # assign individually to avoid update conflicts
        st.session_state.df_hist = df_hist
        st.session_state.fc_idx = idx
        st.session_state.fc_vals = vals
        st.session_state.fc_ci = ci
        st.session_state.intraday = intraday
        st.session_state.ticker = sel
        st.session_state.chart = chart
        # only update if changed
        if st.session_state.get("hour_range") != hour_range:
            st.session_state.hour_range = hour_range
        st.session_state.run_all = True

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )

        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up
        expected_mean = float(vals.mean())
        pred_trend_pct = ((expected_mean - last_price) / last_price) * 100 if last_price != 0 else 0.0
        trend_label_daily = f"+{pred_trend_pct:.2f}%" if pred_trend_pct >= 0 else f"{pred_trend_pct:.2f}%"

        # --- Intraday ---
        if chart in ("Hourly", "Both"):
            intraday = st.session_state.intraday
            hc = intraday.get("Close", intraday).ffill()
            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            trend_h, coeff_h = safe_trend(xh, hc.values.flatten())
            res_h = hc.rolling(60, min_periods=1).max()
            sup_h = hc.rolling(60, min_periods=1).min()

            slope_pct = 0.0
            try:
                base = float(hc.iloc[0])
                if base != 0:
                    total_change = coeff_h[0] * (len(hc) - 1)
                    slope_pct = (total_change / base) * 100
            except Exception:
                slope_pct = 0.0
            trend_label_hourly = f"{slope_pct:.2f}%"

            fig2, ax2 = plt.subplots(figsize=(14,4))
            ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{p_up:.1%}  ↓{p_dn:.1%}  Predicted Trend: {trend_label_hourly}")
            ax2.plot(hc.index, hc, label="Intraday")
            ax2.plot(hc.index, he, "--", label="20 EMA")
            ax2.plot(hc.index, res_h, ":", label="Resistance")
            ax2.plot(hc.index, sup_h, ":", label="Support")
            ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)
            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig2)

        # --- Daily ---
        if chart in ("Daily", "Both"):
            ema200 = df.ewm(span=200).mean()
            ma30 = df.rolling(30).mean()
            lb, mb, ub = compute_bb(df)
            res = df.rolling(30, min_periods=1).max()
            sup = df.rolling(30, min_periods=1).min()
            x_fc = np.arange(len(vals))
            trend_fc, _ = safe_trend(x_fc, vals.to_numpy().flatten())
            macd_line, signal_line, hist = compute_macd(df)

            if isinstance(hist, (pd.Series, np.ndarray, list)):
                hist_series = pd.Series(hist)
                hist_vals = pd.to_numeric(hist_series, errors="coerce").fillna(0).to_numpy()
            else:
                hist_vals = np.zeros(len(df))

            fig, axes = plt.subplots(2, 1, figsize=(14,8), sharex=False)
            axes[0].set_title(f"{sel} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}  Predicted Trend: {trend_label_daily}")
            axes[0].plot(df[-360:], label="History")
            axes[0].plot(ema200[-360:], "--", label="200 EMA")
            axes[0].plot(ma30[-360:], "--", label="30 MA")
            axes[0].plot(res[-360:], ":", label="30 Resistance")
            axes[0].plot(sup[-360:], ":", label="30 Support")
            axes[0].plot(idx, vals, label="Forecast")
            axes[0].plot(idx, trend_fc, "--", label="Forecast Trend", linewidth=2)
            axes[0].fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            axes[0].plot(lb[-360:], "--", label="Lower BB")
            axes[0].plot(ub[-360:], "--", label="Upper BB")
            axes[0].set_xlabel("Date (PST)")
            axes[0].legend(loc="lower left", framealpha=0.5)

            axes[1].plot(df.index, macd_line.to_numpy(), label="MACD Line")
            axes[1].plot(df.index, signal_line.to_numpy(), "--", label="Signal Line")
            axes[1].bar(df.index, hist_vals, label="Histogram", alpha=0.5)
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].set_ylabel("MACD")
            axes[1].legend(loc="lower left", framealpha=0.5)
            axes[1].set_xlabel("Date (PST)")

            st.pyplot(fig)

        # Forecast summary table
        st.write(pd.DataFrame({
            "Forecast": st.session_state.fc_vals,
            "Lower": st.session_state.fc_ci.iloc[:,0],
            "Upper": st.session_state.fc_ci.iloc[:,1]
        }, index=st.session_state.fc_idx))

# --- Tab 2: Enhanced Forecast ---
with tab2:
    st.header("Enhanced Forecast")
    if not st.session_state.run_all:
        st.info("Run Tab 1 first.")
    else:
        df = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30 = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi = compute_rsi(df)
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up
        expected_mean = float(vals.mean())
        pred_trend_pct = ((expected_mean - last_price) / last_price) * 100 if last_price != 0 else 0.0
        trend_label = f"+{pred_trend_pct:.2f}%" if pred_trend_pct >= 0 else f"{pred_trend_pct:.2f}%"

        view = st.radio("View:", ["Daily", "Intraday", "Both"], key="enh_view")

        if view in ("Intraday", "Both"):
            ic = st.session_state.intraday.get("Close", st.session_state.intraday).ffill()
            ie = ic.ewm(span=20).mean()
            xi = np.arange(len(ic))
            trend_i, coeff_i = safe_trend(xi, ic.values.flatten())
            res_i = ic.rolling(60, min_periods=1).max()
            sup_i = ic.rolling(60, min_periods=1).min()

            slope_pct_i = 0.0
            try:
                base_i = float(ic.iloc[0])
                if base_i != 0:
                    total_change_i = coeff_i[0] * (len(ic) - 1)
                    slope_pct_i = (total_change_i / base_i) * 100
            except Exception:
                slope_pct_i = 0.0

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(f"{st.session_state.ticker} Intraday  ↑{p_up:.1%}  ↓{p_dn:.1%}  Predicted Trend: {slope_pct_i:.2f}%")
            ax3.plot(ic.index, ic, label="Intraday")
            ax3.plot(ic.index, ie, "--", label="20 EMA")
            ax3.plot(ic.index, res_i, ":", label="Resistance")
            ax3.plot(ic.index, sup_i, ":", label="Support")
            ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)
            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(14,3))
            ri = compute_rsi(ic)
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)")
            ax4.legend()
            st.pyplot(fig4)

        if view in ("Daily", "Both"):
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}  Predicted Trend: {trend_label}")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(mb[-360:], "--", label="Mid BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            for lev in (0.236, 0.382, 0.5, 0.618):
                ax.hlines(
                    df[-360:].max() - (df[-360:].max() - df[-360:].min()) * lev,
                    df.index[-360], df.index[-1],
                    linestyles="dotted"
                )
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        st.write(pd.DataFrame({
            "Forecast": vals,
            "Lower": ci.iloc[:,0],
            "Upper": ci.iloc[:,1]
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
        expected_mean = float(vals.mean())
        pred_trend_pct = ((expected_mean - last_price) / last_price) * 100 if last_price != 0 else 0.0
        trend_label = f"+{pred_trend_pct:.2f}%" if pred_trend_pct >= 0 else f"{pred_trend_pct:.2f}%"

        st.subheader(f"Last 3 Months  ↑{p_up:.1%}  ↓{p_dn:.1%}  Predicted Trend: {trend_label}")
        cutoff = df_hist.index.max() - pd.Timedelta(days=90)
        df3m = df_hist[df_hist.index >= cutoff]
        ma30_3m = df3m.rolling(30, min_periods=1).mean()
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        x3m = np.arange(len(df3m))
        trend3m, _ = safe_trend(x3m, df3m.values.flatten())

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
        df0['Bull'] = df0['PctChange'] > 0
        df0['MA30'] = df0['Close'].rolling(30, min_periods=1).mean()

        st.subheader("Close + 30-day MA + Trend")
        x0 = np.arange(len(df0))
        trend0, _ = safe_trend(x0, df0['Close'].values.flatten())
        res0 = df0['Close'].rolling(30, min_periods=1).max()
        sup0 = df0['Close'].rolling(30, min_periods=1).min()

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
            "Type": ["Bull", "Bear"],
            "Days": [int(df0['Bull'].sum()), int((~df0['Bull']).sum())]
        }).set_index("Type")
        st.bar_chart(dist, use_container_width=True)
