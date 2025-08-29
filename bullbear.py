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
    .css-18e3th9 {
      transform: none !important;
      visibility: visible !important;
      width: 100% !important;
      position: relative !important;
      margin-bottom: 1rem;
    }
    .css-1v3fvcr { margin-left: 0 !important; }
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
            # newer streamlit
            if hasattr(st, "rerun"): st.rerun()
            else: st.experimental_rerun()
        except Exception:
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
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
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
    idx = pd.date_range(series.index[-1] + timedelta(1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---------- Indicator helpers ----------
def compute_rsi(data, window=14):
    d = data.diff()
    gain = d.where(d>0,0).rolling(window).mean()
    loss = -d.where(d<0,0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_bb(data, window=20, num_sd=2):
    m = data.rolling(window).mean()
    s = data.rolling(window).std()
    return m - num_sd*s, m, m + num_sd*s

def compute_supertrend(ohlc: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    ATR-based Supertrend. Returns columns: 'supertrend', 'direction' (1 bull, -1 bear), 'signal'.
    Signal is +1 on bullish flip, -1 on bearish flip, else 0.
    """
    df = ohlc[['High','Low','Close']].copy().dropna()
    hl2 = (df['High'] + df['Low']) / 2.0

    # True Range and ATR (EMA for smoother response)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    st_line.iloc[0] = upper.iloc[0]
    direction.iloc[0] = -1  # start bearish by default

    for i in range(1, len(df)):
        prev_dir = direction.iloc[i-1]
        prev_st  = st_line.iloc[i-1]

        # band persistence
        if upper.iloc[i] < prev_st and prev_dir == -1:
            upper.iloc[i] = prev_st
        if lower.iloc[i] > prev_st and prev_dir == 1:
            lower.iloc[i] = prev_st

        if prev_dir == -1:
            if df['Close'].iloc[i] > upper.iloc[i]:
                direction.iloc[i] = 1
                st_line.iloc[i] = lower.iloc[i]
            else:
                direction.iloc[i] = -1
                st_line.iloc[i] = upper.iloc[i]
        else:
            if df['Close'].iloc[i] < lower.iloc[i]:
                direction.iloc[i] = -1
                st_line.iloc[i] = upper.iloc[i]
            else:
                direction.iloc[i] = 1
                st_line.iloc[i] = lower.iloc[i]

    signal = direction.diff().fillna(0)
    signal = signal.map({2:1, -2:-1}).fillna(0).astype(int)  # +1 buy, -1 sell, 0 none

    out = pd.DataFrame({
        'supertrend': st_line,
        'direction': direction,
        'signal': signal
    }, index=df.index)
    return out

# Session state init
if 'run_all' not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.hour_range = "24h"

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
    st.info("Pick a ticker; data will be cached for 15 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

    # Hourly lookback selector
    hour_range = st.selectbox(
        "Hourly lookback:",
        ["24h", "48h", "96h"],
        index=["24h","48h","96h"].index(st.session_state.get("hour_range","24h")),
        key="hour_range_select"
    )
    period_map = {"24h": "1d", "48h": "2d", "96h": "4d"}

    auto_run = (
        st.session_state.run_all and (
            sel != st.session_state.ticker or
            hour_range != st.session_state.get("hour_range")
        )
    )

    if st.button("Run Forecast") or auto_run:
        df_hist = fetch_hist(sel)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist,
            "fc_idx": idx,
            "fc_vals": vals,
            "fc_ci": ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

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

        x_fc = np.arange(len(vals))
        slope_fc, intercept_fc = np.polyfit(x_fc, vals.to_numpy(), 1)
        trend_fc = slope_fc * x_fc + intercept_fc

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
            ax.plot(idx, trend_fc, "--", label="Forecast Trend", linewidth=2)
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            ax.plot(lb[-360:], "--", label="Lower BB")
            ax.plot(ub[-360:], "--", label="Upper BB")
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

        if chart in ("Hourly","Both"):
            # 5m series for visual context
            intr = st.session_state.intraday[['Open','High','Low','Close']].dropna()
            hc = intr['Close'].ffill()

            # ---- Supertrend on HOURLY candles ----
            h1 = intr.resample('1H').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            st_df = compute_supertrend(h1, period=10, multiplier=3.0)

            # BUY/SELL markers (based on hourly signal)
            buys  = st_df[st_df['signal'] == 1].index
            sells = st_df[st_df['signal'] == -1].index

            # Latest signal
            latest_dir = int(st_df['direction'].iloc[-1])
            latest_sig = int(st_df['signal'].iloc[-1])
            label = "BUY" if latest_dir == 1 else "SELL"
            delta = "Triggered" if latest_sig != 0 else "—"
            st.metric("Supertrend (1H)", label, delta=delta)

            he = hc.ewm(span=20).mean()
            xh = np.arange(len(hc))
            slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
            trend_h = slope_h * xh + intercept_h
            res_h = hc.rolling(60, min_periods=1).max()
            sup_h = hc.rolling(60, min_periods=1).min()

            fig2, ax2 = plt.subplots(figsize=(14,5))
            ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range}) — with Supertrend (1H)  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax2.plot(hc.index, hc, label="5-min Close", alpha=0.65)
            ax2.plot(hc.index, he, "--", label="20 EMA", alpha=0.7)
            ax2.plot(hc.index, res_h, ":", label="Resistance", alpha=0.5)
            ax2.plot(hc.index, sup_h, ":", label="Support", alpha=0.5)

            # Supertrend line & markers (on hourly timeline)
            ax2.plot(st_df.index, st_df['supertrend'], linewidth=2, label="Supertrend (1H)")

            # Marker y-values from matching close price near those timestamps
            buy_y  = h1['Close'].reindex(buys)
            sell_y = h1['Close'].reindex(sells)
            ax2.scatter(buy_y.index,  buy_y.values,  marker="^", s=120, label="BUY",  zorder=5)
            ax2.scatter(sell_y.index, sell_y.values, marker="v", s=120, label="SELL", zorder=5)

            ax2.set_xlabel("Time (PST)")
            ax2.legend(loc="lower left", framealpha=0.5, ncol=3)
            st.pyplot(fig2)

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
        df     = st.session_state.df_hist
        ema200 = df.ewm(span=200).mean()
        ma30   = df.rolling(30).mean()
        lb, mb, ub = compute_bb(df)
        rsi    = compute_rsi(df)
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = float(df.iloc[-1])
        p_up = np.mean(vals.to_numpy() > last_price)
        p_dn = 1 - p_up
        res = df.rolling(30, min_periods=1).max()
        sup = df.rolling(30, min_periods=1).min()

        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}** (change in Tab 1)")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")
        if view in ("Daily","Both"):
            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax.plot(df[-360:], label="History")
            ax.plot(ema200[-360:], "--", label="200 EMA")
            ax.plot(ma30[-360:], "--", label="30 MA")
            ax.plot(res[-360:], ":", label="Resistance")
            ax.plot(sup[-360:], ":", label="Support")
            ax.plot(idx, vals, label="Forecast")
            ax.fill_between(idx, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
            for lev in (0.236,0.382,0.5,0.618):
                ax.hlines(
                    df[-360:].max() - (df[-360:].max() - df[-360:].min())*lev,
                    df.index[-360], df.index[-1], linestyles="dotted"
                )
            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(14,3))
            ax2.plot(rsi[-360:], label="RSI(14)")
            ax2.axhline(70, linestyle="--"); ax2.axhline(30, linestyle="--")
            ax2.set_xlabel("Date (PST)")
            ax2.legend()
            st.pyplot(fig2)

        if view in ("Intraday","Both"):
            intr = st.session_state.intraday[['Open','High','Low','Close']].dropna()
            ic = intr['Close'].ffill()
            ie = ic.ewm(span=20).mean()
            xi = np.arange(len(ic))
            slope_i, intercept_i = np.polyfit(xi, ic.values, 1)
            trend_i = slope_i * xi + intercept_i
            res_i = ic.rolling(60, min_periods=1).max()
            sup_i = ic.rolling(60, min_periods=1).min()

            # Supertrend (1H) again
            h1 = intr.resample('1H').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
            st_df = compute_supertrend(h1, period=10, multiplier=3.0)
            buys  = st_df[st_df['signal'] == 1].index
            sells = st_df[st_df['signal'] == -1].index

            fig3, ax3 = plt.subplots(figsize=(14,4))
            ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range}) + Supertrend (1H)  ↑{p_up:.1%}  ↓{p_dn:.1%}")
            ax3.plot(ic.index, ic, label="5-min Close", alpha=0.65)
            ax3.plot(ic.index, ie, "--", label="20 EMA", alpha=0.7)
            ax3.plot(ic.index, res_i, ":", label="Resistance", alpha=0.5)
            ax3.plot(ic.index, sup_i, ":", label="Support", alpha=0.5)
            ax3.plot(st_df.index, st_df['supertrend'], linewidth=2, label="Supertrend (1H)")
            ax3.scatter(buys,  h1['Close'].reindex(buys),  marker="^", s=120, label="BUY",  zorder=5)
            ax3.scatter(sells, h1['Close'].reindex(sells), marker="v", s=120, label="SELL", zorder=5)
            ax3.set_xlabel("Time (PST)")
            ax3.legend(loc="lower left", framealpha=0.5, ncol=3)
            st.pyplot(fig3)

            fig4, ax4 = plt.subplots(figsize=(14,3))
            ri = compute_rsi(ic)
            ax4.plot(ri, label="RSI(14)")
            ax4.axhline(70, linestyle="--"); ax4.axhline(30, linestyle="--")
            ax4.set_xlabel("Time (PST)")
            ax4.legend()
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
        res3m = df3m.rolling(30, min_periods=1).max()
        sup3m = df3m.rolling(30, min_periods=1).min()
        x3m = np.arange(len(df3m))
        slope3m, intercept3m = np.polyfit(x3m, df3m.values, 1)
        trend3m = slope3m * x3m + intercept3m

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
        slope0, intercept0 = np.polyfit(x0, df0['Close'], 1)
        trend0 = slope0 * x0 + intercept0
        res0 = df0.rolling(30, min_periods=1).max()
        sup0 = df0.rolling(30, min_periods=1).min()

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
