# bullbear.py — Stocks/Forex dashboard: tested signals only.
# (UPDATED 2026-11) Rebuilt around what held up in testing. Every daily scanner tab was replayed on
# past dates (2019-2026, every 5th trading day, 33 stocks and 17 pairs, each date seeing only its own
# past) and the intraday BUY/SELL banner on 5-minute bars (last 60 days). None beat a random symbol on
# a random day reliably; several "bullish" stock labels did worse. Those tabs were removed (see Rules).
# The Range Chart, RSI <= 30 and FX Stochastic Dip come from tested_charts.py (built in bullbears.py).
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import pytz

import tested_charts as tc

# --- Page config ---
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal CSS ---
st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}
  @media (max-width: 600px) {
    .css-18e3th9 { transform: none !important; visibility: visible !important; width: 100% !important; position: relative !important; margin-bottom: 1rem; }
    .css-1v3fvcr { margin-left: 0 !important; }
  }
</style>
""", unsafe_allow_html=True)


# --- Wrapped tabs CSS for better UX when many tabs are present ---
st.markdown("""
<style>
  div[data-testid="stTabs"] div[role="tablist"] {
    flex-wrap: wrap;
    gap: 0.15rem 0.35rem;
  }
  div[data-testid="stTabs"] button[role="tab"] {
    flex: 0 0 auto;
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
            st.experimental_rerun()
        except Exception:
            pass

auto_refresh()
elapsed = time.time() - st.session_state.last_refresh
remaining = max(0, int(REFRESH_INTERVAL - elapsed))
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(
    f"**Auto-refresh:** every {REFRESH_INTERVAL//60} min  \n"
    f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST  \n"
    f"**Next in:** ~{remaining}s"
)

# --- Sidebar ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Market:", ["Stock", "Forex"], key="sb_mode")
market = "Stocks" if mode == "Stock" else "Forex"

if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL','TLT'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X','AUDJPY=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X','EURCAD=X'
    ]

sel = st.sidebar.selectbox("Ticker:", universe, key=f"sb_ticker_{mode}")
period = st.sidebar.selectbox("Intraday chart period:", ["1d", "2d", "5d", "7d"], index=2, key="sb_intraday_period")
interval = st.sidebar.selectbox("Intraday bar interval:", ["5m", "15m", "30m", "60m"], index=1, key="sb_intraday_interval")
daily_period = st.sidebar.selectbox("Daily chart period:", ["3mo", "6mo", "1y", "2y", "5y"], index=2, key="sb_daily_period")
cfg = {"period": period, "interval": interval, "daily_period": daily_period}

st.title("📈 Stocks / FX: tested signals only")
st.caption(
    "Only what held up in testing: a likely price range (not a direction), the tested dip buy signals with "
    "each symbol's own past record, and an intraday view for timing. Tabs whose signals did no better than "
    "chance were removed; see Rules."
)

tab_range, tab_rsi, tab_stoch, tab_bull, tab_rules = st.tabs([
    "Range Chart",
    "RSI ≤ 30",
    "FX Stochastic Dip",
    "Bull vs Bears",
    "Rules",
])

with tab_range:
    tc.render_range_chart(sel, market, cfg)

with tab_rsi:
    st.subheader(f"{market} — RSI at or below 30 (daily)")
    tc.render_rsi_scan(universe, market)

with tab_stoch:
    st.subheader("Forex Stochastic Dip (daily)")
    if mode == "Forex":
        tc.render_fx_stochastic_scan(universe)
    else:
        st.info("Switch the market to Forex to scan pairs. For stocks, the tested dip signal is RSI ≤ 30 in a downtrend.")

with tab_bull:
    st.header("Bull vs Bear Summary")
    bb_period = st.selectbox("Lookback:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2, key="bb_period")
    df3 = tc.weekday_bars(tc.fetch_market_ohlc(sel, bb_period, "1d"))
    if df3.empty or len(df3) < 3:
        st.warning(f"No daily data for {sel}.")
    else:
        chg = df3["Close"].astype(float).pct_change().dropna()
        bull, bear = int((chg > 0).sum()), int((chg <= 0).sum())
        total = bull + bear
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Days", total)
        c2.metric("Bull Days", bull, f"{bull / total * 100:.1f}%")
        c3.metric("Bear Days", bear, f"{bear / total * 100:.1f}%")
        c4.metric("Lookback", bb_period)
        st.caption(
            "This is the base rate: how often a day closed higher. In the tests, no direction forecast or BUY/SELL "
            "label in this app beat it reliably, so treat any claim of 'bullish' or 'bearish' against this number."
        )

with tab_rules:
    st.subheader("What this app shows, and why")
    st.markdown(
        f"""
Everything here held up when tested on past data. Signals that did no better than chance were removed.

**Range Chart**
- **Likely range (blue cone)**: where the price is likely to be over the next {tc.RANGE_DAYS} trading days, with 80%
  and 50% bands, built from how far this market's symbols have moved in the past, scaled by the symbol's last
  {tc.RANGE_VOL_DAYS} days of volatility. Checked year by year (2018–2026), each year using only earlier years: the real
  close landed inside the 80% band 77–84% of the time (USD/CNY 75–91%). It does **not** say which way the price will go.
- **Dip buy signal (lower panel)**: Forex pairs when daily Stochastic %K(14) is {tc.FX_STOCH_LOW} or lower, sold when RSI
  reaches {tc.FX_STOCH_EXIT_RSI} (at most {tc.FX_STOCH_MAX_DAYS} days); stocks when daily RSI is {tc.STOCK_DIP_RSI} or below
  with the 120-day trend down, held {tc.STOCK_DIP_HOLD} trading days. Every past signal is marked green (rose) or red (fell).
- **Your position**: optional profit or loss, carry and the profit-or-loss range implied by the likely range.
- **Intraday**: for timing only, with today's range against a typical day and the trading sessions. No buy or sell labels.

**RSI ≤ 30** and **FX Stochastic Dip** scan every symbol for the tested dip signals.

**Bull vs Bears** shows how often days closed higher: the base rate any signal has to beat.

**What was removed, and why.** Each daily scanner was replayed on past dates (every 5th trading day, 2019–2026, all 33
stocks and 17 pairs here, each date seeing only its own past) and compared with a random symbol on a random day over the
next 20 trading days:
- Stocks: "BUY / Bullish" in Cumulative Frequency did 2.2% worse than a random stock, Trade Momentum's "BUY / Bullish"
  1.7% worse, Price Trend Bars' 12-month highs 3.0% worse and Buy/Sell Timing's "BUY Watch" 3.1% worse, each in both
  2019–22 and 2023–26. Shorting them didn't pay either, because stocks rose overall.
- Forex: no label in Cumulative Frequency, Trade Momentum, Rolling Return Strength, Buy/Sell Timing, -0.75 Picks,
  Smoothed NPX Cross or S/R Reversal Index was more than 0.3% better or worse than a random pair over 20 days, and none of
  the small leads held in both halves.
- The Original Forecast chart's BUY/SELL banner, replayed every 30 minutes on 5-minute bars (Aug–Nov 2026, 70,000
  readings): no better than chance; "near support" was on 90% of the time for Forex. Its 30-day SARIMAX forecast was
  right about half the time (2020–2026). HMA crosses and BB divergence showed no edge in the bullbears backtest.
The removed code is in the repository history.
"""
    )
    st.info("Size each trade so that a bad outcome costs a small, fixed part of the account. "
            "Not financial advice. Past results do not guarantee future results.")
