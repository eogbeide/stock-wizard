import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime

# --- Page config ---
st.set_page_config(
    page_title="🐂🐻 Bull vs Bear Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Sidebar: symbol selector ---
st.sidebar.title("Configuration")
symbol = st.sidebar.text_input("Ticker symbol", value="SPY", max_chars=10)
period = st.sidebar.selectbox("Lookback period", ["1mo", "3mo", "6mo", "1y"], index=2)

# --- Fetch data ---
@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period)
    data = data[['Close']].dropna()
    data['PctChange'] = data['Close'].pct_change()
    data['Bull'] = data['PctChange'] > 0
    return data

df = load_data(symbol, period)

# --- Compute metrics ---
bull_days = int(df['Bull'].sum())
bear_days = int((~df['Bull']).sum())
total_days = bull_days + bear_days
bull_pct = bull_days / total_days * 100 if total_days else 0
bear_pct = bear_days / total_days * 100 if total_days else 0

# --- Define tabs ---
tab1, tab2 = st.tabs([
    "🐂 Bull vs 🐻 Bear Summary",
    "📊 Detailed Metrics"
])

# --- Tab 1: Summary ---
with tab1:
    st.header("🐂 Bull vs 🐻 Bear Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Days", total_days)
    col2.metric("Bull Days", bull_days, f"{bull_pct:.1f}%")
    col3.metric("Bear Days", bear_days, f"{bear_pct:.1f}%")
    col4.metric("Period", period)

    st.markdown("---")
    st.write(f"Data for **{symbol}** over the past **{period}** (as of {df.index[-1].date()}):")

# --- Tab 2: Detailed Metrics ---
with tab2:
    st.header("📊 Detailed Metrics")

    st.subheader("Price Chart")
    st.line_chart(df['Close'], use_container_width=True)

    st.subheader("Bull/Bear Distribution")
    dist_df = pd.DataFrame({
        "Type": ["Bull", "Bear"],
        "Days": [bull_days, bear_days]
    })
    st.bar_chart(dist_df.set_index("Type"), use_container_width=True)

    st.subheader("Daily Percentage Change")
    st.line_chart(df['PctChange'], use_container_width=True)
