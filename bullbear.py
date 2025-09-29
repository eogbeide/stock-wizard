# bullbear.py — Stocks/Forex Dashboard + Forecasts (+ Normalized RSI on EW panels)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory

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

# --- Auto-refresh ---
REFRESH_INTERVAL = 120
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
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# ---------- Helpers ----------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None: return pd.Series(dtype=float)
    if isinstance(obj, pd.Series): return pd.to_numeric(obj, errors="coerce")
    if isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        return pd.to_numeric(obj[num_cols[0]], errors="coerce") if num_cols else pd.Series(dtype=float)
    try: return pd.Series(obj)
    except: return pd.Series(dtype=float)

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_price_val(y: float) -> str:
    try: return f"{float(y):,.3f}"
    except: return "n/a"

def fmt_pct(x, digits: int = 1) -> str:
    try: xv = float(x)
    except: return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

# --- Sidebar Controls ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"])
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical","6M","12M","24M"], index=2)

st.sidebar.subheader("EW Overlays")
show_npo = st.sidebar.checkbox("Show NPO", value=True)
show_ntd = st.sidebar.checkbox("Show NTD", value=True)
show_ichi = st.sidebar.checkbox("Show Ichimoku", value=True)
show_nrsi = st.sidebar.checkbox("Show Normalized RSI", value=True)

# RSI params
nrsi_period = st.sidebar.slider("RSI period", 5, 50, 14, 1)
nrsi_norm_win = st.sidebar.slider("RSI normalization window", 30, 600, 240, 10)

# --- Indicator functions ---
def compute_npo(c, fast=12, slow=26, norm_win=240):
    s = _coerce_1d_series(c); 
    if s.empty: return pd.Series(index=s.index, dtype=float)
    ema_fast, ema_slow = s.ewm(span=fast).mean(), s.ewm(span=slow).mean()
    ppo = (ema_fast - ema_slow)/ema_slow*100
    mean, std = ppo.rolling(norm_win).mean(), ppo.rolling(norm_win).std()
    return np.tanh(((ppo-mean)/std)/2).reindex(s.index)

def compute_ntd(c, window=60):
    s = _coerce_1d_series(c)
    slope = s.rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x)>2 else np.nan)
    vol = s.rolling(window).std()
    return np.tanh((slope*window/vol)/2).reindex(s.index)

def compute_normalized_rsi(close: pd.Series, period: int = 14, norm_win: int = 240) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    delta = s.diff()
    gain, loss = np.where(delta>0, delta, 0.0), np.where(delta<0, -delta, 0.0)
    roll_up = pd.Series(gain, index=s.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=s.index).rolling(period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100/(1+rs))
    mean, std = rsi.rolling(norm_win).mean(), rsi.rolling(norm_win).std()
    return np.tanh(((rsi-mean)/std)/2).reindex(s.index)

def ichimoku_lines(h,l,c,conv=9,base=26,span_b=52):
    tenkan=(h.rolling(conv).max()+l.rolling(conv).min())/2
    kijun=(h.rolling(base).max()+l.rolling(base).min())/2
    span_a=(tenkan+kijun)/2
    span_b_=(h.rolling(span_b).max()+l.rolling(span_b).min())/2
    chikou=c.shift(-base)
    return tenkan,kijun,span_a,span_b_,chikou

def compute_normalized_ichimoku(h,l,c,conv=9,base=26,span_b=52,norm_win=240,w=0.6):
    tenkan,kijun,sa,sb,_=ichimoku_lines(h,l,c,conv,base,span_b)
    cloud=((sa+sb)/2).reindex(c.index)
    vol=c.rolling(norm_win).std()
    z1=(c-cloud)/vol; z2=(tenkan-kijun)/vol
    return np.tanh((w*z1+(1-w)*z2)/2)
# --- Cache helpers ---
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = yf.download(ticker, start="2018-01-01")['Close'].asfreq("D").ffill()
    try: s = s.tz_localize(PACIFIC)
    except TypeError: s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01")[['Open','High','Low','Close']].dropna()
    try: df = df.tz_localize(PACIFIC)
    except TypeError: df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1h")
    try: df = df.tz_localize("UTC")
    except TypeError: pass
    return df.tz_convert(PACIFIC)

# --- Shading helpers ---
def shade_series(ax, s, color_up="green", color_down="red", alpha=0.15):
    if s is None or s.empty: return
    pos, neg = s.where(s > 0), s.where(s < 0)
    ax.fill_between(pos.index, 0, pos, color=color_up, alpha=alpha)
    ax.fill_between(neg.index, 0, neg, color=color_down, alpha=alpha)

# --- Price chart helpers ---
def draw_daily_chart(df, df_ohlc, kijun=None):
    ema30 = df.ewm(span=30).mean()
    res30, sup30 = df.rolling(30).max(), df.rolling(30).min()
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df, label="History")
    ax.plot(ema30, "--", label="30 EMA")
    ax.plot(res30, ":", label="30 Resistance")
    ax.plot(sup30, ":", label="30 Support")
    if kijun is not None and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun, "-", color="black", linewidth=1.8, label="Ichimoku Kijun")
    ax.legend(loc="lower left", framealpha=0.5)
    ax.set_ylabel("Price")
    return fig, ax

def draw_hourly_chart(df, kijun=None):
    he = df.ewm(span=20).mean()
    fig, ax = plt.subplots(figsize=(14,3))
    ax.plot(df, label="Intraday")
    ax.plot(he, "--", label="20 EMA")
    if kijun is not None and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun, "-", color="black", linewidth=1.8, label="Ichimoku Kijun")
    ax.legend(loc="lower left", framealpha=0.5)
    ax.set_ylabel("Price")
    return fig, ax
# --- EW Panels ---
def draw_ew_panel(df, wave_norm, npo=None, ntd=None, ichi=None, nrsi=None, title="EW Panel"):
    fig, ax = plt.subplots(figsize=(14,3))
    ax.set_title(title)
    ax.plot(wave_norm.index, wave_norm, label="Norm EW", linewidth=1.8)

    if npo is not None and not npo.dropna().empty:
        shade_series(ax, npo, color_up="red", color_down="red", alpha=0.1)
        ax.plot(npo, "--", label="NPO")
    if ntd is not None and not ntd.dropna().empty:
        shade_series(ax, ntd, color_up="green", color_down="red", alpha=0.1)
        ax.plot(ntd, ":", label="NTD")
    if ichi is not None and not ichi.dropna().empty:
        ax.plot(ichi, "-", color="black", linewidth=1.4, label="IchimokuN")
    if nrsi is not None and not nrsi.dropna().empty:
        shade_series(ax, nrsi, color_up="green", color_down="red", alpha=0.08)
        ax.plot(nrsi, "--", color="purple", linewidth=1.2, label="NRSI")

    # Reference lines
    for y, c, l in [(0,"grey","EW 0"),(0.5,"red","+0.5"),(-0.5,"green","-0.5"),(0.75,"black","+0.75"),(-0.75,"black","-0.75")]:
        ax.axhline(y, linestyle="--", linewidth=1, color=c, label=l)
    ax.set_ylim(-1.1,1.1); ax.legend(loc="lower left", framealpha=0.5)
    return fig, ax

# --- Main run (simplified example) ---
ticker = st.sidebar.selectbox("Ticker", ["AAPL","MSFT","TSLA","EURUSD=X","USDJPY=X"])
if st.sidebar.button("Run"):
    df = fetch_hist(ticker)
    ohlc = fetch_ohlc(ticker)
    intraday = fetch_intraday(ticker)

    # Daily indicators
    wave_norm_d = compute_ntd(df)  # placeholder EW
    npo_d = compute_npo(df) if show_npo else None
    ntd_d = compute_ntd(df) if show_ntd else None
    nrsi_d = compute_normalized_rsi(df, nrsi_period, nrsi_norm_win) if show_nrsi else None
    ichi_d = compute_normalized_ichimoku(ohlc["High"],ohlc["Low"],ohlc["Close"]) if show_ichi else None
    _, kijun_d,_,_,_ = ichimoku_lines(ohlc["High"],ohlc["Low"],ohlc["Close"])

    # Hourly indicators
    hc = intraday["Close"]
    wave_norm_h = compute_ntd(hc)  # placeholder EW
    npo_h = compute_npo(hc) if show_npo else None
    ntd_h = compute_ntd(hc) if show_ntd else None
    nrsi_h = compute_normalized_rsi(hc, nrsi_period, nrsi_norm_win) if show_nrsi else None
    ichi_h = compute_normalized_ichimoku(intraday["High"],intraday["Low"],intraday["Close"]) if show_ichi else None
    _, kijun_h,_,_,_ = ichimoku_lines(intraday["High"],intraday["Low"],intraday["Close"])

    # Render
    st.subheader("Daily Chart")
    fig_d, _ = draw_daily_chart(df, ohlc, kijun_d)
    st.pyplot(fig_d)

    st.subheader("Daily EW Panel")
    fig_dw, _ = draw_ew_panel(df, wave_norm_d, npo_d, ntd_d, ichi_d, nrsi_d, "Daily EW + Overlays")
    st.pyplot(fig_dw)

    st.subheader("Hourly Chart")
    fig_h, _ = draw_hourly_chart(hc, kijun_h)
    st.pyplot(fig_h)

    st.subheader("Hourly EW Panel")
    fig_hw, _ = draw_ew_panel(hc, wave_norm_h, npo_h, ntd_h, ichi_h, nrsi_h, "Hourly EW + Overlays")
    st.pyplot(fig_hw)
