# bullbear.py — Stocks/Forex Dashboard + Forecasts
# - Forex news markers on intraday charts
# - Hourly momentum indicator (ROC%) with robust handling
# - Momentum trendline & momentum S/R
# - Daily shows: History, 30 EMA, 30 S/R, Daily slope, Pivots (P, R1/S1, R2/S2) + value labels
# - EMA30 slope overlay on Daily
# - Hourly includes Supertrend overlay (configurable ATR period & multiplier)
# - Fixes tz_localize error by using tz-aware UTC timestamps
# - Auto-refresh, SARIMAX (for probabilities)
# - Cache TTLs = 2 minutes (120s)
# - Hourly BUY/SELL logic (near S/R + confidence threshold)
# - Value labels on intraday Resistance/Support placed on the LEFT; price label outside chart (top-right)
# - All displayed price values formatted to 3 decimal places
# - Hourly Support/Resistance drawn as STRAIGHT LINES across the entire chart
# - Current price shown OUTSIDE of chart area (top-right above axes)
# - Removed BUY/SELL triangles from charts; keep indicators in the title ("symbol area") with ▲/▼ and price values
# - Normalized Elliott Wave panel for Hourly (dates aligned to hourly chart)
# - Normalized Elliott Wave panel for Daily (dates aligned to daily chart)
# - NEW: EW panels show BUY/SELL signals when forecast confidence > 95% and display current price on top
# - NEW: EW panels draw a red line at +0.5 and a green line at -0.5

import streamlit as stc
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import time
import pytz
from matplotlib.transforms import blended_transform_factory  # for left-side labels

# --- Page config (must be the first Streamlit call) ---
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
pst_dt = datetime.fromtimestamp(st.session_state.last_refresh, tz=PACIFIC)
st.sidebar.markdown(f"**Last refresh:** {pst_dt.strftime('%Y-%m-%d %H:%M:%S')} PST")

# ---------- Helpers ----------
def _coerce_1d_series(obj) -> pd.Series:
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.Series):
        s = obj
    elif isinstance(obj, pd.DataFrame):
        num_cols = [c for c in obj.columns if pd.api.types.is_numeric_dtype(obj[c])]
        if not num_cols:
            return pd.Series(dtype=float)
        s = obj[num_cols[0]]
    else:
        try:
            s = pd.Series(obj)
        except Exception:
            return pd.Series(dtype=float)
    return pd.to_numeric(s, errors="coerce")

def _safe_last_float(obj) -> float:
    s = _coerce_1d_series(obj).dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")

def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_price_val(y: float) -> str:
    """Format price values to exactly 3 decimal places with thousands separators."""
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.3f}"

def fmt_slope(m: float) -> str:
    return f"{m:.4f}" if np.isfinite(m) else "n/a"

# Place text at the left edge (x in axes coords, y in data coords)
def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    try:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.01, y_val, text,
            transform=trans, ha="left", va="center",
            color=color, fontsize=fontsize, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=6
        )
    except Exception:
        pass

# --- Sidebar config (explicit keys everywhere) ---
st.sidebar.title("Configuration")
mode = st.sidebar.selectbox("Forecast Mode:", ["Stock", "Forex"], key="sb_mode")
bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=False, key="sb_show_fibs")

slope_lb_daily   = st.sidebar.slider("Daily slope lookback (bars)",   10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly  = st.sidebar.slider("Hourly slope lookback (bars)",  12, 480, 120, 6, key="sb_slope_lb_hourly")

# Hourly Momentum controls
st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=True, key="sb_show_mom_hourly")
mom_lb_hourly   = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

# Hourly Supertrend controls
st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult   = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

# Signal logic controls
st.sidebar.subheader("Signal Logic (Hourly)")
signal_threshold = st.sidebar.slider("Signal confidence threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

# Elliott Wave controls (Hourly)
st.sidebar.subheader("Normalized Elliott Wave (Hourly)")
pivot_lookback = st.sidebar.slider("Pivot lookback (bars)", 3, 21, 7, 2, key="sb_pivot_lb")
norm_window    = st.sidebar.slider("Normalization window (bars)", 30, 600, 240, 10, key="sb_norm_win")
waves_to_annotate = st.sidebar.slider("Annotate recent waves", 3, 12, 7, 1, key="sb_wave_ann")

# Elliott Wave controls (Daily)
st.sidebar.subheader("Normalized Elliott Wave (Daily)")
pivot_lookback_d = st.sidebar.slider("Pivot lookback (days)", 3, 31, 9, 2, key="sb_pivot_lb_d")
norm_window_d    = st.sidebar.slider("Normalization window (days)", 30, 1200, 360, 10, key="sb_norm_win_d")
waves_to_annotate_d = st.sidebar.slider("Annotate recent waves (daily)", 3, 12, 7, 1, key="sb_wave_ann_d")

# Forex news controls (only shown in Forex mode)
if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
else:
    show_fx_news = False
    news_window_days = 7

# Universe
if mode == "Stock":
    universe = sorted([
        'AAPL','SPY','AMZN','DIA','TSLA','SPGI','JPM','VTWG','PLTR','NVDA',
        'META','SITM','MARA','GOOG','HOOD','BABA','IBM','AVGO','GUSH','VOO',
        'MSFT','TSM','NFLX','MP','AAL','URI','DAL','BBAI','QUBT','AMD','SMCI','ORCL'
    ])
else:
    universe = [
        'EURUSD=X','EURJPY=X','GBPUSD=X','USDJPY=X','AUDUSD=X','NZDUSD=X',
        'HKDJPY=X','USDCAD=X','USDCNY=X','USDCHF=X','EURGBP=X',
        'USDHKD=X','EURHKD=X','GBPHKD=X','GBPJPY=X'
    ]

# --- Cache helpers (TTL = 120 seconds) ---
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (
        yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))['Close']
        .asfreq("D").fillna(method="ffill")
    )
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[['Open','High','Low','Close']].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    try:
        df = df.tz_localize('UTC')
    except TypeError:
        pass
    return df.tz_convert(PACIFIC)

@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like):
    series = _coerce_1d_series(series_like).dropna()
    if isinstance(series.index, pd.DatetimeIndex):
        if series.index.tz is None:
            series.index = series.index.tz_localize(PACIFIC)
        else:
            series.index = series.index.tz_convert(PACIFIC)
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except np.linalg.LinAlgError:
        model = SARIMAX(
            series, order=(1,1,1), seasonal_order=(1,1,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1),
                        periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---- Indicators (no RSI) ----
def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    if s.empty:
        return {}
    hi = float(s.max()); lo = float(s.min())
    diff = hi - lo
    if diff == 0:
        return {"100%": lo}
    return {
        "0%": hi,
        "23.6%": hi - 0.236 * diff,
        "38.2%": hi - 0.382 * diff,
        "50%": hi - 0.5   * diff,
        "61.8%": hi - 0.618 * diff,
        "78.6%": hi - 0.786 * diff,
        "100%": lo,
    }

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty:
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    try:
        H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    except Exception:
        return {}
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}

def slope_line(series_like, lookback: int):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    s = s.iloc[-lookback:] if lookback > 0 else s
    if s.shape[0] < 2:
        return pd.Series(dtype=float), float("nan")
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.to_numpy(dtype=float), 1)
    yhat = pd.Series(m * x + b, index=s.index)
    return yhat, float(m)

def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

# ---- Supertrend helpers (hourly overlay) ----
def _true_range(df: pd.DataFrame):
    hl = (df["High"] - df["Low"]).abs()
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    tr = _true_range(df[['High','Low','Close']])
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    if df is None or df.empty or not {'High','Low','Close'}.issubset(df.columns):
        idx = df.index if df is not None else pd.Index([])
        return pd.DataFrame(index=idx, columns=["ST","in_uptrend","upperband","lowerband"])
    ohlc = df[['High','Low','Close']].copy()
    hl2 = (ohlc['High'] + ohlc['Low']) / 2.0
    atr = compute_atr(ohlc, atr_period)
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr
    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_up   = pd.Series(index=ohlc.index, dtype=bool)
    st_line.iloc[0] = upperband.iloc[0]
    in_up.iloc[0]   = True
    for i in range(1, len(ohlc)):
        prev_st = st_line.iloc[i-1]
        prev_up = in_up.iloc[i-1]
        up_i = min(upperband.iloc[i], prev_st) if prev_up else upperband.iloc[i]
        dn_i = max(lowerband.iloc[i], prev_st) if not prev_up else lowerband.iloc[i]
        close_i = ohlc['Close'].iloc[i]
        if close_i > up_i:
            curr_up = True
        elif close_i < dn_i:
            curr_up = False
        else:
            curr_up = prev_up
        in_up.iloc[i]   = curr_up
        st_line.iloc[i] = dn_i if curr_up else up_i
    return pd.DataFrame({
        "ST": st_line, "in_uptrend": in_up,
        "upperband": upperband, "lowerband": lowerband
    })

# ---- Forex News (Yahoo Finance) ----
@st.cache_data(ttl=120, show_spinner=False)
def fetch_yf_news(symbol: str, window_days: int = 7) -> pd.DataFrame:
    rows = []
    try:
        news_list = yf.Ticker(symbol).news or []
    except Exception:
        news_list = []
    for item in news_list:
        ts = item.get("providerPublishTime") or item.get("pubDate")
        if ts is None:
            continue
        try:
            dt_utc = pd.to_datetime(ts, unit="s", utc=True)
        except (ValueError, OverflowError, TypeError):
            try:
                dt_utc = pd.to_datetime(ts, utc=True)
            except Exception:
                continue
        dt_pst = dt_utc.tz_convert(PACIFIC)
        rows.append({
            "time": dt_pst,
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", "")
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    now_utc = pd.Timestamp.now(tz="UTC")
    d1 = (now_utc - pd.Timedelta(days=window_days)).tz_convert(PACIFIC)
    return df[df["time"] >= d1].sort_values("time")

def draw_news_markers(ax, times, ymin, ymax, label="News"):
    for t in times:
        try:
            ax.axvline(t, color="tab:red", alpha=0.18, linewidth=1)
        except Exception:
            pass
    ax.plot([], [], color="tab:red", alpha=0.5, linewidth=2, label=label)

# --- Signal helpers ---
def sr_proximity_signal(hc: pd.Series, res_h: pd.Series, sup_h: pd.Series,
                        fc_vals: pd.Series, threshold: float, prox: float):
    """Return signal info if last price is near hourly S/R and model confidence passes threshold."""
    try:
        last_close = float(hc.iloc[-1])
        res = float(res_h.iloc[-1])
        sup = float(sup_h.iloc[-1])
    except Exception:
        return None

    if not np.all(np.isfinite([last_close, res, sup])) or res <= sup:
        return None

    near_support = last_close <= sup * (1.0 + prox)
    near_resist  = last_close >= res * (1.0 - prox)

    fc = np.asarray(_coerce_1d_series(fc_vals).dropna(), dtype=float)
    if fc.size == 0:
        return None
    p_up_from_here = float(np.mean(fc > last_close))
    p_dn_from_here = float(np.mean(fc < last_close))

    if near_support and p_up_from_here >= threshold:
        return {
            "side": "BUY",
            "prob": p_up_from_here,
            "level": sup,
            "reason": f"Near support {fmt_price_val(sup)} with {fmt_pct(p_up_from_here)} up-confidence ≥ {fmt_pct(threshold)}"
        }
    if near_resist and p_dn_from_here >= threshold:
        return {
            "side": "SELL",
            "prob": p_dn_from_here,
            "level": res,
            "reason": f"Near resistance {fmt_price_val(res)} with {fmt_pct(p_dn_from_here)} down-confidence ≥ {fmt_pct(threshold)}"
        }
    return None

EW_CONFIDENCE = 0.95  # >95% confidence for EW signals

def elliott_conf_signal(price_now: float, fc_vals: pd.Series, conf: float = EW_CONFIDENCE):
    """Signal for EW panel using SARIMAX distribution vs current price."""
    fc = _coerce_1d_series(fc_vals).dropna().to_numpy(dtype=float)
    if fc.size == 0 or not np.isfinite(price_now):
        return None
    p_up = float(np.mean(fc > price_now))
    p_dn = float(np.mean(fc < price_now))
    if p_up >= conf:
        return {"side": "BUY", "prob": p_up}
    if p_dn >= conf:
        return {"side": "SELL", "prob": p_dn}
    return None

# --- Normalized Elliott Wave (simple, dependency-free) ----
def compute_normalized_elliott_wave(close: pd.Series,
                                    pivot_lb: int = 7,
                                    norm_win: int = 240) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      wave_norm: pd.Series in [-1,1] (tanh(zscore) of close vs rolling mean/std)
      pivots_df: DataFrame with 'time','price','type' ('H'/'L') and 'wave' labels (1..5 repeating)
    """
    s = _coerce_1d_series(close).dropna()
    if s.empty:
        return pd.Series(index=close.index, dtype=float), pd.DataFrame(columns=["time","price","type","wave"])

    # Normalization: rolling z-score, then squash to [-1,1]
    minp = max(10, norm_win//10)
    mean = s.rolling(norm_win, min_periods=minp).mean()
    std  = s.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (s - mean) / std
    wave_norm = np.tanh(z / 2.0)  # smoother, bounded [-1,1]
    wave_norm = wave_norm.reindex(close.index)

    # Pivot detection via centered rolling extrema
    if pivot_lb % 2 == 0:
        pivot_lb += 1
    half = pivot_lb // 2
    roll_max = s.rolling(pivot_lb, center=True).max()
    roll_min = s.rolling(pivot_lb, center=True).min()

    pivots = []
    for i in range(half, len(s)-half):
        if not np.isfinite(s.iloc[i]):
            continue
        if s.iloc[i] == roll_max.iloc[i]:
            pivots.append((s.index[i], float(s.iloc[i]), 'H'))
        elif s.iloc[i] == roll_min.iloc[i]:
            pivots.append((s.index[i], float(s.iloc[i]), 'L'))

    # De-duplicate consecutive same-type pivots
    dedup = []
    for t, p, typ in pivots:
        if not dedup:
            dedup.append((t,p,typ))
        else:
            pt, pp, ptyp = dedup[-1]
            if typ == ptyp:
                if (typ == 'H' and p > pp) or (typ == 'L' and p < pp):
                    dedup[-1] = (t,p,typ)
            else:
                dedup.append((t,p,typ))

    # Assign simple 1..5 wave counting
    waves = []
    wave_num = 1
    for t, p, typ in dedup:
        waves.append((t,p,typ,wave_num))
        wave_num += 1
        if wave_num > 5:
            wave_num = 1

    pivots_df = pd.DataFrame(waves, columns=["time","price","type","wave"])
    return wave_norm, pivots_df

# --- Session init ---
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
    st.info("Pick a ticker; data will be cached for 2 minutes after first fetch.")

    sel = st.selectbox("Ticker:", universe, key="orig_ticker")
    chart = st.radio("Chart View:", ["Daily","Hourly","Both"], key="orig_chart")

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

    if st.button("Run Forecast", key="btn_run_forecast") or auto_run:
        df_hist = fetch_hist(sel)
        df_ohlc = fetch_hist_ohlc(sel)
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_hist)
        intraday = fetch_intraday(sel, period=period_map[hour_range])
        st.session_state.update({
            "df_hist": df_hist,
            "df_ohlc": df_ohlc,
            "fc_idx": fc_idx,
            "fc_vals": fc_vals,
            "fc_ci": fc_ci,
            "intraday": intraday,
            "ticker": sel,
            "chart": chart,
            "hour_range": hour_range,
            "run_all": True
        })

    if st.session_state.run_all and st.session_state.ticker == sel:
        df = st.session_state.df_hist
        df_ohlc = st.session_state.df_ohlc

        last_price = _safe_last_float(df)
        p_up = np.mean(st.session_state.fc_vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        # Pre-fetch Forex news (intraday only)
        fx_news = pd.DataFrame()
        if mode == "Forex" and show_fx_news:
            fx_news = fetch_yf_news(sel, window_days=news_window_days)

        # ----- Daily -----
        if chart in ("Daily","Both"):
            df_show = df[-360:]
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)

            piv = current_daily_pivots(df_ohlc)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{sel} Daily — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30[-360:], "--", label="30 EMA")
            ax.plot(res30[-360:], ":", label="30 Resistance")
            ax.plot(sup30[-360:], ":", label="30 Support")

            if not yhat_d.empty:
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema30.empty:
                ax.plot(yhat_ema30.index, yhat_ema30.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            if piv:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            r30_last = float(res30.iloc[-1]); s30_last = float(sup30.iloc[-1])
            ax.text(df_show.index[-1], r30_last, f"  30R = {fmt_price_val(r30_last)}", va="bottom")
            ax.text(df_show.index[-1], s30_last, f"  30S = {fmt_price_val(s30_last)}", va="top")

            ax.set_xlabel("Date (PST)")
            ax.legend(loc="lower left", framealpha=0.5)
            xlim_daily = ax.get_xlim()
            st.pyplot(fig)

            # --- Normalized Elliott Wave panel (Daily) aligned with daily chart x-axis + signals ---
            wave_norm_d, piv_df_d = compute_normalized_elliott_wave(df, pivot_lb=pivot_lookback_d, norm_win=norm_window_d)
            figdw, axdw = plt.subplots(figsize=(14,2.6))
            plt.subplots_adjust(top=0.88, right=0.93)

            axdw.set_title("Daily Normalized Elliott Wave (tanh(z-score) & swing pivots)")
            axdw.plot(wave_norm_d.index, wave_norm_d, label="Norm EW (Daily)", linewidth=1.8)
            axdw.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
            # NEW lines:
            axdw.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
            axdw.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")

            axdw.set_ylim(-1.1, 1.1)
            axdw.set_xlabel("Date (PST)")
            axdw.set_xlim(xlim_daily)

            # Annotate most recent pivots
            if not piv_df_d.empty:
                show_df_d = piv_df_d.tail(int(waves_to_annotate_d))
                for _, r in show_df_d.iterrows():
                    t = r["time"]; w = r["wave"]; typ = r["type"]
                    ylab = 0.9 if typ == 'H' else -0.9
                    axdw.annotate(str(int(w)), (t, ylab),
                                  xytext=(0, 0), textcoords="offset points",
                                  ha="center", va="center",
                                  fontsize=9, fontweight="bold")

            # Price + EW signal (top-right inside chart area)
            px_daily = _safe_last_float(df)
            ew_sig_d = elliott_conf_signal(px_daily, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw = axdw.get_position()
            label_txt = f"Price: {fmt_price_val(px_daily)}"
            if ew_sig_d is not None:
                side = ew_sig_d['side']
                prob = fmt_pct(ew_sig_d['prob'], digits=0)
                label_txt += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily)}  ({prob})"
            figdw.text(posdw.x1, posdw.y1 + 0.01, label_txt, ha="right", va="bottom",
                       fontsize=10, fontweight="bold")

            axdw.legend(loc="lower left", framealpha=0.5)
            st.pyplot(figdw)

        # ----- Hourly (no triangles; title shows ▲ / ▼ with values) -----
        if chart in ("Hourly","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                hc = intr["Close"].ffill()
                he = hc.ewm(span=20).mean()
                xh = np.arange(len(hc))
                slope_h, intercept_h = np.polyfit(xh, hc.values, 1)
                trend_h = slope_h * xh + intercept_h
                # rolling S/R just for latest level computation
                res_h = hc.rolling(60, min_periods=1).max()
                sup_h = hc.rolling(60, min_periods=1).min()

                # Supertrend from intraday OHLC
                st_intraday = compute_supertrend(intr, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday else pd.Series(index=hc.index, dtype=float)

                # Slope on hourly close
                yhat_h, m_h = slope_line(hc, slope_lb_hourly)

                fig2, ax2 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax2.plot(hc.index, hc, label="Intraday")
                ax2.plot(hc.index, he, "--", label="20 EMA")
                ax2.plot(hc.index, trend_h, "--", label="Trend", linewidth=2)

                # STRAIGHT Support/Resistance lines across entire chart
                res_val = np.nan
                sup_val = np.nan
                px_val  = np.nan
                try:
                    res_val = float(res_h.iloc[-1])
                    sup_val = float(sup_h.iloc[-1])
                    px_val  = float(hc.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val) and np.isfinite(sup_val):
                    ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1],
                               colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
                    label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

                # Dynamic title area
                buy_sell_text = ""
                if np.isfinite(sup_val):
                    buy_sell_text += f" — ▲ BUY @{fmt_price_val(sup_val)}"
                if np.isfinite(res_val):
                    buy_sell_text += f"  ▼ SELL @{fmt_price_val(res_val)}"
                ax2.set_title(f"{sel} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text}")

                # Current price label OUTSIDE (top-right above axes)
                if np.isfinite(px_val):
                    pos = ax2.get_position()
                    fig2.text(
                        pos.x1, pos.y1 + 0.02,
                        f"Current price: {fmt_price_val(px_val)}",
                        ha="right", va="bottom",
                        fontsize=11, fontweight="bold"
                    )

                if not st_line_intr.dropna().empty:
                    ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                if show_fibs and not hc.empty:
                    fibs_h = fibonacci_levels(hc)
                    for lbl, y in fibs_h.items():
                        ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax2.text(hc.index[-1], y, f" {lbl}", va="center")

                if mode == "Forex" and show_fx_news and not hc.empty and 'time' in fx_news:
                    t0, t1 = hc.index[0], hc.index[-1]
                    times = [t for t in fx_news["time"] if t0 <= t <= t1]
                    if times:
                        draw_news_markers(ax2, times, float(hc.min()), float(hc.max()), label="News")

                # Signal (text only)
                signal = sr_proximity_signal(hc, res_h, sup_h, st.session_state.fc_vals,
                                             threshold=signal_threshold, prox=sr_prox_pct)
                if signal is not None and np.isfinite(px_val):
                    if signal["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal['level'])} — {signal['reason']}")
                    elif signal["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal['level'])} — {signal['reason']}")

                ax2.set_xlabel("Time (PST)")
                ax2.legend(loc="lower left", framealpha=0.5)
                xlim_price = ax2.get_xlim()
                st.pyplot(fig2)

                # Momentum panel (ROC%)
                if show_mom_hourly:
                    roc = compute_roc(hc, n=mom_lb_hourly)
                    res_m = roc.rolling(60, min_periods=1).max()
                    sup_m = roc.rolling(60, min_periods=1).min()
                    fig2m, ax2m = plt.subplots(figsize=(14,2.8))
                    ax2m.set_title(f"Momentum (ROC% over {mom_lb_hourly} bars)")
                    ax2m.plot(roc.index, roc, label=f"ROC%({mom_lb_hourly})")
                    yhat_m, m_m = slope_line(roc, slope_lb_hourly)
                    if not yhat_m.empty:
                        ax2m.plot(yhat_m.index, yhat_m.values, "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m)}%/bar)")
                    ax2m.plot(res_m.index, res_m, ":", label="Mom Resistance")
                    ax2m.plot(sup_m.index, sup_m, ":", label="Mom Support")
                    ax2m.axhline(0, linestyle="--", linewidth=1)
                    ax2m.set_xlabel("Time (PST)")
                    ax2m.legend(loc="lower left", framealpha=0.5)
                    ax2m.set_xlim(xlim_price)
                    st.pyplot(fig2m)

                # --- Normalized Elliott Wave panel (Hourly) + signals ---
                wave_norm, piv_df = compute_normalized_elliott_wave(hc, pivot_lb=pivot_lookback, norm_win=norm_window)
                fig2w, ax2w = plt.subplots(figsize=(14,2.6))
                plt.subplots_adjust(top=0.88, right=0.93)

                ax2w.set_title("Normalized Elliott Wave (tanh(z-score) & swing pivots)")
                ax2w.plot(wave_norm.index, wave_norm, label="Norm EW", linewidth=1.8)
                ax2w.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
                # NEW lines:
                ax2w.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
                ax2w.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")

                ax2w.set_ylim(-1.1, 1.1)
                ax2w.set_xlabel("Time (PST)")
                ax2w.set_xlim(xlim_price)

                if not piv_df.empty:
                    show_df = piv_df.tail(int(waves_to_annotate))
                    for _, r in show_df.iterrows():
                        t = r["time"]; w = r["wave"]; typ = r["type"]
                        ylab = 0.9 if typ == 'H' else -0.9
                        ax2w.annotate(str(int(w)), (t, ylab),
                                      xytext=(0, 0), textcoords="offset points",
                                      ha="center", va="center",
                                      fontsize=9, fontweight="bold")

                # Price + EW signal label
                px_intr = _safe_last_float(hc)
                ew_sig_h = elliott_conf_signal(px_intr, st.session_state.fc_vals, EW_CONFIDENCE)
                pos2w = ax2w.get_position()
                label_txt_h = f"Price: {fmt_price_val(px_intr)}"
                if ew_sig_h is not None:
                    side = ew_sig_h['side']
                    prob = fmt_pct(ew_sig_h['prob'], digits=0)
                    label_txt_h += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_intr)}  ({prob})"
                fig2w.text(pos2w.x1, pos2w.y1 + 0.01, label_txt_h, ha="right", va="bottom",
                           fontsize=10, fontweight="bold")

                ax2w.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig2w)

        if mode == "Forex" and show_fx_news:
            st.subheader("Recent Forex News (Yahoo Finance)")
            if fx_news.empty:
                st.write("No recent news available.")
            else:
                show_cols = fx_news.copy()
                show_cols["time"] = show_cols["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show_cols[["time","publisher","title","link"]].reset_index(drop=True), use_container_width=True)

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
        df_ohlc = st.session_state.df_ohlc
        idx, vals, ci = (
            st.session_state.fc_idx,
            st.session_state.fc_vals,
            st.session_state.fc_ci
        )
        last_price = _safe_last_float(df)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.caption(f"Intraday lookback: **{st.session_state.get('hour_range','24h')}** (change in 'Original Forecast' tab and rerun)")

        view = st.radio("View:", ["Daily","Intraday","Both"], key="enh_view")

        if view in ("Daily","Both"):
            df_show = df[-360:]
            ema30 = df.ewm(span=30).mean()
            res30 = df.rolling(30, min_periods=1).max()
            sup30 = df.rolling(30, min_periods=1).min()
            yhat_d, m_d = slope_line(df, slope_lb_daily)
            yhat_ema30, m_ema30 = slope_line(ema30, slope_lb_daily)

            piv = current_daily_pivots(df_ohlc)

            fig, ax = plt.subplots(figsize=(14,6))
            ax.set_title(f"{st.session_state.ticker} Daily — History, 30 EMA, 30 S/R, Slope, Pivots")
            ax.plot(df_show, label="History")
            ax.plot(ema30[-360:], "--", label="30 EMA")
            ax.plot(res30[-360:], ":", label="30 Resistance")
            ax.plot(sup30[-360:], ":", label="30 Support")

            if not yhat_d.empty:
                ax.plot(yhat_d.index, yhat_d.values, "-", linewidth=2,
                        label=f"Daily Slope {slope_lb_daily} ({fmt_slope(m_d)}/bar)")
            if not yhat_ema30.empty:
                ax.plot(yhat_ema30.index, yhat_ema30.values, "-", linewidth=2,
                        label=f"EMA30 Slope {slope_lb_daily} ({fmt_slope(m_ema30)}/bar)")

            if piv:
                x0, x1 = df_show.index[0], df_show.index[-1]
                for lbl, y in piv.items():
                    ax.hlines(y, xmin=x0, xmax=x1, linestyles="dashed", linewidth=1.0)
                for lbl, y in piv.items():
                    ax.text(x1, y, f" {lbl} = {fmt_price_val(y)}", va="center")

            r30_last = float(res30.iloc[-1]); s30_last = float(sup30.iloc[-1])
            ax.text(df_show.index[-1], r30_last, f"  30R = {fmt_price_val(r30_last)}", va="bottom")
            ax.text(df_show.index[-1], s30_last, f"  30S = {fmt_price_val(s30_last)}", va="top")

            ax.set_xlabel("Date (PST)")
            ax.legend()
            xlim_daily2 = ax.get_xlim()
            st.pyplot(fig)

            # --- Daily EW panel + signals ---
            wave_norm_d2, piv_df_d2 = compute_normalized_elliott_wave(df, pivot_lb=pivot_lookback_d, norm_win=norm_window_d)
            figdw2, axdw2 = plt.subplots(figsize=(14,2.6))
            plt.subplots_adjust(top=0.88, right=0.93)

            axdw2.set_title("Daily Normalized Elliott Wave (tanh(z-score) & swing pivots)")
            axdw2.plot(wave_norm_d2.index, wave_norm_d2, label="Norm EW (Daily)", linewidth=1.8)
            axdw2.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
            # NEW lines:
            axdw2.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
            axdw2.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")

            axdw2.set_ylim(-1.1, 1.1)
            axdw2.set_xlabel("Date (PST)")
            axdw2.set_xlim(xlim_daily2)

            if not piv_df_d2.empty:
                show_df_d2 = piv_df_d2.tail(int(waves_to_annotate_d))
                for _, r in show_df_d2.iterrows():
                    t = r["time"]; w = r["wave"]; typ = r["type"]
                    ylab = 0.9 if typ == 'H' else -0.9
                    axdw2.annotate(str(int(w)), (t, ylab),
                                   xytext=(0, 0), textcoords="offset points",
                                   ha="center", va="center",
                                   fontsize=9, fontweight="bold")

            # Price + EW signal label
            px_daily2 = _safe_last_float(df)
            ew_sig_d2 = elliott_conf_signal(px_daily2, st.session_state.fc_vals, EW_CONFIDENCE)
            posdw2 = axdw2.get_position()
            label_txt_d2 = f"Price: {fmt_price_val(px_daily2)}"
            if ew_sig_d2 is not None:
                side = ew_sig_d2['side']
                prob = fmt_pct(ew_sig_d2['prob'], digits=0)
                label_txt_d2 += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_daily2)}  ({prob})"
            figdw2.text(posdw2.x1, posdw2.y1 + 0.01, label_txt_d2, ha="right", va="bottom",
                        fontsize=10, fontweight="bold")

            axdw2.legend(loc="lower left", framealpha=0.5)
            st.pyplot(figdw2)

        if view in ("Intraday","Both"):
            intr = st.session_state.intraday
            if intr is None or intr.empty or "Close" not in intr:
                st.warning("No intraday data available.")
            else:
                ic = intr["Close"].ffill()
                ie = ic.ewm(span=20).mean()
                xi = np.arange(len(ic))
                slope_i, intercept_i = np.polyfit(xi, ic.values, 1)
                trend_i = slope_i * xi + intercept_i
                # rolling for latest level only
                res_i = ic.rolling(60, min_periods=1).max()
                sup_i = ic.rolling(60, min_periods=1).min()
                st_intraday = compute_supertrend(intr, atr_period=atr_period, atr_mult=atr_mult)
                st_line_intr = st_intraday["ST"].reindex(ic.index) if "ST" in st_intraday else pd.Series(index=ic.index, dtype=float)
                yhat_h, m_h = slope_line(ic, slope_lb_hourly)

                fig3, ax3 = plt.subplots(figsize=(14,4))
                plt.subplots_adjust(top=0.85, right=0.93)

                ax3.plot(ic.index, ic, label="Intraday")
                ax3.plot(ic.index, ie, "--", label="20 EMA")
                ax3.plot(ic.index, trend_i, "--", label="Trend", linewidth=2)

                # STRAIGHT Support/Resistance lines
                res_val2 = np.nan
                sup_val2 = np.nan
                px_val2  = np.nan
                try:
                    res_val2 = float(res_i.iloc[-1])
                    sup_val2 = float(sup_i.iloc[-1])
                    px_val2  = float(ic.iloc[-1])
                except Exception:
                    pass

                if np.isfinite(res_val2) and np.isfinite(sup_val2):
                    ax3.hlines(res_val2, xmin=ic.index[0], xmax=ic.index[-1],
                               colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
                    ax3.hlines(sup_val2, xmin=ic.index[0], xmax=ic.index[-1],
                               colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
                    label_on_left(ax3, res_val2, f"R {fmt_price_val(res_val2)}", color="tab:red")
                    label_on_left(ax3, sup_val2, f"S {fmt_price_val(sup_val2)}", color="tab:green")

                buy_sell_text2 = ""
                if np.isfinite(sup_val2):
                    buy_sell_text2 += f" — ▲ BUY @{fmt_price_val(sup_val2)}"
                if np.isfinite(res_val2):
                    buy_sell_text2 += f"  ▼ SELL @{fmt_price_val(res_val2)}"
                ax3.set_title(f"{st.session_state.ticker} Intraday ({st.session_state.hour_range})  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}{buy_sell_text2}")

                if np.isfinite(px_val2):
                    pos2 = ax3.get_position()
                    fig3.text(
                        pos2.x1, pos2.y1 + 0.02,
                        f"Current price: {fmt_price_val(px_val2)}",
                        ha="right", va="bottom",
                        fontsize=11, fontweight="bold"
                    )

                if not st_line_intr.dropna().empty:
                    ax3.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")
                if not yhat_h.empty:
                    ax3.plot(yhat_h.index, yhat_h.values, "-", linewidth=2,
                             label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")

                if show_fibs and not ic.empty:
                    fibs_h = fibonacci_levels(ic)
                    for lbl, y in fibs_h.items():
                        ax3.hlines(y, xmin=ic.index[0], xmax=ic.index[-1], linestyles="dotted", linewidth=1)
                    for lbl, y in fibs_h.items():
                        ax3.text(ic.index[-1], y, f" {lbl}", va="center")

                signal2 = sr_proximity_signal(ic, res_i, sup_i, st.session_state.fc_vals,
                                              threshold=signal_threshold, prox=sr_prox_pct)
                if signal2 is not None and np.isfinite(px_val2):
                    if signal2["side"] == "BUY":
                        st.success(f"**BUY** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")
                    elif signal2["side"] == "SELL":
                        st.error(f"**SELL** @ {fmt_price_val(signal2['level'])} — {signal2['reason']}")

                ax3.set_xlabel("Time (PST)")
                ax3.legend(loc="lower left", framealpha=0.5)
                xlim_price2 = ax3.get_xlim()
                st.pyplot(fig3)

                # Momentum panel (ROC%)
                if show_mom_hourly:
                    roc_i = compute_roc(ic, n=mom_lb_hourly)
                    res_m2 = roc_i.rolling(60, min_periods=1).max()
                    sup_m2 = roc_i.rolling(60, min_periods=1).min()
                    fig3m, ax3m = plt.subplots(figsize=(14,2.8))
                    ax3m.set_title(f"Momentum (ROC% over {mom_lb_hourly} bars)")
                    ax3m.plot(roc_i.index, roc_i, label=f"ROC%({mom_lb_hourly})")
                    yhat_m2, m_m2 = slope_line(roc_i, slope_lb_hourly)
                    if not yhat_m2.empty:
                        ax3m.plot(yhat_m2.index, yhat_m2.values, "--", linewidth=2, label=f"Trend {slope_lb_hourly} ({fmt_slope(m_m2)}%/bar)")
                    ax3m.plot(res_m2.index, res_m2, ":", label="Mom Resistance")
                    ax3m.plot(sup_m2.index, sup_m2, ":", label="Mom Support")
                    ax3m.axhline(0, linestyle="--", linewidth=1)
                    ax3m.set_xlabel("Time (PST)")
                    ax3m.legend(loc="lower left", framealpha=0.5)
                    ax3m.set_xlim(xlim_price2)
                    st.pyplot(fig3m)

                # --- Hourly EW panel + signals ---
                wave_norm2, piv_df2 = compute_normalized_elliott_wave(ic, pivot_lb=pivot_lookback, norm_win=norm_window)
                fig3w, ax3w = plt.subplots(figsize=(14,2.6))
                plt.subplots_adjust(top=0.88, right=0.93)

                ax3w.set_title("Normalized Elliott Wave (tanh(z-score) & swing pivots)")
                ax3w.plot(wave_norm2.index, wave_norm2, label="Norm EW", linewidth=1.8)
                ax3w.axhline(0.0, linestyle="--", linewidth=1, label="EW 0")
                # NEW lines:
                ax3w.axhline(0.5, color="tab:red", linestyle="-", linewidth=1, label="EW +0.5")
                ax3w.axhline(-0.5, color="tab:green", linestyle="-", linewidth=1, label="EW -0.5")

                ax3w.set_ylim(-1.1, 1.1)
                ax3w.set_xlabel("Time (PST)")
                ax3w.set_xlim(xlim_price2)

                if not piv_df2.empty:
                    show_df2 = piv_df2.tail(int(waves_to_annotate))
                    for _, r in show_df2.iterrows():
                        t = r["time"]; w = r["wave"]; typ = r["type"]
                        ylab = 0.9 if typ == 'H' else -0.9
                        ax3w.annotate(str(int(w)), (t, ylab),
                                      xytext=(0, 0), textcoords="offset points",
                                      ha="center", va="center",
                                      fontsize=9, fontweight="bold")

                # Price + EW signal label
                px_intr2 = _safe_last_float(ic)
                ew_sig_h2 = elliott_conf_signal(px_intr2, st.session_state.fc_vals, EW_CONFIDENCE)
                pos3w = ax3w.get_position()
                label_txt_h2 = f"Price: {fmt_price_val(px_intr2)}"
                if ew_sig_h2 is not None:
                    side = ew_sig_h2['side']
                    prob = fmt_pct(ew_sig_h2['prob'], digits=0)
                    label_txt_h2 += f"  |  {('▲ BUY' if side=='BUY' else '▼ SELL')} @ {fmt_price_val(px_intr2)}  ({prob})"
                fig3w.text(pos3w.x1, pos3w.y1 + 0.01, label_txt_h2, ha="right", va="bottom",
                           fontsize=10, fontweight="bold")

                ax3w.legend(loc="lower left", framealpha=0.5)
                st.pyplot(fig3w)

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
        last_price = _safe_last_float(df_hist)
        idx, vals, ci = compute_sarimax_forecast(df_hist)
        p_up = np.mean(vals.to_numpy() > last_price) if np.isfinite(last_price) else np.nan
        p_dn = 1 - p_up if np.isfinite(p_up) else np.nan

        st.subheader(f"Last 3 Months  ↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}")
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
