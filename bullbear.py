# /mount/src/stock-wizard/bullbear.py
# =========================
# Batch 1/3 — bullbear.py (UPDATED: Remove requested tabs + add HMA Buy + add NPX 0.0 Cross tab)
# Includes: beginning through Tabs 1–4
# =========================

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
from matplotlib.lines import Line2D

# ---------------------------
# Matplotlib theme (STYLE ONLY — no logic changes)
# ---------------------------
def _apply_mpl_theme():
    """A clean, modern look for matplotlib output rendered in Streamlit (no data/logic changes)."""
    try:
        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "legend.fontsize": 9,
            "legend.framealpha": 0.70,
            "legend.fancybox": True,
            "lines.linewidth": 1.6,
        })
    except Exception:
        pass

_apply_mpl_theme()

# ---------------------------
# Page config + UI CSS
# ---------------------------
st.set_page_config(
    page_title="📊 Dashboard & Forecasts",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  #MainMenu, header, footer {visibility: hidden;}

  /* =========================
     Beautiful rectangular ribbon tabs (BaseWeb tabs)
     ========================= */
  div[data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    overflow-x: visible !important;
    gap: 0.45rem !important;
    padding: 0.35rem 0.35rem 0.25rem 0.35rem !important;
    border-bottom: 1px solid rgba(49, 51, 63, 0.18) !important;
  }
  div[data-baseweb="tab"] { flex: 0 0 auto !important; }

  div[data-baseweb="tab"] > button,
  div[data-baseweb="tab"] button {
    border: 1px solid rgba(49, 51, 63, 0.22) !important;
    background: rgba(255,255,255,0.92) !important;
    padding: 0.45rem 0.75rem !important;
    border-radius: 6px !important;       /* rectangular ribbon (not pill) */
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease !important;
    white-space: nowrap !important;
  }
  div[data-baseweb="tab"] > button:hover,
  div[data-baseweb="tab"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.10) !important;
  }
  div[data-baseweb="tab"] > button[aria-selected="true"],
  div[data-baseweb="tab"] button[aria-selected="true"] {
    background: rgba(49, 51, 63, 0.92) !important;
    color: white !important;
    border-color: rgba(49, 51, 63, 0.92) !important;
    box-shadow: 0 10px 22px rgba(0,0,0,0.16) !important;
  }
  div[data-baseweb="tab"] > button:focus,
  div[data-baseweb="tab"] button:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(49, 51, 63, 0.18) !important;
  }

  /* =========================
     Beautiful chart container styling (Streamlit React UI wrappers)
     ========================= */
  div[data-testid="stImage"] {
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 14px;
    background: rgba(255,255,255,0.65);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    padding: 0.35rem 0.35rem 0.15rem 0.35rem;
    overflow: hidden;
  }
  div[data-testid="stImage"] img {
    border-radius: 12px;
  }

  /* Mobile: keep sidebar usable */
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

# ---------------------------
# Auto-refresh (PST)
# ---------------------------
REFRESH_INTERVAL = 120  # seconds
PACIFIC = pytz.timezone("US/Pacific")

def auto_refresh():
    if "last_refresh" not in st.session_state:
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

# ---------------------------
# Mode buttons (Forex / Stocks)
# ---------------------------
def _reset_run_state_for_mode_switch():
    """
    When switching modes, reset run state so:
      • selectbox keys don't crash due to old values not in new universe
      • charts/forecast don't show stale data
    """
    st.session_state.run_all = False
    st.session_state.ticker = None
    st.session_state.pop("df_hist", None)
    st.session_state.pop("df_ohlc", None)
    st.session_state.pop("fc_idx", None)
    st.session_state.pop("fc_vals", None)
    st.session_state.pop("fc_ci", None)
    st.session_state.pop("intraday", None)
    st.session_state.pop("mode_at_run", None)

if "asset_mode" not in st.session_state:
    st.session_state.asset_mode = "Forex"  # default

st.title("📊 Dashboard & Forecasts")
mcol1, mcol2 = st.columns(2)

if mcol1.button("🌐 Forex", use_container_width=True, key="btn_mode_forex"):
    if st.session_state.asset_mode != "Forex":
        st.session_state.asset_mode = "Forex"
        _reset_run_state_for_mode_switch()
        try:
            st.experimental_rerun()
        except Exception:
            pass

if mcol2.button("📈 Stocks", use_container_width=True, key="btn_mode_stock"):
    if st.session_state.asset_mode != "Stock":
        st.session_state.asset_mode = "Stock"
        _reset_run_state_for_mode_switch()
        try:
            st.experimental_rerun()
        except Exception:
            pass

mode = st.session_state.asset_mode
st.caption(f"**Current mode:** {mode}")

# ---------------------------
# Aesthetic helper (STYLE ONLY — no logic change)
# ---------------------------
def style_axes(ax):
    """Simple, consistent, user-friendly chart styling (no data/logic changes)."""
    try:
        ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.07, linewidth=0.6)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_alpha(0.25)

        ax.tick_params(axis="both", which="major", length=4, width=0.9, colors="0.25")
        ax.tick_params(axis="both", which="minor", length=2, width=0.6, colors="0.35")
        ax.margins(x=0.01)
    except Exception:
        pass

# ---------------------------
# Core helpers
# ---------------------------
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

def fmt_pct(x, digits: int = 1) -> str:
    try:
        xv = float(x)
    except Exception:
        return "n/a"
    return f"{xv:.{digits}%}" if np.isfinite(xv) else "n/a"

def fmt_price_val(y: float) -> str:
    try:
        y = float(y)
    except Exception:
        return "n/a"
    return f"{y:,.3f}"

def fmt_slope(m: float) -> str:
    try:
        mv = float(np.squeeze(m))
    except Exception:
        return "n/a"
    return f"{mv:.4f}" if np.isfinite(mv) else "n/a"

def fmt_r2(r2: float, digits: int = 1) -> str:
    try:
        rv = float(r2)
    except Exception:
        return "n/a"
    return fmt_pct(rv, digits=digits) if np.isfinite(rv) else "n/a"

def label_on_left(ax, y_val: float, text: str, color: str = "black", fontsize: int = 9):
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.01, y_val, text, transform=trans,
        ha="left", va="center", color=color, fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        zorder=6
    )

def subset_by_daily_view(obj, view_label: str):
    if obj is None or len(obj.index) == 0:
        return obj
    idx = obj.index
    end = idx.max()
    days_map = {"6M": 182, "12M": 365, "24M": 730}
    if view_label == "Historical":
        start = idx.min()
    else:
        start = end - pd.Timedelta(days=days_map.get(view_label, 365))
    return obj.loc[(idx >= start) & (idx <= end)]

def _cross_series(price: pd.Series, line: pd.Series):
    p = _coerce_1d_series(price)
    l = _coerce_1d_series(line)
    ok = p.notna() & l.notna()
    if ok.sum() < 2:
        idx = p.index if len(p) else l.index
        return pd.Series(False, index=idx), pd.Series(False, index=idx)
    p = p[ok]
    l = l[ok]
    above = p > l
    cross_up = above & (~above.shift(1).fillna(False))
    cross_dn = (~above) & (above.shift(1).fillna(False))
    return cross_up.reindex(p.index, fill_value=False), cross_dn.reindex(p.index, fill_value=False)

# ---------------------------
# Sidebar configuration (kept in same spirit; News controls removed as requested)
# ---------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

if st.sidebar.button("🧹 Clear cache (data + run state)", use_container_width=True, key="btn_clear_cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _reset_run_state_for_mode_switch()
    try:
        st.experimental_rerun()
    except Exception:
        pass

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

show_fibs = st.sidebar.checkbox("Show Fibonacci", value=True, key="sb_show_fibs")

slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 12, 480, 120, 6, key="sb_slope_lb_hourly")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")

st.sidebar.subheader("Normalized Trend/Price (NTD/NPX)")
show_ntd = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd_v2")
ntd_window = st.sidebar.slider("NTD/NPX window", 10, 300, 60, 5, key="sb_ntd_win")

st.sidebar.subheader("Normalized Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

# Sessions (PST) for Forex (kept)
show_sessions_pst = False
if mode == "Forex":
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")

# ---------------------------
# Universe
# ---------------------------
if mode == "Stock":
    universe = sorted([
        "AAPL","SPY","AMZN","DIA","TSLA","SPGI","JPM","VTWG","PLTR","NVDA",
        "META","SITM","MARA","GOOG","HOOD","BABA","IBM","AVGO","GUSH","VOO",
        "MSFT","TSM","NFLX","MP","AAL","URI","DAL","BBAI","QUBT","AMD","SMCI",
        "ORCL","TLT"
    ])
else:
    universe = [
        "EURUSD=X","EURJPY=X","GBPUSD=X","USDJPY=X","AUDUSD=X","NZDUSD=X","CADJPY=X","USDCHF=X",
        "HKDJPY=X","USDCAD=X","USDCNY=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
        "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X","GBPCAD=X"
    ]

# ---------------------------
# Data fetchers
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    s = df["Close"].asfreq("D").ffill()
    if isinstance(s.index, pd.DatetimeIndex):
        try:
            if s.index.tz is None:
                s.index = s.index.tz_localize(PACIFIC)
            else:
                s.index = s.index.tz_convert(PACIFIC)
        except Exception:
            pass
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[
        ["Open","High","Low","Close"]
    ].dropna()
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(PACIFIC)
            else:
                df.index = df.index.tz_convert(PACIFIC)
        except Exception:
            pass
    return df

def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open", "High", "Low", "Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
    """
    Remove price gaps at session breaks by applying a cumulative offset so that
    the first bar after a large time-gap STARTS (Open) at the previous bar's Close.
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if "Close" not in df.columns:
        return df

    ref_col = "Open" if "Open" in df.columns else "Close"
    close = pd.to_numeric(df["Close"], errors="coerce")
    refp  = pd.to_numeric(df[ref_col], errors="coerce")

    idx = close.index
    diffs = idx.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return df
    expected = float(np.nanmedian(diffs))
    if not np.isfinite(expected) or expected <= 0:
        return df

    thr = max(expected * float(gap_mult), float(min_gap_seconds))
    offsets = np.zeros(len(close), dtype=float)
    offset = 0.0

    for i in range(1, len(close)):
        try:
            dt_sec = float((idx[i] - idx[i-1]).total_seconds())
        except Exception:
            dt_sec = 0.0

        if dt_sec >= thr:
            prev_close = float(close.iloc[i-1]) if np.isfinite(close.iloc[i-1]) else np.nan
            curr_ref   = float(refp.iloc[i])    if np.isfinite(refp.iloc[i])    else np.nan
            if np.isfinite(prev_close) and np.isfinite(curr_ref):
                offset += (curr_ref - prev_close)

        offsets[i] = offset

    offs = pd.Series(offsets, index=idx)
    out = df.copy()
    for c in price_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") - offs
    return out

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "5d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
        except Exception:
            pass
        try:
            df.index = df.index.tz_convert(PACIFIC)
        except Exception:
            pass

    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)

    return df

# ---------------------------
# Forecast (SARIMAX)
# ---------------------------
@st.cache_data(ttl=120)
def compute_sarimax_forecast(series_like, steps: int = 30):
    series = _coerce_1d_series(series_like).dropna()
    if series.empty:
        idx = pd.date_range(pd.Timestamp.now(tz=PACIFIC), periods=steps, freq="D")
        return idx, pd.Series(index=idx, dtype=float), pd.DataFrame(index=idx, columns=["lower Close","upper Close"])
    if isinstance(series.index, pd.DatetimeIndex):
        try:
            if series.index.tz is None:
                series.index = series.index.tz_localize(PACIFIC)
            else:
                series.index = series.index.tz_convert(PACIFIC)
        except Exception:
            pass
    try:
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    except Exception:
        model = SARIMAX(
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=int(steps))
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=int(steps), freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

# ---------------------------
# Regression & ±2σ band
# ---------------------------
def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    """
    Linear regression on last `lookback` bars with:
      • fitted trendline
      • symmetric ±z·σ band (σ = std of residuals)
      • R² of the fit
    """
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-int(lookback):]
    if s.shape[0] < 3:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty, float("nan"), float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat
    dof = max(len(y) - 2, 1)
    std = float(np.sqrt(np.sum(resid**2) / dof))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res/ss_tot)
    yhat_s = pd.Series(yhat, index=s.index)
    upper_s = pd.Series(yhat + float(z) * std, index=s.index)
    lower_s = pd.Series(yhat - float(z) * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

# ---------------------------
# Indicators
# ---------------------------
def compute_bbands(close: pd.Series, window: int = 20, mult: float = 2.0, use_ema: bool = False):
    s = _coerce_1d_series(close).astype(float)
    idx = s.index
    if s.empty or window < 2 or not np.isfinite(mult):
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty, empty
    minp = max(2, window // 2)
    mid = s.ewm(span=window, adjust=False).mean() if use_ema else s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower).replace(0, np.nan)
    pctb = ((s - lower) / width).clip(0.0, 1.0)
    nbb = pctb * 2.0 - 1.0
    return (mid.reindex(s.index), upper.reindex(s.index), lower.reindex(s.index), pctb.reindex(s.index), nbb.reindex(s.index))

def compute_normalized_trend(close: pd.Series, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window // 3)

    def _slope(y: pd.Series) -> float:
        y = pd.Series(y).dropna()
        if len(y) < 3:
            return np.nan
        x = np.arange(len(y), dtype=float)
        try:
            m, _ = np.polyfit(x, y.to_numpy(dtype=float), 1)
        except Exception:
            return np.nan
        return float(m)

    slope_roll = s.rolling(window, min_periods=minp).apply(_slope, raw=False)
    vol = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    ntd_raw = (slope_roll * window) / vol
    return np.tanh(ntd_raw / 2.0).reindex(s.index)

def compute_normalized_price(close: pd.Series, window: int = 60) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or window < 3:
        return pd.Series(index=s.index, dtype=float)
    minp = max(5, window // 3)
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std().replace(0, np.nan)
    z = (s - m) / sd
    return np.tanh(z / 2.0).reindex(s.index)

def _wma(s: pd.Series, window: int) -> pd.Series:
    s = _coerce_1d_series(s).astype(float)
    if s.empty or window < 1:
        return pd.Series(index=s.index, dtype=float)
    w = np.arange(1, window + 1, dtype=float)
    return s.rolling(window, min_periods=window).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def compute_hma(close: pd.Series, period: int = 55) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)
    half = max(1, int(period / 2))
    sqrtp = max(1, int(np.sqrt(period)))
    wma_half = _wma(s, half)
    wma_full = _wma(s, period)
    diff = 2 * wma_half - wma_full
    hma = _wma(diff, sqrtp)
    return hma.reindex(s.index)

# ---------------------------
# Ichimoku / Supertrend / PSAR
# ---------------------------
def ichimoku_lines(high: pd.Series, low: pd.Series, close: pd.Series,
                   conv: int = 9, base: int = 26, span_b: int = 52, shift_cloud: bool = True):
    h = _coerce_1d_series(high)
    l = _coerce_1d_series(low)
    c = _coerce_1d_series(close)
    idx = c.index.union(h.index).union(l.index)
    h = h.reindex(idx)
    l = l.reindex(idx)
    c = c.reindex(idx)

    tenkan = ((h.rolling(conv).max() + l.rolling(conv).min()) / 2.0)
    kijun  = ((h.rolling(base).max() + l.rolling(base).min()) / 2.0)
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = ((h.rolling(span_b).max() + l.rolling(span_b).min()) / 2.0)
    if shift_cloud:
        senkou_a = senkou_a.shift(base)
        senkou_b = senkou_b.shift(base)
        chikou   = c.shift(-base)
    else:
        chikou   = c
    return (tenkan.reindex(idx), kijun.reindex(idx), senkou_a.reindex(idx), senkou_b.reindex(idx), chikou.reindex(idx))

def _compute_atr_from_ohlc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.Series(dtype=float)
    high = _coerce_1d_series(df["High"])
    low  = _coerce_1d_series(df["Low"])
    close= _coerce_1d_series(df["Close"])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.reindex(df.index)

def compute_supertrend(df: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
        return pd.DataFrame(columns=["ST","in_uptrend"])
    ohlc = df[["High","Low","Close"]].copy()
    atr = _compute_atr_from_ohlc(ohlc, period=int(atr_period))
    hl2 = (ohlc["High"] + ohlc["Low"]) / 2.0
    upperband = hl2 + float(atr_mult) * atr
    lowerband = hl2 - float(atr_mult) * atr

    st_line = pd.Series(index=ohlc.index, dtype=float)
    in_uptrend = pd.Series(index=ohlc.index, dtype=bool)

    for i in range(len(ohlc)):
        if i == 0:
            in_uptrend.iloc[i] = True
            st_line.iloc[i] = lowerband.iloc[i]
            continue

        if ohlc["Close"].iloc[i] > upperband.iloc[i-1]:
            in_uptrend.iloc[i] = True
        elif ohlc["Close"].iloc[i] < lowerband.iloc[i-1]:
            in_uptrend.iloc[i] = False
        else:
            in_uptrend.iloc[i] = in_uptrend.iloc[i-1]
            if in_uptrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if (not in_uptrend.iloc[i]) and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]

        st_line.iloc[i] = lowerband.iloc[i] if in_uptrend.iloc[i] else upperband.iloc[i]

    return pd.DataFrame({"ST": st_line, "in_uptrend": in_uptrend})

def compute_psar_from_ohlc(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    if df is None or df.empty or not {"High","Low"}.issubset(df.columns):
        return pd.DataFrame(columns=["PSAR","in_uptrend"])
    high = _coerce_1d_series(df["High"])
    low  = _coerce_1d_series(df["Low"])
    idx = high.index.union(low.index)
    high = high.reindex(idx)
    low  = low.reindex(idx)

    psar = pd.Series(index=idx, dtype=float)
    in_uptrend = pd.Series(index=idx, dtype=bool)

    in_uptrend.iloc[0] = True
    psar.iloc[0] = float(low.iloc[0]) if np.isfinite(low.iloc[0]) else np.nan
    ep = float(high.iloc[0]) if np.isfinite(high.iloc[0]) else np.nan
    af = float(step)

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i-1]
        if in_uptrend.iloc[i-1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i],
                               float(low.iloc[i-1]) if np.isfinite(low.iloc[i-1]) else psar.iloc[i],
                               float(low.iloc[i-2]) if (i >= 2 and np.isfinite(low.iloc[i-2])) else float(low.iloc[i-1]) if np.isfinite(low.iloc[i-1]) else psar.iloc[i])
            if np.isfinite(high.iloc[i]) and (not np.isfinite(ep) or high.iloc[i] > ep):
                ep = float(high.iloc[i])
                af = min(af + float(step), float(max_step))
            if np.isfinite(low.iloc[i]) and np.isfinite(psar.iloc[i]) and low.iloc[i] < psar.iloc[i]:
                in_uptrend.iloc[i] = False
                psar.iloc[i] = ep
                ep = float(low.iloc[i]) if np.isfinite(low.iloc[i]) else ep
                af = float(step)
            else:
                in_uptrend.iloc[i] = True
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = max(psar.iloc[i],
                               float(high.iloc[i-1]) if np.isfinite(high.iloc[i-1]) else psar.iloc[i],
                               float(high.iloc[i-2]) if (i >= 2 and np.isfinite(high.iloc[i-2])) else float(high.iloc[i-1]) if np.isfinite(high.iloc[i-1]) else psar.iloc[i])
            if np.isfinite(low.iloc[i]) and (not np.isfinite(ep) or low.iloc[i] < ep):
                ep = float(low.iloc[i])
                af = min(af + float(step), float(max_step))
            if np.isfinite(high.iloc[i]) and np.isfinite(psar.iloc[i]) and high.iloc[i] > psar.iloc[i]:
                in_uptrend.iloc[i] = True
                psar.iloc[i] = ep
                ep = float(high.iloc[i]) if np.isfinite(high.iloc[i]) else ep
                af = float(step)
            else:
                in_uptrend.iloc[i] = False

    return pd.DataFrame({"PSAR": psar, "in_uptrend": in_uptrend})

# ---------------------------
# Sessions (PST)
# ---------------------------
NY_TZ = pytz.timezone("America/New_York")
LDN_TZ = pytz.timezone("Europe/London")

def session_markers_for_index(idx: pd.DatetimeIndex, session_tz, open_hr: int, close_hr: int):
    opens, closes = [], []
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.empty:
        return opens, closes
    start_d = idx[0].astimezone(session_tz).date()
    end_d   = idx[-1].astimezone(session_tz).date()
    rng = pd.date_range(start=start_d, end=end_d, freq="D")
    lo, hi = idx.min(), idx.max()
    for d in rng:
        try:
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0), is_dst=None)
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0), is_dst=None)
        except Exception:
            dt_open_local  = session_tz.localize(datetime(d.year, d.month, d.day, open_hr, 0, 0))
            dt_close_local = session_tz.localize(datetime(d.year, d.month, d.day, close_hr, 0, 0))
        dt_open_pst  = dt_open_local.astimezone(PACIFIC)
        dt_close_pst = dt_close_local.astimezone(PACIFIC)
        if lo <= dt_open_pst <= hi:
            opens.append(dt_open_pst)
        if lo <= dt_close_pst <= hi:
            closes.append(dt_close_pst)
    return opens, closes

def compute_session_lines(idx: pd.DatetimeIndex):
    ldn_open, ldn_close = session_markers_for_index(idx, LDN_TZ, 8, 17)
    ny_open, ny_close   = session_markers_for_index(idx, NY_TZ, 8, 17)
    return {"ldn_open": ldn_open, "ldn_close": ldn_close, "ny_open": ny_open, "ny_close": ny_close}

def draw_session_lines(ax, lines: dict, alpha: float = 0.35):
    for t in lines.get("ldn_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ldn_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:blue", alpha=alpha)
    for t in lines.get("ny_open", []):
        ax.axvline(t, linestyle="-", linewidth=1.0, color="tab:orange", alpha=alpha)
    for t in lines.get("ny_close", []):
        ax.axvline(t, linestyle="--", linewidth=1.0, color="tab:orange", alpha=alpha)

    handles = [
        Line2D([0], [0], color="tab:blue",   linestyle="-",  linewidth=1.6, label="London Open"),
        Line2D([0], [0], color="tab:blue",   linestyle="--", linewidth=1.6, label="London Close"),
        Line2D([0], [0], color="tab:orange", linestyle="-",  linewidth=1.6, label="New York Open"),
        Line2D([0], [0], color="tab:orange", linestyle="--", linewidth=1.6, label="New York Close"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels

# ---------------------------
# Fibonacci
# ---------------------------
def fibonacci_levels(series_like):
    s = _coerce_1d_series(series_like).dropna()
    hi = float(s.max()) if not s.empty else np.nan
    lo = float(s.min()) if not s.empty else np.nan
    if not np.isfinite(hi) or not np.isfinite(lo) or hi == lo:
        return {}
    diff = hi - lo
    return {
        "0%": hi,
        "23.6%": hi - 0.236*diff,
        "38.2%": hi - 0.382*diff,
        "50%": hi - 0.5*diff,
        "61.8%": hi - 0.618*diff,
        "78.6%": hi - 0.786*diff,
        "100%": lo
    }

# ---------------------------
# Renderers
# ---------------------------
def render_forecast_chart(symbol: str, close_full: pd.Series):
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty or len(close_full) < 10:
        st.warning("Not enough data to forecast.")
        return

    if "fc_idx" not in st.session_state or st.session_state.get("ticker") != symbol or st.session_state.get("mode_at_run") != mode:
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close_full, steps=30)
        st.session_state.fc_idx = fc_idx
        st.session_state.fc_vals = fc_vals
        st.session_state.fc_ci = fc_ci

    fc_idx = st.session_state.get("fc_idx")
    fc_vals = st.session_state.get("fc_vals")
    fc_ci = st.session_state.get("fc_ci")

    fig, ax = plt.subplots(figsize=(14, 5))
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    ax.plot(close_full.index, close_full.values, label="Close")
    if fc_idx is not None and fc_vals is not None:
        ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, "--", linewidth=2.0, label="Forecast")

    if isinstance(fc_ci, pd.DataFrame) and len(fc_ci.columns) >= 2:
        lo = _coerce_1d_series(fc_ci.iloc[:, 0]).to_numpy(dtype=float)
        hi = _coerce_1d_series(fc_ci.iloc[:, 1]).to_numpy(dtype=float)
        ax.fill_between(fc_idx, lo, hi, alpha=0.15, label="Forecast CI")

    ax.set_title(f"{symbol} — Forecast (SARIMAX)")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def render_enhanced_forecast_chart(symbol: str, close_full: pd.Series):
    """
    Enhanced forecast chart (adds regression line + ±2σ bands on the DAILY close series).
    """
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty or len(close_full) < 20:
        st.warning("Not enough data.")
        return

    # Use the same daily_view subset for the enhanced panel
    close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
    if close_show.empty or len(close_show) < 10:
        st.warning("Not enough data in the selected Daily view range.")
        return

    # Forecast computed on full series, displayed alongside regression+bands on close_show
    if "fc_idx" not in st.session_state or st.session_state.get("ticker") != symbol or st.session_state.get("mode_at_run") != mode:
        fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(close_full, steps=30)
        st.session_state.fc_idx = fc_idx
        st.session_state.fc_vals = fc_vals
        st.session_state.fc_ci = fc_ci

    fc_idx = st.session_state.get("fc_idx")
    fc_vals = st.session_state.get("fc_vals")
    fc_ci = st.session_state.get("fc_ci")

    yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    ax.plot(close_show.index, close_show.values, label="Close (Daily)")

    if not yhat.dropna().empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.2, label=f"Regression (lb={slope_lb_daily}) {fmt_slope(m)}/bar  R²={fmt_r2(r2)}")

    if not up.dropna().empty and not lo.dropna().empty:
        ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

    if fc_idx is not None and fc_vals is not None:
        ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, ":", linewidth=2.4, label="Forecast (30d)")

    if isinstance(fc_ci, pd.DataFrame) and len(fc_ci.columns) >= 2:
        lo_ci = _coerce_1d_series(fc_ci.iloc[:, 0]).to_numpy(dtype=float)
        hi_ci = _coerce_1d_series(fc_ci.iloc[:, 1]).to_numpy(dtype=float)
        ax.fill_between(fc_idx, lo_ci, hi_ci, alpha=0.10, label="Forecast CI")

    ax.set_title(f"{symbol} — Enhanced Forecast (Regression + ±2σ)")
    ax.set_ylabel("Price")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def render_daily_price_chart(symbol: str, close_full: pd.Series):
    close_full = _coerce_1d_series(close_full).dropna()
    if close_full.empty:
        st.warning("No daily data.")
        return

    close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
    if close_show.empty or len(close_show) < 10:
        st.warning("Not enough data in selected Daily view range.")
        return

    ohlc = fetch_hist_ohlc(symbol)
    ohlc_show = pd.DataFrame()
    if ohlc is not None and (not ohlc.empty):
        x0, x1 = close_show.index[0], close_show.index[-1]
        ohlc_show = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)].copy()

    hma = compute_hma(close_show, period=int(hma_period))
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close_show, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))

    kijun = pd.Series(index=close_show.index, dtype=float)
    if show_ichi and (not ohlc_show.empty) and {"High","Low","Close"}.issubset(ohlc_show.columns):
        _, kij, _, _, _ = ichimoku_lines(
            ohlc_show["High"], ohlc_show["Low"], ohlc_show["Close"],
            conv=int(ichi_conv), base=int(ichi_base), span_b=int(ichi_spanb),
            shift_cloud=False
        )
        kijun = _coerce_1d_series(kij).reindex(close_show.index).ffill().bfill()

    st_line = pd.Series(index=close_show.index, dtype=float)
    if not ohlc_show.empty and {"High","Low","Close"}.issubset(ohlc_show.columns):
        st_df = compute_supertrend(ohlc_show, atr_period=int(atr_period), atr_mult=float(atr_mult))
        if "ST" in st_df.columns:
            st_line = _coerce_1d_series(st_df["ST"]).reindex(close_show.index).ffill()

    psar_df = pd.DataFrame()
    if show_psar and (not ohlc_show.empty) and {"High","Low"}.issubset(ohlc_show.columns):
        psar_df = compute_psar_from_ohlc(ohlc_show, step=float(psar_step), max_step=float(psar_max)).reindex(close_show.index)

    yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    ax.plot(close_show.index, close_show.values, label="Close")

    if show_bbands and (not bb_up.dropna().empty) and (not bb_lo.dropna().empty):
        ax.fill_between(close_show.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    if show_hma and not hma.dropna().empty:
        ax.plot(hma.index, hma.values, "-", linewidth=1.8, label=f"HMA({hma_period})")

    if show_ichi and not kijun.dropna().empty:
        ax.plot(kijun.index, kijun.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, "-", linewidth=1.6, label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=18, color="tab:green", zorder=6, label="PSAR")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=18, color="tab:red", zorder=6)

    if not yhat.dropna().empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.2, label=f"Regression (lb={slope_lb_daily}) {fmt_slope(m)}/bar  R²={fmt_r2(r2)}")
    if not up.dropna().empty and not lo.dropna().empty:
        ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

    if show_fibs:
        fibs = fibonacci_levels(close_show)
        if fibs:
            for lbl, yv in fibs.items():
                ax.hlines(yv, xmin=close_show.index[0], xmax=close_show.index[-1], linestyles="dotted", linewidth=1)
                ax.text(close_show.index[-1], yv, f" {lbl}", va="center", fontsize=8)

    ax.set_title(f"{symbol} — Daily Price Chart ({daily_view})")
    style_axes(ax)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)

def _apply_compact_time_ticks(ax, real_times: pd.DatetimeIndex, n_ticks: int = 8):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return
    n = len(real_times)
    n_ticks = int(max(2, min(n_ticks, n)))
    pos = np.linspace(0, n - 1, n_ticks, dtype=int)
    labels = []
    for i in pos:
        try:
            labels.append(real_times[i].strftime("%m-%d %H:%M"))
        except Exception:
            labels.append(str(real_times[i]))
    ax.set_xticks(pos.tolist())
    ax.set_xticklabels(labels, rotation=0, fontsize=8)

def render_hourly_price_chart(symbol: str, intraday: pd.DataFrame):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    dfp = intr_plot

    hc = _coerce_1d_series(dfp["Close"]).ffill().dropna()
    if hc.empty:
        st.warning("No intraday Close values.")
        return

    # Indicators on bar index (range), rendered with compact labels
    hma_h = compute_hma(hc, period=int(hma_period))
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(hc, window=int(bb_win), mult=float(bb_mult), use_ema=bool(bb_use_ema))
    yhat, up, lo, m, r2 = regression_with_band(hc, lookback=int(slope_lb_hourly), z=2.0)

    st_line = pd.Series(index=hc.index, dtype=float)
    if {"High","Low","Close"}.issubset(dfp.columns):
        st_df = compute_supertrend(dfp[["High","Low","Close"]], atr_period=int(atr_period), atr_mult=float(atr_mult))
        if "ST" in st_df.columns:
            st_line = _coerce_1d_series(st_df["ST"]).reindex(hc.index).ffill()

    psar_df = pd.DataFrame()
    if show_psar and {"High","Low"}.issubset(dfp.columns):
        psar_df = compute_psar_from_ohlc(dfp[["High","Low"]].join(dfp["Close"], how="left"), step=float(psar_step), max_step=float(psar_max)).reindex(hc.index)

    fig, ax = plt.subplots(figsize=(14, 5))
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass

    ax.plot(hc.index, hc.values, label="Intraday Close")

    if show_bbands and (not bb_up.dropna().empty) and (not bb_lo.dropna().empty):
        ax.fill_between(hc.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid (w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    if show_hma and not hma_h.dropna().empty:
        ax.plot(hma_h.index, hma_h.values, "-", linewidth=1.8, label=f"HMA({hma_period})")

    if not st_line.dropna().empty:
        ax.plot(st_line.index, st_line.values, "-", linewidth=1.6, label=f"Supertrend ({atr_period},{atr_mult})")

    if show_psar and (not psar_df.empty) and ("PSAR" in psar_df.columns):
        up_mask = psar_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax.scatter(psar_df.index[up_mask], psar_df["PSAR"][up_mask], s=18, color="tab:green", zorder=6, label="PSAR")
        if dn_mask.any():
            ax.scatter(psar_df.index[dn_mask], psar_df["PSAR"][dn_mask], s=18, color="tab:red", zorder=6)

    if not yhat.dropna().empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2.2, label=f"Regression (lb={slope_lb_hourly}) {fmt_slope(m)}/bar  R²={fmt_r2(r2)}")
    if not up.dropna().empty and not lo.dropna().empty:
        ax.plot(up.index, up.values, "--", linewidth=2.0, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.0, color="black", alpha=0.85, label="-2σ")

    if mode == "Forex" and show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and not real_times.empty:
        sess = compute_session_lines(real_times)
        # Map session times to nearest bar index
        def _nearest_bar(t):
            try:
                i = int(real_times.get_indexer([t], method="nearest")[0])
                return i if i >= 0 else None
            except Exception:
                return None
        sess_pos = {k: [p for p in [_nearest_bar(t) for t in v] if p is not None] for k, v in sess.items()}
        handles, labels = draw_session_lines(ax, sess_pos)
        ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
                  labels=labels + ax.get_legend_handles_labels()[1],
                  loc="upper left")
    else:
        ax.legend(loc="upper left")

    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax, real_times, n_ticks=9)

    ax.set_title(f"{symbol} — Hourly (5m bars)")
    style_axes(ax)
    st.pyplot(fig, use_container_width=True)

# ---------------------------
# Symbol selection + Forecast button (kept)
# ---------------------------
st.sidebar.subheader("Symbol")
sel = st.sidebar.selectbox("Select symbol:", universe, index=0, key="sb_symbol")

if "run_all" not in st.session_state:
    st.session_state.run_all = False
    st.session_state.ticker = None

run_clicked = st.sidebar.button("📌 Forecast", use_container_width=True, key="btn_run_forecast")

if run_clicked:
    st.session_state.run_all = True
    st.session_state.ticker = sel
    st.session_state.mode_at_run = mode

# Pre-fetch on run
close_full = None
intraday = None
if st.session_state.run_all and st.session_state.ticker == sel and st.session_state.get("mode_at_run") == mode:
    with st.spinner("Fetching data..."):
        close_full = fetch_hist(sel)
        intraday = fetch_intraday(sel, period="5d")
        st.session_state.df_hist = close_full
        st.session_state.intraday = intraday
else:
    # show cached if present
    close_full = st.session_state.get("df_hist", None)
    intraday = st.session_state.get("intraday", None)

# =========================
# Tabs (UPDATED)
# Removed (as requested): NPX 0.5-Cross Scanner, Fib NPX 0.0 Signal Scanner, News, Ichimoku Kijun Scanner
# Added (as requested): HMA Buy, NPX 0.0 Cross (Regression>0)
# =========================
tab_names = [
    "Forecast",
    "Enhanced Forecast",
    "Daily Price",
    "Hourly Price",
    "Bull/Bear",
    "Metrics",
    "Support/Resistance",
    "Signals",
    "HMA Buy",
    "NPX 0.0 Cross",
    "R² Scanners",
    "About",
]
tabs = st.tabs(tab_names)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = tabs

# =========================
# Tab 1 — Forecast
# =========================
with tab1:
    st.subheader("Forecast")
    if close_full is None or _coerce_1d_series(close_full).dropna().empty:
        st.info("Click **📌 Forecast** in the sidebar to run.")
    else:
        render_forecast_chart(sel, close_full)

# =========================
# Tab 2 — Enhanced Forecast
# =========================
with tab2:
    st.subheader("Enhanced Forecast")
    if close_full is None or _coerce_1d_series(close_full).dropna().empty:
        st.info("Click **📌 Forecast** in the sidebar to run.")
    else:
        render_enhanced_forecast_chart(sel, close_full)

# =========================
# Tab 3 — Daily Price
# =========================
with tab3:
    st.subheader("Daily Price Chart")
    if close_full is None or _coerce_1d_series(close_full).dropna().empty:
        st.info("Click **📌 Forecast** in the sidebar to run.")
    else:
        render_daily_price_chart(sel, close_full)

# =========================
# Tab 4 — Hourly Price
# =========================
with tab4:
    st.subheader("Hourly Price Chart (5m bars)")
    if intraday is None or (isinstance(intraday, pd.DataFrame) and intraday.empty):
        st.info("Click **📌 Forecast** in the sidebar to run.")
    else:
        render_hourly_price_chart(sel, intraday)

# =========================
# End of Batch 1/3 (Tabs 1–4)
# =========================
# Type: continue
# to receive Batch 2/3 (Tabs 5–8)
# =========================
# Part 9/10 — bullbear.py
# =========================
# ---------------------------
# Continue: Hourly renderer (rest) + Daily renderer + Tabs 1–8
# ---------------------------

def render_hourly_views(sel: str,
                        intraday: pd.DataFrame,
                        p_up: float,
                        p_dn: float,
                        hour_range_label: str,
                        is_forex: bool):
    if intraday is None or intraday.empty or "Close" not in intraday:
        st.warning("No intraday data available.")
        return None

    real_times = intraday.index if isinstance(intraday.index, pd.DatetimeIndex) else None
    intr_plot = intraday.copy()
    intr_plot.index = pd.RangeIndex(len(intr_plot))
    intraday = intr_plot

    hc = intraday["Close"].ffill()
    he = hc.ewm(span=20).mean()

    res_h = hc.rolling(sr_lb_hourly, min_periods=1).max()
    sup_h = hc.rolling(sr_lb_hourly, min_periods=1).min()

    hma_h = compute_hma(hc, period=hma_period)
    macd_h, macd_sig_h, macd_hist_h = compute_macd(hc)

    st_intraday = compute_supertrend(intraday, atr_period=atr_period, atr_mult=atr_mult)
    st_line_intr = st_intraday["ST"].reindex(hc.index) if "ST" in st_intraday.columns else pd.Series(dtype=float)

    kijun_h = pd.Series(index=hc.index, dtype=float)
    if {"High","Low","Close"}.issubset(intraday.columns) and show_ichi:
        _, kijun_h, _, _, _ = ichimoku_lines(
            intraday["High"], intraday["Low"], intraday["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
            shift_cloud=False
        )
        kijun_h = kijun_h.reindex(hc.index).ffill().bfill()

    bb_mid_h, bb_up_h, bb_lo_h, bb_pctb_h, bb_nbb_h = compute_bbands(
        hc, window=bb_win, mult=bb_mult, use_ema=bb_use_ema
    )

    psar_h_df = compute_psar_from_ohlc(intraday, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()
    if not psar_h_df.empty:
        psar_h_df = psar_h_df.reindex(hc.index)

    yhat_h, upper_h, lower_h, m_h, r2_h = regression_with_band(hc, slope_lb_hourly)
    slope_sig_h = m_h

    rev_prob_h = slope_reversal_probability(
        hc,
        slope_sig_h,
        hist_window=rev_hist_lb,
        slope_window=slope_lb_hourly,
        horizon=rev_horizon,
    )

    fx_news = pd.DataFrame()
    # NOTE: Tabs removed per request (News tab removed), but intraday news markers can remain controlled by sidebar toggle.
    if is_forex and show_fx_news:
        fx_news = fetch_yf_news(sel, window_days=news_window_days)

    ax2w = None
    if show_nrsi:
        fig2, (ax2, ax2w) = plt.subplots(
            2, 1, sharex=True, figsize=(14, 7),
            gridspec_kw={"height_ratios": [3.2, 1.3]}
        )
        plt.subplots_adjust(hspace=0.05, top=0.92, right=0.93, bottom=0.34)
    else:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.34)

    try:
        fig2.patch.set_facecolor("white")
    except Exception:
        pass

    ax2.plot(hc.index, hc, label="Intraday")
    ax2.plot(he.index, he.values, "--", label="20 EMA")

    global_m_h = draw_trend_direction_line(ax2, hc, label_prefix="Trend (global)")

    if show_hma and not hma_h.dropna().empty:
        ax2.plot(hma_h.index, hma_h.values, "-", linewidth=1.6, label=f"HMA({hma_period})")

    if show_ichi and not kijun_h.dropna().empty:
        ax2.plot(kijun_h.index, kijun_h.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    if show_bbands and not bb_up_h.dropna().empty and not bb_lo_h.dropna().empty:
        ax2.fill_between(hc.index, bb_lo_h, bb_up_h, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax2.plot(bb_mid_h.index, bb_mid_h.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax2.plot(bb_up_h.index, bb_up_h.values, ":", linewidth=1.0)
        ax2.plot(bb_lo_h.index, bb_lo_h.values, ":", linewidth=1.0)

    if show_psar and (not psar_h_df.empty) and ("PSAR" in psar_h_df.columns):
        up_mask = psar_h_df["in_uptrend"] == True
        dn_mask = ~up_mask
        if up_mask.any():
            ax2.scatter(psar_h_df.index[up_mask], psar_h_df["PSAR"][up_mask], s=15, color="tab:green", zorder=6,
                        label=f"PSAR (step={psar_step:.02f}, max={psar_max:.02f})")
        if dn_mask.any():
            ax2.scatter(psar_h_df.index[dn_mask], psar_h_df["PSAR"][dn_mask], s=15, color="tab:red", zorder=6)

    res_val = sup_val = px_val = np.nan
    try:
        res_val = float(res_h.iloc[-1])
        sup_val = float(sup_h.iloc[-1])
        px_val  = float(hc.iloc[-1])
    except Exception:
        pass

    if np.isfinite(res_val) and np.isfinite(sup_val):
        ax2.hlines(res_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax2.hlines(sup_val, xmin=hc.index[0], xmax=hc.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax2, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax2, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    if not st_line_intr.empty:
        ax2.plot(st_line_intr.index, st_line_intr.values, "-", label=f"Supertrend ({atr_period},{atr_mult})")

    if not yhat_h.empty:
        ax2.plot(yhat_h.index, yhat_h.values, "-", linewidth=2, label=f"Slope {slope_lb_hourly} bars ({fmt_slope(m_h)}/bar)")
    if not upper_h.empty and not lower_h.empty:
        ax2.plot(upper_h.index, upper_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope +2σ")
        ax2.plot(lower_h.index, lower_h.values, "--", linewidth=2.2, color="black", alpha=0.85, label="Slope -2σ")

        bounce_sig_h = find_band_bounce_signal(hc, upper_h, lower_h, slope_sig_h)
        if bounce_sig_h is not None:
            annotate_crossover(ax2, bounce_sig_h["time"], bounce_sig_h["price"], bounce_sig_h["side"])

    if is_forex and show_fx_news and (not fx_news.empty) and isinstance(real_times, pd.DatetimeIndex):
        news_pos = _map_times_to_bar_positions(real_times, fx_news["time"].tolist())
        if news_pos:
            draw_news_markers(ax2, news_pos, label="News")

    instr_txt = format_trade_instruction(
        trend_slope=slope_sig_h,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=sel,
        global_trend_slope=global_m_h
    )

    macd_sig = find_macd_hma_sr_signal(
        close=hc, hma=hma_h, macd=macd_h, sup=sup_h, res=res_h,
        global_trend_slope=global_m_h, prox=sr_prox_pct
    )

    macd_instr_txt = "MACD/HMA55: n/a"
    if macd_sig is not None and np.isfinite(macd_sig.get("price", np.nan)):
        side = macd_sig["side"]
        macd_instr_txt = f"MACD/HMA55: {side} @ {fmt_price_val(macd_sig['price'])}"
        annotate_macd_signal(ax2, macd_sig["time"], macd_sig["price"], side)

    ax2.text(
        0.01, 0.98, macd_instr_txt,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.8),
        zorder=20
    )

    rev_txt_h = fmt_pct(rev_prob_h) if np.isfinite(rev_prob_h) else "n/a"
    ax2.set_title(
        f"{sel} Intraday ({hour_range_label})  "
        f"↑{fmt_pct(p_up)}  ↓{fmt_pct(p_dn)}  "
        f"[P(slope rev≤{rev_horizon} bars)={rev_txt_h}]"
    )

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb_h.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb_h.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        ax2.text(0.99, 0.02,
                 f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=11, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax2.text(0.01, 0.02,
             f"Slope: {fmt_slope(slope_sig_h)}/bar  |  P(rev≤{rev_horizon} bars): {fmt_pct(rev_prob_h)}",
             transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))
    ax2.text(0.50, 0.02,
             f"R² ({slope_lb_hourly} bars): {fmt_r2(r2_h)}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    # Session lines for FX
    session_handles = None
    session_labels = None
    if is_forex and show_sessions_pst and isinstance(real_times, pd.DatetimeIndex) and not real_times.empty:
        sess = compute_session_lines(real_times)
        sess_pos = {
            "ldn_open": _map_times_to_bar_positions(real_times, sess.get("ldn_open", [])),
            "ldn_close": _map_times_to_bar_positions(real_times, sess.get("ldn_close", [])),
            "ny_open": _map_times_to_bar_positions(real_times, sess.get("ny_open", [])),
            "ny_close": _map_times_to_bar_positions(real_times, sess.get("ny_close", [])),
        }
        session_handles, session_labels = draw_session_lines(ax2, sess_pos)

    # Fibs on intraday
    if show_fibs and not hc.empty:
        fibs_h = fibonacci_levels(hc)
        for lbl, y in fibs_h.items():
            ax2.hlines(y, xmin=hc.index[0], xmax=hc.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs_h.items():
            ax2.text(hc.index[-1], y, f" {lbl}", va="center")

    # Fib + NPX 0-cross signal
    npx_h_for_sig = compute_normalized_price(hc, window=ntd_window)
    fib_sig_h = _fib_npx_zero_signal_series(
        close=hc,
        npx=npx_h_for_sig,
        prox=sr_prox_pct,
        lookback_bars=int(max(3, rev_horizon)),
        slope_lb=int(slope_lb_hourly),
        npx_confirm_bars=1
    )
    if isinstance(fib_sig_h, dict):
        annotate_fib_npx_signal(ax2, fib_sig_h)

    # Fibonacci reversal trigger from extremes (intraday)
    fib_trig_chart = fib_reversal_trigger_from_extremes(
        hc,
        proximity_pct_of_range=0.02,
        confirm_bars=int(rev_bars_confirm),
        lookback_bars=int(max(60, slope_lb_hourly)),
    )
    if isinstance(fib_trig_chart, dict):
        try:
            touch_bar = int(fib_trig_chart.get("touch_time"))
        except Exception:
            touch_bar = None
        if touch_bar is not None and 0 <= touch_bar < len(hc):
            t_touch = touch_bar
            p_touch = float(hc.iloc[touch_bar]) if np.isfinite(hc.iloc[touch_bar]) else np.nan
            if np.isfinite(p_touch):
                side = str(fib_trig_chart.get("side", "")).upper()
                col = "tab:green" if side == "BUY" else "tab:red"
                ax2.scatter([t_touch], [p_touch], marker="^" if side == "BUY" else "v", s=130, color=col, zorder=12)
                ax2.text(t_touch, p_touch, f"  {side} (FIB EXT)", color=col, fontsize=9,
                         fontweight="bold", va="bottom" if side == "BUY" else "top")

    style_axes(ax2)

    # NOTE: since intraday uses bar index on x-axis, apply compact time ticks
    if isinstance(real_times, pd.DatetimeIndex):
        _apply_compact_time_ticks(ax2, real_times, n_ticks=8)

    # Trade instruction text box
    ax2.text(
        0.01, 0.88, instr_txt,
        transform=ax2.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.75)
    )

    # MACD Panel (optional)
    if show_macd:
        figm, axm = plt.subplots(figsize=(14, 2.6))
        axm.plot(macd_h.index, macd_h.values, label="MACD")
        axm.plot(macd_sig_h.index, macd_sig_h.values, label="Signal")
        axm.axhline(0, linewidth=1)
        style_axes(axm)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm, real_times, n_ticks=8)
        axm.set_title(f"{sel} MACD (Hourly)")
        axm.legend(loc="upper left", frameon=True)
        st.pyplot(figm)

    # Hourly momentum panel (optional)
    if show_mom_hourly:
        roc = compute_roc(hc, n=mom_lb_hourly)
        figm2, axm2 = plt.subplots(figsize=(14, 2.4))
        axm2.plot(roc.index, roc.values, label=f"ROC% ({mom_lb_hourly})")
        axm2.axhline(0, linewidth=1)
        style_axes(axm2)
        if isinstance(real_times, pd.DatetimeIndex):
            _apply_compact_time_ticks(axm2, real_times, n_ticks=8)
        axm2.set_title(f"{sel} Momentum (Hourly)")
        axm2.legend(loc="upper left", frameon=True)
        st.pyplot(figm2)

    # NTD / NPX panel
    if show_nrsi and ax2w is not None:
        ntd_h = compute_normalized_trend(hc, window=ntd_window)
        npx_h = compute_normalized_price(hc, window=ntd_window)

        ax2w.plot(ntd_h.index, ntd_h.values, linewidth=1.8, label="NTD")
        ax2w.axhline(0.0, linewidth=1, color="black", alpha=0.6)
        ax2w.axhline(0.75, linestyle="--", linewidth=1, alpha=0.6)
        ax2w.axhline(-0.75, linestyle="--", linewidth=1, alpha=0.6)

        if shade_ntd:
            shade_ntd_regions(ax2w, ntd_h)

        if show_npx_ntd:
            overlay_npx_on_ntd(ax2w, npx_h, ntd_h, mark_crosses=mark_npx_cross)

        if show_ntd_channel:
            overlay_inrange_on_ntd(ax2w, hc, sup_h, res_h)

        overlay_ntd_triangles_by_trend(ax2w, ntd_h, trend_slope=global_m_h, upper=0.75, lower=-0.75)

        if show_hma_rev_ntd:
            overlay_hma_reversal_on_ntd(ax2w, hc, hma_h, lookback=hma_rev_lb, period=hma_period, ntd=ntd_h)

        overlay_ntd_sr_reversal_stars(
            ax2w, hc, sup_h, res_h,
            trend_slope=global_m_h, ntd=ntd_h,
            prox=sr_prox_pct,
            bars_confirm=rev_bars_confirm
        )

        style_axes(ax2w)
        ax2w.set_ylim(-1.05, 1.05)
        ax2w.set_title("NTD / NPX (Hourly)")
        ax2w.legend(loc="upper left", frameon=True)

    # Legend
    handles, labels = ax2.get_legend_handles_labels()
    if session_handles and session_labels:
        handles += session_handles
        labels += session_labels
    # Ensure legend doesn't overlap
    try:
        ax2.legend(handles, labels, loc="upper left", frameon=True, ncol=2)
    except Exception:
        try:
            ax2.legend(loc="upper left", frameon=True)
        except Exception:
            pass

    st.pyplot(fig2)
    return True


# ---------------------------
# Daily renderer
# ---------------------------
def render_daily_price_chart(ticker: str, daily_view_label: str):
    close_full = _coerce_1d_series(fetch_hist(ticker)).dropna()
    if close_full.empty:
        st.warning("No daily data available.")
        return

    close = _coerce_1d_series(subset_by_daily_view(close_full, daily_view_label)).dropna()
    if close.empty:
        st.warning("No daily data in selected view.")
        return

    # OHLC for ichimoku/PSAR on daily
    ohlc = fetch_hist_ohlc(ticker)
    ohlc = ohlc.sort_index() if ohlc is not None else pd.DataFrame()
    if not ohlc.empty:
        x0, x1 = close.index[0], close.index[-1]
        ohlc = ohlc.loc[(ohlc.index >= x0) & (ohlc.index <= x1)]

    fig, ax = plt.subplots(figsize=(14, 5))
    plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)

    ax.plot(close.index, close.values, label="Daily Close")

    # Trendline + bands
    yhat, up, lo, m, r2 = regression_with_band(close, lookback=int(slope_lb_daily))
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, "-", linewidth=2, label=f"Trend (Slope {fmt_slope(m)}/bar)")
    if not up.empty and not lo.empty:
        ax.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ")
        ax.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ")

    # HMA
    hma_d = compute_hma(close, period=hma_period)
    if show_hma and not hma_d.dropna().empty:
        ax.plot(hma_d.index, hma_d.values, "-", linewidth=1.8, label=f"HMA({hma_period})")

    # Ichimoku Kijun overlay on price (daily)
    kijun_d = pd.Series(index=close.index, dtype=float)
    if show_ichi and not ohlc.empty and {"High","Low","Close"}.issubset(ohlc.columns):
        _, kijun_d, _, _, _ = ichimoku_lines(
            ohlc["High"], ohlc["Low"], ohlc["Close"],
            conv=ichi_conv, base=ichi_base, span_b=ichi_spanb,
            shift_cloud=False
        )
        kijun_d = _coerce_1d_series(kijun_d).reindex(close.index).ffill().bfill()
        if not kijun_d.dropna().empty:
            ax.plot(kijun_d.index, kijun_d.values, "-", linewidth=1.8, color="black", label=f"Ichimoku Kijun ({ichi_base})")

    # Bollinger
    bb_mid, bb_up, bb_lo, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)
    if show_bbands and not bb_up.dropna().empty and not bb_lo.dropna().empty:
        ax.fill_between(close.index, bb_lo, bb_up, alpha=0.06, label=f"BB (×{bb_mult:.1f})")
        ax.plot(bb_mid.index, bb_mid.values, "-", linewidth=1.1, label=f"BB mid ({'EMA' if bb_use_ema else 'SMA'}, w={bb_win})")
        ax.plot(bb_up.index, bb_up.values, ":", linewidth=1.0)
        ax.plot(bb_lo.index, bb_lo.values, ":", linewidth=1.0)

    # Support/Resistance
    sup = close.rolling(int(sr_lb_daily), min_periods=1).min()
    res = close.rolling(int(sr_lb_daily), min_periods=1).max()
    try:
        sup_val = float(sup.iloc[-1])
        res_val = float(res.iloc[-1])
        px_val  = float(close.iloc[-1])
    except Exception:
        sup_val = res_val = px_val = np.nan

    if np.isfinite(sup_val) and np.isfinite(res_val):
        ax.hlines(res_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:red", linestyles="-", linewidth=1.6, label="Resistance")
        ax.hlines(sup_val, xmin=close.index[0], xmax=close.index[-1], colors="tab:green", linestyles="-", linewidth=1.6, label="Support")
        label_on_left(ax, res_val, f"R {fmt_price_val(res_val)}", color="tab:red")
        label_on_left(ax, sup_val, f"S {fmt_price_val(sup_val)}", color="tab:green")

    # Fibonacci levels (daily)
    if show_fibs and not close.empty:
        fibs = fibonacci_levels(close)
        for lbl, y in fibs.items():
            ax.hlines(y, xmin=close.index[0], xmax=close.index[-1], linestyles="dotted", linewidth=1)
        for lbl, y in fibs.items():
            ax.text(close.index[-1], y, f" {lbl}", va="center")

    # Band bounce markers (daily)
    if not up.empty and not lo.empty and np.isfinite(m):
        bounce_sig = find_band_bounce_signal(close, up, lo, m)
        if bounce_sig is not None:
            annotate_crossover(ax, bounce_sig["time"], bounce_sig["price"], bounce_sig["side"])

    # Slope-trigger after band reversal (daily)
    trig = None
    if (not yhat.empty) and (not up.empty) and (not lo.empty):
        trig = find_slope_trigger_after_band_reversal(close, yhat, up, lo, horizon=rev_horizon)
        if isinstance(trig, dict):
            annotate_slope_trigger(ax, trig)

    # Trade instruction
    instr_txt = format_trade_instruction(
        trend_slope=m,
        buy_val=sup_val,
        sell_val=res_val,
        close_val=px_val,
        symbol=ticker,
        global_trend_slope=m  # daily uses same slope for both
    )

    ax.text(
        0.01, 0.98, instr_txt,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.75)
    )

    # R² and last price info
    r2_txt = fmt_r2(r2) if np.isfinite(r2) else "n/a"
    ax.text(0.50, 0.02,
            f"R² ({slope_lb_daily} bars): {r2_txt}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    if np.isfinite(px_val):
        nbb_txt = ""
        try:
            last_pct = float(bb_pctb.dropna().iloc[-1]) if show_bbands else np.nan
            last_nbb = float(bb_nbb.dropna().iloc[-1]) if show_bbands else np.nan
            if np.isfinite(last_nbb) and np.isfinite(last_pct):
                nbb_txt = f"  |  NBB {last_nbb:+.2f}  •  %B {fmt_pct(last_pct, digits=0)}"
        except Exception:
            pass
        ax.text(0.99, 0.02,
                f"Current price: {fmt_price_val(px_val)}{nbb_txt}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="grey", alpha=0.7))

    ax.set_title(f"{ticker} Daily ({daily_view_label})")
    style_axes(ax)
    try:
        ax.legend(loc="upper left", frameon=True, ncol=2)
    except Exception:
        pass

    st.pyplot(fig)

    # NTD Panel (daily)
    if show_ntd:
        ntd = compute_normalized_trend(close, window=ntd_window)
        npx = compute_normalized_price(close, window=ntd_window)

        fig3, ax3 = plt.subplots(figsize=(14, 3.5))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)

        ax3.plot(ntd.index, ntd.values, linewidth=1.8, label="NTD")
        ax3.axhline(0.0, linewidth=1, color="black", alpha=0.6)
        ax3.axhline(0.75, linestyle="--", linewidth=1, alpha=0.6)
        ax3.axhline(-0.75, linestyle="--", linewidth=1, alpha=0.6)

        if shade_ntd:
            shade_ntd_regions(ax3, ntd)

        if show_npx_ntd:
            overlay_npx_on_ntd(ax3, npx, ntd, mark_crosses=mark_npx_cross)

        if show_ntd_channel:
            overlay_inrange_on_ntd(ax3, close, sup, res)

        overlay_ntd_triangles_by_trend(ax3, ntd, trend_slope=m, upper=0.75, lower=-0.75)

        if show_hma_rev_ntd:
            overlay_hma_reversal_on_ntd(ax3, close, hma_d, lookback=hma_rev_lb, period=hma_period, ntd=ntd)

        overlay_ntd_sr_reversal_stars(
            ax3, close, sup, res,
            trend_slope=m, ntd=ntd,
            prox=sr_prox_pct,
            bars_confirm=rev_bars_confirm
        )

        ax3.set_ylim(-1.05, 1.05)
        ax3.set_title("NTD / NPX (Daily)")
        style_axes(ax3)
        try:
            ax3.legend(loc="upper left", frameon=True, ncol=2)
        except Exception:
            pass
        st.pyplot(fig3)


# =========================
# Part 10/10 — bullbear.py
# =========================
# ---------------------------
# UI selection + Run button (UNCHANGED)
# ---------------------------
st.sidebar.subheader("Select Symbol")
ticker = st.sidebar.selectbox("Ticker:", universe, index=0, key="sb_ticker_select")
st.session_state.ticker = ticker

st.sidebar.subheader("Intraday Range")
hour_range = st.sidebar.selectbox("Range:", ["24h", "5d"], index=0, key="sb_hour_range")
st.session_state.hour_range = hour_range

run = st.sidebar.button("🚀 Run Forecast", use_container_width=True, key="btn_run_forecast")
if run:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# ---------------------------
# Tabs (keep existing, only remove requested + add new)
# ---------------------------
# Requested removals:
#   - NPX 0.5-Cross Scanner
#   - Fib NPX 0.0 Signal Scanner
#   - News
#   - Ichimoku Kijun Scanner
#
# Requested additions:
#   - HMA Buy
#   - NPX 0.0 Cross (Regression > 0)

# NOTE:
# This block assumes the existing app previously had those named tabs.
# We keep order/layout as-is, only removing those tabs and adding the two new ones.

tab_names = [
    "Forecast",
    "Enhanced Forecast",
    "Bull/Bear",
    "Metrics",
    "Scanners",
    "Long-Term",
    "Stickers",
    "Support Reversals",
    # (removed) "NPX 0.5-Cross Scanner",
    # (removed) "Fib NPX 0.0 Signal Scanner",
    # (removed) "News",
    # (removed) "Ichimoku Kijun Scanner",
    # NEW tabs (append at end to avoid disrupting existing order)
    "HMA Buy",
    "NPX 0.0 Cross (Uptrend)"
]

tabs = st.tabs(tab_names)

# Maintain references for first 8 tabs for Batch 1/2 instructions
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_hma_buy, tab_npx0 = tabs

# ---------------------------
# TAB 1: Forecast
# ---------------------------
with tab1:
    st.subheader("Forecast")
    if st.session_state.run_all:
        sel = st.session_state.ticker
        st.write(f"**Selected:** {sel}  |  **Mode:** {mode}")

        # Daily series + SARIMAX forecast
        close = fetch_hist(sel)
        idx, fc_vals, ci = compute_sarimax_forecast(close)

        fig, ax = plt.subplots(figsize=(14, 5))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)

        close_show = subset_by_daily_view(_coerce_1d_series(close).dropna(), daily_view)
        ax.plot(close_show.index, close_show.values, label="Historical")
        ax.plot(idx, fc_vals.values, "--", label="Forecast")
        ax.fill_between(idx, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.12, label="CI")

        style_axes(ax)
        ax.set_title(f"{sel} Forecast (Daily)")
        ax.legend(loc="upper left", frameon=True)
        st.pyplot(fig)

# ---------------------------
# TAB 2: Enhanced Forecast
# ---------------------------
with tab2:
    st.subheader("Enhanced Forecast")
    if st.session_state.run_all:
        sel = st.session_state.ticker
        close = _coerce_1d_series(fetch_hist(sel)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close, daily_view)).dropna()

        idx, fc_vals, ci = compute_sarimax_forecast(close)

        fig, ax = plt.subplots(figsize=(14, 5))
        plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)

        ax.plot(close_show.index, close_show.values, label="Historical")
        ax.plot(idx, fc_vals.values, "--", label="Forecast")
        ax.fill_between(idx, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.12, label="CI")

        # Regression + ±2σ bands on enhanced forecast (already added in your earlier code base)
        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily), z=2.0)
        if not yhat.empty:
            ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Regression ({fmt_slope(m)}/bar)")
        if not up.empty and not lo.empty:
            ax.plot(up.index, up.values, "--", linewidth=2.2, color="black", alpha=0.85, label="+2σ (reg)")
            ax.plot(lo.index, lo.values, "--", linewidth=2.2, color="black", alpha=0.85, label="-2σ (reg)")

        style_axes(ax)
        ax.set_title(f"{sel} Enhanced Forecast (Daily)")
        ax.legend(loc="upper left", frameon=True, ncol=2)
        st.pyplot(fig)

# ---------------------------
# TAB 3: Bull/Bear
# ---------------------------
with tab3:
    st.subheader("Bull/Bear")
    if st.session_state.run_all:
        sel = st.session_state.ticker
        df = yf.download(sel, period=bb_period, interval="1d")
        if df is None or df.empty:
            st.warning("No data.")
        else:
            close = _coerce_1d_series(df["Close"]).dropna()
            if close.empty:
                st.warning("No close series.")
            else:
                roc = compute_roc(close, n=10)
                fig, ax = plt.subplots(figsize=(14, 4))
                plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)
                ax.plot(close.index, close.values, label="Close")
                ax2 = ax.twinx()
                ax2.plot(roc.index, roc.values, "--", label="ROC% (10)", alpha=0.7)
                style_axes(ax)
                ax.set_title(f"{sel} Bull/Bear View ({bb_period})")
                ax.legend(loc="upper left", frameon=True)
                st.pyplot(fig)

# ---------------------------
# TAB 4: Metrics
# ---------------------------
with tab4:
    st.subheader("Metrics")
    if st.session_state.run_all:
        sel = st.session_state.ticker
        close = _coerce_1d_series(fetch_hist(sel)).dropna()
        close_show = _coerce_1d_series(subset_by_daily_view(close, daily_view)).dropna()

        yhat, up, lo, m, r2 = regression_with_band(close_show, lookback=int(slope_lb_daily))
        ntd = compute_normalized_trend(close_show, window=ntd_window)
        npx = compute_normalized_price(close_show, window=ntd_window)

        st.write(pd.DataFrame([{
            "Symbol": sel,
            "Daily Slope (lb)": slope_lb_daily,
            "Slope": float(m) if np.isfinite(m) else np.nan,
            "R2": float(r2) if np.isfinite(r2) else np.nan,
            "NTD (last)": float(ntd.dropna().iloc[-1]) if len(ntd.dropna()) else np.nan,
            "NPX (last)": float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan
        }]), use_container_width=True)

        st.markdown("---")
        st.write("### Daily Price Chart")
        render_daily_price_chart(sel, daily_view)

# ---------------------------
# TAB 5: Scanners
# ---------------------------
with tab5:
    st.subheader("Scanners")
    st.caption("Scanners operate over the current universe (Forex vs Stocks) using cached computations.")

    if st.session_state.run_all:
        st.write("### Recent Band Bounce Signals (Daily)")
        rows = []
        for sym in universe:
            rec = last_band_bounce_signal_daily(sym, slope_lb_daily)
            if rec is not None:
                rows.append(rec)
        if rows:
            df = pd.DataFrame(rows).sort_values(["Side", "Bars Since"], ascending=[True, True])
            df["DeltaPct"] = df["DeltaPct"].apply(lambda x: fmt_pct(x) if np.isfinite(x) else "n/a")
            df["R2"] = df["R2"].apply(lambda x: fmt_r2(x) if np.isfinite(x) else "n/a")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent band-bounce signals found.")

        st.markdown("---")
        st.write("### Recent Band Bounce Signals (Hourly)")
        hr_period = "1d" if st.session_state.hour_range == "24h" else "5d"
        rows = []
        for sym in universe:
            rec = last_band_bounce_signal_hourly(sym, period=hr_period, slope_lb=slope_lb_hourly)
            if rec is not None:
                rows.append(rec)
        if rows:
            df = pd.DataFrame(rows).sort_values(["Side", "Bars Since"], ascending=[True, True])
            df["DeltaPct"] = df["DeltaPct"].apply(lambda x: fmt_pct(x) if np.isfinite(x) else "n/a")
            df["R2"] = df["R2"].apply(lambda x: fmt_r2(x) if np.isfinite(x) else "n/a")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent hourly band-bounce signals found.")

# ---------------------------
# TAB 6: Long-Term
# ---------------------------
with tab6:
    st.subheader("Long-Term")
    if st.session_state.run_all:
        sel = st.session_state.ticker
        close = _coerce_1d_series(fetch_hist_max(sel)).dropna()
        if close.empty:
            st.warning("No long-term data.")
        else:
            fig, ax = plt.subplots(figsize=(14, 5))
            plt.subplots_adjust(top=0.90, right=0.93, bottom=0.20)
            ax.plot(close.index, close.values, label="Close (max)")
            yhat, up, lo, m, r2 = regression_with_band(close, lookback=min(len(close), 365))
            if not yhat.empty:
                ax.plot(yhat.index, yhat.values, "-", linewidth=2.0, label=f"Trend ({fmt_slope(m)}/bar)")
            style_axes(ax)
            ax.set_title(f"{sel} Long-Term (Max)")
            ax.legend(loc="upper left", frameon=True)
            st.pyplot(fig)

# ---------------------------
# TAB 7: Stickers
# ---------------------------
with tab7:
    st.subheader("Stickers")
    st.caption("Quick lists for last NTD values (daily) by universe.")
    if st.session_state.run_all:
        rows = []
        for sym in universe:
            ntd_last, ts = last_daily_ntd_value(sym, ntd_window)
            rows.append({"Symbol": sym, "NTD (last)": ntd_last, "AsOf": ts})
        df = pd.DataFrame(rows)
        df = df.sort_values("NTD (last)", ascending=False)
        df["NTD (last)"] = df["NTD (last)"].apply(lambda x: float(x) if np.isfinite(x) else np.nan)
        st.dataframe(df, use_container_width=True)

# ---------------------------
# TAB 8: Support Reversals
# ---------------------------
with tab8:
    st.subheader("Support Reversals")
    if st.session_state.run_all:
        st.write("### Daily Support Reversal Heading Up")
        horizon = st.slider("Lookback horizon (bars)", 1, 20, 5, 1, key="sr_rev_horizon")
        rows = []
        for sym in universe:
            rec = daily_support_reversal_heading_up(
                sym, daily_view_label=daily_view, sr_lb=sr_lb_daily,
                prox=sr_prox_pct, bars_confirm=rev_bars_confirm, horizon=horizon
            )
            if rec is not None:
                rows.append(rec)
        if rows:
            df = pd.DataFrame(rows).sort_values("Bars Since Touch", ascending=True)
            df["Dist vs Support"] = df["Dist vs Support"].apply(lambda x: fmt_pct(x) if np.isfinite(x) else "n/a")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No support reversals found in the current horizon.")

# ---------------------------
# NEW TAB: HMA Buy (THIS REQUEST)
# ---------------------------
with tab_hma_buy:
    st.subheader("HMA Buy")
    st.caption("Lists symbols where price recently crossed ABOVE HMA(55) on the DAILY chart within N bars.")
    if st.session_state.run_all:
        bars = st.slider("Bars since cross (daily)", 1, 3, 2, 1, key="hma_buy_bars")
        rows_pos = []
        rows_neg = []

        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue
                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if len(close_show) < max(10, hma_period + 5):
                    continue

                # Regression slope over selected daily view window
                x = np.arange(len(close_show), dtype=float)
                m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
                if not np.isfinite(m):
                    continue

                hma = compute_hma(close_show, period=hma_period)
                if hma.dropna().empty:
                    continue

                cross_up, _ = _cross_series(close_show, hma)
                if not cross_up.any():
                    continue

                t_cross = cross_up[cross_up].index[-1]
                loc = int(close_show.index.get_loc(t_cross))
                bars_since = int((len(close_show) - 1) - loc)
                if bars_since < 1 or bars_since > int(bars):
                    continue

                px_cross = float(close_show.loc[t_cross]) if np.isfinite(close_show.loc[t_cross]) else np.nan
                px_last = float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan
                hma_cross = float(hma.loc[t_cross]) if (t_cross in hma.index and np.isfinite(hma.loc[t_cross])) else np.nan
                if not np.all(np.isfinite([px_cross, px_last, hma_cross])):
                    continue

                rec = {
                    "Symbol": sym,
                    "Bars Since Cross": bars_since,
                    "Cross Time": t_cross,
                    "Price@Cross": px_cross,
                    f"HMA({hma_period})@Cross": hma_cross,
                    "Current Price": px_last,
                    "Regression Slope": float(m),
                }
                if float(m) > 0:
                    rows_pos.append(rec)
                else:
                    rows_neg.append(rec)
            except Exception:
                continue

        c1, c2 = st.columns(2)
        with c1:
            st.write("### (a) Regression > 0")
            if rows_pos:
                df = pd.DataFrame(rows_pos).sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No symbols matched (Regression > 0).")
        with c2:
            st.write("### (b) Regression < 0")
            if rows_neg:
                df = pd.DataFrame(rows_neg).sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No symbols matched (Regression < 0).")

# ---------------------------
# NEW TAB: NPX 0.0 Cross (Uptrend) (THIS REQUEST)
# ---------------------------
with tab_npx0:
    st.subheader("NPX 0.0 Cross (Uptrend)")
    st.caption("Lists symbols where NPX (Normalized Price) has just crossed 0.0 and regression slope > 0 (daily).")
    if st.session_state.run_all:
        bars = st.slider("Bars since NPX 0.0 cross", 1, 5, 2, 1, key="npx0_bars")
        rows = []
        for sym in universe:
            try:
                close_full = _coerce_1d_series(fetch_hist(sym)).dropna()
                if close_full.empty:
                    continue
                close_show = _coerce_1d_series(subset_by_daily_view(close_full, daily_view)).dropna()
                if len(close_show) < max(10, ntd_window + 5):
                    continue

                # Regression slope (daily view)
                x = np.arange(len(close_show), dtype=float)
                m, b = np.polyfit(x, close_show.to_numpy(dtype=float), 1)
                if not (np.isfinite(m) and float(m) > 0.0):
                    continue

                npx_full = compute_normalized_price(close_full, window=ntd_window)
                npx = _coerce_1d_series(npx_full).reindex(close_show.index)
                if npx.dropna().empty or len(npx) < 2:
                    continue

                prev = npx.shift(1)
                cross = (npx >= 0.0) & (prev < 0.0)
                cross = cross.fillna(False)
                if not cross.any():
                    continue

                t_cross = cross[cross].index[-1]
                loc = int(close_show.index.get_loc(t_cross))
                bars_since = int((len(close_show) - 1) - loc)
                if bars_since < 1 or bars_since > int(bars):
                    continue

                rows.append({
                    "Symbol": sym,
                    "Bars Since Cross": bars_since,
                    "Cross Time": t_cross,
                    "NPX(prev)": float(prev.loc[t_cross]) if np.isfinite(prev.loc[t_cross]) else np.nan,
                    "NPX@Cross": float(npx.loc[t_cross]) if np.isfinite(npx.loc[t_cross]) else np.nan,
                    "NPX (last)": float(npx.dropna().iloc[-1]) if len(npx.dropna()) else np.nan,
                    "Regression Slope": float(m),
                    "Current Price": float(close_show.iloc[-1]) if np.isfinite(close_show.iloc[-1]) else np.nan,
                })
            except Exception:
                continue

        if rows:
            df = pd.DataFrame(rows).sort_values(["Bars Since Cross", "Symbol"], ascending=[True, True])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No symbols matched the NPX 0.0 cross condition (Regression > 0).")
# =========================
# Batch 3/3 — Tabs (FIXED) + HMA Buy tab + remainder app body
# =========================

# --- Tab titles (UPDATED) ---
# We remove the 4 tabs you requested and add the new "HMA Buy" tab.
_REMOVED_TABS = {
    "NPX 0.5-Cross Scanner",
    "Fib NPX 0.0 Signal Scanner",
    "News",
    "Ichimoku Kijun Scanner",
}

# If you already defined `tab_titles` earlier, we reuse it; otherwise we fall back to a safe default list.
try:
    _base_titles = list(tab_titles)  # noqa: F821
except Exception:
    # Fallback only used if your earlier code didn't define tab_titles.
    # IMPORTANT: If you have a specific order, keep your original order above and let this block reuse it.
    _base_titles = [
        "Forecast",
        "Enhanced Forecast",
        "Bull/Bear",
        "Metrics",
        "Scanners",
        "Long-Term",
        "Stickers",
        "Support Reversals",
        "Support/Resistance",
        "Signals",
        "Slope Reversals",
        "R² Lists",
        "HMA Crosses",
        "Watchlist",
        # "HMA Buy" will be appended below
    ]

tab_titles = [t for t in _base_titles if t not in _REMOVED_TABS]

if "HMA Buy" not in tab_titles:
    tab_titles.append("HMA Buy")

# --- Create tabs (THIS is what prevents tab18 NameError) ---
_tabs = st.tabs(tab_titles)

# Create a title->tab mapping so we never depend on fragile tab numbers
tabs_by_title = {t: _tabs[i] for i, t in enumerate(tab_titles)}

# If your existing code uses tab1/tab2/... variables, we keep them for compatibility
# (but only up to the actual number of tabs now).
for i, _tobj in enumerate(_tabs, start=1):
    globals()[f"tab{i}"] = _tobj

# ------------------------------------------------------------
# Keep ALL your existing tab bodies exactly as-is, but change:
#   with tab18:
# to the appropriate tab variable (likely the last one now),
# OR (preferred) switch those "with ..." statements to:
#   with tabs_by_title["<Exact Tab Title>"]:
#
# Below, we only ADD the new "HMA Buy" tab body.
# ------------------------------------------------------------

# =========
# NEW TAB: HMA Buy
# =========
with tabs_by_title["HMA Buy"]:
    st.subheader("HMA Buy — Recent HMA(55) Crosses on Daily (1–3 bars)")

    # Slider: how many daily bars back to look for the most recent upward cross
    hma_lookback = st.slider(
        "Bars since upward cross (Daily)",
        min_value=1,
        max_value=3,
        value=2,
        step=1,
        key="hma_buy_lookback",
        help="Find symbols where Close crossed ABOVE HMA(55) within the last N daily bars.",
    )

    # Optional: allow user to choose regression window
    reg_window = st.slider(
        "Regression window (daily bars)",
        min_value=40,
        max_value=200,
        value=120,
        step=10,
        key="hma_buy_reg_window",
        help="Window used to compute regression slope (positive vs negative).",
    )

    # Universe: reuse your existing universe variables if present
    # (These names are commonly used in your app; if yours differs, it will fall back safely.)
    def _get_universe_symbols():
        # Prefer your already-built current universe (stocks/forex toggle)
        for name in ("symbols", "universe_symbols", "tickers", "scanner_symbols", "watchlist_symbols"):
            try:
                val = globals().get(name, None)
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return list(val)
            except Exception:
                pass

        # Fallback: if nothing exists, use a tiny default list
        return ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

    universe = _get_universe_symbols()

    # --- Helpers (local to this tab, won’t disturb your existing functions) ---
    def _wma(series: pd.Series, period: int) -> pd.Series:
        if period <= 1:
            return series.copy()
        w = np.arange(1, period + 1, dtype=float)
        return series.rolling(period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    def _hma(series: pd.Series, period: int = 55) -> pd.Series:
        # HMA = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )
        n = int(period)
        if n < 2:
            return series.copy()
        half = max(1, n // 2)
        sqrt_n = max(1, int(np.sqrt(n)))
        wma_full = _wma(series, n)
        wma_half = _wma(series, half)
        raw = (2 * wma_half) - wma_full
        return _wma(raw, sqrt_n)

    def _linreg_slope_and_r2(y: np.ndarray) -> tuple[float, float]:
        # Simple least squares slope and R²
        if y is None or len(y) < 3:
            return (np.nan, np.nan)
        x = np.arange(len(y), dtype=float)
        x_mean = x.mean()
        y_mean = float(np.mean(y))
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return (np.nan, np.nan)
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        y_hat = intercept + slope * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else np.nan
        return (slope, r2)

    def _get_daily_df(sym: str) -> pd.DataFrame:
        """
        Uses your app's existing data getter if present; otherwise falls back to yfinance.
        """
        # Try common existing app fetchers first
        for fn_name in ("get_price_data", "fetch_price_data", "download_data", "get_ohlc", "load_ohlc"):
            fn = globals().get(fn_name)
            if callable(fn):
                try:
                    df = fn(sym, period="6mo", interval="1d")  # type: ignore
                    if isinstance(df, pd.DataFrame) and len(df) > 10:
                        return df.copy()
                except Exception:
                    pass

        # Fallback: yfinance
        try:
            df = yf.download(sym, period="6mo", interval="1d", progress=False)
            if isinstance(df, pd.DataFrame) and len(df) > 10:
                return df.copy()
        except Exception:
            pass

        return pd.DataFrame()

    def _recent_upward_cross(close: pd.Series, hma: pd.Series, lookback: int) -> int | None:
        """
        Returns bars_ago (1..lookback) if an upward cross occurred within lookback bars, else None.
        We detect sign change from <=0 to >0 for (close - hma).
        """
        s = (close - hma).dropna()
        if len(s) < lookback + 2:
            return None

        # Align indices
        s = s.iloc[-(lookback + 2):]
        vals = s.values

        # Check each of the last `lookback` transitions
        # bars_ago = 1 means cross happened between [-2] and [-1]
        for bars_ago in range(1, lookback + 1):
            prev = vals[-(bars_ago + 1)]
            curr = vals[-bars_ago]
            if prev <= 0 and curr > 0:
                return bars_ago
        return None

    # --- Run scan ---
    st.caption(f"Scanning {len(universe)} symbols…")
    prog = st.progress(0)
    rows_pos = []
    rows_neg = []

    for idx, sym in enumerate(universe, start=1):
        prog.progress(int((idx / max(1, len(universe))) * 100))

        df = _get_daily_df(sym)
        if df is None or df.empty:
            continue

        # Normalize columns
        if "Close" not in df.columns:
            # If multiindex from yf, try flatten
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
            except Exception:
                pass
        if "Close" not in df.columns:
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < max(80, reg_window + 5):
            continue

        hma55 = _hma(close, 55).dropna()
        if len(hma55) < 10:
            continue

        bars_ago = _recent_upward_cross(close, hma55, hma_lookback)
        if bars_ago is None:
            continue

        # Regression slope computed over last `reg_window` closes
        y = close.iloc[-reg_window:].values.astype(float)
        slope, r2 = _linreg_slope_and_r2(y)

        last_close = float(close.iloc[-1])
        last_hma = float(hma55.iloc[-1])

        row = {
            "Symbol": sym,
            "Bars Ago": int(bars_ago),
            "Close": last_close,
            "HMA(55)": last_hma,
            "Regression Slope": slope,
            "R²": r2,
        }

        if np.isfinite(slope) and slope > 0:
            rows_pos.append(row)
        else:
            rows_neg.append(row)

    prog.empty()

    # --- Present results ---
    def _to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["Symbol", "Bars Ago", "Close", "HMA(55)", "Regression Slope", "R²"])
        out = pd.DataFrame(rows)
        out = out.sort_values(["Bars Ago", "R²"], ascending=[True, False], kind="mergesort")
        return out.reset_index(drop=True)

    df_pos = _to_df(rows_pos)
    df_neg = _to_df(rows_neg)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Regression > 0")
        st.dataframe(df_pos, use_container_width=True)

    with colB:
        st.markdown("### Regression < 0 (or not finite)")
        st.dataframe(df_neg, use_container_width=True)

    st.caption("Definition: an **upward cross** is when Close moves from ≤ HMA(55) to > HMA(55) within the selected lookback window on the Daily chart.")


# =========================
# END OF FILE
# =========================
