# =========================
# Batch 1/12 — bullbear.py
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
    st.session_state.pop("chart", None)
    st.session_state.pop("hour_range", None)
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
# Aesthetic helper (no logic change)
# ---------------------------
def style_axes(ax):
    """Simple, consistent, user-friendly chart styling."""
    try:
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
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

# FX helpers
def pip_size_for_symbol(symbol: str):
    s = str(symbol).upper()
    if "=X" not in s:
        return None
    return 0.01 if "JPY" in s else 0.0001

def _diff_text(a: float, b: float, symbol: str) -> str:
    try:
        av = float(a); bv = float(b)
    except Exception:
        return ""
    ps = pip_size_for_symbol(symbol)
    diff = abs(bv - av)
    if ps:
        return f"{diff/ps:.1f} pips"
    return f"Δ {diff:.3f}"

ALERT_TEXT = "ALERT: Trend may be changing - Open trade position with caution while still following the signals on the chat."

def format_trade_instruction(trend_slope: float,
                             buy_val: float,
                             sell_val: float,
                             close_val: float,
                             symbol: str,
                             global_trend_slope: float = None) -> str:
    """
    UPDATED (prior request):
      - Show BUY instruction only when Global Trendline slope and Local Slope agree (both UP)
      - Show SELL instruction only when Global Trendline slope and Local Slope agree (both DOWN)
      - Otherwise show an alert message.

    Backward-compatibility:
      - If global_trend_slope is None, falls back to the prior behavior (uses only trend_slope).
    """
    def _finite(x):
        try:
            return np.isfinite(float(x))
        except Exception:
            return False

    entry_buy = float(buy_val) if _finite(buy_val) else float(close_val)
    exit_sell = float(sell_val) if _finite(sell_val) else float(close_val)

    if global_trend_slope is None:
        uptrend = False
        try:
            uptrend = float(trend_slope) >= 0.0
        except Exception:
            pass

        if uptrend:
            leg_a_val, leg_b_val = entry_buy, exit_sell
            text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        else:
            leg_a_val, leg_b_val = exit_sell, entry_buy
            text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"

        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    try:
        g = float(global_trend_slope)
        l = float(trend_slope)
    except Exception:
        g = np.nan
        l = np.nan

    alert_txt = ALERT_TEXT

    if (not np.isfinite(g)) or (not np.isfinite(l)):
        return alert_txt

    sg = float(np.sign(g))
    sl = float(np.sign(l))

    if sg == 0.0 or sl == 0.0:
        return alert_txt

    if sg > 0 and sl > 0:
        leg_a_val, leg_b_val = entry_buy, exit_sell
        text = f"▲ BUY @{fmt_price_val(leg_a_val)} → ▼ SELL @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    if sg < 0 and sl < 0:
        leg_a_val, leg_b_val = exit_sell, entry_buy
        text = f"▼ SELL @{fmt_price_val(leg_a_val)} → ▲ BUY @{fmt_price_val(leg_b_val)}"
        text += f" • {_diff_text(leg_a_val, leg_b_val, symbol)}"
        return text

    return alert_txt
# =========================
# Batch 2/12 — bullbear.py
# =========================
# ---------------------------
# Gapless (continuous) intraday prices
# ---------------------------
def make_gapless_ohlc(df: pd.DataFrame,
                      price_cols=("Open", "High", "Low", "Close"),
                      gap_mult: float = 12.0,
                      min_gap_seconds: float = 3600.0) -> pd.DataFrame:
    """
    Remove *price gaps* at session breaks by applying a cumulative offset so that
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

def _map_times_to_bar_positions(real_times: pd.DatetimeIndex, times_list):
    if not isinstance(real_times, pd.DatetimeIndex) or real_times.empty:
        return []
    if times_list is None:
        return []
    try:
        t = pd.to_datetime(list(times_list))
    except Exception:
        return []
    if len(t) == 0:
        return []
    try:
        idxer = real_times.get_indexer(t, method="nearest")
    except Exception:
        return []
    pos = [int(i) for i in idxer if int(i) >= 0]
    return pos

# ---------------------------
# Sidebar configuration
# ---------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown(f"### Asset Class: **{mode}**")

if st.sidebar.button("🧹 Clear cache (data + run state)", use_container_width=True, key="btn_clear_cache"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    _reset_run_state_for_mode_switch()
    for k in ["sb_show_fibs", "sb_show_mom_hourly", "sb_show_macd"]:
        if k in st.session_state:
            try:
                del st.session_state[k]
            except Exception:
                pass
    try:
        st.experimental_rerun()
    except Exception:
        pass

bb_period = st.sidebar.selectbox("Bull/Bear Lookback:", ["1mo", "3mo", "6mo", "1y"], index=2, key="sb_bb_period")
daily_view = st.sidebar.selectbox("Daily view range:", ["Historical", "6M", "12M", "24M"], index=2, key="sb_daily_view")

show_fibs = st.sidebar.checkbox("Show Fibonacci (hourly only)", value=True, key="sb_show_fibs")

slope_lb_daily  = st.sidebar.slider("Daily slope lookback (bars)", 10, 360, 90, 10, key="sb_slope_lb_daily")
slope_lb_hourly = st.sidebar.slider("Hourly slope lookback (bars)", 12, 480, 120, 6, key="sb_slope_lb_hourly")

st.sidebar.subheader("MACD")
show_macd = st.sidebar.checkbox("Show MACD chart", value=False, key="sb_show_macd")

st.sidebar.subheader("Slope Reversal Probability (experimental)")
rev_hist_lb = st.sidebar.slider("History window for reversal stats (bars)", 30, 720, 240, 30, key="sb_rev_hist_lb")
rev_horizon = st.sidebar.slider("Forward horizon for reversal (bars)", 3, 60, 15, 1, key="sb_rev_horizon")

st.sidebar.subheader("Daily Support/Resistance Window")
sr_lb_daily = st.sidebar.slider("Daily S/R lookback (bars)", 20, 252, 60, 5, key="sb_sr_lb_daily")

st.sidebar.subheader("Hourly Support/Resistance Window")
sr_lb_hourly = st.sidebar.slider("Hourly S/R lookback (bars)", 20, 240, 60, 5, key="sb_sr_lb_hourly")

st.sidebar.subheader("Hourly Momentum")
show_mom_hourly = st.sidebar.checkbox("Show hourly momentum (ROC%)", value=False, key="sb_show_mom_hourly")
mom_lb_hourly = st.sidebar.slider("Momentum lookback (bars)", 3, 120, 12, 1, key="sb_mom_lb_hourly")

st.sidebar.subheader("Hourly Indicator Panel")
show_nrsi = st.sidebar.checkbox("Show Hourly NTD panel", value=True, key="sb_show_nrsi")
nrsi_period = st.sidebar.slider("RSI period (unused)", 5, 60, 14, 1, key="sb_nrsi_period")

st.sidebar.subheader("NTD Channel (Hourly)")
show_ntd_channel = st.sidebar.checkbox(
    "Highlight when price is between S/R (S↔R) on NTD",
    value=True, key="sb_ntd_channel"
)

st.sidebar.subheader("Hourly Supertrend")
atr_period = st.sidebar.slider("ATR period", 5, 50, 10, 1, key="sb_atr_period")
atr_mult = st.sidebar.slider("ATR multiplier", 1.0, 5.0, 3.0, 0.5, key="sb_atr_mult")

st.sidebar.subheader("Parabolic SAR")
show_psar = st.sidebar.checkbox("Show Parabolic SAR", value=True, key="sb_psar_show")
psar_step = st.sidebar.slider("PSAR acceleration step", 0.01, 0.20, 0.02, 0.01, key="sb_psar_step")
psar_max  = st.sidebar.slider("PSAR max acceleration", 0.10, 1.00, 0.20, 0.10, key="sb_psar_max")

st.sidebar.subheader("Signal Logic")
signal_threshold = st.sidebar.slider("S/R proximity signal threshold", 0.50, 0.99, 0.90, 0.01, key="sb_sig_thr")
sr_prox_pct = st.sidebar.slider("S/R proximity (%)", 0.05, 1.00, 0.25, 0.05, key="sb_sr_prox") / 100.0

st.sidebar.subheader("NTD (Daily/Hourly)")
# UPDATED (THIS REQUEST): use a new key so Daily NTD displays ON by default again
show_ntd = st.sidebar.checkbox("Show NTD overlay", value=True, key="sb_show_ntd_v2")
ntd_window = st.sidebar.slider("NTD slope window", 10, 300, 60, 5, key="sb_ntd_win")
shade_ntd = st.sidebar.checkbox("Shade NTD (green=up, red=down)", value=True, key="sb_ntd_shade")
show_npx_ntd = st.sidebar.checkbox("Overlay normalized price (NPX) on NTD", value=True, key="sb_show_npx_ntd")
mark_npx_cross = st.sidebar.checkbox("Mark NPX↔NTD crosses (dots)", value=True, key="sb_mark_npx_cross")

st.sidebar.subheader("Normalized Ichimoku (Kijun on price)")
show_ichi = st.sidebar.checkbox("Show Ichimoku Kijun on price", value=True, key="sb_show_ichi")
ichi_conv = st.sidebar.slider("Conversion (Tenkan)", 5, 20, 9, 1, key="sb_ichi_conv")
ichi_base = st.sidebar.slider("Base (Kijun)", 20, 40, 26, 1, key="sb_ichi_base")
ichi_spanb = st.sidebar.slider("Span B", 40, 80, 52, 1, key="sb_ichi_spanb")

st.sidebar.subheader("Bollinger Bands (Price Charts)")
show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True, key="sb_show_bbands")
bb_win = st.sidebar.slider("BB window", 5, 120, 20, 1, key="sb_bb_win")
bb_mult = st.sidebar.slider("BB multiplier (σ)", 1.0, 4.0, 2.0, 0.1, key="sb_bb_mult")
bb_use_ema = st.sidebar.checkbox("Use EMA midline (vs SMA)", value=False, key="sb_bb_ema")

st.sidebar.subheader("Probabilistic HMA Crossover (Price Charts)")
show_hma = st.sidebar.checkbox("Show HMA crossover signal", value=True, key="sb_hma_show")
hma_period = st.sidebar.slider("HMA period", 5, 120, 55, 1, key="sb_hma_period")
hma_conf = st.sidebar.slider("Crossover confidence (unused label-only)", 0.50, 0.99, 0.95, 0.01, key="sb_hma_conf")

st.sidebar.subheader("HMA(55) Reversal on NTD")
show_hma_rev_ntd = st.sidebar.checkbox("Mark HMA cross + slope reversal on NTD", value=True, key="sb_hma_rev_ntd")
hma_rev_lb = st.sidebar.slider("HMA reversal slope lookback (bars)", 2, 10, 3, 1, key="sb_hma_rev_lb")

st.sidebar.subheader("Reversal Stars (on NTD panel)")
rev_bars_confirm = st.sidebar.slider("Consecutive bars to confirm reversal", 1, 4, 2, 1, key="sb_rev_bars")

if mode == "Forex":
    show_fx_news = st.sidebar.checkbox("Show Forex news markers (intraday)", value=True, key="sb_show_fx_news")
    news_window_days = st.sidebar.slider("Forex news window (days)", 1, 14, 7, key="sb_news_window_days")
    st.sidebar.subheader("Sessions (PST)")
    show_sessions_pst = st.sidebar.checkbox("Show London/NY session times (PST)", value=True, key="sb_show_sessions_pst")
else:
    show_fx_news = False
    news_window_days = 7
    show_sessions_pst = False
# =========================
# Batch 3/12 — bullbear.py
# =========================
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
        "HKDJPY=X","USDCAD=X","USDCNY=X","USDCHF=X","EURGBP=X","EURCAD=X","NZDJPY=X","USDKRW=X",
        "USDHKD=X","EURHKD=X","GBPHKD=X","GBPJPY=X","CNHJPY=X","AUDJPY=X","GBPCAD=X"
    ]

# ---------------------------
# Data fetchers
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hist(ticker: str) -> pd.Series:
    s = (yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))["Close"]
         .asfreq("D").ffill())
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_max(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="max")[["Close"]].dropna()
    s = df["Close"].asfreq("D").ffill()
    try:
        s = s.tz_localize(PACIFIC)
    except TypeError:
        s = s.tz_convert(PACIFIC)
    return s

@st.cache_data(ttl=120)
def fetch_hist_ohlc(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2018-01-01", end=pd.to_datetime("today"))[
        ["Open","High","Low","Close"]
    ].dropna()
    try:
        df = df.tz_localize(PACIFIC)
    except TypeError:
        df = df.tz_convert(PACIFIC)
    return df

@st.cache_data(ttl=120)
def fetch_intraday(ticker: str, period: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="5m")
    if df is None or df.empty:
        return df
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)

    if {"Open","High","Low","Close"}.issubset(df.columns):
        df = make_gapless_ohlc(df)

    return df

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
            series,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
    fc = model.get_forecast(steps=30)
    idx = pd.date_range(series.index[-1] + timedelta(days=1), periods=30, freq="D", tz=PACIFIC)
    return idx, fc.predicted_mean, fc.conf_int()

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

def current_daily_pivots(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty or not {"High","Low","Close"}.issubset(ohlc.columns):
        return {}
    ohlc = ohlc.sort_index()
    row = ohlc.iloc[-2] if len(ohlc) >= 2 else ohlc.iloc[-1]
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P  = (H + L + C) / 3.0
    R1 = 2 * P - L; S1 = 2 * P - H
    R2 = P + (H - L); S2 = P - (H - L)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2}
# =========================
# Batch 4/12 — bullbear.py
# =========================
# ---------------------------
# Regression & ±2σ band
# ---------------------------
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

def regression_r2(series_like, lookback: int):
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
    if s.shape[0] < 2:
        return float("nan")
    x = np.arange(len(s), dtype=float)
    y = s.to_numpy(dtype=float)
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res/ss_tot)

def regression_with_band(series_like, lookback: int = 0, z: float = 2.0):
    """
    Linear regression on last `lookback` bars with:
      • fitted trendline
      • symmetric ±z·σ band (σ = std of residuals)
      • R² of the fit
    """
    s = _coerce_1d_series(series_like).dropna()
    if lookback > 0:
        s = s.iloc[-lookback:]
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
    upper_s = pd.Series(yhat + z * std, index=s.index)
    lower_s = pd.Series(yhat - z * std, index=s.index)
    return yhat_s, upper_s, lower_s, float(m), r2

def slope_reversal_probability(series_like,
                               current_slope: float,
                               hist_window: int = 240,
                               slope_window: int = 60,
                               horizon: int = 15) -> float:
    s = _coerce_1d_series(series_like).dropna()
    n = len(s)
    if n < slope_window + horizon + 5:
        return float("nan")

    try:
        sign_curr = np.sign(float(current_slope))
    except Exception:
        return float("nan")
    if not np.isfinite(sign_curr) or sign_curr == 0.0:
        return float("nan")

    start = max(slope_window - 1, n - hist_window - horizon)
    end = n - horizon - 1
    if end <= start:
        return float("nan")

    match = 0
    flips = 0
    for i in range(start, end + 1):
        past_start = i - slope_window + 1
        if past_start < 0:
            continue
        past_change = s.iloc[i] - s.iloc[past_start]
        sign_past = np.sign(past_change)
        if not np.isfinite(sign_past) or sign_past == 0.0:
            continue
        if sign_past != sign_curr:
            continue
        future_change = s.iloc[i + horizon] - s.iloc[i]
        sign_future = np.sign(future_change)
        if not np.isfinite(sign_future) or sign_future == 0.0:
            continue
        match += 1
        if sign_future != sign_past:
            flips += 1

    if match == 0:
        return float("nan")
    return float(flips / match)

def find_band_bounce_signal(price: pd.Series,
                            upper_band: pd.Series,
                            lower_band: pd.Series,
                            slope_val: float):
    """
    Detect the most recent BUY/SELL signal based on a 'bounce' off the ±2σ band.
    """
    p = _coerce_1d_series(price)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    mask = p.notna() & u.notna() & l.notna()
    if mask.sum() < 2:
        return None

    p = p[mask]
    u = u.reindex(p.index)
    l = l.reindex(p.index)

    inside = (p <= u) & (p >= l)
    below  = p < l
    above  = p > u

    try:
        slope = float(slope_val)
    except Exception:
        slope = np.nan
    if not np.isfinite(slope) or slope == 0.0:
        return None

    if slope > 0:
        candidates = inside & below.shift(1, fill_value=False)
        idx = list(candidates[candidates].index)
        if not idx:
            return None
        t = idx[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "BUY"}
    else:
        candidates = inside & above.shift(1, fill_value=False)
        idx = list(candidates[candidates].index)
        if not idx:
            return None
        t = idx[-1]
        return {"time": t, "price": float(p.loc[t]), "side": "SELL"}

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

def annotate_crossover(ax, ts, px, side: str, note: str = ""):
    if side == "BUY":
        ax.scatter([ts], [px], marker="P", s=90, color="tab:green", zorder=7)
        label = "BUY" if not note else f"BUY {note}"
        ax.text(ts, px, f"  {label}", va="bottom", fontsize=9,
                color="tab:green", fontweight="bold")
    else:
        ax.scatter([ts], [px], marker="X", s=90, color="tab:red", zorder=7)
        label = "SELL" if not note else f"SELL {note}"
        ax.text(ts, px, f"  {label}", va="top", fontsize=9,
                color="tab:red", fontweight="bold")
# =========================
# Batch 5/12 — bullbear.py
# =========================
# ---------------------------
# NEW (THIS REQUEST): Slope BUY/SELL Trigger (leaderline + legend)
# ---------------------------
def find_slope_trigger_after_band_reversal(price: pd.Series,
                                          yhat: pd.Series,
                                          upper_band: pd.Series,
                                          lower_band: pd.Series,
                                          horizon: int = 15):
    """
    BUY trigger:
      - price touches/breaches LOWER band, then crosses ABOVE the slope line (yhat)
    SELL trigger:
      - price touches/breaches UPPER band, then crosses BELOW the slope line (yhat)
    Returns the most recent trigger dict or None.
    """
    p = _coerce_1d_series(price)
    y = _coerce_1d_series(yhat).reindex(p.index)
    u = _coerce_1d_series(upper_band).reindex(p.index)
    l = _coerce_1d_series(lower_band).reindex(p.index)

    ok = p.notna() & y.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None
    p = p[ok]; y = y[ok]; u = u[ok]; l = l[ok]

    cross_up, cross_dn = _cross_series(p, y)
    below = (p <= l)
    above = (p >= u)

    hz = max(1, int(horizon))

    def _last_touch_before(t_idx, touch_mask: pd.Series):
        try:
            loc = int(p.index.get_loc(t_idx))
        except Exception:
            return None
        j0 = max(0, loc - hz)
        window = touch_mask.iloc[j0:loc+1]
        if not window.any():
            return None
        return window[window].index[-1]

    last_buy_cross = cross_up[cross_up].index[-1] if cross_up.any() else None
    last_sell_cross = cross_dn[cross_dn].index[-1] if cross_dn.any() else None

    buy_tr = None
    if last_buy_cross is not None:
        t_touch = _last_touch_before(last_buy_cross, below)
        if t_touch is not None:
            buy_tr = {
                "side": "BUY",
                "touch_time": t_touch,
                "touch_price": float(p.loc[t_touch]),
                "cross_time": last_buy_cross,
                "cross_price": float(p.loc[last_buy_cross]),
            }

    sell_tr = None
    if last_sell_cross is not None:
        t_touch = _last_touch_before(last_sell_cross, above)
        if t_touch is not None:
            sell_tr = {
                "side": "SELL",
                "touch_time": t_touch,
                "touch_price": float(p.loc[t_touch]),
                "cross_time": last_sell_cross,
                "cross_price": float(p.loc[last_sell_cross]),
            }

    if buy_tr is None and sell_tr is None:
        return None
    if buy_tr is None:
        return sell_tr
    if sell_tr is None:
        return buy_tr

    return buy_tr if buy_tr["cross_time"] >= sell_tr["cross_time"] else sell_tr

def annotate_slope_trigger(ax, trig: dict):
    if trig is None:
        return
    side = trig.get("side", "")
    t0 = trig.get("touch_time")
    p0 = trig.get("touch_price")
    t1 = trig.get("cross_time")
    p1 = trig.get("cross_price")
    if t0 is None or t1 is None:
        return
    if not (np.isfinite(p0) and np.isfinite(p1)):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    lbl = f"Slope {side} Trigger"
    ax.annotate(
        "",
        xy=(t1, p1),
        xytext=(t0, p0),
        arrowprops=dict(arrowstyle="->", color=col, lw=2.0, alpha=0.85),
        zorder=9
    )
    ax.scatter([t1], [p1], marker="o", s=90, color=col, zorder=10, label=lbl)
    ax.text(
        t1, p1,
        f"  {lbl}",
        color=col,
        fontsize=9,
        fontweight="bold",
        va="bottom" if side == "BUY" else "top",
        zorder=10
    )

# ---------------------------
# Other indicators
# ---------------------------
def compute_roc(series_like, n: int = 10) -> pd.Series:
    s = _coerce_1d_series(series_like)
    base = s.dropna()
    if base.empty:
        return pd.Series(index=s.index, dtype=float)
    roc = base.pct_change(n) * 100.0
    return roc.reindex(s.index)

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    s = _coerce_1d_series(close).astype(float)
    if s.empty or period < 2:
        return pd.Series(index=s.index, dtype=float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.reindex(s.index)

def compute_nrsi(close: pd.Series, period: int = 14) -> pd.Series:
    rsi = compute_rsi(close, period=period)
    return ((rsi - 50.0) / 50.0).clip(-1.0, 1.0).reindex(rsi.index)

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty, empty
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    hist = macd - sig
    return macd.reindex(s.index), sig.reindex(s.index), hist.reindex(s.index)
# =========================
# Batch 6/12 — bullbear.py
# =========================
def compute_nmacd(close: pd.Series, fast: int = 12, slow: int = 26,
                  signal: int = 9, norm_win: int = 240):
    s = _coerce_1d_series(close).astype(float)
    if s.empty:
        return (pd.Series(index=s.index, dtype=float),)*3
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean()
    macd = ema_fast - ema_slow
    sig  = macd.ewm(span=int(signal), adjust=False).mean()
    minp = max(10, norm_win//10)

    def _norm(x):
        m = x.rolling(norm_win, min_periods=minp).mean()
        sd = x.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
        z = (x - m) / sd
        return np.tanh(z / 2.0)

    nmacd = _norm(macd)
    nsignal = _norm(sig)
    nhist = nmacd - nsignal
    return (nmacd.reindex(s.index), nsignal.reindex(s.index), nhist.reindex(s.index))

def compute_nvol(volume: pd.Series, norm_win: int = 240) -> pd.Series:
    v = _coerce_1d_series(volume).astype(float)
    if v.empty:
        return pd.Series(index=v.index, dtype=float)
    minp = max(10, norm_win//10)
    m = v.rolling(norm_win, min_periods=minp).mean()
    sd = v.rolling(norm_win, min_periods=minp).std().replace(0, np.nan)
    z = (v - m) / sd
    return np.tanh(z / 3.0)

def compute_npo(close: pd.Series, fast: int = 12, slow: int = 26, norm_win: int = 240) -> pd.Series:
    s = _coerce_1d_series(close)
    if s.empty or fast <= 0 or slow <= 0:
        return pd.Series(index=s.index, dtype=float)
    if fast >= slow:
        fast = max(1, slow - 1)
        if fast >= slow:
            return pd.Series(index=s.index, dtype=float)
    ema_fast = s.ewm(span=int(fast), adjust=False).mean()
    ema_slow = s.ewm(span=int(slow), adjust=False).mean().replace(0, np.nan)
    ppo = (ema_fast - ema_slow) / ema_slow * 100.0
    minp = max(10, int(norm_win)//10)
    mean = ppo.rolling(int(norm_win), min_periods=minp).mean()
    std  = ppo.rolling(int(norm_win), min_periods=minp).std().replace(0, np.nan)
    z = (ppo - mean) / std
    return np.tanh(z / 2.0).reindex(s.index)

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

def shade_ntd_regions(ax, ntd: pd.Series):
    if ntd is None or ntd.empty:
        return
    ntd = ntd.copy()
    pos = ntd.where(ntd > 0)
    neg = ntd.where(ntd < 0)
    ax.fill_between(ntd.index, 0, pos, alpha=0.12, color="tab:green")
    ax.fill_between(ntd.index, 0, neg, alpha=0.12, color="tab:red")

def draw_trend_direction_line(ax, series_like: pd.Series, label_prefix: str = "Trend"):
    s = _coerce_1d_series(series_like).dropna()
    if s.shape[0] < 2:
        return np.nan
    x = np.arange(len(s), dtype=float)
    m, b = np.polyfit(x, s.values, 1)
    yhat = m * x + b
    color = "green" if m >= 0 else "red"
    ax.plot(s.index, yhat, linestyle="--", linewidth=2.4, color=color,
            label=f"{label_prefix} ({fmt_slope(m)}/bar)")
    return float(m)

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
# =========================
# Batch 7/12 — bullbear.py
# =========================
def find_macd_hma_sr_signal(close: pd.Series,
                            hma: pd.Series,
                            macd: pd.Series,
                            sup: pd.Series,
                            res: pd.Series,
                            global_trend_slope: float,
                            prox: float = 0.0025):
    c = _coerce_1d_series(close).astype(float)
    h = _coerce_1d_series(hma).reindex(c.index)
    m = _coerce_1d_series(macd).reindex(c.index)
    s_sup = _coerce_1d_series(sup).reindex(c.index).ffill()
    s_res = _coerce_1d_series(res).reindex(c.index).ffill()

    ok = c.notna() & h.notna() & m.notna() & s_sup.notna() & s_res.notna()
    if ok.sum() < 3:
        return None

    c = c[ok]; h = h[ok]; m = m[ok]; s_sup = s_sup[ok]; s_res = s_res[ok]

    cross_up, cross_dn = _cross_series(c, h)
    cross_up = cross_up.reindex(c.index, fill_value=False)
    cross_dn = cross_dn.reindex(c.index, fill_value=False)

    near_support = c <= s_sup * (1.0 + prox)
    away_from_support = (c - s_sup) > (c.shift(1) - s_sup.shift(1))
    near_resist = c >= s_res * (1.0 - prox)
    away_from_resist = (s_res - c) > (s_res.shift(1) - c.shift(1))

    uptrend = np.isfinite(global_trend_slope) and float(global_trend_slope) > 0
    downtrend = np.isfinite(global_trend_slope) and float(global_trend_slope) < 0

    buy_mask = uptrend & (m < 0.0) & cross_up & near_support & away_from_support
    sell_mask = downtrend & (m > 0.0) & cross_dn & near_resist & away_from_resist

    last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
    last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None
    if last_buy is None and last_sell is None:
        return None

    if last_sell is None:
        t = last_buy; side = "BUY"
    elif last_buy is None:
        t = last_sell; side = "SELL"
    else:
        t = last_buy if last_buy >= last_sell else last_sell
        side = "BUY" if t == last_buy else "SELL"

    px = float(c.loc[t]) if np.isfinite(c.loc[t]) else np.nan
    note = "MACD/HMA55 + S/R"
    return {"time": t, "price": px, "side": side, "note": note}

def annotate_macd_signal(ax, ts, px, side: str):
    if side == "BUY":
        ax.scatter([ts], [px], marker="*", s=180, color="tab:green", zorder=10, label="MACD BUY (HMA55+S/R)")
    else:
        ax.scatter([ts], [px], marker="*", s=180, color="tab:red", zorder=10, label="MACD SELL (HMA55+S/R)")

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

# ---------------------------
# NEW (THIS REQUEST): BB "move-away" triangles with 99% confidence (R² >= 0.99)
# ---------------------------
def find_bb_moveaway_signal(close: pd.Series,
                            bb_upper: pd.Series,
                            bb_lower: pd.Series,
                            slope_val: float,
                            r2_val: float,
                            conf_r2: float = 0.99):
    """
    BB Buy:
      - slope > 0 (upward slope)
      - R² >= conf_r2 (99% confidence)
      - prior bar touched/breached LOWER BB
      - current bar moved away upward: close > lower BB, close increasing, distance from lower increasing

    BB Sell:
      - slope < 0 (downward slope)
      - R² >= conf_r2 (99% confidence)
      - prior bar touched/breached UPPER BB
      - current bar moved away downward: close < upper BB, close decreasing, distance from upper increasing

    Returns most recent signal dict or None.
    """
    c = _coerce_1d_series(close).astype(float)
    u = _coerce_1d_series(bb_upper).reindex(c.index).astype(float)
    l = _coerce_1d_series(bb_lower).reindex(c.index).astype(float)

    ok = c.notna() & u.notna() & l.notna()
    if ok.sum() < 3:
        return None

    c = c[ok]; u = u[ok]; l = l[ok]

    try:
        m = float(slope_val)
        r2 = float(r2_val)
    except Exception:
        return None

    if (not np.isfinite(m)) or (not np.isfinite(r2)) or float(r2) < float(conf_r2):
        return None

    prev_c = c.shift(1)
    prev_u = u.shift(1)
    prev_l = l.shift(1)

    if m > 0.0:
        prev_touch = prev_c <= prev_l
        now_inside = c > l
        moving_up = c > prev_c
        away_from_lower = (c - l) > (prev_c - prev_l)
        mask = (prev_touch & now_inside & moving_up & away_from_lower).fillna(False)
        if not mask.any():
            return None
        t = mask[mask].index[-1]
        return {"time": t, "price": float(c.loc[t]), "side": "BB_BUY"}

    if m < 0.0:
        prev_touch = prev_c >= prev_u
        now_inside = c < u
        moving_dn = c < prev_c
        away_from_upper = (u - c) > (prev_u - prev_c)
        mask = (prev_touch & now_inside & moving_dn & away_from_upper).fillna(False)
        if not mask.any():
            return None
        t = mask[mask].index[-1]
        return {"time": t, "price": float(c.loc[t]), "side": "BB_SELL"}

    return None

def annotate_bb_moveaway(ax, sig: dict):
    if sig is None:
        return
    t = sig.get("time", None)
    px = sig.get("price", None)
    side = sig.get("side", "")
    if t is None or px is None:
        return
    try:
        px = float(px)
    except Exception:
        return
    if not np.isfinite(px):
        return

    if side == "BB_BUY":
        ax.scatter([t], [px], marker="^", s=140, color="tab:green", zorder=12, label="BB Buy")
        ax.text(t, px, "  BB Buy", va="bottom", fontsize=9, color="tab:green", fontweight="bold", zorder=12)
    elif side == "BB_SELL":
        ax.scatter([t], [px], marker="v", s=140, color="tab:red", zorder=12, label="BB Sell")
        ax.text(t, px, "  BB Sell", va="top", fontsize=9, color="tab:red", fontweight="bold", zorder=12)
# =========================
# Batch 8/12 — bullbear.py
# =========================
# ---------------------------
# Ichimoku (Kijun) + PSAR + Supertrend
# ---------------------------
def compute_kijun(high: pd.Series, low: pd.Series, base: int = 26) -> pd.Series:
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    idx = h.index if len(h) else l.index
    if h.empty or l.empty or base < 2:
        return pd.Series(index=idx, dtype=float)
    hh = h.rolling(base, min_periods=max(2, base//2)).max()
    ll = l.rolling(base, min_periods=max(2, base//2)).min()
    kijun = (hh + ll) / 2.0
    return kijun.reindex(idx)

def compute_psar(df_ohlc: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    """
    Classic Parabolic SAR.
    Returns a Series aligned to df_ohlc.index.
    """
    if df_ohlc is None or df_ohlc.empty:
        return pd.Series(dtype=float)
    if not {"High","Low","Close"}.issubset(df_ohlc.columns):
        return pd.Series(index=df_ohlc.index, dtype=float)

    high = pd.to_numeric(df_ohlc["High"], errors="coerce").to_numpy(dtype=float)
    low  = pd.to_numeric(df_ohlc["Low"], errors="coerce").to_numpy(dtype=float)
    close= pd.to_numeric(df_ohlc["Close"], errors="coerce").to_numpy(dtype=float)
    n = len(df_ohlc.index)
    if n < 3:
        return pd.Series(index=df_ohlc.index, dtype=float)

    psar = np.full(n, np.nan, dtype=float)

    # Initialize trend by first movement
    bull = close[1] >= close[0]
    af = float(step)
    ep = high[0] if bull else low[0]
    psar[0] = low[0] if bull else high[0]

    for i in range(1, n):
        prev = psar[i-1]
        if not np.isfinite(prev) or not np.isfinite(ep):
            psar[i] = np.nan
            continue

        # core step
        psar[i] = prev + af * (ep - prev)

        # clamp to prior extremes
        if bull:
            if i >= 2:
                psar[i] = min(psar[i], low[i-1], low[i-2])
            else:
                psar[i] = min(psar[i], low[i-1])
        else:
            if i >= 2:
                psar[i] = max(psar[i], high[i-1], high[i-2])
            else:
                psar[i] = max(psar[i], high[i-1])

        # reversal test
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = float(step)
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(float(max_step), af + float(step))
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = float(step)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(float(max_step), af + float(step))

    return pd.Series(psar, index=df_ohlc.index)

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h = _coerce_1d_series(high).astype(float)
    l = _coerce_1d_series(low).astype(float)
    c = _coerce_1d_series(close).astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.reindex(c.index)

def compute_atr(df_ohlc: pd.DataFrame, period: int = 10) -> pd.Series:
    if df_ohlc is None or df_ohlc.empty or not {"High","Low","Close"}.issubset(df_ohlc.columns):
        return pd.Series(dtype=float)
    tr = _true_range(df_ohlc["High"], df_ohlc["Low"], df_ohlc["Close"])
    atr = tr.rolling(int(period), min_periods=max(2, int(period)//2)).mean()
    return atr.reindex(df_ohlc.index)

def compute_supertrend(df_ohlc: pd.DataFrame, atr_period: int = 10, atr_mult: float = 3.0):
    """
    Returns:
      st_line (Series), st_dir (Series +1 bull / -1 bear),
      upperband (Series), lowerband (Series)
    """
    if df_ohlc is None or df_ohlc.empty or not {"High","Low","Close"}.issubset(df_ohlc.columns):
        idx = df_ohlc.index if isinstance(df_ohlc, pd.DataFrame) else None
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty, empty, empty

    high = _coerce_1d_series(df_ohlc["High"]).astype(float)
    low  = _coerce_1d_series(df_ohlc["Low"]).astype(float)
    close= _coerce_1d_series(df_ohlc["Close"]).astype(float)

    atr = compute_atr(df_ohlc, period=int(atr_period)).astype(float)
    hl2 = (high + low) / 2.0
    upper = hl2 + float(atr_mult) * atr
    lower = hl2 - float(atr_mult) * atr

    # final bands
    fub = upper.copy()
    flb = lower.copy()

    for i in range(1, len(close)):
        if not np.isfinite(fub.iloc[i-1]) or not np.isfinite(flb.iloc[i-1]):
            continue
        if np.isfinite(fub.iloc[i]) and np.isfinite(upper.iloc[i]):
            if (upper.iloc[i] < fub.iloc[i-1]) or (close.iloc[i-1] > fub.iloc[i-1]):
                fub.iloc[i] = upper.iloc[i]
            else:
                fub.iloc[i] = fub.iloc[i-1]
        if np.isfinite(flb.iloc[i]) and np.isfinite(lower.iloc[i]):
            if (lower.iloc[i] > flb.iloc[i-1]) or (close.iloc[i-1] < flb.iloc[i-1]):
                flb.iloc[i] = lower.iloc[i]
            else:
                flb.iloc[i] = flb.iloc[i-1]

    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    # initialize
    st.iloc[0] = np.nan
    direction.iloc[0] = np.nan

    for i in range(1, len(close)):
        prev_st = st.iloc[i-1]
        if not np.isfinite(prev_st):
            # first computable
            st.iloc[i] = fub.iloc[i] if close.iloc[i] < fub.iloc[i] else flb.iloc[i]
        else:
            if prev_st == fub.iloc[i-1]:
                st.iloc[i] = fub.iloc[i] if close.iloc[i] <= fub.iloc[i] else flb.iloc[i]
            else:
                st.iloc[i] = flb.iloc[i] if close.iloc[i] >= flb.iloc[i] else fub.iloc[i]

        direction.iloc[i] = 1.0 if np.isfinite(st.iloc[i]) and st.iloc[i] == flb.iloc[i] else -1.0

    return st, direction, fub, flb

# ---------------------------
# Forex Sessions (PST) + optional (placeholder) news events
# ---------------------------
def london_ny_session_lines_pst(intraday_index: pd.DatetimeIndex):
    """
    Returns approximate London and NY session open times in PST for each date.
    Uses timezone-aware conversion and DST automatically via pytz.
    """
    if not isinstance(intraday_index, pd.DatetimeIndex) or intraday_index.empty:
        return [], []
    idx = intraday_index.tz_convert(PACIFIC) if intraday_index.tz is not None else intraday_index.tz_localize(PACIFIC)

    # London open ~ 08:00 London, NY open ~ 08:00 New York
    london_tz = pytz.timezone("Europe/London")
    ny_tz = pytz.timezone("America/New_York")

    days = pd.to_datetime(sorted(set(idx.normalize())))
    london_opens = []
    ny_opens = []

    for d in days:
        try:
            # 08:00 local session open
            lon_dt = london_tz.localize(datetime(d.year, d.month, d.day, 8, 0, 0)).astimezone(PACIFIC)
            ny_dt  = ny_tz.localize(datetime(d.year, d.month, d.day, 8, 0, 0)).astimezone(PACIFIC)
            london_opens.append(pd.Timestamp(lon_dt))
            ny_opens.append(pd.Timestamp(ny_dt))
        except Exception:
            continue

    return london_opens, ny_opens

def get_forex_news_events_placeholder(ticker: str, lookback_days: int = 7):
    """
    Placeholder: returns empty list (no external news calendar pull).
    Keep this so the rest of the app doesn't break if toggled on.
    """
    return []  # list of dicts: [{"time": pd.Timestamp, "label": "NFP"}, ...]

def annotate_session_and_news(ax, real_times: pd.DatetimeIndex,
                             show_sessions: bool,
                             show_news: bool,
                             ticker: str,
                             news_days: int = 7):
    if (not isinstance(real_times, pd.DatetimeIndex)) or real_times.empty:
        return

    # Sessions
    if show_sessions:
        lon_opens, ny_opens = london_ny_session_lines_pst(real_times)
        lon_pos = _map_times_to_bar_positions(real_times, lon_opens)
        ny_pos  = _map_times_to_bar_positions(real_times, ny_opens)

        for p in lon_pos:
            ax.axvline(p, linestyle="--", linewidth=1.0, alpha=0.35)
        for p in ny_pos:
            ax.axvline(p, linestyle=":", linewidth=1.1, alpha=0.40)

    # News (placeholder)
    if show_news:
        events = get_forex_news_events_placeholder(ticker, lookback_days=news_days)
        if events:
            times = [e.get("time") for e in events if e.get("time") is not None]
            pos = _map_times_to_bar_positions(real_times, times)
            for i, p in enumerate(pos):
                lbl = events[i].get("label", "News")
                ax.axvline(p, linestyle="-.", linewidth=1.0, alpha=0.35)
                ax.text(p, 0.02, lbl, transform=ax.get_xaxis_transform(),
                        rotation=90, va="bottom", fontsize=8, alpha=0.65)
# =========================
# Batch 9/12 — bullbear.py
# =========================
# ---------------------------
# Cached hourly fetch (for charts + scanners)
# ---------------------------
@st.cache_data(ttl=120)
def fetch_hourly_ohlc(ticker: str, period: str = "60d", interval: str = "60m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval)
    if df is None or df.empty:
        return df
    # yfinance often returns naive -> treat as UTC then convert
    try:
        df = df.tz_localize("UTC")
    except TypeError:
        pass
    df = df.tz_convert(PACIFIC)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[cols].dropna()

# ---------------------------
# Support / Resistance helpers
# ---------------------------
def compute_sr(df_ohlc: pd.DataFrame, lookback: int = 60):
    if df_ohlc is None or df_ohlc.empty or not {"High","Low","Close"}.issubset(df_ohlc.columns):
        idx = df_ohlc.index if isinstance(df_ohlc, pd.DataFrame) else None
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    hi = pd.to_numeric(df_ohlc["High"], errors="coerce")
    lo = pd.to_numeric(df_ohlc["Low"], errors="coerce")
    sup = lo.rolling(int(lookback), min_periods=max(3, int(lookback)//3)).min()
    res = hi.rolling(int(lookback), min_periods=max(3, int(lookback)//3)).max()
    return sup.reindex(df_ohlc.index), res.reindex(df_ohlc.index)

# ---------------------------
# R² Trend Scanner (Daily + Hourly)
# ---------------------------
def scan_r2_trends_daily(symbols, lookback: int = 90, r2_min: float = 0.45):
    up = []
    dn = []
    for sym in symbols:
        try:
            s = fetch_hist(sym)
            r2 = regression_r2(s, lookback=lookback)
            _, m = slope_line(s, lookback=lookback)
            if np.isfinite(r2) and np.isfinite(m) and (r2 >= r2_min):
                if m > 0:
                    up.append((sym, r2, m))
                elif m < 0:
                    dn.append((sym, r2, m))
        except Exception:
            continue
    up.sort(key=lambda x: x[1], reverse=True)
    dn.sort(key=lambda x: x[1], reverse=True)
    return up, dn

def scan_r2_trends_hourly(symbols, lookback: int = 120, r2_min: float = 0.45):
    up = []
    dn = []
    for sym in symbols:
        try:
            df = fetch_hourly_ohlc(sym)
            if df is None or df.empty:
                continue
            c = df["Close"]
            r2 = regression_r2(c, lookback=lookback)
            _, m = slope_line(c, lookback=lookback)
            if np.isfinite(r2) and np.isfinite(m) and (r2 >= r2_min):
                if m > 0:
                    up.append((sym, r2, m))
                elif m < 0:
                    dn.append((sym, r2, m))
        except Exception:
            continue
    up.sort(key=lambda x: x[1], reverse=True)
    dn.sort(key=lambda x: x[1], reverse=True)
    return up, dn
# =========================
# Batch 10/12 — bullbear.py
# =========================
# ---------------------------
# Plotting helpers (Daily + Hourly)
# ---------------------------
def plot_price_panel(ax,
                     df_ohlc: pd.DataFrame,
                     symbol: str,
                     slope_lb: int,
                     sr_lb: int,
                     show_fibs_local: bool = False,
                     show_sessions: bool = False,
                     show_news: bool = False,
                     hourly_compact_x: bool = False):
    """
    Main price panel:
      - Close + BBands
      - Regression trendline + ±2σ band + R² badge
      - Supertrend (default ON)
      - Kijun (if enabled)
      - PSAR (if enabled)
      - Support/Resistance
      - Signals:
          • BB move-away triangles (R²>=0.99, slope sign, touch band prior bar, move away)
          • Optional slope trigger arrow (existing helper)
    """
    if df_ohlc is None or df_ohlc.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return {}

    df = df_ohlc.copy().dropna()
    df = df.sort_index()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high  = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else close
    low   = pd.to_numeric(df["Low"], errors="coerce") if "Low"  in df.columns else close

    # BBands
    bb_mid, bb_u, bb_l, bb_pctb, bb_nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # Regression band (used for slope + r2 confidence)
    yhat, up2, lo2, slope_val, r2_val = regression_with_band(close, lookback=int(slope_lb), z=2.0)

    # Support / Resistance
    sup, res = compute_sr(df, lookback=int(sr_lb))

    # Kijun
    kijun = compute_kijun(high, low, base=int(ichi_base)) if show_ichi else pd.Series(index=df.index, dtype=float)

    # PSAR
    psar = compute_psar(df, step=float(psar_step), max_step=float(psar_max)) if show_psar else pd.Series(index=df.index, dtype=float)

    # Supertrend (DEFAULT ON)
    st_line, st_dir, st_ub, st_lb = compute_supertrend(df, atr_period=int(atr_period), atr_mult=float(atr_mult))

    # --- Plot base price
    ax.plot(close.index, close.values, linewidth=2.0, label="Close")
    if show_bbands:
        ax.plot(bb_mid.index, bb_mid.values, linewidth=1.5, alpha=0.9, label=f"BB Mid ({'EMA' if bb_use_ema else 'SMA'})")
        ax.plot(bb_u.index, bb_u.values, linewidth=1.0, alpha=0.65, label="BB Upper")
        ax.plot(bb_l.index, bb_l.values, linewidth=1.0, alpha=0.65, label="BB Lower")

    # --- Regression + ±2σ
    if yhat is not None and len(yhat.dropna()):
        ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=2.2, alpha=0.85, label=f"Slope Fit ({fmt_slope(slope_val)}/bar)")
    if up2 is not None and lo2 is not None and len(up2.dropna()) and len(lo2.dropna()):
        ax.plot(up2.index, up2.values, linestyle="--", linewidth=1.2, alpha=0.55, label="+2σ")
        ax.plot(lo2.index, lo2.values, linestyle="--", linewidth=1.2, alpha=0.55, label="-2σ")

    # --- R² badge
    if np.isfinite(r2_val):
        ax.text(
            0.99, 0.98, f"R² {fmt_r2(r2_val, digits=1)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7),
            zorder=20
        )

    # --- Supertrend
    if st_line is not None and len(st_line.dropna()):
        ax.plot(st_line.index, st_line.values, linewidth=2.0, alpha=0.9, label="Supertrend")

    # --- Kijun
    if show_ichi and kijun is not None and len(kijun.dropna()):
        ax.plot(kijun.index, kijun.values, linewidth=1.8, alpha=0.9, label=f"Kijun({ichi_base})")

    # --- PSAR
    if show_psar and psar is not None and len(psar.dropna()):
        # plot dots
        ax.scatter(psar.index, psar.values, s=14, alpha=0.65, label="PSAR")

    # --- Support/Resistance
    if sup is not None and len(sup.dropna()):
        ax.plot(sup.index, sup.values, linewidth=1.2, alpha=0.6, label=f"Support({sr_lb})")
    if res is not None and len(res.dropna()):
        ax.plot(res.index, res.values, linewidth=1.2, alpha=0.6, label=f"Resistance({sr_lb})")

    # --- Optional fibs (hourly requested)
    fibs = {}
    if show_fibs_local:
        fibs = fibonacci_levels(close)
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=0.9, alpha=0.25)
                label_on_left(ax, v, f"Fib {k}: {fmt_price_val(v)}", fontsize=8)

    # --- Signals: BB move-away triangles (THIS REQUEST)
    bb_sig = find_bb_moveaway_signal(close, bb_u, bb_l, slope_val=slope_val, r2_val=r2_val, conf_r2=0.99)
    annotate_bb_moveaway(ax, bb_sig)

    # --- Optional: slope trigger arrow (existing helper)
    try:
        trig = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=15)
        annotate_slope_trigger(ax, trig)
    except Exception:
        trig = None

    style_axes(ax)
    ax.set_title(f"{symbol} — Price")
    ax.set_ylabel("Price")

    # Hourly compact x-axis: use bar positions instead of time gaps
    meta = {
        "slope": float(slope_val) if np.isfinite(slope_val) else np.nan,
        "r2": float(r2_val) if np.isfinite(r2_val) else np.nan,
        "bb_sig": bb_sig,
        "slope_trig": trig,
        "fibs": fibs
    }

    return meta

def plot_ntd_panel(ax, close: pd.Series, df_ohlc: pd.DataFrame, sr_lb: int):
    """
    NTD panel (trend direction) with optional NPX overlay.
    Also supports S/R channel highlighting.
    """
    c = _coerce_1d_series(close).astype(float)
    if c.empty:
        ax.text(0.5, 0.5, "No NTD", ha="center", va="center", transform=ax.transAxes)
        return

    ntd = compute_normalized_trend(c, window=int(ntd_window))
    ax.plot(ntd.index, ntd.values, linewidth=2.0, label="NTD")

    if shade_ntd:
        shade_ntd_regions(ax, ntd)

    ax.axhline(0.0, linewidth=1.2, alpha=0.5)
    ax.axhline(0.75, linewidth=1.0, alpha=0.35)
    ax.axhline(-0.75, linewidth=1.0, alpha=0.35)

    if show_npx_ntd:
        npx = compute_normalized_price(c, window=int(ntd_window))
        ax.plot(npx.index, npx.values, linewidth=1.3, alpha=0.85, label="NPX")

        if mark_npx_cross:
            # mark where NPX crosses NTD (simple sign of diff cross)
            diff = (npx - ntd)
            cross = (diff * diff.shift(1) < 0).fillna(False)
            if cross.any():
                t = cross[cross].index
                ax.scatter(t, ntd.reindex(t), s=18, alpha=0.6)

    # S/R channel highlighting on NTD (optional)
    if show_ntd_channel and (df_ohlc is not None) and (not df_ohlc.empty) and {"High","Low","Close"}.issubset(df_ohlc.columns):
        sup, res = compute_sr(df_ohlc, lookback=int(sr_lb))
        sup = sup.reindex(c.index).ffill()
        res = res.reindex(c.index).ffill()
        in_channel = (c >= sup) & (c <= res)
        if in_channel.any():
            ax.fill_between(c.index, -1.0, 1.0, where=in_channel.values, alpha=0.06)

    style_axes(ax)
    ax.set_title("NTD / NPX")
    ax.set_ylim(-1.05, 1.05)
# =========================
# Batch 11/12 — bullbear.py
# =========================
# ---------------------------
# Render views
# ---------------------------
def render_daily_view(symbol: str):
    df_daily = fetch_hist_ohlc(symbol)
    if df_daily is None or df_daily.empty:
        st.warning("No daily data returned.")
        return

    df_daily = subset_by_daily_view(df_daily, daily_view)

    close = df_daily["Close"]
    piv = current_daily_pivots(df_daily)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 1, 1)

    meta = plot_price_panel(
        ax=ax1,
        df_ohlc=df_daily,
        symbol=symbol,
        slope_lb=int(slope_lb_daily),
        sr_lb=int(sr_lb_daily),
        show_fibs_local=False,
        show_sessions=False,
        show_news=False,
        hourly_compact_x=False
    )

    # Pivot lines
    if piv:
        for k, v in piv.items():
            if np.isfinite(v):
                ax1.axhline(v, linewidth=0.9, alpha=0.25)
                label_on_left(ax1, v, f"{k}: {fmt_price_val(v)}", fontsize=8)

    # Instruction line (uses global slope as regression slope on same panel for now)
    instr = format_trade_instruction(
        trend_slope=meta.get("slope", np.nan),
        buy_val=_safe_last_float(df_daily["Close"]),
        sell_val=_safe_last_float(df_daily["Close"]),
        close_val=_safe_last_float(df_daily["Close"]),
        symbol=symbol,
        global_trend_slope=meta.get("slope", np.nan)
    )
    st.markdown(f"**Trade instruction:** {instr}")

    ax2 = fig.add_subplot(2, 1, 2)
    if show_ntd:
        plot_ntd_panel(ax2, close=close, df_ohlc=df_daily, sr_lb=int(sr_lb_daily))
        ax2.legend(loc="upper left", fontsize=8)
    else:
        ax2.axis("off")

    ax1.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

def render_hourly_view(symbol: str):
    df_h = fetch_hourly_ohlc(symbol)
    if df_h is None or df_h.empty:
        st.warning("No hourly data returned (try another symbol or timeframe limitations).")
        return

    close = df_h["Close"]

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 1, 1)

    meta = plot_price_panel(
        ax=ax1,
        df_ohlc=df_h,
        symbol=symbol,
        slope_lb=int(slope_lb_hourly),
        sr_lb=int(sr_lb_hourly),
        show_fibs_local=bool(show_fibs),
        show_sessions=bool(show_sessions_pst) if mode == "Forex" else False,
        show_news=bool(show_fx_news) if mode == "Forex" else False,
        hourly_compact_x=False
    )

    # Optional momentum & MACD quick lines as text
    if show_mom_hourly:
        roc = compute_roc(close, n=int(mom_lb_hourly))
        ax1.text(
            0.99, 0.05, f"ROC({mom_lb_hourly}) {roc.dropna().iloc[-1]:.2f}%",
            transform=ax1.transAxes, ha="right", va="bottom",
            fontsize=9, alpha=0.75,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.6),
        )

    # NTD panel
    ax2 = fig.add_subplot(2, 1, 2)
    if show_ntd:
        plot_ntd_panel(ax2, close=close, df_ohlc=df_h, sr_lb=int(sr_lb_hourly))
        ax2.legend(loc="upper left", fontsize=8)
    else:
        ax2.axis("off")

    ax1.legend(loc="upper left", fontsize=8)
    st.pyplot(fig)

    # Optional MACD chart
    if show_macd:
        macd, sig, hist = compute_macd(close)
        fig2 = plt.figure(figsize=(14, 3.5))
        axm = fig2.add_subplot(1, 1, 1)
        axm.plot(macd.index, macd.values, linewidth=1.6, label="MACD")
        axm.plot(sig.index, sig.values, linewidth=1.3, label="Signal")
        axm.axhline(0.0, linewidth=1.0, alpha=0.5)
        style_axes(axm)
        axm.set_title("MACD")
        axm.legend(loc="upper left", fontsize=8)
        st.pyplot(fig2)

def render_scanner_tab(symbols):
    st.subheader("R² Trend Scanner (threshold: 45%)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Daily — Uptrend (R² ≥ 45%)")
        up_d, dn_d = scan_r2_trends_daily(symbols, lookback=int(slope_lb_daily), r2_min=0.45)
        if up_d:
            df = pd.DataFrame(up_d, columns=["Symbol", "R²", "Slope"])
            df["R²"] = df["R²"].astype(float)
            df["Slope"] = df["Slope"].astype(float)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No symbols found.")

    with c2:
        st.markdown("### Daily — Downtrend (R² ≥ 45%)")
        up_d, dn_d = scan_r2_trends_daily(symbols, lookback=int(slope_lb_daily), r2_min=0.45)
        if dn_d:
            df = pd.DataFrame(dn_d, columns=["Symbol", "R²", "Slope"])
            df["R²"] = df["R²"].astype(float)
            df["Slope"] = df["Slope"].astype(float)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No symbols found.")

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Hourly — Uptrend (R² ≥ 45%)")
        up_h, dn_h = scan_r2_trends_hourly(symbols, lookback=int(slope_lb_hourly), r2_min=0.45)
        if up_h:
            df = pd.DataFrame(up_h, columns=["Symbol", "R²", "Slope"])
            df["R²"] = df["R²"].astype(float)
            df["Slope"] = df["Slope"].astype(float)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No symbols found.")

    with c4:
        st.markdown("### Hourly — Downtrend (R² ≥ 45%)")
        up_h, dn_h = scan_r2_trends_hourly(symbols, lookback=int(slope_lb_hourly), r2_min=0.45)
        if dn_h:
            df = pd.DataFrame(dn_h, columns=["Symbol", "R²", "Slope"])
            df["R²"] = df["R²"].astype(float)
            df["Slope"] = df["Slope"].astype(float)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No symbols found.")
# =========================
# Batch 12/12 — bullbear.py
# =========================
# ---------------------------
# Main UI (All Tabs)
# ---------------------------
st.sidebar.subheader("Run")
default_sym = universe[0] if universe else None

# Protect selectbox against old invalid session value
if ("ticker" in st.session_state) and (st.session_state.ticker not in universe):
    st.session_state.ticker = None

ticker = st.sidebar.selectbox(
    "Select symbol:",
    universe,
    index=0 if default_sym in universe else 0,
    key="ticker_select"
)

run = st.sidebar.button("▶ Run / Refresh Charts", use_container_width=True, key="btn_run")
if run:
    st.session_state.run_all = True
    st.session_state.mode_at_run = mode

# If never run yet, keep UI visible but don't render heavy charts
tabs = st.tabs([
    "📅 Daily",
    "⏱ Hourly",
    "📉 Forecast",
    "🧭 Scanner (R²)",
    "ℹ️ About / Settings"
])

with tabs[0]:
    if st.session_state.get("run_all", False):
        render_daily_view(ticker)
    else:
        st.info("Click **Run / Refresh Charts** in the sidebar to load Daily view.")

with tabs[1]:
    if st.session_state.get("run_all", False):
        render_hourly_view(ticker)
    else:
        st.info("Click **Run / Refresh Charts** in the sidebar to load Hourly view.")

with tabs[2]:
    st.subheader("SARIMAX Forecast (Daily Close)")
    if st.session_state.get("run_all", False):
        try:
            s = fetch_hist(ticker)
            idx, mean, ci = compute_sarimax_forecast(s)
            fig = plt.figure(figsize=(14, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(s.index, s.values, linewidth=1.8, label="History")
            ax.plot(idx, mean.values, linewidth=2.2, label="Forecast")
            # CI
            try:
                lo = ci.iloc[:, 0].values
                hi = ci.iloc[:, 1].values
                ax.fill_between(idx, lo, hi, alpha=0.18)
            except Exception:
                pass
            style_axes(ax)
            ax.set_title(f"{ticker} — 30D Forecast")
            ax.legend(loc="upper left", fontsize=8)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Forecast error: {e}")
    else:
        st.info("Click **Run / Refresh Charts** to compute the forecast.")

with tabs[3]:
    if st.session_state.get("run_all", False):
        render_scanner_tab(universe)
    else:
        st.info("Click **Run / Refresh Charts** to run the scanner.")

with tabs[4]:
    st.markdown("""
### What’s included (in this version)
- **Daily + Hourly charts** with:
  - **Bollinger Bands**
  - **Regression slope trendline + ±2σ band**
  - **R² badge**
  - **Supertrend (default ON)**
  - **Ichimoku Kijun (optional)**
  - **Parabolic SAR (optional)**
  - **NTD panel + optional NPX overlay**
- **Signals**
  - **BB Buy / BB Sell triangles**:
    - Requires **R² ≥ 0.99** and slope sign to match (up for buy, down for sell)
    - Requires **prior bar touch/breach** of BB band and **current bar moving away**
- **Scanner tab**
  - Shows symbols with **R² ≥ 45%** for **uptrend** and **downtrend** on both **daily** and **hourly**
- **SARIMAX Forecast tab**

If you want the BB signal to use *your regression ±2σ band* instead of Bollinger Bands, tell me and I’ll swap it cleanly.
""")
