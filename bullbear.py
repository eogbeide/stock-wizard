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

# NEW (THIS REQUEST): user note
st.sidebar.info('📝 Note: Place Buy Trade Closer to 0% Fibonnaci and Sell trade closer to 100% Fibonacci.')

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

# NEW (THIS REQUEST): Supertrend toggle (ON by default)
show_supertrend = st.sidebar.checkbox("Show Supertrend line", value=True, key="sb_show_supertrend")

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
# NEW (THIS REQUEST): CONFIRMATION TRIGGER (Fib 0%/100% + reversal + R²≥0.999)
# ---------------------------
def find_fib_confirmation_trigger(price: pd.Series,
                                 fibs: dict,
                                 r2_val: float,
                                 prox: float = 0.0025,
                                 bars_confirm: int = 2,
                                 horizon: int = 15,
                                 min_r2: float = 0.999):
    """
    Confirmation Trigger rules:
      • Requires regression R² ≥ min_r2 (99.9% confidence)
      • If price touched/breached Fib 100% (low) within last `horizon` bars AND
        has `bars_confirm` consecutive UP diffs -> CONFIRM BUY
      • If price touched/breached Fib 0% (high) within last `horizon` bars AND
        has `bars_confirm` consecutive DOWN diffs -> CONFIRM SELL
    Returns most recent trigger dict or None.
    """
    try:
        r2 = float(r2_val)
    except Exception:
        r2 = np.nan
    if not (np.isfinite(r2) and r2 >= float(min_r2)):
        return None

    p = _coerce_1d_series(price).dropna()
    if p.empty or len(p) < (max(3, int(bars_confirm) + 2)):
        return None

    hi0 = fibs.get("0%") if isinstance(fibs, dict) else None
    lo100 = fibs.get("100%") if isinstance(fibs, dict) else None
    try:
        hi0 = float(hi0)
        lo100 = float(lo100)
    except Exception:
        return None
    if not (np.isfinite(hi0) and np.isfinite(lo100)):
        return None

    hz = max(1, int(horizon))
    n = max(1, int(bars_confirm))

    d = p.diff()
    up_confirm = (d > 0).rolling(n, min_periods=n).apply(lambda x: 1.0 if np.all(x) else 0.0, raw=True) > 0
    dn_confirm = (d < 0).rolling(n, min_periods=n).apply(lambda x: 1.0 if np.all(x) else 0.0, raw=True) > 0
    up_confirm = up_confirm.fillna(False)
    dn_confirm = dn_confirm.fillna(False)

    touch_low = (p <= lo100 * (1.0 + float(prox))).fillna(False)
    touch_high = (p >= hi0 * (1.0 - float(prox))).fillna(False)

    touch_low_recent = touch_low.rolling(hz + 1, min_periods=1).max() > 0
    touch_high_recent = touch_high.rolling(hz + 1, min_periods=1).max() > 0

    buy_mask = up_confirm & touch_low_recent
    sell_mask = dn_confirm & touch_high_recent

    last_buy = buy_mask[buy_mask].index[-1] if buy_mask.any() else None
    last_sell = sell_mask[sell_mask].index[-1] if sell_mask.any() else None

    if last_buy is None and last_sell is None:
        return None

    if last_sell is None:
        t_conf = last_buy
        side = "BUY"
        touch_series = touch_low
        level_lbl = "100%"
    elif last_buy is None:
        t_conf = last_sell
        side = "SELL"
        touch_series = touch_high
        level_lbl = "0%"
    else:
        t_conf = last_buy if last_buy >= last_sell else last_sell
        if t_conf == last_buy:
            side = "BUY"
            touch_series = touch_low
            level_lbl = "100%"
        else:
            side = "SELL"
            touch_series = touch_high
            level_lbl = "0%"

    try:
        loc = int(p.index.get_loc(t_conf))
    except Exception:
        return None
    j0 = max(0, loc - hz)
    win = touch_series.iloc[j0:loc+1]
    if not win.any():
        return None
    t_touch = win[win].index[-1]

    return {
        "side": side,
        "touch_time": t_touch,
        "touch_price": float(p.loc[t_touch]) if np.isfinite(p.loc[t_touch]) else np.nan,
        "confirm_time": t_conf,
        "confirm_price": float(p.loc[t_conf]) if np.isfinite(p.loc[t_conf]) else np.nan,
        "fib_level": level_lbl,
        "r2": float(r2),
    }

def annotate_fib_confirmation_trigger(ax, trig: dict):
    if trig is None:
        return
    side = trig.get("side", "")
    t = trig.get("confirm_time")
    px = trig.get("confirm_price")
    lvl = trig.get("fib_level", "")
    if t is None or not np.isfinite(px):
        return

    col = "tab:green" if side == "BUY" else "tab:red"
    lbl = f"CONFIRM {side} (Fib {lvl}, R²≥99.9%)"
    ax.scatter([t], [px], marker="o", s=140, color=col, zorder=12, label=lbl)
    ax.text(
        t, px,
        f"  {lbl}",
        color=col,
        fontsize=9,
        fontweight="bold",
        va="bottom" if side == "BUY" else "top",
        zorder=13
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
# Ichimoku, Supertrend, PSAR
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
    atr = _compute_atr_from_ohlc(ohlc, period=atr_period)
    hl2 = (ohlc["High"] + ohlc["Low"]) / 2.0
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr

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
    psar.iloc[0] = float(low.iloc[0])
    ep = float(high.iloc[0])
    af = step

    for i in range(1, len(idx)):
        prev_psar = psar.iloc[i-1]
        if in_uptrend.iloc[i-1]:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = min(psar.iloc[i],
                               float(low.iloc[i-1]),
                               float(low.iloc[i-2]) if i >= 2 else float(low.iloc[i-1]))
            if high.iloc[i] > ep:
                ep = float(high.iloc[i])
                af = min(af + step, max_step)
            if low.iloc[i] < psar.iloc[i]:
                in_uptrend.iloc[i] = False
                psar.iloc[i] = ep
                ep = float(low.iloc[i])
                af = step
            else:
                in_uptrend.iloc[i] = True
        else:
            psar.iloc[i] = prev_psar + af * (ep - prev_psar)
            psar.iloc[i] = max(psar.iloc[i],
                               float(high.iloc[i-1]),
                               float(high.iloc[i-2]) if i >= 2 else float(high.iloc[i-1]))
            if low.iloc[i] < ep:
                ep = float(low.iloc[i])
                af = min(af + step, max_step)
            if high.iloc[i] > psar.iloc[i]:
                in_uptrend.iloc[i] = True
                psar.iloc[i] = ep
                ep = float(high.iloc[i])
                af = step
            else:
                in_uptrend.iloc[i] = False

    return pd.DataFrame({"PSAR": psar, "in_uptrend": in_uptrend})

def detect_hma_reversal_masks(price: pd.Series, hma: pd.Series, lookback: int = 3):
    h = _coerce_1d_series(hma)
    slope = h.diff().rolling(lookback, min_periods=1).mean()
    sign_now = np.sign(slope)
    sign_prev = np.sign(slope.shift(1))
    cross_up, cross_dn = _cross_series(price, hma)
    buy_rev  = cross_up & (sign_now > 0) & (sign_prev < 0)
    sell_rev = cross_dn & (sign_now < 0) & (sign_prev > 0)
    return buy_rev.fillna(False), sell_rev.fillna(False)

def overlay_hma_reversal_on_ntd(ax, price: pd.Series, hma: pd.Series,
                               lookback: int = 3, y_up: float = 0.95, y_dn: float = -0.95,
                               period: int = 55, ntd: pd.Series = None):
    buy_rev, sell_rev = detect_hma_reversal_masks(price, hma, lookback=lookback)
    idx_up = list(buy_rev[buy_rev].index)
    idx_dn = list(sell_rev[sell_rev].index)
    if len(idx_up):
        ax.scatter(idx_up, [y_up]*len(idx_up), marker="s", s=70, color="tab:green", zorder=8, label=f"HMA({period}) REV")
    if len(idx_dn):
        ax.scatter(idx_dn, [y_dn]*len(idx_dn), marker="D", s=70, color="tab:red", zorder=8, label=f"HMA({period}) REV")

def overlay_npx_on_ntd(ax, npx: pd.Series, ntd: pd.Series, mark_crosses: bool = True):
    npx = _coerce_1d_series(npx)
    ntd = _coerce_1d_series(ntd)
    idx = ntd.index.union(npx.index)
    npx = npx.reindex(idx)
    ntd = ntd.reindex(idx)
    if npx.dropna().empty:
        return
    ax.plot(npx.index, npx.values, "-", linewidth=1.2, color="tab:gray", alpha=0.9, label="NPX (Norm Price)")
    if mark_crosses and not ntd.dropna().empty:
        up_mask, dn_mask = _cross_series(npx, ntd)
        up_idx = list(up_mask[up_mask].index)
        dn_idx = list(dn_mask[dn_mask].index)
        if len(up_idx):
            ax.scatter(up_idx, ntd.loc[up_idx], marker="o", s=40, color="tab:green", zorder=9, label="Price↑NTD")
        if len(dn_idx):
            ax.scatter(dn_idx, ntd.loc[dn_idx], marker="x", s=60, color="tab:red", zorder=9, label="Price↓NTD")

def overlay_ntd_triangles_by_trend(ax, ntd: pd.Series, trend_slope: float, upper: float = 0.75, lower: float = -0.75):
    s = _coerce_1d_series(ntd).dropna()
    if s.empty or not np.isfinite(trend_slope):
        return
    uptrend = trend_slope > 0
    downtrend = trend_slope < 0

    cross_up0 = (s >= 0.0) & (s.shift(1) < 0.0)
    cross_dn0 = (s <= 0.0) & (s.shift(1) > 0.0)
    idx_up0 = list(cross_up0[cross_up0].index)
    idx_dn0 = list(cross_dn0[cross_dn0].index)

    cross_out_hi = (s >= upper) & (s.shift(1) < upper)
    cross_out_lo = (s <= lower) & (s.shift(1) > lower)
    idx_hi = list(cross_out_hi[cross_out_hi].index)
    idx_lo = list(cross_out_lo[cross_out_lo].index)

    if uptrend:
        if idx_up0:
            ax.scatter(idx_up0, [0.0]*len(idx_up0), marker="^", s=95, color="tab:green", zorder=10, label="NTD 0↑")
        if idx_lo:
            ax.scatter(idx_lo, s.loc[idx_lo], marker="^", s=85, color="tab:green", zorder=10, label="NTD < -0.75")
    if downtrend:
        if idx_dn0:
            ax.scatter(idx_dn0, [0.0]*len(idx_dn0), marker="v", s=95, color="tab:red", zorder=10, label="NTD 0↓")
        if idx_hi:
            ax.scatter(idx_hi, s.loc[idx_hi], marker="v", s=85, color="tab:red", zorder=10, label="NTD > +0.75")

def _n_consecutive_increasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas > 0))

def _n_consecutive_decreasing(series: pd.Series, n: int = 2) -> bool:
    s = _coerce_1d_series(series).dropna()
    if len(s) < n+1:
        return False
    deltas = np.diff(s.iloc[-(n+1):])
    return bool(np.all(deltas < 0))

def overlay_ntd_sr_reversal_stars(ax,
                                 price: pd.Series,
                                 sup: pd.Series,
                                 res: pd.Series,
                                 trend_slope: float,
                                 ntd: pd.Series,
                                 prox: float = 0.0025,
                                 bars_confirm: int = 2):
    p = _coerce_1d_series(price).dropna()
    if p.empty:
        return
    s_sup = _coerce_1d_series(sup).reindex(p.index).ffill().bfill()
    s_res = _coerce_1d_series(res).reindex(p.index).ffill().bfill()
    s_ntd = _coerce_1d_series(ntd).reindex(p.index)

    t = p.index[-1]
    if not (t in s_sup.index and t in s_res.index and t in s_ntd.index):
        return
    c0 = float(p.iloc[-1])
    c1 = float(p.iloc[-2]) if len(p) >= 2 else np.nan
    S0 = float(s_sup.loc[t]) if pd.notna(s_sup.loc[t]) else np.nan
    R0 = float(s_res.loc[t]) if pd.notna(s_res.loc[t]) else np.nan
    ntd0 = float(s_ntd.loc[t]) if pd.notna(s_ntd.loc[t]) else np.nan
    if not np.all(np.isfinite([c0, S0, R0, ntd0])):
        return

    near_support = c0 <= S0 * (1.0 + prox)
    near_resist  = c0 >= R0 * (1.0 - prox)

    toward_res = toward_sup = False
    if np.isfinite(c1):
        toward_res = (R0 - c0) < (R0 - c1)
        toward_sup = (c0 - S0) < (c1 - S0)

    buy_cond  = (trend_slope > 0) and near_support and _n_consecutive_increasing(p, bars_confirm) and toward_res
    sell_cond = (trend_slope < 0) and near_resist  and _n_consecutive_decreasing(p, bars_confirm) and toward_sup

    if buy_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:green", zorder=12, label="BUY ★ (Support reversal)")
    if sell_cond:
        ax.scatter([t], [ntd0], marker="*", s=170, color="tab:red", zorder=12, label="SELL ★ (Resistance reversal)")
# ---------------------------
# Support / Resistance helpers
# ---------------------------
def rolling_support_resistance(close: pd.Series, lookback: int = 60):
    s = _coerce_1d_series(close).astype(float)
    if s.empty or lookback < 2:
        empty = pd.Series(index=s.index, dtype=float)
        return empty, empty
    minp = max(2, lookback // 3)
    sup = s.rolling(int(lookback), min_periods=minp).min()
    res = s.rolling(int(lookback), min_periods=minp).max()
    return sup.reindex(s.index), res.reindex(s.index)

def _safe_axvline(ax, x, **kwargs):
    try:
        ax.axvline(x=x, **kwargs)
    except Exception:
        pass

# ---------------------------
# FX Session markers (PST)
# ---------------------------
def add_fx_sessions_pst(ax, idx: pd.DatetimeIndex):
    """
    Draw approximate London + NY session start times in PST.
    Note: DST shifts make this approximate; this is a visual guide only.
    """
    if not isinstance(idx, pd.DatetimeIndex) or idx.empty:
        return
    try:
        tz = PACIFIC
        local = idx.tz_convert(tz) if idx.tz is not None else idx.tz_localize(tz)
    except Exception:
        local = idx

    # approximate PST starts (will be off during DST; kept simple by design)
    # London open ~ 00:00 PST, NY open ~ 06:30 PST
    dates = pd.to_datetime(local.date).unique()
    for d in dates:
        try:
            d = pd.Timestamp(d).tz_localize(PACIFIC)
        except Exception:
            try:
                d = pd.Timestamp(d)
            except Exception:
                continue
        london = d + pd.Timedelta(hours=0)
        ny = d + pd.Timedelta(hours=6, minutes=30)
        _safe_axvline(ax, london, linestyle=":", linewidth=1.0, alpha=0.22)
        _safe_axvline(ax, ny, linestyle=":", linewidth=1.0, alpha=0.22)

# ---------------------------
# Leaderline annotation outside chart area (right side)
# ---------------------------
def annotate_outside_right(ax, x, y, text: str, color: str = "black", y_frac: float = 0.85):
    """
    Draw label outside to the right with an arrow leader line to (x,y).
    y_frac controls vertical placement of the outside label in axes fraction.
    """
    try:
        ax.annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=(1.02, y_frac),
            textcoords="axes fraction",
            ha="left",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, alpha=0.85),
            zorder=20,
            clip_on=False
        )
    except Exception:
        pass

# ---------------------------
# Supertrend plotting helper
# ---------------------------
def plot_supertrend(ax, st_df: pd.DataFrame, label: str = "Supertrend"):
    if st_df is None or st_df.empty or "ST" not in st_df.columns:
        return
    st_line = _coerce_1d_series(st_df["ST"])
    in_up = st_df["in_uptrend"] if "in_uptrend" in st_df.columns else pd.Series(True, index=st_line.index)
    st_line = st_line.dropna()
    if st_line.empty:
        return

    # plot as two segments for visual clarity
    up_mask = in_up.reindex(st_line.index).fillna(True)
    dn_mask = ~up_mask

    ax.plot(st_line.index[up_mask], st_line.loc[up_mask], linewidth=2.0, alpha=0.9,
            label=f"{label} (UP)")
    ax.plot(st_line.index[dn_mask], st_line.loc[dn_mask], linewidth=2.0, alpha=0.9,
            label=f"{label} (DOWN)")

# ---------------------------
# BB Cross scanner helpers (kept for backward compatibility)
# ---------------------------
def find_bb_mid_cross_signal(close: pd.Series, pctb: pd.Series, mid: pd.Series,
                            global_trend_slope: float, local_slope: float):
    """
    Prior BB Mid Cross logic (kept).
    BUY: price reverses from 100% (upper band) then crosses BB mid upward
    SELL: price reverses from 0% (lower band) then crosses BB mid downward
    AND global + local slopes must align (UP for BUY, DOWN for SELL).
    """
    c = _coerce_1d_series(close).astype(float)
    p = _coerce_1d_series(pctb).reindex(c.index)
    m = _coerce_1d_series(mid).reindex(c.index)

    ok = c.notna() & p.notna() & m.notna()
    if ok.sum() < 3:
        return None
    c = c[ok]; p = p[ok]; m = m[ok]

    above_mid = c > m
    cross_up = above_mid & (~above_mid.shift(1).fillna(False))
    cross_dn = (~above_mid) & (above_mid.shift(1).fillna(False))

    hit_top = (p >= 0.999).rolling(10, min_periods=1).max() > 0
    hit_bot = (p <= 0.001).rolling(10, min_periods=1).max() > 0

    g = float(global_trend_slope) if np.isfinite(global_trend_slope) else np.nan
    l = float(local_slope) if np.isfinite(local_slope) else np.nan
    if not (np.isfinite(g) and np.isfinite(l)):
        return None

    buy_mask = (g > 0) & (l > 0) & hit_bot & cross_up
    sell_mask = (g < 0) & (l < 0) & hit_top & cross_dn

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

    return {"time": t, "price": float(c.loc[t]), "side": side, "note": "BB Mid Cross"}
# ---------------------------
# Chart builders
# ---------------------------
def build_daily_chart(ticker: str, df_ohlc: pd.DataFrame):
    """
    Daily chart:
      • price + regression + ±2σ band
      • BBands
      • Ichimoku Kijun
      • PSAR
      • Supertrend (NEW: plotted on Daily when toggle ON)
      • NTD panel optionally
      • Confirmation trigger (Fib 0/100 + reversal + R²≥0.999) - uses daily fibs ONLY if show_fibs is enabled
        (show_fibs label says hourly-only, but confirmation trigger is based on fib endpoints; we keep fib lines hidden here
         unless you later choose to show them on daily as well.)
    """
    if df_ohlc is None or df_ohlc.empty:
        st.warning("No daily data returned.")
        return None

    dfx = df_ohlc.copy().sort_index()
    dfx = subset_by_daily_view(dfx, daily_view)

    close = _coerce_1d_series(dfx["Close"]).dropna()
    if close.empty:
        st.warning("Daily close series empty.")
        return None

    # regression + band (local)
    yhat, up2, lo2, m_local, r2_local = regression_with_band(close, lookback=slope_lb_daily, z=2.0)

    # global trend (longer)
    m_global = draw_trend_direction_line(plt.gca(), close, label_prefix="")  # placeholder; we re-draw below properly
    # we don't want to pollute a dummy axes; compute directly:
    try:
        g_line, g_m = slope_line(close, lookback=len(close))
        m_global = float(g_m)
    except Exception:
        m_global = float("nan")

    # BBands
    bb_mid, bb_up, bb_lo, pctb, nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # SR
    sup, res = rolling_support_resistance(close, lookback=sr_lb_daily)

    # Ichimoku Kijun
    if show_ichi and {"High", "Low"}.issubset(dfx.columns):
        tenkan, kijun, spA, spB, chikou = ichimoku_lines(dfx["High"], dfx["Low"], close,
                                                         conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
    else:
        kijun = pd.Series(index=close.index, dtype=float)

    # PSAR
    psar_df = compute_psar_from_ohlc(dfx, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()

    # Supertrend (NEW on daily)
    st_df = compute_supertrend(dfx, atr_period=atr_period, atr_mult=atr_mult) if show_supertrend else pd.DataFrame()

    # NTD
    ntd = compute_normalized_trend(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)

    # Confirmation trigger (uses fib endpoints)
    fibs = fibonacci_levels(close)  # daily fib endpoints
    conf_tr = find_fib_confirmation_trigger(
        price=close,
        fibs=fibs,
        r2_val=r2_local,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm,
        horizon=rev_horizon,
        min_r2=0.999
    )

    # Slope trigger after band reversal (daily)
    slope_tr = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=rev_horizon)

    # MACD / HMA / SR signal
    hma = compute_hma(close, period=hma_period) if show_hma else pd.Series(index=close.index, dtype=float)
    macd, macd_sig, macd_hist = compute_macd(close)
    macd_sig_tr = find_macd_hma_sr_signal(close, hma, macd, sup, res, global_trend_slope=m_global, prox=sr_prox_pct)

    # -------------- Figure layout --------------
    nrows = 2 if (show_ntd and show_nrsi) else 1
    fig_h = 7.6 if nrows == 1 else 10.6
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, fig_h), sharex=True)
    ax = axes if nrows == 1 else axes[0]

    # Price plot
    ax.plot(close.index, close.values, linewidth=1.6, label="Close")
    style_axes(ax)

    # Regression band
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=2.2,
                label=f"Local Trend ({fmt_slope(m_local)}/bar, R²={fmt_r2(r2_local, 1)})")
    if not up2.empty:
        ax.plot(up2.index, up2.values, linewidth=1.4, alpha=0.8, label="+2σ band")
    if not lo2.empty:
        ax.plot(lo2.index, lo2.values, linewidth=1.4, alpha=0.8, label="-2σ band")

    # Global trendline
    try:
        x = np.arange(len(close), dtype=float)
        m, b = np.polyfit(x, close.values, 1)
        g = m*x + b
        ax.plot(close.index, g, linestyle="--", linewidth=2.8, alpha=0.85,
                label=f"Global Trend ({fmt_slope(m)}/bar)")
        m_global = float(m)
    except Exception:
        pass

    # Bollinger Bands
    if show_bbands and bb_mid.notna().any():
        ax.plot(bb_mid.index, bb_mid.values, linewidth=1.2, alpha=0.9, label="BB Mid")
        ax.plot(bb_up.index, bb_up.values, linewidth=1.0, alpha=0.7, label="BB Upper")
        ax.plot(bb_lo.index, bb_lo.values, linewidth=1.0, alpha=0.7, label="BB Lower")

    # Ichimoku Kijun
    if show_ichi and kijun.notna().any():
        ax.plot(kijun.index, kijun.values, linewidth=1.6, alpha=0.85, label=f"Kijun({ichi_base})")

    # PSAR
    if show_psar and not psar_df.empty and "PSAR" in psar_df.columns:
        ps = _coerce_1d_series(psar_df["PSAR"]).reindex(close.index)
        ax.scatter(ps.index, ps.values, s=9, alpha=0.7, label="PSAR")

    # Supertrend
    if show_supertrend and not st_df.empty:
        plot_supertrend(ax, st_df.reindex(close.index), label="Supertrend")

    # Support/Resistance lines
    if sup.notna().any():
        ax.plot(sup.index, sup.values, linewidth=1.0, alpha=0.5, label="Support (roll min)")
    if res.notna().any():
        ax.plot(res.index, res.values, linewidth=1.0, alpha=0.5, label="Resistance (roll max)")

    # Annotate triggers (outside right with leaderline)
    if conf_tr is not None:
        side = conf_tr.get("side", "")
        t = conf_tr.get("confirm_time")
        px = conf_tr.get("confirm_price")
        lvl = conf_tr.get("fib_level", "")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"CONFIRM {side}\nFib {lvl}\nR²≥99.9%", color=col, y_frac=0.88)

    if slope_tr is not None:
        side = slope_tr.get("side", "")
        t = slope_tr.get("cross_time")
        px = slope_tr.get("cross_price")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"Slope {side}\n(after ±2σ)", color=col, y_frac=0.72)

    if macd_sig_tr is not None:
        side = macd_sig_tr.get("side", "")
        t = macd_sig_tr.get("time")
        px = macd_sig_tr.get("price")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"{side}\nMACD/HMA55\n+ S/R", color=col, y_frac=0.56)

    # Title & legend
    ax.set_title(f"{ticker} — Daily", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    # ---- NTD panel (optional) ----
    if nrows == 2:
        ax2 = axes[1]
        s_ntd = _coerce_1d_series(ntd).dropna()
        if not s_ntd.empty:
            ax2.plot(s_ntd.index, s_ntd.values, linewidth=1.4, label="NTD")
            ax2.axhline(0, linewidth=1.0, alpha=0.5)
            ax2.axhline(0.75, linewidth=1.0, alpha=0.25)
            ax2.axhline(-0.75, linewidth=1.0, alpha=0.25)
            if shade_ntd:
                shade_ntd_regions(ax2, s_ntd)
            if show_npx_ntd:
                overlay_npx_on_ntd(ax2, npx, s_ntd, mark_crosses=mark_npx_cross)
            overlay_ntd_triangles_by_trend(ax2, s_ntd, m_global)
            overlay_ntd_sr_reversal_stars(ax2, close, sup, res, m_global, s_ntd, prox=sr_prox_pct, bars_confirm=rev_bars_confirm)

            if show_hma_rev_ntd and show_hma and hma.notna().any():
                overlay_hma_reversal_on_ntd(ax2, close, hma, lookback=hma_rev_lb, period=hma_period, ntd=s_ntd)

            ax2.set_title("NTD Panel", fontsize=11, fontweight="bold")
            style_axes(ax2)
            ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    return fig


def build_hourly_chart(ticker: str, df_intraday_5m: pd.DataFrame):
    """
    Hourly chart:
      • uses 5m intraday downloaded data, resampled to 1H OHLC
      • plots regression + ±2σ band on close
      • shows Fibonacci lines (hourly) when toggle ON
      • Supertrend (toggle ON by default)
      • Confirmation trigger (Fib 0/100 + reversal + R²≥0.999)
    """
    if df_intraday_5m is None or df_intraday_5m.empty:
        st.warning("No intraday data returned.")
        return None

    # Resample 5m to 1H OHLC
    df = df_intraday_5m.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        st.warning("Intraday index is not DatetimeIndex.")
        return None

    o = df["Open"].resample("1H").first()
    h = df["High"].resample("1H").max()
    l = df["Low"].resample("1H").min()
    c = df["Close"].resample("1H").last()
    v = df["Volume"].resample("1H").sum() if "Volume" in df.columns else None

    dfx = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
    close = _coerce_1d_series(dfx["Close"]).dropna()
    if close.empty:
        st.warning("Hourly close series empty.")
        return None

    # Local regression + band
    yhat, up2, lo2, m_local, r2_local = regression_with_band(close, lookback=slope_lb_hourly, z=2.0)

    # Global slope (on entire hourly series)
    try:
        g_line, g_m = slope_line(close, lookback=len(close))
        m_global = float(g_m)
    except Exception:
        m_global = float("nan")

    # Fibonacci (hourly)
    fibs = fibonacci_levels(close)

    # BBands (hourly)
    bb_mid, bb_up, bb_lo, pctb, nbb = compute_bbands(close, window=bb_win, mult=bb_mult, use_ema=bb_use_ema)

    # SR (hourly)
    sup, res = rolling_support_resistance(close, lookback=sr_lb_hourly)

    # Ichimoku Kijun (hourly)
    if show_ichi:
        tenkan, kijun, spA, spB, chikou = ichimoku_lines(dfx["High"], dfx["Low"], close,
                                                         conv=ichi_conv, base=ichi_base, span_b=ichi_spanb)
    else:
        kijun = pd.Series(index=close.index, dtype=float)

    # PSAR (hourly)
    psar_df = compute_psar_from_ohlc(dfx, step=psar_step, max_step=psar_max) if show_psar else pd.DataFrame()

    # Supertrend (hourly)
    st_df = compute_supertrend(dfx, atr_period=atr_period, atr_mult=atr_mult) if show_supertrend else pd.DataFrame()

    # NTD (hourly)
    ntd = compute_normalized_trend(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)
    npx = compute_normalized_price(close, window=ntd_window) if show_ntd else pd.Series(index=close.index, dtype=float)

    # Confirmation trigger (THIS REQUEST)
    conf_tr = find_fib_confirmation_trigger(
        price=close,
        fibs=fibs,
        r2_val=r2_local,
        prox=sr_prox_pct,
        bars_confirm=rev_bars_confirm,
        horizon=rev_horizon,
        min_r2=0.999
    )

    # Slope trigger after band reversal (hourly)
    slope_tr = find_slope_trigger_after_band_reversal(close, yhat, up2, lo2, horizon=rev_horizon)

    # MACD / HMA / SR signal (hourly)
    hma = compute_hma(close, period=hma_period) if show_hma else pd.Series(index=close.index, dtype=float)
    macd, macd_sig, macd_hist = compute_macd(close)
    macd_sig_tr = find_macd_hma_sr_signal(close, hma, macd, sup, res, global_trend_slope=m_global, prox=sr_prox_pct)

    # -------------- Figure layout --------------
    nrows = 2 if (show_ntd and show_nrsi) else 1
    fig_h = 7.6 if nrows == 1 else 10.6
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, fig_h), sharex=True)
    ax = axes if nrows == 1 else axes[0]

    # Price
    ax.plot(close.index, close.values, linewidth=1.6, label="Close")
    style_axes(ax)

    # Regression line + bands
    if not yhat.empty:
        ax.plot(yhat.index, yhat.values, linestyle="--", linewidth=2.2,
                label=f"Local Trend ({fmt_slope(m_local)}/bar, R²={fmt_r2(r2_local, 1)})")
    if not up2.empty:
        ax.plot(up2.index, up2.values, linewidth=1.4, alpha=0.8, label="+2σ band")
    if not lo2.empty:
        ax.plot(lo2.index, lo2.values, linewidth=1.4, alpha=0.8, label="-2σ band")

    # Global trendline
    try:
        x = np.arange(len(close), dtype=float)
        m, b = np.polyfit(x, close.values, 1)
        g = m*x + b
        ax.plot(close.index, g, linestyle="--", linewidth=2.8, alpha=0.85,
                label=f"Global Trend ({fmt_slope(m)}/bar)")
        m_global = float(m)
    except Exception:
        pass

    # Bollinger
    if show_bbands and bb_mid.notna().any():
        ax.plot(bb_mid.index, bb_mid.values, linewidth=1.2, alpha=0.9, label="BB Mid")
        ax.plot(bb_up.index, bb_up.values, linewidth=1.0, alpha=0.7, label="BB Upper")
        ax.plot(bb_lo.index, bb_lo.values, linewidth=1.0, alpha=0.7, label="BB Lower")

    # Ichimoku
    if show_ichi and kijun.notna().any():
        ax.plot(kijun.index, kijun.values, linewidth=1.6, alpha=0.85, label=f"Kijun({ichi_base})")

    # PSAR
    if show_psar and not psar_df.empty and "PSAR" in psar_df.columns:
        ps = _coerce_1d_series(psar_df["PSAR"]).reindex(close.index)
        ax.scatter(ps.index, ps.values, s=9, alpha=0.7, label="PSAR")

    # Supertrend (ON by default)
    if show_supertrend and not st_df.empty:
        plot_supertrend(ax, st_df.reindex(close.index), label="Supertrend")

    # Support/Resistance
    if sup.notna().any():
        ax.plot(sup.index, sup.values, linewidth=1.0, alpha=0.5, label="Support (roll min)")
    if res.notna().any():
        ax.plot(res.index, res.values, linewidth=1.0, alpha=0.5, label="Resistance (roll max)")

    # Fibonacci lines (hourly only toggle)
    if show_fibs and isinstance(fibs, dict) and fibs:
        for k, v in fibs.items():
            if np.isfinite(v):
                ax.axhline(v, linewidth=1.0, alpha=0.22)
                label_on_left(ax, v, f"Fib {k}: {fmt_price_val(v)}", fontsize=8)

    # Optional FX sessions
    if mode == "Forex" and show_sessions_pst:
        add_fx_sessions_pst(ax, close.index)

    # Annotate triggers outside-right
    if conf_tr is not None:
        side = conf_tr.get("side", "")
        t = conf_tr.get("confirm_time")
        px = conf_tr.get("confirm_price")
        lvl = conf_tr.get("fib_level", "")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"CONFIRM {side}\nFib {lvl}\nR²≥99.9%", color=col, y_frac=0.88)

    if slope_tr is not None:
        side = slope_tr.get("side", "")
        t = slope_tr.get("cross_time")
        px = slope_tr.get("cross_price")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"Slope {side}\n(after ±2σ)", color=col, y_frac=0.72)

    if macd_sig_tr is not None:
        side = macd_sig_tr.get("side", "")
        t = macd_sig_tr.get("time")
        px = macd_sig_tr.get("price")
        col = "tab:green" if side == "BUY" else "tab:red"
        annotate_outside_right(ax, t, px, f"{side}\nMACD/HMA55\n+ S/R", color=col, y_frac=0.56)

    ax.set_title(f"{ticker} — Hourly", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    # ---- NTD panel ----
    if nrows == 2:
        ax2 = axes[1]
        s_ntd = _coerce_1d_series(ntd).dropna()
        if not s_ntd.empty:
            ax2.plot(s_ntd.index, s_ntd.values, linewidth=1.4, label="NTD")
            ax2.axhline(0, linewidth=1.0, alpha=0.5)
            ax2.axhline(0.75, linewidth=1.0, alpha=0.25)
            ax2.axhline(-0.75, linewidth=1.0, alpha=0.25)
            if shade_ntd:
                shade_ntd_regions(ax2, s_ntd)
            if show_npx_ntd:
                overlay_npx_on_ntd(ax2, npx, s_ntd, mark_crosses=mark_npx_cross)
            overlay_ntd_triangles_by_trend(ax2, s_ntd, m_global)
            overlay_ntd_sr_reversal_stars(ax2, close, sup, res, m_global, s_ntd, prox=sr_prox_pct, bars_confirm=rev_bars_confirm)

            if show_hma_rev_ntd and show_hma and hma.notna().any():
                overlay_hma_reversal_on_ntd(ax2, close, hma, lookback=hma_rev_lb, period=hma_period, ntd=s_ntd)

            ax2.set_title("NTD Panel", fontsize=11, fontweight="bold")
            style_axes(ax2)
            ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    return fig
# ---------------------------
# Scanner: R² Trend (Daily + Hourly)
# ---------------------------
@st.cache_data(ttl=180)
def _scan_r2_trends(universe_list, mode: str, daily_lb: int, hourly_lb: int, r2_min: float = 0.45):
    """
    Returns:
      daily_up, daily_dn, hourly_up, hourly_dn as DataFrames
    """
    rows_daily = []
    rows_hourly = []

    for sym in universe_list:
        try:
            # Daily
            d_ohlc = fetch_hist_ohlc(sym)
            d_close = _coerce_1d_series(d_ohlc["Close"]).dropna()
            if len(d_close) >= max(30, daily_lb):
                r2_d = regression_r2(d_close, lookback=daily_lb)
                _, m_d = slope_line(d_close, lookback=daily_lb)
                rows_daily.append({"Symbol": sym, "Slope": float(m_d), "R2": float(r2_d)})

            # Hourly (from 5m)
            i5 = fetch_intraday(sym, period="5d")
            if i5 is not None and not i5.empty and {"Open","High","Low","Close"}.issubset(i5.columns):
                o = i5["Open"].resample("1H").first()
                h = i5["High"].resample("1H").max()
                l = i5["Low"].resample("1H").min()
                c = i5["Close"].resample("1H").last()
                h_ohlc = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
                h_close = _coerce_1d_series(h_ohlc["Close"]).dropna()
                if len(h_close) >= max(30, hourly_lb):
                    r2_h = regression_r2(h_close, lookback=hourly_lb)
                    _, m_h = slope_line(h_close, lookback=hourly_lb)
                    rows_hourly.append({"Symbol": sym, "Slope": float(m_h), "R2": float(r2_h)})
        except Exception:
            continue

    df_d = pd.DataFrame(rows_daily)
    df_h = pd.DataFrame(rows_hourly)

    def _split(df):
        if df is None or df.empty:
            empty = pd.DataFrame(columns=["Symbol","Slope","R2"])
            return empty, empty
        df = df.copy()
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
        df["Slope"] = pd.to_numeric(df["Slope"], errors="coerce")
        df = df.dropna(subset=["R2","Slope"])
        df = df[df["R2"] >= float(r2_min)]
        up = df[df["Slope"] > 0].sort_values(["R2","Slope"], ascending=[False, False]).reset_index(drop=True)
        dn = df[df["Slope"] < 0].sort_values(["R2","Slope"], ascending=[False, True]).reset_index(drop=True)
        return up, dn

    daily_up, daily_dn = _split(df_d)
    hourly_up, hourly_dn = _split(df_h)
    return daily_up, daily_dn, hourly_up, hourly_dn


# ---------------------------
# Main UI: ticker, run, tabs
# ---------------------------
st.subheader("Select symbol")
ticker = st.selectbox("Symbol:", universe, index=0, key="sb_ticker_select")

run_all = st.button("▶️ Run / Refresh", type="primary", use_container_width=True, key="btn_run_all")
if run_all:
    st.session_state.run_all = True
    st.session_state.ticker = ticker
    st.session_state.mode_at_run = mode

# If switched mode after run, reset
if st.session_state.get("mode_at_run") is not None and st.session_state.mode_at_run != mode:
    _reset_run_state_for_mode_switch()

if not st.session_state.get("run_all", False):
    st.info("Click **Run / Refresh** to load charts.")
    st.stop()

ticker = st.session_state.get("ticker", ticker)

# Load data
df_daily_ohlc = fetch_hist_ohlc(ticker)
df_close_daily = fetch_hist(ticker)

df_intraday_5m = fetch_intraday(ticker, period=bb_period)

# Forecast (daily)
try:
    fc_idx, fc_vals, fc_ci = compute_sarimax_forecast(df_close_daily)
except Exception:
    fc_idx, fc_vals, fc_ci = None, None, None

# Tabs
tabs = st.tabs([
    "📅 Daily Chart",
    "🕒 Hourly Chart",
    "📉 Forecast (SARIMAX)",
    "📍 Signals Summary",
    "🧭 R² Trend Scanner"
])

# 1) Daily
with tabs[0]:
    st.markdown("### Daily")
    fig = build_daily_chart(ticker, df_daily_ohlc)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)

# 2) Hourly
with tabs[1]:
    st.markdown("### Hourly (1H resampled from 5m)")
    fig = build_hourly_chart(ticker, df_intraday_5m)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)

# 3) Forecast
with tabs[2]:
    st.markdown("### 30-Day Forecast (SARIMAX)")
    if fc_idx is None or fc_vals is None:
        st.warning("Forecast not available.")
    else:
        fig, ax = plt.subplots(figsize=(14, 5))
        hist = _coerce_1d_series(df_close_daily).dropna()
        ax.plot(hist.index, hist.values, linewidth=1.4, label="Historical Close")
        ax.plot(fc_idx, _coerce_1d_series(fc_vals).values, linewidth=2.2, label="Forecast")

        try:
            ci = fc_ci
            if isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
                low = _coerce_1d_series(ci.iloc[:, 0]).values
                high = _coerce_1d_series(ci.iloc[:, 1]).values
                ax.fill_between(fc_idx, low, high, alpha=0.18)
        except Exception:
            pass

        ax.set_title(f"{ticker} — SARIMAX Forecast", fontsize=13, fontweight="bold")
        style_axes(ax)
        ax.legend(loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

# 4) Signals Summary
with tabs[3]:
    st.markdown("### Latest Signals (computed from current views)")
    # Recompute minimal signals using the same internal logic (hourly is usually the action view)
    # Hourly derived series
    sig_rows = []

    # Daily signals (confirmation + slope)
    try:
        close_d = _coerce_1d_series(df_daily_ohlc["Close"]).dropna()
        yhat_d, up2_d, lo2_d, m_d, r2_d = regression_with_band(close_d, lookback=slope_lb_daily, z=2.0)
        fibs_d = fibonacci_levels(close_d)
        conf_d = find_fib_confirmation_trigger(close_d, fibs_d, r2_d, prox=sr_prox_pct,
                                               bars_confirm=rev_bars_confirm, horizon=rev_horizon, min_r2=0.999)
        if conf_d:
            sig_rows.append({
                "Timeframe": "Daily",
                "Signal": f"CONFIRM {conf_d.get('side')} (Fib {conf_d.get('fib_level')})",
                "Time": str(conf_d.get("confirm_time")),
                "Price": conf_d.get("confirm_price"),
                "R2": conf_d.get("r2")
            })
        slope_d = find_slope_trigger_after_band_reversal(close_d, yhat_d, up2_d, lo2_d, horizon=rev_horizon)
        if slope_d:
            sig_rows.append({
                "Timeframe": "Daily",
                "Signal": f"Slope {slope_d.get('side')} (after ±2σ)",
                "Time": str(slope_d.get("cross_time")),
                "Price": slope_d.get("cross_price"),
                "R2": r2_d
            })
    except Exception:
        pass

    # Hourly signals (confirmation + slope)
    try:
        if df_intraday_5m is not None and not df_intraday_5m.empty:
            o = df_intraday_5m["Open"].resample("1H").first()
            h = df_intraday_5m["High"].resample("1H").max()
            l = df_intraday_5m["Low"].resample("1H").min()
            c = df_intraday_5m["Close"].resample("1H").last()
            h_ohlc = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}).dropna()
            close_h = _coerce_1d_series(h_ohlc["Close"]).dropna()
            yhat_h, up2_h, lo2_h, m_h, r2_h = regression_with_band(close_h, lookback=slope_lb_hourly, z=2.0)
            fibs_h = fibonacci_levels(close_h)
            conf_h = find_fib_confirmation_trigger(close_h, fibs_h, r2_h, prox=sr_prox_pct,
                                                   bars_confirm=rev_bars_confirm, horizon=rev_horizon, min_r2=0.999)
            if conf_h:
                sig_rows.append({
                    "Timeframe": "Hourly",
                    "Signal": f"CONFIRM {conf_h.get('side')} (Fib {conf_h.get('fib_level')})",
                    "Time": str(conf_h.get("confirm_time")),
                    "Price": conf_h.get("confirm_price"),
                    "R2": conf_h.get("r2")
                })
            slope_h = find_slope_trigger_after_band_reversal(close_h, yhat_h, up2_h, lo2_h, horizon=rev_horizon)
            if slope_h:
                sig_rows.append({
                    "Timeframe": "Hourly",
                    "Signal": f"Slope {slope_h.get('side')} (after ±2σ)",
                    "Time": str(slope_h.get("cross_time")),
                    "Price": slope_h.get("cross_price"),
                    "R2": r2_h
                })
    except Exception:
        pass

    if not sig_rows:
        st.info("No signals detected with current thresholds.")
    else:
        sdf = pd.DataFrame(sig_rows)
        st.dataframe(sdf, use_container_width=True)

# 5) R² Trend Scanner
with tabs[4]:
    st.markdown("### R² Trend Scanner (R² ≥ 45%)")
    st.caption("This scans your universe and returns symbols with strong linear trend fits over the selected lookbacks.")

    daily_up, daily_dn, hourly_up, hourly_dn = _scan_r2_trends(
        universe_list=universe,
        mode=mode,
        daily_lb=slope_lb_daily,
        hourly_lb=slope_lb_hourly,
        r2_min=0.45
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Daily — Uptrend (R² ≥ 45%)")
        st.dataframe(daily_up, use_container_width=True)
        st.markdown("#### Daily — Downtrend (R² ≥ 45%)")
        st.dataframe(daily_dn, use_container_width=True)

    with c2:
        st.markdown("#### Hourly — Uptrend (R² ≥ 45%)")
        st.dataframe(hourly_up, use_container_width=True)
        st.markdown("#### Hourly — Downtrend (R² ≥ 45%)")
        st.dataframe(hourly_dn, use_container_width=True)

st.markdown("---")
st.caption("Reminder: This dashboard provides signals for informational purposes only. Use risk management.")
